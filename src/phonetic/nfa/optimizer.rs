//! NFA optimization passes for reducing size and improving matching performance.
//!
//! This module provides optimization algorithms that transform NFAs to be smaller
//! and faster to execute:
//!
//! - **Epsilon Elimination**: Remove ε-transitions by computing transitive closure
//! - **Unreachable State Removal**: Remove states not reachable from start
//! - **Dead State Removal**: Remove states that cannot reach any final state
//!
//! # Design
//!
//! Optimization is applied automatically after Thompson construction. The order
//! of passes matters:
//!
//! 1. Epsilon elimination (changes transition structure)
//! 2. Unreachable state removal (benefits from epsilon elimination)
//! 3. Dead state removal (benefits from unreachable removal)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::nfa::{
//!     NFACompilerChar, NfaOptimizerChar, OptimizationConfig,
//! };
//! use liblevenshtein::phonetic::regex::parse;
//!
//! let regex = parse("(a|b)*c").expect("doc: regex parse must succeed");
//! let mut compiler = NFACompilerChar::new().without_optimization();
//! let nfa = compiler
//!     .compile(&regex)
//!     .expect("doc: NFA compilation must succeed");
//!
//! let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
//! let (optimized, stats) = optimizer.optimize(nfa.clone());
//!
//! assert_eq!(stats.original_states, nfa.num_states());
//! assert_eq!(stats.final_states, optimized.num_states());
//! assert!(stats.original_epsilon_count > 0);
//! assert_eq!(optimized.count_epsilon_transitions(), 0);
//! for input in ["c", "abc", "bbaac", "ab"] {
//!     assert_eq!(optimized.accepts(input), nfa.accepts(input));
//! }
//! ```

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use std::collections::VecDeque;

use rustc_hash::{FxHashMap, FxHashSet};

use super::state_set::StateSet;
use super::types::{state_id_from_len, state_index, StateId, TransitionLabel, TransitionLabelChar};
use super::{NFAChar, NFA};

fn state_ids_for_count(count: usize) -> Option<Vec<StateId>> {
    (0..count).map(state_id_from_len).collect()
}

fn closure_for_state(closures: &[StateSet], state: StateId) -> Option<&StateSet> {
    state_index(state).and_then(|index| closures.get(index))
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for NFA optimization passes.
///
/// By default, all optimization passes are enabled.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct OptimizationConfig {
    /// Eliminate epsilon transitions by computing transitive closure.
    ///
    /// This is the most impactful optimization, removing all ε-transitions
    /// and adding direct transitions to bypass them.
    pub eliminate_epsilon: bool,

    /// Remove states not reachable from the start state.
    ///
    /// After epsilon elimination, some states may become unreachable.
    pub remove_unreachable: bool,

    /// Remove states that cannot reach any final state.
    ///
    /// These "dead" states can never lead to acceptance.
    pub remove_dead: bool,

    /// Remove duplicate transitions (same source, label, and destination).
    ///
    /// Epsilon elimination can create duplicates that waste memory.
    pub deduplicate_transitions: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self::full()
    }
}

impl OptimizationConfig {
    /// Full optimization with all passes enabled.
    pub fn full() -> Self {
        Self {
            eliminate_epsilon: true,
            remove_unreachable: true,
            remove_dead: true,
            deduplicate_transitions: true,
        }
    }

    /// Quick optimization without epsilon elimination.
    ///
    /// Epsilon elimination is O(|Q|² * |δ|), so skipping it is faster
    /// for cases where the NFA will only be used once.
    pub fn quick() -> Self {
        Self {
            eliminate_epsilon: false,
            remove_unreachable: true,
            remove_dead: true,
            deduplicate_transitions: false,
        }
    }

    /// No optimization (useful for testing/debugging).
    pub fn none() -> Self {
        Self {
            eliminate_epsilon: false,
            remove_unreachable: false,
            remove_dead: false,
            deduplicate_transitions: false,
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics collected during NFA optimization.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct OptimizationStats {
    /// Number of states in the original NFA.
    pub original_states: usize,
    /// Number of transitions in the original NFA.
    pub original_transitions: usize,
    /// Number of epsilon transitions in the original NFA.
    pub original_epsilon_count: usize,

    /// Number of states in the optimized NFA.
    pub final_states: usize,
    /// Number of transitions in the optimized NFA.
    pub final_transitions: usize,

    /// Total states removed (unreachable + dead).
    pub states_removed: usize,
    /// Epsilon transitions eliminated.
    pub epsilon_transitions_eliminated: usize,
    /// Unreachable states removed.
    pub unreachable_states_removed: usize,
    /// Dead states removed.
    pub dead_states_removed: usize,
    /// Duplicate transitions removed.
    pub duplicate_transitions_removed: usize,
}

impl OptimizationStats {
    /// Calculate the percentage reduction in states.
    pub fn state_reduction_percent(&self) -> f64 {
        if self.original_states == 0 {
            0.0
        } else {
            100.0 * (self.original_states - self.final_states) as f64 / self.original_states as f64
        }
    }

    /// Calculate the percentage reduction in transitions.
    pub fn transition_reduction_percent(&self) -> f64 {
        if self.original_transitions == 0 {
            0.0
        } else {
            100.0 * (self.original_transitions - self.final_transitions) as f64
                / self.original_transitions as f64
        }
    }
}

// ============================================================================
// Optimizer (Character-level)
// ============================================================================

/// NFA optimizer for character-level NFAs.
///
/// Applies optimization passes to reduce NFA size and improve matching performance.
#[derive(Debug, Clone)]
pub struct NfaOptimizerChar {
    config: OptimizationConfig,
}

impl NfaOptimizerChar {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// Optimize an NFA, returning the optimized NFA and statistics.
    pub fn optimize(&self, nfa: NFAChar) -> (NFAChar, OptimizationStats) {
        // H9: Finalize the input NFA to ensure all transitions are accessible
        let mut nfa = nfa;
        nfa.finalize();

        let mut stats = OptimizationStats {
            original_states: nfa.num_states(),
            original_transitions: nfa.num_transitions(),
            original_epsilon_count: count_epsilon_transitions_char(&nfa),
            ..Default::default()
        };

        let mut result = nfa;

        // Step 1: Epsilon elimination (changes transition structure)
        if self.config.eliminate_epsilon {
            let before_transitions = result.num_transitions();
            result = eliminate_epsilon_char(result);
            // H9: Finalize after each step to make transitions accessible
            result.finalize();
            let epsilon_after = count_epsilon_transitions_char(&result);
            stats.epsilon_transitions_eliminated =
                stats.original_epsilon_count.saturating_sub(epsilon_after);
            // Note: transition count may increase due to transitive closure
            let _ = before_transitions; // suppress unused warning
        }

        // Step 2: Unreachable state removal
        if self.config.remove_unreachable {
            let before_states = result.num_states();
            result = remove_unreachable_char(result);
            // H9: Finalize after each step
            result.finalize();
            stats.unreachable_states_removed = before_states.saturating_sub(result.num_states());
        }

        // Step 3: Dead state removal
        if self.config.remove_dead {
            let before_states = result.num_states();
            result = remove_dead_char(result);
            // H9: Finalize after each step
            result.finalize();
            stats.dead_states_removed = before_states.saturating_sub(result.num_states());
        }

        // Step 4: Deduplicate transitions
        if self.config.deduplicate_transitions {
            let before_transitions = result.num_transitions();
            result = deduplicate_transitions_char(result);
            // H9: Finalize after each step
            result.finalize();
            stats.duplicate_transitions_removed =
                before_transitions.saturating_sub(result.num_transitions());
        }

        stats.final_states = result.num_states();
        stats.final_transitions = result.num_transitions();
        stats.states_removed = stats.original_states.saturating_sub(stats.final_states);

        (result, stats)
    }
}

// ============================================================================
// Optimizer (Byte-level)
// ============================================================================

/// NFA optimizer for byte-level NFAs.
#[derive(Debug, Clone)]
pub struct NfaOptimizer {
    config: OptimizationConfig,
}

impl NfaOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: OptimizationConfig) -> Self {
        Self { config }
    }

    /// Optimize an NFA, returning the optimized NFA and statistics.
    pub fn optimize(&self, nfa: NFA) -> (NFA, OptimizationStats) {
        // H9: Finalize the input NFA to ensure all transitions are accessible
        let mut nfa = nfa;
        nfa.finalize();

        let mut stats = OptimizationStats {
            original_states: nfa.num_states(),
            original_transitions: nfa.num_transitions(),
            original_epsilon_count: count_epsilon_transitions(&nfa),
            ..Default::default()
        };

        let mut result = nfa;

        if self.config.eliminate_epsilon {
            result = eliminate_epsilon(result);
            // H9: Finalize after each step to make transitions accessible
            result.finalize();
            let epsilon_after = count_epsilon_transitions(&result);
            stats.epsilon_transitions_eliminated =
                stats.original_epsilon_count.saturating_sub(epsilon_after);
        }

        if self.config.remove_unreachable {
            let before_states = result.num_states();
            result = remove_unreachable(result);
            // H9: Finalize after each step
            result.finalize();
            stats.unreachable_states_removed = before_states.saturating_sub(result.num_states());
        }

        if self.config.remove_dead {
            let before_states = result.num_states();
            result = remove_dead(result);
            // H9: Finalize after each step
            result.finalize();
            stats.dead_states_removed = before_states.saturating_sub(result.num_states());
        }

        if self.config.deduplicate_transitions {
            let before_transitions = result.num_transitions();
            result = deduplicate_transitions(result);
            // H9: Finalize after each step
            result.finalize();
            stats.duplicate_transitions_removed =
                before_transitions.saturating_sub(result.num_transitions());
        }

        stats.final_states = result.num_states();
        stats.final_transitions = result.num_transitions();
        stats.states_removed = stats.original_states.saturating_sub(stats.final_states);

        (result, stats)
    }
}

// ============================================================================
// Helper Functions (Character-level)
// ============================================================================

/// Count epsilon transitions in an NFA.
fn count_epsilon_transitions_char(nfa: &NFAChar) -> usize {
    nfa.transitions()
        .iter()
        .filter(|t| t.label.is_epsilon())
        .count()
}

/// Remove states not reachable from the start state.
fn remove_unreachable_char(nfa: NFAChar) -> NFAChar {
    // BFS from start state to find reachable states
    let mut reachable = FxHashSet::with_capacity_and_hasher(nfa.num_states(), Default::default());
    let mut queue = VecDeque::with_capacity(nfa.num_states());

    reachable.insert(nfa.start());
    queue.push_back(nfa.start());

    while let Some(state) = queue.pop_front() {
        for trans in nfa.transitions_from(state) {
            if reachable.insert(trans.to) {
                queue.push_back(trans.to);
            }
        }
    }

    // If all states are reachable, return as-is
    if reachable.len() == nfa.num_states() {
        return nfa;
    }

    // Build new NFA with only reachable states
    build_nfa_with_states_char(&nfa, &reachable)
}

/// Remove states that cannot reach any final state.
fn remove_dead_char(nfa: NFAChar) -> NFAChar {
    // Build reverse transition graph
    let mut reverse_index: FxHashMap<StateId, Vec<StateId>> =
        FxHashMap::with_capacity_and_hasher(nfa.num_states(), Default::default());
    for trans in nfa.transitions() {
        reverse_index.entry(trans.to).or_default().push(trans.from);
    }

    // Backward BFS from all final states
    let mut can_reach_final =
        FxHashSet::with_capacity_and_hasher(nfa.num_states(), Default::default());
    let mut queue = VecDeque::with_capacity(nfa.num_states());

    for &final_state in nfa.finals() {
        if can_reach_final.insert(final_state) {
            queue.push_back(final_state);
        }
    }

    while let Some(state) = queue.pop_front() {
        if let Some(predecessors) = reverse_index.get(&state) {
            for &pred in predecessors {
                if can_reach_final.insert(pred) {
                    queue.push_back(pred);
                }
            }
        }
    }

    // Start state must be kept even if it can't reach final (edge case)
    can_reach_final.insert(nfa.start());

    // If all states can reach final, return as-is
    if can_reach_final.len() == nfa.num_states() {
        return nfa;
    }

    build_nfa_with_states_char(&nfa, &can_reach_final)
}

/// Eliminate epsilon transitions by computing transitive closure.
fn eliminate_epsilon_char(nfa: NFAChar) -> NFAChar {
    // Precompute epsilon closures for all states
    let state_ids =
        state_ids_for_count(nfa.num_states()).expect("NFA state count exceeded StateId capacity");
    let closures: Vec<StateSet> = state_ids
        .iter()
        .copied()
        .map(|s| nfa.epsilon_closure_single(s))
        .collect();

    // Build new NFA
    let mut new_nfa = NFAChar::new();

    // Add states (same number as original)
    // State 0 already exists from NFAChar::new()
    for _ in 1..nfa.num_states() {
        new_nfa.add_state(false);
    }

    // Set start state
    // (new_nfa.start is already 0, same as nfa.start() typically)

    // Update final states: state is final if any state in its epsilon closure is final
    for &state_id in &state_ids {
        let Some(closure) = closure_for_state(&closures, state_id) else {
            continue;
        };
        let is_final = closure.iter().any(|s| nfa.is_final(s));
        new_nfa.set_final(state_id, is_final);
    }

    // Add non-epsilon transitions with epsilon bypass
    for &state_id in &state_ids {
        let Some(closure) = closure_for_state(&closures, state_id) else {
            continue;
        };

        for reachable in closure.iter() {
            for trans in nfa.transitions_from(reachable) {
                // Skip epsilon transitions
                if trans.label.is_epsilon() {
                    continue;
                }

                // For non-epsilon transitions, add transition from original state
                // to all states in epsilon closure of destination
                let Some(dest_closure) = closure_for_state(&closures, trans.to) else {
                    continue;
                };
                for dest in dest_closure.iter() {
                    new_nfa.add_transition_weighted(
                        state_id,
                        trans.label.clone(),
                        dest,
                        trans.weight,
                    );
                }
            }
        }
    }

    new_nfa
}

/// Remove duplicate transitions.
fn deduplicate_transitions_char(nfa: NFAChar) -> NFAChar {
    let unique_transitions = {
        let transitions = nfa.transitions();
        let mut seen: FxHashSet<(StateId, StateId, &TransitionLabelChar)> =
            FxHashSet::with_capacity_and_hasher(transitions.len(), Default::default());
        let mut unique = Vec::with_capacity(transitions.len());

        for trans in transitions {
            let key = (trans.from, trans.to, &trans.label);
            if seen.insert(key) {
                unique.push(trans);
            }
        }

        if unique.len() == transitions.len() {
            None
        } else {
            Some(unique.into_iter().cloned().collect::<Vec<_>>())
        }
    };

    let Some(unique_transitions) = unique_transitions else {
        return nfa;
    };

    // Build new NFA with deduplicated transitions
    let mut new_nfa = NFAChar::new();

    // Add states
    for _i in 1..nfa.num_states() {
        new_nfa.add_state(false);
    }

    // Set final states
    for &final_state in nfa.finals() {
        new_nfa.set_final(final_state, true);
    }

    // Add unique transitions
    for trans in unique_transitions {
        new_nfa.add_transition_weighted(trans.from, trans.label, trans.to, trans.weight);
    }

    new_nfa
}

/// Build a new NFA containing only the specified states.
fn build_nfa_with_states_char(nfa: &NFAChar, keep_states: &FxHashSet<StateId>) -> NFAChar {
    // Build mapping from old state IDs to new contiguous IDs
    let mut old_to_new: FxHashMap<StateId, StateId> =
        FxHashMap::with_capacity_and_hasher(keep_states.len(), Default::default());
    let mut sorted_states = Vec::with_capacity(keep_states.len());
    sorted_states.extend(keep_states.iter().copied());
    sorted_states.sort_unstable();

    for (new_id, &old_id) in sorted_states.iter().enumerate() {
        let new_id =
            state_id_from_len(new_id).expect("optimized NFA state count exceeded StateId capacity");
        old_to_new.insert(old_id, new_id);
    }

    // Build new NFA
    let mut new_nfa = NFAChar::new();

    // Add states (state 0 already exists)
    for _i in 1..sorted_states.len() {
        new_nfa.add_state(false);
    }

    debug_assert_eq!(
        nfa.start(),
        0,
        "NFAChar currently stores a fixed start state at q0"
    );

    // Set final states
    for &old_id in nfa.finals() {
        if let Some(&new_id) = old_to_new.get(&old_id) {
            new_nfa.set_final(new_id, true);
        }
    }

    // Add transitions (only those between kept states)
    for trans in nfa.transitions() {
        if let (Some(&new_from), Some(&new_to)) =
            (old_to_new.get(&trans.from), old_to_new.get(&trans.to))
        {
            new_nfa.add_transition_weighted(new_from, trans.label.clone(), new_to, trans.weight);
        }
    }

    new_nfa
}

// ============================================================================
// Helper Functions (Byte-level)
// ============================================================================

/// Count epsilon transitions in a byte-level NFA.
fn count_epsilon_transitions(nfa: &NFA) -> usize {
    nfa.transitions()
        .iter()
        .filter(|t| t.label.is_epsilon())
        .count()
}

/// Remove states not reachable from the start state (byte-level).
fn remove_unreachable(nfa: NFA) -> NFA {
    let mut reachable = FxHashSet::with_capacity_and_hasher(nfa.num_states(), Default::default());
    let mut queue = VecDeque::with_capacity(nfa.num_states());

    reachable.insert(nfa.start());
    queue.push_back(nfa.start());

    while let Some(state) = queue.pop_front() {
        for trans in nfa.transitions_from(state) {
            if reachable.insert(trans.to) {
                queue.push_back(trans.to);
            }
        }
    }

    if reachable.len() == nfa.num_states() {
        return nfa;
    }

    build_nfa_with_states(&nfa, &reachable)
}

/// Remove states that cannot reach any final state (byte-level).
fn remove_dead(nfa: NFA) -> NFA {
    let mut reverse_index: FxHashMap<StateId, Vec<StateId>> =
        FxHashMap::with_capacity_and_hasher(nfa.num_states(), Default::default());
    for trans in nfa.transitions() {
        reverse_index.entry(trans.to).or_default().push(trans.from);
    }

    let mut can_reach_final =
        FxHashSet::with_capacity_and_hasher(nfa.num_states(), Default::default());
    let mut queue = VecDeque::with_capacity(nfa.num_states());

    for &final_state in nfa.finals() {
        if can_reach_final.insert(final_state) {
            queue.push_back(final_state);
        }
    }

    while let Some(state) = queue.pop_front() {
        if let Some(predecessors) = reverse_index.get(&state) {
            for &pred in predecessors {
                if can_reach_final.insert(pred) {
                    queue.push_back(pred);
                }
            }
        }
    }

    can_reach_final.insert(nfa.start());

    if can_reach_final.len() == nfa.num_states() {
        return nfa;
    }

    build_nfa_with_states(&nfa, &can_reach_final)
}

/// Eliminate epsilon transitions (byte-level).
fn eliminate_epsilon(nfa: NFA) -> NFA {
    let state_ids =
        state_ids_for_count(nfa.num_states()).expect("NFA state count exceeded StateId capacity");
    let closures: Vec<StateSet> = state_ids
        .iter()
        .copied()
        .map(|s| nfa.epsilon_closure_single(s))
        .collect();

    let mut new_nfa = NFA::new();

    for _ in 1..nfa.num_states() {
        new_nfa.add_state(false);
    }

    for &state_id in &state_ids {
        let Some(closure) = closure_for_state(&closures, state_id) else {
            continue;
        };
        let is_final = closure.iter().any(|s| nfa.is_final(s));
        new_nfa.set_final(state_id, is_final);
    }

    for &state_id in &state_ids {
        let Some(closure) = closure_for_state(&closures, state_id) else {
            continue;
        };

        for reachable in closure.iter() {
            for trans in nfa.transitions_from(reachable) {
                if trans.label.is_epsilon() {
                    continue;
                }

                let Some(dest_closure) = closure_for_state(&closures, trans.to) else {
                    continue;
                };
                for dest in dest_closure.iter() {
                    new_nfa.add_transition_weighted(
                        state_id,
                        trans.label.clone(),
                        dest,
                        trans.weight,
                    );
                }
            }
        }
    }

    new_nfa
}

/// Remove duplicate transitions (byte-level).
fn deduplicate_transitions(nfa: NFA) -> NFA {
    let unique_transitions = {
        let transitions = nfa.transitions();
        let mut seen: FxHashSet<(StateId, StateId, &TransitionLabel)> =
            FxHashSet::with_capacity_and_hasher(transitions.len(), Default::default());
        let mut unique = Vec::with_capacity(transitions.len());

        for trans in transitions {
            let key = (trans.from, trans.to, &trans.label);
            if seen.insert(key) {
                unique.push(trans);
            }
        }

        if unique.len() == transitions.len() {
            None
        } else {
            Some(unique.into_iter().cloned().collect::<Vec<_>>())
        }
    };

    let Some(unique_transitions) = unique_transitions else {
        return nfa;
    };

    let mut new_nfa = NFA::new();

    for _ in 1..nfa.num_states() {
        new_nfa.add_state(false);
    }

    for &final_state in nfa.finals() {
        new_nfa.set_final(final_state, true);
    }

    for trans in unique_transitions {
        new_nfa.add_transition_weighted(trans.from, trans.label, trans.to, trans.weight);
    }

    new_nfa
}

/// Build a new byte-level NFA containing only the specified states.
fn build_nfa_with_states(nfa: &NFA, keep_states: &FxHashSet<StateId>) -> NFA {
    let mut old_to_new: FxHashMap<StateId, StateId> =
        FxHashMap::with_capacity_and_hasher(keep_states.len(), Default::default());
    let mut sorted_states = Vec::with_capacity(keep_states.len());
    sorted_states.extend(keep_states.iter().copied());
    sorted_states.sort_unstable();

    for (new_id, &old_id) in sorted_states.iter().enumerate() {
        let new_id =
            state_id_from_len(new_id).expect("optimized NFA state count exceeded StateId capacity");
        old_to_new.insert(old_id, new_id);
    }

    let mut new_nfa = NFA::new();

    for _ in 1..sorted_states.len() {
        new_nfa.add_state(false);
    }

    for &old_id in nfa.finals() {
        if let Some(&new_id) = old_to_new.get(&old_id) {
            new_nfa.set_final(new_id, true);
        }
    }

    for trans in nfa.transitions() {
        if let (Some(&new_from), Some(&new_to)) =
            (old_to_new.get(&trans.from), old_to_new.get(&trans.to))
        {
            new_nfa.add_transition_weighted(new_from, trans.label.clone(), new_to, trans.weight);
        }
    }

    new_nfa
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::nfa::{compile, CharClass, CharClassChar, ThompsonBuilderChar};
    use crate::phonetic::regex::parse;

    #[test]
    fn test_optimization_config_full() {
        let config = OptimizationConfig::full();
        assert!(config.eliminate_epsilon);
        assert!(config.remove_unreachable);
        assert!(config.remove_dead);
        assert!(config.deduplicate_transitions);
    }

    #[test]
    fn test_optimization_config_quick() {
        let config = OptimizationConfig::quick();
        assert!(!config.eliminate_epsilon);
        assert!(config.remove_unreachable);
        assert!(config.remove_dead);
        assert!(!config.deduplicate_transitions);
    }

    #[test]
    fn test_optimization_config_none() {
        let config = OptimizationConfig::none();
        assert!(!config.eliminate_epsilon);
        assert!(!config.remove_unreachable);
        assert!(!config.remove_dead);
        assert!(!config.deduplicate_transitions);
    }

    #[test]
    fn test_count_epsilon_transitions() {
        let builder = ThompsonBuilderChar::new();

        // a|b has epsilon transitions for alternation
        let a = builder.single_char('a');
        let b = builder.single_char('b');
        let nfa = builder.alternation(a, b);

        let count = count_epsilon_transitions_char(&nfa);
        assert!(count > 0, "alternation should have epsilon transitions");
    }

    #[test]
    fn test_remove_unreachable_no_change() {
        // All states reachable - no change expected
        let builder = ThompsonBuilderChar::new();
        let nfa = builder.single_char('a');

        let optimized = remove_unreachable_char(nfa.clone());

        assert_eq!(nfa.num_states(), optimized.num_states());
        assert_eq!(nfa.num_transitions(), optimized.num_transitions());
    }

    #[test]
    fn test_epsilon_elimination_simple() {
        let builder = ThompsonBuilderChar::new();

        // a* has epsilon transitions
        let a = builder.single_char('a');
        let nfa = builder.kleene_star(a);

        let before_epsilon = count_epsilon_transitions_char(&nfa);
        assert!(
            before_epsilon > 0,
            "kleene_star should have epsilon transitions"
        );

        let optimized = eliminate_epsilon_char(nfa.clone());
        let after_epsilon = count_epsilon_transitions_char(&optimized);

        assert_eq!(
            after_epsilon, 0,
            "epsilon elimination should remove all epsilon transitions"
        );

        // Verify language preservation
        assert!(nfa.accepts(""));
        assert!(optimized.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(optimized.accepts("a"));
        assert!(nfa.accepts("aaa"));
        assert!(optimized.accepts("aaa"));
        assert!(!nfa.accepts("b"));
        assert!(!optimized.accepts("b"));
    }

    #[test]
    fn test_epsilon_elimination_alternation() {
        let builder = ThompsonBuilderChar::new();

        // a|b
        let a = builder.single_char('a');
        let b = builder.single_char('b');
        let nfa = builder.alternation(a, b);

        let optimized = eliminate_epsilon_char(nfa.clone());

        // Language preservation
        assert!(nfa.accepts("a"));
        assert!(optimized.accepts("a"));
        assert!(nfa.accepts("b"));
        assert!(optimized.accepts("b"));
        assert!(!nfa.accepts("c"));
        assert!(!optimized.accepts("c"));
        assert!(!nfa.accepts("ab"));
        assert!(!optimized.accepts("ab"));
    }

    #[test]
    fn test_full_optimization_preserves_language() {
        use crate::phonetic::nfa::compiler::NFACompilerChar;

        let patterns = ["a", "ab", "a|b", "a*", "a+", "a?", "(ab)+", "(a|b)*c"];

        for pattern in patterns {
            let regex = parse(pattern).expect("parse");
            // Use compiler without optimization to get unoptimized NFA
            let mut compiler = NFACompilerChar::new().without_optimization();
            let nfa = compiler.compile(&regex).expect("compile");

            let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
            let (optimized, stats) = optimizer.optimize(nfa.clone());

            // Test various inputs
            let test_inputs = ["", "a", "b", "c", "ab", "abc", "aaa", "bbb", "abababc"];
            for input in test_inputs {
                assert_eq!(
                    nfa.accepts(input),
                    optimized.accepts(input),
                    "language mismatch for pattern '{}' on input '{}'",
                    pattern,
                    input
                );
            }

            // Verify some optimization happened (at least for non-trivial patterns)
            if pattern.contains('|') || pattern.contains('*') || pattern.contains('+') {
                assert!(
                    stats.original_epsilon_count > 0,
                    "pattern '{}' should have epsilon transitions",
                    pattern
                );
            }
        }
    }

    #[test]
    fn test_optimization_stats() {
        let builder = ThompsonBuilderChar::new();

        // a* has epsilon transitions
        let a = builder.single_char('a');
        let nfa = builder.kleene_star(a);

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (_optimized, stats) = optimizer.optimize(nfa);

        assert!(stats.original_states > 0);
        assert!(stats.original_transitions > 0);
        assert!(stats.original_epsilon_count > 0);
        assert!(stats.epsilon_transitions_eliminated > 0);
        assert!(stats.final_states > 0);
    }

    #[test]
    fn test_deduplicate_transitions() {
        // Create NFA with duplicate transitions manually
        let mut nfa = NFAChar::new();
        let q1 = nfa.add_state(true);

        // Add duplicate transitions
        nfa.add_transition_char(0, 'a', q1);
        nfa.add_transition_char(0, 'a', q1);
        nfa.add_transition_char(0, 'a', q1);

        // H9: Finalize to move pending transitions to the main list
        nfa.finalize();

        assert_eq!(nfa.num_transitions(), 3);

        let mut optimized = deduplicate_transitions_char(nfa);
        // H9: Finalize the result to make transitions accessible
        optimized.finalize();
        assert_eq!(optimized.num_transitions(), 1);
    }

    #[test]
    fn test_deduplicate_transitions_preserves_distinct_char_classes() {
        let mut nfa = NFAChar::new();
        let q1 = nfa.add_state(true);
        let a_class = CharClassChar::from_chars(&['a']);
        let b_class = CharClassChar::from_chars(&['b']);

        nfa.add_transition_class(0, a_class.clone(), q1);
        nfa.add_transition_class(0, a_class, q1);
        nfa.add_transition_class(0, b_class, q1);
        nfa.finalize();

        let mut optimized = deduplicate_transitions_char(nfa);
        optimized.finalize();

        assert_eq!(optimized.num_transitions(), 2);
        assert!(optimized.accepts("a"));
        assert!(optimized.accepts("b"));
        assert!(!optimized.accepts("c"));
    }

    #[test]
    fn test_deduplicate_byte_transitions_preserves_distinct_classes() {
        let mut nfa = NFA::new();
        let q1 = nfa.add_state(true);
        let a_class = CharClass::from_bytes(b"a");
        let b_class = CharClass::from_bytes(b"b");

        nfa.add_transition_class(0, a_class.clone(), q1);
        nfa.add_transition_class(0, a_class, q1);
        nfa.add_transition_class(0, b_class, q1);
        nfa.finalize();

        let mut optimized = deduplicate_transitions(nfa);
        optimized.finalize();

        assert_eq!(optimized.num_transitions(), 2);
        assert!(optimized.accepts(b"a"));
        assert!(optimized.accepts(b"b"));
        assert!(!optimized.accepts(b"c"));
    }

    #[test]
    fn test_char_epsilon_elimination_skips_invalid_transition_destination() {
        let mut nfa = NFAChar::new();
        nfa.add_transition_char(nfa.start(), 'a', u32::MAX);
        nfa.finalize();

        let optimizer = NfaOptimizerChar::new(OptimizationConfig {
            eliminate_epsilon: true,
            remove_unreachable: false,
            remove_dead: false,
            deduplicate_transitions: false,
        });
        let (mut optimized, _stats) = optimizer.optimize(nfa);
        optimized.finalize();

        assert_eq!(optimized.num_transitions(), 0);
        assert!(!optimized.accepts("a"));
    }

    #[test]
    fn test_byte_epsilon_elimination_skips_invalid_transition_destination() {
        let mut nfa = NFA::new();
        nfa.add_transition_byte(nfa.start(), b'a', u32::MAX);
        nfa.finalize();

        let optimizer = NfaOptimizer::new(OptimizationConfig {
            eliminate_epsilon: true,
            remove_unreachable: false,
            remove_dead: false,
            deduplicate_transitions: false,
        });
        let (mut optimized, _stats) = optimizer.optimize(nfa);
        optimized.finalize();

        assert_eq!(optimized.num_transitions(), 0);
        assert!(!optimized.accepts(b"a"));
    }

    #[test]
    fn test_optimization_with_anchors() {
        // ^hello$ - anchors should be preserved
        let regex = parse("^hello$").expect("parse");
        let nfa = compile(&regex).expect("compile");

        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(nfa.clone());

        // Both should accept "hello" (anchors are zero-width)
        // Note: The actual anchor matching depends on the NFA simulation
        // Here we just verify the optimization doesn't break the NFA
        assert!(optimized.num_states() > 0);
        assert!(optimized.num_transitions() > 0);
    }
}
