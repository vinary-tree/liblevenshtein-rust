//! Core NFA data structure for phonetic regular expressions (byte-level).
//!
//! This module implements the byte-level NFA (Nondeterministic Finite Automaton)
//! data structure used for phonetic pattern matching over ASCII byte sequences.
//! It is the byte-oriented counterpart of [`super::char_nfa::NFAChar`] and supports
//! the same operations:
//!
//! - Epsilon transitions (ε-moves)
//! - Byte and character class transitions
//! - Weighted transitions for phonetic cost modeling
//! - Efficient epsilon closure computation
//! - Union, concatenation, and Kleene star operations

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use std::collections::VecDeque;
use std::fmt;

use rustc_hash::FxHashSet;

use super::state_set::StateSet;
use super::types::{
    CharClass, NFAState, StateId, Transition, TransitionLabel,
};

// ============================================================================
// Transitions Iterator Types (H9) - byte-level
// ============================================================================

/// Iterator wrapper for transitions from a state (byte-level).
pub enum TransitionsFrom<'a> {
    /// Fast path: direct slice from CSR structure
    Slice(&'a [Transition]),
    /// Fallback: references to pending and finalized transitions for filtering
    Pending(
        &'a [Transition],
        &'a [Transition],
        &'a [usize],
        StateId,
    ),
}

impl<'a> TransitionsFrom<'a> {
    /// Iterate over all transitions from this state.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &'a Transition> + '_ {
        TransitionsFromIter::new(self)
    }
}

impl<'a> IntoIterator for TransitionsFrom<'a> {
    type Item = &'a Transition;
    type IntoIter = TransitionsFromOwned<'a>;

    fn into_iter(self) -> Self::IntoIter {
        TransitionsFromOwned::new(self)
    }
}

/// Owned iterator for TransitionsFrom (for `for` loops).
pub struct TransitionsFromOwned<'a> {
    source: TransitionsFrom<'a>,
    idx: usize,
    phase: u8,
}

impl<'a> TransitionsFromOwned<'a> {
    fn new(source: TransitionsFrom<'a>) -> Self {
        Self {
            source,
            idx: 0,
            phase: 0,
        }
    }
}

impl<'a> Iterator for TransitionsFromOwned<'a> {
    type Item = &'a Transition;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.source {
            TransitionsFrom::Slice(slice) => {
                if self.idx < slice.len() {
                    let item = &slice[self.idx];
                    self.idx += 1;
                    Some(item)
                } else {
                    None
                }
            }
            TransitionsFrom::Pending(pending, transitions, offsets, state) => {
                if self.phase == 0 {
                    let state_idx = *state as usize;
                    if state_idx + 1 < offsets.len() {
                        let start = offsets[state_idx];
                        let end = offsets[state_idx + 1];
                        while start + self.idx < end {
                            let item = &transitions[start + self.idx];
                            self.idx += 1;
                            return Some(item);
                        }
                    }
                    self.phase = 1;
                    self.idx = 0;
                }
                while self.idx < pending.len() {
                    let trans = &pending[self.idx];
                    self.idx += 1;
                    if trans.from == *state {
                        return Some(trans);
                    }
                }
                None
            }
        }
    }
}

/// Iterator for TransitionsFrom.
struct TransitionsFromIter<'a, 'b> {
    source: &'b TransitionsFrom<'a>,
    idx: usize,
    phase: u8,
}

impl<'a, 'b> TransitionsFromIter<'a, 'b> {
    fn new(source: &'b TransitionsFrom<'a>) -> Self {
        Self {
            source,
            idx: 0,
            phase: 0,
        }
    }
}

impl<'a, 'b> Iterator for TransitionsFromIter<'a, 'b> {
    type Item = &'a Transition;

    fn next(&mut self) -> Option<Self::Item> {
        match self.source {
            TransitionsFrom::Slice(slice) => {
                if self.idx < slice.len() {
                    let item = &slice[self.idx];
                    self.idx += 1;
                    Some(item)
                } else {
                    None
                }
            }
            TransitionsFrom::Pending(pending, transitions, offsets, state) => {
                if self.phase == 0 {
                    let state_idx = *state as usize;
                    if state_idx + 1 < offsets.len() {
                        let start = offsets[state_idx];
                        let end = offsets[state_idx + 1];
                        while start + self.idx < end {
                            let item = &transitions[start + self.idx];
                            self.idx += 1;
                            return Some(item);
                        }
                    }
                    self.phase = 1;
                    self.idx = 0;
                }
                while self.idx < pending.len() {
                    let trans = &pending[self.idx];
                    self.idx += 1;
                    if trans.from == *state {
                        return Some(trans);
                    }
                }
                None
            }
        }
    }
}

// ============================================================================
// NFA (Byte-level)
// ============================================================================

/// Nondeterministic Finite Automaton (byte-level).
///
/// An NFA for matching phonetic patterns over ASCII bytes.
/// Optimized for ~5% faster matching and ~4× less memory per edge label.
///
/// # H9 Optimization: CSR Transition Table
///
/// Uses Compressed Sparse Row (CSR) format for transitions:
/// - `transitions`: contiguous array sorted by source state
/// - `transition_offsets`: `offsets[s]..offsets[s+1]` = transitions from state s
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct NFA {
    /// All states in the NFA
    states: Vec<NFAState>,
    /// All transitions (H9: sorted by source state for CSR access)
    transitions: Vec<Transition>,
    /// Initial state
    start: StateId,
    /// Set of final (accepting) states
    finals: FxHashSet<StateId>,
    /// H9: CSR offsets - transitions for state s are at transitions[offsets[s]..offsets[s+1]]
    transition_offsets: Vec<usize>,
    /// Pending transitions before finalization
    pending_transitions: Vec<Transition>,
    /// Whether the CSR structure is finalized
    finalized: bool,
}

impl NFA {
    /// Create a new empty NFA with a single non-final initial state.
    pub fn new() -> Self {
        let initial = NFAState::non_final(0);
        let mut finals = FxHashSet::default();
        finals.reserve(4);
        Self {
            states: vec![initial],
            transitions: Vec::new(),
            start: 0,
            finals,
            transition_offsets: vec![0, 0],
            pending_transitions: Vec::new(),
            finalized: true,
        }
    }

    /// Create an NFA with specified initial state finality.
    pub fn with_initial_final(is_final: bool) -> Self {
        let initial = NFAState::new(0, is_final);
        let mut finals = FxHashSet::default();
        if is_final {
            finals.insert(0);
        }
        Self {
            states: vec![initial],
            transitions: Vec::new(),
            start: 0,
            finals,
            transition_offsets: vec![0, 0],
            pending_transitions: Vec::new(),
            finalized: true,
        }
    }

    /// Finalize the NFA by building CSR transition table (H9 optimization).
    pub fn finalize(&mut self) {
        if self.finalized && self.pending_transitions.is_empty() {
            return;
        }

        if !self.pending_transitions.is_empty() {
            self.transitions.append(&mut self.pending_transitions);
        }

        self.transitions.sort_by_key(|t| t.from);

        let num_states = self.states.len();
        let mut offsets = vec![0usize; num_states + 1];

        for trans in &self.transitions {
            if (trans.from as usize) < num_states {
                offsets[trans.from as usize + 1] += 1;
            }
        }

        for i in 1..=num_states {
            offsets[i] += offsets[i - 1];
        }

        self.transition_offsets = offsets;
        self.finalized = true;
    }

    /// Get the initial state.
    #[inline]
    pub fn start(&self) -> StateId {
        self.start
    }

    /// Get the set of final states.
    #[inline]
    pub fn finals(&self) -> &FxHashSet<StateId> {
        &self.finals
    }

    /// Get all states.
    #[inline]
    pub fn states(&self) -> &[NFAState] {
        &self.states
    }

    /// Get all transitions.
    #[inline]
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }

    /// Get the number of states.
    #[inline]
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Get the number of transitions.
    #[inline]
    pub fn num_transitions(&self) -> usize {
        self.transitions.len()
    }

    /// Check if the NFA is empty (accepts no strings including epsilon).
    ///
    /// An NFA is considered empty if it has no final states.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.finals.is_empty()
    }

    /// Check if a state is final.
    #[inline]
    pub fn is_final(&self, state: StateId) -> bool {
        self.finals.contains(&state)
    }

    /// Add a new state and return its ID.
    pub fn add_state(&mut self, is_final: bool) -> StateId {
        let id = self.states.len() as StateId;
        let state = NFAState::new(id, is_final);
        self.states.push(state);
        if is_final {
            self.finals.insert(id);
        }
        // Extend offsets array for new state (H9)
        self.transition_offsets
            .push(self.transition_offsets.last().copied().unwrap_or(0));
        id
    }

    /// Mark a state as final.
    pub fn set_final(&mut self, state: StateId, is_final: bool) {
        if let Some(s) = self.states.get_mut(state as usize) {
            s.is_final = is_final;
            if is_final {
                self.finals.insert(state);
            } else {
                self.finals.remove(&state);
            }
        }
    }

    /// Add a transition with the given label.
    pub fn add_transition(&mut self, from: StateId, label: TransitionLabel, to: StateId) {
        self.add_transition_weighted(from, label, to, 0.0);
    }

    /// Add a weighted transition.
    pub fn add_transition_weighted(
        &mut self,
        from: StateId,
        label: TransitionLabel,
        to: StateId,
        weight: f64,
    ) {
        let trans = Transition::with_weight(from, label, to, weight);
        // H9: Add to pending transitions (will be sorted on finalize)
        self.pending_transitions.push(trans);
        self.finalized = false;
    }

    /// Add an epsilon transition.
    #[inline]
    pub fn add_epsilon(&mut self, from: StateId, to: StateId) {
        self.add_transition(from, TransitionLabel::Epsilon, to);
    }

    /// Add a transition on a single byte.
    #[inline]
    pub fn add_transition_byte(&mut self, from: StateId, b: u8, to: StateId) {
        self.add_transition(from, TransitionLabel::Byte(b), to);
    }

    /// Add a transition on a character class.
    #[inline]
    pub fn add_transition_class(&mut self, from: StateId, class: CharClass, to: StateId) {
        self.add_transition(from, TransitionLabel::CharClass(class), to);
    }

    /// Get all transitions from a given state (H9: O(1) CSR lookup when finalized).
    #[inline]
    pub fn transitions_from(&self, state: StateId) -> TransitionsFrom<'_> {
        if self.finalized && self.pending_transitions.is_empty() {
            // CSR fast path
            let state_idx = state as usize;
            if state_idx + 1 < self.transition_offsets.len() {
                let start = self.transition_offsets[state_idx];
                let end = self.transition_offsets[state_idx + 1];
                TransitionsFrom::Slice(&self.transitions[start..end])
            } else {
                TransitionsFrom::Slice(&[])
            }
        } else {
            // Fallback: linear scan through pending_transitions
            TransitionsFrom::Pending(
                &self.pending_transitions,
                &self.transitions,
                &self.transition_offsets,
                state,
            )
        }
    }

    /// Compute the epsilon closure of a single state.
    pub fn epsilon_closure_single(&self, state: StateId) -> StateSet {
        let mut closure = StateSet::new();
        let mut queue = VecDeque::new();

        closure.insert(state);
        queue.push_back(state);

        while let Some(current) = queue.pop_front() {
            for trans in self.transitions_from(current) {
                if trans.label.is_epsilon() && !closure.contains(&trans.to) {
                    closure.insert(trans.to);
                    queue.push_back(trans.to);
                }
            }
        }

        closure
    }

    /// Compute the epsilon closure of a set of states.
    pub fn epsilon_closure(&self, states: &StateSet) -> StateSet {
        let mut closure = StateSet::new();
        let mut queue = VecDeque::new();

        for state in states.iter() {
            if closure.insert(state) {
                queue.push_back(state);
            }
        }

        while let Some(current) = queue.pop_front() {
            for trans in self.transitions_from(current) {
                if trans.label.is_epsilon() && !closure.contains(&trans.to) {
                    closure.insert(trans.to);
                    queue.push_back(trans.to);
                }
            }
        }

        closure
    }

    /// Compute the set of states reachable from a state set on input byte `b`.
    pub fn move_on_byte(&self, states: &StateSet, b: u8) -> StateSet {
        let mut result = StateSet::new();

        for state in states.iter() {
            for trans in self.transitions_from(state) {
                if !trans.label.is_epsilon() && !trans.label.is_anchor() && trans.label.matches(b) {
                    result.insert(trans.to);
                }
            }
        }

        result
    }

    /// Compute the set of states reachable from a state set on anchor assertions.
    pub fn move_on_anchors(
        &self,
        states: &StateSet,
        input: &[u8],
        pos: usize,
        multiline: bool,
    ) -> StateSet {
        let mut result = StateSet::new();

        for state in states.iter() {
            for trans in self.transitions_from(state) {
                if trans.label.is_anchor() && trans.label.matches_at_position(input, pos, multiline)
                {
                    result.insert(trans.to);
                }
            }
        }

        result
    }

    /// Get the number of states (alias for num_states for compatibility).
    #[inline]
    pub fn state_count(&self) -> usize {
        self.num_states()
    }

    /// Check if the NFA accepts a byte string.
    pub fn accepts(&self, input: &[u8]) -> bool {
        self.accepts_with_flags(input, false, false)
    }

    /// Check if the NFA accepts a byte string with flag support (full-match semantics).
    pub fn accepts_with_flags(&self, input: &[u8], multiline: bool, _dotall: bool) -> bool {
        let mut current = self.epsilon_closure_single(self.start);

        // Process anchors at start position (position 0)
        loop {
            let anchor_moved = self.move_on_anchors(&current, input, 0, multiline);
            if anchor_moved.is_empty() {
                break;
            }
            let anchor_closure = self.epsilon_closure(&anchor_moved);
            if anchor_closure.is_subset(&current) {
                break;
            }
            current.extend(&anchor_closure);
        }

        for (pos, &b) in input.iter().enumerate() {
            // Move on byte
            let byte_moved = self.move_on_byte(&current, b);
            if byte_moved.is_empty() {
                // No transition - check if \Z with trailing newlines
                if current.iter().any(|s| self.is_final(s))
                    && input[pos..].iter().all(|&byte| byte == b'\n')
                {
                    return true;
                }
                return false;
            }
            current = self.epsilon_closure(&byte_moved);

            // Process anchors at position after this byte
            let next_pos = pos + 1;
            loop {
                let anchor_moved = self.move_on_anchors(&current, input, next_pos, multiline);
                if anchor_moved.is_empty() {
                    break;
                }
                let anchor_closure = self.epsilon_closure(&anchor_moved);
                let prev_len = current.len();
                current.extend(&anchor_closure);
                if current.len() == prev_len {
                    break;
                }
            }
        }

        current.iter().any(|s| self.is_final(s))
    }

    /// Search for the pattern anywhere in the input (search semantics).
    pub fn search_with_flags(&self, input: &[u8], multiline: bool, dotall: bool) -> bool {
        let start_closure = self.epsilon_closure_single(self.start);
        let has_start_anchor = start_closure.iter().any(|s| {
            self.transitions_from(s).iter().any(|t| {
                matches!(
                    t.label,
                    TransitionLabel::StartOfLine | TransitionLabel::StartOfInput
                )
            })
        });

        if has_start_anchor {
            self.try_match_from_byte(input, 0, multiline, dotall)
        } else {
            for start_pos in 0..=input.len() {
                if self.try_match_from_byte(input, start_pos, multiline, dotall) {
                    return true;
                }
            }
            false
        }
    }

    /// Try to match the NFA starting from a specific byte position.
    fn try_match_from_byte(
        &self,
        input: &[u8],
        start_pos: usize,
        multiline: bool,
        _dotall: bool,
    ) -> bool {
        let mut current = self.epsilon_closure_single(self.start);

        // Process anchors at start position
        loop {
            let anchor_moved = self.move_on_anchors(&current, input, start_pos, multiline);
            if anchor_moved.is_empty() {
                break;
            }
            let anchor_closure = self.epsilon_closure(&anchor_moved);
            if anchor_closure.is_subset(&current) {
                break;
            }
            current.extend(&anchor_closure);
        }

        for (idx, &b) in input[start_pos..].iter().enumerate() {
            let pos = start_pos + idx;

            // Move on byte
            let byte_moved = self.move_on_byte(&current, b);
            if byte_moved.is_empty() {
                // Check if we can accept here (search semantics)
                if current.iter().any(|s| self.is_final(s)) {
                    let needs_end_anchor = current.iter().any(|s| {
                        self.is_final(s)
                            && self.transitions_from(s).iter().any(|t| {
                                matches!(
                                    t.label,
                                    TransitionLabel::EndOfLine
                                        | TransitionLabel::EndOfInput
                                        | TransitionLabel::EndOfInputStrict
                                )
                            })
                    });

                    if !needs_end_anchor {
                        return true;
                    }

                    if input[pos..].iter().all(|&byte| byte == b'\n') {
                        return true;
                    }
                }
                return false;
            }
            current = self.epsilon_closure(&byte_moved);

            // Process anchors
            let next_pos = pos + 1;
            loop {
                let anchor_moved = self.move_on_anchors(&current, input, next_pos, multiline);
                if anchor_moved.is_empty() {
                    break;
                }
                let anchor_closure = self.epsilon_closure(&anchor_moved);
                let prev_len = current.len();
                current.extend(&anchor_closure);
                if current.len() == prev_len {
                    break;
                }
            }
        }

        current.iter().any(|s| self.is_final(s))
    }

    /// Search for the pattern anywhere in the input (convenience method).
    pub fn search(&self, input: &[u8]) -> bool {
        self.search_with_flags(input, false, false)
    }

    /// Search for the pattern in a string (as UTF-8 bytes).
    pub fn search_str(&self, input: &str) -> bool {
        self.search(input.as_bytes())
    }

    /// Check if the NFA accepts a string (as UTF-8 bytes).
    pub fn accepts_str(&self, input: &str) -> bool {
        self.accepts(input.as_bytes())
    }

    /// Check if the NFA accepts a string with flags (as UTF-8 bytes).
    pub fn accepts_str_with_flags(&self, input: &str, multiline: bool, dotall: bool) -> bool {
        self.accepts_with_flags(input.as_bytes(), multiline, dotall)
    }

    /// Rebuild the CSR transition table (H9: call after modifying transitions directly).
    pub fn rebuild_index(&mut self) {
        self.finalized = false;
        self.finalize();
    }

    /// Shift all state IDs by an offset.
    fn shift_states(&mut self, offset: StateId) {
        self.start += offset;

        for state in &mut self.states {
            state.id += offset;
        }

        for trans in &mut self.transitions {
            trans.from += offset;
            trans.to += offset;
        }

        // H9: Also shift pending_transitions
        for trans in &mut self.pending_transitions {
            trans.from += offset;
            trans.to += offset;
        }

        let old_finals: Vec<_> = self.finals.drain().collect();
        for f in old_finals {
            self.finals.insert(f + offset);
        }

        self.rebuild_index();
    }

    /// Create a new NFA that is the union of `self` and `other`.
    pub fn union(self, other: NFA) -> NFA {
        let mut result = NFA::new();

        let new_start = 0;

        let offset_a = 1;
        let mut nfa_a = self;
        nfa_a.shift_states(offset_a);
        // H9: Ensure pending transitions are merged before copying
        nfa_a.finalize();

        let offset_b = offset_a + nfa_a.num_states() as StateId;
        let mut nfa_b = other;
        nfa_b.shift_states(offset_b);
        // H9: Ensure pending transitions are merged before copying
        nfa_b.finalize();

        let new_final = offset_b + nfa_b.num_states() as StateId;

        result.states.clear();
        result.states.push(NFAState::non_final(new_start));
        result.states.extend(nfa_a.states.iter().cloned());
        result.states.extend(nfa_b.states.iter().cloned());
        result.states.push(NFAState::final_state(new_final));

        result.start = new_start;
        result.finals.clear();
        result.finals.insert(new_final);

        result.transitions = nfa_a.transitions;
        result.transitions.extend(nfa_b.transitions);

        result.transitions.push(Transition::epsilon(new_start, nfa_a.start));
        result.transitions.push(Transition::epsilon(new_start, nfa_b.start));

        for &f in &nfa_a.finals {
            result.transitions.push(Transition::epsilon(f, new_final));
        }
        for &f in &nfa_b.finals {
            result.transitions.push(Transition::epsilon(f, new_final));
        }

        result.rebuild_index();
        result
    }

    /// Create a new NFA that is the concatenation of `self` and `other`.
    pub fn concatenate(self, other: NFA) -> NFA {
        let mut result = NFA::new();

        // H9: Ensure pending transitions are merged before copying
        let mut nfa_a = self;
        nfa_a.finalize();

        let offset_b = nfa_a.num_states() as StateId;
        let mut nfa_b = other;
        nfa_b.shift_states(offset_b);
        // H9: Ensure pending transitions are merged before copying
        nfa_b.finalize();

        result.states = nfa_a.states;
        result.states.extend(nfa_b.states.iter().cloned());

        result.start = nfa_a.start;
        result.finals = nfa_b.finals.clone();

        result.transitions = nfa_a.transitions;
        result.transitions.extend(nfa_b.transitions);

        for &f in &nfa_a.finals {
            result.transitions.push(Transition::epsilon(f, nfa_b.start));
        }

        result.rebuild_index();
        result
    }

    /// Create a new NFA that is the Kleene star of `self`.
    pub fn kleene_star(self) -> NFA {
        let mut result = NFA::new();

        let offset_a = 1;
        let mut nfa_a = self;
        nfa_a.shift_states(offset_a);
        // H9: Ensure pending transitions are merged before copying
        nfa_a.finalize();

        let new_start = 0;
        let new_final = offset_a + nfa_a.num_states() as StateId;

        result.states.clear();
        result.states.push(NFAState::non_final(new_start));
        result.states.extend(nfa_a.states.iter().cloned());
        result.states.push(NFAState::final_state(new_final));

        result.start = new_start;
        result.finals.clear();
        result.finals.insert(new_final);

        result.transitions = nfa_a.transitions;

        result.transitions.push(Transition::epsilon(new_start, nfa_a.start));
        result.transitions.push(Transition::epsilon(new_start, new_final));

        for &f in &nfa_a.finals {
            result.transitions.push(Transition::epsilon(f, new_final));
        }
        for &f in &nfa_a.finals {
            result.transitions.push(Transition::epsilon(f, nfa_a.start));
        }

        result.rebuild_index();
        result
    }

    /// Create a new NFA that is the Kleene plus of `self`.
    pub fn kleene_plus(self) -> NFA {
        let star = self.clone().kleene_star();
        self.concatenate(star)
    }

    /// Create a new NFA that is `self` made optional.
    pub fn optional(self) -> NFA {
        let mut result = NFA::new();

        let offset_a = 1;
        let mut nfa_a = self;
        nfa_a.shift_states(offset_a);
        // H9: Ensure pending transitions are merged before copying
        nfa_a.finalize();

        let new_start = 0;
        let new_final = offset_a + nfa_a.num_states() as StateId;

        result.states.clear();
        result.states.push(NFAState::non_final(new_start));
        result.states.extend(nfa_a.states.iter().cloned());
        result.states.push(NFAState::final_state(new_final));

        result.start = new_start;
        result.finals.clear();
        result.finals.insert(new_final);

        result.transitions = nfa_a.transitions;

        result.transitions.push(Transition::epsilon(new_start, nfa_a.start));
        result.transitions.push(Transition::epsilon(new_start, new_final));

        for &f in &nfa_a.finals {
            result.transitions.push(Transition::epsilon(f, new_final));
        }

        result.rebuild_index();
        result
    }

    // ========================================================================
    // Optimization Methods
    // ========================================================================

    /// Optimize the NFA with full optimization (all passes enabled).
    ///
    /// This is the recommended method for most use cases. It eliminates epsilon
    /// transitions, removes unreachable states, removes dead states, and
    /// deduplicates transitions.
    ///
    /// # Returns
    ///
    /// A new, optimized NFA that accepts the same language.
    pub fn optimize(&self) -> NFA {
        use super::optimizer::{NfaOptimizer, OptimizationConfig};
        let optimizer = NfaOptimizer::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(self.clone());
        optimized
    }

    /// Optimize the NFA with custom configuration, returning statistics.
    ///
    /// This method allows fine-grained control over which optimization passes
    /// to apply and returns detailed statistics about the optimization.
    pub fn optimize_with(
        &self,
        config: super::optimizer::OptimizationConfig,
    ) -> (NFA, super::optimizer::OptimizationStats) {
        use super::optimizer::NfaOptimizer;
        let optimizer = NfaOptimizer::new(config);
        optimizer.optimize(self.clone())
    }

    /// Count the number of epsilon transitions in this NFA.
    pub fn count_epsilon_transitions(&self) -> usize {
        self.transitions.iter().filter(|t| t.label.is_epsilon()).count()
    }
}

impl Default for NFA {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NFA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "NFA (byte-level):")?;
        writeln!(f, "  States: {}", self.num_states())?;
        writeln!(f, "  Transitions: {}", self.num_transitions())?;
        writeln!(f, "  Start: q{}", self.start)?;
        writeln!(
            f,
            "  Finals: {{{}}}",
            self.finals
                .iter()
                .map(|s| format!("q{}", s))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        writeln!(f, "  Transitions:")?;
        for trans in &self.transitions {
            writeln!(f, "    {}", trans)?;
        }
        Ok(())
    }
}
