//! Core NFA data structure for phonetic regular expressions (character-level).
//!
//! This module implements the character-level NFA (Nondeterministic Finite
//! Automaton) data structure used for phonetic pattern matching. The NFA
//! supports:
//!
//! - Epsilon transitions (ε-moves)
//! - Character and character class transitions
//! - Weighted transitions for phonetic cost modeling
//! - Efficient epsilon closure computation
//! - Union, concatenation, and Kleene star operations
//!
//! # Design
//!
//! The NFA uses an adjacency list representation for transitions, optimized for:
//! - Fast transition lookup by source state
//! - Efficient epsilon closure computation
//! - Memory-efficient storage of sparse transition graphs
//!
//! # Formal Specification
//!
//! From `docs/wfst/nfa_phonetic_regex.md` Section 3:
//!
//! ```text
//! NFA = ⟨Q, Σ, δ, q₀, F⟩
//! Where:
//!   Q  = finite set of states
//!   Σ  = input alphabet
//!   δ  = transition function: Q × (Σ ∪ {ε}) → P(Q)
//!   q₀ = initial state
//!   F  = set of final (accepting) states
//! ```

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use std::collections::VecDeque;
use std::fmt;

use rustc_hash::FxHashSet;

use super::state_set::StateSet;
use super::types::{
    CharClassChar, NFAState, StateId, TransitionChar, TransitionLabelChar,
};

// ============================================================================
// Transitions Iterator Types (H9)
// ============================================================================

/// Iterator wrapper for transitions from a state (character-level).
///
/// This enum allows `transitions_from()` to return either a CSR slice (fast path)
/// or fall back to linear scanning for non-finalized NFAs.
pub enum TransitionsFromChar<'a> {
    /// Fast path: direct slice from CSR structure
    Slice(&'a [TransitionChar]),
    /// Fallback: references to pending and finalized transitions for filtering
    Pending(
        &'a [TransitionChar], // pending_transitions
        &'a [TransitionChar], // transitions (finalized)
        &'a [usize],          // transition_offsets
        StateId,
    ),
}

impl<'a> TransitionsFromChar<'a> {
    /// Iterate over all transitions from this state.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &'a TransitionChar> + '_ {
        TransitionsFromCharIter::new(self)
    }
}

impl<'a> IntoIterator for TransitionsFromChar<'a> {
    type Item = &'a TransitionChar;
    type IntoIter = TransitionsFromCharOwned<'a>;

    fn into_iter(self) -> Self::IntoIter {
        TransitionsFromCharOwned::new(self)
    }
}

/// Owned iterator for TransitionsFromChar (for `for` loops).
pub struct TransitionsFromCharOwned<'a> {
    source: TransitionsFromChar<'a>,
    idx: usize,
    phase: u8,
}

impl<'a> TransitionsFromCharOwned<'a> {
    fn new(source: TransitionsFromChar<'a>) -> Self {
        Self {
            source,
            idx: 0,
            phase: 0,
        }
    }
}

impl<'a> Iterator for TransitionsFromCharOwned<'a> {
    type Item = &'a TransitionChar;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.source {
            TransitionsFromChar::Slice(slice) => {
                if self.idx < slice.len() {
                    let item = &slice[self.idx];
                    self.idx += 1;
                    Some(item)
                } else {
                    None
                }
            }
            TransitionsFromChar::Pending(pending, transitions, offsets, state) => {
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

/// Iterator for TransitionsFromChar.
struct TransitionsFromCharIter<'a, 'b> {
    source: &'b TransitionsFromChar<'a>,
    idx: usize,
    phase: u8, // 0 = CSR/slice, 1 = pending
}

impl<'a, 'b> TransitionsFromCharIter<'a, 'b> {
    fn new(source: &'b TransitionsFromChar<'a>) -> Self {
        Self {
            source,
            idx: 0,
            phase: 0,
        }
    }
}

impl<'a, 'b> Iterator for TransitionsFromCharIter<'a, 'b> {
    type Item = &'a TransitionChar;

    fn next(&mut self) -> Option<Self::Item> {
        match self.source {
            TransitionsFromChar::Slice(slice) => {
                if self.idx < slice.len() {
                    let item = &slice[self.idx];
                    self.idx += 1;
                    Some(item)
                } else {
                    None
                }
            }
            TransitionsFromChar::Pending(pending, transitions, offsets, state) => {
                // Phase 0: iterate CSR portion (if any)
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
                    // Switch to pending phase
                    self.phase = 1;
                    self.idx = 0;
                }
                // Phase 1: iterate pending_transitions with filter
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
// NFA (Character-level)
// ============================================================================

/// Nondeterministic Finite Automaton (character-level).
///
/// An NFA for matching phonetic patterns over Unicode characters.
///
/// # H9 Optimization: CSR Transition Table
///
/// Uses Compressed Sparse Row (CSR) format for transitions:
/// - `transitions`: contiguous array sorted by source state
/// - `transition_offsets`: `offsets[s]..offsets[s+1]` = transitions from state s
///
/// This provides O(1) lookup with no hash overhead and better cache locality.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::phonetic::nfa::NFAChar;
///
/// // Create empty NFA
/// let mut nfa = NFAChar::new();
///
/// // Add states
/// let q0 = nfa.add_state(false);  // initial, non-final
/// let q1 = nfa.add_state(true);   // final
///
/// // Add transition on 'a'
/// nfa.add_transition_char(q0, 'a', q1);
///
/// // Check acceptance
/// assert!(nfa.accepts("a"));
/// assert!(!nfa.accepts("b"));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct NFAChar {
    /// All states in the NFA
    states: Vec<NFAState>,
    /// All transitions (H9: sorted by source state for CSR access)
    transitions: Vec<TransitionChar>,
    /// Initial state
    start: StateId,
    /// Set of final (accepting) states
    finals: FxHashSet<StateId>,
    /// H9: CSR offsets - transitions for state s are at transitions[offsets[s]..offsets[s+1]]
    /// Length = num_states + 1 (last element is transitions.len())
    transition_offsets: Vec<usize>,
    /// Pending transitions before finalization (used during construction)
    pending_transitions: Vec<TransitionChar>,
    /// Whether the CSR structure is finalized
    finalized: bool,
}

impl NFAChar {
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
            transition_offsets: vec![0, 0], // One state, no transitions
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
    ///
    /// This must be called after all states and transitions are added,
    /// before any matching operations.
    pub fn finalize(&mut self) {
        if self.finalized && self.pending_transitions.is_empty() {
            return;
        }

        // Merge pending transitions into the main list
        if !self.pending_transitions.is_empty() {
            self.transitions.append(&mut self.pending_transitions);
        }

        // Sort transitions by source state for CSR format
        self.transitions.sort_by_key(|t| t.from);

        // Build offset array
        let num_states = self.states.len();
        let mut offsets = vec![0usize; num_states + 1];

        // Count transitions per state
        for trans in &self.transitions {
            if (trans.from as usize) < num_states {
                offsets[trans.from as usize + 1] += 1;
            }
        }

        // Convert counts to cumulative offsets
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
    pub fn transitions(&self) -> &[TransitionChar] {
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
        self.transition_offsets.push(self.transition_offsets.last().copied().unwrap_or(0));
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
    pub fn add_transition(&mut self, from: StateId, label: TransitionLabelChar, to: StateId) {
        self.add_transition_weighted(from, label, to, 0.0);
    }

    /// Add a weighted transition.
    pub fn add_transition_weighted(
        &mut self,
        from: StateId,
        label: TransitionLabelChar,
        to: StateId,
        weight: f64,
    ) {
        let trans = TransitionChar::with_weight(from, label, to, weight);
        // H9: Add to pending transitions (will be sorted on finalize)
        self.pending_transitions.push(trans);
        self.finalized = false;
    }

    /// Add an epsilon transition.
    #[inline]
    pub fn add_epsilon(&mut self, from: StateId, to: StateId) {
        self.add_transition(from, TransitionLabelChar::Epsilon, to);
    }

    /// Add a transition on a single character.
    #[inline]
    pub fn add_transition_char(&mut self, from: StateId, c: char, to: StateId) {
        self.add_transition(from, TransitionLabelChar::Char(c), to);
    }

    /// Add a transition on a character class.
    #[inline]
    pub fn add_transition_class(&mut self, from: StateId, class: CharClassChar, to: StateId) {
        self.add_transition(from, TransitionLabelChar::CharClass(class), to);
    }

    /// Get all transitions from a given state (H9: O(1) CSR lookup when finalized).
    ///
    /// When the NFA is finalized, this uses CSR for O(1) lookup.
    /// When not finalized, returns a collected slice from pending_transitions.
    #[inline]
    pub fn transitions_from(&self, state: StateId) -> TransitionsFromChar<'_> {
        if self.finalized && self.pending_transitions.is_empty() {
            // CSR fast path
            let state_idx = state as usize;
            if state_idx + 1 < self.transition_offsets.len() {
                let start = self.transition_offsets[state_idx];
                let end = self.transition_offsets[state_idx + 1];
                TransitionsFromChar::Slice(&self.transitions[start..end])
            } else {
                TransitionsFromChar::Slice(&[])
            }
        } else {
            // Fallback: linear scan through pending_transitions
            // This maintains backwards compatibility for tests that don't call finalize()
            TransitionsFromChar::Pending(
                &self.pending_transitions,
                &self.transitions,
                &self.transition_offsets,
                state,
            )
        }
    }

    /// Compute the epsilon closure of a single state.
    ///
    /// The epsilon closure is the set of all states reachable from the given state
    /// by following zero or more epsilon transitions.
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

    /// Compute the set of states reachable from a state set on input character `c`.
    ///
    /// This does NOT include epsilon closure - call `epsilon_closure` on the result
    /// if needed.
    pub fn move_on_char(&self, states: &StateSet, c: char) -> StateSet {
        let mut result = StateSet::new();

        for state in states.iter() {
            for trans in self.transitions_from(state) {
                if !trans.label.is_epsilon() && !trans.label.is_anchor() && trans.label.matches(c) {
                    result.insert(trans.to);
                }
            }
        }

        result
    }

    /// Compute the set of states reachable from a state set on anchor assertions.
    ///
    /// Anchors are zero-width assertions that match based on position in the input.
    /// This does NOT include epsilon closure - call `epsilon_closure` on the result.
    pub fn move_on_anchors(
        &self,
        states: &StateSet,
        input: &str,
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

    /// Check if the NFA accepts a string.
    ///
    /// Uses the standard NFA simulation algorithm:
    /// 1. Start with epsilon closure of initial state
    /// 2. For each input character, compute move and epsilon closure
    /// 3. Accept if final state set intersects with accepting states
    ///
    /// Note: This method does NOT handle anchor assertions. Use `accepts_with_flags`
    /// for patterns containing anchors like `^`, `$`, `\A`, `\Z`, `\z`.
    pub fn accepts(&self, input: &str) -> bool {
        self.accepts_with_flags(input, false, false)
    }

    /// Check if the NFA accepts a string with flag support (full-match semantics).
    ///
    /// This method requires the pattern to consume the entire input string.
    /// For search semantics (finding pattern anywhere in string), use `search_with_flags`.
    ///
    /// # Arguments
    ///
    /// * `input` - The string to match against
    /// * `multiline` - If true, `^` and `$` match at line boundaries (after/before `\n`)
    /// * `_dotall` - If true, `.` matches newlines (currently unused - handled at compile time)
    pub fn accepts_with_flags(&self, input: &str, multiline: bool, _dotall: bool) -> bool {
        let mut current = self.epsilon_closure_single(self.start);

        // Process anchors at start position (position 0)
        loop {
            let anchor_moved = self.move_on_anchors(&current, input, 0, multiline);
            if anchor_moved.is_empty() {
                break;
            }
            let anchor_closure = self.epsilon_closure(&anchor_moved);
            if anchor_closure.is_subset(&current) {
                break; // No new states reached
            }
            current.extend(&anchor_closure);
        }

        let chars: Vec<char> = input.chars().collect();
        let mut byte_pos = 0usize;

        for (char_idx, &c) in chars.iter().enumerate() {
            // Move on character
            let char_moved = self.move_on_char(&current, c);
            if char_moved.is_empty() {
                // No transition available - check if this is \Z with trailing newline
                // \Z matches before optional trailing newline, so we accept if:
                // 1. We're in a final state, AND
                // 2. Remaining input is only trailing newlines
                if current.iter().any(|s| self.is_final(s))
                    && chars[char_idx..].iter().all(|&ch| ch == '\n')
                {
                    return true;
                }
                return false;
            }
            current = self.epsilon_closure(&char_moved);

            // Update byte position for next anchor check
            byte_pos += c.len_utf8();

            // Process anchors at position after this character
            loop {
                let anchor_moved = self.move_on_anchors(&current, input, byte_pos, multiline);
                if anchor_moved.is_empty() {
                    break;
                }
                let anchor_closure = self.epsilon_closure(&anchor_moved);
                let prev_len = current.len();
                current.extend(&anchor_closure);
                if current.len() == prev_len {
                    break; // No new states reached
                }
            }

            // For multiline mode, check anchors after newlines
            if multiline && c == '\n' && char_idx + 1 < chars.len() {
                loop {
                    let anchor_moved = self.move_on_anchors(&current, input, byte_pos, multiline);
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
        }

        current.iter().any(|s| self.is_final(s))
    }

    /// Search for the pattern anywhere in the input string (search semantics).
    ///
    /// This method implements search semantics:
    /// - Patterns with `^` only match at the start (or line starts in multiline)
    /// - Patterns without `^` can match at any position
    /// - Patterns with `$` must end at the end (or line ends in multiline)
    /// - Patterns without `$` can end at any position
    ///
    /// # Arguments
    ///
    /// * `input` - The string to search in
    /// * `multiline` - If true, `^` and `$` match at line boundaries
    /// * `dotall` - If true, `.` matches newlines
    pub fn search_with_flags(&self, input: &str, multiline: bool, dotall: bool) -> bool {
        // Check if the pattern has a start anchor
        let start_closure = self.epsilon_closure_single(self.start);
        let has_start_anchor = start_closure.iter().any(|s| {
            self.transitions_from(s).iter().any(|t| {
                matches!(
                    t.label,
                    TransitionLabelChar::StartOfLine | TransitionLabelChar::StartOfInput
                )
            })
        });

        let chars: Vec<char> = input.chars().collect();

        if has_start_anchor {
            // Pattern has start anchor - only try matching from position 0
            self.try_match_from(&chars, input, 0, multiline, dotall)
        } else {
            // Pattern has no start anchor - try all positions
            for start_pos in 0..=chars.len() {
                if self.try_match_from(&chars, input, start_pos, multiline, dotall) {
                    return true;
                }
            }
            false
        }
    }

    /// Try to match the NFA starting from a specific character position.
    fn try_match_from(
        &self,
        chars: &[char],
        input: &str,
        start_char_pos: usize,
        multiline: bool,
        _dotall: bool,
    ) -> bool {
        let mut current = self.epsilon_closure_single(self.start);

        // Calculate byte position for the starting character position
        let start_byte_pos: usize = chars[..start_char_pos]
            .iter()
            .map(|c| c.len_utf8())
            .sum();

        // Process anchors at start position
        loop {
            let anchor_moved = self.move_on_anchors(&current, input, start_byte_pos, multiline);
            if anchor_moved.is_empty() {
                break;
            }
            let anchor_closure = self.epsilon_closure(&anchor_moved);
            if anchor_closure.is_subset(&current) {
                break;
            }
            current.extend(&anchor_closure);
        }

        let mut byte_pos = start_byte_pos;

        for (idx, &c) in chars[start_char_pos..].iter().enumerate() {
            let char_idx = start_char_pos + idx;

            // Move on character
            let char_moved = self.move_on_char(&current, c);
            if char_moved.is_empty() {
                // No character transition - check if we can accept here (search semantics)
                if current.iter().any(|s| self.is_final(s)) {
                    // Check if any final state has pending end anchor requirements
                    let needs_end_anchor = current.iter().any(|s| {
                        self.is_final(s)
                            && self.transitions_from(s).iter().any(|t| {
                                matches!(
                                    t.label,
                                    TransitionLabelChar::EndOfLine
                                        | TransitionLabelChar::EndOfInput
                                        | TransitionLabelChar::EndOfInputStrict
                                )
                            })
                    });

                    // If no end anchor requirement, accept
                    if !needs_end_anchor {
                        return true;
                    }

                    // If remaining is just trailing newline(s), accept for \Z
                    if chars[char_idx..].iter().all(|&ch| ch == '\n') {
                        return true;
                    }
                }
                return false;
            }
            current = self.epsilon_closure(&char_moved);

            // Update byte position
            byte_pos += c.len_utf8();

            // Process anchors
            loop {
                let anchor_moved = self.move_on_anchors(&current, input, byte_pos, multiline);
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

            // Multiline mode anchor check after newlines
            if multiline && c == '\n' && char_idx + 1 < chars.len() {
                loop {
                    let anchor_moved = self.move_on_anchors(&current, input, byte_pos, multiline);
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
        }

        current.iter().any(|s| self.is_final(s))
    }

    /// Search for the pattern anywhere in the input (convenience method).
    pub fn search(&self, input: &str) -> bool {
        self.search_with_flags(input, false, false)
    }

    /// Rebuild the CSR transition table (H9: call after modifying transitions directly).
    ///
    /// This re-finalizes the NFA, sorting transitions and rebuilding the offset table.
    pub fn rebuild_index(&mut self) {
        self.finalized = false;
        self.finalize();
    }

    /// Shift all state IDs by an offset.
    ///
    /// Used when combining NFAs to avoid state ID collisions.
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

        // Rebuild index since state IDs changed
        self.rebuild_index();
    }

    // ========================================================================
    // NFA Combination Operations
    // ========================================================================

    /// Create a new NFA that is the union of `self` and `other`.
    ///
    /// The resulting NFA accepts strings accepted by either `self` or `other`.
    ///
    /// ```text
    /// Union: L(A) ∪ L(B)
    ///
    ///        ε──→[A]──ε──→
    ///       /             \
    /// [q0]─┤               ├──→[qf]
    ///       \             /
    ///        ε──→[B]──ε──→
    /// ```
    pub fn union(self, other: NFAChar) -> NFAChar {
        let mut result = NFAChar::new();

        // New initial state (non-final)
        let new_start = 0;

        // Offset for self's states (after new start)
        let offset_a = 1;
        let mut nfa_a = self;
        nfa_a.shift_states(offset_a);
        // H9: Ensure pending transitions are merged before copying
        nfa_a.finalize();

        // Offset for other's states (after self)
        let offset_b = offset_a + nfa_a.num_states() as StateId;
        let mut nfa_b = other;
        nfa_b.shift_states(offset_b);
        // H9: Ensure pending transitions are merged before copying
        nfa_b.finalize();

        // New final state
        let new_final = offset_b + nfa_b.num_states() as StateId;

        // Build combined NFA
        result.states.clear();
        result.states.push(NFAState::non_final(new_start));
        result.states.extend(nfa_a.states.iter().cloned());
        result.states.extend(nfa_b.states.iter().cloned());
        result.states.push(NFAState::final_state(new_final));

        result.start = new_start;
        result.finals.clear();
        result.finals.insert(new_final);

        // Add all transitions from both NFAs
        result.transitions = nfa_a.transitions;
        result.transitions.extend(nfa_b.transitions);

        // Add epsilon transitions: new_start -> old starts
        result
            .transitions
            .push(TransitionChar::epsilon(new_start, nfa_a.start));
        result
            .transitions
            .push(TransitionChar::epsilon(new_start, nfa_b.start));

        // Add epsilon transitions: old finals -> new_final
        for &f in &nfa_a.finals {
            result
                .transitions
                .push(TransitionChar::epsilon(f, new_final));
        }
        for &f in &nfa_b.finals {
            result
                .transitions
                .push(TransitionChar::epsilon(f, new_final));
        }

        result.rebuild_index();
        result
    }

    /// Create a new NFA that is the concatenation of `self` and `other`.
    ///
    /// The resulting NFA accepts strings of the form `xy` where `x` is accepted
    /// by `self` and `y` is accepted by `other`.
    ///
    /// ```text
    /// Concatenation: L(A) · L(B)
    ///
    /// [A.start]──→[A]──→[A.finals]──ε──→[B.start]──→[B]──→[B.finals]
    /// ```
    pub fn concatenate(self, other: NFAChar) -> NFAChar {
        let mut result = NFAChar::new();

        // Self starts at offset 0
        // H9: Ensure pending transitions are merged before copying
        let mut nfa_a = self;
        nfa_a.finalize();

        // Other starts after self's states
        let offset_b = nfa_a.num_states() as StateId;
        let mut nfa_b = other;
        nfa_b.shift_states(offset_b);
        // H9: Ensure pending transitions are merged before copying
        nfa_b.finalize();

        // Build combined NFA
        result.states = nfa_a.states;
        result.states.extend(nfa_b.states.iter().cloned());

        result.start = nfa_a.start;
        result.finals = nfa_b.finals.clone();

        // Add all transitions
        result.transitions = nfa_a.transitions;
        result.transitions.extend(nfa_b.transitions);

        // Add epsilon transitions: A's finals -> B's start
        for &f in &nfa_a.finals {
            result
                .transitions
                .push(TransitionChar::epsilon(f, nfa_b.start));
        }

        result.rebuild_index();
        result
    }

    /// Create a new NFA that is the Kleene star of `self`.
    ///
    /// The resulting NFA accepts zero or more repetitions of strings accepted by `self`.
    ///
    /// ```text
    /// Kleene Star: L(A)*
    ///
    ///             ε (loop back)
    ///            ┌───────────┐
    ///            │           │
    /// [q0]──ε──→[A.start]──→[A.finals]──ε──→[qf]
    ///   │                                    ↑
    ///   └────────────ε───────────────────────┘
    /// ```
    pub fn kleene_star(self) -> NFAChar {
        let mut result = NFAChar::new();

        // Offset for self's states (after new start)
        let offset_a = 1;
        let mut nfa_a = self;
        nfa_a.shift_states(offset_a);
        // H9: Ensure pending transitions are merged before copying
        nfa_a.finalize();

        // New start and final states
        let new_start = 0;
        let new_final = offset_a + nfa_a.num_states() as StateId;

        // Build NFA
        result.states.clear();
        result.states.push(NFAState::non_final(new_start));
        result.states.extend(nfa_a.states.iter().cloned());
        result.states.push(NFAState::final_state(new_final));

        result.start = new_start;
        result.finals.clear();
        result.finals.insert(new_final);

        result.transitions = nfa_a.transitions;

        // Epsilon: new_start -> old_start
        result
            .transitions
            .push(TransitionChar::epsilon(new_start, nfa_a.start));

        // Epsilon: new_start -> new_final (accept empty string)
        result
            .transitions
            .push(TransitionChar::epsilon(new_start, new_final));

        // Epsilon: old_finals -> new_final
        for &f in &nfa_a.finals {
            result
                .transitions
                .push(TransitionChar::epsilon(f, new_final));
        }

        // Epsilon: old_finals -> old_start (loop back)
        for &f in &nfa_a.finals {
            result
                .transitions
                .push(TransitionChar::epsilon(f, nfa_a.start));
        }

        result.rebuild_index();
        result
    }

    /// Create a new NFA that is the Kleene plus of `self`.
    ///
    /// The resulting NFA accepts one or more repetitions of strings accepted by `self`.
    pub fn kleene_plus(self) -> NFAChar {
        // A+ = A · A*
        let star = self.clone().kleene_star();
        self.concatenate(star)
    }

    /// Create a new NFA that is `self` made optional.
    ///
    /// The resulting NFA accepts either the empty string or strings accepted by `self`.
    pub fn optional(self) -> NFAChar {
        let mut result = NFAChar::new();

        // Offset for self's states
        let offset_a = 1;
        let mut nfa_a = self;
        nfa_a.shift_states(offset_a);
        // H9: Ensure pending transitions are merged before copying
        nfa_a.finalize();

        // New start and final states
        let new_start = 0;
        let new_final = offset_a + nfa_a.num_states() as StateId;

        // Build NFA
        result.states.clear();
        result.states.push(NFAState::non_final(new_start));
        result.states.extend(nfa_a.states.iter().cloned());
        result.states.push(NFAState::final_state(new_final));

        result.start = new_start;
        result.finals.clear();
        result.finals.insert(new_final);

        result.transitions = nfa_a.transitions;

        // Epsilon: new_start -> old_start
        result
            .transitions
            .push(TransitionChar::epsilon(new_start, nfa_a.start));

        // Epsilon: new_start -> new_final (accept empty string)
        result
            .transitions
            .push(TransitionChar::epsilon(new_start, new_final));

        // Epsilon: old_finals -> new_final
        for &f in &nfa_a.finals {
            result
                .transitions
                .push(TransitionChar::epsilon(f, new_final));
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
    ///
    /// # Example
    ///
    /// ```ignore
    /// let nfa = compile(&regex)?;
    /// let optimized = nfa.optimize();
    /// assert!(optimized.accepts("hello") == nfa.accepts("hello"));
    /// ```
    pub fn optimize(&self) -> NFAChar {
        use super::optimizer::{NfaOptimizerChar, OptimizationConfig};
        let optimizer = NfaOptimizerChar::new(OptimizationConfig::full());
        let (optimized, _stats) = optimizer.optimize(self.clone());
        optimized
    }

    /// Optimize the NFA with custom configuration, returning statistics.
    ///
    /// This method allows fine-grained control over which optimization passes
    /// to apply and returns detailed statistics about the optimization.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying which passes to apply
    ///
    /// # Returns
    ///
    /// A tuple of (optimized NFA, optimization statistics).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use liblevenshtein::phonetic::nfa::optimizer::OptimizationConfig;
    ///
    /// let nfa = compile(&regex)?;
    /// let (optimized, stats) = nfa.optimize_with(OptimizationConfig::quick());
    /// println!("Removed {} states ({:.1}% reduction)",
    ///     stats.states_removed,
    ///     stats.state_reduction_percent());
    /// ```
    pub fn optimize_with(
        &self,
        config: super::optimizer::OptimizationConfig,
    ) -> (NFAChar, super::optimizer::OptimizationStats) {
        use super::optimizer::NfaOptimizerChar;
        let optimizer = NfaOptimizerChar::new(config);
        optimizer.optimize(self.clone())
    }

    /// Count the number of epsilon transitions in this NFA.
    pub fn count_epsilon_transitions(&self) -> usize {
        self.transitions.iter().filter(|t| t.label.is_epsilon()).count()
    }
}

impl Default for NFAChar {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NFAChar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "NFA (char-level):")?;
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
