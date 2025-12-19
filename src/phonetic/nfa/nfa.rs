//! Core NFA data structure for phonetic regular expressions.
//!
//! This module implements the NFA (Nondeterministic Finite Automaton) data structure
//! used for phonetic pattern matching. The NFA supports:
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
    CharClass, CharClassChar, NFAState, StateId, Transition, TransitionChar, TransitionLabel,
    TransitionLabelChar,
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

    /// Ensure the NFA is finalized before access.
    #[inline]
    fn ensure_finalized(&mut self) {
        if !self.finalized {
            self.finalize();
        }
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

    /// Ensure the NFA is finalized before access.
    #[inline]
    fn ensure_finalized(&mut self) {
        if !self.finalized {
            self.finalize();
        }
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- NFAChar basic tests ---

    #[test]
    fn test_nfa_char_new() {
        let nfa = NFAChar::new();
        assert_eq!(nfa.num_states(), 1);
        assert_eq!(nfa.num_transitions(), 0);
        assert_eq!(nfa.start(), 0);
        assert!(!nfa.is_final(0));
    }

    #[test]
    fn test_nfa_char_add_state() {
        let mut nfa = NFAChar::new();
        let q1 = nfa.add_state(false);
        let q2 = nfa.add_state(true);

        assert_eq!(q1, 1);
        assert_eq!(q2, 2);
        assert_eq!(nfa.num_states(), 3);
        assert!(!nfa.is_final(q1));
        assert!(nfa.is_final(q2));
    }

    #[test]
    fn test_nfa_char_simple_accept() {
        let mut nfa = NFAChar::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(true);

        nfa.add_transition_char(q0, 'a', q1);

        assert!(nfa.accepts("a"));
        assert!(!nfa.accepts("b"));
        assert!(!nfa.accepts(""));
        assert!(!nfa.accepts("aa"));
    }

    #[test]
    fn test_nfa_char_epsilon_closure() {
        let mut nfa = NFAChar::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(false);
        let q2 = nfa.add_state(true);

        nfa.add_epsilon(q0, q1);
        nfa.add_epsilon(q1, q2);

        let closure = nfa.epsilon_closure_single(q0);
        assert!(closure.contains(&q0));
        assert!(closure.contains(&q1));
        assert!(closure.contains(&q2));
        assert_eq!(closure.len(), 3);
    }

    #[test]
    fn test_nfa_char_accepts_with_epsilon() {
        let mut nfa = NFAChar::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(false);
        let q2 = nfa.add_state(true);

        nfa.add_epsilon(q0, q1);
        nfa.add_transition_char(q1, 'x', q2);

        assert!(nfa.accepts("x"));
        assert!(!nfa.accepts("y"));
    }

    // --- NFA combination tests ---

    #[test]
    fn test_nfa_char_union() {
        // NFA for "a"
        let mut nfa_a = NFAChar::new();
        let q0 = nfa_a.start();
        let q1 = nfa_a.add_state(true);
        nfa_a.add_transition_char(q0, 'a', q1);

        // NFA for "b"
        let mut nfa_b = NFAChar::new();
        let q0 = nfa_b.start();
        let q1 = nfa_b.add_state(true);
        nfa_b.add_transition_char(q0, 'b', q1);

        // Union: a | b
        let union = nfa_a.union(nfa_b);

        assert!(union.accepts("a"));
        assert!(union.accepts("b"));
        assert!(!union.accepts("c"));
        assert!(!union.accepts("ab"));
        assert!(!union.accepts(""));
    }

    #[test]
    fn test_nfa_char_concatenate() {
        // NFA for "a"
        let mut nfa_a = NFAChar::new();
        let q0 = nfa_a.start();
        let q1 = nfa_a.add_state(true);
        nfa_a.add_transition_char(q0, 'a', q1);

        // NFA for "b"
        let mut nfa_b = NFAChar::new();
        let q0 = nfa_b.start();
        let q1 = nfa_b.add_state(true);
        nfa_b.add_transition_char(q0, 'b', q1);

        // Concatenation: ab
        let concat = nfa_a.concatenate(nfa_b);

        assert!(concat.accepts("ab"));
        assert!(!concat.accepts("a"));
        assert!(!concat.accepts("b"));
        assert!(!concat.accepts("ba"));
        assert!(!concat.accepts(""));
    }

    #[test]
    fn test_nfa_char_kleene_star() {
        // NFA for "a"
        let mut nfa_a = NFAChar::new();
        let q0 = nfa_a.start();
        let q1 = nfa_a.add_state(true);
        nfa_a.add_transition_char(q0, 'a', q1);

        // Kleene star: a*
        let star = nfa_a.kleene_star();

        assert!(star.accepts(""));
        assert!(star.accepts("a"));
        assert!(star.accepts("aa"));
        assert!(star.accepts("aaa"));
        assert!(!star.accepts("b"));
        assert!(!star.accepts("ab"));
    }

    #[test]
    fn test_nfa_char_kleene_plus() {
        // NFA for "a"
        let mut nfa_a = NFAChar::new();
        let q0 = nfa_a.start();
        let q1 = nfa_a.add_state(true);
        nfa_a.add_transition_char(q0, 'a', q1);

        // Kleene plus: a+
        let plus = nfa_a.kleene_plus();

        assert!(!plus.accepts(""));
        assert!(plus.accepts("a"));
        assert!(plus.accepts("aa"));
        assert!(plus.accepts("aaa"));
        assert!(!plus.accepts("b"));
    }

    #[test]
    fn test_nfa_char_optional() {
        // NFA for "a"
        let mut nfa_a = NFAChar::new();
        let q0 = nfa_a.start();
        let q1 = nfa_a.add_state(true);
        nfa_a.add_transition_char(q0, 'a', q1);

        // Optional: a?
        let opt = nfa_a.optional();

        assert!(opt.accepts(""));
        assert!(opt.accepts("a"));
        assert!(!opt.accepts("aa"));
        assert!(!opt.accepts("b"));
    }

    // --- Byte-level NFA tests ---

    #[test]
    fn test_nfa_byte_simple() {
        let mut nfa = NFA::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(true);

        nfa.add_transition_byte(q0, b'a', q1);

        assert!(nfa.accepts(b"a"));
        assert!(nfa.accepts_str("a"));
        assert!(!nfa.accepts(b"b"));
        assert!(!nfa.accepts(b""));
    }

    #[test]
    fn test_nfa_byte_union() {
        let mut nfa_a = NFA::new();
        let q0 = nfa_a.start();
        let q1 = nfa_a.add_state(true);
        nfa_a.add_transition_byte(q0, b'a', q1);

        let mut nfa_b = NFA::new();
        let q0 = nfa_b.start();
        let q1 = nfa_b.add_state(true);
        nfa_b.add_transition_byte(q0, b'b', q1);

        let union = nfa_a.union(nfa_b);

        assert!(union.accepts_str("a"));
        assert!(union.accepts_str("b"));
        assert!(!union.accepts_str("c"));
    }

    #[test]
    fn test_nfa_byte_concatenate() {
        let mut nfa_a = NFA::new();
        let q0 = nfa_a.start();
        let q1 = nfa_a.add_state(true);
        nfa_a.add_transition_byte(q0, b'a', q1);

        let mut nfa_b = NFA::new();
        let q0 = nfa_b.start();
        let q1 = nfa_b.add_state(true);
        nfa_b.add_transition_byte(q0, b'b', q1);

        let concat = nfa_a.concatenate(nfa_b);

        assert!(concat.accepts_str("ab"));
        assert!(!concat.accepts_str("a"));
        assert!(!concat.accepts_str("b"));
    }

    // --- Complex pattern tests ---

    #[test]
    fn test_nfa_char_complex_pattern() {
        // Pattern: (a|b)*c
        // Build NFA for 'a'
        let mut nfa_a = NFAChar::new();
        let q0 = nfa_a.start();
        let q1 = nfa_a.add_state(true);
        nfa_a.add_transition_char(q0, 'a', q1);

        // Build NFA for 'b'
        let mut nfa_b = NFAChar::new();
        let q0 = nfa_b.start();
        let q1 = nfa_b.add_state(true);
        nfa_b.add_transition_char(q0, 'b', q1);

        // Build NFA for 'c'
        let mut nfa_c = NFAChar::new();
        let q0 = nfa_c.start();
        let q1 = nfa_c.add_state(true);
        nfa_c.add_transition_char(q0, 'c', q1);

        // (a|b)*
        let union = nfa_a.union(nfa_b);
        let star = union.kleene_star();

        // (a|b)*c
        let pattern = star.concatenate(nfa_c);

        assert!(pattern.accepts("c"));
        assert!(pattern.accepts("ac"));
        assert!(pattern.accepts("bc"));
        assert!(pattern.accepts("aac"));
        assert!(pattern.accepts("abc"));
        assert!(pattern.accepts("bac"));
        assert!(pattern.accepts("aabc"));
        assert!(!pattern.accepts(""));
        assert!(!pattern.accepts("a"));
        assert!(!pattern.accepts("b"));
        assert!(!pattern.accepts("ca"));
    }

    // --- Anchor tests ---

    #[test]
    fn test_nfa_char_start_of_line_anchor() {
        // Build NFA for ^hello
        let mut nfa = NFAChar::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(false);
        let q2 = nfa.add_state(false);
        let q3 = nfa.add_state(false);
        let q4 = nfa.add_state(false);
        let q5 = nfa.add_state(false);
        let q6 = nfa.add_state(true);

        // ^
        nfa.add_transition(q0, TransitionLabelChar::StartOfLine, q1);
        // hello
        nfa.add_transition_char(q1, 'h', q2);
        nfa.add_transition_char(q2, 'e', q3);
        nfa.add_transition_char(q3, 'l', q4);
        nfa.add_transition_char(q4, 'l', q5);
        nfa.add_transition_char(q5, 'o', q6);

        // Should match "hello" at start of input
        assert!(nfa.accepts("hello"));
        // Should not match with prefix
        assert!(!nfa.accepts("xhello"));
    }

    #[test]
    fn test_nfa_char_end_of_line_anchor() {
        // Build NFA for hello$
        let mut nfa = NFAChar::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(false);
        let q2 = nfa.add_state(false);
        let q3 = nfa.add_state(false);
        let q4 = nfa.add_state(false);
        let q5 = nfa.add_state(false);
        let q6 = nfa.add_state(true);

        // hello
        nfa.add_transition_char(q0, 'h', q1);
        nfa.add_transition_char(q1, 'e', q2);
        nfa.add_transition_char(q2, 'l', q3);
        nfa.add_transition_char(q3, 'l', q4);
        nfa.add_transition_char(q4, 'o', q5);
        // $
        nfa.add_transition(q5, TransitionLabelChar::EndOfLine, q6);

        // Should match "hello" at end of input
        assert!(nfa.accepts("hello"));
    }

    #[test]
    fn test_nfa_char_anchored_pattern() {
        // Build NFA for ^hello$
        let mut nfa = NFAChar::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(false);
        let q2 = nfa.add_state(false);
        let q3 = nfa.add_state(false);
        let q4 = nfa.add_state(false);
        let q5 = nfa.add_state(false);
        let q6 = nfa.add_state(false);
        let q7 = nfa.add_state(true);

        // ^
        nfa.add_transition(q0, TransitionLabelChar::StartOfLine, q1);
        // hello
        nfa.add_transition_char(q1, 'h', q2);
        nfa.add_transition_char(q2, 'e', q3);
        nfa.add_transition_char(q3, 'l', q4);
        nfa.add_transition_char(q4, 'l', q5);
        nfa.add_transition_char(q5, 'o', q6);
        // $
        nfa.add_transition(q6, TransitionLabelChar::EndOfLine, q7);

        // Should match exact "hello"
        assert!(nfa.accepts("hello"));
        // Should not match with extra chars
        assert!(!nfa.accepts("xhello"));
    }

    #[test]
    fn test_nfa_char_multiline_start_of_line() {
        // Build NFA for ^line
        let mut nfa = NFAChar::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(false);
        let q2 = nfa.add_state(false);
        let q3 = nfa.add_state(false);
        let q4 = nfa.add_state(false);
        let q5 = nfa.add_state(true);

        // ^
        nfa.add_transition(q0, TransitionLabelChar::StartOfLine, q1);
        // line
        nfa.add_transition_char(q1, 'l', q2);
        nfa.add_transition_char(q2, 'i', q3);
        nfa.add_transition_char(q3, 'n', q4);
        nfa.add_transition_char(q4, 'e', q5);

        // Without multiline, should only match at start
        assert!(nfa.accepts_with_flags("line", false, false));

        // In multiline mode, ^ should match after newline
        // But the NFA expects the entire input to match, not a substring
        // So "first\nline" won't match ^line in our whole-string matching
    }

    #[test]
    fn test_nfa_char_start_of_input_anchor() {
        // Build NFA for \Ahello
        let mut nfa = NFAChar::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(false);
        let q2 = nfa.add_state(false);
        let q3 = nfa.add_state(false);
        let q4 = nfa.add_state(false);
        let q5 = nfa.add_state(false);
        let q6 = nfa.add_state(true);

        // \A
        nfa.add_transition(q0, TransitionLabelChar::StartOfInput, q1);
        // hello
        nfa.add_transition_char(q1, 'h', q2);
        nfa.add_transition_char(q2, 'e', q3);
        nfa.add_transition_char(q3, 'l', q4);
        nfa.add_transition_char(q4, 'l', q5);
        nfa.add_transition_char(q5, 'o', q6);

        // \A only matches at absolute start, regardless of multiline mode
        assert!(nfa.accepts_with_flags("hello", true, false));
    }

    #[test]
    fn test_nfa_char_end_of_input_anchor() {
        // Build NFA for hello\Z
        let mut nfa = NFAChar::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(false);
        let q2 = nfa.add_state(false);
        let q3 = nfa.add_state(false);
        let q4 = nfa.add_state(false);
        let q5 = nfa.add_state(false);
        let q6 = nfa.add_state(true);

        // hello
        nfa.add_transition_char(q0, 'h', q1);
        nfa.add_transition_char(q1, 'e', q2);
        nfa.add_transition_char(q2, 'l', q3);
        nfa.add_transition_char(q3, 'l', q4);
        nfa.add_transition_char(q4, 'o', q5);
        // \Z
        nfa.add_transition(q5, TransitionLabelChar::EndOfInput, q6);

        // \Z matches at end, optionally with trailing newline
        assert!(nfa.accepts("hello"));
        assert!(nfa.accepts("hello\n"));
    }

    #[test]
    fn test_nfa_char_strict_end_of_input_anchor() {
        // Build NFA for hello\z
        let mut nfa = NFAChar::new();
        let q0 = nfa.start();
        let q1 = nfa.add_state(false);
        let q2 = nfa.add_state(false);
        let q3 = nfa.add_state(false);
        let q4 = nfa.add_state(false);
        let q5 = nfa.add_state(false);
        let q6 = nfa.add_state(true);

        // hello
        nfa.add_transition_char(q0, 'h', q1);
        nfa.add_transition_char(q1, 'e', q2);
        nfa.add_transition_char(q2, 'l', q3);
        nfa.add_transition_char(q3, 'l', q4);
        nfa.add_transition_char(q4, 'o', q5);
        // \z
        nfa.add_transition(q5, TransitionLabelChar::EndOfInputStrict, q6);

        // \z matches only at absolute end, no trailing newline
        assert!(nfa.accepts("hello"));
        assert!(!nfa.accepts("hello\n")); // \z doesn't allow trailing newline
    }
}
