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

use std::collections::{HashMap, VecDeque};
use std::fmt;

use rustc_hash::FxHashSet;

use super::types::{
    CharClass, CharClassChar, NFAState, StateId, Transition, TransitionChar, TransitionLabel,
    TransitionLabelChar,
};

// ============================================================================
// NFA (Character-level)
// ============================================================================

/// Nondeterministic Finite Automaton (character-level).
///
/// An NFA for matching phonetic patterns over Unicode characters.
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
pub struct NFAChar {
    /// All states in the NFA
    states: Vec<NFAState>,
    /// All transitions
    transitions: Vec<TransitionChar>,
    /// Initial state
    start: StateId,
    /// Set of final (accepting) states
    finals: FxHashSet<StateId>,
    /// Index: state -> indices of outgoing transitions
    transition_index: HashMap<StateId, Vec<usize>>,
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
            transition_index: HashMap::new(),
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
            transition_index: HashMap::new(),
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
        let idx = self.transitions.len();
        self.transitions.push(trans);

        self.transition_index
            .entry(from)
            .or_insert_with(Vec::new)
            .push(idx);
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

    /// Get all transitions from a given state.
    pub fn transitions_from(&self, state: StateId) -> impl Iterator<Item = &TransitionChar> {
        self.transition_index
            .get(&state)
            .map(|indices| indices.iter().map(|&i| &self.transitions[i]))
            .into_iter()
            .flatten()
    }

    /// Compute the epsilon closure of a single state.
    ///
    /// The epsilon closure is the set of all states reachable from the given state
    /// by following zero or more epsilon transitions.
    pub fn epsilon_closure_single(&self, state: StateId) -> FxHashSet<StateId> {
        let mut closure = FxHashSet::default();
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
    pub fn epsilon_closure(&self, states: &FxHashSet<StateId>) -> FxHashSet<StateId> {
        let mut closure = FxHashSet::default();
        let mut queue = VecDeque::new();

        for &state in states {
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
    pub fn move_on_char(&self, states: &FxHashSet<StateId>, c: char) -> FxHashSet<StateId> {
        let mut result = FxHashSet::default();

        for &state in states {
            for trans in self.transitions_from(state) {
                if !trans.label.is_epsilon() && trans.label.matches(c) {
                    result.insert(trans.to);
                }
            }
        }

        result
    }

    /// Check if the NFA accepts a string.
    ///
    /// Uses the standard NFA simulation algorithm:
    /// 1. Start with epsilon closure of initial state
    /// 2. For each input character, compute move and epsilon closure
    /// 3. Accept if final state set intersects with accepting states
    pub fn accepts(&self, input: &str) -> bool {
        let mut current = self.epsilon_closure_single(self.start);

        for c in input.chars() {
            let moved = self.move_on_char(&current, c);
            if moved.is_empty() {
                return false;
            }
            current = self.epsilon_closure(&moved);
        }

        current.iter().any(|&s| self.is_final(s))
    }

    /// Rebuild the transition index (call after modifying transitions directly).
    pub fn rebuild_index(&mut self) {
        self.transition_index.clear();
        for (idx, trans) in self.transitions.iter().enumerate() {
            self.transition_index
                .entry(trans.from)
                .or_insert_with(Vec::new)
                .push(idx);
        }
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

        // Offset for other's states (after self)
        let offset_b = offset_a + nfa_a.num_states() as StateId;
        let mut nfa_b = other;
        nfa_b.shift_states(offset_b);

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
        let nfa_a = self;

        // Other starts after self's states
        let offset_b = nfa_a.num_states() as StateId;
        let mut nfa_b = other;
        nfa_b.shift_states(offset_b);

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
#[derive(Debug, Clone)]
pub struct NFA {
    /// All states in the NFA
    states: Vec<NFAState>,
    /// All transitions
    transitions: Vec<Transition>,
    /// Initial state
    start: StateId,
    /// Set of final (accepting) states
    finals: FxHashSet<StateId>,
    /// Index: state -> indices of outgoing transitions
    transition_index: HashMap<StateId, Vec<usize>>,
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
            transition_index: HashMap::new(),
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
            transition_index: HashMap::new(),
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
        let idx = self.transitions.len();
        self.transitions.push(trans);

        self.transition_index
            .entry(from)
            .or_insert_with(Vec::new)
            .push(idx);
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

    /// Get all transitions from a given state.
    pub fn transitions_from(&self, state: StateId) -> impl Iterator<Item = &Transition> {
        self.transition_index
            .get(&state)
            .map(|indices| indices.iter().map(|&i| &self.transitions[i]))
            .into_iter()
            .flatten()
    }

    /// Compute the epsilon closure of a single state.
    pub fn epsilon_closure_single(&self, state: StateId) -> FxHashSet<StateId> {
        let mut closure = FxHashSet::default();
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
    pub fn epsilon_closure(&self, states: &FxHashSet<StateId>) -> FxHashSet<StateId> {
        let mut closure = FxHashSet::default();
        let mut queue = VecDeque::new();

        for &state in states {
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
    pub fn move_on_byte(&self, states: &FxHashSet<StateId>, b: u8) -> FxHashSet<StateId> {
        let mut result = FxHashSet::default();

        for &state in states {
            for trans in self.transitions_from(state) {
                if !trans.label.is_epsilon() && trans.label.matches(b) {
                    result.insert(trans.to);
                }
            }
        }

        result
    }

    /// Check if the NFA accepts a byte string.
    pub fn accepts(&self, input: &[u8]) -> bool {
        let mut current = self.epsilon_closure_single(self.start);

        for &b in input {
            let moved = self.move_on_byte(&current, b);
            if moved.is_empty() {
                return false;
            }
            current = self.epsilon_closure(&moved);
        }

        current.iter().any(|&s| self.is_final(s))
    }

    /// Check if the NFA accepts a string (as UTF-8 bytes).
    pub fn accepts_str(&self, input: &str) -> bool {
        self.accepts(input.as_bytes())
    }

    /// Rebuild the transition index.
    pub fn rebuild_index(&mut self) {
        self.transition_index.clear();
        for (idx, trans) in self.transitions.iter().enumerate() {
            self.transition_index
                .entry(trans.from)
                .or_insert_with(Vec::new)
                .push(idx);
        }
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

        let offset_b = offset_a + nfa_a.num_states() as StateId;
        let mut nfa_b = other;
        nfa_b.shift_states(offset_b);

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

        let nfa_a = self;

        let offset_b = nfa_a.num_states() as StateId;
        let mut nfa_b = other;
        nfa_b.shift_states(offset_b);

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
}
