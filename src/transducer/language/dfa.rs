//! A compact explicit DFA for small regular languages.

use super::LanguageAutomaton;
use std::error::Error;
use std::fmt::{self, Debug, Display};

/// Maximum number of real states representable by [`SmallDfa`].
///
/// Frontier sets use a dynamically sized `u64` bit-vector, so this limit is a
/// resource policy rather than an integer-width accident.
pub const SMALL_DFA_MAX_STATES: usize = 4_096;

/// Construction error for [`SmallDfa`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SmallDfaError {
    /// Adding another state would exceed the public state-set resource policy.
    TooManyStates {
        /// Number of states the requested operation would create.
        requested: usize,
        /// Fixed representable limit.
        maximum: usize,
    },
    /// A state ID does not identify a state in this DFA.
    UnknownState(u32),
}

impl Display for SmallDfaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyStates { requested, maximum } => write!(
                formatter,
                "small DFA requires {requested} states but supports at most {maximum}"
            ),
            Self::UnknownState(state) => write!(formatter, "unknown small-DFA state {state}"),
        }
    }
}

impl Error for SmallDfaError {}

/// A dynamically sized bit-set of active [`SmallDfa`] states.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmallDfaStateSet {
    words: Vec<u64>,
}

impl SmallDfaStateSet {
    fn zeroed(word_count: usize) -> Self {
        Self {
            words: vec![0; word_count],
        }
    }

    fn insert(&mut self, state: u32) {
        let state = state as usize;
        self.words[state / 64] |= 1_u64 << (state % 64);
    }
}

/// An explicit deterministic finite automaton with bit-vector frontier sets.
///
/// Transitions are stored as compact per-state tables. Missing transitions go
/// to an implicit sink that is never inserted into a state set. Although a DFA
/// occupies one state during exact recognition, edit operations can produce a
/// union of states, hence the bit-set representation.
#[derive(Clone, Debug)]
pub struct SmallDfa<U> {
    transitions: Vec<Vec<(U, u32)>>,
    accepting: Vec<u64>,
    start: u32,
}

impl<U> Default for SmallDfa<U> {
    fn default() -> Self {
        Self::new()
    }
}

impl<U> SmallDfa<U> {
    /// Construct a DFA with one non-accepting start state (`0`).
    pub fn new() -> Self {
        Self {
            transitions: vec![Vec::new()],
            accepting: vec![0],
            start: 0,
        }
    }

    /// Number of real states, excluding the implicit sink.
    pub fn state_count(&self) -> usize {
        self.transitions.len()
    }

    /// Add a state and return its numeric ID.
    pub fn add_state(&mut self, accepting: bool) -> Result<u32, SmallDfaError> {
        let requested = self.transitions.len() + 1;
        if requested > SMALL_DFA_MAX_STATES {
            return Err(SmallDfaError::TooManyStates {
                requested,
                maximum: SMALL_DFA_MAX_STATES,
            });
        }
        let id = self.transitions.len() as u32;
        self.transitions.push(Vec::new());
        let required_words = self.transitions.len().div_ceil(64);
        if self.accepting.len() < required_words {
            self.accepting.push(0);
        }
        self.set_accepting(id, accepting)?;
        Ok(id)
    }

    /// Select the start state.
    pub fn set_start(&mut self, state: u32) -> Result<(), SmallDfaError> {
        self.require_state(state)?;
        self.start = state;
        Ok(())
    }

    /// Mark or unmark an accepting state.
    pub fn set_accepting(&mut self, state: u32, accepting: bool) -> Result<(), SmallDfaError> {
        self.require_state(state)?;
        let state = state as usize;
        let bit = 1_u64 << (state % 64);
        if accepting {
            self.accepting[state / 64] |= bit;
        } else {
            self.accepting[state / 64] &= !bit;
        }
        Ok(())
    }

    fn require_state(&self, state: u32) -> Result<(), SmallDfaError> {
        if (state as usize) < self.transitions.len() {
            Ok(())
        } else {
            Err(SmallDfaError::UnknownState(state))
        }
    }
}

impl<U: Eq> SmallDfa<U> {
    /// Add or replace the deterministic transition `from --unit--> to`.
    pub fn add_transition(&mut self, from: u32, unit: U, to: u32) -> Result<(), SmallDfaError> {
        self.require_state(from)?;
        self.require_state(to)?;
        let edges = &mut self.transitions[from as usize];
        if let Some((_, target)) = edges.iter_mut().find(|(label, _)| label == &unit) {
            *target = to;
        } else {
            edges.push((unit, to));
        }
        Ok(())
    }
}

impl<U: Clone + Debug + Eq> LanguageAutomaton<U> for SmallDfa<U> {
    type StateSet = SmallDfaStateSet;

    fn empty(&self) -> Self::StateSet {
        SmallDfaStateSet::zeroed(self.accepting.len())
    }

    fn initial(&self) -> Self::StateSet {
        let mut states = self.empty();
        states.insert(self.start);
        states
    }

    fn is_empty(&self, states: &Self::StateSet) -> bool {
        states.words.iter().all(|word| *word == 0)
    }

    fn union_into(&self, target: &mut Self::StateSet, source: &Self::StateSet) {
        for (target_word, source_word) in target.words.iter_mut().zip(&source.words) {
            *target_word |= source_word;
        }
    }

    fn subtract(&self, target: &mut Self::StateSet, covered: &Self::StateSet) {
        for (target_word, covered_word) in target.words.iter_mut().zip(&covered.words) {
            *target_word &= !covered_word;
        }
    }

    fn step(&self, states: &Self::StateSet, unit: &U) -> Self::StateSet {
        let mut result = self.empty();
        for (word_index, word) in states.words.iter().copied().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let state = word_index * 64 + bit;
                if let Some((_, target)) = self.transitions[state]
                    .iter()
                    .find(|(label, _)| label == unit)
                {
                    result.insert(*target);
                }
            }
        }
        result
    }

    fn advance(&self, states: &Self::StateSet) -> Self::StateSet {
        let mut result = self.empty();
        for (word_index, word) in states.words.iter().copied().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let state = word_index * 64 + bit;
                for (_, target) in &self.transitions[state] {
                    result.insert(*target);
                }
            }
        }
        result
    }

    fn is_accepting(&self, states: &Self::StateSet) -> bool {
        states
            .words
            .iter()
            .zip(&self.accepting)
            .any(|(states, accepting)| states & accepting != 0)
    }

    fn state_count(&self) -> usize {
        self.transitions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::language::LanguageProduct;

    #[test]
    fn u64_literal_language_is_reachable_without_phonetic_features() {
        let mut dfa = SmallDfa::new();
        let one = dfa.add_state(false).unwrap();
        let two = dfa.add_state(true).unwrap();
        dfa.add_transition(0, 10u64, one).unwrap();
        dfa.add_transition(one, 20u64, two).unwrap();

        let product = LanguageProduct::new(dfa, 1);
        assert_eq!(product.distance_to_language([10u64, 20]), Some(0));
        assert_eq!(product.distance_to_language([10u64, 30]), Some(1));
        assert_eq!(product.distance_to_language([10u64]), Some(1));
        assert_eq!(product.distance_to_language([30u64, 40]), None);
    }

    #[test]
    fn state_limit_and_unknown_states_are_checked() {
        let mut dfa = SmallDfa::<u8>::new();
        for _ in 1..SMALL_DFA_MAX_STATES {
            dfa.add_state(false).unwrap();
        }
        assert_eq!(
            dfa.add_state(false),
            Err(SmallDfaError::TooManyStates {
                requested: SMALL_DFA_MAX_STATES + 1,
                maximum: SMALL_DFA_MAX_STATES,
            })
        );
        assert_eq!(
            dfa.set_start(SMALL_DFA_MAX_STATES as u32),
            Err(SmallDfaError::UnknownState(SMALL_DFA_MAX_STATES as u32))
        );
    }
}
