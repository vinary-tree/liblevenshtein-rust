//! [`LanguageAutomaton`]
//! adapters for the byte and Unicode phonetic NFAs.
//!
//! The NFAs themselves remain separate because their anchor and character-class
//! implementations differ. Only the product-facing state-set operations are
//! unified here.

use super::{NFAChar, StateSet, NFA};
use crate::transducer::language::LanguageAutomaton;

fn subtract_state_set(target: &mut StateSet, covered: &StateSet) {
    for state in covered {
        target.remove(&state);
    }
}

impl LanguageAutomaton<char> for NFAChar {
    type StateSet = StateSet;

    fn empty(&self) -> Self::StateSet {
        StateSet::new()
    }

    fn initial(&self) -> Self::StateSet {
        self.epsilon_closure_single(self.start())
    }

    fn is_empty(&self, states: &Self::StateSet) -> bool {
        states.is_empty()
    }

    fn union_into(&self, target: &mut Self::StateSet, source: &Self::StateSet) {
        target.extend(source);
    }

    fn subtract(&self, target: &mut Self::StateSet, covered: &Self::StateSet) {
        subtract_state_set(target, covered);
    }

    fn step(&self, states: &Self::StateSet, unit: &char) -> Self::StateSet {
        let moved = self.move_on_char(states, *unit);
        self.epsilon_closure(&moved)
    }

    fn advance(&self, states: &Self::StateSet) -> Self::StateSet {
        let mut moved = StateSet::new();
        for state in states {
            for transition in self.transitions_from(state) {
                if transition.label.consumes_input() {
                    moved.insert(transition.to);
                }
            }
        }
        self.epsilon_closure(&moved)
    }

    fn is_accepting(&self, states: &Self::StateSet) -> bool {
        states.iter().any(|state| self.is_final(state))
    }

    fn state_count(&self) -> usize {
        self.num_states()
    }
}

impl LanguageAutomaton<u8> for NFA {
    type StateSet = StateSet;

    fn empty(&self) -> Self::StateSet {
        StateSet::new()
    }

    fn initial(&self) -> Self::StateSet {
        self.epsilon_closure_single(self.start())
    }

    fn is_empty(&self, states: &Self::StateSet) -> bool {
        states.is_empty()
    }

    fn union_into(&self, target: &mut Self::StateSet, source: &Self::StateSet) {
        target.extend(source);
    }

    fn subtract(&self, target: &mut Self::StateSet, covered: &Self::StateSet) {
        subtract_state_set(target, covered);
    }

    fn step(&self, states: &Self::StateSet, unit: &u8) -> Self::StateSet {
        let moved = self.move_on_byte(states, *unit);
        self.epsilon_closure(&moved)
    }

    fn advance(&self, states: &Self::StateSet) -> Self::StateSet {
        let mut moved = StateSet::new();
        for state in states {
            for transition in self.transitions_from(state) {
                if transition.label.consumes_input() {
                    moved.insert(transition.to);
                }
            }
        }
        self.epsilon_closure(&moved)
    }

    fn is_accepting(&self, states: &Self::StateSet) -> bool {
        states.iter().any(|state| self.is_final(state))
    }

    fn state_count(&self) -> usize {
        self.num_states()
    }
}
