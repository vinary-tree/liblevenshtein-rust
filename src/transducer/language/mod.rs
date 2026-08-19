//! Generic finite-language automata and edit-distance products.
//!
//! A [`LanguageAutomaton`] recognizes a regular language while
//! [`LanguageProduct`] computes standard Levenshtein distance from a unit
//! sequence to that language. The trait is deliberately independent of the
//! phonetic feature: callers can use [`SmallDfa`] over bytes, Unicode scalar
//! values, token IDs, or another equality-comparable unit.

mod bracket;
mod dfa;
mod dyck;
mod product;
mod query;

pub use bracket::{balance_lower_bound, balanced_depth_dfa, BracketError, BRACKET_DFA_MAX_STATES};
pub use dfa::{SmallDfa, SmallDfaError, SmallDfaStateSet, SMALL_DFA_MAX_STATES};
pub use dyck::{
    is_dyck_word, DyckCorrection, DyckCorrectionError, DyckCorrector, DyckEdit,
    DYCK_CORRECTION_MAX_WORK,
};
pub use product::{Frontier, LanguageProduct};
#[cfg(any(feature = "bindings-phonetic", feature = "phonetic-rules"))]
pub(crate) use query::MappedLanguageQueryIterator;
pub use query::{LanguageMatch, LanguageQueryIterator, LanguageQueryStats};

/// Default state-count ceiling for automata compiled from untrusted input.
///
/// Generic callers may deliberately use a larger automaton through
/// [`LanguageProduct::new`]. Convenience compilers such as `query_regex`
/// enforce this ceiling before beginning a product traversal.
pub const LANGUAGE_PRODUCT_MAX_STATES: usize = 4_096;

use std::fmt::Debug;

/// The finite-state operations required by a Levenshtein language product.
///
/// `StateSet` represents a union of language states. Implementations must make
/// `union_into` set union and `subtract` set difference. `step` consumes one
/// matching symbol; `advance` consumes any one language symbol and is used for
/// substitution and deletion.
pub trait LanguageAutomaton<U>: Clone {
    /// A canonical set of active language states.
    type StateSet: Clone + Debug;

    /// Empty state set.
    fn empty(&self) -> Self::StateSet;

    /// Initial state set, including any zero-width closure.
    fn initial(&self) -> Self::StateSet;

    /// Whether `states` contains no active language state.
    fn is_empty(&self, states: &Self::StateSet) -> bool;

    /// Union `source` into `target`.
    fn union_into(&self, target: &mut Self::StateSet, source: &Self::StateSet);

    /// Remove from `target` every state present in `covered`.
    fn subtract(&self, target: &mut Self::StateSet, covered: &Self::StateSet);

    /// Consume `unit` on matching transitions, including zero-width closure.
    fn step(&self, states: &Self::StateSet, unit: &U) -> Self::StateSet;

    /// Consume any one language symbol, including zero-width closure.
    fn advance(&self, states: &Self::StateSet) -> Self::StateSet;

    /// Whether at least one active state accepts the empty continuation.
    fn is_accepting(&self, states: &Self::StateSet) -> bool;

    /// Number of states in the language automaton.
    ///
    /// This is used for explicit resource-policy checks; the product never
    /// attempts hidden subset construction.
    fn state_count(&self) -> usize;
}
