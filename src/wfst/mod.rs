//! WFST integration with lling-llang.
//!
//! This module exposes liblevenshtein's Levenshtein automata as lling-llang
//! `Wfst` implementers, enabling participation in WFST composition pipelines.
//!
//! # Overview
//!
//! The integration provides:
//!
//! - [`LevenshteinWfst`]: A lazy WFST wrapper that presents the Levenshtein
//!   transducer × dictionary product as a `Wfst<char, TropicalWeight>`.
//!   States encode (dictionary_node, automaton_state) pairs.
//!
//! - [`LevenshteinStateSource`]: A [`StateSource`] implementation for lazy
//!   state computation, enabling efficient composition without materializing
//!   the full product state space.
//!
//! - [`DictionaryBackend`]: An adapter that implements lling-llang's
//!   [`LatticeBackend`] trait using a liblevenshtein dictionary.
//!
//! # Architecture
//!
//! The Levenshtein transducer produces a weighted transducer where:
//! - **Input**: Query characters (the misspelled word)
//! - **Output**: Dictionary characters (corrections)
//! - **Weight**: Edit distance as `TropicalWeight` (min-plus semiring)
//!
//! State encoding uses a composite ID: `state_id = dict_node * automaton_size + automaton_state`
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::wfst::{LevenshteinWfst, DictionaryBackend};
//! use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
//! use lling_llang::composition::compose;
//! use lling_llang::prelude::*;
//!
//! // Build a dictionary
//! let dict = DynamicDawgChar::from_terms(vec!["hello", "help", "world"]);
//!
//! // Create a Levenshtein WFST for the query "helo" with max distance 2
//! let lev_wfst = LevenshteinWfst::new(&dict, "helo", 2);
//!
//! // Compose with a language model WFST
//! let composed = compose(lev_wfst, language_model);
//!
//! // Find best corrections using composed transducer
//! for path in composed.accepting_paths() {
//!     println!("{:?} (weight: {:?})", path.labels(), path.weight());
//! }
//! ```
//!
//! # Feature Gate
//!
//! This module requires the `wfst` feature to be enabled:
//!
//! ```toml
//! [dependencies]
//! liblevenshtein = { version = "0.8", features = ["wfst"] }
//! ```

mod backend;
mod state_source;
mod universal_state_source;
mod universal_wrapper;
mod wrapper;

// Phonetic WFST modules
mod composed_phonetic;
#[cfg(feature = "phonetic-rules")]
mod phonetic_nfa_wfst;
mod phonetic_rewrite_wfst;
#[cfg(feature = "phonetic-rules")]
mod phonetic_state_source;
#[cfg(feature = "phonetic-rules")]
mod phonetic_wfst;

// Time Series MSM WFST module

// Generalized and WallBreaker WFST modules
mod generalized_wfst;
mod wallbreaker_wfst;

pub use backend::DictionaryBackend;
pub use state_source::LevenshteinStateSource;
pub use universal_state_source::{UniversalLevenshteinStateSource, UniversalStateRegistry};
pub use universal_wrapper::{BoundUniversalWfst, UniversalLevenshteinWfst};
pub use wrapper::LevenshteinWfst;

// Phonetic WFST exports
pub use composed_phonetic::{PhoneticMatch, PhoneticPipelineBuilder, PhoneticPipelineConfig};
#[cfg(feature = "phonetic-rules")]
pub use phonetic_nfa_wfst::PhoneticNfaWfst;
pub use phonetic_rewrite_wfst::{CommonPhoneticRules, RewriteRule, RewriteWfst};
#[cfg(feature = "phonetic-rules")]
pub use phonetic_state_source::PhoneticStateSource;
#[cfg(feature = "phonetic-rules")]
pub use phonetic_wfst::{PhoneticWfst, PhoneticWfstBuilder};

// Generalized and WallBreaker WFST exports
pub use generalized_wfst::{GeneralizedWfst, GeneralizedWfstBuilder};
pub use wallbreaker_wfst::{WallBreakerWfst, WallBreakerWfstBuilder};

// Re-export commonly used lling-llang types for convenience
pub use lling_llang::prelude::{
    LazyState, LazyWfst, LazyWfstWrapper, Semiring, StateId, StateSource, TropicalWeight, VocabId,
    WeightedTransition, Wfst,
};

/// Composite state ID encoding for the product automaton.
///
/// States in the Levenshtein WFST encode (dictionary_node, automaton_state) pairs
/// as a single `StateId` using the formula:
///
/// ```text
/// state_id = dict_node_id * max_automaton_states + automaton_state_id
/// ```
///
/// This module provides utilities for encoding and decoding these composite states.
pub mod state_encoding {
    use lling_llang::wfst::StateId;

    /// Encode a (dictionary_node, automaton_state) pair into a single StateId.
    #[inline]
    pub fn encode(dict_node: u32, automaton_state: u32, max_automaton_states: u32) -> StateId {
        dict_node * max_automaton_states + automaton_state
    }

    /// Decode a StateId back into (dictionary_node, automaton_state).
    #[inline]
    pub fn decode(state_id: StateId, max_automaton_states: u32) -> (u32, u32) {
        let automaton_state = state_id % max_automaton_states;
        let dict_node = state_id / max_automaton_states;
        (dict_node, automaton_state)
    }

    /// Estimate the maximum automaton states for a given query length and max distance.
    ///
    /// The Levenshtein automaton for a query of length n with max distance k
    /// has at most O((n+1) * (2k+1)) states (bounded by the position-distance lattice).
    #[inline]
    pub fn estimate_automaton_states(query_len: usize, max_distance: usize) -> u32 {
        let positions = query_len + 1;
        let distances = 2 * max_distance + 1;
        (positions * distances) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_encoding_roundtrip() {
        let max_states = 100u32;

        for dict_node in 0..10 {
            for auto_state in 0..10 {
                let encoded = state_encoding::encode(dict_node, auto_state, max_states);
                let (dec_dict, dec_auto) = state_encoding::decode(encoded, max_states);
                assert_eq!(dec_dict, dict_node);
                assert_eq!(dec_auto, auto_state);
            }
        }
    }

    #[test]
    fn test_estimate_automaton_states() {
        // Query "hello" (len 5) with max distance 2
        // Positions: 0..5 (6 positions)
        // Distances: -2..2 (5 distances per position conceptually, but bounded)
        let estimate = state_encoding::estimate_automaton_states(5, 2);
        assert!(estimate > 0);
        assert!(estimate <= 100); // Reasonable upper bound
    }
}
