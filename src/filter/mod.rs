//! Pre-filtering for approximate string matching.
//!
//! This module provides fast candidate filtering algorithms that can reject
//! 90%+ of candidates before expensive Levenshtein automata traversal.
//!
//! # Filtering Pipeline
//!
//! The hybrid filtering approach combines multiple stages:
//!
//! 1. **N-gram Index** (fastest, coarsest)
//!    - Indexes terms by n-grams (bigrams/trigrams)
//!    - Rejects candidates with insufficient n-gram overlap
//!    - Time: O(|query| + |candidates|)
//!
//! 2. **Jaro-Winkler** (fast, finer)
//!    - Computes string similarity with prefix bonus
//!    - Rejects candidates below similarity threshold
//!    - Time: O(|query| * |term|) per candidate
//!
//! 3. **Levenshtein Automaton** (expensive, exact)
//!    - Full automaton traversal for remaining candidates
//!    - Produces final matches with exact distances
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::filter::{HybridMatcher, NgramIndex};
//!
//! // Build index from dictionary
//! let mut index = NgramIndex::new(2); // Bigrams
//! for term in ["apple", "application", "banana", "apply"] {
//!     index.insert(term);
//! }
//!
//! // Find candidates for "aple" with max distance 2
//! let candidates = index.find_candidates("aple", 2);
//! // candidates: ["apple", "apply"] (rejected "banana", "application")
//! ```
//!
//! # Performance
//!
//! For a 100K word dictionary with max_distance=2:
//! - N-gram filter: ~1-5ms, rejects 85-95% of candidates
//! - Jaro-Winkler refinement: ~0.5-2ms, rejects 50-80% of remaining
//! - Combined: ~2-7ms total filtering vs ~50-200ms full automaton

pub mod jaro_winkler;
pub mod ngram;

mod hybrid;

pub use hybrid::{HybridMatcher, HybridMatcherBuilder};
pub use jaro_winkler::{
    distance_to_similarity_approx, is_similar, jaro_similarity, jaro_winkler_similarity,
    jaro_winkler_similarity_scaled, similarity_to_distance_approx,
};
pub use ngram::NgramIndex;
