//! Universal Levenshtein Automata
//!
//! This module implements Universal Levenshtein Automata as described in
//! Petar Mitankin's 2005 thesis "Universal Levenshtein Automata - Building and Properties".
//!
//! # Overview
//!
//! Universal Levenshtein Automata (A^{∀,χ}_n) are parameter-free, deterministic finite automata
//! that recognize the Levenshtein neighborhood L^χ_{Lev}(n, w) for **any word w** without
//! modification. Unlike traditional Levenshtein automata that must be built for each specific
//! query word, universal automata can be precomputed once and reused for all words.
//!
//! # Key Concepts
//!
//! - **Universal Positions**: Positions parameterized by I (non-final) or M (final) instead of
//!   concrete word indices
//! - **Subsumption Relation**: ≤^χ_s partial order for state minimization
//! - **Bit Vector Encoding**: h_n(w, x) encodes word pairs as bit vector sequences
//! - **Three Variants**: ε (standard), t (with transposition), ms (with merge/split)
//!
//! # Documentation References
//!
//! Complete theory documentation is available in `/docs/research/universal-levenshtein/`:
//! - PAPER_SUMMARY.md - Complete thesis analysis
//! - ALGORITHMS.md - Pseudocode and construction algorithms
//! - THEORETICAL_FOUNDATIONS.md - Proofs and mathematical foundations
//! - IMPLEMENTATION_MAPPING.md - Theory-to-Rust mapping guide
//!
//! # Phase 1 Status
//!
//! Currently implementing:
//! - Core position types (UniversalPosition<V>)
//! - Subsumption relation (≤^χ_s)
//! - State management (UniversalState<V>)
//!
//! # Example (Future API)
//!
//! ```ignore
//! use liblevenshtein::transducer::universal::UniversalAutomaton;
//!
//! // Precompute once for maximum distance n=2
//! let automaton = UniversalAutomaton::<Standard, 2>::build();
//!
//! // Use for any word
//! assert!(automaton.accepts("test", "text"));  // distance 1
//! assert!(automaton.accepts("test", "best"));  // distance 1
//! assert!(!automaton.accepts("test", "hello")); // distance > 2
//! ```

pub mod automaton;
pub mod bit_vector;
pub mod diagonal;
pub mod position;
pub mod state;
pub mod subsumption;

// Re-exports
pub use automaton::UniversalAutomaton;

pub use bit_vector::{characteristic_vector, encode_word_pair, CharacteristicVector};

pub use diagonal::{convert_position, diagonal_crossed, right_most};

pub use position::{
    MergeAndSplit, PositionError, PositionVariant, Standard, Transposition, UniversalPosition,
};

pub use state::UniversalState;
pub use subsumption::subsumes;
