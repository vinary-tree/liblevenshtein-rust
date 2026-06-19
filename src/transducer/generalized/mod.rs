//! Generalized Levenshtein Automata with Runtime-Configurable Operations
//!
//! This module provides `GeneralizedAutomaton`, a Levenshtein automaton that accepts
//! runtime-configurable operations via `OperationSet`. This enables:
//! - Phonetic corrections (ph↔f, ch↔k, etc.)
//! - Custom edit distance metrics
//! - Weighted operations represented by operation costs
//! - Multi-character operations such as transposition, merge, and split
//!
//! # Design Philosophy
//!
//! `GeneralizedAutomaton` complements `UniversalAutomaton` by trading compile-time
//! specialization for runtime flexibility:
//!
//! - **UniversalAutomaton**: Compile-time operations (Standard, Transposition, MergeAndSplit)
//!   - Zero runtime overhead
//!   - Fixed operation sets
//!   - Perfect for standard Levenshtein variants
//!
//! - **GeneralizedAutomaton**: Runtime operations via `OperationSet`
//!   - Small runtime overhead (+10-20%)
//!   - Custom operation sets
//!   - Perfect for phonetic corrections and custom metrics
//!
//! # Example
//!
//! ```ignore
//! use liblevenshtein::transducer::{GeneralizedAutomaton, OperationSet};
//! use liblevenshtein::transducer::phonetic::phonetic_english_basic;
//!
//! // Standard operations
//! let ops = OperationSet::standard();
//! let automaton = GeneralizedAutomaton::with_operations(2, ops);
//!
//! // Phonetic operations
//! let ops = phonetic_english_basic();
//! let automaton = GeneralizedAutomaton::with_operations(2, ops);
//! ```
//!
//! # Implementation Status
//!
//! **Current:**
//! - Generalized positions and states with same-variant subsumption.
//! - Runtime `OperationSet` support for standard, transposition, merge, split,
//!   and restricted phonetic operations.
//! - `accepts()` support for configurable operations over UTF-8 input.
//! - Focused performance work remains ongoing for heavily weighted and
//!   corpus-scale workloads.
//!
//! See: `docs/design/generalized-automaton.md`

mod automaton;
pub mod position;
mod state;
mod subsumption;

pub use crate::transducer::universal::bit_vector::CharacteristicVector;
pub use automaton::GeneralizedAutomaton;
pub use position::{GeneralizedPosition, PositionError};
pub use state::GeneralizedState;
pub use subsumption::subsumes;
