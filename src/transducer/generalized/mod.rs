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
//! specialization for an exact runtime operation grid:
//!
//! - **UniversalAutomaton**: Compile-time operations (Standard, Transposition, MergeAndSplit)
//!   - Zero runtime overhead
//!   - Fixed operation sets
//!   - Perfect for standard Levenshtein variants
//!
//! - **GeneralizedAutomaton**: Runtime operations via `OperationSet`
//!   - Exact decimal weights through [`crate::cost::CostScale`]
//!   - Sparse, resource-bounded alignment graph
//!   - Custom operation sets
//!   - Perfect for phonetic corrections and custom metrics
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
//! use liblevenshtein::transducer::phonetic::phonetic_english_basic;
//! use liblevenshtein::transducer::{OperationSet, OperationSetBuilder};
//!
//! // Standard operations
//! let standard = GeneralizedAutomaton::with_operations(2, OperationSet::standard());
//! assert!(standard.accepts("test", "text"));
//!
//! // Add phonetic rules to the standard operations needed to traverse exact spans.
//! let mut builder = OperationSetBuilder::new().with_standard_ops();
//! for operation in phonetic_english_basic().operations() {
//!     builder = builder.with_operation(operation.clone());
//! }
//! let phonetic = GeneralizedAutomaton::with_operations(1, builder.build());
//! assert!(phonetic.accepts("phone", "fone"));
//! ```
//!
//! # Implementation Status
//!
//! **Current:**
//! - `GeneralizedAutomaton` exactly accumulates standard, transposition,
//!   merge/split, weighted, and restricted phonetic operations.
//! - Consumption counts Unicode scalar values; restrictions use exact UTF-8
//!   slices.
//! - Fallible APIs report invalid costs, overflow, and resource exhaustion.
//! - Generalized positions/states provide a bounded streaming compatibility
//!   API for one-scalar operations and the historical 2-to-2, 2-to-1, and
//!   1-to-2 intermediates. They accumulate weights exactly and report an
//!   unsupported-arity error instead of silently ignoring a rule. The sparse
//!   alignment graph remains the operation-complete API for arbitrary
//!   non-zero consumption pairs.
//!
//! See: `docs/design/generalized-automaton-repair.md`

mod automaton;
pub mod position;
mod state;
mod subsumption;

pub use crate::transducer::universal::bit_vector::CharacteristicVector;
pub use automaton::{
    GeneralizedAutomaton, GeneralizedAutomatonError, MAX_GENERALIZED_ALIGNMENT_STATES,
};
pub use position::{GeneralizedPosition, PositionError};
pub use state::{GeneralizedState, GeneralizedStateError, GeneralizedTransitionInput};
pub use subsumption::subsumes;
