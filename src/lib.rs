//! # liblevenshtein
//!
//! Fast approximate string matching using Levenshtein automata.
//!
//! This library provides efficient fuzzy string matching against dictionaries
//! using Universal Levenshtein Automata, based on the algorithm described in:
//!
//! > Schulz, Klaus U., and Stoyan Mihov. "Fast string correction with
//! > Levenshtein automata." International Journal on Document Analysis and
//! > Recognition 5.1 (2002): 67-85.
//!
//! ## Example
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//!
//! let terms = vec!["test", "testing", "tested"];
//! let dict = PathMapDictionary::from_terms(terms);
//! let transducer = Transducer::new(dict, Algorithm::Standard);
//!
//! for term in transducer.query("tset", 2) {
//!     println!("Match: {}", term);
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod commands;
#[cfg(feature = "pathmap-backend")]
pub mod contextual;
pub mod dictionary;
pub mod distance;
/// Cross-platform synchronization primitives (parking_lot on native, std::sync on WASM)
pub mod sync_compat;
pub mod transducer;

/// Phonetic rewrite rules for approximate string matching
///
/// This module provides verified phonetic transformation rules with formal
/// correctness guarantees from Coq/Rocq proofs. It implements Zompist's
/// English spelling rules for normalizing text before fuzzy matching.
#[cfg(feature = "phonetic-rules")]
pub mod phonetic;

#[cfg(feature = "serialization")]
pub mod serialization;

/// Fuzzy cache with composable eviction strategies
#[cfg(feature = "pathmap-backend")]
pub mod cache;

/// Interactive REPL for exploring Levenshtein dictionaries
#[cfg(all(feature = "cli", not(target_arch = "wasm32")))]
pub mod repl;

/// CLI interface and utilities
#[cfg(all(feature = "cli", not(target_arch = "wasm32")))]
pub mod cli;

/// Grep support for compressed, archived, and document files
///
/// This module provides streaming decompression and archive support
/// for searching through .gz, .zst, .xz, .bz2, .tar, and .zip files.
/// Also provides document extraction support for PDF, DOCX, XLSX, EPUB, and ODT files.
#[cfg(all(
    not(target_arch = "wasm32"),
    any(
        feature = "grep-compression",
        feature = "grep-archives",
        feature = "grep-pdf",
        feature = "grep-docx",
        feature = "grep-xlsx",
        feature = "grep-epub",
        feature = "grep-odt"
    )
))]
pub mod grep;

/// WebAssembly bindings for browser and Node.js via wasm-bindgen
///
/// This module provides JavaScript-friendly APIs for all core functionality:
/// - Distance functions (Levenshtein, Damerau-Levenshtein)
/// - Dictionary backends (DoubleArrayTrie, DynamicDawg)
/// - Levenshtein transducers for fuzzy search
/// - Phonetic rules (with `wasm-phonetic` feature)
#[cfg(feature = "wasm")]
pub mod wasm;

/// C-compatible FFI for WASI runtimes and native language bindings
///
/// This module provides raw C-compatible functions (`extern "C"`) for:
/// - Distance calculations
/// - Dictionary construction and querying
/// - Transducer operations
/// - Memory management (string/array allocation and freeing)
///
/// Suitable for WASI runtimes (Wasmtime, WasmEdge) and native FFI from
/// other languages (Python, Ruby, Go, etc.)
#[cfg(feature = "ffi")]
pub mod ffi;

/// Test corpus utilities
///
/// This module provides parsers and generators for standard spelling
/// correction test corpora. It requires the `rand` feature.
#[cfg(feature = "rand")]
#[doc(hidden)]
pub mod corpus;

/// Common imports for convenient usage
pub mod prelude {
    pub use crate::dictionary::dawg::DawgDictionary;

    /// **DEPRECATED**: Use `DynamicDawg` instead - 11× faster with full feature support.
    ///
    /// OptimizedDawg was an experimental DAWG implementation with arena-based edge storage,
    /// but benchmarks show DynamicDawg is significantly faster while providing more features
    /// (MappedDictionary, ValuedDictZipper, mutability).
    #[deprecated(
        since = "0.7.0",
        note = "Use DynamicDawg instead - 11× faster construction with full feature support"
    )]
    pub use crate::dictionary::dawg_optimized::OptimizedDawg;

    pub use crate::dictionary::double_array_trie::DoubleArrayTrie;
    pub use crate::dictionary::dynamic_dawg::DynamicDawg;
    pub use crate::dictionary::factory::{
        DictionaryBackend, DictionaryContainer, DictionaryFactory,
    };
    #[cfg(feature = "pathmap-backend")]
    pub use crate::dictionary::pathmap::PathMapDictionary;
    pub use crate::dictionary::suffix_automaton::SuffixAutomaton;
    pub use crate::dictionary::{Dictionary, DictionaryNode, SyncStrategy};
    pub use crate::transducer::{
        Algorithm, Candidate, QueryBuilder, Transducer, TransducerBuilder,
    };

    #[cfg(feature = "serialization")]
    pub use crate::serialization::{
        BincodeSerializer, DictionaryFromTerms, DictionarySerializer, JsonSerializer,
        PlainTextSerializer,
    };

    #[cfg(feature = "protobuf")]
    pub use crate::serialization::{
        DatProtobufSerializer, OptimizedProtobufSerializer, ProtobufSerializer,
        SuffixAutomatonProtobufSerializer,
    };

    #[cfg(feature = "compression")]
    pub use crate::serialization::GzipSerializer;

    #[cfg(feature = "pathmap-backend")]
    pub use crate::cache::eviction;

    #[cfg(all(feature = "pathmap-backend", feature = "phonetic-rules"))]
    pub use crate::dictionary::phonetic_normalized::{
        PhoneticNormalizedCandidate, PhoneticNormalizedDictionary, PhoneticNormalizedDictionaryChar,
        PhoneticNormalizedNode, PhoneticNormalizedZipper, RegexQueryError,
    };
}
