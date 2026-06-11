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
pub mod filter;
/// Cross-platform synchronization primitives (parking_lot on native, std::sync on WASM)
pub mod sync_compat;
pub mod transducer;

/// WallBreaker algorithm for approximate string matching with large error bounds.
///
/// This module implements the WallBreaker algorithm from:
/// > "WallBreaker - overcoming the wall effect in similarity search"
/// > (Gerdjikov, Mihov, Mitankin, Schulz - EDBT/ICDT 2013)
///
/// WallBreaker overcomes the "wall effect" in traditional Levenshtein automata
/// by using the pigeonhole principle and SCDAWG substring search. For large
/// error bounds, it achieves significant speedups (5600× for 100-char patterns
/// with 16 errors in a 750K dictionary).
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::scdawg::Scdawg;
/// use liblevenshtein::wallbreaker::WallBreaker;
///
/// let dict = Scdawg::<()>::from_terms(vec!["hello", "world", "help"]);
/// let wb = WallBreaker::new(&dict, 1);
///
/// for result in wb.query("helo") {
///     println!("{} (distance {})", result.term, result.distance);
/// }
/// ```
pub mod wallbreaker;

/// Time series distance metrics and indexing
///
/// This module provides implementations for time series similarity measures,
/// particularly the Move-Split-Merge (MSM) metric. It includes:
/// - Direct O(mn) dynamic programming implementation
/// - Space-optimized O(min(m,n)) variant
/// - Full DP matrix output for debugging/alignment
///
/// # Example
///
/// ```rust
/// use liblevenshtein::time_series::MsmConfig;
///
/// let config = MsmConfig::new(1.0);  // c = 1.0
/// let x = vec![1.0, 2.0, 3.0, 2.0];
/// let y = vec![1.0, 2.5, 2.0];
///
/// let distance = config.distance(&x, &y);
/// println!("MSM distance: {}", distance);
/// ```
///
/// # References
///
/// Stefan, Alexandra, et al. "The move-split-merge metric for time series."
/// IEEE transactions on Knowledge and Data Engineering 25.6 (2012): 1425-1438.
pub mod time_series;

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
///
/// This module provides:
/// - Composable eviction wrappers (LRU, LFU, TTL, etc.)
/// - `FuzzyMultiMap` for fuzzy queries returning aggregated collections
///
/// Works with any `MappedDictionary` implementation including `DynamicDawgChar`,
/// `PathMapDictionary`, `SuffixAutomaton`, etc.
#[cfg(any(feature = "pathmap-backend", feature = "phonetic-rules"))]
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
    pub use crate::transducer::{
        Algorithm, Candidate, QueryBuilder, Transducer, TransducerBuilder,
    };
    pub use libdictenstein::double_array_trie::DoubleArrayTrie;
    pub use libdictenstein::dynamic_dawg::DynamicDawg;
    pub use libdictenstein::factory::{DictionaryBackend, DictionaryContainer, DictionaryFactory};
    #[cfg(feature = "pathmap-backend")]
    pub use libdictenstein::pathmap::PathMapDictionary;
    #[cfg(feature = "persistent-artrie")]
    pub use libdictenstein::persistent_artrie::{PersistentARTrie, PersistentARTrieZipper};
    pub use libdictenstein::suffix_automaton::SuffixAutomaton;
    pub use libdictenstein::{Dictionary, DictionaryNode, SyncStrategy};

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

    // Eviction wrappers available when cache module is public
    #[cfg(feature = "pathmap-backend")]
    pub use crate::cache::eviction;

    // PhoneticNormalizedDictionary only requires phonetic-rules (uses DynamicDawgChar internally)
    // This module remains in liblevenshtein since it depends on phonetic NFAs
    #[cfg(feature = "phonetic-rules")]
    pub use crate::dictionary::phonetic_normalized::{
        PhoneticNormalizedCandidate, PhoneticNormalizedDictionary,
        PhoneticNormalizedDictionaryChar, PhoneticNormalizedNode, PhoneticNormalizedZipper,
        RegexQueryError,
    };

    // WallBreaker for large error bounds
    pub use crate::wallbreaker::{
        PatternPiece, PatternSplitter, WallBreaker, WallBreakerQuery, WallBreakerResult,
    };
    pub use libdictenstein::scdawg::Scdawg;
    pub use libdictenstein::scdawg_char::ScdawgChar;
    pub use libdictenstein::substring::{
        BidirectionalDictionaryNode, SubstringDictionary, SubstringMatch,
    };
}
