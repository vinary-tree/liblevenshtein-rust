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
//! ```rust
//! use liblevenshtein::prelude::*;
//!
//! let terms = vec!["test", "testing", "tested"];
//! let dict = DoubleArrayTrie::from_terms(terms);
//! let transducer = Transducer::new(dict, Algorithm::Standard);
//!
//! let matches: Vec<_> = transducer.query_terms("tset", 2).collect();
//! assert_eq!(matches, ["test"]);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod causal_perf;

#[doc(hidden)]
pub use causal_perf::{causal_perf_stats, reset_causal_perf_stats, CausalPerfStats};

/// Stable, non-generic streaming primitives used by generated language bindings.
#[cfg(feature = "bindings-core")]
pub mod bindings;
#[cfg(feature = "pathmap-backend")]
pub mod contextual;
/// Ordered cost monoids and exact decimal fixed-point scaling.
pub mod cost;
pub mod dictionary;
pub mod distance;
pub mod filter;
/// Crate-internal home for pure checked/saturating numeric conversions
/// (`f64 → usize`, `usize → i64`, `Duration → u64`) shared across the
/// `transducer`, `time_series`, and `cache` modules.
pub(crate) mod numeric;
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
/// ```rust
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
pub mod cache;

/// WebAssembly bindings for browser and Node.js via wasm-bindgen
///
/// This module provides JavaScript-friendly APIs for all core functionality:
/// - Distance functions (Levenshtein, optimal string alignment)
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
    pub use crate::cost::{BottleneckCost, CostMonoid, CostScale, UnitCost, WeightedCost};
    pub use crate::transducer::{
        Algorithm, Candidate, MatchMode, MatchModeError, PrefixQueryIterator, PrefixQueryMatch,
        PrefixQueryStats, QueryBuilder, Transducer, TransducerBuilder,
    };
    // ----------------------------------------------------------------------
    // Legacy dictionary re-exports (deprecated since 0.9.1).
    //
    // The dictionary data structures live in the `libdictenstein` crate. These
    // re-exports are a backwards-compatibility convenience; depend on
    // `libdictenstein` directly and import each type from its canonical module
    // (named in the deprecation notes below). They will be removed in a future
    // release.
    // ----------------------------------------------------------------------
    #[deprecated(
        since = "0.9.1",
        note = "import from `libdictenstein::double_array_trie` instead; liblevenshtein's dictionary re-exports are deprecated and will be removed"
    )]
    pub use libdictenstein::double_array_trie::{DoubleArrayTrie, DoubleArrayTrieChar};
    #[deprecated(
        since = "0.9.1",
        note = "import from `libdictenstein::dynamic_dawg` instead; liblevenshtein's dictionary re-exports are deprecated and will be removed"
    )]
    pub use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgChar};
    #[deprecated(
        since = "0.9.1",
        note = "import from `libdictenstein::factory` instead; liblevenshtein's dictionary re-exports are deprecated and will be removed"
    )]
    pub use libdictenstein::factory::{DictionaryBackend, DictionaryContainer, DictionaryFactory};
    #[cfg(feature = "pathmap-backend")]
    #[deprecated(
        since = "0.9.1",
        note = "import from `libdictenstein::pathmap` instead; liblevenshtein's dictionary re-exports are deprecated and will be removed"
    )]
    pub use libdictenstein::pathmap::PathMapDictionary;
    #[cfg(feature = "persistent-artrie")]
    #[deprecated(
        since = "0.9.1",
        note = "import from `libdictenstein::persistent_artrie` instead; liblevenshtein's dictionary re-exports are deprecated and will be removed"
    )]
    pub use libdictenstein::persistent_artrie::{PersistentARTrie, PersistentARTrieZipper};
    #[deprecated(
        since = "0.9.1",
        note = "import from `libdictenstein::suffix_automaton` instead; liblevenshtein's dictionary re-exports are deprecated and will be removed"
    )]
    pub use libdictenstein::suffix_automaton::{SuffixAutomaton, SuffixAutomatonChar};
    #[deprecated(
        since = "0.9.1",
        note = "import from `libdictenstein` instead; liblevenshtein's dictionary re-exports are deprecated and will be removed"
    )]
    pub use libdictenstein::{Dictionary, DictionaryNode, SyncStrategy};

    #[cfg(feature = "serialization")]
    pub use crate::serialization::{BincodeSerializer, DictionaryFromTerms, DictionarySerializer};

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
    #[deprecated(
        since = "0.9.1",
        note = "import from `libdictenstein::scdawg` instead; liblevenshtein's dictionary re-exports are deprecated and will be removed"
    )]
    pub use libdictenstein::scdawg::{char::ScdawgChar, Scdawg};
    #[deprecated(
        since = "0.9.1",
        note = "import from `libdictenstein::substring` instead; liblevenshtein's dictionary re-exports are deprecated and will be removed"
    )]
    pub use libdictenstein::substring::{
        BidirectionalDictionaryNode, SubstringDictionary, SubstringMatch,
    };
}
