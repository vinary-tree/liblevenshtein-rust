//! Persistent Adaptive Radix Trie (PART) Dictionary
//!
//! This module implements a disk-based dictionary using a hybrid of:
//! - **Adaptive Radix Tree (ART)**: For optimal trie traversal with adaptive node sizes
//! - **B-trie Buckets**: For efficient leaf storage with multiple strings per disk page
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    PersistentARTrie<V>                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Index Layer (ART)                                           │
//! │  - Node4: 1-4 children (linear scan)                        │
//! │  - Node16: 5-16 children (SIMD accelerated)                 │
//! │  - Node48: 17-48 children (index array)                     │
//! │  - Node256: 49-256 children (direct array)                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Leaf Layer (B-trie Buckets)                                 │
//! │  - StringBucket: 8KB pages with multiple strings            │
//! │  - Binary search within buckets                              │
//! │  - B-trie style splits when full                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Storage Layer                                               │
//! │  - BufferManager: LRU cache with Clock eviction             │
//! │  - DiskManager: Memory-mapped 256KB blocks                  │
//! │  - WAL: Write-ahead logging for crash recovery              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **Pointer Swizzling**: Lazy loading - nodes start as disk references,
//!   swizzled to memory pointers on access
//! - **Path Compression**: Up to 12 bytes of prefix stored inline
//! - **Adaptive Node Sizes**: Optimal memory/performance trade-off
//! - **SIMD Acceleration**: SSE4.1 for Node16 key lookup
//! - **Crash Recovery**: WAL with redo-only recovery
//!
//! # Usage
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
//!
//! // Create a new persistent dictionary
//! let dict = PersistentARTrie::create("words.part")?;
//!
//! // Insert terms
//! dict.insert("hello", ())?;
//! dict.insert("world", ())?;
//!
//! // Query with Levenshtein automaton
//! let transducer = Transducer::new(&dict, Algorithm::Standard);
//! for result in transducer.query("helo", 1) {
//!     println!("{}: distance {}", result.term, result.distance);
//! }
//! ```
//!
//! # Feature Flag
//!
//! Enable with the `persistent-artrie` feature:
//!
//! ```toml
//! [dependencies]
//! liblevenshtein = { version = "0.8", features = ["persistent-artrie"] }
//! ```
//!
//! # References
//!
//! - [B-tries for disk-based string management](https://link.springer.com/article/10.1007/s00778-008-0094-1)
//!   (Askitis & Zobel, VLDBJ 2009)
//! - [The Adaptive Radix Tree](https://db.in.tum.de/~leis/papers/ART.pdf)
//!   (Leis et al., ICDE 2013)
//! - [Persistent Storage of ART in DuckDB](https://duckdb.org/2022/07/27/art-storage)
//!   (DuckDB, 2022)

// Core modules (storage foundation)
pub mod error;
pub mod swizzled_ptr;

#[cfg(feature = "persistent-artrie")]
pub mod disk_manager;

#[cfg(feature = "persistent-artrie")]
pub mod buffer_manager;

// ART node types
pub mod nodes;

// Path compression operations
pub mod path_compression;

// Node serialization
pub mod serialization;

// B-trie string buckets
pub mod bucket;

// Bucket ↔ ART transitions
pub mod transitions;

// Dictionary node implementation
pub mod node_impl;

// Dictionary trait implementation
pub mod dict_impl;

// Zipper implementation
pub mod zipper;

// Write-ahead log for crash recovery
#[cfg(feature = "persistent-artrie")]
pub mod wal;

// Crash recovery
#[cfg(feature = "persistent-artrie")]
pub mod recovery;

// Prefetching for DFS traversal
#[cfg(feature = "persistent-artrie")]
pub mod prefetch;

// Concurrency controls - optimistic lock coupling
#[cfg(feature = "persistent-artrie")]
pub mod concurrency;

// Re-exports for convenience
pub use error::{PersistentARTrieError, Result, SwizzleError};
pub use path_compression::{PrefixMatchResult, SplitPrefix};
pub use swizzled_ptr::{DiskLocation, NodeType, SwizzledPtr};

// Bucket types
pub use bucket::{BucketError, BucketHeader, SplitByByteResult, SplitResult, StringBucket, StringEntry};

// Transition types
pub use transitions::{
    art_node_to_bucket, bucket_to_art_node, should_convert_bucket_to_art,
    should_merge_art_to_bucket, ArtToBucketResult, BucketToArtResult, ChildNode, TransitionError,
};

// Node types
pub use node_impl::PersistentARTrieNode;

// Dictionary types
pub use dict_impl::{PersistentARTrie, TermIterator, TermValueIterator};

// Zipper types
pub use zipper::PersistentARTrieZipper;

#[cfg(feature = "persistent-artrie")]
pub use buffer_manager::{BufferManager, BufferPoolStats, PageReadGuard, PageWriteGuard};
#[cfg(feature = "persistent-artrie")]
pub use disk_manager::{DiskManager, FileHeader, BLOCK_SIZE, MAX_BLOCK_COUNT};

// WAL types
#[cfg(feature = "persistent-artrie")]
pub use wal::{GroupCommit, Lsn, WalHeader, WalReader, WalRecord, WalRecordType, WalWriter};

// Recovery types
#[cfg(feature = "persistent-artrie")]
pub use recovery::{
    IncrementalRecovery, RecoveredOperation, RecoveredState, RecoveryError, RecoveryManager,
    RecoveryStats,
};

// Prefetch types
#[cfg(feature = "persistent-artrie")]
pub use prefetch::{
    AdaptivePrefetcher, PrefetchHint, PrefetchRequest, PrefetchStats, PrefetchStatsSnapshot,
    PrefetchStrategy, Prefetcher,
};

// Concurrency types
#[cfg(feature = "persistent-artrie")]
pub use concurrency::{
    EpochGuard, EpochManager, LockCoupling, OptimisticCell, OptimisticReadGuard, OptimisticVersion,
    RetryStats, WriteGuard,
};

/// Maximum key length supported (64KB - 1)
pub const MAX_KEY_LENGTH: usize = 65535;

/// Maximum value size supported (limited by bucket size)
pub const MAX_VALUE_SIZE: usize = 8192;

/// Default buffer pool size (256 pages = 64MB)
pub const DEFAULT_BUFFER_POOL_SIZE: usize = 256;

/// B-trie bucket page size (8KB)
pub const BUCKET_PAGE_SIZE: usize = 8192;

/// Maximum entries per bucket before split
pub const MAX_BUCKET_ENTRIES: usize = 256;

/// Path compression prefix maximum length
pub const MAX_PREFIX_LENGTH: usize = 12;
