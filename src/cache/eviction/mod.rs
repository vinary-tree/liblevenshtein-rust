//! Eviction wrapper infrastructure for fuzzy dictionaries.
//!
//! This module provides composable eviction strategy wrappers that implement
//! the `MappedDictionary` trait, allowing them to be stacked and composed.
//!
//! # Overview
//!
//! The eviction wrappers use a decorator pattern to add caching behavior to any
//! dictionary implementation. Each wrapper maintains separate metadata (access times,
//! hit counts, sizes, etc.) in thread-safe storage (`Arc<RwLock<HashMap>>`).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Eviction Wrapper (e.g., Lru<D>)                 │
//! │  ┌────────────────────────────────────────────────────────┐ │
//! │  │  inner: D                                               │ │
//! │  │  metadata: Arc<RwLock<HashMap<String, Metadata>>>      │ │
//! │  └────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//!                            │
//!                            ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │           Inner Dictionary (PathMapDictionary, etc.)         │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Available Wrappers
//!
//! ## Core Wrappers
//!
//! - **`Noop`**: Zero-cost passthrough wrapper (identity function)
//! - **`LazyInit`**: Deferred dictionary initialization (Default, Fn, or Full closure)
//!
//! ## Time-Based Eviction
//!
//! - **`TTL`**: Time-to-live filtering - expires entries after a duration
//! - **`Age`**: FIFO (First In, First Out) - evicts oldest entries
//! - **`LRU`**: Least Recently Used - evicts entries not accessed recently
//!
//! ## Frequency-Based Eviction
//!
//! - **`LFU`**: Least Frequently Used - evicts entries with lowest access count
//!
//! ## Cost-Based Eviction
//!
//! - **`CostAware`**: Balances age, size, and hit count - formula: `(age * size) / (hits + 1)`
//! - **`MemoryPressure`**: Memory-aware eviction - formula: `size / (hit_rate + 0.1)`
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::Lru;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("hello", 1),
//!     ("world", 2),
//! ]);
//!
//! let lru = Lru::new(dict);
//! assert_eq!(lru.get_value("hello"), Some(1));
//!
//! // Find least recently used entry
//! let lru_term = lru.find_lru(&["hello", "world"]);
//! ```
//!
//! ## Composing Wrappers
//!
//! Wrappers can be composed to combine multiple eviction strategies:
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::{Lru, Ttl};
//! use std::time::Duration;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("foo", 42),
//!     ("bar", 99),
//! ]);
//!
//! // Compose TTL + LRU: entries expire after 5 minutes AND track recency
//! let ttl = Ttl::new(dict, Duration::from_secs(300));
//! let lru = Lru::new(ttl);
//!
//! assert_eq!(lru.get_value("foo"), Some(42));
//! ```
//!
//! ## Memory-Aware Caching
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::MemoryPressure;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("large_data", vec![1, 2, 3, 4, 5]),
//!     ("small_data", vec![1]),
//! ]);
//!
//! let memory = MemoryPressure::new(dict);
//!
//! // Access both entries
//! memory.get_value("large_data");
//! memory.get_value("small_data");
//!
//! // Find entry with highest memory pressure
//! let high_pressure = memory.find_highest_pressure(&["large_data", "small_data"]);
//! ```
//!
//! ## Cost-Based Eviction
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::CostAware;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("old_rarely_used", 1),
//!     ("new_frequently_used", 2),
//! ]);
//!
//! let cost_aware = CostAware::new(dict);
//!
//! // Access patterns affect cost scores
//! cost_aware.get_value("new_frequently_used");
//! cost_aware.get_value("new_frequently_used");
//! cost_aware.get_value("new_frequently_used");
//!
//! // Old, rarely used entries have higher cost scores
//! let highest_cost = cost_aware.find_highest_cost(&[
//!     "old_rarely_used",
//!     "new_frequently_used"
//! ]);
//! ```
//!
//! ## Lazy Initialization
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::LazyInit;
//!
//! // Create dictionary with lazy initializer for missing values
//! let dict: PathMapDictionary<i32> = PathMapDictionary::from_terms_with_values([
//!     ("lazy_value", 42),
//! ]);
//!
//! // Wrap dictionary with lazy initializer
//! let mut lazy = LazyInit::new(dict, || {
//!     0i32  // Default value for missing terms
//! });
//!
//! // Access existing value
//! assert_eq!(lazy.get_value("lazy_value"), Some(42));
//! ```
//!
//! # Use Cases
//!
//! - **Code Completion**: Use `Lru` or `Lfu` to keep frequently accessed identifiers
//! - **Documentation Search**: Use `CostAware` to balance relevance and recency
//! - **AI Code Chat**: Use `Ttl + MemoryPressure` to manage LLM response caching
//! - **Error Solutions**: Use `Lfu` to keep common error solutions cached
//! - **Large Value Caching**: Use `MemoryPressure` for embeddings, ASTs, etc.
//!
//! # Thread Safety
//!
//! All wrappers are thread-safe. Metadata is protected by `Arc<RwLock<>>`, allowing:
//! - Multiple concurrent readers (shared read locks)
//! - Exclusive writer access for updates (write locks)
//!
//! # Performance
//!
//! - **Zero-cost abstraction**: `Noop` wrapper has no runtime overhead
//! - **Metadata overhead**: Each wrapper adds ~100 bytes per entry for metadata
//! - **Lock contention**: Write-heavy workloads may experience contention on metadata locks
//!
//! # Design Patterns
//!
//! ## Wrapper Stacking Order
//!
//! Stack wrappers from most specific to least specific:
//!
//! ```text
//! Lru<Ttl<MemoryPressure<PathMapDictionary>>>
//!  ^   ^        ^              ^
//!  |   |        |              └─ Base dictionary
//!  |   |        └─ Memory tracking
//!  |   └─ Time-based filtering
//!  └─ Recency tracking
//! ```

mod lazy_init;
mod noop;

pub use lazy_init::{LazyInit, LazyInitDefault, LazyInitFn};
pub use noop::Noop;

// Placeholder modules for other eviction strategies
// These will be implemented in subsequent steps
pub mod age;
pub mod cost_aware;
pub mod lfu;
pub mod lru;
pub mod lru_optimized;
pub mod memory_pressure;
pub mod ttl;

pub use age::Age;
pub use cost_aware::CostAware;
pub use lfu::Lfu;
pub use lru::Lru;
pub use lru_optimized::LruOptimized;
pub use memory_pressure::MemoryPressure;
pub use ttl::Ttl;
