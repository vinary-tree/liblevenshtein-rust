//! Dictionary abstractions for pluggable backends.
//!
//! This module provides traits that abstract over different dictionary
//! implementations (tries, DAWGs, etc.) for use with Levenshtein automata.
//!
//! # Choosing a Dictionary Backend
//!
//! The library provides multiple dictionary backends optimized for different use cases.
//! Here's a quick decision guide:
//!
//! ## Quick Start (Pick One)
//!
//! - **Just getting started?** Use [`DoubleArrayTrie`](double_array_trie::DoubleArrayTrie)
//!   - Best overall performance (3x faster queries, 30x faster contains checks)
//!   - Lowest memory footprint (~8 bytes/state)
//!   - Supports runtime insertions (append-only)
//!   - Default choice for most applications
//!
//! - **Working with Unicode text?** Use [`DoubleArrayTrieChar`](double_array_trie_char::DoubleArrayTrieChar)
//!   - Correct character-level Levenshtein distances
//!   - Handles accented characters, CJK, emoji correctly
//!   - Supports runtime insertions (append-only)
//!   - ~5% performance overhead, 4x memory for edge labels
//!
//! - **Need to remove terms?** Use [`DynamicDawg`](dynamic_dawg::DynamicDawg)
//!   - Thread-safe insert AND remove operations
//!   - Good fuzzy matching performance for fully dynamic use
//!   - Active queries see updates immediately
//!
//! - **Need Unicode + remove terms?** Use [`DynamicDawgChar`](dynamic_dawg_char::DynamicDawgChar)
//!   - Correct character-level Levenshtein distances for Unicode
//!   - Thread-safe insert AND remove operations
//!   - ~5% performance overhead vs byte-level, 4x memory for edges
//!
//! - **Need u64 sequences or f64 time series?** Use [`DynamicDawgU64`](dynamic_dawg_u64::DynamicDawgU64)
//!   - 8-byte edge labels for token sequences, vocabulary IDs, or hash values
//!   - f64 convenience API: `insert_f64()`, `contains_f64()` for time series
//!   - Thread-safe insert AND remove operations
//!
//! - **Need substring/infix search?** Use [`SuffixAutomaton`](suffix_automaton::SuffixAutomaton)
//!   - Find patterns anywhere in text (not just prefixes)
//!   - Specialized for suffix-based matching
//!   - Byte-level operation
//!
//! - **Need substring search with Unicode?** Use [`SuffixAutomatonChar`](suffix_automaton_char::SuffixAutomatonChar)
//!   - Correct character-level substring matching
//!   - Handles accented characters, CJK, emoji correctly
//!   - ~5% performance overhead vs byte-level
//!
//! ## Detailed Comparison
//!
//! | Backend | Best For | Performance | Memory | Dynamic Updates | Unicode |
//! |---------|----------|-------------|--------|-----------------|---------|
//! | **[DoubleArrayTrie]** | General use (recommended) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Insert-only | Byte-level |
//! | **[DoubleArrayTrieChar]** | Unicode text | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Insert-only | ✅ Character-level |
//! | **[DynamicDawg]** | Insert + Remove | ⭐⭐⭐ | ⭐⭐⭐ | ✅ Thread-safe | Byte-level |
//! | **[DynamicDawgChar]** | Unicode + Insert + Remove | ⭐⭐⭐ | ⭐⭐⭐ | ✅ Thread-safe | ✅ Character-level |
//! | **[DynamicDawgU64]** | Token sequences, time series | ⭐⭐⭐ | ⭐⭐ | ✅ Thread-safe | 64-bit labels |
//! | **[PathMapDictionary]** | Frequent updates (requires `pathmap-backend` feature) | ⭐⭐ | ⭐⭐ | ✅ Thread-safe | Byte-level |
//! | **[PathMapDictionaryChar]** | Unicode + updates (requires `pathmap-backend` feature) | ⭐⭐ | ⭐⭐ | ✅ Thread-safe | ✅ Character-level |
//! | **[DawgDictionary]** | Static dictionaries | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | Byte-level |
//! | **[OptimizedDawg]** | Fast construction | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | Byte-level |
//! | **[SuffixAutomaton]** | Substring search | ⭐⭐⭐ | ⭐⭐ | ✅ Insert + Remove | Byte-level |
//! | **[SuffixAutomatonChar]** | Unicode substring search | ⭐⭐⭐ | ⭐⭐ | ✅ Insert + Remove | ✅ Character-level |
//!
//! [DoubleArrayTrie]: double_array_trie::DoubleArrayTrie
//! [DoubleArrayTrieChar]: double_array_trie_char::DoubleArrayTrieChar
//! [DynamicDawg]: dynamic_dawg::DynamicDawg
//! [DynamicDawgChar]: dynamic_dawg_char::DynamicDawgChar
//! [DynamicDawgU64]: dynamic_dawg_u64::DynamicDawgU64
//! [PathMapDictionary]: pathmap::PathMapDictionary
//! [PathMapDictionaryChar]: pathmap_char::PathMapDictionaryChar
//! [DawgDictionary]: dawg::DawgDictionary
//! [OptimizedDawg]: dawg_optimized::OptimizedDawg
//! [SuffixAutomaton]: suffix_automaton::SuffixAutomaton
//! [SuffixAutomatonChar]: suffix_automaton_char::SuffixAutomatonChar
//!
//! ## Performance Benchmarks (10,000 words)
//!
//! ```text
//! Construction:  DAT: 3.2ms    PathMap: 3.5ms    DAWG: 7.2ms
//! Exact Match:   DAT: 6.6µs    DAWG: 19.8µs      PathMap: 71.1µs
//! Contains (100):DAT: 0.22µs   DAWG: 6.7µs       PathMap: 132µs
//! Distance 1:    DAT: 12.9µs   DAWG: 319µs       PathMap: 888µs
//! Distance 2:    DAT: 16.3µs   DAWG: 2,150µs     PathMap: 5,919µs
//! ```
//!
//! ## Common Scenarios
//!
//! **Static or append-only dictionary, ASCII/Latin-1 text:**
//! ```rust,ignore
//! use liblevenshtein::dictionary::double_array_trie::DoubleArrayTrie;
//! let mut dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);
//! dict.insert("new_term");  // Can add terms at runtime
//! ```
//!
//! **Multi-language application with Unicode:**
//! ```rust,ignore
//! use liblevenshtein::dictionary::double_array_trie_char::DoubleArrayTrieChar;
//! let mut dict = DoubleArrayTrieChar::from_terms(vec!["café", "naïve", "中文", "🎉"]);
//! dict.insert("新しい");  // Can add Unicode terms at runtime
//! ```
//!
//! **Application that needs to add AND remove terms at runtime:**
//! ```rust,ignore
//! use liblevenshtein::dictionary::dynamic_dawg::DynamicDawg;
//! let dict = DynamicDawg::from_terms(vec!["initial", "terms"]);
//! dict.insert("new_term");  // Thread-safe
//! dict.remove("old_term");  // Also supports removal
//! ```
//!
//! **Unicode application with dynamic insert AND remove:**
//! ```rust,ignore
//! use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
//! let dict = DynamicDawgChar::from_terms(vec!["café", "naïve", "中文"]);
//! dict.insert("新しい");  // Thread-safe Unicode insertion
//! dict.remove("café");    // Thread-safe removal
//! ```
//!
//! **Finding patterns anywhere in text (not just as prefixes):**
//! ```rust,ignore
//! use liblevenshtein::dictionary::suffix_automaton::SuffixAutomaton;
//! let dict = SuffixAutomaton::<()>::from_text("the quick brown fox");
//! // Can find "quick" even though it's not a prefix
//! ```
//!
//! **Unicode substring search (finding patterns in multi-language text):**
//! ```rust,ignore
//! use liblevenshtein::dictionary::suffix_automaton_char::SuffixAutomatonChar;
//! let dict = SuffixAutomatonChar::<()>::from_text("café naïve 中文 🎉");
//! // Can find "café", "naï", "中", "🎉" anywhere in the text
//! ```

pub mod char_unit;
pub mod dawg;
pub mod dawg_optimized;
pub mod dawg_query;
pub mod double_array_trie;
pub mod double_array_trie_char;
pub mod double_array_trie_char_zipper;
pub mod double_array_trie_zipper;
pub mod dynamic_dawg;
pub mod dynamic_dawg_char;
pub mod dynamic_dawg_char_zipper;
pub mod dynamic_dawg_u64;
pub mod dynamic_dawg_u64_zipper;
pub mod dynamic_dawg_zipper;
pub mod factory;
pub mod iterator;
#[cfg(feature = "pathmap-backend")]
pub mod pathmap;
#[cfg(feature = "pathmap-backend")]
pub mod pathmap_char;
#[cfg(feature = "pathmap-backend")]
pub mod pathmap_zipper;
#[cfg(feature = "persistent-artrie")]
pub mod persistent_artrie;
#[cfg(feature = "persistent-artrie")]
pub mod persistent_artrie_char;
#[cfg(feature = "phonetic-rules")]
pub mod phonetic_normalized;
pub mod prefix_zipper;
pub mod scdawg;
pub mod scdawg_char;
pub mod substring;
pub mod suffix_automaton;
pub mod suffix_automaton_char;
pub mod suffix_automaton_char_zipper;
pub mod suffix_automaton_zipper;
pub mod value;
pub mod zipper;

pub use char_unit::CharUnit;
pub use iterator::{DictionaryIterator, DictionaryTermIterator};
#[cfg(feature = "persistent-artrie")]
pub use persistent_artrie::{PersistentARTrie, PersistentARTrieZipper};
#[cfg(feature = "persistent-artrie")]
pub use persistent_artrie_char::{PersistentARTrieChar, PersistentARTrieCharNode, PersistentARTrieCharZipper};
pub use substring::{BidirectionalDictionaryNode, ExtensionResult, SubstringDictionary, SubstringMatch};
pub use value::DictionaryValue;
pub use zipper::{DictZipper, ValuedDictZipper};

/// Synchronization strategy for dictionary operations.
///
/// Different dictionary backends may have different thread-safety guarantees.
/// This trait allows backends to specify their synchronization requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// Backend requires external synchronization (e.g., RwLock).
    ///
    /// Use this for backends that use interior mutability without
    /// internal synchronization.
    ExternalSync,

    /// Backend is internally synchronized and safe for concurrent access.
    ///
    /// Use this for backends that use atomic operations, locks, or
    /// lock-free data structures internally.
    InternalSync,

    /// Backend is a persistent/immutable data structure.
    ///
    /// Mutations create new versions with structural sharing.
    /// Reads require no synchronization. Writes can use atomic swaps.
    Persistent,
}

/// Core dictionary abstraction for approximate string matching.
///
/// A dictionary represents a collection of terms that can be efficiently
/// traversed character-by-character via graph-like nodes. This trait
/// allows different backend implementations (trie, DAWG, double-array trie,
/// etc.) to be used interchangeably.
pub trait Dictionary {
    /// The node type used for dictionary traversal
    type Node: DictionaryNode;

    /// Get the root node of the dictionary
    fn root(&self) -> Self::Node;

    /// Check if a term exists in the dictionary
    fn contains(&self, term: &str) -> bool {
        let mut node = self.root();
        for unit in <Self::Node as DictionaryNode>::Unit::iter_str(term) {
            match node.transition(unit) {
                Some(next) => node = next,
                None => return false,
            }
        }
        node.is_final()
    }

    /// Get the total number of terms (if available efficiently)
    fn len(&self) -> Option<usize>;

    /// Check if the dictionary is empty
    fn is_empty(&self) -> bool {
        self.len().map(|n| n == 0).unwrap_or(false)
    }

    /// Get the synchronization strategy for this dictionary backend.
    ///
    /// This allows wrappers to optimize synchronization based on
    /// the backend's thread-safety guarantees.
    ///
    /// Default: `ExternalSync` (conservative, always safe)
    fn sync_strategy(&self) -> SyncStrategy {
        SyncStrategy::ExternalSync
    }

    /// Check if this dictionary uses suffix-based matching (substring search).
    ///
    /// Suffix-based dictionaries (like SuffixAutomaton) match substrings anywhere
    /// in the indexed text, whereas prefix-based dictionaries match complete words
    /// from the beginning.
    ///
    /// This affects how the Levenshtein automaton computes match distances:
    /// - Prefix-based: penalizes unmatched query suffix
    /// - Suffix-based: allows partial query matches without penalty
    ///
    /// Default: `false` (prefix-based matching)
    fn is_suffix_based(&self) -> bool {
        false
    }
}

/// Traversable dictionary node.
///
/// Nodes form a graph structure representing the dictionary, where edges
/// are labeled with character units (bytes or Unicode characters) and final
/// nodes mark valid terms.
///
/// # Type Parameters
///
/// The node is generic over [`CharUnit`], which can be:
/// - [`u8`] for byte-level matching (faster, ASCII-optimized)
/// - [`char`] for character-level matching (correct Unicode semantics)
pub trait DictionaryNode: Clone + Send + Sync {
    /// The character unit type for edge labels.
    ///
    /// Use `u8` for byte-level (existing behavior, fastest).
    /// Use `char` for character-level (proper Unicode support).
    type Unit: CharUnit;

    /// Check if this node marks the end of a valid term
    fn is_final(&self) -> bool;

    /// Transition to a child node via the given character unit
    ///
    /// Returns `None` if no such transition exists
    fn transition(&self, label: Self::Unit) -> Option<Self>;

    /// Iterate over all outgoing edges as (unit, child_node) pairs
    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_>;

    /// Check if a specific edge exists
    fn has_edge(&self, label: Self::Unit) -> bool {
        self.transition(label).is_some()
    }

    /// Get the number of outgoing edges (if efficiently available)
    fn edge_count(&self) -> Option<usize> {
        None
    }
}

/// Extension trait for dictionaries that map terms to values.
///
/// This trait enables "fuzzy maps" - dictionaries that associate arbitrary values
/// with terms, allowing efficient filtered queries based on those values. This is
/// particularly useful for contextual code completion where terms are mapped to
/// scope IDs, categories, or other metadata.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::dictionary::value::DictionaryValue;
///
/// // Implement for a dictionary backend that supports values
/// # struct MyDict;
/// # struct MyNode;
/// # impl Clone for MyNode { fn clone(&self) -> Self { MyNode } }
/// # impl DictionaryNode for MyNode {
/// #     fn is_final(&self) -> bool { false }
/// #     fn transition(&self, _label: u8) -> Option<Self> { None }
/// #     fn edges(&self) -> Box<dyn Iterator<Item = (u8, Self)> + '_> { Box::new(std::iter::empty()) }
/// # }
/// # impl Dictionary for MyDict {
/// #     type Node = MyNode;
/// #     fn root(&self) -> Self::Node { MyNode }
/// #     fn len(&self) -> Option<usize> { None }
/// # }
/// impl MappedDictionary for MyDict {
///     type Value = u32; // Map terms to scope IDs
///
///     fn get_value(&self, term: &str) -> Option<Self::Value> {
///         // Look up the value for this term
///         # None
///     }
/// }
/// ```
///
/// # Performance
///
/// Filtering during graph traversal (using values) provides 10-100x speedups
/// compared to post-filtering, especially when filters are highly selective.
pub trait MappedDictionary: Dictionary {
    /// The type of values associated with dictionary terms
    type Value: DictionaryValue;

    /// Get the value associated with a term
    ///
    /// Returns `None` if the term doesn't exist in the dictionary.
    fn get_value(&self, term: &str) -> Option<Self::Value> {
        // Default implementation: traverse to find the term, but return no value
        // (for backward compatibility with non-mapped dictionaries)
        let _ = self.contains(term);
        None
    }

    /// Check if a term exists and its value matches a predicate
    ///
    /// This is more efficient than `get_value` + predicate test, as it can
    /// short-circuit early if the term doesn't exist.
    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool,
    {
        self.get_value(term).is_some_and(|v| predicate(&v))
    }
}

/// Extension trait for dictionary nodes that provide access to values.
///
/// This trait allows nodes to expose values during graph traversal, enabling
/// efficient filtering at query time without materializing all results first.
pub trait MappedDictionaryNode: DictionaryNode {
    /// The type of values associated with terms at this node
    type Value: DictionaryValue;

    /// Get the value at this node if it's a final node
    ///
    /// Returns `None` if this is not a final node, or if no value is associated.
    fn value(&self) -> Option<Self::Value>;
}

/// Extension trait for dictionaries that support inserting values.
///
/// This trait enables mutation of mapped dictionaries, allowing terms to be
/// added or updated with associated values. Used by components like
/// `ContextualCompletionEngine` that need to dynamically add terms.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::dictionary::{MappedDictionary, MutableMappedDictionary};
///
/// # struct MyDict;
/// # impl Dictionary for MyDict {
/// #     type Node = ();
/// #     fn root(&self) -> Self::Node { () }
/// #     fn len(&self) -> Option<usize> { None }
/// # }
/// # impl MappedDictionary for MyDict {
/// #     type Value = u32;
/// # }
/// impl MutableMappedDictionary for MyDict {
///     fn insert_with_value(&self, term: &str, value: Self::Value) -> bool {
///         // Add or update term with value
///         # true
///     }
/// }
/// ```
pub trait MutableMappedDictionary: MappedDictionary {
    /// Insert or update a term with an associated value.
    ///
    /// # Arguments
    ///
    /// * `term` - The term to insert
    /// * `value` - The value to associate with the term
    ///
    /// # Returns
    ///
    /// `true` if this is a new term, `false` if updating an existing term's value.
    fn insert_with_value(&self, term: &str, value: Self::Value) -> bool;

    /// Union this dictionary with another, applying a merge function for conflicting values.
    ///
    /// Iterates through all terms in `other` and:
    /// - Inserts new terms directly
    /// - For existing terms, merges values using `merge_fn`
    ///
    /// # Arguments
    ///
    /// * `other` - The dictionary to union with
    /// * `merge_fn` - Function to merge values when term exists in both dictionaries.
    ///   Takes `(existing_value, other_value)` and returns the merged value.
    ///
    /// # Returns
    ///
    /// Number of terms processed from `other`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    ///
    /// let dict1 = PathMapDictionary::from_terms_with_values([
    ///     ("foo", vec![1, 2]),
    ///     ("bar", vec![1]),
    /// ]);
    ///
    /// let dict2 = PathMapDictionary::from_terms_with_values([
    ///     ("foo", vec![2, 3]),  // Overlap
    ///     ("baz", vec![3]),     // New
    /// ]);
    ///
    /// // Union with custom merge (concatenate)
    /// dict1.union_with(&dict2, |left, right| {
    ///     let mut merged = left.clone();
    ///     merged.extend(right.clone());
    ///     merged.sort();
    ///     merged.dedup();
    ///     merged
    /// });
    ///
    /// // Result: foo -> [1,2,3], bar -> [1], baz -> [3]
    /// ```
    fn union_with<F>(&self, other: &Self, merge_fn: F) -> usize
    where
        F: Fn(&Self::Value, &Self::Value) -> Self::Value,
        Self::Value: Clone;

    /// Union with another dictionary, keeping the right (other's) value on conflicts.
    ///
    /// Convenience method equivalent to `union_with(other, |_, right| right.clone())`.
    fn union_replace(&self, other: &Self) -> usize
    where
        Self::Value: Clone,
    {
        self.union_with(other, |_, right| right.clone())
    }

    /// Update an existing term's value in place, or insert a new term with a default value.
    ///
    /// This method is useful when you want to incrementally modify a value (e.g., adding
    /// elements to a `HashSet` or `Vec`) without replacing it entirely.
    ///
    /// # Arguments
    ///
    /// * `term` - The term to update or insert
    /// * `default_value` - The value to use if the term doesn't exist
    /// * `update_fn` - Function to apply to the existing value if the term exists
    ///
    /// # Returns
    ///
    /// `true` if this was a new term (inserted with default), `false` if an existing term was updated.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::HashSet;
    ///
    /// let dict: DynamicDawgChar<HashSet<u32>> = DynamicDawgChar::new();
    ///
    /// // First call: inserts with default value {1}
    /// dict.update_or_insert("foo", HashSet::from([1]), |set| { set.insert(1); });
    ///
    /// // Second call: updates existing value to {1, 2}
    /// dict.update_or_insert("foo", HashSet::from([2]), |set| { set.insert(2); });
    ///
    /// assert_eq!(dict.get_value("foo"), Some(HashSet::from([1, 2])));
    /// ```
    fn update_or_insert<F>(&self, term: &str, default_value: Self::Value, update_fn: F) -> bool
    where
        F: FnOnce(&mut Self::Value);
}
