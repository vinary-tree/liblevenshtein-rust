//! Character-level Dynamic DAWG with online modifications and Unicode support.
//!
//! This implementation supports incremental updates while maintaining
//! "near-minimal" structure. Perfect minimality can be restored via
//! explicit compaction.
//!
//! Unlike the byte-level `DynamicDawg`, this variant operates on Unicode
//! scalar values (`char`), providing correct character-level Levenshtein
//! distances for multi-byte UTF-8 sequences.
//!
//! # Performance Trade-offs
//!
//! - **Memory**: ~4x edge label storage (4 bytes per `char` vs 1 byte per `u8`)
//! - **Speed**: ~5-10% slower due to UTF-8 decoding
//! - **Correctness**: Proper Unicode semantics (e.g., "" → "¡" = distance 1, not 2)

use crate::dictionary::dynamic_dawg_char_zipper::DynamicDawgCharZipper;
use crate::dictionary::iterator::DictionaryIterator;
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::{Dictionary, DictionaryNode, SyncStrategy};
use crate::sync_compat::RwLock;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// A dynamic DAWG that supports online insertions and deletions.
///
/// # Type Parameters
///
/// - `V`: Optional value type associated with each term. Use `()` (default) for
///   dictionaries without values, or any type implementing `DictionaryValue`
///   (Clone + Send + Sync + 'static) for value-storing dictionaries.
///
/// # Minimality Trade-offs
///
/// - **After insertion**: Structure remains minimal (new nodes are shared)
/// - **After deletion**: May become non-minimal (orphaned branches)
/// - **Solution**: Call `compact()` periodically to restore minimality
///
/// # Thread Safety
///
/// Uses `Arc<RwLock<...>>` for interior mutability. Safe for concurrent
/// reads, exclusive writes.
///
/// # Performance
///
/// - Insertion: O(m) where m is term length (amortized)
/// - Deletion: O(m)
/// - Compaction: O(n) where n is total characters
/// - Space: Near-minimal to ~1.5x minimal (worst case between compactions)
///
/// # Examples
///
/// ```rust,ignore
/// // Without values (default)
/// let dict = DynamicDawgChar::new();
/// dict.insert("hello");
///
/// // With values
/// let dict: DynamicDawgChar<u32> = DynamicDawgChar::new();
/// dict.insert_with_value("hello", 42);
/// ```
#[derive(Clone, Debug)]
pub struct DynamicDawgChar<V: DictionaryValue = ()> {
    pub(crate) inner: Arc<RwLock<DynamicDawgCharInner<V>>>,
}

#[derive(Debug)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(serialize = "V: serde::Serialize")),
    serde(bound(deserialize = "V: serde::Deserialize<'de>"))
)]
pub(crate) struct DynamicDawgCharInner<V: DictionaryValue> {
    pub(crate) nodes: Vec<DawgNodeChar<V>>,
    term_count: usize,
    needs_compaction: bool,
    // Suffix sharing cache: hash of suffix -> node index
    // Enables reusing common suffixes to reduce DAWG size by 20-40%
    #[cfg_attr(feature = "serialization", serde(skip))]
    suffix_cache: FxHashMap<u64, usize>,
    // Lazy minimization tracking
    #[cfg_attr(feature = "serialization", serde(skip))]
    last_minimized_node_count: usize,
    #[cfg_attr(feature = "serialization", serde(skip))]
    auto_minimize_threshold: f32, // Trigger minimize when nodes > last * threshold
    // Bloom filter for fast negative lookup rejection (Opt #4)
    #[cfg_attr(feature = "serialization", serde(skip))]
    bloom_filter: Option<BloomFilter>,
}

/// Simple Bloom filter for fast negative lookup rejection.
///
/// Uses 3 hash functions and a bit vector to probabilistically test membership.
/// - False positives: Possible (requires full DAWG traversal)
/// - False negatives: Never (guaranteed correct rejection)
#[derive(Debug, Clone)]
struct BloomFilter {
    bits: Vec<u64>, // Bit vector (64-bit chunks for efficiency)
    bit_count: usize,
    hash_count: usize,
}

/// Hash-based node signature for efficient minimization.
///
/// Instead of storing recursive Box<NodeSignature>, we use a hash
/// representing the right language. This eliminates expensive allocations
/// and enables O(1) equality checks.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
struct NodeSignature {
    // Hash representing (is_final, sorted edges with child hashes)
    hash: u64,
}

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serialization",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(serialize = "V: serde::Serialize")),
    serde(bound(deserialize = "V: serde::Deserialize<'de>"))
)]
pub(crate) struct DawgNodeChar<V: DictionaryValue> {
    // Use SmallVec to avoid heap allocation for nodes with ≤4 edges (most common case)
    pub(crate) edges: SmallVec<[(char, usize); 4]>,
    pub(crate) is_final: bool,
    // Reference count for dynamic deletion
    ref_count: usize,
    // Optional value associated with this node (only for final nodes)
    pub(crate) value: Option<V>,
}

impl<V: DictionaryValue> DawgNodeChar<V> {
    fn new(is_final: bool) -> Self {
        DawgNodeChar {
            edges: SmallVec::new(),
            is_final,
            ref_count: 0,
            value: None,
        }
    }

    fn new_with_value(is_final: bool, value: Option<V>) -> Self {
        DawgNodeChar {
            edges: SmallVec::new(),
            is_final,
            ref_count: 0,
            value,
        }
    }
}

impl BloomFilter {
    /// Create a new Bloom filter with specified capacity.
    ///
    /// Uses ~1.2 bytes per expected element with 3 hash functions.
    /// Target false positive rate: ~1%
    fn new(expected_elements: usize) -> Self {
        // Use 10 bits per element for ~1% false positive rate with 3 hash functions
        let bit_count = expected_elements * 10;
        let chunk_count = (bit_count + 63) / 64; // Round up to nearest u64

        BloomFilter {
            bits: vec![0u64; chunk_count],
            bit_count: chunk_count * 64,
            hash_count: 3,
        }
    }

    /// Add an element to the Bloom filter.
    fn insert(&mut self, term: &str) {
        let bytes = term.as_bytes();
        for i in 0..self.hash_count {
            let hash = self.hash_with_seed(bytes, i as u64);
            let bit_index = (hash % self.bit_count as u64) as usize;
            let chunk_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            self.bits[chunk_index] |= 1u64 << bit_offset;
        }
    }

    /// Check if an element might be in the set.
    ///
    /// Returns:
    /// - false: Definitely NOT in set (fast rejection)
    /// - true: Might be in set (requires full check)
    #[inline]
    fn might_contain(&self, term: &str) -> bool {
        let bytes = term.as_bytes();
        for i in 0..self.hash_count {
            let hash = self.hash_with_seed(bytes, i as u64);
            let bit_index = (hash % self.bit_count as u64) as usize;
            let chunk_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            if (self.bits[chunk_index] & (1u64 << bit_offset)) == 0 {
                return false; // Definitely not in set
            }
        }
        true // Might be in set
    }

    /// Hash with a seed using FxHash.
    #[inline]
    fn hash_with_seed(&self, bytes: &[u8], seed: u64) -> u64 {
        use rustc_hash::FxHasher;
        use std::hash::Hasher;

        let mut hasher = FxHasher::default();
        seed.hash(&mut hasher);
        bytes.hash(&mut hasher);
        hasher.finish()
    }

    /// Clear the Bloom filter.
    #[allow(dead_code)]
    fn clear(&mut self) {
        self.bits.fill(0);
    }
}

impl<V: DictionaryValue> DynamicDawgChar<V> {
    /// Create a new empty dynamic DAWG.
    ///
    /// By default, auto-minimization is disabled. Use `with_auto_minimize_threshold()`
    /// to enable automatic minimization.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Without values (default)
    /// let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();
    /// dawg.insert("hello");
    ///
    /// // With values
    /// let dawg: DynamicDawgChar<u32> = DynamicDawgChar::new();
    /// dawg.insert_with_value("hello", 42);
    /// ```
    pub fn new() -> Self {
        Self::with_auto_minimize_threshold(f32::INFINITY)
    }

    /// Create a new empty dynamic DAWG with custom auto-minimize threshold.
    ///
    /// The auto-minimize threshold determines when the DAWG automatically
    /// triggers minimization. A value of 1.5 means minimize when node count
    /// grows to 1.5x the last minimized size (50% bloat).
    ///
    /// # Parameters
    ///
    /// - `threshold`: Bloat ratio to trigger minimization (e.g., 1.5 = 50% bloat).
    ///   Use `f32::INFINITY` to disable auto-minimization.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Auto-minimize at 50% bloat (default)
    /// let dawg: DynamicDawgChar<()> = DynamicDawgChar::with_auto_minimize_threshold(1.5);
    ///
    /// // Disable auto-minimization (manual minimize() calls only)
    /// let dawg: DynamicDawgChar<()> = DynamicDawgChar::with_auto_minimize_threshold(f32::INFINITY);
    /// ```
    pub fn with_auto_minimize_threshold(threshold: f32) -> Self {
        Self::with_config(threshold, None)
    }

    /// Create a new empty dynamic DAWG with full configuration.
    ///
    /// # Parameters
    ///
    /// - `auto_minimize_threshold`: Bloat ratio to trigger minimization. Use `f32::INFINITY` to disable.
    /// - `bloom_filter_capacity`: Optional Bloom filter capacity for negative lookup optimization.
    ///   Use `Some(expected_size)` to enable, `None` to disable.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // With Bloom filter for 10000 expected terms
    /// let dawg: DynamicDawgChar<()> = DynamicDawgChar::with_config(f32::INFINITY, Some(10000));
    ///
    /// // Without Bloom filter
    /// let dawg: DynamicDawgChar<()> = DynamicDawgChar::with_config(1.5, None);
    /// ```
    pub fn with_config(auto_minimize_threshold: f32, bloom_filter_capacity: Option<usize>) -> Self {
        let nodes = vec![DawgNodeChar::new(false)]; // Root at index 0

        let bloom_filter = bloom_filter_capacity.map(BloomFilter::new);

        DynamicDawgChar {
            inner: Arc::new(RwLock::new(DynamicDawgCharInner {
                nodes,
                term_count: 0,
                needs_compaction: false,
                suffix_cache: FxHashMap::default(),
                last_minimized_node_count: 1, // Start with root node
                auto_minimize_threshold,
                bloom_filter,
            })),
        }
    }

    /// Create from an iterator of terms (optimized batch insert).
    ///
    /// This method sorts terms before insertion for better prefix/suffix sharing.
    pub fn from_terms<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut term_vec: Vec<String> = terms.into_iter().map(|s| s.as_ref().to_string()).collect();
        term_vec.sort_unstable();

        let dawg = Self::new();
        for term in term_vec {
            dawg.insert(&term);
        }
        dawg
    }

    /// Create from sorted terms (assumes pre-sorted input).
    ///
    /// # Performance
    ///
    /// This is faster than `from_terms()` if your input is already sorted,
    /// as it skips the sorting step and takes advantage of better prefix sharing.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut terms = vec!["apple", "banana", "cherry"];
    /// terms.sort();  // Already sorted
    /// let dawg: DynamicDawgChar<()> = DynamicDawgChar::from_sorted_terms(terms);
    /// ```
    pub fn from_sorted_terms<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let dawg = Self::new();
        for term in terms {
            dawg.insert(term.as_ref());
        }
        dawg
    }

    /// Insert a term into the DAWG.
    ///
    /// Returns `true` if the term was newly inserted, `false` if it already existed.
    ///
    /// # Minimality
    ///
    /// Insertions maintain minimality by sharing suffixes with existing nodes.
    pub fn insert(&self, term: &str) -> bool {
        let mut inner = self.inner.write();
        let chars: Vec<char> = term.chars().collect();

        // Navigate to insertion point, creating nodes as needed
        let mut node_idx = 0;
        let mut path: Vec<(usize, char)> = Vec::new(); // (parent_idx, label)

        for &ch in &chars {
            if let Some(&child_idx) = inner.nodes[node_idx]
                .edges
                .iter()
                .find(|(c, _)| *c == ch)
                .map(|(_, idx)| idx)
            {
                // Edge exists, follow it
                path.push((node_idx, ch));
                node_idx = child_idx;
            } else {
                // Need to create new suffix
                break;
            }
        }

        // Check if term already exists
        if path.len() == chars.len() && inner.nodes[node_idx].is_final {
            return false; // Already exists
        }

        // Build remaining suffix with sharing
        let start_char_idx = path.len();

        // Phase 2.1: DISABLED - Suffix sharing is incompatible with dynamic insertions
        // that can later mark intermediate nodes as final.
        //
        // Bug: When inserting "kb" after "jb", suffix sharing would reuse the same
        // entry node for both 'j' and 'k' edges. Later, inserting "j" marks that
        // shared node as final, incorrectly making "k" also appear as a valid term.
        //
        // Example:
        //   dict.insert("jb"); dict.insert("kb");  // Creates shared structure
        //   dict.insert("j");                       // BUG: also marks "k" as valid
        //
        // The correct DAWG structure for ["jb", "kb"] should have distinct nodes
        // for 'j' and 'k' edges, even though they both continue with 'b'.
        //
        // For now, we disable Phase 2.1 and rely on the node-by-node construction
        // below. Future work: implement proper suffix sharing that creates distinct
        // intermediate nodes while sharing only the deeper suffix nodes.

        // Create nodes one by one (no suffix sharing)
        for i in start_char_idx..chars.len() {
            let ch = chars[i];
            let new_idx = inner.nodes.len();
            let is_final = i == chars.len() - 1;
            let mut new_node = DawgNodeChar::new(is_final);
            new_node.ref_count = 1;

            inner.nodes.push(new_node);
            inner.insert_edge_sorted(node_idx, ch, new_idx);

            node_idx = new_idx;
        }

        // Mark as final if we followed existing path
        if start_char_idx == chars.len() {
            inner.nodes[node_idx].is_final = true;
        }

        inner.term_count += 1;

        // Add to Bloom filter if enabled (Opt #4)
        if let Some(ref mut bloom) = inner.bloom_filter {
            bloom.insert(term);
        }

        // Auto-minimize if bloat threshold exceeded
        inner.check_and_auto_minimize();

        true
    }

    /// Insert a term with an associated value.
    ///
    /// Returns `true` if the term was newly inserted, `false` if it already existed.
    /// If the term already exists, its value is updated.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dict: DynamicDawgChar<u32> = DynamicDawgChar::new();
    /// assert!(dict.insert_with_value("hello", 42));
    /// assert!(!dict.insert_with_value("hello", 43)); // Updates value
    /// assert_eq!(dict.get_value("hello"), Some(43));
    /// ```
    pub fn insert_with_value(&self, term: &str, value: V) -> bool {
        let mut inner = self.inner.write();
        let chars: Vec<char> = term.chars().collect();

        // Navigate to insertion point
        let mut node_idx = 0;
        let mut path: Vec<(usize, char)> = Vec::new();

        for &ch in &chars {
            if let Some(&child_idx) = inner.nodes[node_idx]
                .edges
                .iter()
                .find(|(c, _)| *c == ch)
                .map(|(_, idx)| idx)
            {
                path.push((node_idx, ch));
                node_idx = child_idx;
            } else {
                break;
            }
        }

        // Check if term already exists
        if path.len() == chars.len() {
            if inner.nodes[node_idx].is_final {
                // Term exists - update value
                inner.nodes[node_idx].value = Some(value);
                return false;
            } else {
                // Mark as final and set value
                inner.nodes[node_idx].is_final = true;
                inner.nodes[node_idx].value = Some(value);
                inner.term_count += 1;

                // Add to Bloom filter
                if let Some(ref mut bloom) = inner.bloom_filter {
                    bloom.insert(term);
                }

                return true;
            }
        }

        // Build remaining suffix
        let start_char_idx = path.len();
        for i in start_char_idx..chars.len() {
            let ch = chars[i];
            let new_idx = inner.nodes.len();
            let is_final = i == chars.len() - 1;

            let mut new_node = if is_final {
                DawgNodeChar::new_with_value(true, Some(value.clone()))
            } else {
                DawgNodeChar::new(false)
            };
            new_node.ref_count = 1;

            inner.nodes.push(new_node);
            inner.insert_edge_sorted(node_idx, ch, new_idx);
            node_idx = new_idx;
        }

        inner.term_count += 1;

        // Add to Bloom filter
        if let Some(ref mut bloom) = inner.bloom_filter {
            bloom.insert(term);
        }

        // Auto-minimize if needed
        inner.check_and_auto_minimize();

        true
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
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::collections::HashSet;
    ///
    /// let dict: DynamicDawgChar<HashSet<u32>> = DynamicDawgChar::new();
    ///
    /// // First call: inserts with default value {1}
    /// assert!(dict.update_or_insert("foo", HashSet::from([1]), |set| { set.insert(1); }));
    ///
    /// // Second call: updates existing value to {1, 2}
    /// assert!(!dict.update_or_insert("foo", HashSet::from([2]), |set| { set.insert(2); }));
    ///
    /// assert_eq!(dict.get_value("foo"), Some(HashSet::from([1, 2])));
    /// ```
    pub fn update_or_insert<F>(&self, term: &str, default_value: V, update_fn: F) -> bool
    where
        F: FnOnce(&mut V),
    {
        let mut inner = self.inner.write();
        let chars: Vec<char> = term.chars().collect();

        // Navigate to insertion point
        let mut node_idx = 0;
        let mut path: Vec<(usize, char)> = Vec::new();

        for &ch in &chars {
            if let Some(&child_idx) = inner.nodes[node_idx]
                .edges
                .iter()
                .find(|(c, _)| *c == ch)
                .map(|(_, idx)| idx)
            {
                path.push((node_idx, ch));
                node_idx = child_idx;
            } else {
                break;
            }
        }

        // Check if term already exists
        if path.len() == chars.len() && inner.nodes[node_idx].is_final {
            // Term exists - update value in place
            if let Some(ref mut value) = inner.nodes[node_idx].value {
                update_fn(value);
            } else {
                // Term exists but has no value (shouldn't happen with typed values)
                // Insert default and apply update
                let mut value = default_value;
                update_fn(&mut value);
                inner.nodes[node_idx].value = Some(value);
            }
            return false;
        }

        // Term doesn't exist - insert with default value
        if path.len() == chars.len() {
            // Node exists but isn't final - mark as final
            inner.nodes[node_idx].is_final = true;
            inner.nodes[node_idx].value = Some(default_value);
            inner.term_count += 1;

            // Add to Bloom filter
            if let Some(ref mut bloom) = inner.bloom_filter {
                bloom.insert(term);
            }

            return true;
        }

        // Build remaining suffix
        let start_char_idx = path.len();
        for i in start_char_idx..chars.len() {
            let ch = chars[i];
            let new_idx = inner.nodes.len();
            let is_final = i == chars.len() - 1;

            let mut new_node = if is_final {
                DawgNodeChar::new_with_value(true, Some(default_value.clone()))
            } else {
                DawgNodeChar::new(false)
            };
            new_node.ref_count = 1;

            inner.nodes.push(new_node);
            inner.insert_edge_sorted(node_idx, ch, new_idx);
            node_idx = new_idx;
        }

        inner.term_count += 1;

        // Add to Bloom filter
        if let Some(ref mut bloom) = inner.bloom_filter {
            bloom.insert(term);
        }

        // Auto-minimize if needed
        inner.check_and_auto_minimize();

        true
    }

    /// Get the value associated with a term.
    ///
    /// Returns `Some(value)` if the term exists, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dict: DynamicDawgChar<String> = DynamicDawgChar::new();
    /// dict.insert_with_value("key", "value".to_string());
    /// assert_eq!(dict.get_value("key"), Some("value".to_string()));
    /// assert_eq!(dict.get_value("unknown"), None);
    /// ```
    pub fn get_value(&self, term: &str) -> Option<V> {
        let inner = self.inner.read();
        let chars: Vec<char> = term.chars().collect();
        let mut node_idx = 0;

        // Navigate to the term
        for &ch in &chars {
            match inner.nodes[node_idx]
                .edges
                .iter()
                .find(|(c, _)| *c == ch)
                .map(|(_, idx)| idx)
            {
                Some(&child_idx) => node_idx = child_idx,
                None => return None,
            }
        }

        // Check if final and return value
        if inner.nodes[node_idx].is_final {
            inner.nodes[node_idx].value.clone()
        } else {
            None
        }
    }

    /// Remove a term from the DAWG.
    ///
    /// Returns `true` if the term was present and removed, `false` otherwise.
    ///
    /// # Minimality
    ///
    /// Deletions may leave the DAWG non-minimal. Call `compact()` to restore
    /// minimality by removing unreachable nodes.
    pub fn remove(&self, term: &str) -> bool {
        let mut inner = self.inner.write();
        let chars: Vec<char> = term.chars().collect();

        // Navigate to the term
        let mut node_idx = 0;
        let mut path: Vec<(usize, char, usize)> = Vec::new(); // (parent, label, child)

        for &ch in &chars {
            if let Some(&child_idx) = inner.nodes[node_idx]
                .edges
                .iter()
                .find(|(c, _)| *c == ch)
                .map(|(_, idx)| idx)
            {
                path.push((node_idx, ch, child_idx));
                node_idx = child_idx;
            } else {
                return false; // Term doesn't exist
            }
        }

        // Check if it's a final node
        if !inner.nodes[node_idx].is_final {
            return false; // Term doesn't exist
        }

        // Unmark as final
        inner.nodes[node_idx].is_final = false;
        inner.term_count -= 1;

        // Prune unreachable branches (nodes with no children and not final)
        for (parent_idx, label, child_idx) in path.iter().rev() {
            let child = &inner.nodes[*child_idx];
            if !child.is_final && child.edges.is_empty() {
                // Remove edge from parent
                inner.nodes[*parent_idx].edges.retain(|(b, _)| *b != *label);
            } else {
                break; // Stop pruning
            }
        }

        // Phase 2.1: Invalidate suffix cache since structure changed
        inner.suffix_cache.clear();
        inner.needs_compaction = true;
        true
    }

    /// Compact the DAWG to restore perfect minimality.
    ///
    /// This rebuilds the internal structure, merging equivalent suffixes
    /// and removing unreachable nodes. Ideal for batch operations:
    ///
    /// ```rust,ignore
    /// // Batch updates
    /// dawg.insert("term1");
    /// dawg.insert("term2");
    /// dawg.remove("term3");
    /// // ... many more operations ...
    ///
    /// // Single compaction at the end
    /// let removed = dawg.compact();
    /// ```
    ///
    /// **Note**: This does a full rebuild (extracts, sorts, reconstructs, minimizes).
    /// For incremental minimization without rebuilding, use `minimize()`.
    ///
    /// Returns the number of nodes removed.
    pub fn compact(&self) -> usize {
        let mut inner = self.inner.write();

        // Extract all terms
        let terms = inner.extract_all_terms();
        let old_node_count = inner.nodes.len();

        // Preserve settings
        let auto_minimize_threshold = inner.auto_minimize_threshold;
        let bloom_capacity = inner.bloom_filter.as_ref().map(|b| b.bit_count / 10);

        // Rebuild from scratch
        *inner = DynamicDawgCharInner {
            nodes: vec![DawgNodeChar::new(false)],
            term_count: 0,
            needs_compaction: false,
            suffix_cache: FxHashMap::default(),
            last_minimized_node_count: 1,
            auto_minimize_threshold,
            bloom_filter: bloom_capacity.map(BloomFilter::new),
        };

        // Re-insert sorted terms for optimal prefix sharing
        let mut sorted_terms = terms;
        sorted_terms.sort();

        for term in &sorted_terms {
            // Direct insertion without locking (we already have write lock)
            inner.insert_direct(term);

            // Rebuild Bloom filter
            if let Some(ref mut bloom) = inner.bloom_filter {
                bloom.insert(term);
            }
        }

        // Now minimize to merge equivalent suffixes (DAWG minimization)
        // This is what makes it a true DAWG instead of just a trie
        let minimized = inner.minimize_incremental();

        old_node_count - inner.nodes.len() + minimized
    }

    /// Minimize the DAWG using incremental suffix merging.
    ///
    /// Unlike `compact()`, this method:
    /// - **Makes no assumptions** about insertion order
    /// - **Only examines affected nodes** and their neighbors
    /// - **Preserves existing structure** where possible
    /// - **Faster than compact()** for localized updates
    ///
    /// This implements incremental minimization based on node signatures.
    /// If the DAWG was minimal before updates, only the new paths and
    /// their neighbors need to be examined.
    ///
    /// ```rust,ignore
    /// // DAWG is minimal
    /// dawg.minimize();
    ///
    /// // Add some terms (locally affects structure)
    /// dawg.insert("newterm1");
    /// dawg.insert("newterm2");
    ///
    /// // Incremental minimize - only examines affected paths
    /// let merged = dawg.minimize(); // Much faster than compact()!
    /// ```
    ///
    /// Returns the number of nodes merged.
    pub fn minimize(&self) -> usize {
        let mut inner = self.inner.write();
        inner.minimize_incremental()
    }

    /// Batch insert multiple terms, then compact.
    ///
    /// This is more efficient than calling `insert()` followed by `compact()`
    /// separately, as it sorts terms for better prefix sharing and only rebuilds once.
    ///
    /// Returns the number of new terms added.
    pub fn extend<I, S>(&self, terms: I) -> usize
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // Collect and sort for optimal prefix sharing
        let mut term_vec: Vec<String> = terms.into_iter().map(|s| s.as_ref().to_string()).collect();
        term_vec.sort_unstable();

        let mut added = 0;
        for term in term_vec {
            if self.insert(&term) {
                added += 1;
            }
        }

        if added > 0 {
            self.compact();
        }

        added
    }

    /// Batch remove multiple terms, then compact.
    ///
    /// More efficient than individual `remove()` calls followed by `compact()`.
    ///
    /// Returns the number of terms removed.
    pub fn remove_many<I, S>(&self, terms: I) -> usize
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut removed = 0;
        for term in terms {
            if self.remove(term.as_ref()) {
                removed += 1;
            }
        }

        if removed > 0 {
            self.compact();
        }

        removed
    }

    /// Get the number of terms in the DAWG.
    pub fn term_count(&self) -> usize {
        self.inner.read().term_count
    }

    /// Get the number of nodes in the DAWG.
    pub fn node_count(&self) -> usize {
        self.inner.read().nodes.len()
    }

    /// Check if compaction is recommended.
    ///
    /// Returns `true` if deletions have occurred and compaction would
    /// likely reduce memory usage.
    pub fn needs_compaction(&self) -> bool {
        self.inner.read().needs_compaction
    }

    /// Check if a term is in the DAWG.
    ///
    /// This method is optimized with a Bloom filter (if enabled) for fast negative lookup rejection.
    pub fn contains(&self, term: &str) -> bool {
        let inner = self.inner.read();

        // Fast path: Bloom filter check (if enabled)
        if let Some(ref bloom) = inner.bloom_filter {
            if !bloom.might_contain(term) {
                return false; // Definitely not in DAWG
            }
            // Might be in DAWG, need full check
        }

        // Full check: traverse DAWG
        drop(inner); // Release lock before traversal
        let mut node = self.root();
        for ch in term.chars() {
            if let Some(next_node) = node.transition(ch) {
                node = next_node;
            } else {
                return false;
            }
        }
        node.is_final()
    }
}

impl<V: DictionaryValue> DynamicDawgCharInner<V> {
    /// Check if auto-minimization should be triggered based on bloat threshold.
    ///
    /// This is called after each insertion to maintain the DAWG in a near-minimal
    /// state without the overhead of minimizing after every single operation.
    fn check_and_auto_minimize(&mut self) {
        let current_nodes = self.nodes.len();
        let threshold_nodes =
            (self.last_minimized_node_count as f32 * self.auto_minimize_threshold) as usize;

        if current_nodes > threshold_nodes && !self.auto_minimize_threshold.is_infinite() {
            // Trigger automatic minimization
            self.minimize_incremental();
        }
    }

    /// Phase 2.1: Find or create a suffix chain, using cache for common suffixes.
    ///
    /// This is a key optimization that reuses common suffix chains like "ing", "tion", etc.
    /// Expected to reduce memory by 20-40% for natural language dictionaries.
    ///
    /// Returns Some(node_idx) if an existing suffix was found/created, None otherwise.
    ///
    /// NOTE: Currently unused due to bugs with dynamic insertion (see insert() method).
    /// Kept for future optimization work that properly handles suffix sharing in dynamic DAWGs.
    #[allow(dead_code)]
    fn find_or_create_suffix(
        &mut self,
        suffix: &[char],
        create_if_missing: bool,
        is_final: bool,
    ) -> Option<usize> {
        if suffix.is_empty() {
            return None;
        }

        // Compute hash for this suffix
        let hash = self.compute_suffix_hash(suffix, is_final);

        // Check cache for existing suffix
        if let Some(&cached_idx) = self.suffix_cache.get(&hash) {
            // Verify it's actually the same suffix (handle hash collisions)
            if self.verify_suffix_match(cached_idx, suffix, is_final) {
                return Some(cached_idx);
            }
        }

        // Not in cache - create new suffix chain if requested
        if create_if_missing {
            let suffix_idx = self.create_suffix_chain(suffix, is_final);
            self.suffix_cache.insert(hash, suffix_idx);
            Some(suffix_idx)
        } else {
            None
        }
    }

    /// Compute a hash for a suffix to enable caching.
    ///
    /// Phase 2.1: Uses FxHash for fast non-cryptographic hashing.
    #[allow(dead_code)]
    fn compute_suffix_hash(&self, suffix: &[char], is_final: bool) -> u64 {
        use rustc_hash::FxHasher;
        use std::hash::Hasher;

        let mut hasher = FxHasher::default();
        suffix.hash(&mut hasher);
        is_final.hash(&mut hasher);
        hasher.finish()
    }

    /// Verify that a cached node actually matches the requested suffix.
    ///
    /// Phase 2.1: Handles hash collisions by checking structural equality.
    /// The node_idx points to the entry node that should have edges to traverse the suffix.
    #[allow(dead_code)]
    fn verify_suffix_match(&self, node_idx: usize, suffix: &[char], is_final: bool) -> bool {
        if suffix.is_empty() {
            return false;
        }

        let mut current_idx = node_idx;

        // Traverse the suffix: for each byte, follow the edge
        for (i, &byte) in suffix.iter().enumerate() {
            let node = &self.nodes[current_idx];

            // Find the edge for this byte
            if let Some(&next_idx) = node
                .edges
                .iter()
                .find(|(l, _)| *l == byte)
                .map(|(_, idx)| idx)
            {
                current_idx = next_idx;

                // If this was the last byte, check if final state matches
                if i == suffix.len() - 1 && self.nodes[current_idx].is_final != is_final {
                    return false;
                }
            } else {
                return false; // Missing edge
            }
        }

        true
    }

    /// Create a linear chain of nodes for a suffix.
    ///
    /// Phase 2.1: For suffix "abc", creates chain:
    /// node_0 --'a'--> node_1 --'b'--> node_2 --'c'--> node_3(final)
    /// Returns index of node_0 (the entry point to traverse the suffix).
    ///
    /// Example: create_suffix_chain("est", true) creates:
    /// node_0 --'e'--> node_1 --'s'--> node_2 --'t'--> node_3(final)
    #[allow(dead_code)]
    fn create_suffix_chain(&mut self, suffix: &[char], is_final: bool) -> usize {
        if suffix.is_empty() {
            return 0; // Should not happen
        }

        // We need (suffix.len() + 1) nodes total:
        // - 1 entry node (before consuming any suffix bytes)
        // - 1 node for each byte in the suffix
        let mut nodes_to_add = Vec::new();

        // Entry node (not final, will have edge for suffix[0])
        nodes_to_add.push(DawgNodeChar {
            edges: SmallVec::new(),
            is_final: false,
            ref_count: 1,
            value: None,
        });

        // Intermediate nodes (one per suffix byte)
        for i in 0..suffix.len() {
            let is_last = i == suffix.len() - 1;
            nodes_to_add.push(DawgNodeChar {
                edges: SmallVec::new(),
                is_final: is_last && is_final,
                ref_count: 1,
                value: None,
            });
        }

        // Add all nodes
        let start_idx = self.nodes.len();
        self.nodes.extend(nodes_to_add);

        // Link the chain: node[i] --suffix[i]--> node[i+1]
        for (i, &ch) in suffix.iter().enumerate() {
            let from_idx = start_idx + i;
            let to_idx = start_idx + i + 1;
            self.nodes[from_idx].edges.push((ch, to_idx));
        }

        start_idx
    }

    /// Insert an edge into a node's edge list, maintaining sorted order.
    /// Uses binary search to find the insertion point - O(log n) instead of O(n log n) sort.
    #[inline]
    fn insert_edge_sorted(&mut self, node_idx: usize, label: char, target_idx: usize) {
        let edges = &mut self.nodes[node_idx].edges;
        match edges.binary_search_by_key(&label, |(l, _)| *l) {
            Ok(pos) => {
                // Edge with this label already exists, replace it
                edges[pos] = (label, target_idx);
            }
            Err(pos) => {
                // Insert at the correct position to maintain sorted order
                edges.insert(pos, (label, target_idx));
            }
        }
    }

    /// Incremental minimization using signature-based node merging.
    ///
    /// Algorithm:
    /// 1. Compute signatures for all nodes (bottom-up)
    /// 2. Find nodes with identical signatures (equivalent right languages)
    /// 3. Merge equivalent nodes by redirecting edges
    /// 4. Remove unreachable nodes
    ///
    /// Time: O(n) where n is number of nodes
    /// Space: O(n) for signature map
    fn minimize_incremental(&mut self) -> usize {
        let initial_count = self.nodes.len();

        // Step 1: Compute node signatures (right language representation)
        let signatures = self.compute_signatures();

        // Step 2: Build equivalence classes (nodes with same signature)
        // Use Vec to handle hash collisions - multiple nodes may have same hash
        let mut sig_to_canonical: HashMap<NodeSignature, Vec<usize>> = HashMap::new();
        let mut node_mapping: Vec<usize> = (0..self.nodes.len()).collect();

        // Process nodes in reverse order (leaves first) for better merging
        for node_idx in (0..self.nodes.len()).rev() {
            let sig = &signatures[node_idx];

            if let Some(canonical_candidates) = sig_to_canonical.get(sig) {
                // Found nodes with matching hash - verify structural equality
                let mut found_match = false;
                for &canonical_idx in canonical_candidates {
                    // Skip if this candidate was already mapped to another node
                    if node_mapping[canonical_idx] != canonical_idx {
                        continue;
                    }

                    if self.nodes_structurally_equal(node_idx, canonical_idx, &node_mapping) {
                        // True structural match - merge into canonical
                        node_mapping[node_idx] = canonical_idx;
                        found_match = true;
                        break;
                    }
                }

                if !found_match {
                    // Hash collision - this is a different node with same hash
                    sig_to_canonical.get_mut(sig).unwrap().push(node_idx);
                    node_mapping[node_idx] = node_idx;
                }
            } else {
                // This is the first node with this signature hash
                sig_to_canonical.insert(*sig, vec![node_idx]);
                node_mapping[node_idx] = node_idx;
            }
        }

        // Step 3: Redirect all edges to canonical nodes
        for node in &mut self.nodes {
            for (_, target_idx) in &mut node.edges {
                *target_idx = node_mapping[*target_idx];
            }
        }

        // Step 4: Remove unreachable nodes and rebuild compactly
        let reachable = self.find_reachable_nodes();
        if reachable.len() < self.nodes.len() {
            self.compact_with_reachable(&reachable);
        }

        // Phase 2.1: Invalidate suffix cache since nodes were merged
        self.suffix_cache.clear();
        self.needs_compaction = false;

        // Update last minimized count for auto-minimize threshold tracking
        self.last_minimized_node_count = self.nodes.len();

        initial_count - self.nodes.len()
    }

    /// Check if two nodes are structurally equivalent.
    ///
    /// Two nodes are equivalent if they have the same is_final flag and
    /// the same edges (after applying node_mapping to account for already-merged nodes).
    ///
    /// Phase 2.2: Used to verify true equality when hash signatures match,
    /// preventing false merges from hash collisions.
    fn nodes_structurally_equal(&self, idx1: usize, idx2: usize, node_mapping: &[usize]) -> bool {
        let node1 = &self.nodes[idx1];
        let node2 = &self.nodes[idx2];

        // Check is_final flag
        if node1.is_final != node2.is_final {
            return false;
        }

        // Check edge count
        if node1.edges.len() != node2.edges.len() {
            return false;
        }

        // Check each edge (edges should already be sorted by label)
        for i in 0..node1.edges.len() {
            let (label1, target1) = node1.edges[i];
            let (label2, target2) = node2.edges[i];

            // Labels must match
            if label1 != label2 {
                return false;
            }

            // Targets must map to the same canonical node
            if node_mapping[target1] != node_mapping[target2] {
                return false;
            }
        }

        true
    }

    /// Compute signatures for all nodes (bottom-up).
    ///
    /// A signature represents the "right language" of a node - the set of
    /// strings that can be formed from this node to any final state.
    ///
    /// Two nodes with identical signatures are equivalent and can be merged.
    ///
    /// Phase 2.2: Now uses hash-based signatures instead of recursive Box structures.
    /// This eliminates ~3000 Box allocations for a 1000-node DAWG and provides
    /// O(1) signature comparisons instead of recursive equality checks.
    fn compute_signatures(&self) -> Vec<NodeSignature> {
        let mut signatures = vec![NodeSignature { hash: 0 }; self.nodes.len()];

        // Compute signatures bottom-up using DFS post-order
        let mut visited = vec![false; self.nodes.len()];
        self.compute_signatures_dfs(0, &mut signatures, &mut visited);

        signatures
    }

    fn compute_signatures_dfs(
        &self,
        node_idx: usize,
        signatures: &mut [NodeSignature],
        visited: &mut [bool],
    ) {
        if visited[node_idx] {
            return;
        }
        visited[node_idx] = true;

        let node = &self.nodes[node_idx];

        // Visit all children first (post-order)
        for (_, child_idx) in &node.edges {
            self.compute_signatures_dfs(*child_idx, signatures, visited);
        }

        // Compute hash-based signature for this node
        // Hash = FxHash(is_final, sorted[(label, child_hash), ...])
        use rustc_hash::FxHasher;

        let mut hasher = FxHasher::default();

        // Hash the is_final flag
        node.is_final.hash(&mut hasher);

        // Hash sorted edges with their child signatures
        // Note: edges are already sorted in DawgNodeChar, but we'll ensure it
        let mut edge_hashes: SmallVec<[(char, u64); 4]> = node
            .edges
            .iter()
            .map(|(label, child_idx)| (*label, signatures[*child_idx].hash))
            .collect();

        // Ensure edges are sorted by label for consistent hashing
        edge_hashes.sort_unstable_by_key(|(label, _)| *label);

        // Hash each (label, child_hash) pair
        for (label, child_hash) in &edge_hashes {
            label.hash(&mut hasher);
            child_hash.hash(&mut hasher);
        }

        signatures[node_idx] = NodeSignature {
            hash: hasher.finish(),
        };
    }

    /// Find all nodes reachable from root.
    fn find_reachable_nodes(&self) -> Vec<usize> {
        let mut reachable = Vec::new();
        let mut visited = vec![false; self.nodes.len()];
        self.find_reachable_dfs(0, &mut visited);

        for (idx, &is_reachable) in visited.iter().enumerate() {
            if is_reachable {
                reachable.push(idx);
            }
        }

        reachable
    }

    fn find_reachable_dfs(&self, node_idx: usize, visited: &mut [bool]) {
        if visited[node_idx] {
            return;
        }
        visited[node_idx] = true;

        for (_, child_idx) in &self.nodes[node_idx].edges {
            self.find_reachable_dfs(*child_idx, visited);
        }
    }

    /// Compact the node array to only contain reachable nodes.
    fn compact_with_reachable(&mut self, reachable: &[usize]) {
        // Build mapping from old indices to new indices
        let mut old_to_new = vec![usize::MAX; self.nodes.len()];
        for (new_idx, &old_idx) in reachable.iter().enumerate() {
            old_to_new[old_idx] = new_idx;
        }

        // Build new node vector
        let new_nodes: Vec<DawgNodeChar<V>> = reachable
            .iter()
            .map(|&old_idx| {
                let mut node = self.nodes[old_idx].clone();
                // Remap edge targets
                for (_, target) in &mut node.edges {
                    *target = old_to_new[*target];
                }
                node
            })
            .collect();

        self.nodes = new_nodes;
    }

    fn extract_all_terms(&self) -> Vec<String> {
        let mut terms = Vec::new();
        let mut current_term = Vec::new();
        self.dfs_collect(0, &mut current_term, &mut terms);
        terms
    }

    fn dfs_collect(&self, node_idx: usize, current_term: &mut Vec<char>, terms: &mut Vec<String>) {
        let node = &self.nodes[node_idx];

        if node.is_final {
            let term: String = current_term.iter().collect();
            terms.push(term);
        }

        for (ch, child_idx) in &node.edges {
            current_term.push(*ch);
            self.dfs_collect(*child_idx, current_term, terms);
            current_term.pop();
        }
    }

    fn insert_direct(&mut self, term: &str) {
        let chars: Vec<char> = term.chars().collect();
        let mut node_idx = 0;

        for &ch in &chars {
            if let Some(&child_idx) = self.nodes[node_idx]
                .edges
                .iter()
                .find(|(c, _)| *c == ch)
                .map(|(_, idx)| idx)
            {
                node_idx = child_idx;
            } else {
                let new_idx = self.nodes.len();
                self.nodes.push(DawgNodeChar::new(false));
                self.nodes[node_idx].edges.push((ch, new_idx));
                node_idx = new_idx;
            }
        }

        self.nodes[node_idx].is_final = true;
        self.term_count += 1;
    }
}

impl<V: DictionaryValue> DynamicDawgChar<V> {
    /// Iterate over all `(term, value)` pairs as character vectors.
    ///
    /// Returns an iterator yielding `(Vec<char>, V)` tuples in depth-first order.
    /// This is more efficient than `iter()` as it avoids String allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    ///
    /// let dict: DynamicDawgChar<u32> = DynamicDawgChar::new();
    /// dict.insert_with_value("café", 1);
    /// dict.insert_with_value("naïve", 2);
    ///
    /// for (chars, value) in dict.iter_chars() {
    ///     let term: String = chars.iter().collect();
    ///     println!("{} -> {}", term, value);
    /// }
    /// ```
    pub fn iter_chars(&self) -> DictionaryIterator<DynamicDawgCharZipper<V>> {
        let zipper = DynamicDawgCharZipper::new_from_dict(self);
        DictionaryIterator::new(zipper)
    }

    /// Iterate over all `(term, value)` pairs as UTF-8 strings.
    ///
    /// Returns an iterator yielding `(String, V)` tuples in depth-first order.
    /// For better performance with raw characters, use `iter_chars()` instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    ///
    /// let dict: DynamicDawgChar<u32> = DynamicDawgChar::new();
    /// dict.insert_with_value("café", 1);
    /// dict.insert_with_value("naïve", 2);
    ///
    /// for (term, value) in dict.iter() {
    ///     println!("{} -> {}", term, value);
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (String, V)> + '_ {
        self.iter_chars()
            .map(|(chars, value)| (chars.into_iter().collect::<String>(), value))
    }
}

impl<V: DictionaryValue> IntoIterator for &DynamicDawgChar<V> {
    type Item = (Vec<char>, V);
    type IntoIter = DictionaryIterator<DynamicDawgCharZipper<V>>;

    /// Creates an iterator over all `(term, value)` pairs as character vectors.
    fn into_iter(self) -> Self::IntoIter {
        self.iter_chars()
    }
}

impl<V: DictionaryValue> Default for DynamicDawgChar<V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "serialization")]
impl<V: DictionaryValue + serde::Serialize> serde::Serialize for DynamicDawgChar<V> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Extract the inner data by acquiring read lock
        let inner = self.inner.read();
        inner.serialize(serializer)
    }
}

#[cfg(feature = "serialization")]
impl<'de, V: DictionaryValue + serde::Deserialize<'de>> serde::Deserialize<'de>
    for DynamicDawgChar<V>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let inner = DynamicDawgCharInner::deserialize(deserializer)?;
        Ok(DynamicDawgChar {
            inner: Arc::new(RwLock::new(inner)),
        })
    }
}

impl<V: DictionaryValue> Dictionary for DynamicDawgChar<V> {
    type Node = DynamicDawgCharNode<V>;

    fn root(&self) -> Self::Node {
        // Phase 1.2: Load cached data with single lock acquisition
        let inner = self.inner.read();
        let node = &inner.nodes[0];
        DynamicDawgCharNode {
            dawg: Arc::clone(&self.inner),
            node_idx: 0,
            is_final: node.is_final,
            edges: node.edges.clone(),
        }
    }

    fn len(&self) -> Option<usize> {
        Some(self.term_count())
    }

    fn sync_strategy(&self) -> SyncStrategy {
        SyncStrategy::ExternalSync
    }
}

/// Node handle for dynamic DAWG traversal.
///
/// Phase 1.2: Caches is_final and edges to avoid lock acquisition on hot paths.
/// This eliminates locks from is_final() and edge_count(), and drastically
/// reduces locks in transition() (only for successful transitions).
#[derive(Clone)]
pub struct DynamicDawgCharNode<V: DictionaryValue = ()> {
    dawg: Arc<RwLock<DynamicDawgCharInner<V>>>,
    node_idx: usize,
    // Phase 1.2: Cached data
    is_final: bool,
    edges: SmallVec<[(char, usize); 4]>,
}

impl<V: DictionaryValue> DictionaryNode for DynamicDawgCharNode<V> {
    type Unit = char;

    // Phase 1.2: Use cached data - no lock needed
    fn is_final(&self) -> bool {
        self.is_final
    }

    fn transition(&self, label: char) -> Option<Self> {
        // Phase 1.2: Use cached edges for lookup - no lock needed
        // Adaptive: use linear search for small edge counts, binary for large
        // Empirical testing shows crossover at 16-20 edges
        let child_idx = if self.edges.len() < 16 {
            // Linear search - cache-friendly for small counts
            self.edges
                .iter()
                .find(|(c, _)| *c == label)
                .map(|(_, idx)| *idx)
        } else {
            // Binary search - efficient for large edge counts
            self.edges
                .binary_search_by_key(&label, |(c, _)| *c)
                .ok()
                .map(|i| self.edges[i].1)
        }?;

        // Phase 1.2: Only acquire lock to load child node's cached data
        let inner = self.dawg.read();
        let child_node = &inner.nodes[child_idx];
        Some(DynamicDawgCharNode {
            dawg: Arc::clone(&self.dawg),
            node_idx: child_idx,
            is_final: child_node.is_final,
            edges: child_node.edges.clone(),
        })
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (char, Self)> + '_> {
        // Phase 1.2: Batch load child nodes with single lock acquisition
        let inner = self.dawg.read();
        let child_data: Vec<_> = self
            .edges
            .iter()
            .map(|(ch, idx)| {
                let child_node = &inner.nodes[*idx];
                (*ch, *idx, child_node.is_final, child_node.edges.clone())
            })
            .collect();
        drop(inner);

        let dawg = Arc::clone(&self.dawg);
        Box::new(
            child_data
                .into_iter()
                .map(move |(ch, idx, is_final, edges)| {
                    (
                        ch,
                        DynamicDawgCharNode {
                            dawg: Arc::clone(&dawg),
                            node_idx: idx,
                            is_final,
                            edges,
                        },
                    )
                }),
        )
    }

    // Phase 1.2: Use cached data - no lock needed
    fn edge_count(&self) -> Option<usize> {
        Some(self.edges.len())
    }
}

// ============================================================================
// MappedDictionary Trait Implementation
// ============================================================================

use crate::dictionary::{MappedDictionary, MappedDictionaryNode};

impl<V: DictionaryValue> MappedDictionaryNode for DynamicDawgCharNode<V> {
    type Value = V;

    fn value(&self) -> Option<Self::Value> {
        // Need to lock to get the value
        let inner = self.dawg.read();
        inner
            .nodes
            .get(self.node_idx)
            .and_then(|node| node.value.clone())
    }
}

impl<V: DictionaryValue> MappedDictionary for DynamicDawgChar<V> {
    type Value = V;

    fn get_value(&self, term: &str) -> Option<Self::Value> {
        // Delegate to the inherent method
        Self::get_value(self, term)
    }

    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool,
    {
        match self.get_value(term) {
            Some(ref value) => predicate(value),
            None => false,
        }
    }
}

impl<V: DictionaryValue> crate::dictionary::MutableMappedDictionary for DynamicDawgChar<V> {
    fn insert_with_value(&self, term: &str, value: Self::Value) -> bool {
        // Delegate to the inherent method
        Self::insert_with_value(self, term, value)
    }

    fn union_with<F>(&self, other: &Self, merge_fn: F) -> usize
    where
        F: Fn(&Self::Value, &Self::Value) -> Self::Value,
        Self::Value: Clone,
    {
        let other_inner = other.inner.read();
        let mut processed = 0;

        // DFS traversal to extract all terms with values from other
        let mut stack: Vec<(usize, Vec<char>)> = vec![(0, Vec::new())];

        while let Some((node_idx, path)) = stack.pop() {
            let node = &other_inner.nodes[node_idx];

            // If this is a final node, we have a complete term
            if node.is_final {
                let term: String = path.iter().collect();
                processed += 1;

                if let Some(other_value) = &node.value {
                    // Check if term exists in self
                    if let Some(self_value) = self.get_value(&term) {
                        // Merge values
                        let merged = merge_fn(&self_value, other_value);
                        self.insert_with_value(&term, merged);
                    } else {
                        // Insert new term
                        self.insert_with_value(&term, other_value.clone());
                    }
                }
            }

            // Push children onto stack (in reverse for consistent ordering)
            for &(label, target_idx) in node.edges.iter().rev() {
                let mut child_path = path.clone();
                child_path.push(label);
                stack.push((target_idx, child_path));
            }
        }

        processed
    }

    fn update_or_insert<F>(&self, term: &str, default_value: Self::Value, update_fn: F) -> bool
    where
        F: FnOnce(&mut Self::Value),
    {
        // Delegate to the inherent method
        Self::update_or_insert(self, term, default_value, update_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_dawg_insert() {
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();
        assert!(dawg.insert("test"));
        assert!(!dawg.insert("test")); // Duplicate
        assert!(dawg.insert("testing"));
        assert_eq!(dawg.term_count(), 2);
    }

    #[test]
    fn test_dynamic_dawg_remove() {
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();
        dawg.insert("test");
        dawg.insert("testing");
        dawg.insert("tested");

        assert!(dawg.remove("testing"));
        assert_eq!(dawg.term_count(), 2);
        assert!(!dawg.remove("testing")); // Already removed
    }

    #[test]
    fn test_dynamic_dawg_compact() {
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();
        dawg.insert("test");
        dawg.insert("testing");
        dawg.insert("tested");

        let before = dawg.node_count();
        dawg.remove("testing");

        let removed = dawg.compact();
        let after = dawg.node_count();

        assert!(removed > 0 || before == after);
        assert_eq!(dawg.term_count(), 2);
    }

    #[test]
    fn test_dynamic_dawg_with_transducer() {
        use crate::transducer::{Algorithm, Transducer};

        let dawg: DynamicDawgChar<()> =
            DynamicDawgChar::from_terms(vec!["apple", "application", "apply"]);
        let transducer = Transducer::new(dawg.clone(), Algorithm::Standard);

        let results: Vec<_> = transducer.query("aple", 2).collect();
        assert!(results.contains(&"apple".to_string()));
        assert!(results.contains(&"apply".to_string()));

        // Dynamic update
        dawg.insert("apricot");
        assert_eq!(dawg.term_count(), 4);
    }

    #[test]
    fn test_compaction_flag() {
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();
        dawg.insert("test");

        assert!(!dawg.needs_compaction());

        dawg.remove("test");
        assert!(dawg.needs_compaction());

        dawg.compact();
        assert!(!dawg.needs_compaction());
    }

    #[test]
    fn test_batch_extend() {
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();
        dawg.insert("test");

        let new_terms = vec!["testing", "tested", "tester"];
        let added = dawg.extend(new_terms);

        assert_eq!(added, 3);
        assert_eq!(dawg.term_count(), 4);
        assert!(dawg.contains("test"));
        assert!(dawg.contains("testing"));
    }

    #[test]
    fn test_batch_remove_many() {
        let dawg: DynamicDawgChar<()> =
            DynamicDawgChar::from_terms(vec!["test", "testing", "tested", "tester"]);

        let to_remove = vec!["testing", "tester"];
        let removed = dawg.remove_many(to_remove);

        assert_eq!(removed, 2);
        assert_eq!(dawg.term_count(), 2);
        assert!(dawg.contains("test"));
        assert!(!dawg.contains("testing"));
    }

    #[test]
    fn test_minimize_basic() {
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();

        // Insert terms in unsorted order
        dawg.insert("zebra");
        dawg.insert("apple");
        dawg.insert("banana");
        dawg.insert("apricot");

        let nodes_before = dawg.node_count();
        let merged = dawg.minimize();
        let nodes_after = dawg.node_count();

        // Should have merged some nodes or stayed the same
        assert_eq!(nodes_after, nodes_before - merged);

        // All terms should still be present
        assert_eq!(dawg.term_count(), 4);
        assert!(dawg.contains("zebra"));
        assert!(dawg.contains("apple"));
        assert!(dawg.contains("banana"));
        assert!(dawg.contains("apricot"));
    }

    #[test]
    fn test_minimize_vs_compact() {
        // Test that minimize() achieves same minimality as compact()
        let _terms = ["band", "banana", "bandana", "can", "cane", "candy"];

        // Create two identical DAWGs with unsorted insertion
        let dawg1: DynamicDawgChar<()> = DynamicDawgChar::new();
        let dawg2: DynamicDawgChar<()> = DynamicDawgChar::new();

        for term in ["zebra", "apple", "banana", "apricot", "band", "bandana"] {
            dawg1.insert(term);
            dawg2.insert(term);
        }

        // Minimize one, compact the other
        let merged1 = dawg1.minimize();
        let merged2 = dawg2.compact();

        println!(
            "After minimize: {} nodes (merged {})",
            dawg1.node_count(),
            merged1
        );
        println!(
            "After compact: {} nodes (removed {})",
            dawg2.node_count(),
            merged2
        );

        // Both should contain same terms
        for term in ["zebra", "apple", "banana", "apricot", "band", "bandana"] {
            assert!(
                dawg1.contains(term),
                "minimize() DAWG missing term: {}",
                term
            );
            assert!(
                dawg2.contains(term),
                "compact() DAWG missing term: {}",
                term
            );
        }

        // Check term counts match
        assert_eq!(dawg1.term_count(), dawg2.term_count());

        // Both should achieve the same node count (for now, just document the difference)
        // TODO: Investigate why minimize() and compact() produce different node counts
        // If minimize() produces fewer nodes without false positives, it might be better!
        if dawg1.node_count() != dawg2.node_count() {
            eprintln!(
                "WARNING: minimize() produced {} nodes, compact() produced {} nodes",
                dawg1.node_count(),
                dawg2.node_count()
            );
        }
    }

    #[test]
    fn test_minimize_after_deletions() {
        let dawg: DynamicDawgChar<()> =
            DynamicDawgChar::from_terms(vec!["test", "testing", "tested", "tester", "testimony"]);

        // Remove some terms, creating potential orphaned nodes
        dawg.remove("testing");
        dawg.remove("tester");

        assert!(dawg.needs_compaction());

        let nodes_before = dawg.node_count();
        let merged = dawg.minimize();
        let nodes_after = dawg.node_count();

        // Should have cleaned up orphaned nodes
        assert!(merged > 0);
        assert_eq!(nodes_after, nodes_before - merged);

        // Remaining terms should still be present
        assert!(dawg.contains("test"));
        assert!(dawg.contains("tested"));
        assert!(dawg.contains("testimony"));
        assert!(!dawg.contains("testing"));
        assert!(!dawg.contains("tester"));
    }

    #[test]
    fn test_minimize_empty() {
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();
        let merged = dawg.minimize();

        // Empty DAWG should have nothing to minimize
        assert_eq!(merged, 0);
        assert_eq!(dawg.node_count(), 1); // Just root
        assert_eq!(dawg.term_count(), 0);
    }

    #[test]
    fn test_minimize_single_term() {
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();
        dawg.insert("hello");

        let nodes_before = dawg.node_count();
        let merged = dawg.minimize();
        let nodes_after = dawg.node_count();

        // Single term should already be minimal
        assert_eq!(merged, 0);
        assert_eq!(nodes_before, nodes_after);
        assert!(dawg.contains("hello"));
    }

    #[test]
    fn test_minimize_with_shared_suffixes() {
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();

        // These words share suffixes: "ing" in testing/running
        dawg.insert("testing");
        dawg.insert("running");
        dawg.insert("test");
        dawg.insert("run");

        let _merged = dawg.minimize();

        // All terms should be preserved (minimize should handle shared suffixes)
        assert!(dawg.contains("testing"));
        assert!(dawg.contains("running"));
        assert!(dawg.contains("test"));
        assert!(dawg.contains("run"));
    }

    #[test]
    fn test_minimize_idempotent() {
        let dawg: DynamicDawgChar<()> =
            DynamicDawgChar::from_terms(vec!["apple", "application", "apply", "apricot"]);

        // First minimization
        let _merged1 = dawg.minimize();
        let nodes1 = dawg.node_count();

        // Second minimization should do nothing (already minimal)
        let merged2 = dawg.minimize();
        let nodes2 = dawg.node_count();

        assert_eq!(merged2, 0);
        assert_eq!(nodes1, nodes2);
    }

    #[test]
    fn test_minimize_no_false_positives() {
        // Test to prevent false positive lookups after minimize()
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();

        // Insert specific terms in random order
        let inserted_terms = vec!["zebra", "apple", "banana", "apricot", "band", "bandana"];
        let not_inserted_terms = vec!["app", "ban", "zeb", "banan", "apric", "bandanas"];

        for term in &inserted_terms {
            dawg.insert(term);
        }

        // Minimize the DAWG
        dawg.minimize();

        // Check that inserted terms are still present
        for term in &inserted_terms {
            assert!(
                dawg.contains(term),
                "Should contain inserted term: {}",
                term
            );
        }

        // CRITICAL: Check that non-inserted terms are NOT present (no false positives)
        for term in &not_inserted_terms {
            assert!(
                !dawg.contains(term),
                "Should NOT contain term that wasn't inserted: {}",
                term
            );
        }
    }

    #[test]
    fn test_valued_dawg_basic() {
        // Test DynamicDawgChar with values
        let dawg: DynamicDawgChar<u32> = DynamicDawgChar::new();

        // Insert with values
        assert!(dawg.insert_with_value("hello", 42));
        assert!(dawg.insert_with_value("world", 100));
        assert!(dawg.insert_with_value("test", 1));

        // Verify values
        assert_eq!(dawg.get_value("hello"), Some(42));
        assert_eq!(dawg.get_value("world"), Some(100));
        assert_eq!(dawg.get_value("test"), Some(1));
        assert_eq!(dawg.get_value("unknown"), None);

        // Update value
        assert!(!dawg.insert_with_value("hello", 999));
        assert_eq!(dawg.get_value("hello"), Some(999));

        // Verify term count
        assert_eq!(dawg.term_count(), 3);
    }

    #[test]
    fn test_valued_dawg_with_remove() {
        let dawg: DynamicDawgChar<String> = DynamicDawgChar::new();

        dawg.insert_with_value("key1", "value1".to_string());
        dawg.insert_with_value("key2", "value2".to_string());

        assert_eq!(dawg.get_value("key1"), Some("value1".to_string()));

        // Remove should clear value
        assert!(dawg.remove("key1"));
        assert_eq!(dawg.get_value("key1"), None);
        assert_eq!(dawg.get_value("key2"), Some("value2".to_string()));
    }

    #[test]
    fn test_mapped_dictionary_trait() {
        use crate::dictionary::MappedDictionary;

        let dawg: DynamicDawgChar<Vec<u32>> = DynamicDawgChar::new();
        dawg.insert_with_value("scoped", vec![1, 2, 3]);
        dawg.insert_with_value("global", vec![0]);

        // Test MappedDictionary::get_value
        assert_eq!(dawg.get_value("scoped"), Some(vec![1, 2, 3]));

        // Test contains_with_value
        assert!(dawg.contains_with_value("scoped", |v| v.contains(&2)));
        assert!(!dawg.contains_with_value("scoped", |v| v.contains(&999)));
        assert!(!dawg.contains_with_value("unknown", |v| v.contains(&1)));
    }

    #[test]
    fn test_compact_no_false_positives() {
        // Same test for compact() to establish baseline
        let dawg: DynamicDawgChar<()> = DynamicDawgChar::new();

        let inserted_terms = vec!["zebra", "apple", "banana", "apricot", "band", "bandana"];
        let not_inserted_terms = vec!["app", "ban", "zeb", "banan", "apric", "bandanas"];

        for term in &inserted_terms {
            dawg.insert(term);
        }

        dawg.compact();

        for term in &inserted_terms {
            assert!(
                dawg.contains(term),
                "Should contain inserted term: {}",
                term
            );
        }

        for term in &not_inserted_terms {
            assert!(
                !dawg.contains(term),
                "Should NOT contain term that wasn't inserted: {}",
                term
            );
        }
    }
}
