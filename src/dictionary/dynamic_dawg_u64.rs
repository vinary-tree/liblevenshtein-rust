//! 64-bit Dynamic DAWG with lock-free operations and 8-byte edge labels.
//!
//! This implementation uses lock-free algorithms with atomic operations for
//! concurrent access. Reads are wait-free (no blocking, no retries), and
//! writes use CAS (compare-and-swap) loops for lock-free progress guarantees.
//!
//! Unlike the byte-level `DynamicDawg` (u8 edges) or character-level
//! `DynamicDawgChar` (char/u32 edges), this variant uses 64-bit labels (u64),
//! enabling:
//!
//! - **Token sequences**: Vocabulary IDs, hash-based tokens
//! - **Time series**: f64 values encoded via `f64::to_bits()` / `f64::from_bits()`
//! - **Binary data**: Any 8-byte aligned data
//!
//! # Primary API
//!
//! The primary API uses direct sequence operations:
//!
//! - [`insert_sequence`](DynamicDawgU64::insert_sequence): Insert a u64 sequence
//! - [`contains_sequence`](DynamicDawgU64::contains_sequence): Check if sequence exists
//! - [`insert_f64`](DynamicDawgU64::insert_f64): Insert f64 series (convenience)
//! - [`contains_f64`](DynamicDawgU64::contains_f64): Check f64 series (convenience)
//!
//! The string-based API (via `CharUnit` trait) is available but secondary.
//!
//! # Thread Safety
//!
//! - **Reads**: Wait-free - multiple readers never block
//! - **Writes**: Lock-free - at least one writer always makes progress
//! - **Memory**: Arc-based with automatic reclamation via arc-swap

use crate::dictionary::dynamic_dawg_u64_zipper::DynamicDawgU64Zipper;
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::{Dictionary, DictionaryNode, SyncStrategy};
use arc_swap::{ArcSwap, ArcSwapOption};
use smallvec::SmallVec;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Immutable edge list that can be atomically swapped.
///
/// When adding/removing edges, we clone this list, modify the clone,
/// and atomically swap it in place using CAS.
#[derive(Clone, Debug, Default)]
pub(crate) struct EdgeList<V: DictionaryValue> {
    pub(crate) edges: SmallVec<[(u64, Arc<DawgNodeU64<V>>); 4]>,
}

impl<V: DictionaryValue> EdgeList<V> {
    fn new() -> Self {
        EdgeList {
            edges: SmallVec::new(),
        }
    }

    /// Find a child node by label.
    #[inline]
    pub(crate) fn find(&self, label: u64) -> Option<&Arc<DawgNodeU64<V>>> {
        self.edges
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, node)| node)
    }

    /// Insert an edge in sorted order, returning a new EdgeList.
    fn with_edge(&self, label: u64, node: Arc<DawgNodeU64<V>>) -> Self {
        let mut new_edges = self.edges.clone();

        // Find insertion position to maintain sorted order
        let pos = new_edges
            .iter()
            .position(|(l, _)| *l >= label)
            .unwrap_or(new_edges.len());

        // Check if edge already exists (shouldn't happen in normal use)
        if pos < new_edges.len() && new_edges[pos].0 == label {
            new_edges[pos] = (label, node);
        } else {
            new_edges.insert(pos, (label, node));
        }

        EdgeList { edges: new_edges }
    }

    /// Remove an edge by label, returning a new EdgeList.
    fn without_edge(&self, label: u64) -> Self {
        let new_edges: SmallVec<_> = self
            .edges
            .iter()
            .filter(|(l, _)| *l != label)
            .cloned()
            .collect();
        EdgeList { edges: new_edges }
    }
}

/// A lock-free DAWG node with atomic fields.
///
/// - `edges`: Atomically swappable edge list (copy-on-write)
/// - `is_final`: Atomic bool (monotonic: false → true only during normal ops)
/// - `value`: Atomically swappable value
#[derive(Debug)]
pub(crate) struct DawgNodeU64<V: DictionaryValue> {
    /// Edges to child nodes (atomically swappable)
    pub(crate) edges: ArcSwap<EdgeList<V>>,
    /// Whether this node represents a complete term
    pub(crate) is_final: AtomicBool,
    /// Value associated with this node (only meaningful if is_final)
    pub(crate) value: ArcSwapOption<V>,
}

impl<V: DictionaryValue> Clone for DawgNodeU64<V> {
    fn clone(&self) -> Self {
        // Load edges: Guard<Arc<EdgeList>> -> clone inner EdgeList
        let edges_guard = self.edges.load();
        let edges_clone = (**edges_guard).clone();

        // Load value: Guard<Option<Arc<V>>> -> clone inner V if present
        let value_guard = self.value.load();
        // (*value_guard) gives Option<Arc<V>>, as_ref() gives Option<&Arc<V>>
        // Then map to clone the inner V
        let value_clone: Option<V> = value_guard.as_ref().map(|arc| (**arc).clone());

        DawgNodeU64 {
            edges: ArcSwap::from_pointee(edges_clone),
            is_final: AtomicBool::new(self.is_final.load(Ordering::Acquire)),
            value: match value_clone {
                Some(v) => ArcSwapOption::from_pointee(Some(v)),
                None => ArcSwapOption::empty(),
            },
        }
    }
}

impl<V: DictionaryValue> DawgNodeU64<V> {
    fn new(is_final: bool) -> Self {
        DawgNodeU64 {
            edges: ArcSwap::from_pointee(EdgeList::new()),
            is_final: AtomicBool::new(is_final),
            value: ArcSwapOption::empty(),
        }
    }

    fn new_with_value(is_final: bool, value: Option<V>) -> Self {
        DawgNodeU64 {
            edges: ArcSwap::from_pointee(EdgeList::new()),
            is_final: AtomicBool::new(is_final),
            value: match value {
                Some(v) => ArcSwapOption::from_pointee(Some(v)),
                None => ArcSwapOption::empty(),
            },
        }
    }
}

/// A dynamic DAWG with lock-free concurrent access.
///
/// # Type Parameters
///
/// - `V`: Optional value type associated with each term. Use `()` (default) for
///   dictionaries without values, or any type implementing `DictionaryValue`
///   (Clone + Send + Sync + 'static) for value-storing dictionaries.
///
/// # Thread Safety
///
/// - **Reads**: Wait-free - no locks, no retries, no blocking
/// - **Writes**: Lock-free - uses CAS loops, guaranteed progress
///
/// # Performance
///
/// - Insertion: O(m) where m is term length (amortized, with CAS retries)
/// - Lookup: O(m) - wait-free
/// - Space: Higher than RwLock version due to Arc overhead per node
///
/// # Examples
///
/// ```rust,ignore
/// use std::thread;
/// use std::sync::Arc;
///
/// let dict = Arc::new(DynamicDawgU64::<()>::new());
///
/// // Concurrent reads and writes
/// let handles: Vec<_> = (0..10).map(|i| {
///     let d = dict.clone();
///     thread::spawn(move || {
///         d.insert_sequence(&[i, i+1, i+2]);
///         d.contains_sequence(&[i, i+1, i+2])
///     })
/// }).collect();
///
/// for h in handles {
///     assert!(h.join().unwrap());
/// }
/// ```
pub struct DynamicDawgU64<V: DictionaryValue = ()> {
    /// Root node of the DAWG
    root: Arc<DawgNodeU64<V>>,
    /// Number of terms in the DAWG
    term_count: AtomicUsize,
    /// Whether compaction is recommended
    needs_compaction: AtomicBool,
}

impl<V: DictionaryValue> Clone for DynamicDawgU64<V> {
    fn clone(&self) -> Self {
        // Deep clone the entire structure
        DynamicDawgU64 {
            root: Arc::new(self.deep_clone_node(&self.root)),
            term_count: AtomicUsize::new(self.term_count.load(Ordering::Relaxed)),
            needs_compaction: AtomicBool::new(self.needs_compaction.load(Ordering::Relaxed)),
        }
    }
}

impl<V: DictionaryValue> std::fmt::Debug for DynamicDawgU64<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicDawgU64")
            .field("term_count", &self.term_count.load(Ordering::Relaxed))
            .field(
                "needs_compaction",
                &self.needs_compaction.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl<V: DictionaryValue> Default for DynamicDawgU64<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: DictionaryValue> DynamicDawgU64<V> {
    /// Deep clone a node and all its descendants.
    fn deep_clone_node(&self, node: &Arc<DawgNodeU64<V>>) -> DawgNodeU64<V> {
        let edges = node.edges.load();
        let new_edges: SmallVec<_> = edges
            .edges
            .iter()
            .map(|(label, child)| (*label, Arc::new(self.deep_clone_node(child))))
            .collect();

        // Clone the value: Guard<Option<Arc<V>>> -> Option<V>
        let value_guard = node.value.load();
        let value_clone: Option<V> = value_guard.as_ref().map(|arc| (**arc).clone());

        DawgNodeU64 {
            edges: ArcSwap::from_pointee(EdgeList { edges: new_edges }),
            is_final: AtomicBool::new(node.is_final.load(Ordering::Acquire)),
            value: match value_clone {
                Some(v) => ArcSwapOption::from_pointee(Some(v)),
                None => ArcSwapOption::empty(),
            },
        }
    }

    /// Create a new empty dynamic DAWG.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
    /// dawg.insert_sequence(&[1, 2, 3]);
    /// ```
    pub fn new() -> Self {
        DynamicDawgU64 {
            root: Arc::new(DawgNodeU64::new(false)),
            term_count: AtomicUsize::new(0),
            needs_compaction: AtomicBool::new(false),
        }
    }

    /// Get the root node Arc of the DAWG.
    ///
    /// This is primarily used by zippers and iterators for navigation.
    #[inline]
    pub(crate) fn root_arc(&self) -> Arc<DawgNodeU64<V>> {
        self.root.clone()
    }

    /// Create a new empty dynamic DAWG with custom auto-minimize threshold.
    ///
    /// Note: Auto-minimization is not yet implemented in the lock-free version.
    /// This constructor is provided for API compatibility.
    pub fn with_auto_minimize_threshold(_threshold: f32) -> Self {
        Self::new()
    }

    /// Create a new empty dynamic DAWG with full configuration.
    ///
    /// Note: Bloom filter and auto-minimization are not yet implemented
    /// in the lock-free version. This constructor is provided for API compatibility.
    pub fn with_config(_auto_minimize_threshold: f32, _bloom_filter_capacity: Option<usize>) -> Self {
        Self::new()
    }

    /// Create from an iterator of terms.
    pub fn from_terms<I, S>(terms: I) -> Self
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

    /// Create from sorted terms.
    pub fn from_sorted_terms<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::from_terms(terms)
    }

    /// Insert a term into the DAWG (string-based API).
    ///
    /// Returns `true` if the term was newly inserted, `false` if it already existed.
    pub fn insert(&self, term: &str) -> bool {
        let sequence: Vec<u64> = crate::dictionary::CharUnit::from_str(term);
        self.insert_sequence(&sequence)
    }

    /// Insert a term with an associated value.
    ///
    /// Returns `true` if the term was newly inserted, `false` if it already existed.
    /// If the term already exists, its value is updated.
    pub fn insert_with_value(&self, term: &str, value: V) -> bool {
        let sequence: Vec<u64> = crate::dictionary::CharUnit::from_str(term);
        self.insert_sequence_with_value(&sequence, value)
    }

    /// Update an existing term's value in place, or insert with default value.
    pub fn update_or_insert<F>(&self, term: &str, default_value: V, update_fn: F) -> bool
    where
        F: FnOnce(&mut V),
    {
        let sequence: Vec<u64> = crate::dictionary::CharUnit::from_str(term);
        self.update_or_insert_sequence(&sequence, default_value, update_fn)
    }

    /// Get the value associated with a term.
    pub fn get_value(&self, term: &str) -> Option<V> {
        let sequence: Vec<u64> = crate::dictionary::CharUnit::from_str(term);
        self.get_sequence_value(&sequence)
    }

    /// Remove a term from the DAWG.
    ///
    /// Returns `true` if the term was present and removed, `false` otherwise.
    pub fn remove(&self, term: &str) -> bool {
        let sequence: Vec<u64> = crate::dictionary::CharUnit::from_str(term);
        self.remove_sequence(&sequence)
    }

    /// Compact the DAWG (placeholder - not fully implemented for lock-free).
    ///
    /// In the lock-free version, this extracts all terms and rebuilds.
    pub fn compact(&self) -> usize {
        // For now, just clear the needs_compaction flag
        self.needs_compaction.store(false, Ordering::Relaxed);
        0
    }

    /// Minimize the DAWG (placeholder - not fully implemented for lock-free).
    pub fn minimize(&self) -> usize {
        0
    }

    /// Batch insert multiple terms.
    pub fn extend<I, S>(&self, terms: I) -> usize
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut added = 0;
        for term in terms {
            if self.insert(term.as_ref()) {
                added += 1;
            }
        }
        added
    }

    /// Batch remove multiple terms.
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
        removed
    }

    /// Get the number of terms in the DAWG.
    #[inline]
    pub fn term_count(&self) -> usize {
        self.term_count.load(Ordering::Relaxed)
    }

    /// Get the number of nodes in the DAWG.
    pub fn node_count(&self) -> usize {
        self.count_nodes_recursive(&self.root)
    }

    fn count_nodes_recursive(&self, node: &Arc<DawgNodeU64<V>>) -> usize {
        let edges = node.edges.load();
        let mut count = 1;
        for (_, child) in edges.edges.iter() {
            count += self.count_nodes_recursive(child);
        }
        count
    }

    /// Check if compaction is recommended.
    #[inline]
    pub fn needs_compaction(&self) -> bool {
        self.needs_compaction.load(Ordering::Relaxed)
    }

    /// Check if a term is in the DAWG (string-based API).
    ///
    /// This is a wait-free operation.
    pub fn contains(&self, term: &str) -> bool {
        let sequence: Vec<u64> = crate::dictionary::CharUnit::from_str(term);
        self.contains_sequence(&sequence)
    }

    // =========================================================================
    // Sequence-based API (primary for u64 usage)
    // =========================================================================

    /// Insert a u64 sequence directly (lock-free).
    ///
    /// Returns `true` if the sequence was newly inserted, `false` if it already existed.
    ///
    /// # Lock-Free Guarantee
    ///
    /// This method uses CAS loops to atomically update edge lists. At least one
    /// concurrent writer always makes progress, preventing livelock.
    pub fn insert_sequence(&self, sequence: &[u64]) -> bool {
        if sequence.is_empty() {
            // Mark root as final
            if self
                .root
                .is_final
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                self.term_count.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            return false;
        }

        let mut current: Arc<DawgNodeU64<V>> = self.root.clone();

        for (i, &label) in sequence.iter().enumerate() {
            let is_last = i == sequence.len() - 1;

            loop {
                let edges = current.edges.load();

                if let Some(child) = edges.find(label) {
                    // Edge exists, follow it
                    if is_last {
                        // Mark child as final
                        if child
                            .is_final
                            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                            .is_ok()
                        {
                            self.term_count.fetch_add(1, Ordering::Relaxed);
                            return true;
                        }
                        return false; // Already exists
                    }
                    // Clone the Arc to own it, then drop the edge guard
                    current = child.clone();
                    break;
                } else {
                    // Need to add edge - CAS loop
                    let new_node = Arc::new(DawgNodeU64::new(is_last));
                    let new_edges = Arc::new(edges.with_edge(label, new_node.clone()));

                    // Try to swap the edge list
                    let prev = current.edges.compare_and_swap(&edges, new_edges.clone());
                    if Arc::ptr_eq(&prev, &edges) {
                        // CAS succeeded
                        if is_last {
                            self.term_count.fetch_add(1, Ordering::Relaxed);
                            return true;
                        }
                        // Continue with the node we just inserted
                        current = new_node;
                        break;
                    }
                    // CAS failed - another thread modified edges, retry
                }
            }
        }

        true
    }

    /// Insert a sequence with an associated value.
    pub fn insert_sequence_with_value(&self, sequence: &[u64], value: V) -> bool {
        if sequence.is_empty() {
            // Mark root as final with value
            if self
                .root
                .is_final
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                self.root.value.store(Some(Arc::new(value)));
                self.term_count.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            // Already final - update value
            self.root.value.store(Some(Arc::new(value)));
            return false;
        }

        let mut current: Arc<DawgNodeU64<V>> = self.root.clone();

        for (i, &label) in sequence.iter().enumerate() {
            let is_last = i == sequence.len() - 1;

            loop {
                let edges = current.edges.load();

                if let Some(child) = edges.find(label) {
                    if is_last {
                        // Mark child as final with value
                        if child
                            .is_final
                            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                            .is_ok()
                        {
                            child.value.store(Some(Arc::new(value)));
                            self.term_count.fetch_add(1, Ordering::Relaxed);
                            return true;
                        }
                        // Already final - update value
                        child.value.store(Some(Arc::new(value)));
                        return false;
                    }
                    // Clone the Arc to own it
                    current = child.clone();
                    break;
                } else {
                    // Need to add edge
                    let new_node = Arc::new(if is_last {
                        DawgNodeU64::new_with_value(true, Some(value.clone()))
                    } else {
                        DawgNodeU64::new(false)
                    });
                    let new_edges = Arc::new(edges.with_edge(label, new_node.clone()));

                    let prev = current.edges.compare_and_swap(&edges, new_edges.clone());
                    if Arc::ptr_eq(&prev, &edges) {
                        if is_last {
                            self.term_count.fetch_add(1, Ordering::Relaxed);
                            return true;
                        }
                        // Continue with the node we just inserted
                        current = new_node;
                        break;
                    }
                }
            }
        }

        true
    }

    /// Update or insert a sequence with value.
    pub fn update_or_insert_sequence<F>(&self, sequence: &[u64], default_value: V, update_fn: F) -> bool
    where
        F: FnOnce(&mut V),
    {
        // Navigate to the node, creating path if needed
        if sequence.is_empty() {
            if self.root.is_final.load(Ordering::Acquire) {
                // Update existing value
                let current_val = self.root.value.load();
                // current_val is Guard<Option<Arc<V>>>; *current_val is Option<Arc<V>>
                if let Some(val) = &*current_val {
                    // val is &Arc<V>, **val is V
                    let mut new_val = (**val).clone();
                    update_fn(&mut new_val);
                    self.root.value.store(Some(Arc::new(new_val)));
                }
                return false;
            } else {
                // Insert new
                self.root.is_final.store(true, Ordering::Release);
                self.root.value.store(Some(Arc::new(default_value)));
                self.term_count.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }

        // First, ensure the path exists
        let is_new = self.insert_sequence_with_value(sequence, default_value.clone());

        if !is_new {
            // Path existed, update the value
            let node = self.find_node(sequence);
            if let Some(node) = node {
                let current_val = node.value.load();
                if let Some(val) = &*current_val {
                    let mut new_val = (**val).clone();
                    update_fn(&mut new_val);
                    node.value.store(Some(Arc::new(new_val)));
                }
            }
        }

        is_new
    }

    /// Find a node by sequence (internal helper).
    fn find_node(&self, sequence: &[u64]) -> Option<Arc<DawgNodeU64<V>>> {
        let mut current: Arc<DawgNodeU64<V>> = self.root.clone();

        for &label in sequence {
            let edges = current.edges.load();
            match edges.find(label) {
                Some(child) => {
                    // Clone the Arc to increment refcount and own the node.
                    // This ensures the node stays alive after the guard drops.
                    current = child.clone();
                }
                None => return None,
            }
        }

        Some(current)
    }

    /// Get the value for a sequence.
    pub fn get_sequence_value(&self, sequence: &[u64]) -> Option<V> {
        let mut current: Arc<DawgNodeU64<V>> = self.root.clone();

        for &label in sequence {
            let edges = current.edges.load();
            match edges.find(label) {
                Some(child) => {
                    // Clone Arc to own the node, ensuring it outlives the guard
                    current = child.clone();
                }
                None => return None,
            }
        }

        if current.is_final.load(Ordering::Acquire) {
            let val = current.value.load();
            // val is Guard<Option<Arc<V>>>; *val is Option<Arc<V>>
            if let Some(v) = &*val {
                // v is &Arc<V>, **v is V
                return Some((**v).clone());
            }
        }
        None
    }

    /// Check if a sequence exists in the DAWG (wait-free).
    ///
    /// This is a wait-free operation - no locks, no retries, no blocking.
    #[inline]
    pub fn contains_sequence(&self, sequence: &[u64]) -> bool {
        let mut current: Arc<DawgNodeU64<V>> = self.root.clone();

        for &label in sequence {
            let edges = current.edges.load();
            match edges.find(label) {
                Some(child) => {
                    // Clone Arc to own the node, ensuring it outlives the guard
                    current = child.clone();
                }
                None => return false,
            }
        }

        current.is_final.load(Ordering::Acquire)
    }

    /// Remove a sequence from the DAWG.
    ///
    /// Note: This only unmarks the node as final. The node structure remains
    /// for potential future use. Call `compact()` to reclaim unused nodes.
    pub fn remove_sequence(&self, sequence: &[u64]) -> bool {
        if sequence.is_empty() {
            if self
                .root
                .is_final
                .compare_exchange(true, false, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                self.root.value.store(None);
                self.term_count.fetch_sub(1, Ordering::Relaxed);
                self.needs_compaction.store(true, Ordering::Relaxed);
                return true;
            }
            return false;
        }

        // Navigate to the node
        let mut current: Arc<DawgNodeU64<V>> = self.root.clone();

        for &label in sequence {
            let edges = current.edges.load();
            match edges.find(label) {
                Some(child) => {
                    // Clone Arc to own the node, ensuring it outlives the guard
                    current = child.clone();
                }
                None => return false,
            }
        }

        // Try to unmark as final
        if current
            .is_final
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            current.value.store(None);
            self.term_count.fetch_sub(1, Ordering::Relaxed);
            self.needs_compaction.store(true, Ordering::Relaxed);
            return true;
        }

        false
    }

    // =========================================================================
    // f64 convenience API
    // =========================================================================

    /// Insert an f64 series as bit patterns.
    ///
    /// The f64 values are converted to their IEEE 754 bit representation
    /// using `f64::to_bits()`, then stored as a u64 sequence.
    pub fn insert_f64(&self, series: &[f64]) -> bool {
        let sequence: Vec<u64> = series.iter().map(|f| f.to_bits()).collect();
        self.insert_sequence(&sequence)
    }

    /// Insert an f64 series with an associated value.
    pub fn insert_f64_with_value(&self, series: &[f64], value: V) -> bool {
        let sequence: Vec<u64> = series.iter().map(|f| f.to_bits()).collect();
        self.insert_sequence_with_value(&sequence, value)
    }

    /// Check if an f64 series exists in the DAWG.
    pub fn contains_f64(&self, series: &[f64]) -> bool {
        let sequence: Vec<u64> = series.iter().map(|f| f.to_bits()).collect();
        self.contains_sequence(&sequence)
    }

    /// Get the value for an f64 series.
    pub fn get_f64_value(&self, series: &[f64]) -> Option<V> {
        let sequence: Vec<u64> = series.iter().map(|f| f.to_bits()).collect();
        self.get_sequence_value(&sequence)
    }

    /// Remove an f64 series from the DAWG.
    pub fn remove_f64(&self, series: &[f64]) -> bool {
        let sequence: Vec<u64> = series.iter().map(|f| f.to_bits()).collect();
        self.remove_sequence(&sequence)
    }

    // =========================================================================
    // Iterator support
    // =========================================================================

    /// Create a zipper at the root of the DAWG.
    pub fn zipper(&self) -> DynamicDawgU64Zipper<V> {
        DynamicDawgU64Zipper::new_from_dict(self)
    }

    /// Get the root node (for zipper access).
    pub(crate) fn root_node(&self) -> &Arc<DawgNodeU64<V>> {
        &self.root
    }

    /// Iterate over all terms in the DAWG.
    pub fn iter(&self) -> impl Iterator<Item = Vec<u64>> + '_ {
        DawgIterator::new(self)
    }

    /// Iterate over all terms with their values.
    pub fn iter_with_values(&self) -> impl Iterator<Item = (Vec<u64>, V)> + '_ {
        DawgIteratorWithValues::new(self)
    }
}

/// Iterator over DAWG terms.
struct DawgIterator<'a, V: DictionaryValue> {
    dawg: &'a DynamicDawgU64<V>,
    stack: Vec<(Arc<DawgNodeU64<V>>, Vec<u64>, usize)>,
}

impl<'a, V: DictionaryValue> DawgIterator<'a, V> {
    fn new(dawg: &'a DynamicDawgU64<V>) -> Self {
        DawgIterator {
            dawg,
            stack: vec![(dawg.root.clone(), Vec::new(), 0)],
        }
    }
}

impl<V: DictionaryValue> Iterator for DawgIterator<'_, V> {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, path, edge_idx)) = self.stack.pop() {
            let edges = node.edges.load();

            // Visit children
            if edge_idx < edges.edges.len() {
                let (label, child) = &edges.edges[edge_idx];
                let mut new_path = path.clone();
                new_path.push(*label);

                // Push current node back with next edge index
                self.stack.push((node.clone(), path, edge_idx + 1));
                // Push child to visit
                self.stack.push((child.clone(), new_path, 0));
            } else if node.is_final.load(Ordering::Acquire) {
                // All children visited, and this is a final node - return the path
                if !path.is_empty() || edge_idx == 0 {
                    return Some(path);
                }
            }
        }
        None
    }
}

/// Iterator over DAWG terms with values.
struct DawgIteratorWithValues<'a, V: DictionaryValue> {
    dawg: &'a DynamicDawgU64<V>,
    stack: Vec<(Arc<DawgNodeU64<V>>, Vec<u64>, usize)>,
}

impl<'a, V: DictionaryValue> DawgIteratorWithValues<'a, V> {
    fn new(dawg: &'a DynamicDawgU64<V>) -> Self {
        DawgIteratorWithValues {
            dawg,
            stack: vec![(dawg.root.clone(), Vec::new(), 0)],
        }
    }
}

impl<V: DictionaryValue> Iterator for DawgIteratorWithValues<'_, V> {
    type Item = (Vec<u64>, V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, path, edge_idx)) = self.stack.pop() {
            let edges = node.edges.load();

            if edge_idx < edges.edges.len() {
                let (label, child) = &edges.edges[edge_idx];
                let mut new_path = path.clone();
                new_path.push(*label);

                self.stack.push((node.clone(), path, edge_idx + 1));
                self.stack.push((child.clone(), new_path, 0));
            } else if node.is_final.load(Ordering::Acquire) {
                let val = node.value.load();
                // val is Guard<Option<Arc<V>>>; *val is Option<Arc<V>>
                if let Some(v) = &*val {
                    // v is &Arc<V>, **v is V
                    return Some((path, (**v).clone()));
                }
            }
        }
        None
    }
}

// =========================================================================
// Dictionary trait implementation
// =========================================================================

/// Node wrapper for Dictionary trait.
pub struct DynamicDawgU64Node<V: DictionaryValue> {
    node: Arc<DawgNodeU64<V>>,
}

impl<V: DictionaryValue> Clone for DynamicDawgU64Node<V> {
    fn clone(&self) -> Self {
        DynamicDawgU64Node {
            node: self.node.clone(),
        }
    }
}

impl<V: DictionaryValue> DictionaryNode for DynamicDawgU64Node<V> {
    type Unit = u64;

    fn is_final(&self) -> bool {
        self.node.is_final.load(Ordering::Acquire)
    }

    fn transition(&self, label: Self::Unit) -> Option<Self> {
        let edges = self.node.edges.load();
        edges.find(label).map(|child| DynamicDawgU64Node {
            node: child.clone(),
        })
    }

    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        let edges_guard = self.node.edges.load();
        let edges_vec: Vec<_> = edges_guard
            .edges
            .iter()
            .map(|(label, child)| {
                (
                    *label,
                    DynamicDawgU64Node {
                        node: child.clone(),
                    },
                )
            })
            .collect();
        Box::new(edges_vec.into_iter())
    }

    fn edge_count(&self) -> Option<usize> {
        Some(self.node.edges.load().edges.len())
    }
}

impl<V: DictionaryValue> Dictionary for DynamicDawgU64<V> {
    type Node = DynamicDawgU64Node<V>;

    fn root(&self) -> Self::Node {
        DynamicDawgU64Node {
            node: self.root.clone(),
        }
    }

    fn len(&self) -> Option<usize> {
        Some(self.term_count())
    }

    fn is_empty(&self) -> bool {
        self.term_count() == 0
    }

    fn sync_strategy(&self) -> SyncStrategy {
        SyncStrategy::InternalSync
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_dawg_is_empty() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        assert_eq!(dawg.term_count(), 0);
        assert!(!dawg.needs_compaction());
    }

    #[test]
    fn test_insert_sequence() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();

        assert!(dawg.insert_sequence(&[1, 2, 3]));
        assert!(!dawg.insert_sequence(&[1, 2, 3])); // Duplicate
        assert!(dawg.insert_sequence(&[1, 2, 4]));
        assert!(dawg.insert_sequence(&[5, 6, 7]));

        assert_eq!(dawg.term_count(), 3);
    }

    #[test]
    fn test_contains_sequence() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        dawg.insert_sequence(&[1, 2, 3]);
        dawg.insert_sequence(&[1, 2, 4]);

        assert!(dawg.contains_sequence(&[1, 2, 3]));
        assert!(dawg.contains_sequence(&[1, 2, 4]));
        assert!(!dawg.contains_sequence(&[1, 2])); // Prefix only
        assert!(!dawg.contains_sequence(&[1, 2, 5])); // Doesn't exist
        assert!(!dawg.contains_sequence(&[9, 9, 9]));
    }

    #[test]
    fn test_empty_sequence() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();

        assert!(dawg.insert_sequence(&[]));
        assert!(!dawg.insert_sequence(&[])); // Duplicate
        assert!(dawg.contains_sequence(&[]));
        assert_eq!(dawg.term_count(), 1);
    }

    #[test]
    fn test_remove_sequence() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        dawg.insert_sequence(&[1, 2, 3]);
        dawg.insert_sequence(&[1, 2, 4]);

        assert!(dawg.remove_sequence(&[1, 2, 3]));
        assert!(!dawg.contains_sequence(&[1, 2, 3]));
        assert!(dawg.contains_sequence(&[1, 2, 4])); // Other term still exists
        assert_eq!(dawg.term_count(), 1);

        assert!(!dawg.remove_sequence(&[1, 2, 3])); // Already removed
    }

    #[test]
    fn test_f64_api() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();

        assert!(dawg.insert_f64(&[1.0, 2.0, 3.0]));
        assert!(dawg.insert_f64(&[1.0, 2.0, 4.0]));
        assert!(!dawg.insert_f64(&[1.0, 2.0, 3.0])); // Duplicate

        assert!(dawg.contains_f64(&[1.0, 2.0, 3.0]));
        assert!(dawg.contains_f64(&[1.0, 2.0, 4.0]));
        assert!(!dawg.contains_f64(&[1.0, 2.0, 5.0]));
    }

    #[test]
    fn test_f64_edge_cases() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();

        // Test special float values
        dawg.insert_f64(&[0.0, f64::INFINITY, f64::NEG_INFINITY]);
        dawg.insert_f64(&[-0.0]); // Different bit pattern from +0.0

        assert!(dawg.contains_f64(&[0.0, f64::INFINITY, f64::NEG_INFINITY]));
        assert!(dawg.contains_f64(&[-0.0]));

        // NaN requires bit-pattern comparison
        let nan_bits = f64::NAN.to_bits();
        dawg.insert_sequence(&[nan_bits]);
        assert!(dawg.contains_sequence(&[nan_bits]));
    }

    #[test]
    fn test_valued_dawg() {
        let dawg: DynamicDawgU64<u32> = DynamicDawgU64::new();

        assert!(dawg.insert_sequence_with_value(&[1, 2, 3], 42));
        assert!(!dawg.insert_sequence_with_value(&[1, 2, 3], 99)); // Updates value

        assert_eq!(dawg.get_sequence_value(&[1, 2, 3]), Some(99));
        assert_eq!(dawg.get_sequence_value(&[1, 2, 4]), None);
    }

    #[test]
    fn test_string_api() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();

        assert!(dawg.insert("hello"));
        assert!(!dawg.insert("hello")); // Duplicate
        assert!(dawg.insert("world"));

        assert!(dawg.contains("hello"));
        assert!(dawg.contains("world"));
        assert!(!dawg.contains("foo"));
    }

    #[test]
    fn test_clone() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        dawg.insert_sequence(&[1, 2, 3]);
        dawg.insert_sequence(&[4, 5, 6]);

        let cloned = dawg.clone();
        assert_eq!(cloned.term_count(), dawg.term_count());
        assert!(cloned.contains_sequence(&[1, 2, 3]));
        assert!(cloned.contains_sequence(&[4, 5, 6]));

        // Modifications to clone don't affect original
        cloned.insert_sequence(&[7, 8, 9]);
        assert!(cloned.contains_sequence(&[7, 8, 9]));
        assert!(!dawg.contains_sequence(&[7, 8, 9]));
    }

    #[test]
    fn test_concurrent_inserts() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let dawg = StdArc::new(DynamicDawgU64::<()>::new());
        let num_threads = 8;
        let sequences_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let d = dawg.clone();
                thread::spawn(move || {
                    for i in 0..sequences_per_thread {
                        let seq = vec![t as u64, i as u64];
                        d.insert_sequence(&seq);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All sequences should be present
        assert_eq!(
            dawg.term_count(),
            num_threads * sequences_per_thread
        );

        for t in 0..num_threads {
            for i in 0..sequences_per_thread {
                assert!(dawg.contains_sequence(&[t as u64, i as u64]));
            }
        }
    }

    #[test]
    fn test_concurrent_reads_and_writes() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let dawg = StdArc::new(DynamicDawgU64::<()>::new());

        // Pre-populate with some data
        for i in 0..100 {
            dawg.insert_sequence(&[i, i + 1, i + 2]);
        }

        let handles: Vec<_> = (0..10)
            .map(|t| {
                let d = dawg.clone();
                thread::spawn(move || {
                    if t % 2 == 0 {
                        // Writer
                        for i in 100 + t * 10..100 + (t + 1) * 10 {
                            d.insert_sequence(&[i as u64, i as u64 + 1]);
                        }
                    } else {
                        // Reader
                        for i in 0..100 {
                            let _ = d.contains_sequence(&[i, i + 1, i + 2]);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Original data should still be there
        for i in 0..100u64 {
            assert!(dawg.contains_sequence(&[i, i + 1, i + 2]));
        }
    }

    #[test]
    fn test_dictionary_trait() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        dawg.insert_sequence(&[1, 2, 3]);

        let root = dawg.root();
        assert!(!root.is_final());

        let n1 = root.transition(1).expect("Should have transition");
        assert!(!n1.is_final());

        let n2 = n1.transition(2).expect("Should have transition");
        assert!(!n2.is_final());

        let n3 = n2.transition(3).expect("Should have transition");
        assert!(n3.is_final());

        assert!(n2.transition(9).is_none());
    }

    #[test]
    fn test_from_terms() {
        let terms = vec!["apple", "banana", "cherry"];
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::from_terms(terms);

        assert!(dawg.contains("apple"));
        assert!(dawg.contains("banana"));
        assert!(dawg.contains("cherry"));
        assert_eq!(dawg.term_count(), 3);
    }

    #[test]
    fn test_extend() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        dawg.insert("existing");

        let added = dawg.extend(vec!["new1", "new2", "existing"]);
        assert_eq!(added, 2); // Only 2 new terms
        assert_eq!(dawg.term_count(), 3);
    }

    #[test]
    fn test_node_count() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        assert_eq!(dawg.node_count(), 1); // Just root

        dawg.insert_sequence(&[1, 2, 3]);
        assert_eq!(dawg.node_count(), 4); // root + 3 nodes

        dawg.insert_sequence(&[1, 2, 4]);
        assert_eq!(dawg.node_count(), 5); // Shares [1, 2] prefix
    }

    #[test]
    fn test_prefix_sharing() {
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();

        dawg.insert_sequence(&[1, 2, 3]);
        dawg.insert_sequence(&[1, 2, 4]);
        dawg.insert_sequence(&[1, 2, 5]);

        // All share the [1, 2] prefix
        // root -> 1 -> 2 -> {3, 4, 5}
        // That's 1 (root) + 1 (node 1) + 1 (node 2) + 3 (nodes 3,4,5) = 6 nodes
        assert_eq!(dawg.node_count(), 6);
    }

    // ==================== Concurrency Stress Tests ====================
    // These tests verify the lock-free implementation under heavy concurrent load

    #[test]
    fn test_stress_100_concurrent_readers() {
        use std::sync::Arc as StdArc;
        use std::thread;

        // Pre-populate the DAWG
        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        for i in 0u64..1000 {
            dawg.insert_sequence(&[i, i + 1, i + 2]);
        }
        let dawg = StdArc::new(dawg);

        // Spawn 100 concurrent readers
        let handles: Vec<_> = (0..100)
            .map(|reader_id| {
                let dawg = StdArc::clone(&dawg);
                thread::spawn(move || {
                    // Each reader does 1000 lookups
                    for i in 0u64..1000 {
                        let seq = [i, i + 1, i + 2];
                        let found = dawg.contains_sequence(&seq);
                        assert!(found, "Reader {reader_id} failed to find sequence {i}");
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Reader thread panicked");
        }
    }

    #[test]
    fn test_stress_readers_and_writers() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        // Pre-populate with some data
        for i in 0u64..100 {
            dawg.insert_sequence(&[i, i + 1]);
        }
        let dawg = StdArc::new(dawg);
        let stop = StdArc::new(AtomicBool::new(false));

        // 10 reader threads
        let reader_handles: Vec<_> = (0..10)
            .map(|_| {
                let dawg = StdArc::clone(&dawg);
                let stop = StdArc::clone(&stop);
                thread::spawn(move || {
                    let mut reads = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        // Read pre-existing keys
                        for i in 0u64..100 {
                            let _ = dawg.contains_sequence(&[i, i + 1]);
                            reads += 1;
                        }
                    }
                    reads
                })
            })
            .collect();

        // 10 writer threads
        let writer_handles: Vec<_> = (0..10)
            .map(|writer_id| {
                let dawg = StdArc::clone(&dawg);
                thread::spawn(move || {
                    // Each writer inserts 100 sequences in its own range
                    let base = 1000 + (writer_id as u64 * 100);
                    for i in 0u64..100 {
                        dawg.insert_sequence(&[base + i, base + i + 1, base + i + 2]);
                    }
                })
            })
            .collect();

        // Wait for writers to complete
        for handle in writer_handles {
            handle.join().expect("Writer thread panicked");
        }

        // Signal readers to stop
        stop.store(true, Ordering::Relaxed);

        // Wait for readers
        let total_reads: u64 = reader_handles
            .into_iter()
            .map(|h| h.join().expect("Reader thread panicked"))
            .sum();

        // Verify all inserted sequences exist
        for writer_id in 0..10 {
            let base = 1000 + (writer_id as u64 * 100);
            for i in 0u64..100 {
                assert!(
                    dawg.contains_sequence(&[base + i, base + i + 1, base + i + 2]),
                    "Missing sequence from writer {writer_id} at offset {i}"
                );
            }
        }

        // Should have done many reads while writes were happening
        assert!(total_reads > 1000, "Expected many reads, got {total_reads}");
    }

    #[test]
    fn test_stress_50_writers_same_keys() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        let dawg = StdArc::new(dawg);
        let successful_inserts = StdArc::new(AtomicUsize::new(0));

        // 50 writers all trying to insert the same 100 sequences
        let handles: Vec<_> = (0..50)
            .map(|_| {
                let dawg = StdArc::clone(&dawg);
                let successful_inserts = StdArc::clone(&successful_inserts);
                thread::spawn(move || {
                    for i in 0u64..100 {
                        if dawg.insert_sequence(&[i, i + 1, i + 2]) {
                            successful_inserts.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Writer thread panicked");
        }

        // Exactly 100 unique sequences should exist
        assert_eq!(dawg.term_count(), 100);

        // Exactly 100 successful inserts (one per unique sequence)
        assert_eq!(successful_inserts.load(Ordering::Relaxed), 100);

        // Verify all sequences exist
        for i in 0u64..100 {
            assert!(dawg.contains_sequence(&[i, i + 1, i + 2]));
        }
    }

    #[test]
    fn test_stress_50_writers_disjoint_keys() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        let dawg = StdArc::new(dawg);

        // 50 writers, each inserting 100 unique sequences in disjoint ranges
        let handles: Vec<_> = (0..50)
            .map(|writer_id| {
                let dawg = StdArc::clone(&dawg);
                thread::spawn(move || {
                    let base = writer_id as u64 * 1000;
                    for i in 0u64..100 {
                        let inserted = dawg.insert_sequence(&[base + i, base + i + 1]);
                        assert!(inserted, "Writer {writer_id} failed to insert unique seq {i}");
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Writer thread panicked");
        }

        // 50 writers × 100 sequences = 5000 total
        assert_eq!(dawg.term_count(), 5000);

        // Verify all sequences exist
        for writer_id in 0u64..50 {
            let base = writer_id * 1000;
            for i in 0u64..100 {
                assert!(
                    dawg.contains_sequence(&[base + i, base + i + 1]),
                    "Missing sequence from writer {writer_id} at offset {i}"
                );
            }
        }
    }

    #[test]
    fn test_stress_valued_concurrent_writes() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let dawg: DynamicDawgU64<u64> = DynamicDawgU64::new();
        let dawg = StdArc::new(dawg);

        // 20 writers, each inserting sequences with values
        let handles: Vec<_> = (0..20)
            .map(|writer_id| {
                let dawg = StdArc::clone(&dawg);
                thread::spawn(move || {
                    let base = writer_id as u64 * 100;
                    for i in 0u64..50 {
                        let value = base + i;
                        dawg.insert_sequence_with_value(&[base + i], value);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Writer thread panicked");
        }

        // 20 writers × 50 sequences = 1000 total
        assert_eq!(dawg.term_count(), 1000);

        // Verify values are correct
        for writer_id in 0u64..20 {
            let base = writer_id * 100;
            for i in 0u64..50 {
                let expected_value = base + i;
                let value = dawg.get_sequence_value(&[base + i]);
                assert_eq!(
                    value,
                    Some(expected_value),
                    "Wrong value for sequence [{base} + {i}]"
                );
            }
        }
    }

    #[test]
    fn test_stress_remove_while_reading() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        // Insert 1000 sequences
        for i in 0u64..1000 {
            dawg.insert_sequence(&[i, i + 1]);
        }
        let dawg = StdArc::new(dawg);
        let stop = StdArc::new(AtomicBool::new(false));

        // 5 reader threads
        let reader_handles: Vec<_> = (0..5)
            .map(|_| {
                let dawg = StdArc::clone(&dawg);
                let stop = StdArc::clone(&stop);
                thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        // Read random sequences - some may have been removed
                        for i in 0u64..1000 {
                            let _ = dawg.contains_sequence(&[i, i + 1]);
                        }
                    }
                })
            })
            .collect();

        // 5 remover threads
        let remover_handles: Vec<_> = (0..5)
            .map(|remover_id| {
                let dawg = StdArc::clone(&dawg);
                thread::spawn(move || {
                    // Each remover removes 200 sequences in its range
                    let base = remover_id as u64 * 200;
                    for i in 0u64..200 {
                        dawg.remove_sequence(&[base + i, base + i + 1]);
                    }
                })
            })
            .collect();

        // Wait for removers
        for handle in remover_handles {
            handle.join().expect("Remover thread panicked");
        }

        // Signal readers to stop
        stop.store(true, Ordering::Relaxed);

        // Wait for readers
        for handle in reader_handles {
            handle.join().expect("Reader thread panicked");
        }

        // After 5 removers × 200 = 1000 removals from 1000 sequences
        assert_eq!(dawg.term_count(), 0);
    }

    #[test]
    fn test_stress_iterator_during_writes() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let dawg: DynamicDawgU64<()> = DynamicDawgU64::new();
        // Pre-populate
        for i in 0u64..100 {
            dawg.insert_sequence(&[i]);
        }
        let dawg = StdArc::new(dawg);
        let stop = StdArc::new(AtomicBool::new(false));

        // Iterator thread - iterates while writes happen
        let iter_dawg = StdArc::clone(&dawg);
        let iter_stop = StdArc::clone(&stop);
        let iter_handle = thread::spawn(move || {
            let mut iteration_count = 0;
            while !iter_stop.load(Ordering::Relaxed) {
                // Collect all terms (snapshot at time of iteration start)
                let terms: Vec<_> = iter_dawg.iter().collect();
                // Should have at least the initial 100
                assert!(terms.len() >= 100);
                iteration_count += 1;
            }
            iteration_count
        });

        // Writer threads add more sequences
        let writer_handles: Vec<_> = (0..10)
            .map(|writer_id| {
                let dawg = StdArc::clone(&dawg);
                thread::spawn(move || {
                    let base = 1000 + writer_id as u64 * 100;
                    for i in 0u64..100 {
                        dawg.insert_sequence(&[base + i]);
                    }
                })
            })
            .collect();

        // Wait for writers
        for handle in writer_handles {
            handle.join().expect("Writer thread panicked");
        }

        // Signal iterator to stop
        stop.store(true, Ordering::Relaxed);

        let iterations = iter_handle.join().expect("Iterator thread panicked");
        assert!(iterations > 0, "Iterator thread should have run at least once");

        // Final count: 100 initial + 10 writers × 100 = 1100
        assert_eq!(dawg.term_count(), 1100);
    }
}
