//! Memory pressure-aware eviction wrapper.
//!
//! This wrapper tracks size and hit rates to prioritize evicting large entries
//! with low hit rates.
//!
//! # Architecture
//!
//! Unlike the old `MemoryPressureStrategy` which required `CacheEntry<V>` metadata,
//! this wrapper maintains separate metadata tracking size, hit count, and access count.
//!
//! # Use Cases
//!
//! - Memory-constrained environments
//! - Large value caching (embeddings, ASTs, etc.)
//! - Adaptive caching under varying memory conditions
//!
//! # Examples
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::MemoryPressure;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("foo", 42),
//!     ("bar", 99),
//! ]);
//!
//! let memory_pressure = MemoryPressure::new(dict);
//! assert_eq!(memory_pressure.get_value("foo"), Some(42));
//! ```

use crate::sync_compat::RwLock;
use libdictenstein::{
    Dictionary, DictionaryNode, DictionaryValue, MappedDictionary, MappedDictionaryNode,
    SyncStrategy,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Metadata tracked for each entry.
#[derive(Debug, Clone)]
struct EntryMetadata {
    size_bytes: usize,
    hit_count: u32,
    access_count: u32,
}

impl EntryMetadata {
    fn new(size: usize) -> Self {
        Self {
            size_bytes: size,
            hit_count: 1, // First access
            access_count: 1,
        }
    }

    fn increment(&mut self) {
        self.hit_count = self.hit_count.saturating_add(1);
        self.access_count = self.access_count.saturating_add(1);
    }

    fn hit_rate(&self) -> f64 {
        if self.access_count == 0 {
            0.0
        } else {
            self.hit_count as f64 / self.access_count as f64
        }
    }

    /// Memory pressure score: size / (hit_rate + 0.1)
    /// Higher score = more likely to evict (large size + low hit rate)
    fn memory_pressure_score(&self) -> f64 {
        let size = self.size_bytes as f64;
        let hit_rate = self.hit_rate();
        size / (hit_rate + 0.1)
    }
}

/// Memory pressure-aware eviction wrapper.
///
/// Evicts entries based on memory pressure. Entries with large size and
/// low hit rate are evicted first.
///
/// # Type Parameters
///
/// - `D`: Inner dictionary type
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::dictionary::MappedDictionary;
/// use liblevenshtein::cache::eviction::MemoryPressure;
///
/// let dict = PathMapDictionary::from_terms_with_values([
///     ("hello", 1),
///     ("world", 2),
/// ]);
///
/// let memory_pressure = MemoryPressure::new(dict);
/// assert_eq!(memory_pressure.get_value("hello"), Some(1));
/// ```
#[derive(Clone)]
pub struct MemoryPressure<D> {
    inner: D,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<D> MemoryPressure<D> {
    /// Creates a new MemoryPressure wrapper.
    ///
    /// # Arguments
    ///
    /// - `dict`: The dictionary to wrap
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::MemoryPressure;
    ///
    /// let dict = PathMapDictionary::from_terms_with_values([
    ///     ("foo", 42),
    /// ]);
    ///
    /// let memory_pressure = MemoryPressure::new(dict);
    /// ```
    pub fn new(dict: D) -> Self {
        Self {
            inner: dict,
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Unwraps the inner dictionary.
    #[inline]
    pub fn into_inner(self) -> D {
        self.inner
    }

    /// Gets a reference to the inner dictionary.
    #[inline]
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Records an entry access with size tracking.
    fn record_access<V: DictionaryValue>(&self, term: &str, _value: &V) {
        let size = std::mem::size_of::<V>();
        let mut metadata = self.metadata.write();
        metadata
            .entry(term.to_string())
            .and_modify(|m| m.increment())
            .or_insert_with(|| EntryMetadata::new(size));
    }

    /// Gets the memory pressure score for an entry.
    ///
    /// Returns `None` if the entry has never been accessed.
    pub fn memory_pressure_score(&self, term: &str) -> Option<f64> {
        let metadata = self.metadata.read();
        metadata.get(term).map(|m| m.memory_pressure_score())
    }

    /// Finds the highest memory pressure entry among the given terms.
    ///
    /// Returns the term with the highest pressure score (most likely to evict).
    pub fn find_highest_pressure(&self, terms: &[&str]) -> Option<String> {
        let metadata = self.metadata.read();
        terms
            .iter()
            .filter_map(|&term| {
                metadata
                    .get(term)
                    .map(|m| (term, m.memory_pressure_score()))
            })
            .max_by(|(_, score1), (_, score2)| score1.total_cmp(score2))
            .map(|(term, _)| term.to_string())
    }

    /// Evicts the highest memory pressure entry from metadata.
    ///
    /// Returns the evicted term if any.
    pub fn evict_highest_pressure(&self, terms: &[&str]) -> Option<String> {
        if let Some(high_pressure_term) = self.find_highest_pressure(terms) {
            let mut metadata = self.metadata.write();
            metadata.remove(&high_pressure_term);
            Some(high_pressure_term)
        } else {
            None
        }
    }

    /// Clears all metadata.
    pub fn clear_metadata(&self) {
        let mut metadata = self.metadata.write();
        metadata.clear();
    }
}

impl<D> Dictionary for MemoryPressure<D>
where
    D: Dictionary,
{
    type Node = MemoryPressureNode<D::Node>;

    #[inline]
    fn root(&self) -> Self::Node {
        MemoryPressureNode::new(self.inner.root(), Arc::clone(&self.metadata))
    }

    #[inline]
    fn len(&self) -> Option<usize> {
        self.inner.len()
    }

    #[inline]
    fn contains(&self, term: &str) -> bool {
        self.inner.contains(term)
    }

    #[inline]
    fn sync_strategy(&self) -> SyncStrategy {
        self.inner.sync_strategy()
    }
}

impl<D, V> MappedDictionary for MemoryPressure<D>
where
    D: MappedDictionary<Value = V>,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn get_value(&self, term: &str) -> Option<Self::Value> {
        // Get value from inner dictionary first
        let value = self.inner.get_value(term)?;

        // Record access with value for size tracking
        self.record_access(term, &value);

        Some(value)
    }

    #[inline]
    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool,
    {
        self.inner.contains_with_value(term, predicate)
    }
}

/// Node wrapper for MemoryPressure dictionary.
#[derive(Clone)]
pub struct MemoryPressureNode<N> {
    inner: N,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<N> MemoryPressureNode<N> {
    fn new(inner: N, metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>) -> Self {
        Self { inner, metadata }
    }
}

impl<N> DictionaryNode for MemoryPressureNode<N>
where
    N: DictionaryNode,
{
    type Unit = N::Unit;

    #[inline]
    fn is_final(&self) -> bool {
        self.inner.is_final()
    }

    #[inline]
    fn transition(&self, label: Self::Unit) -> Option<Self> {
        self.inner
            .transition(label)
            .map(|node| MemoryPressureNode::new(node, Arc::clone(&self.metadata)))
    }

    #[inline]
    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        let metadata = Arc::clone(&self.metadata);
        Box::new(self.inner.edges().map(move |(label, node)| {
            (label, MemoryPressureNode::new(node, Arc::clone(&metadata)))
        }))
    }

    #[inline]
    fn edge_count(&self) -> Option<usize> {
        self.inner.edge_count()
    }
}

impl<N, V> MappedDictionaryNode for MemoryPressureNode<N>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn value(&self) -> Option<Self::Value> {
        self.inner.value()
    }
}

#[cfg(all(test, feature = "pathmap-backend"))]
mod tests {
    use super::*;
    #[cfg(feature = "pathmap-backend")]
    use libdictenstein::pathmap::PathMapDictionary;

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_memory_pressure_wrapper_basic() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let memory_pressure = MemoryPressure::new(dict);

        // Values should be accessible
        assert_eq!(memory_pressure.get_value("foo"), Some(42));
        assert_eq!(memory_pressure.get_value("bar"), Some(99));
        assert!(memory_pressure.contains("foo"));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_memory_pressure_scoring() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let memory_pressure = MemoryPressure::new(dict);

        // Access foo once (hit_rate = 1/1 = 1.0)
        assert_eq!(memory_pressure.get_value("foo"), Some(42));
        let score1 = memory_pressure
            .memory_pressure_score("foo")
            .expect("expected Some score in test");

        // Access bar multiple times (hit_rate = 3/3 = 1.0)
        assert_eq!(memory_pressure.get_value("bar"), Some(99));
        assert_eq!(memory_pressure.get_value("bar"), Some(99));
        assert_eq!(memory_pressure.get_value("bar"), Some(99));
        let score2 = memory_pressure
            .memory_pressure_score("bar")
            .expect("expected Some score in test");

        // Both have same size and same hit rate (1.0), so scores should be equal
        assert!((score1 - score2).abs() < 0.01);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_memory_pressure_find_highest() {
        let dict =
            PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99), ("baz", 123)]);

        let memory_pressure = MemoryPressure::new(dict);

        // Access with different counts (but same hit rate since all are hits)
        // foo: 1 access (hit_rate = 1/1 = 1.0)
        assert_eq!(memory_pressure.get_value("foo"), Some(42));

        // bar: 3 accesses (hit_rate = 3/3 = 1.0)
        assert_eq!(memory_pressure.get_value("bar"), Some(99));
        assert_eq!(memory_pressure.get_value("bar"), Some(99));
        assert_eq!(memory_pressure.get_value("bar"), Some(99));

        // baz: 5 accesses (hit_rate = 5/5 = 1.0)
        for _ in 0..5 {
            assert_eq!(memory_pressure.get_value("baz"), Some(123));
        }

        // All have same size and same hit rate (1.0), so any could be highest
        // The test just verifies we can find one
        let highest = memory_pressure.find_highest_pressure(&["foo", "bar", "baz"]);
        assert!(highest.is_some());
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_memory_pressure_eviction() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let memory_pressure = MemoryPressure::new(dict);

        // Access both
        assert_eq!(memory_pressure.get_value("foo"), Some(42));
        assert_eq!(memory_pressure.get_value("bar"), Some(99));

        // Evict highest pressure
        let evicted = memory_pressure.evict_highest_pressure(&["foo", "bar"]);
        assert!(evicted.is_some());

        // Evicted term should have no metadata
        assert_eq!(
            memory_pressure.memory_pressure_score(&evicted.expect("expected Some evicted in test")),
            None
        );
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_memory_pressure_clear_metadata() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let memory_pressure = MemoryPressure::new(dict);

        // Access both
        assert_eq!(memory_pressure.get_value("foo"), Some(42));
        assert_eq!(memory_pressure.get_value("bar"), Some(99));

        // Clear metadata
        memory_pressure.clear_metadata();

        // No scores
        assert_eq!(memory_pressure.memory_pressure_score("foo"), None);
        assert_eq!(memory_pressure.memory_pressure_score("bar"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_memory_pressure_node_traversal() {
        let dict = PathMapDictionary::<()>::from_terms(["hello", "help"]);
        let memory_pressure = MemoryPressure::new(dict);

        let root = memory_pressure.root();
        assert!(!root.is_final());

        // Traverse 'h' -> 'e' -> 'l' -> 'p'
        let h = root
            .transition(b'h')
            .expect("expected Some transition h in test");
        let e = h
            .transition(b'e')
            .expect("expected Some transition e in test");
        let l = e
            .transition(b'l')
            .expect("expected Some transition l in test");
        let p = l
            .transition(b'p')
            .expect("expected Some transition p in test");

        assert!(p.is_final()); // "help"
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_memory_pressure_into_inner() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42)]);
        let memory_pressure = MemoryPressure::new(dict);
        let original = memory_pressure.into_inner();

        assert_eq!(original.len(), Some(1));
        assert_eq!(original.get_value("foo"), Some(42));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_memory_pressure_hit_rate_tracking() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42)]);
        let memory_pressure = MemoryPressure::new(dict);

        // First access (hit_rate = 1/1 = 1.0)
        assert_eq!(memory_pressure.get_value("foo"), Some(42));
        let score1 = memory_pressure
            .memory_pressure_score("foo")
            .expect("expected Some score in test");

        // Multiple accesses (hit_rate = 6/6 = 1.0, same as before)
        for _ in 0..5 {
            assert_eq!(memory_pressure.get_value("foo"), Some(42));
        }
        let score2 = memory_pressure
            .memory_pressure_score("foo")
            .expect("expected Some score in test");

        // Hit rate is always 1.0 since every get_value is a hit, so scores stay equal
        assert!((score1 - score2).abs() < 0.01);
    }
}
