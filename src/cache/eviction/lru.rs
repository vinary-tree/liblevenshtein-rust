//! LRU (Least Recently Used) eviction wrapper.
//!
//! This wrapper tracks access times for entries and can evict least recently
//! used entries. It maintains metadata for tracking last access time.
//!
//! # Architecture
//!
//! Unlike the old `LruStrategy` which required `CacheEntry<V>` metadata,
//! this wrapper maintains a separate metadata map tracking access times.
//!
//! # Use Cases
//!
//! - Code completion: Favor recently used identifiers
//! - General-purpose caching with access patterns
//! - Memory-efficient caching (evict cold entries)
//!
//! # Examples
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::Lru;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("foo", 42),
//!     ("bar", 99),
//! ]);
//!
//! let lru = Lru::new(dict);
//! assert_eq!(lru.get_value("foo"), Some(42));
//! ```

use libdictenstein::{
    Dictionary, DictionaryNode, DictionaryValue, MappedDictionary, MappedDictionaryNode,
    SyncStrategy,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Metadata tracked for each entry.
#[derive(Debug, Clone)]
struct EntryMetadata {
    last_accessed: Instant,
}

impl EntryMetadata {
    fn new() -> Self {
        Self {
            last_accessed: Instant::now(),
        }
    }

    fn update_access(&mut self) {
        self.last_accessed = Instant::now();
    }

    fn recency(&self) -> std::time::Duration {
        self.last_accessed.elapsed()
    }
}

/// LRU (Least Recently Used) eviction wrapper.
///
/// Tracks access times for entries. Provides methods to identify least recently
/// used entries for eviction.
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
/// use liblevenshtein::cache::eviction::Lru;
///
/// let dict = PathMapDictionary::from_terms_with_values([
///     ("hello", 1),
///     ("world", 2),
/// ]);
///
/// let lru = Lru::new(dict);
/// assert_eq!(lru.get_value("hello"), Some(1));
/// ```
#[derive(Clone)]
pub struct Lru<D> {
    inner: D,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<D> Lru<D> {
    /// Creates a new LRU wrapper.
    ///
    /// # Arguments
    ///
    /// - `dict`: The dictionary to wrap
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::Lru;
    ///
    /// let dict = PathMapDictionary::from_terms_with_values([
    ///     ("foo", 42),
    /// ]);
    ///
    /// let lru = Lru::new(dict);
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

    /// Records an entry access (updates metadata).
    fn record_access(&self, term: &str) {
        let mut metadata = self
            .metadata
            .write()
            .expect("poisoned RwLock; only fatal if writer panicked");
        metadata
            .entry(term.to_string())
            .and_modify(|m| m.update_access())
            .or_insert_with(EntryMetadata::new);
    }

    /// Gets the recency score for an entry (time since last access).
    ///
    /// Returns `None` if the entry has never been accessed.
    pub fn recency(&self, term: &str) -> Option<std::time::Duration> {
        let metadata = self
            .metadata
            .read()
            .expect("poisoned RwLock; only fatal if writer panicked");
        metadata.get(term).map(|m| m.recency())
    }

    /// Finds the least recently used entry among the given terms.
    ///
    /// Returns the term with the longest time since last access.
    pub fn find_lru(&self, terms: &[&str]) -> Option<String> {
        let metadata = self
            .metadata
            .read()
            .expect("poisoned RwLock; only fatal if writer panicked");
        terms
            .iter()
            .filter_map(|&term| metadata.get(term).map(|m| (term, m.recency())))
            .max_by_key(|(_, recency)| *recency)
            .map(|(term, _)| term.to_string())
    }

    /// Evicts the least recently used entry from metadata.
    ///
    /// Returns the evicted term if any.
    pub fn evict_lru(&self, terms: &[&str]) -> Option<String> {
        if let Some(lru_term) = self.find_lru(terms) {
            let mut metadata = self
                .metadata
                .write()
                .expect("poisoned RwLock; only fatal if writer panicked");
            metadata.remove(&lru_term);
            Some(lru_term)
        } else {
            None
        }
    }

    /// Clears all metadata.
    pub fn clear_metadata(&self) {
        let mut metadata = self
            .metadata
            .write()
            .expect("poisoned RwLock; only fatal if writer panicked");
        metadata.clear();
    }
}

impl<D> Dictionary for Lru<D>
where
    D: Dictionary,
{
    type Node = LruNode<D::Node>;

    #[inline]
    fn root(&self) -> Self::Node {
        LruNode::new(self.inner.root(), Arc::clone(&self.metadata))
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

impl<D, V> MappedDictionary for Lru<D>
where
    D: MappedDictionary<Value = V>,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn get_value(&self, term: &str) -> Option<Self::Value> {
        // Record access
        self.record_access(term);

        // Get value from inner dictionary
        self.inner.get_value(term)
    }

    #[inline]
    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool,
    {
        self.inner.contains_with_value(term, predicate)
    }
}

/// Node wrapper for LRU dictionary.
#[derive(Clone)]
pub struct LruNode<N> {
    inner: N,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<N> LruNode<N> {
    fn new(inner: N, metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>) -> Self {
        Self { inner, metadata }
    }
}

impl<N> DictionaryNode for LruNode<N>
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
            .map(|node| LruNode::new(node, Arc::clone(&self.metadata)))
    }

    #[inline]
    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        let metadata = Arc::clone(&self.metadata);
        Box::new(
            self.inner
                .edges()
                .map(move |(label, node)| (label, LruNode::new(node, Arc::clone(&metadata)))),
        )
    }

    #[inline]
    fn edge_count(&self) -> Option<usize> {
        self.inner.edge_count()
    }
}

impl<N, V> MappedDictionaryNode for LruNode<N>
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
    use std::thread;
    use std::time::Duration;

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lru_wrapper_basic() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let lru = Lru::new(dict);

        // Values should be accessible
        assert_eq!(lru.get_value("foo"), Some(42));
        assert_eq!(lru.get_value("bar"), Some(99));
        assert!(lru.contains("foo"));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lru_access_tracking() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let lru = Lru::new(dict);

        // Access foo
        assert_eq!(lru.get_value("foo"), Some(42));

        // Wait a bit
        thread::sleep(Duration::from_millis(10));

        // Access bar
        assert_eq!(lru.get_value("bar"), Some(99));

        // foo should have higher recency (more time elapsed)
        let foo_recency = lru.recency("foo").expect("expected Some recency in test");
        let bar_recency = lru.recency("bar").expect("expected Some recency in test");

        assert!(foo_recency > bar_recency);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lru_find_lru() {
        let dict =
            PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99), ("baz", 123)]);

        let lru = Lru::new(dict);

        // Access in order: foo, bar, baz
        assert_eq!(lru.get_value("foo"), Some(42));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(lru.get_value("bar"), Some(99));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(lru.get_value("baz"), Some(123));

        // foo is least recently used
        let lru_term = lru.find_lru(&["foo", "bar", "baz"]);
        assert_eq!(lru_term, Some("foo".to_string()));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lru_eviction() {
        let dict =
            PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99), ("baz", 123)]);

        let lru = Lru::new(dict);

        // Access all entries
        assert_eq!(lru.get_value("foo"), Some(42));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(lru.get_value("bar"), Some(99));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(lru.get_value("baz"), Some(123));

        // Evict LRU
        let evicted = lru.evict_lru(&["foo", "bar", "baz"]);
        assert_eq!(evicted, Some("foo".to_string()));

        // foo should have no recency data now
        assert_eq!(lru.recency("foo"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lru_update_on_access() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let lru = Lru::new(dict);

        // Access foo
        assert_eq!(lru.get_value("foo"), Some(42));
        thread::sleep(Duration::from_millis(10));

        // Access bar
        assert_eq!(lru.get_value("bar"), Some(99));
        thread::sleep(Duration::from_millis(10));

        // foo is LRU
        assert_eq!(lru.find_lru(&["foo", "bar"]), Some("foo".to_string()));

        // Access foo again
        assert_eq!(lru.get_value("foo"), Some(42));

        // Now bar is LRU
        assert_eq!(lru.find_lru(&["foo", "bar"]), Some("bar".to_string()));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lru_clear_metadata() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let lru = Lru::new(dict);

        // Access both
        assert_eq!(lru.get_value("foo"), Some(42));
        assert_eq!(lru.get_value("bar"), Some(99));

        // Clear metadata
        lru.clear_metadata();

        // No recency data
        assert_eq!(lru.recency("foo"), None);
        assert_eq!(lru.recency("bar"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lru_node_traversal() {
        let dict = PathMapDictionary::<()>::from_terms(["hello", "help"]);
        let lru = Lru::new(dict);

        let root = lru.root();
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
    fn test_lru_into_inner() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42)]);
        let lru = Lru::new(dict);
        let original = lru.into_inner();

        assert_eq!(original.len(), Some(1));
        assert_eq!(original.get_value("foo"), Some(42));
    }
}
