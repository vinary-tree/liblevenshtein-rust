//! LFU (Least Frequently Used) eviction wrapper.
//!
//! This wrapper tracks access counts for entries to identify the least
//! frequently used entries for eviction.
//!
//! # Architecture
//!
//! Unlike the old `LfuStrategy` which required `CacheEntry<V>` metadata,
//! this wrapper maintains a separate metadata map tracking access counts.
//!
//! # Use Cases
//!
//! - Code completion: Favor frequently used identifiers
//! - Documentation search: Keep popular docs cached
//! - Long-lived caches with stable access patterns
//!
//! # Examples
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::cache::eviction::Lfu;
//! use liblevenshtein::dictionary::MappedDictionary;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("foo", 42),
//!     ("bar", 99),
//! ]);
//!
//! let lfu = Lfu::new(dict);
//! assert_eq!(lfu.get_value("foo"), Some(42));
//! ```

use libdictenstein::{
    Dictionary, DictionaryNode, DictionaryValue, MappedDictionary, MappedDictionaryNode,
    SyncStrategy,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Metadata tracked for each entry.
#[derive(Debug, Clone)]
struct EntryMetadata {
    access_count: u32,
}

impl EntryMetadata {
    fn new() -> Self {
        Self {
            access_count: 1, // First access
        }
    }

    fn increment(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
    }
}

/// LFU (Least Frequently Used) eviction wrapper.
///
/// Evicts entries with the lowest access frequency.
/// Tracks access count for each entry.
///
/// # Type Parameters
///
/// - `D`: Inner dictionary type
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::cache::eviction::Lfu;
/// use liblevenshtein::dictionary::MappedDictionary;
///
/// let dict = PathMapDictionary::from_terms_with_values([
///     ("hello", 1),
///     ("world", 2),
/// ]);
///
/// let lfu = Lfu::new(dict);
/// assert_eq!(lfu.get_value("hello"), Some(1));
/// ```
#[derive(Clone)]
pub struct Lfu<D> {
    inner: D,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<D> Lfu<D> {
    /// Creates a new LFU wrapper.
    ///
    /// # Arguments
    ///
    /// - `dict`: The dictionary to wrap
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::Lfu;
    ///
    /// let dict = PathMapDictionary::from_terms_with_values([
    ///     ("foo", 42),
    /// ]);
    ///
    /// let lfu = Lfu::new(dict);
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
            .and_modify(|m| m.increment())
            .or_insert_with(EntryMetadata::new);
    }

    /// Gets the access count for an entry.
    ///
    /// Returns `None` if the entry has never been accessed.
    pub fn access_count(&self, term: &str) -> Option<u32> {
        let metadata = self
            .metadata
            .read()
            .expect("poisoned RwLock; only fatal if writer panicked");
        metadata.get(term).map(|m| m.access_count)
    }

    /// Finds the least frequently used entry among the given terms.
    ///
    /// Returns the term with the lowest access count.
    pub fn find_lfu(&self, terms: &[&str]) -> Option<String> {
        let metadata = self
            .metadata
            .read()
            .expect("poisoned RwLock; only fatal if writer panicked");
        terms
            .iter()
            .filter_map(|&term| metadata.get(term).map(|m| (term, m.access_count)))
            .min_by_key(|(_, count)| *count)
            .map(|(term, _)| term.to_string())
    }

    /// Evicts the least frequently used entry from metadata.
    ///
    /// Returns the evicted term if any.
    pub fn evict_lfu(&self, terms: &[&str]) -> Option<String> {
        if let Some(lfu_term) = self.find_lfu(terms) {
            let mut metadata = self
                .metadata
                .write()
                .expect("poisoned RwLock; only fatal if writer panicked");
            metadata.remove(&lfu_term);
            Some(lfu_term)
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

impl<D> Dictionary for Lfu<D>
where
    D: Dictionary,
{
    type Node = LfuNode<D::Node>;

    #[inline]
    fn root(&self) -> Self::Node {
        LfuNode::new(self.inner.root(), Arc::clone(&self.metadata))
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

impl<D, V> MappedDictionary for Lfu<D>
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

/// Node wrapper for LFU dictionary.
#[derive(Clone)]
pub struct LfuNode<N> {
    inner: N,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<N> LfuNode<N> {
    fn new(inner: N, metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>) -> Self {
        Self { inner, metadata }
    }
}

impl<N> DictionaryNode for LfuNode<N>
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
            .map(|node| LfuNode::new(node, Arc::clone(&self.metadata)))
    }

    #[inline]
    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        let metadata = Arc::clone(&self.metadata);
        Box::new(
            self.inner
                .edges()
                .map(move |(label, node)| (label, LfuNode::new(node, Arc::clone(&metadata)))),
        )
    }

    #[inline]
    fn edge_count(&self) -> Option<usize> {
        self.inner.edge_count()
    }
}

impl<N, V> MappedDictionaryNode for LfuNode<N>
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "pathmap-backend")]
    use libdictenstein::pathmap::PathMapDictionary;

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lfu_wrapper_basic() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let lfu = Lfu::new(dict);

        // Values should be accessible
        assert_eq!(lfu.get_value("foo"), Some(42));
        assert_eq!(lfu.get_value("bar"), Some(99));
        assert!(lfu.contains("foo"));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lfu_access_tracking() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let lfu = Lfu::new(dict);

        // Access foo once
        assert_eq!(lfu.get_value("foo"), Some(42));
        assert_eq!(lfu.access_count("foo"), Some(1));

        // Access bar three times
        assert_eq!(lfu.get_value("bar"), Some(99));
        assert_eq!(lfu.get_value("bar"), Some(99));
        assert_eq!(lfu.get_value("bar"), Some(99));
        assert_eq!(lfu.access_count("bar"), Some(3));

        // foo has lower frequency
        assert!(
            lfu.access_count("foo")
                .expect("expected Some count in test")
                < lfu
                    .access_count("bar")
                    .expect("expected Some count in test")
        );
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lfu_find_lfu() {
        let dict =
            PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99), ("baz", 123)]);

        let lfu = Lfu::new(dict);

        // Access with different frequencies
        assert_eq!(lfu.get_value("foo"), Some(42));

        assert_eq!(lfu.get_value("bar"), Some(99));
        assert_eq!(lfu.get_value("bar"), Some(99));

        assert_eq!(lfu.get_value("baz"), Some(123));
        assert_eq!(lfu.get_value("baz"), Some(123));
        assert_eq!(lfu.get_value("baz"), Some(123));

        // foo is least frequently used (count = 1)
        let lfu_term = lfu.find_lfu(&["foo", "bar", "baz"]);
        assert_eq!(lfu_term, Some("foo".to_string()));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lfu_eviction() {
        let dict =
            PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99), ("baz", 123)]);

        let lfu = Lfu::new(dict);

        // Access all with different frequencies
        assert_eq!(lfu.get_value("foo"), Some(42));

        assert_eq!(lfu.get_value("bar"), Some(99));
        assert_eq!(lfu.get_value("bar"), Some(99));

        assert_eq!(lfu.get_value("baz"), Some(123));
        assert_eq!(lfu.get_value("baz"), Some(123));
        assert_eq!(lfu.get_value("baz"), Some(123));

        // Evict LFU
        let evicted = lfu.evict_lfu(&["foo", "bar", "baz"]);
        assert_eq!(evicted, Some("foo".to_string()));

        // foo should have no count data now
        assert_eq!(lfu.access_count("foo"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lfu_increment_on_access() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let lfu = Lfu::new(dict);

        // Initial access
        assert_eq!(lfu.get_value("foo"), Some(42));
        assert_eq!(lfu.access_count("foo"), Some(1));

        assert_eq!(lfu.get_value("bar"), Some(99));
        assert_eq!(lfu.access_count("bar"), Some(1));

        // foo is LFU (both have count 1, but foo comes first alphabetically)
        let lfu_term = lfu.find_lfu(&["foo", "bar"]);
        assert!(lfu_term == Some("foo".to_string()) || lfu_term == Some("bar".to_string()));

        // Access foo again
        assert_eq!(lfu.get_value("foo"), Some(42));
        assert_eq!(lfu.access_count("foo"), Some(2));

        // Now bar is definitely LFU
        assert_eq!(lfu.find_lfu(&["foo", "bar"]), Some("bar".to_string()));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lfu_clear_metadata() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let lfu = Lfu::new(dict);

        // Access both
        assert_eq!(lfu.get_value("foo"), Some(42));
        assert_eq!(lfu.get_value("bar"), Some(99));

        // Clear metadata
        lfu.clear_metadata();

        // No count data
        assert_eq!(lfu.access_count("foo"), None);
        assert_eq!(lfu.access_count("bar"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lfu_node_traversal() {
        let dict = PathMapDictionary::<()>::from_terms(["hello", "help"]);
        let lfu = Lfu::new(dict);

        let root = lfu.root();
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
    fn test_lfu_into_inner() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42)]);
        let lfu = Lfu::new(dict);
        let original = lfu.into_inner();

        assert_eq!(original.len(), Some(1));
        assert_eq!(original.get_value("foo"), Some(42));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_lfu_saturation() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42)]);
        let lfu = Lfu::new(dict);

        // Access many times to test saturation
        for _ in 0..100 {
            assert_eq!(lfu.get_value("foo"), Some(42));
        }

        // Should have count of 100
        assert_eq!(lfu.access_count("foo"), Some(100));
    }
}
