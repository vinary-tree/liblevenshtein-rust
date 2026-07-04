//! Age-based eviction wrapper.
//!
//! This wrapper tracks insertion times for entries to identify the oldest
//! entries for eviction (FIFO - First In, First Out).
//!
//! # Architecture
//!
//! Unlike the old `AgeStrategy` which required `CacheEntry<V>` metadata,
//! this wrapper maintains a separate metadata map tracking insertion times.
//!
//! # Use Cases
//!
//! - Simple time-based eviction
//! - Fair eviction regardless of access patterns
//! - Predictable cache behavior for testing
//!
//! # Examples
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::Age;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("foo", 42),
//!     ("bar", 99),
//! ]);
//!
//! let age = Age::new(dict);
//! assert_eq!(age.get_value("foo"), Some(42));
//! ```

use crate::sync_compat::RwLock;
use libdictenstein::{
    Dictionary, DictionaryNode, DictionaryValue, MappedDictionary, MappedDictionaryNode,
    SyncStrategy,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Metadata tracked for each entry.
#[derive(Debug, Clone)]
struct EntryMetadata {
    inserted_at: Instant,
}

impl EntryMetadata {
    fn new() -> Self {
        Self {
            inserted_at: Instant::now(),
        }
    }

    fn age(&self) -> std::time::Duration {
        self.inserted_at.elapsed()
    }
}

/// Age-based eviction wrapper.
///
/// Evicts oldest entries first (FIFO - First In, First Out).
/// Tracks insertion time for each entry.
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
/// use liblevenshtein::cache::eviction::Age;
///
/// let dict = PathMapDictionary::from_terms_with_values([
///     ("hello", 1),
///     ("world", 2),
/// ]);
///
/// let age = Age::new(dict);
/// assert_eq!(age.get_value("hello"), Some(1));
/// ```
#[derive(Clone)]
pub struct Age<D> {
    inner: D,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<D> Age<D> {
    /// Creates a new Age wrapper.
    ///
    /// # Arguments
    ///
    /// - `dict`: The dictionary to wrap
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::Age;
    ///
    /// let dict = PathMapDictionary::from_terms_with_values([
    ///     ("foo", 42),
    /// ]);
    ///
    /// let age = Age::new(dict);
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
        let mut metadata = self.metadata.write();
        metadata
            .entry(term.to_string())
            .or_insert_with(EntryMetadata::new);
    }

    /// Gets the age for an entry (time since insertion).
    ///
    /// Returns `None` if the entry has never been accessed.
    pub fn age(&self, term: &str) -> Option<std::time::Duration> {
        let metadata = self.metadata.read();
        metadata.get(term).map(|m| m.age())
    }

    /// Finds the oldest entry among the given terms.
    ///
    /// Returns the term with the longest time since insertion.
    pub fn find_oldest(&self, terms: &[&str]) -> Option<String> {
        let metadata = self.metadata.read();
        terms
            .iter()
            .filter_map(|&term| metadata.get(term).map(|m| (term, m.age())))
            .max_by_key(|(_, age)| *age)
            .map(|(term, _)| term.to_string())
    }

    /// Evicts the oldest entry from metadata.
    ///
    /// Returns the evicted term if any.
    pub fn evict_oldest(&self, terms: &[&str]) -> Option<String> {
        if let Some(oldest_term) = self.find_oldest(terms) {
            let mut metadata = self.metadata.write();
            metadata.remove(&oldest_term);
            Some(oldest_term)
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

impl<D> Dictionary for Age<D>
where
    D: Dictionary,
{
    type Node = AgeNode<D::Node>;

    #[inline]
    fn root(&self) -> Self::Node {
        AgeNode::new(self.inner.root(), Arc::clone(&self.metadata))
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

impl<D, V> MappedDictionary for Age<D>
where
    D: MappedDictionary<Value = V>,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn get_value(&self, term: &str) -> Option<Self::Value> {
        // Record access (first time sets insertion time)
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

/// Node wrapper for Age dictionary.
#[derive(Clone)]
pub struct AgeNode<N> {
    inner: N,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<N> AgeNode<N> {
    fn new(inner: N, metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>) -> Self {
        Self { inner, metadata }
    }
}

impl<N> DictionaryNode for AgeNode<N>
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
            .map(|node| AgeNode::new(node, Arc::clone(&self.metadata)))
    }

    #[inline]
    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        let metadata = Arc::clone(&self.metadata);
        Box::new(
            self.inner
                .edges()
                .map(move |(label, node)| (label, AgeNode::new(node, Arc::clone(&metadata)))),
        )
    }

    #[inline]
    fn edge_count(&self) -> Option<usize> {
        self.inner.edge_count()
    }
}

impl<N, V> MappedDictionaryNode for AgeNode<N>
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
    fn test_age_wrapper_basic() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let age = Age::new(dict);

        // Values should be accessible
        assert_eq!(age.get_value("foo"), Some(42));
        assert_eq!(age.get_value("bar"), Some(99));
        assert!(age.contains("foo"));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_age_tracking() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let age_wrapper = Age::new(dict);

        // Access foo
        assert_eq!(age_wrapper.get_value("foo"), Some(42));

        // Wait a bit
        thread::sleep(Duration::from_millis(10));

        // Access bar
        assert_eq!(age_wrapper.get_value("bar"), Some(99));

        // foo should be older
        let foo_age = age_wrapper.age("foo").expect("expected Some age in test");
        let bar_age = age_wrapper.age("bar").expect("expected Some age in test");

        assert!(foo_age > bar_age);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_age_find_oldest() {
        let dict =
            PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99), ("baz", 123)]);

        let age_wrapper = Age::new(dict);

        // Access in order: foo, bar, baz
        assert_eq!(age_wrapper.get_value("foo"), Some(42));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(age_wrapper.get_value("bar"), Some(99));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(age_wrapper.get_value("baz"), Some(123));

        // foo is oldest
        let oldest = age_wrapper.find_oldest(&["foo", "bar", "baz"]);
        assert_eq!(oldest, Some("foo".to_string()));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_age_eviction() {
        let dict =
            PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99), ("baz", 123)]);

        let age_wrapper = Age::new(dict);

        // Access all entries
        assert_eq!(age_wrapper.get_value("foo"), Some(42));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(age_wrapper.get_value("bar"), Some(99));
        thread::sleep(Duration::from_millis(10));
        assert_eq!(age_wrapper.get_value("baz"), Some(123));

        // Evict oldest
        let evicted = age_wrapper.evict_oldest(&["foo", "bar", "baz"]);
        assert_eq!(evicted, Some("foo".to_string()));

        // foo should have no age data now
        assert_eq!(age_wrapper.age("foo"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_age_no_update_on_access() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let age_wrapper = Age::new(dict);

        // Access foo
        assert_eq!(age_wrapper.get_value("foo"), Some(42));
        let foo_age_1 = age_wrapper.age("foo").expect("expected Some age in test");

        thread::sleep(Duration::from_millis(10));

        // Access bar
        assert_eq!(age_wrapper.get_value("bar"), Some(99));

        thread::sleep(Duration::from_millis(10));

        // Access foo again - age should increase (not reset like LRU)
        assert_eq!(age_wrapper.get_value("foo"), Some(42));
        let foo_age_2 = age_wrapper.age("foo").expect("expected Some age in test");

        // Age increases over time (not reset on access)
        assert!(foo_age_2 > foo_age_1);

        // foo is still oldest
        assert_eq!(
            age_wrapper.find_oldest(&["foo", "bar"]),
            Some("foo".to_string())
        );
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_age_clear_metadata() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let age_wrapper = Age::new(dict);

        // Access both
        assert_eq!(age_wrapper.get_value("foo"), Some(42));
        assert_eq!(age_wrapper.get_value("bar"), Some(99));

        // Clear metadata
        age_wrapper.clear_metadata();

        // No age data
        assert_eq!(age_wrapper.age("foo"), None);
        assert_eq!(age_wrapper.age("bar"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_age_node_traversal() {
        let dict = PathMapDictionary::<()>::from_terms(["hello", "help"]);
        let age_wrapper = Age::new(dict);

        let root = age_wrapper.root();
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
    fn test_age_into_inner() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42)]);
        let age_wrapper = Age::new(dict);
        let original = age_wrapper.into_inner();

        assert_eq!(original.len(), Some(1));
        assert_eq!(original.get_value("foo"), Some(42));
    }
}
