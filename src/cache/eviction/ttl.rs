//! TTL (Time-To-Live) eviction wrapper.
//!
//! This wrapper filters out expired entries based on their age. It tracks
//! insertion timestamps for all entries and returns `None` for expired values.
//!
//! # Architecture
//!
//! Unlike the old `TtlStrategy` which required `CacheEntry<V>` metadata,
//! this wrapper maintains a separate metadata map tracking insertion times.
//!
//! # Thread Safety
//!
//! The metadata map is wrapped in `Arc<RwLock<...>>` for thread-safe access.
//!
//! # Use Cases
//!
//! - AI code chat: 5-10 minute TTL for session-based caching
//! - Documentation search: 1 hour TTL for stable content
//! - Error solutions: 30 minutes TTL for evolving solutions
//!
//! # Examples
//!
//! ```rust
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::Ttl;
//! use std::time::Duration;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("foo", 42),
//!     ("bar", 99),
//! ]);
//!
//! let ttl_dict = Ttl::new(dict, Duration::from_secs(300)); // 5 minutes
//!
//! // Values expire after 5 minutes
//! assert_eq!(ttl_dict.get_value("foo"), Some(42));
//! ```

use crate::dictionary::node_adapter::{
    impl_semantic_dictionary_node, map_transparent_traversal_root,
};
use crate::sync_compat::RwLock;
use libdictenstein::{
    CharUnit, Dictionary, DictionaryNode, DictionaryTraversalRoot, DictionaryValue,
    MappedDictionary, MappedDictionaryNode, SyncStrategy,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Metadata tracked for each entry.
#[derive(Debug, Clone)]
struct EntryMetadata {
    inserted_at: Instant,
}

impl EntryMetadata {
    fn new(inserted_at: Instant) -> Self {
        Self { inserted_at }
    }

    fn is_expired_at(&self, ttl: Duration, observed_at: Instant) -> bool {
        observed_at
            .checked_duration_since(self.inserted_at)
            .is_some_and(|age| age > ttl)
    }
}

/// TTL (Time-To-Live) eviction wrapper.
///
/// Filters out entries that have exceeded their time-to-live duration.
/// Returns `None` for expired entries, effectively evicting them.
///
/// # Type Parameters
///
/// - `D`: Inner dictionary type
///
/// # Examples
///
/// ```rust
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::dictionary::MappedDictionary;
/// use liblevenshtein::cache::eviction::Ttl;
/// use std::time::Duration;
///
/// let dict = PathMapDictionary::from_terms_with_values([
///     ("hello", 1),
///     ("world", 2),
/// ]);
///
/// let ttl = Ttl::new(dict, Duration::from_secs(300));
/// assert_eq!(ttl.get_value("hello"), Some(1));
/// ```
#[derive(Clone)]
pub struct Ttl<D>
where
    D: Dictionary,
{
    inner: D,
    ttl: Duration,
    metadata: TtlMetadata<<D::Node as DictionaryNode>::Unit>,
}

type TtlMetadata<U> = Arc<RwLock<HashMap<Vec<U>, EntryMetadata>>>;

impl<D> Ttl<D>
where
    D: Dictionary,
{
    /// Creates a new TTL wrapper with the given duration.
    ///
    /// # Arguments
    ///
    /// - `dict`: The dictionary to wrap
    /// - `ttl`: Time-to-live duration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::Ttl;
    /// use std::time::Duration;
    ///
    /// let dict = PathMapDictionary::from_terms_with_values([
    ///     ("foo", 42),
    /// ]);
    ///
    /// let ttl = Ttl::new(dict, Duration::from_secs(300));
    /// ```
    pub fn new(dict: D, ttl: Duration) -> Self {
        Self {
            inner: dict,
            ttl,
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

    /// Gets the TTL duration.
    #[inline]
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Checks if an entry is expired.
    fn is_expired_at(
        &self,
        units: &[<D::Node as DictionaryNode>::Unit],
        observed_at: Instant,
    ) -> bool {
        let metadata = self.metadata.read();
        if let Some(entry_meta) = metadata.get(units) {
            entry_meta.is_expired_at(self.ttl, observed_at)
        } else {
            // No metadata means entry was never accessed, treat as not expired
            false
        }
    }

    /// Start the lifetime of one successfully resolved entry, if necessary.
    fn record_access_at(&self, units: &[<D::Node as DictionaryNode>::Unit], observed_at: Instant) {
        let mut metadata = self.metadata.write();
        metadata
            .entry(units.to_vec())
            .or_insert_with(|| EntryMetadata::new(observed_at));
    }

    /// Removes expired entries from metadata.
    ///
    /// This is a maintenance operation to prevent unbounded metadata growth.
    /// Because this wrapper is a non-owning view and cannot delete the inner
    /// value, a subsequent successful access begins a fresh lifetime.
    pub fn cleanup_expired(&self) {
        let observed_at = Instant::now();
        let mut metadata = self.metadata.write();
        metadata.retain(|_, entry_meta| !entry_meta.is_expired_at(self.ttl, observed_at));
    }

    #[cfg(test)]
    pub(crate) fn set_entry_age_for_test(&self, term: &str, age: Duration) {
        let units = <D::Node as DictionaryNode>::Unit::from_str(term);
        self.set_units_age_for_test(&units, age);
    }

    #[cfg(test)]
    pub(crate) fn set_units_age_for_test(
        &self,
        units: &[<D::Node as DictionaryNode>::Unit],
        age: Duration,
    ) {
        let inserted_at = Instant::now()
            .checked_sub(age)
            .expect("test age must fit within the monotonic clock epoch");
        self.metadata
            .write()
            .insert(units.to_vec(), EntryMetadata::new(inserted_at));
    }
}

impl<D> Dictionary for Ttl<D>
where
    D: Dictionary,
{
    type Node = TtlNode<D::Node>;

    #[inline]
    fn root(&self) -> Self::Node {
        TtlNode::new(
            self.inner.root(),
            self.ttl,
            Arc::clone(&self.metadata),
            Instant::now(),
        )
    }

    #[inline]
    fn traversal_root(&self) -> DictionaryTraversalRoot<Self::Node> {
        let ttl = self.ttl;
        let metadata = Arc::clone(&self.metadata);
        let observed_at = Instant::now();
        map_transparent_traversal_root(self.inner.traversal_root(), |inner| {
            TtlNode::new(inner, ttl, metadata, observed_at)
        })
    }

    #[inline]
    fn len(&self) -> Option<usize> {
        self.inner.len()
    }

    #[inline]
    fn contains(&self, term: &str) -> bool {
        let units = <D::Node as DictionaryNode>::Unit::from_str(term);
        if self.is_expired_at(&units, Instant::now()) {
            return false;
        }
        self.inner.contains(term)
    }

    #[inline]
    fn sync_strategy(&self) -> SyncStrategy {
        self.inner.sync_strategy()
    }

    #[inline]
    fn is_suffix_based(&self) -> bool {
        self.inner.is_suffix_based()
    }
}

impl<D, V> MappedDictionary for Ttl<D>
where
    D: MappedDictionary<Value = V>,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn get_value(&self, term: &str) -> Option<Self::Value> {
        let observed_at = Instant::now();
        let units = <D::Node as DictionaryNode>::Unit::from_str(term);
        // Check if expired first
        if self.is_expired_at(&units, observed_at) {
            return None;
        }

        // Only existing, valued terms consume metadata capacity.
        let value = self.inner.get_value(term)?;
        self.record_access_at(&units, observed_at);
        Some(value)
    }

    #[inline]
    fn contains_with_value<F>(&self, term: &str, predicate: F) -> bool
    where
        F: Fn(&Self::Value) -> bool,
    {
        let observed_at = Instant::now();
        let units = <D::Node as DictionaryNode>::Unit::from_str(term);
        if self.is_expired_at(&units, observed_at) {
            return false;
        }
        let matches = self.inner.contains_with_value(term, predicate);
        if matches {
            self.record_access_at(&units, observed_at);
        }
        matches
    }
}

/// Node wrapper for TTL dictionary.
#[derive(Clone)]
pub struct TtlNode<N: DictionaryNode> {
    inner: N,
    ttl: Duration,
    metadata: Arc<RwLock<HashMap<Vec<N::Unit>, EntryMetadata>>>,
    observed_at: Instant,
}

impl<N: DictionaryNode> TtlNode<N> {
    fn new(
        inner: N,
        ttl: Duration,
        metadata: Arc<RwLock<HashMap<Vec<N::Unit>, EntryMetadata>>>,
        observed_at: Instant,
    ) -> Self {
        Self {
            inner,
            ttl,
            metadata,
            observed_at,
        }
    }

    #[inline]
    fn is_expired(&self, units: &[N::Unit]) -> bool {
        self.metadata
            .read()
            .get(units)
            .is_some_and(|entry| entry.is_expired_at(self.ttl, self.observed_at))
    }

    #[inline]
    fn record_successful_access(&self, units: &[N::Unit]) {
        self.metadata
            .write()
            .entry(units.to_vec())
            .or_insert_with(|| EntryMetadata::new(Instant::now()));
    }
}

impl_semantic_dictionary_node!(
    TtlNode,
    |owner, child| TtlNode::new(
        child,
        owner.ttl,
        Arc::clone(&owner.metadata),
        owner.observed_at,
    ),
    |_owner| true,
    |owner, units| owner.inner.accepts_final_units(units) && !owner.is_expired(units)
);

impl<N, V> MappedDictionaryNode for TtlNode<N>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn value(&self) -> Option<Self::Value> {
        self.inner.value()
    }

    #[inline]
    fn value_at_final(&self) -> Option<Self::Value> {
        self.inner.value_at_final()
    }

    #[inline]
    fn value_at_final_with_units(&self, units: &[Self::Unit]) -> Option<Self::Value> {
        if !self.accepts_final_units(units) {
            return None;
        }
        let value = self.inner.value_at_final_with_units(units)?;
        self.record_successful_access(units);
        Some(value)
    }

    #[inline]
    fn supports_snapshot_cursor_values(&self) -> bool {
        self.inner.supports_snapshot_cursor_values()
    }

    #[inline]
    fn supports_snapshot_graph_values(&self) -> bool {
        self.inner.supports_snapshot_graph_values()
    }

    #[inline]
    fn snapshot_traversal_graph(
        &self,
    ) -> Option<
        Arc<libdictenstein::SnapshotTraversalGraph<Self::Unit, Self::SnapshotGraphValueHandle>>,
    > {
        self.inner.snapshot_traversal_graph()
    }

    #[inline]
    unsafe fn snapshot_cursor_value(
        &self,
        cursor: Self::SnapshotCursor,
    ) -> Option<Option<Self::Value>> {
        // SAFETY: this node retains the same inner revision and cursor owner.
        unsafe { self.inner.snapshot_cursor_value(cursor) }
    }

    #[inline]
    unsafe fn snapshot_cursor_value_with_units(
        &self,
        cursor: Self::SnapshotCursor,
        units: &[Self::Unit],
    ) -> Option<Option<Self::Value>> {
        if !self.accepts_final_units(units) {
            return Some(None);
        }
        // SAFETY: cursor and units come from this exact retained revision.
        let value = unsafe { self.inner.snapshot_cursor_value_with_units(cursor, units) }?;
        if value.is_some() {
            self.record_successful_access(units);
        }
        Some(value)
    }

    #[inline]
    unsafe fn snapshot_graph_cursor_value(
        &self,
        graph: &libdictenstein::SnapshotTraversalGraph<Self::Unit, Self::SnapshotGraphValueHandle>,
        cursor: libdictenstein::SnapshotTraversalCursor,
    ) -> Option<Option<Self::Value>> {
        // SAFETY: this node retains the same graph, cursor, and inner owner.
        unsafe { self.inner.snapshot_graph_cursor_value(graph, cursor) }
    }

    #[inline]
    unsafe fn snapshot_graph_cursor_value_with_units(
        &self,
        graph: &libdictenstein::SnapshotTraversalGraph<Self::Unit, Self::SnapshotGraphValueHandle>,
        cursor: libdictenstein::SnapshotTraversalCursor,
        units: &[Self::Unit],
    ) -> Option<Option<Self::Value>> {
        if !self.accepts_final_units(units) {
            return Some(None);
        }
        // SAFETY: graph, cursor, and units come from this retained revision.
        let value = unsafe {
            self.inner
                .snapshot_graph_cursor_value_with_units(graph, cursor, units)
        }?;
        if value.is_some() {
            self.record_successful_access(units);
        }
        Some(value)
    }
}

#[cfg(all(test, feature = "pathmap-backend"))]
mod tests {
    use super::*;
    #[cfg(feature = "pathmap-backend")]
    use libdictenstein::pathmap::PathMapDictionary;
    use std::thread;

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_ttl_wrapper_basic() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let ttl = Ttl::new(dict, Duration::from_secs(300));

        // Values should be accessible
        assert_eq!(ttl.get_value("foo"), Some(42));
        assert_eq!(ttl.get_value("bar"), Some(99));
        assert!(ttl.contains("foo"));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_ttl_expiration() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let ttl = Ttl::new(dict, Duration::from_millis(50));

        // Access to record metadata
        assert_eq!(ttl.get_value("foo"), Some(42));

        // Wait for expiration
        thread::sleep(Duration::from_millis(60));

        // Should be expired now
        assert_eq!(ttl.get_value("foo"), None);
        assert!(!ttl.contains("foo"));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_ttl_unaccessed_entries() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let ttl = Ttl::new(dict, Duration::from_millis(50));

        // Don't access "bar" - it has no metadata
        thread::sleep(Duration::from_millis(60));

        // Unaccessed entries should still be accessible
        // (no metadata means not expired)
        assert_eq!(ttl.get_value("bar"), Some(99));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_ttl_cleanup() {
        let dict =
            PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99), ("baz", 123)]);

        let ttl = Ttl::new(dict, Duration::from_millis(50));

        // Access all entries
        assert_eq!(ttl.get_value("foo"), Some(42));
        assert_eq!(ttl.get_value("bar"), Some(99));
        assert_eq!(ttl.get_value("baz"), Some(123));

        // Wait for expiration
        thread::sleep(Duration::from_millis(60));

        // All should be expired
        assert_eq!(ttl.get_value("foo"), None);
        assert_eq!(ttl.get_value("bar"), None);
        assert_eq!(ttl.get_value("baz"), None);

        // Cleanup should remove expired metadata
        ttl.cleanup_expired();

        // Metadata map should be empty now
        let metadata = ttl.metadata.read();
        assert_eq!(metadata.len(), 0);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_ttl_node_traversal() {
        let dict = PathMapDictionary::<()>::from_terms(["hello", "help"]);
        let ttl = Ttl::new(dict, Duration::from_secs(300));

        let root = ttl.root();
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
    fn test_ttl_contains_with_value() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let ttl = Ttl::new(dict, Duration::from_millis(50));

        // Access to record metadata
        assert_eq!(ttl.get_value("foo"), Some(42));

        // Should match predicate
        assert!(ttl.contains_with_value("foo", |v| *v == 42));

        // Wait for expiration
        thread::sleep(Duration::from_millis(60));

        // Expired entry should not match
        assert!(!ttl.contains_with_value("foo", |v| *v == 42));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_ttl_into_inner() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42)]);
        let ttl = Ttl::new(dict, Duration::from_secs(300));
        let original = ttl.into_inner();

        assert_eq!(original.len(), Some(1));
        assert_eq!(original.get_value("foo"), Some(42));
    }
}

#[cfg(test)]
mod semantic_query_tests {
    use super::*;
    use crate::transducer::{Algorithm, NoPruning, SubsequenceQueryIterator, Transducer};
    use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
    use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgU64};
    use std::collections::HashSet;

    fn expired_ascii() -> Ttl<DynamicDawg<u64>> {
        let dictionary = DynamicDawg::from_sorted_terms_with_values([("cat", 7_u64), ("dog", 8)]);
        let ttl = Ttl::new(dictionary, Duration::from_secs(1));
        ttl.set_entry_age_for_test("cat", Duration::from_secs(2));
        ttl
    }

    #[test]
    fn every_query_surface_hides_expired_terminals() {
        let ttl = expired_ascii();
        assert!(!ttl.contains("cat"));
        assert_eq!(ttl.get_value("cat"), None);

        let subsequence: Vec<_> =
            SubsequenceQueryIterator::from_dictionary(&ttl, b"cat".to_vec()).collect();
        assert!(subsequence.is_empty());

        let transducer = Transducer::new(ttl, Algorithm::Standard);
        assert!(transducer.query_with_distance("cat", 0).next().is_none());
        assert!(transducer.query_ordered("cat", 0).next().is_none());
        assert!(transducer
            .query_with_pruner("cat", 0, NoPruning)
            .next()
            .is_none());
        assert!(transducer.query_values("cat", 0).next().is_none());
        assert!(transducer
            .query_filtered("cat", 0, |_| true)
            .next()
            .is_none());
        let values = HashSet::from([7_u64]);
        assert!(transducer
            .query_by_value_set("cat", 0, &values)
            .next()
            .is_none());
        assert!(transducer
            .query_suggestions("cat", 0, |_term: &str, _distance, value: &u64| {
                *value as f64
            })
            .next()
            .is_none());
    }

    #[test]
    fn ttl_uses_exact_unicode_and_native_u64_keys() {
        let unicode = Ttl::new(
            DynamicDawgChar::from_sorted_terms_with_values([("cafe", 1_u64), ("café", 2)]),
            Duration::from_secs(1),
        );
        unicode.set_entry_age_for_test("café", Duration::from_secs(2));
        let unicode_query = Transducer::new(unicode, Algorithm::Standard);
        assert!(unicode_query.query_values("café", 0).next().is_none());
        assert_eq!(
            unicode_query.query_values("cafe", 0).collect::<Vec<_>>(),
            [("cafe".to_owned(), 0, 1)]
        );

        let tokens = DynamicDawgU64::new();
        tokens.insert_sequence_with_value(&[1], 10_u64);
        tokens.insert_sequence_with_value(&[1, 0], 20_u64);
        let tokens = Ttl::new(tokens, Duration::from_secs(1));
        tokens.set_units_age_for_test(&[1], Duration::from_secs(2));
        let token_query = Transducer::new(tokens, Algorithm::Standard);
        assert!(token_query.query_units_values(&[1], 0).next().is_none());
        assert_eq!(
            token_query
                .query_units_values(&[1, 0], 0)
                .collect::<Vec<_>>(),
            [(vec![1, 0], 0, 20)]
        );
    }
}
