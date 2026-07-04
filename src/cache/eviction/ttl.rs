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
//! ```rust,ignore
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

use crate::sync_compat::RwLock;
use libdictenstein::{
    Dictionary, DictionaryNode, DictionaryValue, MappedDictionary, MappedDictionaryNode,
    SyncStrategy,
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
    fn new() -> Self {
        Self {
            inserted_at: Instant::now(),
        }
    }

    fn age(&self) -> Duration {
        self.inserted_at.elapsed()
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.age() > ttl
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
/// ```rust,ignore
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
pub struct Ttl<D> {
    inner: D,
    ttl: Duration,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<D> Ttl<D> {
    /// Creates a new TTL wrapper with the given duration.
    ///
    /// # Arguments
    ///
    /// - `dict`: The dictionary to wrap
    /// - `ttl`: Time-to-live duration
    ///
    /// # Examples
    ///
    /// ```rust,ignore
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
    fn is_expired(&self, term: &str) -> bool {
        let metadata = self.metadata.read();
        if let Some(entry_meta) = metadata.get(term) {
            entry_meta.is_expired(self.ttl)
        } else {
            // No metadata means entry was never accessed, treat as not expired
            false
        }
    }

    /// Records an entry access (updates metadata).
    fn record_access(&self, term: &str) {
        let mut metadata = self.metadata.write();
        metadata
            .entry(term.to_string())
            .or_insert_with(EntryMetadata::new);
    }

    /// Removes expired entries from metadata.
    ///
    /// This is a maintenance operation to prevent unbounded metadata growth.
    pub fn cleanup_expired(&self) {
        let mut metadata = self.metadata.write();
        metadata.retain(|_, entry_meta| !entry_meta.is_expired(self.ttl));
    }
}

impl<D> Dictionary for Ttl<D>
where
    D: Dictionary,
{
    type Node = TtlNode<D::Node>;

    #[inline]
    fn root(&self) -> Self::Node {
        TtlNode::new(self.inner.root(), self.ttl, Arc::clone(&self.metadata))
    }

    #[inline]
    fn len(&self) -> Option<usize> {
        self.inner.len()
    }

    #[inline]
    fn contains(&self, term: &str) -> bool {
        if self.is_expired(term) {
            return false;
        }
        self.inner.contains(term)
    }

    #[inline]
    fn sync_strategy(&self) -> SyncStrategy {
        self.inner.sync_strategy()
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
        // Check if expired first
        if self.is_expired(term) {
            return None;
        }

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
        if self.is_expired(term) {
            return false;
        }
        self.inner.contains_with_value(term, predicate)
    }
}

/// Node wrapper for TTL dictionary.
#[derive(Clone)]
pub struct TtlNode<N> {
    inner: N,
    ttl: Duration,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<N> TtlNode<N> {
    fn new(inner: N, ttl: Duration, metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>) -> Self {
        Self {
            inner,
            ttl,
            metadata,
        }
    }
}

impl<N> DictionaryNode for TtlNode<N>
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
            .map(|node| TtlNode::new(node, self.ttl, Arc::clone(&self.metadata)))
    }

    #[inline]
    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        let ttl = self.ttl;
        let metadata = Arc::clone(&self.metadata);
        Box::new(
            self.inner
                .edges()
                .map(move |(label, node)| (label, TtlNode::new(node, ttl, Arc::clone(&metadata)))),
        )
    }

    #[inline]
    fn edge_count(&self) -> Option<usize> {
        self.inner.edge_count()
    }
}

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
