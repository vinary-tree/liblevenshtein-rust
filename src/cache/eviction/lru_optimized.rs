//! Optimized LRU implementation demonstrating all optimization techniques.
//!
//! This module shows performance optimizations using feature flags:
//! - `eviction-parking-lot`: Use parking_lot::RwLock instead of std::sync::RwLock
//! - `eviction-dashmap`: Use DashMap for lock-free concurrent access
//! - `eviction-arc-str`: Use Arc<str> instead of String for keys
//! - `eviction-compact-metadata`: Compact metadata representation
//! - `eviction-coarse-timestamps`: Coarse-grained timestamps (reduce syscalls)

use crate::dictionary::{
    Dictionary, DictionaryNode, DictionaryValue, MappedDictionary, MappedDictionaryNode,
    SyncStrategy,
};
use std::sync::Arc;

// Conditional imports based on feature flags
#[cfg(all(feature = "eviction-parking-lot", not(feature = "eviction-dashmap")))]
use crate::sync_compat::RwLock;
#[cfg(all(
    not(feature = "eviction-parking-lot"),
    not(feature = "eviction-dashmap")
))]
use std::sync::RwLock;

#[cfg(feature = "eviction-dashmap")]
use dashmap::DashMap;
#[cfg(not(feature = "eviction-dashmap"))]
use std::collections::HashMap;

#[cfg(feature = "eviction-coarse-timestamps")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(feature = "eviction-coarse-timestamps"))]
use std::time::Instant;

// Coarse timestamp management
#[cfg(feature = "eviction-coarse-timestamps")]
static COARSE_TIMESTAMP_MS: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "eviction-coarse-timestamps")]
pub(crate) fn init_coarse_timestamp_thread() {
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    thread::spawn(|| loop {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        COARSE_TIMESTAMP_MS.store(now_ms, Ordering::Relaxed);
        thread::sleep(Duration::from_millis(100));
    });
}

// Metadata representation
#[cfg(not(feature = "eviction-compact-metadata"))]
#[derive(Debug, Clone)]
struct EntryMetadata {
    #[cfg(not(feature = "eviction-coarse-timestamps"))]
    last_accessed: Instant,
    #[cfg(feature = "eviction-coarse-timestamps")]
    last_accessed_ms: u64,
}

#[cfg(feature = "eviction-compact-metadata")]
#[derive(Debug, Clone)]
#[repr(C)]
struct EntryMetadata {
    last_accessed_ms: u64,
}

impl EntryMetadata {
    fn new() -> Self {
        #[cfg(all(
            not(feature = "eviction-compact-metadata"),
            not(feature = "eviction-coarse-timestamps")
        ))]
        {
            Self {
                last_accessed: Instant::now(),
            }
        }

        #[cfg(all(
            not(feature = "eviction-compact-metadata"),
            feature = "eviction-coarse-timestamps"
        ))]
        {
            Self {
                last_accessed_ms: COARSE_TIMESTAMP_MS.load(Ordering::Relaxed),
            }
        }

        #[cfg(all(
            feature = "eviction-compact-metadata",
            not(feature = "eviction-coarse-timestamps")
        ))]
        {
            Self {
                last_accessed_ms: get_timestamp_ms(),
            }
        }

        #[cfg(all(
            feature = "eviction-compact-metadata",
            feature = "eviction-coarse-timestamps"
        ))]
        {
            Self {
                last_accessed_ms: COARSE_TIMESTAMP_MS.load(Ordering::Relaxed),
            }
        }
    }

    fn update_access(&mut self) {
        #[cfg(all(
            not(feature = "eviction-compact-metadata"),
            not(feature = "eviction-coarse-timestamps")
        ))]
        {
            self.last_accessed = Instant::now();
        }

        #[cfg(feature = "eviction-coarse-timestamps")]
        {
            self.last_accessed_ms = COARSE_TIMESTAMP_MS.load(Ordering::Relaxed);
        }

        #[cfg(all(
            feature = "eviction-compact-metadata",
            not(feature = "eviction-coarse-timestamps")
        ))]
        {
            self.last_accessed_ms = get_timestamp_ms();
        }
    }

    fn recency_score(&self) -> u64 {
        #[cfg(all(
            not(feature = "eviction-compact-metadata"),
            not(feature = "eviction-coarse-timestamps")
        ))]
        {
            self.last_accessed.elapsed().as_millis() as u64
        }

        #[cfg(feature = "eviction-coarse-timestamps")]
        {
            let now = COARSE_TIMESTAMP_MS.load(Ordering::Relaxed);
            now.saturating_sub(self.last_accessed_ms)
        }

        #[cfg(all(
            feature = "eviction-compact-metadata",
            not(feature = "eviction-coarse-timestamps")
        ))]
        {
            let now = get_timestamp_ms();
            now.saturating_sub(self.last_accessed_ms)
        }
    }
}

#[cfg(all(
    feature = "eviction-compact-metadata",
    not(feature = "eviction-coarse-timestamps")
))]
fn get_timestamp_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

// Key type
#[cfg(feature = "eviction-arc-str")]
type MetadataKey = Arc<str>;
#[cfg(not(feature = "eviction-arc-str"))]
type MetadataKey = String;

#[cfg(feature = "eviction-arc-str")]
fn make_key(term: &str) -> MetadataKey {
    Arc::from(term)
}
#[cfg(not(feature = "eviction-arc-str"))]
fn make_key(term: &str) -> MetadataKey {
    term.to_string()
}

// Storage type
#[cfg(feature = "eviction-dashmap")]
type MetadataStorage = Arc<DashMap<MetadataKey, EntryMetadata>>;
#[cfg(not(feature = "eviction-dashmap"))]
type MetadataStorage = Arc<RwLock<HashMap<MetadataKey, EntryMetadata>>>;

/// Optimized LRU wrapper with conditional optimizations.
#[derive(Clone)]
pub struct LruOptimized<D> {
    inner: D,
    metadata: MetadataStorage,
}

impl<D> LruOptimized<D> {
    /// Creates a new LRU-optimized wrapper around the given dictionary.
    ///
    /// # Arguments
    ///
    /// * `dict` - The dictionary to wrap with LRU tracking
    ///
    /// # Returns
    ///
    /// A new `LruOptimized` instance
    pub fn new(dict: D) -> Self {
        #[cfg(feature = "eviction-coarse-timestamps")]
        {
            static TIMESTAMP_THREAD_INIT: std::sync::Once = std::sync::Once::new();
            TIMESTAMP_THREAD_INIT.call_once(|| {
                init_coarse_timestamp_thread();
            });
        }

        Self {
            inner: dict,
            #[cfg(feature = "eviction-dashmap")]
            metadata: Arc::new(DashMap::new()),
            #[cfg(not(feature = "eviction-dashmap"))]
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Consumes the wrapper and returns the inner dictionary.
    ///
    /// # Returns
    ///
    /// The wrapped dictionary
    #[inline]
    pub fn into_inner(self) -> D {
        self.inner
    }

    /// Returns a reference to the inner dictionary.
    ///
    /// # Returns
    ///
    /// Reference to the wrapped dictionary
    #[inline]
    pub fn inner(&self) -> &D {
        &self.inner
    }

    fn record_access(&self, term: &str) {
        #[cfg(feature = "eviction-dashmap")]
        {
            self.metadata
                .entry(make_key(term))
                .and_modify(|m| m.update_access())
                .or_insert_with(EntryMetadata::new);
        }

        #[cfg(not(feature = "eviction-dashmap"))]
        {
            #[cfg(feature = "eviction-parking-lot")]
            let mut metadata = self.metadata.write();
            #[cfg(not(feature = "eviction-parking-lot"))]
            let mut metadata = self.metadata.write().unwrap();

            metadata
                .entry(make_key(term))
                .and_modify(|m| m.update_access())
                .or_insert_with(EntryMetadata::new);
        }
    }

    /// Gets the recency score for a term (lower is more recent).
    ///
    /// # Arguments
    ///
    /// * `term` - The term to query
    ///
    /// # Returns
    ///
    /// The recency score if the term exists, `None` otherwise
    pub fn recency(&self, term: &str) -> Option<u64> {
        #[cfg(feature = "eviction-dashmap")]
        {
            self.metadata.get(term).map(|m| m.recency_score())
        }

        #[cfg(not(feature = "eviction-dashmap"))]
        {
            #[cfg(feature = "eviction-parking-lot")]
            let metadata = self.metadata.read();
            #[cfg(not(feature = "eviction-parking-lot"))]
            let metadata = self.metadata.read().unwrap();

            metadata.get(term).map(|m| m.recency_score())
        }
    }

    /// Finds the least recently used term from a list of candidates.
    ///
    /// # Arguments
    ///
    /// * `terms` - Slice of term candidates to check
    ///
    /// # Returns
    ///
    /// The LRU term if any have metadata, `None` if none are tracked
    pub fn find_lru(&self, terms: &[&str]) -> Option<String> {
        #[cfg(feature = "eviction-dashmap")]
        {
            terms
                .iter()
                .filter_map(|&term| self.metadata.get(term).map(|m| (term, m.recency_score())))
                .max_by_key(|(_, recency)| *recency)
                .map(|(term, _)| term.to_string())
        }

        #[cfg(not(feature = "eviction-dashmap"))]
        {
            #[cfg(feature = "eviction-parking-lot")]
            let metadata = self.metadata.read();
            #[cfg(not(feature = "eviction-parking-lot"))]
            let metadata = self.metadata.read().unwrap();

            terms
                .iter()
                .filter_map(|&term| metadata.get(term).map(|m| (term, m.recency_score())))
                .max_by_key(|(_, recency)| *recency)
                .map(|(term, _)| term.to_string())
        }
    }

    /// Evicts the least recently used term from a list of candidates.
    ///
    /// Removes the LRU term's metadata from tracking.
    ///
    /// # Arguments
    ///
    /// * `terms` - Slice of term candidates to check
    ///
    /// # Returns
    ///
    /// The evicted term if any were tracked, `None` otherwise
    pub fn evict_lru(&self, terms: &[&str]) -> Option<String> {
        if let Some(lru_term) = self.find_lru(terms) {
            #[cfg(feature = "eviction-dashmap")]
            {
                self.metadata.remove(&*lru_term);
            }

            #[cfg(not(feature = "eviction-dashmap"))]
            {
                #[cfg(feature = "eviction-parking-lot")]
                let mut metadata = self.metadata.write();
                #[cfg(not(feature = "eviction-parking-lot"))]
                let mut metadata = self.metadata.write().unwrap();

                metadata.remove(lru_term.as_str());
            }

            Some(lru_term)
        } else {
            None
        }
    }

    /// Clears all LRU metadata, resetting tracking state.
    ///
    /// This removes all recency information for all terms.
    pub fn clear_metadata(&self) {
        #[cfg(feature = "eviction-dashmap")]
        {
            self.metadata.clear();
        }

        #[cfg(not(feature = "eviction-dashmap"))]
        {
            #[cfg(feature = "eviction-parking-lot")]
            let mut metadata = self.metadata.write();
            #[cfg(not(feature = "eviction-parking-lot"))]
            let mut metadata = self.metadata.write().unwrap();

            metadata.clear();
        }
    }
}

impl<D> Dictionary for LruOptimized<D>
where
    D: Dictionary,
{
    type Node = LruOptimizedNode<D::Node>;

    #[inline]
    fn root(&self) -> Self::Node {
        LruOptimizedNode::new(self.inner.root(), Arc::clone(&self.metadata))
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

impl<D, V> MappedDictionary for LruOptimized<D>
where
    D: MappedDictionary<Value = V>,
    V: DictionaryValue,
{
    type Value = V;

    #[inline]
    fn get_value(&self, term: &str) -> Option<Self::Value> {
        self.record_access(term);
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

/// Dictionary node wrapper that tracks LRU metadata.
///
/// Wraps a dictionary node and maintains shared access to LRU tracking metadata.
#[derive(Clone)]
pub struct LruOptimizedNode<N> {
    inner: N,
    metadata: MetadataStorage,
}

impl<N> LruOptimizedNode<N> {
    fn new(inner: N, metadata: MetadataStorage) -> Self {
        Self { inner, metadata }
    }
}

impl<N> DictionaryNode for LruOptimizedNode<N>
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
            .map(|node| LruOptimizedNode::new(node, Arc::clone(&self.metadata)))
    }

    #[inline]
    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        let metadata = Arc::clone(&self.metadata);
        Box::new(
            self.inner.edges().map(move |(label, node)| {
                (label, LruOptimizedNode::new(node, Arc::clone(&metadata)))
            }),
        )
    }

    #[inline]
    fn edge_count(&self) -> Option<usize> {
        self.inner.edge_count()
    }
}

impl<N, V> MappedDictionaryNode for LruOptimizedNode<N>
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
