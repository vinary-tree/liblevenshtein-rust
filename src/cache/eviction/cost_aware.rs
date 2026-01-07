//! Cost-aware eviction wrapper.
//!
//! This wrapper tracks age, size, and hit counts to make cost-based eviction
//! decisions. It balances regeneration cost against cache space.
//!
//! # Architecture
//!
//! Unlike the old `CostAwareStrategy` which required `CacheEntry<V>` metadata,
//! this wrapper maintains separate metadata tracking age, size, and hits.
//!
//! # Use Cases
//!
//! - AI code chat: Keep expensive LLM responses cached longer
//! - Documentation search: Evict large, rarely-hit results
//! - Error solutions: Balance between recomputation cost and cache space
//!
//! # Examples
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::dictionary::MappedDictionary;
//! use liblevenshtein::cache::eviction::CostAware;
//!
//! let dict = PathMapDictionary::from_terms_with_values([
//!     ("foo", 42),
//!     ("bar", 99),
//! ]);
//!
//! let cost_aware = CostAware::new(dict);
//! assert_eq!(cost_aware.get_value("foo"), Some(42));
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
    inserted_at: Instant,
    hit_count: u32,
    size_bytes: usize,
}

impl EntryMetadata {
    fn new(size: usize) -> Self {
        Self {
            inserted_at: Instant::now(),
            hit_count: 1, // First access
            size_bytes: size,
        }
    }

    fn increment(&mut self) {
        self.hit_count = self.hit_count.saturating_add(1);
    }

    fn age(&self) -> std::time::Duration {
        self.inserted_at.elapsed()
    }

    /// Cost-aware score: (age * size) / (hits + 1)
    /// Higher score = more likely to evict
    fn cost_score(&self) -> f64 {
        let age = self.age().as_secs_f64();
        let size = self.size_bytes as f64;
        let hits = self.hit_count as f64;
        (age * size) / (hits + 1.0)
    }
}

/// Cost-aware eviction wrapper.
///
/// Evicts entries based on cost-to-value ratio. Entries with high age,
/// large size, and low hit count are evicted first.
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
/// use liblevenshtein::cache::eviction::CostAware;
///
/// let dict = PathMapDictionary::from_terms_with_values([
///     ("hello", 1),
///     ("world", 2),
/// ]);
///
/// let cost_aware = CostAware::new(dict);
/// assert_eq!(cost_aware.get_value("hello"), Some(1));
/// ```
#[derive(Clone)]
pub struct CostAware<D> {
    inner: D,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<D> CostAware<D> {
    /// Creates a new CostAware wrapper.
    ///
    /// # Arguments
    ///
    /// - `dict`: The dictionary to wrap
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use liblevenshtein::prelude::*;
    /// use liblevenshtein::cache::eviction::CostAware;
    ///
    /// let dict = PathMapDictionary::from_terms_with_values([
    ///     ("foo", 42),
    /// ]);
    ///
    /// let cost_aware = CostAware::new(dict);
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
        let mut metadata = self.metadata.write().unwrap();
        metadata
            .entry(term.to_string())
            .and_modify(|m| m.increment())
            .or_insert_with(|| EntryMetadata::new(size));
    }

    /// Gets the cost score for an entry.
    ///
    /// Returns `None` if the entry has never been accessed.
    pub fn cost_score(&self, term: &str) -> Option<f64> {
        let metadata = self.metadata.read().unwrap();
        metadata.get(term).map(|m| m.cost_score())
    }

    /// Finds the highest cost entry among the given terms.
    ///
    /// Returns the term with the highest cost score (most likely to evict).
    pub fn find_highest_cost(&self, terms: &[&str]) -> Option<String> {
        let metadata = self.metadata.read().unwrap();
        terms
            .iter()
            .filter_map(|&term| metadata.get(term).map(|m| (term, m.cost_score())))
            .max_by(|(_, score1), (_, score2)| {
                score1
                    .partial_cmp(score2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(term, _)| term.to_string())
    }

    /// Evicts the highest cost entry from metadata.
    ///
    /// Returns the evicted term if any.
    pub fn evict_highest_cost(&self, terms: &[&str]) -> Option<String> {
        if let Some(high_cost_term) = self.find_highest_cost(terms) {
            let mut metadata = self.metadata.write().unwrap();
            metadata.remove(&high_cost_term);
            Some(high_cost_term)
        } else {
            None
        }
    }

    /// Clears all metadata.
    pub fn clear_metadata(&self) {
        let mut metadata = self.metadata.write().unwrap();
        metadata.clear();
    }
}

impl<D> Dictionary for CostAware<D>
where
    D: Dictionary,
{
    type Node = CostAwareNode<D::Node>;

    #[inline]
    fn root(&self) -> Self::Node {
        CostAwareNode::new(self.inner.root(), Arc::clone(&self.metadata))
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

impl<D, V> MappedDictionary for CostAware<D>
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

/// Node wrapper for CostAware dictionary.
#[derive(Clone)]
pub struct CostAwareNode<N> {
    inner: N,
    metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>,
}

impl<N> CostAwareNode<N> {
    fn new(inner: N, metadata: Arc<RwLock<HashMap<String, EntryMetadata>>>) -> Self {
        Self { inner, metadata }
    }
}

impl<N> DictionaryNode for CostAwareNode<N>
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
            .map(|node| CostAwareNode::new(node, Arc::clone(&self.metadata)))
    }

    #[inline]
    fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
        let metadata = Arc::clone(&self.metadata);
        Box::new(
            self.inner
                .edges()
                .map(move |(label, node)| (label, CostAwareNode::new(node, Arc::clone(&metadata)))),
        )
    }

    #[inline]
    fn edge_count(&self) -> Option<usize> {
        self.inner.edge_count()
    }
}

impl<N, V> MappedDictionaryNode for CostAwareNode<N>
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
    use std::thread;
    use std::time::Duration;

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_cost_aware_wrapper_basic() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let cost_aware = CostAware::new(dict);

        // Values should be accessible
        assert_eq!(cost_aware.get_value("foo"), Some(42));
        assert_eq!(cost_aware.get_value("bar"), Some(99));
        assert!(cost_aware.contains("foo"));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_cost_aware_scoring() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let cost_aware = CostAware::new(dict);

        // Access foo once
        assert_eq!(cost_aware.get_value("foo"), Some(42));
        let score1 = cost_aware.cost_score("foo").unwrap();

        // Wait to increase age
        thread::sleep(Duration::from_millis(10));

        // Score should increase with age
        let score2 = cost_aware.cost_score("foo").unwrap();
        assert!(score2 > score1);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_cost_aware_hits_lower_score() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42)]);

        let cost_aware = CostAware::new(dict);

        // First access
        assert_eq!(cost_aware.get_value("foo"), Some(42));
        thread::sleep(Duration::from_millis(10));
        let score1 = cost_aware.cost_score("foo").unwrap();

        // Multiple additional accesses
        for _ in 0..5 {
            assert_eq!(cost_aware.get_value("foo"), Some(42));
        }

        // More hits should lower the cost score (despite increased age)
        let score2 = cost_aware.cost_score("foo").unwrap();
        assert!(score2 < score1);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_cost_aware_find_highest_cost() {
        let dict =
            PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99), ("baz", 123)]);

        let cost_aware = CostAware::new(dict);

        // Access with different patterns
        // foo: 1 access, oldest
        assert_eq!(cost_aware.get_value("foo"), Some(42));
        thread::sleep(Duration::from_millis(10));

        // bar: 3 accesses, middle age
        assert_eq!(cost_aware.get_value("bar"), Some(99));
        assert_eq!(cost_aware.get_value("bar"), Some(99));
        assert_eq!(cost_aware.get_value("bar"), Some(99));
        thread::sleep(Duration::from_millis(10));

        // baz: 5 accesses, newest
        for _ in 0..5 {
            assert_eq!(cost_aware.get_value("baz"), Some(123));
        }

        // foo should have highest cost (oldest, fewest hits)
        let highest = cost_aware.find_highest_cost(&["foo", "bar", "baz"]);
        assert_eq!(highest, Some("foo".to_string()));
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_cost_aware_eviction() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let cost_aware = CostAware::new(dict);

        // Access both
        assert_eq!(cost_aware.get_value("foo"), Some(42));
        assert_eq!(cost_aware.get_value("bar"), Some(99));

        // Evict highest cost
        let evicted = cost_aware.evict_highest_cost(&["foo", "bar"]);
        assert!(evicted.is_some());

        // Evicted term should have no metadata
        assert_eq!(cost_aware.cost_score(&evicted.unwrap()), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_cost_aware_clear_metadata() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42), ("bar", 99)]);

        let cost_aware = CostAware::new(dict);

        // Access both
        assert_eq!(cost_aware.get_value("foo"), Some(42));
        assert_eq!(cost_aware.get_value("bar"), Some(99));

        // Clear metadata
        cost_aware.clear_metadata();

        // No cost scores
        assert_eq!(cost_aware.cost_score("foo"), None);
        assert_eq!(cost_aware.cost_score("bar"), None);
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_cost_aware_node_traversal() {
        let dict = PathMapDictionary::<()>::from_terms(["hello", "help"]);
        let cost_aware = CostAware::new(dict);

        let root = cost_aware.root();
        assert!(!root.is_final());

        // Traverse 'h' -> 'e' -> 'l' -> 'p'
        let h = root.transition(b'h').unwrap();
        let e = h.transition(b'e').unwrap();
        let l = e.transition(b'l').unwrap();
        let p = l.transition(b'p').unwrap();

        assert!(p.is_final()); // "help"
    }

    #[test]
    #[cfg(feature = "pathmap-backend")]
    fn test_cost_aware_into_inner() {
        let dict = PathMapDictionary::from_terms_with_values([("foo", 42)]);
        let cost_aware = CostAware::new(dict);
        let original = cost_aware.into_inner();

        assert_eq!(original.len(), Some(1));
        assert_eq!(original.get_value("foo"), Some(42));
    }
}
