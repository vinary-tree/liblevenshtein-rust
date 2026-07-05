//! Version-tied cross-query result cache (Phase 6 novel optimization #4).
//!
//! Fuzzy query evaluation is dominated by the automaton state-transition inner
//! loop (`transition_state_pooled_ref` — ~80% of query CPU per AMD uProf). For
//! workloads that repeat queries — spell-checkers, autocomplete, batch
//! re-scans — a cache that memoizes `(query, max_distance) → results` skips that
//! loop entirely on a hit.
//!
//! Correctness for **dynamic** dictionaries is preserved by tying invalidation
//! to a monotonically-increasing **dictionary version**: the caller bumps the
//! version whenever the dictionary mutates, and the cache clears itself the next
//! time it observes a newer version. Cache hits therefore never return results
//! that predate a mutation.
//!
//! ```rust
//! use liblevenshtein::transducer::VersionedQueryCache;
//!
//! let mut cache: VersionedQueryCache<String> = VersionedQueryCache::new();
//! let mut version = 0u64;
//!
//! // First lookup misses and runs `compute`.
//! let a = cache.get_or_compute("foo", 2, version, || vec!["foo".to_string()]);
//! // Repeat hits the cache; `compute` is not called.
//! let b = cache.get_or_compute("foo", 2, version, || panic!("should not run"));
//! assert_eq!(a, b);
//!
//! // A dictionary mutation bumps the version and invalidates the cache.
//! version += 1;
//! let c = cache.get_or_compute("foo", 2, version, || vec!["foo".to_string(), "food".to_string()]);
//! assert_eq!(c.len(), 2);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

/// A cache of fuzzy-query results keyed by `(query, max_distance)` and scoped to
/// a dictionary version. Results are shared via `Arc<[V]>` so a hit is a
/// pointer clone, not a deep copy.
///
/// `V` is the query result type (e.g. `String`, or a `(term, distance)` pair).
#[derive(Debug, Clone)]
pub struct VersionedQueryCache<V> {
    /// The dictionary version these entries are valid for.
    version: u64,
    /// `(query, max_distance) → shared results`.
    entries: HashMap<(String, usize), Arc<[V]>>,
}

impl<V> Default for VersionedQueryCache<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> VersionedQueryCache<V> {
    /// Create an empty cache pinned to version 0.
    #[inline]
    pub fn new() -> Self {
        Self {
            version: 0,
            entries: HashMap::new(),
        }
    }

    /// Number of cached `(query, max_distance)` entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The dictionary version the cache is currently scoped to.
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Drop all entries (retaining the current version).
    #[inline]
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Invalidate the cache if `dict_version` is newer than what the cache was
    /// built against, adopting the new version. Returns `true` if it cleared.
    #[inline]
    fn reconcile_version(&mut self, dict_version: u64) -> bool {
        if dict_version != self.version {
            self.entries.clear();
            self.version = dict_version;
            true
        } else {
            false
        }
    }

    /// Look up `(query, max_distance)` at `dict_version`. If the version moved,
    /// the cache is invalidated first. On a miss, `compute` runs and its result
    /// is stored; on a hit, `compute` is **not** called. Returns the shared
    /// result slice (an `Arc` clone, so repeated hits are allocation-free apart
    /// from the lookup key).
    #[inline]
    pub fn get_or_compute<F>(
        &mut self,
        query: &str,
        max_distance: usize,
        dict_version: u64,
        compute: F,
    ) -> Arc<[V]>
    where
        F: FnOnce() -> Vec<V>,
    {
        self.reconcile_version(dict_version);
        // Two-phase to avoid allocating the owned key on the hit path.
        if let Some(hit) = self.entries.get(&(query.to_owned(), max_distance)) {
            return Arc::clone(hit);
        }
        let result: Arc<[V]> = compute().into();
        self.entries
            .insert((query.to_owned(), max_distance), Arc::clone(&result));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn hit_does_not_recompute_and_returns_same_results() {
        let mut cache: VersionedQueryCache<u32> = VersionedQueryCache::new();
        let calls = Cell::new(0);
        let a = cache.get_or_compute("q", 1, 0, || {
            calls.set(calls.get() + 1);
            vec![1, 2, 3]
        });
        let b = cache.get_or_compute("q", 1, 0, || {
            calls.set(calls.get() + 1);
            vec![9, 9]
        });
        assert_eq!(&*a, &[1, 2, 3]);
        assert_eq!(&*a, &*b); // hit returns the cached value, ignores the new closure
        assert_eq!(calls.get(), 1); // computed exactly once
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn distinct_keys_are_cached_separately() {
        let mut cache: VersionedQueryCache<u32> = VersionedQueryCache::new();
        let _ = cache.get_or_compute("q", 1, 0, || vec![1]);
        let _ = cache.get_or_compute("q", 2, 0, || vec![2, 2]); // same query, diff distance
        let _ = cache.get_or_compute("r", 1, 0, || vec![3, 3, 3]); // diff query
        assert_eq!(cache.len(), 3);
        assert_eq!(&*cache.get_or_compute("q", 2, 0, || vec![]), &[2, 2]);
    }

    #[test]
    fn version_bump_invalidates() {
        let mut cache: VersionedQueryCache<u32> = VersionedQueryCache::new();
        let _ = cache.get_or_compute("q", 1, 0, || vec![1]);
        assert_eq!(cache.len(), 1);
        // Newer dictionary version clears the stale entry before serving.
        let recomputed = cache.get_or_compute("q", 1, 1, || vec![1, 2]);
        assert_eq!(&*recomputed, &[1, 2]);
        assert_eq!(cache.version(), 1);
        assert_eq!(cache.len(), 1);
    }
}
