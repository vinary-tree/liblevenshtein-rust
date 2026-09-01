//! Memoized caching for repeated NFA/DFA queries.
//!
//! This module provides caching infrastructure for fuzzy regex queries,
//! particularly useful when querying the same patterns repeatedly with
//! similar inputs.
//!
//! # Use Cases
//!
//! - **Spell checking**: Same words queried multiple times
//! - **Autocomplete**: Prefix queries with overlapping inputs
//! - **Search**: Repeated queries in interactive sessions
//!
//! # Design
//!
//! The cache uses LRU (Least Recently Used) eviction when the cache
//! reaches its maximum size. Each matcher owns one product automaton—and
//! therefore one maximum edit distance—so cache keys are query strings and
//! values are the matching results (plus an optional computed distance).
//!
//! # Examples
//!
//! ```rust
//! use liblevenshtein::phonetic::nfa::{
//!     compile, MemoizedMatcherChar, ProductAutomatonChar,
//! };
//! use liblevenshtein::phonetic::regex::parse;
//!
//! let regex = parse("(ph|f)one").expect("doc: regex parse must succeed");
//! let nfa = compile(&regex).expect("doc: NFA compilation must succeed");
//! let product = ProductAutomatonChar::new(nfa, 2);
//! let mut cache = MemoizedMatcherChar::new(product, 1000);
//!
//! // First query computes result
//! let result1 = cache.accepts("phone");
//! assert!(result1);
//! assert_eq!(cache.stats().misses, 1);
//!
//! // Second query uses cached result
//! let result2 = cache.accepts("phone");
//! assert_eq!(result1, result2);
//! let stats = cache.stats();
//! assert_eq!(stats.size, 1);
//! assert_eq!(stats.hits, 1);
//! assert_eq!(stats.misses, 1);
//! ```

use super::lazy_dfa::{LazyDFA, LazyDFAChar};
use super::product::{ProductAutomaton, ProductAutomatonChar};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::hash::Hash;

#[derive(Debug)]
struct LruTracker<K> {
    order: VecDeque<(K, u64)>,
    generations: FxHashMap<K, u64>,
    clock: u64,
}

impl<K> LruTracker<K>
where
    K: Clone + Eq + Hash,
{
    fn new() -> Self {
        Self {
            order: VecDeque::new(),
            generations: FxHashMap::default(),
            clock: 0,
        }
    }

    fn touch(&mut self, key: K, capacity: usize) {
        if self.clock == u64::MAX {
            self.rebase_generations();
        }
        self.clock += 1;
        let generation = self.clock;
        self.generations.insert(key.clone(), generation);
        self.order.push_front((key, generation));
        self.compact_if_sparse(capacity);
    }

    fn evict_until_below<V>(&mut self, cache: &mut FxHashMap<K, V>, capacity: usize) {
        if capacity == 0 {
            cache.clear();
            self.clear();
            return;
        }

        while cache.len() >= capacity {
            let Some((key, generation)) = self.order.pop_back() else {
                cache.clear();
                self.generations.clear();
                return;
            };

            if self.generations.get(&key).copied() == Some(generation) {
                self.generations.remove(&key);
                cache.remove(&key);
            }
        }
    }

    fn clear(&mut self) {
        self.order.clear();
        self.generations.clear();
        self.clock = 0;
    }

    fn compact_if_sparse(&mut self, capacity: usize) {
        let live_len = self.generations.len();
        let compact_threshold = capacity.max(live_len).saturating_mul(2).max(64);
        if self.order.len() <= compact_threshold {
            return;
        }

        let mut compacted = VecDeque::with_capacity(live_len);
        for (key, generation) in self.order.drain(..) {
            if self.generations.get(&key).copied() == Some(generation) {
                compacted.push_back((key, generation));
            }
        }
        self.order = compacted;
    }

    fn rebase_generations(&mut self) {
        let mut next_generation = 0;
        let mut compacted = VecDeque::with_capacity(self.generations.len());
        let mut generations = FxHashMap::default();

        for (key, generation) in self.order.drain(..).rev() {
            if self.generations.get(&key).copied() == Some(generation) {
                next_generation += 1;
                generations.insert(key.clone(), next_generation);
                compacted.push_front((key, next_generation));
            }
        }

        self.order = compacted;
        self.generations = generations;
        self.clock = next_generation;
    }
}

// ============================================================================
// Character-level Memoized Matcher
// ============================================================================

/// Cache entry for memoized results.
#[derive(Debug, Clone)]
struct CacheEntryChar {
    /// The cached result
    result: bool,
    /// Minimum distance (if computed)
    min_distance: Option<u8>,
    min_distance_computed: bool,
}

/// Memoized matcher for character-level fuzzy regex.
///
/// Wraps a `ProductAutomatonChar` with a caching layer for efficient
/// repeated queries.
#[derive(Debug)]
pub struct MemoizedMatcherChar {
    /// The underlying product automaton
    product: ProductAutomatonChar,
    /// Cache: query -> result
    cache: FxHashMap<String, CacheEntryChar>,
    /// LRU order for eviction
    lru_order: LruTracker<String>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Cache hit count
    hits: usize,
    /// Cache miss count
    misses: usize,
}

impl MemoizedMatcherChar {
    /// Create a new memoized matcher with the given cache size.
    pub fn new(product: ProductAutomatonChar, max_cache_size: usize) -> Self {
        Self {
            product,
            cache: FxHashMap::default(),
            lru_order: LruTracker::new(),
            max_cache_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Check if input is accepted, using cache if available.
    pub fn accepts(&mut self, input: &str) -> bool {
        let key = input.to_string();

        // Check cache - get result first, then update LRU
        if let Some(entry) = self.cache.get(&key).cloned() {
            self.hits += 1;
            self.lru_order.touch(key, self.max_cache_size);
            return entry.result;
        }

        // Cache miss - compute result
        self.misses += 1;
        let result = self.product.accepts(input);

        // Store in cache
        self.insert_cache(
            key,
            CacheEntryChar {
                result,
                min_distance: None,
                min_distance_computed: false,
            },
        );

        result
    }

    /// Get minimum distance, using cache if available.
    pub fn min_distance(&mut self, input: &str) -> Option<u8> {
        let key = input.to_string();

        // Check cache - clone to release borrow
        if let Some(entry) = self.cache.get(&key).cloned() {
            if entry.min_distance_computed {
                self.hits += 1;
                self.lru_order.touch(key, self.max_cache_size);
                return entry.min_distance;
            }
        }

        // Compute min distance
        self.misses += 1;
        let min_dist = self.product.min_distance(input);
        let result = min_dist.is_some();

        // Update cache
        self.insert_cache(
            key,
            CacheEntryChar {
                result,
                min_distance: min_dist,
                min_distance_computed: true,
            },
        );

        min_dist
    }

    /// Insert into cache with LRU eviction.
    fn insert_cache(&mut self, key: String, entry: CacheEntryChar) {
        if self.max_cache_size == 0 {
            return;
        }

        // Evict if at capacity
        if !self.cache.contains_key(&key) {
            self.lru_order
                .evict_until_below(&mut self.cache, self.max_cache_size);
        }

        // Insert new entry
        self.cache.insert(key.clone(), entry);
        self.lru_order.touch(key, self.max_cache_size);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> MemoizedStats {
        MemoizedStats {
            size: self.cache.len(),
            max_size: self.max_cache_size,
            hits: self.hits,
            misses: self.misses,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get the underlying product automaton.
    pub fn product(&self) -> &ProductAutomatonChar {
        &self.product
    }
}

// ============================================================================
// Byte-level Memoized Matcher
// ============================================================================

/// Cache entry for byte-level results.
#[derive(Debug, Clone)]
struct CacheEntry {
    result: bool,
    min_distance: Option<u8>,
    min_distance_computed: bool,
}

/// Memoized matcher for byte-level fuzzy regex.
#[derive(Debug)]
pub struct MemoizedMatcher {
    product: ProductAutomaton,
    cache: FxHashMap<Vec<u8>, CacheEntry>,
    lru_order: LruTracker<Vec<u8>>,
    max_cache_size: usize,
    hits: usize,
    misses: usize,
}

impl MemoizedMatcher {
    /// Create a new memoized matcher.
    pub fn new(product: ProductAutomaton, max_cache_size: usize) -> Self {
        Self {
            product,
            cache: FxHashMap::default(),
            lru_order: LruTracker::new(),
            max_cache_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Check if input is accepted.
    pub fn accepts(&mut self, input: &[u8]) -> bool {
        let key = input.to_vec();

        if let Some(entry) = self.cache.get(&key).cloned() {
            self.hits += 1;
            self.lru_order.touch(key, self.max_cache_size);
            return entry.result;
        }

        self.misses += 1;
        let result = self.product.accepts(input);

        self.insert_cache(
            key,
            CacheEntry {
                result,
                min_distance: None,
                min_distance_computed: false,
            },
        );

        result
    }

    /// Get minimum distance.
    pub fn min_distance(&mut self, input: &[u8]) -> Option<u8> {
        let key = input.to_vec();

        if let Some(entry) = self.cache.get(&key).cloned() {
            if entry.min_distance_computed {
                self.hits += 1;
                self.lru_order.touch(key, self.max_cache_size);
                return entry.min_distance;
            }
        }

        self.misses += 1;
        let min_dist = self.product.min_distance(input);
        let result = min_dist.is_some();

        self.insert_cache(
            key,
            CacheEntry {
                result,
                min_distance: min_dist,
                min_distance_computed: true,
            },
        );

        min_dist
    }

    fn insert_cache(&mut self, key: Vec<u8>, entry: CacheEntry) {
        if self.max_cache_size == 0 {
            return;
        }

        if !self.cache.contains_key(&key) {
            self.lru_order
                .evict_until_below(&mut self.cache, self.max_cache_size);
        }

        self.cache.insert(key.clone(), entry);
        self.lru_order.touch(key, self.max_cache_size);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> MemoizedStats {
        MemoizedStats {
            size: self.cache.len(),
            max_size: self.max_cache_size,
            hits: self.hits,
            misses: self.misses,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get the underlying product automaton.
    pub fn product(&self) -> &ProductAutomaton {
        &self.product
    }
}

// ============================================================================
// Memoized Lazy DFA
// ============================================================================

/// Memoized wrapper for character-level lazy DFA.
///
/// Caches complete accept/reject decisions for strings.
#[derive(Debug)]
pub struct MemoizedLazyDFAChar {
    dfa: LazyDFAChar,
    result_cache: FxHashMap<String, bool>,
    lru_order: LruTracker<String>,
    max_cache_size: usize,
    hits: usize,
    misses: usize,
}

impl MemoizedLazyDFAChar {
    /// Create a new memoized lazy DFA.
    pub fn new(dfa: LazyDFAChar, max_cache_size: usize) -> Self {
        Self {
            dfa,
            result_cache: FxHashMap::default(),
            lru_order: LruTracker::new(),
            max_cache_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Check if input is accepted.
    pub fn accepts(&mut self, input: &str) -> bool {
        let key = input.to_string();

        if let Some(&result) = self.result_cache.get(&key) {
            self.hits += 1;
            self.lru_order.touch(key, self.max_cache_size);
            return result;
        }

        self.misses += 1;
        let result = self.dfa.accepts(input);

        self.insert_cache(key, result);
        result
    }

    fn insert_cache(&mut self, key: String, result: bool) {
        if self.max_cache_size == 0 {
            return;
        }

        if !self.result_cache.contains_key(&key) {
            self.lru_order
                .evict_until_below(&mut self.result_cache, self.max_cache_size);
        }

        self.result_cache.insert(key.clone(), result);
        self.lru_order.touch(key, self.max_cache_size);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> MemoizedStats {
        MemoizedStats {
            size: self.result_cache.len(),
            max_size: self.max_cache_size,
            hits: self.hits,
            misses: self.misses,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all caches (both result cache and DFA transition cache).
    pub fn clear(&mut self) {
        self.result_cache.clear();
        self.lru_order.clear();
        self.dfa.clear_cache();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get the underlying lazy DFA.
    pub fn dfa(&self) -> &LazyDFAChar {
        &self.dfa
    }

    /// Get mutable access to the underlying lazy DFA.
    pub fn dfa_mut(&mut self) -> &mut LazyDFAChar {
        &mut self.dfa
    }
}

/// Memoized wrapper for byte-level lazy DFA.
#[derive(Debug)]
pub struct MemoizedLazyDFA {
    dfa: LazyDFA,
    result_cache: FxHashMap<Vec<u8>, bool>,
    lru_order: LruTracker<Vec<u8>>,
    max_cache_size: usize,
    hits: usize,
    misses: usize,
}

impl MemoizedLazyDFA {
    /// Create a new memoized lazy DFA.
    pub fn new(dfa: LazyDFA, max_cache_size: usize) -> Self {
        Self {
            dfa,
            result_cache: FxHashMap::default(),
            lru_order: LruTracker::new(),
            max_cache_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Check if input is accepted.
    pub fn accepts(&mut self, input: &[u8]) -> bool {
        let key = input.to_vec();

        if let Some(&result) = self.result_cache.get(&key) {
            self.hits += 1;
            self.lru_order.touch(key, self.max_cache_size);
            return result;
        }

        self.misses += 1;
        let result = self.dfa.accepts(input);

        self.insert_cache(key, result);
        result
    }

    fn insert_cache(&mut self, key: Vec<u8>, result: bool) {
        if self.max_cache_size == 0 {
            return;
        }

        if !self.result_cache.contains_key(&key) {
            self.lru_order
                .evict_until_below(&mut self.result_cache, self.max_cache_size);
        }

        self.result_cache.insert(key.clone(), result);
        self.lru_order.touch(key, self.max_cache_size);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> MemoizedStats {
        MemoizedStats {
            size: self.result_cache.len(),
            max_size: self.max_cache_size,
            hits: self.hits,
            misses: self.misses,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all caches.
    pub fn clear(&mut self) {
        self.result_cache.clear();
        self.lru_order.clear();
        self.dfa.clear_cache();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get the underlying lazy DFA.
    pub fn dfa(&self) -> &LazyDFA {
        &self.dfa
    }

    /// Get mutable access to the underlying lazy DFA.
    pub fn dfa_mut(&mut self) -> &mut LazyDFA {
        &mut self.dfa
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for memoized caches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoizedStats {
    /// Current cache size
    pub size: usize,
    /// Maximum cache size
    pub max_size: usize,
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::nfa::compiler::{compile, compile_bytes};
    use crate::phonetic::regex::{parse, parse_bytes};

    #[test]
    fn test_memoized_matcher_accepts() {
        let nfa = compile(&parse("(ph|f)one").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let product = ProductAutomatonChar::new(nfa, 1);
        let mut cache = MemoizedMatcherChar::new(product, 100);

        // First query - miss
        assert!(cache.accepts("phone"));
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Second query - hit
        assert!(cache.accepts("phone"));
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert!(stats.hit_rate > 0.4); // ~50%
    }

    #[test]
    fn test_memoized_matcher_min_distance() {
        let nfa = compile(&parse("phone").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let product = ProductAutomatonChar::new(nfa, 2);
        let mut cache = MemoizedMatcherChar::new(product, 100);

        assert_eq!(cache.min_distance("phone"), Some(0));
        assert_eq!(cache.min_distance("phon"), Some(1));

        // Second query should hit
        assert_eq!(cache.min_distance("phone"), Some(0));
        let stats = cache.stats();
        assert!(stats.hits >= 1);
    }

    #[test]
    fn test_memoized_matcher_lru_eviction() {
        let nfa = compile(&parse("test").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let product = ProductAutomatonChar::new(nfa, 1);
        let mut cache = MemoizedMatcherChar::new(product, 3);

        // Fill cache
        cache.accepts("a");
        cache.accepts("b");
        cache.accepts("c");
        assert_eq!(cache.stats().size, 3);

        // Add one more - should evict "a"
        cache.accepts("d");
        assert_eq!(cache.stats().size, 3);

        // "a" should be evicted, so this is a miss
        let hits_before = cache.stats().hits;
        cache.accepts("a");
        assert_eq!(cache.stats().hits, hits_before); // No new hit
    }

    #[test]
    fn mixed_accepts_and_min_distance_preserve_recent_lru_entry() {
        let nfa = compile(&parse("a").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let product = ProductAutomatonChar::new(nfa, 1);
        let mut cache = MemoizedMatcherChar::new(product, 2);

        assert!(cache.accepts("a"));
        assert_eq!(cache.min_distance("a"), Some(0));
        assert!(cache.accepts("b"));
        assert!(cache.accepts("a"));
        let hits_before_insert = cache.stats().hits;

        assert!(cache.accepts("c"));
        assert!(cache.accepts("a"));
        assert_eq!(cache.stats().hits, hits_before_insert + 1);

        let hits_before_evicted = cache.stats().hits;
        assert!(cache.accepts("b"));
        assert_eq!(cache.stats().hits, hits_before_evicted);
    }

    #[test]
    fn lru_tracker_rebases_saturated_generation_counter() {
        let mut tracker = LruTracker::new();
        let mut cache = FxHashMap::default();

        cache.insert("a".to_string(), ());
        tracker.touch("a".to_string(), 2);
        cache.insert("b".to_string(), ());
        tracker.touch("b".to_string(), 2);

        tracker.clock = u64::MAX;
        tracker.evict_until_below(&mut cache, 2);
        cache.insert("c".to_string(), ());
        tracker.touch("c".to_string(), 2);

        assert!(!cache.contains_key("a"));
        assert!(cache.contains_key("b"));
        assert!(cache.contains_key("c"));
        assert_eq!(tracker.generations.len(), 2);
        assert_eq!(tracker.clock, 2);
    }

    #[test]
    fn min_distance_caches_absent_results() {
        let nfa = compile(&parse("phone").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let product = ProductAutomatonChar::new(nfa, 1);
        let mut cache = MemoizedMatcherChar::new(product, 8);

        assert_eq!(cache.min_distance("zzzz"), None);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.min_distance("zzzz"), None);

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.size, 1);
    }

    #[test]
    fn test_memoized_matcher_clear() {
        let nfa = compile(&parse("test").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let product = ProductAutomatonChar::new(nfa, 1);
        let mut cache = MemoizedMatcherChar::new(product, 100);

        cache.accepts("a");
        cache.accepts("b");
        assert!(cache.stats().size > 0);

        cache.clear();
        let stats = cache.stats();
        assert_eq!(stats.size, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_memoized_lazy_dfa() {
        let nfa = compile(&parse("hello").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let dfa = LazyDFAChar::new(nfa);
        let mut cache = MemoizedLazyDFAChar::new(dfa, 100);

        // First query - miss
        assert!(cache.accepts("hello"));
        assert_eq!(cache.stats().misses, 1);

        // Second query - hit
        assert!(cache.accepts("hello"));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_memoized_bytes() {
        let nfa = compile_bytes(&parse_bytes(b"test").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let product = ProductAutomaton::new(nfa, 1);
        let mut cache = MemoizedMatcher::new(product, 100);

        assert!(cache.accepts(b"test"));
        assert!(cache.accepts(b"test")); // Hit
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn byte_min_distance_caches_absent_results() {
        let nfa = compile_bytes(&parse_bytes(b"phone").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let product = ProductAutomaton::new(nfa, 1);
        let mut cache = MemoizedMatcher::new(product, 8);

        assert_eq!(cache.min_distance(b"zzzz"), None);
        assert_eq!(cache.min_distance(b"zzzz"), None);

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.size, 1);
    }

    #[test]
    fn test_memoized_lazy_dfa_bytes() {
        let nfa = compile_bytes(&parse_bytes(b"world").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let dfa = LazyDFA::new(nfa);
        let mut cache = MemoizedLazyDFA::new(dfa, 100);

        assert!(cache.accepts(b"world"));
        assert!(cache.accepts(b"world")); // Hit
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn zero_capacity_caches_do_not_store_entries() {
        let nfa = compile(&parse("x").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let product = ProductAutomatonChar::new(nfa.clone(), 0);
        let mut matcher = MemoizedMatcherChar::new(product, 0);

        assert!(matcher.accepts("x"));
        assert!(matcher.accepts("x"));
        assert_eq!(matcher.stats().size, 0);
        assert_eq!(matcher.stats().hits, 0);
        assert_eq!(matcher.stats().misses, 2);

        let mut lazy = MemoizedLazyDFAChar::new(LazyDFAChar::new(nfa), 0);
        assert!(lazy.accepts("x"));
        assert!(lazy.accepts("x"));
        assert_eq!(lazy.stats().size, 0);
        assert_eq!(lazy.stats().hits, 0);
        assert_eq!(lazy.stats().misses, 2);

        let byte_nfa = compile_bytes(&parse_bytes(b"x").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let byte_product = ProductAutomaton::new(byte_nfa.clone(), 0);
        let mut byte_matcher = MemoizedMatcher::new(byte_product, 0);

        assert!(byte_matcher.accepts(b"x"));
        assert!(byte_matcher.accepts(b"x"));
        assert_eq!(byte_matcher.stats().size, 0);
        assert_eq!(byte_matcher.stats().hits, 0);
        assert_eq!(byte_matcher.stats().misses, 2);

        let mut byte_lazy = MemoizedLazyDFA::new(LazyDFA::new(byte_nfa), 0);
        assert!(byte_lazy.accepts(b"x"));
        assert!(byte_lazy.accepts(b"x"));
        assert_eq!(byte_lazy.stats().size, 0);
        assert_eq!(byte_lazy.stats().hits, 0);
        assert_eq!(byte_lazy.stats().misses, 2);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let nfa = compile(&parse("x").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let product = ProductAutomatonChar::new(nfa, 0);
        let mut cache = MemoizedMatcherChar::new(product, 100);

        // 1 miss
        cache.accepts("a");
        // 3 hits
        cache.accepts("a");
        cache.accepts("a");
        cache.accepts("a");

        let stats = cache.stats();
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.75).abs() < 0.001);
    }
}
