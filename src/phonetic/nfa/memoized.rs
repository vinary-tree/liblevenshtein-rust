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
//! reaches its maximum size. Cache keys are (query_string, max_distance)
//! pairs, and values are the matching results.
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::nfa::{MemoizedMatcherChar, ProductAutomatonChar};
//!
//! let product = ProductAutomatonChar::new(nfa, 2);
//! let mut cache = MemoizedMatcherChar::new(product, 1000);
//!
//! // First query computes result
//! let result1 = cache.accepts("phone");
//!
//! // Second query uses cached result
//! let result2 = cache.accepts("phone");
//! assert_eq!(result1, result2);
//! ```

use super::lazy_dfa::{LazyDFA, LazyDFAChar};
use super::product::{ProductAutomaton, ProductAutomatonChar};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

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
    lru_order: VecDeque<String>,
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
            lru_order: VecDeque::new(),
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
            self.update_lru(&key);
            return entry.result;
        }

        // Cache miss - compute result
        self.misses += 1;
        let result = self.product.accepts(input);

        // Store in cache
        self.insert_cache(key, CacheEntryChar {
            result,
            min_distance: None,
        });

        result
    }

    /// Get minimum distance, using cache if available.
    pub fn min_distance(&mut self, input: &str) -> Option<u8> {
        let key = input.to_string();

        // Check cache - clone to release borrow
        if let Some(entry) = self.cache.get(&key).cloned() {
            if entry.min_distance.is_some() {
                self.hits += 1;
                self.update_lru(&key);
                return entry.min_distance;
            }
        }

        // Compute min distance
        self.misses += 1;
        let min_dist = self.product.min_distance(input);
        let result = min_dist.is_some();

        // Update cache
        self.insert_cache(key, CacheEntryChar {
            result,
            min_distance: min_dist,
        });

        min_dist
    }

    /// Update LRU order for a key.
    fn update_lru(&mut self, key: &str) {
        // Remove from current position and add to front
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_front(key.to_string());
    }

    /// Insert into cache with LRU eviction.
    fn insert_cache(&mut self, key: String, entry: CacheEntryChar) {
        // Evict if at capacity
        while self.cache.len() >= self.max_cache_size && !self.lru_order.is_empty() {
            if let Some(evict_key) = self.lru_order.pop_back() {
                self.cache.remove(&evict_key);
            }
        }

        // Insert new entry
        self.cache.insert(key.clone(), entry);
        self.lru_order.push_front(key);
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
}

/// Memoized matcher for byte-level fuzzy regex.
#[derive(Debug)]
pub struct MemoizedMatcher {
    product: ProductAutomaton,
    cache: FxHashMap<Vec<u8>, CacheEntry>,
    lru_order: VecDeque<Vec<u8>>,
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
            lru_order: VecDeque::new(),
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
            self.update_lru(&key);
            return entry.result;
        }

        self.misses += 1;
        let result = self.product.accepts(input);

        self.insert_cache(key, CacheEntry {
            result,
            min_distance: None,
        });

        result
    }

    /// Get minimum distance.
    pub fn min_distance(&mut self, input: &[u8]) -> Option<u8> {
        let key = input.to_vec();

        if let Some(entry) = self.cache.get(&key).cloned() {
            if entry.min_distance.is_some() {
                self.hits += 1;
                self.update_lru(&key);
                return entry.min_distance;
            }
        }

        self.misses += 1;
        let min_dist = self.product.min_distance(input);
        let result = min_dist.is_some();

        self.insert_cache(key, CacheEntry {
            result,
            min_distance: min_dist,
        });

        min_dist
    }

    fn update_lru(&mut self, key: &[u8]) {
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_front(key.to_vec());
    }

    fn insert_cache(&mut self, key: Vec<u8>, entry: CacheEntry) {
        while self.cache.len() >= self.max_cache_size && !self.lru_order.is_empty() {
            if let Some(evict_key) = self.lru_order.pop_back() {
                self.cache.remove(&evict_key);
            }
        }

        self.cache.insert(key.clone(), entry);
        self.lru_order.push_front(key);
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
    lru_order: VecDeque<String>,
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
            lru_order: VecDeque::new(),
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
            self.update_lru(&key);
            return result;
        }

        self.misses += 1;
        let result = self.dfa.accepts(input);

        self.insert_cache(key, result);
        result
    }

    fn update_lru(&mut self, key: &str) {
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_front(key.to_string());
    }

    fn insert_cache(&mut self, key: String, result: bool) {
        while self.result_cache.len() >= self.max_cache_size && !self.lru_order.is_empty() {
            if let Some(evict_key) = self.lru_order.pop_back() {
                self.result_cache.remove(&evict_key);
            }
        }

        self.result_cache.insert(key.clone(), result);
        self.lru_order.push_front(key);
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
    lru_order: VecDeque<Vec<u8>>,
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
            lru_order: VecDeque::new(),
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
            self.update_lru(&key);
            return result;
        }

        self.misses += 1;
        let result = self.dfa.accepts(input);

        self.insert_cache(key, result);
        result
    }

    fn update_lru(&mut self, key: &[u8]) {
        if let Some(pos) = self.lru_order.iter().position(|k| k == key) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_front(key.to_vec());
    }

    fn insert_cache(&mut self, key: Vec<u8>, result: bool) {
        while self.result_cache.len() >= self.max_cache_size && !self.lru_order.is_empty() {
            if let Some(evict_key) = self.lru_order.pop_back() {
                self.result_cache.remove(&evict_key);
            }
        }

        self.result_cache.insert(key.clone(), result);
        self.lru_order.push_front(key);
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
    use crate::phonetic::nfa::nfa::{NFAChar, NFA};
    use crate::phonetic::regex::{parse, parse_bytes};

    #[test]
    fn test_memoized_matcher_accepts() {
        let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();
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
        let nfa = compile(&parse("phone").unwrap()).unwrap();
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
        let nfa = compile(&parse("test").unwrap()).unwrap();
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
    fn test_memoized_matcher_clear() {
        let nfa = compile(&parse("test").unwrap()).unwrap();
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
        let nfa = compile(&parse("hello").unwrap()).unwrap();
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
        let nfa = compile_bytes(&parse_bytes(b"test").unwrap()).unwrap();
        let product = ProductAutomaton::new(nfa, 1);
        let mut cache = MemoizedMatcher::new(product, 100);

        assert!(cache.accepts(b"test"));
        assert!(cache.accepts(b"test")); // Hit
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_memoized_lazy_dfa_bytes() {
        let nfa = compile_bytes(&parse_bytes(b"world").unwrap()).unwrap();
        let dfa = LazyDFA::new(nfa);
        let mut cache = MemoizedLazyDFA::new(dfa, 100);

        assert!(cache.accepts(b"world"));
        assert!(cache.accepts(b"world")); // Hit
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let nfa = compile(&parse("x").unwrap()).unwrap();
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
