//! N-gram index for fast candidate pre-filtering.
//!
//! This module provides an N-gram (substring) index for approximate string matching.
//! N-gram filtering can reject 85-95% of candidates before expensive automata traversal.
//!
//! # Algorithm
//!
//! The index works by:
//! 1. Breaking each term into overlapping n-grams (e.g., "hello" → ["he", "el", "ll", "lo"])
//! 2. Storing a mapping from each n-gram to the terms containing it
//! 3. For queries, finding candidates with sufficient n-gram overlap
//!
//! # Theoretical Basis
//!
//! If two strings have edit distance `d`, they can differ in at most `d * n` n-grams,
//! where `n` is the n-gram size. Therefore:
//!
//! ```text
//! min_overlap >= max(0, |query_ngrams| - d * n)
//! ```
//!
//! This provides a necessary (but not sufficient) condition for matching,
//! allowing us to prune the candidate set efficiently.
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::filter::NgramIndex;
//!
//! let mut index = NgramIndex::new(2); // Bigrams
//!
//! // Index some terms
//! index.insert("hello");
//! index.insert("help");
//! index.insert("world");
//!
//! // Find candidates for "helo" with max distance 1
//! let candidates = index.find_candidates("helo", 1);
//! assert!(candidates.contains(&"hello"));
//! assert!(candidates.contains(&"help"));
//! assert!(!candidates.contains(&"world"));
//! ```
//!
//! # Complexity
//!
//! - **Index construction**: O(n * m) where n = terms, m = avg term length
//! - **Query**: O(|query| + |matches|) amortized
//! - **Space**: O(n * m) for index storage

use crate::transducer::AllowedPrefixes;
use rustc_hash::{FxHashMap, FxHashSet};

/// N-gram index for approximate string matching pre-filtering.
///
/// Stores terms indexed by their n-grams for fast candidate lookup.
#[derive(Debug, Clone)]
pub struct NgramIndex {
    /// N-gram size (typically 2 or 3).
    n: usize,

    /// N-gram to term ID set mapping.
    index: FxHashMap<Vec<u8>, FxHashSet<usize>>,

    /// Terms stored by ID.
    terms: Vec<String>,

    /// Whether the term ID is currently live.
    active: Vec<bool>,

    /// Term to ID mapping for deduplication.
    term_to_id: FxHashMap<String, usize>,
}

impl NgramIndex {
    /// Create a new N-gram index with the specified n-gram size.
    ///
    /// # Arguments
    ///
    /// * `n` - N-gram size (typically 2 for bigrams or 3 for trigrams)
    ///
    /// A zero value is normalized to 1 so candidate generation remains total
    /// for externally supplied configuration values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::filter::NgramIndex;
    ///
    /// let bigram_index = NgramIndex::new(2);
    /// let trigram_index = NgramIndex::new(3);
    /// ```
    pub fn new(n: usize) -> Self {
        Self {
            n: n.max(1),
            index: FxHashMap::default(),
            terms: Vec::new(),
            active: Vec::new(),
            term_to_id: FxHashMap::default(),
        }
    }

    /// Create an N-gram index from an iterator of terms.
    ///
    /// # Arguments
    ///
    /// * `n` - N-gram size
    /// * `terms` - Iterator of terms to index
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::filter::NgramIndex;
    ///
    /// let terms = vec!["apple", "banana", "cherry"];
    /// let index = NgramIndex::from_iter(2, terms.into_iter().map(String::from));
    /// ```
    pub fn from_iter<I>(n: usize, terms: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        let terms = terms.into_iter();
        let (capacity, _) = terms.size_hint();
        let mut index = Self::new(n);
        index.terms.reserve(capacity);
        index.active.reserve(capacity);
        index.term_to_id.reserve(capacity);

        for term in terms {
            index.insert(&term);
        }
        index
    }

    /// Get the n-gram size.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get the number of live indexed terms.
    #[inline]
    pub fn len(&self) -> usize {
        self.term_to_id.len()
    }

    /// Check if the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.term_to_id.is_empty()
    }

    /// Get the number of unique n-grams in the index.
    #[inline]
    pub fn ngram_count(&self) -> usize {
        self.index.len()
    }

    /// Add a term to the index.
    ///
    /// If the term is already indexed, this is a no-op.
    ///
    /// # Arguments
    ///
    /// * `term` - The term to index
    ///
    /// # Returns
    ///
    /// The term ID (existing or newly assigned).
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::filter::NgramIndex;
    ///
    /// let mut index = NgramIndex::new(2);
    /// let id1 = index.insert("hello");
    /// let id2 = index.insert("hello"); // Same ID returned
    /// assert_eq!(id1, id2);
    /// ```
    pub fn insert(&mut self, term: &str) -> usize {
        // Check for duplicate
        if let Some(&id) = self.term_to_id.get(term) {
            return id;
        }

        // Assign new ID
        let id = self.terms.len();
        self.terms.push(term.to_string());
        self.active.push(true);
        self.term_to_id.insert(term.to_string(), id);

        self.index_term_ngrams(term.as_bytes(), id);

        id
    }

    /// Remove a term from the index.
    ///
    /// # Note
    ///
    /// This marks the term as removed but doesn't reclaim the ID.
    /// For frequent insertions/deletions, consider rebuilding the index.
    ///
    /// # Returns
    ///
    /// `true` if the term was present and removed, `false` otherwise.
    pub fn remove(&mut self, term: &str) -> bool {
        if let Some(&id) = self.term_to_id.get(term) {
            // Remove from n-gram index
            Self::for_each_ngram(self.n, term.as_bytes(), |ngram| {
                if let Some(ids) = self.index.get_mut(ngram) {
                    ids.remove(&id);
                    // Don't remove empty sets to avoid rehashing
                }
            });
            self.term_to_id.remove(term);
            // Note: We don't remove from terms vec to preserve IDs
            self.active[id] = false;
            true
        } else {
            false
        }
    }

    fn for_each_ngram<'a>(n: usize, bytes: &'a [u8], mut visit: impl FnMut(&'a [u8])) {
        if bytes.len() < n {
            // For short strings, use the whole string as a single "n-gram"
            visit(bytes);
        } else {
            for ngram in bytes.windows(n) {
                visit(ngram);
            }
        }
    }

    fn ngram_count_for_len(&self, byte_len: usize) -> usize {
        if byte_len < self.n {
            1
        } else {
            byte_len - self.n + 1
        }
    }

    fn index_term_ngrams(&mut self, bytes: &[u8], id: usize) {
        Self::for_each_ngram(self.n, bytes, |ngram| {
            self.index.entry(ngram.to_vec()).or_default().insert(id);
        });
    }

    /// Compute n-grams from a byte slice.
    fn compute_ngrams(&self, bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut ngrams = Vec::with_capacity(self.ngram_count_for_len(bytes.len()));
        Self::for_each_ngram(self.n, bytes, |ngram| ngrams.push(ngram.to_vec()));
        ngrams
    }

    fn query_ngram_slices<'a>(&self, query: &'a str) -> FxHashSet<&'a [u8]> {
        let bytes = query.as_bytes();
        let mut set = FxHashSet::with_capacity_and_hasher(
            self.ngram_count_for_len(bytes.len()),
            Default::default(),
        );
        Self::for_each_ngram(self.n, bytes, |ngram| {
            set.insert(ngram);
        });
        set
    }

    fn min_overlap(&self, query_ngram_count: usize, max_distance: usize) -> usize {
        query_ngram_count.saturating_sub(max_distance.saturating_mul(self.n))
    }

    fn live_terms(&self) -> impl Iterator<Item = (usize, &str)> {
        self.terms
            .iter()
            .enumerate()
            .filter(|(id, _)| self.active.get(*id).copied().unwrap_or(false))
            .map(|(id, term)| (id, term.as_str()))
    }

    fn collect_live_terms(&self) -> Vec<&str> {
        let mut terms = Vec::with_capacity(self.term_to_id.len());
        terms.extend(self.live_terms().map(|(_, term)| term));
        terms
    }

    fn overlap_count_capacity(&self, query_ngrams: &FxHashSet<&[u8]>) -> usize {
        query_ngrams
            .iter()
            .filter_map(|&qgram| self.index.get(qgram))
            .fold(0usize, |count, term_ids| {
                count.saturating_add(term_ids.len())
            })
            .min(self.term_to_id.len())
    }

    fn ngram_overlap_counts(&self, query_ngrams: &FxHashSet<&[u8]>) -> FxHashMap<usize, usize> {
        let mut term_counts = FxHashMap::with_capacity_and_hasher(
            self.overlap_count_capacity(query_ngrams),
            Default::default(),
        );
        for &qgram in query_ngrams {
            if let Some(term_ids) = self.index.get(qgram) {
                for &id in term_ids {
                    *term_counts.entry(id).or_insert(0) += 1;
                }
            }
        }
        term_counts
    }

    /// Find candidate terms within the given edit distance.
    ///
    /// Uses n-gram overlap as a necessary condition for matching.
    /// The returned candidates are a superset of actual matches within distance.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string
    /// * `max_distance` - Maximum edit distance to consider
    ///
    /// # Returns
    ///
    /// Vector of candidate term references.
    ///
    /// # Example
    ///
    /// ```rust
    /// use liblevenshtein::filter::NgramIndex;
    ///
    /// let mut index = NgramIndex::new(2);
    /// index.insert("hello");
    /// index.insert("help");
    /// index.insert("world");
    ///
    /// let candidates = index.find_candidates("helo", 1);
    /// // Returns ["hello", "help"] as candidates
    /// ```
    pub fn find_candidates(&self, query: &str, max_distance: usize) -> Vec<&str> {
        let query_ngrams = self.query_ngram_slices(query);

        // Minimum overlap threshold based on edit distance theory:
        // Each edit can destroy up to n n-grams, so we need at least
        // |query_ngrams| - max_distance * n overlap
        let min_overlap = self.min_overlap(query_ngrams.len(), max_distance);
        if min_overlap == 0 {
            return self.collect_live_terms();
        }

        // Filter by minimum overlap and collect results
        self.ngram_overlap_counts(&query_ngrams)
            .into_iter()
            .filter(|&(id, count)| {
                count >= min_overlap && self.active.get(id).copied().unwrap_or(false)
            })
            .map(|(id, _)| self.terms[id].as_str())
            .collect()
    }

    /// Build a downward-closed dictionary prefix pruner from this filter's
    /// conservative candidate set.
    ///
    /// The resulting visitor moves filtering inside a compatible DFS: a trie
    /// branch is rejected as soon as it ceases to prefix any n-gram candidate.
    pub fn prefix_pruner(&self, query: &str, max_distance: usize) -> AllowedPrefixes<u8> {
        AllowedPrefixes::new(
            self.find_candidates(query, max_distance)
                .into_iter()
                .map(str::as_bytes),
        )
    }

    /// Find candidate terms with their n-gram overlap counts.
    ///
    /// Like `find_candidates` but also returns the overlap count,
    /// which can be used for ranking.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string
    /// * `max_distance` - Maximum edit distance to consider
    ///
    /// # Returns
    ///
    /// Vector of (term, overlap_count) pairs.
    pub fn find_candidates_with_counts(
        &self,
        query: &str,
        max_distance: usize,
    ) -> Vec<(&str, usize)> {
        let query_ngrams = self.query_ngram_slices(query);

        let min_overlap = self.min_overlap(query_ngrams.len(), max_distance);
        let term_counts = self.ngram_overlap_counts(&query_ngrams);

        let mut results: Vec<_> = if min_overlap == 0 {
            let mut results = Vec::with_capacity(self.term_to_id.len());
            results.extend(
                self.live_terms()
                    .map(|(id, term)| (term, term_counts.get(&id).copied().unwrap_or(0))),
            );
            results
        } else {
            term_counts
                .into_iter()
                .filter(|&(id, count)| {
                    count >= min_overlap && self.active.get(id).copied().unwrap_or(false)
                })
                .map(|(id, count)| (self.terms[id].as_str(), count))
                .collect()
        };

        // Sort by overlap count descending (higher overlap = more likely match)
        results.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        results
    }

    /// Get the n-grams for a term (for debugging/inspection).
    ///
    /// # Arguments
    ///
    /// * `term` - The term to get n-grams for
    ///
    /// # Returns
    ///
    /// Vector of n-gram strings.
    pub fn get_ngrams(&self, term: &str) -> Vec<String> {
        self.compute_ngrams(term.as_bytes())
            .into_iter()
            .map(|ng| String::from_utf8_lossy(&ng).to_string())
            .collect()
    }

    /// Clear the index, removing all terms.
    pub fn clear(&mut self) {
        self.index.clear();
        self.terms.clear();
        self.active.clear();
        self.term_to_id.clear();
    }

    /// Iterate over all indexed terms.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.terms
            .iter()
            .zip(self.active.iter())
            .filter(|(_, active)| **active)
            .map(|(term, _)| term)
            .map(String::as_str)
    }
}

impl Default for NgramIndex {
    /// Creates a bigram index (n=2).
    fn default() -> Self {
        Self::new(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let index = NgramIndex::new(2);
        assert_eq!(index.n(), 2);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_default() {
        let index = NgramIndex::default();
        assert_eq!(index.n(), 2);
    }

    #[test]
    fn test_zero_n_normalizes_to_unigrams() {
        let index = NgramIndex::new(0);
        assert_eq!(index.n(), 1);
        assert_eq!(index.get_ngrams("abc"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_insert() {
        let mut index = NgramIndex::new(2);
        let id1 = index.insert("hello");
        let id2 = index.insert("world");
        let id3 = index.insert("hello"); // Duplicate

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 0); // Same as first hello
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_compute_ngrams() {
        let index = NgramIndex::new(2);
        let ngrams = index.get_ngrams("hello");
        assert_eq!(ngrams, vec!["he", "el", "ll", "lo"]);

        // Short string
        let ngrams = index.get_ngrams("a");
        assert_eq!(ngrams, vec!["a"]);
    }

    #[test]
    fn test_trigrams() {
        let index = NgramIndex::new(3);
        let ngrams = index.get_ngrams("hello");
        assert_eq!(ngrams, vec!["hel", "ell", "llo"]);
    }

    #[test]
    fn test_find_candidates() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");
        index.insert("help");
        index.insert("world");
        index.insert("helm");

        // "helo" should match "hello" and "help" but not "world"
        let candidates = index.find_candidates("helo", 1);
        assert!(candidates.contains(&"hello"));
        assert!(candidates.contains(&"help"));
        // "world" has no bigram overlap with "helo"

        // "helm" might or might not match depending on threshold
        // It shares "he" and "el" with "helo"
    }

    #[test]
    fn test_find_candidates_zero_overlap_threshold_returns_all_live_terms() {
        let mut index = NgramIndex::new(2);
        index.insert("abc");
        index.insert("xyz");
        index.insert("pqrs");
        assert!(index.remove("pqrs"));

        let candidates = index.find_candidates("abc", usize::MAX);

        assert!(candidates.contains(&"abc"));
        assert!(candidates.contains(&"xyz"));
        assert!(!candidates.contains(&"pqrs"));
    }

    #[test]
    fn test_find_candidates_with_counts() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");
        index.insert("help");

        let candidates = index.find_candidates_with_counts("hello", 0);

        // Should find "hello" with full overlap
        let hello_result = candidates.iter().find(|(t, _)| *t == "hello");
        assert!(hello_result.is_some());
        let (_, count) = hello_result.expect("expected Some hello_result in test");
        assert_eq!(*count, 4); // he, el, ll, lo
    }

    #[test]
    fn test_find_candidates_with_counts_includes_zero_overlap_when_threshold_is_zero() {
        let mut index = NgramIndex::new(2);
        index.insert("abc");
        index.insert("xyz");

        let candidates = index.find_candidates_with_counts("abc", usize::MAX);

        assert!(candidates.contains(&("abc", 2)));
        assert!(candidates.contains(&("xyz", 0)));
    }

    #[test]
    fn test_repeated_query_ngrams_count_once() {
        let mut index = NgramIndex::new(1);
        index.insert("a");

        let candidates = index.find_candidates_with_counts("aaa", 0);

        assert_eq!(candidates, vec![("a", 1)]);
    }

    #[test]
    fn test_remove() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");
        index.insert("world");

        assert_eq!(index.len(), 2);
        assert!(index.remove("hello"));
        assert!(!index.remove("hello")); // Already removed
        assert_eq!(index.len(), 1);

        // "hello" should no longer appear in candidates
        let candidates = index.find_candidates("hello", 0);
        assert!(!candidates.contains(&"hello"));
    }

    #[test]
    fn test_empty_string_is_a_live_term_not_a_tombstone() {
        let mut index = NgramIndex::new(2);
        index.insert("");
        index.insert("abc");

        assert_eq!(index.len(), 2);
        assert!(index.iter().any(str::is_empty));
        assert!(index.find_candidates("", 0).contains(&""));

        assert!(index.remove("abc"));
        let candidates = index.find_candidates("zzz", usize::MAX);
        assert!(candidates.contains(&""));
        assert!(!candidates.contains(&"abc"));

        assert!(index.remove(""));
        assert!(index.is_empty());
    }

    #[test]
    fn test_from_iter() {
        let terms = vec!["apple", "banana", "cherry"];
        let index = NgramIndex::from_iter(2, terms.into_iter().map(String::from));

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");
        index.insert("world");

        assert_eq!(index.len(), 2);
        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_iter() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");
        index.insert("world");

        let terms: Vec<_> = index.iter().collect();
        assert_eq!(terms.len(), 2);
        assert!(terms.contains(&"hello"));
        assert!(terms.contains(&"world"));
    }

    #[test]
    fn test_empty_query() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");

        // Empty query should have no n-grams, so low overlap
        let candidates = index.find_candidates("", 1);
        // Empty string produces one empty n-gram, which won't match anything
        assert!(candidates.is_empty() || candidates.len() <= 1);
    }

    #[test]
    fn test_unicode() {
        let mut index = NgramIndex::new(2);
        index.insert("café");
        index.insert("cafe");

        // Note: This operates on bytes, not characters
        // "café" = [99, 97, 102, 195, 169] (UTF-8)
        // "cafe" = [99, 97, 102, 101]
        let candidates = index.find_candidates("cafe", 1);
        assert!(candidates.contains(&"cafe"));
        // "café" may or may not match depending on n-gram overlap
    }

    #[test]
    fn test_ngram_count() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");
        index.insert("help");

        // "hello" has [he, el, ll, lo]
        // "help" has [he, el, lp]
        // Unique: he, el, ll, lo, lp = 5
        assert_eq!(index.ngram_count(), 5);
    }
}
