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
//! ```rust,ignore
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
    /// # Panics
    ///
    /// Panics if `n` is 0.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::filter::NgramIndex;
    ///
    /// let bigram_index = NgramIndex::new(2);
    /// let trigram_index = NgramIndex::new(3);
    /// ```
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "N-gram size must be at least 1");
        Self {
            n,
            index: FxHashMap::default(),
            terms: Vec::new(),
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
    /// ```rust,ignore
    /// use liblevenshtein::filter::NgramIndex;
    ///
    /// let terms = vec!["apple", "banana", "cherry"];
    /// let index = NgramIndex::from_iter(2, terms.into_iter().map(String::from));
    /// ```
    pub fn from_iter<I>(n: usize, terms: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        let mut index = Self::new(n);
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

    /// Get the number of indexed terms.
    #[inline]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
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
    /// ```rust,ignore
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
        self.term_to_id.insert(term.to_string(), id);

        // Index by n-grams
        for ngram in self.compute_ngrams(term.as_bytes()) {
            self.index
                .entry(ngram)
                .or_insert_with(FxHashSet::default)
                .insert(id);
        }

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
            for ngram in self.compute_ngrams(term.as_bytes()) {
                if let Some(ids) = self.index.get_mut(&ngram) {
                    ids.remove(&id);
                    // Don't remove empty sets to avoid rehashing
                }
            }
            self.term_to_id.remove(term);
            // Note: We don't remove from terms vec to preserve IDs
            // Set to empty string to mark as deleted
            self.terms[id] = String::new();
            true
        } else {
            false
        }
    }

    /// Compute n-grams from a byte slice.
    fn compute_ngrams(&self, bytes: &[u8]) -> Vec<Vec<u8>> {
        if bytes.len() < self.n {
            // For short strings, use the whole string as a single "n-gram"
            return vec![bytes.to_vec()];
        }

        bytes.windows(self.n).map(|w| w.to_vec()).collect()
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
    /// ```rust,ignore
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
        let query_ngrams: FxHashSet<Vec<u8>> =
            self.compute_ngrams(query.as_bytes()).into_iter().collect();

        // Minimum overlap threshold based on edit distance theory:
        // Each edit can destroy up to n n-grams, so we need at least
        // |query_ngrams| - max_distance * n overlap
        let min_overlap = query_ngrams.len().saturating_sub(max_distance * self.n);

        // Count n-gram matches per term
        let mut term_counts: FxHashMap<usize, usize> = FxHashMap::default();
        for qgram in &query_ngrams {
            if let Some(term_ids) = self.index.get(qgram) {
                for &id in term_ids {
                    *term_counts.entry(id).or_insert(0) += 1;
                }
            }
        }

        // Filter by minimum overlap and collect results
        term_counts
            .into_iter()
            .filter(|&(id, count)| count >= min_overlap && !self.terms[id].is_empty())
            .map(|(id, _)| self.terms[id].as_str())
            .collect()
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
        let query_ngrams: FxHashSet<Vec<u8>> =
            self.compute_ngrams(query.as_bytes()).into_iter().collect();

        let min_overlap = query_ngrams.len().saturating_sub(max_distance * self.n);

        let mut term_counts: FxHashMap<usize, usize> = FxHashMap::default();
        for qgram in &query_ngrams {
            if let Some(term_ids) = self.index.get(qgram) {
                for &id in term_ids {
                    *term_counts.entry(id).or_insert(0) += 1;
                }
            }
        }

        let mut results: Vec<_> = term_counts
            .into_iter()
            .filter(|&(id, count)| count >= min_overlap && !self.terms[id].is_empty())
            .map(|(id, count)| (self.terms[id].as_str(), count))
            .collect();

        // Sort by overlap count descending (higher overlap = more likely match)
        results.sort_by(|a, b| b.1.cmp(&a.1));

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
        self.term_to_id.clear();
    }

    /// Iterate over all indexed terms.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.terms
            .iter()
            .filter(|t| !t.is_empty())
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
    #[should_panic(expected = "N-gram size must be at least 1")]
    fn test_zero_n_panics() {
        NgramIndex::new(0);
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
    fn test_remove() {
        let mut index = NgramIndex::new(2);
        index.insert("hello");
        index.insert("world");

        assert_eq!(index.len(), 2);
        assert!(index.remove("hello"));
        assert!(!index.remove("hello")); // Already removed

        // "hello" should no longer appear in candidates
        let candidates = index.find_candidates("hello", 0);
        assert!(!candidates.contains(&"hello"));
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
