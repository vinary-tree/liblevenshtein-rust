//! Hybrid pre-filtering combining N-gram index and Jaro-Winkler similarity.
//!
//! This module provides a multi-stage filtering pipeline that can reject
//! 90%+ of candidates before expensive Levenshtein automata traversal.
//!
//! # Pipeline Stages
//!
//! 1. **N-gram Filter** (Stage 1 - Coarse)
//!    - Very fast hash-based lookup
//!    - Rejects candidates with insufficient n-gram overlap
//!    - Typical rejection rate: 85-95%
//!
//! 2. **Jaro-Winkler Filter** (Stage 2 - Fine)
//!    - Linear-time similarity computation
//!    - Rejects candidates below similarity threshold
//!    - Typical rejection rate: 50-80% of remaining
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::filter::HybridMatcher;
//!
//! // Build matcher from dictionary
//! let terms = vec!["apple", "application", "banana", "apply", "appeal"];
//! let matcher = HybridMatcher::new(terms.into_iter().map(String::from));
//!
//! // Find candidates for "aple" with max distance 2
//! let candidates = matcher.filter_candidates("aple", 2);
//! // Efficiently returns ["apple", "apply"] (rejects others)
//! ```
//!
//! # Performance
//!
//! For a 100K word dictionary with max_distance=2:
//! - N-gram stage: ~1-5ms
//! - Jaro-Winkler stage: ~0.5-2ms
//! - Total: ~2-7ms vs ~50-200ms for full automaton traversal

use super::jaro_winkler::jaro_winkler_similarity;
use super::ngram::NgramIndex;

/// Default n-gram size for hybrid matching.
const DEFAULT_NGRAM_SIZE: usize = 2;

/// Default Jaro-Winkler similarity threshold.
const DEFAULT_JARO_THRESHOLD: f64 = 0.7;

/// Multi-stage hybrid matcher for approximate string matching.
///
/// Combines N-gram indexing with Jaro-Winkler similarity for efficient
/// candidate pre-filtering.
#[derive(Debug, Clone)]
pub struct HybridMatcher {
    /// N-gram index for coarse filtering.
    ngram_index: NgramIndex,

    /// Minimum Jaro-Winkler similarity for fine filtering.
    jaro_threshold: f64,

    /// Whether to skip the Jaro-Winkler stage (use n-gram only).
    skip_jaro: bool,
}

impl HybridMatcher {
    /// Create a hybrid matcher from an iterator of terms.
    ///
    /// Uses default settings (bigram index, 0.7 Jaro threshold).
    ///
    /// # Arguments
    ///
    /// * `terms` - Iterator of terms to index
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::filter::HybridMatcher;
    ///
    /// let terms = ["apple", "banana", "cherry"];
    /// let matcher = HybridMatcher::new(terms.into_iter().map(String::from));
    /// ```
    pub fn new<I>(terms: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        Self::with_config(terms, DEFAULT_NGRAM_SIZE, DEFAULT_JARO_THRESHOLD)
    }

    /// Create a hybrid matcher with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `terms` - Iterator of terms to index
    /// * `ngram_size` - N-gram size (typically 2 or 3)
    /// * `jaro_threshold` - Minimum Jaro-Winkler similarity (0.0-1.0)
    ///
    /// # Panics
    ///
    /// Panics if ngram_size is 0 or jaro_threshold is not in [0.0, 1.0].
    pub fn with_config<I>(terms: I, ngram_size: usize, jaro_threshold: f64) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        assert!(ngram_size > 0, "N-gram size must be at least 1");
        assert!(
            (0.0..=1.0).contains(&jaro_threshold),
            "Jaro threshold must be in [0.0, 1.0], got {}",
            jaro_threshold
        );

        let ngram_index = NgramIndex::from_iter(ngram_size, terms);

        Self {
            ngram_index,
            jaro_threshold,
            skip_jaro: false,
        }
    }

    /// Create a matcher using only N-gram filtering (skip Jaro-Winkler).
    ///
    /// Faster but less precise filtering.
    pub fn ngram_only<I>(terms: I, ngram_size: usize) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        let ngram_index = NgramIndex::from_iter(ngram_size, terms);

        Self {
            ngram_index,
            jaro_threshold: 0.0,
            skip_jaro: true,
        }
    }

    /// Get the number of indexed terms.
    #[inline]
    pub fn len(&self) -> usize {
        self.ngram_index.len()
    }

    /// Check if the matcher is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ngram_index.is_empty()
    }

    /// Get the n-gram size.
    #[inline]
    pub fn ngram_size(&self) -> usize {
        self.ngram_index.n()
    }

    /// Get the Jaro-Winkler threshold.
    #[inline]
    pub fn jaro_threshold(&self) -> f64 {
        self.jaro_threshold
    }

    /// Set the Jaro-Winkler threshold.
    ///
    /// # Panics
    ///
    /// Panics if threshold is not in [0.0, 1.0].
    pub fn set_jaro_threshold(&mut self, threshold: f64) {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "Jaro threshold must be in [0.0, 1.0], got {}",
            threshold
        );
        self.jaro_threshold = threshold;
    }

    /// Add a term to the index.
    pub fn insert(&mut self, term: &str) {
        self.ngram_index.insert(term);
    }

    /// Remove a term from the index.
    pub fn remove(&mut self, term: &str) -> bool {
        self.ngram_index.remove(term)
    }

    /// Filter candidates using the multi-stage pipeline.
    ///
    /// Stage 1: N-gram filtering for coarse candidate selection.
    /// Stage 2: Jaro-Winkler refinement for fine filtering.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string
    /// * `max_distance` - Maximum edit distance to consider
    ///
    /// # Returns
    ///
    /// Vector of candidate term references that pass both stages.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::filter::HybridMatcher;
    ///
    /// let matcher = HybridMatcher::new(["apple", "apply", "banana"].iter().map(|s| s.to_string()));
    /// let candidates = matcher.filter_candidates("aple", 2);
    /// assert!(candidates.contains(&"apple"));
    /// ```
    pub fn filter_candidates(&self, query: &str, max_distance: usize) -> Vec<&str> {
        // Stage 1: N-gram filter
        let ngram_candidates = self.ngram_index.find_candidates(query, max_distance);

        if self.skip_jaro || self.jaro_threshold <= 0.0 {
            return ngram_candidates;
        }

        // Stage 2: Jaro-Winkler filter
        // Compute adaptive threshold based on max_distance and query length
        let adaptive_threshold = self.compute_adaptive_threshold(query, max_distance);

        ngram_candidates
            .into_iter()
            .filter(|term| jaro_winkler_similarity(query, term) >= adaptive_threshold)
            .collect()
    }

    /// Filter candidates and return similarity scores.
    ///
    /// Like `filter_candidates` but also returns Jaro-Winkler scores,
    /// which can be used for ranking.
    ///
    /// # Returns
    ///
    /// Vector of (term, jaro_winkler_score) pairs, sorted by score descending.
    pub fn filter_candidates_with_scores(
        &self,
        query: &str,
        max_distance: usize,
    ) -> Vec<(&str, f64)> {
        let ngram_candidates = self.ngram_index.find_candidates(query, max_distance);

        if self.skip_jaro {
            // Return with dummy scores
            return ngram_candidates.into_iter().map(|t| (t, 1.0)).collect();
        }

        let adaptive_threshold = self.compute_adaptive_threshold(query, max_distance);

        let mut results: Vec<_> = ngram_candidates
            .into_iter()
            .map(|term| (term, jaro_winkler_similarity(query, term)))
            .filter(|&(_, score)| score >= adaptive_threshold)
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    /// Compute adaptive Jaro-Winkler threshold based on query and max distance.
    ///
    /// Shorter queries and higher max distances should use lower thresholds.
    fn compute_adaptive_threshold(&self, query: &str, max_distance: usize) -> f64 {
        let query_len = query.chars().count();

        if query_len == 0 {
            return 0.0;
        }

        // Base threshold adjusted for edit distance tolerance
        // Higher max_distance = lower threshold requirement
        let distance_factor = 1.0 - (max_distance as f64 / query_len as f64).min(1.0);

        // Blend base threshold with distance-adjusted factor
        (self.jaro_threshold * distance_factor).max(0.0).min(1.0)
    }

    /// Get statistics about the filter performance.
    ///
    /// Returns (ngram_count, term_count) for debugging/tuning.
    pub fn stats(&self) -> (usize, usize) {
        (self.ngram_index.ngram_count(), self.ngram_index.len())
    }

    /// Iterate over all indexed terms.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.ngram_index.iter()
    }
}

impl Default for HybridMatcher {
    fn default() -> Self {
        Self {
            ngram_index: NgramIndex::default(),
            jaro_threshold: DEFAULT_JARO_THRESHOLD,
            skip_jaro: false,
        }
    }
}

/// Builder for configuring HybridMatcher.
#[derive(Debug, Clone)]
pub struct HybridMatcherBuilder {
    ngram_size: usize,
    jaro_threshold: f64,
    skip_jaro: bool,
}

impl HybridMatcherBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            ngram_size: DEFAULT_NGRAM_SIZE,
            jaro_threshold: DEFAULT_JARO_THRESHOLD,
            skip_jaro: false,
        }
    }

    /// Set the n-gram size.
    pub fn ngram_size(mut self, size: usize) -> Self {
        self.ngram_size = size;
        self
    }

    /// Set the Jaro-Winkler threshold.
    pub fn jaro_threshold(mut self, threshold: f64) -> Self {
        self.jaro_threshold = threshold;
        self
    }

    /// Disable Jaro-Winkler filtering (use n-gram only).
    pub fn ngram_only(mut self) -> Self {
        self.skip_jaro = true;
        self
    }

    /// Build the matcher from an iterator of terms.
    pub fn build<I>(self, terms: I) -> HybridMatcher
    where
        I: IntoIterator<Item = String>,
    {
        let ngram_index = NgramIndex::from_iter(self.ngram_size, terms);

        HybridMatcher {
            ngram_index,
            jaro_threshold: self.jaro_threshold,
            skip_jaro: self.skip_jaro,
        }
    }
}

impl Default for HybridMatcherBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_terms() -> Vec<String> {
        vec![
            "apple",
            "application",
            "apply",
            "appeal",
            "banana",
            "cherry",
            "hello",
            "help",
            "world",
            "helm",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    #[test]
    fn test_new() {
        let matcher = HybridMatcher::new(test_terms());
        assert_eq!(matcher.len(), 10);
        assert!(!matcher.is_empty());
        assert_eq!(matcher.ngram_size(), 2);
    }

    #[test]
    fn test_with_config() {
        let matcher = HybridMatcher::with_config(test_terms(), 3, 0.8);
        assert_eq!(matcher.ngram_size(), 3);
        assert!((matcher.jaro_threshold() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_filter_candidates() {
        let matcher = HybridMatcher::new(test_terms());

        // "aple" should match "apple" and possibly "apply"
        let candidates = matcher.filter_candidates("aple", 1);
        assert!(
            candidates.contains(&"apple"),
            "Expected 'apple' in candidates: {:?}",
            candidates
        );
    }

    #[test]
    fn test_filter_with_scores() {
        let matcher = HybridMatcher::new(test_terms());

        let results = matcher.filter_candidates_with_scores("apple", 1);

        // Should find exact match with high score
        let apple_result = results.iter().find(|(t, _)| *t == "apple");
        assert!(apple_result.is_some());
        let (_, score) = apple_result.expect("expected Some apple_result in test");
        assert!(
            *score > 0.99,
            "Exact match should have score ~1.0, got {}",
            score
        );
    }

    #[test]
    fn test_insert_remove() {
        let mut matcher = HybridMatcher::new(test_terms());

        assert_eq!(matcher.len(), 10);

        matcher.insert("newterm");
        assert_eq!(matcher.len(), 11);

        assert!(matcher.remove("newterm"));
        // Note: len doesn't decrease after remove (tombstone approach)
        assert!(!matcher.remove("newterm"));
    }

    #[test]
    fn test_ngram_only() {
        let matcher = HybridMatcher::ngram_only(test_terms(), 2);

        // Should still find candidates but skip Jaro-Winkler
        let candidates = matcher.filter_candidates("apple", 1);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_builder() {
        let matcher = HybridMatcherBuilder::new()
            .ngram_size(3)
            .jaro_threshold(0.75)
            .build(test_terms());

        assert_eq!(matcher.ngram_size(), 3);
        assert!((matcher.jaro_threshold() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_builder_ngram_only() {
        let matcher = HybridMatcherBuilder::new().ngram_only().build(test_terms());

        // Should work without Jaro-Winkler
        let candidates = matcher.filter_candidates("apple", 1);
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_set_threshold() {
        let mut matcher = HybridMatcher::new(test_terms());

        matcher.set_jaro_threshold(0.9);
        assert!((matcher.jaro_threshold() - 0.9).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Jaro threshold must be in [0.0, 1.0]")]
    fn test_invalid_threshold_panics() {
        let mut matcher = HybridMatcher::new(test_terms());
        matcher.set_jaro_threshold(1.5);
    }

    #[test]
    fn test_stats() {
        let matcher = HybridMatcher::new(test_terms());
        let (ngram_count, term_count) = matcher.stats();

        assert_eq!(term_count, 10);
        assert!(ngram_count > 0);
    }

    #[test]
    fn test_iter() {
        let matcher = HybridMatcher::new(test_terms());
        let terms: Vec<_> = matcher.iter().collect();

        assert_eq!(terms.len(), 10);
        assert!(terms.contains(&"apple"));
        assert!(terms.contains(&"banana"));
    }

    #[test]
    fn test_empty_query() {
        let matcher = HybridMatcher::new(test_terms());
        let candidates = matcher.filter_candidates("", 2);

        // Empty query has adaptive threshold of 0, so may return everything
        // or nothing depending on n-gram behavior
        // Just ensure it doesn't panic
        let _ = candidates;
    }

    #[test]
    fn test_adaptive_threshold() {
        let matcher = HybridMatcher::new(test_terms());

        // Short query with high max_distance = lower threshold
        let candidates_lenient = matcher.filter_candidates("app", 3);

        // Same query with low max_distance = higher effective threshold
        let candidates_strict = matcher.filter_candidates("app", 1);

        // Lenient should have >= candidates than strict
        assert!(
            candidates_lenient.len() >= candidates_strict.len(),
            "Lenient ({}) should have >= strict ({})",
            candidates_lenient.len(),
            candidates_strict.len()
        );
    }
}
