//! WallBreaker algorithm for approximate string matching with large error bounds.
//!
//! This module implements the WallBreaker algorithm from:
//! > "WallBreaker - overcoming the wall effect in similarity search"
//! > (Gerdjikov, Mihov, Mitankin, Schulz - EDBT/ICDT 2013)
//!
//! # The Wall Effect Problem
//!
//! Traditional Levenshtein automata traverse dictionaries left-to-right.
//! With error bound `b`, the first `b` steps must explore ALL prefixes up
//! to length `b` before any filtering occurs. This creates a "wall" that
//! limits performance for large error bounds.
//!
//! # WallBreaker Solution
//!
//! WallBreaker overcomes the wall by:
//!
//! 1. **Splitting** the query into pieces based on the pigeonhole principle
//! 2. **Finding exact matches** for each piece using SCDAWG substring search
//! 3. **Extending bidirectionally** from matches using Levenshtein filters
//! 4. **Verifying** total distance and deduplicating results
//!
//! # Piece Count by Algorithm (Formally Verified)
//!
//! The number of pieces depends on the edit distance algorithm:
//!
//! - **Standard Levenshtein**: `k+1` pieces suffice
//!   - Each operation (insert, delete, substitute) corrupts at most 1 piece
//!
//! - **Transposition (optimal string alignment)**: `2k+1` pieces required
//!   - Adjacent transpositions can corrupt 2 pieces when spanning boundaries
//!   - Proven in `WallBreakerPigeonhole.v` with counterexample for k=2
//!
//! - **MergeAndSplit**: `2k+1` pieces required
//!   - Merge operations span 2 characters, can corrupt 2 pieces at boundaries
//!   - Proven in `WallBreakerPigeonhole.v` with counterexample for k=2
//!
//! # Performance
//!
//! For 100-character patterns with 16 errors in a 750K dictionary:
//! - Traditional approach: ~500ms
//! - WallBreaker: ~0.088ms (5600x speedup)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::scdawg::Scdawg;
//! use liblevenshtein::wallbreaker::WallBreaker;
//! use liblevenshtein::transducer::Algorithm;
//!
//! // Build SCDAWG dictionary
//! let dict = Scdawg::<()>::from_terms(vec!["cathedral", "category", "catering"]);
//!
//! // Create WallBreaker with max distance 2 (Standard algorithm by default)
//! let wb = WallBreaker::new(&dict, 2);
//!
//! // Or explicitly specify an algorithm for Transposition or MergeAndSplit
//! let wb = WallBreaker::with_algorithm(&dict, 2, Algorithm::Transposition);
//!
//! // Find approximate matches
//! for result in wb.query("cathedrel") {
//!     println!("{} (distance {})", result.term, result.distance);
//! }
//! // Output: cathedral (distance 1)
//! ```

mod extension;
mod pattern_splitter;
mod query_iterator;

pub use extension::{BidirectionalExtension, ExtensionState};
pub use pattern_splitter::{PatternPiece, PatternSplitter};
pub use query_iterator::{WallBreakerQuery, WallBreakerResult};

use crate::transducer::Algorithm;
use libdictenstein::substring::{BidirectionalDictionaryNode, SubstringDictionary};
use libdictenstein::Dictionary;

/// WallBreaker approximate string matcher.
///
/// Wraps a [`SubstringDictionary`] (typically an SCDAWG) and provides
/// approximate matching using the WallBreaker algorithm.
///
/// # Algorithm Support
///
/// WallBreaker supports all three Levenshtein algorithm variants with
/// algorithm-specific piece counts (formally verified in `WallBreakerPigeonhole.v`):
///
/// - **Standard**: `k+1` pieces (default)
/// - **Transposition**: `2k+1` pieces
/// - **MergeAndSplit**: `2k+1` pieces
///
/// # Type Parameters
///
/// * `D` - Dictionary type that implements both [`Dictionary`] and [`SubstringDictionary`]
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::scdawg::Scdawg;
/// use liblevenshtein::wallbreaker::WallBreaker;
/// use liblevenshtein::transducer::Algorithm;
///
/// let dict = Scdawg::<()>::from_terms(["hello", "world", "help"]);
///
/// // Standard algorithm (default)
/// let wb = WallBreaker::new(&dict, 1);
///
/// // Transposition algorithm
/// let wb = WallBreaker::with_algorithm(&dict, 1, Algorithm::Transposition);
///
/// let results: Vec<_> = wb.query("helo").collect();
/// assert!(results.iter().any(|r| r.term == "hello"));
/// ```
pub struct WallBreaker<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    dictionary: &'a D,
    max_distance: usize,
    algorithm: Algorithm,
    splitter: PatternSplitter,
}

impl<'a, D> WallBreaker<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    /// Create a new WallBreaker with the given dictionary and max distance.
    ///
    /// Uses the Standard Levenshtein algorithm by default.
    /// For Transposition or MergeAndSplit, use [`with_algorithm`](Self::with_algorithm).
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The SCDAWG or other substring-searchable dictionary
    /// * `max_distance` - Maximum Levenshtein distance for matches
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dict = Scdawg::<()>::from_terms(["test"]);
    /// let wb = WallBreaker::new(&dict, 2);
    /// ```
    pub fn new(dictionary: &'a D, max_distance: usize) -> Self {
        Self::with_algorithm(dictionary, max_distance, Algorithm::Standard)
    }

    /// Create a new WallBreaker with a specific algorithm.
    ///
    /// # Algorithm-Specific Piece Counts (Formally Verified)
    ///
    /// - **Standard**: `k+1` pieces (each operation corrupts ≤1 piece)
    /// - **Transposition**: `2k+1` pieces (transpositions can corrupt 2 pieces)
    /// - **MergeAndSplit**: `2k+1` pieces (merge/split can corrupt 2 pieces)
    ///
    /// These piece counts are proven correct in `WallBreakerPigeonhole.v`.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The SCDAWG or other substring-searchable dictionary
    /// * `max_distance` - Maximum edit distance for matches
    /// * `algorithm` - The edit distance algorithm to use
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::transducer::Algorithm;
    ///
    /// let dict = Scdawg::<()>::from_terms(["test"]);
    ///
    /// // For OSA (adjacent-transposition) matching
    /// let wb = WallBreaker::with_algorithm(&dict, 2, Algorithm::Transposition);
    /// ```
    pub fn with_algorithm(dictionary: &'a D, max_distance: usize, algorithm: Algorithm) -> Self {
        WallBreaker {
            dictionary,
            max_distance,
            algorithm,
            splitter: PatternSplitter::new(max_distance, algorithm),
        }
    }

    /// Query the dictionary for approximate matches.
    ///
    /// Returns an iterator over (term, distance) pairs for all dictionary
    /// terms within `max_distance` of the query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string to match
    ///
    /// # Returns
    ///
    /// An iterator yielding `(String, usize)` pairs where:
    /// - `String` is the matched dictionary term
    /// - `usize` is the Levenshtein distance from query to term
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// for (term, distance) in wb.query("tset") {
    ///     println!("{} at distance {}", term, distance);
    /// }
    /// ```
    pub fn query(&self, query: &str) -> WallBreakerQuery<'_, D> {
        WallBreakerQuery::new(self.dictionary, query, self.max_distance, &self.splitter)
    }

    /// Get the maximum distance configured for this WallBreaker.
    pub fn max_distance(&self) -> usize {
        self.max_distance
    }

    /// Get the algorithm configured for this WallBreaker.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Update the maximum distance.
    ///
    /// This preserves the current algorithm and updates the pattern splitter
    /// with the algorithm-specific piece count.
    pub fn set_max_distance(&mut self, max_distance: usize) {
        self.max_distance = max_distance;
        self.splitter = PatternSplitter::new(max_distance, self.algorithm);
    }

    /// Update the algorithm.
    ///
    /// This updates the pattern splitter with the new algorithm-specific piece count.
    pub fn set_algorithm(&mut self, algorithm: Algorithm) {
        self.algorithm = algorithm;
        self.splitter = PatternSplitter::new(self.max_distance, algorithm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::scdawg::Scdawg;

    #[test]
    fn test_wallbreaker_basic() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world", "help"]);
        let wb = WallBreaker::new(&dict, 1);

        let results: Vec<_> = wb.query("helo").collect();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.term == "hello"));
    }

    #[test]
    fn test_wallbreaker_exact_match() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        let wb = WallBreaker::new(&dict, 0);

        let results: Vec<_> = wb.query("hello").collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "hello");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_wallbreaker_no_match() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world"]);
        let wb = WallBreaker::new(&dict, 1);

        let results: Vec<_> = wb.query("xyz").collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_wallbreaker_distance_2() {
        let dict = Scdawg::<()>::from_terms(vec!["cathedral"]);
        let wb = WallBreaker::new(&dict, 2);

        // "cathedrel" has distance 1 from "cathedral" (e->a)
        let results: Vec<_> = wb.query("cathedrel").collect();
        assert!(results.iter().any(|r| r.term == "cathedral"));
    }

    #[test]
    fn test_wallbreaker_multiple_terms() {
        let terms = vec![
            "cathedral",
            "category",
            "catering",
            "catastrophe",
            "catalog",
        ];
        let dict = Scdawg::<()>::from_terms(terms);
        let wb = WallBreaker::new(&dict, 2);

        // Test various queries
        let results: Vec<_> = wb.query("cathedrel").collect();
        assert!(results.iter().any(|r| r.term == "cathedral"));

        let results: Vec<_> = wb.query("caterng").collect();
        assert!(results.iter().any(|r| r.term == "catering"));
    }

    // Algorithm-specific tests

    #[test]
    fn test_wallbreaker_with_algorithm() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world", "help"]);

        // Test all algorithm types can be created
        let wb_std = WallBreaker::with_algorithm(&dict, 1, Algorithm::Standard);
        let wb_trans = WallBreaker::with_algorithm(&dict, 1, Algorithm::Transposition);
        let wb_ms = WallBreaker::with_algorithm(&dict, 1, Algorithm::MergeAndSplit);

        // Verify algorithms are set correctly
        assert!(matches!(wb_std.algorithm(), Algorithm::Standard));
        assert!(matches!(wb_trans.algorithm(), Algorithm::Transposition));
        assert!(matches!(wb_ms.algorithm(), Algorithm::MergeAndSplit));
    }

    #[test]
    fn test_wallbreaker_algorithm_getter() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);

        // Default is Standard
        let wb = WallBreaker::new(&dict, 1);
        assert!(matches!(wb.algorithm(), Algorithm::Standard));

        // Explicit algorithm
        let wb = WallBreaker::with_algorithm(&dict, 1, Algorithm::Transposition);
        assert!(matches!(wb.algorithm(), Algorithm::Transposition));
    }

    #[test]
    fn test_wallbreaker_set_algorithm() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let mut wb = WallBreaker::new(&dict, 2);

        // Initial algorithm is Standard
        assert!(matches!(wb.algorithm(), Algorithm::Standard));

        // Change to Transposition
        wb.set_algorithm(Algorithm::Transposition);
        assert!(matches!(wb.algorithm(), Algorithm::Transposition));

        // Change to MergeAndSplit
        wb.set_algorithm(Algorithm::MergeAndSplit);
        assert!(matches!(wb.algorithm(), Algorithm::MergeAndSplit));
    }

    #[test]
    fn test_wallbreaker_set_max_distance_preserves_algorithm() {
        let dict = Scdawg::<()>::from_terms(vec!["test"]);
        let mut wb = WallBreaker::with_algorithm(&dict, 2, Algorithm::Transposition);

        // Change max distance
        wb.set_max_distance(4);

        // Algorithm should be preserved
        assert!(matches!(wb.algorithm(), Algorithm::Transposition));
        assert_eq!(wb.max_distance(), 4);
    }

    #[test]
    fn test_wallbreaker_transposition_finds_matches() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world", "help"]);
        let wb = WallBreaker::with_algorithm(&dict, 1, Algorithm::Transposition);

        // Should find matches using the configured transposition verifier.
        let results: Vec<_> = wb.query("helo").collect();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_wallbreaker_merge_and_split_finds_matches() {
        let dict = Scdawg::<()>::from_terms(vec!["hello", "world", "help"]);
        let wb = WallBreaker::with_algorithm(&dict, 1, Algorithm::MergeAndSplit);

        // Should find matches using the configured merge/split verifier.
        let results: Vec<_> = wb.query("helo").collect();
        assert!(!results.is_empty());
    }
}
