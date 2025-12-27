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
//! 1. **Splitting** the query into `b+1` pieces (pigeonhole principle:
//!    at least one piece must match exactly)
//! 2. **Finding exact matches** for each piece using SCDAWG substring search
//! 3. **Extending bidirectionally** from matches using Levenshtein filters
//! 4. **Verifying** total distance and deduplicating results
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
//!
//! // Build SCDAWG dictionary
//! let dict = Scdawg::<()>::from_terms(vec!["cathedral", "category", "catering"]);
//!
//! // Create WallBreaker with max distance 2
//! let wb = WallBreaker::new(&dict, 2);
//!
//! // Find approximate matches
//! for (term, distance) in wb.query("cathedrel") {
//!     println!("{} (distance {})", term, distance);
//! }
//! // Output: cathedral (distance 1)
//! ```

mod extension;
mod pattern_splitter;
mod query_iterator;

pub use extension::{BidirectionalExtension, ExtensionState};
pub use pattern_splitter::{PatternPiece, PatternSplitter};
pub use query_iterator::{WallBreakerQuery, WallBreakerResult};

use crate::dictionary::substring::{BidirectionalDictionaryNode, SubstringDictionary};
use crate::dictionary::Dictionary;

/// WallBreaker approximate string matcher.
///
/// Wraps a [`SubstringDictionary`] (typically an SCDAWG) and provides
/// approximate matching using the WallBreaker algorithm.
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
///
/// let dict = Scdawg::<()>::from_terms(["hello", "world", "help"]);
/// let wb = WallBreaker::new(&dict, 1);
///
/// let results: Vec<_> = wb.query("helo").collect();
/// assert!(results.iter().any(|(t, _)| t == "hello"));
/// ```
pub struct WallBreaker<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    dictionary: &'a D,
    max_distance: usize,
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
        WallBreaker {
            dictionary,
            max_distance,
            splitter: PatternSplitter::new(max_distance),
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

    /// Update the maximum distance.
    ///
    /// This also updates the pattern splitter to use `max_distance + 1` pieces.
    pub fn set_max_distance(&mut self, max_distance: usize) {
        self.max_distance = max_distance;
        self.splitter = PatternSplitter::new(max_distance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::scdawg::Scdawg;

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
}
