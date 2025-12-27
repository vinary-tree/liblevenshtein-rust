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
pub use pattern_splitter::{FrequencyPatternSplitter, PatternPiece, PatternSplitter};
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

/// WallBreaker with frequency-based pattern splitting.
///
/// This variant uses character frequency analysis to optimize split points,
/// placing rare characters within pieces to reduce false-positive matches
/// during substring search.
///
/// # Performance Characteristics
///
/// Compared to standard WallBreaker:
/// - **Construction**: Slightly slower (must analyze dictionary frequencies)
/// - **Query time**: May be faster for queries containing rare characters
/// - **Memory**: Additional memory for frequency table (~100-200 bytes typical)
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::scdawg::Scdawg;
/// use liblevenshtein::wallbreaker::FrequencyWallBreaker;
///
/// let dict = Scdawg::<()>::from_terms(vec!["cathedral", "category"]);
/// let wb = FrequencyWallBreaker::new(&dict, 2);
///
/// for result in wb.query("cathedrel") {
///     println!("{} (distance {})", result.term, result.distance);
/// }
/// ```
pub struct FrequencyWallBreaker<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    dictionary: &'a D,
    max_distance: usize,
    splitter: FrequencyPatternSplitter,
}

impl<'a, D> FrequencyWallBreaker<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    /// Create a new FrequencyWallBreaker from an iterator of dictionary terms.
    ///
    /// This analyzes the terms to compute character frequencies, which
    /// are then used to optimize pattern splitting.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The SCDAWG or other substring-searchable dictionary
    /// * `terms` - Iterator over dictionary terms for frequency analysis
    /// * `max_distance` - Maximum Levenshtein distance for matches
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dict = Scdawg::<()>::from_terms(vec!["hello", "world"]);
    /// let terms = dict.iter(); // Scdawg has iter() method
    /// let wb = FrequencyWallBreaker::from_terms(&dict, terms, 2);
    /// ```
    pub fn from_terms<I, S>(dictionary: &'a D, terms: I, max_distance: usize) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let splitter = FrequencyPatternSplitter::from_terms(terms, max_distance);

        FrequencyWallBreaker {
            dictionary,
            max_distance,
            splitter,
        }
    }

    /// Create with a pre-built frequency splitter.
    ///
    /// This is useful when you want to reuse the same frequency analysis
    /// across multiple FrequencyWallBreaker instances.
    pub fn with_splitter(dictionary: &'a D, splitter: FrequencyPatternSplitter) -> Self {
        let max_distance = splitter.max_distance();
        FrequencyWallBreaker {
            dictionary,
            max_distance,
            splitter,
        }
    }

    /// Query the dictionary for approximate matches using frequency-optimized splitting.
    ///
    /// Returns an iterator over results for all dictionary terms within
    /// `max_distance` of the query.
    pub fn query(&self, query: &str) -> FrequencyWallBreakerQuery<'_, D> {
        FrequencyWallBreakerQuery::new(self.dictionary, query, self.max_distance, &self.splitter)
    }

    /// Get the maximum distance configured for this FrequencyWallBreaker.
    pub fn max_distance(&self) -> usize {
        self.max_distance
    }
}

/// Query iterator for FrequencyWallBreaker.
///
/// Similar to WallBreakerQuery but uses frequency-based pattern splitting.
pub struct FrequencyWallBreakerQuery<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    dictionary: &'a D,
    query: String,
    max_distance: usize,
    pieces: Vec<PatternPiece>,
    current_piece_idx: usize,
    current_matches: Vec<WallBreakerResult>,
    current_match_idx: usize,
    seen_terms: std::collections::HashSet<String>,
}

impl<'a, D> FrequencyWallBreakerQuery<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    fn new(
        dictionary: &'a D,
        query: &str,
        max_distance: usize,
        splitter: &FrequencyPatternSplitter,
    ) -> Self {
        let pieces = splitter.split(query);

        FrequencyWallBreakerQuery {
            dictionary,
            query: query.to_string(),
            max_distance,
            pieces,
            current_piece_idx: 0,
            current_matches: Vec::new(),
            current_match_idx: 0,
            seen_terms: std::collections::HashSet::new(),
        }
    }

    fn process_next_piece(&mut self) {
        self.current_matches.clear();
        self.current_match_idx = 0;

        if self.current_piece_idx >= self.pieces.len() {
            return;
        }

        let piece = &self.pieces[self.current_piece_idx];
        self.current_piece_idx += 1;

        // Find substring matches for this piece
        let substring_matches = self.dictionary.find_exact_substring(&piece.content);

        // For each match, verify the full query distance
        for match_info in substring_matches {
            let term = &match_info.term;

            // Skip if we've already seen this term
            if self.seen_terms.contains(term) {
                continue;
            }

            // Compute actual Levenshtein distance
            let distance = crate::distance::standard_distance(&self.query, term);

            if distance <= self.max_distance {
                self.seen_terms.insert(term.clone());
                self.current_matches.push(WallBreakerResult {
                    term: term.clone(),
                    distance,
                });
            }
        }
    }
}

impl<D> Iterator for FrequencyWallBreakerQuery<'_, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    type Item = WallBreakerResult;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Return next match from current piece
            if self.current_match_idx < self.current_matches.len() {
                let result = self.current_matches[self.current_match_idx].clone();
                self.current_match_idx += 1;
                return Some(result);
            }

            // No more matches from current piece, try next piece
            if self.current_piece_idx >= self.pieces.len() {
                return None;
            }

            self.process_next_piece();
        }
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

    // ============================================================================
    // FrequencyWallBreaker tests
    // ============================================================================

    #[test]
    fn test_frequency_wallbreaker_basic() {
        let terms = vec!["hello", "world", "help"];
        let dict = Scdawg::<()>::from_terms(terms.clone());
        let wb = FrequencyWallBreaker::from_terms(&dict, terms.iter().copied(), 1);

        let results: Vec<_> = wb.query("helo").collect();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.term == "hello"));
    }

    #[test]
    fn test_frequency_wallbreaker_exact_match() {
        let terms = vec!["hello", "world"];
        let dict = Scdawg::<()>::from_terms(terms.clone());
        let wb = FrequencyWallBreaker::from_terms(&dict, terms.iter().copied(), 0);

        let results: Vec<_> = wb.query("hello").collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "hello");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_frequency_wallbreaker_no_match() {
        let terms = vec!["hello", "world"];
        let dict = Scdawg::<()>::from_terms(terms.clone());
        let wb = FrequencyWallBreaker::from_terms(&dict, terms.iter().copied(), 1);

        let results: Vec<_> = wb.query("xyz").collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_frequency_wallbreaker_same_results_as_standard() {
        let terms = vec![
            "cathedral", "category", "catering", "caterpillar", "catastrophe"
        ];
        let dict = Scdawg::<()>::from_terms(terms.clone());

        let std_wb = WallBreaker::new(&dict, 2);
        let freq_wb = FrequencyWallBreaker::from_terms(&dict, terms.iter().copied(), 2);

        let query = "cathedrel";

        let std_results: std::collections::HashSet<_> = std_wb.query(query)
            .map(|r| (r.term, r.distance))
            .collect();
        let freq_results: std::collections::HashSet<_> = freq_wb.query(query)
            .map(|r| (r.term, r.distance))
            .collect();

        // Both should return the same matches
        assert_eq!(std_results, freq_results);
    }
}
