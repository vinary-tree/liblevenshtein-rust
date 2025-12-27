//! Query iterator for WallBreaker algorithm.
//!
//! The main query iterator that orchestrates:
//! 1. Pattern splitting into `b+1` pieces
//! 2. Finding exact substring matches for each piece
//! 3. Bidirectional extension from matches
//! 4. Deduplication of results

use std::collections::HashSet;

use crate::dictionary::substring::{BidirectionalDictionaryNode, SubstringDictionary};
use crate::dictionary::Dictionary;

use super::extension::BidirectionalExtension;
use super::pattern_splitter::{PatternPiece, PatternSplitter};

/// Result from WallBreaker query.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WallBreakerResult {
    /// The matched dictionary term.
    pub term: String,

    /// Levenshtein distance from query to term.
    pub distance: usize,
}

impl WallBreakerResult {
    /// Create a new result.
    pub fn new(term: String, distance: usize) -> Self {
        WallBreakerResult { term, distance }
    }
}

/// Query iterator for WallBreaker algorithm.
///
/// Iterates through all dictionary terms within the specified edit distance
/// of the query string. Uses the pigeonhole principle and SCDAWG substring
/// search for efficient matching.
///
/// # Algorithm
///
/// 1. Split query into `b+1` pieces (where `b` is max_distance)
/// 2. For each piece, find exact substring matches in dictionary
/// 3. Extend each match bidirectionally using Levenshtein filters
/// 4. Deduplicate and yield results
pub struct WallBreakerQuery<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    /// Reference to the dictionary.
    dictionary: &'a D,

    /// The query string.
    query: String,

    /// Maximum edit distance.
    max_distance: usize,

    /// Pattern pieces from splitting.
    pieces: Vec<PatternPiece>,

    /// Current piece index being processed.
    current_piece_idx: usize,

    /// Results from current piece (buffered).
    current_results: Vec<WallBreakerResult>,

    /// Index into current_results.
    result_idx: usize,

    /// Terms already seen (for deduplication).
    seen_terms: HashSet<String>,

    /// Whether iteration is complete.
    exhausted: bool,
}

impl<'a, D> WallBreakerQuery<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    /// Create a new WallBreaker query.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The dictionary to search
    /// * `query` - The query string
    /// * `max_distance` - Maximum Levenshtein distance
    /// * `splitter` - Pattern splitter to use
    pub fn new(
        dictionary: &'a D,
        query: &str,
        max_distance: usize,
        splitter: &PatternSplitter,
    ) -> Self {
        let pieces = splitter.split(query);

        WallBreakerQuery {
            dictionary,
            query: query.to_string(),
            max_distance,
            pieces,
            current_piece_idx: 0,
            current_results: Vec::new(),
            result_idx: 0,
            seen_terms: HashSet::new(),
            exhausted: false,
        }
    }

    /// Process the next piece and populate current_results.
    fn process_next_piece(&mut self) -> bool {
        while self.current_piece_idx < self.pieces.len() {
            let piece = &self.pieces[self.current_piece_idx];
            self.current_piece_idx += 1;

            // Skip empty pieces
            if piece.is_empty() {
                continue;
            }

            // Find exact substring matches for this piece
            let substring_matches = self.dictionary.find_exact_substring(&piece.content);

            // Extend each match bidirectionally
            for match_info in &substring_matches {
                let extension = BidirectionalExtension::new(
                    match_info,
                    &self.query,
                    piece.start_offset,
                    piece.end_offset,
                    self.max_distance,
                );

                let extensions = extension.extend();

                for (term, _distance) in extensions {
                    // Skip if already seen
                    if self.seen_terms.contains(&term) {
                        continue;
                    }

                    // Verify the distance is within bounds
                    // The extension may have computed partial distances,
                    // so we verify with actual Levenshtein computation
                    let actual_distance = levenshtein_distance(&self.query, &term);
                    if actual_distance <= self.max_distance {
                        self.seen_terms.insert(term.clone());
                        self.current_results
                            .push(WallBreakerResult::new(term, actual_distance));
                    }
                }
            }

            // If we found results, return
            if !self.current_results.is_empty() {
                self.result_idx = 0;
                return true;
            }
        }

        false
    }
}

impl<'a, D> Iterator for WallBreakerQuery<'a, D>
where
    D: Dictionary + SubstringDictionary,
    D::Node: BidirectionalDictionaryNode,
    <D::Node as crate::dictionary::DictionaryNode>::Unit: Into<u32>,
{
    type Item = WallBreakerResult;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        loop {
            // Try to get next result from current batch
            if self.result_idx < self.current_results.len() {
                let result = self.current_results[self.result_idx].clone();
                self.result_idx += 1;
                return Some(result);
            }

            // Clear current batch and process next piece
            self.current_results.clear();
            self.result_idx = 0;

            if !self.process_next_piece() {
                self.exhausted = true;
                return None;
            }
        }
    }
}

/// Compute Levenshtein distance between two strings.
///
/// Uses the standard Wagner-Fischer dynamic programming algorithm.
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();

    let m = chars1.len();
    let n = chars2.len();

    // Handle empty strings
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two-row optimization
    let mut prev_row: Vec<usize> = (0..=n).collect();
    let mut curr_row: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr_row[0] = i;

        for j in 1..=n {
            let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };

            curr_row[j] = (prev_row[j] + 1) // deletion
                .min(curr_row[j - 1] + 1) // insertion
                .min(prev_row[j - 1] + cost); // substitution
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("saturday", "sunday"), 3);
        assert_eq!(levenshtein_distance("hello", "helo"), 1);
        assert_eq!(levenshtein_distance("cathedral", "cathedrel"), 1);
    }

    #[test]
    fn test_wallbreaker_result() {
        let result = WallBreakerResult::new("hello".to_string(), 1);
        assert_eq!(result.term, "hello");
        assert_eq!(result.distance, 1);
    }
}
