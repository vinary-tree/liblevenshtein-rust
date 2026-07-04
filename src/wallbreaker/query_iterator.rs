//! Query iterator for WallBreaker algorithm.
//!
//! The main query iterator that orchestrates:
//! 1. Pattern splitting into pieces (algorithm-dependent count)
//! 2. Finding exact substring matches for each piece
//! 3. Bidirectional extension from matches
//! 4. Deduplication of results

use std::collections::VecDeque;

use crate::distance::{
    create_memo_cache, merge_and_split_distance, standard_distance_bounded,
    transposition_distance_bounded,
};
#[cfg(test)]
use crate::distance::{standard_distance, transposition_distance};
use crate::transducer::Algorithm;
use libdictenstein::substring::{BidirectionalDictionaryNode, SubstringDictionary};
use libdictenstein::Dictionary;
use rustc_hash::FxHashSet;

use super::extension::BidirectionalExtension;
use super::pattern_splitter::{PatternPiece, PatternSplitter};

type SeenTerms = FxHashSet<Box<str>>;

fn compute_distance_within(
    algorithm: Algorithm,
    max_distance: usize,
    source: &str,
    target: &str,
) -> Option<usize> {
    match algorithm {
        Algorithm::Standard => standard_distance_bounded(source, target, max_distance),
        Algorithm::Transposition => transposition_distance_bounded(source, target, max_distance),
        Algorithm::MergeAndSplit => {
            if source.chars().count().abs_diff(target.chars().count()) > max_distance {
                return None;
            }

            let cache = create_memo_cache();
            let distance = merge_and_split_distance(source, target, &cache);
            (distance <= max_distance).then_some(distance)
        }
    }
}

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
/// 1. Split query into pieces (algorithm-dependent count)
/// 2. For each piece, find exact substring matches in dictionary
/// 3. Extend each match bidirectionally using Levenshtein filters
/// 4. Verify distance using the correct algorithm-specific function
/// 5. Deduplicate and yield results
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

    /// The edit distance algorithm to use for verification.
    algorithm: Algorithm,

    /// Pattern pieces from splitting.
    pieces: Vec<PatternPiece>,

    /// Current piece index being processed.
    current_piece_idx: usize,

    /// Results from current piece (buffered).
    current_results: VecDeque<WallBreakerResult>,

    /// Terms already seen (for deduplication).
    seen_terms: SeenTerms,

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
    /// * `max_distance` - Maximum edit distance
    /// * `splitter` - Pattern splitter to use (contains algorithm info)
    pub fn new(
        dictionary: &'a D,
        query: &str,
        max_distance: usize,
        splitter: &PatternSplitter,
    ) -> Self {
        let pieces = splitter.split(query);
        let algorithm = splitter.algorithm();

        WallBreakerQuery {
            dictionary,
            query: query.to_string(),
            max_distance,
            algorithm,
            pieces,
            current_piece_idx: 0,
            current_results: VecDeque::new(),
            seen_terms: SeenTerms::default(),
            exhausted: false,
        }
    }

    /// Compute edit distance using the configured algorithm.
    fn compute_distance_within(&self, s1: &str, s2: &str) -> Option<usize> {
        compute_distance_within(self.algorithm, self.max_distance, s1, s2)
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
                    if self.seen_terms.contains(term.as_str()) {
                        continue;
                    }

                    // Verify the distance is within bounds using the correct algorithm
                    // The extension may have computed partial distances,
                    // so we verify with actual distance computation
                    if let Some(actual_distance) = self.compute_distance_within(&self.query, &term)
                    {
                        self.seen_terms
                            .insert(term.as_str().to_owned().into_boxed_str());
                        self.current_results
                            .push_back(WallBreakerResult::new(term, actual_distance));
                    }
                }
            }

            // If we found results, return
            if !self.current_results.is_empty() {
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
            if let Some(result) = self.current_results.pop_front() {
                return Some(result);
            }

            if !self.process_next_piece() {
                self.exhausted = true;
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        // Now using the distance module functions
        assert_eq!(standard_distance("", ""), 0);
        assert_eq!(standard_distance("abc", ""), 3);
        assert_eq!(standard_distance("", "abc"), 3);
        assert_eq!(standard_distance("abc", "abc"), 0);
        assert_eq!(standard_distance("kitten", "sitting"), 3);
        assert_eq!(standard_distance("saturday", "sunday"), 3);
        assert_eq!(standard_distance("hello", "helo"), 1);
        assert_eq!(standard_distance("cathedral", "cathedrel"), 1);
    }

    #[test]
    fn test_transposition_distance() {
        assert_eq!(transposition_distance("ab", "ba"), 1);
        assert_eq!(transposition_distance("test", "tset"), 1);
        // Transposition is more efficient than standard for swaps
        assert_eq!(transposition_distance("abc", "acb"), 1);
    }

    #[test]
    fn test_merge_and_split_distance() {
        let cache = create_memo_cache();
        assert_eq!(merge_and_split_distance("", "", &cache), 0);
        assert_eq!(merge_and_split_distance("abc", "abc", &cache), 0);
        assert_eq!(merge_and_split_distance("test", "best", &cache), 1);
    }

    #[test]
    fn test_compute_distance_within() {
        assert_eq!(
            compute_distance_within(Algorithm::Standard, 2, "kitten", "sitting"),
            None
        );
        assert_eq!(
            compute_distance_within(Algorithm::Standard, 3, "kitten", "sitting"),
            Some(3)
        );
        assert_eq!(
            compute_distance_within(Algorithm::Transposition, 1, "ab", "ba"),
            Some(1)
        );
        assert_eq!(
            compute_distance_within(Algorithm::MergeAndSplit, 0, "a", "abc"),
            None
        );
        assert_eq!(
            compute_distance_within(Algorithm::MergeAndSplit, 1, "m", "rn"),
            Some(1)
        );
    }

    #[test]
    fn test_wallbreaker_result() {
        let result = WallBreakerResult::new("hello".to_string(), 1);
        assert_eq!(result.term, "hello");
        assert_eq!(result.distance, 1);
    }
}
