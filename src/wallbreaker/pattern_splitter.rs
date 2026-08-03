//! Pattern splitting for WallBreaker algorithm.
//!
//! The WallBreaker algorithm exploits the pigeonhole principle: if a query
//! has at most `b` errors compared to a dictionary term, then at least one
//! piece must match exactly.
//!
//! ## Piece Count by Algorithm (Formally Verified)
//!
//! The number of pieces required depends on the edit distance algorithm:
//!
//! - **Standard Levenshtein**: `b+1` pieces suffice
//!   - Each operation (insert, delete, substitute) corrupts at most 1 piece
//!
//! - **Transposition (optimal string alignment)**: `2b+1` pieces required
//!   - Adjacent transpositions can corrupt 2 pieces when spanning boundaries
//!   - Counterexample: "ABCDE" → "ACBDX" (k=2) corrupts all 3 pieces of (k+1)
//!
//! - **MergeAndSplit**: `2b+1` pieces required
//!   - Merge operations span 2 characters, can corrupt 2 pieces at boundaries
//!   - Counterexample: "abcdef" → "aXYf" (k=2) corrupts all 3 pieces of (k+1)
//!
//! These bounds are formally verified in:
//! `docs/verification/wallbreaker/theories/Pigeonhole/WallBreakerPigeonhole.v`
//!
//! This module provides the [`PatternSplitter`] which divides queries into
//! these pieces for the WallBreaker algorithm.

use crate::transducer::Algorithm;

/// A piece of a split pattern.
///
/// Contains the substring content and its position within the original query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternPiece {
    /// The substring content of this piece.
    pub content: String,

    /// Start position (character index) in the original query.
    pub start_offset: usize,

    /// End position (character index, exclusive) in the original query.
    pub end_offset: usize,

    /// The index of this piece (0 to b).
    pub piece_index: usize,
}

impl PatternPiece {
    /// Create a new pattern piece.
    pub fn new(
        content: String,
        start_offset: usize,
        end_offset: usize,
        piece_index: usize,
    ) -> Self {
        PatternPiece {
            content,
            start_offset,
            end_offset,
            piece_index,
        }
    }

    /// Get the length of this piece in characters.
    #[inline]
    pub fn len(&self) -> usize {
        self.content.chars().count()
    }

    /// Check if this piece is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// Pattern splitter for WallBreaker algorithm.
///
/// Divides queries into pieces based on the pigeonhole principle. The number
/// of pieces depends on the algorithm:
///
/// - **Standard**: `b+1` pieces (each op corrupts ≤1 piece)
/// - **Transposition**: `2b+1` pieces (transpositions can corrupt 2 pieces)
/// - **MergeAndSplit**: `2b+1` pieces (merge/split can corrupt 2 pieces)
///
/// # Splitting Strategy
///
/// For a query of length `n` with max distance `b`:
/// - Number of pieces: algorithm-dependent (see above)
/// - Base piece size: `n / num_pieces`
/// - Remainder characters: distributed among first pieces
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wallbreaker::PatternSplitter;
/// use liblevenshtein::transducer::Algorithm;
///
/// // Standard with b=2 → 3 pieces
/// let splitter = PatternSplitter::new(2, Algorithm::Standard);
/// let pieces = splitter.split("cathedral");
/// assert_eq!(pieces.len(), 3);
///
/// // Transposition with b=2 → 5 pieces
/// let splitter = PatternSplitter::new(2, Algorithm::Transposition);
/// let pieces = splitter.split("cathedral");
/// assert_eq!(pieces.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct PatternSplitter {
    /// Maximum edit distance (b).
    max_distance: usize,
    /// Algorithm type (determines piece count formula).
    algorithm: Algorithm,
}

impl PatternSplitter {
    /// Create a new pattern splitter.
    ///
    /// # Arguments
    ///
    /// * `max_distance` - The maximum edit distance (b)
    /// * `algorithm` - The edit distance algorithm (determines piece count)
    ///
    /// # Piece Count Formula (Formally Verified)
    ///
    /// - Standard: `b + 1` pieces
    /// - Transposition: `2b + 1` pieces
    /// - MergeAndSplit: `2b + 1` pieces
    ///
    /// These formulas are proven in `WallBreakerPigeonhole.v`.
    pub fn new(max_distance: usize, algorithm: Algorithm) -> Self {
        PatternSplitter {
            max_distance,
            algorithm,
        }
    }

    /// Create a pattern splitter using the Standard algorithm.
    ///
    /// Equivalent to `PatternSplitter::new(max_distance, Algorithm::Standard)`.
    pub fn standard(max_distance: usize) -> Self {
        Self::new(max_distance, Algorithm::Standard)
    }

    /// Split a query into `b + 1` pieces.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string to split
    ///
    /// # Returns
    ///
    /// A vector of [`PatternPiece`]s. The vector will have `min(b + 1, query_len)`
    /// elements. If the query is shorter than `b + 1`, some pieces may be empty.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let splitter = PatternSplitter::new(2);
    /// let pieces = splitter.split("hello");
    /// // With b=2: 3 pieces from 5 chars
    /// // "he" (2), "ll" (2), "o" (1)
    /// ```
    pub fn split(&self, query: &str) -> Vec<PatternPiece> {
        let chars: Vec<char> = query.chars().collect();
        let query_len = chars.len();

        if query_len == 0 {
            return Vec::new();
        }

        let num_pieces = self.num_pieces();

        // Handle case where query is shorter than number of pieces
        if query_len < num_pieces {
            // Return individual characters as pieces
            return chars
                .iter()
                .enumerate()
                .map(|(i, &c)| PatternPiece::new(c.to_string(), i, i + 1, i))
                .collect();
        }

        let base_size = query_len / num_pieces;
        let remainder = query_len % num_pieces;

        let mut pieces = Vec::with_capacity(num_pieces);
        let mut start = 0;

        for i in 0..num_pieces {
            // Distribute remainder among first pieces
            let piece_size = base_size + if i < remainder { 1 } else { 0 };
            let end = start + piece_size;

            let content: String = chars[start..end].iter().collect();
            pieces.push(PatternPiece::new(content, start, end, i));

            start = end;
        }

        pieces
    }

    /// Get the number of pieces that will be created.
    ///
    /// Returns the algorithm-specific piece count:
    /// - Standard: `b + 1`
    /// - Transposition or unrestricted Damerau: `2b + 1`
    /// - MergeAndSplit: `2b + 1`
    #[inline]
    pub fn num_pieces(&self) -> usize {
        match self.algorithm {
            Algorithm::Standard => self.max_distance + 1,
            Algorithm::Transposition | Algorithm::MergeAndSplit | Algorithm::DamerauLevenshtein => {
                self.max_distance.saturating_mul(2).saturating_add(1)
            }
        }
    }

    /// Get the algorithm.
    #[inline]
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get the maximum distance.
    #[inline]
    pub fn max_distance(&self) -> usize {
        self.max_distance
    }

    /// Calculate the minimum piece length for a given query length.
    ///
    /// This is useful for filtering: pieces shorter than this cannot
    /// guarantee a match within the error bound.
    #[inline]
    pub fn min_piece_length(&self, query_len: usize) -> usize {
        if query_len < self.num_pieces() {
            1
        } else {
            query_len / self.num_pieces()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_even_standard() {
        // Standard algorithm with k=2 → 3 pieces
        let splitter = PatternSplitter::standard(2);
        let pieces = splitter.split("cathedral"); // 9 chars

        assert_eq!(pieces.len(), 3);
        assert_eq!(pieces[0].content, "cat");
        assert_eq!(pieces[1].content, "hed");
        assert_eq!(pieces[2].content, "ral");

        // Check offsets
        assert_eq!(pieces[0].start_offset, 0);
        assert_eq!(pieces[0].end_offset, 3);
        assert_eq!(pieces[1].start_offset, 3);
        assert_eq!(pieces[1].end_offset, 6);
        assert_eq!(pieces[2].start_offset, 6);
        assert_eq!(pieces[2].end_offset, 9);
    }

    #[test]
    fn test_split_uneven_standard() {
        // Standard algorithm with k=2 → 3 pieces
        let splitter = PatternSplitter::standard(2);
        let pieces = splitter.split("hello"); // 5 chars

        assert_eq!(pieces.len(), 3);
        // 5 / 3 = 1, remainder 2
        // First 2 pieces get extra char
        assert_eq!(pieces[0].content, "he"); // 2 chars
        assert_eq!(pieces[1].content, "ll"); // 2 chars
        assert_eq!(pieces[2].content, "o"); // 1 char
    }

    #[test]
    fn test_split_short_query() {
        // Standard algorithm with k=5 → 6 pieces
        let splitter = PatternSplitter::standard(5);
        let pieces = splitter.split("abc"); // 3 chars < 6 pieces

        // Should get individual characters
        assert_eq!(pieces.len(), 3);
        assert_eq!(pieces[0].content, "a");
        assert_eq!(pieces[1].content, "b");
        assert_eq!(pieces[2].content, "c");
    }

    #[test]
    fn test_split_empty() {
        let splitter = PatternSplitter::standard(2);
        let pieces = splitter.split("");
        assert!(pieces.is_empty());
    }

    #[test]
    fn test_split_single_char() {
        // Standard algorithm with k=0 → 1 piece
        let splitter = PatternSplitter::standard(0);
        let pieces = splitter.split("x");

        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].content, "x");
    }

    #[test]
    fn test_split_unicode() {
        // Standard algorithm with k=2 → 3 pieces
        let splitter = PatternSplitter::standard(2);
        let pieces = splitter.split("café🎉"); // 5 chars

        assert_eq!(pieces.len(), 3);
        // 5 / 3 = 1, remainder 2
        assert_eq!(pieces[0].content, "ca"); // 2 chars
        assert_eq!(pieces[1].content, "fé"); // 2 chars
        assert_eq!(pieces[2].content, "🎉"); // 1 char
    }

    #[test]
    fn test_piece_indices() {
        let splitter = PatternSplitter::standard(2);
        let pieces = splitter.split("abcdef");

        for (i, piece) in pieces.iter().enumerate() {
            assert_eq!(piece.piece_index, i);
        }
    }

    #[test]
    fn test_min_piece_length_standard() {
        // Standard algorithm with k=2 → 3 pieces
        let splitter = PatternSplitter::standard(2);

        assert_eq!(splitter.min_piece_length(9), 3); // 9/3 = 3
        assert_eq!(splitter.min_piece_length(10), 3); // 10/3 = 3
        assert_eq!(splitter.min_piece_length(2), 1); // short query
    }

    // Algorithm-specific piece count tests (formally verified in WallBreakerPigeonhole.v)

    #[test]
    fn test_num_pieces_standard() {
        // Standard: k+1 pieces
        assert_eq!(PatternSplitter::new(0, Algorithm::Standard).num_pieces(), 1);
        assert_eq!(PatternSplitter::new(1, Algorithm::Standard).num_pieces(), 2);
        assert_eq!(PatternSplitter::new(2, Algorithm::Standard).num_pieces(), 3);
        assert_eq!(PatternSplitter::new(5, Algorithm::Standard).num_pieces(), 6);
    }

    #[test]
    fn test_num_pieces_transposition() {
        // Transposition: 2k+1 pieces (proven in WallBreakerPigeonhole.v)
        assert_eq!(
            PatternSplitter::new(0, Algorithm::Transposition).num_pieces(),
            1
        );
        assert_eq!(
            PatternSplitter::new(1, Algorithm::Transposition).num_pieces(),
            3
        );
        assert_eq!(
            PatternSplitter::new(2, Algorithm::Transposition).num_pieces(),
            5
        );
        assert_eq!(
            PatternSplitter::new(5, Algorithm::Transposition).num_pieces(),
            11
        );
    }

    #[test]
    fn test_num_pieces_merge_and_split() {
        // MergeAndSplit: 2k+1 pieces (proven in WallBreakerPigeonhole.v)
        assert_eq!(
            PatternSplitter::new(0, Algorithm::MergeAndSplit).num_pieces(),
            1
        );
        assert_eq!(
            PatternSplitter::new(1, Algorithm::MergeAndSplit).num_pieces(),
            3
        );
        assert_eq!(
            PatternSplitter::new(2, Algorithm::MergeAndSplit).num_pieces(),
            5
        );
        assert_eq!(
            PatternSplitter::new(5, Algorithm::MergeAndSplit).num_pieces(),
            11
        );
    }

    #[test]
    fn test_split_transposition_more_pieces() {
        // Transposition with k=2 → 5 pieces (not 3)
        let splitter = PatternSplitter::new(2, Algorithm::Transposition);
        let pieces = splitter.split("cathedral"); // 9 chars into 5 pieces

        assert_eq!(pieces.len(), 5);
        // 9 / 5 = 1, remainder 4 → first 4 pieces get 2 chars, last gets 1
        assert_eq!(pieces[0].content, "ca"); // 2 chars
        assert_eq!(pieces[1].content, "th"); // 2 chars
        assert_eq!(pieces[2].content, "ed"); // 2 chars
        assert_eq!(pieces[3].content, "ra"); // 2 chars
        assert_eq!(pieces[4].content, "l"); // 1 char
    }

    #[test]
    fn test_split_merge_and_split_more_pieces() {
        // MergeAndSplit with k=2 → 5 pieces (not 3)
        let splitter = PatternSplitter::new(2, Algorithm::MergeAndSplit);
        let pieces = splitter.split("cathedral"); // 9 chars into 5 pieces

        assert_eq!(pieces.len(), 5);
        // Same distribution as Transposition
        assert_eq!(pieces[0].content, "ca");
        assert_eq!(pieces[1].content, "th");
        assert_eq!(pieces[2].content, "ed");
        assert_eq!(pieces[3].content, "ra");
        assert_eq!(pieces[4].content, "l");
    }

    #[test]
    fn test_algorithm_getter() {
        let standard = PatternSplitter::standard(2);
        assert!(matches!(standard.algorithm(), Algorithm::Standard));

        let transposition = PatternSplitter::new(2, Algorithm::Transposition);
        assert!(matches!(
            transposition.algorithm(),
            Algorithm::Transposition
        ));

        let merge_split = PatternSplitter::new(2, Algorithm::MergeAndSplit);
        assert!(matches!(merge_split.algorithm(), Algorithm::MergeAndSplit));
    }

    #[test]
    fn test_min_piece_length_transposition() {
        // Transposition with k=2 → 5 pieces
        let splitter = PatternSplitter::new(2, Algorithm::Transposition);

        assert_eq!(splitter.min_piece_length(10), 2); // 10/5 = 2
        assert_eq!(splitter.min_piece_length(15), 3); // 15/5 = 3
        assert_eq!(splitter.min_piece_length(4), 1); // short query
    }
}
