//! Pattern splitting for WallBreaker algorithm.
//!
//! The WallBreaker algorithm exploits the pigeonhole principle: if a query
//! has at most `b` errors compared to a dictionary term, then at least one
//! of `b+1` equal-sized pieces must match exactly.
//!
//! This module provides the [`PatternSplitter`] which divides queries into
//! these pieces for the WallBreaker algorithm.

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
    pub fn new(content: String, start_offset: usize, end_offset: usize, piece_index: usize) -> Self {
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
/// Divides queries into `b+1` approximately equal pieces, where `b` is the
/// maximum allowed edit distance. By the pigeonhole principle, at least
/// one piece must match exactly in any valid match.
///
/// # Splitting Strategy
///
/// For a query of length `n` with max distance `b`:
/// - Number of pieces: `b + 1`
/// - Base piece size: `n / (b + 1)`
/// - Remainder characters: distributed among first pieces
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wallbreaker::PatternSplitter;
///
/// let splitter = PatternSplitter::new(2); // max_distance = 2
///
/// // "cathedral" (9 chars) with b=2 → 3 pieces
/// let pieces = splitter.split("cathedral");
/// assert_eq!(pieces.len(), 3);
/// // pieces[0] = "cat" (3 chars)
/// // pieces[1] = "hed" (3 chars)
/// // pieces[2] = "ral" (3 chars)
/// ```
#[derive(Debug, Clone)]
pub struct PatternSplitter {
    /// Maximum edit distance (b).
    max_distance: usize,
}

impl PatternSplitter {
    /// Create a new pattern splitter.
    ///
    /// # Arguments
    ///
    /// * `max_distance` - The maximum edit distance (b). The splitter will
    ///   create `b + 1` pieces.
    pub fn new(max_distance: usize) -> Self {
        PatternSplitter { max_distance }
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

        let num_pieces = self.max_distance + 1;

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
    #[inline]
    pub fn num_pieces(&self) -> usize {
        self.max_distance + 1
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
    fn test_split_even() {
        let splitter = PatternSplitter::new(2); // 3 pieces
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
    fn test_split_uneven() {
        let splitter = PatternSplitter::new(2); // 3 pieces
        let pieces = splitter.split("hello"); // 5 chars

        assert_eq!(pieces.len(), 3);
        // 5 / 3 = 1, remainder 2
        // First 2 pieces get extra char
        assert_eq!(pieces[0].content, "he"); // 2 chars
        assert_eq!(pieces[1].content, "ll"); // 2 chars
        assert_eq!(pieces[2].content, "o");  // 1 char
    }

    #[test]
    fn test_split_short_query() {
        let splitter = PatternSplitter::new(5); // 6 pieces
        let pieces = splitter.split("abc"); // 3 chars < 6 pieces

        // Should get individual characters
        assert_eq!(pieces.len(), 3);
        assert_eq!(pieces[0].content, "a");
        assert_eq!(pieces[1].content, "b");
        assert_eq!(pieces[2].content, "c");
    }

    #[test]
    fn test_split_empty() {
        let splitter = PatternSplitter::new(2);
        let pieces = splitter.split("");
        assert!(pieces.is_empty());
    }

    #[test]
    fn test_split_single_char() {
        let splitter = PatternSplitter::new(0); // 1 piece
        let pieces = splitter.split("x");

        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].content, "x");
    }

    #[test]
    fn test_split_unicode() {
        let splitter = PatternSplitter::new(2); // 3 pieces
        let pieces = splitter.split("café🎉"); // 5 chars

        assert_eq!(pieces.len(), 3);
        // 5 / 3 = 1, remainder 2
        assert_eq!(pieces[0].content, "ca"); // 2 chars
        assert_eq!(pieces[1].content, "fé"); // 2 chars
        assert_eq!(pieces[2].content, "🎉"); // 1 char
    }

    #[test]
    fn test_piece_indices() {
        let splitter = PatternSplitter::new(2);
        let pieces = splitter.split("abcdef");

        for (i, piece) in pieces.iter().enumerate() {
            assert_eq!(piece.piece_index, i);
        }
    }

    #[test]
    fn test_min_piece_length() {
        let splitter = PatternSplitter::new(2); // 3 pieces

        assert_eq!(splitter.min_piece_length(9), 3);  // 9/3 = 3
        assert_eq!(splitter.min_piece_length(10), 3); // 10/3 = 3
        assert_eq!(splitter.min_piece_length(2), 1);  // short query
    }
}
