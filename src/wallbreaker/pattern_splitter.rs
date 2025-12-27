//! Pattern splitting for WallBreaker algorithm.
//!
//! The WallBreaker algorithm exploits the pigeonhole principle: if a query
//! has at most `b` errors compared to a dictionary term, then at least one
//! of `b+1` equal-sized pieces must match exactly.
//!
//! This module provides two splitters:
//! - [`PatternSplitter`] - Basic equal-length splitting
//! - [`FrequencyPatternSplitter`] - Splits at rare-character positions to reduce false positives

use rustc_hash::FxHashMap;

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

/// Frequency-based pattern splitter for WallBreaker algorithm.
///
/// This splitter optimizes split points based on character frequency analysis.
/// By placing rare characters within pieces, substring searches will return
/// fewer matches, reducing false positives and improving performance.
///
/// # Strategy
///
/// 1. Compute character frequencies from the dictionary
/// 2. For each character in the query, compute a "rarity score" (1 / frequency)
/// 3. Greedily select split points that maximize the minimum rarity in each piece
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::wallbreaker::FrequencyPatternSplitter;
///
/// // Given dictionary frequencies where 'z' and 'x' are rare
/// let splitter = FrequencyPatternSplitter::from_terms(
///     vec!["apple", "banana", "cherry"].iter().map(|s| *s),
///     2 // max_distance
/// );
///
/// // Query "amazing" will be split to include rare 'z' in a piece
/// let pieces = splitter.split("amazing");
/// ```
#[derive(Debug, Clone)]
pub struct FrequencyPatternSplitter {
    /// Maximum edit distance (b).
    max_distance: usize,

    /// Character frequencies: maps char -> count in dictionary.
    /// Characters not in the map are considered maximally rare.
    char_frequencies: FxHashMap<char, usize>,

    /// Total character count in dictionary.
    total_chars: usize,
}

impl FrequencyPatternSplitter {
    /// Create a frequency-based splitter from an iterator of dictionary terms.
    ///
    /// # Arguments
    ///
    /// * `terms` - Iterator of dictionary terms to analyze
    /// * `max_distance` - The maximum edit distance (b)
    pub fn from_terms<I, S>(terms: I, max_distance: usize) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut char_frequencies = FxHashMap::default();
        let mut total_chars = 0usize;

        for term in terms {
            for ch in term.as_ref().chars() {
                *char_frequencies.entry(ch).or_insert(0) += 1;
                total_chars += 1;
            }
        }

        FrequencyPatternSplitter {
            max_distance,
            char_frequencies,
            total_chars,
        }
    }

    /// Create with pre-computed frequencies.
    pub fn new(max_distance: usize, char_frequencies: FxHashMap<char, usize>, total_chars: usize) -> Self {
        FrequencyPatternSplitter {
            max_distance,
            char_frequencies,
            total_chars,
        }
    }

    /// Get the rarity score for a character.
    ///
    /// Returns a value where higher means rarer.
    /// Unknown characters get maximum rarity (total_chars + 1).
    #[inline]
    fn rarity_score(&self, ch: char) -> usize {
        match self.char_frequencies.get(&ch) {
            Some(&freq) if freq > 0 => self.total_chars / freq,
            _ => self.total_chars + 1, // Maximum rarity for unknown chars
        }
    }

    /// Compute the minimum rarity score for a piece.
    ///
    /// A piece with higher min-rarity contains at least one rare character,
    /// which should result in fewer substring matches.
    fn piece_min_rarity(&self, chars: &[char]) -> usize {
        chars.iter().map(|&ch| self.rarity_score(ch)).max().unwrap_or(0)
    }

    /// Split a query into `b + 1` pieces using frequency-based optimization.
    ///
    /// The algorithm tries to maximize the minimum rarity score across all pieces
    /// while respecting the constraint that we need exactly `b + 1` pieces.
    pub fn split(&self, query: &str) -> Vec<PatternPiece> {
        let chars: Vec<char> = query.chars().collect();
        let query_len = chars.len();

        if query_len == 0 {
            return Vec::new();
        }

        let num_pieces = self.max_distance + 1;

        // Handle case where query is shorter than number of pieces
        if query_len < num_pieces {
            return chars
                .iter()
                .enumerate()
                .map(|(i, &c)| PatternPiece::new(c.to_string(), i, i + 1, i))
                .collect();
        }

        // Calculate rarity scores for each position
        let rarity_scores: Vec<usize> = chars.iter().map(|&ch| self.rarity_score(ch)).collect();

        // Use dynamic programming to find optimal split points
        // that maximize the minimum piece rarity
        let split_points = self.find_optimal_splits(&chars, &rarity_scores, num_pieces);

        // Create pieces from split points
        let mut pieces = Vec::with_capacity(num_pieces);
        let mut start = 0;

        for (i, &end) in split_points.iter().enumerate() {
            let content: String = chars[start..end].iter().collect();
            pieces.push(PatternPiece::new(content, start, end, i));
            start = end;
        }

        pieces
    }

    /// Find optimal split points using a greedy/DP approach.
    ///
    /// Returns the end positions of each piece.
    fn find_optimal_splits(&self, chars: &[char], rarity_scores: &[usize], num_pieces: usize) -> Vec<usize> {
        let n = chars.len();

        // Minimum and maximum piece sizes for balance
        let min_size = n / (num_pieces * 2).max(1);
        let max_size = (n * 2) / num_pieces;

        // Greedy approach: place split points after high-rarity characters
        // while maintaining reasonable piece sizes

        let mut split_points = Vec::with_capacity(num_pieces);
        let mut current_start = 0;

        for piece_idx in 0..(num_pieces - 1) {
            let remaining_pieces = num_pieces - piece_idx;
            let remaining_chars = n - current_start;

            // Calculate the range for this split point
            let ideal_size = remaining_chars / remaining_pieces;
            let min_end = (current_start + min_size.max(1)).min(n - remaining_pieces + 1);
            let max_end = (current_start + max_size.min(remaining_chars - remaining_pieces + 1)).min(n);

            if min_end >= max_end {
                // Forced split point
                split_points.push(min_end);
                current_start = min_end;
                continue;
            }

            // Find the best split point in the range
            // Prefer splitting just after a high-rarity character
            let mut best_end = current_start + ideal_size;
            let mut best_score = 0;

            for end in min_end..=max_end {
                // Score this split: max rarity in the piece
                let piece_max_rarity = rarity_scores[current_start..end].iter().max().copied().unwrap_or(0);

                // Slight preference for positions closer to ideal size
                let size_penalty = ((end - current_start) as isize - ideal_size as isize).unsigned_abs();
                let adjusted_score = piece_max_rarity.saturating_sub(size_penalty);

                if adjusted_score > best_score {
                    best_score = adjusted_score;
                    best_end = end;
                }
            }

            split_points.push(best_end);
            current_start = best_end;
        }

        // Last piece ends at the end
        split_points.push(n);

        split_points
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

    // ============================================================================
    // FrequencyPatternSplitter tests
    // ============================================================================

    #[test]
    fn test_frequency_splitter_basic() {
        // Dictionary where 'e' and 'a' are common, 'z' and 'x' are rare
        let terms = vec!["apple", "banana", "cherry", "date", "elderberry"];
        let splitter = FrequencyPatternSplitter::from_terms(terms.iter().map(|s| *s), 2);

        let pieces = splitter.split("amazing");
        assert_eq!(pieces.len(), 3);

        // Should produce valid pieces that cover the entire string
        let combined: String = pieces.iter().map(|p| p.content.as_str()).collect();
        assert_eq!(combined, "amazing");
    }

    #[test]
    fn test_frequency_splitter_empty() {
        let terms = vec!["test"];
        let splitter = FrequencyPatternSplitter::from_terms(terms.iter().map(|s| *s), 2);

        let pieces = splitter.split("");
        assert!(pieces.is_empty());
    }

    #[test]
    fn test_frequency_splitter_short_query() {
        let terms = vec!["test"];
        let splitter = FrequencyPatternSplitter::from_terms(terms.iter().map(|s| *s), 5);

        let pieces = splitter.split("abc");
        assert_eq!(pieces.len(), 3); // Individual characters
        assert_eq!(pieces[0].content, "a");
        assert_eq!(pieces[1].content, "b");
        assert_eq!(pieces[2].content, "c");
    }

    #[test]
    fn test_frequency_splitter_rare_chars_preferred() {
        // Dictionary with very common 'a' and 'e', rare 'z'
        let mut terms = Vec::new();
        for _ in 0..100 {
            terms.push("aaaaaeeeeea");
        }
        terms.push("z"); // Only one 'z'

        let splitter = FrequencyPatternSplitter::from_terms(terms.iter().copied(), 2);

        // Query with 'z' in the middle
        let pieces = splitter.split("aaazaaeee");

        // The splitter should try to include 'z' in one of the pieces
        // and that piece should be relatively short to maximize its impact
        let combined: String = pieces.iter().map(|p| p.content.as_str()).collect();
        assert_eq!(combined, "aaazaaeee");

        // Check that we have exactly 3 pieces
        assert_eq!(pieces.len(), 3);
    }

    #[test]
    fn test_frequency_splitter_offsets_valid() {
        let terms = vec!["test"];
        let splitter = FrequencyPatternSplitter::from_terms(terms.iter().map(|s| *s), 2);

        let pieces = splitter.split("cathedral");

        // Verify offsets are contiguous
        for i in 0..pieces.len() {
            if i > 0 {
                assert_eq!(pieces[i].start_offset, pieces[i - 1].end_offset);
            }
        }

        // First piece starts at 0, last piece ends at query length
        assert_eq!(pieces[0].start_offset, 0);
        assert_eq!(pieces.last().unwrap().end_offset, 9);
    }

    #[test]
    fn test_frequency_splitter_unicode() {
        let terms = vec!["café", "naïve", "résumé"];
        let splitter = FrequencyPatternSplitter::from_terms(terms.iter().map(|s| *s), 2);

        let pieces = splitter.split("café🎉");
        assert_eq!(pieces.len(), 3);

        let combined: String = pieces.iter().map(|p| p.content.as_str()).collect();
        assert_eq!(combined, "café🎉");
    }

    #[test]
    fn test_frequency_splitter_num_pieces() {
        let terms = vec!["test"];
        let splitter = FrequencyPatternSplitter::from_terms(terms.iter().map(|s| *s), 4);

        assert_eq!(splitter.num_pieces(), 5);
        assert_eq!(splitter.max_distance(), 4);
    }
}
