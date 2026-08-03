//! Position tracking for streaming grep operations.
//!
//! This module provides utilities to track byte positions, line numbers,
//! and column offsets when processing text in chunks.

use crate::grep::result::{MatchLocation, SourceId};

/// Tracks positions (line numbers, columns, byte offsets) across streaming chunks.
///
/// Used to convert byte offsets in grep matches to human-readable line:column positions.
///
/// # Example
///
/// ```
/// use liblevenshtein::grep::PositionTracker;
///
/// let mut tracker = PositionTracker::new();
/// tracker.advance("Hello\nWorld\n");
///
/// // Byte offset 0 is line 1, column 1
/// assert_eq!(tracker.line_number(0), 1);
/// assert_eq!(tracker.column(0), 1);
///
/// // Byte offset 6 (start of "World") is line 2, column 1
/// assert_eq!(tracker.line_number(6), 2);
/// assert_eq!(tracker.column(6), 1);
/// ```
#[derive(Debug, Clone)]
pub struct PositionTracker {
    /// Total bytes processed so far.
    total_bytes: usize,

    /// Byte offsets where each line starts.
    /// `line_starts[0] = 0` (the first line starts at offset 0).
    /// `line_starts[n]` is the byte offset where line `n + 1` starts.
    line_starts: Vec<usize>,

    /// Lines of text (for context in match output).
    /// Only populated if `store_lines` is true.
    lines: Vec<String>,

    /// Current line being accumulated.
    current_line: String,

    /// Whether to store line content for later retrieval.
    store_lines: bool,
}

impl Default for PositionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionTracker {
    /// Create a new position tracker.
    pub fn new() -> Self {
        Self {
            total_bytes: 0,
            line_starts: vec![0], // First line starts at offset 0
            lines: Vec::new(),
            current_line: String::new(),
            store_lines: true,
        }
    }

    /// Create a position tracker that doesn't store line content.
    ///
    /// Use this for lower memory usage when you don't need line context.
    pub fn without_line_storage() -> Self {
        Self {
            total_bytes: 0,
            line_starts: vec![0],
            lines: Vec::new(),
            current_line: String::new(),
            store_lines: false,
        }
    }

    /// Process a chunk of text and update position tracking.
    pub fn advance(&mut self, chunk: &str) {
        for c in chunk.chars() {
            let char_len = c.len_utf8();
            self.total_bytes += char_len;

            if c == '\n' {
                // End of line - record where next line starts
                self.line_starts.push(self.total_bytes);
                if self.store_lines {
                    self.lines.push(std::mem::take(&mut self.current_line));
                }
            } else if self.store_lines {
                self.current_line.push(c);
            }
        }
    }

    /// Finalize tracking (call after all chunks have been processed).
    ///
    /// This handles the case where the last line doesn't end with a newline.
    pub fn finish(&mut self) {
        // If there's content in current_line, it's an unterminated final line
        if self.store_lines && !self.current_line.is_empty() {
            self.lines.push(std::mem::take(&mut self.current_line));
        }
    }

    /// Get the total number of bytes processed.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Get the total number of lines seen.
    pub fn line_count(&self) -> usize {
        // Number of line starts minus 1, plus 1 if there's unterminated content
        self.line_starts.len()
    }

    /// Get the line number (1-indexed) for a byte offset.
    ///
    /// # Example
    ///
    /// ```
    /// use liblevenshtein::grep::PositionTracker;
    ///
    /// let mut tracker = PositionTracker::new();
    /// tracker.advance("line1\nline2\n");
    ///
    /// assert_eq!(tracker.line_number(0), 1);   // 'l' of "line1"
    /// assert_eq!(tracker.line_number(5), 1);   // '\n' after "line1"
    /// assert_eq!(tracker.line_number(6), 2);   // 'l' of "line2"
    /// ```
    pub fn line_number(&self, byte_offset: usize) -> usize {
        // Binary search for the line containing this offset
        match self.line_starts.binary_search(&byte_offset) {
            Ok(idx) => idx + 1, // Exact match on line start
            Err(idx) => idx,    // Falls within line (idx-1), return 1-indexed
        }
    }

    /// Get the column (1-indexed) for a byte offset.
    ///
    /// Note: This is a byte column, not a character column.
    /// For multi-byte UTF-8 characters, this may differ from visual column.
    pub fn column(&self, byte_offset: usize) -> usize {
        let line_num = self.line_number(byte_offset);
        let line_start = self.line_starts.get(line_num - 1).copied().unwrap_or(0);
        byte_offset - line_start + 1
    }

    /// Get the text of a specific line (1-indexed).
    ///
    /// Returns `None` if line storage is disabled or line doesn't exist.
    pub fn get_line(&self, line_number: usize) -> Option<&str> {
        if !self.store_lines {
            return None;
        }
        // line_number is 1-indexed
        self.lines
            .get(line_number.saturating_sub(1))
            .map(|s| s.as_str())
    }

    /// Get the text of the line containing a byte offset.
    pub fn get_line_at(&self, byte_offset: usize) -> Option<&str> {
        let line_num = self.line_number(byte_offset);
        self.get_line(line_num)
    }

    /// Create a MatchLocation from a byte offset.
    pub fn locate(&self, byte_offset: usize, source: &SourceId) -> MatchLocation {
        MatchLocation {
            file_path: source.file_path.clone(),
            archive_entry: source.archive_entry.clone(),
            line_number: self.line_number(byte_offset),
            column: self.column(byte_offset),
            byte_offset,
        }
    }

    /// Get the byte range for a specific line (1-indexed).
    ///
    /// Returns `(start, end)` where `end` is exclusive.
    pub fn line_byte_range(&self, line_number: usize) -> Option<(usize, usize)> {
        if line_number == 0 || line_number > self.line_starts.len() {
            return None;
        }

        let start = self.line_starts[line_number - 1];
        let end = self
            .line_starts
            .get(line_number)
            .copied()
            .unwrap_or(self.total_bytes);

        Some((start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_line() {
        let mut tracker = PositionTracker::new();
        tracker.advance("Hello, World!");
        tracker.finish();

        assert_eq!(tracker.line_number(0), 1);
        assert_eq!(tracker.column(0), 1);
        assert_eq!(tracker.line_number(7), 1);
        assert_eq!(tracker.column(7), 8);
        assert_eq!(tracker.get_line(1), Some("Hello, World!"));
    }

    #[test]
    fn test_multiple_lines() {
        let mut tracker = PositionTracker::new();
        tracker.advance("Hello\nWorld\nTest\n");
        tracker.finish();

        // Line 1: "Hello" (bytes 0-5, newline at 5)
        assert_eq!(tracker.line_number(0), 1);
        assert_eq!(tracker.column(0), 1);
        assert_eq!(tracker.line_number(4), 1);
        assert_eq!(tracker.column(4), 5);

        // Line 2: "World" (bytes 6-11, newline at 11)
        assert_eq!(tracker.line_number(6), 2);
        assert_eq!(tracker.column(6), 1);

        // Line 3: "Test" (bytes 12-16, newline at 16)
        assert_eq!(tracker.line_number(12), 3);
        assert_eq!(tracker.column(12), 1);

        assert_eq!(tracker.get_line(1), Some("Hello"));
        assert_eq!(tracker.get_line(2), Some("World"));
        assert_eq!(tracker.get_line(3), Some("Test"));
    }

    #[test]
    fn test_chunked_processing() {
        let mut tracker = PositionTracker::new();
        tracker.advance("Hello\n");
        tracker.advance("World");
        tracker.finish();

        assert_eq!(tracker.line_number(0), 1);
        assert_eq!(tracker.line_number(6), 2);
        assert_eq!(tracker.get_line(1), Some("Hello"));
        assert_eq!(tracker.get_line(2), Some("World"));
    }

    #[test]
    fn test_empty_lines() {
        let mut tracker = PositionTracker::new();
        tracker.advance("A\n\nB\n");
        tracker.finish();

        assert_eq!(tracker.get_line(1), Some("A"));
        assert_eq!(tracker.get_line(2), Some(""));
        assert_eq!(tracker.get_line(3), Some("B"));
    }

    #[test]
    fn test_line_byte_range() {
        let mut tracker = PositionTracker::new();
        tracker.advance("Hello\nWorld\n");
        tracker.finish();

        assert_eq!(tracker.line_byte_range(1), Some((0, 6))); // "Hello\n"
        assert_eq!(tracker.line_byte_range(2), Some((6, 12))); // "World\n"
    }

    #[test]
    fn test_without_line_storage() {
        let mut tracker = PositionTracker::without_line_storage();
        tracker.advance("Hello\nWorld\n");
        tracker.finish();

        // Position tracking still works
        assert_eq!(tracker.line_number(0), 1);
        assert_eq!(tracker.line_number(6), 2);

        // But line content is not available
        assert!(tracker.get_line(1).is_none());
    }

    #[test]
    fn test_unicode() {
        let mut tracker = PositionTracker::new();
        tracker.advance("héllo\nwörld\n");
        tracker.finish();

        // "héllo" is 6 bytes (h=1, é=2, l=1, l=1, o=1), newline at byte 6
        assert_eq!(tracker.line_number(0), 1);
        assert_eq!(tracker.line_number(7), 2); // Start of "wörld"
    }

    #[test]
    fn test_get_line_at() {
        let mut tracker = PositionTracker::new();
        tracker.advance("Hello\nWorld\n");
        tracker.finish();

        assert_eq!(tracker.get_line_at(0), Some("Hello"));
        assert_eq!(tracker.get_line_at(3), Some("Hello"));
        assert_eq!(tracker.get_line_at(6), Some("World"));
        assert_eq!(tracker.get_line_at(8), Some("World"));
    }
}
