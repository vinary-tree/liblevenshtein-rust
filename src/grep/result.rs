//! Result types for grep operations.
//!
//! This module provides types for representing grep match results,
//! including source location information for archives.

use std::path::PathBuf;

/// Identifier for a source within grep results.
///
/// For plain files, only `file_path` is set.
/// For archive entries, both `file_path` (the archive) and `archive_entry` are set.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceId {
    /// Path to the file or archive on the filesystem.
    pub file_path: PathBuf,

    /// Path within the archive (if applicable).
    ///
    /// Uses forward slashes regardless of platform.
    pub archive_entry: Option<String>,
}

impl SourceId {
    /// Create a SourceId for a plain file.
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self {
            file_path: path.into(),
            archive_entry: None,
        }
    }

    /// Create a SourceId for an archive entry.
    pub fn archive_entry(archive: impl Into<PathBuf>, entry: impl Into<String>) -> Self {
        Self {
            file_path: archive.into(),
            archive_entry: Some(entry.into()),
        }
    }

    /// Check if this is an archive entry.
    pub fn is_archive_entry(&self) -> bool {
        self.archive_entry.is_some()
    }

    /// Get a display string for this source.
    ///
    /// Format: `path` for files, `archive:entry` for archive entries.
    pub fn display(&self) -> String {
        match &self.archive_entry {
            Some(entry) => format!("{}:{}", self.file_path.display(), entry),
            None => self.file_path.display().to_string(),
        }
    }
}

impl std::fmt::Display for SourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

/// Location of a match within a source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchLocation {
    /// Path to the file or archive on the filesystem.
    pub file_path: PathBuf,

    /// Path within the archive (if applicable).
    pub archive_entry: Option<String>,

    /// Line number (1-indexed).
    pub line_number: usize,

    /// Column offset (1-indexed, byte offset within line).
    pub column: usize,

    /// Byte offset from start of (decompressed) content.
    pub byte_offset: usize,
}

impl MatchLocation {
    /// Create a MatchLocation for a plain file.
    pub fn file(
        path: impl Into<PathBuf>,
        line_number: usize,
        column: usize,
        byte_offset: usize,
    ) -> Self {
        Self {
            file_path: path.into(),
            archive_entry: None,
            line_number,
            column,
            byte_offset,
        }
    }

    /// Create a MatchLocation for an archive entry.
    pub fn archive(
        archive: impl Into<PathBuf>,
        entry: impl Into<String>,
        line_number: usize,
        column: usize,
        byte_offset: usize,
    ) -> Self {
        Self {
            file_path: archive.into(),
            archive_entry: Some(entry.into()),
            line_number,
            column,
            byte_offset,
        }
    }

    /// Get a display string in the format used by grep tools.
    ///
    /// Format: `file:line:col` or `archive:entry:line:col`
    pub fn display(&self) -> String {
        match &self.archive_entry {
            Some(entry) => format!(
                "{}:{}:{}:{}",
                self.file_path.display(),
                entry,
                self.line_number,
                self.column
            ),
            None => format!(
                "{}:{}:{}",
                self.file_path.display(),
                self.line_number,
                self.column
            ),
        }
    }

    /// Get just the line:col part.
    pub fn position_display(&self) -> String {
        format!("{}:{}", self.line_number, self.column)
    }

    /// Get the source ID for this location.
    pub fn source_id(&self) -> SourceId {
        SourceId {
            file_path: self.file_path.clone(),
            archive_entry: self.archive_entry.clone(),
        }
    }
}

impl std::fmt::Display for MatchLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

/// A grep match result with full context.
#[derive(Debug, Clone)]
pub struct GrepMatchResult {
    /// Location of the match.
    pub location: MatchLocation,

    /// The matched text (what was actually found).
    pub matched_text: String,

    /// The entire line containing the match.
    pub line_text: String,

    /// Edit distance (for fuzzy matches, 0 for exact).
    pub distance: u8,

    /// Start position within the line (byte offset).
    pub match_start_in_line: usize,

    /// End position within the line (byte offset, exclusive).
    pub match_end_in_line: usize,
}

impl GrepMatchResult {
    /// Create a new grep match result.
    pub fn new(
        location: MatchLocation,
        matched_text: String,
        line_text: String,
        distance: u8,
    ) -> Self {
        // Find match position in line
        let match_start = line_text.find(&matched_text).unwrap_or(0);
        let match_end = match_start + matched_text.len();

        Self {
            location,
            matched_text,
            line_text,
            distance,
            match_start_in_line: match_start,
            match_end_in_line: match_end,
        }
    }

    /// Get the text before the match on the same line.
    pub fn prefix(&self) -> &str {
        &self.line_text[..self.match_start_in_line]
    }

    /// Get the text after the match on the same line.
    pub fn suffix(&self) -> &str {
        &self.line_text[self.match_end_in_line..]
    }

    /// Check if this is an exact match (distance = 0).
    pub fn is_exact(&self) -> bool {
        self.distance == 0
    }
}

impl std::fmt::Display for GrepMatchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.location, self.line_text.trim_end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_id_file() {
        let id = SourceId::file("path/to/file.txt");
        assert!(!id.is_archive_entry());
        assert_eq!(id.display(), "path/to/file.txt");
    }

    #[test]
    fn test_source_id_archive() {
        let id = SourceId::archive_entry("archive.tar.gz", "path/inside/file.txt");
        assert!(id.is_archive_entry());
        assert_eq!(id.display(), "archive.tar.gz:path/inside/file.txt");
    }

    #[test]
    fn test_match_location_file() {
        let loc = MatchLocation::file("file.txt", 10, 5, 100);
        assert_eq!(loc.display(), "file.txt:10:5");
        assert_eq!(loc.position_display(), "10:5");
    }

    #[test]
    fn test_match_location_archive() {
        let loc = MatchLocation::archive("archive.tar.gz", "dir/file.txt", 10, 5, 100);
        assert_eq!(loc.display(), "archive.tar.gz:dir/file.txt:10:5");
    }

    #[test]
    fn test_grep_match_result() {
        let loc = MatchLocation::file("file.txt", 1, 7, 6);
        let result = GrepMatchResult::new(
            loc,
            "World".to_string(),
            "Hello World!".to_string(),
            0,
        );

        assert_eq!(result.prefix(), "Hello ");
        assert_eq!(result.suffix(), "!");
        assert!(result.is_exact());
    }

    #[test]
    fn test_grep_match_result_fuzzy() {
        let loc = MatchLocation::file("file.txt", 1, 7, 6);
        let result = GrepMatchResult {
            location: loc,
            matched_text: "Wold".to_string(),
            line_text: "Hello Wold!".to_string(),
            distance: 1,
            match_start_in_line: 6,
            match_end_in_line: 10,
        };

        assert!(!result.is_exact());
        assert_eq!(result.distance, 1);
    }
}
