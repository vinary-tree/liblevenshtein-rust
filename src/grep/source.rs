//! Source path parsing and handling for grep operations.
//!
//! This module provides the `GrepPath` type for parsing path specifiers
//! with optional archive entry filters.
//!
//! # Path Syntax
//!
//! The path syntax supports:
//! - Plain files: `file.txt`, `data.log`
//! - Compressed files: `file.txt.gz`, `data.log.zst`
//! - Archives: `archive.tar`, `archive.zip`
//! - Compressed archives: `archive.tar.gz`, `archive.tar.zst`
//! - Archive with filter: `archive.tar.gz:*.log`, `archive.zip:src/*.rs`
//!
//! The colon (`:`) separates the filesystem path from the archive filter.
//! Use `\\:` to escape a literal colon in the path.

use std::path::{Path, PathBuf};

#[cfg(any(feature = "tar", feature = "zip"))]
use crate::grep::archive::ArchiveFormat;
use crate::grep::compression::CompressionFormat;
use crate::grep::error::{GrepError, GrepResult};

/// Stub archive format when archive features are not enabled.
/// Used only for type compatibility.
#[cfg(not(any(feature = "tar", feature = "zip")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFormat {
    /// Not an archive (this is the only variant when archives are disabled).
    None,
}

#[cfg(not(any(feature = "tar", feature = "zip")))]
impl ArchiveFormat {
    /// Always returns `None` when archive features are disabled.
    pub fn from_path(_path: &Path) -> Self {
        Self::None
    }
}

/// A parsed grep path specifier.
///
/// Represents a path to a file (possibly compressed) or archive,
/// with an optional filter for archive entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepPath {
    /// Path to the file or archive on the filesystem.
    pub filesystem_path: PathBuf,

    /// Optional glob filter for archive entries.
    ///
    /// If `None`, all entries in an archive are searched.
    /// For non-archives, this should always be `None`.
    pub archive_filter: Option<String>,

    /// Detected compression format.
    pub compression: CompressionFormat,

    /// Detected archive format.
    pub archive: ArchiveFormat,
}

impl GrepPath {
    /// Parse a path specifier string.
    ///
    /// # Syntax
    ///
    /// - `path` - Plain file or auto-detected compressed/archive
    /// - `path:filter` - Archive with entry filter (glob pattern)
    ///
    /// # Escaping
    ///
    /// - Use `\\:` to include a literal colon in the path
    /// - On Windows, drive letters like `C:` are handled automatically
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::grep::GrepPath;
    ///
    /// // Plain file
    /// let path = GrepPath::parse("file.txt").unwrap();
    /// assert_eq!(path.filesystem_path.to_str(), Some("file.txt"));
    /// assert!(path.archive_filter.is_none());
    ///
    /// // Compressed file
    /// let path = GrepPath::parse("file.txt.gz").unwrap();
    /// assert_eq!(path.filesystem_path.to_str(), Some("file.txt.gz"));
    ///
    /// // Archive with filter
    /// let path = GrepPath::parse("logs.tar.gz:*.log").unwrap();
    /// assert_eq!(path.archive_filter, Some("*.log".to_string()));
    /// ```
    pub fn parse(input: &str) -> GrepResult<Self> {
        if input.is_empty() {
            return Err(GrepError::invalid_path(input, "path cannot be empty"));
        }

        let (path_str, filter) = Self::split_path_filter(input)?;

        if path_str.is_empty() {
            return Err(GrepError::invalid_path(
                input,
                "filesystem path cannot be empty",
            ));
        }

        let path = PathBuf::from(path_str);
        let archive = ArchiveFormat::from_path(&path);
        #[cfg(any(feature = "tar", feature = "zip"))]
        let compression = match &archive {
            ArchiveFormat::Tar { compression } => *compression,
            ArchiveFormat::Zip => CompressionFormat::None, // Zip handles compression internally
            ArchiveFormat::None => CompressionFormat::from_extension(&path),
        };
        #[cfg(not(any(feature = "tar", feature = "zip")))]
        let compression = CompressionFormat::from_extension(&path);

        // Validate: filters only make sense for archives
        if filter.is_some() && matches!(archive, ArchiveFormat::None) {
            return Err(GrepError::invalid_path(
                input,
                "archive filter specified but file is not an archive",
            ));
        }

        Ok(Self {
            filesystem_path: path,
            archive_filter: filter,
            compression,
            archive,
        })
    }

    /// Create a GrepPath from a filesystem path without parsing.
    ///
    /// This bypasses the filter syntax parsing and just wraps the path.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let archive = ArchiveFormat::from_path(&path);
        #[cfg(any(feature = "tar", feature = "zip"))]
        let compression = match &archive {
            ArchiveFormat::Tar { compression } => *compression,
            ArchiveFormat::Zip => CompressionFormat::None,
            ArchiveFormat::None => CompressionFormat::from_extension(&path),
        };
        #[cfg(not(any(feature = "tar", feature = "zip")))]
        let compression = CompressionFormat::from_extension(&path);

        Self {
            filesystem_path: path,
            archive_filter: None,
            compression,
            archive,
        }
    }

    /// Check if this path refers to an archive.
    pub fn is_archive(&self) -> bool {
        !matches!(self.archive, ArchiveFormat::None)
    }

    /// Check if this path refers to a compressed file.
    pub fn is_compressed(&self) -> bool {
        !matches!(self.compression, CompressionFormat::None)
    }

    /// Check if this is a plain (uncompressed, non-archive) file.
    pub fn is_plain(&self) -> bool {
        !self.is_archive() && !self.is_compressed()
    }

    /// Split input into filesystem path and optional filter.
    fn split_path_filter(input: &str) -> GrepResult<(String, Option<String>)> {
        let bytes = input.as_bytes();
        let len = bytes.len();

        // Handle Windows drive letters (e.g., "C:\path")
        let skip = if len >= 2
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (len == 2 || bytes[2] == b'\\' || bytes[2] == b'/')
        {
            2 // Skip drive letter
        } else {
            0
        };

        // Find first unescaped colon after any drive letter
        let mut i = skip;
        while i < len {
            if bytes[i] == b':' {
                // Check if escaped
                if i > 0 && bytes[i - 1] == b'\\' {
                    i += 1;
                    continue;
                }
                // Found unescaped colon - split here
                let path = Self::unescape_path(&input[..i]);
                let filter = &input[i + 1..];
                if filter.is_empty() {
                    return Err(GrepError::invalid_path(
                        input,
                        "archive filter cannot be empty (remove trailing colon)",
                    ));
                }
                return Ok((path, Some(filter.to_string())));
            }
            i += 1;
        }

        // No filter separator found
        Ok((Self::unescape_path(input), None))
    }

    /// Remove escape sequences from path.
    fn unescape_path(path: &str) -> String {
        path.replace("\\:", ":")
    }
}

impl std::fmt::Display for GrepPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.filesystem_path.display())?;
        if let Some(filter) = &self.archive_filter {
            write!(f, ":{}", filter)?;
        }
        Ok(())
    }
}

impl From<PathBuf> for GrepPath {
    fn from(path: PathBuf) -> Self {
        Self::from_path(path)
    }
}

impl From<&Path> for GrepPath {
    fn from(path: &Path) -> Self {
        Self::from_path(path)
    }
}

impl AsRef<Path> for GrepPath {
    fn as_ref(&self) -> &Path {
        &self.filesystem_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_plain_file() {
        let path = GrepPath::parse("file.txt").expect("should parse");
        assert_eq!(path.filesystem_path, PathBuf::from("file.txt"));
        assert!(path.archive_filter.is_none());
        assert_eq!(path.compression, CompressionFormat::None);
        assert!(matches!(path.archive, ArchiveFormat::None));
        assert!(path.is_plain());
    }

    #[test]
    fn test_parse_gzip_file() {
        let path = GrepPath::parse("file.txt.gz").expect("should parse");
        assert_eq!(path.filesystem_path, PathBuf::from("file.txt.gz"));
        assert!(path.archive_filter.is_none());
        assert_eq!(path.compression, CompressionFormat::Gzip);
        assert!(matches!(path.archive, ArchiveFormat::None));
        assert!(path.is_compressed());
        assert!(!path.is_archive());
    }

    #[cfg(any(feature = "tar", feature = "zip"))]
    #[test]
    fn test_parse_tar_archive() {
        let path = GrepPath::parse("archive.tar").expect("should parse");
        assert_eq!(path.filesystem_path, PathBuf::from("archive.tar"));
        assert!(path.archive_filter.is_none());
        assert_eq!(path.compression, CompressionFormat::None);
        assert!(matches!(
            path.archive,
            ArchiveFormat::Tar {
                compression: CompressionFormat::None
            }
        ));
        assert!(path.is_archive());
    }

    #[cfg(any(feature = "tar", feature = "zip"))]
    #[test]
    fn test_parse_tar_gz_archive() {
        let path = GrepPath::parse("archive.tar.gz").expect("should parse");
        assert_eq!(path.filesystem_path, PathBuf::from("archive.tar.gz"));
        assert!(path.archive_filter.is_none());
        assert_eq!(path.compression, CompressionFormat::Gzip);
        assert!(matches!(
            path.archive,
            ArchiveFormat::Tar {
                compression: CompressionFormat::Gzip
            }
        ));
    }

    #[cfg(any(feature = "tar", feature = "zip"))]
    #[test]
    fn test_parse_archive_with_filter() {
        let path = GrepPath::parse("logs.tar.gz:*.log").expect("should parse");
        assert_eq!(path.filesystem_path, PathBuf::from("logs.tar.gz"));
        assert_eq!(path.archive_filter, Some("*.log".to_string()));
    }

    #[cfg(any(feature = "tar", feature = "zip"))]
    #[test]
    fn test_parse_archive_with_path_filter() {
        let path = GrepPath::parse("src.zip:src/**/*.rs").expect("should parse");
        assert_eq!(path.filesystem_path, PathBuf::from("src.zip"));
        assert_eq!(path.archive_filter, Some("src/**/*.rs".to_string()));
    }

    #[test]
    fn test_parse_escaped_colon() {
        let path = GrepPath::parse("file\\:name.txt").expect("should parse");
        assert_eq!(path.filesystem_path, PathBuf::from("file:name.txt"));
        assert!(path.archive_filter.is_none());
    }

    #[test]
    fn test_parse_empty_fails() {
        let result = GrepPath::parse("");
        assert!(result.is_err());
    }

    #[cfg(any(feature = "tar", feature = "zip"))]
    #[test]
    fn test_parse_empty_filter_fails() {
        let result = GrepPath::parse("archive.tar:");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_filter_on_non_archive_fails() {
        let result = GrepPath::parse("file.txt:*.log");
        assert!(result.is_err());
    }

    #[cfg(any(feature = "tar", feature = "zip"))]
    #[test]
    fn test_display_with_archive() {
        let path = GrepPath::parse("logs.tar.gz:*.log").expect("should parse");
        assert_eq!(path.to_string(), "logs.tar.gz:*.log");
    }

    #[test]
    fn test_display() {
        let path = GrepPath::parse("file.txt").expect("should parse");
        assert_eq!(path.to_string(), "file.txt");
    }

    #[test]
    fn test_from_pathbuf() {
        let path: GrepPath = PathBuf::from("file.txt.gz").into();
        assert_eq!(path.compression, CompressionFormat::Gzip);
    }

    #[cfg(windows)]
    #[test]
    fn test_parse_windows_drive() {
        let path = GrepPath::parse("C:\\logs\\file.txt").expect("should parse");
        assert_eq!(path.filesystem_path, PathBuf::from("C:\\logs\\file.txt"));
        assert!(path.archive_filter.is_none());

        let path = GrepPath::parse("C:\\archive.tar:*.log").expect("should parse");
        assert_eq!(path.filesystem_path, PathBuf::from("C:\\archive.tar"));
        assert_eq!(path.archive_filter, Some("*.log".to_string()));
    }
}
