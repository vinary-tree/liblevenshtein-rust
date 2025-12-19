//! Error types for grep operations on compressed and archived files.

use std::path::PathBuf;
use thiserror::Error;

/// Error type for grep operations.
#[derive(Debug, Error)]
pub enum GrepError {
    /// IO error during file operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid path specifier syntax.
    #[error("Invalid path specifier '{path}': {reason}")]
    InvalidPath {
        /// The problematic path string
        path: String,
        /// Description of the problem
        reason: String,
    },

    /// Unsupported compression format.
    #[error("Unsupported compression format: {0}")]
    UnsupportedCompression(String),

    /// Unsupported archive format.
    #[error("Unsupported archive format: {0}")]
    UnsupportedArchive(String),

    /// Archive entry not found.
    #[error("Archive entry not found: {entry} in {archive}")]
    EntryNotFound {
        /// The archive path
        archive: PathBuf,
        /// The entry path that was not found
        entry: String,
    },

    /// No entries matched the filter pattern.
    #[error("No entries in {archive} match filter '{filter}'")]
    NoMatchingEntries {
        /// The archive path
        archive: PathBuf,
        /// The filter pattern
        filter: String,
    },

    /// Invalid UTF-8 content in file or archive entry.
    #[error("Invalid UTF-8 in {file_path}: {message}")]
    InvalidUtf8 {
        /// Source file or archive entry path
        file_path: String,
        /// Error message
        message: String,
    },

    /// Decompression error.
    #[error("Decompression error for {file_path}: {message}")]
    Decompression {
        /// Source file path
        file_path: PathBuf,
        /// Error message
        message: String,
    },

    /// Archive format error.
    #[error("Archive error for {file_path}: {message}")]
    Archive {
        /// Archive file path
        file_path: PathBuf,
        /// Error message
        message: String,
    },

    /// Glob pattern error.
    #[error("Invalid glob pattern '{pattern}': {message}")]
    GlobPattern {
        /// The invalid pattern
        pattern: String,
        /// Error message
        message: String,
    },

    /// Password-protected archive.
    #[error("Archive {archive} is password-protected")]
    PasswordProtected {
        /// The archive path
        archive: PathBuf,
    },

    /// Nested archives are not supported.
    #[error("Nested archives are not supported: {inner} inside {outer}")]
    NestedArchive {
        /// Outer archive path
        outer: PathBuf,
        /// Inner archive entry path
        inner: String,
    },

    /// File appears to be binary (non-text).
    #[error("Binary file detected: {0}")]
    BinaryFile(PathBuf),

    /// Feature not enabled.
    #[error("Feature '{feature}' is required but not enabled. Rebuild with --features {feature}")]
    FeatureNotEnabled {
        /// The required feature name
        feature: String,
    },

    /// Document extraction error.
    #[error("Failed to extract text from {file_path}: {message}")]
    DocumentExtraction {
        /// Source file path
        file_path: PathBuf,
        /// Error message
        message: String,
    },

    /// Document contains no extractable text.
    #[error("Document contains no extractable text: {0}")]
    DocumentEmpty(PathBuf),

    /// Unsupported document format.
    #[error("Unsupported document format: {0}")]
    UnsupportedDocument(String),

    /// OCR error.
    #[error("OCR failed for {file_path}: {message}")]
    OcrError {
        /// Source file path
        file_path: PathBuf,
        /// Error message
        message: String,
    },

    /// OCR not available (feature not enabled or Tesseract not installed).
    #[error("OCR not available: {0}")]
    OcrNotAvailable(String),
}

/// Result type for grep operations.
pub type GrepResult<T> = Result<T, GrepError>;

impl GrepError {
    /// Create an InvalidPath error.
    pub fn invalid_path(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidPath {
            path: path.into(),
            reason: reason.into(),
        }
    }

    /// Create a Decompression error.
    pub fn decompression(file_path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::Decompression {
            file_path: file_path.into(),
            message: message.into(),
        }
    }

    /// Create an Archive error.
    pub fn archive(file_path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::Archive {
            file_path: file_path.into(),
            message: message.into(),
        }
    }

    /// Create an InvalidUtf8 error.
    pub fn invalid_utf8(file_path: impl Into<String>, err: std::str::Utf8Error) -> Self {
        Self::InvalidUtf8 {
            file_path: file_path.into(),
            message: err.to_string(),
        }
    }

    /// Create a GlobPattern error.
    pub fn glob_pattern(pattern: impl Into<String>, message: impl Into<String>) -> Self {
        Self::GlobPattern {
            pattern: pattern.into(),
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GrepError::invalid_path("foo:bar:baz", "too many colons");
        assert!(err.to_string().contains("foo:bar:baz"));
        assert!(err.to_string().contains("too many colons"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let grep_err: GrepError = io_err.into();
        assert!(matches!(grep_err, GrepError::Io(_)));
    }
}
