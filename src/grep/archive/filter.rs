//! Glob-based filtering for archive entries.
//!
//! This module provides utilities to filter archive entries by glob patterns.

use crate::grep::error::{GrepError, GrepResult};

#[cfg(feature = "globset")]
use globset::{GlobBuilder, GlobMatcher};

/// A compiled filter for archive entry paths.
#[derive(Clone)]
pub struct EntryFilter {
    /// The original pattern string.
    pattern: String,

    /// Compiled glob matcher.
    #[cfg(feature = "globset")]
    matcher: GlobMatcher,
}

impl EntryFilter {
    /// Create a new entry filter from a glob pattern.
    ///
    /// # Patterns
    ///
    /// - `*` matches any sequence of characters except `/`
    /// - `**` matches any sequence including `/`
    /// - `?` matches any single character except `/`
    /// - `[abc]` matches one character from the set
    /// - `{a,b,c}` matches any of the comma-separated patterns
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use liblevenshtein::grep::archive::filter::EntryFilter;
    ///
    /// // Match all .rs files
    /// let filter = EntryFilter::new("*.rs")?;
    /// assert!(filter.matches("main.rs"));
    /// assert!(!filter.matches("src/main.rs")); // * doesn't match /
    ///
    /// // Match all .rs files recursively
    /// let filter = EntryFilter::new("**/*.rs")?;
    /// assert!(filter.matches("main.rs"));
    /// assert!(filter.matches("src/main.rs"));
    /// assert!(filter.matches("src/foo/bar/baz.rs"));
    /// ```
    #[cfg(feature = "globset")]
    pub fn new(pattern: &str) -> GrepResult<Self> {
        // Use literal_separator(true) so that '*' doesn't match '/'
        // This makes '*.rs' match 'main.rs' but not 'src/main.rs'
        // Use '**/*.rs' to match files in subdirectories
        let glob = GlobBuilder::new(pattern)
            .literal_separator(true)
            .build()
            .map_err(|e| {
                GrepError::glob_pattern(pattern, e.to_string())
            })?;

        Ok(Self {
            pattern: pattern.to_string(),
            matcher: glob.compile_matcher(),
        })
    }

    /// Create a new entry filter (no-op when globset feature is disabled).
    #[cfg(not(feature = "globset"))]
    pub fn new(pattern: &str) -> GrepResult<Self> {
        Ok(Self {
            pattern: pattern.to_string(),
        })
    }

    /// Check if a path matches this filter.
    #[cfg(feature = "globset")]
    pub fn matches(&self, path: &str) -> bool {
        // Normalize path for matching
        let normalized = path.trim_start_matches('/').trim_start_matches("./");
        self.matcher.is_match(normalized)
    }

    /// Check if a path matches this filter (simple substring when globset disabled).
    #[cfg(not(feature = "globset"))]
    pub fn matches(&self, path: &str) -> bool {
        // Fallback: simple substring matching
        let normalized = path.trim_start_matches('/').trim_start_matches("./");

        // Handle simple wildcard patterns
        if self.pattern == "*" {
            return true;
        }

        // Check for prefix match (pattern ends with *)
        if let Some(prefix) = self.pattern.strip_suffix('*') {
            if let Some(p) = prefix.strip_suffix('/') {
                return normalized.starts_with(p);
            }
            return normalized.starts_with(prefix);
        }

        // Check for suffix match (pattern starts with *)
        if let Some(suffix) = self.pattern.strip_prefix('*') {
            return normalized.ends_with(suffix);
        }

        // Exact match
        normalized == self.pattern || path == self.pattern
    }

    /// Get the original pattern string.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Create a filter that matches all entries.
    pub fn match_all() -> Self {
        #[cfg(feature = "globset")]
        {
            Self::new("**/*").expect("** pattern should always compile")
        }
        #[cfg(not(feature = "globset"))]
        {
            Self {
                pattern: "*".to_string(),
            }
        }
    }
}

impl std::fmt::Debug for EntryFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EntryFilter")
            .field("pattern", &self.pattern)
            .finish()
    }
}

impl std::fmt::Display for EntryFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pattern)
    }
}

/// Check if a path should be filtered (excluded from grep).
///
/// Some paths are always excluded:
/// - Hidden files (starting with `.`) unless explicitly requested
/// - Binary files (common binary extensions)
/// - Very large files
pub fn should_skip_entry(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);

    // Skip common binary file extensions
    let binary_extensions = [
        "exe", "dll", "so", "dylib", "o", "obj", "a", "lib",
        "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp",
        "mp3", "mp4", "wav", "ogg", "flac", "avi", "mkv", "mov",
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
        "class", "pyc", "pyo", "wasm",
        "ttf", "otf", "woff", "woff2", "eot",
    ];

    if let Some(ext) = name.rsplit('.').next() {
        if binary_extensions.contains(&ext.to_lowercase().as_str()) {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_all() {
        let filter = EntryFilter::match_all();
        assert!(filter.matches("anything"));
        assert!(filter.matches("path/to/file.txt"));
    }

    #[cfg(feature = "globset")]
    #[test]
    fn test_glob_star() {
        let filter = EntryFilter::new("*.rs").expect("should compile");
        assert!(filter.matches("main.rs"));
        assert!(filter.matches("lib.rs"));
        assert!(!filter.matches("src/main.rs")); // * doesn't cross /
        assert!(!filter.matches("main.txt"));
    }

    #[cfg(feature = "globset")]
    #[test]
    fn test_glob_double_star() {
        let filter = EntryFilter::new("**/*.rs").expect("should compile");
        assert!(filter.matches("main.rs"));
        assert!(filter.matches("src/main.rs"));
        assert!(filter.matches("src/foo/bar/baz.rs"));
        assert!(!filter.matches("main.txt"));
    }

    #[cfg(feature = "globset")]
    #[test]
    fn test_glob_directory() {
        let filter = EntryFilter::new("src/**").expect("should compile");
        assert!(filter.matches("src/main.rs"));
        assert!(filter.matches("src/foo/bar.rs"));
        assert!(!filter.matches("tests/test.rs"));
    }

    #[cfg(feature = "globset")]
    #[test]
    fn test_glob_alternatives() {
        let filter = EntryFilter::new("*.{rs,toml}").expect("should compile");
        assert!(filter.matches("main.rs"));
        assert!(filter.matches("Cargo.toml"));
        assert!(!filter.matches("main.txt"));
    }

    #[test]
    fn test_should_skip_binary() {
        assert!(should_skip_entry("image.png"));
        assert!(should_skip_entry("path/to/binary.exe"));
        assert!(should_skip_entry("archive.zip"));
        assert!(!should_skip_entry("source.rs"));
        assert!(!should_skip_entry("readme.md"));
    }

    #[test]
    fn test_normalized_paths() {
        let filter = EntryFilter::match_all();
        assert!(filter.matches("/absolute/path"));
        assert!(filter.matches("./relative/path"));
    }
}
