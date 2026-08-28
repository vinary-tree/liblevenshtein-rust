//! File loader for `.llev` rule files with include resolution.
//!
//! This module handles loading `.llev` files from disk, resolving `@include`
//! directives, and detecting circular includes.
//!
//! # Include Resolution
//!
//! When an `@include "path"` directive is encountered:
//!
//! 1. If the path is absolute, it is used directly
//! 2. If relative, it is resolved relative to the file containing the `@include`
//! 3. If not found, the loader searches through configured include paths
//! 4. Circular includes are detected and reported as errors
//!
//! # Usage
//!
//! ```rust,no_run
//! use liblevenshtein::phonetic::llev::{Loader, LoaderConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Default configuration
//! let loader = Loader::new();
//! let file = loader.load("rules.llev")?;
//! assert!(!file.rules.is_empty());
//!
//! // Custom configuration
//! let config = LoaderConfig::new()
//!     .with_include_path("./rules")
//!     .with_include_path("/usr/share/llev")
//!     .with_max_include_depth(20);
//! let loader = Loader::with_config(config);
//! let file = loader.load("rules.llev")?;
//! assert!(!file.rules.is_empty());
//! # Ok(())
//! # }
//! ```

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::ast::LLevFile;
use super::error::{LLevError, LLevErrorKind, LLevResult};
use super::parser::Parser;

/// Default maximum include depth.
pub const DEFAULT_MAX_INCLUDE_DEPTH: usize = 10;

/// Configuration for the `.llev` file loader.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Paths to search for included files.
    pub include_paths: Vec<PathBuf>,

    /// Maximum depth of nested includes.
    pub max_include_depth: usize,

    /// Whether to allow missing includes (skip instead of error).
    pub allow_missing_includes: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            include_paths: Vec::new(),
            max_include_depth: DEFAULT_MAX_INCLUDE_DEPTH,
            allow_missing_includes: false,
        }
    }
}

impl LoaderConfig {
    /// Create a new loader configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an include search path.
    pub fn with_include_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.include_paths.push(path.into());
        self
    }

    /// Set the maximum include depth.
    pub fn with_max_include_depth(mut self, depth: usize) -> Self {
        self.max_include_depth = depth;
        self
    }

    /// Set whether to allow missing includes.
    pub fn with_allow_missing_includes(mut self, allow: bool) -> Self {
        self.allow_missing_includes = allow;
        self
    }
}

/// Loader for `.llev` files with include resolution.
///
/// The loader handles reading files from disk, parsing them, and resolving
/// `@include` directives while detecting circular includes.
#[derive(Debug, Clone)]
pub struct Loader {
    config: LoaderConfig,
}

impl Default for Loader {
    fn default() -> Self {
        Self::new()
    }
}

impl Loader {
    /// Create a new loader with default configuration.
    pub fn new() -> Self {
        Self {
            config: LoaderConfig::default(),
        }
    }

    /// Create a new loader with the given configuration.
    pub fn with_config(config: LoaderConfig) -> Self {
        Self { config }
    }

    /// Load a `.llev` file and resolve all includes.
    ///
    /// This is the main entry point for loading rule files. It:
    /// 1. Reads the file from disk
    /// 2. Parses the content
    /// 3. Recursively resolves all `@include` directives
    /// 4. Merges included rules into the main file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The file contains syntax errors
    /// - A circular include is detected
    /// - Include depth is exceeded
    /// - An included file cannot be found
    pub fn load<P: AsRef<Path>>(&self, path: P) -> LLevResult<LLevFile> {
        let path = path.as_ref();
        let canonical = self.canonicalize_path(path)?;
        let mut visited = HashSet::new();
        self.load_recursive(&canonical, &mut visited, 0)
    }

    /// Load a `.llev` file from a string with a given base path.
    ///
    /// The base path is used for resolving relative `@include` paths.
    pub fn load_str(&self, content: &str, base_path: Option<&Path>) -> LLevResult<LLevFile> {
        let mut parser = Parser::new(content);
        let mut file = parser.parse_file()?;

        // Set source file if provided
        if let Some(path) = base_path {
            file.source_file = Some(path.to_path_buf());
        }

        // Resolve includes
        let mut visited = HashSet::new();
        if let Some(path) = base_path {
            let canonical = self.canonicalize_path(path)?;
            visited.insert(canonical);
        }
        self.resolve_includes(&mut file, &mut visited, 0)?;

        Ok(file)
    }

    /// Internal recursive loader.
    fn load_recursive(
        &self,
        path: &Path,
        visited: &mut HashSet<PathBuf>,
        depth: usize,
    ) -> LLevResult<LLevFile> {
        // Check include depth
        if depth > self.config.max_include_depth {
            return Err(LLevError::include_depth_exceeded(
                self.config.max_include_depth,
                path.to_path_buf(),
            ));
        }

        // Check for circular include
        if visited.contains(path) {
            return Err(LLevError::circular_include(path.to_path_buf()));
        }
        visited.insert(path.to_path_buf());

        // Read the file
        let content = self.read_file(path)?;

        // Parse the content
        let mut parser = Parser::new(&content);
        let mut file = parser
            .parse_file()
            .map_err(|e| e.in_file(path.to_path_buf()))?;
        file.source_file = Some(path.to_path_buf());

        // Resolve includes
        self.resolve_includes(&mut file, visited, depth)?;

        // Remove this file from visited (allow re-including in different branches)
        // Actually, we keep it to prevent circular includes within the same load tree
        // visited.remove(path);

        Ok(file)
    }

    /// Resolve all `@include` directives in a file.
    fn resolve_includes(
        &self,
        file: &mut LLevFile,
        visited: &mut HashSet<PathBuf>,
        depth: usize,
    ) -> LLevResult<()> {
        // Process includes
        let includes = std::mem::take(&mut file.includes);

        for include in includes {
            // Resolve the include path
            let include_path =
                self.resolve_include_path(&include.path, file.source_file.as_deref())?;

            match include_path {
                Some(resolved_path) => {
                    // Load the included file
                    let included = self.load_recursive(&resolved_path, visited, depth + 1)?;

                    // Merge the included file
                    file.merge(included);

                    // Track the resolved path
                    file.resolved_includes.push(resolved_path);
                }
                None if self.config.allow_missing_includes => {
                    // Skip missing includes if configured to allow
                    continue;
                }
                None => {
                    // Report include not found
                    let search_paths = self.get_search_paths(file.source_file.as_deref());
                    return Err(LLevError::include_not_found(&include.path, search_paths)
                        .at_position(include.position)
                        .in_file(file.source_file.clone().unwrap_or_default()));
                }
            }
        }

        Ok(())
    }

    /// Resolve an include path to an absolute path.
    ///
    /// Returns `None` if the file cannot be found.
    fn resolve_include_path(
        &self,
        path: &str,
        source_file: Option<&Path>,
    ) -> LLevResult<Option<PathBuf>> {
        let include_path = Path::new(path);

        // If absolute, use directly
        if include_path.is_absolute() {
            if include_path.exists() {
                return Ok(Some(self.canonicalize_path(include_path)?));
            }
            return Ok(None);
        }

        // Try relative to source file first
        if let Some(source) = source_file {
            if let Some(parent) = source.parent() {
                let relative = parent.join(include_path);
                if relative.exists() {
                    return Ok(Some(self.canonicalize_path(&relative)?));
                }
            }
        }

        // Search include paths
        for search_path in &self.config.include_paths {
            let candidate = search_path.join(include_path);
            if candidate.exists() {
                return Ok(Some(self.canonicalize_path(&candidate)?));
            }
        }

        Ok(None)
    }

    /// Get all search paths for error reporting.
    fn get_search_paths(&self, source_file: Option<&Path>) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // Add source file directory
        if let Some(source) = source_file {
            if let Some(parent) = source.parent() {
                paths.push(parent.to_path_buf());
            }
        }

        // Add configured include paths
        paths.extend(self.config.include_paths.iter().cloned());

        paths
    }

    /// Read a file's contents.
    fn read_file(&self, path: &Path) -> LLevResult<String> {
        fs::read_to_string(path).map_err(|e| match e.kind() {
            io::ErrorKind::NotFound => LLevError::file_not_found(path.display().to_string()),
            io::ErrorKind::PermissionDenied => {
                LLevError::new(LLevErrorKind::PermissionDenied(path.display().to_string()))
            }
            _ => LLevError::io_error(format!("{}: {}", path.display(), e)),
        })
    }

    /// Canonicalize a path for consistent comparison.
    fn canonicalize_path(&self, path: &Path) -> LLevResult<PathBuf> {
        path.canonicalize().map_err(|e| {
            LLevError::io_error(format!(
                "failed to canonicalize path {}: {}",
                path.display(),
                e
            ))
        })
    }
}

/// Load a `.llev` file from disk.
///
/// This is a convenience function that uses a default loader configuration.
///
/// # Example
///
/// ```rust,no_run
/// use liblevenshtein::phonetic::llev::load_file;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let file = load_file("rules.llev")?;
/// assert!(!file.rules.is_empty());
/// # Ok(())
/// # }
/// ```
pub fn load_file<P: AsRef<Path>>(path: P) -> LLevResult<LLevFile> {
    Loader::new().load(path)
}

/// Load a `.llev` file with custom include paths.
///
/// # Example
///
/// ```rust,no_run
/// use liblevenshtein::phonetic::llev::load_file_with_includes;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let file = load_file_with_includes("rules.llev", &["./rules", "/usr/share/llev"])?;
/// assert!(!file.rules.is_empty());
/// # Ok(())
/// # }
/// ```
pub fn load_file_with_includes<P: AsRef<Path>, I: AsRef<Path>>(
    path: P,
    include_paths: &[I],
) -> LLevResult<LLevFile> {
    let mut config = LoaderConfig::new();
    for p in include_paths {
        config = config.with_include_path(p.as_ref());
    }
    Loader::with_config(config).load(path)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let path = dir.path().join(name);
        let mut file = File::create(&path).expect("Failed to create test file");
        file.write_all(content.as_bytes())
            .expect("Failed to write test file");
        path
    }

    #[test]
    fn test_load_simple_file() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = create_test_file(&dir, "simple.llev", "ph -> f");

        let loader = Loader::new();
        let file = loader.load(&path).expect("Failed to load file");

        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_load_file_with_metadata() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = r#"
@name "Test Rules"
@version "1.0"

ph -> f
"#;
        let path = create_test_file(&dir, "meta.llev", content);

        let loader = Loader::new();
        let file = loader.load(&path).expect("Failed to load file");

        assert_eq!(file.metadata.name, Some("Test Rules".to_string()));
        assert_eq!(file.metadata.version, Some("1.0".to_string()));
        assert_eq!(file.rules.len(), 1);
    }

    #[test]
    fn test_load_file_with_include() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // Create included file
        create_test_file(&dir, "included.llev", "gh -> f");

        // Create main file with include
        let main_content = r#"
ph -> f
@include "included.llev"
"#;
        let main_path = create_test_file(&dir, "main.llev", main_content);

        let loader = Loader::new();
        let file = loader.load(&main_path).expect("Failed to load file");

        // Should have rules from both files
        assert_eq!(file.rules.len(), 2);
    }

    #[test]
    fn test_circular_include_detection() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // Create files that include each other
        let a_content = r#"
@include "b.llev"
"#;
        let b_content = r#"
@include "a.llev"
"#;
        create_test_file(&dir, "a.llev", a_content);
        create_test_file(&dir, "b.llev", b_content);

        let a_path = dir.path().join("a.llev");
        let loader = Loader::new();
        let result = loader.load(&a_path);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind, LLevErrorKind::CircularInclude(_)));
        }
    }

    #[test]
    fn test_include_depth_exceeded() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // Create a chain of includes
        for i in 0..15 {
            let content = if i < 14 {
                format!("@include \"{}.llev\"", i + 1)
            } else {
                "ph -> f".to_string()
            };
            create_test_file(&dir, &format!("{}.llev", i), &content);
        }

        let start_path = dir.path().join("0.llev");
        let config = LoaderConfig::new().with_max_include_depth(5);
        let loader = Loader::with_config(config);
        let result = loader.load(&start_path);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind, LLevErrorKind::IncludeDepthExceeded { .. }));
        }
    }

    #[test]
    fn test_include_not_found() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = r#"
@include "nonexistent.llev"
"#;
        let path = create_test_file(&dir, "main.llev", content);

        let loader = Loader::new();
        let result = loader.load(&path);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(matches!(e.kind, LLevErrorKind::IncludeNotFound { .. }));
        }
    }

    #[test]
    fn test_allow_missing_includes() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = r#"
ph -> f
@include "nonexistent.llev"
gh -> g
"#;
        let path = create_test_file(&dir, "main.llev", content);

        let config = LoaderConfig::new().with_allow_missing_includes(true);
        let loader = Loader::with_config(config);
        let file = loader
            .load(&path)
            .expect("Should succeed with missing include");

        // Should have rules from main file only
        assert_eq!(file.rules.len(), 2);
    }

    #[test]
    fn test_include_paths() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let include_dir = TempDir::new().expect("Failed to create include dir");

        // Create included file in separate directory
        create_test_file(&include_dir, "external.llev", "gh -> f");

        // Create main file referencing external include
        let main_content = r#"
ph -> f
@include "external.llev"
"#;
        let main_path = create_test_file(&dir, "main.llev", main_content);

        // Load with include path
        let config = LoaderConfig::new().with_include_path(include_dir.path());
        let loader = Loader::with_config(config);
        let file = loader
            .load(&main_path)
            .expect("Failed to load with include path");

        assert_eq!(file.rules.len(), 2);
    }

    #[test]
    fn test_nested_includes() {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // Create nested include chain: main -> a -> b
        create_test_file(&dir, "b.llev", "th -> t");
        create_test_file(&dir, "a.llev", "gh -> g\n@include \"b.llev\"");
        let main_content = r#"
ph -> f
@include "a.llev"
"#;
        let main_path = create_test_file(&dir, "main.llev", main_content);

        let loader = Loader::new();
        let file = loader
            .load(&main_path)
            .expect("Failed to load nested includes");

        // Should have rules from all three files
        assert_eq!(file.rules.len(), 3);
    }

    #[test]
    fn test_file_not_found() {
        let loader = Loader::new();
        let result = loader.load("/nonexistent/path/to/file.llev");

        assert!(result.is_err());
    }
}
