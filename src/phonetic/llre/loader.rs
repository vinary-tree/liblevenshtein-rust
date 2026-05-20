//! File loader for `.llre` files with import resolution.
//!
//! This module handles loading .llre files from disk and resolving @import
//! directives by loading the referenced .llev files.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::phonetic::llev::{self, LLevFile};

use super::ast::{ImportDirective, LLreFile, ResolvedImport, SymbolTable};
use super::error::{LLreError, LLreErrorKind, LLreResult};
use super::parser;

/// Configuration for the .llre file loader.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Search paths for @import directives
    pub search_paths: Vec<PathBuf>,

    /// Maximum import depth (to detect circular imports)
    pub max_import_depth: usize,

    /// Whether to resolve imports (false = parse only)
    pub resolve_imports: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            search_paths: vec![PathBuf::from(".")],
            max_import_depth: 10,
            resolve_imports: true,
        }
    }
}

impl LoaderConfig {
    /// Create a new loader config with custom search paths.
    pub fn with_search_paths(search_paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths,
            ..Default::default()
        }
    }

    /// Add a search path.
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        self.search_paths.push(path.into());
    }
}

/// Loader for .llre files.
pub struct Loader {
    config: LoaderConfig,
    /// Tracks files being loaded to detect circular imports
    loading_stack: HashSet<PathBuf>,
}

impl Loader {
    /// Create a new loader with default configuration.
    pub fn new() -> Self {
        Self {
            config: LoaderConfig::default(),
            loading_stack: HashSet::new(),
        }
    }

    /// Create a loader with custom configuration.
    pub fn with_config(config: LoaderConfig) -> Self {
        Self {
            config,
            loading_stack: HashSet::new(),
        }
    }

    /// Load a .llre file from the given path.
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> LLreResult<LLreFile> {
        let path = path.as_ref();
        let canonical = self.canonicalize_path(path)?;

        // Check for circular imports
        if self.loading_stack.contains(&canonical) {
            return Err(LLreError::new(LLreErrorKind::CircularImport(canonical)));
        }

        // Read the file
        let content = std::fs::read_to_string(path).map_err(|e| {
            LLreError::with_file(
                match e.kind() {
                    std::io::ErrorKind::NotFound => {
                        LLreErrorKind::FileNotFound(path.display().to_string())
                    }
                    std::io::ErrorKind::PermissionDenied => {
                        LLreErrorKind::PermissionDenied(path.display().to_string())
                    }
                    _ => LLreErrorKind::IoError(e.to_string()),
                },
                path,
            )
        })?;

        // Parse the file
        let mut file = parser::parse_str(&content)?;
        file.source_file = Some(canonical.clone());

        // Resolve imports if configured
        if self.config.resolve_imports && !file.imports.is_empty() {
            self.loading_stack.insert(canonical.clone());
            self.resolve_imports(&mut file, path.parent())?;
            self.loading_stack.remove(&canonical);
        }

        Ok(file)
    }

    /// Load a .llre file from a string (no import resolution).
    pub fn load_str(&self, content: &str) -> LLreResult<LLreFile> {
        parser::parse_str(content)
    }

    /// Resolve @import directives in the file.
    fn resolve_imports(&mut self, file: &mut LLreFile, base_dir: Option<&Path>) -> LLreResult<()> {
        let mut resolved_imports = Vec::new();
        let mut symbol_table = SymbolTable::new();

        for import in &file.imports {
            let resolved = self.resolve_import(import, base_dir)?;

            // Merge symbols from the imported file
            symbol_table.merge(&resolved.1);

            resolved_imports.push(ResolvedImport {
                directive: import.clone(),
                resolved_path: resolved.0,
                symbols: resolved.1.symbol_names(),
                rules: Vec::new(), // Rules from .llev files
            });
        }

        file.resolved_imports = resolved_imports;
        file.symbol_table = symbol_table;

        Ok(())
    }

    /// Resolve a single @import directive.
    fn resolve_import(
        &mut self,
        import: &ImportDirective,
        base_dir: Option<&Path>,
    ) -> LLreResult<(PathBuf, SymbolTable)> {
        // Build search paths (base directory first, then configured paths)
        let mut search_paths = Vec::new();
        if let Some(base) = base_dir {
            search_paths.push(base.to_path_buf());
        }
        search_paths.extend(self.config.search_paths.clone());

        // Find the file
        let resolved_path = self.find_file(&import.path, &search_paths).ok_or_else(|| {
            LLreError::with_position(
                LLreErrorKind::ImportNotFound {
                    path: import.path.clone(),
                    search_paths: search_paths.clone(),
                },
                import.position,
            )
        })?;

        // Check import depth
        if self.loading_stack.len() >= self.config.max_import_depth {
            return Err(LLreError::new(LLreErrorKind::ImportDepthExceeded {
                max: self.config.max_import_depth,
                path: resolved_path,
            }));
        }

        // Load the .llev file
        let llev_config = llev::LoaderConfig {
            include_paths: search_paths.clone(),
            max_include_depth: self.config.max_import_depth,
            allow_missing_includes: false,
        };

        let llev_loader = llev::Loader::with_config(llev_config);
        let llev_file = llev_loader
            .load(&resolved_path)
            .map_err(|e| LLreError::from_llev(&e))?;

        // Extract symbols from the .llev file
        let symbol_table = self.extract_symbols(&llev_file, import.alias.as_deref())?;

        Ok((resolved_path, symbol_table))
    }

    /// Find a file in the search paths.
    fn find_file(&self, path: &str, search_paths: &[PathBuf]) -> Option<PathBuf> {
        let path = Path::new(path);

        // If absolute, check directly
        if path.is_absolute() {
            if path.exists() {
                return Some(path.to_path_buf());
            }
            return None;
        }

        // Search in each path
        for base in search_paths {
            let full_path = base.join(path);
            if full_path.exists() {
                return Some(full_path);
            }
        }

        None
    }

    /// Canonicalize a path for comparison.
    fn canonicalize_path(&self, path: &Path) -> LLreResult<PathBuf> {
        std::fs::canonicalize(path).or_else(|_| {
            // If the file doesn't exist yet, just make it absolute
            if path.is_absolute() {
                Ok(path.to_path_buf())
            } else {
                std::env::current_dir()
                    .map(|cwd| cwd.join(path))
                    .map_err(|e| LLreError::new(LLreErrorKind::IoError(e.to_string())))
            }
        })
    }

    /// Extract symbols from an LLev file into a symbol table.
    fn extract_symbols(
        &self,
        file: &LLevFile,
        alias: Option<&str>,
    ) -> LLreResult<SymbolTable> {
        let mut table = SymbolTable::new();
        let source = file.source_file.clone();

        for symbol in &file.symbols {
            let name = if let Some(alias) = alias {
                format!("{}_{}", alias, symbol.name)
            } else {
                symbol.name.clone()
            };

            // Convert the symbol expression to a character class
            // This is a simplified extraction - full implementation would
            // evaluate the expression properly
            match &symbol.value {
                llev::Expression::CharClass { chars, .. } => {
                    table.add_char_class(name, chars.clone(), source.clone());
                }
                llev::Expression::Char(c) => {
                    // Single character as a one-element class
                    table.add_char_class(name, vec![*c], source.clone());
                }
                _ => {
                    // For complex expressions, we'd need to evaluate them
                    // For now, skip or convert to simple form
                }
            }
        }

        Ok(table)
    }
}

impl Default for Loader {
    fn default() -> Self {
        Self::new()
    }
}

/// Load a .llre file from disk.
pub fn load_file<P: AsRef<Path>>(path: P) -> LLreResult<LLreFile> {
    let mut loader = Loader::new();
    loader.load(path)
}

/// Load a .llre file with custom configuration.
pub fn load_file_with_config<P: AsRef<Path>>(path: P, config: LoaderConfig) -> LLreResult<LLreFile> {
    let mut loader = Loader::with_config(config);
    loader.load(path)
}

/// Parse a .llre file from a string (no import resolution).
pub fn parse_str(content: &str) -> LLreResult<LLreFile> {
    parser::parse_str(content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_loader_config_default() {
        let config = LoaderConfig::default();
        assert_eq!(config.search_paths, vec![PathBuf::from(".")]);
        assert_eq!(config.max_import_depth, 10);
        assert!(config.resolve_imports);
    }

    #[test]
    fn test_load_simple_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("test.llre");

        let content = r#"
            @name "Test"
            ^hello$
        "#;

        std::fs::write(&file_path, content).expect("Failed to write file");

        let file = load_file(&file_path).expect("Failed to load file");
        assert_eq!(file.metadata.name, Some("Test".to_string()));
        assert!(file.source_file.is_some());
    }

    #[test]
    fn test_load_file_not_found() {
        let result = load_file("/nonexistent/path/test.llre");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, LLreErrorKind::FileNotFound(_)));
    }

    #[test]
    fn test_load_with_import() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create a .llev file with symbols
        let llev_path = temp_dir.path().join("symbols.llev");
        let llev_content = r#"
            @name "Symbols"
            @define VOWEL = [aeiou]
        "#;
        std::fs::write(&llev_path, llev_content).expect("Failed to write llev file");

        // Create a .llre file that imports it
        let llre_path = temp_dir.path().join("test.llre");
        let llre_content = r#"
            @import "symbols.llev"
            ^[a-z]+$
        "#;
        std::fs::write(&llre_path, llre_content).expect("Failed to write llre file");

        // Configure loader with the temp directory in search paths
        let config = LoaderConfig {
            search_paths: vec![temp_dir.path().to_path_buf()],
            ..Default::default()
        };

        let file = load_file_with_config(&llre_path, config).expect("Failed to load file");
        assert_eq!(file.imports.len(), 1);
        assert_eq!(file.resolved_imports.len(), 1);
        // The VOWEL symbol should be in the symbol table
        assert!(file.symbol_table.contains("VOWEL"));
    }

    #[test]
    fn test_load_with_aliased_import() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create a .llev file with symbols
        let llev_path = temp_dir.path().join("english.llev");
        let llev_content = r#"
            @name "English"
            @define VOWEL = [aeiou]
        "#;
        std::fs::write(&llev_path, llev_content).expect("Failed to write llev file");

        // Create a .llre file that imports it with alias
        let llre_path = temp_dir.path().join("test.llre");
        let llre_content = r#"
            @import "english.llev" as en
            ^[a-z]+$
        "#;
        std::fs::write(&llre_path, llre_content).expect("Failed to write llre file");

        let config = LoaderConfig {
            search_paths: vec![temp_dir.path().to_path_buf()],
            ..Default::default()
        };

        let file = load_file_with_config(&llre_path, config).expect("Failed to load file");

        // The symbol should be prefixed with the alias
        assert!(file.symbol_table.contains("en_VOWEL"));
        assert!(!file.symbol_table.contains("VOWEL"));
    }

    #[test]
    fn test_import_not_found() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        // Create a .llre file that imports a nonexistent file
        let llre_path = temp_dir.path().join("test.llre");
        let llre_content = r#"
            @import "nonexistent.llev"
            ^test$
        "#;
        std::fs::write(&llre_path, llre_content).expect("Failed to write llre file");

        let result = load_file(&llre_path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, LLreErrorKind::ImportNotFound { .. }));
    }

    #[test]
    fn test_parse_str_no_imports() {
        let content = r#"
            @name "Test"
            @import "symbols.llev"  # This won't be resolved
            ^hello$
        "#;

        // parse_str doesn't resolve imports
        let file = parse_str(content).expect("Failed to parse");
        assert_eq!(file.imports.len(), 1);
        // Imports are parsed but not resolved
        assert!(file.resolved_imports.is_empty());
    }
}
