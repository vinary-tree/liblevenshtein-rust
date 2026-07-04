//! File loader for `.llre` files with import resolution.
//!
//! This module handles loading .llre files from disk and resolving @import
//! directives by loading the referenced .llev files.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::phonetic::common::flags::ParsedFlags;
use crate::phonetic::llev::{self, LLevFile};
use crate::phonetic::nfa::types::CharClassChar;
use crate::phonetic::regex::ast::{Regex, RegexFlags, UnicodeNormalization};

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
            let result = self.resolve_imports(&mut file, path.parent());
            self.loading_stack.remove(&canonical);
            result?;
        }

        Ok(file)
    }

    /// Load a .llre file from a string (no import resolution).
    pub fn load_str(&self, content: &str) -> LLreResult<LLreFile> {
        parser::parse_str(content)
    }

    /// Resolve @import directives in the file.
    fn resolve_imports(&mut self, file: &mut LLreFile, base_dir: Option<&Path>) -> LLreResult<()> {
        let mut resolved_imports = Vec::with_capacity(file.imports.len());
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
    fn extract_symbols(&self, file: &LLevFile, alias: Option<&str>) -> LLreResult<SymbolTable> {
        let mut table = SymbolTable::new();
        let source = file.source_file.clone();

        for symbol in &file.symbols {
            let name = if let Some(alias) = alias {
                format!("{}_{}", alias, symbol.name)
            } else {
                symbol.name.clone()
            };

            if let Some(chars) = expression_to_char_class(&symbol.value) {
                table.add_char_class(name, chars, source.clone());
            } else {
                let pattern = llev_expression_to_regex(&symbol.value, alias).map_err(|err| {
                    err.with_context(format!("while importing LLev symbol '{}'", symbol.name))
                })?;
                table.add_pattern(name, pattern, source.clone());
            }
        }

        Ok(table)
    }
}

fn expression_to_char_class(expr: &llev::Expression) -> Option<Vec<char>> {
    match expr {
        llev::Expression::Char(c) => Some(vec![*c]),
        llev::Expression::CharClass {
            chars,
            negated: false,
        } => Some(chars.clone()),
        llev::Expression::Alt(left, right) => {
            let mut chars = expression_to_char_class(left)?;
            chars.extend(expression_to_char_class(right)?);
            chars.sort_unstable();
            chars.dedup();
            Some(chars)
        }
        llev::Expression::Group(inner) => expression_to_char_class(inner),
        _ => None,
    }
}

fn llev_expression_to_regex(expr: &llev::Expression, alias: Option<&str>) -> LLreResult<Regex> {
    match expr {
        llev::Expression::Empty => Ok(Regex::empty()),
        llev::Expression::Char(c) => Ok(Regex::char(*c)),
        llev::Expression::CharClass { chars, negated } => {
            let class = CharClassChar::from_chars(chars);
            Ok(Regex::char_class(if *negated {
                class.negated()
            } else {
                class
            }))
        }
        llev::Expression::CharRange { start, end } => {
            Ok(Regex::char_class(CharClassChar::from_range(*start, *end)))
        }
        llev::Expression::Any => Ok(Regex::any()),
        llev::Expression::Concat(left, right) => Ok(Regex::concat(
            llev_expression_to_regex(left, alias)?,
            llev_expression_to_regex(right, alias)?,
        )),
        llev::Expression::Alt(left, right) => Ok(Regex::alt(
            llev_expression_to_regex(left, alias)?,
            llev_expression_to_regex(right, alias)?,
        )),
        llev::Expression::Star(inner) => Ok(Regex::star(llev_expression_to_regex(inner, alias)?)),
        llev::Expression::Plus(inner) => Ok(Regex::plus(llev_expression_to_regex(inner, alias)?)),
        llev::Expression::Optional(inner) => {
            Ok(Regex::optional(llev_expression_to_regex(inner, alias)?))
        }
        llev::Expression::RepeatExact(inner, count) => Ok(Regex::repeat_exact(
            llev_expression_to_regex(inner, alias)?,
            *count,
        )),
        llev::Expression::RepeatRange { inner, min, max } => Ok(Regex::repeat_range(
            llev_expression_to_regex(inner, alias)?,
            *min,
            *max,
        )),
        llev::Expression::Group(inner) => Ok(Regex::non_capturing_group(llev_expression_to_regex(
            inner, alias,
        )?)),
        llev::Expression::ScopedFlags { flags, inner } => Ok(Regex::flags_group(
            parsed_flags_to_regex_flags(flags)?,
            llev_expression_to_regex(inner, alias)?,
        )),
        llev::Expression::WordBoundary => Ok(Regex::word_boundary()),
        llev::Expression::SymbolRef(name) => Ok(Regex::group_ref(match alias {
            Some(alias) => format!("{}_{}", alias, name),
            None => name.clone(),
        })),
    }
}

fn parsed_flags_to_regex_flags(parsed: &ParsedFlags) -> LLreResult<RegexFlags> {
    let unicode_normalization = parsed
        .unicode_normalization
        .as_deref()
        .map(|norm| {
            norm.parse::<UnicodeNormalization>().map_err(|err| {
                LLreError::new(LLreErrorKind::InvalidFlag(format!(
                    "{} in imported LLev scoped flag",
                    err
                )))
            })
        })
        .transpose()?;

    Ok(RegexFlags {
        case_insensitive: parsed.case_insensitive,
        unicode_normalization,
        feature_based: parsed.feature_based,
        accent_insensitive: parsed.accent_insensitive,
        multiline: parsed.multiline,
        dotall: parsed.dotall,
        local_distance: parsed.levenshtein_distance,
    })
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
pub fn load_file_with_config<P: AsRef<Path>>(
    path: P,
    config: LoaderConfig,
) -> LLreResult<LLreFile> {
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
    fn test_load_imports_composite_llev_symbol_as_pattern() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let llev_path = temp_dir.path().join("symbols.llev");
        let llev_content = r#"
            @name "Symbols"
            @define PHONE_START = (ph|f)o+
        "#;
        std::fs::write(&llev_path, llev_content).expect("Failed to write llev file");

        let llre_path = temp_dir.path().join("test.llre");
        let llre_content = r#"
            @import "symbols.llev"
            ^(?&PHONE_START)ne$
        "#;
        std::fs::write(&llre_path, llre_content).expect("Failed to write llre file");

        let config = LoaderConfig {
            search_paths: vec![temp_dir.path().to_path_buf()],
            ..Default::default()
        };

        let file = load_file_with_config(&llre_path, config).expect("Failed to load file");
        let pattern = file
            .symbol_table
            .get_pattern("PHONE_START")
            .expect("composite symbol should import as a pattern");

        assert!(matches!(pattern, Regex::Concat(_, _)));
        assert!(file.symbol_table.get_char_class("PHONE_START").is_none());

        let compiled =
            crate::phonetic::llre::compile(&file).expect("imported pattern should compile");
        assert!(compiled.matches_full("phone"));
        assert!(compiled.matches_full("fone"));
        assert!(!compiled.matches_full("pone"));
    }

    #[test]
    fn test_load_imports_llev_range_symbol_without_expanding_to_vec() {
        let expr = llev::Expression::CharRange {
            start: 'a',
            end: 'z',
        };

        assert!(expression_to_char_class(&expr).is_none());

        let regex = llev_expression_to_regex(&expr, None).expect("range should convert");
        match regex {
            Regex::CharClass(class) => {
                assert_eq!(class.ranges, vec![('a', 'z')]);
                assert!(!class.negated);
            }
            other => panic!("expected compact char class range, got {:?}", other),
        }
    }

    #[test]
    fn test_load_aliased_composite_symbol_rewrites_internal_refs() {
        let expr = llev::Expression::Concat(
            Box::new(llev::Expression::SymbolRef("VOWEL".to_string())),
            Box::new(llev::Expression::Char('r')),
        );

        let regex =
            llev_expression_to_regex(&expr, Some("en")).expect("symbol refs should convert");

        match regex {
            Regex::Concat(left, right) => {
                assert!(matches!(*left, Regex::GroupRef(ref name) if name == "en_VOWEL"));
                assert_eq!(*right, Regex::Char('r'));
            }
            other => panic!("expected concat, got {:?}", other),
        }
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
    fn test_failed_import_does_not_poison_loader_stack() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let llre_path = temp_dir.path().join("test.llre");
        let llre_content = r#"
            @import "nonexistent.llev"
            ^test$
        "#;
        std::fs::write(&llre_path, llre_content).expect("Failed to write llre file");

        let mut loader = Loader::with_config(LoaderConfig {
            search_paths: vec![temp_dir.path().to_path_buf()],
            ..Default::default()
        });

        for _ in 0..2 {
            let err = loader
                .load(&llre_path)
                .expect_err("missing import should fail consistently");
            assert!(matches!(err.kind, LLreErrorKind::ImportNotFound { .. }));
        }
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
