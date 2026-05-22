//! AST types for `.llre` (LibLevenshtein Regex Expression) files.
//!
//! A `.llre` file contains a single regex pattern with optional metadata, imports,
//! and global flags.
//!
//! # File Format
//!
//! ```text
//! # Example .llre file
//! @name "Email Validator"
//! @version "1.0"
//! @author "LibLevenshtein Team"
//! @description "Validates email addresses"
//!
//! # Import symbols from .llev files
//! @import "phonetic/symbols.llev"
//! @import "phonetic/english.llev" as english
//!
//! # Global flags (optional)
//! @flags multiline, dotall
//!
//! # The regex pattern (required, single pattern per file)
//! ^[[:alpha:]][[:alnum:]._-]*@$DOMAIN+\.$TLD$
//! ```
//!
//! # Grammar (EBNF)
//!
//! ```ebnf
//! llre_file     = { directive } , pattern ;
//! directive     = "@" , directive_name , directive_value ;
//! directive_name = "name" | "version" | "author" | "description" | "import" | "flags" ;
//! import_spec   = string_literal , [ "as" , identifier ] ;
//! flag_list     = flag_name , { "," , flag_name } ;
//! flag_name     = "multiline" | "dotall" | "case_insensitive" ;
//! pattern       = regex_expression ;
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use crate::phonetic::common::Position;
use crate::phonetic::regex::ast::{Regex, RegexFlags};

/// A complete `.llre` file AST.
#[derive(Debug, Clone)]
pub struct LLreFile {
    /// File metadata (name, version, author, description)
    pub metadata: FileMetadata,

    /// Import directives
    pub imports: Vec<ImportDirective>,

    /// Global flags from @flags directive
    pub global_flags: LLreFlags,

    /// The regex pattern (parsed AST)
    pub pattern: Regex,

    /// Raw pattern string (for error messages and debugging)
    pub pattern_source: String,

    /// Position where the pattern starts
    pub pattern_position: Position,

    /// Source file path (if loaded from file)
    pub source_file: Option<PathBuf>,

    /// Resolved import paths (after loader processing)
    pub resolved_imports: Vec<ResolvedImport>,

    /// Merged symbol table from all imports
    pub symbol_table: SymbolTable,
}

impl LLreFile {
    /// Create a new empty LLreFile (for builder pattern).
    pub fn new(pattern: Regex, pattern_source: String, pattern_position: Position) -> Self {
        Self {
            metadata: FileMetadata::default(),
            imports: Vec::new(),
            global_flags: LLreFlags::default(),
            pattern,
            pattern_source,
            pattern_position,
            source_file: None,
            resolved_imports: Vec::new(),
            symbol_table: SymbolTable::default(),
        }
    }

    /// Set the source file path.
    pub fn with_source_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.source_file = Some(path.into());
        self
    }

    /// Get the effective regex flags (combining global flags and inline flags).
    pub fn effective_flags(&self) -> RegexFlags {
        RegexFlags {
            case_insensitive: self.global_flags.case_insensitive,
            multiline: self.global_flags.multiline,
            dotall: self.global_flags.dotall,
            ..Default::default()
        }
    }

    /// Check if the pattern uses multiline mode.
    pub fn is_multiline(&self) -> bool {
        self.global_flags.multiline.unwrap_or(false)
    }

    /// Check if the pattern uses dotall mode.
    pub fn is_dotall(&self) -> bool {
        self.global_flags.dotall.unwrap_or(false)
    }
}

/// File metadata from directives.
#[derive(Debug, Clone, Default)]
pub struct FileMetadata {
    /// @name directive value
    pub name: Option<String>,

    /// @version directive value
    pub version: Option<String>,

    /// @author directive value
    pub author: Option<String>,

    /// @description directive value
    pub description: Option<String>,
}

impl FileMetadata {
    /// Create new metadata with a name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Default::default()
        }
    }
}

/// An @import directive.
#[derive(Debug, Clone)]
pub struct ImportDirective {
    /// The path to import (relative or absolute)
    pub path: String,

    /// Optional alias (e.g., @import "file.llev" as english)
    pub alias: Option<String>,

    /// Position in the source file
    pub position: Position,
}

impl ImportDirective {
    /// Create a new import directive.
    pub fn new(path: impl Into<String>, position: Position) -> Self {
        Self {
            path: path.into(),
            alias: None,
            position,
        }
    }

    /// Create an import directive with an alias.
    pub fn with_alias(
        path: impl Into<String>,
        alias: impl Into<String>,
        position: Position,
    ) -> Self {
        Self {
            path: path.into(),
            alias: Some(alias.into()),
            position,
        }
    }
}

/// A resolved import (after loader processing).
#[derive(Debug, Clone)]
pub struct ResolvedImport {
    /// Original import directive
    pub directive: ImportDirective,

    /// Resolved absolute path
    pub resolved_path: PathBuf,

    /// Symbols imported from this file
    pub symbols: Vec<String>,

    /// Rules imported from this file (if any)
    pub rules: Vec<String>,
}

/// Global flags for .llre files.
#[derive(Debug, Clone, Default)]
pub struct LLreFlags {
    /// Multiline mode (?m) - ^ and $ match line boundaries
    pub multiline: Option<bool>,

    /// Dotall mode (?s) - . matches newlines
    pub dotall: Option<bool>,

    /// Case insensitive mode (?i)
    pub case_insensitive: Option<bool>,

    /// Unicode mode (?u) - \w, \d, etc. match Unicode
    pub unicode: Option<bool>,
}

impl LLreFlags {
    /// Create flags with multiline mode enabled.
    pub fn multiline() -> Self {
        Self {
            multiline: Some(true),
            ..Default::default()
        }
    }

    /// Create flags with dotall mode enabled.
    pub fn dotall() -> Self {
        Self {
            dotall: Some(true),
            ..Default::default()
        }
    }

    /// Merge another set of flags (other takes precedence).
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            multiline: other.multiline.or(self.multiline),
            dotall: other.dotall.or(self.dotall),
            case_insensitive: other.case_insensitive.or(self.case_insensitive),
            unicode: other.unicode.or(self.unicode),
        }
    }

    /// Check if any flags are set.
    pub fn is_empty(&self) -> bool {
        self.multiline.is_none()
            && self.dotall.is_none()
            && self.case_insensitive.is_none()
            && self.unicode.is_none()
    }
}

/// Symbol table containing symbols from all imports.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    /// Character class symbols (e.g., VOWEL -> ['a', 'e', 'i', 'o', 'u'])
    pub char_classes: HashMap<String, Vec<char>>,

    /// Pattern symbols (e.g., DOMAIN -> regex pattern)
    pub patterns: HashMap<String, Regex>,

    /// Source file for each symbol (for error messages)
    pub symbol_sources: HashMap<String, PathBuf>,
}

impl SymbolTable {
    /// Create a new empty symbol table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a character class symbol.
    pub fn add_char_class(
        &mut self,
        name: impl Into<String>,
        chars: Vec<char>,
        source: Option<PathBuf>,
    ) {
        let name = name.into();
        if let Some(src) = source {
            self.symbol_sources.insert(name.clone(), src);
        }
        self.char_classes.insert(name, chars);
    }

    /// Add a pattern symbol.
    pub fn add_pattern(
        &mut self,
        name: impl Into<String>,
        pattern: Regex,
        source: Option<PathBuf>,
    ) {
        let name = name.into();
        if let Some(src) = source {
            self.symbol_sources.insert(name.clone(), src);
        }
        self.patterns.insert(name, pattern);
    }

    /// Look up a character class symbol.
    pub fn get_char_class(&self, name: &str) -> Option<&Vec<char>> {
        self.char_classes.get(name)
    }

    /// Look up a pattern symbol.
    pub fn get_pattern(&self, name: &str) -> Option<&Regex> {
        self.patterns.get(name)
    }

    /// Check if a symbol exists (either char class or pattern).
    pub fn contains(&self, name: &str) -> bool {
        self.char_classes.contains_key(name) || self.patterns.contains_key(name)
    }

    /// Get all symbol names.
    pub fn symbol_names(&self) -> Vec<String> {
        let mut names: Vec<_> = self.char_classes.keys().cloned().collect();
        names.extend(self.patterns.keys().cloned());
        names.sort();
        names.dedup();
        names
    }

    /// Merge another symbol table (other takes precedence on conflicts).
    pub fn merge(&mut self, other: &Self) {
        for (name, chars) in &other.char_classes {
            self.char_classes.insert(name.clone(), chars.clone());
        }
        for (name, pattern) in &other.patterns {
            self.patterns.insert(name.clone(), pattern.clone());
        }
        for (name, source) in &other.symbol_sources {
            self.symbol_sources.insert(name.clone(), source.clone());
        }
    }

    /// Get the source file for a symbol.
    pub fn get_source(&self, name: &str) -> Option<&PathBuf> {
        self.symbol_sources.get(name)
    }
}

/// A parsed directive from an .llre file.
#[derive(Debug, Clone)]
pub enum Directive {
    /// @name "value"
    Name(String, Position),

    /// @version "value"
    Version(String, Position),

    /// @author "value"
    Author(String, Position),

    /// @description "value"
    Description(String, Position),

    /// @import "path" [as alias]
    Import(ImportDirective),

    /// @flags flag1, flag2, ...
    Flags(LLreFlags, Position),
}

impl Directive {
    /// Get the position of this directive.
    pub fn position(&self) -> Position {
        match self {
            Directive::Name(_, pos) => *pos,
            Directive::Version(_, pos) => *pos,
            Directive::Author(_, pos) => *pos,
            Directive::Description(_, pos) => *pos,
            Directive::Import(import) => import.position,
            Directive::Flags(_, pos) => *pos,
        }
    }

    /// Get the directive name (for error messages).
    pub fn name(&self) -> &'static str {
        match self {
            Directive::Name(..) => "name",
            Directive::Version(..) => "version",
            Directive::Author(..) => "author",
            Directive::Description(..) => "description",
            Directive::Import(..) => "import",
            Directive::Flags(..) => "flags",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_metadata() {
        let meta = FileMetadata::with_name("Test Pattern");
        assert_eq!(meta.name, Some("Test Pattern".to_string()));
        assert!(meta.version.is_none());
    }

    #[test]
    fn test_import_directive() {
        let import = ImportDirective::new("symbols.llev", Position::new(1, 1, 0));
        assert_eq!(import.path, "symbols.llev");
        assert!(import.alias.is_none());

        let aliased = ImportDirective::with_alias("english.llev", "en", Position::new(2, 1, 20));
        assert_eq!(aliased.path, "english.llev");
        assert_eq!(aliased.alias, Some("en".to_string()));
    }

    #[test]
    fn test_llre_flags() {
        let flags = LLreFlags::multiline();
        assert_eq!(flags.multiline, Some(true));
        assert!(flags.dotall.is_none());

        let dotall = LLreFlags::dotall();
        let merged = flags.merge(&dotall);
        assert_eq!(merged.multiline, Some(true));
        assert_eq!(merged.dotall, Some(true));
    }

    #[test]
    fn test_symbol_table() {
        let mut table = SymbolTable::new();
        table.add_char_class("VOWEL", vec!['a', 'e', 'i', 'o', 'u'], None);

        assert!(table.contains("VOWEL"));
        assert!(!table.contains("CONSONANT"));

        let vowels = table
            .get_char_class("VOWEL")
            .expect("expected Some VOWEL class in test");
        assert_eq!(vowels.len(), 5);
        assert!(vowels.contains(&'a'));
    }

    #[test]
    fn test_symbol_table_merge() {
        let mut table1 = SymbolTable::new();
        table1.add_char_class("A", vec!['a'], None);

        let mut table2 = SymbolTable::new();
        table2.add_char_class("B", vec!['b'], None);
        table2.add_char_class("A", vec!['x'], None); // Override

        table1.merge(&table2);

        assert!(table1.contains("A"));
        assert!(table1.contains("B"));
        // table2's A should take precedence
        assert_eq!(
            table1
                .get_char_class("A")
                .expect("expected Some A class in test"),
            &vec!['x']
        );
    }
}
