//! Error types for `.llev` file parsing.
//!
//! This module provides comprehensive error types for parsing `.llev` files,
//! including file-level errors (I/O, circular includes) and parse-level errors
//! (syntax, undefined symbols).

use std::fmt;
use std::path::PathBuf;

// Re-export Position from common module for backward compatibility
pub use crate::phonetic::common::Position;

/// Error type for `.llev` file parsing and loading.
#[derive(Debug, Clone)]
pub struct LLevError {
    /// The kind of error
    pub kind: LLevErrorKind,
    /// Position where the error occurred (if applicable)
    pub position: Option<Position>,
    /// File where the error occurred (if applicable)
    pub file: Option<PathBuf>,
    /// Additional context about the error
    pub context: Option<String>,
}

/// The kind of `.llev` error.
#[derive(Debug, Clone, PartialEq)]
pub enum LLevErrorKind {
    // ==================== I/O Errors ====================
    /// File not found
    FileNotFound(String),

    /// Permission denied
    PermissionDenied(String),

    /// General I/O error
    IoError(String),

    // ==================== Include Errors ====================
    /// Circular include detected
    CircularInclude(PathBuf),

    /// Include depth exceeded
    IncludeDepthExceeded {
        max: usize,
        path: PathBuf,
    },

    /// Include file not found
    IncludeNotFound {
        path: String,
        search_paths: Vec<PathBuf>,
    },

    // ==================== Lexer Errors ====================
    /// Unexpected end of input
    UnexpectedEof,

    /// Unexpected character
    UnexpectedChar(char),

    /// Invalid escape sequence
    InvalidEscape(char),

    /// Unterminated string literal
    UnterminatedString,

    /// Unterminated block comment
    UnterminatedComment,

    /// Invalid Unicode code point
    InvalidCodePoint(u32),

    // ==================== Parser Errors ====================
    /// Expected a specific token
    ExpectedToken {
        expected: String,
        found: String,
    },

    /// Invalid directive
    InvalidDirective(String),

    /// Invalid metadata key
    InvalidMetadataKey(String),

    /// Invalid metadata value
    InvalidMetadataValue {
        key: String,
        value: String,
    },

    /// Missing required field
    MissingField(String),

    /// Duplicate field
    DuplicateField(String),

    /// Invalid rule syntax
    InvalidRuleSyntax(String),

    /// Invalid pattern
    InvalidPattern(String),

    /// Invalid replacement
    InvalidReplacement(String),

    /// Invalid context
    InvalidContext(String),

    /// Invalid weight value
    InvalidWeight(String),

    /// Unclosed group
    UnclosedGroup,

    /// Unclosed character class
    UnclosedCharClass,

    /// Unclosed metadata block
    UnclosedMetadata,

    // ==================== Conversion Errors ====================
    /// Invalid rule (e.g., empty pattern)
    InvalidRule(String),

    /// Unsupported pattern for conversion (requires NFA-based matching)
    UnsupportedPattern(String),

    /// Non-ASCII character in byte-level rule set
    ///
    /// The byte-level `RuleSet` only supports ASCII characters (0-127).
    /// For rules containing non-ASCII characters (Unicode), use `RuleSetChar` instead.
    NonAsciiInByteLevel {
        /// The non-ASCII character that was found
        character: char,
        /// The rule name, if available
        rule_name: Option<String>,
    },

    // ==================== Semantic Errors ====================
    /// Undefined symbol reference
    UndefinedSymbol(String),

    /// Duplicate symbol definition
    DuplicateSymbol(String),

    /// Invalid symbol name
    InvalidSymbolName(String),

    /// Type mismatch
    TypeMismatch {
        /// The expected type
        expected: String,
        /// The found type
        found: String,
    },

    /// Attempt to define a symbol that conflicts with a built-in class name
    BuiltinClassConflict {
        /// The name that conflicts with a built-in
        name: String,
    },

    /// Symbol name must be UPPERCASE (user-defined symbols must be uppercase)
    SymbolNameMustBeUppercase {
        /// The invalid symbol name
        name: String,
    },

    /// Unknown built-in named class (e.g., `[:invalid:]`)
    UnknownNamedClass {
        /// The class name that was not found
        name: String,
        /// Available built-in class names
        available: Vec<String>,
    },

    // ==================== Compilation Errors ====================
    /// Serialization error
    SerializationError(String),

    /// Deserialization error
    DeserializationError(String),

    /// Invalid compiled format (version mismatch)
    InvalidCompiledFormat {
        expected_version: u32,
        found_version: u32,
    },

    /// Invalid format (generic format error)
    InvalidFormat(String),
}

impl LLevError {
    /// Create a new error.
    pub fn new(kind: LLevErrorKind) -> Self {
        Self {
            kind,
            position: None,
            file: None,
            context: None,
        }
    }

    /// Create an error with position.
    pub fn with_position(kind: LLevErrorKind, position: Position) -> Self {
        Self {
            kind,
            position: Some(position),
            file: None,
            context: None,
        }
    }

    /// Create an error with file and position.
    pub fn with_file_position(kind: LLevErrorKind, file: PathBuf, position: Position) -> Self {
        Self {
            kind,
            position: Some(position),
            file: Some(file),
            context: None,
        }
    }

    /// Create an error with context.
    pub fn with_context(kind: LLevErrorKind, context: impl Into<String>) -> Self {
        Self {
            kind,
            position: None,
            file: None,
            context: Some(context.into()),
        }
    }

    /// Add position to an existing error.
    pub fn at_position(mut self, position: Position) -> Self {
        self.position = Some(position);
        self
    }

    /// Add file to an existing error.
    pub fn in_file(mut self, file: PathBuf) -> Self {
        self.file = Some(file);
        self
    }

    /// Add context to an existing error.
    pub fn with_added_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    // ==================== Convenience Constructors ====================

    /// Create a "file not found" error.
    pub fn file_not_found(path: impl Into<String>) -> Self {
        Self::new(LLevErrorKind::FileNotFound(path.into()))
    }

    /// Create an "I/O error" error.
    pub fn io_error(msg: impl Into<String>) -> Self {
        Self::new(LLevErrorKind::IoError(msg.into()))
    }

    /// Create a "circular include" error.
    pub fn circular_include(path: PathBuf) -> Self {
        Self::new(LLevErrorKind::CircularInclude(path))
    }

    /// Create an "include depth exceeded" error.
    pub fn include_depth_exceeded(max: usize, path: PathBuf) -> Self {
        Self::new(LLevErrorKind::IncludeDepthExceeded { max, path })
    }

    /// Create an "include not found" error.
    pub fn include_not_found(path: impl Into<String>, search_paths: Vec<PathBuf>) -> Self {
        Self::new(LLevErrorKind::IncludeNotFound {
            path: path.into(),
            search_paths,
        })
    }

    /// Create an "unexpected EOF" error.
    pub fn unexpected_eof(position: Position) -> Self {
        Self::with_position(LLevErrorKind::UnexpectedEof, position)
    }

    /// Create an "unexpected character" error.
    pub fn unexpected_char(c: char, position: Position) -> Self {
        Self::with_position(LLevErrorKind::UnexpectedChar(c), position)
    }

    /// Create an "invalid escape" error.
    pub fn invalid_escape(c: char, position: Position) -> Self {
        Self::with_position(LLevErrorKind::InvalidEscape(c), position)
    }

    /// Create an "unterminated string" error.
    pub fn unterminated_string(position: Position) -> Self {
        Self::with_position(LLevErrorKind::UnterminatedString, position)
    }

    /// Create an "unterminated comment" error.
    pub fn unterminated_comment(position: Position) -> Self {
        Self::with_position(LLevErrorKind::UnterminatedComment, position)
    }

    /// Create an "expected token" error.
    pub fn expected_token(expected: impl Into<String>, found: impl Into<String>, position: Position) -> Self {
        Self::with_position(
            LLevErrorKind::ExpectedToken {
                expected: expected.into(),
                found: found.into(),
            },
            position,
        )
    }

    /// Create an "invalid directive" error.
    pub fn invalid_directive(directive: impl Into<String>, position: Position) -> Self {
        Self::with_position(LLevErrorKind::InvalidDirective(directive.into()), position)
    }

    /// Create an "undefined symbol" error.
    pub fn undefined_symbol(symbol: impl Into<String>, position: Position) -> Self {
        Self::with_position(LLevErrorKind::UndefinedSymbol(symbol.into()), position)
    }

    /// Create an "undefined symbol" error with a suggestion for a similar symbol.
    ///
    /// Uses Levenshtein distance to find the most similar defined symbol.
    /// If a close match is found (distance <= 2), includes "did you mean 'X'?" in the error.
    pub fn undefined_symbol_with_suggestion(
        symbol: impl Into<String>,
        defined_symbols: &[&str],
        position: Position,
    ) -> Self {
        let name = symbol.into();

        // Find the closest matching defined symbol
        let suggestion = find_closest_symbol(&name, defined_symbols);

        let mut err = Self::with_position(LLevErrorKind::UndefinedSymbol(name), position);

        if let Some(suggested) = suggestion {
            err.context = Some(format!("did you mean '{}'?", suggested));
        }

        err
    }

    /// Create a "duplicate symbol" error.
    pub fn duplicate_symbol(symbol: impl Into<String>, position: Position) -> Self {
        Self::with_position(LLevErrorKind::DuplicateSymbol(symbol.into()), position)
    }

    /// Create a "built-in class conflict" error.
    ///
    /// This error indicates that a user-defined symbol name conflicts with
    /// a built-in character class name (e.g., "vowel", "consonant", "alpha").
    pub fn builtin_class_conflict(name: impl Into<String>, position: Position) -> Self {
        Self::with_position(
            LLevErrorKind::BuiltinClassConflict { name: name.into() },
            position,
        )
    }

    /// Create a "symbol name must be uppercase" error.
    ///
    /// User-defined symbols must be UPPERCASE to distinguish them from
    /// built-in character classes (which are lowercase).
    pub fn symbol_name_must_be_uppercase(name: impl Into<String>, position: Position) -> Self {
        Self::with_position(
            LLevErrorKind::SymbolNameMustBeUppercase { name: name.into() },
            position,
        )
    }

    /// Create a "non-ASCII in byte-level" error.
    ///
    /// This error indicates that a non-ASCII character was found in a rule
    /// when using the byte-level `RuleSet`. The user should use `RuleSetChar`
    /// for Unicode support.
    pub fn non_ascii_in_byte_level(
        character: char,
        rule_name: Option<String>,
        position: Position,
    ) -> Self {
        Self::with_position(
            LLevErrorKind::NonAsciiInByteLevel {
                character,
                rule_name,
            },
            position,
        )
    }

    /// Create a "serialization error" error.
    pub fn serialization_error(msg: impl Into<String>) -> Self {
        Self::new(LLevErrorKind::SerializationError(msg.into()))
    }

    /// Create a "deserialization error" error.
    pub fn deserialization_error(msg: impl Into<String>) -> Self {
        Self::new(LLevErrorKind::DeserializationError(msg.into()))
    }
}

impl fmt::Display for LLevError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Write file and position if available
        if let Some(ref file) = self.file {
            write!(f, "{}:", file.display())?;
            if let Some(ref pos) = self.position {
                write!(f, "{}:{}: ", pos.line, pos.column)?;
            } else {
                write!(f, " ")?;
            }
        } else if let Some(ref pos) = self.position {
            write!(f, "{}: ", pos)?;
        }

        // Write the error kind
        write!(f, "{}", self.kind)?;

        // Write context if available
        if let Some(ref ctx) = self.context {
            write!(f, " ({})", ctx)?;
        }

        Ok(())
    }
}

impl fmt::Display for LLevErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // I/O Errors
            LLevErrorKind::FileNotFound(path) => {
                write!(f, "file not found: {}", path)
            }
            LLevErrorKind::PermissionDenied(path) => {
                write!(f, "permission denied: {}", path)
            }
            LLevErrorKind::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }

            // Include Errors
            LLevErrorKind::CircularInclude(path) => {
                write!(f, "circular include detected: {}", path.display())
            }
            LLevErrorKind::IncludeDepthExceeded { max, path } => {
                write!(
                    f,
                    "include depth exceeded (max {}): {}",
                    max,
                    path.display()
                )
            }
            LLevErrorKind::IncludeNotFound { path, search_paths } => {
                write!(f, "include file not found: '{}' (searched: ", path)?;
                for (i, sp) in search_paths.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", sp.display())?;
                }
                write!(f, ")")
            }

            // Lexer Errors
            LLevErrorKind::UnexpectedEof => {
                write!(f, "unexpected end of input")
            }
            LLevErrorKind::UnexpectedChar(c) => {
                write!(f, "unexpected character '{}'", c)
            }
            LLevErrorKind::InvalidEscape(c) => {
                write!(f, "invalid escape sequence '\\{}'", c)
            }
            LLevErrorKind::UnterminatedString => {
                write!(f, "unterminated string literal")
            }
            LLevErrorKind::UnterminatedComment => {
                write!(f, "unterminated block comment")
            }
            LLevErrorKind::InvalidCodePoint(cp) => {
                write!(f, "invalid Unicode code point: U+{:04X}", cp)
            }

            // Parser Errors
            LLevErrorKind::ExpectedToken { expected, found } => {
                write!(f, "expected {}, found {}", expected, found)
            }
            LLevErrorKind::InvalidDirective(directive) => {
                write!(f, "invalid directive: @{}", directive)
            }
            LLevErrorKind::InvalidMetadataKey(key) => {
                write!(f, "invalid metadata key: '{}'", key)
            }
            LLevErrorKind::InvalidMetadataValue { key, value } => {
                write!(f, "invalid value for '{}': '{}'", key, value)
            }
            LLevErrorKind::MissingField(field) => {
                write!(f, "missing required field: {}", field)
            }
            LLevErrorKind::DuplicateField(field) => {
                write!(f, "duplicate field: {}", field)
            }
            LLevErrorKind::InvalidRuleSyntax(msg) => {
                write!(f, "invalid rule syntax: {}", msg)
            }
            LLevErrorKind::InvalidPattern(msg) => {
                write!(f, "invalid pattern: {}", msg)
            }
            LLevErrorKind::InvalidReplacement(msg) => {
                write!(f, "invalid replacement: {}", msg)
            }
            LLevErrorKind::InvalidContext(msg) => {
                write!(f, "invalid context: {}", msg)
            }
            LLevErrorKind::InvalidWeight(msg) => {
                write!(f, "invalid weight: {}", msg)
            }
            LLevErrorKind::UnclosedGroup => {
                write!(f, "unclosed group '('")
            }
            LLevErrorKind::UnclosedCharClass => {
                write!(f, "unclosed character class '['")
            }
            LLevErrorKind::UnclosedMetadata => {
                write!(f, "unclosed metadata block '['")
            }

            // Conversion Errors
            LLevErrorKind::InvalidRule(msg) => {
                write!(f, "invalid rule: {}", msg)
            }
            LLevErrorKind::UnsupportedPattern(msg) => {
                write!(f, "unsupported pattern: {}", msg)
            }
            LLevErrorKind::NonAsciiInByteLevel {
                character,
                rule_name,
            } => {
                write!(
                    f,
                    "non-ASCII character '{}' (U+{:04X}) in byte-level rule",
                    character,
                    *character as u32
                )?;
                if let Some(name) = rule_name {
                    write!(f, " '{}'", name)?;
                }
                write!(f, "; use RuleSetChar for Unicode support")
            }

            // Semantic Errors
            LLevErrorKind::UndefinedSymbol(symbol) => {
                write!(f, "undefined symbol: {}", symbol)
            }
            LLevErrorKind::DuplicateSymbol(symbol) => {
                write!(f, "duplicate symbol definition: {}", symbol)
            }
            LLevErrorKind::InvalidSymbolName(name) => {
                write!(f, "invalid symbol name: {}", name)
            }
            LLevErrorKind::TypeMismatch { expected, found } => {
                write!(f, "type mismatch: expected {}, found {}", expected, found)
            }
            LLevErrorKind::BuiltinClassConflict { name } => {
                write!(
                    f,
                    "cannot define symbol '{}': conflicts with built-in character class",
                    name
                )
            }
            LLevErrorKind::SymbolNameMustBeUppercase { name } => {
                write!(
                    f,
                    "symbol name '{}' must be UPPERCASE (built-in classes are lowercase)",
                    name
                )
            }
            LLevErrorKind::UnknownNamedClass { name, available } => {
                write!(f, "unknown built-in character class '[:{}:]'", name)?;
                if !available.is_empty() {
                    write!(f, " (available: ")?;
                    for (i, avail) in available.iter().take(5).enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", avail)?;
                    }
                    if available.len() > 5 {
                        write!(f, ", ... ({} more)", available.len() - 5)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }

            // Compilation Errors
            LLevErrorKind::SerializationError(msg) => {
                write!(f, "serialization error: {}", msg)
            }
            LLevErrorKind::DeserializationError(msg) => {
                write!(f, "deserialization error: {}", msg)
            }
            LLevErrorKind::InvalidCompiledFormat {
                expected_version,
                found_version,
            } => {
                write!(
                    f,
                    "invalid compiled format: expected version {}, found {}",
                    expected_version, found_version
                )
            }
            LLevErrorKind::InvalidFormat(msg) => {
                write!(f, "invalid format: {}", msg)
            }
        }
    }
}

impl std::error::Error for LLevError {}

impl From<std::io::Error> for LLevError {
    fn from(err: std::io::Error) -> Self {
        use std::io::ErrorKind;
        match err.kind() {
            ErrorKind::NotFound => Self::new(LLevErrorKind::FileNotFound(err.to_string())),
            ErrorKind::PermissionDenied => {
                Self::new(LLevErrorKind::PermissionDenied(err.to_string()))
            }
            _ => Self::new(LLevErrorKind::IoError(err.to_string())),
        }
    }
}

/// Result type for `.llev` operations.
pub type LLevResult<T> = Result<T, LLevError>;

// ============================================================================
// Symbol Suggestion Helper Functions
// ============================================================================

/// Find the closest matching symbol from a list of defined symbols.
///
/// Returns `Some(symbol)` if a close match is found (Levenshtein distance <= 2),
/// otherwise returns `None`.
fn find_closest_symbol<'a>(target: &str, candidates: &[&'a str]) -> Option<&'a str> {
    const MAX_DISTANCE: usize = 2;

    let target_lower = target.to_lowercase();

    candidates
        .iter()
        .filter_map(|&candidate| {
            let candidate_lower = candidate.to_lowercase();
            let distance = levenshtein_distance(&target_lower, &candidate_lower);
            if distance <= MAX_DISTANCE {
                Some((candidate, distance))
            } else {
                None
            }
        })
        .min_by_key(|(_, d)| *d)
        .map(|(s, _)| s)
}

/// Compute the Levenshtein (edit) distance between two strings.
///
/// This is a simple O(n*m) implementation sufficient for symbol suggestions.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two rows instead of full matrix for O(min(m,n)) space
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    // Initialize first row
    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;

        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            curr[j] = (prev[j] + 1) // deletion
                .min(curr[j - 1] + 1) // insertion
                .min(prev[j - 1] + cost); // substitution
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_simple() {
        let err = LLevError::unexpected_char('x', Position::new(1, 5, 4));
        assert_eq!(err.to_string(), "line 1, column 5: unexpected character 'x'");
    }

    #[test]
    fn test_error_display_with_file() {
        let err = LLevError::unexpected_char('x', Position::new(1, 5, 4))
            .in_file(PathBuf::from("test.llev"));
        assert_eq!(
            err.to_string(),
            "test.llev:1:5: unexpected character 'x'"
        );
    }

    #[test]
    fn test_error_display_with_context() {
        let err = LLevError::invalid_directive("foo", Position::new(1, 3, 2))
            .with_added_context("expected @include, @define, etc.");
        assert!(err.to_string().contains("invalid directive: @foo"));
        assert!(err.to_string().contains("expected @include"));
    }

    #[test]
    fn test_circular_include() {
        let err = LLevError::circular_include(PathBuf::from("rules.llev"));
        assert!(err.to_string().contains("circular include"));
        assert!(err.to_string().contains("rules.llev"));
    }

    #[test]
    fn test_include_not_found() {
        let err = LLevError::include_not_found(
            "missing.llev",
            vec![PathBuf::from("./rules"), PathBuf::from("/usr/share/llev")],
        );
        let s = err.to_string();
        assert!(s.contains("include file not found"));
        assert!(s.contains("missing.llev"));
        assert!(s.contains("./rules"));
        assert!(s.contains("/usr/share/llev"));
    }

    #[test]
    fn test_undefined_symbol() {
        let err = LLevError::undefined_symbol("VOWEL", Position::new(10, 5, 100));
        assert!(err.to_string().contains("undefined symbol: VOWEL"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let llev_err: LLevError = io_err.into();
        assert!(matches!(llev_err.kind, LLevErrorKind::FileNotFound(_)));
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", "abd"), 1);
        assert_eq!(levenshtein_distance("abc", "adc"), 1);
        assert_eq!(levenshtein_distance("abc", "abcd"), 1);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("VOWEL", "vowel"), 5); // Case sensitive
    }

    #[test]
    fn test_find_closest_symbol() {
        let symbols = &["FRONT_VOWEL", "BACK_VOWEL", "CONSONANT"];

        // Exact match (after lowercase comparison)
        assert_eq!(find_closest_symbol("front_vowel", symbols), Some("FRONT_VOWEL"));

        // Close match (1 character difference)
        assert_eq!(find_closest_symbol("FRONT_VOWL", symbols), Some("FRONT_VOWEL"));

        // Close match (case difference + typo)
        assert_eq!(find_closest_symbol("front_vowl", symbols), Some("FRONT_VOWEL"));

        // No match (too different)
        assert_eq!(find_closest_symbol("xyz", symbols), None);

        // Close match for CONSONANT
        assert_eq!(find_closest_symbol("CONSNANT", symbols), Some("CONSONANT"));
    }

    #[test]
    fn test_undefined_symbol_with_suggestion() {
        let symbols = &["FRONT_VOWEL", "BACK_VOWEL", "CONSONANT"];
        let err = LLevError::undefined_symbol_with_suggestion(
            "front_vowel",
            symbols,
            Position::new(10, 5, 100),
        );

        let s = err.to_string();
        assert!(s.contains("undefined symbol: front_vowel"));
        assert!(s.contains("did you mean 'FRONT_VOWEL'?"));
    }

    #[test]
    fn test_undefined_symbol_no_suggestion() {
        let symbols = &["FRONT_VOWEL", "BACK_VOWEL", "CONSONANT"];
        let err = LLevError::undefined_symbol_with_suggestion(
            "xyz",
            symbols,
            Position::new(10, 5, 100),
        );

        let s = err.to_string();
        assert!(s.contains("undefined symbol: xyz"));
        assert!(!s.contains("did you mean"));
    }

    #[test]
    fn test_non_ascii_in_byte_level() {
        let err = LLevError::non_ascii_in_byte_level('ü', None, Position::new(5, 10, 50));
        let s = err.to_string();

        // Check error message components
        assert!(s.contains("non-ASCII character"));
        assert!(s.contains("ü"));
        assert!(s.contains("U+00FC")); // Unicode code point for ü
        assert!(s.contains("RuleSetChar"));
    }

    #[test]
    fn test_non_ascii_in_byte_level_with_rule_name() {
        let err =
            LLevError::non_ascii_in_byte_level('é', Some("french-vowels".to_string()), Position::new(5, 10, 50));
        let s = err.to_string();

        // Check error message includes rule name
        assert!(s.contains("non-ASCII character"));
        assert!(s.contains("é"));
        assert!(s.contains("french-vowels"));
        assert!(s.contains("RuleSetChar"));
    }
}
