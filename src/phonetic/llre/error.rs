//! Error types for `.llre` file parsing and compilation.
//!
//! This module provides comprehensive error types for parsing `.llre` files,
//! including file-level errors (I/O, circular imports) and parse-level errors
//! (syntax, undefined symbols).

use std::fmt;
use std::path::PathBuf;

// Re-export Position from common module
pub use crate::phonetic::common::Position;

/// Error type for `.llre` file parsing and loading.
#[derive(Debug, Clone)]
pub struct LLreError {
    /// The kind of error
    pub kind: LLreErrorKind,
    /// Position where the error occurred (if applicable)
    pub position: Option<Position>,
    /// File where the error occurred (if applicable)
    pub file: Option<PathBuf>,
    /// Additional context about the error
    pub context: Option<String>,
}

/// The kind of `.llre` error.
#[derive(Debug, Clone, PartialEq)]
pub enum LLreErrorKind {
    // ==================== I/O Errors ====================
    /// File not found
    FileNotFound(String),

    /// Permission denied
    PermissionDenied(String),

    /// General I/O error
    IoError(String),

    // ==================== Import Errors ====================
    /// Circular import detected
    CircularImport(PathBuf),

    /// Import depth exceeded
    ImportDepthExceeded {
        max: usize,
        path: PathBuf,
    },

    /// Import file not found
    ImportNotFound {
        path: String,
        search_paths: Vec<PathBuf>,
    },

    /// Import resolution failed
    ImportResolutionFailed {
        path: String,
        reason: String,
    },

    // ==================== Lexer/Parse Errors ====================
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

    /// Expected a specific token
    ExpectedToken {
        expected: String,
        found: String,
    },

    // ==================== Directive Errors ====================
    /// Invalid directive
    InvalidDirective(String),

    /// Unknown directive name
    UnknownDirective(String),

    /// Duplicate directive (e.g., multiple @name)
    DuplicateDirective(String),

    /// Invalid directive value
    InvalidDirectiveValue {
        directive: String,
        value: String,
        reason: String,
    },

    // ==================== Pattern Errors ====================
    /// Missing pattern (empty file)
    MissingPattern,

    /// Multiple patterns (only one pattern per file)
    MultiplePatterns,

    /// Invalid regex pattern
    InvalidPattern(String),

    /// Pattern parsing error (delegates to regex parser)
    PatternParseError(String),

    // ==================== Flag Errors ====================
    /// Invalid flag name
    InvalidFlag(String),

    /// Duplicate flag
    DuplicateFlag(String),

    /// Conflicting flags
    ConflictingFlags {
        flag1: String,
        flag2: String,
    },

    // ==================== Symbol/Import Errors ====================
    /// Undefined symbol reference
    UndefinedSymbol {
        name: String,
        available: Vec<String>,
    },

    /// Symbol type mismatch
    SymbolTypeMismatch {
        name: String,
        expected: String,
        found: String,
    },

    /// Alias conflict (two imports with same alias)
    AliasConflict {
        alias: String,
        path1: String,
        path2: String,
    },

    /// Cyclic pattern reference detected during symbol expansion
    CyclicPatternReference {
        /// The pattern name that completes the cycle
        name: String,
        /// The chain of pattern references that form the cycle
        chain: Vec<String>,
    },

    // ==================== Compilation Errors ====================
    /// NFA compilation failed
    NfaCompilationFailed(String),

    /// Pattern too complex
    PatternTooComplex {
        size: usize,
        max: usize,
    },

    /// Recursion depth exceeded during compilation
    RecursionDepthExceeded {
        depth: usize,
        max: usize,
    },

    // ==================== Serialization Errors ====================
    /// Invalid binary format
    InvalidBinaryFormat(String),

    /// Version mismatch
    VersionMismatch {
        expected: u8,
        found: u8,
    },

    /// Serialization failed
    SerializationFailed(String),

    /// Deserialization failed
    DeserializationFailed(String),

    // ==================== Wrapped Errors ====================
    /// LLev error (from imported .llev files)
    LLevError(String),

    /// Regex parse error
    RegexParseError(String),
}

impl LLreError {
    /// Create a new error with the given kind.
    pub fn new(kind: LLreErrorKind) -> Self {
        Self {
            kind,
            position: None,
            file: None,
            context: None,
        }
    }

    /// Create an error with position.
    pub fn with_position(kind: LLreErrorKind, position: Position) -> Self {
        Self {
            kind,
            position: Some(position),
            file: None,
            context: None,
        }
    }

    /// Create an error with file path.
    pub fn with_file(kind: LLreErrorKind, file: impl Into<PathBuf>) -> Self {
        Self {
            kind,
            position: None,
            file: Some(file.into()),
            context: None,
        }
    }

    /// Create an error with both position and file.
    pub fn with_position_and_file(
        kind: LLreErrorKind,
        position: Position,
        file: impl Into<PathBuf>,
    ) -> Self {
        Self {
            kind,
            position: Some(position),
            file: Some(file.into()),
            context: None,
        }
    }

    /// Add context to an error.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Create a "file not found" error.
    pub fn file_not_found(path: impl Into<String>) -> Self {
        Self::new(LLreErrorKind::FileNotFound(path.into()))
    }

    /// Create an "unexpected EOF" error.
    pub fn unexpected_eof(position: Position) -> Self {
        Self::with_position(LLreErrorKind::UnexpectedEof, position)
    }

    /// Create an "unexpected character" error.
    pub fn unexpected_char(c: char, position: Position) -> Self {
        Self::with_position(LLreErrorKind::UnexpectedChar(c), position)
    }

    /// Create a "missing pattern" error.
    pub fn missing_pattern() -> Self {
        Self::new(LLreErrorKind::MissingPattern)
    }

    /// Create an "invalid pattern" error.
    pub fn invalid_pattern(reason: impl Into<String>, position: Position) -> Self {
        Self::with_position(LLreErrorKind::InvalidPattern(reason.into()), position)
    }

    /// Create from an LLev error.
    pub fn from_llev(err: &crate::phonetic::llev::LLevError) -> Self {
        Self {
            kind: LLreErrorKind::LLevError(err.to_string()),
            position: err.position,
            file: err.file.clone(),
            context: err.context.clone(),
        }
    }

    /// Create from a regex parse error.
    pub fn from_regex_parse(err: &crate::phonetic::regex::ParseError) -> Self {
        Self {
            kind: LLreErrorKind::RegexParseError(err.to_string()),
            position: Some(err.position),
            file: None,
            context: err.context.clone(),
        }
    }
}

impl fmt::Display for LLreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format location
        match (&self.file, &self.position) {
            (Some(file), Some(pos)) => {
                write!(f, "{}:{}: ", file.display(), pos)?;
            }
            (Some(file), None) => {
                write!(f, "{}: ", file.display())?;
            }
            (None, Some(pos)) => {
                write!(f, "{}: ", pos)?;
            }
            (None, None) => {}
        }

        // Format error kind
        write!(f, "{}", self.kind)?;

        // Add context if available
        if let Some(ref ctx) = self.context {
            write!(f, " (near '{}')", ctx)?;
        }

        Ok(())
    }
}

impl fmt::Display for LLreErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // I/O Errors
            LLreErrorKind::FileNotFound(path) => {
                write!(f, "file not found: {}", path)
            }
            LLreErrorKind::PermissionDenied(path) => {
                write!(f, "permission denied: {}", path)
            }
            LLreErrorKind::IoError(msg) => {
                write!(f, "I/O error: {}", msg)
            }

            // Import Errors
            LLreErrorKind::CircularImport(path) => {
                write!(f, "circular import detected: {}", path.display())
            }
            LLreErrorKind::ImportDepthExceeded { max, path } => {
                write!(
                    f,
                    "import depth exceeded (max {}) at: {}",
                    max,
                    path.display()
                )
            }
            LLreErrorKind::ImportNotFound { path, search_paths } => {
                write!(f, "import not found: '{}' (searched: ", path)?;
                for (i, sp) in search_paths.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", sp.display())?;
                }
                write!(f, ")")
            }
            LLreErrorKind::ImportResolutionFailed { path, reason } => {
                write!(f, "failed to resolve import '{}': {}", path, reason)
            }

            // Lexer/Parse Errors
            LLreErrorKind::UnexpectedEof => {
                write!(f, "unexpected end of input")
            }
            LLreErrorKind::UnexpectedChar(c) => {
                write!(f, "unexpected character '{}'", c)
            }
            LLreErrorKind::InvalidEscape(c) => {
                write!(f, "invalid escape sequence '\\{}'", c)
            }
            LLreErrorKind::UnterminatedString => {
                write!(f, "unterminated string literal")
            }
            LLreErrorKind::UnterminatedComment => {
                write!(f, "unterminated block comment")
            }
            LLreErrorKind::InvalidCodePoint(cp) => {
                write!(f, "invalid Unicode code point: U+{:04X}", cp)
            }
            LLreErrorKind::ExpectedToken { expected, found } => {
                write!(f, "expected {}, found {}", expected, found)
            }

            // Directive Errors
            LLreErrorKind::InvalidDirective(msg) => {
                write!(f, "invalid directive: {}", msg)
            }
            LLreErrorKind::UnknownDirective(name) => {
                write!(f, "unknown directive '@{}'", name)
            }
            LLreErrorKind::DuplicateDirective(name) => {
                write!(f, "duplicate '@{}' directive", name)
            }
            LLreErrorKind::InvalidDirectiveValue {
                directive,
                value,
                reason,
            } => {
                write!(
                    f,
                    "invalid value '{}' for @{}: {}",
                    value, directive, reason
                )
            }

            // Pattern Errors
            LLreErrorKind::MissingPattern => {
                write!(f, "missing regex pattern (each .llre file must contain exactly one pattern)")
            }
            LLreErrorKind::MultiplePatterns => {
                write!(f, "multiple patterns found (only one pattern allowed per .llre file)")
            }
            LLreErrorKind::InvalidPattern(msg) => {
                write!(f, "invalid pattern: {}", msg)
            }
            LLreErrorKind::PatternParseError(msg) => {
                write!(f, "pattern parse error: {}", msg)
            }

            // Flag Errors
            LLreErrorKind::InvalidFlag(flag) => {
                write!(f, "invalid flag '{}' (valid: multiline, dotall, case_insensitive)", flag)
            }
            LLreErrorKind::DuplicateFlag(flag) => {
                write!(f, "duplicate flag '{}'", flag)
            }
            LLreErrorKind::ConflictingFlags { flag1, flag2 } => {
                write!(f, "conflicting flags '{}' and '{}'", flag1, flag2)
            }

            // Symbol/Import Errors
            LLreErrorKind::UndefinedSymbol { name, available } => {
                if available.is_empty() {
                    write!(f, "undefined symbol '${}'", name)
                } else {
                    write!(
                        f,
                        "undefined symbol '${}'; available: {}",
                        name,
                        available.iter().map(|s| format!("${}", s)).collect::<Vec<_>>().join(", ")
                    )
                }
            }
            LLreErrorKind::SymbolTypeMismatch {
                name,
                expected,
                found,
            } => {
                write!(
                    f,
                    "symbol '{}' has wrong type: expected {}, found {}",
                    name, expected, found
                )
            }
            LLreErrorKind::AliasConflict {
                alias,
                path1,
                path2,
            } => {
                write!(
                    f,
                    "alias '{}' already used by '{}', cannot assign to '{}'",
                    alias, path1, path2
                )
            }
            LLreErrorKind::CyclicPatternReference { name, chain } => {
                write!(
                    f,
                    "cyclic pattern reference: '{}' forms a cycle (chain: {})",
                    name,
                    chain.join(" -> ")
                )
            }

            // Compilation Errors
            LLreErrorKind::NfaCompilationFailed(msg) => {
                write!(f, "NFA compilation failed: {}", msg)
            }
            LLreErrorKind::PatternTooComplex { size, max } => {
                write!(
                    f,
                    "pattern too complex: size {} exceeds maximum {}",
                    size, max
                )
            }
            LLreErrorKind::RecursionDepthExceeded { depth, max } => {
                write!(
                    f,
                    "recursion depth {} exceeded maximum {} during compilation",
                    depth, max
                )
            }

            // Serialization Errors
            LLreErrorKind::InvalidBinaryFormat(msg) => {
                write!(f, "invalid binary format: {}", msg)
            }
            LLreErrorKind::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "version mismatch: expected {}, found {}",
                    expected, found
                )
            }
            LLreErrorKind::SerializationFailed(msg) => {
                write!(f, "serialization failed: {}", msg)
            }
            LLreErrorKind::DeserializationFailed(msg) => {
                write!(f, "deserialization failed: {}", msg)
            }

            // Wrapped Errors
            LLreErrorKind::LLevError(msg) => {
                write!(f, "llev error: {}", msg)
            }
            LLreErrorKind::RegexParseError(msg) => {
                write!(f, "{}", msg)
            }
        }
    }
}

impl std::error::Error for LLreError {}

// Conversion from LLev errors
impl From<crate::phonetic::llev::LLevError> for LLreError {
    fn from(err: crate::phonetic::llev::LLevError) -> Self {
        Self::from_llev(&err)
    }
}

// Conversion from regex parse errors
impl From<crate::phonetic::regex::ParseError> for LLreError {
    fn from(err: crate::phonetic::regex::ParseError) -> Self {
        Self::from_regex_parse(&err)
    }
}

// Conversion from std::io::Error
impl From<std::io::Error> for LLreError {
    fn from(err: std::io::Error) -> Self {
        use std::io::ErrorKind;
        let kind = match err.kind() {
            ErrorKind::NotFound => LLreErrorKind::FileNotFound(err.to_string()),
            ErrorKind::PermissionDenied => LLreErrorKind::PermissionDenied(err.to_string()),
            _ => LLreErrorKind::IoError(err.to_string()),
        };
        Self::new(kind)
    }
}

/// Result type for `.llre` operations.
pub type LLreResult<T> = Result<T, LLreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LLreError::missing_pattern();
        assert!(err.to_string().contains("missing regex pattern"));
    }

    #[test]
    fn test_error_with_position() {
        let err = LLreError::unexpected_char('x', Position::new(2, 5, 15));
        assert!(err.to_string().contains("line 2"));
        assert!(err.to_string().contains("unexpected character 'x'"));
    }

    #[test]
    fn test_error_with_file() {
        let err = LLreError::with_file(
            LLreErrorKind::FileNotFound("test.llre".into()),
            "path/to/test.llre",
        );
        assert!(err.to_string().contains("path/to/test.llre"));
    }

    #[test]
    fn test_undefined_symbol_display() {
        let err = LLreError::new(LLreErrorKind::UndefinedSymbol {
            name: "VOWEL".into(),
            available: vec!["CONSONANT".into(), "DIGIT".into()],
        });
        let msg = err.to_string();
        assert!(msg.contains("$VOWEL"));
        assert!(msg.contains("$CONSONANT"));
        assert!(msg.contains("$DIGIT"));
    }
}
