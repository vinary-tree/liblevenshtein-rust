//! Error types for phonetic regex parsing.

use std::fmt;

// Re-export Position from common module for backward compatibility
pub use crate::phonetic::common::Position;

/// Error type for regex parsing.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    /// The kind of error
    pub kind: ParseErrorKind,
    /// Position where the error occurred
    pub position: Position,
    /// The problematic input (if available)
    pub context: Option<String>,
}

/// The kind of parse error.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    /// Unexpected end of input
    UnexpectedEof,

    /// Unexpected character
    UnexpectedChar(char),

    /// Expected a specific character
    ExpectedChar(char),

    /// Expected one of several characters
    ExpectedOneOf(Vec<char>),

    /// Invalid escape sequence
    InvalidEscape(char),

    /// Invalid character class
    InvalidCharClass(String),

    /// Invalid quantifier
    InvalidQuantifier(String),

    /// Invalid repetition bounds
    InvalidRepetition {
        min: usize,
        max: usize,
    },

    /// Invalid weight value
    InvalidWeight(String),

    /// Empty alternation branch
    EmptyAlternation,

    /// Empty character class
    EmptyCharClass,

    /// Unclosed group
    UnclosedGroup,

    /// Unclosed character class
    UnclosedCharClass,

    /// Unclosed quantifier
    UnclosedQuantifier,

    /// Invalid rewrite rule syntax
    InvalidRewriteRule(String),

    /// Invalid context syntax
    InvalidContext(String),

    /// Nested quantifiers are not allowed
    NestedQuantifier,

    /// Missing pattern before arrow in rewrite rule
    MissingPattern,

    /// Missing replacement after arrow in rewrite rule
    MissingReplacement,

    /// Invalid Unicode code point
    InvalidCodePoint(u32),

    /// Pattern too complex
    PatternTooComplex {
        size: usize,
        max: usize,
    },

    /// Unknown named character class
    UnknownNamedClass(String),

    /// Undefined symbol reference
    UndefinedSymbol {
        /// The symbol name that was not found
        name: String,
        /// Available symbols (for error message suggestions)
        available: Vec<String>,
    },

    /// Symbol type mismatch (e.g., using non-char-class symbol in char class)
    SymbolTypeMismatch {
        /// The symbol name
        name: String,
        /// Expected type
        expected: String,
        /// Actual type
        found: String,
    },

    /// Invalid group syntax in (?...)
    InvalidGroupSyntax(String),

    /// Invalid group name
    InvalidGroupName(String),

    /// Duplicate named group
    DuplicateGroupName(String),

    /// Invalid group reference
    InvalidGroupReference(String),

    /// Undefined group reference
    UndefinedGroupReference(String),

    /// Invalid flag syntax
    InvalidFlag(String),

    /// Recursion depth exceeded during NFA compilation
    RecursionDepthExceeded {
        depth: usize,
        max: usize,
    },

    /// Internal compilation error (should not occur in normal use)
    InternalError(String),
}

impl ParseError {
    /// Create a new parse error.
    pub fn new(kind: ParseErrorKind, position: Position) -> Self {
        Self {
            kind,
            position,
            context: None,
        }
    }

    /// Create a parse error with context.
    pub fn with_context(kind: ParseErrorKind, position: Position, context: impl Into<String>) -> Self {
        Self {
            kind,
            position,
            context: Some(context.into()),
        }
    }

    /// Create an "unexpected EOF" error.
    pub fn unexpected_eof(position: Position) -> Self {
        Self::new(ParseErrorKind::UnexpectedEof, position)
    }

    /// Create an "unexpected character" error.
    pub fn unexpected_char(c: char, position: Position) -> Self {
        Self::new(ParseErrorKind::UnexpectedChar(c), position)
    }

    /// Create an "expected character" error.
    pub fn expected_char(expected: char, position: Position) -> Self {
        Self::new(ParseErrorKind::ExpectedChar(expected), position)
    }

    /// Create an "invalid escape" error.
    pub fn invalid_escape(c: char, position: Position) -> Self {
        Self::new(ParseErrorKind::InvalidEscape(c), position)
    }

    /// Create an "unclosed group" error.
    pub fn unclosed_group(position: Position) -> Self {
        Self::new(ParseErrorKind::UnclosedGroup, position)
    }

    /// Create an "unclosed character class" error.
    pub fn unclosed_char_class(position: Position) -> Self {
        Self::new(ParseErrorKind::UnclosedCharClass, position)
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at {}: {}", self.position, self.kind)?;
        if let Some(ref ctx) = self.context {
            write!(f, " (near '{}')", ctx)?;
        }
        Ok(())
    }
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErrorKind::UnexpectedEof => {
                write!(f, "unexpected end of input")
            }
            ParseErrorKind::UnexpectedChar(c) => {
                write!(f, "unexpected character '{}'", c)
            }
            ParseErrorKind::ExpectedChar(c) => {
                write!(f, "expected '{}'", c)
            }
            ParseErrorKind::ExpectedOneOf(chars) => {
                let chars_str: Vec<String> = chars.iter().map(|c| format!("'{}'", c)).collect();
                write!(f, "expected one of: {}", chars_str.join(", "))
            }
            ParseErrorKind::InvalidEscape(c) => {
                write!(f, "invalid escape sequence '\\{}'", c)
            }
            ParseErrorKind::InvalidCharClass(msg) => {
                write!(f, "invalid character class: {}", msg)
            }
            ParseErrorKind::InvalidQuantifier(msg) => {
                write!(f, "invalid quantifier: {}", msg)
            }
            ParseErrorKind::InvalidRepetition { min, max } => {
                write!(
                    f,
                    "invalid repetition: min ({}) > max ({})",
                    min, max
                )
            }
            ParseErrorKind::InvalidWeight(msg) => {
                write!(f, "invalid weight: {}", msg)
            }
            ParseErrorKind::EmptyAlternation => {
                write!(f, "empty alternation branch")
            }
            ParseErrorKind::EmptyCharClass => {
                write!(f, "empty character class")
            }
            ParseErrorKind::UnclosedGroup => {
                write!(f, "unclosed group '('")
            }
            ParseErrorKind::UnclosedCharClass => {
                write!(f, "unclosed character class '['")
            }
            ParseErrorKind::UnclosedQuantifier => {
                write!(f, "unclosed quantifier '{{'")
            }
            ParseErrorKind::InvalidRewriteRule(msg) => {
                write!(f, "invalid rewrite rule: {}", msg)
            }
            ParseErrorKind::InvalidContext(msg) => {
                write!(f, "invalid context: {}", msg)
            }
            ParseErrorKind::NestedQuantifier => {
                write!(f, "nested quantifiers are not allowed")
            }
            ParseErrorKind::MissingPattern => {
                write!(f, "missing pattern before '->'")
            }
            ParseErrorKind::MissingReplacement => {
                write!(f, "missing replacement after '->'")
            }
            ParseErrorKind::InvalidCodePoint(cp) => {
                write!(f, "invalid Unicode code point: U+{:04X}", cp)
            }
            ParseErrorKind::PatternTooComplex { size, max } => {
                write!(
                    f,
                    "pattern too complex: size {} exceeds maximum {}",
                    size, max
                )
            }
            ParseErrorKind::UnknownNamedClass(name) => {
                write!(f, "unknown named character class '{}'", name)
            }
            ParseErrorKind::UndefinedSymbol { name, available } => {
                if available.is_empty() {
                    write!(f, "undefined symbol '${}'", name)
                } else {
                    write!(
                        f,
                        "undefined symbol '${}'; available symbols: {}",
                        name,
                        available.iter().map(|s| format!("${}", s)).collect::<Vec<_>>().join(", ")
                    )
                }
            }
            ParseErrorKind::SymbolTypeMismatch { name, expected, found } => {
                write!(
                    f,
                    "symbol '${}' has wrong type: expected {}, found {}",
                    name, expected, found
                )
            }
            ParseErrorKind::InvalidGroupSyntax(msg) => {
                write!(f, "invalid group syntax: {}", msg)
            }
            ParseErrorKind::InvalidGroupName(msg) => {
                write!(f, "invalid group name: {}", msg)
            }
            ParseErrorKind::DuplicateGroupName(name) => {
                write!(f, "duplicate named group '{}' (group names must be unique)", name)
            }
            ParseErrorKind::InvalidGroupReference(msg) => {
                write!(f, "invalid group reference: {}", msg)
            }
            ParseErrorKind::UndefinedGroupReference(name) => {
                write!(f, "undefined group reference '(?&{})' (group '{}' was never defined)", name, name)
            }
            ParseErrorKind::InvalidFlag(msg) => {
                write!(f, "invalid flag: {}", msg)
            }
            ParseErrorKind::RecursionDepthExceeded { depth, max } => {
                write!(
                    f,
                    "recursion depth {} exceeded maximum {} during group reference expansion",
                    depth, max
                )
            }
            ParseErrorKind::InternalError(msg) => {
                write!(f, "internal error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// Result type for regex parsing.
pub type ParseResult<T> = Result<T, ParseError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_display() {
        let err = ParseError::unexpected_char('x', Position::new(1, 5, 4));
        assert_eq!(
            err.to_string(),
            "parse error at line 1, column 5: unexpected character 'x'"
        );
    }

    #[test]
    fn test_parse_error_with_context() {
        let err = ParseError::with_context(
            ParseErrorKind::InvalidEscape('q'),
            Position::new(1, 3, 2),
            "\\q",
        );
        assert_eq!(
            err.to_string(),
            "parse error at line 1, column 3: invalid escape sequence '\\q' (near '\\q')"
        );
    }

    #[test]
    fn test_error_kinds() {
        assert_eq!(
            ParseErrorKind::UnexpectedEof.to_string(),
            "unexpected end of input"
        );
        assert_eq!(
            ParseErrorKind::UnclosedGroup.to_string(),
            "unclosed group '('"
        );
        assert_eq!(
            ParseErrorKind::InvalidRepetition { min: 5, max: 3 }.to_string(),
            "invalid repetition: min (5) > max (3)"
        );
    }
}
