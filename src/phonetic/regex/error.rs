//! Error types for phonetic regex parsing.

use std::fmt;

/// Position in the input string where an error occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
    /// Byte offset in the input
    pub offset: usize,
}

impl Position {
    /// Create a new position.
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }

    /// Create a position at the start of input.
    pub fn start() -> Self {
        Self {
            line: 1,
            column: 1,
            offset: 0,
        }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

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
    fn test_position_display() {
        let pos = Position::new(5, 10, 42);
        assert_eq!(pos.to_string(), "line 5, column 10");
    }

    #[test]
    fn test_position_start() {
        let pos = Position::start();
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 1);
        assert_eq!(pos.offset, 0);
    }

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
