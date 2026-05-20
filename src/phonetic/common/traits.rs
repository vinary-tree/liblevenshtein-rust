//! Trait hierarchy for shared parsing infrastructure.
//!
//! This module defines traits that abstract over the differences between
//! the regex and llev parsers, enabling shared parsing logic for syllable
//! expressions, context expressions, and grammar constructs.

use super::position::Position;
use super::syllable::SyllableCondition;

/// Trait for token types that can be used with shared parsing infrastructure.
///
/// This trait abstracts over the differences between regex `Token` and llev `Token`,
/// allowing shared parsing logic to work with both.
pub trait TokenLike: Clone + PartialEq + std::fmt::Debug {
    /// Check if this is a pipe `|` token (alternation).
    fn is_pipe(&self) -> bool;

    /// Check if this is an ampersand `&` token (context AND).
    fn is_ampersand(&self) -> bool;

    /// Check if this is an exclamation/bang `!` token (context NOT).
    fn is_exclamation(&self) -> bool;

    /// Check if this is a group start `(` token.
    fn is_group_start(&self) -> bool;

    /// Check if this is a group end `)` token.
    fn is_group_end(&self) -> bool;

    /// Check if this is a hash `#` token (word boundary).
    fn is_hash(&self) -> bool;

    /// Check if this is a star `*` token (zero or more).
    fn is_star(&self) -> bool;

    /// Check if this is a plus `+` token (one or more).
    fn is_plus(&self) -> bool;

    /// Check if this is a question `?` token (optional).
    fn is_question(&self) -> bool;

    /// Check if this is a brace/quantifier start `{` token.
    fn is_brace_start(&self) -> bool;

    /// Check if this is an EOF token.
    fn is_eof(&self) -> bool;

    /// Check if this is an "if" keyword token.
    fn is_if_keyword(&self) -> bool;

    /// Try to interpret this token as a syllable condition.
    ///
    /// Returns `Some(condition)` if this token represents a syllable keyword
    /// (monosyllable, polysyllable, etc.), `None` otherwise.
    ///
    /// # Implementation Notes
    ///
    /// - Regex tokens: match against dedicated token variants like `Token::Monosyllable`
    /// - LLev tokens: match against `Token::Identifier(name)` and parse the string
    fn as_syllable_condition(&self) -> Option<SyllableCondition>;

    /// Check if this token can start a primary expression.
    ///
    /// This is parser-specific and should return true for tokens like:
    /// - Char literals
    /// - Character class starts `[`
    /// - Group starts `(`
    /// - Dot `.`
    /// - Hash `#` (word boundary)
    /// - Symbol references
    fn can_start_primary(&self) -> bool;
}

/// Trait for lexers that can be used with shared parsing infrastructure.
///
/// This trait abstracts over the differences between regex `Lexer` and llev `Lexer`.
pub trait LexerLike {
    /// The token type produced by this lexer.
    type Token: TokenLike;

    /// The error type returned by this lexer.
    type Error;

    /// Peek at the next token without consuming it.
    fn peek(&mut self) -> Result<&Self::Token, Self::Error>;

    /// Advance to the next token and return it.
    fn advance(&mut self) -> Result<Self::Token, Self::Error>;

    /// Get the current position in the input.
    fn position(&self) -> Position;
}

/// Trait for parsers that support syllable expression parsing.
///
/// This trait provides default implementations for syllable expression parsing
/// (OR, AND, NOT, primary) that work with any `LexerLike` implementation.
pub trait SyllableParser {
    /// The lexer type used by this parser.
    type Lexer: LexerLike;

    /// The error type returned by this parser.
    type Error;

    /// Get a mutable reference to the lexer.
    fn lexer_mut(&mut self) -> &mut Self::Lexer;

    /// Create an error for unexpected tokens.
    fn make_unexpected_token_error(
        &self,
        expected: &str,
        found: &<Self::Lexer as LexerLike>::Token,
        position: Position,
    ) -> Self::Error;

    /// Create an error from a lexer error.
    fn from_lexer_error(&self, err: <Self::Lexer as LexerLike>::Error) -> Self::Error;
}

/// Trait for parsers that support context expression parsing.
///
/// This trait provides default implementations for context expression parsing
/// (OR, AND, NOT, primary) that work with any `LexerLike` implementation.
pub trait ContextParser: SyllableParser {
    /// The pattern type used in context expressions.
    type Pattern;

    /// Parse a pattern for use in a context expression.
    fn parse_pattern_for_context(&mut self) -> Result<Self::Pattern, Self::Error>;

    /// Check if the current token can start a context expression.
    fn can_start_context_expr(&mut self) -> Result<bool, Self::Error>;
}

/// Trait for parsers that support grammar/expression parsing.
///
/// This trait provides hooks for building pattern AST nodes that work with
/// the shared parsing logic for alternation, concatenation, and quantifiers.
pub trait GrammarParser: SyllableParser {
    /// The pattern type produced by this parser.
    type Pattern;

    /// Parse a primary expression (char, char class, group, dot, etc.).
    fn parse_primary(&mut self) -> Result<Self::Pattern, Self::Error>;

    /// Check if the current token can start a primary expression.
    fn can_start_primary(&mut self) -> Result<bool, Self::Error>;

    /// Build an empty pattern (epsilon).
    fn build_empty() -> Self::Pattern;

    /// Build an alternation pattern.
    fn build_alternation(left: Self::Pattern, right: Self::Pattern) -> Self::Pattern;

    /// Build a concatenation pattern.
    fn build_concat(left: Self::Pattern, right: Self::Pattern) -> Self::Pattern;

    /// Build a Kleene star pattern.
    fn build_star(inner: Self::Pattern) -> Self::Pattern;

    /// Build a Kleene plus pattern.
    fn build_plus(inner: Self::Pattern) -> Self::Pattern;

    /// Build an optional pattern.
    fn build_optional(inner: Self::Pattern) -> Self::Pattern;

    /// Build a repetition pattern with bounds.
    fn build_repetition(inner: Self::Pattern, min: usize, max: Option<usize>) -> Self::Pattern;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock token for testing
    #[derive(Debug, Clone, PartialEq)]
    enum MockToken {
        Pipe,
        Monosyllable,
        Char(char),
    }

    impl TokenLike for MockToken {
        fn is_pipe(&self) -> bool {
            matches!(self, MockToken::Pipe)
        }

        fn is_ampersand(&self) -> bool {
            false
        }

        fn is_exclamation(&self) -> bool {
            false
        }

        fn is_group_start(&self) -> bool {
            false
        }

        fn is_group_end(&self) -> bool {
            false
        }

        fn is_hash(&self) -> bool {
            false
        }

        fn is_star(&self) -> bool {
            false
        }

        fn is_plus(&self) -> bool {
            false
        }

        fn is_question(&self) -> bool {
            false
        }

        fn is_brace_start(&self) -> bool {
            false
        }

        fn is_eof(&self) -> bool {
            false
        }

        fn is_if_keyword(&self) -> bool {
            false
        }

        fn as_syllable_condition(&self) -> Option<SyllableCondition> {
            match self {
                MockToken::Monosyllable => Some(SyllableCondition::Monosyllable),
                _ => None,
            }
        }

        fn can_start_primary(&self) -> bool {
            matches!(self, MockToken::Char(_))
        }
    }

    #[test]
    fn test_mock_token_like() {
        let pipe = MockToken::Pipe;
        assert!(pipe.is_pipe());
        assert!(!pipe.is_ampersand());

        let mono = MockToken::Monosyllable;
        assert_eq!(mono.as_syllable_condition(), Some(SyllableCondition::Monosyllable));

        let char_tok = MockToken::Char('a');
        assert!(char_tok.can_start_primary());
        assert_eq!(char_tok.as_syllable_condition(), None);
    }
}
