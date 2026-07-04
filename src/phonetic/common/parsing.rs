//! Shared parsing functions for syllable and context expressions.
//!
//! This module provides free functions that implement shared parsing logic
//! for syllable expressions, context expressions, and grammar constructs.
//! These functions work with any parser that implements the appropriate traits.

use super::context::ContextExpr;
use super::syllable::SyllableExpr;
use super::traits::{ContextParser, LexerLike, SyllableParser, TokenLike};

// =============================================================================
// Syllable Expression Parsing
// =============================================================================

/// Parse a syllable expression (entry point).
///
/// Grammar: `syllable_expr ::= syllable_or`
pub fn parse_syllable_expr<P>(parser: &mut P) -> Result<SyllableExpr, P::Error>
where
    P: SyllableParser,
{
    parse_syllable_or(parser)
}

/// Parse a syllable OR expression.
///
/// Grammar: `syllable_or ::= syllable_and ("|" syllable_and)*`
pub fn parse_syllable_or<P>(parser: &mut P) -> Result<SyllableExpr, P::Error>
where
    P: SyllableParser,
{
    let mut left = parse_syllable_and(parser)?;

    loop {
        let is_pipe = match parser.lexer_mut().peek() {
            Ok(peek) => peek.is_pipe(),
            Err(e) => return Err(parser.map_lexer_error(e)),
        };

        if is_pipe {
            if let Err(e) = parser.lexer_mut().advance() {
                return Err(parser.map_lexer_error(e));
            }
            let right = parse_syllable_and(parser)?;
            left = SyllableExpr::or(left, right);
        } else {
            break;
        }
    }

    Ok(left)
}

/// Parse a syllable AND expression.
///
/// Grammar: `syllable_and ::= syllable_not ("&" syllable_not)*`
pub fn parse_syllable_and<P>(parser: &mut P) -> Result<SyllableExpr, P::Error>
where
    P: SyllableParser,
{
    let mut left = parse_syllable_not(parser)?;

    loop {
        let is_ampersand = match parser.lexer_mut().peek() {
            Ok(peek) => peek.is_ampersand(),
            Err(e) => return Err(parser.map_lexer_error(e)),
        };

        if is_ampersand {
            if let Err(e) = parser.lexer_mut().advance() {
                return Err(parser.map_lexer_error(e));
            }
            let right = parse_syllable_not(parser)?;
            left = SyllableExpr::and(left, right);
        } else {
            break;
        }
    }

    Ok(left)
}

/// Parse a syllable NOT expression.
///
/// Grammar: `syllable_not ::= "!" syllable_not | syllable_primary`
pub fn parse_syllable_not<P>(parser: &mut P) -> Result<SyllableExpr, P::Error>
where
    P: SyllableParser,
{
    let is_exclamation = match parser.lexer_mut().peek() {
        Ok(peek) => peek.is_exclamation(),
        Err(e) => return Err(parser.map_lexer_error(e)),
    };

    if is_exclamation {
        if let Err(e) = parser.lexer_mut().advance() {
            return Err(parser.map_lexer_error(e));
        }
        let inner = parse_syllable_not(parser)?;
        Ok(SyllableExpr::negate(inner))
    } else {
        parse_syllable_primary(parser)
    }
}

/// Parse a syllable primary expression.
///
/// Grammar: `syllable_primary ::= syllable_keyword | "(" syllable_expr ")"`
pub fn parse_syllable_primary<P>(parser: &mut P) -> Result<SyllableExpr, P::Error>
where
    P: SyllableParser,
{
    // Check for grouped expression
    let is_group_start = match parser.lexer_mut().peek() {
        Ok(peek) => peek.is_group_start(),
        Err(e) => return Err(parser.map_lexer_error(e)),
    };

    if is_group_start {
        if let Err(e) = parser.lexer_mut().advance() {
            return Err(parser.map_lexer_error(e));
        }
        let expr = parse_syllable_or(parser)?;

        // Expect closing paren
        let (is_group_end, found_token) = match parser.lexer_mut().peek() {
            Ok(peek) => (peek.is_group_end(), peek.clone()),
            Err(e) => return Err(parser.map_lexer_error(e)),
        };

        if !is_group_end {
            let position = parser.lexer_mut().position();
            return Err(parser.make_unexpected_token_error(")", &found_token, position));
        }
        if let Err(e) = parser.lexer_mut().advance() {
            return Err(parser.map_lexer_error(e));
        }
        return Ok(expr);
    }

    // Parse syllable keyword
    let token = match parser.lexer_mut().advance() {
        Ok(t) => t,
        Err(e) => return Err(parser.map_lexer_error(e)),
    };

    match token.as_syllable_condition() {
        Some(cond) => Ok(SyllableExpr::cond(cond)),
        None => {
            let position = parser.lexer_mut().position();
            Err(parser.make_unexpected_token_error(
                "syllable keyword (monosyllable, polysyllable, open_syllable, closed_syllable, final_syllable, initial_syllable)",
                &token,
                position,
            ))
        }
    }
}

// =============================================================================
// Context Expression Parsing
// =============================================================================

/// Parse a context expression (entry point).
///
/// Grammar: `context_expr ::= context_or`
pub fn parse_context_expr<P>(parser: &mut P) -> Result<ContextExpr<P::Pattern>, P::Error>
where
    P: ContextParser,
{
    parse_context_or(parser)
}

/// Parse a context OR expression.
///
/// Grammar: `context_or ::= context_and ("|" context_and)*`
pub fn parse_context_or<P>(parser: &mut P) -> Result<ContextExpr<P::Pattern>, P::Error>
where
    P: ContextParser,
{
    let mut left = parse_context_and(parser)?;

    loop {
        let is_pipe = match parser.lexer_mut().peek() {
            Ok(peek) => peek.is_pipe(),
            Err(e) => return Err(parser.map_lexer_error(e)),
        };

        if is_pipe {
            if let Err(e) = parser.lexer_mut().advance() {
                return Err(parser.map_lexer_error(e));
            }
            let right = parse_context_and(parser)?;
            left = ContextExpr::Or(Box::new(left), Box::new(right));
        } else {
            break;
        }
    }

    Ok(left)
}

/// Parse a context AND expression.
///
/// Grammar: `context_and ::= context_not ("&" context_not)*`
pub fn parse_context_and<P>(parser: &mut P) -> Result<ContextExpr<P::Pattern>, P::Error>
where
    P: ContextParser,
{
    let mut left = parse_context_not(parser)?;

    loop {
        let is_ampersand = match parser.lexer_mut().peek() {
            Ok(peek) => peek.is_ampersand(),
            Err(e) => return Err(parser.map_lexer_error(e)),
        };

        if is_ampersand {
            if let Err(e) = parser.lexer_mut().advance() {
                return Err(parser.map_lexer_error(e));
            }
            let right = parse_context_not(parser)?;
            left = ContextExpr::And(Box::new(left), Box::new(right));
        } else {
            break;
        }
    }

    Ok(left)
}

/// Parse a context NOT expression.
///
/// Grammar: `context_not ::= "!" context_not | context_primary`
pub fn parse_context_not<P>(parser: &mut P) -> Result<ContextExpr<P::Pattern>, P::Error>
where
    P: ContextParser,
{
    let is_exclamation = match parser.lexer_mut().peek() {
        Ok(peek) => peek.is_exclamation(),
        Err(e) => return Err(parser.map_lexer_error(e)),
    };

    if is_exclamation {
        if let Err(e) = parser.lexer_mut().advance() {
            return Err(parser.map_lexer_error(e));
        }
        let inner = parse_context_not(parser)?;
        Ok(ContextExpr::Not(Box::new(inner)))
    } else {
        parse_context_primary(parser)
    }
}

/// Parse a context primary expression.
///
/// Grammar: `context_primary ::= "#" | "(" context_expr ")" | pattern`
pub fn parse_context_primary<P>(parser: &mut P) -> Result<ContextExpr<P::Pattern>, P::Error>
where
    P: ContextParser,
{
    // Check for word boundary
    let is_hash = match parser.lexer_mut().peek() {
        Ok(peek) => peek.is_hash(),
        Err(e) => return Err(parser.map_lexer_error(e)),
    };

    if is_hash {
        if let Err(e) = parser.lexer_mut().advance() {
            return Err(parser.map_lexer_error(e));
        }
        return Ok(ContextExpr::WordBoundary);
    }

    // Check for grouped expression
    let is_group_start = match parser.lexer_mut().peek() {
        Ok(peek) => peek.is_group_start(),
        Err(e) => return Err(parser.map_lexer_error(e)),
    };

    if is_group_start {
        if let Err(e) = parser.lexer_mut().advance() {
            return Err(parser.map_lexer_error(e));
        }
        let expr = parse_context_or(parser)?;

        // Expect closing paren
        let (is_group_end, found_token) = match parser.lexer_mut().peek() {
            Ok(peek) => (peek.is_group_end(), peek.clone()),
            Err(e) => return Err(parser.map_lexer_error(e)),
        };

        if !is_group_end {
            let position = parser.lexer_mut().position();
            return Err(parser.make_unexpected_token_error(")", &found_token, position));
        }
        if let Err(e) = parser.lexer_mut().advance() {
            return Err(parser.map_lexer_error(e));
        }
        return Ok(expr);
    }

    // Parse pattern
    let pattern = parser.parse_pattern_for_context()?;
    Ok(ContextExpr::Pattern(pattern))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::common::position::Position;
    use crate::phonetic::common::syllable::SyllableCondition;

    // Mock token for testing
    #[derive(Debug, Clone, PartialEq)]
    enum MockToken {
        Pipe,
        Ampersand,
        Exclamation,
        GroupStart,
        GroupEnd,
        Monosyllable,
        Polysyllable,
        OpenSyllable,
        FinalSyllable,
        Eof,
    }

    impl TokenLike for MockToken {
        fn is_pipe(&self) -> bool {
            matches!(self, MockToken::Pipe)
        }
        fn is_ampersand(&self) -> bool {
            matches!(self, MockToken::Ampersand)
        }
        fn is_exclamation(&self) -> bool {
            matches!(self, MockToken::Exclamation)
        }
        fn is_group_start(&self) -> bool {
            matches!(self, MockToken::GroupStart)
        }
        fn is_group_end(&self) -> bool {
            matches!(self, MockToken::GroupEnd)
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
            matches!(self, MockToken::Eof)
        }
        fn is_if_keyword(&self) -> bool {
            false
        }
        fn as_syllable_condition(&self) -> Option<SyllableCondition> {
            match self {
                MockToken::Monosyllable => Some(SyllableCondition::Monosyllable),
                MockToken::Polysyllable => Some(SyllableCondition::Polysyllable),
                MockToken::OpenSyllable => Some(SyllableCondition::OpenSyllable),
                MockToken::FinalSyllable => Some(SyllableCondition::FinalSyllable),
                _ => None,
            }
        }
        fn can_start_primary(&self) -> bool {
            matches!(self, MockToken::GroupStart)
        }
    }

    // Mock lexer for testing
    struct MockLexer {
        tokens: Vec<MockToken>,
        index: usize,
    }

    impl MockLexer {
        fn new(tokens: Vec<MockToken>) -> Self {
            Self { tokens, index: 0 }
        }
    }

    impl LexerLike for MockLexer {
        type Token = MockToken;
        type Error = String;

        fn peek(&mut self) -> Result<&Self::Token, Self::Error> {
            self.tokens
                .get(self.index)
                .ok_or_else(|| "unexpected end of input".to_string())
        }

        fn advance(&mut self) -> Result<Self::Token, Self::Error> {
            if self.index < self.tokens.len() {
                let token = self.tokens[self.index].clone();
                self.index += 1;
                Ok(token)
            } else {
                Err("unexpected end of input".to_string())
            }
        }

        fn position(&self) -> Position {
            Position::start()
        }
    }

    // Mock parser for testing syllable parsing
    struct MockSyllableParser {
        lexer: MockLexer,
    }

    impl MockSyllableParser {
        fn new(tokens: Vec<MockToken>) -> Self {
            Self {
                lexer: MockLexer::new(tokens),
            }
        }
    }

    impl SyllableParser for MockSyllableParser {
        type Lexer = MockLexer;
        type Error = String;

        fn lexer_mut(&mut self) -> &mut Self::Lexer {
            &mut self.lexer
        }

        fn make_unexpected_token_error(
            &self,
            expected: &str,
            found: &MockToken,
            _position: Position,
        ) -> String {
            format!("expected {}, found {:?}", expected, found)
        }

        fn map_lexer_error(&self, err: String) -> String {
            err
        }
    }

    #[test]
    fn test_parse_syllable_simple() {
        let mut parser = MockSyllableParser::new(vec![MockToken::Monosyllable, MockToken::Eof]);
        let result = parse_syllable_expr(&mut parser);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("test fixture: parse must be Ok"),
            SyllableExpr::Cond(SyllableCondition::Monosyllable)
        );
    }

    #[test]
    fn test_parse_syllable_or() {
        let mut parser = MockSyllableParser::new(vec![
            MockToken::Monosyllable,
            MockToken::Pipe,
            MockToken::Polysyllable,
            MockToken::Eof,
        ]);
        let result = parse_syllable_expr(&mut parser);
        assert!(result.is_ok());
        let expr = result.expect("test fixture: parse must be Ok");
        match expr {
            SyllableExpr::Or(left, right) => {
                assert_eq!(*left, SyllableExpr::Cond(SyllableCondition::Monosyllable));
                assert_eq!(*right, SyllableExpr::Cond(SyllableCondition::Polysyllable));
            }
            _ => panic!("expected Or expression"),
        }
    }

    #[test]
    fn test_parse_syllable_and() {
        let mut parser = MockSyllableParser::new(vec![
            MockToken::OpenSyllable,
            MockToken::Ampersand,
            MockToken::FinalSyllable,
            MockToken::Eof,
        ]);
        let result = parse_syllable_expr(&mut parser);
        assert!(result.is_ok());
        let expr = result.expect("test fixture: parse must be Ok");
        match expr {
            SyllableExpr::And(left, right) => {
                assert_eq!(*left, SyllableExpr::Cond(SyllableCondition::OpenSyllable));
                assert_eq!(*right, SyllableExpr::Cond(SyllableCondition::FinalSyllable));
            }
            _ => panic!("expected And expression"),
        }
    }

    #[test]
    fn test_parse_syllable_not() {
        let mut parser = MockSyllableParser::new(vec![
            MockToken::Exclamation,
            MockToken::Monosyllable,
            MockToken::Eof,
        ]);
        let result = parse_syllable_expr(&mut parser);
        assert!(result.is_ok());
        let expr = result.expect("test fixture: parse must be Ok");
        match expr {
            SyllableExpr::Not(inner) => {
                assert_eq!(*inner, SyllableExpr::Cond(SyllableCondition::Monosyllable));
            }
            _ => panic!("expected Not expression"),
        }
    }

    #[test]
    fn test_parse_syllable_grouped() {
        let mut parser = MockSyllableParser::new(vec![
            MockToken::GroupStart,
            MockToken::Monosyllable,
            MockToken::Pipe,
            MockToken::Polysyllable,
            MockToken::GroupEnd,
            MockToken::Eof,
        ]);
        let result = parse_syllable_expr(&mut parser);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_syllable_complex() {
        // monosyllable | (open_syllable & final_syllable)
        let mut parser = MockSyllableParser::new(vec![
            MockToken::Monosyllable,
            MockToken::Pipe,
            MockToken::GroupStart,
            MockToken::OpenSyllable,
            MockToken::Ampersand,
            MockToken::FinalSyllable,
            MockToken::GroupEnd,
            MockToken::Eof,
        ]);
        let result = parse_syllable_expr(&mut parser);
        assert!(result.is_ok());
    }
}
