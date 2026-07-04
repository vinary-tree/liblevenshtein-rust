//! Byte-level (ASCII) recursive descent parser for phonetic regular expressions.
//!
//! This parser is a leaner duplicate of the char-level parser with no group
//! tracking, a separate lexer (`LexerByte`), and a separate AST type
//! (`RegexByte`).

use super::common::MAX_PATTERN_SIZE;
use crate::phonetic::common::traits::SyllableParser;
use crate::phonetic::nfa::types::CharClass;
use crate::phonetic::regex::ast::{
    ContextExprByte, ContextPredicateByte, RegexByte, SyllableCondition, SyllableExpr,
};
use crate::phonetic::regex::error::{ParseError, ParseErrorKind, ParseResult, Position};
use crate::phonetic::regex::lexer::{LexerByte, TokenByte};

/// Byte-level parser for ASCII patterns.
pub struct ParserByte<'a> {
    lexer: LexerByte<'a>,
}

impl<'a> ParserByte<'a> {
    /// Create a new byte-level parser.
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            lexer: LexerByte::new(input),
        }
    }

    /// Parse a complete regex pattern.
    pub fn parse(&mut self) -> ParseResult<RegexByte> {
        let result = self.parse_alternation()?;

        if !self.lexer.is_eof() {
            let token = self.lexer.next_token()?;
            if token != TokenByte::Eof {
                return Err(ParseError::unexpected_char(
                    self.token_to_char(&token),
                    self.lexer.position(),
                ));
            }
        }

        let size = result.size();
        if size > MAX_PATTERN_SIZE {
            return Err(ParseError::new(
                ParseErrorKind::PatternTooComplex {
                    size,
                    max: MAX_PATTERN_SIZE,
                },
                self.lexer.position(),
            ));
        }

        Ok(result)
    }

    /// Parse a rewrite rule.
    pub fn parse_rewrite_rule(&mut self) -> ParseResult<RegexByte> {
        let pattern = self.parse_alternation()?;

        let token = self.lexer.next_token()?;
        if token != TokenByte::Arrow {
            return Err(ParseError::with_context(
                ParseErrorKind::InvalidRewriteRule("expected '->'".to_string()),
                self.lexer.position(),
                format!("got {:?}", token),
            ));
        }

        let replacement = if self.is_at_context_or_weight_or_end()? {
            RegexByte::Empty
        } else {
            self.parse_alternation()?
        };

        let context = if self.lexer.peek()? == &TokenByte::Slash {
            self.lexer.next_token()?;
            Some(self.parse_context()?)
        } else {
            None
        };

        let weight = 0.0; // Simplified for byte-level

        Ok(RegexByte::rewrite_rule(
            pattern,
            replacement,
            context,
            weight,
        ))
    }

    /// Parse an alternation.
    fn parse_alternation(&mut self) -> ParseResult<RegexByte> {
        let mut left = self.parse_concatenation()?;

        while self.lexer.peek()? == &TokenByte::Pipe {
            self.lexer.next_token()?;
            let right = self.parse_concatenation()?;
            left = RegexByte::alt(left, right);
        }

        Ok(left)
    }

    /// Parse a concatenation.
    fn parse_concatenation(&mut self) -> ParseResult<RegexByte> {
        let mut result = self.parse_quantified()?;

        loop {
            let peek = self.lexer.peek()?;
            let can_continue = Self::can_start_primary_token(peek);
            if !can_continue {
                break;
            }

            let next = self.parse_quantified()?;
            result = RegexByte::concat(result, next);
        }

        Ok(result)
    }

    /// Parse a quantified expression.
    fn parse_quantified(&mut self) -> ParseResult<RegexByte> {
        let primary = self.parse_primary()?;

        let peek = self.lexer.peek()?;
        match peek {
            TokenByte::Star => {
                self.lexer.next_token()?;
                Ok(RegexByte::star(primary))
            }
            TokenByte::Plus => {
                self.lexer.next_token()?;
                Ok(RegexByte::plus(primary))
            }
            TokenByte::Question => {
                self.lexer.next_token()?;
                Ok(RegexByte::optional(primary))
            }
            TokenByte::QuantifierStart => {
                self.lexer.next_token()?;
                self.parse_repetition(primary)
            }
            _ => Ok(primary),
        }
    }

    /// Parse a repetition quantifier: `{n}`, `{n,}`, `{,m}`, `{n,m}`
    fn parse_repetition(&mut self, inner: RegexByte) -> ParseResult<RegexByte> {
        // Check for {,m} syntax (at most m, min defaults to 0)
        let peek = self.lexer.peek()?;
        if peek == &TokenByte::Comma {
            self.lexer.next_token()?; // consume ','
            let max = match self.lexer.next_token()? {
                TokenByte::Number(n) => n,
                _ => {
                    return Err(ParseError::new(
                        ParseErrorKind::InvalidQuantifier("expected number after ','".to_string()),
                        self.lexer.position(),
                    ))
                }
            };
            self.expect_token(TokenByte::QuantifierEnd)?;
            return Ok(RegexByte::repeat_range(inner, 0, Some(max)));
        }

        // Expect a number for {n}, {n,}, {n,m}
        let min = match self.lexer.next_token()? {
            TokenByte::Number(n) => n,
            token => {
                return Err(ParseError::with_context(
                    ParseErrorKind::InvalidQuantifier("expected number".to_string()),
                    self.lexer.position(),
                    format!("got {:?}", token),
                ))
            }
        };

        let peek = self.lexer.peek()?;
        match peek {
            TokenByte::QuantifierEnd => {
                self.lexer.next_token()?;
                Ok(RegexByte::repeat_exact(inner, min))
            }
            TokenByte::Comma => {
                self.lexer.next_token()?;

                let peek = self.lexer.peek()?;
                if peek == &TokenByte::QuantifierEnd {
                    self.lexer.next_token()?;
                    Ok(RegexByte::repeat_range(inner, min, None))
                } else if let TokenByte::Number(max) = self.lexer.next_token()? {
                    if max < min {
                        return Err(ParseError::new(
                            ParseErrorKind::InvalidRepetition { min, max },
                            self.lexer.position(),
                        ));
                    }
                    self.expect_token(TokenByte::QuantifierEnd)?;
                    Ok(RegexByte::repeat_range(inner, min, Some(max)))
                } else {
                    Err(ParseError::new(
                        ParseErrorKind::InvalidQuantifier("expected number or '}'".to_string()),
                        self.lexer.position(),
                    ))
                }
            }
            _ => Err(ParseError::new(
                ParseErrorKind::UnclosedQuantifier,
                self.lexer.position(),
            )),
        }
    }

    /// Parse a primary expression.
    fn parse_primary(&mut self) -> ParseResult<RegexByte> {
        let token = self.lexer.next_token()?;

        match token {
            TokenByte::GroupStart => {
                let inner = self.parse_alternation()?;
                self.expect_token(TokenByte::GroupEnd)?;
                Ok(RegexByte::non_capturing_group(inner))
            }
            TokenByte::ByteClassStart => self.parse_byte_class(),
            TokenByte::Dot => Ok(RegexByte::any()),
            TokenByte::Hash => Ok(RegexByte::word_boundary()),
            TokenByte::Byte(b) => Ok(RegexByte::byte(b)),
            TokenByte::Eof => Err(ParseError::unexpected_eof(self.lexer.position())),
            _ => Err(ParseError::unexpected_char(
                self.token_to_char(&token),
                self.lexer.position(),
            )),
        }
    }

    /// Parse a byte class.
    fn parse_byte_class(&mut self) -> ParseResult<RegexByte> {
        let mut bytes = Vec::new();
        let mut negated = false;

        if self.lexer.peek()? == &TokenByte::Caret {
            self.lexer.next_token()?;
            negated = true;
        }

        loop {
            let token = self.lexer.next_token()?;

            match token {
                TokenByte::ByteClassEnd => break,
                TokenByte::Byte(b) => {
                    if self.lexer.peek()? == &TokenByte::Dash {
                        self.lexer.next_token()?;
                        let end_token = self.lexer.next_token()?;
                        if let TokenByte::Byte(end) = end_token {
                            for byte in b..=end {
                                bytes.push(byte);
                            }
                        } else if end_token == TokenByte::ByteClassEnd {
                            bytes.push(b);
                            bytes.push(b'-');
                            break;
                        } else {
                            return Err(ParseError::unexpected_char(
                                self.token_to_char(&end_token),
                                self.lexer.position(),
                            ));
                        }
                    } else {
                        bytes.push(b);
                    }
                }
                TokenByte::Dash => {
                    bytes.push(b'-');
                }
                TokenByte::Eof => {
                    return Err(ParseError::unclosed_char_class(self.lexer.position()));
                }
                _ => {
                    return Err(ParseError::unexpected_char(
                        self.token_to_char(&token),
                        self.lexer.position(),
                    ));
                }
            }
        }

        if bytes.is_empty() {
            return Err(ParseError::new(
                ParseErrorKind::EmptyCharClass,
                self.lexer.position(),
            ));
        }

        let class = if negated {
            CharClass::from_bytes(&bytes).negated()
        } else {
            CharClass::from_bytes(&bytes)
        };

        Ok(RegexByte::byte_class(class))
    }

    /// Parse a context predicate: `left_context? _ right_context? syllable_clause?`
    fn parse_context(&mut self) -> ParseResult<ContextPredicateByte> {
        let mut left = None;
        let mut right = None;

        let peek = self.lexer.peek()?;
        if peek != &TokenByte::Underscore {
            left = Some(self.parse_context_expr()?);
        }

        self.expect_token(TokenByte::Underscore)?;

        if self.can_start_context_expr()? {
            right = Some(self.parse_context_expr()?);
        }

        // Parse optional syllable clause
        let syllable = self.parse_syllable_clause()?;

        Ok(ContextPredicateByte::new_with_exprs(left, right, syllable))
    }

    /// Parse a context expression with logical operators.
    /// Precedence: NOT > AND > OR
    fn parse_context_expr(&mut self) -> ParseResult<ContextExprByte> {
        self.parse_context_or()
    }

    /// Parse OR-level context expression: `a | b`
    fn parse_context_or(&mut self) -> ParseResult<ContextExprByte> {
        let mut left = self.parse_context_and()?;

        while self.lexer.peek()? == &TokenByte::Pipe {
            self.lexer.next_token()?; // consume '|'
            let right = self.parse_context_and()?;
            left = ContextExprByte::or(left, right);
        }

        Ok(left)
    }

    /// Parse AND-level context expression: `a & b`
    fn parse_context_and(&mut self) -> ParseResult<ContextExprByte> {
        let mut left = self.parse_context_not()?;

        while self.lexer.peek()? == &TokenByte::Ampersand {
            self.lexer.next_token()?; // consume '&'
            let right = self.parse_context_not()?;
            left = ContextExprByte::and(left, right);
        }

        Ok(left)
    }

    /// Parse NOT-level context expression: `!a`
    fn parse_context_not(&mut self) -> ParseResult<ContextExprByte> {
        if self.lexer.peek()? == &TokenByte::Exclamation {
            self.lexer.next_token()?; // consume '!'
            let inner = self.parse_context_not()?;
            Ok(ContextExprByte::negate(inner))
        } else {
            self.parse_context_primary()
        }
    }

    /// Parse a primary context expression.
    fn parse_context_primary(&mut self) -> ParseResult<ContextExprByte> {
        let peek = self.lexer.peek()?;

        match peek {
            TokenByte::Hash => {
                self.lexer.next_token()?;
                Ok(ContextExprByte::word_boundary())
            }
            TokenByte::ByteClassStart => {
                self.lexer.next_token()?;
                let regex = self.parse_byte_class()?;
                Ok(ContextExprByte::pattern(regex))
            }
            TokenByte::Byte(_) | TokenByte::Dot => {
                let token = self.lexer.next_token()?;
                let regex = match token {
                    TokenByte::Byte(b) => RegexByte::byte(b),
                    TokenByte::Dot => RegexByte::any(),
                    _ => {
                        return Err(ParseError::new(
                            ParseErrorKind::InvalidContext("expected pattern".to_string()),
                            self.lexer.position(),
                        ));
                    }
                };
                Ok(ContextExprByte::pattern(regex))
            }
            TokenByte::GroupStart => {
                self.lexer.next_token()?; // consume '('
                let inner = self.parse_context_expr()?;
                self.expect_token(TokenByte::GroupEnd)?;
                Ok(inner)
            }
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidContext("expected pattern".to_string()),
                self.lexer.position(),
            )),
        }
    }

    /// Parse an optional syllable clause.
    fn parse_syllable_clause(&mut self) -> ParseResult<Option<SyllableExpr>> {
        if self.lexer.peek()? != &TokenByte::IfKeyword {
            return Ok(None);
        }

        self.lexer.next_token()?; // consume 'if'
        let expr = self.parse_syllable_expr()?;
        Ok(Some(expr))
    }

    /// Parse a syllable expression with logical operators.
    fn parse_syllable_expr(&mut self) -> ParseResult<SyllableExpr> {
        self.parse_syllable_or()
    }

    /// Parse OR-level syllable expression.
    fn parse_syllable_or(&mut self) -> ParseResult<SyllableExpr> {
        let mut left = self.parse_syllable_and()?;

        while self.lexer.peek()? == &TokenByte::Pipe {
            self.lexer.next_token()?;
            let right = self.parse_syllable_and()?;
            left = SyllableExpr::or(left, right);
        }

        Ok(left)
    }

    /// Parse AND-level syllable expression.
    fn parse_syllable_and(&mut self) -> ParseResult<SyllableExpr> {
        let mut left = self.parse_syllable_not()?;

        while self.lexer.peek()? == &TokenByte::Ampersand {
            self.lexer.next_token()?;
            let right = self.parse_syllable_not()?;
            left = SyllableExpr::and(left, right);
        }

        Ok(left)
    }

    /// Parse NOT-level syllable expression.
    fn parse_syllable_not(&mut self) -> ParseResult<SyllableExpr> {
        if self.lexer.peek()? == &TokenByte::Exclamation {
            self.lexer.next_token()?;
            let inner = self.parse_syllable_not()?;
            Ok(SyllableExpr::negate(inner))
        } else {
            self.parse_syllable_primary()
        }
    }

    /// Parse a primary syllable expression.
    fn parse_syllable_primary(&mut self) -> ParseResult<SyllableExpr> {
        let token = self.lexer.next_token()?;

        match token {
            TokenByte::Monosyllable => Ok(SyllableExpr::cond(SyllableCondition::Monosyllable)),
            TokenByte::Polysyllable => Ok(SyllableExpr::cond(SyllableCondition::Polysyllable)),
            TokenByte::OpenSyllable => Ok(SyllableExpr::cond(SyllableCondition::OpenSyllable)),
            TokenByte::ClosedSyllable => Ok(SyllableExpr::cond(SyllableCondition::ClosedSyllable)),
            TokenByte::FinalSyllable => Ok(SyllableExpr::cond(SyllableCondition::FinalSyllable)),
            TokenByte::InitialSyllable => {
                Ok(SyllableExpr::cond(SyllableCondition::InitialSyllable))
            }
            TokenByte::GroupStart => {
                let inner = self.parse_syllable_expr()?;
                self.expect_token(TokenByte::GroupEnd)?;
                Ok(inner)
            }
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidContext(format!(
                    "expected syllable condition, got {:?}",
                    token
                )),
                self.lexer.position(),
            )),
        }
    }

    /// Check if we can start a context expression.
    fn can_start_context_expr(&mut self) -> ParseResult<bool> {
        let peek = self.lexer.peek()?;
        Ok(matches!(
            peek,
            TokenByte::Hash
                | TokenByte::ByteClassStart
                | TokenByte::Byte(_)
                | TokenByte::Dot
                | TokenByte::GroupStart
                | TokenByte::Exclamation
        ))
    }

    fn is_at_context_or_weight_or_end(&mut self) -> ParseResult<bool> {
        let peek = self.lexer.peek()?;
        Ok(matches!(
            peek,
            TokenByte::Slash | TokenByte::ByteClassStart | TokenByte::Eof
        ))
    }

    fn can_start_primary_token(token: &TokenByte) -> bool {
        matches!(
            token,
            TokenByte::Byte(_)
                | TokenByte::ByteClassStart
                | TokenByte::GroupStart
                | TokenByte::Dot
                | TokenByte::Hash
        )
    }

    fn expect_token(&mut self, expected: TokenByte) -> ParseResult<()> {
        let token = self.lexer.next_token()?;
        if token == expected {
            Ok(())
        } else {
            Err(ParseError::with_context(
                ParseErrorKind::ExpectedChar(self.token_to_char(&expected)),
                self.lexer.position(),
                format!("got {:?}", token),
            ))
        }
    }

    fn token_to_char(&self, token: &TokenByte) -> char {
        match token {
            TokenByte::Byte(b) => *b as char,
            TokenByte::ByteClassStart => '[',
            TokenByte::ByteClassEnd => ']',
            TokenByte::Caret => '^',
            TokenByte::Dash => '-',
            TokenByte::GroupStart => '(',
            TokenByte::GroupEnd => ')',
            TokenByte::Pipe => '|',
            TokenByte::Star => '*',
            TokenByte::Plus => '+',
            TokenByte::Question => '?',
            TokenByte::Dot => '.',
            TokenByte::QuantifierStart => '{',
            TokenByte::QuantifierEnd => '}',
            TokenByte::Comma => ',',
            TokenByte::Number(_) => '0',
            TokenByte::Float(_) => '0',
            TokenByte::Arrow => '>',
            TokenByte::Slash => '/',
            TokenByte::Underscore => '_',
            TokenByte::Hash => '#',
            TokenByte::WeightStart => '[',
            TokenByte::WeightEnd => ']',
            TokenByte::Ampersand => '&',
            TokenByte::Exclamation => '!',
            TokenByte::IfKeyword => 'i',
            TokenByte::Monosyllable => 'm',
            TokenByte::Polysyllable => 'p',
            TokenByte::OpenSyllable => 'o',
            TokenByte::ClosedSyllable => 'c',
            TokenByte::FinalSyllable => 'f',
            TokenByte::InitialSyllable => 'i',
            TokenByte::Eof => '\0',
        }
    }
}

impl<'a> SyllableParser for ParserByte<'a> {
    type Lexer = LexerByte<'a>;
    type Error = ParseError;

    fn lexer_mut(&mut self) -> &mut Self::Lexer {
        &mut self.lexer
    }

    fn make_unexpected_token_error(
        &self,
        expected: &str,
        found: &TokenByte,
        position: Position,
    ) -> Self::Error {
        ParseError::new(
            ParseErrorKind::InvalidContext(format!("expected {}, got {:?}", expected, found)),
            position,
        )
    }

    fn map_lexer_error(&self, err: ParseError) -> Self::Error {
        err
    }
}
