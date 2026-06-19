//! Lexical helpers and `SyllableParser` trait impl.
//!
//! These are the low-level token-stream utilities used by every other
//! parser sub-module: advance/check/expect_*/skip_terminator, plus the
//! `SyllableParser` trait implementation used by shared syllable logic.

use crate::phonetic::common::traits::SyllableParser;

use super::super::error::{LLevError, LLevErrorKind, LLevResult, Position};
use super::super::lexer::{Lexer, Token};
use super::Parser;

impl<'a> Parser<'a> {
    // ==================== Helper Methods ====================

    /// Advance and return the next token.
    pub(super) fn advance(&mut self) -> LLevResult<Token> {
        self.lexer.next_token()
    }

    /// Check if the next token matches (without consuming).
    pub(super) fn check(&mut self, expected: &Token) -> bool {
        self.lexer.peek().ok() == Some(expected)
    }

    /// Check whether the next raw token sequence is an inline rule weight suffix.
    pub(super) fn next_token_is_weight_suffix(&mut self) -> bool {
        matches!(self.lexer.peek().ok(), Some(Token::CharClassStart))
            && self.peek_weight_suffix().is_some()
    }

    /// Return the numeric value of an upcoming inline rule weight suffix, if present.
    pub(super) fn peek_weight_suffix(&mut self) -> Option<f64> {
        let remaining = self.lexer.remaining_input();
        let Some(after_open) = remaining.strip_prefix('[') else {
            return None;
        };
        let close = after_open.find(']')?;
        let value = after_open[..close].trim();
        if !is_inline_weight_literal(value) {
            return None;
        }

        value
            .parse::<f64>()
            .ok()
            .filter(|weight| weight.is_finite() && *weight >= 0.0)
    }

    /// Expect and consume a specific token.
    pub(super) fn expect(&mut self, expected: &Token) -> LLevResult<Token> {
        let token = self.advance()?;
        if &token == expected {
            Ok(token)
        } else {
            Err(LLevError::expected_token(
                format!("{:?}", expected),
                format!("{:?}", token),
                self.lexer.position(),
            ))
        }
    }

    /// Expect and consume a string literal.
    pub(super) fn expect_string(&mut self) -> LLevResult<String> {
        match self.advance()? {
            Token::String(s) => Ok(s),
            other => Err(LLevError::expected_token(
                "string".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Expect and consume an identifier.
    pub(super) fn expect_identifier(&mut self) -> LLevResult<String> {
        match self.advance()? {
            Token::Identifier(s) => Ok(s),
            other => Err(LLevError::expected_token(
                "identifier".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Expect and consume an identifier or string literal.
    ///
    /// This is useful for metadata fields like `group` where users might
    /// write either `group: orthography` or `group: "orthography"`.
    pub(super) fn expect_identifier_or_string(&mut self) -> LLevResult<String> {
        match self.advance()? {
            Token::Identifier(s) => Ok(s),
            Token::String(s) => Ok(s),
            other => Err(LLevError::expected_token(
                "identifier or string".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Expect and consume a number.
    /// In Pattern mode, digits are Char tokens - collect them into a number.
    pub(super) fn expect_number(&mut self) -> LLevResult<usize> {
        match self.advance()? {
            Token::Number(n) => Ok(n),
            // In Pattern mode, digits are Char tokens - collect them into a number
            Token::Char(c) if c.is_ascii_digit() => {
                let mut num_str = String::new();
                num_str.push(c);
                // Collect consecutive digit chars
                while let Ok(Token::Char(next_c)) = self.lexer.peek() {
                    if next_c.is_ascii_digit() {
                        num_str.push(*next_c);
                        self.advance()?;
                    } else {
                        break;
                    }
                }
                num_str.parse::<usize>().map_err(|_| {
                    LLevError::expected_token(
                        "number".to_string(),
                        format!("'{}'", num_str),
                        self.lexer.position(),
                    )
                })
            }
            other => Err(LLevError::expected_token(
                "number".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Expect and consume a float or number (as f64).
    pub(super) fn expect_float_or_number(&mut self) -> LLevResult<f64> {
        match self.advance()? {
            Token::Float(f) => Ok(f),
            Token::Number(n) => Ok(n as f64),
            other => Err(LLevError::expected_token(
                "number or float".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Expect and consume a boolean (identifier "true" or "false").
    pub(super) fn expect_bool(&mut self) -> LLevResult<bool> {
        match self.advance()? {
            Token::Identifier(s) if s == "true" => Ok(true),
            Token::Identifier(s) if s == "false" => Ok(false),
            other => Err(LLevError::expected_token(
                "true or false".to_string(),
                format!("{:?}", other),
                self.lexer.position(),
            )),
        }
    }

    /// Skip optional terminator (semicolon or newline).
    pub(super) fn skip_terminator(&mut self) {
        // Use raw lookahead to avoid consuming characters in TopLevel mode
        loop {
            let remaining = self.lexer.remaining_input();
            if remaining.starts_with(';') || remaining.starts_with('\n') {
                // We're in TopLevel mode, so advance will correctly handle these
                self.lexer.enter_top_level();
                let _ = self.advance();
            } else {
                break;
            }
        }
    }
}

fn is_inline_weight_literal(value: &str) -> bool {
    let mut chars = value.chars().peekable();
    let mut saw_digit = false;

    while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
        saw_digit = true;
        chars.next();
    }

    if chars.peek() == Some(&'.') {
        chars.next();
        while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
            saw_digit = true;
            chars.next();
        }
    }

    saw_digit && chars.next().is_none()
}

impl<'a> SyllableParser for Parser<'a> {
    type Lexer = Lexer<'a>;
    type Error = LLevError;

    fn lexer_mut(&mut self) -> &mut Self::Lexer {
        &mut self.lexer
    }

    fn make_unexpected_token_error(
        &self,
        expected: &str,
        found: &Token,
        position: Position,
    ) -> Self::Error {
        LLevError::with_position(
            LLevErrorKind::InvalidPattern(format!("expected {}, got {:?}", expected, found)),
            position,
        )
    }

    fn from_lexer_error(&self, err: LLevError) -> Self::Error {
        err
    }
}
