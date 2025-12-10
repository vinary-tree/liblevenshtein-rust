//! Recursive descent parser for phonetic regular expressions.
//!
//! This parser implements the grammar defined in the AST module,
//! supporting both standard regex constructs and phonetic rewrite rules.
//!
//! # Grammar
//!
//! ```text
//! regex       ::= alternation
//! alternation ::= concatenation ('|' concatenation)*
//! concatenation ::= quantified+
//! quantified  ::= primary quantifier?
//! quantifier  ::= '*' | '+' | '?' | '{' number '}' | '{' number ',' number? '}'
//! primary     ::= '(' regex ')' | char_class | literal | '.' | '#'
//! char_class  ::= '[' '^'? (char | char '-' char)+ ']'
//!
//! rewrite_rule ::= pattern '->' replacement context? weight?
//! context      ::= '/' left_context? '_' right_context?
//! weight       ::= '[' float ']'
//! ```

use super::ast::{
    ContextExpr, ContextExprByte, ContextPredicate, ContextPredicateByte, Regex, RegexByte,
    SyllableCondition, SyllableExpr,
};
use super::error::{ParseError, ParseErrorKind, ParseResult};
use super::lexer::{Lexer, LexerByte, Token, TokenByte};
use crate::phonetic::nfa::types::{CharClass, CharClassChar};

/// Maximum complexity for parsed patterns (prevents DoS).
const MAX_PATTERN_SIZE: usize = 10_000;

/// Parser for phonetic regular expressions.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
}

impl<'a> Parser<'a> {
    /// Create a new parser for the given input.
    pub fn new(input: &'a str) -> Self {
        Self {
            lexer: Lexer::new(input),
        }
    }

    /// Parse a complete regex pattern.
    pub fn parse(&mut self) -> ParseResult<Regex> {
        let result = self.parse_alternation()?;

        // Check for trailing content
        if !self.lexer.is_eof() {
            let token = self.lexer.next_token()?;
            if token != Token::Eof {
                return Err(ParseError::unexpected_char(
                    self.token_to_char(&token),
                    self.lexer.position(),
                ));
            }
        }

        // Check complexity
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

    /// Parse a rewrite rule: `pattern -> replacement context? weight?`
    pub fn parse_rewrite_rule(&mut self) -> ParseResult<Regex> {
        let pattern = self.parse_alternation()?;

        // Expect arrow
        let token = self.lexer.next_token()?;
        if token != Token::Arrow {
            return Err(ParseError::with_context(
                ParseErrorKind::InvalidRewriteRule("expected '->'".to_string()),
                self.lexer.position(),
                format!("got {:?}", token),
            ));
        }

        // Parse replacement (can be empty)
        let replacement = if self.is_at_context_or_weight_or_end()? {
            Regex::Empty
        } else {
            self.parse_alternation()?
        };

        // Parse optional context
        let context = if self.lexer.peek()? == &Token::Slash {
            self.lexer.next_token()?; // consume '/'
            Some(self.parse_context()?)
        } else {
            None
        };

        // Parse optional weight - weight syntax is [number] like [0.15]
        // We need to distinguish from char class [abc]
        // For simplicity, weights must start with a digit, so we can't have weights here
        // after context since the context already consumed any [...]
        // Weight parsing only applies when there's no context or after context ends
        let weight = 0.0; // TODO: implement weight parsing with better syntax

        Ok(Regex::rewrite_rule(pattern, replacement, context, weight))
    }

    /// Parse multiple rewrite rules separated by newlines.
    pub fn parse_rule_set(&mut self) -> ParseResult<Vec<Regex>> {
        let mut rules = Vec::new();

        while !self.lexer.is_eof() {
            // Skip empty lines
            while self.lexer.peek()? == &Token::Eof {
                break;
            }

            if self.lexer.is_eof() {
                break;
            }

            let rule = self.parse_rewrite_rule()?;
            rules.push(rule);
        }

        Ok(rules)
    }

    /// Parse an alternation: `a | b | c`
    fn parse_alternation(&mut self) -> ParseResult<Regex> {
        let mut left = self.parse_concatenation()?;

        while self.lexer.peek()? == &Token::Pipe {
            self.lexer.next_token()?; // consume '|'
            let right = self.parse_concatenation()?;
            left = Regex::alt(left, right);
        }

        Ok(left)
    }

    /// Parse a concatenation: `abc`
    fn parse_concatenation(&mut self) -> ParseResult<Regex> {
        let mut result = self.parse_quantified()?;

        loop {
            // Check if next token can start a primary
            let peek = self.lexer.peek()?;
            let can_continue = Self::can_start_primary_token(peek);
            if !can_continue {
                break;
            }

            let next = self.parse_quantified()?;
            result = Regex::concat(result, next);
        }

        Ok(result)
    }

    /// Parse a quantified expression: `a*`, `a+`, `a?`, `a{2,4}`
    fn parse_quantified(&mut self) -> ParseResult<Regex> {
        let primary = self.parse_primary()?;

        let peek = self.lexer.peek()?;
        match peek {
            Token::Star => {
                self.lexer.next_token()?;
                Ok(Regex::star(primary))
            }
            Token::Plus => {
                self.lexer.next_token()?;
                Ok(Regex::plus(primary))
            }
            Token::Question => {
                self.lexer.next_token()?;
                Ok(Regex::optional(primary))
            }
            Token::QuantifierStart => {
                self.lexer.next_token()?;
                self.parse_repetition(primary)
            }
            _ => Ok(primary),
        }
    }

    /// Parse a repetition quantifier: `{n}`, `{n,}`, `{n,m}`
    fn parse_repetition(&mut self, inner: Regex) -> ParseResult<Regex> {
        // Expect a number
        let min = match self.lexer.next_token()? {
            Token::Number(n) => n,
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
            Token::QuantifierEnd => {
                // {n} - exact repetition
                self.lexer.next_token()?;
                Ok(Regex::repeat_exact(inner, min))
            }
            Token::Comma => {
                // {n,} or {n,m}
                self.lexer.next_token()?; // consume ','

                let peek = self.lexer.peek()?;
                if peek == &Token::QuantifierEnd {
                    // {n,} - unbounded
                    self.lexer.next_token()?;
                    Ok(Regex::repeat_range(inner, min, None))
                } else if let Token::Number(max) = self.lexer.next_token()? {
                    // {n,m} - bounded
                    if max < min {
                        return Err(ParseError::new(
                            ParseErrorKind::InvalidRepetition { min, max },
                            self.lexer.position(),
                        ));
                    }
                    self.expect_token(Token::QuantifierEnd)?;
                    Ok(Regex::repeat_range(inner, min, Some(max)))
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

    /// Parse a primary expression: `(...)`, `[...]`, `.`, `#`, or literal
    fn parse_primary(&mut self) -> ParseResult<Regex> {
        let token = self.lexer.next_token()?;

        match token {
            Token::GroupStart => {
                let inner = self.parse_alternation()?;
                self.expect_token(Token::GroupEnd)?;
                Ok(Regex::group(inner))
            }
            Token::CharClassStart => self.parse_char_class(),
            Token::Dot => Ok(Regex::any()),
            Token::Hash => Ok(Regex::word_boundary()),
            Token::Char(c) => Ok(Regex::char(c)),
            Token::Eof => Err(ParseError::unexpected_eof(self.lexer.position())),
            _ => Err(ParseError::unexpected_char(
                self.token_to_char(&token),
                self.lexer.position(),
            )),
        }
    }

    /// Parse a character class: `[abc]`, `[^abc]`, `[a-z]`
    fn parse_char_class(&mut self) -> ParseResult<Regex> {
        let mut chars = Vec::new();
        let mut negated = false;

        // Check for negation
        if self.lexer.peek()? == &Token::Caret {
            self.lexer.next_token()?;
            negated = true;
        }

        loop {
            let token = self.lexer.next_token()?;

            match token {
                Token::CharClassEnd => break,
                Token::Char(c) => {
                    // Check for range
                    if self.lexer.peek()? == &Token::Dash {
                        self.lexer.next_token()?; // consume '-'
                        let end_token = self.lexer.next_token()?;
                        if let Token::Char(end) = end_token {
                            // Add range
                            for ch in c..=end {
                                chars.push(ch);
                            }
                        } else if end_token == Token::CharClassEnd {
                            // Trailing dash - add both the char and the dash
                            chars.push(c);
                            chars.push('-');
                            break;
                        } else {
                            return Err(ParseError::unexpected_char(
                                self.token_to_char(&end_token),
                                self.lexer.position(),
                            ));
                        }
                    } else {
                        chars.push(c);
                    }
                }
                Token::Dash => {
                    // Dash at start of class is literal
                    chars.push('-');
                }
                Token::Eof => {
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

        if chars.is_empty() {
            return Err(ParseError::new(
                ParseErrorKind::EmptyCharClass,
                self.lexer.position(),
            ));
        }

        let class = if negated {
            CharClassChar::from_chars(&chars).negated()
        } else {
            CharClassChar::from_chars(&chars)
        };

        Ok(Regex::char_class(class))
    }

    /// Parse a context predicate: `left_context? _ right_context? syllable_clause?`
    fn parse_context(&mut self) -> ParseResult<ContextPredicate> {
        let mut left = None;
        let mut right = None;

        // Check for left context
        let peek = self.lexer.peek()?;
        if peek != &Token::Underscore {
            // Parse left context expression (may include And/Or/Not)
            left = Some(self.parse_context_expr()?);
        }

        // Expect underscore
        self.expect_token(Token::Underscore)?;

        // Check for right context
        // A right context can be: #, [charclass], char, ., group, or !/& for compound
        if self.can_start_context_expr()? {
            right = Some(self.parse_context_expr()?);
        }

        // Parse optional syllable clause: "if monosyllable", etc.
        let syllable = self.parse_syllable_clause()?;

        Ok(ContextPredicate::new_with_exprs(left, right, syllable))
    }

    /// Parse a context expression with logical operators.
    /// Precedence: NOT > AND > OR
    fn parse_context_expr(&mut self) -> ParseResult<ContextExpr> {
        self.parse_context_or()
    }

    /// Parse OR-level context expression: `a | b`
    fn parse_context_or(&mut self) -> ParseResult<ContextExpr> {
        let mut left = self.parse_context_and()?;

        while self.lexer.peek()? == &Token::Pipe {
            self.lexer.next_token()?; // consume '|'
            let right = self.parse_context_and()?;
            left = ContextExpr::or(left, right);
        }

        Ok(left)
    }

    /// Parse AND-level context expression: `a & b`
    fn parse_context_and(&mut self) -> ParseResult<ContextExpr> {
        let mut left = self.parse_context_not()?;

        while self.lexer.peek()? == &Token::Ampersand {
            self.lexer.next_token()?; // consume '&'
            let right = self.parse_context_not()?;
            left = ContextExpr::and(left, right);
        }

        Ok(left)
    }

    /// Parse NOT-level context expression: `!a`
    fn parse_context_not(&mut self) -> ParseResult<ContextExpr> {
        if self.lexer.peek()? == &Token::Exclamation {
            self.lexer.next_token()?; // consume '!'
            let inner = self.parse_context_not()?;
            Ok(ContextExpr::not(inner))
        } else {
            self.parse_context_primary()
        }
    }

    /// Parse a primary context expression: pattern, word boundary, or grouped expression.
    fn parse_context_primary(&mut self) -> ParseResult<ContextExpr> {
        let peek = self.lexer.peek()?;

        match peek {
            Token::Hash => {
                self.lexer.next_token()?;
                Ok(ContextExpr::word_boundary())
            }
            Token::CharClassStart => {
                self.lexer.next_token()?;
                let regex = self.parse_char_class()?;
                Ok(ContextExpr::pattern(regex))
            }
            Token::Char(_) | Token::Dot => {
                // Single character or any
                let token = self.lexer.next_token()?;
                let regex = match token {
                    Token::Char(c) => Regex::char(c),
                    Token::Dot => Regex::any(),
                    _ => unreachable!(),
                };
                Ok(ContextExpr::pattern(regex))
            }
            Token::GroupStart => {
                // Could be grouped context expression or pattern
                self.lexer.next_token()?; // consume '('

                // Check if this looks like a context expression (contains & or !)
                // For simplicity, parse as context expression which can handle patterns too
                let inner = self.parse_context_expr()?;
                self.expect_token(Token::GroupEnd)?;
                Ok(inner)
            }
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidContext("expected pattern".to_string()),
                self.lexer.position(),
            )),
        }
    }

    /// Parse an optional syllable clause: `if monosyllable`, `if polysyllable & final_syllable`, etc.
    fn parse_syllable_clause(&mut self) -> ParseResult<Option<SyllableExpr>> {
        if self.lexer.peek()? != &Token::IfKeyword {
            return Ok(None);
        }

        self.lexer.next_token()?; // consume 'if'
        let expr = self.parse_syllable_expr()?;
        Ok(Some(expr))
    }

    /// Parse a syllable expression with logical operators.
    /// Precedence: NOT > AND > OR
    fn parse_syllable_expr(&mut self) -> ParseResult<SyllableExpr> {
        self.parse_syllable_or()
    }

    /// Parse OR-level syllable expression: `a | b`
    fn parse_syllable_or(&mut self) -> ParseResult<SyllableExpr> {
        let mut left = self.parse_syllable_and()?;

        while self.lexer.peek()? == &Token::Pipe {
            self.lexer.next_token()?; // consume '|'
            let right = self.parse_syllable_and()?;
            left = SyllableExpr::or(left, right);
        }

        Ok(left)
    }

    /// Parse AND-level syllable expression: `a & b`
    fn parse_syllable_and(&mut self) -> ParseResult<SyllableExpr> {
        let mut left = self.parse_syllable_not()?;

        while self.lexer.peek()? == &Token::Ampersand {
            self.lexer.next_token()?; // consume '&'
            let right = self.parse_syllable_not()?;
            left = SyllableExpr::and(left, right);
        }

        Ok(left)
    }

    /// Parse NOT-level syllable expression: `!a`
    fn parse_syllable_not(&mut self) -> ParseResult<SyllableExpr> {
        if self.lexer.peek()? == &Token::Exclamation {
            self.lexer.next_token()?; // consume '!'
            let inner = self.parse_syllable_not()?;
            Ok(SyllableExpr::not(inner))
        } else {
            self.parse_syllable_primary()
        }
    }

    /// Parse a primary syllable expression: keyword or grouped expression.
    fn parse_syllable_primary(&mut self) -> ParseResult<SyllableExpr> {
        let token = self.lexer.next_token()?;

        match token {
            Token::Monosyllable => Ok(SyllableExpr::cond(SyllableCondition::Monosyllable)),
            Token::Polysyllable => Ok(SyllableExpr::cond(SyllableCondition::Polysyllable)),
            Token::OpenSyllable => Ok(SyllableExpr::cond(SyllableCondition::OpenSyllable)),
            Token::ClosedSyllable => Ok(SyllableExpr::cond(SyllableCondition::ClosedSyllable)),
            Token::FinalSyllable => Ok(SyllableExpr::cond(SyllableCondition::FinalSyllable)),
            Token::InitialSyllable => Ok(SyllableExpr::cond(SyllableCondition::InitialSyllable)),
            Token::GroupStart => {
                let inner = self.parse_syllable_expr()?;
                self.expect_token(Token::GroupEnd)?;
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
            Token::Hash
                | Token::CharClassStart
                | Token::Char(_)
                | Token::Dot
                | Token::GroupStart
                | Token::Exclamation
        ))
    }

    /// Parse an optional weight: `[0.15]`
    fn parse_weight(&mut self) -> ParseResult<f64> {
        // Check if this is actually a weight bracket
        // We need to be careful not to consume a char class
        self.lexer.enter_weight_mode();
        self.lexer.next_token()?; // consume '['

        let token = self.lexer.next_token()?;
        let weight = match token {
            Token::Float(f) => f,
            Token::Number(n) => n as f64,
            _ => {
                return Err(ParseError::new(
                    ParseErrorKind::InvalidWeight(format!("expected number, got {:?}", token)),
                    self.lexer.position(),
                ))
            }
        };

        // This should be WeightEnd but lexer mode changed
        let end_token = self.lexer.next_token()?;
        if end_token != Token::CharClassEnd {
            // Weight mode changed it back
            return Err(ParseError::new(
                ParseErrorKind::InvalidWeight("expected ']'".to_string()),
                self.lexer.position(),
            ));
        }

        Ok(weight)
    }

    /// Check if we're at context, weight, or end.
    fn is_at_context_or_weight_or_end(&mut self) -> ParseResult<bool> {
        let peek = self.lexer.peek()?;
        Ok(matches!(
            peek,
            Token::Slash | Token::Eof
        ))
    }

    /// Check if a token can start a primary expression.
    fn can_start_primary_token(token: &Token) -> bool {
        matches!(
            token,
            Token::Char(_)
                | Token::CharClassStart
                | Token::GroupStart
                | Token::Dot
                | Token::Hash
        )
    }

    /// Expect a specific token, returning an error if not found.
    fn expect_token(&mut self, expected: Token) -> ParseResult<()> {
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

    /// Convert a token to a character for error messages.
    fn token_to_char(&self, token: &Token) -> char {
        match token {
            Token::Char(c) => *c,
            Token::CharClassStart => '[',
            Token::CharClassEnd => ']',
            Token::Caret => '^',
            Token::Dash => '-',
            Token::GroupStart => '(',
            Token::GroupEnd => ')',
            Token::Pipe => '|',
            Token::Star => '*',
            Token::Plus => '+',
            Token::Question => '?',
            Token::Dot => '.',
            Token::QuantifierStart => '{',
            Token::QuantifierEnd => '}',
            Token::Comma => ',',
            Token::Number(_) => '0',
            Token::Float(_) => '0',
            Token::Arrow => '>',
            Token::Slash => '/',
            Token::Underscore => '_',
            Token::Hash => '#',
            Token::WeightStart => '[',
            Token::WeightEnd => ']',
            Token::Ampersand => '&',
            Token::Exclamation => '!',
            Token::IfKeyword => 'i',
            Token::Monosyllable => 'm',
            Token::Polysyllable => 'p',
            Token::OpenSyllable => 'o',
            Token::ClosedSyllable => 'c',
            Token::FinalSyllable => 'f',
            Token::InitialSyllable => 'i',
            Token::Eof => '\0',
        }
    }
}

// ============================================================================
// Byte-level Parser
// ============================================================================

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

        Ok(RegexByte::rewrite_rule(pattern, replacement, context, weight))
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

    /// Parse a repetition quantifier.
    fn parse_repetition(&mut self, inner: RegexByte) -> ParseResult<RegexByte> {
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
                Ok(RegexByte::group(inner))
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
            Ok(ContextExprByte::not(inner))
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
                    _ => unreachable!(),
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
            Ok(SyllableExpr::not(inner))
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
            TokenByte::InitialSyllable => Ok(SyllableExpr::cond(SyllableCondition::InitialSyllable)),
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

// ============================================================================
// Convenience functions
// ============================================================================

/// Parse a regex pattern string.
pub fn parse(input: &str) -> ParseResult<Regex> {
    Parser::new(input).parse()
}

/// Parse a rewrite rule string.
pub fn parse_rule(input: &str) -> ParseResult<Regex> {
    Parser::new(input).parse_rewrite_rule()
}

/// Parse multiple rewrite rules.
pub fn parse_rules(input: &str) -> ParseResult<Vec<Regex>> {
    Parser::new(input).parse_rule_set()
}

/// Parse a byte-level regex pattern.
pub fn parse_bytes(input: &[u8]) -> ParseResult<RegexByte> {
    ParserByte::new(input).parse()
}

/// Parse a byte-level rewrite rule.
pub fn parse_rule_bytes(input: &[u8]) -> ParseResult<RegexByte> {
    ParserByte::new(input).parse_rewrite_rule()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_literal() {
        let r = parse("phone").unwrap();
        assert_eq!(r.to_string(), "phone");
    }

    #[test]
    fn test_parse_alternation() {
        let r = parse("ph|f").unwrap();
        assert_eq!(r.to_string(), "(ph|f)");
    }

    #[test]
    fn test_parse_group() {
        let r = parse("(ph|f)one").unwrap();
        assert_eq!(r.to_string(), "((ph|f))one");
    }

    #[test]
    fn test_parse_star() {
        let r = parse("a*").unwrap();
        assert_eq!(r.to_string(), "a*");
    }

    #[test]
    fn test_parse_plus() {
        let r = parse("a+").unwrap();
        assert_eq!(r.to_string(), "a+");
    }

    #[test]
    fn test_parse_optional() {
        let r = parse("a?").unwrap();
        assert_eq!(r.to_string(), "a?");
    }

    #[test]
    fn test_parse_char_class() {
        let r = parse("[aeiou]").unwrap();
        assert_eq!(r.to_string(), "[aeiou]");
    }

    #[test]
    fn test_parse_char_class_negated() {
        let r = parse("[^aeiou]").unwrap();
        assert_eq!(r.to_string(), "[^aeiou]");
    }

    #[test]
    fn test_parse_char_class_range() {
        let r = parse("[a-z]").unwrap();
        // The display will show all characters in the range
        assert!(r.to_string().starts_with('['));
        assert!(r.to_string().ends_with(']'));
    }

    #[test]
    fn test_parse_any() {
        let r = parse("a.b").unwrap();
        assert_eq!(r.to_string(), "a.b");
    }

    #[test]
    fn test_parse_repetition_exact() {
        let r = parse("a{3}").unwrap();
        assert_eq!(r.to_string(), "a{3}");
    }

    #[test]
    fn test_parse_repetition_range() {
        let r = parse("a{2,4}").unwrap();
        assert_eq!(r.to_string(), "a{2,4}");
    }

    #[test]
    fn test_parse_repetition_unbounded() {
        let r = parse("a{2,}").unwrap();
        assert_eq!(r.to_string(), "a{2,}");
    }

    #[test]
    fn test_parse_escape() {
        let r = parse("\\[\\]").unwrap();
        assert_eq!(r.to_string(), "\\[\\]");
    }

    #[test]
    fn test_parse_word_boundary() {
        let r = parse("#abc#").unwrap();
        assert_eq!(r.to_string(), "#abc#");
    }

    #[test]
    fn test_parse_rewrite_rule_simple() {
        let r = parse_rule("ph -> f").unwrap();
        assert!(r.is_rewrite_rule());
        assert_eq!(r.to_string(), "ph -> f");
    }

    #[test]
    fn test_parse_rewrite_rule_with_context() {
        let r = parse_rule("c -> s / _[ei]").unwrap();
        assert!(r.is_rewrite_rule());
        assert_eq!(r.to_string(), "c -> s / _[ei]");
    }

    #[test]
    fn test_parse_rewrite_rule_word_end() {
        let r = parse_rule("e -> / _#").unwrap();
        assert!(r.is_rewrite_rule());
        assert_eq!(r.to_string(), "e ->  / _#");
    }

    #[test]
    fn test_parse_rewrite_rule_word_start() {
        let r = parse_rule("k -> c / #_").unwrap();
        assert!(r.is_rewrite_rule());
        assert_eq!(r.to_string(), "k -> c / #_");
    }

    #[test]
    fn test_parse_complex_pattern() {
        let r = parse("(ph|f)one[s]?").unwrap();
        // Should parse without error
        assert!(!r.is_empty());
    }

    #[test]
    fn test_parse_error_unclosed_group() {
        let result = parse("(abc");
        assert!(result.is_err());
        let err = result.unwrap_err();
        // The error could be UnclosedGroup or ExpectedChar(')')
        assert!(
            matches!(err.kind, ParseErrorKind::UnclosedGroup)
                || matches!(err.kind, ParseErrorKind::ExpectedChar(')'))
        );
    }

    #[test]
    fn test_parse_error_unclosed_char_class() {
        let result = parse("[abc");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UnclosedCharClass));
    }

    #[test]
    fn test_parse_error_empty_char_class() {
        let result = parse("[]");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::EmptyCharClass));
    }

    #[test]
    fn test_parse_error_invalid_repetition() {
        let result = parse("a{5,3}");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind,
            ParseErrorKind::InvalidRepetition { min: 5, max: 3 }
        ));
    }

    // Byte-level tests

    #[test]
    fn test_parse_bytes_literal() {
        let r = parse_bytes(b"phone").unwrap();
        assert_eq!(r.to_string(), "phone");
    }

    #[test]
    fn test_parse_bytes_alternation() {
        let r = parse_bytes(b"ph|f").unwrap();
        assert_eq!(r.to_string(), "(ph|f)");
    }

    #[test]
    fn test_parse_bytes_rewrite_rule() {
        let r = parse_rule_bytes(b"ph -> f").unwrap();
        assert!(r.is_rewrite_rule());
        assert_eq!(r.to_string(), "ph -> f");
    }
}
