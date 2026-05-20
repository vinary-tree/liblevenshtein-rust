//! Byte-level lexer for phonetic regular expressions (ASCII-only, leaner subset).

use crate::phonetic::common::syllable::SyllableCondition;
use crate::phonetic::common::traits::{LexerLike, TokenLike};
use crate::phonetic::regex::error::{ParseError, ParseErrorKind, ParseResult, Position};

/// Byte-level lexer for ASCII patterns.
pub struct LexerByte<'a> {
    input: &'a [u8],
    pos: usize,
    position: Position,
    in_char_class: bool,
    peeked: Vec<(TokenByte, Position)>,
}

/// Byte-level token.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenByte {
    /// A literal byte
    Byte(u8),

    /// Start of byte class `[`
    ByteClassStart,

    /// End of byte class `]`
    ByteClassEnd,

    /// Negation in byte class `^`
    Caret,

    /// Range in byte class `-`
    Dash,

    /// Start of group `(`
    GroupStart,

    /// End of group `)`
    GroupEnd,

    /// Alternation `|`
    Pipe,

    /// Kleene star `*`
    Star,

    /// Kleene plus `+`
    Plus,

    /// Optional `?`
    Question,

    /// Any byte `.`
    Dot,

    /// Start of quantifier `{`
    QuantifierStart,

    /// End of quantifier `}`
    QuantifierEnd,

    /// Comma in quantifier `,`
    Comma,

    /// Number (for quantifiers and weights)
    Number(usize),

    /// Float number (for weights)
    Float(f64),

    /// Arrow `->` for rewrite rules
    Arrow,

    /// Context separator `/`
    Slash,

    /// Underscore `_` for context position
    Underscore,

    /// Word boundary `#`
    Hash,

    /// Start of weight `[`
    WeightStart,

    /// End of weight `]`
    WeightEnd,

    /// Ampersand `&` for context AND
    Ampersand,

    /// Exclamation `!` for context NOT
    Exclamation,

    /// "if" keyword for syllable conditions
    IfKeyword,

    /// Monosyllable condition keyword
    Monosyllable,

    /// Polysyllable condition keyword
    Polysyllable,

    /// Open syllable condition keyword
    OpenSyllable,

    /// Closed syllable condition keyword
    ClosedSyllable,

    /// Final syllable condition keyword
    FinalSyllable,

    /// Initial syllable condition keyword
    InitialSyllable,

    /// End of input
    Eof,
}

impl<'a> LexerByte<'a> {
    /// Create a new byte-level lexer.
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            position: Position::start(),
            in_char_class: false,
            peeked: Vec::new(),
        }
    }

    /// Get the current position.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Peek at the next token without consuming it.
    pub fn peek(&mut self) -> ParseResult<&TokenByte> {
        if self.peeked.is_empty() {
            let token = self.next_token_internal()?;
            let pos = self.position;
            self.peeked.push((token, pos));
        }
        Ok(&self.peeked.last().expect("just pushed").0)
    }

    /// Get the next token.
    pub fn next_token(&mut self) -> ParseResult<TokenByte> {
        if let Some((token, pos)) = self.peeked.pop() {
            self.position = pos;
            Ok(token)
        } else {
            self.next_token_internal()
        }
    }

    /// Advance to the next byte.
    fn advance(&mut self) -> Option<u8> {
        if self.pos < self.input.len() {
            let b = self.input[self.pos];
            self.pos += 1;
            self.position.offset = self.pos;
            if b == b'\n' {
                self.position.line += 1;
                self.position.column = 1;
            } else {
                self.position.column += 1;
            }
            Some(b)
        } else {
            None
        }
    }

    /// Peek at the next byte.
    fn peek_byte(&self) -> Option<u8> {
        if self.pos < self.input.len() {
            Some(self.input[self.pos])
        } else {
            None
        }
    }

    /// Skip whitespace.
    fn skip_whitespace(&mut self) {
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Parse an escape sequence.
    fn parse_escape(&mut self) -> ParseResult<u8> {
        match self.advance() {
            Some(b'n') => Ok(b'\n'),
            Some(b'r') => Ok(b'\r'),
            Some(b't') => Ok(b'\t'),
            Some(b'0') => Ok(0),
            Some(b'\\') => Ok(b'\\'),
            Some(b'[') => Ok(b'['),
            Some(b']') => Ok(b']'),
            Some(b'(') => Ok(b'('),
            Some(b')') => Ok(b')'),
            Some(b'{') => Ok(b'{'),
            Some(b'}') => Ok(b'}'),
            Some(b'|') => Ok(b'|'),
            Some(b'*') => Ok(b'*'),
            Some(b'+') => Ok(b'+'),
            Some(b'?') => Ok(b'?'),
            Some(b'.') => Ok(b'.'),
            Some(b'^') => Ok(b'^'),
            Some(b'$') => Ok(b'$'),
            Some(b'-') => Ok(b'-'),
            Some(b'/') => Ok(b'/'),
            Some(b'#') => Ok(b'#'),
            Some(b'x') => self.parse_hex_escape(),
            Some(b) => Err(ParseError::invalid_escape(b as char, self.position)),
            None => Err(ParseError::unexpected_eof(self.position)),
        }
    }

    /// Parse a hex escape sequence (\xNN).
    fn parse_hex_escape(&mut self) -> ParseResult<u8> {
        let mut value: u8 = 0;
        for _ in 0..2 {
            match self.advance() {
                Some(b) if b.is_ascii_hexdigit() => {
                    let digit = if b.is_ascii_digit() {
                        b - b'0'
                    } else if b.is_ascii_lowercase() {
                        b - b'a' + 10
                    } else {
                        b - b'A' + 10
                    };
                    value = value * 16 + digit;
                }
                Some(b) => {
                    return Err(ParseError::with_context(
                        ParseErrorKind::InvalidEscape(b as char),
                        self.position,
                        "expected hex digit",
                    ))
                }
                None => return Err(ParseError::unexpected_eof(self.position)),
            }
        }
        Ok(value)
    }

    /// Parse a number.
    fn parse_number(&mut self, first_digit: u8) -> ParseResult<usize> {
        let mut value = (first_digit - b'0') as usize;
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_digit() {
                self.advance();
                value = value * 10 + (b - b'0') as usize;
            } else {
                break;
            }
        }
        Ok(value)
    }

    /// Internal method to get the next token.
    fn next_token_internal(&mut self) -> ParseResult<TokenByte> {
        if !self.in_char_class {
            self.skip_whitespace();
        }

        let b = match self.advance() {
            Some(b) => b,
            None => return Ok(TokenByte::Eof),
        };

        if self.in_char_class {
            return match b {
                b']' => {
                    self.in_char_class = false;
                    Ok(TokenByte::ByteClassEnd)
                }
                b'^' => Ok(TokenByte::Caret),
                b'-' => Ok(TokenByte::Dash),
                b'\\' => {
                    let escaped = self.parse_escape()?;
                    Ok(TokenByte::Byte(escaped))
                }
                _ => Ok(TokenByte::Byte(b)),
            };
        }

        match b {
            b'[' => {
                self.in_char_class = true;
                Ok(TokenByte::ByteClassStart)
            }
            b']' => Ok(TokenByte::ByteClassEnd),
            b'(' => Ok(TokenByte::GroupStart),
            b')' => Ok(TokenByte::GroupEnd),
            b'|' => Ok(TokenByte::Pipe),
            b'*' => Ok(TokenByte::Star),
            b'+' => Ok(TokenByte::Plus),
            b'?' => Ok(TokenByte::Question),
            b'.' => Ok(TokenByte::Dot),
            b'{' => Ok(TokenByte::QuantifierStart),
            b'}' => Ok(TokenByte::QuantifierEnd),
            b',' => Ok(TokenByte::Comma),
            b'/' => Ok(TokenByte::Slash),
            b'_' => Ok(TokenByte::Underscore),
            b'#' => Ok(TokenByte::Hash),
            b'-' => {
                if self.peek_byte() == Some(b'>') {
                    self.advance();
                    Ok(TokenByte::Arrow)
                } else {
                    Ok(TokenByte::Dash)
                }
            }
            b'\\' => {
                let escaped = self.parse_escape()?;
                Ok(TokenByte::Byte(escaped))
            }
            b'&' => Ok(TokenByte::Ampersand),
            b'!' => Ok(TokenByte::Exclamation),
            b if b.is_ascii_digit() => {
                let n = self.parse_number(b)?;
                Ok(TokenByte::Number(n))
            }
            b if b.is_ascii_alphabetic() => {
                self.parse_keyword_or_byte(b)
            }
            _ => Ok(TokenByte::Byte(b)),
        }
    }

    /// Parse a keyword or return byte tokens.
    fn parse_keyword_or_byte(&mut self, first: u8) -> ParseResult<TokenByte> {
        // Collect the full identifier
        let mut word = Vec::new();
        word.push(first);
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.advance();
                word.push(b);
            } else {
                break;
            }
        }

        // Check for known keywords
        match word.as_slice() {
            b"if" => Ok(TokenByte::IfKeyword),
            b"monosyllable" => Ok(TokenByte::Monosyllable),
            b"polysyllable" => Ok(TokenByte::Polysyllable),
            b"open_syllable" => Ok(TokenByte::OpenSyllable),
            b"closed_syllable" => Ok(TokenByte::ClosedSyllable),
            b"final_syllable" => Ok(TokenByte::FinalSyllable),
            b"initial_syllable" => Ok(TokenByte::InitialSyllable),
            _ => {
                // Not a keyword - push remaining bytes to peeked queue
                let pos = self.position;
                for b in word.iter().skip(1).rev() {
                    self.peeked.push((TokenByte::Byte(*b), pos));
                }
                Ok(TokenByte::Byte(first))
            }
        }
    }

    /// Check if we're at end of input.
    pub fn is_eof(&mut self) -> bool {
        self.skip_whitespace();
        self.pos >= self.input.len()
    }
}

impl TokenLike for TokenByte {
    fn is_pipe(&self) -> bool {
        matches!(self, TokenByte::Pipe)
    }

    fn is_ampersand(&self) -> bool {
        matches!(self, TokenByte::Ampersand)
    }

    fn is_exclamation(&self) -> bool {
        matches!(self, TokenByte::Exclamation)
    }

    fn is_group_start(&self) -> bool {
        matches!(self, TokenByte::GroupStart)
    }

    fn is_group_end(&self) -> bool {
        matches!(self, TokenByte::GroupEnd)
    }

    fn is_hash(&self) -> bool {
        matches!(self, TokenByte::Hash)
    }

    fn is_star(&self) -> bool {
        matches!(self, TokenByte::Star)
    }

    fn is_plus(&self) -> bool {
        matches!(self, TokenByte::Plus)
    }

    fn is_question(&self) -> bool {
        matches!(self, TokenByte::Question)
    }

    fn is_brace_start(&self) -> bool {
        matches!(self, TokenByte::QuantifierStart)
    }

    fn is_eof(&self) -> bool {
        matches!(self, TokenByte::Eof)
    }

    fn is_if_keyword(&self) -> bool {
        matches!(self, TokenByte::IfKeyword)
    }

    fn as_syllable_condition(&self) -> Option<SyllableCondition> {
        match self {
            TokenByte::Monosyllable => Some(SyllableCondition::Monosyllable),
            TokenByte::Polysyllable => Some(SyllableCondition::Polysyllable),
            TokenByte::OpenSyllable => Some(SyllableCondition::OpenSyllable),
            TokenByte::ClosedSyllable => Some(SyllableCondition::ClosedSyllable),
            TokenByte::FinalSyllable => Some(SyllableCondition::FinalSyllable),
            TokenByte::InitialSyllable => Some(SyllableCondition::InitialSyllable),
            _ => None,
        }
    }

    fn can_start_primary(&self) -> bool {
        matches!(
            self,
            TokenByte::Byte(_)
                | TokenByte::ByteClassStart
                | TokenByte::GroupStart
                | TokenByte::Dot
                | TokenByte::Hash
        )
    }
}

impl<'a> LexerLike for LexerByte<'a> {
    type Token = TokenByte;
    type Error = ParseError;

    fn peek(&mut self) -> Result<&Self::Token, Self::Error> {
        LexerByte::peek(self)
    }

    fn advance(&mut self) -> Result<Self::Token, Self::Error> {
        LexerByte::next_token(self)
    }

    fn position(&self) -> Position {
        LexerByte::position(self)
    }
}
