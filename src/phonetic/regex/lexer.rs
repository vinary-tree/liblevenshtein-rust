//! Lexer for phonetic regular expressions.
//!
//! Tokenizes input strings into a stream of tokens for the parser.

use super::error::{ParseError, ParseErrorKind, ParseResult, Position};

/// A token in the phonetic regex language.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// A literal character
    Char(char),

    /// Start of character class `[`
    CharClassStart,

    /// End of character class `]`
    CharClassEnd,

    /// Negation in character class `^`
    Caret,

    /// Range in character class `-`
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

    /// Any character `.`
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

    /// User-defined symbol reference ($NAME or ${NAME})
    SymbolRef(String),

    /// End of input
    Eof,
}

impl Token {
    /// Check if this token is a quantifier.
    pub fn is_quantifier(&self) -> bool {
        matches!(
            self,
            Token::Star | Token::Plus | Token::Question | Token::QuantifierStart
        )
    }

    /// Check if this token can start a primary expression.
    pub fn can_start_primary(&self) -> bool {
        matches!(
            self,
            Token::Char(_)
                | Token::CharClassStart
                | Token::GroupStart
                | Token::Dot
                | Token::Hash
                | Token::SymbolRef(_)
        )
    }
}

/// Lexer for phonetic regex patterns.
pub struct Lexer<'a> {
    input: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    position: Position,
    /// Whether we're currently inside a character class `[...]`
    in_char_class: bool,
    /// Whether we're currently inside a weight bracket `[...]`
    in_weight: bool,
    /// Stack of peeked tokens for lookahead (token, position, in_char_class, in_weight)
    peeked: Vec<(Token, Position, bool, bool)>,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given input.
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.char_indices().peekable(),
            position: Position::start(),
            in_char_class: false,
            in_weight: false,
            peeked: Vec::new(),
        }
    }

    /// Get the current position in the input.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Peek at the next token without consuming it.
    pub fn peek(&mut self) -> ParseResult<&Token> {
        if self.peeked.is_empty() {
            // Save state before tokenizing
            let saved_in_char_class = self.in_char_class;
            let saved_in_weight = self.in_weight;
            let token = self.next_token_internal()?;
            let pos = self.position;
            // Store the state AFTER tokenizing (this is what we'll restore when consuming)
            let new_in_char_class = self.in_char_class;
            let new_in_weight = self.in_weight;
            // Restore to state before tokenizing for peek
            self.in_char_class = saved_in_char_class;
            self.in_weight = saved_in_weight;
            self.peeked.push((token, pos, new_in_char_class, new_in_weight));
        }
        Ok(&self.peeked.last().expect("just pushed").0)
    }

    /// Get the next token.
    pub fn next_token(&mut self) -> ParseResult<Token> {
        if let Some((token, pos, in_char_class, in_weight)) = self.peeked.pop() {
            // Restore the state to what it was after tokenizing
            self.in_char_class = in_char_class;
            self.in_weight = in_weight;
            self.position = pos;
            Ok(token)
        } else {
            self.next_token_internal()
        }
    }

    /// Consume the next token if it matches the expected token.
    pub fn expect(&mut self, expected: &Token) -> ParseResult<Token> {
        let token = self.next_token()?;
        if &token == expected {
            Ok(token)
        } else {
            Err(ParseError::with_context(
                ParseErrorKind::UnexpectedChar(self.token_to_char(&token)),
                self.position,
                format!("expected {:?}", expected),
            ))
        }
    }

    /// Skip whitespace and comments.
    fn skip_whitespace(&mut self) {
        while let Some(&(_, c)) = self.chars.peek() {
            if c.is_whitespace() {
                self.advance();
            } else if c == '#' && !self.in_char_class {
                // In rewrite rule syntax, # is word boundary, not comment
                // Only treat as comment at line start or after whitespace
                // For simplicity, we'll handle # as a token, not a comment starter
                break;
            } else {
                break;
            }
        }
    }

    /// Advance to the next character.
    fn advance(&mut self) -> Option<char> {
        if let Some((offset, c)) = self.chars.next() {
            self.position.offset = offset;
            if c == '\n' {
                self.position.line += 1;
                self.position.column = 1;
            } else {
                self.position.column += 1;
            }
            Some(c)
        } else {
            None
        }
    }

    /// Peek at the next character.
    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().map(|&(_, c)| c)
    }

    /// Convert a token to a representative character (for error messages).
    fn token_to_char(&self, token: &Token) -> char {
        match token {
            Token::Char(c) => *c,
            Token::CharClassStart | Token::WeightStart => '[',
            Token::CharClassEnd | Token::WeightEnd => ']',
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
            Token::Arrow => '-',
            Token::Slash => '/',
            Token::Underscore => '_',
            Token::Hash => '#',
            Token::Ampersand => '&',
            Token::Exclamation => '!',
            Token::IfKeyword
            | Token::Monosyllable
            | Token::Polysyllable
            | Token::OpenSyllable
            | Token::ClosedSyllable
            | Token::FinalSyllable
            | Token::InitialSyllable => 'i', // keywords
            Token::SymbolRef(_) => '$',
            Token::Eof => '\0',
        }
    }

    /// Parse an escape sequence.
    fn parse_escape(&mut self) -> ParseResult<char> {
        match self.advance() {
            Some('n') => Ok('\n'),
            Some('r') => Ok('\r'),
            Some('t') => Ok('\t'),
            Some('0') => Ok('\0'),
            Some('\\') => Ok('\\'),
            Some('[') => Ok('['),
            Some(']') => Ok(']'),
            Some('(') => Ok('('),
            Some(')') => Ok(')'),
            Some('{') => Ok('{'),
            Some('}') => Ok('}'),
            Some('|') => Ok('|'),
            Some('*') => Ok('*'),
            Some('+') => Ok('+'),
            Some('?') => Ok('?'),
            Some('.') => Ok('.'),
            Some('^') => Ok('^'),
            Some('$') => Ok('$'),
            Some('-') => Ok('-'),
            Some('/') => Ok('/'),
            Some('#') => Ok('#'),
            Some('x') => self.parse_hex_escape(2),
            Some('u') => self.parse_hex_escape(4),
            Some('U') => self.parse_hex_escape(8),
            Some(c) => Err(ParseError::invalid_escape(c, self.position)),
            None => Err(ParseError::unexpected_eof(self.position)),
        }
    }

    /// Parse a hex escape sequence (\xNN, \uNNNN, \UNNNNNNNN).
    fn parse_hex_escape(&mut self, num_digits: usize) -> ParseResult<char> {
        let mut value: u32 = 0;
        for _ in 0..num_digits {
            match self.advance() {
                Some(c) if c.is_ascii_hexdigit() => {
                    value = value * 16 + c.to_digit(16).expect("is_ascii_hexdigit");
                }
                Some(c) => {
                    return Err(ParseError::with_context(
                        ParseErrorKind::InvalidEscape(c),
                        self.position,
                        format!("expected hex digit, got '{}'", c),
                    ))
                }
                None => return Err(ParseError::unexpected_eof(self.position)),
            }
        }
        char::from_u32(value).ok_or_else(|| {
            ParseError::new(ParseErrorKind::InvalidCodePoint(value), self.position)
        })
    }

    /// Parse a number.
    fn parse_number(&mut self, first_digit: char) -> ParseResult<usize> {
        let mut value = first_digit.to_digit(10).expect("is digit") as usize;
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                self.advance();
                value = value * 10 + c.to_digit(10).expect("is digit") as usize;
            } else {
                break;
            }
        }
        Ok(value)
    }

    /// Parse a float number (for weights).
    fn parse_float(&mut self, integer_part: usize) -> ParseResult<f64> {
        let mut value = integer_part as f64;

        // Check for decimal point
        if self.peek_char() == Some('.') {
            self.advance(); // consume '.'

            let mut decimal_place = 0.1;
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() {
                    self.advance();
                    value += c.to_digit(10).expect("is digit") as f64 * decimal_place;
                    decimal_place *= 0.1;
                } else {
                    break;
                }
            }
        }

        Ok(value)
    }

    /// Parse a symbol reference ($NAME or ${NAME}).
    fn parse_symbol_ref(&mut self) -> ParseResult<Token> {
        let pos = self.position;
        if self.peek_char() == Some('{') {
            // Explicit form: ${NAME}
            self.advance(); // consume '{'
            let mut name = String::new();
            while let Some(c) = self.peek_char() {
                if c == '}' {
                    self.advance();
                    if name.is_empty() {
                        return Err(ParseError::with_context(
                            ParseErrorKind::InvalidCharClass("empty symbol name in ${...}".to_string()),
                            pos,
                            "expected symbol name",
                        ));
                    }
                    return Ok(Token::SymbolRef(name));
                } else if c.is_alphanumeric() || c == '_' {
                    self.advance();
                    name.push(c);
                } else {
                    return Err(ParseError::with_context(
                        ParseErrorKind::InvalidCharClass(format!("invalid character '{}' in symbol name", c)),
                        self.position,
                        "symbol names must be alphanumeric",
                    ));
                }
            }
            Err(ParseError::with_context(
                ParseErrorKind::UnexpectedEof,
                pos,
                "unclosed ${...} - expected '}'",
            ))
        } else {
            // Simple form: $NAME
            let mut name = String::new();
            while let Some(c) = self.peek_char() {
                if c.is_alphanumeric() || c == '_' {
                    self.advance();
                    name.push(c);
                } else {
                    break;
                }
            }
            if name.is_empty() {
                return Err(ParseError::with_context(
                    ParseErrorKind::InvalidCharClass("expected symbol name after '$'".to_string()),
                    pos,
                    "symbol name required",
                ));
            }
            Ok(Token::SymbolRef(name))
        }
    }

    /// Internal method to get the next token.
    fn next_token_internal(&mut self) -> ParseResult<Token> {
        // Skip whitespace unless inside character class
        if !self.in_char_class {
            self.skip_whitespace();
        }

        let c = match self.advance() {
            Some(c) => c,
            None => return Ok(Token::Eof),
        };

        // Inside character class, most characters are literal
        if self.in_char_class {
            return match c {
                ']' => {
                    self.in_char_class = false;
                    Ok(Token::CharClassEnd)
                }
                '^' => Ok(Token::Caret),
                '-' => Ok(Token::Dash),
                '\\' => {
                    let escaped = self.parse_escape()?;
                    Ok(Token::Char(escaped))
                }
                '$' => self.parse_symbol_ref(),
                _ => Ok(Token::Char(c)),
            };
        }

        // Inside weight bracket
        if self.in_weight {
            return match c {
                ']' => {
                    self.in_weight = false;
                    Ok(Token::WeightEnd)
                }
                c if c.is_ascii_digit() => {
                    let int_part = self.parse_number(c)?;
                    let float_val = self.parse_float(int_part)?;
                    Ok(Token::Float(float_val))
                }
                '.' => {
                    // Handle .5 style floats
                    let float_val = self.parse_float(0)?;
                    Ok(Token::Float(float_val))
                }
                _ if c.is_whitespace() => self.next_token_internal(),
                _ => Err(ParseError::unexpected_char(c, self.position)),
            };
        }

        // Normal context
        match c {
            '[' => {
                // Check if this starts a weight (after a rewrite rule)
                // For now, treat all [ as character class start
                // The parser will handle weight context
                self.in_char_class = true;
                Ok(Token::CharClassStart)
            }
            ']' => Ok(Token::CharClassEnd),
            '(' => Ok(Token::GroupStart),
            ')' => Ok(Token::GroupEnd),
            '|' => Ok(Token::Pipe),
            '*' => Ok(Token::Star),
            '+' => Ok(Token::Plus),
            '?' => Ok(Token::Question),
            '.' => Ok(Token::Dot),
            '{' => Ok(Token::QuantifierStart),
            '}' => Ok(Token::QuantifierEnd),
            ',' => Ok(Token::Comma),
            '/' => Ok(Token::Slash),
            '_' => Ok(Token::Underscore),
            '#' => Ok(Token::Hash),
            '-' => {
                // Check for arrow `->`
                if self.peek_char() == Some('>') {
                    self.advance();
                    Ok(Token::Arrow)
                } else {
                    Ok(Token::Dash)
                }
            }
            '\\' => {
                let escaped = self.parse_escape()?;
                Ok(Token::Char(escaped))
            }
            '&' => Ok(Token::Ampersand),
            '!' => Ok(Token::Exclamation),
            '$' => self.parse_symbol_ref(),
            c if c.is_ascii_digit() => {
                let n = self.parse_number(c)?;
                Ok(Token::Number(n))
            }
            c if c.is_ascii_alphabetic() => {
                // Check for keywords
                self.parse_keyword_or_char(c)
            }
            _ => Ok(Token::Char(c)),
        }
    }

    /// Parse a keyword or return character tokens.
    fn parse_keyword_or_char(&mut self, first: char) -> ParseResult<Token> {
        // Collect the full identifier
        let mut word = String::new();
        word.push(first);
        while let Some(c) = self.peek_char() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.advance();
                word.push(c);
            } else {
                break;
            }
        }

        // Check for known keywords
        match word.as_str() {
            "if" => Ok(Token::IfKeyword),
            "monosyllable" => Ok(Token::Monosyllable),
            "polysyllable" => Ok(Token::Polysyllable),
            "open_syllable" => Ok(Token::OpenSyllable),
            "closed_syllable" => Ok(Token::ClosedSyllable),
            "final_syllable" => Ok(Token::FinalSyllable),
            "initial_syllable" => Ok(Token::InitialSyllable),
            _ => {
                // Not a keyword - return as sequence of character tokens
                // Push all remaining chars (in reverse order) to peeked queue
                // so they'll be returned as Char tokens on subsequent calls
                let pos = self.position;
                let in_cc = self.in_char_class;
                let in_w = self.in_weight;
                // Collect to Vec first since Chars doesn't implement ExactSizeIterator
                let remaining: Vec<char> = word.chars().skip(1).collect();
                for c in remaining.into_iter().rev() {
                    self.peeked.push((Token::Char(c), pos, in_cc, in_w));
                }
                Ok(Token::Char(first))
            }
        }
    }

    /// Enter weight parsing mode (after seeing a rewrite rule).
    pub fn enter_weight_mode(&mut self) {
        self.in_char_class = false;
        self.in_weight = true;
    }

    /// Enter char class mode (used when re-entering after nested constructs).
    pub fn enter_char_class_mode(&mut self) {
        self.in_char_class = true;
    }

    /// Check if we're at end of input.
    pub fn is_eof(&mut self) -> bool {
        self.skip_whitespace();
        self.chars.peek().is_none()
    }
}

/// Byte-level lexer for ASCII patterns.
pub struct LexerByte<'a> {
    input: &'a [u8],
    pos: usize,
    position: Position,
    in_char_class: bool,
    in_weight: bool,
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
            in_weight: false,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_simple() {
        let mut lexer = Lexer::new("abc");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('c'));
        assert_eq!(lexer.next_token().unwrap(), Token::Eof);
    }

    #[test]
    fn test_lexer_operators() {
        let mut lexer = Lexer::new("a*b+c?");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Star);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
        assert_eq!(lexer.next_token().unwrap(), Token::Plus);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('c'));
        assert_eq!(lexer.next_token().unwrap(), Token::Question);
    }

    #[test]
    fn test_lexer_alternation() {
        let mut lexer = Lexer::new("a|b");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Pipe);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
    }

    #[test]
    fn test_lexer_groups() {
        let mut lexer = Lexer::new("(ab)");
        assert_eq!(lexer.next_token().unwrap(), Token::GroupStart);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
        assert_eq!(lexer.next_token().unwrap(), Token::GroupEnd);
    }

    #[test]
    fn test_lexer_char_class() {
        let mut lexer = Lexer::new("[abc]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('c'));
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
    }

    #[test]
    fn test_lexer_char_class_negated() {
        let mut lexer = Lexer::new("[^abc]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        assert_eq!(lexer.next_token().unwrap(), Token::Caret);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('c'));
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
    }

    #[test]
    fn test_lexer_char_class_range() {
        let mut lexer = Lexer::new("[a-z]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Dash);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('z'));
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
    }

    #[test]
    fn test_lexer_quantifier() {
        let mut lexer = Lexer::new("a{2,5}");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::QuantifierStart);
        assert_eq!(lexer.next_token().unwrap(), Token::Number(2));
        assert_eq!(lexer.next_token().unwrap(), Token::Comma);
        assert_eq!(lexer.next_token().unwrap(), Token::Number(5));
        assert_eq!(lexer.next_token().unwrap(), Token::QuantifierEnd);
    }

    #[test]
    fn test_lexer_arrow() {
        let mut lexer = Lexer::new("ph -> f");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('p'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('h'));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('f'));
    }

    #[test]
    fn test_lexer_context() {
        let mut lexer = Lexer::new("c -> s / _[ei]");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('c'));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('s'));
        assert_eq!(lexer.next_token().unwrap(), Token::Slash);
        assert_eq!(lexer.next_token().unwrap(), Token::Underscore);
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('e'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('i'));
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
    }

    #[test]
    fn test_lexer_word_boundary() {
        let mut lexer = Lexer::new("e -> / _#");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('e'));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Slash);
        assert_eq!(lexer.next_token().unwrap(), Token::Underscore);
        assert_eq!(lexer.next_token().unwrap(), Token::Hash);
    }

    #[test]
    fn test_lexer_escape() {
        let mut lexer = Lexer::new("\\[\\]\\*");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('['));
        assert_eq!(lexer.next_token().unwrap(), Token::Char(']'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('*'));
    }

    #[test]
    fn test_lexer_hex_escape() {
        let mut lexer = Lexer::new("\\x41\\x42");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('A'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('B'));
    }

    #[test]
    fn test_lexer_unicode_escape() {
        let mut lexer = Lexer::new("\\u00E9");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('é'));
    }

    #[test]
    fn test_lexer_dot() {
        let mut lexer = Lexer::new("a.b");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Dot);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
    }

    #[test]
    fn test_lexer_peek() {
        let mut lexer = Lexer::new("ab");
        assert_eq!(*lexer.peek().unwrap(), Token::Char('a'));
        assert_eq!(*lexer.peek().unwrap(), Token::Char('a')); // Still 'a'
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
    }

    #[test]
    fn test_lexer_whitespace() {
        let mut lexer = Lexer::new("a b c");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('c'));
    }

    // Symbol reference tests

    #[test]
    fn test_lexer_symbol_ref_simple() {
        let mut lexer = Lexer::new("$VOWEL");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::SymbolRef("VOWEL".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::Eof);
    }

    #[test]
    fn test_lexer_symbol_ref_braced() {
        let mut lexer = Lexer::new("${FRONT_VOWEL}");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::SymbolRef("FRONT_VOWEL".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::Eof);
    }

    #[test]
    fn test_lexer_symbol_ref_in_pattern() {
        let mut lexer = Lexer::new("a$VOWEL+");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::SymbolRef("VOWEL".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::Plus);
        assert_eq!(lexer.next_token().unwrap(), Token::Eof);
    }

    #[test]
    fn test_lexer_symbol_ref_in_char_class() {
        let mut lexer = Lexer::new("[$FRONT$BACK]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::SymbolRef("FRONT".to_string())
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::SymbolRef("BACK".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
    }

    #[test]
    fn test_lexer_symbol_ref_braced_adjacent() {
        // Test that ${NAME}x parses correctly
        let mut lexer = Lexer::new("${FRONT}y");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::SymbolRef("FRONT".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::Char('y'));
        assert_eq!(lexer.next_token().unwrap(), Token::Eof);
    }

    #[test]
    fn test_lexer_symbol_ref_empty_name_error() {
        let mut lexer = Lexer::new("$ ");
        let err = lexer.next_token().unwrap_err();
        assert!(
            format!("{:?}", err).contains("symbol name"),
            "Error should mention symbol name"
        );
    }

    #[test]
    fn test_lexer_symbol_ref_empty_braced_error() {
        let mut lexer = Lexer::new("${}");
        let err = lexer.next_token().unwrap_err();
        assert!(
            format!("{:?}", err).contains("empty"),
            "Error should mention empty"
        );
    }

    // Byte-level tests

    #[test]
    fn test_lexer_byte_simple() {
        let mut lexer = LexerByte::new(b"abc");
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Byte(b'a'));
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Byte(b'b'));
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Byte(b'c'));
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Eof);
    }

    #[test]
    fn test_lexer_byte_operators() {
        let mut lexer = LexerByte::new(b"a*b+");
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Byte(b'a'));
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Star);
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Byte(b'b'));
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Plus);
    }

    #[test]
    fn test_lexer_byte_arrow() {
        let mut lexer = LexerByte::new(b"ph -> f");
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Byte(b'p'));
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Byte(b'h'));
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Arrow);
        assert_eq!(lexer.next_token().unwrap(), TokenByte::Byte(b'f'));
    }
}
