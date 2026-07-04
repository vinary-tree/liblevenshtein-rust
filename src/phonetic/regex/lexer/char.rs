//! Char-level lexer for phonetic regular expressions (full Unicode + phonetic features).

use crate::phonetic::common::flags::ParsedFlags;
use crate::phonetic::common::syllable::SyllableCondition;
use crate::phonetic::common::traits::{LexerLike, TokenLike};
use crate::phonetic::regex::error::{ParseError, ParseErrorKind, ParseResult, Position};

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

    /// Start of capturing group `(`
    GroupStart,

    /// End of group `)`
    GroupEnd,

    /// Start of non-capturing group `(?:`
    NonCapturingGroupStart,

    /// Start of named group `(?<name>` - contains the group name
    NamedGroupStart(String),

    /// Group reference (subroutine call) `(?&name)` - contains the group name
    GroupReference(String),

    /// Inline flags `(?i)` - applies to rest of current scope
    InlineFlags(ParsedFlags),

    /// Scoped flags start `(?i:` - contains flags, pattern follows until `)`
    ScopedFlagsStart(ParsedFlags),

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

    /// Phonetic/standard class shortcut (\v for vowel, \d for digit, etc.)
    /// Lowercase = match class, Uppercase = negate class
    /// Uses interned &'static str for zero allocation
    PhoneticShortcut {
        /// Interned class name to look up (e.g. `"v"` for vowel).
        class_name: &'static str,
        /// `true` for uppercase (negated), `false` for lowercase (positive).
        negated: bool,
    },

    /// User-defined symbol reference ($NAME or ${NAME})
    SymbolRef(String),

    /// Start of line anchor `^` (outside char class)
    StartOfLine,

    /// End of line anchor `$` (outside char class)
    EndOfLine,

    /// Start of input anchor `\A`
    StartOfInput,

    /// End of input anchor `\Z` (allows trailing newline)
    EndOfInput,

    /// Strict end of input anchor `\z` (no trailing newline)
    EndOfInputStrict,

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
                | Token::NonCapturingGroupStart
                | Token::NamedGroupStart(_)
                | Token::GroupReference(_)
                | Token::InlineFlags(_)
                | Token::ScopedFlagsStart(_)
                | Token::Dot
                | Token::Hash
                | Token::SymbolRef(_)
                | Token::PhoneticShortcut { .. }
                | Token::StartOfLine
                | Token::EndOfLine
                | Token::StartOfInput
                | Token::EndOfInput
                | Token::EndOfInputStrict
        )
    }
}

impl TokenLike for Token {
    fn is_pipe(&self) -> bool {
        matches!(self, Token::Pipe)
    }

    fn is_ampersand(&self) -> bool {
        matches!(self, Token::Ampersand)
    }

    fn is_exclamation(&self) -> bool {
        matches!(self, Token::Exclamation)
    }

    fn is_group_start(&self) -> bool {
        matches!(
            self,
            Token::GroupStart
                | Token::NonCapturingGroupStart
                | Token::NamedGroupStart(_)
                | Token::ScopedFlagsStart(_)
        )
    }

    fn is_group_end(&self) -> bool {
        matches!(self, Token::GroupEnd)
    }

    fn is_hash(&self) -> bool {
        matches!(self, Token::Hash)
    }

    fn is_star(&self) -> bool {
        matches!(self, Token::Star)
    }

    fn is_plus(&self) -> bool {
        matches!(self, Token::Plus)
    }

    fn is_question(&self) -> bool {
        matches!(self, Token::Question)
    }

    fn is_brace_start(&self) -> bool {
        matches!(self, Token::QuantifierStart)
    }

    fn is_eof(&self) -> bool {
        matches!(self, Token::Eof)
    }

    fn is_if_keyword(&self) -> bool {
        matches!(self, Token::IfKeyword)
    }

    fn as_syllable_condition(&self) -> Option<SyllableCondition> {
        match self {
            Token::Monosyllable => Some(SyllableCondition::Monosyllable),
            Token::Polysyllable => Some(SyllableCondition::Polysyllable),
            Token::OpenSyllable => Some(SyllableCondition::OpenSyllable),
            Token::ClosedSyllable => Some(SyllableCondition::ClosedSyllable),
            Token::FinalSyllable => Some(SyllableCondition::FinalSyllable),
            Token::InitialSyllable => Some(SyllableCondition::InitialSyllable),
            _ => None,
        }
    }

    fn can_start_primary(&self) -> bool {
        Token::can_start_primary(self)
    }
}

fn ascii_digit_value(c: char) -> Option<u8> {
    if !c.is_ascii_digit() {
        return None;
    }

    c.to_digit(10).and_then(|digit| u8::try_from(digit).ok())
}

/// Lexer for phonetic regex patterns.
pub struct Lexer<'a> {
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
            self.peeked
                .push((token, pos, new_in_char_class, new_in_weight));
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
            Token::GroupStart
            | Token::NonCapturingGroupStart
            | Token::NamedGroupStart(_)
            | Token::ScopedFlagsStart(_) => '(',
            Token::GroupEnd => ')',
            Token::GroupReference(_) => '(',
            Token::InlineFlags(_) => '(',
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
            Token::PhoneticShortcut { .. } => '\\', // escape sequence
            Token::SymbolRef(_) => '$',
            Token::StartOfLine => '^',
            Token::EndOfLine => '$',
            Token::StartOfInput => '\\',     // \A
            Token::EndOfInput => '\\',       // \Z
            Token::EndOfInputStrict => '\\', // \z
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
        char::from_u32(value)
            .ok_or_else(|| ParseError::new(ParseErrorKind::InvalidCodePoint(value), self.position))
    }

    /// Parse an escape sequence, returning a shortcut token for class escapes.
    ///
    /// Handles:
    /// - Phonetic shortcuts: `\v`/`\V` (vowel), `\c`/`\C` (consonant), etc.
    /// - Standard regex shortcuts: `\d`/`\D` (digit), `\w`/`\W` (word), `\s`/`\S` (space)
    /// - Literal escapes fall back to `parse_escape`
    fn parse_escape_or_shortcut(&mut self) -> ParseResult<Token> {
        let next = self.peek_char();

        match next {
            // Standard regex class shortcuts
            Some('d') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "digit",
                    negated: false,
                })
            }
            Some('D') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "digit",
                    negated: true,
                })
            }
            Some('w') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "word",
                    negated: false,
                })
            }
            Some('W') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "word",
                    negated: true,
                })
            }
            Some('s') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "space",
                    negated: false,
                })
            }
            Some('S') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "space",
                    negated: true,
                })
            }
            // Phonetic class shortcuts
            // v/V - vowel
            Some('v') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "vowel",
                    negated: false,
                })
            }
            Some('V') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "vowel",
                    negated: true,
                })
            }
            // c/C - consonant
            Some('c') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "consonant",
                    negated: false,
                })
            }
            Some('C') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "consonant",
                    negated: true,
                })
            }
            // f/F - front_vowel
            Some('f') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "front_vowel",
                    negated: false,
                })
            }
            Some('F') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "front_vowel",
                    negated: true,
                })
            }
            // k/K - back_vowel (can't use b/B - word boundary)
            Some('k') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "back_vowel",
                    negated: false,
                })
            }
            Some('K') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "back_vowel",
                    negated: true,
                })
            }
            // h/H - high_vowel
            Some('h') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "high_vowel",
                    negated: false,
                })
            }
            Some('H') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "high_vowel",
                    negated: true,
                })
            }
            // l/L - low_vowel
            Some('l') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "low_vowel",
                    negated: false,
                })
            }
            Some('L') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "low_vowel",
                    negated: true,
                })
            }
            // m/M - mid_vowel
            Some('m') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "mid_vowel",
                    negated: false,
                })
            }
            Some('M') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "mid_vowel",
                    negated: true,
                })
            }
            // p/P - stop/plosive (can't use s/S - whitespace)
            Some('p') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "stop",
                    negated: false,
                })
            }
            Some('P') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "stop",
                    negated: true,
                })
            }
            // g/G - glide
            Some('g') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "glide",
                    negated: false,
                })
            }
            Some('G') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "glide",
                    negated: true,
                })
            }
            // z - nasal (inside char class) or EndOfInputStrict anchor (outside)
            Some('z') => {
                self.advance();
                // \z is EndOfInputStrict anchor outside char class, nasal inside
                if self.in_char_class {
                    Ok(Token::PhoneticShortcut {
                        class_name: "nasal",
                        negated: false,
                    })
                } else {
                    Ok(Token::EndOfInputStrict)
                }
            }
            // q/Q - liquid (can't use l/L - low_vowel)
            Some('q') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "liquid",
                    negated: false,
                })
            }
            Some('Q') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "liquid",
                    negated: true,
                })
            }
            // o/O - voiced
            Some('o') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "voiced",
                    negated: false,
                })
            }
            Some('O') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "voiced",
                    negated: true,
                })
            }
            // e/E - fricative
            Some('e') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "fricative",
                    negated: false,
                })
            }
            Some('E') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "fricative",
                    negated: true,
                })
            }
            // a/A - affricate (inside char class) or anchor (outside)
            Some('a') => {
                self.advance();
                Ok(Token::PhoneticShortcut {
                    class_name: "affricate",
                    negated: false,
                })
            }
            Some('A') => {
                self.advance();
                // \A is StartOfInput anchor outside char class, affricate negation inside
                if self.in_char_class {
                    Ok(Token::PhoneticShortcut {
                        class_name: "affricate",
                        negated: true,
                    })
                } else {
                    Ok(Token::StartOfInput)
                }
            }
            // Z/z - nasal (inside char class) or anchor (outside)
            Some('Z') => {
                self.advance();
                // \Z is EndOfInput anchor outside char class, nasal negation inside
                if self.in_char_class {
                    Ok(Token::PhoneticShortcut {
                        class_name: "nasal",
                        negated: true,
                    })
                } else {
                    Ok(Token::EndOfInput)
                }
            }
            // Default: fall back to standard escape handling
            _ => {
                let escaped = self.parse_escape()?;
                Ok(Token::Char(escaped))
            }
        }
    }

    /// Parse a number.
    fn parse_number(&mut self, first_digit: char) -> ParseResult<usize> {
        let Some(first_digit) = ascii_digit_value(first_digit) else {
            return Err(ParseError::unexpected_char(first_digit, self.position));
        };
        let mut value = usize::from(first_digit);
        while let Some(c) = self.peek_char() {
            if let Some(digit) = ascii_digit_value(c) {
                self.advance();
                let digit = usize::from(digit);
                value = value
                    .checked_mul(10)
                    .and_then(|value| value.checked_add(digit))
                    .ok_or_else(|| {
                        ParseError::with_context(
                            ParseErrorKind::InvalidQuantifier(
                                "numeric literal exceeds usize::MAX".to_string(),
                            ),
                            self.position,
                            "numeric literal is too large",
                        )
                    })?;
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
                if let Some(digit) = ascii_digit_value(c) {
                    self.advance();
                    value += f64::from(digit) * decimal_place;
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
                            ParseErrorKind::InvalidCharClass(
                                "empty symbol name in ${...}".to_string(),
                            ),
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
                        ParseErrorKind::InvalidCharClass(format!(
                            "invalid character '{}' in symbol name",
                            c
                        )),
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
        // Note: $ is a literal character inside character classes (unlike outside)
        if self.in_char_class {
            return match c {
                ']' => {
                    self.in_char_class = false;
                    Ok(Token::CharClassEnd)
                }
                '^' => Ok(Token::Caret),
                '-' => Ok(Token::Dash),
                '\\' => self.parse_escape_or_shortcut(),
                _ => Ok(Token::Char(c)), // $ falls through here as literal
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
                // The parser distinguishes character classes from rewrite-rule
                // weight suffixes based on syntactic context.
                self.in_char_class = true;
                Ok(Token::CharClassStart)
            }
            ']' => Ok(Token::CharClassEnd),
            '(' => {
                // Check for special group syntax (?...)
                if self.peek_char() == Some('?') {
                    self.advance(); // consume '?'
                    self.parse_special_group()
                } else {
                    Ok(Token::GroupStart)
                }
            }
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
            '\\' => self.parse_escape_or_shortcut(),
            '&' => Ok(Token::Ampersand),
            '!' => Ok(Token::Exclamation),
            '^' => Ok(Token::StartOfLine), // Anchor outside char class
            '$' => {
                // $ is EndOfLine anchor unless followed by identifier (then SymbolRef)
                match self.peek_char() {
                    Some(c) if c.is_alphanumeric() || c == '_' || c == '{' => {
                        self.parse_symbol_ref()
                    }
                    _ => Ok(Token::EndOfLine),
                }
            }
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

    /// Parse a special group `(?...)` after consuming `(?`.
    ///
    /// Handles:
    /// - `(?:` - non-capturing group
    /// - `(?<name>` - named group (PCRE/JS style)
    /// - `(?&name)` - group reference (subroutine call)
    /// - `(?i)` - inline flags
    /// - `(?i:` - scoped flags
    fn parse_special_group(&mut self) -> ParseResult<Token> {
        let pos = self.position;

        match self.peek_char() {
            Some(':') => {
                // Non-capturing group (?:
                self.advance();
                Ok(Token::NonCapturingGroupStart)
            }
            Some('<') => {
                // Named group (?<name>
                self.advance();
                self.parse_named_group_start()
            }
            Some('&') => {
                // Group reference (?&name)
                self.advance();
                self.parse_group_reference()
            }
            Some(c)
                if c == 'i'
                    || c == 'm'
                    || c == 's'
                    || c == 'u'
                    || c == 'f'
                    || c == 'a'
                    || c == '-'
                    || c == ';' =>
            {
                // Flags: (?i), (?m), (?s), (?-i), (?i:...), (?u:NFC:...), (?;N), (?i;N:...), etc.
                self.parse_flags_group()
            }
            Some(c) => Err(ParseError::with_context(
                ParseErrorKind::InvalidGroupSyntax(format!("unexpected '{}' after '(?'", c)),
                pos,
                "expected ':', '<', '&', or flag (i, m, s, u, f, a, -, ;N)",
            )),
            None => Err(ParseError::with_context(
                ParseErrorKind::UnexpectedEof,
                pos,
                "unexpected end of input after '(?'",
            )),
        }
    }

    /// Parse a named group start `(?<name>` after consuming `(?<`.
    fn parse_named_group_start(&mut self) -> ParseResult<Token> {
        let pos = self.position;
        let mut name = String::new();

        // Collect the group name
        while let Some(c) = self.peek_char() {
            if c == '>' {
                self.advance();
                if name.is_empty() {
                    return Err(ParseError::with_context(
                        ParseErrorKind::InvalidGroupName("empty group name".to_string()),
                        pos,
                        "group name cannot be empty",
                    ));
                }
                return Ok(Token::NamedGroupStart(name));
            } else if c.is_ascii_alphanumeric() || c == '_' {
                self.advance();
                name.push(c);
            } else {
                return Err(ParseError::with_context(
                    ParseErrorKind::InvalidGroupName(format!(
                        "invalid character '{}' in group name",
                        c
                    )),
                    self.position,
                    "group names must be alphanumeric or underscore",
                ));
            }
        }

        Err(ParseError::with_context(
            ParseErrorKind::UnclosedGroup,
            pos,
            "unclosed named group - expected '>'",
        ))
    }

    /// Parse a group reference `(?&name)` after consuming `(?&`.
    fn parse_group_reference(&mut self) -> ParseResult<Token> {
        let pos = self.position;
        let mut name = String::new();

        // Collect the group name
        while let Some(c) = self.peek_char() {
            if c == ')' {
                self.advance();
                if name.is_empty() {
                    return Err(ParseError::with_context(
                        ParseErrorKind::InvalidGroupReference("empty group reference".to_string()),
                        pos,
                        "group reference name cannot be empty",
                    ));
                }
                return Ok(Token::GroupReference(name));
            } else if c.is_ascii_alphanumeric() || c == '_' {
                self.advance();
                name.push(c);
            } else {
                return Err(ParseError::with_context(
                    ParseErrorKind::InvalidGroupReference(format!(
                        "invalid character '{}' in group reference",
                        c
                    )),
                    self.position,
                    "group reference names must be alphanumeric or underscore",
                ));
            }
        }

        Err(ParseError::with_context(
            ParseErrorKind::UnclosedGroup,
            pos,
            "unclosed group reference - expected ')'",
        ))
    }

    /// Parse flags group `(?flags)` or `(?flags:...)` after consuming `(?`.
    ///
    /// Also supports local Levenshtein distance syntax:
    /// - `(?i;0:pattern)` - case-insensitive with max 0 edits
    /// - `(?;2:pattern)` - distance only (2 edits)
    /// - `(?i;N)` - inline flags + distance
    /// - `(?;N)` - inline distance only
    fn parse_flags_group(&mut self) -> ParseResult<Token> {
        let pos = self.position;
        let mut flags = ParsedFlags::default();
        let mut negating = false;

        // Parse flags until we see ':' (scoped) or ')' (inline)
        loop {
            match self.peek_char() {
                Some(':') => {
                    // Scoped flags (?flags:...)
                    self.advance();
                    return Ok(Token::ScopedFlagsStart(flags));
                }
                Some(')') => {
                    // Inline flags (?flags)
                    self.advance();
                    return Ok(Token::InlineFlags(flags));
                }
                Some('-') => {
                    // Start negating all following flags until ')' or ':'
                    self.advance();
                    negating = true;
                }
                Some(';') => {
                    // Local Levenshtein distance: (?flags;N...) or (?;N...)
                    self.advance();
                    let distance = self.parse_levenshtein_distance()?;
                    flags.levenshtein_distance = Some(distance);
                    // After parsing distance, must be ':' (scoped) or ')' (inline)
                }
                Some('i') => {
                    self.advance();
                    flags.case_insensitive = Some(!negating);
                    // Note: Don't reset negating - it applies to all following flags
                }
                Some('f') => {
                    self.advance();
                    flags.feature_based = Some(!negating);
                }
                Some('a') => {
                    self.advance();
                    flags.accent_insensitive = Some(!negating);
                }
                Some('m') => {
                    // Multiline mode: ^ and $ match line boundaries
                    self.advance();
                    flags.multiline = Some(!negating);
                }
                Some('s') => {
                    // Dotall (single-line) mode: . matches newlines
                    self.advance();
                    flags.dotall = Some(!negating);
                }
                Some('u') => {
                    // Unicode normalization: (?u:NFC) or (?u:NFC:...)
                    self.advance();
                    if self.peek_char() == Some(':') {
                        self.advance();
                        let norm_form = self.parse_normalization_form()?;
                        flags.unicode_normalization = Some(norm_form);
                    } else {
                        return Err(ParseError::with_context(
                            ParseErrorKind::InvalidFlag(
                                "expected ':' after 'u' for normalization form".to_string(),
                            ),
                            self.position,
                            "use (?u:NFC:...) or (?u:NFD:...) syntax",
                        ));
                    }
                }
                Some(c) => {
                    return Err(ParseError::with_context(
                        ParseErrorKind::InvalidFlag(format!("unknown flag '{}'", c)),
                        self.position,
                        "valid flags are: i (case-insensitive), m (multiline), s (dotall), f (feature-based), a (accent-insensitive), u:NFC/NFD/NFKC/NFKD, ;N (distance)",
                    ));
                }
                None => {
                    return Err(ParseError::with_context(
                        ParseErrorKind::UnclosedGroup,
                        pos,
                        "unclosed flags group - expected ':' or ')'",
                    ));
                }
            }
        }
    }

    /// Parse a local Levenshtein distance value (digits after `;`).
    fn parse_levenshtein_distance(&mut self) -> ParseResult<u8> {
        let pos = self.position;
        let mut digits = String::new();

        // Collect digits
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                self.advance();
                digits.push(c);
            } else {
                break;
            }
        }

        if digits.is_empty() {
            return Err(ParseError::with_context(
                ParseErrorKind::InvalidFlag("expected distance number after ';'".to_string()),
                pos,
                "use (?;N) or (?flags;N:pattern) syntax where N is a number (0-255)",
            ));
        }

        digits.parse::<u8>().map_err(|_| {
            ParseError::with_context(
                ParseErrorKind::InvalidFlag(format!("distance '{}' out of range", digits)),
                pos,
                "distance must be a number between 0 and 255",
            )
        })
    }

    /// Parse a Unicode normalization form (NFC, NFD, NFKC, NFKD).
    fn parse_normalization_form(&mut self) -> ParseResult<String> {
        let pos = self.position;
        let mut form = String::new();

        // Collect uppercase letters for the normalization form
        while let Some(c) = self.peek_char() {
            if c.is_ascii_uppercase() {
                self.advance();
                form.push(c);
            } else {
                break;
            }
        }

        // Validate the form
        match form.as_str() {
            "NFC" | "NFD" | "NFKC" | "NFKD" => Ok(form),
            "" => Err(ParseError::with_context(
                ParseErrorKind::InvalidFlag("missing normalization form after 'u:'".to_string()),
                pos,
                "expected NFC, NFD, NFKC, or NFKD",
            )),
            _ => Err(ParseError::with_context(
                ParseErrorKind::InvalidFlag(format!("unknown normalization form '{}'", form)),
                pos,
                "expected NFC, NFD, NFKC, or NFKD",
            )),
        }
    }

    /// Enter weight parsing mode (after seeing a rewrite rule).
    pub fn enter_weight_mode(&mut self) {
        self.in_char_class = false;
        self.in_weight = true;
    }

    /// Check if the current position looks like a weight start `[digit` or `[-digit`.
    /// This is used to distinguish weights `[0.15]` from character classes `[abc]`.
    pub fn is_at_weight_start(&mut self) -> bool {
        // Save current state
        let saved_chars = self.chars.clone();
        self.skip_whitespace();

        // Check for '[' followed by digit or '-' digit
        let result = if self.peek_char() == Some('[') {
            // Peek ahead without consuming
            let mut chars_copy = self.chars.clone();
            chars_copy.next(); // skip '['
            if let Some((_, c)) = chars_copy.peek() {
                c.is_ascii_digit()
                    || (*c == '-' && {
                        chars_copy.next();
                        chars_copy
                            .peek()
                            .map(|(_, c)| c.is_ascii_digit())
                            .unwrap_or(false)
                    })
            } else {
                false
            }
        } else {
            false
        };

        // Restore state
        self.chars = saved_chars;
        result
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

impl<'a> LexerLike for Lexer<'a> {
    type Token = Token;
    type Error = ParseError;

    fn peek(&mut self) -> Result<&Self::Token, Self::Error> {
        Lexer::peek(self)
    }

    fn advance(&mut self) -> Result<Self::Token, Self::Error> {
        Lexer::next_token(self)
    }

    fn position(&self) -> Position {
        Lexer::position(self)
    }
}

#[cfg(test)]
mod digit_conversion_tests {
    use super::*;

    #[test]
    fn ascii_digit_value_accepts_only_ascii_decimal_digits() {
        assert_eq!(ascii_digit_value('0'), Some(0));
        assert_eq!(ascii_digit_value('9'), Some(9));
        assert_eq!(ascii_digit_value('a'), None);
        assert_eq!(ascii_digit_value('٣'), None);
    }

    #[test]
    fn parse_float_uses_checked_ascii_digit_values() {
        let mut lexer = Lexer::new(".75x");
        let value = lexer
            .parse_float(0)
            .expect("test: decimal float parsing must succeed");
        assert!((value - 0.75).abs() < f64::EPSILON);
        assert_eq!(
            lexer
                .next_token()
                .expect("test: remaining token must be readable"),
            Token::Char('x')
        );
    }
}
