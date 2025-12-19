//! Lexer for `.llev` rule files.
//!
//! This lexer extends the regex lexer with additional tokens for:
//! - Directives (`@name`, `@include`, `@define`, etc.)
//! - Comments (line `//` and block `/* */`)
//! - Metadata blocks (`[id: 1, name: "..."]`)
//! - String literals (`"..."`)
//! - Identifiers (for symbol references)

use crate::phonetic::common::traits::{LexerLike, TokenLike};
use crate::phonetic::common::syllable::SyllableCondition;
use super::error::{LLevError, LLevErrorKind, LLevResult, Position};

/// A token in the `.llev` file format.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // ==================== Literals ====================
    /// A literal character (in patterns/replacements)
    Char(char),

    /// A string literal (`"..."`)
    String(String),

    /// An identifier (for symbols, metadata keys)
    Identifier(String),

    /// A symbol reference (`$NAME` or `${NAME}`)
    SymbolRef(String),

    /// A number
    Number(usize),

    /// A float number
    Float(f64),

    // ==================== Directives ====================
    /// `@name` directive
    DirectiveName,

    /// `@version` directive
    DirectiveVersion,

    /// `@author` directive
    DirectiveAuthor,

    /// `@description` directive
    DirectiveDescription,

    /// `@include` directive
    DirectiveInclude,

    /// `@define` directive
    DirectiveDefine,

    // ==================== Regex Operators ====================
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
    BraceStart,

    /// End of quantifier `}`
    BraceEnd,

    /// Comma `,`
    Comma,

    // ==================== Rewrite Rule Operators ====================
    /// Arrow `->` or `→`
    Arrow,

    /// Context separator `/`
    Slash,

    /// Context position marker `_`
    Underscore,

    /// Word boundary `#`
    Hash,

    // ==================== Context Logical Operators ====================
    /// Ampersand `&` (AND in context expressions)
    Ampersand,

    /// Bang `!` (NOT in context expressions)
    Bang,

    // ==================== Keywords ====================
    /// `if` keyword (for syllable conditions)
    KeywordIf,

    // ==================== Metadata ====================
    /// Start of metadata block `[`
    MetadataStart,

    /// End of metadata block `]`
    MetadataEnd,

    /// Colon `:` (in metadata)
    Colon,

    // ==================== Punctuation ====================
    /// Assignment `=`
    Equals,

    /// Statement terminator `;`
    Semicolon,

    /// Newline (significant for line-based parsing)
    Newline,

    // ==================== Special ====================
    /// End of input
    Eof,

    // ==================== Phonetic Shortcuts ====================
    /// A phonetic class shortcut (e.g., `\v` for vowel, `\V` for non-vowel)
    /// - `class_name`: The full class name to look up
    /// - `negated`: true for uppercase (negated), false for lowercase (positive)
    PhoneticShortcut { class_name: String, negated: bool },
}

impl Token {
    /// Check if this token is a quantifier operator.
    pub fn is_quantifier(&self) -> bool {
        matches!(
            self,
            Token::Star | Token::Plus | Token::Question | Token::BraceStart
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
                | Token::Identifier(_)
                | Token::SymbolRef(_)
        )
    }

    /// Check if this token is a directive.
    pub fn is_directive(&self) -> bool {
        matches!(
            self,
            Token::DirectiveName
                | Token::DirectiveVersion
                | Token::DirectiveAuthor
                | Token::DirectiveDescription
                | Token::DirectiveInclude
                | Token::DirectiveDefine
        )
    }

    /// Get the directive name (if this is a directive token).
    pub fn directive_name(&self) -> Option<&'static str> {
        match self {
            Token::DirectiveName => Some("name"),
            Token::DirectiveVersion => Some("version"),
            Token::DirectiveAuthor => Some("author"),
            Token::DirectiveDescription => Some("description"),
            Token::DirectiveInclude => Some("include"),
            Token::DirectiveDefine => Some("define"),
            _ => None,
        }
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
        matches!(self, Token::Bang)
    }

    fn is_group_start(&self) -> bool {
        matches!(self, Token::GroupStart)
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
        matches!(self, Token::BraceStart)
    }

    fn is_eof(&self) -> bool {
        matches!(self, Token::Eof)
    }

    fn is_if_keyword(&self) -> bool {
        matches!(self, Token::KeywordIf)
    }

    fn as_syllable_condition(&self) -> Option<SyllableCondition> {
        match self {
            Token::Identifier(name) => match name.as_str() {
                "monosyllable" => Some(SyllableCondition::Monosyllable),
                "polysyllable" => Some(SyllableCondition::Polysyllable),
                "open_syllable" => Some(SyllableCondition::OpenSyllable),
                "closed_syllable" => Some(SyllableCondition::ClosedSyllable),
                "final_syllable" => Some(SyllableCondition::FinalSyllable),
                "initial_syllable" => Some(SyllableCondition::InitialSyllable),
                _ => None,
            },
            _ => None,
        }
    }

    fn can_start_primary(&self) -> bool {
        Token::can_start_primary(self)
    }
}

/// Lexer state for context-sensitive tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LexerState {
    /// Top-level parsing (directives, metadata blocks, etc.)
    TopLevel,
    /// Inside a pattern/replacement expression (characters are literals)
    Pattern,
    /// Inside a character class `[...]`
    CharClass,
    /// Inside a metadata block `[...]`
    Metadata,
}

/// Lexer for `.llev` files.
pub struct Lexer<'a> {
    input: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    position: Position,
    state: LexerState,
    /// Stack of peeked tokens for lookahead
    peeked: Vec<(Token, Position, LexerState)>,
    /// Track if we're at line start (for comment detection)
    at_line_start: bool,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given input.
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.char_indices().peekable(),
            position: Position::start(),
            state: LexerState::Pattern, // Default to pattern mode (chars are literals)
            peeked: Vec::new(),
            at_line_start: true,
        }
    }

    /// Create a new lexer for a full `.llev` file (starts at top-level).
    pub fn new_file(input: &'a str) -> Self {
        Self {
            input,
            chars: input.char_indices().peekable(),
            position: Position::start(),
            state: LexerState::TopLevel,
            peeked: Vec::new(),
            at_line_start: true,
        }
    }

    /// Get the current position in the input.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Peek at the next token without consuming it.
    pub fn peek(&mut self) -> LLevResult<&Token> {
        if self.peeked.is_empty() {
            let state_before = self.state;
            let token = self.next_token_internal()?;
            let pos = self.position;
            let state_after = self.state;
            // Restore state for peeking (we didn't consume the token)
            self.state = state_before;
            self.peeked.push((token, pos, state_after));
        }
        Ok(&self.peeked.last().expect("just pushed").0)
    }

    /// Peek at the next token's position.
    pub fn peek_position(&mut self) -> LLevResult<Position> {
        if self.peeked.is_empty() {
            let state_before = self.state;
            let token = self.next_token_internal()?;
            let pos = self.position;
            let state_after = self.state;
            // Restore state for peeking
            self.state = state_before;
            self.peeked.push((token, pos, state_after));
        }
        Ok(self.peeked.last().expect("just pushed").1)
    }

    /// Get the next token.
    pub fn next_token(&mut self) -> LLevResult<Token> {
        if let Some((token, pos, state)) = self.peeked.pop() {
            self.position = pos;
            self.state = state;
            Ok(token)
        } else {
            self.next_token_internal()
        }
    }

    /// Consume the next token if it matches the expected token.
    pub fn expect(&mut self, expected: &Token) -> LLevResult<Token> {
        let token = self.next_token()?;
        if &token == expected {
            Ok(token)
        } else {
            Err(LLevError::expected_token(
                format!("{:?}", expected),
                format!("{:?}", token),
                self.position.clone(),
            ))
        }
    }

    /// Check if we're at end of input.
    pub fn is_eof(&mut self) -> bool {
        self.skip_whitespace_and_comments();
        self.chars.peek().is_none()
    }

    /// Enter character class mode.
    pub fn enter_char_class(&mut self) {
        self.state = LexerState::CharClass;
    }

    /// Exit character class mode.
    pub fn exit_char_class(&mut self) {
        self.state = LexerState::Pattern;
    }

    /// Enter metadata mode.
    pub fn enter_metadata(&mut self) {
        self.state = LexerState::Metadata;
        self.peeked.clear();
    }

    /// Exit metadata mode.
    pub fn exit_metadata(&mut self) {
        self.state = LexerState::Pattern;
        self.peeked.clear();
    }

    /// Enter pattern mode (characters are literals).
    pub fn enter_pattern(&mut self) {
        self.state = LexerState::Pattern;
        self.peeked.clear();
    }

    /// Enter top-level mode (identifiers, directives).
    pub fn enter_top_level(&mut self) {
        self.state = LexerState::TopLevel;
        self.peeked.clear();
    }

    /// Get the current byte offset in the input.
    pub fn current_offset(&self) -> usize {
        self.position.offset
    }

    /// Get the remaining input from current position.
    ///
    /// Note: This clears any peeked tokens and returns the remaining input
    /// starting at the position where the next token would begin.
    ///
    /// IMPORTANT: This method resets the lexer state to the position of the first
    /// peeked token (if any), so subsequent tokenization will re-lex from that point.
    pub fn remaining_input(&mut self) -> &'a str {
        // If there are peeked tokens, return from the start of the first peeked token.
        // We also need to reset the lexer to re-lex from that position.
        if let Some((_, pos, _)) = self.peeked.first() {
            let offset = pos.offset;
            // Return remaining input from the peeked position
            // Don't skip whitespace - just return the raw remaining input
            // This avoids issues with iterator state
            self.peeked.clear();
            // Reset iterator to start from this offset
            // Use a proper reset that maintains absolute offsets
            self.reset_to_offset(offset);
        }

        // Skip whitespace (and comments if in TopLevel mode) from current position
        if self.state == LexerState::TopLevel {
            self.skip_whitespace_and_comments();
        } else {
            self.skip_whitespace_only();
        }

        // Get the current byte offset and return remaining input
        if let Some(&(idx, _)) = self.chars.peek() {
            &self.input[idx..]
        } else {
            ""
        }
    }

    /// Reset the lexer to start reading from a specific offset in the original input.
    fn reset_to_offset(&mut self, offset: usize) {
        // Re-create the char iterator from the beginning and skip to offset
        self.chars = self.input.char_indices().peekable();
        self.position.offset = 0;
        self.position.line = 1;
        self.position.column = 1;

        // Advance to the desired offset
        while let Some(&(idx, _)) = self.chars.peek() {
            if idx >= offset {
                break;
            }
            self.advance();
        }
    }

    // ==================== Internal Methods ====================

    /// Advance to the next character.
    fn advance(&mut self) -> Option<char> {
        if let Some((offset, c)) = self.chars.next() {
            self.position.offset = offset;
            if c == '\n' {
                self.position.line += 1;
                self.position.column = 1;
                self.at_line_start = true;
            } else {
                self.position.column += 1;
                if !c.is_whitespace() {
                    self.at_line_start = false;
                }
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

    /// Peek at the second next character.
    fn peek_char2(&self) -> Option<char> {
        let mut iter = self.chars.clone();
        iter.next();
        iter.next().map(|(_, c)| c)
    }

    /// Skip whitespace only (not comments).
    /// Used in pattern mode where `#` is word boundary and `/` is context separator.
    fn skip_whitespace_only(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() && c != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Skip whitespace and comments.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace (but not newlines in some contexts)
            while let Some(c) = self.peek_char() {
                if c.is_whitespace() && c != '\n' {
                    self.advance();
                } else {
                    break;
                }
            }

            // Check for comments (// line comments and /* */ block comments)
            match self.peek_char() {
                Some('/') => {
                    match self.peek_char2() {
                        Some('/') => {
                            // Line comment //
                            self.skip_line_comment();
                        }
                        Some('*') => {
                            // Block comment /* */
                            if self.skip_block_comment().is_err() {
                                // Error handled in skip_block_comment
                                break;
                            }
                        }
                        _ => break,
                    }
                }
                _ => break,
            }
        }
    }

    /// Skip a line comment (//).
    fn skip_line_comment(&mut self) {
        while let Some(c) = self.advance() {
            if c == '\n' {
                break;
            }
        }
    }

    /// Skip a block comment (/* */).
    fn skip_block_comment(&mut self) -> LLevResult<()> {
        let start_pos = self.position.clone();

        // Consume /*
        self.advance(); // /
        self.advance(); // *

        let mut depth = 1;
        while depth > 0 {
            match self.advance() {
                Some('*') if self.peek_char() == Some('/') => {
                    self.advance();
                    depth -= 1;
                }
                Some('/') if self.peek_char() == Some('*') => {
                    self.advance();
                    depth += 1;
                }
                Some(_) => {}
                None => {
                    return Err(LLevError::unterminated_comment(start_pos));
                }
            }
        }

        Ok(())
    }

    /// Parse an escape sequence.
    ///
    /// Supports:
    /// - Standard escapes: `\n`, `\r`, `\t`, `\0`, `\\`, `\"`, etc.
    /// - Hex escapes: `\xNN` (2 hex digits)
    /// - Unicode escapes: `\uNNNN` (4 hex digits), `\UNNNNNNNN` (8 hex digits)
    /// - Literal uppercase: `\A`, `\B`, ..., `\Z` for literal uppercase characters
    ///
    /// This allows writing `\A -> a;` to match literal 'A' instead of treating
    /// it as a symbol reference. For `\U`, we check if it's followed by 8 hex
    /// digits (unicode escape) or not (literal 'U').
    fn parse_escape(&mut self) -> LLevResult<char> {
        match self.advance() {
            Some('n') => Ok('\n'),
            Some('r') => Ok('\r'),
            Some('t') => Ok('\t'),
            Some('0') => Ok('\0'),
            Some('\\') => Ok('\\'),
            Some('"') => Ok('"'),
            Some('\'') => Ok('\''),
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
            Some('@') => Ok('@'),
            Some('x') => self.parse_hex_escape(2),
            Some('u') => self.parse_hex_escape(4),
            Some('U') => {
                // Check if followed by 8 hex digits (unicode escape) or not (literal 'U')
                // Peek at the next character to decide
                if self.peek_char().map_or(false, |c| c.is_ascii_hexdigit()) {
                    self.parse_hex_escape(8)
                } else {
                    // Return literal 'U'
                    Ok('U')
                }
            }
            // Uppercase letters (A-Z, except U handled above) become literals
            // This allows \A, \B, etc. for literal uppercase characters
            Some(c) if c.is_ascii_uppercase() => Ok(c),
            Some(c) => Err(LLevError::invalid_escape(c, self.position.clone())),
            None => Err(LLevError::unexpected_eof(self.position.clone())),
        }
    }

    /// Parse a hex escape sequence.
    fn parse_hex_escape(&mut self, num_digits: usize) -> LLevResult<char> {
        let mut value: u32 = 0;
        for _ in 0..num_digits {
            match self.advance() {
                Some(c) if c.is_ascii_hexdigit() => {
                    value = value * 16 + c.to_digit(16).expect("is_ascii_hexdigit");
                }
                Some(c) => {
                    return Err(LLevError::with_context(
                        LLevErrorKind::InvalidEscape(c),
                        format!("expected hex digit, got '{}'", c),
                    )
                    .at_position(self.position.clone()));
                }
                None => {
                    return Err(LLevError::unexpected_eof(self.position.clone()));
                }
            }
        }
        char::from_u32(value).ok_or_else(|| {
            LLevError::new(LLevErrorKind::InvalidCodePoint(value))
                .at_position(self.position.clone())
        })
    }

    /// Parse an escape sequence or phonetic shortcut.
    ///
    /// Phonetic shortcuts provide quick access to named character classes.
    /// Following standard regex convention:
    /// - Lowercase = positive match (e.g., `\v` matches vowels)
    /// - Uppercase = negated match (e.g., `\V` matches non-vowels)
    ///
    /// Available phonetic shortcuts:
    /// - `\v` / `\V` → vowel / non-vowel
    /// - `\c` / `\C` → consonant / non-consonant
    /// - `\f` / `\F` → front_vowel / non-front_vowel
    /// - `\k` / `\K` → back_vowel / non-back_vowel (K for bacK; `\b`/`\B` are word boundary)
    /// - `\h` / `\H` → high_vowel / non-high_vowel
    /// - `\l` / `\L` → low_vowel / non-low_vowel
    /// - `\m` / `\M` → mid_vowel / non-mid_vowel
    /// - `\p` / `\P` → stop/plosive / non-stop (P for Plosive; `\s`/`\S` are whitespace)
    /// - `\g` / `\G` → glide / non-glide
    /// - `\z` / `\Z` → nasal / non-nasal (Z since `\n` is newline)
    /// - `\q` / `\Q` → liquid / non-liquid (Q since L is taken for low_vowel)
    ///
    /// Standard regex escapes are preserved and NOT overridden:
    /// - `\d`, `\D` → digit / non-digit (POSIX)
    /// - `\w`, `\W` → word character / non-word (POSIX)
    /// - `\s`, `\S` → whitespace / non-whitespace (POSIX)
    /// - `\b`, `\B` → word boundary / non-word boundary
    /// - `\n`, `\r`, `\t` → newline, carriage return, tab
    ///
    /// For non-shortcut escapes, falls back to `parse_escape`.
    fn parse_escape_or_shortcut(&mut self) -> LLevResult<Token> {
        // Peek at the next character to check for phonetic shortcuts
        let next = self.peek_char();

        match next {
            // Phonetic class shortcuts
            // v/V - vowel
            Some('v') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "vowel".to_string(), negated: false })
            }
            Some('V') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "vowel".to_string(), negated: true })
            }
            // c/C - consonant
            Some('c') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "consonant".to_string(), negated: false })
            }
            Some('C') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "consonant".to_string(), negated: true })
            }
            // f/F - front_vowel
            Some('f') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "front_vowel".to_string(), negated: false })
            }
            Some('F') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "front_vowel".to_string(), negated: true })
            }
            // k/K - back_vowel (can't use b/B - word boundary)
            Some('k') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "back_vowel".to_string(), negated: false })
            }
            Some('K') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "back_vowel".to_string(), negated: true })
            }
            // h/H - high_vowel
            Some('h') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "high_vowel".to_string(), negated: false })
            }
            Some('H') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "high_vowel".to_string(), negated: true })
            }
            // l/L - low_vowel
            Some('l') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "low_vowel".to_string(), negated: false })
            }
            Some('L') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "low_vowel".to_string(), negated: true })
            }
            // m/M - mid_vowel
            Some('m') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "mid_vowel".to_string(), negated: false })
            }
            Some('M') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "mid_vowel".to_string(), negated: true })
            }
            // p/P - stop/plosive (can't use s/S - whitespace)
            Some('p') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "stop".to_string(), negated: false })
            }
            Some('P') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "stop".to_string(), negated: true })
            }
            // g/G - glide
            Some('g') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "glide".to_string(), negated: false })
            }
            Some('G') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "glide".to_string(), negated: true })
            }
            // z/Z - nasal (can't use n - newline)
            Some('z') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "nasal".to_string(), negated: false })
            }
            Some('Z') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "nasal".to_string(), negated: true })
            }
            // q/Q - liquid (can't use l/L - low_vowel)
            Some('q') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "liquid".to_string(), negated: false })
            }
            Some('Q') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "liquid".to_string(), negated: true })
            }
            // o/O - voiced
            Some('o') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "voiced".to_string(), negated: false })
            }
            Some('O') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "voiced".to_string(), negated: true })
            }
            // e/E - fricative
            Some('e') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "fricative".to_string(), negated: false })
            }
            Some('E') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "fricative".to_string(), negated: true })
            }
            // a/A - affricate
            Some('a') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "affricate".to_string(), negated: false })
            }
            Some('A') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "affricate".to_string(), negated: true })
            }
            // Standard regex class shortcuts
            // d/D - digit
            Some('d') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "digit".to_string(), negated: false })
            }
            Some('D') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "digit".to_string(), negated: true })
            }
            // w/W - word character
            Some('w') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "word".to_string(), negated: false })
            }
            Some('W') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "word".to_string(), negated: true })
            }
            // s/S - whitespace (can't use for stop - uses p/P instead)
            Some('s') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "space".to_string(), negated: false })
            }
            Some('S') => {
                self.advance();
                Ok(Token::PhoneticShortcut { class_name: "space".to_string(), negated: true })
            }
            // Other escapes - pass through to parse_escape
            // \b, \B, \n, \r, \t, etc.
            _ => {
                let escaped = self.parse_escape()?;
                Ok(Token::Char(escaped))
            }
        }
    }

    /// Parse a string literal.
    fn parse_string(&mut self) -> LLevResult<String> {
        let start_pos = self.position.clone();
        let mut result = String::new();

        loop {
            match self.advance() {
                Some('"') => break,
                Some('\\') => {
                    let c = self.parse_escape()?;
                    result.push(c);
                }
                Some('\n') => {
                    return Err(LLevError::unterminated_string(start_pos));
                }
                Some(c) => result.push(c),
                None => {
                    return Err(LLevError::unterminated_string(start_pos));
                }
            }
        }

        Ok(result)
    }

    /// Parse an identifier.
    fn parse_identifier(&mut self, first: char) -> String {
        let mut result = String::new();
        result.push(first);

        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
                result.push(c);
            } else {
                break;
            }
        }

        result
    }

    /// Parse a symbol reference: `$NAME` or `${NAME}`.
    /// Called after `$` has been consumed.
    fn parse_symbol_ref(&mut self) -> LLevResult<Token> {
        let pos = self.position;

        if self.peek_char() == Some('{') {
            // Explicit form: ${NAME}
            self.advance(); // consume '{'

            let mut name = String::new();
            while let Some(c) = self.peek_char() {
                if c == '}' {
                    self.advance(); // consume '}'
                    if name.is_empty() {
                        return Err(LLevError::new(LLevErrorKind::InvalidSymbolName(
                            "empty symbol name in ${...}".to_string(),
                        ))
                        .at_position(pos));
                    }
                    return Ok(Token::SymbolRef(name));
                } else if c.is_alphanumeric() || c == '_' {
                    self.advance();
                    name.push(c);
                } else {
                    return Err(LLevError::new(LLevErrorKind::InvalidPattern(format!(
                        "invalid character '{}' in symbol name",
                        c
                    )))
                    .at_position(self.position));
                }
            }
            // Hit EOF before closing brace
            Err(LLevError::new(LLevErrorKind::InvalidPattern(
                "unclosed ${...} - expected '}'".to_string(),
            ))
            .at_position(pos))
        } else {
            // Simple form: $NAME
            // If not followed by a valid identifier, $ is a literal character
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
                // $ not followed by identifier = literal '$' character
                // This allows [$] to match literal '$' (standard regex convention)
                return Ok(Token::Char('$'));
            }

            Ok(Token::SymbolRef(name))
        }
    }

    /// Parse a number.
    fn parse_number(&mut self, first_digit: char) -> LLevResult<Token> {
        let mut value = first_digit.to_digit(10).expect("is digit") as usize;

        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                self.advance();
                value = value * 10 + c.to_digit(10).expect("is digit") as usize;
            } else {
                break;
            }
        }

        // Check for float
        if self.peek_char() == Some('.') {
            let next = self.peek_char2();
            if next.map_or(false, |c| c.is_ascii_digit()) {
                self.advance(); // consume '.'
                let float_val = self.parse_float_decimal(value as f64)?;
                return Ok(Token::Float(float_val));
            }
        }

        Ok(Token::Number(value))
    }

    /// Parse the decimal part of a float.
    fn parse_float_decimal(&mut self, integer_part: f64) -> LLevResult<f64> {
        let mut value = integer_part;
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

        Ok(value)
    }

    /// Parse a directive (starting with @).
    fn parse_directive(&mut self) -> LLevResult<Token> {
        let mut name = String::new();

        while let Some(c) = self.peek_char() {
            if c.is_alphabetic() || c == '_' {
                self.advance();
                name.push(c);
            } else {
                break;
            }
        }

        match name.as_str() {
            "name" => Ok(Token::DirectiveName),
            "version" => Ok(Token::DirectiveVersion),
            "author" => Ok(Token::DirectiveAuthor),
            "description" => Ok(Token::DirectiveDescription),
            "include" => Ok(Token::DirectiveInclude),
            "define" => Ok(Token::DirectiveDefine),
            _ => Err(LLevError::invalid_directive(name, self.position.clone())),
        }
    }

    /// Internal method to get the next token.
    fn next_token_internal(&mut self) -> LLevResult<Token> {
        // Handle state-specific tokenization
        match self.state {
            LexerState::CharClass => return self.next_token_char_class(),
            LexerState::Metadata => return self.next_token_metadata(),
            LexerState::Pattern => return self.next_token_pattern(),
            LexerState::TopLevel => {} // Continue below
        }

        self.skip_whitespace_and_comments();

        let c = match self.advance() {
            Some(c) => c,
            None => return Ok(Token::Eof),
        };

        match c {
            // String literal
            '"' => {
                let s = self.parse_string()?;
                Ok(Token::String(s))
            }

            // Directive
            '@' => self.parse_directive(),

            // Arrow or dash
            '-' => {
                if self.peek_char() == Some('>') {
                    self.advance();
                    Ok(Token::Arrow)
                } else {
                    Ok(Token::Dash)
                }
            }

            // Unicode arrow
            '→' => Ok(Token::Arrow),

            // Character class or metadata start
            '[' => {
                // We'll let the parser decide which mode to enter
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
            '{' => Ok(Token::BraceStart),
            '}' => Ok(Token::BraceEnd),
            ',' => Ok(Token::Comma),
            '/' => Ok(Token::Slash),
            '_' => Ok(Token::Underscore),
            '#' => Ok(Token::Hash),
            ':' => Ok(Token::Colon),
            '=' => Ok(Token::Equals),
            ';' => Ok(Token::Semicolon),
            '\n' => Ok(Token::Newline),
            '^' => Ok(Token::Caret),

            // Context logical operators
            '&' => Ok(Token::Ampersand),
            '!' => Ok(Token::Bang),

            // Escape sequence or phonetic shortcut
            '\\' => self.parse_escape_or_shortcut(),

            // Number
            c if c.is_ascii_digit() => self.parse_number(c),

            // Identifier (at top-level, any alphabetic sequence is an identifier)
            // "if" becomes KeywordIf
            c if c.is_alphabetic() || c == '_' => {
                let ident = self.parse_identifier(c);
                if ident == "if" {
                    Ok(Token::KeywordIf)
                } else {
                    Ok(Token::Identifier(ident))
                }
            }

            // Other characters are literals
            _ => Ok(Token::Char(c)),
        }
    }

    /// Get the next token inside a pattern expression (characters are literals).
    fn next_token_pattern(&mut self) -> LLevResult<Token> {
        // In pattern mode, only skip whitespace (not comments).
        // `#` is the word boundary marker, not a comment.
        // `/` is the context separator, not part of a comment.
        self.skip_whitespace_only();

        let c = match self.advance() {
            Some(c) => c,
            None => return Ok(Token::Eof),
        };

        match c {
            // String literal (for complex patterns)
            '"' => {
                let s = self.parse_string()?;
                Ok(Token::String(s))
            }

            // Arrow or dash
            '-' => {
                if self.peek_char() == Some('>') {
                    self.advance();
                    Ok(Token::Arrow)
                } else {
                    Ok(Token::Dash)
                }
            }

            // Unicode arrow
            '→' => Ok(Token::Arrow),

            // Character class start
            '[' => Ok(Token::CharClassStart),

            ']' => Ok(Token::CharClassEnd),
            '(' => Ok(Token::GroupStart),
            ')' => Ok(Token::GroupEnd),
            '|' => Ok(Token::Pipe),
            '*' => Ok(Token::Star),
            '+' => Ok(Token::Plus),
            '?' => Ok(Token::Question),
            '.' => Ok(Token::Dot),
            '{' => Ok(Token::BraceStart),
            '}' => Ok(Token::BraceEnd),
            ',' => Ok(Token::Comma),
            '/' => Ok(Token::Slash),
            '_' => Ok(Token::Underscore),
            '#' => Ok(Token::Hash),
            ':' => Ok(Token::Colon),
            '=' => Ok(Token::Equals),
            ';' => Ok(Token::Semicolon),
            '\n' => Ok(Token::Newline),
            '^' => Ok(Token::Caret),

            // Context logical operators
            '&' => Ok(Token::Ampersand),
            '!' => Ok(Token::Bang),

            // Symbol reference: $NAME or ${NAME}
            '$' => self.parse_symbol_ref(),

            // Escape sequence or phonetic shortcut
            '\\' => self.parse_escape_or_shortcut(),

            // Digits are literal characters in patterns (e.g., "2 -> to" for text-speak)
            // Numbers are only parsed in metadata mode (e.g., "[id: 210]")
            c if c.is_ascii_digit() => Ok(Token::Char(c)),

            // Check for "if" keyword
            'i' if self.peek_char() == Some('f') => {
                // Check if "if" is followed by a non-identifier char (space, &, |, etc.)
                let next2 = self.peek_char2();
                if next2.map_or(true, |c| !c.is_alphanumeric() && c != '_') {
                    self.advance(); // consume 'f'
                    Ok(Token::KeywordIf)
                } else {
                    // It's part of a longer identifier like "iffy", treat as literal
                    Ok(Token::Char(c))
                }
            }

            // All other characters (including lowercase letters) are literals
            _ => Ok(Token::Char(c)),
        }
    }

    /// Get the next token inside a character class.
    fn next_token_char_class(&mut self) -> LLevResult<Token> {
        let c = match self.advance() {
            Some(c) => c,
            None => return Ok(Token::Eof),
        };

        match c {
            ']' => {
                self.state = LexerState::Pattern;
                Ok(Token::CharClassEnd)
            }
            // '[' inside a character class signals nested class: [[...]]
            // Note: escaped \[ is handled below and returns Token::Char('[')
            '[' => Ok(Token::CharClassStart),
            '^' => Ok(Token::Caret),
            '-' => Ok(Token::Dash),
            ':' => Ok(Token::Colon),
            // Escape sequences or phonetic shortcuts
            // \[ returns Token::Char('[') (literal bracket)
            // \v returns Token::PhoneticShortcut("vowel")
            '\\' => self.parse_escape_or_shortcut(),
            // $ is a literal character inside character classes (unlike outside)
            _ => Ok(Token::Char(c)),
        }
    }

    /// Get the next token inside a metadata block.
    fn next_token_metadata(&mut self) -> LLevResult<Token> {
        // Skip whitespace
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() && c != '\n' {
                self.advance();
            } else {
                break;
            }
        }

        let c = match self.advance() {
            Some(c) => c,
            None => return Ok(Token::Eof),
        };

        match c {
            ']' => {
                self.state = LexerState::Pattern;
                Ok(Token::MetadataEnd)
            }
            ':' => Ok(Token::Colon),
            ',' => Ok(Token::Comma),
            '"' => {
                let s = self.parse_string()?;
                Ok(Token::String(s))
            }
            c if c.is_ascii_digit() => self.parse_number(c),
            c if c.is_alphabetic() || c == '_' => {
                let ident = self.parse_identifier(c);

                // Handle boolean keywords
                match ident.as_str() {
                    "true" => Ok(Token::Identifier("true".to_string())),
                    "false" => Ok(Token::Identifier("false".to_string())),
                    _ => Ok(Token::Identifier(ident)),
                }
            }
            '\n' => {
                // Newline in metadata - might be an error
                Ok(Token::Newline)
            }
            _ => Err(LLevError::unexpected_char(c, self.position.clone())),
        }
    }
}

impl<'a> LexerLike for Lexer<'a> {
    type Token = Token;
    type Error = LLevError;

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_simple_chars() {
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
    fn test_lexer_arrow() {
        let mut lexer = Lexer::new("ph -> f");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('p'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('h'));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('f'));
    }

    #[test]
    fn test_lexer_unicode_arrow() {
        let mut lexer = Lexer::new("ph → f");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('p'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('h'));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('f'));
    }

    #[test]
    fn test_lexer_string_literal() {
        let mut lexer = Lexer::new("\"hello world\"");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::String("hello world".to_string())
        );
    }

    #[test]
    fn test_lexer_string_with_escapes() {
        let mut lexer = Lexer::new("\"hello\\nworld\"");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::String("hello\nworld".to_string())
        );
    }

    #[test]
    fn test_lexer_directives() {
        // Directives appear at top-level, not inside patterns
        let mut lexer = Lexer::new_file("@name @version @include");
        assert_eq!(lexer.next_token().unwrap(), Token::DirectiveName);
        assert_eq!(lexer.next_token().unwrap(), Token::DirectiveVersion);
        assert_eq!(lexer.next_token().unwrap(), Token::DirectiveInclude);
    }

    #[test]
    fn test_lexer_directive_with_string() {
        // Directives appear at top-level, not inside patterns
        let mut lexer = Lexer::new_file("@name \"English Rules\"");
        assert_eq!(lexer.next_token().unwrap(), Token::DirectiveName);
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::String("English Rules".to_string())
        );
    }

    #[test]
    fn test_lexer_symbol_ref_simple() {
        let mut lexer = Lexer::new("$FRONT_VOWEL");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::SymbolRef("FRONT_VOWEL".to_string())
        );
    }

    #[test]
    fn test_lexer_symbol_ref_braced() {
        let mut lexer = Lexer::new("${FRONT_VOWEL}");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::SymbolRef("FRONT_VOWEL".to_string())
        );
    }

    #[test]
    fn test_dollar_literal_in_char_class() {
        // $ is now a literal character inside character classes
        let mut lexer = Lexer::new("[$abc]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        lexer.enter_char_class();
        assert_eq!(lexer.next_token().unwrap(), Token::Char('$'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('c'));
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
    }

    #[test]
    fn test_lexer_uppercase_now_literal_in_pattern() {
        // After sigil change, bare uppercase is literal, not symbol
        let mut lexer = Lexer::new("FRONT");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('F'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('R'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('O'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('N'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('T'));
    }

    #[test]
    fn test_lexer_number() {
        // Use new_file() for TopLevel mode where numbers are tokenized
        let mut lexer = Lexer::new_file("123");
        assert_eq!(lexer.next_token().unwrap(), Token::Number(123));
    }

    #[test]
    fn test_lexer_float() {
        // Use new_file() for TopLevel mode where floats are tokenized
        let mut lexer = Lexer::new_file("0.15");
        match lexer.next_token().unwrap() {
            Token::Float(f) => assert!((f - 0.15).abs() < 0.001),
            t => panic!("expected Float, got {:?}", t),
        }
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
    }

    #[test]
    fn test_lexer_char_class() {
        let mut lexer = Lexer::new("[aeiou]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);

        // Enter char class mode
        lexer.enter_char_class();

        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('e'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('i'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('o'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('u'));
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
    }

    #[test]
    fn test_lexer_char_class_negated() {
        let mut lexer = Lexer::new("[^abc]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        lexer.enter_char_class();
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
        lexer.enter_char_class();
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Dash);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('z'));
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
    }

    #[test]
    fn test_lexer_hash_is_not_comment() {
        // # is NOT a comment character - it's a word boundary marker (Hash token)
        // Use // for line comments instead
        let mut lexer = Lexer::new_file("a # b");
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("a".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Hash);
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("b".to_string()));
    }

    #[test]
    fn test_lexer_line_comment_double_slash() {
        // Comments are only recognized at top-level, not in patterns
        // Note: The newline is consumed as part of the comment
        let mut lexer = Lexer::new_file("a // comment\nb");
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("a".to_string()));
        // Newline is consumed by skip_line_comment()
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("b".to_string()));
    }

    #[test]
    fn test_lexer_block_comment() {
        // Block comments are only recognized at top-level
        let mut lexer = Lexer::new_file("a /* comment */ b");
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("a".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("b".to_string()));
    }

    #[test]
    fn test_lexer_nested_block_comment() {
        // Block comments are only recognized at top-level
        let mut lexer = Lexer::new_file("a /* outer /* inner */ outer */ b");
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("a".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("b".to_string()));
    }

    #[test]
    fn test_lexer_metadata_mode() {
        let mut lexer = Lexer::new("[id: 1, name: \"test\"]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);

        // Enter metadata mode (parser would do this)
        lexer.state = LexerState::Metadata;

        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Identifier("id".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        assert_eq!(lexer.next_token().unwrap(), Token::Number(1));
        assert_eq!(lexer.next_token().unwrap(), Token::Comma);
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Identifier("name".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::String("test".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::MetadataEnd);
    }

    #[test]
    fn test_lexer_semicolon() {
        let mut lexer = Lexer::new("ph -> f;");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('p'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('h'));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('f'));
        assert_eq!(lexer.next_token().unwrap(), Token::Semicolon);
    }

    #[test]
    fn test_lexer_define_directive() {
        // Directives appear at top-level
        let mut lexer = Lexer::new_file("@define VOWEL = [aeiou]");
        assert_eq!(lexer.next_token().unwrap(), Token::DirectiveDefine);
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Identifier("VOWEL".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::Equals);
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
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
    fn test_lexer_escape_sequences() {
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
    fn test_lexer_word_boundary() {
        let mut lexer = Lexer::new("e -> / _#");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('e'));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Slash);
        assert_eq!(lexer.next_token().unwrap(), Token::Underscore);
        assert_eq!(lexer.next_token().unwrap(), Token::Hash);
    }

    #[test]
    fn test_lexer_groups() {
        let mut lexer = Lexer::new("(ph|f)");
        assert_eq!(lexer.next_token().unwrap(), Token::GroupStart);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('p'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('h'));
        assert_eq!(lexer.next_token().unwrap(), Token::Pipe);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('f'));
        assert_eq!(lexer.next_token().unwrap(), Token::GroupEnd);
    }

    #[test]
    fn test_token_is_directive() {
        assert!(Token::DirectiveName.is_directive());
        assert!(Token::DirectiveInclude.is_directive());
        assert!(!Token::Char('a').is_directive());
    }

    #[test]
    fn test_token_directive_name() {
        assert_eq!(Token::DirectiveName.directive_name(), Some("name"));
        assert_eq!(Token::DirectiveInclude.directive_name(), Some("include"));
        assert_eq!(Token::Char('a').directive_name(), None);
    }

    #[test]
    fn test_token_can_start_primary() {
        assert!(Token::Char('a').can_start_primary());
        assert!(Token::CharClassStart.can_start_primary());
        assert!(Token::GroupStart.can_start_primary());
        assert!(Token::Dot.can_start_primary());
        assert!(Token::Hash.can_start_primary());
        assert!(Token::Identifier("VOWEL".to_string()).can_start_primary());
        assert!(!Token::Star.can_start_primary());
        assert!(!Token::Arrow.can_start_primary());
    }

    #[test]
    fn test_lexer_invalid_directive() {
        // Directives are only parsed at top-level
        let mut lexer = Lexer::new_file("@invalid");
        let result = lexer.next_token();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, LLevErrorKind::InvalidDirective(_)));
    }

    #[test]
    fn test_lexer_unterminated_string() {
        let mut lexer = Lexer::new("\"unterminated");
        let result = lexer.next_token();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, LLevErrorKind::UnterminatedString));
    }

    #[test]
    fn test_lexer_escaped_uppercase() {
        // Test that escaped uppercase letters (not phonetic shortcuts) produce literal character tokens
        // Note: Phonetic shortcuts are: A, C, D, E, F, G, H, K, L, M, O, P, Q, S, V, W, Z
        // So we use non-shortcut letters: B, I, J, N, R, T, X, Y
        let mut lexer = Lexer::new("\\B\\I\\J\\N\\X");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('B'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('I'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('J'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('N'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('X'));
    }

    #[test]
    fn test_lexer_escaped_u_literal() {
        // Test that \U followed by non-hex produces literal 'U'
        let mut lexer = Lexer::new("\\Upper");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('U'));
        // 'pper' should follow as lowercase literals
        assert_eq!(lexer.next_token().unwrap(), Token::Char('p'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('p'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('e'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('r'));
    }

    #[test]
    fn test_lexer_unicode_escape_still_works() {
        // Test that \U followed by hex digits still works as unicode escape
        let mut lexer = Lexer::new("\\U00000041");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('A'));
    }

    #[test]
    fn test_lexer_phonetic_shortcuts() {
        // Test phonetic shortcuts: lowercase = positive, uppercase = negated
        let mut lexer = Lexer::new("\\v\\V\\c\\C");
        // \v = vowel (positive)
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::PhoneticShortcut {
                class_name: "vowel".to_string(),
                negated: false
            }
        );
        // \V = vowel (negated)
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::PhoneticShortcut {
                class_name: "vowel".to_string(),
                negated: true
            }
        );
        // \c = consonant (positive)
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::PhoneticShortcut {
                class_name: "consonant".to_string(),
                negated: false
            }
        );
        // \C = consonant (negated)
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::PhoneticShortcut {
                class_name: "consonant".to_string(),
                negated: true
            }
        );
    }

    #[test]
    fn test_lexer_phonetic_shortcuts_all() {
        // Test all phonetic shortcuts
        let shortcuts = [
            ("\\f", "front_vowel", false),
            ("\\F", "front_vowel", true),
            ("\\k", "back_vowel", false),
            ("\\K", "back_vowel", true),
            ("\\h", "high_vowel", false),
            ("\\H", "high_vowel", true),
            ("\\l", "low_vowel", false),
            ("\\L", "low_vowel", true),
            ("\\m", "mid_vowel", false),
            ("\\M", "mid_vowel", true),
            ("\\p", "stop", false),
            ("\\P", "stop", true),
            ("\\g", "glide", false),
            ("\\G", "glide", true),
            ("\\z", "nasal", false),
            ("\\Z", "nasal", true),
            ("\\q", "liquid", false),
            ("\\Q", "liquid", true),
        ];

        for (input, expected_class, expected_negated) in shortcuts.iter() {
            let mut lexer = Lexer::new(input);
            assert_eq!(
                lexer.next_token().unwrap(),
                Token::PhoneticShortcut {
                    class_name: expected_class.to_string(),
                    negated: *expected_negated
                },
                "Failed for input: {}",
                input
            );
        }
    }

    #[test]
    fn test_lexer_uppercase_pattern_rule() {
        // Test a complete pattern with escaped uppercase (using non-shortcut letter)
        let mut lexer = Lexer::new("\\B -> b");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('B'));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('b'));
    }

    // ========================================================================
    // Tests for compound context operators (Phase 1B)
    // ========================================================================

    #[test]
    fn test_lexer_ampersand() {
        let mut lexer = Lexer::new("[aeiou] & [bcdf]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        lexer.enter_char_class();
        assert_eq!(lexer.next_token().unwrap(), Token::Char('a'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('e'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('i'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('o'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('u'));
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
        assert_eq!(lexer.next_token().unwrap(), Token::Ampersand);
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
    }

    #[test]
    fn test_lexer_bang() {
        let mut lexer = Lexer::new("![aeiou]");
        assert_eq!(lexer.next_token().unwrap(), Token::Bang);
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
    }

    #[test]
    fn test_lexer_if_keyword() {
        let mut lexer = Lexer::new("/ _# if monosyllable");
        assert_eq!(lexer.next_token().unwrap(), Token::Slash);
        assert_eq!(lexer.next_token().unwrap(), Token::Underscore);
        assert_eq!(lexer.next_token().unwrap(), Token::Hash);
        assert_eq!(lexer.next_token().unwrap(), Token::KeywordIf);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('m'));
    }

    #[test]
    fn test_lexer_if_not_keyword_in_word() {
        // "iffy" should not trigger the if keyword
        let mut lexer = Lexer::new("iffy");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('i'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('f'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('f'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('y'));
    }

    #[test]
    fn test_lexer_if_keyword_top_level() {
        // At top-level, "if" becomes KeywordIf identifier
        let mut lexer = Lexer::new_file("if");
        assert_eq!(lexer.next_token().unwrap(), Token::KeywordIf);
    }

    #[test]
    fn test_lexer_compound_context_rule() {
        // Test a rule with compound context: x -> gz / [aeiou]_[aeiou]
        let mut lexer = Lexer::new("x -> gz / [aeiou]_[aeiou] & ![y]");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('x'));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('g'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('z'));
        assert_eq!(lexer.next_token().unwrap(), Token::Slash);
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        lexer.enter_char_class();
        // Skip to end of char class
        while lexer.next_token().unwrap() != Token::CharClassEnd {}
        assert_eq!(lexer.next_token().unwrap(), Token::Underscore);
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        lexer.enter_char_class();
        while lexer.next_token().unwrap() != Token::CharClassEnd {}
        assert_eq!(lexer.next_token().unwrap(), Token::Ampersand);
        assert_eq!(lexer.next_token().unwrap(), Token::Bang);
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
    }

    #[test]
    fn test_lexer_syllable_condition_rule() {
        // Test a rule with syllable condition: y -> i / _# if monosyllable
        let mut lexer = Lexer::new("y -> i / _# if monosyllable;");
        assert_eq!(lexer.next_token().unwrap(), Token::Char('y'));
        assert_eq!(lexer.next_token().unwrap(), Token::Arrow);
        assert_eq!(lexer.next_token().unwrap(), Token::Char('i'));
        assert_eq!(lexer.next_token().unwrap(), Token::Slash);
        assert_eq!(lexer.next_token().unwrap(), Token::Underscore);
        assert_eq!(lexer.next_token().unwrap(), Token::Hash);
        assert_eq!(lexer.next_token().unwrap(), Token::KeywordIf);
        // "monosyllable" as lowercase chars
        assert_eq!(lexer.next_token().unwrap(), Token::Char('m'));
    }

    #[test]
    fn test_lexer_colon_in_char_class() {
        // Test that ':' is recognized as Token::Colon in char class mode
        // The full POSIX syntax [[:NAME:]] is parsed by the parser, not the lexer
        let mut lexer = Lexer::new("[[:VOWEL:]]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);

        // Enter char class mode
        lexer.enter_char_class();

        // '[' inside char class mode returns CharClassStart (signals nested class)
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        // ':' is recognized as Token::Colon
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        // Characters of the name
        assert_eq!(lexer.next_token().unwrap(), Token::Char('V'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('O'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('W'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('E'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('L'));
        // Second ':' ends the name
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        // ']' ends inner named class reference
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
        // We're now back in pattern mode, second ']' is NOT CharClassEnd
    }

    #[test]
    fn test_lexer_mixed_char_class_with_named() {
        // Test mixed syntax: [[:VOWEL:]xyz]
        // The lexer just tokenizes; the parser handles named class resolution
        let mut lexer = Lexer::new("[[:VOWEL:]xyz]");
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);

        lexer.enter_char_class();

        // '[' inside char class mode returns CharClassStart (signals nested class)
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassStart);
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        // Name characters
        assert_eq!(lexer.next_token().unwrap(), Token::Char('V'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('O'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('W'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('E'));
        assert_eq!(lexer.next_token().unwrap(), Token::Char('L'));
        // End delimiter
        assert_eq!(lexer.next_token().unwrap(), Token::Colon);
        // This ']' closes the named class reference and exits char class mode
        assert_eq!(lexer.next_token().unwrap(), Token::CharClassEnd);
    }
}
