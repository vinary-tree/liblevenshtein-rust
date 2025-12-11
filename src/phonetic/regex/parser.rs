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

use std::collections::HashMap;

use super::ast::{
    ContextExpr, ContextExprByte, ContextPredicate, ContextPredicateByte, Regex, RegexByte,
    SyllableCondition, SyllableExpr,
};
use super::error::{ParseError, ParseErrorKind, ParseResult};
use super::lexer::{Lexer, LexerByte, Token, TokenByte};
use crate::phonetic::nfa::types::{CharClass, CharClassChar};

/// Maximum complexity for parsed patterns (prevents DoS).
const MAX_PATTERN_SIZE: usize = 10_000;

/// Symbol table for user-defined character classes.
/// Maps symbol names to their character sets.
pub type SymbolTable = HashMap<String, Vec<char>>;

/// Check if a name represents a user-defined symbol (all uppercase).
///
/// User-defined symbols use UPPERCASE names to distinguish them from
/// built-in classes which use lowercase (e.g., `[:alpha:]`, `[:vowel:]`).
///
/// A name is considered a user symbol if:
/// - It starts with an uppercase letter
/// - All alphabetic characters are uppercase
/// - Non-alphabetic characters (digits, underscores) are allowed after the first character
///
/// # Examples
/// - `"VOWEL"` → true (user symbol)
/// - `"FRONT_V"` → true (user symbol)
/// - `"V2"` → true (user symbol with digit)
/// - `"alpha"` → false (built-in)
/// - `"Vowel"` → false (mixed case, treated as built-in)
/// - `"_FOO"` → false (must start with a letter)
/// - `"123"` → false (must start with a letter)
fn is_user_symbol_name(name: &str) -> bool {
    let mut chars = name.chars();
    // Must start with an uppercase letter
    match chars.next() {
        Some(first) if first.is_uppercase() => {
            // Remaining characters: alphabetic must be uppercase, non-alphabetic are allowed
            chars.all(|c| c.is_uppercase() || !c.is_alphabetic())
        }
        _ => false,
    }
}

/// Compute complement of a character class using printable ASCII.
///
/// This is used for negated named classes like `[^[:vowel:]]`.
/// Returns all printable ASCII characters (0x20-0x7E) that are NOT in the input set.
fn negate_char_class(chars: &[char]) -> Vec<char> {
    (0x20u8..=0x7Eu8)
        .map(|b| b as char)
        .filter(|c| !chars.contains(c))
        .collect()
}

/// Parser for phonetic regular expressions.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    /// Optional symbol table for user-defined symbols ($NAME references)
    symbols: Option<&'a SymbolTable>,
}

impl<'a> Parser<'a> {
    /// Create a new parser for the given input.
    pub fn new(input: &'a str) -> Self {
        Self {
            lexer: Lexer::new(input),
            symbols: None,
        }
    }

    /// Create a new parser with a symbol table for user-defined symbols.
    ///
    /// This allows the regex to reference symbols defined in an LLev grammar
    /// using the `$SYMBOL` syntax.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::collections::HashMap;
    /// use liblevenshtein::phonetic::regex::parser::{Parser, SymbolTable};
    ///
    /// let mut symbols: SymbolTable = HashMap::new();
    /// symbols.insert("VOWEL".to_string(), vec!['a', 'e', 'i', 'o', 'u']);
    ///
    /// let mut parser = Parser::new_with_symbols("[$VOWEL]+", &symbols);
    /// let regex = parser.parse().unwrap();
    /// ```
    pub fn new_with_symbols(input: &'a str, symbols: &'a SymbolTable) -> Self {
        Self {
            lexer: Lexer::new(input),
            symbols: Some(symbols),
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

    /// Parse a primary expression: `(...)`, `[...]`, `.`, `#`, `$SYMBOL`, or literal
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
            Token::SymbolRef(name) => self.expand_symbol_ref(&name),
            Token::Eof => Err(ParseError::unexpected_eof(self.lexer.position())),
            _ => Err(ParseError::unexpected_char(
                self.token_to_char(&token),
                self.lexer.position(),
            )),
        }
    }

    /// Expand a symbol reference to a character class.
    fn expand_symbol_ref(&self, name: &str) -> ParseResult<Regex> {
        let chars = self.get_symbol_chars(name)?;
        Ok(Regex::CharClass(CharClassChar::from_chars(&chars)))
    }

    /// Get the characters for a symbol reference.
    fn get_symbol_chars(&self, name: &str) -> ParseResult<Vec<char>> {
        match &self.symbols {
            Some(symbols) => {
                if let Some(chars) = symbols.get(name) {
                    Ok(chars.clone())
                } else {
                    // Symbol not found - provide helpful error with available symbols
                    let available: Vec<String> = symbols.keys().cloned().collect();
                    Err(ParseError::new(
                        ParseErrorKind::UndefinedSymbol { name: name.to_string(), available },
                        self.lexer.position(),
                    ))
                }
            }
            None => {
                // No symbol table provided
                Err(ParseError::with_context(
                    ParseErrorKind::UndefinedSymbol {
                        name: name.to_string(),
                        available: vec![],
                    },
                    self.lexer.position(),
                    "no symbol table provided to parser",
                ))
            }
        }
    }

    /// Parse a character class: `[abc]`, `[^abc]`, `[a-z]`, `[:NAME:]`, `[[:NAME:]abc]`
    ///
    /// Also supports:
    /// - Inline named classes: `[x[:vowel:]]` - `:NAME:` anywhere in a char class
    /// - Negated nested classes: `[^[:vowel:]]` - negates the named class
    /// - Arbitrary nesting: `[$FRONT[[:BACK:][^[:SYMBOL:]]]]` - all unioned together
    fn parse_char_class(&mut self) -> ParseResult<Regex> {
        let mut chars = Vec::new();
        let mut negated = false;

        // Check for negation
        if self.lexer.peek()? == &Token::Caret {
            self.lexer.next_token()?;
            negated = true;
        }

        // Check for standalone named class syntax: [:NAME:]
        if self.lexer.peek()? == &Token::Char(':') {
            self.lexer.next_token()?; // consume ':'
            return self.parse_standalone_named_class(negated);
        }

        loop {
            let token = self.lexer.next_token()?;

            match token {
                Token::CharClassEnd => break,
                Token::Char(c) => {
                    // Check for nested character class: [[...]] or [^[...]]
                    if c == '[' {
                        // Parse the nested class content
                        let nested_chars = self.parse_nested_char_class_content()?;
                        chars.extend(nested_chars);
                        continue;
                    }

                    // Check for inline named class: :NAME:
                    if c == ':' {
                        if let Some(named_chars) = self.try_parse_inline_named_class()? {
                            chars.extend(named_chars);
                            continue;
                        } else {
                            // Not a valid :NAME: pattern, treat ':' as literal
                            chars.push(':');
                            continue;
                        }
                    }

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
                Token::SymbolRef(name) => {
                    // Expand symbol reference into chars
                    let symbol_chars = self.get_symbol_chars(&name)?;
                    chars.extend(symbol_chars);
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

    /// Parse nested character class content after `[` has been consumed.
    /// Handles `[[:NAME:]]`, `[^[:NAME:]]`, and arbitrary nesting.
    /// Returns the characters to union with the parent class.
    fn parse_nested_char_class_content(&mut self) -> ParseResult<Vec<char>> {
        // Check for negation inside the nested class
        let inner_negated = self.lexer.peek()? == &Token::Char('^');
        if inner_negated {
            self.lexer.next_token()?; // consume '^'
        }

        // Check if this is a named class: [[:NAME:]] or [^[:NAME:]]
        if self.lexer.peek()? == &Token::Char(':') {
            self.lexer.next_token()?; // consume ':'
            let named_chars = self.parse_posix_named_class()?;
            if inner_negated {
                Ok(negate_char_class(&named_chars))
            } else {
                Ok(named_chars)
            }
        } else if self.lexer.peek()? == &Token::Char('[') {
            // Another level of nesting: [[[...]]]
            self.lexer.next_token()?; // consume '['
            let nested_chars = self.parse_nested_char_class_content()?;

            // Expect closing ']' for this level
            let token = self.lexer.next_token()?;
            if token != Token::CharClassEnd {
                return Err(ParseError::with_context(
                    ParseErrorKind::ExpectedChar(']'),
                    self.lexer.position(),
                    "closing nested character class",
                ));
            }
            // Re-enter char class mode for the outer class
            self.lexer.enter_char_class_mode();

            if inner_negated {
                Ok(negate_char_class(&nested_chars))
            } else {
                Ok(nested_chars)
            }
        } else {
            // Just a literal '[' followed by something else - treat '[' as literal
            // But we already consumed '[', so we need to handle this case
            // Actually if we get here, it's not a valid nested class syntax
            // For backwards compatibility, treat it as literal '['
            // Put back what we consumed and return just '['
            let mut result = vec!['['];
            if inner_negated {
                result.push('^');
            }
            Ok(result)
        }
    }

    /// Try to parse inline :NAME: syntax. Called after ':' has been consumed.
    /// Returns Some(chars) if successful, None if not a valid :NAME: pattern.
    fn try_parse_inline_named_class(&mut self) -> ParseResult<Option<Vec<char>>> {
        let mut name = String::new();

        // Collect name characters until we hit ':' or something else
        loop {
            let token = self.lexer.peek()?;
            match token {
                Token::Char(':') => {
                    self.lexer.next_token()?; // consume closing ':'
                    break;
                }
                Token::Char(c) if c.is_alphanumeric() || *c == '_' => {
                    let c = *c;
                    self.lexer.next_token()?;
                    name.push(c);
                }
                _ => {
                    // Not a valid :NAME: pattern
                    // We've already consumed some tokens, but we can't easily backtrack
                    // Return the collected name characters as literals instead
                    if name.is_empty() {
                        return Ok(None);
                    }
                    // This is actually an error - we saw :NAME but no closing :
                    return Err(ParseError::with_context(
                        ParseErrorKind::InvalidCharClass("expected ':' to close named class".to_string()),
                        self.lexer.position(),
                        format!(":{}...", name),
                    ));
                }
            }
        }

        if name.is_empty() {
            // Just "::" - treat as two literal colons
            return Ok(Some(vec![':']));
        }

        // Resolve the name: built-in first, then user symbol
        if let Some(builtin_chars) = crate::phonetic::named_classes::get_chars_only(&name) {
            Ok(Some(builtin_chars))
        } else if is_user_symbol_name(&name) {
            let chars = self.get_symbol_chars(&name)?;
            Ok(Some(chars))
        } else {
            Err(ParseError::new(
                ParseErrorKind::UnknownNamedClass(name),
                self.lexer.position(),
            ))
        }
    }

    /// Parse a standalone named class after `[:` has been consumed: `NAME:]`
    fn parse_standalone_named_class(&mut self, negated: bool) -> ParseResult<Regex> {
        let mut name = String::new();

        // Collect name characters
        loop {
            let token = self.lexer.next_token()?;
            match token {
                Token::Char(':') => break, // End of name
                Token::Char(c) if c.is_alphanumeric() || c == '_' => {
                    name.push(c);
                }
                Token::Eof => {
                    return Err(ParseError::unclosed_char_class(self.lexer.position()));
                }
                _ => {
                    return Err(ParseError::with_context(
                        ParseErrorKind::InvalidCharClass(format!(
                            "invalid character '{}' in named class",
                            self.token_to_char(&token)
                        )),
                        self.lexer.position(),
                        format!("in [:{}...", name),
                    ));
                }
            }
        }

        // Expect closing ']'
        let token = self.lexer.next_token()?;
        if token != Token::CharClassEnd {
            return Err(ParseError::with_context(
                ParseErrorKind::ExpectedChar(']'),
                self.lexer.position(),
                format!("after [:{}:", name),
            ));
        }

        if name.is_empty() {
            return Err(ParseError::new(
                ParseErrorKind::InvalidCharClass("empty named class [::]".to_string()),
                self.lexer.position(),
            ));
        }

        // First check built-in classes (case-insensitive lookup)
        // Then check user-defined symbols if not found and name is uppercase
        let chars = if let Some(builtin_chars) = crate::phonetic::named_classes::get_chars_only(&name) {
            // Built-in class found
            builtin_chars
        } else if is_user_symbol_name(&name) {
            // Not a built-in class, but has uppercase name - treat as user symbol
            self.get_symbol_chars(&name)?
        } else {
            // Unknown built-in class (lowercase name that doesn't exist)
            return Err(ParseError::new(
                ParseErrorKind::UnknownNamedClass(name.clone()),
                self.lexer.position(),
            ));
        };

        let class = if negated {
            CharClassChar::from_chars(&chars).negated()
        } else {
            CharClassChar::from_chars(&chars)
        };
        Ok(Regex::char_class(class))
    }

    /// Parse a POSIX-style named class after `[[:` has been consumed: `NAME:]]`
    /// Returns the characters from the named class (without creating a Regex).
    fn parse_posix_named_class(&mut self) -> ParseResult<Vec<char>> {
        let mut name = String::new();

        // Collect name characters
        loop {
            let token = self.lexer.next_token()?;
            match token {
                Token::Char(':') => break, // End of name
                Token::Char(c) if c.is_alphanumeric() || c == '_' => {
                    name.push(c);
                }
                Token::Eof => {
                    return Err(ParseError::unclosed_char_class(self.lexer.position()));
                }
                _ => {
                    return Err(ParseError::with_context(
                        ParseErrorKind::InvalidCharClass(format!(
                            "invalid character '{}' in named class",
                            self.token_to_char(&token)
                        )),
                        self.lexer.position(),
                        format!("in [[:{}...", name),
                    ));
                }
            }
        }

        // Expect closing ']' for the inner bracket.
        // Note: The lexer returns CharClassEnd for ']' when in char class mode,
        // and exits char class mode. We accept CharClassEnd here and re-enter
        // char class mode so parsing can continue for the outer char class.
        let token = self.lexer.next_token()?;
        if token != Token::CharClassEnd {
            return Err(ParseError::with_context(
                ParseErrorKind::ExpectedChar(']'),
                self.lexer.position(),
                format!("expected ']]' after [[:{}:", name),
            ));
        }

        // Re-enter char class mode since we're still parsing the outer [...]
        self.lexer.enter_char_class_mode();

        if name.is_empty() {
            return Err(ParseError::new(
                ParseErrorKind::InvalidCharClass("empty named class [[::]...]".to_string()),
                self.lexer.position(),
            ));
        }

        // First check built-in classes (case-insensitive lookup)
        // Then check user-defined symbols if not found and name is uppercase
        if let Some(builtin_chars) = crate::phonetic::named_classes::get_chars_only(&name) {
            // Built-in class found
            Ok(builtin_chars)
        } else if is_user_symbol_name(&name) {
            // Not a built-in class, but has uppercase name - treat as user symbol
            self.get_symbol_chars(&name)
        } else {
            // Unknown built-in class (lowercase name that doesn't exist)
            Err(ParseError::new(
                ParseErrorKind::UnknownNamedClass(name.clone()),
                self.lexer.position(),
            ))
        }
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
                | Token::SymbolRef(_)
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
            Token::SymbolRef(_) => '$',
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

    // ========================================================================
    // Named character class tests
    // ========================================================================

    #[test]
    fn test_parse_standalone_named_class_vowel() {
        let r = parse("[:vowel:]").unwrap();
        // Should parse as a character class containing vowels
        assert!(r.to_string().starts_with('['));
        assert!(r.to_string().contains('a'));
        assert!(r.to_string().contains('e'));
        assert!(r.to_string().contains('i'));
        assert!(r.to_string().contains('o'));
        assert!(r.to_string().contains('u'));
    }

    #[test]
    fn test_parse_standalone_named_class_negated() {
        let r = parse("[^:vowel:]").unwrap();
        // Should parse as a negated character class
        assert!(r.to_string().starts_with("[^"));
    }

    #[test]
    fn test_parse_standalone_named_class_full_name() {
        // Use full name since shorthand aliases were removed
        let r = parse("[:vowel:]").unwrap();
        // Should contain vowels
        assert!(r.to_string().contains('a'));
        assert!(r.to_string().contains('e'));
    }

    #[test]
    fn test_parse_standalone_named_class_alpha() {
        let r = parse("[:alpha:]").unwrap();
        // Should contain a-z, A-Z
        let s = r.to_string();
        assert!(s.contains('a'));
        assert!(s.contains('z'));
        assert!(s.contains('A'));
        assert!(s.contains('Z'));
    }

    #[test]
    fn test_parse_standalone_named_class_digit() {
        let r = parse("[:digit:]").unwrap();
        // Should contain 0-9
        let s = r.to_string();
        assert!(s.contains('0'));
        assert!(s.contains('9'));
    }

    #[test]
    fn test_parse_posix_named_class_mixed() {
        let r = parse("[[:vowel:]y]").unwrap();
        // Should contain vowels plus 'y'
        let s = r.to_string();
        assert!(s.contains('a'));
        assert!(s.contains('y'));
    }

    #[test]
    fn test_parse_posix_named_class_multiple() {
        let r = parse("[[:vowel:][:digit:]]").unwrap();
        // Should contain vowels and digits
        let s = r.to_string();
        assert!(s.contains('a'));
        assert!(s.contains('0'));
    }

    #[test]
    fn test_parse_named_class_case_insensitive() {
        let r1 = parse("[:VOWEL:]").unwrap();
        let r2 = parse("[:vowel:]").unwrap();
        // Both should work and produce similar results
        assert!(r1.to_string().contains('a'));
        assert!(r2.to_string().contains('a'));
    }

    #[test]
    fn test_parse_named_class_unknown() {
        let result = parse("[:unknown_class:]");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UnknownNamedClass(_)));
    }

    #[test]
    fn test_parse_named_class_empty() {
        let result = parse("[::]");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_named_class_in_rewrite_rule() {
        let r = parse_rule("c -> s / _[:front_vowel:]").unwrap();
        assert!(r.is_rewrite_rule());
        // The context should contain front vowels
    }

    #[test]
    fn test_parse_named_class_consonant() {
        let r = parse("[:consonant:]").unwrap();
        let s = r.to_string();
        // Should contain consonants
        assert!(s.contains('b'));
        assert!(s.contains('c'));
        assert!(s.contains('d'));
        // Should NOT contain vowels
        // (Actually the string representation shows all chars, so we can't easily test exclusion)
    }

    #[test]
    fn test_parse_named_class_stop() {
        let r = parse("[:stop:]").unwrap();
        let s = r.to_string();
        // Should contain stop consonants
        assert!(s.contains('p'));
        assert!(s.contains('t'));
        assert!(s.contains('k'));
        assert!(s.contains('b'));
        assert!(s.contains('d'));
        assert!(s.contains('g'));
    }

    #[test]
    fn test_parse_literal_bracket_in_char_class() {
        // Make sure [[] still works (literal '[' in class)
        let r = parse("[[ab]").unwrap();
        let s = r.to_string();
        assert!(s.contains('['));
        assert!(s.contains('a'));
        assert!(s.contains('b'));
    }

    // ========================================================================
    // Symbol reference tests
    // ========================================================================

    #[test]
    fn test_parse_symbol_ref_standalone() {
        let mut symbols = SymbolTable::new();
        symbols.insert("VOWEL".to_string(), vec!['a', 'e', 'i', 'o', 'u']);

        let mut parser = Parser::new_with_symbols("$VOWEL", &symbols);
        let r = parser.parse().unwrap();
        let s = r.to_string();
        assert!(s.contains('a'));
        assert!(s.contains('e'));
        assert!(s.contains('i'));
        assert!(s.contains('o'));
        assert!(s.contains('u'));
    }

    #[test]
    fn test_parse_symbol_ref_in_pattern() {
        let mut symbols = SymbolTable::new();
        symbols.insert("VOWEL".to_string(), vec!['a', 'e', 'i', 'o', 'u']);

        let mut parser = Parser::new_with_symbols("$VOWEL+", &symbols);
        let r = parser.parse().unwrap();
        // Pattern should match one or more vowels
        assert!(r.to_string().contains('+'));
    }

    #[test]
    fn test_parse_symbol_ref_in_char_class() {
        let mut symbols = SymbolTable::new();
        symbols.insert("FRONT".to_string(), vec!['e', 'i']);
        symbols.insert("BACK".to_string(), vec!['o', 'u']);

        let mut parser = Parser::new_with_symbols("[$FRONT$BACK]", &symbols);
        let r = parser.parse().unwrap();
        let s = r.to_string();
        assert!(s.contains('e'));
        assert!(s.contains('i'));
        assert!(s.contains('o'));
        assert!(s.contains('u'));
    }

    #[test]
    fn test_parse_symbol_ref_braced() {
        let mut symbols = SymbolTable::new();
        symbols.insert("FRONT_VOWEL".to_string(), vec!['e', 'i']);

        let mut parser = Parser::new_with_symbols("${FRONT_VOWEL}y", &symbols);
        let r = parser.parse().unwrap();
        // Should parse as character class followed by 'y'
        assert!(r.to_string().contains('y'));
    }

    #[test]
    fn test_parse_symbol_ref_undefined_error() {
        let symbols = SymbolTable::new();

        let mut parser = Parser::new_with_symbols("$UNDEFINED", &symbols);
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UndefinedSymbol { .. }));
    }

    #[test]
    fn test_parse_symbol_ref_undefined_with_suggestions() {
        let mut symbols = SymbolTable::new();
        symbols.insert("VOWEL".to_string(), vec!['a', 'e', 'i', 'o', 'u']);
        symbols.insert("CONSONANT".to_string(), vec!['b', 'c', 'd']);

        let mut parser = Parser::new_with_symbols("$UNDEFINED", &symbols);
        let result = parser.parse();
        assert!(result.is_err());
        let err = result.unwrap_err();
        if let ParseErrorKind::UndefinedSymbol { name, available } = &err.kind {
            assert_eq!(name, "UNDEFINED");
            // Available should contain our defined symbols
            assert!(available.contains(&"VOWEL".to_string()) || available.contains(&"CONSONANT".to_string()));
        } else {
            panic!("Expected UndefinedSymbol error");
        }
    }

    #[test]
    fn test_parse_symbol_ref_no_symbols_error() {
        // Using regular parser without symbols
        let result = parse("$VOWEL");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UndefinedSymbol { .. }));
    }

    #[test]
    fn test_parse_symbol_ref_mixed_with_literals() {
        let mut symbols = SymbolTable::new();
        symbols.insert("V".to_string(), vec!['a', 'e', 'i', 'o', 'u']);

        // Use braced form ${V} to separate symbol from following 'z'
        let mut parser = Parser::new_with_symbols("[x${V}z]", &symbols);
        let r = parser.parse().unwrap();
        let s = r.to_string();
        assert!(s.contains('x'));
        assert!(s.contains('a'));
        assert!(s.contains('z'));
    }

    #[test]
    fn test_parse_symbol_ref_simple_form_consumes_alphanum() {
        // Verify that $Vz parses as symbol "Vz", not "V" + "z"
        let mut symbols = SymbolTable::new();
        symbols.insert("V".to_string(), vec!['a', 'e', 'i', 'o', 'u']);

        let mut parser = Parser::new_with_symbols("[$Vz]", &symbols);
        let result = parser.parse();
        // Should fail because "Vz" is not defined, only "V"
        assert!(result.is_err());
        if let ParseErrorKind::UndefinedSymbol { name, .. } = &result.unwrap_err().kind {
            assert_eq!(name, "Vz");
        } else {
            panic!("Expected UndefinedSymbol error");
        }
    }
}
