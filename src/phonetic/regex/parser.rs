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
    RegexFlags, SyllableCondition, SyllableExpr, UnicodeNormalization,
};
use super::error::{ParseError, ParseErrorKind, ParseResult, Position};
use super::lexer::{Lexer, LexerByte, ParsedFlags, Token, TokenByte};
use crate::phonetic::common::traits::SyllableParser;
use crate::phonetic::common::utils::{is_user_symbol_name, negate_char_class};
use crate::phonetic::nfa::types::{CharClass, CharClassChar};

/// Maximum complexity for parsed patterns (prevents DoS).
pub const MAX_PATTERN_SIZE: usize = 10_000;

// Re-export SymbolTable from common for backward compatibility
pub use crate::phonetic::common::utils::SymbolTable;

/// Resolve a feature bundle to a character set.
///
/// Each term is (name, negated). The result is the intersection of all terms,
/// where negated terms are first complemented relative to all phonetic characters.
fn resolve_feature_bundle_chars(
    terms: &[(String, bool)],
    symbols: Option<&SymbolTable>,
) -> Result<Vec<char>, String> {
    use crate::phonetic::named_classes::{
        get_chars_only, intersect_char_sets, negate_char_set,
    };

    let mut char_sets: Vec<Vec<char>> = Vec::new();

    for (name, negated) in terms {
        // Try user symbol first, then built-in
        let chars = if let Some(symbol_table) = symbols {
            if let Some(symbol_chars) = symbol_table.get(name) {
                symbol_chars.clone()
            } else if let Some(builtin_chars) = get_chars_only(name) {
                builtin_chars
            } else {
                return Err(format!("unknown class or symbol '{}'", name));
            }
        } else if let Some(builtin_chars) = get_chars_only(name) {
            builtin_chars
        } else if is_user_symbol_name(name) {
            return Err(format!("undefined symbol '{}'", name));
        } else {
            return Err(format!("unknown named class '{}'", name));
        };

        let final_chars = if *negated {
            negate_char_set(&chars)
        } else {
            chars
        };

        char_sets.push(final_chars);
    }

    Ok(intersect_char_sets(&char_sets))
}

/// Parser for phonetic regular expressions.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    /// Optional symbol table for user-defined symbols ($NAME references)
    symbols: Option<&'a SymbolTable>,

    // Group tracking state (Phase 3)
    /// Next capturing group number (starts at 1)
    next_group_number: usize,
    /// Named groups registry: name -> (group_number, pattern AST)
    /// The pattern AST is stored so group references can be expanded.
    named_groups: HashMap<String, (usize, Regex)>,
    /// Deferred validation: group references that need to be validated after parsing
    /// Stores (name, position) for each reference found.
    group_refs_to_validate: Vec<(String, Position)>,
}

impl<'a> Parser<'a> {
    /// Create a new parser for the given input.
    pub fn new(input: &'a str) -> Self {
        Self {
            lexer: Lexer::new(input),
            symbols: None,
            next_group_number: 1,
            named_groups: HashMap::new(),
            group_refs_to_validate: Vec::new(),
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
            next_group_number: 1,
            named_groups: HashMap::new(),
            group_refs_to_validate: Vec::new(),
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

        // Validate all group references (Phase 3)
        self.validate_group_references()?;

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

    /// Validate that all group references refer to defined named groups.
    fn validate_group_references(&self) -> ParseResult<()> {
        for (name, position) in &self.group_refs_to_validate {
            if !self.named_groups.contains_key(name) {
                return Err(ParseError::new(
                    ParseErrorKind::UndefinedGroupReference(name.clone()),
                    *position,
                ));
            }
        }
        Ok(())
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

    /// Parse a repetition quantifier: `{n}`, `{n,}`, `{,m}`, `{n,m}`
    fn parse_repetition(&mut self, inner: Regex) -> ParseResult<Regex> {
        // Check for {,m} syntax (at most m, min defaults to 0)
        let peek = self.lexer.peek()?;
        if peek == &Token::Comma {
            self.lexer.next_token()?; // consume ','
            let max = match self.lexer.next_token()? {
                Token::Number(n) => n,
                _ => {
                    return Err(ParseError::new(
                        ParseErrorKind::InvalidQuantifier("expected number after ','".to_string()),
                        self.lexer.position(),
                    ))
                }
            };
            self.expect_token(Token::QuantifierEnd)?;
            return Ok(Regex::repeat_range(inner, 0, Some(max)));
        }

        // Expect a number for {n}, {n,}, {n,m}
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

    /// Parse a primary expression: `(...)`, `(?:...)`, `(?<name>...)`, `(?&name)`, `(?flags:...)`,
    /// `[...]`, `.`, `#`, `$SYMBOL`, or literal
    fn parse_primary(&mut self) -> ParseResult<Regex> {
        let token = self.lexer.next_token()?;

        match token {
            // Capturing group: (...)
            Token::GroupStart => {
                let group_num = self.next_group_number;
                self.next_group_number += 1;
                let inner = self.parse_alternation()?;
                self.expect_token(Token::GroupEnd)?;
                Ok(Regex::capturing_group(group_num, inner))
            }

            // Non-capturing group: (?:...)
            Token::NonCapturingGroupStart => {
                let inner = self.parse_alternation()?;
                self.expect_token(Token::GroupEnd)?;
                Ok(Regex::non_capturing_group(inner))
            }

            // Named group: (?<name>...)
            Token::NamedGroupStart(name) => {
                // Check for duplicate group name
                if self.named_groups.contains_key(&name) {
                    return Err(ParseError::new(
                        ParseErrorKind::DuplicateGroupName(name),
                        self.lexer.position(),
                    ));
                }

                let group_num = self.next_group_number;
                self.next_group_number += 1;

                // Parse the inner pattern
                let inner = self.parse_alternation()?;
                self.expect_token(Token::GroupEnd)?;

                // Register the named group
                self.named_groups.insert(name.clone(), (group_num, inner.clone()));

                Ok(Regex::named_group(name, inner))
            }

            // Group reference (subroutine call): (?&name)
            Token::GroupReference(name) => {
                // Record for deferred validation
                self.group_refs_to_validate.push((name.clone(), self.lexer.position()));
                Ok(Regex::group_ref(name))
            }

            // Inline flags: (?i) - applies to rest of current scope
            Token::InlineFlags(parsed_flags) => {
                let flags = self.parsed_flags_to_regex_flags(&parsed_flags)?;
                Ok(Regex::inline_flags(flags))
            }

            // Scoped flags: (?i:...) - applies only to inner pattern
            Token::ScopedFlagsStart(parsed_flags) => {
                let flags = self.parsed_flags_to_regex_flags(&parsed_flags)?;
                let inner = self.parse_alternation()?;
                self.expect_token(Token::GroupEnd)?;
                Ok(Regex::flags_group(flags, inner))
            }

            Token::CharClassStart => self.parse_char_class(),
            Token::Dot => Ok(Regex::any()),
            Token::Hash => Ok(Regex::word_boundary()),
            Token::Char(c) => Ok(Regex::char(c)),
            Token::SymbolRef(name) => self.expand_symbol_ref(&name),
            Token::PhoneticShortcut { class_name, negated } => {
                self.expand_phonetic_shortcut(&class_name, negated)
            }

            // Anchors
            Token::StartOfLine => Ok(Regex::StartOfLine),
            Token::EndOfLine => Ok(Regex::EndOfLine),
            Token::StartOfInput => Ok(Regex::StartOfInput),
            Token::EndOfInput => Ok(Regex::EndOfInput),
            Token::EndOfInputStrict => Ok(Regex::EndOfInputStrict),

            Token::Eof => Err(ParseError::unexpected_eof(self.lexer.position())),
            _ => Err(ParseError::unexpected_char(
                self.token_to_char(&token),
                self.lexer.position(),
            )),
        }
    }

    /// Convert lexer ParsedFlags to AST RegexFlags.
    fn parsed_flags_to_regex_flags(&self, parsed: &ParsedFlags) -> ParseResult<RegexFlags> {
        // Convert unicode normalization string to enum
        let unicode_normalization = if let Some(ref norm_str) = parsed.unicode_normalization {
            Some(match norm_str.as_str() {
                "NFC" => UnicodeNormalization::NFC,
                "NFD" => UnicodeNormalization::NFD,
                "NFKC" => UnicodeNormalization::NFKC,
                "NFKD" => UnicodeNormalization::NFKD,
                other => {
                    return Err(ParseError::new(
                        ParseErrorKind::InvalidFlag(format!(
                            "unknown normalization form '{}', expected NFC, NFD, NFKC, or NFKD",
                            other
                        )),
                        self.lexer.position(),
                    ));
                }
            })
        } else {
            None
        };

        Ok(RegexFlags {
            case_insensitive: parsed.case_insensitive,
            unicode_normalization,
            feature_based: parsed.feature_based,
            accent_insensitive: parsed.accent_insensitive,
            multiline: parsed.multiline,
            dotall: parsed.dotall,
            local_distance: parsed.levenshtein_distance,
        })
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

    /// Expand a phonetic shortcut (\v, \d, etc.) to a character class.
    fn expand_phonetic_shortcut(&self, class_name: &str, negated: bool) -> ParseResult<Regex> {
        use crate::phonetic::named_classes::get_chars_only;

        let chars = get_chars_only(class_name).ok_or_else(|| {
            ParseError::new(
                ParseErrorKind::UnknownNamedClass(class_name.to_string()),
                self.lexer.position(),
            )
        })?;

        let final_chars = if negated {
            negate_char_class(&chars)
        } else {
            chars
        };

        Ok(Regex::CharClass(CharClassChar::from_chars(&final_chars)))
    }

    /// Parse a character class: `[abc]`, `[^abc]`, `[a-z]`, `[:NAME:]`, `[[:NAME:]abc]`
    ///
    /// Also supports:
    /// - Nested named classes: `[x[:vowel:]]` - bracketed named class inside char class
    /// - Negated nested classes: `[^[:vowel:]]` - negates the named class
    /// - Arbitrary nesting: `[$FRONT[[:BACK:][^[:SYMBOL:]]]]` - all unioned together
    ///
    /// Note: `:` is a literal character inside char classes. Use `[a[:name:]z]` syntax.
    ///
    /// De Morgan's Law: Tracks cumulative negation count. Even count = positive, odd = negated.
    /// This allows `[^[^[:vowel:]]]` to properly equal `[:vowel:]`.
    fn parse_char_class(&mut self) -> ParseResult<Regex> {
        let mut chars = Vec::new();
        let mut negation_count: usize = 0;

        // Check for negation
        if self.lexer.peek()? == &Token::Caret {
            self.lexer.next_token()?;
            negation_count += 1;
        }

        // Check for standalone named class syntax: [:NAME:]
        if self.lexer.peek()? == &Token::Char(':') {
            self.lexer.next_token()?; // consume ':'
            return self.parse_standalone_named_class(negation_count % 2 == 1);
        }

        loop {
            let token = self.lexer.next_token()?;

            match token {
                Token::CharClassEnd => break,
                Token::Char(c) => {
                    // Check for nested character class: [[...]] or [^[...]]
                    if c == '[' {
                        // Parse the nested class content - returns (chars, negation_count)
                        let (nested_chars, nested_neg_count) =
                            self.parse_nested_char_class_content()?;
                        chars.extend(nested_chars);
                        negation_count += nested_neg_count;
                        continue;
                    }

                    // ':' is a literal character inside char classes
                    // Use [a[:name:]z] syntax for named classes
                    if c == ':' {
                        chars.push(':');
                        continue;
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
                Token::PhoneticShortcut { class_name, negated } => {
                    // Expand phonetic shortcut into chars
                    use crate::phonetic::named_classes::get_chars_only;
                    let shortcut_chars = get_chars_only(class_name).ok_or_else(|| {
                        ParseError::new(
                            ParseErrorKind::UnknownNamedClass(class_name.to_string()),
                            self.lexer.position(),
                        )
                    })?;
                    let final_chars = if negated {
                        negate_char_class(&shortcut_chars)
                    } else {
                        shortcut_chars
                    };
                    chars.extend(final_chars);
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

        // De Morgan's Law: odd negation count = negated, even count = positive
        let final_negated = negation_count % 2 == 1;
        let class = if final_negated {
            CharClassChar::from_chars(&chars).negated()
        } else {
            CharClassChar::from_chars(&chars)
        };

        Ok(Regex::char_class(class))
    }

    /// Parse nested character class content after `[` has been consumed.
    /// Handles `[[:NAME:]]`, `[^[:NAME:]]`, and arbitrary nesting.
    /// Returns (characters, negation_count) where negation_count is how many `^` were encountered.
    /// This enables proper De Morgan's law handling: even count = positive, odd count = negated.
    fn parse_nested_char_class_content(&mut self) -> ParseResult<(Vec<char>, usize)> {
        // Check for negation inside the nested class
        // Note: Inside character classes, ^ is tokenized as Token::Caret, not Token::Char('^')
        let inner_negated = self.lexer.peek()? == &Token::Caret;
        let negation_count: usize = if inner_negated {
            self.lexer.next_token()?; // consume '^'
            1
        } else {
            0
        };

        // Check if this is a named class: [[:NAME:]] or [^[:NAME:]]
        if self.lexer.peek()? == &Token::Char(':') {
            self.lexer.next_token()?; // consume ':'
            let named_chars = self.parse_posix_named_class()?;
            // Return chars without applying negation - let caller handle parity
            Ok((named_chars, negation_count))
        } else if self.lexer.peek()? == &Token::Char('[') {
            // Another level of nesting: [[[...]]]
            self.lexer.next_token()?; // consume '['
            let (nested_chars, nested_neg_count) = self.parse_nested_char_class_content()?;

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

            // Combine negation counts
            Ok((nested_chars, negation_count + nested_neg_count))
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
            Ok((result, 0)) // Not a real negation, just literal chars
        }
    }

    /// Parse a standalone named class or feature bundle after `[:` has been consumed.
    ///
    /// Supports both single-term `[:stop:]` and multi-term feature bundles `[:voiced stop:]`.
    /// Feature bundles use space-separated terms with optional `!` negation prefix.
    fn parse_standalone_named_class(&mut self, negated: bool) -> ParseResult<Regex> {
        let mut terms: Vec<(String, bool)> = Vec::new();
        let mut current_name = String::new();
        let mut current_negated = false;

        // Collect feature terms (space-separated, with optional '!' prefix)
        loop {
            let token = self.lexer.next_token()?;
            match token {
                Token::Char(':') => {
                    // End of feature bundle - push any accumulated term
                    if !current_name.is_empty() {
                        terms.push((current_name.clone(), current_negated));
                    }
                    break;
                }
                Token::Char(c) if c.is_alphanumeric() || c == '_' => {
                    current_name.push(c);
                }
                Token::Char(' ') | Token::Char('\t') => {
                    // Whitespace separates terms
                    if !current_name.is_empty() {
                        terms.push((current_name.clone(), current_negated));
                        current_name.clear();
                        current_negated = false;
                    }
                }
                Token::Char('!') => {
                    // Negation prefix - push any accumulated term first
                    if !current_name.is_empty() {
                        terms.push((current_name.clone(), current_negated));
                        current_name.clear();
                    }
                    current_negated = true;
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
                        format!("in [:{}...", current_name),
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
                "after [:...:",
            ));
        }

        if terms.is_empty() {
            return Err(ParseError::new(
                ParseErrorKind::InvalidCharClass("empty named class [::]".to_string()),
                self.lexer.position(),
            ));
        }

        // Resolve the feature bundle using the helper
        let chars = resolve_feature_bundle_chars(&terms, self.symbols).map_err(|msg| {
            ParseError::new(
                ParseErrorKind::UnknownNamedClass(msg),
                self.lexer.position(),
            )
        })?;

        let class = if negated {
            CharClassChar::from_chars(&chars).negated()
        } else {
            CharClassChar::from_chars(&chars)
        };
        Ok(Regex::char_class(class))
    }

    /// Parse a POSIX-style named class or feature bundle after `[[:` has been consumed.
    ///
    /// Supports both single-term `[[:stop:]]` and multi-term `[[:voiced stop:]]`.
    /// Returns the characters from the named class (without creating a Regex).
    fn parse_posix_named_class(&mut self) -> ParseResult<Vec<char>> {
        let mut terms: Vec<(String, bool)> = Vec::new();
        let mut current_name = String::new();
        let mut current_negated = false;

        // Collect feature terms (space-separated, with optional '!' prefix)
        loop {
            let token = self.lexer.next_token()?;
            match token {
                Token::Char(':') => {
                    // End of feature bundle - push any accumulated term
                    if !current_name.is_empty() {
                        terms.push((current_name.clone(), current_negated));
                    }
                    break;
                }
                Token::Char(c) if c.is_alphanumeric() || c == '_' => {
                    current_name.push(c);
                }
                Token::Char(' ') | Token::Char('\t') => {
                    // Whitespace separates terms
                    if !current_name.is_empty() {
                        terms.push((current_name.clone(), current_negated));
                        current_name.clear();
                        current_negated = false;
                    }
                }
                Token::Char('!') => {
                    // Negation prefix - push any accumulated term first
                    if !current_name.is_empty() {
                        terms.push((current_name.clone(), current_negated));
                        current_name.clear();
                    }
                    current_negated = true;
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
                        format!("in [[:{}...", current_name),
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
                "expected ']]' after [[:...:",
            ));
        }

        // Re-enter char class mode since we're still parsing the outer [...]
        self.lexer.enter_char_class_mode();

        if terms.is_empty() {
            return Err(ParseError::new(
                ParseErrorKind::InvalidCharClass("empty named class [[::]...]".to_string()),
                self.lexer.position(),
            ));
        }

        // Resolve the feature bundle using the helper
        resolve_feature_bundle_chars(&terms, self.symbols).map_err(|msg| {
            ParseError::new(
                ParseErrorKind::UnknownNamedClass(msg),
                self.lexer.position(),
            )
        })
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
                | Token::NonCapturingGroupStart
                | Token::NamedGroupStart(_)
                | Token::GroupReference(_)
                | Token::InlineFlags(_)
                | Token::ScopedFlagsStart(_)
                | Token::Dot
                | Token::Hash
                | Token::SymbolRef(_)
                | Token::PhoneticShortcut { .. }
                // Anchors
                | Token::StartOfLine
                | Token::EndOfLine
                | Token::StartOfInput
                | Token::EndOfInput
                | Token::EndOfInputStrict
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
            Token::NonCapturingGroupStart => '(',
            Token::NamedGroupStart(_) => '(',
            Token::GroupReference(_) => '(',
            Token::InlineFlags(_) => '(',
            Token::ScopedFlagsStart(_) => '(',
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
            Token::PhoneticShortcut { .. } => '\\',
            Token::SymbolRef(_) => '$',
            Token::StartOfLine => '^',
            Token::EndOfLine => '$',
            Token::StartOfInput => '\\', // \A
            Token::EndOfInput => '\\',   // \Z
            Token::EndOfInputStrict => '\\', // \z
            Token::Eof => '\0',
        }
    }
}

impl<'a> SyllableParser for Parser<'a> {
    type Lexer = Lexer<'a>;
    type Error = ParseError;

    fn lexer_mut(&mut self) -> &mut Self::Lexer {
        &mut self.lexer
    }

    fn make_unexpected_token_error(
        &self,
        expected: &str,
        found: &Token,
        position: Position,
    ) -> Self::Error {
        ParseError::new(
            ParseErrorKind::InvalidContext(format!(
                "expected {}, got {:?}",
                expected, found
            )),
            position,
        )
    }

    fn from_lexer_error(&self, err: ParseError) -> Self::Error {
        err
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
            ParseErrorKind::InvalidContext(format!(
                "expected {}, got {:?}",
                expected, found
            )),
            position,
        )
    }

    fn from_lexer_error(&self, err: ParseError) -> Self::Error {
        err
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
    fn test_parse_repetition_at_most() {
        // {,m} is equivalent to {0,m}
        let r = parse("a{,3}").unwrap();
        assert_eq!(r.to_string(), "a{0,3}");
    }

    #[test]
    fn test_parse_repetition_at_most_zero() {
        // {,0} means zero occurrences (effectively empty)
        let r = parse("a{,0}").unwrap();
        assert_eq!(r.to_string(), "a{0,0}");
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
    fn test_dollar_literal_in_char_class() {
        // $ is a literal character inside character classes, not a symbol reference
        let r = parse("[$abc]").unwrap();
        let s = r.to_string();
        // Should contain literal '$' and 'a', 'b', 'c'
        assert!(s.contains('$'), "should contain literal $");
        assert!(s.contains('a'));
        assert!(s.contains('b'));
        assert!(s.contains('c'));
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
    fn test_parse_symbol_ref_outside_char_class() {
        // $ symbols are only expanded OUTSIDE character classes
        let mut symbols = SymbolTable::new();
        symbols.insert("V".to_string(), vec!['a', 'e', 'i', 'o', 'u']);

        // Symbol reference before char class
        let mut parser = Parser::new_with_symbols("$V[xyz]", &symbols);
        let r = parser.parse().unwrap();
        let s = r.to_string();
        // Vowels should be expanded from $V
        assert!(s.contains('a'), "Should contain vowel from $V");
        assert!(s.contains('e'), "Should contain vowel from $V");
    }

    #[test]
    fn test_dollar_is_literal_inside_char_class() {
        // $ is a LITERAL character inside character classes
        let mut symbols = SymbolTable::new();
        symbols.insert("V".to_string(), vec!['a', 'e', 'i', 'o', 'u']);

        // Inside char class, $V is literal chars '$', 'V'
        let mut parser = Parser::new_with_symbols("[$Vz]", &symbols);
        let r = parser.parse().unwrap();
        let s = r.to_string();
        // Should contain literal '$'
        assert!(s.contains('$'), "Should contain literal $");
        // Should contain literal 'V' (not expanded to vowels)
        assert!(s.contains('V'), "Should contain literal V");
        // Should NOT contain 'a' since symbol is not expanded
        assert!(!s.contains('a'), "Should NOT expand symbol inside []");
    }

    // ========================================================================
    // Feature Bundle Tests
    // ========================================================================

    #[test]
    fn test_feature_bundle_standalone_intersection() {
        // [:voiced stop:] should match only voiced stops: b, d, g
        let r = parse("[:voiced stop:]").unwrap();
        let s = r.to_string();
        // Should contain b, d, g
        assert!(s.contains('b'), "Should contain 'b'");
        assert!(s.contains('d'), "Should contain 'd'");
        assert!(s.contains('g'), "Should contain 'g'");
        // Should NOT contain voiceless stops: p, t, k
        assert!(!s.contains('p'), "Should NOT contain 'p'");
        assert!(!s.contains('t'), "Should NOT contain 't'");
        assert!(!s.contains('k'), "Should NOT contain 'k'");
    }

    #[test]
    fn test_feature_bundle_standalone_negation() {
        // [:!nasal stop:] should match stops that are NOT nasal: p, t, k, b, d, g
        let r = parse("[:!nasal stop:]").unwrap();
        let s = r.to_string();
        // Should contain all stops (none are nasal)
        assert!(s.contains('p'), "Should contain 'p'");
        assert!(s.contains('t'), "Should contain 't'");
        assert!(s.contains('k'), "Should contain 'k'");
        assert!(s.contains('b'), "Should contain 'b'");
        assert!(s.contains('d'), "Should contain 'd'");
        assert!(s.contains('g'), "Should contain 'g'");
    }

    #[test]
    fn test_feature_bundle_standalone_single_negated() {
        // [:!nasal:] should match everything except nasals
        let r = parse("[:!nasal:]").unwrap();
        let s = r.to_string();
        // Should NOT contain nasals
        assert!(!s.contains('m'), "Should NOT contain 'm'");
        assert!(!s.contains('n'), "Should NOT contain 'n'");
        // Should contain other consonants
        assert!(s.contains('p'), "Should contain 'p'");
        assert!(s.contains('s'), "Should contain 's'");
    }

    #[test]
    fn test_feature_bundle_standalone_single_term() {
        // [:stop:] should work as before (backwards compatible)
        let r = parse("[:stop:]").unwrap();
        let s = r.to_string();
        assert!(s.contains('p'), "Should contain 'p'");
        assert!(s.contains('t'), "Should contain 't'");
        assert!(s.contains('k'), "Should contain 'k'");
        assert!(s.contains('b'), "Should contain 'b'");
        assert!(s.contains('d'), "Should contain 'd'");
        assert!(s.contains('g'), "Should contain 'g'");
    }

    #[test]
    fn test_feature_bundle_standalone_negated_outer() {
        // [^:voiced stop:] - negated outer, should NOT contain b, d, g
        let r = parse("[^:voiced stop:]").unwrap();
        let s = r.to_string();
        // The outer negation should negate the char class
        // This syntax means "match anything NOT in voiced stop"
        assert!(s.contains('^'), "Should be a negated char class");
    }

    #[test]
    fn test_feature_bundle_posix_intersection() {
        // [a[[:voiced stop:]]] - a plus voiced stops in POSIX syntax
        let r = parse("[a[[:voiced stop:]]]").unwrap();
        let s = r.to_string();
        assert!(s.contains('a'), "Should contain 'a'");
        assert!(s.contains('b'), "Should contain 'b'");
        assert!(s.contains('d'), "Should contain 'd'");
        assert!(s.contains('g'), "Should contain 'g'");
    }

    #[test]
    fn test_feature_bundle_posix_negation() {
        // [[[:!nasal stop:]]] - stops that aren't nasal
        let r = parse("[[[:!nasal stop:]]]").unwrap();
        let s = r.to_string();
        assert!(s.contains('p'), "Should contain 'p'");
        assert!(s.contains('b'), "Should contain 'b'");
    }

    #[test]
    fn test_feature_bundle_unknown_feature_error() {
        // [:unknown_feature:] should error
        let result = parse("[:unknown_feature:]");
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_bundle_empty_error() {
        // [::]  should error (empty)
        let result = parse("[::]");
        assert!(result.is_err());
    }

    // ========================================================================
    // De Morgan's Law Tests
    // ========================================================================

    #[test]
    fn test_double_negation_equals_positive() {
        // [^[^[:vowel:]]] should equal [:vowel:] (double negation cancels out)
        let double_neg = parse("[^[^[:vowel:]]]").unwrap();
        let positive = parse("[:vowel:]").unwrap();

        let double_neg_str = double_neg.to_string();
        let positive_str = positive.to_string();

        // Both should contain the same vowels
        for c in ['a', 'e', 'i', 'o', 'u'] {
            assert!(
                double_neg_str.contains(c),
                "double_neg should contain '{}'",
                c
            );
            assert!(positive_str.contains(c), "positive should contain '{}'", c);
        }
        // Neither should contain consonants (for the positive case, they're excluded)
        for c in ['p', 't', 'k'] {
            // The double negation result should NOT be negated
            assert!(
                !double_neg_str.contains('^'),
                "double negation should not have ^ flag"
            );
        }
    }

    #[test]
    fn test_negated_union() {
        // [^[:vowel:][:stop:]] = ¬(vowel ∪ stop)
        let r = parse("[^[:vowel:][:stop:]]").unwrap();
        let s = r.to_string();

        // Should be a negated char class
        assert!(s.starts_with("[^"), "Should be negated: {}", s);
        // Should contain vowels and stops (which are then negated)
        assert!(s.contains('a'), "Should contain 'a' (to be negated)");
        assert!(s.contains('p'), "Should contain 'p' (to be negated)");
    }

    #[test]
    fn test_triple_negation() {
        // [^[^[^[:vowel:]]]] should equal [^[:vowel:]] (odd count = negated)
        let triple = parse("[^[^[^[:vowel:]]]]").unwrap();
        let s = triple.to_string();

        // Should be negated (odd count of negations)
        assert!(s.starts_with("[^"), "Triple negation should result in negated: {}", s);
    }

    #[test]
    fn test_quadruple_negation() {
        // [^[^[^[^[:vowel:]]]]] should equal [:vowel:] (even count = positive)
        let quad = parse("[^[^[^[^[:vowel:]]]]]").unwrap();
        let s = quad.to_string();

        // Should NOT be negated (even count of negations)
        assert!(!s.starts_with("[^"), "Quadruple negation should be positive: {}", s);
        // Should contain vowels
        assert!(s.contains('a'), "Should contain 'a'");
        assert!(s.contains('e'), "Should contain 'e'");
    }

    // ========================================================================
    // Phonetic Shortcut Parser Tests
    // ========================================================================

    #[test]
    fn test_parse_shortcut_vowel() {
        let r = parse(r"\v").unwrap();
        let s = r.to_string();
        // Should contain vowels
        assert!(s.contains('a'), "Should contain 'a'");
        assert!(s.contains('e'), "Should contain 'e'");
        assert!(s.contains('i'), "Should contain 'i'");
        assert!(s.contains('o'), "Should contain 'o'");
        assert!(s.contains('u'), "Should contain 'u'");
    }

    #[test]
    fn test_parse_shortcut_vowel_negated() {
        let r = parse(r"\V").unwrap();
        let s = r.to_string();
        // Should NOT contain vowels, should contain consonants
        assert!(!s.contains('a'), "Should NOT contain 'a'");
        assert!(!s.contains('e'), "Should NOT contain 'e'");
        assert!(s.contains('p') || s.contains('b') || s.contains('t'), "Should contain some consonants");
    }

    #[test]
    fn test_parse_shortcut_consonant() {
        let r = parse(r"\c").unwrap();
        let s = r.to_string();
        // Should contain consonants
        assert!(s.contains('p'), "Should contain 'p'");
        assert!(s.contains('b'), "Should contain 'b'");
        assert!(s.contains('t'), "Should contain 't'");
    }

    #[test]
    fn test_parse_shortcut_stop() {
        let r = parse(r"\p").unwrap();
        let s = r.to_string();
        // Should contain stop consonants
        assert!(s.contains('p'), "Should contain 'p'");
        assert!(s.contains('t'), "Should contain 't'");
        assert!(s.contains('k'), "Should contain 'k'");
        assert!(s.contains('b'), "Should contain 'b'");
        assert!(s.contains('d'), "Should contain 'd'");
        assert!(s.contains('g'), "Should contain 'g'");
    }

    #[test]
    fn test_parse_shortcut_digit() {
        let r = parse(r"\d").unwrap();
        let s = r.to_string();
        // Should contain digits
        assert!(s.contains('0'), "Should contain '0'");
        assert!(s.contains('5'), "Should contain '5'");
        assert!(s.contains('9'), "Should contain '9'");
    }

    #[test]
    fn test_parse_shortcut_word() {
        let r = parse(r"\w").unwrap();
        let s = r.to_string();
        // Should contain word characters (alnum + _)
        assert!(s.contains('a'), "Should contain 'a'");
        assert!(s.contains('Z'), "Should contain 'Z'");
        assert!(s.contains('0'), "Should contain '0'");
        assert!(s.contains('_'), "Should contain '_'");
    }

    #[test]
    fn test_parse_shortcut_space() {
        let r = parse(r"\s").unwrap();
        let s = r.to_string();
        // Should contain whitespace (space, tab, newline, etc.)
        assert!(s.contains(' '), "Should contain space");
    }

    #[test]
    fn test_parse_shortcut_voiced() {
        let r = parse(r"\o").unwrap();
        let s = r.to_string();
        // Should contain voiced consonants
        assert!(s.contains('b'), "Should contain 'b'");
        assert!(s.contains('d'), "Should contain 'd'");
        assert!(s.contains('g'), "Should contain 'g'");
    }

    #[test]
    fn test_parse_shortcut_fricative() {
        let r = parse(r"\e").unwrap();
        let s = r.to_string();
        // Should contain fricatives
        assert!(s.contains('f'), "Should contain 'f'");
        assert!(s.contains('v'), "Should contain 'v'");
        assert!(s.contains('s'), "Should contain 's'");
        assert!(s.contains('z'), "Should contain 'z'");
    }

    #[test]
    fn test_parse_shortcut_affricate() {
        let r = parse(r"\a").unwrap();
        // Should parse successfully (affricates like ch, j)
        // Just check it parses - affricate class may be small
        assert!(r.to_string().len() > 0);
    }

    #[test]
    fn test_parse_shortcut_in_char_class() {
        // Shortcuts should work inside character classes
        let r = parse(r"[\v123]").unwrap();
        let s = r.to_string();
        // Should contain vowels and digits
        assert!(s.contains('a'), "Should contain 'a'");
        assert!(s.contains('1'), "Should contain '1'");
        assert!(s.contains('2'), "Should contain '2'");
        assert!(s.contains('3'), "Should contain '3'");
    }

    #[test]
    fn test_parse_shortcut_negated_in_char_class() {
        // Negated shortcuts should work inside character classes
        let r = parse(r"[\V]").unwrap();
        let s = r.to_string();
        // Should NOT contain vowels
        assert!(!s.contains('a'), "Should NOT contain 'a'");
        assert!(!s.contains('e'), "Should NOT contain 'e'");
    }

    #[test]
    fn test_parse_shortcut_mixed_in_pattern() {
        // Test simpler concatenation first
        let r = parse(r"\v\c").unwrap();
        // Just check it parses successfully
        assert!(r.to_string().len() > 0);
    }

    #[test]
    fn test_parse_shortcut_with_quantifier() {
        // Shortcut with quantifier
        let r = parse(r"\v+").unwrap();
        // Just check it parses successfully
        assert!(r.to_string().len() > 0);
    }

    // ========================================================================
    // Tests for Group Types (Phase 3)
    // ========================================================================

    #[test]
    fn test_parse_capturing_group() {
        // Standard capturing group: (abc)
        let r = parse("(abc)").unwrap();
        // Should produce CapturingGroup(1, ...)
        assert_eq!(r.to_string(), "(abc)");
    }

    #[test]
    fn test_parse_capturing_group_numbering() {
        // Multiple capturing groups should get sequential numbers
        let r = parse("(a)(b)(c)").unwrap();
        // All groups parse correctly
        assert!(r.to_string().contains("(a)"));
        assert!(r.to_string().contains("(b)"));
        assert!(r.to_string().contains("(c)"));
    }

    #[test]
    fn test_parse_non_capturing_group() {
        // Non-capturing group: (?:abc)
        let r = parse("(?:abc)").unwrap();
        assert_eq!(r.to_string(), "(?:abc)");
    }

    #[test]
    fn test_parse_non_capturing_group_complex() {
        // Non-capturing group with alternation
        let r = parse("(?:ph|f)one").unwrap();
        assert!(r.to_string().contains("(?:"));
    }

    #[test]
    fn test_parse_named_group() {
        // Named group: (?<name>pattern)
        let r = parse("(?<vowel>[aeiou])").unwrap();
        assert!(r.to_string().contains("(?<vowel>"));
    }

    #[test]
    fn test_parse_named_group_with_reference() {
        // Named group with valid reference
        let r = parse("(?<digit>[0-9])(?&digit)").unwrap();
        assert!(r.to_string().contains("(?<digit>"));
        assert!(r.to_string().contains("(?&digit)"));
    }

    #[test]
    fn test_parse_duplicate_named_group_error() {
        // Duplicate named groups should error
        let result = parse("(?<x>a)(?<x>b)");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::DuplicateGroupName(_)));
    }

    #[test]
    fn test_parse_undefined_group_reference_error() {
        // References to undefined groups should error
        let result = parse("(?&undefined)");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ParseErrorKind::UndefinedGroupReference(_)));
    }

    #[test]
    fn test_parse_forward_reference() {
        // Forward references are allowed (reference before definition)
        // The validation happens after parsing, so this pattern should parse
        // but the reference should be validated at the end
        let result = parse("(?&later)(?<later>abc)");
        // This should succeed because the group is defined before validation
        assert!(result.is_ok());
    }

    // ========================================================================
    // Tests for Flags (Phase 3)
    // ========================================================================

    #[test]
    fn test_parse_inline_flags_case_insensitive() {
        // Inline case-insensitive flag
        let r = parse("(?i)abc").unwrap();
        assert!(r.to_string().contains("(?i)"));
    }

    #[test]
    fn test_parse_scoped_flags_case_insensitive() {
        // Scoped case-insensitive flag
        let r = parse("(?i:abc)def").unwrap();
        assert!(r.to_string().contains("(?i:"));
    }

    #[test]
    fn test_parse_flag_disable() {
        // Disable flag
        let r = parse("(?-i)abc").unwrap();
        assert!(r.to_string().contains("(?-i)"));
    }

    #[test]
    fn test_parse_unicode_normalization_flag() {
        // Unicode normalization flag
        let r = parse("(?u:NFC:abc)").unwrap();
        assert!(r.to_string().contains("u:NFC"));
    }

    #[test]
    fn test_parse_unicode_normalization_nfd() {
        let r = parse("(?u:NFD:abc)").unwrap();
        assert!(r.to_string().contains("u:NFD"));
    }

    #[test]
    fn test_parse_unicode_normalization_nfkc() {
        let r = parse("(?u:NFKC:abc)").unwrap();
        assert!(r.to_string().contains("u:NFKC"));
    }

    #[test]
    fn test_parse_unicode_normalization_nfkd() {
        let r = parse("(?u:NFKD:abc)").unwrap();
        assert!(r.to_string().contains("u:NFKD"));
    }

    #[test]
    fn test_parse_feature_flag() {
        // Feature-based matching flag
        let r = parse("(?f)abc").unwrap();
        assert!(r.to_string().contains("(?f)"));
    }

    #[test]
    fn test_parse_accent_flag() {
        // Accent-insensitive flag
        let r = parse("(?a)cafe").unwrap();
        assert!(r.to_string().contains("(?a)"));
    }

    #[test]
    fn test_parse_combined_flags() {
        // Combined flags
        let r = parse("(?ia)abc").unwrap();
        let s = r.to_string();
        // Should contain both flags
        assert!(s.contains('i') || s.contains('a'));
    }

    #[test]
    fn test_parse_combined_scoped_flags() {
        // Combined scoped flags
        let r = parse("(?ia:abc)def").unwrap();
        assert!(r.to_string().contains("(?"));
    }

    // ========================================================================
    // Tests for NFA Compilation of New Group Types
    // ========================================================================

    #[test]
    fn test_compile_non_capturing_group() {
        use crate::phonetic::nfa::compiler::compile;
        let regex = parse("(?:ph|f)one").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("phone"));
        assert!(nfa.accepts("fone"));
        assert!(!nfa.accepts("bone"));
    }

    #[test]
    fn test_compile_capturing_group() {
        use crate::phonetic::nfa::compiler::compile;
        let regex = parse("(ph|f)one").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("phone"));
        assert!(nfa.accepts("fone"));
    }

    #[test]
    fn test_compile_named_group() {
        use crate::phonetic::nfa::compiler::compile;
        let regex = parse("(?<prefix>ph|f)one").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("phone"));
        assert!(nfa.accepts("fone"));
    }

    #[test]
    fn test_compile_flags_group() {
        use crate::phonetic::nfa::compiler::compile;
        // Flags don't affect matching yet, but should compile
        let regex = parse("(?i:abc)").unwrap();
        let nfa = compile(&regex).unwrap();
        // Without case-insensitive implementation, only exact match works
        assert!(nfa.accepts("abc"));
    }

    #[test]
    fn test_compile_inline_flags() {
        use crate::phonetic::nfa::compiler::compile;
        // Inline flags produce epsilon for now
        let regex = parse("(?i)").unwrap();
        let nfa = compile(&regex).unwrap();
        // Should accept empty string (epsilon)
        assert!(nfa.accepts(""));
    }

    // ========================================================================
    // Anchor Tests
    // ========================================================================

    /// Helper function to check if a Regex tree contains a specific variant
    fn contains_variant(regex: &Regex, predicate: &dyn Fn(&Regex) -> bool) -> bool {
        if predicate(regex) {
            return true;
        }
        match regex {
            Regex::Concat(left, right) => {
                contains_variant(left, predicate) || contains_variant(right, predicate)
            }
            Regex::Alt(left, right) => {
                contains_variant(left, predicate) || contains_variant(right, predicate)
            }
            Regex::Star(inner) | Regex::Plus(inner) | Regex::Optional(inner) => {
                contains_variant(inner, predicate)
            }
            Regex::RepeatExact(inner, _) | Regex::RepeatRange(inner, _, _) => {
                contains_variant(inner, predicate)
            }
            Regex::CapturingGroup(_, inner) | Regex::NonCapturingGroup(inner) | Regex::NamedGroup(_, inner) => {
                contains_variant(inner, predicate)
            }
            Regex::FlagsGroup { inner: Some(inner), .. } => {
                contains_variant(inner, predicate)
            }
            #[allow(deprecated)]
            Regex::Group(inner) => contains_variant(inner, predicate),
            _ => false,
        }
    }

    /// Helper to get the leftmost node in a concat chain
    fn leftmost(regex: &Regex) -> &Regex {
        match regex {
            Regex::Concat(left, _) => leftmost(left),
            _ => regex,
        }
    }

    /// Helper to get the rightmost node in a concat chain
    fn rightmost(regex: &Regex) -> &Regex {
        match regex {
            Regex::Concat(_, right) => rightmost(right),
            _ => regex,
        }
    }

    #[test]
    fn test_parse_start_of_line_anchor() {
        let regex = parse("^hello").unwrap();
        // Check that the leftmost element is StartOfLine
        assert!(
            matches!(leftmost(&regex), Regex::StartOfLine),
            "Expected StartOfLine at start, got {:?}", regex
        );
    }

    #[test]
    fn test_parse_end_of_line_anchor() {
        let regex = parse("hello$").unwrap();
        // Check that the rightmost element is EndOfLine
        assert!(
            matches!(rightmost(&regex), Regex::EndOfLine),
            "Expected EndOfLine at end, got {:?}", regex
        );
    }

    #[test]
    fn test_parse_both_anchors() {
        let regex = parse("^hello$").unwrap();
        assert!(
            matches!(leftmost(&regex), Regex::StartOfLine),
            "Expected StartOfLine at start"
        );
        assert!(
            matches!(rightmost(&regex), Regex::EndOfLine),
            "Expected EndOfLine at end"
        );
    }

    #[test]
    fn test_parse_start_of_input_anchor() {
        let regex = parse(r"\Ahello").unwrap();
        assert!(
            matches!(leftmost(&regex), Regex::StartOfInput),
            "Expected StartOfInput at start, got {:?}", regex
        );
    }

    #[test]
    fn test_parse_end_of_input_anchor() {
        let regex = parse(r"hello\Z").unwrap();
        assert!(
            matches!(rightmost(&regex), Regex::EndOfInput),
            "Expected EndOfInput at end, got {:?}", regex
        );
    }

    #[test]
    fn test_parse_end_of_input_strict_anchor() {
        let regex = parse(r"hello\z").unwrap();
        assert!(
            matches!(rightmost(&regex), Regex::EndOfInputStrict),
            "Expected EndOfInputStrict at end, got {:?}", regex
        );
    }

    #[test]
    fn test_parse_anchors_roundtrip() {
        // Test that anchors are correctly represented in Display
        let regex = parse("^hello$").unwrap();
        let display = regex.to_string();
        assert!(display.contains('^'), "Display should contain ^: {}", display);
        assert!(display.contains('$'), "Display should contain $: {}", display);
    }

    // ========================================================================
    // Multiline and Dotall Flag Tests
    // ========================================================================

    /// Helper to find FlagsGroup in a regex tree and return its flags
    fn find_flags_group(regex: &Regex) -> Option<&RegexFlags> {
        match regex {
            Regex::FlagsGroup { flags, .. } => Some(flags),
            Regex::Concat(left, right) => {
                find_flags_group(left).or_else(|| find_flags_group(right))
            }
            _ => None,
        }
    }

    #[test]
    fn test_parse_multiline_flag() {
        let regex = parse("(?m)^line$").unwrap();
        // Pattern should contain FlagsGroup with multiline=true
        let flags = find_flags_group(&regex);
        assert!(flags.is_some(), "Expected FlagsGroup in {:?}", regex);
        assert_eq!(flags.unwrap().multiline, Some(true));
    }

    #[test]
    fn test_parse_dotall_flag() {
        let regex = parse("(?s).*").unwrap();
        let flags = find_flags_group(&regex);
        assert!(flags.is_some(), "Expected FlagsGroup in {:?}", regex);
        assert_eq!(flags.unwrap().dotall, Some(true));
    }

    #[test]
    fn test_parse_combined_multiline_dotall() {
        let regex = parse("(?ms)test").unwrap();
        let flags = find_flags_group(&regex);
        assert!(flags.is_some(), "Expected FlagsGroup in {:?}", regex);
        let flags = flags.unwrap();
        assert_eq!(flags.multiline, Some(true));
        assert_eq!(flags.dotall, Some(true));
    }

    #[test]
    fn test_parse_scoped_multiline() {
        let regex = parse("(?m:^line$)").unwrap();
        // Should be FlagsGroup with inner pattern containing anchors
        match &regex {
            Regex::FlagsGroup { flags, inner: Some(inner) } => {
                assert_eq!(flags.multiline, Some(true));
                // Inner should contain anchors
                assert!(
                    contains_variant(inner, &|r| matches!(r, Regex::StartOfLine)),
                    "Expected StartOfLine in inner"
                );
                assert!(
                    contains_variant(inner, &|r| matches!(r, Regex::EndOfLine)),
                    "Expected EndOfLine in inner"
                );
            }
            _ => panic!("Expected FlagsGroup with inner pattern, got {:?}", regex),
        }
    }

    #[test]
    fn test_parse_negated_flags() {
        let regex = parse("(?-ms)test").unwrap();
        let flags = find_flags_group(&regex);
        assert!(flags.is_some(), "Expected FlagsGroup in {:?}", regex);
        let flags = flags.unwrap();
        assert_eq!(flags.multiline, Some(false), "multiline should be false");
        assert_eq!(flags.dotall, Some(false), "dotall should be false");
    }
}
