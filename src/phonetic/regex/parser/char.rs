//! Char-level (Unicode) recursive descent parser for phonetic regular expressions.
//!
//! This parser supports the full feature set including capturing groups, named
//! groups, group references, scoped flags, symbol references, and named classes.

use std::collections::HashMap;

use super::common::{resolve_feature_bundle_chars, SymbolTable, MAX_PATTERN_SIZE};
use crate::phonetic::common::traits::SyllableParser;
use crate::phonetic::common::utils::negate_char_class;
use crate::phonetic::nfa::types::CharClassChar;
use crate::phonetic::regex::ast::{
    ContextExpr, ContextPredicate, Regex, RegexFlags, SyllableCondition, SyllableExpr,
    UnicodeNormalization,
};
use crate::phonetic::regex::error::{ParseError, ParseErrorKind, ParseResult, Position};
use crate::phonetic::regex::lexer::{Lexer, ParsedFlags, Token};

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
    /// Whether undefined group references should be rejected after parsing.
    validate_group_refs: bool,
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
            validate_group_refs: true,
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
    /// let regex = parser.parse().expect("doc: parser.parse must succeed");
    /// ```
    pub fn new_with_symbols(input: &'a str, symbols: &'a SymbolTable) -> Self {
        Self {
            lexer: Lexer::new(input),
            symbols: Some(symbols),
            next_group_number: 1,
            named_groups: HashMap::new(),
            group_refs_to_validate: Vec::new(),
            validate_group_refs: true,
        }
    }

    /// Allow group references that are not defined as local named groups.
    ///
    /// This is used by higher-level formats such as LLRE, where `(?&NAME)` can
    /// refer to an imported pattern symbol that is resolved after the file-level
    /// parser has loaded imports. Standalone regex parsing remains strict by
    /// default.
    pub fn allow_external_group_refs(mut self) -> Self {
        self.validate_group_refs = false;
        self
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
        if self.validate_group_refs {
            self.validate_group_references()?;
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
        // We distinguish from char class [abc] by checking if '[' is followed by a digit
        let weight = if self.lexer.is_at_weight_start() {
            self.parse_weight()?
        } else {
            0.0
        };

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
                self.named_groups
                    .insert(name.clone(), (group_num, inner.clone()));

                Ok(Regex::named_group(name, inner))
            }

            // Group reference (subroutine call): (?&name)
            Token::GroupReference(name) => {
                // Record for deferred validation
                self.group_refs_to_validate
                    .push((name.clone(), self.lexer.position()));
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
            Token::PhoneticShortcut {
                class_name,
                negated,
            } => self.expand_phonetic_shortcut(&class_name, negated),

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
                        ParseErrorKind::UndefinedSymbol {
                            name: name.to_string(),
                            available,
                        },
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
                Token::PhoneticShortcut {
                    class_name,
                    negated,
                } => {
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
                    _ => unreachable!(
                        "outer match peeked Token::Char|Token::Dot; next_token cannot return a different variant"
                    ),
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
        if matches!(peek, Token::Slash | Token::Eof) {
            return Ok(true);
        }
        // Also check for weight start: '[' followed by digit
        Ok(self.lexer.is_at_weight_start())
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
            Token::StartOfInput => '\\',     // \A
            Token::EndOfInput => '\\',       // \Z
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
            ParseErrorKind::InvalidContext(format!("expected {}, got {:?}", expected, found)),
            position,
        )
    }

    fn from_lexer_error(&self, err: ParseError) -> Self::Error {
        err
    }
}
