//! Token-aware fuzzy grep with per-token Levenshtein distance.
//!
//! This module implements a principled token-aware matching system that applies
//! Levenshtein distance per-token rather than per-character/substring. Key features:
//!
//! - **Per-token Levenshtein**: Each token has independent distance budget
//! - **Wildcard support**: `.*` matches arbitrary content between tokens (zero cost)
//! - **Hybrid query syntax**: `hello world` or `error:0 .* failed:1`
//! - **Phonetic integration**: Normalize tokens before fuzzy matching
//!
//! # Query Syntax
//!
//! ```text
//! hello world           # Two tokens, default distance each
//! hello:1 world:0       # Explicit distance per token
//! error .* failed       # Wildcard matches anything between tokens
//! (phone|fone):1 call   # Alternation with distance budget
//! "stack overflow"      # Quoted phrase as single token
//! error\:404:1          # Escaped colon: literal "error:404" with distance 1
//! ```
//!
//! ## Escape Sequences
//!
//! | Sequence | Meaning |
//! |----------|---------|
//! | `\:` | Literal colon (not distance specifier) |
//! | `\\` | Literal backslash |
//! | `\.` | Literal dot (prevents `.*` interpretation) |
//! | `\(`, `\)` | Literal parentheses |
//! | `\|` | Literal pipe |
//! | `\"` | Literal quote |
//!
//! # Examples
//!
//! ## Basic Multi-Token
//!
//! ```ignore
//! let grep = TokenGrep::new("hello world", 1)?;
//! grep.scan("hello world")   // distance 0+0=0 ✓
//! grep.scan("helo wrld")     // distance 1+1=2 ✓
//! grep.scan("helloworld")    // no separator ✗
//! ```
//!
//! ## Explicit Distance
//!
//! ```ignore
//! let grep = TokenGrep::new("error:0 .* failed:1", 0)?;
//! grep.scan("error: test failed")   // exact+wildcard+d1 ✓
//! grep.scan("erro: test failed")    // d1+wildcard+d0 ✗ (error:0 violated)
//! ```
//!
//! ## Phonetic + Fuzzy
//!
//! ```ignore
//! let rules = vec![make_rule("ph", "f", Anywhere)];
//! let grep = TokenGrep::with_rules("phone call", rules, 1)?;
//! grep.scan("fone call")     // d0+d0=0 (phonetic equiv) ✓
//! grep.scan("fon call")      // d1+d0=1 ✓
//! ```

use crate::phonetic::grep::{GrepError, WordBoundaryIterator};
use crate::phonetic::nfa::{compile, NFAChar, ProductAutomatonChar, ThompsonBuilderChar};
use crate::phonetic::types::RewriteRuleChar;
use crate::phonetic::{apply_rules_seq_char, PhoneChar};

/// Helper to create a parse error from a message string.
fn parse_error(msg: &str) -> GrepError {
    GrepError::Compile(format!("parse error: {}", msg))
}

// ============================================================================
// Data Structures
// ============================================================================

/// A parsed token query.
#[derive(Debug, Clone)]
pub struct TokenQuery {
    /// The tokens to match in sequence
    pub tokens: Vec<TokenSpec>,
    /// Separators between tokens (one less than tokens.len())
    pub separators: Vec<Separator>,
    /// Default distance for tokens without explicit `:N` suffix
    pub default_distance: u8,
}

/// Specification for a single token in the query.
#[derive(Debug, Clone)]
pub struct TokenSpec {
    /// The pattern to match
    pub pattern: TokenPattern,
    /// Maximum edit distance for this token
    pub max_distance: u8,
}

/// Pattern types for tokens.
#[derive(Debug, Clone)]
pub enum TokenPattern {
    /// A literal string (after escape processing)
    Literal(String),
    /// Alternation: (A|B|C)
    Alternation(Vec<TokenPattern>),
}

/// Separator types between tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Separator {
    /// One or more whitespace characters
    Whitespace,
    /// Wildcard: .* matches any sequence of characters (zero cost)
    Wildcard,
}

/// A compiled token query ready for matching.
pub struct CompiledTokenQuery {
    /// Product automata for each token
    token_automata: Vec<ProductAutomatonChar>,
    /// Separators between tokens
    separators: Vec<Separator>,
    /// Optional phonetic rules for normalization
    rules: Option<Vec<RewriteRuleChar>>,
}

/// Result of a token match.
#[derive(Debug, Clone)]
pub struct TokenMatch {
    /// Byte range in the original document
    pub byte_range: (usize, usize),
    /// Details for each token match
    pub token_matches: Vec<TokenMatchDetail>,
    /// Sum of per-token distances
    pub total_distance: u8,
    /// The matched text from the document
    pub matched_text: String,
}

/// Details about a single token match within a TokenMatch.
#[derive(Debug, Clone)]
pub struct TokenMatchDetail {
    /// Index of the token in the query
    pub token_index: usize,
    /// Byte range in the original document
    pub byte_range: (usize, usize),
    /// Original text from the document
    pub original_text: String,
    /// Text after phonetic normalization
    pub normalized_text: String,
    /// Edit distance for this token
    pub distance: u8,
}

// ============================================================================
// Parser
// ============================================================================

/// Parse a token query string into a TokenQuery structure.
///
/// # Grammar
///
/// ```text
/// query   ::= token (sep token)*
/// sep     ::= WHITESPACE | '.*'
/// token   ::= pattern (':' distance)?
/// pattern ::= WORD | '(' pattern ('|' pattern)* ')' | '"' PHRASE '"'
/// WORD    ::= (CHAR | ESCAPE)+
/// ESCAPE  ::= '\' ( ':' | '\' | '.' | '(' | ')' | '|' | '"' )
/// ```
pub fn parse_query(input: &str, default_distance: u8) -> Result<TokenQuery, GrepError> {
    let mut parser = QueryParser::new(input, default_distance);
    parser.parse()
}

struct QueryParser<'a> {
    input: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    default_distance: u8,
}

impl<'a> QueryParser<'a> {
    fn new(input: &'a str, default_distance: u8) -> Self {
        Self {
            input,
            chars: input.char_indices().peekable(),
            default_distance,
        }
    }

    fn parse(&mut self) -> Result<TokenQuery, GrepError> {
        let mut tokens = Vec::new();
        let mut separators = Vec::new();

        self.skip_whitespace();

        // Parse first token
        if self.peek().is_some() {
            tokens.push(self.parse_token_spec()?);
        }

        // Parse remaining tokens with separators
        while self.peek().is_some() {
            let sep = self.parse_separator()?;

            // Check if there's another token
            if self.peek().is_none() {
                break;
            }

            separators.push(sep);
            tokens.push(self.parse_token_spec()?);
        }

        if tokens.is_empty() {
            return Err(parse_error("empty query"));
        }

        Ok(TokenQuery {
            tokens,
            separators,
            default_distance: self.default_distance,
        })
    }

    fn parse_token_spec(&mut self) -> Result<TokenSpec, GrepError> {
        let pattern = self.parse_pattern()?;

        // Check for :N suffix
        let max_distance = if self.peek_char() == Some(':') {
            self.advance(); // consume ':'
            self.parse_distance()?
        } else {
            self.default_distance
        };

        Ok(TokenSpec {
            pattern,
            max_distance,
        })
    }

    fn parse_pattern(&mut self) -> Result<TokenPattern, GrepError> {
        match self.peek_char() {
            Some('(') => self.parse_alternation(),
            Some('"') => self.parse_quoted(),
            _ => self.parse_literal(),
        }
    }

    fn parse_alternation(&mut self) -> Result<TokenPattern, GrepError> {
        self.expect('(')?;
        let mut alternatives = Vec::new();

        loop {
            // Parse alternative pattern (can be literal or nested)
            let alt = self.parse_alternative_pattern()?;
            alternatives.push(alt);

            match self.peek_char() {
                Some('|') => {
                    self.advance(); // consume '|'
                }
                Some(')') => {
                    self.advance(); // consume ')'
                    break;
                }
                Some(c) => {
                    return Err(parse_error(&format!(
                        "expected '|' or ')' in alternation, found '{}'",
                        c
                    )));
                }
                None => {
                    return Err(parse_error("unclosed alternation"));
                }
            }
        }

        if alternatives.is_empty() {
            return Err(parse_error("empty alternation"));
        }

        Ok(TokenPattern::Alternation(alternatives))
    }

    fn parse_alternative_pattern(&mut self) -> Result<TokenPattern, GrepError> {
        // Within alternation, parse until we hit | or )
        let mut chars = String::new();

        while let Some(c) = self.peek_char() {
            match c {
                '|' | ')' => break,
                '\\' => {
                    self.advance(); // consume '\'
                    if let Some(escaped) = self.peek_char() {
                        chars.push(self.unescape(escaped)?);
                        self.advance();
                    } else {
                        return Err(parse_error("trailing backslash"));
                    }
                }
                _ => {
                    chars.push(c);
                    self.advance();
                }
            }
        }

        if chars.is_empty() {
            return Err(parse_error("empty pattern in alternation"));
        }

        Ok(TokenPattern::Literal(chars))
    }

    fn parse_quoted(&mut self) -> Result<TokenPattern, GrepError> {
        self.expect('"')?;
        let mut chars = String::new();

        loop {
            match self.peek_char() {
                Some('"') => {
                    self.advance(); // consume closing '"'
                    break;
                }
                Some('\\') => {
                    self.advance(); // consume '\'
                    if let Some(escaped) = self.peek_char() {
                        chars.push(self.unescape(escaped)?);
                        self.advance();
                    } else {
                        return Err(parse_error("trailing backslash in quoted string"));
                    }
                }
                Some(c) => {
                    chars.push(c);
                    self.advance();
                }
                None => {
                    return Err(parse_error("unclosed quoted string"));
                }
            }
        }

        Ok(TokenPattern::Literal(chars))
    }

    fn parse_literal(&mut self) -> Result<TokenPattern, GrepError> {
        let mut chars = String::new();

        while let Some(c) = self.peek_char() {
            match c {
                // Stop at separator-like characters
                ' ' | '\t' | '\n' | '\r' => break,
                // Stop at special characters that start other constructs
                ':' => {
                    // A raw colon starts the :N distance suffix. Use `\:` for a
                    // literal colon inside a token.
                    break;
                }
                '(' | ')' | '|' | '"' => break,
                // Check for wildcard .*
                '.' => {
                    // Look ahead to see if this is .*
                    if self.is_wildcard_ahead() {
                        break;
                    }
                    // Otherwise, treat . as literal (unusual but valid)
                    chars.push(c);
                    self.advance();
                }
                // Handle escapes
                '\\' => {
                    self.advance(); // consume '\'
                    if let Some(escaped) = self.peek_char() {
                        chars.push(self.unescape(escaped)?);
                        self.advance();
                    } else {
                        return Err(parse_error("trailing backslash"));
                    }
                }
                // Regular character
                _ => {
                    chars.push(c);
                    self.advance();
                }
            }
        }

        if chars.is_empty() {
            return Err(parse_error("empty token pattern"));
        }

        Ok(TokenPattern::Literal(chars))
    }

    fn parse_separator(&mut self) -> Result<Separator, GrepError> {
        let start_whitespace = self.skip_whitespace();

        // Check for .* wildcard
        if self.peek_char() == Some('.') && self.is_wildcard_ahead() {
            self.advance(); // consume '.'
            self.advance(); // consume '*'
            self.skip_whitespace();
            return Ok(Separator::Wildcard);
        }

        // If we consumed whitespace, that's our separator
        if start_whitespace {
            return Ok(Separator::Whitespace);
        }

        // No separator found - this might be an error or end of input
        Err(parse_error("expected separator between tokens"))
    }

    fn parse_distance(&mut self) -> Result<u8, GrepError> {
        let mut digits = String::new();

        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                digits.push(c);
                self.advance();
            } else {
                break;
            }
        }

        if digits.is_empty() {
            return Err(parse_error("expected distance after ':'"));
        }

        digits
            .parse::<u8>()
            .map_err(|_| parse_error(&format!("invalid distance: {} (must be 0-255)", digits)))
    }

    fn unescape(&self, c: char) -> Result<char, GrepError> {
        match c {
            ':' | '\\' | '.' | '(' | ')' | '|' | '"' => Ok(c),
            _ => Err(parse_error(&format!("invalid escape sequence: \\{}", c))),
        }
    }

    fn is_wildcard_ahead(&mut self) -> bool {
        // Check if current position is at '.' followed by '*'
        let chars_copy: Vec<_> = self.input[self.current_pos()..].chars().take(2).collect();
        chars_copy.len() >= 2 && chars_copy[0] == '.' && chars_copy[1] == '*'
    }

    fn current_pos(&mut self) -> usize {
        self.chars
            .peek()
            .map(|(pos, _)| *pos)
            .unwrap_or(self.input.len())
    }

    fn peek(&mut self) -> Option<(usize, char)> {
        self.chars.peek().copied()
    }

    fn peek_char(&mut self) -> Option<char> {
        self.chars.peek().map(|(_, c)| *c)
    }

    fn advance(&mut self) -> Option<(usize, char)> {
        self.chars.next()
    }

    fn expect(&mut self, expected: char) -> Result<(), GrepError> {
        match self.advance() {
            Some((_, c)) if c == expected => Ok(()),
            Some((_, c)) => Err(parse_error(&format!(
                "expected '{}', found '{}'",
                expected, c
            ))),
            None => Err(parse_error(&format!(
                "expected '{}', found end of input",
                expected
            ))),
        }
    }

    fn skip_whitespace(&mut self) -> bool {
        let mut skipped = false;
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.advance();
                skipped = true;
            } else {
                break;
            }
        }
        skipped
    }
}

// ============================================================================
// Compiler
// ============================================================================

impl CompiledTokenQuery {
    /// Compile a parsed TokenQuery into automata.
    pub fn compile(query: &TokenQuery) -> Result<Self, GrepError> {
        Self::compile_with_rules(query, None)
    }

    /// Compile with phonetic rules.
    pub fn compile_with_rules(
        query: &TokenQuery,
        rules: Option<Vec<RewriteRuleChar>>,
    ) -> Result<Self, GrepError> {
        let mut token_automata = Vec::with_capacity(query.tokens.len());

        for token in &query.tokens {
            let nfa = Self::pattern_to_nfa(&token.pattern)?;
            let product = ProductAutomatonChar::new(nfa, token.max_distance);
            token_automata.push(product);
        }

        Ok(Self {
            token_automata,
            separators: query.separators.clone(),
            rules,
        })
    }

    /// Convert a TokenPattern to an NFA.
    fn pattern_to_nfa(pattern: &TokenPattern) -> Result<NFAChar, GrepError> {
        match pattern {
            TokenPattern::Literal(s) => {
                // Parse as regex (escaping special chars)
                let escaped = Self::escape_for_regex(s);
                let regex = crate::phonetic::regex::parse(&escaped)?;
                compile(&regex).map_err(|e| GrepError::Compile(e.to_string()))
            }
            TokenPattern::Alternation(alts) => {
                if alts.is_empty() {
                    return Err(GrepError::Compile("empty alternation".to_string()));
                }

                let builder = ThompsonBuilderChar::new();
                let mut nfas = Vec::new();

                for alt in alts {
                    nfas.push(Self::pattern_to_nfa(alt)?);
                }

                // Build alternation NFA
                let mut result = nfas.remove(0);
                for nfa in nfas {
                    result = builder.alternation(result, nfa);
                }

                Ok(result)
            }
        }
    }

    /// Escape special regex characters in a literal.
    fn escape_for_regex(s: &str) -> String {
        let mut result = String::with_capacity(s.len() * 2);
        for c in s.chars() {
            match c {
                '.' | '*' | '+' | '?' | '[' | ']' | '(' | ')' | '{' | '}' | '|' | '^' | '$'
                | '\\' => {
                    result.push('\\');
                    result.push(c);
                }
                _ => result.push(c),
            }
        }
        result
    }
}

// ============================================================================
// Main API: TokenGrep
// ============================================================================

/// Token-aware grep with per-token Levenshtein distance.
///
/// Each token in the query has its own maximum edit distance, allowing
/// precise control over fuzzy matching granularity.
pub struct TokenGrep {
    compiled: CompiledTokenQuery,
}

impl TokenGrep {
    /// Create a new TokenGrep from a query string.
    ///
    /// # Arguments
    ///
    /// * `query` - The query string (see module docs for syntax)
    /// * `default_distance` - Default max distance for tokens without `:N` suffix
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let grep = TokenGrep::new("hello world", 1)?;
    /// let grep = TokenGrep::new("error:0 .* failed:1", 0)?;
    /// ```
    pub fn new(query: &str, default_distance: u8) -> Result<Self, GrepError> {
        let parsed = parse_query(query, default_distance)?;
        let compiled = CompiledTokenQuery::compile(&parsed)?;
        Ok(Self { compiled })
    }

    /// Create a TokenGrep with phonetic rules.
    ///
    /// Tokens are normalized using the phonetic rules before matching,
    /// allowing phonetically equivalent strings to match with distance 0.
    pub fn with_rules(
        query: &str,
        rules: Vec<RewriteRuleChar>,
        default_distance: u8,
    ) -> Result<Self, GrepError> {
        let parsed = parse_query(query, default_distance)?;
        let compiled = CompiledTokenQuery::compile_with_rules(&parsed, Some(rules))?;
        Ok(Self { compiled })
    }

    /// Scan a document for matches.
    ///
    /// Returns all non-overlapping matches found in the document.
    pub fn scan(&self, document: &str) -> Vec<TokenMatch> {
        let mut matches = Vec::new();

        // Get all word positions in the document
        let words: Vec<(usize, &str, usize)> = WordBoundaryIterator::new(document).collect();

        if words.is_empty() || self.compiled.token_automata.is_empty() {
            return matches;
        }

        // Try starting from each word position
        for start_idx in 0..words.len() {
            if let Some(m) = self.try_match_from(&words, start_idx, document) {
                // Check for overlap with previous match
                let overlaps = matches.last().map_or(false, |last: &TokenMatch| {
                    m.byte_range.0 < last.byte_range.1
                });

                if !overlaps {
                    matches.push(m);
                }
            }
        }

        matches
    }

    /// Try to match the token sequence starting from a word index.
    fn try_match_from(
        &self,
        words: &[(usize, &str, usize)],
        start_idx: usize,
        document: &str,
    ) -> Option<TokenMatch> {
        let mut token_matches = Vec::new();
        let mut word_idx = start_idx;
        let mut total_distance: u8 = 0;

        for (token_idx, product) in self.compiled.token_automata.iter().enumerate() {
            // Check if we have more words
            if word_idx >= words.len() {
                return None;
            }

            // Match separator if not the first token
            if token_idx > 0 {
                let sep = &self.compiled.separators[token_idx - 1];

                match sep {
                    Separator::Whitespace => {
                        // Whitespace separator: next word must be adjacent (just whitespace between)
                        // The word iterator already handles word boundaries
                        // Just move to next word
                    }
                    Separator::Wildcard => {
                        // Wildcard: try all remaining words to find a match
                        let mut found = false;
                        for try_idx in word_idx..words.len() {
                            if let Some(detail) =
                                self.try_match_token(product, words, try_idx, token_idx)
                            {
                                word_idx = try_idx;
                                found = true;
                                total_distance = total_distance.saturating_add(detail.distance);
                                token_matches.push(detail);
                                break;
                            }
                        }
                        if !found {
                            return None;
                        }
                        word_idx += 1;
                        continue; // Skip the normal token matching below
                    }
                }
            }

            // Try to match current token at current word
            if let Some(detail) = self.try_match_token(product, words, word_idx, token_idx) {
                total_distance = total_distance.saturating_add(detail.distance);
                token_matches.push(detail);
                word_idx += 1;
            } else {
                return None;
            }
        }

        if token_matches.is_empty() {
            return None;
        }

        let first = &token_matches[0];
        let last = &token_matches[token_matches.len() - 1];
        let byte_range = (first.byte_range.0, last.byte_range.1);
        let matched_text = document[byte_range.0..byte_range.1].to_string();

        Some(TokenMatch {
            byte_range,
            token_matches,
            total_distance,
            matched_text,
        })
    }

    /// Try to match a single token at a word position.
    fn try_match_token(
        &self,
        product: &ProductAutomatonChar,
        words: &[(usize, &str, usize)],
        word_idx: usize,
        token_idx: usize,
    ) -> Option<TokenMatchDetail> {
        let (start, word, end) = words[word_idx];
        let normalized = self.normalize(word);

        product
            .min_distance(&normalized)
            .map(|distance| TokenMatchDetail {
                token_index: token_idx,
                byte_range: (start, end),
                original_text: word.to_string(),
                normalized_text: normalized,
                distance,
            })
    }

    /// Normalize text using phonetic rules if available.
    fn normalize(&self, text: &str) -> String {
        match &self.compiled.rules {
            Some(rules) if !rules.is_empty() => {
                let vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'];
                let input_phones: Vec<PhoneChar> = text
                    .chars()
                    .map(|c| {
                        if vowels.contains(&c) {
                            PhoneChar::Vowel(c)
                        } else {
                            PhoneChar::Consonant(c)
                        }
                    })
                    .collect();

                match apply_rules_seq_char(rules, &input_phones, 100) {
                    Some(phones) => {
                        let mut s = String::new();
                        for p in phones.iter() {
                            match p {
                                PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => s.push(*c),
                                PhoneChar::Digraph(c1, c2) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                }
                                PhoneChar::Trigraph(c1, c2, c3) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                }
                                PhoneChar::Tetragraph(c1, c2, c3, c4) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                }
                                PhoneChar::Pentagraph(c1, c2, c3, c4, c5) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                    s.push(*c5);
                                }
                                PhoneChar::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                    s.push(*c5);
                                    s.push(*c6);
                                }
                                PhoneChar::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                    s.push(*c5);
                                    s.push(*c6);
                                    s.push(*c7);
                                }
                                PhoneChar::Sequence(chars) => {
                                    for c in chars {
                                        s.push(*c);
                                    }
                                }
                                PhoneChar::Silent => {}
                            }
                        }
                        s
                    }
                    None => text.to_string(),
                }
            }
            _ => text.to_string(),
        }
    }
}

// ============================================================================
// Parallel Scanning (Rayon)
// ============================================================================

/// Result of scanning a document with an ID.
#[derive(Debug, Clone)]
pub struct DocumentMatch<I> {
    /// Document identifier
    pub doc_id: I,
    /// Matches found in this document
    pub matches: Vec<TokenMatch>,
}

#[cfg(feature = "parallel-grep")]
impl TokenGrep {
    /// Scan multiple documents in parallel using Rayon.
    ///
    /// Each document is processed independently on a separate thread.
    /// Results are returned in arbitrary order.
    ///
    /// # Arguments
    ///
    /// * `documents` - Iterator of (id, document) pairs
    ///
    /// # Returns
    ///
    /// Vector of document matches (only documents with matches are included).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let grep = TokenGrep::new("error .* failed", 0)?;
    /// let docs = vec![
    ///     (1, "error in module failed"),
    ///     (2, "success complete"),
    ///     (3, "error test failed"),
    /// ];
    /// let results = grep.scan_documents_parallel(docs);
    /// // results contains DocumentMatch for docs 1 and 3
    /// ```
    pub fn scan_documents_parallel<I, S>(
        &self,
        documents: impl IntoIterator<Item = (I, S)>,
    ) -> Vec<DocumentMatch<I>>
    where
        I: Send + Clone,
        S: AsRef<str> + Send,
    {
        use rayon::prelude::*;

        let docs: Vec<_> = documents.into_iter().collect();

        docs.into_par_iter()
            .filter_map(|(id, doc)| {
                let matches = self.scan(doc.as_ref());
                if matches.is_empty() {
                    None
                } else {
                    Some(DocumentMatch {
                        doc_id: id,
                        matches,
                    })
                }
            })
            .collect()
    }

    /// Filter documents that contain at least one match.
    ///
    /// More efficient than `scan_documents_parallel` when you only need
    /// to know which documents match, not the match details.
    pub fn filter_documents_parallel<I, S>(
        &self,
        documents: impl IntoIterator<Item = (I, S)>,
    ) -> Vec<I>
    where
        I: Send + Clone,
        S: AsRef<str> + Send,
    {
        use rayon::prelude::*;

        let docs: Vec<_> = documents.into_iter().collect();

        docs.into_par_iter()
            .filter_map(|(id, doc)| {
                let matches = self.scan(doc.as_ref());
                if matches.is_empty() {
                    None
                } else {
                    Some(id)
                }
            })
            .collect()
    }

    /// Count total matches across all documents in parallel.
    pub fn count_documents_parallel<I, S>(
        &self,
        documents: impl IntoIterator<Item = (I, S)>,
    ) -> usize
    where
        I: Send,
        S: AsRef<str> + Send,
    {
        use rayon::prelude::*;

        let docs: Vec<_> = documents.into_iter().collect();

        docs.into_par_iter()
            .map(|(_, doc)| self.scan(doc.as_ref()).len())
            .sum()
    }
}

// ============================================================================
// Streaming Token Matcher
// ============================================================================

use std::collections::VecDeque;

/// Maximum number of words to buffer in streaming mode.
const MAX_BUFFERED_WORDS: usize = 1000;

/// Maximum number of concurrent match attempts.
const MAX_ACTIVE_ATTEMPTS: usize = 100;

/// A buffered word with position information.
#[derive(Debug, Clone)]
struct BufferedWord {
    /// Byte offset where the word starts in the document
    byte_start: usize,
    /// Byte offset where the word ends in the document
    byte_end: usize,
    /// Original word text
    original: String,
    /// Normalized word text (after phonetic rules)
    normalized: String,
}

/// State of a match attempt after processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptState {
    /// Match is still in progress
    InProgress,
    /// All tokens matched successfully
    Complete,
    /// This attempt failed (no valid match)
    Failed,
    /// No match for current word, but attempt may continue (wildcard)
    NoMatch,
}

/// An active attempt to match the token sequence.
#[derive(Debug, Clone)]
struct MatchAttempt {
    /// Index of first word in this attempt (relative to current buffer)
    start_word_idx: usize,
    /// Current token index being matched (0-based)
    current_token_idx: usize,
    /// Last word index that matched a token
    last_matched_word_idx: usize,
    /// Accumulated token match details
    token_details: Vec<TokenMatchDetail>,
    /// Total distance accumulated so far
    total_distance: u8,
}

/// Streaming token matcher for incremental document processing.
///
/// This struct allows processing large documents word-by-word without
/// loading the entire document into memory. It maintains internal state
/// to track partial matches across multiple `feed_word` calls.
///
/// # Example
///
/// ```ignore
/// let grep = TokenGrep::new("hello world", 1)?;
/// let mut stream = grep.streaming();
///
/// // Feed words one at a time
/// let m1 = stream.feed_word("hello", 0, 5);
/// assert!(m1.is_empty());  // Incomplete match
///
/// let m2 = stream.feed_word("world", 6, 11);
/// assert_eq!(m2.len(), 1);  // Complete match!
///
/// // Finalize and get any remaining matches
/// let final_matches = stream.finish();
/// ```
pub struct StreamingTokenMatcher {
    /// Compiled query with token automata
    compiled: CompiledTokenQuery,

    /// Buffered words awaiting matching
    word_buffer: VecDeque<BufferedWord>,

    /// Active match attempts starting from different positions
    match_attempts: Vec<MatchAttempt>,

    /// Completed matches ready to return
    completed_matches: Vec<TokenMatch>,

    /// Global word index counter (for tracking across buffer pruning)
    global_word_idx: usize,

    /// Offset applied to word indices after pruning
    word_idx_offset: usize,
}

impl StreamingTokenMatcher {
    /// Create a new streaming matcher from a compiled query.
    pub fn new(compiled: CompiledTokenQuery) -> Self {
        Self {
            compiled,
            word_buffer: VecDeque::with_capacity(64),
            match_attempts: Vec::with_capacity(16),
            completed_matches: Vec::new(),
            global_word_idx: 0,
            word_idx_offset: 0,
        }
    }

    /// Feed a word to the matcher.
    ///
    /// Returns any matches that have been completed by this word.
    /// The word should be pre-extracted using word boundary detection.
    ///
    /// # Arguments
    ///
    /// * `word` - The word text
    /// * `byte_start` - Byte offset where the word starts in the document
    /// * `byte_end` - Byte offset where the word ends in the document
    pub fn feed_word(&mut self, word: &str, byte_start: usize, byte_end: usize) -> Vec<TokenMatch> {
        // Normalize the word
        let normalized = self.normalize(word);

        // Add to buffer
        let buffered = BufferedWord {
            byte_start,
            byte_end,
            original: word.to_string(),
            normalized,
        };
        self.word_buffer.push_back(buffered);

        let current_word_idx = self.global_word_idx;
        self.global_word_idx += 1;

        // Start a new match attempt at this word position
        if !self.compiled.token_automata.is_empty() {
            self.match_attempts.push(MatchAttempt {
                start_word_idx: current_word_idx,
                current_token_idx: 0,
                last_matched_word_idx: current_word_idx.wrapping_sub(1), // Will be set on first match
                token_details: Vec::new(),
                total_distance: 0,
            });
        }

        // Process all active attempts with the new word
        self.process_attempts(current_word_idx);

        // Prune old words and attempts
        self.prune_if_needed();

        // Return completed matches
        std::mem::take(&mut self.completed_matches)
    }

    /// Signal end of document and return remaining matches.
    ///
    /// This should be called after all words have been fed to finalize
    /// any pending matches.
    pub fn finish(&mut self) -> Vec<TokenMatch> {
        // Any remaining attempts that haven't completed are abandoned
        self.match_attempts.clear();
        self.word_buffer.clear();

        std::mem::take(&mut self.completed_matches)
    }

    /// Reset state for reuse with a new document.
    pub fn reset(&mut self) {
        self.word_buffer.clear();
        self.match_attempts.clear();
        self.completed_matches.clear();
        self.global_word_idx = 0;
        self.word_idx_offset = 0;
    }

    /// Process all active match attempts with the current word.
    fn process_attempts(&mut self, current_word_idx: usize) {
        let mut i = 0;
        while i < self.match_attempts.len() {
            let state = self.try_advance_attempt(i, current_word_idx);

            match state {
                AttemptState::Complete => {
                    // Extract and finalize the match
                    let attempt = self.match_attempts.remove(i);
                    if let Some(token_match) = self.finalize_attempt(&attempt) {
                        self.completed_matches.push(token_match);
                    }
                    // Don't increment i since we removed an element
                }
                AttemptState::Failed => {
                    self.match_attempts.remove(i);
                    // Don't increment i
                }
                AttemptState::InProgress | AttemptState::NoMatch => {
                    i += 1;
                }
            }
        }
    }

    /// Try to advance a match attempt with the current word.
    fn try_advance_attempt(&mut self, attempt_idx: usize, current_word_idx: usize) -> AttemptState {
        let attempt = &self.match_attempts[attempt_idx];
        let token_idx = attempt.current_token_idx;

        // Check if we've matched all tokens
        if token_idx >= self.compiled.token_automata.len() {
            return AttemptState::Complete;
        }

        // Get the buffer index for the current word
        let buffer_idx = current_word_idx - self.word_idx_offset;
        if buffer_idx >= self.word_buffer.len() {
            return AttemptState::InProgress; // Word not in buffer yet
        }

        // Check separator constraints
        if token_idx > 0 {
            let sep_idx = token_idx - 1;
            let sep = &self.compiled.separators[sep_idx];

            match sep {
                Separator::Whitespace => {
                    // For whitespace separator, words must be consecutive
                    // (i.e., current word immediately follows the last matched word)
                    let last_matched = self.match_attempts[attempt_idx].last_matched_word_idx;
                    if current_word_idx != last_matched + 1 {
                        // Not consecutive - this attempt cannot use this word
                        // If we're past the expected position, fail the attempt
                        if current_word_idx > last_matched + 1 {
                            return AttemptState::Failed;
                        }
                        return AttemptState::NoMatch;
                    }
                }
                Separator::Wildcard => {
                    // Wildcard allows skipping words - no constraint
                }
            }
        } else {
            // First token - only process on the starting word
            let start_idx = self.match_attempts[attempt_idx].start_word_idx;
            if current_word_idx != start_idx {
                return AttemptState::NoMatch;
            }
        }

        // Try to match the current token
        let word = &self.word_buffer[buffer_idx];
        let product = &self.compiled.token_automata[token_idx];

        match product.min_distance(&word.normalized) {
            Some(distance) => {
                // Token matched! Update the attempt
                let attempt = &mut self.match_attempts[attempt_idx];
                attempt.token_details.push(TokenMatchDetail {
                    token_index: token_idx,
                    byte_range: (word.byte_start, word.byte_end),
                    original_text: word.original.clone(),
                    normalized_text: word.normalized.clone(),
                    distance,
                });
                attempt.total_distance = attempt.total_distance.saturating_add(distance);
                attempt.current_token_idx += 1;
                attempt.last_matched_word_idx = current_word_idx;

                // Check if all tokens are now matched
                if attempt.current_token_idx >= self.compiled.token_automata.len() {
                    AttemptState::Complete
                } else {
                    AttemptState::InProgress
                }
            }
            None => {
                // No match for this word
                if token_idx > 0 {
                    let sep_idx = token_idx - 1;
                    if self.compiled.separators[sep_idx] == Separator::Wildcard {
                        // Wildcard: keep trying with subsequent words
                        AttemptState::NoMatch
                    } else {
                        // Whitespace: consecutive required, so fail
                        AttemptState::Failed
                    }
                } else {
                    // First token didn't match at start position
                    AttemptState::Failed
                }
            }
        }
    }

    /// Finalize a completed match attempt into a TokenMatch.
    fn finalize_attempt(&self, attempt: &MatchAttempt) -> Option<TokenMatch> {
        if attempt.token_details.is_empty() {
            return None;
        }

        let first = &attempt.token_details[0];
        let last = &attempt.token_details[attempt.token_details.len() - 1];
        let byte_range = (first.byte_range.0, last.byte_range.1);

        // Build matched text from token details
        // Note: This may not capture text between tokens (for wildcards)
        // For a complete solution, we'd need to track all words in the range
        let matched_text = attempt
            .token_details
            .iter()
            .map(|d| d.original_text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        Some(TokenMatch {
            byte_range,
            token_matches: attempt.token_details.clone(),
            total_distance: attempt.total_distance,
            matched_text,
        })
    }

    /// Prune old words and cull excessive attempts.
    fn prune_if_needed(&mut self) {
        // Find the minimum start index across all active attempts
        let min_start = self
            .match_attempts
            .iter()
            .map(|a| a.start_word_idx)
            .min()
            .unwrap_or(self.global_word_idx);

        // Calculate how many words to remove from the front
        let words_to_remove = min_start.saturating_sub(self.word_idx_offset);

        if words_to_remove > 0 && words_to_remove <= self.word_buffer.len() {
            // Remove old words
            for _ in 0..words_to_remove {
                self.word_buffer.pop_front();
            }
            self.word_idx_offset += words_to_remove;
        }

        // Enforce maximum buffer size
        while self.word_buffer.len() > MAX_BUFFERED_WORDS {
            self.word_buffer.pop_front();
            self.word_idx_offset += 1;

            // Remove attempts that reference removed words
            self.match_attempts
                .retain(|a| a.start_word_idx >= self.word_idx_offset);
        }

        // Cull low-priority attempts if over limit
        if self.match_attempts.len() > MAX_ACTIVE_ATTEMPTS {
            // Prioritize attempts with more progress (higher current_token_idx)
            self.match_attempts
                .sort_by_key(|a| std::cmp::Reverse(a.current_token_idx));
            self.match_attempts.truncate(MAX_ACTIVE_ATTEMPTS);
        }
    }

    /// Normalize text using phonetic rules if available.
    fn normalize(&self, text: &str) -> String {
        match &self.compiled.rules {
            Some(rules) if !rules.is_empty() => {
                let vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'];
                let input_phones: Vec<PhoneChar> = text
                    .chars()
                    .map(|c| {
                        if vowels.contains(&c) {
                            PhoneChar::Vowel(c)
                        } else {
                            PhoneChar::Consonant(c)
                        }
                    })
                    .collect();

                match apply_rules_seq_char(rules, &input_phones, 100) {
                    Some(phones) => {
                        let mut s = String::new();
                        for p in phones.iter() {
                            match p {
                                PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => s.push(*c),
                                PhoneChar::Digraph(c1, c2) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                }
                                PhoneChar::Trigraph(c1, c2, c3) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                }
                                PhoneChar::Tetragraph(c1, c2, c3, c4) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                }
                                PhoneChar::Pentagraph(c1, c2, c3, c4, c5) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                    s.push(*c5);
                                }
                                PhoneChar::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                    s.push(*c5);
                                    s.push(*c6);
                                }
                                PhoneChar::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                                    s.push(*c1);
                                    s.push(*c2);
                                    s.push(*c3);
                                    s.push(*c4);
                                    s.push(*c5);
                                    s.push(*c6);
                                    s.push(*c7);
                                }
                                PhoneChar::Sequence(chars) => {
                                    for c in chars {
                                        s.push(*c);
                                    }
                                }
                                PhoneChar::Silent => {}
                            }
                        }
                        s
                    }
                    None => text.to_string(),
                }
            }
            _ => text.to_string(),
        }
    }
}

impl TokenGrep {
    /// Create a streaming matcher for this query.
    ///
    /// The streaming matcher allows processing documents word-by-word,
    /// which is useful for large documents that shouldn't be loaded
    /// entirely into memory.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let grep = TokenGrep::new("error .* failed", 0)?;
    /// let mut stream = grep.streaming();
    ///
    /// // Process words from a file reader
    /// for (word, start, end) in word_boundary_iterator(reader) {
    ///     for m in stream.feed_word(&word, start, end) {
    ///         println!("Match: {:?}", m);
    ///     }
    /// }
    /// let remaining = stream.finish();
    /// ```
    pub fn streaming(&self) -> StreamingTokenMatcher {
        StreamingTokenMatcher::new(self.compiled.clone())
    }
}

impl Clone for CompiledTokenQuery {
    fn clone(&self) -> Self {
        Self {
            token_automata: self.token_automata.clone(),
            separators: self.separators.clone(),
            rules: self.rules.clone(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_query() {
        let query = parse_query("hello world", 1).expect("should parse");
        assert_eq!(query.tokens.len(), 2);
        assert_eq!(query.separators.len(), 1);
        assert_eq!(query.separators[0], Separator::Whitespace);

        match &query.tokens[0].pattern {
            TokenPattern::Literal(s) => assert_eq!(s, "hello"),
            _ => panic!("expected literal"),
        }
        assert_eq!(query.tokens[0].max_distance, 1);

        match &query.tokens[1].pattern {
            TokenPattern::Literal(s) => assert_eq!(s, "world"),
            _ => panic!("expected literal"),
        }
        assert_eq!(query.tokens[1].max_distance, 1);
    }

    #[test]
    fn test_parse_explicit_distance() {
        let query = parse_query("error:0 failed:1", 2).expect("should parse");
        assert_eq!(query.tokens.len(), 2);

        match &query.tokens[0].pattern {
            TokenPattern::Literal(s) => assert_eq!(s, "error"),
            _ => panic!("expected literal"),
        }
        assert_eq!(query.tokens[0].max_distance, 0);

        match &query.tokens[1].pattern {
            TokenPattern::Literal(s) => assert_eq!(s, "failed"),
            _ => panic!("expected literal"),
        }
        assert_eq!(query.tokens[1].max_distance, 1);
    }

    #[test]
    fn test_parse_wildcard() {
        let query = parse_query("error .* failed", 1).expect("should parse");
        assert_eq!(query.tokens.len(), 2);
        assert_eq!(query.separators.len(), 1);
        assert_eq!(query.separators[0], Separator::Wildcard);
    }

    #[test]
    fn test_parse_escaped_colon() {
        let query = parse_query(r"error\:404:1", 0).expect("should parse");
        assert_eq!(query.tokens.len(), 1);

        match &query.tokens[0].pattern {
            TokenPattern::Literal(s) => assert_eq!(s, "error:404"),
            _ => panic!("expected literal"),
        }
        assert_eq!(query.tokens[0].max_distance, 1);
    }

    #[test]
    fn test_parse_alternation() {
        let query = parse_query("(phone|fone):1 call", 0).expect("should parse");
        assert_eq!(query.tokens.len(), 2);

        match &query.tokens[0].pattern {
            TokenPattern::Alternation(alts) => {
                assert_eq!(alts.len(), 2);
                match &alts[0] {
                    TokenPattern::Literal(s) => assert_eq!(s, "phone"),
                    _ => panic!("expected literal in alternation"),
                }
                match &alts[1] {
                    TokenPattern::Literal(s) => assert_eq!(s, "fone"),
                    _ => panic!("expected literal in alternation"),
                }
            }
            _ => panic!("expected alternation"),
        }
        assert_eq!(query.tokens[0].max_distance, 1);
    }

    #[test]
    fn test_parse_quoted() {
        let query = parse_query(r#""stack overflow""#, 1).expect("should parse");
        assert_eq!(query.tokens.len(), 1);

        match &query.tokens[0].pattern {
            TokenPattern::Literal(s) => assert_eq!(s, "stack overflow"),
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn test_scan_simple() {
        let grep = TokenGrep::new("hello world", 1).expect("should compile");
        let matches = grep.scan("hello world");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].total_distance, 0);
        assert_eq!(matches[0].matched_text, "hello world");
    }

    #[test]
    fn test_scan_fuzzy() {
        let grep = TokenGrep::new("hello world", 1).expect("should compile");
        let matches = grep.scan("helo wrld");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].total_distance, 2); // 1 + 1
    }

    #[test]
    fn test_scan_no_separator() {
        let grep = TokenGrep::new("hello world", 1).expect("should compile");
        let matches = grep.scan("helloworld"); // No space
        assert_eq!(matches.len(), 0); // Should not match
    }

    #[test]
    fn test_scan_wildcard() {
        let grep = TokenGrep::new("error .* failed", 0).expect("should compile");
        let matches = grep.scan("error in module failed");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].token_matches.len(), 2);
        assert_eq!(matches[0].token_matches[0].original_text, "error");
        assert_eq!(matches[0].token_matches[1].original_text, "failed");
    }

    #[test]
    fn test_scan_explicit_distance() {
        let grep = TokenGrep::new("error:0 failed:1", 0).expect("should compile");

        // Exact "error" + fuzzy "failed"
        let matches = grep.scan("error faild");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].token_matches[0].distance, 0);
        assert_eq!(matches[0].token_matches[1].distance, 1);

        // Fuzzy "error" should fail with :0
        let matches2 = grep.scan("erro failed");
        assert_eq!(matches2.len(), 0);
    }

    #[test]
    fn test_scan_alternation() {
        let grep = TokenGrep::new("(phone|fone) call", 0).expect("should compile");

        let matches1 = grep.scan("phone call");
        assert_eq!(matches1.len(), 1);

        let matches2 = grep.scan("fone call");
        assert_eq!(matches2.len(), 1);
    }

    #[test]
    fn test_escaped_characters() {
        // Test escaped backslash
        let query = parse_query(r"C\\Users", 0).expect("should parse");
        match &query.tokens[0].pattern {
            TokenPattern::Literal(s) => assert_eq!(s, r"C\Users"),
            _ => panic!("expected literal"),
        }

        // Test escaped dot
        let query2 = parse_query(r"1\.0", 0).expect("should parse");
        match &query2.tokens[0].pattern {
            TokenPattern::Literal(s) => assert_eq!(s, "1.0"),
            _ => panic!("expected literal"),
        }
    }

    // ========================================================================
    // Streaming Tests
    // ========================================================================

    #[test]
    fn test_streaming_basic() {
        let grep = TokenGrep::new("hello world", 1).expect("should compile");
        let mut stream = grep.streaming();

        // Feed first word - no match yet
        let m1 = stream.feed_word("hello", 0, 5);
        assert!(m1.is_empty(), "Should not match with only first word");

        // Feed second word - should complete match
        let m2 = stream.feed_word("world", 6, 11);
        assert_eq!(m2.len(), 1, "Should match after second word");
        assert_eq!(m2[0].total_distance, 0);
        assert_eq!(m2[0].token_matches.len(), 2);
        assert_eq!(m2[0].token_matches[0].original_text, "hello");
        assert_eq!(m2[0].token_matches[1].original_text, "world");

        // Finish should return nothing (all matches already returned)
        let final_matches = stream.finish();
        assert!(final_matches.is_empty());
    }

    #[test]
    fn test_streaming_fuzzy() {
        let grep = TokenGrep::new("hello world", 1).expect("should compile");
        let mut stream = grep.streaming();

        stream.feed_word("helo", 0, 4);
        let matches = stream.feed_word("wrld", 5, 9);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].total_distance, 2); // 1 + 1
    }

    #[test]
    fn test_streaming_wildcard() {
        let grep = TokenGrep::new("error .* failed", 0).expect("should compile");
        let mut stream = grep.streaming();

        // Feed words one at a time
        let m1 = stream.feed_word("error", 0, 5);
        assert!(m1.is_empty());

        let m2 = stream.feed_word("in", 6, 8);
        assert!(m2.is_empty());

        let m3 = stream.feed_word("module", 9, 15);
        assert!(m3.is_empty());

        let m4 = stream.feed_word("failed", 16, 22);
        assert_eq!(m4.len(), 1, "Should match with wildcard between tokens");
        assert_eq!(m4[0].token_matches.len(), 2);
        assert_eq!(m4[0].token_matches[0].original_text, "error");
        assert_eq!(m4[0].token_matches[1].original_text, "failed");
    }

    #[test]
    fn test_streaming_no_match() {
        let grep = TokenGrep::new("hello world", 0).expect("should compile");
        let mut stream = grep.streaming();

        stream.feed_word("goodbye", 0, 7);
        let matches = stream.feed_word("world", 8, 13);

        assert!(matches.is_empty(), "Should not match 'goodbye world'");
    }

    #[test]
    fn test_streaming_multiple_matches() {
        let grep = TokenGrep::new("hello world", 1).expect("should compile");
        let mut stream = grep.streaming();

        // First match
        stream.feed_word("hello", 0, 5);
        let m1 = stream.feed_word("world", 6, 11);
        assert_eq!(m1.len(), 1);

        // Second match in same document
        stream.feed_word("hello", 12, 17);
        let m2 = stream.feed_word("world", 18, 23);
        assert_eq!(m2.len(), 1);
    }

    #[test]
    fn test_streaming_reset() {
        let grep = TokenGrep::new("hello world", 1).expect("should compile");
        let mut stream = grep.streaming();

        // Partial match
        stream.feed_word("hello", 0, 5);

        // Reset before completing
        stream.reset();

        // New document
        stream.feed_word("hello", 0, 5);
        let matches = stream.feed_word("world", 6, 11);
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_streaming_explicit_distance() {
        let grep = TokenGrep::new("error:0 failed:1", 0).expect("should compile");
        let mut stream = grep.streaming();

        // Exact "error" + fuzzy "failed"
        stream.feed_word("error", 0, 5);
        let matches = stream.feed_word("faild", 6, 11);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].token_matches[0].distance, 0);
        assert_eq!(matches[0].token_matches[1].distance, 1);

        // Reset and try fuzzy "error" - should fail with :0
        stream.reset();
        stream.feed_word("erro", 0, 4);
        let matches2 = stream.feed_word("failed", 5, 11);
        assert!(matches2.is_empty());
    }

    #[test]
    fn test_streaming_single_token() {
        let grep = TokenGrep::new("hello", 1).expect("should compile");
        let mut stream = grep.streaming();

        let matches = stream.feed_word("hello", 0, 5);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].total_distance, 0);

        // Fuzzy match
        stream.reset();
        let matches2 = stream.feed_word("helo", 0, 4);
        assert_eq!(matches2.len(), 1);
        assert_eq!(matches2[0].total_distance, 1);
    }
}
