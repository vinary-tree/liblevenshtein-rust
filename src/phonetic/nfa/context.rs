//! Contextual pattern matching for phonetic rewrite rules.
//!
//! This module provides context matching (lookahead/lookbehind) for phonetic
//! rewrite rules. Context predicates restrict when a rule can apply based on
//! the surrounding text.
//!
//! # Context Types
//!
//! - **Lookahead** (`_[pattern]`): Pattern must follow the match
//! - **Lookbehind** (`[pattern]_`): Pattern must precede the match
//! - **Word boundary** (`#`): Match at word start or end
//!
//! # Examples
//!
//! ```rust
//! use liblevenshtein::phonetic::nfa::{compile, ContextMatcherChar};
//! use liblevenshtein::phonetic::regex::parse;
//!
//! // c -> s before front vowels (e, i)
//! // Rule: c -> s / _[ei]
//! let regex = parse("[ei]").expect("doc: regex parse must succeed");
//! let right_nfa = compile(&regex).expect("doc: context NFA compilation must succeed");
//! let matcher = ContextMatcherChar::new(None, Some(right_nfa));
//!
//! // "city" at position 0: c before i -> matches
//! assert!(matcher.matches_at("city", 0, 1));
//!
//! // "cat" at position 0: c before a -> doesn't match
//! assert!(!matcher.matches_at("cat", 0, 1));
//! ```
//!
//! # Word Boundary Handling
//!
//! Word boundaries (`#`) match:
//! - At position 0 (start of string) for left context
//! - After the last character for right context
//! - Before/after whitespace or punctuation
//!
//! ```rust
//! use liblevenshtein::phonetic::nfa::ContextMatcherChar;
//!
//! // e -> (empty) at word end
//! // Rule: e -> / _#
//! let matcher = ContextMatcherChar::word_end();
//!
//! // "phone" at position 4: e at end -> matches
//! assert!(matcher.matches_at("phone", 4, 5));
//!
//! // "phonetic" at position 4: e not at end -> doesn't match
//! assert!(!matcher.matches_at("phonetic", 4, 5));
//! ```

use super::{NFAChar, NFA};

/// Kind of context boundary condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundaryKind {
    /// Start of word (`#_`)
    WordStart,
    /// End of word (`_#`)
    WordEnd,
}

impl BoundaryKind {
    /// Check if this boundary matches at the given position.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text
    /// * `pos` - Position in text to check
    ///
    /// # Returns
    ///
    /// `true` if the boundary condition is satisfied at this position.
    pub fn matches_at(&self, text: &str, pos: usize) -> bool {
        match self {
            BoundaryKind::WordStart => {
                pos == 0 || Self::is_word_boundary_char(char_before(text, pos))
            }
            BoundaryKind::WordEnd => Self::is_word_boundary_char(char_at(text, pos)),
        }
    }

    /// Check if this boundary matches at the given byte position.
    pub fn matches_at_bytes(&self, text: &[u8], pos: usize) -> bool {
        match self {
            BoundaryKind::WordStart => {
                pos == 0 || Self::is_word_boundary_byte(text.get(pos.saturating_sub(1)).copied())
            }
            BoundaryKind::WordEnd => {
                pos >= text.len() || Self::is_word_boundary_byte(text.get(pos).copied())
            }
        }
    }

    /// Check if a character is a word boundary character.
    fn is_word_boundary_char(c: Option<char>) -> bool {
        match c {
            None => true, // End of string is a boundary
            Some(c) => !c.is_alphanumeric() && c != '_',
        }
    }

    /// Check if a byte is a word boundary byte.
    fn is_word_boundary_byte(b: Option<u8>) -> bool {
        match b {
            None => true,
            Some(b) => !b.is_ascii_alphanumeric() && b != b'_',
        }
    }
}

fn char_at(text: &str, char_pos: usize) -> Option<char> {
    text.chars().nth(char_pos)
}

fn char_before(text: &str, char_pos: usize) -> Option<char> {
    char_pos
        .checked_sub(1)
        .and_then(|prev_pos| char_at(text, prev_pos))
}

fn byte_index_for_char_pos(text: &str, char_pos: usize) -> Option<usize> {
    if char_pos == 0 {
        return Some(0);
    }

    let mut chars_seen = 0usize;
    for (byte_idx, _) in text.char_indices() {
        if chars_seen == char_pos {
            return Some(byte_idx);
        }
        chars_seen += 1;
    }

    (chars_seen == char_pos).then_some(text.len())
}

fn byte_range_for_char_range(
    text: &str,
    match_start: usize,
    match_end: usize,
) -> Option<(usize, usize)> {
    if match_start > match_end {
        return None;
    }

    let start_byte = byte_index_for_char_pos(text, match_start)?;
    let end_byte = byte_index_for_char_pos(text, match_end)?;
    Some((start_byte, end_byte))
}

#[inline]
fn replacement_splice_capacity(
    text_len: usize,
    removed_len: usize,
    replacement_len: usize,
) -> Option<usize> {
    text_len
        .checked_sub(removed_len)?
        .checked_add(replacement_len)
}

fn char_boundary_indices(text: &str) -> Vec<usize> {
    let char_count = text.chars().count();
    let mut boundaries = Vec::with_capacity(char_count.saturating_add(1));
    boundaries.push(0);
    if text.is_empty() {
        return boundaries;
    }
    boundaries.extend(text.char_indices().skip(1).map(|(byte_idx, _)| byte_idx));
    boundaries.push(text.len());
    boundaries
}

/// Character-level context matcher for rewrite rules.
///
/// Matches context predicates (lookahead/lookbehind) against input text
/// to determine if a rewrite rule should apply.
#[derive(Debug, Clone)]
pub struct ContextMatcherChar {
    /// Left context (lookbehind) NFA - matches text before the pattern
    pub left: Option<ContextPatternChar>,
    /// Right context (lookahead) NFA - matches text after the pattern
    pub right: Option<ContextPatternChar>,
}

/// A context pattern that can be an NFA, boundary, or compound expression.
#[derive(Debug, Clone)]
pub enum ContextPatternChar {
    /// NFA pattern
    Nfa(NFAChar),
    /// Word boundary
    Boundary(BoundaryKind),
    /// Both patterns must match
    And(Box<ContextPatternChar>, Box<ContextPatternChar>),
    /// Either pattern must match
    Or(Box<ContextPatternChar>, Box<ContextPatternChar>),
    /// Pattern must NOT match
    Not(Box<ContextPatternChar>),
}

impl ContextPatternChar {
    /// Check if this context pattern accepts the given input string.
    ///
    /// This is used for testing context patterns in isolation.
    pub fn accepts(&self, input: &str) -> bool {
        match self {
            ContextPatternChar::Nfa(nfa) => nfa.accepts(input),
            ContextPatternChar::Boundary(kind) => match kind {
                BoundaryKind::WordStart => input.is_empty(),
                BoundaryKind::WordEnd => input.is_empty(),
            },
            ContextPatternChar::And(a, b) => a.accepts(input) && b.accepts(input),
            ContextPatternChar::Or(a, b) => a.accepts(input) || b.accepts(input),
            ContextPatternChar::Not(inner) => !inner.accepts(input),
        }
    }
}

impl ContextMatcherChar {
    /// Create a new context matcher with the given left and right contexts.
    ///
    /// # Arguments
    ///
    /// * `left` - Optional left context (lookbehind) NFA
    /// * `right` - Optional right context (lookahead) NFA
    pub fn new(left: Option<NFAChar>, right: Option<NFAChar>) -> Self {
        Self {
            left: left.map(ContextPatternChar::Nfa),
            right: right.map(ContextPatternChar::Nfa),
        }
    }

    /// Create a context matcher that matches at word start.
    pub fn word_start() -> Self {
        Self {
            left: Some(ContextPatternChar::Boundary(BoundaryKind::WordStart)),
            right: None,
        }
    }

    /// Create a context matcher that matches at word end.
    pub fn word_end() -> Self {
        Self {
            left: None,
            right: Some(ContextPatternChar::Boundary(BoundaryKind::WordEnd)),
        }
    }

    /// Create a context matcher with no constraints (always matches).
    pub fn none() -> Self {
        Self {
            left: None,
            right: None,
        }
    }

    /// Create from compiled rewrite context NFAs.
    pub fn from_compiled(left_nfa: Option<NFAChar>, right_nfa: Option<NFAChar>) -> Self {
        // Check if the NFAs represent word boundaries
        // Word boundary NFAs are epsilon-only (accept empty string)
        let left = left_nfa.map(|nfa| {
            if Self::is_word_boundary_nfa(&nfa) {
                ContextPatternChar::Boundary(BoundaryKind::WordStart)
            } else {
                ContextPatternChar::Nfa(nfa)
            }
        });

        let right = right_nfa.map(|nfa| {
            if Self::is_word_boundary_nfa(&nfa) {
                ContextPatternChar::Boundary(BoundaryKind::WordEnd)
            } else {
                ContextPatternChar::Nfa(nfa)
            }
        });

        Self { left, right }
    }

    /// Check if an NFA represents a word boundary (epsilon-only).
    fn is_word_boundary_nfa(nfa: &NFAChar) -> bool {
        // Word boundary NFAs accept empty string and have no character transitions
        nfa.accepts("") && nfa.transitions().iter().all(|t| t.label.is_epsilon())
    }

    /// Check if the context matches at a given position in the text.
    ///
    /// # Arguments
    ///
    /// * `text` - The full input text
    /// * `match_start` - Start position of the pattern match
    /// * `match_end` - End position of the pattern match (exclusive)
    ///
    /// # Returns
    ///
    /// `true` if both left and right contexts (if any) are satisfied.
    pub fn matches_at(&self, text: &str, match_start: usize, match_end: usize) -> bool {
        // Check left context (lookbehind)
        if let Some(ref left_pattern) = self.left {
            if !self.matches_left_context(left_pattern, text, match_start) {
                return false;
            }
        }

        // Check right context (lookahead)
        if let Some(ref right_pattern) = self.right {
            if !self.matches_right_context(right_pattern, text, match_end) {
                return false;
            }
        }

        true
    }

    /// Check if the left context (lookbehind) matches.
    ///
    /// For lookbehind, we need to find a suffix of text[0..match_start]
    /// that the NFA accepts.
    fn matches_left_context(
        &self,
        pattern: &ContextPatternChar,
        text: &str,
        match_start: usize,
    ) -> bool {
        match pattern {
            ContextPatternChar::Boundary(kind) => kind.matches_at(text, match_start),
            ContextPatternChar::Nfa(nfa) => {
                let Some(match_byte) = byte_index_for_char_pos(text, match_start) else {
                    return false;
                };
                let prefix = &text[..match_byte];

                for (start_byte, _) in prefix.char_indices() {
                    if nfa.accepts(&text[start_byte..match_byte]) {
                        return true;
                    }
                }
                nfa.accepts("")
            }
            ContextPatternChar::And(a, b) => {
                self.matches_left_context(a, text, match_start)
                    && self.matches_left_context(b, text, match_start)
            }
            ContextPatternChar::Or(a, b) => {
                self.matches_left_context(a, text, match_start)
                    || self.matches_left_context(b, text, match_start)
            }
            ContextPatternChar::Not(inner) => !self.matches_left_context(inner, text, match_start),
        }
    }

    /// Check if the right context (lookahead) matches.
    ///
    /// For lookahead, we need to find a prefix of text[match_end..]
    /// that the NFA accepts.
    fn matches_right_context(
        &self,
        pattern: &ContextPatternChar,
        text: &str,
        match_end: usize,
    ) -> bool {
        match pattern {
            ContextPatternChar::Boundary(kind) => kind.matches_at(text, match_end),
            ContextPatternChar::Nfa(nfa) => {
                let Some(match_byte) = byte_index_for_char_pos(text, match_end) else {
                    return false;
                };
                if nfa.accepts("") {
                    return true;
                }

                let suffix = &text[match_byte..];
                for (offset, ch) in suffix.char_indices() {
                    let end_byte = match_byte + offset + ch.len_utf8();
                    if nfa.accepts(&text[match_byte..end_byte]) {
                        return true;
                    }
                }
                false
            }
            ContextPatternChar::And(a, b) => {
                self.matches_right_context(a, text, match_end)
                    && self.matches_right_context(b, text, match_end)
            }
            ContextPatternChar::Or(a, b) => {
                self.matches_right_context(a, text, match_end)
                    || self.matches_right_context(b, text, match_end)
            }
            ContextPatternChar::Not(inner) => !self.matches_right_context(inner, text, match_end),
        }
    }
}

/// Byte-level context matcher for rewrite rules.
#[derive(Debug, Clone)]
pub struct ContextMatcher {
    /// Left context (lookbehind) NFA
    pub left: Option<ContextPattern>,
    /// Right context (lookahead) NFA
    pub right: Option<ContextPattern>,
}

/// A byte-level context pattern.
#[derive(Debug, Clone)]
pub enum ContextPattern {
    /// NFA pattern
    Nfa(NFA),
    /// Word boundary
    Boundary(BoundaryKind),
    /// Both patterns must match
    And(Box<ContextPattern>, Box<ContextPattern>),
    /// Either pattern must match
    Or(Box<ContextPattern>, Box<ContextPattern>),
    /// Pattern must NOT match
    Not(Box<ContextPattern>),
}

impl ContextMatcher {
    /// Create a new context matcher.
    pub fn new(left: Option<NFA>, right: Option<NFA>) -> Self {
        Self {
            left: left.map(ContextPattern::Nfa),
            right: right.map(ContextPattern::Nfa),
        }
    }

    /// Create a context matcher that matches at word start.
    pub fn word_start() -> Self {
        Self {
            left: Some(ContextPattern::Boundary(BoundaryKind::WordStart)),
            right: None,
        }
    }

    /// Create a context matcher that matches at word end.
    pub fn word_end() -> Self {
        Self {
            left: None,
            right: Some(ContextPattern::Boundary(BoundaryKind::WordEnd)),
        }
    }

    /// Create a context matcher with no constraints.
    pub fn none() -> Self {
        Self {
            left: None,
            right: None,
        }
    }

    /// Create from compiled rewrite context NFAs.
    pub fn from_compiled(left_nfa: Option<NFA>, right_nfa: Option<NFA>) -> Self {
        let left = left_nfa.map(|nfa| {
            if Self::is_word_boundary_nfa(&nfa) {
                ContextPattern::Boundary(BoundaryKind::WordStart)
            } else {
                ContextPattern::Nfa(nfa)
            }
        });

        let right = right_nfa.map(|nfa| {
            if Self::is_word_boundary_nfa(&nfa) {
                ContextPattern::Boundary(BoundaryKind::WordEnd)
            } else {
                ContextPattern::Nfa(nfa)
            }
        });

        Self { left, right }
    }

    /// Check if an NFA represents a word boundary (epsilon-only).
    fn is_word_boundary_nfa(nfa: &NFA) -> bool {
        nfa.accepts(b"") && nfa.transitions().iter().all(|t| t.label.is_epsilon())
    }

    /// Check if the context matches at a given position.
    pub fn matches_at(&self, text: &[u8], match_start: usize, match_end: usize) -> bool {
        // Check left context
        if let Some(ref left_pattern) = self.left {
            if !self.matches_left_context(left_pattern, text, match_start) {
                return false;
            }
        }

        // Check right context
        if let Some(ref right_pattern) = self.right {
            if !self.matches_right_context(right_pattern, text, match_end) {
                return false;
            }
        }

        true
    }

    /// Check if the left context matches.
    fn matches_left_context(
        &self,
        pattern: &ContextPattern,
        text: &[u8],
        match_start: usize,
    ) -> bool {
        match pattern {
            ContextPattern::Boundary(kind) => kind.matches_at_bytes(text, match_start),
            ContextPattern::Nfa(nfa) => {
                let prefix = &text[..match_start];
                // Try matching any suffix of the prefix
                for start in 0..=prefix.len() {
                    let suffix = &prefix[start..];
                    if nfa.accepts(suffix) {
                        return true;
                    }
                }
                false
            }
            ContextPattern::And(a, b) => {
                self.matches_left_context(a, text, match_start)
                    && self.matches_left_context(b, text, match_start)
            }
            ContextPattern::Or(a, b) => {
                self.matches_left_context(a, text, match_start)
                    || self.matches_left_context(b, text, match_start)
            }
            ContextPattern::Not(inner) => !self.matches_left_context(inner, text, match_start),
        }
    }

    /// Check if the right context matches.
    fn matches_right_context(
        &self,
        pattern: &ContextPattern,
        text: &[u8],
        match_end: usize,
    ) -> bool {
        match pattern {
            ContextPattern::Boundary(kind) => kind.matches_at_bytes(text, match_end),
            ContextPattern::Nfa(nfa) => {
                let suffix = &text[match_end..];
                // Try matching any prefix of the suffix
                for len in 0..=suffix.len() {
                    let prefix = &suffix[..len];
                    if nfa.accepts(prefix) {
                        return true;
                    }
                }
                false
            }
            ContextPattern::And(a, b) => {
                self.matches_right_context(a, text, match_end)
                    && self.matches_right_context(b, text, match_end)
            }
            ContextPattern::Or(a, b) => {
                self.matches_right_context(a, text, match_end)
                    || self.matches_right_context(b, text, match_end)
            }
            ContextPattern::Not(inner) => !self.matches_right_context(inner, text, match_end),
        }
    }
}

/// A rewrite rule with context matching capability.
#[derive(Debug, Clone)]
pub struct ContextualRewriteRuleChar {
    /// NFA matching the source pattern
    pub source: NFAChar,
    /// Characters to replace with
    pub replacement: Vec<char>,
    /// Context matcher for lookahead/lookbehind
    pub context: ContextMatcherChar,
    /// Weight/cost for this rule
    pub weight: f64,
}

impl ContextualRewriteRuleChar {
    /// Create a new contextual rewrite rule.
    pub fn new(
        source: NFAChar,
        replacement: Vec<char>,
        left_context: Option<NFAChar>,
        right_context: Option<NFAChar>,
        weight: f64,
    ) -> Self {
        Self {
            source,
            replacement,
            context: ContextMatcherChar::from_compiled(left_context, right_context),
            weight,
        }
    }

    /// Check if this rule can apply at the given position in text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text
    /// * `match_start` - Start position of potential match
    /// * `match_end` - End position of potential match
    ///
    /// # Returns
    ///
    /// `true` if both the source pattern and context predicates match.
    pub fn can_apply_at(&self, text: &str, match_start: usize, match_end: usize) -> bool {
        // First check if the source pattern matches the substring
        let Some((start_byte, end_byte)) = byte_range_for_char_range(text, match_start, match_end)
        else {
            return false;
        };
        let substring = &text[start_byte..end_byte];
        if !self.source.accepts(substring) {
            return false;
        }

        // Then check context
        self.context.matches_at(text, match_start, match_end)
    }

    /// Apply this rule at a given position, returning the modified string.
    ///
    /// Assumes `can_apply_at` has already returned true.
    pub fn apply_at(&self, text: &str, match_start: usize, match_end: usize) -> String {
        let Some((start_byte, end_byte)) = byte_range_for_char_range(text, match_start, match_end)
        else {
            return text.to_owned();
        };

        let removed_len = end_byte - start_byte;
        let replacement_len: usize = self.replacement.iter().map(|c| c.len_utf8()).sum();
        let mut result = String::with_capacity(
            replacement_splice_capacity(text.len(), removed_len, replacement_len).unwrap_or(0),
        );
        result.push_str(&text[..start_byte]);
        result.extend(self.replacement.iter().copied());
        result.push_str(&text[end_byte..]);
        result
    }

    /// Find the first position where this rule can apply.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text
    /// * `start_from` - Position to start searching from
    ///
    /// # Returns
    ///
    /// `Some((match_start, match_end))` if a match is found, `None` otherwise.
    pub fn find_first_match(&self, text: &str, start_from: usize) -> Option<(usize, usize)> {
        let boundaries = char_boundary_indices(text);
        let char_count = boundaries.len().saturating_sub(1);
        if start_from >= char_count {
            return None;
        }

        for start in start_from..char_count {
            // Try different match lengths starting from this position
            for end in (start + 1)..=char_count {
                let substring = &text[boundaries[start]..boundaries[end]];
                if self.source.accepts(substring) && self.context.matches_at(text, start, end) {
                    return Some((start, end));
                }
            }
        }

        None
    }
}

/// Byte-level contextual rewrite rule.
#[derive(Debug, Clone)]
pub struct ContextualRewriteRule {
    /// NFA matching the source pattern
    pub source: NFA,
    /// Bytes to replace with
    pub replacement: Vec<u8>,
    /// Context matcher
    pub context: ContextMatcher,
    /// Weight/cost for this rule
    pub weight: f64,
}

impl ContextualRewriteRule {
    /// Create a new contextual rewrite rule.
    pub fn new(
        source: NFA,
        replacement: Vec<u8>,
        left_context: Option<NFA>,
        right_context: Option<NFA>,
        weight: f64,
    ) -> Self {
        Self {
            source,
            replacement,
            context: ContextMatcher::from_compiled(left_context, right_context),
            weight,
        }
    }

    /// Check if this rule can apply at the given position.
    pub fn can_apply_at(&self, text: &[u8], match_start: usize, match_end: usize) -> bool {
        let substring = &text[match_start..match_end];
        if !self.source.accepts(substring) {
            return false;
        }
        self.context.matches_at(text, match_start, match_end)
    }

    /// Apply this rule at a given position.
    pub fn apply_at(&self, text: &[u8], match_start: usize, match_end: usize) -> Vec<u8> {
        let removed_len = match_end.saturating_sub(match_start);
        let mut result = Vec::with_capacity(
            replacement_splice_capacity(text.len(), removed_len, self.replacement.len())
                .unwrap_or(0),
        );
        result.extend_from_slice(&text[..match_start]);
        result.extend_from_slice(&self.replacement);
        result.extend_from_slice(&text[match_end..]);
        result
    }

    /// Find the first position where this rule can apply.
    pub fn find_first_match(&self, text: &[u8], start_from: usize) -> Option<(usize, usize)> {
        for start in start_from..text.len() {
            for end in (start + 1)..=text.len() {
                let substring = &text[start..end];
                if self.source.accepts(substring) && self.context.matches_at(text, start, end) {
                    return Some((start, end));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::nfa::compiler::{compile, compile_bytes};
    use crate::phonetic::regex::{parse, parse_bytes};

    // ============================================================================
    // BoundaryKind Tests
    // ============================================================================

    #[test]
    fn test_word_start_at_beginning() {
        assert!(BoundaryKind::WordStart.matches_at("hello", 0));
    }

    #[test]
    fn test_word_start_after_space() {
        assert!(BoundaryKind::WordStart.matches_at("hello world", 6));
    }

    #[test]
    fn test_word_start_mid_word() {
        assert!(!BoundaryKind::WordStart.matches_at("hello", 2));
    }

    #[test]
    fn test_word_end_at_end() {
        assert!(BoundaryKind::WordEnd.matches_at("hello", 5));
    }

    #[test]
    fn test_word_end_before_space() {
        assert!(BoundaryKind::WordEnd.matches_at("hello world", 5));
    }

    #[test]
    fn test_word_end_mid_word() {
        assert!(!BoundaryKind::WordEnd.matches_at("hello", 2));
    }

    // ============================================================================
    // ContextMatcherChar Tests
    // ============================================================================

    #[test]
    fn test_context_matcher_none() {
        let matcher = ContextMatcherChar::none();
        assert!(matcher.matches_at("anything", 0, 8));
    }

    #[test]
    fn test_context_matcher_word_start() {
        let matcher = ContextMatcherChar::word_start();

        // "hello" at position 0 - should match (word start)
        assert!(matcher.matches_at("hello", 0, 1));

        // "hello" at position 2 - should not match (mid-word)
        assert!(!matcher.matches_at("hello", 2, 3));

        // "hello world" at position 6 - should match (after space)
        assert!(matcher.matches_at("hello world", 6, 7));
    }

    #[test]
    fn test_context_matcher_word_end() {
        let matcher = ContextMatcherChar::word_end();

        // "hello" ending at position 5 - should match (word end)
        assert!(matcher.matches_at("hello", 4, 5));

        // "hello" ending at position 2 - should not match (mid-word)
        assert!(!matcher.matches_at("hello", 1, 2));

        // "hello world" ending at position 5 - should match (before space)
        assert!(matcher.matches_at("hello world", 4, 5));
    }

    #[test]
    fn test_context_matcher_lookahead() {
        // c -> s / _[ei]
        let nfa = compile(&parse("[ei]").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let matcher = ContextMatcherChar::new(None, Some(nfa));

        // "city" - c before i at position 0
        assert!(matcher.matches_at("city", 0, 1));

        // "cent" - c before e at position 0
        assert!(matcher.matches_at("cent", 0, 1));

        // "cat" - c before a at position 0
        assert!(!matcher.matches_at("cat", 0, 1));
    }

    #[test]
    fn test_context_matcher_lookbehind() {
        // s -> z / [aeiou]_
        let nfa = compile(&parse("[aeiou]").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let matcher = ContextMatcherChar::new(Some(nfa), None);

        // "roses" - s after o at position 2
        assert!(matcher.matches_at("roses", 2, 3));

        // "star" - s after nothing at position 0
        assert!(!matcher.matches_at("star", 0, 1));
    }

    #[test]
    fn test_context_matcher_both_contexts() {
        // Match s between two vowels
        let left_nfa = compile(&parse("[aeiou]").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let right_nfa = compile(&parse("[aeiou]").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let matcher = ContextMatcherChar::new(Some(left_nfa), Some(right_nfa));

        // "roses" - s between o and e at position 2
        assert!(matcher.matches_at("roses", 2, 3));

        // "star" - s at start (no left vowel)
        assert!(!matcher.matches_at("star", 0, 1));

        // "fast" - s before t (not a vowel on right)
        assert!(!matcher.matches_at("fast", 2, 3));
    }

    // ============================================================================
    // ContextualRewriteRuleChar Tests
    // ============================================================================

    #[test]
    fn test_contextual_rule_simple() {
        // ph -> f (no context)
        let source = compile(&parse("ph").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let rule = ContextualRewriteRuleChar::new(source, vec!['f'], None, None, 0.0);

        assert!(rule.can_apply_at("phone", 0, 2));
        assert_eq!(rule.apply_at("phone", 0, 2), "fone");
    }

    #[test]
    fn test_contextual_rule_rejects_invalid_char_ranges() {
        let source = compile(&parse("e").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let rule = ContextualRewriteRuleChar::new(source, vec![], None, None, 0.0);

        assert!(!rule.can_apply_at("phone", 5, 4));
        assert!(!rule.can_apply_at("phone", 10, 11));
    }

    #[test]
    fn test_contextual_rule_uses_char_offsets_for_utf8_slices() {
        let source = compile(&parse("é").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let right = compile(&parse("β").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let rule = ContextualRewriteRuleChar::new(source, vec!['e'], None, Some(right), 0.0);

        assert!(rule.can_apply_at("aéβ", 1, 2));
        assert_eq!(rule.apply_at("aéβ", 1, 2), "aeβ");
        assert_eq!(rule.find_first_match("xaéβ", 0), Some((2, 3)));
        assert!(!rule.can_apply_at("aéγ", 1, 2));
    }

    #[test]
    fn test_replacement_splice_capacity_rejects_overflow() {
        assert_eq!(replacement_splice_capacity(5, 2, 3), Some(6));
        assert_eq!(replacement_splice_capacity(2, 3, 1), None);
        assert_eq!(replacement_splice_capacity(usize::MAX, 0, 1), None);
    }

    #[test]
    fn test_contextual_rule_with_lookahead() {
        // c -> s / _[ei]
        let source = compile(&parse("c").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let right = compile(&parse("[ei]").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let rule = ContextualRewriteRuleChar::new(source, vec!['s'], None, Some(right), 0.0);

        // "city" - c before i
        assert!(rule.can_apply_at("city", 0, 1));
        assert_eq!(rule.apply_at("city", 0, 1), "sity");

        // "cat" - c before a
        assert!(!rule.can_apply_at("cat", 0, 1));
    }

    #[test]
    fn test_contextual_rule_find_first_match() {
        // c -> s / _[ei]
        let source = compile(&parse("c").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let right = compile(&parse("[ei]").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let rule = ContextualRewriteRuleChar::new(source, vec!['s'], None, Some(right), 0.0);

        // "soccer" - first c is before another c (no match), second c is before e
        let result = rule.find_first_match("soccer", 0);
        assert_eq!(result, Some((3, 4))); // The 'c' before 'e'
    }

    #[test]
    fn test_contextual_rule_word_boundary() {
        // e -> (empty) / _# (silent e at word end)
        let source = compile(&parse("e").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let rule = ContextualRewriteRuleChar {
            source,
            replacement: vec![],
            context: ContextMatcherChar::word_end(),
            weight: 0.0,
        };

        // "phone" - e at word end
        assert!(rule.can_apply_at("phone", 4, 5));
        assert_eq!(rule.apply_at("phone", 4, 5), "phon");

        // "phonetic" - e not at word end
        assert!(!rule.can_apply_at("phonetic", 4, 5));
    }

    // ============================================================================
    // Byte-level Tests
    // ============================================================================

    #[test]
    fn test_byte_context_matcher_none() {
        let matcher = ContextMatcher::none();
        assert!(matcher.matches_at(b"anything", 0, 8));
    }

    #[test]
    fn test_byte_context_matcher_word_start() {
        let matcher = ContextMatcher::word_start();
        assert!(matcher.matches_at(b"hello", 0, 1));
        assert!(!matcher.matches_at(b"hello", 2, 3));
    }

    #[test]
    fn test_byte_context_matcher_lookahead() {
        let nfa = compile_bytes(&parse_bytes(b"[ei]").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let matcher = ContextMatcher::new(None, Some(nfa));

        assert!(matcher.matches_at(b"city", 0, 1));
        assert!(!matcher.matches_at(b"cat", 0, 1));
    }

    #[test]
    fn test_byte_contextual_rule() {
        let source = compile_bytes(&parse_bytes(b"ph").expect("test fixture: parse must be Ok"))
            .expect("test fixture: compile must be Ok");
        let rule = ContextualRewriteRule::new(source, vec![b'f'], None, None, 0.0);

        assert!(rule.can_apply_at(b"phone", 0, 2));
        assert_eq!(rule.apply_at(b"phone", 0, 2), b"fone");
    }
}
