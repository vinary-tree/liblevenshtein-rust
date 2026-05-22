//! Core NFA types for phonetic regular expressions.
//!
//! This module defines the fundamental types for NFA-based phonetic pattern matching:
//! - [`StateId`]: Unique identifier for NFA states
//! - [`NFAState`]: NFA state with acceptance status
//! - [`Transition`]: Labeled transition between states
//! - [`TransitionLabel`]: Transition label (epsilon, char, char class, any)
//! - [`CharClass`]: Character class with ranges and negation
//!
//! # Design
//!
//! Following the existing codebase pattern, this module provides both
//! byte-level (`u8`) and character-level (`char`) implementations:
//!
//! - Byte-level types are optimized for ASCII text (~5% faster, ~4× less memory)
//! - Character-level types support full Unicode (accented chars, CJK, emoji)
//!
//! # Formal Specification
//!
//! See `docs/wfst/nfa_phonetic_regex.md` Section 3 for the formal definition
//! of NFA construction algorithms.
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::nfa::{StateId, NFAState, Transition, TransitionLabel};
//!
//! // Create a simple NFA state
//! let state = NFAState::new(0, false);
//! assert!(!state.is_final);
//!
//! // Create an epsilon transition
//! let trans = Transition::epsilon(0, 1);
//! assert!(trans.label.is_epsilon());
//! ```

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use std::fmt;

// ============================================================================
// State Types
// ============================================================================

/// Unique identifier for NFA states.
///
/// Uses `u32` for memory efficiency (same as existing practice in the codebase).
/// This allows up to ~4 billion states, which is sufficient for any practical NFA.
pub type StateId = u32;

/// An NFA state with unique identifier and acceptance status.
///
/// # Fields
///
/// - `id`: Unique identifier for this state
/// - `is_final`: Whether this is an accepting state
///
/// # Examples
///
/// ```ignore
/// let initial = NFAState::new(0, false);
/// let accepting = NFAState::new(1, true);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct NFAState {
    /// Unique identifier for this state
    pub id: StateId,
    /// Whether this is an accepting (final) state
    pub is_final: bool,
}

impl NFAState {
    /// Create a new NFA state.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this state
    /// * `is_final` - Whether this is an accepting state
    #[inline]
    pub const fn new(id: StateId, is_final: bool) -> Self {
        Self { id, is_final }
    }

    /// Create a new non-final state.
    #[inline]
    pub const fn non_final(id: StateId) -> Self {
        Self::new(id, false)
    }

    /// Create a new final (accepting) state.
    #[inline]
    pub const fn final_state(id: StateId) -> Self {
        Self::new(id, true)
    }
}

impl fmt::Display for NFAState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_final {
            write!(f, "q{}*", self.id)
        } else {
            write!(f, "q{}", self.id)
        }
    }
}

// ============================================================================
// Character Class Types (Character-level)
// ============================================================================

/// A character class representing a set of characters.
///
/// Character classes can be:
/// - Positive: `[abc]` matches a, b, or c
/// - Negative: `[^abc]` matches anything except a, b, or c
/// - Ranges: `[a-z]` matches lowercase letters
///
/// # Examples
///
/// ```ignore
/// // Vowels: [aeiou]
/// let vowels = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']);
///
/// // Non-vowels: [^aeiou]
/// let consonants = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']).negated();
///
/// // Lowercase: [a-z]
/// let lowercase = CharClassChar::from_range('a', 'z');
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct CharClassChar {
    /// Character ranges (inclusive on both ends)
    pub ranges: Vec<(char, char)>,
    /// If true, matches characters NOT in the ranges
    pub negated: bool,
}

impl CharClassChar {
    /// Create an empty character class.
    #[inline]
    pub fn new() -> Self {
        Self {
            ranges: Vec::new(),
            negated: false,
        }
    }

    /// Create a character class from a single range.
    ///
    /// # Arguments
    ///
    /// * `start` - Start of the range (inclusive)
    /// * `end` - End of the range (inclusive)
    #[inline]
    pub fn from_range(start: char, end: char) -> Self {
        Self {
            ranges: vec![(start, end)],
            negated: false,
        }
    }

    /// Create a character class from individual characters.
    pub fn from_chars(chars: &[char]) -> Self {
        let ranges = chars.iter().map(|&c| (c, c)).collect();
        Self {
            ranges,
            negated: false,
        }
    }

    /// Add a range to this character class.
    pub fn add_range(&mut self, start: char, end: char) {
        self.ranges.push((start, end));
    }

    /// Add a single character to this character class.
    pub fn add_char(&mut self, c: char) {
        self.ranges.push((c, c));
    }

    /// Return a negated version of this character class.
    #[inline]
    pub fn negated(mut self) -> Self {
        self.negated = !self.negated;
        self
    }

    /// Check if a character matches this class.
    pub fn matches(&self, c: char) -> bool {
        let in_ranges = self
            .ranges
            .iter()
            .any(|&(start, end)| c >= start && c <= end);
        if self.negated {
            !in_ranges
        } else {
            in_ranges
        }
    }

    /// Check if this is an empty character class.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

impl Default for CharClassChar {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CharClassChar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        if self.negated {
            write!(f, "^")?;
        }
        for (start, end) in &self.ranges {
            if start == end {
                write!(f, "{}", start)?;
            } else {
                write!(f, "{}-{}", start, end)?;
            }
        }
        write!(f, "]")
    }
}

// ============================================================================
// Character Class Types (Byte-level)
// ============================================================================

/// Byte-level character class for ASCII patterns.
///
/// Optimized for ASCII text processing. Uses `u8` instead of `char`
/// for ~4× less memory per edge label and ~5% faster matching.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct CharClass {
    /// Byte ranges (inclusive on both ends)
    pub ranges: Vec<(u8, u8)>,
    /// If true, matches bytes NOT in the ranges
    pub negated: bool,
}

impl CharClass {
    /// Create an empty character class.
    #[inline]
    pub fn new() -> Self {
        Self {
            ranges: Vec::new(),
            negated: false,
        }
    }

    /// Create a character class from a single range.
    #[inline]
    pub fn from_range(start: u8, end: u8) -> Self {
        Self {
            ranges: vec![(start, end)],
            negated: false,
        }
    }

    /// Create a character class from individual bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let ranges = bytes.iter().map(|&b| (b, b)).collect();
        Self {
            ranges,
            negated: false,
        }
    }

    /// Add a range to this character class.
    pub fn add_range(&mut self, start: u8, end: u8) {
        self.ranges.push((start, end));
    }

    /// Add a single byte to this character class.
    pub fn add_byte(&mut self, b: u8) {
        self.ranges.push((b, b));
    }

    /// Return a negated version of this character class.
    #[inline]
    pub fn negated(mut self) -> Self {
        self.negated = !self.negated;
        self
    }

    /// Check if a byte matches this class.
    pub fn matches(&self, b: u8) -> bool {
        let in_ranges = self
            .ranges
            .iter()
            .any(|&(start, end)| b >= start && b <= end);
        if self.negated {
            !in_ranges
        } else {
            in_ranges
        }
    }

    /// Check if this is an empty character class.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

impl Default for CharClass {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CharClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        if self.negated {
            write!(f, "^")?;
        }
        for (start, end) in &self.ranges {
            if start == end {
                write!(f, "{}", *start as char)?;
            } else {
                write!(f, "{}-{}", *start as char, *end as char)?;
            }
        }
        write!(f, "]")
    }
}

// ============================================================================
// Transition Label Types (Character-level)
// ============================================================================

/// Label for an NFA transition (character-level).
///
/// Transition labels determine what input symbols cause a transition:
/// - `Epsilon`: No input consumed (ε-transition)
/// - `Char`: Matches a single character
/// - `CharClass`: Matches any character in the class
/// - `Any`: Matches any single character (.)
///
/// # Formal Definition
///
/// From `docs/wfst/nfa_phonetic_regex.md` Section 3.1:
/// ```text
/// δ: Q × (Σ ∪ {ε}) → P(Q)
/// ```
/// Where Σ is the input alphabet and ε represents epsilon transitions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub enum TransitionLabelChar {
    /// Epsilon transition (no input consumed)
    Epsilon,
    /// Matches a single character
    Char(char),
    /// Matches any character in the class
    CharClass(CharClassChar),
    /// Matches any single character
    Any,

    // Zero-width anchor assertions
    /// Start of line anchor (^) - matches at line start or input start
    StartOfLine,
    /// End of line anchor ($) - matches at line end or input end
    EndOfLine,
    /// Start of input anchor (\A) - matches only at absolute input start
    StartOfInput,
    /// End of input anchor (\Z) - matches at input end, allows trailing newline
    EndOfInput,
    /// Strict end of input anchor (\z) - matches only at absolute input end
    EndOfInputStrict,
}

impl TransitionLabelChar {
    /// Check if this is an epsilon transition.
    #[inline]
    pub fn is_epsilon(&self) -> bool {
        matches!(self, TransitionLabelChar::Epsilon)
    }

    /// Check if this is a zero-width assertion (anchor).
    ///
    /// Zero-width assertions don't consume input but impose positional constraints.
    #[inline]
    pub fn is_anchor(&self) -> bool {
        matches!(
            self,
            TransitionLabelChar::StartOfLine
                | TransitionLabelChar::EndOfLine
                | TransitionLabelChar::StartOfInput
                | TransitionLabelChar::EndOfInput
                | TransitionLabelChar::EndOfInputStrict
        )
    }

    /// Check if a character matches this label.
    ///
    /// Returns `true` for epsilon transitions (they always "match" without consuming input).
    /// Anchors require position-aware matching via `matches_at_position`.
    pub fn matches(&self, c: char) -> bool {
        match self {
            TransitionLabelChar::Epsilon => true,
            TransitionLabelChar::Char(expected) => c == *expected,
            TransitionLabelChar::CharClass(class) => class.matches(c),
            TransitionLabelChar::Any => true,
            // Anchors don't match characters - use matches_at_position
            TransitionLabelChar::StartOfLine
            | TransitionLabelChar::EndOfLine
            | TransitionLabelChar::StartOfInput
            | TransitionLabelChar::EndOfInput
            | TransitionLabelChar::EndOfInputStrict => false,
        }
    }

    /// Check if this anchor matches at the given position in the input.
    ///
    /// # Arguments
    ///
    /// * `input` - The full input string
    /// * `pos` - Current position in the input (byte offset)
    /// * `multiline` - Whether multiline mode is enabled (affects ^ and $)
    ///
    /// # Returns
    ///
    /// `true` if the anchor matches at this position, `false` otherwise.
    /// Returns `false` for non-anchor labels.
    pub fn matches_at_position(&self, input: &str, pos: usize, multiline: bool) -> bool {
        let len = input.len();

        match self {
            TransitionLabelChar::StartOfLine => {
                // ^ matches at start of input
                if pos == 0 {
                    return true;
                }
                // In multiline mode, also matches after newline
                if multiline && pos > 0 {
                    let bytes = input.as_bytes();
                    if pos <= len && bytes.get(pos - 1) == Some(&b'\n') {
                        return true;
                    }
                }
                false
            }
            TransitionLabelChar::EndOfLine => {
                // $ matches at end of input
                if pos == len {
                    return true;
                }
                // In multiline mode, also matches before newline
                if multiline {
                    let bytes = input.as_bytes();
                    if bytes.get(pos) == Some(&b'\n') {
                        return true;
                    }
                }
                false
            }
            TransitionLabelChar::StartOfInput => {
                // \A matches only at absolute start
                pos == 0
            }
            TransitionLabelChar::EndOfInput => {
                // \Z matches at end, optionally before trailing newline
                if pos == len {
                    return true;
                }
                // Also matches before single trailing \n
                if pos == len.saturating_sub(1) {
                    let bytes = input.as_bytes();
                    if bytes.last() == Some(&b'\n') {
                        return true;
                    }
                }
                false
            }
            TransitionLabelChar::EndOfInputStrict => {
                // \z matches only at absolute end
                pos == len
            }
            // Non-anchor labels don't match positions
            _ => false,
        }
    }

    /// Check if this label consumes input.
    ///
    /// Epsilon transitions and anchors don't consume input; all others do.
    #[inline]
    pub fn consumes_input(&self) -> bool {
        !self.is_epsilon() && !self.is_anchor()
    }

    /// Get the expected character if this is a single-character label.
    ///
    /// Returns `Some(c)` for `Char(c)` labels, `None` for all other label types
    /// (Epsilon, Any, CharClass, anchors). This is useful for computing
    /// articulatory distance between input and pattern characters.
    #[inline]
    pub fn expected_char(&self) -> Option<char> {
        match self {
            TransitionLabelChar::Char(c) => Some(*c),
            _ => None,
        }
    }
}

impl fmt::Display for TransitionLabelChar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransitionLabelChar::Epsilon => write!(f, "ε"),
            TransitionLabelChar::Char(c) => write!(f, "{}", c),
            TransitionLabelChar::CharClass(class) => write!(f, "{}", class),
            TransitionLabelChar::Any => write!(f, "."),
            TransitionLabelChar::StartOfLine => write!(f, "^"),
            TransitionLabelChar::EndOfLine => write!(f, "$"),
            TransitionLabelChar::StartOfInput => write!(f, "\\A"),
            TransitionLabelChar::EndOfInput => write!(f, "\\Z"),
            TransitionLabelChar::EndOfInputStrict => write!(f, "\\z"),
        }
    }
}

// ============================================================================
// Transition Label Types (Byte-level)
// ============================================================================

/// Label for an NFA transition (byte-level).
///
/// Byte-level version optimized for ASCII text processing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub enum TransitionLabel {
    /// Epsilon transition (no input consumed)
    Epsilon,
    /// Matches a single byte
    Byte(u8),
    /// Matches any byte in the class
    CharClass(CharClass),
    /// Matches any single byte
    Any,

    // Zero-width anchor assertions
    /// Start of line anchor (^) - matches at line start or input start
    StartOfLine,
    /// End of line anchor ($) - matches at line end or input end
    EndOfLine,
    /// Start of input anchor (\A) - matches only at absolute input start
    StartOfInput,
    /// End of input anchor (\Z) - matches at input end, allows trailing newline
    EndOfInput,
    /// Strict end of input anchor (\z) - matches only at absolute input end
    EndOfInputStrict,
}

impl TransitionLabel {
    /// Check if this is an epsilon transition.
    #[inline]
    pub fn is_epsilon(&self) -> bool {
        matches!(self, TransitionLabel::Epsilon)
    }

    /// Check if this is a zero-width assertion (anchor).
    #[inline]
    pub fn is_anchor(&self) -> bool {
        matches!(
            self,
            TransitionLabel::StartOfLine
                | TransitionLabel::EndOfLine
                | TransitionLabel::StartOfInput
                | TransitionLabel::EndOfInput
                | TransitionLabel::EndOfInputStrict
        )
    }

    /// Check if a byte matches this label.
    pub fn matches(&self, b: u8) -> bool {
        match self {
            TransitionLabel::Epsilon => true,
            TransitionLabel::Byte(expected) => b == *expected,
            TransitionLabel::CharClass(class) => class.matches(b),
            TransitionLabel::Any => true,
            // Anchors don't match bytes - use matches_at_position
            TransitionLabel::StartOfLine
            | TransitionLabel::EndOfLine
            | TransitionLabel::StartOfInput
            | TransitionLabel::EndOfInput
            | TransitionLabel::EndOfInputStrict => false,
        }
    }

    /// Check if this anchor matches at the given position in the input.
    ///
    /// # Arguments
    ///
    /// * `input` - The full input byte slice
    /// * `pos` - Current position in the input
    /// * `multiline` - Whether multiline mode is enabled (affects ^ and $)
    pub fn matches_at_position(&self, input: &[u8], pos: usize, multiline: bool) -> bool {
        let len = input.len();

        match self {
            TransitionLabel::StartOfLine => {
                if pos == 0 {
                    return true;
                }
                if multiline && pos > 0 && input.get(pos - 1) == Some(&b'\n') {
                    return true;
                }
                false
            }
            TransitionLabel::EndOfLine => {
                if pos == len {
                    return true;
                }
                if multiline && input.get(pos) == Some(&b'\n') {
                    return true;
                }
                false
            }
            TransitionLabel::StartOfInput => pos == 0,
            TransitionLabel::EndOfInput => {
                if pos == len {
                    return true;
                }
                if pos == len.saturating_sub(1) && input.last() == Some(&b'\n') {
                    return true;
                }
                false
            }
            TransitionLabel::EndOfInputStrict => pos == len,
            _ => false,
        }
    }

    /// Check if this label consumes input.
    #[inline]
    pub fn consumes_input(&self) -> bool {
        !self.is_epsilon() && !self.is_anchor()
    }
}

impl fmt::Display for TransitionLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransitionLabel::Epsilon => write!(f, "ε"),
            TransitionLabel::Byte(b) => write!(f, "{}", *b as char),
            TransitionLabel::CharClass(class) => write!(f, "{}", class),
            TransitionLabel::Any => write!(f, "."),
            TransitionLabel::StartOfLine => write!(f, "^"),
            TransitionLabel::EndOfLine => write!(f, "$"),
            TransitionLabel::StartOfInput => write!(f, "\\A"),
            TransitionLabel::EndOfInput => write!(f, "\\Z"),
            TransitionLabel::EndOfInputStrict => write!(f, "\\z"),
        }
    }
}

// ============================================================================
// Transition Types (Character-level)
// ============================================================================

/// An NFA transition (character-level).
///
/// A transition represents an edge in the NFA graph from one state to another,
/// labeled with a condition on the input.
///
/// # Fields
///
/// - `from`: Source state
/// - `label`: Transition label (what input triggers this transition)
/// - `to`: Destination state
/// - `weight`: Cost/weight of this transition (for weighted NFAs)
///
/// # Examples
///
/// ```ignore
/// // Epsilon transition from state 0 to state 1
/// let eps = TransitionChar::epsilon(0, 1);
///
/// // Transition on 'a' from state 1 to state 2
/// let on_a = TransitionChar::on_char(1, 'a', 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct TransitionChar {
    /// Source state
    pub from: StateId,
    /// Transition label
    pub label: TransitionLabelChar,
    /// Destination state
    pub to: StateId,
    /// Weight/cost of this transition (default: 0.0)
    pub weight: f64,
}

impl TransitionChar {
    /// Create a new transition with the given label and default weight.
    #[inline]
    pub fn new(from: StateId, label: TransitionLabelChar, to: StateId) -> Self {
        Self {
            from,
            label,
            to,
            weight: 0.0,
        }
    }

    /// Create a new transition with a specific weight.
    #[inline]
    pub fn with_weight(
        from: StateId,
        label: TransitionLabelChar,
        to: StateId,
        weight: f64,
    ) -> Self {
        Self {
            from,
            label,
            to,
            weight,
        }
    }

    /// Create an epsilon transition.
    #[inline]
    pub fn epsilon(from: StateId, to: StateId) -> Self {
        Self::new(from, TransitionLabelChar::Epsilon, to)
    }

    /// Create a transition on a single character.
    #[inline]
    pub fn on_char(from: StateId, c: char, to: StateId) -> Self {
        Self::new(from, TransitionLabelChar::Char(c), to)
    }

    /// Create a transition on a character class.
    #[inline]
    pub fn on_class(from: StateId, class: CharClassChar, to: StateId) -> Self {
        Self::new(from, TransitionLabelChar::CharClass(class), to)
    }

    /// Create a transition that matches any character.
    #[inline]
    pub fn on_any(from: StateId, to: StateId) -> Self {
        Self::new(from, TransitionLabelChar::Any, to)
    }
}

impl fmt::Display for TransitionChar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.weight != 0.0 {
            write!(
                f,
                "q{} --{}[{:.2}]--> q{}",
                self.from, self.label, self.weight, self.to
            )
        } else {
            write!(f, "q{} --{}--> q{}", self.from, self.label, self.to)
        }
    }
}

// ============================================================================
// Transition Types (Byte-level)
// ============================================================================

/// An NFA transition (byte-level).
///
/// Byte-level version optimized for ASCII text processing.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct Transition {
    /// Source state
    pub from: StateId,
    /// Transition label
    pub label: TransitionLabel,
    /// Destination state
    pub to: StateId,
    /// Weight/cost of this transition (default: 0.0)
    pub weight: f64,
}

impl Transition {
    /// Create a new transition with the given label and default weight.
    #[inline]
    pub fn new(from: StateId, label: TransitionLabel, to: StateId) -> Self {
        Self {
            from,
            label,
            to,
            weight: 0.0,
        }
    }

    /// Create a new transition with a specific weight.
    #[inline]
    pub fn with_weight(from: StateId, label: TransitionLabel, to: StateId, weight: f64) -> Self {
        Self {
            from,
            label,
            to,
            weight,
        }
    }

    /// Create an epsilon transition.
    #[inline]
    pub fn epsilon(from: StateId, to: StateId) -> Self {
        Self::new(from, TransitionLabel::Epsilon, to)
    }

    /// Create a transition on a single byte.
    #[inline]
    pub fn on_byte(from: StateId, b: u8, to: StateId) -> Self {
        Self::new(from, TransitionLabel::Byte(b), to)
    }

    /// Create a transition on a character class.
    #[inline]
    pub fn on_class(from: StateId, class: CharClass, to: StateId) -> Self {
        Self::new(from, TransitionLabel::CharClass(class), to)
    }

    /// Create a transition that matches any byte.
    #[inline]
    pub fn on_any(from: StateId, to: StateId) -> Self {
        Self::new(from, TransitionLabel::Any, to)
    }
}

impl fmt::Display for Transition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.weight != 0.0 {
            write!(
                f,
                "q{} --{}[{:.2}]--> q{}",
                self.from, self.label, self.weight, self.to
            )
        } else {
            write!(f, "q{} --{}--> q{}", self.from, self.label, self.to)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- NFAState tests ---

    #[test]
    fn test_nfa_state_new() {
        let state = NFAState::new(0, false);
        assert_eq!(state.id, 0);
        assert!(!state.is_final);
    }

    #[test]
    fn test_nfa_state_final() {
        let state = NFAState::final_state(5);
        assert_eq!(state.id, 5);
        assert!(state.is_final);
    }

    #[test]
    fn test_nfa_state_display() {
        assert_eq!(format!("{}", NFAState::non_final(0)), "q0");
        assert_eq!(format!("{}", NFAState::final_state(1)), "q1*");
    }

    // --- CharClassChar tests ---

    #[test]
    fn test_char_class_from_chars() {
        let vowels = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']);
        assert!(vowels.matches('a'));
        assert!(vowels.matches('e'));
        assert!(!vowels.matches('b'));
    }

    #[test]
    fn test_char_class_from_range() {
        let lowercase = CharClassChar::from_range('a', 'z');
        assert!(lowercase.matches('a'));
        assert!(lowercase.matches('m'));
        assert!(lowercase.matches('z'));
        assert!(!lowercase.matches('A'));
        assert!(!lowercase.matches('0'));
    }

    #[test]
    fn test_char_class_negated() {
        let non_vowels = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']).negated();
        assert!(!non_vowels.matches('a'));
        assert!(non_vowels.matches('b'));
        assert!(non_vowels.matches('z'));
    }

    #[test]
    fn test_char_class_display() {
        let vowels = CharClassChar::from_chars(&['a', 'e', 'i']);
        assert_eq!(format!("{}", vowels), "[aei]");

        let range = CharClassChar::from_range('a', 'z');
        assert_eq!(format!("{}", range), "[a-z]");

        let negated = CharClassChar::from_chars(&['x']).negated();
        assert_eq!(format!("{}", negated), "[^x]");
    }

    // --- CharClass (byte) tests ---

    #[test]
    fn test_byte_class_from_bytes() {
        let vowels = CharClass::from_bytes(&[b'a', b'e', b'i', b'o', b'u']);
        assert!(vowels.matches(b'a'));
        assert!(!vowels.matches(b'b'));
    }

    #[test]
    fn test_byte_class_from_range() {
        let lowercase = CharClass::from_range(b'a', b'z');
        assert!(lowercase.matches(b'a'));
        assert!(lowercase.matches(b'z'));
        assert!(!lowercase.matches(b'A'));
    }

    // --- TransitionLabelChar tests ---

    #[test]
    fn test_label_epsilon() {
        let label = TransitionLabelChar::Epsilon;
        assert!(label.is_epsilon());
        assert!(!label.consumes_input());
        assert!(label.matches('x')); // epsilon always matches
    }

    #[test]
    fn test_label_char() {
        let label = TransitionLabelChar::Char('a');
        assert!(!label.is_epsilon());
        assert!(label.consumes_input());
        assert!(label.matches('a'));
        assert!(!label.matches('b'));
    }

    #[test]
    fn test_label_char_class() {
        let class = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']);
        let label = TransitionLabelChar::CharClass(class);
        assert!(label.matches('a'));
        assert!(!label.matches('b'));
    }

    #[test]
    fn test_label_any() {
        let label = TransitionLabelChar::Any;
        assert!(label.matches('a'));
        assert!(label.matches('z'));
        assert!(label.matches(' '));
    }

    // --- TransitionChar tests ---

    #[test]
    fn test_transition_epsilon() {
        let trans = TransitionChar::epsilon(0, 1);
        assert_eq!(trans.from, 0);
        assert_eq!(trans.to, 1);
        assert!(trans.label.is_epsilon());
        assert_eq!(trans.weight, 0.0);
    }

    #[test]
    fn test_transition_on_char() {
        let trans = TransitionChar::on_char(1, 'a', 2);
        assert_eq!(trans.from, 1);
        assert_eq!(trans.to, 2);
        assert!(trans.label.matches('a'));
    }

    #[test]
    fn test_transition_with_weight() {
        let trans = TransitionChar::with_weight(0, TransitionLabelChar::Char('x'), 1, 0.5);
        assert_eq!(trans.weight, 0.5);
    }

    #[test]
    fn test_transition_display() {
        let eps = TransitionChar::epsilon(0, 1);
        assert_eq!(format!("{}", eps), "q0 --ε--> q1");

        let on_a = TransitionChar::on_char(1, 'a', 2);
        assert_eq!(format!("{}", on_a), "q1 --a--> q2");

        let weighted = TransitionChar::with_weight(0, TransitionLabelChar::Char('x'), 1, 0.15);
        assert_eq!(format!("{}", weighted), "q0 --x[0.15]--> q1");
    }

    // --- Transition (byte) tests ---

    #[test]
    fn test_byte_transition_epsilon() {
        let trans = Transition::epsilon(0, 1);
        assert!(trans.label.is_epsilon());
    }

    #[test]
    fn test_byte_transition_on_byte() {
        let trans = Transition::on_byte(0, b'a', 1);
        assert!(trans.label.matches(b'a'));
        assert!(!trans.label.matches(b'b'));
    }
}
