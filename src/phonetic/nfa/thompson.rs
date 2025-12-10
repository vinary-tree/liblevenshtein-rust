//! Thompson's Construction algorithm for building NFAs from patterns.
//!
//! Thompson's Construction is a systematic method for converting regular expressions
//! to NFAs. It builds NFAs inductively:
//!
//! - **Base cases**: Empty string (ε), single character
//! - **Inductive cases**: Concatenation, alternation (union), Kleene star
//!
//! # Algorithm Properties
//!
//! The resulting NFA has these properties:
//! - O(n) states where n is the pattern length
//! - O(n) transitions
//! - Exactly one start state
//! - Exactly one final state (for each sub-NFA)
//! - No transitions into the start state
//! - No transitions out of the final state
//!
//! # Formal Specification
//!
//! From `docs/wfst/nfa_phonetic_regex.md` Section 3.2:
//!
//! ```text
//! Thompson(ε) = single state that is both start and final
//! Thompson(a) = two states with transition on 'a'
//! Thompson(RS) = Thompson(R) concatenated with Thompson(S)
//! Thompson(R|S) = Thompson(R) unioned with Thompson(S)
//! Thompson(R*) = Kleene star of Thompson(R)
//! ```
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::nfa::ThompsonBuilderChar;
//!
//! let mut builder = ThompsonBuilderChar::new();
//!
//! // Build NFA for "a"
//! let nfa_a = builder.single_char('a');
//!
//! // Build NFA for "b"
//! let nfa_b = builder.single_char('b');
//!
//! // Build NFA for "a|b" (alternation)
//! let nfa_union = builder.alternation(nfa_a, nfa_b);
//!
//! assert!(nfa_union.accepts("a"));
//! assert!(nfa_union.accepts("b"));
//! ```

use super::nfa::{NFAChar, NFA};
use super::types::{CharClass, CharClassChar, TransitionLabel, TransitionLabelChar};

// ============================================================================
// Thompson Builder (Character-level)
// ============================================================================

/// Builder for constructing NFAs using Thompson's Construction (character-level).
///
/// This builder provides methods for creating basic NFAs and combining them
/// using standard regular expression operations.
///
/// # Examples
///
/// ```ignore
/// let mut builder = ThompsonBuilderChar::new();
///
/// // Pattern: (a|b)*c
/// let a = builder.single_char('a');
/// let b = builder.single_char('b');
/// let ab = builder.alternation(a, b);
/// let ab_star = builder.kleene_star(ab);
/// let c = builder.single_char('c');
/// let pattern = builder.concatenate(ab_star, c);
///
/// assert!(pattern.accepts("c"));
/// assert!(pattern.accepts("abc"));
/// assert!(pattern.accepts("aabc"));
/// ```
#[derive(Debug, Default)]
pub struct ThompsonBuilderChar {
    // Currently stateless, but allows for future extensions
    // like state ID generation or statistics tracking
}

impl ThompsonBuilderChar {
    /// Create a new Thompson builder.
    pub fn new() -> Self {
        Self {}
    }

    // ========================================================================
    // Base Cases
    // ========================================================================

    /// Create an NFA that accepts only the empty string (ε).
    ///
    /// ```text
    /// [q0*] (single state, both start and final)
    /// ```
    pub fn epsilon(&self) -> NFAChar {
        NFAChar::with_initial_final(true)
    }

    /// Create an NFA that accepts a single character.
    ///
    /// ```text
    /// [q0] --a--> [q1*]
    /// ```
    pub fn single_char(&self, c: char) -> NFAChar {
        let mut nfa = NFAChar::new();
        let q1 = nfa.add_state(true);
        nfa.add_transition_char(0, c, q1);
        nfa
    }

    /// Create an NFA that accepts any single character (.).
    ///
    /// ```text
    /// [q0] --.--> [q1*]
    /// ```
    pub fn any_char(&self) -> NFAChar {
        let mut nfa = NFAChar::new();
        let q1 = nfa.add_state(true);
        nfa.add_transition(0, TransitionLabelChar::Any, q1);
        nfa
    }

    /// Create an NFA that accepts any character in a class.
    ///
    /// ```text
    /// [q0] --[class]--> [q1*]
    /// ```
    pub fn char_class(&self, class: CharClassChar) -> NFAChar {
        let mut nfa = NFAChar::new();
        let q1 = nfa.add_state(true);
        nfa.add_transition_class(0, class, q1);
        nfa
    }

    /// Create an NFA that accepts a literal string.
    ///
    /// ```text
    /// [q0] --a--> [q1] --b--> [q2] --c--> [q3*]
    /// ```
    pub fn literal(&self, s: &str) -> NFAChar {
        if s.is_empty() {
            return self.epsilon();
        }

        let mut nfa = NFAChar::new();
        let mut current = 0;

        for (i, c) in s.chars().enumerate() {
            let is_last = i == s.chars().count() - 1;
            let next = nfa.add_state(is_last);
            nfa.add_transition_char(current, c, next);
            current = next;
        }

        nfa
    }

    // ========================================================================
    // Inductive Cases
    // ========================================================================

    /// Create an NFA that is the concatenation of two NFAs.
    ///
    /// Accepts strings of the form `xy` where `x ∈ L(a)` and `y ∈ L(b)`.
    ///
    /// ```text
    /// [A] --ε--> [B]
    /// ```
    #[inline]
    pub fn concatenate(&self, a: NFAChar, b: NFAChar) -> NFAChar {
        a.concatenate(b)
    }

    /// Create an NFA that is the alternation (union) of two NFAs.
    ///
    /// Accepts strings in `L(a) ∪ L(b)`.
    ///
    /// ```text
    ///        ε──→[A]──ε──→
    ///       /             \
    /// [q0]─┤               ├──→[qf]
    ///       \             /
    ///        ε──→[B]──ε──→
    /// ```
    #[inline]
    pub fn alternation(&self, a: NFAChar, b: NFAChar) -> NFAChar {
        a.union(b)
    }

    /// Create an NFA that is the Kleene star of an NFA.
    ///
    /// Accepts zero or more repetitions of strings in `L(a)`.
    ///
    /// ```text
    ///             ε (loop)
    ///            ┌───────┐
    ///            │       │
    /// [q0]──ε──→[A]──ε──→[qf]
    ///   │                  ↑
    ///   └────────ε─────────┘
    /// ```
    #[inline]
    pub fn kleene_star(&self, a: NFAChar) -> NFAChar {
        a.kleene_star()
    }

    /// Create an NFA that is the Kleene plus of an NFA.
    ///
    /// Accepts one or more repetitions of strings in `L(a)`.
    /// Equivalent to `a · a*`.
    #[inline]
    pub fn kleene_plus(&self, a: NFAChar) -> NFAChar {
        a.kleene_plus()
    }

    /// Create an NFA that makes the input optional.
    ///
    /// Accepts either the empty string or strings in `L(a)`.
    /// Equivalent to `ε | a`.
    #[inline]
    pub fn optional(&self, a: NFAChar) -> NFAChar {
        a.optional()
    }

    /// Create an NFA that accepts exactly `n` repetitions.
    ///
    /// Equivalent to `a · a · ... · a` (n times).
    pub fn repeat_exact(&self, a: NFAChar, n: usize) -> NFAChar {
        if n == 0 {
            return self.epsilon();
        }

        let mut result = a.clone();
        for _ in 1..n {
            result = self.concatenate(result, a.clone());
        }
        result
    }

    /// Create an NFA that accepts between `min` and `max` repetitions.
    ///
    /// If `max` is `None`, accepts `min` or more repetitions.
    pub fn repeat_range(&self, a: NFAChar, min: usize, max: Option<usize>) -> NFAChar {
        match max {
            Some(max_val) if max_val < min => {
                // Invalid range - return NFA that accepts nothing
                // (empty NFA with no final states)
                NFAChar::new()
            }
            Some(max_val) => {
                // a{min,max} = a^min · (a?)^(max-min)
                let required = self.repeat_exact(a.clone(), min);
                let optional_count = max_val - min;

                if optional_count == 0 {
                    required
                } else {
                    let mut optional_part = self.optional(a.clone());
                    for _ in 1..optional_count {
                        optional_part = self.concatenate(optional_part, self.optional(a.clone()));
                    }
                    self.concatenate(required, optional_part)
                }
            }
            None => {
                // a{min,} = a^min · a*
                let required = self.repeat_exact(a.clone(), min);
                let star = self.kleene_star(a);
                self.concatenate(required, star)
            }
        }
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Create an NFA for a union of multiple alternatives.
    ///
    /// Equivalent to `a | b | c | ...`.
    pub fn union_all(&self, nfas: Vec<NFAChar>) -> NFAChar {
        if nfas.is_empty() {
            return NFAChar::new(); // Empty language
        }

        let mut iter = nfas.into_iter();
        let mut result = iter.next().expect("at least one NFA");

        for nfa in iter {
            result = self.alternation(result, nfa);
        }

        result
    }

    /// Create an NFA for a concatenation of multiple parts.
    ///
    /// Equivalent to `a · b · c · ...`.
    pub fn concat_all(&self, nfas: Vec<NFAChar>) -> NFAChar {
        if nfas.is_empty() {
            return self.epsilon();
        }

        let mut iter = nfas.into_iter();
        let mut result = iter.next().expect("at least one NFA");

        for nfa in iter {
            result = self.concatenate(result, nfa);
        }

        result
    }
}

// ============================================================================
// Thompson Builder (Byte-level)
// ============================================================================

/// Builder for constructing NFAs using Thompson's Construction (byte-level).
///
/// Byte-level version optimized for ASCII text processing.
#[derive(Debug, Default)]
pub struct ThompsonBuilder {
    // Currently stateless
}

impl ThompsonBuilder {
    /// Create a new Thompson builder.
    pub fn new() -> Self {
        Self {}
    }

    // ========================================================================
    // Base Cases
    // ========================================================================

    /// Create an NFA that accepts only the empty string (ε).
    pub fn epsilon(&self) -> NFA {
        NFA::with_initial_final(true)
    }

    /// Create an NFA that accepts a single byte.
    pub fn single_byte(&self, b: u8) -> NFA {
        let mut nfa = NFA::new();
        let q1 = nfa.add_state(true);
        nfa.add_transition_byte(0, b, q1);
        nfa
    }

    /// Create an NFA that accepts any single byte (.).
    pub fn any_byte(&self) -> NFA {
        let mut nfa = NFA::new();
        let q1 = nfa.add_state(true);
        nfa.add_transition(0, TransitionLabel::Any, q1);
        nfa
    }

    /// Create an NFA that accepts any byte in a class.
    pub fn byte_class(&self, class: CharClass) -> NFA {
        let mut nfa = NFA::new();
        let q1 = nfa.add_state(true);
        nfa.add_transition_class(0, class, q1);
        nfa
    }

    /// Create an NFA that accepts a literal byte string.
    pub fn literal(&self, s: &[u8]) -> NFA {
        if s.is_empty() {
            return self.epsilon();
        }

        let mut nfa = NFA::new();
        let mut current = 0;

        for (i, &b) in s.iter().enumerate() {
            let is_last = i == s.len() - 1;
            let next = nfa.add_state(is_last);
            nfa.add_transition_byte(current, b, next);
            current = next;
        }

        nfa
    }

    /// Create an NFA that accepts a literal string (as UTF-8 bytes).
    pub fn literal_str(&self, s: &str) -> NFA {
        self.literal(s.as_bytes())
    }

    // ========================================================================
    // Inductive Cases
    // ========================================================================

    /// Create an NFA that is the concatenation of two NFAs.
    #[inline]
    pub fn concatenate(&self, a: NFA, b: NFA) -> NFA {
        a.concatenate(b)
    }

    /// Create an NFA that is the alternation (union) of two NFAs.
    #[inline]
    pub fn alternation(&self, a: NFA, b: NFA) -> NFA {
        a.union(b)
    }

    /// Create an NFA that is the Kleene star of an NFA.
    #[inline]
    pub fn kleene_star(&self, a: NFA) -> NFA {
        a.kleene_star()
    }

    /// Create an NFA that is the Kleene plus of an NFA.
    #[inline]
    pub fn kleene_plus(&self, a: NFA) -> NFA {
        a.kleene_plus()
    }

    /// Create an NFA that makes the input optional.
    #[inline]
    pub fn optional(&self, a: NFA) -> NFA {
        a.optional()
    }

    /// Create an NFA that accepts exactly `n` repetitions.
    pub fn repeat_exact(&self, a: NFA, n: usize) -> NFA {
        if n == 0 {
            return self.epsilon();
        }

        let mut result = a.clone();
        for _ in 1..n {
            result = self.concatenate(result, a.clone());
        }
        result
    }

    /// Create an NFA that accepts between `min` and `max` repetitions.
    pub fn repeat_range(&self, a: NFA, min: usize, max: Option<usize>) -> NFA {
        match max {
            Some(max_val) if max_val < min => {
                NFA::new() // Invalid range
            }
            Some(max_val) => {
                let required = self.repeat_exact(a.clone(), min);
                let optional_count = max_val - min;

                if optional_count == 0 {
                    required
                } else {
                    let mut optional_part = self.optional(a.clone());
                    for _ in 1..optional_count {
                        optional_part = self.concatenate(optional_part, self.optional(a.clone()));
                    }
                    self.concatenate(required, optional_part)
                }
            }
            None => {
                let required = self.repeat_exact(a.clone(), min);
                let star = self.kleene_star(a);
                self.concatenate(required, star)
            }
        }
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Create an NFA for a union of multiple alternatives.
    pub fn union_all(&self, nfas: Vec<NFA>) -> NFA {
        if nfas.is_empty() {
            return NFA::new();
        }

        let mut iter = nfas.into_iter();
        let mut result = iter.next().expect("at least one NFA");

        for nfa in iter {
            result = self.alternation(result, nfa);
        }

        result
    }

    /// Create an NFA for a concatenation of multiple parts.
    pub fn concat_all(&self, nfas: Vec<NFA>) -> NFA {
        if nfas.is_empty() {
            return self.epsilon();
        }

        let mut iter = nfas.into_iter();
        let mut result = iter.next().expect("at least one NFA");

        for nfa in iter {
            result = self.concatenate(result, nfa);
        }

        result
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- ThompsonBuilderChar tests ---

    #[test]
    fn test_thompson_epsilon() {
        let builder = ThompsonBuilderChar::new();
        let nfa = builder.epsilon();

        assert!(nfa.accepts(""));
        assert!(!nfa.accepts("a"));
    }

    #[test]
    fn test_thompson_single_char() {
        let builder = ThompsonBuilderChar::new();
        let nfa = builder.single_char('a');

        assert!(nfa.accepts("a"));
        assert!(!nfa.accepts("b"));
        assert!(!nfa.accepts(""));
        assert!(!nfa.accepts("aa"));
    }

    #[test]
    fn test_thompson_any_char() {
        let builder = ThompsonBuilderChar::new();
        let nfa = builder.any_char();

        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("b"));
        assert!(nfa.accepts("z"));
        assert!(!nfa.accepts(""));
        assert!(!nfa.accepts("ab"));
    }

    #[test]
    fn test_thompson_char_class() {
        let builder = ThompsonBuilderChar::new();
        let vowels = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']);
        let nfa = builder.char_class(vowels);

        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("e"));
        assert!(nfa.accepts("i"));
        assert!(!nfa.accepts("b"));
        assert!(!nfa.accepts(""));
    }

    #[test]
    fn test_thompson_literal() {
        let builder = ThompsonBuilderChar::new();
        let nfa = builder.literal("hello");

        assert!(nfa.accepts("hello"));
        assert!(!nfa.accepts("hell"));
        assert!(!nfa.accepts("helloo"));
        assert!(!nfa.accepts(""));
    }

    #[test]
    fn test_thompson_literal_empty() {
        let builder = ThompsonBuilderChar::new();
        let nfa = builder.literal("");

        assert!(nfa.accepts(""));
        assert!(!nfa.accepts("a"));
    }

    #[test]
    fn test_thompson_concatenate() {
        let builder = ThompsonBuilderChar::new();
        let nfa_a = builder.single_char('a');
        let nfa_b = builder.single_char('b');
        let nfa = builder.concatenate(nfa_a, nfa_b);

        assert!(nfa.accepts("ab"));
        assert!(!nfa.accepts("a"));
        assert!(!nfa.accepts("b"));
        assert!(!nfa.accepts("ba"));
    }

    #[test]
    fn test_thompson_alternation() {
        let builder = ThompsonBuilderChar::new();
        let nfa_a = builder.single_char('a');
        let nfa_b = builder.single_char('b');
        let nfa = builder.alternation(nfa_a, nfa_b);

        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("b"));
        assert!(!nfa.accepts("c"));
        assert!(!nfa.accepts("ab"));
        assert!(!nfa.accepts(""));
    }

    #[test]
    fn test_thompson_kleene_star() {
        let builder = ThompsonBuilderChar::new();
        let nfa_a = builder.single_char('a');
        let nfa = builder.kleene_star(nfa_a);

        assert!(nfa.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(nfa.accepts("aaaa"));
        assert!(!nfa.accepts("b"));
        assert!(!nfa.accepts("ab"));
    }

    #[test]
    fn test_thompson_kleene_plus() {
        let builder = ThompsonBuilderChar::new();
        let nfa_a = builder.single_char('a');
        let nfa = builder.kleene_plus(nfa_a);

        assert!(!nfa.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(!nfa.accepts("b"));
    }

    #[test]
    fn test_thompson_optional() {
        let builder = ThompsonBuilderChar::new();
        let nfa_a = builder.single_char('a');
        let nfa = builder.optional(nfa_a);

        assert!(nfa.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(!nfa.accepts("aa"));
        assert!(!nfa.accepts("b"));
    }

    #[test]
    fn test_thompson_repeat_exact() {
        let builder = ThompsonBuilderChar::new();
        let nfa_a = builder.single_char('a');

        // a{3}
        let nfa = builder.repeat_exact(nfa_a, 3);

        assert!(!nfa.accepts(""));
        assert!(!nfa.accepts("a"));
        assert!(!nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(!nfa.accepts("aaaa"));
    }

    #[test]
    fn test_thompson_repeat_range() {
        let builder = ThompsonBuilderChar::new();
        let nfa_a = builder.single_char('a');

        // a{2,4}
        let nfa = builder.repeat_range(nfa_a, 2, Some(4));

        assert!(!nfa.accepts(""));
        assert!(!nfa.accepts("a"));
        assert!(nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(nfa.accepts("aaaa"));
        assert!(!nfa.accepts("aaaaa"));
    }

    #[test]
    fn test_thompson_repeat_range_unbounded() {
        let builder = ThompsonBuilderChar::new();
        let nfa_a = builder.single_char('a');

        // a{2,}
        let nfa = builder.repeat_range(nfa_a, 2, None);

        assert!(!nfa.accepts(""));
        assert!(!nfa.accepts("a"));
        assert!(nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(nfa.accepts("aaaa"));
        assert!(nfa.accepts("aaaaa"));
    }

    #[test]
    fn test_thompson_union_all() {
        let builder = ThompsonBuilderChar::new();
        let nfas = vec![
            builder.single_char('a'),
            builder.single_char('b'),
            builder.single_char('c'),
        ];
        let nfa = builder.union_all(nfas);

        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("b"));
        assert!(nfa.accepts("c"));
        assert!(!nfa.accepts("d"));
        assert!(!nfa.accepts("ab"));
    }

    #[test]
    fn test_thompson_concat_all() {
        let builder = ThompsonBuilderChar::new();
        let nfas = vec![
            builder.single_char('a'),
            builder.single_char('b'),
            builder.single_char('c'),
        ];
        let nfa = builder.concat_all(nfas);

        assert!(nfa.accepts("abc"));
        assert!(!nfa.accepts("a"));
        assert!(!nfa.accepts("ab"));
        assert!(!nfa.accepts("abcd"));
    }

    // --- Complex patterns ---

    #[test]
    fn test_thompson_complex_pattern() {
        // Pattern: (a|b)*c
        let builder = ThompsonBuilderChar::new();

        let a = builder.single_char('a');
        let b = builder.single_char('b');
        let c = builder.single_char('c');

        let ab = builder.alternation(a, b);
        let ab_star = builder.kleene_star(ab);
        let pattern = builder.concatenate(ab_star, c);

        assert!(pattern.accepts("c"));
        assert!(pattern.accepts("ac"));
        assert!(pattern.accepts("bc"));
        assert!(pattern.accepts("aac"));
        assert!(pattern.accepts("abc"));
        assert!(pattern.accepts("bac"));
        assert!(pattern.accepts("aabbc"));
        assert!(!pattern.accepts(""));
        assert!(!pattern.accepts("a"));
        assert!(!pattern.accepts("ab"));
        assert!(!pattern.accepts("ca"));
    }

    #[test]
    fn test_thompson_phonetic_pattern() {
        // Pattern: (ph|f)one - matches "phone" or "fone"
        let builder = ThompsonBuilderChar::new();

        let ph = builder.literal("ph");
        let f = builder.single_char('f');
        let ph_or_f = builder.alternation(ph, f);
        let one = builder.literal("one");
        let pattern = builder.concatenate(ph_or_f, one);

        assert!(pattern.accepts("phone"));
        assert!(pattern.accepts("fone"));
        assert!(!pattern.accepts("bone"));
        assert!(!pattern.accepts("phon"));
        assert!(!pattern.accepts(""));
    }

    // --- Byte-level tests ---

    #[test]
    fn test_thompson_byte_literal() {
        let builder = ThompsonBuilder::new();
        let nfa = builder.literal_str("hello");

        assert!(nfa.accepts_str("hello"));
        assert!(!nfa.accepts_str("hell"));
    }

    #[test]
    fn test_thompson_byte_alternation() {
        let builder = ThompsonBuilder::new();
        let a = builder.single_byte(b'a');
        let b = builder.single_byte(b'b');
        let nfa = builder.alternation(a, b);

        assert!(nfa.accepts_str("a"));
        assert!(nfa.accepts_str("b"));
        assert!(!nfa.accepts_str("c"));
    }

    #[test]
    fn test_thompson_byte_kleene_star() {
        let builder = ThompsonBuilder::new();
        let a = builder.single_byte(b'a');
        let nfa = builder.kleene_star(a);

        assert!(nfa.accepts_str(""));
        assert!(nfa.accepts_str("a"));
        assert!(nfa.accepts_str("aaa"));
    }
}
