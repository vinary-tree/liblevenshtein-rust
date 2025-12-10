//! Type definitions for phonetic rewrite rules.
//!
//! This module provides the core types for representing phonetic rewrite rules,
//! directly translated from the Coq/Rocq verification in
//! `docs/verification/phonetic/rewrite_rules.v`.
//!
//! Two versions of each type are provided:
//! - Byte-level (`Phone`, `Context`, `RewriteRule`) using `u8` - optimized for ASCII
//! - Character-level (`PhoneChar`, `ContextChar`, `RewriteRuleChar`) using `char` - proper Unicode support
//!
//! # Serialization
//!
//! All types support serde serialization when the `serialization` feature is enabled.
//! This allows compiled rule sets to be saved to binary format for faster loading.
//!
//! # Formal Specification
//!
//! These types are direct translations of the Coq definitions:
//!
//! ```coq
//! Inductive Phone : Type :=
//!   | Vowel : ascii -> Phone
//!   | Consonant : ascii -> Phone
//!   | Digraph : ascii -> ascii -> Phone
//!   | Silent : Phone.
//!
//! Inductive Context : Type :=
//!   | Initial : Context
//!   | Final : Context
//!   | BeforeVowel : list ascii -> Context
//!   | AfterConsonant : list ascii -> Context
//!   | BeforeConsonant : list ascii -> Context
//!   | AfterVowel : list ascii -> Context
//!   | Anywhere : Context.
//!
//! Record RewriteRule : Type := {
//!   rule_id : nat;
//!   rule_name : string;
//!   pattern : list Phone;
//!   replacement : list Phone;
//!   context : Context;
//!   weight : Q
//! }.
//! ```
//!
//! See `docs/verification/phonetic/rewrite_rules.v` for the complete formal specification.

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Byte-level types (u8) - optimized for ASCII text
// ============================================================================

/// A phonetic unit representing a single sound.
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:30-34`
///
/// # Variants
///
/// - `Vowel(u8)` - A vowel sound (e.g., 'a', 'e', 'i', 'o', 'u')
/// - `Consonant(u8)` - A consonant sound (e.g., 'b', 'k', 'p')
/// - `Digraph(u8, u8)` - A two-character sound unit (e.g., 'ch', 'sh', 'th')
/// - `Silent` - A silent letter (not pronounced)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub enum Phone {
    /// A vowel sound
    Vowel(u8),
    /// A consonant sound
    Consonant(u8),
    /// A two-character sound unit (digraph)
    Digraph(u8, u8),
    /// A silent letter
    Silent,
}

/// Context specification for when a rule applies (byte-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:48-55`
///
/// # Variants
///
/// - `Initial` - At the beginning of a word
/// - `Final` - At the end of a word
/// - `BeforeVowel(Vec<u8>)` - Before specific vowels
/// - `AfterConsonant(Vec<u8>)` - After specific consonants
/// - `BeforeConsonant(Vec<u8>)` - Before specific consonants
/// - `AfterVowel(Vec<u8>)` - After specific vowels
/// - `Anywhere` - No context restriction
/// - `And(Box<Context>, Box<Context>)` - Both contexts must match
/// - `Or(Box<Context>, Box<Context>)` - Either context must match
/// - `Not(Box<Context>)` - Context must NOT match
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub enum Context {
    /// At the beginning of a word
    Initial,
    /// At the end of a word
    Final,
    /// Before specific vowels
    BeforeVowel(Vec<u8>),
    /// After specific consonants
    AfterConsonant(Vec<u8>),
    /// Before specific consonants
    BeforeConsonant(Vec<u8>),
    /// After specific vowels
    AfterVowel(Vec<u8>),
    /// No context restriction
    Anywhere,
    /// Compound: both contexts must match
    And(Box<Context>, Box<Context>),
    /// Compound: either context must match
    Or(Box<Context>, Box<Context>),
    /// Negated: context must NOT match
    Not(Box<Context>),
}

impl Context {
    /// Returns true if this context depends on string length.
    ///
    /// **Formal Specification**: `docs/verification/phonetic/position_skipping_proof.v:1202-1211`
    ///
    /// Only `Final` is position-dependent because:
    /// - `Final` matches when `pos == s.len()` (depends on string length)
    /// - All other contexts depend only on local structure (position within bounds, adjacent characters)
    ///
    /// This method is used to determine whether position skipping optimization is safe.
    /// Position skipping is SAFE when no rules use `Context::Final`.
    ///
    /// # Counter-example (why Final is position-dependent)
    ///
    /// From `position_skipping_proof.v:3424-3444`:
    /// - Original: `s = [a, b, c]` (length 3), position 2 is NOT final
    /// - After shortening: `s' = [a, b]` (length 2), position 2 IS now final
    /// - A rule with `Final` context might match at a position that was previously skipped
    ///
    /// # Compound contexts
    ///
    /// For compound contexts (And, Or, Not), position-dependence is propagated:
    /// - `And(a, b)` is position-dependent if either `a` or `b` is
    /// - `Or(a, b)` is position-dependent if either `a` or `b` is
    /// - `Not(inner)` is position-dependent if `inner` is
    #[inline]
    pub fn is_position_dependent(&self) -> bool {
        match self {
            Context::Final => true,
            Context::And(a, b) => a.is_position_dependent() || b.is_position_dependent(),
            Context::Or(a, b) => a.is_position_dependent() || b.is_position_dependent(),
            Context::Not(inner) => inner.is_position_dependent(),
            _ => false,
        }
    }
}

/// A phonetic rewrite rule (byte-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:62-68`
///
/// Represents a transformation from a pattern of phones to a replacement
/// sequence, applicable in a specific context.
///
/// # Fields
///
/// - `rule_id` - Unique identifier for the rule
/// - `rule_name` - Human-readable name (for debugging/documentation)
/// - `pattern` - Sequence of phones to match
/// - `replacement` - Sequence of phones to substitute
/// - `context` - Context in which the rule applies
/// - `weight` - Priority weight (higher = applied first)
///
/// # Formal Properties
///
/// Well-formed rules satisfy (Theorem 1, `zompist_rules.v:285`):
/// - Pattern is non-empty: `pattern.len() > 0`
/// - Replacement is bounded: `replacement.len() <= pattern.len() + MAX_EXPANSION_FACTOR`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct RewriteRule {
    /// Unique identifier for the rule
    pub rule_id: usize,
    /// Human-readable name
    pub rule_name: String,
    /// Pattern to match (sequence of phones)
    pub pattern: Vec<Phone>,
    /// Replacement sequence
    pub replacement: Vec<Phone>,
    /// Context specification
    pub context: Context,
    /// Priority weight (higher = applied first)
    pub weight: f64,
}

// ============================================================================
// Character-level types (char) - proper Unicode support
// ============================================================================

/// A phonetic unit representing a single sound (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:30-34`
///
/// This is the character-level variant of [`Phone`] for proper Unicode support.
/// Use when working with non-ASCII text (accented characters, CJK, emoji, etc.).
///
/// # Variants
///
/// - `Vowel(char)` - A vowel sound
/// - `Consonant(char)` - A consonant sound
/// - `Digraph(char, char)` - A two-character sound unit
/// - `Silent` - A silent letter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub enum PhoneChar {
    /// A vowel sound
    Vowel(char),
    /// A consonant sound
    Consonant(char),
    /// A two-character sound unit (digraph)
    Digraph(char, char),
    /// A silent letter
    Silent,
}

/// Context specification for when a rule applies (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:48-55`
///
/// This is the character-level variant of [`Context`] for proper Unicode support.
///
/// # Variants
///
/// - `Initial` - At the beginning of a word
/// - `Final` - At the end of a word
/// - `BeforeVowel(Vec<char>)` - Before specific vowels
/// - `AfterConsonant(Vec<char>)` - After specific consonants
/// - `BeforeConsonant(Vec<char>)` - Before specific consonants
/// - `AfterVowel(Vec<char>)` - After specific vowels
/// - `Anywhere` - No context restriction
/// - `And(Box<ContextChar>, Box<ContextChar>)` - Both contexts must match
/// - `Or(Box<ContextChar>, Box<ContextChar>)` - Either context must match
/// - `Not(Box<ContextChar>)` - Context must NOT match
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub enum ContextChar {
    /// At the beginning of a word
    Initial,
    /// At the end of a word
    Final,
    /// Before specific vowels
    BeforeVowel(Vec<char>),
    /// After specific consonants
    AfterConsonant(Vec<char>),
    /// Before specific consonants
    BeforeConsonant(Vec<char>),
    /// After specific vowels
    AfterVowel(Vec<char>),
    /// No context restriction
    Anywhere,
    /// Compound: both contexts must match
    And(Box<ContextChar>, Box<ContextChar>),
    /// Compound: either context must match
    Or(Box<ContextChar>, Box<ContextChar>),
    /// Negated: context must NOT match
    Not(Box<ContextChar>),
}

impl ContextChar {
    /// Returns true if this context depends on string length.
    ///
    /// **Formal Specification**: `docs/verification/phonetic/position_skipping_proof.v:1202-1211`
    ///
    /// Character-level variant of [`Context::is_position_dependent`].
    /// See that method for detailed documentation.
    ///
    /// # Compound contexts
    ///
    /// For compound contexts (And, Or, Not), position-dependence is propagated:
    /// - `And(a, b)` is position-dependent if either `a` or `b` is
    /// - `Or(a, b)` is position-dependent if either `a` or `b` is
    /// - `Not(inner)` is position-dependent if `inner` is
    #[inline]
    pub fn is_position_dependent(&self) -> bool {
        match self {
            ContextChar::Final => true,
            ContextChar::And(a, b) => a.is_position_dependent() || b.is_position_dependent(),
            ContextChar::Or(a, b) => a.is_position_dependent() || b.is_position_dependent(),
            ContextChar::Not(inner) => inner.is_position_dependent(),
            _ => false,
        }
    }
}

/// A phonetic rewrite rule (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:62-68`
///
/// This is the character-level variant of [`RewriteRule`] for proper Unicode support.
///
/// # Fields
///
/// - `rule_id` - Unique identifier for the rule
/// - `rule_name` - Human-readable name (for debugging/documentation)
/// - `pattern` - Sequence of phones to match
/// - `replacement` - Sequence of phones to substitute
/// - `context` - Context in which the rule applies
/// - `weight` - Priority weight (higher = applied first)
///
/// # Formal Properties
///
/// Well-formed rules satisfy (Theorem 1, `zompist_rules.v:285`):
/// - Pattern is non-empty: `pattern.len() > 0`
/// - Replacement is bounded: `replacement.len() <= pattern.len() + MAX_EXPANSION_FACTOR`
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct RewriteRuleChar {
    /// Unique identifier for the rule
    pub rule_id: usize,
    /// Human-readable name
    pub rule_name: String,
    /// Pattern to match (sequence of phones)
    pub pattern: Vec<PhoneChar>,
    /// Replacement sequence
    pub replacement: Vec<PhoneChar>,
    /// Context specification
    pub context: ContextChar,
    /// Priority weight (higher = applied first)
    pub weight: f64,
}

// ============================================================================
// Display implementations
// ============================================================================

impl std::fmt::Display for Phone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Phone::Vowel(c) => write!(f, "V({})", *c as char),
            Phone::Consonant(c) => write!(f, "C({})", *c as char),
            Phone::Digraph(c1, c2) => write!(f, "D({},{})", *c1 as char, *c2 as char),
            Phone::Silent => write!(f, "Silent"),
        }
    }
}

impl std::fmt::Display for PhoneChar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhoneChar::Vowel(c) => write!(f, "V({})", c),
            PhoneChar::Consonant(c) => write!(f, "C({})", c),
            PhoneChar::Digraph(c1, c2) => write!(f, "D({},{})", c1, c2),
            PhoneChar::Silent => write!(f, "Silent"),
        }
    }
}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Context::Initial => write!(f, "Initial"),
            Context::Final => write!(f, "Final"),
            Context::BeforeVowel(cs) => {
                write!(f, "BeforeVowel({})", String::from_utf8_lossy(cs))
            }
            Context::AfterConsonant(cs) => {
                write!(f, "AfterConsonant({})", String::from_utf8_lossy(cs))
            }
            Context::BeforeConsonant(cs) => {
                write!(f, "BeforeConsonant({})", String::from_utf8_lossy(cs))
            }
            Context::AfterVowel(cs) => {
                write!(f, "AfterVowel({})", String::from_utf8_lossy(cs))
            }
            Context::Anywhere => write!(f, "Anywhere"),
            Context::And(a, b) => write!(f, "And({}, {})", a, b),
            Context::Or(a, b) => write!(f, "Or({}, {})", a, b),
            Context::Not(inner) => write!(f, "Not({})", inner),
        }
    }
}

impl std::fmt::Display for ContextChar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextChar::Initial => write!(f, "Initial"),
            ContextChar::Final => write!(f, "Final"),
            ContextChar::BeforeVowel(cs) => {
                write!(f, "BeforeVowel({})", cs.iter().collect::<String>())
            }
            ContextChar::AfterConsonant(cs) => {
                write!(f, "AfterConsonant({})", cs.iter().collect::<String>())
            }
            ContextChar::BeforeConsonant(cs) => {
                write!(f, "BeforeConsonant({})", cs.iter().collect::<String>())
            }
            ContextChar::AfterVowel(cs) => {
                write!(f, "AfterVowel({})", cs.iter().collect::<String>())
            }
            ContextChar::Anywhere => write!(f, "Anywhere"),
            ContextChar::And(a, b) => write!(f, "And({}, {})", a, b),
            ContextChar::Or(a, b) => write!(f, "Or({}, {})", a, b),
            ContextChar::Not(inner) => write!(f, "Not({})", inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phone_display() {
        assert_eq!(Phone::Vowel(b'a').to_string(), "V(a)");
        assert_eq!(Phone::Consonant(b'k').to_string(), "C(k)");
        assert_eq!(Phone::Digraph(b'c', b'h').to_string(), "D(c,h)");
        assert_eq!(Phone::Silent.to_string(), "Silent");
    }

    #[test]
    fn test_phone_char_display() {
        assert_eq!(PhoneChar::Vowel('a').to_string(), "V(a)");
        assert_eq!(PhoneChar::Consonant('k').to_string(), "C(k)");
        assert_eq!(PhoneChar::Digraph('c', 'h').to_string(), "D(c,h)");
        assert_eq!(PhoneChar::Silent.to_string(), "Silent");
    }

    #[test]
    fn test_context_display() {
        assert_eq!(Context::Initial.to_string(), "Initial");
        assert_eq!(Context::Final.to_string(), "Final");
        assert_eq!(Context::Anywhere.to_string(), "Anywhere");
        assert_eq!(
            Context::BeforeVowel(vec![b'a', b'e', b'i']).to_string(),
            "BeforeVowel(aei)"
        );
    }

    #[test]
    fn test_context_char_display() {
        assert_eq!(ContextChar::Initial.to_string(), "Initial");
        assert_eq!(ContextChar::Final.to_string(), "Final");
        assert_eq!(ContextChar::Anywhere.to_string(), "Anywhere");
        assert_eq!(
            ContextChar::BeforeVowel(vec!['a', 'e', 'i']).to_string(),
            "BeforeVowel(aei)"
        );
    }

    #[test]
    fn test_phone_equality() {
        assert_eq!(Phone::Vowel(b'a'), Phone::Vowel(b'a'));
        assert_ne!(Phone::Vowel(b'a'), Phone::Vowel(b'e'));
        assert_ne!(Phone::Vowel(b'a'), Phone::Consonant(b'a'));
        assert_eq!(Phone::Silent, Phone::Silent);
    }

    #[test]
    fn test_phone_char_equality() {
        assert_eq!(PhoneChar::Vowel('a'), PhoneChar::Vowel('a'));
        assert_ne!(PhoneChar::Vowel('a'), PhoneChar::Vowel('e'));
        assert_ne!(PhoneChar::Vowel('a'), PhoneChar::Consonant('a'));
        assert_eq!(PhoneChar::Silent, PhoneChar::Silent);
    }

    #[test]
    fn test_context_equality() {
        assert_eq!(Context::Initial, Context::Initial);
        assert_ne!(Context::Initial, Context::Final);
        assert_eq!(
            Context::BeforeVowel(vec![b'a', b'e']),
            Context::BeforeVowel(vec![b'a', b'e'])
        );
        assert_ne!(
            Context::BeforeVowel(vec![b'a']),
            Context::BeforeVowel(vec![b'e'])
        );
    }

    #[test]
    fn test_rewrite_rule_creation() {
        let rule = RewriteRule {
            rule_id: 1,
            rule_name: "Test Rule".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 1.0,
        };
        assert_eq!(rule.rule_id, 1);
        assert_eq!(rule.pattern.len(), 2);
        assert_eq!(rule.replacement.len(), 1);
    }

    #[test]
    fn test_rewrite_rule_char_creation() {
        let rule = RewriteRuleChar {
            rule_id: 1,
            rule_name: "Test Rule".to_string(),
            pattern: vec![PhoneChar::Consonant('g'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 1.0,
        };
        assert_eq!(rule.rule_id, 1);
        assert_eq!(rule.pattern.len(), 2);
        assert_eq!(rule.replacement.len(), 1);
    }

    // ============================================================================
    // Tests for compound context types (And, Or, Not)
    // ============================================================================

    #[test]
    fn test_compound_context_and() {
        let ctx = Context::And(
            Box::new(Context::AfterVowel(vec![b'a', b'e', b'i', b'o', b'u'])),
            Box::new(Context::BeforeVowel(vec![b'a', b'e', b'i', b'o', b'u'])),
        );
        assert_eq!(
            ctx.to_string(),
            "And(AfterVowel(aeiou), BeforeVowel(aeiou))"
        );
        // Not position-dependent since neither child is
        assert!(!ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_or() {
        let ctx = Context::Or(
            Box::new(Context::Initial),
            Box::new(Context::Final),
        );
        assert_eq!(ctx.to_string(), "Or(Initial, Final)");
        // Position-dependent because Final is
        assert!(ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_not() {
        let ctx = Context::Not(Box::new(Context::BeforeVowel(vec![b'a', b'e', b'i', b'o', b'u'])));
        assert_eq!(ctx.to_string(), "Not(BeforeVowel(aeiou))");
        // Not position-dependent
        assert!(!ctx.is_position_dependent());
    }

    #[test]
    fn test_nested_compound_context() {
        // (!BeforeVowel) & (AfterVowel | Final)
        let ctx = Context::And(
            Box::new(Context::Not(Box::new(Context::BeforeVowel(vec![b'a', b'e'])))),
            Box::new(Context::Or(
                Box::new(Context::AfterVowel(vec![b'a', b'e'])),
                Box::new(Context::Final),
            )),
        );
        // Position-dependent because nested Final is
        assert!(ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_char_and() {
        let ctx = ContextChar::And(
            Box::new(ContextChar::AfterVowel(vec!['a', 'e', 'i', 'o', 'u'])),
            Box::new(ContextChar::BeforeVowel(vec!['a', 'e', 'i', 'o', 'u'])),
        );
        assert_eq!(
            ctx.to_string(),
            "And(AfterVowel(aeiou), BeforeVowel(aeiou))"
        );
        assert!(!ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_char_or() {
        let ctx = ContextChar::Or(
            Box::new(ContextChar::Initial),
            Box::new(ContextChar::Final),
        );
        assert_eq!(ctx.to_string(), "Or(Initial, Final)");
        assert!(ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_char_not() {
        let ctx = ContextChar::Not(Box::new(ContextChar::BeforeVowel(vec!['a', 'e', 'i', 'o', 'u'])));
        assert_eq!(ctx.to_string(), "Not(BeforeVowel(aeiou))");
        assert!(!ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_equality() {
        let ctx1 = Context::And(
            Box::new(Context::Initial),
            Box::new(Context::BeforeVowel(vec![b'a'])),
        );
        let ctx2 = Context::And(
            Box::new(Context::Initial),
            Box::new(Context::BeforeVowel(vec![b'a'])),
        );
        let ctx3 = Context::And(
            Box::new(Context::Initial),
            Box::new(Context::BeforeVowel(vec![b'e'])),
        );
        assert_eq!(ctx1, ctx2);
        assert_ne!(ctx1, ctx3);
    }
}
