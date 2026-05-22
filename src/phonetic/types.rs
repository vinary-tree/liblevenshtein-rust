//! Type definitions for phonetic rewrite rules.
//!
//! This module provides the core types for representing phonetic rewrite rules,
//! directly translated from the Coq/Rocq verification in
//! `docs/verification/phonetic/rewrite_rules.v`.
//!
//! # Generic Types
//!
//! All types are generic over the character unit type `U: PhoneticUnit`:
//! - `Phone<U>` - A phonetic unit (vowel, consonant, digraph, etc.)
//! - `Context<U>` - Context specification for rule matching
//! - `RewriteRule<U>` - A phonetic transformation rule
//!
//! # Type Aliases
//!
//! For convenience and backward compatibility, type aliases are provided:
//! - `PhoneByte`, `ContextByte`, `RewriteRuleByte` - Byte-level (u8) types
//! - `PhoneChar`, `ContextChar`, `RewriteRuleChar` - Character-level (char) types
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

use super::common::phonetic_unit::PhoneticUnit;
use super::common::syllable::SyllableExpr;

// ============================================================================
// Generic Phone Type
// ============================================================================

/// A phonetic unit representing a single sound.
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:30-34`
///
/// This type is generic over the character unit type `U: PhoneticUnit`,
/// enabling both byte-level (`u8`) and character-level (`char`) representations.
///
/// # Variants
///
/// - `Vowel(U)` - A vowel sound (e.g., 'a', 'e', 'i', 'o', 'u')
/// - `Consonant(U)` - A consonant sound (e.g., 'b', 'k', 'p')
/// - `Digraph(U, U)` - A two-character sound unit (e.g., 'ch', 'sh', 'th')
/// - `Trigraph(U, U, U)` - A three-character sound unit (e.g., ejective affricates)
/// - `Tetragraph(U, U, U, U)` - A four-character sound unit (e.g., prenasalized aspirated clicks)
/// - `Pentagraph(U, U, U, U, U)` - A five-character sound unit (e.g., prenasalized labialized clicks)
/// - `Hexagraph(U, U, U, U, U, U)` - A six-character sound unit (e.g., complex clusters)
/// - `Heptagraph(U, U, U, U, U, U, U)` - A seven-character sound unit (theoretical maximum)
/// - `Sequence(Vec<U>)` - An 8+ character sound unit (rare complex clusters)
/// - `Silent` - A silent letter (not pronounced)
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::types::Phone;
///
/// // Byte-level (ASCII)
/// let vowel_a: Phone<u8> = Phone::Vowel(b'a');
/// let digraph_ch: Phone<u8> = Phone::Digraph(b'c', b'h');
///
/// // Character-level (Unicode)
/// let vowel_e: Phone<char> = Phone::Vowel('e');
/// let vowel_umlaut: Phone<char> = Phone::Vowel('ü');
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialization",
    serde(bound = "U: Serialize + for<'a> Deserialize<'a>")
)]
pub enum Phone<U: PhoneticUnit> {
    /// A vowel sound
    Vowel(U),
    /// A consonant sound
    Consonant(U),
    /// A two-character sound unit (digraph)
    Digraph(U, U),
    /// A three-character sound unit (trigraph, e.g., ejective affricates)
    Trigraph(U, U, U),
    /// A four-character sound unit (tetragraph, e.g., prenasalized aspirated clicks)
    Tetragraph(U, U, U, U),
    /// A five-character sound unit (pentagraph, e.g., prenasalized labialized clicks)
    Pentagraph(U, U, U, U, U),
    /// A six-character sound unit (hexagraph, e.g., complex clusters)
    Hexagraph(U, U, U, U, U, U),
    /// A seven-character sound unit (heptagraph, theoretical maximum)
    Heptagraph(U, U, U, U, U, U, U),
    /// An 8+ character sound unit (rare complex clusters) - heap allocated
    Sequence(Vec<U>),
    /// A silent letter
    Silent,
}

impl<U: PhoneticUnit> Phone<U> {
    /// Check if this phone is a vowel.
    #[inline]
    pub fn is_vowel(&self) -> bool {
        matches!(self, Phone::Vowel(_))
    }

    /// Check if this phone is a consonant.
    ///
    /// Includes `Consonant`, `Digraph`, `Trigraph`, `Tetragraph`, `Pentagraph`,
    /// `Hexagraph`, `Heptagraph`, and `Sequence` variants.
    #[inline]
    pub fn is_consonant(&self) -> bool {
        matches!(
            self,
            Phone::Consonant(_)
                | Phone::Digraph(_, _)
                | Phone::Trigraph(_, _, _)
                | Phone::Tetragraph(_, _, _, _)
                | Phone::Pentagraph(_, _, _, _, _)
                | Phone::Hexagraph(_, _, _, _, _, _)
                | Phone::Heptagraph(_, _, _, _, _, _, _)
                | Phone::Sequence(_)
        )
    }

    /// Check if this phone is silent.
    #[inline]
    pub fn is_silent(&self) -> bool {
        matches!(self, Phone::Silent)
    }

    /// Get the first character of this phone, if any.
    pub fn first_char(&self) -> Option<U> {
        match self {
            Phone::Vowel(c)
            | Phone::Consonant(c)
            | Phone::Digraph(c, _)
            | Phone::Trigraph(c, _, _)
            | Phone::Tetragraph(c, _, _, _)
            | Phone::Pentagraph(c, _, _, _, _)
            | Phone::Hexagraph(c, _, _, _, _, _)
            | Phone::Heptagraph(c, _, _, _, _, _, _) => Some(*c),
            Phone::Sequence(s) => s.first().copied(),
            Phone::Silent => None,
        }
    }

    /// Get all characters of this phone as a vector.
    pub fn chars(&self) -> Vec<U> {
        match self {
            Phone::Vowel(c) | Phone::Consonant(c) => vec![*c],
            Phone::Digraph(c1, c2) => vec![*c1, *c2],
            Phone::Trigraph(c1, c2, c3) => vec![*c1, *c2, *c3],
            Phone::Tetragraph(c1, c2, c3, c4) => vec![*c1, *c2, *c3, *c4],
            Phone::Pentagraph(c1, c2, c3, c4, c5) => vec![*c1, *c2, *c3, *c4, *c5],
            Phone::Hexagraph(c1, c2, c3, c4, c5, c6) => vec![*c1, *c2, *c3, *c4, *c5, *c6],
            Phone::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                vec![*c1, *c2, *c3, *c4, *c5, *c6, *c7]
            }
            Phone::Sequence(s) => s.clone(),
            Phone::Silent => vec![],
        }
    }
}

impl<U: PhoneticUnit> std::fmt::Display for Phone<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Phone::Vowel(c) => write!(f, "V({})", U::to_char(*c)),
            Phone::Consonant(c) => write!(f, "C({})", U::to_char(*c)),
            Phone::Digraph(c1, c2) => write!(f, "D({},{})", U::to_char(*c1), U::to_char(*c2)),
            Phone::Trigraph(c1, c2, c3) => {
                write!(
                    f,
                    "T({},{},{})",
                    U::to_char(*c1),
                    U::to_char(*c2),
                    U::to_char(*c3)
                )
            }
            Phone::Tetragraph(c1, c2, c3, c4) => {
                write!(
                    f,
                    "Q({},{},{},{})",
                    U::to_char(*c1),
                    U::to_char(*c2),
                    U::to_char(*c3),
                    U::to_char(*c4)
                )
            }
            Phone::Pentagraph(c1, c2, c3, c4, c5) => {
                write!(
                    f,
                    "P5({},{},{},{},{})",
                    U::to_char(*c1),
                    U::to_char(*c2),
                    U::to_char(*c3),
                    U::to_char(*c4),
                    U::to_char(*c5)
                )
            }
            Phone::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                write!(
                    f,
                    "H6({},{},{},{},{},{})",
                    U::to_char(*c1),
                    U::to_char(*c2),
                    U::to_char(*c3),
                    U::to_char(*c4),
                    U::to_char(*c5),
                    U::to_char(*c6)
                )
            }
            Phone::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                write!(
                    f,
                    "H7({},{},{},{},{},{},{})",
                    U::to_char(*c1),
                    U::to_char(*c2),
                    U::to_char(*c3),
                    U::to_char(*c4),
                    U::to_char(*c5),
                    U::to_char(*c6),
                    U::to_char(*c7)
                )
            }
            Phone::Sequence(s) => {
                write!(f, "S(")?;
                for (i, c) in s.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", U::to_char(*c))?;
                }
                write!(f, ")")
            }
            Phone::Silent => write!(f, "Silent"),
        }
    }
}

// ============================================================================
// Generic Context Type
// ============================================================================

/// Context specification for when a rule applies.
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:48-55`
///
/// This type is generic over the character unit type `U: PhoneticUnit`,
/// enabling both byte-level (`u8`) and character-level (`char`) representations.
///
/// # Variants
///
/// - `Initial` - At the beginning of a word
/// - `Final` - At the end of a word
/// - `BeforeVowel(Vec<U>)` - Before specific vowels
/// - `AfterConsonant(Vec<U>)` - After specific consonants
/// - `BeforeConsonant(Vec<U>)` - Before specific consonants
/// - `AfterVowel(Vec<U>)` - After specific vowels
/// - `Anywhere` - No context restriction
/// - `And(Box<Context<U>>, Box<Context<U>>)` - Both contexts must match
/// - `Or(Box<Context<U>>, Box<Context<U>>)` - Either context must match
/// - `Not(Box<Context<U>>)` - Context must NOT match
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::types::Context;
///
/// // Byte-level
/// let ctx: Context<u8> = Context::BeforeVowel(vec![b'a', b'e', b'i']);
///
/// // Character-level
/// let ctx: Context<char> = Context::BeforeVowel(vec!['a', 'e', 'i', 'o', 'u']);
///
/// // Compound context
/// let ctx: Context<char> = Context::And(
///     Box::new(Context::Initial),
///     Box::new(Context::BeforeVowel(vec!['a', 'e'])),
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialization",
    serde(bound = "U: Serialize + for<'a> Deserialize<'a>")
)]
pub enum Context<U: PhoneticUnit> {
    /// At the beginning of a word
    Initial,
    /// At the end of a word
    Final,
    /// Before specific vowels
    BeforeVowel(Vec<U>),
    /// After specific consonants
    AfterConsonant(Vec<U>),
    /// Before specific consonants
    BeforeConsonant(Vec<U>),
    /// After specific vowels
    AfterVowel(Vec<U>),
    /// No context restriction
    Anywhere,
    /// Compound: both contexts must match
    And(Box<Context<U>>, Box<Context<U>>),
    /// Compound: either context must match
    Or(Box<Context<U>>, Box<Context<U>>),
    /// Negated: context must NOT match
    Not(Box<Context<U>>),
}

impl<U: PhoneticUnit> Context<U> {
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

impl<U: PhoneticUnit> std::fmt::Display for Context<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Context::Initial => write!(f, "Initial"),
            Context::Final => write!(f, "Final"),
            Context::BeforeVowel(cs) => {
                write!(f, "BeforeVowel({})", U::units_to_string(cs))
            }
            Context::AfterConsonant(cs) => {
                write!(f, "AfterConsonant({})", U::units_to_string(cs))
            }
            Context::BeforeConsonant(cs) => {
                write!(f, "BeforeConsonant({})", U::units_to_string(cs))
            }
            Context::AfterVowel(cs) => {
                write!(f, "AfterVowel({})", U::units_to_string(cs))
            }
            Context::Anywhere => write!(f, "Anywhere"),
            Context::And(a, b) => write!(f, "And({}, {})", a, b),
            Context::Or(a, b) => write!(f, "Or({}, {})", a, b),
            Context::Not(inner) => write!(f, "Not({})", inner),
        }
    }
}

// ============================================================================
// Generic RewriteRule Type
// ============================================================================

/// A phonetic rewrite rule.
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:62-68`
///
/// Represents a transformation from a pattern of phones to a replacement
/// sequence, applicable in a specific context.
///
/// This type is generic over the character unit type `U: PhoneticUnit`,
/// enabling both byte-level (`u8`) and character-level (`char`) representations.
///
/// # Fields
///
/// - `rule_id` - Unique identifier for the rule
/// - `rule_name` - Human-readable name (for debugging/documentation)
/// - `pattern` - Sequence of phones to match
/// - `replacement` - Sequence of phones to substitute
/// - `context` - Context in which the rule applies
/// - `weight` - Priority weight (higher = applied first)
/// - `syllable_condition` - Optional syllable-based condition
///
/// # Formal Properties
///
/// Well-formed rules satisfy (Theorem 1, `zompist_rules.v:285`):
/// - Pattern is non-empty: `pattern.len() > 0`
/// - Replacement is bounded: `replacement.len() <= pattern.len() + MAX_EXPANSION_FACTOR`
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::types::{Phone, Context, RewriteRule};
///
/// // Byte-level rule: "gh" -> "f" (anywhere)
/// let rule: RewriteRule<u8> = RewriteRule {
///     rule_id: 1,
///     rule_name: "gh → f".to_string(),
///     pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
///     replacement: vec![Phone::Consonant(b'f')],
///     context: Context::Anywhere,
///     weight: 1.0,
///     syllable_condition: None,
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialization",
    serde(bound = "U: Serialize + for<'a> Deserialize<'a>")
)]
pub struct RewriteRule<U: PhoneticUnit> {
    /// Unique identifier for the rule
    pub rule_id: usize,
    /// Human-readable name
    pub rule_name: String,
    /// Pattern to match (sequence of phones)
    pub pattern: Vec<Phone<U>>,
    /// Replacement sequence
    pub replacement: Vec<Phone<U>>,
    /// Context specification
    pub context: Context<U>,
    /// Priority weight (higher = applied first)
    pub weight: f64,
    /// Optional syllable condition
    /// When present, the rule only applies if the word satisfies this condition
    pub syllable_condition: Option<SyllableExpr>,
}

// ============================================================================
// Type Aliases for Backward Compatibility
// ============================================================================

// Note: To maintain full backward compatibility with existing code that uses
// `Phone`, `Context`, and `RewriteRule` directly (without type parameters),
// we need to handle import strategies carefully. Code that explicitly imports
// `Phone<u8>` or uses turbofish syntax will continue to work.
//
// For code using the old non-generic names:
// - Use `PhoneByte`/`PhoneChar` for explicit byte/char variants
// - Or import with: `use types::{Phone, PhoneChar}` and use `Phone::<u8>::...`

/// Byte-level phone (ASCII) - alias for `Phone<u8>`
///
/// Use this when working with ASCII-only text for better performance.
pub type PhoneByte = Phone<u8>;

/// Character-level phone (Unicode) - alias for `Phone<char>`
///
/// Use this when working with Unicode text (accented characters, CJK, etc.).
pub type PhoneChar = Phone<char>;

/// Byte-level context (ASCII) - alias for `Context<u8>`
pub type ContextByte = Context<u8>;

/// Character-level context (Unicode) - alias for `Context<char>`
pub type ContextChar = Context<char>;

/// Byte-level rewrite rule (ASCII) - alias for `RewriteRule<u8>`
pub type RewriteRuleByte = RewriteRule<u8>;

/// Character-level rewrite rule (Unicode) - alias for `RewriteRule<char>`
pub type RewriteRuleChar = RewriteRule<char>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Phone tests
    // ========================================================================

    #[test]
    fn test_phone_display_byte() {
        assert_eq!(Phone::<u8>::Vowel(b'a').to_string(), "V(a)");
        assert_eq!(Phone::<u8>::Consonant(b'k').to_string(), "C(k)");
        assert_eq!(Phone::<u8>::Digraph(b'c', b'h').to_string(), "D(c,h)");
        assert_eq!(Phone::<u8>::Silent.to_string(), "Silent");
    }

    #[test]
    fn test_phone_display_char() {
        assert_eq!(Phone::<char>::Vowel('a').to_string(), "V(a)");
        assert_eq!(Phone::<char>::Consonant('k').to_string(), "C(k)");
        assert_eq!(Phone::<char>::Digraph('c', 'h').to_string(), "D(c,h)");
        assert_eq!(Phone::<char>::Silent.to_string(), "Silent");
    }

    #[test]
    fn test_phone_equality_byte() {
        assert_eq!(Phone::<u8>::Vowel(b'a'), Phone::<u8>::Vowel(b'a'));
        assert_ne!(Phone::<u8>::Vowel(b'a'), Phone::<u8>::Vowel(b'e'));
        assert_ne!(Phone::<u8>::Vowel(b'a'), Phone::<u8>::Consonant(b'a'));
        assert_eq!(Phone::<u8>::Silent, Phone::<u8>::Silent);
    }

    #[test]
    fn test_phone_equality_char() {
        assert_eq!(Phone::<char>::Vowel('a'), Phone::<char>::Vowel('a'));
        assert_ne!(Phone::<char>::Vowel('a'), Phone::<char>::Vowel('e'));
        assert_ne!(Phone::<char>::Vowel('a'), Phone::<char>::Consonant('a'));
        assert_eq!(Phone::<char>::Silent, Phone::<char>::Silent);
    }

    #[test]
    fn test_phone_is_vowel() {
        assert!(Phone::<u8>::Vowel(b'a').is_vowel());
        assert!(!Phone::<u8>::Consonant(b'k').is_vowel());
        assert!(!Phone::<u8>::Silent.is_vowel());
    }

    #[test]
    fn test_phone_is_consonant() {
        assert!(Phone::<u8>::Consonant(b'k').is_consonant());
        assert!(Phone::<u8>::Digraph(b'c', b'h').is_consonant());
        assert!(!Phone::<u8>::Vowel(b'a').is_consonant());
        assert!(!Phone::<u8>::Silent.is_consonant());
    }

    #[test]
    fn test_phone_first_char() {
        assert_eq!(Phone::<u8>::Vowel(b'a').first_char(), Some(b'a'));
        assert_eq!(Phone::<u8>::Digraph(b'c', b'h').first_char(), Some(b'c'));
        assert_eq!(Phone::<u8>::Silent.first_char(), None);
    }

    #[test]
    fn test_phone_chars() {
        assert_eq!(Phone::<u8>::Vowel(b'a').chars(), vec![b'a']);
        assert_eq!(Phone::<u8>::Digraph(b'c', b'h').chars(), vec![b'c', b'h']);
        assert_eq!(Phone::<u8>::Silent.chars(), Vec::<u8>::new());
    }

    // ========================================================================
    // Context tests
    // ========================================================================

    #[test]
    fn test_context_display_byte() {
        assert_eq!(Context::<u8>::Initial.to_string(), "Initial");
        assert_eq!(Context::<u8>::Final.to_string(), "Final");
        assert_eq!(Context::<u8>::Anywhere.to_string(), "Anywhere");
        assert_eq!(
            Context::<u8>::BeforeVowel(vec![b'a', b'e', b'i']).to_string(),
            "BeforeVowel(aei)"
        );
    }

    #[test]
    fn test_context_display_char() {
        assert_eq!(Context::<char>::Initial.to_string(), "Initial");
        assert_eq!(Context::<char>::Final.to_string(), "Final");
        assert_eq!(Context::<char>::Anywhere.to_string(), "Anywhere");
        assert_eq!(
            Context::<char>::BeforeVowel(vec!['a', 'e', 'i']).to_string(),
            "BeforeVowel(aei)"
        );
    }

    #[test]
    fn test_context_equality_byte() {
        assert_eq!(Context::<u8>::Initial, Context::<u8>::Initial);
        assert_ne!(Context::<u8>::Initial, Context::<u8>::Final);
        assert_eq!(
            Context::<u8>::BeforeVowel(vec![b'a', b'e']),
            Context::<u8>::BeforeVowel(vec![b'a', b'e'])
        );
        assert_ne!(
            Context::<u8>::BeforeVowel(vec![b'a']),
            Context::<u8>::BeforeVowel(vec![b'e'])
        );
    }

    #[test]
    fn test_context_is_position_dependent() {
        // Final is position-dependent
        assert!(Context::<u8>::Final.is_position_dependent());

        // Others are not
        assert!(!Context::<u8>::Initial.is_position_dependent());
        assert!(!Context::<u8>::Anywhere.is_position_dependent());
        assert!(!Context::<u8>::BeforeVowel(vec![b'a']).is_position_dependent());

        // And propagates
        let ctx = Context::<u8>::And(Box::new(Context::Initial), Box::new(Context::Final));
        assert!(ctx.is_position_dependent());

        // Or propagates
        let ctx = Context::<u8>::Or(Box::new(Context::Initial), Box::new(Context::Final));
        assert!(ctx.is_position_dependent());

        // Not propagates
        let ctx = Context::<u8>::Not(Box::new(Context::Final));
        assert!(ctx.is_position_dependent());
    }

    // ========================================================================
    // RewriteRule tests
    // ========================================================================

    #[test]
    fn test_rewrite_rule_creation_byte() {
        let rule: RewriteRule<u8> = RewriteRule {
            rule_id: 1,
            rule_name: "Test Rule".to_string(),
            pattern: vec![Phone::Consonant(b'g'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 1.0,
            syllable_condition: None,
        };
        assert_eq!(rule.rule_id, 1);
        assert_eq!(rule.pattern.len(), 2);
        assert_eq!(rule.replacement.len(), 1);
    }

    #[test]
    fn test_rewrite_rule_creation_char() {
        let rule: RewriteRule<char> = RewriteRule {
            rule_id: 1,
            rule_name: "Test Rule".to_string(),
            pattern: vec![Phone::Consonant('g'), Phone::Consonant('h')],
            replacement: vec![Phone::Consonant('f')],
            context: Context::Anywhere,
            weight: 1.0,
            syllable_condition: None,
        };
        assert_eq!(rule.rule_id, 1);
        assert_eq!(rule.pattern.len(), 2);
        assert_eq!(rule.replacement.len(), 1);
    }

    // ========================================================================
    // Type alias tests
    // ========================================================================

    #[test]
    fn test_type_aliases() {
        // Verify type aliases work correctly
        let _phone_byte: PhoneByte = Phone::Vowel(b'a');
        let _phone_char: PhoneChar = Phone::Vowel('a');
        let _context_byte: ContextByte = Context::Initial;
        let _context_char: ContextChar = Context::Initial;
    }

    // ========================================================================
    // Compound context tests
    // ========================================================================

    #[test]
    fn test_compound_context_and_byte() {
        let ctx: Context<u8> = Context::And(
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
    fn test_compound_context_or_byte() {
        let ctx: Context<u8> = Context::Or(Box::new(Context::Initial), Box::new(Context::Final));
        assert_eq!(ctx.to_string(), "Or(Initial, Final)");
        // Position-dependent because Final is
        assert!(ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_not_byte() {
        let ctx: Context<u8> = Context::Not(Box::new(Context::BeforeVowel(vec![
            b'a', b'e', b'i', b'o', b'u',
        ])));
        assert_eq!(ctx.to_string(), "Not(BeforeVowel(aeiou))");
        // Not position-dependent
        assert!(!ctx.is_position_dependent());
    }

    #[test]
    fn test_nested_compound_context() {
        // (!BeforeVowel) & (AfterVowel | Final)
        let ctx: Context<u8> = Context::And(
            Box::new(Context::Not(Box::new(Context::BeforeVowel(vec![
                b'a', b'e',
            ])))),
            Box::new(Context::Or(
                Box::new(Context::AfterVowel(vec![b'a', b'e'])),
                Box::new(Context::Final),
            )),
        );
        // Position-dependent because nested Final is
        assert!(ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_and_char() {
        let ctx: Context<char> = Context::And(
            Box::new(Context::AfterVowel(vec!['a', 'e', 'i', 'o', 'u'])),
            Box::new(Context::BeforeVowel(vec!['a', 'e', 'i', 'o', 'u'])),
        );
        assert_eq!(
            ctx.to_string(),
            "And(AfterVowel(aeiou), BeforeVowel(aeiou))"
        );
        assert!(!ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_or_char() {
        let ctx: Context<char> = Context::Or(Box::new(Context::Initial), Box::new(Context::Final));
        assert_eq!(ctx.to_string(), "Or(Initial, Final)");
        assert!(ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_not_char() {
        let ctx: Context<char> = Context::Not(Box::new(Context::BeforeVowel(vec![
            'a', 'e', 'i', 'o', 'u',
        ])));
        assert_eq!(ctx.to_string(), "Not(BeforeVowel(aeiou))");
        assert!(!ctx.is_position_dependent());
    }

    #[test]
    fn test_compound_context_equality() {
        let ctx1: Context<u8> = Context::And(
            Box::new(Context::Initial),
            Box::new(Context::BeforeVowel(vec![b'a'])),
        );
        let ctx2: Context<u8> = Context::And(
            Box::new(Context::Initial),
            Box::new(Context::BeforeVowel(vec![b'a'])),
        );
        let ctx3: Context<u8> = Context::And(
            Box::new(Context::Initial),
            Box::new(Context::BeforeVowel(vec![b'e'])),
        );
        assert_eq!(ctx1, ctx2);
        assert_ne!(ctx1, ctx3);
    }
}
