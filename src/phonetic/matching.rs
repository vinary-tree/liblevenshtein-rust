//! Pattern and context matching for phonetic rewrite rules.
//!
//! This module implements the matching logic for phonetic rewrite rules,
//! directly translated from the Coq/Rocq verification in
//! `docs/verification/phonetic/rewrite_rules.v`.
//!
//! # Generic Functions
//!
//! All functions are generic over `U: PhoneticUnit`, enabling use with both
//! byte-level (`u8`) and character-level (`char`) types:
//!
//! - [`context_matches`] - Context satisfaction check
//! - [`pattern_matches_at`] - Pattern matching at position
//!
//! # Legacy Functions (Deprecated)
//!
//! For backward compatibility, type-specific aliases are provided:
//! - `context_matches_char`, `pattern_matches_at_char` - Character-level aliases
//! - `phone_eq`, `phone_eq_char` - Phone equality (now use `==` directly)
//!
//! # Formal Specification
//!
//! All functions are direct translations of Coq functions with proven properties.

use super::common::phonetic_unit::PhoneticUnit;
use super::types::{Context, ContextChar, Phone, PhoneByte, PhoneChar};

// ============================================================================
// Generic Functions
// ============================================================================

/// Check if a context is satisfied at a position in a phonetic string.
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:117-157`
///
/// This is the generic implementation that works with any `PhoneticUnit` type.
///
/// # Arguments
///
/// - `ctx` - The context to check
/// - `s` - The phonetic string
/// - `match_start` - The position where the pattern match starts
/// - `pattern_len` - The length of the matched pattern
///
/// # Returns
///
/// `true` if the context is satisfied, `false` otherwise.
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::{Context, Phone, context_matches};
///
/// let s: Vec<Phone<u8>> = vec![Phone::Consonant(b'k'), Phone::Vowel(b'a'), Phone::Consonant(b't')];
/// assert!(context_matches(&Context::Initial, &s, 0, 1));     // pattern at start
/// assert!(context_matches(&Context::Final, &s, 2, 1));       // pattern at end
/// assert!(context_matches(&Context::Anywhere, &s, 1, 1));
/// ```
pub fn context_matches<U: PhoneticUnit>(
    ctx: &Context<U>,
    s: &[Phone<U>],
    match_start: usize,
    pattern_len: usize,
) -> bool {
    // Compute the appropriate position based on context type:
    // - "Before" and "Final" contexts check at the END of the pattern (match_start + pattern_len)
    // - "After" and "Initial" contexts check at the START of the pattern (match_start)
    match ctx {
        Context::Initial => match_start == 0,
        Context::Final => match_start + pattern_len == s.len(),
        Context::BeforeVowel(vowels) => {
            let pos = match_start + pattern_len;
            if let Some(Phone::Vowel(v)) = s.get(pos) {
                vowels.contains(v)
            } else {
                false
            }
        }
        Context::AfterConsonant(consonants) => {
            if match_start == 0 {
                false
            } else if let Some(phone) = s.get(match_start - 1) {
                match phone {
                    Phone::Consonant(c) => consonants.contains(c),
                    Phone::Digraph(c1, _) => consonants.contains(c1),
                    Phone::Trigraph(c1, _, _) => consonants.contains(c1),
                    _ => false,
                }
            } else {
                false
            }
        }
        Context::BeforeConsonant(consonants) => {
            let pos = match_start + pattern_len;
            if let Some(phone) = s.get(pos) {
                match phone {
                    Phone::Consonant(c) => consonants.contains(c),
                    Phone::Digraph(c1, _) => consonants.contains(c1),
                    Phone::Trigraph(c1, _, _) => consonants.contains(c1),
                    _ => false,
                }
            } else {
                false
            }
        }
        Context::AfterVowel(vowels) => {
            if match_start == 0 {
                false
            } else if let Some(Phone::Vowel(v)) = s.get(match_start - 1) {
                vowels.contains(v)
            } else {
                false
            }
        }
        Context::Anywhere => true,
        // Compound context: both must match (each with correct position based on type)
        Context::And(a, b) => {
            context_matches(a, s, match_start, pattern_len)
                && context_matches(b, s, match_start, pattern_len)
        }
        // Compound context: either must match
        Context::Or(a, b) => {
            context_matches(a, s, match_start, pattern_len)
                || context_matches(b, s, match_start, pattern_len)
        }
        // Negated context: must NOT match
        Context::Not(inner) => !context_matches(inner, s, match_start, pattern_len),
    }
}

/// Check if a pattern matches at a specific position in a phonetic string.
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:162-174`
///
/// This is the generic implementation that works with any `PhoneticUnit` type.
///
/// # Arguments
///
/// - `pattern` - The pattern to match
/// - `s` - The phonetic string
/// - `pos` - The starting position
///
/// # Returns
///
/// `true` if the pattern matches at the given position, `false` otherwise.
///
/// # Examples
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::{Phone, pattern_matches_at};
///
/// let pattern: Vec<Phone<u8>> = vec![Phone::Consonant(b'c'), Phone::Consonant(b'h')];
/// let s: Vec<Phone<u8>> = vec![Phone::Consonant(b'c'), Phone::Consonant(b'h'), Phone::Vowel(b'a')];
/// assert!(pattern_matches_at(&pattern, &s, 0));
/// assert!(!pattern_matches_at(&pattern, &s, 1));
/// ```
pub fn pattern_matches_at<U: PhoneticUnit>(
    pattern: &[Phone<U>],
    s: &[Phone<U>],
    pos: usize,
) -> bool {
    if pattern.is_empty() {
        return true;
    }

    // Check bounds first
    if pos + pattern.len() > s.len() {
        return false;
    }

    // Use PartialEq derived on Phone<U>
    for (i, p) in pattern.iter().enumerate() {
        if s.get(pos + i) != Some(p) {
            return false;
        }
    }
    true
}

// ============================================================================
// Backward Compatibility: Type-Specific Aliases
// ============================================================================

/// Check if a context is satisfied (character-level alias).
///
/// This is an alias for `context_matches::<char>` for backward compatibility.
#[inline]
pub fn context_matches_char(
    ctx: &ContextChar,
    s: &[PhoneChar],
    match_start: usize,
    pattern_len: usize,
) -> bool {
    context_matches(ctx, s, match_start, pattern_len)
}

/// Check if a pattern matches at a position (character-level alias).
///
/// This is an alias for `pattern_matches_at::<char>` for backward compatibility.
#[inline]
pub fn pattern_matches_at_char(pattern: &[PhoneChar], s: &[PhoneChar], pos: usize) -> bool {
    pattern_matches_at(pattern, s, pos)
}

// ============================================================================
// Legacy Functions (for backward compatibility)
// ============================================================================

/// Check if two phones are equal (byte-level).
///
/// **Deprecated**: Use `p1 == p2` directly instead. `Phone<U>` derives `PartialEq`.
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:97-105`
#[inline]
pub fn phone_eq(p1: &PhoneByte, p2: &PhoneByte) -> bool {
    p1 == p2
}

/// Check if two phones are equal (character-level).
///
/// **Deprecated**: Use `p1 == p2` directly instead. `Phone<U>` derives `PartialEq`.
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:97-105`
#[inline]
pub fn phone_eq_char(p1: &PhoneChar, p2: &PhoneChar) -> bool {
    p1 == p2
}

// ============================================================================
// Helper functions (for backward compatibility with existing code)
// ============================================================================

/// Check if a character is a vowel (byte-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:72-76`
///
/// Checks if a byte represents one of the five English vowels: a, e, i, o, u.
#[inline]
pub fn is_vowel_char(c: u8) -> bool {
    u8::is_vowel(c)
}

/// Check if a phone is a vowel (byte-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:82-87`
#[inline]
pub fn is_vowel(p: &PhoneByte) -> bool {
    p.is_vowel()
}

/// Check if a phone is a consonant (byte-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:89-95`
#[inline]
pub fn is_consonant(p: &PhoneByte) -> bool {
    p.is_consonant()
}

/// Check if a character is a vowel (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:72-76`
#[inline]
pub fn is_vowel_char_char(c: char) -> bool {
    char::is_vowel(c)
}

/// Check if a phone is a vowel (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:82-87`
#[inline]
pub fn is_vowel_char_type(p: &PhoneChar) -> bool {
    p.is_vowel()
}

/// Check if a phone is a consonant (character-level).
///
/// **Formal Specification**: `docs/verification/phonetic/rewrite_rules.v:89-95`
#[inline]
pub fn is_consonant_char(p: &PhoneChar) -> bool {
    p.is_consonant()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Byte-level tests
    // ========================================================================

    #[test]
    fn test_is_vowel_char() {
        assert!(is_vowel_char(b'a'));
        assert!(is_vowel_char(b'e'));
        assert!(is_vowel_char(b'i'));
        assert!(is_vowel_char(b'o'));
        assert!(is_vowel_char(b'u'));
        assert!(!is_vowel_char(b'k'));
        assert!(!is_vowel_char(b'b'));
    }

    #[test]
    fn test_is_vowel() {
        assert!(is_vowel(&Phone::Vowel(b'a')));
        assert!(!is_vowel(&Phone::Consonant(b'k')));
        assert!(!is_vowel(&Phone::Silent));
    }

    #[test]
    fn test_is_consonant() {
        assert!(is_consonant(&Phone::Consonant(b'k')));
        assert!(is_consonant(&Phone::Digraph(b'c', b'h')));
        assert!(!is_consonant(&Phone::Vowel(b'a')));
        assert!(!is_consonant(&Phone::Silent));
    }

    #[test]
    fn test_phone_eq() {
        assert!(phone_eq(&Phone::Vowel(b'a'), &Phone::Vowel(b'a')));
        assert!(!phone_eq(&Phone::Vowel(b'a'), &Phone::Vowel(b'e')));
        assert!(phone_eq(&Phone::Silent, &Phone::Silent));
        assert!(phone_eq(
            &Phone::Digraph(b'c', b'h'),
            &Phone::Digraph(b'c', b'h')
        ));
        assert!(!phone_eq(
            &Phone::Digraph(b'c', b'h'),
            &Phone::Digraph(b's', b'h')
        ));
    }

    #[test]
    fn test_context_matches_initial() {
        let s: Vec<Phone<u8>> = vec![Phone::Consonant(b'k'), Phone::Vowel(b'a')];
        // Pattern of length 1 at position 0 - Initial should match
        assert!(context_matches(&Context::Initial, &s, 0, 1));
        // Pattern at position 1 - Initial should not match
        assert!(!context_matches(&Context::Initial, &s, 1, 1));
    }

    #[test]
    fn test_context_matches_final() {
        let s: Vec<Phone<u8>> = vec![Phone::Consonant(b'k'), Phone::Vowel(b'a')];
        // Pattern of length 1 at position 1 (ends at 2 = s.len()) - Final matches
        assert!(context_matches(&Context::Final, &s, 1, 1));
        // Pattern of length 1 at position 0 (ends at 1 != 2) - Final doesn't match
        assert!(!context_matches(&Context::Final, &s, 0, 1));
    }

    #[test]
    fn test_context_matches_anywhere() {
        let s: Vec<Phone<u8>> = vec![Phone::Consonant(b'k'), Phone::Vowel(b'a')];
        // Anywhere always matches regardless of position
        assert!(context_matches(&Context::Anywhere, &s, 0, 1));
        assert!(context_matches(&Context::Anywhere, &s, 1, 1));
    }

    #[test]
    fn test_context_matches_before_vowel() {
        let s: Vec<Phone<u8>> = vec![Phone::Consonant(b'k'), Phone::Vowel(b'a')];
        let ctx: Context<u8> = Context::BeforeVowel(vec![b'a', b'e', b'i']);
        // Pattern at position 0 with length 1 - checks position 1 which is vowel 'a'
        assert!(context_matches(&ctx, &s, 0, 1));
        // Pattern at position 1 with length 1 - checks position 2 which is out of bounds
        assert!(!context_matches(&ctx, &s, 1, 1));
    }

    #[test]
    fn test_pattern_matches_at() {
        let pattern: Vec<Phone<u8>> = vec![Phone::Consonant(b'c'), Phone::Consonant(b'h')];
        let s: Vec<Phone<u8>> = vec![
            Phone::Consonant(b'c'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'a'),
        ];
        assert!(pattern_matches_at(&pattern, &s, 0));
        assert!(!pattern_matches_at(&pattern, &s, 1));
        assert!(!pattern_matches_at(&pattern, &s, 2));
    }

    #[test]
    fn test_pattern_matches_at_empty() {
        let pattern: Vec<Phone<u8>> = vec![];
        let s: Vec<Phone<u8>> = vec![Phone::Vowel(b'a')];
        assert!(pattern_matches_at(&pattern, &s, 0));
    }

    // ========================================================================
    // Character-level tests
    // ========================================================================

    #[test]
    fn test_is_vowel_char_char() {
        assert!(is_vowel_char_char('a'));
        assert!(is_vowel_char_char('e'));
        assert!(!is_vowel_char_char('k'));
    }

    #[test]
    fn test_phone_eq_char() {
        assert!(phone_eq_char(
            &PhoneChar::Vowel('a'),
            &PhoneChar::Vowel('a')
        ));
        assert!(!phone_eq_char(
            &PhoneChar::Vowel('a'),
            &PhoneChar::Vowel('e')
        ));
        assert!(phone_eq_char(&PhoneChar::Silent, &PhoneChar::Silent));
    }

    #[test]
    fn test_context_matches_char_initial() {
        let s: Vec<PhoneChar> = vec![PhoneChar::Consonant('k'), PhoneChar::Vowel('a')];
        // Pattern at position 0 - Initial should match
        assert!(context_matches_char(&ContextChar::Initial, &s, 0, 1));
        // Pattern at position 1 - Initial should not match
        assert!(!context_matches_char(&ContextChar::Initial, &s, 1, 1));
    }

    #[test]
    fn test_pattern_matches_at_char() {
        let pattern: Vec<PhoneChar> = vec![PhoneChar::Consonant('c'), PhoneChar::Consonant('h')];
        let s: Vec<PhoneChar> = vec![
            PhoneChar::Consonant('c'),
            PhoneChar::Consonant('h'),
            PhoneChar::Vowel('a'),
        ];
        assert!(pattern_matches_at_char(&pattern, &s, 0));
        assert!(!pattern_matches_at_char(&pattern, &s, 1));
    }

    // ========================================================================
    // Compound context tests (byte-level)
    // ========================================================================

    #[test]
    fn test_context_matches_and() {
        // Test: x -> gz when after vowel AND before vowel (exact -> egzact)
        // String: e x a c t (positions 0-4)
        let s: Vec<Phone<u8>> = vec![
            Phone::Vowel(b'e'),     // pos 0
            Phone::Consonant(b'x'), // pos 1
            Phone::Vowel(b'a'),     // pos 2
            Phone::Consonant(b'c'), // pos 3
            Phone::Consonant(b't'), // pos 4
        ];

        // The x->gz rule with pattern 'x' (length 1) at position 1:
        // - AfterVowel checks s[match_start-1] = s[0] = 'e' is vowel? YES
        // - BeforeVowel checks s[match_start+pattern_len] = s[2] = 'a' is vowel? YES
        let ctx_exact: Context<u8> = Context::And(
            Box::new(Context::AfterVowel(vec![b'a', b'e', b'i', b'o', b'u'])),
            Box::new(Context::BeforeVowel(vec![b'a', b'e', b'i', b'o', b'u'])),
        );
        // Pattern 'x' at position 1 with length 1: AfterVowel and BeforeVowel both pass
        assert!(context_matches(&ctx_exact, &s, 1, 1));

        // Clear test for AND with vowel sequence:
        let s2: Vec<Phone<u8>> = vec![
            Phone::Vowel(b'a'), // pos 0
            Phone::Vowel(b'e'), // pos 1
        ];

        // Pattern at position 1 with length 1:
        // - AfterVowel checks s[0] = 'a'
        // - BeforeVowel checks s[2] which is out of bounds (fails)
        let ctx2: Context<u8> = Context::And(
            Box::new(Context::AfterVowel(vec![b'a'])),
            Box::new(Context::BeforeVowel(vec![b'e'])),
        );
        assert!(!context_matches(&ctx2, &s2, 1, 1)); // BeforeVowel fails (out of bounds)

        // Pattern at position 0 with length 1:
        // - AfterVowel checks s[-1] which fails (at start)
        // - BeforeVowel checks s[1] = 'e'
        assert!(!context_matches(&ctx2, &s2, 0, 1)); // AfterVowel fails (at start)

        // Test where AND succeeds: vowel-consonant-vowel pattern
        let s3: Vec<Phone<u8>> = vec![
            Phone::Vowel(b'a'),     // pos 0
            Phone::Consonant(b'x'), // pos 1
            Phone::Vowel(b'e'),     // pos 2
        ];
        // Pattern at position 1 with length 1:
        // - AfterVowel checks s[0] = 'a'
        // - BeforeVowel checks s[2] = 'e'
        let ctx3: Context<u8> = Context::And(
            Box::new(Context::AfterVowel(vec![b'a'])),
            Box::new(Context::BeforeVowel(vec![b'e'])),
        );
        assert!(context_matches(&ctx3, &s3, 1, 1));
    }

    #[test]
    fn test_context_matches_or() {
        let s: Vec<Phone<u8>> = vec![Phone::Consonant(b'k'), Phone::Vowel(b'a')];

        // Either initial OR final
        let ctx: Context<u8> = Context::Or(Box::new(Context::Initial), Box::new(Context::Final));

        // Pattern at position 0 with length 1: Initial matches (0 == 0)
        assert!(context_matches(&ctx, &s, 0, 1));
        // Pattern at position 1 with length 1: Final matches (1+1 == 2 == s.len())
        assert!(context_matches(&ctx, &s, 1, 1));
        // Pattern at position 0 with length 2: Final matches (0+2 == 2 == s.len())
        assert!(context_matches(&ctx, &s, 0, 2));
    }

    #[test]
    fn test_context_matches_not() {
        let s: Vec<Phone<u8>> = vec![Phone::Consonant(b'k'), Phone::Vowel(b'a')];

        // NOT initial
        let ctx: Context<u8> = Context::Not(Box::new(Context::Initial));

        // Pattern at position 0 - Initial matches, so NOT fails
        assert!(!context_matches(&ctx, &s, 0, 1));
        // Pattern at position 1 - Initial doesn't match, so NOT succeeds
        assert!(context_matches(&ctx, &s, 1, 1));
    }

    #[test]
    fn test_context_matches_nested_compound() {
        let s: Vec<Phone<u8>> = vec![
            Phone::Vowel(b'a'),     // pos 0
            Phone::Consonant(b'k'), // pos 1
            Phone::Vowel(b'e'),     // pos 2
        ];

        // (NOT Initial) AND (AfterVowel OR BeforeVowel)
        let ctx: Context<u8> = Context::And(
            Box::new(Context::Not(Box::new(Context::Initial))),
            Box::new(Context::Or(
                Box::new(Context::AfterVowel(vec![b'a', b'e', b'i', b'o', b'u'])),
                Box::new(Context::BeforeVowel(vec![b'a', b'e', b'i', b'o', b'u'])),
            )),
        );

        // Pattern at pos 0: Initial matches, so NOT Initial = false -> AND fails
        assert!(!context_matches(&ctx, &s, 0, 1));

        // Pattern at pos 1 with length 1:
        // - NOT Initial = true (1 != 0)
        // - AfterVowel checks s[0]='a' = true
        // - OR succeeds, AND succeeds
        assert!(context_matches(&ctx, &s, 1, 1));

        // Pattern at pos 2 with length 1:
        // - NOT Initial = true (2 != 0)
        // - AfterVowel checks s[1]='k' = false (consonant)
        // - BeforeVowel checks s[3] = out of bounds = false
        // - OR fails, AND fails
        assert!(!context_matches(&ctx, &s, 2, 1));
    }

    // ========================================================================
    // Compound context tests (character-level)
    // ========================================================================

    #[test]
    fn test_context_matches_char_and() {
        // vowel-consonant-vowel sequence
        let s: Vec<PhoneChar> = vec![
            PhoneChar::Vowel('a'),     // pos 0
            PhoneChar::Consonant('x'), // pos 1
            PhoneChar::Vowel('e'),     // pos 2
        ];

        let ctx: ContextChar = ContextChar::And(
            Box::new(ContextChar::AfterVowel(vec!['a'])),
            Box::new(ContextChar::BeforeVowel(vec!['e'])),
        );

        // Pattern at position 1 with length 1:
        // - AfterVowel checks s[0] = 'a'
        // - BeforeVowel checks s[2] = 'e'
        assert!(context_matches_char(&ctx, &s, 1, 1));
    }

    #[test]
    fn test_context_matches_char_or() {
        let s: Vec<PhoneChar> = vec![PhoneChar::Consonant('k'), PhoneChar::Vowel('a')];

        let ctx: ContextChar =
            ContextChar::Or(Box::new(ContextChar::Initial), Box::new(ContextChar::Final));

        // Pattern at position 0 with length 1: Initial matches
        assert!(context_matches_char(&ctx, &s, 0, 1));
        // Pattern at position 1 with length 1: Final matches (1+1 == 2 == s.len())
        assert!(context_matches_char(&ctx, &s, 1, 1));
        // Pattern at position 0 with length 2: Both match (Initial AND Final)
        assert!(context_matches_char(&ctx, &s, 0, 2));
    }

    #[test]
    fn test_context_matches_char_not() {
        let s: Vec<PhoneChar> = vec![PhoneChar::Consonant('k'), PhoneChar::Vowel('a')];

        let ctx: ContextChar = ContextChar::Not(Box::new(ContextChar::Initial));

        // Pattern at position 0 - Initial matches, so NOT fails
        assert!(!context_matches_char(&ctx, &s, 0, 1));
        // Pattern at position 1 - Initial doesn't match, so NOT succeeds
        assert!(context_matches_char(&ctx, &s, 1, 1));
    }
}
