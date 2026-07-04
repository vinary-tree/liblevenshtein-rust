//! Phonetic unit trait for abstracting over u8 and char types.
//!
//! This module provides a trait (`PhoneticUnit`) that enables generic
//! phonetic types like `Phone<U>`, `Context<U>`, and `RewriteRule<U>`.
//! By implementing this trait for both `u8` and `char`, we unify the
//! byte-level and character-level phonetic code paths, eliminating
//! significant code duplication.
//!
//! # Design
//!
//! The `PhoneticUnit` trait captures the essential operations needed for
//! phonetic processing:
//!
//! - **From/to string conversion** - For parsing and serialization
//! - **Case folding** - For case-insensitive matching
//! - **Vowel/consonant classification** - For context-dependent rules
//! - **Display formatting** - For error messages and diagnostics
//!
//! # ASCII vs Unicode
//!
//! - `u8` implementation: Tuned for ASCII text, single-byte operations
//! - `char` implementation: Full Unicode support, multi-byte characters
//!
//! Choose `u8` when processing ASCII-only text for better performance.
//! Choose `char` when Unicode support is needed (accented characters, CJK, etc.).

use std::fmt::{Debug, Display};
use std::hash::Hash;

/// Trait for phonetic character units.
///
/// This trait abstracts over `u8` (ASCII bytes) and `char` (Unicode characters),
/// enabling generic phonetic types that work with both representations.
///
/// # Required Bounds
///
/// Implementations must satisfy:
/// - `Copy + Clone` - Efficient value semantics
/// - `Eq + PartialEq + Ord + Hash` - Comparison and collection support
/// - `Debug + Display` - Formatting for errors and diagnostics
/// - `Send + Sync + 'static` - Thread safety for shared rule sets
///
/// # Examples
///
/// Using generic phonetic types:
///
/// ```rust,ignore
/// use liblevenshtein::phonetic::common::PhoneticUnit;
/// use liblevenshtein::phonetic::types::Phone;
///
/// fn process_phones<U: PhoneticUnit>(phones: &[Phone<U>]) {
///     for phone in phones {
///         println!("{:?}", phone);
///     }
/// }
/// ```
pub trait PhoneticUnit:
    Copy + Clone + Eq + PartialEq + Ord + Hash + Debug + Display + Send + Sync + 'static
{
    /// The display width for alignment in error messages.
    const DISPLAY_WIDTH: usize;

    /// Convert a string to a vector of units.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to convert
    ///
    /// # Returns
    ///
    /// A vector of units representing the string.
    ///
    /// # Panics (u8 only)
    ///
    /// For `u8`, this will panic if the string contains non-ASCII characters.
    /// Use `try_from_str` for fallible conversion.
    fn from_str(s: &str) -> Vec<Self>;

    /// Try to convert a string to a vector of units.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to convert
    ///
    /// # Returns
    ///
    /// `Some(Vec<Self>)` if conversion succeeds, `None` if it fails
    /// (e.g., non-ASCII characters when converting to `u8`).
    fn try_from_str(s: &str) -> Option<Vec<Self>>;

    /// Convert a slice of units to a string.
    ///
    /// # Arguments
    ///
    /// * `units` - The units to convert
    ///
    /// # Returns
    ///
    /// A string representation of the units.
    fn units_to_string(units: &[Self]) -> String;

    /// Convert a single unit to lowercase for case-insensitive matching.
    ///
    /// # Arguments
    ///
    /// * `unit` - The unit to lowercase
    ///
    /// # Returns
    ///
    /// The lowercase version of the unit.
    fn to_lowercase(unit: Self) -> Self;

    /// Check if this unit is an English vowel (a, e, i, o, u).
    ///
    /// For multi-script support, use the `VowelClassifier` trait instead.
    /// This method provides a quick default for English/Latin text.
    ///
    /// # Arguments
    ///
    /// * `unit` - The unit to check
    ///
    /// # Returns
    ///
    /// `true` if the unit is a vowel, `false` otherwise.
    fn is_vowel(unit: Self) -> bool;

    /// Check if this unit is an alphabetic character.
    ///
    /// Used to determine if a unit is a consonant (alphabetic but not vowel).
    fn is_alphabetic(unit: Self) -> bool;

    /// Create a unit from a char.
    ///
    /// # Arguments
    ///
    /// * `c` - The character to convert
    ///
    /// # Returns
    ///
    /// `Some(unit)` if the conversion is valid, `None` otherwise.
    /// For `u8`, returns `None` for non-ASCII characters.
    fn from_char(c: char) -> Option<Self>;

    /// Convert a unit to a char.
    ///
    /// This always succeeds since both `u8` and `char` can be represented as `char`.
    fn to_char(unit: Self) -> char;

    /// Format a unit for display in error messages.
    ///
    /// For `u8`, displays the character if printable, otherwise the hex code.
    /// For `char`, displays the character directly.
    fn format_for_display(unit: Self) -> String {
        Self::to_char(unit).to_string()
    }
}

// ============================================================================
// u8 Implementation (ASCII/byte-level)
// ============================================================================

impl PhoneticUnit for u8 {
    const DISPLAY_WIDTH: usize = 1;

    #[inline]
    fn from_str(s: &str) -> Vec<Self> {
        s.bytes().collect()
    }

    #[inline]
    fn try_from_str(s: &str) -> Option<Vec<Self>> {
        if s.is_ascii() {
            Some(s.bytes().collect())
        } else {
            None
        }
    }

    #[inline]
    fn units_to_string(units: &[Self]) -> String {
        String::from_utf8_lossy(units).into_owned()
    }

    #[inline]
    fn to_lowercase(unit: Self) -> Self {
        unit.to_ascii_lowercase()
    }

    #[inline]
    fn is_vowel(unit: Self) -> bool {
        matches!(
            unit,
            b'a' | b'e' | b'i' | b'o' | b'u' | b'A' | b'E' | b'I' | b'O' | b'U'
        )
    }

    #[inline]
    fn is_alphabetic(unit: Self) -> bool {
        unit.is_ascii_alphabetic()
    }

    #[inline]
    fn from_char(c: char) -> Option<Self> {
        u8::try_from(u32::from(c)).ok().filter(u8::is_ascii)
    }

    #[inline]
    fn to_char(unit: Self) -> char {
        char::from(unit)
    }

    fn format_for_display(unit: Self) -> String {
        if unit.is_ascii_graphic() || unit == b' ' {
            char::from(unit).to_string()
        } else {
            format!("0x{:02X}", unit)
        }
    }
}

// ============================================================================
// char Implementation (Unicode/character-level)
// ============================================================================

impl PhoneticUnit for char {
    const DISPLAY_WIDTH: usize = 1;

    #[inline]
    fn from_str(s: &str) -> Vec<Self> {
        s.chars().collect()
    }

    #[inline]
    fn try_from_str(s: &str) -> Option<Vec<Self>> {
        Some(s.chars().collect())
    }

    #[inline]
    fn units_to_string(units: &[Self]) -> String {
        units.iter().collect()
    }

    #[inline]
    fn to_lowercase(unit: Self) -> Self {
        // Unicode case folding - take first character of lowercase mapping
        unit.to_lowercase().next().unwrap_or(unit)
    }

    #[inline]
    fn is_vowel(unit: Self) -> bool {
        matches!(
            unit,
            'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U'
        )
    }

    #[inline]
    fn is_alphabetic(unit: Self) -> bool {
        unit.is_alphabetic()
    }

    #[inline]
    fn from_char(c: char) -> Option<Self> {
        Some(c)
    }

    #[inline]
    fn to_char(unit: Self) -> char {
        unit
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Check if a unit is a consonant (alphabetic but not a vowel).
///
/// This is a convenience function that uses the `PhoneticUnit` trait methods.
#[inline]
pub fn is_consonant<U: PhoneticUnit>(unit: U) -> bool {
    U::is_alphabetic(unit) && !U::is_vowel(unit)
}

/// Convert a slice of units to lowercase.
///
/// # Arguments
///
/// * `units` - The units to convert
///
/// # Returns
///
/// A vector of lowercase units.
pub fn to_lowercase_vec<U: PhoneticUnit>(units: &[U]) -> Vec<U> {
    units.iter().map(|&u| U::to_lowercase(u)).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // u8 (byte-level) tests
    // ========================================================================

    #[test]
    fn test_u8_from_str() {
        let units: Vec<u8> = PhoneticUnit::from_str("hello");
        assert_eq!(units, vec![b'h', b'e', b'l', b'l', b'o']);
    }

    #[test]
    fn test_u8_try_from_str_ascii() {
        let units: Option<Vec<u8>> = PhoneticUnit::try_from_str("hello");
        assert_eq!(units, Some(vec![b'h', b'e', b'l', b'l', b'o']));
    }

    #[test]
    fn test_u8_try_from_str_non_ascii() {
        let units: Option<Vec<u8>> = PhoneticUnit::try_from_str("héllo");
        assert_eq!(units, None);
    }

    #[test]
    fn test_u8_units_to_string() {
        let s = u8::units_to_string(b"hello");
        assert_eq!(s, "hello");
    }

    #[test]
    fn test_u8_to_lowercase() {
        assert_eq!(u8::to_lowercase(b'A'), b'a');
        assert_eq!(u8::to_lowercase(b'Z'), b'z');
        assert_eq!(u8::to_lowercase(b'a'), b'a');
        assert_eq!(u8::to_lowercase(b'5'), b'5');
    }

    #[test]
    fn test_u8_is_vowel() {
        assert!(u8::is_vowel(b'a'));
        assert!(u8::is_vowel(b'e'));
        assert!(u8::is_vowel(b'i'));
        assert!(u8::is_vowel(b'o'));
        assert!(u8::is_vowel(b'u'));
        assert!(u8::is_vowel(b'A'));
        assert!(u8::is_vowel(b'E'));
        assert!(!u8::is_vowel(b'b'));
        assert!(!u8::is_vowel(b'z'));
    }

    #[test]
    fn test_u8_from_char() {
        assert_eq!(u8::from_char('\0'), Some(0x00));
        assert_eq!(u8::from_char('a'), Some(b'a'));
        assert_eq!(u8::from_char('Z'), Some(b'Z'));
        assert_eq!(u8::from_char('\u{7F}'), Some(0x7F));
        assert_eq!(u8::from_char('\u{80}'), None);
        assert_eq!(u8::from_char('é'), None); // Non-ASCII
    }

    #[test]
    fn test_u8_to_char() {
        assert_eq!(u8::to_char(b'a'), 'a');
        assert_eq!(u8::to_char(b'Z'), 'Z');
        assert_eq!(u8::to_char(0xFF), 'ÿ');
    }

    #[test]
    fn test_u8_format_for_display() {
        assert_eq!(u8::format_for_display(b'a'), "a");
        assert_eq!(u8::format_for_display(b' '), " ");
        assert_eq!(u8::format_for_display(0x00), "0x00");
        assert_eq!(u8::format_for_display(0x7F), "0x7F");
    }

    // ========================================================================
    // char (Unicode) tests
    // ========================================================================

    #[test]
    fn test_char_from_str() {
        let units: Vec<char> = PhoneticUnit::from_str("hello");
        assert_eq!(units, vec!['h', 'e', 'l', 'l', 'o']);
    }

    #[test]
    fn test_char_from_str_unicode() {
        let units: Vec<char> = PhoneticUnit::from_str("héllo");
        assert_eq!(units, vec!['h', 'é', 'l', 'l', 'o']);
    }

    #[test]
    fn test_char_try_from_str() {
        let units: Option<Vec<char>> = PhoneticUnit::try_from_str("héllo");
        assert_eq!(units, Some(vec!['h', 'é', 'l', 'l', 'o']));
    }

    #[test]
    fn test_char_units_to_string() {
        let s = char::units_to_string(&['h', 'é', 'l', 'l', 'o']);
        assert_eq!(s, "héllo");
    }

    #[test]
    fn test_char_to_lowercase() {
        assert_eq!(<char as PhoneticUnit>::to_lowercase('A'), 'a');
        assert_eq!(<char as PhoneticUnit>::to_lowercase('Z'), 'z');
        assert_eq!(<char as PhoneticUnit>::to_lowercase('É'), 'é');
        assert_eq!(<char as PhoneticUnit>::to_lowercase('a'), 'a');
    }

    #[test]
    fn test_char_is_vowel() {
        assert!(char::is_vowel('a'));
        assert!(char::is_vowel('e'));
        assert!(char::is_vowel('A'));
        assert!(!char::is_vowel('b'));
        assert!(!char::is_vowel('é')); // Accented vowels need VowelClassifier
    }

    #[test]
    fn test_char_from_char() {
        assert_eq!(char::from_char('a'), Some('a'));
        assert_eq!(char::from_char('é'), Some('é'));
    }

    #[test]
    fn test_char_to_char() {
        assert_eq!(char::to_char('a'), 'a');
        assert_eq!(char::to_char('é'), 'é');
    }

    // ========================================================================
    // Helper function tests
    // ========================================================================

    #[test]
    fn test_is_consonant_u8() {
        assert!(is_consonant(b'b'));
        assert!(is_consonant(b'k'));
        assert!(!is_consonant(b'a'));
        assert!(!is_consonant(b'5'));
    }

    #[test]
    fn test_is_consonant_char() {
        assert!(is_consonant('b'));
        assert!(is_consonant('k'));
        assert!(!is_consonant('a'));
        assert!(!is_consonant('5'));
    }

    #[test]
    fn test_to_lowercase_vec_u8() {
        let input: Vec<u8> = vec![b'H', b'E', b'L', b'L', b'O'];
        let result = to_lowercase_vec(&input);
        assert_eq!(result, vec![b'h', b'e', b'l', b'l', b'o']);
    }

    #[test]
    fn test_to_lowercase_vec_char() {
        let input: Vec<char> = vec!['H', 'É', 'L', 'L', 'O'];
        let result = to_lowercase_vec(&input);
        assert_eq!(result, vec!['h', 'é', 'l', 'l', 'o']);
    }
}
