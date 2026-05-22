//! Gurmukhi script vowel classifier.
//!
//! Supports Punjabi (ਪੰਜਾਬੀ) written in Gurmukhi script (used in India).
//! Gurmukhi is a Brahmic abugida with unique features distinct from Devanagari.

use super::VowelClassifier;

/// Vowel classifier for Gurmukhi script.
///
/// Gurmukhi has two types of vowels:
/// - **Independent vowels (ਸਵਰ)**: Stand-alone vowel letters (ਅ, ਆ, ਇ, ਈ, etc.)
/// - **Dependent vowels (ਲਗਾਂ/ਮਾਤਰਾ)**: Vowel signs attached to consonants (ਾ, ਿ, ੀ, etc.)
///
/// The inherent vowel is /a/. The virama (ਹਲੰਤ ੍) removes it.
///
/// Note: Punjabi is a tonal language with 3 tones, but these are not
/// fully marked in the script.
#[derive(Debug, Clone, Copy, Default)]
pub struct GurmukhiClassifier;

impl GurmukhiClassifier {
    /// Create a new Gurmukhi classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Gurmukhi vowels (independent and dependent).
static GURMUKHI_VOWELS: &[char] = &[
    // Independent vowels (ਸਵਰ)
    'ਅ', // a
    'ਆ', // aa
    'ਇ', // i
    'ਈ', // ii
    'ਉ', // u
    'ਊ', // uu
    'ਏ', // e
    'ਐ', // ai
    'ਓ', // o
    'ਔ', // au
    // Dependent vowels (ਲਗਾਂ/ਮਾਤਰਾ)
    'ਾ', // aa matra
    'ਿ', // i matra
    'ੀ', // ii matra
    'ੁ',  // u matra
    'ੂ',  // uu matra
    'ੇ',  // e matra
    'ੈ',  // ai matra
    'ੋ',  // o matra
    'ੌ',  // au matra
];

impl VowelClassifier for GurmukhiClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Independent vowels (Gurmukhi block)
            0x0A05..=0x0A0A => true, // ਅ through ਊ
            0x0A0F..=0x0A10 => true, // ਏ, ਐ
            0x0A13..=0x0A14 => true, // ਓ, ਔ

            // Dependent vowel signs (matras)
            0x0A3E..=0x0A42 => true, // ਾ through ੂ
            0x0A47..=0x0A48 => true, // ੇ, ੈ
            0x0A4B..=0x0A4C => true, // ੋ, ੌ

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Gurmukhi"
    }

    fn vowels(&self) -> &[char] {
        GURMUKHI_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Consonants (ka through ha)
            0x0A15..=0x0A28 => true, // ਕ through ਨ
            0x0A2A..=0x0A30 => true, // ਪ through ਰ
            0x0A32..=0x0A33 => true, // ਲ, ਲ਼
            0x0A35..=0x0A36 => true, // ਵ, ਸ਼
            0x0A38..=0x0A39 => true, // ਸ, ਹ

            // Nukta consonants
            0x0A59..=0x0A5C => true, // ਖ਼, ਗ਼, ਜ਼, ੜ
            0x0A5E => true,          // ਫ਼

            // Virama is neither vowel nor consonant
            0x0A4D => false,

            // Anusvara, visarga, etc. are special marks
            0x0A01..=0x0A03 => false,

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_vowels() {
        let c = GurmukhiClassifier::new();
        assert!(c.is_vowel('ਅ')); // a
        assert!(c.is_vowel('ਆ')); // aa
        assert!(c.is_vowel('ਇ')); // i
        assert!(c.is_vowel('ਈ')); // ii
        assert!(c.is_vowel('ਉ')); // u
        assert!(c.is_vowel('ਊ')); // uu
        assert!(c.is_vowel('ਏ')); // e
        assert!(c.is_vowel('ਐ')); // ai
        assert!(c.is_vowel('ਓ')); // o
        assert!(c.is_vowel('ਔ')); // au
    }

    #[test]
    fn test_dependent_vowels() {
        let c = GurmukhiClassifier::new();
        assert!(c.is_vowel('ਾ')); // aa matra
        assert!(c.is_vowel('ਿ')); // i matra
        assert!(c.is_vowel('ੀ')); // ii matra
        assert!(c.is_vowel('ੁ')); // u matra
        assert!(c.is_vowel('ੂ')); // uu matra
        assert!(c.is_vowel('ੇ')); // e matra
        assert!(c.is_vowel('ੈ')); // ai matra
    }

    #[test]
    fn test_consonants() {
        let c = GurmukhiClassifier::new();
        assert!(!c.is_vowel('ਕ')); // ka
        assert!(!c.is_vowel('ਖ')); // kha
        assert!(!c.is_vowel('ਗ')); // ga
        assert!(!c.is_vowel('ਘ')); // gha
        assert!(!c.is_vowel('ਚ')); // cha
        assert!(!c.is_vowel('ਜ')); // ja
        assert!(!c.is_vowel('ਤ')); // ta
        assert!(!c.is_vowel('ਦ')); // da
        assert!(!c.is_vowel('ਨ')); // na
        assert!(!c.is_vowel('ਪ')); // pa
        assert!(!c.is_vowel('ਮ')); // ma
        assert!(!c.is_vowel('ਹ')); // ha
    }

    #[test]
    fn test_nukta_consonants() {
        let c = GurmukhiClassifier::new();
        // Persian/Arabic loanword consonants - nukta combinations are two codepoints
        // Test base consonants separately
        assert!(!c.is_vowel('\u{0A16}')); // ਖ kha (base of ਖ਼)
        assert!(!c.is_vowel('\u{0A17}')); // ਗ ga (base of ਗ਼)
        assert!(!c.is_vowel('\u{0A1C}')); // ਜ ja (base of ਜ਼)
        assert!(!c.is_vowel('\u{0A5C}')); // ੜ rra (single codepoint)
        assert!(!c.is_vowel('\u{0A2B}')); // ਫ pha (base of ਫ਼)
                                          // The nukta (U+0A3C) is a combining mark
        assert!(!c.is_consonant('\u{0A3C}')); // nukta
    }

    #[test]
    fn test_is_consonant() {
        let c = GurmukhiClassifier::new();
        assert!(c.is_consonant('ਕ')); // ka
        assert!(c.is_consonant('ਗ')); // ga
        assert!(c.is_consonant('ਤ')); // ta
        assert!(c.is_consonant('ਨ')); // na
        assert!(c.is_consonant('ਮ')); // ma
        assert!(c.is_consonant('ਹ')); // ha
    }

    #[test]
    fn test_virama_not_vowel() {
        let c = GurmukhiClassifier::new();
        assert!(!c.is_vowel('੍')); // virama
        assert!(!c.is_consonant('੍')); // virama is neither
    }

    #[test]
    fn test_diacritics() {
        let c = GurmukhiClassifier::new();
        // Bindi, tippi, etc. are not vowels or consonants
        assert!(!c.is_vowel('ਁ')); // adak bindi
        assert!(!c.is_vowel('ਂ')); // bindi
        assert!(!c.is_vowel('ਃ')); // visarga
        assert!(!c.is_consonant('ਁ'));
        assert!(!c.is_consonant('ਂ'));
        assert!(!c.is_consonant('ਃ'));
    }
}
