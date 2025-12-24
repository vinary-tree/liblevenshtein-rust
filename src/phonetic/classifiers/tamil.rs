//! Tamil script vowel classifier.
//!
//! Supports Tamil (தமிழ்) and related languages.
//! Tamil is the oldest Dravidian script, with unique features distinct
//! from other Brahmic scripts.

use super::VowelClassifier;

/// Vowel classifier for Tamil script.
///
/// Tamil has two types of vowels:
/// - **Independent vowels (உயிரெழுத்து)**: Stand-alone vowel letters (அ, ஆ, இ, ஈ, etc.)
/// - **Dependent vowels (உயிர் குறி)**: Vowel signs attached to consonants (ா, ி, ீ, etc.)
///
/// Unlike other Brahmic scripts, Tamil has a limited consonant set with
/// unique letters like ழ (retroflex approximant).
#[derive(Debug, Clone, Copy, Default)]
pub struct TamilClassifier;

impl TamilClassifier {
    /// Create a new Tamil classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Tamil vowels (independent and dependent).
static TAMIL_VOWELS: &[char] = &[
    // Independent vowels (உயிரெழுத்து)
    'அ', // a (short)
    'ஆ', // aa (long)
    'இ', // i (short)
    'ஈ', // ii (long)
    'உ', // u (short)
    'ஊ', // uu (long)
    'எ', // e (short)
    'ஏ', // ee (long)
    'ஐ', // ai
    'ஒ', // o (short)
    'ஓ', // oo (long)
    'ஔ', // au
    // Dependent vowels (உயிர் குறி)
    'ா', // aa matra
    'ி', // i matra
    'ீ', // ii matra
    'ு', // u matra
    'ூ', // uu matra
    'ெ', // e matra
    'ே', // ee matra
    'ை', // ai matra
    'ொ', // o matra
    'ோ', // oo matra
    'ௌ', // au matra
];

impl VowelClassifier for TamilClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Independent vowels (Tamil block)
            0x0B85..=0x0B94 => true, // அ through ஔ

            // Dependent vowel signs (matras)
            0x0BBE..=0x0BC8 => true, // ா through ை
            0x0BCA..=0x0BCC => true, // ொ through ௌ

            // Au length mark
            0x0BD7 => true, // ௗ

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Tamil"
    }

    fn vowels(&self) -> &[char] {
        TAMIL_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Native Tamil consonants
            0x0B95 => true, // க (ka)
            0x0B99 => true, // ங (nga)
            0x0B9A => true, // ச (ca)
            0x0B9C => true, // ஜ (ja - Grantha)
            0x0B9E => true, // ஞ (nya)
            0x0B9F => true, // ட (Ta)
            0x0BA3 => true, // ண (Na)
            0x0BA4 => true, // த (ta)
            0x0BA8 => true, // ந (na dental)
            0x0BAA => true, // ப (pa)
            0x0BAE => true, // ம (ma)
            0x0BAF => true, // ய (ya)
            0x0BB0 => true, // ர (ra)
            0x0BB1 => true, // ற (rra - alveolar)
            0x0BB2 => true, // ல (la)
            0x0BB3 => true, // ள (lla - retroflex)
            0x0BB4 => true, // ழ (zha - retroflex approximant)
            0x0BB5 => true, // வ (va)
            0x0BB6 => true, // ஶ (sha - Grantha)
            0x0BB7 => true, // ஷ (Sha - Grantha)
            0x0BB8 => true, // ஸ (sa - Grantha)
            0x0BB9 => true, // ஹ (ha - Grantha)
            0x0BA9 => true, // ன (na alveolar)

            // Virama is neither vowel nor consonant
            0x0BCD => false,

            // Anusvara, visarga (rare in Tamil)
            0x0B82..=0x0B83 => false,

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_vowels() {
        let c = TamilClassifier::new();
        assert!(c.is_vowel('அ')); // a
        assert!(c.is_vowel('ஆ')); // aa
        assert!(c.is_vowel('இ')); // i
        assert!(c.is_vowel('ஈ')); // ii
        assert!(c.is_vowel('உ')); // u
        assert!(c.is_vowel('ஊ')); // uu
        assert!(c.is_vowel('எ')); // e
        assert!(c.is_vowel('ஏ')); // ee
        assert!(c.is_vowel('ஐ')); // ai
        assert!(c.is_vowel('ஒ')); // o
        assert!(c.is_vowel('ஓ')); // oo
        assert!(c.is_vowel('ஔ')); // au
    }

    #[test]
    fn test_dependent_vowels() {
        let c = TamilClassifier::new();
        assert!(c.is_vowel('ா')); // aa matra
        assert!(c.is_vowel('ி')); // i matra
        assert!(c.is_vowel('ீ')); // ii matra
        assert!(c.is_vowel('ு')); // u matra
        assert!(c.is_vowel('ூ')); // uu matra
        assert!(c.is_vowel('ெ')); // e matra
        assert!(c.is_vowel('ே')); // ee matra
        assert!(c.is_vowel('ை')); // ai matra
    }

    #[test]
    fn test_native_consonants() {
        let c = TamilClassifier::new();
        assert!(!c.is_vowel('க')); // ka
        assert!(!c.is_vowel('ங')); // nga
        assert!(!c.is_vowel('ச')); // ca
        assert!(!c.is_vowel('ட')); // Ta
        assert!(!c.is_vowel('த')); // ta
        assert!(!c.is_vowel('ப')); // pa
        assert!(!c.is_vowel('ம')); // ma
        assert!(!c.is_vowel('ய')); // ya
        assert!(!c.is_vowel('ர')); // ra
        assert!(!c.is_vowel('ல')); // la
        assert!(!c.is_vowel('வ')); // va
    }

    #[test]
    fn test_unique_tamil_consonants() {
        let c = TamilClassifier::new();
        // Unique Tamil sounds
        assert!(!c.is_vowel('ழ')); // zha - retroflex approximant
        assert!(!c.is_vowel('ற')); // rra - alveolar trill
        assert!(!c.is_vowel('ன')); // na - alveolar nasal
        assert!(!c.is_vowel('ள')); // lla - retroflex lateral
    }

    #[test]
    fn test_grantha_consonants() {
        let c = TamilClassifier::new();
        // Grantha letters for Sanskrit loanwords
        assert!(!c.is_vowel('ஜ')); // ja
        assert!(!c.is_vowel('ஶ')); // sha
        assert!(!c.is_vowel('ஷ')); // Sha
        assert!(!c.is_vowel('ஸ')); // sa
        assert!(!c.is_vowel('ஹ')); // ha
    }

    #[test]
    fn test_is_consonant() {
        let c = TamilClassifier::new();
        assert!(c.is_consonant('க')); // ka
        assert!(c.is_consonant('த')); // ta
        assert!(c.is_consonant('ம')); // ma
        assert!(c.is_consonant('ழ')); // zha
        assert!(c.is_consonant('ற')); // rra
        assert!(c.is_consonant('ன')); // na alveolar
    }

    #[test]
    fn test_virama_not_vowel() {
        let c = TamilClassifier::new();
        assert!(!c.is_vowel('்')); // pulli/virama
        assert!(!c.is_consonant('்')); // virama is neither
    }

    #[test]
    fn test_diacritics() {
        let c = TamilClassifier::new();
        // Anusvara, visarga (rare in Tamil)
        assert!(!c.is_vowel('ஂ')); // anusvara
        assert!(!c.is_vowel('ஃ')); // visarga (aytham)
        assert!(!c.is_consonant('ஂ'));
        assert!(!c.is_consonant('ஃ'));
    }
}
