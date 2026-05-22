//! Telugu script vowel classifier.
//!
//! Supports Telugu (తెలుగు) and related languages.
//! Telugu is a Brahmic abugida with a distinctive rounded appearance,
//! closely related to Kannada script.

use super::VowelClassifier;

/// Vowel classifier for Telugu script.
///
/// Telugu has two types of vowels:
/// - **Independent vowels (అచ్చులు)**: Stand-alone vowel letters (అ, ఆ, ఇ, ఈ, etc.)
/// - **Dependent vowels (మాత్రలు)**: Vowel signs attached to consonants (ా, ి, ీ, etc.)
///
/// The inherent vowel is /a/. The virama (halant ్) removes it.
#[derive(Debug, Clone, Copy, Default)]
pub struct TeluguClassifier;

impl TeluguClassifier {
    /// Create a new Telugu classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Telugu vowels (independent and dependent).
static TELUGU_VOWELS: &[char] = &[
    // Independent vowels (అచ్చులు)
    'అ', // a (short)
    'ఆ', // aa (long)
    'ఇ', // i (short)
    'ఈ', // ii (long)
    'ఉ', // u (short)
    'ఊ', // uu (long)
    'ఋ', // ri (vocalic r)
    'ౠ', // rii (long vocalic r)
    'ఌ', // li (vocalic l)
    'ౡ', // lii (long vocalic l)
    'ఎ', // e (short)
    'ఏ', // ee (long)
    'ఐ', // ai
    'ఒ', // o (short)
    'ఓ', // oo (long)
    'ఔ', // au
    // Dependent vowels (మాత్రలు)
    'ా',  // aa matra
    'ి',  // i matra
    'ీ',  // ii matra
    'ు', // u matra
    'ూ', // uu matra
    'ృ', // ri matra
    'ౄ', // rii matra
    'ె',  // e matra
    'ే',  // ee matra
    'ై',  // ai matra
    'ొ',  // o matra
    'ో',  // oo matra
    'ౌ',  // au matra
];

impl VowelClassifier for TeluguClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Independent vowels (Telugu block)
            0x0C05..=0x0C14 => true, // అ through ఔ

            // Dependent vowel signs (matras)
            0x0C3E..=0x0C4C => true, // ా through ౌ

            // Vowel signs for vocalic r/l (long forms)
            0x0C60..=0x0C63 => true, // ౠ, ౡ, ౢ, ౣ

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Telugu"
    }

    fn vowels(&self) -> &[char] {
        TELUGU_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Consonants (ka through ha)
            0x0C15..=0x0C39 => true,

            // Additional consonants
            0x0C58..=0x0C5A => true, // ౘ, ౙ, ౚ (tsa, dza, rrra)

            // Virama is neither vowel nor consonant
            0x0C4D => false,

            // Chandrabindu, anusvara, visarga are special marks
            0x0C00..=0x0C04 => false,

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_vowels() {
        let c = TeluguClassifier::new();
        assert!(c.is_vowel('అ')); // a
        assert!(c.is_vowel('ఆ')); // aa
        assert!(c.is_vowel('ఇ')); // i
        assert!(c.is_vowel('ఈ')); // ii
        assert!(c.is_vowel('ఉ')); // u
        assert!(c.is_vowel('ఊ')); // uu
        assert!(c.is_vowel('ఋ')); // ri
        assert!(c.is_vowel('ఎ')); // e
        assert!(c.is_vowel('ఏ')); // ee
        assert!(c.is_vowel('ఐ')); // ai
        assert!(c.is_vowel('ఒ')); // o
        assert!(c.is_vowel('ఓ')); // oo
        assert!(c.is_vowel('ఔ')); // au
    }

    #[test]
    fn test_dependent_vowels() {
        let c = TeluguClassifier::new();
        assert!(c.is_vowel('ా')); // aa matra
        assert!(c.is_vowel('ి')); // i matra
        assert!(c.is_vowel('ీ')); // ii matra
        assert!(c.is_vowel('ు')); // u matra
        assert!(c.is_vowel('ూ')); // uu matra
        assert!(c.is_vowel('ృ')); // ri matra
        assert!(c.is_vowel('ె')); // e matra
        assert!(c.is_vowel('ే')); // ee matra
        assert!(c.is_vowel('ై')); // ai matra
    }

    #[test]
    fn test_consonants() {
        let c = TeluguClassifier::new();
        assert!(!c.is_vowel('క')); // ka
        assert!(!c.is_vowel('ఖ')); // kha
        assert!(!c.is_vowel('గ')); // ga
        assert!(!c.is_vowel('ఘ')); // gha
        assert!(!c.is_vowel('చ')); // cha
        assert!(!c.is_vowel('జ')); // ja
        assert!(!c.is_vowel('త')); // ta
        assert!(!c.is_vowel('ద')); // da
        assert!(!c.is_vowel('న')); // na
        assert!(!c.is_vowel('ప')); // pa
        assert!(!c.is_vowel('మ')); // ma
        assert!(!c.is_vowel('హ')); // ha
    }

    #[test]
    fn test_is_consonant() {
        let c = TeluguClassifier::new();
        assert!(c.is_consonant('క')); // ka
        assert!(c.is_consonant('గ')); // ga
        assert!(c.is_consonant('త')); // ta
        assert!(c.is_consonant('న')); // na
        assert!(c.is_consonant('మ')); // ma
        assert!(c.is_consonant('హ')); // ha
        assert!(c.is_consonant('ళ')); // lla
    }

    #[test]
    fn test_virama_not_vowel() {
        let c = TeluguClassifier::new();
        assert!(!c.is_vowel('్')); // virama
        assert!(!c.is_consonant('్')); // virama is neither
    }

    #[test]
    fn test_diacritics() {
        let c = TeluguClassifier::new();
        // Chandrabindu, anusvara, visarga are not vowels or consonants
        assert!(!c.is_vowel('ఁ')); // chandrabindu
        assert!(!c.is_vowel('ం')); // anusvara
        assert!(!c.is_vowel('ః')); // visarga
        assert!(!c.is_consonant('ఁ'));
        assert!(!c.is_consonant('ం'));
        assert!(!c.is_consonant('ః'));
    }
}
