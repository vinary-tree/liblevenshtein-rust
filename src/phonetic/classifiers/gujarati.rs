//! Gujarati script vowel classifier.
//!
//! Supports Gujarati (ગુજરાતી) and related languages.
//! Gujarati is an abugida derived from Devanagari but without the
//! characteristic headline (shirorekha).

use super::VowelClassifier;

/// Vowel classifier for Gujarati script.
///
/// Gujarati has two types of vowels:
/// - **Independent vowels (svara)**: Stand-alone vowel letters (અ, આ, ઇ, ઈ, etc.)
/// - **Dependent vowels (matra)**: Vowel signs attached to consonants (ા, િ, ી, etc.)
///
/// Like Devanagari, the inherent vowel is /a/. The virama (્) removes it.
#[derive(Debug, Clone, Copy, Default)]
pub struct GujaratiClassifier;

impl GujaratiClassifier {
    /// Create a new Gujarati classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Gujarati vowels (independent and dependent).
static GUJARATI_VOWELS: &[char] = &[
    // Independent vowels (svara)
    'અ', // a (short)
    'આ', // aa (long)
    'ઇ', // i (short)
    'ઈ', // ii (long)
    'ઉ', // u (short)
    'ઊ', // uu (long)
    'ઋ', // ri (vocalic r)
    'એ', // e
    'ઐ', // ai
    'ઓ', // o
    'ઔ', // au
    // Dependent vowels (matra)
    'ા', // aa matra
    'િ', // i matra
    'ી', // ii matra
    'ુ',  // u matra
    'ૂ',  // uu matra
    'ૃ',  // ri matra
    'ે',  // e matra
    'ૈ',  // ai matra
    'ો', // o matra
    'ૌ', // au matra
];

impl VowelClassifier for GujaratiClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Independent vowels (Gujarati block)
            0x0A85..=0x0A94 => true, // અ through ઔ

            // Dependent vowel signs (matras)
            0x0ABE..=0x0ACC => true, // ા through ૌ

            // Vowel sign for vocalic r (long form)
            0x0AE0..=0x0AE3 => true, // ૠ, ૡ, ૢ, ૣ

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Gujarati"
    }

    fn vowels(&self) -> &[char] {
        GUJARATI_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Consonants (ka through ha)
            0x0A95..=0x0AB9 => true,

            // Virama is neither vowel nor consonant
            0x0ACD => false,

            // Chandrabindu, anusvara, visarga are special marks
            0x0A81..=0x0A83 => false,

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_vowels() {
        let c = GujaratiClassifier::new();
        assert!(c.is_vowel('અ')); // a
        assert!(c.is_vowel('આ')); // aa
        assert!(c.is_vowel('ઇ')); // i
        assert!(c.is_vowel('ઈ')); // ii
        assert!(c.is_vowel('ઉ')); // u
        assert!(c.is_vowel('ઊ')); // uu
        assert!(c.is_vowel('ઋ')); // ri
        assert!(c.is_vowel('એ')); // e
        assert!(c.is_vowel('ઐ')); // ai
        assert!(c.is_vowel('ઓ')); // o
        assert!(c.is_vowel('ઔ')); // au
    }

    #[test]
    fn test_dependent_vowels() {
        let c = GujaratiClassifier::new();
        assert!(c.is_vowel('ા')); // aa matra
        assert!(c.is_vowel('િ')); // i matra
        assert!(c.is_vowel('ી')); // ii matra
        assert!(c.is_vowel('ુ')); // u matra
        assert!(c.is_vowel('ૂ')); // uu matra
        assert!(c.is_vowel('ૃ')); // ri matra
        assert!(c.is_vowel('ે')); // e matra
        assert!(c.is_vowel('ૈ')); // ai matra
    }

    #[test]
    fn test_consonants() {
        let c = GujaratiClassifier::new();
        assert!(!c.is_vowel('ક')); // ka
        assert!(!c.is_vowel('ખ')); // kha
        assert!(!c.is_vowel('ગ')); // ga
        assert!(!c.is_vowel('ઘ')); // gha
        assert!(!c.is_vowel('ચ')); // cha
        assert!(!c.is_vowel('જ')); // ja
        assert!(!c.is_vowel('ત')); // ta
        assert!(!c.is_vowel('દ')); // da
        assert!(!c.is_vowel('ન')); // na
        assert!(!c.is_vowel('પ')); // pa
        assert!(!c.is_vowel('મ')); // ma
        assert!(!c.is_vowel('હ')); // ha
    }

    #[test]
    fn test_is_consonant() {
        let c = GujaratiClassifier::new();
        assert!(c.is_consonant('ક')); // ka
        assert!(c.is_consonant('ગ')); // ga
        assert!(c.is_consonant('ત')); // ta
        assert!(c.is_consonant('ન')); // na
        assert!(c.is_consonant('મ')); // ma
        assert!(c.is_consonant('હ')); // ha
    }

    #[test]
    fn test_virama_not_vowel() {
        let c = GujaratiClassifier::new();
        assert!(!c.is_vowel('્')); // virama
        assert!(!c.is_consonant('્')); // virama is neither
    }

    #[test]
    fn test_diacritics() {
        let c = GujaratiClassifier::new();
        // Chandrabindu, anusvara, visarga are not vowels or consonants
        assert!(!c.is_vowel('ઁ')); // chandrabindu
        assert!(!c.is_vowel('ં')); // anusvara
        assert!(!c.is_vowel('ઃ')); // visarga
        assert!(!c.is_consonant('ઁ'));
        assert!(!c.is_consonant('ં'));
        assert!(!c.is_consonant('ઃ'));
    }
}
