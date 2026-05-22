//! Bengali script vowel classifier.
//!
//! Supports Bengali (বাংলা), Assamese, and related languages.
//! Bengali is an abugida where consonants carry an inherent /ɔ/ vowel
//! (unlike Hindi's /a/).

use super::VowelClassifier;

/// Vowel classifier for Bengali script.
///
/// Bengali has two types of vowels:
/// - **Independent vowels (svara)**: Stand-alone vowel letters (অ, আ, ই, ঈ, etc.)
/// - **Dependent vowels (matra)**: Vowel signs attached to consonants (া, ি, ী, etc.)
///
/// Key difference from Devanagari: The inherent vowel is /ɔ/ (like 'o' in "hot"),
/// not /a/ as in Hindi.
///
/// The hasanta (্) explicitly removes the inherent vowel.
#[derive(Debug, Clone, Copy, Default)]
pub struct BengaliClassifier;

impl BengaliClassifier {
    /// Create a new Bengali classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Bengali vowels (independent and dependent).
static BENGALI_VOWELS: &[char] = &[
    // Independent vowels (svara)
    'অ', // o (inherent vowel, short)
    'আ', // a (long)
    'ই', // i (short)
    'ঈ', // ii (long)
    'উ', // u (short)
    'ঊ', // uu (long)
    'ঋ', // ri (vocalic r)
    'এ', // e
    'ঐ', // oi (diphthong)
    'ও', // o
    'ঔ', // ou (diphthong)
    // Dependent vowels (matra)
    'া',  // a matra
    'ি', // i matra
    'ী', // ii matra
    'ু',  // u matra
    'ূ',  // uu matra
    'ৃ',  // ri matra
    'ে', // e matra
    'ৈ', // oi matra
    'ো', // o matra
    'ৌ', // ou matra
];

impl VowelClassifier for BengaliClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Independent vowels (Bengali block)
            0x0985..=0x0994 => true, // অ through ঔ

            // Dependent vowel signs (matras)
            0x09BE..=0x09CC => true, // া through ৌ

            // Vowel sign for vocalic r (long form)
            0x09E0..=0x09E3 => true, // ৠ, ৡ, ৢ, ৣ

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Bengali"
    }

    fn vowels(&self) -> &[char] {
        BENGALI_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Consonants (ka through ha)
            0x0995..=0x09B9 => true,

            // Additional consonants (with nukta)
            0x09DC..=0x09DF => true, // ড়, ঢ়, য়

            // Hasanta is neither vowel nor consonant (it removes vowel)
            0x09CD => false,

            // Chandrabindu, anusvara, visarga are special marks
            0x0981..=0x0983 => false,

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_vowels() {
        let c = BengaliClassifier::new();
        assert!(c.is_vowel('অ')); // o (inherent)
        assert!(c.is_vowel('আ')); // a
        assert!(c.is_vowel('ই')); // i
        assert!(c.is_vowel('ঈ')); // ii
        assert!(c.is_vowel('উ')); // u
        assert!(c.is_vowel('ঊ')); // uu
        assert!(c.is_vowel('ঋ')); // ri
        assert!(c.is_vowel('এ')); // e
        assert!(c.is_vowel('ঐ')); // oi
        assert!(c.is_vowel('ও')); // o
        assert!(c.is_vowel('ঔ')); // ou
    }

    #[test]
    fn test_dependent_vowels() {
        let c = BengaliClassifier::new();
        assert!(c.is_vowel('া')); // a matra
        assert!(c.is_vowel('ি')); // i matra
        assert!(c.is_vowel('ী')); // ii matra
        assert!(c.is_vowel('ু')); // u matra
        assert!(c.is_vowel('ূ')); // uu matra
        assert!(c.is_vowel('ৃ')); // ri matra
        assert!(c.is_vowel('ে')); // e matra
        assert!(c.is_vowel('ৈ')); // oi matra
    }

    #[test]
    fn test_consonants() {
        let c = BengaliClassifier::new();
        assert!(!c.is_vowel('ক')); // ka
        assert!(!c.is_vowel('খ')); // kha
        assert!(!c.is_vowel('গ')); // ga
        assert!(!c.is_vowel('ঘ')); // gha
        assert!(!c.is_vowel('চ')); // cha
        assert!(!c.is_vowel('জ')); // ja
        assert!(!c.is_vowel('ত')); // ta
        assert!(!c.is_vowel('দ')); // da
        assert!(!c.is_vowel('ন')); // na
        assert!(!c.is_vowel('প')); // pa
        assert!(!c.is_vowel('ম')); // ma
        assert!(!c.is_vowel('হ')); // ha
    }

    #[test]
    fn test_is_consonant() {
        let c = BengaliClassifier::new();
        assert!(c.is_consonant('ক')); // ka
        assert!(c.is_consonant('গ')); // ga
        assert!(c.is_consonant('ত')); // ta
        assert!(c.is_consonant('ন')); // na
        assert!(c.is_consonant('ম')); // ma
        assert!(c.is_consonant('হ')); // ha
    }

    #[test]
    fn test_special_consonants() {
        let c = BengaliClassifier::new();
        // Note: Nukta consonants like ড় are two codepoints (base + nukta)
        // We test the base consonants and the nukta separately
        assert!(c.is_consonant('\u{09A1}')); // ড (da) - base of ড়
        assert!(c.is_consonant('\u{09A2}')); // ঢ (dha) - base of ঢ়
        assert!(c.is_consonant('\u{09AF}')); // য (ya) - base of য়
                                             // The nukta (U+09BC) is a combining mark, not a consonant
        assert!(!c.is_consonant('\u{09BC}')); // nukta
    }

    #[test]
    fn test_hasanta_not_vowel() {
        let c = BengaliClassifier::new();
        assert!(!c.is_vowel('্')); // hasanta
        assert!(!c.is_consonant('্')); // hasanta is neither
    }

    #[test]
    fn test_diacritics() {
        let c = BengaliClassifier::new();
        // Chandrabindu, anusvara, visarga are not vowels or consonants
        assert!(!c.is_vowel('ঁ')); // chandrabindu
        assert!(!c.is_vowel('ং')); // anusvara
        assert!(!c.is_vowel('ঃ')); // visarga
        assert!(!c.is_consonant('ঁ'));
        assert!(!c.is_consonant('ং'));
        assert!(!c.is_consonant('ঃ'));
    }
}
