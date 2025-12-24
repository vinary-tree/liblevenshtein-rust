//! Hebrew script vowel classifier.
//!
//! Hebrew is an abjad where consonants are written but vowels are typically
//! omitted or indicated with optional diacritical marks (niqqud).

use super::VowelClassifier;

/// Vowel classifier for Hebrew script.
///
/// Hebrew has a complex vowel system:
/// - **Niqqud (vowel points)**: Optional diacritics that indicate vowels
/// - **Matres lectionis**: Consonant letters used to indicate vowels
///   - ו (vav) can indicate /o/ or /u/
///   - י (yod) can indicate /i/ or /e/
///   - ה (he) at word end can indicate /a/ or /e/
///   - א (alef) can carry vowels
///
/// This classifier recognizes niqqud as vowels but treats consonant letters
/// as consonants (even when they serve as matres lectionis, since context
/// is needed to determine their function).
#[derive(Debug, Clone, Copy, Default)]
pub struct HebrewClassifier;

impl HebrewClassifier {
    /// Create a new Hebrew classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Hebrew niqqud (vowel points).
static HEBREW_VOWELS: &[char] = &[
    '\u{05B0}', // Sheva
    '\u{05B1}', // Hataf Segol
    '\u{05B2}', // Hataf Patah
    '\u{05B3}', // Hataf Qamats
    '\u{05B4}', // Hiriq
    '\u{05B5}', // Tsere
    '\u{05B6}', // Segol
    '\u{05B7}', // Patah
    '\u{05B8}', // Qamats
    '\u{05B9}', // Holam
    '\u{05BA}', // Holam Haser (for vav)
    '\u{05BB}', // Qubuts
    '\u{05BC}', // Dagesh (not a vowel, but included for completeness)
];

impl VowelClassifier for HebrewClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Hebrew niqqud (vowel points)
            0x05B0..=0x05BB => true, // Sheva through Qubuts

            // Holam Haser for vav
            0x05BA => true,

            // Note: Dagesh (0x05BC) is NOT a vowel - it's a consonant modifier

            // Meteg (0x05BD) is a cantillation mark, not a vowel

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Hebrew"
    }

    fn vowels(&self) -> &[char] {
        HEBREW_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Hebrew letters (consonants)
            0x05D0..=0x05EA => true, // Alef through Tav

            // Final letter forms
            0x05DA | 0x05DD | 0x05DF | 0x05E3 | 0x05E5 => true, // Final Kaf, Mem, Nun, Pe, Tsadi

            _ => false,
        }
    }

    fn normalize(&self, input: &str) -> String {
        // NFD decomposition to separate base letters from combining marks
        use unicode_normalization::UnicodeNormalization;
        input.nfd().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_niqqud_vowels() {
        let c = HebrewClassifier::new();
        assert!(c.is_vowel('\u{05B0}')); // Sheva
        assert!(c.is_vowel('\u{05B4}')); // Hiriq
        assert!(c.is_vowel('\u{05B5}')); // Tsere
        assert!(c.is_vowel('\u{05B6}')); // Segol
        assert!(c.is_vowel('\u{05B7}')); // Patah
        assert!(c.is_vowel('\u{05B8}')); // Qamats
        assert!(c.is_vowel('\u{05B9}')); // Holam
        assert!(c.is_vowel('\u{05BB}')); // Qubuts
    }

    #[test]
    fn test_consonant_letters() {
        let c = HebrewClassifier::new();
        assert!(!c.is_vowel('א')); // Alef
        assert!(!c.is_vowel('ב')); // Bet
        assert!(!c.is_vowel('ג')); // Gimel
        assert!(!c.is_vowel('ד')); // Dalet
        assert!(!c.is_vowel('ה')); // He
        assert!(!c.is_vowel('ו')); // Vav
        assert!(!c.is_vowel('ז')); // Zayin
        assert!(!c.is_vowel('ח')); // Het
        assert!(!c.is_vowel('ט')); // Tet
        assert!(!c.is_vowel('י')); // Yod
    }

    #[test]
    fn test_is_consonant() {
        let c = HebrewClassifier::new();
        assert!(c.is_consonant('א')); // Alef
        assert!(c.is_consonant('ב')); // Bet
        assert!(c.is_consonant('כ')); // Kaf
        assert!(c.is_consonant('ך')); // Final Kaf
        assert!(c.is_consonant('מ')); // Mem
        assert!(c.is_consonant('ם')); // Final Mem
        assert!(c.is_consonant('ת')); // Tav
    }

    #[test]
    fn test_dagesh_not_vowel() {
        let c = HebrewClassifier::new();
        assert!(!c.is_vowel('\u{05BC}')); // Dagesh
    }
}
