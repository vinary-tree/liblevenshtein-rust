//! Arabic script vowel classifier.
//!
//! Arabic is an abjad where consonants are primary and vowels are indicated
//! by optional diacritical marks (harakat) or long vowel letters.

use super::VowelClassifier;

/// Vowel classifier for Arabic script.
///
/// Arabic has a dual vowel system:
/// - **Short vowels (harakat)**: Diacritical marks written above/below consonants
///   - Fatha (ـَ) = /a/
///   - Kasra (ـِ) = /i/
///   - Damma (ـُ) = /u/
/// - **Long vowels (matres lectionis)**: Consonant letters serving as vowels
///   - Alif (ا) = /aː/
///   - Waw (و) = /uː/ or consonant /w/
///   - Ya (ي) = /iː/ or consonant /j/
///
/// This classifier recognizes harakat as vowels. The long vowel letters (alif,
/// waw, ya) are treated as consonants by default since they have dual functions
/// and context is needed to determine their role.
///
/// Also used for Urdu (with extensions).
#[derive(Debug, Clone, Copy, Default)]
pub struct ArabicClassifier {
    /// Treat alif/waw/ya as vowels (for contexts where they're matres lectionis)
    pub long_vowels_as_vowels: bool,
}

impl ArabicClassifier {
    /// Create a new Arabic classifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a classifier that treats long vowel letters as vowels.
    ///
    /// Use this for contexts where alif, waw, and ya are known to be
    /// functioning as vowels rather than consonants.
    pub fn with_long_vowels() -> Self {
        Self {
            long_vowels_as_vowels: true,
        }
    }
}

/// Arabic harakat (vowel marks).
static ARABIC_VOWELS: &[char] = &[
    '\u{064E}', // Fatha
    '\u{064F}', // Damma
    '\u{0650}', // Kasra
    '\u{064B}', // Fathatan (tanwin)
    '\u{064C}', // Dammatan (tanwin)
    '\u{064D}', // Kasratan (tanwin)
    '\u{0670}', // Superscript Alef
];

impl VowelClassifier for ArabicClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        matches!(code, 0x064B..=0x0650 | 0x0670)
            || (self.long_vowels_as_vowels && matches!(code, 0x0627 | 0x0648 | 0x0649 | 0x064A))
    }

    fn script_name(&self) -> &'static str {
        "Arabic"
    }

    fn vowels(&self) -> &[char] {
        ARABIC_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        let base_letter =
            (0x0621..=0x063A).contains(&code) && (code != 0x0627 || !self.long_vowels_as_vowels);
        let medial_letter = (0x0641..=0x0647).contains(&code)
            || (!self.long_vowels_as_vowels && matches!(code, 0x0648 | 0x064A));
        let urdu_extension = matches!(
            code,
            0x0679 | 0x067E | 0x0686 | 0x0688 | 0x0691 | 0x0698 | 0x06AF | 0x06BA
        );

        base_letter || medial_letter || urdu_extension
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
    fn test_harakat_vowels() {
        let c = ArabicClassifier::new();
        assert!(c.is_vowel('\u{064E}')); // Fatha
        assert!(c.is_vowel('\u{064F}')); // Damma
        assert!(c.is_vowel('\u{0650}')); // Kasra
    }

    #[test]
    fn test_tanwin() {
        let c = ArabicClassifier::new();
        assert!(c.is_vowel('\u{064B}')); // Fathatan
        assert!(c.is_vowel('\u{064C}')); // Dammatan
        assert!(c.is_vowel('\u{064D}')); // Kasratan
    }

    #[test]
    fn test_consonant_letters() {
        let c = ArabicClassifier::new();
        assert!(!c.is_vowel('ب')); // Ba
        assert!(!c.is_vowel('ت')); // Ta
        assert!(!c.is_vowel('ث')); // Tha
        assert!(!c.is_vowel('ج')); // Jim
        assert!(!c.is_vowel('ح')); // Ha
        assert!(!c.is_vowel('خ')); // Kha
    }

    #[test]
    fn test_long_vowel_letters_default() {
        let c = ArabicClassifier::new();
        // By default, long vowel letters are treated as consonants
        assert!(!c.is_vowel('ا')); // Alif
        assert!(!c.is_vowel('و')); // Waw
        assert!(!c.is_vowel('ي')); // Ya
    }

    #[test]
    fn test_long_vowel_letters_enabled() {
        let c = ArabicClassifier::with_long_vowels();
        assert!(c.is_vowel('ا')); // Alif
        assert!(c.is_vowel('و')); // Waw
        assert!(c.is_vowel('ي')); // Ya
    }

    #[test]
    fn test_is_consonant() {
        let c = ArabicClassifier::new();
        assert!(c.is_consonant('ب')); // Ba
        assert!(c.is_consonant('ك')); // Kaf
        assert!(c.is_consonant('ل')); // Lam
        assert!(c.is_consonant('م')); // Mim
        assert!(c.is_consonant('ن')); // Nun
    }

    #[test]
    fn test_urdu_consonants() {
        let c = ArabicClassifier::new();
        assert!(c.is_consonant('پ')); // Pe
        assert!(c.is_consonant('چ')); // Che
        assert!(c.is_consonant('ژ')); // Zhe
        assert!(c.is_consonant('گ')); // Gaf
    }
}
