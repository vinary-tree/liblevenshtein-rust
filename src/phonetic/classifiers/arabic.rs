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
        match code {
            // Short vowels (harakat)
            0x064E => true, // Fatha
            0x064F => true, // Damma
            0x0650 => true, // Kasra

            // Tanwin (nunation) - nasal vowel endings
            0x064B => true, // Fathatan
            0x064C => true, // Dammatan
            0x064D => true, // Kasratan

            // Superscript Alef (dagger alif)
            0x0670 => true,

            // Long vowel letters (when configured)
            0x0627 if self.long_vowels_as_vowels => true, // Alif
            0x0648 if self.long_vowels_as_vowels => true, // Waw
            0x064A if self.long_vowels_as_vowels => true, // Ya
            0x0649 if self.long_vowels_as_vowels => true, // Alif Maksura

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Arabic"
    }

    fn vowels(&self) -> &[char] {
        ARABIC_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Arabic consonants
            0x0621 => true, // Hamza
            0x0622 => true, // Alif with Madda
            0x0623 => true, // Alif with Hamza Above
            0x0624 => true, // Waw with Hamza
            0x0625 => true, // Alif with Hamza Below
            0x0626 => true, // Ya with Hamza
            0x0628 => true, // Ba
            0x062A => true, // Ta
            0x062B => true, // Tha
            0x062C => true, // Jim
            0x062D => true, // Ha
            0x062E => true, // Kha
            0x062F => true, // Dal
            0x0630 => true, // Dhal
            0x0631 => true, // Ra
            0x0632 => true, // Zay
            0x0633 => true, // Sin
            0x0634 => true, // Shin
            0x0635 => true, // Sad
            0x0636 => true, // Dad
            0x0637 => true, // Ta (emphatic)
            0x0638 => true, // Za (emphatic)
            0x0639 => true, // Ayn
            0x063A => true, // Ghayn
            0x0641 => true, // Fa
            0x0642 => true, // Qaf
            0x0643 => true, // Kaf
            0x0644 => true, // Lam
            0x0645 => true, // Mim
            0x0646 => true, // Nun
            0x0647 => true, // Ha (final)
            0x0629 => true, // Ta Marbuta

            // Base letters that can be vowels (treat as consonants by default)
            0x0627 if !self.long_vowels_as_vowels => true, // Alif
            0x0648 if !self.long_vowels_as_vowels => true, // Waw
            0x064A if !self.long_vowels_as_vowels => true, // Ya

            // Urdu extensions
            0x067E => true, // Pe
            0x0686 => true, // Che
            0x0698 => true, // Zhe
            0x06AF => true, // Gaf
            0x0679 => true, // Tte
            0x0688 => true, // Ddal
            0x0691 => true, // Rreh
            0x06BA => true, // Noon Ghunna

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
