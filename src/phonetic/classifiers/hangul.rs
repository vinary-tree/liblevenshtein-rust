//! Korean Hangul vowel classifier.
//!
//! Supports Korean jamo (vowel) components in Hangul syllable blocks.

use super::VowelClassifier;

/// Vowel classifier for Korean Hangul script.
///
/// Hangul is an alphabetic script where syllable blocks are composed of:
/// - Initial consonant (choseong)
/// - Vowel (jungseong) - the nucleus
/// - Optional final consonant (jongseong)
///
/// This classifier identifies:
/// - Hangul Jamo vowels (U+1161-U+1175, U+11A8-U+11C2)
/// - Hangul Compatibility Jamo vowels (U+314F-U+3163)
/// - Vowel components within syllable blocks (U+AC00-U+D7A3)
#[derive(Debug, Clone, Copy, Default)]
pub struct HangulClassifier;

impl HangulClassifier {
    /// Create a new Hangul classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Static list of Hangul jamo vowels.
static HANGUL_VOWELS: &[char] = &[
    // Basic vowels (monophthongs)
    'ㅏ', // a
    'ㅓ', // eo
    'ㅗ', // o
    'ㅜ', // u
    'ㅡ', // eu
    'ㅣ', // i
    'ㅐ', // ae
    'ㅔ', // e
    // Y-vowels (iotized)
    'ㅑ', // ya
    'ㅕ', // yeo
    'ㅛ', // yo
    'ㅠ', // yu
    'ㅒ', // yae
    'ㅖ', // ye
    // Compound vowels (diphthongs)
    'ㅘ', // wa
    'ㅙ', // wae
    'ㅚ', // oe
    'ㅝ', // wo
    'ㅞ', // we
    'ㅟ', // wi
    'ㅢ', // ui
];

impl VowelClassifier for HangulClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Hangul Compatibility Jamo vowels (ㅏ-ㅣ)
            0x314F..=0x3163 => true,

            // Hangul Jamo vowels (medial vowels)
            0x1161..=0x1175 => true,

            // Hangul Jamo Extended-A (medial vowels)
            0xA960..=0xA97C => false, // These are initial consonants

            // Hangul Jamo Extended-B (some vowels)
            0xD7B0..=0xD7C6 => true,

            // For complete syllable blocks, we can't easily tell
            // if it "contains" a vowel - the block itself is a syllable
            // with an inherent vowel. Return false for syllable blocks.
            0xAC00..=0xD7A3 => false,

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Hangul"
    }

    fn vowels(&self) -> &[char] {
        HANGUL_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Hangul Compatibility Jamo consonants (ㄱ-ㅎ)
            0x3131..=0x314E => true,

            // Hangul Jamo initial consonants
            0x1100..=0x1112 => true,

            // Hangul Jamo final consonants
            0x11A8..=0x11C2 => true,

            // Syllable blocks are neither pure vowel nor consonant
            0xAC00..=0xD7A3 => false,

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_vowels() {
        let c = HangulClassifier::new();
        assert!(c.is_vowel('ㅏ')); // a
        assert!(c.is_vowel('ㅓ')); // eo
        assert!(c.is_vowel('ㅗ')); // o
        assert!(c.is_vowel('ㅜ')); // u
        assert!(c.is_vowel('ㅡ')); // eu
        assert!(c.is_vowel('ㅣ')); // i
    }

    #[test]
    fn test_iotized_vowels() {
        let c = HangulClassifier::new();
        assert!(c.is_vowel('ㅑ')); // ya
        assert!(c.is_vowel('ㅕ')); // yeo
        assert!(c.is_vowel('ㅛ')); // yo
        assert!(c.is_vowel('ㅠ')); // yu
    }

    #[test]
    fn test_compound_vowels() {
        let c = HangulClassifier::new();
        assert!(c.is_vowel('ㅘ')); // wa
        assert!(c.is_vowel('ㅙ')); // wae
        assert!(c.is_vowel('ㅚ')); // oe
        assert!(c.is_vowel('ㅝ')); // wo
        assert!(c.is_vowel('ㅞ')); // we
        assert!(c.is_vowel('ㅟ')); // wi
        assert!(c.is_vowel('ㅢ')); // ui
    }

    #[test]
    fn test_consonants() {
        let c = HangulClassifier::new();
        assert!(!c.is_vowel('ㄱ')); // g/k
        assert!(!c.is_vowel('ㄴ')); // n
        assert!(!c.is_vowel('ㄷ')); // d/t
        assert!(!c.is_vowel('ㄹ')); // r/l
        assert!(!c.is_vowel('ㅁ')); // m
        assert!(!c.is_vowel('ㅂ')); // b/p
        assert!(!c.is_vowel('ㅅ')); // s
        assert!(!c.is_vowel('ㅎ')); // h
    }

    #[test]
    fn test_double_consonants() {
        let c = HangulClassifier::new();
        assert!(!c.is_vowel('ㄲ')); // kk
        assert!(!c.is_vowel('ㄸ')); // tt
        assert!(!c.is_vowel('ㅃ')); // pp
        assert!(!c.is_vowel('ㅆ')); // ss
        assert!(!c.is_vowel('ㅉ')); // jj
    }

    #[test]
    fn test_syllable_blocks() {
        let c = HangulClassifier::new();
        // Complete syllable blocks are not classified as vowels
        // because they contain both consonant and vowel components
        assert!(!c.is_vowel('가')); // ga
        assert!(!c.is_vowel('나')); // na
        assert!(!c.is_vowel('한')); // han
        assert!(!c.is_vowel('글')); // geul
    }
}
