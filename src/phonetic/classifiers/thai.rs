//! Thai script vowel classifier.
//!
//! Supports Thai (ภาษาไทย) language.
//! Thai is a Brahmic-derived abugida with unique features:
//! - No spaces between words
//! - 5 tones (not fully marked in writing)
//! - Vowels can appear before, after, above, below, or around consonants

use super::VowelClassifier;

/// Vowel classifier for Thai script.
///
/// Thai has complex vowel positioning:
/// - **Leading vowels**: Appear before the consonant (เ, แ, โ, ใ, ไ)
/// - **Following vowels**: Appear after the consonant (ะ, า, อ)
/// - **Upper vowels**: Appear above the consonant (◌ิ, ◌ี, ◌ึ, ◌ื, ◌ั)
/// - **Lower vowels**: Appear below the consonant (◌ุ, ◌ู)
/// - **Surrounding vowels**: Consonant appears inside (เ◌ะ, แ◌ะ, โ◌ะ, etc.)
///
/// The inherent vowel is /o/ or /a/ depending on syllable structure.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThaiClassifier;

impl ThaiClassifier {
    /// Create a new Thai classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Thai vowels (all forms).
static THAI_VOWELS: &[char] = &[
    // Leading vowels
    'เ', // e (sara e)
    'แ', // ae (sara ae)
    'โ', // o (sara o)
    'ใ', // ai (sara ai mai muan)
    'ไ', // ai (sara ai mai malai)
    // Following vowels
    'ะ', // a short (sara a)
    'า', // aa long (sara aa)
    'ำ', // am (sara am)
    // Upper vowels (combining)
    'ิ',  // i (sara i)
    'ี',  // ii (sara ii)
    'ึ',  // ue (sara ue)
    'ื',  // uee (sara uee)
    'ั',  // mai han akat (short vowel marker)
    '\u{0E47}', // mai tai khu (shortens vowel)
    // Lower vowels (combining)
    'ุ', // u (sara u)
    'ู', // uu (sara uu)
];

impl VowelClassifier for ThaiClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Leading vowels
            0x0E40..=0x0E44 => true, // เ through ไ

            // Following vowels
            0x0E30..=0x0E31 => true, // ะ, ◌ั
            0x0E32..=0x0E33 => true, // า, ำ

            // Upper vowels (above consonant)
            0x0E34..=0x0E39 => true, // ◌ิ through ◌ู

            // Mai tai khu (shortening mark, functions as vowel modifier)
            0x0E47 => true,

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Thai"
    }

    fn vowels(&self) -> &[char] {
        THAI_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Thai consonants (44 characters)
            0x0E01..=0x0E2E => true, // ก through ฮ

            // Obsolete consonants (rare)
            0x0E2F => false, // ฯ (paiyannoi - abbreviation)

            // Tone marks are neither vowels nor consonants
            0x0E48..=0x0E4B => false,

            // Thai currency and other symbols
            0x0E3F => false,

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leading_vowels() {
        let c = ThaiClassifier::new();
        assert!(c.is_vowel('เ')); // sara e
        assert!(c.is_vowel('แ')); // sara ae
        assert!(c.is_vowel('โ')); // sara o
        assert!(c.is_vowel('ใ')); // sara ai mai muan
        assert!(c.is_vowel('ไ')); // sara ai mai malai
    }

    #[test]
    fn test_following_vowels() {
        let c = ThaiClassifier::new();
        assert!(c.is_vowel('ะ')); // sara a (short)
        assert!(c.is_vowel('า')); // sara aa (long)
        assert!(c.is_vowel('ำ')); // sara am
    }

    #[test]
    fn test_upper_vowels() {
        let c = ThaiClassifier::new();
        assert!(c.is_vowel('ิ')); // sara i
        assert!(c.is_vowel('ี')); // sara ii
        assert!(c.is_vowel('ึ')); // sara ue
        assert!(c.is_vowel('ื')); // sara uee
        assert!(c.is_vowel('ั')); // mai han akat
    }

    #[test]
    fn test_lower_vowels() {
        let c = ThaiClassifier::new();
        assert!(c.is_vowel('ุ')); // sara u
        assert!(c.is_vowel('ู')); // sara uu
    }

    #[test]
    fn test_consonants() {
        let c = ThaiClassifier::new();
        assert!(!c.is_vowel('ก')); // ko kai
        assert!(!c.is_vowel('ข')); // kho khai
        assert!(!c.is_vowel('ค')); // kho khwai
        assert!(!c.is_vowel('ง')); // ngo ngu
        assert!(!c.is_vowel('จ')); // cho chan
        assert!(!c.is_vowel('ท')); // tho thahan
        assert!(!c.is_vowel('น')); // no nu
        assert!(!c.is_vowel('ม')); // mo ma
        assert!(!c.is_vowel('ย')); // yo yak
        assert!(!c.is_vowel('ร')); // ro ruea
        assert!(!c.is_vowel('ล')); // lo ling
        assert!(!c.is_vowel('ว')); // wo waen
        assert!(!c.is_vowel('ส')); // so suea
        assert!(!c.is_vowel('ห')); // ho hip
        assert!(!c.is_vowel('อ')); // o ang
        assert!(!c.is_vowel('ฮ')); // ho nokhuk
    }

    #[test]
    fn test_is_consonant() {
        let c = ThaiClassifier::new();
        assert!(c.is_consonant('ก')); // ko kai
        assert!(c.is_consonant('ข')); // kho khai
        assert!(c.is_consonant('ง')); // ngo ngu
        assert!(c.is_consonant('ม')); // mo ma
        assert!(c.is_consonant('น')); // no nu
        assert!(c.is_consonant('ฮ')); // ho nokhuk
    }

    #[test]
    fn test_tone_marks_not_vowel() {
        let c = ThaiClassifier::new();
        assert!(!c.is_vowel('่')); // mai ek
        assert!(!c.is_vowel('้')); // mai tho
        assert!(!c.is_vowel('๊')); // mai tri
        assert!(!c.is_vowel('๋')); // mai chattawa
        assert!(!c.is_consonant('่'));
        assert!(!c.is_consonant('้'));
    }

    #[test]
    fn test_thanthakhat_not_vowel() {
        let c = ThaiClassifier::new();
        // Thanthakhat (cancellation mark) - silences consonant
        assert!(!c.is_vowel('์'));
        assert!(!c.is_consonant('์'));
    }
}
