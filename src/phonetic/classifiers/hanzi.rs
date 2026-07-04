//! Chinese Hanzi classifier.
//!
//! Chinese characters (Hanzi) are logographic - each character represents
//! a morpheme rather than a phoneme. The concept of "vowel" doesn't directly
//! apply to individual characters.

use super::VowelClassifier;

/// Classifier for Chinese Hanzi characters.
///
/// Chinese is logographic, so the vowel/consonant distinction doesn't apply
/// to individual characters. However, when characters are converted to
/// phonetic representations (Pinyin), the resulting syllables have vowels.
///
/// This classifier:
/// - Treats all CJK ideographs as "consonants" (non-vowels) since they're
///   complete syllables or morphemes, not phonemes
/// - Returns `true` for Pinyin vowel letters when encountered
///
/// For phonetic matching, Chinese text should first be converted to Pinyin,
/// then the Latin classifier can be used on the result.
#[derive(Debug, Clone, Copy, Default)]
pub struct HanziClassifier;

impl HanziClassifier {
    /// Create a new Hanzi classifier.
    pub fn new() -> Self {
        Self
    }
}

// Keep Han block membership centralized so vowel and consonant contexts cannot drift.
#[inline]
fn is_cjk_ideograph(c: char) -> bool {
    matches!(
        c as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x2CEB0..=0x2EBEF
            | 0x2EBF0..=0x2EE5F
            | 0x2F800..=0x2FA1F
            | 0x30000..=0x3134F
            | 0x31350..=0x323AF
            | 0x323B0..=0x3347F
    )
}

#[inline]
fn is_pinyin_vowel(c: char) -> bool {
    matches!(
        c,
        // If we encounter Latin vowels (from Pinyin), classify them correctly.
        'A' | 'E' | 'I' | 'O' | 'U' | 'a' | 'e' | 'i' | 'o' | 'u'
            // Pinyin vowels with tone marks.
            | '\u{0101}'
            | '\u{00E1}'
            | '\u{01CE}'
            | '\u{00E0}'
            | '\u{0113}'
            | '\u{00E9}'
            | '\u{011B}'
            | '\u{00E8}'
            | '\u{012B}'
            | '\u{00ED}'
            | '\u{01D0}'
            | '\u{00EC}'
            | '\u{014D}'
            | '\u{00F3}'
            | '\u{01D2}'
            | '\u{00F2}'
            | '\u{016B}'
            | '\u{00FA}'
            | '\u{01D4}'
            | '\u{00F9}'
            | '\u{01D6}'
            | '\u{01D8}'
            | '\u{01DA}'
            | '\u{01DC}'
            | '\u{00FC}'
    )
}

#[inline]
fn is_ascii_pinyin_consonant(c: char) -> bool {
    c.is_ascii_alphabetic()
        && !matches!(c, 'A' | 'E' | 'I' | 'O' | 'U' | 'a' | 'e' | 'i' | 'o' | 'u')
}

impl VowelClassifier for HanziClassifier {
    fn is_vowel(&self, c: char) -> bool {
        is_pinyin_vowel(c)
    }

    fn script_name(&self) -> &'static str {
        "Hanzi"
    }

    fn is_consonant(&self, c: char) -> bool {
        is_ascii_pinyin_consonant(c) || is_cjk_ideograph(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cjk_ideographs_not_vowels() {
        let c = HanziClassifier::new();
        assert!(!c.is_vowel('我')); // wo (I/me)
        assert!(!c.is_vowel('你')); // ni (you)
        assert!(!c.is_vowel('好')); // hao (good)
        assert!(!c.is_vowel('中')); // zhong (middle)
        assert!(!c.is_vowel('国')); // guo (country)
    }

    #[test]
    fn test_pinyin_vowels() {
        let c = HanziClassifier::new();
        assert!(c.is_vowel('a'));
        assert!(c.is_vowel('e'));
        assert!(c.is_vowel('i'));
        assert!(c.is_vowel('o'));
        assert!(c.is_vowel('u'));
    }

    #[test]
    fn test_pinyin_tone_vowels() {
        let c = HanziClassifier::new();
        assert!(c.is_vowel('ā')); // a with macron (1st tone)
        assert!(c.is_vowel('á')); // a with acute (2nd tone)
        assert!(c.is_vowel('ǎ')); // a with caron (3rd tone)
        assert!(c.is_vowel('à')); // a with grave (4th tone)
        assert!(c.is_vowel('ü')); // u with umlaut
    }

    #[test]
    fn test_pinyin_consonants() {
        let c = HanziClassifier::new();
        assert!(c.is_consonant('b'));
        assert!(c.is_consonant('p'));
        assert!(c.is_consonant('m'));
        assert!(c.is_consonant('f'));
        assert!(c.is_consonant('z'));
        assert!(c.is_consonant('c'));
        assert!(c.is_consonant('s'));
    }

    #[test]
    fn test_cjk_as_consonant() {
        let c = HanziClassifier::new();
        // CJK ideographs treated as consonants for rule context purposes
        assert!(c.is_consonant('我'));
        assert!(c.is_consonant('你'));
        assert!(c.is_consonant('他'));
    }

    #[test]
    fn test_cjk_extensions_as_consonants() {
        let c = HanziClassifier::new();
        assert!(c.is_consonant('\u{2A700}')); // Extension C
        assert!(c.is_consonant('\u{2EBF0}')); // Extension I
        assert!(c.is_consonant('\u{31350}')); // Extension H
        assert!(c.is_consonant('\u{323B0}')); // Extension J
        assert!(c.is_consonant('\u{F900}')); // Compatibility Ideographs
        assert!(c.is_consonant('\u{2F800}')); // Compatibility Ideographs Supplement
    }
}
