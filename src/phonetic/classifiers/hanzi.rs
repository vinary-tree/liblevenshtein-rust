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

impl VowelClassifier for HanziClassifier {
    fn is_vowel(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // If we encounter Latin vowels (from Pinyin), classify them correctly
            0x0041 | 0x0045 | 0x0049 | 0x004F | 0x0055 => true, // A, E, I, O, U
            0x0061 | 0x0065 | 0x0069 | 0x006F | 0x0075 => true, // a, e, i, o, u

            // Pinyin vowels with tone marks
            0x0101 | 0x00E1 | 0x01CE | 0x00E0 => true, // ā á ǎ à
            0x0113 | 0x00E9 | 0x011B | 0x00E8 => true, // ē é ě è
            0x012B | 0x00ED | 0x01D0 | 0x00EC => true, // ī í ǐ ì
            0x014D | 0x00F3 | 0x01D2 | 0x00F2 => true, // ō ó ǒ ò
            0x016B | 0x00FA | 0x01D4 | 0x00F9 => true, // ū ú ǔ ù
            0x01D6 | 0x01D8 | 0x01DA | 0x01DC | 0x00FC => true, // ǖ ǘ ǚ ǜ ü

            // CJK ideographs are not vowels (they're complete syllables)
            0x4E00..=0x9FFF => false,   // CJK Unified Ideographs
            0x3400..=0x4DBF => false,   // CJK Extension A
            0x20000..=0x2A6DF => false, // CJK Extension B
            0x2A700..=0x2B73F => false, // CJK Extension C
            0x2B740..=0x2B81F => false, // CJK Extension D
            0x2B820..=0x2CEAF => false, // CJK Extension E
            0x2CEB0..=0x2EBEF => false, // CJK Extension F
            0x30000..=0x3134F => false, // CJK Extension G
            0xF900..=0xFAFF => false,   // CJK Compatibility Ideographs

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Hanzi"
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Latin consonants (from Pinyin)
            0x0042..=0x0044
            | 0x0046..=0x0048
            | 0x004A..=0x004E
            | 0x0050..=0x0054
            | 0x0056..=0x005A => true, // B-D, F-H, J-N, P-T, V-Z
            0x0062..=0x0064
            | 0x0066..=0x0068
            | 0x006A..=0x006E
            | 0x0070..=0x0074
            | 0x0076..=0x007A => true, // b-d, f-h, j-n, p-t, v-z

            // CJK ideographs are treated as "consonants" (non-vowels)
            // for the purpose of phonetic rule contexts
            0x4E00..=0x9FFF => true,   // CJK Unified Ideographs
            0x3400..=0x4DBF => true,   // CJK Extension A
            0x20000..=0x2A6DF => true, // CJK Extension B

            _ => false,
        }
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
}
