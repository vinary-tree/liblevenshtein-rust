//! Armenian script vowel classifier.
//!
//! Supports Armenian (Հdelays երdelays) language.
//! Armenian uses a unique alphabet created in 405 AD with:
//! - 39 letters (originally 36, 3 added in Middle Ages)
//! - Uppercase and lowercase forms
//! - 7 vowels and 31 consonants (+ 1 combining mark)
//! - Aspirated consonant series
//! - Two major dialect pronunciations (Eastern vs Western Armenian)

use super::VowelClassifier;

/// Vowel classifier for Armenian script.
///
/// Armenian has 7 vowels:
/// - ** Delays ա (ayb)**: /a/
/// - **Delays delays (ech)**: /e/ (word-initial) or /ye/ (Eastern)
/// - **Է է (e)**: /e/ (schwa in some dialects)
/// - **Delays delays (ini)**: /i/
/// - **Delays delays (oh)**: /o/ (word-initial) or /vo/ (Eastern)
/// - **Delays delays (u)**: /u/ (digraph delays delays)
/// - **Delays delays (yiwn)**: /y/ (historically) or /i/ (modern)
///
/// The difference between Eastern and Western Armenian is significant:
/// - Eastern: Official in Armenia and Iran
/// - Western: Diaspora Armenian (historically Ottoman Empire)
#[derive(Debug, Clone, Copy, Default)]
pub struct ArmenianClassifier;

impl ArmenianClassifier {
    /// Create a new Armenian classifier.
    pub fn new() -> Self {
        Self
    }
}

/// Armenian vowels (uppercase and lowercase).
static ARMENIAN_VOWELS: &[char] = &[
    // Uppercase vowels
    '\u{0531}', // Ա Ayb (A)
    '\u{0535}', // Delays Ech (E/Ye)
    '\u{0537}', // Է E (E)
    '\u{053B}', // Delays Ini (I)
    '\u{0548}', // Delays Oh (O/Vo)
    '\u{0555}', // Օ Yiwn (O)
    // Lowercase vowels
    '\u{0561}', // ա ayb (a)
    '\u{0565}', // delays ech (e/ye)
    '\u{0567}', // է e (e)
    '\u{056B}', // delays ini (i)
    '\u{0578}', // delays oh (o/vo)
    '\u{0585}', // delays yiwn (o)
];

impl VowelClassifier for ArmenianClassifier {
    fn is_vowel(&self, c: char) -> bool {
        matches!(
            c,
            '\u{0531}'
                | '\u{0535}'
                | '\u{0537}'
                | '\u{053B}'
                | '\u{0548}'
                | '\u{0555}'
                | '\u{0561}'
                | '\u{0565}'
                | '\u{0567}'
                | '\u{056B}'
                | '\u{0578}'
                | '\u{0585}'
        )
    }

    fn script_name(&self) -> &'static str {
        "Armenian"
    }

    fn vowels(&self) -> &[char] {
        ARMENIAN_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        ((0x0531..=0x0556).contains(&code) || (0x0561..=0x0586).contains(&code))
            && !self.is_vowel(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uppercase_vowels() {
        let c = ArmenianClassifier::new();
        assert!(c.is_vowel('\u{0531}')); // ayb - A
        assert!(c.is_vowel('\u{0535}')); // ech - E
        assert!(c.is_vowel('\u{0537}')); // e - E
        assert!(c.is_vowel('\u{053B}')); // ini - I
        assert!(c.is_vowel('\u{0548}')); // oh - O
        assert!(c.is_vowel('\u{0555}')); // o - O
    }

    #[test]
    fn test_lowercase_vowels() {
        let c = ArmenianClassifier::new();
        assert!(c.is_vowel('\u{0561}')); // ayb - a
        assert!(c.is_vowel('\u{0565}')); // ech - e
        assert!(c.is_vowel('\u{0567}')); // e - e
        assert!(c.is_vowel('\u{056B}')); // ini - i
        assert!(c.is_vowel('\u{0578}')); // oh - o
        assert!(c.is_vowel('\u{0585}')); // o - o
    }

    #[test]
    fn test_consonants_not_vowels() {
        let c = ArmenianClassifier::new();
        assert!(!c.is_vowel('\u{0532}')); // ben - B
        assert!(!c.is_vowel('\u{0533}')); // gim - G
        assert!(!c.is_vowel('\u{0534}')); // da - D
        assert!(!c.is_vowel('\u{0544}')); // men - M
        assert!(!c.is_vowel('\u{0546}')); // nu - N
        assert!(!c.is_vowel('\u{054D}')); // se - S
    }

    #[test]
    fn test_is_consonant_uppercase() {
        let c = ArmenianClassifier::new();
        assert!(c.is_consonant('\u{0532}')); // ben - B
        assert!(c.is_consonant('\u{0533}')); // gim - G
        assert!(c.is_consonant('\u{0534}')); // da - D
        assert!(c.is_consonant('\u{053C}')); // liwn - L
        assert!(c.is_consonant('\u{0544}')); // men - M
        assert!(c.is_consonant('\u{0546}')); // nu - N
        assert!(c.is_consonant('\u{0547}')); // sha - SH
        assert!(c.is_consonant('\u{054D}')); // se - S
        assert!(c.is_consonant('\u{054C}')); // ra - RR (trilled)
        assert!(c.is_consonant('\u{0556}')); // fe - F
    }

    #[test]
    fn test_is_consonant_lowercase() {
        let c = ArmenianClassifier::new();
        assert!(c.is_consonant('\u{0562}')); // ben - b
        assert!(c.is_consonant('\u{0563}')); // gim - g
        assert!(c.is_consonant('\u{0564}')); // da - d
        assert!(c.is_consonant('\u{056C}')); // liwn - l
        assert!(c.is_consonant('\u{0574}')); // men - m
        assert!(c.is_consonant('\u{0576}')); // nu - n
        assert!(c.is_consonant('\u{0577}')); // sha - sh
        assert!(c.is_consonant('\u{057D}')); // se - s
    }

    #[test]
    fn test_aspirated_consonants() {
        let c = ArmenianClassifier::new();
        // Aspirated consonants in Armenian
        assert!(c.is_consonant('\u{0539}')); // to - aspirated t
        assert!(c.is_consonant('\u{0553}')); // piwr - aspirated p
        assert!(c.is_consonant('\u{0554}')); // ke - aspirated k
        assert!(c.is_consonant('\u{0549}')); // cha - aspirated ch
    }

    #[test]
    fn test_vowels_not_consonants() {
        let c = ArmenianClassifier::new();
        assert!(!c.is_consonant('\u{0531}')); // ayb - A
        assert!(!c.is_consonant('\u{0535}')); // ech - E
        assert!(!c.is_consonant('\u{0561}')); // ayb - a
        assert!(!c.is_consonant('\u{0565}')); // ech - e
        assert!(!c.is_consonant('\u{056B}')); // ini - i
    }

    #[test]
    fn test_script_name() {
        let c = ArmenianClassifier::new();
        assert_eq!(c.script_name(), "Armenian");
    }
}
