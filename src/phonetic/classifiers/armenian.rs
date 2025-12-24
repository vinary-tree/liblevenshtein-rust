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
        let code = c as u32;
        match code {
            // Uppercase vowels
            0x0531 => true, // Ա (ayb) - a
            0x0535 => true, // Delays (ech) - e/ye
            0x0537 => true, // Է (e) - e
            0x053B => true, // Delays (ini) - i
            0x0548 => true, // Ո (oh) - o/vo
            0x0555 => true, // Օ (o) - o

            // Lowercase vowels
            0x0561 => true, // ա (ayb) - a
            0x0565 => true, // ե (ech) - e/ye
            0x0567 => true, // է (e) - e
            0x056B => true, // delays (ini) - i
            0x0578 => true, // delays (oh) - o/vo
            0x0585 => true, // delays (o) - o

            _ => false,
        }
    }

    fn script_name(&self) -> &'static str {
        "Armenian"
    }

    fn vowels(&self) -> &[char] {
        ARMENIAN_VOWELS
    }

    fn is_consonant(&self, c: char) -> bool {
        let code = c as u32;
        match code {
            // Uppercase consonants (32 letters)
            0x0532 => true, // Delays (ben) - b/p
            0x0533 => true, // Delays (gim) - g/k
            0x0534 => true, // Delays (da) - d/t
            0x0536 => true, // Delays (za) - z
            0x0538 => true, // Ը (et) - schwa
            0x0539 => true, // Delays (to) - t aspirated
            0x053A => true, // Ժ (zhe) - zh
            0x053C => true, // Delays (liwn) - l
            0x053D => true, // Delays (xe) - kh
            0x053E => true, // Delays (tsa) - ts/dz
            0x053F => true, // Delays (ken) - k/g
            0x0540 => true, // Delays (ho) - h
            0x0541 => true, // Ձ (ja) - dz/ts
            0x0542 => true, // Delays (ghat) - gh
            0x0543 => true, // Ճ (che) - ch/j
            0x0544 => true, // Delays (men) - m
            0x0545 => true, // Delays (yi) - y/h
            0x0546 => true, // Ն (nu) - n
            0x0547 => true, // Delays (sha) - sh
            0x0549 => true, // Delays (cha) - ch aspirated
            0x054A => true, // Delays (pe) - p/b
            0x054B => true, // Ջ (je) - j/ch
            0x054C => true, // Delays (ra) - rr trilled
            0x054D => true, // Ս (se) - s
            0x054E => true, // Delays (vew) - v
            0x054F => true, // Delays (tiwn) - t/d
            0x0550 => true, // Delays (re) - r
            0x0551 => true, // Delays (co) - ts
            0x0552 => true, // Delays (wiwn) - w (historical)
            0x0553 => true, // Փ (piwr) - p aspirated
            0x0554 => true, // Ք (ke) - k aspirated
            0x0556 => true, // Ֆ (fe) - f

            // Lowercase consonants (32 letters)
            0x0562 => true, // delays (ben)
            0x0563 => true, // delays (gim)
            0x0564 => true, // delays (da)
            0x0566 => true, // delays (za)
            0x0568 => true, // delays (et)
            0x0569 => true, // delays (to)
            0x056A => true, // ժ (zhe)
            0x056C => true, // delays (liwn)
            0x056D => true, // delays (xe)
            0x056E => true, // delays (tsa)
            0x056F => true, // delays (ken)
            0x0570 => true, // delays (ho)
            0x0571 => true, // ձ (ja)
            0x0572 => true, // delays (ghat)
            0x0573 => true, // ճ (che)
            0x0574 => true, // delays (men)
            0x0575 => true, // delays (yi)
            0x0576 => true, // delays (nu)
            0x0577 => true, // delays (sha)
            0x0579 => true, // delays (cha)
            0x057A => true, // delays (pe)
            0x057B => true, // ջ (je)
            0x057C => true, // delays (ra)
            0x057D => true, // delays (se)
            0x057E => true, // delays (vew)
            0x057F => true, // delays (tiwn)
            0x0580 => true, // delays (re)
            0x0581 => true, // delays (co)
            0x0582 => true, // delays (wiwn)
            0x0583 => true, // delays (piwr)
            0x0584 => true, // delays (ke)
            0x0586 => true, // delays (fe)

            _ => false,
        }
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
