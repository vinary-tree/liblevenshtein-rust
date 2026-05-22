//! Thai phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Thai (ภาษาไทย) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Thai phonetic normalization handles:
//! - **Thai script**: Brahmic-derived abugida with unique features
//! - **No word spacing**: Thai doesn't use spaces between words
//! - **44 consonants**: Divided into high, mid, and low classes
//! - **Complex vowel positioning**: Before, after, above, below, or around consonants
//! - **5 tones**: Determined by consonant class and tone marks
//!
//! # Thai Consonant Classes
//!
//! Thai consonants are divided into 3 classes that affect tone:
//! - **High class** (อักษรสูง): ข, ฃ, ฉ, ฐ, ถ, ผ, ฝ, ศ, ษ, ส, ห
//! - **Mid class** (อักษรกลาง): ก, จ, ฎ, ฏ, ด, ต, บ, ป, อ
//! - **Low class** (อักษรต่ำ): All remaining consonants
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers for Thai sounds:
//! - AE = sara ae (แ)
//! - AI = sara ai (ใ, ไ)
//! - AM = sara am (ำ)
//! - UE = sara ue (ึ)
//! - UEE = sara uee (ื)
//! - M = nikhahit nasalization (ํ)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Thai phonetic rules (~85 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::thai;
//!
//! let rules = thai::base();
//!
//! // Thai consonants
//! let result = rules.apply("ก");
//! assert!(result.contains('k'), "ก → k");
//!
//! // Leading vowel (appears before consonant)
//! let result = rules.apply("เ");
//! assert!(result.contains('e'), "เ → e");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Thai base phonetic rules.
///
/// Complete phonetic normalization rules for Thai:
///
/// ## Leading Vowels (weight 0.02)
/// - เ→e, แ→AE, โ→o, ใ→AI, ไ→AI
///
/// ## Following Vowels (weight 0.05)
/// - ะ→a, า→A, ำ→AM
///
/// ## Upper Vowels (weight 0.05)
/// - ิ→i, ี→I, ึ→UE, ื→UEE, ั→a
///
/// ## Lower Vowels (weight 0.05)
/// - ุ→u, ู→U
///
/// ## High Class Consonants (weight 0.05)
/// - ข→kh, ฃ→kh, ฉ→ch, ฐ→th, ถ→th, ผ→ph, ฝ→f
/// - ศ→s, ษ→s, ส→s, ห→h
///
/// ## Mid Class Consonants (weight 0.05)
/// - ก→k, จ→c, ฎ→d, ฏ→t, ด→d, ต→t, บ→b, ป→p, อ→'
///
/// ## Low Class Consonants (weight 0.05)
/// - ง→ng, ญ→y, ณ→n, น→n, ม→m, ย→y, ร→r, ล→l, ว→w
/// - ค→kh, ฅ→kh, ฆ→kh, ช→ch, ฌ→ch, ฑ→th, ฒ→th, ท→th, ธ→th
/// - พ→ph, ฟ→f, ภ→ph, ฬ→l, ฮ→h
///
/// ## Tone Marks (weight 0.1)
/// - ่, ้, ๊, ๋ → ∅ (silent in romanization)
///
/// ## Numerals (weight 0.1)
/// - Thai digits: ๐-๙ → 0-9
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/thai/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded thai/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Thai base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Thai base rules should not be empty");
        assert!(
            rules.len() >= 70,
            "expected >=70 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // LEADING VOWEL TESTS
    // ============================================================

    #[test]
    fn test_leading_vowels() {
        let rules = base();
        let result = rules.apply("เ");
        assert!(result.contains('e'), "เ should become e, got: {}", result);
        let result = rules.apply("แ");
        assert!(result.contains("æ"), "แ should become ae, got: {}", result);
        let result = rules.apply("โ");
        assert!(result.contains('o'), "โ should become o, got: {}", result);
        let result = rules.apply("ใ");
        assert!(result.contains("aɪ"), "ใ should become ai, got: {}", result);
        let result = rules.apply("ไ");
        assert!(result.contains("aɪ"), "ไ should become ai, got: {}", result);
    }

    // ============================================================
    // FOLLOWING VOWEL TESTS
    // ============================================================

    #[test]
    fn test_following_vowels() {
        let rules = base();
        let result = rules.apply("ะ");
        assert!(result.contains('a'), "ะ should become a, got: {}", result);
        let result = rules.apply("า");
        assert!(result.contains("aː"), "า should become A, got: {}", result);
        let result = rules.apply("ำ");
        assert!(result.contains("am"), "ำ should become AM, got: {}", result);
    }

    // ============================================================
    // UPPER/LOWER VOWEL TESTS
    // ============================================================

    #[test]
    fn test_upper_vowels() {
        let rules = base();
        let result = rules.apply("ิ");
        assert!(result.contains('i'), "ิ should become i, got: {}", result);
        let result = rules.apply("ี");
        assert!(
            result.contains("iː") || result.contains('i'),
            "ี should become iː, got: {}",
            result
        );
        let result = rules.apply("ึ");
        assert!(result.contains("ɯ"), "ึ should become ɯ, got: {}", result);
    }

    #[test]
    fn test_lower_vowels() {
        let rules = base();
        let result = rules.apply("ุ");
        assert!(result.contains('u'), "ุ should become u, got: {}", result);
        let result = rules.apply("ู");
        assert!(
            result.contains("uː") || result.contains('u'),
            "ู should become uː, got: {}",
            result
        );
    }

    // ============================================================
    // CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_mid_class_consonants() {
        let rules = base();
        let result = rules.apply("ก");
        assert!(result.contains('k'), "ก should become k, got: {}", result);
        let result = rules.apply("จ");
        assert!(result.contains('c'), "จ should become c, got: {}", result);
        let result = rules.apply("บ");
        assert!(result.contains('b'), "บ should become b, got: {}", result);
        let result = rules.apply("ป");
        assert!(result.contains('p'), "ป should become p, got: {}", result);
    }

    #[test]
    fn test_high_class_consonants() {
        let rules = base();
        let result = rules.apply("ข");
        assert!(
            result.contains("kh") || result.contains('k'),
            "ข should become kh, got: {}",
            result
        );
        let result = rules.apply("ส");
        assert!(result.contains('s'), "ส should become s, got: {}", result);
        let result = rules.apply("ห");
        assert!(result.contains('h'), "ห should become h, got: {}", result);
    }

    #[test]
    fn test_low_class_consonants() {
        let rules = base();
        let result = rules.apply("ง");
        assert!(result.contains("ŋ"), "ง should become ng, got: {}", result);
        let result = rules.apply("ม");
        assert!(result.contains('m'), "ม should become m, got: {}", result);
        let result = rules.apply("ร");
        assert!(result.contains('r'), "ร should become r, got: {}", result);
        let result = rules.apply("ล");
        assert!(result.contains('l'), "ล should become l, got: {}", result);
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_thai() {
        let rules = base();
        // ไทย (Thai)
        let result = rules.apply_full("ไทย");
        // ไ→AI, ท→th, ย→y
        assert!(
            result.contains("aɪ") && result.contains('y'),
            "ไทย should contain AI, y, got: {}",
            result
        );
    }

    #[test]
    fn test_word_bangkok() {
        let rules = base();
        // กรุงเทพ (Bangkok short form)
        let result = rules.apply_full("กรุงเทพ");
        // ก→k, ร→r, ุ→u, ง→ng, เ→e, ท→th, พ→ph
        assert!(
            result.contains('k') && result.contains('r'),
            "กรุงเทพ should contain k, r, got: {}",
            result
        );
    }

    #[test]
    fn test_word_sawatdi() {
        let rules = base();
        // สวัสดี (Hello)
        let result = rules.apply_full("สวัสดี");
        // ส→s, ว→w, ั→a, ส→s, ด→d, ี→I
        assert!(
            result.contains('s') && result.contains('w') && result.contains('d'),
            "สวัสดี should contain s, w, d, got: {}",
            result
        );
    }

    // ============================================================
    // NUMERAL TESTS
    // ============================================================

    #[test]
    fn test_numerals() {
        let rules = base();
        let result = rules.apply("๐");
        assert!(result.contains('0'), "๐ should become 0, got: {}", result);
        let result = rules.apply("๕");
        assert!(result.contains('5'), "๕ should become 5, got: {}", result);
        let result = rules.apply("๙");
        assert!(result.contains('9'), "๙ should become 9, got: {}", result);
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================
}
