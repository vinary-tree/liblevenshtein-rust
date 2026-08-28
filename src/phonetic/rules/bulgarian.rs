//! Bulgarian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Bulgarian (Български език) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Bulgarian phonetic normalization handles:
//! - **Cyrillic script**: Standard Bulgarian Cyrillic alphabet
//! - **Щ = sht**: Unlike Russian (shch), Bulgarian Щ is pronounced "sht"
//! - **Ъ = schwa**: Very common vowel sound, NOT a hard sign like Russian!
//! - **No Ы, Ё, Э**: These Russian letters don't exist in Bulgarian
//!
//! # Bulgarian vs Russian Differences
//!
//! Bulgarian Cyrillic differs from Russian in important ways:
//! - Щ → sht (Russian: shch)
//! - Ъ → schwa vowel (Russian: hard sign, mostly silent)
//! - No Ы (Bulgarian uses И for this sound)
//! - No Ё (Bulgarian uses Е or Йо)
//! - No Э (Bulgarian uses Е)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Bulgarian transliteration rules (~65 rules)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::rules::bulgarian;
//!
//! let rules = bulgarian::base();
//!
//! // Щ becomes sht (not shch like Russian!)
//! let result = rules.apply("щ");
//! assert_eq!(result, "ʃt");
//!
//! // Ъ is a schwa vowel (very common in Bulgarian)
//! let result = rules.apply("ъ");
//! assert_eq!(result, "ə");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Bulgarian phonetic rules.
///
/// Complete phonetic normalization rules for Bulgarian:
///
/// ## Complex Consonants
/// - Щ → sht (Bulgarian-specific! Unlike Russian shch)
/// - Ж → zh (voiced postalveolar fricative)
/// - Ш → sh (voiceless postalveolar fricative)
/// - Х → kh (voiceless velar fricative)
/// - Ц → ts (voiceless alveolar affricate)
/// - Ч → ch (voiceless palatal affricate)
///
/// ## Schwa Vowel (Bulgarian-specific!)
/// - Ъ → a (schwa sound, very common - NOT a hard sign!)
///
/// ## Iotated Vowels
/// - Е → e (just e, not ye like Russian)
/// - Ю → yu
/// - Я → ya
///
/// ## Standard Vowels and Consonants
/// - А, И, О, У → a, i, o, u
/// - Б, В, Г, Д, З, Й, К, Л, М, Н, П, Р, С, Т, Ф → b, v, g, d, z, y, k, l, m, n, p, r, s, t, f
///
/// ## Soft Sign
/// - Ь → (silent, marks palatalization)
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/bulgarian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded bulgarian/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Bulgarian base rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(
            !rules.is_empty(),
            "Bulgarian base rules should not be empty"
        );
        assert!(
            rules.len() >= 40,
            "expected >=40 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_sht_sound() {
        let rules = base();
        // Щ → sht (Bulgarian-specific!)
        let result = rules.apply("щ");
        assert!(
            result.contains("ʃt"),
            "щ should become sht, got: {}",
            result
        );
    }

    #[test]
    fn test_schwa_vowel() {
        let rules = base();
        // Ъ → ə (schwa sound, NOT hard sign!)
        let result = rules.apply("ъ");
        assert!(
            result.contains('ə') || result.contains('a'),
            "ъ should become ə (schwa) or a, got: {}",
            result
        );
    }

    #[test]
    fn test_zh_sound() {
        let rules = base();
        // Ж → zh
        let result = rules.apply("ж");
        assert!(result.contains("ʒ"), "ж should become zh, got: {}", result);
    }

    #[test]
    fn test_sh_sound() {
        let rules = base();
        // Ш → sh
        let result = rules.apply("ш");
        assert!(result.contains("ʃ"), "ш should become sh, got: {}", result);
    }

    #[test]
    fn test_ts_sound() {
        let rules = base();
        // Ц → ts
        let result = rules.apply("ц");
        assert!(result.contains("t͡s"), "ц should become ts, got: {}", result);
    }

    #[test]
    fn test_ch_sound() {
        let rules = base();
        // Ч → ch
        let result = rules.apply("ч");
        assert!(result.contains("t͡ʃ"), "ч should become ch, got: {}", result);
    }

    #[test]
    fn test_kh_sound() {
        let rules = base();
        // Х → kh
        let result = rules.apply("х");
        assert!(result.contains("x"), "х should become kh, got: {}", result);
    }

    #[test]
    fn test_iotated_yu() {
        let rules = base();
        // Ю → yu
        let result = rules.apply("ю");
        assert!(result.contains("ju"), "ю should become yu, got: {}", result);
    }

    #[test]
    fn test_iotated_ya() {
        let rules = base();
        // Я → ya
        let result = rules.apply("я");
        assert!(result.contains("ja"), "я should become ya, got: {}", result);
    }

    #[test]
    fn test_simple_vowel_a() {
        let rules = base();
        // А → a
        let result = rules.apply("а");
        assert!(result.contains('a'), "а should become a, got: {}", result);
    }

    #[test]
    fn test_simple_consonant_b() {
        let rules = base();
        // Б → b
        let result = rules.apply("б");
        assert!(result.contains('b'), "б should become b, got: {}", result);
    }

    #[test]
    fn test_word_bulgaria() {
        let rules = base();
        // България (Bulgaria) - test full word
        let result = rules.apply_full("българия");
        let lower = result.to_lowercase();
        // Note: г -> ɡ (IPA g U+0261), not ASCII 'g'
        assert!(
            lower.contains('b')
                && lower.contains('l')
                && (lower.contains('ɡ') || lower.contains('g')),
            "българия should contain b, l, ɡ/g, got: {}",
            result
        );
    }

    #[test]
    fn test_word_sofia() {
        let rules = base();
        // София (Sofia) - test full word
        let result = rules.apply_full("софия");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('s') && lower.contains('f'),
            "софия should contain s, f, got: {}",
            result
        );
    }

    #[test]
    fn test_soft_sign_silent() {
        let rules = base();
        // Ь → (silent)
        let result = rules.apply("ь");
        assert!(
            result.is_empty(),
            "ь should be silent (empty), got: '{}'",
            result
        );
    }
}
