//! Belarusian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Belarusian (Беларуская мова) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Belarusian phonetic normalization handles:
//! - **Unique Ў**: Short u/w sound - ONLY Belarusian has this letter!
//! - **І instead of И**: Like Ukrainian, uses dotted І
//! - **No Щ**: Uses ШЧ digraph instead
//! - **Г = h sound**: Unlike Russian [g], Belarusian Г is [ɦ] (voiced glottal fricative)
//! - **Ё commonly used**: Unlike Russian where it's often omitted
//!
//! # Belarusian vs Russian Differences
//!
//! - Ў (short u/w) - unique to Belarusian
//! - Г → h (not g like Russian)
//! - І (not И)
//! - No Щ (uses ШЧ)
//! - Ё always written (not optional like Russian)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Belarusian transliteration rules (~70 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::belarusian;
//!
//! let rules = belarusian::base();
//!
//! // Ў becomes w (unique Belarusian letter!)
//! let result = rules.apply("ў");
//! assert!(result.contains("w"), "ў → w");
//!
//! // Г becomes h (not g like Russian)
//! let result = rules.apply("г");
//! assert!(result.contains("h"), "г → h");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Belarusian phonetic rules.
///
/// Complete phonetic normalization rules for Belarusian:
///
/// ## Unique Belarusian Letter
/// - Ў → w (short u/w sound - only in Belarusian!)
///
/// ## Complex Consonants
/// - ШЧ → shch (used instead of Щ)
/// - Ж → zh (voiced postalveolar fricative)
/// - Ш → sh (voiceless postalveolar fricative)
/// - Х → kh (voiceless velar fricative)
/// - Ц → ts (voiceless alveolar affricate)
/// - Ч → ch (voiceless palatal affricate)
/// - Г → h (voiced glottal fricative, unlike Russian g!)
///
/// ## Iotated Vowels
/// - Е → ye
/// - Ё → yo
/// - Ю → yu
/// - Я → ya
/// - Э → e
///
/// ## Standard Vowels
/// - А, О, У, Ы, І → a, o, u, y, i
///
/// ## Signs
/// - Ь → (silent, marks palatalization)
/// - ' → (silent, separates consonant from iotated vowel)
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/belarusian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded belarusian/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Belarusian base rules - this is a bug in liblevenshtein")
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
            "Belarusian base rules should not be empty"
        );
        assert!(
            rules.len() >= 40,
            "expected >=40 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_unique_u_short() {
        let rules = base();
        // Ў → w (unique Belarusian letter!)
        let result = rules.apply("ў");
        assert!(result.contains('w'), "ў should become w, got: {}", result);
    }

    #[test]
    fn test_g_to_h() {
        let rules = base();
        // Г → ɦ (voiced glottal fricative, Belarusian-specific! Unlike Russian g)
        let result = rules.apply("г");
        assert!(
            result.contains('h') || result.contains('ɦ'),
            "г should become h or ɦ (voiced glottal fricative) in Belarusian, got: {}",
            result
        );
    }

    #[test]
    fn test_shch_digraph() {
        let rules = base();
        // ШЧ → shch (Belarusian uses this instead of Щ)
        let result = rules.apply("шч");
        assert!(
            result.contains("ʃt͡ʃ"),
            "шч should become shch, got: {}",
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
    fn test_iotated_yo() {
        let rules = base();
        // Ё → yo (commonly used in Belarusian)
        let result = rules.apply("ё");
        assert!(result.contains("jo"), "ё should become yo, got: {}", result);
    }

    #[test]
    fn test_iotated_ye() {
        let rules = base();
        // Е → ye
        let result = rules.apply("е");
        assert!(result.contains("je"), "е should become ye, got: {}", result);
    }

    #[test]
    fn test_dotted_i() {
        let rules = base();
        // І → i (Belarusian uses І instead of И)
        let result = rules.apply("і");
        assert!(result.contains('i'), "і should become i, got: {}", result);
    }

    #[test]
    fn test_y_sound() {
        let rules = base();
        // Ы → ɨ (close central unrounded vowel in IPA)
        let result = rules.apply("ы");
        assert!(
            result.contains('y') || result.contains('ɨ'),
            "ы should become y or ɨ (close central unrounded vowel), got: {}",
            result
        );
    }

    #[test]
    fn test_word_belarus() {
        let rules = base();
        // Беларусь (Belarus) - test full word
        let result = rules.apply_full("беларусь");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('b') && lower.contains('l') && lower.contains('r'),
            "беларусь should contain b, l, r, got: {}",
            result
        );
    }

    #[test]
    fn test_word_minsk() {
        let rules = base();
        // Мінск (Minsk) - test full word
        let result = rules.apply_full("мінск");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('m')
                && lower.contains('n')
                && lower.contains('s')
                && lower.contains('k'),
            "мінск should contain m, n, s, k, got: {}",
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
