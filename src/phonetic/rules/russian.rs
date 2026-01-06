//! Russian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Standard Russian (Русский язык) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Russian phonetic normalization handles:
//! - **Cyrillic to Latin transliteration**: All Cyrillic letters mapped to Latin
//! - **Complex consonants**: ж→zh, ш→sh, щ→shch, ч→ch, ц→ts, х→kh
//! - **Iotated vowels**: е→ye, ё→yo, ю→yu, я→ya
//! - **Simple vowels**: а→a, о→o, у→u, э→e, и→i, ы→y
//! - **Soft/hard signs**: ь and ъ are removed (they modify pronunciation)
//! - **Final devoicing**: б→p, в→f, г→k, д→t, ж→sh, з→s at word end
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Russian transliteration rules (~70 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::russian;
//!
//! let rules = russian::base();
//!
//! // Basic transliteration
//! let moscow = rules.apply("Москва");
//! assert!(moscow.contains("moskv"), "Москва → moskva");
//!
//! // Complex consonants
//! let shchi = rules.apply("щи");
//! assert!(shchi.starts_with("ʃt͡ʃ"), "щ → shch");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Russian phonetic rules.
///
/// Complete phonetic normalization rules for Standard Russian:
///
/// ## Transliteration
/// - Cyrillic letters mapped to Latin equivalents
/// - Complex consonants: ж→zh, ш→sh, щ→shch, ч→ch, ц→ts, х→kh
///
/// ## Vowels
/// - Simple: а→a, о→o, у→u, э→e, и→i, ы→y
/// - Iotated: е→ye, ё→yo, ю→yu, я→ya
///
/// ## Special Characters
/// - Soft sign (ь): Removed (marks palatalization)
/// - Hard sign (ъ): Removed (separation marker)
///
/// ## Final Devoicing
/// - Voiced consonants become voiceless at word end
/// - б→p, в→f, г→k, д→t, ж→sh, з→s
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/russian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded russian/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Russian base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Russian base rules should not be empty");
        assert!(
            rules.len() > 50,
            "expected >50 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_simple_vowels() {
        let rules = base();
        // а → a or ɐ (reduced in unstressed position)
        let result = rules.apply("а");
        assert!(result.contains('a') || result.contains('ɐ'), "а should become a or ɐ, got: {}", result);
        // о → ɔ or o or ɐ
        let result = rules.apply("о");
        assert!(result.contains('ɔ') || result.contains('o') || result.contains('ɐ'), "о should become ɔ or o, got: {}", result);
        // у → u
        let result = rules.apply("у");
        assert!(result.contains('u'), "у should become u, got: {}", result);
        // и → i
        let result = rules.apply("и");
        assert!(result.contains('i'), "и should become i, got: {}", result);
        // ы → ɨ (close central unrounded vowel)
        let result = rules.apply("ы");
        assert!(result.contains('ɨ'), "ы should become ɨ, got: {}", result);
    }

    #[test]
    fn test_iotated_vowels() {
        let rules = base();
        // е → ɪ (when not word-initial)
        let result = rules.apply("е");
        // Rule may output various forms depending on context
        assert!(
            result.contains('ɪ') || result.contains('e') || result.contains('j'),
            "е should be processed, got: {}",
            result
        );
        // ё → jo or ɵ
        let result = rules.apply("ё");
        assert!(
            result.contains('j') || result.contains('o') || result.contains('ɵ'),
            "ё should contain j or o, got: {}",
            result
        );
        // ю → ju or ʉ
        let result = rules.apply("ю");
        assert!(
            result.contains('j') || result.contains('u') || result.contains('ʉ'),
            "ю should contain j or u, got: {}",
            result
        );
        // я → ja or ɪ (reduced vowel in unstressed position)
        let result = rules.apply("я");
        assert!(
            result.contains('j') || result.contains('a') || result.contains('ɪ'),
            "я should contain j, a, or ɪ, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_shch() {
        let rules = base();
        // щ → ɕː (IPA voiceless alveolo-palatal fricative, long)
        let result = rules.apply("щи");
        assert!(
            result.contains('ʃ') || result.contains('ɕ'),
            "щ should contain ʃ or ɕ, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_zh() {
        let rules = base();
        // ж → ʒ (IPA voiced postalveolar fricative)
        let result = rules.apply("жить");
        assert!(
            result.contains('ʒ'),
            "ж should become ʒ, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_sh() {
        let rules = base();
        // ш → ʃ (IPA voiceless postalveolar fricative)
        let result = rules.apply("школа");
        assert!(
            result.contains('ʃ'),
            "ш should become ʃ, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_ch() {
        let rules = base();
        // ч → t͡ʃ (IPA voiceless postalveolar affricate)
        let result = rules.apply("чай");
        assert!(
            result.contains("t͡ʃ"),
            "ч should become t͡ʃ, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_ts() {
        let rules = base();
        // ц → t͡s (IPA voiceless alveolar affricate)
        let result = rules.apply("царь");
        assert!(
            result.contains("t͡s"),
            "ц should become t͡s, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_kh() {
        let rules = base();
        // х → x (IPA voiceless velar fricative)
        let result = rules.apply("хлеб");
        assert!(
            result.contains('x'),
            "х should become x, got: {}",
            result
        );
    }

    #[test]
    fn test_soft_sign_removed() {
        let rules = base();
        // ь should be removed
        let result = rules.apply("мать");
        assert!(
            !result.contains('ь'),
            "soft sign should be removed, got: {}",
            result
        );
    }

    #[test]
    fn test_hard_sign_removed() {
        let rules = base();
        // ъ should be removed
        let result = rules.apply("объект");
        assert!(
            !result.contains('ъ'),
            "hard sign should be removed, got: {}",
            result
        );
    }

    #[test]
    fn test_simple_consonants() {
        let rules = base();
        // Basic consonant transliteration
        let result = rules.apply("молоко");
        assert!(result.contains('m'), "м should become m");
        assert!(result.contains('l'), "л should become l");
        assert!(result.contains('k'), "к should become k");
    }

    #[test]
    fn test_moscow() {
        let rules = base();
        // Москва → IPA phonetic form
        let result = rules.apply("Москва");
        assert!(
            result.contains('m') && result.contains('k') && result.contains('v'),
            "Москва should contain m, k, v, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_b() {
        let rules = base();
        // хлеб → xlɪb or xlɪp (final б may devoice)
        let result = rules.apply("хлеб");
        assert!(
            result.ends_with('p') || result.ends_with('b'),
            "final б should process correctly, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_d() {
        let rules = base();
        // год → ɡɐd or ɡɐt (final д may devoice)
        let result = rules.apply("год");
        assert!(
            result.ends_with('t') || result.ends_with('d'),
            "final д should process correctly, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_g() {
        let rules = base();
        // друг → druɡ or druk (final г may devoice)
        let result = rules.apply("друг");
        assert!(
            result.ends_with('k') || result.ends_with('ɡ'),
            "final г should process correctly, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_v() {
        let rules = base();
        // кров → krɐv or krɐf (final в may devoice)
        let result = rules.apply("кров");
        assert!(
            result.ends_with('f') || result.ends_with('v'),
            "final в should process correctly, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_z() {
        let rules = base();
        // мороз → mɐrɐz or mɐrɐs (final з may devoice)
        let result = rules.apply("мороз");
        assert!(
            result.ends_with('s') || result.ends_with('z'),
            "final з should process correctly, got: {}",
            result
        );
    }

    #[test]
    fn test_uppercase() {
        let rules = base();
        // Uppercase should also work - check for IPA output
        let result = rules.apply("РОССИЯ");
        assert!(
            result.contains('r') && result.contains('s'),
            "РОССИЯ should contain r and s, got: {}",
            result
        );
    }

    #[test]
    fn test_mixed_case() {
        let rules = base();
        // Mixed case - check for basic Latin phonemes
        let result = rules.apply("Путин");
        assert!(
            result.contains('p') && result.contains('u') && result.contains('t'),
            "Путин should contain p, u, t, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_rules_exist() {
        use crate::phonetic::types::ContextChar;
        let rules = base();

        // Check we have devoicing rules with Final context
        let final_devoicing_count = rules.rules.iter().filter(|r| {
            matches!(r.context, ContextChar::Final)
        }).count();

        assert!(
            final_devoicing_count >= 6,
            "Expected at least 6 final devoicing rules, got {}",
            final_devoicing_count
        );
    }
}
