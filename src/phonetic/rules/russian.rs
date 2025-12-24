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
//! assert!(shchi.starts_with("shch"), "щ → shch");
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
        // а → a
        assert!(rules.apply("а").contains('a'), "а should become a");
        // о → o
        assert!(rules.apply("о").contains('o'), "о should become o");
        // у → u
        assert!(rules.apply("у").contains('u'), "у should become u");
        // и → i
        assert!(rules.apply("и").contains('i'), "и should become i");
        // ы → y
        assert!(rules.apply("ы").contains('y'), "ы should become y");
    }

    #[test]
    fn test_iotated_vowels() {
        let rules = base();
        // е → ye
        let result = rules.apply("е");
        assert!(
            result.contains("ye"),
            "е should become ye, got: {}",
            result
        );
        // ё → yo
        let result = rules.apply("ё");
        assert!(
            result.contains("yo"),
            "ё should become yo, got: {}",
            result
        );
        // ю → yu
        let result = rules.apply("ю");
        assert!(
            result.contains("yu"),
            "ю should become yu, got: {}",
            result
        );
        // я → ya
        let result = rules.apply("я");
        assert!(
            result.contains("ya"),
            "я should become ya, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_shch() {
        let rules = base();
        // щ → shch
        let result = rules.apply("щи");
        assert!(
            result.starts_with("shch"),
            "щ should become shch, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_zh() {
        let rules = base();
        // ж → zh
        let result = rules.apply("жить");
        assert!(
            result.starts_with("zh"),
            "ж should become zh, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_sh() {
        let rules = base();
        // ш → sh
        let result = rules.apply("школа");
        assert!(
            result.starts_with("sh"),
            "ш should become sh, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_ch() {
        let rules = base();
        // ч → ch
        let result = rules.apply("чай");
        assert!(
            result.starts_with("ch"),
            "ч should become ch, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_ts() {
        let rules = base();
        // ц → ts
        let result = rules.apply("царь");
        assert!(
            result.starts_with("ts"),
            "ц should become ts, got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants_kh() {
        let rules = base();
        // х → kh
        let result = rules.apply("хлеб");
        assert!(
            result.starts_with("kh"),
            "х should become kh, got: {}",
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
        // Москва → moskva
        let result = rules.apply("Москва");
        assert!(
            result.to_lowercase().contains("moskv"),
            "Москва should transliterate to moskva, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_b() {
        let rules = base();
        // хлеб → khlyep (final б → p)
        let result = rules.apply("хлеб");
        assert!(
            result.ends_with('p'),
            "final б should become p, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_d() {
        let rules = base();
        // год → got (final д → t)
        let result = rules.apply("год");
        assert!(
            result.ends_with('t'),
            "final д should become t, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_g() {
        let rules = base();
        // друг → druk (final г → k)
        let result = rules.apply("друг");
        assert!(
            result.ends_with('k'),
            "final г should become k, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_v() {
        let rules = base();
        // кров → krof (final в → f)
        let result = rules.apply("кров");
        assert!(
            result.ends_with('f'),
            "final в should become f, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_z() {
        let rules = base();
        // мороз → moros (final з → s)
        let result = rules.apply("мороз");
        assert!(
            result.ends_with('s'),
            "final з should become s, got: {}",
            result
        );
    }

    #[test]
    fn test_uppercase() {
        let rules = base();
        // Uppercase should also work
        let result = rules.apply("РОССИЯ");
        assert!(
            result.to_lowercase().contains("ros"),
            "РОССИЯ should transliterate, got: {}",
            result
        );
    }

    #[test]
    fn test_mixed_case() {
        let rules = base();
        // Mixed case
        let result = rules.apply("Путин");
        assert!(
            result.to_lowercase().contains("putin"),
            "Путин should become putin, got: {}",
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

        // Check that the rules are sorted by weight
        let weights: Vec<_> = rules.rules.iter().map(|r| r.weight).collect();
        let mut sorted_weights = weights.clone();
        sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(weights, sorted_weights, "Rules should be sorted by weight");
    }
}
