//! Ukrainian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Ukrainian (Українська мова) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Ukrainian phonetic normalization handles:
//! - **Unique Ukrainian letters**: ї→yi, є→ye, і→i, ґ→g
//! - **Complex consonants**: щ→shch, ж→zh, ш→sh, х→kh, ц→ts, ч→ch
//! - **Iotated vowels**: ю→yu, я→ya
//! - **Simple vowels**: а→a, е→e, и→y, о→o, у→u
//! - **Consonants**: Standard Cyrillic consonant mappings
//! - **Special г→h**: Ukrainian г is [ɦ] (voiced glottal), not [g]
//! - **Soft sign removal**: ь removed (palatalization marker)
//! - **Final devoicing**: б→p, в→f, г→kh, д→t at word end
//!
//! # Differences from Russian
//!
//! - No ы, ё, э (Russian letters not used in Ukrainian)
//! - Ukrainian и sounds like Russian ы (→ y)
//! - Ukrainian г = [ɦ] (→ h), while ґ = [g] (→ g)
//! - Ukrainian е is not iotated (→ e, not ye)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Ukrainian transliteration rules (~55 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::ukrainian;
//!
//! let rules = ukrainian::base();
//!
//! // Unique Ukrainian letters
//! let result = rules.apply("ї");
//! assert!(result.contains("yi"), "ї → yi");
//!
//! // Ukrainian г → h (not g!)
//! let result = rules.apply("г");
//! assert!(result.contains("h"), "г → h");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Ukrainian phonetic rules.
///
/// Complete phonetic normalization rules for Ukrainian Cyrillic:
///
/// ## Unique Ukrainian Letters
/// - ї → yi, є → ye, і → i, ґ → g
///
/// ## Complex Consonants
/// - щ → shch, ж → zh, ш → sh, х → kh, ц → ts, ч → ch
///
/// ## Iotated Vowels
/// - ю → yu, я → ya
///
/// ## Simple Vowels
/// - а → a, е → e, и → y (like Russian ы!), о → o, у → u
///
/// ## Consonants
/// - г → h (voiced glottal, NOT plosive g)
/// - б → b, в → v, д → d, з → z, й → y, к → k, л → l
/// - м → m, н → n, п → p, р → r, с → s, т → t, ф → f
///
/// ## Final Devoicing
/// - б → p, в → f, г → kh, ґ → k, д → t, ж → sh, з → s at word end
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/ukrainian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded ukrainian/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Ukrainian base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Ukrainian base rules should not be empty");
        assert!(
            rules.len() >= 40,
            "expected >=40 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_unique_yi() {
        let rules = base();
        // ї → yi
        let result = rules.apply("ї");
        assert!(
            result.contains("yi"),
            "ї should become yi, got: {}",
            result
        );
    }

    #[test]
    fn test_unique_ye() {
        let rules = base();
        // є → ye
        let result = rules.apply("є");
        assert!(
            result.contains("ye"),
            "є should become ye, got: {}",
            result
        );
    }

    #[test]
    fn test_unique_i() {
        let rules = base();
        // і → i
        let result = rules.apply("і");
        assert!(
            result.contains('i'),
            "і should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_unique_plosive_g() {
        let rules = base();
        // ґ → g (plosive g)
        let result = rules.apply("ґ");
        assert!(
            result.contains('g'),
            "ґ should become g, got: {}",
            result
        );
    }

    #[test]
    fn test_ukrainian_h() {
        let rules = base();
        // г → h (voiced glottal, NOT g!)
        let result = rules.apply("г");
        assert!(
            result.contains('h'),
            "г should become h (not g!), got: {}",
            result
        );
    }

    #[test]
    fn test_ukrainian_y_vowel() {
        let rules = base();
        // Ukrainian и → y (sounds like Russian ы)
        let result = rules.apply("и");
        assert!(
            result.contains('y'),
            "и should become y (like Russian ы), got: {}",
            result
        );
    }

    #[test]
    fn test_complex_consonants() {
        let rules = base();
        // щ → shch
        let result = rules.apply("щ");
        assert!(
            result.contains("shch"),
            "щ should become shch, got: {}",
            result
        );
        // ж → zh (test with following vowel to avoid word-final devoicing)
        let result = rules.apply("жа");
        assert!(
            result.contains("zh"),
            "ж should become zh, got: {}",
            result
        );
        // ш → sh
        let result = rules.apply("ш");
        assert!(
            result.contains("sh"),
            "ш should become sh, got: {}",
            result
        );
    }

    #[test]
    fn test_iotated_vowels() {
        let rules = base();
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
    fn test_simple_vowels() {
        let rules = base();
        // а → a
        let result = rules.apply("а");
        assert!(
            result.contains('a'),
            "а should become a, got: {}",
            result
        );
        // о → o
        let result = rules.apply("о");
        assert!(
            result.contains('o'),
            "о should become o, got: {}",
            result
        );
        // у → u
        let result = rules.apply("у");
        assert!(
            result.contains('u'),
            "у should become u, got: {}",
            result
        );
    }

    #[test]
    fn test_word_kyiv() {
        let rules = base();
        // Київ (Kyiv) - capital of Ukraine
        let result = rules.apply("Київ");
        // Should contain k, y (from и), i (from і), v (from в)
        assert!(
            result.contains('k') && result.contains('y'),
            "Київ should contain k and y, got: {}",
            result
        );
    }

    #[test]
    fn test_word_ukraina() {
        let rules = base();
        // Україна (Ukraine)
        let result = rules.apply("Україна");
        // Should contain u, k, r, a, y, n
        assert!(
            result.contains('u') && result.contains('k') && result.contains('n'),
            "Україна should contain u, k, n, got: {}",
            result
        );
    }

    #[test]
    fn test_rules_sorted_by_weight() {
        let rules = base();
        let weights: Vec<_> = rules.rules.iter().map(|r| r.weight).collect();
        let mut sorted_weights = weights.clone();
        sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(weights, sorted_weights, "Rules should be sorted by weight");
    }
}
