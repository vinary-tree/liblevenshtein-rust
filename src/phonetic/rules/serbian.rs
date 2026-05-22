//! Serbian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Serbian (Српски језик / Srpski jezik) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Serbian phonetic normalization handles:
//! - **Dual script support**: Both Cyrillic and Latin scripts are official
//! - **Perfect phonemic orthography**: One letter = one sound
//! - **Unique Cyrillic letters**: Љ(lj), Њ(nj), Џ(dž), Ћ(ć), Ђ(đ)
//! - **Latin digraphs**: lj, nj, dž (treated as single phonemes)
//! - **Latin diacritics**: č, ć, š, ž, đ
//!
//! # Script Variants
//!
//! Serbian uniquely uses both Cyrillic and Latin scripts officially:
//! - **Cyrillic** (`base()`): Traditional script, used in official documents
//! - **Latin** (`latin()`): Common in everyday/casual use
//!
//! Both scripts represent exactly the same sounds - Serbian has perfect
//! phonemic orthography where every letter corresponds to exactly one sound.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Serbian Cyrillic transliteration (~64 rules)
//! - [`latin()`] - Serbian Latin normalization (~39 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::serbian;
//!
//! // Cyrillic script
//! let cyrillic_rules = serbian::base();
//! let result = cyrillic_rules.apply("љ");
//! assert!(result.contains("lj"), "љ → lj");
//!
//! // Latin script
//! let latin_rules = serbian::latin();
//! let result = latin_rules.apply("č");
//! assert!(result.contains("t͡ʃ"), "č → ch");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Serbian phonetic rules (Cyrillic script).
///
/// Complete phonetic normalization rules for Serbian Cyrillic:
///
/// ## Unique Cyrillic Letters
/// - Љ → lj (palatal lateral approximant)
/// - Њ → nj (palatal nasal)
/// - Џ → dz (voiced postalveolar affricate)
/// - Ћ → c (voiceless alveolo-palatal affricate, soft)
/// - Ђ → dj (voiced alveolo-palatal affricate)
///
/// ## Complex Consonants
/// - Ж → zh (voiced postalveolar fricative)
/// - Ш → sh (voiceless postalveolar fricative)
/// - Х → kh (voiceless velar fricative)
/// - Ц → ts (voiceless alveolar affricate)
/// - Ч → ch (voiceless postalveolar affricate)
///
/// ## Standard Vowels and Consonants
/// - А, Е, И, О, У → a, e, i, o, u
/// - Б, В, Г, Д, З, Ј, К, Л, М, Н, П, Р, С, Т, Ф → b, v, g, d, z, j, k, l, m, n, p, r, s, t, f
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/serbian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded serbian/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Serbian base rules - this is a bug in liblevenshtein")
    })
}

/// Serbian Latin script phonetic rules.
///
/// Phonetic normalization rules for Serbian Latin script:
///
/// ## Digraphs (single phonemes)
/// - lj → lj (palatal lateral approximant)
/// - nj → nj (palatal nasal)
/// - dž → dz (voiced postalveolar affricate)
///
/// ## Diacritical Consonants
/// - č → ch (voiceless postalveolar affricate)
/// - ć → c (voiceless alveolo-palatal affricate, softer than č)
/// - š → sh (voiceless postalveolar fricative)
/// - ž → zh (voiced postalveolar fricative)
/// - đ → dj (voiced alveolo-palatal affricate)
///
/// ## Case Normalization
/// - Uppercase letters → lowercase equivalents
pub fn latin() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/serbian/latin.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded serbian/latin.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Serbian Latin rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // CYRILLIC SCRIPT TESTS
    // ============================================================

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Serbian base rules should not be empty");
        assert!(
            rules.len() >= 30,
            "expected >=30 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_cyrillic_lj() {
        let rules = base();
        // Љ → ʎ (IPA palatal lateral approximant)
        let result = rules.apply("љ");
        assert!(
            result.contains('ʎ') || result.contains("lj"),
            "љ should become ʎ, got: {}",
            result
        );
    }

    #[test]
    fn test_cyrillic_nj() {
        let rules = base();
        // Њ → ɲ (IPA palatal nasal)
        let result = rules.apply("њ");
        assert!(
            result.contains('ɲ') || result.contains("nj"),
            "њ should become ɲ, got: {}",
            result
        );
    }

    #[test]
    fn test_cyrillic_dzh() {
        let rules = base();
        // Џ → d͡ʒ (IPA voiced postalveolar affricate)
        let result = rules.apply("џ");
        assert!(
            result.contains("d͡ʒ") || result.contains("dz"),
            "џ should become d͡ʒ, got: {}",
            result
        );
    }

    #[test]
    fn test_cyrillic_soft_ch() {
        let rules = base();
        // Ћ → tɕ (IPA voiceless alveolo-palatal affricate)
        let result = rules.apply("ћ");
        assert!(
            result.contains("tɕ") || result.contains('c'),
            "ћ should become tɕ, got: {}",
            result
        );
    }

    #[test]
    fn test_cyrillic_dj() {
        let rules = base();
        // Ђ → dʑ (IPA voiced alveolo-palatal affricate)
        let result = rules.apply("ђ");
        assert!(
            result.contains("dʑ") || result.contains("d͡ʒ"),
            "ђ should become dʑ, got: {}",
            result
        );
    }

    #[test]
    fn test_cyrillic_zh() {
        let rules = base();
        // Ж → zh
        let result = rules.apply("ж");
        assert!(result.contains("ʒ"), "ж should become zh, got: {}", result);
    }

    #[test]
    fn test_cyrillic_sh() {
        let rules = base();
        // Ш → sh
        let result = rules.apply("ш");
        assert!(result.contains("ʃ"), "ш should become sh, got: {}", result);
    }

    #[test]
    fn test_cyrillic_ch() {
        let rules = base();
        // Ч → ch
        let result = rules.apply("ч");
        assert!(result.contains("t͡ʃ"), "ч should become ch, got: {}", result);
    }

    #[test]
    fn test_cyrillic_word_beograd() {
        let rules = base();
        // Београд (Belgrade)
        let result = rules.apply_full("београд");
        assert!(
            result.contains('b') && result.contains('r') && result.contains('d'),
            "београд should contain b, r, d, got: {}",
            result
        );
    }

    #[test]
    fn test_cyrillic_word_srbija() {
        let rules = base();
        // Србија (Serbia)
        let result = rules.apply_full("србија");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('s') && lower.contains('r') && lower.contains('b'),
            "србија should contain s, r, b, got: {}",
            result
        );
    }

    // ============================================================
    // LATIN SCRIPT TESTS
    // ============================================================

    #[test]
    fn test_latin_loads() {
        let rules = latin();
        assert!(!rules.is_empty(), "Serbian Latin rules should not be empty");
        assert!(
            rules.len() >= 8,
            "expected >=8 latin rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_latin_c_caron() {
        let rules = latin();
        // č → ch
        let result = rules.apply("č");
        assert!(result.contains("t͡ʃ"), "č should become ch, got: {}", result);
    }

    #[test]
    fn test_latin_c_acute() {
        let rules = latin();
        // ć → tɕ (IPA voiceless alveolo-palatal affricate)
        let result = rules.apply("ć");
        assert!(
            result.contains("tɕ") || result.contains('c'),
            "ć should become tɕ, got: {}",
            result
        );
    }

    #[test]
    fn test_latin_s_caron() {
        let rules = latin();
        // š → sh
        let result = rules.apply("š");
        assert!(result.contains("ʃ"), "š should become sh, got: {}", result);
    }

    #[test]
    fn test_latin_z_caron() {
        let rules = latin();
        // ž → zh
        let result = rules.apply("ž");
        assert!(result.contains("ʒ"), "ž should become zh, got: {}", result);
    }

    #[test]
    fn test_latin_d_stroke() {
        let rules = latin();
        // đ → dʑ (IPA voiced alveolo-palatal affricate)
        let result = rules.apply("đ");
        assert!(
            result.contains("dʑ") || result.contains("d͡ʒ"),
            "đ should become dʑ, got: {}",
            result
        );
    }

    #[test]
    fn test_latin_dz_digraph() {
        let rules = latin();
        // dž → d͡ʒ (IPA voiced postalveolar affricate)
        let result = rules.apply("dž");
        assert!(
            result.contains("d͡ʒ") || result.contains("dz"),
            "dž should become d͡ʒ, got: {}",
            result
        );
    }

    #[test]
    fn test_latin_word_beograd() {
        let rules = latin();
        // Beograd (Belgrade in Latin script)
        let result = rules.apply_full("Beograd");
        assert!(
            result.contains('b') && result.contains('r') && result.contains('d'),
            "Beograd should contain b, r, d, got: {}",
            result
        );
    }

    #[test]
    fn test_latin_word_srbija() {
        let rules = latin();
        // Srbija (Serbia in Latin script)
        let result = rules.apply_full("Srbija");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('s') && lower.contains('r') && lower.contains('b'),
            "Srbija should contain s, r, b, got: {}",
            result
        );
    }

    // ============================================================
    // WEIGHT ORDERING TESTS
    // ============================================================
}
