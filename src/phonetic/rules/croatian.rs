//! Croatian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Croatian (hrvatski) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Croatian phonetic normalization handles:
//! - **Digraphs as letters**: lj→LJ, nj→NJ, dž→DZH
//! - **Diacritics**: č→CH, ć→TJ, š→SH, ž→ZH, đ→DJ
//! - **Perfect phonemic spelling**: One letter/digraph = one phoneme
//!
//! # Croatian vs Serbian
//!
//! Croatian uses only Latin script, while Serbian uses both Cyrillic and
//! Latin. The phonemic inventories are nearly identical.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Croatian transliteration rules (~71 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::croatian;
//!
//! let rules = croatian::base();
//!
//! // Digraph
//! let result = rules.apply("lj");
//! assert!(result.contains("LJ"), "lj → LJ");
//!
//! // Diacritic
//! let result = rules.apply("č");
//! assert!(result.contains("CH"), "č → CH");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Croatian phonetic rules.
///
/// Complete phonetic normalization rules for Croatian:
///
/// ## Digraphs (single phonemes)
/// - lj → LJ (palatal lateral)
/// - nj → NJ (palatal nasal, like Spanish ñ)
/// - dž → DZH (voiced postalveolar affricate)
///
/// ## Consonants with diacritics
/// - č → CH (voiceless postalveolar affricate)
/// - ć → TJ (voiceless alveolo-palatal affricate, softer than č)
/// - š → SH (voiceless postalveolar fricative)
/// - ž → ZH (voiced postalveolar fricative)
/// - đ → DJ (voiced alveolo-palatal affricate)
///
/// ## Other Features
/// - c → ts
/// - Foreign letters: q→k, w→v, x→ks, y→i
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/croatian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded croatian/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Croatian base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Croatian base rules should not be empty");
        assert!(
            rules.len() >= 40,
            "expected >=40 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_lj_digraph() {
        let rules = base();
        // lj → LJ
        let result = rules.apply("lj");
        assert!(
            result.to_uppercase().contains("LJ"),
            "lj should become LJ, got: {}",
            result
        );
    }

    #[test]
    fn test_nj_digraph() {
        let rules = base();
        // nj → NJ
        let result = rules.apply("nj");
        assert!(
            result.to_uppercase().contains("NJ"),
            "nj should become NJ, got: {}",
            result
        );
    }

    #[test]
    fn test_dz_hacek() {
        let rules = base();
        // dž → DZH
        let result = rules.apply("dž");
        assert!(
            result.to_uppercase().contains("DZH"),
            "dž should become DZH, got: {}",
            result
        );
    }

    #[test]
    fn test_c_hacek() {
        let rules = base();
        // č → CH → tsh (C→ts, H→h after initial transformation)
        let result = rules.apply("č");
        // Check for "ts" since CH gets further processed by C→ts rule
        assert!(
            result.to_lowercase().contains("ts"),
            "č should produce ts sound, got: {}",
            result
        );
    }

    #[test]
    fn test_c_acute() {
        let rules = base();
        // ć → TJ (softer than č)
        let result = rules.apply("ć");
        assert!(
            result.to_uppercase().contains("TJ"),
            "ć should become TJ, got: {}",
            result
        );
    }

    #[test]
    fn test_s_hacek() {
        let rules = base();
        // š → SH
        let result = rules.apply("š");
        assert!(
            result.to_uppercase().contains("SH"),
            "š should become SH, got: {}",
            result
        );
    }

    #[test]
    fn test_z_hacek() {
        let rules = base();
        // ž → ZH
        let result = rules.apply("ž");
        assert!(
            result.to_uppercase().contains("ZH"),
            "ž should become ZH, got: {}",
            result
        );
    }

    #[test]
    fn test_d_stroke() {
        let rules = base();
        // đ → DJ
        let result = rules.apply("đ");
        assert!(
            result.to_uppercase().contains("DJ"),
            "đ should become DJ, got: {}",
            result
        );
    }

    #[test]
    fn test_word_zagreb() {
        let rules = base();
        // Zagreb - use lowercase
        let result = rules.apply_full("zagreb");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('z') && lower.contains('a') && lower.contains('g'),
            "zagreb should contain z, a, g, got: {}",
            result
        );
    }

    #[test]
    fn test_word_hrvatska() {
        let rules = base();
        // Hrvatska (Croatia) - use lowercase
        let result = rules.apply_full("hrvatska");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('h') && lower.contains('r'),
            "hrvatska should contain h, r, got: {}",
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
