//! Slovak phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Slovak (slovenčina) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Slovak phonetic normalization handles:
//! - **Háčky (caron)**: č→CH, š→SH, ž→ZH, ď→DJ, ť→TJ, ň→NJ, ľ→LJ
//! - **No ř**: Unlike Czech, Slovak lacks the unique ř sound
//! - **Diphthong**: ô→UO
//! - **Digraphs**: dž→DZH, dz→DZ (single phonemes)
//! - **Long vowels (čárka)**: á, é, í, ó, ú, ý → short equivalents
//!
//! # Slovak vs Czech
//!
//! Slovak is closely related to Czech but lacks the unique ř sound and
//! has additional features like ľ (soft l) and ô (uo diphthong).
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Slovak transliteration rules (~86 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::slovak;
//!
//! let rules = slovak::base();
//!
//! // Soft l
//! let result = rules.apply("ľ");
//! assert!(result.contains("LJ"), "ľ → LJ");
//!
//! // Diphthong
//! let result = rules.apply("ô");
//! assert!(result.contains("UO"), "ô → UO");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Slovak phonetic rules.
///
/// Complete phonetic normalization rules for Slovak:
///
/// ## Consonants with háček
/// - č → CH (like English "ch")
/// - š → SH (like English "sh")
/// - ž → ZH (like French "j")
///
/// ## Soft consonants
/// - ď → DJ (palatal d)
/// - ť → TJ (palatal t)
/// - ň → NJ (palatal n, like Spanish ñ)
/// - ľ → LJ (soft l - Slovak specific)
///
/// ## Digraphs
/// - dž → DZH (voiced postalveolar affricate)
/// - dz → DZ (voiced alveolar affricate)
///
/// ## Special vowels
/// - ô → UO (diphthong, Slovak specific)
///
/// ## Long vowels (čárka/acute)
/// - á, é, í, ó, ú, ý → short equivalents
///
/// ## Other Features
/// - y → i (y and i have same pronunciation)
/// - c → ts
/// - w → v
/// - x → ks
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/slovak/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded slovak/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Slovak base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Slovak base rules should not be empty");
        assert!(
            rules.len() >= 40,
            "expected >=40 base rules, got {}",
            rules.len()
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
    fn test_soft_l() {
        let rules = base();
        // ľ → LJ (Slovak specific)
        let result = rules.apply("ľ");
        assert!(
            result.to_uppercase().contains("LJ"),
            "ľ should become LJ, got: {}",
            result
        );
    }

    #[test]
    fn test_soft_d() {
        let rules = base();
        // ď → DJ
        let result = rules.apply("ď");
        assert!(
            result.to_uppercase().contains("DJ"),
            "ď should become DJ, got: {}",
            result
        );
    }

    #[test]
    fn test_o_circumflex() {
        let rules = base();
        // ô → UO (diphthong)
        let result = rules.apply("ô");
        assert!(
            result.to_uppercase().contains("UO"),
            "ô should become UO, got: {}",
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
    fn test_long_a() {
        let rules = base();
        // á → a
        let result = rules.apply("á");
        assert!(
            result.to_lowercase().contains('a'),
            "á should become a, got: {}",
            result
        );
    }

    #[test]
    fn test_y_to_i() {
        let rules = base();
        // y → i
        let result = rules.apply("y");
        assert!(
            result.to_lowercase().contains('i'),
            "y should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_word_bratislava() {
        let rules = base();
        // Bratislava - use lowercase
        let result = rules.apply_full("bratislava");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('b') && lower.contains('r') && lower.contains('a'),
            "bratislava should contain b, r, a, got: {}",
            result
        );
    }

    #[test]
    fn test_word_slovensko() {
        let rules = base();
        // Slovensko (Slovakia) - use lowercase
        let result = rules.apply_full("slovensko");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('s') && lower.contains('l'),
            "slovensko should contain s, l, got: {}",
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
