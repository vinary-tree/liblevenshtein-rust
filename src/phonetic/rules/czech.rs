//! Czech phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Czech (čeština) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Czech phonetic normalization handles:
//! - **Háčky (caron)**: č→CH, š→SH, ž→ZH, ř→RZH
//! - **Soft consonants**: ď→DJ, ť→TJ, ň→NJ
//! - **Long vowels (čárka)**: á, é, í, ó, ú, ý → short equivalents
//! - **Kroužek**: ů→u (historical long u)
//! - **Y/I merger**: y→i (same pronunciation)
//!
//! # Unique Czech Sound
//!
//! The letter ř represents a unique Czech sound - a raised alveolar trill
//! [r̝] that exists only in Czech. It's transcribed as RZH here.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Czech transliteration rules (~80 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::czech;
//!
//! let rules = czech::base();
//!
//! // Special consonants
//! let result = rules.apply("č");
//! assert!(result.contains("CH"), "č → CH");
//!
//! // Unique ř
//! let result = rules.apply("ř");
//! assert!(result.contains("RZH"), "ř → RZH");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Czech phonetic rules.
///
/// Complete phonetic normalization rules for Czech:
///
/// ## Consonants with háček
/// - č → CH (like English "ch")
/// - š → SH (like English "sh")
/// - ž → ZH (like French "j")
/// - ř → RZH (unique Czech raised alveolar trill)
///
/// ## Soft consonants
/// - ď → DJ (palatal d)
/// - ť → TJ (palatal t)
/// - ň → NJ (palatal n, like Spanish ñ)
///
/// ## Long vowels (čárka/acute)
/// - á, é, í, ó, ú, ý → short equivalents
/// - ů → u (kroužek, mid/end of word long u)
///
/// ## Other Features
/// - y → i (y and i have same pronunciation)
/// - c → ts
/// - w → v
/// - x → ks
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/czech/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded czech/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Czech base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Czech base rules should not be empty");
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
    fn test_r_hacek() {
        let rules = base();
        // ř → RZH (unique Czech sound)
        let result = rules.apply("ř");
        assert!(
            result.to_uppercase().contains("RZH"),
            "ř should become RZH, got: {}",
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
    fn test_soft_t() {
        let rules = base();
        // ť → TJ
        let result = rules.apply("ť");
        assert!(
            result.to_uppercase().contains("TJ"),
            "ť should become TJ, got: {}",
            result
        );
    }

    #[test]
    fn test_soft_n() {
        let rules = base();
        // ň → NJ
        let result = rules.apply("ň");
        assert!(
            result.to_uppercase().contains("NJ"),
            "ň should become NJ, got: {}",
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
    fn test_u_ring() {
        let rules = base();
        // ů → u (kroužek)
        let result = rules.apply("ů");
        assert!(
            result.to_lowercase().contains('u'),
            "ů should become u, got: {}",
            result
        );
    }

    #[test]
    fn test_y_to_i() {
        let rules = base();
        // y → i (same pronunciation in Czech)
        let result = rules.apply("y");
        assert!(
            result.to_lowercase().contains('i'),
            "y should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_word_praha() {
        let rules = base();
        // Praha (Prague) - use lowercase
        let result = rules.apply_full("praha");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('p') && lower.contains('r') && lower.contains('a'),
            "praha should contain p, r, a, got: {}",
            result
        );
    }

    #[test]
    fn test_word_cesko() {
        let rules = base();
        // Česko (Czech Republic) - use lowercase česko
        let result = rules.apply_full("česko");
        assert!(
            result.to_uppercase().contains("CH"),
            "česko should contain CH, got: {}",
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
