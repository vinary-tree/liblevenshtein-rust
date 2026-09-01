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
//! ```rust
//! use liblevenshtein::phonetic::rules::czech;
//!
//! let rules = czech::base();
//!
//! // Special consonants
//! let result = rules.apply("č");
//! assert!(result.contains("t͡ʃ"), "č → CH");
//!
//! // Unique ř
//! let result = rules.apply("ř");
//! assert!(result.contains("r̝"), "ř → RZH");
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
        let file = crate::phonetic::llev::parse_str(content).expect(
            "Invalid embedded czech/base.llev - this indicates an internal invariant violation",
        );
        RuleSetChar::from_llev(&file).expect(
            "Failed to compile Czech base rules - this indicates an internal invariant violation",
        )
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
            rules.len() >= 15,
            "expected >=15 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_c_hacek() {
        let rules = base();
        // č → t͡ʃ (voiceless postalveolar affricate)
        let result = rules.apply("č");
        assert!(
            result.contains("t͡ʃ"),
            "č should produce t͡ʃ sound, got: {}",
            result
        );
    }

    #[test]
    fn test_s_hacek() {
        let rules = base();
        // š → SH
        let result = rules.apply("š");
        assert!(result.contains("ʃ"), "š should become SH, got: {}", result);
    }

    #[test]
    fn test_z_hacek() {
        let rules = base();
        // ž → ZH
        let result = rules.apply("ž");
        assert!(result.contains("ʒ"), "ž should become ZH, got: {}", result);
    }

    #[test]
    fn test_r_hacek() {
        let rules = base();
        // ř → RZH (unique Czech sound)
        let result = rules.apply("ř");
        assert!(result.contains("r̝"), "ř should become RZH, got: {}", result);
    }

    #[test]
    fn test_soft_d() {
        let rules = base();
        // ď → ɟ (voiced palatal plosive)
        let result = rules.apply("ď");
        assert!(
            result.contains("ɟ") || result.contains("d͡ʒ"),
            "ď should become ɟ or d͡ʒ, got: {}",
            result
        );
    }

    #[test]
    fn test_soft_t() {
        let rules = base();
        // ť → c (voiceless palatal plosive), which may then become t͡s via c -> t͡s rule
        let result = rules.apply("ť");
        // Note: ť first becomes c (palatal plosive), then c -> t͡s rule may apply
        assert!(
            result.contains('c') || result.contains("tɕ") || result.contains("t͡s"),
            "ť should become c (palatal plosive), tɕ, or t͡s, got: {}",
            result
        );
    }

    #[test]
    fn test_soft_n() {
        let rules = base();
        // ň → NJ
        let result = rules.apply("ň");
        assert!(result.contains("ɲ"), "ň should become NJ, got: {}", result);
    }

    #[test]
    fn test_long_a() {
        let rules = base();
        // á → aː (long a in IPA)
        let result = rules.apply("á");
        assert!(
            result.contains("aː") || result.contains('a'),
            "á should become aː or a, got: {}",
            result
        );
    }

    #[test]
    fn test_u_ring() {
        let rules = base();
        // ů → uː (kroužek, long u in IPA)
        let result = rules.apply("ů");
        assert!(
            result.contains("uː") || result.contains('u'),
            "ů should become uː or u, got: {}",
            result
        );
    }

    #[test]
    fn test_y_to_i() {
        let rules = base();
        // y → i (same pronunciation in Czech)
        let result = rules.apply("y");
        assert!(result.contains('i'), "y should become i, got: {}", result);
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
            result.contains("t͡ʃ"),
            "česko should contain CH, got: {}",
            result
        );
    }
}
