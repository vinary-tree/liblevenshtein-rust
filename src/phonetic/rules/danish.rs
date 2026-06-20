//! Danish phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Danish (dansk) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Danish phonetic normalization handles:
//! - **Three extra vowels**: æ→AE, ø→OE, å→O
//! - **Old spelling**: aa→O (historical spelling for å)
//! - **SJ-sound**: sj→SJ
//! - **Silent clusters**: hv→v, hj→j
//! - **Velar nasal**: ng→NG
//!
//! # Stød
//!
//! Danish has a distinctive feature called "stød" (glottal stop) that is
//! not represented in spelling and therefore not handled by these rules.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Danish transliteration rules (~70 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::danish;
//!
//! let rules = danish::base();
//!
//! // Special vowels
//! let result = rules.apply("æ");
//! assert!(result.contains("æ"), "æ → AE");
//!
//! // Old spelling for å
//! let result = rules.apply("aa");
//! assert!(result.contains('O'), "aa → O");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Danish phonetic rules.
///
/// Complete phonetic normalization rules for Danish:
///
/// ## Digraphs
/// - sj → SJ (sj-sound)
/// - ng → NG (velar nasal)
/// - aa → O (old spelling for å)
///
/// ## Silent Clusters
/// - hv → v (h silent before v)
/// - hj → j (h silent before j)
///
/// ## Special Vowels
/// - æ → AE (front open vowel)
/// - ø → OE (front rounded vowel)
/// - å → O (rounded back vowel)
///
/// ## Other Features
/// - y → Y (front rounded vowel)
/// - c → k
/// - w → v
/// - x → ks
/// - z → s
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/danish/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded danish/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Danish base rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Danish base rules should not be empty");
        assert!(
            rules.len() >= 25,
            "expected >=25 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_ae_ligature() {
        let rules = base();
        // æ → AE
        let result = rules.apply("æ");
        assert!(result.contains("æ"), "æ should become AE, got: {}", result);
    }

    #[test]
    fn test_o_slash() {
        let rules = base();
        // ø → OE
        let result = rules.apply("ø");
        assert!(result.contains("ø"), "ø should become OE, got: {}", result);
    }

    #[test]
    fn test_a_ring() {
        let rules = base();
        // å → O → o (normalized to lowercase)
        let result = rules.apply("å");
        assert!(result.contains('ɔ'), "å should become o, got: {}", result);
    }

    #[test]
    fn test_aa_old_spelling() {
        let rules = base();
        // aa → O → o (old spelling for å, normalized to lowercase)
        let result = rules.apply("aa");
        assert!(result.contains('ɔ'), "aa should become o, got: {}", result);
    }

    #[test]
    fn test_sj_digraph() {
        let rules = base();
        // sj → SJ
        let result = rules.apply("sj");
        assert!(result.contains("ʃ"), "sj should become SJ, got: {}", result);
    }

    #[test]
    fn test_silent_hv() {
        let rules = base();
        // hv → v
        let result = rules.apply("hv");
        assert!(
            result.contains('v') && !result.contains('h'),
            "hv should become v, got: {}",
            result
        );
    }

    #[test]
    fn test_silent_hj() {
        let rules = base();
        // hj → J
        let result = rules.apply("hj");
        assert!(result.contains('j'), "hj should become J, got: {}", result);
    }

    #[test]
    fn test_word_kobenhavn() {
        let rules = base();
        // København (Copenhagen) - ø is preserved as IPA ø
        let result = rules.apply_full("københavn");
        assert!(
            result.contains('k') && result.contains('ø'),
            "københavn should contain k and ø, got: {}",
            result
        );
    }

    #[test]
    fn test_word_danmark() {
        let rules = base();
        // Danmark (Denmark) - use lowercase input
        let result = rules.apply_full("danmark");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('d') && lower.contains('a'),
            "danmark should contain d, a, got: {}",
            result
        );
    }
}
