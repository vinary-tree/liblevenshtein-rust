//! Romanian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Romanian (română) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Romanian phonetic normalization handles:
//! - **Special vowels**: ă (schwa), â/î (close central unrounded)
//! - **Special consonants**: ș → SH, ț → TS
//! - **Digraphs**: ch → K (before e,i), gh → G (before e,i)
//! - **Romance language**: j → ZH (like French "j")
//!
//! # â vs î
//!
//! Romanian has two spellings for the same sound [ɨ]:
//! - **â** is used in the middle of words
//! - **î** is used at the beginning and end of words
//! Both normalize to 'i' for phonetic matching.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Romanian transliteration rules (~50 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::romanian;
//!
//! let rules = romanian::base();
//!
//! // Special consonants
//! let result = rules.apply("ș");
//! assert!(result.contains("SH"), "ș → SH");
//!
//! // Special vowels
//! let result = rules.apply("România");
//! assert!(result.contains('i'), "â → i");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Romanian phonetic rules.
///
/// Complete phonetic normalization rules for Romanian:
///
/// ## Special Consonants
/// - ș → SH (like English "sh")
/// - ț → TS (like "ts" in "cats")
///
/// ## Special Vowels
/// - ă → a (schwa, reduced to a)
/// - â → i (close central unrounded)
/// - î → i (same as â, word initial/final)
///
/// ## Digraphs
/// - ch → K (before e, i - like Italian)
/// - gh → G (before e, i - like Italian)
///
/// ## Other Features
/// - j → ZH (like French "j")
/// - x → ks
/// - w → v (loanwords)
/// - y → i (loanwords)
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/romanian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded romanian/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Romanian base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Romanian base rules should not be empty");
        assert!(
            rules.len() >= 30,
            "expected >=30 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_s_cedilla() {
        let rules = base();
        // ș → SH
        let result = rules.apply("ș");
        assert!(
            result.to_uppercase().contains("SH"),
            "ș should become SH, got: {}",
            result
        );
    }

    #[test]
    fn test_t_cedilla() {
        let rules = base();
        // ț → TS
        let result = rules.apply("ț");
        assert!(
            result.to_uppercase().contains("TS"),
            "ț should become TS, got: {}",
            result
        );
    }

    #[test]
    fn test_a_breve() {
        let rules = base();
        // ă → a
        let result = rules.apply("ă");
        assert!(
            result.contains('a'),
            "ă should become a, got: {}",
            result
        );
    }

    #[test]
    fn test_a_circumflex() {
        let rules = base();
        // â → i
        let result = rules.apply("â");
        assert!(
            result.contains('i'),
            "â should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_i_circumflex() {
        let rules = base();
        // î → i
        let result = rules.apply("î");
        assert!(
            result.contains('i'),
            "î should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_ch_digraph() {
        let rules = base();
        // ch → K
        let result = rules.apply("ch");
        assert!(
            result.to_uppercase().contains('K'),
            "ch should become K, got: {}",
            result
        );
    }

    #[test]
    fn test_gh_digraph() {
        let rules = base();
        // gh → G
        let result = rules.apply("gh");
        assert!(
            result.to_uppercase().contains('G'),
            "gh should become G, got: {}",
            result
        );
    }

    #[test]
    fn test_j_to_zh() {
        let rules = base();
        // j → ZH
        let result = rules.apply("j");
        assert!(
            result.to_uppercase().contains("ZH"),
            "j should become ZH, got: {}",
            result
        );
    }

    #[test]
    fn test_word_romania() {
        let rules = base();
        // România
        let result = rules.apply("România");
        // â should become i
        assert!(
            result.contains('i'),
            "România should contain i (from â), got: {}",
            result
        );
    }

    #[test]
    fn test_word_bucuresti() {
        let rules = base();
        // București (Bucharest)
        let result = rules.apply("București");
        // Should contain SH (from ș) and TS (from ț)
        assert!(
            result.contains("SH"),
            "București should contain SH, got: {}",
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
