//! Turkish phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Turkish (Türkçe) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Turkish phonetic normalization handles:
//! - **Dotted/undotted I**: ı→I, i→i, I→I, İ→i (four-way distinction)
//! - **Special consonants**: Ş→S, Ç→C, Ğ→G (soft g)
//! - **Front vowels**: Ö→O, Ü→U (like German umlauts)
//! - **C pronunciation**: c→DJ (like English "j" in "judge", uses DJ marker)
//! - **J pronunciation**: j→Z (like French "j" or English "zh")
//!
//! # Phonetic Markers
//!
//! Uses uppercase single-character markers to avoid rule reprocessing:
//! - S = postalveolar fricative (ş)
//! - C = postalveolar affricate (ç)
//! - G = soft g marker (ğ)
//! - I = back/undotted vowel (ı, I)
//! - O = front rounded o (ö)
//! - U = front rounded u (ü)
//! - Z = voiced postalveolar fricative (j)
//!
//! # Turkish Alphabet
//!
//! Turkish uses 29 letters: standard Latin modified with:
//! - Added: Ç, Ğ, I/ı, İ/i, Ö, Ş, Ü
//! - Note: Q, W, X only appear in loanwords
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Turkish phonetic rules (~25 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::turkish;
//!
//! let rules = turkish::base();
//!
//! // Dotted I distinction
//! let result = rules.apply("İstanbul");
//! assert!(result.contains('i'), "İ → i");
//!
//! // Soft g
//! let result = rules.apply("dağ");
//! assert!(result.contains('G'), "ğ → G");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Turkish phonetic rules.
///
/// Complete phonetic normalization rules for Turkish:
///
/// ## Special Consonants (weight 0.05)
/// - ş→S, ç→C, ğ→G
///
/// ## Dotted/Undotted I (weight 0.1)
/// - ı→I (undotted lowercase → I marker)
/// - I→I (undotted uppercase → I marker)
/// - İ→i (dotted uppercase → lowercase i)
///
/// ## Front Vowels (weight 0.1)
/// - ö→O, ü→U
///
/// ## Consonant Transforms (weight 0.15)
/// - c→DJ (Turkish c = English j, uses DJ marker to avoid chaining)
/// - j→Z (Turkish j = French j / English zh)
///
/// ## Simplifications (weight 0.2)
/// - SS→S, CC→C, GG→G, II→I, OO→O, UU→U, ZZ→Z, DJDJ→DJ
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/turkish/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded turkish/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Turkish base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Turkish base rules should not be empty");
        assert!(
            rules.len() > 20,
            "expected >20 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_special_consonants() {
        let rules = base();
        // ş → S
        let result = rules.apply("ş");
        assert!(
            result.contains('S'),
            "ş should become S, got: {}",
            result
        );
        // ç → C
        let result = rules.apply("ç");
        assert!(
            result.contains('C'),
            "ç should become C, got: {}",
            result
        );
        // ğ → G
        let result = rules.apply("ğ");
        assert!(
            result.contains('G'),
            "ğ should become G, got: {}",
            result
        );
    }

    #[test]
    fn test_dotted_undotted_i() {
        let rules = base();
        // ı (undotted lowercase) → I
        let result = rules.apply("ı");
        assert!(
            result.contains('I'),
            "ı should become I, got: {}",
            result
        );
        // İ (dotted uppercase) → i
        let result = rules.apply("İ");
        assert!(
            result.contains('i'),
            "İ should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_front_vowels() {
        let rules = base();
        // ö → O
        let result = rules.apply("ö");
        assert!(
            result.contains('O'),
            "ö should become O, got: {}",
            result
        );
        // ü → U
        let result = rules.apply("ü");
        assert!(
            result.contains('U'),
            "ü should become U, got: {}",
            result
        );
    }

    #[test]
    fn test_consonant_transforms() {
        let rules = base();
        // c → DJ (Turkish c sounds like English j, using DJ marker)
        let result = rules.apply("c");
        assert!(
            result.contains("DJ"),
            "c should become DJ, got: {}",
            result
        );
        // j → Z (Turkish j sounds like French j)
        let result = rules.apply("j");
        assert!(
            result.contains('Z'),
            "j should become Z, got: {}",
            result
        );
    }

    #[test]
    fn test_word_istanbul() {
        let rules = base();
        // İstanbul - Turkey's largest city
        let result = rules.apply("İstanbul");
        // İ → i, s stays, t stays, a stays, n stays, b stays, u stays, l stays
        assert!(
            result.contains('i') && result.contains('s'),
            "İstanbul should normalize properly, got: {}",
            result
        );
    }

    #[test]
    fn test_word_turkiye() {
        let rules = base();
        // Türkiye (Turkey in Turkish)
        let result = rules.apply("Türkiye");
        // T stays, ü→U, r stays, k stays, i stays, y stays, e stays
        assert!(
            result.contains('U') && result.contains('k'),
            "Türkiye should have U (from ü), got: {}",
            result
        );
    }

    #[test]
    fn test_word_dag() {
        let rules = base();
        // dağ (mountain) - has soft g
        let result = rules.apply("dağ");
        // d stays, a stays, ğ→G
        assert!(
            result.contains('G') && result.contains('d') && result.contains('a'),
            "dağ should have G (from ğ), got: {}",
            result
        );
    }

    #[test]
    fn test_word_gunes() {
        let rules = base();
        // güneş (sun)
        let result = rules.apply("güneş");
        // g stays, ü→U, n stays, e stays, ş→S
        assert!(
            result.contains('U') && result.contains('S'),
            "güneş should have U and S, got: {}",
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
