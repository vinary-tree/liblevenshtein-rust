//! Catalan phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Catalan (català) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Catalan phonetic normalization handles:
//! - **L·L (ela geminada)**: Geminated l, unique to Catalan (→L)
//! - **Digraphs**: ny→Ñ, tx→Č, ll→Λ, ss→S, tg/tj→Ž
//! - **X = "sh" sound**: Like English "sh" (normalized to X)
//! - **Ç = "s" sound**: C cedilla
//! - **Accents**: à, è, é, í, ò, ó, ú, ï, ü (normalized)
//! - **Silent H**: H is not pronounced
//! - **B/V merger**: v→b in most dialects
//!
//! # Regional Varieties
//!
//! Catalan has several dialects (Central, Valencian, Balearic) with some
//! phonological differences. These rules normalize to a standard form.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Catalan transliteration rules (~60 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::catalan;
//!
//! let rules = catalan::base();
//!
//! // Ela geminada
//! let result = rules.apply("l·l");
//! assert!(result.contains("LL"), "l·l → LL");
//!
//! // NY digraph
//! let result = rules.apply("Catalunya");
//! assert!(result.contains("NY"), "ny → NY");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Catalan phonetic rules.
///
/// Complete phonetic normalization rules for Catalan:
///
/// ## Special Features
/// - l·l → L (ela geminada, unique to Catalan)
///
/// ## Digraphs
/// - ny → Ñ (palatal nasal, like Spanish ñ)
/// - tx → Č (like English "ch")
/// - ll → Λ (palatal lateral)
/// - ss → S (geminate s)
/// - tg, tj → Ž (like English "j")
/// - ig → Č (in final position)
///
/// ## Special Consonants
/// - ç → S (c cedilla)
/// - x → X (represents "sh" sound, kept as X to avoid H-stripping)
/// - h → (silent)
/// - v → b (b/v merger)
///
/// ## Accented Vowels
/// - à, è, é → a, e (accent marks normalized)
/// - í, ï → i
/// - ò, ó → o
/// - ú, ü → u
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/catalan/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded catalan/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Catalan base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Catalan base rules should not be empty");
        assert!(
            rules.len() >= 40,
            "expected >=40 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_ela_geminada() {
        let rules = base();
        // l·l → L (geminated l normalized to single L)
        let result = rules.apply("l·l");
        assert!(
            result.to_uppercase().contains('L'),
            "l·l should become L, got: {}",
            result
        );
    }

    #[test]
    fn test_ny_digraph() {
        let rules = base();
        // ny → Ñ (palatal nasal)
        let result = rules.apply("ny");
        assert!(
            result.contains('Ñ'),
            "ny should become Ñ, got: {}",
            result
        );
    }

    #[test]
    fn test_tx_digraph() {
        let rules = base();
        // tx → Č (ch sound)
        let result = rules.apply("tx");
        assert!(
            result.contains('Č'),
            "tx should become Č, got: {}",
            result
        );
    }

    #[test]
    fn test_ll_digraph() {
        let rules = base();
        // ll → Λ (palatal lateral)
        let result = rules.apply("ll");
        assert!(
            result.contains('Λ'),
            "ll should become Λ, got: {}",
            result
        );
    }

    #[test]
    fn test_c_cedilla() {
        let rules = base();
        // ç → S
        let result = rules.apply("ç");
        assert!(
            result.contains('S'),
            "ç should become S, got: {}",
            result
        );
    }

    #[test]
    fn test_x_to_x() {
        let rules = base();
        // x → X (represents "sh" sound, kept as X to avoid H-stripping)
        let result = rules.apply("x");
        assert!(
            result.contains('X'),
            "x should become X, got: {}",
            result
        );
    }

    #[test]
    fn test_v_to_b() {
        let rules = base();
        // v → b
        let result = rules.apply("v");
        assert!(
            result.contains('b'),
            "v should become b, got: {}",
            result
        );
    }

    #[test]
    fn test_accented_vowels() {
        let rules = base();
        // à → a
        let result = rules.apply("à");
        assert!(
            result.contains('a'),
            "à should become a, got: {}",
            result
        );
        // é → e
        let result = rules.apply("é");
        assert!(
            result.contains('e'),
            "é should become e, got: {}",
            result
        );
        // ó → o
        let result = rules.apply("ó");
        assert!(
            result.contains('o'),
            "ó should become o, got: {}",
            result
        );
    }

    #[test]
    fn test_word_catalunya() {
        let rules = base();
        // Catalunya
        let result = rules.apply("Catalunya");
        // Should contain Ñ (from ny)
        assert!(
            result.contains('Ñ'),
            "Catalunya should contain Ñ, got: {}",
            result
        );
    }

    #[test]
    fn test_word_barcelona() {
        let rules = base();
        // Barcelona
        let result = rules.apply("Barcelona");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('b') && lower.contains('a'),
            "Barcelona should contain b and a, got: {}",
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
