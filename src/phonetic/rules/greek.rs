//! Greek phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Modern Greek (Ελληνικά) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Greek phonetic normalization handles:
//! - **Greek alphabet**: 24 letters (7 vowels, 17 consonants)
//! - **Vowel digraphs**: αι(e), ει(i), οι(i), υι(i), αυ(av), ευ(ev), ου(u)
//! - **Consonant combinations**: μπ(b), ντ(d), γκ(g), γγ(ng), τσ(ts), τζ(dz)
//! - **Accent stripping**: Tonos and diaeresis marks removed
//!
//! # Modern Greek Phonology
//!
//! In Modern Greek, several vowels have merged:
//! - η, ι, υ, ει, οι, υι → all pronounced /i/
//! - ο, ω → both pronounced /o/
//! - αι → pronounced /e/
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Greek to Latin phonetic transliteration (~107 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::greek;
//!
//! let rules = greek::base();
//! let result = rules.apply_full("Ελλάδα");
//! // Result contains Latin phonetic representation
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Greek phonetic rules.
///
/// Complete phonetic normalization rules for Modern Greek:
///
/// ## Vowel Digraphs (highest priority)
/// - αυ → av (before voiced consonants or at word end)
/// - ευ → ev (before voiced consonants or at word end)
/// - ου → u
/// - αι → e
/// - ει → i
/// - οι → i
/// - υι → i
///
/// ## Consonant Combinations
/// - μπ → b (word-initial) / mb (mid-word)
/// - ντ → d (word-initial) / nd (mid-word)
/// - γκ → g (word-initial) / ng (mid-word)
/// - γγ → ng
/// - τσ → ts
/// - τζ → dz
///
/// ## Single Letters
/// - **Vowels**: α(a), ε(e), η(i), ι(i), ο(o), υ(i), ω(o)
/// - **Consonants**: β(v), γ(g), δ(th), ζ(z), θ(th), κ(k), λ(l), μ(m),
///   ν(n), ξ(x), π(p), ρ(r), σ/ς(s), τ(t), φ(f), χ(ch), ψ(ps)
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/greek/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded greek/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Greek base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Greek base rules should not be empty");
        assert!(
            rules.len() >= 80,
            "expected >=80 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // VOWEL DIGRAPH TESTS
    // ============================================================

    #[test]
    fn test_digraph_au() {
        let rules = base();
        // αυ → av
        let result = rules.apply("αυ");
        assert!(
            result.to_lowercase().contains("av"),
            "αυ should become av, got: {}",
            result
        );
    }

    #[test]
    fn test_digraph_eu() {
        let rules = base();
        // ευ → ev
        let result = rules.apply("ευ");
        assert!(
            result.to_lowercase().contains("ev"),
            "ευ should become ev, got: {}",
            result
        );
    }

    #[test]
    fn test_digraph_ou() {
        let rules = base();
        // ου → u
        let result = rules.apply("ου");
        assert!(
            result.to_lowercase().contains('u'),
            "ου should become u, got: {}",
            result
        );
    }

    #[test]
    fn test_digraph_ai() {
        let rules = base();
        // αι → e
        let result = rules.apply("αι");
        assert!(
            result.to_lowercase().contains('e'),
            "αι should become e, got: {}",
            result
        );
    }

    #[test]
    fn test_digraph_ei() {
        let rules = base();
        // ει → i
        let result = rules.apply("ει");
        assert!(
            result.to_lowercase().contains('i'),
            "ει should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_digraph_oi() {
        let rules = base();
        // οι → i
        let result = rules.apply("οι");
        assert!(
            result.to_lowercase().contains('i'),
            "οι should become i, got: {}",
            result
        );
    }

    // ============================================================
    // CONSONANT COMBINATION TESTS
    // ============================================================

    #[test]
    fn test_combo_mp() {
        let rules = base();
        // μπ → b
        let result = rules.apply("μπ");
        assert!(
            result.to_lowercase().contains('b'),
            "μπ should become b, got: {}",
            result
        );
    }

    #[test]
    fn test_combo_nt() {
        let rules = base();
        // ντ → d
        let result = rules.apply("ντ");
        assert!(
            result.to_lowercase().contains('d'),
            "ντ should become d, got: {}",
            result
        );
    }

    #[test]
    fn test_combo_gk() {
        let rules = base();
        // γκ → g
        let result = rules.apply("γκ");
        assert!(
            result.to_lowercase().contains('g'),
            "γκ should become g, got: {}",
            result
        );
    }

    #[test]
    fn test_combo_gg() {
        let rules = base();
        // γγ → ng
        let result = rules.apply("γγ");
        assert!(
            result.to_lowercase().contains("ng"),
            "γγ should become ng, got: {}",
            result
        );
    }

    #[test]
    fn test_combo_ts() {
        let rules = base();
        // τσ → ts
        let result = rules.apply("τσ");
        assert!(
            result.to_lowercase().contains("ts"),
            "τσ should become ts, got: {}",
            result
        );
    }

    #[test]
    fn test_combo_tz() {
        let rules = base();
        // τζ → dz
        let result = rules.apply("τζ");
        assert!(
            result.to_lowercase().contains("dz"),
            "τζ should become dz, got: {}",
            result
        );
    }

    // ============================================================
    // SINGLE VOWEL TESTS
    // ============================================================

    #[test]
    fn test_vowel_alpha() {
        let rules = base();
        let result = rules.apply("α");
        assert!(
            result.contains('a'),
            "α should become a, got: {}",
            result
        );
    }

    #[test]
    fn test_vowel_epsilon() {
        let rules = base();
        let result = rules.apply("ε");
        assert!(
            result.contains('e'),
            "ε should become e, got: {}",
            result
        );
    }

    #[test]
    fn test_vowel_eta() {
        let rules = base();
        // η → i (Modern Greek)
        let result = rules.apply("η");
        assert!(
            result.contains('i'),
            "η should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_vowel_iota() {
        let rules = base();
        let result = rules.apply("ι");
        assert!(
            result.contains('i'),
            "ι should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_vowel_omicron() {
        let rules = base();
        let result = rules.apply("ο");
        assert!(
            result.contains('o'),
            "ο should become o, got: {}",
            result
        );
    }

    #[test]
    fn test_vowel_upsilon() {
        let rules = base();
        // υ → i (Modern Greek)
        let result = rules.apply("υ");
        assert!(
            result.contains('i'),
            "υ should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_vowel_omega() {
        let rules = base();
        let result = rules.apply("ω");
        assert!(
            result.contains('o'),
            "ω should become o, got: {}",
            result
        );
    }

    // ============================================================
    // ACCENTED VOWEL TESTS
    // ============================================================

    #[test]
    fn test_accented_alpha() {
        let rules = base();
        let result = rules.apply("ά");
        assert!(
            result.contains('a'),
            "ά should become a, got: {}",
            result
        );
    }

    #[test]
    fn test_accented_epsilon() {
        let rules = base();
        let result = rules.apply("έ");
        assert!(
            result.contains('e'),
            "έ should become e, got: {}",
            result
        );
    }

    #[test]
    fn test_diaeresis_iota() {
        let rules = base();
        let result = rules.apply("ϊ");
        assert!(
            result.contains('i'),
            "ϊ should become i, got: {}",
            result
        );
    }

    // ============================================================
    // CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_consonant_beta() {
        let rules = base();
        // β → v (Modern Greek)
        let result = rules.apply("β");
        assert!(
            result.contains('v'),
            "β should become v, got: {}",
            result
        );
    }

    #[test]
    fn test_consonant_gamma() {
        let rules = base();
        let result = rules.apply("γ");
        assert!(
            result.contains('g'),
            "γ should become g, got: {}",
            result
        );
    }

    #[test]
    fn test_consonant_delta() {
        let rules = base();
        // δ → th (voiced dental fricative)
        let result = rules.apply("δ");
        assert!(
            result.contains("th"),
            "δ should become th, got: {}",
            result
        );
    }

    #[test]
    fn test_consonant_theta() {
        let rules = base();
        // θ → th (voiceless dental fricative)
        let result = rules.apply("θ");
        assert!(
            result.contains("th"),
            "θ should become th, got: {}",
            result
        );
    }

    #[test]
    fn test_consonant_xi() {
        let rules = base();
        // ξ → x (ks)
        let result = rules.apply("ξ");
        assert!(
            result.contains('x'),
            "ξ should become x, got: {}",
            result
        );
    }

    #[test]
    fn test_consonant_phi() {
        let rules = base();
        let result = rules.apply("φ");
        assert!(
            result.contains('f'),
            "φ should become f, got: {}",
            result
        );
    }

    #[test]
    fn test_consonant_chi() {
        let rules = base();
        // χ → ch
        let result = rules.apply("χ");
        assert!(
            result.contains("ch"),
            "χ should become ch, got: {}",
            result
        );
    }

    #[test]
    fn test_consonant_psi() {
        let rules = base();
        // ψ → ps
        let result = rules.apply("ψ");
        assert!(
            result.contains("ps"),
            "ψ should become ps, got: {}",
            result
        );
    }

    #[test]
    fn test_final_sigma() {
        let rules = base();
        // ς → s (final sigma)
        let result = rules.apply("ς");
        assert!(
            result.contains('s'),
            "ς should become s, got: {}",
            result
        );
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_ellada() {
        let rules = base();
        // Ελλάδα (Greece)
        let result = rules.apply_full("Ελλάδα");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('e') && lower.contains('l') && lower.contains("th") && lower.contains('a'),
            "Ελλάδα should contain e, l, th, a, got: {}",
            result
        );
    }

    #[test]
    fn test_word_athina() {
        let rules = base();
        // Αθήνα (Athens)
        let result = rules.apply_full("Αθήνα");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('a') && lower.contains("th") && lower.contains('i') && lower.contains('n'),
            "Αθήνα should contain a, th, i, n, got: {}",
            result
        );
    }

    #[test]
    fn test_word_olympos() {
        let rules = base();
        // Όλυμπος (Olympus)
        let result = rules.apply_full("Όλυμπος");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('o') && lower.contains('l') && lower.contains('i') && lower.contains('b'),
            "Όλυμπος should normalize with μπ → b, got: {}",
            result
        );
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================

    #[test]
    fn test_base_rules_sorted_by_weight() {
        let rules = base();
        let weights: Vec<_> = rules.rules.iter().map(|r| r.weight).collect();
        let mut sorted_weights = weights.clone();
        sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(weights, sorted_weights, "Base rules should be sorted by weight");
    }
}
