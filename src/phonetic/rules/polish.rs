//! Polish phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Polish (polski) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Polish phonetic normalization handles:
//! - **Special letters**: ą, ę, ć, ś, ź, ż, ł, ó, ń with diacritics
//! - **Nasal vowels**: ą→on, ę→en (context-dependent nasalization)
//! - **Digraphs**: sz→S, cz→C, rz→Z, dz→dz, dź→J, dż→DZ, ch→X
//! - **Ł**: ł→w (like English "w")
//! - **Ó equivalence**: ó→u (identical pronunciation to u)
//!
//! # Phonetic Markers
//!
//! Uses uppercase single-character markers to avoid rule reprocessing:
//! - S = postalveolar fricative (sz, ś)
//! - C = postalveolar affricate (cz, ć)
//! - Z = postalveolar/palatal fricative (rz, ź, ż)
//! - X = velar fricative (ch, h)
//! - N = palatal nasal (ń)
//! - J = palatal affricate (dź)
//! - W = labio-velar approximant (ł) - like English "w"
//!
//! # Polish Alphabet
//!
//! Polish uses 32 letters: standard Latin minus Q, V, X plus 9 with diacritics.
//! The special characters are: ą, ć, ę, ł, ń, ó, ś, ź, ż
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Polish phonetic rules (~35 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::polish;
//!
//! let rules = polish::base();
//!
//! // Nasal vowel
//! let result = rules.apply("ą");
//! assert!(result.contains("on"), "ą → on");
//!
//! // Digraph (S marker for postalveolar fricative)
//! let result = rules.apply("sz");
//! assert!(result.contains('S'), "sz → S");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Polish phonetic rules.
///
/// Complete phonetic normalization rules for Polish:
///
/// ## Digraphs (highest priority, weight 0.05)
/// - sz→S, cz→C, rz→Z, dz→dz, dź→J, dż→DZ, ch→X
///
/// ## Special Polish Letters (weight 0.1)
/// - ą→on, ę→en (nasal vowels)
/// - ć→C, ś→S, ź→Z, ż→Z (palatalized sibilants → markers)
/// - ł→W, ó→u, ń→N
///
/// ## Consonant Transforms (weight 0.15)
/// - c→ts, h→X, j→y, w→v
///
/// ## Marker Simplifications (weight 0.2)
/// - SS→S, CC→C, ZZ→Z, XX→X, NN→N, WW→W, tsts→ts
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/polish/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded polish/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Polish base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Polish base rules should not be empty");
        assert!(
            rules.len() > 25,
            "expected >25 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_digraphs() {
        let rules = base();
        // sz → S (postalveolar fricative marker)
        let result = rules.apply("sz");
        assert!(
            result.contains('S'),
            "sz should become S, got: {}",
            result
        );
        // cz → C (postalveolar affricate marker)
        let result = rules.apply("cz");
        assert!(
            result.contains('C'),
            "cz should become C, got: {}",
            result
        );
        // rz → Z (voiced postalveolar fricative marker)
        let result = rules.apply("rz");
        assert!(
            result.contains('Z'),
            "rz should become Z, got: {}",
            result
        );
        // ch → X (velar fricative marker)
        let result = rules.apply("ch");
        assert!(
            result.contains('X'),
            "ch should become X, got: {}",
            result
        );
    }

    #[test]
    fn test_nasal_vowels() {
        let rules = base();
        // ą → on
        let result = rules.apply("ą");
        assert!(
            result.contains("on"),
            "ą should become on, got: {}",
            result
        );
        // ę → en
        let result = rules.apply("ę");
        assert!(
            result.contains("en"),
            "ę should become en, got: {}",
            result
        );
    }

    #[test]
    fn test_special_letters() {
        let rules = base();
        // ć → C (affricate marker)
        let result = rules.apply("ć");
        assert!(
            result.contains('C'),
            "ć should become C, got: {}",
            result
        );
        // ś → S (fricative marker)
        let result = rules.apply("ś");
        assert!(
            result.contains('S'),
            "ś should become S, got: {}",
            result
        );
        // ź → Z (fricative marker)
        let result = rules.apply("ź");
        assert!(
            result.contains('Z'),
            "ź should become Z, got: {}",
            result
        );
        // ż → Z (fricative marker)
        let result = rules.apply("ż");
        assert!(
            result.contains('Z'),
            "ż should become Z, got: {}",
            result
        );
        // ł → W (marker for English "w" sound)
        let result = rules.apply("ł");
        assert!(
            result.contains('W'),
            "ł should become W, got: {}",
            result
        );
        // ó → u
        let result = rules.apply("ó");
        assert!(
            result.contains('u'),
            "ó should become u, got: {}",
            result
        );
        // ń → N (palatal nasal marker)
        let result = rules.apply("ń");
        assert!(
            result.contains('N'),
            "ń should become N, got: {}",
            result
        );
    }

    #[test]
    fn test_basic_consonants() {
        let rules = base();
        // c → ts
        let result = rules.apply("c");
        assert!(
            result.contains("ts"),
            "c should become ts, got: {}",
            result
        );
        // j → y
        let result = rules.apply("j");
        assert!(result.contains('y'), "j should become y, got: {}", result);
        // w → v
        let result = rules.apply("w");
        assert!(result.contains('v'), "w should become v, got: {}", result);
        // h → X (velar fricative marker)
        let result = rules.apply("h");
        assert!(
            result.contains('X'),
            "h should become X, got: {}",
            result
        );
    }

    #[test]
    fn test_word_warszawa() {
        let rules = base();
        // Warszawa (Warsaw) - capital of Poland
        let result = rules.apply("Warszawa");
        // Should contain v (from W→w→v), a, r, S (from sz), a, v (from w), a
        assert!(
            result.contains('v') && result.contains('S') && result.contains('a'),
            "Warszawa should normalize properly, got: {}",
            result
        );
    }

    #[test]
    fn test_word_lodz() {
        let rules = base();
        // Łódź - city name with both Ł and ó and dź digraph
        let result = rules.apply("Łódź");
        // Should become something like "WuJ" (W from Ł, u from ó, J from dź)
        assert!(
            result.contains('W') && result.contains('u'),
            "Łódź should have W (from Ł) and u (from ó), got: {}",
            result
        );
    }

    #[test]
    fn test_word_szczecin() {
        let rules = base();
        // Szczecin - city with szcz cluster
        let result = rules.apply("Szczecin");
        // Should have S (from sz) and C (from cz)
        assert!(
            result.contains('S') && result.contains('C'),
            "Szczecin should have S and C markers, got: {}",
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
