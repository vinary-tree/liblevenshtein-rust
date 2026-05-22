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
//! assert!(result.contains('ʃ'), "sz → S");
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
        // sz → ʃ (IPA voiceless postalveolar fricative)
        let result = rules.apply("sz");
        assert!(
            result.contains('ʃ') || result.contains('S'),
            "sz should become ʃ, got: {}",
            result
        );
        // cz → t͡ʃ (IPA voiceless postalveolar affricate)
        let result = rules.apply("cz");
        assert!(
            result.contains("t͡ʃ") || result.contains('C'),
            "cz should become t͡ʃ, got: {}",
            result
        );
        // rz → ʃ (IPA postalveolar fricative - often devoiced to ʃ)
        let result = rules.apply("rz");
        assert!(
            result.contains('ʃ') || result.contains('ʒ') || result.contains('Z'),
            "rz should become ʃ/ʒ, got: {}",
            result
        );
        // ch → x (IPA voiceless velar fricative)
        let result = rules.apply("ch");
        assert!(
            result.contains('x') || result.contains('X'),
            "ch should become x, got: {}",
            result
        );
    }

    #[test]
    fn test_nasal_vowels() {
        let rules = base();
        // ą → ɔ̃ (IPA nasal o) or on
        let result = rules.apply("ą");
        assert!(
            result.contains('ɔ') || result.contains("on"),
            "ą should become ɔ̃ or on, got: {}",
            result
        );
        // ę → ɛ̃ (IPA nasal e) or en
        let result = rules.apply("ę");
        assert!(
            result.contains('ɛ') || result.contains("en"),
            "ę should become ɛ̃ or en, got: {}",
            result
        );
    }

    #[test]
    fn test_special_letters() {
        let rules = base();
        // ć → tɕ (IPA voiceless alveolo-palatal affricate)
        let result = rules.apply("ć");
        assert!(
            result.contains("tɕ") || result.contains('C'),
            "ć should become tɕ, got: {}",
            result
        );
        // ś → ɕ (IPA voiceless alveolo-palatal fricative)
        let result = rules.apply("ś");
        assert!(
            result.contains('ɕ') || result.contains('ʃ'),
            "ś should become ɕ, got: {}",
            result
        );
        // ź → ɕ (IPA alveolo-palatal fricative - often devoiced)
        let result = rules.apply("ź");
        assert!(
            result.contains('ɕ') || result.contains('ʑ') || result.contains('Z'),
            "ź should become ɕ/ʑ, got: {}",
            result
        );
        // ż → ʃ/ʒ (IPA postalveolar fricative - often devoiced to ʃ)
        let result = rules.apply("ż");
        assert!(
            result.contains('ʃ') || result.contains('ʒ') || result.contains('Z'),
            "ż should become ʃ/ʒ, got: {}",
            result
        );
        // ł → w/f (like English "w" but can be devoiced)
        let result = rules.apply("ł");
        assert!(
            result.contains('w') || result.contains('W') || result.contains('f'),
            "ł should become w/f, got: {}",
            result
        );
        // ó → u
        let result = rules.apply("ó");
        assert!(result.contains('u'), "ó should become u, got: {}", result);
        // ń → ɲ (IPA palatal nasal)
        let result = rules.apply("ń");
        assert!(
            result.contains('ɲ') || result.contains('ŋ') || result.contains('N'),
            "ń should become ɲ, got: {}",
            result
        );
    }

    #[test]
    fn test_basic_consonants() {
        let rules = base();
        // c → t͡s (IPA voiceless alveolar affricate)
        let result = rules.apply("c");
        assert!(result.contains("t͡s"), "c should become t͡s, got: {}", result);
        // j → j (IPA palatal approximant - stays as j)
        let result = rules.apply("j");
        assert!(
            result.contains('j') || result.contains('y'),
            "j should become j or y, got: {}",
            result
        );
        // w → f/v (can be devoiced to f in some contexts)
        let result = rules.apply("w");
        assert!(
            result.contains('v') || result.contains('f'),
            "w should become v/f, got: {}",
            result
        );
        // h → x (IPA voiceless velar fricative)
        let result = rules.apply("h");
        assert!(
            result.contains('x') || result.contains('X'),
            "h should become x, got: {}",
            result
        );
    }

    #[test]
    fn test_word_warszawa() {
        let rules = base();
        // Warszawa (Warsaw) - capital of Poland
        let result = rules.apply("Warszawa");
        // Should contain v (from W→w→v), a, r, ʃ (from sz), a, v (from w), a
        assert!(
            result.contains('v')
                && (result.contains('ʃ') || result.contains('S'))
                && result.contains('a'),
            "Warszawa should normalize properly, got: {}",
            result
        );
    }

    #[test]
    fn test_word_lodz() {
        let rules = base();
        // Łódź - city name with both Ł and ó and dź digraph
        let result = rules.apply("Łódź");
        // Should become something like "wud͡ʑ" (w from Ł, u from ó, d͡ʑ from dź)
        assert!(
            (result.contains('w') || result.contains('W')) && result.contains('u'),
            "Łódź should have w (from Ł) and u (from ó), got: {}",
            result
        );
    }

    #[test]
    fn test_word_szczecin() {
        let rules = base();
        // Szczecin - city with szcz cluster
        let result = rules.apply("Szczecin");
        // Should have ʃ (from sz) and t͡ʃ (from cz)
        assert!(
            (result.contains('ʃ') || result.contains('S'))
                && (result.contains("t͡ʃ") || result.contains('C')),
            "Szczecin should have ʃ and t͡ʃ markers, got: {}",
            result
        );
    }
}
