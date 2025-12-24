//! Indonesian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Indonesian (Bahasa Indonesia) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Indonesian phonetic normalization handles:
//! - **Nearly phonemic orthography**: Indonesian spelling is very regular
//! - **Special C**: c → CH (always like English "ch")
//! - **Digraphs**: ng (velar nasal), ny (palatal nasal), sy (sh), kh (Arabic loans)
//! - **No diacritics**: Standard Indonesian uses plain Latin letters
//! - **V pronunciation**: v → f (common Indonesian pronunciation)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Indonesian transliteration rules (~35 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::indonesian;
//!
//! let rules = indonesian::base();
//!
//! // C is always "ch" sound
//! let result = rules.apply("cari");
//! assert!(result.contains("CH"), "c → CH");
//!
//! // Ng digraph
//! let result = rules.apply("dengan");
//! assert!(result.contains("NG"), "ng → NG");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Indonesian phonetic rules.
///
/// Complete phonetic normalization rules for Indonesian:
///
/// ## Digraphs
/// - ng → NG (velar nasal, very common)
/// - ny → NY (palatal nasal, like Spanish ñ)
/// - sy → SH (like English "sh", Arabic loanwords)
/// - kh → KH (voiceless velar fricative, Arabic loanwords)
///
/// ## Special Consonants
/// - c → CH (always like English "ch")
/// - j → J (like English "j" in "judge")
/// - v → f (common pronunciation in Indonesian)
///
/// ## Vowels
/// - a, e, i, o, u (simple Latin vowels)
///
/// ## Simple Consonants
/// - Standard Latin consonants (b, d, f, g, h, k, l, m, n, p, r, s, t, w, y, z)
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/indonesian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded indonesian/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Indonesian base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Indonesian base rules should not be empty");
        assert!(
            rules.len() >= 30,
            "expected >=30 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_ng_digraph() {
        let rules = base();
        // ng → NG
        let result = rules.apply("ng");
        assert!(
            result.contains("NG"),
            "ng should become NG, got: {}",
            result
        );
    }

    #[test]
    fn test_ny_digraph() {
        let rules = base();
        // ny → NY
        let result = rules.apply("ny");
        assert!(
            result.contains("NY"),
            "ny should become NY, got: {}",
            result
        );
    }

    #[test]
    fn test_sy_digraph() {
        let rules = base();
        // sy → SH
        let result = rules.apply("sy");
        assert!(
            result.to_uppercase().contains("SH"),
            "sy should become SH, got: {}",
            result
        );
    }

    #[test]
    fn test_kh_digraph() {
        let rules = base();
        // kh → KH
        let result = rules.apply("kh");
        assert!(
            result.contains("KH"),
            "kh should become KH, got: {}",
            result
        );
    }

    #[test]
    fn test_c_to_ch() {
        let rules = base();
        // c → CH (always like English "ch")
        let result = rules.apply("c");
        assert!(
            result.contains("CH"),
            "c should become CH, got: {}",
            result
        );
    }

    #[test]
    fn test_v_to_f() {
        let rules = base();
        // v → f (common Indonesian pronunciation)
        let result = rules.apply("v");
        assert!(
            result.contains('f'),
            "v should become f, got: {}",
            result
        );
    }

    #[test]
    fn test_word_dengan() {
        let rules = base();
        // dengan (with) - contains ng digraph
        let result = rules.apply("dengan");
        assert!(
            result.contains("NG"),
            "dengan should contain NG, got: {}",
            result
        );
    }

    #[test]
    fn test_word_cari() {
        let rules = base();
        // cari (search) - c becomes CH
        let result = rules.apply("cari");
        assert!(
            result.contains("CH"),
            "cari should contain CH, got: {}",
            result
        );
    }

    #[test]
    fn test_word_indonesia() {
        let rules = base();
        // Indonesia
        let result = rules.apply("indonesia");
        // Should contain i, n, d, o, n, e, s, i, a
        assert!(
            result.contains('i') && result.contains('n') && result.contains('a'),
            "indonesia should contain i, n, a, got: {}",
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
