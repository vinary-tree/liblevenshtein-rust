//! Hungarian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Hungarian (magyar) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Hungarian phonetic normalization handles:
//! - **Digraphs** (9 digraphs treated as single letters):
//!   - cs→CH, dz→DZ, dzs→DZS, gy→GY, ly→Y, ny→NY, sz→S, ty→TY, zs→ZS
//! - **Unique S sound**: Hungarian s→SH (like English "sh")!
//! - **Long vowels**: á→A, é→E, í→I, ó→O, ú→U
//! - **Front rounded vowels**: ö→OE, ő→OE, ü→UE, ű→UE
//! - **Double-acute accent**: ő, ű (long front rounded vowels)
//! - **Geminate digraphs**: ccs→CH, ssz→S, nny→NY, etc.
//!
//! # Unique Hungarian Feature
//!
//! In Hungarian, standalone **S = "sh" sound** (like English "sh").
//! This is unusual among European languages where S typically = "s".
//! The "s" sound in Hungarian is written as SZ (digraph).
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Hungarian transliteration rules (~70 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::hungarian;
//!
//! let rules = hungarian::base();
//!
//! // CS digraph
//! let result = rules.apply("cs");
//! assert!(result.contains("CH"), "cs → CH");
//!
//! // S = "sh" sound (unique to Hungarian!)
//! let result = rules.apply("s");
//! assert!(result.contains("SH"), "s → SH");
//!
//! // SZ digraph = "s" sound
//! let result = rules.apply("sz");
//! assert!(result.contains('S'), "sz → S");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Hungarian phonetic rules.
///
/// Complete phonetic normalization rules for Hungarian:
///
/// ## Trigraph
/// - dzs → DZS (like English "j")
///
/// ## Digraphs (treated as single letters in Hungarian)
/// - cs → CH (like English "ch")
/// - dz → DZ (voiced alveolar affricate)
/// - gy → GY (palatalized d)
/// - ly → Y (palatal lateral, [j] in modern Hungarian)
/// - ny → NY (palatal nasal, like Spanish ñ)
/// - sz → S (voiceless "s" - this is the normal "s" sound!)
/// - ty → TY (palatalized t)
/// - zs → ZS (voiced postalveolar, like French "j")
///
/// ## S Sound (unique Hungarian feature!)
/// - s → SH (Hungarian S = "sh" sound, unlike most languages!)
///
/// ## Long Vowels (acute accent)
/// - á → A, é → E, í → I, ó → O, ú → U
///
/// ## Front Rounded Vowels
/// - ö → OE, ő → OE (short/long front rounded o)
/// - ü → UE, ű → UE (short/long front rounded u)
///
/// ## Geminate Digraphs
/// - ccs → CH, ssz → S, nny → NY, ggy → GY, etc.
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/hungarian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded hungarian/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Hungarian base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(
            !rules.is_empty(),
            "Hungarian base rules should not be empty"
        );
        assert!(
            rules.len() >= 50,
            "expected >=50 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_dzs_trigraph() {
        let rules = base();
        // dzs → DZS (trigraph, like English "j")
        let result = rules.apply("dzs");
        assert!(
            result.contains("DZS"),
            "dzs should become DZS, got: {}",
            result
        );
    }

    #[test]
    fn test_cs_digraph() {
        let rules = base();
        // cs → CH (like English "ch")
        let result = rules.apply("cs");
        assert!(
            result.contains("CH"),
            "cs should become CH, got: {}",
            result
        );
    }

    #[test]
    fn test_sz_digraph() {
        let rules = base();
        // sz → S (this is the normal "s" sound in Hungarian!)
        let result = rules.apply("sz");
        assert!(
            result.contains('S'),
            "sz should become S, got: {}",
            result
        );
    }

    #[test]
    fn test_zs_digraph() {
        let rules = base();
        // zs → ZS (like French "j")
        let result = rules.apply("zs");
        assert!(
            result.contains("ZS"),
            "zs should become ZS, got: {}",
            result
        );
    }

    #[test]
    fn test_gy_digraph() {
        let rules = base();
        // gy → GY (palatalized d)
        let result = rules.apply("gy");
        assert!(
            result.contains("GY"),
            "gy should become GY, got: {}",
            result
        );
    }

    #[test]
    fn test_ny_digraph() {
        let rules = base();
        // ny → NY (like Spanish ñ)
        let result = rules.apply("ny");
        assert!(
            result.contains("NY"),
            "ny should become NY, got: {}",
            result
        );
    }

    #[test]
    fn test_ly_digraph() {
        let rules = base();
        // ly → Y (palatal lateral)
        let result = rules.apply("ly");
        assert!(
            result.contains('Y'),
            "ly should become Y, got: {}",
            result
        );
    }

    #[test]
    fn test_ty_digraph() {
        let rules = base();
        // ty → TY (palatalized t)
        let result = rules.apply("ty");
        assert!(
            result.contains("TY"),
            "ty should become TY, got: {}",
            result
        );
    }

    #[test]
    fn test_s_to_sh() {
        let rules = base();
        // s → SH (unique Hungarian feature!)
        let result = rules.apply("s");
        assert!(
            result.contains("SH"),
            "s should become SH (unique Hungarian!), got: {}",
            result
        );
    }

    #[test]
    fn test_long_vowels() {
        let rules = base();
        // á → A
        let result = rules.apply("á");
        assert!(
            result.contains('A'),
            "á should become A, got: {}",
            result
        );
        // é → E
        let result = rules.apply("é");
        assert!(
            result.contains('E'),
            "é should become E, got: {}",
            result
        );
        // ó → O
        let result = rules.apply("ó");
        assert!(
            result.contains('O'),
            "ó should become O, got: {}",
            result
        );
    }

    #[test]
    fn test_front_rounded_vowels() {
        let rules = base();
        // ö → OE
        let result = rules.apply("ö");
        assert!(
            result.contains("OE"),
            "ö should become OE, got: {}",
            result
        );
        // ő → OE (double acute)
        let result = rules.apply("ő");
        assert!(
            result.contains("OE"),
            "ő should become OE, got: {}",
            result
        );
        // ü → UE
        let result = rules.apply("ü");
        assert!(
            result.contains("UE"),
            "ü should become UE, got: {}",
            result
        );
        // ű → UE (double acute)
        let result = rules.apply("ű");
        assert!(
            result.contains("UE"),
            "ű should become UE, got: {}",
            result
        );
    }

    #[test]
    fn test_geminate_digraphs() {
        let rules = base();
        // ccs → CH (geminate cs)
        let result = rules.apply("ccs");
        assert!(
            result.contains("CH"),
            "ccs should become CH, got: {}",
            result
        );
        // ssz → S (geminate sz)
        let result = rules.apply("ssz");
        assert!(
            result.contains('S'),
            "ssz should become S, got: {}",
            result
        );
    }

    #[test]
    fn test_word_magyar() {
        let rules = base();
        // magyar (Hungarian) - contains gy digraph
        let result = rules.apply("magyar");
        // Should contain m, a, GY, a, r
        assert!(
            result.contains("GY"),
            "magyar should contain GY, got: {}",
            result
        );
    }

    #[test]
    fn test_word_budapest() {
        let rules = base();
        // Budapest - contains s (→SH) and sz would be different
        let result = rules.apply("Budapest");
        // The 's' in Budapest should become SH
        assert!(
            result.contains("SH"),
            "Budapest should contain SH (from s), got: {}",
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
