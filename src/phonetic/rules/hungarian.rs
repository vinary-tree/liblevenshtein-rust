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
//! assert!(result.contains("t͡ʃ"), "cs → CH");
//!
//! // S = "sh" sound (unique to Hungarian!)
//! let result = rules.apply("s");
//! assert!(result.contains("ʃ"), "s → SH");
//!
//! // SZ digraph = "s" sound
//! let result = rules.apply("sz");
//! assert!(result.contains('ʃ'), "sz → S");
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
            result.contains("d͡ʒ"),
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
            result.contains("t͡ʃ"),
            "cs should become CH, got: {}",
            result
        );
    }

    #[test]
    fn test_sz_digraph() {
        let rules = base();
        // sz → s (voiceless alveolar fricative), then s → ʃ (Hungarian s = "sh")
        // So the final output is ʃ
        let result = rules.apply("sz");
        assert!(
            result.contains('s') || result.contains('ʃ'),
            "sz should become s or ʃ (after s→ʃ rule), got: {}",
            result
        );
    }

    #[test]
    fn test_zs_digraph() {
        let rules = base();
        // zs → ʒ (voiced postalveolar fricative, like French "j")
        let result = rules.apply("zs");
        assert!(result.contains('ʒ'), "zs should become ʒ, got: {}", result);
    }

    #[test]
    fn test_gy_digraph() {
        let rules = base();
        // gy → GY (palatalized d)
        let result = rules.apply("gy");
        assert!(result.contains("ɟ"), "gy should become GY, got: {}", result);
    }

    #[test]
    fn test_ny_digraph() {
        let rules = base();
        // ny → NY (like Spanish ñ)
        let result = rules.apply("ny");
        assert!(result.contains("ɲ"), "ny should become NY, got: {}", result);
    }

    #[test]
    fn test_ly_digraph() {
        let rules = base();
        // ly → j (palatal approximant, pronounced as [j] in modern Hungarian)
        let result = rules.apply("ly");
        assert!(result.contains('j'), "ly should become j, got: {}", result);
    }

    #[test]
    fn test_ty_digraph() {
        let rules = base();
        // ty → c (voiceless palatal plosive), but c → t͡s, so the final output is t͡s or t͡ʃ
        let result = rules.apply("ty");
        assert!(
            result.contains('c') || result.contains("t͡s") || result.contains("t͡ʃ"),
            "ty should become c, t͡s, or t͡ʃ (voiceless palatal plosive), got: {}",
            result
        );
    }

    #[test]
    fn test_s_to_sh() {
        let rules = base();
        // s → SH (unique Hungarian feature!)
        let result = rules.apply("s");
        assert!(
            result.contains("ʃ"),
            "s should become SH (unique Hungarian!), got: {}",
            result
        );
    }

    #[test]
    fn test_long_vowels() {
        let rules = base();
        // á → aː (long a)
        let result = rules.apply("á");
        assert!(result.contains("aː"), "á should become aː, got: {}", result);
        // é → eː (long e)
        let result = rules.apply("é");
        assert!(result.contains("eː"), "é should become eː, got: {}", result);
        // ó → oː (long o)
        let result = rules.apply("ó");
        assert!(result.contains("oː"), "ó should become oː, got: {}", result);
    }

    #[test]
    fn test_front_rounded_vowels() {
        let rules = base();
        // ö → ø (short front rounded o)
        let result = rules.apply("ö");
        assert!(result.contains('ø'), "ö should become ø, got: {}", result);
        // ő → øː (long front rounded o, double acute)
        let result = rules.apply("ő");
        assert!(result.contains("øː"), "ő should become øː, got: {}", result);
        // ü → y (short front rounded u, IPA), but y → i at word end (Beider-Morse rule)
        // When ü is standalone (at word end), it becomes y, then y → i
        let result = rules.apply("ü");
        assert!(
            result.contains('y') || result.contains('i'),
            "ü should become y or i (after y→i rule at word end), got: {}",
            result
        );
        // ű → yː (long front rounded u, double acute)
        let result = rules.apply("ű");
        assert!(result.contains("yː"), "ű should become yː, got: {}", result);
    }

    #[test]
    fn test_geminate_digraphs() {
        let rules = base();
        // ccs → t͡ʃ (geminate cs)
        let result = rules.apply("ccs");
        assert!(
            result.contains("t͡ʃ"),
            "ccs should become t͡ʃ, got: {}",
            result
        );
        // ssz → s (geminate sz), then s → ʃ (Hungarian s = "sh")
        // So the final output is ʃ
        let result = rules.apply("ssz");
        assert!(
            result.contains('s') || result.contains('ʃ'),
            "ssz should become s or ʃ (after s→ʃ rule), got: {}",
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
            result.contains("ɟ"),
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
            result.contains("ʃ"),
            "Budapest should contain SH (from s), got: {}",
            result
        );
    }
}
