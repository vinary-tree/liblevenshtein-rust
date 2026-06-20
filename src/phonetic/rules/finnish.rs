//! Finnish phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Finnish (suomi) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Finnish phonetic normalization handles:
//! - **Nearly phonemic**: Finnish spelling is very regular
//! - **Front vowels**: ä → AE, ö → OE, y → Y
//! - **Vowel harmony**: Front (ä, ö, y) vs back (a, o, u)
//! - **Digraphs**: ng → NG (velar nasal)
//! - **Loanword consonants**: b→p, d→t, g→k, z→ts
//!
//! # Vowel Harmony
//!
//! Finnish has vowel harmony - back vowels (a, o, u) and front vowels
//! (ä, ö, y) don't typically appear in the same word (except compounds).
//! Neutral vowels (e, i) can appear with either group.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Finnish transliteration rules (~45 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::finnish;
//!
//! let rules = finnish::base();
//!
//! // Front vowels
//! let result = rules.apply("äiti");
//! assert!(result.contains("æ"), "ä → AE");
//!
//! // Ng digraph
//! let result = rules.apply("kengät");
//! assert!(result.contains("ŋ"), "ng → NG");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Finnish phonetic rules.
///
/// Complete phonetic normalization rules for Finnish:
///
/// ## Digraphs
/// - ng → NG (geminate velar nasal)
/// - nk → NK (velar nasal + k)
///
/// ## Front Vowels
/// - ä → AE (front open vowel)
/// - ö → OE (front rounded vowel)
/// - y → Y (front rounded vowel, like German ü)
///
/// ## Back Vowels
/// - a, e, i, o, u (standard Latin vowels)
///
/// ## Loanword Consonant Adaptations
/// - b → p (voiced → voiceless)
/// - d → t (voiced → voiceless, dialectal)
/// - g → k (voiced → voiceless)
/// - z → ts
/// - w → v
/// - x → ks
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/finnish/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded finnish/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Finnish base rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Finnish base rules should not be empty");
        assert!(
            rules.len() >= 15,
            "expected >=15 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_a_umlaut() {
        let rules = base();
        // ä → AE
        let result = rules.apply("ä");
        assert!(result.contains("æ"), "ä should become AE, got: {}", result);
    }

    #[test]
    fn test_o_umlaut() {
        let rules = base();
        // ö → OE
        let result = rules.apply("ö");
        assert!(result.contains("ø"), "ö should become OE, got: {}", result);
    }

    #[test]
    fn test_y_vowel() {
        let rules = base();
        // y → y (single y stays as y; only yy → yː)
        let result = rules.apply("y");
        assert!(
            result.contains('y'),
            "y should stay as y (or become y), got: {}",
            result
        );
    }

    #[test]
    fn test_ng_digraph() {
        let rules = base();
        // ng → ŋː (geminate velar nasal)
        let result = rules.apply("ng");
        assert!(
            result.contains("ŋː") || result.contains("ŋ"),
            "ng should become ŋː (or ŋ), got: {}",
            result
        );
    }

    #[test]
    fn test_nk_digraph() {
        let rules = base();
        // nk → ŋk (velar nasal + k)
        let result = rules.apply("nk");
        assert!(
            result.contains("ŋk"),
            "nk should become ŋk, got: {}",
            result
        );
    }

    #[test]
    fn test_b_to_p() {
        let rules = base();
        // b → p (loanwords)
        let result = rules.apply("b");
        assert!(result.contains('p'), "b should become p, got: {}", result);
    }

    #[test]
    fn test_z_to_ts() {
        let rules = base();
        // z → ts (loanwords)
        let result = rules.apply("z");
        assert!(result.contains("t͡s"), "z should become ts, got: {}", result);
    }

    #[test]
    fn test_word_suomi() {
        let rules = base();
        // suomi (Finland) - no transformation for o in Finnish rules
        let result = rules.apply("suomi");
        assert!(
            result.contains('s') && result.contains('u') && result.contains('o'),
            "suomi should contain s, u, o, got: {}",
            result
        );
    }

    #[test]
    fn test_word_helsinki() {
        let rules = base();
        // Helsinki
        let result = rules.apply("Helsinki");
        let upper = result.to_uppercase();
        assert!(
            upper.contains('H') && upper.contains('E') && upper.contains('L'),
            "Helsinki should contain h, e, l, got: {}",
            result
        );
    }

    #[test]
    fn test_word_aiti() {
        let rules = base();
        // äiti (mother)
        let result = rules.apply("äiti");
        assert!(
            result.contains("æ"),
            "äiti should contain AE, got: {}",
            result
        );
    }
}
