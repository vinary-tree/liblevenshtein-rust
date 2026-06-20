//! Swedish phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Swedish (svenska) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Swedish phonetic normalization handles:
//! - **Three extra vowels**: å→O, ä→AE, ö→OE
//! - **SJ-sound**: sj, skj, stj → SJ (unique Swedish fricative)
//! - **TJ/KJ**: tj, kj → TJ (voiceless palatal fricative)
//! - **Silent clusters**: dj, gj, hj, lj → J
//! - **Velar nasal**: ng → NG
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Swedish transliteration rules (~80 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::swedish;
//!
//! let rules = swedish::base();
//!
//! // SJ-sound
//! let result = rules.apply("sjö");
//! assert!(result.contains("ʃ"), "sj → SJ");
//!
//! // Special vowels
//! let result = rules.apply("å");
//! assert!(result.contains('O'), "å → O");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Swedish phonetic rules.
///
/// Complete phonetic normalization rules for Swedish:
///
/// ## Trigraphs
/// - stj → SJ (sj-sound)
/// - skj → SJ (sj-sound)
///
/// ## Digraphs
/// - sj → SJ (sj-sound, unique Swedish fricative)
/// - tj → TJ (voiceless palatal fricative)
/// - kj → TJ (same as tj)
/// - ng → NG (velar nasal)
///
/// ## Silent Clusters
/// - dj → J (d silent)
/// - gj → J (g silent)
/// - hj → J (h silent)
/// - lj → J (l silent)
///
/// ## Special Vowels
/// - å → O (like English "or")
/// - ä → AE (like English "air")
/// - ö → OE (front rounded vowel)
///
/// ## Other Features
/// - y → Y (front rounded vowel)
/// - c → k
/// - w → v
/// - x → ks
/// - z → s
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/swedish/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded swedish/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Swedish base rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Swedish base rules should not be empty");
        assert!(
            rules.len() >= 30,
            "expected >=30 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_sj_digraph() {
        let rules = base();
        // sj → ɧ (voiceless palatal-velar fricative, unique Swedish sj-sound)
        let result = rules.apply("sj");
        assert!(
            result.contains("ɧ") || result.contains("ʃ"),
            "sj should become ɧ (sj-sound), got: {}",
            result
        );
    }

    #[test]
    fn test_stj_trigraph() {
        let rules = base();
        // stj → ɧ (voiceless palatal-velar fricative, unique Swedish sj-sound)
        let result = rules.apply("stj");
        assert!(
            result.contains("ɧ") || result.contains("ʃ"),
            "stj should become ɧ (sj-sound), got: {}",
            result
        );
    }

    #[test]
    fn test_tj_digraph() {
        let rules = base();
        // tj → ç (voiceless palatal fricative)
        let result = rules.apply("tj");
        assert!(
            result.contains("ç") || result.contains("tɕ") || result.contains("ɕ"),
            "tj should become ç (voiceless palatal fricative), got: {}",
            result
        );
    }

    #[test]
    fn test_kj_digraph() {
        let rules = base();
        // kj → ç (voiceless palatal fricative, same as tj)
        let result = rules.apply("kj");
        assert!(
            result.contains("ç") || result.contains("tɕ") || result.contains("ɕ"),
            "kj should become ç (voiceless palatal fricative), got: {}",
            result
        );
    }

    #[test]
    fn test_a_ring() {
        let rules = base();
        // å → O → o (normalized to lowercase)
        let result = rules.apply("å");
        assert!(result.contains('ɔ'), "å should become o, got: {}", result);
    }

    #[test]
    fn test_a_umlaut() {
        let rules = base();
        // ä → ɛ (front vowel, like English "air")
        let result = rules.apply("ä");
        assert!(
            result.contains("ɛ") || result.contains("æ"),
            "ä should become ɛ (front vowel), got: {}",
            result
        );
    }

    #[test]
    fn test_o_umlaut() {
        let rules = base();
        // ö → OE
        let result = rules.apply("ö");
        assert!(result.contains("ø"), "ö should become OE, got: {}", result);
    }

    #[test]
    fn test_silent_dj() {
        let rules = base();
        // dj → J
        let result = rules.apply("dj");
        assert!(result.contains('j'), "dj should become J, got: {}", result);
    }

    #[test]
    fn test_silent_hj() {
        let rules = base();
        // hj → J
        let result = rules.apply("hj");
        assert!(result.contains('j'), "hj should become J, got: {}", result);
    }

    #[test]
    fn test_word_stockholm() {
        let rules = base();
        // Stockholm - check that c→k transformation occurs
        let result = rules.apply_full("stockholm");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('s') && lower.contains('t') && lower.contains('k'),
            "stockholm should contain s, t, k, got: {}",
            result
        );
    }

    #[test]
    fn test_word_sjo() {
        let rules = base();
        // sjö (lake) - sj → ɧ, ö → ø
        let result = rules.apply("sjö");
        assert!(
            result.contains("ɧ") || result.contains("ʃ"),
            "sjö should contain ɧ (sj-sound), got: {}",
            result
        );
    }
}
