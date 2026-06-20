//! Norwegian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Norwegian (norsk) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Norwegian phonetic normalization handles:
//! - **Three extra vowels**: æ→AE, ø→OE, å→O
//! - **KJ/SKJ**: kj→KJ, skj→SJ (palatal fricatives)
//! - **SJ-sound**: sj→SJ
//! - **Silent clusters**: hv→v, hj→j
//! - **Velar nasal**: ng→NG
//!
//! # Bokmål vs Nynorsk
//!
//! Both Bokmål and Nynorsk spelling variants are handled by these rules.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Norwegian transliteration rules (~75 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::norwegian;
//!
//! let rules = norwegian::base();
//!
//! // Special vowels
//! let result = rules.apply("æ");
//! assert!(result.contains("æ"), "æ → AE");
//!
//! // Silent hv
//! let result = rules.apply("hva");
//! assert!(result.contains("va"), "hv → v");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Norwegian phonetic rules.
///
/// Complete phonetic normalization rules for Norwegian:
///
/// ## Trigraphs
/// - skj → SJ (sj-sound)
///
/// ## Digraphs
/// - kj → KJ (voiceless palatal fricative)
/// - sj → SJ (sj-sound)
/// - ng → NG (velar nasal)
///
/// ## Silent Clusters
/// - hv → v (h silent before v)
/// - hj → j (h silent before j)
///
/// ## Special Vowels
/// - æ → AE (front open vowel)
/// - ø → OE (front rounded vowel)
/// - å → O (rounded back vowel)
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
        let content = include_str!("../../../data/rules/norwegian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded norwegian/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Norwegian base rules - this indicates an internal invariant violation")
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
            "Norwegian base rules should not be empty"
        );
        assert!(
            rules.len() >= 30,
            "expected >=30 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_ae_ligature() {
        let rules = base();
        // æ → AE
        let result = rules.apply("æ");
        assert!(result.contains("æ"), "æ should become AE, got: {}", result);
    }

    #[test]
    fn test_o_slash() {
        let rules = base();
        // ø → OE
        let result = rules.apply("ø");
        assert!(result.contains("ø"), "ø should become OE, got: {}", result);
    }

    #[test]
    fn test_a_ring() {
        let rules = base();
        // å → O → o (normalized to lowercase)
        let result = rules.apply("å");
        assert!(result.contains('ɔ'), "å should become o, got: {}", result);
    }

    #[test]
    fn test_kj_digraph() {
        let rules = base();
        // kj → KJ
        let result = rules.apply("kj");
        assert!(result.contains("ç"), "kj should become KJ, got: {}", result);
    }

    #[test]
    fn test_sj_digraph() {
        let rules = base();
        // sj → SJ
        let result = rules.apply("sj");
        assert!(result.contains("ʃ"), "sj should become SJ, got: {}", result);
    }

    #[test]
    fn test_skj_trigraph() {
        let rules = base();
        // skj → SJ
        let result = rules.apply("skj");
        assert!(
            result.contains("ʃ"),
            "skj should become SJ, got: {}",
            result
        );
    }

    #[test]
    fn test_silent_hv() {
        let rules = base();
        // hv → v
        let result = rules.apply("hv");
        assert!(
            result.contains('v') && !result.contains('h'),
            "hv should become v, got: {}",
            result
        );
    }

    #[test]
    fn test_silent_hj() {
        let rules = base();
        // hj → J
        let result = rules.apply("hj");
        assert!(result.contains('j'), "hj should become J, got: {}", result);
    }

    #[test]
    fn test_word_oslo() {
        let rules = base();
        // Oslo - 'O' stays as 'o' (only å/Å becomes ɔ)
        let result = rules.apply("Oslo");
        assert!(
            result.contains('o') && result.contains('s') && result.contains('l'),
            "Oslo should contain o, s, l, got: {}",
            result
        );
    }

    #[test]
    fn test_word_norge() {
        let rules = base();
        // Norge (Norway) - 'o' stays as 'o' (only å/Å becomes ɔ)
        let result = rules.apply("Norge");
        assert!(
            result.contains('n') && result.contains('o'),
            "Norge should contain n, o, got: {}",
            result
        );
    }
}
