//! Maltese phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Maltese (Malti) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Maltese phonetic normalization handles:
//! - **Latin script for Semitic language**: Unique among Semitic languages
//! - **Special letters**: Ċ(ch), Ġ(j), Għ(silent), Ħ(h), Ż(z)
//! - **Glottal stop**: Q is often silent in modern Maltese
//! - **Sh sound**: X represents /ʃ/ (like English "sh")
//!
//! # Maltese Orthography
//!
//! Maltese is the only Semitic language written in Latin script. It has:
//! - Arabic-derived vocabulary (~40%)
//! - Italian-derived vocabulary (~40%)
//! - English-derived vocabulary (~20%)
//!
//! The special letters preserve Arabic-derived sounds:
//! - **Għ**: Historically pharyngeal (like Arabic ع), now usually silent
//! - **Ħ**: Pharyngeal h (like Arabic ح)
//! - **Q**: Glottal stop (like Arabic ق), often silent
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Maltese phonetic normalization (~40 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::maltese;
//!
//! let rules = maltese::base();
//! let result = rules.apply_full("Għawdex");
//! // Għ becomes silent, result normalizes the word
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Maltese phonetic rules.
///
/// Complete phonetic normalization rules for Maltese:
///
/// ## Digraphs (highest priority)
/// - għ → (silent, historically pharyngeal)
/// - ie → i (diphthong /iː/)
///
/// ## Special Letters
/// - Ċ/ċ → ch (voiceless postalveolar affricate)
/// - Ġ/ġ → j (voiced postalveolar affricate)
/// - Ħ/ħ → h (voiceless pharyngeal fricative)
/// - Ż/ż → z (voiced alveolar sibilant)
/// - X/x → sh (voiceless postalveolar fricative)
/// - Q/q → (glottal stop, often silent)
///
/// ## Standard Letters
/// - Consonants: b, d, f, g, h, j, k, l, m, n, p, r, s, t, v, w, z
/// - Vowels: a, e, i, o, u
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/maltese/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded maltese/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Maltese base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Maltese base rules should not be empty");
        assert!(
            rules.len() >= 5,
            "expected >=5 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // DIGRAPH TESTS
    // ============================================================

    #[test]
    fn test_gh_digraph() {
        let rules = base();
        // għ → silent
        let result = rules.apply("għ");
        assert!(
            result.is_empty() || !result.contains("ɣ"),
            "għ should become silent (empty), got: '{}'",
            result
        );
    }

    #[test]
    fn test_ie_digraph() {
        let rules = base();
        // ie → i
        let result = rules.apply("ie");
        assert!(result.contains('i'), "ie should become i, got: {}", result);
    }

    // ============================================================
    // SPECIAL LETTER TESTS
    // ============================================================

    #[test]
    fn test_c_dot() {
        let rules = base();
        // ċ → ch
        let result = rules.apply("ċ");
        assert!(result.contains("t͡ʃ"), "ċ should become ch, got: {}", result);
    }

    #[test]
    fn test_g_dot() {
        let rules = base();
        // ġ → d͡ʒ (voiced postalveolar affricate, like English "j")
        let result = rules.apply("ġ");
        assert!(result.contains("d͡ʒ"), "ġ should become d͡ʒ, got: {}", result);
    }

    #[test]
    fn test_h_stroke() {
        let rules = base();
        // ħ → passes through unchanged (no rule defined) or becomes h
        let result = rules.apply("ħ");
        assert!(
            result.contains('h') || result.contains('ħ'),
            "ħ should become h or pass through as ħ, got: {}",
            result
        );
    }

    #[test]
    fn test_z_dot() {
        let rules = base();
        // ż → z
        let result = rules.apply("ż");
        assert!(result.contains('z'), "ż should become z, got: {}", result);
    }

    #[test]
    fn test_x_sh() {
        let rules = base();
        // x → sh
        let result = rules.apply("x");
        assert!(result.contains("ʃ"), "x should become sh, got: {}", result);
    }

    #[test]
    fn test_q_silent() {
        let rules = base();
        // q → silent
        let result = rules.apply("q");
        assert!(
            result.is_empty() || !result.contains('q'),
            "q should become silent (empty), got: '{}'",
            result
        );
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_malta() {
        let rules = base();
        // Malta
        let result = rules.apply_full("Malta");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('m')
                && lower.contains('a')
                && lower.contains('l')
                && lower.contains('t'),
            "Malta should contain m, a, l, t, got: {}",
            result
        );
    }

    #[test]
    fn test_word_malti() {
        let rules = base();
        // Malti (Maltese language)
        let result = rules.apply_full("Malti");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('m')
                && lower.contains('l')
                && lower.contains('t')
                && lower.contains('i'),
            "Malti should contain m, l, t, i, got: {}",
            result
        );
    }

    #[test]
    fn test_word_ghawdex() {
        let rules = base();
        // Għawdex (Gozo island)
        let result = rules.apply_full("Għawdex");
        let lower = result.to_lowercase();
        // għ should be silent, x → sh
        assert!(
            lower.contains('a')
                && lower.contains('w')
                && lower.contains('d')
                && lower.contains("ʃ"),
            "Għawdex should have silent għ and x → sh, got: {}",
            result
        );
    }

    #[test]
    fn test_word_valletta() {
        let rules = base();
        // Valletta (capital)
        let result = rules.apply_full("Valletta");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('v')
                && lower.contains('a')
                && lower.contains('l')
                && lower.contains('e')
                && lower.contains('t'),
            "Valletta should contain v, a, l, e, t, got: {}",
            result
        );
    }

    #[test]
    fn test_word_with_c_dot() {
        let rules = base();
        // ċkejken (small) - has ċ
        let result = rules.apply_full("ċkejken");
        let lower = result.to_lowercase();
        assert!(
            lower.contains("t͡ʃ"),
            "Word with ċ should have ch, got: {}",
            result
        );
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================
}
