//! Basque phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Basque (Euskara) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Basque phonetic normalization handles:
//! - **Digraphs**: tx→CH, ts→TS, tz→TZ, tt→TT, dd→DD, rr→RR
//! - **X = "sh" sound**: Unlike most European languages
//! - **Z = "s" sound**: NOT like English "z"!
//! - **Palatalized consonants**: tt, dd
//! - **No F, V**: In native Basque words
//!
//! # Language Isolate
//!
//! Basque is a language isolate - unrelated to any other known language.
//! It has unique phonological features not found in neighboring Spanish or French.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Basque transliteration rules (~45 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::basque;
//!
//! let rules = basque::base();
//!
//! // X = "sh" sound
//! let result = rules.apply("etxe");
//! assert!(result.contains("t͡ʃ"), "tx → t͡ʃ");
//!
//! // Z = "s" sound (not z!)
//! let result = rules.apply("zu");
//! assert!(result.contains("s̻") || result.contains('s'), "z → s̻");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Basque phonetic rules.
///
/// Complete phonetic normalization rules for Basque:
///
/// ## Digraphs
/// - tx → CH (like English "ch")
/// - ts → TS (voiceless alveolar affricate)
/// - tz → TZ (laminal alveolar affricate)
/// - tt → TT (palatalized t)
/// - dd → DD (palatalized d)
/// - rr → RR (trilled r)
///
/// ## Special Consonants
/// - x → SH (like English "sh")
/// - z → S (like "s", NOT English "z"!)
/// - ñ → NY (palatal nasal)
///
/// ## Loanword Adaptations
/// - v → b (v rare in Basque)
/// - f → f (loanwords only)
/// - w → u
/// - y → i
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/basque/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded basque/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Basque base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Basque base rules should not be empty");
        assert!(
            rules.len() >= 10,
            "expected >=10 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_tx_digraph() {
        let rules = base();
        // tx → CH
        let result = rules.apply("tx");
        assert!(
            result.contains("t͡ʃ"),
            "tx should become CH, got: {}",
            result
        );
    }

    #[test]
    fn test_ts_digraph() {
        let rules = base();
        // ts → TS
        let result = rules.apply("ts");
        assert!(
            result.contains("t͡s"),
            "ts should become TS, got: {}",
            result
        );
    }

    #[test]
    fn test_tz_digraph() {
        let rules = base();
        // tz → t͡s̻ (laminal alveolar affricate)
        let result = rules.apply("tz");
        assert!(
            result.contains("t͡s̻") || result.contains("t͡s"),
            "tz should become t͡s̻ or t͡s, got: {}",
            result
        );
    }

    #[test]
    fn test_rr_digraph() {
        let rules = base();
        // rr → r (trilled r) which may then become ɾ (alveolar tap) via further rule application
        let result = rules.apply("rr");
        assert!(
            result.contains("r") || result.contains("ɾ"),
            "rr should become r or ɾ (trilled/tapped r), got: {}",
            result
        );
    }

    #[test]
    fn test_x_to_sh() {
        let rules = base();
        // x → SH
        let result = rules.apply("x");
        assert!(
            result.contains("ʃ"),
            "x should become SH, got: {}",
            result
        );
    }

    #[test]
    fn test_z_to_s() {
        let rules = base();
        // z → s̻ (laminal alveolar, NOT English "z"!)
        let result = rules.apply("z");
        assert!(
            result.contains("s̻") || result.contains('s'),
            "z should become s̻ or s (not z!), got: {}",
            result
        );
    }

    #[test]
    fn test_n_tilde() {
        let rules = base();
        // ñ → NY
        let result = rules.apply("ñ");
        assert!(
            result.contains("ɲ"),
            "ñ should become NY, got: {}",
            result
        );
    }

    #[test]
    fn test_word_etxe() {
        let rules = base();
        // etxe (house)
        let result = rules.apply("etxe");
        assert!(
            result.contains("t͡ʃ"),
            "etxe should contain CH (from tx), got: {}",
            result
        );
    }

    #[test]
    fn test_word_euskara() {
        let rules = base();
        // Euskara (Basque language)
        let result = rules.apply("euskara");
        assert!(
            result.contains('e') && result.contains('u') && result.contains('a'),
            "euskara should contain e, u, a, got: {}",
            result
        );
    }

    #[test]
    fn test_word_bilbo() {
        let rules = base();
        // Bilbo (Bilbao)
        let result = rules.apply("bilbo");
        assert!(
            result.contains('b') && result.contains('i') && result.contains('l'),
            "bilbo should contain b, i, l, got: {}",
            result
        );
    }

}
