//! Irish phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Irish (Gaeilge) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Irish phonetic normalization handles:
//! - **Séimhiú (lenition)**: bh→v, ch→CH, dh→GH, fh→(silent!), gh→GH, mh→v, ph→f, sh→h, th→h
//! - **Silent FH**: Lenited f is completely silent in Irish!
//! - **Fadas (acute accent)**: á, é, í, ó, ú (long vowels)
//!
//! # Lenition (Séimhiú)
//!
//! In Irish, adding 'h' after certain consonants changes their pronunciation:
//! - bh, mh → v sound
//! - ch, dh, gh → guttural sounds
//! - fh → completely silent!
//! - ph → f sound
//! - sh, th → h sound
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Irish transliteration rules (~73 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::irish;
//!
//! let rules = irish::base();
//!
//! // Lenition - bh becomes v
//! let result = rules.apply("bh");
//! assert!(result.contains("v"), "bh → v");
//!
//! // Silent fh
//! let result = rules.apply("fh");
//! assert!(result.is_empty(), "fh → (silent)");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Irish phonetic rules.
///
/// Complete phonetic normalization rules for Irish:
///
/// ## Séimhiú (Lenition)
/// - bh → v (like English "v")
/// - ch → CH (voiceless velar fricative, like Scottish "loch")
/// - dh → GH (voiced velar fricative)
/// - fh → (silent!) - completely silent in Irish
/// - gh → GH (voiced velar fricative)
/// - mh → v (like English "v")
/// - ph → f (like English "f")
/// - sh → h (like English "h")
/// - th → h (like English "h")
///
/// ## Fadas (Long Vowels)
/// - á, é, í, ó, ú → short equivalents (a, e, i, o, u)
///
/// ## Standard Consonants
/// - c → k (always hard)
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/irish/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded irish/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Irish base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Irish base rules should not be empty");
        assert!(
            rules.len() >= 20,
            "expected >=20 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_bh_lenition() {
        let rules = base();
        // bh → v
        let result = rules.apply("bh");
        assert!(
            result.contains('v'),
            "bh should become v, got: {}",
            result
        );
    }

    #[test]
    fn test_ch_lenition() {
        let rules = base();
        // ch → x (voiceless velar fricative, like Scottish "loch")
        let result = rules.apply("ch");
        assert!(
            result.contains('x'),
            "ch should become x, got: {}",
            result
        );
    }

    #[test]
    fn test_fh_silent() {
        let rules = base();
        // fh → (silent!) - should produce empty or very minimal output
        let result = rules.apply("fh");
        assert!(
            result.is_empty(),
            "fh should be silent (empty), got: '{}'",
            result
        );
    }

    #[test]
    fn test_mh_lenition() {
        let rules = base();
        // mh → v
        let result = rules.apply("mh");
        assert!(
            result.contains('v'),
            "mh should become v, got: {}",
            result
        );
    }

    #[test]
    fn test_ph_lenition() {
        let rules = base();
        // ph → f
        let result = rules.apply("ph");
        assert!(
            result.contains('f'),
            "ph should become f, got: {}",
            result
        );
    }

    #[test]
    fn test_sh_lenition() {
        let rules = base();
        // sh → h
        let result = rules.apply("sh");
        assert!(
            result.contains('h'),
            "sh should become h, got: {}",
            result
        );
    }

    #[test]
    fn test_th_lenition() {
        let rules = base();
        // th → h
        let result = rules.apply("th");
        assert!(
            result.contains('h'),
            "th should become h, got: {}",
            result
        );
    }

    #[test]
    fn test_a_fada() {
        let rules = base();
        // á → a
        let result = rules.apply("á");
        assert!(
            result.contains('a'),
            "á should become a, got: {}",
            result
        );
    }

    #[test]
    fn test_o_fada() {
        let rules = base();
        // ó → o
        let result = rules.apply("ó");
        assert!(
            result.contains('o'),
            "ó should become o, got: {}",
            result
        );
    }

    #[test]
    fn test_c_to_k() {
        let rules = base();
        // c → k
        let result = rules.apply("c");
        assert!(
            result.contains('k'),
            "c should become k, got: {}",
            result
        );
    }

    #[test]
    fn test_word_eire() {
        let rules = base();
        // Éire (Ireland) - use lowercase éire
        let result = rules.apply_full("éire");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('e') && lower.contains('i') && lower.contains('r'),
            "éire should contain e, i, r, got: {}",
            result
        );
    }

    #[test]
    fn test_word_baile() {
        let rules = base();
        // Baile (town) - use lowercase
        let result = rules.apply_full("baile");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('b') && lower.contains('a') && lower.contains('l'),
            "baile should contain b, a, l, got: {}",
            result
        );
    }

    #[test]
    fn test_word_slainte() {
        let rules = base();
        // Sláinte (health/cheers) - use lowercase
        let result = rules.apply_full("sláinte");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('s') && lower.contains('l') && lower.contains('a'),
            "sláinte should contain s, l, a, got: {}",
            result
        );
    }

}
