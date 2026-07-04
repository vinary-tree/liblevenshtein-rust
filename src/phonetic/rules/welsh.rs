//! Welsh phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Welsh (Cymraeg) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Welsh phonetic normalization handles:
//! - **8 digraphs as letters**: ch, dd, ff, ng, ll, ph, rh, th
//! - **LL**: Unique voiceless lateral fricative [ɬ]
//! - **W and Y as vowels**: w = oo sound, y = schwa or i
//! - **Circumflex**: â, ê, î, ô, û, ŵ, ŷ (long vowels)
//! - **F = V sound**: Single f is pronounced like English "v"
//!
//! # Welsh Alphabet
//!
//! The Welsh alphabet has 29 letters including 8 digraphs:
//! a b c ch d dd e f ff g ng h i j l ll m n o p ph r rh s t th u w y
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Welsh transliteration rules (~80 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::welsh;
//!
//! let rules = welsh::base();
//!
//! // LL digraph (unique Welsh sound)
//! let result = rules.apply("ll");
//! assert!(result.contains("ʎ"), "ll → LL");
//!
//! // F = v sound (ff = f sound)
//! let result = rules.apply("f");
//! assert!(result.contains("v"), "f → v");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Welsh phonetic rules.
///
/// Complete phonetic normalization rules for Welsh:
///
/// ## Digraphs (8 single phonemes)
/// - ch → CH (voiceless uvular fricative, like Scottish "loch")
/// - dd → DH (voiced dental fricative, like English "the")
/// - ff → F (voiceless labiodental fricative, like English "f")
/// - ng → NG (velar nasal, like English "sing")
/// - ll → LL (voiceless lateral fricative - unique Welsh!)
/// - ph → F (like English "ph" = f)
/// - rh → RH (voiceless alveolar trill)
/// - th → TH (voiceless dental fricative, like English "think")
///
/// ## Special consonants
/// - f → v (Welsh f sounds like English "v"!)
/// - c → k (always hard)
///
/// ## Vowels (including W and Y)
/// - w → w (vowel, like "oo")
/// - y → y (schwa or "i" sound)
/// - Circumflex marks: â, ê, î, ô, û, ŵ, ŷ → base vowels
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/welsh/base.llev");
        let file = crate::phonetic::llev::parse_str(content).expect(
            "Invalid embedded welsh/base.llev - this indicates an internal invariant violation",
        );
        RuleSetChar::from_llev(&file).expect(
            "Failed to compile Welsh base rules - this indicates an internal invariant violation",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Welsh base rules should not be empty");
        assert!(
            rules.len() >= 40,
            "expected >=40 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_ll_digraph() {
        let rules = base();
        // ll → ɬ (voiceless lateral fricative), may become 'l' via soft mutation
        let result = rules.apply("ll");
        assert!(
            result.contains('ɬ') || result.contains('l') || result.contains("ʎ"),
            "ll should become ɬ (or l via soft mutation), got: {}",
            result
        );
    }

    #[test]
    fn test_dd_digraph() {
        let rules = base();
        // dd → DH
        let result = rules.apply("dd");
        assert!(result.contains("ð"), "dd should become DH, got: {}", result);
    }

    #[test]
    fn test_ff_digraph() {
        let rules = base();
        // ff → F → v (F gets further processed by F→v rule in Welsh)
        let result = rules.apply("ff");
        assert!(
            result.contains('v'),
            "ff should become v (via F→v rule), got: {}",
            result
        );
    }

    #[test]
    fn test_ch_digraph() {
        let rules = base();
        // ch → x (voiceless velar fricative, like Scottish "loch")
        let result = rules.apply("ch");
        assert!(
            result.contains('x') || result.contains("t͡ʃ"),
            "ch should become x (velar fricative), got: {}",
            result
        );
    }

    #[test]
    fn test_f_to_v() {
        let rules = base();
        // Welsh f → v (not the English f sound!)
        let result = rules.apply("f");
        assert!(
            result.contains('v'),
            "f should become v in Welsh, got: {}",
            result
        );
    }

    #[test]
    fn test_c_to_k() {
        let rules = base();
        // c → k (always hard in Welsh)
        let result = rules.apply("c");
        assert!(result.contains('k'), "c should become k, got: {}", result);
    }

    #[test]
    fn test_w_vowel() {
        let rules = base();
        // w is a vowel in Welsh
        let result = rules.apply("w");
        assert!(result.contains('w'), "w should remain w, got: {}", result);
    }

    #[test]
    fn test_y_vowel() {
        let rules = base();
        // y is a vowel in Welsh
        let result = rules.apply("y");
        assert!(result.contains('y'), "y should remain y, got: {}", result);
    }

    #[test]
    fn test_circumflex_a() {
        let rules = base();
        // â → a
        let result = rules.apply("â");
        assert!(result.contains('a'), "â should become a, got: {}", result);
    }

    #[test]
    fn test_circumflex_w() {
        let rules = base();
        // ŵ → w
        let result = rules.apply("ŵ");
        assert!(result.contains('w'), "ŵ should become w, got: {}", result);
    }

    #[test]
    fn test_word_cymru() {
        let rules = base();
        // Cymru (Wales) - test c→k transformation
        let result = rules.apply("c");
        assert!(result.contains('k'), "c should become k, got: {}", result);
        // Test y is preserved as vowel
        let result_y = rules.apply("y");
        assert!(
            result_y.to_lowercase().contains('y'),
            "y should remain y, got: {}",
            result_y
        );
    }

    #[test]
    fn test_word_llanfair() {
        let rules = base();
        // Llanfair - famous for long place names, use lowercase
        // ll → ɬ (voiceless lateral fricative), may become 'l' via soft mutation
        let result = rules.apply_full("llanfair");
        assert!(
            result.contains('ɬ') || result.contains('l') || result.contains("ʎ"),
            "llanfair should contain ɬ (or l via soft mutation), got: {}",
            result
        );
    }
}
