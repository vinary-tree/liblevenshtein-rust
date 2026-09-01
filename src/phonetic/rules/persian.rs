//! Persian (Farsi) phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Persian (فارسی / Farsi) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Persian phonetic normalization handles:
//! - **Arabic script** with 4 additional Persian letters
//! - **Persian additions**: پ(p), چ(ch), ژ(zh), گ(g)
//! - **Short vowels**: Usually not written
//! - **Silent letters**: ع (ayn) often silent in Persian
//! - **Word-final ه**: Can be silent or pronounced /e/
//!
//! # Persian vs Arabic
//!
//! Persian uses the Arabic script but with different pronunciation:
//! - ث, ذ, ظ → /z/ or /s/ (not emphatic like Arabic)
//! - ع → often silent or glottal stop (not pharyngeal)
//! - ق → /gh/ (not uvular stop like Arabic)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Persian phonetic normalization (~46 rules)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::rules::persian;
//!
//! let rules = persian::base();
//! let result = rules.apply_full("فارسی");
//! // Result contains Latin phonetic representation
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Persian phonetic rules.
///
/// Complete phonetic normalization rules for Persian:
///
/// ## Persian-Specific Letters
/// - پ → p (not in Arabic)
/// - چ → ch (not in Arabic)
/// - ژ → zh (not in Arabic)
/// - گ → g (not in Arabic)
///
/// ## Persian Pronunciation Differences
/// - ث → s (not th like Arabic)
/// - ذ → z (not th like Arabic)
/// - ص → s (not emphatic)
/// - ض → z (not emphatic)
/// - ط → t (not emphatic)
/// - ظ → z (not emphatic)
/// - ع → a (often silent in Persian)
/// - غ, ق → gh (uvular fricative)
///
/// ## Standard Consonants
/// - ب(b), ت(t), ج(j), ح(h), خ(kh), د(d), ر(r), ز(z)
/// - س(s), ش(sh), ف(f), ک(k), ل(l), م(m), ن(n)
///
/// ## Vowels
/// - ا, آ → a (long vowel)
/// - و → v/u/o
/// - ی → y/i
/// - ه → h/e
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/persian/base.llev");
        let file = crate::phonetic::llev::parse_str(content).expect(
            "Invalid embedded persian/base.llev - this indicates an internal invariant violation",
        );
        RuleSetChar::from_llev(&file).expect(
            "Failed to compile Persian base rules - this indicates an internal invariant violation",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Persian base rules should not be empty");
        assert!(
            rules.len() >= 35,
            "expected >=35 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // PERSIAN-SPECIFIC LETTER TESTS
    // ============================================================

    #[test]
    fn test_pe() {
        let rules = base();
        // پ → p
        let result = rules.apply("پ");
        assert!(result.contains('p'), "پ should become p, got: {}", result);
    }

    #[test]
    fn test_che() {
        let rules = base();
        // چ → ch
        let result = rules.apply("چ");
        assert!(result.contains("t͡ʃ"), "چ should become ch, got: {}", result);
    }

    #[test]
    fn test_zhe() {
        let rules = base();
        // ژ → zh
        let result = rules.apply("ژ");
        assert!(result.contains("ʒ"), "ژ should become zh, got: {}", result);
    }

    #[test]
    fn test_gaf() {
        let rules = base();
        // گ → g
        let result = rules.apply("گ");
        assert!(result.contains('ɡ'), "گ should become g, got: {}", result);
    }

    // ============================================================
    // PERSIAN PRONUNCIATION DIFFERENCE TESTS
    // ============================================================

    #[test]
    fn test_se() {
        let rules = base();
        // ث → s (not th)
        let result = rules.apply("ث");
        assert!(
            result.contains('s'),
            "ث should become s in Persian, got: {}",
            result
        );
    }

    #[test]
    fn test_zal() {
        let rules = base();
        // ذ → z (not th)
        let result = rules.apply("ذ");
        assert!(
            result.contains('z'),
            "ذ should become z in Persian, got: {}",
            result
        );
    }

    #[test]
    fn test_ayn() {
        let rules = base();
        // ع → a (often silent)
        let result = rules.apply("ع");
        assert!(
            result.contains('a'),
            "ع should become a in Persian, got: {}",
            result
        );
    }

    #[test]
    fn test_ghayn() {
        let rules = base();
        // غ → gh
        let result = rules.apply("غ");
        assert!(result.contains("ɣ"), "غ should become gh, got: {}", result);
    }

    #[test]
    fn test_qaf() {
        let rules = base();
        // ق → gh (Persian pronunciation)
        let result = rules.apply("ق");
        assert!(
            result.contains("ɣ"),
            "ق should become gh in Persian, got: {}",
            result
        );
    }

    #[test]
    fn test_khe() {
        let rules = base();
        // خ → kh
        let result = rules.apply("خ");
        assert!(result.contains("x"), "خ should become kh, got: {}", result);
    }

    #[test]
    fn test_shin() {
        let rules = base();
        // ش → sh
        let result = rules.apply("ش");
        assert!(result.contains("ʃ"), "ش should become sh, got: {}", result);
    }

    // ============================================================
    // STANDARD CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_be() {
        let rules = base();
        let result = rules.apply("ب");
        assert!(result.contains('b'), "ب should become b, got: {}", result);
    }

    #[test]
    fn test_te() {
        let rules = base();
        let result = rules.apply("ت");
        assert!(result.contains('t'), "ت should become t, got: {}", result);
    }

    #[test]
    fn test_jim() {
        let rules = base();
        let result = rules.apply("ج");
        // Persian ج -> j (ASCII j, not IPA ʝ)
        assert!(
            result.contains('j') || result.contains('ʝ') || result.contains("d͡ʒ"),
            "ج should become j, got: {}",
            result
        );
    }

    #[test]
    fn test_dal() {
        let rules = base();
        let result = rules.apply("د");
        assert!(result.contains('d'), "د should become d, got: {}", result);
    }

    #[test]
    fn test_sin() {
        let rules = base();
        let result = rules.apply("س");
        assert!(result.contains('s'), "س should become s, got: {}", result);
    }

    #[test]
    fn test_fe() {
        let rules = base();
        let result = rules.apply("ف");
        assert!(result.contains('f'), "ف should become f, got: {}", result);
    }

    #[test]
    fn test_kaf() {
        let rules = base();
        let result = rules.apply("ک");
        assert!(result.contains('k'), "ک should become k, got: {}", result);
    }

    // ============================================================
    // VOWEL TESTS
    // ============================================================

    #[test]
    fn test_alef() {
        let rules = base();
        let result = rules.apply("ا");
        assert!(result.contains('a'), "ا should become a, got: {}", result);
    }

    #[test]
    fn test_alef_madda() {
        let rules = base();
        let result = rules.apply("آ");
        assert!(result.contains('a'), "آ should become a, got: {}", result);
    }

    #[test]
    fn test_vav() {
        let rules = base();
        let result = rules.apply("و");
        assert!(result.contains('v'), "و should become v, got: {}", result);
    }

    #[test]
    fn test_ye() {
        let rules = base();
        let result = rules.apply("ی");
        assert!(result.contains('y'), "ی should become y, got: {}", result);
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_farsi() {
        let rules = base();
        // فارسی (Persian)
        let result = rules.apply_full("فارسی");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('f')
                && lower.contains('a')
                && lower.contains('r')
                && lower.contains('s')
                && lower.contains('y'),
            "فارسی should contain f, a, r, s, y, got: {}",
            result
        );
    }

    #[test]
    fn test_word_iran() {
        let rules = base();
        // ایران (Iran)
        let result = rules.apply_full("ایران");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('a')
                && lower.contains('y')
                && lower.contains('r')
                && lower.contains('n'),
            "ایران should contain a, y, r, n, got: {}",
            result
        );
    }

    #[test]
    fn test_word_tehran() {
        let rules = base();
        // تهران (Tehran)
        let result = rules.apply_full("تهران");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('t')
                && lower.contains('h')
                && lower.contains('r')
                && lower.contains('a')
                && lower.contains('n'),
            "تهران should contain t, h, r, a, n, got: {}",
            result
        );
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================
}
