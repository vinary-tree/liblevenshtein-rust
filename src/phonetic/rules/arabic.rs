//! Arabic phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Arabic (العربية) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Arabic phonetic normalization handles:
//! - **Basic consonants**: 28 Arabic letters mapped to Latin equivalents
//! - **Emphatic consonants**: ص→S, ض→D, ط→T, ظ→Z (pharyngealized)
//! - **Pharyngeal consonants**: ح→H, ع→E
//! - **Interdentals**: ث→TH, ذ→DH
//! - **Velar fricatives**: خ→KH, غ→GH
//! - **Postalveolar**: ش→SH
//! - **Hamza variants**: ء, أ, إ, ؤ, ئ (glottal stop carriers)
//! - **Diacritics**: fatha→a, kasra→i, damma→u, sukun/shadda→∅
//! - **Numerals**: Arabic-Indic digits (٠-٩ → 0-9)
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers for Arabic-specific sounds:
//! - TH = voiceless dental fricative (ث)
//! - DH = voiced dental fricative (ذ)
//! - SH = postalveolar fricative (ش)
//! - KH = voiceless velar fricative (خ)
//! - GH = voiced velar fricative (غ)
//! - H = pharyngeal fricative (ح)
//! - E = voiced pharyngeal (ع)
//! - S, D, T, Z = emphatic consonants (ص, ض, ط, ظ)
//! - A = long a (آ)
//!
//! # Arabic Script
//!
//! Arabic is an abjad (consonantal alphabet) with 28 letters, written
//! right-to-left. Vowels are typically:
//! - **Short vowels**: Unmarked or shown with diacritics (harakat)
//! - **Long vowels**: ا (aa), و (uu), ي (ii)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Arabic phonetic rules (~55 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::arabic;
//!
//! let rules = arabic::base();
//!
//! // Basic consonants
//! let result = rules.apply("ب");
//! assert!(result.contains('b'), "ب → b");
//!
//! // Emphatic consonants
//! let result = rules.apply("ص");
//! assert!(result.contains('ʃ'), "ص → S");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Arabic base phonetic rules.
///
/// Complete phonetic normalization rules for Arabic:
///
/// ## Basic Consonants (weight 0.05)
/// - ا→a, ب→b, ت→t, ج→j, د→d, ر→r, ز→z, س→s, ف→f, ق→q, ك→k, ل→l, م→m, ن→n, ه→h, و→w, ي→y
///
/// ## Special Consonants (weight 0.05)
/// - ث→TH, ذ→DH (interdentals)
/// - ح→H, ع→E (pharyngeals)
/// - خ→KH, غ→GH (velar fricatives)
/// - ش→SH (postalveolar)
///
/// ## Emphatic Consonants (weight 0.05)
/// - ص→S, ض→D, ط→T, ظ→Z
///
/// ## Hamza Variants (weight 0.05)
/// - ء→', أ→a, إ→i, ؤ→u, ئ→i
///
/// ## Special Forms (weight 0.05)
/// - آ→A (alif madda), ة→a (ta marbuta), ى→a (alif maqsura)
///
/// ## Diacritics (weight 0.1)
/// - Fatha→a, Kasra→i, Damma→u
/// - Sukun→∅, Shadda→∅
/// - Tanwin: ً→an, ٌ→un, ٍ→in
///
/// ## Numerals (weight 0.1)
/// - Arabic-Indic digits: ٠-٩ → 0-9
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/arabic/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded arabic/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Arabic base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Arabic base rules should not be empty");
        assert!(
            rules.len() > 50,
            "expected >50 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_basic_consonants() {
        let rules = base();
        // ب → b
        let result = rules.apply("ب");
        assert!(
            result.contains('b'),
            "ب should become b, got: {}",
            result
        );
        // ت → t
        let result = rules.apply("ت");
        assert!(
            result.contains('t'),
            "ت should become t, got: {}",
            result
        );
        // س → s
        let result = rules.apply("س");
        assert!(
            result.contains('s'),
            "س should become s, got: {}",
            result
        );
        // ك → k
        let result = rules.apply("ك");
        assert!(
            result.contains('k'),
            "ك should become k, got: {}",
            result
        );
    }

    #[test]
    fn test_special_consonants() {
        let rules = base();
        // ش → SH
        let result = rules.apply("ش");
        assert!(
            result.contains("ʃ"),
            "ش should become SH, got: {}",
            result
        );
        // خ → KH
        let result = rules.apply("خ");
        assert!(
            result.contains("x"),
            "خ should become KH, got: {}",
            result
        );
        // ث → TH
        let result = rules.apply("ث");
        assert!(
            result.contains("θ"),
            "ث should become TH, got: {}",
            result
        );
        // غ → GH
        let result = rules.apply("غ");
        assert!(
            result.contains("ɣ"),
            "غ should become GH, got: {}",
            result
        );
    }

    #[test]
    fn test_emphatic_consonants() {
        let rules = base();
        // ص → sˤ (emphatic s)
        let result = rules.apply("ص");
        assert!(
            result.contains("sˤ"),
            "ص should become sˤ (emphatic s), got: {}",
            result
        );
        // ض → dˤ (emphatic d - unique to Arabic)
        let result = rules.apply("ض");
        assert!(
            result.contains("dˤ"),
            "ض should become dˤ (emphatic d), got: {}",
            result
        );
        // ط → tˤ (emphatic t)
        let result = rules.apply("ط");
        assert!(
            result.contains("tˤ"),
            "ط should become tˤ (emphatic t), got: {}",
            result
        );
        // ظ → ðˤ (emphatic z)
        let result = rules.apply("ظ");
        assert!(
            result.contains("ðˤ"),
            "ظ should become ðˤ (emphatic dh), got: {}",
            result
        );
    }

    #[test]
    fn test_pharyngeal_consonants() {
        let rules = base();
        // ح → ħ (voiceless pharyngeal)
        let result = rules.apply("ح");
        assert!(
            result.contains('ħ'),
            "ح should become ħ (voiceless pharyngeal), got: {}",
            result
        );
        // ع → ʕ (voiced pharyngeal)
        let result = rules.apply("ع");
        assert!(
            result.contains('ʕ'),
            "ع should become ʕ (voiced pharyngeal), got: {}",
            result
        );
    }

    #[test]
    fn test_hamza() {
        let rules = base();
        // أ → a (hamza on alif above)
        let result = rules.apply("أ");
        assert!(
            result.contains('a'),
            "أ should become a, got: {}",
            result
        );
        // إ → i (hamza on alif below)
        let result = rules.apply("إ");
        assert!(
            result.contains('i'),
            "إ should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_special_forms() {
        let rules = base();
        // آ → aː (alif madda - long a)
        let result = rules.apply("آ");
        assert!(
            result.contains("aː"),
            "آ should become aː (long a), got: {}",
            result
        );
        // ة → a (ta marbuta)
        let result = rules.apply("ة");
        assert!(
            result.contains('a'),
            "ة should become a, got: {}",
            result
        );
    }

    #[test]
    fn test_numerals() {
        let rules = base();
        // ٠ → 0
        let result = rules.apply("٠");
        assert!(
            result.contains('0'),
            "٠ should become 0, got: {}",
            result
        );
        // ٥ → 5
        let result = rules.apply("٥");
        assert!(
            result.contains('5'),
            "٥ should become 5, got: {}",
            result
        );
        // ٩ → 9
        let result = rules.apply("٩");
        assert!(
            result.contains('9'),
            "٩ should become 9, got: {}",
            result
        );
    }

    #[test]
    fn test_word_marhaba() {
        let rules = base();
        // مرحبا (marhaba - hello)
        let result = rules.apply("مرحبا");
        // م→m, ر→r, ح→ħ, ب→b, ا→a
        assert!(
            result.contains('m') && result.contains('r') && result.contains('ħ') && result.contains('b'),
            "مرحبا should contain m, r, ħ, b, got: {}",
            result
        );
    }

    #[test]
    fn test_word_salam() {
        let rules = base();
        // سلام (salam - peace)
        let result = rules.apply("سلام");
        // س→s, ل→l, ا→a, م→m
        assert!(
            result.contains('s') && result.contains('l') && result.contains('a') && result.contains('m'),
            "سلام should contain s, l, a, m, got: {}",
            result
        );
    }

}
