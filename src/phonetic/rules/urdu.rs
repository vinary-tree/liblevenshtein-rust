//! Urdu phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Urdu (اردو) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Urdu phonetic normalization handles:
//! - **Arabic-inherited consonants**: Based on Arabic alphabet
//! - **Persian additions**: پ (p), چ (ch), ژ (zh), گ (g)
//! - **Retroflex consonants**: ٹ→TT, ڈ→DD, ڑ→RR (Indo-Aryan sounds)
//! - **Aspirated consonants**: بھ→bh, پھ→ph, تھ→th, کھ→kh, گھ→gh
//! - **Special letters**: ں (nasal N), ے (bari ye), ھ (aspiration marker)
//! - **Diacritics**: zabar, zer, pesh (vowel marks)
//! - **Extended numerals**: Urdu numerals (۰-۹ → 0-9)
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers for Urdu-specific sounds:
//! - TT = retroflex t (ٹ)
//! - DD = retroflex d (ڈ)
//! - RR = retroflex r/flap (ڑ)
//! - C = ch affricate (چ)
//! - ZH = voiced postalveolar fricative (ژ)
//! - N = nasal n (ں)
//! - TH, DH, SH, KH, GH = Arabic-derived digraphs
//! - H, E, S, D, T, Z = emphatic/pharyngeal markers (like Arabic)
//!
//! # Urdu Script
//!
//! Urdu uses a modified Perso-Arabic script (Nastaliq calligraphy style)
//! written right-to-left. It includes:
//! - All Arabic consonants
//! - Persian additions for sounds not in Arabic
//! - Indo-Aryan retroflex consonants unique to South Asian languages
//! - Aspirated consonant sequences (consonant + do-chashmi he)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Urdu phonetic rules (~65 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::urdu;
//!
//! let rules = urdu::base();
//!
//! // Basic consonants
//! let result = rules.apply("ب");
//! assert!(result.contains('b'), "ب → b");
//!
//! // Retroflex consonants
//! let result = rules.apply("ٹ");
//! assert!(result.contains("TT"), "ٹ → TT");
//!
//! // Persian additions
//! let result = rules.apply("پ");
//! assert!(result.contains('p'), "پ → p");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Urdu base phonetic rules.
///
/// Complete phonetic normalization rules for Urdu:
///
/// ## Arabic-Inherited Consonants (weight 0.05)
/// - ا→a, ب→b, ت→t, ث→TH, ج→j, ح→H, خ→KH, د→d, ذ→DH, ر→r, ز→z
/// - س→s, ش→SH, ص→S, ض→D, ط→T, ظ→Z, ع→E, غ→GH, ف→f, ق→q, ک→k
/// - ل→l, م→m, ن→n, و→w, ی→y
///
/// ## Persian/Urdu-Specific Consonants (weight 0.05)
/// - پ→p (pe), چ→C (che), ژ→ZH (zhe), گ→g (gaf)
///
/// ## Retroflex Consonants (weight 0.05)
/// - ٹ→TT (retroflex t), ڈ→DD (retroflex d), ڑ→RR (retroflex r/flap)
///
/// ## Special Letters (weight 0.05)
/// - ں→N (nasal nun ghunna), ے→e (bari ye), ہ→h, ھ→h (aspiration)
///
/// ## Aspirated Consonants (weight 0.08)
/// - بھ→bh, پھ→ph, تھ→th, ٹھ→TTh, جھ→jh, چھ→Ch
/// - دھ→dh, ڈھ→DDh, ڑھ→RRh, کھ→kh, گھ→gh
///
/// ## Diacritics (weight 0.1)
/// - Zabar→a, Zer→i, Pesh→u
/// - Jazm→∅, Tashdid→∅
/// - Tanwin: ً→an, ٌ→un, ٍ→in
///
/// ## Numerals (weight 0.1)
/// - Extended Arabic-Indic digits: ۰-۹ → 0-9
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/urdu/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded urdu/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Urdu base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Urdu base rules should not be empty");
        assert!(
            rules.len() > 60,
            "expected >60 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_arabic_inherited_consonants() {
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
        // ک → k
        let result = rules.apply("ک");
        assert!(
            result.contains('k'),
            "ک should become k, got: {}",
            result
        );
    }

    #[test]
    fn test_persian_additions() {
        let rules = base();
        // پ → p
        let result = rules.apply("پ");
        assert!(
            result.contains('p'),
            "پ should become p, got: {}",
            result
        );
        // چ → C
        let result = rules.apply("چ");
        assert!(
            result.contains('C'),
            "چ should become C, got: {}",
            result
        );
        // گ → g
        let result = rules.apply("گ");
        assert!(
            result.contains('g'),
            "گ should become g, got: {}",
            result
        );
        // ژ → ZH
        let result = rules.apply("ژ");
        assert!(
            result.contains("ZH"),
            "ژ should become ZH, got: {}",
            result
        );
    }

    #[test]
    fn test_retroflex_consonants() {
        let rules = base();
        // ٹ → TT
        let result = rules.apply("ٹ");
        assert!(
            result.contains("TT"),
            "ٹ should become TT, got: {}",
            result
        );
        // ڈ → DD
        let result = rules.apply("ڈ");
        assert!(
            result.contains("DD"),
            "ڈ should become DD, got: {}",
            result
        );
        // ڑ → RR
        let result = rules.apply("ڑ");
        assert!(
            result.contains("RR"),
            "ڑ should become RR, got: {}",
            result
        );
    }

    #[test]
    fn test_special_letters() {
        let rules = base();
        // ں → N (nasal nun)
        let result = rules.apply("ں");
        assert!(
            result.contains('N'),
            "ں should become N, got: {}",
            result
        );
        // ے → e (bari ye)
        let result = rules.apply("ے");
        assert!(
            result.contains('e'),
            "ے should become e, got: {}",
            result
        );
        // ہ → h (gol he)
        let result = rules.apply("ہ");
        assert!(
            result.contains('h'),
            "ہ should become h, got: {}",
            result
        );
    }

    #[test]
    fn test_aspirated_consonants() {
        let rules = base();
        // بھ → bh
        let result = rules.apply("بھ");
        assert!(
            result.contains("bh"),
            "بھ should become bh, got: {}",
            result
        );
        // پھ → ph
        let result = rules.apply("پھ");
        assert!(
            result.contains("ph"),
            "پھ should become ph, got: {}",
            result
        );
        // کھ → kh
        let result = rules.apply("کھ");
        assert!(
            result.contains("kh"),
            "کھ should become kh, got: {}",
            result
        );
        // گھ → gh
        let result = rules.apply("گھ");
        assert!(
            result.contains("gh"),
            "گھ should become gh, got: {}",
            result
        );
    }

    #[test]
    fn test_special_consonants() {
        let rules = base();
        // ش → SH
        let result = rules.apply("ش");
        assert!(
            result.contains("SH"),
            "ش should become SH, got: {}",
            result
        );
        // خ → KH
        let result = rules.apply("خ");
        assert!(
            result.contains("KH"),
            "خ should become KH, got: {}",
            result
        );
        // ث → TH
        let result = rules.apply("ث");
        assert!(
            result.contains("TH"),
            "ث should become TH, got: {}",
            result
        );
        // غ → GH
        let result = rules.apply("غ");
        assert!(
            result.contains("GH"),
            "غ should become GH, got: {}",
            result
        );
    }

    #[test]
    fn test_emphatic_consonants() {
        let rules = base();
        // ص → S (emphatic s)
        let result = rules.apply("ص");
        assert!(
            result.contains('S'),
            "ص should become S, got: {}",
            result
        );
        // ض → D (emphatic d)
        let result = rules.apply("ض");
        assert!(
            result.contains('D'),
            "ض should become D, got: {}",
            result
        );
        // ط → T (emphatic t)
        let result = rules.apply("ط");
        assert!(
            result.contains('T'),
            "ط should become T, got: {}",
            result
        );
        // ظ → Z (emphatic z)
        let result = rules.apply("ظ");
        assert!(
            result.contains('Z'),
            "ظ should become Z, got: {}",
            result
        );
    }

    #[test]
    fn test_pharyngeal_consonants() {
        let rules = base();
        // ح → H (voiceless pharyngeal)
        let result = rules.apply("ح");
        assert!(
            result.contains('H'),
            "ح should become H, got: {}",
            result
        );
        // ع → E (voiced pharyngeal)
        let result = rules.apply("ع");
        assert!(
            result.contains('E'),
            "ع should become E, got: {}",
            result
        );
    }

    #[test]
    fn test_numerals() {
        let rules = base();
        // ۰ → 0
        let result = rules.apply("۰");
        assert!(
            result.contains('0'),
            "۰ should become 0, got: {}",
            result
        );
        // ۵ → 5
        let result = rules.apply("۵");
        assert!(
            result.contains('5'),
            "۵ should become 5, got: {}",
            result
        );
        // ۹ → 9
        let result = rules.apply("۹");
        assert!(
            result.contains('9'),
            "۹ should become 9, got: {}",
            result
        );
    }

    #[test]
    fn test_word_pakistan() {
        let rules = base();
        // پاکستان (Pakistan)
        let result = rules.apply("پاکستان");
        // پ→p, ا→a, ک→k, س→s, ت→t, ا→a, ن→n
        assert!(
            result.contains('p') && result.contains('k') && result.contains('s'),
            "پاکستان should contain p, k, s, got: {}",
            result
        );
    }

    #[test]
    fn test_word_urdu() {
        let rules = base();
        // اردو (Urdu)
        let result = rules.apply("اردو");
        // ا→a, ر→r, د→d, و→w
        assert!(
            result.contains('a') && result.contains('r') && result.contains('d'),
            "اردو should contain a, r, d, got: {}",
            result
        );
    }

    #[test]
    fn test_rules_sorted_by_weight() {
        let rules = base();
        let weights: Vec<_> = rules.rules.iter().map(|r| r.weight).collect();
        let mut sorted_weights = weights.clone();
        sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(weights, sorted_weights, "Rules should be sorted by weight");
    }
}
