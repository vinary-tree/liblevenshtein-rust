//! Marathi phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Marathi (मराठी) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Marathi phonetic normalization handles:
//! - **Devanagari script**: Same script as Hindi with additions
//! - **Retroflex lateral**: ळ (ḷa) - unique to Marathi
//! - **English loanword vowels**: ऍ (æ), ऑ (ɔ)
//! - All standard Devanagari consonants and vowels
//!
//! # Marathi vs Hindi
//!
//! Marathi uses the same Devanagari script as Hindi with:
//! - **ळ (ḷa)**: Retroflex lateral approximant - unique to Marathi
//! - **Different schwa deletion**: More aggressive than Hindi
//! - **Same consonants, vowels, matras, and diacritics**
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers for Marathi-specific sounds:
//! - LL = retroflex lateral (ळ) - unique to Marathi
//! - TT, DD, NN = retroflex consonants (ट, ड, ण)
//! - SH = palatal fricative (श)
//! - SS = retroflex fricative (ष)
//! - NY = palatal nasal (ञ)
//! - AE = caret vowel (ऍ) for English loanwords
//! - AW = open o vowel (ऑ) for English loanwords
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Marathi phonetic rules (~88 rules)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::rules::marathi;
//!
//! let rules = marathi::base();
//!
//! // Retroflex lateral (unique to Marathi)
//! let result = rules.apply("ळ");
//! assert!(result.contains("ɭ"), "ळ → ɭ");
//!
//! // Standard Devanagari consonants
//! let result = rules.apply("क");
//! assert!(result.contains('k'), "क → k");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Marathi base phonetic rules.
///
/// Complete phonetic normalization rules for Marathi:
///
/// ## Marathi-Specific (weight 0.02)
/// - ळ→LL (retroflex lateral - unique to Marathi)
/// - ऍ→AE, ऑ→AW (English loanword vowels)
///
/// ## Independent Vowels (weight 0.05)
/// - अ→a, आ→A, इ→i, ई→I, उ→u, ऊ→U, ऋ→RI
/// - ए→e, ऐ→AI, ओ→o, औ→AU
///
/// ## Consonants (weight 0.05)
/// - Same as Hindi: velars, palatals, retroflexes, dentals, labials
/// - Semi-vowels: य→y, र→r, ल→l, व→v
/// - Sibilants: श→SH, ष→SS, स→s, ह→h
///
/// ## Vowel Matras (weight 0.05)
/// - Same as Hindi plus ॅ→AE, ॉ→AW
///
/// ## Diacritics (weight 0.1)
/// - Virama (्)→∅, Anusvara (ं)→M, Chandrabindu (ँ)→M, Visarga (ः)→H
///
/// ## Numerals (weight 0.1)
/// - Devanagari digits: ०-९ → 0-9
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/marathi/base.llev");
        let file = crate::phonetic::llev::parse_str(content).expect(
            "Invalid embedded marathi/base.llev - this indicates an internal invariant violation",
        );
        RuleSetChar::from_llev(&file).expect(
            "Failed to compile Marathi base rules - this indicates an internal invariant violation",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Marathi base rules should not be empty");
        assert!(
            rules.len() >= 70,
            "expected >=70 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // MARATHI-SPECIFIC TESTS
    // ============================================================

    #[test]
    fn test_retroflex_lateral() {
        let rules = base();
        // ळ → ɭ (retroflex lateral approximant, unique to Marathi)
        let result = rules.apply("ळ");
        assert!(
            result.contains("ɭ"),
            "ळ should become ɭ (retroflex lateral), got: {}",
            result
        );
    }

    #[test]
    fn test_ae_vowel() {
        let rules = base();
        // ऍ → AE (English loanword)
        let result = rules.apply("ऍ");
        assert!(result.contains("æ"), "ऍ should become AE, got: {}", result);
    }

    #[test]
    fn test_aw_vowel() {
        let rules = base();
        // ऑ → AW (English loanword)
        let result = rules.apply("ऑ");
        assert!(result.contains("ɔ"), "ऑ should become AW, got: {}", result);
    }

    // ============================================================
    // STANDARD DEVANAGARI TESTS (shared with Hindi)
    // ============================================================

    #[test]
    fn test_independent_vowels() {
        let rules = base();
        let result = rules.apply("अ");
        assert!(result.contains('a'), "अ should become a, got: {}", result);
        let result = rules.apply("आ");
        assert!(result.contains("aː"), "आ should become A, got: {}", result);
        let result = rules.apply("इ");
        assert!(result.contains('i'), "इ should become i, got: {}", result);
    }

    #[test]
    fn test_velar_consonants() {
        let rules = base();
        let result = rules.apply("क");
        assert!(result.contains('k'), "क should become k, got: {}", result);
        let result = rules.apply("ख");
        assert!(result.contains("kh"), "ख should become kh, got: {}", result);
        let result = rules.apply("ग");
        assert!(result.contains('ɡ'), "ग should become g, got: {}", result);
    }

    #[test]
    fn test_retroflex_consonants() {
        let rules = base();
        let result = rules.apply("ट");
        assert!(result.contains("ʈ"), "ट should become TT, got: {}", result);
        let result = rules.apply("ड");
        assert!(result.contains("ɖ"), "ड should become DD, got: {}", result);
        let result = rules.apply("ण");
        assert!(result.contains("ɳ"), "ण should become NN, got: {}", result);
    }

    #[test]
    fn test_sibilants() {
        let rules = base();
        let result = rules.apply("श");
        assert!(result.contains("ʃ"), "श should become SH, got: {}", result);
        let result = rules.apply("ष");
        assert!(result.contains("ʂ"), "ष should become SS, got: {}", result);
        let result = rules.apply("स");
        assert!(result.contains('s'), "स should become s, got: {}", result);
    }

    #[test]
    fn test_vowel_matras() {
        let rules = base();
        let result = rules.apply("ा");
        assert!(result.contains("aː"), "ा should become aː, got: {}", result);
        let result = rules.apply("ि");
        assert!(result.contains('i'), "ि should become i, got: {}", result);
        let result = rules.apply("ै");
        assert!(
            result.contains("ɛː"),
            "ै should become ɛː (ai diphthong), got: {}",
            result
        );
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_marathi() {
        let rules = base();
        // मराठी (Marathi)
        let result = rules.apply_full("मराठी");
        // म→m, र→r, ा→A, ठ→TTh, ी→I
        assert!(
            result.contains('m') && result.contains('r'),
            "मराठी should contain m, r, got: {}",
            result
        );
    }

    #[test]
    fn test_word_mumbai() {
        let rules = base();
        // मुंबई (Mumbai)
        let result = rules.apply_full("मुंबई");
        // म→m, ु→u, ं→M, ब→b, ई→I
        assert!(
            result.contains('m') && result.contains('b'),
            "मुंबई should contain m, b, got: {}",
            result
        );
    }

    #[test]
    fn test_word_with_retroflex_lateral() {
        let rules = base();
        // बाळ (child)
        let result = rules.apply_full("बाळ");
        // ब→b, ा→aː, ळ→ɭ
        assert!(
            result.contains('b') && result.contains("ɭ"),
            "बाळ should contain b, ɭ, got: {}",
            result
        );
    }

    #[test]
    fn test_word_maharashtra() {
        let rules = base();
        // महाराष्ट्र (Maharashtra)
        let result = rules.apply_full("महाराष्ट्र");
        // म→m, ह→h, ा→A, र→r, ा→A, ष→SS, ्→∅, ट→TT, ्→∅, र→r
        assert!(
            result.contains('m') && result.contains('h') && result.contains('r'),
            "महाराष्ट्र should contain m, h, r, got: {}",
            result
        );
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================
}
