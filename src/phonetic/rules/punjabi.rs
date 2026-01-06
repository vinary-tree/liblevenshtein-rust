//! Punjabi phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Punjabi (ਪੰਜਾਬੀ / پنجابی) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Punjabi is a **dual-script language**:
//! - **Gurmukhi** (ਗੁਰਮੁਖੀ): Used in India (Punjab)
//! - **Shahmukhi** (شاہ مکھی): Used in Pakistan (Punjab)
//!
//! Punjabi is also a **tonal language** with 3 tones:
//! - High falling
//! - Low rising
//! - Level
//!
//! Tones are not fully marked in either script.
//!
//! # Available Rule Sets
//!
//! - [`gurmukhi()`] - Gurmukhi script rules (~75 rules)
//! - [`shahmukhi()`] - Shahmukhi script rules (~50 rules)
//!
//! # Phonetic Markers
//!
//! ## Gurmukhi
//! - RR = retroflex flap (ੜ)
//! - LL = retroflex lateral (ਲ਼)
//! - TT, DD, NN = retroflex consonants
//! - SH = palatal fricative (ਸ਼)
//! - NY = palatal nasal (ਞ)
//! - AI, AU = diphthongs
//! - M = nasalization (bindi/tippi)
//!
//! ## Shahmukhi
//! - Uses Arabic-style markers (similar to Urdu)
//! - RR = retroflex flap (ڑ)
//! - TT, DD = retroflex consonants
//! - SH = shin (ش)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::punjabi;
//!
//! // Gurmukhi (India)
//! let gurmukhi_rules = punjabi::gurmukhi();
//! let result = gurmukhi_rules.apply("ਪੰਜਾਬੀ");
//! assert!(result.contains('p'));
//!
//! // Shahmukhi (Pakistan)
//! let shahmukhi_rules = punjabi::shahmukhi();
//! let result = shahmukhi_rules.apply("پنجابی");
//! assert!(result.contains('p'));
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Punjabi Gurmukhi phonetic rules.
///
/// Complete phonetic normalization rules for Punjabi in Gurmukhi script (India):
///
/// ## Nukta Consonants (weight 0.02)
/// - ਸ਼→SH (sha)
/// - ਖ਼→x (kha with nukta)
/// - ਗ਼→G (ga with nukta)
/// - ਜ਼→z (za)
/// - ਫ਼→f (fa)
/// - ੜ→RR (retroflex flap)
/// - ਲ਼→LL (retroflex lateral)
///
/// ## Independent Vowels (weight 0.05)
/// - ਅ→a, ਆ→A, ਇ→i, ਈ→I, ਉ→u, ਊ→U
/// - ਏ→e, ਐ→AI, ਓ→o, ਔ→AU
///
/// ## Consonants (weight 0.05)
/// - Velars: ਕ→k, ਖ→kh, ਗ→g, ਘ→gh, ਙ→N
/// - Palatals: ਚ→c, ਛ→ch, ਜ→j, ਝ→jh, ਞ→NY
/// - Retroflexes: ਟ→TT, ਠ→TTh, ਡ→DD, ਢ→DDh, ਣ→NN
/// - Dentals: ਤ→t, ਥ→th, ਦ→d, ਧ→dh, ਨ→n
/// - Labials: ਪ→p, ਫ→ph, ਬ→b, ਭ→bh, ਮ→m
/// - Semi-vowels: ਯ→y, ਰ→r, ਲ→l, ਵ→v
/// - Sibilants: ਸ→s, ਹ→h
///
/// ## Vowel Matras (weight 0.05)
/// - ਾ→A, ਿ→i, ੀ→I, ੁ→u, ੂ→U, ੇ→e, ੈ→AI, ੋ→o, ੌ→AU
///
/// ## Diacritics (weight 0.1)
/// - Virama (੍)→∅, Bindi (ਂ)→M, Tippi (ੰ)→M, Visarga (ਃ)→H
///
/// ## Numerals (weight 0.1)
/// - Gurmukhi digits: ੦-੯ → 0-9
pub fn gurmukhi() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/punjabi/gurmukhi.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded punjabi/gurmukhi.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Punjabi Gurmukhi rules - this is a bug in liblevenshtein")
    })
}

/// Punjabi Shahmukhi phonetic rules.
///
/// Complete phonetic normalization rules for Punjabi in Shahmukhi script (Pakistan):
///
/// ## Consonants (weight 0.05)
/// - Arabic-derived letters with Punjabi additions
/// - پ→p, ٹ→TT, چ→c, ڈ→DD, ڑ→RR, ژ→zh, گ→g, ں→M
/// - Standard Arabic letters: ب, ت, ج, د, ر, ز, س, ش, ف, ق, ک, ل, م, ن, و, ہ, ی
///
/// ## Vowel Marks (weight 0.1)
/// - Fatha (◌َ)→a, Kasra (◌ِ)→i, Damma (◌ُ)→u
/// - Sukun (◌ْ)→∅, Shadda (◌ّ)→∅
///
/// Note: Short vowels are often not written in Shahmukhi.
pub fn shahmukhi() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/punjabi/shahmukhi.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded punjabi/shahmukhi.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Punjabi Shahmukhi rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // GURMUKHI TESTS
    // ============================================================

    #[test]
    fn test_gurmukhi_loads() {
        let rules = gurmukhi();
        assert!(
            !rules.is_empty(),
            "Punjabi Gurmukhi rules should not be empty"
        );
        assert!(
            rules.len() >= 65,
            "expected >=65 Gurmukhi rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_gurmukhi_nukta_consonants() {
        let rules = gurmukhi();
        // ਸ਼ → ʃ (IPA voiceless postalveolar fricative)
        let result = rules.apply("ਸ਼");
        assert!(
            result.contains("ʃ"),
            "ਸ਼ should become ʃ, got: {}",
            result
        );
        // ੜ → ɽ (IPA retroflex flap)
        let result = rules.apply("ੜ");
        assert!(
            result.contains("ɽ") || result.contains("r"),
            "ੜ should become ɽ (retroflex flap), got: {}",
            result
        );
    }

    #[test]
    fn test_gurmukhi_vowels() {
        let rules = gurmukhi();
        let result = rules.apply("ਅ");
        assert!(result.contains('a'), "ਅ should become a, got: {}", result);
        let result = rules.apply("ਆ");
        assert!(result.contains("aː"), "ਆ should become aː, got: {}", result);
        // ਐ → ɛː (IPA long open-mid front unrounded vowel)
        let result = rules.apply("ਐ");
        assert!(
            result.contains("ɛː") || result.contains("aɪ"),
            "ਐ should become ɛː (IPA), got: {}",
            result
        );
    }

    #[test]
    fn test_gurmukhi_consonants() {
        let rules = gurmukhi();
        let result = rules.apply("ਕ");
        assert!(result.contains('k'), "ਕ should become k, got: {}", result);
        let result = rules.apply("ਪ");
        assert!(result.contains('p'), "ਪ should become p, got: {}", result);
        let result = rules.apply("ਮ");
        assert!(result.contains('m'), "ਮ should become m, got: {}", result);
    }

    #[test]
    fn test_gurmukhi_word_punjabi() {
        let rules = gurmukhi();
        // ਪੰਜਾਬੀ (Punjabi)
        let result = rules.apply_full("ਪੰਜਾਬੀ");
        // ਪ→p, ੰ→̃ (nasalization), ਜ→j, ਾ→aː, ਬ→b, ੀ→iː
        assert!(
            result.contains('p') && (result.contains('j') || result.contains('ʝ')) && result.contains('b'),
            "ਪੰਜਾਬੀ should contain p, j, b, got: {}",
            result
        );
    }

    #[test]
    fn test_gurmukhi_word_lahore() {
        let rules = gurmukhi();
        // ਲਾਹੌਰ (Lahore)
        let result = rules.apply_full("ਲਾਹੌਰ");
        // ਲ→l, ਾ→A, ਹ→h, ੌ→AU, ਰ→r
        assert!(
            result.contains('l') && result.contains('h') && result.contains('r'),
            "ਲਾਹੌਰ should contain l, h, r, got: {}",
            result
        );
    }

    #[test]
    fn test_gurmukhi_numerals() {
        let rules = gurmukhi();
        let result = rules.apply("੦");
        assert!(result.contains('0'), "੦ should become 0, got: {}", result);
        let result = rules.apply("੫");
        assert!(result.contains('5'), "੫ should become 5, got: {}", result);
    }

    // ============================================================
    // SHAHMUKHI TESTS
    // ============================================================

    #[test]
    fn test_shahmukhi_loads() {
        let rules = shahmukhi();
        assert!(
            !rules.is_empty(),
            "Punjabi Shahmukhi rules should not be empty"
        );
        assert!(
            rules.len() >= 40,
            "expected >=40 Shahmukhi rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_shahmukhi_consonants() {
        let rules = shahmukhi();
        let result = rules.apply("پ");
        assert!(result.contains('p'), "پ should become p, got: {}", result);
        let result = rules.apply("ب");
        assert!(result.contains('b'), "ب should become b, got: {}", result);
        let result = rules.apply("ش");
        assert!(
            result.contains("ʃ"),
            "ش should become SH, got: {}",
            result
        );
    }

    #[test]
    fn test_shahmukhi_retroflex() {
        let rules = shahmukhi();
        // ٹ → ʈ (IPA retroflex stop)
        let result = rules.apply("ٹ");
        assert!(
            result.contains("ʈ"),
            "ٹ should become ʈ, got: {}",
            result
        );
        // ڑ → ɽ (IPA retroflex flap)
        let result = rules.apply("ڑ");
        assert!(
            result.contains("ɽ") || result.contains("r"),
            "ڑ should become ɽ (retroflex flap), got: {}",
            result
        );
    }

    #[test]
    fn test_shahmukhi_word_punjabi() {
        let rules = shahmukhi();
        // پنجابی (Punjabi)
        let result = rules.apply_full("پنجابی");
        // پ→p, ن→n, ج→j, ا→a, ب→b, ی→y
        assert!(
            result.contains('p') && result.contains('n') && (result.contains('j') || result.contains('ʝ')),
            "پنجابی should contain p, n, j, got: {}",
            result
        );
    }

    // ============================================================
    // WEIGHT ORDERING TESTS
    // ============================================================


}
