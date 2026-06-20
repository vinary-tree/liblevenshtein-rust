//! Gujarati phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Gujarati (ગુજરાતી) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Gujarati phonetic normalization handles:
//! - **Gujarati script**: Brahmic abugida derived from Devanagari
//! - **No headline**: Unlike Devanagari, no shirorekha
//! - **Retroflex lateral**: ળ (like Marathi)
//! - **Same phonology as Hindi**: Inherent /a/ vowel
//!
//! # Gujarati Script
//!
//! Gujarati is derived from Devanagari but lacks the horizontal line
//! (shirorekha) that connects letters in Devanagari. The phonology is
//! very similar to Hindi.
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers for Gujarati sounds:
//! - LL = retroflex lateral (ળ)
//! - TT, DD, NN = retroflex consonants
//! - SH = palatal fricative (શ)
//! - SS = retroflex fricative (ષ)
//! - NY = palatal nasal (ઞ)
//! - AI, AU = diphthongs
//! - RI = vocalic r (ઋ)
//! - M = nasalization (anusvara/chandrabindu)
//! - H = visarga
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Gujarati phonetic rules (~75 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::gujarati;
//!
//! let rules = gujarati::base();
//!
//! // Retroflex lateral
//! let result = rules.apply("ળ");
//! assert!(result.contains("ʎ"), "ળ → LL");
//!
//! // Standard consonants
//! let result = rules.apply("ક");
//! assert!(result.contains('k'), "ક → k");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Gujarati base phonetic rules.
///
/// Complete phonetic normalization rules for Gujarati:
///
/// ## Gujarati-Specific (weight 0.02)
/// - ળ→LL (retroflex lateral)
///
/// ## Independent Vowels (weight 0.05)
/// - અ→a, આ→A, ઇ→i, ઈ→I, ઉ→u, ઊ→U, ઋ→RI
/// - એ→e, ઐ→AI, ઓ→o, ઔ→AU
///
/// ## Consonants (weight 0.05)
/// - Velars: ક→k, ખ→kh, ગ→g, ઘ→gh, ઙ→N
/// - Palatals: ચ→c, છ→ch, જ→j, ઝ→jh, ઞ→NY
/// - Retroflexes: ટ→TT, ઠ→TTh, ડ→DD, ઢ→DDh, ણ→NN
/// - Dentals: ત→t, થ→th, દ→d, ધ→dh, ન→n
/// - Labials: પ→p, ફ→ph, બ→b, ભ→bh, મ→m
/// - Semi-vowels: ય→y, ર→r, લ→l, વ→v
/// - Sibilants: શ→SH, ષ→SS, સ→s, હ→h
///
/// ## Vowel Matras (weight 0.05)
/// - ા→A, િ→i, ી→I, ુ→u, ૂ→U, ૃ→RI, ે→e, ૈ→AI, ો→o, ૌ→AU
///
/// ## Diacritics (weight 0.1)
/// - Virama (્)→∅, Anusvara (ં)→M, Chandrabindu (ઁ)→M, Visarga (ઃ)→H
///
/// ## Numerals (weight 0.1)
/// - Gujarati digits: ૦-૯ → 0-9
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/gujarati/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded gujarati/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Gujarati base rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Gujarati base rules should not be empty");
        assert!(
            rules.len() >= 65,
            "expected >=65 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // GUJARATI-SPECIFIC TESTS
    // ============================================================

    #[test]
    fn test_retroflex_lateral() {
        let rules = base();
        // ળ → ɭ (retroflex lateral)
        let result = rules.apply("ળ");
        assert!(
            result.contains("ɭ") || result.contains("ʎ"),
            "ળ should become ɭ (retroflex lateral), got: {}",
            result
        );
    }

    // ============================================================
    // VOWEL TESTS
    // ============================================================

    #[test]
    fn test_independent_vowels() {
        let rules = base();
        let result = rules.apply("અ");
        assert!(result.contains('a'), "અ should become a, got: {}", result);
        let result = rules.apply("આ");
        assert!(result.contains("aː"), "આ should become A, got: {}", result);
        let result = rules.apply("ઇ");
        assert!(result.contains('i'), "ઇ should become i, got: {}", result);
    }

    #[test]
    fn test_diphthongs() {
        let rules = base();
        let result = rules.apply("ઐ");
        assert!(
            result.contains("ɛː") || result.contains("aɪ"),
            "ઐ should become ɛː (AI), got: {}",
            result
        );
        let result = rules.apply("ઔ");
        assert!(
            result.contains("ɔː") || result.contains("aʊ"),
            "ઔ should become ɔː (AU), got: {}",
            result
        );
    }

    // ============================================================
    // CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_velar_consonants() {
        let rules = base();
        let result = rules.apply("ક");
        assert!(result.contains('k'), "ક should become k, got: {}", result);
        let result = rules.apply("ખ");
        assert!(
            result.contains("kh") || result.contains("x"),
            "ખ should become kh, got: {}",
            result
        );
        let result = rules.apply("ગ");
        assert!(result.contains('ɡ'), "ગ should become g, got: {}", result);
    }

    #[test]
    fn test_retroflex_consonants() {
        let rules = base();
        let result = rules.apply("ટ");
        assert!(result.contains("ʈ"), "ટ should become TT, got: {}", result);
        let result = rules.apply("ડ");
        assert!(result.contains("ɖ"), "ડ should become DD, got: {}", result);
        let result = rules.apply("ણ");
        assert!(result.contains("ɳ"), "ણ should become NN, got: {}", result);
    }

    #[test]
    fn test_sibilants() {
        let rules = base();
        let result = rules.apply("શ");
        assert!(result.contains("ʃ"), "શ should become SH, got: {}", result);
        let result = rules.apply("ષ");
        assert!(result.contains("ʂ"), "ષ should become SS, got: {}", result);
        let result = rules.apply("સ");
        assert!(result.contains('s'), "સ should become s, got: {}", result);
    }

    // ============================================================
    // MATRA TESTS
    // ============================================================

    #[test]
    fn test_vowel_matras() {
        let rules = base();
        let result = rules.apply("ા");
        assert!(result.contains("aː"), "ા should become A, got: {}", result);
        let result = rules.apply("િ");
        assert!(result.contains('i'), "િ should become i, got: {}", result);
        let result = rules.apply("ૈ");
        assert!(
            result.contains("ɛː") || result.contains("aɪ"),
            "ૈ should become ɛː (AI), got: {}",
            result
        );
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_gujarati() {
        let rules = base();
        // ગુજરાતી (Gujarati)
        let result = rules.apply_full("ગુજરાતી");
        // ગ→ɡ, ુ→u, જ→j, ર→r, ા→aː, ત→t, ી→iː
        assert!(
            result.contains('ɡ') && result.contains('j') && result.contains('r'),
            "ગુજરાતી should contain ɡ, j, r, got: {}",
            result
        );
    }

    #[test]
    fn test_word_ahmedabad() {
        let rules = base();
        // અમદાવાદ (Ahmedabad)
        let result = rules.apply_full("અમદાવાદ");
        // અ→a, મ→m, દ→d, ા→A, વ→v, ા→A, દ→d
        assert!(
            result.contains('a') && result.contains('m') && result.contains('d'),
            "અમદાવાદ should contain a, m, d, got: {}",
            result
        );
    }

    #[test]
    fn test_word_gujarat() {
        let rules = base();
        // ગુજરાત (Gujarat)
        let result = rules.apply_full("ગુજરાત");
        // ગ→ɡ, ુ→u, જ→j, ર→r, ા→aː, ત→t
        assert!(
            result.contains('ɡ') && result.contains('j') && result.contains('t'),
            "ગુજરાત should contain ɡ, j, t, got: {}",
            result
        );
    }

    // ============================================================
    // NUMERAL TESTS
    // ============================================================

    #[test]
    fn test_numerals() {
        let rules = base();
        let result = rules.apply("૦");
        assert!(result.contains('0'), "૦ should become 0, got: {}", result);
        let result = rules.apply("૫");
        assert!(result.contains('5'), "૫ should become 5, got: {}", result);
        let result = rules.apply("૯");
        assert!(result.contains('9'), "૯ should become 9, got: {}", result);
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================
}
