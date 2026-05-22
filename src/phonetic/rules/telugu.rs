//! Telugu phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Telugu (తెలుగు) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Telugu phonetic normalization handles:
//! - **Telugu script**: Brahmic abugida with distinctive rounded letters
//! - **Short/long vowel distinction**: ఎ/ఏ (e/E), ఒ/ఓ (o/O)
//! - **Retroflex lateral**: ళ (like Marathi/Gujarati)
//! - **Alveolar trill**: ఱ (historical, now rare)
//! - **Inherent /a/ vowel**: Like Hindi
//!
//! # Telugu Script
//!
//! Telugu is closely related to Kannada script. Both share the distinctive
//! rounded letterforms. Telugu has 16 vowels and 36+ consonants.
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers for Telugu sounds:
//! - LL = retroflex lateral (ళ)
//! - RR = alveolar trill (ఱ)
//! - TT, DD, NN = retroflex consonants
//! - SH = palatal fricative (శ)
//! - SS = retroflex fricative (ష)
//! - NY = palatal nasal (ఞ)
//! - AI, AU = diphthongs
//! - RI = vocalic r (ఋ)
//! - LI = vocalic l (ఌ)
//! - M = nasalization (anusvara/chandrabindu)
//! - H = visarga
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Telugu phonetic rules (~85 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::telugu;
//!
//! let rules = telugu::base();
//!
//! // Retroflex lateral
//! let result = rules.apply("ళ");
//! assert!(result.contains("ɭ"), "ళ → ɭ");
//!
//! // Standard consonants
//! let result = rules.apply("క");
//! assert!(result.contains('k'), "క → k");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Telugu base phonetic rules.
///
/// Complete phonetic normalization rules for Telugu:
///
/// ## Telugu-Specific (weight 0.02)
/// - ళ→LL (retroflex lateral)
/// - ఱ→RR (alveolar trill)
///
/// ## Independent Vowels (weight 0.05)
/// - అ→a, ఆ→A, ఇ→i, ఈ→I, ఉ→u, ఊ→U, ఋ→RI, ఌ→LI
/// - ఎ→e, ఏ→E, ఐ→AI, ఒ→o, ఓ→O, ఔ→AU
///
/// ## Consonants (weight 0.05)
/// - Velars: క→k, ఖ→kh, గ→g, ఘ→gh, ఙ→N
/// - Palatals: చ→c, ఛ→ch, జ→j, ఝ→jh, ఞ→NY
/// - Retroflexes: ట→TT, ఠ→TTh, డ→DD, ఢ→DDh, ణ→NN
/// - Dentals: త→t, థ→th, ద→d, ధ→dh, న→n
/// - Labials: ప→p, ఫ→ph, బ→b, భ→bh, మ→m
/// - Semi-vowels: య→y, ర→r, ల→l, వ→v
/// - Sibilants: శ→SH, ష→SS, స→s, హ→h
///
/// ## Vowel Matras (weight 0.05)
/// - ా→A, ి→i, ీ→I, ు→u, ూ→U, ృ→RI, ె→e, ే→E, ై→AI, ొ→o, ో→O, ౌ→AU
///
/// ## Diacritics (weight 0.1)
/// - Virama (్)→∅, Anusvara (ం)→M, Chandrabindu (ఁ)→M, Visarga (ః)→H
///
/// ## Numerals (weight 0.1)
/// - Telugu digits: ౦-౯ → 0-9
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/telugu/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded telugu/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Telugu base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Telugu base rules should not be empty");
        assert!(
            rules.len() >= 70,
            "expected >=70 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // TELUGU-SPECIFIC TESTS
    // ============================================================

    #[test]
    fn test_retroflex_lateral() {
        let rules = base();
        // ళ → ɭ (retroflex lateral approximant)
        let result = rules.apply("ళ");
        assert!(
            result.contains("ɭ"),
            "ళ should become ɭ (retroflex lateral), got: {}",
            result
        );
    }

    #[test]
    fn test_alveolar_trill() {
        let rules = base();
        // ఱ → ɽ (retroflex flap)
        let result = rules.apply("ఱ");
        assert!(
            result.contains("ɽ"),
            "ఱ should become ɽ (retroflex flap), got: {}",
            result
        );
    }

    // ============================================================
    // VOWEL TESTS
    // ============================================================

    #[test]
    fn test_independent_vowels() {
        let rules = base();
        let result = rules.apply("అ");
        assert!(result.contains('a'), "అ should become a, got: {}", result);
        let result = rules.apply("ఆ");
        assert!(result.contains("aː"), "ఆ should become A, got: {}", result);
        let result = rules.apply("ఇ");
        assert!(result.contains('i'), "ఇ should become i, got: {}", result);
    }

    #[test]
    fn test_short_long_vowels() {
        let rules = base();
        // Telugu distinguishes short and long e/o
        let result = rules.apply("ఎ");
        assert!(result.contains('e'), "ఎ should become e, got: {}", result);
        let result = rules.apply("ఏ");
        assert!(
            result.contains("eː"),
            "ఏ should become eː (long e), got: {}",
            result
        );
        let result = rules.apply("ఒ");
        assert!(result.contains('o'), "ఒ should become o, got: {}", result);
        let result = rules.apply("ఓ");
        assert!(
            result.contains("oː"),
            "ఓ should become oː (long o), got: {}",
            result
        );
    }

    #[test]
    fn test_diphthongs() {
        let rules = base();
        let result = rules.apply("ఐ");
        assert!(result.contains("aɪ"), "ఐ should become AI, got: {}", result);
        let result = rules.apply("ఔ");
        assert!(result.contains("aʊ"), "ఔ should become AU, got: {}", result);
    }

    // ============================================================
    // CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_velar_consonants() {
        let rules = base();
        let result = rules.apply("క");
        assert!(result.contains('k'), "క should become k, got: {}", result);
        let result = rules.apply("ఖ");
        assert!(result.contains("kh"), "ఖ should become kh, got: {}", result);
        let result = rules.apply("గ");
        assert!(
            result.contains('ɡ') || result.contains('g'),
            "గ should become ɡ (IPA g), got: {}",
            result
        );
    }

    #[test]
    fn test_retroflex_consonants() {
        let rules = base();
        let result = rules.apply("ట");
        assert!(result.contains("ʈ"), "ట should become TT, got: {}", result);
        let result = rules.apply("డ");
        assert!(result.contains("ɖ"), "డ should become DD, got: {}", result);
        let result = rules.apply("ణ");
        assert!(result.contains("ɳ"), "ణ should become NN, got: {}", result);
    }

    #[test]
    fn test_sibilants() {
        let rules = base();
        let result = rules.apply("శ");
        assert!(result.contains("ʃ"), "శ should become SH, got: {}", result);
        let result = rules.apply("ష");
        assert!(result.contains("ʂ"), "ష should become SS, got: {}", result);
        let result = rules.apply("స");
        assert!(result.contains('s'), "స should become s, got: {}", result);
    }

    // ============================================================
    // MATRA TESTS
    // ============================================================

    #[test]
    fn test_vowel_matras() {
        let rules = base();
        let result = rules.apply("ా");
        assert!(result.contains("aː"), "ా should become A, got: {}", result);
        let result = rules.apply("ి");
        assert!(result.contains('i'), "ి should become i, got: {}", result);
        let result = rules.apply("ై");
        assert!(result.contains("aɪ"), "ై should become AI, got: {}", result);
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_telugu() {
        let rules = base();
        // తెలుగు (Telugu)
        let result = rules.apply_full("తెలుగు");
        // త→t, ె→e, ల→l, ు→u, గ→ɡ (IPA g), ు→u
        assert!(
            result.contains('t')
                && result.contains('l')
                && (result.contains('ɡ') || result.contains('g')),
            "తెలుగు should contain t, l, ɡ (IPA g), got: {}",
            result
        );
    }

    #[test]
    fn test_word_hyderabad() {
        let rules = base();
        // హైదరాబాద్ (Hyderabad)
        let result = rules.apply_full("హైదరాబాద్");
        // హ→h, ై→AI, ద→d, ర→r, ా→A, బ→b, ా→A, ద→d, ్→(virama)
        assert!(
            result.contains('h') && result.contains('d') && result.contains('r'),
            "హైదరాబాద్ should contain h, d, r, got: {}",
            result
        );
    }

    #[test]
    fn test_word_andhra() {
        let rules = base();
        // ఆంధ్ర (Andhra)
        let result = rules.apply_full("ఆంధ్ర");
        // ఆ→A, ం→M, ధ→dh, ్→(virama), ర→r
        assert!(
            result.contains("aː") && result.contains('r'),
            "ఆంధ్ర should contain A, r, got: {}",
            result
        );
    }

    // ============================================================
    // NUMERAL TESTS
    // ============================================================

    #[test]
    fn test_numerals() {
        let rules = base();
        let result = rules.apply("౦");
        assert!(result.contains('0'), "౦ should become 0, got: {}", result);
        let result = rules.apply("౫");
        assert!(result.contains('5'), "౫ should become 5, got: {}", result);
        let result = rules.apply("౯");
        assert!(result.contains('9'), "౯ should become 9, got: {}", result);
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================
}
