//! Tamil phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Tamil (தமிழ்) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Tamil phonetic normalization handles:
//! - **Tamil script**: Oldest Dravidian script with unique features
//! - **Limited consonant set**: 18 native consonants (no aspirated consonants)
//! - **Unique sounds**: ழ (retroflex approximant), ற (alveolar trill), ன (alveolar nasal)
//! - **Grantha letters**: For Sanskrit loanwords (ஜ, ஶ, ஷ, ஸ, ஹ)
//!
//! # Tamil Script
//!
//! Tamil has a simpler consonant system than other Brahmic scripts.
//! Native Tamil has no aspirated consonants - aspiration is indicated
//! by doubling or using Grantha letters for Sanskrit loanwords.
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers for Tamil sounds:
//! - ZH = retroflex approximant (ழ) - unique Tamil sound!
//! - RR = alveolar trill (ற)
//! - LL = retroflex lateral (ள)
//! - TT = retroflex stop (ட)
//! - NN = retroflex nasal (ண)
//! - NY = palatal nasal (ஞ)
//! - SH = palatal fricative (ஶ - Grantha)
//! - SS = retroflex fricative (ஷ - Grantha)
//! - AI, AU = diphthongs
//! - M = nasalization (anusvara)
//! - H = aytham (aspirated marker)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Tamil phonetic rules (~65 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::tamil;
//!
//! let rules = tamil::base();
//!
//! // Unique retroflex approximant
//! let result = rules.apply("ழ");
//! assert!(result.contains("ʒ"), "ழ → ZH");
//!
//! // Standard consonants
//! let result = rules.apply("க");
//! assert!(result.contains('k'), "க → k");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Tamil base phonetic rules.
///
/// Complete phonetic normalization rules for Tamil:
///
/// ## Tamil-Specific (weight 0.02)
/// - ழ→ZH (retroflex approximant - unique!)
/// - ற→RR (alveolar trill)
/// - ன→nn (alveolar nasal)
/// - ள→LL (retroflex lateral)
///
/// ## Independent Vowels (weight 0.05)
/// - அ→a, ஆ→A, இ→i, ஈ→I, உ→u, ஊ→U
/// - எ→e, ஏ→E, ஐ→AI, ஒ→o, ஓ→O, ஔ→AU
///
/// ## Native Consonants (weight 0.05)
/// - Vallinam (hard): க→k, ச→c, ட→TT, த→t, ப→p
/// - Mellinam (nasals): ங→N, ஞ→NY, ண→NN, ந→n, ம→m
/// - Idaiyinam (medium): ய→y, ர→r, ல→l, வ→v
///
/// ## Grantha Consonants (weight 0.05)
/// - ஜ→j, ஶ→SH, ஷ→SS, ஸ→s, ஹ→h
///
/// ## Vowel Matras (weight 0.05)
/// - ா→A, ி→i, ீ→I, ு→u, ூ→U, ெ→e, ே→E, ை→AI, ொ→o, ோ→O, ௌ→AU
///
/// ## Diacritics (weight 0.1)
/// - Pulli (்)→∅, Anusvara (ஂ)→M, Aytham (ஃ)→H
///
/// ## Numerals (weight 0.1)
/// - Tamil digits: ௦-௯ → 0-9
/// - Special: ௰→10, ௱→100, ௲→1000
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/tamil/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded tamil/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Tamil base rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Tamil base rules should not be empty");
        assert!(
            rules.len() >= 55,
            "expected >=55 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // TAMIL-SPECIFIC TESTS
    // ============================================================

    #[test]
    fn test_retroflex_approximant() {
        let rules = base();
        // ழ → ɻ (retroflex approximant, IPA)
        let result = rules.apply("ழ");
        assert!(
            result.contains("ɻ") || result.contains("ʒ"),
            "ழ should become ɻ (retroflex approximant), got: {}",
            result
        );
    }

    #[test]
    fn test_alveolar_trill() {
        let rules = base();
        // ற → RR
        let result = rules.apply("ற");
        assert!(result.contains("r"), "ற should become RR, got: {}", result);
    }

    #[test]
    fn test_alveolar_nasal() {
        let rules = base();
        // ன → n̺ (alveolar nasal with bridge below, IPA)
        let result = rules.apply("ன");
        assert!(
            result.contains("n̺") || result.contains("nn") || result.contains('n'),
            "ன should become n̺ (alveolar nasal), got: {}",
            result
        );
    }

    #[test]
    fn test_retroflex_lateral() {
        let rules = base();
        // ள → ɭ (retroflex lateral, IPA)
        let result = rules.apply("ள");
        assert!(
            result.contains("ɭ") || result.contains("ʎ"),
            "ள should become ɭ (retroflex lateral), got: {}",
            result
        );
    }

    // ============================================================
    // VOWEL TESTS
    // ============================================================

    #[test]
    fn test_independent_vowels() {
        let rules = base();
        let result = rules.apply("அ");
        assert!(result.contains('a'), "அ should become a, got: {}", result);
        let result = rules.apply("ஆ");
        assert!(result.contains("aː"), "ஆ should become A, got: {}", result);
        let result = rules.apply("இ");
        assert!(result.contains('i'), "இ should become i, got: {}", result);
    }

    #[test]
    fn test_short_long_vowels() {
        let rules = base();
        // Tamil distinguishes short and long e/o
        let result = rules.apply("எ");
        assert!(result.contains('e'), "எ should become e, got: {}", result);
        let result = rules.apply("ஏ");
        assert!(
            result.contains("eː") || result.contains('E'),
            "ஏ should become eː (long e), got: {}",
            result
        );
        let result = rules.apply("ஒ");
        assert!(
            result.contains('o') || result.contains('ɔ'),
            "ஒ should become o, got: {}",
            result
        );
        let result = rules.apply("ஓ");
        assert!(
            result.contains("oː") || result.contains('O'),
            "ஓ should become oː (long o), got: {}",
            result
        );
    }

    #[test]
    fn test_diphthongs() {
        let rules = base();
        let result = rules.apply("ஐ");
        assert!(result.contains("aɪ"), "ஐ should become AI, got: {}", result);
        let result = rules.apply("ஔ");
        assert!(result.contains("aʊ"), "ஔ should become AU, got: {}", result);
    }

    // ============================================================
    // CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_vallinam_consonants() {
        let rules = base();
        // Hard consonants (no aspiration in native Tamil)
        let result = rules.apply("க");
        assert!(result.contains('k'), "க should become k, got: {}", result);
        let result = rules.apply("ச");
        assert!(result.contains('c'), "ச should become c, got: {}", result);
        let result = rules.apply("ட");
        assert!(result.contains("ʈ"), "ட should become TT, got: {}", result);
        let result = rules.apply("த");
        assert!(result.contains('t'), "த should become t, got: {}", result);
        let result = rules.apply("ப");
        assert!(result.contains('p'), "ப should become p, got: {}", result);
    }

    #[test]
    fn test_mellinam_consonants() {
        let rules = base();
        // Nasal consonants
        let result = rules.apply("ங");
        assert!(result.contains('ŋ'), "ங should become N, got: {}", result);
        let result = rules.apply("ஞ");
        assert!(result.contains("ɲ"), "ஞ should become NY, got: {}", result);
        let result = rules.apply("ண");
        assert!(result.contains("ɳ"), "ண should become NN, got: {}", result);
    }

    #[test]
    fn test_grantha_consonants() {
        let rules = base();
        // Sanskrit loanword consonants
        let result = rules.apply("ஜ");
        assert!(
            result.contains('j') || result.contains('ʝ'),
            "ஜ should become j, got: {}",
            result
        );
        let result = rules.apply("ஶ");
        assert!(
            result.contains("ʃ"),
            "ஶ should become ʃ (SH), got: {}",
            result
        );
        let result = rules.apply("ஸ");
        assert!(result.contains('s'), "ஸ should become s, got: {}", result);
        let result = rules.apply("ஹ");
        assert!(result.contains('h'), "ஹ should become h, got: {}", result);
    }

    // ============================================================
    // MATRA TESTS
    // ============================================================

    #[test]
    fn test_vowel_matras() {
        let rules = base();
        let result = rules.apply("ா");
        assert!(result.contains("aː"), "ா should become A, got: {}", result);
        let result = rules.apply("ி");
        assert!(result.contains('i'), "ி should become i, got: {}", result);
        let result = rules.apply("ை");
        assert!(result.contains("aɪ"), "ை should become AI, got: {}", result);
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_tamil() {
        let rules = base();
        // தமிழ் (Tamil)
        let result = rules.apply_full("தமிழ்");
        // த→t, ம→m, ி→i, ழ→ɻ (retroflex approximant), ்→(pulli)
        assert!(
            result.contains('t') && result.contains('m'),
            "தமிழ் should contain t, m, got: {}",
            result
        );
        assert!(
            result.contains("ɻ") || result.contains("ʒ"),
            "தமிழ் should contain ɻ (retroflex approximant), got: {}",
            result
        );
    }

    #[test]
    fn test_word_chennai() {
        let rules = base();
        // சென்னை (Chennai)
        let result = rules.apply_full("சென்னை");
        // ச→c, ெ→e, ன→nn, ன→nn, ை→AI
        assert!(
            result.contains('c') && result.contains('e'),
            "சென்னை should contain c, e, got: {}",
            result
        );
    }

    #[test]
    fn test_word_madurai() {
        let rules = base();
        // மதுரை (Madurai)
        let result = rules.apply_full("மதுரை");
        // ம→m, த→t, ு→u, ர→r, ை→AI
        assert!(
            result.contains('m') && result.contains('t') && result.contains('r'),
            "மதுரை should contain m, t, r, got: {}",
            result
        );
    }

    // ============================================================
    // NUMERAL TESTS
    // ============================================================

    #[test]
    fn test_numerals() {
        let rules = base();
        let result = rules.apply("௦");
        assert!(result.contains('0'), "௦ should become 0, got: {}", result);
        let result = rules.apply("௫");
        assert!(result.contains('5'), "௫ should become 5, got: {}", result);
        let result = rules.apply("௯");
        assert!(result.contains('9'), "௯ should become 9, got: {}", result);
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================
}
