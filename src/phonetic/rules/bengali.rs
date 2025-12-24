//! Bengali phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Bengali (বাংলা) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Bengali phonetic normalization handles:
//! - **Bengali script**: Brahmic abugida with inherent /ɔ/ vowel
//! - **Independent vowels**: অ→o, আ→a, ই→i, ঈ→I, উ→u, ঊ→U, ঋ→RI, এ→e, ঐ→OI, ও→O, ঔ→OU
//! - **Nukta consonants**: ড়→RR, ঢ়→RRh, য়→YY
//! - **Khanda ta**: ৎ→t (final t without inherent vowel)
//!
//! # Bengali vs Hindi
//!
//! Key differences from Hindi/Devanagari:
//! - **Inherent vowel**: Bengali has /ɔ/ (like 'o'), Hindi has /a/
//! - **য (ya)**: Pronounced /dʒ/ (like 'j'), not /y/ like Hindi
//! - **Different script**: Bengali/Assamese script family
//! - **Diphthongs**: OI and OU instead of AI and AU
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers for Bengali-specific sounds:
//! - TT, DD, NN = retroflex consonants
//! - SH = palatal fricative (শ)
//! - SS = retroflex fricative (ষ)
//! - NY = palatal nasal (ঞ)
//! - RR = retroflex flap (ড়)
//! - YY = nukta ya (য়)
//! - OI, OU = diphthongs (ঐ, ঔ)
//! - RI = vocalic r (ঋ)
//! - M = nasalization (anusvara/chandrabindu)
//! - H = visarga
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Bengali phonetic rules (~80 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::bengali;
//!
//! let rules = bengali::base();
//!
//! // Inherent vowel is ɔ (o sound)
//! let result = rules.apply("অ");
//! assert!(result.contains('o'), "অ → o (not 'a' like Hindi!)");
//!
//! // Nukta consonants
//! let result = rules.apply("ড়");
//! assert!(result.contains("RR"), "ড় → RR");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Bengali base phonetic rules.
///
/// Complete phonetic normalization rules for Bengali:
///
/// ## Nukta Consonants (weight 0.02)
/// - ড়→RR (retroflex flap)
/// - ঢ়→RRh (aspirated retroflex flap)
/// - য়→YY (nukta ya)
/// - ৎ→t (khanda ta)
///
/// ## Independent Vowels (weight 0.05)
/// - অ→o (inherent vowel, NOT 'a'!)
/// - আ→a, ই→i, ঈ→I, উ→u, ঊ→U, ঋ→RI
/// - এ→e, ঐ→OI, ও→O, ঔ→OU
///
/// ## Consonants (weight 0.05)
/// - Velars: ক→k, খ→kh, গ→g, ঘ→gh, ঙ→N
/// - Palatals: চ→c, ছ→ch, জ→j, ঝ→jh, ঞ→NY
/// - Retroflexes: ট→TT, ঠ→TTh, ড→DD, ঢ→DDh, ণ→NN
/// - Dentals: ত→t, থ→th, দ→d, ধ→dh, ন→n
/// - Labials: প→p, ফ→ph, ব→b, ভ→bh, ম→m
/// - Semi-vowels: য→j (not y!), র→r, ল→l
/// - Sibilants: শ→SH, ষ→SS, স→s, হ→h
///
/// ## Vowel Matras (weight 0.05)
/// - া→a, ি→i, ী→I, ু→u, ূ→U, ৃ→RI, ে→e, ৈ→OI, ো→O, ৌ→OU
///
/// ## Diacritics (weight 0.1)
/// - Hasanta (্)→∅, Anusvara (ং)→M, Chandrabindu (ঁ)→M, Visarga (ঃ)→H
///
/// ## Numerals (weight 0.1)
/// - Bengali digits: ০-৯ → 0-9
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/bengali/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded bengali/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Bengali base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Bengali base rules should not be empty");
        assert!(
            rules.len() >= 70,
            "expected >=70 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // INHERENT VOWEL TEST (key difference from Hindi)
    // ============================================================

    #[test]
    fn test_inherent_vowel_is_o_not_a() {
        let rules = base();
        // অ → o (not 'a' like Hindi!)
        let result = rules.apply("অ");
        assert!(
            result.contains('o'),
            "অ should become o (Bengali inherent vowel), got: {}",
            result
        );
    }

    // ============================================================
    // NUKTA CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_retroflex_flap() {
        let rules = base();
        // ড় → RR
        let result = rules.apply("ড়");
        assert!(
            result.contains("RR"),
            "ড় should become RR, got: {}",
            result
        );
    }

    #[test]
    fn test_nukta_ya() {
        let rules = base();
        // য় → YY
        let result = rules.apply("য়");
        assert!(
            result.contains("YY"),
            "য় should become YY, got: {}",
            result
        );
    }

    #[test]
    fn test_khanda_ta() {
        let rules = base();
        // ৎ → t
        let result = rules.apply("ৎ");
        assert!(
            result.contains('t'),
            "ৎ should become t, got: {}",
            result
        );
    }

    // ============================================================
    // VOWEL TESTS
    // ============================================================

    #[test]
    fn test_independent_vowels() {
        let rules = base();
        let result = rules.apply("আ");
        assert!(result.contains('a'), "আ should become a, got: {}", result);
        let result = rules.apply("ই");
        assert!(result.contains('i'), "ই should become i, got: {}", result);
        let result = rules.apply("এ");
        assert!(result.contains('e'), "এ should become e, got: {}", result);
    }

    #[test]
    fn test_diphthongs() {
        let rules = base();
        // ঐ → OI (not AI like Hindi)
        let result = rules.apply("ঐ");
        assert!(
            result.contains("OI"),
            "ঐ should become OI, got: {}",
            result
        );
        // ঔ → OU (not AU like Hindi)
        let result = rules.apply("ঔ");
        assert!(
            result.contains("OU"),
            "ঔ should become OU, got: {}",
            result
        );
    }

    // ============================================================
    // CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_velar_consonants() {
        let rules = base();
        let result = rules.apply("ক");
        assert!(result.contains('k'), "ক should become k, got: {}", result);
        let result = rules.apply("গ");
        assert!(result.contains('g'), "গ should become g, got: {}", result);
    }

    #[test]
    fn test_ya_becomes_j() {
        let rules = base();
        // য → j (not y! Bengali y is pronounced as j)
        let result = rules.apply("য");
        assert!(
            result.contains('j'),
            "য should become j (Bengali pronunciation), got: {}",
            result
        );
    }

    #[test]
    fn test_retroflex_consonants() {
        let rules = base();
        let result = rules.apply("ট");
        assert!(
            result.contains("TT"),
            "ট should become TT, got: {}",
            result
        );
        let result = rules.apply("ড");
        assert!(
            result.contains("DD"),
            "ড should become DD, got: {}",
            result
        );
    }

    #[test]
    fn test_sibilants() {
        let rules = base();
        let result = rules.apply("শ");
        assert!(
            result.contains("SH"),
            "শ should become SH, got: {}",
            result
        );
        let result = rules.apply("স");
        assert!(result.contains('s'), "স should become s, got: {}", result);
    }

    // ============================================================
    // MATRA TESTS
    // ============================================================

    #[test]
    fn test_vowel_matras() {
        let rules = base();
        let result = rules.apply("া");
        assert!(result.contains('a'), "া should become a, got: {}", result);
        let result = rules.apply("ি");
        assert!(result.contains('i'), "ি should become i, got: {}", result);
        let result = rules.apply("ৈ");
        assert!(
            result.contains("OI"),
            "ৈ should become OI, got: {}",
            result
        );
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_bangla() {
        let rules = base();
        // বাংলা (Bangla/Bengali)
        let result = rules.apply_full("বাংলা");
        // ব→b, া→a, ং→M, ল→l, া→a
        assert!(
            result.contains('b') && result.contains('l'),
            "বাংলা should contain b, l, got: {}",
            result
        );
    }

    #[test]
    fn test_word_kolkata() {
        let rules = base();
        // কলকাতা (Kolkata)
        let result = rules.apply_full("কলকাতা");
        // ক→k, ল→l, ক→k, া→a, ত→t, া→a
        assert!(
            result.contains('k') && result.contains('l') && result.contains('t'),
            "কলকাতা should contain k, l, t, got: {}",
            result
        );
    }

    #[test]
    fn test_word_dhaka() {
        let rules = base();
        // ঢাকা (Dhaka)
        let result = rules.apply_full("ঢাকা");
        // ঢ→DDh, া→a, ক→k, া→a
        assert!(
            result.contains('k'),
            "ঢাকা should contain k, got: {}",
            result
        );
    }

    // ============================================================
    // NUMERAL TESTS
    // ============================================================

    #[test]
    fn test_numerals() {
        let rules = base();
        let result = rules.apply("০");
        assert!(result.contains('0'), "০ should become 0, got: {}", result);
        let result = rules.apply("৫");
        assert!(result.contains('5'), "৫ should become 5, got: {}", result);
        let result = rules.apply("৯");
        assert!(result.contains('9'), "৯ should become 9, got: {}", result);
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================

    #[test]
    fn test_rules_sorted_by_weight() {
        let rules = base();
        let weights: Vec<_> = rules.rules.iter().map(|r| r.weight).collect();
        let mut sorted_weights = weights.clone();
        sorted_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(weights, sorted_weights, "Rules should be sorted by weight");
    }
}
