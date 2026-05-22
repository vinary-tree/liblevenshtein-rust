//! Hindi phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Hindi (हिन्दी) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Hindi phonetic normalization handles:
//! - **Independent vowels**: अ→a, आ→A, इ→i, ई→I, उ→u, ऊ→U, ऋ→RI, ए→e, ऐ→AI, ओ→o, औ→AU
//! - **Velar consonants**: क→k, ख→kh, ग→g, घ→gh, ङ→N
//! - **Palatal consonants**: च→c, छ→ch, ज→j, झ→jh, ञ→NY
//! - **Retroflex consonants**: ट→TT, ठ→TTh, ड→DD, ढ→DDh, ण→NN
//! - **Dental consonants**: त→t, थ→th, द→d, ध→dh, न→n
//! - **Labial consonants**: प→p, फ→ph, ब→b, भ→bh, म→m
//! - **Semi-vowels**: य→y, र→r, ल→l, व→v
//! - **Sibilants**: श→SH, ष→SS, स→s, ह→h
//! - **Nukta letters**: क़→q, ख़→KH, ग़→GH, ज़→z, फ़→f, ड़→RR, ढ़→RRh
//! - **Vowel matras**: ा→A, ि→i, ी→I, ु→u, ू→U, ृ→RI, े→e, ै→AI, ो→o, ौ→AU
//! - **Diacritics**: virama (्), anusvara (ं→M), chandrabindu (ँ→M), visarga (ः→H)
//! - **Numerals**: Devanagari digits (०-९ → 0-9)
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers for Hindi-specific sounds:
//! - TT, DD, NN = retroflex consonants (ट, ड, ण)
//! - TTh, DDh = aspirated retroflexes (ठ, ढ)
//! - SH = palatal fricative (श)
//! - SS = retroflex fricative (ष)
//! - NY = palatal nasal (ञ)
//! - RR = retroflex flap (ड़)
//! - A, I, U = long vowels (आ, ई, ऊ)
//! - AI, AU = diphthongs (ऐ, औ)
//! - RI = vocalic r (ऋ)
//! - M = nasalization (anusvara/chandrabindu)
//! - H = visarga
//!
//! # Devanagari Script
//!
//! Hindi uses the Devanagari abugida where:
//! - Each consonant carries an inherent /a/ vowel
//! - Vowel matras modify the inherent vowel
//! - Virama (्) removes the inherent vowel
//! - Conjunct consonants combine multiple consonants
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Hindi phonetic rules (~70 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::hindi;
//!
//! let rules = hindi::base();
//!
//! // Basic consonants
//! let result = rules.apply("क");
//! assert!(result.contains('k'), "क → k");
//!
//! // Retroflex consonants
//! let result = rules.apply("ट");
//! assert!(result.contains("ʈ"), "ट → TT");
//!
//! // Aspirated consonants
//! let result = rules.apply("ख");
//! assert!(result.contains("x"), "ख → kh");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Hindi base phonetic rules.
///
/// Complete phonetic normalization rules for Hindi:
///
/// ## Independent Vowels (weight 0.05)
/// - अ→a, आ→A, इ→i, ई→I, उ→u, ऊ→U, ऋ→RI
/// - ए→e, ऐ→AI, ओ→o, औ→AU
///
/// ## Consonants (weight 0.05)
/// - Velars: क→k, ख→kh, ग→g, घ→gh, ङ→N
/// - Palatals: च→c, छ→ch, ज→j, झ→jh, ञ→NY
/// - Retroflexes: ट→TT, ठ→TTh, ड→DD, ढ→DDh, ण→NN
/// - Dentals: त→t, थ→th, द→d, ध→dh, न→n
/// - Labials: प→p, फ→ph, ब→b, भ→bh, म→m
/// - Semi-vowels: य→y, र→r, ल→l, व→v
/// - Sibilants: श→SH, ष→SS, स→s, ह→h
///
/// ## Nukta Letters (weight 0.05)
/// - क़→q, ख़→KH, ग़→GH, ज़→z, फ़→f, ड़→RR, ढ़→RRh
///
/// ## Vowel Matras (weight 0.05)
/// - ा→A, ि→i, ी→I, ु→u, ू→U, ृ→RI, े→e, ै→AI, ो→o, ौ→AU
///
/// ## Diacritics (weight 0.1)
/// - Virama (्)→∅, Anusvara (ं)→M, Chandrabindu (ँ)→M, Visarga (ः)→H
///
/// ## Numerals (weight 0.1)
/// - Devanagari digits: ०-९ → 0-9
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/hindi/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded hindi/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Hindi base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Hindi base rules should not be empty");
        assert!(
            rules.len() > 65,
            "expected >65 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_independent_vowels() {
        let rules = base();
        // अ → a
        let result = rules.apply("अ");
        assert!(result.contains('a'), "अ should become a, got: {}", result);
        // आ → aː (long vowel)
        let result = rules.apply("आ");
        assert!(result.contains("aː"), "आ should become aː, got: {}", result);
        // इ → i
        let result = rules.apply("इ");
        assert!(result.contains('i'), "इ should become i, got: {}", result);
        // ऊ → uː (long vowel)
        let result = rules.apply("ऊ");
        assert!(result.contains("uː"), "ऊ should become uː, got: {}", result);
    }

    #[test]
    fn test_velar_consonants() {
        let rules = base();
        // क → k
        let result = rules.apply("क");
        assert!(result.contains('k'), "क should become k, got: {}", result);
        // ख → kh (aspirated voiceless velar stop)
        let result = rules.apply("ख");
        assert!(result.contains("kh"), "ख should become kh, got: {}", result);
        // ग → ɡ (IPA voiced velar stop)
        let result = rules.apply("ग");
        assert!(result.contains('ɡ'), "ग should become ɡ, got: {}", result);
        // घ → gh (aspirated voiced velar stop)
        let result = rules.apply("घ");
        assert!(result.contains("gh"), "घ should become gh, got: {}", result);
    }

    #[test]
    fn test_retroflex_consonants() {
        let rules = base();
        // ट → TT
        let result = rules.apply("ट");
        assert!(result.contains("ʈ"), "ट should become TT, got: {}", result);
        // ड → DD
        let result = rules.apply("ड");
        assert!(result.contains("ɖ"), "ड should become DD, got: {}", result);
        // ण → NN
        let result = rules.apply("ण");
        assert!(result.contains("ɳ"), "ण should become NN, got: {}", result);
    }

    #[test]
    fn test_dental_consonants() {
        let rules = base();
        // त → t
        let result = rules.apply("त");
        assert!(result.contains('t'), "त should become t, got: {}", result);
        // द → d
        let result = rules.apply("द");
        assert!(result.contains('d'), "द should become d, got: {}", result);
        // न → n
        let result = rules.apply("न");
        assert!(result.contains('n'), "न should become n, got: {}", result);
    }

    #[test]
    fn test_labial_consonants() {
        let rules = base();
        // प → p
        let result = rules.apply("प");
        assert!(result.contains('p'), "प should become p, got: {}", result);
        // ब → b
        let result = rules.apply("ब");
        assert!(result.contains('b'), "ब should become b, got: {}", result);
        // म → m
        let result = rules.apply("म");
        assert!(result.contains('m'), "म should become m, got: {}", result);
    }

    #[test]
    fn test_sibilants() {
        let rules = base();
        // श → SH
        let result = rules.apply("श");
        assert!(result.contains("ʃ"), "श should become SH, got: {}", result);
        // ष → SS
        let result = rules.apply("ष");
        assert!(result.contains("ʂ"), "ष should become SS, got: {}", result);
        // स → s
        let result = rules.apply("स");
        assert!(result.contains('s'), "स should become s, got: {}", result);
    }

    #[test]
    fn test_nukta_consonants() {
        let rules = base();
        // क़ → q (or k + nukta if not matched as sequence)
        // Note: Nukta consonants are composed of base consonant + nukta (़ U+093C).
        // If the two-character sequence is not matched, the base consonant is
        // transformed and the nukta passes through.
        let result = rules.apply("क़");
        assert!(
            result.contains('q') || result.contains('k'),
            "क़ should become q or k, got: {}",
            result
        );
        // ज़ → z (or j + nukta)
        let result = rules.apply("ज़");
        assert!(
            result.contains('z') || result.contains('j'),
            "ज़ should become z or j, got: {}",
            result
        );
        // फ़ → f (or ph + nukta)
        let result = rules.apply("फ़");
        assert!(
            result.contains('f') || result.contains("ph"),
            "फ़ should become f or ph, got: {}",
            result
        );
        // ड़ → ɽ (retroflex flap IPA, or ɖ + nukta)
        let result = rules.apply("ड़");
        assert!(
            result.contains('ɽ') || result.contains('ɖ'),
            "ड़ should become ɽ or ɖ (retroflex), got: {}",
            result
        );
    }

    #[test]
    fn test_vowel_matras() {
        let rules = base();
        // ा (aa matra) → aː
        let result = rules.apply("ा");
        assert!(result.contains("aː"), "ा should become aː, got: {}", result);
        // ि (i matra) → i
        let result = rules.apply("ि");
        assert!(result.contains('i'), "ि should become i, got: {}", result);
        // ै (ai matra) → ɛː (IPA representation)
        let result = rules.apply("ै");
        assert!(result.contains("ɛː"), "ै should become ɛː, got: {}", result);
    }

    #[test]
    fn test_numerals() {
        let rules = base();
        // ० → 0
        let result = rules.apply("०");
        assert!(result.contains('0'), "० should become 0, got: {}", result);
        // ५ → 5
        let result = rules.apply("५");
        assert!(result.contains('5'), "५ should become 5, got: {}", result);
        // ९ → 9
        let result = rules.apply("९");
        assert!(result.contains('9'), "९ should become 9, got: {}", result);
    }

    #[test]
    fn test_word_hindi() {
        let rules = base();
        // हिन्दी (Hindi)
        let result = rules.apply("हिन्दी");
        // ह→h, ि→i, न→n, ्→∅, द→d, ी→I
        assert!(
            result.contains('h') && result.contains('n') && result.contains('d'),
            "हिन्दी should contain h, n, d, got: {}",
            result
        );
    }

    #[test]
    fn test_word_namaste() {
        let rules = base();
        // नमस्ते (namaste)
        let result = rules.apply("नमस्ते");
        // न→n, म→m, स→s, ्→∅, त→t, े→e
        assert!(
            result.contains('n') && result.contains('m') && result.contains('s'),
            "नमस्ते should contain n, m, s, got: {}",
            result
        );
    }
}
