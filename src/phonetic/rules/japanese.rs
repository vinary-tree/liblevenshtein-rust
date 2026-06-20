//! Japanese phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Japanese Romaji (ローマ字) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Japanese Romaji phonetic normalization handles:
//! - **Long vowels**: ā→A, ē→E, ī→I, ō→O, ū→U (macron markers)
//! - **Romanization variants**: ti→C, tu→TS, si→S, hu→F (Kunrei→Hepburn)
//! - **Common digraphs**: shi→S, chi→C, tsu→TS, fu→F
//! - **Gemination**: kk→K, pp→P, tt→T, ss→S (double consonant simplification)
//! - **Syllabic N**: n'→N (before vowels)
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers to avoid rule reprocessing:
//! - A, E, I, O, U = long vowels (macrons)
//! - S = postalveolar fricative (shi)
//! - C = postalveolar affricate (chi)
//! - TS = alveolar affricate (tsu)
//! - F = bilabial fricative (fu)
//! - J = voiced affricate (ji)
//! - N = syllabic n (before vowel)
//! - K, P, T = geminated stops (っ)
//!
//! # Romanization Systems
//!
//! This ruleset normalizes between different romanization systems:
//! - **Hepburn** (most common): shi, chi, tsu, fu, ji
//! - **Kunrei-shiki/Nihon-shiki**: si, ti, tu, hu, zi
//!
//! # Available Rule Sets
//!
//! - [`romaji()`] - Complete Japanese Romaji phonetic rules (~40 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::japanese;
//!
//! let rules = japanese::romaji();
//!
//! // Long vowel normalization
//! let result = rules.apply("Tōkyō");
//! assert!(result.contains('O'), "ō → O");
//!
//! // Romanization variant normalization
//! let result = rules.apply("sushi");
//! assert!(result.contains('ʃ'), "shi → S");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Japanese Romaji phonetic rules.
///
/// Complete phonetic normalization rules for Japanese romanization:
///
/// ## Long Vowels (weight 0.05)
/// - ā→A, ē→E, ī→I, ō→O, ū→U (macron to marker)
///
/// ## Romanization Variants (weight 0.1)
/// - ti→C (chi), tu→TS (tsu), si→S (shi), hu→F (fu)
/// - zi→J (ji), di→J, du→Z (zu)
///
/// ## Digraphs (weight 0.1)
/// - shi→S, chi→C, tsu→TS, fu→F
/// - sha→Sa, shu→Su, sho→So, cha→Ca, chu→Cu, cho→Co
///
/// ## Gemination (weight 0.15)
/// - kk→K, pp→P, tt→T, ss→S, cc→C
///
/// ## Syllabic N (weight 0.1)
/// - n'→N (before vowel distinction)
pub fn romaji() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/japanese/romaji.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded japanese/romaji.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Japanese romaji rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_romaji_loads() {
        let rules = romaji();
        assert!(
            !rules.is_empty(),
            "Japanese romaji rules should not be empty"
        );
        assert!(
            rules.len() > 35,
            "expected >35 romaji rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_long_vowels() {
        let rules = romaji();
        // ā → aː (IPA long vowel)
        let result = rules.apply("ā");
        assert!(
            result.contains("aː") || result.contains('A'),
            "ā should become aː, got: {}",
            result
        );
        // ō → oː (IPA long vowel)
        let result = rules.apply("ō");
        assert!(
            result.contains("oː") || result.contains('O'),
            "ō should become oː, got: {}",
            result
        );
        // ū → uː (IPA long vowel)
        let result = rules.apply("ū");
        assert!(
            result.contains("uː") || result.contains('U'),
            "ū should become uː, got: {}",
            result
        );
    }

    #[test]
    fn test_romanization_variants() {
        let rules = romaji();
        // ti → t͡ɕ (IPA alveolo-palatal affricate, chi equivalent)
        let result = rules.apply("ti");
        assert!(
            result.contains("t͡ɕ") || result.contains('C'),
            "ti should become t͡ɕ, got: {}",
            result
        );
        // tu → t͡s (IPA alveolar affricate, tsu equivalent)
        let result = rules.apply("tu");
        assert!(
            result.contains("t͡s"),
            "tu should become t͡s, got: {}",
            result
        );
        // si → ɕ (IPA alveolo-palatal fricative, shi equivalent)
        let result = rules.apply("si");
        assert!(
            result.contains('ɕ') || result.contains('ʃ'),
            "si should become ɕ, got: {}",
            result
        );
        // hu → ɸ (IPA bilabial fricative, fu equivalent)
        let result = rules.apply("hu");
        assert!(
            result.contains('ɸ') || result.contains('F'),
            "hu should become ɸ, got: {}",
            result
        );
    }

    #[test]
    fn test_digraphs() {
        let rules = romaji();
        // shi → ɕ (IPA alveolo-palatal fricative)
        let result = rules.apply("shi");
        assert!(
            result.contains('ɕ') || result.contains('ʃ'),
            "shi should become ɕ, got: {}",
            result
        );
        // chi → t͡ɕ (IPA alveolo-palatal affricate)
        let result = rules.apply("chi");
        assert!(
            result.contains("t͡ɕ") || result.contains('C'),
            "chi should become t͡ɕ, got: {}",
            result
        );
        // tsu → t͡s (IPA alveolar affricate)
        let result = rules.apply("tsu");
        assert!(
            result.contains("t͡s"),
            "tsu should become t͡s, got: {}",
            result
        );
        // fu → ɸ (IPA bilabial fricative)
        let result = rules.apply("fu");
        assert!(
            result.contains('ɸ') || result.contains('F'),
            "fu should become ɸ, got: {}",
            result
        );
    }

    #[test]
    fn test_gemination() {
        let rules = romaji();
        // kk → K
        let result = rules.apply("kk");
        assert!(
            result.contains('K') && !result.contains("k͈"),
            "kk should become K, got: {}",
            result
        );
        // pp → P
        let result = rules.apply("pp");
        assert!(
            result.contains('P') && !result.contains("pp"),
            "pp should become P, got: {}",
            result
        );
    }

    #[test]
    fn test_syllabic_n() {
        let rules = romaji();
        // n' → N (syllable-final n before vowel)
        let result = rules.apply("n'");
        assert!(
            result.contains('N') || result.contains('ŋ'),
            "n' should become N, got: {}",
            result
        );
    }

    #[test]
    fn test_word_tokyo() {
        let rules = romaji();
        // Tōkyō - capital of Japan
        let result = rules.apply("Tōkyō");
        // T stays, ō→oː, k stays, y stays, ō→oː
        assert!(
            (result.contains("oː") || result.contains('O')) && result.contains('k'),
            "Tōkyō should have oː markers, got: {}",
            result
        );
    }

    #[test]
    fn test_word_sushi() {
        let rules = romaji();
        // sushi - famous Japanese food
        let result = rules.apply("sushi");
        // su stays, shi→ɕ, i stays
        assert!(
            result.contains('ɕ') || result.contains('ʃ'),
            "sushi should have ɕ (from shi), got: {}",
            result
        );
    }

    #[test]
    fn test_word_nippon() {
        let rules = romaji();
        // Nippon (Japan) - has geminated pp
        let result = rules.apply("Nippon");
        // N, i, pp→P, o, n
        assert!(
            result.contains('P'),
            "Nippon should have P (from pp), got: {}",
            result
        );
    }
}
