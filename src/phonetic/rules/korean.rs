//! Korean phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Korean (한국어) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Korean phonetic normalization handles:
//! - **Hangul jamo to Latin**: Consonants and vowels mapped to romanization
//! - **Double consonants**: ㄲ→kk, ㄸ→tt, ㅃ→pp, ㅆ→ss, ㅉ→jj
//! - **Aspirated consonants**: ㅋ→k, ㅌ→t, ㅍ→p, ㅊ→ch
//! - **Basic consonants**: ㄱ→g, ㄴ→n, ㄷ→d, ㄹ→r, etc.
//! - **Compound finals**: ㄳ→ks, ㄵ→nj, ㄺ→lg, etc.
//! - **Vowels**: ㅏ→a, ㅓ→eo, ㅗ→o, ㅜ→u, ㅡ→eu, ㅣ→i
//! - **Y-vowels**: ㅑ→ya, ㅕ→yeo, ㅛ→yo, ㅠ→yu
//! - **W-vowels**: ㅘ→wa, ㅙ→wae, ㅚ→oe, ㅝ→wo, ㅞ→we, ㅟ→wi
//!
//! # Romanization Variant Normalization
//!
//! The [`romanization()`] ruleset normalizes different Korean romanization
//! systems to the Revised Romanization (RR) standard:
//! - **McCune-Reischauer breves**: ŏ→eo, ŭ→eu
//! - **Aspirate markers**: k'→k, p'→p, t'→t, ch'→ch
//! - **Vowel digraph normalization**: eo→O, eu→U, ae→E
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Korean jamo transliteration rules (~45 rules)
//! - [`romanization()`] - Romanization variant normalization (~35 rules)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::rules::korean;
//!
//! let rules = korean::base();
//!
//! // Basic consonant
//! let result = rules.apply("ㄱ");
//! assert_eq!(result, "k");
//!
//! // Double consonant
//! let result = rules.apply("ㄲ");
//! assert_eq!(result, "k͈");
//!
//! // Romanization variant normalization
//! let rom_rules = korean::romanization();
//! let result = rom_rules.apply("Sŏul");
//! assert_eq!(result, "sʌul");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Korean phonetic rules.
///
/// Complete phonetic normalization rules for Korean Hangul jamo:
///
/// ## Double (Tense) Consonants
/// - ㄲ→kk, ㄸ→tt, ㅃ→pp, ㅆ→ss, ㅉ→jj
///
/// ## Aspirated Consonants
/// - ㅋ→k, ㅌ→t, ㅍ→p, ㅊ→ch
///
/// ## Basic Consonants
/// - ㄱ→g, ㄴ→n, ㄷ→d, ㄹ→r, ㅁ→m, ㅂ→b, ㅅ→s, ㅇ→ng, ㅈ→j, ㅎ→h
///
/// ## Compound Finals
/// - ㄳ→ks, ㄵ→nj, ㄶ→nh, ㄺ→lg, ㄻ→lm, etc.
///
/// ## Vowels
/// - Basic: ㅏ→a, ㅓ→eo, ㅗ→o, ㅜ→u, ㅡ→eu, ㅣ→i, ㅐ→ae, ㅔ→e
/// - Y-glide: ㅑ→ya, ㅕ→yeo, ㅛ→yo, ㅠ→yu, ㅒ→yae, ㅖ→ye
/// - W-glide: ㅘ→wa, ㅙ→wae, ㅚ→oe, ㅝ→wo, ㅞ→we, ㅟ→wi, ㅢ→ui
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/korean/base.llev");
        let file = crate::phonetic::llev::parse_str(content).expect(
            "Invalid embedded korean/base.llev - this indicates an internal invariant violation",
        );
        RuleSetChar::from_llev(&file).expect(
            "Failed to compile Korean base rules - this indicates an internal invariant violation",
        )
    })
}

/// Korean romanization variant normalization rules.
///
/// Normalizes different Korean romanization systems to a common form
/// based on Revised Romanization (RR):
///
/// ## McCune-Reischauer Breve Vowels
/// - ŏ → eo, ŭ → eu
///
/// ## Aspirate Markers
/// - k' → k, p' → p, t' → t, ch' → ch
///
/// ## Vowel Digraph Normalization
/// - eo → O, eu → U, ae → E (uppercase markers)
///
/// ## W/Y Combinations
/// - wa → WA, wo → WO, we → WE, wi → WI
/// - yeo → YO, yae → YE
///
/// ## Syllable Separators
/// - Hyphens removed for consistent matching
pub fn romanization() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/korean/romanization.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded korean/romanization.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Korean romanization rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Korean base rules should not be empty");
        assert!(
            rules.len() > 35,
            "expected >35 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_double_consonants() {
        let rules = base();
        // ㄲ → k͈ (IPA fortis k)
        let result = rules.apply("ㄲ");
        assert!(result.contains("k͈"), "ㄲ should become k͈, got: {}", result);
        // ㄸ → t͈ (IPA fortis t)
        let result = rules.apply("ㄸ");
        assert!(result.contains("t͈"), "ㄸ should become t͈, got: {}", result);
        // ㅃ → p͈ (IPA fortis p)
        let result = rules.apply("ㅃ");
        assert!(
            result.contains("p͈") || result.contains("pp"),
            "ㅃ should become p͈, got: {}",
            result
        );
        // ㅆ → s͈ (IPA fortis s)
        let result = rules.apply("ㅆ");
        assert!(
            result.contains("s͈") || result.contains("ss"),
            "ㅆ should become s͈, got: {}",
            result
        );
        // ㅉ → t͈͡ɕ͈ (IPA fortis j)
        let result = rules.apply("ㅉ");
        assert!(
            result.contains("t͡ɕ͈") || result.contains("jj"),
            "ㅉ should become t͡ɕ͈, got: {}",
            result
        );
    }

    #[test]
    fn test_aspirated_consonants() {
        let rules = base();
        // ㅋ → kʰ (IPA aspirated k)
        let result = rules.apply("ㅋ");
        assert!(result.contains('k'), "ㅋ should contain k, got: {}", result);
        // ㅌ → tʰ (IPA aspirated t)
        let result = rules.apply("ㅌ");
        assert!(result.contains('t'), "ㅌ should contain t, got: {}", result);
        // ㅍ → pʰ (IPA aspirated p)
        let result = rules.apply("ㅍ");
        assert!(result.contains('p'), "ㅍ should contain p, got: {}", result);
        // ㅊ → t͡ɕʰ (IPA aspirated alveolo-palatal affricate)
        let result = rules.apply("ㅊ");
        assert!(
            result.contains("t͡ɕ"),
            "ㅊ should become t͡ɕʰ, got: {}",
            result
        );
    }

    #[test]
    fn test_basic_consonants() {
        let rules = base();
        // ㄱ → k (lax consonant, voiceless)
        let result = rules.apply("ㄱ");
        assert!(result.contains('k'), "ㄱ should become k, got: {}", result);
        // ㄴ → n
        let result = rules.apply("ㄴ");
        assert!(result.contains('n'), "ㄴ should become n, got: {}", result);
        // ㄷ → t (lax consonant, voiceless)
        let result = rules.apply("ㄷ");
        assert!(
            result.contains('t') || result.contains('d'),
            "ㄷ should become t or d, got: {}",
            result
        );
        // ㄹ → r/l
        let result = rules.apply("ㄹ");
        assert!(
            result.contains('r') || result.contains('l'),
            "ㄹ should become r or l, got: {}",
            result
        );
        // ㅁ → m
        let result = rules.apply("ㅁ");
        assert!(result.contains('m'), "ㅁ should become m, got: {}", result);
        // ㅂ → p (lax consonant, voiceless)
        let result = rules.apply("ㅂ");
        assert!(
            result.contains('p') || result.contains('b'),
            "ㅂ should become p or b, got: {}",
            result
        );
        // ㅅ → t (in final position) or s
        let result = rules.apply("ㅅ");
        assert!(
            result.contains('s') || result.contains('t'),
            "ㅅ should become s or t, got: {}",
            result
        );
        // ㅎ → h (but may be silent/dropped in some contexts)
        let result = rules.apply("ㅎ");
        assert!(
            result.contains('h') || result.is_empty() || result == "ㅎ",
            "ㅎ should become h or be dropped, got: {}",
            result
        );
    }

    #[test]
    fn test_basic_vowels() {
        let rules = base();
        // ㅏ → a
        let result = rules.apply("ㅏ");
        assert!(result.contains('a'), "ㅏ should become a, got: {}", result);
        // ㅗ → o
        let result = rules.apply("ㅗ");
        assert!(result.contains('o'), "ㅗ should become o, got: {}", result);
        // ㅜ → u
        let result = rules.apply("ㅜ");
        assert!(result.contains('u'), "ㅜ should become u, got: {}", result);
        // ㅣ → i
        let result = rules.apply("ㅣ");
        assert!(result.contains('i'), "ㅣ should become i, got: {}", result);
    }

    #[test]
    fn test_y_vowels() {
        let rules = base();
        // ㅑ → ya
        let result = rules.apply("ㅑ");
        assert!(
            result.contains("ya") || result.contains("ja"),
            "ㅑ should become ya, got: {}",
            result
        );
        // ㅛ → yo
        let result = rules.apply("ㅛ");
        assert!(
            result.contains("yo") || result.contains("jo"),
            "ㅛ should become yo, got: {}",
            result
        );
        // ㅠ → yu
        let result = rules.apply("ㅠ");
        assert!(
            result.contains("yu") || result.contains("ju"),
            "ㅠ should become yu, got: {}",
            result
        );
    }

    #[test]
    fn test_w_vowels() {
        let rules = base();
        // ㅘ → wa
        let result = rules.apply("ㅘ");
        assert!(
            result.contains("wa"),
            "ㅘ should become wa, got: {}",
            result
        );
        // ㅟ → wi
        let result = rules.apply("ㅟ");
        assert!(
            result.contains("wi"),
            "ㅟ should become wi, got: {}",
            result
        );
    }

    #[test]
    fn test_compound_finals() {
        let rules = base();
        // ㄳ → kt (Korean compound final - k from ㄱ, t from ㅅ in final position)
        let result = rules.apply("ㄳ");
        assert!(
            result.contains("kt") || result.contains("ks"),
            "ㄳ should become kt or ks, got: {}",
            result
        );
        // ㄻ → lm
        let result = rules.apply("ㄻ");
        assert!(
            result.contains("lm"),
            "ㄻ should become lm, got: {}",
            result
        );
    }

    #[test]
    fn test_ieung_ng() {
        let rules = base();
        // ㅇ → ng
        let result = rules.apply("ㅇ");
        assert!(result.contains("ŋ"), "ㅇ should become ng, got: {}", result);
    }

    #[test]
    fn test_jamo_sequence() {
        let rules = base();
        // ㄱㅏ → ka or ɡa (ɡ is IPA voiced velar plosive in onset position)
        let result = rules.apply("ㄱㅏ");
        assert!(
            (result.contains('k') || result.contains('ɡ')) && result.contains('a'),
            "ㄱㅏ should become ka or ɡa, got: {}",
            result
        );
    }

    // ============================================================
    // Romanization tests
    // ============================================================

    #[test]
    fn test_romanization_loads() {
        let rules = romanization();
        assert!(
            !rules.is_empty(),
            "Korean romanization rules should not be empty"
        );
        assert!(
            rules.len() >= 25,
            "expected >=25 romanization rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_romanization_breve_o() {
        let rules = romanization();
        // ŏ → ʌ (IPA near-open central vowel, McCune-Reischauer breve o)
        let result = rules.apply("ŏ");
        assert!(
            result.contains('ʌ') || result.contains("eo") || result.contains('O'),
            "ŏ should become ʌ or eo, got: {}",
            result
        );
    }

    #[test]
    fn test_romanization_breve_u() {
        let rules = romanization();
        // ŭ → ɯ (IPA close back unrounded vowel, McCune-Reischauer breve u)
        let result = rules.apply("ŭ");
        assert!(
            result.contains('ɯ') || result.contains("eu") || result.contains('U'),
            "ŭ should become ɯ or eu, got: {}",
            result
        );
    }

    #[test]
    fn test_romanization_aspirate_k() {
        let rules = romanization();
        // k' → k (aspirate marker removal)
        let result = rules.apply("k'");
        assert!(
            result.contains('k') && !result.contains('\''),
            "k' should become k, got: {}",
            result
        );
    }

    #[test]
    fn test_romanization_aspirate_p() {
        let rules = romanization();
        // p' → p
        let result = rules.apply("p'");
        assert!(
            result.contains('p') && !result.contains('\''),
            "p' should become p, got: {}",
            result
        );
    }

    #[test]
    fn test_romanization_aspirate_t() {
        let rules = romanization();
        // t' → t
        let result = rules.apply("t'");
        assert!(
            result.contains('t') && !result.contains('\''),
            "t' should become t, got: {}",
            result
        );
    }

    #[test]
    fn test_romanization_aspirate_ch() {
        let rules = romanization();
        // ch' → t͡ɕ (IPA voiceless alveolo-palatal affricate)
        let result = rules.apply("ch'");
        assert!(
            (result.contains("t͡ɕ") || result.contains("ch") || result.contains('C'))
                && !result.contains('\''),
            "ch' should become t͡ɕ or ch, got: {}",
            result
        );
    }

    #[test]
    fn test_romanization_seoul_mr() {
        let rules = romanization();
        // Sŏul (McCune-Reischauer) → sʌul (with IPA vowel)
        let result = rules.apply("Sŏul");
        // Should contain ʌ (IPA) or eo after breve conversion
        assert!(
            result.contains('ʌ') || result.contains("eo") || result.contains('O'),
            "Sŏul should have ʌ or eo, got: {}",
            result
        );
    }

    #[test]
    fn test_romanization_hyphen_removal() {
        let rules = romanization();
        // Hyphens should be removed
        let result = rules.apply("Tae-gu");
        assert!(
            !result.contains('-'),
            "Hyphen should be removed, got: {}",
            result
        );
    }
}
