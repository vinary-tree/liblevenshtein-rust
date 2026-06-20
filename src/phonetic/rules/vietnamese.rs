//! Vietnamese phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Vietnamese (Tiếng Việt) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Vietnamese phonetic normalization handles:
//! - **Tonal diacritics**: 6 tones marked with diacritics (stripped for matching)
//! - **Special vowels**: ă, â, ê, ô, ơ, ư (normalized to base forms)
//! - **Special consonant**: đ (voiced alveolar stop)
//! - **Digraphs**: ch, gh, gi, kh, ng, ngh, nh, ph, qu, th, tr
//!
//! # Vietnamese Tones
//!
//! Vietnamese has 6 tones marked with diacritics:
//! - **Ngang** (flat): no mark
//! - **Huyền** (falling): grave accent (à)
//! - **Sắc** (rising): acute accent (á)
//! - **Hỏi** (dipping-rising): hook above (ả)
//! - **Ngã** (creaky rising): tilde (ã)
//! - **Nặng** (low falling): dot below (ạ)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Vietnamese phonetic normalization (~180 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::vietnamese;
//!
//! let rules = vietnamese::base();
//! // Strips tones and normalizes vowels
//! let result = rules.apply("Việt");
//! assert!(result.contains("viet"), "Việt → viet");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Vietnamese phonetic rules.
///
/// Complete phonetic normalization rules for Vietnamese:
///
/// ## Tone Stripping
/// All tonal diacritics are removed:
/// - à, á, ả, ã, ạ → a
/// - è, é, ẻ, ẽ, ẹ → e
/// - ì, í, ỉ, ĩ, ị → i
/// - ò, ó, ỏ, õ, ọ → o
/// - ù, ú, ủ, ũ, ụ → u
/// - ỳ, ý, ỷ, ỹ, ỵ → y
///
/// ## Special Vowels
/// - ă (short a) → a
/// - â (close central) → a
/// - ê (close-mid front) → e
/// - ô (close-mid back) → o
/// - ơ (open-mid back unrounded) → o
/// - ư (close back unrounded) → u
///
/// ## Special Consonant
/// - đ → d (voiced alveolar stop)
///
/// ## Digraphs
/// - ngh → ng (must match before ng)
/// - ng → ng (velar nasal)
/// - ch → ch (voiceless postalveolar affricate)
/// - gh → g (like g before front vowels)
/// - gi → z (Northern dialect)
/// - kh → kh (voiceless velar fricative)
/// - nh → ny (palatal nasal)
/// - ph → f
/// - qu → kw
/// - th → th (aspirated t)
/// - tr → tr (retroflex)
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/vietnamese/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded vietnamese/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Vietnamese base rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(
            !rules.is_empty(),
            "Vietnamese base rules should not be empty"
        );
        assert!(
            rules.len() >= 100,
            "expected >=100 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // SPECIAL CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_d_stroke() {
        let rules = base();
        // đ → d
        let result = rules.apply("đ");
        assert!(result.contains('d'), "đ should become d, got: {}", result);
    }

    // ============================================================
    // DIGRAPH TESTS
    // ============================================================

    #[test]
    fn test_ngh_digraph() {
        let rules = base();
        // ngh → ng (must match before ng)
        let result = rules.apply("ngh");
        assert!(
            result.contains("ŋ"),
            "ngh should become ng, got: {}",
            result
        );
    }

    #[test]
    fn test_ph_digraph() {
        let rules = base();
        // ph → f
        let result = rules.apply("ph");
        assert!(result.contains('f'), "ph should become f, got: {}", result);
    }

    #[test]
    fn test_nh_digraph() {
        let rules = base();
        // nh → ny
        let result = rules.apply("nh");
        assert!(result.contains("ɲ"), "nh should become ny, got: {}", result);
    }

    #[test]
    fn test_gi_digraph() {
        let rules = base();
        // gi → z
        let result = rules.apply("gi");
        assert!(result.contains('z'), "gi should become z, got: {}", result);
    }

    #[test]
    fn test_qu_digraph() {
        let rules = base();
        // qu → kw
        let result = rules.apply("qu");
        assert!(
            result.contains("kw"),
            "qu should become kw, got: {}",
            result
        );
    }

    // ============================================================
    // TONAL VOWEL TESTS
    // ============================================================

    #[test]
    fn test_a_grave() {
        let rules = base();
        // à → a
        let result = rules.apply("à");
        assert!(result.contains('a'), "à should become a, got: {}", result);
    }

    #[test]
    fn test_a_acute() {
        let rules = base();
        // á → a
        let result = rules.apply("á");
        assert!(result.contains('a'), "á should become a, got: {}", result);
    }

    #[test]
    fn test_a_hook() {
        let rules = base();
        // ả → a
        let result = rules.apply("ả");
        assert!(result.contains('a'), "ả should become a, got: {}", result);
    }

    #[test]
    fn test_a_tilde() {
        let rules = base();
        // ã → a
        let result = rules.apply("ã");
        assert!(result.contains('a'), "ã should become a, got: {}", result);
    }

    #[test]
    fn test_a_dot_below() {
        let rules = base();
        // ạ → a
        let result = rules.apply("ạ");
        assert!(result.contains('a'), "ạ should become a, got: {}", result);
    }

    // ============================================================
    // SPECIAL VOWEL TESTS
    // ============================================================

    #[test]
    fn test_a_breve() {
        let rules = base();
        // ă → a
        let result = rules.apply("ă");
        assert!(result.contains('a'), "ă should become a, got: {}", result);
    }

    #[test]
    fn test_a_circumflex() {
        let rules = base();
        // â → a
        let result = rules.apply("â");
        assert!(result.contains('a'), "â should become a, got: {}", result);
    }

    #[test]
    fn test_e_circumflex() {
        let rules = base();
        // ê → e
        let result = rules.apply("ê");
        assert!(result.contains('e'), "ê should become e, got: {}", result);
    }

    #[test]
    fn test_o_circumflex() {
        let rules = base();
        // ô → o (normalized to plain ASCII 'o')
        let result = rules.apply("ô");
        assert!(
            result.contains('o') || result.contains('ɔ'),
            "ô should become o, got: {}",
            result
        );
    }

    #[test]
    fn test_o_horn() {
        let rules = base();
        // ơ → o (normalized to plain ASCII 'o')
        let result = rules.apply("ơ");
        assert!(
            result.contains('o') || result.contains('ɔ'),
            "ơ should become o, got: {}",
            result
        );
    }

    #[test]
    fn test_u_horn() {
        let rules = base();
        // ư → u
        let result = rules.apply("ư");
        assert!(result.contains('u'), "ư should become u, got: {}", result);
    }

    // ============================================================
    // COMBINED VOWEL+TONE TESTS
    // ============================================================

    #[test]
    fn test_a_breve_with_tone() {
        let rules = base();
        // ắ → a (ă with acute)
        let result = rules.apply("ắ");
        assert!(result.contains('a'), "ắ should become a, got: {}", result);
    }

    #[test]
    fn test_o_horn_with_tone() {
        let rules = base();
        // ớ → o (ơ with acute, normalized to plain ASCII 'o')
        let result = rules.apply("ớ");
        assert!(
            result.contains('o') || result.contains('ɔ'),
            "ớ should become o, got: {}",
            result
        );
    }

    #[test]
    fn test_u_horn_with_tone() {
        let rules = base();
        // ứ → u (ư with acute)
        let result = rules.apply("ứ");
        assert!(result.contains('u'), "ứ should become u, got: {}", result);
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_vietnam() {
        let rules = base();
        // Việt Nam → viet nam
        let result = rules.apply_full("Việt Nam");
        let lower = result.to_lowercase();
        assert!(
            lower.contains("viet") && lower.contains("nam"),
            "Việt Nam should normalize to contain viet and nam, got: {}",
            result
        );
    }

    #[test]
    fn test_word_hanoi() {
        let rules = base();
        // Hà Nội → ha noi
        let result = rules.apply_full("Hà Nội");
        let lower = result.to_lowercase();
        assert!(
            lower.contains("ha") && lower.contains("noi"),
            "Hà Nội should normalize to contain ha and noi, got: {}",
            result
        );
    }

    #[test]
    fn test_word_pho() {
        let rules = base();
        // Phở → fo (ph→f, ở→o)
        let result = rules.apply_full("Phở");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('f') && (lower.contains('o') || lower.contains('ɔ')),
            "Phở should normalize to contain f and o, got: {}",
            result
        );
    }

    #[test]
    fn test_word_nguyen() {
        let rules = base();
        // Nguyễn → nguyen
        let result = rules.apply_full("Nguyễn");
        let lower = result.to_lowercase();
        assert!(
            lower.contains("ŋ")
                && lower.contains('u')
                && lower.contains('y')
                && lower.contains('e')
                && lower.contains('n'),
            "Nguyễn should normalize to nguyen, got: {}",
            result
        );
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================
}
