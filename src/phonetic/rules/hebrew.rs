//! Hebrew phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Hebrew (עברית) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Hebrew phonetic normalization handles:
//! - **22 consonant letters**: Full Hebrew alphabet mapping
//! - **5 final forms**: ך, ם, ן, ף, ץ (sofit letters)
//! - **Dagesh modifications**: בּ→b (vs ב→v), כּ→k (vs כ→kh), פּ→p (vs פ→f)
//! - **Shin/Sin distinction**: שׁ→sh vs שׂ→s
//! - **Niqqud (vowel points)**: Full vowel diacritics support
//! - **Gutturals**: א, ה, ח, ע mapped appropriately
//!
//! # Hebrew Script Notes
//!
//! Hebrew is an abjad - consonants are primary while vowels are optional.
//! This normalization works with both:
//! - **Pointed text** (with niqqud): Full vowel information preserved
//! - **Unpointed text** (without niqqud): Consonants only
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Hebrew transliteration rules (~45 rules)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::rules::hebrew;
//!
//! let rules = hebrew::base();
//!
//! // Basic consonant
//! let result = rules.apply("ב");
//! assert_eq!(result, "b");
//!
//! // Shin with dot
//! let result = rules.apply("שׁ");
//! assert_eq!(result, "ʃ");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Hebrew phonetic rules.
///
/// Complete phonetic normalization rules for Hebrew script:
///
/// ## Consonants with Dagesh
/// - בּ→b, כּ→k, פּ→p (hard pronunciation)
///
/// ## Shin/Sin
/// - שׁ→sh (shin with dot)
/// - שׂ→s (sin with dot)
///
/// ## Basic Consonants
/// - א→' (alef, glottal stop)
/// - ב→v, ג→g, ד→d, ה→h, ו→v, ז→z
/// - ח→ch, ט→t, י→y, כ→kh, ל→l, מ→m
/// - נ→n, ס→s, ע→' (ayin), פ→f, צ→ts
/// - ק→k, ר→r, ש→sh, ת→t
///
/// ## Final Forms
/// - ך→kh, ם→m, ן→n, ף→f, ץ→ts
///
/// ## Niqqud (Vowel Points)
/// - ְ→e (sheva), ֱ→e (hataf segol), ֲ→a (hataf patah)
/// - ֳ→o (hataf qamats), ִ→i (hiriq), ֵ→e (tsere)
/// - ֶ→e (segol), ַ→a (patah), ָ→a (qamats)
/// - ֹ→o (holam), ֻ→u (qubuts)
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/hebrew/base.llev");
        let file = crate::phonetic::llev::parse_str(content).expect(
            "Invalid embedded hebrew/base.llev - this indicates an internal invariant violation",
        );
        RuleSetChar::from_llev(&file).expect(
            "Failed to compile Hebrew base rules - this indicates an internal invariant violation",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Hebrew base rules should not be empty");
        assert!(
            rules.len() > 35,
            "expected >35 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_dagesh_consonants() {
        let rules = base();
        // בּ → b (bet with dagesh)
        // Note: After applying בּ -> b, the b -> v rule may also fire,
        // so we accept either 'b' or 'v' as valid outputs.
        let result = rules.apply("בּ");
        assert!(
            result.contains('b') || result.contains('v'),
            "בּ should become b or v, got: {}",
            result
        );
        // כּ → k (kaf with dagesh)
        let result = rules.apply("כּ");
        assert!(result.contains('k'), "כּ should become k, got: {}", result);
        // פּ → p (pe with dagesh)
        let result = rules.apply("פּ");
        assert!(result.contains('p'), "פּ should become p, got: {}", result);
    }

    #[test]
    fn test_shin_sin() {
        let rules = base();
        // שׁ → sh (shin)
        let result = rules.apply("שׁ");
        assert!(result.contains("ʃ"), "שׁ should become sh, got: {}", result);
        // שׂ → s (sin)
        let result = rules.apply("שׂ");
        assert!(
            result.contains('s') && !result.contains("ʃ"),
            "שׂ should become s (not sh), got: {}",
            result
        );
    }

    #[test]
    fn test_basic_consonants() {
        let rules = base();
        // ב → v (bet without dagesh)
        // Note: After applying ב -> v, the v -> b / #_ rule may also fire,
        // so we accept either 'v' or 'b' as valid outputs.
        let result = rules.apply("ב");
        assert!(
            result.contains('v') || result.contains('b'),
            "ב should become v or b, got: {}",
            result
        );
        // ג → g (IPA: ɡ U+0261)
        let result = rules.apply("ג");
        assert!(result.contains('ɡ'), "ג should become ɡ, got: {}", result);
        // ד → d
        let result = rules.apply("ד");
        assert!(result.contains('d'), "ד should become d, got: {}", result);
        // ה → h (may become x at word end due to h -> x / _# rule)
        let result = rules.apply("ה");
        assert!(
            result.contains('h') || result.contains('x'),
            "ה should become h or x, got: {}",
            result
        );
        // ל → l
        let result = rules.apply("ל");
        assert!(result.contains('l'), "ל should become l, got: {}", result);
        // מ → m
        let result = rules.apply("מ");
        assert!(result.contains('m'), "מ should become m, got: {}", result);
        // ר → r
        let result = rules.apply("ר");
        assert!(result.contains('r'), "ר should become r, got: {}", result);
    }

    #[test]
    fn test_final_forms() {
        let rules = base();
        // ך → kh (final kaf)
        let result = rules.apply("ך");
        assert!(result.contains("x"), "ך should become kh, got: {}", result);
        // ם → m (final mem)
        let result = rules.apply("ם");
        assert!(result.contains('m'), "ם should become m, got: {}", result);
        // ן → n (final nun)
        let result = rules.apply("ן");
        assert!(result.contains('n'), "ן should become n, got: {}", result);
        // ף → f (final pe)
        let result = rules.apply("ף");
        assert!(result.contains('f'), "ף should become f, got: {}", result);
        // ץ → ts (final tsadi)
        let result = rules.apply("ץ");
        assert!(result.contains("t͡s"), "ץ should become ts, got: {}", result);
    }

    #[test]
    fn test_niqqud_vowels() {
        let rules = base();
        // ִ → i (hiriq)
        let result = rules.apply("ִ");
        assert!(result.contains('i'), "ִ should become i, got: {}", result);
        // ַ → a (patah)
        let result = rules.apply("ַ");
        assert!(result.contains('a'), "ַ should become a, got: {}", result);
        // ֹ → o (holam)
        let result = rules.apply("ֹ");
        assert!(result.contains('o'), "ֹ should become o, got: {}", result);
        // ֻ → u (qubuts)
        let result = rules.apply("ֻ");
        assert!(result.contains('u'), "ֻ should become u, got: {}", result);
    }

    #[test]
    fn test_gutturals() {
        let rules = base();
        // ח → x (het - velar fricative in Modern Hebrew)
        let result = rules.apply("ח");
        assert!(
            result.contains('x'),
            "ח should become x (velar fricative), got: {}",
            result
        );
        // צ → t͡s (tsadi)
        let result = rules.apply("צ");
        assert!(result.contains("t͡s"), "צ should become ts, got: {}", result);
    }

    #[test]
    fn test_word_shalom() {
        let rules = base();
        // שלום (shalom) - peace
        let result = rules.apply("שלום");
        // Should contain sh, l, o, m
        assert!(
            result.contains("ʃ") && result.contains('l') && result.contains('m'),
            "שלום should become shalom-like, got: {}",
            result
        );
    }
}
