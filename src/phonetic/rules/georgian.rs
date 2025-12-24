//! Georgian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Georgian (ქართული) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Georgian phonetic normalization handles:
//! - **Mkhedruli script**: Unique alphabet with 33 letters
//! - **No case distinction**: Modern Georgian has no uppercase/lowercase
//! - **Ejective consonants**: კ, პ, ტ, წ, ჭ, ყ (voiced with glottal closure)
//! - **Aspirated consonants**: თ, ფ, ქ, ც, ჩ
//! - **Rich consonant inventory**: 28 consonants, 5 vowels
//!
//! # Georgian Consonant Series
//!
//! Georgian distinguishes three series of stops and affricates:
//! - **Voiced**: ბ(b), გ(g), დ(d), ძ(dz), ჯ(j)
//! - **Aspirated**: თ(T), ფ(P), ქ(K), ც(TS), ჩ(CH)
//! - **Ejective**: კ(k'), პ(p'), ტ(t'), წ(ts'), ჭ(ch'), ყ(q')
//!
//! # Phonetic Markers
//!
//! Uses apostrophe for ejectives and uppercase for aspirated:
//! - k', p', t' = ejective stops
//! - ts', ch', q' = ejective affricates/uvular
//! - T, P, K = aspirated stops
//! - TS, CH = aspirated affricates
//! - GH = voiced velar fricative (ღ)
//! - SH, ZH = post-alveolar fricatives
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Georgian phonetic rules (~50 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::georgian;
//!
//! let rules = georgian::base();
//!
//! // Georgian vowels
//! let result = rules.apply("ა");
//! assert!(result.contains('a'), "ა → a");
//!
//! // Ejective consonant
//! let result = rules.apply("კ");
//! assert!(result.contains("k'"), "კ → k' (ejective)");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Georgian base phonetic rules.
///
/// Complete phonetic normalization rules for Georgian:
///
/// ## Vowels (weight 0.05)
/// - ა→a, ე→e, ი→i, ო→o, უ→u
///
/// ## Voiced Stops (weight 0.05)
/// - ბ→b, გ→g, დ→d
///
/// ## Aspirated Stops (weight 0.05)
/// - თ→T, ფ→P, ქ→K
///
/// ## Ejective Stops (weight 0.02)
/// - კ→k', პ→p', ტ→t'
///
/// ## Voiced Affricates (weight 0.05)
/// - ძ→dz, ჯ→j
///
/// ## Aspirated Affricates (weight 0.05)
/// - ც→TS, ჩ→CH
///
/// ## Ejective Affricates (weight 0.02)
/// - წ→ts', ჭ→ch'
///
/// ## Sibilants (weight 0.05)
/// - ზ→z, ს→s, ჟ→ZH, შ→SH
///
/// ## Uvular/Velar (weight 0.02-0.05)
/// - ყ→q' (ejective uvular), ღ→GH, ხ→x
///
/// ## Sonorants (weight 0.05)
/// - მ→m, ნ→n, ლ→l, რ→r, ვ→v
///
/// ## Simplification (weight 0.2)
/// - Ejective markers removed: k'→k, p'→p, etc.
/// - Aspirated simplified: T→t, P→p, etc.
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/georgian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded georgian/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Georgian base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Georgian base rules should not be empty");
        assert!(
            rules.len() >= 40,
            "expected >=40 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // VOWEL TESTS
    // ============================================================

    #[test]
    fn test_vowels() {
        let rules = base();
        let result = rules.apply("ა");
        assert!(result.contains('a'), "ა should become a, got: {}", result);
        let result = rules.apply("ე");
        assert!(result.contains('e'), "ე should become e, got: {}", result);
        let result = rules.apply("ი");
        assert!(result.contains('i'), "ი should become i, got: {}", result);
        let result = rules.apply("ო");
        assert!(result.contains('o'), "ო should become o, got: {}", result);
        let result = rules.apply("უ");
        assert!(result.contains('u'), "უ should become u, got: {}", result);
    }

    // ============================================================
    // STOP CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_voiced_stops() {
        let rules = base();
        let result = rules.apply("ბ");
        assert!(result.contains('b'), "ბ should become b, got: {}", result);
        let result = rules.apply("გ");
        assert!(result.contains('g'), "გ should become g, got: {}", result);
        let result = rules.apply("დ");
        assert!(result.contains('d'), "დ should become d, got: {}", result);
    }

    #[test]
    fn test_aspirated_stops() {
        let rules = base();
        // Note: Aspirated markers (T, P, K) are simplified to lowercase (t, p, k)
        let result = rules.apply("თ");
        assert!(result.contains('t'), "თ should become t (aspirated T simplified), got: {}", result);
        let result = rules.apply("ფ");
        assert!(result.contains('p'), "ფ should become p (aspirated P simplified), got: {}", result);
        let result = rules.apply("ქ");
        assert!(result.contains('k'), "ქ should become k (aspirated K simplified), got: {}", result);
    }

    #[test]
    fn test_ejective_stops() {
        let rules = base();
        // Note: Ejective markers (k', p', t') are simplified to plain consonants
        let result = rules.apply("კ");
        assert!(
            result.contains('k'),
            "კ should become k (ejective k' simplified), got: {}",
            result
        );
        let result = rules.apply("პ");
        assert!(
            result.contains('p'),
            "პ should become p (ejective p' simplified), got: {}",
            result
        );
        let result = rules.apply("ტ");
        assert!(
            result.contains('t'),
            "ტ should become t (ejective t' simplified), got: {}",
            result
        );
    }

    // ============================================================
    // AFFRICATE TESTS
    // ============================================================

    #[test]
    fn test_voiced_affricates() {
        let rules = base();
        let result = rules.apply("ძ");
        assert!(
            result.contains("dz"),
            "ძ should become dz, got: {}",
            result
        );
        let result = rules.apply("ჯ");
        assert!(result.contains('j'), "ჯ should become j, got: {}", result);
    }

    #[test]
    fn test_aspirated_affricates() {
        let rules = base();
        // Note: Aspirated affricates (TS, CH) are simplified to lowercase
        let result = rules.apply("ც");
        assert!(
            result.contains("ts"),
            "ც should become ts (aspirated TS simplified), got: {}",
            result
        );
        let result = rules.apply("ჩ");
        assert!(
            result.contains("ch"),
            "ჩ should become ch (aspirated CH simplified), got: {}",
            result
        );
    }

    #[test]
    fn test_ejective_affricates() {
        let rules = base();
        // Note: Ejective affricates (ts', ch') are simplified to plain affricates
        let result = rules.apply("წ");
        assert!(
            result.contains("ts"),
            "წ should become ts (ejective ts' simplified), got: {}",
            result
        );
        let result = rules.apply("ჭ");
        assert!(
            result.contains("ch"),
            "ჭ should become ch (ejective ch' simplified), got: {}",
            result
        );
    }

    // ============================================================
    // FRICATIVE TESTS
    // ============================================================

    #[test]
    fn test_sibilants() {
        let rules = base();
        let result = rules.apply("ზ");
        assert!(result.contains('z'), "ზ should become z, got: {}", result);
        let result = rules.apply("ს");
        assert!(result.contains('s'), "ს should become s, got: {}", result);
        // Note: SH and ZH are simplified to lowercase
        let result = rules.apply("შ");
        assert!(
            result.contains("sh"),
            "შ should become sh (SH simplified), got: {}",
            result
        );
        let result = rules.apply("ჟ");
        assert!(
            result.contains("zh"),
            "ჟ should become zh (ZH simplified), got: {}",
            result
        );
    }

    #[test]
    fn test_velar_fricatives() {
        let rules = base();
        // Note: GH is simplified to lowercase
        let result = rules.apply("ღ");
        assert!(
            result.contains("gh"),
            "ღ should become gh (GH simplified), got: {}",
            result
        );
        let result = rules.apply("ხ");
        assert!(result.contains('x'), "ხ should become x, got: {}", result);
    }

    #[test]
    fn test_uvular_ejective() {
        let rules = base();
        // Note: Ejective q' is simplified to plain q
        let result = rules.apply("ყ");
        assert!(
            result.contains('q'),
            "ყ should become q (ejective q' simplified), got: {}",
            result
        );
    }

    // ============================================================
    // SONORANT TESTS
    // ============================================================

    #[test]
    fn test_nasals() {
        let rules = base();
        let result = rules.apply("მ");
        assert!(result.contains('m'), "მ should become m, got: {}", result);
        let result = rules.apply("ნ");
        assert!(result.contains('n'), "ნ should become n, got: {}", result);
    }

    #[test]
    fn test_liquids() {
        let rules = base();
        let result = rules.apply("ლ");
        assert!(result.contains('l'), "ლ should become l, got: {}", result);
        let result = rules.apply("რ");
        assert!(result.contains('r'), "რ should become r, got: {}", result);
    }

    #[test]
    fn test_approximants() {
        let rules = base();
        let result = rules.apply("ვ");
        assert!(result.contains('v'), "ვ should become v, got: {}", result);
        let result = rules.apply("ჰ");
        assert!(result.contains('h'), "ჰ should become h, got: {}", result);
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_georgia() {
        let rules = base();
        // საქართველო (Sakartvelo - Georgia)
        let result = rules.apply_full("საქართველო");
        // ს→s, ა→a, ქ→K, ა→a, რ→r, თ→T, ვ→v, ე→e, ლ→l, ო→o
        assert!(
            result.contains('s') && result.contains('r') && result.contains('v'),
            "საქართველო should contain s, r, v, got: {}",
            result
        );
    }

    #[test]
    fn test_word_tbilisi() {
        let rules = base();
        // თბილისი (Tbilisi)
        let result = rules.apply_full("თბილისი");
        // თ→T, ბ→b, ი→i, ლ→l, ი→i, ს→s, ი→i
        assert!(
            result.contains('b') && result.contains('l') && result.contains('s'),
            "თბილისი should contain b, l, s, got: {}",
            result
        );
    }

    #[test]
    fn test_word_kartuli() {
        let rules = base();
        // ქართული (Georgian language)
        let result = rules.apply_full("ქართული");
        // ქ→K→k, ა→a, რ→r, თ→T→t, უ→u, ლ→l, ი→i (simplified)
        assert!(
            result.contains('k') && result.contains('r') && result.contains('l'),
            "ქართული should contain k, r, l (simplified), got: {}",
            result
        );
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
