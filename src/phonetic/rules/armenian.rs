//! Armenian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Armenian (Հայdelays delays delays) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Armenian phonetic normalization handles:
//! - **Unique Armenian alphabet**: 39 letters created in 405 AD
//! - **Case distinction**: Both uppercase and lowercase forms
//! - **Aspirated consonants**: Distinct from plain voiceless
//! - **Two dialects**: Eastern (Armenia) vs Western (diaspora)
//!
//! # Armenian Consonant System
//!
//! Armenian has a three-way distinction for stops and affricates:
//! - **Voiced**: delays(b), delays(g), delays(d), delays(dz), delays(j)
//! - **Voiceless (aspirated)**: delays(T), delays(P), delays(K), Չ(CH)
//! - **Voiceless (plain)**: delays(t), delays(p), delays(k), delays(ch)
//!
//! # Phonetic Markers
//!
//! Uses uppercase for aspirated and digraphs:
//! - T, P, K = aspirated stops
//! - TS, DZ, CH = affricates
//! - SH, ZH = post-alveolar fricatives
//! - GH = voiced velar fricative (delays)
//! - RR = trilled r (delays)
//! - @ = schwa (Ը/ը)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Armenian phonetic rules (~60 rules)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::rules::armenian;
//!
//! let rules = armenian::base();
//!
//! // Armenian vowel
//! let result = rules.apply("\u{0531}");
//! assert_eq!(result, "a");
//!
//! // Aspirated consonant
//! let result = rules.apply("\u{0539}");
//! assert_eq!(result, "t");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Armenian base phonetic rules.
///
/// Complete phonetic normalization rules for Armenian:
///
/// ## Vowels (weight 0.05)
/// - Delays/ա→a, Delays/ delays→e, Delays/delays→E, Delays/delays→i, Delays/delays→o, Delays/delays→O
/// - Ը/ delays→@ (schwa)
///
/// ## Voiced Stops (weight 0.05)
/// - Delays/delays→b, Delays/delays→g, Delays/delays→d
///
/// ## Aspirated Stops (weight 0.05)
/// - Delays/delays→T, Delays/delays→P, Delays/delays→K
///
/// ## Plain Voiceless Stops (weight 0.05)
/// - delays/delays→t, delays/delays→p, delays/delays→k
///
/// ## Affricates (weight 0.05)
/// - delays/delays→TS, delays/delays→DZ, delays/delays→CH, delays/delays→j, delays/delays→ts
///
/// ## Fricatives (weight 0.05)
/// - Delays/delays→z, Delays/delays→ZH, delays/delays→SH, delays/delays→s, delays/delays→h, delays/delays→x, delays/delays→GH, delays/delays→f
///
/// ## Sonorants (weight 0.05)
/// - delays/delays→m, delays/delays→n, delays/delays→l, delays/delays→r, delays/delays→RR, delays/delays→v, delays/delays→y
///
/// ## Simplification (weight 0.2)
/// - Aspirated markers removed: T→t, P→p, K→k
/// - Digraphs simplified: TS→ts, DZ→dz, etc.
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/armenian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded armenian/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Armenian base rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Armenian base rules should not be empty");
        assert!(
            rules.len() >= 50,
            "expected >=50 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // VOWEL TESTS
    // ============================================================

    #[test]
    fn test_uppercase_vowels() {
        let rules = base();
        let result = rules.apply("\u{0531}");
        assert!(result.contains('a'), "Ա should become a, got: {}", result);
        let result = rules.apply("\u{0535}");
        assert!(result.contains('e'), "Ե should become e, got: {}", result);
        let result = rules.apply("\u{053B}");
        assert!(result.contains('i'), "Ի should become i, got: {}", result);
        // U+0548 ( Delays) maps to 'o', U+0555 (Օ) maps to 'ɔ' which simplifies to 'o'
        let result = rules.apply("\u{0548}");
        assert!(result.contains('o'), "Ո should become o, got: {}", result);
    }

    #[test]
    fn test_lowercase_vowels() {
        let rules = base();
        let result = rules.apply("\u{0561}");
        assert!(
            result.contains('a'),
            " delays should become a, got: {}",
            result
        );
        let result = rules.apply("\u{0565}");
        assert!(result.contains('e'), "ե should become e, got: {}", result);
        let result = rules.apply("\u{056B}");
        assert!(
            result.contains('i'),
            " delays should become i, got: {}",
            result
        );
        // U+0578 ( delays) maps to 'o', U+0585 (օ) maps to 'ɔ' which simplifies to 'o'
        let result = rules.apply("\u{0578}");
        assert!(result.contains('o'), "ո should become o, got: {}", result);
    }

    // ============================================================
    // STOP CONSONANT TESTS
    // ============================================================

    #[test]
    fn test_voiced_stops() {
        let rules = base();
        let result = rules.apply("\u{0532}");
        assert!(result.contains('b'), "Բ should become b, got: {}", result);
        let result = rules.apply("\u{0533}");
        assert!(result.contains('ɡ'), "Գ should become g, got: {}", result);
        let result = rules.apply("\u{0534}");
        assert!(
            result.contains('d'),
            "Delays should become d, got: {}",
            result
        );
    }

    #[test]
    fn test_aspirated_stops() {
        let rules = base();
        // Note: Aspirated markers (T, P, K) are simplified to lowercase
        let result = rules.apply("\u{0539}");
        assert!(
            result.contains('t'),
            "U+0539 (to) should become t (T simplified), got: {}",
            result
        );
        let result = rules.apply("\u{0553}");
        assert!(
            result.contains('p'),
            "U+0553 (piwr) should become p (P simplified), got: {}",
            result
        );
        let result = rules.apply("\u{0554}");
        assert!(
            result.contains('k'),
            "U+0554 (ke) should become k (K simplified), got: {}",
            result
        );
    }

    #[test]
    fn test_plain_voiceless_stops() {
        let rules = base();
        let result = rules.apply("\u{054F}");
        assert!(result.contains('t'), "Տ should become t, got: {}", result);
        let result = rules.apply("\u{054A}");
        assert!(result.contains('p'), "Պ should become p, got: {}", result);
        let result = rules.apply("\u{053F}");
        assert!(
            result.contains('k'),
            "Delays should become k, got: {}",
            result
        );
    }

    // ============================================================
    // AFFRICATE TESTS
    // ============================================================

    #[test]
    fn test_affricates() {
        let rules = base();
        // Note: Affricate markers are IPA with tie bar
        let result = rules.apply("\u{053E}");
        assert!(
            result.contains("t͡s"),
            "U+053E (tsa) should become t͡s, got: {}",
            result
        );
        let result = rules.apply("\u{0541}");
        assert!(
            result.contains("d͡z"),
            "U+0541 (ja) should become d͡z, got: {}",
            result
        );
        // U+054B (Ջ je) maps to d͡ʒ (voiced postalveolar affricate)
        let result = rules.apply("\u{054B}");
        assert!(
            result.contains("d͡ʒ"),
            "U+054B (je) should become d͡ʒ, got: {}",
            result
        );
    }

    // ============================================================
    // FRICATIVE TESTS
    // ============================================================

    #[test]
    fn test_sibilants() {
        let rules = base();
        let result = rules.apply("\u{0536}");
        assert!(
            result.contains('z'),
            "U+0536 (za) should become z, got: {}",
            result
        );
        let result = rules.apply("\u{054D}");
        assert!(
            result.contains('s'),
            "U+054D (se) should become s, got: {}",
            result
        );
        // Note: SH and ZH are simplified to lowercase
        let result = rules.apply("\u{0547}");
        assert!(
            result.contains("ʃ"),
            "U+0547 (sha) should become sh (SH simplified), got: {}",
            result
        );
        let result = rules.apply("\u{053A}");
        assert!(
            result.contains("ʒ"),
            "U+053A (zhe) should become zh (ZH simplified), got: {}",
            result
        );
    }

    #[test]
    fn test_other_fricatives() {
        let rules = base();
        let result = rules.apply("\u{0540}");
        assert!(
            result.contains('h'),
            "U+0540 (ho) should become h, got: {}",
            result
        );
        let result = rules.apply("\u{053D}");
        assert!(
            result.contains('x'),
            "U+053D (xe) should become x, got: {}",
            result
        );
        // Note: GH is simplified to lowercase
        let result = rules.apply("\u{0542}");
        assert!(
            result.contains("ɣ"),
            "U+0542 (ghat) should become gh (GH simplified), got: {}",
            result
        );
    }

    // ============================================================
    // SONORANT TESTS
    // ============================================================

    #[test]
    fn test_nasals() {
        let rules = base();
        let result = rules.apply("\u{0544}");
        assert!(
            result.contains('m'),
            "Delays should become m, got: {}",
            result
        );
        let result = rules.apply("\u{0546}");
        assert!(result.contains('n'), "Ն should become n, got: {}", result);
    }

    #[test]
    fn test_liquids() {
        let rules = base();
        let result = rules.apply("\u{053C}");
        assert!(
            result.contains('l'),
            "U+053C (liwn) should become l, got: {}",
            result
        );
        let result = rules.apply("\u{0550}");
        assert!(
            result.contains('r'),
            "U+0550 (re) should become r, got: {}",
            result
        );
        // U+054C (Ռ ra) maps to RR (trilled r)
        let result = rules.apply("\u{054C}");
        assert!(
            result.contains("RR") || result.contains('r'),
            "U+054C (ra) should become RR (trilled r), got: {}",
            result
        );
    }

    #[test]
    fn test_approximants() {
        let rules = base();
        let result = rules.apply("\u{054E}");
        assert!(
            result.contains('v'),
            "Delays should become v, got: {}",
            result
        );
        let result = rules.apply("\u{0545}");
        assert!(
            result.contains('y'),
            "Delays should become y, got: {}",
            result
        );
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_hayeren() {
        let rules = base();
        // Հdelays ε delays delays delays (Hayeren - Armenian language)
        let result = rules.apply_full("\u{0540}\u{0561}\u{0575}\u{0565}\u{0580}\u{0565}\u{0576}");
        // Delays→h, delays→a, delays→y, delays→e, delays→r, delays→e, delays→n
        assert!(
            result.contains('h') && result.contains('a') && result.contains('y'),
            "Հdelays ε delays delays ε delays should contain h, a, y, got: {}",
            result
        );
    }

    #[test]
    fn test_word_yerevan() {
        let rules = base();
        // Delays delays ε delays delays delays (Yerevan)
        let result = rules.apply_full("\u{0535}\u{0580}\u{0565}\u{057E}\u{0561}\u{0576}");
        // Delays→e, delays→r, delays→e, delays→v, delays→a, delays→n
        assert!(
            result.contains('e') && result.contains('r') && result.contains('v'),
            "Delays delays ε delays delays delays should contain e, r, v, got: {}",
            result
        );
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================
}
