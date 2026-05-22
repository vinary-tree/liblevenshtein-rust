//! German phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Standard German (Hochdeutsch) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! German phonetic normalization handles:
//! - **Umlauts**: ä → ae, ö → oe, ü → ue
//! - **Eszett**: ß → ss
//! - **CH variations**: ich-Laut (after front vowels) vs ach-Laut (after back vowels)
//! - **Final devoicing**: b → p, d → t, g → k at word end
//! - **W/V pronunciation**: w → v, v → f
//! - **SP/ST at word start**: sp → shp, st → sht
//! - **Silent H**: Lengthening H after vowels is deleted
//! - **Z pronunciation**: z → ts (affricate)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete German orthographic rules (~50 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::german;
//!
//! let rules = german::base();
//!
//! // Umlaut normalization
//! let muenchen = rules.apply("München");
//! assert!(muenchen.contains("ue"), "ü → ue");
//!
//! // Eszett handling
//! let strasse = rules.apply("Straße");
//! assert!(strasse.contains("ss"), "ß → ss");
//!
//! // Final devoicing
//! let hund = rules.apply("Hund");
//! assert!(hund.ends_with('t'), "d → t at word end");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base German phonetic rules.
///
/// Complete phonetic normalization rules for Standard German (Hochdeutsch):
///
/// ## Digraphs
/// - SCH → sh (Schule → shule)
/// - TSCH → tsh (Deutsch → doytsh)
/// - CK → k (Stück → shtuek)
/// - PH → f (Philosophie → filosofi)
/// - TH → t (Theater → teater)
///
/// ## CH Sounds
/// - **ich-Laut**: CH after front vowels (i, e, ä, ö, ü) → X (palatal)
/// - **ach-Laut**: CH after back vowels (a, o, u) → K (velar)
/// - CH at word start → k (Charakter → karakter)
///
/// ## Special Characters
/// - ß → ss (Straße → strasse)
/// - ä → ae, ö → oe, ü → ue
///
/// ## Final Devoicing (Auslautverhärtung)
/// - b → p, d → t, g → k at word end
///
/// ## Other Features
/// - W → v (German W sounds like English V)
/// - V → f (in native words)
/// - Z → ts (Zeitung → tsaitung)
/// - QU → kv (Quelle → kvele)
/// - SP/ST at word start → shp/sht (spät → shpaet, Stein → shtain)
/// - Silent H after vowels (Sahne → sane)
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/german/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded german/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile German base rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "German base rules should not be empty");
        assert!(
            rules.len() > 40,
            "expected >40 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_eszett() {
        let rules = base();
        // Straße → shtrase (St→sht, ß→ss→s via double consonant simplification)
        // For phonetic matching, both Straße and Strasse normalize to the same output
        let result = rules.apply("Straße");
        // The ß produces ss which then simplifies to s for phonetic matching
        assert!(
            result.contains('s'),
            "ß should become s (via ss→s), got: {}",
            result
        );
        // Verify both forms normalize the same
        let strasse_result = rules.apply("Strasse");
        assert_eq!(
            result, strasse_result,
            "Straße and Strasse should normalize identically"
        );
    }

    #[test]
    fn test_umlaut_a() {
        let rules = base();
        // Käse → kɛse (ä → ɛ IPA open-mid front unrounded vowel)
        let result = rules.apply("Käse");
        assert!(
            result.contains('ɛ') || result.contains("ae"),
            "ä should become ɛ, got: {}",
            result
        );
    }

    #[test]
    fn test_umlaut_o() {
        let rules = base();
        // schön → ʃøn (ö → ø IPA front rounded vowel)
        let result = rules.apply("schön");
        assert!(
            result.contains('ø') || result.contains("oe"),
            "ö should become ø, got: {}",
            result
        );
    }

    #[test]
    fn test_umlaut_u() {
        let rules = base();
        // München → myŋken (ü → y IPA front rounded high vowel)
        let result = rules.apply("München");
        assert!(
            result.contains('y') || result.contains("ue"),
            "ü should become y, got: {}",
            result
        );
    }

    #[test]
    fn test_sch_digraph() {
        let rules = base();
        // Schule → shule
        let result = rules.apply("Schule");
        assert!(
            result.starts_with("ʃ"),
            "sch should become sh, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_d() {
        let rules = base();
        // Hund → hunt (d→t at end)
        let result = rules.apply("Hund");
        assert!(
            result.ends_with('t'),
            "final d should become t, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_b() {
        let rules = base();
        // gelb → gelp (b→p at end)
        let result = rules.apply("gelb");
        assert!(
            result.ends_with('p'),
            "final b should become p, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_g() {
        let rules = base();
        // Tag → tak (g→k at end)
        let result = rules.apply("Tag");
        assert!(
            result.ends_with('k'),
            "final g should become k, got: {}",
            result
        );
    }

    #[test]
    fn test_w_pronunciation() {
        let rules = base();
        // Wasser → vaser or faser (W→v, v may become f in German)
        let result = rules.apply("Wasser");
        assert!(
            result.starts_with('v') || result.starts_with('V') || result.starts_with('f'),
            "W should become v or f, got: {}",
            result
        );
    }

    #[test]
    fn test_z_pronunciation() {
        let rules = base();
        // Zeit → tsait
        let result = rules.apply("Zeit");
        assert!(
            result.starts_with("t͡s"),
            "z should become ts, got: {}",
            result
        );
    }

    #[test]
    fn test_sp_initial() {
        let rules = base();
        // Spiel → ʃpiːl (sp → ʃp at word start)
        let result = rules.apply("Spiel");
        assert!(
            result.starts_with("ʃp") || result.starts_with("shp"),
            "initial sp should become ʃp, got: {}",
            result
        );
    }

    #[test]
    fn test_st_initial() {
        let rules = base();
        // Stein → ʃtaɪn (st → ʃt at word start)
        let result = rules.apply("Stein");
        assert!(
            result.starts_with("ʃt") || result.starts_with("sht"),
            "initial st should become ʃt, got: {}",
            result
        );
    }

    #[test]
    fn test_ie_vowel() {
        let rules = base();
        // Liebe → libe
        let result = rules.apply("Liebe");
        // ie becomes i
        assert!(
            result.contains('i') && !result.contains("ie"),
            "ie should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_ei_diphthong() {
        let rules = base();
        // Eis → AIs (Ei→AI uppercase to prevent rule chaining)
        let result = rules.apply("Eis");
        assert!(
            result.contains("aɪ"),
            "ei should become AI (uppercase), got: {}",
            result
        );
    }

    #[test]
    fn test_eu_diphthong() {
        let rules = base();
        // neu → nOY (eu→OY uppercase to prevent rule chaining)
        let result = rules.apply("neu");
        assert!(
            result.contains("ɔʏ"),
            "eu should become OY (uppercase), got: {}",
            result
        );
    }

    #[test]
    fn test_au_diphthong() {
        let rules = base();
        // Haus → HAUs (au→AU uppercase to prevent W→V chaining)
        let result = rules.apply("Haus");
        assert!(
            result.contains("aʊ"),
            "au should become AU (uppercase), got: {}",
            result
        );
    }

    #[test]
    fn test_qu_pattern() {
        let rules = base();
        // Quelle → kfele or kvele (qu → kv or kf)
        let result = rules.apply("Quelle");
        assert!(
            result.starts_with("kv") || result.starts_with("kV") || result.starts_with("kf"),
            "qu should become kv or kf, got: {}",
            result
        );
    }

    #[test]
    fn test_ph_pattern() {
        let rules = base();
        // Philosophie → filosofi
        let result = rules.apply("Philosophie");
        assert!(result.contains('f'), "ph should become f, got: {}", result);
        assert!(
            !result.contains("ph"),
            "ph should not remain, got: {}",
            result
        );
    }

    #[test]
    fn test_silent_h() {
        let rules = base();
        // Sahne → sane
        let result = rules.apply("Sahne");
        assert!(
            !result.contains("ah"),
            "ah should become a (silent h), got: {}",
            result
        );
    }

    #[test]
    fn test_ck_digraph() {
        let rules = base();
        // Stück → shtueK (ck→k, st→sht, ü→ue)
        let result = rules.apply("Stück");
        assert!(
            !result.contains("ck"),
            "ck should become k, got: {}",
            result
        );
    }

    #[test]
    fn test_double_consonants() {
        let rules = base();
        // Mutter → muter
        let result = rules.apply("Mutter");
        assert!(
            !result.contains("tt"),
            "tt should become t, got: {}",
            result
        );
    }
}
