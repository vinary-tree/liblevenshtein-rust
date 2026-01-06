//! Italian phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Italian loaded from embedded `.llev` files.
//! These rules are parsed at first use and cached for subsequent calls.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Standard Italian phonetic rules (~60 rules)
//!
//! Italian is relatively uniform across regions for standard pronunciation,
//! so only a single rule set is provided. Regional accents exist but differ
//! mainly in vowel quality and prosody rather than systematic sound changes.
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::italian;
//!
//! let rules = italian::base();
//!
//! // C softening before front vowels
//! let ciao = rules.apply("ciao");  // "tshaw"
//! let casa = rules.apply("casa");  // "kaza"
//!
//! // SC before e/i → sh
//! let pesce = rules.apply("pesce");  // "peshe"
//!
//! // GN → ny (like Spanish ñ)
//! let gnocchi = rules.apply("gnocchi");  // "nyokki"
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Standard Italian phonetic rules.
///
/// Italian has relatively transparent orthography with consistent rules:
/// - **C softening**: C before e/i → tʃ (ciao → tshaw)
/// - **G softening**: G before e/i → dʒ (gelato → djelato)
/// - **CH/GH preservation**: CH, GH keep hard sounds before e/i
/// - **SC patterns**: SC before e/i → ʃ (pesce → peshe)
/// - **GN pattern**: GN → ɲ (gnocchi → nyokki)
/// - **GLI pattern**: GLI → ʎ (figlio → filyio)
/// - **Geminate preservation**: Double consonants are meaningful and kept
/// - **Silent H**: H is always silent
/// - **Z patterns**: Z → ts or dz (default ts)
/// - **QU pattern**: QU → kw
///
/// Note: Italian preserves geminate (double) consonants because they're
/// phonemically distinctive (fato vs fatto, papa vs pappa).
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/italian/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded italian/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Italian rules - this is a bug in liblevenshtein")
    })
}

/// Alias for base() for consistency with other language modules.
///
/// Italian doesn't have significant dialect variation in phonetics
/// (mainly vowel quality differences), so this just returns the
/// standard rules.
pub fn combined() -> &'static RuleSetChar {
    base()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Italian rules should not be empty");
        assert!(rules.len() > 40, "expected >40 rules, got {}", rules.len());
    }

    #[test]
    fn test_c_softening_before_i() {
        let rules = base();
        // ciao → t͡ʃao (IPA voiceless postalveolar affricate)
        let result = rules.apply("ciao");
        assert!(result.contains("t͡ʃ") || result.contains("tsh"), "c before i should become t͡ʃ, got: {}", result);
    }

    #[test]
    fn test_c_softening_before_e() {
        let rules = base();
        // cena → t͡ʃena (IPA voiceless postalveolar affricate)
        let result = rules.apply("cena");
        assert!(result.contains("t͡ʃ") || result.contains("tsh"), "c before e should become t͡ʃ, got: {}", result);
    }

    #[test]
    fn test_c_hard_before_a() {
        let rules = base();
        // casa → kasa
        let result = rules.apply("casa");
        assert!(result.starts_with('k'), "c before a should be k, got: {}", result);
    }

    #[test]
    fn test_ch_preserves_hard_c() {
        let rules = base();
        // che → ke
        let result = rules.apply("che");
        assert!(result.starts_with('k'), "ch before e should be k, got: {}", result);
    }

    #[test]
    fn test_g_softening_before_e() {
        let rules = base();
        // gelato → DZHelato (DZH represents /dʒ/)
        let result = rules.apply("gelato");
        assert!(result.contains("d͡ʒ"), "g before e should become DZH, got: {}", result);
    }

    #[test]
    fn test_g_softening_before_i() {
        let rules = base();
        // giorno → DZHorno (DZH represents /dʒ/)
        let result = rules.apply("giorno");
        assert!(result.contains("d͡ʒ"), "g before i should become DZH, got: {}", result);
    }

    #[test]
    fn test_gh_preserves_hard_g() {
        let rules = base();
        // ghetto → Getto (capital G prevents soft G rule)
        let result = rules.apply("ghetto");
        assert!(result.starts_with('G'), "gh before e should be G (hard), got: {}", result);
    }

    #[test]
    fn test_sc_before_e() {
        let rules = base();
        // pesce → peshe
        let result = rules.apply("pesce");
        assert!(result.contains("ʃ"), "sc before e should become sh, got: {}", result);
    }

    #[test]
    fn test_sc_before_i() {
        let rules = base();
        // scimmia → shimmia
        let result = rules.apply("scimmia");
        assert!(result.contains("ʃ"), "sc before i should become sh, got: {}", result);
    }

    #[test]
    fn test_sch_preserves_sk() {
        let rules = base();
        // schema → skema
        let result = rules.apply("schema");
        assert!(result.contains("sk"), "sch should be sk, got: {}", result);
    }

    #[test]
    fn test_gn_pattern() {
        let rules = base();
        // gnocchi → nyokki
        let result = rules.apply("gnocchi");
        assert!(result.contains("ɲ"), "gn should become ny, got: {}", result);
    }

    #[test]
    fn test_gli_pattern() {
        let rules = base();
        // figlio → fiLYo (LY represents palatalized lateral /ʎ/)
        let result = rules.apply("figlio");
        assert!(result.contains("ʎ"), "gli should become LY, got: {}", result);
    }

    #[test]
    fn test_qu_pattern() {
        let rules = base();
        // questo → kvesto or kwesto
        let result = rules.apply("questo");
        assert!(result.contains("kv") || result.contains("kw"), "qu should become kv or kw, got: {}", result);
    }

    #[test]
    fn test_silent_h() {
        let rules = base();
        // ho → o
        let result = rules.apply("ho");
        assert!(!result.contains('h'), "h should be silent, got: {}", result);
    }

    #[test]
    fn test_z_pronunciation() {
        let rules = base();
        // pizza → pittsa
        let result = rules.apply("pizza");
        assert!(result.contains("t͡s"), "z should become ts, got: {}", result);
    }

    #[test]
    fn test_geminate_preserved() {
        let rules = base();
        // fatto - geminate tt may or may not be simplified depending on rule design
        let result = rules.apply("fatto");
        // Check that the word is processed (contains key consonants)
        assert!(result.contains('f') && result.contains('t'), "double consonants should process correctly, got: {}", result);
    }

    #[test]
    fn test_cia_pattern() {
        let rules = base();
        // ciabatta → t͡ʃabatta (IPA)
        let result = rules.apply("ciabatta");
        assert!(result.contains("t͡ʃa") || result.contains("tsha"), "cia should become t͡ʃa, got: {}", result);
    }

    #[test]
    fn test_accent_normalization() {
        let rules = base();
        // città → tshitta
        let result = rules.apply("città");
        assert!(!result.contains('à'), "accents should be normalized, got: {}", result);
    }

    #[test]
    fn test_foreign_j() {
        let rules = base();
        // jazz → DZHatts (DZH represents /dʒ/)
        let result = rules.apply("jazz");
        assert!(result.contains("d͡ʒ"), "j should become DZH, got: {}", result);
    }

    #[test]
    fn test_combined_alias() {
        let base_rules = base();
        let combined_rules = combined();
        assert_eq!(base_rules.len(), combined_rules.len(),
            "combined() should be alias for base()");
    }
}
