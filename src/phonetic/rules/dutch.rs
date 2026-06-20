//! Dutch phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Standard Dutch (Nederlands) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Dutch phonetic normalization handles:
//! - **IJ digraph**: ij → EI (treated as single letter, like German ü)
//! - **G/CH**: g, ch → X (guttural fricative like Scottish "loch")
//! - **OE**: oe → U (like German "Buch")
//! - **EU**: eu → OE (like German ö)
//! - **UI**: ui → OY (unique Dutch diphthong)
//! - **W**: w → V (approximant between v and w)
//! - **SCH**: sch → sX (NOT /ʃ/ like German)
//! - **Final devoicing**: b → p, d → t at word end
//! - **Long vowels**: aa → a, ee → e, oo → o, uu → u
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Dutch orthographic rules (~50 rules)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::dutch;
//!
//! let rules = dutch::base();
//!
//! // IJ digraph
//! let wijn = rules.apply("wijn");
//! assert!(wijn.contains("eɪ"), "ij → EI");
//!
//! // Guttural G
//! let goed = rules.apply("goed");
//! assert!(goed.starts_with("X"), "g → X");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Dutch phonetic rules.
///
/// Complete phonetic normalization rules for Standard Dutch (Nederlands):
///
/// ## Digraphs
/// - IJ → EI (treated as single unit)
/// - SCH → sX (guttural s + x, not /ʃ/)
/// - OE → U (like German /u/)
/// - EU → OE (like German /ø/)
/// - UI → OY (Dutch diphthong /œy/)
///
/// ## Consonants
/// - G, CH → X (guttural fricative)
/// - W → V (approximant)
/// - V → V (voiced, unlike German v→f)
/// - PH → f, TH → t (loanwords)
///
/// ## Final Devoicing
/// - b → p, d → t at word end
///
/// ## Long Vowels
/// - aa → a, ee → e, oo → o, uu → u
/// - ie → i
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/dutch/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded dutch/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Dutch base rules - this indicates an internal invariant violation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Dutch base rules should not be empty");
        assert!(
            rules.len() > 40,
            "expected >40 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_ij_digraph() {
        let rules = base();
        // wijn → ʋɛin (w→ʋ, ij→ɛi)
        let result = rules.apply("wijn");
        assert!(result.contains('ɛ'), "ij should become ɛi, got: {}", result);
    }

    #[test]
    fn test_ij_uppercase() {
        let rules = base();
        // IJsberg → ɛisberx (IJ→ɛi, g→x)
        let result = rules.apply("IJsberg");
        assert!(
            result.starts_with('ɛ'),
            "IJ should become ɛi, got: {}",
            result
        );
    }

    #[test]
    fn test_g_guttural() {
        let rules = base();
        // goed → xut (g→x, oe→u, d→t at end)
        let result = rules.apply("goed");
        assert!(
            result.starts_with('x'),
            "g should become x (guttural), got: {}",
            result
        );
    }

    #[test]
    fn test_ch_guttural() {
        let rules = base();
        // acht → axt
        let result = rules.apply("acht");
        assert!(
            result.contains('x'),
            "ch should become x (guttural), got: {}",
            result
        );
    }

    #[test]
    fn test_sch_pattern() {
        let rules = base();
        // school → sxol (sch→sx, oo→o)
        let result = rules.apply("school");
        assert!(
            result.starts_with("sx"),
            "sch should become sx (not sh like German), got: {}",
            result
        );
    }

    #[test]
    fn test_oe_vowel() {
        let rules = base();
        // boek → buk (oe→u)
        let result = rules.apply("boek");
        assert!(result.contains('u'), "oe should become u, got: {}", result);
    }

    #[test]
    fn test_eu_vowel() {
        let rules = base();
        // neus → nøs (eu→ø)
        let result = rules.apply("neus");
        assert!(result.contains('ø'), "eu should become ø, got: {}", result);
    }

    #[test]
    fn test_ui_diphthong() {
        let rules = base();
        // huis → hœys (ui→œy)
        let result = rules.apply("huis");
        assert!(result.contains('œ'), "ui should become œy, got: {}", result);
    }

    #[test]
    fn test_w_pronunciation() {
        let rules = base();
        // water → ʋater (w→ʋ)
        let result = rules.apply("water");
        assert!(
            result.starts_with('ʋ'),
            "w should become ʋ, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_d() {
        let rules = base();
        // hond → hont (d→t at end)
        let result = rules.apply("hond");
        assert!(
            result.ends_with('t'),
            "final d should become t, got: {}",
            result
        );
    }

    #[test]
    fn test_final_devoicing_b() {
        let rules = base();
        // web → wep (b→p at end) → Vep (w→V)
        let result = rules.apply("web");
        assert!(
            result.ends_with('p'),
            "final b should become p, got: {}",
            result
        );
    }

    #[test]
    fn test_long_vowel_aa() {
        let rules = base();
        // naam → nam (aa→a)
        let result = rules.apply("naam");
        assert!(
            !result.contains("aa"),
            "aa should become a, got: {}",
            result
        );
    }

    #[test]
    fn test_long_vowel_ee() {
        let rules = base();
        // been → ben (ee→e)
        let result = rules.apply("been");
        assert!(
            !result.contains("ee"),
            "ee should become e, got: {}",
            result
        );
    }

    #[test]
    fn test_long_vowel_oo() {
        let rules = base();
        // boom → bom (oo→o)
        let result = rules.apply("boom");
        assert!(
            !result.contains("oo"),
            "oo should become o, got: {}",
            result
        );
    }

    #[test]
    fn test_ei_diphthong() {
        let rules = base();
        // klein → klɛin (ei→ɛi)
        let result = rules.apply("klein");
        assert!(result.contains('ɛ'), "ei should become ɛi, got: {}", result);
    }

    #[test]
    fn test_au_diphthong() {
        let rules = base();
        // blauw → blɔuʋ (au→ɔu, w→ʋ)
        let result = rules.apply("blauw");
        assert!(result.contains('ɔ'), "au should become ɔu, got: {}", result);
    }

    #[test]
    fn test_ou_diphthong() {
        let rules = base();
        // oud → ɔut (ou→ɔu, d→t at end)
        let result = rules.apply("oud");
        assert!(result.contains('ɔ'), "ou should become ɔu, got: {}", result);
    }

    #[test]
    fn test_ie_vowel() {
        let rules = base();
        // niet → nit (ie→i)
        let result = rules.apply("niet");
        assert!(
            !result.contains("ie"),
            "ie should become i, got: {}",
            result
        );
    }

    #[test]
    fn test_double_consonants() {
        let rules = base();
        // koffie → kofi (ff→f)
        let result = rules.apply("koffie");
        assert!(
            !result.contains("ff"),
            "ff should become f, got: {}",
            result
        );
    }
}
