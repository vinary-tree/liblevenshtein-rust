//! French phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for French loaded from embedded `.llev` files.
//! These rules are parsed at first use and cached for subsequent calls.
//!
//! # Available Rule Sets
//!
//! ## Base Rules (shared across dialects)
//! - [`base()`] - Core French orthographic rules (~80 rules)
//!
//! ## Dialect-Specific Rules
//! - [`standard()`] - Metropolitan/Standard French (Parisian)
//! - [`canadian()`] - Canadian French (Québécois)
//!
//! ## Combined Rule Sets
//! - [`combined_standard()`] - base + standard
//! - [`combined_canadian()`] - base + canadian
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::french;
//!
//! // Standard French - no affrication
//! let standard = french::combined_standard();
//! let tu = standard.apply("tu");  // "tu" preserved
//!
//! // Canadian French - affrication before high front vowels
//! let quebec = french::combined_canadian();
//! let tu = quebec.apply("tu");  // "tsu" (affricated)
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base French phonetic rules.
///
/// Shared rules for all French dialects covering:
/// - Nasal vowels (on, an, en, in, un → nasalized)
/// - Consonant digraphs (ch, ph, th, gn, qu)
/// - Vowel digraphs (ou, oi, au, eau, ai, ei, eu)
/// - -tion/-sion endings
/// - C/G softening before front vowels
/// - Silent final consonants (e, s, t, d, x, z, p)
/// - Silent H
/// - Accent normalization
/// - Double consonant simplification
///
/// Note: T/D affrication differs between Standard and Canadian French.
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/french/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded french/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile French base rules - this is a bug in liblevenshtein")
    })
}

/// Standard French (Metropolitan) dialect rules.
///
/// Dialect-specific transformations for Standard/Metropolitan French:
/// - No affrication of T/D before high front vowels
/// - /a/-/ɑ/ merger in most contexts
/// - IL/ILL → /j/ patterns
/// - -ER/-EZ verb endings → /e/
///
/// Standard French is based on Parisian (Île-de-France) pronunciation
/// and is the prestige variety used in formal contexts.
///
/// Use [`combined_standard()`] for a complete rule set that includes
/// base French rules plus Standard-specific rules.
pub fn standard() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/french/standard.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded french/standard.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Standard French rules - this is a bug in liblevenshtein")
    })
}

/// Canadian French (Québécois) dialect rules.
///
/// Dialect-specific transformations for Canadian French:
/// - **Affrication**: T/D before high front vowels (i, u, y) → ts/dz
///   - tu → tsu, dire → dzire, petit → petsit
/// - Lax vowels in closed syllables
/// - Diphthongization of long vowels
/// - IL/ILL patterns
///
/// Québécois is the variety spoken in Quebec, Canada, and has
/// distinctive phonetic features not found in European French.
///
/// Use [`combined_canadian()`] for a complete rule set that includes
/// base French rules plus Canadian-specific rules.
pub fn canadian() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/french/canadian.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded french/canadian.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Canadian French rules - this is a bug in liblevenshtein")
    })
}

/// Combined Standard French rules.
///
/// Returns a complete rule set for Standard/Metropolitan French normalization:
/// - base (shared French orthographic rules)
/// - standard (no affrication, vowel mergers)
///
/// This is the rule set used when `rules_for_language("fr")` is called.
pub fn combined_standard() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let mut rules = Vec::new();
        rules.extend(base().rules.iter().cloned());
        rules.extend(standard().rules.iter().cloned());
        RuleSetChar {
            rules,
            name: Some("Combined Standard French".to_string()),
            version: None,
        }
    })
}

/// Combined Canadian French rules.
///
/// Returns a complete rule set for Canadian French (Québécois) normalization:
/// - base (shared French orthographic rules)
/// - canadian (affrication, lax vowels, diphthongization)
///
/// This is the rule set used when `rules_for_language("fr-ca")` is called.
pub fn combined_canadian() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let mut rules = Vec::new();
        rules.extend(base().rules.iter().cloned());
        rules.extend(canadian().rules.iter().cloned());
        RuleSetChar {
            rules,
            name: Some("Combined Canadian French".to_string()),
            version: None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "French base rules should not be empty");
        assert!(rules.len() > 50, "expected >50 base rules, got {}", rules.len());
    }

    #[test]
    fn test_standard_loads() {
        let rules = standard();
        assert!(!rules.is_empty(), "Standard French rules should not be empty");
    }

    #[test]
    fn test_canadian_loads() {
        let rules = canadian();
        assert!(!rules.is_empty(), "Canadian French rules should not be empty");
    }

    #[test]
    fn test_combined_standard_loads() {
        let rules = combined_standard();
        assert!(!rules.is_empty(), "Combined Standard French rules should not be empty");
        let total = base().len() + standard().len();
        assert_eq!(rules.len(), total, "combined_standard should have all rules");
    }

    #[test]
    fn test_combined_canadian_loads() {
        let rules = combined_canadian();
        assert!(!rules.is_empty(), "Combined Canadian French rules should not be empty");
        let total = base().len() + canadian().len();
        assert_eq!(rules.len(), total, "combined_canadian should have all rules");
    }

    #[test]
    fn test_silent_h() {
        let rules = base();
        // homme → omme (silent h)
        let result = rules.apply("homme");
        assert!(!result.contains('h'), "h should be silent, got: {}", result);
    }

    #[test]
    fn test_ch_digraph() {
        let rules = base();
        // chat → SHa (ch → SH, capitals to avoid re-matching)
        let result = rules.apply("chat");
        assert!(result.contains("SH"), "ch should become SH, got: {}", result);
    }

    #[test]
    fn test_ph_digraph() {
        let rules = base();
        // photo → foto
        let result = rules.apply("photo");
        assert!(result.contains('f'), "ph should become f, got: {}", result);
    }

    #[test]
    fn test_nasal_vowels() {
        let rules = base();
        // bon → bo~ (nasal)
        let result = rules.apply("bon");
        assert!(result.contains("o~"), "on should become nasal, got: {}", result);
    }

    #[test]
    fn test_oi_digraph() {
        let rules = base();
        // moi → mWA (capitals to avoid w->v re-matching)
        let result = rules.apply("moi");
        assert!(result.contains("WA"), "oi should become WA, got: {}", result);
    }

    #[test]
    fn test_silent_final_consonants() {
        let rules = base();
        // petit → peti (silent t)
        let result = rules.apply("petit");
        assert!(!result.ends_with('t'), "final t should be silent, got: {}", result);
    }

    #[test]
    fn test_c_softening() {
        let rules = base();
        // centre → sentre
        let result = rules.apply("centre");
        assert!(result.starts_with('s'), "c before e should become s, got: {}", result);
    }

    #[test]
    fn test_g_softening() {
        let rules = base();
        // geste → ZHeste (capitals to avoid re-matching)
        let result = rules.apply("geste");
        assert!(result.contains("ZH"), "g before e should become ZH, got: {}", result);
    }

    #[test]
    fn test_canadian_affrication() {
        let rules = combined_canadian();
        // tu → tsu (affrication in Québécois)
        let result = rules.apply("tu");
        assert!(result.contains("ts"), "t before u should affricate in Québécois, got: {}", result);
    }

    #[test]
    fn test_standard_no_affrication() {
        let rules = combined_standard();
        // tu → tu (no affrication in Standard French)
        let result = rules.apply("tu");
        // Standard French should NOT have affrication
        assert!(!result.contains("ts"), "Standard French should not affricate, got: {}", result);
    }

    #[test]
    fn test_dialect_differentiation() {
        let standard_rules = combined_standard();
        let canadian_rules = combined_canadian();

        // "petit" should differ between dialects (affrication in Canadian)
        let petit_fr = standard_rules.apply("petit");
        let petit_ca = canadian_rules.apply("petit");

        // The difference may be subtle due to rule ordering
        // At minimum, Canadian should have some affrication marker
        assert!(petit_ca.contains("ts") || petit_fr != petit_ca,
            "Dialects should normalize 'petit' differently: fr='{}', fr-ca='{}'",
            petit_fr, petit_ca);
    }

    #[test]
    fn test_qu_digraph() {
        let rules = base();
        // que → ke
        let result = rules.apply("que");
        assert!(result.contains('k'), "qu should become k, got: {}", result);
    }

    #[test]
    fn test_gn_digraph() {
        let rules = base();
        // montagne → mo~taNY (capitals to avoid an->a~ re-matching)
        let result = rules.apply("montagne");
        assert!(result.contains("NY"), "gn should become NY, got: {}", result);
    }

    #[test]
    fn test_eau_trigraph() {
        let rules = base();
        // eau → o
        let result = rules.apply("eau");
        assert_eq!(result, "o", "eau should become o, got: {}", result);
    }

    #[test]
    fn test_tion_ending() {
        let rules = base();
        // nation → naSYO~ (capitals to avoid re-matching)
        let result = rules.apply("nation");
        assert!(result.contains("SYO"), "tion should become SYO~, got: {}", result);
    }
}
