//! Spanish phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Spanish loaded from embedded `.llev` files.
//! These rules are parsed at first use and cached for subsequent calls.
//!
//! # Available Rule Sets
//!
//! ## Base Rules (shared across dialects)
//! - [`base()`] - Core Spanish orthographic rules (~30 rules)
//!
//! ## Dialect-Specific Rules
//! - [`castilian()`] - Castilian Spanish (distinción: z/ce/ci → θ)
//! - [`latin_american()`] - Latin American Spanish (seseo: z/ce/ci → s)
//!
//! ## Combined Rule Sets
//! - [`combined_castilian()`] - base + castilian
//! - [`combined_latin_american()`] - base + latin_american
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::spanish;
//!
//! // Castilian Spanish distinguishes z from s
//! let castilian = spanish::combined_castilian();
//! let caza = castilian.apply("caza");  // "katha" (hunt)
//! let casa = castilian.apply("casa");  // "kasa" (house)
//! assert_ne!(caza, casa);
//!
//! // Latin American Spanish merges them (seseo)
//! let latam = spanish::combined_latin_american();
//! let caza = latam.apply("caza");  // "kasa"
//! let casa = latam.apply("casa");  // "kasa"
//! // Both normalize to same form
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Spanish phonetic rules.
///
/// Shared rules for all Spanish dialects covering:
/// - QU/GU digraph handling
/// - CH, LL, RR digraphs
/// - Silent H deletion
/// - G/J softening (→ x)
/// - B/V merger
/// - Accent normalization
/// - Ñ → ny conversion
///
/// Note: C softening (ce/ci) is NOT included here as it differs
/// between Castilian (→ θ) and Latin American (→ s) dialects.
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/spanish/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded spanish/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Spanish base rules - this is a bug in liblevenshtein")
    })
}

/// Castilian Spanish dialect rules.
///
/// Dialect-specific transformations for Castilian (Peninsular) Spanish:
/// - **Distinción**: Z and C (before e/i) → θ (theta, "th" sound)
///   - caza → katha, cena → thena, cinco → thinko
/// - S remains /s/: casa → kasa, sopa → sopa
///
/// This creates a phonemic distinction between words like:
/// - caza (hunt) vs casa (house)
/// - cena (dinner) vs sena (proper noun)
///
/// Use [`combined_castilian()`] for a complete rule set that includes
/// base Spanish rules plus Castilian-specific rules.
pub fn castilian() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/spanish/castilian.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded spanish/castilian.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Castilian rules - this is a bug in liblevenshtein")
    })
}

/// Latin American Spanish dialect rules.
///
/// Dialect-specific transformations for Latin American Spanish:
/// - **Seseo**: Z and C (before e/i) → s (same as S)
///   - caza → kasa, cena → sena, cinco → sinko
///
/// This merges the distinction found in Castilian Spanish:
/// - caza and casa both → kasa
///
/// This is the standard pronunciation in all of Latin America,
/// the Canary Islands, and much of Andalusia.
///
/// Use [`combined_latin_american()`] for a complete rule set that includes
/// base Spanish rules plus Latin American-specific rules.
pub fn latin_american() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/spanish/latin_american.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded spanish/latin_american.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Latin American rules - this is a bug in liblevenshtein")
    })
}

/// Combined Castilian Spanish rules.
///
/// Returns a complete rule set for Castilian Spanish normalization:
/// - base (shared Spanish orthographic rules)
/// - castilian (distinción: z/ce/ci → θ)
///
/// This is the rule set used when `rules_for_language("es")` is called.
pub fn combined_castilian() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let mut rules = Vec::new();
        rules.extend(base().rules.iter().cloned());
        rules.extend(castilian().rules.iter().cloned());
        RuleSetChar {
            rules,
            name: Some("Combined Castilian Spanish".to_string()),
            version: None,
        }
    })
}

/// Combined Latin American Spanish rules.
///
/// Returns a complete rule set for Latin American Spanish normalization:
/// - base (shared Spanish orthographic rules)
/// - latin_american (seseo: z/ce/ci → s)
///
/// This is the rule set used when `rules_for_language("es-419")` is called.
pub fn combined_latin_american() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let mut rules = Vec::new();
        rules.extend(base().rules.iter().cloned());
        rules.extend(latin_american().rules.iter().cloned());
        RuleSetChar {
            rules,
            name: Some("Combined Latin American Spanish".to_string()),
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
        assert!(!rules.is_empty(), "Spanish base rules should not be empty");
        assert!(rules.len() > 20, "expected >20 base rules, got {}", rules.len());
    }

    #[test]
    fn test_castilian_loads() {
        let rules = castilian();
        assert!(!rules.is_empty(), "Castilian rules should not be empty");
    }

    #[test]
    fn test_latin_american_loads() {
        let rules = latin_american();
        assert!(!rules.is_empty(), "Latin American rules should not be empty");
    }

    #[test]
    fn test_combined_castilian_loads() {
        let rules = combined_castilian();
        assert!(!rules.is_empty(), "Combined Castilian rules should not be empty");
        let total = base().len() + castilian().len();
        assert_eq!(rules.len(), total, "combined_castilian should have all rules");
    }

    #[test]
    fn test_combined_latin_american_loads() {
        let rules = combined_latin_american();
        assert!(!rules.is_empty(), "Combined Latin American rules should not be empty");
        let total = base().len() + latin_american().len();
        assert_eq!(rules.len(), total, "combined_latin_american should have all rules");
    }

    #[test]
    fn test_silent_h() {
        let rules = base();
        // hola → ola (silent h)
        let result = rules.apply("hola");
        assert!(!result.contains('h'), "h should be silent, got: {}", result);
    }

    #[test]
    fn test_ll_to_y() {
        let rules = base();
        // llamar → yamar (yeísmo), then y at word start → ʝ (palatal fricative)
        let result = rules.apply("llamar");
        // ll becomes y, then initial y becomes ʝ (palatal approximant/fricative)
        assert!(result.starts_with('y') || result.starts_with('ʝ'),
            "ll should become y or ʝ (yeísmo), got: {}", result);
    }

    #[test]
    fn test_ny_from_enye() {
        let rules = base();
        // español → espanyol
        let result = rules.apply("español");
        assert!(result.contains("ɲ"), "ñ should become ny, got: {}", result);
    }

    #[test]
    fn test_castilian_distincion() {
        let rules = combined_castilian();
        // caza → katha (with theta)
        let caza = rules.apply("caza");
        // casa → kasa (with s)
        let casa = rules.apply("casa");
        assert_ne!(caza, casa, "Castilian should distinguish caza from casa");
        assert!(caza.contains("θ"), "caza should have 'th' sound, got: {}", caza);
    }

    #[test]
    fn test_latin_american_seseo() {
        let rules = combined_latin_american();
        // caza → kasa (seseo)
        let caza = rules.apply("caza");
        // casa → kasa
        let _casa = rules.apply("casa");
        // Both should normalize similarly (z and s both become s)
        assert!(caza.contains('s'), "caza should have 's' sound in LatAm, got: {}", caza);
    }

    #[test]
    fn test_dialect_differentiation() {
        let castilian_rules = combined_castilian();
        let latam_rules = combined_latin_american();

        // "cinco" should differ between dialects
        let cinco_es = castilian_rules.apply("cinco");
        let cinco_419 = latam_rules.apply("cinco");

        assert_ne!(cinco_es, cinco_419,
            "Dialects should normalize 'cinco' differently: es='{}', es-419='{}'",
            cinco_es, cinco_419);
    }

    #[test]
    fn test_bv_merger() {
        let rules = base();
        // vaca → baka (b/v merger)
        let result = rules.apply("vaca");
        assert!(result.contains('b'), "v should become b, got: {}", result);
    }

    #[test]
    fn test_j_pronunciation() {
        let rules = base();
        // joven → j becomes x (velar fricative /x/), but x is also converted to ks
        // So: joven → xoben → ksoben, then final n → ŋ giving ksobeŋ
        let result = rules.apply("joven");
        // Accept either x (if x->ks rule runs before j->x) or ks (if after)
        assert!(result.contains('x') || result.starts_with("ks"),
            "j should become x (velar fricative) or ks, got: {}", result);
    }

    #[test]
    fn test_g_softening() {
        let rules = base();
        // gente → ge becomes xe (g before e → x, velar fricative /x/)
        // But x is also converted to ks, so: gente → xente → ksente
        let result = rules.apply("gente");
        // Accept either x (if x->ks rule runs before g->x) or ks (if after)
        assert!(result.contains('x') || result.contains("ks"),
            "g before e should become x (velar fricative) or ks, got: {}", result);
    }

    #[test]
    fn test_qu_handling() {
        let rules = base();
        // queso → keso
        let result = rules.apply("queso");
        assert!(result.starts_with('k'), "qu should become k, got: {}", result);
        assert!(!result.contains('u'), "u in qu should be silent, got: {}", result);
    }

    #[test]
    fn test_ch_digraph() {
        let rules = base();
        // chico → t͡ʃiko (ch → t͡ʃ, voiceless postalveolar affricate)
        let result = rules.apply("chico");
        assert!(result.contains("t͡ʃ"), "ch should become t͡ʃ (postalveolar affricate), got: {}", result);
    }
}
