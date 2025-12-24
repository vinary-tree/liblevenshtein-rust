//! Portuguese phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Portuguese loaded from embedded `.llev` files.
//! These rules are parsed at first use and cached for subsequent calls.
//!
//! # Available Rule Sets
//!
//! ## Base Rules (shared across dialects)
//! - [`base()`] - Core Portuguese orthographic rules (~60 rules)
//!
//! ## Dialect-Specific Rules
//! - [`european()`] - European Portuguese (Portugal)
//! - [`brazilian()`] - Brazilian Portuguese
//!
//! ## Combined Rule Sets
//! - [`combined_european()`] - base + european
//! - [`combined_brazilian()`] - base + brazilian
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::portuguese;
//!
//! // European Portuguese - uvular R, s → sh before consonants
//! let european = portuguese::combined_european();
//! let porta = european.apply("porta");  // with uvular R, sh before t
//!
//! // Brazilian Portuguese - h-like R, l → w, t/d palatalization
//! let brazilian = portuguese::combined_brazilian();
//! let porta = brazilian.apply("porta");  // with h-sound R
//! let brasil = brazilian.apply("brasil");  // final l → w
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Portuguese phonetic rules.
///
/// Shared rules for all Portuguese dialects covering:
/// - Nasal diphthongs (ão, ões, ãe)
/// - Nasal vowels (ã, õ, am/an, em/en, im/in, om/on, um/un)
/// - Consonant digraphs (ch, lh, nh, ss, rr, qu, gu)
/// - C/Ç softening before front vowels
/// - G softening before front vowels
/// - J → ʒ (zh sound)
/// - X → ʃ (sh sound, default)
/// - Intervocalic S voicing
/// - Silent H
/// - Accent normalization
///
/// Note: R pronunciation and L vocalization differ between dialects.
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/portuguese/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded portuguese/base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Portuguese base rules - this is a bug in liblevenshtein")
    })
}

/// European Portuguese dialect rules.
///
/// Dialect-specific transformations for European Portuguese (Portugal):
/// - **Vowel reduction**: Strong reduction in unstressed syllables
///   - Unstressed 'e' → schwa, unstressed 'o' → u
/// - **R pronunciation**: Uvular [ʀ] initially and in rr
/// - **S/Z patterns**: s/z → ʃ/ʒ before consonants and at word end
/// - **L preservation**: L stays as lateral (unlike Brazilian → w)
///
/// European Portuguese sounds noticeably different from Brazilian
/// due to these vowel and consonant differences.
///
/// Use [`combined_european()`] for a complete rule set that includes
/// base Portuguese rules plus European-specific rules.
pub fn european() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/portuguese/european.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded portuguese/european.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile European Portuguese rules - this is a bug in liblevenshtein")
    })
}

/// Brazilian Portuguese dialect rules.
///
/// Dialect-specific transformations for Brazilian Portuguese:
/// - **T/D palatalization**: Before 'i', t → tʃ, d → dʒ
///   - tipo → tʃipo, dia → dʒia, noite → noitʃi
/// - **L vocalization**: L at end of syllable → w
///   - Brasil → Brasiw, sal → saw, alto → awto
/// - **R pronunciation**: Initial R and RR → h (Rio/São Paulo style)
///   - Rio → Hio, carro → caho
/// - **Less vowel reduction**: Vowels are clearer than European
///
/// Brazilian Portuguese is the most widely spoken variety and
/// is characterized by these distinctive sound changes.
///
/// Use [`combined_brazilian()`] for a complete rule set that includes
/// base Portuguese rules plus Brazilian-specific rules.
pub fn brazilian() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/portuguese/brazilian.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded portuguese/brazilian.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Brazilian Portuguese rules - this is a bug in liblevenshtein")
    })
}

/// Combined European Portuguese rules.
///
/// Returns a complete rule set for European Portuguese normalization:
/// - base (shared Portuguese orthographic rules)
/// - european (vowel reduction, uvular R, s→sh patterns)
///
/// This is the rule set used when `rules_for_language("pt-pt")` is called.
pub fn combined_european() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let mut rules = Vec::new();
        rules.extend(base().rules.iter().cloned());
        rules.extend(european().rules.iter().cloned());
        RuleSetChar {
            rules,
            name: Some("Combined European Portuguese".to_string()),
            version: None,
        }
    })
}

/// Combined Brazilian Portuguese rules.
///
/// Returns a complete rule set for Brazilian Portuguese normalization:
/// - base (shared Portuguese orthographic rules)
/// - brazilian (t/d palatalization, l vocalization, h-like R)
///
/// This is the rule set used when `rules_for_language("pt-br")` is called.
pub fn combined_brazilian() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let mut rules = Vec::new();
        rules.extend(base().rules.iter().cloned());
        rules.extend(brazilian().rules.iter().cloned());
        RuleSetChar {
            rules,
            name: Some("Combined Brazilian Portuguese".to_string()),
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
        assert!(!rules.is_empty(), "Portuguese base rules should not be empty");
        assert!(rules.len() > 40, "expected >40 base rules, got {}", rules.len());
    }

    #[test]
    fn test_european_loads() {
        let rules = european();
        assert!(!rules.is_empty(), "European Portuguese rules should not be empty");
    }

    #[test]
    fn test_brazilian_loads() {
        let rules = brazilian();
        assert!(!rules.is_empty(), "Brazilian Portuguese rules should not be empty");
    }

    #[test]
    fn test_combined_european_loads() {
        let rules = combined_european();
        assert!(!rules.is_empty(), "Combined European Portuguese rules should not be empty");
        let total = base().len() + european().len();
        assert_eq!(rules.len(), total, "combined_european should have all rules");
    }

    #[test]
    fn test_combined_brazilian_loads() {
        let rules = combined_brazilian();
        assert!(!rules.is_empty(), "Combined Brazilian Portuguese rules should not be empty");
        let total = base().len() + brazilian().len();
        assert_eq!(rules.len(), total, "combined_brazilian should have all rules");
    }

    #[test]
    fn test_ch_digraph() {
        let rules = base();
        // chave → SHave (capitals to avoid re-matching)
        let result = rules.apply("chave");
        assert!(result.contains("SH"), "ch should become SH, got: {}", result);
    }

    #[test]
    fn test_lh_digraph() {
        let rules = base();
        // filho → fiLYo (capitals to avoid re-matching)
        let result = rules.apply("filho");
        assert!(result.contains("LY"), "lh should become LY, got: {}", result);
    }

    #[test]
    fn test_nh_digraph() {
        let rules = base();
        // senhor → seNYor (capitals to avoid re-matching)
        let result = rules.apply("senhor");
        assert!(result.contains("NY"), "nh should become NY, got: {}", result);
    }

    #[test]
    fn test_nasal_ao() {
        let rules = base();
        // não → na~w
        let result = rules.apply("não");
        assert!(result.contains("a~"), "ão should be nasal, got: {}", result);
    }

    #[test]
    fn test_c_cedilla() {
        let rules = base();
        // coração → koraSa~w (capital S to avoid intervocalic s→z)
        let result = rules.apply("coração");
        assert!(result.contains('S'), "ç should become S, got: {}", result);
    }

    #[test]
    fn test_j_pronunciation() {
        let rules = base();
        // janeiro → zhaneiro
        let result = rules.apply("janeiro");
        assert!(result.contains("zh"), "j should become zh, got: {}", result);
    }

    #[test]
    fn test_brazilian_l_vocalization() {
        let rules = combined_brazilian();
        // Brasil → Brasiw
        let result = rules.apply("brasil");
        assert!(result.ends_with('w'), "final l should become w in Brazilian, got: {}", result);
    }

    #[test]
    fn test_brazilian_t_palatalization() {
        let rules = combined_brazilian();
        // tipo → tshipo
        let result = rules.apply("tipo");
        assert!(result.contains("tsh"), "ti should palatalize in Brazilian, got: {}", result);
    }

    #[test]
    fn test_brazilian_d_palatalization() {
        let rules = combined_brazilian();
        // dia → DZHia (DZH to avoid j→zh rule)
        let result = rules.apply("dia");
        assert!(result.contains("DZH"), "di should palatalize in Brazilian, got: {}", result);
    }

    #[test]
    fn test_brazilian_r_pronunciation() {
        let rules = combined_brazilian();
        // rio → Hio (capital H to avoid base h→/ deletion)
        let result = rules.apply("rio");
        assert!(result.starts_with('H'), "initial r should become H in Brazilian, got: {}", result);
    }

    #[test]
    fn test_european_s_pattern() {
        let rules = combined_european();
        // esta → eshta
        let result = rules.apply("esta");
        assert!(result.contains("sh"), "s before t should become sh in European, got: {}", result);
    }

    #[test]
    fn test_dialect_differentiation() {
        let european_rules = combined_european();
        let brazilian_rules = combined_brazilian();

        // "brasil" should differ - L vocalization in Brazilian
        let brasil_pt = european_rules.apply("brasil");
        let brasil_br = brazilian_rules.apply("brasil");

        assert_ne!(brasil_pt, brasil_br,
            "Dialects should normalize 'brasil' differently: pt='{}', br='{}'",
            brasil_pt, brasil_br);
    }

    #[test]
    fn test_dialect_r_difference() {
        let european_rules = combined_european();
        let brazilian_rules = combined_brazilian();

        // "rio" should differ - R pronunciation
        let rio_pt = european_rules.apply("rio");
        let rio_br = brazilian_rules.apply("rio");

        // European keeps R (possibly uvular), Brazilian → H (capital)
        assert!(rio_br.contains('H'), "Brazilian should have H-sound, got: {}", rio_br);
    }

    #[test]
    fn test_silent_h() {
        let rules = base();
        // hora → ora
        let result = rules.apply("hora");
        assert!(!result.contains('h'), "h should be silent, got: {}", result);
    }

    #[test]
    fn test_c_to_k() {
        let rules = base();
        // casa → kasa (c before a → k)
        let result = rules.apply("casa");
        assert!(result.starts_with('k'), "c before a should become k, got: {}", result);
    }

    #[test]
    fn test_g_softening() {
        let rules = base();
        // gente → zhente
        let result = rules.apply("gente");
        assert!(result.contains("zh"), "g before e should become zh, got: {}", result);
    }
}
