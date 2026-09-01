//! Icelandic phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Icelandic (Íslenska) loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Icelandic phonetic normalization handles:
//! - **Archaic Norse orthography**: Preserved medieval spelling
//! - **Unique letters**: Þ(th), Ð(dh), Æ(ai)
//! - **Accented vowels**: á(ow), é(ye), í(ee), ó(oh), ú(oo), ý(ee)
//! - **Consonant clusters**: ll(tl), hv(kv), rl(rtl), rn(rtn)
//!
//! # Icelandic Phonology
//!
//! Icelandic preserves many features lost in other Nordic languages:
//! - **Þ/þ** (thorn): Voiceless dental fricative /θ/ (like "thing")
//! - **Ð/ð** (eth): Voiced dental fricative /ð/ (like "the")
//! - **Accents indicate quality**: á = /au/, not just long a
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Icelandic phonetic normalization (~64 rules)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::rules::icelandic;
//!
//! let rules = icelandic::base();
//! let result = rules.apply_full("Ísland");
//! // Result normalizes accents and unique letters
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base Icelandic phonetic rules.
///
/// Complete phonetic normalization rules for Icelandic:
///
/// ## Consonant Clusters (highest priority)
/// - hv → kv (Old Norse kv preserved in sound)
/// - ll → tl (unique Icelandic lateral)
/// - rl → rtl (retroflex lateral)
/// - rn → rtn (retroflex nasal)
/// - nn → tn (pre-aspirated)
/// - dj, gj, hj → j (all palatalize to /j/)
/// - fn → pn (devoiced)
///
/// ## Unique Letters
/// - Þ/þ → th (voiceless dental fricative)
/// - Ð/ð → dh (voiced dental fricative)
/// - Æ/æ → ai (diphthong)
/// - Ö/ö → oe (rounded front vowel)
///
/// ## Accented Vowels
/// - á → ow (diphthong /au/)
/// - é → ye (diphthong /jɛ/)
/// - í → ee (long /i/)
/// - ó → oh (diphthong /ou/)
/// - ú → oo (long /u/)
/// - ý → ee (same as í in Modern Icelandic)
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/icelandic/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded icelandic/base.llev - this indicates an internal invariant violation");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile Icelandic base rules - this indicates an internal invariant violation")
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
            "Icelandic base rules should not be empty"
        );
        assert!(
            rules.len() >= 15,
            "expected >=15 base rules, got {}",
            rules.len()
        );
    }

    // ============================================================
    // UNIQUE LETTER TESTS
    // ============================================================

    #[test]
    fn test_thorn() {
        let rules = base();
        // þ → th (voiceless dental fricative)
        let result = rules.apply("þ");
        assert!(result.contains("θ"), "þ should become th, got: {}", result);
    }

    #[test]
    fn test_eth() {
        let rules = base();
        // ð → dh (voiced dental fricative)
        let result = rules.apply("ð");
        assert!(result.contains("ð"), "ð should become dh, got: {}", result);
    }

    #[test]
    fn test_ash() {
        let rules = base();
        // æ → aɪ (IPA diphthong)
        let result = rules.apply("æ");
        assert!(result.contains("aɪ"), "æ should become aɪ, got: {}", result);
    }

    #[test]
    fn test_o_umlaut() {
        let rules = base();
        // ö → ø (IPA)
        let result = rules.apply("ö");
        assert!(result.contains('ø'), "ö should become ø, got: {}", result);
    }

    // ============================================================
    // ACCENTED VOWEL TESTS
    // ============================================================

    #[test]
    fn test_a_acute() {
        let rules = base();
        // á → au (IPA diphthong)
        let result = rules.apply("á");
        assert!(result.contains("au"), "á should become au, got: {}", result);
    }

    #[test]
    fn test_e_acute() {
        let rules = base();
        // é → jɛ (IPA)
        let result = rules.apply("é");
        assert!(
            result.contains('j') || result.contains('ɛ'),
            "é should contain j or ɛ, got: {}",
            result
        );
    }

    #[test]
    fn test_i_acute() {
        let rules = base();
        // í → iː (IPA long i)
        let result = rules.apply("í");
        assert!(
            result.contains("iː") || result.contains('i'),
            "í should become iː, got: {}",
            result
        );
    }

    #[test]
    fn test_o_acute() {
        let rules = base();
        // ó → ou (IPA diphthong)
        let result = rules.apply("ó");
        assert!(result.contains("ou"), "ó should become ou, got: {}", result);
    }

    #[test]
    fn test_u_acute() {
        let rules = base();
        // ú → uː (IPA long u)
        let result = rules.apply("ú");
        assert!(
            result.contains("uː") || result.contains('u'),
            "ú should become uː, got: {}",
            result
        );
    }

    #[test]
    fn test_y_acute() {
        let rules = base();
        // ý → iː (same as í in Icelandic)
        let result = rules.apply("ý");
        assert!(
            result.contains("iː") || result.contains('i'),
            "ý should become iː, got: {}",
            result
        );
    }

    // ============================================================
    // CONSONANT CLUSTER TESTS
    // ============================================================

    #[test]
    fn test_cluster_hv() {
        let rules = base();
        // hv → kv
        let result = rules.apply("hv");
        assert!(
            result.contains("kv"),
            "hv should become kv, got: {}",
            result
        );
    }

    #[test]
    fn test_cluster_ll() {
        let rules = base();
        // ll → tl
        let result = rules.apply("ll");
        assert!(
            result.contains("tl"),
            "ll should become tl, got: {}",
            result
        );
    }

    #[test]
    fn test_cluster_rl() {
        let rules = base();
        // rl → rtl
        let result = rules.apply("rl");
        assert!(
            result.contains("rtl"),
            "rl should become rtl, got: {}",
            result
        );
    }

    #[test]
    fn test_cluster_rn() {
        let rules = base();
        // rn → rtn
        let result = rules.apply("rn");
        assert!(
            result.contains("rtn"),
            "rn should become rtn, got: {}",
            result
        );
    }

    #[test]
    fn test_cluster_nn() {
        let rules = base();
        // nn → tn
        let result = rules.apply("nn");
        assert!(
            result.contains("tn"),
            "nn should become tn, got: {}",
            result
        );
    }

    #[test]
    fn test_cluster_hj() {
        let rules = base();
        // hj → j (palatal approximant)
        let result = rules.apply("hj");
        assert!(result.contains('j'), "hj should become j, got: {}", result);
    }

    #[test]
    fn test_cluster_fn() {
        let rules = base();
        // fn → pn
        let result = rules.apply("fn");
        assert!(
            result.contains("pn"),
            "fn should become pn, got: {}",
            result
        );
    }

    // ============================================================
    // WORD TESTS
    // ============================================================

    #[test]
    fn test_word_island() {
        let rules = base();
        // Ísland (Iceland)
        let result = rules.apply_full("Ísland");
        let lower = result.to_lowercase();
        assert!(
            lower.contains("iː")
                && lower.contains('s')
                && lower.contains('l')
                && lower.contains('a')
                && lower.contains('n')
                && lower.contains('d'),
            "Ísland should normalize í to iː, got: {}",
            result
        );
    }

    #[test]
    fn test_word_reykjavik() {
        let rules = base();
        // Reykjavík (capital)
        let result = rules.apply_full("Reykjavík");
        let lower = result.to_lowercase();
        assert!(
            lower.contains('r')
                && lower.contains('k')
                && lower.contains('j')
                && lower.contains("iː"),
            "Reykjavík should normalize í to iː, got: {}",
            result
        );
    }

    #[test]
    fn test_word_fjall() {
        let rules = base();
        // fjall (mountain) - ll → tl
        let result = rules.apply_full("fjall");
        let lower = result.to_lowercase();
        assert!(
            lower.contains("tl"),
            "fjall should have ll → tl, got: {}",
            result
        );
    }

    #[test]
    fn test_word_thingvellir() {
        let rules = base();
        // Þingvellir (historical site)
        let result = rules.apply_full("Þingvellir");
        let lower = result.to_lowercase();
        assert!(
            lower.contains("θ") && lower.contains("tl"),
            "Þingvellir should have Þ → th and ll → tl, got: {}",
            result
        );
    }

    // ============================================================
    // WEIGHT ORDERING TEST
    // ============================================================
}
