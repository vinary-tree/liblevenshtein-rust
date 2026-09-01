//! Tagalog phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for Tagalog/Filipino loaded from
//! embedded `.llev` files. These rules are parsed at first use and cached for
//! subsequent calls.
//!
//! # Key Features
//!
//! Tagalog phonetic normalization handles:
//! - **NG digraph**: ng→NG (velar nasal, very common in Tagalog)
//! - **Spanish loanword adaptations**: ll→ly, ñ→ny, qu→k
//! - **Borrowed consonant normalization**: f→p, v→b, z→s
//! - **Simple vowel system**: a, e, i, o, u
//! - **Glottal stop markers**: apostrophe, hyphen → removed
//!
//! # Phonetic Markers
//!
//! Uses uppercase markers to avoid rule reprocessing:
//! - NG = velar nasal (as in "nang", "ng")
//!
//! # Tagalog Language
//!
//! Tagalog is an Austronesian language spoken as a first language by about
//! a quarter of the Philippine population. It forms the basis of Filipino,
//! the national language of the Philippines.
//!
//! Key phonetic characteristics:
//! - 15 native consonants + borrowed sounds from Spanish/English
//! - Velar nasal /ŋ/ represented by "ng" digraph
//! - Spanish influence in vocabulary and some phonology
//! - No native f, v, z (borrowed words use p, b, s)
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Complete Tagalog phonetic rules (~50 rules)
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::rules::tagalog;
//!
//! let rules = tagalog::base();
//!
//! // NG digraph
//! let result = rules.apply("ang");
//! assert!(result.contains("ŋ"), "ng → NG");
//!
//! // Spanish loanword adaptation
//! let result = rules.apply("familia");
//! assert!(result.contains('p'), "f → p");
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Tagalog base phonetic rules.
///
/// Complete phonetic normalization rules for Tagalog/Filipino:
///
/// ## NG Digraph (weight 0.02)
/// - ng, Ng, NG → NG (velar nasal)
///
/// ## Spanish Loanword Adaptations (weight 0.02)
/// - ll → ly, ñ → ny, qu → k
///
/// ## Native Consonants (weight 0.05)
/// - b, d, g, h, k, l, m, n, p, r, s, t, w, y
///
/// ## Borrowed Consonant Normalization (weight 0.05)
/// - f → p, v → b, z → s, c → k, j → h, x → ks
///
/// ## Vowels (weight 0.05)
/// - a, e, i, o, u (simple 5-vowel system)
///
/// ## Glottal Stop Markers (weight 0.1)
/// - Apostrophe ('), backtick (\`), hyphen (-) → removed
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/tagalog/base.llev");
        let file = crate::phonetic::llev::parse_str(content).expect(
            "Invalid embedded tagalog/base.llev - this indicates an internal invariant violation",
        );
        RuleSetChar::from_llev(&file).expect(
            "Failed to compile Tagalog base rules - this indicates an internal invariant violation",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        assert!(!rules.is_empty(), "Tagalog base rules should not be empty");
        // ~25 rules: digraphs, Spanish adaptations, borrowed consonant transformations,
        // glottal markers, and simplification rules. No identity or case normalization rules.
        assert!(
            rules.len() >= 20,
            "expected >=20 base rules, got {}",
            rules.len()
        );
    }

    #[test]
    fn test_ng_digraph() {
        let rules = base();
        // ng → NG
        let result = rules.apply("ng");
        assert!(result.contains("ŋ"), "ng should become NG, got: {}", result);
        // ang
        let result = rules.apply("ang");
        assert!(
            result.contains("ŋ"),
            "ang should contain NG, got: {}",
            result
        );
    }

    #[test]
    fn test_spanish_ll_adaptation() {
        let rules = base();
        // ll → ʎ (palatal lateral in IPA)
        let result = rules.apply("ll");
        assert!(
            result.contains("ʎ") || result.contains("ly"),
            "ll should become ʎ or ly, got: {}",
            result
        );
    }

    #[test]
    fn test_spanish_ene_adaptation() {
        let rules = base();
        // ñ → ny
        let result = rules.apply("ñ");
        assert!(result.contains("ɲ"), "ñ should become ny, got: {}", result);
    }

    #[test]
    fn test_spanish_qu_adaptation() {
        let rules = base();
        // qu → k
        let result = rules.apply("qu");
        assert!(result.contains('k'), "qu should become k, got: {}", result);
    }

    #[test]
    fn test_borrowed_f_to_p() {
        let rules = base();
        // f -> p (native adaptation)
        let result = rules.apply("familia");
        assert!(
            result.contains('p') && !result.contains('f'),
            "familia should have p not f, got: {}",
            result
        );
    }

    #[test]
    fn test_borrowed_v_to_b() {
        let rules = base();
        // v → b (native adaptation)
        let result = rules.apply("vino");
        assert!(
            result.contains('b') && !result.contains('v'),
            "vino should have b not v, got: {}",
            result
        );
    }

    #[test]
    fn test_borrowed_z_to_s() {
        let rules = base();
        // z → s (native adaptation)
        let result = rules.apply("zona");
        assert!(
            result.contains('s') && !result.contains('z'),
            "zona should have s not z, got: {}",
            result
        );
    }

    #[test]
    fn test_native_consonants() {
        let rules = base();
        // Test some native consonants
        let result = rules.apply("bahay");
        assert!(
            result.contains('b') && result.contains('h') && result.contains('y'),
            "bahay should contain b, h, y, got: {}",
            result
        );
    }

    #[test]
    fn test_vowels() {
        let rules = base();
        let result = rules.apply("aeiou");
        // Vowels pass through unchanged in Tagalog (no explicit vowel rules)
        assert!(
            result.contains('a')
                && result.contains('e')
                && result.contains('i')
                && (result.contains('o') || result.contains('ɔ'))
                && result.contains('u'),
            "aeiou should remain, got: {}",
            result
        );
    }

    #[test]
    fn test_word_maganda() {
        let rules = base();
        // maganda (beautiful)
        let result = rules.apply("maganda");
        // 'g' passes through unchanged (no rule transforms it)
        assert!(
            result.contains('m')
                && (result.contains('g') || result.contains('ɡ'))
                && result.contains('d'),
            "maganda should contain m, g, d, got: {}",
            result
        );
    }

    #[test]
    fn test_word_pangalan() {
        let rules = base();
        // pangalan (name) - contains ng
        let result = rules.apply("pangalan");
        assert!(
            result.contains("ŋ"),
            "pangalan should contain NG, got: {}",
            result
        );
    }
}
