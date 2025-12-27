//! English phonetic normalization rules (embedded).
//!
//! Pre-compiled phonetic rules for English loaded from embedded `.llev` files.
//! These rules are parsed at first use and cached for subsequent calls.
//!
//! # Available Rule Sets
//!
//! - [`base()`] - Comprehensive phonetic transformations (62 rules)
//! - [`homophones()`] - Words that sound alike
//! - [`text_speak()`] - SMS/text abbreviations (2 -> to, u -> you, etc.)
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::rules::english;
//!
//! let rules = english::base();
//! // Use rules.apply("knight") to get normalized form
//! ```

use crate::phonetic::llev::RuleSetChar;
use std::sync::OnceLock;

/// Base phonetic rules for English.
///
/// Based on Mark Rosenfelder's (Zompist) English spelling rules for
/// phonetic similarity matching. These rules transform English orthography
/// to a normalized phonetic form.
///
/// 62 rules covering:
/// - Affrication patterns (tion, sion, cious, tious)
/// - Multi-char patterns (ough, aught, ought, tch, dge)
/// - GH rules (ghost -> gost)
/// - Digraph conversions (ch, sh, ph, th)
/// - Initial clusters (wr, wh, gn, kn, mn, pt, ps, tm)
/// - X pronunciation (exam -> egzam, box -> boks)
/// - Contextual rules (soft c/g before front vowels)
/// - Double consonant simplification
/// - Vowel digraph simplification
///
/// # Reference
///
/// Original specification: <https://zompist.com/spell.html>
pub fn base() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/english/base.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded base.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile base rules - this is a bug in liblevenshtein")
    })
}

/// Deprecated alias for [`base()`].
///
/// This function has been renamed to `base()` for consistency with other
/// language modules. The "Zompist" name refers to Mark Rosenfelder's original
/// specification at <https://zompist.com/spell.html>.
#[deprecated(since = "3.1.0", note = "Use `base()` instead for consistency with other languages")]
pub fn zompist() -> &'static RuleSetChar {
    base()
}

/// Homophone rules for English.
///
/// Maps words that sound identical to a canonical spelling for matching.
/// This allows queries like "fone" to match "phone" after normalization.
///
/// Includes:
/// - Letter-name homophones (oh -> o, aye -> i)
/// - Common homophones (too/two -> to, their/there -> their)
/// - Sound-alike words (write/right, hear/here, etc.)
pub fn homophones() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/english/homophones.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded homophones.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile homophones rules - this is a bug in liblevenshtein")
    })
}

/// Text speak rules for English.
///
/// Maps common text/SMS abbreviations to their full forms for matching.
/// This allows queries with text-speak abbreviations to match dictionary words.
///
/// Includes:
/// - Single-letter substitutions (c -> see, r -> are, u -> you)
/// - Number substitutions (2 -> to, 4 -> for, 8 -> ate)
/// - Common abbreviations (b4 -> before, l8r -> later, gr8 -> great)
/// - Letter-based abbreviations (thx -> thanks, pls -> please)
/// - Phonetic spellings (nite -> night, thru -> through)
pub fn text_speak() -> &'static RuleSetChar {
    static RULESET: OnceLock<RuleSetChar> = OnceLock::new();
    RULESET.get_or_init(|| {
        let content = include_str!("../../../data/rules/english/text_speak.llev");
        let file = crate::phonetic::llev::parse_str(content)
            .expect("Invalid embedded text_speak.llev - this is a bug in liblevenshtein");
        RuleSetChar::from_llev(&file)
            .expect("Failed to compile text_speak rules - this is a bug in liblevenshtein")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_loads() {
        let rules = base();
        // Should have rules (62 minus disabled ones)
        assert!(!rules.is_empty(), "base rules should not be empty");
        assert!(rules.len() > 50, "expected >50 base rules, got {}", rules.len());
    }

    #[test]
    #[allow(deprecated)]
    fn test_zompist_alias() {
        // Ensure deprecated alias still works
        let rules = zompist();
        assert!(!rules.is_empty(), "zompist alias should return base rules");
    }

    #[test]
    fn test_homophones_loads() {
        let rules = homophones();
        assert!(!rules.is_empty(), "homophones rules should not be empty");
    }

    #[test]
    fn test_text_speak_loads() {
        let rules = text_speak();
        assert!(!rules.is_empty(), "text_speak rules should not be empty");
    }

    #[test]
    fn test_base_applies() {
        let rules = base();
        // phone -> foʊn (ph -> f, o_e -> oʊ with magic e)
        // Note: IPA output includes diphthong /oʊ/
        let result = rules.apply("phone");
        assert!(result.contains("f") && (result.contains("oʊn") || result.contains("on")),
            "expected 'f' and 'oʊn' or 'on' in result, got: {}", result);
    }

    #[test]
    fn test_homophones_applies() {
        let rules = homophones();
        // too -> to
        let result = rules.apply("too");
        assert_eq!(result, "to", "expected 'to', got: {}", result);
    }

    #[test]
    fn test_text_speak_applies() {
        let rules = text_speak();
        // thx -> thanks (test a multi-character abbreviation that still exists)
        let result = rules.apply("thx");
        assert_eq!(result, "thanks", "expected 'thanks', got: {}", result);
    }
}
