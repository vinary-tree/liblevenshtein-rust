//! Embedded dictionary and rules for WASM builds.
//!
//! Data is embedded at compile time using include_str! to avoid
//! filesystem access, which is not available in browser WASM.

use liblevenshtein::phonetic::{parse_str, Phone, RewriteRule, RuleSet};
use std::sync::OnceLock;

/// Embedded English dictionary (one word per line, ~124k words)
const DICTIONARY_STR: &str = include_str!("../../phonetic_spellcheck/data/english_words.txt");

/// Embedded phonetic rules
const ZOMPIST_LLEV: &str = include_str!("../../../data/rules/english/zompist.llev");
const HOMOPHONES_LLEV: &str = include_str!("../../../data/rules/english/homophones.llev");
const TEXT_SPEAK_LLEV: &str = include_str!("../../../data/rules/english/text_speak.llev");

/// Lazily parsed dictionary - returns a static reference to avoid repeated parsing
pub fn dictionary() -> &'static Vec<String> {
    static DICT: OnceLock<Vec<String>> = OnceLock::new();
    DICT.get_or_init(|| {
        DICTIONARY_STR
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| line.trim().to_lowercase())
            .filter(|line| line.chars().all(|c| c.is_ascii_alphabetic()))
            .collect()
    })
}

/// Lazily parsed combined rules - returns a static reference
pub fn rules() -> &'static Vec<RewriteRule> {
    static RULES: OnceLock<Vec<RewriteRule>> = OnceLock::new();
    RULES.get_or_init(|| {
        let mut combined = RuleSet::default();

        // Load rules in order: text_speak, homophones, then zompist
        // This order matches the native example
        for (name, content) in [
            ("text_speak", TEXT_SPEAK_LLEV),
            ("homophones", HOMOPHONES_LLEV),
            ("zompist", ZOMPIST_LLEV),
        ] {
            let file = parse_str(content).unwrap_or_else(|e| {
                panic!("Failed to parse {}: {:?}", name, e);
            });
            let ruleset = RuleSet::from_llev(&file).unwrap_or_else(|e| {
                panic!("Failed to compile {} rules: {:?}", name, e);
            });
            combined.merge(ruleset);
        }

        combined.rules
    })
}

/// Get dictionary size without parsing (counts lines)
pub fn dictionary_size() -> usize {
    dictionary().len()
}

/// Get number of rules without parsing
pub fn rules_count() -> usize {
    rules().len()
}

/// Convert a string to a vector of Phones (used for normalization)
pub fn string_to_phones(s: &str) -> Vec<Phone> {
    s.bytes()
        .map(|b| {
            let lower = b.to_ascii_lowercase();
            if matches!(lower, b'a' | b'e' | b'i' | b'o' | b'u') {
                Phone::Vowel(lower)
            } else if b.is_ascii_alphabetic() {
                Phone::Consonant(lower)
            } else {
                Phone::Consonant(b)
            }
        })
        .collect()
}

/// Convert a vector of Phones back to a string
pub fn phones_to_string(phones: &[Phone]) -> String {
    phones
        .iter()
        .filter_map(|p| match p {
            Phone::Vowel(c) | Phone::Consonant(c) => Some(*c as char),
            Phone::Digraph(c1, _c2) => Some(*c1 as char),
            Phone::Silent => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_loads() {
        let dict = dictionary();
        // After filtering to ASCII-only words, we get ~90k words
        assert!(dict.len() > 80_000, "Dictionary should have 80k+ words, got {}", dict.len());
    }

    #[test]
    fn test_rules_load() {
        let r = rules();
        assert!(r.len() > 50, "Should have 50+ rules");
    }

    #[test]
    fn test_string_to_phones() {
        let phones = string_to_phones("phone");
        assert_eq!(phones.len(), 5);
    }
}
