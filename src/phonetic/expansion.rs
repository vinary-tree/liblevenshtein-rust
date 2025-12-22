//! Phonetic pattern expansion for reverse phonetic matching.
//!
//! This module provides functions to expand a normalized phonetic string back
//! into a regex pattern that matches all possible original spellings.
//!
//! # Motivation
//!
//! Given phonetic rules like:
//! - `ph → f`
//! - `ough → o`
//! - `tion → shun`
//!
//! If we see the normalized form "fon", we want to generate a pattern that
//! matches both "fon" and "phon" (and potentially other variants).
//!
//! This is the **inverse** of normalization: instead of collapsing spellings
//! to a canonical form, we expand from a canonical form to all possible spellings.
//!
//! # Algorithm
//!
//! For each position in the input string:
//! 1. Check if any rule's **replacement** matches at this position
//! 2. If so, create an alternation: `(original_pattern|replacement)`
//! 3. If not, emit the literal character
//!
//! # Example
//!
//! ```ignore
//! use liblevenshtein::phonetic::expansion::expand_phonetic_alternatives_char;
//! use liblevenshtein::phonetic::zompist_rules_char;
//!
//! let rules = zompist_rules_char();
//!
//! // "f" could have come from "ph"
//! let pattern = expand_phonetic_alternatives_char("fone", &rules);
//! // pattern might be "(ph|f)one" or similar
//! ```

use crate::phonetic::types::{PhoneChar, RewriteRuleChar};

/// Expand a normalized string into a regex pattern matching phonetic variants.
///
/// Given rules like `ph → f`, the string "fone" becomes "(ph|f)one"
/// because anywhere we see "f" in the output, the input could have been "ph".
///
/// # Arguments
///
/// * `input` - The normalized string to expand
/// * `rules` - The phonetic rules (used in reverse: replacement → pattern)
///
/// # Returns
///
/// A regex pattern string that matches the input and all its phonetic variants.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::phonetic::expansion::expand_phonetic_alternatives_char;
/// use liblevenshtein::phonetic::zompist_rules_char;
///
/// let rules = zompist_rules_char();
/// let pattern = expand_phonetic_alternatives_char("fone", &rules);
///
/// // The pattern will match "fone", "phone", etc.
/// ```
pub fn expand_phonetic_alternatives_char(input: &str, rules: &[RewriteRuleChar]) -> String {
    // Build a map of replacement → original patterns for efficient lookup
    let reverse_map = build_reverse_map(rules);

    let mut pattern = String::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let remaining = &input[char_byte_index(input, i)..];
        let mut matched = false;

        // Check if any replacement matches at this position
        // Try longer replacements first to handle overlapping rules
        for (replacement, originals) in &reverse_map {
            if remaining.starts_with(replacement.as_str()) {
                // This position could be the result of one of these rules
                // Create alternation: (original1|original2|...|replacement)
                let mut alternatives: Vec<&str> = originals.iter().map(|s| s.as_str()).collect();

                // Add the replacement itself as an alternative (identity case)
                if !alternatives.contains(&replacement.as_str()) {
                    alternatives.push(replacement);
                }

                // Only create alternation if we have multiple alternatives
                if alternatives.len() > 1 {
                    pattern.push('(');
                    for (j, alt) in alternatives.iter().enumerate() {
                        if j > 0 {
                            pattern.push('|');
                        }
                        pattern.push_str(&regex_escape(alt));
                    }
                    pattern.push(')');
                } else {
                    pattern.push_str(&regex_escape(alternatives[0]));
                }

                i += replacement.chars().count();
                matched = true;
                break;
            }
        }

        if !matched {
            // No rule applies, emit literal character (escaped if necessary)
            pattern.push_str(&regex_escape_char(chars[i]));
            i += 1;
        }
    }

    pattern
}

/// A reverse mapping entry: replacement string → list of original patterns
type ReverseMap = Vec<(String, Vec<String>)>;

/// Build a reverse map from replacement strings to original patterns.
///
/// This allows efficient lookup: given a replacement substring, find all
/// original patterns that could have produced it.
fn build_reverse_map(rules: &[RewriteRuleChar]) -> ReverseMap {
    use std::collections::HashMap;

    let mut map: HashMap<String, Vec<String>> = HashMap::new();

    for rule in rules {
        let original = phones_to_string(&rule.pattern);
        let replacement = phones_to_string(&rule.replacement);

        // Skip identity rules and rules with empty replacement
        if original != replacement && !replacement.is_empty() {
            map.entry(replacement)
                .or_default()
                .push(original);
        }
    }

    // Convert to Vec and sort by replacement length (descending) for greedy matching
    let mut entries: Vec<(String, Vec<String>)> = map.into_iter().collect();
    entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    entries
}

/// Convert a sequence of PhoneChar to a string.
fn phones_to_string(phones: &[PhoneChar]) -> String {
    let mut result = String::new();
    for phone in phones {
        match phone {
            PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => result.push(*c),
            PhoneChar::Digraph(c1, c2) => {
                result.push(*c1);
                result.push(*c2);
            }
            PhoneChar::Silent => {}
        }
    }
    result
}

/// Get the byte index for a character position in a UTF-8 string.
fn char_byte_index(s: &str, char_index: usize) -> usize {
    s.char_indices()
        .nth(char_index)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

/// Escape a string for use in a regex pattern.
fn regex_escape(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        escaped.push_str(&regex_escape_char(c));
    }
    escaped
}

/// Escape a single character for use in a regex pattern.
fn regex_escape_char(c: char) -> String {
    match c {
        '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' | '\\' => {
            format!("\\{}", c)
        }
        _ => c.to_string(),
    }
}

/// Expand a string using rules with optional cost tracking.
///
/// This variant keeps track of which rules were applied, allowing for
/// cost-weighted pattern matching.
///
/// # Arguments
///
/// * `input` - The normalized string to expand
/// * `rules` - The phonetic rules
///
/// # Returns
///
/// A tuple of (pattern, max_phonetic_cost) where the cost is the sum of
/// weights of all rules that could have been applied.
pub fn expand_with_costs(input: &str, rules: &[RewriteRuleChar]) -> (String, f64) {
    let reverse_map = build_reverse_map_with_costs(rules);

    let mut pattern = String::new();
    let mut total_cost = 0.0;
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let remaining = &input[char_byte_index(input, i)..];
        let mut matched = false;

        for (replacement, originals_with_costs) in &reverse_map {
            if remaining.starts_with(replacement.as_str()) {
                let mut alternatives: Vec<&str> =
                    originals_with_costs.iter().map(|(s, _)| s.as_str()).collect();

                // Track the maximum cost among alternatives
                let max_cost = originals_with_costs
                    .iter()
                    .map(|(_, cost)| *cost)
                    .fold(0.0_f64, f64::max);

                total_cost += max_cost;

                if !alternatives.contains(&replacement.as_str()) {
                    alternatives.push(replacement);
                }

                if alternatives.len() > 1 {
                    pattern.push('(');
                    for (j, alt) in alternatives.iter().enumerate() {
                        if j > 0 {
                            pattern.push('|');
                        }
                        pattern.push_str(&regex_escape(alt));
                    }
                    pattern.push(')');
                } else {
                    pattern.push_str(&regex_escape(alternatives[0]));
                }

                i += replacement.chars().count();
                matched = true;
                break;
            }
        }

        if !matched {
            pattern.push_str(&regex_escape_char(chars[i]));
            i += 1;
        }
    }

    (pattern, total_cost)
}

/// Reverse map with costs: replacement → [(original, cost), ...]
type ReverseMapWithCosts = Vec<(String, Vec<(String, f64)>)>;

/// Build a reverse map that includes rule weights.
fn build_reverse_map_with_costs(rules: &[RewriteRuleChar]) -> ReverseMapWithCosts {
    use std::collections::HashMap;

    let mut map: HashMap<String, Vec<(String, f64)>> = HashMap::new();

    for rule in rules {
        let original = phones_to_string(&rule.pattern);
        let replacement = phones_to_string(&rule.replacement);

        if original != replacement && !replacement.is_empty() {
            map.entry(replacement)
                .or_default()
                .push((original, rule.weight));
        }
    }

    let mut entries: Vec<(String, Vec<(String, f64)>)> = map.into_iter().collect();
    entries.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::types::{ContextChar, PhoneChar, RewriteRuleChar};

    fn make_rule(id: usize, pattern: &str, replacement: &str, weight: f64) -> RewriteRuleChar {
        RewriteRuleChar {
            rule_id: id,
            rule_name: format!("{} -> {}", pattern, replacement),
            pattern: pattern
                .chars()
                .map(|c| PhoneChar::Consonant(c))
                .collect(),
            replacement: replacement
                .chars()
                .map(|c| PhoneChar::Consonant(c))
                .collect(),
            context: ContextChar::Anywhere,
            weight,
        }
    }

    #[test]
    fn test_expand_single_rule() {
        // Rule: ph -> f
        let rules = vec![make_rule(1, "ph", "f", 0.1)];

        let pattern = expand_phonetic_alternatives_char("fone", &rules);

        // Should match "f" with "(ph|f)"
        assert!(pattern.contains("(ph|f)") || pattern.contains("(f|ph)"));
        assert!(pattern.ends_with("one"));
    }

    #[test]
    fn test_expand_no_rules() {
        let rules: Vec<RewriteRuleChar> = vec![];
        let pattern = expand_phonetic_alternatives_char("hello", &rules);

        // No rules means literal string
        assert_eq!(pattern, "hello");
    }

    #[test]
    fn test_expand_multiple_alternatives() {
        // Multiple rules that produce the same replacement
        // Rule 1: ph -> f
        // Rule 2: gh -> f (hypothetical)
        let rules = vec![make_rule(1, "ph", "f", 0.1), make_rule(2, "gh", "f", 0.2)];

        let pattern = expand_phonetic_alternatives_char("f", &rules);

        // Should have all three alternatives: (ph|gh|f)
        assert!(pattern.contains("ph"));
        assert!(pattern.contains("gh"));
        assert!(pattern.contains('|'));
    }

    #[test]
    fn test_expand_with_special_chars() {
        let rules: Vec<RewriteRuleChar> = vec![];

        // Special regex characters should be escaped
        let pattern = expand_phonetic_alternatives_char("a.b*c?", &rules);

        assert_eq!(pattern, "a\\.b\\*c\\?");
    }

    #[test]
    fn test_expand_longer_replacement_first() {
        // Rule 1: tion -> shun (longer replacement)
        // Rule 2: ti -> sh (shorter replacement)
        let rules = vec![
            make_rule(1, "tion", "shun", 0.1),
            make_rule(2, "ti", "sh", 0.1),
        ];

        let pattern = expand_phonetic_alternatives_char("shun", &rules);

        // Should match the longer "shun" -> "tion", not break it up
        assert!(pattern.contains("(tion|shun)") || pattern.contains("(shun|tion)"));
    }

    #[test]
    fn test_expand_with_costs() {
        let rules = vec![
            make_rule(1, "ph", "f", 0.1),
            make_rule(2, "tion", "shun", 0.2),
        ];

        let (pattern, cost) = expand_with_costs("fashun", &rules);

        // Should have expanded "f" and "shun"
        assert!(pattern.contains("(ph|f)") || pattern.contains("(f|ph)"));
        assert!(pattern.contains("shun"));

        // Cost should be sum of max costs at each expansion point
        assert!(cost > 0.0);
    }

    #[test]
    fn test_expand_identity_rule_excluded() {
        // Rule: f -> f (identity, should be excluded from reverse map)
        let rules = vec![make_rule(1, "f", "f", 0.1)];

        let pattern = expand_phonetic_alternatives_char("fone", &rules);

        // No alternation needed for identity rule
        assert_eq!(pattern, "fone");
    }

    #[test]
    fn test_phones_to_string() {
        let phones = vec![
            PhoneChar::Consonant('p'),
            PhoneChar::Consonant('h'),
            PhoneChar::Vowel('o'),
            PhoneChar::Consonant('n'),
            PhoneChar::Vowel('e'),
        ];

        assert_eq!(phones_to_string(&phones), "phone");
    }

    #[test]
    fn test_phones_to_string_with_digraph() {
        let phones = vec![
            PhoneChar::Digraph('s', 'h'),
            PhoneChar::Vowel('i'),
            PhoneChar::Consonant('p'),
        ];

        assert_eq!(phones_to_string(&phones), "ship");
    }

    #[test]
    fn test_phones_to_string_with_silent() {
        let phones = vec![
            PhoneChar::Consonant('k'),
            PhoneChar::Silent,
            PhoneChar::Consonant('n'),
            PhoneChar::Vowel('o'),
            PhoneChar::Consonant('w'),
        ];

        assert_eq!(phones_to_string(&phones), "know");
    }

    #[test]
    fn test_regex_escape() {
        assert_eq!(regex_escape("."), "\\.");
        assert_eq!(regex_escape("*"), "\\*");
        assert_eq!(regex_escape("+"), "\\+");
        assert_eq!(regex_escape("?"), "\\?");
        assert_eq!(regex_escape("("), "\\(");
        assert_eq!(regex_escape(")"), "\\)");
        assert_eq!(regex_escape("["), "\\[");
        assert_eq!(regex_escape("]"), "\\]");
        assert_eq!(regex_escape("{"), "\\{");
        assert_eq!(regex_escape("}"), "\\}");
        assert_eq!(regex_escape("|"), "\\|");
        assert_eq!(regex_escape("^"), "\\^");
        assert_eq!(regex_escape("$"), "\\$");
        assert_eq!(regex_escape("\\"), "\\\\");
        assert_eq!(regex_escape("abc"), "abc");
    }

    #[test]
    fn test_char_byte_index() {
        // ASCII string
        assert_eq!(char_byte_index("hello", 0), 0);
        assert_eq!(char_byte_index("hello", 2), 2);
        assert_eq!(char_byte_index("hello", 5), 5);

        // UTF-8 string with multi-byte characters
        let s = "héllo";
        assert_eq!(char_byte_index(s, 0), 0); // 'h' at byte 0
        assert_eq!(char_byte_index(s, 1), 1); // 'é' at byte 1 (2 bytes)
        assert_eq!(char_byte_index(s, 2), 3); // 'l' at byte 3
    }
}
