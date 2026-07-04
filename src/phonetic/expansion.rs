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
use std::cmp::Reverse;
use std::rc::Rc;

/// Expand a normalized string into a regex pattern matching phonetic variants.
///
/// Given rules like `ph → f`, the string "fone" becomes "(ph|f)one"
/// because anywhere we see "f" in the output, the input could have been "ph".
///
/// # Algorithm
///
/// This uses dynamic programming to find ALL possible segmentations of the input,
/// not just greedy longest-first matching. This is critical for cases like:
/// - "naɪt" which could come from "n"+"igh"+"t" (night) OR "n"+"ite" (nite)
///
/// The algorithm builds all valid parse trees and combines them into a single
/// regex pattern with nested alternations.
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

    // Use DP to find all possible expansions
    let (chars, byte_indices) = chars_and_byte_indices(input);
    let n = chars.len();

    if n == 0 {
        return String::new();
    }

    // dp[i] = ids of all possible pattern-prefix nodes that expand chars[0..i].
    // Prefix nodes avoid repeatedly cloning whole partial strings at each
    // reachable position; complete strings are rendered once at the end.
    let Some(dp_len) = one_extra_capacity(n) else {
        return regex_escape(input);
    };
    let mut nodes = vec![ExpansionNode::root()];
    let mut dp: Vec<Vec<usize>> = vec![Vec::new(); dp_len];
    dp[0].push(0); // Empty prefix has one expansion: the root node.

    for i in 0..n {
        if dp[i].is_empty() {
            continue; // No way to reach this position
        }

        let remaining = &input[byte_indices[i]..];
        let mut has_single = false;

        // Check all replacements (not just the longest)
        for entry in &reverse_map {
            if remaining.starts_with(entry.replacement.as_str()) {
                has_single |= entry.replacement_char_len == 1;
                let next_pos = i + entry.replacement_char_len;
                if next_pos <= n {
                    push_segment_nodes(&mut dp, &mut nodes, i, next_pos, &entry.segment);
                }
            }
        }

        // Always allow single character match (identity)
        if !has_single {
            let single_char = regex_escape_char(chars[i]);
            push_segment_nodes(&mut dp, &mut nodes, i, i + 1, &single_char);
        }
    }

    // Collect all complete expansions and deduplicate
    let mut final_patterns: Vec<String> = dp[n]
        .iter()
        .map(|&node_id| materialize_expansion(node_id, &nodes))
        .collect();
    final_patterns.sort();
    final_patterns.dedup();

    if final_patterns.is_empty() {
        // Fallback: just escape the input
        return regex_escape(input);
    }

    if final_patterns.len() == 1 {
        return final_patterns
            .into_iter()
            .next()
            .expect("len==1 checked above");
    }

    // Multiple complete expansions: combine with alternation
    // But first, try to simplify by finding common prefixes/suffixes
    format!("({})", final_patterns.join("|"))
}

struct ReverseMapEntry {
    replacement: String,
    replacement_char_len: usize,
    segment: String,
}

type ReverseMap = Vec<ReverseMapEntry>;

struct ExpansionNode {
    parent: Option<usize>,
    segment: Rc<str>,
    len: usize,
}

impl ExpansionNode {
    fn root() -> Self {
        Self {
            parent: None,
            segment: Rc::from(""),
            len: 0,
        }
    }
}

#[inline]
fn one_extra_capacity(len: usize) -> Option<usize> {
    len.checked_add(1)
}

#[inline]
fn escaped_regex_capacity(byte_len: usize) -> Option<usize> {
    byte_len.checked_mul(2)
}

#[inline]
fn expansion_node_len(parent_len: usize, segment_len: usize) -> Option<usize> {
    parent_len.checked_add(segment_len)
}

fn alternation_segment_capacity<I>(
    alternative_count: usize,
    alternative_lengths: I,
) -> Option<usize>
where
    I: IntoIterator<Item = usize>,
{
    let payload_len = alternative_lengths
        .into_iter()
        .try_fold(0usize, |total, len| total.checked_add(len))?;
    let separator_count = alternative_count.saturating_sub(1);

    payload_len.checked_add(separator_count)?.checked_add(2)
}

/// Build a reverse map from replacement strings to original patterns.
///
/// This allows efficient lookup: given a replacement substring, find all
/// original patterns that could have produced it.
fn build_reverse_map(rules: &[RewriteRuleChar]) -> ReverseMap {
    use std::collections::HashMap;

    let mut map: HashMap<String, Vec<String>> = HashMap::with_capacity(rules.len());

    for rule in rules {
        let original = phones_to_string(&rule.pattern);
        let replacement = phones_to_string(&rule.replacement);

        // Skip identity rules and rules with empty replacement
        if original != replacement && !replacement.is_empty() {
            map.entry(replacement).or_default().push(original);
        }
    }

    let mut entries: Vec<ReverseMapEntry> = map
        .into_iter()
        .map(|(replacement, originals)| {
            let replacement_char_len = replacement.chars().count();
            let mut alternatives: Vec<String> =
                Vec::with_capacity(one_extra_capacity(originals.len()).unwrap_or(0));
            alternatives.extend(originals.iter().map(|s| regex_escape(s)));

            // Add the replacement itself as an alternative
            let escaped_replacement = regex_escape(&replacement);
            if !alternatives.contains(&escaped_replacement) {
                alternatives.push(escaped_replacement);
            }

            ReverseMapEntry {
                replacement,
                replacement_char_len,
                segment: segment_from_escaped_alternatives(alternatives),
            }
        })
        .collect();
    entries.sort_by_key(|entry| {
        (
            Reverse(entry.replacement_char_len),
            Reverse(entry.replacement.len()),
        )
    });

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
            PhoneChar::Trigraph(c1, c2, c3) => {
                result.push(*c1);
                result.push(*c2);
                result.push(*c3);
            }
            PhoneChar::Tetragraph(c1, c2, c3, c4) => {
                result.push(*c1);
                result.push(*c2);
                result.push(*c3);
                result.push(*c4);
            }
            PhoneChar::Pentagraph(c1, c2, c3, c4, c5) => {
                result.push(*c1);
                result.push(*c2);
                result.push(*c3);
                result.push(*c4);
                result.push(*c5);
            }
            PhoneChar::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                result.push(*c1);
                result.push(*c2);
                result.push(*c3);
                result.push(*c4);
                result.push(*c5);
                result.push(*c6);
            }
            PhoneChar::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                result.push(*c1);
                result.push(*c2);
                result.push(*c3);
                result.push(*c4);
                result.push(*c5);
                result.push(*c6);
                result.push(*c7);
            }
            PhoneChar::Sequence(s) => {
                for c in s {
                    result.push(*c);
                }
            }
            PhoneChar::Silent => {}
        }
    }
    result
}

fn chars_and_byte_indices(s: &str) -> (Vec<char>, Vec<usize>) {
    let char_count = s.chars().count();
    let mut chars = Vec::with_capacity(char_count);
    let mut byte_indices = Vec::with_capacity(one_extra_capacity(char_count).unwrap_or(0));

    for (byte_index, ch) in s.char_indices() {
        byte_indices.push(byte_index);
        chars.push(ch);
    }
    byte_indices.push(s.len());

    (chars, byte_indices)
}

#[cfg(test)]
fn char_byte_index(s: &str, char_index: usize) -> usize {
    let (_, byte_indices) = chars_and_byte_indices(s);
    byte_indices.get(char_index).copied().unwrap_or(s.len())
}

fn segment_from_escaped_alternatives(alternatives: Vec<String>) -> String {
    if alternatives.len() == 1 {
        return alternatives
            .into_iter()
            .next()
            .expect("len==1 checked above");
    }

    let capacity =
        alternation_segment_capacity(alternatives.len(), alternatives.iter().map(String::len))
            .unwrap_or(0);
    let mut segment = String::with_capacity(capacity);
    segment.push('(');
    for (index, alternative) in alternatives.iter().enumerate() {
        if index > 0 {
            segment.push('|');
        }
        segment.push_str(alternative);
    }
    segment.push(')');
    segment
}

fn push_segment_nodes(
    dp: &mut [Vec<usize>],
    nodes: &mut Vec<ExpansionNode>,
    current_pos: usize,
    next_pos: usize,
    segment: &str,
) {
    debug_assert!(current_pos < next_pos);

    let (before_target, target_and_after) = dp.split_at_mut(next_pos);
    let prefixes = &before_target[current_pos];
    let target = &mut target_and_after[0];
    target.reserve(prefixes.len());
    let shared_segment: Rc<str> = Rc::from(segment);

    for &parent in prefixes {
        let node_id = nodes.len();
        let parent_len = nodes[parent].len;
        let len = expansion_node_len(parent_len, segment.len()).unwrap_or(0);
        nodes.push(ExpansionNode {
            parent: Some(parent),
            segment: Rc::clone(&shared_segment),
            len,
        });
        target.push(node_id);
    }
}

fn materialize_expansion(node_id: usize, nodes: &[ExpansionNode]) -> String {
    let mut parts = Vec::new();
    let mut cursor = node_id;
    while let Some(parent) = nodes[cursor].parent {
        parts.push(nodes[cursor].segment.as_ref());
        cursor = parent;
    }

    let mut expansion = String::with_capacity(nodes[node_id].len);
    for segment in parts.into_iter().rev() {
        expansion.push_str(segment);
    }
    expansion
}

/// Escape a string for use in a regex pattern.
fn regex_escape(s: &str) -> String {
    let mut escaped = String::with_capacity(escaped_regex_capacity(s.len()).unwrap_or(0));
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

    let mut pattern = String::with_capacity(input.len());
    let mut total_cost = 0.0;
    let (chars, byte_indices) = chars_and_byte_indices(input);
    let mut i = 0;

    while i < chars.len() {
        let remaining = &input[byte_indices[i]..];
        let mut matched = false;

        for entry in &reverse_map {
            if remaining.starts_with(entry.replacement.as_str()) {
                total_cost += entry.max_cost;
                pattern.push_str(&entry.segment);
                i += entry.replacement_char_len;
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

struct ReverseMapEntryWithCost {
    replacement: String,
    replacement_char_len: usize,
    segment: String,
    max_cost: f64,
}

type ReverseMapWithCosts = Vec<ReverseMapEntryWithCost>;

/// Build a reverse map that includes rule weights.
fn build_reverse_map_with_costs(rules: &[RewriteRuleChar]) -> ReverseMapWithCosts {
    use std::collections::HashMap;

    let mut map: HashMap<String, Vec<(String, f64)>> = HashMap::with_capacity(rules.len());

    for rule in rules {
        let original = phones_to_string(&rule.pattern);
        let replacement = phones_to_string(&rule.replacement);

        if original != replacement && !replacement.is_empty() {
            map.entry(replacement)
                .or_default()
                .push((original, rule.weight));
        }
    }

    let mut entries: Vec<ReverseMapEntryWithCost> = map
        .into_iter()
        .map(|(replacement, originals_with_costs)| {
            let replacement_char_len = replacement.chars().count();
            let max_cost = originals_with_costs
                .iter()
                .map(|(_, cost)| *cost)
                .fold(0.0_f64, f64::max);
            let mut alternatives: Vec<String> =
                Vec::with_capacity(one_extra_capacity(originals_with_costs.len()).unwrap_or(0));
            alternatives.extend(originals_with_costs.iter().map(|(s, _)| regex_escape(s)));

            let escaped_replacement = regex_escape(&replacement);
            if !alternatives.contains(&escaped_replacement) {
                alternatives.push(escaped_replacement);
            }

            ReverseMapEntryWithCost {
                replacement,
                replacement_char_len,
                segment: segment_from_escaped_alternatives(alternatives),
                max_cost,
            }
        })
        .collect();
    entries.sort_by_key(|entry| {
        (
            Reverse(entry.replacement_char_len),
            Reverse(entry.replacement.len()),
        )
    });

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
            pattern: pattern.chars().map(PhoneChar::Consonant).collect(),
            replacement: replacement.chars().map(PhoneChar::Consonant).collect(),
            context: ContextChar::Anywhere,
            weight,
            syllable_condition: None,
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
    fn test_expand_unicode_replacement_uses_character_offsets() {
        let rules = vec![make_rule(1, "eh", "é", 0.3)];

        let pattern = expand_phonetic_alternatives_char("éx", &rules);

        assert!(pattern.contains("(eh|é)") || pattern.contains("(é|eh)"));
        assert!(pattern.ends_with('x'));
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
    fn test_expand_with_costs_unicode_replacement_uses_character_offsets() {
        let rules = vec![make_rule(1, "eh", "é", 0.3)];

        let (pattern, cost) = expand_with_costs("éx", &rules);

        assert!(pattern.contains("(eh|é)") || pattern.contains("(é|eh)"));
        assert!(pattern.ends_with('x'));
        assert_eq!(cost, 0.3);
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
    fn test_capacity_helpers_reject_overflow() {
        assert_eq!(one_extra_capacity(0), Some(1));
        assert_eq!(one_extra_capacity(usize::MAX), None);

        assert_eq!(escaped_regex_capacity(4), Some(8));
        assert_eq!(escaped_regex_capacity(usize::MAX), None);

        assert_eq!(expansion_node_len(5, 7), Some(12));
        assert_eq!(expansion_node_len(usize::MAX, 1), None);

        assert_eq!(alternation_segment_capacity(3, [2, 2, 1]), Some(9));
        assert_eq!(alternation_segment_capacity(2, [usize::MAX, 1]), None);
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
