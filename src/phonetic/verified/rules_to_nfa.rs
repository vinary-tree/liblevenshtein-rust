//! Convert verified phonetic rewrite rules to NFA representation.
//!
//! This module provides functions to convert the Coq-verified phonetic rules
//! (from `src/phonetic/rules.rs`) to NFA-based pattern matching automata.
//!
//! # Conversion Strategy
//!
//! A `RewriteRule` defines a transformation:
//! ```text
//! pattern → replacement / context
//! ```
//!
//! We convert this to an NFA that:
//! 1. Recognizes the pattern (as a literal sequence)
//! 2. Optionally includes context constraints (lookahead/lookbehind)
//!
//! The resulting NFA can be:
//! - Used directly for pattern matching
//! - Combined with other rules using union (alternation)
//! - Composed with Levenshtein automaton for fuzzy matching
//!
//! # Context Handling
//!
//! Contexts are converted as follows:
//! - `Anywhere`: No additional constraints
//! - `Initial`: Word-start marker (handled at match time)
//! - `Final`: Word-end marker (handled at match time)
//! - `BeforeVowel([v1, v2, ...])`: Lookahead for vowel class
//! - `AfterConsonant([c1, c2, ...])`: Lookbehind for consonant class
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::verified::{rule_to_nfa_char, zompist_nfa_char};
//! use liblevenshtein::phonetic::rules::zompist_rules_char;
//!
//! // Convert all Zompist rules to a single NFA
//! let nfa = zompist_nfa_char();
//!
//! // Test pattern matching
//! assert!(nfa.accepts("ph"));  // matches "ph → f" pattern
//! assert!(nfa.accepts("gh"));  // matches "gh → ∅" pattern
//! ```

use crate::phonetic::nfa::nfa::{NFAChar, NFA};
use crate::phonetic::nfa::thompson::{ThompsonBuilder, ThompsonBuilderChar};
use crate::phonetic::rules::{zompist_rules, zompist_rules_char};
use crate::phonetic::types::{
    Context, ContextChar, Phone, PhoneChar, RewriteRule, RewriteRuleChar,
};

// ============================================================================
// Character-level conversions
// ============================================================================

/// Convert a PhoneChar to a character for pattern matching.
///
/// This extracts the primary character from a phone unit:
/// - Vowels and consonants return their character
/// - Digraphs return their first character (the pattern NFA handles the full digraph)
/// - Trigraphs return their first character
/// - Tetragraphs return their first character
/// - Sequences return their first character
/// - Silent returns None (no character to match)
#[allow(dead_code)]
fn phone_to_char(phone: &PhoneChar) -> Option<char> {
    match phone {
        PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => Some(*c),
        PhoneChar::Digraph(c1, _) => Some(*c1),
        PhoneChar::Trigraph(c1, _, _) => Some(*c1),
        PhoneChar::Tetragraph(c1, _, _, _) => Some(*c1),
        PhoneChar::Pentagraph(c1, _, _, _, _) => Some(*c1),
        PhoneChar::Hexagraph(c1, _, _, _, _, _) => Some(*c1),
        PhoneChar::Heptagraph(c1, _, _, _, _, _, _) => Some(*c1),
        PhoneChar::Sequence(s) => s.first().copied(),
        PhoneChar::Silent => None,
    }
}

/// Convert a pattern of PhoneChar to a string for NFA construction.
fn pattern_to_string_char(pattern: &[PhoneChar]) -> String {
    let mut result = String::new();
    for phone in pattern {
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

/// Convert a single character-level rewrite rule to an NFA.
///
/// The NFA recognizes the pattern portion of the rule.
/// Context information is preserved in a separate structure for matching.
///
/// # Arguments
///
/// * `rule` - The rewrite rule to convert
///
/// # Returns
///
/// An NFA that accepts exactly the pattern strings defined by the rule.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::phonetic::verified::rule_to_nfa_char;
/// use liblevenshtein::phonetic::types::{RewriteRuleChar, PhoneChar, ContextChar};
///
/// // Rule: ph → f
/// let rule = RewriteRuleChar {
///     rule_id: 1,
///     rule_name: "ph → f".to_string(),
///     pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
///     replacement: vec![PhoneChar::Consonant('f')],
///     context: ContextChar::Anywhere,
///     weight: 0.0,
/// };
///
/// let nfa = rule_to_nfa_char(&rule);
/// assert!(nfa.accepts("ph"));
/// assert!(!nfa.accepts("f"));
/// ```
pub fn rule_to_nfa_char(rule: &RewriteRuleChar) -> NFAChar {
    let builder = ThompsonBuilderChar::new();
    let pattern_str = pattern_to_string_char(&rule.pattern);

    if pattern_str.is_empty() {
        return builder.epsilon();
    }

    // Build base pattern NFA
    let pattern_nfa = builder.literal(&pattern_str);

    // Context handling is done at match time, not in the NFA itself
    // The NFA just recognizes the pattern; context is checked separately
    pattern_nfa
}

/// Convert multiple character-level rewrite rules to a single NFA.
///
/// Creates an NFA that matches any of the rule patterns using alternation.
/// This is equivalent to `pattern1 | pattern2 | ... | patternN`.
///
/// # Arguments
///
/// * `rules` - Slice of rewrite rules to convert
///
/// # Returns
///
/// An NFA that accepts strings matching any of the rule patterns.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::phonetic::verified::rules_to_nfa_char;
/// use liblevenshtein::phonetic::rules::orthography_rules_char;
///
/// let rules = orthography_rules_char();
/// let nfa = rules_to_nfa_char(&rules);
///
/// // Matches any of the orthography rule patterns
/// assert!(nfa.accepts("ph"));  // Rule 3
/// assert!(nfa.accepts("gh"));  // Rule 34
/// assert!(nfa.accepts("ch"));  // Rule 1
/// ```
pub fn rules_to_nfa_char(rules: &[RewriteRuleChar]) -> NFAChar {
    if rules.is_empty() {
        let builder = ThompsonBuilderChar::new();
        return builder.epsilon();
    }

    let builder = ThompsonBuilderChar::new();
    let nfas: Vec<NFAChar> = rules.iter().map(rule_to_nfa_char).collect();

    builder.union_all(nfas)
}

/// Get a pre-compiled NFA for all Zompist rules (character-level).
///
/// This is a convenience function that returns an NFA recognizing
/// all 13 Zompist rule patterns.
///
/// # Returns
///
/// An NFA that accepts any Zompist rule pattern.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::phonetic::verified::zompist_nfa_char;
///
/// let nfa = zompist_nfa_char();
///
/// // Orthography patterns
/// assert!(nfa.accepts("ch"));  // Rule 1: ch → ç
/// assert!(nfa.accepts("sh"));  // Rule 2: sh → $
/// assert!(nfa.accepts("ph"));  // Rule 3: ph → f
/// assert!(nfa.accepts("c"));   // Rule 20/21: c → s/k
/// assert!(nfa.accepts("g"));   // Rule 22: g → j
/// assert!(nfa.accepts("e"));   // Rule 33: silent e
/// assert!(nfa.accepts("gh"));  // Rule 34: gh → ∅
///
/// // Phonetic patterns
/// assert!(nfa.accepts("th"));  // Rule 100: th → t
/// assert!(nfa.accepts("qu"));  // Rule 101: qu → kw
/// assert!(nfa.accepts("kw"));  // Rule 102: kw → qu
///
/// // Test patterns
/// assert!(nfa.accepts("x"));   // Rule 200: x → yy
/// assert!(nfa.accepts("y"));   // Rule 201: y → z
/// ```
pub fn zompist_nfa_char() -> NFAChar {
    rules_to_nfa_char(&zompist_rules_char())
}

// ============================================================================
// Byte-level conversions
// ============================================================================

/// Convert a Phone to a byte for pattern matching.
#[allow(dead_code)]
fn phone_to_byte(phone: &Phone) -> Option<u8> {
    match phone {
        Phone::Vowel(b) | Phone::Consonant(b) => Some(*b),
        Phone::Digraph(b1, _) => Some(*b1),
        Phone::Trigraph(b1, _, _) => Some(*b1),
        Phone::Tetragraph(b1, _, _, _) => Some(*b1),
        Phone::Pentagraph(b1, _, _, _, _) => Some(*b1),
        Phone::Hexagraph(b1, _, _, _, _, _) => Some(*b1),
        Phone::Heptagraph(b1, _, _, _, _, _, _) => Some(*b1),
        Phone::Sequence(s) => s.first().copied(),
        Phone::Silent => None,
    }
}

/// Convert a pattern of Phone to a byte sequence for NFA construction.
fn pattern_to_bytes(pattern: &[Phone]) -> Vec<u8> {
    let mut result = Vec::new();
    for phone in pattern {
        match phone {
            Phone::Vowel(b) | Phone::Consonant(b) => result.push(*b),
            Phone::Digraph(b1, b2) => {
                result.push(*b1);
                result.push(*b2);
            }
            Phone::Trigraph(b1, b2, b3) => {
                result.push(*b1);
                result.push(*b2);
                result.push(*b3);
            }
            Phone::Tetragraph(b1, b2, b3, b4) => {
                result.push(*b1);
                result.push(*b2);
                result.push(*b3);
                result.push(*b4);
            }
            Phone::Pentagraph(b1, b2, b3, b4, b5) => {
                result.push(*b1);
                result.push(*b2);
                result.push(*b3);
                result.push(*b4);
                result.push(*b5);
            }
            Phone::Hexagraph(b1, b2, b3, b4, b5, b6) => {
                result.push(*b1);
                result.push(*b2);
                result.push(*b3);
                result.push(*b4);
                result.push(*b5);
                result.push(*b6);
            }
            Phone::Heptagraph(b1, b2, b3, b4, b5, b6, b7) => {
                result.push(*b1);
                result.push(*b2);
                result.push(*b3);
                result.push(*b4);
                result.push(*b5);
                result.push(*b6);
                result.push(*b7);
            }
            Phone::Sequence(s) => {
                result.extend_from_slice(s);
            }
            Phone::Silent => {}
        }
    }
    result
}

/// Convert a single byte-level rewrite rule to an NFA.
///
/// The NFA recognizes the pattern portion of the rule.
///
/// # Arguments
///
/// * `rule` - The rewrite rule to convert
///
/// # Returns
///
/// An NFA that accepts exactly the pattern bytes defined by the rule.
pub fn rule_to_nfa(rule: &RewriteRule) -> NFA {
    let builder = ThompsonBuilder::new();
    let pattern_bytes = pattern_to_bytes(&rule.pattern);

    if pattern_bytes.is_empty() {
        return builder.epsilon();
    }

    // Build base pattern NFA
    builder.literal(&pattern_bytes)
}

/// Convert multiple byte-level rewrite rules to a single NFA.
///
/// Creates an NFA that matches any of the rule patterns using alternation.
///
/// # Arguments
///
/// * `rules` - Slice of rewrite rules to convert
///
/// # Returns
///
/// An NFA that accepts bytes matching any of the rule patterns.
pub fn rules_to_nfa(rules: &[RewriteRule]) -> NFA {
    if rules.is_empty() {
        let builder = ThompsonBuilder::new();
        return builder.epsilon();
    }

    let builder = ThompsonBuilder::new();
    let nfas: Vec<NFA> = rules.iter().map(rule_to_nfa).collect();

    builder.union_all(nfas)
}

/// Get a pre-compiled NFA for all Zompist rules (byte-level).
///
/// This is a convenience function that returns an NFA recognizing
/// all 13 Zompist rule patterns at the byte level.
///
/// # Returns
///
/// An NFA that accepts any Zompist rule pattern as bytes.
pub fn zompist_nfa() -> NFA {
    rules_to_nfa(&zompist_rules())
}

// ============================================================================
// Contextual NFA construction
// ============================================================================

/// Information about rule context for matching.
///
/// While the basic NFA recognizes patterns, context information
/// determines where those patterns can be applied.
#[derive(Debug, Clone)]
pub struct RuleContextInfoChar {
    /// The compiled NFA for the pattern
    pub pattern_nfa: NFAChar,
    /// Context constraint
    pub context: ContextChar,
    /// Rule weight (for prioritization)
    pub weight: f64,
    /// Original rule name (for debugging)
    pub rule_name: String,
}

/// Information about rule context for matching (byte-level).
#[derive(Debug, Clone)]
pub struct RuleContextInfo {
    /// The compiled NFA for the pattern
    pub pattern_nfa: NFA,
    /// Context constraint
    pub context: Context,
    /// Rule weight (for prioritization)
    pub weight: f64,
    /// Original rule name (for debugging)
    pub rule_name: String,
}

/// Convert a rule to NFA with context information preserved.
pub fn rule_to_nfa_with_context_char(rule: &RewriteRuleChar) -> RuleContextInfoChar {
    RuleContextInfoChar {
        pattern_nfa: rule_to_nfa_char(rule),
        context: rule.context.clone(),
        weight: rule.weight,
        rule_name: rule.rule_name.clone(),
    }
}

/// Convert a rule to NFA with context information preserved (byte-level).
pub fn rule_to_nfa_with_context(rule: &RewriteRule) -> RuleContextInfo {
    RuleContextInfo {
        pattern_nfa: rule_to_nfa(rule),
        context: rule.context.clone(),
        weight: rule.weight,
        rule_name: rule.rule_name.clone(),
    }
}

/// Convert multiple rules to NFAs with context information.
pub fn rules_to_nfa_with_context_char(rules: &[RewriteRuleChar]) -> Vec<RuleContextInfoChar> {
    rules.iter().map(rule_to_nfa_with_context_char).collect()
}

/// Convert multiple rules to NFAs with context information (byte-level).
pub fn rules_to_nfa_with_context(rules: &[RewriteRule]) -> Vec<RuleContextInfo> {
    rules.iter().map(rule_to_nfa_with_context).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::rules::{
        orthography_rules, orthography_rules_char, phonetic_rules, phonetic_rules_char,
        test_rules, test_rules_char,
    };

    // ============================================================================
    // Character-level tests
    // ============================================================================

    #[test]
    fn test_pattern_to_string_simple() {
        let pattern = vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')];
        assert_eq!(pattern_to_string_char(&pattern), "ph");
    }

    #[test]
    fn test_pattern_to_string_digraph() {
        let pattern = vec![PhoneChar::Digraph('c', 'h')];
        assert_eq!(pattern_to_string_char(&pattern), "ch");
    }

    #[test]
    fn test_pattern_to_string_mixed() {
        let pattern = vec![
            PhoneChar::Consonant('a'),
            PhoneChar::Digraph('c', 'h'),
            PhoneChar::Vowel('e'),
        ];
        assert_eq!(pattern_to_string_char(&pattern), "ache");
    }

    #[test]
    fn test_pattern_to_string_silent() {
        let pattern = vec![PhoneChar::Consonant('a'), PhoneChar::Silent];
        assert_eq!(pattern_to_string_char(&pattern), "a");
    }

    #[test]
    fn test_rule_to_nfa_char_ph_to_f() {
        let rule = RewriteRuleChar {
            rule_id: 3,
            rule_name: "ph → f".to_string(),
            pattern: vec![PhoneChar::Consonant('p'), PhoneChar::Consonant('h')],
            replacement: vec![PhoneChar::Consonant('f')],
            context: ContextChar::Anywhere,
            weight: 0.0,
            syllable_condition: None,
        };

        let nfa = rule_to_nfa_char(&rule);
        assert!(nfa.accepts("ph"));
        assert!(!nfa.accepts("p"));
        assert!(!nfa.accepts("h"));
        assert!(!nfa.accepts("f"));
        assert!(!nfa.accepts(""));
    }

    #[test]
    fn test_rule_to_nfa_char_single_char() {
        let rule = RewriteRuleChar {
            rule_id: 20,
            rule_name: "c → s".to_string(),
            pattern: vec![PhoneChar::Consonant('c')],
            replacement: vec![PhoneChar::Consonant('s')],
            context: ContextChar::BeforeVowel(vec!['e', 'i']),
            weight: 0.0,
            syllable_condition: None,
        };

        let nfa = rule_to_nfa_char(&rule);
        assert!(nfa.accepts("c"));
        assert!(!nfa.accepts("s"));
        assert!(!nfa.accepts(""));
    }

    #[test]
    fn test_rules_to_nfa_char_orthography() {
        let rules = orthography_rules_char();
        let nfa = rules_to_nfa_char(&rules);

        // Test each orthography rule pattern
        assert!(nfa.accepts("ch")); // Rule 1
        assert!(nfa.accepts("sh")); // Rule 2
        assert!(nfa.accepts("ph")); // Rule 3
        assert!(nfa.accepts("c")); // Rule 20, 21
        assert!(nfa.accepts("g")); // Rule 22
        assert!(nfa.accepts("e")); // Rule 33
        assert!(nfa.accepts("gh")); // Rule 34

        // Non-matching patterns
        assert!(!nfa.accepts("xyz"));
        assert!(!nfa.accepts("abc"));
    }

    #[test]
    fn test_rules_to_nfa_char_phonetic() {
        let rules = phonetic_rules_char();
        let nfa = rules_to_nfa_char(&rules);

        assert!(nfa.accepts("th")); // Rule 100
        assert!(nfa.accepts("qu")); // Rule 101
        assert!(nfa.accepts("kw")); // Rule 102

        assert!(!nfa.accepts("xyz"));
    }

    #[test]
    fn test_rules_to_nfa_char_test_rules() {
        let rules = test_rules_char();
        let nfa = rules_to_nfa_char(&rules);

        assert!(nfa.accepts("x")); // Rule 200
        assert!(nfa.accepts("y")); // Rule 201

        assert!(!nfa.accepts("z"));
        assert!(!nfa.accepts("a"));
    }

    #[test]
    fn test_zompist_nfa_char() {
        let nfa = zompist_nfa_char();

        // All patterns should be recognized
        assert!(nfa.accepts("ch"));
        assert!(nfa.accepts("sh"));
        assert!(nfa.accepts("ph"));
        assert!(nfa.accepts("c"));
        assert!(nfa.accepts("g"));
        assert!(nfa.accepts("e"));
        assert!(nfa.accepts("gh"));
        assert!(nfa.accepts("th"));
        assert!(nfa.accepts("qu"));
        assert!(nfa.accepts("kw"));
        assert!(nfa.accepts("x"));
        assert!(nfa.accepts("y"));

        // Non-Zompist patterns
        assert!(!nfa.accepts("xyz"));
        assert!(!nfa.accepts("hello"));
    }

    #[test]
    fn test_rules_to_nfa_char_empty() {
        let rules: Vec<RewriteRuleChar> = vec![];
        let nfa = rules_to_nfa_char(&rules);

        // Empty rules should give epsilon NFA
        assert!(nfa.accepts(""));
    }

    #[test]
    fn test_rule_context_info_char() {
        let rule = RewriteRuleChar {
            rule_id: 20,
            rule_name: "c → s / _[ie]".to_string(),
            pattern: vec![PhoneChar::Consonant('c')],
            replacement: vec![PhoneChar::Consonant('s')],
            context: ContextChar::BeforeVowel(vec!['e', 'i']),
            weight: 0.0,
            syllable_condition: None,
        };

        let info = rule_to_nfa_with_context_char(&rule);

        assert!(info.pattern_nfa.accepts("c"));
        assert_eq!(info.weight, 0.0);
        assert_eq!(info.rule_name, "c → s / _[ie]");
        assert!(matches!(info.context, ContextChar::BeforeVowel(_)));
    }

    // ============================================================================
    // Byte-level tests
    // ============================================================================

    #[test]
    fn test_pattern_to_bytes_simple() {
        let pattern = vec![Phone::Consonant(b'p'), Phone::Consonant(b'h')];
        assert_eq!(pattern_to_bytes(&pattern), vec![b'p', b'h']);
    }

    #[test]
    fn test_pattern_to_bytes_digraph() {
        let pattern = vec![Phone::Digraph(b'c', b'h')];
        assert_eq!(pattern_to_bytes(&pattern), vec![b'c', b'h']);
    }

    #[test]
    fn test_rule_to_nfa_ph_to_f() {
        let rule = RewriteRule {
            rule_id: 3,
            rule_name: "ph → f".to_string(),
            pattern: vec![Phone::Consonant(b'p'), Phone::Consonant(b'h')],
            replacement: vec![Phone::Consonant(b'f')],
            context: Context::Anywhere,
            weight: 0.0,
            syllable_condition: None,
        };

        let nfa = rule_to_nfa(&rule);
        assert!(nfa.accepts_str("ph"));
        assert!(!nfa.accepts_str("p"));
        assert!(!nfa.accepts_str("f"));
    }

    #[test]
    fn test_rules_to_nfa_orthography() {
        let rules = orthography_rules();
        let nfa = rules_to_nfa(&rules);

        assert!(nfa.accepts_str("ch"));
        assert!(nfa.accepts_str("sh"));
        assert!(nfa.accepts_str("ph"));
        assert!(nfa.accepts_str("c"));
        assert!(nfa.accepts_str("g"));
        assert!(nfa.accepts_str("e"));
        assert!(nfa.accepts_str("gh"));
    }

    #[test]
    fn test_rules_to_nfa_phonetic() {
        let rules = phonetic_rules();
        let nfa = rules_to_nfa(&rules);

        assert!(nfa.accepts_str("th"));
        assert!(nfa.accepts_str("qu"));
        assert!(nfa.accepts_str("kw"));
    }

    #[test]
    fn test_zompist_nfa() {
        let nfa = zompist_nfa();

        // All patterns
        assert!(nfa.accepts_str("ch"));
        assert!(nfa.accepts_str("sh"));
        assert!(nfa.accepts_str("ph"));
        assert!(nfa.accepts_str("c"));
        assert!(nfa.accepts_str("g"));
        assert!(nfa.accepts_str("e"));
        assert!(nfa.accepts_str("gh"));
        assert!(nfa.accepts_str("th"));
        assert!(nfa.accepts_str("qu"));
        assert!(nfa.accepts_str("kw"));
        assert!(nfa.accepts_str("x"));
        assert!(nfa.accepts_str("y"));
    }

    #[test]
    fn test_rules_to_nfa_empty() {
        let rules: Vec<RewriteRule> = vec![];
        let nfa = rules_to_nfa(&rules);

        assert!(nfa.accepts_str(""));
    }

    // ============================================================================
    // Rule count verification
    // ============================================================================

    #[test]
    fn test_rule_counts() {
        // Verify our conversions handle all rules
        let ortho_char = orthography_rules_char();
        let phonetic_char = phonetic_rules_char();
        let test_char = test_rules_char();
        let zompist_char = zompist_rules_char();

        assert_eq!(ortho_char.len(), 45);
        assert_eq!(phonetic_char.len(), 3);
        assert_eq!(test_char.len(), 2);
        assert_eq!(zompist_char.len(), 62);

        // Byte-level
        let ortho = orthography_rules();
        let phonetic = phonetic_rules();
        let test = test_rules();
        let zompist = zompist_rules();

        assert_eq!(ortho.len(), 45);
        assert_eq!(phonetic.len(), 3);
        assert_eq!(test.len(), 2);
        assert_eq!(zompist.len(), 62);
    }
}
