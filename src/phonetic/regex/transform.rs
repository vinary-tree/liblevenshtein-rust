//! AST transformations for regex flags.
//!
//! This module implements flag semantics by transforming the regex AST before
//! NFA compilation. Flags like `(?i)` (case-insensitive) and `(?a)` (accent-insensitive)
//! are applied by expanding characters into character classes.
//!
//! # Supported Flags
//!
//! | Flag | Description | Implementation |
//! |------|-------------|----------------|
//! | `(?i)` | Case-insensitive | `a` → `[aA]` |
//! | `(?a)` | Accent-insensitive | `e` → `[eéèêë]` |
//! | `(?f)` | Feature-based | `p` → `[pb]` (voiced/voiceless) |
//! | `(?ia)` | Both combined | `e` → `[eéèêëEÉÈÊË]` |
//! | `(?u:NFC)` | Unicode normalization | Applied at runtime |
//! | `(?m)` | Multiline | Handled at NFA runtime |
//! | `(?s)` | Dotall | Handled at NFA runtime |
//!
//! # Example
//!
//! ```rust
//! use liblevenshtein::phonetic::regex::{parse, transform::apply_flags};
//!
//! let regex = parse("(?i:hello)").unwrap();
//! let transformed = apply_flags(&regex);
//! // Now "hello" is transformed to match "HELLO", "Hello", etc.
//! ```

use super::ast::{ContextExpr, ContextPredicate, Regex, RegexFlags, UnicodeNormalization};
use crate::phonetic::features::expand_feature_based;
use crate::phonetic::nfa::types::CharClassChar;
use unicode_normalization::UnicodeNormalization as UnicodeNormTrait;

/// Result of applying flags to a regex.
///
/// Contains the transformed regex and any extracted runtime flags
/// (like unicode normalization) that can't be applied via AST transformation.
#[derive(Debug, Clone)]
pub struct TransformResult {
    /// The transformed regex with flag semantics applied.
    pub regex: Regex,
    /// Unicode normalization to apply to input at runtime.
    pub unicode_normalization: Option<UnicodeNormalization>,
    /// Whether multiline mode is enabled (for `^` and `$`).
    pub multiline: bool,
    /// Whether dotall mode is enabled (`.` matches newlines).
    pub dotall: bool,
    /// Local Levenshtein distance override (from `(?;N)` syntax).
    ///
    /// When set, this distance limit should be used instead of the global
    /// distance parameter for matching against this pattern.
    pub local_distance: Option<u8>,
}

impl TransformResult {
    /// Create a new transform result with just the regex.
    pub fn new(regex: Regex) -> Self {
        Self {
            regex,
            unicode_normalization: None,
            multiline: false,
            dotall: false,
            local_distance: None,
        }
    }
}

/// Transform regex AST to apply flag semantics.
///
/// This is the main entry point for flag transformation. It walks the AST
/// and expands characters/classes based on active flags.
///
/// # Arguments
///
/// * `regex` - The regex to transform
///
/// # Returns
///
/// A `TransformResult` containing the transformed regex and any runtime flags.
pub fn apply_flags(regex: &Regex) -> TransformResult {
    let mut result = TransformResult::new(Regex::Empty);
    let transformed = apply_flags_with_context(regex, &RegexFlags::default(), &mut result);
    result.regex = transformed;
    result
}

/// Extract inline flags from the leftmost position of a regex pattern.
///
/// This handles nested left-associative Concat structures like:
/// `Concat(Concat(FlagsGroup(?i), Char(a)), Concat(Char(b), Char(c)))`
///
/// Returns the extracted flags (if any) and the remaining pattern with the flags removed.
fn extract_leftmost_inline_flags(regex: &Regex) -> (Option<RegexFlags>, Regex) {
    match regex {
        Regex::FlagsGroup { flags, inner: None } => {
            // Found inline flags - return them and Empty as remaining
            (Some(flags.clone()), Regex::Empty)
        }
        Regex::Concat(a, b) => {
            // Recursively check the left side
            let (flags, remaining_a) = extract_leftmost_inline_flags(a);
            if flags.is_some() {
                // Found flags in left subtree
                match remaining_a {
                    Regex::Empty => {
                        // Left was just flags - return right as remaining
                        (flags, (**b).clone())
                    }
                    _ => {
                        // Left had flags + more content - rebuild Concat
                        (flags, Regex::Concat(Box::new(remaining_a), b.clone()))
                    }
                }
            } else {
                // No flags found - return unchanged
                (None, regex.clone())
            }
        }
        _ => (None, regex.clone()),
    }
}

/// Apply flags with inherited context.
fn apply_flags_with_context(
    regex: &Regex,
    inherited: &RegexFlags,
    result: &mut TransformResult,
) -> Regex {
    match regex {
        // FlagsGroup: merge flags and transform inner pattern
        Regex::FlagsGroup { flags, inner } => {
            let merged = inherited.merge(flags);

            // Extract runtime flags
            if let Some(norm) = merged.unicode_normalization {
                result.unicode_normalization = Some(norm);
            }
            if merged.multiline == Some(true) {
                result.multiline = true;
            }
            if merged.dotall == Some(true) {
                result.dotall = true;
            }
            if let Some(dist) = merged.local_distance {
                result.local_distance = Some(dist);
            }

            match inner {
                Some(inner) => apply_flags_with_context(inner, &merged, result),
                None => Regex::Empty,
            }
        }

        // Char: expand for case/accent/feature insensitive
        Regex::Char(c) => {
            let case_insensitive = inherited.case_insensitive == Some(true);
            let accent_insensitive = inherited.accent_insensitive == Some(true);
            let feature_based = inherited.feature_based == Some(true);

            if case_insensitive || accent_insensitive || feature_based {
                expand_char(*c, case_insensitive, accent_insensitive, feature_based)
            } else {
                Regex::Char(*c)
            }
        }

        // CharClass: expand each character in the class
        Regex::CharClass(class) => {
            let case_insensitive = inherited.case_insensitive == Some(true);
            let accent_insensitive = inherited.accent_insensitive == Some(true);
            let feature_based = inherited.feature_based == Some(true);

            if case_insensitive || accent_insensitive || feature_based {
                expand_char_class(class, case_insensitive, accent_insensitive, feature_based)
            } else {
                Regex::CharClass(class.clone())
            }
        }

        // Concat: transform both sides, propagating inline flags
        Regex::Concat(a, b) => {
            // Extract inline flags from leftmost position (handles nested Concats)
            let (leftmost_flags, remaining) = extract_leftmost_inline_flags(regex);

            if let Some(flags) = leftmost_flags {
                let merged = inherited.merge(&flags);
                // Extract runtime flags
                if let Some(norm) = merged.unicode_normalization {
                    result.unicode_normalization = Some(norm);
                }
                if merged.multiline == Some(true) {
                    result.multiline = true;
                }
                if merged.dotall == Some(true) {
                    result.dotall = true;
                }
                if let Some(dist) = merged.local_distance {
                    result.local_distance = Some(dist);
                }
                // Transform remaining pattern with merged flags
                apply_flags_with_context(&remaining, &merged, result)
            } else {
                // No inline flags - transform both sides normally
                let a_transformed = apply_flags_with_context(a, inherited, result);
                let b_transformed = apply_flags_with_context(b, inherited, result);
                // Eliminate Empty nodes
                match (&a_transformed, &b_transformed) {
                    (Regex::Empty, _) => b_transformed,
                    (_, Regex::Empty) => a_transformed,
                    _ => Regex::Concat(Box::new(a_transformed), Box::new(b_transformed)),
                }
            }
        }

        // Alt: transform both alternatives
        Regex::Alt(a, b) => {
            let a_transformed = apply_flags_with_context(a, inherited, result);
            let b_transformed = apply_flags_with_context(b, inherited, result);
            Regex::Alt(Box::new(a_transformed), Box::new(b_transformed))
        }

        // Quantifiers: transform inner pattern
        Regex::Star(inner) => {
            let inner_transformed = apply_flags_with_context(inner, inherited, result);
            Regex::Star(Box::new(inner_transformed))
        }
        Regex::Plus(inner) => {
            let inner_transformed = apply_flags_with_context(inner, inherited, result);
            Regex::Plus(Box::new(inner_transformed))
        }
        Regex::Optional(inner) => {
            let inner_transformed = apply_flags_with_context(inner, inherited, result);
            Regex::Optional(Box::new(inner_transformed))
        }
        Regex::RepeatExact(inner, n) => {
            let inner_transformed = apply_flags_with_context(inner, inherited, result);
            Regex::RepeatExact(Box::new(inner_transformed), *n)
        }
        Regex::RepeatRange(inner, min, max) => {
            let inner_transformed = apply_flags_with_context(inner, inherited, result);
            Regex::RepeatRange(Box::new(inner_transformed), *min, *max)
        }

        // Groups: transform inner pattern
        Regex::CapturingGroup(num, inner) => {
            let inner_transformed = apply_flags_with_context(inner, inherited, result);
            Regex::CapturingGroup(*num, Box::new(inner_transformed))
        }
        Regex::NonCapturingGroup(inner) => {
            let inner_transformed = apply_flags_with_context(inner, inherited, result);
            Regex::NonCapturingGroup(Box::new(inner_transformed))
        }
        Regex::NamedGroup(name, inner) => {
            let inner_transformed = apply_flags_with_context(inner, inherited, result);
            Regex::NamedGroup(name.clone(), Box::new(inner_transformed))
        }

        // RewriteRule: transform pattern and replacement
        Regex::RewriteRule {
            pattern,
            replacement,
            context,
            weight,
        } => {
            let pattern_transformed = apply_flags_with_context(pattern, inherited, result);
            let replacement_transformed = apply_flags_with_context(replacement, inherited, result);
            let context_transformed = context
                .as_ref()
                .map(|ctx| Box::new(transform_context_predicate(ctx, inherited, result)));
            Regex::RewriteRule {
                pattern: Box::new(pattern_transformed),
                replacement: Box::new(replacement_transformed),
                context: context_transformed,
                weight: *weight,
            }
        }

        // Pass-through: no transformation needed
        Regex::Empty => Regex::Empty,
        // Any (.): depends on dotall flag
        // Without dotall: . matches any char EXCEPT \r and \n
        // With dotall: . matches ANY char including newlines
        Regex::Any => {
            let dotall = inherited.dotall == Some(true);
            if dotall {
                // Dotall mode: . matches everything (default NFA behavior)
                Regex::Any
            } else {
                // Standard mode: . matches everything except newlines
                // Transform to [^\r\n]
                let mut class = CharClassChar::new();
                class.add_char('\r');
                class.add_char('\n');
                class.negated = true;
                Regex::CharClass(class)
            }
        }
        Regex::GroupRef(name) => Regex::GroupRef(name.clone()),
        Regex::WordBoundary => Regex::WordBoundary,
        Regex::StartOfLine => Regex::StartOfLine,
        Regex::EndOfLine => Regex::EndOfLine,
        Regex::StartOfInput => Regex::StartOfInput,
        Regex::EndOfInput => Regex::EndOfInput,
        Regex::EndOfInputStrict => Regex::EndOfInputStrict,
    }
}

/// Transform a context predicate.
fn transform_context_predicate(
    ctx: &ContextPredicate,
    inherited: &RegexFlags,
    result: &mut TransformResult,
) -> ContextPredicate {
    ContextPredicate {
        left: ctx
            .left
            .as_ref()
            .map(|e| transform_context_expr(e, inherited, result)),
        right: ctx
            .right
            .as_ref()
            .map(|e| transform_context_expr(e, inherited, result)),
        syllable: ctx.syllable.clone(),
    }
}

/// Transform a context expression.
fn transform_context_expr(
    expr: &ContextExpr,
    inherited: &RegexFlags,
    result: &mut TransformResult,
) -> ContextExpr {
    match expr {
        ContextExpr::Pattern(regex) => {
            ContextExpr::Pattern(apply_flags_with_context(regex, inherited, result))
        }
        ContextExpr::WordBoundary => ContextExpr::WordBoundary,
        ContextExpr::And(a, b) => ContextExpr::And(
            Box::new(transform_context_expr(a, inherited, result)),
            Box::new(transform_context_expr(b, inherited, result)),
        ),
        ContextExpr::Or(a, b) => ContextExpr::Or(
            Box::new(transform_context_expr(a, inherited, result)),
            Box::new(transform_context_expr(b, inherited, result)),
        ),
        ContextExpr::Not(inner) => {
            ContextExpr::Not(Box::new(transform_context_expr(inner, inherited, result)))
        }
    }
}

/// Expand a single character based on flags.
fn expand_char(
    c: char,
    case_insensitive: bool,
    accent_insensitive: bool,
    feature_based: bool,
) -> Regex {
    let mut chars = vec![c];

    // Collect all variants
    if case_insensitive {
        add_case_variants(&mut chars);
    }
    if accent_insensitive {
        add_accent_variants(&mut chars);
    }
    if feature_based {
        add_feature_variants(&mut chars);
    }

    // Remove duplicates
    chars.sort();
    chars.dedup();

    if chars.len() == 1 {
        Regex::Char(chars[0])
    } else {
        // Create character class with all variants
        let class = CharClassChar::from_chars(&chars);
        Regex::CharClass(class)
    }
}

/// Expand a character class based on flags.
fn expand_char_class(
    class: &CharClassChar,
    case_insensitive: bool,
    accent_insensitive: bool,
    feature_based: bool,
) -> Regex {
    let base_capacity = char_class_expansion_capacity(class);
    let variant_factor = 1
        + usize::from(case_insensitive)
        + usize::from(accent_insensitive)
        + usize::from(feature_based);
    let mut expanded = Vec::with_capacity(base_capacity.saturating_mul(variant_factor));

    for &(start, end) in &class.ranges {
        for c in start..=end {
            expanded.push(c);
            if case_insensitive {
                add_case_variants_for_char(c, &mut expanded);
            }
            if accent_insensitive {
                add_accent_variants_for_char(c, &mut expanded);
            }
            if feature_based {
                add_feature_variants_for_char(c, &mut expanded);
            }
        }
    }

    // Remove duplicates
    expanded.sort();
    expanded.dedup();

    // Create new character class
    let new_class = CharClassChar {
        ranges: chars_to_ranges(&expanded),
        negated: class.negated,
    };

    Regex::CharClass(new_class)
}

fn inclusive_char_range_len(start: char, end: char) -> Option<usize> {
    let start = u32::from(start);
    let end = u32::from(end);
    let len = end.checked_sub(start)?.checked_add(1)?;
    usize::try_from(len).ok()
}

fn char_class_expansion_capacity(class: &CharClassChar) -> usize {
    class
        .ranges
        .iter()
        .filter_map(|&(start, end)| inclusive_char_range_len(start, end))
        .try_fold(0usize, |total, len| total.checked_add(len))
        .unwrap_or(usize::MAX)
}

fn adjacent_scalars(previous: char, current: char) -> bool {
    u32::from(previous).checked_add(1) == Some(u32::from(current))
}

/// Add case variants to a character list.
fn add_case_variants(chars: &mut Vec<char>) {
    let original: Vec<char> = chars.clone();
    for c in original {
        add_case_variants_for_char(c, chars);
    }
}

/// Add case variants for a single character.
fn add_case_variants_for_char(c: char, chars: &mut Vec<char>) {
    // Add lowercase variant
    for lower in c.to_lowercase() {
        if !chars.contains(&lower) {
            chars.push(lower);
        }
    }
    // Add uppercase variant
    for upper in c.to_uppercase() {
        if !chars.contains(&upper) {
            chars.push(upper);
        }
    }
}

/// Add accent variants to a character list.
fn add_accent_variants(chars: &mut Vec<char>) {
    let original: Vec<char> = chars.clone();
    for c in original {
        add_accent_variants_for_char(c, chars);
    }
}

/// Add accent variants for a single character.
fn add_accent_variants_for_char(c: char, chars: &mut Vec<char>) {
    // Get base character by NFD decomposition
    let base = get_base_char(c);

    // Add the base character
    if !chars.contains(&base) {
        chars.push(base);
    }

    // Add common accent variants for this base
    let variants = get_accent_variants(base);
    for v in variants {
        if !chars.contains(&v) {
            chars.push(v);
        }
    }
}

/// Add phonetic feature variants to a character list.
fn add_feature_variants(chars: &mut Vec<char>) {
    let original: Vec<char> = chars.clone();
    for c in original {
        add_feature_variants_for_char(c, chars);
    }
}

/// Add phonetic feature variants for a single character.
///
/// Uses the phonetic features module to expand a character to all
/// phonetically similar characters (same place/manner of articulation,
/// differing in voicing).
fn add_feature_variants_for_char(c: char, chars: &mut Vec<char>) {
    let variants = expand_feature_based(c);
    for v in variants {
        if !chars.contains(&v) {
            chars.push(v);
        }
    }
}

/// Get the base character by stripping accents (NFD decomposition).
fn get_base_char(c: char) -> char {
    // NFD decompose and take the first character (the base)
    c.to_string().nfd().next().unwrap_or(c)
}

/// Get common accent variants for a base character.
fn get_accent_variants(base: char) -> Vec<char> {
    match base {
        // Lowercase vowels
        'a' => vec!['a', 'à', 'á', 'â', 'ã', 'ä', 'å', 'ā', 'ă', 'ą'],
        'e' => vec!['e', 'è', 'é', 'ê', 'ë', 'ē', 'ĕ', 'ė', 'ę', 'ě'],
        'i' => vec!['i', 'ì', 'í', 'î', 'ï', 'ĩ', 'ī', 'ĭ', 'į', 'ı'],
        'o' => vec!['o', 'ò', 'ó', 'ô', 'õ', 'ö', 'ō', 'ŏ', 'ő', 'ø'],
        'u' => vec!['u', 'ù', 'ú', 'û', 'ü', 'ũ', 'ū', 'ŭ', 'ů', 'ű', 'ų'],
        'y' => vec!['y', 'ý', 'ÿ', 'ŷ'],

        // Lowercase consonants with common diacritics
        'c' => vec!['c', 'ç', 'ć', 'ĉ', 'č'],
        'd' => vec!['d', 'ď', 'đ'],
        'g' => vec!['g', 'ĝ', 'ğ', 'ġ', 'ģ'],
        'h' => vec!['h', 'ĥ', 'ħ'],
        'j' => vec!['j', 'ĵ'],
        'k' => vec!['k', 'ķ'],
        'l' => vec!['l', 'ĺ', 'ļ', 'ľ', 'ł'],
        'n' => vec!['n', 'ñ', 'ń', 'ņ', 'ň'],
        'r' => vec!['r', 'ŕ', 'ŗ', 'ř'],
        's' => vec!['s', 'ś', 'ŝ', 'ş', 'š'],
        't' => vec!['t', 'ţ', 'ť', 'ŧ'],
        'w' => vec!['w', 'ŵ'],
        'z' => vec!['z', 'ź', 'ż', 'ž'],

        // Uppercase vowels
        'A' => vec!['A', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Ā', 'Ă', 'Ą'],
        'E' => vec!['E', 'È', 'É', 'Ê', 'Ë', 'Ē', 'Ĕ', 'Ė', 'Ę', 'Ě'],
        'I' => vec!['I', 'Ì', 'Í', 'Î', 'Ï', 'Ĩ', 'Ī', 'Ĭ', 'Į'],
        'O' => vec!['O', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Ō', 'Ŏ', 'Ő', 'Ø'],
        'U' => vec!['U', 'Ù', 'Ú', 'Û', 'Ü', 'Ũ', 'Ū', 'Ŭ', 'Ů', 'Ű', 'Ų'],
        'Y' => vec!['Y', 'Ý', 'Ÿ', 'Ŷ'],

        // Uppercase consonants
        'C' => vec!['C', 'Ç', 'Ć', 'Ĉ', 'Č'],
        'D' => vec!['D', 'Ď', 'Đ'],
        'G' => vec!['G', 'Ĝ', 'Ğ', 'Ġ', 'Ģ'],
        'H' => vec!['H', 'Ĥ', 'Ħ'],
        'J' => vec!['J', 'Ĵ'],
        'K' => vec!['K', 'Ķ'],
        'L' => vec!['L', 'Ĺ', 'Ļ', 'Ľ', 'Ł'],
        'N' => vec!['N', 'Ñ', 'Ń', 'Ņ', 'Ň'],
        'R' => vec!['R', 'Ŕ', 'Ŗ', 'Ř'],
        'S' => vec!['S', 'Ś', 'Ŝ', 'Ş', 'Š'],
        'T' => vec!['T', 'Ţ', 'Ť', 'Ŧ'],
        'W' => vec!['W', 'Ŵ'],
        'Z' => vec!['Z', 'Ź', 'Ż', 'Ž'],

        // Special characters
        'æ' => vec!['æ', 'ǽ'],
        'Æ' => vec!['Æ', 'Ǽ'],
        'œ' => vec!['œ'],
        'Œ' => vec!['Œ'],
        'ß' => vec!['ß'],

        // No variants for other characters
        _ => vec![base],
    }
}

/// Convert a sorted list of characters to optimized ranges.
fn chars_to_ranges(chars: &[char]) -> Vec<(char, char)> {
    if chars.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::with_capacity(chars.len());
    let mut start = chars[0];
    let mut end = chars[0];

    for &c in chars.iter().skip(1) {
        if adjacent_scalars(end, c) {
            // Extend current range
            end = c;
        } else {
            // Start new range
            ranges.push((start, end));
            start = c;
            end = c;
        }
    }

    // Don't forget the last range
    ranges.push((start, end));

    ranges
}

/// Normalize input string according to Unicode normalization form.
///
/// This is applied at runtime before matching.
pub fn normalize_input(input: &str, form: UnicodeNormalization) -> String {
    match form {
        UnicodeNormalization::NFC => input.nfc().collect(),
        UnicodeNormalization::NFD => input.nfd().collect(),
        UnicodeNormalization::NFKC => input.nfkc().collect(),
        UnicodeNormalization::NFKD => input.nfkd().collect(),
    }
}

/// Extract flags from a regex AST.
///
/// This extracts the effective flags from the top level of the regex,
/// useful for determining runtime behavior.
pub fn extract_flags(regex: &Regex) -> RegexFlags {
    match regex {
        Regex::FlagsGroup { flags, .. } => flags.clone(),
        _ => RegexFlags::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::regex::parse;

    #[test]
    fn test_case_insensitive_char() {
        let expanded = expand_char('a', true, false, false);
        match expanded {
            Regex::CharClass(class) => {
                assert!(class.matches('a'));
                assert!(class.matches('A'));
            }
            _ => panic!("Expected CharClass"),
        }
    }

    #[test]
    fn test_accent_insensitive_char() {
        let expanded = expand_char('e', false, true, false);
        match expanded {
            Regex::CharClass(class) => {
                assert!(class.matches('e'));
                assert!(class.matches('é'));
                assert!(class.matches('è'));
                assert!(class.matches('ê'));
                assert!(class.matches('ë'));
            }
            _ => panic!("Expected CharClass"),
        }
    }

    #[test]
    fn test_combined_flags() {
        let expanded = expand_char('e', true, true, false);
        match expanded {
            Regex::CharClass(class) => {
                // Should have lowercase and uppercase with accents
                assert!(class.matches('e'));
                assert!(class.matches('E'));
                assert!(class.matches('é'));
                assert!(class.matches('É'));
            }
            _ => panic!("Expected CharClass"),
        }
    }

    #[test]
    fn test_no_expansion_for_digit() {
        let expanded = expand_char('5', true, true, false);
        match expanded {
            Regex::Char('5') => {}
            _ => panic!("Expected unchanged Char('5')"),
        }
    }

    #[test]
    fn test_feature_based_char() {
        // p -> [bp] (voiced/voiceless bilabial stops)
        let expanded = expand_char('p', false, false, true);
        match expanded {
            Regex::CharClass(class) => {
                assert!(class.matches('p'));
                assert!(class.matches('b')); // voiced counterpart
            }
            _ => panic!("Expected CharClass"),
        }
    }

    #[test]
    fn test_feature_based_voiced_unvoiced_pairs() {
        // Test several voiced/voiceless pairs
        for (voiceless, voiced) in [('p', 'b'), ('t', 'd'), ('k', 'g'), ('f', 'v'), ('s', 'z')] {
            let expanded = expand_char(voiceless, false, false, true);
            match expanded {
                Regex::CharClass(class) => {
                    assert!(class.matches(voiceless), "Expected {} in class", voiceless);
                    assert!(
                        class.matches(voiced),
                        "Expected {} in class for {}",
                        voiced,
                        voiceless
                    );
                }
                _ => panic!("Expected CharClass for {}", voiceless),
            }
        }
    }

    #[test]
    fn test_apply_flags_case_insensitive() {
        let regex = parse("(?i:abc)").expect("should parse");
        let result = apply_flags(&regex);

        // The transformed regex should have 'a', 'b', 'c' expanded to include uppercase
        let regex_str = format!("{}", result.regex);
        // It should contain character classes now
        assert!(regex_str.contains('[') || regex_str.len() > 3);
    }

    #[test]
    fn test_apply_flags_unicode_normalization() {
        let regex = parse("(?u:NFC:test)").expect("should parse");
        let result = apply_flags(&regex);

        assert_eq!(
            result.unicode_normalization,
            Some(UnicodeNormalization::NFC)
        );
    }

    #[test]
    fn test_normalize_input() {
        // Composed vs decomposed é
        let composed = "café"; // é as single codepoint
        let decomposed = "cafe\u{0301}"; // e + combining acute

        let normalized_composed = normalize_input(composed, UnicodeNormalization::NFC);
        let normalized_decomposed = normalize_input(decomposed, UnicodeNormalization::NFC);

        assert_eq!(normalized_composed, normalized_decomposed);
    }

    #[test]
    fn test_chars_to_ranges() {
        let chars = vec!['a', 'b', 'c', 'x', 'y', 'z'];
        let ranges = chars_to_ranges(&chars);

        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], ('a', 'c'));
        assert_eq!(ranges[1], ('x', 'z'));
    }

    #[test]
    fn inclusive_char_range_len_handles_edges() {
        assert_eq!(inclusive_char_range_len('a', 'a'), Some(1));
        assert_eq!(inclusive_char_range_len('a', 'c'), Some(3));
        assert_eq!(inclusive_char_range_len('c', 'a'), None);
        assert_eq!(inclusive_char_range_len(char::MAX, char::MAX), Some(1));
    }

    #[test]
    fn char_class_expansion_capacity_ignores_invalid_ranges() {
        let class = CharClassChar {
            ranges: vec![('z', 'a'), ('a', 'c'), (char::MAX, char::MAX)],
            negated: false,
        };

        assert_eq!(char_class_expansion_capacity(&class), 4);
    }

    #[test]
    fn adjacent_scalars_handles_unicode_maximum() {
        let before_max = char::from_u32(u32::from(char::MAX) - 1).expect("valid scalar");

        assert!(adjacent_scalars('a', 'b'));
        assert!(adjacent_scalars(before_max, char::MAX));
        assert!(!adjacent_scalars(char::MAX, char::MAX));
        assert!(!adjacent_scalars(char::MAX, '\0'));
    }

    #[test]
    fn chars_to_ranges_coalesces_up_to_unicode_maximum() {
        let before_max = char::from_u32(u32::from(char::MAX) - 1).expect("valid scalar");
        let ranges = chars_to_ranges(&[before_max, char::MAX]);

        assert_eq!(ranges, vec![(before_max, char::MAX)]);
    }

    #[test]
    fn expand_char_class_preserves_unicode_maximum() {
        let class = CharClassChar {
            ranges: vec![(char::MAX, char::MAX)],
            negated: false,
        };

        match expand_char_class(&class, true, true, true) {
            Regex::CharClass(expanded) => {
                assert!(expanded.matches(char::MAX));
                assert_eq!(expanded.ranges, vec![(char::MAX, char::MAX)]);
            }
            other => panic!("expected CharClass, got {other:?}"),
        }
    }

    #[test]
    fn test_multiline_flag_extracted() {
        let regex = parse("(?m:^test$)").expect("should parse");
        let result = apply_flags(&regex);

        assert!(result.multiline);
    }

    #[test]
    fn test_dotall_flag_extracted() {
        let regex = parse("(?s:a.b)").expect("should parse");
        let result = apply_flags(&regex);

        assert!(result.dotall);
    }

    #[test]
    fn test_inline_flags_propagate_to_subsequent_pattern() {
        // Test that (?i)abc propagates case-insensitivity to "abc"
        let regex = parse("(?i)abc").expect("should parse");
        let result = apply_flags(&regex);

        // The transformed regex should have 'a', 'b', 'c' expanded to include uppercase
        // Check that the result is not just empty - it should contain the pattern
        let regex_str = format!("{}", result.regex);
        // Should contain character classes for case-insensitive matching
        assert!(
            regex_str.contains('['),
            "Expected character classes in: {}",
            regex_str
        );
        assert!(!regex_str.is_empty(), "Result should not be empty");
    }

    #[test]
    fn test_inline_flags_case_insensitive() {
        // (?i)abc should be equivalent to (?i:abc)
        let inline_regex = parse("(?i)abc").expect("should parse");
        let scoped_regex = parse("(?i:abc)").expect("should parse");

        let inline_result = apply_flags(&inline_regex);
        let scoped_result = apply_flags(&scoped_regex);

        // Both should produce equivalent transformed regexes
        let inline_str = format!("{}", inline_result.regex);
        let scoped_str = format!("{}", scoped_result.regex);

        assert_eq!(inline_str, scoped_str,
            "Inline (?i)abc and scoped (?i:abc) should produce same result.\nInline: {}\nScoped: {}",
            inline_str, scoped_str);
    }

    #[test]
    fn test_inline_flags_multiline() {
        // (?m)^test$ should extract multiline flag
        let regex = parse("(?m)^test$").expect("should parse");
        let result = apply_flags(&regex);

        assert!(
            result.multiline,
            "Multiline flag should be extracted from inline (?m)"
        );
    }

    #[test]
    fn test_inline_flags_dotall() {
        // (?s)a.b should extract dotall flag
        let regex = parse("(?s)a.b").expect("should parse");
        let result = apply_flags(&regex);

        assert!(
            result.dotall,
            "Dotall flag should be extracted from inline (?s)"
        );
    }

    #[test]
    fn test_inline_unicode_normalization() {
        // (?u:NFC)test should extract unicode normalization
        let regex = parse("(?u:NFC)test").expect("should parse");
        let result = apply_flags(&regex);

        assert_eq!(
            result.unicode_normalization,
            Some(UnicodeNormalization::NFC),
            "Unicode normalization should be extracted from inline (?u:NFC)"
        );
    }

    #[test]
    fn test_local_distance_scoped() {
        // (?;2:test) should extract local_distance = 2
        let regex = parse("(?;2:test)").expect("should parse");
        let result = apply_flags(&regex);

        assert_eq!(
            result.local_distance,
            Some(2),
            "Local distance should be extracted from scoped (?;2:...)"
        );
    }

    #[test]
    fn test_local_distance_inline() {
        // (?;1)test should extract local_distance = 1
        let regex = parse("(?;1)test").expect("should parse");
        let result = apply_flags(&regex);

        assert_eq!(
            result.local_distance,
            Some(1),
            "Local distance should be extracted from inline (?;N)"
        );
    }

    #[test]
    fn test_local_distance_with_flags() {
        // (?i;0:test) should extract both case_insensitive and local_distance = 0
        let regex = parse("(?i;0:test)").expect("should parse");
        let result = apply_flags(&regex);

        assert_eq!(
            result.local_distance,
            Some(0),
            "Local distance should be extracted from combined flags"
        );
        // The pattern should also be case-insensitive expanded
        let display = format!("{}", result.regex);
        assert!(
            display.contains("[tT]") || display.contains("Tt"),
            "Case insensitive flag should also apply: {}",
            display
        );
    }
}
