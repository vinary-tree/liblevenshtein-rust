//! Runtime rule set conversion from `.llev` AST.
//!
//! This module converts parsed `.llev` files into runtime-usable rule sets.
//! It bridges the gap between the flexible AST representation and the
//! efficient runtime types used by the phonetic application functions.
//!
//! # Conversion Process
//!
//! 1. **Symbol Expansion**: `@define` symbols are expanded in patterns/contexts
//! 2. **Pattern Conversion**: AST expressions → `Vec<Phone<U>>`
//! 3. **Context Mapping**: AST contexts → runtime `Context<U>`
//! 4. **Rule Assembly**: Combines metadata with converted rule data
//!
//! # Limitations
//!
//! The runtime rule format is simpler than the AST format. Not all AST
//! patterns can be converted:
//!
//! - **Supported**: Literal sequences, word boundaries, simple character classes
//! - **Unsupported**: Quantifiers (`*`, `+`, `?`), alternation (`|`), complex regex
//!
//! For unsupported patterns, use the NFA-based application functions instead.
//!
//! # Usage
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::llev::{RuleSetChar, parse_str};
//! use liblevenshtein::phonetic::apply_rules_seq_char;
//!
//! let llev = parse_str(r#"
//!     ph -> f;
//!     c -> s / _[ei];
//! "#)?;
//!
//! let ruleset = RuleSetChar::from_llev(&llev)?;
//! let result = apply_rules_seq_char(&ruleset.rules, "phone");
//! ```

use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::ast::{ContextAST, ContextExpr, Expression, LLevFile, RuleDefinition, SymbolDef};
use super::error::{LLevError, LLevErrorKind, LLevResult, Position};
use crate::phonetic::common::PhoneticUnit;
use crate::phonetic::types::{Context, Phone, RewriteRule};

// ============================================================================
// Generic Rule Set
// ============================================================================

/// A collection of rewrite rules converted from a `.llev` file.
///
/// This is the generic rule set type that works with any phonetic unit type
/// (u8 for ASCII/byte-level, char for Unicode/character-level).
///
/// For convenience, use the type aliases:
/// - [`RuleSet`] for byte-level (ASCII) rules
/// - [`RuleSetChar`] for character-level (Unicode) rules
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serialization",
    serde(bound = "U: Serialize + for<'a> Deserialize<'a>")
)]
pub struct RuleSetGeneric<U: PhoneticUnit> {
    /// Converted rules ready for application
    pub rules: Vec<RewriteRule<U>>,

    /// Source file metadata (if available)
    pub name: Option<String>,
    /// Version string declared in the source file, when present.
    pub version: Option<String>,
}

impl<U: PhoneticUnit> RuleSetGeneric<U> {
    /// Create a new empty rule set.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            name: None,
            version: None,
        }
    }

    /// Get the number of rules in the set.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if the rule set is empty.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Merge another rule set into this one.
    ///
    /// Rules from the other set are appended to this set.
    pub fn merge(&mut self, other: RuleSetGeneric<U>) {
        self.rules.extend(other.rules);
    }
}

impl<U: PhoneticUnit> Default for RuleSetGeneric<U> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Byte-level Rule Set (backward-compatible alias)
// ============================================================================

/// A collection of byte-level rewrite rules converted from a `.llev` file.
///
/// Use this for ASCII-only text where byte-level operations are sufficient.
/// For Unicode support, use [`RuleSetChar`] instead.
pub type RuleSet = RuleSetGeneric<u8>;

impl RuleSet {
    /// Convert a parsed `.llev` file into a byte-level rule set.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A pattern contains unsupported AST nodes (quantifiers, alternation, etc.)
    /// - A symbol reference cannot be resolved
    /// - A character cannot be converted to a single byte
    pub fn from_llev(file: &LLevFile) -> LLevResult<Self> {
        let converter = RuleConverter::<u8>::new(&file.symbols);
        let mut rules = Vec::with_capacity(file.rules.len());

        for (index, rule_def) in file.rules.iter().enumerate() {
            if !rule_def.metadata.enabled {
                continue;
            }

            let rule = converter.convert_rule(rule_def, index)?;
            rules.push(rule);
        }

        Ok(Self {
            rules,
            name: file.metadata.name.clone(),
            version: file.metadata.version.clone(),
        })
    }
}

// ============================================================================
// Character-level Rule Set (backward-compatible alias)
// ============================================================================

/// A collection of character-level rewrite rules converted from a `.llev` file.
///
/// Use this for Unicode text with accented characters, CJK, emoji, etc.
/// For ASCII-only text, [`RuleSet`] is slightly more efficient.
pub type RuleSetChar = RuleSetGeneric<char>;

impl RuleSetChar {
    /// Convert a parsed `.llev` file into a character-level rule set.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A pattern contains unsupported AST nodes (quantifiers, alternation, etc.)
    /// - A symbol reference cannot be resolved
    pub fn from_llev(file: &LLevFile) -> LLevResult<Self> {
        let converter = RuleConverter::<char>::new(&file.symbols);
        let mut rules = Vec::with_capacity(file.rules.len());

        for (index, rule_def) in file.rules.iter().enumerate() {
            if !rule_def.metadata.enabled {
                continue;
            }

            let rule = converter.convert_rule(rule_def, index)?;
            rules.push(rule);
        }

        Ok(Self {
            rules,
            name: file.metadata.name.clone(),
            version: file.metadata.version.clone(),
        })
    }

    /// Apply rules to a string, returning the transformed result.
    ///
    /// This is a convenience method that handles conversion between strings
    /// and PhoneChar arrays internally.
    ///
    /// Matching is case-insensitive by default. Input is lowercased before
    /// matching against patterns (which are also lowercased during ruleset
    /// conversion). Replacement output uses the exact case specified in rules.
    ///
    /// # Arguments
    ///
    /// - `input` - The input string to transform
    ///
    /// # Returns
    ///
    /// The transformed string with all rules applied to fixed point.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let ruleset = RuleSetChar::from_llev(&file)?;
    /// let result = ruleset.apply("phone"); // "fone"
    /// let result = ruleset.apply("PHONE"); // "fone" (case-insensitive match)
    /// ```
    pub fn apply(&self, input: &str) -> String {
        // Convert string to PhoneChar array with case folding for matching
        // Input is lowercased to match against lowercased patterns
        let phones: Vec<Phone<char>> = input
            .chars()
            .map(|c| {
                // Lowercase for case-insensitive matching
                let c_lower = c.to_lowercase().next().unwrap_or(c);
                if is_vowel_char(c_lower) {
                    Phone::Vowel(c_lower)
                } else {
                    Phone::Consonant(c_lower)
                }
            })
            .collect();

        // Apply rules with generous fuel
        let fuel = input.len() * 10 + 100;
        if let Some(result) = crate::phonetic::apply_rules_seq(&self.rules, &phones, fuel) {
            phones_to_string(&result)
        } else {
            // Fuel exhausted - return input unchanged
            input.to_string()
        }
    }

    /// Apply rules to a string with multi-symbol output expansion.
    ///
    /// This is retained as an explicit name for callers that want to emphasize
    /// that graph and sequence phones are expanded to their constituent
    /// characters. It has the same semantics as [`apply`](Self::apply).
    ///
    /// Matching is case-insensitive by default (same as [`apply`]).
    pub fn apply_full(&self, input: &str) -> String {
        self.apply(input)
    }
}

fn phones_to_string(phones: &[Phone<char>]) -> String {
    let mut output = String::with_capacity(phones.len() * 4);
    for phone in phones {
        match phone {
            Phone::Vowel(c) | Phone::Consonant(c) => output.push(*c),
            Phone::Digraph(c1, c2) => {
                output.push(*c1);
                output.push(*c2);
            }
            Phone::Trigraph(c1, c2, c3) => {
                output.push(*c1);
                output.push(*c2);
                output.push(*c3);
            }
            Phone::Tetragraph(c1, c2, c3, c4) => {
                output.push(*c1);
                output.push(*c2);
                output.push(*c3);
                output.push(*c4);
            }
            Phone::Pentagraph(c1, c2, c3, c4, c5) => {
                output.push(*c1);
                output.push(*c2);
                output.push(*c3);
                output.push(*c4);
                output.push(*c5);
            }
            Phone::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                output.push(*c1);
                output.push(*c2);
                output.push(*c3);
                output.push(*c4);
                output.push(*c5);
                output.push(*c6);
            }
            Phone::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                output.push(*c1);
                output.push(*c2);
                output.push(*c3);
                output.push(*c4);
                output.push(*c5);
                output.push(*c6);
                output.push(*c7);
            }
            Phone::Sequence(s) => {
                for c in s {
                    output.push(*c);
                }
            }
            Phone::Silent => {}
        }
    }
    output
}

// ============================================================================
// Generic Rule Converter
// ============================================================================

/// Internal converter for generic rules.
///
/// This unified converter works with both byte-level (u8) and character-level (char)
/// rule conversion, using the `PhoneticUnit` trait to abstract over the unit type.
struct RuleConverter<'a, U: PhoneticUnit> {
    /// Symbol table for expanding symbol references
    symbols: HashMap<&'a str, &'a Expression>,
    /// List of symbol names for error suggestions
    symbol_names: Vec<&'a str>,
    /// Phantom data to track the unit type
    _phantom: PhantomData<U>,
}

impl<'a, U: PhoneticUnit> RuleConverter<'a, U> {
    /// Create a new converter with the given symbol definitions.
    fn new(symbols: &'a [SymbolDef]) -> Self {
        let symbol_map: HashMap<_, _> = symbols
            .iter()
            .map(|s| (s.name.as_str(), &s.value))
            .collect();
        let symbol_names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        Self {
            symbols: symbol_map,
            symbol_names,
            _phantom: PhantomData,
        }
    }

    /// Convert an AST rule definition to a runtime rule.
    ///
    /// Patterns and contexts are converted with case folding enabled by default
    /// for case-insensitive matching. Replacements preserve original case.
    fn convert_rule(&self, rule_def: &RuleDefinition, index: usize) -> LLevResult<RewriteRule<U>> {
        let pos = rule_def.position;

        // Convert pattern with case folding for case-insensitive matching
        let pattern = self.convert_pattern(&rule_def.rule.pattern, pos, true)?;
        if pattern.is_empty() {
            return Err(LLevError::with_position(
                LLevErrorKind::InvalidRule("Pattern cannot be empty".into()),
                pos,
            ));
        }

        // Convert replacement WITHOUT case folding - preserve original output
        let replacement = self.convert_pattern(&rule_def.rule.replacement, pos, false)?;

        // Convert context with case folding for case-insensitive matching
        let context = self.convert_context(&rule_def.rule.context, pos)?;

        // Extract syllable condition from context AST
        let syllable_condition = rule_def
            .rule
            .context
            .as_ref()
            .and_then(|ctx| ctx.syllable.clone());

        // Determine weight (inline weight takes precedence over metadata weight)
        let weight = rule_def
            .rule
            .weight
            .or(rule_def.metadata.weight)
            .unwrap_or(1.0);

        // Determine ID (metadata ID or auto-assign from index)
        let rule_id = rule_def.metadata.id.unwrap_or(index);

        // Determine name
        let rule_name = rule_def
            .metadata
            .name
            .clone()
            .unwrap_or_else(|| format!("rule_{}", rule_id));

        Ok(RewriteRule {
            rule_id,
            rule_name,
            pattern,
            replacement,
            context,
            weight,
            syllable_condition,
        })
    }

    /// Convert an AST expression to a sequence of phones.
    ///
    /// # Arguments
    /// - `expr`: The expression to convert
    /// - `pos`: Position for error reporting
    /// - `case_fold`: If true, convert characters to lowercase for case-insensitive matching
    fn convert_pattern(
        &self,
        expr: &Expression,
        pos: Position,
        case_fold: bool,
    ) -> LLevResult<Vec<Phone<U>>> {
        match expr {
            Expression::Empty => Ok(Vec::new()),

            Expression::Char(c) => {
                let phone = self.char_to_phone(*c, pos, case_fold)?;
                Ok(vec![phone])
            }

            Expression::Concat(a, b) => {
                let mut result = self.convert_pattern(a, pos, case_fold)?;
                result.extend(self.convert_pattern(b, pos, case_fold)?);
                Ok(result)
            }

            Expression::Group(inner) => self.convert_pattern(inner, pos, case_fold),

            Expression::ScopedFlags { flags, inner } => {
                // Override case_fold based on the scoped flags
                let new_case_fold = match flags.case_insensitive {
                    Some(false) => false, // (?c:...) or (?-i:...) - case-sensitive
                    Some(true) => true,   // Explicit case-insensitive
                    None => case_fold,    // Use parent context's setting
                };
                self.convert_pattern(inner, pos, new_case_fold)
            }

            Expression::SymbolRef(name) => {
                if let Some(symbol_expr) = self.symbols.get(name.as_str()) {
                    self.convert_pattern(symbol_expr, pos, case_fold)
                } else {
                    // Use suggestion-aware error for better developer experience
                    Err(LLevError::undefined_symbol_with_suggestion(
                        name.clone(),
                        &self.symbol_names,
                        pos,
                    ))
                }
            }

            // Unsupported patterns for direct conversion
            Expression::CharClass { .. } => Err(LLevError::with_position(
                LLevErrorKind::UnsupportedPattern(
                    "Character classes in patterns require NFA-based matching".into(),
                ),
                pos,
            )),

            Expression::CharRange { .. } => Err(LLevError::with_position(
                LLevErrorKind::UnsupportedPattern(
                    "Character ranges in patterns require NFA-based matching".into(),
                ),
                pos,
            )),

            Expression::Any => Err(LLevError::with_position(
                LLevErrorKind::UnsupportedPattern(
                    "Wildcard (.) in patterns requires NFA-based matching".into(),
                ),
                pos,
            )),

            Expression::Alt(_, _) => Err(LLevError::with_position(
                LLevErrorKind::UnsupportedPattern(
                    "Alternation (|) in patterns requires NFA-based matching".into(),
                ),
                pos,
            )),

            Expression::Star(_) => Err(LLevError::with_position(
                LLevErrorKind::UnsupportedPattern(
                    "Kleene star (*) in patterns requires NFA-based matching".into(),
                ),
                pos,
            )),

            Expression::Plus(_) => Err(LLevError::with_position(
                LLevErrorKind::UnsupportedPattern(
                    "Kleene plus (+) in patterns requires NFA-based matching".into(),
                ),
                pos,
            )),

            Expression::Optional(_) => Err(LLevError::with_position(
                LLevErrorKind::UnsupportedPattern(
                    "Optional (?) in patterns requires NFA-based matching".into(),
                ),
                pos,
            )),

            Expression::RepeatExact(_, _) => Err(LLevError::with_position(
                LLevErrorKind::UnsupportedPattern(
                    "Repetition {n} in patterns requires NFA-based matching".into(),
                ),
                pos,
            )),

            Expression::RepeatRange { .. } => Err(LLevError::with_position(
                LLevErrorKind::UnsupportedPattern(
                    "Repetition {n,m} in patterns requires NFA-based matching".into(),
                ),
                pos,
            )),

            Expression::WordBoundary => Err(LLevError::with_position(
                LLevErrorKind::UnsupportedPattern(
                    "Word boundary (#) is only valid in contexts, not patterns".into(),
                ),
                pos,
            )),
        }
    }

    /// Convert a character to a Phone.
    ///
    /// # Arguments
    /// - `c`: The character to convert
    /// - `pos`: Position for error reporting
    /// - `case_fold`: If true, convert to lowercase for case-insensitive matching
    fn char_to_phone(&self, c: char, pos: Position, case_fold: bool) -> LLevResult<Phone<U>> {
        // Apply case folding if requested
        let c_folded = if case_fold {
            U::to_lowercase(
                U::from_char(c).ok_or_else(|| LLevError::non_ascii_in_byte_level(c, None, pos))?,
            )
        } else {
            U::from_char(c).ok_or_else(|| LLevError::non_ascii_in_byte_level(c, None, pos))?
        };

        // Classify as vowel or consonant
        if U::is_vowel(c_folded) {
            Ok(Phone::Vowel(c_folded))
        } else {
            Ok(Phone::Consonant(c_folded))
        }
    }

    /// Convert an AST context to a runtime Context.
    ///
    /// Supports:
    /// - Word boundaries (#_ and _#)
    /// - Character class contexts ([aeiou]_ and _[aeiou])
    /// - Compound contexts (And, Or, Not)
    /// - Both left AND right contexts simultaneously
    fn convert_context(
        &self,
        ctx_opt: &Option<ContextAST>,
        pos: Position,
    ) -> LLevResult<Context<U>> {
        let ctx = match ctx_opt {
            None => return Ok(Context::Anywhere),
            Some(ctx) => ctx,
        };

        // Convert left and right contexts separately, then combine
        let left_ctx = match &ctx.left {
            None => None,
            Some(expr) => Some(self.convert_context_expr(expr, pos, true)?),
        };

        let right_ctx = match &ctx.right {
            None => None,
            Some(expr) => Some(self.convert_context_expr(expr, pos, false)?),
        };

        // Combine left and right contexts
        match (left_ctx, right_ctx) {
            (None, None) => Ok(Context::Anywhere),
            (Some(left), None) => Ok(left),
            (None, Some(right)) => Ok(right),
            (Some(left), Some(right)) => {
                // Both left and right context - combine with And
                Ok(Context::And(Box::new(left), Box::new(right)))
            }
        }
    }

    /// Convert a context expression to a runtime Context.
    ///
    /// This handles compound contexts (And, Or, Not) recursively.
    fn convert_context_expr(
        &self,
        ctx_expr: &ContextExpr,
        pos: Position,
        is_left_context: bool,
    ) -> LLevResult<Context<U>> {
        match ctx_expr {
            ContextExpr::WordBoundary => {
                if is_left_context {
                    Ok(Context::Initial)
                } else {
                    Ok(Context::Final)
                }
            }
            ContextExpr::Pattern(expr) => {
                // Try to extract as character class
                if let Some(chars) = self.extract_char_class_from_expr(expr, pos)? {
                    if is_left_context {
                        // Left context - "after" the characters
                        if chars.iter().all(|&c| U::is_vowel(c)) {
                            Ok(Context::AfterVowel(chars))
                        } else if chars.iter().all(|&c| !U::is_vowel(c)) {
                            Ok(Context::AfterConsonant(chars))
                        } else {
                            // Mixed - use AfterVowel as fallback
                            Ok(Context::AfterVowel(chars))
                        }
                    } else {
                        // Right context - "before" the characters
                        if chars.iter().all(|&c| U::is_vowel(c)) {
                            Ok(Context::BeforeVowel(chars))
                        } else if chars.iter().all(|&c| !U::is_vowel(c)) {
                            Ok(Context::BeforeConsonant(chars))
                        } else {
                            // Mixed - use BeforeVowel as fallback
                            Ok(Context::BeforeVowel(chars))
                        }
                    }
                } else {
                    Err(LLevError::with_position(
                        LLevErrorKind::UnsupportedPattern(
                            "Complex pattern in context requires NFA-based matching".into(),
                        ),
                        pos,
                    ))
                }
            }
            ContextExpr::And(a, b) => {
                let left = self.convert_context_expr(a, pos, is_left_context)?;
                let right = self.convert_context_expr(b, pos, is_left_context)?;
                Ok(Context::And(Box::new(left), Box::new(right)))
            }
            ContextExpr::Or(a, b) => {
                let left = self.convert_context_expr(a, pos, is_left_context)?;
                let right = self.convert_context_expr(b, pos, is_left_context)?;
                Ok(Context::Or(Box::new(left), Box::new(right)))
            }
            ContextExpr::Not(inner) => {
                let converted = self.convert_context_expr(inner, pos, is_left_context)?;
                Ok(Context::Not(Box::new(converted)))
            }
        }
    }

    /// Extract character class from an Expression.
    ///
    /// Characters are lowercased for case-insensitive context matching.
    fn extract_char_class_from_expr(
        &self,
        expr: &Expression,
        pos: Position,
    ) -> LLevResult<Option<Vec<U>>> {
        match expr {
            Expression::CharClass { chars, negated } => {
                if *negated {
                    return Ok(None); // Negated classes not supported
                }
                let mut units = Vec::with_capacity(chars.len());
                for &c in chars {
                    let unit = U::from_char(c)
                        .ok_or_else(|| LLevError::non_ascii_in_byte_level(c, None, pos))?;
                    // Lowercase for case-insensitive matching
                    units.push(U::to_lowercase(unit));
                }
                Ok(Some(units))
            }

            Expression::CharRange { start, end } => {
                // Convert range to character class
                let mut units = Vec::new();
                for c in *start..=*end {
                    let unit = U::from_char(c)
                        .ok_or_else(|| LLevError::non_ascii_in_byte_level(c, None, pos))?;
                    // Lowercase for case-insensitive matching
                    units.push(U::to_lowercase(unit));
                }
                Ok(Some(units))
            }

            Expression::SymbolRef(name) => {
                if let Some(symbol_expr) = self.symbols.get(name.as_str()) {
                    self.extract_char_class_from_expr(symbol_expr, pos)
                } else {
                    // Use suggestion-aware error for better developer experience
                    Err(LLevError::undefined_symbol_with_suggestion(
                        name.clone(),
                        &self.symbol_names,
                        pos,
                    ))
                }
            }

            _ => Ok(None),
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if a character is an English vowel.
#[inline]
fn is_vowel_char(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U')
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::llev::parser::parse_str;

    #[test]
    fn test_ruleset_empty() {
        let ruleset = RuleSet::new();
        assert!(ruleset.is_empty());
        assert_eq!(ruleset.len(), 0);
    }

    #[test]
    fn test_ruleset_char_empty() {
        let ruleset = RuleSetChar::new();
        assert!(ruleset.is_empty());
        assert_eq!(ruleset.len(), 0);
    }

    #[test]
    fn test_simple_rule_conversion() {
        let file = parse_str("ph -> f;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert_eq!(rule.pattern.len(), 2);
        assert_eq!(rule.replacement.len(), 1);
        assert_eq!(rule.context, Context::Anywhere);
    }

    #[test]
    fn test_deletion_rule_conversion() {
        let file = parse_str("gh -> ;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert_eq!(rule.pattern.len(), 2);
        assert!(rule.replacement.is_empty());
    }

    #[test]
    fn test_apply_expands_multi_symbol_outputs() {
        let file = parse_str("x -> ks;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.apply("box"), "boks");
        assert_eq!(ruleset.apply_full("box"), "boks");
    }

    #[test]
    fn test_context_final_conversion() {
        let file = parse_str("e -> / _#;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert_eq!(rule.context, Context::Final);
    }

    #[test]
    fn test_context_initial_conversion() {
        let file = parse_str("k -> c / #_;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert_eq!(rule.context, Context::Initial);
    }

    #[test]
    fn test_context_before_vowel_conversion() {
        let file = parse_str("c -> s / _[ei];").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        match &rule.context {
            Context::BeforeVowel(chars) => {
                assert_eq!(chars.len(), 2);
                assert!(chars.contains(&'e'));
                assert!(chars.contains(&'i'));
            }
            _ => panic!("Expected BeforeVowel context"),
        }
    }

    #[test]
    fn test_context_after_vowel_conversion() {
        let file = parse_str("s -> z / [aeiou]_;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        match &rule.context {
            Context::AfterVowel(chars) => {
                assert_eq!(chars.len(), 5);
            }
            _ => panic!("Expected AfterVowel context"),
        }
    }

    #[test]
    fn test_rule_with_metadata() {
        let file = parse_str(
            r#"
            [id: 42, name: "soft c", weight: 0.5]
            c -> s / _[ei];
            "#,
        )
        .expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert_eq!(rule.rule_id, 42);
        assert_eq!(rule.rule_name, "soft c");
        assert_eq!(rule.weight, 0.5);
    }

    #[test]
    fn test_rule_with_inline_weight() {
        let file = parse_str("c -> s [0.3];").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert!((rule.weight - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_rule_with_context_and_inline_weight() {
        let file = parse_str("c -> s / _[ei] [0.25];").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert!((rule.weight - 0.25).abs() < 1e-10);
        assert!(matches!(rule.context, Context::BeforeVowel(_)));
    }

    #[test]
    fn test_inline_weight_overrides_metadata_weight() {
        let file = parse_str(
            r#"
            [weight: 0.9]
            c -> s [0.3];
            "#,
        )
        .expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert!((rule.weight - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_default_weight() {
        let file = parse_str("c -> s;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        // Default weight is 1.0
        assert_eq!(ruleset.rules[0].weight, 1.0);
    }

    #[test]
    fn test_disabled_rule_skipped() {
        let file = parse_str(
            r#"
            [enabled: false]
            ph -> f;
            gh -> f;
            "#,
        )
        .expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        // First rule is disabled, so only second rule is converted
        assert_eq!(ruleset.len(), 1);
    }

    #[test]
    fn test_symbol_expansion() {
        let file = parse_str(
            r#"
            @define FRONT = [ei]
            c -> s / _$FRONT;
            "#,
        )
        .expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        match &rule.context {
            Context::BeforeVowel(chars) => {
                assert_eq!(chars.len(), 2);
            }
            _ => panic!("Expected BeforeVowel context from symbol expansion"),
        }
    }

    #[test]
    fn test_undefined_symbol_error() {
        // Undefined symbols are now caught at parse time
        let result = parse_str("c -> s / _$UNDEFINED;");

        assert!(result.is_err());
        match result.unwrap_err().kind {
            LLevErrorKind::UndefinedSymbol(name) => {
                assert_eq!(name, "UNDEFINED");
            }
            _ => panic!("Expected UndefinedSymbol error"),
        }
    }

    #[test]
    fn test_undefined_symbol_with_suggestion() {
        // Define FRONT_VOWEL but use FRONTVOWEL (missing underscore - typo)
        // User-defined symbols require $ prefix
        // Undefined symbols are now caught at parse time with suggestions
        let result = parse_str(
            r#"
            @define FRONT_VOWEL = [ei]
            c -> s / _$FRONTVOWEL;
            "#,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();

        // Check that the error contains the suggestion
        let err_string = err.to_string();
        assert!(err_string.contains("undefined symbol: FRONTVOWEL"));
        assert!(err_string.contains("did you mean 'FRONT_VOWEL'?"));
    }

    #[test]
    fn test_undefined_symbol_typo_suggestion() {
        // Define CONSONANT but use CONSNANT (typo)
        // Undefined symbols are now caught at parse time with suggestions
        let result = parse_str(
            r#"
            @define CONSONANT = [bcdfghjklmnpqrstvwxyz]
            x -> y / _$CONSNANT;
            "#,
        );

        assert!(result.is_err());
        let err = result.unwrap_err();

        // Check that the error contains the suggestion
        let err_string = err.to_string();
        assert!(err_string.contains("undefined symbol: CONSNANT"));
        assert!(err_string.contains("did you mean 'CONSONANT'?"));
    }

    #[test]
    fn test_multiple_rules() {
        let file = parse_str(
            r#"
            ph -> f;
            gh -> ;
            c -> s / _[ei];
            "#,
        )
        .expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 3);
    }

    #[test]
    fn test_auto_id_assignment() {
        let file = parse_str(
            r#"
            ph -> f;
            gh -> ;
            "#,
        )
        .expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        // Auto-assigned IDs based on index
        assert_eq!(ruleset.rules[0].rule_id, 0);
        assert_eq!(ruleset.rules[1].rule_id, 1);
    }

    #[test]
    fn test_byte_level_conversion() {
        let file = parse_str("ph -> f;").expect("parse failed");
        let ruleset = RuleSet::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert_eq!(rule.pattern.len(), 2);
        assert_eq!(rule.pattern[0], Phone::Consonant(b'p'));
        assert_eq!(rule.pattern[1], Phone::Consonant(b'h'));
        assert_eq!(rule.replacement.len(), 1);
        assert_eq!(rule.replacement[0], Phone::Consonant(b'f'));
    }

    #[test]
    fn test_byte_level_non_ascii_error() {
        let file = parse_str("ü -> u;").expect("parse failed");
        let result = RuleSet::from_llev(&file);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err.kind {
            LLevErrorKind::NonAsciiInByteLevel { character, .. } => {
                assert_eq!(*character, 'ü');
            }
            _ => panic!("Expected NonAsciiInByteLevel error, got {:?}", err.kind),
        }

        // Check that the error message suggests RuleSetChar
        let err_string = err.to_string();
        assert!(err_string.contains("RuleSetChar"));
        assert!(err_string.contains("ü"));
    }

    #[test]
    fn test_char_level_unicode() {
        let file = parse_str("ü -> u;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert_eq!(rule.pattern.len(), 1);
        assert_eq!(rule.pattern[0], Phone::Consonant('ü'));
    }

    #[test]
    fn test_byte_level_non_ascii_in_context() {
        // Non-ASCII in context should also fail for byte-level
        let file = parse_str("a -> b / _[aäo];").expect("parse failed");
        let result = RuleSet::from_llev(&file);

        assert!(result.is_err());
        let err = result.unwrap_err();
        match &err.kind {
            LLevErrorKind::NonAsciiInByteLevel { character, .. } => {
                assert_eq!(*character, 'ä');
            }
            _ => panic!("Expected NonAsciiInByteLevel error, got {:?}", err.kind),
        }
    }

    #[test]
    fn test_unsupported_pattern_star() {
        let file = parse_str("a* -> b;").expect("parse failed");
        let result = RuleSetChar::from_llev(&file);

        assert!(result.is_err());
        match &result.unwrap_err().kind {
            LLevErrorKind::UnsupportedPattern(msg) => {
                assert!(msg.contains("Kleene star"));
            }
            _ => panic!("Expected UnsupportedPattern error"),
        }
    }

    #[test]
    fn test_unsupported_pattern_alternation() {
        let file = parse_str("(a|b) -> c;").expect("parse failed");
        let result = RuleSetChar::from_llev(&file);

        assert!(result.is_err());
        match &result.unwrap_err().kind {
            LLevErrorKind::UnsupportedPattern(msg) => {
                assert!(msg.contains("Alternation"));
            }
            _ => panic!("Expected UnsupportedPattern error"),
        }
    }

    #[test]
    fn test_ruleset_merge() {
        let file1 = parse_str("ph -> f;").expect("parse failed");
        let file2 = parse_str("gh -> ;").expect("parse failed");

        let mut ruleset1 = RuleSetChar::from_llev(&file1).expect("conversion failed");
        let ruleset2 = RuleSetChar::from_llev(&file2).expect("conversion failed");

        ruleset1.merge(ruleset2);
        assert_eq!(ruleset1.len(), 2);
    }

    #[test]
    fn test_metadata_preserved() {
        let file = parse_str(
            r#"
            @name "English Rules"
            @version "1.0"
            ph -> f;
            "#,
        )
        .expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.name, Some("English Rules".to_string()));
        assert_eq!(ruleset.version, Some("1.0".to_string()));
    }

    #[test]
    fn test_vowel_classification() {
        assert!(is_vowel_char('a'));
        assert!(is_vowel_char('E'));
        assert!(!is_vowel_char('z'));
    }

    #[test]
    fn test_empty_pattern_error() {
        let file = parse_str(" -> f;").expect("parse failed");
        let result = RuleSetChar::from_llev(&file);

        assert!(result.is_err());
        match &result.unwrap_err().kind {
            LLevErrorKind::InvalidRule(msg) => {
                assert!(msg.contains("empty"));
            }
            _ => panic!("Expected InvalidRule error"),
        }
    }

    // ========================================================================
    // Compound Context Tests
    // ========================================================================

    #[test]
    fn test_context_both_left_and_right() {
        // Test that rules with both left and right contexts work
        // x -> gz / [aeiou]_[aeiou] (voiced between vowels)
        let file = parse_str("x -> gz / [aeiou]_[aeiou];").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];

        // Should be And(AfterVowel, BeforeVowel)
        match &rule.context {
            Context::And(left, right) => {
                match left.as_ref() {
                    Context::AfterVowel(chars) => {
                        assert_eq!(chars.len(), 5); // a, e, i, o, u
                    }
                    _ => panic!("Expected AfterVowel on left, got {:?}", left),
                }
                match right.as_ref() {
                    Context::BeforeVowel(chars) => {
                        assert_eq!(chars.len(), 5); // a, e, i, o, u
                    }
                    _ => panic!("Expected BeforeVowel on right, got {:?}", right),
                }
            }
            _ => panic!("Expected And context, got {:?}", rule.context),
        }
    }

    #[test]
    fn test_context_word_boundary_both() {
        // Test that #_# works (whole word match)
        let file = parse_str("the -> ðə / #_#;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];

        // Should be And(Initial, Final)
        match &rule.context {
            Context::And(left, right) => {
                assert_eq!(left.as_ref(), &Context::Initial);
                assert_eq!(right.as_ref(), &Context::Final);
            }
            _ => panic!(
                "Expected And(Initial, Final) context, got {:?}",
                rule.context
            ),
        }
    }

    #[test]
    fn test_context_initial_with_char_class() {
        // Test #_[ei] - word-initial before front vowel
        let file = parse_str("c -> s / #_[ei];").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];

        // Should be And(Initial, BeforeVowel([e, i]))
        match &rule.context {
            Context::And(left, right) => {
                assert_eq!(left.as_ref(), &Context::Initial);
                match right.as_ref() {
                    Context::BeforeVowel(chars) => {
                        assert!(chars.contains(&'e'));
                        assert!(chars.contains(&'i'));
                    }
                    _ => panic!("Expected BeforeVowel on right, got {:?}", right),
                }
            }
            _ => panic!("Expected And context, got {:?}", rule.context),
        }
    }

    #[test]
    fn test_context_char_class_before_final() {
        // Test [aeiou]_# - after vowel at word end
        let file = parse_str("s -> z / [aeiou]_#;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];

        // Should be And(AfterVowel, Final)
        match &rule.context {
            Context::And(left, right) => {
                match left.as_ref() {
                    Context::AfterVowel(chars) => {
                        assert_eq!(chars.len(), 5);
                    }
                    _ => panic!("Expected AfterVowel on left, got {:?}", left),
                }
                assert_eq!(right.as_ref(), &Context::Final);
            }
            _ => panic!("Expected And context, got {:?}", rule.context),
        }
    }

    #[test]
    fn test_byte_level_compound_context() {
        // Test compound context for byte-level converter
        let file = parse_str("x -> gz / [aeiou]_[aeiou];").expect("parse failed");
        let ruleset = RuleSet::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];

        // Should be And(AfterVowel, BeforeVowel)
        match &rule.context {
            Context::And(left, right) => {
                match left.as_ref() {
                    Context::AfterVowel(chars) => {
                        assert_eq!(chars.len(), 5);
                    }
                    _ => panic!("Expected AfterVowel on left, got {:?}", left),
                }
                match right.as_ref() {
                    Context::BeforeVowel(chars) => {
                        assert_eq!(chars.len(), 5);
                    }
                    _ => panic!("Expected BeforeVowel on right, got {:?}", right),
                }
            }
            _ => panic!("Expected And context, got {:?}", rule.context),
        }
    }

    #[test]
    fn test_byte_level_word_boundary_both() {
        // Test #_# for byte-level
        let file = parse_str("a -> b / #_#;").expect("parse failed");
        let ruleset = RuleSet::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];

        match &rule.context {
            Context::And(left, right) => {
                assert_eq!(left.as_ref(), &Context::Initial);
                assert_eq!(right.as_ref(), &Context::Final);
            }
            _ => panic!(
                "Expected And(Initial, Final) context, got {:?}",
                rule.context
            ),
        }
    }
}
