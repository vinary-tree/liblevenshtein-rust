//! Runtime rule set conversion from `.llev` AST.
//!
//! This module converts parsed `.llev` files into runtime-usable rule sets.
//! It bridges the gap between the flexible AST representation and the
//! efficient runtime types used by the phonetic application functions.
//!
//! # Conversion Process
//!
//! 1. **Symbol Expansion**: `@define` symbols are expanded in patterns/contexts
//! 2. **Pattern Conversion**: AST expressions → `Vec<Phone>`/`Vec<PhoneChar>`
//! 3. **Context Mapping**: AST contexts → runtime `Context`/`ContextChar`
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
//! use liblevenshtein::phonetic::llev::{RuleSet, parse_str};
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

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

use super::ast::{ContextAST, ContextExpr, Expression, LLevFile, RuleDefinition, SymbolDef};
use super::error::{LLevError, LLevErrorKind, LLevResult, Position};
use crate::phonetic::types::{Context, ContextChar, Phone, PhoneChar, RewriteRule, RewriteRuleChar};

// ============================================================================
// Rule Set (Byte-level)
// ============================================================================

/// A collection of byte-level rewrite rules converted from a `.llev` file.
///
/// Use this for ASCII-only text where byte-level operations are sufficient.
/// For Unicode support, use [`RuleSetChar`] instead.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct RuleSet {
    /// Converted rules ready for application
    pub rules: Vec<RewriteRule>,

    /// Source file metadata (if available)
    pub name: Option<String>,
    pub version: Option<String>,
}

impl RuleSet {
    /// Create a new empty rule set.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            name: None,
            version: None,
        }
    }

    /// Convert a parsed `.llev` file into a byte-level rule set.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A pattern contains unsupported AST nodes (quantifiers, alternation, etc.)
    /// - A symbol reference cannot be resolved
    /// - A character cannot be converted to a single byte
    pub fn from_llev(file: &LLevFile) -> LLevResult<Self> {
        let converter = RuleConverter::new(&file.symbols);
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
    pub fn merge(&mut self, other: RuleSet) {
        self.rules.extend(other.rules);
    }
}

impl Default for RuleSet {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Rule Set (Character-level)
// ============================================================================

/// A collection of character-level rewrite rules converted from a `.llev` file.
///
/// Use this for Unicode text with accented characters, CJK, emoji, etc.
/// For ASCII-only text, [`RuleSet`] is slightly more efficient.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct RuleSetChar {
    /// Converted rules ready for application
    pub rules: Vec<RewriteRuleChar>,

    /// Source file metadata (if available)
    pub name: Option<String>,
    pub version: Option<String>,
}

impl RuleSetChar {
    /// Create a new empty rule set.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            name: None,
            version: None,
        }
    }

    /// Convert a parsed `.llev` file into a character-level rule set.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A pattern contains unsupported AST nodes (quantifiers, alternation, etc.)
    /// - A symbol reference cannot be resolved
    pub fn from_llev(file: &LLevFile) -> LLevResult<Self> {
        let converter = RuleConverterChar::new(&file.symbols);
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
    pub fn merge(&mut self, other: RuleSetChar) {
        self.rules.extend(other.rules);
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
        let phones: Vec<PhoneChar> = input.chars().map(|c| {
            // Lowercase for case-insensitive matching
            let c_lower = c.to_lowercase().next().unwrap_or(c);
            if is_vowel_char(c_lower) {
                PhoneChar::Vowel(c_lower)
            } else {
                PhoneChar::Consonant(c_lower)
            }
        }).collect();

        // Apply rules with generous fuel
        let fuel = input.len() * 10 + 100;
        if let Some(result) = crate::phonetic::apply_rules_seq_char(&self.rules, &phones, fuel) {
            // Convert back to string
            result.iter().filter_map(|p| match p {
                PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => Some(*c),
                PhoneChar::Digraph(_c1, _c2) => {
                    // For now, output both chars; could be refined
                    None // Skip digraphs in string output
                }
                PhoneChar::Trigraph(_c1, _c2, _c3) => {
                    // For now, skip trigraphs in string output
                    None
                }
                PhoneChar::Tetragraph(_c1, _c2, _c3, _c4) => {
                    // For now, skip tetragraphs in string output
                    None
                }
                PhoneChar::Pentagraph(_c1, _c2, _c3, _c4, _c5) => {
                    // For now, skip pentagraphs in string output
                    None
                }
                PhoneChar::Hexagraph(_c1, _c2, _c3, _c4, _c5, _c6) => {
                    // For now, skip hexagraphs in string output
                    None
                }
                PhoneChar::Heptagraph(_c1, _c2, _c3, _c4, _c5, _c6, _c7) => {
                    // For now, skip heptagraphs in string output
                    None
                }
                PhoneChar::Sequence(_) => {
                    // For now, skip sequences in string output
                    None
                }
                PhoneChar::Silent => None,
            }).collect()
        } else {
            // Fuel exhausted - return input unchanged
            input.to_string()
        }
    }

    /// Apply rules to a string with digraph support.
    ///
    /// Like [`apply`], but properly handles digraph output by expanding
    /// them to their constituent characters.
    ///
    /// Matching is case-insensitive by default (same as [`apply`]).
    pub fn apply_full(&self, input: &str) -> String {
        // Convert string to PhoneChar array with case folding for matching
        let phones: Vec<PhoneChar> = input.chars().map(|c| {
            // Lowercase for case-insensitive matching
            let c_lower = c.to_lowercase().next().unwrap_or(c);
            if is_vowel_char(c_lower) {
                PhoneChar::Vowel(c_lower)
            } else {
                PhoneChar::Consonant(c_lower)
            }
        }).collect();

        // Apply rules with generous fuel
        let fuel = input.len() * 10 + 100;
        if let Some(result) = crate::phonetic::apply_rules_seq_char(&self.rules, &phones, fuel) {
            // Convert back to string with digraph/trigraph/tetragraph/sequence expansion
            let mut output = String::with_capacity(result.len() * 4);
            for p in result.iter() {
                match p {
                    PhoneChar::Vowel(c) | PhoneChar::Consonant(c) => output.push(*c),
                    PhoneChar::Digraph(c1, c2) => {
                        output.push(*c1);
                        output.push(*c2);
                    }
                    PhoneChar::Trigraph(c1, c2, c3) => {
                        output.push(*c1);
                        output.push(*c2);
                        output.push(*c3);
                    }
                    PhoneChar::Tetragraph(c1, c2, c3, c4) => {
                        output.push(*c1);
                        output.push(*c2);
                        output.push(*c3);
                        output.push(*c4);
                    }
                    PhoneChar::Pentagraph(c1, c2, c3, c4, c5) => {
                        output.push(*c1);
                        output.push(*c2);
                        output.push(*c3);
                        output.push(*c4);
                        output.push(*c5);
                    }
                    PhoneChar::Hexagraph(c1, c2, c3, c4, c5, c6) => {
                        output.push(*c1);
                        output.push(*c2);
                        output.push(*c3);
                        output.push(*c4);
                        output.push(*c5);
                        output.push(*c6);
                    }
                    PhoneChar::Heptagraph(c1, c2, c3, c4, c5, c6, c7) => {
                        output.push(*c1);
                        output.push(*c2);
                        output.push(*c3);
                        output.push(*c4);
                        output.push(*c5);
                        output.push(*c6);
                        output.push(*c7);
                    }
                    PhoneChar::Sequence(s) => {
                        for c in s {
                            output.push(*c);
                        }
                    }
                    PhoneChar::Silent => {}
                }
            }
            output
        } else {
            // Fuel exhausted - return input unchanged
            input.to_string()
        }
    }
}

impl Default for RuleSetChar {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Byte-level Converter
// ============================================================================

/// Internal converter for byte-level rules.
struct RuleConverter<'a> {
    /// Symbol table for expanding symbol references
    symbols: HashMap<&'a str, &'a Expression>,
    /// List of symbol names for error suggestions
    symbol_names: Vec<&'a str>,
}

impl<'a> RuleConverter<'a> {
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
        }
    }

    /// Convert an AST rule definition to a runtime rule.
    ///
    /// Patterns and contexts are converted with case folding enabled by default
    /// for case-insensitive matching. Replacements preserve original case.
    fn convert_rule(&self, rule_def: &RuleDefinition, index: usize) -> LLevResult<RewriteRule> {
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
    ) -> LLevResult<Vec<Phone>> {
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

            // Unsupported patterns for byte-level conversion
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
    fn char_to_phone(&self, c: char, pos: Position, case_fold: bool) -> LLevResult<Phone> {
        // Apply case folding if requested (for ASCII only)
        let c = if case_fold && c.is_ascii() {
            c.to_ascii_lowercase()
        } else {
            c
        };

        // Check if character fits in a single byte
        if !c.is_ascii() {
            return Err(LLevError::non_ascii_in_byte_level(c, None, pos));
        }

        let b = c as u8;

        // Classify as vowel or consonant
        if is_vowel_byte(b) {
            Ok(Phone::Vowel(b))
        } else {
            Ok(Phone::Consonant(b))
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
    ) -> LLevResult<Context> {
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
    ) -> LLevResult<Context> {
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
                if let Some(chars) = self.extract_char_class_bytes_from_expr(expr, pos)? {
                    if is_left_context {
                        // Left context - "after" the characters
                        if chars.iter().all(|&c| is_vowel_byte(c)) {
                            Ok(Context::AfterVowel(chars))
                        } else if chars.iter().all(|&c| !is_vowel_byte(c)) {
                            Ok(Context::AfterConsonant(chars))
                        } else {
                            // Mixed - use AfterVowel as fallback
                            Ok(Context::AfterVowel(chars))
                        }
                    } else {
                        // Right context - "before" the characters
                        if chars.iter().all(|&c| is_vowel_byte(c)) {
                            Ok(Context::BeforeVowel(chars))
                        } else if chars.iter().all(|&c| !is_vowel_byte(c)) {
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

    /// Extract character class bytes from a context expression.
    ///
    /// Returns `Some(chars)` if the expression is a simple character class or symbol reference
    /// that resolves to a character class. Returns `None` if it's not a character class.
    fn extract_char_class_bytes(
        &self,
        ctx_expr: &ContextExpr,
        pos: Position,
    ) -> LLevResult<Option<Vec<u8>>> {
        match ctx_expr {
            ContextExpr::Pattern(expr) => self.extract_char_class_bytes_from_expr(expr, pos),
            ContextExpr::WordBoundary => Ok(None),
            ContextExpr::And(_, _) | ContextExpr::Or(_, _) | ContextExpr::Not(_) => {
                // Compound contexts are not supported for simple char class extraction
                Ok(None)
            }
        }
    }

    /// Extract character class bytes from an Expression.
    ///
    /// Characters are lowercased for case-insensitive context matching.
    fn extract_char_class_bytes_from_expr(
        &self,
        expr: &Expression,
        pos: Position,
    ) -> LLevResult<Option<Vec<u8>>> {
        match expr {
            Expression::CharClass { chars, negated } => {
                if *negated {
                    return Ok(None); // Negated classes not supported
                }
                let mut bytes = Vec::with_capacity(chars.len());
                for &c in chars {
                    if !c.is_ascii() {
                        return Err(LLevError::non_ascii_in_byte_level(c, None, pos));
                    }
                    // Lowercase for case-insensitive matching
                    bytes.push(c.to_ascii_lowercase() as u8);
                }
                Ok(Some(bytes))
            }

            Expression::CharRange { start, end } => {
                // Check both start and end for non-ASCII
                if !start.is_ascii() {
                    return Err(LLevError::non_ascii_in_byte_level(*start, None, pos));
                }
                if !end.is_ascii() {
                    return Err(LLevError::non_ascii_in_byte_level(*end, None, pos));
                }
                // Lowercase the range for case-insensitive matching
                let bytes: Vec<u8> = (*start as u8..=*end as u8)
                    .map(|b| (b as char).to_ascii_lowercase() as u8)
                    .collect();
                Ok(Some(bytes))
            }

            Expression::SymbolRef(name) => {
                if let Some(symbol_expr) = self.symbols.get(name.as_str()) {
                    self.extract_char_class_bytes_from_expr(symbol_expr, pos)
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
// Character-level Converter
// ============================================================================

/// Internal converter for character-level rules.
struct RuleConverterChar<'a> {
    /// Symbol table for expanding symbol references
    symbols: HashMap<&'a str, &'a Expression>,
    /// List of symbol names for error suggestions
    symbol_names: Vec<&'a str>,
}

impl<'a> RuleConverterChar<'a> {
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
        }
    }

    /// Convert an AST rule definition to a runtime rule.
    ///
    /// Patterns and contexts are converted with case folding enabled by default
    /// for case-insensitive matching. Replacements preserve original case.
    fn convert_rule(&self, rule_def: &RuleDefinition, index: usize) -> LLevResult<RewriteRuleChar> {
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

        Ok(RewriteRuleChar {
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
    ) -> LLevResult<Vec<PhoneChar>> {
        match expr {
            Expression::Empty => Ok(Vec::new()),

            Expression::Char(c) => {
                let phone = self.char_to_phone(*c, case_fold);
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

            // Unsupported patterns for character-level conversion
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

    /// Convert a character to a PhoneChar.
    ///
    /// # Arguments
    /// - `c`: The character to convert
    /// - `case_fold`: If true, convert to lowercase for case-insensitive matching
    fn char_to_phone(&self, c: char, case_fold: bool) -> PhoneChar {
        let c = if case_fold {
            // Unicode case folding - convert to lowercase for matching
            c.to_lowercase().next().unwrap_or(c)
        } else {
            c
        };

        // Classify as vowel or consonant
        if is_vowel_char(c) {
            PhoneChar::Vowel(c)
        } else {
            PhoneChar::Consonant(c)
        }
    }

    /// Convert an AST context to a runtime ContextChar.
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
    ) -> LLevResult<ContextChar> {
        let ctx = match ctx_opt {
            None => return Ok(ContextChar::Anywhere),
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
            (None, None) => Ok(ContextChar::Anywhere),
            (Some(left), None) => Ok(left),
            (None, Some(right)) => Ok(right),
            (Some(left), Some(right)) => {
                // Both left and right context - combine with And
                Ok(ContextChar::And(Box::new(left), Box::new(right)))
            }
        }
    }

    /// Extract character class chars from a context expression.
    fn extract_char_class(
        &self,
        ctx_expr: &ContextExpr,
        pos: Position,
    ) -> LLevResult<Option<Vec<char>>> {
        match ctx_expr {
            ContextExpr::Pattern(expr) => self.extract_char_class_from_expr(expr, pos),
            ContextExpr::WordBoundary => Ok(None),
            ContextExpr::And(_, _) | ContextExpr::Or(_, _) | ContextExpr::Not(_) => {
                // Compound contexts are handled by convert_context_expr, not char class extraction
                Ok(None)
            }
        }
    }

    /// Convert a context expression to a runtime ContextChar.
    ///
    /// This handles compound contexts (And, Or, Not) recursively.
    fn convert_context_expr(
        &self,
        ctx_expr: &ContextExpr,
        pos: Position,
        is_left_context: bool,
    ) -> LLevResult<ContextChar> {
        match ctx_expr {
            ContextExpr::WordBoundary => {
                if is_left_context {
                    Ok(ContextChar::Initial)
                } else {
                    Ok(ContextChar::Final)
                }
            }
            ContextExpr::Pattern(expr) => {
                // Try to extract as character class
                if let Some(chars) = self.extract_char_class_from_expr(expr, pos)? {
                    if is_left_context {
                        // Left context - "after" the characters
                        if chars.iter().all(|&c| is_vowel_char(c)) {
                            Ok(ContextChar::AfterVowel(chars))
                        } else if chars.iter().all(|&c| !is_vowel_char(c)) {
                            Ok(ContextChar::AfterConsonant(chars))
                        } else {
                            // Mixed - use AfterVowel as fallback
                            Ok(ContextChar::AfterVowel(chars))
                        }
                    } else {
                        // Right context - "before" the characters
                        if chars.iter().all(|&c| is_vowel_char(c)) {
                            Ok(ContextChar::BeforeVowel(chars))
                        } else if chars.iter().all(|&c| !is_vowel_char(c)) {
                            Ok(ContextChar::BeforeConsonant(chars))
                        } else {
                            // Mixed - use BeforeVowel as fallback
                            Ok(ContextChar::BeforeVowel(chars))
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
                Ok(ContextChar::And(Box::new(left), Box::new(right)))
            }
            ContextExpr::Or(a, b) => {
                let left = self.convert_context_expr(a, pos, is_left_context)?;
                let right = self.convert_context_expr(b, pos, is_left_context)?;
                Ok(ContextChar::Or(Box::new(left), Box::new(right)))
            }
            ContextExpr::Not(inner) => {
                let converted = self.convert_context_expr(inner, pos, is_left_context)?;
                Ok(ContextChar::Not(Box::new(converted)))
            }
        }
    }

    /// Extract character class chars from an Expression.
    ///
    /// Characters are lowercased for case-insensitive context matching.
    fn extract_char_class_from_expr(
        &self,
        expr: &Expression,
        pos: Position,
    ) -> LLevResult<Option<Vec<char>>> {
        match expr {
            Expression::CharClass { chars, negated } => {
                if *negated {
                    return Ok(None); // Negated classes not supported
                }
                // Lowercase chars for case-insensitive matching
                let lowered: Vec<char> = chars
                    .iter()
                    .map(|c| c.to_lowercase().next().unwrap_or(*c))
                    .collect();
                Ok(Some(lowered))
            }

            Expression::CharRange { start, end } => {
                // Lowercase the range for case-insensitive matching
                let chars: Vec<char> = (*start..=*end)
                    .map(|c| c.to_lowercase().next().unwrap_or(c))
                    .collect();
                Ok(Some(chars))
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

/// Check if a byte is an English vowel.
#[inline]
fn is_vowel_byte(b: u8) -> bool {
    matches!(
        b,
        b'a' | b'e' | b'i' | b'o' | b'u' | b'A' | b'E' | b'I' | b'O' | b'U'
    )
}

/// Check if a character is an English vowel.
#[inline]
fn is_vowel_char(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U'
    )
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
        assert_eq!(rule.context, ContextChar::Anywhere);
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
    fn test_context_final_conversion() {
        let file = parse_str("e -> / _#;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert_eq!(rule.context, ContextChar::Final);
    }

    #[test]
    fn test_context_initial_conversion() {
        let file = parse_str("k -> c / #_;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        assert_eq!(rule.context, ContextChar::Initial);
    }

    #[test]
    fn test_context_before_vowel_conversion() {
        let file = parse_str("c -> s / _[ei];").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        assert_eq!(ruleset.len(), 1);
        let rule = &ruleset.rules[0];
        match &rule.context {
            ContextChar::BeforeVowel(chars) => {
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
            ContextChar::AfterVowel(chars) => {
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
    fn test_rule_with_metadata_weight() {
        // Note: Inline weight syntax `[0.3]` after replacement is not yet implemented
        // in the parser. For now, weights must be specified in metadata blocks.
        let file = parse_str(
            r#"
            [weight: 0.3]
            c -> s;
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
            ContextChar::BeforeVowel(chars) => {
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
        assert_eq!(rule.pattern[0], PhoneChar::Consonant('ü'));
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
        assert!(is_vowel_byte(b'a'));
        assert!(is_vowel_byte(b'e'));
        assert!(is_vowel_byte(b'i'));
        assert!(is_vowel_byte(b'o'));
        assert!(is_vowel_byte(b'u'));
        assert!(is_vowel_byte(b'A'));
        assert!(!is_vowel_byte(b'b'));
        assert!(!is_vowel_byte(b'c'));

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
            ContextChar::And(left, right) => {
                match left.as_ref() {
                    ContextChar::AfterVowel(chars) => {
                        assert_eq!(chars.len(), 5); // a, e, i, o, u
                    }
                    _ => panic!("Expected AfterVowel on left, got {:?}", left),
                }
                match right.as_ref() {
                    ContextChar::BeforeVowel(chars) => {
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
            ContextChar::And(left, right) => {
                assert_eq!(left.as_ref(), &ContextChar::Initial);
                assert_eq!(right.as_ref(), &ContextChar::Final);
            }
            _ => panic!("Expected And(Initial, Final) context, got {:?}", rule.context),
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
            ContextChar::And(left, right) => {
                assert_eq!(left.as_ref(), &ContextChar::Initial);
                match right.as_ref() {
                    ContextChar::BeforeVowel(chars) => {
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
            ContextChar::And(left, right) => {
                match left.as_ref() {
                    ContextChar::AfterVowel(chars) => {
                        assert_eq!(chars.len(), 5);
                    }
                    _ => panic!("Expected AfterVowel on left, got {:?}", left),
                }
                assert_eq!(right.as_ref(), &ContextChar::Final);
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
            _ => panic!("Expected And(Initial, Final) context, got {:?}", rule.context),
        }
    }
}
