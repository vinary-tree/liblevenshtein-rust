//! NFA compiler for phonetic regular expressions.
//!
//! This module compiles parsed regex ASTs into NFAs using Thompson's construction.
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::nfa::compiler::compile;
//! use liblevenshtein::phonetic::regex::parse;
//!
//! // Compile a simple pattern
//! let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();
//! assert!(nfa.accepts("phone"));
//! assert!(nfa.accepts("fone"));
//! assert!(!nfa.accepts("bone"));
//!
//! // Compile a rewrite rule
//! let rule = parse_rule("ph -> f").unwrap();
//! let rewrite = compile_rewrite(&rule).unwrap();
//! ```

use super::context::{BoundaryKind, ContextPattern, ContextPatternChar};
use super::nfa::{NFAChar, NFA};
use super::thompson::{ThompsonBuilder, ThompsonBuilderChar};
use crate::phonetic::regex::ast::{ContextExpr, ContextExprByte, Regex, RegexByte};
use crate::phonetic::regex::error::{ParseError, ParseErrorKind, ParseResult, Position};

/// Compile a character-level regex to an NFA.
///
/// # Arguments
///
/// * `regex` - The regex AST to compile
///
/// # Returns
///
/// An NFA that accepts the language defined by the regex.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::phonetic::regex::parse;
/// use liblevenshtein::phonetic::nfa::compiler::compile;
///
/// let regex = parse("a+b*c").unwrap();
/// let nfa = compile(&regex).unwrap();
/// assert!(nfa.accepts("ac"));
/// assert!(nfa.accepts("aaac"));
/// assert!(nfa.accepts("abbc"));
/// ```
pub fn compile(regex: &Regex) -> ParseResult<NFAChar> {
    let mut compiler = NFACompilerChar::new();
    compiler.compile(regex)
}

/// Compile a byte-level regex to an NFA.
pub fn compile_bytes(regex: &RegexByte) -> ParseResult<NFA> {
    let mut compiler = NFACompilerByte::new();
    compiler.compile(regex)
}

/// A compiled rewrite rule.
///
/// Contains the source pattern NFA, replacement string, and optional context.
#[derive(Debug, Clone)]
pub struct CompiledRewriteChar {
    /// NFA matching the source pattern
    pub source: NFAChar,
    /// Characters to replace with
    pub replacement: Vec<char>,
    /// Optional left context (lookbehind)
    pub left_context: Option<ContextPatternChar>,
    /// Optional right context (lookahead)
    pub right_context: Option<ContextPatternChar>,
    /// Weight/cost for this rule
    pub weight: f64,
}

/// A compiled byte-level rewrite rule.
#[derive(Debug, Clone)]
pub struct CompiledRewrite {
    /// NFA matching the source pattern
    pub source: NFA,
    /// Bytes to replace with
    pub replacement: Vec<u8>,
    /// Optional left context (lookbehind)
    pub left_context: Option<ContextPattern>,
    /// Optional right context (lookahead)
    pub right_context: Option<ContextPattern>,
    /// Weight/cost for this rule
    pub weight: f64,
}

/// Compile a rewrite rule regex to a compiled rewrite structure.
pub fn compile_rewrite(regex: &Regex) -> ParseResult<CompiledRewriteChar> {
    let mut compiler = NFACompilerChar::new();
    compiler.compile_rewrite(regex)
}

/// Compile a byte-level rewrite rule.
pub fn compile_rewrite_bytes(regex: &RegexByte) -> ParseResult<CompiledRewrite> {
    let mut compiler = NFACompilerByte::new();
    compiler.compile_rewrite(regex)
}

/// Character-level NFA compiler.
pub struct NFACompilerChar {
    builder: ThompsonBuilderChar,
}

impl NFACompilerChar {
    /// Create a new compiler.
    pub fn new() -> Self {
        Self {
            builder: ThompsonBuilderChar::new(),
        }
    }

    /// Compile a regex AST to an NFA.
    pub fn compile(&mut self, regex: &Regex) -> ParseResult<NFAChar> {
        self.compile_regex(regex)
    }

    /// Compile a rewrite rule.
    pub fn compile_rewrite(&mut self, regex: &Regex) -> ParseResult<CompiledRewriteChar> {
        match regex {
            Regex::RewriteRule {
                pattern,
                replacement,
                context,
                weight,
            } => {
                let source = self.compile_regex(pattern)?;
                let replacement_chars = self.regex_to_literal(replacement)?;

                let (left_context, right_context) = if let Some(ctx) = context {
                    (
                        ctx.left
                            .as_ref()
                            .map(|l| self.compile_context_expr(l))
                            .transpose()?,
                        ctx.right
                            .as_ref()
                            .map(|r| self.compile_context_expr(r))
                            .transpose()?,
                    )
                } else {
                    (None, None)
                };

                Ok(CompiledRewriteChar {
                    source,
                    replacement: replacement_chars,
                    left_context,
                    right_context,
                    weight: *weight,
                })
            }
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidRewriteRule("expected rewrite rule".to_string()),
                Position::start(),
            )),
        }
    }

    /// Compile a context expression to a context pattern.
    fn compile_context_expr(&mut self, expr: &ContextExpr) -> ParseResult<ContextPatternChar> {
        match expr {
            ContextExpr::Pattern(regex) => {
                let nfa = self.compile_regex(regex)?;
                Ok(ContextPatternChar::Nfa(nfa))
            }
            ContextExpr::WordBoundary => {
                // Word boundary is represented as a boundary kind
                // The actual direction (start/end) is determined by position in the context
                Ok(ContextPatternChar::Boundary(BoundaryKind::WordStart))
            }
            ContextExpr::And(a, b) => {
                let left = self.compile_context_expr(a)?;
                let right = self.compile_context_expr(b)?;
                Ok(ContextPatternChar::And(Box::new(left), Box::new(right)))
            }
            ContextExpr::Or(a, b) => {
                let left = self.compile_context_expr(a)?;
                let right = self.compile_context_expr(b)?;
                Ok(ContextPatternChar::Or(Box::new(left), Box::new(right)))
            }
            ContextExpr::Not(inner) => {
                let pattern = self.compile_context_expr(inner)?;
                Ok(ContextPatternChar::Not(Box::new(pattern)))
            }
        }
    }

    /// Compile a regex to NFA.
    fn compile_regex(&mut self, regex: &Regex) -> ParseResult<NFAChar> {
        match regex {
            Regex::Empty => Ok(self.builder.epsilon()),
            Regex::Char(c) => Ok(self.builder.single_char(*c)),
            Regex::CharClass(class) => Ok(self.builder.char_class(class.clone())),
            Regex::Any => Ok(self.builder.any_char()),
            Regex::Concat(a, b) => {
                let nfa_a = self.compile_regex(a)?;
                let nfa_b = self.compile_regex(b)?;
                Ok(self.builder.concatenate(nfa_a, nfa_b))
            }
            Regex::Alt(a, b) => {
                let nfa_a = self.compile_regex(a)?;
                let nfa_b = self.compile_regex(b)?;
                Ok(self.builder.alternation(nfa_a, nfa_b))
            }
            Regex::Star(inner) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.kleene_star(nfa))
            }
            Regex::Plus(inner) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.kleene_plus(nfa))
            }
            Regex::Optional(inner) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.optional(nfa))
            }
            Regex::RepeatExact(inner, n) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.repeat_exact(nfa, *n))
            }
            Regex::RepeatRange(inner, min, max) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.repeat_range(nfa, *min, *max))
            }
            Regex::Group(inner) => {
                // Groups don't affect NFA structure (capturing is not implemented yet)
                self.compile_regex(inner)
            }
            Regex::WordBoundary => {
                // Word boundary is handled specially in context matching
                // For now, return epsilon (will be handled by context logic)
                Ok(self.builder.epsilon())
            }
            Regex::RewriteRule { pattern, .. } => {
                // When compiling a rewrite rule as a pattern, just compile the pattern part
                self.compile_regex(pattern)
            }
        }
    }

    /// Convert a regex to a literal string (for replacement).
    fn regex_to_literal(&self, regex: &Regex) -> ParseResult<Vec<char>> {
        match regex {
            Regex::Empty => Ok(Vec::new()),
            Regex::Char(c) => Ok(vec![*c]),
            Regex::Concat(a, b) => {
                let mut chars = self.regex_to_literal(a)?;
                chars.extend(self.regex_to_literal(b)?);
                Ok(chars)
            }
            Regex::Group(inner) => self.regex_to_literal(inner),
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidRewriteRule(
                    "replacement must be a literal string".to_string(),
                ),
                Position::start(),
            )),
        }
    }
}

impl Default for NFACompilerChar {
    fn default() -> Self {
        Self::new()
    }
}

/// Byte-level NFA compiler.
pub struct NFACompilerByte {
    builder: ThompsonBuilder,
}

impl NFACompilerByte {
    /// Create a new compiler.
    pub fn new() -> Self {
        Self {
            builder: ThompsonBuilder::new(),
        }
    }

    /// Compile a regex AST to an NFA.
    pub fn compile(&mut self, regex: &RegexByte) -> ParseResult<NFA> {
        self.compile_regex(regex)
    }

    /// Compile a rewrite rule.
    pub fn compile_rewrite(&mut self, regex: &RegexByte) -> ParseResult<CompiledRewrite> {
        match regex {
            RegexByte::RewriteRule {
                pattern,
                replacement,
                context,
                weight,
            } => {
                let source = self.compile_regex(pattern)?;
                let replacement_bytes = self.regex_to_literal(replacement)?;

                let (left_context, right_context) = if let Some(ctx) = context {
                    (
                        ctx.left
                            .as_ref()
                            .map(|l| self.compile_context_expr(l))
                            .transpose()?,
                        ctx.right
                            .as_ref()
                            .map(|r| self.compile_context_expr(r))
                            .transpose()?,
                    )
                } else {
                    (None, None)
                };

                Ok(CompiledRewrite {
                    source,
                    replacement: replacement_bytes,
                    left_context,
                    right_context,
                    weight: *weight,
                })
            }
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidRewriteRule("expected rewrite rule".to_string()),
                Position::start(),
            )),
        }
    }

    /// Compile a context expression to a context pattern.
    fn compile_context_expr(&mut self, expr: &ContextExprByte) -> ParseResult<ContextPattern> {
        match expr {
            ContextExprByte::Pattern(regex) => {
                let nfa = self.compile_regex(regex)?;
                Ok(ContextPattern::Nfa(nfa))
            }
            ContextExprByte::WordBoundary => {
                Ok(ContextPattern::Boundary(BoundaryKind::WordStart))
            }
            ContextExprByte::And(a, b) => {
                let left = self.compile_context_expr(a)?;
                let right = self.compile_context_expr(b)?;
                Ok(ContextPattern::And(Box::new(left), Box::new(right)))
            }
            ContextExprByte::Or(a, b) => {
                let left = self.compile_context_expr(a)?;
                let right = self.compile_context_expr(b)?;
                Ok(ContextPattern::Or(Box::new(left), Box::new(right)))
            }
            ContextExprByte::Not(inner) => {
                let pattern = self.compile_context_expr(inner)?;
                Ok(ContextPattern::Not(Box::new(pattern)))
            }
        }
    }

    /// Compile a regex to NFA.
    fn compile_regex(&mut self, regex: &RegexByte) -> ParseResult<NFA> {
        match regex {
            RegexByte::Empty => Ok(self.builder.epsilon()),
            RegexByte::Byte(b) => Ok(self.builder.single_byte(*b)),
            RegexByte::ByteClass(class) => Ok(self.builder.byte_class(class.clone())),
            RegexByte::Any => Ok(self.builder.any_byte()),
            RegexByte::Concat(a, b) => {
                let nfa_a = self.compile_regex(a)?;
                let nfa_b = self.compile_regex(b)?;
                Ok(self.builder.concatenate(nfa_a, nfa_b))
            }
            RegexByte::Alt(a, b) => {
                let nfa_a = self.compile_regex(a)?;
                let nfa_b = self.compile_regex(b)?;
                Ok(self.builder.alternation(nfa_a, nfa_b))
            }
            RegexByte::Star(inner) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.kleene_star(nfa))
            }
            RegexByte::Plus(inner) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.kleene_plus(nfa))
            }
            RegexByte::Optional(inner) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.optional(nfa))
            }
            RegexByte::RepeatExact(inner, n) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.repeat_exact(nfa, *n))
            }
            RegexByte::RepeatRange(inner, min, max) => {
                let nfa = self.compile_regex(inner)?;
                Ok(self.builder.repeat_range(nfa, *min, *max))
            }
            RegexByte::Group(inner) => {
                self.compile_regex(inner)
            }
            RegexByte::WordBoundary => {
                Ok(self.builder.epsilon())
            }
            RegexByte::RewriteRule { pattern, .. } => {
                self.compile_regex(pattern)
            }
        }
    }

    /// Convert a regex to a literal byte string (for replacement).
    fn regex_to_literal(&self, regex: &RegexByte) -> ParseResult<Vec<u8>> {
        match regex {
            RegexByte::Empty => Ok(Vec::new()),
            RegexByte::Byte(b) => Ok(vec![*b]),
            RegexByte::Concat(a, b) => {
                let mut bytes = self.regex_to_literal(a)?;
                bytes.extend(self.regex_to_literal(b)?);
                Ok(bytes)
            }
            RegexByte::Group(inner) => self.regex_to_literal(inner),
            _ => Err(ParseError::new(
                ParseErrorKind::InvalidRewriteRule(
                    "replacement must be a literal string".to_string(),
                ),
                Position::start(),
            )),
        }
    }
}

impl Default for NFACompilerByte {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::regex::{parse, parse_rule};

    #[test]
    fn test_compile_literal() {
        let regex = parse("phone").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("phone"));
        assert!(!nfa.accepts("fone"));
    }

    #[test]
    fn test_compile_alternation() {
        let regex = parse("ph|f").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("ph"));
        assert!(nfa.accepts("f"));
        assert!(!nfa.accepts("g"));
    }

    #[test]
    fn test_compile_group() {
        let regex = parse("(ph|f)one").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("phone"));
        assert!(nfa.accepts("fone"));
        assert!(!nfa.accepts("bone"));
    }

    #[test]
    fn test_compile_star() {
        let regex = parse("a*").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("aaa"));
        assert!(!nfa.accepts("b"));
    }

    #[test]
    fn test_compile_plus() {
        let regex = parse("a+").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(!nfa.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("aaa"));
    }

    #[test]
    fn test_compile_optional() {
        let regex = parse("a?b").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("b"));
        assert!(nfa.accepts("ab"));
        assert!(!nfa.accepts("aab"));
    }

    #[test]
    fn test_compile_char_class() {
        let regex = parse("[aeiou]").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("e"));
        assert!(!nfa.accepts("b"));
    }

    #[test]
    fn test_compile_any() {
        let regex = parse("a.c").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("abc"));
        assert!(nfa.accepts("axc"));
        assert!(!nfa.accepts("ac"));
    }

    #[test]
    fn test_compile_repeat_exact() {
        let regex = parse("a{3}").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(!nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(!nfa.accepts("aaaa"));
    }

    #[test]
    fn test_compile_repeat_range() {
        let regex = parse("a{2,4}").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(!nfa.accepts("a"));
        assert!(nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(nfa.accepts("aaaa"));
        assert!(!nfa.accepts("aaaaa"));
    }

    #[test]
    fn test_compile_rewrite_rule_simple() {
        let regex = parse_rule("ph -> f").unwrap();
        let rewrite = compile_rewrite(&regex).unwrap();
        assert!(rewrite.source.accepts("ph"));
        assert_eq!(rewrite.replacement, vec!['f']);
        assert!(rewrite.left_context.is_none());
        assert!(rewrite.right_context.is_none());
    }

    #[test]
    fn test_compile_rewrite_rule_with_context() {
        let regex = parse_rule("c -> s / _[ei]").unwrap();
        let rewrite = compile_rewrite(&regex).unwrap();
        assert!(rewrite.source.accepts("c"));
        assert_eq!(rewrite.replacement, vec!['s']);
        assert!(rewrite.left_context.is_none());
        assert!(rewrite.right_context.is_some());
        let right = rewrite.right_context.unwrap();
        assert!(right.accepts("e"));
        assert!(right.accepts("i"));
        assert!(!right.accepts("a"));
    }

    #[test]
    fn test_compile_rewrite_rule_empty_replacement() {
        let regex = parse_rule("e -> / _#").unwrap();
        let rewrite = compile_rewrite(&regex).unwrap();
        assert!(rewrite.source.accepts("e"));
        assert!(rewrite.replacement.is_empty());
    }

    #[test]
    fn test_compile_complex_pattern() {
        let regex = parse("(ph|f)one[s]?").unwrap();
        let nfa = compile(&regex).unwrap();
        assert!(nfa.accepts("phone"));
        assert!(nfa.accepts("phones"));
        assert!(nfa.accepts("fone"));
        assert!(nfa.accepts("fones"));
        assert!(!nfa.accepts("bone"));
    }

    // Byte-level tests

    #[test]
    fn test_compile_bytes_literal() {
        let regex = crate::phonetic::regex::parse_bytes(b"phone").unwrap();
        let nfa = compile_bytes(&regex).unwrap();
        assert!(nfa.accepts(b"phone"));
        assert!(!nfa.accepts(b"fone"));
    }

    #[test]
    fn test_compile_bytes_alternation() {
        let regex = crate::phonetic::regex::parse_bytes(b"ph|f").unwrap();
        let nfa = compile_bytes(&regex).unwrap();
        assert!(nfa.accepts(b"ph"));
        assert!(nfa.accepts(b"f"));
    }

    #[test]
    fn test_compile_bytes_rewrite() {
        let regex = crate::phonetic::regex::parse_rule_bytes(b"ph -> f").unwrap();
        let rewrite = compile_rewrite_bytes(&regex).unwrap();
        assert!(rewrite.source.accepts(b"ph"));
        assert_eq!(rewrite.replacement, vec![b'f']);
    }
}
