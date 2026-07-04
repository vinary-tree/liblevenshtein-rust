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
//! let nfa = compile(&parse("(ph|f)one").expect("doc: regex parse must succeed")).expect("doc: nfa compile must succeed");
//! assert!(nfa.accepts("phone"));
//! assert!(nfa.accepts("fone"));
//! assert!(!nfa.accepts("bone"));
//!
//! // Compile a rewrite rule
//! let rule = parse_rule("ph -> f").expect("doc: rewrite rule parse must succeed");
//! let rewrite = compile_rewrite(&rule).expect("doc: rewrite compile must succeed");
//! ```

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use super::context::{BoundaryKind, ContextPattern, ContextPatternChar};
use super::optimizer::{NfaOptimizerChar, OptimizationConfig};
use super::thompson::{ThompsonBuilder, ThompsonBuilderChar};
use super::{NFAChar, NFA};
use crate::phonetic::regex::ast::{
    ContextExpr, ContextExprByte, Regex, RegexByte, RegexFlags, UnicodeNormalization,
};
use crate::phonetic::regex::error::{ParseError, ParseErrorKind, ParseResult, Position};
use crate::phonetic::regex::transform::apply_flags;

const MAX_GROUP_EXPANSION_DEPTH: usize = 100;

#[derive(Debug, Default, Clone, Copy)]
struct GroupReferenceStats {
    named_groups: usize,
    group_refs: usize,
}

impl GroupReferenceStats {
    fn has_groups(self) -> bool {
        self.named_groups > 0 || self.group_refs > 0
    }

    fn needs_expansion(self) -> bool {
        self.group_refs > 0
    }

    fn stack_capacity(self) -> usize {
        self.group_refs.min(MAX_GROUP_EXPANSION_DEPTH + 1)
    }
}

fn invalid_literal_replacement_error() -> ParseError {
    ParseError::new(
        ParseErrorKind::InvalidRewriteRule("replacement must be a literal string".to_string()),
        Position::start(),
    )
}

fn replacement_literal_too_long_error() -> ParseError {
    ParseError::new(
        ParseErrorKind::InvalidRewriteRule("replacement literal is too long".to_string()),
        Position::start(),
    )
}

fn increment_literal_len(len: &mut usize) -> ParseResult<()> {
    *len = len
        .checked_add(1)
        .ok_or_else(replacement_literal_too_long_error)?;
    Ok(())
}

/// Result of compiling a regex with flag support.
///
/// Contains both the compiled NFA and runtime settings extracted from flags.
#[derive(Debug, Clone)]
pub struct CompileResultChar {
    /// The compiled NFA.
    pub nfa: NFAChar,
    /// Unicode normalization to apply to input at runtime.
    pub unicode_normalization: Option<UnicodeNormalization>,
    /// Whether multiline mode is enabled (for `^` and `$`).
    pub multiline: bool,
    /// Whether dotall mode is enabled (`.` matches newlines).
    pub dotall: bool,
    /// Local Levenshtein distance override (from `(?;N)` syntax).
    ///
    /// When set, this distance limit should be used for fuzzy matching
    /// instead of the global distance parameter.
    pub local_distance: Option<u8>,
}

/// Compile a character-level regex to an NFA.
///
/// This applies regex flag transformations (like `(?i)` case-insensitive)
/// before compilation. For flags that affect runtime behavior (unicode
/// normalization, multiline, dotall), use [`compile_with_flags`] instead
/// to access those settings.
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
/// let regex = parse("a+b*c").expect("doc: regex parse must succeed");
/// let nfa = compile(&regex).expect("doc: nfa compile must succeed");
/// assert!(nfa.accepts("ac"));
/// assert!(nfa.accepts("aaac"));
/// assert!(nfa.accepts("abbc"));
///
/// // Case-insensitive matching via (?i) flag
/// let regex = parse("(?i:hello)").expect("doc: regex parse must succeed");
/// let nfa = compile(&regex).expect("doc: nfa compile must succeed");
/// assert!(nfa.accepts("hello"));
/// assert!(nfa.accepts("HELLO"));
/// assert!(nfa.accepts("HeLLo"));
/// ```
pub fn compile(regex: &Regex) -> ParseResult<NFAChar> {
    let mut compiler = NFACompilerChar::new();
    compiler.compile(regex)
}

/// Compile a character-level regex to an NFA with full flag support.
///
/// This applies regex flag transformations and returns both the NFA and
/// runtime settings extracted from flags like `(?u:NFC)`, `(?m)`, and `(?s)`.
///
/// # Arguments
///
/// * `regex` - The regex AST to compile
///
/// # Returns
///
/// A [`CompileResultChar`] containing the NFA and runtime settings.
///
/// # Examples
///
/// ```ignore
/// use liblevenshtein::phonetic::regex::parse;
/// use liblevenshtein::phonetic::nfa::compiler::compile_with_flags;
///
/// let regex = parse("(?iu:café)").expect("doc: regex parse must succeed");
/// let result = compile_with_flags(&regex).expect("doc: compile_with_flags must succeed");
///
/// // NFA matches case-insensitively (transformed)
/// assert!(result.nfa.accepts("café"));
/// assert!(result.nfa.accepts("CAFÉ"));
///
/// // Unicode normalization available for runtime use
/// if let Some(norm) = result.unicode_normalization {
///     // Apply normalization to input before matching
/// }
/// ```
pub fn compile_with_flags(regex: &Regex) -> ParseResult<CompileResultChar> {
    let mut compiler = NFACompilerChar::new();
    compiler.compile_with_flags(regex)
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

// ============================================================================
// Trampolining support for stack-safe compilation
// ============================================================================

/// Work item for trampolined regex compilation.
///
/// This enum represents the operations that need to be performed during
/// iterative (trampolined) compilation. Using an explicit work stack instead
/// of the call stack allows compilation of deeply nested patterns without
/// risk of stack overflow.
#[derive(Debug)]
enum CompileWork<'a> {
    /// Compile this regex node and push result to value stack.
    Compile(&'a Regex),
    /// Pop two values, concatenate them, push result.
    DoConcatenate,
    /// Pop two values, create alternation, push result.
    DoAlternation,
    /// Pop one value, apply Kleene star, push result.
    DoStar,
    /// Pop one value, apply Kleene plus, push result.
    DoPlus,
    /// Pop one value, apply optional, push result.
    DoOptional,
    /// Pop one value, apply exact repetition, push result.
    DoRepeatExact(usize),
    /// Pop one value, apply range repetition, push result.
    DoRepeatRange(usize, Option<usize>),
}

fn prepare_regex_for_compile(regex: &Regex) -> ParseResult<Cow<'_, Regex>> {
    let stats = collect_group_reference_stats(regex);
    if !stats.has_groups() {
        return Ok(Cow::Borrowed(regex));
    }

    if !stats.needs_expansion() {
        let mut seen = HashSet::with_capacity(stats.named_groups);
        validate_unique_named_groups(regex, &mut seen)?;
        return Ok(Cow::Borrowed(regex));
    }

    let mut named_groups = HashMap::with_capacity(stats.named_groups);
    collect_named_groups(regex, &mut named_groups)?;
    let mut stack = Vec::with_capacity(stats.stack_capacity());
    expand_group_refs(regex, &named_groups, &mut stack, 0).map(Cow::Owned)
}

fn collect_group_reference_stats(regex: &Regex) -> GroupReferenceStats {
    let mut stats = GroupReferenceStats::default();
    collect_group_reference_stats_into(regex, &mut stats);
    stats
}

fn collect_group_reference_stats_into(regex: &Regex, stats: &mut GroupReferenceStats) {
    match regex {
        Regex::Concat(a, b) | Regex::Alt(a, b) => {
            collect_group_reference_stats_into(a, stats);
            collect_group_reference_stats_into(b, stats);
        }
        Regex::Star(inner)
        | Regex::Plus(inner)
        | Regex::Optional(inner)
        | Regex::RepeatExact(inner, _)
        | Regex::RepeatRange(inner, _, _)
        | Regex::CapturingGroup(_, inner)
        | Regex::NonCapturingGroup(inner) => {
            collect_group_reference_stats_into(inner, stats);
        }
        Regex::NamedGroup(_, inner) => {
            stats.named_groups += 1;
            collect_group_reference_stats_into(inner, stats);
        }
        Regex::GroupRef(_) => {
            stats.group_refs += 1;
        }
        Regex::FlagsGroup {
            inner: Some(inner), ..
        } => {
            collect_group_reference_stats_into(inner, stats);
        }
        Regex::RewriteRule {
            pattern,
            replacement,
            context,
            ..
        } => {
            collect_group_reference_stats_into(pattern, stats);
            collect_group_reference_stats_into(replacement, stats);
            if let Some(context) = context {
                collect_group_reference_stats_context(context.left.as_ref(), stats);
                collect_group_reference_stats_context(context.right.as_ref(), stats);
            }
        }
        Regex::Empty
        | Regex::Char(_)
        | Regex::CharClass(_)
        | Regex::Any
        | Regex::FlagsGroup { inner: None, .. }
        | Regex::WordBoundary
        | Regex::StartOfLine
        | Regex::EndOfLine
        | Regex::StartOfInput
        | Regex::EndOfInput
        | Regex::EndOfInputStrict => {}
    }
}

fn collect_group_reference_stats_context(
    expr: Option<&ContextExpr>,
    stats: &mut GroupReferenceStats,
) {
    let Some(expr) = expr else {
        return;
    };

    match expr {
        ContextExpr::Pattern(regex) => collect_group_reference_stats_into(regex, stats),
        ContextExpr::WordBoundary => {}
        ContextExpr::And(a, b) | ContextExpr::Or(a, b) => {
            collect_group_reference_stats_context(Some(a.as_ref()), stats);
            collect_group_reference_stats_context(Some(b.as_ref()), stats);
        }
        ContextExpr::Not(inner) => {
            collect_group_reference_stats_context(Some(inner.as_ref()), stats);
        }
    }
}

fn validate_unique_named_groups(regex: &Regex, seen: &mut HashSet<String>) -> ParseResult<()> {
    match regex {
        Regex::Concat(a, b) | Regex::Alt(a, b) => {
            validate_unique_named_groups(a, seen)?;
            validate_unique_named_groups(b, seen)?;
        }
        Regex::Star(inner)
        | Regex::Plus(inner)
        | Regex::Optional(inner)
        | Regex::RepeatExact(inner, _)
        | Regex::RepeatRange(inner, _, _)
        | Regex::CapturingGroup(_, inner)
        | Regex::NonCapturingGroup(inner) => {
            validate_unique_named_groups(inner, seen)?;
        }
        Regex::NamedGroup(name, inner) => {
            if !seen.insert(name.clone()) {
                return Err(ParseError::new(
                    ParseErrorKind::DuplicateGroupName(name.clone()),
                    Position::start(),
                ));
            }
            validate_unique_named_groups(inner, seen)?;
        }
        Regex::FlagsGroup {
            inner: Some(inner), ..
        } => {
            validate_unique_named_groups(inner, seen)?;
        }
        Regex::RewriteRule {
            pattern,
            replacement,
            context,
            ..
        } => {
            validate_unique_named_groups(pattern, seen)?;
            validate_unique_named_groups(replacement, seen)?;
            if let Some(context) = context {
                validate_unique_named_groups_context(context.left.as_ref(), seen)?;
                validate_unique_named_groups_context(context.right.as_ref(), seen)?;
            }
        }
        Regex::Empty
        | Regex::Char(_)
        | Regex::CharClass(_)
        | Regex::Any
        | Regex::GroupRef(_)
        | Regex::FlagsGroup { inner: None, .. }
        | Regex::WordBoundary
        | Regex::StartOfLine
        | Regex::EndOfLine
        | Regex::StartOfInput
        | Regex::EndOfInput
        | Regex::EndOfInputStrict => {}
    }

    Ok(())
}

fn validate_unique_named_groups_context(
    expr: Option<&ContextExpr>,
    seen: &mut HashSet<String>,
) -> ParseResult<()> {
    let Some(expr) = expr else {
        return Ok(());
    };

    match expr {
        ContextExpr::Pattern(regex) => validate_unique_named_groups(regex, seen),
        ContextExpr::WordBoundary => Ok(()),
        ContextExpr::And(a, b) | ContextExpr::Or(a, b) => {
            validate_unique_named_groups_context(Some(a.as_ref()), seen)?;
            validate_unique_named_groups_context(Some(b.as_ref()), seen)
        }
        ContextExpr::Not(inner) => validate_unique_named_groups_context(Some(inner.as_ref()), seen),
    }
}

fn collect_named_groups<'a>(
    regex: &'a Regex,
    named_groups: &mut HashMap<String, &'a Regex>,
) -> ParseResult<()> {
    match regex {
        Regex::Concat(a, b) | Regex::Alt(a, b) => {
            collect_named_groups(a, named_groups)?;
            collect_named_groups(b, named_groups)?;
        }
        Regex::Star(inner)
        | Regex::Plus(inner)
        | Regex::Optional(inner)
        | Regex::RepeatExact(inner, _)
        | Regex::RepeatRange(inner, _, _)
        | Regex::CapturingGroup(_, inner)
        | Regex::NonCapturingGroup(inner) => {
            collect_named_groups(inner, named_groups)?;
        }
        Regex::NamedGroup(name, inner) => {
            if named_groups.insert(name.clone(), inner.as_ref()).is_some() {
                return Err(ParseError::new(
                    ParseErrorKind::DuplicateGroupName(name.clone()),
                    Position::start(),
                ));
            }
            collect_named_groups(inner, named_groups)?;
        }
        Regex::FlagsGroup {
            inner: Some(inner), ..
        } => {
            collect_named_groups(inner, named_groups)?;
        }
        Regex::RewriteRule {
            pattern,
            replacement,
            context,
            ..
        } => {
            collect_named_groups(pattern, named_groups)?;
            collect_named_groups(replacement, named_groups)?;
            if let Some(context) = context {
                collect_named_groups_context(context.left.as_ref(), named_groups)?;
                collect_named_groups_context(context.right.as_ref(), named_groups)?;
            }
        }
        Regex::Empty
        | Regex::Char(_)
        | Regex::CharClass(_)
        | Regex::Any
        | Regex::GroupRef(_)
        | Regex::FlagsGroup { inner: None, .. }
        | Regex::WordBoundary
        | Regex::StartOfLine
        | Regex::EndOfLine
        | Regex::StartOfInput
        | Regex::EndOfInput
        | Regex::EndOfInputStrict => {}
    }

    Ok(())
}

fn collect_named_groups_context<'a>(
    expr: Option<&'a ContextExpr>,
    named_groups: &mut HashMap<String, &'a Regex>,
) -> ParseResult<()> {
    let Some(expr) = expr else {
        return Ok(());
    };

    match expr {
        ContextExpr::Pattern(regex) => collect_named_groups(regex, named_groups),
        ContextExpr::WordBoundary => Ok(()),
        ContextExpr::And(a, b) | ContextExpr::Or(a, b) => {
            collect_named_groups_context(Some(a.as_ref()), named_groups)?;
            collect_named_groups_context(Some(b.as_ref()), named_groups)
        }
        ContextExpr::Not(inner) => collect_named_groups_context(Some(inner.as_ref()), named_groups),
    }
}

fn expand_group_refs<'a>(
    regex: &'a Regex,
    named_groups: &HashMap<String, &'a Regex>,
    stack: &mut Vec<&'a str>,
    depth: usize,
) -> ParseResult<Regex> {
    if depth > MAX_GROUP_EXPANSION_DEPTH {
        return Err(ParseError::new(
            ParseErrorKind::InvalidGroupReference(format!(
                "group reference expansion exceeded maximum depth {}",
                MAX_GROUP_EXPANSION_DEPTH
            )),
            Position::start(),
        ));
    }

    match regex {
        Regex::GroupRef(name) => {
            if stack.contains(&name.as_str()) {
                let mut chain = stack.join(" -> ");
                if !chain.is_empty() {
                    chain.push_str(" -> ");
                }
                chain.push_str(name);
                return Err(ParseError::new(
                    ParseErrorKind::InvalidGroupReference(format!(
                        "cyclic group reference expansion: {}",
                        chain
                    )),
                    Position::start(),
                ));
            }

            let Some(inner) = named_groups.get(name).copied() else {
                return Err(ParseError::new(
                    ParseErrorKind::UndefinedGroupReference(name.clone()),
                    Position::start(),
                ));
            };

            stack.push(name.as_str());
            let expanded = expand_group_refs(inner, named_groups, stack, depth + 1)?;
            stack.pop();

            Ok(Regex::NonCapturingGroup(Box::new(expanded)))
        }
        Regex::Concat(a, b) => Ok(Regex::Concat(
            Box::new(expand_group_refs(a, named_groups, stack, depth)?),
            Box::new(expand_group_refs(b, named_groups, stack, depth)?),
        )),
        Regex::Alt(a, b) => Ok(Regex::Alt(
            Box::new(expand_group_refs(a, named_groups, stack, depth)?),
            Box::new(expand_group_refs(b, named_groups, stack, depth)?),
        )),
        Regex::Star(inner) => Ok(Regex::Star(Box::new(expand_group_refs(
            inner,
            named_groups,
            stack,
            depth,
        )?))),
        Regex::Plus(inner) => Ok(Regex::Plus(Box::new(expand_group_refs(
            inner,
            named_groups,
            stack,
            depth,
        )?))),
        Regex::Optional(inner) => Ok(Regex::Optional(Box::new(expand_group_refs(
            inner,
            named_groups,
            stack,
            depth,
        )?))),
        Regex::RepeatExact(inner, n) => Ok(Regex::RepeatExact(
            Box::new(expand_group_refs(inner, named_groups, stack, depth)?),
            *n,
        )),
        Regex::RepeatRange(inner, min, max) => Ok(Regex::RepeatRange(
            Box::new(expand_group_refs(inner, named_groups, stack, depth)?),
            *min,
            *max,
        )),
        Regex::CapturingGroup(n, inner) => Ok(Regex::CapturingGroup(
            *n,
            Box::new(expand_group_refs(inner, named_groups, stack, depth)?),
        )),
        Regex::NonCapturingGroup(inner) => Ok(Regex::NonCapturingGroup(Box::new(
            expand_group_refs(inner, named_groups, stack, depth)?,
        ))),
        Regex::NamedGroup(name, inner) => Ok(Regex::NamedGroup(
            name.clone(),
            Box::new(expand_group_refs(inner, named_groups, stack, depth)?),
        )),
        Regex::FlagsGroup { flags, inner } => Ok(Regex::FlagsGroup {
            flags: flags.clone(),
            inner: inner
                .as_ref()
                .map(|inner| expand_group_refs(inner, named_groups, stack, depth).map(Box::new))
                .transpose()?,
        }),
        Regex::RewriteRule {
            pattern,
            replacement,
            context,
            weight,
        } => Ok(Regex::RewriteRule {
            pattern: Box::new(expand_group_refs(pattern, named_groups, stack, depth)?),
            replacement: Box::new(expand_group_refs(replacement, named_groups, stack, depth)?),
            context: context
                .as_ref()
                .map(|context| expand_group_refs_context(context, named_groups, stack, depth))
                .transpose()?
                .map(Box::new),
            weight: *weight,
        }),
        Regex::Empty
        | Regex::Char(_)
        | Regex::CharClass(_)
        | Regex::Any
        | Regex::WordBoundary
        | Regex::StartOfLine
        | Regex::EndOfLine
        | Regex::StartOfInput
        | Regex::EndOfInput
        | Regex::EndOfInputStrict => Ok(regex.clone()),
    }
}

fn expand_group_refs_context<'a>(
    context: &'a crate::phonetic::regex::ast::ContextPredicate,
    named_groups: &HashMap<String, &'a Regex>,
    stack: &mut Vec<&'a str>,
    depth: usize,
) -> ParseResult<crate::phonetic::regex::ast::ContextPredicate> {
    Ok(crate::phonetic::regex::ast::ContextPredicate {
        left: context
            .left
            .as_ref()
            .map(|expr| expand_group_refs_context_expr(expr, named_groups, stack, depth))
            .transpose()?,
        right: context
            .right
            .as_ref()
            .map(|expr| expand_group_refs_context_expr(expr, named_groups, stack, depth))
            .transpose()?,
        syllable: context.syllable.clone(),
    })
}

fn expand_group_refs_context_expr<'a>(
    expr: &'a ContextExpr,
    named_groups: &HashMap<String, &'a Regex>,
    stack: &mut Vec<&'a str>,
    depth: usize,
) -> ParseResult<ContextExpr> {
    match expr {
        ContextExpr::Pattern(regex) => Ok(ContextExpr::Pattern(expand_group_refs(
            regex,
            named_groups,
            stack,
            depth,
        )?)),
        ContextExpr::WordBoundary => Ok(ContextExpr::WordBoundary),
        ContextExpr::And(a, b) => Ok(ContextExpr::And(
            Box::new(expand_group_refs_context_expr(
                a,
                named_groups,
                stack,
                depth,
            )?),
            Box::new(expand_group_refs_context_expr(
                b,
                named_groups,
                stack,
                depth,
            )?),
        )),
        ContextExpr::Or(a, b) => Ok(ContextExpr::Or(
            Box::new(expand_group_refs_context_expr(
                a,
                named_groups,
                stack,
                depth,
            )?),
            Box::new(expand_group_refs_context_expr(
                b,
                named_groups,
                stack,
                depth,
            )?),
        )),
        ContextExpr::Not(inner) => Ok(ContextExpr::Not(Box::new(expand_group_refs_context_expr(
            inner,
            named_groups,
            stack,
            depth,
        )?))),
    }
}

fn prepare_byte_regex_for_compile(regex: &RegexByte) -> ParseResult<Cow<'_, RegexByte>> {
    let stats = collect_byte_group_reference_stats(regex);
    if !stats.has_groups() {
        return Ok(Cow::Borrowed(regex));
    }

    if !stats.needs_expansion() {
        let mut seen = HashSet::with_capacity(stats.named_groups);
        validate_unique_byte_named_groups(regex, &mut seen)?;
        return Ok(Cow::Borrowed(regex));
    }

    let mut named_groups = HashMap::with_capacity(stats.named_groups);
    collect_byte_named_groups(regex, &mut named_groups)?;
    let mut stack = Vec::with_capacity(stats.stack_capacity());
    expand_byte_group_refs(regex, &named_groups, &mut stack, 0).map(Cow::Owned)
}

fn collect_byte_group_reference_stats(regex: &RegexByte) -> GroupReferenceStats {
    let mut stats = GroupReferenceStats::default();
    collect_byte_group_reference_stats_into(regex, &mut stats);
    stats
}

fn collect_byte_group_reference_stats_into(regex: &RegexByte, stats: &mut GroupReferenceStats) {
    match regex {
        RegexByte::Concat(a, b) | RegexByte::Alt(a, b) => {
            collect_byte_group_reference_stats_into(a, stats);
            collect_byte_group_reference_stats_into(b, stats);
        }
        RegexByte::Star(inner)
        | RegexByte::Plus(inner)
        | RegexByte::Optional(inner)
        | RegexByte::RepeatExact(inner, _)
        | RegexByte::RepeatRange(inner, _, _)
        | RegexByte::CapturingGroup(_, inner)
        | RegexByte::NonCapturingGroup(inner) => {
            collect_byte_group_reference_stats_into(inner, stats);
        }
        RegexByte::NamedGroup(_, inner) => {
            stats.named_groups += 1;
            collect_byte_group_reference_stats_into(inner, stats);
        }
        RegexByte::GroupRef(_) => {
            stats.group_refs += 1;
        }
        RegexByte::FlagsGroup {
            inner: Some(inner), ..
        } => {
            collect_byte_group_reference_stats_into(inner, stats);
        }
        RegexByte::RewriteRule {
            pattern,
            replacement,
            context,
            ..
        } => {
            collect_byte_group_reference_stats_into(pattern, stats);
            collect_byte_group_reference_stats_into(replacement, stats);
            if let Some(context) = context {
                collect_byte_group_reference_stats_context(context.left.as_ref(), stats);
                collect_byte_group_reference_stats_context(context.right.as_ref(), stats);
            }
        }
        RegexByte::Empty
        | RegexByte::Byte(_)
        | RegexByte::ByteClass(_)
        | RegexByte::Any
        | RegexByte::FlagsGroup { inner: None, .. }
        | RegexByte::WordBoundary
        | RegexByte::StartOfLine
        | RegexByte::EndOfLine
        | RegexByte::StartOfInput
        | RegexByte::EndOfInput
        | RegexByte::EndOfInputStrict => {}
    }
}

fn collect_byte_group_reference_stats_context(
    expr: Option<&ContextExprByte>,
    stats: &mut GroupReferenceStats,
) {
    let Some(expr) = expr else {
        return;
    };

    match expr {
        ContextExprByte::Pattern(regex) => collect_byte_group_reference_stats_into(regex, stats),
        ContextExprByte::WordBoundary => {}
        ContextExprByte::And(a, b) | ContextExprByte::Or(a, b) => {
            collect_byte_group_reference_stats_context(Some(a.as_ref()), stats);
            collect_byte_group_reference_stats_context(Some(b.as_ref()), stats);
        }
        ContextExprByte::Not(inner) => {
            collect_byte_group_reference_stats_context(Some(inner.as_ref()), stats);
        }
    }
}

fn validate_unique_byte_named_groups(
    regex: &RegexByte,
    seen: &mut HashSet<String>,
) -> ParseResult<()> {
    match regex {
        RegexByte::Concat(a, b) | RegexByte::Alt(a, b) => {
            validate_unique_byte_named_groups(a, seen)?;
            validate_unique_byte_named_groups(b, seen)?;
        }
        RegexByte::Star(inner)
        | RegexByte::Plus(inner)
        | RegexByte::Optional(inner)
        | RegexByte::RepeatExact(inner, _)
        | RegexByte::RepeatRange(inner, _, _)
        | RegexByte::CapturingGroup(_, inner)
        | RegexByte::NonCapturingGroup(inner) => {
            validate_unique_byte_named_groups(inner, seen)?;
        }
        RegexByte::NamedGroup(name, inner) => {
            if !seen.insert(name.clone()) {
                return Err(ParseError::new(
                    ParseErrorKind::DuplicateGroupName(name.clone()),
                    Position::start(),
                ));
            }
            validate_unique_byte_named_groups(inner, seen)?;
        }
        RegexByte::FlagsGroup {
            inner: Some(inner), ..
        } => {
            validate_unique_byte_named_groups(inner, seen)?;
        }
        RegexByte::RewriteRule {
            pattern,
            replacement,
            context,
            ..
        } => {
            validate_unique_byte_named_groups(pattern, seen)?;
            validate_unique_byte_named_groups(replacement, seen)?;
            if let Some(context) = context {
                validate_unique_byte_named_groups_context(context.left.as_ref(), seen)?;
                validate_unique_byte_named_groups_context(context.right.as_ref(), seen)?;
            }
        }
        RegexByte::Empty
        | RegexByte::Byte(_)
        | RegexByte::ByteClass(_)
        | RegexByte::Any
        | RegexByte::GroupRef(_)
        | RegexByte::FlagsGroup { inner: None, .. }
        | RegexByte::WordBoundary
        | RegexByte::StartOfLine
        | RegexByte::EndOfLine
        | RegexByte::StartOfInput
        | RegexByte::EndOfInput
        | RegexByte::EndOfInputStrict => {}
    }

    Ok(())
}

fn validate_unique_byte_named_groups_context(
    expr: Option<&ContextExprByte>,
    seen: &mut HashSet<String>,
) -> ParseResult<()> {
    let Some(expr) = expr else {
        return Ok(());
    };

    match expr {
        ContextExprByte::Pattern(regex) => validate_unique_byte_named_groups(regex, seen),
        ContextExprByte::WordBoundary => Ok(()),
        ContextExprByte::And(a, b) | ContextExprByte::Or(a, b) => {
            validate_unique_byte_named_groups_context(Some(a.as_ref()), seen)?;
            validate_unique_byte_named_groups_context(Some(b.as_ref()), seen)
        }
        ContextExprByte::Not(inner) => {
            validate_unique_byte_named_groups_context(Some(inner.as_ref()), seen)
        }
    }
}

fn collect_byte_named_groups<'a>(
    regex: &'a RegexByte,
    named_groups: &mut HashMap<String, &'a RegexByte>,
) -> ParseResult<()> {
    match regex {
        RegexByte::Concat(a, b) | RegexByte::Alt(a, b) => {
            collect_byte_named_groups(a, named_groups)?;
            collect_byte_named_groups(b, named_groups)?;
        }
        RegexByte::Star(inner)
        | RegexByte::Plus(inner)
        | RegexByte::Optional(inner)
        | RegexByte::RepeatExact(inner, _)
        | RegexByte::RepeatRange(inner, _, _)
        | RegexByte::CapturingGroup(_, inner)
        | RegexByte::NonCapturingGroup(inner) => {
            collect_byte_named_groups(inner, named_groups)?;
        }
        RegexByte::NamedGroup(name, inner) => {
            if named_groups.insert(name.clone(), inner.as_ref()).is_some() {
                return Err(ParseError::new(
                    ParseErrorKind::DuplicateGroupName(name.clone()),
                    Position::start(),
                ));
            }
            collect_byte_named_groups(inner, named_groups)?;
        }
        RegexByte::FlagsGroup {
            inner: Some(inner), ..
        } => {
            collect_byte_named_groups(inner, named_groups)?;
        }
        RegexByte::RewriteRule {
            pattern,
            replacement,
            context,
            ..
        } => {
            collect_byte_named_groups(pattern, named_groups)?;
            collect_byte_named_groups(replacement, named_groups)?;
            if let Some(context) = context {
                collect_byte_named_groups_context(context.left.as_ref(), named_groups)?;
                collect_byte_named_groups_context(context.right.as_ref(), named_groups)?;
            }
        }
        RegexByte::Empty
        | RegexByte::Byte(_)
        | RegexByte::ByteClass(_)
        | RegexByte::Any
        | RegexByte::GroupRef(_)
        | RegexByte::FlagsGroup { inner: None, .. }
        | RegexByte::WordBoundary
        | RegexByte::StartOfLine
        | RegexByte::EndOfLine
        | RegexByte::StartOfInput
        | RegexByte::EndOfInput
        | RegexByte::EndOfInputStrict => {}
    }

    Ok(())
}

fn collect_byte_named_groups_context<'a>(
    expr: Option<&'a ContextExprByte>,
    named_groups: &mut HashMap<String, &'a RegexByte>,
) -> ParseResult<()> {
    let Some(expr) = expr else {
        return Ok(());
    };

    match expr {
        ContextExprByte::Pattern(regex) => collect_byte_named_groups(regex, named_groups),
        ContextExprByte::WordBoundary => Ok(()),
        ContextExprByte::And(a, b) | ContextExprByte::Or(a, b) => {
            collect_byte_named_groups_context(Some(a.as_ref()), named_groups)?;
            collect_byte_named_groups_context(Some(b.as_ref()), named_groups)
        }
        ContextExprByte::Not(inner) => {
            collect_byte_named_groups_context(Some(inner.as_ref()), named_groups)
        }
    }
}

fn expand_byte_group_refs<'a>(
    regex: &'a RegexByte,
    named_groups: &HashMap<String, &'a RegexByte>,
    stack: &mut Vec<&'a str>,
    depth: usize,
) -> ParseResult<RegexByte> {
    if depth > MAX_GROUP_EXPANSION_DEPTH {
        return Err(ParseError::new(
            ParseErrorKind::InvalidGroupReference(format!(
                "group reference expansion exceeded maximum depth {}",
                MAX_GROUP_EXPANSION_DEPTH
            )),
            Position::start(),
        ));
    }

    match regex {
        RegexByte::GroupRef(name) => {
            if stack.contains(&name.as_str()) {
                let mut chain = stack.join(" -> ");
                if !chain.is_empty() {
                    chain.push_str(" -> ");
                }
                chain.push_str(name);
                return Err(ParseError::new(
                    ParseErrorKind::InvalidGroupReference(format!(
                        "cyclic group reference expansion: {}",
                        chain
                    )),
                    Position::start(),
                ));
            }

            let Some(inner) = named_groups.get(name).copied() else {
                return Err(ParseError::new(
                    ParseErrorKind::UndefinedGroupReference(name.clone()),
                    Position::start(),
                ));
            };

            stack.push(name.as_str());
            let expanded = expand_byte_group_refs(inner, named_groups, stack, depth + 1)?;
            stack.pop();

            Ok(RegexByte::NonCapturingGroup(Box::new(expanded)))
        }
        RegexByte::Concat(a, b) => Ok(RegexByte::Concat(
            Box::new(expand_byte_group_refs(a, named_groups, stack, depth)?),
            Box::new(expand_byte_group_refs(b, named_groups, stack, depth)?),
        )),
        RegexByte::Alt(a, b) => Ok(RegexByte::Alt(
            Box::new(expand_byte_group_refs(a, named_groups, stack, depth)?),
            Box::new(expand_byte_group_refs(b, named_groups, stack, depth)?),
        )),
        RegexByte::Star(inner) => Ok(RegexByte::Star(Box::new(expand_byte_group_refs(
            inner,
            named_groups,
            stack,
            depth,
        )?))),
        RegexByte::Plus(inner) => Ok(RegexByte::Plus(Box::new(expand_byte_group_refs(
            inner,
            named_groups,
            stack,
            depth,
        )?))),
        RegexByte::Optional(inner) => Ok(RegexByte::Optional(Box::new(expand_byte_group_refs(
            inner,
            named_groups,
            stack,
            depth,
        )?))),
        RegexByte::RepeatExact(inner, n) => Ok(RegexByte::RepeatExact(
            Box::new(expand_byte_group_refs(inner, named_groups, stack, depth)?),
            *n,
        )),
        RegexByte::RepeatRange(inner, min, max) => Ok(RegexByte::RepeatRange(
            Box::new(expand_byte_group_refs(inner, named_groups, stack, depth)?),
            *min,
            *max,
        )),
        RegexByte::CapturingGroup(n, inner) => Ok(RegexByte::CapturingGroup(
            *n,
            Box::new(expand_byte_group_refs(inner, named_groups, stack, depth)?),
        )),
        RegexByte::NonCapturingGroup(inner) => Ok(RegexByte::NonCapturingGroup(Box::new(
            expand_byte_group_refs(inner, named_groups, stack, depth)?,
        ))),
        RegexByte::NamedGroup(name, inner) => Ok(RegexByte::NamedGroup(
            name.clone(),
            Box::new(expand_byte_group_refs(inner, named_groups, stack, depth)?),
        )),
        RegexByte::FlagsGroup { flags, inner } => Ok(RegexByte::FlagsGroup {
            flags: flags.clone(),
            inner: inner
                .as_ref()
                .map(|inner| {
                    expand_byte_group_refs(inner, named_groups, stack, depth).map(Box::new)
                })
                .transpose()?,
        }),
        RegexByte::RewriteRule {
            pattern,
            replacement,
            context,
            weight,
        } => Ok(RegexByte::RewriteRule {
            pattern: Box::new(expand_byte_group_refs(pattern, named_groups, stack, depth)?),
            replacement: Box::new(expand_byte_group_refs(
                replacement,
                named_groups,
                stack,
                depth,
            )?),
            context: context
                .as_ref()
                .map(|context| expand_byte_group_refs_context(context, named_groups, stack, depth))
                .transpose()?
                .map(Box::new),
            weight: *weight,
        }),
        RegexByte::Empty
        | RegexByte::Byte(_)
        | RegexByte::ByteClass(_)
        | RegexByte::Any
        | RegexByte::WordBoundary
        | RegexByte::StartOfLine
        | RegexByte::EndOfLine
        | RegexByte::StartOfInput
        | RegexByte::EndOfInput
        | RegexByte::EndOfInputStrict => Ok(regex.clone()),
    }
}

fn expand_byte_group_refs_context<'a>(
    context: &'a crate::phonetic::regex::ast::ContextPredicateByte,
    named_groups: &HashMap<String, &'a RegexByte>,
    stack: &mut Vec<&'a str>,
    depth: usize,
) -> ParseResult<crate::phonetic::regex::ast::ContextPredicateByte> {
    Ok(crate::phonetic::regex::ast::ContextPredicateByte {
        left: context
            .left
            .as_ref()
            .map(|expr| expand_byte_group_refs_context_expr(expr, named_groups, stack, depth))
            .transpose()?,
        right: context
            .right
            .as_ref()
            .map(|expr| expand_byte_group_refs_context_expr(expr, named_groups, stack, depth))
            .transpose()?,
        syllable: context.syllable.clone(),
    })
}

fn expand_byte_group_refs_context_expr<'a>(
    expr: &'a ContextExprByte,
    named_groups: &HashMap<String, &'a RegexByte>,
    stack: &mut Vec<&'a str>,
    depth: usize,
) -> ParseResult<ContextExprByte> {
    match expr {
        ContextExprByte::Pattern(regex) => Ok(ContextExprByte::Pattern(expand_byte_group_refs(
            regex,
            named_groups,
            stack,
            depth,
        )?)),
        ContextExprByte::WordBoundary => Ok(ContextExprByte::WordBoundary),
        ContextExprByte::And(a, b) => Ok(ContextExprByte::And(
            Box::new(expand_byte_group_refs_context_expr(
                a,
                named_groups,
                stack,
                depth,
            )?),
            Box::new(expand_byte_group_refs_context_expr(
                b,
                named_groups,
                stack,
                depth,
            )?),
        )),
        ContextExprByte::Or(a, b) => Ok(ContextExprByte::Or(
            Box::new(expand_byte_group_refs_context_expr(
                a,
                named_groups,
                stack,
                depth,
            )?),
            Box::new(expand_byte_group_refs_context_expr(
                b,
                named_groups,
                stack,
                depth,
            )?),
        )),
        ContextExprByte::Not(inner) => Ok(ContextExprByte::Not(Box::new(
            expand_byte_group_refs_context_expr(inner, named_groups, stack, depth)?,
        ))),
    }
}

/// Character-level NFA compiler.
pub struct NFACompilerChar {
    builder: ThompsonBuilderChar,
    /// Symbol table for pattern expansion
    symbols: HashMap<String, Vec<char>>,
    /// Regex flags (multiline, dotall, etc.)
    flags: RegexFlags,
    /// Use trampolining for stack-safe compilation of deep patterns.
    /// When false (default), uses recursive compilation which is faster
    /// but may stack overflow on deeply nested patterns.
    use_trampolining: bool,
    /// Optimization configuration. When Some, the NFA is optimized after compilation.
    /// Default: full optimization enabled.
    optimization: Option<OptimizationConfig>,
}

impl NFACompilerChar {
    /// Create a new compiler with recursive compilation and optimization enabled (default).
    pub fn new() -> Self {
        Self {
            builder: ThompsonBuilderChar::new(),
            symbols: HashMap::new(),
            flags: RegexFlags::default(),
            use_trampolining: false,
            optimization: Some(OptimizationConfig::full()),
        }
    }

    /// Enable trampolining for stack-safe compilation.
    ///
    /// Use this when compiling potentially deep patterns from untrusted input.
    /// Trampolining uses an explicit heap-allocated stack instead of the call stack,
    /// preventing stack overflow on deeply nested patterns like `((((a))))`.
    ///
    /// # Example
    /// ```
    /// use liblevenshtein::phonetic::nfa::NFACompilerChar;
    ///
    /// // For untrusted input, use trampolining
    /// let compiler = NFACompilerChar::new().with_trampolining();
    /// ```
    pub fn with_trampolining(mut self) -> Self {
        self.use_trampolining = true;
        self
    }

    /// Set trampolining mode.
    ///
    /// When `enabled` is true, compilation uses an iterative trampoline-based
    /// approach that is safe for deeply nested patterns but slightly slower.
    /// When false (default), uses recursive compilation which is faster.
    pub fn set_trampolining(&mut self, enabled: bool) {
        self.use_trampolining = enabled;
    }

    /// Check if trampolining is enabled.
    pub fn is_trampolining(&self) -> bool {
        self.use_trampolining
    }

    /// Enable optimization with the given configuration.
    ///
    /// By default, full optimization is enabled. Use this to customize
    /// which optimization passes are applied.
    ///
    /// # Example
    /// ```
    /// use liblevenshtein::phonetic::nfa::{NFACompilerChar, OptimizationConfig};
    ///
    /// // Use quick optimization (no epsilon elimination)
    /// let compiler = NFACompilerChar::new()
    ///     .with_optimization(OptimizationConfig::quick());
    /// ```
    pub fn with_optimization(mut self, config: OptimizationConfig) -> Self {
        self.optimization = Some(config);
        self
    }

    /// Disable optimization entirely.
    ///
    /// This is useful for debugging or when the NFA will be used only once.
    /// The resulting NFA may have epsilon transitions and unreachable states.
    ///
    /// # Example
    /// ```
    /// use liblevenshtein::phonetic::nfa::NFACompilerChar;
    ///
    /// let compiler = NFACompilerChar::new().without_optimization();
    /// ```
    pub fn without_optimization(mut self) -> Self {
        self.optimization = None;
        self
    }

    /// Set the optimization configuration.
    ///
    /// Pass `None` to disable optimization, or `Some(config)` to enable it.
    pub fn set_optimization(&mut self, config: Option<OptimizationConfig>) {
        self.optimization = config;
    }

    /// Check if optimization is enabled.
    pub fn is_optimization_enabled(&self) -> bool {
        self.optimization.is_some()
    }

    /// Get the current optimization configuration.
    pub fn optimization_config(&self) -> Option<&OptimizationConfig> {
        self.optimization.as_ref()
    }

    /// Add a symbol to the symbol table.
    ///
    /// Symbols can be referenced in patterns and will be expanded during compilation.
    pub fn add_symbol(&mut self, name: impl Into<String>, chars: Vec<char>) {
        self.symbols.insert(name.into(), chars);
    }

    /// Set the regex flags for compilation.
    pub fn set_flags(&mut self, flags: RegexFlags) {
        self.flags = flags;
    }

    /// Get the current flags.
    pub fn flags(&self) -> &RegexFlags {
        &self.flags
    }

    /// Get a symbol by name.
    pub fn get_symbol(&self, name: &str) -> Option<&Vec<char>> {
        self.symbols.get(name)
    }

    /// Compile a regex AST to an NFA.
    ///
    /// This applies flag transformations (like `(?i)` case-insensitive) before
    /// compilation. If optimization is enabled (default), the NFA is automatically
    /// optimized after Thompson construction.
    ///
    /// For access to runtime flags (unicode normalization, multiline, dotall),
    /// use [`compile_with_flags`] instead.
    pub fn compile(&mut self, regex: &Regex) -> ParseResult<NFAChar> {
        // Apply flag transformations before compilation
        let transform_result = apply_flags(regex);
        let prepared = prepare_regex_for_compile(&transform_result.regex)?;
        let nfa = self.compile_regex(prepared.as_ref())?;

        // Apply optimization if configured
        let mut nfa = if let Some(ref config) = self.optimization {
            let optimizer = NfaOptimizerChar::new(config.clone());
            let (optimized, _stats) = optimizer.optimize(nfa);
            optimized
        } else {
            nfa
        };

        // H9: Finalize CSR transition table
        nfa.finalize();
        Ok(nfa)
    }

    /// Compile a regex AST to an NFA with full flag support.
    ///
    /// This applies flag transformations and returns both the NFA and runtime
    /// settings extracted from flags like `(?u:NFC)`, `(?m)`, and `(?s)`.
    pub fn compile_with_flags(&mut self, regex: &Regex) -> ParseResult<CompileResultChar> {
        // Apply flag transformations before compilation
        let transform_result = apply_flags(regex);
        let prepared = prepare_regex_for_compile(&transform_result.regex)?;
        let nfa = self.compile_regex(prepared.as_ref())?;

        // Apply optimization if configured
        let nfa = if let Some(ref config) = self.optimization {
            let optimizer = NfaOptimizerChar::new(config.clone());
            let (optimized, _stats) = optimizer.optimize(nfa);
            optimized
        } else {
            nfa
        };

        // H9: Finalize CSR transition table
        let mut nfa = nfa;
        nfa.finalize();

        Ok(CompileResultChar {
            nfa,
            unicode_normalization: transform_result.unicode_normalization,
            multiline: transform_result.multiline,
            dotall: transform_result.dotall,
            local_distance: transform_result.local_distance,
        })
    }

    /// Compile a rewrite rule.
    ///
    /// The source pattern NFA is automatically optimized if optimization is enabled.
    pub fn compile_rewrite(&mut self, regex: &Regex) -> ParseResult<CompiledRewriteChar> {
        let prepared = prepare_regex_for_compile(regex)?;
        match prepared.as_ref() {
            Regex::RewriteRule {
                pattern,
                replacement,
                context,
                weight,
            } => {
                // Compile and optimize the source pattern
                let source = self.compile_regex(pattern)?;
                let mut source = if let Some(ref config) = self.optimization {
                    let optimizer = NfaOptimizerChar::new(config.clone());
                    let (optimized, _stats) = optimizer.optimize(source);
                    optimized
                } else {
                    source
                };
                // H9: Finalize CSR transition table
                source.finalize();

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
                // Apply optimization if configured
                let mut nfa = if let Some(ref config) = self.optimization {
                    let optimizer = NfaOptimizerChar::new(config.clone());
                    let (optimized, _stats) = optimizer.optimize(nfa);
                    optimized
                } else {
                    nfa
                };
                // H9: Finalize CSR transition table
                nfa.finalize();
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
    ///
    /// This method dispatches to either the recursive or trampolined
    /// implementation based on the `use_trampolining` flag.
    fn compile_regex(&mut self, regex: &Regex) -> ParseResult<NFAChar> {
        if self.use_trampolining {
            self.compile_regex_trampolined(regex)
        } else {
            self.compile_regex_recursive(regex)
        }
    }

    /// Recursive implementation of regex compilation.
    ///
    /// This is the default, faster implementation that uses the call stack.
    /// For deeply nested patterns, use `with_trampolining()` instead.
    #[allow(deprecated)]
    fn compile_regex_recursive(&mut self, regex: &Regex) -> ParseResult<NFAChar> {
        match regex {
            Regex::Empty => Ok(self.builder.epsilon()),
            Regex::Char(c) => Ok(self.builder.single_char(*c)),
            Regex::CharClass(class) => Ok(self.builder.char_class(class.clone())),
            Regex::Any => Ok(self.builder.any_char()),
            Regex::Concat(a, b) => {
                let nfa_a = self.compile_regex_recursive(a)?;
                let nfa_b = self.compile_regex_recursive(b)?;
                Ok(self.builder.concatenate(nfa_a, nfa_b))
            }
            Regex::Alt(a, b) => {
                let nfa_a = self.compile_regex_recursive(a)?;
                let nfa_b = self.compile_regex_recursive(b)?;
                Ok(self.builder.alternation(nfa_a, nfa_b))
            }
            Regex::Star(inner) => {
                let nfa = self.compile_regex_recursive(inner)?;
                Ok(self.builder.kleene_star(nfa))
            }
            Regex::Plus(inner) => {
                let nfa = self.compile_regex_recursive(inner)?;
                Ok(self.builder.kleene_plus(nfa))
            }
            Regex::Optional(inner) => {
                let nfa = self.compile_regex_recursive(inner)?;
                Ok(self.builder.optional(nfa))
            }
            Regex::RepeatExact(inner, n) => {
                let nfa = self.compile_regex_recursive(inner)?;
                Ok(self.builder.repeat_exact(nfa, *n))
            }
            Regex::RepeatRange(inner, min, max) => {
                let nfa = self.compile_regex_recursive(inner)?;
                Ok(self.builder.repeat_range(nfa, *min, *max))
            }
            // Capturing group: (...)
            Regex::CapturingGroup(_, inner) => {
                // Captures are parse-time structure here; the Thompson NFA
                // accepts the group's language without exposing capture spans.
                self.compile_regex_recursive(inner)
            }
            // Non-capturing group: (?:...)
            Regex::NonCapturingGroup(inner) => self.compile_regex_recursive(inner),
            // Named group: (?<name>...)
            Regex::NamedGroup(_, inner) => {
                // Named references are expanded by prepare_regex_for_compile();
                // the named group itself contributes only its inner language.
                self.compile_regex_recursive(inner)
            }
            // Group reference: (?&name)
            Regex::GroupRef(name) => Err(ParseError::new(
                ParseErrorKind::InvalidGroupReference(format!(
                    "unexpanded group reference (?&{}) reached recursive NFA compilation",
                    name
                )),
                Position::start(),
            )),
            // Flags group: (?flags:...) or (?flags)
            Regex::FlagsGroup { inner, .. } => {
                // Public compile entry points apply flag transformations before
                // recursive Thompson construction. If a flag wrapper remains
                // here, its inner structure is already the matchable portion.
                match inner {
                    Some(inner_regex) => self.compile_regex_recursive(inner_regex),
                    None => Ok(self.builder.epsilon()),
                }
            }
            Regex::WordBoundary => {
                // Word boundary is a zero-width assertion in this NFA layer.
                // Context matching carries directional boundary semantics.
                Ok(self.builder.epsilon())
            }
            // Anchor assertions (zero-width)
            Regex::StartOfLine => Ok(self.builder.start_of_line()),
            Regex::EndOfLine => Ok(self.builder.end_of_line()),
            Regex::StartOfInput => Ok(self.builder.start_of_input()),
            Regex::EndOfInput => Ok(self.builder.end_of_input()),
            Regex::EndOfInputStrict => Ok(self.builder.end_of_input_strict()),
            Regex::RewriteRule { pattern, .. } => {
                // When compiling a rewrite rule as a pattern, just compile the pattern part
                self.compile_regex_recursive(pattern)
            }
        }
    }

    fn pop_trampolined_value(
        value_stack: &mut Vec<NFAChar>,
        operand: &'static str,
    ) -> ParseResult<NFAChar> {
        value_stack.pop().ok_or_else(|| {
            ParseError::new(
                ParseErrorKind::InternalError(format!(
                    "missing {operand} during trampolined NFA compilation"
                )),
                Position::start(),
            )
        })
    }

    /// Trampolined (iterative) implementation of regex compilation.
    ///
    /// This implementation uses an explicit heap-allocated work stack instead
    /// of the call stack, making it safe for deeply nested patterns that would
    /// otherwise cause stack overflow.
    #[allow(deprecated)]
    fn compile_regex_trampolined(&mut self, regex: &Regex) -> ParseResult<NFAChar> {
        let mut work_stack: Vec<CompileWork<'_>> = vec![CompileWork::Compile(regex)];
        let mut value_stack: Vec<NFAChar> = Vec::new();

        while let Some(work) = work_stack.pop() {
            match work {
                CompileWork::Compile(node) => {
                    self.process_compile_node(node, &mut work_stack, &mut value_stack)?;
                }
                CompileWork::DoConcatenate => {
                    let b = Self::pop_trampolined_value(&mut value_stack, "second concat operand")?;
                    let a = Self::pop_trampolined_value(&mut value_stack, "first concat operand")?;
                    value_stack.push(self.builder.concatenate(a, b));
                }
                CompileWork::DoAlternation => {
                    let b = Self::pop_trampolined_value(
                        &mut value_stack,
                        "second alternation operand",
                    )?;
                    let a =
                        Self::pop_trampolined_value(&mut value_stack, "first alternation operand")?;
                    value_stack.push(self.builder.alternation(a, b));
                }
                CompileWork::DoStar => {
                    let inner = Self::pop_trampolined_value(&mut value_stack, "star operand")?;
                    value_stack.push(self.builder.kleene_star(inner));
                }
                CompileWork::DoPlus => {
                    let inner = Self::pop_trampolined_value(&mut value_stack, "plus operand")?;
                    value_stack.push(self.builder.kleene_plus(inner));
                }
                CompileWork::DoOptional => {
                    let inner = Self::pop_trampolined_value(&mut value_stack, "optional operand")?;
                    value_stack.push(self.builder.optional(inner));
                }
                CompileWork::DoRepeatExact(n) => {
                    let inner =
                        Self::pop_trampolined_value(&mut value_stack, "exact repeat operand")?;
                    value_stack.push(self.builder.repeat_exact(inner, n));
                }
                CompileWork::DoRepeatRange(min, max) => {
                    let inner =
                        Self::pop_trampolined_value(&mut value_stack, "repeat range operand")?;
                    value_stack.push(self.builder.repeat_range(inner, min, max));
                }
            }
        }

        value_stack.pop().ok_or_else(|| {
            ParseError::new(
                ParseErrorKind::InternalError("empty value stack after compilation".into()),
                Position::start(),
            )
        })
    }

    /// Process a single regex node for trampolined compilation.
    ///
    /// This method handles each regex variant by either:
    /// - Pushing the result directly to the value stack (base cases)
    /// - Pushing work items for children to the work stack (recursive cases)
    #[allow(deprecated)]
    fn process_compile_node<'a>(
        &mut self,
        node: &'a Regex,
        work_stack: &mut Vec<CompileWork<'a>>,
        value_stack: &mut Vec<NFAChar>,
    ) -> ParseResult<()> {
        match node {
            // === Base cases: push result directly ===
            Regex::Empty => value_stack.push(self.builder.epsilon()),
            Regex::Char(c) => value_stack.push(self.builder.single_char(*c)),
            Regex::CharClass(class) => value_stack.push(self.builder.char_class(class.clone())),
            Regex::Any => value_stack.push(self.builder.any_char()),
            Regex::WordBoundary => value_stack.push(self.builder.epsilon()),
            Regex::StartOfLine => value_stack.push(self.builder.start_of_line()),
            Regex::EndOfLine => value_stack.push(self.builder.end_of_line()),
            Regex::StartOfInput => value_stack.push(self.builder.start_of_input()),
            Regex::EndOfInput => value_stack.push(self.builder.end_of_input()),
            Regex::EndOfInputStrict => value_stack.push(self.builder.end_of_input_strict()),

            // === Binary operations: push operation, then children in reverse ===
            Regex::Concat(a, b) => {
                work_stack.push(CompileWork::DoConcatenate);
                work_stack.push(CompileWork::Compile(b));
                work_stack.push(CompileWork::Compile(a));
            }
            Regex::Alt(a, b) => {
                work_stack.push(CompileWork::DoAlternation);
                work_stack.push(CompileWork::Compile(b));
                work_stack.push(CompileWork::Compile(a));
            }

            // === Unary operations: push operation, then child ===
            Regex::Star(inner) => {
                work_stack.push(CompileWork::DoStar);
                work_stack.push(CompileWork::Compile(inner));
            }
            Regex::Plus(inner) => {
                work_stack.push(CompileWork::DoPlus);
                work_stack.push(CompileWork::Compile(inner));
            }
            Regex::Optional(inner) => {
                work_stack.push(CompileWork::DoOptional);
                work_stack.push(CompileWork::Compile(inner));
            }
            Regex::RepeatExact(inner, n) => {
                work_stack.push(CompileWork::DoRepeatExact(*n));
                work_stack.push(CompileWork::Compile(inner));
            }
            Regex::RepeatRange(inner, min, max) => {
                work_stack.push(CompileWork::DoRepeatRange(*min, *max));
                work_stack.push(CompileWork::Compile(inner));
            }

            // === Group wrappers: pass-through to inner ===
            Regex::CapturingGroup(_, inner)
            | Regex::NonCapturingGroup(inner)
            | Regex::NamedGroup(_, inner) => {
                work_stack.push(CompileWork::Compile(inner));
            }
            Regex::FlagsGroup {
                inner: Some(inner_regex),
                ..
            } => {
                work_stack.push(CompileWork::Compile(inner_regex));
            }
            Regex::FlagsGroup { inner: None, .. } => {
                value_stack.push(self.builder.epsilon());
            }
            Regex::RewriteRule { pattern, .. } => {
                work_stack.push(CompileWork::Compile(pattern));
            }

            // === Error case ===
            Regex::GroupRef(name) => {
                return Err(ParseError::new(
                    ParseErrorKind::InvalidGroupReference(format!(
                        "unexpanded group reference (?&{}) reached trampolined NFA compilation",
                        name
                    )),
                    Position::start(),
                ));
            }
        }
        Ok(())
    }

    /// Convert a regex to a literal string (for replacement).
    #[allow(deprecated)]
    fn regex_to_literal(&self, regex: &Regex) -> ParseResult<Vec<char>> {
        let mut chars = Vec::with_capacity(Self::literal_char_len(regex)?);
        Self::push_regex_literal_chars(regex, &mut chars)?;
        Ok(chars)
    }

    fn literal_char_len(regex: &Regex) -> ParseResult<usize> {
        let mut len = 0;
        let mut stack = Vec::with_capacity(8);
        stack.push(regex);

        while let Some(node) = stack.pop() {
            match node {
                Regex::Empty | Regex::FlagsGroup { inner: None, .. } => {}
                Regex::Char(_) => increment_literal_len(&mut len)?,
                Regex::Concat(a, b) => {
                    stack.push(b);
                    stack.push(a);
                }
                Regex::CapturingGroup(_, inner)
                | Regex::NonCapturingGroup(inner)
                | Regex::NamedGroup(_, inner) => stack.push(inner),
                Regex::FlagsGroup {
                    inner: Some(inner), ..
                } => stack.push(inner),
                _ => return Err(invalid_literal_replacement_error()),
            }
        }

        Ok(len)
    }

    fn push_regex_literal_chars(regex: &Regex, chars: &mut Vec<char>) -> ParseResult<()> {
        let mut stack = Vec::with_capacity(8);
        stack.push(regex);

        while let Some(node) = stack.pop() {
            match node {
                Regex::Empty | Regex::FlagsGroup { inner: None, .. } => {}
                Regex::Char(c) => chars.push(*c),
                Regex::Concat(a, b) => {
                    stack.push(b);
                    stack.push(a);
                }
                Regex::CapturingGroup(_, inner)
                | Regex::NonCapturingGroup(inner)
                | Regex::NamedGroup(_, inner) => stack.push(inner),
                Regex::FlagsGroup {
                    inner: Some(inner), ..
                } => stack.push(inner),
                _ => return Err(invalid_literal_replacement_error()),
            }
        }

        Ok(())
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
        let prepared = prepare_byte_regex_for_compile(regex)?;
        let mut nfa = self.compile_regex(prepared.as_ref())?;
        // H9: Finalize CSR transition table
        nfa.finalize();
        Ok(nfa)
    }

    /// Compile a rewrite rule.
    pub fn compile_rewrite(&mut self, regex: &RegexByte) -> ParseResult<CompiledRewrite> {
        let prepared = prepare_byte_regex_for_compile(regex)?;
        match prepared.as_ref() {
            RegexByte::RewriteRule {
                pattern,
                replacement,
                context,
                weight,
            } => {
                let mut source = self.compile_regex(pattern)?;
                // H9: Finalize CSR transition table
                source.finalize();
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
                let mut nfa = self.compile_regex(regex)?;
                // H9: Finalize CSR transition table
                nfa.finalize();
                Ok(ContextPattern::Nfa(nfa))
            }
            ContextExprByte::WordBoundary => Ok(ContextPattern::Boundary(BoundaryKind::WordStart)),
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
    #[allow(deprecated)]
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
            // Capturing group: (...)
            RegexByte::CapturingGroup(_, inner) => self.compile_regex(inner),
            // Non-capturing group: (?:...)
            RegexByte::NonCapturingGroup(inner) => self.compile_regex(inner),
            // Named group: (?<name>...)
            RegexByte::NamedGroup(_, inner) => self.compile_regex(inner),
            // Group reference: (?&name)
            RegexByte::GroupRef(name) => Err(ParseError::new(
                ParseErrorKind::InvalidGroupReference(format!(
                    "unexpanded group reference (?&{}) reached byte NFA compilation",
                    name
                )),
                Position::start(),
            )),
            // Flags group: (?flags:...) or (?flags)
            RegexByte::FlagsGroup { inner, .. } => match inner {
                Some(inner_regex) => self.compile_regex(inner_regex),
                None => Ok(self.builder.epsilon()),
            },
            RegexByte::WordBoundary => Ok(self.builder.epsilon()),
            // Anchor assertions (zero-width)
            RegexByte::StartOfLine => Ok(self.builder.start_of_line()),
            RegexByte::EndOfLine => Ok(self.builder.end_of_line()),
            RegexByte::StartOfInput => Ok(self.builder.start_of_input()),
            RegexByte::EndOfInput => Ok(self.builder.end_of_input()),
            RegexByte::EndOfInputStrict => Ok(self.builder.end_of_input_strict()),
            RegexByte::RewriteRule { pattern, .. } => self.compile_regex(pattern),
        }
    }

    /// Convert a regex to a literal byte string (for replacement).
    #[allow(deprecated)]
    fn regex_to_literal(&self, regex: &RegexByte) -> ParseResult<Vec<u8>> {
        let mut bytes = Vec::with_capacity(Self::literal_byte_len(regex)?);
        Self::push_regex_literal_bytes(regex, &mut bytes)?;
        Ok(bytes)
    }

    fn literal_byte_len(regex: &RegexByte) -> ParseResult<usize> {
        let mut len = 0;
        let mut stack = Vec::with_capacity(8);
        stack.push(regex);

        while let Some(node) = stack.pop() {
            match node {
                RegexByte::Empty | RegexByte::FlagsGroup { inner: None, .. } => {}
                RegexByte::Byte(_) => increment_literal_len(&mut len)?,
                RegexByte::Concat(a, b) => {
                    stack.push(b);
                    stack.push(a);
                }
                RegexByte::CapturingGroup(_, inner)
                | RegexByte::NonCapturingGroup(inner)
                | RegexByte::NamedGroup(_, inner) => stack.push(inner),
                RegexByte::FlagsGroup {
                    inner: Some(inner), ..
                } => stack.push(inner),
                _ => return Err(invalid_literal_replacement_error()),
            }
        }

        Ok(len)
    }

    fn push_regex_literal_bytes(regex: &RegexByte, bytes: &mut Vec<u8>) -> ParseResult<()> {
        let mut stack = Vec::with_capacity(8);
        stack.push(regex);

        while let Some(node) = stack.pop() {
            match node {
                RegexByte::Empty | RegexByte::FlagsGroup { inner: None, .. } => {}
                RegexByte::Byte(b) => bytes.push(*b),
                RegexByte::Concat(a, b) => {
                    stack.push(b);
                    stack.push(a);
                }
                RegexByte::CapturingGroup(_, inner)
                | RegexByte::NonCapturingGroup(inner)
                | RegexByte::NamedGroup(_, inner) => stack.push(inner),
                RegexByte::FlagsGroup {
                    inner: Some(inner), ..
                } => stack.push(inner),
                _ => return Err(invalid_literal_replacement_error()),
            }
        }

        Ok(())
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
    fn test_trampolined_operand_underflow_returns_error() {
        let mut stack = Vec::new();
        let Err(err) = NFACompilerChar::pop_trampolined_value(&mut stack, "test operand") else {
            panic!("empty stack should produce an internal compile error");
        };

        assert!(matches!(
            err.kind,
            ParseErrorKind::InternalError(ref message)
                if message.contains("test operand")
                    && message.contains("trampolined NFA compilation")
        ));
    }

    #[test]
    fn test_compile_literal() {
        let regex = parse("phone").expect("test: parse phone");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.accepts("phone"));
        assert!(!nfa.accepts("fone"));
    }

    #[test]
    fn test_compile_alternation() {
        let regex = parse("ph|f").expect("test: parse ph|f");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.accepts("ph"));
        assert!(nfa.accepts("f"));
        assert!(!nfa.accepts("g"));
    }

    #[test]
    fn test_compile_group() {
        let regex = parse("(ph|f)one").expect("test: parse (ph|f)one");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.accepts("phone"));
        assert!(nfa.accepts("fone"));
        assert!(!nfa.accepts("bone"));
    }

    #[test]
    fn test_prepare_regex_without_groups_borrows_original() {
        let regex = parse("(ab|cd)+").expect("test: parse regex without named groups");
        let prepared = prepare_regex_for_compile(&regex).expect("test: prepare regex");

        assert!(matches!(prepared, Cow::Borrowed(_)));
    }

    #[test]
    fn test_prepare_named_groups_without_refs_borrows_and_validates() {
        let regex = Regex::Alt(
            Box::new(Regex::named_group("left", Regex::Char('a'))),
            Box::new(Regex::named_group("right", Regex::Char('b'))),
        );
        let prepared =
            prepare_regex_for_compile(&regex).expect("test: prepare regex with named groups");
        assert!(matches!(prepared, Cow::Borrowed(_)));

        let duplicate = Regex::Alt(
            Box::new(Regex::named_group("dup", Regex::Char('a'))),
            Box::new(Regex::named_group("dup", Regex::Char('b'))),
        );
        let err = prepare_regex_for_compile(&duplicate)
            .expect_err("duplicate named groups should still fail without references");
        assert!(matches!(err.kind, ParseErrorKind::DuplicateGroupName(_)));
    }

    #[test]
    fn test_prepare_regex_with_group_refs_owns_expansion() {
        let regex = parse("(?<digit>[0-9])(?&digit)").expect("test: parse named group reference");
        let prepared = prepare_regex_for_compile(&regex).expect("test: prepare group reference");

        assert!(matches!(prepared, Cow::Owned(_)));
    }

    #[test]
    fn test_compile_named_group_reference() {
        let regex = parse("(?<digit>[0-9])(?&digit)").expect("test: parse named group reference");
        let nfa = compile(&regex).expect("test: compile named group reference");

        assert!(nfa.accepts("42"));
        assert!(nfa.accepts("00"));
        assert!(!nfa.accepts("4"));
        assert!(!nfa.accepts("4a"));
    }

    #[test]
    fn test_compile_forward_named_group_reference() {
        let regex = parse("(?&pair)(?<pair>ab|cd)").expect("test: parse forward group reference");
        let nfa = compile(&regex).expect("test: compile forward group reference");

        assert!(nfa.accepts("abab"));
        assert!(nfa.accepts("cdab"));
        assert!(!nfa.accepts("ab"));
        assert!(!nfa.accepts("abcdx"));
    }

    #[test]
    fn test_compile_named_group_reference_with_trampolining() {
        let regex =
            parse("(?<unit>ab|cd)(?&unit)+").expect("test: parse named group reference with plus");
        let mut compiler = NFACompilerChar::new().with_trampolining();
        let nfa = compiler
            .compile(&regex)
            .expect("test: trampolined compile named group reference");

        assert!(nfa.accepts("abab"));
        assert!(nfa.accepts("abcdcd"));
        assert!(!nfa.accepts("ab"));
        assert!(!nfa.accepts("abef"));
    }

    #[test]
    fn test_compile_named_group_reference_cycle_error() {
        let regex = Regex::NamedGroup("self".to_string(), Box::new(Regex::group_ref("self")));
        let err = compile(&regex).expect_err("recursive group reference should fail");

        assert!(matches!(err.kind, ParseErrorKind::InvalidGroupReference(_)));
    }

    #[test]
    fn test_compile_manual_undefined_group_reference_error() {
        let err = compile(&Regex::group_ref("missing"))
            .expect_err("undefined manual group reference should fail");

        assert!(matches!(
            err.kind,
            ParseErrorKind::UndefinedGroupReference(_)
        ));
    }

    #[test]
    fn test_compile_star() {
        let regex = parse("a*").expect("test: parse a*");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("aaa"));
        assert!(!nfa.accepts("b"));
    }

    #[test]
    fn test_compile_plus() {
        let regex = parse("a+").expect("test: parse a+");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(!nfa.accepts(""));
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("aaa"));
    }

    #[test]
    fn test_compile_optional() {
        let regex = parse("a?b").expect("test: parse a?b");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.accepts("b"));
        assert!(nfa.accepts("ab"));
        assert!(!nfa.accepts("aab"));
    }

    #[test]
    fn test_compile_char_class() {
        let regex = parse("[aeiou]").expect("test: parse [aeiou]");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("e"));
        assert!(!nfa.accepts("b"));
    }

    #[test]
    fn test_compile_any() {
        let regex = parse("a.c").expect("test: parse a.c");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.accepts("abc"));
        assert!(nfa.accepts("axc"));
        assert!(!nfa.accepts("ac"));
    }

    #[test]
    fn test_compile_repeat_exact() {
        let regex = parse("a{3}").expect("test: parse a{3}");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(!nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(!nfa.accepts("aaaa"));
    }

    #[test]
    fn test_compile_repeat_range() {
        let regex = parse("a{2,4}").expect("test: parse a{2,4}");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(!nfa.accepts("a"));
        assert!(nfa.accepts("aa"));
        assert!(nfa.accepts("aaa"));
        assert!(nfa.accepts("aaaa"));
        assert!(!nfa.accepts("aaaaa"));
    }

    #[test]
    fn test_compile_rewrite_rule_simple() {
        let regex = parse_rule("ph -> f").expect("test: parse rule ph -> f");
        let rewrite = compile_rewrite(&regex).expect("test: compile_rewrite");
        assert!(rewrite.source.accepts("ph"));
        assert_eq!(rewrite.replacement, vec!['f']);
        assert!(rewrite.left_context.is_none());
        assert!(rewrite.right_context.is_none());
    }

    #[test]
    fn test_compile_rewrite_rule_with_context() {
        let regex = parse_rule("c -> s / _[ei]").expect("test: parse rule c -> s / _[ei]");
        let rewrite = compile_rewrite(&regex).expect("test: compile_rewrite");
        assert!(rewrite.source.accepts("c"));
        assert_eq!(rewrite.replacement, vec!['s']);
        assert!(rewrite.left_context.is_none());
        assert!(rewrite.right_context.is_some());
        let right = rewrite
            .right_context
            .expect("test: right_context is_some checked above");
        assert!(right.accepts("e"));
        assert!(right.accepts("i"));
        assert!(!right.accepts("a"));
    }

    #[test]
    fn test_compile_rewrite_rule_empty_replacement() {
        let regex = parse_rule("e -> / _#").expect("test: parse rule e -> / _#");
        let rewrite = compile_rewrite(&regex).expect("test: compile_rewrite");
        assert!(rewrite.source.accepts("e"));
        assert!(rewrite.replacement.is_empty());
    }

    #[test]
    fn test_compile_rewrite_rule_long_literal_replacement() {
        let replacement = "abcdefghijklmnopqrstuvwxyz".repeat(8);
        let rule = format!("x -> {replacement}");
        let regex = parse_rule(&rule).expect("test: parse long literal replacement rule");
        let rewrite = compile_rewrite(&regex).expect("test: compile_rewrite");

        assert!(rewrite.source.accepts("x"));
        assert_eq!(rewrite.replacement, replacement.chars().collect::<Vec<_>>());
    }

    #[test]
    fn test_compile_complex_pattern() {
        let regex = parse("(ph|f)one[s]?").expect("test: parse (ph|f)one[s]?");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.accepts("phone"));
        assert!(nfa.accepts("phones"));
        assert!(nfa.accepts("fone"));
        assert!(nfa.accepts("fones"));
        assert!(!nfa.accepts("bone"));
    }

    // Byte-level tests

    #[test]
    fn test_compile_bytes_literal() {
        let regex = crate::phonetic::regex::parse_bytes(b"phone").expect("test: parse_bytes phone");
        let nfa = compile_bytes(&regex).expect("test: compile_bytes nfa");
        assert!(nfa.accepts(b"phone"));
        assert!(!nfa.accepts(b"fone"));
    }

    #[test]
    fn test_compile_bytes_alternation() {
        let regex = crate::phonetic::regex::parse_bytes(b"ph|f").expect("test: parse_bytes ph|f");
        let nfa = compile_bytes(&regex).expect("test: compile_bytes nfa");
        assert!(nfa.accepts(b"ph"));
        assert!(nfa.accepts(b"f"));
    }

    #[test]
    fn test_prepare_byte_regex_without_group_refs_borrows_original() {
        let regex = RegexByte::Concat(
            Box::new(RegexByte::Byte(b'a')),
            Box::new(RegexByte::Byte(b'b')),
        );
        let prepared =
            prepare_byte_regex_for_compile(&regex).expect("test: prepare byte regex without refs");

        assert!(matches!(prepared, Cow::Borrowed(_)));
    }

    #[test]
    fn test_compile_bytes_named_group_reference() {
        let regex = RegexByte::Concat(
            Box::new(RegexByte::NamedGroup(
                "x".to_string(),
                Box::new(RegexByte::Byte(b'a')),
            )),
            Box::new(RegexByte::GroupRef("x".to_string())),
        );
        let nfa = compile_bytes(&regex).expect("test: compile byte group reference");

        assert!(nfa.accepts(b"aa"));
        assert!(!nfa.accepts(b"a"));
        assert!(!nfa.accepts(b"ab"));
    }

    #[test]
    fn test_compile_bytes_rewrite() {
        let regex = crate::phonetic::regex::parse_rule_bytes(b"ph -> f")
            .expect("test: parse_rule_bytes ph -> f");
        let rewrite = compile_rewrite_bytes(&regex).expect("test: compile_rewrite_bytes");
        assert!(rewrite.source.accepts(b"ph"));
        assert_eq!(rewrite.replacement, vec![b'f']);
    }

    #[test]
    fn test_compile_bytes_rewrite_long_literal_replacement() {
        let replacement = b"abcdefghijklmnopqrstuvwxyz".repeat(8);
        let mut rule = b"x -> ".to_vec();
        rule.extend_from_slice(&replacement);
        let regex = crate::phonetic::regex::parse_rule_bytes(&rule)
            .expect("test: parse_rule_bytes long literal replacement");
        let rewrite = compile_rewrite_bytes(&regex).expect("test: compile_rewrite_bytes");

        assert!(rewrite.source.accepts(b"x"));
        assert_eq!(rewrite.replacement, replacement);
    }

    // --- Anchor tests ---

    #[test]
    fn test_compile_start_of_line() {
        let regex = parse("^hello").expect("test: parse ^hello");
        let nfa = compile(&regex).expect("test: compile nfa");
        // NFA should have anchor transition followed by literal
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_compile_end_of_line() {
        let regex = parse("hello$").expect("test: parse hello$");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_compile_anchored_pattern() {
        let regex = parse("^hello$").expect("test: parse ^hello$");
        let nfa = compile(&regex).expect("test: compile nfa");
        // start_of_line + hello (5 chars) + end_of_line
        assert!(nfa.state_count() >= 4);
    }

    #[test]
    fn test_compile_start_of_input() {
        let regex = parse(r"\Ahello").expect("test: parse \\Ahello");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_compile_end_of_input() {
        let regex = parse(r"hello\Z").expect("test: parse hello\\Z");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_compile_strict_end_of_input() {
        let regex = parse(r"hello\z").expect("test: parse hello\\z");
        let nfa = compile(&regex).expect("test: compile nfa");
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_compile_multiline_flags() {
        // Pattern with multiline flag
        let regex = parse("(?m)^line$").expect("test: parse (?m)^line$");
        let nfa = compile(&regex).expect("test: compile nfa");
        // Should compile with anchor transitions
        assert!(nfa.state_count() >= 3);
    }

    #[test]
    fn test_trampolining_api() {
        // Test that the trampolining API works correctly
        let mut compiler = NFACompilerChar::new();
        assert!(!compiler.is_trampolining(), "default should be recursive");

        compiler.set_trampolining(true);
        assert!(
            compiler.is_trampolining(),
            "should be trampolining after set"
        );

        compiler.set_trampolining(false);
        assert!(
            !compiler.is_trampolining(),
            "should be recursive after unset"
        );

        let compiler2 = NFACompilerChar::new().with_trampolining();
        assert!(
            compiler2.is_trampolining(),
            "builder should enable trampolining"
        );
    }

    #[test]
    fn test_trampolined_matches_recursive_simple() {
        // Simple patterns should produce equivalent NFAs
        let patterns = ["a", "ab", "a|b", "a*", "a+", "a?", "[aeiou]"];

        for pattern in patterns {
            let regex = parse(pattern).expect("parse");

            // Compile with recursive (default)
            let nfa_recursive = compile(&regex).expect("recursive compile");

            // Compile with trampolining
            let mut compiler = NFACompilerChar::new().with_trampolining();
            let nfa_trampolined = compiler.compile(&regex).expect("trampolined compile");

            // Both should have the same state count
            assert_eq!(
                nfa_recursive.state_count(),
                nfa_trampolined.state_count(),
                "state count mismatch for pattern '{}'",
                pattern
            );

            // Both should accept/reject the same inputs
            let test_inputs = ["", "a", "b", "ab", "aaa", "aeiou"];
            for input in test_inputs {
                assert_eq!(
                    nfa_recursive.accepts(input),
                    nfa_trampolined.accepts(input),
                    "behavior mismatch for pattern '{}' on input '{}'",
                    pattern,
                    input
                );
            }
        }
    }

    #[test]
    fn test_trampolined_matches_recursive_complex() {
        // More complex patterns
        let patterns = [
            "(ab)+",
            "a{2,5}",
            "(?:a|b)*c",
            "(ph|f)one",
            "[a-z]+",
            "a{3}",
        ];

        for pattern in patterns {
            let regex = parse(pattern).expect("parse");

            let nfa_recursive = compile(&regex).expect("recursive compile");

            let mut compiler = NFACompilerChar::new().with_trampolining();
            let nfa_trampolined = compiler.compile(&regex).expect("trampolined compile");

            assert_eq!(
                nfa_recursive.state_count(),
                nfa_trampolined.state_count(),
                "state count mismatch for pattern '{}'",
                pattern
            );
        }
    }

    #[test]
    fn test_deep_nesting_with_trampolining() {
        // Build deeply nested pattern: ((((((((((a))))))))))
        // Note: The depth is limited by the parser's recursion, not the compiler.
        // The trampolining helps with compilation, but parsing is still recursive.
        let depth = 50; // Parser can handle this depth
        let mut pattern = "a".to_string();
        for _ in 0..depth {
            pattern = format!("({})", pattern);
        }

        let regex = parse(&pattern).expect("parse deeply nested pattern");

        // Compile with trampolining
        let mut compiler = NFACompilerChar::new().with_trampolining();
        let nfa = compiler
            .compile(&regex)
            .expect("trampolined compile of deep pattern");

        // Verify it works correctly
        assert!(nfa.accepts("a"), "should accept 'a'");
        assert!(!nfa.accepts("b"), "should reject 'b'");
        assert!(!nfa.accepts("aa"), "should reject 'aa'");

        // Also verify recursive compilation works at this depth
        let nfa_recursive = compile(&regex).expect("recursive compile");
        assert_eq!(nfa.state_count(), nfa_recursive.state_count());
    }

    #[test]
    fn test_deep_alternation_with_trampolining() {
        // Build deeply nested alternation: (((a|b)|c)|d)...
        // Parser depth limited
        let depth = 40;
        let mut pattern = "a".to_string();
        for i in 0..depth {
            let c = char::from(b'a' + ((i + 1) % 26) as u8);
            pattern = format!("({}|{})", pattern, c);
        }

        let regex = parse(&pattern).expect("parse deeply nested alternation");

        let mut compiler = NFACompilerChar::new().with_trampolining();
        let nfa = compiler
            .compile(&regex)
            .expect("trampolined compile of deep alternation");

        // Should accept any single letter
        assert!(nfa.accepts("a"));
        assert!(nfa.accepts("b"));
        assert!(nfa.accepts("c"));

        // Verify equivalence with recursive
        let nfa_recursive = compile(&regex).expect("recursive compile");
        assert_eq!(nfa.state_count(), nfa_recursive.state_count());
    }

    #[test]
    fn test_deep_concat_with_trampolining() {
        // Build deeply concatenated pattern via groups
        // Parser depth limited
        let depth = 100;
        let mut pattern = String::new();
        for _ in 0..depth {
            pattern.push_str("(?:a)");
        }

        let regex = parse(&pattern).expect("parse deeply concatenated pattern");

        let mut compiler = NFACompilerChar::new().with_trampolining();
        let nfa = compiler
            .compile(&regex)
            .expect("trampolined compile of deep concat");

        // Should accept exactly 'depth' 'a' characters
        let expected = "a".repeat(depth);
        assert!(nfa.accepts(&expected), "should accept {} 'a' chars", depth);

        let too_short = "a".repeat(depth - 1);
        assert!(
            !nfa.accepts(&too_short),
            "should reject {} 'a' chars",
            depth - 1
        );

        // Verify equivalence with recursive
        let nfa_recursive = compile(&regex).expect("recursive compile");
        assert_eq!(nfa.state_count(), nfa_recursive.state_count());
    }

    #[test]
    fn test_recursive_mode_still_works() {
        // Verify recursive mode (default) still works for normal patterns
        let patterns = ["hello", "world|earth", "a+b*c?", "[0-9]+"];

        for pattern in patterns {
            let regex = parse(pattern).expect("parse");

            // Default is recursive
            let nfa = compile(&regex).expect("recursive compile");
            assert!(nfa.state_count() > 0);
        }
    }
}
