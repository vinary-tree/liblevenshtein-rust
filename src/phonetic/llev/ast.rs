//! Abstract Syntax Tree for `.llev` rule files.
//!
//! This module defines the AST nodes for representing parsed `.llev` files.
//! The AST captures file-level structure including directives, metadata,
//! and rule definitions.
//!
//! # Structure
//!
//! A `.llev` file consists of:
//! - File metadata (`@name`, `@version`, `@author`, `@description`)
//! - Symbol definitions (`@define`)
//! - Include directives (`@include`)
//! - Rule definitions with optional metadata
//!
//! # Example `.llev` File
//!
//! ```text
//! # English Phonetic Rules
//! @name "English Phonetic Rules"
//! @version "1.0"
//!
//! @define FRONT_VOWEL = [ei]
//!
//! [id: 1, name: "ph to f"]
//! ph -> f;
//!
//! [id: 20, name: "soft c", weight: 0.0]
//! c -> s / _FRONT_VOWEL;
//! ```

use std::fmt;
use std::path::PathBuf;

use super::error::Position;

// ============================================================================
// File-level AST
// ============================================================================

/// A complete `.llev` file AST.
///
/// Represents the top-level structure of a parsed `.llev` file, containing
/// metadata, directives, and rule definitions.
#[derive(Debug, Clone, PartialEq)]
pub struct LLevFile {
    /// File metadata (name, version, author, description)
    pub metadata: FileMetadata,

    /// Symbol definitions from `@define` directives
    pub symbols: Vec<SymbolDef>,

    /// Include directives (paths to other `.llev` files)
    pub includes: Vec<IncludeDirective>,

    /// Rule definitions
    pub rules: Vec<RuleDefinition>,

    /// Source file path (if loaded from file)
    pub source_file: Option<PathBuf>,

    /// Resolved include paths (after loading)
    pub resolved_includes: Vec<PathBuf>,
}

impl LLevFile {
    /// Create a new empty `.llev` file AST.
    pub fn new() -> Self {
        Self {
            metadata: FileMetadata::default(),
            symbols: Vec::new(),
            includes: Vec::new(),
            rules: Vec::new(),
            source_file: None,
            resolved_includes: Vec::new(),
        }
    }

    /// Create a new `.llev` file AST with a source path.
    pub fn with_source(source: PathBuf) -> Self {
        Self {
            source_file: Some(source),
            ..Self::new()
        }
    }

    /// Check if the file has any rules defined.
    pub fn has_rules(&self) -> bool {
        !self.rules.is_empty()
    }

    /// Get the number of rules in the file.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Check if the file has any includes.
    pub fn has_includes(&self) -> bool {
        !self.includes.is_empty()
    }

    /// Check if the file has any symbol definitions.
    pub fn has_symbols(&self) -> bool {
        !self.symbols.is_empty()
    }

    /// Merge another `.llev` file into this one.
    ///
    /// This appends rules, symbols, and resolved includes from the other file.
    /// Metadata from the other file is ignored (the current file's metadata is kept).
    pub fn merge(&mut self, other: LLevFile) {
        // Merge symbols (append)
        self.symbols.extend(other.symbols);

        // Merge rules (append)
        self.rules.extend(other.rules);

        // Merge resolved includes
        self.resolved_includes.extend(other.resolved_includes);
    }
}

impl Default for LLevFile {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Metadata
// ============================================================================

/// File-level metadata from `@name`, `@version`, etc. directives.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FileMetadata {
    /// Human-readable name (`@name "..."`)
    pub name: Option<String>,

    /// Version string (`@version "..."`)
    pub version: Option<String>,

    /// Author information (`@author "..."`)
    pub author: Option<String>,

    /// Description (`@description "..."`)
    pub description: Option<String>,
}

impl FileMetadata {
    /// Create new empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if all metadata fields are empty.
    pub fn is_empty(&self) -> bool {
        self.name.is_none()
            && self.version.is_none()
            && self.author.is_none()
            && self.description.is_none()
    }
}

// ============================================================================
// Directives
// ============================================================================

/// An `@include` directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncludeDirective {
    /// Path to the file to include (may be relative)
    pub path: String,

    /// Position of the directive in the source file
    pub position: Position,
}

impl IncludeDirective {
    /// Create a new include directive.
    pub fn new(path: String, position: Position) -> Self {
        Self { path, position }
    }
}

/// A symbol definition from `@define`.
///
/// Symbols can be used in patterns and replacements by name.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolDef {
    /// Symbol name (e.g., "FRONT_VOWEL")
    pub name: String,

    /// Symbol value as a regex expression
    pub value: Expression,

    /// Position of the definition in the source file
    pub position: Position,
}

impl SymbolDef {
    /// Create a new symbol definition.
    pub fn new(name: String, value: Expression, position: Position) -> Self {
        Self {
            name,
            value,
            position,
        }
    }
}

// ============================================================================
// Rule Definitions
// ============================================================================

/// A complete rule definition with optional metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleDefinition {
    /// Rule metadata (id, name, weight, group, enabled)
    pub metadata: RuleMetadata,

    /// The rewrite rule itself
    pub rule: RewriteRuleAST,

    /// Position of the rule in the source file
    pub position: Position,
}

impl RuleDefinition {
    /// Create a new rule definition.
    pub fn new(metadata: RuleMetadata, rule: RewriteRuleAST, position: Position) -> Self {
        Self {
            metadata,
            rule,
            position,
        }
    }

    /// Create a rule definition with default metadata.
    pub fn simple(rule: RewriteRuleAST, position: Position) -> Self {
        Self {
            metadata: RuleMetadata::default(),
            rule,
            position,
        }
    }
}

/// Metadata for a rule definition.
///
/// Parsed from metadata blocks like `[id: 1, name: "ph to f", weight: 0.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct RuleMetadata {
    /// Unique rule identifier
    pub id: Option<usize>,

    /// Human-readable name
    pub name: Option<String>,

    /// Weight/cost for phonetic distance (lower = more likely)
    pub weight: Option<f64>,

    /// Group name for organizing related rules
    pub group: Option<String>,

    /// Whether the rule is enabled (default: true)
    pub enabled: bool,
}

impl RuleMetadata {
    /// Create new empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create metadata with an ID.
    pub fn with_id(id: usize) -> Self {
        Self {
            id: Some(id),
            ..Self::default()
        }
    }

    /// Create metadata with an ID and name.
    pub fn with_id_name(id: usize, name: impl Into<String>) -> Self {
        Self {
            id: Some(id),
            name: Some(name.into()),
            ..Self::default()
        }
    }
}

impl Default for RuleMetadata {
    fn default() -> Self {
        Self {
            id: None,
            name: None,
            weight: None,
            group: None,
            enabled: true,
        }
    }
}

// ============================================================================
// Rewrite Rule AST
// ============================================================================

/// A rewrite rule: `pattern -> replacement context? weight?`
#[derive(Debug, Clone, PartialEq)]
pub struct RewriteRuleAST {
    /// Pattern to match
    pub pattern: Expression,

    /// Replacement expression (empty for deletion)
    pub replacement: Expression,

    /// Optional context predicate
    pub context: Option<ContextAST>,

    /// Optional inline weight suffix `[0.15]`
    pub weight: Option<f64>,
}

impl RewriteRuleAST {
    /// Create a new rewrite rule.
    pub fn new(
        pattern: Expression,
        replacement: Expression,
        context: Option<ContextAST>,
        weight: Option<f64>,
    ) -> Self {
        Self {
            pattern,
            replacement,
            context,
            weight,
        }
    }

    /// Create a simple rewrite rule without context or weight.
    pub fn simple(pattern: Expression, replacement: Expression) -> Self {
        Self {
            pattern,
            replacement,
            context: None,
            weight: None,
        }
    }

    /// Create a deletion rule (pattern -> empty).
    pub fn deletion(pattern: Expression, context: Option<ContextAST>) -> Self {
        Self {
            pattern,
            replacement: Expression::Empty,
            context,
            weight: None,
        }
    }
}

/// Context predicate: `/ left_context? _ right_context? syllable_clause?`
#[derive(Debug, Clone, PartialEq)]
pub struct ContextAST {
    /// Left context (lookbehind) - what must precede the match
    pub left: Option<Box<ContextExpr>>,

    /// Right context (lookahead) - what must follow the match
    pub right: Option<Box<ContextExpr>>,

    /// Optional syllable condition (`if monosyllable`, etc.)
    pub syllable: Option<SyllableExpr>,
}

// ============================================================================
// Context Expression (with compound operators)
// ============================================================================

/// A context expression with support for compound (And/Or/Not) operators.
///
/// This extends simple `Expression` with logical combinators for more
/// expressive context specifications.
///
/// # Examples
///
/// ```text
/// [aeiou]          -> Pattern (simple char class)
/// #                -> WordBoundary
/// [aeiou] & [bcdf] -> And (both must match)
/// [aeiou] | #      -> Or (either must match)
/// ![aeiou]         -> Not (must NOT match)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ContextExpr {
    /// Simple pattern match
    Pattern(Expression),

    /// Word boundary marker (`#`)
    WordBoundary,

    /// Compound: both must match (`a & b`)
    And(Box<ContextExpr>, Box<ContextExpr>),

    /// Compound: either must match (`a | b`)
    Or(Box<ContextExpr>, Box<ContextExpr>),

    /// Negation: must NOT match (`!a`)
    Not(Box<ContextExpr>),
}

impl ContextExpr {
    /// Create a pattern context expression.
    pub fn pattern(expr: Expression) -> Self {
        ContextExpr::Pattern(expr)
    }

    /// Create a word boundary context expression.
    pub fn word_boundary() -> Self {
        ContextExpr::WordBoundary
    }

    /// Create an AND context expression.
    pub fn and(a: ContextExpr, b: ContextExpr) -> Self {
        ContextExpr::And(Box::new(a), Box::new(b))
    }

    /// Create an OR context expression.
    pub fn or(a: ContextExpr, b: ContextExpr) -> Self {
        ContextExpr::Or(Box::new(a), Box::new(b))
    }

    /// Create a NOT context expression.
    pub fn not(inner: ContextExpr) -> Self {
        ContextExpr::Not(Box::new(inner))
    }

    /// Convert from a simple Expression.
    pub fn from_expression(expr: Expression) -> Self {
        match expr {
            Expression::WordBoundary => ContextExpr::WordBoundary,
            other => ContextExpr::Pattern(other),
        }
    }

    /// Check if this is a simple pattern (no compound operators).
    pub fn is_simple(&self) -> bool {
        matches!(self, ContextExpr::Pattern(_) | ContextExpr::WordBoundary)
    }

    /// Get the inner expression if this is a simple pattern.
    pub fn as_expression(&self) -> Option<&Expression> {
        match self {
            ContextExpr::Pattern(expr) => Some(expr),
            _ => None,
        }
    }
}

impl fmt::Display for ContextExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextExpr::Pattern(expr) => write!(f, "{}", expr),
            ContextExpr::WordBoundary => write!(f, "#"),
            ContextExpr::And(a, b) => write!(f, "({} & {})", a, b),
            ContextExpr::Or(a, b) => write!(f, "({} | {})", a, b),
            ContextExpr::Not(inner) => write!(f, "!{}", inner),
        }
    }
}

// ============================================================================
// Syllable Conditions
// ============================================================================

/// Syllable-based conditions for context-sensitive rules.
///
/// These conditions allow rules to apply based on syllable structure,
/// enabling correct handling of vowel length, Y pronunciation, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyllableCondition {
    /// Word has exactly 1 syllable (e.g., "ply", "fly")
    Monosyllable,

    /// Word has more than 1 syllable (e.g., "happy", "flying")
    Polysyllable,

    /// Current syllable ends in vowel (long vowel context)
    OpenSyllable,

    /// Current syllable ends in consonant (short vowel context)
    ClosedSyllable,

    /// Match is in the last syllable
    FinalSyllable,

    /// Match is in the first syllable
    InitialSyllable,
}

impl SyllableCondition {
    /// Parse a syllable condition from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "monosyllable" => Some(SyllableCondition::Monosyllable),
            "polysyllable" => Some(SyllableCondition::Polysyllable),
            "open_syllable" => Some(SyllableCondition::OpenSyllable),
            "closed_syllable" => Some(SyllableCondition::ClosedSyllable),
            "final_syllable" => Some(SyllableCondition::FinalSyllable),
            "initial_syllable" => Some(SyllableCondition::InitialSyllable),
            _ => None,
        }
    }

    /// Get the string representation of this condition.
    pub fn as_str(&self) -> &'static str {
        match self {
            SyllableCondition::Monosyllable => "monosyllable",
            SyllableCondition::Polysyllable => "polysyllable",
            SyllableCondition::OpenSyllable => "open_syllable",
            SyllableCondition::ClosedSyllable => "closed_syllable",
            SyllableCondition::FinalSyllable => "final_syllable",
            SyllableCondition::InitialSyllable => "initial_syllable",
        }
    }
}

impl fmt::Display for SyllableCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Syllable expression with compound operators.
///
/// Allows combining syllable conditions with And/Or/Not.
///
/// # Examples
///
/// ```text
/// monosyllable                     -> Cond (simple condition)
/// polysyllable & final_syllable    -> And (both must be true)
/// monosyllable | !final_syllable   -> Or with negation
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum SyllableExpr {
    /// Simple syllable condition
    Cond(SyllableCondition),

    /// Both conditions must be true
    And(Box<SyllableExpr>, Box<SyllableExpr>),

    /// Either condition must be true
    Or(Box<SyllableExpr>, Box<SyllableExpr>),

    /// Condition must NOT be true
    Not(Box<SyllableExpr>),
}

impl SyllableExpr {
    /// Create a simple condition expression.
    pub fn cond(condition: SyllableCondition) -> Self {
        SyllableExpr::Cond(condition)
    }

    /// Create an AND expression.
    pub fn and(a: SyllableExpr, b: SyllableExpr) -> Self {
        SyllableExpr::And(Box::new(a), Box::new(b))
    }

    /// Create an OR expression.
    pub fn or(a: SyllableExpr, b: SyllableExpr) -> Self {
        SyllableExpr::Or(Box::new(a), Box::new(b))
    }

    /// Create a NOT expression.
    pub fn not(inner: SyllableExpr) -> Self {
        SyllableExpr::Not(Box::new(inner))
    }

    /// Check if this is a simple condition (no compound operators).
    pub fn is_simple(&self) -> bool {
        matches!(self, SyllableExpr::Cond(_))
    }

    /// Get the inner condition if this is simple.
    pub fn as_condition(&self) -> Option<SyllableCondition> {
        match self {
            SyllableExpr::Cond(c) => Some(*c),
            _ => None,
        }
    }
}

impl fmt::Display for SyllableExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyllableExpr::Cond(c) => write!(f, "{}", c),
            SyllableExpr::And(a, b) => write!(f, "({} & {})", a, b),
            SyllableExpr::Or(a, b) => write!(f, "({} | {})", a, b),
            SyllableExpr::Not(inner) => write!(f, "!{}", inner),
        }
    }
}

impl ContextAST {
    /// Create a new context with left and right predicates (from expressions).
    pub fn new(left: Option<Expression>, right: Option<Expression>) -> Self {
        Self {
            left: left.map(|e| Box::new(ContextExpr::from_expression(e))),
            right: right.map(|e| Box::new(ContextExpr::from_expression(e))),
            syllable: None,
        }
    }

    /// Create a new context with ContextExpr left and right predicates.
    pub fn new_expr(left: Option<ContextExpr>, right: Option<ContextExpr>) -> Self {
        Self {
            left: left.map(Box::new),
            right: right.map(Box::new),
            syllable: None,
        }
    }

    /// Create a new context with syllable condition.
    pub fn new_with_syllable(
        left: Option<ContextExpr>,
        right: Option<ContextExpr>,
        syllable: Option<SyllableExpr>,
    ) -> Self {
        Self {
            left: left.map(Box::new),
            right: right.map(Box::new),
            syllable,
        }
    }

    /// Create a lookahead context (only right side).
    pub fn lookahead(right: Expression) -> Self {
        Self {
            left: None,
            right: Some(Box::new(ContextExpr::from_expression(right))),
            syllable: None,
        }
    }

    /// Create a lookahead context with ContextExpr.
    pub fn lookahead_expr(right: ContextExpr) -> Self {
        Self {
            left: None,
            right: Some(Box::new(right)),
            syllable: None,
        }
    }

    /// Create a lookbehind context (only left side).
    pub fn lookbehind(left: Expression) -> Self {
        Self {
            left: Some(Box::new(ContextExpr::from_expression(left))),
            right: None,
            syllable: None,
        }
    }

    /// Create a lookbehind context with ContextExpr.
    pub fn lookbehind_expr(left: ContextExpr) -> Self {
        Self {
            left: Some(Box::new(left)),
            right: None,
            syllable: None,
        }
    }

    /// Create a word-start context (`#_`).
    pub fn word_start() -> Self {
        Self {
            left: Some(Box::new(ContextExpr::WordBoundary)),
            right: None,
            syllable: None,
        }
    }

    /// Create a word-end context (`_#`).
    pub fn word_end() -> Self {
        Self {
            left: None,
            right: Some(Box::new(ContextExpr::WordBoundary)),
            syllable: None,
        }
    }

    /// Add a syllable condition to this context.
    pub fn with_syllable(mut self, syllable: SyllableExpr) -> Self {
        self.syllable = Some(syllable);
        self
    }

    /// Check if this context has a syllable condition.
    pub fn has_syllable_condition(&self) -> bool {
        self.syllable.is_some()
    }
}

// ============================================================================
// Expressions (Regex-like patterns)
// ============================================================================

/// An expression node (regex-like pattern).
///
/// This is the AST representation of patterns and replacements in rules.
/// It supports standard regex constructs.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Empty expression (matches empty string)
    Empty,

    /// Single character literal
    Char(char),

    /// Character class (e.g., `[aeiou]`)
    CharClass {
        /// Characters in the class
        chars: Vec<char>,
        /// Whether the class is negated (`[^...]`)
        negated: bool,
    },

    /// Character range within a class (e.g., `[a-z]`)
    CharRange {
        /// Start character (inclusive)
        start: char,
        /// End character (inclusive)
        end: char,
    },

    /// Any character (`.`)
    Any,

    /// Concatenation of two expressions
    Concat(Box<Expression>, Box<Expression>),

    /// Alternation of two expressions (`a|b`)
    Alt(Box<Expression>, Box<Expression>),

    /// Kleene star (`a*` - zero or more)
    Star(Box<Expression>),

    /// Kleene plus (`a+` - one or more)
    Plus(Box<Expression>),

    /// Optional (`a?` - zero or one)
    Optional(Box<Expression>),

    /// Exact repetition (`a{n}`)
    RepeatExact(Box<Expression>, usize),

    /// Range repetition (`a{min,max}` or `a{min,}`)
    RepeatRange {
        inner: Box<Expression>,
        min: usize,
        max: Option<usize>,
    },

    /// Grouped expression (parentheses)
    Group(Box<Expression>),

    /// Word boundary (`#`)
    WordBoundary,

    /// Symbol reference (e.g., `FRONT_VOWEL`)
    SymbolRef(String),
}

impl Expression {
    /// Create an empty expression.
    pub fn empty() -> Self {
        Expression::Empty
    }

    /// Create a single character expression.
    pub fn char(c: char) -> Self {
        Expression::Char(c)
    }

    /// Create a literal string expression (concatenation of characters).
    pub fn literal(s: &str) -> Self {
        if s.is_empty() {
            return Expression::Empty;
        }

        let mut chars = s.chars();
        let first = chars.next().expect("non-empty string");
        let mut result = Expression::Char(first);

        for c in chars {
            result = Expression::Concat(Box::new(result), Box::new(Expression::Char(c)));
        }

        result
    }

    /// Create a character class expression.
    pub fn char_class(chars: Vec<char>, negated: bool) -> Self {
        Expression::CharClass { chars, negated }
    }

    /// Create a character class from a slice.
    pub fn char_class_from_slice(chars: &[char]) -> Self {
        Expression::CharClass {
            chars: chars.to_vec(),
            negated: false,
        }
    }

    /// Create a negated character class.
    pub fn negated_char_class(chars: Vec<char>) -> Self {
        Expression::CharClass {
            chars,
            negated: true,
        }
    }

    /// Create a character range.
    pub fn char_range(start: char, end: char) -> Self {
        Expression::CharRange { start, end }
    }

    /// Create an "any character" expression (`.`).
    pub fn any() -> Self {
        Expression::Any
    }

    /// Create a concatenation.
    pub fn concat(a: Expression, b: Expression) -> Self {
        Expression::Concat(Box::new(a), Box::new(b))
    }

    /// Create an alternation.
    pub fn alt(a: Expression, b: Expression) -> Self {
        Expression::Alt(Box::new(a), Box::new(b))
    }

    /// Create a Kleene star (zero or more).
    pub fn star(inner: Expression) -> Self {
        Expression::Star(Box::new(inner))
    }

    /// Create a Kleene plus (one or more).
    pub fn plus(inner: Expression) -> Self {
        Expression::Plus(Box::new(inner))
    }

    /// Create an optional (zero or one).
    pub fn optional(inner: Expression) -> Self {
        Expression::Optional(Box::new(inner))
    }

    /// Create an exact repetition.
    pub fn repeat_exact(inner: Expression, n: usize) -> Self {
        Expression::RepeatExact(Box::new(inner), n)
    }

    /// Create a range repetition.
    pub fn repeat_range(inner: Expression, min: usize, max: Option<usize>) -> Self {
        Expression::RepeatRange {
            inner: Box::new(inner),
            min,
            max,
        }
    }

    /// Create a grouped expression.
    pub fn group(inner: Expression) -> Self {
        Expression::Group(Box::new(inner))
    }

    /// Create a word boundary.
    pub fn word_boundary() -> Self {
        Expression::WordBoundary
    }

    /// Create a symbol reference.
    pub fn symbol_ref(name: impl Into<String>) -> Self {
        Expression::SymbolRef(name.into())
    }

    /// Check if this expression is empty.
    pub fn is_empty(&self) -> bool {
        matches!(self, Expression::Empty)
    }

    /// Check if this expression contains any symbol references.
    pub fn has_symbol_refs(&self) -> bool {
        match self {
            Expression::SymbolRef(_) => true,
            Expression::Concat(a, b) | Expression::Alt(a, b) => {
                a.has_symbol_refs() || b.has_symbol_refs()
            }
            Expression::Star(inner)
            | Expression::Plus(inner)
            | Expression::Optional(inner)
            | Expression::Group(inner)
            | Expression::RepeatExact(inner, _) => inner.has_symbol_refs(),
            Expression::RepeatRange { inner, .. } => inner.has_symbol_refs(),
            _ => false,
        }
    }

    /// Get the estimated size/complexity of this expression.
    pub fn size(&self) -> usize {
        match self {
            Expression::Empty
            | Expression::Char(_)
            | Expression::Any
            | Expression::WordBoundary
            | Expression::SymbolRef(_) => 1,
            Expression::CharClass { chars, .. } => 1 + chars.len(),
            Expression::CharRange { .. } => 1,
            Expression::Concat(a, b) | Expression::Alt(a, b) => 1 + a.size() + b.size(),
            Expression::Star(inner)
            | Expression::Plus(inner)
            | Expression::Optional(inner)
            | Expression::Group(inner)
            | Expression::RepeatExact(inner, _) => 1 + inner.size(),
            Expression::RepeatRange { inner, .. } => 1 + inner.size(),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Empty => write!(f, ""),
            Expression::Char(c) => {
                // Escape special regex characters
                if "\\[](){}|*+?.^$#".contains(*c) {
                    write!(f, "\\{}", c)
                } else {
                    write!(f, "{}", c)
                }
            }
            Expression::CharClass { chars, negated } => {
                write!(f, "[")?;
                if *negated {
                    write!(f, "^")?;
                }
                for c in chars {
                    if "\\[]^-".contains(*c) {
                        write!(f, "\\{}", c)?;
                    } else {
                        write!(f, "{}", c)?;
                    }
                }
                write!(f, "]")
            }
            Expression::CharRange { start, end } => {
                write!(f, "[{}-{}]", start, end)
            }
            Expression::Any => write!(f, "."),
            Expression::Concat(a, b) => write!(f, "{}{}", a, b),
            Expression::Alt(a, b) => write!(f, "({}|{})", a, b),
            Expression::Star(inner) => {
                if needs_parens_for_quantifier(inner) {
                    write!(f, "({})*", inner)
                } else {
                    write!(f, "{}*", inner)
                }
            }
            Expression::Plus(inner) => {
                if needs_parens_for_quantifier(inner) {
                    write!(f, "({})+", inner)
                } else {
                    write!(f, "{}+", inner)
                }
            }
            Expression::Optional(inner) => {
                if needs_parens_for_quantifier(inner) {
                    write!(f, "({})?", inner)
                } else {
                    write!(f, "{}?", inner)
                }
            }
            Expression::RepeatExact(inner, n) => {
                if needs_parens_for_quantifier(inner) {
                    write!(f, "({}){{{}}}", inner, n)
                } else {
                    write!(f, "{}{{{}}}", inner, n)
                }
            }
            Expression::RepeatRange { inner, min, max } => {
                let quantifier = match max {
                    Some(max) => format!("{{{},{}}}", min, max),
                    None => format!("{{{},}}", min),
                };
                if needs_parens_for_quantifier(inner) {
                    write!(f, "({}){}", inner, quantifier)
                } else {
                    write!(f, "{}{}", inner, quantifier)
                }
            }
            Expression::Group(inner) => write!(f, "({})", inner),
            Expression::WordBoundary => write!(f, "#"),
            Expression::SymbolRef(name) => write!(f, "{}", name),
        }
    }
}

/// Check if an expression needs parentheses when applying a quantifier.
fn needs_parens_for_quantifier(expr: &Expression) -> bool {
    !matches!(
        expr,
        Expression::Char(_)
            | Expression::CharClass { .. }
            | Expression::CharRange { .. }
            | Expression::Any
            | Expression::Group(_)
            | Expression::SymbolRef(_)
    )
}

impl fmt::Display for RewriteRuleAST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.pattern, self.replacement)?;

        if let Some(ctx) = &self.context {
            write!(f, " / ")?;
            if let Some(left) = &ctx.left {
                write!(f, "{}", left)?;
            }
            write!(f, "_")?;
            if let Some(right) = &ctx.right {
                write!(f, "{}", right)?;
            }
            if let Some(syllable) = &ctx.syllable {
                write!(f, " if {}", syllable)?;
            }
        }

        if let Some(weight) = self.weight {
            write!(f, " [{:.2}]", weight)?;
        }

        Ok(())
    }
}

impl fmt::Display for LLevFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Metadata
        if let Some(name) = &self.metadata.name {
            writeln!(f, "@name \"{}\"", name)?;
        }
        if let Some(version) = &self.metadata.version {
            writeln!(f, "@version \"{}\"", version)?;
        }
        if let Some(author) = &self.metadata.author {
            writeln!(f, "@author \"{}\"", author)?;
        }
        if let Some(desc) = &self.metadata.description {
            writeln!(f, "@description \"{}\"", desc)?;
        }

        if !self.metadata.is_empty() {
            writeln!(f)?;
        }

        // Symbols
        for sym in &self.symbols {
            writeln!(f, "@define {} = {}", sym.name, sym.value)?;
        }

        if !self.symbols.is_empty() {
            writeln!(f)?;
        }

        // Includes
        for inc in &self.includes {
            writeln!(f, "@include \"{}\"", inc.path)?;
        }

        if !self.includes.is_empty() {
            writeln!(f)?;
        }

        // Rules
        for rule_def in &self.rules {
            // Metadata block
            let meta = &rule_def.metadata;
            let mut meta_parts = Vec::new();

            if let Some(id) = meta.id {
                meta_parts.push(format!("id: {}", id));
            }
            if let Some(name) = &meta.name {
                meta_parts.push(format!("name: \"{}\"", name));
            }
            if let Some(weight) = meta.weight {
                meta_parts.push(format!("weight: {:.2}", weight));
            }
            if let Some(group) = &meta.group {
                meta_parts.push(format!("group: {}", group));
            }
            if !meta.enabled {
                meta_parts.push("enabled: false".to_string());
            }

            if !meta_parts.is_empty() {
                writeln!(f, "[{}]", meta_parts.join(", "))?;
            }

            // Rule
            writeln!(f, "{};", rule_def.rule)?;
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llev_file_new() {
        let file = LLevFile::new();
        assert!(file.metadata.is_empty());
        assert!(!file.has_rules());
        assert!(!file.has_includes());
        assert!(!file.has_symbols());
    }

    #[test]
    fn test_file_metadata() {
        let mut meta = FileMetadata::new();
        assert!(meta.is_empty());

        meta.name = Some("English Rules".to_string());
        assert!(!meta.is_empty());
    }

    #[test]
    fn test_rule_metadata_default() {
        let meta = RuleMetadata::default();
        assert!(meta.id.is_none());
        assert!(meta.name.is_none());
        assert!(meta.weight.is_none());
        assert!(meta.group.is_none());
        assert!(meta.enabled);
    }

    #[test]
    fn test_rule_metadata_with_id() {
        let meta = RuleMetadata::with_id(1);
        assert_eq!(meta.id, Some(1));
        assert!(meta.enabled);
    }

    #[test]
    fn test_rule_metadata_with_id_name() {
        let meta = RuleMetadata::with_id_name(1, "ph to f");
        assert_eq!(meta.id, Some(1));
        assert_eq!(meta.name, Some("ph to f".to_string()));
    }

    #[test]
    fn test_expression_literal() {
        let expr = Expression::literal("phone");
        assert!(!expr.is_empty());
        assert_eq!(expr.to_string(), "phone");
    }

    #[test]
    fn test_expression_empty() {
        let expr = Expression::empty();
        assert!(expr.is_empty());
    }

    #[test]
    fn test_expression_char_class() {
        let expr = Expression::char_class(vec!['a', 'e', 'i', 'o', 'u'], false);
        assert_eq!(expr.to_string(), "[aeiou]");
    }

    #[test]
    fn test_expression_negated_char_class() {
        let expr = Expression::negated_char_class(vec!['a', 'e', 'i', 'o', 'u']);
        assert_eq!(expr.to_string(), "[^aeiou]");
    }

    #[test]
    fn test_expression_char_range() {
        let expr = Expression::char_range('a', 'z');
        assert_eq!(expr.to_string(), "[a-z]");
    }

    #[test]
    fn test_expression_star() {
        let expr = Expression::star(Expression::char('a'));
        assert_eq!(expr.to_string(), "a*");
    }

    #[test]
    fn test_expression_plus() {
        let expr = Expression::plus(Expression::char('a'));
        assert_eq!(expr.to_string(), "a+");
    }

    #[test]
    fn test_expression_optional() {
        let expr = Expression::optional(Expression::char('a'));
        assert_eq!(expr.to_string(), "a?");
    }

    #[test]
    fn test_expression_alternation() {
        let expr = Expression::alt(Expression::literal("ph"), Expression::char('f'));
        assert_eq!(expr.to_string(), "(ph|f)");
    }

    #[test]
    fn test_expression_symbol_ref() {
        let expr = Expression::symbol_ref("FRONT_VOWEL");
        assert_eq!(expr.to_string(), "FRONT_VOWEL");
        assert!(expr.has_symbol_refs());
    }

    #[test]
    fn test_expression_complex_has_symbol_refs() {
        let expr = Expression::concat(
            Expression::literal("c"),
            Expression::symbol_ref("VOWEL"),
        );
        assert!(expr.has_symbol_refs());

        let expr2 = Expression::literal("cat");
        assert!(!expr2.has_symbol_refs());
    }

    #[test]
    fn test_expression_word_boundary() {
        let expr = Expression::word_boundary();
        assert_eq!(expr.to_string(), "#");
    }

    #[test]
    fn test_expression_escape_special_chars() {
        let expr = Expression::char('.');
        assert_eq!(expr.to_string(), "\\.");

        let expr2 = Expression::char('*');
        assert_eq!(expr2.to_string(), "\\*");
    }

    #[test]
    fn test_expression_size() {
        let expr = Expression::literal("phone");
        assert_eq!(expr.size(), 9); // 5 chars + 4 concats

        let expr2 = Expression::star(Expression::char('a'));
        assert_eq!(expr2.size(), 2); // star + char
    }

    #[test]
    fn test_rewrite_rule_ast_simple() {
        let rule = RewriteRuleAST::simple(
            Expression::literal("ph"),
            Expression::char('f'),
        );
        assert!(rule.context.is_none());
        assert!(rule.weight.is_none());
        assert_eq!(rule.to_string(), "ph -> f");
    }

    #[test]
    fn test_rewrite_rule_ast_with_context() {
        let ctx = ContextAST::lookahead(Expression::char_class(vec!['e', 'i'], false));
        let rule = RewriteRuleAST::new(
            Expression::char('c'),
            Expression::char('s'),
            Some(ctx),
            None,
        );
        assert_eq!(rule.to_string(), "c -> s / _[ei]");
    }

    #[test]
    fn test_rewrite_rule_ast_with_weight() {
        let rule = RewriteRuleAST::new(
            Expression::literal("th"),
            Expression::char('t'),
            None,
            Some(0.15),
        );
        assert_eq!(rule.to_string(), "th -> t [0.15]");
    }

    #[test]
    fn test_rewrite_rule_ast_deletion() {
        let ctx = ContextAST::word_end();
        let rule = RewriteRuleAST::deletion(Expression::char('e'), Some(ctx));
        assert!(rule.replacement.is_empty());
        assert_eq!(rule.to_string(), "e ->  / _#");
    }

    #[test]
    fn test_context_ast_word_start() {
        let ctx = ContextAST::word_start();
        assert!(ctx.left.is_some());
        assert!(ctx.right.is_none());
    }

    #[test]
    fn test_context_ast_word_end() {
        let ctx = ContextAST::word_end();
        assert!(ctx.left.is_none());
        assert!(ctx.right.is_some());
    }

    #[test]
    fn test_llev_file_display() {
        let mut file = LLevFile::new();
        file.metadata.name = Some("English Rules".to_string());
        file.metadata.version = Some("1.0".to_string());

        file.symbols.push(SymbolDef::new(
            "VOWEL".to_string(),
            Expression::char_class(vec!['a', 'e', 'i', 'o', 'u'], false),
            Position::start(),
        ));

        let rule = RuleDefinition::new(
            RuleMetadata::with_id_name(1, "ph to f"),
            RewriteRuleAST::simple(Expression::literal("ph"), Expression::char('f')),
            Position::start(),
        );
        file.rules.push(rule);

        let output = file.to_string();
        assert!(output.contains("@name \"English Rules\""));
        assert!(output.contains("@version \"1.0\""));
        assert!(output.contains("@define VOWEL = [aeiou]"));
        assert!(output.contains("[id: 1, name: \"ph to f\"]"));
        assert!(output.contains("ph -> f;"));
    }

    #[test]
    fn test_include_directive() {
        let inc = IncludeDirective::new("extra_rules.llev".to_string(), Position::start());
        assert_eq!(inc.path, "extra_rules.llev");
    }

    #[test]
    fn test_symbol_def() {
        let sym = SymbolDef::new(
            "FRONT_VOWEL".to_string(),
            Expression::char_class(vec!['e', 'i'], false),
            Position::start(),
        );
        assert_eq!(sym.name, "FRONT_VOWEL");
    }

    // ========================================================================
    // Tests for compound context expressions (Phase 1B)
    // ========================================================================

    #[test]
    fn test_context_expr_pattern() {
        let expr = ContextExpr::pattern(Expression::char_class(vec!['a', 'e', 'i', 'o', 'u'], false));
        assert!(expr.is_simple());
        assert!(expr.as_expression().is_some());
        assert_eq!(expr.to_string(), "[aeiou]");
    }

    #[test]
    fn test_context_expr_word_boundary() {
        let expr = ContextExpr::word_boundary();
        assert!(expr.is_simple());
        assert!(expr.as_expression().is_none());
        assert_eq!(expr.to_string(), "#");
    }

    #[test]
    fn test_context_expr_and() {
        let left = ContextExpr::pattern(Expression::char_class(vec!['a', 'e', 'i', 'o', 'u'], false));
        let right = ContextExpr::pattern(Expression::char_class(vec!['b', 'c', 'd', 'f'], false));
        let expr = ContextExpr::and(left, right);
        assert!(!expr.is_simple());
        assert_eq!(expr.to_string(), "([aeiou] & [bcdf])");
    }

    #[test]
    fn test_context_expr_or() {
        let left = ContextExpr::pattern(Expression::char('a'));
        let right = ContextExpr::word_boundary();
        let expr = ContextExpr::or(left, right);
        assert!(!expr.is_simple());
        assert_eq!(expr.to_string(), "(a | #)");
    }

    #[test]
    fn test_context_expr_not() {
        let inner = ContextExpr::pattern(Expression::char_class(vec!['a', 'e', 'i', 'o', 'u'], false));
        let expr = ContextExpr::not(inner);
        assert!(!expr.is_simple());
        assert_eq!(expr.to_string(), "![aeiou]");
    }

    #[test]
    fn test_context_expr_from_expression() {
        let expr1 = ContextExpr::from_expression(Expression::WordBoundary);
        assert!(matches!(expr1, ContextExpr::WordBoundary));

        let expr2 = ContextExpr::from_expression(Expression::char('a'));
        assert!(matches!(expr2, ContextExpr::Pattern(_)));
    }

    #[test]
    fn test_syllable_condition_from_str() {
        assert_eq!(SyllableCondition::from_str("monosyllable"), Some(SyllableCondition::Monosyllable));
        assert_eq!(SyllableCondition::from_str("polysyllable"), Some(SyllableCondition::Polysyllable));
        assert_eq!(SyllableCondition::from_str("open_syllable"), Some(SyllableCondition::OpenSyllable));
        assert_eq!(SyllableCondition::from_str("closed_syllable"), Some(SyllableCondition::ClosedSyllable));
        assert_eq!(SyllableCondition::from_str("final_syllable"), Some(SyllableCondition::FinalSyllable));
        assert_eq!(SyllableCondition::from_str("initial_syllable"), Some(SyllableCondition::InitialSyllable));
        assert_eq!(SyllableCondition::from_str("invalid"), None);
    }

    #[test]
    fn test_syllable_condition_as_str() {
        assert_eq!(SyllableCondition::Monosyllable.as_str(), "monosyllable");
        assert_eq!(SyllableCondition::Polysyllable.as_str(), "polysyllable");
        assert_eq!(SyllableCondition::OpenSyllable.as_str(), "open_syllable");
    }

    #[test]
    fn test_syllable_expr_simple() {
        let expr = SyllableExpr::cond(SyllableCondition::Monosyllable);
        assert!(expr.is_simple());
        assert_eq!(expr.as_condition(), Some(SyllableCondition::Monosyllable));
        assert_eq!(expr.to_string(), "monosyllable");
    }

    #[test]
    fn test_syllable_expr_and() {
        let left = SyllableExpr::cond(SyllableCondition::Polysyllable);
        let right = SyllableExpr::cond(SyllableCondition::FinalSyllable);
        let expr = SyllableExpr::and(left, right);
        assert!(!expr.is_simple());
        assert_eq!(expr.to_string(), "(polysyllable & final_syllable)");
    }

    #[test]
    fn test_syllable_expr_or() {
        let left = SyllableExpr::cond(SyllableCondition::Monosyllable);
        let right = SyllableExpr::cond(SyllableCondition::FinalSyllable);
        let expr = SyllableExpr::or(left, right);
        assert_eq!(expr.to_string(), "(monosyllable | final_syllable)");
    }

    #[test]
    fn test_syllable_expr_not() {
        let inner = SyllableExpr::cond(SyllableCondition::Monosyllable);
        let expr = SyllableExpr::not(inner);
        assert_eq!(expr.to_string(), "!monosyllable");
    }

    #[test]
    fn test_context_ast_with_syllable() {
        let ctx = ContextAST::word_end()
            .with_syllable(SyllableExpr::cond(SyllableCondition::Monosyllable));
        assert!(ctx.has_syllable_condition());
        assert!(ctx.syllable.is_some());
    }

    #[test]
    fn test_rewrite_rule_ast_with_syllable_condition() {
        let ctx = ContextAST::word_end()
            .with_syllable(SyllableExpr::cond(SyllableCondition::Monosyllable));
        let rule = RewriteRuleAST::new(
            Expression::char('y'),
            Expression::char('i'),
            Some(ctx),
            None,
        );
        assert_eq!(rule.to_string(), "y -> i / _# if monosyllable");
    }

    #[test]
    fn test_rewrite_rule_ast_with_compound_context() {
        let left = ContextExpr::pattern(Expression::char_class(vec!['a', 'e', 'i', 'o', 'u'], false));
        let right = ContextExpr::and(
            ContextExpr::pattern(Expression::char_class(vec!['a', 'e', 'i', 'o', 'u'], false)),
            ContextExpr::not(ContextExpr::pattern(Expression::char('y'))),
        );
        let ctx = ContextAST::new_expr(Some(left), Some(right));
        let rule = RewriteRuleAST::new(
            Expression::char('x'),
            Expression::literal("gz"),
            Some(ctx),
            None,
        );
        assert_eq!(rule.to_string(), "x -> gz / [aeiou]_([aeiou] & !y)");
    }

    #[test]
    fn test_context_ast_new_with_syllable() {
        let left = ContextExpr::pattern(Expression::char('a'));
        let right = ContextExpr::word_boundary();
        let syllable = SyllableExpr::and(
            SyllableExpr::cond(SyllableCondition::Polysyllable),
            SyllableExpr::cond(SyllableCondition::FinalSyllable),
        );
        let ctx = ContextAST::new_with_syllable(Some(left), Some(right), Some(syllable));
        assert!(ctx.left.is_some());
        assert!(ctx.right.is_some());
        assert!(ctx.syllable.is_some());
    }
}
