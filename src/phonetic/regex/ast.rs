//! Abstract Syntax Tree for phonetic regular expressions.
//!
//! This module defines the AST nodes for phonetic regex patterns.
//! The AST supports standard regex constructs plus phonetic-specific
//! rewrite rules with context predicates.
//!
//! # Grammar (BNF)
//!
//! ```text
//! regex       ::= alternation
//! alternation ::= concatenation ('|' concatenation)*
//! concatenation ::= quantified+
//! quantified  ::= primary quantifier?
//! quantifier  ::= '*' | '+' | '?' | '{' number '}' | '{' number ',' number? '}'
//! primary     ::= '(' regex ')' | char_class | literal | '.'
//! char_class  ::= '[' '^'? char_range+ ']'
//! char_range  ::= char | char '-' char
//! literal     ::= char+
//! ```
//!
//! # Phonetic Extensions
//!
//! ```text
//! rewrite_rule ::= pattern '->' replacement context? weight?
//! context      ::= '/' left_context? '_' right_context? syllable_clause?
//! left_context ::= context_expr
//! right_context ::= context_expr
//! context_expr ::= context_or
//! context_or   ::= context_and ('|' context_and)*
//! context_and  ::= context_not ('&' context_not)*
//! context_not  ::= '!' context_not | context_primary
//! context_primary ::= regex | '#' | '(' context_expr ')'
//! syllable_clause ::= 'if' syllable_expr
//! weight       ::= '[' number ']'
//! ```

use std::fmt;

use super::super::nfa::types::CharClassChar;

// Re-export syllable types from common module for backward compatibility
pub use crate::phonetic::common::syllable::{SyllableCondition, SyllableExpr};

// ============================================================================
// Regex Flags and Unicode Normalization
// ============================================================================

/// Unicode normalization forms.
///
/// Used with the `(?u:NFC)` flag syntax to normalize input before matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnicodeNormalization {
    /// Canonical Decomposition, followed by Canonical Composition
    NFC,
    /// Canonical Decomposition
    NFD,
    /// Compatibility Decomposition, followed by Canonical Composition
    NFKC,
    /// Compatibility Decomposition
    NFKD,
}

impl fmt::Display for UnicodeNormalization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnicodeNormalization::NFC => write!(f, "NFC"),
            UnicodeNormalization::NFD => write!(f, "NFD"),
            UnicodeNormalization::NFKC => write!(f, "NFKC"),
            UnicodeNormalization::NFKD => write!(f, "NFKD"),
        }
    }
}

impl std::str::FromStr for UnicodeNormalization {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "NFC" => Ok(UnicodeNormalization::NFC),
            "NFD" => Ok(UnicodeNormalization::NFD),
            "NFKC" => Ok(UnicodeNormalization::NFKC),
            "NFKD" => Ok(UnicodeNormalization::NFKD),
            _ => Err(format!("unknown Unicode normalization form: {}", s)),
        }
    }
}

/// Flags/modifiers for regex matching behavior.
///
/// These can be set inline with `(?i)` or scoped with `(?i:pattern)`.
/// `None` means inherit from parent scope, `Some(true)` enables, `Some(false)` disables.
///
/// # Examples
///
/// - `(?i)abc` - case-insensitive matching for the rest of the pattern
/// - `(?i:abc)def` - case-insensitive only for "abc"
/// - `(?-i:abc)` - explicitly case-sensitive
/// - `(?u:NFC:cafe)` - normalize to NFC before matching
/// - `(?ia:cafe)` - case-insensitive AND accent-insensitive
/// - `(?m)^line$` - multiline: `^` and `$` match line boundaries
/// - `(?s).*` - dotall: `.` matches newlines
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RegexFlags {
    /// Case-insensitive matching (`(?i)` or `(?-i)`)
    pub case_insensitive: Option<bool>,

    /// Unicode normalization mode (`(?u:NFC)`, `(?u:NFD)`, etc.)
    pub unicode_normalization: Option<UnicodeNormalization>,

    /// Feature-based matching for phonetic patterns (`(?f)`)
    ///
    /// When enabled, character classes like `[voiced]` match any character
    /// with the "voiced" phonetic feature.
    pub feature_based: Option<bool>,

    /// Accent-insensitive matching (`(?a)`)
    ///
    /// Ignores diacritics when matching (e.g., "é" matches "e").
    pub accent_insensitive: Option<bool>,

    /// Multiline mode (`(?m)` or `(?-m)`)
    ///
    /// When enabled, `^` matches at the start of any line and `$` matches
    /// at the end of any line. When disabled, they only match at input
    /// start/end.
    pub multiline: Option<bool>,

    /// Dotall (single-line) mode (`(?s)` or `(?-s)`)
    ///
    /// When enabled, `.` matches any character including newlines.
    /// When disabled, `.` matches any character except newlines.
    pub dotall: Option<bool>,

    /// Local Levenshtein distance limit for this pattern segment (`(?;N)` or `(?flags;N:pattern)`)
    ///
    /// When set, this specifies the maximum edit distance allowed for matching
    /// this pattern segment. This allows different parts of a pattern to have
    /// different error tolerance levels.
    ///
    /// # Examples
    ///
    /// - `(?;0:exact)` - "exact" must match exactly (0 edits)
    /// - `(?;2:fuzzy)` - "fuzzy" allows up to 2 edits
    /// - `(?i;1:word)` - case-insensitive "word" with up to 1 edit
    pub local_distance: Option<u8>,
}

impl RegexFlags {
    /// Create a new empty flags set (all flags inherit from parent).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create flags with case-insensitive matching enabled.
    pub fn case_insensitive() -> Self {
        Self {
            case_insensitive: Some(true),
            ..Default::default()
        }
    }

    /// Create flags with Unicode normalization.
    pub fn with_normalization(form: UnicodeNormalization) -> Self {
        Self {
            unicode_normalization: Some(form),
            ..Default::default()
        }
    }

    /// Create flags with feature-based matching enabled.
    pub fn feature_based() -> Self {
        Self {
            feature_based: Some(true),
            ..Default::default()
        }
    }

    /// Create flags with accent-insensitive matching enabled.
    pub fn accent_insensitive() -> Self {
        Self {
            accent_insensitive: Some(true),
            ..Default::default()
        }
    }

    /// Create flags with multiline mode enabled.
    ///
    /// In multiline mode, `^` and `$` match at line boundaries,
    /// not just at input start/end.
    pub fn multiline() -> Self {
        Self {
            multiline: Some(true),
            ..Default::default()
        }
    }

    /// Create flags with dotall (single-line) mode enabled.
    ///
    /// In dotall mode, `.` matches any character including newlines.
    pub fn dotall() -> Self {
        Self {
            dotall: Some(true),
            ..Default::default()
        }
    }

    /// Merge flags, with `other` taking precedence for explicitly set values.
    ///
    /// This is used for flag scoping: inner flags override outer flags.
    pub fn merge(&self, other: &RegexFlags) -> RegexFlags {
        RegexFlags {
            case_insensitive: other.case_insensitive.or(self.case_insensitive),
            unicode_normalization: other.unicode_normalization.or(self.unicode_normalization),
            feature_based: other.feature_based.or(self.feature_based),
            accent_insensitive: other.accent_insensitive.or(self.accent_insensitive),
            multiline: other.multiline.or(self.multiline),
            dotall: other.dotall.or(self.dotall),
            local_distance: other.local_distance.or(self.local_distance),
        }
    }

    /// Check if any flags are explicitly set.
    pub fn is_empty(&self) -> bool {
        self.case_insensitive.is_none()
            && self.unicode_normalization.is_none()
            && self.feature_based.is_none()
            && self.accent_insensitive.is_none()
            && self.multiline.is_none()
            && self.dotall.is_none()
            && self.local_distance.is_none()
    }

    /// Create flags with a specific local Levenshtein distance limit.
    pub fn with_local_distance(distance: u8) -> Self {
        Self {
            local_distance: Some(distance),
            ..Default::default()
        }
    }
}

impl fmt::Display for RegexFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();

        if self.case_insensitive == Some(true) {
            parts.push("i".to_string());
        } else if self.case_insensitive == Some(false) {
            parts.push("-i".to_string());
        }

        if let Some(norm) = self.unicode_normalization {
            parts.push(format!("u:{}", norm));
        }

        if self.feature_based == Some(true) {
            parts.push("f".to_string());
        } else if self.feature_based == Some(false) {
            parts.push("-f".to_string());
        }

        if self.accent_insensitive == Some(true) {
            parts.push("a".to_string());
        } else if self.accent_insensitive == Some(false) {
            parts.push("-a".to_string());
        }

        if self.multiline == Some(true) {
            parts.push("m".to_string());
        } else if self.multiline == Some(false) {
            parts.push("-m".to_string());
        }

        if self.dotall == Some(true) {
            parts.push("s".to_string());
        } else if self.dotall == Some(false) {
            parts.push("-s".to_string());
        }

        // Format: flags;N or just ;N if no flags
        let flags_str = parts.join("");
        if let Some(dist) = self.local_distance {
            if flags_str.is_empty() {
                write!(f, ";{}", dist)?;
            } else {
                write!(f, "{};{}", flags_str, dist)?;
            }
        } else {
            write!(f, "{}", flags_str)?;
        }
        Ok(())
    }
}

/// A phonetic regular expression AST node.
///
/// Supports both standard regex constructs and phonetic rewrite rules.
#[derive(Debug, Clone, PartialEq)]
pub enum Regex {
    /// Empty pattern (matches empty string)
    Empty,

    /// Single character literal
    Char(char),

    /// Character class (e.g., `[aeiou]`, `[^aeiou]`, `[a-z]`)
    CharClass(CharClassChar),

    /// Any character (`.`)
    Any,

    /// Concatenation of two patterns
    Concat(Box<Regex>, Box<Regex>),

    /// Alternation of two patterns (`a|b`)
    Alt(Box<Regex>, Box<Regex>),

    /// Kleene star (`a*` - zero or more)
    Star(Box<Regex>),

    /// Kleene plus (`a+` - one or more)
    Plus(Box<Regex>),

    /// Optional (`a?` - zero or one)
    Optional(Box<Regex>),

    /// Exact repetition (`a{3}`)
    RepeatExact(Box<Regex>, usize),

    /// Range repetition (`a{2,4}` or `a{2,}`)
    RepeatRange(Box<Regex>, usize, Option<usize>),

    /// Numbered capturing group: `(pattern)`
    ///
    /// The `usize` is the group number (1-indexed for captures).
    /// Group 0 is reserved for the full match in capture extraction.
    CapturingGroup(usize, Box<Regex>),

    /// Non-capturing group: `(?:pattern)`
    ///
    /// Groups the pattern for precedence without capturing.
    NonCapturingGroup(Box<Regex>),

    /// Named capturing group: `(?<name>pattern)`
    ///
    /// A capturing group that can be referenced by name using `(?&name)`.
    NamedGroup(String, Box<Regex>),

    /// Group reference (subroutine call): `(?&name)`
    ///
    /// References a named group, allowing pattern reuse and recursion.
    /// The referenced group must be defined somewhere in the pattern.
    GroupRef(String),

    /// Scoped flags: `(?flags:pattern)` or standalone `(?flags)`
    ///
    /// Applies flags to the inner pattern (scoped) or to subsequent patterns (inline).
    /// If `inner` is `None`, this is an inline flag that affects the rest of the pattern.
    FlagsGroup {
        /// Flag set being introduced by this group.
        flags: RegexFlags,
        /// Inner pattern when scoped (`Some`); `None` for an inline flag prefix.
        inner: Option<Box<Regex>>,
    },

    /// Word boundary assertion (`#` at start or end)
    WordBoundary,

    /// Start of line anchor (`^`)
    ///
    /// In multiline mode (`(?m)`), matches at the start of any line.
    /// Otherwise, matches only at the start of the input.
    StartOfLine,

    /// End of line anchor (`$`)
    ///
    /// In multiline mode (`(?m)`), matches at the end of any line.
    /// Otherwise, matches only at the end of the input.
    EndOfLine,

    /// Start of input anchor (`\A`)
    ///
    /// Always matches only at the absolute start of the input,
    /// regardless of multiline mode.
    StartOfInput,

    /// End of input anchor (`\Z`)
    ///
    /// Matches at the end of the input, allowing an optional trailing newline.
    /// This is the standard "end of string" anchor.
    EndOfInput,

    /// Strict end of input anchor (`\z`)
    ///
    /// Matches only at the absolute end of the input,
    /// with no trailing newline allowed.
    EndOfInputStrict,

    /// Rewrite rule: pattern -> replacement with optional context and weight
    RewriteRule {
        /// Left-hand side pattern to match.
        pattern: Box<Regex>,
        /// Right-hand side substitution produced when the pattern matches.
        replacement: Box<Regex>,
        /// Optional context predicate constraining where the rule applies.
        context: Option<Box<ContextPredicate>>,
        /// Cost/weight associated with applying this rewrite.
        weight: f64,
    },
}

// ============================================================================
// Context Expressions
// ============================================================================

/// A context expression with logical operators.
///
/// Allows building compound context predicates like:
/// - `[aeiou]` - simple pattern
/// - `[aeiou] & ![y]` - vowel but not y
/// - `([bcdf] | #)` - consonant or word boundary
#[derive(Debug, Clone, PartialEq)]
pub enum ContextExpr {
    /// Simple pattern-based context
    Pattern(Regex),
    /// Word boundary marker
    WordBoundary,
    /// Both contexts must match
    And(Box<ContextExpr>, Box<ContextExpr>),
    /// Either context must match
    Or(Box<ContextExpr>, Box<ContextExpr>),
    /// Context must NOT match
    Not(Box<ContextExpr>),
}

impl ContextExpr {
    /// Create a pattern-based context expression.
    pub fn pattern(regex: Regex) -> Self {
        ContextExpr::Pattern(regex)
    }

    /// Create a word boundary expression.
    pub fn word_boundary() -> Self {
        ContextExpr::WordBoundary
    }

    /// Create an AND expression.
    pub fn and(left: ContextExpr, right: ContextExpr) -> Self {
        ContextExpr::And(Box::new(left), Box::new(right))
    }

    /// Create an OR expression.
    pub fn or(left: ContextExpr, right: ContextExpr) -> Self {
        ContextExpr::Or(Box::new(left), Box::new(right))
    }

    /// Create a NOT expression.
    pub fn not(inner: ContextExpr) -> Self {
        ContextExpr::Not(Box::new(inner))
    }

    /// Get the estimated size/complexity of this context expression.
    pub fn size(&self) -> usize {
        match self {
            ContextExpr::Pattern(regex) => regex.size(),
            ContextExpr::WordBoundary => 1,
            ContextExpr::And(a, b) | ContextExpr::Or(a, b) => 1 + a.size() + b.size(),
            ContextExpr::Not(inner) => 1 + inner.size(),
        }
    }
}

impl fmt::Display for ContextExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextExpr::Pattern(regex) => write!(f, "{}", regex),
            ContextExpr::WordBoundary => write!(f, "#"),
            ContextExpr::And(a, b) => write!(f, "({} & {})", a, b),
            ContextExpr::Or(a, b) => write!(f, "({} | {})", a, b),
            ContextExpr::Not(inner) => write!(f, "!{}", inner),
        }
    }
}

/// Context predicate for rewrite rules.
///
/// Specifies the environment in which a rewrite rule applies.
/// Supports compound contexts with logical operators and syllable conditions.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextPredicate {
    /// Left context (lookbehind) - pattern that must precede the match
    pub left: Option<ContextExpr>,

    /// Right context (lookahead) - pattern that must follow the match
    pub right: Option<ContextExpr>,

    /// Syllable condition - structural constraint based on syllable position
    pub syllable: Option<SyllableExpr>,
}

impl ContextPredicate {
    /// Create a new context predicate with left and right contexts.
    pub fn new(left: Option<Regex>, right: Option<Regex>) -> Self {
        Self {
            left: left.map(ContextExpr::Pattern),
            right: right.map(ContextExpr::Pattern),
            syllable: None,
        }
    }

    /// Create a new context predicate with context expressions.
    pub fn new_with_exprs(
        left: Option<ContextExpr>,
        right: Option<ContextExpr>,
        syllable: Option<SyllableExpr>,
    ) -> Self {
        Self {
            left,
            right,
            syllable,
        }
    }

    /// Create a context predicate with only right context (lookahead).
    ///
    /// Example: `c -> s / _[ei]` (c becomes s before e or i)
    pub fn lookahead(right: Regex) -> Self {
        Self {
            left: None,
            right: Some(ContextExpr::Pattern(right)),
            syllable: None,
        }
    }

    /// Create a context predicate with only left context (lookbehind).
    ///
    /// Example: `s -> z / [aeiou]_` (s becomes z after a vowel)
    pub fn lookbehind(left: Regex) -> Self {
        Self {
            left: Some(ContextExpr::Pattern(left)),
            right: None,
            syllable: None,
        }
    }

    /// Create a word-start context (`#_`).
    pub fn word_start() -> Self {
        Self {
            left: Some(ContextExpr::WordBoundary),
            right: None,
            syllable: None,
        }
    }

    /// Create a word-end context (`_#`).
    pub fn word_end() -> Self {
        Self {
            left: None,
            right: Some(ContextExpr::WordBoundary),
            syllable: None,
        }
    }

    /// Add a syllable condition to this context predicate.
    pub fn with_syllable(mut self, syllable: SyllableExpr) -> Self {
        self.syllable = Some(syllable);
        self
    }
}

impl Regex {
    /// Create an empty regex (matches empty string).
    pub fn empty() -> Self {
        Regex::Empty
    }

    /// Create a single character regex.
    pub fn char(c: char) -> Self {
        Regex::Char(c)
    }

    /// Create a literal string regex (concatenation of characters).
    pub fn literal(s: &str) -> Self {
        if s.is_empty() {
            return Regex::Empty;
        }

        let mut chars = s.chars();
        let first = chars.next().expect("non-empty string");
        let mut result = Regex::Char(first);

        for c in chars {
            result = Regex::Concat(Box::new(result), Box::new(Regex::Char(c)));
        }

        result
    }

    /// Create a character class regex.
    pub fn char_class(class: CharClassChar) -> Self {
        Regex::CharClass(class)
    }

    /// Create a "any character" regex (`.`).
    pub fn any() -> Self {
        Regex::Any
    }

    /// Create a concatenation of two regexes.
    pub fn concat(a: Regex, b: Regex) -> Self {
        Regex::Concat(Box::new(a), Box::new(b))
    }

    /// Create an alternation of two regexes.
    pub fn alt(a: Regex, b: Regex) -> Self {
        Regex::Alt(Box::new(a), Box::new(b))
    }

    /// Create a Kleene star (zero or more).
    pub fn star(inner: Regex) -> Self {
        Regex::Star(Box::new(inner))
    }

    /// Create a Kleene plus (one or more).
    pub fn plus(inner: Regex) -> Self {
        Regex::Plus(Box::new(inner))
    }

    /// Create an optional (zero or one).
    pub fn optional(inner: Regex) -> Self {
        Regex::Optional(Box::new(inner))
    }

    /// Create an exact repetition (`a{n}`).
    pub fn repeat_exact(inner: Regex, n: usize) -> Self {
        Regex::RepeatExact(Box::new(inner), n)
    }

    /// Create a range repetition (`a{min,max}`).
    pub fn repeat_range(inner: Regex, min: usize, max: Option<usize>) -> Self {
        Regex::RepeatRange(Box::new(inner), min, max)
    }

    /// Create a numbered capturing group.
    ///
    /// # Arguments
    ///
    /// * `group_num` - The group number (1-indexed)
    /// * `inner` - The pattern inside the group
    pub fn capturing_group(group_num: usize, inner: Regex) -> Self {
        Regex::CapturingGroup(group_num, Box::new(inner))
    }

    /// Create a non-capturing group.
    ///
    /// Groups the pattern for precedence without capturing.
    pub fn non_capturing_group(inner: Regex) -> Self {
        Regex::NonCapturingGroup(Box::new(inner))
    }

    /// Create a named capturing group.
    ///
    /// # Arguments
    ///
    /// * `name` - The group name (for later reference with `(?&name)`)
    /// * `inner` - The pattern inside the group
    pub fn named_group(name: impl Into<String>, inner: Regex) -> Self {
        Regex::NamedGroup(name.into(), Box::new(inner))
    }

    /// Create a group reference (subroutine call).
    ///
    /// References a named group defined elsewhere in the pattern.
    pub fn group_ref(name: impl Into<String>) -> Self {
        Regex::GroupRef(name.into())
    }

    /// Create a scoped flags group.
    ///
    /// Applies flags only to the inner pattern.
    pub fn flags_group(flags: RegexFlags, inner: Regex) -> Self {
        Regex::FlagsGroup {
            flags,
            inner: Some(Box::new(inner)),
        }
    }

    /// Create inline flags (no inner pattern).
    ///
    /// Applies flags to subsequent patterns in the same scope.
    pub fn inline_flags(flags: RegexFlags) -> Self {
        Regex::FlagsGroup { flags, inner: None }
    }

    /// Create a word boundary assertion.
    pub fn word_boundary() -> Self {
        Regex::WordBoundary
    }

    /// Create a rewrite rule.
    ///
    /// # Arguments
    ///
    /// * `pattern` - The pattern to match
    /// * `replacement` - The replacement pattern
    /// * `context` - Optional context predicate
    /// * `weight` - Optional weight (cost) for phonetic distance, defaults to 0.0
    pub fn rewrite_rule(
        pattern: Regex,
        replacement: Regex,
        context: Option<ContextPredicate>,
        weight: f64,
    ) -> Self {
        Regex::RewriteRule {
            pattern: Box::new(pattern),
            replacement: Box::new(replacement),
            context: context.map(Box::new),
            weight,
        }
    }

    /// Check if this regex is empty (matches only empty string).
    pub fn is_empty(&self) -> bool {
        matches!(self, Regex::Empty)
    }

    /// Check if this regex is a rewrite rule.
    pub fn is_rewrite_rule(&self) -> bool {
        matches!(self, Regex::RewriteRule { .. })
    }

    /// Get the estimated size/complexity of this regex.
    #[allow(deprecated)]
    pub fn size(&self) -> usize {
        match self {
            Regex::Empty
            | Regex::Char(_)
            | Regex::Any
            | Regex::WordBoundary
            | Regex::StartOfLine
            | Regex::EndOfLine
            | Regex::StartOfInput
            | Regex::EndOfInput
            | Regex::EndOfInputStrict => 1,
            Regex::CharClass(_) => 1,
            Regex::GroupRef(_) => 1,
            Regex::Concat(a, b) | Regex::Alt(a, b) => 1 + a.size() + b.size(),
            Regex::Star(inner)
            | Regex::Plus(inner)
            | Regex::Optional(inner)
            | Regex::NonCapturingGroup(inner)
            | Regex::CapturingGroup(_, inner)
            | Regex::NamedGroup(_, inner) => 1 + inner.size(),
            Regex::RepeatExact(inner, _) | Regex::RepeatRange(inner, _, _) => 1 + inner.size(),
            Regex::FlagsGroup { inner, .. } => 1 + inner.as_ref().map_or(0, |i| i.size()),
            Regex::RewriteRule {
                pattern,
                replacement,
                context,
                ..
            } => {
                let ctx_size = context.as_ref().map_or(0, |c| {
                    c.left.as_ref().map_or(0, |l| l.size())
                        + c.right.as_ref().map_or(0, |r| r.size())
                        + c.syllable.as_ref().map_or(0, |s| s.size())
                });
                1 + pattern.size() + replacement.size() + ctx_size
            }
        }
    }
}

impl fmt::Display for Regex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Regex::Empty => write!(f, ""),
            Regex::Char(c) => {
                // Escape special regex characters
                if "\\[](){}|*+?.^$".contains(*c) {
                    write!(f, "\\{}", c)
                } else {
                    write!(f, "{}", c)
                }
            }
            Regex::CharClass(class) => write!(f, "{}", class),
            Regex::Any => write!(f, "."),
            Regex::Concat(a, b) => write!(f, "{}{}", a, b),
            Regex::Alt(a, b) => write!(f, "({}|{})", a, b),
            Regex::Star(inner) => {
                if matches!(**inner, Regex::Char(_) | Regex::CharClass(_) | Regex::Any) {
                    write!(f, "{}*", inner)
                } else {
                    write!(f, "({})*", inner)
                }
            }
            Regex::Plus(inner) => {
                if matches!(**inner, Regex::Char(_) | Regex::CharClass(_) | Regex::Any) {
                    write!(f, "{}+", inner)
                } else {
                    write!(f, "({})+", inner)
                }
            }
            Regex::Optional(inner) => {
                if matches!(**inner, Regex::Char(_) | Regex::CharClass(_) | Regex::Any) {
                    write!(f, "{}?", inner)
                } else {
                    write!(f, "({})?", inner)
                }
            }
            Regex::RepeatExact(inner, n) => {
                if matches!(**inner, Regex::Char(_) | Regex::CharClass(_) | Regex::Any) {
                    write!(f, "{}{{{}}}", inner, n)
                } else {
                    write!(f, "({}){{{}}}", inner, n)
                }
            }
            Regex::RepeatRange(inner, min, max) => {
                let quantifier = match max {
                    Some(max) => format!("{{{},{}}}", min, max),
                    None => format!("{{{},}}", min),
                };
                if matches!(**inner, Regex::Char(_) | Regex::CharClass(_) | Regex::Any) {
                    write!(f, "{}{}", inner, quantifier)
                } else {
                    write!(f, "({}){}", inner, quantifier)
                }
            }
            Regex::CapturingGroup(_, inner) => write!(f, "({})", inner),
            Regex::NonCapturingGroup(inner) => write!(f, "(?:{})", inner),
            Regex::NamedGroup(name, inner) => write!(f, "(?<{}>{})", name, inner),
            Regex::GroupRef(name) => write!(f, "(?&{})", name),
            Regex::FlagsGroup { flags, inner } => match inner {
                Some(inner) => write!(f, "(?{}:{})", flags, inner),
                None => write!(f, "(?{})", flags),
            },
            Regex::WordBoundary => write!(f, "#"),
            Regex::StartOfLine => write!(f, "^"),
            Regex::EndOfLine => write!(f, "$"),
            Regex::StartOfInput => write!(f, "\\A"),
            Regex::EndOfInput => write!(f, "\\Z"),
            Regex::EndOfInputStrict => write!(f, "\\z"),
            Regex::RewriteRule {
                pattern,
                replacement,
                context,
                weight,
            } => {
                write!(f, "{} -> {}", pattern, replacement)?;
                if let Some(ctx) = context {
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
                if *weight != 0.0 {
                    write!(f, " [{:.2}]", weight)?;
                }
                Ok(())
            }
        }
    }
}

// ============================================================================
// Byte-level AST (for ASCII-only patterns)
// ============================================================================

use super::super::nfa::types::CharClass;

/// A byte-level phonetic regular expression AST node.
///
/// Optimized for ASCII text, ~5% faster than char-level.
#[derive(Debug, Clone, PartialEq)]
pub enum RegexByte {
    /// Empty pattern (matches empty string)
    Empty,

    /// Single byte literal
    Byte(u8),

    /// Byte class (e.g., `[aeiou]`, `[^aeiou]`, `[a-z]`)
    ByteClass(CharClass),

    /// Any byte (`.`)
    Any,

    /// Concatenation of two patterns
    Concat(Box<RegexByte>, Box<RegexByte>),

    /// Alternation of two patterns (`a|b`)
    Alt(Box<RegexByte>, Box<RegexByte>),

    /// Kleene star (`a*` - zero or more)
    Star(Box<RegexByte>),

    /// Kleene plus (`a+` - one or more)
    Plus(Box<RegexByte>),

    /// Optional (`a?` - zero or one)
    Optional(Box<RegexByte>),

    /// Exact repetition (`a{3}`)
    RepeatExact(Box<RegexByte>, usize),

    /// Range repetition (`a{2,4}` or `a{2,}`)
    RepeatRange(Box<RegexByte>, usize, Option<usize>),

    /// Numbered capturing group: `(pattern)`
    CapturingGroup(usize, Box<RegexByte>),

    /// Non-capturing group: `(?:pattern)`
    NonCapturingGroup(Box<RegexByte>),

    /// Named capturing group: `(?<name>pattern)`
    NamedGroup(String, Box<RegexByte>),

    /// Group reference (subroutine call): `(?&name)`
    GroupRef(String),

    /// Scoped flags: `(?flags:pattern)` or standalone `(?flags)`
    FlagsGroup {
        /// Flag set being introduced by this group.
        flags: RegexFlags,
        /// Inner pattern when scoped (`Some`); `None` for an inline flag prefix.
        inner: Option<Box<RegexByte>>,
    },

    /// Word boundary assertion (`#` at start or end)
    WordBoundary,

    /// Start of line anchor (`^`)
    ///
    /// In multiline mode (`(?m)`), matches at the start of any line.
    /// Otherwise, matches only at the start of the input.
    StartOfLine,

    /// End of line anchor (`$`)
    ///
    /// In multiline mode (`(?m)`), matches at the end of any line.
    /// Otherwise, matches only at the end of the input.
    EndOfLine,

    /// Start of input anchor (`\A`)
    ///
    /// Always matches only at the absolute start of the input,
    /// regardless of multiline mode.
    StartOfInput,

    /// End of input anchor (`\Z`)
    ///
    /// Matches at the end of the input, allowing an optional trailing newline.
    /// This is the standard "end of string" anchor.
    EndOfInput,

    /// Strict end of input anchor (`\z`)
    ///
    /// Matches only at the absolute end of the input,
    /// with no trailing newline allowed.
    EndOfInputStrict,

    /// Rewrite rule: pattern -> replacement with optional context and weight
    RewriteRule {
        /// Left-hand side pattern to match.
        pattern: Box<RegexByte>,
        /// Right-hand side substitution produced when the pattern matches.
        replacement: Box<RegexByte>,
        /// Optional context predicate constraining where the rule applies.
        context: Option<Box<ContextPredicateByte>>,
        /// Cost/weight associated with applying this rewrite.
        weight: f64,
    },
}

// ============================================================================
// Byte-level Context Expressions
// ============================================================================

/// A byte-level context expression with logical operators.
#[derive(Debug, Clone, PartialEq)]
pub enum ContextExprByte {
    /// Simple pattern-based context
    Pattern(RegexByte),
    /// Word boundary marker
    WordBoundary,
    /// Both contexts must match
    And(Box<ContextExprByte>, Box<ContextExprByte>),
    /// Either context must match
    Or(Box<ContextExprByte>, Box<ContextExprByte>),
    /// Context must NOT match
    Not(Box<ContextExprByte>),
}

impl ContextExprByte {
    /// Create a pattern-based context expression.
    pub fn pattern(regex: RegexByte) -> Self {
        ContextExprByte::Pattern(regex)
    }

    /// Create a word boundary expression.
    pub fn word_boundary() -> Self {
        ContextExprByte::WordBoundary
    }

    /// Create an AND expression.
    pub fn and(left: ContextExprByte, right: ContextExprByte) -> Self {
        ContextExprByte::And(Box::new(left), Box::new(right))
    }

    /// Create an OR expression.
    pub fn or(left: ContextExprByte, right: ContextExprByte) -> Self {
        ContextExprByte::Or(Box::new(left), Box::new(right))
    }

    /// Create a NOT expression.
    pub fn not(inner: ContextExprByte) -> Self {
        ContextExprByte::Not(Box::new(inner))
    }

    /// Get the estimated size/complexity of this context expression.
    pub fn size(&self) -> usize {
        match self {
            ContextExprByte::Pattern(regex) => regex.size(),
            ContextExprByte::WordBoundary => 1,
            ContextExprByte::And(a, b) | ContextExprByte::Or(a, b) => 1 + a.size() + b.size(),
            ContextExprByte::Not(inner) => 1 + inner.size(),
        }
    }
}

impl fmt::Display for ContextExprByte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextExprByte::Pattern(regex) => write!(f, "{}", regex),
            ContextExprByte::WordBoundary => write!(f, "#"),
            ContextExprByte::And(a, b) => write!(f, "({} & {})", a, b),
            ContextExprByte::Or(a, b) => write!(f, "({} | {})", a, b),
            ContextExprByte::Not(inner) => write!(f, "!{}", inner),
        }
    }
}

/// Byte-level context predicate for rewrite rules.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextPredicateByte {
    /// Left context (lookbehind) - pattern that must precede the match
    pub left: Option<ContextExprByte>,

    /// Right context (lookahead) - pattern that must follow the match
    pub right: Option<ContextExprByte>,

    /// Syllable condition - structural constraint based on syllable position
    pub syllable: Option<SyllableExpr>,
}

impl ContextPredicateByte {
    /// Create a new context predicate with left and right contexts.
    pub fn new(left: Option<RegexByte>, right: Option<RegexByte>) -> Self {
        Self {
            left: left.map(ContextExprByte::Pattern),
            right: right.map(ContextExprByte::Pattern),
            syllable: None,
        }
    }

    /// Create a new context predicate with context expressions.
    pub fn new_with_exprs(
        left: Option<ContextExprByte>,
        right: Option<ContextExprByte>,
        syllable: Option<SyllableExpr>,
    ) -> Self {
        Self {
            left,
            right,
            syllable,
        }
    }

    /// Create a context predicate with only right context (lookahead).
    pub fn lookahead(right: RegexByte) -> Self {
        Self {
            left: None,
            right: Some(ContextExprByte::Pattern(right)),
            syllable: None,
        }
    }

    /// Create a context predicate with only left context (lookbehind).
    pub fn lookbehind(left: RegexByte) -> Self {
        Self {
            left: Some(ContextExprByte::Pattern(left)),
            right: None,
            syllable: None,
        }
    }

    /// Create a word-start context (`#_`).
    pub fn word_start() -> Self {
        Self {
            left: Some(ContextExprByte::WordBoundary),
            right: None,
            syllable: None,
        }
    }

    /// Create a word-end context (`_#`).
    pub fn word_end() -> Self {
        Self {
            left: None,
            right: Some(ContextExprByte::WordBoundary),
            syllable: None,
        }
    }

    /// Add a syllable condition to this context predicate.
    pub fn with_syllable(mut self, syllable: SyllableExpr) -> Self {
        self.syllable = Some(syllable);
        self
    }
}

impl RegexByte {
    /// Create an empty regex (matches empty string).
    pub fn empty() -> Self {
        RegexByte::Empty
    }

    /// Create a single byte regex.
    pub fn byte(b: u8) -> Self {
        RegexByte::Byte(b)
    }

    /// Create a literal string regex (concatenation of bytes).
    pub fn literal(s: &[u8]) -> Self {
        if s.is_empty() {
            return RegexByte::Empty;
        }

        let mut result = RegexByte::Byte(s[0]);

        for &b in &s[1..] {
            result = RegexByte::Concat(Box::new(result), Box::new(RegexByte::Byte(b)));
        }

        result
    }

    /// Create a byte class regex.
    pub fn byte_class(class: CharClass) -> Self {
        RegexByte::ByteClass(class)
    }

    /// Create a "any byte" regex (`.`).
    pub fn any() -> Self {
        RegexByte::Any
    }

    /// Create a concatenation of two regexes.
    pub fn concat(a: RegexByte, b: RegexByte) -> Self {
        RegexByte::Concat(Box::new(a), Box::new(b))
    }

    /// Create an alternation of two regexes.
    pub fn alt(a: RegexByte, b: RegexByte) -> Self {
        RegexByte::Alt(Box::new(a), Box::new(b))
    }

    /// Create a Kleene star (zero or more).
    pub fn star(inner: RegexByte) -> Self {
        RegexByte::Star(Box::new(inner))
    }

    /// Create a Kleene plus (one or more).
    pub fn plus(inner: RegexByte) -> Self {
        RegexByte::Plus(Box::new(inner))
    }

    /// Create an optional (zero or one).
    pub fn optional(inner: RegexByte) -> Self {
        RegexByte::Optional(Box::new(inner))
    }

    /// Create an exact repetition (`a{n}`).
    pub fn repeat_exact(inner: RegexByte, n: usize) -> Self {
        RegexByte::RepeatExact(Box::new(inner), n)
    }

    /// Create a range repetition (`a{min,max}`).
    pub fn repeat_range(inner: RegexByte, min: usize, max: Option<usize>) -> Self {
        RegexByte::RepeatRange(Box::new(inner), min, max)
    }

    /// Create a numbered capturing group.
    pub fn capturing_group(group_num: usize, inner: RegexByte) -> Self {
        RegexByte::CapturingGroup(group_num, Box::new(inner))
    }

    /// Create a non-capturing group.
    pub fn non_capturing_group(inner: RegexByte) -> Self {
        RegexByte::NonCapturingGroup(Box::new(inner))
    }

    /// Create a named capturing group.
    pub fn named_group(name: impl Into<String>, inner: RegexByte) -> Self {
        RegexByte::NamedGroup(name.into(), Box::new(inner))
    }

    /// Create a group reference (subroutine call).
    pub fn group_ref(name: impl Into<String>) -> Self {
        RegexByte::GroupRef(name.into())
    }

    /// Create a scoped flags group.
    pub fn flags_group(flags: RegexFlags, inner: RegexByte) -> Self {
        RegexByte::FlagsGroup {
            flags,
            inner: Some(Box::new(inner)),
        }
    }

    /// Create inline flags (no inner pattern).
    pub fn inline_flags(flags: RegexFlags) -> Self {
        RegexByte::FlagsGroup { flags, inner: None }
    }

    /// Create a word boundary assertion.
    pub fn word_boundary() -> Self {
        RegexByte::WordBoundary
    }

    /// Create a rewrite rule.
    pub fn rewrite_rule(
        pattern: RegexByte,
        replacement: RegexByte,
        context: Option<ContextPredicateByte>,
        weight: f64,
    ) -> Self {
        RegexByte::RewriteRule {
            pattern: Box::new(pattern),
            replacement: Box::new(replacement),
            context: context.map(Box::new),
            weight,
        }
    }

    /// Check if this regex is empty (matches only empty string).
    pub fn is_empty(&self) -> bool {
        matches!(self, RegexByte::Empty)
    }

    /// Check if this regex is a rewrite rule.
    pub fn is_rewrite_rule(&self) -> bool {
        matches!(self, RegexByte::RewriteRule { .. })
    }

    /// Get the estimated size/complexity of this regex.
    #[allow(deprecated)]
    pub fn size(&self) -> usize {
        match self {
            RegexByte::Empty
            | RegexByte::Byte(_)
            | RegexByte::Any
            | RegexByte::WordBoundary
            | RegexByte::StartOfLine
            | RegexByte::EndOfLine
            | RegexByte::StartOfInput
            | RegexByte::EndOfInput
            | RegexByte::EndOfInputStrict => 1,
            RegexByte::ByteClass(_) => 1,
            RegexByte::GroupRef(_) => 1,
            RegexByte::Concat(a, b) | RegexByte::Alt(a, b) => 1 + a.size() + b.size(),
            RegexByte::Star(inner)
            | RegexByte::Plus(inner)
            | RegexByte::Optional(inner)
            | RegexByte::NonCapturingGroup(inner)
            | RegexByte::CapturingGroup(_, inner)
            | RegexByte::NamedGroup(_, inner) => 1 + inner.size(),
            RegexByte::RepeatExact(inner, _) | RegexByte::RepeatRange(inner, _, _) => {
                1 + inner.size()
            }
            RegexByte::FlagsGroup { inner, .. } => 1 + inner.as_ref().map_or(0, |i| i.size()),
            RegexByte::RewriteRule {
                pattern,
                replacement,
                context,
                ..
            } => {
                let ctx_size = context.as_ref().map_or(0, |c| {
                    c.left.as_ref().map_or(0, |l| l.size())
                        + c.right.as_ref().map_or(0, |r| r.size())
                        + c.syllable.as_ref().map_or(0, |s| s.size())
                });
                1 + pattern.size() + replacement.size() + ctx_size
            }
        }
    }
}

impl fmt::Display for RegexByte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegexByte::Empty => write!(f, ""),
            RegexByte::Byte(b) => {
                let c = *b as char;
                // Escape special regex characters
                if "\\[](){}|*+?.^$".contains(c) {
                    write!(f, "\\{}", c)
                } else if b.is_ascii_graphic() || *b == b' ' {
                    write!(f, "{}", c)
                } else {
                    write!(f, "\\x{:02x}", b)
                }
            }
            RegexByte::ByteClass(class) => write!(f, "{}", class),
            RegexByte::Any => write!(f, "."),
            RegexByte::Concat(a, b) => write!(f, "{}{}", a, b),
            RegexByte::Alt(a, b) => write!(f, "({}|{})", a, b),
            RegexByte::Star(inner) => {
                if matches!(
                    **inner,
                    RegexByte::Byte(_) | RegexByte::ByteClass(_) | RegexByte::Any
                ) {
                    write!(f, "{}*", inner)
                } else {
                    write!(f, "({})*", inner)
                }
            }
            RegexByte::Plus(inner) => {
                if matches!(
                    **inner,
                    RegexByte::Byte(_) | RegexByte::ByteClass(_) | RegexByte::Any
                ) {
                    write!(f, "{}+", inner)
                } else {
                    write!(f, "({})+", inner)
                }
            }
            RegexByte::Optional(inner) => {
                if matches!(
                    **inner,
                    RegexByte::Byte(_) | RegexByte::ByteClass(_) | RegexByte::Any
                ) {
                    write!(f, "{}?", inner)
                } else {
                    write!(f, "({})?", inner)
                }
            }
            RegexByte::RepeatExact(inner, n) => {
                if matches!(
                    **inner,
                    RegexByte::Byte(_) | RegexByte::ByteClass(_) | RegexByte::Any
                ) {
                    write!(f, "{}{{{}}}", inner, n)
                } else {
                    write!(f, "({}){{{}}}", inner, n)
                }
            }
            RegexByte::RepeatRange(inner, min, max) => {
                let quantifier = match max {
                    Some(max) => format!("{{{},{}}}", min, max),
                    None => format!("{{{},}}", min),
                };
                if matches!(
                    **inner,
                    RegexByte::Byte(_) | RegexByte::ByteClass(_) | RegexByte::Any
                ) {
                    write!(f, "{}{}", inner, quantifier)
                } else {
                    write!(f, "({}){}", inner, quantifier)
                }
            }
            RegexByte::CapturingGroup(_, inner) => write!(f, "({})", inner),
            RegexByte::NonCapturingGroup(inner) => write!(f, "(?:{})", inner),
            RegexByte::NamedGroup(name, inner) => write!(f, "(?<{}>{})", name, inner),
            RegexByte::GroupRef(name) => write!(f, "(?&{})", name),
            RegexByte::FlagsGroup { flags, inner } => match inner {
                Some(inner) => write!(f, "(?{}:{})", flags, inner),
                None => write!(f, "(?{})", flags),
            },
            RegexByte::WordBoundary => write!(f, "#"),
            RegexByte::StartOfLine => write!(f, "^"),
            RegexByte::EndOfLine => write!(f, "$"),
            RegexByte::StartOfInput => write!(f, "\\A"),
            RegexByte::EndOfInput => write!(f, "\\Z"),
            RegexByte::EndOfInputStrict => write!(f, "\\z"),
            RegexByte::RewriteRule {
                pattern,
                replacement,
                context,
                weight,
            } => {
                write!(f, "{} -> {}", pattern, replacement)?;
                if let Some(ctx) = context {
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
                if *weight != 0.0 {
                    write!(f, " [{:.2}]", weight)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regex_literal() {
        let r = Regex::literal("phone");
        assert!(!r.is_empty());
        assert_eq!(r.to_string(), "phone");
    }

    #[test]
    fn test_regex_empty() {
        let r = Regex::empty();
        assert!(r.is_empty());
        assert_eq!(r.to_string(), "");
    }

    #[test]
    fn test_regex_char() {
        let r = Regex::char('a');
        assert_eq!(r.to_string(), "a");
    }

    #[test]
    fn test_regex_any() {
        let r = Regex::any();
        assert_eq!(r.to_string(), ".");
    }

    #[test]
    fn test_regex_char_class() {
        let class = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']);
        let r = Regex::char_class(class);
        assert_eq!(r.to_string(), "[aeiou]");
    }

    #[test]
    fn test_regex_concat() {
        let a = Regex::char('a');
        let b = Regex::char('b');
        let r = Regex::concat(a, b);
        assert_eq!(r.to_string(), "ab");
    }

    #[test]
    fn test_regex_alt() {
        let ph = Regex::literal("ph");
        let f = Regex::char('f');
        let r = Regex::alt(ph, f);
        assert_eq!(r.to_string(), "(ph|f)");
    }

    #[test]
    fn test_regex_star() {
        let a = Regex::char('a');
        let r = Regex::star(a);
        assert_eq!(r.to_string(), "a*");
    }

    #[test]
    fn test_regex_plus() {
        let a = Regex::char('a');
        let r = Regex::plus(a);
        assert_eq!(r.to_string(), "a+");
    }

    #[test]
    fn test_regex_optional() {
        let a = Regex::char('a');
        let r = Regex::optional(a);
        assert_eq!(r.to_string(), "a?");
    }

    #[test]
    fn test_regex_repeat_exact() {
        let a = Regex::char('a');
        let r = Regex::repeat_exact(a, 3);
        assert_eq!(r.to_string(), "a{3}");
    }

    #[test]
    fn test_regex_repeat_range() {
        let a = Regex::char('a');
        let r = Regex::repeat_range(a.clone(), 2, Some(4));
        assert_eq!(r.to_string(), "a{2,4}");

        let r2 = Regex::repeat_range(a, 2, None);
        assert_eq!(r2.to_string(), "a{2,}");
    }

    #[test]
    fn test_regex_word_boundary() {
        let r = Regex::word_boundary();
        assert_eq!(r.to_string(), "#");
    }

    #[test]
    fn test_regex_rewrite_rule_simple() {
        // ph -> f
        let r = Regex::rewrite_rule(Regex::literal("ph"), Regex::char('f'), None, 0.0);
        assert!(r.is_rewrite_rule());
        assert_eq!(r.to_string(), "ph -> f");
    }

    #[test]
    fn test_regex_rewrite_rule_with_context() {
        // c -> s / _[ei]
        let vowels = CharClassChar::from_chars(&['e', 'i']);
        let context = ContextPredicate::lookahead(Regex::char_class(vowels));
        let r = Regex::rewrite_rule(Regex::char('c'), Regex::char('s'), Some(context), 0.0);
        assert_eq!(r.to_string(), "c -> s / _[ei]");
    }

    #[test]
    fn test_regex_rewrite_rule_with_weight() {
        // th -> t [0.15]
        let r = Regex::rewrite_rule(Regex::literal("th"), Regex::char('t'), None, 0.15);
        assert_eq!(r.to_string(), "th -> t [0.15]");
    }

    #[test]
    fn test_regex_rewrite_rule_word_end() {
        // e -> (empty) / _#  (silent e at word end)
        let context = ContextPredicate::word_end();
        let r = Regex::rewrite_rule(Regex::char('e'), Regex::empty(), Some(context), 0.0);
        assert_eq!(r.to_string(), "e ->  / _#");
    }

    #[test]
    fn test_regex_escape_special_chars() {
        let r = Regex::char('.');
        assert_eq!(r.to_string(), "\\.");

        let r2 = Regex::char('*');
        assert_eq!(r2.to_string(), "\\*");

        let r3 = Regex::char('[');
        assert_eq!(r3.to_string(), "\\[");
    }

    #[test]
    fn test_regex_size() {
        let r = Regex::literal("phone");
        assert_eq!(r.size(), 9); // 5 chars + 4 concats

        let r2 = Regex::star(Regex::char('a'));
        assert_eq!(r2.size(), 2); // star + char

        let r3 = Regex::alt(Regex::char('a'), Regex::char('b'));
        assert_eq!(r3.size(), 3); // alt + 2 chars
    }

    // Byte-level tests

    #[test]
    fn test_regex_byte_literal() {
        let r = RegexByte::literal(b"phone");
        assert!(!r.is_empty());
        assert_eq!(r.to_string(), "phone");
    }

    #[test]
    fn test_regex_byte_empty() {
        let r = RegexByte::empty();
        assert!(r.is_empty());
        assert_eq!(r.to_string(), "");
    }

    #[test]
    fn test_regex_byte_rewrite_rule() {
        // ph -> f
        let r =
            RegexByte::rewrite_rule(RegexByte::literal(b"ph"), RegexByte::byte(b'f'), None, 0.0);
        assert!(r.is_rewrite_rule());
        assert_eq!(r.to_string(), "ph -> f");
    }

    #[test]
    fn test_context_predicate_new() {
        let left = Regex::char('a');
        let right = Regex::char('b');
        let ctx = ContextPredicate::new(Some(left), Some(right));
        assert!(ctx.left.is_some());
        assert!(ctx.right.is_some());
    }

    #[test]
    fn test_context_predicate_lookahead() {
        let right = Regex::char('b');
        let ctx = ContextPredicate::lookahead(right);
        assert!(ctx.left.is_none());
        assert!(ctx.right.is_some());
    }

    #[test]
    fn test_context_predicate_lookbehind() {
        let left = Regex::char('a');
        let ctx = ContextPredicate::lookbehind(left);
        assert!(ctx.left.is_some());
        assert!(ctx.right.is_none());
    }
}
