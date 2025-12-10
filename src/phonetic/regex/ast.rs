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

    /// Capturing group (for future use)
    Group(Box<Regex>),

    /// Word boundary assertion (`#` at start or end)
    WordBoundary,

    /// Rewrite rule: pattern -> replacement with optional context and weight
    RewriteRule {
        pattern: Box<Regex>,
        replacement: Box<Regex>,
        context: Option<Box<ContextPredicate>>,
        weight: f64,
    },
}

// ============================================================================
// Syllable Conditions
// ============================================================================

/// Syllable-based condition for phonetic rules.
///
/// These conditions allow rules to be constrained based on syllable structure,
/// which is important for rules like vowel length and Y pronunciation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyllableCondition {
    /// Word has exactly one syllable (e.g., "fly", "ply")
    Monosyllable,
    /// Word has more than one syllable (e.g., "happy", "flying")
    Polysyllable,
    /// Current syllable ends in a vowel (vowel is long)
    OpenSyllable,
    /// Current syllable ends in a consonant (vowel is short)
    ClosedSyllable,
    /// Match position is in the final syllable
    FinalSyllable,
    /// Match position is in the initial syllable
    InitialSyllable,
}

impl fmt::Display for SyllableCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyllableCondition::Monosyllable => write!(f, "monosyllable"),
            SyllableCondition::Polysyllable => write!(f, "polysyllable"),
            SyllableCondition::OpenSyllable => write!(f, "open_syllable"),
            SyllableCondition::ClosedSyllable => write!(f, "closed_syllable"),
            SyllableCondition::FinalSyllable => write!(f, "final_syllable"),
            SyllableCondition::InitialSyllable => write!(f, "initial_syllable"),
        }
    }
}

/// A syllable expression with logical operators.
///
/// Allows combining syllable conditions with AND, OR, NOT.
#[derive(Debug, Clone, PartialEq)]
pub enum SyllableExpr {
    /// Simple syllable condition
    Cond(SyllableCondition),
    /// Both expressions must be true
    And(Box<SyllableExpr>, Box<SyllableExpr>),
    /// Either expression must be true
    Or(Box<SyllableExpr>, Box<SyllableExpr>),
    /// Expression must be false
    Not(Box<SyllableExpr>),
}

impl SyllableExpr {
    /// Create a simple condition expression.
    pub fn cond(cond: SyllableCondition) -> Self {
        SyllableExpr::Cond(cond)
    }

    /// Create an AND expression.
    pub fn and(left: SyllableExpr, right: SyllableExpr) -> Self {
        SyllableExpr::And(Box::new(left), Box::new(right))
    }

    /// Create an OR expression.
    pub fn or(left: SyllableExpr, right: SyllableExpr) -> Self {
        SyllableExpr::Or(Box::new(left), Box::new(right))
    }

    /// Create a NOT expression.
    pub fn not(inner: SyllableExpr) -> Self {
        SyllableExpr::Not(Box::new(inner))
    }

    /// Get the estimated size/complexity of this syllable expression.
    pub fn size(&self) -> usize {
        match self {
            SyllableExpr::Cond(_) => 1,
            SyllableExpr::And(a, b) | SyllableExpr::Or(a, b) => 1 + a.size() + b.size(),
            SyllableExpr::Not(inner) => 1 + inner.size(),
        }
    }
}

impl fmt::Display for SyllableExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyllableExpr::Cond(cond) => write!(f, "{}", cond),
            SyllableExpr::And(a, b) => write!(f, "({} & {})", a, b),
            SyllableExpr::Or(a, b) => write!(f, "({} | {})", a, b),
            SyllableExpr::Not(inner) => write!(f, "!{}", inner),
        }
    }
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

    /// Create a capturing group.
    pub fn group(inner: Regex) -> Self {
        Regex::Group(Box::new(inner))
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
    pub fn size(&self) -> usize {
        match self {
            Regex::Empty | Regex::Char(_) | Regex::Any | Regex::WordBoundary => 1,
            Regex::CharClass(_) => 1,
            Regex::Concat(a, b) | Regex::Alt(a, b) => 1 + a.size() + b.size(),
            Regex::Star(inner)
            | Regex::Plus(inner)
            | Regex::Optional(inner)
            | Regex::Group(inner) => 1 + inner.size(),
            Regex::RepeatExact(inner, _) | Regex::RepeatRange(inner, _, _) => 1 + inner.size(),
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
            Regex::Group(inner) => write!(f, "({})", inner),
            Regex::WordBoundary => write!(f, "#"),
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

    /// Capturing group (for future use)
    Group(Box<RegexByte>),

    /// Word boundary assertion (`#` at start or end)
    WordBoundary,

    /// Rewrite rule: pattern -> replacement with optional context and weight
    RewriteRule {
        pattern: Box<RegexByte>,
        replacement: Box<RegexByte>,
        context: Option<Box<ContextPredicateByte>>,
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

    /// Create a capturing group.
    pub fn group(inner: RegexByte) -> Self {
        RegexByte::Group(Box::new(inner))
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
    pub fn size(&self) -> usize {
        match self {
            RegexByte::Empty | RegexByte::Byte(_) | RegexByte::Any | RegexByte::WordBoundary => 1,
            RegexByte::ByteClass(_) => 1,
            RegexByte::Concat(a, b) | RegexByte::Alt(a, b) => 1 + a.size() + b.size(),
            RegexByte::Star(inner)
            | RegexByte::Plus(inner)
            | RegexByte::Optional(inner)
            | RegexByte::Group(inner) => 1 + inner.size(),
            RegexByte::RepeatExact(inner, _) | RegexByte::RepeatRange(inner, _, _) => {
                1 + inner.size()
            }
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
            RegexByte::Group(inner) => write!(f, "({})", inner),
            RegexByte::WordBoundary => write!(f, "#"),
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
    fn test_regex_group() {
        let inner = Regex::alt(Regex::char('a'), Regex::char('b'));
        let r = Regex::group(inner);
        assert_eq!(r.to_string(), "((a|b))");
    }

    #[test]
    fn test_regex_word_boundary() {
        let r = Regex::word_boundary();
        assert_eq!(r.to_string(), "#");
    }

    #[test]
    fn test_regex_rewrite_rule_simple() {
        // ph -> f
        let r = Regex::rewrite_rule(
            Regex::literal("ph"),
            Regex::char('f'),
            None,
            0.0,
        );
        assert!(r.is_rewrite_rule());
        assert_eq!(r.to_string(), "ph -> f");
    }

    #[test]
    fn test_regex_rewrite_rule_with_context() {
        // c -> s / _[ei]
        let vowels = CharClassChar::from_chars(&['e', 'i']);
        let context = ContextPredicate::lookahead(Regex::char_class(vowels));
        let r = Regex::rewrite_rule(
            Regex::char('c'),
            Regex::char('s'),
            Some(context),
            0.0,
        );
        assert_eq!(r.to_string(), "c -> s / _[ei]");
    }

    #[test]
    fn test_regex_rewrite_rule_with_weight() {
        // th -> t [0.15]
        let r = Regex::rewrite_rule(
            Regex::literal("th"),
            Regex::char('t'),
            None,
            0.15,
        );
        assert_eq!(r.to_string(), "th -> t [0.15]");
    }

    #[test]
    fn test_regex_rewrite_rule_word_end() {
        // e -> (empty) / _#  (silent e at word end)
        let context = ContextPredicate::word_end();
        let r = Regex::rewrite_rule(
            Regex::char('e'),
            Regex::empty(),
            Some(context),
            0.0,
        );
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
        let r = RegexByte::rewrite_rule(
            RegexByte::literal(b"ph"),
            RegexByte::byte(b'f'),
            None,
            0.0,
        );
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
