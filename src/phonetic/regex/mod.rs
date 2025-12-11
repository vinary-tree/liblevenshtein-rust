//! Phonetic regular expression support.
//!
//! This module provides a parser and AST for phonetic regex patterns,
//! supporting both standard regex constructs and phonetic rewrite rules.
//!
//! # Syntax
//!
//! ## Standard Regex
//!
//! | Syntax | Description | Example |
//! |--------|-------------|---------|
//! | `abc` | Literal characters | `phone` |
//! | `a\|b` | Alternation | `ph\|f` |
//! | `(...)` | Grouping | `(ph\|f)one` |
//! | `[abc]` | Character class | `[aeiou]` |
//! | `[^abc]` | Negated class | `[^aeiou]` |
//! | `[a-z]` | Character range | `[a-z]` |
//! | `.` | Any character | `a.b` |
//! | `*` | Zero or more | `a*` |
//! | `+` | One or more | `a+` |
//! | `?` | Zero or one | `a?` |
//! | `{n}` | Exactly n | `a{3}` |
//! | `{n,}` | At least n | `a{2,}` |
//! | `{,m}` | At most m | `a{,3}` |
//! | `{n,m}` | Between n and m | `a{2,4}` |
//!
//! ## Phonetic Extensions
//!
//! | Syntax | Description | Example |
//! |--------|-------------|---------|
//! | `a -> b` | Rewrite rule | `ph -> f` |
//! | `/ _X` | Right context (lookahead) | `c -> s / _[ei]` |
//! | `/ X_` | Left context (lookbehind) | `s -> z / [aeiou]_` |
//! | `#` | Word boundary | `e -> / _#` |
//! | `[w]` | Weight/cost | `th -> t [0.15]` |
//!
//! # Examples
//!
//! ## Parsing a simple pattern
//!
//! ```ignore
//! use liblevenshtein::phonetic::regex::parse;
//!
//! let pattern = parse("(ph|f)one").unwrap();
//! assert_eq!(pattern.to_string(), "((ph|f))one");
//! ```
//!
//! ## Parsing a rewrite rule
//!
//! ```ignore
//! use liblevenshtein::phonetic::regex::parse_rule;
//!
//! // ph -> f (phone -> fone)
//! let rule = parse_rule("ph -> f").unwrap();
//! assert!(rule.is_rewrite_rule());
//!
//! // c -> s before e or i (city -> sity)
//! let rule = parse_rule("c -> s / _[ei]").unwrap();
//!
//! // Silent e at word end (phone -> phon)
//! let rule = parse_rule("e -> / _#").unwrap();
//! ```
//!
//! ## Parsing multiple rules
//!
//! ```ignore
//! use liblevenshtein::phonetic::regex::parse_rules;
//!
//! let rules = parse_rules(r#"
//!     ph -> f
//!     c -> s / _[ei]
//!     e -> / _#
//! "#).unwrap();
//! assert_eq!(rules.len(), 3);
//! ```
//!
//! # Design
//!
//! The parser is a recursive descent parser that produces an AST representation.
//! The AST can then be compiled to an NFA using the `nfa::compiler` module.
//!
//! Following the codebase pattern, both character-level (`Regex`, `Parser`) and
//! byte-level (`RegexByte`, `ParserByte`) implementations are provided.

pub mod ast;
pub mod error;
pub mod lexer;
pub mod parser;

// Re-export main types (character-level)
pub use ast::{ContextPredicate, Regex};
pub use error::{ParseError, ParseErrorKind, ParseResult, Position};
pub use lexer::{Lexer, Token};
pub use parser::{parse, parse_rule, parse_rules, Parser};

// Re-export byte-level types
pub use ast::{ContextPredicateByte, RegexByte};
pub use lexer::{LexerByte, TokenByte};
pub use parser::{parse_bytes, parse_rule_bytes, ParserByte};
