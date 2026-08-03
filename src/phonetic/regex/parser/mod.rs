//! Recursive descent parser for phonetic regular expressions.
//!
//! This parser implements the grammar defined in the AST module,
//! supporting both standard regex constructs and phonetic rewrite rules.
//!
//! The parser is split along the char/byte duality:
//! - `char` holds the Unicode [`Parser`] with full feature support
//!   (capturing groups, scoped flags, group references).
//! - `byte` holds the leaner ASCII [`ParserByte`].
//! - `common` holds shared helpers and the public [`SymbolTable`]
//!   re-export.
//!
//! # Grammar
//!
//! ```text
//! regex       ::= alternation
//! alternation ::= concatenation ('|' concatenation)*
//! concatenation ::= quantified+
//! quantified  ::= primary quantifier?
//! quantifier  ::= '*' | '+' | '?' | '{' number '}' | '{' number ',' number? '}'
//! primary     ::= '(' regex ')' | char_class | literal | '.' | '#'
//! char_class  ::= '[' '^'? (char | char '-' char)+ ']'
//!
//! rewrite_rule ::= pattern '->' replacement context? weight?
//! context      ::= '/' left_context? '_' right_context?
//! weight       ::= '[' float ']'
//! ```

pub mod byte;
pub mod char;
pub mod common;

#[cfg(test)]
mod tests;

pub use byte::ParserByte;
pub use char::Parser;
pub use common::{SymbolTable, MAX_PATTERN_SIZE};

use crate::phonetic::regex::ast::{Regex, RegexByte};
use crate::phonetic::regex::error::ParseResult;

/// Parse a regex pattern string.
pub fn parse(input: &str) -> ParseResult<Regex> {
    Parser::new(input).parse()
}

/// Parse a rewrite rule string.
pub fn parse_rule(input: &str) -> ParseResult<Regex> {
    Parser::new(input).parse_rewrite_rule()
}

/// Parse multiple rewrite rules.
pub fn parse_rules(input: &str) -> ParseResult<Vec<Regex>> {
    Parser::new(input).parse_rule_set()
}

/// Parse a byte-level regex pattern.
pub fn parse_bytes(input: &[u8]) -> ParseResult<RegexByte> {
    ParserByte::new(input).parse()
}

/// Parse a byte-level rewrite rule.
pub fn parse_rule_bytes(input: &[u8]) -> ParseResult<RegexByte> {
    ParserByte::new(input).parse_rewrite_rule()
}
