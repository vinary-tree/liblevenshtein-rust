//! Recursive descent parser for `.llev` rule files.
//!
//! This parser transforms a stream of tokens into an AST representing
//! the `.llev` file structure including directives, metadata, and rules.
//!
//! # Grammar Overview
//!
//! ```text
//! llev_file       ::= (directive | rule_definition | NEWLINE)*
//! directive       ::= "@" DIRECTIVE_NAME directive_value
//! rule_definition ::= metadata_block? rewrite_rule TERMINATOR?
//! rewrite_rule    ::= pattern ARROW replacement context? weight_suffix?
//! pattern         ::= expression
//! replacement     ::= expression | EMPTY
//! context         ::= "/" left_context? "_" right_context?
//! expression      ::= alternation
//! alternation     ::= concatenation ("|" concatenation)*
//! concatenation   ::= quantified*
//! quantified      ::= primary quantifier?
//! primary         ::= char | char_class | group | any | symbol_ref | boundary
//! ```

use std::collections::HashMap;

use super::ast::Expression;
use super::lexer::Lexer;

mod context;
mod expr;
mod file;
mod helpers;

#[cfg(test)]
mod tests;

pub use expr::parse_expression;
pub use file::{parse_str, parse_str_with_symbols};

/// Parser for `.llev` files.
pub struct Parser<'a> {
    pub(super) lexer: Lexer<'a>,
    /// Symbol table for `@define` symbols
    pub(super) symbols: HashMap<String, Expression>,
}
