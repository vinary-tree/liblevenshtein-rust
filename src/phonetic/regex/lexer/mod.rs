//! Lexer for phonetic regular expressions.
//!
//! Tokenizes input strings into a stream of tokens for the parser.

mod char;
mod byte;
#[cfg(test)]
mod tests;

pub use char::{Lexer, Token};
pub use byte::{LexerByte, TokenByte};

// Re-export ParsedFlags for users of this module
pub use crate::phonetic::common::flags::ParsedFlags;
