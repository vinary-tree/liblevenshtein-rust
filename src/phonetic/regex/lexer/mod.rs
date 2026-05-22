//! Lexer for phonetic regular expressions.
//!
//! Tokenizes input strings into a stream of tokens for the parser.

mod byte;
mod char;
#[cfg(test)]
mod tests;

pub use byte::{LexerByte, TokenByte};
pub use char::{Lexer, Token};

// Re-export ParsedFlags for users of this module
pub use crate::phonetic::common::flags::ParsedFlags;
