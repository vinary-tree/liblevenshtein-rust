//! Shared types and utilities for phonetic parsing modules.
//!
//! This module contains types and functions that are shared between the
//! `regex` and `llev` parsing modules, reducing code duplication.

pub mod context;
pub mod directives;
pub mod flags;
pub mod parsing;
pub mod phonetic_unit;
pub mod position;
pub mod syllable;
pub mod traits;
pub mod utils;

// Re-export commonly used types
pub use context::ContextExpr;
pub use directives::{FileMetadata, MetadataDirective};
pub use flags::ParsedFlags;
pub use phonetic_unit::{is_consonant, to_lowercase_vec, PhoneticUnit};
pub use position::Position;
pub use syllable::{SyllableCondition, SyllableExpr};
pub use traits::{ContextParser, GrammarParser, LexerLike, SyllableParser, TokenLike};
pub use utils::{
    is_user_symbol_name, negate_char_class, resolve_feature_bundle_chars, SymbolTable,
};

// Re-export parsing functions
pub use parsing::{
    parse_context_and, parse_context_expr, parse_context_not, parse_context_or,
    parse_context_primary, parse_syllable_and, parse_syllable_expr, parse_syllable_not,
    parse_syllable_or, parse_syllable_primary,
};
