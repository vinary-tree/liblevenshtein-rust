//! Parser and loader for `.llev` rule files.
//!
//! This module provides support for loading custom phonetic rewrite rules from
//! `.llev` (or `.llv`) files. The file format is human-readable and supports:
//!
//! - File metadata (`@name`, `@version`, `@author`, `@description`)
//! - Symbol definitions (`@define`) for reusable patterns
//! - Include directives (`@include`) for modular rule organization
//! - Rule definitions with optional metadata (id, name, weight, group, enabled)
//! - Comments (line `#`, `//` and block `/* */`)
//!
//! # Example `.llev` File
//!
//! ```text
//! # English Phonetic Spelling Rules
//! @name "English Phonetic Rules"
//! @version "1.0"
//!
//! # Define reusable character classes
//! @define FRONT_VOWEL = [ei]
//!
//! # ============================================================
//! # Orthography Rules
//! # ============================================================
//!
//! [id: 1, name: "ph to f"]
//! ph -> f;  # phone -> fone
//!
//! [id: 20, name: "soft c", weight: 0.0]
//! c -> s / _FRONT_VOWEL;  # city -> sity
//!
//! [id: 21, name: "hard c"]
//! c -> k;  # cat -> kat
//!
//! [id: 33, name: "silent final e"]
//! e -> / _#;  # make -> mak
//!
//! /* Silent gh - can be silent (night) or /f/ (enough)
//!    For simplicity, delete everywhere */
//! [id: 34, name: "silent gh"]
//! gh -> ;
//!
//! # Include additional rules
//! @include "additional_rules.llv"
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use liblevenshtein::phonetic::llev::{load_file, RuleSet};
//! use liblevenshtein::phonetic::{
//!     apply_rules_seq, rules_to_nfa, PhoneByte, MAX_EXPANSION_FACTOR,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load rules from a file
//! let file = load_file("english.llev")?;
//! let rules = RuleSet::from_llev(&file)?;
//!
//! // Use the byte-level rules with the low-level rewrite engine.
//! let input = vec![
//!     PhoneByte::Consonant(b'p'),
//!     PhoneByte::Consonant(b'h'),
//!     PhoneByte::Vowel(b'o'),
//!     PhoneByte::Consonant(b'n'),
//!     PhoneByte::Vowel(b'e'),
//! ];
//! let fuel = input.len() * rules.len() * MAX_EXPANSION_FACTOR;
//! let normalized = apply_rules_seq(&rules.rules, &input, fuel)
//!     .expect("rewriting returns the reached fixed point");
//!
//! // Or compile the rule patterns to an NFA.
//! let nfa = rules_to_nfa(&rules.rules);
//! assert!(nfa.accepts(b"ph"));
//! # Ok(())
//! # }
//! ```
//!
//! # AOT Compilation
//!
//! Rules can be compiled to binary format for faster loading:
//!
//! ```bash
//! # Compile to binary
//! liblevenshtein -c english.llev -o english.llev.bin
//!
//! # Load compiled rules (faster startup)
//! let rules = RuleSet::from_compiled("english.llev.bin")?;
//! ```
//!
//! # File Format Grammar
//!
//! See [`ast`] module documentation for the complete eBNF grammar.

pub mod ast;
#[cfg(feature = "serialization")]
pub mod compiled;
pub mod error;
pub mod lexer;
pub mod loader;
pub mod parser;
pub mod ruleset;

// Re-export main types
pub use ast::{
    ContextAST, Expression, FileMetadata, IncludeDirective, LLevFile, RewriteRuleAST,
    RuleDefinition, RuleMetadata, SymbolDef,
};
pub use error::{LLevError, LLevErrorKind, LLevResult, Position};
pub use loader::{load_file, load_file_with_includes, Loader, LoaderConfig};
pub use parser::{parse_expression, parse_str, Parser};
pub use ruleset::{RuleSet, RuleSetChar};

// Re-export compiled module functions (requires serialization feature)
#[cfg(feature = "serialization")]
pub use compiled::{
    // Byte-level
    from_bytes,
    // Character-level
    from_bytes_char,
    load,
    load_char,
    save,
    save_char,
    to_bytes,
    to_bytes_char,
};
