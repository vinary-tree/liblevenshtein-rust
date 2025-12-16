//! Parser and compiler for `.llre` (LibLevenshtein Regex Expression) files.
//!
//! This module provides support for the `.llre` file format, which contains a
//! single regex pattern with optional metadata, imports, and global flags.
//!
//! # Features
//!
//! - **Single pattern per file** with metadata (@name, @version, @author, @description)
//! - **Import symbols** from `.llev` files via @import directive
//! - **Global flags** (@flags multiline, dotall, case_insensitive)
//! - **AOT compilation** to binary format for instant loading
//! - **Full anchor support** (^, $, \A, \Z, \z)
//!
//! # Example `.llre` File
//!
//! ```text
//! # Example .llre file
//! @name "Email Validator"
//! @version "1.0"
//! @author "LibLevenshtein Team"
//! @description "Validates email addresses"
//!
//! # Import symbols from .llev files
//! @import "phonetic/symbols.llev"
//! @import "phonetic/english.llev" as en
//!
//! # Global flags
//! @flags multiline
//!
//! # The regex pattern
//! ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
//! ```
//!
//! # Usage
//!
//! ## Parse and Compile
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::llre::{parse_str, compile};
//!
//! // Parse from string
//! let file = parse_str(r#"
//!     @name "Hello Pattern"
//!     ^hello$
//! "#)?;
//!
//! // Compile to NFA
//! let compiled = compile(&file)?;
//!
//! // Match strings
//! assert!(compiled.matches("hello"));
//! assert!(!compiled.matches("world"));
//! ```
//!
//! ## Load from File
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::llre::load_file;
//!
//! let file = load_file("patterns/email.llre")?;
//! let compiled = compile(&file)?;
//! ```
//!
//! ## AOT Compilation
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::llre::{compile, save, load};
//!
//! // Compile and save to binary
//! let file = parse_str("^hello$")?;
//! let compiled = compile(&file)?;
//! save(&compiled, "pattern.llre.bin")?;
//!
//! // Load pre-compiled (instant, no parsing)
//! let loaded = load("pattern.llre.bin")?;
//! assert!(loaded.matches("hello"));
//! ```
//!
//! ## Quick Matching
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::llre::is_match;
//!
//! assert!(is_match("^hello$", "hello")?);
//! assert!(!is_match("^hello$", "world")?);
//! ```
//!
//! # File Format Grammar
//!
//! ```ebnf
//! llre_file     = { directive } , pattern ;
//! directive     = "@" , directive_name , directive_value ;
//! directive_name = "name" | "version" | "author" | "description" | "import" | "flags" ;
//! import_spec   = string_literal , [ "as" , identifier ] ;
//! flag_list     = flag_name , { "," , flag_name } ;
//! flag_name     = "multiline" | "dotall" | "case_insensitive" | "unicode" ;
//! pattern       = regex_expression ;
//! ```
//!
//! # Flags
//!
//! | Flag | Short | Description |
//! |------|-------|-------------|
//! | multiline | m | `^` and `$` match at line boundaries |
//! | dotall | s | `.` matches newlines |
//! | case_insensitive | i | Case-insensitive matching |
//! | unicode | u | Unicode-aware character classes |

pub mod ast;
pub mod compiled;
pub mod error;
pub mod loader;
pub mod nfa_compiler;
pub mod parser;

// Re-export main types
pub use ast::{
    Directive, FileMetadata, ImportDirective, LLreFile, LLreFlags, ResolvedImport, SymbolTable,
};
pub use error::{LLreError, LLreErrorKind, LLreResult, Position};
pub use loader::{load_file, load_file_with_config, Loader, LoaderConfig};
pub use nfa_compiler::{
    compile, compile_pattern, compile_pattern_with_flags, compile_with_options, is_match,
    is_match_multiline, CompileOptions, CompiledNFA,
};
pub use parser::{parse_str, Parser};

// Re-export serialization functions (requires serialization feature)
#[cfg(feature = "serialization")]
pub use compiled::{from_bytes, load, save, to_bytes, CompiledMetadata, MAGIC, VERSION};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_workflow() {
        // Parse
        let file = parse_str(r#"
            @name "Test Pattern"
            @version "1.0"
            ^hello$
        "#).expect("Failed to parse");

        assert_eq!(file.metadata.name, Some("Test Pattern".to_string()));
        assert_eq!(file.metadata.version, Some("1.0".to_string()));

        // Compile
        let compiled = compile(&file).expect("Failed to compile");

        // Match
        assert!(compiled.matches("hello"));
        assert!(!compiled.matches("world"));
    }

    #[test]
    fn test_quick_match() {
        assert!(is_match("[a-z]+", "hello").expect("Failed"));
        assert!(!is_match("[a-z]+", "123").expect("Failed"));
    }

    #[test]
    fn test_anchors() {
        // Start anchor
        assert!(is_match("^hello", "hello world").expect("Failed"));
        assert!(!is_match("^hello", "say hello").expect("Failed"));

        // End anchor
        assert!(is_match("world$", "hello world").expect("Failed"));
        assert!(!is_match("world$", "world hello").expect("Failed"));

        // Both anchors
        assert!(is_match("^hello$", "hello").expect("Failed"));
        assert!(!is_match("^hello$", "hello world").expect("Failed"));
    }

    #[test]
    fn test_multiline_flag() {
        let file = parse_str(r#"
            @flags multiline
            ^line$
        "#).expect("Failed to parse");

        let compiled = compile(&file).expect("Failed to compile");
        assert!(compiled.multiline);

        // In multiline mode, ^ and $ should match at line boundaries
        assert!(compiled.matches("line"));
    }

    #[test]
    fn test_imports_parsed() {
        let file = parse_str(r#"
            @import "symbols.llev"
            @import "english.llev" as en
            ^test$
        "#).expect("Failed to parse");

        assert_eq!(file.imports.len(), 2);
        assert_eq!(file.imports[0].path, "symbols.llev");
        assert!(file.imports[0].alias.is_none());
        assert_eq!(file.imports[1].path, "english.llev");
        assert_eq!(file.imports[1].alias, Some("en".to_string()));
    }

    #[test]
    fn test_compile_options() {
        let file = parse_str("^[a-z]+$").expect("Failed to parse");
        let options = CompileOptions {
            max_states: Some(1000),
            optimize: true,
            ..Default::default()
        };

        let compiled = compile_with_options(&file, &options).expect("Failed to compile");
        assert!(compiled.matches("hello"));
    }

    #[test]
    fn test_character_classes() {
        assert!(is_match("[aeiou]+", "aeiou").expect("Failed"));
        assert!(!is_match("[aeiou]+", "xyz").expect("Failed"));

        assert!(is_match("[^aeiou]+", "xyz").expect("Failed"));
        assert!(!is_match("[^aeiou]+", "aeiou").expect("Failed"));
    }

    #[test]
    fn test_quantifiers() {
        // Zero or more
        assert!(is_match("a*", "").expect("Failed"));
        assert!(is_match("a*", "aaa").expect("Failed"));

        // One or more
        assert!(!is_match("^a+$", "").expect("Failed"));
        assert!(is_match("a+", "aaa").expect("Failed"));

        // Optional
        assert!(is_match("ab?c", "ac").expect("Failed"));
        assert!(is_match("ab?c", "abc").expect("Failed"));

        // Specific counts
        assert!(is_match("^a{3}$", "aaa").expect("Failed"));
        assert!(!is_match("^a{3}$", "aa").expect("Failed"));
    }

    #[test]
    fn test_alternation() {
        assert!(is_match("cat|dog", "cat").expect("Failed"));
        assert!(is_match("cat|dog", "dog").expect("Failed"));
        assert!(!is_match("^(cat|dog)$", "bird").expect("Failed"));
    }

    #[test]
    fn test_groups() {
        assert!(is_match("(ab)+", "abab").expect("Failed"));
        assert!(!is_match("^(ab)+$", "aba").expect("Failed"));
    }

    #[test]
    fn test_escape_sequences() {
        // Literal dot
        assert!(is_match(r"a\.b", "a.b").expect("Failed"));
        assert!(!is_match(r"^a\.b$", "axb").expect("Failed"));

        // Word boundary would need NFA support
    }

    #[test]
    fn test_complex_pattern() {
        // Email-like pattern
        let pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$";
        assert!(is_match(pattern, "test@example.com").expect("Failed"));
        assert!(is_match(pattern, "user.name@sub.domain.org").expect("Failed"));
        assert!(!is_match(pattern, "invalid").expect("Failed"));
        assert!(!is_match(pattern, "@example.com").expect("Failed"));
    }
}
