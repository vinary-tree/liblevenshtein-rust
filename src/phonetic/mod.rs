//! Phonetic rewrite rules for approximate string matching.
//!
//! This module implements verified phonetic transformation rules based on
//! Zompist's English spelling rules, with formal correctness guarantees from
//! Coq/Rocq proofs.
//!
//! # Rule Application Functions
//!
//! | Function | Use Case |
//! |----------|----------|
//! | [`apply_rules_seq`] | **Default** - use for all typical workloads (dictionary words, phrases) |
//! | [`apply_rules_seq_optimized`] | **Opt-in** - use only for very long strings (100+ chars) with repetitive patterns |
//! | [`apply_rules_seq_opt`] | **Deprecated** - now just calls `apply_rules_seq` |
//!
//! The position skipping optimization in `apply_rules_seq_optimized` provides up to
//! 26.6× speedup for synthetic repetitive strings, but causes 1-15% overhead for
//! typical English dictionary words. See `docs/verification/phonetic/POSITION_SKIPPING_BENCHMARK_RESULTS.md`
//! for detailed benchmark results.
//!
//! # Formal Verification
//!
//! All algorithms in this module are proven correct in Coq/Rocq. The proofs
//! establish five critical properties:
//!
//! 1. **Well-formedness** (Theorem 1, `docs/verification/phonetic/zompist_rules.v:285`)
//!    - All rules satisfy structural constraints
//!    - Pattern and replacement sequences are valid
//!    - Context specifications are well-formed
//!
//! 2. **Bounded Expansion** (Theorem 2, `docs/verification/phonetic/zompist_rules.v:425`)
//!    - String length growth is bounded by a constant factor
//!    - Maximum expansion: `length(output) ≤ length(input) + 20`
//!    - Prevents unbounded memory growth
//!
//! 3. **Non-Confluence** (Theorem 3, `docs/verification/phonetic/zompist_rules.v:491`)
//!    - Rule application order matters
//!    - Different orders can produce different results
//!    - Sequential application is the canonical approach
//!
//! 4. **Termination** (Theorem 4, `docs/verification/phonetic/zompist_rules.v:569`)
//!    - Sequential application always terminates
//!    - No infinite rewrite loops
//!    - Guaranteed to reach a fixed point
//!
//! 5. **Idempotence** (Theorem 5, `docs/verification/phonetic/zompist_rules.v:615`)
//!    - Fixed points remain unchanged
//!    - Applying rules to the output produces the same output
//!    - Stable transformation semantics
//!
//! # Usage
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::{apply_rules_seq, ORTHOGRAPHY_RULES};
//!
//! let input = "enough";
//! let phonetic = apply_rules_seq(&ORTHOGRAPHY_RULES, input);
//! assert_eq!(phonetic, "enuf");
//! ```
//!
//! # Rule Sets
//!
//! Three built-in rule sets are provided:
//!
//! - [`ORTHOGRAPHY_RULES`] - Simplifies English spelling (e.g., "gh" → "f")
//! - [`PHONETIC_RULES`] - Maps to phonetic representations
//! - [`TEST_RULES`] - Example rules for testing and validation
//!
//! # Custom Rules with `.llev` Files
//!
//! Load custom phonetic rules from `.llev` files:
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::{parse_str, RuleSetChar, apply_rules_seq_char};
//!
//! // Parse rules from a string
//! let file = parse_str(r#"
//!     @name "Custom Rules"
//!
//!     [id: 1, name: "ph to f"]
//!     ph -> f;
//!
//!     [id: 2, name: "soft c"]
//!     c -> s / _[ei];
//! "#)?;
//!
//! // Convert to runtime rule set
//! let ruleset = RuleSetChar::from_llev(&file)?;
//!
//! // Apply rules
//! let result = apply_rules_seq_char(&ruleset.rules, "phone");
//! assert_eq!(result, "fone");
//! ```
//!
//! See the [`llev`] module for complete documentation on the file format.
//!
//! # Implementation Note
//!
//! Following the existing codebase pattern, this module provides both
//! byte-level (`u8`) and character-level (`char`) implementations:
//!
//! - Byte-level types (`Phone`, `Context`, `RewriteRule`) for ASCII text
//! - Character-level types (`PhoneChar`, `ContextChar`, `RewriteRuleChar`) for Unicode
//!
//! Byte-level operations are ~5% faster and use ~4× less memory for edge labels,
//! but character-level operations correctly handle accented characters, CJK, emoji, etc.
//!
//! # See Also
//!
//! - Original specification: <https://zompist.com/spell.html>
//! - Verification proofs: `docs/verification/phonetic/`
//! - Architecture documentation: `docs/verification/ARCHITECTURE.md`

pub mod application;
pub mod llev;
pub mod matching;
pub mod named_classes;
pub mod nfa;
pub mod regex;
pub mod rules;
pub mod syllable;
pub mod types;
pub mod verified;

#[cfg(test)]
mod properties;

// Re-export main types (byte-level)
pub use application::{
    apply_rule_at, apply_rules_seq, apply_rules_seq_opt, apply_rules_seq_optimized,
    apply_rules_with_cycle_detection, can_apply_at, find_first_match_from,
    has_position_dependent_rules, NormalizationResult, MAX_EXPANSION_FACTOR,
};
pub use matching::{context_matches, pattern_matches_at, phone_eq};
pub use rules::{orthography_rules, phonetic_rules, test_rules, zompist_rules};
pub use types::{Context, Phone, RewriteRule};

// Re-export character-level types
pub use application::{
    apply_rule_at_char, apply_rules_seq_char, apply_rules_seq_opt_char,
    apply_rules_seq_optimized_char, apply_rules_with_cycle_detection_char,
    can_apply_at_char, find_first_match_from_char, has_position_dependent_rules_char,
    NormalizationResultChar,
};
pub use matching::{context_matches_char, pattern_matches_at_char, phone_eq_char};
pub use rules::{
    orthography_rules_char, phonetic_rules_char, test_rules_char, zompist_rules_char,
};
pub use types::{ContextChar, PhoneChar, RewriteRuleChar};

// Re-export verified rules integration
pub use verified::{
    rule_to_nfa, rule_to_nfa_char, rules_to_nfa, rules_to_nfa_char, zompist_nfa, zompist_nfa_char,
};

// Re-export syllable detection functions
pub use syllable::{
    is_before_doubled_consonant, is_final_syllable, is_initial_syllable, is_open_syllable,
    syllable_boundaries, syllable_count,
};

// Re-export LLev file format types for custom rule definitions
pub use llev::{
    // AST types
    ContextAST, Expression, FileMetadata, IncludeDirective, LLevFile, RewriteRuleAST,
    RuleDefinition, RuleMetadata, SymbolDef,
    // Error types
    LLevError, LLevErrorKind, LLevResult, Position,
    // Loader
    load_file, load_file_with_includes, Loader, LoaderConfig,
    // Parser
    parse_expression, parse_str, Parser,
    // Ruleset conversion
    RuleSet, RuleSetChar,
};

// Re-export compiled module for AOT compilation (requires serialization feature)
#[cfg(feature = "serialization")]
pub use llev::{
    // Byte-level serialization
    from_bytes, load, save, to_bytes,
    // Character-level serialization
    from_bytes_char, load_char, save_char, to_bytes_char,
};

// Re-export named character classes
pub use named_classes::{
    all_builtin_class_names, get_chars_only, get_digraphs_only, get_named_class, is_builtin_class,
    NamedClass, PhonePattern, NAMED_CLASSES,
};
