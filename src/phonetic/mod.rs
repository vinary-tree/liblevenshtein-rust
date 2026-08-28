//! Phonetic rewrite rules for approximate string matching.
//!
//! This module implements phonetic transformation rules based on Zompist's
//! English spelling rules. The legacy byte-level rewrite core has partial
//! Rocq coverage in `docs/verification/phonetic/`; the current runtime also
//! includes expanded rule sets, compound contexts, syllable conditions, and
//! Unicode-oriented variants that are validated by Rust tests.
//!
//! # Rule Application Functions
//!
//! | Function | Use Case |
//! |----------|----------|
//! | [`apply_rules_seq`](crate::phonetic::apply_rules_seq) | **Default** - use for all typical workloads (dictionary words, phrases) |
//! | [`apply_rules_seq_optimized`](crate::phonetic::apply_rules_seq_optimized) | **Opt-in** - use only for very long strings (100+ chars) with repetitive patterns |
//!
//! The position skipping optimization in `apply_rules_seq_optimized` provides up to
//! 26.6× speedup for synthetic repetitive strings, but causes 1-15% overhead for
//! typical English dictionary words. See `docs/verification/phonetic/POSITION_SKIPPING_BENCHMARK_RESULTS.md`
//! for detailed benchmark results.
//!
//! # Formal Verification
//!
//! The authoritative verification status is
//! `docs/verification/FORMAL_VERIFICATION_MANIFEST.tsv`. The phonetic proof
//! tree is currently marked partial: it proves theorem-shaped properties for a
//! legacy modeled subset, while Rust tests cover the expanded runtime API.
//!
//! The legacy proof model establishes these properties for its closed-world
//! rule set:
//!
//! 1. **Well-formedness** (Theorem 1, `docs/verification/phonetic/zompist_rules.v:285`)
//!    - Modeled rules satisfy structural constraints
//!    - Pattern and replacement sequences are valid
//!    - Context specifications are well-formed
//!
//! 2. **Bounded Expansion** (Theorem 2, `docs/verification/phonetic/zompist_rules.v:425`)
//!    - String length growth is bounded by a constant factor
//!    - Prevents unbounded growth in the modeled subset
//!
//! 3. **Non-Confluence** (Theorem 3, `docs/verification/phonetic/zompist_rules.v:491`)
//!    - Rule application order matters
//!    - Different orders can produce different results
//!    - Sequential application is the canonical approach
//!
//! 4. **Termination** (Theorem 4, `docs/verification/phonetic/zompist_rules.v:569`)
//!    - Sequential application is fuel-bounded
//!
//! 5. **Fixed-point stability** (Theorem 5, `docs/verification/phonetic/zompist_rules.v:615`)
//!    - Fixed points remain unchanged
//!    - Applying rules to the output produces the same output
//!    - Stable transformation semantics
//!
//! # Usage
//!
//! ```rust
//! use liblevenshtein::phonetic::{
//!     apply_rules_seq, orthography_rules, PhoneByte, MAX_EXPANSION_FACTOR,
//! };
//!
//! let input = vec![
//!     PhoneByte::Consonant(b'p'),
//!     PhoneByte::Consonant(b'h'),
//!     PhoneByte::Vowel(b'o'),
//!     PhoneByte::Consonant(b'n'),
//!     PhoneByte::Vowel(b'e'),
//! ];
//! let rules = orthography_rules();
//! let fuel = input.len() * rules.len() * MAX_EXPANSION_FACTOR;
//! let phonetic = apply_rules_seq(&rules, &input, fuel)
//!     .expect("phonetic rewriting returns its fixed point");
//! assert_eq!(
//!     phonetic,
//!     vec![
//!         PhoneByte::Consonant(b'f'),
//!         PhoneByte::Vowel(b'o'),
//!         PhoneByte::Consonant(b'n'),
//!         PhoneByte::Silent,
//!     ],
//! );
//! ```
//!
//! # Rule Sets
//!
//! Three built-in rule sets are provided:
//!
//! - [`orthography_rules()`](crate::phonetic::orthography_rules) - Simplifies English spelling (e.g., "gh" → "f")
//! - [`phonetic_rules()`](crate::phonetic::phonetic_rules) - Maps to phonetic representations
//! - [`test_rules()`](crate::phonetic::test_rules) - Example rules for testing and validation
//!
//! # Custom Rules with `.llev` Files
//!
//! Load custom phonetic rules from `.llev` files:
//!
//! ```rust
//! use liblevenshtein::phonetic::{parse_str, RuleSetChar};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
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
//! // Apply rules through the string-oriented convenience API.
//! let result = ruleset.apply("phone");
//! assert_eq!(result, "fone");
//! # Ok(())
//! # }
//! ```
//!
//! See the [`llev`](crate::phonetic::llev) module for complete documentation on the file format.
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
pub mod classifiers;
pub mod common;
pub mod expansion;
pub mod feature_distance;
pub mod features;
pub mod grep;
pub mod grep_online;
pub mod ipa_syllable;
pub mod language;
pub mod llev;
pub mod llre;
pub mod matching;
pub mod named_classes;
pub mod nfa;
pub mod online_scanner;
pub mod online_transducer;
pub mod regex;
pub mod rules;
pub mod syllable;
pub mod token_grep;
pub mod types;
pub mod verified;

#[cfg(test)]
mod properties;

// Re-export main types (byte-level)
pub use application::{
    apply_rule_at, apply_rules_seq, apply_rules_seq_optimized, apply_rules_with_cycle_detection,
    can_apply_at, find_first_match_from, has_position_dependent_rules, NormalizationResult,
    MAX_EXPANSION_FACTOR,
};
pub use matching::{context_matches, pattern_matches_at, phone_eq};
pub use rules::{orthography_rules, phonetic_rules, test_rules, zompist_rules};
pub use types::{Context, ContextByte, Phone, PhoneByte, RewriteRule, RewriteRuleByte};

// Re-export character-level types
pub use application::{
    apply_rule_at_char, apply_rules_seq_char, apply_rules_seq_optimized_char,
    apply_rules_with_cycle_detection_char, can_apply_at_char, find_first_match_from_char,
    has_position_dependent_rules_char, NormalizationResultChar,
};
pub use matching::{context_matches_char, pattern_matches_at_char, phone_eq_char};
pub use rules::{orthography_rules_char, phonetic_rules_char, test_rules_char, zompist_rules_char};
pub use types::{ContextChar, PhoneChar, RewriteRuleChar};

// Re-export verified rules integration
pub use verified::{
    rule_to_nfa, rule_to_nfa_char, rules_to_nfa, rules_to_nfa_char, zompist_nfa, zompist_nfa_char,
};

// Re-export syllable detection functions (orthographic, English-based)
pub use syllable::{
    evaluate_syllable_condition, evaluate_syllable_condition_ipa, evaluate_syllable_expr,
    evaluate_syllable_expr_ipa, is_before_doubled_consonant, is_final_syllable,
    is_initial_syllable, is_open_syllable, syllable_boundaries, syllable_count,
};

// Re-export IPA-based syllable functions (language-agnostic)
pub use ipa_syllable::{
    ipa_syllable_boundaries,
    ipa_syllable_count,
    // Position checks using IPA
    is_final_syllable as is_final_syllable_ipa,
    is_initial_syllable as is_initial_syllable_ipa,
    is_ipa_consonant,
    is_ipa_vowel,
    is_length_marker,
    is_open_syllable as is_open_syllable_ipa,
    is_stress_marker,
    is_syllable_boundary,
};

// Re-export phonetic feature types for (?f) flag support
pub use features::{
    are_similar, chars_with_any_feature, chars_with_features, expand_feature_based, get_features,
    get_similar_chars, get_voicing_pair, PhoneticFeature,
};

// Re-export articulatory feature distance computation
pub use feature_distance::{
    articulatory_distance, articulatory_edit_distance, feature_set_distance, is_free_substitution,
};

// Re-export LLev file format types for custom rule definitions
pub use llev::{
    // Loader
    load_file,
    load_file_with_includes,
    // Parser
    parse_expression,
    parse_str,
    // AST types
    ContextAST,
    Expression,
    FileMetadata,
    IncludeDirective,
    // Error types
    LLevError,
    LLevErrorKind,
    LLevFile,
    LLevResult,
    Loader,
    LoaderConfig,
    Parser,
    Position,
    RewriteRuleAST,
    RuleDefinition,
    RuleMetadata,
    // Ruleset conversion
    RuleSet,
    RuleSetChar,
    SymbolDef,
};

// Re-export compiled module for AOT compilation (requires serialization feature)
#[cfg(feature = "serialization")]
pub use llev::{
    // Byte-level serialization
    from_bytes,
    // Character-level serialization
    from_bytes_char,
    load,
    load_char,
    save,
    save_char,
    to_bytes,
    to_bytes_char,
};

// Re-export named character classes
pub use named_classes::{
    all_builtin_class_names, get_chars_only, get_digraphs_only, get_named_class, is_builtin_class,
    NamedClass, PhonePattern, NAMED_CLASSES,
};

// Re-export LLRE (LibLevenshtein Regex Expression) types
pub use llre::{
    // Compiler
    compile as compile_llre,
    compile_pattern,
    compile_pattern_with_flags,
    compile_with_options,
    is_match as llre_is_match,
    is_match_multiline as llre_is_match_multiline,
    // Loader
    load_file as load_llre_file,
    load_file_with_config as load_llre_file_with_config,
    // Parser
    parse_str as parse_llre_str,
    CompileOptions as LLreCompileOptions,
    CompiledNFA,
    // AST types
    Directive as LLreDirective,
    FileMetadata as LLreMetadata,
    ImportDirective as LLreImport,
    // Error types
    LLreError,
    LLreErrorKind,
    LLreFile,
    LLreFlags,
    LLreResult,
    Loader as LLreLoader,
    LoaderConfig as LLreLoaderConfig,
    Parser as LLreParser,
    Position as LLrePosition,
    ResolvedImport,
    SymbolTable,
};

// Re-export LLRE serialization functions (requires serialization feature)
#[cfg(feature = "serialization")]
pub use llre::{
    from_bytes as llre_from_bytes, load as load_compiled_llre, save as save_compiled_llre,
    to_bytes as llre_to_bytes, CompiledMetadata as LLreCompiledMetadata,
};

// Re-export grep types for on-the-fly phonetic pattern matching
pub use grep::{GrepError, GrepMatch, LineMatch, PhoneticGrep, WordBoundaryIterator};

// Re-export online grep for character-level streaming matching
pub use grep_online::{PhoneticGrepOnline, StreamingScanner};

// Re-export online transducer for streaming phonetic normalization
pub use online_transducer::OnlinePhoneticTransducerChar;

// Re-export online scanner for streaming document matching
pub use online_scanner::{OnlinePhoneticScannerChar, ScanMatch, ScannerStats};

// Re-export token-aware grep for per-token Levenshtein matching
pub use token_grep::{
    parse_query as parse_token_query, CompiledTokenQuery, DocumentMatch, Separator,
    StreamingTokenMatcher, TokenGrep, TokenMatch, TokenMatchDetail, TokenPattern, TokenQuery,
    TokenSpec,
};

// Re-export phonetic pattern expansion for reverse matching
pub use expansion::{expand_phonetic_alternatives_char, expand_with_costs};
