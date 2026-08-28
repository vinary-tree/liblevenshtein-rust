//! NFA-based phonetic regular expression support.
//!
//! This module provides NFA (Nondeterministic Finite Automaton) infrastructure
//! for phonetic pattern matching. It implements:
//!
//! - Core NFA data structures ([`NFA`], [`NFAChar`])
//! - Thompson's Construction for building NFAs ([`ThompsonBuilder`], [`ThompsonBuilderChar`])
//! - Epsilon closure and NFA simulation algorithms
//! - Union, concatenation, and Kleene star operations
//!
//! # Design
//!
//! Following the existing codebase pattern, this module provides both
//! byte-level (`u8`) and character-level (`char`) implementations:
//!
//! - **Byte-level** (e.g., [`NFA`], [`ThompsonBuilder`]): Optimized for ASCII text,
//!   ~5% faster matching, ~4× less memory per edge label
//! - **Character-level** (e.g., [`NFAChar`], [`ThompsonBuilderChar`]): Full Unicode support
//!   for accented characters, CJK, emoji, etc.
//!
//! # Examples
//!
//! ## Basic NFA Construction
//!
//! ```rust
//! use liblevenshtein::phonetic::nfa::{ThompsonBuilderChar, NFAChar};
//!
//! let builder = ThompsonBuilderChar::new();
//!
//! // Build NFA for pattern: (ph|f)one
//! let ph = builder.literal("ph");
//! let f = builder.single_char('f');
//! let ph_or_f = builder.alternation(ph, f);
//! let one = builder.literal("one");
//! let pattern = builder.concatenate(ph_or_f, one);
//!
//! // Test acceptance
//! assert!(pattern.accepts("phone"));
//! assert!(pattern.accepts("fone"));
//! assert!(!pattern.accepts("bone"));
//! ```
//!
//! ## Character Classes
//!
//! ```rust
//! use liblevenshtein::phonetic::nfa::{ThompsonBuilderChar, CharClassChar};
//!
//! let builder = ThompsonBuilderChar::new();
//!
//! // Pattern: [aeiou]+ (one or more vowels)
//! let vowels = CharClassChar::from_chars(&['a', 'e', 'i', 'o', 'u']);
//! let pattern = builder.kleene_plus(builder.char_class(vowels));
//!
//! assert!(pattern.accepts("a"));
//! assert!(pattern.accepts("aeiou"));
//! assert!(!pattern.accepts(""));
//! assert!(!pattern.accepts("xyz"));
//! ```
//!
//! # Formal Specification
//!
//! See `docs/wfst/nfa_phonetic_regex.md` for the complete formal specification
//! of NFA construction and operations.

pub mod byte_nfa;
pub mod char_nfa;
pub mod compiler;
pub mod context;
pub mod incremental;
mod language_impl;
pub mod lazy_dfa;
pub mod memoized;
pub mod optimizer;
pub mod product;
pub mod state_set;
pub mod thompson;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export main types (character-level)
pub use char_nfa::NFAChar;
pub use compiler::{
    compile, compile_rewrite, compile_with_flags, estimate_thompson_states, CompileResultChar,
    CompiledRewriteChar, NFACompilerChar,
};
pub use context::{
    BoundaryKind, ContextMatcherChar, ContextPatternChar, ContextualRewriteRuleChar,
};
pub use incremental::{IncrementalMatcherChar, IncrementalProductMatcherChar, MatcherSnapshotChar};
pub use lazy_dfa::{CacheStats, DFAStateChar, LazyDFAChar};
pub use memoized::{MemoizedLazyDFAChar, MemoizedMatcherChar, MemoizedStats};
pub use product::{ProductAutomatonChar, ProductStateChar};
pub use thompson::ThompsonBuilderChar;
pub use types::{CharClassChar, NFAState, StateId, TransitionChar, TransitionLabelChar};

// Re-export byte-level types
pub use byte_nfa::NFA;
pub use compiler::{compile_bytes, compile_rewrite_bytes, CompiledRewrite, NFACompilerByte};
pub use context::{ContextMatcher, ContextPattern, ContextualRewriteRule};
pub use incremental::{IncrementalMatcher, MatcherSnapshot};
pub use lazy_dfa::{DFAState, LazyDFA};
pub use memoized::{MemoizedLazyDFA, MemoizedMatcher};
pub use product::{ProductAutomaton, ProductState};
pub use thompson::ThompsonBuilder;
pub use types::{CharClass, Transition, TransitionLabel};

// Re-export state set
pub use state_set::StateSet;

// Re-export optimizer types
pub use optimizer::{NfaOptimizer, NfaOptimizerChar, OptimizationConfig, OptimizationStats};
