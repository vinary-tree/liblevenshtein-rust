//! Verified rules integration with NFA compilation.
//!
//! This module provides the bridge between phonetic rules and NFA-based pattern
//! matching.
//!
//! # Design
//!
//! Rather than Coq→JSON→Rust code generation, we compile the existing Rust
//! rules directly to NFA representation. The current verification boundary is:
//!
//! 1. Rust rules in `rules.rs` are the implementation
//! 2. Rocq proofs in `docs/verification/phonetic/zompist_rules.v` verify
//!    properties for the legacy modeled subset
//! 3. Rust tests cover full 62-rule NFA compilation behavior
//!
//! # Properties Verified in Rocq for the Legacy Subset
//!
//! - **Well-formedness**: Patterns are non-empty
//! - **Bounded expansion**: Replacement length is bounded
//! - **Termination**: Rule application terminates
//! - **Non-commutativity**: Some rule orderings matter (Theorem 3)
//!
//! # Usage
//!
//! ```ignore
//! use liblevenshtein::phonetic::verified::{zompist_nfa_char, rules_to_nfa_char};
//! use liblevenshtein::phonetic::rules::zompist_rules_char;
//!
//! // Get pre-compiled NFA for all Zompist rules
//! let nfa = zompist_nfa_char();
//!
//! // Or compile a custom rule set
//! let custom_rules = vec![...];
//! let custom_nfa = rules_to_nfa_char(&custom_rules);
//! ```

pub mod rules_to_nfa;

pub use rules_to_nfa::{
    rule_to_nfa, rule_to_nfa_char, rules_to_nfa, rules_to_nfa_char, zompist_nfa, zompist_nfa_char,
};
