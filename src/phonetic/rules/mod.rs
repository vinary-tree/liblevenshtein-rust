//! Concrete phonetic rewrite rule definitions.
//!
//! This module contains the actual rule definitions from the Zompist English
//! spelling-to-pronunciation system. The legacy Rocq file
//! `docs/verification/phonetic/zompist_rules.v` proves properties for a
//! 13-rule subset; the current Rust aggregate contains 62 rules.
//!
//! # Rule Sets
//!
//! - [`orthography_rules()`] - Exact orthographic transformations (45 rules, weight=0.0)
//! - [`vowel_digraph_rules()`] - Vowel digraph simplifications (12 rules, weight=0.1)
//! - [`phonetic_rules()`] - Phonetic approximations (3 rules, weight=0.15)
//! - [`test_rules()`] - Test rules for non-commutativity (2 rules)
//! - [`zompist_rules()`] - Complete combined rule set (62 rules)
//!
//! # Rule Categories
//!
//! The orthography rules are organized by priority (high to low):
//! 1. **Affrication patterns**: tion, sion, cious, tious endings
//! 2. **Multi-char patterns**: ough, aught, ought, tch, dge
//! 3. **GH rules**: gh before vowels
//! 4. **Digraph conversions**: ch, sh, ph, th
//! 5. **Initial clusters**: wr, wh, gn, kn, mn, pt, ps, tm
//! 6. **X pronunciation**: x→gz (compound context), x→ks
//! 7. **Contextual rules**: soft c/g before front vowels
//! 8. **Additional ortho**: ck, mb, bt, mn
//! 9. **Double consonants**: bb, cc, dd, ff, gg, ll, mm, nn, pp, rr, ss, tt, zz
//! 10. **Default rules**: silent e, silent gh
//!
//! # Rule Application Order
//!
//! Rules must be applied in the order defined in the rule set, as some rules
//! depend on transformations made by earlier rules (e.g., rule 21 "c → k" must
//! follow rule 20 `c → s before [ie]`).
//!
//! # Formal Specification
//!
//! The first legacy subset is represented in
//! `docs/verification/phonetic/zompist_rules.v`. Current aggregate rule-set
//! invariants, including count, unique IDs, and expansion limits, are enforced
//! by Rust tests.
//!
//! # Reference
//!
//! Original specification: <https://zompist.com/spell.html>

// Submodules - language-specific phonetic rules (require embedded-rules feature)
#[cfg(feature = "embedded-rules")]
pub mod arabic;
#[cfg(feature = "embedded-rules")]
pub mod armenian;
#[cfg(feature = "embedded-rules")]
pub mod basque;
#[cfg(feature = "embedded-rules")]
pub mod belarusian;
#[cfg(feature = "embedded-rules")]
pub mod bengali;
#[cfg(feature = "embedded-rules")]
pub mod bulgarian;
#[cfg(feature = "embedded-rules")]
pub mod catalan;
#[cfg(feature = "embedded-rules")]
pub mod chinese;
#[cfg(feature = "embedded-rules")]
pub mod croatian;
#[cfg(feature = "embedded-rules")]
pub mod czech;
#[cfg(feature = "embedded-rules")]
pub mod danish;
#[cfg(feature = "embedded-rules")]
pub mod dutch;
#[cfg(feature = "embedded-rules")]
pub mod english;
#[cfg(feature = "embedded-rules")]
pub mod finnish;
#[cfg(feature = "embedded-rules")]
pub mod french;
#[cfg(feature = "embedded-rules")]
pub mod georgian;
#[cfg(feature = "embedded-rules")]
pub mod german;
#[cfg(feature = "embedded-rules")]
pub mod greek;
#[cfg(feature = "embedded-rules")]
pub mod gujarati;
#[cfg(feature = "embedded-rules")]
pub mod hebrew;
#[cfg(feature = "embedded-rules")]
pub mod hindi;
#[cfg(feature = "embedded-rules")]
pub mod hungarian;
#[cfg(feature = "embedded-rules")]
pub mod icelandic;
#[cfg(feature = "embedded-rules")]
pub mod indonesian;
#[cfg(feature = "embedded-rules")]
pub mod irish;
#[cfg(feature = "embedded-rules")]
pub mod italian;
#[cfg(feature = "embedded-rules")]
pub mod japanese;
#[cfg(feature = "embedded-rules")]
pub mod korean;
#[cfg(feature = "embedded-rules")]
pub mod maltese;
#[cfg(feature = "embedded-rules")]
pub mod marathi;
#[cfg(feature = "embedded-rules")]
pub mod norwegian;
#[cfg(feature = "embedded-rules")]
pub mod persian;
#[cfg(feature = "embedded-rules")]
pub mod polish;
#[cfg(feature = "embedded-rules")]
pub mod portuguese;
#[cfg(feature = "embedded-rules")]
pub mod punjabi;
#[cfg(feature = "embedded-rules")]
pub mod romanian;
#[cfg(feature = "embedded-rules")]
pub mod russian;
#[cfg(feature = "embedded-rules")]
pub mod serbian;
#[cfg(feature = "embedded-rules")]
pub mod slovak;
#[cfg(feature = "embedded-rules")]
pub mod spanish;
#[cfg(feature = "embedded-rules")]
pub mod swedish;
#[cfg(feature = "embedded-rules")]
pub mod tagalog;
#[cfg(feature = "embedded-rules")]
pub mod tamil;
#[cfg(feature = "embedded-rules")]
pub mod telugu;
#[cfg(feature = "embedded-rules")]
pub mod thai;
#[cfg(feature = "embedded-rules")]
pub mod turkish;
#[cfg(feature = "embedded-rules")]
pub mod ukrainian;
#[cfg(feature = "embedded-rules")]
pub mod urdu;
#[cfg(feature = "embedded-rules")]
pub mod vietnamese;
#[cfg(feature = "embedded-rules")]
pub mod welsh;

// Zompist English rule submodules
mod zompist_byte;
mod zompist_char;

#[cfg(test)]
mod tests;

// Re-export the Zompist English aggregators so external callers continue to
// resolve `crate::phonetic::rules::{orthography_rules, ..., zompist_rules}` and
// `crate::phonetic::rules::{orthography_rules_char, ..., zompist_rules_char}`.
pub use zompist_byte::{
    orthography_rules, phonetic_rules, test_rules, vowel_digraph_rules, zompist_rules,
};
pub use zompist_char::{
    orthography_rules_char, phonetic_rules_char, test_rules_char, vowel_digraph_rules_char,
    zompist_rules_char,
};
