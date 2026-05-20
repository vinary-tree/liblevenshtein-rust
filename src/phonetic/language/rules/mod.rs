//! Per-language phonetic rule aggregator functions, grouped by language family.
//!
//! Each `<lang>_rules()` function builds a `Vec<RewriteRuleChar>` for one
//! supported language by pulling from `crate::phonetic::rules::<lang>`. The
//! aggregators are independent (no shared state) and are dispatched from
//! [`super::dispatch::rules_for_language`].

pub mod celtic;
pub mod cjk;
pub mod germanic;
pub mod indic;
pub mod other;
pub mod romance;
pub mod semitic;
pub mod slavic;
pub mod southeast_asian;
