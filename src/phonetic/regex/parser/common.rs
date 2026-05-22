//! Shared parser helpers and re-exports.
//!
//! Items here are used by both the char-level parser (`char.rs`) and the
//! byte-level parser (`byte.rs`), or are public re-exports that historically
//! lived in `parser.rs`.

use crate::phonetic::common::utils::is_user_symbol_name;

/// Maximum complexity for parsed patterns (prevents DoS).
pub const MAX_PATTERN_SIZE: usize = 10_000;

// Re-export SymbolTable from common for backward compatibility
pub use crate::phonetic::common::utils::SymbolTable;

/// Resolve a feature bundle to a character set.
///
/// Each term is (name, negated). The result is the intersection of all terms,
/// where negated terms are first complemented relative to all phonetic characters.
pub(super) fn resolve_feature_bundle_chars(
    terms: &[(String, bool)],
    symbols: Option<&SymbolTable>,
) -> Result<Vec<char>, String> {
    use crate::phonetic::named_classes::{get_chars_only, intersect_char_sets, negate_char_set};

    let mut char_sets: Vec<Vec<char>> = Vec::new();

    for (name, negated) in terms {
        // Try user symbol first, then built-in
        let chars = if let Some(symbol_table) = symbols {
            if let Some(symbol_chars) = symbol_table.get(name) {
                symbol_chars.clone()
            } else if let Some(builtin_chars) = get_chars_only(name) {
                builtin_chars
            } else {
                return Err(format!("unknown class or symbol '{}'", name));
            }
        } else if let Some(builtin_chars) = get_chars_only(name) {
            builtin_chars
        } else if is_user_symbol_name(name) {
            return Err(format!("undefined symbol '{}'", name));
        } else {
            return Err(format!("unknown named class '{}'", name));
        };

        let final_chars = if *negated {
            negate_char_set(&chars)
        } else {
            chars
        };

        char_sets.push(final_chars);
    }

    Ok(intersect_char_sets(&char_sets))
}
