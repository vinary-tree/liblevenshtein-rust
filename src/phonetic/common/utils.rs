//! Shared utility functions for phonetic parsing modules.

use std::collections::HashMap;

/// Compute complement of a character class using printable ASCII.
///
/// This is used for negated named classes like `[^[:vowel:]]`.
/// Returns all printable ASCII characters (0x20-0x7E) that are NOT in the input set.
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::phonetic::common::utils::negate_char_class;
///
/// let vowels = vec!['a', 'e', 'i', 'o', 'u'];
/// let non_vowels = negate_char_class(&vowels);
/// assert!(!non_vowels.contains(&'a'));
/// assert!(non_vowels.contains(&'b'));
/// ```
pub fn negate_char_class(chars: &[char]) -> Vec<char> {
    (0x20u8..=0x7Eu8)
        .map(|b| b as char)
        .filter(|c| !chars.contains(c))
        .collect()
}

/// Check if a name represents a user-defined symbol (all uppercase).
///
/// User symbols follow the convention of starting with an uppercase letter
/// and containing only uppercase letters or non-alphabetic characters.
///
/// # Examples
///
/// - `"VOWEL"` → true
/// - `"FRONT_VOWEL"` → true
/// - `"V1"` → true
/// - `"vowel"` → false (lowercase)
/// - `"Vowel"` → false (mixed case)
/// - `"_FOO"` → false (must start with a letter)
/// - `"123"` → false (must start with a letter)
pub fn is_user_symbol_name(name: &str) -> bool {
    let mut chars = name.chars();
    // Must start with an uppercase letter
    match chars.next() {
        Some(first) if first.is_uppercase() => {
            // Remaining characters: alphabetic must be uppercase, non-alphabetic are allowed
            chars.all(|c| c.is_uppercase() || !c.is_alphabetic())
        }
        _ => false,
    }
}

/// Alias for symbol table: maps symbol names to character vectors.
pub type SymbolTable = HashMap<String, Vec<char>>;

/// Resolve a feature bundle to a character set.
///
/// Each term is (name, negated). The result is the intersection of all terms,
/// where negated terms are first complemented relative to all phonetic characters.
///
/// # Arguments
///
/// * `terms` - List of (name, negated) pairs
/// * `symbols` - Optional user symbol table
/// * `get_builtin` - Function to resolve built-in named classes
/// * `negate_set` - Function to negate a character set
/// * `intersect_sets` - Function to intersect character sets
///
/// # Returns
///
/// Returns `Ok(chars)` on success, or `Err(message)` if a symbol/class is not found.
pub fn resolve_feature_bundle_chars<F, N, I>(
    terms: &[(String, bool)],
    symbols: Option<&SymbolTable>,
    get_builtin: F,
    negate_set: N,
    intersect_sets: I,
) -> Result<Vec<char>, String>
where
    F: Fn(&str) -> Option<Vec<char>>,
    N: Fn(&[char]) -> Vec<char>,
    I: Fn(&[Vec<char>]) -> Vec<char>,
{
    let mut char_sets: Vec<Vec<char>> = Vec::with_capacity(terms.len());

    for (name, negated) in terms {
        // Try user symbol first, then built-in
        let chars = if let Some(symbol_table) = symbols {
            if let Some(symbol_chars) = symbol_table.get(name) {
                symbol_chars.clone()
            } else if let Some(builtin_chars) = get_builtin(name) {
                builtin_chars
            } else {
                return Err(format!("unknown class or symbol '{}'", name));
            }
        } else if let Some(builtin_chars) = get_builtin(name) {
            builtin_chars
        } else if is_user_symbol_name(name) {
            return Err(format!("undefined symbol '{}'", name));
        } else {
            return Err(format!("unknown named class '{}'", name));
        };

        let final_chars = if *negated { negate_set(&chars) } else { chars };

        char_sets.push(final_chars);
    }

    Ok(intersect_sets(&char_sets))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negate_char_class() {
        let vowels = vec!['a', 'e', 'i', 'o', 'u'];
        let non_vowels = negate_char_class(&vowels);

        // Vowels should not be in the result
        assert!(!non_vowels.contains(&'a'));
        assert!(!non_vowels.contains(&'e'));
        assert!(!non_vowels.contains(&'i'));
        assert!(!non_vowels.contains(&'o'));
        assert!(!non_vowels.contains(&'u'));

        // Consonants should be in the result
        assert!(non_vowels.contains(&'b'));
        assert!(non_vowels.contains(&'c'));
        assert!(non_vowels.contains(&'z'));

        // Space and symbols should be in the result
        assert!(non_vowels.contains(&' '));
        assert!(non_vowels.contains(&'!'));
        assert!(non_vowels.contains(&'~'));
    }

    #[test]
    fn test_negate_char_class_empty() {
        let empty: Vec<char> = vec![];
        let result = negate_char_class(&empty);

        // Should contain all printable ASCII (95 characters: 0x20-0x7E)
        assert_eq!(result.len(), 95);
    }

    #[test]
    fn test_is_user_symbol_name_valid() {
        assert!(is_user_symbol_name("VOWEL"));
        assert!(is_user_symbol_name("FRONT_VOWEL"));
        assert!(is_user_symbol_name("V1"));
        assert!(is_user_symbol_name("A"));
        assert!(is_user_symbol_name("ABC123"));
    }

    #[test]
    fn test_is_user_symbol_name_invalid() {
        assert!(!is_user_symbol_name("vowel"));
        assert!(!is_user_symbol_name("Vowel"));
        assert!(!is_user_symbol_name("_FOO"));
        assert!(!is_user_symbol_name("123"));
        assert!(!is_user_symbol_name(""));
        assert!(!is_user_symbol_name("foo_BAR"));
    }

    #[test]
    fn test_resolve_feature_bundle_chars_builtin_only() {
        let terms = vec![("vowel".to_string(), false)];
        let result = resolve_feature_bundle_chars(
            &terms,
            None,
            |name| {
                if name == "vowel" {
                    Some(vec!['a', 'e', 'i', 'o', 'u'])
                } else {
                    None
                }
            },
            negate_char_class,
            |sets| {
                if sets.is_empty() {
                    vec![]
                } else {
                    sets[0].clone()
                }
            },
        );

        assert!(result.is_ok());
        let chars = result.expect("test fixture: resolve must be Ok");
        assert!(chars.contains(&'a'));
        assert!(chars.contains(&'e'));
    }

    #[test]
    fn test_resolve_feature_bundle_chars_negated() {
        let terms = vec![("vowel".to_string(), true)];
        let result = resolve_feature_bundle_chars(
            &terms,
            None,
            |name| {
                if name == "vowel" {
                    Some(vec!['a', 'e', 'i', 'o', 'u'])
                } else {
                    None
                }
            },
            negate_char_class,
            |sets| {
                if sets.is_empty() {
                    vec![]
                } else {
                    sets[0].clone()
                }
            },
        );

        assert!(result.is_ok());
        let chars = result.expect("test fixture: resolve must be Ok");
        // Negated - should NOT contain vowels
        assert!(!chars.contains(&'a'));
        assert!(!chars.contains(&'e'));
        // Should contain consonants
        assert!(chars.contains(&'b'));
    }

    #[test]
    fn test_resolve_feature_bundle_chars_with_symbols() {
        let mut symbols = SymbolTable::new();
        symbols.insert("MY_CLASS".to_string(), vec!['x', 'y', 'z']);

        let terms = vec![("MY_CLASS".to_string(), false)];
        let result = resolve_feature_bundle_chars(
            &terms,
            Some(&symbols),
            |_| None,
            negate_char_class,
            |sets| {
                if sets.is_empty() {
                    vec![]
                } else {
                    sets[0].clone()
                }
            },
        );

        assert!(result.is_ok());
        let chars = result.expect("test fixture: resolve must be Ok");
        assert!(chars.contains(&'x'));
        assert!(chars.contains(&'y'));
        assert!(chars.contains(&'z'));
    }

    #[test]
    fn test_resolve_feature_bundle_chars_unknown() {
        let terms = vec![("unknown".to_string(), false)];
        let result = resolve_feature_bundle_chars(
            &terms,
            None,
            |_| None,
            negate_char_class,
            |sets| {
                if sets.is_empty() {
                    vec![]
                } else {
                    sets[0].clone()
                }
            },
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown named class"));
    }

    #[test]
    fn test_resolve_feature_bundle_chars_undefined_symbol() {
        let terms = vec![("UNDEFINED_SYMBOL".to_string(), false)];
        let result = resolve_feature_bundle_chars(
            &terms,
            None,
            |_| None,
            negate_char_class,
            |sets| {
                if sets.is_empty() {
                    vec![]
                } else {
                    sets[0].clone()
                }
            },
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("undefined symbol"));
    }
}
