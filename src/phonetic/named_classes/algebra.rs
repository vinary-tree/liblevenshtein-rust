//! Set-algebra helpers over phonetic character classes.
//!
//! Provides the union/negation/intersection primitives used to combine named classes —
//! e.g. the universe of all phonetic characters (vowels ∪ consonants, ASCII and IPA) that
//! serves as the complement base for negating a character set. Underpins the intersection
//! semantics of compound class expressions.

// ============================================================================
// Feature Bundle Helpers (for intersection semantics)
// ============================================================================

use std::collections::HashSet;

use super::lookup::get_chars_only;

#[inline]
fn phonetic_universe_capacity(vowel_count: usize, consonant_count: usize) -> Option<usize> {
    vowel_count.checked_add(consonant_count)
}

/// Get all phonetic characters (union of all vowels and consonants).
///
/// This is used as the universe for computing negation of character sets.
/// Returns both ASCII and IPA characters.
pub fn get_all_phonetic_chars() -> Vec<char> {
    let vowels = get_chars_only("vowel").unwrap_or_default();
    let consonants = get_chars_only("consonant").unwrap_or_default();

    let mut chars: HashSet<char> = HashSet::with_capacity(
        phonetic_universe_capacity(vowels.len(), consonants.len()).unwrap_or(0),
    );
    chars.extend(vowels);
    chars.extend(consonants);
    chars.into_iter().collect()
}

/// Compute the intersection of multiple character sets.
///
/// Returns characters that appear in ALL of the provided sets.
/// An empty input returns an empty result.
///
/// # Example
///
/// ```
/// use liblevenshtein::phonetic::named_classes::{get_chars_only, intersect_char_sets};
///
/// let voiced = get_chars_only("voiced").unwrap();
/// let stop = get_chars_only("stop").unwrap();
/// let result = intersect_char_sets(&[voiced, stop]);
/// // result contains only voiced stops: b, d, g
/// assert!(result.contains(&'b'));
/// assert!(result.contains(&'d'));
/// assert!(result.contains(&'g'));
/// assert!(!result.contains(&'p')); // voiceless
/// ```
pub fn intersect_char_sets(sets: &[Vec<char>]) -> Vec<char> {
    if sets.is_empty() {
        return Vec::new();
    }

    let Some((smallest_index, smallest)) = sets.iter().enumerate().min_by_key(|(_, set)| set.len())
    else {
        return Vec::new();
    };
    if smallest.is_empty() {
        return Vec::new();
    }

    let mut result: HashSet<char> = HashSet::with_capacity(smallest.len());
    result.extend(smallest.iter().copied());

    for (index, set) in sets.iter().enumerate() {
        if index == smallest_index {
            continue;
        }
        let mut other: HashSet<char> = HashSet::with_capacity(set.len());
        other.extend(set.iter().copied());
        result.retain(|c| other.contains(c));
        if result.is_empty() {
            break;
        }
    }

    result.into_iter().collect()
}

/// Negate a character set (relative to all phonetic characters).
///
/// Returns all phonetic characters that are NOT in the provided set.
///
/// # Example
///
/// ```
/// use liblevenshtein::phonetic::named_classes::{get_chars_only, negate_char_set};
///
/// let nasal = get_chars_only("nasal").unwrap();
/// let not_nasal = negate_char_set(&nasal);
/// // not_nasal contains everything except m, n, ŋ
/// assert!(!not_nasal.contains(&'m'));
/// assert!(!not_nasal.contains(&'n'));
/// assert!(not_nasal.contains(&'p'));
/// assert!(not_nasal.contains(&'a'));
/// ```
pub fn negate_char_set(chars: &[char]) -> Vec<char> {
    let all = get_all_phonetic_chars();
    let mut excluded: HashSet<char> = HashSet::with_capacity(chars.len());
    excluded.extend(chars.iter().copied());

    let mut negated = Vec::with_capacity(all.len());
    negated.extend(all.into_iter().filter(|c| !excluded.contains(c)));
    negated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phonetic_universe_capacity_rejects_overflow() {
        assert_eq!(phonetic_universe_capacity(2, 3), Some(5));
        assert_eq!(phonetic_universe_capacity(usize::MAX, 1), None);
    }
}
