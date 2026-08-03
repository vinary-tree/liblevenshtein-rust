//! Hamming distance over equal-length unit sequences.
//!
//! Hamming distance counts mismatched positions and is undefined when the two
//! sequences have different lengths. It is not ordinary Levenshtein distance
//! with an after-the-fact length check: insertion and deletion are absent from
//! its operation set.
//!
//! ```rust
//! use liblevenshtein::distance::{hamming_distance, standard_distance};
//!
//! assert_eq!(standard_distance("abc", "bca"), 2);
//! assert_eq!(hamming_distance("abc", "bca"), Some(3));
//! assert_eq!(hamming_distance("abc", "ab"), None);
//! ```

use libdictenstein::CharUnit;

/// Count unequal positions in two equal-length native-unit sequences.
///
/// Returns `None` when the lengths differ.
pub fn hamming_distance_units<U: CharUnit>(left: &[U], right: &[U]) -> Option<usize> {
    (left.len() == right.len()).then(|| {
        left.iter()
            .zip(right)
            .filter(|(left, right)| left != right)
            .count()
    })
}

/// Count unequal Unicode-scalar positions in two equal-length strings.
///
/// Returns `None` when the strings contain different numbers of Unicode scalar
/// values. Canonically equivalent strings are not normalized implicitly.
pub fn hamming_distance(left: &str, right: &str) -> Option<usize> {
    let mut left = left.chars();
    let mut right = right.chars();
    let mut mismatches = 0usize;
    loop {
        match (left.next(), right.next()) {
            (Some(left), Some(right)) => {
                mismatches = mismatches.saturating_add(usize::from(left != right));
            }
            (None, None) => return Some(mismatches),
            _ => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn examples_and_boundaries() {
        assert_eq!(hamming_distance("", ""), Some(0));
        assert_eq!(hamming_distance("abc", "abc"), Some(0));
        assert_eq!(hamming_distance("abc", "bca"), Some(3));
        assert_eq!(hamming_distance("abc", "ab"), None);
        assert_eq!(hamming_distance("é", "e"), Some(1));
        assert_eq!(hamming_distance("é", "e\u{301}"), None);
        assert_eq!(hamming_distance_units(&[1_u64, 2, 3], &[1, 4, 3]), Some(1));
    }
}
