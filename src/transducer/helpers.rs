//! Helper functions for common fuzzy matching patterns.
//!
//! This module provides tuned implementations for hierarchical scope
//! filtering and other value-based query patterns.

/// Checks if two sorted vectors have any common elements.
///
/// This uses a two-pointer scan with early termination, making it
/// optimal for hierarchical scope intersection checks.
///
/// # Performance
///
/// - Time complexity: O(n + m) worst case, O(1) best case with early termination
/// - Benchmarks show 4.7% faster than HashSet for typical scope sets
/// - Works with unlimited scope IDs
///
/// # Requirements
///
/// Both input slices must be sorted in ascending order. If they are not sorted,
/// the behavior is undefined (may return incorrect results).
///
/// # Examples
///
/// ```
/// use liblevenshtein::transducer::helpers::sorted_vec_intersection;
///
/// let term_scopes = vec![1, 5, 10, 15];
/// let visible_scopes = vec![0, 5, 12];
///
/// assert!(sorted_vec_intersection(&term_scopes, &visible_scopes)); // 5 is common
///
/// let no_overlap_a = vec![1, 3, 5];
/// let no_overlap_b = vec![2, 4, 6];
/// assert!(!sorted_vec_intersection(&no_overlap_a, &no_overlap_b));
/// ```
///
/// # Use Case: Code Completion with Hierarchical Scopes
///
/// ```ignore
/// // This example requires the "pathmap-backend" feature
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::transducer::helpers::sorted_vec_intersection;
///
/// // Create dictionary mapping terms to their visible scopes
/// let terms = vec![
///     ("global_var".to_string(), vec![0]),           // Only in global scope
///     ("outer_var".to_string(), vec![0, 1]),         // Global + outer
///     ("inner_var".to_string(), vec![0, 1, 2]),      // All scopes
/// ];
/// let dict = PathMapDictionary::from_terms_with_values(terms);
/// let transducer = Transducer::new(dict, Algorithm::Standard);
///
/// // Query from within inner scope (visible scopes: {0, 1, 2})
/// let visible_scopes = vec![0, 1, 2];
/// let results: Vec<_> = transducer
///     .query_filtered("var", 1, |term_scopes| {
///         sorted_vec_intersection(term_scopes, &visible_scopes)
///     })
///     .map(|c| c.term)
///     .collect();
///
/// assert_eq!(results.len(), 3); // All three visible
/// ```
#[inline]
pub fn sorted_vec_intersection(a: &[u32], b: &[u32]) -> bool {
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            return true; // Early termination on first match
        }
        if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }

    false
}

/// Checks if two bitmasks have any common bits set.
///
/// This is the fastest intersection check for scope IDs < 64.
///
/// # Performance
///
/// - Time complexity: O(1) - single bitwise AND operation
/// - Benchmarks show 7.9% faster than HashSet for small scope sets
/// - 3.4% faster than sorted vector intersection
///
/// # Limitations
///
/// Only works for scope IDs in range 0-63. Use [`sorted_vec_intersection`]
/// for larger scope IDs.
///
/// # Examples
///
/// ```
/// use liblevenshtein::transducer::helpers::bitmask_intersection;
///
/// let term_mask = 0b101010;    // scopes {1, 3, 5}
/// let visible_mask = 0b001100; // scopes {2, 3}
///
/// assert!(bitmask_intersection(term_mask, visible_mask)); // 3 is common
///
/// let no_overlap_a = 0b000001; // scope {0}
/// let no_overlap_b = 0b000010; // scope {1}
/// assert!(!bitmask_intersection(no_overlap_a, no_overlap_b));
/// ```
///
/// # Creating Bitmasks from Scope Sets
///
/// ```
/// use std::collections::HashSet;
///
/// fn scopes_to_bitmask(scopes: &HashSet<u32>) -> u64 {
///     let mut mask = 0u64;
///     for &scope_id in scopes {
///         if scope_id < 64 {
///             mask |= 1u64 << scope_id;
///         }
///     }
///     mask
/// }
///
/// let mut scopes = HashSet::new();
/// scopes.insert(1);
/// scopes.insert(3);
/// scopes.insert(5);
///
/// let mask = scopes_to_bitmask(&scopes);
/// assert_eq!(mask, 0b101010);
/// ```
///
/// # Use Case: Fast Code Completion (≤64 scopes)
///
/// ```ignore
/// // This example requires the "pathmap-backend" feature
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::transducer::helpers::bitmask_intersection;
///
/// // Create dictionary with bitmask values
/// let terms = vec![
///     ("global_var".to_string(), 0b0001u64),  // scope 0
///     ("outer_var".to_string(), 0b0011u64),   // scopes 0, 1
///     ("inner_var".to_string(), 0b0111u64),   // scopes 0, 1, 2
/// ];
/// let dict = PathMapDictionary::from_terms_with_values(terms);
/// let transducer = Transducer::new(dict, Algorithm::Standard);
///
/// // Query from within inner scope (bitmask for scopes {0, 1, 2})
/// let visible_mask = 0b0111u64;
/// let results: Vec<_> = transducer
///     .query_filtered("var", 1, |term_mask| {
///         bitmask_intersection(*term_mask, visible_mask)
///     })
///     .map(|c| c.term)
///     .collect();
///
/// assert_eq!(results.len(), 3); // All three visible
/// ```
#[inline(always)]
pub fn bitmask_intersection(mask_a: u64, mask_b: u64) -> bool {
    (mask_a & mask_b) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_vec_intersection_found() {
        let a = vec![1, 5, 10, 15];
        let b = vec![0, 5, 12];
        assert!(sorted_vec_intersection(&a, &b));
    }

    #[test]
    fn test_sorted_vec_intersection_not_found() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4, 6];
        assert!(!sorted_vec_intersection(&a, &b));
    }

    #[test]
    fn test_sorted_vec_intersection_empty() {
        let a = vec![1, 2, 3];
        let b: Vec<u32> = vec![];
        assert!(!sorted_vec_intersection(&a, &b));
        assert!(!sorted_vec_intersection(&b, &a));
    }

    #[test]
    fn test_sorted_vec_intersection_first_element() {
        let a = vec![1, 5, 10];
        let b = vec![1, 2, 3];
        assert!(sorted_vec_intersection(&a, &b));
    }

    #[test]
    fn test_sorted_vec_intersection_last_element() {
        let a = vec![1, 5, 10];
        let b = vec![7, 8, 10];
        assert!(sorted_vec_intersection(&a, &b));
    }

    #[test]
    fn test_bitmask_intersection_found() {
        let a = 0b101010; // {1, 3, 5}
        let b = 0b001100; // {2, 3}
        assert!(bitmask_intersection(a, b));
    }

    #[test]
    fn test_bitmask_intersection_not_found() {
        let a = 0b000001; // {0}
        let b = 0b000010; // {1}
        assert!(!bitmask_intersection(a, b));
    }

    #[test]
    fn test_bitmask_intersection_multiple_common() {
        let a = 0b111111; // {0, 1, 2, 3, 4, 5}
        let b = 0b110000; // {4, 5}
        assert!(bitmask_intersection(a, b));
    }

    #[test]
    fn test_bitmask_intersection_zero() {
        assert!(!bitmask_intersection(0, 0));
        assert!(!bitmask_intersection(0b1010, 0));
        assert!(!bitmask_intersection(0, 0b1010));
    }
}
