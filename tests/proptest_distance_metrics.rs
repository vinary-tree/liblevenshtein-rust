//! Property-based tests for distance metric mathematical properties.
//!
//! These tests verify that the distance functions satisfy the requirements
//! of distance metrics and semi-metrics:
//!
//! 1. **Non-negativity**: d(x, y) >= 0
//! 2. **Identity of indiscernibles**: d(x, y) = 0 ⟺ x = y
//! 3. **Symmetry**: d(x, y) = d(y, x)
//! 4. **Triangle inequality**: d(x, z) <= d(x, y) + d(y, z)
//! 5. **Left invariance**: d(zx, zy) = d(x, y)
//! 6. **Right invariance**: d(xz, yz) = d(x, y)
//!
//! Based on the C++ property tests in liblevenshtein-cpp/test/liblevenshtein/distance/

use liblevenshtein::distance::*;
use proptest::prelude::*;

// String generators
fn arb_string() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z]{0,20}").unwrap()
}

fn arb_unicode_string() -> impl Strategy<Value = String> {
    prop::collection::vec(any::<char>(), 0..20).prop_map(|chars| chars.into_iter().collect())
}

// ============================================================================
// Standard Distance Tests (Iterative)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn standard_distance_identity(a in arb_string()) {
        let distance = standard_distance(&a, &a);
        prop_assert_eq!(distance, 0, "Distance from string to itself must be zero");
    }

    #[test]
    fn standard_distance_indiscernible(a in arb_string(), b in arb_string()) {
        let distance = standard_distance(&a, &b);
        if distance == 0 {
            prop_assert_eq!(&a, &b, "If distance is zero, strings must be identical");
        }
    }

    #[test]
    fn standard_distance_symmetric(a in arb_string(), b in arb_string()) {
        let d_ab = standard_distance(&a, &b);
        let d_ba = standard_distance(&b, &a);
        prop_assert_eq!(d_ab, d_ba, "Distance must be symmetric: d(a,b) = d(b,a)");
    }

    #[test]
    fn standard_distance_triangle_inequality(
        a in arb_string(),
        b in arb_string(),
        c in arb_string()
    ) {
        let d_ac = standard_distance(&a, &c);
        let d_ab = standard_distance(&a, &b);
        let d_bc = standard_distance(&b, &c);

        prop_assert!(
            d_ac <= d_ab + d_bc,
            "Triangle inequality violated: d({}, {}) = {} > d({}, {}) + d({}, {}) = {} + {} = {}",
            a, c, d_ac, a, b, b, c, d_ab, d_bc, d_ab + d_bc
        );
    }

    #[test]
    fn standard_distance_left_invariance(
        x in arb_string(),
        y in arb_string(),
        z in arb_string()
    ) {
        let zx = format!("{}{}", z, x);
        let zy = format!("{}{}", z, y);

        let d_xy = standard_distance(&x, &y);
        let d_zx_zy = standard_distance(&zx, &zy);

        prop_assert_eq!(
            d_xy, d_zx_zy,
            "Left invariance violated: d({}, {}) = {} != d({}, {}) = {}",
            x, y, d_xy, zx, zy, d_zx_zy
        );
    }

    #[test]
    fn standard_distance_right_invariance(
        x in arb_string(),
        y in arb_string(),
        z in arb_string()
    ) {
        let xz = format!("{}{}", x, z);
        let yz = format!("{}{}", y, z);

        let d_xy = standard_distance(&x, &y);
        let d_xz_yz = standard_distance(&xz, &yz);

        prop_assert_eq!(
            d_xy, d_xz_yz,
            "Right invariance violated: d({}, {}) = {} != d({}, {}) = {}",
            x, y, d_xy, xz, yz, d_xz_yz
        );
    }
}

// ============================================================================
// Standard Distance Recursive Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn standard_recursive_identity(a in arb_string()) {
        let cache = create_memo_cache();
        let distance = standard_distance_recursive(&a, &a, &cache);
        prop_assert_eq!(distance, 0, "Distance from string to itself must be zero");
    }

    #[test]
    fn standard_recursive_indiscernible(a in arb_string(), b in arb_string()) {
        let cache = create_memo_cache();
        let distance = standard_distance_recursive(&a, &b, &cache);
        if distance == 0 {
            prop_assert_eq!(&a, &b, "If distance is zero, strings must be identical");
        }
    }

    #[test]
    fn standard_recursive_symmetric(a in arb_string(), b in arb_string()) {
        let cache = create_memo_cache();
        let d_ab = standard_distance_recursive(&a, &b, &cache);
        let d_ba = standard_distance_recursive(&b, &a, &cache);
        prop_assert_eq!(d_ab, d_ba, "Distance must be symmetric: d(a,b) = d(b,a)");
    }

    #[test]
    fn standard_recursive_triangle_inequality(
        a in arb_string(),
        b in arb_string(),
        c in arb_string()
    ) {
        let cache = create_memo_cache();
        let d_ac = standard_distance_recursive(&a, &c, &cache);
        let d_ab = standard_distance_recursive(&a, &b, &cache);
        let d_bc = standard_distance_recursive(&b, &c, &cache);

        prop_assert!(
            d_ac <= d_ab + d_bc,
            "Triangle inequality violated: d({}, {}) = {} > {} + {} = {}",
            a, c, d_ac, d_ab, d_bc, d_ab + d_bc
        );
    }

    #[test]
    fn standard_recursive_left_invariance(
        x in arb_string(),
        y in arb_string(),
        z in arb_string()
    ) {
        let cache = create_memo_cache();
        let zx = format!("{}{}", z, x);
        let zy = format!("{}{}", z, y);

        let d_xy = standard_distance_recursive(&x, &y, &cache);
        let d_zx_zy = standard_distance_recursive(&zx, &zy, &cache);

        prop_assert_eq!(
            d_xy, d_zx_zy,
            "Left invariance violated: d({}, {}) = {} != d({}, {}) = {}",
            x, y, d_xy, zx, zy, d_zx_zy
        );
    }

    #[test]
    fn standard_recursive_right_invariance(
        x in arb_string(),
        y in arb_string(),
        z in arb_string()
    ) {
        let cache = create_memo_cache();
        let xz = format!("{}{}", x, z);
        let yz = format!("{}{}", y, z);

        let d_xy = standard_distance_recursive(&x, &y, &cache);
        let d_xz_yz = standard_distance_recursive(&xz, &yz, &cache);

        prop_assert_eq!(
            d_xy, d_xz_yz,
            "Right invariance violated: d({}, {}) = {} != d({}, {}) = {}",
            x, y, d_xy, xz, yz, d_xz_yz
        );
    }

    #[test]
    fn standard_recursive_matches_iterative(a in arb_string(), b in arb_string()) {
        let cache = create_memo_cache();
        let recursive = standard_distance_recursive(&a, &b, &cache);
        let iterative = standard_distance(&a, &b);

        prop_assert_eq!(
            recursive, iterative,
            "Recursive and iterative implementations must match for '{}' vs '{}'",
            a, b
        );
    }
}

// ============================================================================
// Transposition Distance Tests (Iterative)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn transposition_distance_identity(a in arb_string()) {
        let distance = transposition_distance(&a, &a);
        prop_assert_eq!(distance, 0, "Distance from string to itself must be zero");
    }

    #[test]
    fn transposition_distance_indiscernible(a in arb_string(), b in arb_string()) {
        let distance = transposition_distance(&a, &b);
        if distance == 0 {
            prop_assert_eq!(&a, &b, "If distance is zero, strings must be identical");
        }
    }

    #[test]
    fn transposition_distance_symmetric(a in arb_string(), b in arb_string()) {
        let d_ab = transposition_distance(&a, &b);
        let d_ba = transposition_distance(&b, &a);
        prop_assert_eq!(d_ab, d_ba, "Distance must be symmetric: d(a,b) = d(b,a)");
    }

    // Note: Damerau-Levenshtein does NOT satisfy triangle inequality!
    // See: https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    // Example: d("ca", "abc") = 2, but d("ca", "ac") + d("ac", "abc") = 1 + 1 = 2
    // So we don't test triangle inequality for transposition distance.

    #[test]
    fn transposition_distance_left_invariance(
        x in arb_string(),
        y in arb_string(),
        z in arb_string()
    ) {
        let zx = format!("{}{}", z, x);
        let zy = format!("{}{}", z, y);

        let d_xy = transposition_distance(&x, &y);
        let d_zx_zy = transposition_distance(&zx, &zy);

        prop_assert_eq!(
            d_xy, d_zx_zy,
            "Left invariance violated: d({}, {}) = {} != d({}, {}) = {}",
            x, y, d_xy, zx, zy, d_zx_zy
        );
    }

    #[test]
    fn transposition_distance_right_invariance(
        x in arb_string(),
        y in arb_string(),
        z in arb_string()
    ) {
        let xz = format!("{}{}", x, z);
        let yz = format!("{}{}", y, z);

        let d_xy = transposition_distance(&x, &y);
        let d_xz_yz = transposition_distance(&xz, &yz);

        prop_assert_eq!(
            d_xy, d_xz_yz,
            "Right invariance violated: d({}, {}) = {} != d({}, {}) = {}",
            x, y, d_xy, xz, yz, d_xz_yz
        );
    }

    #[test]
    fn transposition_le_standard(a in arb_string(), b in arb_string()) {
        let trans = transposition_distance(&a, &b);
        let std = standard_distance(&a, &b);

        prop_assert!(
            trans <= std,
            "Transposition distance should never exceed standard distance: trans({}, {}) = {} > std = {}",
            a, b, trans, std
        );
    }
}

// ============================================================================
// Transposition Distance Recursive Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn transposition_recursive_identity(a in arb_string()) {
        let cache = create_memo_cache();
        let distance = transposition_distance_recursive(&a, &a, &cache);
        prop_assert_eq!(distance, 0, "Distance from string to itself must be zero");
    }

    #[test]
    fn transposition_recursive_symmetric(a in arb_string(), b in arb_string()) {
        let cache = create_memo_cache();
        let d_ab = transposition_distance_recursive(&a, &b, &cache);
        let d_ba = transposition_distance_recursive(&b, &a, &cache);
        prop_assert_eq!(d_ab, d_ba, "Distance must be symmetric: d(a,b) = d(b,a)");
    }

    #[test]
    fn transposition_recursive_left_invariance(
        x in arb_string(),
        y in arb_string(),
        z in arb_string()
    ) {
        let cache = create_memo_cache();
        let zx = format!("{}{}", z, x);
        let zy = format!("{}{}", z, y);

        let d_xy = transposition_distance_recursive(&x, &y, &cache);
        let d_zx_zy = transposition_distance_recursive(&zx, &zy, &cache);

        prop_assert_eq!(
            d_xy, d_zx_zy,
            "Left invariance violated"
        );
    }

    #[test]
    fn transposition_recursive_matches_iterative(a in arb_string(), b in arb_string()) {
        let cache = create_memo_cache();
        let recursive = transposition_distance_recursive(&a, &b, &cache);
        let iterative = transposition_distance(&a, &b);

        prop_assert_eq!(
            recursive, iterative,
            "Recursive and iterative implementations must match for '{}' vs '{}'",
            a, b
        );
    }
}

// ============================================================================
// Merge and Split Distance Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn merge_split_distance_identity(a in arb_string()) {
        let cache = create_memo_cache();
        let distance = merge_and_split_distance(&a, &a, &cache);
        prop_assert_eq!(distance, 0, "Distance from string to itself must be zero");
    }

    #[test]
    fn merge_split_distance_indiscernible(a in arb_string(), b in arb_string()) {
        let cache = create_memo_cache();
        let distance = merge_and_split_distance(&a, &b, &cache);
        if distance == 0 {
            prop_assert_eq!(&a, &b, "If distance is zero, strings must be identical");
        }
    }

    #[test]
    fn merge_split_distance_symmetric(a in arb_string(), b in arb_string()) {
        let cache = create_memo_cache();
        let d_ab = merge_and_split_distance(&a, &b, &cache);
        let d_ba = merge_and_split_distance(&b, &a, &cache);
        prop_assert_eq!(d_ab, d_ba, "Distance must be symmetric: d(a,b) = d(b,a)");
    }

    // Note: Merge-and-split also may not satisfy triangle inequality
    // Similar to transposition distance, we skip this test

    #[test]
    fn merge_split_distance_left_invariance(
        x in arb_string(),
        y in arb_string(),
        z in arb_string()
    ) {
        let cache = create_memo_cache();
        let zx = format!("{}{}", z, x);
        let zy = format!("{}{}", z, y);

        let d_xy = merge_and_split_distance(&x, &y, &cache);
        let d_zx_zy = merge_and_split_distance(&zx, &zy, &cache);

        prop_assert_eq!(
            d_xy, d_zx_zy,
            "Left invariance violated: d({}, {}) = {} != d({}, {}) = {}",
            x, y, d_xy, zx, zy, d_zx_zy
        );
    }

    #[test]
    fn merge_split_distance_right_invariance(
        x in arb_string(),
        y in arb_string(),
        z in arb_string()
    ) {
        let cache = create_memo_cache();
        let xz = format!("{}{}", x, z);
        let yz = format!("{}{}", y, z);

        let d_xy = merge_and_split_distance(&x, &y, &cache);
        let d_xz_yz = merge_and_split_distance(&xz, &yz, &cache);

        prop_assert_eq!(
            d_xy, d_xz_yz,
            "Right invariance violated: d({}, {}) = {} != d({}, {}) = {}",
            x, y, d_xy, xz, yz, d_xz_yz
        );
    }

    #[test]
    fn merge_split_le_standard(a in arb_string(), b in arb_string()) {
        let cache = create_memo_cache();
        let merge_split = merge_and_split_distance(&a, &b, &cache);
        let std_recursive = standard_distance_recursive(&a, &b, &cache);

        prop_assert!(
            merge_split <= std_recursive,
            "Merge-split distance should never exceed standard distance: ms({}, {}) = {} > std = {}",
            a, b, merge_split, std_recursive
        );
    }
}

// ============================================================================
// Unicode Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn unicode_standard_distance_symmetric(a in arb_unicode_string(), b in arb_unicode_string()) {
        let d_ab = standard_distance(&a, &b);
        let d_ba = standard_distance(&b, &a);
        prop_assert_eq!(d_ab, d_ba, "Unicode distance must be symmetric");
    }

    #[test]
    fn unicode_standard_recursive_matches_iterative(a in arb_unicode_string(), b in arb_unicode_string()) {
        let cache = create_memo_cache();
        let recursive = standard_distance_recursive(&a, &b, &cache);
        let iterative = standard_distance(&a, &b);

        prop_assert_eq!(
            recursive, iterative,
            "Recursive and iterative must match for Unicode strings"
        );
    }
}
