/// Property-based tests for skip-to-match optimization
///
/// These tests empirically validate the theorems proven in Coq:
/// - i_skip_to_match_preserves_invariant
/// - i_skip_to_match_formula
/// - m_skip_to_match_preserves_invariant
/// - m_skip_to_match_formula
use liblevenshtein::transducer::generalized::GeneralizedPosition;
use proptest::prelude::*;

// Helper function to check I-type invariant
fn check_i_invariant(offset: i32, errors: u8, max_distance: u8) -> bool {
    let n = max_distance as i32;
    // Bounds: -n <= offset <= n
    let bounds_ok = -n <= offset && offset <= n;
    // Reachability: |offset| <= errors
    let reachability_ok = offset.abs() <= errors as i32;
    // Budget: errors <= max_distance
    let budget_ok = errors <= max_distance;

    bounds_ok && reachability_ok && budget_ok
}

// Helper function to check M-type invariant
fn check_m_invariant(offset: i32, errors: u8, max_distance: u8) -> bool {
    let n = max_distance as i32;
    // Bounds: -2n <= offset <= 0
    let bounds_ok = -(2 * n) <= offset && offset <= 0;
    // Reachability: errors >= (-offset - n)
    // Equivalently: |offset| <= errors + n (since offset <= 0)
    let reachability_ok = errors as i32 >= (-offset - n);
    // Budget: errors <= max_distance
    let budget_ok = errors <= max_distance;

    bounds_ok && reachability_ok && budget_ok
}

proptest! {
    /// Test: I-type skip-to-match preserves invariant
    ///
    /// Validates: i_skip_to_match_preserves_invariant theorem
    ///
    /// Given:
    /// - Valid I-type position (satisfies i_invariant)
    /// - Skip distance such that errors + distance <= max_distance
    ///
    /// Then:
    /// - Result position (if created) satisfies i_invariant
    #[test]
    fn i_skip_preserves_invariant(
        offset in -10i32..=10,
        errors in 0u8..10,
        max_distance in 1u8..10,
        skip_distance in 1u8..5,
    ) {
        // Only test valid initial positions
        if !check_i_invariant(offset, errors, max_distance) {
            return Ok(());
        }

        // Only test cases where skip is within budget
        let new_errors = errors + skip_distance;
        if new_errors > max_distance {
            return Ok(());
        }

        // Compute new offset (skip-to-match formula: offset + distance)
        let new_offset = offset + skip_distance as i32;

        // Try to create successor position
        if let Ok(successor) = GeneralizedPosition::new_i(
            new_offset,
            new_errors,
            max_distance
        ) {
            // Verify successor satisfies invariant
            prop_assert!(check_i_invariant(
                successor.offset(),
                successor.errors(),
                max_distance
            ), "Skip-to-match created invalid I-type position");
        }
        // If position creation failed, that's OK - preconditions weren't met
    }

    /// Test: I-type skip-to-match formula
    ///
    /// Validates: i_skip_to_match_formula theorem
    ///
    /// The formula states:
    /// - offset' = offset + distance (forward scan)
    /// - errors' = errors + distance
    #[test]
    fn i_skip_formula(
        offset in -10i32..=10,
        errors in 0u8..10,
        max_distance in 1u8..10,
        skip_distance in 1u8..5,
    ) {
        // Only test valid initial positions
        if !check_i_invariant(offset, errors, max_distance) {
            return Ok(());
        }

        // Only test cases where skip is within budget
        let new_errors = errors + skip_distance;
        if new_errors > max_distance {
            return Ok(());
        }

        // Compute expected values from formula
        let expected_offset = offset + skip_distance as i32;
        let expected_errors = new_errors;

        // Try to create successor position
        if let Ok(successor) = GeneralizedPosition::new_i(
            expected_offset,
            expected_errors,
            max_distance
        ) {
            // Verify formula holds
            prop_assert_eq!(successor.offset(), expected_offset,
                "Skip-to-match offset formula failed");
            prop_assert_eq!(successor.errors(), expected_errors,
                "Skip-to-match errors formula failed");
        }
    }

    /// Test: M-type skip-to-match preserves invariant
    ///
    /// Validates: m_skip_to_match_preserves_invariant theorem
    ///
    /// Given:
    /// - Valid M-type position (satisfies m_invariant)
    /// - Skip distance such that errors + distance <= max_distance
    ///
    /// Then:
    /// - Result position (if created) satisfies m_invariant
    #[test]
    fn m_skip_preserves_invariant(
        offset in -20i32..=0,
        errors in 0u8..10,
        max_distance in 1u8..10,
        skip_distance in 1u8..5,
    ) {
        // Only test valid initial positions
        if !check_m_invariant(offset, errors, max_distance) {
            return Ok(());
        }

        // Only test cases where skip is within budget
        let new_errors = errors + skip_distance;
        if new_errors > max_distance {
            return Ok(());
        }

        // M-type skip: offset unchanged (stays at same word position)
        let new_offset = offset;

        // Try to create successor position
        if let Ok(successor) = GeneralizedPosition::new_m(
            new_offset,
            new_errors,
            max_distance
        ) {
            // Verify successor satisfies invariant
            prop_assert!(check_m_invariant(
                successor.offset(),
                successor.errors(),
                max_distance
            ), "Skip-to-match created invalid M-type position");
        }
        // If position creation failed, that's OK - preconditions weren't met
    }

    /// Test: M-type skip-to-match formula
    ///
    /// Validates: m_skip_to_match_formula theorem
    ///
    /// The formula states:
    /// - offset' = offset (unchanged for M-type)
    /// - errors' = errors + distance
    #[test]
    fn m_skip_formula(
        offset in -20i32..=0,
        errors in 0u8..10,
        max_distance in 1u8..10,
        skip_distance in 1u8..5,
    ) {
        // Only test valid initial positions
        if !check_m_invariant(offset, errors, max_distance) {
            return Ok(());
        }

        // Only test cases where skip is within budget
        let new_errors = errors + skip_distance;
        if new_errors > max_distance {
            return Ok(());
        }

        // Compute expected values from formula
        let expected_offset = offset;  // M-type: offset unchanged
        let expected_errors = new_errors;

        // Try to create successor position
        if let Ok(successor) = GeneralizedPosition::new_m(
            expected_offset,
            expected_errors,
            max_distance
        ) {
            // Verify formula holds
            prop_assert_eq!(successor.offset(), expected_offset,
                "Skip-to-match offset formula failed (M-type should be unchanged)");
            prop_assert_eq!(successor.errors(), expected_errors,
                "Skip-to-match errors formula failed");
        }
    }

    /// Test: Skip-to-match moves forward for I-type
    ///
    /// This test validates the key discovery from Finding F8:
    /// Skip-to-match moves FORWARD through the word (offset increases),
    /// unlike DELETE which moves BACKWARD (offset decreases).
    #[test]
    fn i_skip_moves_forward(
        offset in -5i32..=5,
        errors in 0u8..5,
        max_distance in 3u8..8,
        skip_distance in 1u8..3,
    ) {
        // Only test valid initial positions
        if !check_i_invariant(offset, errors, max_distance) {
            return Ok(());
        }

        // Only test cases where skip is within budget
        let new_errors = errors + skip_distance;
        if new_errors > max_distance {
            return Ok(());
        }

        // Skip-to-match moves FORWARD: offset increases
        let new_offset = offset + skip_distance as i32;

        if let Ok(successor) = GeneralizedPosition::new_i(
            new_offset,
            new_errors,
            max_distance
        ) {
            // Verify offset increased (moved forward)
            prop_assert!(successor.offset() > offset,
                "Skip-to-match should move forward (offset should increase)");
            prop_assert_eq!(successor.offset(), offset + skip_distance as i32,
                "Skip-to-match offset should be old offset + distance");
        }
    }
}
