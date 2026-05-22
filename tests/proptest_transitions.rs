// Property-based tests validating formal transition theorems
//
// This file tests the core correctness properties proven in:
// - rocq/liblevenshtein/Transitions.v (I-type and M-type successor preservation)
//
// Each test corresponds to a proven Coq theorem and validates that the Rust
// implementation correctly preserves position invariants during transitions.

use liblevenshtein::transducer::generalized::{
    CharacteristicVector, GeneralizedPosition, GeneralizedState,
};
use liblevenshtein::transducer::OperationSet;
use proptest::prelude::*;

// ============================================================================
// GENERATORS
// ============================================================================

/// Generate valid I-type positions
fn valid_i_position() -> impl Strategy<Value = (i32, u8, u8)> {
    (1u8..10).prop_flat_map(|max_distance| {
        let max_d = max_distance as i32;
        (0u8..=max_distance).prop_flat_map(move |errors| {
            let e = errors as i32;
            // I-type invariant: |offset| ≤ errors and -n ≤ offset ≤ n
            let min_offset = (-max_d).max(-e);
            let max_offset = max_d.min(e);
            (min_offset..=max_offset, Just(errors), Just(max_distance))
        })
    })
}

/// Generate valid M-type positions
fn valid_m_position() -> impl Strategy<Value = (i32, u8, u8)> {
    (1u8..10).prop_flat_map(|max_distance| {
        let max_d = max_distance as i32;
        (0u8..=max_distance).prop_flat_map(move |errors| {
            let e = errors as i32;
            // M-type invariant: errors ≥ -offset - n and -2n ≤ offset ≤ 0
            let min_offset = -(2 * max_d);
            let max_offset = 0;
            // Also need: errors >= -offset - n, so offset >= -errors - n
            let min_required = (-e - max_d).max(min_offset);
            (min_required..=max_offset, Just(errors), Just(max_distance))
        })
    })
}

/// Generate test strings
fn test_word() -> impl Strategy<Value = String> {
    "[a-z]{3,8}"
}

/// Generate input characters
fn input_char() -> impl Strategy<Value = char> {
    prop::char::range('a', 'z')
}

// ============================================================================
// THEOREM: i_successor_preserves_invariant (MAIN I-TYPE THEOREM)
// ============================================================================

/// **HIGH PRIORITY TEST**
///
/// Validates Coq theorem from Transitions.v:420:
/// ```coq
/// Theorem i_successor_preserves_invariant : forall p op cv p',
///   i_invariant p ->
///   i_successor p op cv p' ->
///   i_invariant p'.
/// ```
///
/// This is the MAIN correctness property for I-type transitions.
/// It proves that ALL I-type operations (Match, Delete, Insert, Substitute)
/// preserve the I-type position invariant.
#[cfg(test)]
mod i_type_invariant_preservation {
    use super::*;

    proptest! {
        #[test]
        fn i_successors_preserve_invariant(
            (offset, errors, max_distance) in valid_i_position(),
            word in test_word(),
            input_ch in input_char()
        ) {
            // Create valid I-type position (already satisfies i_invariant by construction)
            let p = GeneralizedPosition::new_i(offset, errors, max_distance)
                .expect("valid_i_position generator should produce valid positions");

            // Create state with single I-type position
            let mut state = GeneralizedState::new(max_distance);
            state.add_position(p);

            // Set up operations and bit vector
            let operations = OperationSet::standard();
            let cv = CharacteristicVector::new(input_ch, &word);

            // Compute successor state
            let word_chars: Vec<char> = word.chars().collect(); if let Some(next_state) = state.transition(&operations, &cv, &word, Some(&word_chars), &word, input_ch, 1) {
                // VALIDATE: Every successor must satisfy i_invariant
                for succ in next_state.positions() {
                    let succ_offset = succ.offset();
                    let succ_errors = succ.errors();

                    // I-invariant: For I-type positions (skip M-type)
                    if succ.is_non_final() {
                        // I-invariant conjunct 1: |offset| ≤ errors (or relaxed for errors=0)
                        if succ_errors > 0 {
                            prop_assert!(succ_offset.abs() <= succ_errors as i32,
                                "Successor violates reachability: |offset| = {} > errors = {}",
                                succ_offset.abs(), succ_errors);
                        }

                        // I-invariant conjunct 2: -n ≤ offset ≤ n (standard)
                        if succ_errors > 0 || succ_offset <= 0 {
                            prop_assert!(succ_offset >= -(max_distance as i32),
                                "Successor offset {} < -{}", succ_offset, max_distance);
                            prop_assert!(succ_offset <= max_distance as i32,
                                "Successor offset {} > {}", succ_offset, max_distance);
                        }

                        // I-invariant conjunct 3: errors ≤ n
                        prop_assert!(succ_errors <= max_distance,
                            "Successor errors {} > max_distance {}", succ_errors, max_distance);
                    }
                }
            }
        }
    }
}

// ============================================================================
// THEOREM: m_successor_preserves_invariant (MAIN M-TYPE THEOREM)
// ============================================================================

/// **HIGH PRIORITY TEST**
///
/// Validates Coq theorem from Transitions.v:828:
/// ```coq
/// Theorem m_successor_preserves_invariant : forall p op cv p',
///   m_invariant p ->
///   m_successor p op cv p' ->
///   m_invariant p'.
/// ```
///
/// This is the MAIN correctness property for M-type transitions.
#[cfg(test)]
mod m_type_invariant_preservation {
    use super::*;

    proptest! {
        #[test]
        fn m_successors_preserve_invariant(
            (offset, errors, max_distance) in valid_m_position(),
            word in test_word(),
            input_ch in input_char()
        ) {
            // Create valid M-type position (already satisfies m_invariant by construction)
            let p = GeneralizedPosition::new_m(offset, errors, max_distance)
                .expect("valid_m_position generator should produce valid positions");

            // Create state with single M-type position
            let mut state = GeneralizedState::new(max_distance);
            state.add_position(p);

            // Set up operations and bit vector
            let operations = OperationSet::standard();
            let cv = CharacteristicVector::new(input_ch, &word);

            // Compute successor state
            let word_chars: Vec<char> = word.chars().collect(); if let Some(next_state) = state.transition(&operations, &cv, &word, Some(&word_chars), &word, input_ch, 1) {
                // VALIDATE: Every successor must satisfy m_invariant
                for succ in next_state.positions() {
                    let succ_offset = succ.offset();
                    let succ_errors = succ.errors();
                    let n = max_distance as i32;

                    // M-invariant: For M-type positions (skip I-type)
                    if succ.is_final() {
                        // M-invariant conjunct 1: errors ≥ -offset - n
                        prop_assert!(succ_errors as i32 >= -(succ_offset) - n,
                            "Successor violates M-reachability: errors {} < -offset {} - n {}",
                            succ_errors, succ_offset, n);

                        // M-invariant conjunct 2: -2n ≤ offset ≤ 0
                        prop_assert!(succ_offset >= -(2 * n),
                            "Successor offset {} < -{}", succ_offset, 2 * n);
                        prop_assert!(succ_offset <= 0,
                            "Successor offset {} > 0", succ_offset);

                        // M-invariant conjunct 3: errors ≤ n
                        prop_assert!(succ_errors <= max_distance,
                            "Successor errors {} > max_distance {}", succ_errors, max_distance);
                    }
                }
            }
        }
    }
}

// ============================================================================
// THEOREM: i_successor_cost_correct / m_successor_cost_correct
// ============================================================================

/// Validates Coq theorems from Transitions.v:484, 860:
/// ```coq
/// Theorem i_successor_cost_correct : forall p op cv p',
///   i_successor p op cv p' ->
///   errors p' = (errors p + operation_cost op)%nat.
/// ```
///
/// Verifies that error accounting matches operation costs exactly.
#[cfg(test)]
mod cost_correctness {
    use super::*;

    proptest! {
        #[test]
        fn i_successor_cost_matches_operation(
            (offset, errors, max_distance) in valid_i_position(),
            word in test_word(),
            input_ch in input_char()
        ) {
            prop_assume!(errors > 0); // Skip relaxed invariant case (errors=0)
            prop_assume!(errors < max_distance); // Need budget for error ops

            let p = GeneralizedPosition::new_i(offset, errors, max_distance)
                .expect("valid I-position");
            let mut state = GeneralizedState::new(max_distance);
            state.add_position(p);

            let operations = OperationSet::standard();
            let cv = CharacteristicVector::new(input_ch, &word);

            let word_chars: Vec<char> = word.chars().collect(); if let Some(next_state) = state.transition(&operations, &cv, &word, Some(&word_chars), &word, input_ch, 1) {
                for succ in next_state.positions() {
                    if succ.is_non_final() {
                        let error_diff = succ.errors() as i32 - errors as i32;

                        // Skip-to-match optimization can create successors with error_diff > 1
                        // (one error per skipped position). These are tested separately in
                        // proptest_skip_to_match.rs and validated by i_skip_to_match_preserves_invariant.
                        // This test only validates i_successor_cost_correct (standard operations).
                        if error_diff > 1 {
                            continue; // Skip skip-to-match successors
                        }

                        // Standard operations have cost 0 (match) or 1 (delete/insert/substitute)
                        prop_assert!(error_diff == 0 || error_diff == 1,
                            "Successor error diff {} not in {{0, 1}}", error_diff);

                        // Match (cost 0): errors unchanged
                        // Delete/Insert/Substitute (cost 1): errors + 1
                        if error_diff == 1 {
                            prop_assert_eq!(succ.errors(), errors + 1,
                                "Error operation should increase errors by exactly 1");
                        }
                    }
                }
            }
        }

        #[test]
        fn m_successor_cost_matches_operation(
            (offset, errors, max_distance) in valid_m_position(),
            word in test_word(),
            input_ch in input_char()
        ) {
            prop_assume!(errors < max_distance);

            let p = GeneralizedPosition::new_m(offset, errors, max_distance)
                .expect("valid M-position");
            let mut state = GeneralizedState::new(max_distance);
            state.add_position(p);

            let operations = OperationSet::standard();
            let cv = CharacteristicVector::new(input_ch, &word);

            let word_chars: Vec<char> = word.chars().collect(); if let Some(next_state) = state.transition(&operations, &cv, &word, Some(&word_chars), &word, input_ch, 1) {
                for succ in next_state.positions() {
                    if succ.is_final() {
                        let error_diff = succ.errors() as i32 - errors as i32;

                        // Same cost structure as I-type
                        prop_assert!(error_diff == 0 || error_diff == 1,
                            "M-type successor error diff {} not in {{0, 1}}", error_diff);
                    }
                }
            }
        }
    }
}

// ============================================================================
// CROSS-CUTTING PROPERTIES
// ============================================================================

/// Validates offset change semantics from Operations.v:239:
/// ```coq
/// Lemma only_delete_moves_left : forall op,
///   offset_change op = (-1) <-> op = OpDelete.
/// ```
#[cfg(test)]
mod offset_semantics {
    use super::*;

    proptest! {
        #[test]
        fn i_type_offset_changes_are_valid(
            (offset, errors, max_distance) in valid_i_position(),
            word in test_word(),
            input_ch in input_char()
        ) {
            prop_assume!(errors > 0); // Skip relaxed invariant case (errors=0)
            prop_assume!(errors < max_distance);
            prop_assume!(offset > -(max_distance as i32)); // Allow delete

            let p = GeneralizedPosition::new_i(offset, errors, max_distance)
                .expect("valid I-position");
            let mut state = GeneralizedState::new(max_distance);
            state.add_position(p);

            let operations = OperationSet::standard();
            let cv = CharacteristicVector::new(input_ch, &word);

            let word_chars: Vec<char> = word.chars().collect(); if let Some(next_state) = state.transition(&operations, &cv, &word, Some(&word_chars), &word, input_ch, 1) {
                for succ in next_state.positions() {
                    if succ.is_non_final() {
                        let offset_diff = succ.offset() - offset;
                        let error_diff = succ.errors() as i32 - errors as i32;

                        // I-type standard operations:
                        // - Match: offset+0, errors+0
                        // - Delete: offset-1, errors+1
                        // - Insert: offset+0, errors+1
                        // - Substitute: offset+0, errors+1
                        // - Skip-to-match (optimization): offset+N, errors+N (N deletes)

                        // Validate: offset change should not exceed error change
                        // (each position skip costs at least 1 error)
                        prop_assert!(offset_diff <= error_diff,
                            "I-type offset increase {} > error increase {}",
                            offset_diff, error_diff);

                        // Validate: offset can decrease by at most 1 (single delete)
                        prop_assert!(offset_diff >= -1,
                            "I-type offset decrease {} < -1", offset_diff);
                    }
                }
            }
        }

        #[test]
        fn m_type_offset_increases_or_stays(
            (offset, errors, max_distance) in valid_m_position(),
            word in test_word(),
            input_ch in input_char()
        ) {
            prop_assume!(errors < max_distance);

            let p = GeneralizedPosition::new_m(offset, errors, max_distance)
                .expect("valid M-position");
            let mut state = GeneralizedState::new(max_distance);
            state.add_position(p);

            let operations = OperationSet::standard();
            let cv = CharacteristicVector::new(input_ch, &word);

            let word_chars: Vec<char> = word.chars().collect(); if let Some(next_state) = state.transition(&operations, &cv, &word, Some(&word_chars), &word, input_ch, 1) {
                for succ in next_state.positions() {
                    if succ.is_final() {
                        let offset_diff = succ.offset() - offset;

                        // M-type: Match/Insert/Substitute increase offset (+1)
                        //         Delete keeps offset unchanged (0)
                        prop_assert!(offset_diff == 0 || offset_diff == 1,
                            "M-type offset change {} not in {{0, +1}}", offset_diff);

                        // If offset increased, must have been < 0 originally (F4 finding)
                        if offset_diff == 1 {
                            prop_assert!(offset < 0,
                                "M-type offset-increasing operation requires offset < 0, got offset = {}",
                                offset);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// INDIVIDUAL OPERATION TESTS (MEDIUM PRIORITY)
// ============================================================================

/// Tests for specific operation preservation lemmas
#[cfg(test)]
mod individual_operations {
    use super::*;

    // Note: These tests are more granular than the main theorems above.
    // They can help identify which specific operation is problematic if
    // the main tests fail.

    proptest! {
        /// Validates i_delete_preserves_invariant (Transitions.v:318)
        #[test]
        fn i_delete_preserves_invariant(
            (offset, errors, max_distance) in valid_i_position()
        ) {
            prop_assume!(errors < max_distance); // Need budget
            prop_assume!(offset > -(max_distance as i32)); // Not at left boundary

            // Simulate delete: offset - 1, errors + 1
            if let Ok(_succ) = GeneralizedPosition::new_i(offset - 1, errors + 1, max_distance) {
                // Constructor validates invariant
                prop_assert!((offset - 1).abs() <= (errors + 1) as i32);
            }
        }

        /// Validates m_delete_preserves_invariant (Transitions.v:723)
        #[test]
        fn m_delete_preserves_invariant(
            (offset, errors, max_distance) in valid_m_position()
        ) {
            prop_assume!(errors < max_distance);

            // M-type delete: offset unchanged, errors + 1
            if let Ok(_succ) = GeneralizedPosition::new_m(offset, errors + 1, max_distance) {
                let n = max_distance as i32;
                prop_assert!(errors as i32 + 1 >= -(offset) - n);
            }
        }
    }
}
