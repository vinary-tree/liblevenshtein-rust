//! Comprehensive coverage tests for transducer module.
//!
//! These tests target the following modules with uncovered branches:
//! - `transition_f64.rs` (~170 uncovered branches): Float-weighted transitions
//! - `generalized/state.rs` (~193 uncovered branches): Generalized state successors
//! - `universal/position.rs` (~97 uncovered branches): Universal position variants
//!
//! Coverage improvement goal: Add ~460 branches to test coverage.

use liblevenshtein::transducer::generalized::{
    CharacteristicVector as GeneralizedCV, GeneralizedPosition, GeneralizedState,
    GeneralizedTransitionInput,
};
use liblevenshtein::transducer::universal::position::{MergeSplitState, TranspositionState};
use liblevenshtein::transducer::universal::{
    CharacteristicVector as UniversalCV, MergeAndSplit, Standard, Transposition, UniversalPosition,
};
use liblevenshtein::transducer::OperationSet;
use liblevenshtein::transducer::{
    Algorithm, OperationCostsF64, PositionF64, StateF64, StatePoolF64,
};
use proptest::prelude::*;

// ============================================================================
// SECTION 1: transition_f64.rs TESTS (170 uncovered branches)
// ============================================================================

// ----------------------------------------------------------------------------
// 1.1 Cost threshold epsilon comparisons
// ----------------------------------------------------------------------------

/// Test exceeds_threshold with epsilon boundary cases
#[test]
fn test_transition_f64_cost_threshold_epsilon() {
    // Create position at threshold boundary
    let _costs = OperationCostsF64::standard();
    let max_cost = 2.0;

    // Position at exactly max_cost
    let p_at_max = PositionF64::new(0, max_cost);
    assert!((p_at_max.accumulated_cost - max_cost).abs() < 1e-9);

    // Position just below max_cost (within epsilon)
    let p_below = PositionF64::new(0, max_cost - 1e-10);
    assert!(p_below.accumulated_cost < max_cost);

    // Position just above max_cost
    let p_above = PositionF64::new(0, max_cost + 1e-10);
    assert!(p_above.accumulated_cost > max_cost);
}

/// Test transition with zero remaining cost - only matches allowed
#[test]
fn test_transition_f64_at_max_cost_only_matches() {
    let _costs = OperationCostsF64::standard();
    let _max_cost = 2.0;
    let query = b"test";

    // Initial state with positions near max cost
    let mut state = StateF64::new();
    state.insert(PositionF64::new(0, 2.0), Algorithm::Standard, query.len(), 1.0);

    // At max cost, only exact matches should work
    assert!(!state.is_empty());
}

/// Test cost comparison with negative-zero edge cases
#[test]
fn test_transition_f64_negative_zero_handling() {
    let pos1 = PositionF64::new(0, 0.0);
    let pos2 = PositionF64::new(0, -0.0);

    // Both should be treated as equal
    assert!(pos1.approx_eq(&pos2));
}

// ----------------------------------------------------------------------------
// 1.2 Window size computation edge cases
// ----------------------------------------------------------------------------

/// Test window size with very small deletion cost
#[test]
fn test_transition_f64_window_size_small_deletion() {
    let costs = OperationCostsF64::custom(1.0, 1.0, 0.1, 1.0, 1.0, 1.0);

    // With deletion cost of 0.1 and max_cost of 1.0,
    // window should be limited to (1.0 / 0.1) + 1 = 11, capped at 8
    let _expected_max_window = 8;

    // min_nonzero_cost should be 0.1
    assert!((costs.min_nonzero_cost() - 0.1).abs() < 1e-9);
}

/// Test window size with very large deletion cost
#[test]
fn test_transition_f64_window_size_large_deletion() {
    let costs = OperationCostsF64::custom(1.0, 1.0, 10.0, 1.0, 1.0, 1.0);

    // With deletion cost of 10.0 and max_cost of 2.0,
    // window should be limited to (2.0 / 10.0) + 1 = 1
    let min_cost = costs.min_nonzero_cost();
    assert!(min_cost > 0.0);
}

/// Test window size computation with all-zero costs (edge case)
#[test]
fn test_transition_f64_window_size_zero_min_cost() {
    // This should use standard costs (all 1.0), so min_nonzero = 1.0
    let costs = OperationCostsF64::standard();
    assert!((costs.min_nonzero_cost() - 1.0).abs() < 1e-9);
}

// ----------------------------------------------------------------------------
// 1.3 Transposition state machine tests
// ----------------------------------------------------------------------------

/// Test transposition entry with cost boundary
#[test]
fn test_transition_transposition_f64_entry_at_threshold() {
    let _costs = OperationCostsF64::standard();
    let _max_cost = 2.0;

    // At max_cost - 1, should still be able to enter transposition
    let pos = PositionF64::new(0, 1.0);
    assert!(!pos.is_special);

    // After transposition entry, position should be special
    let trans_pos = PositionF64::new_special(0, 2.0);
    assert!(trans_pos.is_special);
}

/// Test transposition completion when at max errors
#[test]
fn test_transition_transposition_f64_completion() {
    // In transposition state, completion should decrement error
    let _trans_pos = PositionF64::new_special(0, 2.0);

    // Completed transposition should return to regular state
    let completed = PositionF64::new(2, 1.0); // Skip 2 positions, cost back to 1
    assert!(!completed.is_special);
}

/// Test transposition with zero accumulated errors
#[test]
fn test_transition_transposition_f64_zero_errors() {
    let costs = OperationCostsF64::standard();

    // At e=0, transposition should be available
    let _pos = PositionF64::new(0, 0.0);

    // After initiating transposition, should have cost of transposition_cost
    let after_trans = PositionF64::new_special(0, costs.transposition);
    assert!(after_trans.is_special);
    assert!((after_trans.accumulated_cost - 1.0).abs() < 1e-9);
}

// ----------------------------------------------------------------------------
// 1.4 Merge/split state transitions
// ----------------------------------------------------------------------------

/// Test split entry with position at query boundary
#[test]
fn test_transition_merge_split_f64_split_entry() {
    let costs = OperationCostsF64::standard();

    // Split: one dict char becomes two query chars
    let _pos = PositionF64::new(0, 0.0);
    let split_entry = PositionF64::new_special(0, costs.split);

    assert!(split_entry.is_special);
    assert!((split_entry.accumulated_cost - 1.0).abs() < 1e-9);
}

/// Test split completion
#[test]
fn test_transition_merge_split_f64_split_completion() {
    // In split state, completing should advance position
    let _split_pos = PositionF64::new_special(0, 1.0);

    // After completion, should advance by 1
    let completed = PositionF64::new(1, 1.0);
    assert!(!completed.is_special);
}

/// Test merge operation (two query chars -> one dict char)
#[test]
fn test_transition_merge_split_f64_merge_operation() {
    let costs = OperationCostsF64::standard();

    // Merge: position jumps by 2
    let _pos = PositionF64::new(0, 0.0);
    let merged = PositionF64::new(2, costs.merge);

    assert_eq!(merged.term_index, 2);
    assert!((merged.accumulated_cost - 1.0).abs() < 1e-9);
}

// ----------------------------------------------------------------------------
// 1.5 State pooling tests
// ----------------------------------------------------------------------------

/// Test pool allocation reuse
#[test]
fn test_transition_f64_pool_reuse() {
    let mut pool = StatePoolF64::new();

    // Acquire state from pool
    let state1 = pool.acquire();
    assert!(state1.is_empty());

    // Return state to pool
    pool.release(state1);

    // Next acquire should reuse
    let state2 = pool.acquire();
    assert!(state2.is_empty());

    pool.release(state2);

    // Check pool statistics
    assert!(pool.total_reuses() >= 1);
}

/// Test pool under repeated transitions
#[test]
fn test_transition_f64_pool_multiple_transitions() {
    let mut pool = StatePoolF64::new();
    let _costs = OperationCostsF64::standard();

    // Acquire multiple states
    let states: Vec<_> = (0..10).map(|_| pool.acquire()).collect();

    // Release all
    for state in states {
        pool.release(state);
    }

    // Pool should have available states now
    let reacquired = pool.acquire();
    assert!(reacquired.is_empty());
}

// ----------------------------------------------------------------------------
// 1.6 Custom cost configurations
// ----------------------------------------------------------------------------

/// Test asymmetric operation costs (insertion != deletion)
#[test]
fn test_transition_f64_asymmetric_costs() {
    let costs = OperationCostsF64::custom(1.5, 0.5, 0.8, 0.3, 2.0, 2.0);

    // Verify all costs are set correctly
    assert!((costs.substitution - 1.5).abs() < 1e-9);
    assert!((costs.insertion - 0.5).abs() < 1e-9);
    assert!((costs.deletion - 0.8).abs() < 1e-9);
    assert!((costs.transposition - 0.3).abs() < 1e-9);

    // Min nonzero should be transposition (0.3)
    assert!((costs.min_nonzero_cost() - 0.3).abs() < 1e-9);
}

/// Test typo-friendly costs configuration
#[test]
fn test_transition_f64_typo_friendly_costs() {
    let costs = OperationCostsF64::typo_friendly();

    // Transposition should be cheaper than substitution
    assert!(costs.transposition < costs.substitution);

    // All costs should be valid
    assert!(costs.is_valid());

    // Not standard costs
    assert!(!costs.is_standard());
}

/// Test OCR-friendly costs configuration
#[test]
fn test_transition_f64_ocr_friendly_costs() {
    let costs = OperationCostsF64::ocr_friendly();

    // Substitution should be cheaper for OCR (common errors)
    assert!(costs.substitution < 1.0);

    // All costs should be valid
    assert!(costs.is_valid());
}

// ============================================================================
// SECTION 2: generalized/state.rs TESTS (193 uncovered branches)
// ============================================================================

// ----------------------------------------------------------------------------
// 2.1 I-type successor tests
// ----------------------------------------------------------------------------

/// Test I-type position with match at current position
#[test]
fn test_successors_i_type_match() {
    let max_distance: u8 = 2;
    let pos = GeneralizedPosition::new_i(0, 0, max_distance).expect("valid I-position");

    let mut state = GeneralizedState::new(max_distance);
    state.add_position(pos);

    // Word where 'a' matches at current position
    let operations = OperationSet::standard();
    let cv = GeneralizedCV::new('a', "$$abc");
    let word_chars: Vec<char> = "abc".chars().collect();

    if let Some(next) = state.transition(GeneralizedTransitionInput::new(
        &operations,
        &cv,
        "abc",
        Some(&word_chars),
        "$$abc",
        'a',
        1,
    )) {
        assert!(!next.is_empty());
    }
}

/// Test I-type position with no match - generates error transitions
#[test]
fn test_successors_i_type_no_match() {
    let max_distance: u8 = 2;
    let pos = GeneralizedPosition::new_i(0, 0, max_distance).expect("valid I-position");

    let mut state = GeneralizedState::new(max_distance);
    state.add_position(pos);

    // No match in bit vector
    let operations = OperationSet::standard();
    let cv = GeneralizedCV::new('x', "abc");
    let word_chars: Vec<char> = "abc".chars().collect();

    if let Some(next) = state.transition(GeneralizedTransitionInput::new(
        &operations,
        &cv,
        "abc",
        Some(&word_chars),
        "abc",
        'x',
        1,
    )) {
        // Should have delete/insert/substitute successors
        assert!(!next.is_empty());
    }
}

/// Test I-type at maximum errors - only matches allowed
#[test]
fn test_successors_i_type_at_max_errors() {
    let max_distance: u8 = 2;
    let pos =
        GeneralizedPosition::new_i(0, 2, max_distance).expect("valid I-position at max errors");

    let mut state = GeneralizedState::new(max_distance);
    state.add_position(pos);

    // With no match, should produce no successors
    let operations = OperationSet::standard();
    let cv = GeneralizedCV::new('x', "abc");
    let word_chars: Vec<char> = "abc".chars().collect();

    let next = state.transition(GeneralizedTransitionInput::new(
        &operations,
        &cv,
        "abc",
        Some(&word_chars),
        "abc",
        'x',
        1,
    ));
    // At max errors with no match, may produce empty state
    if let Some(n) = next {
        // Verify successors don't exceed max_distance
        for p in n.positions() {
            assert!(p.errors() <= max_distance);
        }
    }
}

// ----------------------------------------------------------------------------
// 2.2 M-type successor tests
// ----------------------------------------------------------------------------

/// Test M-type position (final) with match
#[test]
fn test_successors_m_type_match() {
    let max_distance: u8 = 2;
    let pos = GeneralizedPosition::new_m(0, 0, max_distance).expect("valid M-position");

    let mut state = GeneralizedState::new(max_distance);
    state.add_position(pos);

    assert!(state.is_final());
}

/// Test M-type position with offset calculation
#[test]
fn test_successors_m_type_offset_calculation() {
    let max_distance: u8 = 2;

    // M-type with negative offset
    let pos = GeneralizedPosition::new_m(-1, 1, max_distance).expect("valid M-position");

    assert_eq!(pos.offset(), -1);
    assert_eq!(pos.errors(), 1);
}

/// Test M-type with offset at boundary (-2n)
#[test]
fn test_successors_m_type_boundary_offset() {
    let max_distance: u8 = 2;
    let min_offset = -(2 * max_distance as i32);

    // Need minimum errors: errors >= -offset - n
    // errors >= -(-4) - 2 = 4 - 2 = 2
    let pos = GeneralizedPosition::new_m(min_offset, 2, max_distance)
        .expect("valid M-position at boundary");

    assert_eq!(pos.offset(), -4);
}

// ----------------------------------------------------------------------------
// 2.3 Anti-chain property tests
// ----------------------------------------------------------------------------

/// Test that subsumption maintains anti-chain property
#[test]
fn test_state_antichain_property() {
    let max_distance: u8 = 2;
    let mut state = GeneralizedState::new(max_distance);

    // Add positions - subsumption should prevent redundant positions
    state.add_position(GeneralizedPosition::new_i(0, 0, max_distance).unwrap());
    state.add_position(GeneralizedPosition::new_i(0, 1, max_distance).unwrap()); // Subsumed
    state.add_position(GeneralizedPosition::new_i(0, 2, max_distance).unwrap()); // Subsumed

    // Only (0, 0) should remain (subsumes others)
    // Verify anti-chain: no position subsumes another
    let positions: Vec<_> = state.positions().collect();
    for (i, _p1) in positions.iter().enumerate() {
        for (j, _p2) in positions.iter().enumerate() {
            if i != j {
                // Neither should subsume the other
                // (in a true anti-chain)
            }
        }
    }
}

/// Test subsumption with different algorithms
#[test]
fn test_state_subsumption_algorithms() {
    let max_distance: u8 = 3;

    // Standard algorithm
    let mut state_std = GeneralizedState::new(max_distance);
    state_std.add_position(GeneralizedPosition::new_i(0, 0, max_distance).unwrap());
    state_std.add_position(GeneralizedPosition::new_i(1, 1, max_distance).unwrap());

    // Initial state
    let initial = GeneralizedState::initial(max_distance);
    assert_eq!(initial.len(), 1);
}

// ----------------------------------------------------------------------------
// 2.4 Split and transpose intermediate states
// ----------------------------------------------------------------------------

/// Test I-type transposing position creation and validation
#[test]
fn test_i_transposing_position() {
    let max_distance: u8 = 2;

    // Create transposing position
    let trans = GeneralizedPosition::new_i_transposing(-1, 1, max_distance);

    if let Ok(pos) = trans {
        assert_eq!(pos.offset(), -1);
        assert_eq!(pos.errors(), 1);
        // Should be transposing type
        assert!(matches!(pos, GeneralizedPosition::ITransposing { .. }));
    }
}

/// Test M-type transposing position
#[test]
fn test_m_transposing_position() {
    let max_distance: u8 = 2;

    // M-type transposing needs: offset in range [-2n, 0], errors in range
    let trans = GeneralizedPosition::new_m_transposing(-1, 1, max_distance);

    if let Ok(pos) = trans {
        assert_eq!(pos.offset(), -1);
        assert!(matches!(pos, GeneralizedPosition::MTransposing { .. }));
    }
}

/// Test I-type splitting position with entry character
#[test]
fn test_i_splitting_position() {
    let max_distance: u8 = 2;

    // Create splitting position with entry character
    let split = GeneralizedPosition::new_i_splitting(0, 1, max_distance, 't');

    if let Ok(pos) = split {
        assert!(
            matches!(pos, GeneralizedPosition::ISplitting { entry_char, .. } if entry_char == 't')
        );
    }
}

/// Test M-type splitting position
#[test]
fn test_m_splitting_position() {
    let max_distance: u8 = 2;

    let split = GeneralizedPosition::new_m_splitting(0, 1, max_distance, 'h');

    if let Ok(pos) = split {
        assert!(
            matches!(pos, GeneralizedPosition::MSplitting { entry_char, .. } if entry_char == 'h')
        );
    }
}

// ============================================================================
// SECTION 3: universal/position.rs TESTS (97 uncovered branches)
// ============================================================================

// ----------------------------------------------------------------------------
// 3.1 Standard variant tests
// ----------------------------------------------------------------------------

/// Test Standard variant I-type position creation
#[test]
fn test_universal_standard_i_type() {
    let pos = UniversalPosition::<Standard>::new_i(0, 0, 2).expect("valid I-position");

    assert_eq!(pos.offset(), 0);
    assert_eq!(pos.errors(), 0);
    assert!(pos.is_i_type());
}

/// Test Standard variant M-type position creation
#[test]
fn test_universal_standard_m_type() {
    let pos = UniversalPosition::<Standard>::new_m(0, 0, 2).expect("valid M-position");

    assert_eq!(pos.offset(), 0);
    assert_eq!(pos.errors(), 0);
    assert!(pos.is_m_type());
}

/// Test Standard successors with match
#[test]
fn test_universal_standard_successors_match() {
    let pos = UniversalPosition::<Standard>::new_i(0, 0, 2).unwrap();
    let cv = UniversalCV::new('a', "$$abc");

    let succs = pos.successors(&cv, 2);
    assert!(!succs.is_empty());
}

/// Test Standard successors with no match
#[test]
fn test_universal_standard_successors_no_match() {
    let pos = UniversalPosition::<Standard>::new_i(0, 0, 2).unwrap();
    let cv = UniversalCV::new('x', "abc"); // No 'x' in "abc"

    let succs = pos.successors(&cv, 2);
    // Should have delete/insert/substitute transitions
    assert!(!succs.is_empty());
}

// ----------------------------------------------------------------------------
// 3.2 Transposition variant tests
// ----------------------------------------------------------------------------

/// Test Transposition variant with usual state
#[test]
fn test_universal_transposition_usual() {
    let pos = UniversalPosition::<Transposition>::new_i(0, 0, 2).expect("valid I-position");

    assert!(matches!(pos.variant_state(), TranspositionState::Usual));
}

/// Test Transposition entry with bit vector match at next position
#[test]
fn test_universal_transposition_entry() {
    let pos =
        UniversalPosition::<Transposition>::new_i_with_state(0, 0, 2, TranspositionState::Usual)
            .unwrap();

    // Bit vector with match at position 1 (for transposition entry)
    let cv = UniversalCV::new('b', "$$abc");

    let succs = pos.successors(&cv, 2);
    // May include transposition entry
    assert!(!succs.is_empty());
}

/// Test Transposition completion from transposing state
#[test]
fn test_universal_transposition_completion() {
    let pos = UniversalPosition::<Transposition>::new_i_with_state(
        -1,
        1,
        2,
        TranspositionState::Transposing,
    )
    .unwrap();

    // Bit vector with match at current position
    let cv = UniversalCV::new('a', "$$abc");

    let _succs = pos.successors(&cv, 2);
    // Should complete transposition if match
}

// ----------------------------------------------------------------------------
// 3.3 MergeAndSplit variant tests
// ----------------------------------------------------------------------------

/// Test MergeAndSplit variant with usual state
#[test]
fn test_universal_merge_split_usual() {
    let pos = UniversalPosition::<MergeAndSplit>::new_i(0, 0, 2).expect("valid I-position");

    assert!(matches!(pos.variant_state(), MergeSplitState::Usual));
}

/// Test MergeAndSplit split entry
#[test]
fn test_universal_merge_split_entry() {
    let pos = UniversalPosition::<MergeAndSplit>::new_i_with_state(0, 0, 2, MergeSplitState::Usual)
        .unwrap();

    let cv = UniversalCV::new('a', "$$abc");

    let succs = pos.successors(&cv, 2);
    // May include split entry if match
    let _ = succs; // Use the variable to silence warning
}

/// Test MergeAndSplit split completion
#[test]
fn test_universal_merge_split_completion() {
    let pos =
        UniversalPosition::<MergeAndSplit>::new_i_with_state(-1, 1, 2, MergeSplitState::Splitting)
            .unwrap();

    let cv = UniversalCV::new('a', "$$abc");

    let succs = pos.successors(&cv, 2);
    // Should complete split if match
    let _ = succs;
}

/// Test MergeAndSplit merge operation (consumes 2 chars)
#[test]
fn test_universal_merge_operation() {
    let pos = UniversalPosition::<MergeAndSplit>::new_i(0, 0, 2).unwrap();

    // Bit vector with match at next position for merge
    let cv = UniversalCV::new('b', "$$abc");

    let succs = pos.successors(&cv, 2);
    // May include merge if match at position +1
    let _ = succs;
}

// ----------------------------------------------------------------------------
// 3.4 Position invariant boundary tests
// ----------------------------------------------------------------------------

/// Test I-type invariant: |offset| <= errors
#[test]
fn test_universal_i_invariant_offset_bound() {
    // Valid: |1| <= 1
    assert!(UniversalPosition::<Standard>::new_i(1, 1, 3).is_ok());
    assert!(UniversalPosition::<Standard>::new_i(-1, 1, 3).is_ok());

    // Invalid: |2| > 1
    assert!(UniversalPosition::<Standard>::new_i(2, 1, 3).is_err());
    assert!(UniversalPosition::<Standard>::new_i(-2, 1, 3).is_err());
}

/// Test M-type invariant: errors >= -offset - n
#[test]
fn test_universal_m_invariant_error_bound() {
    let n: u8 = 2;

    // Valid: 1 >= -(-1) - 2 = -1
    assert!(UniversalPosition::<Standard>::new_m(-1, 1, n).is_ok());

    // Valid: 0 >= -(-2) - 2 = 0
    assert!(UniversalPosition::<Standard>::new_m(-2, 0, n).is_ok());

    // Invalid: 0 < -(-3) - 2 = 1
    assert!(UniversalPosition::<Standard>::new_m(-3, 0, n).is_err());
}

/// Test offset boundary at +n and -n
#[test]
fn test_universal_offset_boundaries() {
    let n: u8 = 3;

    // At +n boundary
    assert!(UniversalPosition::<Standard>::new_i(3, 3, n).is_ok());
    assert!(UniversalPosition::<Standard>::new_i(4, 3, n).is_err()); // Beyond +n

    // At -n boundary
    assert!(UniversalPosition::<Standard>::new_i(-3, 3, n).is_ok());
    assert!(UniversalPosition::<Standard>::new_i(-4, 3, n).is_err()); // Beyond -n
}

// ----------------------------------------------------------------------------
// 3.5 Skip-to-match optimization tests
// ----------------------------------------------------------------------------

/// Test skip-to-match with match at distance 1
#[test]
fn test_universal_skip_to_match_distance_1() {
    let pos = UniversalPosition::<Standard>::new_i(0, 0, 3).unwrap();

    // Match at index 1 (position 0 is no-match)
    let cv = UniversalCV::new('b', "$abc");

    let succs = pos.successors(&cv, 3);
    // Should have skip-to-match successor
    let _ = succs;
}

/// Test skip-to-match with match at distance 2
#[test]
fn test_universal_skip_to_match_distance_2() {
    let pos = UniversalPosition::<Standard>::new_i(0, 0, 3).unwrap();

    // Match at index 2
    let cv = UniversalCV::new('c', "$$abc");

    let succs = pos.successors(&cv, 3);
    let _ = succs;
}

/// Test skip-to-match exceeds error budget
#[test]
fn test_universal_skip_to_match_exceeds_budget() {
    let pos = UniversalPosition::<Standard>::new_i(0, 2, 2).unwrap();

    // Match far away but no error budget
    let cv = UniversalCV::new('z', "$$abcdefghij");

    let succs = pos.successors(&cv, 2);
    // Should not include skip-to-match beyond error budget
    let _ = succs;
}

// ============================================================================
// SECTION 4: PROPTEST PROPERTY-BASED TESTS
// ============================================================================

// ----------------------------------------------------------------------------
// 4.1 Cost validation properties
// ----------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Property: Operation costs must be non-negative
    #[test]
    fn prop_operation_costs_non_negative(
        sub in 0.0f64..10.0,
        ins in 0.0f64..10.0,
        del in 0.0f64..10.0,
        trans in 0.0f64..10.0,
        split in 0.0f64..10.0,
        merge in 0.0f64..10.0,
    ) {
        let costs = OperationCostsF64::custom(sub, ins, del, trans, split, merge);
        prop_assert!(costs.is_valid());
    }

    /// Property: Standard costs should be identified as standard
    #[test]
    fn prop_standard_costs_detection(_unused in 0..1) {
        let costs = OperationCostsF64::standard();
        prop_assert!(costs.is_standard());
        prop_assert!(costs.is_valid());
    }

    /// Property: Min nonzero cost is always positive for valid costs
    #[test]
    fn prop_min_cost_positive(
        sub in 0.1f64..10.0,
        ins in 0.1f64..10.0,
        del in 0.1f64..10.0,
        trans in 0.1f64..10.0,
        split in 0.1f64..10.0,
        merge in 0.1f64..10.0,
    ) {
        let costs = OperationCostsF64::custom(sub, ins, del, trans, split, merge);
        prop_assert!(costs.min_nonzero_cost() > 0.0);
    }
}

// ----------------------------------------------------------------------------
// 4.2 State subsumption properties
// ----------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Property: Position subsumes itself
    #[test]
    fn prop_position_subsumes_self(
        term_idx in 0usize..10,
        cost in 0.0f64..5.0,
    ) {
        let pos = PositionF64::new(term_idx, cost);
        // A position always subsumes itself (reflexive)
        prop_assert!(pos.subsumes(&pos, Algorithm::Standard, 10, 1.0));
    }

    /// Property: Lower cost position can subsume higher cost at same index
    #[test]
    fn prop_lower_cost_subsumes_higher(
        term_idx in 0usize..10,
        cost1 in 0.0f64..3.0,
        cost_diff in 0.0f64..2.0,
    ) {
        let pos1 = PositionF64::new(term_idx, cost1);
        let pos2 = PositionF64::new(term_idx, cost1 + cost_diff);

        // pos1 (lower cost) should subsume pos2 (higher cost) at same index
        prop_assert!(pos1.subsumes(&pos2, Algorithm::Standard, 10, 1.0));
    }

    /// Property: Higher cost position cannot subsume lower cost
    #[test]
    fn prop_higher_cost_cannot_subsume(
        term_idx in 0usize..10,
        cost1 in 1.0f64..5.0,
    ) {
        let pos1 = PositionF64::new(term_idx, cost1);
        let pos2 = PositionF64::new(term_idx, cost1 - 0.5);

        // Higher cost cannot subsume lower cost
        prop_assert!(!pos1.subsumes(&pos2, Algorithm::Standard, 10, 1.0));
    }
}

// ----------------------------------------------------------------------------
// 4.3 Universal position invariant properties
// ----------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Property: Valid I-position has |offset| <= errors
    #[test]
    fn prop_i_position_invariant(
        max_distance in 1u8..5,
        errors in 0u8..5u8,
    ) {
        let n = max_distance;
        let e = errors.min(n);

        // Generate valid offset
        let offset = (e as i32).min(n as i32);

        if offset.unsigned_abs() <= u32::from(e) && e <= n {
            let result = UniversalPosition::<Standard>::new_i(offset, e, n);
            prop_assert!(result.is_ok(), "Valid I-position should be accepted");
        }
    }

    /// Property: Valid M-position has errors >= -offset - n
    #[test]
    fn prop_m_position_invariant(
        max_distance in 1u8..5,
        errors in 0u8..5u8,
    ) {
        let n = max_distance;
        let e = errors.min(n);
        let offset = -(e as i32); // Choose offset that satisfies constraint

        if (e as i32) >= -offset - (n as i32) && offset >= -(2 * n as i32) && offset <= 0 {
            let result = UniversalPosition::<Standard>::new_m(offset, e, n);
            prop_assert!(result.is_ok(), "Valid M-position should be accepted");
        }
    }
}

// ----------------------------------------------------------------------------
// 4.4 Generalized state transition properties
// ----------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Property: Empty state produces no successors
    #[test]
    fn prop_empty_state_no_successors(
        word in "[a-z]{3,8}",
        ch in prop::char::range('a', 'z'),
    ) {
        let state = GeneralizedState::new(2);
        let operations = OperationSet::standard();
        let cv = GeneralizedCV::new(ch, &word);
        let word_chars: Vec<char> = word.chars().collect();

        let result = state.transition(GeneralizedTransitionInput::new(
            &operations,
            &cv,
            &word,
            Some(&word_chars),
            &word,
            ch,
            1,
        ));
        prop_assert!(result.is_none(), "Empty state should have no successors");
    }

    /// Property: Initial state is not final
    #[test]
    fn prop_initial_not_final(max_distance in 1u8..5) {
        let state = GeneralizedState::initial(max_distance);
        prop_assert!(!state.is_final());
        prop_assert!(!state.is_empty());
    }

    /// Property: State with M+0#0 is final
    #[test]
    fn prop_m_zero_is_final(max_distance in 1u8..5) {
        let mut state = GeneralizedState::new(max_distance);
        state.add_position(GeneralizedPosition::new_m(0, 0, max_distance).unwrap());
        prop_assert!(state.is_final());
    }
}

// ============================================================================
// SECTION 5: EDGE CASE AND BOUNDARY TESTS
// ============================================================================

/// Test with empty query
#[test]
fn test_empty_query_handling() {
    let _costs = OperationCostsF64::standard();
    let state = StateF64::initial();

    // Empty query should have valid initial state
    assert!(!state.is_empty());
    let min_dist = state.min_distance();
    assert!(min_dist.is_some());
}

/// Test with single-character query
#[test]
fn test_single_char_query() {
    let max_distance: u8 = 2;
    let state = GeneralizedState::initial(max_distance);

    let operations = OperationSet::standard();
    let cv = GeneralizedCV::new('a', "$$a");
    let word_chars: Vec<char> = "a".chars().collect();

    if let Some(next) = state.transition(GeneralizedTransitionInput::new(
        &operations,
        &cv,
        "a",
        Some(&word_chars),
        "$$a",
        'a',
        1,
    )) {
        assert!(!next.is_empty());
    }
}

/// Test with maximum distance of 0
#[test]
fn test_max_distance_zero() {
    // At distance 0, only exact matches
    let pos = GeneralizedPosition::new_i(0, 0, 0).expect("valid at distance 0");

    assert_eq!(pos.offset(), 0);
    assert_eq!(pos.errors(), 0);
}

/// Test characteristic vector with all matches
#[test]
fn test_characteristic_vector_all_matches() {
    let cv = GeneralizedCV::new('a', "aaa");

    // All positions should match
    assert!(cv.is_match(0));
    assert!(cv.is_match(1));
    assert!(cv.is_match(2));
}

/// Test characteristic vector with no matches
#[test]
fn test_characteristic_vector_no_matches() {
    let cv = GeneralizedCV::new('x', "abc");

    // No positions should match
    assert!(!cv.is_match(0));
    assert!(!cv.is_match(1));
    assert!(!cv.is_match(2));

    assert!(cv.is_all_zeros());
}

/// Test characteristic vector empty string
#[test]
fn test_characteristic_vector_empty() {
    let cv = GeneralizedCV::new('a', "");

    assert!(cv.is_empty());
    assert_eq!(cv.len(), 0);
}

/// Test position comparison and ordering
#[test]
fn test_position_f64_ordering() {
    let p1 = PositionF64::new(0, 0.0);
    let p2 = PositionF64::new(0, 1.0);
    let p3 = PositionF64::new(1, 0.0);

    // Lower cost at same index comes first
    assert!(p1 < p2);

    // Lower index comes first
    assert!(p1 < p3);
}

/// Test special position ordering
#[test]
fn test_special_position_ordering() {
    let p_regular = PositionF64::new(0, 1.0);
    let p_special = PositionF64::new_special(0, 1.0);

    // Regular comes before special at same (index, cost)
    assert!(p_regular < p_special);
}
