//! Property contracts for the runtime-to-static automaton-variant seam.

use liblevenshtein::transducer::transition::transition_position;
use liblevenshtein::transducer::{Algorithm, Position, PositionKind};
use proptest::prelude::*;

fn position_for(algorithm: Algorithm, special: bool, index: usize, errors: usize) -> Position {
    if !special {
        return Position::new(index, errors);
    }
    match algorithm {
        Algorithm::Standard | Algorithm::Transposition => {
            Position::new_osa_transposing(index, errors)
        }
        Algorithm::MergeAndSplit => Position::new_splitting(index, errors),
        _ => unreachable!("strategy emits only built-in Phase 5 algorithms"),
    }
}

fn reference_subsumes(
    lhs: &Position,
    rhs: &Position,
    algorithm: Algorithm,
    query_length: usize,
) -> bool {
    if lhs.num_errors > rhs.num_errors {
        return false;
    }

    match algorithm {
        Algorithm::Standard => {
            lhs.term_index.abs_diff(rhs.term_index) <= rhs.num_errors - lhs.num_errors
        }
        Algorithm::Transposition => match (lhs.is_special(), rhs.is_special()) {
            (true, true) => lhs.term_index == rhs.term_index,
            (true, false) | (false, true) => false,
            (false, false) => {
                lhs.term_index.abs_diff(rhs.term_index) <= rhs.num_errors - lhs.num_errors
            }
        },
        Algorithm::MergeAndSplit => {
            lhs.is_special() == rhs.is_special()
                && lhs.term_index <= query_length
                && !(lhs.is_special()
                    && lhs.term_index >= query_length
                    && rhs.term_index < query_length)
                && lhs.num_errors < rhs.num_errors
                && lhs.term_index == rhs.term_index
        }
        _ => unreachable!("strategy emits only built-in Phase 5 algorithms"),
    }
}

fn algorithm_strategy() -> impl Strategy<Value = Algorithm> {
    prop_oneof![
        Just(Algorithm::Standard),
        Just(Algorithm::Transposition),
        Just(Algorithm::MergeAndSplit),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    #[test]
    fn public_dispatch_matches_the_frozen_subsumption_formula(
        algorithm in algorithm_strategy(),
        query_length in 0usize..24,
        lhs_index in 0usize..28,
        rhs_index in 0usize..28,
        lhs_errors in 0usize..8,
        rhs_errors in 0usize..8,
        lhs_special in any::<bool>(),
        rhs_special in any::<bool>(),
    ) {
        let lhs = position_for(algorithm, lhs_special, lhs_index, lhs_errors);
        let rhs = position_for(algorithm, rhs_special, rhs_index, rhs_errors);
        prop_assert_eq!(
            lhs.subsumes(&rhs, algorithm, query_length),
            reference_subsumes(&lhs, &rhs, algorithm, query_length),
        );
    }

    #[test]
    fn typed_transition_dispatch_is_deterministic_bounded_and_kind_safe(
        algorithm in algorithm_strategy(),
        query_length in 0usize..16,
        term_index in 0usize..18,
        max_distance in 0usize..5,
        error_seed in any::<usize>(),
        special in any::<bool>(),
        prefix_mode in any::<bool>(),
        cv in prop::collection::vec(any::<bool>(), 0..8),
    ) {
        // Construct a valid budget directly instead of rejecting independent
        // samples. This keeps all 2,000 cases productive, including distance 0.
        let errors = error_seed % (max_distance + 1);
        let position = position_for(algorithm, special, term_index, errors);
        let first = transition_position(
            &position,
            &cv,
            query_length,
            max_distance,
            algorithm,
            prefix_mode,
        );
        let second = transition_position(
            &position,
            &cv,
            query_length,
            max_distance,
            algorithm,
            prefix_mode,
        );

        prop_assert_eq!(&first, &second);
        for successor in &first {
            prop_assert!(successor.num_errors <= max_distance);
            prop_assert_eq!(successor.aux(), 0);
            match algorithm {
                Algorithm::Standard => prop_assert_eq!(successor.kind(), PositionKind::Normal),
                Algorithm::Transposition => prop_assert!(matches!(
                    successor.kind(),
                    PositionKind::Normal | PositionKind::OsaTransposing
                )),
                Algorithm::MergeAndSplit => prop_assert!(matches!(
                    successor.kind(),
                    PositionKind::Normal | PositionKind::Splitting
                )),
                _ => unreachable!("strategy emits only built-in Phase 5 algorithms"),
            }
        }
    }

    #[test]
    fn position_order_is_total_and_distinguishes_continuation_languages(
        index in 0usize..32,
        errors in 0usize..8,
    ) {
        let normal = Position::new(index, errors);
        let osa = Position::new_osa_transposing(index, errors);
        let split = Position::new_splitting(index, errors);

        prop_assert!(normal < osa);
        prop_assert!(osa < split);
        prop_assert_ne!(normal, osa);
        prop_assert_ne!(osa, split);
        prop_assert_eq!(normal.cmp(&split), split.cmp(&normal).reverse());
    }
}

#[test]
fn position_layout_and_public_accessors_are_stable() {
    #[cfg(target_pointer_width = "64")]
    assert_eq!(std::mem::size_of::<Position>(), 24);

    let normal = Position::new(4, 2);
    assert_eq!(normal.kind(), PositionKind::Normal);
    assert_eq!(normal.aux(), 0);
    assert!(!normal.is_special());

    let osa = Position::new_osa_transposing(4, 2);
    assert_eq!(osa.kind(), PositionKind::OsaTransposing);
    assert_eq!(osa.aux(), 0);
    assert!(osa.is_special());

    let split = Position::new_splitting(4, 2);
    assert_eq!(split.kind(), PositionKind::Splitting);
    assert_eq!(split.aux(), 0);
    assert!(split.is_special());
}
