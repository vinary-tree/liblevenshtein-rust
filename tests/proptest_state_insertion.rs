//! Executable invariants for canonical transducer-state insertion.
//!
//! These properties mirror the assumption-free Rocq model in
//! `docs/verification/core/theories/Conformance/RustSubsumption.v` and the
//! bounded lifecycle model in `docs/verification/tla/Subsumption.tla`.

use liblevenshtein::transducer::{Algorithm, Position, PositionF64, State, StateF64};
use proptest::prelude::*;

fn algorithm() -> impl Strategy<Value = Algorithm> {
    prop_oneof![
        Just(Algorithm::Standard),
        Just(Algorithm::Transposition),
        Just(Algorithm::MergeAndSplit),
    ]
}

fn position() -> impl Strategy<Value = Position> {
    (0usize..=8, 0usize..=5, any::<bool>()).prop_map(|(term_index, errors, special)| {
        if special {
            Position::new_osa_transposing(term_index, errors)
        } else {
            Position::new(term_index, errors)
        }
    })
}

fn position_f64() -> impl Strategy<Value = PositionF64> {
    (0usize..=8, 0u8..=10, any::<bool>()).prop_map(|(term_index, half_cost, special)| {
        let cost = f64::from(half_cost) / 2.0;
        if special {
            PositionF64::new_special(term_index, cost)
        } else {
            PositionF64::new(term_index, cost)
        }
    })
}

fn pairwise_canonical(positions: &[Position], algorithm: Algorithm, query_length: usize) -> bool {
    positions.iter().enumerate().all(|(index, left)| {
        positions.iter().skip(index + 1).all(|right| {
            left != right
                && !left.subsumes(right, algorithm, query_length)
                && !right.subsumes(left, algorithm, query_length)
        })
    })
}

fn pairwise_canonical_f64(
    positions: &[PositionF64],
    algorithm: Algorithm,
    query_length: usize,
    max_index_op_cost: f64,
) -> bool {
    positions.iter().enumerate().all(|(index, left)| {
        positions.iter().skip(index + 1).all(|right| {
            !left.approx_eq(right)
                && !left.subsumes(right, algorithm, query_length, max_index_op_cost)
                && !right.subsumes(left, algorithm, query_length, max_index_op_cost)
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    /// `insert` returns true exactly when no equal or dominating representative
    /// was present before the call. A true result is therefore the worklist
    /// event used by epsilon-closure, regardless of the post-state length.
    #[test]
    fn integer_insert_result_is_a_retention_witness(
        raw in prop::collection::vec(position(), 0..24),
        candidate in position(),
        algorithm in algorithm(),
        query_length in 0usize..=8,
    ) {
        let mut state = State::new();
        for position in raw {
            state.insert(position, algorithm, query_length);
        }
        let before = state.positions().to_vec();
        let expected_retained = !before.iter().any(|existing| {
            existing == &candidate || existing.subsumes(&candidate, algorithm, query_length)
        });

        let retained = state.insert(candidate, algorithm, query_length);
        prop_assert_eq!(retained, expected_retained);

        if retained {
            prop_assert!(state.positions().contains(&candidate));
            for existing in &before {
                let should_remain = !candidate.subsumes(existing, algorithm, query_length);
                prop_assert_eq!(state.positions().contains(existing), should_remain);
            }
        } else {
            prop_assert_eq!(state.positions(), before.as_slice());
        }
        prop_assert!(pairwise_canonical(state.positions(), algorithm, query_length));
    }

    #[test]
    fn weighted_insert_result_is_a_retention_witness(
        raw in prop::collection::vec(position_f64(), 0..24),
        candidate in position_f64(),
        algorithm in algorithm(),
        query_length in 0usize..=8,
        max_index_half_cost in 1u8..=8,
    ) {
        let max_index_op_cost = f64::from(max_index_half_cost) / 2.0;
        let mut state = StateF64::new();
        for position in raw {
            state.insert(position, algorithm, query_length, max_index_op_cost);
        }
        let before = state.positions().to_vec();
        let expected_retained = !before.iter().any(|existing| {
            existing.approx_eq(&candidate)
                || existing.subsumes(
                    &candidate,
                    algorithm,
                    query_length,
                    max_index_op_cost,
                )
        });

        let retained = state.insert(
            candidate,
            algorithm,
            query_length,
            max_index_op_cost,
        );
        prop_assert_eq!(retained, expected_retained);

        if retained {
            prop_assert!(state.positions().iter().any(|position| position.approx_eq(&candidate)));
            for existing in &before {
                let should_remain = !candidate.subsumes(
                    existing,
                    algorithm,
                    query_length,
                    max_index_op_cost,
                );
                prop_assert_eq!(
                    state.positions().iter().any(|position| position.approx_eq(existing)),
                    should_remain,
                );
            }
        } else {
            prop_assert_eq!(state.positions().len(), before.len());
            prop_assert!(state.positions().iter().zip(&before).all(|(left, right)| left.approx_eq(right)));
        }
        prop_assert!(pairwise_canonical_f64(
            state.positions(),
            algorithm,
            query_length,
            max_index_op_cost,
        ));
    }
}
