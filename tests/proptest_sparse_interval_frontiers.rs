//! Production sparse dictionary-edge frontiers against dense interval columns.

use liblevenshtein::cost::CostMonoid;
use liblevenshtein::time_series::elastic::{ElasticKernel, PointFrontierStep};
use liblevenshtein::time_series::{ErpConfig, FrechetConfig, MsmConfig, MsmKernel, TwedConfig};
use proptest::prelude::*;

fn assert_sparse_matches_dense<K>(kernel: K, query: &[f64], intervals: &[(f64, f64)], cutoff: f64)
where
    K: ElasticKernel,
    K::Monoid: CostMonoid<Cost = f64>,
{
    let width = kernel
        .column_len(query.len())
        .expect("generated query belongs to the kernel domain");
    let plan = kernel.plan(query);
    let mut dense_previous = vec![K::Monoid::TOP; width];
    let mut sparse_previous = vec![K::Monoid::TOP; width];
    let mut sparse_previous_active = Vec::new();
    let mut dense_carry = None;
    let mut sparse_carry = None;

    for (index, interval) in intervals.iter().copied().enumerate() {
        let depth = index + 1;
        let mut dense_next = Vec::with_capacity(width);
        let (_, next_dense_carry) = kernel.step_column(
            &dense_previous,
            query,
            interval,
            dense_carry,
            depth,
            &plan,
            &mut dense_next,
        );
        let mut sparse_next = vec![K::Monoid::TOP; width];
        let mut sparse_next_active = Vec::with_capacity(width);
        let outcome = kernel
            .step_interval_frontier(
                &sparse_previous,
                &sparse_previous_active,
                query,
                interval,
                sparse_carry,
                depth,
                &plan,
                cutoff,
                usize::MAX,
                &mut sparse_next,
                &mut sparse_next_active,
            )
            .expect("built-in metric automata implement sparse interval transitions");
        let PointFrontierStep::Advanced {
            lower_bound,
            carry: next_sparse_carry,
            work,
        } = outcome
        else {
            panic!("an unlimited generated transition cannot exhaust work");
        };
        assert!(work <= width);
        assert!(sparse_next_active.windows(2).all(|pair| pair[0] < pair[1]));

        let mut expected_active = Vec::new();
        let mut expected_lower = K::Monoid::TOP;
        for (row, dense) in dense_next.iter().copied().enumerate() {
            let expected = if K::Monoid::within(dense, cutoff) {
                expected_active.push(row);
                expected_lower = K::Monoid::select(expected_lower, dense);
                dense
            } else {
                K::Monoid::TOP
            };
            assert_eq!(
                K::Monoid::compare(sparse_next[row], expected),
                std::cmp::Ordering::Equal,
                "row {row} differs at target depth {depth}"
            );
        }
        assert_eq!(sparse_next_active, expected_active);
        assert_eq!(
            K::Monoid::compare(lower_bound, expected_lower),
            std::cmp::Ordering::Equal
        );

        dense_previous = dense_next;
        sparse_previous = sparse_next;
        sparse_previous_active = sparse_next_active;
        dense_carry = Some(next_dense_carry);
        sparse_carry = Some(next_sparse_carry);
    }
}

fn finite_intervals(raw: &[(i8, u8)]) -> Vec<(f64, f64)> {
    raw.iter()
        .map(|(low, width)| {
            let low = f64::from(*low);
            (low, low + f64::from(*width % 5))
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn erp_sparse_interval_frontier_equals_cutoff_pruned_dense_column(
        query in prop::collection::vec(-12i8..=12, 0..16),
        target in prop::collection::vec((-12i8..=12, 0u8..=8), 0..20),
        gap in -4i8..=4,
        cutoff in 0u16..=160,
    ) {
        let query: Vec<_> = query.into_iter().map(f64::from).collect();
        assert_sparse_matches_dense(
            ErpConfig::new(f64::from(gap)),
            &query,
            &finite_intervals(&target),
            f64::from(cutoff),
        );
    }

    #[test]
    fn msm_sparse_interval_frontier_equals_cutoff_pruned_dense_column(
        query in prop::collection::vec(-12i8..=12, 1..16),
        target in prop::collection::vec((-12i8..=12, 0u8..=8), 0..20),
        split_merge in 0u8..=8,
        cutoff in 0u16..=160,
    ) {
        let query: Vec<_> = query.into_iter().map(f64::from).collect();
        assert_sparse_matches_dense(
            MsmKernel::new(MsmConfig::new(f64::from(split_merge))),
            &query,
            &finite_intervals(&target),
            f64::from(cutoff),
        );
    }

    #[test]
    fn twed_sparse_interval_frontier_equals_cutoff_pruned_dense_column(
        query in prop::collection::vec(-10i8..=10, 0..16),
        target in prop::collection::vec((-10i8..=10, 0u8..=8), 0..20),
        cutoff in 0u16..=220,
    ) {
        let query: Vec<_> = query.into_iter().map(f64::from).collect();
        assert_sparse_matches_dense(
            TwedConfig::new(0.5, 1.0),
            &query,
            &finite_intervals(&target),
            f64::from(cutoff),
        );
    }

    #[test]
    fn frechet_sparse_interval_frontier_equals_cutoff_pruned_dense_column(
        query in prop::collection::vec(-10i8..=10, 1..16),
        target in prop::collection::vec((-10i8..=10, 0u8..=8), 0..20),
        cutoff in 0u16..=80,
    ) {
        let query: Vec<_> = query.into_iter().map(f64::from).collect();
        assert_sparse_matches_dense(
            FrechetConfig::new(),
            &query,
            &finite_intervals(&target),
            f64::from(cutoff),
        );
    }
}
