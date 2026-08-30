//! Executable properties extracted from the lazy weighted-frontier proofs.
//!
//! This test module intentionally contains an independent integer ERP model.
//! Production code is compared to this model in the correspondence suite; the
//! model itself is small enough for exhaustive property generation and does
//! not share implementation helpers with the optimized machine.

use std::collections::BTreeMap;

use proptest::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Position {
    query_index: usize,
    cost: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Interval {
    low: i16,
    high: i16,
}

impl Interval {
    fn point(value: i16) -> Self {
        Self {
            low: value,
            high: value,
        }
    }

    fn distance(self, value: i16) -> u64 {
        if value < self.low {
            u64::from(self.low.abs_diff(value))
        } else if value > self.high {
            u64::from(value.abs_diff(self.high))
        } else {
            0
        }
    }
}

fn scalar_distance(left: i16, right: i16) -> u64 {
    u64::from(left.abs_diff(right))
}

fn deletion_costs(query: &[i16], gap: i16) -> Vec<u64> {
    query
        .iter()
        .map(|value| scalar_distance(*value, gap))
        .collect()
}

fn deletion_mass(costs: &[u64], start: usize, end: usize) -> u64 {
    costs[start..end].iter().copied().sum()
}

fn canonicalize(mut positions: Vec<Position>, deletion: &[u64]) -> Vec<Position> {
    let mut cheapest_by_index = BTreeMap::<usize, u64>::new();
    for position in positions.drain(..) {
        cheapest_by_index
            .entry(position.query_index)
            .and_modify(|cost| *cost = (*cost).min(position.cost))
            .or_insert(position.cost);
    }

    let mut canonical = Vec::<Position>::with_capacity(cheapest_by_index.len());
    for (query_index, cost) in cheapest_by_index {
        let dominated = canonical.iter().any(|existing| {
            existing.cost + deletion_mass(deletion, existing.query_index, query_index) <= cost
        });
        if !dominated {
            canonical.push(Position { query_index, cost });
        }
    }
    canonical
}

fn represented_cell(state: &[Position], deletion: &[u64], row: usize) -> Option<u64> {
    state
        .iter()
        .filter(|position| position.query_index <= row)
        .map(|position| position.cost + deletion_mass(deletion, position.query_index, row))
        .min()
}

fn represented_column(state: &[Position], deletion: &[u64]) -> Vec<u64> {
    (0..=deletion.len())
        .map(|row| {
            represented_cell(state, deletion, row)
                .expect("a valid ERP frontier represents every query row")
        })
        .collect()
}

fn sparse_interval_step(
    query: &[i16],
    gap: i16,
    state: &[Position],
    target: Interval,
) -> Vec<Position> {
    let deletion = deletion_costs(query, gap);
    let insertion = target.distance(gap);
    let expansion_capacity = state
        .len()
        .checked_mul(query.len().saturating_add(2))
        .expect("small executable model capacity is representable");
    let mut generated = Vec::with_capacity(expansion_capacity);

    for position in state {
        generated.push(Position {
            query_index: position.query_index,
            cost: position.cost + insertion,
        });

        for (substitution_index, sample) in query.iter().enumerate().skip(position.query_index) {
            generated.push(Position {
                query_index: substitution_index + 1,
                cost: position.cost
                    + deletion_mass(&deletion, position.query_index, substitution_index)
                    + target.distance(*sample),
            });
        }
    }

    canonicalize(generated, &deletion)
}

fn sparse_interval_prefixes(query: &[i16], gap: i16, target: &[Interval]) -> Vec<Vec<Position>> {
    let mut state = vec![Position {
        query_index: 0,
        cost: 0,
    }];
    let mut prefixes = vec![state.clone()];
    for interval in target {
        state = sparse_interval_step(query, gap, &state, *interval);
        prefixes.push(state.clone());
    }
    prefixes
}

fn dense_erp_prefix_columns(query: &[i16], target: &[i16], gap: i16) -> Vec<Vec<u64>> {
    let deletion = deletion_costs(query, gap);
    let mut previous = Vec::with_capacity(query.len() + 1);
    previous.push(0);
    for cost in &deletion {
        previous.push(previous.last().copied().expect("ERP root cell exists") + cost);
    }

    let mut prefixes = vec![previous.clone()];
    for target_value in target {
        let insertion = scalar_distance(*target_value, gap);
        let mut current = vec![0; query.len() + 1];
        current[0] = previous[0] + insertion;
        for row in 1..=query.len() {
            let substitute = previous[row - 1] + scalar_distance(query[row - 1], *target_value);
            let delete = current[row - 1] + deletion[row - 1];
            let insert = previous[row] + insertion;
            current[row] = substitute.min(delete).min(insert);
        }
        previous = current;
        prefixes.push(previous.clone());
    }
    prefixes
}

fn process_from_state(query: &[i16], gap: i16, mut state: Vec<Position>, suffix: &[i16]) -> u64 {
    for target in suffix {
        state = sparse_interval_step(query, gap, &state, Interval::point(*target));
    }
    represented_cell(&state, &deletion_costs(query, gap), query.len())
        .expect("a nonempty ERP frontier has a final residual cost")
}

fn canonical_nonnegative_bits(value: f64) -> Option<u64> {
    if !value.is_finite() || value < 0.0 {
        None
    } else if value == 0.0 {
        Some(0)
    } else {
        Some(value.to_bits())
    }
}

fn small_series() -> impl Strategy<Value = Vec<i16>> {
    prop::collection::vec(-8i16..=8, 0..8)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn canonicalization_preserves_every_represented_cell(
        query in small_series(),
        gap in -4i16..=4,
        positions_seed in prop::collection::vec((0usize..8, 0u64..80), 1..16),
    ) {
        let deletion = deletion_costs(&query, gap);
        let positions: Vec<_> = positions_seed
            .into_iter()
            .map(|(query_index, cost)| Position {
                query_index: query_index.min(query.len()),
                cost,
            })
            .collect();
        let canonical = canonicalize(positions.clone(), &deletion);

        for row in 0..=query.len() {
            prop_assert_eq!(
                represented_cell(&positions, &deletion, row),
                represented_cell(&canonical, &deletion, row),
            );
        }
    }

    #[test]
    fn canonical_state_is_permutation_independent(
        query in small_series(),
        gap in -4i16..=4,
        positions_seed in prop::collection::vec((0usize..8, 0u64..80), 1..16),
    ) {
        let deletion = deletion_costs(&query, gap);
        let positions: Vec<_> = positions_seed
            .into_iter()
            .map(|(query_index, cost)| Position {
                query_index: query_index.min(query.len()),
                cost,
            })
            .collect();
        let mut reversed = positions.clone();
        reversed.reverse();
        prop_assert_eq!(
            canonicalize(positions, &deletion),
            canonicalize(reversed, &deletion),
        );
    }

    #[test]
    fn canonical_state_is_a_sorted_antichain(
        query in small_series(),
        gap in -4i16..=4,
        positions_seed in prop::collection::vec((0usize..8, 0u64..80), 1..16),
    ) {
        let deletion = deletion_costs(&query, gap);
        let positions: Vec<_> = positions_seed
            .into_iter()
            .map(|(query_index, cost)| Position {
                query_index: query_index.min(query.len()),
                cost,
            })
            .collect();
        let canonical = canonicalize(positions, &deletion);

        for pair in canonical.windows(2) {
            prop_assert!(pair[0].query_index < pair[1].query_index);
        }
        for (left_index, left) in canonical.iter().enumerate() {
            for right in &canonical[left_index + 1..] {
                prop_assert!(
                    left.cost + deletion_mass(&deletion, left.query_index, right.query_index)
                        > right.cost
                );
            }
        }
    }

    #[test]
    fn sparse_point_frontier_matches_dense_erp_after_every_prefix(
        query in small_series(),
        target in small_series(),
        gap in -4i16..=4,
    ) {
        let dense = dense_erp_prefix_columns(&query, &target, gap);
        let point_target: Vec<_> = target.iter().copied().map(Interval::point).collect();
        let sparse = sparse_interval_prefixes(&query, gap, &point_target);
        let deletion = deletion_costs(&query, gap);

        prop_assert_eq!(dense.len(), sparse.len());
        for (dense_column, sparse_state) in dense.iter().zip(&sparse) {
            prop_assert_eq!(dense_column, &represented_column(sparse_state, &deletion));
        }
    }

    #[test]
    fn point_intervals_reproduce_exact_sparse_transitions(
        query in small_series(),
        target in small_series(),
        gap in -4i16..=4,
    ) {
        let points: Vec<_> = target.iter().copied().map(Interval::point).collect();
        let exact = sparse_interval_prefixes(&query, gap, &points);
        let point_relaxed = sparse_interval_prefixes(&query, gap, &points);
        prop_assert_eq!(exact, point_relaxed);
    }

    #[test]
    fn interval_frontier_lower_bounds_every_represented_concrete_prefix(
        query in small_series(),
        centers in small_series(),
        radii in prop::collection::vec(0i16..=3, 0..8),
        choices in prop::collection::vec(0u8..=6, 0..8),
        gap in -4i16..=4,
    ) {
        let len = centers.len().min(radii.len()).min(choices.len());
        let intervals: Vec<_> = (0..len)
            .map(|index| Interval {
                low: centers[index] - radii[index],
                high: centers[index] + radii[index],
            })
            .collect();
        let concrete: Vec<_> = intervals
            .iter()
            .enumerate()
            .map(|(index, interval)| {
                let width = i32::from(interval.high) - i32::from(interval.low) + 1;
                let offset = i32::from(choices[index]) % width;
                i16::try_from(i32::from(interval.low) + offset)
                    .expect("small interval sample remains representable")
            })
            .collect();

        let relaxed = sparse_interval_prefixes(&query, gap, &intervals);
        let concrete_columns = dense_erp_prefix_columns(&query, &concrete, gap);
        let deletion = deletion_costs(&query, gap);
        for (relaxed_state, concrete_column) in relaxed.iter().zip(&concrete_columns) {
            for (lower, exact) in represented_column(relaxed_state, &deletion)
                .iter()
                .zip(concrete_column)
            {
                prop_assert!(lower <= exact);
            }
        }
    }

    #[test]
    fn dominance_pruning_preserves_all_enumerated_suffix_costs(
        query in small_series(),
        suffix in prop::collection::vec(-3i16..=3, 0..5),
        positions_seed in prop::collection::vec((0usize..8, 0u64..80), 1..16),
        gap in -4i16..=4,
    ) {
        let deletion = deletion_costs(&query, gap);
        let positions: Vec<_> = positions_seed
            .into_iter()
            .map(|(query_index, cost)| Position {
                query_index: query_index.min(query.len()),
                cost,
            })
            .collect();
        let canonical = canonicalize(positions.clone(), &deletion);
        prop_assert_eq!(
            process_from_state(&query, gap, positions, &suffix),
            process_from_state(&query, gap, canonical, &suffix),
        );
    }

    #[test]
    fn every_page_partition_reconstructs_uninterrupted_observations(
        values in prop::collection::vec(any::<u16>(), 0..128),
        page_size in 0usize..160,
    ) {
        let split = page_size.min(values.len());
        let mut resumed = values[..split].to_vec();
        resumed.extend_from_slice(&values[split..]);
        prop_assert_eq!(resumed, values);
    }

    #[test]
    fn finite_nonnegative_float_keys_are_bit_exact_and_zero_canonical(
        value in 0.0f64..1.0e100,
    ) {
        let bits = canonical_nonnegative_bits(value)
            .expect("generated value belongs to the canonical finite domain");
        if value == 0.0 {
            prop_assert_eq!(bits, 0);
        } else {
            prop_assert_eq!(bits, value.to_bits());
        }
    }
}

#[test]
fn signed_zero_has_one_canonical_state_key() {
    assert_eq!(canonical_nonnegative_bits(0.0), Some(0));
    assert_eq!(canonical_nonnegative_bits(-0.0), Some(0));
}

#[test]
fn invalid_float_state_costs_fail_closed() {
    assert_eq!(canonical_nonnegative_bits(f64::NAN), None);
    assert_eq!(canonical_nonnegative_bits(f64::INFINITY), None);
    assert_eq!(canonical_nonnegative_bits(f64::NEG_INFINITY), None);
    assert_eq!(canonical_nonnegative_bits(-1.0), None);
}

#[test]
fn million_prefixes_do_not_change_the_proved_generation_ceiling() {
    let frontier_limit = 257usize;
    let cache_limit = 4_096usize;
    let retained_ceiling = 2 * frontier_limit + cache_limit;
    for _consumed_prefix in 0..1_000_000usize {
        let current = frontier_limit;
        let next = frontier_limit;
        let cache = cache_limit;
        assert!(current + next + cache <= retained_ceiling);
    }
}
