use liblevenshtein::time_series::{
    ExactDecision, MetricTimestampedTwedConfig, OperationOutcome, PageBudget, ResourceKind,
    ResourceLimits, TimestampUnit, TimestampedSeries, TimestampedTwedIndex,
    TimestampedTwedProductLimits, TimestampedTwedQuantizer, TimestampedTwedRangeMatch,
    TimestampedTwedRangeOutcome,
};
use proptest::prelude::*;

fn series(values: &[f64], timestamps: &[f64]) -> TimestampedSeries {
    TimestampedSeries::try_new(
        values,
        timestamps,
        TimestampUnit::Seconds,
        ResourceLimits::default(),
    )
    .unwrap()
}

fn irregular(values: &[i8]) -> TimestampedSeries {
    let values_f64: Vec<f64> = values.iter().map(|value| f64::from(*value)).collect();
    let mut timestamps = Vec::with_capacity(values.len());
    let mut current = 0.0;
    for value in values {
        current += 0.5 + f64::from(value.unsigned_abs() % 4) * 0.125;
        timestamps.push(current);
    }
    series(&values_f64, &timestamps)
}

fn quantizer(value_bins: u32, time_bins: u32) -> TimestampedTwedQuantizer {
    TimestampedTwedQuantizer::try_new(
        TimestampUnit::Seconds,
        0.0,
        (-8.0, 8.0),
        (0.0, 16.0),
        value_bins,
        time_bins,
    )
    .unwrap()
}

fn config() -> MetricTimestampedTwedConfig {
    MetricTimestampedTwedConfig::try_new(0.75, 0.5).unwrap()
}

fn drain<'a>(
    mut outcome: TimestampedTwedRangeOutcome<'a, usize>,
    page: PageBudget,
) -> (Vec<TimestampedTwedRangeMatch<'a, usize>>, usize) {
    let mut pauses = 0;
    loop {
        match outcome {
            OperationOutcome::Complete { value, .. } => return (value, pauses),
            OperationOutcome::Incomplete {
                continuation: Some(continuation),
                ..
            } => {
                pauses += 1;
                assert_eq!(
                    continuation.exact_partial().len(),
                    continuation.usage().results
                );
                outcome = continuation.resume(page);
            }
            OperationOutcome::Incomplete {
                reason,
                continuation: None,
                ..
            } => panic!("unexpected terminal incomplete query: {reason:?}"),
        }
    }
}

fn brute_force_twed(
    config: MetricTimestampedTwedConfig,
    left: &TimestampedSeries,
    right: &TimestampedSeries,
) -> f64 {
    let rows = left.values().len() + 1;
    let columns = right.values().len() + 1;
    let mut matrix = vec![f64::INFINITY; rows * columns];
    matrix[0] = 0.0;

    for row in 1..rows {
        matrix[row * columns] = matrix[(row - 1) * columns]
            + deletion(
                left.values()[row - 1],
                if row == 1 {
                    0.0
                } else {
                    left.values()[row - 2]
                },
                left.timestamps()[row - 1],
                if row == 1 {
                    left.origin()
                } else {
                    left.timestamps()[row - 2]
                },
                config,
            );
    }
    for column in 1..columns {
        matrix[column] = matrix[column - 1]
            + deletion(
                right.values()[column - 1],
                if column == 1 {
                    0.0
                } else {
                    right.values()[column - 2]
                },
                right.timestamps()[column - 1],
                if column == 1 {
                    right.origin()
                } else {
                    right.timestamps()[column - 2]
                },
                config,
            );
    }

    for row in 1..rows {
        for column in 1..columns {
            let left_previous_value = if row == 1 {
                0.0
            } else {
                left.values()[row - 2]
            };
            let left_previous_time = if row == 1 {
                left.origin()
            } else {
                left.timestamps()[row - 2]
            };
            let right_previous_value = if column == 1 {
                0.0
            } else {
                right.values()[column - 2]
            };
            let right_previous_time = if column == 1 {
                right.origin()
            } else {
                right.timestamps()[column - 2]
            };
            let pair = matrix[(row - 1) * columns + column - 1]
                + (left.values()[row - 1] - right.values()[column - 1]).abs()
                + (left_previous_value - right_previous_value).abs()
                + config.stiffness()
                    * ((left.timestamps()[row - 1] - right.timestamps()[column - 1]).abs()
                        + (left_previous_time - right_previous_time).abs());
            let delete_left = matrix[(row - 1) * columns + column]
                + deletion(
                    left.values()[row - 1],
                    left_previous_value,
                    left.timestamps()[row - 1],
                    left_previous_time,
                    config,
                );
            let delete_right = matrix[row * columns + column - 1]
                + deletion(
                    right.values()[column - 1],
                    right_previous_value,
                    right.timestamps()[column - 1],
                    right_previous_time,
                    config,
                );
            matrix[row * columns + column] = pair.min(delete_left).min(delete_right);
        }
    }
    matrix[rows * columns - 1]
}

fn deletion(
    value: f64,
    previous_value: f64,
    time: f64,
    previous_time: f64,
    config: MetricTimestampedTwedConfig,
) -> f64 {
    (value - previous_value).abs()
        + config.stiffness() * (time - previous_time)
        + config.gap_penalty()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(192))]

    #[test]
    fn exact_product_equals_independent_full_matrix_oracle(
        query_values in prop::collection::vec(-6_i8..=6, 1..6),
        candidate_values in prop::collection::vec(
            prop::collection::vec(-6_i8..=6, 1..6),
            0..8,
        ),
        cutoff_quarters in 0_u8..80,
    ) {
        let query = irregular(&query_values);
        let candidates: Vec<_> = candidate_values.iter().map(|values| irregular(values)).collect();
        let cutoff = f64::from(cutoff_quarters) / 4.0;
        let mut index = TimestampedTwedIndex::new(quantizer(5, 7), config());
        for (id, candidate) in candidates.iter().cloned().enumerate() {
            prop_assert_eq!(index.insert(id, candidate).unwrap(), id as u64);
        }

        let outcome = index.search_range_bounded(
            &query,
            cutoff,
            TimestampedTwedProductLimits::default(),
            PageBudget::default(),
        ).unwrap();
        let (actual, _) = drain(outcome, PageBudget::default());
        let mut expected: Vec<_> = candidates.iter().enumerate().filter_map(|(id, candidate)| {
            let distance = brute_force_twed(config(), &query, candidate);
            (distance <= cutoff).then_some((id as u64, distance))
        }).collect();
        expected.sort_by(|left, right| left.1.total_cmp(&right.1).then_with(|| left.0.cmp(&right.0)));
        prop_assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            prop_assert_eq!(actual.episode_id, expected.0);
            prop_assert!((actual.distance - expected.1).abs() <= 1e-12);
        }
    }

    #[test]
    fn decoded_box_contains_every_encoded_point(
        value in -8.0_f64..=8.0,
        time in 0.0_f64..=16.0,
        value_bins in 1_u32..64,
        time_bins in 1_u32..64,
    ) {
        let quantizer = quantizer(value_bins, time_bins);
        let decoded = quantizer.decode(quantizer.encode(value, time).unwrap()).unwrap();
        prop_assert!(decoded.value_interval().0 <= value);
        prop_assert!(value <= decoded.value_interval().1);
        prop_assert!(decoded.time_interval().0 <= time);
        prop_assert!(time <= decoded.time_interval().1);
    }

    #[test]
    fn range_membership_is_monotone_in_cutoff(
        query_values in prop::collection::vec(-6_i8..=6, 1..6),
        candidate_values in prop::collection::vec(
            prop::collection::vec(-6_i8..=6, 1..6),
            0..8,
        ),
        first_cutoff_quarters in 0_u8..40,
        extra_cutoff_quarters in 0_u8..40,
    ) {
        let query = irregular(&query_values);
        let mut index = TimestampedTwedIndex::new(quantizer(5, 7), config());
        for (id, values) in candidate_values.iter().enumerate() {
            index.insert(id, irregular(values)).unwrap();
        }
        let first_cutoff = f64::from(first_cutoff_quarters) / 4.0;
        let second_cutoff = first_cutoff + f64::from(extra_cutoff_quarters) / 4.0;
        let first = index.search_range_bounded(
            &query,
            first_cutoff,
            TimestampedTwedProductLimits::default(),
            PageBudget::default(),
        ).unwrap();
        let second = index.search_range_bounded(
            &query,
            second_cutoff,
            TimestampedTwedProductLimits::default(),
            PageBudget::default(),
        ).unwrap();
        let (first, _) = drain(first, PageBudget::default());
        let (second, _) = drain(second, PageBudget::default());
        let second_ids: std::collections::BTreeSet<_> =
            second.iter().map(|entry| entry.episode_id).collect();
        prop_assert!(first
            .iter()
            .all(|entry| second_ids.contains(&entry.episode_id)));
    }

    #[test]
    fn bounded_knn_equals_independent_full_matrix_oracle(
        query_values in prop::collection::vec(-6_i8..=6, 1..6),
        candidate_values in prop::collection::vec(
            prop::collection::vec(-6_i8..=6, 1..7),
            0..10,
        ),
        requested_k in 0_usize..12,
    ) {
        let query = irregular(&query_values);
        let candidates: Vec<_> = candidate_values.iter().map(|values| irregular(values)).collect();
        let mut index = TimestampedTwedIndex::new(quantizer(5, 7), config());
        for (id, candidate) in candidates.iter().cloned().enumerate() {
            index.insert(id, candidate).unwrap();
        }
        let outcome = index
            .search_knn_bounded(&query, requested_k, ResourceLimits::default())
            .unwrap();
        let actual = match outcome {
            OperationOutcome::Complete { value, .. } => value,
            other => panic!("default limits must complete a small generated kNN: {other:?}"),
        };
        let mut expected: Vec<_> = candidates
            .iter()
            .enumerate()
            .map(|(id, candidate)| (id as u64, brute_force_twed(config(), &query, candidate)))
            .collect();
        expected.sort_by(|left, right| {
            left.1.total_cmp(&right.1).then_with(|| left.0.cmp(&right.0))
        });
        expected.truncate(requested_k);
        prop_assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            prop_assert_eq!(actual.episode_id, expected.0);
            prop_assert_eq!(actual.distance.to_bits(), expected.1.to_bits());
        }
    }

    #[test]
    fn arbitrary_paged_resume_is_bitwise_equal_to_uninterrupted(
        query_values in prop::collection::vec(-6_i8..=6, 1..5),
        candidate_values in prop::collection::vec(
            prop::collection::vec(-6_i8..=6, 1..5),
            0..8,
        ),
        cutoff_quarters in 0_u8..80,
        page_work in 32_usize..96,
        page_results in 1_usize..5,
    ) {
        let query = irregular(&query_values);
        let cutoff = f64::from(cutoff_quarters) / 4.0;
        let mut index = TimestampedTwedIndex::new(quantizer(5, 7), config());
        for (id, values) in candidate_values.iter().enumerate() {
            index.insert(id, irregular(values)).unwrap();
        }

        let uninterrupted = index.search_range_bounded(
            &query,
            cutoff,
            TimestampedTwedProductLimits::default(),
            PageBudget::default(),
        ).unwrap();
        let (uninterrupted, _) = drain(uninterrupted, PageBudget::default());

        let page = PageBudget {
            max_work_units: page_work,
            max_results: page_results,
        };
        let paged = index.search_range_bounded(
            &query,
            cutoff,
            TimestampedTwedProductLimits::default(),
            page,
        ).unwrap();
        let (paged, _) = drain(paged, page);

        prop_assert_eq!(
            paged.iter()
                .map(|entry| (entry.episode_id, entry.distance.to_bits()))
                .collect::<Vec<_>>(),
            uninterrupted.iter()
                .map(|entry| (entry.episode_id, entry.distance.to_bits()))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn every_quantization_collision_is_verified_at_full_precision() {
    let candidates = [
        series(&[0.0, 1.0], &[1.0, 2.0]),
        series(&[0.2, 1.1], &[1.1, 2.1]),
        series(&[4.0, 4.5], &[4.0, 5.0]),
    ];
    let query = candidates[0].clone();
    let mut index = TimestampedTwedIndex::new(quantizer(1, 1), config());
    for (id, candidate) in candidates.iter().cloned().enumerate() {
        index.insert(id, candidate).unwrap();
    }
    let outcome = index
        .search_range_bounded(
            &query,
            0.5,
            TimestampedTwedProductLimits::default(),
            PageBudget::default(),
        )
        .unwrap();
    let (actual, _) = drain(outcome, PageBudget::default());
    let expected: Vec<_> = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| brute_force_twed(config(), &query, candidate) <= 0.5)
        .map(|(id, _)| id as u64)
        .collect();
    assert_eq!(
        actual
            .iter()
            .map(|entry| entry.episode_id)
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(index.len(), 3);
}

#[test]
fn resumed_query_equals_uninterrupted_and_retains_root_revision() {
    let query = series(&[0.0, 1.5, 2.0], &[0.5, 1.7, 3.0]);
    let candidates = [
        series(&[0.0, 1.0, 2.0], &[0.5, 1.5, 3.0]),
        series(&[0.0, 2.0], &[0.6, 2.8]),
        series(&[5.0, 6.0], &[1.0, 2.0]),
    ];
    let mut index = TimestampedTwedIndex::new(quantizer(4, 4), config());
    for (id, candidate) in candidates.into_iter().enumerate() {
        index.insert(id, candidate).unwrap();
    }

    let full = index
        .search_range_bounded(
            &query,
            20.0,
            TimestampedTwedProductLimits::default(),
            PageBudget::default(),
        )
        .unwrap();
    let (full, _) = drain(full, PageBudget::default());

    let first = index
        .search_range_bounded(
            &query,
            20.0,
            TimestampedTwedProductLimits::default(),
            PageBudget {
                max_work_units: 4,
                max_results: 1,
            },
        )
        .unwrap();
    let continuation = match first {
        OperationOutcome::Incomplete {
            continuation: Some(continuation),
            ..
        } => continuation,
        other => panic!("expected a resumable first page, got {other:?}"),
    };
    let revision = continuation.captured_revision_identity();
    assert_eq!(continuation.captured_term_count(), 3);
    let resumed = continuation.resume(PageBudget {
        max_work_units: 20,
        max_results: 1,
    });
    if let OperationOutcome::Incomplete {
        continuation: Some(ref continuation),
        ..
    } = resumed
    {
        assert_eq!(continuation.captured_revision_identity(), revision);
        assert_eq!(continuation.captured_term_count(), 3);
    }
    let (resumed, pauses) = drain(
        resumed,
        PageBudget {
            max_work_units: 20,
            max_results: 1,
        },
    );
    assert!(pauses > 0);
    assert_eq!(
        resumed
            .iter()
            .map(|entry| (entry.episode_id, entry.distance.to_bits()))
            .collect::<Vec<_>>(),
        full.iter()
            .map(|entry| (entry.episode_id, entry.distance.to_bits()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn disabling_transition_cache_preserves_exact_results() {
    let query = series(&[0.0, 1.5, 2.0], &[0.5, 1.7, 3.0]);
    let candidates = [
        series(&[0.0, 1.0, 2.0], &[0.5, 1.5, 3.0]),
        series(&[0.0, 2.0], &[0.6, 2.8]),
        series(&[5.0, 6.0], &[1.0, 2.0]),
    ];
    let mut index = TimestampedTwedIndex::new(quantizer(4, 4), config());
    for (id, candidate) in candidates.into_iter().enumerate() {
        index.insert(id, candidate).unwrap();
    }
    let cached = index
        .search_range_bounded(
            &query,
            20.0,
            TimestampedTwedProductLimits::default(),
            PageBudget::default(),
        )
        .unwrap();
    let uncached = index
        .search_range_bounded(
            &query,
            20.0,
            TimestampedTwedProductLimits {
                max_transition_cache_entries: 0,
                ..TimestampedTwedProductLimits::default()
            },
            PageBudget::default(),
        )
        .unwrap();
    let (cached, _) = drain(cached, PageBudget::default());
    let (uncached, _) = drain(uncached, PageBudget::default());
    assert_eq!(
        cached
            .iter()
            .map(|entry| (entry.episode_id, entry.distance.to_bits()))
            .collect::<Vec<_>>(),
        uncached
            .iter()
            .map(|entry| (entry.episode_id, entry.distance.to_bits()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn broad_root_fanout_is_paged_not_retained_as_a_queue() {
    let mut index = TimestampedTwedIndex::new(quantizer(128, 1), config());
    for id in 0..128_usize {
        let value = -7.9375 + id as f64 * 0.125;
        index
            .insert(id, series(&[value], &[1.0]))
            .expect("distinct one-token episode");
    }
    let limits = TimestampedTwedProductLimits {
        resources: ResourceLimits {
            max_queue_entries: 2,
            max_results: 128,
            ..ResourceLimits::default()
        },
        ..TimestampedTwedProductLimits::default()
    };
    let query = series(&[0.0], &[1.0]);
    let outcome = index
        .search_range_bounded(&query, f64::INFINITY, limits, PageBudget::default())
        .unwrap();
    let (matches, _) = drain(outcome, PageBudget::default());
    assert_eq!(matches.len(), 128);
}

#[test]
fn interval_refinement_is_monotone_and_point_boxes_reproduce_local_costs() {
    let coarse_quantizer = quantizer(2, 2);
    let fine_quantizer = quantizer(32, 32);
    let previous_point = (0.75, 1.25);
    let current_point = (1.5, 2.75);
    let coarse_previous = coarse_quantizer
        .decode(
            coarse_quantizer
                .encode(previous_point.0, previous_point.1)
                .unwrap(),
        )
        .unwrap();
    let coarse_current = coarse_quantizer
        .decode(
            coarse_quantizer
                .encode(current_point.0, current_point.1)
                .unwrap(),
        )
        .unwrap();
    let fine_previous = fine_quantizer
        .decode(
            fine_quantizer
                .encode(previous_point.0, previous_point.1)
                .unwrap(),
        )
        .unwrap();
    let fine_current = fine_quantizer
        .decode(
            fine_quantizer
                .encode(current_point.0, current_point.1)
                .unwrap(),
        )
        .unwrap();
    assert!(fine_previous.refines(coarse_previous).unwrap());
    assert!(fine_current.refines(coarse_current).unwrap());
    let coarse = config()
        .interval_delete_lower_bound(coarse_current, coarse_previous)
        .unwrap();
    let fine = config()
        .interval_delete_lower_bound(fine_current, fine_previous)
        .unwrap();
    let exact = deletion(
        current_point.0,
        previous_point.0,
        current_point.1,
        previous_point.1,
        config(),
    );
    assert!(coarse <= fine + 1e-12);
    assert!(fine <= exact + 1e-12);
}

#[test]
fn emitted_distances_agree_with_the_independent_bounded_scalar_scorer() {
    let query = series(&[0.0, 1.0], &[1.0, 2.5]);
    let candidates = [
        series(&[0.0, 1.1], &[1.0, 2.4]),
        series(&[3.0, 4.0], &[1.2, 3.0]),
    ];
    let mut index = TimestampedTwedIndex::new(quantizer(4, 4), config());
    for (id, candidate) in candidates.iter().cloned().enumerate() {
        index.insert(id, candidate).unwrap();
    }
    let outcome = index
        .search_range_bounded(
            &query,
            f64::INFINITY,
            TimestampedTwedProductLimits::default(),
            PageBudget::default(),
        )
        .unwrap();
    let (actual, _) = drain(outcome, PageBudget::default());
    for entry in actual {
        let scalar = config()
            .distance_bounded(
                &query,
                entry.series,
                f64::INFINITY,
                ResourceLimits::default(),
            )
            .unwrap();
        let expected = match scalar {
            OperationOutcome::Complete {
                value: ExactDecision::WithinCutoff { distance, .. },
                ..
            } => distance,
            other => panic!("expected exact scalar result, got {other:?}"),
        };
        assert_eq!(entry.distance.to_bits(), expected.to_bits());
    }
}

#[test]
fn state_ceiling_and_zero_page_are_never_complete_empty() {
    let query = series(&[0.0, 1.0], &[1.0, 2.0]);
    let mut index = TimestampedTwedIndex::new(quantizer(2, 2), config());
    index.insert(0, query.clone()).unwrap();

    let zero_page = index
        .search_range_bounded(
            &query,
            0.0,
            TimestampedTwedProductLimits::default(),
            PageBudget {
                max_work_units: 0,
                max_results: 0,
            },
        )
        .unwrap();
    assert!(matches!(
        zero_page,
        OperationOutcome::Incomplete {
            continuation: Some(_),
            ..
        }
    ));

    let limited = index
        .search_range_bounded(
            &query,
            0.0,
            TimestampedTwedProductLimits {
                max_product_states: 1,
                ..TimestampedTwedProductLimits::default()
            },
            PageBudget::default(),
        )
        .unwrap();
    assert!(matches!(
        limited,
        OperationOutcome::Incomplete {
            reason: liblevenshtein::time_series::IncompleteReason::BudgetExceeded {
                resource: ResourceKind::QueueEntries,
                ..
            },
            continuation: None,
            ..
        }
    ));
}

#[test]
fn bounded_knn_resource_exhaustion_is_never_complete() {
    let query = series(&[0.0, 1.0], &[1.0, 2.0]);
    let mut index = TimestampedTwedIndex::new(quantizer(2, 2), config());
    index.insert(0, query.clone()).unwrap();
    let outcome = index
        .search_knn_bounded(
            &query,
            1,
            ResourceLimits {
                max_candidates: 0,
                ..ResourceLimits::default()
            },
        )
        .unwrap();
    assert!(matches!(
        outcome,
        OperationOutcome::Incomplete {
            reason: liblevenshtein::time_series::IncompleteReason::BudgetExceeded {
                resource: ResourceKind::Candidates,
                limit: 0,
                requested: 1,
            },
            continuation: None,
            ..
        }
    ));
}

#[test]
fn long_dictionary_path_is_stack_safe_on_a_small_thread_stack() {
    let handle = std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(|| {
            let length = 100_000;
            let values: Vec<f64> = (0..length).map(|index| f64::from(index % 7)).collect();
            let timestamps: Vec<f64> = (0..length).map(|index| index as f64 + 1.0).collect();
            let candidate = series(&values, &timestamps);
            let query = series(&[0.0], &[1.0]);
            let quantizer = TimestampedTwedQuantizer::try_new(
                TimestampUnit::Seconds,
                0.0,
                (-1.0, 8.0),
                (0.0, length as f64 + 1.0),
                16,
                4_096,
            )
            .unwrap();
            let mut index = TimestampedTwedIndex::new(quantizer, config());
            index.insert(7_usize, candidate).unwrap();
            let outcome = index
                .search_range_bounded(
                    &query,
                    f64::INFINITY,
                    TimestampedTwedProductLimits::default(),
                    PageBudget {
                        max_work_units: 1_000_000,
                        max_results: 2,
                    },
                )
                .unwrap();
            let (matches, _) = drain(outcome, PageBudget::default());
            assert_eq!(matches.len(), 1);
        })
        .unwrap();
    handle.join().unwrap();
}
