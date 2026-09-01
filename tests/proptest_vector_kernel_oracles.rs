//! Independent full-matrix correspondence oracles for vector temporal kernels.
//!
//! These tests deliberately do not call the production rolling-row helpers.
//! Each oracle materializes the complete mathematical matrix and computes the
//! fixed-channel point cost directly from the public immutable configuration.

use liblevenshtein::time_series::{
    ChannelIdentity, ExactDecision, FixedChannelMetric, FoldLocalScaleProvenance, MetricChannel,
    OperationOutcome, ResourceLimits, TimestampUnit, TimestampedVectorSeries,
    VectorBandedDtwScorer, VectorErpMetric, VectorErpSeries, VectorFrechetMetric,
    VectorFrechetPath, VectorSample, VectorTimestampedTwedMetric,
};
use proptest::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
enum OracleDecision {
    Within(f64),
    Above,
    NoFiniteAlignment,
}

fn metric_with(scales: [f64; 2], weights: [f64; 2]) -> FixedChannelMetric {
    FixedChannelMetric::try_new(
        vec![
            MetricChannel::try_new(
                ChannelIdentity::try_new("position-x", "metre").unwrap(),
                scales[0],
                weights[0],
            )
            .unwrap(),
            MetricChannel::try_new(
                ChannelIdentity::try_new("position-y", "metre").unwrap(),
                scales[1],
                weights[1],
            )
            .unwrap(),
        ],
        FoldLocalScaleProvenance::try_new("training-fold-0", "oracle-test-v1").unwrap(),
    )
    .unwrap()
}

fn metric() -> FixedChannelMetric {
    metric_with([2.0, 3.0], [5.0, 7.0])
}

fn sample(coordinates: [i8; 2]) -> VectorSample {
    VectorSample::try_new(
        &[f64::from(coordinates[0]), f64::from(coordinates[1])],
        ResourceLimits::default(),
    )
    .unwrap()
}

fn samples(rows: Vec<[i8; 2]>) -> Vec<VectorSample> {
    rows.into_iter().map(sample).collect()
}

fn oracle_point_distance(
    ground: &FixedChannelMetric,
    left: &VectorSample,
    right: &VectorSample,
) -> f64 {
    ground
        .channels()
        .iter()
        .zip(left.coordinates().iter().zip(right.coordinates()))
        .fold(0.0, |total, (channel, (left, right))| {
            total + channel.weight() * (left - right).abs() / channel.scale()
        })
}

fn oracle_erp(
    ground: &FixedChannelMetric,
    gap: &VectorSample,
    left: &VectorErpSeries,
    right: &VectorErpSeries,
) -> f64 {
    let rows = left.samples().len() + 1;
    let columns = right.samples().len() + 1;
    let mut matrix = vec![f64::INFINITY; rows * columns];
    let cell = |row: usize, column: usize| row * columns + column;
    matrix[cell(0, 0)] = 0.0;

    for row in 1..rows {
        matrix[cell(row, 0)] =
            matrix[cell(row - 1, 0)] + oracle_point_distance(ground, &left.samples()[row - 1], gap);
    }
    for column in 1..columns {
        matrix[cell(0, column)] = matrix[cell(0, column - 1)]
            + oracle_point_distance(ground, &right.samples()[column - 1], gap);
    }
    for row in 1..rows {
        for column in 1..columns {
            let substitute = matrix[cell(row - 1, column - 1)]
                + oracle_point_distance(
                    ground,
                    &left.samples()[row - 1],
                    &right.samples()[column - 1],
                );
            let delete = matrix[cell(row - 1, column)]
                + oracle_point_distance(ground, &left.samples()[row - 1], gap);
            let insert = matrix[cell(row, column - 1)]
                + oracle_point_distance(ground, &right.samples()[column - 1], gap);
            matrix[cell(row, column)] = substitute.min(delete).min(insert);
        }
    }
    matrix[cell(rows - 1, columns - 1)]
}

fn twed_delete_cost(
    ground: &FixedChannelMetric,
    current: &VectorSample,
    previous: &VectorSample,
    current_time: f64,
    previous_time: f64,
    stiffness: f64,
    gap_penalty: f64,
) -> f64 {
    oracle_point_distance(ground, current, previous)
        + stiffness * (current_time - previous_time)
        + gap_penalty
}

#[allow(clippy::too_many_arguments)]
fn twed_match_cost(
    ground: &FixedChannelMetric,
    left: &VectorSample,
    left_previous: &VectorSample,
    left_time: f64,
    left_previous_time: f64,
    right: &VectorSample,
    right_previous: &VectorSample,
    right_time: f64,
    right_previous_time: f64,
    stiffness: f64,
) -> f64 {
    oracle_point_distance(ground, left, right)
        + oracle_point_distance(ground, left_previous, right_previous)
        + stiffness
            * ((left_time - right_time).abs() + (left_previous_time - right_previous_time).abs())
}

fn oracle_timestamped_twed(
    ground: &FixedChannelMetric,
    sentinel: &VectorSample,
    stiffness: f64,
    gap_penalty: f64,
    left: &TimestampedVectorSeries,
    right: &TimestampedVectorSeries,
) -> f64 {
    let rows = left.samples().len() + 1;
    let columns = right.samples().len() + 1;
    let mut matrix = vec![f64::INFINITY; rows * columns];
    let cell = |row: usize, column: usize| row * columns + column;
    matrix[cell(0, 0)] = 0.0;

    for row in 1..rows {
        let previous = if row == 1 {
            sentinel
        } else {
            &left.samples()[row - 2]
        };
        let previous_time = if row == 1 {
            left.origin()
        } else {
            left.timestamps()[row - 2]
        };
        matrix[cell(row, 0)] = matrix[cell(row - 1, 0)]
            + twed_delete_cost(
                ground,
                &left.samples()[row - 1],
                previous,
                left.timestamps()[row - 1],
                previous_time,
                stiffness,
                gap_penalty,
            );
    }
    for column in 1..columns {
        let previous = if column == 1 {
            sentinel
        } else {
            &right.samples()[column - 2]
        };
        let previous_time = if column == 1 {
            right.origin()
        } else {
            right.timestamps()[column - 2]
        };
        matrix[cell(0, column)] = matrix[cell(0, column - 1)]
            + twed_delete_cost(
                ground,
                &right.samples()[column - 1],
                previous,
                right.timestamps()[column - 1],
                previous_time,
                stiffness,
                gap_penalty,
            );
    }

    for row in 1..rows {
        let left_previous = if row == 1 {
            sentinel
        } else {
            &left.samples()[row - 2]
        };
        let left_previous_time = if row == 1 {
            left.origin()
        } else {
            left.timestamps()[row - 2]
        };
        let delete_left = twed_delete_cost(
            ground,
            &left.samples()[row - 1],
            left_previous,
            left.timestamps()[row - 1],
            left_previous_time,
            stiffness,
            gap_penalty,
        );
        for column in 1..columns {
            let right_previous = if column == 1 {
                sentinel
            } else {
                &right.samples()[column - 2]
            };
            let right_previous_time = if column == 1 {
                right.origin()
            } else {
                right.timestamps()[column - 2]
            };
            let match_pair = matrix[cell(row - 1, column - 1)]
                + twed_match_cost(
                    ground,
                    &left.samples()[row - 1],
                    left_previous,
                    left.timestamps()[row - 1],
                    left_previous_time,
                    &right.samples()[column - 1],
                    right_previous,
                    right.timestamps()[column - 1],
                    right_previous_time,
                    stiffness,
                );
            let delete_left = matrix[cell(row - 1, column)] + delete_left;
            let delete_right = matrix[cell(row, column - 1)]
                + twed_delete_cost(
                    ground,
                    &right.samples()[column - 1],
                    right_previous,
                    right.timestamps()[column - 1],
                    right_previous_time,
                    stiffness,
                    gap_penalty,
                );
            matrix[cell(row, column)] = match_pair.min(delete_left).min(delete_right);
        }
    }
    matrix[cell(rows - 1, columns - 1)]
}

fn oracle_frechet(
    ground: &FixedChannelMetric,
    left: &VectorFrechetPath,
    right: &VectorFrechetPath,
) -> f64 {
    let rows = left.samples().len();
    let columns = right.samples().len();
    let mut matrix = vec![f64::INFINITY; rows * columns];
    let cell = |row: usize, column: usize| row * columns + column;

    for row in 0..rows {
        for column in 0..columns {
            let local =
                oracle_point_distance(ground, &left.samples()[row], &right.samples()[column]);
            matrix[cell(row, column)] = match (row, column) {
                (0, 0) => local,
                (0, _) => matrix[cell(row, column - 1)].max(local),
                (_, 0) => matrix[cell(row - 1, column)].max(local),
                _ => matrix[cell(row - 1, column - 1)]
                    .min(matrix[cell(row - 1, column)])
                    .min(matrix[cell(row, column - 1)])
                    .max(local),
            };
        }
    }
    matrix[cell(rows - 1, columns - 1)]
}

fn oracle_banded_dtw(
    ground: &FixedChannelMetric,
    band: usize,
    left: &[VectorSample],
    right: &[VectorSample],
) -> Option<f64> {
    match (left.is_empty(), right.is_empty()) {
        (true, true) => return Some(0.0),
        (true, false) | (false, true) => return None,
        (false, false) => {}
    }
    if left.len().abs_diff(right.len()) > band {
        return None;
    }

    let rows = left.len() + 1;
    let columns = right.len() + 1;
    let mut matrix = vec![f64::INFINITY; rows * columns];
    let cell = |row: usize, column: usize| row * columns + column;
    matrix[cell(0, 0)] = 0.0;
    for row in 1..rows {
        let start = row.saturating_sub(band).max(1);
        let end = row.saturating_add(band).min(right.len());
        for column in start..=end {
            let distance = oracle_point_distance(ground, &left[row - 1], &right[column - 1]);
            let predecessor = matrix[cell(row - 1, column - 1)]
                .min(matrix[cell(row - 1, column)])
                .min(matrix[cell(row, column - 1)]);
            matrix[cell(row, column)] = predecessor + distance * distance;
        }
    }
    matrix[cell(rows - 1, columns - 1)]
        .is_finite()
        .then_some(matrix[cell(rows - 1, columns - 1)].sqrt())
}

fn observe(outcome: OperationOutcome<ExactDecision>) -> OracleDecision {
    match outcome {
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, .. },
            ..
        } => OracleDecision::Within(distance),
        OperationOutcome::Complete {
            value: ExactDecision::AboveCutoff,
            ..
        } => OracleDecision::Above,
        OperationOutcome::Complete {
            value: ExactDecision::NoFiniteAlignment,
            ..
        } => OracleDecision::NoFiniteAlignment,
        OperationOutcome::Incomplete { reason, .. } => {
            panic!("small finite oracle case unexpectedly incomplete: {reason:?}")
        }
    }
}

fn close(left: f64, right: f64) -> bool {
    left.to_bits() == right.to_bits()
        || (left - right).abs() <= 1.0e-10 * (1.0 + left.abs().max(right.abs()))
}

fn previous_nonnegative_float(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

fn expected_cutoff(exact: f64, cutoff: f64) -> OracleDecision {
    if exact <= cutoff {
        OracleDecision::Within(exact)
    } else {
        OracleDecision::Above
    }
}

fn decisions_agree(observed: OracleDecision, expected: OracleDecision) -> bool {
    match (observed, expected) {
        (OracleDecision::Within(observed), OracleDecision::Within(expected)) => {
            close(observed, expected)
        }
        (OracleDecision::Above, OracleDecision::Above)
        | (OracleDecision::NoFiniteAlignment, OracleDecision::NoFiniteAlignment) => true,
        _ => false,
    }
}

fn cumulative_timestamps(deltas: &[u8]) -> Vec<f64> {
    let mut elapsed = 0.0;
    deltas
        .iter()
        .map(|delta| {
            elapsed += f64::from(*delta);
            elapsed
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn vector_erp_matches_independent_full_matrix(
        left in prop::collection::vec(any::<[i8; 2]>(), 0..7),
        right in prop::collection::vec(any::<[i8; 2]>(), 0..7),
    ) {
        let ground = metric();
        let erp = VectorErpMetric::try_new(ground.clone(), sample([0, 0])).unwrap();
        let left = erp.try_series(samples(left), ResourceLimits::default()).unwrap();
        let right = erp.try_series(samples(right), ResourceLimits::default()).unwrap();
        let exact = oracle_erp(&ground, erp.gap_sample(), &left, &right);

        for cutoff in [f64::INFINITY, exact, previous_nonnegative_float(exact)] {
            let observed = observe(
                erp.distance_bounded(&left, &right, cutoff, ResourceLimits::default())
                    .unwrap(),
            );
            prop_assert!(decisions_agree(observed, expected_cutoff(exact, cutoff)));
        }
    }

    #[test]
    fn vector_explicit_timestamp_twed_matches_independent_full_matrix(
        left in prop::collection::vec(any::<[i8; 2]>(), 1..7),
        right in prop::collection::vec(any::<[i8; 2]>(), 1..7),
        left_deltas in prop::collection::vec(1u8..=8, 1..7),
        right_deltas in prop::collection::vec(1u8..=8, 1..7),
        stiffness in 1u8..=8,
        gap_penalty in 0u8..=8,
    ) {
        let left_len = left.len().min(left_deltas.len());
        let right_len = right.len().min(right_deltas.len());
        let left = &left[..left_len];
        let right = &right[..right_len];
        let left_times = cumulative_timestamps(&left_deltas[..left_len]);
        let right_times = cumulative_timestamps(&right_deltas[..right_len]);
        let ground = metric();
        let sentinel = sample([0, 0]);
        let twed = VectorTimestampedTwedMetric::try_new(
            ground.clone(),
            sentinel.clone(),
            f64::from(stiffness),
            f64::from(gap_penalty),
        ).unwrap();
        let left = twed.try_series(
            samples(left.to_vec()),
            &left_times,
            TimestampUnit::Seconds,
            0.0,
            ResourceLimits::default(),
        ).unwrap();
        let right = twed.try_series(
            samples(right.to_vec()),
            &right_times,
            TimestampUnit::Seconds,
            0.0,
            ResourceLimits::default(),
        ).unwrap();
        let exact = oracle_timestamped_twed(
            &ground,
            &sentinel,
            twed.stiffness(),
            twed.gap_penalty(),
            &left,
            &right,
        );

        for cutoff in [f64::INFINITY, exact, previous_nonnegative_float(exact)] {
            let observed = observe(
                twed.distance_bounded(&left, &right, cutoff, ResourceLimits::default())
                    .unwrap(),
            );
            prop_assert!(decisions_agree(observed, expected_cutoff(exact, cutoff)));
        }
    }

    #[test]
    fn vector_discrete_frechet_matches_independent_full_matrix(
        left in prop::collection::vec(any::<[i8; 2]>(), 1..8),
        right in prop::collection::vec(any::<[i8; 2]>(), 1..8),
    ) {
        let ground = metric();
        let frechet = VectorFrechetMetric::new(ground.clone());
        let left = VectorFrechetPath::try_new(samples(left), ResourceLimits::default()).unwrap();
        let right = VectorFrechetPath::try_new(samples(right), ResourceLimits::default()).unwrap();
        let exact = oracle_frechet(&ground, &left, &right);

        for cutoff in [f64::INFINITY, exact, previous_nonnegative_float(exact)] {
            let observed = observe(
                frechet.distance_bounded(&left, &right, cutoff, ResourceLimits::default())
                    .unwrap(),
            );
            prop_assert!(decisions_agree(observed, expected_cutoff(exact, cutoff)));
        }
    }

    #[test]
    fn vector_banded_dtw_matches_independent_full_matrix(
        left in prop::collection::vec(any::<[i8; 2]>(), 0..8),
        right in prop::collection::vec(any::<[i8; 2]>(), 0..8),
        band in 0usize..8,
    ) {
        let ground = metric();
        let dtw = VectorBandedDtwScorer::new(ground.clone(), band);
        let left = dtw.try_series(samples(left), ResourceLimits::default()).unwrap();
        let right = dtw.try_series(samples(right), ResourceLimits::default()).unwrap();
        let exact = oracle_banded_dtw(&ground, band, left.samples(), right.samples());

        let cutoffs = match exact {
            Some(exact) => [f64::INFINITY, exact, previous_nonnegative_float(exact)],
            None => [f64::INFINITY; 3],
        };
        for cutoff in cutoffs {
            let observed = observe(
                dtw.distance_bounded(&left, &right, cutoff, ResourceLimits::default())
                    .unwrap(),
            );
            let expected = exact
                .map(|exact| expected_cutoff(exact, cutoff))
                .unwrap_or(OracleDecision::NoFiniteAlignment);
            prop_assert!(
                decisions_agree(observed, expected),
                "cutoff={cutoff:?}, exact={exact:?}, observed={observed:?}, expected={expected:?}"
            );
        }
    }
}

#[test]
fn vector_dtw_public_score_cutoff_is_exactly_inclusive_after_square_root_rounding() {
    let ground = metric();
    let dtw = VectorBandedDtwScorer::new(ground.clone(), 1);
    let left = dtw
        .try_series(
            samples(vec![[95, 79], [82, 1], [-65, 95]]),
            ResourceLimits::default(),
        )
        .unwrap();
    let right = dtw
        .try_series(
            samples(vec![[120, 16], [105, -94]]),
            ResourceLimits::default(),
        )
        .unwrap();
    let exact = oracle_banded_dtw(&ground, dtw.band(), left.samples(), right.samples()).unwrap();

    assert_eq!(
        observe(
            dtw.distance_bounded(&left, &right, exact, ResourceLimits::default())
                .unwrap(),
        ),
        OracleDecision::Within(exact),
    );
    assert_eq!(
        observe(
            dtw.distance_bounded(
                &left,
                &right,
                previous_nonnegative_float(exact),
                ResourceLimits::default(),
            )
            .unwrap(),
        ),
        OracleDecision::Above,
    );
}

fn fit_training_fold(training: &[[i8; 2]]) -> FixedChannelMetric {
    let scales = std::array::from_fn(|coordinate| {
        1.0 + training
            .iter()
            .map(|sample| f64::from(sample[coordinate]).abs())
            .fold(0.0, f64::max)
    });
    metric_with(scales, [2.0, 3.0])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn held_out_mutation_cannot_change_owned_fold_local_configuration_or_scores(
        training in prop::collection::vec(any::<[i8; 2]>(), 1..12),
        held_out in prop::collection::vec(any::<[i8; 2]>(), 0..12),
        mutation in prop::collection::vec(any::<[i8; 2]>(), 0..12),
        probe_left in any::<[i8; 2]>(),
        probe_right in any::<[i8; 2]>(),
    ) {
        // Fitting belongs outside this crate. This test models the API boundary:
        // construction receives only training-fold-derived values, while
        // arbitrary held-out storage remains causally disconnected.
        let before = fit_training_fold(&training);
        let before_channels: Vec<_> = before.channels().iter()
            .map(|channel| (channel.identity().clone(), channel.scale(), channel.weight()))
            .collect();
        let before_provenance = before.scale_provenance().clone();
        let left = sample(probe_left);
        let right = sample(probe_right);
        let before_score = before.distance_checked(&left, &right).unwrap();

        let mut mutated_held_out = held_out;
        let mutation_len = mutation.len();
        mutated_held_out.clear();
        mutated_held_out.extend(mutation);
        let after = fit_training_fold(&training);
        let after_channels: Vec<_> = after.channels().iter()
            .map(|channel| (channel.identity().clone(), channel.scale(), channel.weight()))
            .collect();
        let after_score = after.distance_checked(&left, &right).unwrap();

        prop_assert_eq!(&before, &after);
        prop_assert_eq!(&before_channels, &after_channels);
        prop_assert_eq!(&before_provenance, after.scale_provenance());
        prop_assert_eq!(before_score.to_bits(), after_score.to_bits());
        prop_assert_eq!(mutated_held_out.len(), mutation_len);
    }
}
