//! Lawful-domain and interval-refinement properties for vector kernels.

use liblevenshtein::time_series::{
    ChannelIdentity, ExactDecision, FixedChannelMetric, FoldLocalScaleProvenance, IncompleteReason,
    MetricChannel, OperationOutcome, ResourceKind, ResourceLimits, TimestampUnit,
    TimestampedVectorBox, VectorBandedDtwScorer, VectorBox, VectorErpMetric, VectorFrechetMetric,
    VectorFrechetPath, VectorMetricError, VectorMsmSupportDecision, VectorSample,
    VectorTimestampedTwedMetric, VECTOR_MSM_SUPPORT,
};
use proptest::prelude::*;

fn metric() -> FixedChannelMetric {
    FixedChannelMetric::try_new(
        vec![
            MetricChannel::try_new(
                ChannelIdentity::try_new("position-x", "metre").unwrap(),
                2.0,
                3.0,
            )
            .unwrap(),
            MetricChannel::try_new(
                ChannelIdentity::try_new("position-y", "metre").unwrap(),
                4.0,
                5.0,
            )
            .unwrap(),
        ],
        FoldLocalScaleProvenance::try_new("training-fold-0", "mad-v1").unwrap(),
    )
    .unwrap()
}

fn sample(coordinates: [i8; 2]) -> VectorSample {
    VectorSample::try_new(
        &[f64::from(coordinates[0]), f64::from(coordinates[1])],
        ResourceLimits::default(),
    )
    .unwrap()
}

fn completed_distance(outcome: OperationOutcome<ExactDecision>) -> f64 {
    match outcome {
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, .. },
            ..
        } => distance,
        other => panic!("expected complete finite distance, got {other:?}"),
    }
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9 * (1.0 + left.abs().max(right.abs()))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn fixed_positive_channel_sum_is_a_metric(
        x in any::<[i8; 2]>(),
        y in any::<[i8; 2]>(),
        z in any::<[i8; 2]>(),
    ) {
        let metric = metric();
        let (x, y, z) = (sample(x), sample(y), sample(z));
        let dxy = metric.distance_checked(&x, &y).unwrap();
        let dyx = metric.distance_checked(&y, &x).unwrap();
        let dxz = metric.distance_checked(&x, &z).unwrap();
        let dzy = metric.distance_checked(&z, &y).unwrap();
        prop_assert!(dxy >= 0.0);
        prop_assert!(close(dxy, dyx));
        prop_assert_eq!(dxy == 0.0, x == y);
        prop_assert!(dzy <= dxz + dxy + 1.0e-9);
    }

    #[test]
    fn vector_box_k1_and_refinement_hold_coordinatewise(
        query in any::<[i8; 2]>(),
        concrete in any::<[i8; 2]>(),
        coarse_radius in 0u8..=16,
        fine_radius in 0u8..=16,
    ) {
        let metric = metric();
        let query = sample(query);
        let concrete = sample(concrete);
        let coarse_radius = f64::from(coarse_radius.max(fine_radius));
        let fine_radius = f64::from(fine_radius.min(coarse_radius as u8));
        let coarse_bounds: Vec<_> = concrete.coordinates().iter()
            .map(|value| (*value - coarse_radius, *value + coarse_radius))
            .collect();
        let fine_bounds: Vec<_> = concrete.coordinates().iter()
            .map(|value| (*value - fine_radius, *value + fine_radius))
            .collect();
        let coarse = VectorBox::try_new(metric.channel_layout().clone(), &coarse_bounds).unwrap();
        let fine = VectorBox::try_new(metric.channel_layout().clone(), &fine_bounds).unwrap();
        let exact_box = VectorBox::from_sample(metric.channel_layout().clone(), &concrete).unwrap();
        prop_assert!(fine.refines(&coarse).unwrap());
        let coarse_lb = metric.point_box_lower_bound(&query, &coarse).unwrap();
        let fine_lb = metric.point_box_lower_bound(&query, &fine).unwrap();
        let exact = metric.distance_checked(&query, &concrete).unwrap();
        prop_assert!(coarse_lb <= fine_lb + 1.0e-12);
        prop_assert!(fine_lb <= exact + 1.0e-12);
        prop_assert!(close(
            metric.point_box_lower_bound(&query, &exact_box).unwrap(),
            exact,
        ));
    }

    #[test]
    fn vector_erp_metric_laws_hold_on_the_gap_quotient(
        x in prop::collection::vec(any::<[i8; 2]>(), 0..5),
        y in prop::collection::vec(any::<[i8; 2]>(), 0..5),
        z in prop::collection::vec(any::<[i8; 2]>(), 0..5),
    ) {
        let erp = VectorErpMetric::try_new(metric(), sample([0, 0])).unwrap();
        let x = erp.try_series(x.into_iter().map(sample).collect(), ResourceLimits::default()).unwrap();
        let y = erp.try_series(y.into_iter().map(sample).collect(), ResourceLimits::default()).unwrap();
        let z = erp.try_series(z.into_iter().map(sample).collect(), ResourceLimits::default()).unwrap();
        let dxy = completed_distance(erp.distance_bounded(&x, &y, f64::INFINITY, ResourceLimits::default()).unwrap());
        let dyx = completed_distance(erp.distance_bounded(&y, &x, f64::INFINITY, ResourceLimits::default()).unwrap());
        let dxz = completed_distance(erp.distance_bounded(&x, &z, f64::INFINITY, ResourceLimits::default()).unwrap());
        let dzy = completed_distance(erp.distance_bounded(&z, &y, f64::INFINITY, ResourceLimits::default()).unwrap());
        prop_assert!(close(dxy, dyx));
        prop_assert_eq!(dxy == 0.0, x == y);
        prop_assert!(dzy <= dxz + dxy + 1.0e-8);
        prop_assert!(erp.candidate_lower_bound(&x, &y).unwrap() <= dxy + 1.0e-12);
    }

    #[test]
    fn timestamped_vector_twed_metric_laws_hold_on_one_typed_time_domain(
        x in any::<[i8; 2]>(),
        y in any::<[i8; 2]>(),
        z in any::<[i8; 2]>(),
    ) {
        let twed = VectorTimestampedTwedMetric::try_new(metric(), sample([0, 0]), 1.0, 1.0).unwrap();
        let x = twed.try_series(vec![sample(x)], &[1.0], TimestampUnit::Seconds, 0.0, ResourceLimits::default()).unwrap();
        let y = twed.try_series(vec![sample(y)], &[2.0], TimestampUnit::Seconds, 0.0, ResourceLimits::default()).unwrap();
        let z = twed.try_series(vec![sample(z)], &[3.0], TimestampUnit::Seconds, 0.0, ResourceLimits::default()).unwrap();
        let dxy = completed_distance(twed.distance_bounded(&x, &y, f64::INFINITY, ResourceLimits::default()).unwrap());
        let dyx = completed_distance(twed.distance_bounded(&y, &x, f64::INFINITY, ResourceLimits::default()).unwrap());
        let dxz = completed_distance(twed.distance_bounded(&x, &z, f64::INFINITY, ResourceLimits::default()).unwrap());
        let dzy = completed_distance(twed.distance_bounded(&z, &y, f64::INFINITY, ResourceLimits::default()).unwrap());
        prop_assert!(close(dxy, dyx));
        prop_assert!(dxy > 0.0 || x == y);
        prop_assert!(dzy <= dxz + dxy + 1.0e-8);
    }
}

#[test]
fn gap_insertion_is_canonicalized_before_erp_metric_identity_is_claimed() {
    let erp = VectorErpMetric::try_new(metric(), sample([0, 0])).unwrap();
    let with_gap = erp
        .try_series(
            vec![sample([1, 2]), sample([0, 0]), sample([3, 4])],
            ResourceLimits::default(),
        )
        .unwrap();
    let without_gap = erp
        .try_series(
            vec![sample([1, 2]), sample([3, 4])],
            ResourceLimits::default(),
        )
        .unwrap();
    assert_eq!(with_gap, without_gap);
}

#[test]
fn missing_channel_pair_renormalization_is_a_pinned_nonmetric_control() {
    fn bad_pair_distance(left: &[Option<f64>; 3], right: &[Option<f64>; 3]) -> f64 {
        let mut sum = 0.0;
        let mut present = 0usize;
        for (left, right) in left.iter().zip(right) {
            if let (Some(left), Some(right)) = (left, right) {
                sum += (left - right).abs();
                present += 1;
            }
        }
        sum * 3.0 / present as f64
    }

    let x = [Some(0.0), Some(0.0), None];
    let y = [Some(0.0), None, Some(0.0)];
    let z = [None, Some(100.0), Some(0.0)];
    let dxy = bad_pair_distance(&x, &y);
    let dyz = bad_pair_distance(&y, &z);
    let dxz = bad_pair_distance(&x, &z);
    assert_eq!((dxy, dyz, dxz), (0.0, 0.0, 300.0));
    assert!(dxz > dxy + dyz);
}

#[test]
fn vector_dtw_is_bounded_exact_but_deliberately_not_metric_qualified() {
    let dtw = VectorBandedDtwScorer::new(metric(), 1);
    let left = dtw
        .try_series(
            vec![sample([0, 0]), sample([1, 1])],
            ResourceLimits::default(),
        )
        .unwrap();
    let right = dtw
        .try_series(
            vec![sample([0, 0]), sample([0, 0]), sample([1, 1])],
            ResourceLimits::default(),
        )
        .unwrap();
    assert_eq!(
        completed_distance(
            dtw.distance_bounded(&left, &right, f64::INFINITY, ResourceLimits::default())
                .unwrap(),
        ),
        0.0,
    );
    assert_ne!(left, right);
}

#[test]
fn timestamped_local_k1_bounds_strengthen_under_box_refinement() {
    let metric = metric();
    let twed =
        VectorTimestampedTwedMetric::try_new(metric.clone(), sample([0, 0]), 2.0, 1.0).unwrap();
    let query_previous = sample([1, 2]);
    let query_current = sample([3, 5]);
    let candidate_previous = sample([2, 2]);
    let candidate_current = sample([4, 7]);
    let exact_previous = TimestampedVectorBox::try_new(
        VectorBox::from_sample(metric.channel_layout().clone(), &candidate_previous).unwrap(),
        (1.0, 1.0),
        TimestampUnit::Seconds,
    )
    .unwrap();
    let exact_current = TimestampedVectorBox::try_new(
        VectorBox::from_sample(metric.channel_layout().clone(), &candidate_current).unwrap(),
        (3.0, 3.0),
        TimestampUnit::Seconds,
    )
    .unwrap();
    let coarse_previous = TimestampedVectorBox::try_new(
        VectorBox::try_new(metric.channel_layout().clone(), &[(0.0, 4.0), (0.0, 4.0)]).unwrap(),
        (0.5, 1.5),
        TimestampUnit::Seconds,
    )
    .unwrap();
    let coarse_current = TimestampedVectorBox::try_new(
        VectorBox::try_new(metric.channel_layout().clone(), &[(2.0, 6.0), (5.0, 9.0)]).unwrap(),
        (2.5, 3.5),
        TimestampUnit::Seconds,
    )
    .unwrap();
    assert!(exact_previous.refines(&coarse_previous).unwrap());
    assert!(exact_current.refines(&coarse_current).unwrap());
    assert!(
        twed.interval_delete_lower_bound(&coarse_current, &coarse_previous)
            .unwrap()
            <= twed
                .interval_delete_lower_bound(&exact_current, &exact_previous)
                .unwrap()
    );
    assert!(
        twed.interval_match_lower_bound(
            &query_current,
            &query_previous,
            4.0,
            2.0,
            &coarse_current,
            &coarse_previous,
        )
        .unwrap()
            <= twed
                .interval_match_lower_bound(
                    &query_current,
                    &query_previous,
                    4.0,
                    2.0,
                    &exact_current,
                    &exact_previous,
                )
                .unwrap()
    );
}

#[test]
fn frechet_and_dtw_expose_coherent_k1_k4_bounds() {
    let metric = metric();
    let query = sample([1, 2]);
    let candidate = sample([4, 6]);
    let exact_box = VectorBox::from_sample(metric.channel_layout().clone(), &candidate).unwrap();
    let point_distance = metric.distance_checked(&query, &candidate).unwrap();

    let dtw = VectorBandedDtwScorer::new(metric.clone(), 0);
    assert!(close(
        dtw.interval_local_lower_bound_squared(&query, &exact_box)
            .unwrap(),
        point_distance * point_distance,
    ));

    let frechet = VectorFrechetMetric::new(metric);
    assert!(close(
        frechet
            .interval_link_lower_bound(&query, &exact_box)
            .unwrap(),
        point_distance,
    ));
    let left = VectorFrechetPath::try_new(
        vec![query.clone(), sample([2, 3])],
        ResourceLimits::default(),
    )
    .unwrap();
    let right = VectorFrechetPath::try_new(
        vec![candidate.clone(), sample([5, 8])],
        ResourceLimits::default(),
    )
    .unwrap();
    let exact = completed_distance(
        frechet
            .distance_bounded(&left, &right, f64::INFINITY, ResourceLimits::default())
            .unwrap(),
    );
    assert!(frechet.candidate_lower_bound(&left, &right).unwrap() <= exact);
}

#[test]
fn vector_preflight_is_fail_closed_and_msm_decision_is_explicit() {
    let erp = VectorErpMetric::try_new(metric(), sample([0, 0])).unwrap();
    let left = erp
        .try_series(vec![sample([1, 2])], ResourceLimits::default())
        .unwrap();
    let right = erp
        .try_series(vec![sample([3, 4])], ResourceLimits::default())
        .unwrap();
    let limits = ResourceLimits {
        max_dp_cells: 0,
        ..ResourceLimits::default()
    };
    assert!(matches!(
        erp.distance_bounded(&left, &right, f64::INFINITY, limits),
        Ok(OperationOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::DpCells,
                ..
            },
            ..
        })
    ));
    assert_eq!(
        VECTOR_MSM_SUPPORT,
        VectorMsmSupportDecision::UnsupportedNoCanonicalBetweenness
    );
    assert!(matches!(
        MetricChannel::try_new(ChannelIdentity::try_new("x", "metre").unwrap(), 1.0, 0.0,),
        Err(VectorMetricError::InvalidChannelWeight { .. })
    ));

    assert!(matches!(
        VectorSample::try_new(
            &[1.0, 2.0],
            ResourceLimits {
                max_scratch_bytes: 15,
                ..ResourceLimits::default()
            },
        ),
        Err(VectorMetricError::Resource(
            IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: 15,
                requested: 16,
            }
        ))
    ));

    let scratch_limited = ResourceLimits {
        max_scratch_bytes: 31,
        ..ResourceLimits::default()
    };
    assert!(matches!(
        erp.distance_bounded(&left, &right, f64::INFINITY, scratch_limited),
        Ok(OperationOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: 31,
                requested: 32,
            },
            ..
        })
    ));
}
