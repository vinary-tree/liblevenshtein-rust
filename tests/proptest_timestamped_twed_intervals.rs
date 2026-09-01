//! Abstract-interpretation laws for explicit-timestamp TWED labels.

use liblevenshtein::time_series::{
    MetricTimestampedTwedConfig, TimestampUnit, TimestampedScalarBox,
};
use proptest::prelude::*;

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-12 * left.abs().max(right.abs()).max(1.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn singleton_boxes_reproduce_exact_local_recurrence_terms(
        query_previous_value in -100_i16..=100,
        query_current_value in -100_i16..=100,
        candidate_previous_value in -100_i16..=100,
        candidate_current_value in -100_i16..=100,
        query_previous_time in 0_u16..=100,
        query_delta in 1_u16..=100,
        candidate_previous_time in 0_u16..=100,
        candidate_delta in 1_u16..=100,
        stiffness in 1_u16..=20,
        gap in 0_u16..=20,
    ) {
        let qpv = f64::from(query_previous_value);
        let qcv = f64::from(query_current_value);
        let cpv = f64::from(candidate_previous_value);
        let ccv = f64::from(candidate_current_value);
        let qpt = f64::from(query_previous_time);
        let qct = qpt + f64::from(query_delta);
        let cpt = f64::from(candidate_previous_time);
        let cct = cpt + f64::from(candidate_delta);
        let nu = f64::from(stiffness);
        let lambda = f64::from(gap);
        let config = MetricTimestampedTwedConfig::try_new(nu, lambda).unwrap();
        let previous = TimestampedScalarBox::point(cpv, cpt, TimestampUnit::Seconds).unwrap();
        let current = TimestampedScalarBox::point(ccv, cct, TimestampUnit::Seconds).unwrap();

        let abstract_delete = config.interval_delete_lower_bound(current, previous).unwrap();
        let concrete_delete = (ccv - cpv).abs() + nu * (cct - cpt) + lambda;
        prop_assert!(close(abstract_delete, concrete_delete));

        let abstract_match = config.interval_match_lower_bound(
            qcv,
            qpv,
            qct,
            qpt,
            TimestampUnit::Seconds,
            current,
            previous,
        ).unwrap();
        let concrete_match = (qcv - ccv).abs()
            + (qpv - cpv).abs()
            + nu * ((qct - cct).abs() + (qpt - cpt).abs());
        prop_assert!(close(abstract_match, concrete_match));
    }

    #[test]
    fn refinement_is_monotone_and_bounds_every_represented_point(
        query_previous_value in -100_i16..=100,
        query_current_value in -100_i16..=100,
        candidate_previous_value in -100_i16..=100,
        candidate_current_value in -100_i16..=100,
        value_radius in 0_u16..=20,
        time_radius in 0_u16..=1,
    ) {
        let qpv = f64::from(query_previous_value);
        let qcv = f64::from(query_current_value);
        let cpv = f64::from(candidate_previous_value);
        let ccv = f64::from(candidate_current_value);
        let value_radius = f64::from(value_radius);
        let time_radius = f64::from(time_radius) * 0.25;
        let cpt = 1.0;
        let cct = 3.0;
        let qpt = 1.25;
        let qct = 3.25;
        let config = MetricTimestampedTwedConfig::try_new(2.0, 1.0).unwrap();

        let coarse_previous = TimestampedScalarBox::try_new(
            (cpv - value_radius, cpv + value_radius),
            (cpt - time_radius, cpt + time_radius),
            TimestampUnit::Seconds,
        ).unwrap();
        let coarse_current = TimestampedScalarBox::try_new(
            (ccv - value_radius, ccv + value_radius),
            (cct - time_radius, cct + time_radius),
            TimestampUnit::Seconds,
        ).unwrap();
        let exact_previous = TimestampedScalarBox::point(cpv, cpt, TimestampUnit::Seconds).unwrap();
        let exact_current = TimestampedScalarBox::point(ccv, cct, TimestampUnit::Seconds).unwrap();
        prop_assert!(exact_previous.refines(coarse_previous).unwrap());
        prop_assert!(exact_current.refines(coarse_current).unwrap());

        let coarse_delete = config.interval_delete_lower_bound(coarse_current, coarse_previous).unwrap();
        let exact_delete = config.interval_delete_lower_bound(exact_current, exact_previous).unwrap();
        prop_assert!(coarse_delete <= exact_delete + 1.0e-12);

        let coarse_match = config.interval_match_lower_bound(
            qcv, qpv, qct, qpt, TimestampUnit::Seconds, coarse_current, coarse_previous,
        ).unwrap();
        let exact_match = config.interval_match_lower_bound(
            qcv, qpv, qct, qpt, TimestampUnit::Seconds, exact_current, exact_previous,
        ).unwrap();
        prop_assert!(coarse_match <= exact_match + 1.0e-12);
    }
}

#[test]
fn malformed_or_mixed_unit_boxes_fail_closed() {
    assert!(
        TimestampedScalarBox::try_new((1.0, 0.0), (0.0, 1.0), TimestampUnit::Seconds,).is_err()
    );
    assert!(TimestampedScalarBox::try_new(
        (f64::NEG_INFINITY, 0.0),
        (0.0, 1.0),
        TimestampUnit::Seconds,
    )
    .is_ok());
    assert!(TimestampedScalarBox::try_new(
        (0.0, 1.0),
        (0.0, f64::INFINITY),
        TimestampUnit::Seconds,
    )
    .is_err());
    let seconds = TimestampedScalarBox::point(0.0, 1.0, TimestampUnit::Seconds).unwrap();
    let millis = TimestampedScalarBox::point(0.0, 1.0, TimestampUnit::Milliseconds).unwrap();
    assert!(MetricTimestampedTwedConfig::try_new(1.0, 0.0)
        .unwrap()
        .interval_delete_lower_bound(seconds, millis)
        .is_err());
}
