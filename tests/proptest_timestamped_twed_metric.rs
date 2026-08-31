//! Metric-domain and physical-origin laws for explicit-timestamp TWED.

use liblevenshtein::time_series::{
    ExactDecision, MetricTimestampedTwedConfig, OperationOutcome, ResourceLimits, TimestampUnit,
    TimestampedSeries,
};
use proptest::prelude::*;

fn points() -> impl Strategy<Value = Vec<(i8, u8)>> {
    prop::collection::vec((-20_i8..=20, 1_u8..=6), 1..7)
}

fn series(points: &[(i8, u8)], origin: f64) -> TimestampedSeries {
    let mut timestamp = origin;
    let mut values = Vec::with_capacity(points.len());
    let mut timestamps = Vec::with_capacity(points.len());
    for &(value, delta) in points {
        timestamp += f64::from(delta);
        values.push(f64::from(value));
        timestamps.push(timestamp);
    }
    TimestampedSeries::try_new_with_origin(
        &values,
        &timestamps,
        TimestampUnit::Seconds,
        origin,
        ResourceLimits::default(),
    )
    .unwrap()
}

fn exact_distance(
    config: MetricTimestampedTwedConfig,
    left: &TimestampedSeries,
    right: &TimestampedSeries,
) -> f64 {
    match config
        .distance_bounded(left, right, f64::INFINITY, ResourceLimits::default())
        .unwrap()
    {
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, .. },
            ..
        } => distance,
        other => panic!("small finite metric operands must complete exactly: {other:?}"),
    }
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1.0e-9 * left.abs().max(right.abs()).max(1.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn explicit_timestamp_twed_satisfies_the_metric_laws_on_one_physical_frame(
        x in points(),
        y in points(),
        z in points(),
        stiffness in 1_u8..=8,
        gap in 0_u8..=8,
    ) {
        let x = series(&x, 0.0);
        let y = series(&y, 0.0);
        let z = series(&z, 0.0);
        let config = MetricTimestampedTwedConfig::try_new(
            f64::from(stiffness) / 4.0,
            f64::from(gap) / 4.0,
        ).unwrap();

        let dxx = exact_distance(config, &x, &x);
        let dxy = exact_distance(config, &x, &y);
        let dyx = exact_distance(config, &y, &x);
        let dxz = exact_distance(config, &x, &z);
        let dzy = exact_distance(config, &z, &y);

        prop_assert_eq!(dxx.to_bits(), 0.0_f64.to_bits());
        prop_assert!(dxy >= 0.0);
        prop_assert!(close(dxy, dyx));
        prop_assert!(dxy <= dxz + dzy + 1.0e-9);
    }

    #[test]
    fn translating_the_shared_physical_origin_preserves_distance(
        x in points(),
        y in points(),
        shift in -100_i16..=100,
    ) {
        let baseline_x = series(&x, 0.0);
        let baseline_y = series(&y, 0.0);
        let shifted_origin = f64::from(shift);
        let shifted_x = series(&x, shifted_origin);
        let shifted_y = series(&y, shifted_origin);
        let config = MetricTimestampedTwedConfig::try_new(0.75, 1.25).unwrap();

        prop_assert!(close(
            exact_distance(config, &baseline_x, &baseline_y),
            exact_distance(config, &shifted_x, &shifted_y),
        ));
    }
}
