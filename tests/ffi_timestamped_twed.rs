#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_timestamped_twed_distance, LlevStatus, LlevTemporalDistanceResult, LlevTemporalLimits,
    LlevTimestampedSeriesView,
};
use liblevenshtein::time_series::{
    ExactDecision, MetricTimestampedTwedConfig, OperationOutcome, ResourceLimits, TimestampUnit,
    TimestampedSeries,
};

fn view<'a>(
    values: &'a [f64],
    timestamps: &'a [f64],
    unit: u32,
    origin: f64,
) -> LlevTimestampedSeriesView {
    LlevTimestampedSeriesView {
        values: values.as_ptr(),
        timestamps: timestamps.as_ptr(),
        len: values.len(),
        unit,
        reserved: 0,
        origin,
    }
}

fn compare(
    left: &LlevTimestampedSeriesView,
    right: &LlevTimestampedSeriesView,
    cutoff: f64,
    limits: &LlevTemporalLimits,
) -> (LlevStatus, LlevTemporalDistanceResult) {
    let mut result = LlevTemporalDistanceResult::default();
    let status = unsafe {
        llev_timestamped_twed_distance(left, right, 0.5, 1.0, cutoff, limits, &mut result)
    };
    (status, result)
}

#[test]
fn timestamped_twed_bridge_matches_native_physical_time_and_cutoff() {
    let left_values = [1.0, 2.0];
    let right_values = [1.0, 2.5];
    let left_times = [10.0, 13.0];
    let right_times = [10.0, 14.0];
    let left = view(&left_values, &left_times, 2, 10.0);
    let right = view(&right_values, &right_times, 2, 10.0);
    let config = MetricTimestampedTwedConfig::try_new(0.5, 1.0).unwrap();
    let native_left = TimestampedSeries::try_new_with_origin(
        &left_values,
        &left_times,
        TimestampUnit::Milliseconds,
        10.0,
        ResourceLimits::default(),
    )
    .unwrap();
    let native_right = TimestampedSeries::try_new_with_origin(
        &right_values,
        &right_times,
        TimestampUnit::Milliseconds,
        10.0,
        ResourceLimits::default(),
    )
    .unwrap();
    let expected = match config
        .distance_bounded(
            &native_left,
            &native_right,
            f64::INFINITY,
            ResourceLimits::default(),
        )
        .unwrap()
    {
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, .. },
            ..
        } => distance,
        other => panic!("unexpected native outcome: {other:?}"),
    };
    let limits = LlevTemporalLimits::default();
    let (status, result) = compare(&left, &right, f64::INFINITY, &limits);
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(result.kind, 0);
    assert_eq!(result.value, expected);
    assert!(result.work_units > 0);

    let (status, result) = compare(&left, &right, 0.0, &limits);
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(result.kind, 1);
    assert_eq!(result.value, 0.0);
}

#[test]
fn timestamped_twed_bridge_rejects_invalid_domains_and_reports_limits() {
    let values = [1.0, 2.0];
    let times = [0.0, 2.0];
    let left = view(&values, &times, 1, 0.0);
    let right = view(&values, &times, 1, 0.0);
    let limits = LlevTemporalLimits::default();
    let restricted = LlevTemporalLimits {
        max_dp_cells: 1,
        ..limits
    };
    let (status, result) = compare(&left, &right, f64::INFINITY, &restricted);
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(result.kind, 3);
    assert_eq!(result.reason, 1);
    assert_eq!(result.value, 0.0);

    let scratch_restricted = LlevTemporalLimits {
        max_scratch_bytes: 16,
        ..limits
    };
    let (status, result) = compare(&left, &right, f64::INFINITY, &scratch_restricted);
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(result.kind, 3);
    assert_eq!(result.reason, 3);
    assert_eq!(result.value, 0.0);

    let invalid_unit = view(&values, &times, 9, 0.0);
    let (status, result) = compare(&left, &invalid_unit, f64::INFINITY, &limits);
    assert_eq!(status, LlevStatus::InvalidArgument);
    assert_eq!(result.value, 0.0);

    let shifted_origin = view(&values, &times, 1, -1.0);
    let (status, result) = compare(&left, &shifted_origin, f64::INFINITY, &limits);
    assert_eq!(status, LlevStatus::InvalidArgument);
    assert_eq!(result.value, 0.0);

    let nonmonotone = [0.0, 0.0];
    let invalid_time = view(&values, &nonmonotone, 1, 0.0);
    let (status, result) = compare(&left, &invalid_time, f64::INFINITY, &limits);
    assert_eq!(status, LlevStatus::InvalidArgument);
    assert_eq!(result.value, 0.0);
}
