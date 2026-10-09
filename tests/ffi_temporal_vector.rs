#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_vector_frechet_online_advance, llev_vector_frechet_online_free,
    llev_vector_frechet_online_new, llev_vector_frechet_online_observation,
    llev_vector_frechet_online_scratch_bytes, llev_vector_metric_free, llev_vector_metric_new,
    llev_vector_temporal_distance, LlevStatus, LlevTemporalDistanceResult, LlevTemporalLimits,
    LlevTemporalOnlineLimits, LlevTemporalOnlineObservation, LlevTemporalOnlineStep,
    LlevVectorChannelView, LlevVectorMetric, LlevVectorMetricView, LlevVectorSeriesView,
    LlevVectorTemporalConfig, LlevVectorTemporalLimits,
};
use liblevenshtein::time_series::{
    ChannelIdentity, ExactDecision, FixedChannelMetric, FoldLocalScaleProvenance, MetricChannel,
    OperationOutcome, ResourceLimits, TimestampUnit, VectorBandedDtwScorer, VectorErpMetric,
    VectorFrechetMetric, VectorFrechetPath, VectorSample, VectorTimestampedTwedMetric,
};

fn metric() -> FixedChannelMetric {
    FixedChannelMetric::try_new(
        [("x", "metre", 2.0, 3.0), ("y", "metre", 4.0, 5.0)]
            .into_iter()
            .map(|(name, unit, scale, weight)| {
                MetricChannel::try_new(ChannelIdentity::try_new(name, unit).unwrap(), scale, weight)
                    .unwrap()
            })
            .collect(),
        FoldLocalScaleProvenance::try_new("fold-1", "scale-v1").unwrap(),
    )
    .unwrap()
}

fn native_samples(flat: &[f64]) -> Vec<VectorSample> {
    flat.chunks_exact(2)
        .map(|point| VectorSample::try_new(point, ResourceLimits::default()).unwrap())
        .collect()
}

fn exact(outcome: OperationOutcome<ExactDecision>) -> f64 {
    match outcome {
        OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, .. },
            ..
        } => distance,
        other => panic!("expected exact score, got {other:?}"),
    }
}

fn view<'a>(coordinates: &'a [f64], timestamps: &'a [f64]) -> LlevVectorSeriesView {
    LlevVectorSeriesView {
        coordinates: coordinates.as_ptr(),
        sample_count: coordinates.len() / 2,
        dimension: 2,
        timestamps: timestamps.as_ptr(),
        timestamp_unit: 1,
        reserved: 0,
        origin: 0.0,
    }
}

fn ffi_metric() -> *mut LlevVectorMetric {
    let channels = [
        LlevVectorChannelView {
            channel: b"x".as_ptr(),
            channel_len: 1,
            unit: b"metre".as_ptr(),
            unit_len: 5,
            scale: 2.0,
            weight: 3.0,
        },
        LlevVectorChannelView {
            channel: b"y".as_ptr(),
            channel_len: 1,
            unit: b"metre".as_ptr(),
            unit_len: 5,
            scale: 4.0,
            weight: 5.0,
        },
    ];
    let view = LlevVectorMetricView {
        channels: channels.as_ptr(),
        channel_count: 2,
        training_fold: b"fold-1".as_ptr(),
        training_fold_len: 6,
        estimator_revision: b"scale-v1".as_ptr(),
        estimator_revision_len: 8,
    };
    let mut metric = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_vector_metric_new(&view, 2, &mut metric) },
        LlevStatus::Ok
    );
    metric
}

#[test]
fn reusable_vector_metric_matches_native_kernels_and_limits() {
    let channels = [
        LlevVectorChannelView {
            channel: b"x".as_ptr(),
            channel_len: 1,
            unit: b"metre".as_ptr(),
            unit_len: 5,
            scale: 2.0,
            weight: 3.0,
        },
        LlevVectorChannelView {
            channel: b"y".as_ptr(),
            channel_len: 1,
            unit: b"metre".as_ptr(),
            unit_len: 5,
            scale: 4.0,
            weight: 5.0,
        },
    ];
    let raw_metric = LlevVectorMetricView {
        channels: channels.as_ptr(),
        channel_count: 2,
        training_fold: b"fold-1".as_ptr(),
        training_fold_len: 6,
        estimator_revision: b"scale-v1".as_ptr(),
        estimator_revision_len: 8,
    };
    let mut handle = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_vector_metric_new(&raw_metric, 2, &mut handle) },
        LlevStatus::Ok
    );
    assert!(!handle.is_null());

    let x = [0.0, 1.0, 2.0, 3.0, 4.0, 1.0];
    let y = [0.0, 1.0, 2.0, 2.0, 3.0, 1.0];
    let tx = [1.0, 2.0, 3.0];
    let ty = [1.0, 2.0, 3.0];
    let left = view(&x, &tx);
    let right = view(&y, &ty);
    let limits = LlevVectorTemporalLimits {
        scalar: LlevTemporalLimits::default(),
        max_dimension: 2,
        max_band_width: 2,
    };
    let native = metric();
    let gap = [0.0, 0.0];
    let sentinel = VectorSample::try_new(&gap, ResourceLimits::default()).unwrap();
    let x_samples = native_samples(&x);
    let y_samples = native_samples(&y);
    let erp = VectorErpMetric::try_new(native.clone(), sentinel.clone()).unwrap();
    let erp_x = erp
        .try_series(x_samples.clone(), ResourceLimits::default())
        .unwrap();
    let erp_y = erp
        .try_series(y_samples.clone(), ResourceLimits::default())
        .unwrap();
    let expected_erp = exact(
        erp.distance_bounded(&erp_x, &erp_y, f64::INFINITY, ResourceLimits::default())
            .unwrap(),
    );
    let dtw = VectorBandedDtwScorer::new(native.clone(), 1);
    let dtw_x = dtw
        .try_series(x_samples.clone(), ResourceLimits::default())
        .unwrap();
    let dtw_y = dtw
        .try_series(y_samples.clone(), ResourceLimits::default())
        .unwrap();
    let expected_dtw = exact(
        dtw.distance_bounded(&dtw_x, &dtw_y, f64::INFINITY, ResourceLimits::default())
            .unwrap(),
    );
    let frechet = VectorFrechetMetric::new(native.clone());
    let frechet_x =
        VectorFrechetPath::try_new(x_samples.clone(), ResourceLimits::default()).unwrap();
    let frechet_y =
        VectorFrechetPath::try_new(y_samples.clone(), ResourceLimits::default()).unwrap();
    let expected_frechet = exact(
        frechet
            .distance_bounded(
                &frechet_x,
                &frechet_y,
                f64::INFINITY,
                ResourceLimits::default(),
            )
            .unwrap(),
    );
    let twed = VectorTimestampedTwedMetric::try_new(native, sentinel, 0.5, 1.0).unwrap();
    let twed_x = twed
        .try_series(
            x_samples,
            &tx,
            TimestampUnit::Seconds,
            0.0,
            ResourceLimits::default(),
        )
        .unwrap();
    let twed_y = twed
        .try_series(
            y_samples,
            &ty,
            TimestampUnit::Seconds,
            0.0,
            ResourceLimits::default(),
        )
        .unwrap();
    let expected_twed = exact(
        twed.distance_bounded(&twed_x, &twed_y, f64::INFINITY, ResourceLimits::default())
            .unwrap(),
    );
    for (algorithm, parameter0, parameter1, band, expected) in [
        (2, 0.0, 0.0, 0, expected_erp),
        (4, 0.0, 0.0, 1, expected_dtw),
        (5, 0.0, 0.0, 0, expected_frechet),
        (7, 0.5, 1.0, 0, expected_twed),
    ] {
        let config = LlevVectorTemporalConfig {
            algorithm,
            reserved: 0,
            gap_or_sentinel: if algorithm == 2 || algorithm == 7 {
                gap.as_ptr()
            } else {
                std::ptr::null()
            },
            parameter0,
            parameter1,
            band,
            cutoff: f64::INFINITY,
        };
        let mut output = LlevTemporalDistanceResult::default();
        assert_eq!(
            unsafe {
                llev_vector_temporal_distance(handle, &left, &right, &config, &limits, &mut output)
            },
            LlevStatus::Ok
        );
        assert_eq!(output.kind, 0);
        assert_eq!(output.value, expected);
        assert!(output.work_units > 0);

        let insufficient = LlevVectorTemporalLimits {
            scalar: LlevTemporalLimits {
                max_work_units: 0,
                ..LlevTemporalLimits::default()
            },
            ..limits
        };
        assert_eq!(
            unsafe {
                llev_vector_temporal_distance(
                    handle,
                    &left,
                    &right,
                    &config,
                    &insufficient,
                    &mut output,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(output.kind, 3);
        assert_eq!(output.reason, 2);
    }
    unsafe { llev_vector_metric_free(handle) };
}

#[test]
fn vector_boundary_rejects_malformed_metric_and_series() {
    let duplicate = [LlevVectorChannelView {
        channel: b"x".as_ptr(),
        channel_len: 1,
        unit: b"m".as_ptr(),
        unit_len: 1,
        scale: 1.0,
        weight: 1.0,
    }; 2];
    let raw = LlevVectorMetricView {
        channels: duplicate.as_ptr(),
        channel_count: 2,
        training_fold: b"fold".as_ptr(),
        training_fold_len: 4,
        estimator_revision: b"v1".as_ptr(),
        estimator_revision_len: 2,
    };
    let mut handle = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_vector_metric_new(&raw, 2, &mut handle) },
        LlevStatus::InvalidArgument
    );
    assert!(handle.is_null());
    let channels = [duplicate[0]];
    let good = LlevVectorMetricView {
        channels: channels.as_ptr(),
        channel_count: 1,
        ..raw
    };
    assert_eq!(
        unsafe { llev_vector_metric_new(&good, 1, &mut handle) },
        LlevStatus::Ok
    );
    let coordinates = [1.0, 2.0];
    let mut bad = LlevVectorSeriesView {
        coordinates: coordinates.as_ptr(),
        sample_count: 2,
        dimension: 1,
        timestamps: std::ptr::null(),
        timestamp_unit: 0,
        reserved: 0,
        origin: 0.0,
    };
    let config = LlevVectorTemporalConfig {
        algorithm: 5,
        reserved: 0,
        gap_or_sentinel: std::ptr::null(),
        parameter0: 0.0,
        parameter1: 0.0,
        band: 0,
        cutoff: f64::INFINITY,
    };
    let limits = LlevVectorTemporalLimits {
        scalar: LlevTemporalLimits::default(),
        max_dimension: 1,
        max_band_width: 1,
    };
    let mut output = LlevTemporalDistanceResult::default();
    bad.dimension = 2;
    assert_eq!(
        unsafe { llev_vector_temporal_distance(handle, &bad, &bad, &config, &limits, &mut output) },
        LlevStatus::InvalidArgument
    );
    assert_eq!(output.value, 0.0);
    unsafe { llev_vector_metric_free(handle) };
}

#[test]
fn vector_online_prefixes_match_native_metric_and_rollback_on_limits() {
    let metric_handle = ffi_metric();
    let query_data = [0.0, 0.0, 1.0, 1.0];
    let query = LlevVectorSeriesView {
        coordinates: query_data.as_ptr(),
        sample_count: 2,
        dimension: 2,
        timestamps: std::ptr::null(),
        timestamp_unit: 0,
        reserved: 0,
        origin: 0.0,
    };
    let limits = LlevTemporalOnlineLimits {
        max_query_len: 2,
        max_frontier_positions: 2,
        max_step_work_units: 100,
        max_scratch_bytes: 1024,
    };
    let mut machine = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_vector_frechet_online_new(metric_handle, &query, 10.0, &limits, &mut machine)
        },
        LlevStatus::Ok
    );
    let mut limited = std::ptr::null_mut();
    let stopped_limits = LlevTemporalOnlineLimits {
        max_step_work_units: 0,
        ..limits
    };
    assert_eq!(
        unsafe {
            llev_vector_frechet_online_new(
                metric_handle,
                &query,
                10.0,
                &stopped_limits,
                &mut limited,
            )
        },
        LlevStatus::Ok
    );
    unsafe { llev_vector_metric_free(metric_handle) };
    let mut retained = 0;
    assert_eq!(
        unsafe { llev_vector_frechet_online_scratch_bytes(machine, &mut retained) },
        LlevStatus::Ok
    );
    assert!(retained > 0);
    let mut observed = LlevTemporalOnlineObservation::default();
    assert_eq!(
        unsafe { llev_vector_frechet_online_observation(machine, &mut observed) },
        LlevStatus::Ok
    );
    assert_eq!(observed.consumed_target_len, 0);

    let target = [[0.0, 0.0], [1.0, 1.0]];
    let rust_metric = VectorFrechetMetric::new(metric());
    let rust_query =
        VectorFrechetPath::try_new(native_samples(&query_data), ResourceLimits::default()).unwrap();
    let mut prefix = Vec::new();
    for (index, point) in target.iter().enumerate() {
        let mut step = LlevTemporalOnlineStep::default();
        assert_eq!(
            unsafe {
                llev_vector_frechet_online_advance(machine, point.as_ptr(), point.len(), &mut step)
            },
            LlevStatus::Ok
        );
        assert_eq!(step.kind, 0);
        assert_eq!(step.observation.consumed_target_len, index + 1);
        prefix.extend_from_slice(point);
        let rust_prefix =
            VectorFrechetPath::try_new(native_samples(&prefix), ResourceLimits::default()).unwrap();
        let expected = exact(
            rust_metric
                .distance_bounded(&rust_query, &rust_prefix, 10.0, ResourceLimits::default())
                .unwrap(),
        );
        assert_eq!(step.observation.has_distance, 1);
        assert_eq!(step.observation.distance_within_cutoff, expected);
        let mut current = 0;
        assert_eq!(
            unsafe { llev_vector_frechet_online_scratch_bytes(machine, &mut current) },
            LlevStatus::Ok
        );
        assert_eq!(current, retained);
    }

    let mut step = LlevTemporalOnlineStep::default();
    assert_eq!(
        unsafe { llev_vector_frechet_online_advance(limited, target[0].as_ptr(), 2, &mut step) },
        LlevStatus::Ok
    );
    assert_eq!(step.kind, 1);
    assert_eq!(step.reason, 2);
    assert_eq!(
        unsafe { llev_vector_frechet_online_observation(limited, &mut observed) },
        LlevStatus::Ok
    );
    assert_eq!(observed.consumed_target_len, 0);
    assert_eq!(
        unsafe { llev_vector_frechet_online_advance(machine, target[0].as_ptr(), 1, &mut step) },
        LlevStatus::InvalidArgument
    );
    assert_eq!(
        unsafe { llev_vector_frechet_online_observation(machine, &mut observed) },
        LlevStatus::Ok
    );
    assert_eq!(observed.consumed_target_len, 2);
    unsafe {
        llev_vector_frechet_online_free(machine);
        llev_vector_frechet_online_free(limited);
    }
}
