#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_vector_box_box_lower_bound, llev_vector_metric_free, llev_vector_metric_new,
    llev_vector_point_box_lower_bound, llev_vector_temporal_candidate_lower_bound,
    llev_vector_twed_interval_lower_bound, LlevStatus, LlevTemporalLimits,
    LlevTimestampedVectorBoxView, LlevVectorChannelView, LlevVectorInterval, LlevVectorMetric,
    LlevVectorMetricView, LlevVectorSeriesView, LlevVectorTemporalConfig, LlevVectorTemporalLimits,
};
use liblevenshtein::time_series::{
    ChannelIdentity, FixedChannelMetric, FoldLocalScaleProvenance, MetricChannel, ResourceLimits,
    TimestampUnit, TimestampedVectorBox, VectorBandedDtwScorer, VectorBox, VectorErpMetric,
    VectorFrechetMetric, VectorFrechetPath, VectorSample, VectorTimestampedTwedMetric,
};

fn native_metric() -> FixedChannelMetric {
    FixedChannelMetric::try_new(
        [("x", 2.0, 3.0), ("y", 4.0, 5.0)]
            .into_iter()
            .map(|(name, scale, weight)| {
                MetricChannel::try_new(ChannelIdentity::try_new(name, "m").unwrap(), scale, weight)
                    .unwrap()
            })
            .collect(),
        FoldLocalScaleProvenance::try_new("fold", "v1").unwrap(),
    )
    .unwrap()
}

fn ffi_metric() -> *mut LlevVectorMetric {
    let channels = [
        LlevVectorChannelView {
            channel: b"x".as_ptr(),
            channel_len: 1,
            unit: b"m".as_ptr(),
            unit_len: 1,
            scale: 2.0,
            weight: 3.0,
        },
        LlevVectorChannelView {
            channel: b"y".as_ptr(),
            channel_len: 1,
            unit: b"m".as_ptr(),
            unit_len: 1,
            scale: 4.0,
            weight: 5.0,
        },
    ];
    let view = LlevVectorMetricView {
        channels: channels.as_ptr(),
        channel_count: 2,
        training_fold: b"fold".as_ptr(),
        training_fold_len: 4,
        estimator_revision: b"v1".as_ptr(),
        estimator_revision_len: 2,
    };
    let mut handle = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_vector_metric_new(&view, 2, &mut handle) },
        LlevStatus::Ok
    );
    handle
}

fn limits() -> LlevVectorTemporalLimits {
    LlevVectorTemporalLimits {
        scalar: LlevTemporalLimits::default(),
        max_dimension: 2,
        max_band_width: 1,
    }
}

fn native_sample(values: &[f64]) -> VectorSample {
    VectorSample::try_new(values, ResourceLimits::default()).unwrap()
}

fn native_samples(values: &[f64]) -> Vec<VectorSample> {
    values.chunks_exact(2).map(native_sample).collect()
}

fn native_box(metric: &FixedChannelMetric, values: &[(f64, f64)]) -> VectorBox {
    VectorBox::try_new(metric.channel_layout().clone(), values).unwrap()
}

fn series_view(values: &[f64], timestamps: &[f64]) -> LlevVectorSeriesView {
    LlevVectorSeriesView {
        coordinates: values.as_ptr(),
        sample_count: values.len() / 2,
        dimension: 2,
        timestamps: timestamps.as_ptr(),
        timestamp_unit: 1,
        reserved: 0,
        origin: 0.0,
    }
}

#[test]
fn typed_local_bounds_match_native_metric_and_fail_closed() {
    let handle = ffi_metric();
    let metric = native_metric();
    let point = [0.0, 0.0];
    let left = [
        LlevVectorInterval {
            low: -1.0,
            high: 1.0,
        },
        LlevVectorInterval {
            low: 2.0,
            high: 3.0,
        },
    ];
    let right = [
        LlevVectorInterval {
            low: 3.0,
            high: 4.0,
        },
        LlevVectorInterval {
            low: 1.0,
            high: 2.0,
        },
    ];
    let native_left = native_box(&metric, &[(-1.0, 1.0), (2.0, 3.0)]);
    let native_right = native_box(&metric, &[(3.0, 4.0), (1.0, 2.0)]);
    let mut actual = 0.0;
    assert_eq!(
        unsafe {
            llev_vector_point_box_lower_bound(
                handle,
                point.as_ptr(),
                2,
                left.as_ptr(),
                2,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(
        actual,
        metric
            .point_box_lower_bound(&native_sample(&point), &native_left)
            .unwrap()
    );
    assert_eq!(
        unsafe {
            llev_vector_box_box_lower_bound(
                handle,
                left.as_ptr(),
                2,
                right.as_ptr(),
                2,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(
        actual,
        metric
            .box_box_lower_bound(&native_left, &native_right)
            .unwrap()
    );
    let bad = [LlevVectorInterval {
        low: 3.0,
        high: 1.0,
    }; 2];
    assert_eq!(
        unsafe {
            llev_vector_point_box_lower_bound(
                handle,
                point.as_ptr(),
                2,
                bad.as_ptr(),
                2,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::InvalidArgument
    );
    assert_eq!(actual, 0.0);
    let restricted = LlevVectorTemporalLimits {
        scalar: LlevTemporalLimits {
            max_scratch_bytes: 1,
            ..LlevTemporalLimits::default()
        },
        ..limits()
    };
    assert_eq!(
        unsafe {
            llev_vector_box_box_lower_bound(
                handle,
                left.as_ptr(),
                2,
                right.as_ptr(),
                2,
                &restricted,
                &mut actual,
            )
        },
        LlevStatus::LimitExceeded
    );
    unsafe { llev_vector_metric_free(handle) };
}

#[test]
fn candidate_and_timestamped_interval_bounds_match_native() {
    let handle = ffi_metric();
    let metric = native_metric();
    let x = [1.0, 2.0, 2.0, 3.0];
    let y = [1.0, 1.0, 3.0, 3.0];
    let times = [1.0, 2.0];
    let x_view = series_view(&x, &times);
    let y_view = series_view(&y, &times);
    let gap = [0.0, 0.0];
    let mut actual = 0.0;
    let mut config = LlevVectorTemporalConfig {
        algorithm: 2,
        reserved: 0,
        gap_or_sentinel: gap.as_ptr(),
        parameter0: 0.0,
        parameter1: 0.0,
        band: 0,
        cutoff: f64::INFINITY,
    };
    let erp = VectorErpMetric::try_new(metric.clone(), native_sample(&gap)).unwrap();
    let erp_x = erp
        .try_series(native_samples(&x), ResourceLimits::default())
        .unwrap();
    let erp_y = erp
        .try_series(native_samples(&y), ResourceLimits::default())
        .unwrap();
    assert_eq!(
        unsafe {
            llev_vector_temporal_candidate_lower_bound(
                handle,
                &x_view,
                &y_view,
                &config,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(actual, erp.candidate_lower_bound(&erp_x, &erp_y).unwrap());

    config.algorithm = 5;
    config.gap_or_sentinel = std::ptr::null();
    let frechet = VectorFrechetMetric::new(metric.clone());
    let fx = VectorFrechetPath::try_new(native_samples(&x), ResourceLimits::default()).unwrap();
    let fy = VectorFrechetPath::try_new(native_samples(&y), ResourceLimits::default()).unwrap();
    assert_eq!(
        unsafe {
            llev_vector_temporal_candidate_lower_bound(
                handle,
                &x_view,
                &y_view,
                &config,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(actual, frechet.candidate_lower_bound(&fx, &fy).unwrap());

    config.algorithm = 4;
    config.band = 1;
    let dtw = VectorBandedDtwScorer::new(metric.clone(), 1);
    let dx = dtw
        .try_series(native_samples(&x), ResourceLimits::default())
        .unwrap();
    let dy = dtw
        .try_series(native_samples(&y), ResourceLimits::default())
        .unwrap();
    assert_eq!(
        unsafe {
            llev_vector_temporal_candidate_lower_bound(
                handle,
                &x_view,
                &y_view,
                &config,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(actual, dtw.candidate_lower_bound(&dx, &dy).unwrap());
    config.band = 2;
    assert_eq!(
        unsafe {
            llev_vector_temporal_candidate_lower_bound(
                handle,
                &x_view,
                &y_view,
                &config,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(actual, 0.0);
    config.band = 0;

    let box_now = [
        LlevVectorInterval {
            low: 2.0,
            high: 3.0,
        },
        LlevVectorInterval {
            low: 3.0,
            high: 4.0,
        },
    ];
    let box_before = [
        LlevVectorInterval {
            low: 0.0,
            high: 1.0,
        },
        LlevVectorInterval {
            low: 1.0,
            high: 2.0,
        },
    ];
    let current = LlevTimestampedVectorBoxView {
        coordinates: box_now.as_ptr(),
        dimension: 2,
        time_low: 2.0,
        time_high: 2.5,
        unit: 1,
        reserved: 0,
    };
    let previous = LlevTimestampedVectorBoxView {
        coordinates: box_before.as_ptr(),
        dimension: 2,
        time_low: 0.5,
        time_high: 1.0,
        unit: 1,
        reserved: 0,
    };
    config.algorithm = 7;
    config.gap_or_sentinel = gap.as_ptr();
    config.parameter0 = 0.5;
    config.parameter1 = 1.0;
    let twed = VectorTimestampedTwedMetric::try_new(metric.clone(), native_sample(&gap), 0.5, 1.0)
        .unwrap();
    let twed_x = twed
        .try_series(
            native_samples(&x),
            &times,
            TimestampUnit::Seconds,
            0.0,
            ResourceLimits::default(),
        )
        .unwrap();
    let twed_y = twed
        .try_series(
            native_samples(&y),
            &times,
            TimestampUnit::Seconds,
            0.0,
            ResourceLimits::default(),
        )
        .unwrap();
    assert_eq!(
        unsafe {
            llev_vector_temporal_candidate_lower_bound(
                handle,
                &x_view,
                &y_view,
                &config,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(
        actual,
        twed.candidate_lower_bound(&twed_x, &twed_y).unwrap()
    );
    let mismatched_time_unit = LlevVectorSeriesView {
        timestamp_unit: 2,
        ..y_view
    };
    assert_eq!(
        unsafe {
            llev_vector_temporal_candidate_lower_bound(
                handle,
                &x_view,
                &mismatched_time_unit,
                &config,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::InvalidArgument
    );
    assert_eq!(actual, 0.0);
    let native_current = TimestampedVectorBox::try_new(
        native_box(&metric, &[(2.0, 3.0), (3.0, 4.0)]),
        (2.0, 2.5),
        TimestampUnit::Seconds,
    )
    .unwrap();
    let native_previous = TimestampedVectorBox::try_new(
        native_box(&metric, &[(0.0, 1.0), (1.0, 2.0)]),
        (0.5, 1.0),
        TimestampUnit::Seconds,
    )
    .unwrap();
    assert_eq!(
        unsafe {
            llev_vector_twed_interval_lower_bound(
                handle,
                1,
                std::ptr::null(),
                std::ptr::null(),
                0,
                0.0,
                0.0,
                &current,
                &previous,
                &config,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(
        actual,
        twed.interval_delete_lower_bound(&native_current, &native_previous)
            .unwrap()
    );
    let query_current = [2.0, 3.0];
    let query_previous = [0.0, 1.0];
    assert_eq!(
        unsafe {
            llev_vector_twed_interval_lower_bound(
                handle,
                2,
                query_current.as_ptr(),
                query_previous.as_ptr(),
                2,
                2.0,
                1.0,
                &current,
                &previous,
                &config,
                &limits(),
                &mut actual,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(
        actual,
        twed.interval_match_lower_bound(
            &native_sample(&query_current),
            &native_sample(&query_previous),
            2.0,
            1.0,
            &native_current,
            &native_previous,
        )
        .unwrap()
    );
    unsafe { llev_vector_metric_free(handle) };
}
