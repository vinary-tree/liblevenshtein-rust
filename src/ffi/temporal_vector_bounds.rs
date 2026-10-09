//! Typed interval and candidate bounds for native vector temporal kernels.

use super::{
    index::{boundary, slice},
    temporal::timestamp_unit,
    temporal_vector::{
        limits_from_view, samples, vector_error, LlevVectorMetric, LlevVectorSeriesView,
        LlevVectorTemporalConfig, LlevVectorTemporalLimits,
    },
    LlevStatus,
};
use crate::time_series::{
    ResourceLimits, TimestampedVectorBox, VectorBandedDtwScorer, VectorBox, VectorErpMetric,
    VectorFrechetMetric, VectorFrechetPath, VectorSample, VectorTimestampedTwedMetric,
};
use std::mem::size_of;

/// Finite closed interval for one fixed-channel coordinate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevVectorInterval {
    /// Inclusive lower coordinate.
    pub low: f64,
    /// Inclusive upper coordinate.
    pub high: f64,
}

/// One vector box and its finite physical-time interval.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTimestampedVectorBoxView {
    /// Ordered coordinate intervals.
    pub coordinates: *const LlevVectorInterval,
    /// Number of coordinate intervals.
    pub dimension: usize,
    /// Inclusive lower timestamp.
    pub time_low: f64,
    /// Inclusive upper timestamp.
    pub time_high: f64,
    /// Canonical timestamp unit.
    pub unit: u32,
    /// Must be zero.
    pub reserved: u32,
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

fn checked_storage(
    dimension: usize,
    interval_count: usize,
    point_count: usize,
    limits: ResourceLimits,
) -> Result<(), (LlevStatus, String)> {
    if dimension == 0 || dimension > limits.max_dimension {
        return Err((
            LlevStatus::LimitExceeded,
            "vector bound dimension limit".into(),
        ));
    }
    let bytes = (|| {
        let intervals = dimension.checked_mul(interval_count)?;
        let points = dimension.checked_mul(point_count)?;
        intervals
            .checked_mul(size_of::<(f64, f64)>())?
            .checked_add(points.checked_mul(size_of::<f64>())?)
    })()
    .ok_or((
        LlevStatus::LimitExceeded,
        "vector bound storage overflow".into(),
    ))?;
    if bytes > limits.max_scratch_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            "vector bound storage limit".into(),
        ));
    }
    Ok(())
}

unsafe fn vector_box(
    metric: &LlevVectorMetric,
    intervals: *const LlevVectorInterval,
    dimension: usize,
    name: &str,
) -> Result<VectorBox, (LlevStatus, String)> {
    let expected = metric.metric.channel_layout().dimension();
    if dimension != expected {
        return Err(invalid(format!("{name} dimension differs from metric")));
    }
    let raw = slice(intervals, dimension, name)?;
    let mut bounds = Vec::new();
    bounds
        .try_reserve_exact(dimension)
        .map_err(|_| (LlevStatus::LimitExceeded, format!("{name} allocation")))?;
    bounds.extend(raw.iter().map(|interval| (interval.low, interval.high)));
    VectorBox::try_new(
        metric
            .metric
            .channel_layout()
            .try_clone()
            .map_err(vector_error)?,
        &bounds,
    )
    .map_err(vector_error)
}

unsafe fn point(
    coordinates: *const f64,
    dimension: usize,
    expected: usize,
    limits: ResourceLimits,
    name: &str,
) -> Result<VectorSample, (LlevStatus, String)> {
    if dimension != expected {
        return Err(invalid(format!("{name} dimension differs from metric")));
    }
    VectorSample::try_new(slice(coordinates, dimension, name)?, limits).map_err(vector_error)
}

unsafe fn timestamp_box(
    metric: &LlevVectorMetric,
    raw: LlevTimestampedVectorBoxView,
    name: &str,
) -> Result<TimestampedVectorBox, (LlevStatus, String)> {
    if raw.reserved != 0 {
        return Err(invalid(format!("{name} reserved field must be zero")));
    }
    let box_value = vector_box(metric, raw.coordinates, raw.dimension, name)?;
    TimestampedVectorBox::try_new(
        box_value,
        (raw.time_low, raw.time_high),
        timestamp_unit(raw.unit)?,
    )
    .map_err(vector_error)
}

/// Native K1 fixed-channel point-to-box lower bound.
///
/// # Safety
/// The metric, point, box, limits, and output must be valid for this call.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_point_box_lower_bound(
    metric: *const LlevVectorMetric,
    coordinates: *const f64,
    dimension: usize,
    intervals: *const LlevVectorInterval,
    interval_count: usize,
    raw_limits: *const LlevVectorTemporalLimits,
    out_bound: *mut f64,
) -> LlevStatus {
    boundary(|| {
        let output = out_bound
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "point-box output is null".into()))?;
        *output = 0.0;
        let metric = metric
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "vector metric is null".into()))?;
        let limits = limits_from_view(
            *raw_limits
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "point-box limits are null".into()))?,
        );
        let expected = metric.metric.channel_layout().dimension();
        checked_storage(expected, 1, 1, limits)?;
        let point = point(coordinates, dimension, expected, limits, "point")?;
        let box_value = vector_box(metric, intervals, interval_count, "vector box")?;
        *output = metric
            .metric
            .point_box_lower_bound(&point, &box_value)
            .map_err(vector_error)?;
        Ok(LlevStatus::Ok)
    })
}

/// Native K1 fixed-channel box-to-box lower bound.
///
/// # Safety
/// The metric, both boxes, limits, and output must be valid for this call.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_box_box_lower_bound(
    metric: *const LlevVectorMetric,
    left: *const LlevVectorInterval,
    left_len: usize,
    right: *const LlevVectorInterval,
    right_len: usize,
    raw_limits: *const LlevVectorTemporalLimits,
    out_bound: *mut f64,
) -> LlevStatus {
    boundary(|| {
        let output = out_bound
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "box-box output is null".into()))?;
        *output = 0.0;
        let metric = metric
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "vector metric is null".into()))?;
        let limits = limits_from_view(
            *raw_limits
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "box-box limits are null".into()))?,
        );
        checked_storage(metric.metric.channel_layout().dimension(), 2, 0, limits)?;
        let left = vector_box(metric, left, left_len, "left vector box")?;
        let right = vector_box(metric, right, right_len, "right vector box")?;
        *output = metric
            .metric
            .box_box_lower_bound(&left, &right)
            .map_err(vector_error)?;
        Ok(LlevStatus::Ok)
    })
}

/// Native K4 candidate bound for ERP, banded DTW, Fréchet, or timestamped
/// TWED. DTW and TWED currently return their checked identity bound zero.
///
/// # Safety
/// The metric, views, config, limits, and output must be valid and disjoint.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_temporal_candidate_lower_bound(
    metric: *const LlevVectorMetric,
    left: *const LlevVectorSeriesView,
    right: *const LlevVectorSeriesView,
    config: *const LlevVectorTemporalConfig,
    raw_limits: *const LlevVectorTemporalLimits,
    out_bound: *mut f64,
) -> LlevStatus {
    boundary(|| {
        let output = out_bound.as_mut().ok_or((
            LlevStatus::NullPointer,
            "vector candidate output is null".into(),
        ))?;
        *output = 0.0;
        let metric = metric
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "vector metric is null".into()))?;
        let left = *left
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "left series is null".into()))?;
        let right = *right
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "right series is null".into()))?;
        let config = *config
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "candidate config is null".into()))?;
        let limits = limits_from_view(
            *raw_limits
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "candidate limits are null".into()))?,
        );
        if config.reserved != 0
            || !matches!(config.algorithm, 2 | 4 | 5 | 7)
            || config.cutoff != f64::INFINITY
        {
            return Err(invalid("invalid vector candidate-bound configuration"));
        }
        if config.algorithm == 4 && config.band > limits.max_band_width {
            return Err((LlevStatus::LimitExceeded, "vector DTW band limit".into()));
        }
        let dimension = metric.metric.channel_layout().dimension();
        let count = left.sample_count.checked_add(right.sample_count).ok_or((
            LlevStatus::LimitExceeded,
            "candidate input count overflow".into(),
        ))?;
        let interval_count = if matches!(config.algorithm, 2 | 7) {
            1
        } else {
            0
        };
        let storage_bytes = count
            .checked_mul(
                dimension
                    .checked_mul(size_of::<f64>())
                    .and_then(|bytes| bytes.checked_add(size_of::<VectorSample>()))
                    .ok_or((LlevStatus::LimitExceeded, "candidate width overflow".into()))?,
            )
            .and_then(|bytes| {
                dimension
                    .checked_mul(interval_count)
                    .and_then(|gap| gap.checked_mul(size_of::<f64>()))
                    .and_then(|gap| bytes.checked_add(gap))
            })
            .and_then(|bytes| {
                if config.algorithm == 7 {
                    count
                        .checked_mul(size_of::<f64>())
                        .and_then(|time| bytes.checked_add(time))
                } else {
                    Some(bytes)
                }
            })
            .ok_or((
                LlevStatus::LimitExceeded,
                "candidate input size overflow".into(),
            ))?;
        if storage_bytes > limits.max_scratch_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "candidate input storage limit".into(),
            ));
        }
        let left_samples = samples(left, dimension, limits, "left vector series")?;
        let right_samples = samples(right, dimension, limits, "right vector series")?;
        *output = match config.algorithm {
            2 => {
                if config.parameter0 != 0.0 || config.parameter1 != 0.0 || config.band != 0 {
                    return Err(invalid("unused ERP bound parameters must be zero"));
                }
                let gap = point(
                    config.gap_or_sentinel,
                    dimension,
                    dimension,
                    limits,
                    "ERP gap",
                )?;
                let kernel =
                    VectorErpMetric::try_new(metric.metric.clone(), gap).map_err(vector_error)?;
                let x = kernel
                    .try_series(left_samples, limits)
                    .map_err(vector_error)?;
                let y = kernel
                    .try_series(right_samples, limits)
                    .map_err(vector_error)?;
                kernel.candidate_lower_bound(&x, &y).map_err(vector_error)?
            }
            4 => {
                if !config.gap_or_sentinel.is_null()
                    || config.parameter0 != 0.0
                    || config.parameter1 != 0.0
                {
                    return Err(invalid("unused DTW bound parameters must be zero"));
                }
                let kernel = VectorBandedDtwScorer::new(metric.metric.clone(), config.band);
                let x = kernel
                    .try_series(left_samples, limits)
                    .map_err(vector_error)?;
                let y = kernel
                    .try_series(right_samples, limits)
                    .map_err(vector_error)?;
                kernel.candidate_lower_bound(&x, &y).map_err(vector_error)?
            }
            5 => {
                if !config.gap_or_sentinel.is_null()
                    || config.parameter0 != 0.0
                    || config.parameter1 != 0.0
                    || config.band != 0
                {
                    return Err(invalid("unused Fréchet bound parameters must be zero"));
                }
                let kernel = VectorFrechetMetric::new(metric.metric.clone());
                let x = VectorFrechetPath::try_new(left_samples, limits).map_err(vector_error)?;
                let y = VectorFrechetPath::try_new(right_samples, limits).map_err(vector_error)?;
                kernel.candidate_lower_bound(&x, &y).map_err(vector_error)?
            }
            7 => {
                if config.band != 0 {
                    return Err(invalid("unused TWED bound band must be zero"));
                }
                let unit = timestamp_unit(left.timestamp_unit)?;
                if unit != timestamp_unit(right.timestamp_unit)? {
                    return Err(invalid("vector TWED timestamp units differ"));
                }
                let sentinel = point(
                    config.gap_or_sentinel,
                    dimension,
                    dimension,
                    limits,
                    "TWED sentinel",
                )?;
                let kernel = VectorTimestampedTwedMetric::try_new(
                    metric.metric.clone(),
                    sentinel,
                    config.parameter0,
                    config.parameter1,
                )
                .map_err(vector_error)?;
                let x = kernel
                    .try_series(
                        left_samples,
                        slice(left.timestamps, left.sample_count, "left timestamps")?,
                        unit,
                        left.origin,
                        limits,
                    )
                    .map_err(vector_error)?;
                let y = kernel
                    .try_series(
                        right_samples,
                        slice(right.timestamps, right.sample_count, "right timestamps")?,
                        unit,
                        right.origin,
                        limits,
                    )
                    .map_err(vector_error)?;
                kernel.candidate_lower_bound(&x, &y).map_err(vector_error)?
            }
            _ => unreachable!(),
        };
        Ok(LlevStatus::Ok)
    })
}

/// Native K1 timestamped-vector TWED local bound. Mode one is deletion of
/// two consecutive candidate boxes; mode two is matching two exact query
/// points to those boxes.
///
/// # Safety
/// All pointers must address their declared lengths, and the output must be
/// writable and disjoint. Mode two requires both query points.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn llev_vector_twed_interval_lower_bound(
    metric: *const LlevVectorMetric,
    mode: u32,
    query_current: *const f64,
    query_previous: *const f64,
    query_dimension: usize,
    query_current_time: f64,
    query_previous_time: f64,
    candidate_current: *const LlevTimestampedVectorBoxView,
    candidate_previous: *const LlevTimestampedVectorBoxView,
    config: *const LlevVectorTemporalConfig,
    raw_limits: *const LlevVectorTemporalLimits,
    out_bound: *mut f64,
) -> LlevStatus {
    boundary(|| {
        let output = out_bound.as_mut().ok_or((
            LlevStatus::NullPointer,
            "vector TWED bound output is null".into(),
        ))?;
        *output = 0.0;
        let metric = metric
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "vector metric is null".into()))?;
        let current = *candidate_current
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "current box is null".into()))?;
        let previous = *candidate_previous
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "previous box is null".into()))?;
        let config = *config
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "TWED bound config is null".into()))?;
        let limits = limits_from_view(
            *raw_limits
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "TWED bound limits are null".into()))?,
        );
        if !matches!(mode, 1 | 2)
            || config.algorithm != 7
            || config.reserved != 0
            || config.band != 0
            || config.cutoff != f64::INFINITY
        {
            return Err(invalid("invalid vector TWED interval-bound configuration"));
        }
        let dimension = metric.metric.channel_layout().dimension();
        checked_storage(dimension, 2, if mode == 2 { 3 } else { 1 }, limits)?;
        let sentinel = point(
            config.gap_or_sentinel,
            dimension,
            dimension,
            limits,
            "TWED sentinel",
        )?;
        let kernel = VectorTimestampedTwedMetric::try_new(
            metric.metric.clone(),
            sentinel,
            config.parameter0,
            config.parameter1,
        )
        .map_err(vector_error)?;
        let current = timestamp_box(metric, current, "current vector time box")?;
        let previous = timestamp_box(metric, previous, "previous vector time box")?;
        *output = if mode == 1 {
            if !query_current.is_null()
                || !query_previous.is_null()
                || query_dimension != 0
                || query_current_time != 0.0
                || query_previous_time != 0.0
            {
                return Err(invalid("delete mode must have no query points"));
            }
            kernel
                .interval_delete_lower_bound(&current, &previous)
                .map_err(vector_error)?
        } else {
            let query_current = point(
                query_current,
                query_dimension,
                dimension,
                limits,
                "current query point",
            )?;
            let query_previous = point(
                query_previous,
                query_dimension,
                dimension,
                limits,
                "previous query point",
            )?;
            kernel
                .interval_match_lower_bound(
                    &query_current,
                    &query_previous,
                    query_current_time,
                    query_previous_time,
                    &current,
                    &previous,
                )
                .map_err(vector_error)?
        };
        Ok(LlevStatus::Ok)
    })
}
