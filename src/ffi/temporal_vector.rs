//! Typed vector temporal scores through a reusable fixed-channel metric.

use super::{
    index::{boundary, slice},
    temporal::{reason_code, timestamp_unit, LlevTemporalDistanceResult, LlevTemporalLimits},
    temporal_index::incomplete_code,
    temporal_online::{
        LlevTemporalOnlineLimits, LlevTemporalOnlineObservation, LlevTemporalOnlineStep,
    },
    LlevStatus,
};
use crate::time_series::{
    ChannelIdentity, ExactDecision, FixedChannelMetric, FoldLocalScaleProvenance, MetricChannel,
    OnlineAutomatonLimits, OnlineStepOutcome, OperationOutcome, ResourceLimits, ResourceUsage,
    VectorBandedDtwScorer, VectorErpMetric, VectorFrechetMetric, VectorFrechetOnlineAutomaton,
    VectorFrechetOnlineObservation, VectorFrechetPath, VectorMetricError, VectorSample,
    VectorTimestampedTwedMetric,
};

/// Exact channel/unit identity and fixed fold-local scale/weight.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevVectorChannelView {
    /// UTF-8 channel identifier.
    pub channel: *const u8,
    /// Channel identifier byte length.
    pub channel_len: usize,
    /// UTF-8 physical unit.
    pub unit: *const u8,
    /// Physical unit byte length.
    pub unit_len: usize,
    /// Strictly positive fold-local scale.
    pub scale: f64,
    /// Strictly positive channel weight.
    pub weight: f64,
}

/// Immutable metric configuration, copied by construction.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevVectorMetricView {
    /// Ordered channel definitions.
    pub channels: *const LlevVectorChannelView,
    /// Number of channel definitions.
    pub channel_count: usize,
    /// UTF-8 training-fold identity.
    pub training_fold: *const u8,
    /// Training-fold byte length.
    pub training_fold_len: usize,
    /// UTF-8 estimator revision.
    pub estimator_revision: *const u8,
    /// Estimator-revision byte length.
    pub estimator_revision_len: usize,
}

/// Column-major matrix of points: each consecutive `dimension` coordinates
/// form one sample. Time fields are used only by timestamped TWED.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevVectorSeriesView {
    /// Contiguous sample columns.
    pub coordinates: *const f64,
    /// Number of samples.
    pub sample_count: usize,
    /// Coordinates per sample.
    pub dimension: usize,
    /// Physical timestamps for TWED, otherwise null.
    pub timestamps: *const f64,
    /// Timestamp unit for TWED, otherwise zero.
    pub timestamp_unit: u32,
    /// Must be zero.
    pub reserved: u32,
    /// Shared physical origin for TWED.
    pub origin: f64,
}

/// Algorithm 2 ERP, 4 banded DTW, 5 Fréchet, or 7 timestamped TWED.
/// ERP and TWED require a finite `gap_or_sentinel` of metric dimension.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevVectorTemporalConfig {
    /// Algorithm code.
    pub algorithm: u32,
    /// Must be zero.
    pub reserved: u32,
    /// Fixed gap or sentinel point for ERP/TWED.
    pub gap_or_sentinel: *const f64,
    /// TWED stiffness, otherwise zero.
    pub parameter0: f64,
    /// TWED gap penalty, otherwise zero.
    pub parameter1: f64,
    /// DTW half-band, otherwise zero.
    pub band: usize,
    /// Inclusive cutoff or positive infinity.
    pub cutoff: f64,
}

/// Explicit vector and scalar work limits.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevVectorTemporalLimits {
    /// Scalar series, cell, work, and scratch ceilings.
    pub scalar: LlevTemporalLimits,
    /// Maximum coordinates per point.
    pub max_dimension: usize,
    /// Maximum DTW half-band.
    pub max_band_width: usize,
}

/// Reusable, immutable native metric. Concurrent distance calls may share it.
pub struct LlevVectorMetric {
    pub(super) metric: FixedChannelMetric,
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

pub(super) fn vector_error(error: VectorMetricError) -> (LlevStatus, String) {
    match error {
        VectorMetricError::Resource(_) => (LlevStatus::LimitExceeded, error.to_string()),
        _ => invalid(error.to_string()),
    }
}

unsafe fn text_at(data: *const u8, len: usize, name: &str) -> Result<&str, (LlevStatus, String)> {
    let bytes = slice(data, len, name)?;
    std::str::from_utf8(bytes).map_err(|error| (LlevStatus::InvalidUtf8, error.to_string()))
}

/// Copy and validate a fixed typed metric once for reuse across comparisons.
///
/// # Safety
/// All input pointers must address their declared lengths during this call.
/// `out_metric` must be writable and disjoint from inputs.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_metric_new(
    view: *const LlevVectorMetricView,
    max_dimension: usize,
    out_metric: *mut *mut LlevVectorMetric,
) -> LlevStatus {
    boundary(|| {
        let output = out_metric.as_mut().ok_or((
            LlevStatus::NullPointer,
            "vector metric output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let view = *view
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "vector metric view is null".into()))?;
        if view.channel_count == 0 || view.channel_count > max_dimension {
            return Err((
                LlevStatus::LimitExceeded,
                "vector metric dimension limit".into(),
            ));
        }
        let raw_channels = slice(view.channels, view.channel_count, "vector channels")?;
        let mut channels = Vec::new();
        channels
            .try_reserve_exact(raw_channels.len())
            .map_err(|_| {
                (
                    LlevStatus::LimitExceeded,
                    "vector channel allocation".into(),
                )
            })?;
        for raw in raw_channels {
            let identity = ChannelIdentity::try_new(
                text_at(raw.channel, raw.channel_len, "vector channel")?,
                text_at(raw.unit, raw.unit_len, "vector unit")?,
            )
            .map_err(vector_error)?;
            channels.push(
                MetricChannel::try_new(identity, raw.scale, raw.weight).map_err(vector_error)?,
            );
        }
        let provenance = FoldLocalScaleProvenance::try_new(
            text_at(view.training_fold, view.training_fold_len, "training fold")?,
            text_at(
                view.estimator_revision,
                view.estimator_revision_len,
                "estimator revision",
            )?,
        )
        .map_err(vector_error)?;
        let metric = FixedChannelMetric::try_new(channels, provenance).map_err(vector_error)?;
        *output = Box::into_raw(Box::new(LlevVectorMetric { metric }));
        Ok(LlevStatus::Ok)
    })
}

/// Release a vector metric after its concurrent users have finished.
///
/// # Safety
/// The pointer must be null or a live handle returned by `llev_vector_metric_new`.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_metric_free(metric: *mut LlevVectorMetric) {
    if !metric.is_null() {
        drop(Box::from_raw(metric));
    }
}

pub(super) fn limits_from_view(view: LlevVectorTemporalLimits) -> ResourceLimits {
    ResourceLimits {
        max_series_len: view.scalar.max_series_len,
        max_dimension: view.max_dimension,
        max_band_width: view.max_band_width,
        max_dp_cells: view.scalar.max_dp_cells,
        max_work_units: view.scalar.max_work_units,
        max_scratch_bytes: view.scalar.max_scratch_bytes,
        ..ResourceLimits::default()
    }
}

pub(super) unsafe fn samples(
    view: LlevVectorSeriesView,
    expected_dimension: usize,
    limits: ResourceLimits,
    name: &str,
) -> Result<Vec<VectorSample>, (LlevStatus, String)> {
    if view.reserved != 0 || view.dimension != expected_dimension {
        return Err(invalid(format!(
            "{name} vector layout differs from the metric"
        )));
    }
    if view.sample_count > limits.max_series_len {
        return Err((
            LlevStatus::LimitExceeded,
            format!("{name} series length limit"),
        ));
    }
    let coordinate_count = view.sample_count.checked_mul(view.dimension).ok_or((
        LlevStatus::LimitExceeded,
        "vector coordinate count overflow".into(),
    ))?;
    let input_bytes = coordinate_count
        .checked_mul(std::mem::size_of::<f64>())
        .and_then(|bytes| {
            view.sample_count
                .checked_mul(std::mem::size_of::<VectorSample>())
                .and_then(|headers| bytes.checked_add(headers))
        })
        .ok_or((
            LlevStatus::LimitExceeded,
            "vector input size overflow".into(),
        ))?;
    if input_bytes > limits.max_scratch_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            format!("{name} input storage limit"),
        ));
    }
    let coordinates = slice(view.coordinates, coordinate_count, name)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(view.sample_count)
        .map_err(|_| (LlevStatus::LimitExceeded, format!("{name} allocation")))?;
    for row in coordinates.chunks_exact(view.dimension) {
        output.push(VectorSample::try_new(row, limits).map_err(vector_error)?);
    }
    Ok(output)
}

fn publish(outcome: OperationOutcome<ExactDecision>, output: &mut LlevTemporalDistanceResult) {
    let usage = outcome.usage();
    output.dp_cells = usage.dp_cells;
    output.work_units = usage.work_units;
    output.scratch_bytes = usage.scratch_bytes;
    match outcome {
        OperationOutcome::Complete { value, .. } => match value {
            ExactDecision::WithinCutoff { distance, .. } => output.value = distance,
            ExactDecision::AboveCutoff => output.kind = 1,
            ExactDecision::NoFiniteAlignment => output.kind = 2,
        },
        OperationOutcome::Incomplete { reason, .. } => {
            output.kind = 3;
            output.reason = reason_code(reason);
        }
    }
}

/// Compare typed vector series with the existing exact native kernels.
/// Result kind and resource accounting match `llev_temporal_distance`.
///
/// # Safety
/// The metric must stay live throughout the call. All views and buffers must
/// address their declared lengths and be disjoint from writable output.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_temporal_distance(
    metric: *const LlevVectorMetric,
    left: *const LlevVectorSeriesView,
    right: *const LlevVectorSeriesView,
    config: *const LlevVectorTemporalConfig,
    raw_limits: *const LlevVectorTemporalLimits,
    out_result: *mut LlevTemporalDistanceResult,
) -> LlevStatus {
    boundary(|| {
        let output = out_result.as_mut().ok_or((
            LlevStatus::NullPointer,
            "vector distance output is null".into(),
        ))?;
        *output = LlevTemporalDistanceResult::default();
        let metric = metric
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "vector metric is null".into()))?;
        let left = *left
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "left vector series is null".into()))?;
        let right = *right.as_ref().ok_or((
            LlevStatus::NullPointer,
            "right vector series is null".into(),
        ))?;
        let config = *config
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "vector config is null".into()))?;
        let limits = limits_from_view(
            *raw_limits
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "vector limits are null".into()))?,
        );
        if config.reserved != 0 || !matches!(config.algorithm, 2 | 4 | 5 | 7) {
            return Err(invalid(
                "unknown vector temporal algorithm or reserved field",
            ));
        }
        let dimension = metric.metric.channel_layout().dimension();
        if dimension > limits.max_dimension {
            return Err((LlevStatus::LimitExceeded, "vector dimension limit".into()));
        }
        let pair_bytes = (|| {
            let count = left.sample_count.checked_add(right.sample_count)?;
            let point_bytes = dimension
                .checked_mul(std::mem::size_of::<f64>())?
                .checked_add(std::mem::size_of::<VectorSample>())?;
            let per_sample = point_bytes.checked_add(if config.algorithm == 7 {
                std::mem::size_of::<f64>()
            } else {
                0
            })?;
            let fixed_point = if matches!(config.algorithm, 2 | 7) {
                point_bytes
            } else {
                0
            };
            count.checked_mul(per_sample)?.checked_add(fixed_point)
        })()
        .ok_or((
            LlevStatus::LimitExceeded,
            "vector pair storage overflow".into(),
        ))?;
        if pair_bytes > limits.max_scratch_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "vector pair input storage limit".into(),
            ));
        }
        let left_samples = samples(left, dimension, limits, "left vector series")?;
        let right_samples = samples(right, dimension, limits, "right vector series")?;
        let outcome = match config.algorithm {
            2 => {
                if config.parameter0 != 0.0 || config.parameter1 != 0.0 || config.band != 0 {
                    return Err(invalid("unused vector ERP parameters must be zero"));
                }
                let gap = VectorSample::try_new(
                    slice(config.gap_or_sentinel, dimension, "vector gap")?,
                    limits,
                )
                .map_err(vector_error)?;
                let kernel =
                    VectorErpMetric::try_new(metric.metric.clone(), gap).map_err(vector_error)?;
                let x = kernel
                    .try_series(left_samples, limits)
                    .map_err(vector_error)?;
                let y = kernel
                    .try_series(right_samples, limits)
                    .map_err(vector_error)?;
                kernel.distance_bounded(&x, &y, config.cutoff, limits)
            }
            4 => {
                if !config.gap_or_sentinel.is_null()
                    || config.parameter0 != 0.0
                    || config.parameter1 != 0.0
                {
                    return Err(invalid("unused vector DTW parameters must be zero"));
                }
                let kernel = VectorBandedDtwScorer::new(metric.metric.clone(), config.band);
                let x = kernel
                    .try_series(left_samples, limits)
                    .map_err(vector_error)?;
                let y = kernel
                    .try_series(right_samples, limits)
                    .map_err(vector_error)?;
                kernel.distance_bounded(&x, &y, config.cutoff, limits)
            }
            5 => {
                if !config.gap_or_sentinel.is_null()
                    || config.parameter0 != 0.0
                    || config.parameter1 != 0.0
                    || config.band != 0
                {
                    return Err(invalid("unused vector Fréchet parameters must be zero"));
                }
                let kernel = VectorFrechetMetric::new(metric.metric.clone());
                let x = VectorFrechetPath::try_new(left_samples, limits).map_err(vector_error)?;
                let y = VectorFrechetPath::try_new(right_samples, limits).map_err(vector_error)?;
                kernel.distance_bounded(&x, &y, config.cutoff, limits)
            }
            7 => {
                if config.band != 0 {
                    return Err(invalid("unused vector TWED band must be zero"));
                }
                let sentinel = VectorSample::try_new(
                    slice(config.gap_or_sentinel, dimension, "vector sentinel")?,
                    limits,
                )
                .map_err(vector_error)?;
                let unit = timestamp_unit(left.timestamp_unit)?;
                if unit != timestamp_unit(right.timestamp_unit)? {
                    return Err(invalid("vector TWED timestamp units differ"));
                }
                let left_times = slice(left.timestamps, left.sample_count, "left timestamps")?;
                let right_times = slice(right.timestamps, right.sample_count, "right timestamps")?;
                let kernel = VectorTimestampedTwedMetric::try_new(
                    metric.metric.clone(),
                    sentinel,
                    config.parameter0,
                    config.parameter1,
                )
                .map_err(vector_error)?;
                let x = kernel
                    .try_series(left_samples, left_times, unit, left.origin, limits)
                    .map_err(vector_error)?;
                let y = kernel
                    .try_series(right_samples, right_times, unit, right.origin, limits)
                    .map_err(vector_error)?;
                kernel.distance_bounded(&x, &y, config.cutoff, limits)
            }
            _ => unreachable!(),
        }
        .map_err(vector_error)?;
        publish(outcome, output);
        Ok(LlevStatus::Ok)
    })
}

/// Fixed-query online Fréchet automaton over whole vector points.
pub struct LlevVectorFrechetOnline {
    machine: VectorFrechetOnlineAutomaton<FixedChannelMetric>,
    dimension: usize,
    limits: ResourceLimits,
}

fn online_observation(value: VectorFrechetOnlineObservation) -> LlevTemporalOnlineObservation {
    LlevTemporalOnlineObservation {
        consumed_target_len: value.consumed_target_len,
        active_positions: value.active_positions,
        distance_within_cutoff: value.distance_within_cutoff.unwrap_or(0.0),
        minimum_active_cost: value.minimum_active_cost.unwrap_or(0.0),
        has_distance: u8::from(value.distance_within_cutoff.is_some()),
        has_minimum: u8::from(value.minimum_active_cost.is_some()),
        reserved: [0; 6],
    }
}

fn online_step_usage(
    observation: LlevTemporalOnlineObservation,
    kind: u32,
    reason: u32,
    usage: ResourceUsage,
) -> LlevTemporalOnlineStep {
    LlevTemporalOnlineStep {
        observation,
        kind,
        reason,
        dp_cells: usage.dp_cells,
        work_units: usage.work_units,
        scratch_bytes: usage.scratch_bytes,
        queue_entries: usage.queue_entries,
    }
}

/// Copy a typed query and construct a bounded online vector Fréchet machine.
///
/// # Safety
/// The metric and all input buffers must be live for this call; the output
/// must be writable and disjoint from all inputs.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_frechet_online_new(
    metric: *const LlevVectorMetric,
    query: *const LlevVectorSeriesView,
    cutoff: f64,
    raw_limits: *const LlevTemporalOnlineLimits,
    out_machine: *mut *mut LlevVectorFrechetOnline,
) -> LlevStatus {
    boundary(|| {
        let output = out_machine.as_mut().ok_or((
            LlevStatus::NullPointer,
            "vector online output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let metric = metric
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "vector metric is null".into()))?;
        let query = *query.as_ref().ok_or((
            LlevStatus::NullPointer,
            "vector online query is null".into(),
        ))?;
        let raw_limits = *raw_limits.as_ref().ok_or((
            LlevStatus::NullPointer,
            "vector online limits are null".into(),
        ))?;
        if !query.timestamps.is_null() || query.timestamp_unit != 0 || query.origin != 0.0 {
            return Err(invalid(
                "vector Fréchet online query must have no timestamps",
            ));
        }
        let dimension = metric.metric.channel_layout().dimension();
        let limits = ResourceLimits {
            max_series_len: raw_limits.max_query_len,
            max_dimension: dimension,
            max_scratch_bytes: raw_limits.max_scratch_bytes,
            ..ResourceLimits::default()
        };
        let query = VectorFrechetPath::try_new(
            samples(query, dimension, limits, "vector online query")?,
            limits,
        )
        .map_err(vector_error)?;
        let machine = VectorFrechetOnlineAutomaton::new(
            query,
            metric.metric.clone(),
            cutoff,
            OnlineAutomatonLimits::from(raw_limits),
        )
        .map_err(vector_error)?;
        *output = Box::into_raw(Box::new(LlevVectorFrechetOnline {
            machine,
            dimension,
            limits,
        }));
        Ok(LlevStatus::Ok)
    })
}

/// Observe the last committed target prefix.
///
/// # Safety
/// The machine must be live and exclusively accessed; output must be writable.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_frechet_online_observation(
    machine: *const LlevVectorFrechetOnline,
    out_observation: *mut LlevTemporalOnlineObservation,
) -> LlevStatus {
    boundary(|| {
        let output = out_observation.as_mut().ok_or((
            LlevStatus::NullPointer,
            "vector online observation output is null".into(),
        ))?;
        *output = LlevTemporalOnlineObservation::default();
        let machine = machine.as_ref().ok_or((
            LlevStatus::NullPointer,
            "vector online machine is null".into(),
        ))?;
        *output = online_observation(machine.machine.observation());
        Ok(LlevStatus::Ok)
    })
}

/// Consume one vector point transactionally. An incomplete step preserves
/// the prior observation.
///
/// # Safety
/// The machine must be live and exclusively accessed. The point must contain
/// exactly the metric dimension and be disjoint from writable output.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_frechet_online_advance(
    machine: *mut LlevVectorFrechetOnline,
    point: *const f64,
    point_len: usize,
    out_step: *mut LlevTemporalOnlineStep,
) -> LlevStatus {
    boundary(|| {
        let output = out_step.as_mut().ok_or((
            LlevStatus::NullPointer,
            "vector online step output is null".into(),
        ))?;
        *output = LlevTemporalOnlineStep::default();
        let machine = machine.as_mut().ok_or((
            LlevStatus::NullPointer,
            "vector online machine is null".into(),
        ))?;
        if point_len != machine.dimension {
            return Err(invalid("vector online point dimension mismatch"));
        }
        let point = VectorSample::try_new(
            slice(point, point_len, "vector online point")?,
            machine.limits,
        )
        .map_err(vector_error)?;
        *output = match machine.machine.advance(&point).map_err(vector_error)? {
            OnlineStepOutcome::Advanced { value, usage } => {
                online_step_usage(online_observation(value), 0, 0, usage)
            }
            OnlineStepOutcome::Incomplete { reason, usage } => online_step_usage(
                LlevTemporalOnlineObservation::default(),
                1,
                incomplete_code(reason),
                usage,
            ),
        };
        Ok(LlevStatus::Ok)
    })
}

/// Fixed logical storage retained independently of target-prefix length.
///
/// # Safety
/// The machine must be live and exclusively accessed; output must be writable.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_frechet_online_scratch_bytes(
    machine: *const LlevVectorFrechetOnline,
    out_bytes: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let output = out_bytes.as_mut().ok_or((
            LlevStatus::NullPointer,
            "vector online scratch output is null".into(),
        ))?;
        *output = 0;
        let machine = machine.as_ref().ok_or((
            LlevStatus::NullPointer,
            "vector online machine is null".into(),
        ))?;
        *output = machine.machine.scratch_bytes();
        Ok(LlevStatus::Ok)
    })
}

/// Release an online vector Fréchet machine.
///
/// # Safety
/// The pointer must be null or a live handle returned by
/// `llev_vector_frechet_online_new`.
#[no_mangle]
pub unsafe extern "C" fn llev_vector_frechet_online_free(machine: *mut LlevVectorFrechetOnline) {
    if !machine.is_null() {
        drop(Box::from_raw(machine));
    }
}
