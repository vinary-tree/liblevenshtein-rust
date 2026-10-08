//! Bounded scalar temporal metrics at the C boundary.
//!
//! Every request validates both series and reserves its complete dynamic-
//! programming budget before calling a native kernel. No partial distance is
//! published on invalid input, a budget failure, or numeric overflow.

use super::{
    index::{boundary, slice},
    LlevStatus,
};
use crate::time_series::{
    DtwConfig, ErpConfig, ExactDecision, FrechetConfig, IncompleteReason,
    MetricTimestampedTwedConfig, MsmConfig, OperationOutcome, ResourceKind, ResourceLedger,
    ResourceLimits, SoftDtwAnalysis, SoftDtwConfig, TemporalValidationError, TimestampUnit,
    TimestampedSeries, TimestampedTwedError, TwedConfig,
};

/// Scalar temporal kernel selected by `LlevTemporalConfig::algorithm`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlevTemporalAlgorithm {
    /// Move-split-merge distance.
    Msm = 1,
    /// Edit distance with real penalty.
    Erp = 2,
    /// Time warp edit distance on the unit time grid.
    Twed = 3,
    /// Banded dynamic time warping.
    Dtw = 4,
    /// Discrete Fréchet distance.
    Frechet = 5,
    /// Differentiable soft DTW loss.
    SoftDtw = 6,
}

impl TryFrom<u32> for LlevTemporalAlgorithm {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Msm),
            2 => Ok(Self::Erp),
            3 => Ok(Self::Twed),
            4 => Ok(Self::Dtw),
            5 => Ok(Self::Frechet),
            6 => Ok(Self::SoftDtw),
            _ => Err(()),
        }
    }
}

/// Algorithm parameters and an inclusive cutoff. Positive infinity requests
/// the full exact score. Unused parameters must be zero.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTemporalConfig {
    /// One of `LlevTemporalAlgorithm`.
    pub algorithm: u32,
    /// Must be zero.
    pub reserved: u32,
    /// MSM split/merge cost, ERP gap, TWED stiffness, or Soft-DTW gamma.
    pub parameter0: f64,
    /// TWED gap penalty; zero for other kernels.
    pub parameter1: f64,
    /// Explicit DTW band; zero for other kernels.
    pub band: usize,
    /// Inclusive maximum score, or positive infinity for the full result.
    pub cutoff: f64,
}

/// Hard limits for one scalar comparison. Every field has to be explicit.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTemporalLimits {
    /// Maximum samples in either series.
    pub max_series_len: usize,
    /// Maximum dynamic-programming cells charged by the operation.
    pub max_dp_cells: usize,
    /// Maximum logical work units charged by the operation.
    pub max_work_units: usize,
    /// Maximum temporary bytes reserved by the operation.
    pub max_scratch_bytes: usize,
}

impl Default for LlevTemporalLimits {
    fn default() -> Self {
        let limits = ResourceLimits::default();
        Self {
            max_series_len: limits.max_series_len,
            max_dp_cells: limits.max_dp_cells,
            max_work_units: limits.max_work_units,
            max_scratch_bytes: limits.max_scratch_bytes,
        }
    }
}

/// `kind`: 0 finite score, 1 above cutoff, 2 no finite alignment, 3 incomplete.
/// Incomplete results contain no score. `reason` identifies the limit class:
/// 1 DP cells, 2 work units, 3 scratch bytes, 4 arithmetic or numeric overflow.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevTemporalDistanceResult {
    /// Finite score when `kind` is zero; zero otherwise.
    pub value: f64,
    /// Zero finite, one above cutoff, two no path, three incomplete.
    pub kind: u32,
    /// Resource or overflow code when `kind` is three.
    pub reason: u32,
    /// Dynamic-programming cells charged before execution.
    pub dp_cells: usize,
    /// Work units charged before execution.
    pub work_units: usize,
    /// Temporary bytes reserved before execution.
    pub scratch_bytes: usize,
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

fn validation(error: TemporalValidationError) -> (LlevStatus, String) {
    match error {
        TemporalValidationError::SeriesTooLong { .. } => {
            (LlevStatus::LimitExceeded, error.to_string())
        }
        _ => invalid(error.to_string()),
    }
}

pub(crate) fn reason_code(reason: IncompleteReason) -> u32 {
    match reason {
        IncompleteReason::BudgetExceeded { resource, .. }
        | IncompleteReason::ArithmeticOverflow { resource } => match resource {
            ResourceKind::DpCells => 1,
            ResourceKind::WorkUnits => 2,
            ResourceKind::ScratchBytes => 3,
            _ => 4,
        },
        _ => 4,
    }
}

fn limits_from_c(raw: LlevTemporalLimits) -> ResourceLimits {
    ResourceLimits {
        max_series_len: raw.max_series_len,
        max_dp_cells: raw.max_dp_cells,
        max_work_units: raw.max_work_units,
        max_scratch_bytes: raw.max_scratch_bytes,
        ..ResourceLimits::default()
    }
}

/// Borrowed scalar samples and physical timestamps in one canonical unit.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTimestampedSeriesView {
    /// Scalar sample values.
    pub values: *const f64,
    /// Physical timestamps in the declared unit.
    pub timestamps: *const f64,
    /// Equal number of values and timestamps.
    pub len: usize,
    /// 1 seconds, 2 milliseconds, 3 microseconds, 4 nanoseconds.
    pub unit: u32,
    /// Must be zero.
    pub reserved: u32,
    /// Shared physical origin in the declared unit.
    pub origin: f64,
}

fn timestamp_unit(code: u32) -> Result<TimestampUnit, (LlevStatus, String)> {
    match code {
        1 => Ok(TimestampUnit::Seconds),
        2 => Ok(TimestampUnit::Milliseconds),
        3 => Ok(TimestampUnit::Microseconds),
        4 => Ok(TimestampUnit::Nanoseconds),
        _ => Err(invalid("unknown timestamp unit")),
    }
}

fn timestamp_error(error: TimestampedTwedError) -> (LlevStatus, String) {
    match error {
        TimestampedTwedError::InvalidSeries(error) => validation(error),
        TimestampedTwedError::Resource(reason) => (
            LlevStatus::LimitExceeded,
            format!("timestamped series resource: {reason:?}"),
        ),
        _ => invalid(error.to_string()),
    }
}

/// Compare explicit-timestamp TWED series under hard scalar resource limits.
///
/// # Safety
/// Both views, limits, and output must address valid, mutually disjoint
/// storage. Each nonempty value and timestamp buffer must be aligned and
/// readable for its declared length. Buffers are borrowed only for this call.
#[no_mangle]
pub unsafe extern "C" fn llev_timestamped_twed_distance(
    left: *const LlevTimestampedSeriesView,
    right: *const LlevTimestampedSeriesView,
    stiffness: f64,
    gap_penalty: f64,
    cutoff: f64,
    raw_limits: *const LlevTemporalLimits,
    out_result: *mut LlevTemporalDistanceResult,
) -> LlevStatus {
    boundary(|| {
        let output = out_result.as_mut().ok_or((
            LlevStatus::NullPointer,
            "timestamped TWED output is null".into(),
        ))?;
        *output = LlevTemporalDistanceResult::default();
        let left = *left.as_ref().ok_or((
            LlevStatus::NullPointer,
            "left timestamped series is null".into(),
        ))?;
        let right = *right.as_ref().ok_or((
            LlevStatus::NullPointer,
            "right timestamped series is null".into(),
        ))?;
        let raw_limits = *raw_limits.as_ref().ok_or((
            LlevStatus::NullPointer,
            "timestamped TWED limits are null".into(),
        ))?;
        if left.len > raw_limits.max_series_len || right.len > raw_limits.max_series_len {
            return Err((
                LlevStatus::LimitExceeded,
                "timestamped series length limit".into(),
            ));
        }
        if left.reserved != 0 || right.reserved != 0 {
            return Err(invalid("timestamped series reserved field must be zero"));
        }
        let left_unit = timestamp_unit(left.unit)?;
        let right_unit = timestamp_unit(right.unit)?;
        let limits = limits_from_c(raw_limits);
        let config = MetricTimestampedTwedConfig::try_new(stiffness, gap_penalty)
            .map_err(timestamp_error)?;
        // Both validated operands remain owned while the two-row recurrence
        // runs. Charge their combined storage together with its DP rows.
        let scratch_bytes = left
            .len
            .checked_add(right.len)
            .and_then(|length| length.checked_add(left.len.min(right.len)))
            .and_then(|length| length.checked_add(1))
            .and_then(|slots| slots.checked_mul(2 * std::mem::size_of::<f64>()));
        let Some(scratch_bytes) = scratch_bytes else {
            output.kind = 3;
            output.reason = 4;
            return Err((
                LlevStatus::LimitExceeded,
                "timestamped scratch accounting overflow".into(),
            ));
        };
        if scratch_bytes > raw_limits.max_scratch_bytes {
            output.kind = 3;
            output.reason = 3;
            output.scratch_bytes = scratch_bytes;
            return Err((
                LlevStatus::LimitExceeded,
                "timestamped scratch limit".into(),
            ));
        }
        let left = match TimestampedSeries::try_new_with_origin(
            slice(left.values, left.len, "left timestamped values")?,
            slice(left.timestamps, left.len, "left timestamps")?,
            left_unit,
            left.origin,
            limits,
        ) {
            Ok(series) => series,
            Err(TimestampedTwedError::Resource(reason)) => {
                output.kind = 3;
                output.reason = reason_code(reason);
                output.scratch_bytes = scratch_bytes;
                return Err((
                    LlevStatus::LimitExceeded,
                    "left timestamped series incomplete".into(),
                ));
            }
            Err(error) => return Err(timestamp_error(error)),
        };
        let right = match TimestampedSeries::try_new_with_origin(
            slice(right.values, right.len, "right timestamped values")?,
            slice(right.timestamps, right.len, "right timestamps")?,
            right_unit,
            right.origin,
            limits,
        ) {
            Ok(series) => series,
            Err(TimestampedTwedError::Resource(reason)) => {
                output.kind = 3;
                output.reason = reason_code(reason);
                output.scratch_bytes = scratch_bytes;
                return Err((
                    LlevStatus::LimitExceeded,
                    "right timestamped series incomplete".into(),
                ));
            }
            Err(error) => return Err(timestamp_error(error)),
        };
        match config
            .distance_bounded(&left, &right, cutoff, limits)
            .map_err(timestamp_error)?
        {
            OperationOutcome::Complete { value, usage } => {
                let (kind, distance) = match value {
                    ExactDecision::WithinCutoff { distance, .. } => (0, distance),
                    ExactDecision::AboveCutoff => (1, 0.0),
                    ExactDecision::NoFiniteAlignment => (2, 0.0),
                };
                *output = LlevTemporalDistanceResult {
                    value: distance,
                    kind,
                    dp_cells: usage.dp_cells,
                    work_units: usage.work_units,
                    scratch_bytes,
                    ..LlevTemporalDistanceResult::default()
                };
                Ok(LlevStatus::Ok)
            }
            OperationOutcome::Incomplete { reason, usage, .. } => {
                *output = LlevTemporalDistanceResult {
                    kind: 3,
                    reason: reason_code(reason),
                    dp_cells: usage.dp_cells,
                    work_units: usage.work_units,
                    scratch_bytes,
                    ..LlevTemporalDistanceResult::default()
                };
                Err((
                    LlevStatus::LimitExceeded,
                    "timestamped TWED incomplete".into(),
                ))
            }
        }
    })
}

/// Evaluate a complete Soft-DTW loss and both sample gradients. The caller
/// owns output buffers of at least the corresponding operand length. Neither
/// gradient buffer is written unless the operation completes. Incomplete
/// outcomes use the same result tag and reason codes as scalar comparisons.
///
/// # Safety
/// All nonempty buffers must be valid, aligned, and mutually disjoint. Inputs
/// and outputs remain borrowed only for this call.
#[no_mangle]
pub unsafe extern "C" fn llev_soft_dtw_gradient(
    left: *const f64,
    left_len: usize,
    right: *const f64,
    right_len: usize,
    gamma: f64,
    raw_limits: *const LlevTemporalLimits,
    left_gradient: *mut f64,
    left_capacity: usize,
    right_gradient: *mut f64,
    right_capacity: usize,
    out_result: *mut LlevTemporalDistanceResult,
) -> LlevStatus {
    boundary(|| {
        let output = out_result.as_mut().ok_or((
            LlevStatus::NullPointer,
            "Soft-DTW gradient result output is null".into(),
        ))?;
        *output = LlevTemporalDistanceResult::default();
        let raw_limits = *raw_limits.as_ref().ok_or((
            LlevStatus::NullPointer,
            "Soft-DTW gradient limits are null".into(),
        ))?;
        if left_len > raw_limits.max_series_len || right_len > raw_limits.max_series_len {
            return Err((
                LlevStatus::LimitExceeded,
                "Soft-DTW series length limit".into(),
            ));
        }
        if left_len == 0 || right_len == 0 {
            return Err(invalid("Soft-DTW gradient requires two nonempty series"));
        }
        if left_capacity < left_len || right_capacity < right_len {
            return Err(invalid("Soft-DTW gradient output capacity is too small"));
        }
        for (pointer, name) in [
            (left_gradient, "left Soft-DTW gradient output"),
            (right_gradient, "right Soft-DTW gradient output"),
        ] {
            if pointer.is_null() {
                return Err((LlevStatus::NullPointer, format!("{name} is null")));
            }
            if !(pointer as usize).is_multiple_of(std::mem::align_of::<f64>()) {
                return Err(invalid(format!("{name} is not aligned")));
            }
        }
        let left = slice(left, left_len, "left Soft-DTW series")?;
        let right = slice(right, right_len, "right Soft-DTW series")?;
        let config = SoftDtwConfig::try_new(gamma).map_err(|error| invalid(error.to_string()))?;
        match config
            .analyze_with_gradient_bounded(left, right, limits_from_c(raw_limits))
            .map_err(validation)?
        {
            OperationOutcome::Complete { value, usage } => {
                std::ptr::copy_nonoverlapping(
                    value.left_gradient.as_ptr(),
                    left_gradient,
                    left_len,
                );
                std::ptr::copy_nonoverlapping(
                    value.right_gradient.as_ptr(),
                    right_gradient,
                    right_len,
                );
                *output = LlevTemporalDistanceResult {
                    value: value.value,
                    dp_cells: usage.dp_cells,
                    work_units: usage.work_units,
                    scratch_bytes: usage.scratch_bytes,
                    ..LlevTemporalDistanceResult::default()
                };
                Ok(LlevStatus::Ok)
            }
            OperationOutcome::Incomplete { reason, usage, .. } => {
                *output = LlevTemporalDistanceResult {
                    kind: 3,
                    reason: reason_code(reason),
                    dp_cells: usage.dp_cells,
                    work_units: usage.work_units,
                    scratch_bytes: usage.scratch_bytes,
                    ..LlevTemporalDistanceResult::default()
                };
                Err((
                    LlevStatus::LimitExceeded,
                    "Soft-DTW gradient incomplete".into(),
                ))
            }
        }
    })
}

pub(crate) fn check_config(
    raw: LlevTemporalConfig,
    limits: ResourceLimits,
) -> Result<LlevTemporalAlgorithm, (LlevStatus, String)> {
    let algorithm = LlevTemporalAlgorithm::try_from(raw.algorithm)
        .map_err(|()| invalid("unknown temporal algorithm"))?;
    if raw.reserved != 0
        || raw.cutoff.is_nan()
        || raw.cutoff == f64::NEG_INFINITY
        || (raw.cutoff < 0.0 && algorithm != LlevTemporalAlgorithm::SoftDtw)
    {
        return Err(invalid("invalid temporal configuration or cutoff"));
    }
    match algorithm {
        LlevTemporalAlgorithm::Msm => {
            MsmConfig::try_new(raw.parameter0).map_err(|error| invalid(error.to_string()))?;
            if raw.parameter1 != 0.0 || raw.band != 0 {
                return Err(invalid("unused MSM parameters must be zero"));
            }
        }
        LlevTemporalAlgorithm::Erp => {
            if !raw.parameter0.is_finite() || raw.parameter1 != 0.0 || raw.band != 0 {
                return Err(invalid(
                    "ERP requires a finite gap and zero unused parameters",
                ));
            }
        }
        LlevTemporalAlgorithm::Twed => {
            if !raw.parameter0.is_finite()
                || raw.parameter0 < 0.0
                || !raw.parameter1.is_finite()
                || raw.parameter1 < 0.0
                || raw.band != 0
            {
                return Err(invalid(
                    "TWED requires finite nonnegative stiffness and gap penalty",
                ));
            }
        }
        LlevTemporalAlgorithm::Dtw => {
            if raw.parameter0 != 0.0 || raw.parameter1 != 0.0 || raw.band > limits.max_band_width {
                return Err(invalid(
                    "DTW requires zero scalar parameters and a bounded band",
                ));
            }
        }
        LlevTemporalAlgorithm::Frechet => {
            if raw.parameter0 != 0.0 || raw.parameter1 != 0.0 || raw.band != 0 {
                return Err(invalid("Frechet has no configuration parameters"));
            }
        }
        LlevTemporalAlgorithm::SoftDtw => {
            SoftDtwConfig::try_new(raw.parameter0).map_err(|error| invalid(error.to_string()))?;
            if raw.parameter1 != 0.0 || raw.band != 0 {
                return Err(invalid("unused Soft-DTW parameters must be zero"));
            }
        }
    }
    Ok(algorithm)
}

fn complete(value: Option<f64>, cutoff: f64, output: &mut LlevTemporalDistanceResult) {
    match value {
        Some(value) if value.is_finite() && value <= cutoff => {
            output.kind = 0;
            output.value = value;
        }
        Some(value) if value.is_finite() => output.kind = 1,
        None if cutoff.is_finite() => output.kind = 1,
        _ => {
            output.kind = 3;
            output.reason = 4;
        }
    }
}

/// Compare two finite scalar series through the selected native temporal kernel.
///
/// # Safety
/// Nonempty `left` and `right` arrays must address their declared number of
/// aligned doubles, `config` and `limits` must point to initialized values,
/// and `out_result` must point to distinct writable storage. No input may be
/// mutated or freed during this call.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_distance(
    left: *const f64,
    left_len: usize,
    right: *const f64,
    right_len: usize,
    config: *const LlevTemporalConfig,
    limits: *const LlevTemporalLimits,
    out_result: *mut LlevTemporalDistanceResult,
) -> LlevStatus {
    boundary(|| {
        if out_result.is_null() {
            return Err((LlevStatus::NullPointer, "temporal result is null".into()));
        }
        out_result.write(LlevTemporalDistanceResult::default());
        let config = config
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "temporal config is null".into()))?;
        let limits = limits
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "temporal limits are null".into()))?;
        let limits = limits_from_c(*limits);
        let algorithm = check_config(*config, limits)?;
        let left = slice(left, left_len, "left series")?;
        let right = slice(right, right_len, "right series")?;
        let mut ledger = ResourceLedger::new(limits);
        ledger
            .validate_finite_series(crate::time_series::Operand::Query, left)
            .map_err(validation)?;
        ledger
            .validate_finite_series(crate::time_series::Operand::Candidate, right)
            .map_err(validation)?;

        let one_empty = left.is_empty() != right.is_empty();
        if (algorithm == LlevTemporalAlgorithm::Dtw
            && (one_empty || left_len.abs_diff(right_len) > config.band))
            || (algorithm == LlevTemporalAlgorithm::Frechet && one_empty)
        {
            out_result.write(LlevTemporalDistanceResult {
                kind: 2,
                ..LlevTemporalDistanceResult::default()
            });
            return Ok(LlevStatus::Ok);
        }

        let full_cells = left_len.checked_mul(right_len).ok_or((
            LlevStatus::LimitExceeded,
            "temporal DP cell count overflows".into(),
        ))?;
        let cells = if algorithm == LlevTemporalAlgorithm::Dtw {
            let band_width = config
                .band
                .checked_mul(2)
                .and_then(|width| width.checked_add(1))
                .unwrap_or(usize::MAX);
            full_cells.min(
                left_len
                    .max(right_len)
                    .saturating_mul(left_len.min(right_len).min(band_width)),
            )
        } else {
            full_cells
        };
        let scratch = if cells == 0 {
            if matches!(
                algorithm,
                LlevTemporalAlgorithm::Erp | LlevTemporalAlgorithm::Twed
            ) {
                16
            } else {
                0
            }
        } else {
            left_len
                .min(right_len)
                .checked_add(1)
                .and_then(|rows| rows.checked_mul(2))
                .and_then(|rows| rows.checked_mul(8))
                .ok_or((
                    LlevStatus::LimitExceeded,
                    "temporal scratch size overflows".into(),
                ))?
        };
        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::DpCells, cells),
            (ResourceKind::WorkUnits, cells),
            (ResourceKind::ScratchBytes, scratch),
        ]) {
            let output = LlevTemporalDistanceResult {
                kind: 3,
                reason: reason_code(reason),
                ..LlevTemporalDistanceResult::default()
            };
            out_result.write(output);
            return Err((
                LlevStatus::LimitExceeded,
                "temporal resource limit exceeded".into(),
            ));
        }
        let usage = ledger.usage();
        let mut output = LlevTemporalDistanceResult {
            dp_cells: usage.dp_cells,
            work_units: usage.work_units,
            scratch_bytes: usage.scratch_bytes,
            ..LlevTemporalDistanceResult::default()
        };
        let cutoff = config.cutoff;
        match algorithm {
            LlevTemporalAlgorithm::Msm => {
                let metric = MsmConfig::try_new(config.parameter0)
                    .map_err(|error| invalid(error.to_string()))?;
                match metric
                    .distance_bounded(left, right, cutoff, limits)
                    .map_err(validation)?
                {
                    OperationOutcome::Complete { value, usage } => {
                        output.dp_cells = usage.dp_cells;
                        output.work_units = usage.work_units;
                        output.scratch_bytes = usage.scratch_bytes;
                        match value {
                            ExactDecision::WithinCutoff { distance, .. } => {
                                complete(Some(distance), cutoff, &mut output)
                            }
                            ExactDecision::AboveCutoff => output.kind = 1,
                            ExactDecision::NoFiniteAlignment => output.kind = 2,
                        }
                    }
                    OperationOutcome::Incomplete { reason, .. } => {
                        output.kind = 3;
                        output.reason = reason_code(reason);
                        out_result.write(output);
                        return Err((
                            LlevStatus::LimitExceeded,
                            "MSM did not complete within limits".into(),
                        ));
                    }
                }
            }
            LlevTemporalAlgorithm::Erp => complete(
                ErpConfig::new(config.parameter0).distance_with_cutoff(left, right, cutoff),
                cutoff,
                &mut output,
            ),
            LlevTemporalAlgorithm::Twed => complete(
                TwedConfig::new(config.parameter0, config.parameter1)
                    .distance_with_cutoff(left, right, cutoff),
                cutoff,
                &mut output,
            ),
            LlevTemporalAlgorithm::Dtw => {
                complete(
                    DtwConfig::new(config.band).distance_with_cutoff(left, right, cutoff),
                    cutoff,
                    &mut output,
                );
            }
            LlevTemporalAlgorithm::Frechet => {
                complete(
                    FrechetConfig::new().distance_with_cutoff(left, right, cutoff),
                    cutoff,
                    &mut output,
                );
            }
            LlevTemporalAlgorithm::SoftDtw => {
                let metric = SoftDtwConfig::try_new(config.parameter0)
                    .map_err(|error| invalid(error.to_string()))?;
                match metric
                    .analyze_bounded(left, right, limits)
                    .map_err(validation)?
                {
                    OperationOutcome::Complete { value, usage } => {
                        output.dp_cells = usage.dp_cells;
                        output.work_units = usage.work_units;
                        output.scratch_bytes = usage.scratch_bytes;
                        match value {
                            SoftDtwAnalysis::Finite { value } => {
                                complete(Some(value), cutoff, &mut output)
                            }
                            SoftDtwAnalysis::NoFiniteAlignment => output.kind = 2,
                        }
                    }
                    OperationOutcome::Incomplete { reason, .. } => {
                        output.kind = 3;
                        output.reason = reason_code(reason);
                        out_result.write(output);
                        return Err((
                            LlevStatus::LimitExceeded,
                            "Soft-DTW did not complete within limits".into(),
                        ));
                    }
                }
            }
        }
        out_result.write(output);
        if output.kind == 3 {
            Err((
                LlevStatus::LimitExceeded,
                "temporal numeric overflow".into(),
            ))
        } else {
            Ok(LlevStatus::Ok)
        }
    })
}
