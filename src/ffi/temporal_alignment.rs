//! Bounded, page-readable native temporal alignment witnesses.

use super::{
    index::{boundary, slice},
    temporal::{
        check_config, limits_from_c, reason_code, timestamp_error, timestamp_unit,
        LlevTemporalAlgorithm, LlevTemporalConfig, LlevTemporalLimits, LlevTimestampedSeriesView,
    },
    LlevStatus,
};
use crate::time_series::{
    DtwConfig, ErpConfig, ExactDecision, FrechetConfig, IncompleteReason,
    MetricTimestampedTwedConfig, MsmAlignmentStep, MsmAlignmentWitness, MsmConfig,
    OperationOutcome, ResourceKind, ResourceLimits, TemporalAlignmentWitness,
    TemporalValidationError, TimestampedSeries, TwedConfig,
};

/// Scalar limits plus an explicit peak witness-storage ceiling.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTemporalAlignmentLimits {
    /// Scalar sample, DP, work, and scratch ceilings.
    pub temporal: LlevTemporalLimits,
    /// Peak witness storage ceiling.
    pub max_witness_bytes: usize,
}

/// One page-readable operation. `flags` bit 0/1 says the corresponding
/// endpoint is present. MSM uses only `operation` (1 move, 2 merge, 3 split).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevTemporalAlignmentStep {
    /// Kernel-specific operation tag.
    pub operation: u32,
    /// Bit 0: query endpoint present; bit 1: candidate endpoint present.
    pub flags: u32,
    /// Zero-based query endpoint when flag bit 0 is set.
    pub query_endpoint: u64,
    /// Zero-based candidate endpoint when flag bit 1 is set.
    pub candidate_endpoint: u64,
    /// Exact local-cost bits for non-MSM witnesses.
    pub local_cost_bits: u64,
}

/// Kind 0 finite witness, 1 above cutoff, 2 no finite alignment, 3 incomplete.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevTemporalAlignmentOutcome {
    /// 0 finite, 1 above cutoff, 2 no alignment, 3 incomplete.
    pub kind: u32,
    /// Resource reason when incomplete.
    pub reason: u32,
    /// Exact distance only when kind is zero.
    pub distance: f64,
    /// Number of operations in the returned witness.
    pub step_count: usize,
    /// Charged dynamic-programming cells.
    pub dp_cells: usize,
    /// Charged logical work units.
    pub work_units: usize,
    /// Peak scratch bytes.
    pub scratch_bytes: usize,
    /// Peak witness bytes.
    pub witness_bytes: usize,
}

enum Witness {
    Msm(MsmAlignmentWitness),
    Temporal(TemporalAlignmentWitness),
}

/// Opaque immutable witness. A caller may page it after inputs are released.
pub struct LlevTemporalAlignment {
    witness: Witness,
    config: LlevTemporalConfig,
    timestamped: bool,
    max_series_len: usize,
}

impl LlevTemporalAlignment {
    fn len(&self) -> usize {
        match &self.witness {
            Witness::Msm(witness) => witness.len(),
            Witness::Temporal(witness) => witness.len(),
        }
    }
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

fn limits_from_raw(raw: LlevTemporalAlignmentLimits) -> ResourceLimits {
    let mut limits = limits_from_c(raw.temporal);
    limits.max_witness_bytes = raw.max_witness_bytes;
    limits
}

fn alignment_reason_code(reason: IncompleteReason) -> u32 {
    match reason {
        IncompleteReason::BudgetExceeded {
            resource: ResourceKind::WitnessBytes,
            ..
        }
        | IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::WitnessBytes,
        } => 5,
        _ => reason_code(reason),
    }
}

fn publish<W>(
    outcome: OperationOutcome<ExactDecision<W>>,
    config: LlevTemporalConfig,
    timestamped: bool,
    max_series_len: usize,
    wrap: impl FnOnce(W) -> Witness,
    handle: &mut *mut LlevTemporalAlignment,
    output: &mut LlevTemporalAlignmentOutcome,
) {
    let usage = outcome.usage();
    output.dp_cells = usage.dp_cells;
    output.work_units = usage.work_units;
    output.scratch_bytes = usage.scratch_bytes;
    output.witness_bytes = usage.witness_bytes;
    match outcome {
        OperationOutcome::Complete { value, .. } => match value {
            ExactDecision::WithinCutoff { distance, witness } => {
                let owned = Box::new(LlevTemporalAlignment {
                    witness: wrap(witness),
                    config,
                    timestamped,
                    max_series_len,
                });
                output.kind = 0;
                output.distance = distance;
                output.step_count = owned.len();
                *handle = Box::into_raw(owned);
            }
            ExactDecision::AboveCutoff => output.kind = 1,
            ExactDecision::NoFiniteAlignment => output.kind = 2,
        },
        OperationOutcome::Incomplete { reason, .. } => {
            output.kind = 3;
            output.reason = alignment_reason_code(reason);
        }
    }
}

/// Extract a scalar MSM, ERP, unit-grid TWED, banded DTW, or Fréchet witness.
///
/// # Safety
/// All pointers must be valid, properly aligned, and disjoint. Samples are
/// borrowed only during this call. The returned handle is owned by the caller.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_alignment_new(
    query: *const f64,
    query_len: usize,
    candidate: *const f64,
    candidate_len: usize,
    raw_config: *const LlevTemporalConfig,
    raw_limits: *const LlevTemporalAlignmentLimits,
    out_alignment: *mut *mut LlevTemporalAlignment,
    out_outcome: *mut LlevTemporalAlignmentOutcome,
) -> LlevStatus {
    boundary(|| {
        let handle = out_alignment.as_mut().ok_or((
            LlevStatus::NullPointer,
            "alignment output handle is null".into(),
        ))?;
        *handle = std::ptr::null_mut();
        let output = out_outcome
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "alignment outcome is null".into()))?;
        *output = LlevTemporalAlignmentOutcome::default();
        let config = *raw_config
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "alignment config is null".into()))?;
        let limits = limits_from_raw(
            *raw_limits
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "alignment limits are null".into()))?,
        );
        let algorithm = check_config(config, limits)?;
        if algorithm == LlevTemporalAlgorithm::SoftDtw {
            return Err(invalid("Soft-DTW does not have an alignment witness"));
        }
        if query_len > limits.max_series_len || candidate_len > limits.max_series_len {
            return Err((
                LlevStatus::LimitExceeded,
                "alignment series length limit".into(),
            ));
        }
        let query = slice(query, query_len, "alignment query")?;
        let candidate = slice(candidate, candidate_len, "alignment candidate")?;
        match algorithm {
            LlevTemporalAlgorithm::Msm => publish(
                MsmConfig::try_new(config.parameter0)
                    .map_err(|error| invalid(error.to_string()))?
                    .distance_with_alignment_bounded(query, candidate, config.cutoff, limits)
                    .map_err(validation)?,
                config,
                false,
                limits.max_series_len,
                Witness::Msm,
                handle,
                output,
            ),
            LlevTemporalAlgorithm::Erp => publish(
                ErpConfig::new(config.parameter0)
                    .distance_with_alignment_bounded(query, candidate, config.cutoff, limits)
                    .map_err(validation)?,
                config,
                false,
                limits.max_series_len,
                Witness::Temporal,
                handle,
                output,
            ),
            LlevTemporalAlgorithm::Twed => publish(
                TwedConfig::new(config.parameter0, config.parameter1)
                    .distance_with_alignment_bounded(query, candidate, config.cutoff, limits)
                    .map_err(validation)?,
                config,
                false,
                limits.max_series_len,
                Witness::Temporal,
                handle,
                output,
            ),
            LlevTemporalAlgorithm::Dtw => publish(
                DtwConfig::new(config.band)
                    .distance_with_alignment_bounded(query, candidate, config.cutoff, limits)
                    .map_err(validation)?,
                config,
                false,
                limits.max_series_len,
                Witness::Temporal,
                handle,
                output,
            ),
            LlevTemporalAlgorithm::Frechet => publish(
                FrechetConfig::new()
                    .distance_with_alignment_bounded(query, candidate, config.cutoff, limits)
                    .map_err(validation)?,
                config,
                false,
                limits.max_series_len,
                Witness::Temporal,
                handle,
                output,
            ),
            LlevTemporalAlgorithm::SoftDtw => unreachable!(),
        }
        Ok(LlevStatus::Ok)
    })
}

unsafe fn timestamped_series(
    raw: LlevTimestampedSeriesView,
    limits: ResourceLimits,
) -> Result<TimestampedSeries, (LlevStatus, String)> {
    if raw.reserved != 0 {
        return Err(invalid("timestamped alignment reserved field must be zero"));
    }
    if raw.len > limits.max_series_len {
        return Err((
            LlevStatus::LimitExceeded,
            "timestamped series length limit".into(),
        ));
    }
    TimestampedSeries::try_new_with_origin(
        slice(raw.values, raw.len, "timestamped alignment values")?,
        slice(raw.timestamps, raw.len, "timestamped alignment timestamps")?,
        timestamp_unit(raw.unit)?,
        raw.origin,
        limits,
    )
    .map_err(timestamp_error)
}

/// Extract a metric physical-time TWED witness from validated input copies.
///
/// # Safety
/// Both views and their buffers, limits, and outputs must be valid and
/// disjoint. Returned handle ownership transfers to the caller.
#[no_mangle]
pub unsafe extern "C" fn llev_timestamped_twed_alignment_new(
    raw_query: *const LlevTimestampedSeriesView,
    raw_candidate: *const LlevTimestampedSeriesView,
    stiffness: f64,
    gap_penalty: f64,
    cutoff: f64,
    raw_limits: *const LlevTemporalAlignmentLimits,
    out_alignment: *mut *mut LlevTemporalAlignment,
    out_outcome: *mut LlevTemporalAlignmentOutcome,
) -> LlevStatus {
    boundary(|| {
        let handle = out_alignment.as_mut().ok_or((
            LlevStatus::NullPointer,
            "timestamped alignment handle output is null".into(),
        ))?;
        *handle = std::ptr::null_mut();
        let output = out_outcome.as_mut().ok_or((
            LlevStatus::NullPointer,
            "timestamped alignment outcome is null".into(),
        ))?;
        *output = LlevTemporalAlignmentOutcome::default();
        let limits = limits_from_raw(*raw_limits.as_ref().ok_or((
            LlevStatus::NullPointer,
            "timestamped alignment limits are null".into(),
        ))?);
        let query = timestamped_series(
            *raw_query.as_ref().ok_or((
                LlevStatus::NullPointer,
                "timestamped alignment query is null".into(),
            ))?,
            limits,
        )?;
        let candidate = timestamped_series(
            *raw_candidate.as_ref().ok_or((
                LlevStatus::NullPointer,
                "timestamped alignment candidate is null".into(),
            ))?,
            limits,
        )?;
        let config = MetricTimestampedTwedConfig::try_new(stiffness, gap_penalty)
            .map_err(timestamp_error)?;
        let raw_config = LlevTemporalConfig {
            algorithm: 7,
            reserved: 0,
            parameter0: stiffness,
            parameter1: gap_penalty,
            band: 0,
            cutoff,
        };
        publish(
            config
                .distance_with_alignment_bounded(&query, &candidate, cutoff, limits)
                .map_err(timestamp_error)?,
            raw_config,
            true,
            limits.max_series_len,
            Witness::Temporal,
            handle,
            output,
        );
        Ok(LlevStatus::Ok)
    })
}

/// Copy one page of stable witness operations. The returned handle remains
/// immutable, and a page may be read concurrently by distinct callers.
///
/// # Safety
/// The handle must be live; outputs must be valid and disjoint. `out_steps`
/// has `capacity` writable records when capacity is nonzero.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_alignment_page(
    alignment: *const LlevTemporalAlignment,
    start: usize,
    out_steps: *mut LlevTemporalAlignmentStep,
    capacity: usize,
    out_written: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let written = out_written.as_mut().ok_or((
            LlevStatus::NullPointer,
            "alignment page count output is null".into(),
        ))?;
        *written = 0;
        let alignment = alignment
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "alignment handle is null".into()))?;
        if start > alignment.len() {
            return Err(invalid("alignment page start exceeds witness length"));
        }
        if capacity != 0 {
            if out_steps.is_null() {
                return Err((
                    LlevStatus::NullPointer,
                    "alignment page output is null".into(),
                ));
            }
            if !(out_steps as usize)
                .is_multiple_of(std::mem::align_of::<LlevTemporalAlignmentStep>())
            {
                return Err(invalid("alignment page output is not aligned"));
            }
        }
        let count = capacity.min(alignment.len() - start);
        match &alignment.witness {
            Witness::Msm(witness) => {
                for (offset, step) in witness.steps()[start..start + count].iter().enumerate() {
                    out_steps.add(offset).write(LlevTemporalAlignmentStep {
                        operation: match step {
                            MsmAlignmentStep::Move => 1,
                            MsmAlignmentStep::Merge => 2,
                            MsmAlignmentStep::Split => 3,
                        },
                        ..LlevTemporalAlignmentStep::default()
                    });
                }
            }
            Witness::Temporal(witness) => {
                for (offset, step) in witness.steps()[start..start + count].iter().enumerate() {
                    let query = step.query_endpoint();
                    let candidate = step.candidate_endpoint();
                    out_steps.add(offset).write(LlevTemporalAlignmentStep {
                        operation: step.operation() as u32,
                        flags: u32::from(query.is_some()) | (u32::from(candidate.is_some()) << 1),
                        query_endpoint: query.unwrap_or(0),
                        candidate_endpoint: candidate.unwrap_or(0),
                        local_cost_bits: step.local_cost_bits(),
                    });
                }
            }
        }
        *written = count;
        Ok(LlevStatus::Ok)
    })
}

/// Replay the witness against scalar operands and the captured configuration.
///
/// # Safety
/// Handle and input pointers must be live and valid for the duration.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_alignment_replay(
    alignment: *const LlevTemporalAlignment,
    query: *const f64,
    query_len: usize,
    candidate: *const f64,
    candidate_len: usize,
    out_distance: *mut f64,
) -> LlevStatus {
    boundary(|| {
        let output = out_distance.as_mut().ok_or((
            LlevStatus::NullPointer,
            "alignment replay output is null".into(),
        ))?;
        *output = 0.0;
        let alignment = alignment
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "alignment handle is null".into()))?;
        if alignment.timestamped {
            return Err(invalid("timestamped witness needs timestamped replay"));
        }
        if query_len > alignment.max_series_len || candidate_len > alignment.max_series_len {
            return Err((
                LlevStatus::LimitExceeded,
                "alignment replay series limit".into(),
            ));
        }
        let query = slice(query, query_len, "alignment replay query")?;
        let candidate = slice(candidate, candidate_len, "alignment replay candidate")?;
        let raw = alignment.config;
        *output = match &alignment.witness {
            Witness::Msm(witness) => witness
                .replay(
                    query,
                    candidate,
                    &MsmConfig::try_new(raw.parameter0)
                        .map_err(|error| invalid(error.to_string()))?,
                )
                .map_err(|error| invalid(error.to_string()))?,
            Witness::Temporal(witness) => match LlevTemporalAlgorithm::try_from(raw.algorithm) {
                Ok(LlevTemporalAlgorithm::Erp) => {
                    witness.replay_erp(query, candidate, &ErpConfig::new(raw.parameter0))
                }
                Ok(LlevTemporalAlgorithm::Twed) => witness.replay_unit_grid_twed(
                    query,
                    candidate,
                    &TwedConfig::new(raw.parameter0, raw.parameter1),
                ),
                Ok(LlevTemporalAlgorithm::Dtw) => {
                    witness.replay_banded_dtw(query, candidate, &DtwConfig::new(raw.band))
                }
                Ok(LlevTemporalAlgorithm::Frechet) => {
                    witness.replay_discrete_frechet(query, candidate, &FrechetConfig::new())
                }
                _ => return Err(invalid("unsupported temporal alignment kind")),
            }
            .map_err(|error| invalid(error.to_string()))?,
        };
        Ok(LlevStatus::Ok)
    })
}

/// Replay a physical-time TWED witness against supplied timestamped series.
///
/// # Safety
/// Handle, views, buffers, and output must be live, valid, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn llev_timestamped_twed_alignment_replay(
    alignment: *const LlevTemporalAlignment,
    raw_query: *const LlevTimestampedSeriesView,
    raw_candidate: *const LlevTimestampedSeriesView,
    out_distance: *mut f64,
) -> LlevStatus {
    boundary(|| {
        let output = out_distance.as_mut().ok_or((
            LlevStatus::NullPointer,
            "timestamped alignment replay output is null".into(),
        ))?;
        *output = 0.0;
        let alignment = alignment
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "alignment handle is null".into()))?;
        if !alignment.timestamped {
            return Err(invalid("scalar witness needs scalar replay"));
        }
        let limits = ResourceLimits {
            max_series_len: alignment.max_series_len,
            ..ResourceLimits::default()
        };
        let query = timestamped_series(
            *raw_query.as_ref().ok_or((
                LlevStatus::NullPointer,
                "timestamped replay query is null".into(),
            ))?,
            limits,
        )?;
        let candidate = timestamped_series(
            *raw_candidate.as_ref().ok_or((
                LlevStatus::NullPointer,
                "timestamped replay candidate is null".into(),
            ))?,
            limits,
        )?;
        let raw = alignment.config;
        let config = MetricTimestampedTwedConfig::try_new(raw.parameter0, raw.parameter1)
            .map_err(timestamp_error)?;
        let Witness::Temporal(witness) = &alignment.witness else {
            return Err(invalid("timestamped witness type mismatch"));
        };
        *output = witness
            .replay_timestamped_twed(&query, &candidate, &config)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(LlevStatus::Ok)
    })
}

/// Free an owned witness handle once.
///
/// # Safety
/// Pointer must be a live handle returned by an alignment constructor.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_alignment_free(alignment: *mut LlevTemporalAlignment) {
    if !alignment.is_null() {
        drop(Box::from_raw(alignment));
    }
}
