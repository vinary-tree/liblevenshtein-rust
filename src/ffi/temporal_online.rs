//! Fixed-query online temporal automata through a bounded C handle.

use super::{
    index::{boundary, slice},
    temporal::{check_config, LlevTemporalAlgorithm, LlevTemporalConfig},
    temporal_index::incomplete_code,
    LlevStatus,
};
use crate::time_series::{
    DtwConfig, ElasticOnlineAutomaton, ElasticOnlineObservation, ErpConfig, ErpOnlineAutomaton,
    ErpOnlineObservation, FrechetConfig, MsmConfig, MsmKernel, OnlineAutomatonLimits,
    OnlineStepOutcome, ResourceLimits, ResourceUsage, TemporalAutomatonError,
    TemporalValidationError, TwedConfig,
};

/// Fixed query storage and per-step resource limits for an online machine.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTemporalOnlineLimits {
    /// Maximum copied query samples.
    pub max_query_len: usize,
    /// Maximum live frontier positions per generation.
    pub max_frontier_positions: usize,
    /// Maximum work charged by one target sample.
    pub max_step_work_units: usize,
    /// Maximum retained and construction-peak bytes.
    pub max_scratch_bytes: usize,
}

impl From<LlevTemporalOnlineLimits> for OnlineAutomatonLimits {
    fn from(value: LlevTemporalOnlineLimits) -> Self {
        Self {
            max_query_len: value.max_query_len,
            max_frontier_positions: value.max_frontier_positions,
            max_step_work_units: value.max_step_work_units,
            max_scratch_bytes: value.max_scratch_bytes,
        }
    }
}

/// Exact observation of the already committed target prefix.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevTemporalOnlineObservation {
    /// Number of target samples already committed.
    pub consumed_target_len: usize,
    /// Number of active canonical positions.
    pub active_positions: usize,
    /// Exact score when `has_distance` is one; DTW uses squared units.
    pub distance_within_cutoff: f64,
    /// Minimum live frontier score when `has_minimum` is one.
    pub minimum_active_cost: f64,
    /// One when the complete score is within the construction cutoff.
    pub has_distance: u8,
    /// One when a finite live frontier score exists.
    pub has_minimum: u8,
    /// Must be ignored by callers; always zero.
    pub reserved: [u8; 6],
}

impl From<ElasticOnlineObservation> for LlevTemporalOnlineObservation {
    fn from(value: ElasticOnlineObservation) -> Self {
        Self {
            consumed_target_len: value.consumed_target_len,
            active_positions: value.active_positions,
            distance_within_cutoff: value.distance_within_cutoff.unwrap_or(0.0),
            minimum_active_cost: value.minimum_active_cost.unwrap_or(0.0),
            has_distance: u8::from(value.distance_within_cutoff.is_some()),
            has_minimum: u8::from(value.minimum_active_cost.is_some()),
            reserved: [0; 6],
        }
    }
}

impl From<ErpOnlineObservation> for LlevTemporalOnlineObservation {
    fn from(value: ErpOnlineObservation) -> Self {
        Self {
            consumed_target_len: value.consumed_target_len,
            active_positions: value.active_positions,
            distance_within_cutoff: value.distance_within_cutoff.unwrap_or(0.0),
            minimum_active_cost: value.minimum_active_cost.unwrap_or(0.0),
            has_distance: u8::from(value.distance_within_cutoff.is_some()),
            has_minimum: u8::from(value.minimum_active_cost.is_some()),
            reserved: [0; 6],
        }
    }
}

/// `kind`: zero committed, one incomplete without consuming the sample.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevTemporalOnlineStep {
    /// Exact observation for an advanced step; zeroed on incomplete steps.
    pub observation: LlevTemporalOnlineObservation,
    /// Zero for advanced, one for incomplete.
    pub kind: u32,
    /// Native incomplete reason code when `kind` is one.
    pub reason: u32,
    /// Dynamic-programming cells charged in this step.
    pub dp_cells: usize,
    /// Work units charged in this step.
    pub work_units: usize,
    /// Peak scratch bytes charged in this step.
    pub scratch_bytes: usize,
    /// Live frontier positions after the step or its rollback.
    pub queue_entries: usize,
}

fn step<T: Into<LlevTemporalOnlineObservation>>(
    outcome: OnlineStepOutcome<T>,
) -> LlevTemporalOnlineStep {
    match outcome {
        OnlineStepOutcome::Advanced { value, usage } => step_with_usage(value.into(), 0, 0, usage),
        OnlineStepOutcome::Incomplete { reason, usage } => step_with_usage(
            LlevTemporalOnlineObservation::default(),
            1,
            incomplete_code(reason),
            usage,
        ),
    }
}

fn step_with_usage(
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

enum OnlineState {
    Msm(ElasticOnlineAutomaton<MsmKernel>),
    Erp(ErpOnlineAutomaton),
    Twed(ElasticOnlineAutomaton<TwedConfig>),
    Dtw(ElasticOnlineAutomaton<DtwConfig>),
    Frechet(ElasticOnlineAutomaton<FrechetConfig>),
}

impl OnlineState {
    fn observation(&self) -> LlevTemporalOnlineObservation {
        match self {
            Self::Msm(value) => value.observation().into(),
            Self::Erp(value) => value.observation().into(),
            Self::Twed(value) => value.observation().into(),
            Self::Dtw(value) => value.observation().into(),
            Self::Frechet(value) => value.observation().into(),
        }
    }

    fn advance(&mut self, sample: f64) -> Result<LlevTemporalOnlineStep, TemporalValidationError> {
        match self {
            Self::Msm(value) => value.advance(sample).map(step),
            Self::Erp(value) => value.advance(sample).map(step),
            Self::Twed(value) => value.advance(sample).map(step),
            Self::Dtw(value) => value.advance(sample).map(step),
            Self::Frechet(value) => value.advance(sample).map(step),
        }
    }

    fn scratch_bytes(&self) -> usize {
        match self {
            Self::Msm(value) => value.scratch_bytes(),
            Self::Erp(value) => value.scratch_bytes(),
            Self::Twed(value) => value.scratch_bytes(),
            Self::Dtw(value) => value.scratch_bytes(),
            Self::Frechet(value) => value.scratch_bytes(),
        }
    }
}

/// Opaque fixed-query online automaton. Calls require exclusive access.
pub struct LlevTemporalOnlineAutomaton {
    state: OnlineState,
}

fn validation(error: TemporalValidationError) -> (LlevStatus, String) {
    match error {
        TemporalValidationError::SeriesTooLong { .. } => {
            (LlevStatus::LimitExceeded, error.to_string())
        }
        _ => (LlevStatus::InvalidArgument, error.to_string()),
    }
}

fn construction(error: TemporalAutomatonError) -> (LlevStatus, String) {
    match error {
        TemporalAutomatonError::Validation(error) => validation(error),
        TemporalAutomatonError::Resource(reason) => (
            LlevStatus::LimitExceeded,
            format!("online temporal resource: {reason:?}"),
        ),
    }
}

/// Construct a bounded fixed-query online machine. Soft-DTW has no online
/// elastic automaton. Non-ERP kernels require a finite inclusive cutoff.
///
/// # Safety
/// All pointers must be valid and disjoint; query is copied during the call.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_online_new(
    query: *const f64,
    query_len: usize,
    config: *const LlevTemporalConfig,
    limits: *const LlevTemporalOnlineLimits,
    out_machine: *mut *mut LlevTemporalOnlineAutomaton,
) -> LlevStatus {
    boundary(|| {
        let output = out_machine.as_mut().ok_or((
            LlevStatus::NullPointer,
            "online temporal machine output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let raw = *config.as_ref().ok_or((
            LlevStatus::NullPointer,
            "online temporal config is null".into(),
        ))?;
        let raw_limits = *limits.as_ref().ok_or((
            LlevStatus::NullPointer,
            "online temporal limits are null".into(),
        ))?;
        if query_len > raw_limits.max_query_len {
            return Err((
                LlevStatus::LimitExceeded,
                "online query length limit".into(),
            ));
        }
        let query = slice(query, query_len, "online temporal query")?;
        let algorithm = check_config(raw, ResourceLimits::default())?;
        let limits = OnlineAutomatonLimits::from(raw_limits);
        let cutoff = raw.cutoff;
        let state = match algorithm {
            LlevTemporalAlgorithm::Msm => OnlineState::Msm(
                ElasticOnlineAutomaton::new(
                    query,
                    MsmKernel::new(MsmConfig::new(raw.parameter0)),
                    cutoff,
                    limits,
                )
                .map_err(construction)?,
            ),
            LlevTemporalAlgorithm::Erp => OnlineState::Erp(
                ErpOnlineAutomaton::new(query, ErpConfig::new(raw.parameter0), cutoff, limits)
                    .map_err(construction)?,
            ),
            LlevTemporalAlgorithm::Twed => OnlineState::Twed(
                ElasticOnlineAutomaton::new(
                    query,
                    TwedConfig::new(raw.parameter0, raw.parameter1),
                    cutoff,
                    limits,
                )
                .map_err(construction)?,
            ),
            LlevTemporalAlgorithm::Dtw => OnlineState::Dtw(
                ElasticOnlineAutomaton::new(query, DtwConfig::new(raw.band), cutoff, limits)
                    .map_err(construction)?,
            ),
            LlevTemporalAlgorithm::Frechet => OnlineState::Frechet(
                ElasticOnlineAutomaton::new(query, FrechetConfig::new(), cutoff, limits)
                    .map_err(construction)?,
            ),
            LlevTemporalAlgorithm::SoftDtw => {
                return Err((
                    LlevStatus::Unsupported,
                    "Soft-DTW has no online automaton".into(),
                ));
            }
        };
        *output = Box::into_raw(Box::new(LlevTemporalOnlineAutomaton { state }));
        Ok(LlevStatus::Ok)
    })
}

/// Observe the committed prefix without consuming a sample.
///
/// # Safety
/// `machine` must be live and exclusively accessed; output must be writable.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_online_observation(
    machine: *const LlevTemporalOnlineAutomaton,
    out_observation: *mut LlevTemporalOnlineObservation,
) -> LlevStatus {
    boundary(|| {
        let output = out_observation.as_mut().ok_or((
            LlevStatus::NullPointer,
            "online temporal observation output is null".into(),
        ))?;
        *output = LlevTemporalOnlineObservation::default();
        let machine = machine.as_ref().ok_or((
            LlevStatus::NullPointer,
            "online temporal machine is null".into(),
        ))?;
        *output = machine.state.observation();
        Ok(LlevStatus::Ok)
    })
}

/// Consume one finite target sample. Incomplete output leaves the prefix
/// unchanged and carries no observation; inspect the prior prefix separately.
///
/// # Safety
/// `machine` must be live and exclusively accessed; output must be writable.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_online_advance(
    machine: *mut LlevTemporalOnlineAutomaton,
    sample: f64,
    out_step: *mut LlevTemporalOnlineStep,
) -> LlevStatus {
    boundary(|| {
        let output = out_step.as_mut().ok_or((
            LlevStatus::NullPointer,
            "online temporal step output is null".into(),
        ))?;
        *output = LlevTemporalOnlineStep::default();
        let machine = machine.as_mut().ok_or((
            LlevStatus::NullPointer,
            "online temporal machine is null".into(),
        ))?;
        *output = machine.state.advance(sample).map_err(validation)?;
        Ok(LlevStatus::Ok)
    })
}

/// Report fixed logical bytes retained by the online machine.
///
/// # Safety
/// `machine` must be live and exclusively accessed; output must be writable.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_online_scratch_bytes(
    machine: *const LlevTemporalOnlineAutomaton,
    out_bytes: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let output = out_bytes.as_mut().ok_or((
            LlevStatus::NullPointer,
            "online temporal scratch output is null".into(),
        ))?;
        *output = 0;
        let machine = machine.as_ref().ok_or((
            LlevStatus::NullPointer,
            "online temporal machine is null".into(),
        ))?;
        *output = machine.state.scratch_bytes();
        Ok(LlevStatus::Ok)
    })
}

/// Release a live online machine exactly once.
///
/// # Safety
/// The pointer must come from `llev_temporal_online_new` and remain live.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_online_free(machine: *mut LlevTemporalOnlineAutomaton) {
    if !machine.is_null() {
        drop(Box::from_raw(machine));
    }
}
