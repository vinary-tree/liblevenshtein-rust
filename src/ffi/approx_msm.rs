//! Strict C boundary for advisory and exhaustive approximate MSM kNN.

use super::{
    index::{boundary, slice},
    temporal_index::{incomplete_code, LlevTemporalSearchLimits},
    LlevStatus,
};
use crate::time_series::{
    ApproxMsmConfig, ApproxMsmIndex, ApproxMsmSearchOutcome, ApproxMsmSearchResult, MsmConfig,
    ResourceLimits, ResourceUsage, TemporalValidationError,
};

/// PAA and exact-MSM parameters plus hard retained-storage ceilings.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevApproxMsmIndexConfig {
    /// PAA segment count; zero ranks by length alone.
    pub segments: usize,
    /// Maximum feature-ranked entries reranked exactly; zero uses k.
    pub candidate_limit: usize,
    /// Finite nonnegative MSM split/merge cost.
    pub split_merge_cost: f64,
    /// Maximum indexed episodes.
    pub max_entries: usize,
    /// Maximum scalar samples across indexed episodes.
    pub max_total_samples: usize,
    /// Maximum samples in one indexed episode.
    pub max_series_len: usize,
    /// Maximum retained PAA feature values across indexed episodes.
    pub max_total_features: usize,
}

/// One exact MSM distance, tied deterministically by insertion position.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevApproxMsmNeighbor {
    /// Caller metadata identifier.
    pub id: u64,
    /// Zero-based stable insertion position.
    pub insertion_index: usize,
    /// Exact MSM distance after feature selection.
    pub distance: f64,
}

/// Tagged bounded outcome. Kind 1 proves recall; kind 2 is advisory; kind 3
/// is incomplete. An empty advisory list makes no absence claim.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevApproxMsmOutcome {
    /// 1 exhaustive, 2 advisory, 3 incomplete.
    pub kind: u32,
    /// Temporal incompletion reason code when kind is 3; otherwise zero.
    pub reason: u32,
    /// Number of exact neighbors written, including a possible partial list.
    pub neighbor_count: usize,
    /// Indexed episodes in the captured revision.
    pub indexed_entries: usize,
    /// Episodes admitted by feature ranking.
    pub candidate_entries: usize,
    /// Admitted episodes decided by exact MSM.
    pub exact_reranked: usize,
    /// Charged cumulative DP cells.
    pub dp_cells: usize,
    /// Charged cumulative logical work units.
    pub work_units: usize,
    /// Peak scratch bytes.
    pub scratch_bytes: usize,
    /// Charged full-precision candidate inspections.
    pub candidates: usize,
    /// Charged retained exact results.
    pub results: usize,
}

/// Opaque mutable builder, then immutable frozen query source.
pub struct LlevApproxMsmIndex {
    native: ApproxMsmIndex<u64>,
    frozen: bool,
    segments: usize,
    max_entries: usize,
    max_total_samples: usize,
    max_series_len: usize,
    max_total_features: usize,
    total_samples: usize,
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

/// Create a bounded approximate MSM builder.
///
/// # Safety
/// The config and output pointers must be valid and disjoint. The output
/// receives one owning handle on success.
#[no_mangle]
pub unsafe extern "C" fn llev_approx_msm_index_new(
    config: *const LlevApproxMsmIndexConfig,
    out_index: *mut *mut LlevApproxMsmIndex,
) -> LlevStatus {
    boundary(|| {
        let output = out_index.as_mut().ok_or((
            LlevStatus::NullPointer,
            "approximate MSM index output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let raw = *config.as_ref().ok_or((
            LlevStatus::NullPointer,
            "approximate MSM config is null".into(),
        ))?;
        let msm =
            MsmConfig::try_new(raw.split_merge_cost).map_err(|error| invalid(error.to_string()))?;
        let config =
            ApproxMsmConfig::try_new(raw.segments, raw.candidate_limit, msm).map_err(validation)?;
        *output = Box::into_raw(Box::new(LlevApproxMsmIndex {
            native: ApproxMsmIndex::new(config),
            frozen: false,
            segments: raw.segments,
            max_entries: raw.max_entries,
            max_total_samples: raw.max_total_samples,
            max_series_len: raw.max_series_len,
            max_total_features: raw.max_total_features,
            total_samples: 0,
        }));
        Ok(LlevStatus::Ok)
    })
}

/// Copy one finite series into the index and return its insertion position.
///
/// # Safety
/// The index must be live and exclusively accessible. Input and output
/// storage must be valid and disjoint; samples are borrowed only here.
#[no_mangle]
pub unsafe extern "C" fn llev_approx_msm_index_insert(
    index: *mut LlevApproxMsmIndex,
    id: u64,
    samples: *const f64,
    len: usize,
    out_position: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let output = out_position.as_mut().ok_or((
            LlevStatus::NullPointer,
            "approximate MSM insertion position output is null".into(),
        ))?;
        *output = 0;
        let handle = index.as_mut().ok_or((
            LlevStatus::NullPointer,
            "approximate MSM index is null".into(),
        ))?;
        if handle.frozen {
            return Err(invalid("approximate MSM index is frozen"));
        }
        let next_entries = handle.native.len().checked_add(1).ok_or((
            LlevStatus::LimitExceeded,
            "approximate MSM entry count overflow".into(),
        ))?;
        let next_samples = handle.total_samples.checked_add(len).ok_or((
            LlevStatus::LimitExceeded,
            "approximate MSM sample count overflow".into(),
        ))?;
        let next_features = next_entries.checked_mul(handle.segments).ok_or((
            LlevStatus::LimitExceeded,
            "approximate MSM feature count overflow".into(),
        ))?;
        if next_entries > handle.max_entries
            || len > handle.max_series_len
            || next_samples > handle.max_total_samples
            || next_features > handle.max_total_features
        {
            return Err((
                LlevStatus::LimitExceeded,
                "approximate MSM ingestion limit".into(),
            ));
        }
        let samples = slice(samples, len, "approximate MSM samples")?;
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(invalid("approximate MSM samples must be finite"));
        }
        handle.native.insert(id, samples);
        handle.total_samples = next_samples;
        *output = next_entries - 1;
        Ok(LlevStatus::Ok)
    })
}

/// Seal an index for immutable concurrent queries.
///
/// # Safety
/// The index must be live and exclusively accessible during this call.
#[no_mangle]
pub unsafe extern "C" fn llev_approx_msm_index_freeze(
    index: *mut LlevApproxMsmIndex,
) -> LlevStatus {
    boundary(|| {
        let handle = index.as_mut().ok_or((
            LlevStatus::NullPointer,
            "approximate MSM index is null".into(),
        ))?;
        handle.frozen = true;
        Ok(LlevStatus::Ok)
    })
}

fn copy_result(
    kind: u32,
    reason: u32,
    result: Option<&ApproxMsmSearchResult<'_, u64>>,
    usage: ResourceUsage,
    output: &mut [LlevApproxMsmNeighbor],
) -> Result<LlevApproxMsmOutcome, (LlevStatus, String)> {
    let mut raw = LlevApproxMsmOutcome {
        kind,
        reason,
        dp_cells: usage.dp_cells,
        work_units: usage.work_units,
        scratch_bytes: usage.scratch_bytes,
        candidates: usage.candidates,
        results: usage.results,
        ..LlevApproxMsmOutcome::default()
    };
    if let Some(result) = result {
        raw.indexed_entries = result.coverage.indexed_entries;
        raw.candidate_entries = result.coverage.candidate_entries;
        raw.exact_reranked = result.coverage.exact_reranked;
        if result.neighbors.len() > output.len() {
            return Err((
                LlevStatus::LimitExceeded,
                "approximate MSM output capacity exceeded".into(),
            ));
        }
        for (slot, neighbor) in output.iter_mut().zip(&result.neighbors) {
            *slot = LlevApproxMsmNeighbor {
                id: *neighbor.value,
                insertion_index: neighbor.index,
                distance: neighbor.distance,
            };
        }
        raw.neighbor_count = result.neighbors.len();
    }
    Ok(raw)
}

/// Run strict bounded PAA selection and exact MSM reranking. Outcome kind 1
/// proves recall; kind 2 is advisory; kind 3 may carry an exact partial list.
///
/// # Safety
/// The index must be live and frozen. Query and output pointers must be
/// valid and disjoint. `out_neighbors` has `capacity` writable elements when
/// capacity is nonzero. Output storage remains caller owned.
#[no_mangle]
pub unsafe extern "C" fn llev_approx_msm_index_query_knn(
    index: *const LlevApproxMsmIndex,
    query: *const f64,
    query_len: usize,
    k: usize,
    raw_limits: *const LlevTemporalSearchLimits,
    out_neighbors: *mut LlevApproxMsmNeighbor,
    capacity: usize,
    out_outcome: *mut LlevApproxMsmOutcome,
) -> LlevStatus {
    boundary(|| {
        let output = out_outcome.as_mut().ok_or((
            LlevStatus::NullPointer,
            "approximate MSM outcome output is null".into(),
        ))?;
        *output = LlevApproxMsmOutcome::default();
        let handle = index.as_ref().ok_or((
            LlevStatus::NullPointer,
            "approximate MSM index is null".into(),
        ))?;
        if !handle.frozen {
            return Err(invalid("approximate MSM index must be frozen"));
        }
        let raw_limits = *raw_limits.as_ref().ok_or((
            LlevStatus::NullPointer,
            "approximate MSM limits are null".into(),
        ))?;
        if capacity < k.min(handle.native.len()) {
            return Err((
                LlevStatus::LimitExceeded,
                "approximate MSM output capacity is smaller than k".into(),
            ));
        }
        if capacity != 0 {
            if out_neighbors.is_null() {
                return Err((
                    LlevStatus::NullPointer,
                    "approximate MSM neighbor output is null".into(),
                ));
            }
            if !(out_neighbors as usize)
                .is_multiple_of(std::mem::align_of::<LlevApproxMsmNeighbor>())
            {
                return Err(invalid("approximate MSM neighbor output is not aligned"));
            }
        }
        let query = slice(query, query_len, "approximate MSM query")?;
        let neighbors = if capacity == 0 {
            &mut []
        } else {
            std::slice::from_raw_parts_mut(out_neighbors, capacity)
        };
        let outcome = handle
            .native
            .search_knn_bounded(query, k, ResourceLimits::from(raw_limits))
            .map_err(validation)?;
        *output = match outcome {
            ApproxMsmSearchOutcome::Exhaustive { result, usage } => {
                copy_result(1, 0, Some(&result), usage, neighbors)?
            }
            ApproxMsmSearchOutcome::Advisory { result, usage } => {
                copy_result(2, 0, Some(&result), usage, neighbors)?
            }
            ApproxMsmSearchOutcome::Incomplete {
                partial,
                reason,
                usage,
            } => copy_result(
                3,
                incomplete_code(reason),
                partial.as_ref(),
                usage,
                neighbors,
            )?,
        };
        output.indexed_entries = handle.native.len();
        Ok(LlevStatus::Ok)
    })
}

/// Release the owning index handle.
///
/// # Safety
/// The pointer must be a live handle returned by the constructor, freed once.
#[no_mangle]
pub unsafe extern "C" fn llev_approx_msm_index_free(index: *mut LlevApproxMsmIndex) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}
