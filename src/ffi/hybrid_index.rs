//! Frozen hybrid quantized-candidate / exact-MSM index and bounded C cursor.

use super::{
    index::{boundary, slice},
    temporal_index::{incomplete_code, LlevTemporalIndexMatch, LlevTemporalSearchLimits},
    LlevQuantizedIndexConfig, LlevStatus,
};
use crate::time_series::{
    HybridCandidateCursor, HybridSearchIndex, HybridStartError, IncompleteReason, LowerBoundType,
    MsmConfig, PageBudget, QuantizationConfig,
};
use std::{collections::HashMap, sync::Arc};

/// Construction settings for an advisory hybrid candidate and MSM index.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevHybridIndexConfig {
    /// Quantizer and finite source ceilings.
    pub source: LlevQuantizedIndexConfig,
    /// Nonnegative finite MSM split/merge cost.
    pub msm_cost: f64,
    /// Positive finite multiplier, or source-default fallback otherwise.
    pub trie_threshold_multiplier: f64,
    /// 0 length, 1 prefix Euclidean, 2 prefix L1, 3 combined.
    pub lower_bound_type: u32,
    /// Zero disables prefiltering; one enables it.
    pub use_lower_bounds: u32,
}

enum IndexState {
    Empty,
    Building(Box<HybridSearchIndex<u64>>),
    Frozen(Arc<HybridSearchIndex<u64>>),
}

/// Opaque mutable-then-frozen hybrid index handle.
pub struct LlevHybridIndex {
    state: IndexState,
    lengths: HashMap<u64, usize>,
    max_entries: usize,
    max_total_samples: usize,
    max_series_len: usize,
    total_samples: usize,
}

/// Opaque closeable cursor that retains its frozen source snapshot.
pub struct LlevHybridCursor {
    // This borrowed cursor is dropped before its owning Arc.
    state: Option<HybridCursorState>,
    _index: Arc<HybridSearchIndex<u64>>,
}

enum HybridCursorState {
    Range(Box<HybridCandidateCursor<'static, u64>>),
    Knn {
        matches: Vec<(u64, f64)>,
        next: usize,
    },
}

impl Drop for LlevHybridCursor {
    fn drop(&mut self) {
        self.state.take();
    }
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

/// Allocate a bounded mutable hybrid index.
///
/// # Safety
/// Pointers must be valid and output writable.
#[no_mangle]
pub unsafe extern "C" fn llev_hybrid_index_new(
    config: *const LlevHybridIndexConfig,
    out_index: *mut *mut LlevHybridIndex,
) -> LlevStatus {
    boundary(|| {
        let output = out_index.as_mut().ok_or((
            LlevStatus::NullPointer,
            "hybrid index output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let raw = *config.as_ref().ok_or((
            LlevStatus::NullPointer,
            "hybrid index config is null".into(),
        ))?;
        if raw.source.reserved != 0 || !(1..=256).contains(&raw.source.quant_bins) {
            return Err(invalid(
                "hybrid quantizer requires 1..=256 bins and zero reserved",
            ));
        }
        let quant = QuantizationConfig::try_uniform(
            raw.source.quant_min,
            raw.source.quant_max,
            raw.source.quant_bins,
        )
        .ok_or_else(|| invalid("invalid hybrid quantization range"))?;
        let msm = MsmConfig::try_new(raw.msm_cost).map_err(|error| invalid(error.to_string()))?;
        let lower_bound = match raw.lower_bound_type {
            0 => LowerBoundType::LengthOnly,
            1 => LowerBoundType::EuclideanOnly,
            2 => LowerBoundType::L1Only,
            3 => LowerBoundType::Combined,
            _ => return Err(invalid("unknown hybrid lower-bound type")),
        };
        if raw.use_lower_bounds > 1 {
            return Err(invalid("hybrid use_lower_bounds must be zero or one"));
        }
        let mut index = HybridSearchIndex::new(quant, msm);
        index.set_trie_threshold_multiplier(raw.trie_threshold_multiplier);
        index.set_lower_bound_type(lower_bound);
        index.set_use_lower_bounds(raw.use_lower_bounds == 1);
        *output = Box::into_raw(Box::new(LlevHybridIndex {
            state: IndexState::Building(Box::new(index)),
            lengths: HashMap::new(),
            max_entries: raw.source.max_entries,
            max_total_samples: raw.source.max_total_samples,
            max_series_len: raw.source.max_series_len,
            total_samples: 0,
        }));
        Ok(LlevStatus::Ok)
    })
}

/// Insert or replace one finite source series before freeze.
///
/// # Safety
/// The index must be live and exclusively accessed. Samples must be valid and
/// immutable for the call.
#[no_mangle]
pub unsafe extern "C" fn llev_hybrid_index_insert(
    index: *mut LlevHybridIndex,
    id: u64,
    samples: *const f64,
    len: usize,
) -> LlevStatus {
    boundary(|| {
        let handle = index
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "hybrid index is null".into()))?;
        let values = slice(samples, len, "hybrid series")?;
        if len > handle.max_series_len {
            return Err((
                LlevStatus::LimitExceeded,
                "hybrid series length limit".into(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid("hybrid MSM source series must be finite"));
        }
        let building = match &mut handle.state {
            IndexState::Building(value) => value,
            IndexState::Frozen(_) | IndexState::Empty => {
                return Err((LlevStatus::Closed, "hybrid index is frozen".into()));
            }
        };
        let old = handle.lengths.get(&id).copied();
        let entries = handle
            .lengths
            .len()
            .checked_add(usize::from(old.is_none()))
            .ok_or((
                LlevStatus::LimitExceeded,
                "hybrid entry count overflow".into(),
            ))?;
        let total = handle
            .total_samples
            .checked_sub(old.unwrap_or(0))
            .and_then(|count| count.checked_add(len))
            .ok_or((
                LlevStatus::LimitExceeded,
                "hybrid sample count overflow".into(),
            ))?;
        if entries > handle.max_entries || total > handle.max_total_samples {
            return Err((LlevStatus::LimitExceeded, "hybrid source limit".into()));
        }
        building.insert(id, values);
        if building.quantized_key_slots() > handle.max_entries.saturating_mul(2) {
            building.compact_source();
        }
        handle.lengths.insert(id, len);
        handle.total_samples = total;
        Ok(LlevStatus::Ok)
    })
}

/// Freeze one index into an immutable revision.
///
/// # Safety
/// Index must be live and exclusively accessed.
#[no_mangle]
pub unsafe extern "C" fn llev_hybrid_index_freeze(index: *mut LlevHybridIndex) -> LlevStatus {
    boundary(|| {
        let handle = index
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "hybrid index is null".into()))?;
        let old = std::mem::replace(&mut handle.state, IndexState::Empty);
        handle.state = match old {
            IndexState::Building(value) => IndexState::Frozen(Arc::from(value)),
            other => other,
        };
        Ok(LlevStatus::Ok)
    })
}

/// Release an index handle; live cursors retain their frozen snapshot.
///
/// # Safety
/// Pointer must be a live handle returned by `llev_hybrid_index_new` and
/// freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn llev_hybrid_index_free(index: *mut LlevHybridIndex) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}

/// Start a bounded lazy hybrid candidate / exact-MSM range query.
///
/// # Safety
/// Index and output pointers must be valid; query must be immutable and valid
/// for the call. The index may be freed after query construction.
#[no_mangle]
pub unsafe extern "C" fn llev_hybrid_index_query_range(
    index: *const LlevHybridIndex,
    query: *const f64,
    query_len: usize,
    cutoff: f64,
    limits: *const LlevTemporalSearchLimits,
    out_cursor: *mut *mut LlevHybridCursor,
    out_reason: *mut u32,
) -> LlevStatus {
    boundary(|| {
        let output = out_cursor.as_mut().ok_or((
            LlevStatus::NullPointer,
            "hybrid cursor output is null".into(),
        ))?;
        let reason = out_reason.as_mut().ok_or((
            LlevStatus::NullPointer,
            "hybrid reason output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        *reason = 0;
        let handle = index
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "hybrid index is null".into()))?;
        let values = slice(query, query_len, "hybrid query")?;
        let raw = *limits
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "hybrid limits are null".into()))?;
        let frozen = match &handle.state {
            IndexState::Frozen(value) => Arc::clone(value),
            IndexState::Building(_) | IndexState::Empty => {
                return Err(invalid("hybrid index must be frozen"));
            }
        };
        // SAFETY: the retained Arc owns this immutable allocation until the
        // borrowed cursor is explicitly dropped first.
        let borrowed: &'static HybridSearchIndex<u64> = unsafe { &*Arc::as_ptr(&frozen) };
        let cursor = borrowed
            .search_hybrid_bounded(values, cutoff, raw.into())
            .map_err(|error| match error {
                HybridStartError::Validation(error) => invalid(error.to_string()),
                HybridStartError::Resource(incomplete) => {
                    *reason = incomplete_code(incomplete);
                    (LlevStatus::LimitExceeded, format!("{incomplete:?}"))
                }
            })?;
        *output = Box::into_raw(Box::new(LlevHybridCursor {
            state: Some(HybridCursorState::Range(Box::new(cursor))),
            _index: frozen,
        }));
        Ok(LlevStatus::Ok)
    })
}

/// Complete a bounded hybrid kNN search and open a closeable result cursor.
/// No partial neighbor list is published on resource exhaustion.
///
/// # Safety
/// Index and output pointers must be valid; query must be immutable and valid
/// for the call. The index may be freed after query construction.
#[no_mangle]
pub unsafe extern "C" fn llev_hybrid_index_query_knn(
    index: *const LlevHybridIndex,
    query: *const f64,
    query_len: usize,
    k: usize,
    initial_threshold: f64,
    limits: *const LlevTemporalSearchLimits,
    out_cursor: *mut *mut LlevHybridCursor,
    out_reason: *mut u32,
) -> LlevStatus {
    boundary(|| {
        let output = out_cursor.as_mut().ok_or((
            LlevStatus::NullPointer,
            "hybrid cursor output is null".into(),
        ))?;
        let reason = out_reason.as_mut().ok_or((
            LlevStatus::NullPointer,
            "hybrid reason output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        *reason = 0;
        let handle = index
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "hybrid index is null".into()))?;
        let values = slice(query, query_len, "hybrid query")?;
        let raw = *limits
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "hybrid limits are null".into()))?;
        let frozen = match &handle.state {
            IndexState::Frozen(value) => Arc::clone(value),
            IndexState::Building(_) | IndexState::Empty => {
                return Err(invalid("hybrid index must be frozen"));
            }
        };
        let result = frozen
            .search_hybrid_knn_bounded(values, k, initial_threshold, raw.into())
            .map_err(|error| match error {
                HybridStartError::Validation(error) => invalid(error.to_string()),
                HybridStartError::Resource(incomplete) => {
                    *reason = incomplete_code(incomplete);
                    (LlevStatus::LimitExceeded, format!("{incomplete:?}"))
                }
            })?;
        *output = Box::into_raw(Box::new(LlevHybridCursor {
            state: Some(HybridCursorState::Knn {
                matches: result.matches,
                next: 0,
            }),
            _index: frozen,
        }));
        Ok(LlevStatus::Ok)
    })
}

/// Copy at most one bounded page of verified MSM results.
///
/// # Safety
/// Cursor must be live and exclusively accessed. Outputs must be valid,
/// writable and nonoverlapping. Capacity and page budgets must be positive.
#[no_mangle]
pub unsafe extern "C" fn llev_hybrid_cursor_next_batch(
    cursor: *mut LlevHybridCursor,
    out_matches: *mut LlevTemporalIndexMatch,
    capacity: usize,
    page_work_units: usize,
    page_results: usize,
    out_len: *mut usize,
    out_done: *mut u8,
    out_reason: *mut u32,
) -> LlevStatus {
    boundary(|| {
        let len = out_len.as_mut().ok_or((
            LlevStatus::NullPointer,
            "hybrid batch length is null".into(),
        ))?;
        let done = out_done
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "hybrid done flag is null".into()))?;
        let reason = out_reason.as_mut().ok_or((
            LlevStatus::NullPointer,
            "hybrid reason output is null".into(),
        ))?;
        *len = 0;
        *done = 0;
        *reason = 0;
        if capacity == 0 || page_work_units == 0 || page_results == 0 {
            return Err(invalid("hybrid page and batch limits must be positive"));
        }
        let output = std::slice::from_raw_parts_mut(
            out_matches.as_mut().ok_or((
                LlevStatus::NullPointer,
                "hybrid batch output is null".into(),
            ))?,
            capacity,
        );
        let handle = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "hybrid cursor is null".into()))?;
        match handle
            .state
            .as_mut()
            .expect("live hybrid cursor holds its continuation")
        {
            HybridCursorState::Range(cursor) => {
                let page = cursor
                    .next_page(PageBudget {
                        max_work_units: page_work_units,
                        max_results: page_results.min(capacity),
                    })
                    .map_err(|incomplete: IncompleteReason| {
                        *reason = incomplete_code(incomplete);
                        (
                            LlevStatus::LimitExceeded,
                            format!("hybrid query incomplete: {incomplete:?}"),
                        )
                    })?;
                for (slot, (id, distance)) in page.matches.into_iter().enumerate() {
                    output[slot] = LlevTemporalIndexMatch { id, distance };
                    *len += 1;
                }
                *done = u8::from(page.done);
            }
            HybridCursorState::Knn { matches, next } => {
                let count = matches
                    .len()
                    .saturating_sub(*next)
                    .min(capacity)
                    .min(page_results)
                    .min(page_work_units);
                for (slot, &(id, distance)) in matches[*next..*next + count].iter().enumerate() {
                    output[slot] = LlevTemporalIndexMatch { id, distance };
                }
                *next += count;
                *len = count;
                *done = u8::from(*next == matches.len());
            }
        }
        Ok(LlevStatus::Ok)
    })
}

/// Cancel and release a hybrid cursor.
///
/// # Safety
/// Pointer must be a live cursor returned by `llev_hybrid_index_query_range`
/// and freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn llev_hybrid_cursor_free(cursor: *mut LlevHybridCursor) {
    if !cursor.is_null() {
        drop(Box::from_raw(cursor));
    }
}
