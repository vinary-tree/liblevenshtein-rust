//! Frozen quantized temporal candidate index and bounded C page cursor.

use super::{
    index::{boundary, slice},
    temporal_index::{incomplete_code, LlevTemporalSearchLimits},
    LlevAlgorithm, LlevStatus,
};
use crate::{
    time_series::{
        IncompleteReason, PageBudget, QuantizationConfig, QuantizedCandidateCursor,
        QuantizedCandidateStartError, TimeSeriesIndex,
    },
    transducer::Algorithm,
};
use std::{collections::HashMap, sync::Arc};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
/// Bounded construction settings for a quantized byte candidate index.
pub struct LlevQuantizedIndexConfig {
    /// Inclusive lower sample bound for quantization.
    pub quant_min: f64,
    /// Inclusive upper sample bound for quantization.
    pub quant_max: f64,
    /// Number of byte bins, from one through 256.
    pub quant_bins: u32,
    /// Must be zero.
    pub reserved: u32,
    /// Maximum distinct identifiers retained.
    pub max_entries: usize,
    /// Maximum total sample count retained.
    pub max_total_samples: usize,
    /// Maximum samples in one stored series.
    pub max_series_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
/// One quantized edit-distance candidate; advisory for original samples.
pub struct LlevQuantizedMatch {
    /// Stored source identifier.
    pub id: u64,
    /// Exact byte edit distance after quantization.
    pub edit_distance: usize,
}

enum IndexState {
    Empty,
    Building(TimeSeriesIndex<u64>),
    Frozen(Arc<TimeSeriesIndex<u64>>),
}

/// Opaque mutable-then-frozen quantized candidate index handle.
pub struct LlevQuantizedIndex {
    state: IndexState,
    lengths: HashMap<u64, usize>,
    max_entries: usize,
    max_total_samples: usize,
    max_series_len: usize,
    total_samples: usize,
}

/// Opaque bounded, closeable cursor over one frozen index revision.
pub struct LlevQuantizedCursor {
    // Explicitly drop the borrowed cursor before releasing the immutable Arc.
    cursor: Option<QuantizedCandidateCursor<'static, u64>>,
    _index: Arc<TimeSeriesIndex<u64>>,
}

impl Drop for LlevQuantizedCursor {
    fn drop(&mut self) {
        self.cursor.take();
    }
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

/// Allocate one mutable quantized index with finite source ceilings.
///
/// # Safety
/// Pointers must be valid and `out_index` writable.
#[no_mangle]
pub unsafe extern "C" fn llev_quantized_index_new(
    config: *const LlevQuantizedIndexConfig,
    out_index: *mut *mut LlevQuantizedIndex,
) -> LlevStatus {
    boundary(|| {
        let output = out_index.as_mut().ok_or((
            LlevStatus::NullPointer,
            "quantized index output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let raw = *config.as_ref().ok_or((
            LlevStatus::NullPointer,
            "quantized index config is null".into(),
        ))?;
        if raw.reserved != 0 || !(1..=256).contains(&raw.quant_bins) {
            return Err(invalid(
                "quantized index requires 1..=256 bins and zero reserved",
            ));
        }
        let quant = QuantizationConfig::try_uniform(raw.quant_min, raw.quant_max, raw.quant_bins)
            .ok_or_else(|| invalid("invalid quantization range"))?;
        *output = Box::into_raw(Box::new(LlevQuantizedIndex {
            state: IndexState::Building(TimeSeriesIndex::new_with_verification(quant)),
            lengths: HashMap::new(),
            max_entries: raw.max_entries,
            max_total_samples: raw.max_total_samples,
            max_series_len: raw.max_series_len,
            total_samples: 0,
        }));
        Ok(LlevStatus::Ok)
    })
}

/// Insert or replace one identifier before freezing. Failed source-limit
/// checks leave the previous series in place.
///
/// # Safety
/// The index must be live and exclusively accessed. The sample slice must be
/// immutable and valid for the call.
#[no_mangle]
pub unsafe extern "C" fn llev_quantized_index_insert(
    index: *mut LlevQuantizedIndex,
    id: u64,
    samples: *const f64,
    len: usize,
) -> LlevStatus {
    boundary(|| {
        let handle = index
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "quantized index is null".into()))?;
        let values = slice(samples, len, "quantized series")?;
        if len > handle.max_series_len {
            return Err((
                LlevStatus::LimitExceeded,
                "quantized series length limit".into(),
            ));
        }
        let building = match &mut handle.state {
            IndexState::Building(value) => value,
            IndexState::Frozen(_) | IndexState::Empty => {
                return Err((LlevStatus::Closed, "quantized index is frozen".into()));
            }
        };
        let old = handle.lengths.get(&id).copied();
        let entries = handle
            .lengths
            .len()
            .checked_add(usize::from(old.is_none()))
            .ok_or((
                LlevStatus::LimitExceeded,
                "quantized entry count overflow".into(),
            ))?;
        let total = handle
            .total_samples
            .checked_sub(old.unwrap_or(0))
            .and_then(|count| count.checked_add(len))
            .ok_or((
                LlevStatus::LimitExceeded,
                "quantized sample count overflow".into(),
            ))?;
        if entries > handle.max_entries || total > handle.max_total_samples {
            return Err((LlevStatus::LimitExceeded, "quantized source limit".into()));
        }
        building.insert(id, values);
        if building.quantized_key_slots() > handle.max_entries.saturating_mul(2) {
            building.compact_verification();
        }
        handle.lengths.insert(id, len);
        handle.total_samples = total;
        Ok(LlevStatus::Ok)
    })
}

/// Freeze the index into an immutable snapshot for independent cursors.
///
/// # Safety
/// The index must be live and exclusively accessed.
#[no_mangle]
pub unsafe extern "C" fn llev_quantized_index_freeze(index: *mut LlevQuantizedIndex) -> LlevStatus {
    boundary(|| {
        let handle = index
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "quantized index is null".into()))?;
        let old = std::mem::replace(&mut handle.state, IndexState::Empty);
        handle.state = match old {
            IndexState::Building(value) => IndexState::Frozen(Arc::new(value)),
            other => other,
        };
        Ok(LlevStatus::Ok)
    })
}

/// Release an index handle. Existing cursors retain the frozen snapshot.
///
/// # Safety
/// Pointer must be a live handle returned by `llev_quantized_index_new` and
/// freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn llev_quantized_index_free(index: *mut LlevQuantizedIndex) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}

/// Start a quantized-byte candidate query for Standard, OSA transposition,
/// or merge/split edit distance. Candidates are advisory for original samples.
///
/// # Safety
/// Index and output pointers must be valid. Query is borrowed only for the
/// duration of this call. Index may be freed after this call, not during it.
#[no_mangle]
pub unsafe extern "C" fn llev_quantized_index_query(
    index: *const LlevQuantizedIndex,
    query: *const f64,
    query_len: usize,
    max_distance: usize,
    algorithm: u32,
    limits: *const LlevTemporalSearchLimits,
    out_cursor: *mut *mut LlevQuantizedCursor,
    out_reason: *mut u32,
) -> LlevStatus {
    boundary(|| {
        let output = out_cursor.as_mut().ok_or((
            LlevStatus::NullPointer,
            "quantized cursor output is null".into(),
        ))?;
        let reason = out_reason.as_mut().ok_or((
            LlevStatus::NullPointer,
            "quantized reason output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        *reason = 0;
        let handle = index
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "quantized index is null".into()))?;
        let values = slice(query, query_len, "quantized query")?;
        let raw = *limits
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "quantized limits are null".into()))?;
        let algorithm: Algorithm = LlevAlgorithm::try_from(algorithm)
            .map_err(|_| invalid("unknown quantized edit algorithm"))?
            .into();
        let frozen = match &handle.state {
            IndexState::Frozen(value) => Arc::clone(value),
            IndexState::Building(_) | IndexState::Empty => {
                return Err(invalid("quantized index must be frozen"));
            }
        };
        // SAFETY: the retained Arc owns this immutable allocation until the
        // cursor is explicitly dropped before its Arc field.
        let borrowed: &'static TimeSeriesIndex<u64> = unsafe { &*Arc::as_ptr(&frozen) };
        let cursor = borrowed
            .search_quantized_bounded(values, max_distance, algorithm, raw.into())
            .map_err(|error| match error {
                QuantizedCandidateStartError::Validation(error) => invalid(error.to_string()),
                QuantizedCandidateStartError::Resource(incomplete) => {
                    *reason = incomplete_code(incomplete);
                    (LlevStatus::LimitExceeded, format!("{incomplete:?}"))
                }
            })?;
        *output = Box::into_raw(Box::new(LlevQuantizedCursor {
            cursor: Some(cursor),
            _index: frozen,
        }));
        Ok(LlevStatus::Ok)
    })
}

/// Copy at most one bounded native page of quantized candidates.
///
/// # Safety
/// Cursor must be live and exclusively accessed. Outputs must be valid,
/// writable, and nonoverlapping. Capacity and page budgets must be positive.
#[no_mangle]
pub unsafe extern "C" fn llev_quantized_cursor_next_batch(
    cursor: *mut LlevQuantizedCursor,
    out_matches: *mut LlevQuantizedMatch,
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
            "quantized batch length is null".into(),
        ))?;
        let done = out_done.as_mut().ok_or((
            LlevStatus::NullPointer,
            "quantized done flag is null".into(),
        ))?;
        let reason = out_reason.as_mut().ok_or((
            LlevStatus::NullPointer,
            "quantized reason output is null".into(),
        ))?;
        *len = 0;
        *done = 0;
        *reason = 0;
        if capacity == 0 || page_work_units == 0 || page_results == 0 {
            return Err(invalid("quantized page and batch limits must be positive"));
        }
        let output = std::slice::from_raw_parts_mut(
            out_matches.as_mut().ok_or((
                LlevStatus::NullPointer,
                "quantized batch output is null".into(),
            ))?,
            capacity,
        );
        let handle = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "quantized cursor is null".into()))?;
        let page = handle
            .cursor
            .as_mut()
            .expect("live quantized cursor holds its continuation")
            .next_page(PageBudget {
                max_work_units: page_work_units,
                max_results: page_results.min(capacity),
            })
            .map_err(|incomplete: IncompleteReason| {
                *reason = incomplete_code(incomplete);
                (
                    LlevStatus::LimitExceeded,
                    format!("quantized query incomplete: {incomplete:?}"),
                )
            })?;
        for (slot, (id, edit_distance)) in page.matches.into_iter().enumerate() {
            output[slot] = LlevQuantizedMatch { id, edit_distance };
            *len += 1;
        }
        *done = u8::from(page.done);
        Ok(LlevStatus::Ok)
    })
}

/// Copy one original series from the cursor's frozen index snapshot. Calling
/// with null `out_samples` and zero capacity reports the required length.
///
/// # Safety
/// Cursor and output length must be valid; a nonzero output capacity requires
/// writable storage for that many `f64` values.
#[no_mangle]
pub unsafe extern "C" fn llev_quantized_cursor_original(
    cursor: *const LlevQuantizedCursor,
    id: u64,
    out_samples: *mut f64,
    capacity: usize,
    out_len: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let len = out_len.as_mut().ok_or((
            LlevStatus::NullPointer,
            "quantized original length is null".into(),
        ))?;
        *len = 0;
        let handle = cursor
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "quantized cursor is null".into()))?;
        let original = handle
            ._index
            .get_original(&id)
            .ok_or_else(|| invalid("quantized original identifier is absent"))?;
        *len = original.len();
        if capacity == 0 && out_samples.is_null() {
            return Ok(LlevStatus::Ok);
        }
        if capacity < original.len() {
            return Err((
                LlevStatus::LimitExceeded,
                "quantized original buffer is too short".into(),
            ));
        }
        if !original.is_empty() {
            let output = out_samples.as_mut().ok_or((
                LlevStatus::NullPointer,
                "quantized original output is null".into(),
            ))?;
            std::ptr::copy_nonoverlapping(original.as_ptr(), output, original.len());
        }
        Ok(LlevStatus::Ok)
    })
}

/// Cancel and release a quantized candidate cursor.
///
/// # Safety
/// Pointer must be a live cursor returned by `llev_quantized_index_query`
/// and freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn llev_quantized_cursor_free(cursor: *mut LlevQuantizedCursor) {
    if !cursor.is_null() {
        drop(Box::from_raw(cursor));
    }
}
