//! Bounded physical-time TWED index and resumable C range cursors.

use super::{
    index::{boundary, slice},
    temporal::{timestamp_error, timestamp_unit, LlevTimestampedSeriesView},
    temporal_index::{incomplete_code, LlevTemporalSearchLimits},
    LlevStatus,
};
use crate::time_series::{
    IncompleteReason, MetricTimestampedTwedConfig, OperationOutcome, PageBudget, ResourceLimits,
    TimestampedSeries, TimestampedTwedIndex, TimestampedTwedIndexError,
    TimestampedTwedProductLimits, TimestampedTwedQuantizer, TimestampedTwedRangeContinuation,
    TimestampedTwedRangeMatch,
};
use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
};

/// Typed quantization, metric parameters, and hard ingestion ceilings.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTimestampedTwedIndexConfig {
    /// 1 seconds, 2 milliseconds, 3 microseconds, 4 nanoseconds.
    pub unit: u32,
    /// Must be zero.
    pub reserved: u32,
    /// Shared finite physical origin.
    pub origin: f64,
    /// Finite increasing scalar-value domain lower bound.
    pub value_min: f64,
    /// Finite increasing scalar-value domain upper bound.
    pub value_max: f64,
    /// Finite increasing timestamp domain lower bound.
    pub time_min: f64,
    /// Finite increasing timestamp domain upper bound.
    pub time_max: f64,
    /// Number of value bins in 1..=2^31.
    pub value_bins: u32,
    /// Number of timestamp bins in 1..=2^31.
    pub time_bins: u32,
    /// Finite positive cost per physical time unit.
    pub stiffness: f64,
    /// Finite nonnegative gap penalty.
    pub gap_penalty: f64,
    /// Maximum retained episodes.
    pub max_entries: usize,
    /// Maximum retained scalar samples across all episodes.
    pub max_total_samples: usize,
    /// Maximum scalar samples in one episode.
    pub max_series_len: usize,
}

/// Common resource ceilings plus product-specific state ceilings.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTimestampedTwedSearchLimits {
    /// Cumulative and peak search ceilings.
    pub common: LlevTemporalSearchLimits,
    /// Maximum exact query-local residuals.
    pub max_product_states: usize,
    /// Maximum retained recurrence positions.
    pub max_product_positions: usize,
    /// Maximum cached observed transitions.
    pub max_transition_cache_entries: usize,
}

impl From<LlevTimestampedTwedSearchLimits> for TimestampedTwedProductLimits {
    fn from(raw: LlevTimestampedTwedSearchLimits) -> Self {
        Self {
            resources: ResourceLimits::from(raw.common),
            max_product_states: raw.max_product_states,
            max_product_positions: raw.max_product_positions,
            max_transition_cache_entries: raw.max_transition_cache_entries,
        }
    }
}

/// One full-precision exact match from a captured index revision.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevTimestampedTwedMatch {
    /// Caller-provided metadata identifier.
    pub id: u64,
    /// Stable native insertion-order episode identifier.
    pub episode_id: u64,
    /// Exact physical-time TWED distance.
    pub distance: f64,
}

enum IndexState {
    Empty,
    Building(TimestampedTwedIndex<u64>),
    Frozen(Arc<TimestampedTwedIndex<u64>>),
}

/// Opaque index whose frozen revision is retained by independent cursors.
pub struct LlevTimestampedTwedIndex {
    state: IndexState,
    unit: u32,
    origin: f64,
    max_entries: usize,
    max_total_samples: usize,
    max_series_len: usize,
    entries: usize,
    total_samples: usize,
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

fn index_error(error: TimestampedTwedIndexError) -> (LlevStatus, String) {
    match error {
        TimestampedTwedIndexError::Resource(_) | TimestampedTwedIndexError::EpisodeIdOverflow => {
            (LlevStatus::LimitExceeded, error.to_string())
        }
        TimestampedTwedIndexError::MixedUnits | TimestampedTwedIndexError::MixedOrigins => {
            (LlevStatus::DomainMismatch, error.to_string())
        }
        TimestampedTwedIndexError::Timestamped(error) => timestamp_error(error),
        _ => invalid(error.to_string()),
    }
}

unsafe fn read_series(
    raw: LlevTimestampedSeriesView,
    max_series_len: usize,
    limits: ResourceLimits,
) -> Result<TimestampedSeries, (LlevStatus, String)> {
    if raw.len > max_series_len {
        return Err((
            LlevStatus::LimitExceeded,
            "timestamped series length limit".into(),
        ));
    }
    if raw.reserved != 0 {
        return Err(invalid("timestamped series reserved field must be zero"));
    }
    let unit = timestamp_unit(raw.unit)?;
    TimestampedSeries::try_new_with_origin(
        slice(raw.values, raw.len, "timestamped values")?,
        slice(raw.timestamps, raw.len, "timestamps")?,
        unit,
        raw.origin,
        limits,
    )
    .map_err(timestamp_error)
}

/// Construct a bounded, mutable physical-time TWED index.
///
/// # Safety
/// Both pointers must address valid disjoint storage; the output receives
/// one owning handle on success and null on failure.
#[no_mangle]
pub unsafe extern "C" fn llev_timestamped_twed_index_new(
    config: *const LlevTimestampedTwedIndexConfig,
    out_index: *mut *mut LlevTimestampedTwedIndex,
) -> LlevStatus {
    boundary(|| {
        let output = out_index.as_mut().ok_or((
            LlevStatus::NullPointer,
            "timestamped index output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let raw = *config.as_ref().ok_or((
            LlevStatus::NullPointer,
            "timestamped index config is null".into(),
        ))?;
        if raw.reserved != 0 {
            return Err(invalid("timestamped index reserved field must be zero"));
        }
        let unit = timestamp_unit(raw.unit)?;
        let quantizer = TimestampedTwedQuantizer::try_new(
            unit,
            raw.origin,
            (raw.value_min, raw.value_max),
            (raw.time_min, raw.time_max),
            raw.value_bins,
            raw.time_bins,
        )
        .map_err(index_error)?;
        let metric = MetricTimestampedTwedConfig::try_new(raw.stiffness, raw.gap_penalty)
            .map_err(timestamp_error)?;
        let index = LlevTimestampedTwedIndex {
            state: IndexState::Building(TimestampedTwedIndex::new(quantizer, metric)),
            unit: raw.unit,
            origin: raw.origin,
            max_entries: raw.max_entries,
            max_total_samples: raw.max_total_samples,
            max_series_len: raw.max_series_len,
            entries: 0,
            total_samples: 0,
        };
        *output = Box::into_raw(Box::new(index));
        Ok(LlevStatus::Ok)
    })
}

/// Copy and insert one episode, returning its stable native episode ID.
///
/// # Safety
/// The index must be live and exclusively accessible. The input view and
/// output must address valid disjoint storage; input buffers are borrowed
/// only for this call.
#[no_mangle]
pub unsafe extern "C" fn llev_timestamped_twed_index_insert(
    index: *mut LlevTimestampedTwedIndex,
    id: u64,
    series: *const LlevTimestampedSeriesView,
    out_episode_id: *mut u64,
) -> LlevStatus {
    boundary(|| {
        let output = out_episode_id.as_mut().ok_or((
            LlevStatus::NullPointer,
            "timestamped episode output is null".into(),
        ))?;
        *output = 0;
        let handle = index
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "timestamped index is null".into()))?;
        let raw = *series
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "timestamped series is null".into()))?;
        let next_entries = handle.entries.checked_add(1).ok_or((
            LlevStatus::LimitExceeded,
            "timestamped entry count overflow".into(),
        ))?;
        let next_samples = handle.total_samples.checked_add(raw.len).ok_or((
            LlevStatus::LimitExceeded,
            "timestamped sample count overflow".into(),
        ))?;
        if next_entries > handle.max_entries || next_samples > handle.max_total_samples {
            return Err((
                LlevStatus::LimitExceeded,
                "timestamped index ingestion limit".into(),
            ));
        }
        if raw.unit != handle.unit || raw.origin.to_bits() != handle.origin.to_bits() {
            return Err((
                LlevStatus::DomainMismatch,
                "timestamped index domain mismatch".into(),
            ));
        }
        let retained_bytes = raw.len.checked_mul(2 * std::mem::size_of::<f64>()).ok_or((
            LlevStatus::LimitExceeded,
            "timestamped ingestion byte count overflow".into(),
        ))?;
        let ingest_limits = ResourceLimits {
            max_series_len: handle.max_series_len,
            max_scratch_bytes: retained_bytes,
            ..ResourceLimits::default()
        };
        let series = read_series(raw, handle.max_series_len, ingest_limits)?;
        let index = match &mut handle.state {
            IndexState::Building(index) => index,
            _ => return Err(invalid("timestamped index must be mutable for insertion")),
        };
        let episode_id = index.insert(id, series).map_err(index_error)?;
        handle.entries = next_entries;
        handle.total_samples = next_samples;
        *output = episode_id;
        Ok(LlevStatus::Ok)
    })
}

/// Freeze the index. Later insertions fail, while cursors share its snapshot.
///
/// # Safety
/// The index must be live and exclusively accessible during the call.
#[no_mangle]
pub unsafe extern "C" fn llev_timestamped_twed_index_freeze(
    index: *mut LlevTimestampedTwedIndex,
) -> LlevStatus {
    boundary(|| {
        let handle = index
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "timestamped index is null".into()))?;
        match std::mem::replace(&mut handle.state, IndexState::Empty) {
            IndexState::Building(index) => {
                handle.state = IndexState::Frozen(Arc::new(index));
                Ok(LlevStatus::Ok)
            }
            IndexState::Frozen(index) => {
                handle.state = IndexState::Frozen(index);
                Ok(LlevStatus::Ok)
            }
            IndexState::Empty => Err(invalid("timestamped index has no state")),
        }
    })
}

/// Release one owning index handle; existing cursors retain its revision.
///
/// # Safety
/// The pointer must be a live handle returned by the constructor, freed once.
#[no_mangle]
pub unsafe extern "C" fn llev_timestamped_twed_index_free(index: *mut LlevTimestampedTwedIndex) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}

struct NativeCursor {
    // Drop the self-referential native continuation before query and index.
    continuation: Option<TimestampedTwedRangeContinuation<'static, u64>>,
    _query: Box<TimestampedSeries>,
    _index: Arc<TimestampedTwedIndex<u64>>,
    pending: VecDeque<LlevTimestampedTwedMatch>,
    seen: HashSet<u64>,
    done: bool,
    terminal: Option<IncompleteReason>,
}

impl Drop for NativeCursor {
    fn drop(&mut self) {
        self.continuation.take();
    }
}

impl NativeCursor {
    fn new(
        index: Arc<TimestampedTwedIndex<u64>>,
        query: TimestampedSeries,
        cutoff: f64,
        limits: TimestampedTwedProductLimits,
    ) -> Result<Self, (LlevStatus, String)> {
        let query = Box::new(query);
        // SAFETY: both allocations remain owned by the cursor. Drop destroys
        // the continuation before either allocation is released. Moving the
        // Box or Arc values does not move their referents.
        let borrowed_index: &'static TimestampedTwedIndex<u64> = unsafe { &*Arc::as_ptr(&index) };
        let borrowed_query: &'static TimestampedSeries = unsafe { &*(&*query as *const _) };
        let outcome = borrowed_index
            .search_range_bounded(
                borrowed_query,
                cutoff,
                limits,
                PageBudget {
                    max_work_units: 1,
                    max_results: 1,
                },
            )
            .map_err(index_error)?;
        let mut cursor = Self {
            continuation: None,
            _query: query,
            _index: index,
            pending: VecDeque::new(),
            seen: HashSet::new(),
            done: false,
            terminal: None,
        };
        cursor.accept(outcome)?;
        Ok(cursor)
    }

    fn add(
        &mut self,
        matches: &[TimestampedTwedRangeMatch<'static, u64>],
    ) -> Result<(), (LlevStatus, String)> {
        for matched in matches {
            if self.seen.contains(&matched.episode_id) {
                continue;
            }
            self.seen.try_reserve(1).map_err(|_| {
                (
                    LlevStatus::LimitExceeded,
                    "timestamped seen-ID allocation failed".into(),
                )
            })?;
            self.pending.try_reserve(1).map_err(|_| {
                (
                    LlevStatus::LimitExceeded,
                    "timestamped result allocation failed".into(),
                )
            })?;
            self.seen.insert(matched.episode_id);
            self.pending.push_back(LlevTimestampedTwedMatch {
                id: *matched.value,
                episode_id: matched.episode_id,
                distance: matched.distance,
            });
        }
        Ok(())
    }

    fn accept(
        &mut self,
        outcome: OperationOutcome<
            Vec<TimestampedTwedRangeMatch<'static, u64>>,
            TimestampedTwedRangeContinuation<'static, u64>,
        >,
    ) -> Result<(), (LlevStatus, String)> {
        match outcome {
            OperationOutcome::Complete { value, .. } => {
                self.add(&value)?;
                self.done = true;
            }
            OperationOutcome::Incomplete {
                partial,
                reason,
                continuation,
                ..
            } => {
                if let Some(ref value) = continuation {
                    self.add(value.exact_partial())?;
                } else if let Some(ref value) = partial {
                    self.add(value)?;
                }
                self.continuation = continuation;
                if self.continuation.is_none() {
                    self.terminal = Some(reason);
                }
            }
        }
        Ok(())
    }

    fn next(
        &mut self,
        out: &mut [LlevTimestampedTwedMatch],
        page: PageBudget,
    ) -> Result<(usize, bool), (u32, String)> {
        if self.pending.is_empty() && !self.done && self.terminal.is_none() {
            if let Some(continuation) = self.continuation.take() {
                let before = continuation.usage().work_units;
                self.accept(continuation.resume(page))
                    .map_err(|(_, message)| (13, message))?;
                if self.pending.is_empty()
                    && self
                        .continuation
                        .as_ref()
                        .is_some_and(|value| value.usage().work_units == before)
                {
                    return Err((
                        14,
                        "timestamped page work budget cannot advance this query".into(),
                    ));
                }
            }
        }
        let mut written = 0;
        while written < out.len() {
            let Some(value) = self.pending.pop_front() else {
                break;
            };
            out[written] = value;
            written += 1;
        }
        if written == 0 {
            if let Some(reason) = self.terminal {
                return Err((
                    incomplete_code(reason),
                    format!("timestamped range incomplete: {reason:?}"),
                ));
            }
        }
        Ok((written, self.done && self.pending.is_empty()))
    }
}

/// Opaque lazy cursor retaining a frozen native timestamped index revision.
pub struct LlevTimestampedTwedCursor {
    state: NativeCursor,
}

/// Begin a bounded exact range query over a frozen revision.
///
/// # Safety
/// All pointers must be valid and disjoint. The index must not be freed or
/// mutated during this call. The query is copied into the cursor; its input
/// buffers may be released when this function returns.
#[no_mangle]
pub unsafe extern "C" fn llev_timestamped_twed_index_query_range(
    index: *const LlevTimestampedTwedIndex,
    query: *const LlevTimestampedSeriesView,
    cutoff: f64,
    raw_limits: *const LlevTimestampedTwedSearchLimits,
    out_cursor: *mut *mut LlevTimestampedTwedCursor,
) -> LlevStatus {
    boundary(|| {
        let output = out_cursor.as_mut().ok_or((
            LlevStatus::NullPointer,
            "timestamped cursor output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let handle = index
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "timestamped index is null".into()))?;
        let raw_query = *query
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "timestamped query is null".into()))?;
        let raw_limits = *raw_limits.as_ref().ok_or((
            LlevStatus::NullPointer,
            "timestamped search limits are null".into(),
        ))?;
        if cutoff.is_nan() || cutoff < 0.0 {
            return Err(invalid("timestamped index cutoff must be nonnegative"));
        }
        if raw_query.unit != handle.unit || raw_query.origin.to_bits() != handle.origin.to_bits() {
            return Err((
                LlevStatus::DomainMismatch,
                "timestamped index domain mismatch".into(),
            ));
        }
        let frozen = match &handle.state {
            IndexState::Frozen(index) => Arc::clone(index),
            _ => return Err(invalid("timestamped index must be frozen before querying")),
        };
        let mut limits = TimestampedTwedProductLimits::from(raw_limits);
        let query_bytes = raw_query
            .len
            .checked_mul(2 * std::mem::size_of::<f64>())
            .ok_or((
                LlevStatus::LimitExceeded,
                "timestamped query byte count overflow".into(),
            ))?;
        if query_bytes > limits.resources.max_scratch_bytes
            || query_bytes > limits.resources.max_continuation_bytes
        {
            return Err((
                LlevStatus::LimitExceeded,
                "timestamped query copy limit".into(),
            ));
        }
        let query = read_series(raw_query, limits.resources.max_series_len, limits.resources)?;
        limits.resources.max_scratch_bytes -= query_bytes;
        limits.resources.max_continuation_bytes -= query_bytes;
        let state = NativeCursor::new(frozen, query, cutoff, limits)?;
        *output = Box::into_raw(Box::new(LlevTimestampedTwedCursor { state }));
        Ok(LlevStatus::Ok)
    })
}

/// Advance one bounded page. A zero-length page with out_done=0 is paused.
///
/// # Safety
/// The cursor must be live and exclusively accessible. Output pointers must
/// address valid disjoint writable storage; `out_matches` has `capacity`
/// elements when capacity is nonzero.
#[no_mangle]
pub unsafe extern "C" fn llev_timestamped_twed_cursor_next_batch(
    cursor: *mut LlevTimestampedTwedCursor,
    out_matches: *mut LlevTimestampedTwedMatch,
    capacity: usize,
    page_work_units: usize,
    page_results: usize,
    out_len: *mut usize,
    out_done: *mut u8,
    out_reason: *mut u32,
) -> LlevStatus {
    boundary(|| {
        let written = out_len.as_mut().ok_or((
            LlevStatus::NullPointer,
            "timestamped batch length output is null".into(),
        ))?;
        let done = out_done.as_mut().ok_or((
            LlevStatus::NullPointer,
            "timestamped done output is null".into(),
        ))?;
        let reason = out_reason.as_mut().ok_or((
            LlevStatus::NullPointer,
            "timestamped reason output is null".into(),
        ))?;
        *written = 0;
        *done = 0;
        *reason = 0;
        if capacity == 0 || page_work_units == 0 || page_results == 0 {
            return Err(invalid(
                "timestamped page budgets and output capacity must be positive",
            ));
        }
        let handle = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "timestamped cursor is null".into()))?;
        if out_matches.is_null() {
            return Err((
                LlevStatus::NullPointer,
                "timestamped batch output is null".into(),
            ));
        }
        if !(out_matches as usize).is_multiple_of(std::mem::align_of::<LlevTimestampedTwedMatch>())
        {
            return Err(invalid("timestamped batch output is not aligned"));
        }
        let out = std::slice::from_raw_parts_mut(out_matches, capacity);
        let page = PageBudget {
            max_work_units: page_work_units,
            max_results: page_results,
        };
        match handle.state.next(out, page) {
            Ok((count, complete)) => {
                *written = count;
                *done = u8::from(complete);
                Ok(LlevStatus::Ok)
            }
            Err((code, message)) => {
                *reason = code;
                Err((LlevStatus::LimitExceeded, message))
            }
        }
    })
}

/// Release an owning timestamped range cursor.
///
/// # Safety
/// The pointer must be a live cursor returned by query_range, freed once.
#[no_mangle]
pub unsafe extern "C" fn llev_timestamped_twed_cursor_free(cursor: *mut LlevTimestampedTwedCursor) {
    if !cursor.is_null() {
        drop(Box::from_raw(cursor));
    }
}
