//! Frozen native temporal indexes and bounded, resumable C range cursors.

use super::{
    index::{boundary, slice},
    temporal::{check_config, LlevTemporalAlgorithm, LlevTemporalConfig},
    LlevStatus,
};
use crate::{
    cost::CostMonoid,
    time_series::{
        elastic::{ElasticKernel, ElasticTransducer, RangeContinuation},
        DtwConfig, ErpConfig, FrechetConfig, IncompleteReason, MsmConfig, MsmKernel,
        OperationOutcome, PageBudget, QuantizationConfig, ResourceLimits, TemporalValidationError,
        TwedConfig,
    },
};
use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
};

type MsmIndex = ElasticTransducer<MsmKernel, u64>;
type ErpIndex = ElasticTransducer<ErpConfig, u64>;
type TwedIndex = ElasticTransducer<TwedConfig, u64>;
type DtwIndex = ElasticTransducer<DtwConfig, u64>;
type FrechetIndex = ElasticTransducer<FrechetConfig, u64>;

enum Building {
    Msm(MsmIndex),
    Erp(ErpIndex),
    Twed(TwedIndex),
    Dtw(DtwIndex),
    Frechet(FrechetIndex),
}

enum Frozen {
    Msm(Arc<MsmIndex>),
    Erp(Arc<ErpIndex>),
    Twed(Arc<TwedIndex>),
    Dtw(Arc<DtwIndex>),
    Frechet(Arc<FrechetIndex>),
}

enum IndexState {
    Empty,
    Building(Building),
    Frozen(Frozen),
}

/// Opaque temporal index handle. Mutation requires exclusive access; frozen
/// handles may start independent concurrent range cursors.
pub struct LlevTemporalIndex {
    state: IndexState,
    max_entries: usize,
    max_total_samples: usize,
    max_series_len: usize,
    entries: usize,
    total_samples: usize,
}

/// Bounded construction parameters for one frozen index.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTemporalIndexConfig {
    /// Algorithm and kernel parameters; cutoff must be positive infinity.
    pub temporal: LlevTemporalConfig,
    /// Inclusive lower quantization bound.
    pub quant_min: f64,
    /// Inclusive upper quantization bound.
    pub quant_max: f64,
    /// Number of byte bins, between 1 and 256.
    pub quant_bins: u32,
    /// Must be zero.
    pub reserved: u32,
    /// Maximum distinct identifiers retained.
    pub max_entries: usize,
    /// Maximum total samples retained across all identifiers.
    pub max_total_samples: usize,
    /// Maximum samples in one series.
    pub max_series_len: usize,
}

/// Explicit cumulative native search ceilings.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevTemporalSearchLimits {
    /// Maximum samples in a query or indexed series.
    pub max_series_len: usize,
    /// Cumulative dynamic-programming cell ceiling.
    pub max_dp_cells: usize,
    /// Cumulative logical work ceiling.
    pub max_work_units: usize,
    /// Peak temporary byte ceiling.
    pub max_scratch_bytes: usize,
    /// Cumulative trie-node ceiling.
    pub max_trie_nodes: usize,
    /// Cumulative trie-edge ceiling.
    pub max_trie_edges: usize,
    /// Cumulative exact-candidate ceiling.
    pub max_candidates: usize,
    /// Maximum retained exact matches.
    pub max_results: usize,
    /// Peak traversal queue ceiling.
    pub max_queue_entries: usize,
    /// Peak retained continuation byte ceiling.
    pub max_continuation_bytes: usize,
}

impl From<LlevTemporalSearchLimits> for ResourceLimits {
    fn from(raw: LlevTemporalSearchLimits) -> Self {
        Self {
            max_series_len: raw.max_series_len,
            max_dp_cells: raw.max_dp_cells,
            max_work_units: raw.max_work_units,
            max_scratch_bytes: raw.max_scratch_bytes,
            max_trie_nodes: raw.max_trie_nodes,
            max_trie_edges: raw.max_trie_edges,
            max_candidates: raw.max_candidates,
            max_results: raw.max_results,
            max_queue_entries: raw.max_queue_entries,
            max_continuation_bytes: raw.max_continuation_bytes,
            ..ResourceLimits::default()
        }
    }
}

/// An exact native match. DTW scores are returned in root-distance units.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LlevTemporalIndexMatch {
    /// Stored source identifier.
    pub id: u64,
    /// Exact score in the algorithm's public distance units.
    pub distance: f64,
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

fn freeze(building: Building) -> Frozen {
    match building {
        Building::Msm(value) => Frozen::Msm(Arc::new(value)),
        Building::Erp(value) => Frozen::Erp(Arc::new(value)),
        Building::Twed(value) => Frozen::Twed(Arc::new(value)),
        Building::Dtw(value) => Frozen::Dtw(Arc::new(value)),
        Building::Frechet(value) => Frozen::Frechet(Arc::new(value)),
    }
}

/// Allocate an empty index. Soft-DTW has no elastic index and is rejected.
///
/// # Safety
/// Both pointers must be valid; `out_index` must address writable storage.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_index_new(
    config: *const LlevTemporalIndexConfig,
    out_index: *mut *mut LlevTemporalIndex,
) -> LlevStatus {
    boundary(|| {
        let output = out_index.as_mut().ok_or((
            LlevStatus::NullPointer,
            "temporal index output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let raw = *config.as_ref().ok_or((
            LlevStatus::NullPointer,
            "temporal index config is null".into(),
        ))?;
        let algorithm = check_config(raw.temporal, ResourceLimits::default())?;
        if algorithm == LlevTemporalAlgorithm::SoftDtw {
            return Err((
                LlevStatus::Unsupported,
                "Soft-DTW has no elastic index".into(),
            ));
        }
        if raw.temporal.cutoff != f64::INFINITY || raw.reserved != 0 {
            return Err(invalid(
                "index configuration requires infinite cutoff and zero reserved",
            ));
        }
        if raw.quant_bins == 0 || raw.quant_bins > 256 {
            return Err(invalid("temporal quantization requires 1..=256 byte bins"));
        }
        let quant = QuantizationConfig::try_uniform(raw.quant_min, raw.quant_max, raw.quant_bins)
            .ok_or_else(|| invalid("invalid temporal quantization range"))?;
        let state = IndexState::Building(match algorithm {
            LlevTemporalAlgorithm::Msm => Building::Msm(MsmIndex::new(
                quant,
                MsmConfig::new(raw.temporal.parameter0),
            )),
            LlevTemporalAlgorithm::Erp => Building::Erp(ErpIndex::new(
                quant,
                ErpConfig::new(raw.temporal.parameter0),
            )),
            LlevTemporalAlgorithm::Twed => Building::Twed(TwedIndex::new(
                quant,
                TwedConfig::new(raw.temporal.parameter0, raw.temporal.parameter1),
            )),
            LlevTemporalAlgorithm::Dtw => {
                Building::Dtw(DtwIndex::new(quant, DtwConfig::new(raw.temporal.band)))
            }
            LlevTemporalAlgorithm::Frechet => {
                Building::Frechet(FrechetIndex::new(quant, FrechetConfig))
            }
            LlevTemporalAlgorithm::SoftDtw => unreachable!(),
        });
        *output = Box::into_raw(Box::new(LlevTemporalIndex {
            state,
            max_entries: raw.max_entries,
            max_total_samples: raw.max_total_samples,
            max_series_len: raw.max_series_len,
            entries: 0,
            total_samples: 0,
        }));
        Ok(LlevStatus::Ok)
    })
}

/// Insert or replace one ID before freezing. All source limits are checked
/// before mutation; replacement preserves the old entry if insertion fails.
///
/// # Safety
/// `index` must be live and exclusively accessed. `samples` must describe
/// a valid immutable array for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_index_insert(
    index: *mut LlevTemporalIndex,
    id: u64,
    samples: *const f64,
    len: usize,
) -> LlevStatus {
    boundary(|| {
        let handle = index
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "temporal index is null".into()))?;
        let series = slice(samples, len, "temporal series")?;
        if len > handle.max_series_len {
            return Err((
                LlevStatus::LimitExceeded,
                "temporal series length limit".into(),
            ));
        }
        if !series.iter().all(|value| value.is_finite()) {
            return Err(invalid("temporal series contains a nonfinite sample"));
        }
        let building = match &mut handle.state {
            IndexState::Building(value) => value,
            _ => return Err((LlevStatus::Closed, "temporal index is frozen".into())),
        };
        let old_len = match building {
            Building::Msm(value) => value.get_original(&id),
            Building::Erp(value) => value.get_original(&id),
            Building::Twed(value) => value.get_original(&id),
            Building::Dtw(value) => value.get_original(&id),
            Building::Frechet(value) => value.get_original(&id),
        }
        .map_or(0, <[f64]>::len);
        let is_new = old_len == 0
            && match building {
                Building::Msm(value) => value.get_original(&id).is_none(),
                Building::Erp(value) => value.get_original(&id).is_none(),
                Building::Twed(value) => value.get_original(&id).is_none(),
                Building::Dtw(value) => value.get_original(&id).is_none(),
                Building::Frechet(value) => value.get_original(&id).is_none(),
            };
        let count = handle.entries.checked_add(usize::from(is_new)).ok_or((
            LlevStatus::LimitExceeded,
            "temporal entry count overflow".into(),
        ))?;
        let total = handle
            .total_samples
            .checked_sub(old_len)
            .and_then(|value| value.checked_add(len))
            .ok_or((
                LlevStatus::LimitExceeded,
                "temporal sample count overflow".into(),
            ))?;
        if count > handle.max_entries || total > handle.max_total_samples {
            return Err((LlevStatus::LimitExceeded, "temporal source limit".into()));
        }
        let result = match building {
            Building::Msm(value) => value.try_insert(id, series),
            Building::Erp(value) => value.try_insert(id, series),
            Building::Twed(value) => value.try_insert(id, series),
            Building::Dtw(value) => value.try_insert(id, series),
            Building::Frechet(value) => value.try_insert(id, series),
        }
        .map_err(|error| (LlevStatus::ProviderError, error.to_string()))?;
        debug_assert_eq!(result, is_new);
        handle.entries = count;
        handle.total_samples = total;
        Ok(LlevStatus::Ok)
    })
}

/// Freeze construction so independent cursor snapshots can share the index.
///
/// # Safety
/// `index` must be live and exclusively accessed during this call.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_index_freeze(index: *mut LlevTemporalIndex) -> LlevStatus {
    boundary(|| {
        let handle = index
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "temporal index is null".into()))?;
        let old = std::mem::replace(&mut handle.state, IndexState::Empty);
        handle.state = match old {
            IndexState::Building(value) => IndexState::Frozen(freeze(value)),
            other => other,
        };
        Ok(LlevStatus::Ok)
    })
}

/// Release an index handle. Live range cursors keep their frozen snapshot.
///
/// # Safety
/// The pointer must be a live handle returned by `llev_temporal_index_new`
/// and must be freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_index_free(index: *mut LlevTemporalIndex) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}

struct NativeCursor<K>
where
    K: ElasticKernel,
    K::Monoid: CostMonoid<Cost = f64>,
{
    // This field must be dropped before _index. Every continuation borrows
    // the immutable Arc allocation held by _index for exactly this lifetime.
    continuation: Option<RangeContinuation<'static, K, u64>>,
    _index: Arc<ElasticTransducer<K, u64>>,
    pending: VecDeque<LlevTemporalIndexMatch>,
    seen: HashSet<u64>,
    done: bool,
    terminal: Option<IncompleteReason>,
    squared_dtw: bool,
}

impl<K> Drop for NativeCursor<K>
where
    K: ElasticKernel,
    K::Monoid: CostMonoid<Cost = f64>,
{
    fn drop(&mut self) {
        self.continuation.take();
    }
}

impl<K> NativeCursor<K>
where
    K: ElasticKernel,
    K::Monoid: CostMonoid<Cost = f64>,
{
    fn new(
        index: Arc<ElasticTransducer<K, u64>>,
        query: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
        squared_dtw: bool,
    ) -> Result<Self, (LlevStatus, String)> {
        // SAFETY: the Arc is retained in the cursor, the index is immutable,
        // and Drop destroys continuation before releasing the Arc. The
        // continuation never escapes the cursor.
        let borrowed: &'static ElasticTransducer<K, u64> = unsafe { &*Arc::as_ptr(&index) };
        let native_cutoff = if squared_dtw { cutoff * cutoff } else { cutoff };
        let outcome = borrowed
            .search_range_bounded(
                query,
                native_cutoff,
                limits,
                PageBudget {
                    max_work_units: 1,
                    max_results: 1,
                },
            )
            .map_err(validation)?;
        let mut cursor = Self {
            continuation: None,
            _index: index,
            pending: VecDeque::new(),
            seen: HashSet::new(),
            done: false,
            terminal: None,
            squared_dtw,
        };
        cursor.accept(outcome)?;
        Ok(cursor)
    }

    fn add(&mut self, matches: &[(u64, f64)]) -> Result<(), (LlevStatus, String)> {
        for &(id, distance) in matches {
            if self.seen.contains(&id) {
                continue;
            }
            self.seen.try_reserve(1).map_err(|_| {
                (
                    LlevStatus::LimitExceeded,
                    "temporal seen-ID allocation failed".into(),
                )
            })?;
            self.pending.try_reserve(1).map_err(|_| {
                (
                    LlevStatus::LimitExceeded,
                    "temporal result allocation failed".into(),
                )
            })?;
            self.seen.insert(id);
            self.pending.push_back(LlevTemporalIndexMatch {
                id,
                distance: if self.squared_dtw {
                    distance.sqrt()
                } else {
                    distance
                },
            });
        }
        Ok(())
    }

    fn accept(
        &mut self,
        outcome: OperationOutcome<Vec<(u64, f64)>, RangeContinuation<'static, K, u64>>,
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
        out: &mut [LlevTemporalIndexMatch],
        page: PageBudget,
    ) -> Result<(usize, bool), (LlevStatus, String)> {
        if self.pending.is_empty() && !self.done && self.terminal.is_none() {
            if let Some(continuation) = self.continuation.take() {
                let before = continuation.usage().work_units;
                self.accept(continuation.resume(page))?;
                if self.pending.is_empty()
                    && self
                        .continuation
                        .as_ref()
                        .is_some_and(|value| value.usage().work_units == before)
                {
                    return Err((
                        LlevStatus::LimitExceeded,
                        "temporal page work budget cannot advance this query".into(),
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
                    LlevStatus::LimitExceeded,
                    format!("temporal range incomplete: {reason:?}",),
                ));
            }
        }
        Ok((written, self.done && self.pending.is_empty()))
    }
}

enum CursorState {
    Msm(NativeCursor<MsmKernel>),
    Erp(NativeCursor<ErpConfig>),
    Twed(NativeCursor<TwedConfig>),
    Dtw(NativeCursor<DtwConfig>),
    Frechet(NativeCursor<FrechetConfig>),
}

/// Opaque closeable range cursor.
pub struct LlevTemporalIndexCursor {
    state: CursorState,
}

/// Begin one exact range query against a frozen snapshot.
///
/// # Safety
/// The index and output pointers must be valid. The query array is borrowed
/// only during this call; the cursor owns its copy. The index may be freed
/// after this call, but not concurrently with it.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_index_query_range(
    index: *const LlevTemporalIndex,
    query: *const f64,
    query_len: usize,
    cutoff: f64,
    limits: *const LlevTemporalSearchLimits,
    out_cursor: *mut *mut LlevTemporalIndexCursor,
) -> LlevStatus {
    boundary(|| {
        let output = out_cursor.as_mut().ok_or((
            LlevStatus::NullPointer,
            "temporal cursor output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let handle = index
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "temporal index is null".into()))?;
        let query = slice(query, query_len, "temporal query")?;
        let raw = *limits.as_ref().ok_or((
            LlevStatus::NullPointer,
            "temporal search limits are null".into(),
        ))?;
        if cutoff.is_nan() || cutoff < 0.0 {
            return Err(invalid("temporal index cutoff must be nonnegative"));
        }
        let limits = ResourceLimits::from(raw);
        let frozen = match &handle.state {
            IndexState::Frozen(value) => value,
            _ => return Err(invalid("temporal index must be frozen before querying")),
        };
        let cursor = match frozen {
            Frozen::Msm(value) => CursorState::Msm(NativeCursor::new(
                Arc::clone(value),
                query,
                cutoff,
                limits,
                false,
            )?),
            Frozen::Erp(value) => CursorState::Erp(NativeCursor::new(
                Arc::clone(value),
                query,
                cutoff,
                limits,
                false,
            )?),
            Frozen::Twed(value) => CursorState::Twed(NativeCursor::new(
                Arc::clone(value),
                query,
                cutoff,
                limits,
                false,
            )?),
            Frozen::Dtw(value) => CursorState::Dtw(NativeCursor::new(
                Arc::clone(value),
                query,
                cutoff,
                limits,
                true,
            )?),
            Frozen::Frechet(value) => CursorState::Frechet(NativeCursor::new(
                Arc::clone(value),
                query,
                cutoff,
                limits,
                false,
            )?),
        };
        *output = Box::into_raw(Box::new(LlevTemporalIndexCursor { state: cursor }));
        Ok(LlevStatus::Ok)
    })
}

/// Advance by at most one native page and copy up to `capacity` matches.
/// A successful empty nonterminal page means the caller should advance again.
///
/// # Safety
/// All pointers must be live and nonoverlapping. `out_matches` must address
/// `capacity` writable elements. The cursor requires exclusive access.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_index_cursor_next_batch(
    cursor: *mut LlevTemporalIndexCursor,
    out_matches: *mut LlevTemporalIndexMatch,
    capacity: usize,
    page_work_units: usize,
    page_results: usize,
    out_len: *mut usize,
    out_done: *mut u8,
) -> LlevStatus {
    boundary(|| {
        let len = out_len.as_mut().ok_or((
            LlevStatus::NullPointer,
            "temporal batch length is null".into(),
        ))?;
        let done = out_done
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "temporal done flag is null".into()))?;
        *len = 0;
        *done = 0;
        if capacity == 0 || page_work_units == 0 || page_results == 0 {
            return Err(invalid("temporal page and batch limits must be positive"));
        }
        let output = std::slice::from_raw_parts_mut(
            out_matches.as_mut().ok_or((
                LlevStatus::NullPointer,
                "temporal batch output is null".into(),
            ))?,
            capacity,
        );
        let cursor = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "temporal cursor is null".into()))?;
        let page = PageBudget {
            max_work_units: page_work_units,
            max_results: page_results,
        };
        let (count, complete) = match &mut cursor.state {
            CursorState::Msm(value) => value.next(output, page)?,
            CursorState::Erp(value) => value.next(output, page)?,
            CursorState::Twed(value) => value.next(output, page)?,
            CursorState::Dtw(value) => value.next(output, page)?,
            CursorState::Frechet(value) => value.next(output, page)?,
        };
        *len = count;
        *done = u8::from(complete);
        Ok(LlevStatus::Ok)
    })
}

/// Cancel and release a range cursor.
///
/// # Safety
/// The pointer must be live and freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn llev_temporal_index_cursor_free(cursor: *mut LlevTemporalIndexCursor) {
    if !cursor.is_null() {
        drop(Box::from_raw(cursor));
    }
}
