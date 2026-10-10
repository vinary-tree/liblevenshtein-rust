//! Bounded persistent native n-gram and hybrid source-filter indexes.

use super::{
    index::{boundary, utf8},
    LlevStatus,
};
use crate::filter::{HybridMatcher, NgramIndex};
use std::ffi::c_char;

/// Index algorithm and hard source/query byte ceilings.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevSourceFilterIndexConfig {
    /// 1 for n-gram; 2 for n-gram plus Jaro-Winkler.
    pub mode: u32,
    /// Must be zero.
    pub reserved: u32,
    /// Positive n-gram width in UTF-8 bytes.
    pub ngram_size: usize,
    /// Finite threshold between 0 and 1 inclusive; zero for n-gram mode.
    pub jaro_threshold: f64,
    /// Maximum distinct retained terms.
    pub max_terms: usize,
    /// Maximum UTF-8 bytes in one term.
    pub max_term_bytes: usize,
    /// Maximum UTF-8 bytes across distinct retained terms.
    pub max_source_bytes: usize,
    /// Maximum UTF-8 bytes in a query.
    pub max_query_bytes: usize,
}

/// Fail-closed candidate, result, and hybrid comparison ceilings.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevSourceFilterIndexLimits {
    /// Upper bound on the source cardinality admitted to this query.
    pub max_candidates: usize,
    /// Upper bound on the complete candidate set.
    pub max_results: usize,
    /// Upper bound on the conservative query-character × source-byte work.
    pub max_comparisons: usize,
}

enum NativeIndex {
    Ngram(NgramIndex),
    Hybrid(HybridMatcher),
}

impl NativeIndex {
    fn len(&self) -> usize {
        match self {
            Self::Ngram(index) => index.len(),
            Self::Hybrid(index) => index.len(),
        }
    }

    fn term_id(&self, term: &str) -> Option<usize> {
        match self {
            Self::Ngram(index) => index.term_id(term),
            Self::Hybrid(index) => index.term_id(term),
        }
    }

    fn insert(&mut self, term: &str) -> usize {
        match self {
            Self::Ngram(index) => index.insert(term),
            Self::Hybrid(index) => {
                index.insert(term);
                index.term_id(term).expect("inserted hybrid term has an ID")
            }
        }
    }

    fn candidates(&self, query: &str, max_distance: usize) -> Vec<&str> {
        match self {
            Self::Ngram(index) => index.find_candidates(query, max_distance),
            Self::Hybrid(index) => index.filter_candidates(query, max_distance),
        }
    }

    fn needs_comparisons(&self) -> bool {
        matches!(self, Self::Hybrid(index) if index.jaro_threshold() > 0.0)
    }
}

/// Opaque native index. Freeze is idempotent; all query results are copied.
pub struct LlevSourceFilterIndex {
    index: NativeIndex,
    frozen: bool,
    max_terms: usize,
    max_term_bytes: usize,
    max_source_bytes: usize,
    max_query_bytes: usize,
    source_bytes: usize,
}

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

/// Create a persistent native filter index.
///
/// # Safety
/// The config and output pointers must be valid, writable where appropriate,
/// and disjoint. The output receives one owning handle on success.
#[no_mangle]
pub unsafe extern "C" fn llev_source_filter_index_new(
    config: *const LlevSourceFilterIndexConfig,
    out_index: *mut *mut LlevSourceFilterIndex,
) -> LlevStatus {
    boundary(|| {
        let output = out_index.as_mut().ok_or((
            LlevStatus::NullPointer,
            "source filter index output is null".into(),
        ))?;
        *output = std::ptr::null_mut();
        let raw = *config.as_ref().ok_or((
            LlevStatus::NullPointer,
            "source filter index config is null".into(),
        ))?;
        if !matches!(raw.mode, 1 | 2)
            || raw.reserved != 0
            || raw.ngram_size == 0
            || !raw.jaro_threshold.is_finite()
            || !(0.0..=1.0).contains(&raw.jaro_threshold)
            || (raw.mode == 1 && raw.jaro_threshold != 0.0)
        {
            return Err(invalid("invalid persistent source filter configuration"));
        }
        let index = if raw.mode == 1 {
            NativeIndex::Ngram(NgramIndex::new(raw.ngram_size))
        } else {
            NativeIndex::Hybrid(HybridMatcher::with_config(
                std::iter::empty::<String>(),
                raw.ngram_size,
                raw.jaro_threshold,
            ))
        };
        *output = Box::into_raw(Box::new(LlevSourceFilterIndex {
            index,
            frozen: false,
            max_terms: raw.max_terms,
            max_term_bytes: raw.max_term_bytes,
            max_source_bytes: raw.max_source_bytes,
            max_query_bytes: raw.max_query_bytes,
            source_bytes: 0,
        }));
        Ok(LlevStatus::Ok)
    })
}

/// Copy one UTF-8 term and return its stable insertion ID. Duplicate terms
/// retain their old IDs and consume no new source quota.
///
/// # Safety
/// The index must be live and exclusively accessible. The term bytes and
/// output must be valid, disjoint storage; bytes are borrowed only here.
#[no_mangle]
pub unsafe extern "C" fn llev_source_filter_index_insert(
    index: *mut LlevSourceFilterIndex,
    term: *const c_char,
    term_len: usize,
    out_id: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let output = out_id.as_mut().ok_or((
            LlevStatus::NullPointer,
            "source filter term ID output is null".into(),
        ))?;
        *output = 0;
        let handle = index.as_mut().ok_or((
            LlevStatus::NullPointer,
            "source filter index is null".into(),
        ))?;
        if handle.frozen {
            return Err(invalid("source filter index is frozen"));
        }
        if term_len > handle.max_term_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "source filter term byte limit".into(),
            ));
        }
        let term = utf8(term, term_len)?;
        if let Some(id) = handle.index.term_id(term) {
            *output = id;
            return Ok(LlevStatus::Ok);
        }
        let next_terms = handle.index.len().checked_add(1).ok_or((
            LlevStatus::LimitExceeded,
            "source filter term count overflow".into(),
        ))?;
        let next_bytes = handle.source_bytes.checked_add(term_len).ok_or((
            LlevStatus::LimitExceeded,
            "source filter byte count overflow".into(),
        ))?;
        if next_terms > handle.max_terms || next_bytes > handle.max_source_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "source filter index ingestion limit".into(),
            ));
        }
        let id = handle.index.insert(term);
        handle.source_bytes = next_bytes;
        *output = id;
        Ok(LlevStatus::Ok)
    })
}

/// Seal the index for concurrent immutable queries.
///
/// # Safety
/// The index must be live and exclusively accessible during this call.
#[no_mangle]
pub unsafe extern "C" fn llev_source_filter_index_freeze(
    index: *mut LlevSourceFilterIndex,
) -> LlevStatus {
    boundary(|| {
        let handle = index.as_mut().ok_or((
            LlevStatus::NullPointer,
            "source filter index is null".into(),
        ))?;
        handle.frozen = true;
        Ok(LlevStatus::Ok)
    })
}

/// Return the complete native candidate set as stable insertion IDs sorted
/// in source order. No partial result is published on exhaustion.
///
/// # Safety
/// The index must be live and immutable. Input and output pointers must be
/// valid and disjoint. `out_ids` holds `capacity` writable elements when
/// capacity is nonzero. Output storage remains caller owned.
#[no_mangle]
pub unsafe extern "C" fn llev_source_filter_index_query(
    index: *const LlevSourceFilterIndex,
    query: *const c_char,
    query_len: usize,
    max_distance: usize,
    raw_limits: *const LlevSourceFilterIndexLimits,
    out_ids: *mut usize,
    capacity: usize,
    out_len: *mut usize,
    out_reason: *mut u32,
) -> LlevStatus {
    boundary(|| {
        let written = out_len.as_mut().ok_or((
            LlevStatus::NullPointer,
            "source filter candidate count output is null".into(),
        ))?;
        let reason = out_reason.as_mut().ok_or((
            LlevStatus::NullPointer,
            "source filter reason output is null".into(),
        ))?;
        *written = 0;
        *reason = 0;
        let handle = index.as_ref().ok_or((
            LlevStatus::NullPointer,
            "source filter index is null".into(),
        ))?;
        let limits = raw_limits.as_ref().ok_or((
            LlevStatus::NullPointer,
            "source filter query limits are null".into(),
        ))?;
        if !handle.frozen {
            return Err(invalid("source filter index must be frozen before query"));
        }
        if query_len > handle.max_query_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "source filter query byte limit".into(),
            ));
        }
        let query = utf8(query, query_len)?;
        if handle.index.len() > limits.max_candidates {
            *reason = 1;
            return Err((
                LlevStatus::LimitExceeded,
                "source filter candidate ceiling".into(),
            ));
        }
        if handle.index.needs_comparisons() {
            let comparisons = match query.chars().count().checked_mul(handle.source_bytes) {
                Some(comparisons) => comparisons,
                None => {
                    *reason = 4;
                    return Err((
                        LlevStatus::LimitExceeded,
                        "source filter comparison count overflow".into(),
                    ));
                }
            };
            if comparisons > limits.max_comparisons {
                *reason = 4;
                return Err((
                    LlevStatus::LimitExceeded,
                    "source filter comparison ceiling".into(),
                ));
            }
        }
        let candidates = handle.index.candidates(query, max_distance);
        if candidates.len() > limits.max_results {
            *reason = 2;
            return Err((
                LlevStatus::LimitExceeded,
                "source filter result ceiling".into(),
            ));
        }
        if candidates.len() > capacity {
            *reason = 3;
            return Err((
                LlevStatus::LimitExceeded,
                "source filter output capacity".into(),
            ));
        }
        if capacity != 0 {
            if out_ids.is_null() {
                return Err((
                    LlevStatus::NullPointer,
                    "source filter output IDs are null".into(),
                ));
            }
            if !(out_ids as usize).is_multiple_of(std::mem::align_of::<usize>()) {
                return Err(invalid("source filter output IDs are not aligned"));
            }
        }
        let mut ids = Vec::new();
        ids.try_reserve_exact(candidates.len()).map_err(|_| {
            *reason = 2;
            (
                LlevStatus::LimitExceeded,
                "source filter result allocation failed".into(),
            )
        })?;
        for term in candidates {
            ids.push(handle.index.term_id(term).ok_or((
                LlevStatus::Panic,
                "source filter candidate lost its insertion ID".into(),
            ))?);
        }
        ids.sort_unstable();
        if !ids.is_empty() {
            std::ptr::copy_nonoverlapping(ids.as_ptr(), out_ids, ids.len());
        }
        *written = ids.len();
        Ok(LlevStatus::Ok)
    })
}

/// Release an owning index handle.
///
/// # Safety
/// The pointer must be a live handle returned by the constructor, freed once.
#[no_mangle]
pub unsafe extern "C" fn llev_source_filter_index_free(index: *mut LlevSourceFilterIndex) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}
