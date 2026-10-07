//! Bounded leased C cursors for contextual-cost and prefix-pruned traversals.

use super::generated::LlevStatus;
use super::index::{binding, boundary, utf8, LlevTransducer};
use crate::bindings::{
    CallbackContextualCost, CallbackPrefixPruner, ContextualCostCallback, PrefixCallback,
    SpecializedMatch, SpecializedQueryCursor,
};
use std::ffi::{c_char, c_void};
use std::ptr;

pub use crate::bindings::EditContextView as LlevEditContextView;

/// Borrowed Unicode descriptor. `cost` is an exact contextual cost or an
/// integral prefix-query distance represented as f64. `score` is meaningful
/// only when `has_score` is one.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevSpecializedMatch {
    /// Borrowed UTF-8 bytes valid until exact batch release.
    pub term_data: *const c_char,
    /// Number of initialized UTF-8 bytes.
    pub byte_len: usize,
    /// Contextual cost or integral prefix-query distance.
    pub cost: f64,
    /// Prefix visitor score when `has_score` is one.
    pub score: f64,
    /// One if a prefix score was supplied.
    pub has_score: u8,
    /// Fixed to zero for this API revision.
    pub reserved: [u8; 7],
}

/// One generation-checked lease over cursor-owned Unicode match storage.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevSpecializedBatchView {
    /// Borrowed contiguous descriptors.
    pub matches: *const LlevSpecializedMatch,
    /// Number of initialized descriptors.
    pub len: usize,
    /// Nonzero generation token required for release.
    pub generation: u64,
}

impl Default for LlevSpecializedBatchView {
    fn default() -> Self {
        Self {
            matches: ptr::null(),
            len: 0,
            generation: 0,
        }
    }
}

/// Opaque, exclusive specialized cursor retaining a dictionary snapshot.
pub struct LlevSpecializedCursor {
    inner: SpecializedQueryCursor,
    batch: Vec<SpecializedMatch>,
    views: Vec<LlevSpecializedMatch>,
    offsets: Vec<usize>,
    arena: Vec<u8>,
    generation: u64,
    leased: bool,
}

/// Synchronous reducer over borrowed descriptors; return End to stop.
pub type LlevSpecializedBatchReducer = unsafe extern "C" fn(
    context: *mut c_void,
    matches: *const LlevSpecializedMatch,
    len: usize,
) -> u32;

fn write_cursor(
    out: *mut *mut LlevSpecializedCursor,
    inner: SpecializedQueryCursor,
) -> Result<LlevStatus, (LlevStatus, String)> {
    if out.is_null() {
        return Err((LlevStatus::NullPointer, "out_cursor is null".into()));
    }
    let cursor = LlevSpecializedCursor {
        inner,
        batch: Vec::new(),
        views: Vec::new(),
        offsets: Vec::new(),
        arena: Vec::new(),
        generation: 0,
        leased: false,
    };
    unsafe { out.write(Box::into_raw(Box::new(cursor))) };
    Ok(LlevStatus::Ok)
}

/// Capture one Unicode dictionary revision and start a contextual-cost query.
/// The callback and context must remain valid until the cursor is freed.
///
/// # Safety
/// All non-null pointers must be valid. `query` must be UTF-8. The callback
/// must not unwind across the C boundary.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_query_contextual_utf8(
    transducer: *const LlevTransducer,
    query: *const c_char,
    query_len: usize,
    max_cost: f64,
    minimum_nonzero_cost: f64,
    callback: Option<ContextualCostCallback>,
    context: *mut c_void,
    out_cursor: *mut *mut LlevSpecializedCursor,
) -> LlevStatus {
    boundary(|| {
        let transducer = transducer
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
        let callback = callback.ok_or((LlevStatus::NullPointer, "cost callback is null".into()))?;
        if out_cursor.is_null() {
            return Err((LlevStatus::NullPointer, "out_cursor is null".into()));
        }
        let inner = binding(unsafe {
            transducer.inner.query_contextual_utf8(
                utf8(query, query_len)?,
                max_cost,
                CallbackContextualCost::new(context, callback, minimum_nonzero_cost),
            )
        })?;
        write_cursor(out_cursor, inner)
    })
}

/// Capture one Unicode dictionary revision and start a balanced prefix DFS.
/// The callback and context must remain valid until the cursor is freed.
///
/// # Safety
/// All non-null pointers must be valid. `query` must be UTF-8. The callback
/// must not unwind across the C boundary.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_query_pruned_utf8(
    transducer: *const LlevTransducer,
    query: *const c_char,
    query_len: usize,
    max_distance: usize,
    callback: Option<PrefixCallback>,
    context: *mut c_void,
    out_cursor: *mut *mut LlevSpecializedCursor,
) -> LlevStatus {
    boundary(|| {
        let transducer = transducer
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
        let callback =
            callback.ok_or((LlevStatus::NullPointer, "prefix callback is null".into()))?;
        if out_cursor.is_null() {
            return Err((LlevStatus::NullPointer, "out_cursor is null".into()));
        }
        let inner = binding(unsafe {
            transducer.inner.query_pruned_utf8(
                utf8(query, query_len)?,
                max_distance,
                CallbackPrefixPruner { context, callback },
            )
        })?;
        write_cursor(out_cursor, inner)
    })
}

fn fill_batch(
    cursor: &mut LlevSpecializedCursor,
    maximum: usize,
) -> Result<LlevStatus, (LlevStatus, String)> {
    if cursor.leased {
        return Err((
            LlevStatus::BatchInUse,
            "release the current batch before advancing".into(),
        ));
    }
    let count = binding(cursor.inner.next_batch(&mut cursor.batch, maximum))?;
    if count == 0 {
        return Ok(LlevStatus::End);
    }
    cursor.views.clear();
    cursor.offsets.clear();
    cursor.arena.clear();
    cursor.views.reserve(count);
    cursor.offsets.reserve(count);
    for item in &cursor.batch {
        cursor.offsets.push(cursor.arena.len());
        cursor.arena.extend_from_slice(item.term.as_bytes());
        cursor.views.push(LlevSpecializedMatch {
            term_data: ptr::null(),
            byte_len: item.term.len(),
            cost: item.cost,
            score: item.score.unwrap_or(0.0),
            has_score: u8::from(item.score.is_some()),
            reserved: [0; 7],
        });
    }
    for (view, offset) in cursor.views.iter_mut().zip(&cursor.offsets) {
        view.term_data = unsafe { cursor.arena.as_ptr().add(*offset).cast() };
    }
    cursor.generation = cursor.generation.wrapping_add(1).max(1);
    cursor.leased = true;
    Ok(LlevStatus::Ok)
}

/// Borrow at most `maximum` descriptors in one generation-checked batch.
///
/// # Safety
/// Both pointers must be valid and the cursor must be exclusively borrowed.
#[no_mangle]
pub unsafe extern "C" fn llev_specialized_cursor_next_batch(
    cursor: *mut LlevSpecializedCursor,
    maximum: usize,
    out_batch: *mut LlevSpecializedBatchView,
) -> LlevStatus {
    boundary(|| {
        let cursor = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "cursor is null".into()))?;
        if out_batch.is_null() {
            return Err((LlevStatus::NullPointer, "out_batch is null".into()));
        }
        out_batch.write(LlevSpecializedBatchView::default());
        let status = fill_batch(cursor, maximum)?;
        if status == LlevStatus::Ok {
            out_batch.write(LlevSpecializedBatchView {
                matches: cursor.views.as_ptr(),
                len: cursor.views.len(),
                generation: cursor.generation,
            });
        }
        Ok(status)
    })
}

/// Settle the live batch generation.
///
/// # Safety
/// `cursor` must be live and exclusively borrowed.
#[no_mangle]
pub unsafe extern "C" fn llev_specialized_cursor_release_batch(
    cursor: *mut LlevSpecializedCursor,
    generation: u64,
) -> LlevStatus {
    boundary(|| {
        let cursor = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "cursor is null".into()))?;
        if !cursor.leased || cursor.generation != generation {
            return Err((
                LlevStatus::InvalidArgument,
                "batch generation is not live".into(),
            ));
        }
        cursor.leased = false;
        Ok(LlevStatus::Ok)
    })
}

/// Consume bounded borrowed batches with a caller-supplied reducer.
///
/// # Safety
/// All pointers must remain valid for the call; callback borrows expire on
/// return and the callback must not unwind.
#[no_mangle]
pub unsafe extern "C" fn llev_specialized_cursor_reduce(
    cursor: *mut LlevSpecializedCursor,
    batch_size: usize,
    reducer: Option<LlevSpecializedBatchReducer>,
    context: *mut c_void,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        let cursor = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "cursor is null".into()))?;
        let reducer = reducer.ok_or((LlevStatus::NullPointer, "reducer is null".into()))?;
        if out_count.is_null() {
            return Err((LlevStatus::NullPointer, "out_count is null".into()));
        }
        let mut count = 0usize;
        loop {
            match fill_batch(cursor, batch_size)? {
                LlevStatus::End => break,
                LlevStatus::Ok => {
                    let raw = reducer(context, cursor.views.as_ptr(), cursor.views.len());
                    count = count.saturating_add(cursor.views.len());
                    cursor.leased = false;
                    match LlevStatus::try_from(raw) {
                        Ok(LlevStatus::Ok) => {}
                        Ok(LlevStatus::End) => break,
                        Ok(other) => return Err((other, "batch reducer aborted".into())),
                        Err(_) => {
                            return Err((
                                LlevStatus::InvalidArgument,
                                "batch reducer returned an invalid status".into(),
                            ))
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        out_count.write(count);
        Ok(LlevStatus::Ok)
    })
}

/// Free a cursor, rejecting a live batch lease.
///
/// # Safety
/// A non-null pointer must name a live cursor and cannot be reused on success.
#[no_mangle]
pub unsafe extern "C" fn llev_specialized_cursor_free(
    cursor: *mut LlevSpecializedCursor,
) -> LlevStatus {
    boundary(|| {
        if cursor.is_null() {
            return Ok(LlevStatus::Ok);
        }
        if (*cursor).leased {
            return Err((
                LlevStatus::BatchInUse,
                "release the batch before closing".into(),
            ));
        }
        drop(Box::from_raw(cursor));
        Ok(LlevStatus::Ok)
    })
}
