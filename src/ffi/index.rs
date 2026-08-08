//! Stable resource-transducer and leased streaming cursor C ABI.

use super::{
    LlevAlgorithm, LlevQueryOrder, LlevStatus, LLEV_ABI_VERSION, LLEV_API_REVISION,
    LLEV_BUILD_FEATURE_CORE, LLEV_BUILD_FEATURE_PHONETIC,
};
use crate::bindings::{
    BindingError, MatchBatch, MatchTerm, QueryCursor, QueryOrder, ResourceTransducer,
};
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use vinary_tree_interop::{VtResource, VtStatus, VtUnitDomain};

/// Opaque Levenshtein automaton configuration over a retained dictionary
/// resource.
pub struct LlevTransducer {
    pub(crate) inner: ResourceTransducer,
}

/// Domain-neutral borrowed match descriptor.
///
/// For byte and Unicode-scalar dictionaries, `term_data` addresses `byte_len`
/// bytes. Unicode-scalar terms are encoded as UTF-8 and `term_len` is their
/// scalar count. For u64 dictionaries, `term_data` is aligned for `uint64_t`,
/// `term_len` is the number of tokens, and `byte_len == term_len * 8`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevMatch {
    /// Cursor-owned term storage.
    pub term_data: *const c_void,
    /// Number of native units in the term.
    pub term_len: usize,
    /// Number of addressable bytes in `term_data`.
    pub byte_len: usize,
    /// Exact edit distance.
    pub distance: usize,
    /// Optional provider value.
    pub id: u64,
    /// One of [`VtUnitDomain`].
    pub unit_domain: VtUnitDomain,
    /// Zero or one.
    pub has_id: u8,
    /// Reserved; fixed to zero.
    pub reserved: [u8; 3],
}

/// One borrowed, generation-checked view over cursor-owned reusable storage.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevMatchBatchView {
    /// Contiguous match descriptors.
    pub matches: *const LlevMatch,
    /// Number of descriptors.
    pub len: usize,
    /// Lease generation passed to [`llev_query_cursor_release_batch`].
    pub generation: u64,
}

impl Default for LlevMatchBatchView {
    fn default() -> Self {
        Self {
            matches: ptr::null(),
            len: 0,
            generation: 0,
        }
    }
}

/// Opaque lazy cursor. It retains the immutable dictionary revision captured by
/// the query and owns all memory exposed through a batch lease.
pub struct LlevQueryCursor {
    inner: QueryCursor,
    batch: MatchBatch,
    views: Vec<LlevMatch>,
    offsets: Vec<usize>,
    byte_arena: Vec<u8>,
    u64_arena: Vec<u64>,
    generation: u64,
    leased: bool,
}

/// Batch reducer callback. Returning `End` stops reduction successfully;
/// another non-`Ok` status aborts and is returned to the caller.
pub type LlevBatchReducer =
    unsafe extern "C" fn(context: *mut c_void, matches: *const LlevMatch, len: usize) -> LlevStatus;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(message: impl AsRef<str>) {
    let sanitized = message.as_ref().replace('\0', "\\0");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(sanitized).unwrap_or_default();
    });
}

fn map_binding_error(error: &BindingError) -> LlevStatus {
    match error {
        BindingError::NullResource => LlevStatus::NullPointer,
        BindingError::UnitDomainMismatch { .. } => LlevStatus::DomainMismatch,
        BindingError::UnsupportedValueDomain(_) | BindingError::UnsupportedOrdering(_) => {
            LlevStatus::Unsupported
        }
        BindingError::EmptyBatch => LlevStatus::InvalidArgument,
        BindingError::Provider(status) => match status {
            VtStatus::InvalidArgument => LlevStatus::InvalidArgument,
            VtStatus::NullPointer => LlevStatus::NullPointer,
            VtStatus::Unsupported => LlevStatus::Unsupported,
            VtStatus::IoError => LlevStatus::IoError,
            VtStatus::Closed => LlevStatus::Closed,
            VtStatus::LimitExceeded => LlevStatus::LimitExceeded,
            _ => LlevStatus::ProviderError,
        },
        BindingError::IncompatibleResourceAbi
        | BindingError::MissingDictionaryInterface
        | BindingError::IncompatibleDictionaryInterface
        | BindingError::InvalidProviderOutput(_) => LlevStatus::ProviderError,
    }
}

pub(crate) fn boundary(
    operation: impl FnOnce() -> Result<LlevStatus, (LlevStatus, String)>,
) -> LlevStatus {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(status)) => {
            if matches!(status, LlevStatus::Ok | LlevStatus::End) {
                set_last_error("");
            }
            status
        }
        Ok(Err((status, message))) => {
            set_last_error(message);
            status
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("panic in liblevenshtein");
            set_last_error(message);
            LlevStatus::Panic
        }
    }
}

fn binding<T>(result: Result<T, BindingError>) -> Result<T, (LlevStatus, String)> {
    result.map_err(|error| (map_binding_error(&error), error.to_string()))
}

unsafe fn slice<'a, T>(
    data: *const T,
    len: usize,
    name: &str,
) -> Result<&'a [T], (LlevStatus, String)> {
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err((LlevStatus::NullPointer, format!("{name} is null")));
    }
    Ok(std::slice::from_raw_parts(data, len))
}

pub(crate) unsafe fn utf8<'a>(
    data: *const c_char,
    len: usize,
) -> Result<&'a str, (LlevStatus, String)> {
    let bytes = slice(data.cast::<u8>(), len, "UTF-8 input")?;
    std::str::from_utf8(bytes).map_err(|error| (LlevStatus::InvalidUtf8, error.to_string()))
}

fn parse_algorithm(value: u32) -> Result<crate::transducer::Algorithm, (LlevStatus, String)> {
    LlevAlgorithm::try_from(value)
        .map(Into::into)
        .map_err(|()| {
            (
                LlevStatus::InvalidArgument,
                format!("unknown algorithm value {value}"),
            )
        })
}

fn parse_order(value: u32) -> Result<QueryOrder, (LlevStatus, String)> {
    LlevQueryOrder::try_from(value)
        .map(|order| match order {
            LlevQueryOrder::Traversal => QueryOrder::Traversal,
            LlevQueryOrder::DistanceThenTerm => QueryOrder::DistanceThenTerm,
        })
        .map_err(|()| {
            (
                LlevStatus::InvalidArgument,
                format!("unknown query-order value {value}"),
            )
        })
}

/// Return the stable liblevenshtein ABI version.
#[no_mangle]
pub extern "C" fn llev_abi_version() -> u32 {
    LLEV_ABI_VERSION
}

/// Return the additive API revision.
#[no_mangle]
pub extern "C" fn llev_api_revision() -> u32 {
    LLEV_API_REVISION
}

/// Return compiled optional binding features.
#[no_mangle]
pub extern "C" fn llev_build_features() -> u64 {
    let phonetic = if cfg!(feature = "bindings-phonetic") {
        LLEV_BUILD_FEATURE_PHONETIC
    } else {
        0
    };
    LLEV_BUILD_FEATURE_CORE | phonetic
}

/// Return a thread-local error message owned by the library.
#[no_mangle]
pub extern "C" fn llev_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Retain a dictionary resource and construct a Levenshtein transducer.
///
/// # Safety
///
/// `dictionary` and `out_transducer` must be readable/writable respectively;
/// the resource must implement the vinary-tree interop contract.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_new(
    dictionary: *const VtResource,
    algorithm: u32,
    out_transducer: *mut *mut LlevTransducer,
) -> LlevStatus {
    boundary(|| {
        let dictionary = dictionary
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "dictionary is null".into()))?;
        if out_transducer.is_null() {
            return Err((LlevStatus::NullPointer, "out_transducer is null".into()));
        }
        let inner = binding(ResourceTransducer::from_resource(
            *dictionary,
            parse_algorithm(algorithm)?,
        ))?;
        out_transducer.write(Box::into_raw(Box::new(LlevTransducer { inner })));
        Ok(LlevStatus::Ok)
    })
}

/// Release a transducer retain. Existing query cursors remain valid.
///
/// # Safety
///
/// A non-null pointer must be a live handle returned by
/// [`llev_transducer_new`] and cannot be used afterward.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_free(transducer: *mut LlevTransducer) {
    if !transducer.is_null() {
        drop(Box::from_raw(transducer));
    }
}

/// Return the dictionary unit domain.
///
/// # Safety
///
/// Both pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_unit_domain(
    transducer: *const LlevTransducer,
    out_domain: *mut VtUnitDomain,
) -> LlevStatus {
    boundary(|| {
        let transducer = transducer
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
        if out_domain.is_null() {
            return Err((LlevStatus::NullPointer, "out_domain is null".into()));
        }
        out_domain.write(transducer.inner.unit_domain());
        Ok(LlevStatus::Ok)
    })
}

pub(crate) fn write_cursor(
    result: Result<QueryCursor, BindingError>,
    out_cursor: *mut *mut LlevQueryCursor,
) -> Result<LlevStatus, (LlevStatus, String)> {
    if out_cursor.is_null() {
        return Err((LlevStatus::NullPointer, "out_cursor is null".into()));
    }
    let inner = binding(result)?;
    let cursor = LlevQueryCursor {
        inner,
        batch: MatchBatch::default(),
        views: Vec::with_capacity(crate::bindings::DEFAULT_MATCH_BATCH),
        offsets: Vec::with_capacity(crate::bindings::DEFAULT_MATCH_BATCH),
        byte_arena: Vec::new(),
        u64_arena: Vec::new(),
        generation: 0,
        leased: false,
    };
    unsafe { out_cursor.write(Box::into_raw(Box::new(cursor))) };
    Ok(LlevStatus::Ok)
}

/// Start a lazy Unicode-scalar query and capture its dictionary snapshot.
///
/// # Safety
///
/// All non-optional pointers must be valid and query bytes must be UTF-8.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_query_utf8(
    transducer: *const LlevTransducer,
    query: *const c_char,
    query_len: usize,
    max_distance: usize,
    order: u32,
    out_cursor: *mut *mut LlevQueryCursor,
) -> LlevStatus {
    boundary(|| {
        let transducer = transducer
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
        write_cursor(
            transducer
                .inner
                .query_utf8(utf8(query, query_len)?, max_distance, parse_order(order)?),
            out_cursor,
        )
    })
}

/// Start a lazy raw-byte query and capture its dictionary snapshot.
///
/// # Safety
///
/// `query` must address `query_len` bytes when non-empty.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_query_bytes(
    transducer: *const LlevTransducer,
    query: *const u8,
    query_len: usize,
    max_distance: usize,
    order: u32,
    out_cursor: *mut *mut LlevQueryCursor,
) -> LlevStatus {
    boundary(|| {
        let transducer = transducer
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
        write_cursor(
            transducer.inner.query_bytes(
                slice(query, query_len, "byte query")?,
                max_distance,
                parse_order(order)?,
            ),
            out_cursor,
        )
    })
}

/// Start a lazy u64-token query and capture its dictionary snapshot.
///
/// # Safety
///
/// `query` must address `query_len` aligned u64 values when non-empty.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_query_u64(
    transducer: *const LlevTransducer,
    query: *const u64,
    query_len: usize,
    max_distance: usize,
    order: u32,
    out_cursor: *mut *mut LlevQueryCursor,
) -> LlevStatus {
    boundary(|| {
        let transducer = transducer
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
        write_cursor(
            transducer.inner.query_u64(
                slice(query, query_len, "u64 query")?,
                max_distance,
                parse_order(order)?,
            ),
            out_cursor,
        )
    })
}

fn fill_batch(
    cursor: &mut LlevQueryCursor,
    max_matches: usize,
) -> Result<LlevStatus, (LlevStatus, String)> {
    if cursor.leased {
        return Err((
            LlevStatus::BatchInUse,
            "release the current batch before advancing the cursor".into(),
        ));
    }
    let count = binding(cursor.inner.next_batch(&mut cursor.batch, max_matches))?;
    if count == 0 {
        return Ok(LlevStatus::End);
    }

    cursor.views.clear();
    cursor.offsets.clear();
    cursor.byte_arena.clear();
    cursor.u64_arena.clear();
    cursor.views.reserve(count);
    cursor.offsets.reserve(count);

    for item in cursor.batch.as_slice() {
        let (unit_domain, term_len, byte_len, offset) = match &item.term {
            MatchTerm::Utf8(term) => {
                let offset = cursor.byte_arena.len();
                cursor.byte_arena.extend_from_slice(term.as_bytes());
                (
                    VtUnitDomain::UnicodeScalar,
                    term.chars().count(),
                    term.len(),
                    offset,
                )
            }
            MatchTerm::Bytes(term) => {
                let offset = cursor.byte_arena.len();
                cursor.byte_arena.extend_from_slice(term);
                (VtUnitDomain::Byte, term.len(), term.len(), offset)
            }
            MatchTerm::U64(term) => {
                let offset = cursor.u64_arena.len();
                cursor.u64_arena.extend_from_slice(term);
                (
                    VtUnitDomain::U64,
                    term.len(),
                    term.len().saturating_mul(std::mem::size_of::<u64>()),
                    offset,
                )
            }
        };
        cursor.offsets.push(offset);
        cursor.views.push(LlevMatch {
            term_data: ptr::null(),
            term_len,
            byte_len,
            distance: item.distance,
            id: item.id.unwrap_or_default(),
            unit_domain,
            has_id: u8::from(item.id.is_some()),
            reserved: [0; 3],
        });
    }

    for (index, view) in cursor.views.iter_mut().enumerate() {
        let offset = cursor.offsets[index];
        view.term_data = match view.unit_domain {
            VtUnitDomain::Byte | VtUnitDomain::UnicodeScalar => unsafe {
                cursor.byte_arena.as_ptr().add(offset).cast()
            },
            VtUnitDomain::U64 => unsafe { cursor.u64_arena.as_ptr().add(offset).cast() },
        };
    }
    cursor.generation = cursor.generation.wrapping_add(1).max(1);
    cursor.leased = true;
    Ok(LlevStatus::Ok)
}

/// Borrow the next bounded result batch from cursor-owned contiguous arenas.
///
/// The view is valid until it is explicitly released. Advancing or closing the
/// cursor while a view is live returns `LLEV_STATUS_BATCH_IN_USE`.
///
/// # Safety
///
/// `cursor` and `out_batch` must be valid and exclusively accessed.
#[no_mangle]
pub unsafe extern "C" fn llev_query_cursor_next_batch(
    cursor: *mut LlevQueryCursor,
    max_matches: usize,
    out_batch: *mut LlevMatchBatchView,
) -> LlevStatus {
    boundary(|| {
        let cursor = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "cursor is null".into()))?;
        if out_batch.is_null() {
            return Err((LlevStatus::NullPointer, "out_batch is null".into()));
        }
        out_batch.write(LlevMatchBatchView::default());
        let result = fill_batch(cursor, max_matches)?;
        if result == LlevStatus::Ok {
            out_batch.write(LlevMatchBatchView {
                matches: cursor.views.as_ptr(),
                len: cursor.views.len(),
                generation: cursor.generation,
            });
        }
        Ok(result)
    })
}

/// Release a borrowed batch generation.
///
/// # Safety
///
/// `cursor` must be valid and `generation` must identify its live lease.
#[no_mangle]
pub unsafe extern "C" fn llev_query_cursor_release_batch(
    cursor: *mut LlevQueryCursor,
    generation: u64,
) -> LlevStatus {
    boundary(|| {
        let cursor = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "cursor is null".into()))?;
        if !cursor.leased || generation != cursor.generation {
            return Err((
                LlevStatus::InvalidArgument,
                "batch generation is not the cursor's live lease".into(),
            ));
        }
        cursor.leased = false;
        Ok(LlevStatus::Ok)
    })
}

/// Consume a cursor through one callback per reusable batch.
///
/// This is the allocation-minimizing managed-language path: it never creates a
/// managed result object unless the callback chooses to materialize one.
///
/// # Safety
///
/// The cursor and callback must be valid. The callback may read the descriptors
/// only for the duration of the call and must not free them.
#[no_mangle]
pub unsafe extern "C" fn llev_query_cursor_reduce(
    cursor: *mut LlevQueryCursor,
    batch_size: usize,
    reducer: Option<LlevBatchReducer>,
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
        if cursor.leased {
            return Err((
                LlevStatus::BatchInUse,
                "release the current batch before reducing the cursor".into(),
            ));
        }
        let mut count = 0usize;
        loop {
            match fill_batch(cursor, batch_size)? {
                LlevStatus::End => break,
                LlevStatus::Ok => {
                    let callback_status =
                        reducer(context, cursor.views.as_ptr(), cursor.views.len());
                    count = count.saturating_add(cursor.views.len());
                    cursor.leased = false;
                    match callback_status {
                        LlevStatus::Ok => {}
                        LlevStatus::End => break,
                        other => {
                            return Err((other, "batch reducer aborted the query".into()));
                        }
                    }
                }
                _ => unreachable!("fill_batch returns only Ok or End"),
            }
        }
        out_count.write(count);
        Ok(LlevStatus::Ok)
    })
}

/// Close a query cursor. A live batch lease must be released first.
///
/// # Safety
///
/// A non-null pointer must identify a live cursor. On success it cannot be used
/// again. On `BatchInUse`, ownership remains with the caller.
#[no_mangle]
pub unsafe extern "C" fn llev_query_cursor_free(cursor: *mut LlevQueryCursor) -> LlevStatus {
    boundary(|| {
        if cursor.is_null() {
            return Ok(LlevStatus::Ok);
        }
        if (*cursor).leased {
            return Err((
                LlevStatus::BatchInUse,
                "release the current batch before closing the cursor".into(),
            ));
        }
        drop(Box::from_raw(cursor));
        Ok(LlevStatus::Ok)
    })
}
