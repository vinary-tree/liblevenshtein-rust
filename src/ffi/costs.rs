//! Versioned native cost configuration and domain-neutral leased query cursors.

use super::index::{binding, boundary, slice, utf8, LlevTransducer};
use super::LlevStatus;
use crate::bindings::{CostMatch, CostQueryCursor, MatchTerm};
use crate::cost::{CostScale, ScaleError};
use crate::transducer::{AffineGapParams, OperationCostsF64};
use std::ffi::{c_char, c_void};
use std::ptr;
use vinary_tree_interop::VtUnitDomain;

/// Decimal affine-gap configuration. Zero denominator derives the least exact
/// decimal scale; a nonzero denominator requests that exact scale explicitly.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevAffineCosts {
    /// Cost charged once for each nonempty gap.
    pub gap_open: f64,
    /// Cost charged for every unit in a gap, including its first.
    pub gap_extend: f64,
    /// Substitution cost.
    pub substitution: f64,
    /// Zero to derive or an explicitly requested exact denominator.
    pub scale_denominator: u32,
    /// Must be zero.
    pub reserved: u32,
}

/// Per-operation costs used by the native floating weighted automaton.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevOperationCostsF64 {
    /// Exact zero for matching units.
    pub match_cost: f64,
    /// Replacement cost.
    pub substitution: f64,
    /// Query-side insertion cost.
    pub insertion: f64,
    /// Dictionary-side deletion cost.
    pub deletion: f64,
    /// Adjacent swap cost.
    pub transposition: f64,
    /// One-to-two split cost.
    pub split: f64,
    /// Two-to-one merge cost.
    pub merge: f64,
}

impl From<OperationCostsF64> for LlevOperationCostsF64 {
    fn from(value: OperationCostsF64) -> Self {
        Self {
            match_cost: value.match_cost,
            substitution: value.substitution,
            insertion: value.insertion,
            deletion: value.deletion,
            transposition: value.transposition,
            split: value.split,
            merge: value.merge,
        }
    }
}

impl From<LlevOperationCostsF64> for OperationCostsF64 {
    fn from(value: LlevOperationCostsF64) -> Self {
        Self {
            match_cost: value.match_cost,
            substitution: value.substitution,
            insertion: value.insertion,
            deletion: value.deletion,
            transposition: value.transposition,
            split: value.split,
            merge: value.merge,
        }
    }
}

/// Domain-neutral cost match. `scaled_cost` and `scale_denominator` represent
/// an exact rational only when `has_scaled_cost` is one.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevCostMatch {
    /// Cursor-owned UTF-8, byte, or aligned u64 term storage.
    pub term_data: *const c_void,
    /// Logical unit count.
    pub term_len: usize,
    /// Addressable byte count.
    pub byte_len: usize,
    /// Float presentation cost.
    pub cost: f64,
    /// Exact affine numerator when `has_scaled_cost` is one.
    pub scaled_cost: usize,
    /// Exact affine denominator, or one for weighted float costs.
    pub scale_denominator: u32,
    /// Original dictionary unit domain.
    pub unit_domain: VtUnitDomain,
    /// One for affine, zero for weighted float costs.
    pub has_scaled_cost: u8,
    /// Fixed to zero.
    pub reserved: [u8; 7],
}

/// One generation-checked borrow of cursor-owned results.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevCostBatchView {
    /// Borrowed contiguous descriptors.
    pub matches: *const LlevCostMatch,
    /// Number of initialized descriptors.
    pub len: usize,
    /// Nonzero exact lease token.
    pub generation: u64,
}

impl Default for LlevCostBatchView {
    fn default() -> Self {
        Self {
            matches: ptr::null(),
            len: 0,
            generation: 0,
        }
    }
}

/// Exclusive lazy cursor over one immutable provider snapshot.
pub struct LlevCostCursor {
    inner: CostQueryCursor,
    batch: Vec<CostMatch>,
    views: Vec<LlevCostMatch>,
    offsets: Vec<usize>,
    byte_arena: Vec<u8>,
    u64_arena: Vec<u64>,
    generation: u64,
    leased: bool,
}

/// Synchronous reducer over borrowed cost descriptors; End stops early.
pub type LlevCostBatchReducer =
    unsafe extern "C" fn(context: *mut c_void, matches: *const LlevCostMatch, len: usize) -> u32;

fn invalid(message: impl Into<String>) -> (LlevStatus, String) {
    (LlevStatus::InvalidArgument, message.into())
}

fn map_scale(error: ScaleError) -> (LlevStatus, String) {
    let status = match error {
        ScaleError::DenominatorOverflow | ScaleError::CostOverflow => LlevStatus::LimitExceeded,
        _ => LlevStatus::InvalidArgument,
    };
    (status, error.to_string())
}

fn affine_params(costs: LlevAffineCosts) -> Result<AffineGapParams, (LlevStatus, String)> {
    if costs.reserved != 0 {
        return Err(invalid("affine reserved field must be zero"));
    }
    let result = if costs.scale_denominator == 0 {
        AffineGapParams::new(costs.gap_open, costs.gap_extend, costs.substitution)
    } else {
        let scale = CostScale::new(costs.scale_denominator).map_err(map_scale)?;
        AffineGapParams::with_scale(scale, costs.gap_open, costs.gap_extend, costs.substitution)
    };
    result.map_err(map_scale)
}

fn weighted_costs(costs: LlevOperationCostsF64) -> Result<OperationCostsF64, (LlevStatus, String)> {
    let costs: OperationCostsF64 = costs.into();
    costs.is_valid().then_some(costs).ok_or_else(|| {
        invalid("weighted costs must be finite and nonnegative, with zero match cost")
    })
}

/// Validate an affine configuration using the same exact native scale as query.
///
/// # Safety
/// Both pointers must be valid and `out_denominator` writable.
#[no_mangle]
pub unsafe extern "C" fn llev_affine_costs_validate(
    costs: *const LlevAffineCosts,
    out_denominator: *mut u32,
) -> LlevStatus {
    boundary(|| {
        let costs = costs
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "costs is null".into()))?;
        if out_denominator.is_null() {
            return Err((LlevStatus::NullPointer, "out_denominator is null".into()));
        }
        out_denominator.write(affine_params(*costs)?.scale().denominator());
        Ok(LlevStatus::Ok)
    })
}

/// Native presets: zero standard, one typo-friendly, two OCR-friendly.
///
/// # Safety
/// `out_costs` must be writable.
#[no_mangle]
pub unsafe extern "C" fn llev_operation_costs_preset(
    preset: u32,
    out_costs: *mut LlevOperationCostsF64,
) -> LlevStatus {
    boundary(|| {
        if out_costs.is_null() {
            return Err((LlevStatus::NullPointer, "out_costs is null".into()));
        }
        let costs = match preset {
            0 => OperationCostsF64::standard(),
            1 => OperationCostsF64::typo_friendly(),
            2 => OperationCostsF64::ocr_friendly(),
            _ => return Err(invalid("unknown operation-cost preset")),
        };
        out_costs.write(costs.into());
        Ok(LlevStatus::Ok)
    })
}

/// Validate custom floating operation costs before starting a query.
///
/// # Safety
/// `costs` must point to one readable configuration.
#[no_mangle]
pub unsafe extern "C" fn llev_operation_costs_validate(
    costs: *const LlevOperationCostsF64,
) -> LlevStatus {
    boundary(|| {
        let costs = costs
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "costs is null".into()))?;
        weighted_costs(*costs)?;
        Ok(LlevStatus::Ok)
    })
}

enum QueryInput<'a> {
    Text(&'a str),
    Bytes(&'a [u8]),
    U64(&'a [u64]),
}

unsafe fn query_input<'a>(
    domain: u32,
    query: *const c_void,
    len: usize,
) -> Result<QueryInput<'a>, (LlevStatus, String)> {
    match domain {
        value if value == VtUnitDomain::UnicodeScalar as u32 => {
            utf8(query.cast::<c_char>(), len).map(QueryInput::Text)
        }
        value if value == VtUnitDomain::Byte as u32 => {
            slice(query.cast::<u8>(), len, "byte query").map(QueryInput::Bytes)
        }
        value if value == VtUnitDomain::U64 as u32 => {
            slice(query.cast::<u64>(), len, "u64 query").map(QueryInput::U64)
        }
        _ => Err(invalid("unknown query unit domain")),
    }
}

fn write_cursor(
    cursor: CostQueryCursor,
    out: *mut *mut LlevCostCursor,
) -> Result<LlevStatus, (LlevStatus, String)> {
    if out.is_null() {
        return Err((LlevStatus::NullPointer, "out_cursor is null".into()));
    }
    unsafe {
        out.write(Box::into_raw(Box::new(LlevCostCursor {
            inner: cursor,
            batch: Vec::new(),
            views: Vec::new(),
            offsets: Vec::new(),
            byte_arena: Vec::new(),
            u64_arena: Vec::new(),
            generation: 0,
            leased: false,
        })))
    };
    Ok(LlevStatus::Ok)
}

/// Query a captured Unicode, byte, or u64 provider using exact affine costs.
///
/// # Safety
/// Input pointers must match the stated domain and length. Output is writable.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_query_affine(
    transducer: *const LlevTransducer,
    domain: u32,
    query: *const c_void,
    query_len: usize,
    max_cost: f64,
    costs: *const LlevAffineCosts,
    out_cursor: *mut *mut LlevCostCursor,
) -> LlevStatus {
    boundary(|| {
        let transducer = transducer
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
        let costs = costs
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "costs is null".into()))?;
        if out_cursor.is_null() {
            return Err((LlevStatus::NullPointer, "out_cursor is null".into()));
        }
        let params = affine_params(*costs)?;
        let maximum_scaled = params.scale_cost(max_cost).map_err(map_scale)?;
        let cursor = match query_input(domain, query, query_len)? {
            QueryInput::Text(value) => {
                transducer
                    .inner
                    .query_affine_utf8(value, maximum_scaled, params)
            }
            QueryInput::Bytes(value) => {
                transducer
                    .inner
                    .query_affine_bytes(value, maximum_scaled, params)
            }
            QueryInput::U64(value) => {
                transducer
                    .inner
                    .query_affine_u64(value, maximum_scaled, params)
            }
        };
        write_cursor(binding(cursor)?, out_cursor)
    })
}

/// Query a captured Unicode, byte, or u64 provider using floating edit costs.
///
/// # Safety
/// Input pointers must match the stated domain and length. Output is writable.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_query_weighted(
    transducer: *const LlevTransducer,
    domain: u32,
    query: *const c_void,
    query_len: usize,
    max_cost: f64,
    costs: *const LlevOperationCostsF64,
    out_cursor: *mut *mut LlevCostCursor,
) -> LlevStatus {
    boundary(|| {
        let transducer = transducer
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
        let costs = costs
            .as_ref()
            .ok_or((LlevStatus::NullPointer, "costs is null".into()))?;
        if out_cursor.is_null() {
            return Err((LlevStatus::NullPointer, "out_cursor is null".into()));
        }
        let costs = weighted_costs(*costs)?;
        if !max_cost.is_finite() || max_cost < 0.0 {
            return Err(invalid(
                "weighted maximum cost must be finite and nonnegative",
            ));
        }
        let cursor = match query_input(domain, query, query_len)? {
            QueryInput::Text(value) => transducer.inner.query_weighted_utf8(value, max_cost, costs),
            QueryInput::Bytes(value) => transducer
                .inner
                .query_weighted_bytes(value, max_cost, costs),
            QueryInput::U64(value) => transducer.inner.query_weighted_u64(value, max_cost, costs),
        };
        write_cursor(binding(cursor)?, out_cursor)
    })
}

fn fill_batch(
    cursor: &mut LlevCostCursor,
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
    cursor.byte_arena.clear();
    cursor.u64_arena.clear();
    cursor.views.reserve(count);
    cursor.offsets.reserve(count);
    for item in &cursor.batch {
        let (domain, term_len, byte_len, offset) = match &item.term {
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
                    term.len().saturating_mul(8),
                    offset,
                )
            }
        };
        cursor.offsets.push(offset);
        cursor.views.push(LlevCostMatch {
            term_data: ptr::null(),
            term_len,
            byte_len,
            cost: item.cost,
            scaled_cost: item.scaled_cost.unwrap_or(0),
            scale_denominator: item.scale_denominator,
            unit_domain: domain,
            has_scaled_cost: u8::from(item.scaled_cost.is_some()),
            reserved: [0; 7],
        });
    }
    for (view, offset) in cursor.views.iter_mut().zip(&cursor.offsets) {
        view.term_data = match view.unit_domain {
            VtUnitDomain::U64 => unsafe { cursor.u64_arena.as_ptr().add(*offset).cast() },
            _ => unsafe { cursor.byte_arena.as_ptr().add(*offset).cast() },
        };
    }
    cursor.generation = cursor.generation.wrapping_add(1).max(1);
    cursor.leased = true;
    Ok(LlevStatus::Ok)
}

/// Borrow one bounded generation-checked batch.
///
/// # Safety
/// Pointers must be valid and cursor exclusively borrowed.
#[no_mangle]
pub unsafe extern "C" fn llev_cost_cursor_next_batch(
    cursor: *mut LlevCostCursor,
    maximum: usize,
    out_batch: *mut LlevCostBatchView,
) -> LlevStatus {
    boundary(|| {
        let cursor = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "cursor is null".into()))?;
        if out_batch.is_null() {
            return Err((LlevStatus::NullPointer, "out_batch is null".into()));
        }
        out_batch.write(LlevCostBatchView::default());
        let status = fill_batch(cursor, maximum)?;
        if status == LlevStatus::Ok {
            out_batch.write(LlevCostBatchView {
                matches: cursor.views.as_ptr(),
                len: cursor.views.len(),
                generation: cursor.generation,
            });
        }
        Ok(status)
    })
}

/// Release the exact live batch generation.
///
/// # Safety
/// Cursor must be live and exclusively borrowed.
#[no_mangle]
pub unsafe extern "C" fn llev_cost_cursor_release_batch(
    cursor: *mut LlevCostCursor,
    generation: u64,
) -> LlevStatus {
    boundary(|| {
        let cursor = cursor
            .as_mut()
            .ok_or((LlevStatus::NullPointer, "cursor is null".into()))?;
        if !cursor.leased || cursor.generation != generation {
            return Err(invalid("cost batch generation is not live"));
        }
        cursor.leased = false;
        Ok(LlevStatus::Ok)
    })
}

/// Reduce bounded borrowed batches on the caller thread.
///
/// # Safety
/// Callback and context must remain valid and callback must not unwind.
#[no_mangle]
pub unsafe extern "C" fn llev_cost_cursor_reduce(
    cursor: *mut LlevCostCursor,
    batch_size: usize,
    reducer: Option<LlevCostBatchReducer>,
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
                    let response = reducer(context, cursor.views.as_ptr(), cursor.views.len());
                    count = count.saturating_add(cursor.views.len());
                    cursor.leased = false;
                    match LlevStatus::try_from(response) {
                        Ok(LlevStatus::Ok) => {}
                        Ok(LlevStatus::End) => break,
                        Ok(status) => return Err((status, "cost reducer aborted".into())),
                        Err(_) => return Err(invalid("cost reducer returned an invalid status")),
                    }
                }
                _ => unreachable!(),
            }
        }
        out_count.write(count);
        Ok(LlevStatus::Ok)
    })
}

/// Free a cost cursor, refusing a live batch lease.
///
/// # Safety
/// Non-null pointer must name a live cursor and cannot be reused on success.
#[no_mangle]
pub unsafe extern "C" fn llev_cost_cursor_free(cursor: *mut LlevCostCursor) -> LlevStatus {
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
