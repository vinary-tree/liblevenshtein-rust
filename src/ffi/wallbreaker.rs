//! Additive, finite-owned Unicode WallBreaker C surface.
//!
//! This does not consume `vt.dictionary.v1`: that interface has no exact
//! substring lookup. The matcher owns an immutable SCDAWG built from caller
//! terms, and cursors own complete bounded results before publishing a lease.

use super::{index::boundary, index::utf8, LlevAlgorithm, LlevStatus};
use crate::transducer::Algorithm;
use crate::wallbreaker::{PatternSplitter, WallBreaker, WallBreakerResult};
use libdictenstein::scdawg::ScdawgChar;
use libdictenstein::scdawg::ScdawgCharNodeHandle;
use libdictenstein::substring::SubstringMatch;
use std::ffi::c_char;
use std::ptr;

const HARD_TERMS: usize = 4096;
const HARD_TOTAL_BYTES: usize = 1 << 20;
const HARD_SCALARS: usize = 256;
const HARD_CANDIDATE_BYTES: usize = 16 << 20;
const HARD_RESULTS: usize = 4096;
const HARD_RESULT_BYTES: usize = 1 << 20;
const HARD_DISTANCE: usize = 8;

type Failure = (LlevStatus, String);

fn invalid(message: &str) -> Failure {
    (LlevStatus::InvalidArgument, message.into())
}

fn null(message: &str) -> Failure {
    (LlevStatus::NullPointer, message.into())
}

fn limit(message: &str) -> Failure {
    (LlevStatus::LimitExceeded, message.into())
}

fn algorithm(raw: u32) -> Result<Algorithm, Failure> {
    LlevAlgorithm::try_from(raw)
        .map(Into::into)
        .map_err(|_| invalid("unknown WallBreaker algorithm"))
}

/// Borrowed input term; bytes are copied before this call returns.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevWallBreakerTerm {
    /// Borrowed UTF-8 bytes, null only for an empty term.
    pub data: *const c_char,
    /// Number of bytes at `data`.
    pub byte_len: usize,
}

/// Per-matcher logical limits. All fields must be positive and no greater
/// than the implementation maxima documented in the C header.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevWallBreakerLimits {
    /// Maximum number of input term descriptors.
    pub max_terms: usize,
    /// Maximum sum of input term UTF-8 bytes.
    pub max_total_term_bytes: usize,
    /// Maximum Unicode scalar count of any term.
    pub max_term_scalars: usize,
    /// Maximum Unicode scalar count of a query.
    pub max_query_scalars: usize,
    /// Conservative upper bound on selective substring-candidate clone bytes.
    /// Complete-term streaming queries do not materialize those candidates.
    pub max_candidate_clone_bytes: usize,
    /// Maximum complete verified query results.
    pub max_results: usize,
    /// Maximum sum of complete result UTF-8 byte lengths.
    pub max_result_bytes: usize,
}

impl LlevWallBreakerLimits {
    fn validate(self) -> Result<(), Failure> {
        for (value, hard) in [
            (self.max_terms, HARD_TERMS),
            (self.max_total_term_bytes, HARD_TOTAL_BYTES),
            (self.max_term_scalars, HARD_SCALARS),
            (self.max_query_scalars, HARD_SCALARS),
            (self.max_candidate_clone_bytes, HARD_CANDIDATE_BYTES),
            (self.max_results, HARD_RESULTS),
            (self.max_result_bytes, HARD_RESULT_BYTES),
        ] {
            if value == 0 || value > hard {
                return Err(invalid(
                    "WallBreaker limits must be positive and within hard maxima",
                ));
            }
        }
        Ok(())
    }
}

/// Immutable, owned Unicode term set and algorithm configuration.
pub struct LlevWallBreaker {
    dictionary: ScdawgChar<()>,
    limits: LlevWallBreakerLimits,
    algorithm: Algorithm,
    max_distance: usize,
    candidate_clone_bound: usize,
}

/// Borrowed result descriptor, valid only during its batch lease.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevWallBreakerResultView {
    /// Cursor-owned UTF-8 term bytes.
    pub term_data: *const u8,
    /// Byte count at `term_data`.
    pub byte_len: usize,
    /// Selected-algorithm exact distance.
    pub distance: usize,
}

/// One cursor-owned batch lease. Generation must be released before advancing.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevWallBreakerBatchView {
    /// Contiguous cursor-owned descriptors.
    pub results: *const LlevWallBreakerResultView,
    /// Number of descriptors.
    pub len: usize,
    /// Lease generation to release.
    pub generation: u64,
}

impl Default for LlevWallBreakerBatchView {
    fn default() -> Self {
        Self {
            results: ptr::null(),
            len: 0,
            generation: 0,
        }
    }
}

/// Exclusive cursor over fully verified results of one immutable snapshot.
pub struct LlevWallBreakerCursor {
    results: Vec<WallBreakerResult>,
    offset: usize,
    views: Vec<LlevWallBreakerResultView>,
    generation: u64,
    leased: bool,
    cancelled: bool,
}

/// A scalar-indexed pattern piece, with byte coordinates in the original UTF-8.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct LlevPatternPiece {
    /// Zero-based UTF-8 byte offset.
    pub byte_offset: usize,
    /// UTF-8 byte length.
    pub byte_len: usize,
    /// Zero-based Unicode scalar start.
    pub start_scalar: usize,
    /// Exclusive Unicode scalar end.
    pub end_scalar: usize,
    /// Zero-based native piece index.
    pub piece_index: usize,
}

/// Construct a bounded matcher from borrowed Unicode terms.
///
/// # Safety
/// `out` must be writable. `terms` must address `term_count` descriptors,
/// whose nonempty byte spans remain readable for this call.
#[no_mangle]
pub unsafe extern "C" fn llev_wallbreaker_new_utf8(
    terms: *const LlevWallBreakerTerm,
    term_count: usize,
    limits: LlevWallBreakerLimits,
    algorithm_raw: u32,
    max_distance: usize,
    out: *mut *mut LlevWallBreaker,
) -> LlevStatus {
    boundary(|| {
        if out.is_null() {
            return Err(null("null matcher output"));
        }
        unsafe {
            *out = ptr::null_mut();
        }
        limits.validate()?;
        let algorithm = algorithm(algorithm_raw)?;
        if max_distance > HARD_DISTANCE {
            return Err(limit("WallBreaker distance exceeds hard maximum 8"));
        }
        if term_count > limits.max_terms {
            return Err(limit("WallBreaker term count exceeds limit"));
        }
        if term_count > 0 && terms.is_null() {
            return Err(null("null term descriptor array"));
        }
        let input = if term_count == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(terms, term_count) }
        };
        let mut owned = Vec::with_capacity(term_count);
        let mut total_bytes = 0usize;
        let mut candidate_clone_bound = 0usize;
        for item in input {
            total_bytes = total_bytes
                .checked_add(item.byte_len)
                .ok_or_else(|| limit("term byte count overflow"))?;
            if total_bytes > limits.max_total_term_bytes {
                return Err(limit("WallBreaker term bytes exceed limit"));
            }
            let term = unsafe { utf8(item.data, item.byte_len)? };
            let scalars = term.chars().count();
            if scalars > limits.max_term_scalars {
                return Err(limit("WallBreaker term scalar count exceeds limit"));
            }
            // Each piece can occur at no more than scalars+1 positions in a
            // term. The native API clones one full term per occurrence.
            let descriptor_bytes = std::mem::size_of::<SubstringMatch<ScdawgCharNodeHandle<()>>>()
                .checked_add(std::mem::size_of::<(String, usize)>())
                .ok_or_else(|| limit("candidate descriptor bound overflow"))?;
            let per_occurrence = item
                .byte_len
                .checked_add(descriptor_bytes)
                .ok_or_else(|| limit("candidate bound overflow"))?;
            let bound = (scalars + 1)
                .checked_mul(per_occurrence)
                .ok_or_else(|| limit("candidate bound overflow"))?;
            candidate_clone_bound = candidate_clone_bound
                .checked_add(bound)
                .ok_or_else(|| limit("candidate bound overflow"))?;
            owned.push(term.to_owned());
        }
        let matcher = LlevWallBreaker {
            dictionary: ScdawgChar::from_terms(owned),
            limits,
            algorithm,
            max_distance,
            candidate_clone_bound,
        };
        unsafe {
            *out = Box::into_raw(Box::new(matcher));
        }
        Ok(LlevStatus::Ok)
    })
}

/// Free a matcher; completed cursors do not borrow it.
///
/// # Safety
/// `matcher` must be null or a live pointer returned by this constructor,
/// transferred exactly once; no concurrent call may use it.
#[no_mangle]
pub unsafe extern "C" fn llev_wallbreaker_free(matcher: *mut LlevWallBreaker) {
    if !matcher.is_null() {
        unsafe {
            drop(Box::from_raw(matcher));
        }
    }
}

/// Eagerly compute all results, or return LIMIT_EXCEEDED without a cursor.
///
/// # Safety
/// `matcher` must remain live for this call, `query` must address `query_len`
/// readable bytes when nonempty, and `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn llev_wallbreaker_query_utf8(
    matcher: *const LlevWallBreaker,
    query: *const c_char,
    query_len: usize,
    out: *mut *mut LlevWallBreakerCursor,
) -> LlevStatus {
    boundary(|| {
        if out.is_null() {
            return Err(null("null cursor output"));
        }
        unsafe {
            *out = ptr::null_mut();
        }
        if matcher.is_null() {
            return Err(null("null matcher"));
        }
        let matcher = unsafe { &*matcher };
        let query = unsafe { utf8(query, query_len)? };
        if query.chars().count() > matcher.limits.max_query_scalars {
            return Err(limit("WallBreaker query scalar count exceeds limit"));
        }
        let mut results = Vec::new();
        let mut result_bytes = 0usize;
        let search = WallBreaker::with_algorithm(
            &matcher.dictionary,
            matcher.max_distance,
            matcher.algorithm,
        );
        let candidates = search.query(query);
        if !candidates.uses_complete_term_scan()
            && matcher.candidate_clone_bound > matcher.limits.max_candidate_clone_bytes
        {
            return Err(limit("WallBreaker candidate bound exceeds limit"));
        }
        for result in candidates {
            if results.len() >= matcher.limits.max_results {
                return Err(limit("WallBreaker result count exceeds limit"));
            }
            result_bytes = result_bytes
                .checked_add(result.term.len())
                .ok_or_else(|| limit("result bytes overflow"))?;
            if result_bytes > matcher.limits.max_result_bytes {
                return Err(limit("WallBreaker result bytes exceed limit"));
            }
            results.push(result);
        }
        let cursor = LlevWallBreakerCursor {
            results,
            offset: 0,
            views: Vec::new(),
            generation: 0,
            leased: false,
            cancelled: false,
        };
        unsafe {
            *out = Box::into_raw(Box::new(cursor));
        }
        Ok(LlevStatus::Ok)
    })
}

/// Publish at most one bounded borrowed batch lease.
///
/// # Safety
/// `cursor` must be a live exclusive cursor, and `out` must be writable.
/// Borrowed descriptors must not be accessed after release or cursor free.
#[no_mangle]
pub unsafe extern "C" fn llev_wallbreaker_cursor_next_batch(
    cursor: *mut LlevWallBreakerCursor,
    max_entries: usize,
    max_bytes: usize,
    out: *mut LlevWallBreakerBatchView,
) -> LlevStatus {
    boundary(|| {
        if out.is_null() {
            return Err(null("null batch output"));
        }
        unsafe {
            *out = LlevWallBreakerBatchView::default();
        }
        if cursor.is_null() {
            return Err(null("null cursor"));
        }
        let cursor = unsafe { &mut *cursor };
        if cursor.cancelled {
            return Err((LlevStatus::Closed, "WallBreaker cursor is cancelled".into()));
        }
        if cursor.leased {
            return Err((LlevStatus::BatchInUse, "WallBreaker batch is leased".into()));
        }
        if max_entries == 0 || max_bytes == 0 {
            return Err(invalid("batch limits must be positive"));
        }
        if cursor.offset == cursor.results.len() {
            return Ok(LlevStatus::End);
        }
        let mut end = cursor.offset;
        let mut bytes = 0usize;
        while end < cursor.results.len() && end - cursor.offset < max_entries {
            let len = cursor.results[end].term.len();
            if len > max_bytes - bytes {
                break;
            }
            bytes += len;
            end += 1;
        }
        if end == cursor.offset {
            return Err(limit("first WallBreaker result exceeds batch byte limit"));
        }
        cursor.views.clear();
        cursor
            .views
            .extend(cursor.results[cursor.offset..end].iter().map(|result| {
                LlevWallBreakerResultView {
                    term_data: result.term.as_ptr(),
                    byte_len: result.term.len(),
                    distance: result.distance,
                }
            }));
        cursor.offset = end;
        cursor.generation = cursor.generation.wrapping_add(1).max(1);
        cursor.leased = true;
        unsafe {
            *out = LlevWallBreakerBatchView {
                results: cursor.views.as_ptr(),
                len: cursor.views.len(),
                generation: cursor.generation,
            };
        }
        Ok(LlevStatus::Ok)
    })
}

/// Release exactly the active lease generation.
///
/// # Safety
/// `cursor` must be a live exclusive cursor; callers must stop using the
/// released generation's borrowed descriptors before this call.
#[no_mangle]
pub unsafe extern "C" fn llev_wallbreaker_cursor_release_batch(
    cursor: *mut LlevWallBreakerCursor,
    generation: u64,
) -> LlevStatus {
    boundary(|| {
        if cursor.is_null() {
            return Err(null("null cursor"));
        }
        let cursor = unsafe { &mut *cursor };
        if !cursor.leased || cursor.generation != generation {
            return Err(invalid("no matching WallBreaker lease"));
        }
        cursor.leased = false;
        cursor.views.clear();
        Ok(LlevStatus::Ok)
    })
}

/// Stop future cursor advances while preserving an active lease.
///
/// # Safety
/// `cursor` must be a live exclusive cursor, with no concurrent advance.
#[no_mangle]
pub unsafe extern "C" fn llev_wallbreaker_cursor_cancel(
    cursor: *mut LlevWallBreakerCursor,
) -> LlevStatus {
    boundary(|| {
        if cursor.is_null() {
            return Err(null("null cursor"));
        }
        unsafe {
            (*cursor).cancelled = true;
        }
        Ok(LlevStatus::Ok)
    })
}

/// Free an exclusive cursor, invalidating any active lease.
///
/// # Safety
/// `cursor` must be null or a live cursor transferred exactly once. No
/// borrowed batch or concurrent operation may continue using it afterward.
#[no_mangle]
pub unsafe extern "C" fn llev_wallbreaker_cursor_free(cursor: *mut LlevWallBreakerCursor) {
    if !cursor.is_null() {
        unsafe {
            drop(Box::from_raw(cursor));
        }
    }
}

/// Two-phase split projection. Query bytes are borrowed only during the call.
///
/// # Safety
/// `query` must address `query_len` readable bytes when nonempty;
/// `out_required` must be writable, and `pieces` must address `capacity`
/// writable descriptors whenever the capacity is sufficient and nonzero.
#[no_mangle]
pub unsafe extern "C" fn llev_wallbreaker_split_utf8(
    query: *const c_char,
    query_len: usize,
    algorithm_raw: u32,
    max_distance: usize,
    pieces: *mut LlevPatternPiece,
    capacity: usize,
    out_required: *mut usize,
) -> LlevStatus {
    boundary(|| {
        if out_required.is_null() {
            return Err(null("null required count output"));
        }
        unsafe {
            *out_required = 0;
        }
        let algorithm = algorithm(algorithm_raw)?;
        if max_distance > HARD_DISTANCE {
            return Err(limit("WallBreaker distance exceeds hard maximum 8"));
        }
        let query = unsafe { utf8(query, query_len)? };
        if query.chars().count() > HARD_SCALARS {
            return Err(limit("WallBreaker split query exceeds hard scalar limit"));
        }
        let split = PatternSplitter::new(max_distance, algorithm).split(query);
        unsafe {
            *out_required = split.len();
        }
        if capacity < split.len() {
            return Err(limit("WallBreaker split output capacity is too small"));
        }
        if !split.is_empty() && pieces.is_null() {
            return Err(null("null piece output"));
        }
        let byte_offsets: Vec<usize> = query
            .char_indices()
            .map(|(offset, _)| offset)
            .chain(std::iter::once(query.len()))
            .collect();
        for (index, piece) in split.iter().enumerate() {
            unsafe {
                *pieces.add(index) = LlevPatternPiece {
                    byte_offset: byte_offsets[piece.start_offset],
                    byte_len: byte_offsets[piece.end_offset] - byte_offsets[piece.start_offset],
                    start_scalar: piece.start_offset,
                    end_scalar: piece.end_offset,
                    piece_index: piece.piece_index,
                };
            }
        }
        Ok(LlevStatus::Ok)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> LlevWallBreakerLimits {
        LlevWallBreakerLimits {
            max_terms: 16,
            max_total_term_bytes: 1024,
            max_term_scalars: 32,
            max_query_scalars: 32,
            max_candidate_clone_bytes: 1 << 16,
            max_results: 16,
            max_result_bytes: 1024,
        }
    }

    unsafe fn matcher(terms: &[&str], bounds: LlevWallBreakerLimits) -> *mut LlevWallBreaker {
        let descriptors: Vec<_> = terms
            .iter()
            .map(|term| LlevWallBreakerTerm {
                data: term.as_ptr().cast(),
                byte_len: term.len(),
            })
            .collect();
        let mut output = ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_wallbreaker_new_utf8(
                    descriptors.as_ptr(),
                    descriptors.len(),
                    bounds,
                    0,
                    2,
                    &mut output,
                )
            },
            LlevStatus::Ok
        );
        output
    }

    #[test]
    fn unicode_eager_results_lease_cancel_and_source_independence() {
        unsafe {
            let matcher = matcher(&["", "café", "cafe", "café", "αβγ"], limits());
            let mut cursor = ptr::null_mut();
            assert_eq!(
                llev_wallbreaker_query_utf8(matcher, "café".as_ptr().cast(), 5, &mut cursor),
                LlevStatus::Ok
            );
            llev_wallbreaker_free(matcher);
            let mut batch = LlevWallBreakerBatchView::default();
            assert_eq!(
                llev_wallbreaker_cursor_next_batch(cursor, 1, 1, &mut batch),
                LlevStatus::LimitExceeded
            );
            assert_eq!(batch.len, 0);
            assert_eq!(
                llev_wallbreaker_cursor_next_batch(cursor, 2, 64, &mut batch),
                LlevStatus::Ok
            );
            let copied: Vec<_> = std::slice::from_raw_parts(batch.results, batch.len)
                .iter()
                .map(|result| {
                    (
                        std::str::from_utf8(std::slice::from_raw_parts(
                            result.term_data,
                            result.byte_len,
                        ))
                        .unwrap()
                        .to_string(),
                        result.distance,
                    )
                })
                .collect();
            assert!(copied.contains(&("café".into(), 0)));
            assert!(copied.contains(&("cafe".into(), 1)));
            assert_eq!(
                llev_wallbreaker_cursor_next_batch(
                    cursor,
                    2,
                    64,
                    &mut LlevWallBreakerBatchView::default()
                ),
                LlevStatus::BatchInUse
            );
            assert_eq!(llev_wallbreaker_cursor_cancel(cursor), LlevStatus::Ok);
            assert_eq!(
                llev_wallbreaker_cursor_release_batch(cursor, batch.generation + 1),
                LlevStatus::InvalidArgument
            );
            assert_eq!(
                llev_wallbreaker_cursor_release_batch(cursor, batch.generation),
                LlevStatus::Ok
            );
            assert_eq!(
                llev_wallbreaker_cursor_next_batch(cursor, 2, 64, &mut batch),
                LlevStatus::Closed
            );
            llev_wallbreaker_cursor_free(cursor);
        }
    }

    #[test]
    fn budget_and_utf8_fail_without_publishing_partial_output() {
        unsafe {
            let mut output = ptr::null_mut();
            let invalid = [LlevWallBreakerTerm {
                data: [0xffu8].as_ptr().cast(),
                byte_len: 1,
            }];
            assert_eq!(
                llev_wallbreaker_new_utf8(invalid.as_ptr(), 1, limits(), 0, 1, &mut output),
                LlevStatus::InvalidUtf8
            );
            assert!(output.is_null());
            let terms = [
                LlevWallBreakerTerm {
                    data: "abcdefgh".as_ptr().cast(),
                    byte_len: 8,
                },
                LlevWallBreakerTerm {
                    data: "a".as_ptr().cast(),
                    byte_len: 1,
                },
            ];
            let mut small = limits();
            small.max_candidate_clone_bytes = 100;
            assert_eq!(
                llev_wallbreaker_new_utf8(terms.as_ptr(), 2, small, 0, 1, &mut output),
                LlevStatus::Ok
            );
            let mut bounded_cursor = ptr::null_mut();
            assert_eq!(
                llev_wallbreaker_query_utf8(output, "a".as_ptr().cast(), 1, &mut bounded_cursor),
                LlevStatus::Ok,
                "short queries stream complete terms without a candidate-clone set"
            );
            let mut bounded_batch = LlevWallBreakerBatchView::default();
            assert_eq!(
                llev_wallbreaker_cursor_next_batch(bounded_cursor, 2, 16, &mut bounded_batch),
                LlevStatus::Ok
            );
            assert_eq!(bounded_batch.len, 1);
            let only = &*bounded_batch.results;
            assert_eq!(only.distance, 0);
            assert_eq!(
                std::slice::from_raw_parts(only.term_data, only.byte_len),
                b"a"
            );
            assert_eq!(
                llev_wallbreaker_cursor_release_batch(bounded_cursor, bounded_batch.generation),
                LlevStatus::Ok
            );
            llev_wallbreaker_cursor_free(bounded_cursor);
            bounded_cursor = ptr::null_mut();
            assert_eq!(
                llev_wallbreaker_query_utf8(
                    output,
                    "abcdefgh".as_ptr().cast(),
                    8,
                    &mut bounded_cursor,
                ),
                LlevStatus::LimitExceeded,
                "selective queries still enforce the candidate-clone budget"
            );
            assert!(bounded_cursor.is_null());
            llev_wallbreaker_free(output);
            let first_matcher = matcher(&["a", "b"], limits());
            let mut cursor = ptr::null_mut();
            assert_eq!(
                llev_wallbreaker_query_utf8(first_matcher, ptr::null(), 0, &mut cursor),
                LlevStatus::Ok
            );
            llev_wallbreaker_cursor_free(cursor);
            llev_wallbreaker_free(first_matcher);

            let mut one_result = limits();
            one_result.max_results = 1;
            let matcher = matcher(&["a", "b"], one_result);
            cursor = ptr::null_mut();
            assert_eq!(
                llev_wallbreaker_query_utf8(matcher, ptr::null(), 0, &mut cursor),
                LlevStatus::LimitExceeded
            );
            assert!(cursor.is_null());
            llev_wallbreaker_free(matcher);
        }
    }

    #[test]
    fn split_is_scalar_indexed_two_phase_and_nonpartial() {
        unsafe {
            let query = "éa";
            let mut count = 0;
            assert_eq!(
                llev_wallbreaker_split_utf8(
                    query.as_ptr().cast(),
                    query.len(),
                    0,
                    1,
                    ptr::null_mut(),
                    0,
                    &mut count
                ),
                LlevStatus::LimitExceeded
            );
            assert_eq!(count, 2);
            let mut output = [LlevPatternPiece::default(); 2];
            assert_eq!(
                llev_wallbreaker_split_utf8(
                    query.as_ptr().cast(),
                    query.len(),
                    0,
                    1,
                    output.as_mut_ptr(),
                    1,
                    &mut count
                ),
                LlevStatus::LimitExceeded
            );
            assert_eq!(output[0].byte_len, 0);
            assert_eq!(
                llev_wallbreaker_split_utf8(
                    query.as_ptr().cast(),
                    query.len(),
                    0,
                    1,
                    output.as_mut_ptr(),
                    2,
                    &mut count
                ),
                LlevStatus::Ok
            );
            assert_eq!((output[0].byte_offset, output[0].byte_len), (0, 2));
            assert_eq!((output[1].byte_offset, output[1].byte_len), (2, 1));
        }
    }

    #[test]
    fn empty_corpus_and_empty_query_end_without_a_lease() {
        unsafe {
            let mut matcher = ptr::null_mut();
            assert_eq!(
                llev_wallbreaker_new_utf8(ptr::null(), 0, limits(), 0, 0, &mut matcher),
                LlevStatus::Ok
            );
            let mut cursor = ptr::null_mut();
            assert_eq!(
                llev_wallbreaker_query_utf8(matcher, ptr::null(), 0, &mut cursor),
                LlevStatus::Ok
            );
            llev_wallbreaker_free(matcher);
            let mut batch = LlevWallBreakerBatchView::default();
            assert_eq!(
                llev_wallbreaker_cursor_next_batch(cursor, 2, 8, &mut batch),
                LlevStatus::End
            );
            assert!(batch.results.is_null());
            assert_eq!(batch.len, 0);
            llev_wallbreaker_cursor_free(cursor);
        }
    }
}
