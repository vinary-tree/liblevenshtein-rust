//! Bounded token-query phonetic grep ABI with copied per-token details.

use super::index::boundary;
#[cfg(feature = "bindings-phonetic")]
use super::index::utf8;
use super::{LlevOwnedString, LlevPhoneticRuleSet, LlevStatus};
#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::TokenGrep;
use std::ffi::c_char;
use std::ptr;

/// Copied detail for one token in a token-query match.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevPhoneticTokenDetail {
    /// Query token index, zero-based.
    pub token_index: usize,
    /// Start UTF-8 byte offset in the full document.
    pub byte_start: usize,
    /// End-exclusive UTF-8 byte offset in the full document.
    pub byte_end: usize,
    /// Copied original token text.
    pub original_text: LlevOwnedString,
    /// Copied phonetic-normalized token text.
    pub normalized_text: LlevOwnedString,
    /// Native edit distance for this token.
    pub distance: u8,
    /// Reserved zero bytes.
    pub reserved: [u8; 7],
}

/// Copied match with a separately owned nested detail array.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevPhoneticTokenMatch {
    /// Start UTF-8 byte offset in the full document.
    pub byte_start: usize,
    /// End-exclusive UTF-8 byte offset in the full document.
    pub byte_end: usize,
    /// Sum of native per-token distances.
    pub total_distance: u8,
    /// Reserved zero bytes.
    pub reserved: [u8; 7],
    /// Copied matched source span.
    pub matched_text: LlevOwnedString,
    /// Owned detail array; free only through token_matches_free.
    pub details: *mut LlevPhoneticTokenDetail,
    /// Detail count.
    pub detail_count: usize,
}

/// Reusable native token-query matcher.
pub struct LlevPhoneticTokenGrep {
    #[cfg(feature = "bindings-phonetic")]
    inner: TokenGrep,
}

#[cfg(not(feature = "bindings-phonetic"))]
fn unavailable() -> (LlevStatus, String) {
    (
        LlevStatus::Unsupported,
        "phonetic bindings were not compiled".into(),
    )
}

/// Compile a token-query pattern with optional cloned phonetic rules.
///
/// Query syntax and default-distance semantics are identical to native
/// `TokenGrep`. `max_query_bytes` must be positive. The query is not retained.
///
/// # Safety
///
/// Query bytes, optional rules, and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_token_new(
    query: *const c_char,
    query_len: usize,
    rules: *const LlevPhoneticRuleSet,
    default_distance: u8,
    max_query_bytes: usize,
    out_grep: *mut *mut LlevPhoneticTokenGrep,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_grep.is_null() {
                return Err((LlevStatus::NullPointer, "out_grep is null".into()));
            }
            if max_query_bytes == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "max_query_bytes must be positive".into(),
                ));
            }
            if query_len > max_query_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "query exceeds max_query_bytes".into(),
                ));
            }
            let query = utf8(query, query_len)?;
            let inner = if let Some(rules) = rules.as_ref() {
                TokenGrep::with_rules(query, rules.inner.rules().to_vec(), default_distance)
            } else {
                TokenGrep::new(query, default_distance)
            }
            .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            out_grep.write(Box::into_raw(Box::new(LlevPhoneticTokenGrep { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                query,
                query_len,
                rules,
                default_distance,
                max_query_bytes,
                out_grep,
            );
            Err(unavailable())
        }
    })
}

/// Consume a token matcher; null is a no-op.
///
/// # Safety
///
/// Non-null handles must be live and unique.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_token_free(grep: *mut LlevPhoneticTokenGrep) {
    if !grep.is_null() {
        drop(Box::from_raw(grep));
    }
}

/// Scan a document and return fully copied matches and details.
///
/// Positive input, match, and total-detail ceilings are required. Limit
/// failures leave outputs untouched. Free all nested storage using
/// `llev_phonetic_token_matches_free`.
///
/// # Safety
///
/// Handle, document bytes, and output pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_token_scan(
    grep: *const LlevPhoneticTokenGrep,
    document: *const c_char,
    document_len: usize,
    max_input_bytes: usize,
    max_matches: usize,
    max_details: usize,
    out_matches: *mut *mut LlevPhoneticTokenMatch,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let grep = grep
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "grep is null".into()))?;
            if out_matches.is_null() || out_count.is_null() {
                return Err((LlevStatus::NullPointer, "result output is null".into()));
            }
            if max_input_bytes == 0 || max_matches == 0 || max_details == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "token scan ceilings must be positive".into(),
                ));
            }
            if document_len > max_input_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "document exceeds max_input_bytes".into(),
                ));
            }
            let document = utf8(document, document_len)?;
            let matches = grep.inner.scan(document);
            let detail_count = matches
                .iter()
                .try_fold(0usize, |total, item| {
                    total.checked_add(item.token_matches.len())
                })
                .ok_or((LlevStatus::LimitExceeded, "detail count overflow".into()))?;
            if matches.len() > max_matches || detail_count > max_details {
                return Err((
                    LlevStatus::LimitExceeded,
                    "token result ceilings exceeded".into(),
                ));
            }
            let values: Box<[LlevPhoneticTokenMatch]> = matches
                .into_iter()
                .map(|item| {
                    let details: Box<[LlevPhoneticTokenDetail]> = item
                        .token_matches
                        .into_iter()
                        .map(|detail| LlevPhoneticTokenDetail {
                            token_index: detail.token_index,
                            byte_start: detail.byte_range.0,
                            byte_end: detail.byte_range.1,
                            original_text: super::phonetic_dictionary::owned(detail.original_text),
                            normalized_text: super::phonetic_dictionary::owned(
                                detail.normalized_text,
                            ),
                            distance: detail.distance,
                            reserved: [0; 7],
                        })
                        .collect();
                    let detail_count = details.len();
                    LlevPhoneticTokenMatch {
                        byte_start: item.byte_range.0,
                        byte_end: item.byte_range.1,
                        total_distance: item.total_distance,
                        reserved: [0; 7],
                        matched_text: super::phonetic_dictionary::owned(item.matched_text),
                        details: if detail_count == 0 {
                            ptr::null_mut()
                        } else {
                            Box::into_raw(details) as *mut LlevPhoneticTokenDetail
                        },
                        detail_count,
                    }
                })
                .collect();
            let count = values.len();
            out_matches.write(if count == 0 {
                ptr::null_mut()
            } else {
                Box::into_raw(values) as *mut LlevPhoneticTokenMatch
            });
            out_count.write(count);
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                grep,
                document,
                document_len,
                max_input_bytes,
                max_matches,
                max_details,
                out_matches,
                out_count,
            );
            Err(unavailable())
        }
    })
}

/// Consume an exact token-match array/count pair and all nested storage.
///
/// # Safety
///
/// The pointer/count pair must be returned by token_scan and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_token_matches_free(
    matches: *mut LlevPhoneticTokenMatch,
    count: usize,
) {
    if matches.is_null() {
        return;
    }
    let mut values = Box::from_raw(ptr::slice_from_raw_parts_mut(matches, count));
    for item in values.iter_mut() {
        super::llev_owned_string_free(&mut item.matched_text);
        if !item.details.is_null() {
            let mut details = Box::from_raw(ptr::slice_from_raw_parts_mut(
                item.details,
                item.detail_count,
            ));
            for detail in details.iter_mut() {
                super::llev_owned_string_free(&mut detail.original_text);
                super::llev_owned_string_free(&mut detail.normalized_text);
            }
        }
    }
}
