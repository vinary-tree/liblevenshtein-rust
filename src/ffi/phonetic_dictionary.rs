//! Explicitly bounded native phonetic-normalized dictionary ABI.

use super::index::boundary;
#[cfg(feature = "bindings-phonetic")]
use super::index::utf8;
#[cfg(feature = "bindings-phonetic")]
use super::LlevAlgorithm;
use super::{LlevOwnedString, LlevPhoneticRuleSet, LlevStatus};
#[cfg(feature = "bindings-phonetic")]
use crate::dictionary::phonetic_normalized::{
    PhoneticNormalizedCandidate, PhoneticNormalizedDictionary, PhoneticNormalizedTermIdDictionary,
};
#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::zompist_rules_char;
use std::ffi::c_char;
use std::ptr;

/// A borrowed, length-bearing UTF-8 input. Empty strings may use a null data pointer.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevUtf8View {
    /// Borrowed UTF-8 bytes, possibly null for an empty view.
    pub data: *const c_char,
    /// Byte length.
    pub len: usize,
}

/// One result owned by a returned candidate array. Neither string is NUL-terminated.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevPhoneticCandidate {
    /// Copied original spelling.
    pub term: LlevOwnedString,
    /// Edit distance in normalized space.
    pub distance: usize,
    /// Copied normalized form responsible for the match.
    pub normalized_form: LlevOwnedString,
}

/// Opaque normalized index. Mode zero is mutable; mode one is compact term-ID.
pub struct LlevPhoneticDictionary {
    #[cfg(feature = "bindings-phonetic")]
    inner: DictionaryKind,
}

#[cfg(feature = "bindings-phonetic")]
enum DictionaryKind {
    Mutable(PhoneticNormalizedDictionary<()>),
    Compact(PhoneticNormalizedTermIdDictionary),
}

#[cfg(not(feature = "bindings-phonetic"))]
fn unavailable() -> (LlevStatus, String) {
    (
        LlevStatus::Unsupported,
        "phonetic bindings were not compiled".into(),
    )
}

#[cfg(feature = "bindings-phonetic")]
pub(super) fn owned(value: String) -> LlevOwnedString {
    if value.is_empty() {
        return LlevOwnedString::default();
    }
    let boxed = value.into_bytes().into_boxed_slice();
    let len = boxed.len();
    LlevOwnedString {
        data: Box::into_raw(boxed).cast::<u8>().cast(),
        len,
    }
}

#[cfg(feature = "bindings-phonetic")]
unsafe fn borrowed_terms(
    terms: *const LlevUtf8View,
    term_count: usize,
    max_terms: usize,
    max_total_bytes: usize,
) -> Result<Vec<String>, (LlevStatus, String)> {
    if max_terms == 0 || max_total_bytes == 0 {
        return Err((
            LlevStatus::InvalidArgument,
            "term ceilings must be positive".into(),
        ));
    }
    if term_count > max_terms {
        return Err((
            LlevStatus::LimitExceeded,
            "term count exceeds max_terms".into(),
        ));
    }
    if term_count > 0 && terms.is_null() {
        return Err((LlevStatus::NullPointer, "terms is null".into()));
    }
    let views: &[LlevUtf8View] = if term_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(terms, term_count)
    };
    let mut total = 0usize;
    let mut values = Vec::with_capacity(term_count);
    for view in views {
        total = total
            .checked_add(view.len)
            .ok_or((LlevStatus::LimitExceeded, "term byte count overflow".into()))?;
        if total > max_total_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "terms exceed max_total_bytes".into(),
            ));
        }
        values.push(utf8(view.data, view.len)?.to_owned());
    }
    Ok(values)
}

/// Build a mutable or compact native phonetic-normalized dictionary.
///
/// The supplied rule set is cloned. A null rule set selects built-in English
/// Zompist rules. Neither input terms nor rule handles are retained.
///
/// # Safety
///
/// Every input view, optional rule handle, and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_dictionary_new(
    terms: *const LlevUtf8View,
    term_count: usize,
    rules: *const LlevPhoneticRuleSet,
    algorithm: u32,
    mode: u8,
    max_terms: usize,
    max_total_bytes: usize,
    out_dictionary: *mut *mut LlevPhoneticDictionary,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_dictionary.is_null() {
                return Err((LlevStatus::NullPointer, "out_dictionary is null".into()));
            }
            let algorithm = LlevAlgorithm::try_from(algorithm)
                .map_err(|_| (LlevStatus::InvalidArgument, "invalid algorithm".into()))?;
            if mode > 1 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "mode must be zero or one".into(),
                ));
            }
            let terms = borrowed_terms(terms, term_count, max_terms, max_total_bytes)?;
            let rules = rules
                .as_ref()
                .map(|rules| rules.inner.rules().to_vec())
                .unwrap_or_else(zompist_rules_char);
            let inner = if mode == 0 {
                DictionaryKind::Mutable(
                    PhoneticNormalizedDictionary::from_terms_with_rules_and_algorithm(
                        terms,
                        rules,
                        algorithm.into(),
                    ),
                )
            } else {
                let compact =
                    PhoneticNormalizedTermIdDictionary::try_from_terms_with_rules_and_algorithm(
                        terms,
                        rules,
                        algorithm.into(),
                    )
                    .map_err(|error| (LlevStatus::LimitExceeded, error.to_string()))?;
                DictionaryKind::Compact(compact)
            };
            out_dictionary.write(Box::into_raw(Box::new(LlevPhoneticDictionary { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                terms,
                term_count,
                rules,
                algorithm,
                mode,
                max_terms,
                max_total_bytes,
                out_dictionary,
            );
            Err(unavailable())
        }
    })
}

/// Consume a dictionary handle; null is a no-op.
///
/// # Safety
///
/// Non-null handles must be live and unique.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_dictionary_free(dictionary: *mut LlevPhoneticDictionary) {
    if !dictionary.is_null() {
        drop(Box::from_raw(dictionary));
    }
}

/// Query normalized-space fuzzy candidates in native relevance order.
///
/// This returns an owned array of owned UTF-8 strings; call
/// `llev_phonetic_candidates_free` exactly once. On a limit or invalid-input
/// error both outputs remain unchanged. `max_results` bounds returned data,
/// while `max_query_scalars` and the constructor ceilings bound input size.
///
/// # Safety
///
/// The handle, query bytes, and output pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_dictionary_query(
    dictionary: *const LlevPhoneticDictionary,
    query: *const c_char,
    query_len: usize,
    max_distance: usize,
    max_query_scalars: usize,
    max_results: usize,
    out_candidates: *mut *mut LlevPhoneticCandidate,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let dictionary = dictionary
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "dictionary is null".into()))?;
            if out_candidates.is_null() || out_count.is_null() {
                return Err((LlevStatus::NullPointer, "candidate output is null".into()));
            }
            if max_query_scalars == 0 || max_results == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "query ceilings must be positive".into(),
                ));
            }
            let query = utf8(query, query_len)?;
            if query.chars().count() > max_query_scalars {
                return Err((
                    LlevStatus::LimitExceeded,
                    "query exceeds max_query_scalars".into(),
                ));
            }
            let candidates: Vec<PhoneticNormalizedCandidate> = match &dictionary.inner {
                DictionaryKind::Mutable(inner) => inner.query(query, max_distance),
                DictionaryKind::Compact(inner) => inner.query(query, max_distance),
            };
            if candidates.len() > max_results {
                return Err((
                    LlevStatus::LimitExceeded,
                    "candidate count exceeds max_results".into(),
                ));
            }
            let values: Box<[LlevPhoneticCandidate]> = candidates
                .into_iter()
                .map(|candidate| LlevPhoneticCandidate {
                    term: owned(candidate.term),
                    distance: candidate.distance,
                    normalized_form: owned(candidate.normalized_form),
                })
                .collect();
            let count = values.len();
            let pointer = if count == 0 {
                ptr::null_mut()
            } else {
                Box::into_raw(values) as *mut LlevPhoneticCandidate
            };
            out_candidates.write(pointer);
            out_count.write(count);
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                dictionary,
                query,
                query_len,
                max_distance,
                max_query_scalars,
                max_results,
                out_candidates,
                out_count,
            );
            Err(unavailable())
        }
    })
}

/// Free a candidate array and all nested strings; null is valid only with zero count.
///
/// # Safety
///
/// The pointer/count pair must be returned by a successful query and not freed.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_candidates_free(
    candidates: *mut LlevPhoneticCandidate,
    count: usize,
) {
    if candidates.is_null() {
        return;
    }
    let mut values = Box::from_raw(ptr::slice_from_raw_parts_mut(candidates, count));
    for candidate in values.iter_mut() {
        super::llev_owned_string_free(&mut candidate.term);
        super::llev_owned_string_free(&mut candidate.normalized_form);
    }
}

/// Insert or remove a term from the mutable mode. Compact mode is immutable.
///
/// # Safety
///
/// The dictionary, term bytes, and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_dictionary_update(
    dictionary: *mut LlevPhoneticDictionary,
    term: *const c_char,
    term_len: usize,
    remove: u8,
    max_term_scalars: usize,
    out_changed: *mut u8,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let dictionary = dictionary
                .as_mut()
                .ok_or((LlevStatus::NullPointer, "dictionary is null".into()))?;
            if out_changed.is_null() {
                return Err((LlevStatus::NullPointer, "out_changed is null".into()));
            }
            if remove > 1 || max_term_scalars == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "invalid update selector or ceiling".into(),
                ));
            }
            let term = utf8(term, term_len)?;
            if term.chars().count() > max_term_scalars {
                return Err((
                    LlevStatus::LimitExceeded,
                    "term exceeds max_term_scalars".into(),
                ));
            }
            let DictionaryKind::Mutable(inner) = &dictionary.inner else {
                return Err((
                    LlevStatus::Unsupported,
                    "compact term-ID index is immutable".into(),
                ));
            };
            out_changed.write(u8::from(if remove == 1 {
                inner.remove(term)
            } else {
                inner.insert(term)
            }));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                dictionary,
                term,
                term_len,
                remove,
                max_term_scalars,
                out_changed,
            );
            Err(unavailable())
        }
    })
}
