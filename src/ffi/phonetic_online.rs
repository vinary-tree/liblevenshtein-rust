//! Owned, bounded character-level phonetic grep and incremental scanner ABI.

use super::index::boundary;
#[cfg(feature = "bindings-phonetic")]
use super::index::utf8;
use super::{LlevOwnedString, LlevPhoneticRuleSet, LlevStatus};
#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::{PhoneticGrepOnline, ScanMatch, StreamingScanner};
use std::ffi::c_char;
use std::ptr;

/// Copied character-level match with byte and Unicode-scalar spans.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlevPhoneticOnlineMatch {
    /// Inclusive starting UTF-8 byte offset.
    pub byte_start: usize,
    /// Exclusive ending UTF-8 byte offset.
    pub byte_end: usize,
    /// Inclusive starting Unicode-scalar offset.
    pub char_start: usize,
    /// Exclusive ending Unicode-scalar offset.
    pub char_end: usize,
    /// Copied matched source text.
    pub original_text: LlevOwnedString,
    /// Copied normalized match text.
    pub normalized_text: LlevOwnedString,
    /// Native edit distance.
    pub distance: u8,
    /// Reserved zero bytes.
    pub reserved: [u8; 7],
}

/// Reusable immutable native online matcher.
pub struct LlevPhoneticOnlineGrep {
    #[cfg(feature = "bindings-phonetic")]
    inner: PhoneticGrepOnline,
}

/// Owned character-level scanner; feed appends chunks and finish consumes state.
pub struct LlevPhoneticOnlineStream {
    #[cfg(feature = "bindings-phonetic")]
    inner: Option<StreamingScanner>,
    #[cfg(feature = "bindings-phonetic")]
    bytes: usize,
    #[cfg(feature = "bindings-phonetic")]
    max_bytes: usize,
}

#[cfg(not(feature = "bindings-phonetic"))]
fn unavailable() -> (LlevStatus, String) {
    (
        LlevStatus::Unsupported,
        "phonetic bindings were not compiled".into(),
    )
}

#[cfg(feature = "bindings-phonetic")]
unsafe fn owned_matches(
    matches: Vec<ScanMatch>,
    max_matches: usize,
    out_matches: *mut *mut LlevPhoneticOnlineMatch,
    out_count: *mut usize,
) -> Result<LlevStatus, (LlevStatus, String)> {
    if matches.len() > max_matches {
        return Err((
            LlevStatus::LimitExceeded,
            "online match count exceeds max_matches".into(),
        ));
    }
    let values: Box<[LlevPhoneticOnlineMatch]> = matches
        .into_iter()
        .map(|value| LlevPhoneticOnlineMatch {
            byte_start: value.byte_range.0,
            byte_end: value.byte_range.1,
            char_start: value.char_range.0,
            char_end: value.char_range.1,
            original_text: super::phonetic_dictionary::owned(value.original_text),
            normalized_text: super::phonetic_dictionary::owned(value.normalized_text),
            distance: value.distance,
            reserved: [0; 7],
        })
        .collect();
    let count = values.len();
    out_matches.write(if count == 0 {
        ptr::null_mut()
    } else {
        Box::into_raw(values) as *mut LlevPhoneticOnlineMatch
    });
    out_count.write(count);
    Ok(LlevStatus::Ok)
}

/// Compile a character-level phonetic matcher. Optional rules are cloned.
///
/// `max_pattern_scalars` must be positive; case_insensitive must be 0 or 1.
/// The native matcher normalizes its query before scanning and does not require
/// word boundaries. This is distinct from the word-boundary grep ABI.
///
/// # Safety
///
/// Pattern bytes, optional rule handle, and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_online_new(
    pattern: *const c_char,
    pattern_len: usize,
    rules: *const LlevPhoneticRuleSet,
    max_distance: u8,
    case_insensitive: u8,
    max_pattern_scalars: usize,
    out_grep: *mut *mut LlevPhoneticOnlineGrep,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_grep.is_null() {
                return Err((LlevStatus::NullPointer, "out_grep is null".into()));
            }
            if max_pattern_scalars == 0 || case_insensitive > 1 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "invalid online configuration".into(),
                ));
            }
            let pattern = utf8(pattern, pattern_len)?;
            if pattern.chars().count() > max_pattern_scalars {
                return Err((
                    LlevStatus::LimitExceeded,
                    "pattern exceeds max_pattern_scalars".into(),
                ));
            }
            let rules = rules
                .as_ref()
                .map(|value| value.inner.rules().to_vec())
                .unwrap_or_default();
            let inner = PhoneticGrepOnline::with_rules(pattern, rules, max_distance)
                .case_insensitive(case_insensitive != 0);
            out_grep.write(Box::into_raw(Box::new(LlevPhoneticOnlineGrep { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                pattern,
                pattern_len,
                rules,
                max_distance,
                case_insensitive,
                max_pattern_scalars,
                out_grep,
            );
            Err(unavailable())
        }
    })
}

/// Consume an online matcher; null is a no-op.
///
/// # Safety
///
/// A non-null handle must be live and unique.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_online_free(grep: *mut LlevPhoneticOnlineGrep) {
    if !grep.is_null() {
        drop(Box::from_raw(grep));
    }
}

/// Copy the normalized query into an owned string.
///
/// # Safety
///
/// Both pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_online_normalized_query(
    grep: *const LlevPhoneticOnlineGrep,
    out_text: *mut LlevOwnedString,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let grep = grep
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "grep is null".into()))?;
            if out_text.is_null() {
                return Err((LlevStatus::NullPointer, "out_text is null".into()));
            }
            out_text.write(super::phonetic_dictionary::owned(
                grep.inner.normalized_query(),
            ));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (grep, out_text);
            Err(unavailable())
        }
    })
}

/// Scan one document and return a newly owned result array.
///
/// Positive max_input_bytes and max_matches bound supplied data and returned
/// results. Outputs are unchanged on failure. Free with online_matches_free.
///
/// # Safety
///
/// Handle, document bytes, and outputs must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_online_scan(
    grep: *const LlevPhoneticOnlineGrep,
    document: *const c_char,
    document_len: usize,
    max_input_bytes: usize,
    max_matches: usize,
    out_matches: *mut *mut LlevPhoneticOnlineMatch,
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
            if max_input_bytes == 0 || max_matches == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "scan ceilings must be positive".into(),
                ));
            }
            if document_len > max_input_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "document exceeds max_input_bytes".into(),
                ));
            }
            let document = utf8(document, document_len)?;
            owned_matches(
                grep.inner.scan(document),
                max_matches,
                out_matches,
                out_count,
            )
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                grep,
                document,
                document_len,
                max_input_bytes,
                max_matches,
                out_matches,
                out_count,
            );
            Err(unavailable())
        }
    })
}

/// Consume an exact array/count pair returned by online_scan/stream_finish.
///
/// # Safety
///
/// The pair must not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_online_matches_free(
    matches: *mut LlevPhoneticOnlineMatch,
    count: usize,
) {
    if matches.is_null() {
        return;
    }
    let mut values = Box::from_raw(ptr::slice_from_raw_parts_mut(matches, count));
    for value in values.iter_mut() {
        super::llev_owned_string_free(&mut value.original_text);
        super::llev_owned_string_free(&mut value.normalized_text);
    }
}

/// Create a chunk-fed scanner from an immutable matcher clone.
///
/// # Safety
///
/// The matcher and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_online_stream_new(
    grep: *const LlevPhoneticOnlineGrep,
    max_total_bytes: usize,
    out_stream: *mut *mut LlevPhoneticOnlineStream,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let grep = grep
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "grep is null".into()))?;
            if out_stream.is_null() {
                return Err((LlevStatus::NullPointer, "out_stream is null".into()));
            }
            if max_total_bytes == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "max_total_bytes must be positive".into(),
                ));
            }
            out_stream.write(Box::into_raw(Box::new(LlevPhoneticOnlineStream {
                inner: Some(grep.inner.streaming()),
                bytes: 0,
                max_bytes: max_total_bytes,
            })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (grep, max_total_bytes, out_stream);
            Err(unavailable())
        }
    })
}

/// Consume an online stream; null is a no-op.
///
/// # Safety
///
/// A non-null handle must be live and unique.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_online_stream_free(stream: *mut LlevPhoneticOnlineStream) {
    if !stream.is_null() {
        drop(Box::from_raw(stream));
    }
}

/// Append one UTF-8 chunk; returned positions refer to the joined stream.
///
/// # Safety
///
/// Handle and input bytes must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_online_stream_feed(
    stream: *mut LlevPhoneticOnlineStream,
    chunk: *const c_char,
    chunk_len: usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let stream = stream
                .as_mut()
                .ok_or((LlevStatus::NullPointer, "stream is null".into()))?;
            let next = stream.bytes.checked_add(chunk_len).ok_or((
                LlevStatus::LimitExceeded,
                "stream byte count overflow".into(),
            ))?;
            if next > stream.max_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "stream exceeds max_total_bytes".into(),
                ));
            }
            let chunk = utf8(chunk, chunk_len)?;
            let inner = stream
                .inner
                .as_mut()
                .ok_or((LlevStatus::InvalidArgument, "stream was finished".into()))?;
            inner.feed(chunk);
            stream.bytes = next;
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (stream, chunk, chunk_len);
            Err(unavailable())
        }
    })
}

/// Finalize a stream once, copying all native matches. If max_matches is too
/// small the stream is still consumed, but outputs remain unchanged.
///
/// # Safety
///
/// Handle and outputs must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_online_stream_finish(
    stream: *mut LlevPhoneticOnlineStream,
    max_matches: usize,
    out_matches: *mut *mut LlevPhoneticOnlineMatch,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let stream = stream
                .as_mut()
                .ok_or((LlevStatus::NullPointer, "stream is null".into()))?;
            if out_matches.is_null() || out_count.is_null() {
                return Err((LlevStatus::NullPointer, "result output is null".into()));
            }
            if max_matches == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "max_matches must be positive".into(),
                ));
            }
            let inner = stream
                .inner
                .take()
                .ok_or((LlevStatus::InvalidArgument, "stream was finished".into()))?;
            owned_matches(inner.finish(), max_matches, out_matches, out_count)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (stream, max_matches, out_matches, out_count);
            Err(unavailable())
        }
    })
}
