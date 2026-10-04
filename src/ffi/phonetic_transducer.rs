//! Incremental native phonetic rewrite transducer with explicit output ceilings.

use super::index::boundary;
#[cfg(feature = "bindings-phonetic")]
use super::index::utf8;
use super::{LlevOwnedString, LlevPhoneticRuleSet, LlevStatus};
#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::{OnlinePhoneticTransducerChar, RewriteRuleChar};
use std::ffi::c_char;

/// Stateful rule transducer. Finish once or reset to reuse.
pub struct LlevPhoneticTransducer {
    #[cfg(feature = "bindings-phonetic")]
    rules: Vec<RewriteRuleChar>,
    #[cfg(feature = "bindings-phonetic")]
    inner: OnlinePhoneticTransducerChar,
    #[cfg(feature = "bindings-phonetic")]
    finished: bool,
}

#[cfg(not(feature = "bindings-phonetic"))]
fn unavailable() -> (LlevStatus, String) {
    (
        LlevStatus::Unsupported,
        "phonetic bindings were not compiled".into(),
    )
}

/// Construct a native online rewrite transducer from cloned rules.
/// A null rule handle selects an empty rule set (identity rewrite).
///
/// # Safety
///
/// Optional rules and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_transducer_new(
    rules: *const LlevPhoneticRuleSet,
    out_transducer: *mut *mut LlevPhoneticTransducer,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_transducer.is_null() {
                return Err((LlevStatus::NullPointer, "out_transducer is null".into()));
            }
            let rules = rules
                .as_ref()
                .map(|value| value.inner.rules().to_vec())
                .unwrap_or_default();
            let inner = OnlinePhoneticTransducerChar::new(rules.clone());
            out_transducer.write(Box::into_raw(Box::new(LlevPhoneticTransducer {
                rules,
                inner,
                finished: false,
            })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (rules, out_transducer);
            Err(unavailable())
        }
    })
}

/// Consume a transducer; null is a no-op.
///
/// # Safety
///
/// A non-null handle must be live and unique.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_transducer_free(transducer: *mut LlevPhoneticTransducer) {
    if !transducer.is_null() {
        drop(Box::from_raw(transducer));
    }
}

/// Feed a UTF-8 chunk and return only output ready for emission. Contextual
/// rewrite rules may defer characters until a later feed or finish call.
/// A result exceeding max_output_bytes resets the transducer and leaves the
/// output unchanged. Both input/output ceilings must be positive.
///
/// # Safety
///
/// Handle, chunk bytes, and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_transducer_feed(
    transducer: *mut LlevPhoneticTransducer,
    chunk: *const c_char,
    chunk_len: usize,
    max_input_scalars: usize,
    max_output_bytes: usize,
    out_text: *mut LlevOwnedString,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let transducer = transducer
                .as_mut()
                .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
            if out_text.is_null() {
                return Err((LlevStatus::NullPointer, "out_text is null".into()));
            }
            if max_input_scalars == 0 || max_output_bytes == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "transducer ceilings must be positive".into(),
                ));
            }
            if transducer.finished {
                return Err((
                    LlevStatus::InvalidArgument,
                    "transducer was finished".into(),
                ));
            }
            let chunk = utf8(chunk, chunk_len)?;
            if chunk.chars().count() > max_input_scalars {
                return Err((
                    LlevStatus::LimitExceeded,
                    "chunk exceeds max_input_scalars".into(),
                ));
            }
            let mut text = String::new();
            for character in chunk.chars() {
                let mut oversized = false;
                for ready in transducer.inner.feed(character) {
                    text.push(ready);
                    if text.len() > max_output_bytes {
                        oversized = true;
                        break;
                    }
                }
                if oversized {
                    transducer.inner.reset();
                    return Err((
                        LlevStatus::LimitExceeded,
                        "rewrite output exceeds max_output_bytes; transducer reset".into(),
                    ));
                }
            }
            out_text.write(super::phonetic_dictionary::owned(text));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                transducer,
                chunk,
                chunk_len,
                max_input_scalars,
                max_output_bytes,
                out_text,
            );
            Err(unavailable())
        }
    })
}

/// Signal end of input and return any buffered output. A result exceeding
/// max_output_bytes resets the transducer and leaves the output unchanged.
/// Subsequent feed/finish calls require reset.
///
/// # Safety
///
/// Handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_transducer_finish(
    transducer: *mut LlevPhoneticTransducer,
    max_output_bytes: usize,
    out_text: *mut LlevOwnedString,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let transducer = transducer
                .as_mut()
                .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
            if out_text.is_null() {
                return Err((LlevStatus::NullPointer, "out_text is null".into()));
            }
            if max_output_bytes == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "max_output_bytes must be positive".into(),
                ));
            }
            if transducer.finished {
                return Err((
                    LlevStatus::InvalidArgument,
                    "transducer was finished".into(),
                ));
            }
            let mut text = String::new();
            let mut oversized = false;
            for ready in transducer.inner.finish() {
                text.push(ready);
                if text.len() > max_output_bytes {
                    oversized = true;
                    break;
                }
            }
            if oversized {
                transducer.inner.reset();
                return Err((
                    LlevStatus::LimitExceeded,
                    "rewrite output exceeds max_output_bytes; transducer reset".into(),
                ));
            }
            transducer.finished = true;
            out_text.write(super::phonetic_dictionary::owned(text));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (transducer, max_output_bytes, out_text);
            Err(unavailable())
        }
    })
}

/// Reset rewrite state and counters while retaining cloned rules.
///
/// # Safety
///
/// Handle must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_transducer_reset(
    transducer: *mut LlevPhoneticTransducer,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let transducer = transducer
                .as_mut()
                .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
            transducer.inner.reset();
            transducer.finished = false;
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = transducer;
            Err(unavailable())
        }
    })
}

/// Normalize an independent UTF-8 string using this handle's cloned rules.
/// Does not change incremental state. Input and output ceilings are positive;
/// an overlarge output returns LIMIT_EXCEEDED and leaves output unchanged.
///
/// # Safety
///
/// Handle, input bytes, and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_transducer_normalize(
    transducer: *const LlevPhoneticTransducer,
    input: *const c_char,
    input_len: usize,
    max_input_scalars: usize,
    max_output_bytes: usize,
    out_text: *mut LlevOwnedString,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let transducer = transducer
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
            if out_text.is_null() {
                return Err((LlevStatus::NullPointer, "out_text is null".into()));
            }
            if max_input_scalars == 0 || max_output_bytes == 0 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "transducer ceilings must be positive".into(),
                ));
            }
            let input = utf8(input, input_len)?;
            if input.chars().count() > max_input_scalars {
                return Err((
                    LlevStatus::LimitExceeded,
                    "input exceeds max_input_scalars".into(),
                ));
            }
            let mut inner = OnlinePhoneticTransducerChar::new(transducer.rules.clone());
            let text = inner.normalize(input);
            if text.len() > max_output_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "rewrite output exceeds max_output_bytes".into(),
                ));
            }
            out_text.write(super::phonetic_dictionary::owned(text));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                transducer,
                input,
                input_len,
                max_input_scalars,
                max_output_bytes,
                out_text,
            );
            Err(unavailable())
        }
    })
}
