//! Versioned native phonetic AOT bytes with explicit feature negotiation.

use super::index::boundary;
use super::{LlevPhoneticPattern, LlevPhoneticRuleSet, LlevStatus};
#[cfg(all(feature = "bindings-phonetic", feature = "serialization"))]
use crate::bindings::{PhoneticPattern, PhoneticRuleSet};
use std::ptr;

/// Owned arbitrary binary bytes, never interpreted as UTF-8 or NUL-terminated.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevOwnedBytes {
    /// Owned byte storage, null only for zero length.
    pub data: *mut u8,
    /// Exact byte length.
    pub len: usize,
}

impl Default for LlevOwnedBytes {
    fn default() -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
        }
    }
}

/// Release and clear a versioned byte buffer; null is a no-op.
///
/// # Safety
///
/// Nonempty buffers must originate from this library and not yet be freed.
#[no_mangle]
pub unsafe extern "C" fn llev_owned_bytes_free(value: *mut LlevOwnedBytes) {
    let Some(value) = value.as_mut() else {
        return;
    };
    if !value.data.is_null() {
        drop(Box::from_raw(ptr::slice_from_raw_parts_mut(
            value.data, value.len,
        )));
    }
    *value = LlevOwnedBytes::default();
}

#[cfg(all(feature = "bindings-phonetic", feature = "serialization"))]
const MAX_AOT_BYTES: usize = 16 * 1024 * 1024;

#[cfg(all(feature = "bindings-phonetic", feature = "serialization"))]
fn validate_limit(max_bytes: usize) -> Result<(), (LlevStatus, String)> {
    if max_bytes == 0 {
        return Err((
            LlevStatus::InvalidArgument,
            "AOT byte ceiling must be positive".into(),
        ));
    }
    if max_bytes > MAX_AOT_BYTES {
        return Err((
            LlevStatus::LimitExceeded,
            "AOT byte ceiling exceeds 16 MiB hard limit".into(),
        ));
    }
    Ok(())
}

#[cfg(all(feature = "bindings-phonetic", feature = "serialization"))]
unsafe fn input_bytes<'a>(
    data: *const u8,
    len: usize,
    max_bytes: usize,
) -> Result<&'a [u8], (LlevStatus, String)> {
    validate_limit(max_bytes)?;
    if len > max_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            "AOT input exceeds byte ceiling".into(),
        ));
    }
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err((LlevStatus::NullPointer, "AOT data is null".into()));
    }
    Ok(std::slice::from_raw_parts(data, len))
}

#[cfg(all(feature = "bindings-phonetic", feature = "serialization"))]
unsafe fn write_bytes(
    data: Vec<u8>,
    max_bytes: usize,
    output: *mut LlevOwnedBytes,
) -> Result<LlevStatus, (LlevStatus, String)> {
    if data.len() > max_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            "AOT output exceeds byte ceiling".into(),
        ));
    }
    let data = data.into_boxed_slice();
    let len = data.len();
    output.write(if len == 0 {
        LlevOwnedBytes::default()
    } else {
        LlevOwnedBytes {
            data: Box::into_raw(data) as *mut u8,
            len,
        }
    });
    Ok(LlevStatus::Ok)
}

#[cfg(not(all(feature = "bindings-phonetic", feature = "serialization")))]
fn unavailable() -> (LlevStatus, String) {
    (
        LlevStatus::Unsupported,
        "phonetic AOT requires bindings-phonetic and serialization".into(),
    )
}

/// Serialize a native Unicode rewrite-rule set to versioned `.llev` bytes.
/// Requires `LLEV_BUILD_FEATURE_PHONETIC_AOT`.
///
/// # Safety
///
/// Rule handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_rules_to_bytes(
    rules: *const LlevPhoneticRuleSet,
    max_output_bytes: usize,
    out_bytes: *mut LlevOwnedBytes,
) -> LlevStatus {
    boundary(|| {
        #[cfg(all(feature = "bindings-phonetic", feature = "serialization"))]
        {
            let rules = rules
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "rules is null".into()))?;
            if out_bytes.is_null() {
                return Err((LlevStatus::NullPointer, "out_bytes is null".into()));
            }
            validate_limit(max_output_bytes)?;
            let bytes = rules
                .inner
                .to_compiled_bytes()
                .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            write_bytes(bytes, max_output_bytes, out_bytes)
        }
        #[cfg(not(all(feature = "bindings-phonetic", feature = "serialization")))]
        {
            let _ = (rules, max_output_bytes, out_bytes);
            Err(unavailable())
        }
    })
}

/// Restore Unicode rewrite rules from versioned `.llev` bytes.
/// Requires `LLEV_BUILD_FEATURE_PHONETIC_AOT`.
///
/// # Safety
///
/// Input bytes and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_rules_from_bytes(
    data: *const u8,
    data_len: usize,
    max_input_bytes: usize,
    out_rules: *mut *mut LlevPhoneticRuleSet,
) -> LlevStatus {
    boundary(|| {
        #[cfg(all(feature = "bindings-phonetic", feature = "serialization"))]
        {
            if out_rules.is_null() {
                return Err((LlevStatus::NullPointer, "out_rules is null".into()));
            }
            let data = input_bytes(data, data_len, max_input_bytes)?;
            let inner = PhoneticRuleSet::from_compiled_bytes(data)
                .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            out_rules.write(Box::into_raw(Box::new(LlevPhoneticRuleSet { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(all(feature = "bindings-phonetic", feature = "serialization")))]
        {
            let _ = (data, data_len, max_input_bytes, out_rules);
            Err(unavailable())
        }
    })
}

/// Serialize a compiled pattern NFA and available `.llre` metadata.
/// Requires `LLEV_BUILD_FEATURE_PHONETIC_AOT`.
///
/// # Safety
///
/// Pattern handle and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_pattern_to_bytes(
    pattern: *const LlevPhoneticPattern,
    max_output_bytes: usize,
    out_bytes: *mut LlevOwnedBytes,
) -> LlevStatus {
    boundary(|| {
        #[cfg(all(feature = "bindings-phonetic", feature = "serialization"))]
        {
            let pattern = pattern
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "pattern is null".into()))?;
            if out_bytes.is_null() {
                return Err((LlevStatus::NullPointer, "out_bytes is null".into()));
            }
            validate_limit(max_output_bytes)?;
            let bytes = pattern
                .inner
                .to_compiled_bytes()
                .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            write_bytes(bytes, max_output_bytes, out_bytes)
        }
        #[cfg(not(all(feature = "bindings-phonetic", feature = "serialization")))]
        {
            let _ = (pattern, max_output_bytes, out_bytes);
            Err(unavailable())
        }
    })
}

/// Restore a compiled `.llre` pattern from versioned bytes. The decoded NFA
/// is checked against the shared language-product state ceiling.
/// Requires `LLEV_BUILD_FEATURE_PHONETIC_AOT`.
///
/// # Safety
///
/// Input bytes and output pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_pattern_from_bytes(
    data: *const u8,
    data_len: usize,
    max_input_bytes: usize,
    out_pattern: *mut *mut LlevPhoneticPattern,
) -> LlevStatus {
    boundary(|| {
        #[cfg(all(feature = "bindings-phonetic", feature = "serialization"))]
        {
            if out_pattern.is_null() {
                return Err((LlevStatus::NullPointer, "out_pattern is null".into()));
            }
            let data = input_bytes(data, data_len, max_input_bytes)?;
            let inner = PhoneticPattern::from_compiled_bytes(data)
                .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            out_pattern.write(Box::into_raw(Box::new(LlevPhoneticPattern { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(all(feature = "bindings-phonetic", feature = "serialization")))]
        {
            let _ = (data, data_len, max_input_bytes, out_pattern);
            Err(unavailable())
        }
    })
}
