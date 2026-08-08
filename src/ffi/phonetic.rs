//! Optional phonetic automata, language-product query, and rewrite-rule ABI.

use super::index::boundary;
#[cfg(feature = "bindings-phonetic")]
use super::index::{utf8, write_cursor};
#[cfg(feature = "bindings-phonetic")]
use super::LlevPhoneticRuleSetKind;
use super::{LlevQueryCursor, LlevStatus, LlevTransducer};
#[cfg(feature = "bindings-phonetic")]
use crate::bindings::{PhoneticPattern, PhoneticRuleSet};
use std::ffi::c_char;
use std::ptr;

/// Opaque reusable Unicode phonetic language automaton.
pub struct LlevPhoneticPattern {
    #[cfg(feature = "bindings-phonetic")]
    inner: PhoneticPattern,
}

/// Opaque reusable Unicode rewrite-rule set.
pub struct LlevPhoneticRuleSet {
    #[cfg(feature = "bindings-phonetic")]
    inner: PhoneticRuleSet,
}

/// Heap-owned UTF-8 output.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlevOwnedString {
    /// UTF-8 bytes, not NUL-terminated.
    pub data: *mut c_char,
    /// Number of bytes.
    pub len: usize,
}

impl Default for LlevOwnedString {
    fn default() -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
        }
    }
}

#[cfg(not(feature = "bindings-phonetic"))]
fn unavailable() -> (LlevStatus, String) {
    (
        LlevStatus::Unsupported,
        "phonetic bindings were not compiled; enable bindings-phonetic".into(),
    )
}

#[cfg(feature = "bindings-phonetic")]
unsafe fn write_owned(
    value: String,
    output: *mut LlevOwnedString,
) -> Result<(), (LlevStatus, String)> {
    if output.is_null() {
        return Err((LlevStatus::NullPointer, "output is null".into()));
    }
    let bytes = value.into_bytes().into_boxed_slice();
    if bytes.is_empty() {
        output.write(LlevOwnedString::default());
    } else {
        let len = bytes.len();
        output.write(LlevOwnedString {
            data: Box::into_raw(bytes).cast::<u8>().cast(),
            len,
        });
    }
    Ok(())
}

/// Free and clear owned UTF-8 returned by this ABI.
///
/// # Safety
///
/// A non-empty value must have been returned by this library and not already
/// freed.
#[no_mangle]
pub unsafe extern "C" fn llev_owned_string_free(value: *mut LlevOwnedString) {
    let Some(value) = value.as_mut() else {
        return;
    };
    if !value.data.is_null() {
        drop(Box::from_raw(ptr::slice_from_raw_parts_mut(
            value.data.cast::<u8>(),
            value.len,
        )));
    }
    *value = LlevOwnedString::default();
}

/// Compile a Unicode phonetic regular expression.
///
/// # Safety
///
/// Input and output pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_pattern_compile_regex(
    source: *const c_char,
    source_len: usize,
    out_pattern: *mut *mut LlevPhoneticPattern,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_pattern.is_null() {
                return Err((LlevStatus::NullPointer, "out_pattern is null".into()));
            }
            let inner = PhoneticPattern::from_regex(utf8(source, source_len)?)
                .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            out_pattern.write(Box::into_raw(Box::new(LlevPhoneticPattern { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (source, source_len, out_pattern);
            Err(unavailable())
        }
    })
}

/// Compile an import-free `.llre` document.
///
/// # Safety
///
/// Input and output pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_pattern_compile_llre(
    source: *const c_char,
    source_len: usize,
    out_pattern: *mut *mut LlevPhoneticPattern,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_pattern.is_null() {
                return Err((LlevStatus::NullPointer, "out_pattern is null".into()));
            }
            let inner = PhoneticPattern::from_llre(utf8(source, source_len)?)
                .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            out_pattern.write(Box::into_raw(Box::new(LlevPhoneticPattern { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (source, source_len, out_pattern);
            Err(unavailable())
        }
    })
}

/// Free a phonetic pattern. Existing cursors retain their own pattern product.
///
/// # Safety
///
/// A non-null handle must be live and cannot be reused.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_pattern_free(pattern: *mut LlevPhoneticPattern) {
    if !pattern.is_null() {
        drop(Box::from_raw(pattern));
    }
}

/// Return pattern state and transition counts.
///
/// # Safety
///
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_pattern_size(
    pattern: *const LlevPhoneticPattern,
    out_states: *mut usize,
    out_transitions: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let pattern = pattern
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "pattern is null".into()))?;
            if out_states.is_null() || out_transitions.is_null() {
                return Err((LlevStatus::NullPointer, "an output is null".into()));
            }
            out_states.write(pattern.inner.state_count());
            out_transitions.write(pattern.inner.transition_count());
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (pattern, out_states, out_transitions);
            Err(unavailable())
        }
    })
}

/// Test complete-string acceptance.
///
/// # Safety
///
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_pattern_matches(
    pattern: *const LlevPhoneticPattern,
    input: *const c_char,
    input_len: usize,
    out_matches: *mut u8,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let pattern = pattern
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "pattern is null".into()))?;
            if out_matches.is_null() {
                return Err((LlevStatus::NullPointer, "out_matches is null".into()));
            }
            out_matches.write(u8::from(pattern.inner.matches(utf8(input, input_len)?)));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (pattern, input, input_len, out_matches);
            Err(unavailable())
        }
    })
}

/// Query a Unicode dictionary by distance to a phonetic language.
///
/// # Safety
///
/// All pointers must reference live handles or writable output storage.
#[no_mangle]
pub unsafe extern "C" fn llev_transducer_query_pattern(
    transducer: *const LlevTransducer,
    pattern: *const LlevPhoneticPattern,
    max_distance: u8,
    out_cursor: *mut *mut LlevQueryCursor,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let transducer = transducer
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "transducer is null".into()))?;
            let pattern = pattern
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "pattern is null".into()))?;
            write_cursor(
                transducer.inner.query_pattern(&pattern.inner, max_distance),
                out_cursor,
            )
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (transducer, pattern, max_distance, out_cursor);
            Err(unavailable())
        }
    })
}

/// Parse an import-free `.llev` rewrite-rule document.
///
/// # Safety
///
/// Input and output pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_rules_parse(
    source: *const c_char,
    source_len: usize,
    out_rules: *mut *mut LlevPhoneticRuleSet,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_rules.is_null() {
                return Err((LlevStatus::NullPointer, "out_rules is null".into()));
            }
            let inner = PhoneticRuleSet::parse(utf8(source, source_len)?)
                .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            out_rules.write(Box::into_raw(Box::new(LlevPhoneticRuleSet { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (source, source_len, out_rules);
            Err(unavailable())
        }
    })
}

/// Construct a built-in rewrite-rule set.
///
/// # Safety
///
/// `out_rules` must be writable.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_rules_builtin(
    kind: u32,
    out_rules: *mut *mut LlevPhoneticRuleSet,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_rules.is_null() {
                return Err((LlevStatus::NullPointer, "out_rules is null".into()));
            }
            let inner = match LlevPhoneticRuleSetKind::try_from(kind).map_err(|()| {
                (
                    LlevStatus::InvalidArgument,
                    format!("unknown phonetic rule-set kind {kind}"),
                )
            })? {
                LlevPhoneticRuleSetKind::EnglishOrthography => {
                    PhoneticRuleSet::english_orthography()
                }
                LlevPhoneticRuleSetKind::EnglishPhonetic => PhoneticRuleSet::english_phonetic(),
            };
            out_rules.write(Box::into_raw(Box::new(LlevPhoneticRuleSet { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (kind, out_rules);
            Err(unavailable())
        }
    })
}

/// Free a rewrite-rule set.
///
/// # Safety
///
/// A non-null handle must be live and cannot be reused.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_rules_free(rules: *mut LlevPhoneticRuleSet) {
    if !rules.is_null() {
        drop(Box::from_raw(rules));
    }
}

/// Return the number of enabled rules.
///
/// # Safety
///
/// Both pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_rules_len(
    rules: *const LlevPhoneticRuleSet,
    out_len: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let rules = rules
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "rules is null".into()))?;
            if out_len.is_null() {
                return Err((LlevStatus::NullPointer, "out_len is null".into()));
            }
            out_len.write(rules.inner.len());
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (rules, out_len);
            Err(unavailable())
        }
    })
}

/// Apply rewrite rules and return owned UTF-8.
///
/// # Safety
///
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_rules_apply(
    rules: *const LlevPhoneticRuleSet,
    input: *const c_char,
    input_len: usize,
    out_text: *mut LlevOwnedString,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let rules = rules
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "rules is null".into()))?;
            write_owned(rules.inner.apply(utf8(input, input_len)?), out_text)?;
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (rules, input, input_len, out_text);
            Err(unavailable())
        }
    })
}
