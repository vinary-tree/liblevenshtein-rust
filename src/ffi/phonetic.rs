//! Optional phonetic automata, language-product query, and rewrite-rule ABI.

use super::index::boundary;
#[cfg(feature = "bindings-phonetic")]
use super::index::{utf8, write_cursor};
#[cfg(feature = "bindings-phonetic")]
use super::LlevAlgorithm;
#[cfg(feature = "bindings-phonetic")]
use super::LlevPhoneticRuleSetKind;
use super::LlevUtf8View;
use super::{LlevQueryCursor, LlevStatus, LlevTransducer};
#[cfg(feature = "bindings-phonetic")]
use crate::bindings::{validate_phonetic_regex_size, PhoneticPattern, PhoneticRuleSet};
#[cfg(feature = "bindings-phonetic")]
use crate::phonetic::PhoneticGrep;
use std::ffi::c_char;
use std::ptr;

#[cfg(feature = "bindings-phonetic")]
unsafe fn file_paths(
    path: *const c_char,
    path_len: usize,
    search_paths: *const LlevUtf8View,
    search_path_count: usize,
    max_total_path_bytes: usize,
) -> Result<(std::path::PathBuf, Vec<std::path::PathBuf>), (LlevStatus, String)> {
    if max_total_path_bytes == 0 {
        return Err((
            LlevStatus::InvalidArgument,
            "max_total_path_bytes must be positive".into(),
        ));
    }
    if search_path_count > 64 {
        return Err((
            LlevStatus::LimitExceeded,
            "at most 64 search paths are allowed".into(),
        ));
    }
    if search_path_count > 0 && search_paths.is_null() {
        return Err((LlevStatus::NullPointer, "search_paths is null".into()));
    }
    let path = utf8(path, path_len)?;
    if path.is_empty() {
        return Err((LlevStatus::InvalidArgument, "file path is empty".into()));
    }
    let mut total = path_len;
    if total > max_total_path_bytes {
        return Err((
            LlevStatus::LimitExceeded,
            "file paths exceed max_total_path_bytes".into(),
        ));
    }
    let views = if search_path_count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(search_paths, search_path_count)
    };
    let mut paths = Vec::with_capacity(search_path_count);
    for view in views {
        total = total.checked_add(view.len).ok_or((
            LlevStatus::LimitExceeded,
            "file path byte count overflow".into(),
        ))?;
        if total > max_total_path_bytes {
            return Err((
                LlevStatus::LimitExceeded,
                "file paths exceed max_total_path_bytes".into(),
            ));
        }
        let value = utf8(view.data, view.len)?;
        if value.is_empty() {
            return Err((LlevStatus::InvalidArgument, "search path is empty".into()));
        }
        paths.push(std::path::PathBuf::from(value));
    }
    Ok((std::path::PathBuf::from(path), paths))
}

/// Opaque reusable Unicode phonetic language automaton.
pub struct LlevPhoneticPattern {
    #[cfg(feature = "bindings-phonetic")]
    pub(super) inner: PhoneticPattern,
}

/// Opaque reusable Unicode rewrite-rule set.
pub struct LlevPhoneticRuleSet {
    #[cfg(feature = "bindings-phonetic")]
    pub(super) inner: PhoneticRuleSet,
}

/// Immutable word-boundary phonetic grep configuration.
pub struct LlevPhoneticGrep {
    #[cfg(feature = "bindings-phonetic")]
    inner: PhoneticGrep,
}

/// One copied word-boundary grep result; byte columns are zero-based and
/// end-exclusive within the one-based `line_number`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlevPhoneticGrepMatch {
    /// One-based logical line in a scanned document; one for line scans.
    pub line_number: usize,
    /// Zero-based byte offset within the line.
    pub start_byte: usize,
    /// End-exclusive byte offset within the line.
    pub end_byte: usize,
    /// Exact bounded edit distance after the matcher's normalization.
    pub distance: u8,
    /// Reserved for ABI-compatible growth; fixed to zero.
    pub reserved: [u8; 7],
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

/// Compile a `.llre` file with native import resolution and the shared NFA
/// state ceiling. Search paths are UTF-8 directory views; zero paths use
/// the native loader defaults. This reads trusted local files and imports.
///
/// # Safety
///
/// Path buffers, optional search-path array, and output must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_pattern_load_llre_file(
    path: *const c_char,
    path_len: usize,
    search_paths: *const LlevUtf8View,
    search_path_count: usize,
    max_total_path_bytes: usize,
    out_pattern: *mut *mut LlevPhoneticPattern,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_pattern.is_null() {
                return Err((LlevStatus::NullPointer, "out_pattern is null".into()));
            }
            let (path, search_paths) = file_paths(
                path,
                path_len,
                search_paths,
                search_path_count,
                max_total_path_bytes,
            )?;
            let inner = PhoneticPattern::from_llre_file(&path, &search_paths)
                .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            out_pattern.write(Box::into_raw(Box::new(LlevPhoneticPattern { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                path,
                path_len,
                search_paths,
                search_path_count,
                max_total_path_bytes,
                out_pattern,
            );
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

/// Load a `.llev` file with native include resolution. Zero search paths use
/// relative includes and native loader defaults. This reads trusted local
/// files and their includes; the path ceiling does not bound file contents.
///
/// # Safety
///
/// Path buffers, optional search-path array, and output must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_rules_load_file(
    path: *const c_char,
    path_len: usize,
    search_paths: *const LlevUtf8View,
    search_path_count: usize,
    max_total_path_bytes: usize,
    out_rules: *mut *mut LlevPhoneticRuleSet,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_rules.is_null() {
                return Err((LlevStatus::NullPointer, "out_rules is null".into()));
            }
            let (path, search_paths) = file_paths(
                path,
                path_len,
                search_paths,
                search_path_count,
                max_total_path_bytes,
            )?;
            let inner = PhoneticRuleSet::from_file(&path, &search_paths)
                .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            out_rules.write(Box::into_raw(Box::new(LlevPhoneticRuleSet { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                path,
                path_len,
                search_paths,
                search_path_count,
                max_total_path_bytes,
                out_rules,
            );
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

#[cfg(feature = "bindings-phonetic")]
unsafe fn write_grep_matches(
    matches: &[LlevPhoneticGrepMatch],
    out_matches: *mut LlevPhoneticGrepMatch,
    capacity: usize,
    out_count: *mut usize,
) -> Result<(), (LlevStatus, String)> {
    let output_count = out_count
        .as_mut()
        .ok_or((LlevStatus::NullPointer, "out_count is null".into()))?;
    *output_count = matches.len();
    if matches.len() > capacity {
        return Err((
            LlevStatus::LimitExceeded,
            "phonetic grep result capacity is too small".into(),
        ));
    }
    if !matches.is_empty() {
        if out_matches.is_null() {
            return Err((LlevStatus::NullPointer, "out_matches is null".into()));
        }
        ptr::copy_nonoverlapping(matches.as_ptr(), out_matches, matches.len());
    }
    Ok(())
}

/// Compile a bounded Unicode word-boundary grep matcher, optionally sharing
/// the parsed rule set's semantics while owning an independent rule clone.
///
/// # Safety
///
/// Non-empty `pattern`, non-null `rules`, and `out_grep` must be valid. A null
/// `rules` selects no phonetic rewrite rules.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_grep_new(
    pattern: *const c_char,
    pattern_len: usize,
    rules: *const LlevPhoneticRuleSet,
    max_distance: u8,
    algorithm: u32,
    case_insensitive: u8,
    out_grep: *mut *mut LlevPhoneticGrep,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            if out_grep.is_null() {
                return Err((LlevStatus::NullPointer, "out_grep is null".into()));
            }
            let pattern = utf8(pattern, pattern_len)?;
            validate_phonetic_regex_size(pattern)
                .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?;
            let algorithm = LlevAlgorithm::try_from(algorithm)
                .map_err(|()| (LlevStatus::InvalidArgument, "unknown algorithm".into()))?;
            if case_insensitive > 1 {
                return Err((
                    LlevStatus::InvalidArgument,
                    "case_insensitive must be zero or one".into(),
                ));
            }
            let inner = if let Some(rules) = rules.as_ref() {
                PhoneticGrep::with_loaded_rules(pattern, rules.inner.rules().to_vec(), max_distance)
            } else {
                PhoneticGrep::from_pattern_with_algorithm(pattern, max_distance, algorithm.into())
            }
            .map_err(|error| (LlevStatus::InvalidArgument, error.to_string()))?
            .algorithm(algorithm.into())
            .case_insensitive(case_insensitive != 0);
            out_grep.write(Box::into_raw(Box::new(LlevPhoneticGrep { inner })));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                pattern,
                pattern_len,
                rules,
                max_distance,
                algorithm,
                case_insensitive,
                out_grep,
            );
            Err(unavailable())
        }
    })
}

/// Consume a grep configuration. NULL is a no-op.
///
/// # Safety
///
/// A non-null pointer must be owned, live, and not used again.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_grep_free(grep: *mut LlevPhoneticGrep) {
    if !grep.is_null() {
        drop(Box::from_raw(grep));
    }
}

/// Report the effective and local pattern override distance bounds.
///
/// # Safety
///
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_grep_distance_config(
    grep: *const LlevPhoneticGrep,
    out_effective: *mut u8,
    out_local: *mut u8,
    out_has_local: *mut u8,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let grep = grep
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "grep is null".into()))?;
            if out_effective.is_null() || out_local.is_null() || out_has_local.is_null() {
                return Err((LlevStatus::NullPointer, "an output is null".into()));
            }
            out_effective.write(grep.inner.effective_distance());
            out_local.write(grep.inner.local_distance().unwrap_or(0));
            out_has_local.write(u8::from(grep.inner.local_distance().is_some()));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (grep, out_effective, out_local, out_has_local);
            Err(unavailable())
        }
    })
}

/// Match one UTF-8 candidate and return an optional native edit distance.
///
/// # Safety
///
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_grep_matches(
    grep: *const LlevPhoneticGrep,
    candidate: *const c_char,
    candidate_len: usize,
    max_candidate_bytes: usize,
    out_distance: *mut u8,
    out_matches: *mut u8,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let grep = grep
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "grep is null".into()))?;
            if out_distance.is_null() || out_matches.is_null() {
                return Err((LlevStatus::NullPointer, "an output is null".into()));
            }
            if max_candidate_bytes == 0 || candidate_len > max_candidate_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "candidate exceeds max_candidate_bytes".into(),
                ));
            }
            let candidate = utf8(candidate, candidate_len)?;
            let distance = grep.inner.matches(candidate);
            out_distance.write(distance.unwrap_or(0));
            out_matches.write(u8::from(distance.is_some()));
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                grep,
                candidate,
                candidate_len,
                max_candidate_bytes,
                out_distance,
                out_matches,
            );
            Err(unavailable())
        }
    })
}

/// Find non-overlapping word matches within a single line.
///
/// `max_input_bytes` must be positive; results are copied transactionally.
/// `out_count` still receives the required capacity on LIMIT_EXCEEDED.
///
/// # Safety
///
/// All non-null pointers must be valid and output storage must not alias input.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_grep_scan_line(
    grep: *const LlevPhoneticGrep,
    line: *const c_char,
    line_len: usize,
    max_input_bytes: usize,
    out_matches: *mut LlevPhoneticGrepMatch,
    capacity: usize,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let grep = grep
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "grep is null".into()))?;
            if max_input_bytes == 0 || line_len > max_input_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "line exceeds max_input_bytes".into(),
                ));
            }
            let line = utf8(line, line_len)?;
            let matches: Vec<_> = grep
                .inner
                .find_in_line(line)
                .into_iter()
                .map(|found| LlevPhoneticGrepMatch {
                    line_number: 1,
                    start_byte: found.start_column - 1,
                    end_byte: found.end_column,
                    distance: found.distance,
                    reserved: [0; 7],
                })
                .collect();
            write_grep_matches(&matches, out_matches, capacity, out_count)?;
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                grep,
                line,
                line_len,
                max_input_bytes,
                out_matches,
                capacity,
                out_count,
            );
            Err(unavailable())
        }
    })
}

/// Find non-overlapping word matches in each logical line of a UTF-8 document.
///
/// Columns remain zero-based within a one-based line. A bounded caller output
/// array receives all results or none; `out_count` reports required capacity.
///
/// # Safety
///
/// All non-null pointers must be valid and output storage must not alias input.
#[no_mangle]
pub unsafe extern "C" fn llev_phonetic_grep_scan_text(
    grep: *const LlevPhoneticGrep,
    document: *const c_char,
    document_len: usize,
    max_input_bytes: usize,
    out_matches: *mut LlevPhoneticGrepMatch,
    capacity: usize,
    out_count: *mut usize,
) -> LlevStatus {
    boundary(|| {
        #[cfg(feature = "bindings-phonetic")]
        {
            let grep = grep
                .as_ref()
                .ok_or((LlevStatus::NullPointer, "grep is null".into()))?;
            if max_input_bytes == 0 || document_len > max_input_bytes {
                return Err((
                    LlevStatus::LimitExceeded,
                    "document exceeds max_input_bytes".into(),
                ));
            }
            let document = utf8(document, document_len)?;
            let matches: Vec<_> = grep
                .inner
                .grep_file(document)
                .flat_map(|line| {
                    line.matches
                        .into_iter()
                        .map(move |found| LlevPhoneticGrepMatch {
                            line_number: line.line_number,
                            start_byte: found.start_column - 1,
                            end_byte: found.end_column,
                            distance: found.distance,
                            reserved: [0; 7],
                        })
                })
                .collect();
            write_grep_matches(&matches, out_matches, capacity, out_count)?;
            Ok(LlevStatus::Ok)
        }
        #[cfg(not(feature = "bindings-phonetic"))]
        {
            let _ = (
                grep,
                document,
                document_len,
                max_input_bytes,
                out_matches,
                capacity,
                out_count,
            );
            Err(unavailable())
        }
    })
}
