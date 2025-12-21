//! Distance function FFI bindings.

use std::ffi::c_char;

use crate::distance::{standard_distance, transposition_distance};

/// Calculate Levenshtein distance between two strings.
///
/// # Safety
///
/// - Both `source` and `target` must be valid null-terminated UTF-8 strings
/// - Returns `usize::MAX` if either pointer is null or contains invalid UTF-8
#[no_mangle]
pub unsafe extern "C" fn llev_distance(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
) -> usize {
    let source = match super::cstr_to_str(source) {
        Some(s) => &s[..source_len.min(s.len())],
        None => return usize::MAX,
    };
    let target = match super::cstr_to_str(target) {
        Some(s) => &s[..target_len.min(s.len())],
        None => return usize::MAX,
    };

    standard_distance(source, target)
}

/// Calculate Levenshtein distance, returning early if it exceeds threshold.
///
/// # Safety
///
/// - Both `source` and `target` must be valid null-terminated UTF-8 strings
/// - Returns `usize::MAX` if either pointer is null or contains invalid UTF-8
/// - Returns `usize::MAX - 1` if distance exceeds threshold
#[no_mangle]
pub unsafe extern "C" fn llev_distance_threshold(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
    threshold: usize,
) -> usize {
    let source = match super::cstr_to_str(source) {
        Some(s) => &s[..source_len.min(s.len())],
        None => return usize::MAX,
    };
    let target = match super::cstr_to_str(target) {
        Some(s) => &s[..target_len.min(s.len())],
        None => return usize::MAX,
    };

    let dist = standard_distance(source, target);
    if dist > threshold {
        usize::MAX - 1
    } else {
        dist
    }
}

/// Calculate Damerau-Levenshtein distance between two strings.
///
/// This includes transposition as a single edit operation.
///
/// # Safety
///
/// - Both `source` and `target` must be valid null-terminated UTF-8 strings
/// - Returns `usize::MAX` if either pointer is null or contains invalid UTF-8
#[no_mangle]
pub unsafe extern "C" fn llev_damerau_distance(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
) -> usize {
    let source = match super::cstr_to_str(source) {
        Some(s) => &s[..source_len.min(s.len())],
        None => return usize::MAX,
    };
    let target = match super::cstr_to_str(target) {
        Some(s) => &s[..target_len.min(s.len())],
        None => return usize::MAX,
    };

    transposition_distance(source, target)
}

/// Calculate Damerau-Levenshtein distance, returning early if it exceeds threshold.
///
/// # Safety
///
/// - Both `source` and `target` must be valid null-terminated UTF-8 strings
/// - Returns `usize::MAX` if either pointer is null or contains invalid UTF-8
/// - Returns `usize::MAX - 1` if distance exceeds threshold
#[no_mangle]
pub unsafe extern "C" fn llev_damerau_distance_threshold(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
    threshold: usize,
) -> usize {
    let source = match super::cstr_to_str(source) {
        Some(s) => &s[..source_len.min(s.len())],
        None => return usize::MAX,
    };
    let target = match super::cstr_to_str(target) {
        Some(s) => &s[..target_len.min(s.len())],
        None => return usize::MAX,
    };

    let dist = transposition_distance(source, target);
    if dist > threshold {
        usize::MAX - 1
    } else {
        dist
    }
}
