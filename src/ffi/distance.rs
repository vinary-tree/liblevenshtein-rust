//! Distance function FFI bindings.

use std::ffi::c_char;

use crate::distance::{
    standard_distance, standard_distance_bounded, transposition_distance,
    transposition_distance_bounded,
};

/// Calculate Levenshtein distance between two strings.
///
/// # Safety
///
/// - Both `source` and `target` must be valid UTF-8 buffers for their byte lengths
/// - Returns `usize::MAX` if either pointer is null or contains invalid UTF-8
#[no_mangle]
pub unsafe extern "C" fn llev_distance(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
) -> usize {
    let source = match super::cbuf_to_str(source, source_len) {
        Some(s) => s,
        None => return usize::MAX,
    };
    let target = match super::cbuf_to_str(target, target_len) {
        Some(s) => s,
        None => return usize::MAX,
    };

    standard_distance(source, target)
}

/// Calculate Levenshtein distance, returning early if it exceeds threshold.
///
/// # Safety
///
/// - Both `source` and `target` must be valid UTF-8 buffers for their byte lengths
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
    let source = match super::cbuf_to_str(source, source_len) {
        Some(s) => s,
        None => return usize::MAX,
    };
    let target = match super::cbuf_to_str(target, target_len) {
        Some(s) => s,
        None => return usize::MAX,
    };

    standard_distance_bounded(source, target, threshold).unwrap_or(usize::MAX - 1)
}

/// Calculate Damerau-Levenshtein distance between two strings.
///
/// This includes transposition as a single edit operation.
///
/// # Safety
///
/// - Both `source` and `target` must be valid UTF-8 buffers for their byte lengths
/// - Returns `usize::MAX` if either pointer is null or contains invalid UTF-8
#[no_mangle]
pub unsafe extern "C" fn llev_damerau_distance(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
) -> usize {
    let source = match super::cbuf_to_str(source, source_len) {
        Some(s) => s,
        None => return usize::MAX,
    };
    let target = match super::cbuf_to_str(target, target_len) {
        Some(s) => s,
        None => return usize::MAX,
    };

    transposition_distance(source, target)
}

/// Calculate Damerau-Levenshtein distance, returning early if it exceeds threshold.
///
/// # Safety
///
/// - Both `source` and `target` must be valid UTF-8 buffers for their byte lengths
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
    let source = match super::cbuf_to_str(source, source_len) {
        Some(s) => s,
        None => return usize::MAX,
    };
    let target = match super::cbuf_to_str(target, target_len) {
        Some(s) => s,
        None => return usize::MAX,
    };

    transposition_distance_bounded(source, target, threshold).unwrap_or(usize::MAX - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn ffi_distance_rejects_lengths_inside_utf8_codepoint() {
        let source = CString::new("é").unwrap();
        let target = CString::new("e").unwrap();

        unsafe {
            assert_eq!(
                llev_distance(source.as_ptr(), 1, target.as_ptr(), 1),
                usize::MAX
            );
            assert_eq!(
                llev_distance_threshold(source.as_ptr(), 1, target.as_ptr(), 1, 1),
                usize::MAX
            );
            assert_eq!(
                llev_damerau_distance(source.as_ptr(), 1, target.as_ptr(), 1),
                usize::MAX
            );
            assert_eq!(
                llev_damerau_distance_threshold(source.as_ptr(), 1, target.as_ptr(), 1, 1),
                usize::MAX
            );
        }
    }

    #[test]
    fn ffi_distance_accepts_non_null_terminated_buffers() {
        let source = b"kitten";
        let target = b"sitting";

        unsafe {
            assert_eq!(
                llev_distance(
                    source.as_ptr().cast(),
                    source.len(),
                    target.as_ptr().cast(),
                    target.len()
                ),
                3
            );
            assert_eq!(
                llev_distance_threshold(
                    source.as_ptr().cast(),
                    source.len(),
                    target.as_ptr().cast(),
                    target.len(),
                    2
                ),
                usize::MAX - 1
            );
            assert_eq!(
                llev_damerau_distance(source.as_ptr().cast(), 2, b"ik".as_ptr().cast(), 2),
                1
            );
            assert_eq!(
                llev_damerau_distance_threshold(
                    source.as_ptr().cast(),
                    2,
                    b"ik".as_ptr().cast(),
                    2,
                    1
                ),
                1
            );
        }
    }

    #[test]
    fn ffi_thresholds_preserve_unicode_character_semantics() {
        let source = "café";
        let target = "cafe";

        unsafe {
            assert_eq!(
                llev_distance_threshold(
                    source.as_ptr().cast(),
                    source.len(),
                    target.as_ptr().cast(),
                    target.len(),
                    1
                ),
                1
            );
            assert_eq!(
                llev_distance_threshold(
                    source.as_ptr().cast(),
                    source.len(),
                    target.as_ptr().cast(),
                    target.len(),
                    0
                ),
                usize::MAX - 1
            );
            assert_eq!(
                llev_damerau_distance_threshold(
                    "préabΩ".as_ptr().cast(),
                    "préabΩ".len(),
                    "prébaΩ".as_ptr().cast(),
                    "prébaΩ".len(),
                    1
                ),
                1
            );
        }
    }
}
