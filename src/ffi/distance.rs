//! Distance function FFI bindings.

use std::ffi::c_char;

use crate::distance::{
    damerau_levenshtein_distance, damerau_levenshtein_distance_bounded, standard_distance,
    standard_distance_bounded, transposition_distance, transposition_distance_bounded,
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

/// Calculate optimal string alignment distance between two strings.
///
/// This legacy C symbol includes adjacent transposition as one operation but
/// computes restricted Damerau (OSA), not unrestricted Damerau–Levenshtein.
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

/// Calculate optimal string alignment distance, returning early if it exceeds threshold.
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

/// Calculate unrestricted Damerau–Levenshtein distance between two strings.
///
/// # Safety
///
/// - Both buffers must be non-null and valid UTF-8 for their supplied lengths.
/// - Returns `usize::MAX` for a null or invalid UTF-8 buffer.
#[no_mangle]
pub unsafe extern "C" fn llev_true_damerau_distance(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
) -> usize {
    let source = match super::cbuf_to_str(source, source_len) {
        Some(source) => source,
        None => return usize::MAX,
    };
    let target = match super::cbuf_to_str(target, target_len) {
        Some(target) => target,
        None => return usize::MAX,
    };
    damerau_levenshtein_distance(source, target)
}

/// Calculate unrestricted Damerau–Levenshtein distance within a threshold.
///
/// # Safety
///
/// - Both buffers must be non-null and valid UTF-8 for their supplied lengths.
/// - Returns `usize::MAX` for invalid input and `usize::MAX - 1` when the exact
///   distance exceeds `threshold`.
#[no_mangle]
pub unsafe extern "C" fn llev_true_damerau_distance_threshold(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
    threshold: usize,
) -> usize {
    let source = match super::cbuf_to_str(source, source_len) {
        Some(source) => source,
        None => return usize::MAX,
    };
    let target = match super::cbuf_to_str(target, target_len) {
        Some(target) => target,
        None => return usize::MAX,
    };
    damerau_levenshtein_distance_bounded(source, target, threshold).unwrap_or(usize::MAX - 1)
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

    // Regression for the empty-operand defect (LLEV-B18): a caller may pass a
    // null data pointer for an empty string — several host runtimes materialize
    // an empty slice as a null pointer — and the distance functions must treat
    // `(NULL, 0)` as the empty string rather than returning the invalid-input
    // sentinel `usize::MAX`. This matches the transducer query path, which has
    // always accepted `(NULL, 0)` as an empty query.
    #[test]
    fn ffi_distance_treats_null_zero_length_as_empty_string() {
        let ab = b"ab";
        let nul: *const c_char = std::ptr::null();

        unsafe {
            // Empty to empty is distance 0 across every distance variant.
            assert_eq!(llev_distance(nul, 0, nul, 0), 0);
            assert_eq!(llev_distance_threshold(nul, 0, nul, 0, 0), 0);
            assert_eq!(llev_damerau_distance(nul, 0, nul, 0), 0);
            assert_eq!(llev_damerau_distance_threshold(nul, 0, nul, 0, 0), 0);
            assert_eq!(llev_true_damerau_distance(nul, 0, nul, 0), 0);
            assert_eq!(llev_true_damerau_distance_threshold(nul, 0, nul, 0, 0), 0);

            // A null empty operand pairs correctly with a non-empty operand,
            // symmetrically, at cost equal to the non-empty length.
            assert_eq!(llev_distance(nul, 0, ab.as_ptr().cast(), 2), 2);
            assert_eq!(llev_distance(ab.as_ptr().cast(), 2, nul, 0), 2);
            assert_eq!(llev_damerau_distance(nul, 0, ab.as_ptr().cast(), 2), 2);
            assert_eq!(llev_true_damerau_distance(ab.as_ptr().cast(), 2, nul, 0), 2);

            // A non-empty operand fitting under the threshold still reports its
            // true distance; over the bound still yields the threshold sentinel.
            assert_eq!(llev_distance_threshold(nul, 0, ab.as_ptr().cast(), 2, 2), 2);
            assert_eq!(
                llev_distance_threshold(nul, 0, ab.as_ptr().cast(), 2, 1),
                usize::MAX - 1
            );

            // A null pointer with a NON-zero length is still an invalid operand
            // (it cannot denote any bytes) and must keep returning the
            // invalid-input sentinel — the fix must not weaken this guard.
            assert_eq!(llev_distance(nul, 2, ab.as_ptr().cast(), 2), usize::MAX);
            assert_eq!(llev_distance(ab.as_ptr().cast(), 2, nul, 2), usize::MAX);
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

    #[test]
    fn ffi_true_damerau_symbol_separates_from_legacy_osa_symbol() {
        let source = b"CA";
        let target = b"ABC";

        unsafe {
            assert_eq!(
                llev_damerau_distance(
                    source.as_ptr().cast(),
                    source.len(),
                    target.as_ptr().cast(),
                    target.len(),
                ),
                3
            );
            assert_eq!(
                llev_true_damerau_distance(
                    source.as_ptr().cast(),
                    source.len(),
                    target.as_ptr().cast(),
                    target.len(),
                ),
                2
            );
            assert_eq!(
                llev_true_damerau_distance_threshold(
                    source.as_ptr().cast(),
                    source.len(),
                    target.as_ptr().cast(),
                    target.len(),
                    1,
                ),
                usize::MAX - 1
            );
        }
    }
}
