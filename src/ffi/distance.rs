//! Distance function FFI bindings.

use std::ffi::c_char;

use crate::distance::{
    damerau_levenshtein_distance, damerau_levenshtein_distance_bounded,
    damerau_levenshtein_distance_units, damerau_levenshtein_distance_units_bounded,
    merge_and_split_distance_bounded, merge_and_split_distance_units,
    merge_and_split_distance_units_bounded, myers::myers_distance_bytes,
    myers::myers_distance_bytes_bounded, standard_distance, standard_distance_bounded,
    standard_distance_units, standard_distance_units_bounded, transposition_distance,
    transposition_distance_bounded, transposition_distance_units,
    transposition_distance_units_bounded,
};

const INVALID_INPUT: usize = usize::MAX;
const ABOVE_THRESHOLD: usize = usize::MAX - 1;

#[inline]
unsafe fn input_slice<'a, U>(data: *const U, len: usize) -> Option<&'a [U]> {
    if len == 0 {
        return Some(&[]);
    }
    if data.is_null() || !(data as usize).is_multiple_of(std::mem::align_of::<U>()) {
        return None;
    }
    Some(std::slice::from_raw_parts(data, len))
}

macro_rules! raw_unit_distance_functions {
    (
        $exact_name:ident,
        $bounded_name:ident,
        $unit:ty,
        $domain:literal,
        $family:literal,
        $exact:path,
        $bounded:path
    ) => {
        #[doc = concat!("Compute exact ", $family, " distance over ", $domain, ".")]
        ///
        /// # Safety
        ///
        /// Each non-empty input must address its declared number of aligned
        /// units. A zero-length input may use a null pointer. Invalid pointers
        /// or alignment return `usize::MAX`.
        #[no_mangle]
        pub unsafe extern "C" fn $exact_name(
            source: *const $unit,
            source_len: usize,
            target: *const $unit,
            target_len: usize,
        ) -> usize {
            let Some(source) = input_slice(source, source_len) else {
                return INVALID_INPUT;
            };
            let Some(target) = input_slice(target, target_len) else {
                return INVALID_INPUT;
            };
            $exact(source, target)
        }

        #[doc = concat!(
                                                    "Compute thresholded ",
                                                    $family,
                                                    " distance over ",
                                                    $domain,
                                                    "."
                                                )]
        ///
        /// # Safety
        ///
        /// Each non-empty input must address its declared number of aligned
        /// units. A zero-length input may use a null pointer. Invalid pointers
        /// or alignment return `usize::MAX`; a result above `threshold` returns
        /// `usize::MAX - 1`.
        #[no_mangle]
        pub unsafe extern "C" fn $bounded_name(
            source: *const $unit,
            source_len: usize,
            target: *const $unit,
            target_len: usize,
            threshold: usize,
        ) -> usize {
            let Some(source) = input_slice(source, source_len) else {
                return INVALID_INPUT;
            };
            let Some(target) = input_slice(target, target_len) else {
                return INVALID_INPUT;
            };
            $bounded(source, target, threshold).unwrap_or(ABOVE_THRESHOLD)
        }
    };
}

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

/// Calculate Unicode-scalar merge-and-split distance between two strings.
///
/// # Safety
///
/// - Both buffers must be non-null and valid UTF-8 for their supplied lengths.
/// - A zero-length buffer may use a null pointer.
/// - Returns `usize::MAX` for a null or invalid UTF-8 buffer.
#[no_mangle]
pub unsafe extern "C" fn llev_merge_and_split_distance(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
) -> usize {
    let source = match super::cbuf_to_str(source, source_len) {
        Some(source) => source,
        None => return INVALID_INPUT,
    };
    let target = match super::cbuf_to_str(target, target_len) {
        Some(target) => target,
        None => return INVALID_INPUT,
    };
    let source: smallvec::SmallVec<[char; 32]> = source.chars().collect();
    let target: smallvec::SmallVec<[char; 32]> = target.chars().collect();
    merge_and_split_distance_units(&source, &target)
}

/// Calculate Unicode-scalar merge-and-split distance within a threshold.
///
/// # Safety
///
/// - Both buffers must be non-null and valid UTF-8 for their supplied lengths.
/// - A zero-length buffer may use a null pointer.
/// - Returns `usize::MAX` for invalid input and `usize::MAX - 1` when the exact
///   distance exceeds `threshold`.
#[no_mangle]
pub unsafe extern "C" fn llev_merge_and_split_distance_threshold(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
    threshold: usize,
) -> usize {
    let source = match super::cbuf_to_str(source, source_len) {
        Some(source) => source,
        None => return INVALID_INPUT,
    };
    let target = match super::cbuf_to_str(target, target_len) {
        Some(target) => target,
        None => return INVALID_INPUT,
    };
    merge_and_split_distance_bounded(source, target, threshold).unwrap_or(ABOVE_THRESHOLD)
}

raw_unit_distance_functions!(
    llev_distance_bytes,
    llev_distance_bytes_threshold,
    u8,
    "arbitrary bytes",
    "Levenshtein",
    myers_distance_bytes,
    myers_distance_bytes_bounded
);
raw_unit_distance_functions!(
    llev_distance_u64,
    llev_distance_u64_threshold,
    u64,
    "unsigned 64-bit tokens",
    "Levenshtein",
    standard_distance_units,
    standard_distance_units_bounded
);
raw_unit_distance_functions!(
    llev_damerau_distance_bytes,
    llev_damerau_distance_bytes_threshold,
    u8,
    "arbitrary bytes",
    "optimal-string-alignment",
    transposition_distance_units,
    transposition_distance_units_bounded
);
raw_unit_distance_functions!(
    llev_damerau_distance_u64,
    llev_damerau_distance_u64_threshold,
    u64,
    "unsigned 64-bit tokens",
    "optimal-string-alignment",
    transposition_distance_units,
    transposition_distance_units_bounded
);
raw_unit_distance_functions!(
    llev_true_damerau_distance_bytes,
    llev_true_damerau_distance_bytes_threshold,
    u8,
    "arbitrary bytes",
    "unrestricted Damerau--Levenshtein",
    damerau_levenshtein_distance_units,
    damerau_levenshtein_distance_units_bounded
);
raw_unit_distance_functions!(
    llev_true_damerau_distance_u64,
    llev_true_damerau_distance_u64_threshold,
    u64,
    "unsigned 64-bit tokens",
    "unrestricted Damerau--Levenshtein",
    damerau_levenshtein_distance_units,
    damerau_levenshtein_distance_units_bounded
);
raw_unit_distance_functions!(
    llev_merge_and_split_distance_bytes,
    llev_merge_and_split_distance_bytes_threshold,
    u8,
    "arbitrary bytes",
    "merge-and-split",
    merge_and_split_distance_units,
    merge_and_split_distance_units_bounded
);
raw_unit_distance_functions!(
    llev_merge_and_split_distance_u64,
    llev_merge_and_split_distance_u64_threshold,
    u64,
    "unsigned 64-bit tokens",
    "merge-and-split",
    merge_and_split_distance_units,
    merge_and_split_distance_units_bounded
);

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

    fn generated_sequences<U: Copy + Eq>(alphabet: &[U], maximum_len: usize) -> Vec<Vec<U>> {
        let mut sequences = vec![Vec::new()];
        for _ in 0..maximum_len {
            let previous = sequences.clone();
            for prefix in previous {
                for unit in alphabet {
                    let mut sequence = prefix.clone();
                    sequence.push(*unit);
                    sequences.push(sequence);
                }
            }
        }
        sequences.sort_by_key(Vec::len);
        sequences.dedup();
        sequences
    }

    #[test]
    fn raw_domain_ffi_matches_generic_native_kernels() {
        let byte_sequences = generated_sequences(&[0_u8, 0x7f, 0xff], 3);
        for source in &byte_sequences {
            for target in &byte_sequences {
                let source_ptr = source.as_ptr();
                let target_ptr = target.as_ptr();
                unsafe {
                    assert_eq!(
                        llev_distance_bytes(source_ptr, source.len(), target_ptr, target.len()),
                        standard_distance_units(source, target)
                    );
                    assert_eq!(
                        llev_damerau_distance_bytes(
                            source_ptr,
                            source.len(),
                            target_ptr,
                            target.len()
                        ),
                        transposition_distance_units(source, target)
                    );
                    assert_eq!(
                        llev_true_damerau_distance_bytes(
                            source_ptr,
                            source.len(),
                            target_ptr,
                            target.len()
                        ),
                        damerau_levenshtein_distance_units(source, target)
                    );
                    assert_eq!(
                        llev_merge_and_split_distance_bytes(
                            source_ptr,
                            source.len(),
                            target_ptr,
                            target.len()
                        ),
                        merge_and_split_distance_units(source, target)
                    );
                    for threshold in 0..=3 {
                        let sentinel = |value: Option<usize>| value.unwrap_or(ABOVE_THRESHOLD);
                        assert_eq!(
                            llev_distance_bytes_threshold(
                                source_ptr,
                                source.len(),
                                target_ptr,
                                target.len(),
                                threshold
                            ),
                            sentinel(standard_distance_units_bounded(source, target, threshold))
                        );
                        assert_eq!(
                            llev_damerau_distance_bytes_threshold(
                                source_ptr,
                                source.len(),
                                target_ptr,
                                target.len(),
                                threshold
                            ),
                            sentinel(transposition_distance_units_bounded(
                                source, target, threshold
                            ))
                        );
                        assert_eq!(
                            llev_true_damerau_distance_bytes_threshold(
                                source_ptr,
                                source.len(),
                                target_ptr,
                                target.len(),
                                threshold
                            ),
                            sentinel(damerau_levenshtein_distance_units_bounded(
                                source, target, threshold
                            ))
                        );
                        assert_eq!(
                            llev_merge_and_split_distance_bytes_threshold(
                                source_ptr,
                                source.len(),
                                target_ptr,
                                target.len(),
                                threshold
                            ),
                            sentinel(merge_and_split_distance_units_bounded(
                                source, target, threshold
                            ))
                        );
                    }
                }
            }
        }

        let token_sequences = generated_sequences(&[0_u64, 7, u64::MAX], 3);
        for source in &token_sequences {
            for target in &token_sequences {
                unsafe {
                    assert_eq!(
                        llev_distance_u64(
                            source.as_ptr(),
                            source.len(),
                            target.as_ptr(),
                            target.len()
                        ),
                        standard_distance_units(source, target)
                    );
                    assert_eq!(
                        llev_damerau_distance_u64(
                            source.as_ptr(),
                            source.len(),
                            target.as_ptr(),
                            target.len()
                        ),
                        transposition_distance_units(source, target)
                    );
                    assert_eq!(
                        llev_true_damerau_distance_u64(
                            source.as_ptr(),
                            source.len(),
                            target.as_ptr(),
                            target.len()
                        ),
                        damerau_levenshtein_distance_units(source, target)
                    );
                    assert_eq!(
                        llev_merge_and_split_distance_u64(
                            source.as_ptr(),
                            source.len(),
                            target.as_ptr(),
                            target.len()
                        ),
                        merge_and_split_distance_units(source, target)
                    );
                }
            }
        }
    }

    #[test]
    fn raw_domains_preserve_binary_values_and_validate_alignment() {
        unsafe {
            assert_eq!(
                llev_distance_bytes([0xff].as_ptr(), 1, [0x00].as_ptr(), 1),
                1
            );
            assert_eq!(
                llev_distance([0xff].as_ptr().cast(), 1, [0x00].as_ptr().cast(), 1),
                INVALID_INPUT
            );
            assert_eq!(
                llev_distance_u64(std::ptr::null(), 0, std::ptr::null(), 0),
                0
            );

            let storage = [0_u64; 2];
            let misaligned = storage.as_ptr().cast::<u8>().add(1).cast::<u64>();
            assert_eq!(
                llev_distance_u64(misaligned, 1, std::ptr::null(), 0),
                INVALID_INPUT
            );
        }
    }

    #[test]
    fn unicode_merge_split_is_exact_and_thresholded() {
        unsafe {
            assert_eq!(
                llev_merge_and_split_distance("m".as_ptr().cast(), 1, "rn".as_ptr().cast(), 2),
                1
            );
            assert_eq!(
                llev_merge_and_split_distance_threshold(
                    "prézΩ".as_ptr().cast(),
                    "prézΩ".len(),
                    "préxyΩ".as_ptr().cast(),
                    "préxyΩ".len(),
                    1
                ),
                1
            );
            assert_eq!(
                llev_merge_and_split_distance_threshold(
                    "a".as_ptr().cast(),
                    1,
                    "abcd".as_ptr().cast(),
                    4,
                    1
                ),
                ABOVE_THRESHOLD
            );
        }
    }
}
