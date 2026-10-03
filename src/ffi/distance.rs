//! Distance function FFI bindings.

use std::ffi::c_char;

use crate::cost::CostScale;
use crate::distance::{
    affine_gap_distance, affine_gap_distance_units, damerau_levenshtein_distance,
    damerau_levenshtein_distance_bounded, damerau_levenshtein_distance_units,
    damerau_levenshtein_distance_units_bounded, hamming_distance, hamming_distance_units,
    indel_distance, indel_distance_bounded, indel_distance_units, indel_distance_units_bounded,
    merge_and_split_distance_bounded, merge_and_split_distance_units,
    merge_and_split_distance_units_bounded, myers::myers_distance_bytes,
    myers::myers_distance_bytes_bounded, standard_distance, standard_distance_bounded,
    standard_distance_units, standard_distance_units_bounded, transposition_distance,
    transposition_distance_bounded, transposition_distance_units,
    transposition_distance_units_bounded,
};
use crate::transducer::AffineGapParams;

const INVALID_INPUT: usize = usize::MAX;
const ABOVE_THRESHOLD: usize = usize::MAX - 1;
const UNDEFINED_DISTANCE: usize = usize::MAX - 2;

#[inline]
fn affine_params(gap_open: usize, gap_extend: usize, substitution: usize) -> AffineGapParams {
    AffineGapParams::from_scaled(
        CostScale::new(1).expect("unit scale has a nonzero denominator"),
        gap_open,
        gap_extend,
        substitution,
    )
}

#[inline]
fn encode_affine_result(distance: Option<usize>) -> usize {
    distance
        .filter(|value| *value < UNDEFINED_DISTANCE)
        .unwrap_or(UNDEFINED_DISTANCE)
}

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

#[inline]
unsafe fn utf8_pair<'a>(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
) -> Option<(&'a str, &'a str)> {
    let source = super::cbuf_to_str(source, source_len)?;
    let target = super::cbuf_to_str(target, target_len)?;
    Some((source, target))
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

        #[doc = concat!("Compute thresholded ", $family, " distance over ", $domain, ".")]
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

macro_rules! raw_optional_unit_distance_functions {
    ($exact_name:ident, $bounded_name:ident, $unit:ty, $exact:path) => {
        /// Compute Hamming distance; unequal lengths have no defined result.
        ///
        /// # Safety
        ///
        /// Non-empty inputs must point to their declared number of aligned
        /// units. Null pointers are valid only for zero-length inputs.
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
            $exact(source, target).unwrap_or(UNDEFINED_DISTANCE)
        }

        /// Compute thresholded Hamming distance.
        ///
        /// # Safety
        ///
        /// Pointer and length requirements match the exact variant. The
        /// undefined-length sentinel remains distinct from above-threshold.
        #[no_mangle]
        pub unsafe extern "C" fn $bounded_name(
            source: *const $unit,
            source_len: usize,
            target: *const $unit,
            target_len: usize,
            threshold: usize,
        ) -> usize {
            match $exact_name(source, source_len, target, target_len) {
                INVALID_INPUT => INVALID_INPUT,
                UNDEFINED_DISTANCE => UNDEFINED_DISTANCE,
                distance if distance > threshold => ABOVE_THRESHOLD,
                distance => distance,
            }
        }
    };
}

macro_rules! raw_affine_gap_distance_functions {
    ($exact_name:ident, $bounded_name:ident, $unit:ty) => {
        /// Compute exact affine-gap distance over native units.
        ///
        /// # Safety
        ///
        /// Non-empty inputs must point to their declared number of aligned
        /// units. Null pointers are valid only for zero-length inputs.
        #[no_mangle]
        pub unsafe extern "C" fn $exact_name(
            source: *const $unit,
            source_len: usize,
            target: *const $unit,
            target_len: usize,
            gap_open: usize,
            gap_extend: usize,
            substitution: usize,
        ) -> usize {
            let Some(source) = input_slice(source, source_len) else {
                return INVALID_INPUT;
            };
            let Some(target) = input_slice(target, target_len) else {
                return INVALID_INPUT;
            };
            encode_affine_result(affine_gap_distance_units(
                source,
                target,
                affine_params(gap_open, gap_extend, substitution),
            ))
        }

        /// Compute thresholded affine-gap distance over native units.
        ///
        /// # Safety
        ///
        /// Pointer and length requirements match the exact variant.
        #[no_mangle]
        pub unsafe extern "C" fn $bounded_name(
            source: *const $unit,
            source_len: usize,
            target: *const $unit,
            target_len: usize,
            gap_open: usize,
            gap_extend: usize,
            substitution: usize,
            threshold: usize,
        ) -> usize {
            match $exact_name(
                source,
                source_len,
                target,
                target_len,
                gap_open,
                gap_extend,
                substitution,
            ) {
                INVALID_INPUT => INVALID_INPUT,
                UNDEFINED_DISTANCE => UNDEFINED_DISTANCE,
                distance if distance > threshold => ABOVE_THRESHOLD,
                distance => distance,
            }
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

/// Count mismatched Unicode scalar positions for equal-length strings.
///
/// # Safety
///
/// Non-empty inputs must point to valid UTF-8 for their declared byte lengths.
/// Invalid input returns `SIZE_MAX`; unequal scalar counts return
/// `SIZE_MAX - 2`.
#[no_mangle]
pub unsafe extern "C" fn llev_hamming_distance(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
) -> usize {
    let Some((source, target)) = utf8_pair(source, source_len, target, target_len) else {
        return INVALID_INPUT;
    };
    hamming_distance(source, target).unwrap_or(UNDEFINED_DISTANCE)
}

/// Count mismatched Unicode scalar positions within a threshold.
///
/// # Safety
///
/// Input requirements and unequal-length sentinel match the exact variant.
/// A defined result above the threshold returns `SIZE_MAX - 1`.
#[no_mangle]
pub unsafe extern "C" fn llev_hamming_distance_threshold(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
    threshold: usize,
) -> usize {
    match llev_hamming_distance(source, source_len, target, target_len) {
        INVALID_INPUT => INVALID_INPUT,
        UNDEFINED_DISTANCE => UNDEFINED_DISTANCE,
        distance if distance > threshold => ABOVE_THRESHOLD,
        distance => distance,
    }
}

/// Compute exact insertion/deletion distance over Unicode scalars.
///
/// # Safety
///
/// Non-empty inputs must point to valid UTF-8 for their declared byte lengths.
#[no_mangle]
pub unsafe extern "C" fn llev_indel_distance(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
) -> usize {
    let Some((source, target)) = utf8_pair(source, source_len, target, target_len) else {
        return INVALID_INPUT;
    };
    indel_distance(source, target)
}

/// Compute bounded insertion/deletion distance over Unicode scalars.
///
/// # Safety
///
/// Input requirements match the exact variant. Above-bound results return
/// `SIZE_MAX - 1`.
#[no_mangle]
pub unsafe extern "C" fn llev_indel_distance_threshold(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
    threshold: usize,
) -> usize {
    let Some((source, target)) = utf8_pair(source, source_len, target, target_len) else {
        return INVALID_INPUT;
    };
    indel_distance_bounded(source, target, threshold).unwrap_or(ABOVE_THRESHOLD)
}

/// Compute exact affine-gap distance with integer scaled costs.
///
/// A gap of length `k` costs `gap_open + k * gap_extend`. The result uses
/// the same integer scale as all three supplied costs. Arithmetic overflow or
/// an unrepresentable sentinel-range result returns `SIZE_MAX - 2`.
///
/// # Safety
///
/// Non-empty inputs must point to valid UTF-8 for their declared byte lengths.
#[no_mangle]
pub unsafe extern "C" fn llev_affine_gap_distance(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
    gap_open: usize,
    gap_extend: usize,
    substitution: usize,
) -> usize {
    let Some((source, target)) = utf8_pair(source, source_len, target, target_len) else {
        return INVALID_INPUT;
    };
    encode_affine_result(affine_gap_distance(
        source,
        target,
        affine_params(gap_open, gap_extend, substitution),
    ))
}

/// Compute thresholded affine-gap distance with integer scaled costs.
///
/// # Safety
///
/// Input requirements match the exact variant. Above-bound results return
/// `SIZE_MAX - 1`; overflow remains `SIZE_MAX - 2`.
#[no_mangle]
pub unsafe extern "C" fn llev_affine_gap_distance_threshold(
    source: *const c_char,
    source_len: usize,
    target: *const c_char,
    target_len: usize,
    gap_open: usize,
    gap_extend: usize,
    substitution: usize,
    threshold: usize,
) -> usize {
    match llev_affine_gap_distance(
        source,
        source_len,
        target,
        target_len,
        gap_open,
        gap_extend,
        substitution,
    ) {
        INVALID_INPUT => INVALID_INPUT,
        UNDEFINED_DISTANCE => UNDEFINED_DISTANCE,
        distance if distance > threshold => ABOVE_THRESHOLD,
        distance => distance,
    }
}

raw_optional_unit_distance_functions!(
    llev_hamming_distance_bytes,
    llev_hamming_distance_bytes_threshold,
    u8,
    hamming_distance_units
);
raw_optional_unit_distance_functions!(
    llev_hamming_distance_u64,
    llev_hamming_distance_u64_threshold,
    u64,
    hamming_distance_units
);
raw_unit_distance_functions!(
    llev_indel_distance_bytes,
    llev_indel_distance_bytes_threshold,
    u8,
    "arbitrary bytes",
    "insertion/deletion",
    indel_distance_units,
    indel_distance_units_bounded
);
raw_unit_distance_functions!(
    llev_indel_distance_u64,
    llev_indel_distance_u64_threshold,
    u64,
    "unsigned 64-bit tokens",
    "insertion/deletion",
    indel_distance_units,
    indel_distance_units_bounded
);
raw_affine_gap_distance_functions!(
    llev_affine_gap_distance_bytes,
    llev_affine_gap_distance_bytes_threshold,
    u8
);
raw_affine_gap_distance_functions!(
    llev_affine_gap_distance_u64,
    llev_affine_gap_distance_u64_threshold,
    u64
);

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
    fn new_distance_families_match_rust_oracles_across_raw_domains() {
        let costs = affine_params(2, 1, 3);
        let expected_hamming = |source: &[u8], target: &[u8]| {
            hamming_distance_units(source, target).unwrap_or(UNDEFINED_DISTANCE)
        };
        for source in generated_sequences(&[0_u8, 0xff], 3) {
            for target in generated_sequences(&[0_u8, 0xff], 3) {
                let hamming = expected_hamming(&source, &target);
                let indel = indel_distance_units(&source, &target);
                let affine =
                    encode_affine_result(affine_gap_distance_units(&source, &target, costs));
                unsafe {
                    assert_eq!(
                        llev_hamming_distance_bytes(
                            source.as_ptr(),
                            source.len(),
                            target.as_ptr(),
                            target.len()
                        ),
                        hamming
                    );
                    assert_eq!(
                        llev_indel_distance_bytes(
                            source.as_ptr(),
                            source.len(),
                            target.as_ptr(),
                            target.len()
                        ),
                        indel
                    );
                    assert_eq!(
                        llev_affine_gap_distance_bytes(
                            source.as_ptr(),
                            source.len(),
                            target.as_ptr(),
                            target.len(),
                            2,
                            1,
                            3
                        ),
                        affine
                    );
                    for threshold in 0..=6 {
                        assert_eq!(
                            llev_hamming_distance_bytes_threshold(
                                source.as_ptr(),
                                source.len(),
                                target.as_ptr(),
                                target.len(),
                                threshold
                            ),
                            if hamming == UNDEFINED_DISTANCE {
                                hamming
                            } else if hamming > threshold {
                                ABOVE_THRESHOLD
                            } else {
                                hamming
                            }
                        );
                        assert_eq!(
                            llev_indel_distance_bytes_threshold(
                                source.as_ptr(),
                                source.len(),
                                target.as_ptr(),
                                target.len(),
                                threshold
                            ),
                            indel_distance_units_bounded(&source, &target, threshold)
                                .unwrap_or(ABOVE_THRESHOLD)
                        );
                        assert_eq!(
                            llev_affine_gap_distance_bytes_threshold(
                                source.as_ptr(),
                                source.len(),
                                target.as_ptr(),
                                target.len(),
                                2,
                                1,
                                3,
                                threshold
                            ),
                            if affine == UNDEFINED_DISTANCE {
                                affine
                            } else if affine > threshold {
                                ABOVE_THRESHOLD
                            } else {
                                affine
                            }
                        );
                    }
                }
            }
        }

        for source in generated_sequences(&[0_u64, u64::MAX], 3) {
            for target in generated_sequences(&[0_u64, u64::MAX], 3) {
                let hamming =
                    hamming_distance_units(&source, &target).unwrap_or(UNDEFINED_DISTANCE);
                let indel = indel_distance_units(&source, &target);
                let affine =
                    encode_affine_result(affine_gap_distance_units(&source, &target, costs));
                unsafe {
                    assert_eq!(
                        llev_hamming_distance_u64(
                            source.as_ptr(),
                            source.len(),
                            target.as_ptr(),
                            target.len()
                        ),
                        hamming
                    );
                    assert_eq!(
                        llev_indel_distance_u64(
                            source.as_ptr(),
                            source.len(),
                            target.as_ptr(),
                            target.len()
                        ),
                        indel
                    );
                    assert_eq!(
                        llev_affine_gap_distance_u64(
                            source.as_ptr(),
                            source.len(),
                            target.as_ptr(),
                            target.len(),
                            2,
                            1,
                            3
                        ),
                        affine
                    );
                    for threshold in 0..=6 {
                        assert_eq!(
                            llev_hamming_distance_u64_threshold(
                                source.as_ptr(),
                                source.len(),
                                target.as_ptr(),
                                target.len(),
                                threshold
                            ),
                            if hamming == UNDEFINED_DISTANCE {
                                hamming
                            } else if hamming > threshold {
                                ABOVE_THRESHOLD
                            } else {
                                hamming
                            }
                        );
                        assert_eq!(
                            llev_indel_distance_u64_threshold(
                                source.as_ptr(),
                                source.len(),
                                target.as_ptr(),
                                target.len(),
                                threshold
                            ),
                            indel_distance_units_bounded(&source, &target, threshold)
                                .unwrap_or(ABOVE_THRESHOLD)
                        );
                        assert_eq!(
                            llev_affine_gap_distance_u64_threshold(
                                source.as_ptr(),
                                source.len(),
                                target.as_ptr(),
                                target.len(),
                                2,
                                1,
                                3,
                                threshold
                            ),
                            if affine == UNDEFINED_DISTANCE {
                                affine
                            } else if affine > threshold {
                                ABOVE_THRESHOLD
                            } else {
                                affine
                            }
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn new_utf8_families_preserve_scalar_semantics_and_distinct_absences() {
        let source = "pré";
        let target = "prè";
        let source_ptr = source.as_ptr().cast();
        let target_ptr = target.as_ptr().cast();
        unsafe {
            assert_eq!(
                llev_hamming_distance(source_ptr, source.len(), target_ptr, target.len()),
                hamming_distance(source, target).unwrap()
            );
            assert_eq!(
                llev_indel_distance(source_ptr, source.len(), target_ptr, target.len()),
                indel_distance(source, target)
            );
            assert_eq!(
                llev_affine_gap_distance(
                    source_ptr,
                    source.len(),
                    target_ptr,
                    target.len(),
                    2,
                    1,
                    3
                ),
                affine_gap_distance(source, target, affine_params(2, 1, 3)).unwrap()
            );
            assert_eq!(
                llev_hamming_distance(source_ptr, source.len(), "pr".as_ptr().cast(), 2),
                UNDEFINED_DISTANCE
            );
            assert_eq!(
                llev_hamming_distance_threshold(
                    source_ptr,
                    source.len(),
                    "pr".as_ptr().cast(),
                    2,
                    0
                ),
                UNDEFINED_DISTANCE
            );
            assert_eq!(
                llev_hamming_distance_threshold(
                    source_ptr,
                    source.len(),
                    target_ptr,
                    target.len(),
                    0
                ),
                ABOVE_THRESHOLD
            );
            assert_eq!(
                llev_indel_distance_threshold(
                    source_ptr,
                    source.len(),
                    target_ptr,
                    target.len(),
                    0
                ),
                ABOVE_THRESHOLD
            );
            assert_eq!(
                llev_affine_gap_distance_threshold(
                    source_ptr,
                    source.len(),
                    target_ptr,
                    target.len(),
                    2,
                    1,
                    3,
                    0
                ),
                ABOVE_THRESHOLD
            );
            assert_eq!(
                llev_affine_gap_distance(
                    "a".as_ptr().cast(),
                    1,
                    "b".as_ptr().cast(),
                    1,
                    usize::MAX,
                    usize::MAX,
                    usize::MAX
                ),
                UNDEFINED_DISTANCE
            );
            assert_eq!(
                llev_hamming_distance(source_ptr, source.len() - 1, target_ptr, target.len()),
                INVALID_INPUT
            );
            assert_eq!(
                llev_indel_distance(source_ptr, source.len() - 1, target_ptr, target.len()),
                INVALID_INPUT
            );
            assert_eq!(
                llev_affine_gap_distance(
                    source_ptr,
                    source.len() - 1,
                    target_ptr,
                    target.len(),
                    2,
                    1,
                    3
                ),
                INVALID_INPUT
            );
            assert_eq!(
                llev_hamming_distance(std::ptr::null(), 0, std::ptr::null(), 0),
                0
            );
            assert_eq!(
                llev_indel_distance(std::ptr::null(), 0, std::ptr::null(), 0),
                0
            );
            assert_eq!(
                llev_affine_gap_distance(std::ptr::null(), 0, std::ptr::null(), 0, 2, 1, 3),
                0
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
