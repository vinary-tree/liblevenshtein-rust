//! SIMD-accelerated transducer operations
//!
//! This module provides vectorized implementations of hot-path transducer
//! operations using AVX2 and SSE4.1 intrinsics for significant performance
//! improvements.
//!
//! The implementation uses runtime CPU feature detection to automatically select
//! the fastest available implementation (AVX2 > SSE4.1 > scalar fallback).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute characteristic vector using SIMD acceleration
///
/// This function compares a dictionary character against up to 8 query characters
/// in parallel, achieving 3-4x speedup over scalar comparison.
///
/// # Arguments
/// * `dict_char` - Character from the dictionary edge
/// * `query` - Query term bytes
/// * `window_size` - Size of the window (typically max_distance + 1)
/// * `offset` - Base offset in query
/// * `buffer` - Stack-allocated buffer to write results into
///
/// # Returns
/// Slice of booleans indicating matches at each position in window.
#[cfg(target_arch = "x86_64")]
pub fn characteristic_vector_simd<'a>(
    dict_char: u8,
    query: &[u8],
    window_size: usize,
    offset: usize,
    buffer: &'a mut [bool; 8],
) -> &'a [bool] {
    let len = window_size.min(8);

    // Use SIMD for window_size >= 4 (worth the overhead)
    if len >= 8 && is_x86_feature_detected!("avx2") {
        unsafe { characteristic_vector_avx2(dict_char, query, len, offset, buffer) }
    } else if len >= 4 && is_x86_feature_detected!("sse4.1") {
        unsafe { characteristic_vector_sse41(dict_char, query, len, offset, buffer) }
    } else {
        characteristic_vector_scalar(dict_char, query, len, offset, buffer)
    }
}

/// Scalar fallback for characteristic vector computation
#[inline(always)]
fn characteristic_vector_scalar<'a>(
    dict_char: u8,
    query: &[u8],
    len: usize,
    offset: usize,
    buffer: &'a mut [bool; 8],
) -> &'a [bool] {
    for (i, item) in buffer.iter_mut().enumerate().take(len) {
        let query_idx = offset + i;
        *item = query_idx < query.len() && query[query_idx] == dict_char;
    }
    &buffer[..len]
}

/// AVX2-accelerated characteristic vector computation
///
/// Processes 8 characters at once using 256-bit SIMD vectors
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn characteristic_vector_avx2<'a>(
    dict_char: u8,
    query: &[u8],
    len: usize,
    offset: usize,
    buffer: &'a mut [bool; 8],
) -> &'a [bool] {
    debug_assert!(len == 8, "AVX2 path expects exactly 8 elements");

    // Broadcast dictionary character to all 32 bytes (we'll use first 8)
    let dict_vec = _mm256_set1_epi8(dict_char as i8);

    // Load 8 query characters (with bounds checking)
    let mut query_buf = [0u8; 32]; // 256-bit = 32 bytes
    for (i, cell) in query_buf.iter_mut().enumerate().take(8) {
        let query_idx = offset + i;
        *cell = if query_idx < query.len() {
            query[query_idx]
        } else {
            0xFF // Out of bounds = use invalid byte that won't match
        };
    }

    // Load as 256-bit vector (only first 8 bytes matter)
    let query_vec = _mm256_loadu_si256(query_buf.as_ptr() as *const __m256i);

    // Compare all bytes at once
    let cmp_result = _mm256_cmpeq_epi8(dict_vec, query_vec);

    // Extract comparison results (first 8 bits are what we care about)
    let mask = _mm256_movemask_epi8(cmp_result);

    // Write results to buffer (first 8 bits of mask)
    for (i, cell) in buffer.iter_mut().enumerate().take(8) {
        *cell = (mask & (1 << i)) != 0;
    }

    &buffer[..len]
}

/// SSE4.1-accelerated characteristic vector computation
///
/// Processes 4 characters at once using 128-bit SIMD vectors
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn characteristic_vector_sse41<'a>(
    dict_char: u8,
    query: &[u8],
    len: usize,
    offset: usize,
    buffer: &'a mut [bool; 8],
) -> &'a [bool] {
    debug_assert!(len >= 4, "SSE4.1 path expects at least 4 elements");

    // Broadcast dictionary character to all 16 bytes
    let dict_vec = _mm_set1_epi8(dict_char as i8);

    // Process first 4 characters
    let mut query_buf = [0u8; 4];
    for (i, cell) in query_buf.iter_mut().enumerate().take(4) {
        let query_idx = offset + i;
        *cell = if query_idx < query.len() {
            query[query_idx]
        } else {
            0 // Out of bounds = no match
        };
    }

    // Load as 32-bit value
    let query_val = u32::from_le_bytes(query_buf);
    let query_vec = _mm_set1_epi32(query_val as i32);

    // Compare first 4 bytes
    let cmp_result = _mm_cmpeq_epi8(dict_vec, query_vec);
    let mask = _mm_movemask_epi8(cmp_result);

    // Write first 4 results
    for (i, cell) in buffer.iter_mut().enumerate().take(4) {
        *cell = (mask & (1 << i)) != 0;
    }

    // Process remaining elements with scalar
    for (i, cell) in buffer.iter_mut().enumerate().take(len).skip(4) {
        let query_idx = offset + i;
        *cell = query_idx < query.len() && query[query_idx] == dict_char;
    }

    &buffer[..len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_characteristic_vector_simd() {
        let test_cases = vec![
            // (dict_char, query, window_size, offset, expected)
            (b'a', b"aaaaaaa".as_slice(), 7, 0, vec![true; 7]),
            (
                b'a',
                b"abcdefg".as_slice(),
                7,
                0,
                vec![true, false, false, false, false, false, false],
            ),
            (b'x', b"abcdefg".as_slice(), 7, 0, vec![false; 7]),
            (
                b'a',
                b"abc".as_slice(),
                5,
                0,
                vec![true, false, false, false, false],
            ), // Out of bounds
            (
                b'c',
                b"abcdefg".as_slice(),
                5,
                2,
                vec![true, false, false, false, false],
            ), // Offset
            (
                b'a',
                b"aaa".as_slice(),
                8,
                0,
                vec![true, true, true, false, false, false, false, false],
            ), // Full 8
        ];

        for (dict_char, query, window_size, offset, expected) in test_cases {
            let mut buffer = [false; 8];
            let result =
                characteristic_vector_simd(dict_char, query, window_size, offset, &mut buffer);

            assert_eq!(
                result,
                &expected[..],
                "SIMD result mismatch for dict_char={}, query={:?}, window_size={}, offset={}",
                dict_char as char,
                query,
                window_size,
                offset
            );

            // Compare against scalar
            let mut scalar_buffer = [false; 8];
            let scalar_result = characteristic_vector_scalar(
                dict_char,
                query,
                window_size.min(8),
                offset,
                &mut scalar_buffer,
            );

            assert_eq!(
                result, scalar_result,
                "SIMD vs scalar mismatch for dict_char={}, query={:?}, window_size={}, offset={}",
                dict_char as char, query, window_size, offset
            );
        }
    }
}

/// SIMD-accelerated position subsumption checking for Standard algorithm
///
/// This function checks if positions in `lhs_positions` subsume corresponding
/// positions in `rhs_positions` using the Standard algorithm subsumption rule:
/// `|i - j| <= (f - e)` where (i,e) is lhs and (j,f) is rhs.
///
/// # Arguments
/// * `lhs_term_indices` - Array of term indices for left-hand positions
/// * `lhs_errors` - Array of error counts for left-hand positions
/// * `rhs_term_indices` - Array of term indices for right-hand positions
/// * `rhs_errors` - Array of error counts for right-hand positions
/// * `count` - Number of position pairs to check (must be <= 8)
/// * `results` - Output buffer for subsumption results
///
/// # Returns
/// Slice of booleans where `results[i]` indicates if `lhs[i]` subsumes `rhs[i]`
///
/// # Performance
/// - AVX2: Processes 8 pairs simultaneously (3-4x speedup)
/// - SSE4.1: Processes 4 pairs simultaneously (2-3x speedup)
/// - Scalar fallback for count < 4 or when SIMD unavailable
#[cfg(target_arch = "x86_64")]
pub fn check_subsumption_simd<'a>(
    lhs_term_indices: &[usize],
    lhs_errors: &[usize],
    rhs_term_indices: &[usize],
    rhs_errors: &[usize],
    count: usize,
    results: &'a mut [bool; 8],
) -> &'a [bool] {
    debug_assert!(count <= 8, "count must be <= 8");
    debug_assert!(lhs_term_indices.len() >= count);
    debug_assert!(lhs_errors.len() >= count);
    debug_assert!(rhs_term_indices.len() >= count);
    debug_assert!(rhs_errors.len() >= count);

    // Use SIMD for count >= 4 (worth the overhead)
    if count == 8 && is_x86_feature_detected!("avx2") {
        unsafe {
            check_subsumption_avx2(
                lhs_term_indices,
                lhs_errors,
                rhs_term_indices,
                rhs_errors,
                results,
            )
        }
    } else if count >= 4 && is_x86_feature_detected!("sse4.1") {
        unsafe {
            check_subsumption_sse41(
                lhs_term_indices,
                lhs_errors,
                rhs_term_indices,
                rhs_errors,
                count,
                results,
            )
        }
    } else {
        check_subsumption_scalar(
            lhs_term_indices,
            lhs_errors,
            rhs_term_indices,
            rhs_errors,
            count,
            results,
        )
    }
}

/// Scalar fallback for subsumption checking
#[inline(always)]
fn check_subsumption_scalar<'a>(
    lhs_term_indices: &[usize],
    lhs_errors: &[usize],
    rhs_term_indices: &[usize],
    rhs_errors: &[usize],
    count: usize,
    results: &'a mut [bool; 8],
) -> &'a [bool] {
    for idx in 0..count {
        let i = lhs_term_indices[idx];
        let e = lhs_errors[idx];
        let j = rhs_term_indices[idx];
        let f = rhs_errors[idx];

        // Must have fewer or equal errors to subsume
        if e > f {
            results[idx] = false;
            continue;
        }

        // Standard algorithm: |i - j| <= (f - e)
        let index_diff = i.abs_diff(j);
        let error_diff = f - e;
        results[idx] = index_diff <= error_diff;
    }
    &results[..count]
}

/// AVX2-accelerated subsumption checking
///
/// Processes 8 position pairs at once using 256-bit SIMD vectors.
/// Implements: `|i - j| <= (f - e)` for each pair simultaneously.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn check_subsumption_avx2<'a>(
    lhs_term_indices: &[usize],
    lhs_errors: &[usize],
    rhs_term_indices: &[usize],
    rhs_errors: &[usize],
    results: &'a mut [bool; 8],
) -> &'a [bool] {
    // Convert usize to u32 for SIMD processing (assume indices fit in u32)
    let mut lhs_i_buf = [0u32; 8];
    let mut lhs_e_buf = [0u32; 8];
    let mut rhs_j_buf = [0u32; 8];
    let mut rhs_f_buf = [0u32; 8];

    for idx in 0..8 {
        lhs_i_buf[idx] = lhs_term_indices[idx] as u32;
        lhs_e_buf[idx] = lhs_errors[idx] as u32;
        rhs_j_buf[idx] = rhs_term_indices[idx] as u32;
        rhs_f_buf[idx] = rhs_errors[idx] as u32;
    }

    // Load data into SIMD vectors
    let i_vec = _mm256_loadu_si256(lhs_i_buf.as_ptr() as *const __m256i);
    let e_vec = _mm256_loadu_si256(lhs_e_buf.as_ptr() as *const __m256i);
    let j_vec = _mm256_loadu_si256(rhs_j_buf.as_ptr() as *const __m256i);
    let f_vec = _mm256_loadu_si256(rhs_f_buf.as_ptr() as *const __m256i);

    // Check if e > f (early rejection: lhs cannot subsume if more errors)
    let e_gt_f = _mm256_cmpgt_epi32(e_vec, f_vec);

    // Compute |i - j| = max(i - j, j - i)
    let i_sub_j = _mm256_sub_epi32(i_vec, j_vec);
    let j_sub_i = _mm256_sub_epi32(j_vec, i_vec);
    let abs_diff = _mm256_max_epi32(i_sub_j, j_sub_i);

    // Compute (f - e)
    let error_diff = _mm256_sub_epi32(f_vec, e_vec);

    // Check if |i - j| <= (f - e)
    // Note: _mm256_cmple doesn't exist, so we use: !(|i-j| > (f-e))
    let abs_gt_error = _mm256_cmpgt_epi32(abs_diff, error_diff);
    let subsumes_mask = _mm256_andnot_si256(abs_gt_error, _mm256_set1_epi32(-1));

    // Combine with e <= f check (must both be true)
    let final_mask = _mm256_andnot_si256(e_gt_f, subsumes_mask);

    // Extract results as bitmask
    let mask = _mm256_movemask_ps(_mm256_castsi256_ps(final_mask));

    // Write boolean results
    for (idx, result) in results.iter_mut().enumerate().take(8) {
        *result = (mask & (1 << idx)) != 0;
    }

    &results[..8]
}

/// SSE4.1-accelerated subsumption checking
///
/// Processes 4 position pairs at once using 128-bit SIMD vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn check_subsumption_sse41<'a>(
    lhs_term_indices: &[usize],
    lhs_errors: &[usize],
    rhs_term_indices: &[usize],
    rhs_errors: &[usize],
    count: usize,
    results: &'a mut [bool; 8],
) -> &'a [bool] {
    debug_assert!((4..=8).contains(&count));

    // Process first 4 with SIMD
    let mut lhs_i_buf = [0u32; 4];
    let mut lhs_e_buf = [0u32; 4];
    let mut rhs_j_buf = [0u32; 4];
    let mut rhs_f_buf = [0u32; 4];

    for idx in 0..4 {
        lhs_i_buf[idx] = lhs_term_indices[idx] as u32;
        lhs_e_buf[idx] = lhs_errors[idx] as u32;
        rhs_j_buf[idx] = rhs_term_indices[idx] as u32;
        rhs_f_buf[idx] = rhs_errors[idx] as u32;
    }

    let i_vec = _mm_loadu_si128(lhs_i_buf.as_ptr() as *const __m128i);
    let e_vec = _mm_loadu_si128(lhs_e_buf.as_ptr() as *const __m128i);
    let j_vec = _mm_loadu_si128(rhs_j_buf.as_ptr() as *const __m128i);
    let f_vec = _mm_loadu_si128(rhs_f_buf.as_ptr() as *const __m128i);

    // Check if e > f
    let e_gt_f = _mm_cmpgt_epi32(e_vec, f_vec);

    // Compute |i - j|
    let i_sub_j = _mm_sub_epi32(i_vec, j_vec);
    let j_sub_i = _mm_sub_epi32(j_vec, i_vec);
    let abs_diff = _mm_max_epi32(i_sub_j, j_sub_i);

    // Compute (f - e)
    let error_diff = _mm_sub_epi32(f_vec, e_vec);

    // Check if |i - j| <= (f - e)
    let abs_gt_error = _mm_cmpgt_epi32(abs_diff, error_diff);
    let subsumes_mask = _mm_andnot_si128(abs_gt_error, _mm_set1_epi32(-1));

    // Combine with e <= f check
    let final_mask = _mm_andnot_si128(e_gt_f, subsumes_mask);

    // Extract results
    let mask = _mm_movemask_ps(_mm_castsi128_ps(final_mask));

    for (idx, result) in results.iter_mut().enumerate().take(4) {
        *result = (mask & (1 << idx)) != 0;
    }

    // Process remaining elements with scalar (if count > 4)
    for idx in 4..count {
        let i = lhs_term_indices[idx];
        let e = lhs_errors[idx];
        let j = rhs_term_indices[idx];
        let f = rhs_errors[idx];

        results[idx] = e <= f && i.abs_diff(j) <= (f - e);
    }

    &results[..count]
}

#[cfg(test)]
mod subsumption_tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_subsumption_simd_basic() {
        // Test cases from position.rs tests
        let test_cases = vec![
            // (lhs_i, lhs_e, rhs_j, rhs_f, expected)
            // (5, 2) should subsume (5, 3) - same position, fewer errors
            (5, 2, 5, 3, true),
            // (5, 2) should subsume (4, 3) - |5-4| = 1 <= (3-2) = 1
            (5, 2, 4, 3, true),
            // (3, 2) should subsume (3, 2) - same position and errors
            (3, 2, 3, 2, true),
            // (3, 3) should NOT subsume (5, 2) - e > f
            (3, 3, 5, 2, false),
            // (10, 1) should subsume (8, 3) - |10-8| = 2 <= (3-1) = 2
            (10, 1, 8, 3, true),
            // (10, 1) should NOT subsume (5, 3) - |10-5| = 5 > (3-1) = 2
            (10, 1, 5, 3, false),
            // (0, 0) should subsume (0, 0) - identical
            (0, 0, 0, 0, true),
            // (0, 0) should subsume (1, 1) - |0-1| = 1 <= (1-0) = 1
            (0, 0, 1, 1, true),
        ];

        for (lhs_i, lhs_e, rhs_j, rhs_f, expected) in test_cases {
            let lhs_indices = [lhs_i];
            let lhs_errs = [lhs_e];
            let rhs_indices = [rhs_j];
            let rhs_errs = [rhs_f];
            let mut results = [false; 8];

            let result = check_subsumption_simd(
                &lhs_indices,
                &lhs_errs,
                &rhs_indices,
                &rhs_errs,
                1,
                &mut results,
            );

            assert_eq!(
                result[0], expected,
                "SIMD result mismatch for ({}, {}) vs ({}, {}): expected {}, got {}",
                lhs_i, lhs_e, rhs_j, rhs_f, expected, result[0]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_subsumption_simd_batch() {
        // Test processing multiple pairs at once (AVX2: 8 pairs)
        let lhs_indices = [5, 5, 3, 3, 10, 10, 0, 0];
        let lhs_errs = [2, 2, 2, 3, 1, 1, 0, 0];
        let rhs_indices = [5, 4, 3, 5, 8, 5, 0, 1];
        let rhs_errs = [3, 3, 2, 2, 3, 3, 0, 1];
        let expected = [true, true, true, false, true, false, true, true];

        let mut results = [false; 8];
        let result = check_subsumption_simd(
            &lhs_indices,
            &lhs_errs,
            &rhs_indices,
            &rhs_errs,
            8,
            &mut results,
        );

        for i in 0..8 {
            assert_eq!(
                result[i], expected[i],
                "Batch SIMD result mismatch at index {}: ({}, {}) vs ({}, {}) - expected {}, got {}",
                i, lhs_indices[i], lhs_errs[i], rhs_indices[i], rhs_errs[i], expected[i], result[i]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_subsumption_simd_vs_scalar() {
        // Comprehensive comparison of SIMD vs scalar implementation
        let test_cases = vec![
            // Various position pairs to test
            ([0, 1, 5, 10, 3, 7, 2, 15], [0, 0, 2, 1, 3, 2, 1, 5]),
            ([0, 1, 2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 1, 1, 1, 1]),
            ([10, 10, 10, 10, 10, 10, 10, 10], [0, 1, 2, 3, 4, 5, 6, 7]),
        ];

        let rhs_cases = vec![
            ([0, 2, 6, 12, 5, 9, 4, 20], [0, 1, 3, 3, 4, 4, 2, 8]),
            ([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 2, 2, 2, 2, 2, 2]),
            ([10, 11, 12, 13, 14, 15, 16, 17], [1, 2, 3, 4, 5, 6, 7, 8]),
        ];

        for (lhs_i, lhs_e) in &test_cases {
            for (rhs_j, rhs_f) in &rhs_cases {
                let mut simd_results = [false; 8];
                let mut scalar_results = [false; 8];

                // Get SIMD results
                let simd_result =
                    check_subsumption_simd(lhs_i, lhs_e, rhs_j, rhs_f, 8, &mut simd_results);

                // Get scalar results
                let scalar_result =
                    check_subsumption_scalar(lhs_i, lhs_e, rhs_j, rhs_f, 8, &mut scalar_results);

                // Compare
                assert_eq!(
                    simd_result, scalar_result,
                    "SIMD vs scalar mismatch:\nlhs_i: {:?}\nlhs_e: {:?}\nrhs_j: {:?}\nrhs_f: {:?}\nSIMD: {:?}\nScalar: {:?}",
                    lhs_i, lhs_e, rhs_j, rhs_f, simd_result, scalar_result
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_subsumption_simd_edge_cases() {
        // Test edge cases: large indices, zero errors, etc.

        // Large indices (but still fit in u32)
        let lhs_indices = [1000, 2000, 5000, 10000, 100, 200, 300, 400];
        let lhs_errs = [10, 20, 50, 100, 5, 10, 15, 20];
        let rhs_indices = [1010, 2025, 5060, 10150, 110, 215, 325, 440];
        let rhs_errs = [20, 50, 120, 250, 20, 25, 40, 60];

        let mut results = [false; 8];
        let result = check_subsumption_simd(
            &lhs_indices,
            &lhs_errs,
            &rhs_indices,
            &rhs_errs,
            8,
            &mut results,
        );

        // Verify against scalar
        let mut scalar_results = [false; 8];
        check_subsumption_scalar(
            &lhs_indices,
            &lhs_errs,
            &rhs_indices,
            &rhs_errs,
            8,
            &mut scalar_results,
        );

        assert_eq!(
            result,
            &scalar_results[..8],
            "SIMD vs scalar mismatch on large indices"
        );

        // All zero errors
        let lhs_indices = [0, 1, 2, 3, 4, 5, 6, 7];
        let lhs_errs = [0, 0, 0, 0, 0, 0, 0, 0];
        let rhs_indices = [0, 1, 2, 3, 4, 5, 6, 7];
        let rhs_errs = [0, 0, 0, 0, 0, 0, 0, 0];

        let mut results = [false; 8];
        let result = check_subsumption_simd(
            &lhs_indices,
            &lhs_errs,
            &rhs_indices,
            &rhs_errs,
            8,
            &mut results,
        );

        // All should subsume (same position, same errors)
        assert_eq!(
            result,
            &[true, true, true, true, true, true, true, true],
            "Zero errors edge case failed"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_subsumption_simd_partial_batches() {
        // Test with count < 8 (SSE4.1 path or scalar)
        let lhs_indices = [5, 5, 3, 3, 10];
        let lhs_errs = [2, 2, 2, 3, 1];
        let rhs_indices = [5, 4, 3, 5, 8];
        let rhs_errs = [3, 3, 2, 2, 3];
        let expected = [true, true, true, false, true];

        // Test count = 5 (SSE4.1: 4 + 1 scalar)
        let mut results = [false; 8];
        let result = check_subsumption_simd(
            &lhs_indices,
            &lhs_errs,
            &rhs_indices,
            &rhs_errs,
            5,
            &mut results,
        );

        for i in 0..5 {
            assert_eq!(
                result[i], expected[i],
                "Partial batch (count=5) mismatch at index {}: expected {}, got {}",
                i, expected[i], result[i]
            );
        }

        // Test count = 4 (exact SSE4.1)
        let mut results = [false; 8];
        let result = check_subsumption_simd(
            &lhs_indices,
            &lhs_errs,
            &rhs_indices,
            &rhs_errs,
            4,
            &mut results,
        );

        for i in 0..4 {
            assert_eq!(
                result[i], expected[i],
                "Partial batch (count=4) mismatch at index {}: expected {}, got {}",
                i, expected[i], result[i]
            );
        }

        // Test count = 2 (scalar fallback)
        let mut results = [false; 8];
        let result = check_subsumption_simd(
            &lhs_indices,
            &lhs_errs,
            &rhs_indices,
            &rhs_errs,
            2,
            &mut results,
        );

        for i in 0..2 {
            assert_eq!(
                result[i], expected[i],
                "Partial batch (count=2) mismatch at index {}: expected {}, got {}",
                i, expected[i], result[i]
            );
        }
    }
}

/// SIMD-accelerated minimum value computation
///
/// Finds the minimum value among up to 8 usize values using SIMD horizontal
/// reduction. This is used for finding the minimum error count across positions
/// in a state.
///
/// # Arguments
/// * `values` - Array of values to find minimum of
/// * `count` - Number of values to consider (must be > 0 and <= 8)
///
/// # Returns
/// The minimum value among `values[0..count]`
///
/// # Performance
/// - AVX2: Processes 8 values with horizontal minimum (~5-7 ns)
/// - SSE4.1: Processes 4-8 values with horizontal minimum (~4-6 ns)
/// - Scalar fallback for count < 4 or when SIMD unavailable
#[cfg(target_arch = "x86_64")]
pub fn find_minimum_simd(values: &[usize], count: usize) -> usize {
    debug_assert!(count > 0 && count <= 8, "count must be in range 1..=8");
    debug_assert!(values.len() >= count);

    // Fast path for single value
    if count == 1 {
        return values[0];
    }

    // Use SIMD for count >= 4
    if count == 8 && is_x86_feature_detected!("avx2") {
        unsafe { find_minimum_avx2(values) }
    } else if count >= 4 && is_x86_feature_detected!("sse4.1") {
        unsafe { find_minimum_sse41(values, count) }
    } else {
        find_minimum_scalar(values, count)
    }
}

/// Scalar fallback for minimum finding
#[inline(always)]
fn find_minimum_scalar(values: &[usize], count: usize) -> usize {
    values[0..count]
        .iter()
        .copied()
        .min()
        .expect("find_minimum_scalar: count > 0 (caller invariant)")
}

/// AVX2-accelerated minimum finding via horizontal reduction
///
/// Processes 8 usize values (converted to u32) and finds the minimum
/// using a series of horizontal minimum operations.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_minimum_avx2(values: &[usize]) -> usize {
    debug_assert!(values.len() >= 8);

    // Convert usize to u32 (assume values fit in u32)
    let mut buf = [0u32; 8];
    for i in 0..8 {
        buf[i] = values[i] as u32;
    }

    // Load into 256-bit vector
    let vec = _mm256_loadu_si256(buf.as_ptr() as *const __m256i);

    // Horizontal minimum reduction:
    // Step 1: Compare each 128-bit half and take minimum
    // Extract high 128 bits
    let high = _mm256_extracti128_si256(vec, 1);
    // Extract low 128 bits and compare
    let low = _mm256_castsi256_si128(vec);
    let min_half = _mm_min_epu32(low, high);

    // Step 2: Within 128 bits, find minimum of 4 values
    // Shuffle to compare pairs: [0,1,2,3] -> [2,3,0,1]
    let shuffled = _mm_shuffle_epi32(min_half, 0b01_00_11_10);
    let min_pairs = _mm_min_epu32(min_half, shuffled);

    // Step 3: Final reduction to single minimum
    // Shuffle to compare final pairs: [min01, min23, ...] -> [min23, min01, ...]
    let final_shuffle = _mm_shuffle_epi32(min_pairs, 0b00_00_00_01);
    let final_min = _mm_min_epu32(min_pairs, final_shuffle);

    // Extract the minimum value (in lowest 32 bits)
    _mm_extract_epi32(final_min, 0) as usize
}

/// SSE4.1-accelerated minimum finding
///
/// Processes 4-8 values using 128-bit SIMD + scalar remainder
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn find_minimum_sse41(values: &[usize], count: usize) -> usize {
    debug_assert!((4..=8).contains(&count));

    // Process first 4 values with SIMD
    let mut buf = [0u32; 4];
    for i in 0..4 {
        buf[i] = values[i] as u32;
    }

    let vec = _mm_loadu_si128(buf.as_ptr() as *const __m128i);

    // Horizontal minimum reduction for 4 values
    // Step 1: Compare pairs [0,1,2,3] -> [2,3,0,1]
    let shuffled = _mm_shuffle_epi32(vec, 0b01_00_11_10);
    let min_pairs = _mm_min_epu32(vec, shuffled);

    // Step 2: Final reduction [min01, min23, ...] -> [min23, min01, ...]
    let final_shuffle = _mm_shuffle_epi32(min_pairs, 0b00_00_00_01);
    let final_min = _mm_min_epu32(min_pairs, final_shuffle);

    let mut min_val = _mm_extract_epi32(final_min, 0) as usize;

    // Process remaining values (if count > 4) with scalar
    for value in values.iter().take(count).skip(4) {
        min_val = min_val.min(*value);
    }

    min_val
}

#[cfg(test)]
mod minimum_tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_find_minimum_simd_basic() {
        // Test cases: (values, count, expected_min)
        let test_cases = vec![
            // Single value
            (vec![5], 1, 5),
            // Two values
            (vec![10, 3], 2, 3),
            (vec![1, 20], 2, 1),
            // Four values (SSE4.1 threshold)
            (vec![5, 2, 8, 1], 4, 1),
            (vec![100, 50, 75, 25], 4, 25),
            // Eight values (AVX2 full)
            (vec![10, 3, 7, 2, 15, 1, 9, 5], 8, 1),
            (vec![100, 200, 50, 300, 25, 400, 150, 75], 8, 25),
            // Minimum at different positions
            (vec![1, 2, 3, 4, 5, 6, 7, 8], 8, 1), // First
            (vec![8, 7, 6, 5, 4, 3, 2, 1], 8, 1), // Last
            (vec![5, 4, 3, 2, 1, 2, 3, 4], 8, 1), // Middle
        ];

        for (values, count, expected) in test_cases {
            let result = find_minimum_simd(&values, count);
            assert_eq!(
                result, expected,
                "SIMD minimum mismatch for values={:?}, count={}: expected {}, got {}",
                values, count, expected, result
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_find_minimum_simd_vs_scalar() {
        // Comprehensive comparison of SIMD vs scalar
        let test_cases = vec![
            vec![5, 2, 8, 1, 15, 3, 9, 6],
            vec![100, 200, 50, 300, 25, 400, 150, 75],
            vec![10, 10, 10, 10, 10, 10, 10, 10],     // All same
            vec![0, 1, 2, 3, 4, 5, 6, 7],             // Sequential
            vec![1000, 500, 250, 125, 62, 31, 15, 7], // Decreasing
        ];

        for values in test_cases {
            for count in 1..=8 {
                let simd_result = find_minimum_simd(&values, count);
                let scalar_result = find_minimum_scalar(&values, count);

                assert_eq!(
                    simd_result, scalar_result,
                    "SIMD vs scalar mismatch for values={:?}, count={}: SIMD={}, scalar={}",
                    values, count, simd_result, scalar_result
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_find_minimum_edge_cases() {
        // Edge case: All values the same
        let values = vec![42, 42, 42, 42, 42, 42, 42, 42];
        assert_eq!(find_minimum_simd(&values, 8), 42);

        // Edge case: Large values (but fit in u32)
        let values = vec![
            1000000, 2000000, 500000, 3000000, 250000, 4000000, 1500000, 750000,
        ];
        assert_eq!(find_minimum_simd(&values, 8), 250000);

        // Edge case: Zero minimum
        let values = vec![10, 5, 0, 3, 7, 2, 8, 4];
        assert_eq!(find_minimum_simd(&values, 8), 0);

        // Edge case: Partial arrays
        let values = vec![10, 5, 3, 7, 2];
        assert_eq!(find_minimum_simd(&values, 5), 2);
        assert_eq!(find_minimum_simd(&values, 3), 3);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_find_minimum_real_world() {
        // Simulate real error counts from Levenshtein automaton positions
        // Typical state might have 2-5 positions with errors in range 0-10

        // Small state (2 positions)
        let errors = vec![2, 1];
        assert_eq!(find_minimum_simd(&errors, 2), 1);

        // Medium state (4 positions)
        let errors = vec![3, 1, 2, 4];
        assert_eq!(find_minimum_simd(&errors, 4), 1);

        // Large state (8 positions - rare but possible)
        let errors = vec![5, 2, 7, 1, 3, 6, 4, 8];
        assert_eq!(find_minimum_simd(&errors, 8), 1);

        // State with zero errors (exact match path)
        let errors = vec![0, 1, 2, 3];
        assert_eq!(find_minimum_simd(&errors, 4), 0);
    }
}

// ============================================================================
// BATCH 2B: DICTIONARY EDGE LOOKUP SIMD OPTIMIZATION
// ============================================================================
//
// ## Motivation
//
// Dictionary edge lookup is the most frequently called dictionary operation
// during transducer query execution. For every character in every candidate
// term, we must check if an edge exists with that label.
//
// Current implementations use adaptive strategies:
// - **DAWG**: Linear search for <16 edges, binary for ≥16 edges
// - **Optimized DAWG**: Linear search for ≤4 edges, binary for >4 edges
//
// Linear search dominates because most nodes have 1-5 edges. SIMD can
// dramatically accelerate this by checking multiple edge labels in parallel.
//
// ## Algorithm: Parallel Edge Label Comparison
//
// Given a sorted array of edge labels `[l₀, l₁, l₂, ..., l_n]` and a target
// label `target`, find the index `i` where `l_i == target`.
//
// ### SIMD Approach:
//
// 1. **Broadcast**: Replicate target label across all SIMD lanes
//    - AVX2: 32 copies in 256-bit register
//    - SSE4.1: 16 copies in 128-bit register
//
// 2. **Load**: Load up to 32 (AVX2) or 16 (SSE4.1) edge labels
//
// 3. **Compare**: Single instruction compares all lanes simultaneously
//    - `_mm256_cmpeq_epi8`: 32 byte comparisons in ~1 cycle
//    - `_mm_cmpeq_epi8`: 16 byte comparisons in ~1 cycle
//
// 4. **Extract**: Convert comparison mask to bit mask and find first match
//    - `_mm256_movemask_epi8`: Extract 32-bit mask
//    - `trailing_zeros()`: Find first set bit = index
//
// ### Example (AVX2, 8 edges):
//
// ```
// edges = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 0, 0, ..., 0]
// target = 'e'
//
// Step 1 - Broadcast:
//   target_vec = ['e', 'e', 'e', 'e', ..., 'e']  (32 copies)
//
// Step 2 - Load:
//   edges_vec = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 0, 0, ..., 0]
//
// Step 3 - Compare (_mm256_cmpeq_epi8):
//   cmp_result = [0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, ...]
//                  ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑
//                  a≠e   b≠e   c≠e   d≠e   e=e!  f≠e   g≠e   h≠e
//
// Step 4 - Extract (_mm256_movemask_epi8):
//   mask = 0b...00010000 (bit 4 set)
//   index = trailing_zeros(mask) = 4
//   return Some((4, edges[4].1))  // Return edge target at index 4
// ```
//
// ## Performance Characteristics
//
// **Scalar Linear Search** (typical):
// - Best case: 1 comparison (first edge)
// - Average: n/2 comparisons
// - Worst case: n comparisons
// - Cost per comparison: ~1 cycle (load + compare + branch)
//
// **SIMD Linear Search**:
// - Fixed cost: ~10 cycles (broadcast + load + compare + extract)
// - Compares: 16-32 edges simultaneously
// - No branch misprediction penalty
//
// **Crossover Analysis**:
// - For 4 edges: SIMD ~10 cycles vs Scalar ~2 cycles (avg) → Scalar wins
// - For 8 edges: SIMD ~10 cycles vs Scalar ~4 cycles (avg) → SIMD ~2.5x faster
// - For 16 edges: SIMD ~10 cycles vs Scalar ~8 cycles (avg) → SIMD ~1.25x faster
// - For 32 edges: SIMD ~20 cycles vs Scalar ~16 cycles (avg) → SIMD ~1.25x faster
//
// **Optimal Strategy**:
// - edges < 4: Use scalar (SIMD overhead too high)
// - 4 ≤ edges < 16: Use SSE4.1 (16-way comparison)
// - 16 ≤ edges < threshold: Use AVX2 (32-way comparison)
// - edges ≥ threshold: Use binary search (O(log n) better)
//
// ## Implementation Notes
//
// 1. **Padding Safety**: We pad edge arrays to avoid reading past allocation
//    - AVX2: Pad to multiple of 32 bytes
//    - SSE4.1: Pad to multiple of 16 bytes
//    - Padding bytes set to 0xFF (never match valid labels 0x00-0x7F)
//
// 2. **Sorted Edges**: All dictionary implementations maintain sorted edges
//    - This invariant is critical for binary search
//    - SIMD doesn't require sorting but benefits from it (early exit)
//
// 3. **Feature Detection**: Runtime CPU feature detection ensures portability
//    - `is_x86_feature_detected!("avx2")` for AVX2 path
//    - `is_x86_feature_detected!("sse4.1")` for SSE4.1 path
//    - Scalar fallback for unsupported CPUs
//
// ============================================================================

/// Find an edge label in a sorted array of (label, target) tuples using SIMD.
///
/// This function searches for a specific edge label among a node's edges and
/// returns the index if found. It uses SIMD instructions to compare multiple
/// edge labels simultaneously.
///
/// # Algorithm
///
/// 1. **Threshold Check**: For small edge counts (<4), use scalar search
/// 2. **SIMD Selection**: Choose AVX2 (32-way) or SSE4.1 (16-way) based on size
/// 3. **Broadcast**: Replicate target label across all SIMD lanes
/// 4. **Compare**: Single instruction compares all lanes simultaneously
/// 5. **Extract**: Find first matching position using bit mask operations
///
/// # Performance
///
/// - **Scalar** (edges < 4): 1-3 cycles average
/// - **SSE4.1** (4-15 edges): ~10 cycles (2-4x faster than scalar)
/// - **AVX2** (16-31 edges): ~12 cycles (2-3x faster than scalar)
/// - **Binary Search** (≥32 edges): Falls back to O(log n)
///
/// # Safety
///
/// The SIMD implementations use unaligned loads (`loadu`) to safely handle
/// any edge array without alignment requirements. Padding to SIMD width
/// prevents reading past allocation boundaries.
///
/// # Arguments
///
/// * `edges` - Sorted slice of (u8, T) tuples representing (label, target)
/// * `target_label` - The edge label to search for
///
/// # Returns
///
/// * `Some(index)` - Index into edges slice where label was found
/// * `None` - Label not found in edges
///
/// # Examples
///
/// ```
/// # use liblevenshtein::transducer::simd::find_edge_label_simd;
/// let edges = vec![(b'a', 1), (b'c', 2), (b'e', 3), (b'g', 4)];
///
/// assert_eq!(find_edge_label_simd(&edges, b'a'), Some(0));
/// assert_eq!(find_edge_label_simd(&edges, b'e'), Some(2));
/// assert_eq!(find_edge_label_simd(&edges, b'z'), None);
/// ```
#[cfg(target_arch = "x86_64")]
pub fn find_edge_label_simd<T>(edges: &[(u8, T)], target_label: u8) -> Option<usize> {
    let count = edges.len();

    // Data-driven thresholds based on empirical benchmarking:
    //
    // Benchmark results:
    // - count < 12: Scalar is 2-3x faster (SIMD overhead dominates)
    // - count >= 12: SIMD provides 1.2-1.5x speedup
    // - count >= 24: Binary search O(log n) becomes competitive
    //
    // See docs/BATCH2B_PERFORMANCE_ANALYSIS.md for detailed analysis

    // Threshold 1: Scalar wins for < 12 edges (SIMD overhead ~10ns dominates)
    if count < 12 {
        return find_edge_label_scalar(edges, target_label);
    }

    // Threshold 2: SSE4.1 for 12-16 edges (optimal range)
    // SSE4.1 can handle up to 16 bytes (edges) in a single register
    // Benchmarks show SSE4.1 at 16 edges (8.5ns) provides 1.24x speedup
    if count <= 16 && is_x86_feature_detected!("sse4.1") {
        return unsafe { find_edge_label_sse41(edges, target_label, count) };
    }

    // Threshold 3: For edge counts > 16, scalar or binary search is better
    // For 17-23 edges: Linear search overhead is acceptable
    // For ≥24 edges: Binary search O(log₂24) ≈ 4.6 comparisons is optimal
    find_edge_label_scalar(edges, target_label)
}

/// Scalar fallback for edge label search.
///
/// Simple linear search used when SIMD overhead isn't justified (<4 edges)
/// or when SIMD instructions are unavailable.
#[inline]
fn find_edge_label_scalar<T>(edges: &[(u8, T)], target_label: u8) -> Option<usize> {
    edges.iter().position(|(label, _)| *label == target_label)
}

/// SSE4.1 implementation: Compare 16 edge labels simultaneously.
///
/// # Algorithm
///
/// 1. Extract labels into temporary buffer (max 16)
/// 2. Pad buffer to 16 bytes with 0xFF (never matches)
/// 3. Load labels into 128-bit SSE register
/// 4. Broadcast target across 128-bit register
/// 5. Compare all 16 lanes simultaneously
/// 6. Extract bit mask and find first match
///
/// # Safety
///
/// Requires SSE4.1 CPU feature. Caller must verify with
/// `is_x86_feature_detected!("sse4.1")` before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn find_edge_label_sse41<T>(
    edges: &[(u8, T)],
    target_label: u8,
    count: usize,
) -> Option<usize> {
    use std::arch::x86_64::*;

    // Extract edge labels into buffer and pad to 16 bytes
    let mut labels = [0xFFu8; 16]; // 0xFF = padding byte (never matches ASCII)
    for (i, (label, _)) in edges.iter().enumerate().take(16.min(count)) {
        labels[i] = *label;
    }

    // Load labels into SSE register (unaligned load is safe)
    let labels_vec = _mm_loadu_si128(labels.as_ptr() as *const __m128i);

    // Broadcast target label to all 16 lanes
    let target_vec = _mm_set1_epi8(target_label as i8);

    // Compare all 16 lanes simultaneously (each 0xFF if equal, 0x00 if not)
    let cmp_result = _mm_cmpeq_epi8(labels_vec, target_vec);

    // Extract comparison mask (1 bit per lane, 16 bits total)
    let mask = _mm_movemask_epi8(cmp_result);

    // Find first set bit (= first match)
    if mask != 0 {
        let index = mask.trailing_zeros() as usize;
        // Verify index is within actual edge count (not padding)
        if index < count {
            return Some(index);
        }
    }

    None
}

/// AVX2 implementation: Compare 32 edge labels simultaneously.
///
/// # Algorithm
///
/// 1. Extract labels into temporary buffer (max 32)
/// 2. Pad buffer to 32 bytes with 0xFF (never matches)
/// 3. Load labels into 256-bit AVX2 register
/// 4. Broadcast target across 256-bit register
/// 5. Compare all 32 lanes simultaneously
/// 6. Extract bit mask and find first match
///
/// # Performance
///
/// For 16-31 edges, this provides ~2-3x speedup over scalar linear search.
/// The AVX2 path handles twice as many comparisons as SSE4.1 in a single
/// instruction, with minimal additional latency.
///
/// # Safety
///
/// Requires AVX2 CPU feature. Caller must verify with
/// `is_x86_feature_detected!("avx2")` before calling.
///
/// # Note
///
/// Currently unused due to performance analysis showing buffer copy overhead
/// dominates. Kept for future optimization when pre-extracted label arrays
/// are implemented. See docs/BATCH2B_PERFORMANCE_ANALYSIS.md for details.
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_edge_label_avx2<T>(
    edges: &[(u8, T)],
    target_label: u8,
    count: usize,
) -> Option<usize> {
    use std::arch::x86_64::*;

    // Extract edge labels into buffer and pad to 32 bytes
    let mut labels = [0xFFu8; 32]; // 0xFF = padding byte (never matches ASCII)
    for (i, (label, _)) in edges.iter().enumerate().take(32.min(count)) {
        labels[i] = *label;
    }

    // Load labels into AVX2 register (unaligned load is safe)
    let labels_vec = _mm256_loadu_si256(labels.as_ptr() as *const __m256i);

    // Broadcast target label to all 32 lanes
    let target_vec = _mm256_set1_epi8(target_label as i8);

    // Compare all 32 lanes simultaneously (each 0xFF if equal, 0x00 if not)
    let cmp_result = _mm256_cmpeq_epi8(labels_vec, target_vec);

    // Extract comparison mask (1 bit per lane, 32 bits total)
    let mask = _mm256_movemask_epi8(cmp_result);

    // Find first set bit (= first match)
    if mask != 0 {
        let index = mask.trailing_zeros() as usize;
        // Verify index is within actual edge count (not padding)
        if index < count {
            return Some(index);
        }
    }

    None
}

#[cfg(test)]
mod edge_lookup_tests {
    use super::*;

    #[test]
    fn test_edge_not_found() {
        let edges = vec![(b'a', 1), (b'c', 2), (b'e', 3), (b'g', 4)];
        assert_eq!(find_edge_label_simd(&edges, b'z'), None);
        assert_eq!(find_edge_label_simd(&edges, b'b'), None);
    }

    #[test]
    fn test_edge_at_beginning() {
        let edges = vec![(b'a', 10), (b'b', 20), (b'c', 30), (b'd', 40)];
        assert_eq!(find_edge_label_simd(&edges, b'a'), Some(0));
    }

    #[test]
    fn test_edge_at_middle() {
        let edges = vec![(b'a', 10), (b'b', 20), (b'c', 30), (b'd', 40)];
        assert_eq!(find_edge_label_simd(&edges, b'b'), Some(1));
        assert_eq!(find_edge_label_simd(&edges, b'c'), Some(2));
    }

    #[test]
    fn test_edge_at_end() {
        let edges = vec![(b'a', 10), (b'b', 20), (b'c', 30), (b'd', 40)];
        assert_eq!(find_edge_label_simd(&edges, b'd'), Some(3));
    }

    #[test]
    fn test_empty_edges() {
        let edges: Vec<(u8, usize)> = vec![];
        assert_eq!(find_edge_label_simd(&edges, b'a'), None);
    }

    #[test]
    fn test_single_edge() {
        let edges = vec![(b'x', 100)];
        assert_eq!(find_edge_label_simd(&edges, b'x'), Some(0));
        assert_eq!(find_edge_label_simd(&edges, b'y'), None);
    }

    #[test]
    fn test_exactly_4_edges() {
        // Threshold: 4 edges should use SSE4.1 path
        let edges = vec![(b'a', 1), (b'b', 2), (b'c', 3), (b'd', 4)];
        assert_eq!(find_edge_label_simd(&edges, b'a'), Some(0));
        assert_eq!(find_edge_label_simd(&edges, b'c'), Some(2));
        assert_eq!(find_edge_label_simd(&edges, b'd'), Some(3));
        assert_eq!(find_edge_label_simd(&edges, b'z'), None);
    }

    #[test]
    fn test_8_edges() {
        // SSE4.1 path (8 < 16)
        let edges = vec![
            (b'a', 1),
            (b'b', 2),
            (b'c', 3),
            (b'd', 4),
            (b'e', 5),
            (b'f', 6),
            (b'g', 7),
            (b'h', 8),
        ];
        assert_eq!(find_edge_label_simd(&edges, b'a'), Some(0));
        assert_eq!(find_edge_label_simd(&edges, b'e'), Some(4));
        assert_eq!(find_edge_label_simd(&edges, b'h'), Some(7));
        assert_eq!(find_edge_label_simd(&edges, b'z'), None);
    }

    #[test]
    fn test_exactly_16_edges() {
        // Boundary: 16 edges should trigger AVX2 path if available
        let edges: Vec<(u8, usize)> = (0..16).map(|i| (b'a' + i, i as usize)).collect();
        assert_eq!(find_edge_label_simd(&edges, b'a'), Some(0)); // First
        assert_eq!(find_edge_label_simd(&edges, b'h'), Some(7)); // Middle
        assert_eq!(find_edge_label_simd(&edges, b'p'), Some(15)); // Last (b'a' + 15 = b'p')
        assert_eq!(find_edge_label_simd(&edges, b'z'), None); // Not found
    }

    #[test]
    fn test_32_edges() {
        // Full AVX2 path (32 edges)
        let edges: Vec<(u8, usize)> = (0..32).map(|i| (i as u8, i as usize)).collect();
        assert_eq!(find_edge_label_simd(&edges, 0), Some(0)); // First
        assert_eq!(find_edge_label_simd(&edges, 16), Some(16)); // Middle
        assert_eq!(find_edge_label_simd(&edges, 31), Some(31)); // Last
        assert_eq!(find_edge_label_simd(&edges, 255), None); // Not found
    }

    #[test]
    fn test_boundary_15_edges() {
        // Just below AVX2 threshold (should use SSE4.1)
        let edges: Vec<(u8, usize)> = (0..15).map(|i| (b'a' + i, i as usize)).collect();
        assert_eq!(find_edge_label_simd(&edges, b'a'), Some(0));
        assert_eq!(find_edge_label_simd(&edges, b'o'), Some(14)); // b'a' + 14 = b'o'
        assert_eq!(find_edge_label_simd(&edges, b'z'), None);
    }

    #[test]
    fn test_boundary_17_edges() {
        // 17 edges: Exceeds SSE4.1 limit (16 edges), falls back to scalar
        // This validates that >16 edges work correctly via scalar path
        let edges: Vec<(u8, usize)> = (0..17).map(|i| (b'a' + i, i as usize)).collect();
        assert_eq!(find_edge_label_simd(&edges, b'a'), Some(0));
        assert_eq!(find_edge_label_simd(&edges, b'q'), Some(16)); // b'a' + 16 = b'q'
        assert_eq!(find_edge_label_simd(&edges, b'z'), None);
    }

    #[test]
    fn test_all_positions_in_16_edges() {
        // Comprehensive: test finding edge at every position
        let edges: Vec<(u8, usize)> = (0..16).map(|i| (b'a' + i, i as usize * 10)).collect();

        for i in 0..16 {
            let label = b'a' + i;
            assert_eq!(
                find_edge_label_simd(&edges, label),
                Some(i as usize),
                "Failed to find label '{}' at position {}",
                label as char,
                i
            );
        }
    }

    #[test]
    fn test_realistic_english_letters() {
        // Simulate realistic node: edges for common English letters
        let edges = vec![
            (b'a', 1),
            (b'e', 2),
            (b'i', 3),
            (b'n', 4),
            (b'o', 5),
            (b'r', 6),
            (b's', 7),
            (b't', 8),
        ];

        assert_eq!(find_edge_label_simd(&edges, b'a'), Some(0));
        assert_eq!(find_edge_label_simd(&edges, b'e'), Some(1));
        assert_eq!(find_edge_label_simd(&edges, b't'), Some(7));
        assert_eq!(find_edge_label_simd(&edges, b'x'), None);
    }
}
