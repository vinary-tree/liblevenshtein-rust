#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_quantized_cursor_free, llev_quantized_cursor_next_batch, llev_quantized_cursor_original,
    llev_quantized_index_free, llev_quantized_index_freeze, llev_quantized_index_insert,
    llev_quantized_index_new, llev_quantized_index_query, LlevAlgorithm, LlevQuantizedCursor,
    LlevQuantizedIndex, LlevQuantizedIndexConfig, LlevQuantizedMatch, LlevStatus,
    LlevTemporalSearchLimits,
};
use liblevenshtein::time_series::{QuantizationConfig, TimeSeriesIndex};
use std::ptr;

fn config() -> LlevQuantizedIndexConfig {
    LlevQuantizedIndexConfig {
        quant_min: 0.0,
        quant_max: 10.0,
        quant_bins: 5,
        reserved: 0,
        max_entries: 4,
        max_total_samples: 12,
        max_series_len: 3,
    }
}

fn limits() -> LlevTemporalSearchLimits {
    LlevTemporalSearchLimits {
        max_series_len: 3,
        max_dp_cells: 100_000,
        max_work_units: 100_000,
        max_scratch_bytes: 1_000_000,
        max_trie_nodes: 100_000,
        max_trie_edges: 100_000,
        max_candidates: 100,
        max_results: 100,
        max_queue_entries: 100_000,
        max_continuation_bytes: 1_000_000,
    }
}

fn create() -> *mut LlevQuantizedIndex {
    let mut handle = ptr::null_mut();
    assert_eq!(
        unsafe { llev_quantized_index_new(&config(), &mut handle) },
        LlevStatus::Ok
    );
    assert!(!handle.is_null());
    handle
}

fn insert(handle: *mut LlevQuantizedIndex, id: u64, samples: &[f64]) -> LlevStatus {
    unsafe { llev_quantized_index_insert(handle, id, samples.as_ptr(), samples.len()) }
}

fn query(
    handle: *const LlevQuantizedIndex,
    values: &[f64],
    threshold: usize,
    algorithm: LlevAlgorithm,
    limits: &LlevTemporalSearchLimits,
) -> (LlevStatus, *mut LlevQuantizedCursor, u32) {
    let mut cursor = ptr::null_mut();
    let mut reason = u32::MAX;
    let status = unsafe {
        llev_quantized_index_query(
            handle,
            values.as_ptr(),
            values.len(),
            threshold,
            algorithm as u32,
            limits,
            &mut cursor,
            &mut reason,
        )
    };
    (status, cursor, reason)
}

fn collect(cursor: *mut LlevQuantizedCursor) -> (LlevStatus, Vec<(u64, usize)>, u32) {
    let mut matches = Vec::new();
    for _ in 0..1_000 {
        let mut batch = [LlevQuantizedMatch::default(); 1];
        let mut len = usize::MAX;
        let mut done = u8::MAX;
        let mut reason = u32::MAX;
        let status = unsafe {
            llev_quantized_cursor_next_batch(
                cursor,
                batch.as_mut_ptr(),
                1,
                100,
                1,
                &mut len,
                &mut done,
                &mut reason,
            )
        };
        if status != LlevStatus::Ok {
            return (status, matches, reason);
        }
        assert_eq!(reason, 0);
        matches.extend(
            batch[..len]
                .iter()
                .map(|entry| (entry.id, entry.edit_distance)),
        );
        if done != 0 {
            return (status, matches, 0);
        }
    }
    panic!("quantized FFI cursor failed to complete");
}

#[test]
fn frozen_quantized_ffi_pages_match_all_three_rust_candidate_families() {
    let handle = create();
    let mut rust = TimeSeriesIndex::<u64>::new(QuantizationConfig::uniform(0.0, 10.0, 5));
    let entries = [
        (7, [1.0, 3.0, 5.0]),
        (8, [1.1, 3.1, 5.1]),
        (9, [3.0, 1.0, 5.0]),
        (10, [9.0, 9.0, 9.0]),
    ];
    for (id, samples) in entries {
        assert_eq!(insert(handle, id, &samples), LlevStatus::Ok);
        rust.insert(id, &samples);
    }
    assert_eq!(
        unsafe { llev_quantized_index_freeze(handle) },
        LlevStatus::Ok
    );
    let query_values = [1.05, 3.05, 5.05];
    for algorithm in [
        LlevAlgorithm::Standard,
        LlevAlgorithm::Transposition,
        LlevAlgorithm::MergeAndSplit,
    ] {
        let (status, cursor, reason) = query(handle, &query_values, 1, algorithm, &limits());
        assert_eq!(status, LlevStatus::Ok);
        assert_eq!(reason, 0);
        let (status, mut observed, reason) = collect(cursor);
        unsafe { llev_quantized_cursor_free(cursor) };
        assert_eq!(status, LlevStatus::Ok);
        assert_eq!(reason, 0);
        let mut expected = match algorithm {
            LlevAlgorithm::Standard => rust.search(&query_values, 1),
            LlevAlgorithm::Transposition => rust.search_transposition(&query_values, 1),
            LlevAlgorithm::MergeAndSplit => rust.search_merge_split(&query_values, 1),
            _ => unreachable!(),
        };
        observed.sort_unstable();
        expected.sort_unstable();
        assert_eq!(observed, expected, "{algorithm:?}");
    }
    unsafe { llev_quantized_index_free(handle) };
}

#[test]
fn quantized_ffi_keeps_snapshot_and_fails_closed_on_limits() {
    let handle = create();
    assert_eq!(insert(handle, 7, &[1.0, 3.0]), LlevStatus::Ok);
    assert_eq!(insert(handle, 8, &[1.1, 3.1]), LlevStatus::Ok);
    assert_eq!(
        unsafe { llev_quantized_index_freeze(handle) },
        LlevStatus::Ok
    );
    assert_eq!(insert(handle, 9, &[1.0]), LlevStatus::Closed);

    let mut restricted = limits();
    restricted.max_continuation_bytes = 0;
    let (status, cursor, reason) =
        query(handle, &[1.0, 3.0], 0, LlevAlgorithm::Standard, &restricted);
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(reason, 9);
    assert!(cursor.is_null());

    restricted = limits();
    restricted.max_results = 1;
    let (status, cursor, reason) =
        query(handle, &[1.0, 3.0], 0, LlevAlgorithm::Standard, &restricted);
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(reason, 0);
    let (status, partial, reason) = collect(cursor);
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(reason, 7);
    assert_eq!(partial.len(), 1);
    unsafe { llev_quantized_cursor_free(cursor) };

    let (status, cursor, reason) =
        query(handle, &[1.0, 3.0], 0, LlevAlgorithm::Standard, &limits());
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(reason, 0);
    unsafe { llev_quantized_index_free(handle) };
    let mut original_len = 0;
    assert_eq!(
        unsafe { llev_quantized_cursor_original(cursor, 7, ptr::null_mut(), 0, &mut original_len) },
        LlevStatus::Ok
    );
    assert_eq!(original_len, 2);
    let mut original = [0.0; 2];
    assert_eq!(
        unsafe {
            llev_quantized_cursor_original(
                cursor,
                7,
                original.as_mut_ptr(),
                original.len(),
                &mut original_len,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(original, [1.0, 3.0]);
    let (status, matches, reason) = collect(cursor);
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(reason, 0);
    assert_eq!(matches.len(), 2);
    unsafe { llev_quantized_cursor_free(cursor) };
}
