#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_hybrid_cursor_free, llev_hybrid_cursor_next_batch, llev_hybrid_index_free,
    llev_hybrid_index_freeze, llev_hybrid_index_insert, llev_hybrid_index_new,
    llev_hybrid_index_query_knn, llev_hybrid_index_query_range, LlevHybridCursor, LlevHybridIndex,
    LlevHybridIndexConfig, LlevQuantizedIndexConfig, LlevStatus, LlevTemporalIndexMatch,
    LlevTemporalSearchLimits,
};
use liblevenshtein::time_series::{HybridSearchIndex, MsmConfig, QuantizationConfig};
use std::ptr;

fn config() -> LlevHybridIndexConfig {
    LlevHybridIndexConfig {
        source: LlevQuantizedIndexConfig {
            quant_min: 0.0,
            quant_max: 10.0,
            quant_bins: 5,
            reserved: 0,
            max_entries: 4,
            max_total_samples: 12,
            max_series_len: 3,
        },
        msm_cost: 1.0,
        trie_threshold_multiplier: 2.0,
        lower_bound_type: 0,
        use_lower_bounds: 1,
    }
}

fn query_knn(
    index: *const LlevHybridIndex,
    values: &[f64],
    k: usize,
    initial: f64,
    limits: &LlevTemporalSearchLimits,
) -> (LlevStatus, *mut LlevHybridCursor, u32) {
    let mut cursor = ptr::null_mut();
    let mut reason = u32::MAX;
    let status = unsafe {
        llev_hybrid_index_query_knn(
            index,
            values.as_ptr(),
            values.len(),
            k,
            initial,
            limits,
            &mut cursor,
            &mut reason,
        )
    };
    (status, cursor, reason)
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

fn create(config: &LlevHybridIndexConfig) -> *mut LlevHybridIndex {
    let mut index = ptr::null_mut();
    assert_eq!(
        unsafe { llev_hybrid_index_new(config, &mut index) },
        LlevStatus::Ok
    );
    assert!(!index.is_null());
    index
}

fn query(
    index: *const LlevHybridIndex,
    values: &[f64],
    cutoff: f64,
    limits: &LlevTemporalSearchLimits,
) -> (LlevStatus, *mut LlevHybridCursor, u32) {
    let mut cursor = ptr::null_mut();
    let mut reason = u32::MAX;
    let status = unsafe {
        llev_hybrid_index_query_range(
            index,
            values.as_ptr(),
            values.len(),
            cutoff,
            limits,
            &mut cursor,
            &mut reason,
        )
    };
    (status, cursor, reason)
}

fn collect(cursor: *mut LlevHybridCursor) -> (LlevStatus, Vec<(u64, f64)>, u32) {
    let mut matches = Vec::new();
    for _ in 0..1_000 {
        let mut batch = [LlevTemporalIndexMatch::default(); 1];
        let mut len = usize::MAX;
        let mut done = u8::MAX;
        let mut reason = u32::MAX;
        let status = unsafe {
            llev_hybrid_cursor_next_batch(
                cursor,
                batch.as_mut_ptr(),
                batch.len(),
                1_000,
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
        matches.extend(batch[..len].iter().map(|entry| (entry.id, entry.distance)));
        if done != 0 {
            return (status, matches, 0);
        }
    }
    panic!("hybrid FFI cursor failed to complete");
}

#[test]
fn frozen_hybrid_ffi_matches_native_candidate_and_msm_pipeline() {
    let handle = create(&config());
    let mut rust = HybridSearchIndex::<u64>::new(
        QuantizationConfig::uniform(0.0, 10.0, 5),
        MsmConfig::new(1.0),
    );
    for (id, series) in [
        (7, [1.0, 2.0, 3.0]),
        (8, [1.1, 2.1, 3.1]),
        (9, [3.0, 2.0, 1.0]),
        (10, [9.0, 9.0, 9.0]),
    ] {
        assert_eq!(
            unsafe { llev_hybrid_index_insert(handle, id, series.as_ptr(), series.len()) },
            LlevStatus::Ok
        );
        rust.insert(id, &series);
    }
    assert_eq!(unsafe { llev_hybrid_index_freeze(handle) }, LlevStatus::Ok);
    for cutoff in [0.0, 0.5, 1.0, 3.0, f64::INFINITY] {
        let (status, cursor, reason) = query(handle, &[1.0, 2.0, 3.0], cutoff, &limits());
        assert_eq!(status, LlevStatus::Ok);
        assert_eq!(reason, 0);
        let (status, mut observed, reason) = collect(cursor);
        unsafe { llev_hybrid_cursor_free(cursor) };
        assert_eq!(status, LlevStatus::Ok);
        assert_eq!(reason, 0);
        let mut expected = rust.search_exact(&[1.0, 2.0, 3.0], cutoff);
        observed.sort_by_key(|&(id, _)| id);
        expected.sort_by_key(|&(id, _)| id);
        assert_eq!(observed, expected, "cutoff={cutoff}");
    }
    unsafe { llev_hybrid_index_free(handle) };
}

#[test]
fn hybrid_ffi_retains_snapshot_and_reports_incomplete_pages() {
    let handle = create(&config());
    let series = [1.0, 2.0];
    assert_eq!(
        unsafe { llev_hybrid_index_insert(handle, 7, series.as_ptr(), series.len()) },
        LlevStatus::Ok
    );
    assert_eq!(
        unsafe { llev_hybrid_index_insert(handle, 8, series.as_ptr(), series.len()) },
        LlevStatus::Ok
    );
    assert_eq!(unsafe { llev_hybrid_index_freeze(handle) }, LlevStatus::Ok);
    let mut restricted = limits();
    restricted.max_results = 1;
    let (status, cursor, reason) = query(handle, &series, 0.0, &restricted);
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(reason, 0);
    let (status, partial, reason) = collect(cursor);
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(reason, 7);
    assert_eq!(partial.len(), 1);
    unsafe { llev_hybrid_cursor_free(cursor) };

    let (status, cursor, reason) = query(handle, &series, 0.0, &limits());
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(reason, 0);
    unsafe { llev_hybrid_index_free(handle) };
    let (status, matches, reason) = collect(cursor);
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(reason, 0);
    assert_eq!(matches.len(), 2);
    unsafe { llev_hybrid_cursor_free(cursor) };
}

#[test]
fn hybrid_ffi_knn_matches_legacy_expansion_and_fails_closed() {
    let handle = create(&config());
    let mut rust = HybridSearchIndex::<u64>::new(
        QuantizationConfig::uniform(0.0, 10.0, 5),
        MsmConfig::new(1.0),
    );
    for (id, series) in [
        (7, [1.0, 2.0, 3.0]),
        (8, [1.1, 2.1, 3.1]),
        (9, [3.0, 2.0, 1.0]),
        (10, [9.0, 9.0, 9.0]),
    ] {
        assert_eq!(
            unsafe { llev_hybrid_index_insert(handle, id, series.as_ptr(), series.len()) },
            LlevStatus::Ok
        );
        rust.insert(id, &series);
    }
    assert_eq!(unsafe { llev_hybrid_index_freeze(handle) }, LlevStatus::Ok);
    for k in 0..=4 {
        let (status, cursor, reason) = query_knn(handle, &[1.0, 2.0, 3.0], k, 0.0, &limits());
        assert_eq!(status, LlevStatus::Ok);
        assert_eq!(reason, 0);
        let (status, observed, reason) = collect(cursor);
        unsafe { llev_hybrid_cursor_free(cursor) };
        assert_eq!(status, LlevStatus::Ok);
        assert_eq!(reason, 0);
        assert_eq!(observed, rust.search_knn(&[1.0, 2.0, 3.0], k, 0.0));
    }
    let mut restricted = limits();
    restricted.max_results = 1;
    let (status, cursor, reason) = query_knn(handle, &[1.0, 2.0, 3.0], 2, 0.0, &restricted);
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(reason, 7);
    assert!(cursor.is_null());
    unsafe { llev_hybrid_index_free(handle) };
}
