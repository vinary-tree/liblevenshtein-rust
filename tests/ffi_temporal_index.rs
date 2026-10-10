#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_temporal_index_cursor_free, llev_temporal_index_cursor_next_batch,
    llev_temporal_index_free, llev_temporal_index_freeze, llev_temporal_index_insert,
    llev_temporal_index_new, llev_temporal_index_query_erp_automaton_range,
    llev_temporal_index_query_knn, llev_temporal_index_query_range, llev_temporal_knn_cursor_free,
    llev_temporal_knn_cursor_next_batch, LlevStatus, LlevTemporalAlgorithm, LlevTemporalConfig,
    LlevTemporalIndex, LlevTemporalIndexConfig, LlevTemporalIndexCursor, LlevTemporalIndexMatch,
    LlevTemporalSearchLimits,
};
use liblevenshtein::time_series::{DtwConfig, ErpConfig, FrechetConfig, MsmConfig, TwedConfig};
use std::ptr;

fn config(algorithm: LlevTemporalAlgorithm) -> LlevTemporalIndexConfig {
    LlevTemporalIndexConfig {
        temporal: LlevTemporalConfig {
            algorithm: algorithm as u32,
            reserved: 0,
            parameter0: if matches!(algorithm, LlevTemporalAlgorithm::Msm) {
                1.0
            } else {
                0.0
            },
            parameter1: 0.0,
            band: if matches!(algorithm, LlevTemporalAlgorithm::Dtw) {
                2
            } else {
                0
            },
            cutoff: f64::INFINITY,
        },
        quant_min: -10.0,
        quant_max: 10.0,
        quant_bins: 32,
        reserved: 0,
        max_entries: 3,
        max_total_samples: 9,
        max_series_len: 3,
    }
}

fn limits() -> LlevTemporalSearchLimits {
    LlevTemporalSearchLimits {
        max_series_len: 3,
        max_dp_cells: 100_000,
        max_work_units: 100_000,
        max_scratch_bytes: 4 * 1024 * 1024,
        max_trie_nodes: 100_000,
        max_trie_edges: 100_000,
        max_candidates: 100_000,
        max_results: 3,
        max_queue_entries: 100_000,
        max_continuation_bytes: 4 * 1024 * 1024,
    }
}

fn create(config: &LlevTemporalIndexConfig) -> *mut LlevTemporalIndex {
    let mut index = ptr::null_mut();
    assert_eq!(
        unsafe { llev_temporal_index_new(config, &mut index) },
        LlevStatus::Ok
    );
    assert!(!index.is_null());
    index
}

fn insert(index: *mut LlevTemporalIndex, id: u64, values: &[f64]) -> LlevStatus {
    unsafe { llev_temporal_index_insert(index, id, values.as_ptr(), values.len()) }
}

fn query(
    index: *const LlevTemporalIndex,
    values: &[f64],
    cutoff: f64,
    limits: &LlevTemporalSearchLimits,
) -> (LlevStatus, *mut LlevTemporalIndexCursor) {
    let mut cursor = ptr::null_mut();
    let status = unsafe {
        llev_temporal_index_query_range(
            index,
            values.as_ptr(),
            values.len(),
            cutoff,
            limits,
            &mut cursor,
        )
    };
    (status, cursor)
}

fn query_erp_automaton(
    index: *const LlevTemporalIndex,
    values: &[f64],
    cutoff: f64,
    limits: &LlevTemporalSearchLimits,
) -> (LlevStatus, *mut LlevTemporalIndexCursor, u32) {
    let mut cursor = ptr::null_mut();
    let mut reason = u32::MAX;
    let status = unsafe {
        llev_temporal_index_query_erp_automaton_range(
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

fn collect(cursor: *mut LlevTemporalIndexCursor) -> (LlevStatus, Vec<LlevTemporalIndexMatch>) {
    let mut matches = Vec::new();
    for _ in 0..10_000 {
        let mut batch = [LlevTemporalIndexMatch::default(); 1];
        let mut len = usize::MAX;
        let mut done = u8::MAX;
        let mut reason = u32::MAX;
        let status = unsafe {
            llev_temporal_index_cursor_next_batch(
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
            assert_ne!(reason, 0);
            return (status, matches);
        }
        assert_eq!(reason, 0);
        matches.extend_from_slice(&batch[..len]);
        if done != 0 {
            return (status, matches);
        }
    }
    panic!("temporal cursor failed to make bounded progress");
}

fn scalar(algorithm: LlevTemporalAlgorithm, query: &[f64], candidate: &[f64]) -> f64 {
    match algorithm {
        LlevTemporalAlgorithm::Msm => MsmConfig::new(1.0).distance(query, candidate),
        LlevTemporalAlgorithm::Erp => ErpConfig::new(0.0).distance(query, candidate),
        LlevTemporalAlgorithm::Twed => TwedConfig::new(0.0, 0.0).distance(query, candidate),
        LlevTemporalAlgorithm::Dtw => DtwConfig::new(2).distance(query, candidate),
        LlevTemporalAlgorithm::Frechet => FrechetConfig::new().distance(query, candidate),
        LlevTemporalAlgorithm::SoftDtw => unreachable!(),
    }
}

#[test]
fn erp_automaton_pages_match_scalar_and_generic_range_after_index_close() {
    let index = create(&config(LlevTemporalAlgorithm::Erp));
    let query_values = [1.0, 2.0, 3.0];
    let candidates = [(7, [1.0, 2.0, 3.0]), (8, [1.0, 2.5, 3.0])];
    for (id, values) in candidates {
        assert_eq!(insert(index, id, &values), LlevStatus::Ok);
    }
    assert_eq!(unsafe { llev_temporal_index_freeze(index) }, LlevStatus::Ok);
    let (status, automaton, reason) = query_erp_automaton(index, &query_values, 1.0, &limits());
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(reason, 0);
    let (status, generic) = query(index, &query_values, 1.0, &limits());
    assert_eq!(status, LlevStatus::Ok);
    unsafe { llev_temporal_index_free(index) };
    let (status, mut observed) = collect(automaton);
    assert_eq!(status, LlevStatus::Ok);
    let (status, mut expected) = collect(generic);
    assert_eq!(status, LlevStatus::Ok);
    unsafe {
        llev_temporal_index_cursor_free(automaton);
        llev_temporal_index_cursor_free(generic);
    }
    observed.sort_by_key(|value| value.id);
    expected.sort_by_key(|value| value.id);
    assert_eq!(observed.len(), expected.len());
    for (actual, reference) in observed.iter().zip(expected.iter()) {
        assert_eq!(actual.id, reference.id);
        assert_eq!(actual.distance, reference.distance);
        let candidate = candidates.iter().find(|(id, _)| *id == actual.id).unwrap();
        assert_eq!(
            actual.distance,
            scalar(LlevTemporalAlgorithm::Erp, &query_values, &candidate.1)
        );
    }
}

#[test]
fn erp_automaton_rejects_wrong_domain_and_fails_closed_on_limits() {
    let other = create(&config(LlevTemporalAlgorithm::Msm));
    assert_eq!(unsafe { llev_temporal_index_freeze(other) }, LlevStatus::Ok);
    let (status, cursor, _) = query_erp_automaton(other, &[1.0], 1.0, &limits());
    assert_eq!(status, LlevStatus::Unsupported);
    assert!(cursor.is_null());
    unsafe { llev_temporal_index_free(other) };

    let index = create(&config(LlevTemporalAlgorithm::Erp));
    assert_eq!(insert(index, 7, &[1.0, 2.0, 3.0]), LlevStatus::Ok);
    let (status, cursor, _) = query_erp_automaton(index, &[1.0], 1.0, &limits());
    assert_eq!(status, LlevStatus::InvalidArgument);
    assert!(cursor.is_null());
    assert_eq!(unsafe { llev_temporal_index_freeze(index) }, LlevStatus::Ok);

    let mut restricted = limits();
    restricted.max_scratch_bytes = 0;
    let (status, cursor, reason) = query_erp_automaton(index, &[1.0], 1.0, &restricted);
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(reason, 3);
    assert!(cursor.is_null());
    let (status, cursor, _) = query_erp_automaton(index, &[f64::NAN], 1.0, &limits());
    assert_eq!(status, LlevStatus::InvalidArgument);
    assert!(cursor.is_null());
    unsafe { llev_temporal_index_free(index) };
}

#[test]
fn exact_knn_pages_all_five_kernels_after_index_close_and_fails_closed() {
    let query = [1.0, 2.0, 3.0];
    let candidates = [(7, [1.0, 2.0, 3.0]), (8, [1.0, 2.5, 3.0])];
    for algorithm in [
        LlevTemporalAlgorithm::Msm,
        LlevTemporalAlgorithm::Erp,
        LlevTemporalAlgorithm::Twed,
        LlevTemporalAlgorithm::Dtw,
        LlevTemporalAlgorithm::Frechet,
    ] {
        let index = create(&config(algorithm));
        for (id, values) in candidates {
            assert_eq!(insert(index, id, &values), LlevStatus::Ok);
        }
        assert_eq!(unsafe { llev_temporal_index_freeze(index) }, LlevStatus::Ok);
        let mut restricted = limits();
        restricted.max_candidates = 0;
        let mut cursor = ptr::null_mut();
        let mut reason = u32::MAX;
        assert_eq!(
            unsafe {
                llev_temporal_index_query_knn(
                    index,
                    query.as_ptr(),
                    query.len(),
                    2,
                    &restricted,
                    &mut cursor,
                    &mut reason,
                )
            },
            LlevStatus::LimitExceeded
        );
        assert_eq!(reason, 6);
        assert!(cursor.is_null());
        assert_eq!(
            unsafe {
                llev_temporal_index_query_knn(
                    index,
                    query.as_ptr(),
                    query.len(),
                    2,
                    &limits(),
                    &mut cursor,
                    &mut reason,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(reason, 0);
        assert!(!cursor.is_null());
        unsafe { llev_temporal_index_free(index) };
        for (id, values) in candidates {
            let mut output = [LlevTemporalIndexMatch::default(); 1];
            let mut written = usize::MAX;
            let mut done = u8::MAX;
            assert_eq!(
                unsafe {
                    llev_temporal_knn_cursor_next_batch(
                        cursor,
                        output.as_mut_ptr(),
                        1,
                        &mut written,
                        &mut done,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(written, 1);
            assert_eq!(output[0].id, id);
            assert_eq!(output[0].distance, scalar(algorithm, &query, &values));
            assert_eq!(done, u8::from(id == 8));
        }
        unsafe { llev_temporal_knn_cursor_free(cursor) };
    }
}

#[test]
fn frozen_index_pages_match_all_five_native_kernels_and_outlive_handle() {
    let query_values = [1.0, 2.0, 3.0];
    let candidates = [(10, [1.0, 2.0, 3.0]), (20, [1.0, 2.5, 3.0])];
    for algorithm in [
        LlevTemporalAlgorithm::Msm,
        LlevTemporalAlgorithm::Erp,
        LlevTemporalAlgorithm::Twed,
        LlevTemporalAlgorithm::Dtw,
        LlevTemporalAlgorithm::Frechet,
    ] {
        let index = create(&config(algorithm));
        for (id, values) in candidates {
            assert_eq!(insert(index, id, &values), LlevStatus::Ok);
        }
        assert_eq!(unsafe { llev_temporal_index_freeze(index) }, LlevStatus::Ok);
        let (status, cursor) = query(index, &query_values, 100.0, &limits());
        assert_eq!(status, LlevStatus::Ok);
        unsafe { llev_temporal_index_free(index) };
        let (status, matches) = collect(cursor);
        unsafe { llev_temporal_index_cursor_free(cursor) };
        assert_eq!(status, LlevStatus::Ok, "{algorithm:?}");
        assert_eq!(matches.len(), candidates.len(), "{algorithm:?}");
        for (id, candidate) in candidates {
            let matched = matches.iter().find(|value| value.id == id).unwrap();
            let expected = scalar(algorithm, &query_values, &candidate);
            assert!(
                (matched.distance - expected).abs() < 1e-10,
                "{algorithm:?}: {} != {expected}",
                matched.distance
            );
        }
    }
}

#[test]
fn source_and_query_limits_fail_closed_without_losing_snapshot() {
    let index = create(&config(LlevTemporalAlgorithm::Msm));
    assert_eq!(insert(index, 1, &[1.0, 2.0]), LlevStatus::Ok);
    assert_eq!(insert(index, 1, &[1.0, 2.0, 3.0]), LlevStatus::Ok);
    assert_eq!(insert(index, 2, &[4.0, 5.0, 6.0]), LlevStatus::Ok);
    assert_eq!(insert(index, 3, &[7.0, 8.0, 9.0]), LlevStatus::Ok);
    assert_eq!(insert(index, 4, &[0.0]), LlevStatus::LimitExceeded);
    assert_eq!(insert(index, 2, &[f64::NAN]), LlevStatus::InvalidArgument);
    assert_eq!(unsafe { llev_temporal_index_freeze(index) }, LlevStatus::Ok);
    assert_eq!(insert(index, 4, &[0.0]), LlevStatus::Closed);

    let (status, cursor) = query(index, &[1.0, 2.0, 3.0], 0.0, &limits());
    assert_eq!(status, LlevStatus::Ok);
    let (status, matches) = collect(cursor);
    assert_eq!(status, LlevStatus::Ok);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].id, 1);
    unsafe { llev_temporal_index_cursor_free(cursor) };

    let restricted = LlevTemporalSearchLimits {
        max_work_units: 0,
        ..limits()
    };
    let (status, cursor) = query(index, &[1.0, 2.0, 3.0], 10.0, &restricted);
    assert_eq!(status, LlevStatus::Ok);
    let mut batch = [LlevTemporalIndexMatch::default(); 1];
    let mut len = usize::MAX;
    let mut done = u8::MAX;
    let mut reason = u32::MAX;
    let status = unsafe {
        llev_temporal_index_cursor_next_batch(
            cursor,
            batch.as_mut_ptr(),
            1,
            1_000,
            1,
            &mut len,
            &mut done,
            &mut reason,
        )
    };
    assert_eq!(status, LlevStatus::LimitExceeded);
    assert_eq!(reason, 2);
    assert_eq!(len, 0);
    assert_eq!(done, 0);
    unsafe {
        llev_temporal_index_cursor_free(cursor);
        llev_temporal_index_free(index);
    }
}

#[test]
fn independent_cursors_share_one_frozen_snapshot_concurrently() {
    let index = create(&config(LlevTemporalAlgorithm::Erp));
    assert_eq!(insert(index, 7, &[1.0, 2.0, 3.0]), LlevStatus::Ok);
    assert_eq!(unsafe { llev_temporal_index_freeze(index) }, LlevStatus::Ok);
    let address = index as usize;
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(move || {
                    let (status, cursor) = query(
                        address as *const LlevTemporalIndex,
                        &[1.0, 2.0, 3.0],
                        0.0,
                        &limits(),
                    );
                    assert_eq!(status, LlevStatus::Ok);
                    let (status, values) = collect(cursor);
                    unsafe { llev_temporal_index_cursor_free(cursor) };
                    assert_eq!(status, LlevStatus::Ok);
                    assert_eq!(values.len(), 1);
                    assert_eq!(values[0].id, 7);
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    unsafe { llev_temporal_index_free(index) };
}
