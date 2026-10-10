#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_timestamped_twed_cursor_free, llev_timestamped_twed_cursor_next_batch,
    llev_timestamped_twed_index_free, llev_timestamped_twed_index_freeze,
    llev_timestamped_twed_index_insert, llev_timestamped_twed_index_new,
    llev_timestamped_twed_index_query_knn, llev_timestamped_twed_index_query_range, LlevStatus,
    LlevTemporalSearchLimits, LlevTimestampedSeriesView, LlevTimestampedTwedCursor,
    LlevTimestampedTwedIndex, LlevTimestampedTwedIndexConfig, LlevTimestampedTwedMatch,
    LlevTimestampedTwedSearchLimits,
};
use liblevenshtein::time_series::{
    MetricTimestampedTwedConfig, OperationOutcome, PageBudget, ResourceLimits, TimestampUnit,
    TimestampedSeries, TimestampedTwedIndex, TimestampedTwedProductLimits,
    TimestampedTwedQuantizer,
};

fn config() -> LlevTimestampedTwedIndexConfig {
    LlevTimestampedTwedIndexConfig {
        unit: 2,
        reserved: 0,
        origin: 10.0,
        value_min: 0.0,
        value_max: 5.0,
        time_min: 10.0,
        time_max: 20.0,
        value_bins: 1,
        time_bins: 1,
        stiffness: 0.5,
        gap_penalty: 1.0,
        max_entries: 3,
        max_total_samples: 6,
        max_series_len: 2,
    }
}

fn view(values: &[f64], times: &[f64]) -> LlevTimestampedSeriesView {
    LlevTimestampedSeriesView {
        values: values.as_ptr(),
        timestamps: times.as_ptr(),
        len: values.len(),
        unit: 2,
        reserved: 0,
        origin: 10.0,
    }
}

fn limits() -> LlevTimestampedTwedSearchLimits {
    LlevTimestampedTwedSearchLimits {
        common: LlevTemporalSearchLimits {
            max_series_len: 2,
            max_dp_cells: 1_000_000,
            max_work_units: 1_000_000,
            max_scratch_bytes: 1_000_000,
            max_trie_nodes: 1_000_000,
            max_trie_edges: 1_000_000,
            max_candidates: 100,
            max_results: 100,
            max_queue_entries: 1_000_000,
            max_continuation_bytes: 1_000_000,
        },
        max_product_states: 1_000_000,
        max_product_positions: 1_000_000,
        max_transition_cache_entries: 1_000_000,
    }
}

#[test]
fn frozen_timestamped_range_cursor_retains_full_precision_episodes() {
    let mut index: *mut LlevTimestampedTwedIndex = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_timestamped_twed_index_new(&config(), &mut index) },
        LlevStatus::Ok
    );
    let first_values = [1.0, 2.0];
    let second_values = [1.0, 2.0];
    let third_values = [3.0, 4.0];
    let first_times = [10.0, 13.0];
    let second_times = [10.0, 13.0];
    let third_times = [10.0, 14.0];
    let mut episode = u64::MAX;
    assert_eq!(
        unsafe {
            llev_timestamped_twed_index_insert(
                index,
                7,
                &view(&first_values, &first_times),
                &mut episode,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(episode, 0);
    assert_eq!(
        unsafe {
            llev_timestamped_twed_index_insert(
                index,
                7,
                &view(&second_values, &second_times),
                &mut episode,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(episode, 1);
    assert_eq!(
        unsafe {
            llev_timestamped_twed_index_insert(
                index,
                11,
                &view(&third_values, &third_times),
                &mut episode,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(episode, 2);
    assert_eq!(
        unsafe { llev_timestamped_twed_index_freeze(index) },
        LlevStatus::Ok
    );
    let query_values = [1.0, 2.0];
    let query_times = [10.0, 13.0];
    let mut cursor: *mut LlevTimestampedTwedCursor = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_timestamped_twed_index_query_range(
                index,
                &view(&query_values, &query_times),
                0.0,
                &limits(),
                &mut cursor,
            )
        },
        LlevStatus::Ok
    );
    unsafe { llev_timestamped_twed_index_free(index) };
    let mut found = Vec::new();
    let mut done = 0;
    for _ in 0..100 {
        let mut out = [LlevTimestampedTwedMatch::default(); 1];
        let mut len = 0;
        let mut reason = 0;
        let status = unsafe {
            llev_timestamped_twed_cursor_next_batch(
                cursor,
                out.as_mut_ptr(),
                1,
                100_000,
                1,
                &mut len,
                &mut done,
                &mut reason,
            )
        };
        assert_eq!(status, LlevStatus::Ok, "reason={reason}");
        assert_eq!(reason, 0);
        found.extend_from_slice(&out[..len]);
        if done != 0 {
            break;
        }
    }
    unsafe { llev_timestamped_twed_cursor_free(cursor) };
    assert_eq!(done, 1);
    found.sort_by_key(|matched| matched.episode_id);
    assert_eq!(found.len(), 2);
    assert_eq!(
        found.iter().map(|matched| matched.id).collect::<Vec<_>>(),
        [7, 7]
    );
    assert_eq!(
        found
            .iter()
            .map(|matched| matched.episode_id)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert!(found.iter().all(|matched| matched.distance == 0.0));
}

#[test]
fn exact_knn_is_sorted_and_fails_closed_on_resource_exhaustion() {
    let mut index: *mut LlevTimestampedTwedIndex = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_timestamped_twed_index_new(&config(), &mut index) },
        LlevStatus::Ok
    );
    let times = [10.0, 13.0];
    let mut episode = u64::MAX;
    for (id, values) in [(7, [1.0, 2.0]), (7, [1.0, 2.0]), (11, [3.0, 4.0])] {
        assert_eq!(
            unsafe {
                llev_timestamped_twed_index_insert(index, id, &view(&values, &times), &mut episode)
            },
            LlevStatus::Ok
        );
    }
    assert_eq!(
        unsafe { llev_timestamped_twed_index_freeze(index) },
        LlevStatus::Ok
    );
    let query_values = [1.0, 2.0];
    let query = view(&query_values, &times);
    let mut out = [LlevTimestampedTwedMatch::default(); 2];
    let mut len = usize::MAX;
    let mut reason = u32::MAX;
    assert_eq!(
        unsafe {
            llev_timestamped_twed_index_query_knn(
                index,
                &query,
                2,
                &limits().common,
                out.as_mut_ptr(),
                out.len(),
                &mut len,
                &mut reason,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(len, 2);
    assert_eq!(reason, 0);
    assert_eq!(out.map(|matched| matched.id), [7, 7]);
    assert_eq!(out.map(|matched| matched.episode_id), [0, 1]);
    assert_eq!(out.map(|matched| matched.distance), [0.0, 0.0]);

    let mut restricted = limits().common;
    restricted.max_work_units = 0;
    let previous = out;
    assert_eq!(
        unsafe {
            llev_timestamped_twed_index_query_knn(
                index,
                &query,
                2,
                &restricted,
                out.as_mut_ptr(),
                out.len(),
                &mut len,
                &mut reason,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(len, 0);
    assert_ne!(reason, 0);
    assert_eq!(
        out.map(|matched| matched.episode_id),
        previous.map(|matched| matched.episode_id)
    );
    assert_eq!(
        unsafe {
            llev_timestamped_twed_index_query_knn(
                index,
                &query,
                2,
                &limits().common,
                out.as_mut_ptr(),
                1,
                &mut len,
                &mut reason,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(len, 0);
    unsafe { llev_timestamped_twed_index_free(index) };
}

#[test]
fn native_timestamped_one_work_unit_start_can_resume() {
    let quantizer = TimestampedTwedQuantizer::try_new(
        TimestampUnit::Milliseconds,
        10.0,
        (0.0, 5.0),
        (10.0, 20.0),
        1,
        1,
    )
    .unwrap();
    let metric = MetricTimestampedTwedConfig::try_new(0.5, 1.0).unwrap();
    let series = TimestampedSeries::try_new_with_origin(
        &[1.0, 2.0],
        &[11.0, 13.0],
        TimestampUnit::Milliseconds,
        10.0,
        ResourceLimits::default(),
    )
    .unwrap();
    let mut index = TimestampedTwedIndex::new(quantizer, metric);
    index.insert(7, series.clone()).unwrap();
    let full = index
        .search_range_bounded(
            &series,
            0.0,
            TimestampedTwedProductLimits::default(),
            PageBudget::default(),
        )
        .unwrap();
    assert!(
        matches!(full, OperationOutcome::Complete { .. }),
        "full: {full:?}"
    );
    let mut result = index
        .search_range_bounded(
            &series,
            0.0,
            TimestampedTwedProductLimits::default(),
            PageBudget {
                max_work_units: 1,
                max_results: 1,
            },
        )
        .unwrap();
    for _ in 0..10 {
        result = match result {
            OperationOutcome::Incomplete {
                continuation: Some(cursor),
                ..
            } => cursor.resume(PageBudget {
                max_work_units: 100_000,
                max_results: 1,
            }),
            other => {
                assert!(
                    matches!(other, OperationOutcome::Complete { .. }),
                    "{other:?}"
                );
                return;
            }
        };
    }
    panic!("native timestamped cursor did not complete");
}

#[test]
fn timestamped_index_rejects_domain_mutation_and_cumulative_result_limit() {
    let mut index: *mut LlevTimestampedTwedIndex = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_timestamped_twed_index_new(&config(), &mut index) },
        LlevStatus::Ok
    );
    let values = [1.0, 2.0];
    let times = [10.0, 13.0];
    let mut episode = u64::MAX;
    let mut invalid = view(&values, &times);
    invalid.unit = 1;
    assert_eq!(
        unsafe { llev_timestamped_twed_index_insert(index, 7, &invalid, &mut episode) },
        LlevStatus::DomainMismatch
    );
    assert_eq!(
        unsafe {
            llev_timestamped_twed_index_insert(index, 7, &view(&values, &times), &mut episode)
        },
        LlevStatus::Ok
    );
    assert_eq!(
        unsafe { llev_timestamped_twed_index_freeze(index) },
        LlevStatus::Ok
    );
    assert_eq!(
        unsafe {
            llev_timestamped_twed_index_insert(index, 9, &view(&values, &times), &mut episode)
        },
        LlevStatus::InvalidArgument
    );
    let mut restricted = limits();
    restricted.common.max_results = 0;
    let mut cursor: *mut LlevTimestampedTwedCursor = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_timestamped_twed_index_query_range(
                index,
                &view(&values, &times),
                0.0,
                &restricted,
                &mut cursor,
            )
        },
        LlevStatus::Ok
    );
    let mut out = [LlevTimestampedTwedMatch::default(); 1];
    let mut len = 0;
    let mut done = 0;
    let mut reason = 0;
    assert_eq!(
        unsafe {
            llev_timestamped_twed_cursor_next_batch(
                cursor,
                out.as_mut_ptr(),
                1,
                100_000,
                1,
                &mut len,
                &mut done,
                &mut reason,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert_eq!(len, 0);
    assert_eq!(reason, 7);
    unsafe {
        llev_timestamped_twed_cursor_free(cursor);
        llev_timestamped_twed_index_free(index);
    }
}
