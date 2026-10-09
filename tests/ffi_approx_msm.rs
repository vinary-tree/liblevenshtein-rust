#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_approx_msm_index_free, llev_approx_msm_index_freeze, llev_approx_msm_index_insert,
    llev_approx_msm_index_new, llev_approx_msm_index_query_knn, LlevApproxMsmIndex,
    LlevApproxMsmIndexConfig, LlevApproxMsmNeighbor, LlevApproxMsmOutcome, LlevStatus,
    LlevTemporalSearchLimits,
};

fn config(candidate_limit: usize) -> LlevApproxMsmIndexConfig {
    LlevApproxMsmIndexConfig {
        segments: 1,
        candidate_limit,
        split_merge_cost: 1.0,
        max_entries: 3,
        max_total_samples: 3,
        max_series_len: 1,
        max_total_features: 3,
    }
}

fn limits() -> LlevTemporalSearchLimits {
    LlevTemporalSearchLimits {
        max_series_len: 1,
        max_dp_cells: 1000,
        max_work_units: 1000,
        max_scratch_bytes: 1000,
        max_trie_nodes: 1000,
        max_trie_edges: 1000,
        max_candidates: 3,
        max_results: 3,
        max_queue_entries: 1000,
        max_continuation_bytes: 1000,
    }
}

fn index(candidate_limit: usize) -> *mut LlevApproxMsmIndex {
    let mut index = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_approx_msm_index_new(&config(candidate_limit), &mut index) },
        LlevStatus::Ok
    );
    for (position, (id, sample)) in [(7, 1.0), (7, 2.0), (11, 3.0)].into_iter().enumerate() {
        let mut out_position = usize::MAX;
        assert_eq!(
            unsafe { llev_approx_msm_index_insert(index, id, &sample, 1, &mut out_position) },
            LlevStatus::Ok
        );
        assert_eq!(out_position, position);
    }
    assert_eq!(
        unsafe { llev_approx_msm_index_freeze(index) },
        LlevStatus::Ok
    );
    index
}

#[test]
fn exhaustive_and_advisory_outcomes_preserve_exact_distances_and_coverage() {
    for (candidate_limit, expected_kind, expected_coverage) in [(3, 1, 3), (1, 2, 1)] {
        let index = index(candidate_limit);
        let query = [1.0];
        let mut neighbors = [LlevApproxMsmNeighbor::default(); 1];
        let mut outcome = LlevApproxMsmOutcome::default();
        assert_eq!(
            unsafe {
                llev_approx_msm_index_query_knn(
                    index,
                    query.as_ptr(),
                    query.len(),
                    1,
                    &limits(),
                    neighbors.as_mut_ptr(),
                    neighbors.len(),
                    &mut outcome,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(outcome.kind, expected_kind);
        assert_eq!(outcome.indexed_entries, 3);
        assert_eq!(outcome.candidate_entries, expected_coverage);
        assert_eq!(outcome.exact_reranked, expected_coverage);
        assert_eq!(outcome.neighbor_count, 1);
        assert_eq!(neighbors[0].id, 7);
        assert_eq!(neighbors[0].insertion_index, 0);
        assert_eq!(neighbors[0].distance, 0.0);
        unsafe { llev_approx_msm_index_free(index) };
    }
}

#[test]
fn bounded_approximate_msm_distinguishes_empty_advice_and_incompletion() {
    let index = index(1);
    let query = [1.0];
    let mut outcome = LlevApproxMsmOutcome::default();
    assert_eq!(
        unsafe {
            llev_approx_msm_index_query_knn(
                index,
                query.as_ptr(),
                1,
                0,
                &limits(),
                std::ptr::null_mut(),
                0,
                &mut outcome,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(outcome.kind, 2);
    assert_eq!(outcome.neighbor_count, 0);
    let mut restricted = limits();
    restricted.max_dp_cells = 0;
    let mut neighbors = [LlevApproxMsmNeighbor::default(); 1];
    assert_eq!(
        unsafe {
            llev_approx_msm_index_query_knn(
                index,
                query.as_ptr(),
                1,
                1,
                &restricted,
                neighbors.as_mut_ptr(),
                1,
                &mut outcome,
            )
        },
        LlevStatus::Ok
    );
    assert_eq!(outcome.kind, 3);
    assert_eq!(outcome.reason, 1);
    assert_eq!(outcome.indexed_entries, 3);
    assert_eq!(outcome.neighbor_count, 0);
    let mut position = usize::MAX;
    assert_eq!(
        unsafe { llev_approx_msm_index_insert(index, 13, query.as_ptr(), 1, &mut position) },
        LlevStatus::InvalidArgument
    );
    unsafe { llev_approx_msm_index_free(index) };
}
