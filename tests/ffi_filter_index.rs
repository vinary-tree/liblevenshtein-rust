#![cfg(feature = "ffi")]

use liblevenshtein::ffi::{
    llev_source_filter_index_free, llev_source_filter_index_freeze,
    llev_source_filter_index_insert, llev_source_filter_index_new, llev_source_filter_index_query,
    LlevSourceFilterIndex, LlevSourceFilterIndexConfig, LlevSourceFilterIndexLimits, LlevStatus,
};
use liblevenshtein::filter::{HybridMatcher, NgramIndex};

fn config(mode: u32) -> LlevSourceFilterIndexConfig {
    LlevSourceFilterIndexConfig {
        mode,
        reserved: 0,
        ngram_size: 2,
        jaro_threshold: if mode == 1 { 0.0 } else { 0.7 },
        max_terms: 4,
        max_term_bytes: 10,
        max_source_bytes: 32,
        max_query_bytes: 10,
    }
}

fn limits() -> LlevSourceFilterIndexLimits {
    LlevSourceFilterIndexLimits {
        max_candidates: 4,
        max_results: 4,
        max_comparisons: 1000,
    }
}

#[test]
fn persistent_ngram_and_hybrid_match_native_candidates_in_source_order() {
    let terms = ["hello", "help", "world", "héllo"];
    for mode in [1, 2] {
        let mut index: *mut LlevSourceFilterIndex = std::ptr::null_mut();
        assert_eq!(
            unsafe { llev_source_filter_index_new(&config(mode), &mut index) },
            LlevStatus::Ok
        );
        for (position, term) in terms.iter().enumerate() {
            let mut id = usize::MAX;
            assert_eq!(
                unsafe {
                    llev_source_filter_index_insert(
                        index,
                        term.as_ptr().cast(),
                        term.len(),
                        &mut id,
                    )
                },
                LlevStatus::Ok
            );
            assert_eq!(id, position);
        }
        let mut duplicate = usize::MAX;
        assert_eq!(
            unsafe {
                llev_source_filter_index_insert(
                    index,
                    terms[0].as_ptr().cast(),
                    terms[0].len(),
                    &mut duplicate,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(duplicate, 0);
        assert_eq!(
            unsafe { llev_source_filter_index_freeze(index) },
            LlevStatus::Ok
        );
        let query = "helo";
        let mut out = [usize::MAX; 4];
        let mut len = usize::MAX;
        let mut reason = u32::MAX;
        assert_eq!(
            unsafe {
                llev_source_filter_index_query(
                    index,
                    query.as_ptr().cast(),
                    query.len(),
                    1,
                    &limits(),
                    out.as_mut_ptr(),
                    out.len(),
                    &mut len,
                    &mut reason,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(reason, 0);
        let expected = if mode == 1 {
            NgramIndex::from_iter(2, terms.iter().map(|term| (*term).to_owned()))
                .find_candidates(query, 1)
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        } else {
            HybridMatcher::with_config(terms.iter().map(|term| (*term).to_owned()), 2, 0.7)
                .filter_candidates(query, 1)
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        let actual = out[..len]
            .iter()
            .map(|&id| terms[id].to_owned())
            .collect::<Vec<_>>();
        let expected = terms
            .iter()
            .filter(|term| expected.contains(&term.to_string()))
            .map(|term| (*term).to_owned())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(
            unsafe {
                llev_source_filter_index_insert(index, "late".as_ptr().cast(), 4, &mut duplicate)
            },
            LlevStatus::InvalidArgument
        );
        unsafe { llev_source_filter_index_free(index) };
        assert_eq!(actual, expected);
    }
}

#[test]
fn persistent_filter_limits_fail_without_publishing_partial_ids() {
    let mut index: *mut LlevSourceFilterIndex = std::ptr::null_mut();
    assert_eq!(
        unsafe { llev_source_filter_index_new(&config(2), &mut index) },
        LlevStatus::Ok
    );
    let mut id = 0;
    for term in ["hello", "help"] {
        assert_eq!(
            unsafe {
                llev_source_filter_index_insert(index, term.as_ptr().cast(), term.len(), &mut id)
            },
            LlevStatus::Ok
        );
    }
    assert_eq!(
        unsafe { llev_source_filter_index_freeze(index) },
        LlevStatus::Ok
    );
    let query = "helo";
    let mut out = [usize::MAX; 2];
    let mut len = usize::MAX;
    let mut reason = u32::MAX;
    let cases = [
        (
            LlevSourceFilterIndexLimits {
                max_candidates: 1,
                ..limits()
            },
            1,
        ),
        (
            LlevSourceFilterIndexLimits {
                max_results: 0,
                ..limits()
            },
            2,
        ),
        (
            LlevSourceFilterIndexLimits {
                max_comparisons: 0,
                ..limits()
            },
            4,
        ),
    ];
    for (limit, expected_reason) in cases {
        assert_eq!(
            unsafe {
                llev_source_filter_index_query(
                    index,
                    query.as_ptr().cast(),
                    query.len(),
                    1,
                    &limit,
                    out.as_mut_ptr(),
                    out.len(),
                    &mut len,
                    &mut reason,
                )
            },
            LlevStatus::LimitExceeded
        );
        assert_eq!(reason, expected_reason);
        assert_eq!(len, 0);
        assert_eq!(out, [usize::MAX; 2]);
    }
    unsafe { llev_source_filter_index_free(index) };
}
