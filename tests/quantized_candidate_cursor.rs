use liblevenshtein::time_series::{
    IncompleteReason, PageBudget, QuantizationConfig, ResourceKind, ResourceLimits, TimeSeriesIndex,
};
use liblevenshtein::transducer::Algorithm;

fn collect_bounded(
    index: &TimeSeriesIndex<u64>,
    query: &[f64],
    threshold: usize,
    algorithm: Algorithm,
) -> Vec<(u64, usize)> {
    let mut cursor = index
        .search_quantized_bounded(query, threshold, algorithm, ResourceLimits::default())
        .unwrap();
    let mut matches = Vec::new();
    for _ in 0..100 {
        let page = cursor
            .next_page(PageBudget {
                max_work_units: 100,
                max_results: 1,
            })
            .unwrap();
        matches.extend(page.matches);
        if page.done {
            return matches;
        }
    }
    panic!("bounded candidate cursor failed to finish");
}

#[test]
fn bounded_pages_match_all_three_legacy_quantized_search_families() {
    let config = QuantizationConfig::uniform(0.0, 10.0, 5);
    let mut index = TimeSeriesIndex::<u64>::new(config);
    index.insert(7, &[1.0, 3.0, 5.0]);
    index.insert(8, &[1.1, 3.1, 5.1]);
    index.insert(9, &[3.0, 1.0, 5.0]);
    index.insert(10, &[1.0, 3.0, 7.0]);
    index.insert(11, &[1.0, 5.0]);
    index.insert(10, &[9.0, 9.0, 9.0]);
    index.remove(11);
    let query = [1.05, 3.05, 5.05];
    for algorithm in [
        Algorithm::Standard,
        Algorithm::Transposition,
        Algorithm::MergeAndSplit,
    ] {
        for threshold in 0..=3 {
            let mut expected = match algorithm {
                Algorithm::Standard => index.search(&query, threshold),
                Algorithm::Transposition => index.search_transposition(&query, threshold),
                Algorithm::MergeAndSplit => index.search_merge_split(&query, threshold),
                _ => unreachable!(),
            };
            let mut observed = collect_bounded(&index, &query, threshold, algorithm);
            expected.sort_unstable();
            observed.sort_unstable();
            assert_eq!(observed, expected, "{algorithm:?} threshold={threshold}");
        }
    }
}

#[test]
fn quantized_candidate_pages_match_legacy_on_short_word_space() {
    let mut index = TimeSeriesIndex::<u64>::new(QuantizationConfig::uniform(0.0, 3.0, 3));
    let mut words = vec![Vec::new()];
    for a in 0..3 {
        words.push(vec![f64::from(a) + 0.5]);
        for b in 0..3 {
            words.push(vec![f64::from(a) + 0.5, f64::from(b) + 0.5]);
            for c in 0..3 {
                words.push(vec![
                    f64::from(a) + 0.5,
                    f64::from(b) + 0.5,
                    f64::from(c) + 0.5,
                ]);
            }
        }
    }
    for (id, word) in words.iter().enumerate() {
        index.insert(id as u64, word);
    }
    for query in words.iter().step_by(4) {
        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
        ] {
            for threshold in 0..=3 {
                let mut expected = match algorithm {
                    Algorithm::Standard => index.search(query, threshold),
                    Algorithm::Transposition => index.search_transposition(query, threshold),
                    Algorithm::MergeAndSplit => index.search_merge_split(query, threshold),
                    _ => unreachable!(),
                };
                let mut observed = collect_bounded(&index, query, threshold, algorithm);
                expected.sort_unstable();
                observed.sort_unstable();
                assert_eq!(
                    observed, expected,
                    "{algorithm:?} query={query:?} threshold={threshold}"
                );
            }
        }
    }
}

#[test]
fn candidate_limits_fail_without_claiming_a_complete_set() {
    let mut index = TimeSeriesIndex::<u64>::new(QuantizationConfig::uniform(0.0, 10.0, 5));
    index.insert(1, &[1.0, 3.0]);
    index.insert(2, &[1.0, 3.0]);
    index.insert(3, &[1.0, 3.0, 5.0]);
    let mut limited = ResourceLimits {
        max_results: 1,
        ..ResourceLimits::default()
    };
    let mut cursor = index
        .search_quantized_bounded(&[1.0, 3.0], 0, Algorithm::Standard, limited)
        .unwrap();
    let first = cursor
        .next_page(PageBudget {
            max_work_units: 100,
            max_results: 1,
        })
        .unwrap();
    assert_eq!(first.matches.len(), 1);
    assert!(!first.done);
    assert!(matches!(
        cursor.next_page(PageBudget {
            max_work_units: 100,
            max_results: 1,
        }),
        Err(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::Results,
            ..
        })
    ));

    limited.max_results = 10;
    limited.max_dp_cells = 0;
    let mut cursor = index
        .search_quantized_bounded(&[1.0, 3.0], 1, Algorithm::Standard, limited)
        .unwrap();
    assert!(matches!(
        cursor.next_page(PageBudget {
            max_work_units: 100,
            max_results: 10,
        }),
        Err(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::DpCells,
            ..
        })
    ));
}

#[test]
fn page_too_small_can_be_retried_and_source_length_is_checked() {
    let mut index = TimeSeriesIndex::<u64>::new(QuantizationConfig::uniform(0.0, 10.0, 5));
    index.insert(1, &[1.0, 3.0]);
    index.insert(2, &[1.0, 3.0, 5.0]);
    let mut cursor = index
        .search_quantized_bounded(
            &[1.0, 3.0],
            1,
            Algorithm::Standard,
            ResourceLimits {
                max_series_len: 2,
                ..ResourceLimits::default()
            },
        )
        .unwrap();
    assert!(matches!(
        cursor.next_page(PageBudget {
            max_work_units: 1,
            max_results: 1,
        }),
        Err(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::WorkUnits,
            ..
        })
    ));
    let first = cursor
        .next_page(PageBudget {
            max_work_units: 100,
            max_results: 1,
        })
        .unwrap();
    assert_eq!(first.matches, vec![(1, 0)]);
    assert!(!first.done);
    assert!(matches!(
        cursor.next_page(PageBudget {
            max_work_units: 100,
            max_results: 1,
        }),
        Err(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::SeriesLength,
            ..
        })
    ));
}

#[test]
fn zero_threshold_uses_bounded_exact_lookup_without_scanning_other_keys() {
    let mut index = TimeSeriesIndex::<u64>::new(QuantizationConfig::uniform(0.0, 10.0, 5));
    index.insert(1, &[1.0, 3.0]);
    index.insert(2, &[1.0, 3.0, 5.0]);
    let mut cursor = index
        .search_quantized_bounded(
            &[1.0, 3.0],
            0,
            Algorithm::Standard,
            ResourceLimits {
                max_series_len: 2,
                max_dp_cells: 0,
                max_work_units: 4,
                ..ResourceLimits::default()
            },
        )
        .unwrap();
    let first = cursor
        .next_page(PageBudget {
            max_work_units: 1,
            max_results: 1,
        })
        .unwrap();
    assert_eq!(first.matches, vec![(1, 0)]);
    assert!(!first.done);
    let final_page = cursor
        .next_page(PageBudget {
            max_work_units: 1,
            max_results: 1,
        })
        .unwrap();
    assert!(final_page.matches.is_empty());
    assert!(final_page.done);
}
