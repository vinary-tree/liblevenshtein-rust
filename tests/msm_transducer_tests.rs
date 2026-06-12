//! Integration tests for the exact MSM-over-trie [`MsmTransducer`].
//!
//! These exercise the public re-export `liblevenshtein::time_series::MsmTransducer`
//! against a brute-force MSM oracle, covering edge cases the in-module unit tests
//! do not: empty inputs, single-element series, length mismatch, `k > len`, the
//! inclusive threshold boundary, out-of-range values that fold into the ±∞ extreme
//! quantization bins, a differential cross-check against
//! [`HybridSearchIndex::search_exact`] (which re-verifies candidates exactly but
//! whose lossy pre-filter can miss true neighbors, so the transducer's exact set is
//! a complete superset of it), and concurrent read-only queries (documents `Sync`;
//! no `loom` needed because the transducer is immutable after construction).

use std::cmp::Ordering;
use std::collections::HashSet;

use liblevenshtein::time_series::{
    HybridSearchIndex, MsmConfig, MsmTransducer, QuantizationConfig,
};
use proptest::prelude::*;

const EPS: f64 = 1e-9;

/// Brute-force reference: every series within `tau`, sorted ascending by distance.
fn brute_range(series: &[Vec<f64>], query: &[f64], msm: &MsmConfig, tau: f64) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = series
        .iter()
        .enumerate()
        .map(|(i, s)| (i, msm.distance(query, s)))
        .filter(|(_, d)| *d <= tau + EPS)
        .collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    v
}

fn ids(results: &[(usize, f64)]) -> HashSet<usize> {
    results.iter().map(|(i, _)| *i).collect()
}

// ---------------------------------------------------------------------------
// Public-API smoke
// ---------------------------------------------------------------------------

#[test]
fn public_api_smoke_range_and_knn() {
    let series = vec![
        vec![10.0, 20.0, 30.0],
        vec![11.0, 21.0, 29.0],
        vec![50.0, 60.0, 70.0],
    ];
    let quant = QuantizationConfig::for_u8(0.0, 100.0);
    let msm = MsmConfig::new(1.0);
    let idx = MsmTransducer::from_series(quant, msm, &series);
    assert_eq!(idx.len(), 3);
    assert!(!idx.is_empty());

    let query = vec![12.0, 22.0, 31.0];
    let range = idx.search_range(&query, 10.0);
    assert_eq!(ids(&range), ids(&brute_range(&series, &query, &msm, 10.0)));

    let knn = idx.search_knn(&query, 2, 1.0);
    assert_eq!(knn.len(), 2);
    // Returned ascending and exact.
    assert!(knn[0].1 <= knn[1].1 + EPS);
    for (i, d) in &knn {
        assert!((d - msm.distance(&query, &series[*i])).abs() < EPS);
    }
}

// ---------------------------------------------------------------------------
// Empty inputs
// ---------------------------------------------------------------------------

#[test]
fn empty_query_returns_empty_when_only_non_empty_references_exist() {
    let series = vec![vec![1.0, 2.0, 3.0]];
    let idx = MsmTransducer::from_series(
        QuantizationConfig::for_u8(0.0, 10.0),
        MsmConfig::new(1.0),
        &series,
    );
    assert!(idx.search_range(&[], 100.0).is_empty());
    assert!(idx.search_knn(&[], 5, 1.0).is_empty());
}

#[test]
fn empty_query_returns_empty_references_exactly() {
    let series = vec![Vec::new(), vec![1.0, 2.0, 3.0], Vec::new()];
    let idx = MsmTransducer::from_series(
        QuantizationConfig::for_u8(0.0, 10.0),
        MsmConfig::new(1.0),
        &series,
    );

    let range = idx.search_range(&[], 0.0);
    assert_eq!(ids(&range), HashSet::from([0, 2]));
    assert!(range.iter().all(|(_, distance)| distance.abs() < EPS));

    assert!(idx.search_range(&[], -2.0 * EPS).is_empty());

    let knn_one = idx.search_knn(&[], 1, 1.0);
    assert_eq!(knn_one.len(), 1);
    assert!(matches!(knn_one[0].0, 0 | 2));
    assert!(knn_one[0].1.abs() < EPS);

    let knn_all = idx.search_knn(&[], 5, 1.0);
    assert_eq!(ids(&knn_all), HashSet::from([0, 2]));
    assert!(knn_all.iter().all(|(_, distance)| distance.abs() < EPS));
}

#[test]
fn empty_index_returns_empty() {
    let idx: MsmTransducer<usize> =
        MsmTransducer::new(QuantizationConfig::for_u8(0.0, 10.0), MsmConfig::new(1.0));
    assert!(idx.is_empty());
    assert!(idx.search_range(&[1.0, 2.0], 100.0).is_empty());
    assert!(idx.search_knn(&[1.0, 2.0], 3, 1.0).is_empty());
}

// ---------------------------------------------------------------------------
// Single-element series and query
// ---------------------------------------------------------------------------

#[test]
fn single_element_series_and_query() {
    let series = vec![vec![1.0], vec![3.0], vec![7.0]];
    let quant = QuantizationConfig::for_u8(0.0, 100.0);
    let msm = MsmConfig::new(1.0);
    let idx = MsmTransducer::from_series(quant, msm, &series);
    let query = vec![1.0];
    for &tau in &[0.0, 2.0, 6.0, 100.0] {
        let got = idx.search_range(&query, tau);
        let want = brute_range(&series, &query, &msm, tau);
        assert_eq!(ids(&got), ids(&want), "tau={tau}");
        for (i, d) in &got {
            assert!((d - msm.distance(&query, &series[*i])).abs() < EPS);
        }
    }
}

// ---------------------------------------------------------------------------
// Length mismatch (MSM spans different lengths via split/merge)
// ---------------------------------------------------------------------------

#[test]
fn length_mismatch_matches_brute_force() {
    let series = vec![
        vec![10.0, 20.0],
        vec![10.0, 20.0, 30.0, 40.0, 50.0],
        vec![10.0, 11.0, 12.0],
        vec![80.0],
    ];
    let quant = QuantizationConfig::for_u8(0.0, 100.0);
    let msm = MsmConfig::new(1.0);
    let idx = MsmTransducer::from_series(quant, msm, &series);
    let query = vec![10.0, 20.0, 30.0]; // length differs from most references
    for &tau in &[0.0, 5.0, 20.0, 60.0, 1000.0] {
        let got = idx.search_range(&query, tau);
        let want = brute_range(&series, &query, &msm, tau);
        assert_eq!(ids(&got), ids(&want), "tau={tau}");
        for (gi, gd) in &got {
            assert!((gd - msm.distance(&query, &series[*gi])).abs() < EPS);
        }
    }
}

// ---------------------------------------------------------------------------
// k-NN when k exceeds the number of indexed series
// ---------------------------------------------------------------------------

#[test]
fn knn_k_exceeds_len_returns_all_sorted() {
    let series = vec![vec![1.0, 2.0], vec![5.0, 6.0], vec![9.0, 9.0]];
    let quant = QuantizationConfig::for_u8(0.0, 10.0);
    let msm = MsmConfig::new(1.0);
    let idx = MsmTransducer::from_series(quant, msm, &series);
    let query = vec![1.0, 2.0];
    let got = idx.search_knn(&query, 100, 1.0);
    assert_eq!(got.len(), series.len(), "k>len returns every series");
    for w in got.windows(2) {
        assert!(w[0].1 <= w[1].1 + EPS, "results must be ascending");
    }
}

// ---------------------------------------------------------------------------
// Inclusive threshold boundary
// ---------------------------------------------------------------------------

#[test]
fn threshold_boundary_is_inclusive() {
    // query=[1.0], series=[3.0] -> MSM = |1-3| = 2.0 (a Move).
    let series = vec![vec![3.0]];
    let quant = QuantizationConfig::for_u8(0.0, 100.0);
    let msm = MsmConfig::new(1.0);
    let idx = MsmTransducer::from_series(quant, msm, &series);
    let query = vec![1.0];
    let exact = msm.distance(&query, &series[0]);
    assert!((exact - 2.0).abs() < EPS);

    // At tau == exact: included. Just below: excluded.
    assert_eq!(idx.search_range(&query, exact).len(), 1);
    assert!(idx.search_range(&query, exact - 0.5).is_empty());
}

// ---------------------------------------------------------------------------
// Out-of-range values: exercise the ±∞ extreme bins of `bin_bounds`
// ---------------------------------------------------------------------------

#[test]
fn out_of_range_values_use_extreme_bins() {
    // Quantizer covers [0, 100]; several series sit entirely below 0 or above 100,
    // folding into bin 0 (lo = -inf) and bin 255 (hi = +inf) respectively.
    let series = vec![
        vec![-50.0, -40.0], // bin 0 (below min)
        vec![150.0, 160.0], // bin 255 (above max)
        vec![10.0, 20.0],   // interior
        vec![-48.0, -41.0], // near the first, also below min
    ];
    let quant = QuantizationConfig::for_u8(0.0, 100.0);
    let msm = MsmConfig::new(1.0);
    let idx = MsmTransducer::from_series(quant, msm, &series);

    for query in [vec![-45.0, -42.0], vec![155.0, 158.0], vec![12.0, 18.0]] {
        for &tau in &[0.0, 3.0, 10.0, 50.0, 1000.0] {
            let got = idx.search_range(&query, tau);
            let want = brute_range(&series, &query, &msm, tau);
            assert_eq!(ids(&got), ids(&want), "query={query:?} tau={tau}");
            for (i, d) in &got {
                assert!((d - msm.distance(&query, &series[*i])).abs() < EPS);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Collisions: distinct ids that quantize identically are all recovered
// ---------------------------------------------------------------------------

#[test]
fn quantization_collisions_all_recovered() {
    let quant = QuantizationConfig::for_u8(0.0, 100.0);
    let msm = MsmConfig::new(1.0);
    let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);
    // Three near-identical series collapse to the same byte key at 256 bins / [0,100].
    idx.insert(1, &[10.0, 20.0, 30.0]);
    idx.insert(2, &[10.1, 20.1, 30.1]);
    idx.insert(3, &[10.2, 20.2, 30.2]);
    let got = idx.search_range(&[10.1, 20.1, 30.1], 5.0);
    let recovered: HashSet<u32> = got.iter().map(|(v, _)| *v).collect();
    assert!(
        recovered.contains(&1) && recovered.contains(&2) && recovered.contains(&3),
        "all colliding ids must be returned: {recovered:?}"
    );
}

// ---------------------------------------------------------------------------
// Differential cross-check vs HybridSearchIndex::search_exact
// ---------------------------------------------------------------------------

#[test]
fn transducer_is_exact_and_supersets_hybrid() {
    let series = vec![
        vec![10.0, 20.0, 30.0],
        vec![12.0, 19.0, 31.0],
        vec![50.0, 55.0, 60.0],
        vec![10.0, 20.0, 30.0, 40.0],
        vec![90.0, 10.0, 50.0],
    ];
    let quant = QuantizationConfig::for_u8(0.0, 100.0);
    let msm = MsmConfig::new(1.0);

    let transducer = MsmTransducer::from_series(quant.clone(), msm, &series);
    let mut hybrid: HybridSearchIndex<usize> = HybridSearchIndex::new(quant, msm);
    for (i, s) in series.iter().enumerate() {
        hybrid.insert(i, s);
    }

    let query = vec![11.0, 20.0, 30.0];
    for &tau in &[0.0, 2.0, 5.0, 20.0, 100.0] {
        let t_ids = ids(&transducer.search_range(&query, tau));
        let h_ids = ids(&hybrid.search_exact(&query, tau));
        let b_ids = ids(&brute_range(&series, &query, &msm, tau));
        // The transducer is exact: it equals the brute-force set with no
        // false negatives and no false positives.
        assert_eq!(
            t_ids, b_ids,
            "transducer must equal brute force at tau={tau}"
        );
        // HybridSearchIndex re-verifies candidates exactly (so no false
        // positives), but its lossy Levenshtein pre-filter can drop true MSM
        // neighbors whose quantized form is several bin-edits away. Its result
        // set is therefore a subset of the exact set -- never a superset. This
        // completeness gap is precisely why the MsmTransducer exists.
        assert!(
            h_ids.is_subset(&t_ids),
            "hybrid returned ids outside the exact set at tau={tau}: hybrid={h_ids:?} exact={t_ids:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Concurrent read-only queries (Sync; not a loom test by design)
// ---------------------------------------------------------------------------

#[test]
fn concurrent_queries_are_consistent() {
    let series = vec![
        vec![10.0, 20.0, 30.0],
        vec![11.0, 21.0, 29.0],
        vec![50.0, 60.0, 70.0],
        vec![10.0, 20.0, 30.0, 40.0],
    ];
    let quant = QuantizationConfig::for_u8(0.0, 100.0);
    let msm = MsmConfig::new(1.0);
    let idx = MsmTransducer::from_series(quant, msm, &series);
    let query = vec![12.0, 22.0, 31.0];
    let expected = idx.search_range(&query, 15.0);

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| idx.search_range(&query, 15.0)))
            .collect();
        for h in handles {
            let got = h.join().expect("query thread panicked");
            assert_eq!(got, expected, "concurrent query diverged from sequential");
        }
    });
}

// ---------------------------------------------------------------------------
// Property: exactness holds even when values fall outside the quantizer range
// (heavy use of the ±∞ extreme bins).
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_range_exact_with_outliers(
        series in prop::collection::vec(
            prop::collection::vec(-50.0f64..150.0, 1..6),
            1..12,
        ),
        query in prop::collection::vec(-50.0f64..150.0, 1..6),
        c in 0.1f64..3.0,
        tau in 0.0f64..60.0,
    ) {
        // Quantizer range [0,100] is narrower than the value range, so many
        // elements land in the extreme (±∞) bins.
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(c);
        let idx = MsmTransducer::from_series(quant, msm, &series);

        let got = idx.search_range(&query, tau);
        let want = brute_range(&series, &query, &msm, tau);
        prop_assert_eq!(ids(&got), ids(&want));
        for (i, d) in &got {
            let exact = msm.distance(&query, &series[*i]);
            prop_assert!((d - exact).abs() < 1e-9);
        }
    }
}
