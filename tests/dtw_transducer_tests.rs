//! Public-surface integration and property tests for exact banded DTW search.

use std::collections::{HashMap, HashSet};

use liblevenshtein::time_series::{DtwConfig, DtwTransducer, QuantizationConfig};
use proptest::prelude::*;

const EPSILON: f64 = 1.0e-9;

fn brute_range(series: &[Vec<f64>], query: &[f64], band: usize, cutoff: f64) -> Vec<(usize, f64)> {
    let config = DtwConfig::new(band);
    let mut results: Vec<_> = series
        .iter()
        .enumerate()
        .filter_map(|(id, candidate)| {
            config
                .distance_with_cutoff(query, candidate, cutoff)
                .map(|distance| (id, distance))
        })
        .collect();
    results.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    results
}

fn as_map(results: &[(usize, f64)]) -> HashMap<usize, f64> {
    results.iter().copied().collect()
}

#[test]
fn required_band_root_units_collisions_and_exact_survivors() {
    let series = vec![
        Vec::new(),
        vec![0.0, 0.0],
        vec![3.0, 4.0],
        vec![3.01, 4.01],
        vec![3.0, 3.0, 4.0],
        vec![20.0],
    ];
    let index = DtwTransducer::from_series(
        QuantizationConfig::for_u8(0.0, 20.0),
        DtwConfig::new(1),
        &series,
    );

    assert_eq!(index.config().band, 1);
    assert_eq!(index.len(), series.len());
    assert_eq!(
        as_map(&index.search_range(&[0.0, 0.0], 5.0)),
        as_map(&brute_range(&series, &[0.0, 0.0], 1, 5.0))
    );
    assert_eq!(
        as_map(&index.search_range(&[3.0, 4.0], 0.02)),
        as_map(&brute_range(&series, &[3.0, 4.0], 1, 0.02))
    );
    assert_eq!(
        as_map(&index.search_range(&[], f64::INFINITY)),
        as_map(&brute_range(&series, &[], 1, f64::INFINITY))
    );

    let knn = index.search_knn(&[3.0, 4.0], series.len(), 0.0);
    let mut expected: Vec<_> = series
        .iter()
        .map(|candidate| DtwConfig::new(1).distance(&[3.0, 4.0], candidate))
        .filter(|distance| distance.is_finite())
        .collect();
    expected.sort_by(f64::total_cmp);
    assert_eq!(
        knn.iter()
            .map(|(_, distance)| *distance)
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn public_mutation_and_invalid_domain_are_total() {
    let mut index: DtwTransducer<&str> =
        DtwTransducer::new(QuantizationConfig::for_u8(-5.0, 5.0), DtwConfig::new(2));
    assert!(index.is_empty());
    assert!(index.insert("a", &[1.0, 2.0]));
    assert!(index.insert("b", &[1.0, 1.0, 2.0]));
    assert!(index.insert("bad", &[f64::NAN]));
    assert!(!index.insert("a", &[2.0, 3.0]));
    assert_eq!(index.get_original(&"a"), Some(&[2.0, 3.0][..]));
    assert_eq!(index.quant_config().min_value, -5.0);
    assert_eq!(index.quant_config().max_value, 5.0);
    assert_eq!(index.quant_config().num_bins, 256);

    assert_eq!(index.search_range(&[1.0, 2.0], 0.0), vec![("b", 0.0)]);
    assert!(index.search_range(&[1.0], -1.0).is_empty());
    assert!(index.search_range(&[f64::NAN], f64::INFINITY).is_empty());
    assert!(index
        .search_knn(&[f64::INFINITY], 3, f64::INFINITY)
        .is_empty());
    assert!(index.remove("a"));
    assert!(!index.remove("a"));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    #[test]
    fn exact_range_is_differential_monotone_and_deterministic(
        raw_series in prop::collection::vec(
            prop::collection::vec(-20i16..=20, 0..8),
            0..20,
        ),
        raw_query in prop::collection::vec(-20i16..=20, 0..8),
        band in 0usize..9,
        cutoff in 0u16..=40,
        increment in 0u8..=20,
    ) {
        let series: Vec<Vec<f64>> = raw_series
            .into_iter()
            .map(|values| values.into_iter().map(f64::from).collect())
            .collect();
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let cutoff = f64::from(cutoff);
        let index = DtwTransducer::from_series(
            QuantizationConfig::for_u8(-20.0, 20.0),
            DtwConfig::new(band),
            &series,
        );

        let actual = index.search_range(&query, cutoff);
        let expected = brute_range(&series, &query, band, cutoff);
        prop_assert_eq!(as_map(&actual), as_map(&expected));
        prop_assert_eq!(&actual, &index.search_range(&query, cutoff));
        prop_assert!(actual.windows(2).all(|pair| pair[0].1 <= pair[1].1));

        let wider = index.search_range(&query, cutoff + f64::from(increment));
        let wider_ids: HashSet<_> = wider.iter().map(|(id, _)| *id).collect();
        prop_assert!(actual.iter().all(|(id, _)| wider_ids.contains(id)));
    }

    #[test]
    fn knn_distance_multiset_matches_brute_force(
        raw_series in prop::collection::vec(
            prop::collection::vec(-12i16..=12, 0..7),
            0..18,
        ),
        raw_query in prop::collection::vec(-12i16..=12, 0..7),
        band in 0usize..8,
        k in 0usize..22,
    ) {
        let series: Vec<Vec<f64>> = raw_series
            .into_iter()
            .map(|values| values.into_iter().map(f64::from).collect())
            .collect();
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let config = DtwConfig::new(band);
        let index = DtwTransducer::from_series(
            QuantizationConfig::for_u8(-12.0, 12.0),
            config,
            &series,
        );

        let actual = index.search_knn(&query, k, 0.0);
        let (observed, stats) = index.search_knn_with_stats(&query, k, 0.0);
        let mut expected_distances: Vec<_> = series
            .iter()
            .map(|candidate| config.distance(&query, candidate))
            .filter(|distance| distance.is_finite())
            .collect();
        expected_distances.sort_by(f64::total_cmp);
        expected_distances.truncate(k);
        let actual_distances: Vec<_> = actual.iter().map(|(_, distance)| *distance).collect();

        prop_assert_eq!(actual_distances, expected_distances);
        prop_assert_eq!(&observed, &actual);
        prop_assert!(stats.accounting_is_consistent());
        prop_assert_eq!(&actual, &index.search_knn(&query, k, f64::INFINITY));
        for (id, distance) in actual {
            prop_assert!((distance - config.distance(&query, &series[id])).abs() <= EPSILON);
        }
    }
}
