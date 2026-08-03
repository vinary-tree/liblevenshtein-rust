//! Public-surface integration and property tests for discrete Fréchet search.

use std::collections::{HashMap, HashSet};

use liblevenshtein::cost::BottleneckCost;
use liblevenshtein::time_series::elastic::ElasticKernel;
use liblevenshtein::time_series::{FrechetConfig, FrechetTransducer, QuantizationConfig};
use proptest::prelude::*;

const EPSILON: f64 = 1.0e-9;

fn brute_range(series: &[Vec<f64>], query: &[f64], cutoff: f64) -> Vec<(usize, f64)> {
    let config = FrechetConfig::new();
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

fn assert_bottleneck_kernel<K: ElasticKernel<Monoid = BottleneckCost>>() {}

#[test]
fn bottleneck_kernel_range_knn_stutters_and_collisions_are_exact() {
    assert_bottleneck_kernel::<FrechetConfig>();

    let series = vec![
        Vec::new(),
        vec![1.0, 2.0, 3.0],
        vec![1.01, 2.01, 3.01],
        vec![1.0, 1.0, 2.0, 3.0],
        vec![9.0, 9.0],
    ];
    let index = FrechetTransducer::from_series(
        QuantizationConfig::for_u8(0.0, 10.0),
        FrechetConfig::new(),
        &series,
    );
    let query = [1.0, 2.0, 3.0];

    assert_eq!(
        as_map(&index.search_range(&query, 0.1)),
        as_map(&brute_range(&series, &query, 0.1))
    );
    assert_eq!(
        as_map(&index.search_range(&query, f64::INFINITY)),
        as_map(&brute_range(&series, &query, f64::INFINITY))
    );

    let knn = index.search_knn(&query, series.len(), 0.0);
    let expected_finite: HashMap<_, _> = brute_range(&series, &query, f64::INFINITY)
        .into_iter()
        .filter(|(_, distance)| distance.is_finite())
        .collect();
    assert_eq!(as_map(&knn), expected_finite);
    assert!(knn.windows(2).all(|pair| pair[0].1 <= pair[1].1));

    assert_eq!(
        as_map(&index.search_range(&[], f64::INFINITY)),
        as_map(&brute_range(&series, &[], f64::INFINITY))
    );
    assert_eq!(index.search_range(&query, -1.0), Vec::new());
}

#[test]
fn public_configuration_and_mutation_surface_is_complete() {
    let mut index: FrechetTransducer<&str> =
        FrechetTransducer::new(QuantizationConfig::for_u8(-5.0, 5.0), FrechetConfig::new());
    assert_eq!(*index.kernel(), FrechetConfig::new());
    assert!(index.insert("a", &[1.0, 2.0]));
    assert!(index.insert("b", &[1.0, 1.0, 2.0]));
    assert!(!index.insert("a", &[2.0, 3.0]));
    assert_eq!(index.get_original(&"a"), Some(&[2.0, 3.0][..]));
    assert!(index.remove("a"));
    assert!(!index.remove("a"));
    assert_eq!(index.search_range(&[1.0, 2.0], 0.0), vec![("b", 0.0)]);
}

#[test]
fn non_finite_samples_are_outside_the_exact_search_domain() {
    let mut index: FrechetTransducer<usize> = FrechetTransducer::new(
        QuantizationConfig::for_u8(-10.0, 10.0),
        FrechetConfig::new(),
    );
    index.insert(0, &[1.0, 2.0]);
    index.insert(1, &[f64::NAN]);

    assert_eq!(index.search_range(&[1.0, 2.0], 100.0), vec![(0, 0.0)]);
    assert!(index.search_range(&[f64::NAN], f64::INFINITY).is_empty());
    assert!(index
        .search_knn(&[f64::INFINITY], 2, f64::INFINITY)
        .is_empty());
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
        cutoff in 0u16..=40,
        increment in 0u8..=20,
    ) {
        let series: Vec<Vec<f64>> = raw_series
            .into_iter()
            .map(|values| values.into_iter().map(f64::from).collect())
            .collect();
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let cutoff = f64::from(cutoff);
        let index = FrechetTransducer::from_series(
            QuantizationConfig::for_u8(-20.0, 20.0),
            FrechetConfig::new(),
            &series,
        );

        let actual = index.search_range(&query, cutoff);
        let expected = brute_range(&series, &query, cutoff);
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
        k in 0usize..22,
    ) {
        let series: Vec<Vec<f64>> = raw_series
            .into_iter()
            .map(|values| values.into_iter().map(f64::from).collect())
            .collect();
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let config = FrechetConfig::new();
        let index = FrechetTransducer::from_series(
            QuantizationConfig::for_u8(-12.0, 12.0),
            config,
            &series,
        );

        let actual = index.search_knn(&query, k, 0.0);
        let mut expected_distances: Vec<_> = series
            .iter()
            .map(|candidate| config.distance(&query, candidate))
            .filter(|distance| distance.is_finite())
            .collect();
        expected_distances.sort_by(f64::total_cmp);
        expected_distances.truncate(k);
        let actual_distances: Vec<_> = actual.iter().map(|(_, distance)| *distance).collect();

        prop_assert_eq!(actual_distances, expected_distances);
        prop_assert_eq!(&actual, &index.search_knn(&query, k, f64::INFINITY));
        for (id, distance) in actual {
            prop_assert!((distance - config.distance(&query, &series[id])).abs() <= EPSILON);
        }
    }
}
