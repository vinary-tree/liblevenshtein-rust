//! Public-surface integration and property tests for exact TWED trie search.

use std::collections::{HashMap, HashSet};

use liblevenshtein::time_series::{
    MetricTwedConfig, MetricTwedConfigError, MetricTwedTransducer, QuantizationConfig, TwedConfig,
    TwedTransducer,
};
use proptest::prelude::*;

const EPSILON: f64 = 1.0e-9;

fn brute_range(
    series: &[Vec<f64>],
    query: &[f64],
    config: TwedConfig,
    cutoff: f64,
) -> Vec<(usize, f64)> {
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
fn range_knn_empty_boundaries_and_collisions_are_exact() {
    let series = vec![
        Vec::new(),
        vec![1.0, 2.0, 3.0],
        vec![1.01, 2.01, 3.01],
        vec![1.0, 0.0, 2.0, 3.0],
        vec![9.0, 9.0],
    ];
    let config = TwedConfig::new(0.5, 1.0);
    let index = TwedTransducer::from_series(QuantizationConfig::for_u8(0.0, 10.0), config, &series);
    let query = [1.0, 2.0, 3.0];

    assert_eq!(
        as_map(&index.search_range(&query, 0.1)),
        as_map(&brute_range(&series, &query, config, 0.1))
    );
    assert_eq!(
        as_map(&index.search_range(&query, f64::INFINITY)),
        as_map(&brute_range(&series, &query, config, f64::INFINITY))
    );

    let knn = index.search_knn(&query, series.len(), 0.0);
    assert_eq!(
        as_map(&knn),
        as_map(&brute_range(&series, &query, config, f64::INFINITY))
    );
    assert!(knn.windows(2).all(|pair| pair[0].1 <= pair[1].1));

    assert_eq!(
        as_map(&index.search_range(&[], 20.0)),
        as_map(&brute_range(&series, &[], config, 20.0))
    );
    assert!(index.search_range(&query, -1.0).is_empty());
}

#[test]
fn public_configuration_metric_validation_and_mutation_are_total() {
    assert_eq!(
        MetricTwedConfig::try_new(0.0, 1.0),
        Err(MetricTwedConfigError::NonPositiveStiffness)
    );
    assert_eq!(
        MetricTwedConfig::try_new(1.0, -1.0),
        Err(MetricTwedConfigError::InvalidGapPenalty)
    );
    assert!(MetricTwedConfig::try_new(f64::NAN, 1.0).is_err());
    assert!(MetricTwedConfig::try_new(1.0, f64::INFINITY).is_err());

    let metric = MetricTwedConfig::try_new(0.5, 1.0).unwrap();
    let mut index: MetricTwedTransducer<&str> =
        MetricTwedTransducer::new(QuantizationConfig::for_u8(-5.0, 5.0), metric);
    assert_eq!(index.kernel().stiffness(), 0.5);
    assert_eq!(index.kernel().gap_penalty(), 1.0);
    assert!(index.insert("a", &[1.0, 2.0]));
    assert!(index.insert("b", &[1.0, 1.0, 2.0]));
    assert!(!index.insert("a", &[2.0, 3.0]));
    assert_eq!(index.get_original(&"a"), Some(&[2.0, 3.0][..]));
    assert!(index.remove("a"));
    assert!(!index.remove("a"));
    assert_eq!(index.search_range(&[1.0, 1.0, 2.0], 0.0), vec![("b", 0.0)]);
}

#[test]
fn non_finite_samples_are_outside_the_exact_search_domain() {
    let mut index: TwedTransducer<usize> = TwedTransducer::new(
        QuantizationConfig::for_u8(-10.0, 10.0),
        TwedConfig::default(),
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
        raw_nu in 0u8..=4,
        raw_lambda in 0u8..=4,
        cutoff in 0u16..=200,
        increment in 0u8..=40,
    ) {
        let series: Vec<Vec<f64>> = raw_series
            .into_iter()
            .map(|values| values.into_iter().map(f64::from).collect())
            .collect();
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let config = TwedConfig::new(f64::from(raw_nu), f64::from(raw_lambda));
        let cutoff = f64::from(cutoff);
        let index = TwedTransducer::from_series(
            QuantizationConfig::for_u8(-20.0, 20.0),
            config,
            &series,
        );

        let actual = index.search_range(&query, cutoff);
        let expected = brute_range(&series, &query, config, cutoff);
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
        raw_nu in 0u8..=4,
        raw_lambda in 0u8..=4,
        k in 0usize..22,
    ) {
        let series: Vec<Vec<f64>> = raw_series
            .into_iter()
            .map(|values| values.into_iter().map(f64::from).collect())
            .collect();
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let config = TwedConfig::new(f64::from(raw_nu), f64::from(raw_lambda));
        let index = TwedTransducer::from_series(
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
