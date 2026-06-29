//! Profiling benchmark specifically for FuzzyMultiMap operations.
//!
//! This benchmark is designed to run with flamegraph to identify
//! bottlenecks in the FuzzyMultiMap query method.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use liblevenshtein::cache::multimap::FuzzyMultiMap;
use liblevenshtein::prelude::*;
use std::collections::HashSet;

/// Create a large dictionary with HashSet values for profiling
fn create_fuzzy_multimap_dict(
    size: usize,
    values_per_set: usize,
) -> PathMapDictionary<HashSet<i32>> {
    let terms: Vec<(String, HashSet<i32>)> = (0..size)
        .map(|i| {
            let term = format!("term{:04}", i);
            let values: HashSet<i32> = (0..values_per_set as i32)
                .map(|v| i as i32 * 100 + v)
                .collect();
            (term, values)
        })
        .collect();
    PathMapDictionary::from_terms_with_values(terms)
}

/// Profile: FuzzyMultiMap query with HashSet aggregation
fn profile_fuzzy_multimap_query(c: &mut Criterion) {
    let dict = create_fuzzy_multimap_dict(1000, 5);
    let fuzzy_map = FuzzyMultiMap::new(dict, Algorithm::Standard);

    c.bench_function("fuzzy_multimap_query", |b| {
        b.iter(|| {
            let result = fuzzy_map.query(black_box("term"), black_box(2));
            black_box(result)
        })
    });
}

/// Profile: FuzzyMultiMap query with many matches (high aggregation)
fn profile_fuzzy_multimap_high_aggregation(c: &mut Criterion) {
    let dict = create_fuzzy_multimap_dict(1000, 10);
    let fuzzy_map = FuzzyMultiMap::new(dict, Algorithm::Standard);

    c.bench_function("fuzzy_multimap_high_aggregation", |b| {
        b.iter(|| {
            // Query "term" which will match term0000-term9999 at distance 2-3
            let result = fuzzy_map.query(black_box("term"), black_box(3));
            black_box(result)
        })
    });
}

/// Profile: FuzzyMultiMap query with Vec aggregation
fn profile_fuzzy_multimap_vec_concat(c: &mut Criterion) {
    let terms: Vec<(String, Vec<i32>)> = (0..1000)
        .map(|i| {
            let term = format!("term{:04}", i);
            let values = vec![i as i32, i as i32 * 2, i as i32 * 3];
            (term, values)
        })
        .collect();
    let dict = PathMapDictionary::from_terms_with_values(terms);
    let fuzzy_map = FuzzyMultiMap::new(dict, Algorithm::Standard);

    c.bench_function("fuzzy_multimap_vec_concat", |b| {
        b.iter(|| {
            let result = fuzzy_map.query(black_box("term"), black_box(2));
            black_box(result)
        })
    });
}

/// Profile: Transducer query (baseline without aggregation)
fn profile_transducer_query_baseline(c: &mut Criterion) {
    let dict = create_fuzzy_multimap_dict(1000, 5);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("transducer_query_baseline", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer.query(black_box("term"), black_box(2)).collect();
            black_box(results)
        })
    });
}

/// Profile: Dictionary get_value operations (isolation test)
fn profile_dict_get_value_hashset(c: &mut Criterion) {
    let dict = create_fuzzy_multimap_dict(1000, 5);

    c.bench_function("dict_get_value_hashset", |b| {
        b.iter(|| {
            // Access multiple values
            for i in (0..100).step_by(10) {
                let term = format!("term{:04}", i);
                let value = dict.get_value(black_box(&term));
                black_box(value);
            }
        })
    });
}

/// Profile: HashSet aggregation (isolation test)
fn profile_hashset_aggregation(c: &mut Criterion) {
    use liblevenshtein::cache::multimap::CollectionAggregate;

    // Create sample data
    let sets: Vec<HashSet<i32>> = (0..50).map(|i| (i..i + 10).collect()).collect();

    c.bench_function("hashset_aggregation", |b| {
        b.iter(|| {
            let result = HashSet::aggregate(black_box(sets.clone().into_iter()));
            black_box(result)
        })
    });
}

/// Profile: Complete FuzzyMultiMap query with all steps
fn profile_fuzzy_multimap_complete(c: &mut Criterion) {
    let dict = create_fuzzy_multimap_dict(5000, 8);
    let fuzzy_map = FuzzyMultiMap::new(dict, Algorithm::Standard);

    c.bench_function("fuzzy_multimap_complete", |b| {
        b.iter(|| {
            // Multiple queries to get comprehensive profiling data
            for query_str in &["term", "test", "item", "data", "code"] {
                let result = fuzzy_map.query(black_box(query_str), black_box(2));
                black_box(result);
            }
        })
    });
}

criterion_group!(
    benches,
    profile_fuzzy_multimap_query,
    profile_fuzzy_multimap_high_aggregation,
    profile_fuzzy_multimap_vec_concat,
    profile_transducer_query_baseline,
    profile_dict_get_value_hashset,
    profile_hashset_aggregation,
    profile_fuzzy_multimap_complete,
);
criterion_main!(benches);
