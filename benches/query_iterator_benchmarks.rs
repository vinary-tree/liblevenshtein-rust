//! Comprehensive benchmarks for all query iterator types and modifiers.
//!
//! This benchmark suite tests:
//! - Ordered vs unordered queries
//! - Prefix mode
//! - Various distance levels (0-10, 99)
//! - Different dictionary sizes
//! - Different algorithms (Standard, Transposition, MergeAndSplit)

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use liblevenshtein::prelude::*;

/// Create a dictionary of the specified size with realistic English-like words
fn create_dictionary(size: usize) -> DoubleArrayTrie {
    let words: Vec<String> = (0..size)
        .map(|i| {
            // Create varied word patterns
            match i % 5 {
                0 => format!("test{}", i),
                1 => format!("best{}", i),
                2 => format!("rest{}", i),
                3 => format!("word{}", i),
                _ => format!("term{}", i),
            }
        })
        .collect();
    DoubleArrayTrie::from_terms(words)
}

/// Benchmark: Ordered query vs unordered query
fn bench_ordered_vs_unordered(c: &mut Criterion) {
    let dict = create_dictionary(1000);
    let mut group = c.benchmark_group("ordered_vs_unordered");

    for distance in [1, 2, 5].iter() {
        // Unordered query
        group.bench_with_input(
            BenchmarkId::new("unordered", distance),
            distance,
            |b, &d| {
                let transducer = Transducer::new(dict.clone(), Algorithm::Standard);
                b.iter(|| {
                    let results: Vec<_> = transducer
                        .query(black_box("test500"), black_box(d))
                        .collect();
                    black_box(results);
                });
            },
        );

        // Ordered query
        group.bench_with_input(BenchmarkId::new("ordered", distance), distance, |b, &d| {
            let transducer = Transducer::new(dict.clone(), Algorithm::Standard);
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_ordered(black_box("test500"), black_box(d))
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Ordered query with varying distances
fn bench_ordered_query_varying_distance(c: &mut Criterion) {
    let dict = create_dictionary(1000);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("ordered_query_varying_distance");

    for distance in [0, 1, 2, 3, 5, 10, 99].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(distance), distance, |b, &d| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_ordered(black_box("test500"), black_box(d))
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Prefix mode vs standard mode
fn bench_prefix_vs_standard(c: &mut Criterion) {
    let dict = DoubleArrayTrie::from_terms(vec![
        "test", "testing", "tested", "tester", "tests", "best", "rest", "nest", "fest", "west",
    ]);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("prefix_vs_standard");

    for distance in [0, 1, 2].iter() {
        // Standard mode
        group.bench_with_input(BenchmarkId::new("standard", distance), distance, |b, &d| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_ordered(black_box("test"), black_box(d))
                    .collect();
                black_box(results);
            });
        });

        // Prefix mode
        group.bench_with_input(BenchmarkId::new("prefix", distance), distance, |b, &d| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_ordered(black_box("test"), black_box(d))
                    .prefix()
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Ordered query with different algorithms
fn bench_ordered_query_algorithms(c: &mut Criterion) {
    let dict = create_dictionary(500);
    let mut group = c.benchmark_group("ordered_query_algorithms");

    let algorithms = [
        ("standard", Algorithm::Standard),
        ("transposition", Algorithm::Transposition),
        ("merge_split", Algorithm::MergeAndSplit),
    ];

    for (name, algo) in algorithms.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(name), algo, |b, &algo| {
            let transducer = Transducer::new(dict.clone(), algo);
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_ordered(black_box("test250"), black_box(2))
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Early termination with take()
fn bench_ordered_query_early_termination(c: &mut Criterion) {
    let dict = create_dictionary(1000);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("ordered_query_early_termination");

    for limit in [1, 5, 10, 50].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(limit), limit, |b, &lim| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_ordered(black_box("test500"), black_box(5))
                    .take(lim)
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Ordered query with take_while
fn bench_ordered_query_take_while(c: &mut Criterion) {
    let dict = create_dictionary(1000);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("ordered_query_take_while");

    for max_dist in [0, 1, 2, 3].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(max_dist), max_dist, |b, &md| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_ordered(black_box("test500"), black_box(5))
                    .take_while(|c| c.distance <= md)
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Ordered query lexicographic sorting overhead
fn bench_ordered_query_sorting_overhead(c: &mut Criterion) {
    let dict = create_dictionary(1000);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("ordered_query_sorting");

    // Query that produces many results at the same distance
    group.bench_function("many_results_same_distance", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_ordered(black_box("xyz"), black_box(3))
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

/// Benchmark: Prefix mode with varying query lengths
fn bench_prefix_varying_query_length(c: &mut Criterion) {
    let dict = DoubleArrayTrie::from_terms(vec![
        "t", "te", "tes", "test", "testi", "testin", "testing", "testings", "best", "resting",
    ]);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("prefix_varying_query_length");

    let queries = [("t", 1), ("te", 2), ("tes", 3), ("test", 4), ("testi", 5)];

    for (query, len) in queries.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(len), query, |b, &q| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_ordered(black_box(q), black_box(1))
                    .prefix()
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: large-distance queries (regression coverage)
fn bench_large_distance_queries(c: &mut Criterion) {
    let dict = DoubleArrayTrie::from_terms(vec!["foo", "bar", "baz", "quo", "qux"]);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("large_distance_queries");

    for distance in [10, 50, 99].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(distance), distance, |b, &d| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_ordered(black_box("quuo"), black_box(d))
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Ordered query with dictionary size scaling
fn bench_ordered_query_dict_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ordered_query_dict_size_scaling");

    for size in [100, 500, 1000, 5000].iter() {
        let dict = create_dictionary(*size);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_ordered(black_box("test500"), black_box(2))
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ordered_vs_unordered,
    bench_ordered_query_varying_distance,
    bench_prefix_vs_standard,
    bench_ordered_query_algorithms,
    bench_ordered_query_early_termination,
    bench_ordered_query_take_while,
    bench_ordered_query_sorting_overhead,
    bench_prefix_varying_query_length,
    bench_large_distance_queries,
    bench_ordered_query_dict_size_scaling,
);

criterion_main!(benches);
