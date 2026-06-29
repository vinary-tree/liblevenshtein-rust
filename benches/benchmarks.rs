//! Comprehensive benchmarks for liblevenshtein profiling and performance analysis.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::pathmap::PathMapDictionary;
use liblevenshtein::prelude::*;

/// Create a dictionary of the specified size with varied word lengths
fn create_dictionary(size: usize) -> PathMapDictionary {
    let words: Vec<String> = (0..size)
        .map(|i| {
            // Create words of varying lengths (4-12 characters)
            let len = 4 + (i % 9);
            format!("word{:0width$}", i, width = len - 4)
        })
        .collect();
    PathMapDictionary::from_terms(words)
}

/// Benchmark: Query performance with different dictionary sizes
fn bench_query_varying_dict_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_varying_dict_size");

    for size in [100, 500, 1000, 5000].iter() {
        let dict = create_dictionary(*size);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query(black_box("word500"), black_box(2))
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Query performance with different max distances
fn bench_query_varying_distance(c: &mut Criterion) {
    let dict = create_dictionary(1000);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("query_varying_distance");

    for distance in [0, 1, 2, 3, 4].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(distance), distance, |b, &d| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query(black_box("word500"), black_box(d))
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Query performance with different query lengths
fn bench_query_varying_query_length(c: &mut Criterion) {
    let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(vec![
        "a",
        "ab",
        "abc",
        "abcd",
        "abcde",
        "abcdef",
        "abcdefg",
        "abcdefgh",
        "test",
        "testing",
        "extraordinary",
        "incomprehensibilities",
    ]);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("query_varying_query_length");

    let queries = [
        ("a", 1),
        ("abc", 3),
        ("abcde", 5),
        ("testing", 7),
        ("extraordinary", 13),
    ];

    for (query, len) in queries.iter() {
        group.throughput(Throughput::Bytes(*len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), query, |b, &q| {
            b.iter(|| {
                let results: Vec<_> = transducer.query(black_box(q), black_box(2)).collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: High-result queries (memory intensive)
fn bench_query_many_results(c: &mut Criterion) {
    let dict = create_dictionary(1000);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("query_many_results_distance_3", |b| {
        b.iter(|| {
            // This should return many results with distance 3
            let results: Vec<_> = transducer.query(black_box("word"), black_box(3)).collect();
            black_box(results);
        });
    });
}

/// Benchmark: Worst-case queries (maximum state expansion)
fn bench_query_worst_case(c: &mut Criterion) {
    let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(vec![
        "aaaa", "aaab", "aaba", "aabb", "abaa", "abab", "abba", "abbb", "baaa", "baab", "baba",
        "babb", "bbaa", "bbab", "bbba", "bbbb",
    ]);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("query_worst_case_similar_words", |b| {
        b.iter(|| {
            // Query similar to all dictionary words - maximum state expansion
            let results: Vec<_> = transducer.query(black_box("aaaa"), black_box(2)).collect();
            black_box(results);
        });
    });
}

/// Benchmark: Different algorithms
fn bench_algorithms(c: &mut Criterion) {
    let dict = create_dictionary(500);
    let mut group = c.benchmark_group("algorithms");

    for algo in [
        Algorithm::Standard,
        Algorithm::Transposition,
        Algorithm::MergeAndSplit,
    ]
    .iter()
    {
        let transducer = Transducer::new(dict.clone(), *algo);
        group.bench_with_input(
            BenchmarkId::new("algorithm", format!("{:?}", algo)),
            algo,
            |b, _| {
                b.iter(|| {
                    let results: Vec<_> = transducer
                        .query(black_box("word250"), black_box(2))
                        .collect();
                    black_box(results);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Dictionary operations (runtime updates)
fn bench_dictionary_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("dictionary_operations");

    group.bench_function("insert", |b| {
        let dict = create_dictionary(1000);
        let mut counter = 0;
        b.iter(|| {
            dict.insert(&format!("newword{}", counter));
            counter += 1;
        });
    });

    group.bench_function("remove", |b| {
        let dict = create_dictionary(1000);
        let mut counter = 0;
        b.iter(|| {
            dict.remove(&format!("word{}", counter % 1000));
            counter += 1;
        });
    });

    group.bench_function("contains", |b| {
        let dict = create_dictionary(1000);
        b.iter(|| dict.contains(black_box("word500")));
    });

    group.finish();
}

/// Benchmark: Distance computation functions
fn bench_distance_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_computation");

    let test_pairs = [
        ("kitten", "sitting"),
        ("test", "test"),
        ("a", "abcdefgh"),
        ("programming", "programing"),
        ("incomprehensibilities", "miscomprehensions"),
    ];

    for (s1, s2) in test_pairs.iter() {
        group.bench_with_input(
            BenchmarkId::new("standard", format!("{}_{}", s1.len(), s2.len())),
            &(s1, s2),
            |bencher, &(a, b)| {
                bencher.iter(|| {
                    liblevenshtein::distance::standard_distance(black_box(a), black_box(b))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("transposition", format!("{}_{}", s1.len(), s2.len())),
            &(s1, s2),
            |bencher, &(a, b)| {
                bencher.iter(|| {
                    liblevenshtein::distance::transposition_distance(black_box(a), black_box(b))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Insertion/deletion heavy queries
fn bench_insertion_deletion_queries(c: &mut Criterion) {
    let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(vec![
        "apple",
        "application",
        "apply",
        "test",
        "testing",
        "tested",
    ]);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("insertion_deletion");

    // Query missing characters (requires insertions)
    group.bench_function("insertions", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer.query(black_box("aple"), black_box(1)).collect();
            black_box(results);
        });
    });

    // Query with extra characters (requires deletions)
    group.bench_function("deletions", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query(black_box("appple"), black_box(1))
                .collect();
            black_box(results);
        });
    });

    // Mixed operations
    group.bench_function("mixed", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query(black_box("aplication"), black_box(2))
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

/// Benchmark: Query with distance calculation
fn bench_query_with_distance(c: &mut Criterion) {
    let dict = create_dictionary(500);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    c.bench_function("query_with_distance_calculation", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_with_distance(black_box("word250"), black_box(2))
                .collect();
            black_box(results);
        });
    });
}

criterion_group!(
    benches,
    bench_query_varying_dict_size,
    bench_query_varying_distance,
    bench_query_varying_query_length,
    bench_query_many_results,
    bench_query_worst_case,
    bench_algorithms,
    bench_dictionary_operations,
    bench_distance_computation,
    bench_insertion_deletion_queries,
    bench_query_with_distance,
);
criterion_main!(benches);
