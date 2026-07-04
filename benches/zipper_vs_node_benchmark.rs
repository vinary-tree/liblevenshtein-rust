//! Benchmark comparing zipper-based vs node-based query iteration.
//!
//! This benchmark compares the performance of the new zipper-based
//! ZipperQueryIterator against the existing node-based QueryIterator.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::pathmap::zipper::PathMapZipper;
use libdictenstein::pathmap::PathMapDictionary;
use liblevenshtein::transducer::{Algorithm, Transducer, ZipperQueryIterator};
use std::hint::black_box;

/// Load dictionary with common English words
fn load_test_dictionary() -> PathMapDictionary<()> {
    let dict = PathMapDictionary::new();

    // Common English words for realistic testing
    let words = vec![
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        // Add some longer words
        "testing",
        "benchmark",
        "performance",
        "optimization",
        "implementation",
        "algorithm",
        "dictionary",
        "transducer",
        "levenshtein",
        "automaton",
        "intersection",
        "traversal",
        "efficient",
        "comparison",
        "measurement",
    ];

    for word in words {
        dict.insert(word);
    }

    dict
}

/// Benchmark node-based QueryIterator
fn bench_node_based(c: &mut Criterion) {
    let dict = load_test_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut group = c.benchmark_group("node_based");

    for distance in [0, 1, 2] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("query_time", distance),
            &distance,
            |b, &distance| {
                b.iter(|| {
                    let results: Vec<_> = transducer
                        .query(black_box("time"), black_box(distance))
                        .collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark zipper-based ZipperQueryIterator
fn bench_zipper_based(c: &mut Criterion) {
    let dict = load_test_dictionary();

    let mut group = c.benchmark_group("zipper_based");

    for distance in [0, 1, 2] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("query_time", distance),
            &distance,
            |b, &distance| {
                b.iter(|| {
                    let dict_zipper = PathMapZipper::new_from_dict(&dict);
                    let iter = ZipperQueryIterator::new(
                        dict_zipper,
                        black_box("time"),
                        black_box(distance),
                        Algorithm::Standard,
                    );
                    let results: Vec<_> = iter.collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark comparison with identical queries
fn bench_comparison(c: &mut Criterion) {
    let dict = load_test_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let dict2 = load_test_dictionary();

    let queries = vec!["time", "test", "work", "people", "algorithm"];
    let distance = 1;

    let mut group = c.benchmark_group("comparison");
    group.throughput(Throughput::Elements(queries.len() as u64));

    group.bench_function("node_based_batch", |b| {
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer
                    .query(black_box(query), black_box(distance))
                    .collect();
                black_box(results);
            }
        });
    });

    group.bench_function("zipper_based_batch", |b| {
        b.iter(|| {
            for query in &queries {
                let dict_zipper = PathMapZipper::new_from_dict(&dict2);
                let iter = ZipperQueryIterator::new(
                    dict_zipper,
                    black_box(query),
                    black_box(distance),
                    Algorithm::Standard,
                );
                let results: Vec<_> = iter.collect();
                black_box(results);
            }
        });
    });

    group.finish();
}

/// Benchmark memory allocation overhead
fn bench_memory_overhead(c: &mut Criterion) {
    let dict = load_test_dictionary();
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let dict2 = load_test_dictionary();

    let mut group = c.benchmark_group("memory_overhead");

    // Measure iterator creation overhead
    group.bench_function("node_based_create", |b| {
        b.iter(|| {
            let iter = transducer.query(black_box("time"), black_box(1));
            black_box(iter);
        });
    });

    group.bench_function("zipper_based_create", |b| {
        b.iter(|| {
            let dict_zipper = PathMapZipper::new_from_dict(&dict2);
            let iter = ZipperQueryIterator::new(
                dict_zipper,
                black_box("time"),
                black_box(1),
                Algorithm::Standard,
            );
            black_box(iter);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_node_based,
    bench_zipper_based,
    bench_comparison,
    bench_memory_overhead
);
criterion_main!(benches);
