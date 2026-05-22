//! Comprehensive comparison of exact, prefix, and substring matching modes
//!
//! This benchmark systematically compares:
//! - Exact matching: Standard dictionaries without prefix mode
//! - Prefix matching: Standard dictionaries with .prefix()
//! - Substring matching: Suffix automata (inherently substring-based)

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::suffix_automaton::SuffixAutomaton;
use liblevenshtein::transducer::{Algorithm, Transducer};

fn load_dictionary() -> Vec<String> {
    std::fs::read_to_string("/usr/share/dict/words")
        .unwrap_or_else(|_| {
            (0..10000)
                .map(|i| format!("word{}", i))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .lines()
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() >= 3 && s.len() <= 15)
        .take(5000)
        .collect()
}

/// Benchmark: Compare matching modes at different edit distances
fn bench_matching_modes_by_distance(c: &mut Criterion) {
    let words = load_dictionary();
    let pathmap_dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(words.iter().map(|s| s.as_str()));
    let suffix_dict: SuffixAutomaton<()> =
        SuffixAutomaton::from_texts(words.iter().map(|s| s.as_str()));

    let mut group = c.benchmark_group("matching_modes_by_distance");
    group.plot_config(PlotConfiguration::default());

    let query = "test";

    for distance in [0, 1, 2, 3] {
        // Exact matching
        group.bench_with_input(
            BenchmarkId::new("exact", distance),
            &distance,
            |b, &dist| {
                b.iter(|| {
                    let results: Vec<_> =
                        Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                            .query_ordered(black_box(query), dist)
                            .take(10)
                            .collect();
                    black_box(results);
                });
            },
        );

        // Prefix matching
        group.bench_with_input(
            BenchmarkId::new("prefix", distance),
            &distance,
            |b, &dist| {
                b.iter(|| {
                    let results: Vec<_> =
                        Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                            .query_ordered(black_box(query), dist)
                            .prefix()
                            .take(10)
                            .collect();
                    black_box(results);
                });
            },
        );

        // Substring matching
        group.bench_with_input(
            BenchmarkId::new("substring", distance),
            &distance,
            |b, &dist| {
                b.iter(|| {
                    let results: Vec<_> = Transducer::new(suffix_dict.clone(), Algorithm::Standard)
                        .query_ordered(black_box(query), dist)
                        .take(10)
                        .collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Compare matching modes with different query lengths
fn bench_matching_modes_by_query_length(c: &mut Criterion) {
    let words = load_dictionary();
    let pathmap_dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(words.iter().map(|s| s.as_str()));
    let suffix_dict: SuffixAutomaton<()> =
        SuffixAutomaton::from_texts(words.iter().map(|s| s.as_str()));

    let mut group = c.benchmark_group("matching_modes_by_query_length");

    let queries = [("te", 2), ("test", 4), ("testing", 7), ("testability", 11)];

    for (query, len) in queries.iter() {
        // Exact matching
        group.bench_with_input(BenchmarkId::new("exact", len), len, |b, _| {
            b.iter(|| {
                let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                    .query_ordered(black_box(query), 1)
                    .take(10)
                    .collect();
                black_box(results);
            });
        });

        // Prefix matching
        group.bench_with_input(BenchmarkId::new("prefix", len), len, |b, _| {
            b.iter(|| {
                let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                    .query_ordered(black_box(query), 1)
                    .prefix()
                    .take(10)
                    .collect();
                black_box(results);
            });
        });

        // Substring matching
        group.bench_with_input(BenchmarkId::new("substring", len), len, |b, _| {
            b.iter(|| {
                let results: Vec<_> = Transducer::new(suffix_dict.clone(), Algorithm::Standard)
                    .query_ordered(black_box(query), 1)
                    .take(10)
                    .collect();
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark: Result set sizes (how many results are produced)
fn bench_result_set_sizes(c: &mut Criterion) {
    let words = load_dictionary();
    let pathmap_dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(words.iter().map(|s| s.as_str()));
    let suffix_dict: SuffixAutomaton<()> =
        SuffixAutomaton::from_texts(words.iter().map(|s| s.as_str()));

    let mut group = c.benchmark_group("result_set_sizes");

    let query = "test";
    let distance = 1;

    // Exact matching - collect ALL results
    group.bench_function("exact_all_results", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                .query_ordered(black_box(query), distance)
                .collect();
            black_box(results);
        });
    });

    // Prefix matching - collect ALL results
    group.bench_function("prefix_all_results", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                .query_ordered(black_box(query), distance)
                .prefix()
                .collect();
            black_box(results);
        });
    });

    // Substring matching - collect ALL results (may be very large!)
    group.bench_function("substring_all_results", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(suffix_dict.clone(), Algorithm::Standard)
                .query_ordered(black_box(query), distance)
                .take(1000) // Limit to prevent excessive runtime
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

/// Benchmark: Ordered vs unordered queries
fn bench_ordered_vs_unordered(c: &mut Criterion) {
    let words = load_dictionary();
    let pathmap_dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(words.iter().map(|s| s.as_str()));
    let suffix_dict: SuffixAutomaton<()> =
        SuffixAutomaton::from_texts(words.iter().map(|s| s.as_str()));

    let mut group = c.benchmark_group("ordered_vs_unordered");

    let query = "test";
    let distance = 2;

    // Exact - ordered
    group.bench_function("exact_ordered", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                .query_ordered(black_box(query), distance)
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Exact - unordered
    group.bench_function("exact_unordered", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                .query(black_box(query), distance)
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Prefix - ordered
    group.bench_function("prefix_ordered", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                .query_ordered(black_box(query), distance)
                .prefix()
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Prefix - unordered
    group.bench_function("prefix_unordered", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                .query(black_box(query), distance)
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Substring - ordered
    group.bench_function("substring_ordered", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(suffix_dict.clone(), Algorithm::Standard)
                .query_ordered(black_box(query), distance)
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Substring - unordered
    group.bench_function("substring_unordered", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(suffix_dict.clone(), Algorithm::Standard)
                .query(black_box(query), distance)
                .take(10)
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

/// Benchmark: Dictionary construction overhead
fn bench_dictionary_construction(c: &mut Criterion) {
    let words = load_dictionary();
    let mut group = c.benchmark_group("dictionary_construction");

    let total_chars: usize = words.iter().map(|s| s.len()).sum();
    group.throughput(Throughput::Bytes(total_chars as u64));

    group.bench_function("pathmap_construction", |b| {
        b.iter(|| {
            let dict: PathMapDictionary<()> =
                PathMapDictionary::from_terms(black_box(words.iter().map(|s| s.as_str())));
            black_box(dict);
        });
    });

    group.bench_function("suffix_automaton_construction", |b| {
        b.iter(|| {
            let dict: SuffixAutomaton<()> =
                SuffixAutomaton::from_texts(black_box(words.iter().map(|s| s.as_str())));
            black_box(dict);
        });
    });

    group.finish();
}

/// Benchmark: Different algorithms (Standard vs Transposition)
fn bench_algorithms(c: &mut Criterion) {
    let words = load_dictionary();
    let pathmap_dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(words.iter().map(|s| s.as_str()));

    let mut group = c.benchmark_group("algorithms");

    let query = "test";
    let distance = 2;

    // Standard algorithm - exact
    group.bench_function("standard_exact", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                .query_ordered(black_box(query), distance)
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Standard algorithm - prefix
    group.bench_function("standard_prefix", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Standard)
                .query_ordered(black_box(query), distance)
                .prefix()
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Transposition algorithm - exact
    group.bench_function("transposition_exact", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Transposition)
                .query_ordered(black_box(query), distance)
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Transposition algorithm - prefix
    group.bench_function("transposition_prefix", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(pathmap_dict.clone(), Algorithm::Transposition)
                .query_ordered(black_box(query), distance)
                .prefix()
                .take(10)
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_matching_modes_by_distance,
    bench_matching_modes_by_query_length,
    bench_result_set_sizes,
    bench_ordered_vs_unordered,
    bench_dictionary_construction,
    bench_algorithms,
);
criterion_main!(benches);
