//! Benchmark comparing Levenshtein automaton performance vs. linear scan with distance functions.
//!
//! This benchmark demonstrates the performance advantage of using a Levenshtein automaton
//! over a naive linear scan with distance computation for each dictionary word.
//!
//! The automaton exploits the structure of the dictionary (trie/DAWG) to prune the search
//! space, while linear scan must compute distance for every word.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::distance::*;
use liblevenshtein::prelude::*;

// ============================================================================
// Test Data Setup
// ============================================================================

/// Load a small dictionary for benchmarking
fn load_small_dict() -> Vec<String> {
    vec![
        "test", "testing", "tested", "tester", "tests", "best", "rest", "nest", "quest", "fest",
        "apple", "apply", "ample", "maple", "staple", "code", "coder", "decode", "encode", "node",
        "data", "date", "gate", "rate", "state",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Load a medium dictionary (100 words)
fn load_medium_dict() -> Vec<String> {
    let base_words = vec![
        "algorithm",
        "structure",
        "computer",
        "science",
        "program",
        "function",
        "variable",
        "constant",
        "iterator",
        "reference",
        "pattern",
        "matching",
        "distance",
        "levenshtein",
        "automaton",
        "transducer",
        "dictionary",
        "benchmark",
        "performance",
        "optimization",
    ];

    let mut dict = Vec::new();
    for word in &base_words {
        dict.push(word.to_string());
        // Add variations
        dict.push(format!("{}s", word));
        dict.push(format!("{}ed", word));
        dict.push(format!("{}ing", word));
        dict.push(format!("pre{}", word));
    }
    dict
}

/// Load a large dictionary (1000+ words) - realistic scenario
fn load_large_dict() -> Vec<String> {
    // Read from system dictionary if available, otherwise generate
    if let Ok(contents) = std::fs::read_to_string("/usr/share/dict/words") {
        contents
            .lines()
            .take(1000)
            .map(|s| s.trim().to_lowercase())
            .filter(|s| s.len() >= 3 && s.len() <= 15)
            .collect()
    } else {
        // Fallback: generate synthetic dictionary
        let mut dict = Vec::new();
        for i in 0..1000 {
            dict.push(format!("word{:04}", i));
        }
        dict
    }
}

// ============================================================================
// Linear Scan Implementations
// ============================================================================

/// Linear scan using Standard distance
fn linear_scan_standard(dict: &[String], query: &str, max_distance: usize) -> Vec<String> {
    dict.iter()
        .filter(|word| standard_distance(query, word) <= max_distance)
        .cloned()
        .collect()
}

/// Linear scan using Transposition distance
fn linear_scan_transposition(dict: &[String], query: &str, max_distance: usize) -> Vec<String> {
    dict.iter()
        .filter(|word| transposition_distance(query, word) <= max_distance)
        .cloned()
        .collect()
}

/// Linear scan using MergeAndSplit distance
fn linear_scan_merge_split(dict: &[String], query: &str, max_distance: usize) -> Vec<String> {
    let cache = create_memo_cache();
    dict.iter()
        .filter(|word| merge_and_split_distance(query, word, &cache) <= max_distance)
        .cloned()
        .collect()
}

// ============================================================================
// Automaton Query Implementations
// ============================================================================

/// Query using Levenshtein automaton with Standard algorithm
fn automaton_query_standard(
    transducer: &Transducer<DoubleArrayTrie>,
    query: &str,
    max_distance: usize,
) -> Vec<String> {
    transducer.query(query, max_distance).collect()
}

/// Query using Levenshtein automaton with Transposition algorithm
fn automaton_query_transposition(
    transducer: &Transducer<DoubleArrayTrie>,
    query: &str,
    max_distance: usize,
) -> Vec<String> {
    transducer.query(query, max_distance).collect()
}

/// Query using Levenshtein automaton with MergeAndSplit algorithm
fn automaton_query_merge_split(
    transducer: &Transducer<DoubleArrayTrie>,
    query: &str,
    max_distance: usize,
) -> Vec<String> {
    transducer.query(query, max_distance).collect()
}

// ============================================================================
// Benchmarks: Standard Algorithm
// ============================================================================

fn bench_standard_small_dict(c: &mut Criterion) {
    let dict = load_small_dict();
    let dat = DoubleArrayTrie::from_terms(dict.clone());
    let transducer = Transducer::new(dat, Algorithm::Standard);

    let query = "test";
    let max_distance = 2;

    let mut group = c.benchmark_group("standard/small_dict");
    group.throughput(Throughput::Elements(dict.len() as u64));

    group.bench_function("linear_scan", |b| {
        b.iter(|| {
            black_box(linear_scan_standard(
                black_box(&dict),
                black_box(query),
                black_box(max_distance),
            ))
        })
    });

    group.bench_function("automaton", |b| {
        b.iter(|| {
            black_box(automaton_query_standard(
                black_box(&transducer),
                black_box(query),
                black_box(max_distance),
            ))
        })
    });

    group.finish();
}

fn bench_standard_medium_dict(c: &mut Criterion) {
    let dict = load_medium_dict();
    let dat = DoubleArrayTrie::from_terms(dict.clone());
    let transducer = Transducer::new(dat, Algorithm::Standard);

    let query = "algorithm";

    let mut group = c.benchmark_group("standard/medium_dict");
    group.throughput(Throughput::Elements(dict.len() as u64));

    for &max_distance in &[1, 2, 3] {
        group.bench_with_input(
            BenchmarkId::new("linear_scan", max_distance),
            &max_distance,
            |b, &max_dist| {
                b.iter(|| {
                    black_box(linear_scan_standard(
                        black_box(&dict),
                        black_box(query),
                        black_box(max_dist),
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("automaton", max_distance),
            &max_distance,
            |b, &max_dist| {
                b.iter(|| {
                    black_box(automaton_query_standard(
                        black_box(&transducer),
                        black_box(query),
                        black_box(max_dist),
                    ))
                })
            },
        );
    }

    group.finish();
}

fn bench_standard_large_dict(c: &mut Criterion) {
    let dict = load_large_dict();
    let dat = DoubleArrayTrie::from_terms(dict.clone());
    let transducer = Transducer::new(dat, Algorithm::Standard);

    let query = "programming";

    let mut group = c.benchmark_group("standard/large_dict");
    group.throughput(Throughput::Elements(dict.len() as u64));
    group.sample_size(50); // Reduce sample size for large benchmarks

    for &max_distance in &[1, 2] {
        group.bench_with_input(
            BenchmarkId::new("linear_scan", max_distance),
            &max_distance,
            |b, &max_dist| {
                b.iter(|| {
                    black_box(linear_scan_standard(
                        black_box(&dict),
                        black_box(query),
                        black_box(max_dist),
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("automaton", max_distance),
            &max_distance,
            |b, &max_dist| {
                b.iter(|| {
                    black_box(automaton_query_standard(
                        black_box(&transducer),
                        black_box(query),
                        black_box(max_dist),
                    ))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks: Transposition Algorithm
// ============================================================================

fn bench_transposition_medium_dict(c: &mut Criterion) {
    let dict = load_medium_dict();
    let dat = DoubleArrayTrie::from_terms(dict.clone());
    let transducer = Transducer::new(dat, Algorithm::Transposition);

    let query = "algorithm";
    let max_distance = 2;

    let mut group = c.benchmark_group("transposition/medium_dict");
    group.throughput(Throughput::Elements(dict.len() as u64));

    group.bench_function("linear_scan", |b| {
        b.iter(|| {
            black_box(linear_scan_transposition(
                black_box(&dict),
                black_box(query),
                black_box(max_distance),
            ))
        })
    });

    group.bench_function("automaton", |b| {
        b.iter(|| {
            black_box(automaton_query_transposition(
                black_box(&transducer),
                black_box(query),
                black_box(max_distance),
            ))
        })
    });

    group.finish();
}

fn bench_transposition_large_dict(c: &mut Criterion) {
    let dict = load_large_dict();
    let dat = DoubleArrayTrie::from_terms(dict.clone());
    let transducer = Transducer::new(dat, Algorithm::Transposition);

    let query = "programming";
    let max_distance = 2;

    let mut group = c.benchmark_group("transposition/large_dict");
    group.throughput(Throughput::Elements(dict.len() as u64));
    group.sample_size(50);

    group.bench_function("linear_scan", |b| {
        b.iter(|| {
            black_box(linear_scan_transposition(
                black_box(&dict),
                black_box(query),
                black_box(max_distance),
            ))
        })
    });

    group.bench_function("automaton", |b| {
        b.iter(|| {
            black_box(automaton_query_transposition(
                black_box(&transducer),
                black_box(query),
                black_box(max_distance),
            ))
        })
    });

    group.finish();
}

// ============================================================================
// Benchmarks: MergeAndSplit Algorithm
// ============================================================================

fn bench_merge_split_medium_dict(c: &mut Criterion) {
    let dict = load_medium_dict();
    let dat = DoubleArrayTrie::from_terms(dict.clone());
    let transducer = Transducer::new(dat, Algorithm::MergeAndSplit);

    let query = "algorithm";
    let max_distance = 2;

    let mut group = c.benchmark_group("merge_split/medium_dict");
    group.throughput(Throughput::Elements(dict.len() as u64));

    group.bench_function("linear_scan", |b| {
        b.iter(|| {
            black_box(linear_scan_merge_split(
                black_box(&dict),
                black_box(query),
                black_box(max_distance),
            ))
        })
    });

    group.bench_function("automaton", |b| {
        b.iter(|| {
            black_box(automaton_query_merge_split(
                black_box(&transducer),
                black_box(query),
                black_box(max_distance),
            ))
        })
    });

    group.finish();
}

// ============================================================================
// Benchmarks: Dictionary Construction Overhead
// ============================================================================

fn bench_construction_overhead(c: &mut Criterion) {
    let dict = load_medium_dict();

    let mut group = c.benchmark_group("construction");
    group.throughput(Throughput::Elements(dict.len() as u64));

    // Benchmark just the dictionary construction
    group.bench_function("double_array_trie", |b| {
        b.iter(|| black_box(DoubleArrayTrie::from_terms(black_box(dict.clone()))))
    });

    // Benchmark construction + transducer creation
    group.bench_function("transducer_standard", |b| {
        b.iter(|| {
            let dat = DoubleArrayTrie::from_terms(black_box(dict.clone()));
            black_box(Transducer::new(dat, Algorithm::Standard))
        })
    });

    group.finish();
}

// ============================================================================
// Benchmarks: Varying Query Lengths
// ============================================================================

fn bench_query_length_impact(c: &mut Criterion) {
    let dict = load_medium_dict();
    let dat = DoubleArrayTrie::from_terms(dict.clone());
    let transducer = Transducer::new(dat, Algorithm::Standard);

    let queries = vec![
        ("sh", "short"),                  // 2 chars
        ("test", "medium"),               // 4 chars
        ("algorithm", "long"),            // 9 chars
        ("implementations", "very_long"), // 15 chars
    ];

    let max_distance = 2;

    let mut group = c.benchmark_group("query_length");

    for (query, label) in queries {
        group.bench_with_input(BenchmarkId::new("linear_scan", label), &query, |b, &q| {
            b.iter(|| {
                black_box(linear_scan_standard(
                    black_box(&dict),
                    black_box(q),
                    black_box(max_distance),
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("automaton", label), &query, |b, &q| {
            b.iter(|| {
                black_box(automaton_query_standard(
                    black_box(&transducer),
                    black_box(q),
                    black_box(max_distance),
                ))
            })
        });
    }

    group.finish();
}

// ============================================================================
// Benchmarks: Best Case vs Worst Case
// ============================================================================

fn bench_best_worst_case(c: &mut Criterion) {
    let dict = load_medium_dict();
    let dat = DoubleArrayTrie::from_terms(dict.clone());
    let transducer = Transducer::new(dat, Algorithm::Standard);

    let max_distance = 2;

    let mut group = c.benchmark_group("best_worst_case");

    // Best case: query matches exactly (early termination possible)
    let best_case_query = dict[0].as_str();

    group.bench_function("linear_scan/best_case", |b| {
        b.iter(|| {
            black_box(linear_scan_standard(
                black_box(&dict),
                black_box(best_case_query),
                black_box(max_distance),
            ))
        })
    });

    group.bench_function("automaton/best_case", |b| {
        b.iter(|| {
            black_box(automaton_query_standard(
                black_box(&transducer),
                black_box(best_case_query),
                black_box(max_distance),
            ))
        })
    });

    // Worst case: query matches nothing (must check all words)
    let worst_case_query = "zzzzzzzzzz";

    group.bench_function("linear_scan/worst_case", |b| {
        b.iter(|| {
            black_box(linear_scan_standard(
                black_box(&dict),
                black_box(worst_case_query),
                black_box(max_distance),
            ))
        })
    });

    group.bench_function("automaton/worst_case", |b| {
        b.iter(|| {
            black_box(automaton_query_standard(
                black_box(&transducer),
                black_box(worst_case_query),
                black_box(max_distance),
            ))
        })
    });

    group.finish();
}

// ============================================================================
// Benchmark Groups
// ============================================================================

criterion_group!(
    standard_benches,
    bench_standard_small_dict,
    bench_standard_medium_dict,
    bench_standard_large_dict,
);

criterion_group!(
    transposition_benches,
    bench_transposition_medium_dict,
    bench_transposition_large_dict,
);

criterion_group!(merge_split_benches, bench_merge_split_medium_dict,);

criterion_group!(overhead_benches, bench_construction_overhead,);

criterion_group!(
    variation_benches,
    bench_query_length_impact,
    bench_best_worst_case,
);

criterion_main!(
    standard_benches,
    transposition_benches,
    merge_split_benches,
    overhead_benches,
    variation_benches,
);
