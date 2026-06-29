//! Comprehensive Benchmark: Parameterized vs Universal Levenshtein Automata
//!
//! This benchmark compares two approaches to Levenshtein automata:
//!
//! 1. **Parameterized Automata** (current production implementation):
//!    - Runtime/lazy state construction during dictionary traversal
//!    - Each query builds automaton states on-demand
//!    - Optimized with StatePool for allocation reuse
//!
//! 2. **Universal Automata** (precomputed, parameter-free):
//!    - Fixed structure for given max_distance
//!    - Works for ANY word without modification
//!    - Currently implements accepts(word, input) primitive
//!
//! ## Benchmark Dimensions
//!
//! - **Single Query**: Cold start overhead
//! - **Batch Queries**: Amortization analysis (10, 100, 1000 queries)
//! - **Distance Scaling**: Complexity growth (d=1, 2, 3, 4)
//! - **Dictionary Size**: Scalability (100, 1K, 10K terms)
//! - **Algorithm Variants**: Standard & Transposition
//!
//! ## Metrics
//!
//! - Query execution time (mean, stddev)
//! - Memory usage characteristics
//! - Scalability patterns
//!
//! ## Use-Case Recommendations
//!
//! Results will inform when to choose each approach based on:
//! - Number of queries (single vs batch)
//! - Dictionary size and structure
//! - Distance requirements
//! - Memory constraints

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::universal::{Standard as UniversalStandard, UniversalAutomaton};

/// Create a dictionary of the specified size with varied word lengths
fn create_dictionary(size: usize) -> DynamicDawg {
    let words: Vec<String> = (0..size)
        .map(|i| {
            // Create words of varying lengths (4-12 characters)
            let len = 4 + (i % 9);
            format!("word{:0width$}", i, width = len - 4)
        })
        .collect();
    DynamicDawg::from_terms(words)
}

/// Generate dictionary terms as a Vec<String> for universal automaton iteration
fn generate_dictionary_terms(size: usize) -> Vec<String> {
    (0..size)
        .map(|i| {
            let len = 4 + (i % 9);
            format!("word{:0width$}", i, width = len - 4)
        })
        .collect()
}

// ====================================================================================
// Benchmark 1: Single Query Performance (Cold Start)
// ====================================================================================

fn bench_single_query_parameterized_vs_universal(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_query_cold_start");

    let dict = create_dictionary(1000);
    let terms = generate_dictionary_terms(1000);
    let query_word = "word500";
    let distance: usize = 2;

    // Parameterized approach (production)
    group.bench_function("parameterized", |b| {
        b.iter(|| {
            let transducer = Transducer::new(dict.clone(), Algorithm::Standard);
            let results: Vec<_> = transducer
                .query(black_box(query_word), black_box(distance))
                .collect();
            black_box(results);
        });
    });

    // Universal approach (simulated dictionary query)
    group.bench_function("universal", |b| {
        b.iter(|| {
            let automaton = UniversalAutomaton::<UniversalStandard>::new(distance as u8);
            let results: Vec<&str> = terms
                .iter()
                .filter(|dict_word| automaton.accepts(black_box(dict_word), black_box(query_word)))
                .map(|s| s.as_str())
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

// ====================================================================================
// Benchmark 2: Batch Query Performance (Amortization)
// ====================================================================================

fn bench_batch_queries_parameterized_vs_universal(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_queries");

    let dict = create_dictionary(1000);
    let terms = generate_dictionary_terms(1000);
    let distance: usize = 2;

    // Query words to test
    let query_words: Vec<String> = vec![
        "word100", "word200", "word300", "word400", "word500", "word600", "word700", "word800",
        "word900", "test",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();

    for batch_size in [10, 100, 1000].iter() {
        let queries: Vec<&str> = query_words
            .iter()
            .cycle()
            .take(*batch_size)
            .map(|s| s.as_str())
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));

        // Parameterized: Reuse transducer across queries
        group.bench_with_input(
            BenchmarkId::new("parameterized", batch_size),
            batch_size,
            |b, _| {
                let transducer = Transducer::new(dict.clone(), Algorithm::Standard);
                b.iter(|| {
                    for query in &queries {
                        let results: Vec<_> = transducer
                            .query(black_box(query), black_box(distance))
                            .collect();
                        black_box(results);
                    }
                });
            },
        );

        // Universal: Reuse automaton across queries
        group.bench_with_input(
            BenchmarkId::new("universal", batch_size),
            batch_size,
            |b, _| {
                let automaton = UniversalAutomaton::<UniversalStandard>::new(distance as u8);
                b.iter(|| {
                    for query in &queries {
                        let results: Vec<&str> = terms
                            .iter()
                            .filter(|dict_word| {
                                automaton.accepts(black_box(dict_word), black_box(query))
                            })
                            .map(|s| s.as_str())
                            .collect();
                        black_box(results);
                    }
                });
            },
        );
    }

    group.finish();
}

// ====================================================================================
// Benchmark 3: Distance Scaling
// ====================================================================================

fn bench_distance_scaling_parameterized_vs_universal(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_scaling");

    let dict = create_dictionary(1000);
    let terms = generate_dictionary_terms(1000);
    let query_word = "word500";

    for distance in [1usize, 2, 3, 4].iter() {
        // Parameterized
        group.bench_with_input(
            BenchmarkId::new("parameterized", distance),
            distance,
            |b, &d| {
                let transducer = Transducer::new(dict.clone(), Algorithm::Standard);
                b.iter(|| {
                    let results: Vec<_> = transducer
                        .query(black_box(query_word), black_box(d))
                        .collect();
                    black_box(results);
                });
            },
        );

        // Universal
        group.bench_with_input(
            BenchmarkId::new("universal", distance),
            distance,
            |b, &d| {
                let automaton = UniversalAutomaton::<UniversalStandard>::new(d as u8);
                b.iter(|| {
                    let results: Vec<&str> = terms
                        .iter()
                        .filter(|dict_word| {
                            automaton.accepts(black_box(dict_word), black_box(query_word))
                        })
                        .map(|s| s.as_str())
                        .collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

// ====================================================================================
// Benchmark 4: Dictionary Size Scaling
// ====================================================================================

fn bench_dictionary_size_scaling_parameterized_vs_universal(c: &mut Criterion) {
    let mut group = c.benchmark_group("dictionary_size_scaling");

    let query_word = "word500";
    let distance: usize = 2;

    for size in [100, 1000, 10000].iter() {
        let dict = create_dictionary(*size);
        let terms = generate_dictionary_terms(*size);

        group.throughput(Throughput::Elements(*size as u64));

        // Parameterized
        group.bench_with_input(BenchmarkId::new("parameterized", size), size, |b, _| {
            let transducer = Transducer::new(dict.clone(), Algorithm::Standard);
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query(black_box(query_word), black_box(distance))
                    .collect();
                black_box(results);
            });
        });

        // Universal
        group.bench_with_input(BenchmarkId::new("universal", size), size, |b, _| {
            let automaton = UniversalAutomaton::<UniversalStandard>::new(distance as u8);
            b.iter(|| {
                let results: Vec<&str> = terms
                    .iter()
                    .filter(|dict_word| {
                        automaton.accepts(black_box(dict_word), black_box(query_word))
                    })
                    .map(|s| s.as_str())
                    .collect();
                black_box(results);
            });
        });
    }

    group.finish();
}

// ====================================================================================
// Benchmark 5: Primitive Operation - Single accepts() call
// ====================================================================================

fn bench_primitive_accepts_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive_accepts");

    let word1 = "testing";
    let word2 = "text";

    for distance in [1usize, 2, 3].iter() {
        group.bench_with_input(
            BenchmarkId::new("universal_accepts", distance),
            distance,
            |b, &d| {
                let automaton = UniversalAutomaton::<UniversalStandard>::new(d as u8);
                b.iter(|| {
                    let result = automaton.accepts(black_box(word1), black_box(word2));
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_query_parameterized_vs_universal,
    bench_batch_queries_parameterized_vs_universal,
    bench_distance_scaling_parameterized_vs_universal,
    bench_dictionary_size_scaling_parameterized_vs_universal,
    bench_primitive_accepts_operation,
);

criterion_main!(benches);
