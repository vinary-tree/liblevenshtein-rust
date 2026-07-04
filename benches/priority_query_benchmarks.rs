//! Benchmarks comparing PriorityQueryIterator vs OrderedQueryIterator.
//!
//! Measures:
//! - Time to first result
//! - Time to first K results
//! - Time to exhaustive iteration
//! - Memory overhead comparison

use criterion::{criterion_group, criterion_main, Criterion};
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::Dictionary;
use liblevenshtein::transducer::{Algorithm, OrderedQueryIterator, PriorityQueryIterator};
use std::hint::black_box;

// ============================================================================
// Test Data Generation
// ============================================================================

/// Generate a synthetic dictionary of given size
fn generate_dictionary(size: usize) -> Vec<String> {
    let prefixes = [
        "pre", "un", "re", "dis", "over", "mis", "out", "sub", "trans", "inter",
    ];
    let roots = [
        "act", "form", "port", "duct", "ject", "tract", "struct", "scribe", "spect", "vert",
    ];
    let suffixes = [
        "ion", "ive", "ment", "ness", "able", "ible", "ous", "ful", "less", "ly",
    ];

    let mut words = Vec::with_capacity(size);

    for root in &roots {
        words.push(root.to_string());
        if words.len() >= size {
            return words;
        }
    }

    for prefix in &prefixes {
        for root in &roots {
            words.push(format!("{}{}", prefix, root));
            if words.len() >= size {
                return words;
            }
        }
    }

    for root in &roots {
        for suffix in &suffixes {
            words.push(format!("{}{}", root, suffix));
            if words.len() >= size {
                return words;
            }
        }
    }

    for prefix in &prefixes {
        for root in &roots {
            for suffix in &suffixes {
                words.push(format!("{}{}{}", prefix, root, suffix));
                if words.len() >= size {
                    return words;
                }
            }
        }
    }

    while words.len() < size {
        words.push(format!("word{}", words.len()));
    }

    words
}

/// Build a DynamicDawg from a dictionary
fn build_dawg(dictionary: &[String]) -> DynamicDawg {
    let dawg = DynamicDawg::new();
    for term in dictionary {
        dawg.insert(term);
    }
    dawg
}

// ============================================================================
// First-K Result Benchmarks
// ============================================================================

fn bench_first_result(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_iterator/first_result");

    let sizes = [1_000, 10_000];
    let queries = [
        ("exact", "preact"),
        ("typo", "preeact"),
        ("distant", "preeeact"),
    ];

    for size in sizes {
        let dictionary = generate_dictionary(size);
        let dawg = build_dawg(&dictionary);

        for (query_name, query) in &queries {
            let max_dist = 2;

            // Priority iterator - first result
            group.bench_function(format!("{}_priority_{}", size, query_name), |b| {
                b.iter(|| {
                    let iter = PriorityQueryIterator::new(
                        dawg.root(),
                        black_box(query),
                        black_box(max_dist),
                        Algorithm::Standard,
                    );
                    iter.into_iter().next()
                });
            });

            // Ordered iterator - first result
            group.bench_function(format!("{}_ordered_{}", size, query_name), |b| {
                b.iter(|| {
                    let iter = OrderedQueryIterator::new(
                        dawg.root(),
                        black_box(query.to_string()),
                        black_box(max_dist),
                        Algorithm::Standard,
                    );
                    iter.into_iter().next()
                });
            });
        }
    }

    group.finish();
}

fn bench_first_k_results(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_iterator/first_k");

    let size = 10_000;
    let dictionary = generate_dictionary(size);
    let dawg = build_dawg(&dictionary);

    let query = "preaction";
    let max_dist = 2;
    let k_values = [1, 5, 10, 25];

    for k in k_values {
        // Priority iterator - first K results
        group.bench_function(format!("priority_k{}", k), |b| {
            b.iter(|| {
                let iter = PriorityQueryIterator::new(
                    dawg.root(),
                    black_box(query),
                    black_box(max_dist),
                    Algorithm::Standard,
                );
                iter.take(k).collect::<Vec<_>>()
            });
        });

        // Ordered iterator - first K results
        group.bench_function(format!("ordered_k{}", k), |b| {
            b.iter(|| {
                let iter = OrderedQueryIterator::new(
                    dawg.root(),
                    black_box(query.to_string()),
                    black_box(max_dist),
                    Algorithm::Standard,
                );
                iter.take(k).collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Exhaustive Iteration Benchmarks
// ============================================================================

fn bench_exhaustive_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_iterator/exhaustive");
    group.sample_size(30);

    let sizes = [1_000, 5_000];

    for size in sizes {
        let dictionary = generate_dictionary(size);
        let dawg = build_dawg(&dictionary);

        let query = "preact";
        let max_dist = 2;

        // Priority iterator - all results
        group.bench_function(format!("{}_priority_all", size), |b| {
            b.iter(|| {
                let iter = PriorityQueryIterator::new(
                    dawg.root(),
                    black_box(query),
                    black_box(max_dist),
                    Algorithm::Standard,
                );
                iter.collect::<Vec<_>>()
            });
        });

        // Ordered iterator - all results
        group.bench_function(format!("{}_ordered_all", size), |b| {
            b.iter(|| {
                let iter = OrderedQueryIterator::new(
                    dawg.root(),
                    black_box(query.to_string()),
                    black_box(max_dist),
                    Algorithm::Standard,
                );
                iter.collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Algorithm Variant Benchmarks
// ============================================================================

fn bench_algorithm_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_iterator/algorithms");

    let size = 5_000;
    let dictionary = generate_dictionary(size);
    let dawg = build_dawg(&dictionary);

    let queries = [
        ("transposition", "tset", Algorithm::Transposition),
        ("standard", "test", Algorithm::Standard),
    ];

    for (name, query, algorithm) in queries {
        let max_dist = 2;

        // Priority iterator
        group.bench_function(format!("priority_{}", name), |b| {
            b.iter(|| {
                let iter = PriorityQueryIterator::new(
                    dawg.root(),
                    black_box(query),
                    black_box(max_dist),
                    algorithm,
                );
                iter.take(10).collect::<Vec<_>>()
            });
        });

        // Ordered iterator
        group.bench_function(format!("ordered_{}", name), |b| {
            b.iter(|| {
                let iter = OrderedQueryIterator::new(
                    dawg.root(),
                    black_box(query.to_string()),
                    black_box(max_dist),
                    algorithm,
                );
                iter.take(10).collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Distance Threshold Benchmarks
// ============================================================================

fn bench_distance_thresholds(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_iterator/distance");

    let size = 5_000;
    let dictionary = generate_dictionary(size);
    let dawg = build_dawg(&dictionary);

    let query = "preaction";

    for max_dist in [1, 2, 3] {
        // Priority iterator
        group.bench_function(format!("priority_d{}", max_dist), |b| {
            b.iter(|| {
                let iter = PriorityQueryIterator::new(
                    dawg.root(),
                    black_box(query),
                    black_box(max_dist),
                    Algorithm::Standard,
                );
                iter.take(20).collect::<Vec<_>>()
            });
        });

        // Ordered iterator
        group.bench_function(format!("ordered_d{}", max_dist), |b| {
            b.iter(|| {
                let iter = OrderedQueryIterator::new(
                    dawg.root(),
                    black_box(query.to_string()),
                    black_box(max_dist),
                    Algorithm::Standard,
                );
                iter.take(20).collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(first_k_benches, bench_first_result, bench_first_k_results,);

criterion_group!(exhaustive_benches, bench_exhaustive_iteration,);

criterion_group!(
    variant_benches,
    bench_algorithm_variants,
    bench_distance_thresholds,
);

criterion_main!(first_k_benches, exhaustive_benches, variant_benches);
