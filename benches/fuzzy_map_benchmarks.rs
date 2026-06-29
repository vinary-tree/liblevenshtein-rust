//! Comprehensive benchmarks for fuzzy map operations and value-filtered queries.
//!
//! This benchmark suite measures:
//! 1. Memory overhead: PathMapDictionary<()> vs PathMapDictionary<V>
//! 2. Query performance: value-filtered vs post-filtered vs unfiltered
//! 3. Filter selectivity: 1%, 10%, 50%, 100% match rates
//! 4. Different value types: u32, Vec<u32>, String
//! 5. Dictionary operations with values: insert_with_value, get_value

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::prelude::*;
use std::collections::HashSet;

/// Create a dictionary without values (baseline)
fn create_dict_no_values(size: usize) -> PathMapDictionary<()> {
    let words: Vec<String> = (0..size).map(|i| format!("term{:04}", i)).collect();
    PathMapDictionary::from_terms(words)
}

/// Create a dictionary with u32 scope IDs
fn create_dict_with_scopes(size: usize, num_scopes: usize) -> PathMapDictionary<u32> {
    let terms: Vec<(String, u32)> = (0..size)
        .map(|i| {
            let term = format!("term{:04}", i);
            let scope = (i % num_scopes) as u32;
            (term, scope)
        })
        .collect();
    PathMapDictionary::from_terms_with_values(terms)
}

/// Create a dictionary with Vec<u32> metadata
fn create_dict_with_metadata(size: usize) -> PathMapDictionary<Vec<u32>> {
    let terms: Vec<(String, Vec<u32>)> = (0..size)
        .map(|i| {
            let term = format!("term{:04}", i);
            let metadata = vec![i as u32, (i * 2) as u32, (i * 3) as u32];
            (term, metadata)
        })
        .collect();
    PathMapDictionary::from_terms_with_values(terms)
}

/// Benchmark: Memory overhead of storing values
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");

    for size in [100, 500, 1000].iter() {
        // Baseline: no values
        group.bench_with_input(BenchmarkId::new("no_values", size), size, |b, &s| {
            b.iter(|| {
                let dict = create_dict_no_values(s);
                black_box(dict)
            })
        });

        // With u32 values
        group.bench_with_input(BenchmarkId::new("u32_values", size), size, |b, &s| {
            b.iter(|| {
                let dict = create_dict_with_scopes(s, 10);
                black_box(dict)
            })
        });

        // With Vec<u32> metadata
        group.bench_with_input(BenchmarkId::new("vec_metadata", size), size, |b, &s| {
            b.iter(|| {
                let dict = create_dict_with_metadata(s);
                black_box(dict)
            })
        });
    }

    group.finish();
}

/// Benchmark: Value-filtered queries vs post-filtering
fn bench_filtered_vs_postfiltered(c: &mut Criterion) {
    let dict = create_dict_with_scopes(1000, 10);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("filtered_vs_postfiltered");

    // Unfiltered baseline (all results)
    group.bench_function("unfiltered", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer.query(black_box("term"), black_box(2)).collect();
            black_box(results)
        })
    });

    // Value-filtered (during traversal)
    group.bench_function("value_filtered", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_filtered(black_box("term"), black_box(2), |scope| *scope == 5)
                .collect();
            black_box(results)
        })
    });

    // Post-filtered (filter after query)
    group.bench_function("post_filtered", |b| {
        b.iter(|| {
            let dict_ref = transducer.dictionary();
            let results: Vec<_> = transducer
                .query(black_box("term"), black_box(2))
                .filter(|term| dict_ref.get_value(term) == Some(5))
                .collect();
            black_box(results)
        })
    });

    group.finish();
}

/// Benchmark: Filter selectivity (how many terms match the filter)
fn bench_filter_selectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_selectivity");

    // Test different selectivity levels
    for (selectivity, num_scopes) in [
        ("1_percent", 100), // 1% of terms match
        ("10_percent", 10), // 10% of terms match
        ("50_percent", 2),  // 50% of terms match
        ("100_percent", 1), // 100% of terms match
    ]
    .iter()
    {
        let dict = create_dict_with_scopes(1000, *num_scopes);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        group.bench_with_input(
            BenchmarkId::new("value_filtered", selectivity),
            selectivity,
            |b, _| {
                b.iter(|| {
                    let results: Vec<_> = transducer
                        .query_filtered(black_box("term"), black_box(2), |scope| *scope == 0)
                        .collect();
                    black_box(results)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("post_filtered", selectivity),
            selectivity,
            |b, _| {
                b.iter(|| {
                    let dict_ref = transducer.dictionary();
                    let results: Vec<_> = transducer
                        .query(black_box("term"), black_box(2))
                        .filter(|term| dict_ref.get_value(term) == Some(0))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Value set filtering (hierarchical scope queries)
fn bench_value_set_filtering(c: &mut Criterion) {
    let dict = create_dict_with_scopes(1000, 20);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("value_set_filtering");

    // Test different set sizes
    for set_size in [1, 5, 10, 20].iter() {
        let value_set: HashSet<u32> = (0..*set_size).collect();

        group.bench_with_input(BenchmarkId::new("value_set", set_size), set_size, |b, _| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_by_value_set(black_box("term"), black_box(2), &value_set)
                    .collect();
                black_box(results)
            })
        });
    }

    group.finish();
}

/// Benchmark: Dictionary operations with values
fn bench_dict_operations_with_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("dict_operations_with_values");

    // Insert with value (u32)
    group.bench_function("insert_u32", |b| {
        let dict: PathMapDictionary<u32> = PathMapDictionary::new();
        let mut counter = 0;
        b.iter(|| {
            dict.insert_with_value(&format!("term{}", counter), black_box(counter % 10));
            counter += 1;
        })
    });

    // Insert with value (Vec<u32>)
    group.bench_function("insert_vec", |b| {
        let dict: PathMapDictionary<Vec<u32>> = PathMapDictionary::new();
        let mut counter = 0;
        b.iter(|| {
            dict.insert_with_value(
                &format!("term{}", counter),
                black_box(vec![counter, counter * 2]),
            );
            counter += 1;
        })
    });

    // Get value (u32)
    group.bench_function("get_value_u32", |b| {
        let dict = create_dict_with_scopes(1000, 10);
        b.iter(|| {
            let value = dict.get_value(black_box("term0500"));
            black_box(value)
        })
    });

    // Get value (Vec<u32>)
    group.bench_function("get_value_vec", |b| {
        let dict = create_dict_with_metadata(1000);
        b.iter(|| {
            let value = dict.get_value(black_box("term0500"));
            black_box(value)
        })
    });

    // Contains (with values)
    group.bench_function("contains_with_values", |b| {
        let dict = create_dict_with_scopes(1000, 10);
        b.iter(|| {
            let exists = dict.contains(black_box("term0500"));
            black_box(exists)
        })
    });

    group.finish();
}

/// Benchmark: Query performance with different value types
fn bench_query_with_value_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_with_value_types");

    // No values (baseline)
    let dict_no_val = create_dict_no_values(1000);
    let transducer_no_val = Transducer::new(dict_no_val, Algorithm::Standard);
    group.bench_function("no_values", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer_no_val
                .query(black_box("term"), black_box(2))
                .collect();
            black_box(results)
        })
    });

    // u32 values
    let dict_u32 = create_dict_with_scopes(1000, 10);
    let transducer_u32 = Transducer::new(dict_u32, Algorithm::Standard);
    group.bench_function("u32_values", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer_u32
                .query(black_box("term"), black_box(2))
                .collect();
            black_box(results)
        })
    });

    // Vec<u32> values
    let dict_vec = create_dict_with_metadata(1000);
    let transducer_vec = Transducer::new(dict_vec, Algorithm::Standard);
    group.bench_function("vec_values", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer_vec
                .query(black_box("term"), black_box(2))
                .collect();
            black_box(results)
        })
    });

    group.finish();
}

/// Benchmark: Varying dictionary sizes with value filtering
fn bench_dict_size_with_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("dict_size_with_filtering");

    for size in [100, 500, 1000, 5000].iter() {
        let dict = create_dict_with_scopes(*size, 10);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_filtered(black_box("term"), black_box(2), |scope| *scope == 5)
                    .collect();
                black_box(results)
            })
        });
    }

    group.finish();
}

/// Benchmark: Edit distance variation with filtering
fn bench_distance_with_filtering(c: &mut Criterion) {
    let dict = create_dict_with_scopes(1000, 10);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("distance_with_filtering");

    for distance in [0, 1, 2, 3].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(distance), distance, |b, &d| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query_filtered(black_box("term"), black_box(d), |scope| *scope == 5)
                    .collect();
                black_box(results)
            })
        });
    }

    group.finish();
}

/// Benchmark: Realistic code completion scenario
fn bench_code_completion_scenario(c: &mut Criterion) {
    // Simulate 10,000 identifiers across 100 scopes (typical large codebase)
    let dict = create_dict_with_scopes(10000, 100);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Simulate hierarchical scope visibility: current scope + parent scopes
    let visible_scopes: HashSet<u32> = [10, 20, 30, 40, 50].iter().cloned().collect();

    c.bench_function("code_completion_realistic", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_by_value_set(black_box("term"), black_box(2), &visible_scopes)
                .take(10) // IDE typically shows top 10 results
                .collect();
            black_box(results)
        })
    });
}

criterion_group!(
    benches,
    bench_memory_overhead,
    bench_filtered_vs_postfiltered,
    bench_filter_selectivity,
    bench_value_set_filtering,
    bench_dict_operations_with_values,
    bench_query_with_value_types,
    bench_dict_size_with_filtering,
    bench_distance_with_filtering,
    bench_code_completion_scenario,
);
criterion_main!(benches);
