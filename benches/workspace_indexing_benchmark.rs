//! Benchmarks for Parallel Workspace Indexing
//!
//! Compares performance of sequential fold vs binary tree reduction for merging
//! per-document dictionaries in contextual completion scenarios.
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Run all workspace indexing benchmarks
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench workspace_indexing_benchmark
//!
//! # Run specific benchmark group
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench workspace_indexing_benchmark -- merge_strategy
//!
//! # Save baseline for comparison
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench workspace_indexing_benchmark -- --save-baseline main
//!
//! # Compare against baseline
//! RUSTFLAGS="-C target-cpu=native" cargo bench --bench workspace_indexing_benchmark -- --baseline main
//! ```

use std::hint::black_box;
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::{Dictionary, MutableMappedDictionary};
use rayon::prelude::*;
use rustc_hash::FxHashSet;

type ContextId = u32;

/// Generate synthetic terms for benchmarking
fn generate_terms(doc_id: ContextId, terms_per_doc: usize) -> Vec<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    doc_id.hash(&mut hasher);
    let seed = hasher.finish();

    (0..terms_per_doc)
        .map(|i| {
            let pattern = (seed + i as u64) % 100;
            if pattern < 40 {
                format!("fn{}_{}", doc_id, i)
            } else if pattern < 70 {
                format!("var{}_{}", doc_id, i)
            } else if pattern < 90 {
                format!("Class{}_{}", doc_id, i)
            } else {
                format!("doc{}_unique_{}", doc_id, i)
            }
        })
        .collect()
}

/// Build a single document dictionary
fn build_doc_dict(doc_id: ContextId, terms: Vec<String>) -> DynamicDawg<Vec<ContextId>> {
    let dict: DynamicDawg<Vec<ContextId>> = DynamicDawg::new();
    for term in terms {
        dict.insert_with_value(&term, vec![doc_id]);
    }
    dict
}

/// Merge function for context ID vectors
fn merge_contexts(left: &Vec<ContextId>, right: &Vec<ContextId>) -> Vec<ContextId> {
    let total_len = left.len() + right.len();

    if total_len > 50 {
        let mut set: FxHashSet<_> = left.iter().copied().collect();
        set.extend(right.iter().copied());
        let mut merged: Vec<_> = set.into_iter().collect();
        merged.sort_unstable();
        merged
    } else {
        let mut merged = left.clone();
        merged.extend(right.clone());
        merged.sort_unstable();
        merged.dedup();
        merged
    }
}

/// Sequential fold merge strategy
fn merge_sequential(mut dicts: Vec<DynamicDawg<Vec<ContextId>>>) -> DynamicDawg<Vec<ContextId>> {
    if dicts.is_empty() {
        return DynamicDawg::new();
    }

    let merged = dicts.remove(0);
    for dict in dicts {
        merged.union_with(&dict, merge_contexts);
    }
    merged
}

/// Binary tree reduction strategy
fn merge_binary_tree(mut dicts: Vec<DynamicDawg<Vec<ContextId>>>) -> DynamicDawg<Vec<ContextId>> {
    if dicts.is_empty() {
        return DynamicDawg::new();
    }

    while dicts.len() > 1 {
        dicts = dicts
            .par_chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    let merged = chunk[0].clone();
                    merged.union_with(&chunk[1], merge_contexts);
                    merged
                } else {
                    chunk[0].clone()
                }
            })
            .collect();
    }

    dicts.into_iter().next().unwrap()
}

/// Benchmark: Parallel document dictionary construction
fn bench_parallel_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_construction");
    group.plot_config(PlotConfiguration::default());

    for num_docs in [10, 50, 100, 200].iter() {
        let terms_per_doc = 1000;
        group.throughput(Throughput::Elements(*num_docs as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}docs_{}terms", num_docs, terms_per_doc)),
            num_docs,
            |b, &num_docs| {
                b.iter(|| {
                    let dicts: Vec<_> = (0..num_docs as u32)
                        .into_par_iter()
                        .map(|doc_id| {
                            let terms = generate_terms(doc_id, terms_per_doc);
                            build_doc_dict(doc_id, terms)
                        })
                        .collect();
                    black_box(dicts);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Sequential vs Binary Tree merge strategy comparison
fn bench_merge_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_strategy");
    group.plot_config(PlotConfiguration::default());

    for num_docs in [10, 25, 50, 100].iter() {
        let terms_per_doc = 1000;

        // Pre-build dictionaries for fair comparison
        let dicts: Vec<_> = (0..(*num_docs as u32))
            .map(|doc_id| {
                let terms = generate_terms(doc_id, terms_per_doc);
                build_doc_dict(doc_id, terms)
            })
            .collect();

        let dicts_clone = dicts.clone();

        group.bench_with_input(
            BenchmarkId::new("sequential", format!("{}docs", num_docs)),
            num_docs,
            |b, _| {
                b.iter(|| {
                    let result = merge_sequential(dicts.clone());
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_tree", format!("{}docs", num_docs)),
            num_docs,
            |b, _| {
                b.iter(|| {
                    let result = merge_binary_tree(dicts_clone.clone());
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Scaling with document count
fn bench_document_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_scaling");
    group.plot_config(PlotConfiguration::default());
    group.sample_size(20); // Reduce samples for large workloads

    for num_docs in [10, 20, 50, 100, 200].iter() {
        let terms_per_doc = 500;

        group.throughput(Throughput::Elements(*num_docs as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_docs),
            num_docs,
            |b, &num_docs| {
                b.iter(|| {
                    let dicts: Vec<_> = (0..num_docs as u32)
                        .into_par_iter()
                        .map(|doc_id| {
                            let terms = generate_terms(doc_id, terms_per_doc);
                            build_doc_dict(doc_id, terms)
                        })
                        .collect();

                    let result = merge_binary_tree(dicts);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Scaling with terms per document
fn bench_terms_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("terms_scaling");
    group.plot_config(PlotConfiguration::default());
    group.sample_size(20);

    let num_docs = 50;

    for terms_per_doc in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements((num_docs * terms_per_doc) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(terms_per_doc),
            terms_per_doc,
            |b, &terms_per_doc| {
                b.iter(|| {
                    let dicts: Vec<_> = (0..num_docs as u32)
                        .into_par_iter()
                        .map(|doc_id| {
                            let terms = generate_terms(doc_id, terms_per_doc);
                            build_doc_dict(doc_id, terms)
                        })
                        .collect();

                    let result = merge_binary_tree(dicts);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: End-to-end workspace indexing
fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    group.plot_config(PlotConfiguration::default());
    group.sample_size(10);

    for (num_docs, terms_per_doc) in [(50, 500), (100, 1000), (200, 500)].iter() {
        group.throughput(Throughput::Elements((num_docs * terms_per_doc) as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}docs_{}terms", num_docs, terms_per_doc)),
            &(num_docs, terms_per_doc),
            |b, (num_docs, terms_per_doc)| {
                b.iter(|| {
                    // Full pipeline: generate -> build -> merge
                    let dicts: Vec<_> = (0..**num_docs as u32)
                        .into_par_iter()
                        .map(|doc_id| {
                            let terms = generate_terms(doc_id, **terms_per_doc);
                            build_doc_dict(doc_id, terms)
                        })
                        .collect();

                    let result = merge_binary_tree(dicts);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory overhead comparison
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");
    group.plot_config(PlotConfiguration::default());

    for num_docs in [10, 50, 100].iter() {
        let terms_per_doc = 1000;

        group.bench_with_input(
            BenchmarkId::from_parameter(num_docs),
            num_docs,
            |b, &num_docs| {
                b.iter(|| {
                    let dicts: Vec<_> = (0..num_docs as u32)
                        .map(|doc_id| {
                            let terms = generate_terms(doc_id, terms_per_doc);
                            build_doc_dict(doc_id, terms)
                        })
                        .collect();

                    // Measure peak memory during merge
                    let result = merge_binary_tree(dicts);

                    // Access to ensure not optimized away
                    let _ = result.len();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Merge round analysis
fn bench_merge_rounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_rounds");
    group.plot_config(PlotConfiguration::default());

    // Test perfect powers of 2 vs non-powers
    for num_docs in [8, 10, 16, 20, 32, 50, 64, 100].iter() {
        let terms_per_doc = 500;

        let dicts: Vec<_> = (0..(*num_docs as u32))
            .map(|doc_id| {
                let terms = generate_terms(doc_id, terms_per_doc);
                build_doc_dict(doc_id, terms)
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(num_docs), num_docs, |b, _| {
            b.iter(|| {
                let result = merge_binary_tree(dicts.clone());
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark: Context vector deduplication strategies
fn bench_deduplication_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("deduplication_strategy");

    // Create test vectors of different sizes
    for size in [10, 50, 100, 500].iter() {
        let left: Vec<u32> = (0..*size).collect();
        let right: Vec<u32> = (*size / 2..*size + *size / 2).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = merge_contexts(&left, &right);
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    workspace_benches,
    bench_parallel_construction,
    bench_merge_strategy,
    bench_document_scaling,
    bench_terms_scaling,
    bench_end_to_end,
    bench_memory_overhead,
    bench_merge_rounds,
    bench_deduplication_strategy,
);

criterion_main!(workspace_benches);
