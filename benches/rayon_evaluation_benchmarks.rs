//! Benchmarks to evaluate whether Rayon integration is justified.
//!
//! This benchmark suite measures the performance impact of parallelizing
//! batch operations using Rayon, comparing against sequential implementations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::MappedDictionary;
use liblevenshtein::cache::eviction::Lru;
use std::hint::black_box;

// Sample dictionary data
fn create_test_dict(size: usize) -> PathMapDictionary<i32> {
    let terms: Vec<_> = (0..size)
        .map(|i| (format!("term_{:06}", i), i as i32))
        .collect();
    PathMapDictionary::from_terms_with_values(terms)
}

// Populate LRU metadata
fn populate_lru_metadata(lru: &Lru<PathMapDictionary<i32>>, size: usize) {
    for i in 0..size {
        let term = format!("term_{:06}", i);
        lru.get_value(&term);
    }
}

// Sequential batch recency query
fn sequential_batch_recency(
    lru: &Lru<PathMapDictionary<i32>>,
    terms: &[String],
) -> Vec<(String, std::time::Duration)> {
    terms
        .iter()
        .filter_map(|term| lru.recency(term).map(|recency| (term.clone(), recency)))
        .collect()
}

// Parallel batch recency query (simulated - we'll implement this if justified)
#[cfg(feature = "rayon")]
fn parallel_batch_recency(
    lru: &Lru<PathMapDictionary<i32>>,
    terms: &[String],
) -> Vec<(String, std::time::Duration)> {
    use rayon::prelude::*;
    terms
        .par_iter()
        .filter_map(|term| lru.recency(term).map(|recency| (term.clone(), recency)))
        .collect()
}

// Sequential find N least recently used
fn sequential_find_n_lru(
    lru: &Lru<PathMapDictionary<i32>>,
    terms: &[&str],
    n: usize,
) -> Vec<String> {
    let mut candidates: Vec<_> = terms
        .iter()
        .filter_map(|&term| lru.recency(term).map(|recency| (term, recency)))
        .collect();

    candidates.sort_by_key(|(_, recency)| std::cmp::Reverse(*recency));
    candidates
        .into_iter()
        .take(n)
        .map(|(term, _)| term.to_string())
        .collect()
}

// Parallel find N least recently used (simulated)
#[cfg(feature = "rayon")]
fn parallel_find_n_lru(lru: &Lru<PathMapDictionary<i32>>, terms: &[&str], n: usize) -> Vec<String> {
    use rayon::prelude::*;
    let mut candidates: Vec<_> = terms
        .par_iter()
        .filter_map(|&term| lru.recency(term).map(|recency| (term, recency)))
        .collect();

    candidates.sort_by_key(|(_, recency)| std::cmp::Reverse(*recency));
    candidates
        .into_iter()
        .take(n)
        .map(|(term, _)| term.to_string())
        .collect()
}

// Benchmark: Sequential batch recency query
fn bench_sequential_batch_recency(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_recency/sequential");

    for size in [100, 1000, 10000] {
        let dict = create_test_dict(size);
        let lru = Lru::new(dict);
        populate_lru_metadata(&lru, size);

        let terms: Vec<String> = (0..size).map(|i| format!("term_{:06}", i)).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &terms, |b, terms| {
            b.iter(|| {
                black_box(sequential_batch_recency(&lru, terms));
            });
        });
    }

    group.finish();
}

// Benchmark: Parallel batch recency query
#[cfg(feature = "rayon")]
fn bench_parallel_batch_recency(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_recency/parallel");

    for size in [100, 1000, 10000] {
        let dict = create_test_dict(size);
        let lru = Lru::new(dict);
        populate_lru_metadata(&lru, size);

        let terms: Vec<String> = (0..size).map(|i| format!("term_{:06}", i)).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &terms, |b, terms| {
            b.iter(|| {
                black_box(parallel_batch_recency(&lru, terms));
            });
        });
    }

    group.finish();
}

// Benchmark: Sequential find N LRU
fn bench_sequential_find_n_lru(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_n_lru/sequential");

    for size in [100, 1000, 10000] {
        let dict = create_test_dict(size);
        let lru = Lru::new(dict);
        populate_lru_metadata(&lru, size);

        let terms: Vec<String> = (0..size).map(|i| format!("term_{:06}", i)).collect();
        let term_refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();

        let n = size / 10; // Find 10% LRU entries

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_find_{}", size, n)),
            &term_refs,
            |b, terms| {
                b.iter(|| {
                    black_box(sequential_find_n_lru(&lru, terms, n));
                });
            },
        );
    }

    group.finish();
}

// Benchmark: Parallel find N LRU
#[cfg(feature = "rayon")]
fn bench_parallel_find_n_lru(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_n_lru/parallel");

    for size in [100, 1000, 10000] {
        let dict = create_test_dict(size);
        let lru = Lru::new(dict);
        populate_lru_metadata(&lru, size);

        let terms: Vec<String> = (0..size).map(|i| format!("term_{:06}", i)).collect();
        let term_refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();

        let n = size / 10; // Find 10% LRU entries

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_find_{}", size, n)),
            &term_refs,
            |b, terms| {
                b.iter(|| {
                    black_box(parallel_find_n_lru(&lru, terms, n));
                });
            },
        );
    }

    group.finish();
}

// Benchmark: Comparison at different dataset sizes
fn bench_comparison_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("size_comparison");

    for size in [10, 50, 100, 500, 1000, 5000, 10000] {
        let dict = create_test_dict(size);
        let lru = Lru::new(dict);
        populate_lru_metadata(&lru, size);

        let terms: Vec<String> = (0..size).map(|i| format!("term_{:06}", i)).collect();

        group.throughput(Throughput::Elements(size as u64));

        // Sequential version
        group.bench_with_input(BenchmarkId::new("sequential", size), &terms, |b, terms| {
            b.iter(|| {
                black_box(sequential_batch_recency(&lru, terms));
            });
        });

        // Parallel version (if available)
        #[cfg(feature = "rayon")]
        group.bench_with_input(BenchmarkId::new("parallel", size), &terms, |b, terms| {
            b.iter(|| {
                black_box(parallel_batch_recency(&lru, terms));
            });
        });
    }

    group.finish();
}

#[cfg(not(feature = "rayon"))]
criterion_group!(
    benches,
    bench_sequential_batch_recency,
    bench_sequential_find_n_lru,
    bench_comparison_by_size,
);

#[cfg(feature = "rayon")]
criterion_group!(
    benches,
    bench_sequential_batch_recency,
    bench_parallel_batch_recency,
    bench_sequential_find_n_lru,
    bench_parallel_find_n_lru,
    bench_comparison_by_size,
);

criterion_main!(benches);
