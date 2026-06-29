//! Benchmarks for cache eviction wrappers.
//!
//! These benchmarks measure the overhead of eviction wrappers compared to
//! unwrapped dictionaries, and compare performance across wrapper types.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::pathmap::PathMapDictionary;
use libdictenstein::MappedDictionary;
use liblevenshtein::cache::eviction::{
    Age, CostAware, Lfu, Lru, LruOptimized, MemoryPressure, Noop, Ttl,
};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// Sample dictionary data
fn create_test_dict() -> PathMapDictionary<i32> {
    let terms: Vec<_> = (0..1000).map(|i| (format!("term_{:04}", i), i)).collect();
    PathMapDictionary::from_terms_with_values(terms)
}

// Benchmark: Baseline (no wrapper)
fn bench_baseline_get_value(c: &mut Criterion) {
    let dict = create_test_dict();

    c.bench_function("baseline/get_value", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(dict.get_value(&term));
            }
        });
    });
}

// Benchmark: Noop wrapper (should be zero-cost)
fn bench_noop_get_value(c: &mut Criterion) {
    let dict = create_test_dict();
    let noop = Noop::new(dict);

    c.bench_function("noop/get_value", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(noop.get_value(&term));
            }
        });
    });
}

// Benchmark: LRU wrapper
fn bench_lru_get_value(c: &mut Criterion) {
    let dict = create_test_dict();
    let lru = Lru::new(dict);

    c.bench_function("lru/get_value", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(lru.get_value(&term));
            }
        });
    });
}

// Benchmark: LFU wrapper
fn bench_lfu_get_value(c: &mut Criterion) {
    let dict = create_test_dict();
    let lfu = Lfu::new(dict);

    c.bench_function("lfu/get_value", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(lfu.get_value(&term));
            }
        });
    });
}

// Benchmark: TTL wrapper
fn bench_ttl_get_value(c: &mut Criterion) {
    let dict = create_test_dict();
    let ttl = Ttl::new(dict, Duration::from_secs(60));

    c.bench_function("ttl/get_value", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(ttl.get_value(&term));
            }
        });
    });
}

// Benchmark: Age wrapper
fn bench_age_get_value(c: &mut Criterion) {
    let dict = create_test_dict();
    let age = Age::new(dict);

    c.bench_function("age/get_value", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(age.get_value(&term));
            }
        });
    });
}

// Benchmark: CostAware wrapper
fn bench_cost_aware_get_value(c: &mut Criterion) {
    let dict = create_test_dict();
    let cost = CostAware::new(dict);

    c.bench_function("cost_aware/get_value", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(cost.get_value(&term));
            }
        });
    });
}

// Benchmark: MemoryPressure wrapper
fn bench_memory_pressure_get_value(c: &mut Criterion) {
    let dict = create_test_dict();
    let memory = MemoryPressure::new(dict);

    c.bench_function("memory_pressure/get_value", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(memory.get_value(&term));
            }
        });
    });
}

// Benchmark: Wrapper composition (Lru<Ttl<Dict>>)
fn bench_composed_wrappers(c: &mut Criterion) {
    let dict = create_test_dict();
    let ttl = Ttl::new(dict, Duration::from_secs(60));
    let lru = Lru::new(ttl);

    c.bench_function("composed/lru_ttl/get_value", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(lru.get_value(&term));
            }
        });
    });
}

// Benchmark: Multi-threaded access (LRU)
fn bench_lru_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru/concurrent");

    for thread_count in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements((thread_count * 100) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_threads", thread_count)),
            &thread_count,
            |b, &threads| {
                let dict = create_test_dict();
                let lru = Arc::new(Lru::new(dict));

                b.iter(|| {
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let lru_clone = Arc::clone(&lru);
                            thread::spawn(move || {
                                for i in 0..100 {
                                    let term = format!("term_{:04}", i);
                                    black_box(lru_clone.get_value(&term));
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// Benchmark: Eviction operations
fn bench_lru_eviction(c: &mut Criterion) {
    let dict = create_test_dict();
    let lru = Lru::new(dict);

    // Populate metadata
    for i in 0..1000 {
        let term = format!("term_{:04}", i);
        lru.get_value(&term);
    }

    c.bench_function("lru/find_lru", |b| {
        let terms: Vec<_> = (0..100).map(|i| format!("term_{:04}", i)).collect();
        let term_refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();

        b.iter(|| {
            black_box(lru.find_lru(&term_refs));
        });
    });
}

// Benchmark: Metadata operations
fn bench_lru_metadata_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru/metadata");

    for size in [100, 1000, 10000] {
        let dict: PathMapDictionary<i32> = PathMapDictionary::from_terms_with_values(
            (0..size).map(|i| (format!("term_{:04}", i), i)),
        );
        let lru = Lru::new(dict);

        // Populate metadata
        for i in 0..size {
            let term = format!("term_{:04}", i);
            lru.get_value(&term);
        }

        group.bench_with_input(BenchmarkId::new("clear", size), &lru, |b, lru| {
            b.iter(|| {
                lru.clear_metadata();
            });
        });
    }

    group.finish();
}

// Benchmark: String allocation overhead
fn bench_string_allocation_overhead(c: &mut Criterion) {
    let dict = create_test_dict();
    let lru = Lru::new(dict);

    let mut group = c.benchmark_group("string_allocation");

    // Pre-allocate strings (best case)
    let terms: Vec<String> = (0..100).map(|i| format!("term_{:04}", i)).collect();

    group.bench_function("preallocated", |b| {
        b.iter(|| {
            for term in &terms {
                black_box(lru.get_value(term));
            }
        });
    });

    // Allocate on each access (worst case)
    group.bench_function("per_access", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(lru.get_value(&term));
            }
        });
    });

    group.finish();
}

// Benchmark: LruOptimized vs baseline Lru
fn bench_lru_optimized_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_comparison");

    let dict = create_test_dict();
    let lru_baseline = Lru::new(dict.clone());
    let lru_optimized = LruOptimized::new(dict);

    group.bench_function("baseline_lru", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(lru_baseline.get_value(&term));
            }
        });
    });

    group.bench_function("optimized_lru", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{:04}", i);
                black_box(lru_optimized.get_value(&term));
            }
        });
    });

    group.finish();
}

// Benchmark: LruOptimized concurrent access
fn bench_lru_optimized_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_optimized/concurrent");

    for thread_count in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements((thread_count * 100) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_threads", thread_count)),
            &thread_count,
            |b, &threads| {
                let dict = create_test_dict();
                let lru = Arc::new(LruOptimized::new(dict));

                b.iter(|| {
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let lru_clone = Arc::clone(&lru);
                            thread::spawn(move || {
                                for i in 0..100 {
                                    let term = format!("term_{:04}", i);
                                    black_box(lru_clone.get_value(&term));
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_baseline_get_value,
    bench_noop_get_value,
    bench_lru_get_value,
    bench_lfu_get_value,
    bench_ttl_get_value,
    bench_age_get_value,
    bench_cost_aware_get_value,
    bench_memory_pressure_get_value,
    bench_composed_wrappers,
    bench_lru_concurrent_access,
    bench_lru_eviction,
    bench_lru_metadata_operations,
    bench_string_allocation_overhead,
    bench_lru_optimized_comparison,
    bench_lru_optimized_concurrent,
);

criterion_main!(benches);
