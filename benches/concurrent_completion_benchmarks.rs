//! Concurrency benchmarks for the Contextual Completion Engine
//!
//! These benchmarks measure the performance under concurrent access patterns:
//! - Multiple threads querying simultaneously (read-heavy workload)
//! - Multiple threads inserting to different contexts (write-heavy workload)
//! - Mixed read-write workload
//! - Lock contention analysis

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::contextual::DynamicContextualCompletionEngine as ContextualCompletionEngine;
use std::sync::Arc;
use std::thread;

/// Benchmark concurrent queries (read-heavy workload)
fn bench_concurrent_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_queries");

    for num_threads in [2, 4, 8].iter() {
        group.throughput(Throughput::Elements(*num_threads as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, &num_threads| {
                let engine = Arc::new(ContextualCompletionEngine::new());
                let ctx = engine.create_root_context(0);

                // Populate dictionary
                for i in 0..1000 {
                    let term = format!("identifier_{}", i);
                    engine.finalize_direct(ctx, &term).unwrap();
                }
                engine.finalize_direct(ctx, "helper").unwrap();
                engine.finalize_direct(ctx, "helper_fn").unwrap();

                b.iter(|| {
                    let mut handles = vec![];

                    for _ in 0..num_threads {
                        let engine_clone = Arc::clone(&engine);
                        let handle = thread::spawn(move || {
                            for _ in 0..10 {
                                let results = engine_clone.complete(ctx, "help", 2);
                                black_box(results);
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent insertions to different contexts (write-heavy workload)
fn bench_concurrent_insertions(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_insertions");

    for num_threads in [2, 4, 8].iter() {
        group.throughput(Throughput::Elements(*num_threads as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            num_threads,
            |b, &num_threads| {
                let engine = Arc::new(ContextualCompletionEngine::new());

                // Create separate contexts for each thread
                let contexts: Vec<_> = (0..num_threads)
                    .map(|i| engine.create_root_context(i as u32))
                    .collect();

                b.iter(|| {
                    let mut handles = vec![];

                    for (i, &ctx) in contexts.iter().enumerate() {
                        let engine_clone = Arc::clone(&engine);
                        let handle = thread::spawn(move || {
                            // Insert characters
                            for ch in format!("thread_{}_data", i).chars() {
                                engine_clone.insert_char(ctx, ch).unwrap();
                            }
                            // Clear for next iteration
                            engine_clone.clear_draft(ctx).unwrap();
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mixed read-write workload
fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");

    group.bench_function("4_threads_50_50_mix", |b| {
        let engine = Arc::new(ContextualCompletionEngine::new());
        let global = engine.create_root_context(0);

        // Populate dictionary
        for i in 0..500 {
            let term = format!("identifier_{}", i);
            engine.finalize_direct(global, &term).unwrap();
        }

        // Create contexts for writers
        let writer_contexts: Vec<_> = (1..=4)
            .map(|i| engine.create_child_context(i, global).unwrap())
            .collect();

        b.iter(|| {
            let mut handles = vec![];

            // 2 reader threads
            for _ in 0..2 {
                let engine_clone = Arc::clone(&engine);
                let handle = thread::spawn(move || {
                    for _ in 0..5 {
                        let results = engine_clone.complete(global, "ident", 2);
                        black_box(results);
                    }
                });
                handles.push(handle);
            }

            // 2 writer threads
            for (i, &ctx) in writer_contexts.iter().take(2).enumerate() {
                let engine_clone = Arc::clone(&engine);
                let handle = thread::spawn(move || {
                    for ch in format!("draft_{}", i).chars() {
                        engine_clone.insert_char(ctx, ch).unwrap();
                    }
                    engine_clone.clear_draft(ctx).unwrap();
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark concurrent finalization
fn bench_concurrent_finalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_finalization");

    group.bench_function("8_threads_finalize", |b| {
        let engine = Arc::new(ContextualCompletionEngine::new());

        b.iter(|| {
            let mut handles = vec![];

            for i in 0..8 {
                let engine_clone = Arc::clone(&engine);
                let handle = thread::spawn(move || {
                    let ctx = engine_clone.create_root_context(i);
                    for j in 0..10 {
                        let term = format!("term_{}_{}", i, j);
                        engine_clone.finalize_direct(ctx, &term).unwrap();
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark lock contention with same context
fn bench_lock_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("lock_contention");

    group.bench_function("4_threads_same_context", |b| {
        let engine = Arc::new(ContextualCompletionEngine::new());
        let ctx = engine.create_root_context(0);

        // Pre-populate
        engine.finalize_direct(ctx, "test").unwrap();

        b.iter(|| {
            let mut handles = vec![];

            for i in 0..4 {
                let engine_clone = Arc::clone(&engine);
                let handle = thread::spawn(move || {
                    if i % 2 == 0 {
                        // Reader
                        let results = engine_clone.complete(ctx, "te", 1);
                        black_box(results);
                    } else {
                        // Writer
                        engine_clone.insert_char(ctx, 'x').unwrap();
                        engine_clone.delete_char(ctx);
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark hierarchical context queries under concurrency
fn bench_hierarchical_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical_concurrent");

    group.bench_function("8_threads_hierarchy", |b| {
        let engine = Arc::new(ContextualCompletionEngine::new());

        // Create hierarchy: root -> 4 children -> 4 grandchildren each
        let root = engine.create_root_context(0);
        let mut children = vec![];
        let mut grandchildren = vec![];

        for i in 1..=4 {
            let child = engine.create_child_context(i, root).unwrap();
            children.push(child);
            engine
                .finalize_direct(child, &format!("child_{}", i))
                .unwrap();

            for j in 1..=4 {
                let id = i * 10 + j;
                let grandchild = engine.create_child_context(id, child).unwrap();
                grandchildren.push(grandchild);
                engine
                    .finalize_direct(grandchild, &format!("grandchild_{}_{}", i, j))
                    .unwrap();
            }
        }

        // Add global terms
        for i in 0..50 {
            engine
                .finalize_direct(root, &format!("global_{}", i))
                .unwrap();
        }

        b.iter(|| {
            let mut handles = vec![];

            // Query from different levels simultaneously
            for &ctx in grandchildren.iter().take(8) {
                let engine_clone = Arc::clone(&engine);
                let handle = thread::spawn(move || {
                    let results = engine_clone.complete(ctx, "glo", 2);
                    black_box(results);
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_concurrent_queries,
    bench_concurrent_insertions,
    bench_mixed_workload,
    bench_concurrent_finalization,
    bench_lock_contention,
    bench_hierarchical_concurrent
);

criterion_main!(benches);
