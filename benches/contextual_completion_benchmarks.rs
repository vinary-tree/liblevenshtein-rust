//! Benchmarks for the Contextual Completion Engine
//!
//! These benchmarks measure the performance of key operations:
//! - Character insertion (draft building)
//! - Checkpoint creation and undo
//! - Query with finalized terms (automaton-based)
//! - Query with drafts (naive Levenshtein)
//! - Query fusion (both drafts and finalized)

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::contextual::DynamicContextualCompletionEngine as ContextualCompletionEngine;
use liblevenshtein::transducer::Algorithm;

/// Benchmark character-level insertion into draft buffers
fn bench_insert_char(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_char");

    for size in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let engine = ContextualCompletionEngine::new();
            let ctx = engine.create_root_context(0);

            b.iter(|| {
                // Insert characters
                for i in 0..size {
                    let ch = char::from_u32('a' as u32 + (i % 26)).unwrap();
                    engine.insert_char(ctx, ch).unwrap();
                }

                // Clear for next iteration
                engine.clear_draft(ctx).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark checkpoint creation
fn bench_checkpoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint");

    group.bench_function("checkpoint_10x", |b| {
        let engine = ContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        // Build a draft
        engine.insert_str(ctx, "hello_world").unwrap();

        b.iter(|| {
            // Create 10 checkpoints
            for _ in 0..10 {
                engine.checkpoint(ctx).unwrap();
            }

            // Clear checkpoints for next iteration
            engine.clear_checkpoints(ctx).unwrap();
        });
    });

    group.finish();
}

/// Benchmark undo operations
fn bench_undo(c: &mut Criterion) {
    let mut group = c.benchmark_group("undo");

    group.bench_function("undo_10x", |b| {
        let engine = ContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        b.iter(|| {
            // Build draft with checkpoints
            for ch in "hello_world".chars() {
                engine.insert_char(ctx, ch).unwrap();
                engine.checkpoint(ctx).unwrap();
            }

            // Undo 10 times
            for _ in 0..10 {
                let _ = engine.undo(ctx);
            }

            // Clear for next iteration
            engine.clear_draft(ctx).unwrap();
            engine.clear_checkpoints(ctx).unwrap();
        });
    });

    group.finish();
}

/// Benchmark querying finalized terms (transducer-based)
fn bench_complete_finalized(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_finalized");

    // Create engine with various dictionary sizes
    for dict_size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*dict_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(dict_size),
            dict_size,
            |b, &dict_size| {
                let engine = ContextualCompletionEngine::with_algorithm(Algorithm::Standard);
                let ctx = engine.create_root_context(0);

                // Populate dictionary with terms
                for i in 0..dict_size {
                    let term = format!("identifier_{}", i);
                    engine.finalize_direct(ctx, &term).unwrap();
                }

                // Add some terms that will match our query
                engine.finalize_direct(ctx, "helper").unwrap();
                engine.finalize_direct(ctx, "helper_fn").unwrap();
                engine.finalize_direct(ctx, "global_helper").unwrap();

                b.iter(|| {
                    let results = engine.complete_finalized(ctx, black_box("help"), black_box(2));
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark querying drafts (naive Levenshtein)
fn bench_complete_drafts(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_drafts");

    // Create hierarchical contexts with drafts
    for depth in [1, 3, 5].iter() {
        group.throughput(Throughput::Elements(*depth as u64));
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            let engine = ContextualCompletionEngine::new();

            // Create hierarchical contexts
            let root = engine.create_root_context(0);
            let mut current = root;
            for i in 1..depth {
                current = engine.create_child_context(i as u32, current).unwrap();
            }

            // Add drafts to each context
            for i in 0..depth {
                engine
                    .insert_str(i as u32, &format!("draft_{}", i))
                    .unwrap();
            }

            b.iter(|| {
                let results = engine.complete_drafts(current, black_box("dra"), black_box(2));
                black_box(results);
            });

            // Cleanup
            for i in 0..depth {
                engine.clear_draft(i as u32).unwrap();
            }
        });
    }

    group.finish();
}

/// Benchmark full query fusion (drafts + finalized)
fn bench_complete(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_fusion");

    group.bench_function("fusion_100_terms", |b| {
        let engine = ContextualCompletionEngine::new();
        let global = engine.create_root_context(0);
        let func = engine.create_child_context(1, global).unwrap();

        // Add finalized terms
        for i in 0..100 {
            let term = format!("identifier_{}", i);
            engine.finalize_direct(global, &term).unwrap();
        }

        // Add some matching terms
        engine.finalize_direct(global, "helper").unwrap();
        engine.finalize_direct(global, "helper_fn").unwrap();

        // Add drafts
        engine.insert_str(func, "helping_hand").unwrap();

        b.iter(|| {
            let results = engine.complete(func, black_box("help"), black_box(2));
            black_box(results);
        });
    });

    group.finish();
}

/// Benchmark hierarchical visibility resolution
fn bench_visibility(c: &mut Criterion) {
    let mut group = c.benchmark_group("visibility");

    for depth in [3, 5, 10].iter() {
        group.throughput(Throughput::Elements(*depth as u64));
        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, &depth| {
            let engine = ContextualCompletionEngine::new();

            // Create deep hierarchy
            let root = engine.create_root_context(0);
            let mut current = root;
            for i in 1..depth {
                current = engine.create_child_context(i as u32, current).unwrap();
            }

            b.iter(|| {
                let visible = engine.get_visible_contexts(black_box(current));
                black_box(visible);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_char,
    bench_checkpoint,
    bench_undo,
    bench_complete_finalized,
    bench_complete_drafts,
    bench_complete,
    bench_visibility
);

criterion_main!(benches);
