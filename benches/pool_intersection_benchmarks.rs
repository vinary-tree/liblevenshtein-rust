//! Benchmarks for StatePool operations.
//!
//! This benchmark suite measures the performance of:
//! - StatePool acquire/release operations
//! - Pool reuse patterns
//! - Pool pre-warming effectiveness
//!
//! PathNode and Intersection benchmarks are internal-only since those
//! types are in private modules. They are tested via integration tests.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use liblevenshtein::transducer::{Algorithm, Position, StatePool};

// ============================================================================
// StatePool Benchmarks
// ============================================================================

fn bench_pool_acquire_release(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_acquire_release");

    // Benchmark pool acquire (should hit pre-warmed states)
    group.bench_function("acquire_from_prewarm", |b| {
        let mut pool = StatePool::new();
        b.iter(|| {
            let state = pool.acquire();
            black_box(state);
        });
    });

    // Benchmark pool acquire + release cycle
    group.bench_function("acquire_release_cycle", |b| {
        let mut pool = StatePool::new();
        b.iter(|| {
            let state = pool.acquire();
            pool.release(black_box(state));
        });
    });

    // Benchmark pool acquire after exhausting pre-warmed (forces allocation)
    group.bench_function("acquire_after_exhaustion", |b| {
        let mut pool = StatePool::new();
        // Exhaust pre-warmed states
        let states: Vec<_> = (0..4).map(|_| pool.acquire()).collect();
        b.iter(|| {
            let state = pool.acquire();
            black_box(state);
        });
        // Return states to prevent memory leak
        for state in states {
            pool.release(state);
        }
    });

    group.finish();
}

fn bench_pool_reuse_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_reuse_patterns");

    // Realistic pattern: multiple acquires and releases
    for num_states in [2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("concurrent_{}", num_states)),
            num_states,
            |b, &n| {
                let mut pool = StatePool::new();
                b.iter(|| {
                    let states: Vec<_> = (0..n).map(|_| pool.acquire()).collect();
                    for state in states {
                        pool.release(black_box(state));
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Realistic Query Simulation
// ============================================================================

fn bench_realistic_query_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_query");

    // Simulate a realistic query pattern:
    // - Acquire states from pool
    // - Populate states with positions
    // - Release states
    group.bench_function("full_query_cycle", |b| {
        let mut pool = StatePool::new();
        b.iter(|| {
            // Acquire states
            let mut states = Vec::new();
            for _ in 0..4 {
                states.push(pool.acquire());
            }

            // Simulate some work with states
            let query_length = 10;
            for state in &mut states {
                state.insert(Position::new(0, 0), Algorithm::Standard, query_length);
            }

            // Release states
            for state in states {
                pool.release(black_box(state));
            }
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Groups
// ============================================================================

criterion_group!(
    pool_benches,
    bench_pool_acquire_release,
    bench_pool_reuse_patterns,
    bench_realistic_query_simulation,
);

criterion_main!(pool_benches);
