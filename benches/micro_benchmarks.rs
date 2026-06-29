//! Micro-benchmarks to investigate Phase 1 regressions
//!
//! These benchmarks isolate specific optimizations to understand
//! which changes caused regressions and why.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use liblevenshtein::prelude::*;
use smallvec::SmallVec;

/// Benchmark SmallVec vs Vec for position storage
fn bench_smallvec_overhead(c: &mut Criterion) {
    use liblevenshtein::transducer::Position;

    let mut group = c.benchmark_group("smallvec_vs_vec");

    // Test with different sizes
    for size in [2, 4, 6, 8, 10].iter() {
        // SmallVec with stack size 4
        group.bench_with_input(BenchmarkId::new("smallvec_4", size), size, |b, &n| {
            b.iter(|| {
                let mut v: SmallVec<[Position; 4]> = SmallVec::new();
                for i in 0..n {
                    v.push(Position::new(i, 0));
                }
                black_box(v);
            });
        });

        // Plain Vec
        group.bench_with_input(BenchmarkId::new("vec", size), size, |b, &n| {
            b.iter(|| {
                let mut v: Vec<Position> = Vec::new();
                for i in 0..n {
                    v.push(Position::new(i, 0));
                }
                black_box(v);
            });
        });

        // Vec with capacity
        group.bench_with_input(
            BenchmarkId::new("vec_with_capacity", size),
            size,
            |b, &n| {
                b.iter(|| {
                    let mut v: Vec<Position> = Vec::with_capacity(n);
                    for i in 0..n {
                        v.push(Position::new(i, 0));
                    }
                    black_box(v);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark characteristic vector allocation strategies
fn bench_characteristic_vector_allocation(c: &mut Criterion) {
    let query = b"testing";
    let mut group = c.benchmark_group("characteristic_vector");

    // Stack array (current implementation)
    group.bench_function("stack_array", |b| {
        b.iter(|| {
            let mut buffer = [false; 8];
            for i in 0..5 {
                for j in 0..8 {
                    buffer[j] = i + j < query.len() && query[i + j] == b't';
                }
                black_box(&buffer[..5]);
            }
        });
    });

    // Vec allocation (baseline)
    group.bench_function("vec_alloc", |b| {
        b.iter(|| {
            for i in 0..5 {
                let mut vec = Vec::with_capacity(8);
                for j in 0..8 {
                    vec.push(i + j < query.len() && query[i + j] == b't');
                }
                black_box(&vec[..5]);
            }
        });
    });

    group.finish();
}

/// Benchmark epsilon closure implementations
fn bench_epsilon_closure(c: &mut Criterion) {
    use liblevenshtein::transducer::{Position, State};

    let mut group = c.benchmark_group("epsilon_closure");

    // Create a state with various sizes
    for state_size in [1, 3, 5, 8].iter() {
        let mut state = State::new();
        let query_length = 20;
        for i in 0..*state_size {
            state.insert(Position::new(i, 0), Algorithm::Standard, query_length);
        }

        group.bench_with_input(
            BenchmarkId::new("current_impl", state_size),
            &state,
            |b, s| {
                b.iter(|| {
                    // This uses the current epsilon_closure implementation
                    let query_length = 20;
                    let max_distance = 3;

                    // Simulate epsilon closure work
                    let mut result = s.clone();
                    let mut to_process: SmallVec<[Position; 8]> = SmallVec::new();

                    for pos in result.positions() {
                        to_process.push(*pos);
                    }

                    let mut processed = 0;
                    while processed < to_process.len() {
                        let position = &to_process[processed];
                        processed += 1;

                        if position.num_errors < max_distance && position.term_index < query_length
                        {
                            let deleted =
                                Position::new(position.term_index + 1, position.num_errors + 1);
                            if !to_process.contains(&deleted) {
                                result.insert(deleted, Algorithm::Standard, query_length);
                                to_process.push(deleted);
                            }
                        }
                    }

                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark state operations with different sizes
fn bench_state_operations(c: &mut Criterion) {
    use liblevenshtein::transducer::{Position, State};

    let mut group = c.benchmark_group("state_operations");

    for size in [1, 3, 5, 10, 20].iter() {
        // Benchmark insert operation
        group.bench_with_input(BenchmarkId::new("insert", size), size, |b, &n| {
            b.iter(|| {
                let mut state = State::new();
                let query_length = 20;
                for i in 0..n {
                    state.insert(Position::new(i, 0), Algorithm::Standard, query_length);
                }
                black_box(state);
            });
        });

        // Benchmark contains check (for subsumption)
        group.bench_with_input(BenchmarkId::new("contains", size), size, |b, &n| {
            let mut state = State::new();
            let query_length = 20;
            for i in 0..n {
                state.insert(Position::new(i, 0), Algorithm::Standard, query_length);
            }

            b.iter(|| {
                for i in 0..n {
                    let pos = Position::new(i, 0);
                    black_box(state.positions().contains(&pos));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark transition operations
fn bench_transition_overhead(c: &mut Criterion) {
    use liblevenshtein::transducer::transition::{initial_state, transition_state};
    use liblevenshtein::transducer::Unrestricted;

    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(vec!["test", "testing", "tested"]);
    let _root = dict.root();

    let mut group = c.benchmark_group("transition_overhead");

    for distance in [1, 2, 3].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(distance),
            distance,
            |b, &max_dist| {
                let query: &[u8] = b"test";

                b.iter(|| {
                    let state = initial_state(query.len(), max_dist, Algorithm::Standard);

                    // Transition through 't'
                    let state = transition_state(
                        &state,
                        Unrestricted,
                        b't',
                        query,
                        max_dist,
                        Algorithm::Standard,
                        false,
                    );

                    black_box(state);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark inline vs non-inline impact
fn bench_inline_impact(c: &mut Criterion) {
    use liblevenshtein::transducer::Position;

    let mut group = c.benchmark_group("inline_impact");

    // Test Position::new calls (inlined)
    group.bench_function("position_new_inlined", |b| {
        b.iter(|| {
            for i in 0..100 {
                black_box(Position::new(i, 0));
            }
        });
    });

    // Test function call overhead
    #[inline(never)]
    fn position_new_not_inlined(term_index: usize, num_errors: usize) -> Position {
        Position::new(term_index, num_errors)
    }

    group.bench_function("position_new_not_inlined", |b| {
        b.iter(|| {
            for i in 0..100 {
                black_box(position_new_not_inlined(i, 0));
            }
        });
    });

    group.finish();
}

/// Benchmark complete query with profiling
fn bench_query_with_instrumentation(c: &mut Criterion) {
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms((0..1000).map(|i| format!("word{:04}", i)));
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut group = c.benchmark_group("instrumented_query");

    for distance in [1, 2, 3].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(distance), distance, |b, &d| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query(black_box("word0500"), black_box(d))
                    .collect();
                black_box(results);
            });
        });
    }

    group.finish();
}

criterion_group!(
    micro_benches,
    bench_smallvec_overhead,
    bench_characteristic_vector_allocation,
    bench_epsilon_closure,
    bench_state_operations,
    bench_transition_overhead,
    bench_inline_impact,
    bench_query_with_instrumentation,
);
criterion_main!(micro_benches);
