use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::transducer::{Algorithm, Position, Unrestricted};
use smallvec::SmallVec;

// Re-export internal functions for benchmarking
use liblevenshtein::transducer::transition::{
    initial_state, transition_position, transition_state,
};

//==============================================================================
// MICRO-BENCHMARKS: Individual Function Performance
//==============================================================================

/// Benchmark characteristic_vector computation
fn bench_characteristic_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("characteristic_vector");

    let query = b"approximately";

    for &window_size in &[1, 2, 4, 8] {
        group.throughput(Throughput::Elements(window_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("window={}", window_size)),
            &window_size,
            |b, &window_size| {
                b.iter(|| {
                    let mut buffer = [false; 8];
                    // Simulate characteristic_vector (inlined for benchmarking)
                    let len = window_size.min(8);
                    for (i, item) in buffer.iter_mut().enumerate().take(len) {
                        let query_idx = i;
                        *item = query_idx < query.len() && query[query_idx] == black_box(b'a');
                    }
                    black_box(buffer[..len].to_vec())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark index_of_match (multi-character deletion optimization)
fn bench_index_of_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_of_match");

    // Test different match patterns
    let patterns = vec![
        ("immediate", vec![true, false, false, false]), // Match at start
        ("middle", vec![false, false, true, false]),    // Match in middle
        ("end", vec![false, false, false, true]),       // Match at end
        ("no_match", vec![false, false, false, false]), // No match
    ];

    for (name, cv) in patterns {
        group.bench_with_input(BenchmarkId::from_parameter(name), &cv, |b, cv| {
            b.iter(|| {
                // Inline index_of_match for benchmarking
                let start = 0;
                let limit = 4;
                let mut result = None;
                for j in 0..limit {
                    if cv.get(start + j).copied().unwrap_or(false) {
                        result = Some(j);
                        break;
                    }
                }
                black_box(result)
            });
        });
    }

    group.finish();
}

//==============================================================================
// ALGORITHM-SPECIFIC BENCHMARKS
//==============================================================================

/// Benchmark transition_standard
fn bench_transition_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("transition_standard");

    // Test different scenarios
    let scenarios = vec![
        ("match", vec![true, false, false]), // Immediate match (best case)
        ("no_match", vec![false, false, false]), // No match (worst case)
        ("delayed_match", vec![false, false, true]), // Match after deletions
    ];

    for &max_distance in &[1, 2, 3] {
        for (name, cv) in &scenarios {
            let position = Position::new(0, 0);
            let query_length = 10;

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}/d={}", name, max_distance)),
                &(&position, cv, query_length, max_distance),
                |b, (pos, cv, q_len, max_d)| {
                    b.iter(|| {
                        // Call actual transition function
                        let next = transition_position(
                            black_box(pos),
                            black_box(cv),
                            *q_len,
                            *max_d,
                            Algorithm::Standard,
                            false,
                        );
                        black_box(next)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark transition_transposition
fn bench_transition_transposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("transition_transposition");

    let scenarios = vec![
        ("match", vec![true, false, false]),
        ("transposition", vec![false, true, false]), // Potential transposition
        ("no_match", vec![false, false, false]),
    ];

    for &max_distance in &[1, 2, 3] {
        for (name, cv) in &scenarios {
            // Test both regular and special positions
            for is_special in &[false, true] {
                let position = if *is_special {
                    Position::new_special(0, 1)
                } else {
                    Position::new(0, 0)
                };
                let query_length = 10;

                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "{}/d={}/special={}",
                        name, max_distance, is_special
                    )),
                    &(&position, cv, query_length, max_distance),
                    |b, (pos, cv, q_len, max_d)| {
                        b.iter(|| {
                            let next = transition_position(
                                black_box(pos),
                                black_box(cv),
                                *q_len,
                                *max_d,
                                Algorithm::Transposition,
                                false,
                            );
                            black_box(next)
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark transition_merge_split
fn bench_transition_merge_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("transition_merge_split");

    let scenarios = vec![
        ("match", vec![true, false, false]),
        ("merge", vec![false, false, true]), // Potential merge
        ("no_match", vec![false, false, false]),
    ];

    for &max_distance in &[1, 2, 3] {
        for (name, cv) in &scenarios {
            for is_special in &[false, true] {
                let position = if *is_special {
                    Position::new_special(0, 1)
                } else {
                    Position::new(0, 0)
                };
                let query_length = 10;

                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "{}/d={}/special={}",
                        name, max_distance, is_special
                    )),
                    &(&position, cv, query_length, max_distance),
                    |b, (pos, cv, q_len, max_d)| {
                        b.iter(|| {
                            let next = transition_position(
                                black_box(pos),
                                black_box(cv),
                                *q_len,
                                *max_d,
                                Algorithm::MergeAndSplit,
                                false,
                            );
                            black_box(next)
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

//==============================================================================
// STATE TRANSITION BENCHMARKS
//==============================================================================

/// Benchmark full state transition
fn bench_transition_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("transition_state");

    let queries = vec![
        ("short", b"test".to_vec()),
        ("medium", b"approximate".to_vec()),
        ("long", b"approximately".to_vec()),
    ];

    for (q_name, query) in &queries {
        for &max_distance in &[1, 2, 3] {
            for algorithm in [
                Algorithm::Standard,
                Algorithm::Transposition,
                Algorithm::MergeAndSplit,
            ] {
                // Create initial state with varying complexities
                let state = initial_state(query.len(), max_distance, algorithm);

                group.throughput(Throughput::Elements(state.len() as u64));
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "{:?}/{}/d={}/positions={}",
                        algorithm,
                        q_name,
                        max_distance,
                        state.len()
                    )),
                    &(&state, query, max_distance, algorithm),
                    |b, (state, query, max_d, algo)| {
                        b.iter(|| {
                            // Transition on first character
                            let dict_char = query[0];
                            let next = transition_state(
                                black_box(state),
                                Unrestricted,
                                black_box(dict_char),
                                black_box(&query[..]),
                                *max_d,
                                *algo,
                                false,
                            );
                            black_box(next)
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark state transition with different state sizes
fn bench_transition_by_state_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("transition_by_state_size");

    let query = b"approximate";

    for &max_distance in &[1, 2, 3, 5] {
        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
        ] {
            let state = initial_state(query.len(), max_distance, algorithm);

            group.throughput(Throughput::Elements(state.len() as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{:?}/d={}/size={}",
                    algorithm,
                    max_distance,
                    state.len()
                )),
                &(state, query, max_distance, algorithm),
                |b, (state, query, max_d, algo)| {
                    b.iter(|| {
                        let next = transition_state(
                            black_box(state),
                            Unrestricted,
                            black_box(b'a'),
                            black_box(&query[..]),
                            *max_d,
                            *algo,
                            false,
                        );
                        black_box(next)
                    });
                },
            );
        }
    }

    group.finish();
}

//==============================================================================
// PREFIX MODE BENCHMARKS
//==============================================================================

/// Benchmark prefix mode transitions
fn bench_prefix_mode(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_mode");

    let query = b"test";

    for &prefix_mode in &[false, true] {
        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
        ] {
            let state = initial_state(query.len(), 2, algorithm);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{:?}/prefix={}", algorithm, prefix_mode)),
                &(state, query, algorithm, prefix_mode),
                |b, (state, query, algo, prefix)| {
                    b.iter(|| {
                        let next = transition_state(
                            black_box(state),
                            Unrestricted,
                            black_box(b't'),
                            black_box(&query[..]),
                            2,
                            *algo,
                            *prefix,
                        );
                        black_box(next)
                    });
                },
            );
        }
    }

    group.finish();
}

//==============================================================================
// COMPARISON BENCHMARKS
//==============================================================================

/// Compare algorithms head-to-head
fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");

    let query = b"approximate";
    let max_distance = 2;

    for algorithm in [
        Algorithm::Standard,
        Algorithm::Transposition,
        Algorithm::MergeAndSplit,
    ] {
        let state = initial_state(query.len(), max_distance, algorithm);

        group.throughput(Throughput::Elements(state.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", algorithm)),
            &(state, query, algorithm),
            |b, (state, query, algo)| {
                b.iter(|| {
                    let next = transition_state(
                        black_box(state),
                        Unrestricted,
                        black_box(b'a'),
                        black_box(&query[..]),
                        max_distance,
                        *algo,
                        false,
                    );
                    black_box(next)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark position allocation overhead (SmallVec vs Vec)
fn bench_position_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_allocation");

    // Test SmallVec allocation patterns
    for &count in &[1, 2, 3, 4, 5, 8] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("count={}", count)),
            &count,
            |b, &count| {
                b.iter(|| {
                    let mut positions: SmallVec<[Position; 4]> = SmallVec::new();
                    for i in 0..count {
                        positions.push(Position::new(i, 0));
                    }
                    black_box(positions)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_characteristic_vector,
    bench_index_of_match,
    bench_transition_standard,
    bench_transition_transposition,
    bench_transition_merge_split,
    bench_transition_state,
    bench_transition_by_state_size,
    bench_prefix_mode,
    bench_algorithm_comparison,
    bench_position_allocation,
);
criterion_main!(benches);
