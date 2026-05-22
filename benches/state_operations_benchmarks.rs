use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::transducer::{Algorithm, Position, State};

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

/// Generate a state with n positions
fn generate_state(n: usize, max_distance: usize) -> State {
    let mut state = State::new();
    let query_length = n + 10; // Reasonable query length for benchmarks
    for i in 0..n {
        let errors = i % (max_distance + 1);
        state.insert(Position::new(i, errors), Algorithm::Standard, query_length);
    }
    state
}

/// Generate state with specific error distribution
fn generate_state_with_errors(positions: &[(usize, usize)]) -> State {
    let mut state = State::new();
    let query_length = 20; // Reasonable query length for benchmarks
    for &(term_index, num_errors) in positions {
        state.insert(
            Position::new(term_index, num_errors),
            Algorithm::Standard,
            query_length,
        );
    }
    state
}

//==============================================================================
// QUERY OPERATIONS BENCHMARKS
//==============================================================================

/// Benchmark min_distance() with various state sizes
fn bench_min_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_distance");

    for &n in &[1, 2, 3, 5, 10, 20] {
        let state = generate_state(n, 3);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}", n)),
            &state,
            |b, state| {
                b.iter(|| black_box(state.min_distance()));
            },
        );
    }

    group.finish();
}

/// Benchmark infer_distance() with various state sizes
fn bench_infer_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("infer_distance");

    for &n in &[1, 2, 3, 5, 10, 20] {
        let state = generate_state(n, 3);
        let query_length = 10;

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}", n)),
            &(&state, query_length),
            |b, (state, query_length)| {
                b.iter(|| black_box(state.infer_distance(*query_length)));
            },
        );
    }

    group.finish();
}

/// Benchmark infer_prefix_distance() with various state sizes
fn bench_infer_prefix_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("infer_prefix_distance");

    for &n in &[1, 2, 3, 5, 10, 20] {
        // Create state where some positions have consumed the query
        let state = generate_state(n, 3);
        let query_length = 5; // Some positions will have term_index >= query_length

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}", n)),
            &(&state, query_length),
            |b, (state, query_length)| {
                b.iter(|| black_box(state.infer_prefix_distance(*query_length)));
            },
        );
    }

    group.finish();
}

//==============================================================================
// COPY OPERATIONS BENCHMARKS
//==============================================================================

/// Benchmark copy_from() with various state sizes
fn bench_copy_from(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_from");

    for &n in &[1, 2, 3, 5, 10, 20, 50] {
        let source = generate_state(n, 3);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}", n)),
            &source,
            |b, source| {
                let mut dest = State::new();
                b.iter(|| {
                    dest.copy_from(black_box(source));
                    black_box(&dest);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark alternative copy_from using clone
fn bench_copy_from_alternative(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_from_alternative");

    for &n in &[1, 2, 3, 5, 10, 20, 50] {
        let source = generate_state(n, 3);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}", n)),
            &source,
            |b, _source| {
                b.iter(|| {
                    // Alternative: just clone the entire state
                    let dest = source.clone();
                    black_box(&dest);
                });
            },
        );
    }

    group.finish();
}

//==============================================================================
// MERGE OPERATIONS BENCHMARKS
//==============================================================================

/// Benchmark merge() with various state sizes
fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge");

    for &n in &[1, 2, 3, 5, 10] {
        for &m in &[1, 2, 3, 5, 10] {
            let state1 = generate_state(n, 3);
            let state2 = generate_state_with_errors(&[(10, 1), (11, 1), (12, 2), (13, 2), (14, 3)]);

            group.throughput(Throughput::Elements((n + m) as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("n={},m={}", n, m)),
                &(&state1, &state2),
                |b, (state1, state2)| {
                    b.iter(|| {
                        let mut dest = (*state1).clone();
                        let query_length = 20;
                        dest.merge(black_box(*state2), Algorithm::Standard, query_length);
                        black_box(&dest);
                    });
                },
            );
        }
    }

    group.finish();
}

//==============================================================================
// CREATION BENCHMARKS
//==============================================================================

/// Benchmark state creation operations
fn bench_state_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_creation");

    // Benchmark State::new()
    group.bench_function("new", |b| {
        b.iter(|| black_box(State::new()));
    });

    // Benchmark State::single()
    group.bench_function("single", |b| {
        b.iter(|| black_box(State::single(Position::new(0, 0))));
    });

    // Benchmark State::from_positions()
    for &n in &[1, 3, 5, 10, 20] {
        let positions: Vec<Position> = (0..n).map(|i| Position::new(i, i % 3)).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("from_positions/n={}", n)),
            &positions,
            |b, positions| {
                b.iter(|| black_box(State::from_positions(positions.clone())));
            },
        );
    }

    group.finish();
}

//==============================================================================
// ACCESSOR BENCHMARKS
//==============================================================================

/// Benchmark simple accessor operations
fn bench_accessors(c: &mut Criterion) {
    let mut group = c.benchmark_group("accessors");

    let state = generate_state(5, 3);

    group.bench_function("head", |b| {
        b.iter(|| black_box(state.head()));
    });

    group.bench_function("is_empty", |b| {
        b.iter(|| black_box(state.is_empty()));
    });

    group.bench_function("len", |b| {
        b.iter(|| black_box(state.len()));
    });

    group.bench_function("positions", |b| {
        b.iter(|| black_box(state.positions()));
    });

    group.bench_function("iter", |b| {
        b.iter(|| {
            let count = state.iter().count();
            black_box(count)
        });
    });

    group.finish();
}

//==============================================================================
// DISTANCE COMPUTATION COMPARISON
//==============================================================================

/// Compare distance computation methods
fn bench_distance_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_comparison");

    // Test with different state configurations
    let configs = vec![
        ("single_pos", generate_state(1, 0)),
        ("small_state", generate_state(3, 2)),
        ("medium_state", generate_state(5, 3)),
        ("large_state", generate_state(10, 5)),
    ];

    for (name, state) in &configs {
        let query_length = 10;

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}/min_distance", name)),
            state,
            |b, state| {
                b.iter(|| black_box(state.min_distance()));
            },
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}/infer_distance", name)),
            &(state, query_length),
            |b, (state, query_length)| {
                b.iter(|| black_box(state.infer_distance(*query_length)));
            },
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}/infer_prefix_distance", name)),
            &(state, query_length),
            |b, (state, query_length)| {
                b.iter(|| black_box(state.infer_prefix_distance(*query_length)));
            },
        );
    }

    group.finish();
}

//==============================================================================
// REALISTIC USAGE PATTERNS
//==============================================================================

/// Benchmark realistic query pattern: create, populate, query, drop
fn bench_realistic_query_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_query_pattern");

    for &max_distance in &[1, 2, 3] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d={}", max_distance)),
            &max_distance,
            |b, &max_d| {
                b.iter(|| {
                    // Typical pattern in dictionary traversal
                    let mut state = State::single(Position::new(0, 0));
                    let query_length = 10;

                    // Simulate adding a few transitions
                    for i in 1..=3 {
                        state.insert(
                            Position::new(i, (i - 1) % (max_d + 1)),
                            Algorithm::Standard,
                            query_length,
                        );
                    }

                    // Query distance
                    let dist = state.infer_distance(query_length);

                    black_box(dist)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark StatePool-style usage
fn bench_state_pool_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_pool_pattern");

    let source_state = generate_state(5, 3);

    group.bench_function("pool_reuse", |b| {
        let mut pooled_state = State::new();
        b.iter(|| {
            // Simulate pool reuse pattern
            pooled_state.clear();
            pooled_state.copy_from(&source_state);

            // Do some work
            let dist = pooled_state.min_distance();

            black_box(dist)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_min_distance,
    bench_infer_distance,
    bench_infer_prefix_distance,
    bench_copy_from,
    bench_copy_from_alternative,
    bench_merge,
    bench_state_creation,
    bench_accessors,
    bench_distance_comparison,
    bench_realistic_query_pattern,
    bench_state_pool_pattern,
);
criterion_main!(benches);
