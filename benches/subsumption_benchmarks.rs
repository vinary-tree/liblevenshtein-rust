use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::transducer::{Algorithm, Position, State};

/// Generate test positions for benchmarking
fn generate_positions(count: usize, max_distance: usize, query_length: usize) -> Vec<Position> {
    let mut positions = Vec::new();
    for i in 0..count {
        let term_index = i % (query_length + 1);
        let num_errors = (i / (query_length + 1)) % (max_distance + 1);
        let is_special = (i % 3) == 0; // Some special positions
        if is_special {
            positions.push(Position::new_special(term_index, num_errors));
        } else {
            positions.push(Position::new(term_index, num_errors));
        }
    }
    positions
}

/// Benchmark online subsumption (current Rust approach)
fn bench_online_subsumption(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_subsumption");

    for &pos_count in &[10, 50, 100, 200] {
        for &max_distance in &[0, 2, 5, 10] {
            let query_length = 20;
            let positions = generate_positions(pos_count, max_distance, query_length);

            for algorithm in [
                Algorithm::Standard,
                Algorithm::Transposition,
                Algorithm::MergeAndSplit,
            ] {
                group.throughput(Throughput::Elements(pos_count as u64));
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "{:?}/d={}/n={}",
                        algorithm, max_distance, pos_count
                    )),
                    &(&positions, algorithm),
                    |b, (positions, algorithm)| {
                        b.iter(|| {
                            let mut state = State::new();
                            for pos in *positions {
                                state.insert(black_box(*pos), black_box(*algorithm), query_length);
                            }
                            black_box(state)
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark batch subsumption (C++ approach - insert all then unsubsume)
fn bench_batch_subsumption(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_subsumption");

    for &pos_count in &[10, 50, 100, 200] {
        for &max_distance in &[0, 2, 5, 10] {
            let query_length = 20;
            let positions = generate_positions(pos_count, max_distance, query_length);

            for algorithm in [
                Algorithm::Standard,
                Algorithm::Transposition,
                Algorithm::MergeAndSplit,
            ] {
                group.throughput(Throughput::Elements(pos_count as u64));
                group.bench_with_input(
                    BenchmarkId::from_parameter(format!(
                        "{:?}/d={}/n={}",
                        algorithm, max_distance, pos_count
                    )),
                    &(&positions, algorithm),
                    |b, (positions, algorithm)| {
                        b.iter(|| {
                            // Batch approach: insert all positions first
                            let mut state_positions = Vec::new();
                            for pos in *positions {
                                state_positions.push(black_box(*pos));
                            }

                            // Then remove subsumed positions (simulating unsubsume)
                            batch_unsubsume(
                                &mut state_positions,
                                black_box(*algorithm),
                                query_length,
                            );

                            // Sort (like C++ does)
                            state_positions.sort();

                            black_box(state_positions)
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Simulate C++ batch unsubsumption
fn batch_unsubsume(positions: &mut Vec<Position>, algorithm: Algorithm, query_length: usize) {
    let mut to_remove = Vec::new();

    // Nested loop like C++ implementation
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            if positions[i].subsumes(&positions[j], algorithm, query_length) {
                to_remove.push(j);
            } else if positions[j].subsumes(&positions[i], algorithm, query_length) {
                to_remove.push(i);
                break; // Don't check more if outer is subsumed
            }
        }
    }

    // Remove duplicates and sort indices in reverse
    to_remove.sort_unstable();
    to_remove.dedup();
    to_remove.reverse();

    // Remove in reverse order to maintain indices
    for idx in to_remove {
        positions.swap_remove(idx);
    }
}

/// Benchmark subsumption with varying max_distance
fn bench_subsumption_by_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("subsumption_by_distance");

    let pos_count = 100;
    let query_length = 20;

    // Test edge cases: 0, small, medium, large, and max distance
    for &max_distance in &[0, 1, 2, 5, 10, 50, 100, usize::MAX / 2] {
        let positions = generate_positions(pos_count, max_distance.min(100), query_length);

        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
        ] {
            group.throughput(Throughput::Elements(pos_count as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{:?}/d={}", algorithm, max_distance)),
                &(&positions, algorithm, max_distance),
                |b, (positions, algorithm, _max_distance)| {
                    b.iter(|| {
                        let mut state = State::new();
                        for pos in *positions {
                            state.insert(black_box(*pos), black_box(*algorithm), query_length);
                        }
                        black_box(state)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark worst-case scenario: no subsumption possible
fn bench_no_subsumption(c: &mut Criterion) {
    let mut group = c.benchmark_group("no_subsumption");

    for &pos_count in &[10, 50, 100, 200] {
        let query_length = pos_count; // Each position at different index

        // Create positions that don't subsume each other
        let mut positions = Vec::new();
        for i in 0..pos_count {
            positions.push(Position::new(i, 0)); // All at different indices with 0 errors
        }

        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
        ] {
            group.throughput(Throughput::Elements(pos_count as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{:?}/n={}", algorithm, pos_count)),
                &(&positions, algorithm),
                |b, (positions, algorithm)| {
                    b.iter(|| {
                        let mut state = State::new();
                        for pos in *positions {
                            state.insert(black_box(*pos), black_box(*algorithm), query_length);
                        }
                        black_box(state)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark best-case scenario: all positions subsumed by first
fn bench_all_subsumed(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_subsumed");

    for &pos_count in &[10, 50, 100, 200] {
        let max_distance = 10;
        let query_length = 20; // Fixed query length for subsumption testing

        // Create positions where (0,0) subsumes all others
        let mut positions = vec![Position::new(0, 0)];
        for i in 1..pos_count {
            // These will all be subsumed by (0,0)
            positions.push(Position::new(i % 5, i % (max_distance + 1)));
        }

        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
        ] {
            group.throughput(Throughput::Elements(pos_count as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{:?}/n={}", algorithm, pos_count)),
                &(&positions, algorithm),
                |b, (positions, algorithm)| {
                    b.iter(|| {
                        let mut state = State::new();
                        for pos in *positions {
                            state.insert(black_box(*pos), black_box(*algorithm), query_length);
                        }
                        black_box(state)
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_online_subsumption,
    bench_batch_subsumption,
    bench_subsumption_by_distance,
    bench_no_subsumption,
    bench_all_subsumed
);
criterion_main!(benches);
