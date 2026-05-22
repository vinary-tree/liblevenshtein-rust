use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::transducer::universal::{
    Standard, Transposition, UniversalPosition, UniversalState,
};
use smallvec::SmallVec;

/// Generate test positions with varying error distributions
fn generate_positions<V: liblevenshtein::transducer::universal::PositionVariant>(
    count: usize,
    max_distance: u8,
) -> Vec<UniversalPosition<V>> {
    let mut positions = Vec::new();

    // Generate positions with realistic error/offset distribution
    for i in 0..count {
        let offset = (i as i32 % 10) - 5; // Offsets in [-5, 4]
        let errors = (i / 3) as u8 % (max_distance + 1); // Varied errors

        if let Ok(pos) = UniversalPosition::new_i(offset, errors, max_distance) {
            positions.push(pos);
        }
    }

    positions
}

/// Benchmark current BTreeSet approach
fn bench_btreeset_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("universal_btreeset/Standard");

    for &pos_count in &[10, 20, 50, 100] {
        for &max_distance in &[1, 2, 3] {
            let positions = generate_positions::<Standard>(pos_count, max_distance);

            group.throughput(Throughput::Elements(pos_count as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("d={}/n={}", max_distance, pos_count)),
                &(&positions, max_distance),
                |b, (positions, max_distance)| {
                    b.iter(|| {
                        let mut state = UniversalState::<Standard>::new(*max_distance);
                        for pos in *positions {
                            state.add_position(black_box(pos.clone()));
                        }
                        black_box(state)
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_btreeset_transposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("universal_btreeset/Transposition");

    for &pos_count in &[10, 20, 50, 100] {
        for &max_distance in &[1, 2, 3] {
            let positions = generate_positions::<Transposition>(pos_count, max_distance);

            group.throughput(Throughput::Elements(pos_count as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("d={}/n={}", max_distance, pos_count)),
                &(&positions, max_distance),
                |b, (positions, max_distance)| {
                    b.iter(|| {
                        let mut state = UniversalState::<Transposition>::new(*max_distance);
                        for pos in *positions {
                            state.add_position(black_box(pos.clone()));
                        }
                        black_box(state)
                    });
                },
            );
        }
    }

    group.finish();
}

/// SmallVec-based state for comparison
#[derive(Debug, Clone)]
struct SmallVecState<V: liblevenshtein::transducer::universal::PositionVariant> {
    positions: SmallVec<[UniversalPosition<V>; 8]>,
    max_distance: u8,
}

impl<V: liblevenshtein::transducer::universal::PositionVariant> SmallVecState<V> {
    fn new(max_distance: u8) -> Self {
        Self {
            positions: SmallVec::new(),
            max_distance,
        }
    }

    fn add_position(&mut self, pos: UniversalPosition<V>) {
        use liblevenshtein::transducer::universal::subsumes;

        // Check if subsumed by existing position
        for existing in &self.positions {
            if subsumes(existing, &pos, self.max_distance) {
                return; // Already covered
            }
        }

        // Remove positions subsumed by new position
        self.positions
            .retain(|p| !subsumes(&pos, p, self.max_distance));

        // Insert in sorted position (binary search)
        let insert_pos = self.positions.binary_search(&pos).unwrap_or_else(|pos| pos);
        self.positions.insert(insert_pos, pos);
    }
}

/// Benchmark SmallVec approach
fn bench_smallvec_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("universal_smallvec/Standard");

    for &pos_count in &[10, 20, 50, 100] {
        for &max_distance in &[1, 2, 3] {
            let positions = generate_positions::<Standard>(pos_count, max_distance);

            group.throughput(Throughput::Elements(pos_count as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("d={}/n={}", max_distance, pos_count)),
                &(&positions, max_distance),
                |b, (positions, max_distance)| {
                    b.iter(|| {
                        let mut state = SmallVecState::<Standard>::new(*max_distance);
                        for pos in *positions {
                            state.add_position(black_box(pos.clone()));
                        }
                        black_box(state)
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_smallvec_transposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("universal_smallvec/Transposition");

    for &pos_count in &[10, 20, 50, 100] {
        for &max_distance in &[1, 2, 3] {
            let positions = generate_positions::<Transposition>(pos_count, max_distance);

            group.throughput(Throughput::Elements(pos_count as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("d={}/n={}", max_distance, pos_count)),
                &(&positions, max_distance),
                |b, (positions, max_distance)| {
                    b.iter(|| {
                        let mut state = SmallVecState::<Transposition>::new(*max_distance);
                        for pos in *positions {
                            state.add_position(black_box(pos.clone()));
                        }
                        black_box(state)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark state size distribution (how many positions typically remain)
fn bench_state_size_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_size_distribution");

    for &max_distance in &[1, 2, 3] {
        // BTreeSet
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("BTreeSet/d={}", max_distance)),
            &max_distance,
            |b, &max_distance| {
                b.iter(|| {
                    let positions = generate_positions::<Standard>(100, max_distance);
                    let mut state = UniversalState::<Standard>::new(max_distance);
                    for pos in positions {
                        state.add_position(black_box(pos));
                    }
                    black_box(state.len())
                });
            },
        );

        // SmallVec
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("SmallVec/d={}", max_distance)),
            &max_distance,
            |b, &max_distance| {
                b.iter(|| {
                    let positions = generate_positions::<Standard>(100, max_distance);
                    let mut state = SmallVecState::<Standard>::new(max_distance);
                    for pos in positions {
                        state.add_position(black_box(pos));
                    }
                    black_box(state.positions.len())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_btreeset_standard,
    bench_btreeset_transposition,
    bench_smallvec_standard,
    bench_smallvec_transposition,
    bench_state_size_distribution
);
criterion_main!(benches);
