#![cfg_attr(
    not(all(target_arch = "x86_64", feature = "simd")),
    allow(unused_variables)
)]

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use criterion::{black_box, BenchmarkId};
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use liblevenshtein::transducer::simd::check_subsumption_simd;

/// Benchmark subsumption checking with SIMD vs scalar
///
/// This benchmark measures the performance improvement of SIMD-accelerated
/// position subsumption checking, which is critical for the Standard algorithm's
/// state insertion operation.
fn bench_subsumption_simd_vs_scalar(c: &mut Criterion) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        let mut group = c.benchmark_group("subsumption_simd_vs_scalar");

        // Test case 1: Small batch (4 pairs - SSE4.1 threshold)
        let lhs_indices_4 = [5, 3, 10, 0];
        let lhs_errs_4 = [2, 2, 1, 0];
        let rhs_indices_4 = [5, 3, 8, 1];
        let rhs_errs_4 = [3, 2, 3, 1];

        group.bench_function(BenchmarkId::new("SIMD", "4_pairs"), |b| {
            let mut results = [false; 8];
            b.iter(|| {
                check_subsumption_simd(
                    black_box(&lhs_indices_4),
                    black_box(&lhs_errs_4),
                    black_box(&rhs_indices_4),
                    black_box(&rhs_errs_4),
                    black_box(4),
                    black_box(&mut results),
                );
            });
        });

        // Test case 2: Full batch (8 pairs - AVX2)
        let lhs_indices_8 = [5, 5, 3, 3, 10, 10, 0, 0];
        let lhs_errs_8 = [2, 2, 2, 3, 1, 1, 0, 0];
        let rhs_indices_8 = [5, 4, 3, 5, 8, 5, 0, 1];
        let rhs_errs_8 = [3, 3, 2, 2, 3, 3, 0, 1];

        group.bench_function(BenchmarkId::new("SIMD", "8_pairs"), |b| {
            let mut results = [false; 8];
            b.iter(|| {
                check_subsumption_simd(
                    black_box(&lhs_indices_8),
                    black_box(&lhs_errs_8),
                    black_box(&rhs_indices_8),
                    black_box(&rhs_errs_8),
                    black_box(8),
                    black_box(&mut results),
                );
            });
        });

        // Test case 3: Large indices (realistic position values)
        let lhs_indices_large = [100, 200, 300, 500, 1000, 1500, 2000, 2500];
        let lhs_errs_large = [5, 10, 15, 20, 25, 30, 35, 40];
        let rhs_indices_large = [105, 215, 325, 540, 1050, 1580, 2100, 2650];
        let rhs_errs_large = [10, 25, 40, 60, 75, 95, 110, 130];

        group.bench_function(BenchmarkId::new("SIMD", "large_indices"), |b| {
            let mut results = [false; 8];
            b.iter(|| {
                check_subsumption_simd(
                    black_box(&lhs_indices_large),
                    black_box(&lhs_errs_large),
                    black_box(&rhs_indices_large),
                    black_box(&rhs_errs_large),
                    black_box(8),
                    black_box(&mut results),
                );
            });
        });

        group.finish();
    }

    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    {
        println!("SIMD benchmarks require x86_64 architecture and simd feature");
    }
}

/// Benchmark realistic subsumption workload
///
/// This simulates the typical usage pattern: checking multiple candidate positions
/// against existing state positions during state insertion.
fn bench_subsumption_realistic_workload(c: &mut Criterion) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        let mut group = c.benchmark_group("subsumption_realistic_workload");

        // Simulate checking 64 position pairs (8 batches of 8)
        // This represents a moderate-sized state with ~8 positions checking against
        // ~8 candidate positions
        let test_data: Vec<_> = (0..8)
            .map(|batch| {
                let offset = batch * 10;
                let lhs_indices = [
                    offset,
                    offset + 1,
                    offset + 2,
                    offset + 3,
                    offset + 4,
                    offset + 5,
                    offset + 6,
                    offset + 7,
                ];
                let lhs_errs = [
                    batch,
                    batch,
                    batch + 1,
                    batch + 1,
                    batch + 2,
                    batch + 2,
                    batch + 3,
                    batch + 3,
                ];
                let rhs_indices = [
                    offset + 1,
                    offset + 2,
                    offset + 4,
                    offset + 5,
                    offset + 7,
                    offset + 9,
                    offset + 10,
                    offset + 12,
                ];
                let rhs_errs = [
                    batch + 1,
                    batch + 2,
                    batch + 3,
                    batch + 4,
                    batch + 5,
                    batch + 6,
                    batch + 7,
                    batch + 8,
                ];
                (lhs_indices, lhs_errs, rhs_indices, rhs_errs)
            })
            .collect();

        group.bench_function("SIMD_64_pairs", |b| {
            let mut results = [false; 8];
            b.iter(|| {
                for (lhs_i, lhs_e, rhs_j, rhs_f) in &test_data {
                    check_subsumption_simd(
                        black_box(lhs_i),
                        black_box(lhs_e),
                        black_box(rhs_j),
                        black_box(rhs_f),
                        black_box(8),
                        black_box(&mut results),
                    );
                }
            });
        });

        group.finish();
    }
}

/// Benchmark minimum distance finding with SIMD
///
/// This benchmark measures the performance of SIMD horizontal reduction
/// for finding the minimum error count across positions in a state.
fn bench_minimum_distance_simd(c: &mut Criterion) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        use liblevenshtein::transducer::simd::find_minimum_simd;

        let mut group = c.benchmark_group("minimum_distance_simd");

        // Test case 1: Small state (2 positions - common case)
        let errors_2 = vec![2, 1];

        group.bench_function("2_values", |b| {
            b.iter(|| black_box(find_minimum_simd(black_box(&errors_2), black_box(2))));
        });

        // Test case 2: Medium state (4 positions - SSE4.1 threshold)
        let errors_4 = vec![3, 1, 2, 4];

        group.bench_function("4_values", |b| {
            b.iter(|| black_box(find_minimum_simd(black_box(&errors_4), black_box(4))));
        });

        // Test case 3: Large state (8 positions - AVX2 full)
        let errors_8 = vec![5, 2, 7, 1, 3, 6, 4, 8];

        group.bench_function("8_values", |b| {
            b.iter(|| black_box(find_minimum_simd(black_box(&errors_8), black_box(8))));
        });

        // Test case 4: Realistic workload (many states, varying sizes)
        let test_data: Vec<(Vec<usize>, usize)> = vec![
            (vec![2, 1], 2),
            (vec![3, 1, 2], 3),
            (vec![5, 2, 7, 1], 4),
            (vec![3, 1, 2, 4, 6], 5),
            (vec![5, 2, 7, 1, 3, 6], 6),
            (vec![5, 2, 7, 1, 3, 6, 4], 7),
            (vec![5, 2, 7, 1, 3, 6, 4, 8], 8),
        ];

        group.bench_function("realistic_workload", |b| {
            b.iter(|| {
                for (values, count) in &test_data {
                    black_box(find_minimum_simd(black_box(values), black_box(*count)));
                }
            });
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_subsumption_simd_vs_scalar,
    bench_subsumption_realistic_workload,
    bench_minimum_distance_simd,
);
criterion_main!(benches);
