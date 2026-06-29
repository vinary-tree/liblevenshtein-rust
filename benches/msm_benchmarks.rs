//! Benchmarks for MSM (Move-Split-Merge) time series distance metric.
//!
//! This benchmark suite compares:
//! 1. Direct MSM DP algorithm vs automaton-based approach
//! 2. Lower bound pruning effectiveness
//! 3. Hybrid search with different configurations
//! 4. Effect of series length and database size on performance

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::time_series::{
    combined_lb, euclidean_lb, length_lb, msm_distance_automaton, msm_distance_wavefront,
    search_with_lb, HybridSearchIndex, LowerBoundType, MsmConfig, MsmTransducer,
    QuantizationConfig,
};

/// Generate a random time series
fn generate_series(len: usize, seed: u64) -> Vec<f64> {
    // Simple PRNG for reproducibility
    let mut state = seed;
    let mut series = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let value = ((state >> 33) as f64) / (u32::MAX as f64) * 100.0;
        series.push(value);
    }
    series
}

fn bench_msm_dp(c: &mut Criterion) {
    let config = MsmConfig::new(1.0);
    let mut group = c.benchmark_group("msm_dp");

    for len in [10, 50, 100, 500].iter() {
        let x = generate_series(*len, 12345);
        let y = generate_series(*len, 67890);

        group.throughput(Throughput::Elements(*len as u64));
        group.bench_with_input(BenchmarkId::new("standard", len), len, |b, _| {
            b.iter(|| config.distance(black_box(&x), black_box(&y)));
        });

        group.bench_with_input(BenchmarkId::new("optimized", len), len, |b, _| {
            b.iter(|| config.distance_optimized(black_box(&x), black_box(&y)));
        });
    }
    group.finish();
}

fn bench_lower_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("lower_bounds");

    for len in [10, 50, 100, 500].iter() {
        let x = generate_series(*len, 12345);
        let y = generate_series(*len, 67890);

        group.throughput(Throughput::Elements(*len as u64));

        group.bench_with_input(BenchmarkId::new("euclidean", len), len, |b, _| {
            b.iter(|| euclidean_lb(black_box(&x), black_box(&y)));
        });

        group.bench_with_input(BenchmarkId::new("length", len), len, |b, _| {
            b.iter(|| length_lb(black_box(&x), black_box(&y), 1.0));
        });

        group.bench_with_input(BenchmarkId::new("combined", len), len, |b, _| {
            b.iter(|| combined_lb(black_box(&x), black_box(&y), 1.0));
        });
    }
    group.finish();
}

fn bench_lb_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("lb_speedup");

    for len in [50, 100].iter() {
        let x = generate_series(*len, 12345);
        let y = generate_series(*len, 67890);
        let config = MsmConfig::new(1.0);

        // Compare LB computation vs full MSM
        group.bench_with_input(BenchmarkId::new("lower_bound", len), len, |b, _| {
            b.iter(|| combined_lb(black_box(&x), black_box(&y), 1.0));
        });

        group.bench_with_input(BenchmarkId::new("full_msm", len), len, |b, _| {
            b.iter(|| config.distance(black_box(&x), black_box(&y)));
        });
    }
    group.finish();
}

fn bench_hybrid_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_search");
    group.sample_size(30); // Reduce samples for slower benchmarks

    for db_size in [100, 500, 1000].iter() {
        let series_len = 50;
        let database: Vec<Vec<f64>> = (0..*db_size)
            .map(|i| generate_series(series_len, i as u64 * 1000))
            .collect();

        let query = generate_series(series_len, 99999);
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);

        // Build index
        let mut index = HybridSearchIndex::new(quant_config.clone(), msm_config);
        for (i, series) in database.iter().enumerate() {
            index.insert(i, series);
        }

        let threshold = 50.0;

        // Benchmark with lower bounds enabled
        group.bench_with_input(BenchmarkId::new("with_lb", db_size), db_size, |b, _| {
            b.iter(|| index.search_exact(black_box(&query), black_box(threshold)));
        });

        // Benchmark with lower bounds disabled
        let mut index_no_lb = HybridSearchIndex::new(quant_config.clone(), msm_config);
        for (i, series) in database.iter().enumerate() {
            index_no_lb.insert(i, series);
        }
        index_no_lb.set_use_lower_bounds(false);

        group.bench_with_input(BenchmarkId::new("without_lb", db_size), db_size, |b, _| {
            b.iter(|| index_no_lb.search_exact(black_box(&query), black_box(threshold)));
        });
    }
    group.finish();
}

fn bench_brute_force_vs_indexed(c: &mut Criterion) {
    let mut group = c.benchmark_group("brute_force_vs_indexed");
    group.sample_size(20);

    for db_size in [100, 500].iter() {
        let series_len = 50;
        let database: Vec<(usize, Vec<f64>)> = (0..*db_size)
            .map(|i| (i, generate_series(series_len, i as u64 * 1000)))
            .collect();

        let query = generate_series(series_len, 99999);
        let msm_config = MsmConfig::new(1.0);
        let threshold = 50.0;

        // Brute force with lower bounds
        group.bench_with_input(
            BenchmarkId::new("brute_force_lb", db_size),
            db_size,
            |b, _| {
                b.iter(|| {
                    search_with_lb(
                        black_box(&query),
                        black_box(&database),
                        black_box(threshold),
                        black_box(&msm_config),
                    )
                });
            },
        );

        // Indexed search (hybrid)
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);
        for (i, (_, series)) in database.iter().enumerate() {
            index.insert(i, series);
        }

        group.bench_with_input(
            BenchmarkId::new("indexed_hybrid", db_size),
            db_size,
            |b, _| {
                b.iter(|| index.search_exact(black_box(&query), black_box(threshold)));
            },
        );
    }
    group.finish();
}

fn bench_lb_type_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("lb_type_comparison");
    group.sample_size(30);

    let db_size = 500;
    let series_len = 50;
    let database: Vec<Vec<f64>> = (0..db_size)
        .map(|i| generate_series(series_len, i as u64 * 1000))
        .collect();

    let query = generate_series(series_len, 99999);
    let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
    let msm_config = MsmConfig::new(1.0);
    let threshold = 50.0;

    for lb_type in [
        ("length_only", LowerBoundType::LengthOnly),
        ("euclidean", LowerBoundType::EuclideanOnly),
        ("l1", LowerBoundType::L1Only),
        ("combined", LowerBoundType::Combined),
    ] {
        let mut index = HybridSearchIndex::new(quant_config.clone(), msm_config);
        for (i, series) in database.iter().enumerate() {
            index.insert(i, series);
        }
        index.set_lower_bound_type(lb_type.1);

        group.bench_with_input(BenchmarkId::new(lb_type.0, db_size), &db_size, |b, _| {
            b.iter(|| index.search_exact(black_box(&query), black_box(threshold)));
        });
    }
    group.finish();
}

fn bench_quantization_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_levels");
    group.sample_size(30);

    let db_size = 300;
    let series_len = 50;
    let database: Vec<Vec<f64>> = (0..db_size)
        .map(|i| generate_series(series_len, i as u64 * 1000))
        .collect();

    let query = generate_series(series_len, 99999);
    let msm_config = MsmConfig::new(1.0);
    let threshold = 50.0;

    for bins in [16, 64, 256].iter() {
        let quant_config = QuantizationConfig::uniform(0.0, 100.0, *bins as u32);
        let mut index = HybridSearchIndex::new(quant_config, msm_config);
        for (i, series) in database.iter().enumerate() {
            index.insert(i, series);
        }

        group.bench_with_input(BenchmarkId::new("bins", bins), bins, |b, _| {
            b.iter(|| index.search_exact(black_box(&query), black_box(threshold)));
        });
    }
    group.finish();
}

fn generate_prefix_shared_database(db_size: usize, len: usize) -> Vec<Vec<f64>> {
    let base = generate_series(len, 42);
    (0..db_size)
        .map(|i| {
            let mut series = base.clone();
            let pivot = len.saturating_mul(3) / 4;
            for (j, value) in series.iter_mut().enumerate().skip(pivot) {
                let perturb = ((i as f64 + 1.0) * (j as f64 + 0.5)).sin() * 2.0;
                *value = (*value + perturb).clamp(0.0, 100.0);
            }
            series
        })
        .collect()
}

fn bench_exact_msm_transducer(c: &mut Criterion) {
    let mut group = c.benchmark_group("exact_msm_transducer");
    group.sample_size(30);

    for db_size in [128, 512].iter() {
        let series_len = 48;
        let database = generate_prefix_shared_database(*db_size, series_len);
        let query = database[0]
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if i % 7 == 0 {
                    (*v + 1.5).clamp(0.0, 100.0)
                } else {
                    *v
                }
            })
            .collect::<Vec<_>>();
        let database_pairs = database
            .iter()
            .cloned()
            .enumerate()
            .collect::<Vec<(usize, Vec<f64>)>>();
        let quant_config = QuantizationConfig::for_u8(0.0, 100.0);
        let msm_config = MsmConfig::new(1.0);
        let threshold = 24.0;

        let transducer = MsmTransducer::from_series(quant_config.clone(), msm_config, &database);
        let mut hybrid = HybridSearchIndex::new(quant_config, msm_config);
        for (i, series) in database.iter().enumerate() {
            hybrid.insert(i, series);
        }

        group.bench_with_input(
            BenchmarkId::new("brute_force_lb_range", db_size),
            db_size,
            |b, _| {
                b.iter(|| {
                    search_with_lb(
                        black_box(&query),
                        black_box(&database_pairs),
                        black_box(threshold),
                        black_box(&msm_config),
                    )
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("exact_transducer_range", db_size),
            db_size,
            |b, _| {
                b.iter(|| transducer.search_range(black_box(&query), black_box(threshold)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("exact_transducer_knn", db_size),
            db_size,
            |b, _| {
                b.iter(|| transducer.search_knn(black_box(&query), black_box(8), black_box(1.0)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hybrid_exact", db_size),
            db_size,
            |b, _| {
                b.iter(|| hybrid.search_exact(black_box(&query), black_box(threshold)));
            },
        );
    }

    group.finish();
}

fn bench_legacy_msm_automata(c: &mut Criterion) {
    let mut group = c.benchmark_group("legacy_msm_automata");
    group.sample_size(20);
    let config = MsmConfig::new(1.0);

    for len in [12, 24].iter() {
        let x = generate_series(*len, 12345);
        let y = generate_series(*len, 67890);

        group.throughput(Throughput::Elements(*len as u64));
        group.bench_with_input(BenchmarkId::new("optimized_dp", len), len, |b, _| {
            b.iter(|| config.distance_optimized(black_box(&x), black_box(&y)));
        });
        group.bench_with_input(BenchmarkId::new("wavefront", len), len, |b, _| {
            b.iter(|| {
                msm_distance_wavefront(
                    black_box(&x),
                    black_box(&y),
                    black_box(&config),
                    black_box(f64::INFINITY),
                )
            });
        });
        group.bench_with_input(BenchmarkId::new("automaton", len), len, |b, _| {
            b.iter(|| {
                msm_distance_automaton(
                    black_box(&x),
                    black_box(&y),
                    black_box(&config),
                    black_box(f64::INFINITY),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_msm_dp,
    bench_lower_bounds,
    bench_lb_speedup,
    bench_hybrid_search,
    bench_brute_force_vs_indexed,
    bench_lb_type_comparison,
    bench_quantization_levels,
    bench_exact_msm_transducer,
    bench_legacy_msm_automata,
);
criterion_main!(benches);
