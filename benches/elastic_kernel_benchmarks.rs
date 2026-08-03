//! Criterion benchmarks for production elastic kernels.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use liblevenshtein::time_series::{
    erp_gap_mass_lower_bound, frechet_candidate_lower_bound, lb_keogh, twed_length_lower_bound,
    DtwConfig, DtwTransducer, ErpConfig, ErpTransducer, FrechetConfig, FrechetTransducer,
    QuantizationConfig, TwedConfig, TwedTransducer,
};

fn deterministic_series(count: usize, len: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|series_id| {
            (0..len)
                .map(|sample| {
                    let phase = (series_id * 17 + sample * 31) % 257;
                    f64::from(u16::try_from(phase).expect("phase is at most 256")) / 16.0 - 8.0
                })
                .collect()
        })
        .collect()
}

fn erp_exact(c: &mut Criterion) {
    let config = ErpConfig::new(0.0);
    let mut group = c.benchmark_group("erp_exact");
    for len in [16usize, 64, 256] {
        let x = deterministic_series(1, len).pop().expect("one series");
        let y = deterministic_series(2, len).pop().expect("two series");
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| config.distance(black_box(&x), black_box(&y)));
        });
    }
    group.finish();
}

fn erp_candidate_bound(c: &mut Criterion) {
    let x = deterministic_series(1, 256).pop().expect("one series");
    let y = deterministic_series(2, 256).pop().expect("two series");
    c.bench_function("erp_gap_mass_bound/256", |b| {
        b.iter(|| erp_gap_mass_lower_bound(black_box(&x), black_box(&y), 0.0));
    });
}

fn erp_trie_range(c: &mut Criterion) {
    let references = deterministic_series(1_000, 64);
    let query = references[137].clone();
    let index = ErpTransducer::from_series(
        QuantizationConfig::for_u8(-8.0, 8.0),
        ErpConfig::new(0.0),
        &references,
    );
    c.bench_function("erp_trie_range/1000x64", |b| {
        b.iter(|| index.search_range(black_box(&query), black_box(12.0)));
    });
}

fn frechet_exact(c: &mut Criterion) {
    let config = FrechetConfig::new();
    let mut group = c.benchmark_group("frechet_exact");
    for len in [16usize, 64, 256] {
        let x = deterministic_series(1, len).pop().expect("one series");
        let y = deterministic_series(2, len).pop().expect("two series");
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| config.distance(black_box(&x), black_box(&y)));
        });
    }
    group.finish();
}

fn frechet_candidate_bound(c: &mut Criterion) {
    let x = deterministic_series(1, 256).pop().expect("one series");
    let y = deterministic_series(2, 256).pop().expect("two series");
    c.bench_function("frechet_candidate_bound/256", |b| {
        b.iter(|| frechet_candidate_lower_bound(black_box(&x), black_box(&y)));
    });
}

fn frechet_trie_range(c: &mut Criterion) {
    let references = deterministic_series(1_000, 64);
    let query = references[137].clone();
    let index = FrechetTransducer::from_series(
        QuantizationConfig::for_u8(-8.0, 8.0),
        FrechetConfig::new(),
        &references,
    );
    c.bench_function("frechet_trie_range/1000x64", |b| {
        b.iter(|| index.search_range(black_box(&query), black_box(3.0)));
    });
}

fn dtw_exact(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtw_exact");
    for len in [16usize, 64, 256] {
        let x = deterministic_series(1, len).pop().expect("one series");
        let y = deterministic_series(2, len).pop().expect("two series");
        for band in [2usize, 8, 32] {
            let config = DtwConfig::new(band);
            group.bench_with_input(
                BenchmarkId::new(format!("band_{band}"), len),
                &len,
                |b, _| b.iter(|| config.distance(black_box(&x), black_box(&y))),
            );
        }
    }
    group.finish();
}

fn dtw_candidate_bound(c: &mut Criterion) {
    let x = deterministic_series(1, 256).pop().expect("one series");
    let y = deterministic_series(2, 256).pop().expect("two series");
    c.bench_function("dtw_lb_keogh/256_band_16", |b| {
        b.iter(|| lb_keogh(black_box(&x), black_box(&y), black_box(16)));
    });
}

fn dtw_trie_range(c: &mut Criterion) {
    let references = deterministic_series(1_000, 64);
    let query = references[137].clone();
    let index = DtwTransducer::from_series(
        QuantizationConfig::for_u8(-8.0, 8.0),
        DtwConfig::new(8),
        &references,
    );
    c.bench_function("dtw_trie_range/1000x64_band_8", |b| {
        b.iter(|| index.search_range(black_box(&query), black_box(8.0)));
    });
}

fn twed_exact(c: &mut Criterion) {
    let config = TwedConfig::new(0.5, 1.0);
    let mut group = c.benchmark_group("twed_exact");
    for len in [16usize, 64, 256] {
        let x = deterministic_series(1, len).pop().expect("one series");
        let y = deterministic_series(2, len).pop().expect("two series");
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| config.distance(black_box(&x), black_box(&y)));
        });
    }
    group.finish();
}

fn twed_candidate_bound(c: &mut Criterion) {
    c.bench_function("twed_length_bound/192x256", |b| {
        b.iter(|| twed_length_lower_bound(black_box(192), black_box(256), black_box(1.0)));
    });
}

fn twed_trie_range(c: &mut Criterion) {
    let references = deterministic_series(1_000, 64);
    let query = references[137].clone();
    let index = TwedTransducer::from_series(
        QuantizationConfig::for_u8(-8.0, 8.0),
        TwedConfig::new(0.5, 1.0),
        &references,
    );
    c.bench_function("twed_trie_range/1000x64", |b| {
        b.iter(|| index.search_range(black_box(&query), black_box(12.0)));
    });
}

criterion_group!(
    elastic_kernel_benches,
    erp_exact,
    erp_candidate_bound,
    erp_trie_range,
    frechet_exact,
    frechet_candidate_bound,
    frechet_trie_range,
    dtw_exact,
    dtw_candidate_bound,
    dtw_trie_range,
    twed_exact,
    twed_candidate_bound,
    twed_trie_range
);
criterion_main!(elastic_kernel_benches);
