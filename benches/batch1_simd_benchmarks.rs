// Benchmark for Batch 1 SIMD optimizations (SSE4.1 + affix stripping)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use liblevenshtein::distance::standard_distance;
#[cfg(target_arch = "x86_64")]
use liblevenshtein::distance::standard_distance_impl;

#[cfg(target_arch = "x86_64")]
use liblevenshtein::distance::simd::{standard_distance_simd, strip_common_affixes_simd};

#[cfg(target_arch = "x86_64")]
fn bench_sse41_vs_avx2(c: &mut Criterion) {
    let mut group = c.benchmark_group("sse41_vs_avx2");

    let test_cases = vec![
        ("short", "kitten", "sitting"),
        ("medium", "saturday_morning", "sunday_evening"),
        (
            "long",
            "this_is_a_longer_string_for_testing",
            "this_is_another_longer_string_test",
        ),
    ];

    for (name, source, target) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("simd", name),
            &(source, target),
            |b, (s, t)| b.iter(|| standard_distance_simd(black_box(s), black_box(t))),
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &(source, target),
            |b, (s, t)| b.iter(|| standard_distance_impl(black_box(s), black_box(t))),
        );
    }

    group.finish();
}

#[cfg(target_arch = "x86_64")]
fn bench_affix_stripping(c: &mut Criterion) {
    let mut group = c.benchmark_group("affix_stripping");

    let test_cases = vec![
        ("no_common", "abcdefgh", "xyztuvwx"),
        ("common_prefix_8", "prefix12_abc", "prefix12_xyz"),
        (
            "common_prefix_16",
            "long_prefix_here_abc",
            "long_prefix_here_xyz",
        ),
        ("common_suffix_8", "abc_suffix12", "xyz_suffix12"),
        ("common_both", "prefix_middle_suffix", "prefix_other_suffix"),
        ("identical", "abcdefghijklmnop", "abcdefghijklmnop"),
    ];

    for (name, source, target) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("simd", name),
            &(source, target),
            |b, (s, t)| b.iter(|| strip_common_affixes_simd(black_box(s), black_box(t))),
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &(source, target),
            |b, (s, t)| {
                b.iter(|| {
                    liblevenshtein::distance::strip_common_affixes(black_box(s), black_box(t))
                })
            },
        );
    }

    group.finish();
}

fn bench_integrated_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("integrated_distance_batch1");

    let test_cases = vec![
        ("short", "test", "best"),
        ("medium_similar", "saturday", "sunday"),
        (
            "long_prefix",
            "common_prefix_different_ending_abc",
            "common_prefix_different_ending_xyz",
        ),
        (
            "long_identical",
            "this_is_identical_string",
            "this_is_identical_string",
        ),
    ];

    for (name, source, target) in test_cases {
        group.bench_with_input(
            BenchmarkId::new("optimized", name),
            &(source, target),
            |b, (s, t)| b.iter(|| standard_distance(black_box(s), black_box(t))),
        );
    }

    group.finish();
}

#[cfg(target_arch = "x86_64")]
criterion_group!(
    benches,
    bench_sse41_vs_avx2,
    bench_affix_stripping,
    bench_integrated_distance
);

#[cfg(not(target_arch = "x86_64"))]
criterion_group!(benches, bench_integrated_distance);

criterion_main!(benches);
