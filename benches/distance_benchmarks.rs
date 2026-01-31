//! Comprehensive benchmarks for distance functions.
//!
//! Tests various scenarios:
//! - String length variations (short, medium, long)
//! - Similarity patterns (identical, similar, different)
//! - Character sets (ASCII, Unicode)
//! - Cache behavior (cold, warm)
//! - Iterative vs recursive implementations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::distance::{*, myers};

// ============================================================================
// Test Data Generation
// ============================================================================

fn generate_test_pairs() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        // (name, source, target)
        // Short strings (0-10 chars)
        ("empty", "", ""),
        ("short_identical", "test", "test"),
        ("short_1edit", "test", "best"),
        ("short_2edit", "test", "cast"),
        ("short_different", "abc", "xyz"),
        // Medium strings (10-50 chars)
        ("medium_identical", "programming", "programming"),
        ("medium_similar", "programming", "programing"), // 1 deletion
        ("medium_prefix", "commonprefix_abc", "commonprefix_xyz"),
        ("medium_different", "completely", "different"),
        // Long strings (50-200 chars)
        (
            "long_identical",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        ),
        (
            "long_similar",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
        ),
        (
            "long_prefix",
            "In the beginning was the Word and the Word was with God",
            "In the beginning was the Word and the Word was with Bob",
        ),
        (
            "long_different",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump",
        ),
        // Unicode strings
        ("unicode_short", "café", "cafe"),
        ("unicode_japanese", "日本語", "日本語"),
        ("unicode_mixed", "Hello 世界", "Hello World"),
        // Transposition cases
        ("transposition_simple", "ab", "ba"),
        ("transposition_word", "test", "tset"),
        // Common prefix/suffix scenarios
        (
            "long_common_prefix",
            "this_is_a_very_long_common_prefix_abc",
            "this_is_a_very_long_common_prefix_xyz",
        ),
        (
            "long_common_suffix",
            "abc_this_is_a_very_long_common_suffix",
            "xyz_this_is_a_very_long_common_suffix",
        ),
    ]
}

// ============================================================================
// Standard Distance Benchmarks
// ============================================================================

fn bench_standard_distance_iterative(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_distance/iterative");

    for (name, source, target) in generate_test_pairs() {
        let size = source.len() + target.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(source, target),
            |b, &(s, t)| {
                b.iter(|| standard_distance(black_box(s), black_box(t)));
            },
        );
    }

    group.finish();
}

fn bench_standard_distance_recursive(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_distance/recursive");

    for (name, source, target) in generate_test_pairs() {
        let size = source.len() + target.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(source, target),
            |b, &(s, t)| {
                let cache = create_memo_cache();
                b.iter(|| standard_distance_recursive(black_box(s), black_box(t), &cache));
            },
        );
    }

    group.finish();
}

fn bench_standard_distance_recursive_warm_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_distance/recursive_warm_cache");

    for (name, source, target) in generate_test_pairs() {
        let size = source.len() + target.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(source, target),
            |b, &(s, t)| {
                let cache = create_memo_cache();
                // Warm up the cache
                let _ = standard_distance_recursive(s, t, &cache);

                b.iter(|| standard_distance_recursive(black_box(s), black_box(t), &cache));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Transposition Distance Benchmarks
// ============================================================================

fn bench_transposition_distance_iterative(c: &mut Criterion) {
    let mut group = c.benchmark_group("transposition_distance/iterative");

    for (name, source, target) in generate_test_pairs() {
        let size = source.len() + target.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(source, target),
            |b, &(s, t)| {
                b.iter(|| transposition_distance(black_box(s), black_box(t)));
            },
        );
    }

    group.finish();
}

fn bench_transposition_distance_recursive(c: &mut Criterion) {
    let mut group = c.benchmark_group("transposition_distance/recursive");

    for (name, source, target) in generate_test_pairs() {
        let size = source.len() + target.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(source, target),
            |b, &(s, t)| {
                let cache = create_memo_cache();
                b.iter(|| transposition_distance_recursive(black_box(s), black_box(t), &cache));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Merge and Split Distance Benchmarks
// ============================================================================

fn bench_merge_split_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_split_distance");

    for (name, source, target) in generate_test_pairs() {
        let size = source.len() + target.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(source, target),
            |b, &(s, t)| {
                let cache = create_memo_cache();
                b.iter(|| merge_and_split_distance(black_box(s), black_box(t), &cache));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Comparison Benchmarks
// ============================================================================

fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");

    let test_cases = vec![
        ("short", "test", "best"),
        ("medium", "programming", "programing"),
        (
            "long",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
        ),
    ];

    for (name, source, target) in test_cases {
        group.bench_function(format!("{}/standard_iterative", name), |b| {
            b.iter(|| standard_distance(black_box(source), black_box(target)));
        });

        group.bench_function(format!("{}/standard_recursive", name), |b| {
            let cache = create_memo_cache();
            b.iter(|| standard_distance_recursive(black_box(source), black_box(target), &cache));
        });

        group.bench_function(format!("{}/transposition_iterative", name), |b| {
            b.iter(|| transposition_distance(black_box(source), black_box(target)));
        });

        group.bench_function(format!("{}/transposition_recursive", name), |b| {
            let cache = create_memo_cache();
            b.iter(|| {
                transposition_distance_recursive(black_box(source), black_box(target), &cache)
            });
        });

        group.bench_function(format!("{}/merge_split", name), |b| {
            let cache = create_memo_cache();
            b.iter(|| merge_and_split_distance(black_box(source), black_box(target), &cache));
        });
    }

    group.finish();
}

// ============================================================================
// Scaling Benchmarks
// ============================================================================

fn bench_string_length_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/string_length");

    // Generate strings of increasing length
    let lengths = vec![10, 20, 50, 100, 200];

    for len in lengths {
        let source: String = "a".repeat(len);
        let target: String = "b".repeat(len);

        group.throughput(Throughput::Elements(len as u64));

        group.bench_with_input(
            BenchmarkId::new("iterative", len),
            &(&source, &target),
            |b, (s, t)| {
                b.iter(|| standard_distance(black_box(s), black_box(t)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("recursive", len),
            &(&source, &target),
            |b, (s, t)| {
                let cache = create_memo_cache();
                b.iter(|| standard_distance_recursive(black_box(s), black_box(t), &cache));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Cache Performance Benchmarks
// ============================================================================

fn bench_cache_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache/effectiveness");

    let strings = vec!["test", "best", "rest", "fest", "nest"];

    // Benchmark with fresh cache for each operation
    group.bench_function("cold_cache", |b| {
        b.iter(|| {
            let cache = create_memo_cache();
            for &s1 in &strings {
                for &s2 in &strings {
                    standard_distance_recursive(black_box(s1), black_box(s2), &cache);
                }
            }
        });
    });

    // Benchmark with shared cache across all operations
    group.bench_function("warm_cache", |b| {
        let cache = create_memo_cache();
        // Warm up
        for &s1 in &strings {
            for &s2 in &strings {
                standard_distance_recursive(s1, s2, &cache);
            }
        }

        b.iter(|| {
            for &s1 in &strings {
                for &s2 in &strings {
                    standard_distance_recursive(black_box(s1), black_box(s2), &cache);
                }
            }
        });
    });

    group.finish();
}

// ============================================================================
// Unicode Performance Benchmarks
// ============================================================================

fn bench_unicode_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("unicode");

    let test_cases = vec![
        ("ascii", "hello world", "helo world"),
        ("latin", "café résumé", "cafe resume"),
        ("japanese", "こんにちは世界", "こんにちわ世界"),
        ("emoji", "Hello 👋 World 🌍", "Hello 👋 World 🌎"),
        ("mixed", "Hello世界🌍", "Hello世界🌎"),
    ];

    for (name, source, target) in test_cases {
        group.bench_function(format!("standard_iterative/{}", name), |b| {
            b.iter(|| standard_distance(black_box(source), black_box(target)));
        });

        group.bench_function(format!("standard_recursive/{}", name), |b| {
            let cache = create_memo_cache();
            b.iter(|| standard_distance_recursive(black_box(source), black_box(target), &cache));
        });
    }

    group.finish();
}

// ============================================================================
// Myers Bit-Parallel Benchmarks
// ============================================================================

fn bench_myers_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("myers_distance");

    // Test data for Myers (short ASCII strings where it excels)
    let test_cases = vec![
        ("empty", "", ""),
        ("short_identical", "test", "test"),
        ("short_1edit", "test", "best"),
        ("short_2edit", "test", "cast"),
        ("short_different", "abc", "xyz"),
        ("medium_identical", "programming", "programming"),
        ("medium_similar", "programming", "programing"),
        ("classic_kitten", "kitten", "sitting"),
        ("classic_saturday", "saturday", "sunday"),
        ("long_32", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
    ];

    for (name, source, target) in test_cases {
        let size = source.len() + target.len();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(source, target),
            |b, &(s, t)| {
                b.iter(|| myers::myers_distance(black_box(s), black_box(t)));
            },
        );
    }

    group.finish();
}

fn bench_myers_vs_dp(c: &mut Criterion) {
    let mut group = c.benchmark_group("myers_vs_dp");

    // Pre-generate strings for ownership
    let boundary_a = "a".repeat(32);
    let boundary_b = "b".repeat(32);

    // Compare Myers against scalar DP for short strings
    let test_cases: Vec<(&str, &str, &str)> = vec![
        ("short_8", "kitten", "sitting"),
        ("medium_16", "acknowledgement", "acknowledgment"),
        ("boundary_64", &boundary_a, &boundary_b),
    ];

    for (name, source, target) in test_cases {
        group.bench_function(format!("{}/myers", name), |b| {
            b.iter(|| myers::myers_distance(black_box(source), black_box(target)));
        });

        group.bench_function(format!("{}/dp_scalar", name), |b| {
            b.iter(|| standard_distance_impl(black_box(source), black_box(target)));
        });
    }

    group.finish();
}

fn bench_myers_bounded(c: &mut Criterion) {
    let mut group = c.benchmark_group("myers_bounded");

    let test_cases = vec![
        ("close_match", "test", "best", 2),
        ("far_match", "abc", "xyz", 3),
        ("exact_threshold", "kitten", "sitting", 3),
    ];

    for (name, source, target, threshold) in test_cases {
        group.bench_function(name, |b| {
            b.iter(|| myers::myers_distance_bounded(black_box(source), black_box(target), threshold));
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_standard_distance_iterative,
    bench_standard_distance_recursive,
    bench_standard_distance_recursive_warm_cache,
    bench_transposition_distance_iterative,
    bench_transposition_distance_recursive,
    bench_merge_split_distance,
    bench_algorithm_comparison,
    bench_string_length_scaling,
    bench_cache_effectiveness,
    bench_unicode_performance,
    bench_myers_distance,
    bench_myers_vs_dp,
    bench_myers_bounded,
);

criterion_main!(benches);
