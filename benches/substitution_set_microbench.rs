//! Micro-benchmarks for SubstitutionSet and SubstitutionSetChar operations
//!
//! This benchmark suite measures the raw performance of substitution set
//! operations in isolation to identify optimization opportunities.
//!
//! Benchmarks cover:
//! - contains() lookup with varying set sizes
//! - Hit vs miss ratio impact
//! - Insertion performance
//! - Preset builder initialization
//!
//! Run with: cargo bench --bench substitution_set_microbench

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::transducer::{SubstitutionSet, SubstitutionSetChar};
use rand::prelude::*;
use std::hint::black_box;

/// Generate random byte pairs for testing
fn generate_byte_pairs(count: usize, seed: u64) -> Vec<(u8, u8)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let a = rng.gen_range(b'a'..=b'z');
            let b = rng.gen_range(b'a'..=b'z');
            (a, b)
        })
        .collect()
}

/// Generate random char pairs for testing
fn generate_char_pairs(count: usize, seed: u64) -> Vec<(char, char)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzαβγδεζηθικλμνξοπρστυφχψω"
        .chars()
        .collect();
    (0..count)
        .map(|_| {
            let a = chars[rng.gen_range(0..chars.len())];
            let b = chars[rng.gen_range(0..chars.len())];
            (a, b)
        })
        .collect()
}

/// Benchmark: contains() with varying set sizes (byte-level)
fn bench_contains_varying_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_set/contains/size");

    for size in [1, 5, 10, 20, 50, 100, 200, 500].iter() {
        let pairs = generate_byte_pairs(*size, 42);
        let mut set = SubstitutionSet::new();
        for &(a, b) in &pairs {
            set.allow_byte(a, b);
        }

        // Test queries: mix of hits and misses
        let test_queries: Vec<(u8, u8)> = (0..100)
            .map(|i| {
                if i < 50 {
                    // 50% hits - use existing pairs
                    pairs[i % pairs.len()]
                } else {
                    // 50% misses - use non-existing pairs
                    (b'A' + (i as u8), b'Z')
                }
            })
            .collect();

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                for &(a, c) in &test_queries {
                    black_box(set.contains(black_box(a), black_box(c)));
                }
            });
        });
    }
    group.finish();
}

/// Benchmark: contains() with varying hit rates
fn bench_contains_hit_rates(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_set/contains/hit_rate");

    let size = 50;
    let pairs = generate_byte_pairs(size, 42);
    let mut set = SubstitutionSet::new();
    for &(a, b) in &pairs {
        set.allow_byte(a, b);
    }

    for hit_rate in [10, 50, 90].iter() {
        let test_queries: Vec<(u8, u8)> = (0..1000)
            .map(|i| {
                if (i % 100) < *hit_rate {
                    // Hit - use existing pair
                    pairs[i % pairs.len()]
                } else {
                    // Miss - use non-existing pair
                    (b'A' + ((i % 26) as u8), b'Z')
                }
            })
            .collect();

        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("hit_rate", format!("{}%", hit_rate)),
            hit_rate,
            |b, _| {
                b.iter(|| {
                    for &(a, c) in &test_queries {
                        black_box(set.contains(black_box(a), black_box(c)));
                    }
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Insertion performance (allow_byte)
fn bench_insertion_byte(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_set/insertion/byte");

    for size in [10, 50, 100, 500].iter() {
        let pairs = generate_byte_pairs(*size, 42);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut set = SubstitutionSet::new();
                for &(a, c) in &pairs {
                    set.allow_byte(black_box(a), black_box(c));
                }
                black_box(set)
            });
        });
    }
    group.finish();
}

/// Benchmark: Insertion performance (allow for char)
fn bench_insertion_char(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_set/insertion/char");

    for size in [10, 50, 100, 500].iter() {
        let pairs = generate_char_pairs(*size, 42);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut set = SubstitutionSetChar::new();
                for &(a, c) in &pairs {
                    set.allow(black_box(a), black_box(c));
                }
                black_box(set)
            });
        });
    }
    group.finish();
}

/// Benchmark: Preset builder initialization
fn bench_preset_builders_byte(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_set/presets/byte");

    group.bench_function("phonetic_basic", |b| {
        b.iter(|| black_box(SubstitutionSet::phonetic_basic()));
    });

    group.bench_function("keyboard_qwerty", |b| {
        b.iter(|| black_box(SubstitutionSet::keyboard_qwerty()));
    });

    group.bench_function("leet_speak", |b| {
        b.iter(|| black_box(SubstitutionSet::leet_speak()));
    });

    group.bench_function("ocr_friendly", |b| {
        b.iter(|| black_box(SubstitutionSet::ocr_friendly()));
    });

    group.finish();
}

/// Benchmark: allow_str() for ASCII and UTF-8 string substitutions
fn bench_allow_str_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_set/allow_str/encoding");

    for (name, source, target) in [
        ("ascii_multi_to_single", "ph", "f"),
        ("utf8_single_scalar", "α", "β"),
        ("utf8_cjk", "你", "好"),
        ("ascii_to_utf8", "sch", "š"),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut set = SubstitutionSet::new();
                set.allow_str(black_box(source), black_box(target));
                black_box(set)
            });
        });
    }

    group.finish();
}

/// Benchmark: contains_str() for ASCII and UTF-8 string substitutions
fn bench_contains_str_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_set/contains_str/encoding");

    let mut set = SubstitutionSet::new();
    set.allow_str("ph", "f");
    set.allow_str("α", "β");
    set.allow_str("你", "好");
    set.allow_str("sch", "š");

    for (name, source, target) in [
        ("ascii_multi_hit", "ph", "f"),
        ("utf8_single_scalar_hit", "α", "β"),
        ("utf8_cjk_hit", "你", "好"),
        ("ascii_to_utf8_hit", "sch", "š"),
        ("ascii_miss", "ph", "v"),
        ("utf8_miss", "α", "γ"),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(
                    set.contains_str(black_box(source.as_bytes()), black_box(target.as_bytes())),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark: Preset builder initialization for char
fn bench_preset_builders_char(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_set/presets/char");

    group.bench_function("diacritics_latin", |b| {
        b.iter(|| black_box(SubstitutionSetChar::diacritics_latin()));
    });

    group.bench_function("greek_case_insensitive", |b| {
        b.iter(|| black_box(SubstitutionSetChar::greek_case_insensitive()));
    });

    group.bench_function("cyrillic_case_insensitive", |b| {
        b.iter(|| black_box(SubstitutionSetChar::cyrillic_case_insensitive()));
    });

    group.bench_function("japanese_hiragana_katakana", |b| {
        b.iter(|| black_box(SubstitutionSetChar::japanese_hiragana_katakana()));
    });

    group.finish();
}

/// Benchmark: contains() for char (Unicode)
fn bench_contains_char_varying_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_set_char/contains/size");

    for size in [1, 5, 10, 20, 50, 100].iter() {
        let pairs = generate_char_pairs(*size, 42);
        let mut set = SubstitutionSetChar::new();
        for &(a, b) in &pairs {
            set.allow(a, b);
        }

        // Test queries: mix of hits and misses
        let test_queries: Vec<(char, char)> = (0..100)
            .map(|i| {
                if i < 50 {
                    pairs[i % pairs.len()]
                } else {
                    ('X', 'Y') // Misses
                }
            })
            .collect();

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                for &(a, c) in &test_queries {
                    black_box(set.contains(black_box(a), black_box(c)));
                }
            });
        });
    }
    group.finish();
}

/// Benchmark: Single lookup (for profiling)
fn bench_single_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("substitution_set/single_lookup");

    let pairs = generate_byte_pairs(50, 42);
    let mut set = SubstitutionSet::new();
    for &(a, b) in &pairs {
        set.allow_byte(a, b);
    }

    group.bench_function("hit", |b| {
        let (a, c) = pairs[0];
        b.iter(|| black_box(set.contains(black_box(a), black_box(c))));
    });

    group.bench_function("miss", |b| {
        b.iter(|| black_box(set.contains(black_box(b'X'), black_box(b'Y'))));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_contains_varying_sizes,
    bench_contains_hit_rates,
    bench_insertion_byte,
    bench_insertion_char,
    bench_preset_builders_byte,
    bench_allow_str_encoding,
    bench_contains_str_encoding,
    bench_preset_builders_char,
    bench_contains_char_varying_sizes,
    bench_single_lookup,
);

criterion_main!(benches);
