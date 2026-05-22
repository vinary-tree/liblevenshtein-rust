//! Criterion benchmarks for phonetic rewrite rules.
//!
//! This benchmark suite profiles the performance of the phonetic rewrite system,
//! measuring throughput and latency for:
//! - Single rule application
//! - Sequential rule application
//! - Different rule set sizes
//! - Byte-level vs character-level performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::phonetic::*;

// ============================================================================
// Benchmark Fixtures
// ============================================================================

fn sample_phonetic_strings() -> Vec<Vec<Phone>> {
    vec![
        // Short strings
        vec![
            Phone::Consonant(b'c'),
            Phone::Vowel(b'a'),
            Phone::Consonant(b't'),
        ],
        vec![
            Phone::Consonant(b'p'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'o'),
        ],
        // Medium strings
        vec![
            Phone::Consonant(b'c'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'u'),
            Phone::Consonant(b'r'),
            Phone::Consonant(b'c'),
            Phone::Consonant(b'h'),
        ],
        // Longer strings
        vec![
            Phone::Consonant(b'p'),
            Phone::Consonant(b'h'),
            Phone::Vowel(b'o'),
            Phone::Consonant(b'n'),
            Phone::Vowel(b'e'),
            Phone::Consonant(b't'),
            Phone::Vowel(b'i'),
            Phone::Consonant(b'c'),
        ],
    ]
}

// ============================================================================
// Single Rule Application Benchmarks
// ============================================================================

fn bench_single_rule_application(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_rule_application");

    let ortho_rules = orthography_rules();
    let test_strings = sample_phonetic_strings();

    for (i, rule) in ortho_rules.iter().enumerate() {
        group.bench_function(BenchmarkId::new("rule", i), |b| {
            b.iter(|| {
                for s in &test_strings {
                    for pos in 0..=s.len() {
                        black_box(apply_rule_at(black_box(rule), black_box(s), black_box(pos)));
                    }
                }
            });
        });
    }

    group.finish();
}

// ============================================================================
// Sequential Rule Application Benchmarks
// ============================================================================

fn bench_sequential_application(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_application");

    let ortho_rules = orthography_rules();
    let phon_rules = phonetic_rules();
    let all_rules = zompist_rules();

    let test_string = vec![
        Phone::Consonant(b'p'),
        Phone::Consonant(b'h'),
        Phone::Vowel(b'o'),
        Phone::Consonant(b'n'),
        Phone::Vowel(b'e'),
        Phone::Consonant(b't'),
        Phone::Vowel(b'i'),
        Phone::Consonant(b'c'),
    ];

    let fuel = test_string.len() * all_rules.len() * MAX_EXPANSION_FACTOR;

    group.bench_function("orthography_rules", |b| {
        b.iter(|| {
            apply_rules_seq(
                black_box(&ortho_rules),
                black_box(&test_string),
                black_box(fuel),
            )
        });
    });

    group.bench_function("phonetic_rules", |b| {
        b.iter(|| {
            apply_rules_seq(
                black_box(&phon_rules),
                black_box(&test_string),
                black_box(fuel),
            )
        });
    });

    group.bench_function("all_zompist_rules", |b| {
        b.iter(|| {
            apply_rules_seq(
                black_box(&all_rules),
                black_box(&test_string),
                black_box(fuel),
            )
        });
    });

    group.finish();
}

// ============================================================================
// Throughput Benchmarks (varying input size)
// ============================================================================

fn bench_throughput_by_input_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_by_input_size");

    let rules = orthography_rules();

    for size in [5, 10, 20, 50].iter() {
        let input: Vec<Phone> = (0..*size)
            .map(|i| {
                if i % 2 == 0 {
                    Phone::Consonant(b'c')
                } else {
                    Phone::Vowel(b'a')
                }
            })
            .collect();

        let fuel = input.len() * rules.len() * MAX_EXPANSION_FACTOR;

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| apply_rules_seq(black_box(&rules), black_box(&input), black_box(fuel)));
        });
    }

    group.finish();
}

// ============================================================================
// Fuel Variation Benchmarks
// ============================================================================

fn bench_fuel_variation(c: &mut Criterion) {
    let mut group = c.benchmark_group("fuel_variation");

    let rules = orthography_rules();
    let test_string = vec![
        Phone::Consonant(b'c'),
        Phone::Consonant(b'h'),
        Phone::Vowel(b'u'),
        Phone::Consonant(b'r'),
        Phone::Consonant(b'c'),
        Phone::Consonant(b'h'),
    ];

    for fuel in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(fuel), fuel, |b, &fuel| {
            b.iter(|| apply_rules_seq(black_box(&rules), black_box(&test_string), black_box(fuel)));
        });
    }

    group.finish();
}

// ============================================================================
// Pattern Matching Benchmarks
// ============================================================================

fn bench_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");

    let patterns = vec![
        vec![Phone::Consonant(b'c')],
        vec![Phone::Consonant(b'c'), Phone::Consonant(b'h')],
        vec![Phone::Consonant(b'p'), Phone::Consonant(b'h')],
    ];

    let test_string = vec![
        Phone::Consonant(b'c'),
        Phone::Consonant(b'h'),
        Phone::Vowel(b'u'),
        Phone::Consonant(b'r'),
        Phone::Consonant(b'c'),
        Phone::Consonant(b'h'),
    ];

    for (i, pattern) in patterns.iter().enumerate() {
        group.bench_function(BenchmarkId::new("pattern", i), |b| {
            b.iter(|| {
                for pos in 0..=test_string.len() {
                    black_box(pattern_matches_at(
                        black_box(pattern),
                        black_box(&test_string),
                        black_box(pos),
                    ));
                }
            });
        });
    }

    group.finish();
}

// ============================================================================
// Context Matching Benchmarks
// ============================================================================

fn bench_context_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_matching");

    let contexts = vec![
        Context::Initial,
        Context::Final,
        Context::Anywhere,
        Context::BeforeVowel(vec![b'a', b'e', b'i', b'o', b'u']),
        Context::AfterConsonant(vec![b'c', b'k', b'p']),
    ];

    let test_string = vec![
        Phone::Consonant(b'c'),
        Phone::Vowel(b'a'),
        Phone::Consonant(b't'),
    ];

    for (i, context) in contexts.iter().enumerate() {
        group.bench_function(BenchmarkId::new("context", i), |b| {
            b.iter(|| {
                for pos in 0..=test_string.len() {
                    black_box(context_matches(
                        black_box(context),
                        black_box(&test_string),
                        black_box(pos),
                    ));
                }
            });
        });
    }

    group.finish();
}

// ============================================================================
// Byte-level vs Character-level Comparison
// ============================================================================

fn bench_u8_vs_char(c: &mut Criterion) {
    let mut group = c.benchmark_group("u8_vs_char_comparison");

    let ortho_rules = orthography_rules();
    let ortho_rules_char = orthography_rules_char();

    let test_string_u8 = vec![
        Phone::Consonant(b'p'),
        Phone::Consonant(b'h'),
        Phone::Vowel(b'o'),
        Phone::Consonant(b'n'),
        Phone::Vowel(b'e'),
    ];

    let test_string_char = vec![
        PhoneChar::Consonant('p'),
        PhoneChar::Consonant('h'),
        PhoneChar::Vowel('o'),
        PhoneChar::Consonant('n'),
        PhoneChar::Vowel('e'),
    ];

    let fuel = test_string_u8.len() * ortho_rules.len() * MAX_EXPANSION_FACTOR;

    group.bench_function("u8_byte_level", |b| {
        b.iter(|| {
            apply_rules_seq(
                black_box(&ortho_rules),
                black_box(&test_string_u8),
                black_box(fuel),
            )
        });
    });

    group.bench_function("char_unicode_level", |b| {
        b.iter(|| {
            apply_rules_seq_char(
                black_box(&ortho_rules_char),
                black_box(&test_string_char),
                black_box(fuel),
            )
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_single_rule_application,
    bench_sequential_application,
    bench_throughput_by_input_size,
    bench_fuel_variation,
    bench_pattern_matching,
    bench_context_matching,
    bench_u8_vs_char,
);

criterion_main!(benches);
