//! Criterion benchmarks for PhoneticNormalizedDictionary.
//!
//! This benchmark suite profiles the performance of the phonetic normalized dictionary,
//! measuring throughput and latency for:
//! - Dictionary construction from terms
//! - String normalization
//! - Query operations (exact and fuzzy)
//! - Insert operations
//! - Contains lookup
//!
//! Used as the baseline for scientific optimization of the PhoneticNormalizedDictionary.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::Dictionary;
use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedDictionary;

// ============================================================================
// Test Data
// ============================================================================

/// Generate a word list of common English words for benchmarking.
fn common_words() -> Vec<&'static str> {
    vec![
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on",
        "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we",
        "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
        "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into",
        "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now",
        "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
        "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any",
        "these", "give", "day", "most", "us",
    ]
}

/// Generate a larger word list by extending common words.
fn extended_words(count: usize) -> Vec<String> {
    let base = common_words();
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let base_word = base[i % base.len()];
        if i < base.len() {
            result.push(base_word.to_string());
        } else {
            // Generate variations: word, word1, word2, ...
            result.push(format!("{}{}", base_word, i / base.len()));
        }
    }

    result
}

/// Words specifically chosen for phonetic variation.
fn phonetic_words() -> Vec<&'static str> {
    vec![
        "phone", "fone", "elephant", "elefant", "knight", "night", "nite", "through", "thru",
        "threw", "color", "colour", "enough", "enuf", "cough", "rough", "tough", "accept",
        "except", "affect", "effect",
    ]
}

// ============================================================================
// Construction Benchmarks
// ============================================================================

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonetic_normalized_construction");

    // Small dictionary (100 terms)
    let small_words = extended_words(100);
    group.throughput(Throughput::Elements(100));
    group.bench_function(BenchmarkId::new("from_terms", "100"), |b| {
        b.iter(|| {
            let dict = PhoneticNormalizedDictionary::<()>::from_terms(black_box(&small_words));
            black_box(dict)
        });
    });

    // Medium dictionary (10K terms)
    let medium_words = extended_words(10_000);
    group.throughput(Throughput::Elements(10_000));
    group.bench_function(BenchmarkId::new("from_terms", "10k"), |b| {
        b.iter(|| {
            let dict = PhoneticNormalizedDictionary::<()>::from_terms(black_box(&medium_words));
            black_box(dict)
        });
    });

    // Large dictionary (100K terms)
    let large_words = extended_words(100_000);
    group.throughput(Throughput::Elements(100_000));
    group.bench_function(BenchmarkId::new("from_terms", "100k"), |b| {
        b.iter(|| {
            let dict = PhoneticNormalizedDictionary::<()>::from_terms(black_box(&large_words));
            black_box(dict)
        });
    });

    group.finish();
}

// ============================================================================
// Normalization Benchmarks
// ============================================================================

fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonetic_normalized_normalize");

    // Create dictionary once for normalize method access
    let dict = PhoneticNormalizedDictionary::<()>::from_terms(["test"]);

    // Short string (5 chars)
    let short = "phone";
    group.throughput(Throughput::Bytes(short.len() as u64));
    group.bench_function(BenchmarkId::new("normalize", "5_chars"), |b| {
        b.iter(|| black_box(dict.normalize(black_box(short))))
    });

    // Medium string (20 chars)
    let medium = "phonetic_alphabet_xy";
    group.throughput(Throughput::Bytes(medium.len() as u64));
    group.bench_function(BenchmarkId::new("normalize", "20_chars"), |b| {
        b.iter(|| black_box(dict.normalize(black_box(medium))))
    });

    // Long string (100 chars)
    let long = "a".repeat(50) + &"b".repeat(50);
    group.throughput(Throughput::Bytes(long.len() as u64));
    group.bench_function(BenchmarkId::new("normalize", "100_chars"), |b| {
        b.iter(|| black_box(dict.normalize(black_box(&long))))
    });

    // Phonetically interesting string
    let phonetic = "enough_through_knight";
    group.throughput(Throughput::Bytes(phonetic.len() as u64));
    group.bench_function(BenchmarkId::new("normalize", "phonetic"), |b| {
        b.iter(|| black_box(dict.normalize(black_box(phonetic))))
    });

    group.finish();
}

// ============================================================================
// Query Benchmarks
// ============================================================================

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonetic_normalized_query");

    // Create dictionary with phonetic words
    let words = phonetic_words();
    let dict = PhoneticNormalizedDictionary::<()>::from_terms(&words);

    // Exact match (distance = 0)
    group.bench_function("query_exact_hit", |b| {
        b.iter(|| {
            let results = dict.query(black_box("phone"), black_box(0));
            black_box(results)
        });
    });

    group.bench_function("query_exact_miss", |b| {
        b.iter(|| {
            let results = dict.query(black_box("xyz123"), black_box(0));
            black_box(results)
        });
    });

    // Fuzzy match (distance = 1)
    group.bench_function("query_distance_1", |b| {
        b.iter(|| {
            let results = dict.query(black_box("fone"), black_box(1));
            black_box(results)
        });
    });

    // Fuzzy match (distance = 2)
    group.bench_function("query_distance_2", |b| {
        b.iter(|| {
            let results = dict.query(black_box("elefant"), black_box(2));
            black_box(results)
        });
    });

    group.finish();

    // Query on larger dictionary
    let mut group = c.benchmark_group("phonetic_normalized_query_large");

    let large_words = extended_words(10_000);
    let large_dict = PhoneticNormalizedDictionary::<()>::from_terms(&large_words);

    group.bench_function("query_exact_10k", |b| {
        b.iter(|| {
            let results = large_dict.query(black_box("people"), black_box(0));
            black_box(results)
        });
    });

    group.bench_function("query_distance_1_10k", |b| {
        b.iter(|| {
            let results = large_dict.query(black_box("people"), black_box(1));
            black_box(results)
        });
    });

    group.bench_function("query_distance_2_10k", |b| {
        b.iter(|| {
            let results = large_dict.query(black_box("people"), black_box(2));
            black_box(results)
        });
    });

    group.finish();
}

// ============================================================================
// Mutation Benchmarks
// ============================================================================

fn bench_mutation(c: &mut Criterion) {
    let mut group = c.benchmark_group("phonetic_normalized_mutation");

    // Insert single term into empty dictionary
    group.bench_function("insert_single_empty", |b| {
        b.iter(|| {
            let dict = PhoneticNormalizedDictionary::<()>::new();
            black_box(dict.insert(black_box("testword")))
        });
    });

    // Insert into dictionary with 1000 terms
    let base_words = extended_words(1000);
    group.bench_function("insert_single_1k", |b| {
        b.iter_batched(
            || PhoneticNormalizedDictionary::<()>::from_terms(&base_words),
            |dict| black_box(dict.insert(black_box("newword"))),
            criterion::BatchSize::SmallInput,
        );
    });

    // Contains lookup
    let dict = PhoneticNormalizedDictionary::<()>::from_terms(phonetic_words());

    group.bench_function("contains_hit", |b| {
        b.iter(|| black_box(dict.contains(black_box("phone"))))
    });

    group.bench_function("contains_miss", |b| {
        b.iter(|| black_box(dict.contains(black_box("xyz123"))))
    });

    group.finish();
}

// ============================================================================
// Levenshtein Distance Benchmarks (internal function, tested via query)
// ============================================================================

fn bench_levenshtein_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("levenshtein_distance_via_query");

    // We can't directly benchmark the private levenshtein_distance function,
    // but we can measure query performance with different string lengths
    // to understand distance calculation overhead.

    // Create a dictionary with strings of various lengths
    let short_strings: Vec<String> = (0..100).map(|i| format!("w{}", i)).collect();
    let medium_strings: Vec<String> = (0..100).map(|i| format!("word{:05}", i)).collect();
    let long_strings: Vec<String> = (0..100).map(|i| format!("longword{:010}", i)).collect();

    // Short strings (3-4 chars)
    let short_dict = PhoneticNormalizedDictionary::<()>::from_terms(&short_strings);
    group.bench_function("distance_short_strings", |b| {
        b.iter(|| {
            let results = short_dict.query(black_box("w50"), black_box(1));
            black_box(results)
        });
    });

    // Medium strings (9-10 chars)
    let medium_dict = PhoneticNormalizedDictionary::<()>::from_terms(&medium_strings);
    group.bench_function("distance_medium_strings", |b| {
        b.iter(|| {
            let results = medium_dict.query(black_box("word00050"), black_box(1));
            black_box(results)
        });
    });

    // Long strings (18-20 chars)
    let long_dict = PhoneticNormalizedDictionary::<()>::from_terms(&long_strings);
    group.bench_function("distance_long_strings", |b| {
        b.iter(|| {
            let results = long_dict.query(black_box("longword0000000050"), black_box(1));
            black_box(results)
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_construction,
    bench_normalization,
    bench_query,
    bench_mutation,
    bench_levenshtein_distance,
);

criterion_main!(benches);
