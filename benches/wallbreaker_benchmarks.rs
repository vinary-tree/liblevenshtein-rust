//! Comprehensive benchmarks for WallBreaker algorithm.
//!
//! Compares WallBreaker (SCDAWG + pigeonhole) vs traditional Levenshtein automata.
//!
//! Run with:
//!   cargo bench --bench wallbreaker_benchmarks
//!
//! Save baseline:
//!   cargo bench --bench wallbreaker_benchmarks -- --save-baseline wallbreaker-baseline
//!
//! Compare to baseline:
//!   cargo bench --bench wallbreaker_benchmarks -- --baseline wallbreaker-baseline

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::scdawg::Scdawg;
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::Algorithm;
use liblevenshtein::wallbreaker::WallBreaker;
use std::collections::HashSet;

// ============================================================================
// Test Data Setup
// ============================================================================

/// Load dictionary from system words file with configurable size
fn load_dictionary(target_size: usize) -> Vec<String> {
    if let Ok(contents) = std::fs::read_to_string("/usr/share/dict/words") {
        contents
            .lines()
            .map(|s| s.trim().to_lowercase())
            .filter(|s| s.len() >= 3 && s.len() <= 20 && s.chars().all(|c| c.is_ascii_alphabetic()))
            .take(target_size)
            .collect()
    } else {
        // Fallback: generate synthetic dictionary
        generate_synthetic_dictionary(target_size)
    }
}

/// Generate synthetic dictionary for systems without /usr/share/dict/words
fn generate_synthetic_dictionary(size: usize) -> Vec<String> {
    let base_words = [
        "algorithm", "structure", "computer", "science", "program",
        "function", "variable", "constant", "iterator", "reference",
        "pattern", "matching", "distance", "automaton", "transducer",
        "dictionary", "benchmark", "performance", "optimization", "implementation",
        "cathedral", "category", "catering", "catastrophe", "catalyst",
        "application", "approximation", "acceleration", "authentication", "authorization",
    ];

    let suffixes = ["", "s", "ed", "ing", "er", "est", "ly", "tion", "ment", "ness"];
    let prefixes = ["", "un", "re", "pre", "mis", "dis", "over", "under", "out", "sub"];

    let mut words = HashSet::new();

    for base in &base_words {
        for prefix in &prefixes {
            for suffix in &suffixes {
                let word = format!("{}{}{}", prefix, base, suffix);
                if word.len() >= 3 && word.len() <= 20 {
                    words.insert(word);
                }
                if words.len() >= size {
                    return words.into_iter().collect();
                }
            }
        }
    }

    // If still need more words, generate numbered variants
    let mut i = 0;
    while words.len() < size {
        words.insert(format!("word{:06}", i));
        i += 1;
    }

    words.into_iter().collect()
}

/// Generate a query of specified length from random characters
fn generate_query(length: usize, seed: usize) -> String {
    // Use deterministic pseudo-random based on seed for reproducibility
    let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
    (0..length)
        .map(|i| {
            let idx = (seed * 31 + i * 17) % chars.len();
            chars[idx]
        })
        .collect()
}

/// Generate a query based on a dictionary word with some modifications
fn generate_realistic_query(dict: &[String], seed: usize, target_len: usize) -> String {
    if dict.is_empty() {
        return generate_query(target_len, seed);
    }

    // Pick a word from dictionary
    let word = &dict[seed % dict.len()];

    // Modify it to be approximately target_len
    if word.len() >= target_len {
        word.chars().take(target_len).collect()
    } else {
        // Extend with random chars
        let mut query: String = word.clone();
        let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
        while query.len() < target_len {
            let idx = (seed * 31 + query.len() * 17) % chars.len();
            query.push(chars[idx]);
        }
        query
    }
}

// ============================================================================
// WallBreaker Benchmarks
// ============================================================================

/// Benchmark WallBreaker query performance across different configurations
fn bench_wallbreaker_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("wallbreaker_query");
    group.sample_size(50); // Ensure sufficient samples for statistical significance

    // Test configurations: (dict_size, max_distance, query_len)
    let configs = [
        // Small dictionary
        (1_000, 2, 10),
        (1_000, 2, 20),
        (1_000, 4, 10),
        (1_000, 4, 20),
        // Medium dictionary
        (10_000, 2, 20),
        (10_000, 4, 20),
        (10_000, 4, 50),
        (10_000, 8, 50),
        // Large dictionary - wall effect regime
        (100_000, 2, 20),
        (100_000, 4, 50),
        (100_000, 8, 50),
        (100_000, 8, 100),
        (100_000, 16, 100),
    ];

    for (dict_size, max_distance, query_len) in configs {
        let dict_words = load_dictionary(dict_size);
        let actual_dict_size = dict_words.len();

        if actual_dict_size < dict_size / 2 {
            eprintln!("Warning: Only loaded {} words for target {}", actual_dict_size, dict_size);
            continue;
        }

        // Build SCDAWG
        let scdawg = Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
        let wb = WallBreaker::new(&scdawg, max_distance);

        // Generate queries
        let queries: Vec<String> = (0..10)
            .map(|i| generate_realistic_query(&dict_words, i * 7919, query_len))
            .collect();

        let id = format!("d{}_k{}_q{}", actual_dict_size, max_distance, query_len);

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_function(BenchmarkId::new("wallbreaker", &id), |b| {
            b.iter(|| {
                let mut total = 0usize;
                for query in &queries {
                    total += black_box(wb.query(query).count());
                }
                total
            })
        });
    }

    group.finish();
}

/// Benchmark traditional Levenshtein transducer for comparison
fn bench_traditional_transducer(c: &mut Criterion) {
    let mut group = c.benchmark_group("traditional_transducer");
    group.sample_size(50);

    // Same configurations as WallBreaker for fair comparison
    let configs = [
        (1_000, 2, 10),
        (1_000, 2, 20),
        (1_000, 4, 10),
        (1_000, 4, 20),
        (10_000, 2, 20),
        (10_000, 4, 20),
        (10_000, 4, 50),
        (10_000, 8, 50),
        // Note: Large dict with high distance may be slow
        (100_000, 2, 20),
        (100_000, 4, 50),
    ];

    for (dict_size, max_distance, query_len) in configs {
        let dict_words = load_dictionary(dict_size);
        let actual_dict_size = dict_words.len();

        if actual_dict_size < dict_size / 2 {
            continue;
        }

        // Build traditional DAWG
        let dawg = DynamicDawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));

        // Generate same queries
        let queries: Vec<String> = (0..10)
            .map(|i| generate_realistic_query(&dict_words, i * 7919, query_len))
            .collect();

        let id = format!("d{}_k{}_q{}", actual_dict_size, max_distance, query_len);

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_function(BenchmarkId::new("transducer", &id), |b| {
            let transducer = Transducer::new(dawg.clone(), Algorithm::Standard);
            b.iter(|| {
                let mut total = 0usize;
                for query in &queries {
                    total += black_box(transducer.query(query, max_distance).count());
                }
                total
            })
        });
    }

    group.finish();
}

/// Benchmark SCDAWG construction time
fn bench_scdawg_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("scdawg_construction");
    group.sample_size(30);

    for dict_size in [1_000, 10_000, 100_000] {
        let dict_words = load_dictionary(dict_size);
        let actual_size = dict_words.len();

        if actual_size < dict_size / 2 {
            continue;
        }

        let id = format!("d{}", actual_size);

        group.bench_function(BenchmarkId::new("scdawg", &id), |b| {
            b.iter(|| {
                black_box(Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str())))
            })
        });

        // Also benchmark traditional DAWG construction for comparison
        group.bench_function(BenchmarkId::new("dynamic_dawg", &id), |b| {
            b.iter(|| {
                black_box(DynamicDawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str())))
            })
        });
    }

    group.finish();
}

/// Benchmark substring search specifically (isolate for Phase 2 optimization)
fn bench_substring_search(c: &mut Criterion) {
    use libdictenstein::substring::SubstringDictionary;

    let mut group = c.benchmark_group("substring_search");
    group.sample_size(100);

    let dict_words = load_dictionary(50_000);
    let scdawg = Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));

    // Test different pattern lengths
    for pattern_len in [5, 10, 15, 20] {
        let patterns: Vec<String> = (0..20)
            .map(|i| generate_query(pattern_len, i * 1009))
            .collect();

        let id = format!("len{}", pattern_len);

        group.bench_function(BenchmarkId::new("find_substring", &id), |b| {
            b.iter(|| {
                let mut total = 0usize;
                for pattern in &patterns {
                    total += black_box(scdawg.find_exact_substring(pattern).len());
                }
                total
            })
        });
    }

    group.finish();
}

/// Benchmark pattern splitting (isolate for Phase 3 optimization)
fn bench_pattern_splitting(c: &mut Criterion) {
    use liblevenshtein::wallbreaker::PatternSplitter;

    let mut group = c.benchmark_group("pattern_splitting");
    group.sample_size(1000);

    for max_distance in [2, 4, 8, 16] {
        let splitter = PatternSplitter::new(max_distance);

        // Test different query lengths
        for query_len in [20, 50, 100] {
            let queries: Vec<String> = (0..100)
                .map(|i| generate_query(query_len, i * 997))
                .collect();

            let id = format!("k{}_q{}", max_distance, query_len);

            group.bench_function(BenchmarkId::new("split", &id), |b| {
                b.iter(|| {
                    let mut total = 0usize;
                    for query in &queries {
                        total += black_box(splitter.split(query).len());
                    }
                    total
                })
            });
        }
    }

    group.finish();
}

/// Compare WallBreaker vs Traditional directly in same benchmark group
fn bench_wallbreaker_vs_traditional(c: &mut Criterion) {
    let mut group = c.benchmark_group("wallbreaker_vs_traditional");
    group.sample_size(50);

    // Focus on configurations where WallBreaker should excel (high distance, long queries)
    let configs = [
        (10_000, 4, 50, "medium_d4_q50"),
        (10_000, 8, 50, "medium_d8_q50"),
        (50_000, 4, 50, "large_d4_q50"),
        (50_000, 8, 100, "large_d8_q100"),
    ];

    for (dict_size, max_distance, query_len, label) in configs {
        let dict_words = load_dictionary(dict_size);
        let actual_size = dict_words.len();

        if actual_size < dict_size / 2 {
            continue;
        }

        // Build both data structures
        let scdawg = Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
        let dawg = DynamicDawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));

        let wb = WallBreaker::new(&scdawg, max_distance);

        let queries: Vec<String> = (0..10)
            .map(|i| generate_realistic_query(&dict_words, i * 7919, query_len))
            .collect();

        group.throughput(Throughput::Elements(queries.len() as u64));

        group.bench_function(BenchmarkId::new("wallbreaker", label), |b| {
            b.iter(|| {
                let mut total = 0usize;
                for query in &queries {
                    total += black_box(wb.query(query).count());
                }
                total
            })
        });

        group.bench_function(BenchmarkId::new("transducer", label), |b| {
            let transducer = Transducer::new(dawg.clone(), Algorithm::Standard);
            b.iter(|| {
                let mut total = 0usize;
                for query in &queries {
                    total += black_box(transducer.query(query, max_distance).count());
                }
                total
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_wallbreaker_query,
    bench_traditional_transducer,
    bench_scdawg_construction,
    bench_substring_search,
    bench_pattern_splitting,
    bench_wallbreaker_vs_traditional,
);
criterion_main!(benches);
