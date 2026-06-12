//! Comprehensive comparison of fuzzy query backends.
//!
//! Compares:
//! - **WallBreaker** (with Scdawg backend) - pigeonhole + suffix automaton
//! - **DynamicDawg** (with Transducer) - Levenshtein automaton
//! - **DoubleArrayTrie** (with Transducer) - cache-optimized automaton
//!
//! Across all three Levenshtein algorithm variants:
//! - Standard (insert, delete, substitute)
//! - Transposition (+ swap adjacent)
//! - MergeAndSplit (+ merge, split)
//!
//! Run with:
//!   cargo bench --bench backend_fuzzy_comparison
//!
//! Save baseline:
//!   cargo bench --bench backend_fuzzy_comparison -- --save-baseline backend-baseline

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::pathmap::PathMapDictionary;
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
        generate_synthetic_dictionary(target_size)
    }
}

/// Generate synthetic dictionary for systems without /usr/share/dict/words
fn generate_synthetic_dictionary(size: usize) -> Vec<String> {
    let base_words = [
        "algorithm",
        "structure",
        "computer",
        "science",
        "program",
        "function",
        "variable",
        "constant",
        "iterator",
        "reference",
        "pattern",
        "matching",
        "distance",
        "automaton",
        "transducer",
        "dictionary",
        "benchmark",
        "performance",
        "optimization",
        "implementation",
        "cathedral",
        "category",
        "catering",
        "catastrophe",
        "catalyst",
    ];

    let suffixes = [
        "", "s", "ed", "ing", "er", "est", "ly", "tion", "ment", "ness",
    ];
    let prefixes = [
        "", "un", "re", "pre", "mis", "dis", "over", "under", "out", "sub",
    ];

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

    let mut i = 0;
    while words.len() < size {
        words.insert(format!("word{:06}", i));
        i += 1;
    }

    words.into_iter().collect()
}

/// Generate a query based on a dictionary word with modifications
fn generate_realistic_query(dict: &[String], seed: usize, target_len: usize) -> String {
    if dict.is_empty() {
        return (0..target_len)
            .map(|i| {
                let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
                chars[(seed * 31 + i * 17) % chars.len()]
            })
            .collect();
    }

    let word = &dict[seed % dict.len()];

    if word.len() >= target_len {
        word.chars().take(target_len).collect()
    } else {
        let mut query: String = word.clone();
        let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
        while query.len() < target_len {
            let idx = (seed * 31 + query.len() * 17) % chars.len();
            query.push(chars[idx]);
        }
        query
    }
}

/// Generate multiple queries of specified length
fn generate_queries(dict: &[String], count: usize, target_len: usize) -> Vec<String> {
    (0..count)
        .map(|i| generate_realistic_query(dict, i * 7919, target_len))
        .collect()
}

// ============================================================================
// Fuzzy Query Comparison Benchmarks
// ============================================================================

/// Compare fuzzy query performance across backends and algorithms
fn bench_fuzzy_queries(c: &mut Criterion) {
    // Use full dictionary for realistic comparison
    let dict_words = load_dictionary(90_000);
    let actual_size = dict_words.len();

    eprintln!("Loaded {} dictionary words", actual_size);

    // Test configurations: (max_distance, query_len)
    let configs = [
        (1u8, 10),
        (1, 20),
        (2, 10),
        (2, 20),
        (4, 20),
        (4, 50),
        (8, 50),
    ];

    // Test each algorithm
    for algorithm in [
        Algorithm::Standard,
        Algorithm::Transposition,
        Algorithm::MergeAndSplit,
    ] {
        let algo_name = match algorithm {
            Algorithm::Standard => "Standard",
            Algorithm::Transposition => "Transposition",
            Algorithm::MergeAndSplit => "MergeAndSplit",
        };

        for (max_dist, query_len) in configs {
            let group_name = format!("{}_k{}_q{}", algo_name, max_dist, query_len);
            let mut group = c.benchmark_group(&group_name);
            group.sample_size(50);

            // Generate queries once for fair comparison
            let queries = generate_queries(&dict_words, 20, query_len);
            group.throughput(Throughput::Elements(queries.len() as u64));

            // ============ DynamicDawg + Transducer ============
            {
                let dawg = DynamicDawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
                let transducer = Transducer::new(dawg, algorithm);

                group.bench_function(BenchmarkId::new("DynamicDawg", ""), |b| {
                    b.iter(|| {
                        let mut total = 0usize;
                        for q in &queries {
                            total += black_box(transducer.query(q, max_dist as usize).count());
                        }
                        total
                    })
                });
            }

            // ============ DoubleArrayTrie + Transducer ============
            {
                let dat = DoubleArrayTrie::<()>::from_terms(dict_words.clone());
                let transducer = Transducer::new(dat, algorithm);

                group.bench_function(BenchmarkId::new("DoubleArrayTrie", ""), |b| {
                    b.iter(|| {
                        let mut total = 0usize;
                        for q in &queries {
                            total += black_box(transducer.query(q, max_dist as usize).count());
                        }
                        total
                    })
                });
            }

            // ============ WallBreaker + Scdawg ============
            // WallBreaker now supports all algorithms with formally verified piece counts:
            // - Standard: k+1 pieces
            // - Transposition: 2k+1 pieces (proven in WallBreakerPigeonhole.v)
            // - MergeAndSplit: 2k+1 pieces (proven in WallBreakerPigeonhole.v)
            {
                let scdawg = Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
                let wallbreaker =
                    WallBreaker::with_algorithm(&scdawg, max_dist as usize, algorithm);

                group.bench_function(BenchmarkId::new("WallBreaker", ""), |b| {
                    b.iter(|| {
                        let mut total = 0usize;
                        for q in &queries {
                            total += black_box(wallbreaker.query(q).count());
                        }
                        total
                    })
                });
            }

            // ============ PathMapDictionary + Transducer (TrieRef snapshot) ============
            {
                let pm = PathMapDictionary::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
                let transducer = Transducer::new(pm, algorithm);

                group.bench_function(BenchmarkId::new("PathMap", ""), |b| {
                    b.iter(|| {
                        let mut total = 0usize;
                        for q in &queries {
                            total += black_box(transducer.query(q, max_dist as usize).count());
                        }
                        total
                    })
                });
            }

            group.finish();
        }
    }
}

// ============================================================================
// Construction Time Comparison
// ============================================================================

/// Compare construction times for each backend
fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    group.sample_size(30); // Fewer samples for slower construction

    for dict_size in [1_000, 10_000, 89_000] {
        let dict_words = load_dictionary(dict_size);
        let actual_size = dict_words.len();
        let id = format!("d{}", actual_size);

        // DynamicDawg construction
        group.bench_function(BenchmarkId::new("DynamicDawg", &id), |b| {
            b.iter(|| {
                black_box(DynamicDawg::<()>::from_terms(
                    dict_words.iter().map(|s| s.as_str()),
                ))
            })
        });

        // DoubleArrayTrie construction
        group.bench_function(BenchmarkId::new("DoubleArrayTrie", &id), |b| {
            b.iter(|| black_box(DoubleArrayTrie::<()>::from_terms(dict_words.clone())))
        });

        // Scdawg construction
        group.bench_function(BenchmarkId::new("Scdawg", &id), |b| {
            b.iter(|| {
                black_box(Scdawg::<()>::from_terms(
                    dict_words.iter().map(|s| s.as_str()),
                ))
            })
        });
    }

    group.finish();
}

// ============================================================================
// Scaling Analysis
// ============================================================================

/// Test how performance scales with dictionary size (Standard algorithm only)
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");
    group.sample_size(30);

    let max_dist = 2u8;
    let query_len = 20;

    for dict_size in [1_000, 5_000, 10_000, 50_000, 89_000] {
        let dict_words = load_dictionary(dict_size);
        let actual_size = dict_words.len();
        let queries = generate_queries(&dict_words, 10, query_len);

        let id = format!("d{}_k{}", actual_size, max_dist);

        // DynamicDawg
        {
            let dawg = DynamicDawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
            let transducer = Transducer::new(dawg, Algorithm::Standard);

            group.bench_function(BenchmarkId::new("DynamicDawg", &id), |b| {
                b.iter(|| {
                    let mut total = 0usize;
                    for q in &queries {
                        total += black_box(transducer.query(q, max_dist as usize).count());
                    }
                    total
                })
            });
        }

        // DoubleArrayTrie
        {
            let dat = DoubleArrayTrie::<()>::from_terms(dict_words.clone());
            let transducer = Transducer::new(dat, Algorithm::Standard);

            group.bench_function(BenchmarkId::new("DoubleArrayTrie", &id), |b| {
                b.iter(|| {
                    let mut total = 0usize;
                    for q in &queries {
                        total += black_box(transducer.query(q, max_dist as usize).count());
                    }
                    total
                })
            });
        }

        // WallBreaker
        {
            let scdawg = Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
            let wallbreaker = WallBreaker::new(&scdawg, max_dist as usize);

            group.bench_function(BenchmarkId::new("WallBreaker", &id), |b| {
                b.iter(|| {
                    let mut total = 0usize;
                    for q in &queries {
                        total += black_box(wallbreaker.query(q).count());
                    }
                    total
                })
            });
        }
    }

    group.finish();
}

// ============================================================================
// High Distance Comparison (where WallBreaker should excel)
// ============================================================================

/// Test high edit distances where WallBreaker's advantage should be largest
fn bench_high_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_distance");
    group.sample_size(20); // Fewer samples for very slow operations

    let dict_words = load_dictionary(50_000);
    let actual_size = dict_words.len();

    eprintln!("High distance benchmark with {} words", actual_size);

    // High distance configurations
    for (max_dist, query_len) in [(8u8, 50), (8, 100), (16, 100)] {
        let queries = generate_queries(&dict_words, 5, query_len);
        let id = format!("k{}_q{}", max_dist, query_len);

        // DynamicDawg
        {
            let dawg = DynamicDawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
            let transducer = Transducer::new(dawg, Algorithm::Standard);

            group.bench_function(BenchmarkId::new("DynamicDawg", &id), |b| {
                b.iter(|| {
                    let mut total = 0usize;
                    for q in &queries {
                        total += black_box(transducer.query(q, max_dist as usize).count());
                    }
                    total
                })
            });
        }

        // DoubleArrayTrie
        {
            let dat = DoubleArrayTrie::<()>::from_terms(dict_words.clone());
            let transducer = Transducer::new(dat, Algorithm::Standard);

            group.bench_function(BenchmarkId::new("DoubleArrayTrie", &id), |b| {
                b.iter(|| {
                    let mut total = 0usize;
                    for q in &queries {
                        total += black_box(transducer.query(q, max_dist as usize).count());
                    }
                    total
                })
            });
        }

        // WallBreaker
        {
            let scdawg = Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
            let wallbreaker = WallBreaker::new(&scdawg, max_dist as usize);

            group.bench_function(BenchmarkId::new("WallBreaker", &id), |b| {
                b.iter(|| {
                    let mut total = 0usize;
                    for q in &queries {
                        total += black_box(wallbreaker.query(q).count());
                    }
                    total
                })
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fuzzy_queries,
    bench_construction,
    bench_scaling,
    bench_high_distance
);
criterion_main!(benches);
