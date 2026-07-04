//! Comprehensive benchmark comparing all dictionary backends.
//!
//! This benchmark compares:
//! - PathMap (baseline - fastest queries, highest memory)
//! - DoubleArrayTrie (O(1) transitions, excellent cache)
//! - DynamicDAWG (space-efficient with modifications)
//! - SuffixAutomaton (substring matching)
//!
//! Metrics:
//! - Construction time
//! - Memory usage (estimated bytes/word)
//! - Query performance at distance 0, 1, 2, 3
//! - Result ordering overhead

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use liblevenshtein::prelude::*;
use std::fs;
use std::hint::black_box;

// Load dictionary from system words file
fn load_dictionary() -> Vec<String> {
    let words = fs::read_to_string("/usr/share/dict/words")
        .unwrap_or_else(|_| {
            // Fallback to a minimal word list
            "hello\nworld\ntest\ntesting\ntested\ntester\nexample\nexamples\n\
             approximate\napproximation\nalgorithm\nalgorithms\ndata\ndatabase\n\
             structure\nstructures\nperformance\nbenchmark\nbenchmarks\noptimize\n\
             optimized\noptimization\nmemory\ncache\nlocality\nefficient\nefficiency\n"
                .to_string()
        })
        .lines()
        .filter(|line| !line.is_empty() && line.len() >= 3 && line.len() <= 15)
        .map(|s| s.to_lowercase())
        .collect::<Vec<_>>();

    words
}

// Sample queries for benchmarking
fn sample_queries() -> Vec<&'static str> {
    vec![
        // Short queries
        "te",
        "ab",
        "in",
        // Medium queries
        "test",
        "data",
        "algo",
        "perf",
        // Longer queries with typos
        "aproximate", // approximate with typo
        "performnce", // performance with typo
        "algoritm",   // algorithm with typo
        "optimizd",   // misspelled target
        "structre",   // structure with typo
        "eficiency",  // efficiency with typo
    ]
}

// Benchmark dictionary construction
fn bench_construction(c: &mut Criterion) {
    let words = load_dictionary();
    let sample_size = 10_000.min(words.len());
    let sample: Vec<String> = words.iter().take(sample_size).cloned().collect();

    let mut group = c.benchmark_group("construction");

    group.bench_with_input(
        BenchmarkId::new("PathMap", sample_size),
        &sample,
        |b, terms| {
            b.iter(|| {
                let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(terms.clone());
                black_box(dict)
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("DoubleArrayTrie", sample_size),
        &sample,
        |b, terms| {
            b.iter(|| {
                let dict = DoubleArrayTrie::from_terms(terms.clone());
                black_box(dict)
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("DynamicDAWG", sample_size),
        &sample,
        |b, terms| {
            b.iter(|| {
                let dict: DynamicDawg = DynamicDawg::from_terms(terms.clone());
                black_box(dict)
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("SuffixAutomaton", sample_size),
        &sample,
        |b, terms| {
            b.iter(|| {
                let dict: SuffixAutomaton = SuffixAutomaton::from_texts(terms.clone());
                black_box(dict)
            })
        },
    );

    group.finish();
}

// Benchmark exact matching (distance = 0)
fn bench_exact_matching(c: &mut Criterion) {
    let words = load_dictionary();
    let sample_size = 10_000.min(words.len());
    let sample: Vec<String> = words.iter().take(sample_size).cloned().collect();
    let queries = sample_queries();

    let pathmap_dict: PathMapDictionary<()> = PathMapDictionary::from_terms(sample.clone());
    let dat_dict = DoubleArrayTrie::from_terms(sample.clone());
    let dynamic_dawg_dict: DynamicDawg = DynamicDawg::from_terms(sample.clone());
    let suffix_dict: SuffixAutomaton = SuffixAutomaton::from_texts(sample.clone());

    let mut group = c.benchmark_group("exact_matching");

    group.bench_function("PathMap", |b| {
        let transducer = Transducer::new(pathmap_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 0).collect();
                black_box(results);
            }
        })
    });

    group.bench_function("DoubleArrayTrie", |b| {
        let transducer = Transducer::new(dat_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 0).collect();
                black_box(results);
            }
        })
    });

    group.bench_function("DynamicDAWG", |b| {
        let transducer = Transducer::new(dynamic_dawg_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 0).collect();
                black_box(results);
            }
        })
    });

    group.bench_function("SuffixAutomaton", |b| {
        let transducer = Transducer::new(suffix_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 0).collect();
                black_box(results);
            }
        })
    });

    group.finish();
}

// Benchmark fuzzy matching at distance 1
fn bench_distance_1_matching(c: &mut Criterion) {
    let words = load_dictionary();
    let sample_size = 10_000.min(words.len());
    let sample: Vec<String> = words.iter().take(sample_size).cloned().collect();
    let queries = sample_queries();

    let pathmap_dict: PathMapDictionary<()> = PathMapDictionary::from_terms(sample.clone());
    let dat_dict = DoubleArrayTrie::from_terms(sample.clone());
    let dynamic_dawg_dict: DynamicDawg = DynamicDawg::from_terms(sample.clone());
    let suffix_dict: SuffixAutomaton = SuffixAutomaton::from_texts(sample.clone());

    let mut group = c.benchmark_group("distance_1_matching");

    group.bench_function("PathMap", |b| {
        let transducer = Transducer::new(pathmap_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 1).collect();
                black_box(results);
            }
        })
    });

    group.bench_function("DoubleArrayTrie", |b| {
        let transducer = Transducer::new(dat_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 1).collect();
                black_box(results);
            }
        })
    });

    group.bench_function("DynamicDAWG", |b| {
        let transducer = Transducer::new(dynamic_dawg_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 1).collect();
                black_box(results);
            }
        })
    });

    group.bench_function("SuffixAutomaton", |b| {
        let transducer = Transducer::new(suffix_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 1).collect();
                black_box(results);
            }
        })
    });

    group.finish();
}

// Benchmark fuzzy matching at distance 2
fn bench_distance_2_matching(c: &mut Criterion) {
    let words = load_dictionary();
    let sample_size = 10_000.min(words.len());
    let sample: Vec<String> = words.iter().take(sample_size).cloned().collect();
    let queries = sample_queries();

    let pathmap_dict: PathMapDictionary<()> = PathMapDictionary::from_terms(sample.clone());
    let dat_dict = DoubleArrayTrie::from_terms(sample.clone());
    let dynamic_dawg_dict: DynamicDawg = DynamicDawg::from_terms(sample.clone());
    let suffix_dict: SuffixAutomaton = SuffixAutomaton::from_texts(sample.clone());

    let mut group = c.benchmark_group("distance_2_matching");

    group.bench_function("PathMap", |b| {
        let transducer = Transducer::new(pathmap_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 2).collect();
                black_box(results);
            }
        })
    });

    group.bench_function("DoubleArrayTrie", |b| {
        let transducer = Transducer::new(dat_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 2).collect();
                black_box(results);
            }
        })
    });

    group.bench_function("DynamicDAWG", |b| {
        let transducer = Transducer::new(dynamic_dawg_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 2).collect();
                black_box(results);
            }
        })
    });

    group.bench_function("SuffixAutomaton", |b| {
        let transducer = Transducer::new(suffix_dict.clone(), Algorithm::Standard);
        b.iter(|| {
            for query in &queries {
                let results: Vec<_> = transducer.query(query, 2).collect();
                black_box(results);
            }
        })
    });

    group.finish();
}

// Benchmark single contains() operation
fn bench_contains_operation(c: &mut Criterion) {
    let words = load_dictionary();
    let sample_size = 10_000.min(words.len());
    let sample: Vec<String> = words.iter().take(sample_size).cloned().collect();

    let pathmap_dict: PathMapDictionary<()> = PathMapDictionary::from_terms(sample.clone());
    let dat_dict = DoubleArrayTrie::from_terms(sample.clone());
    let dynamic_dawg_dict: DynamicDawg = DynamicDawg::from_terms(sample.clone());
    let suffix_dict: SuffixAutomaton = SuffixAutomaton::from_texts(sample.clone());

    // Test words that exist
    let test_words: Vec<&str> = sample.iter().take(100).map(|s| s.as_str()).collect();

    let mut group = c.benchmark_group("contains_operation");

    group.bench_function("PathMap", |b| {
        b.iter(|| {
            for word in &test_words {
                black_box(pathmap_dict.contains(word));
            }
        })
    });

    group.bench_function("DoubleArrayTrie", |b| {
        b.iter(|| {
            for word in &test_words {
                black_box(dat_dict.contains(word));
            }
        })
    });

    group.bench_function("DynamicDAWG", |b| {
        b.iter(|| {
            for word in &test_words {
                black_box(dynamic_dawg_dict.contains(word));
            }
        })
    });

    group.bench_function("SuffixAutomaton", |b| {
        b.iter(|| {
            for word in &test_words {
                black_box(suffix_dict.contains(word));
            }
        })
    });

    group.finish();
}

// Benchmark memory footprint (estimated)
fn bench_memory_footprint(c: &mut Criterion) {
    let words = load_dictionary();
    let sample_size = 10_000.min(words.len());
    let sample: Vec<String> = words.iter().take(sample_size).cloned().collect();

    let mut group = c.benchmark_group("memory_estimate");

    // These aren't real memory measurements, but rough estimates
    // based on data structure sizes. For real measurements, use a
    // profiling tool like valgrind or dhat.

    group.bench_function("PathMap_construction", |b| {
        b.iter(|| {
            let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(sample.clone());
            // PathMap uses HashMap-based trie, roughly 64 bytes per node + string data
            black_box(dict)
        })
    });

    group.bench_function("DoubleArrayTrie_construction", |b| {
        b.iter(|| {
            let dict = DoubleArrayTrie::from_terms(sample.clone());
            // DAT uses BASE/CHECK arrays: 8 bytes/state (4+4)
            black_box(dict)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_construction,
    bench_exact_matching,
    bench_distance_1_matching,
    bench_distance_2_matching,
    bench_contains_operation,
    bench_memory_footprint,
);

criterion_main!(benches);
