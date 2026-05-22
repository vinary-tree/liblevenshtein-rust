//! Benchmarks for pre-filtering modules: N-gram index, Jaro-Winkler, HybridMatcher.
//!
//! Measures:
//! - N-gram index construction and query performance
//! - Jaro-Winkler similarity computation throughput
//! - Hybrid filter effectiveness vs full Levenshtein automaton

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::filter::{jaro_similarity, jaro_winkler_similarity, HybridMatcher, NgramIndex};

// ============================================================================
// Test Data Generation
// ============================================================================

/// Generate a synthetic dictionary of given size
fn generate_dictionary(size: usize) -> Vec<String> {
    // Common English word patterns for realistic benchmarking
    let prefixes = [
        "pre", "un", "re", "dis", "over", "mis", "out", "sub", "trans", "inter",
    ];
    let roots = [
        "act", "form", "port", "duct", "ject", "tract", "struct", "scribe", "spect", "vert",
    ];
    let suffixes = [
        "ion", "ive", "ment", "ness", "able", "ible", "ous", "ful", "less", "ly",
    ];

    let mut words = Vec::with_capacity(size);

    // Add base roots
    for root in &roots {
        words.push(root.to_string());
        if words.len() >= size {
            return words;
        }
    }

    // Add prefix + root combinations
    for prefix in &prefixes {
        for root in &roots {
            words.push(format!("{}{}", prefix, root));
            if words.len() >= size {
                return words;
            }
        }
    }

    // Add root + suffix combinations
    for root in &roots {
        for suffix in &suffixes {
            words.push(format!("{}{}", root, suffix));
            if words.len() >= size {
                return words;
            }
        }
    }

    // Add prefix + root + suffix combinations
    for prefix in &prefixes {
        for root in &roots {
            for suffix in &suffixes {
                words.push(format!("{}{}{}", prefix, root, suffix));
                if words.len() >= size {
                    return words;
                }
            }
        }
    }

    // Fill remaining with numbered words
    while words.len() < size {
        words.push(format!("word{}", words.len()));
    }

    words
}

/// String pairs for similarity benchmarks
fn similarity_test_pairs() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        // (name, s1, s2)
        ("identical_short", "test", "test"),
        ("identical_medium", "programming", "programming"),
        ("similar_short", "test", "tset"),
        ("similar_medium", "martha", "marhta"),
        ("different_short", "abc", "xyz"),
        ("different_medium", "hello", "world"),
        ("prefix_match", "prefix_abc", "prefix_xyz"),
        ("unicode", "café", "cafe"),
        // Classic Jaro test cases
        ("classic_martha", "MARTHA", "MARHTA"),
        ("classic_dwayne", "DWAYNE", "DUANE"),
        ("classic_dixon", "DIXON", "DICKSONX"),
    ]
}

// ============================================================================
// N-gram Index Benchmarks
// ============================================================================

fn bench_ngram_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ngram/construction");

    for size in [100, 1_000, 10_000, 50_000] {
        let dictionary = generate_dictionary(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("bigram", size), &dictionary, |b, dict| {
            b.iter(|| {
                let mut index = NgramIndex::new(2);
                for term in dict {
                    index.insert(black_box(term));
                }
                index
            });
        });

        group.bench_with_input(BenchmarkId::new("trigram", size), &dictionary, |b, dict| {
            b.iter(|| {
                let mut index = NgramIndex::new(3);
                for term in dict {
                    index.insert(black_box(term));
                }
                index
            });
        });
    }

    group.finish();
}

fn bench_ngram_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("ngram/query");

    let queries = [
        ("short", "test"),
        ("medium", "programming"),
        ("typo", "progamming"),
        ("long", "acknowledgement"),
    ];

    for size in [1_000, 10_000, 50_000] {
        let dictionary = generate_dictionary(size);
        let mut index = NgramIndex::new(2);
        for term in &dictionary {
            index.insert(term);
        }

        for (query_name, query) in &queries {
            for max_dist in [1, 2, 3] {
                let id = format!("{}_{}_d{}", size, query_name, max_dist);
                group.bench_function(&id, |b| {
                    b.iter(|| index.find_candidates(black_box(query), black_box(max_dist)));
                });
            }
        }
    }

    group.finish();
}

fn bench_ngram_rejection_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("ngram/rejection_rate");
    group.sample_size(50);

    let sizes = [1_000, 10_000, 50_000];

    for size in sizes {
        let dictionary = generate_dictionary(size);
        let mut index = NgramIndex::new(2);
        for term in &dictionary {
            index.insert(term);
        }

        // Measure how many candidates pass the filter
        let query = "progamming";
        for max_dist in [1, 2, 3] {
            let candidates = index.find_candidates(query, max_dist);
            let rejection_rate = 1.0 - (candidates.len() as f64 / size as f64);

            let id = format!("dict_{}_d{}", size, max_dist);
            group.bench_function(&id, |b| {
                b.iter(|| {
                    let c = index.find_candidates(black_box(query), black_box(max_dist));
                    (c.len(), rejection_rate)
                });
            });
        }
    }

    group.finish();
}

// ============================================================================
// Jaro-Winkler Benchmarks
// ============================================================================

fn bench_jaro_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("jaro/similarity");

    for (name, s1, s2) in similarity_test_pairs() {
        let bytes = (s1.len() + s2.len()) as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(BenchmarkId::new("jaro", name), &(s1, s2), |b, (s1, s2)| {
            b.iter(|| jaro_similarity(black_box(s1), black_box(s2)));
        });
    }

    group.finish();
}

fn bench_jaro_winkler_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("jaro_winkler/similarity");

    for (name, s1, s2) in similarity_test_pairs() {
        let bytes = (s1.len() + s2.len()) as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(
            BenchmarkId::new("jaro_winkler", name),
            &(s1, s2),
            |b, (s1, s2)| {
                b.iter(|| jaro_winkler_similarity(black_box(s1), black_box(s2)));
            },
        );
    }

    group.finish();
}

fn bench_jaro_vs_jaro_winkler(c: &mut Criterion) {
    let mut group = c.benchmark_group("jaro_vs_winkler");

    let pairs = [
        ("identical", "programming", "programming"),
        ("prefix_match", "MARTHA", "MARHTA"),
        ("no_prefix", "DWAYNE", "DUANE"),
    ];

    for (name, s1, s2) in pairs {
        group.bench_with_input(BenchmarkId::new("jaro", name), &(s1, s2), |b, (s1, s2)| {
            b.iter(|| jaro_similarity(black_box(s1), black_box(s2)));
        });

        group.bench_with_input(
            BenchmarkId::new("jaro_winkler", name),
            &(s1, s2),
            |b, (s1, s2)| {
                b.iter(|| jaro_winkler_similarity(black_box(s1), black_box(s2)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Hybrid Matcher Benchmarks
// ============================================================================

fn bench_hybrid_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid/construction");

    for size in [1_000, 10_000, 50_000] {
        let dictionary = generate_dictionary(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("from_iter", size),
            &dictionary,
            |b, dict| {
                b.iter(|| HybridMatcher::new(black_box(dict.iter().cloned())));
            },
        );
    }

    group.finish();
}

fn bench_hybrid_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid/filter");

    let queries = [
        ("exact", "preact"),
        ("typo", "progamming"),
        ("prefix", "prefor"),
        ("distant", "zzzzz"),
    ];

    for size in [1_000, 10_000, 50_000] {
        let dictionary = generate_dictionary(size);
        let matcher = HybridMatcher::new(dictionary.iter().cloned());

        for (query_name, query) in &queries {
            for max_dist in [1, 2] {
                let id = format!("{}_{}_d{}", size, query_name, max_dist);
                group.bench_function(&id, |b| {
                    b.iter(|| matcher.filter_candidates(black_box(query), black_box(max_dist)));
                });
            }
        }
    }

    group.finish();
}

// ============================================================================
// Comparison: Hybrid Filter vs Full Levenshtein
// ============================================================================

fn bench_hybrid_vs_full_levenshtein(c: &mut Criterion) {
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use liblevenshtein::transducer::{Algorithm, Transducer};

    let mut group = c.benchmark_group("hybrid_vs_full");
    group.sample_size(30);

    let sizes = [1_000, 10_000];

    for size in sizes {
        let dictionary = generate_dictionary(size);

        // Build hybrid matcher
        let matcher = HybridMatcher::new(dictionary.iter().cloned());

        // Build Levenshtein transducer
        let dawg: DynamicDawg<()> = DynamicDawg::new();
        for term in &dictionary {
            dawg.insert(term);
        }
        let transducer = Transducer::new(dawg, Algorithm::Standard);

        let query = "progamming";
        let max_dist = 2;

        // Benchmark hybrid filter only
        group.bench_function(format!("{}_hybrid_filter", size), |b| {
            b.iter(|| matcher.filter_candidates(black_box(query), black_box(max_dist)));
        });

        // Benchmark full Levenshtein traversal
        group.bench_function(format!("{}_full_levenshtein", size), |b| {
            b.iter(|| {
                transducer
                    .query(black_box(query), black_box(max_dist))
                    .collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    ngram_benches,
    bench_ngram_construction,
    bench_ngram_query,
    bench_ngram_rejection_rate,
);

criterion_group!(
    jaro_benches,
    bench_jaro_similarity,
    bench_jaro_winkler_similarity,
    bench_jaro_vs_jaro_winkler,
);

criterion_group!(
    hybrid_benches,
    bench_hybrid_construction,
    bench_hybrid_filter,
    bench_hybrid_vs_full_levenshtein,
);

criterion_main!(ngram_benches, jaro_benches, hybrid_benches);
