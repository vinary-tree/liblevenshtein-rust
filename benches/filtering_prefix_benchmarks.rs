use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::prelude::*;

// Load real dictionary for realistic benchmarks
fn load_dictionary() -> Vec<String> {
    std::fs::read_to_string("/usr/share/dict/words")
        .unwrap_or_else(|_| {
            // Fallback: generate synthetic dictionary
            (0..10000)
                .map(|i| format!("word{}", i))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .lines()
        .map(|s| s.to_lowercase())
        .filter(|s| s.len() >= 3 && s.len() <= 15)
        .collect()
}

// Generate code-like identifiers
fn generate_code_identifiers(count: usize) -> Vec<String> {
    let mut identifiers = Vec::new();

    let prefixes = [
        "get", "set", "is", "has", "create", "delete", "update", "find",
    ];
    let middles = [
        "User", "Value", "Item", "Data", "Record", "Entry", "Element",
    ];
    let suffixes = ["", "ById", "ByName", "FromCache", "Async", "Sync"];

    for prefix in &prefixes {
        for middle in &middles {
            for suffix in &suffixes {
                identifiers.push(format!("{}{}{}", prefix, middle, suffix));
                if identifiers.len() >= count {
                    return identifiers;
                }
            }
        }
    }

    identifiers
}

// Benchmark: Exact matching vs Prefix matching
fn bench_prefix_vs_exact(c: &mut Criterion) {
    let words = load_dictionary();
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(words.iter().map(|s| s.as_str()));

    let mut group = c.benchmark_group("prefix_vs_exact");

    for query_len in [3, 5, 7, 10] {
        let query = "test".chars().take(query_len).collect::<String>();

        // Exact matching
        group.bench_with_input(BenchmarkId::new("exact", query_len), &query, |b, q| {
            b.iter(|| {
                let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                    .query_ordered(black_box(q), 0)
                    .take(10)
                    .collect();
                black_box(results);
            });
        });

        // Prefix matching
        group.bench_with_input(BenchmarkId::new("prefix", query_len), &query, |b, q| {
            b.iter(|| {
                let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                    .query_ordered(black_box(q), 0)
                    .prefix()
                    .take(10)
                    .collect();
                black_box(results);
            });
        });
    }

    group.finish();
}

// Benchmark: Different edit distances with prefix matching
fn bench_prefix_distances(c: &mut Criterion) {
    let words = load_dictionary();
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(words.iter().map(|s| s.as_str()));

    let mut group = c.benchmark_group("prefix_distances");

    for distance in [0, 1, 2, 3] {
        group.bench_with_input(
            BenchmarkId::new("distance", distance),
            &distance,
            |b, &d| {
                b.iter(|| {
                    let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                        .query_ordered(black_box("test"), d)
                        .prefix()
                        .take(10)
                        .collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

// Benchmark: Post-filtering strategies
fn bench_filtering_strategies(c: &mut Criterion) {
    let identifiers = generate_code_identifiers(5000);
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(identifiers.iter().map(|s| s.as_str()));

    // Create metadata for filtering
    let public_identifiers: Vec<_> = identifiers
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 == 0) // 33% are "public"
        .map(|(_, s)| s.as_str())
        .collect();

    let public_set: std::collections::HashSet<_> = public_identifiers.iter().cloned().collect();

    let mut group = c.benchmark_group("filtering_strategies");

    // Post-filtering
    group.bench_function("post_filter", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("getVal"), 1)
                .prefix()
                .filter(|c| public_set.contains(c.term.as_str()))
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Pre-filtering (sub-trie)
    let filtered_dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(public_identifiers.clone());
    group.bench_function("pre_filter", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(filtered_dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("getVal"), 1)
                .prefix()
                .take(10)
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

// Benchmark: Filter complexity impact
fn bench_filter_complexity(c: &mut Criterion) {
    let identifiers = generate_code_identifiers(3000);
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(identifiers.iter().map(|s| s.as_str()));

    let mut group = c.benchmark_group("filter_complexity");

    // Simple filter (O(1) - hashset lookup)
    let public_set: std::collections::HashSet<_> = identifiers
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, s)| s.as_str())
        .collect();

    group.bench_function("simple_filter", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("get"), 0)
                .prefix()
                .filter(|c| public_set.contains(c.term.as_str()))
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Medium filter (string operations)
    group.bench_function("medium_filter", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("get"), 0)
                .prefix()
                .filter(|c| c.term.len() > 5 && c.term.chars().nth(3).unwrap().is_uppercase())
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Complex filter (multiple conditions)
    group.bench_function("complex_filter", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("get"), 0)
                .prefix()
                .filter(|c| {
                    c.term.len() > 5
                        && c.term.contains("User")
                        && c.term.chars().filter(|ch| ch.is_uppercase()).count() >= 2
                })
                .take(10)
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

// Benchmark: Combined operations
fn bench_combined_operations(c: &mut Criterion) {
    let identifiers = generate_code_identifiers(3000);
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(identifiers.iter().map(|s| s.as_str()));

    let public_set: std::collections::HashSet<_> = identifiers
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, s)| s.as_str())
        .collect();

    let mut group = c.benchmark_group("combined_operations");

    // Just prefix
    group.bench_function("prefix_only", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("get"), 0)
                .prefix()
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Prefix + filter
    group.bench_function("prefix_and_filter", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("get"), 0)
                .prefix()
                .filter(|c| public_set.contains(c.term.as_str()))
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Prefix + distance + filter
    group.bench_function("prefix_distance_filter", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("get"), 1)
                .prefix()
                .filter(|c| public_set.contains(c.term.as_str()))
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Prefix + distance + multiple filters
    group.bench_function("prefix_distance_multi_filter", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("get"), 1)
                .prefix()
                .filter(|c| public_set.contains(c.term.as_str()))
                .filter(|c| c.term.len() > 5)
                .take(10)
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

// Benchmark: Scalability with dictionary size
fn bench_scalability(c: &mut Criterion) {
    let all_words = load_dictionary();

    let mut group = c.benchmark_group("scalability");
    group.sample_size(20); // Fewer samples for large benchmarks

    for size in [1000, 5000, 10000, 20000] {
        let words: Vec<_> = all_words.iter().take(size).map(|s| s.as_str()).collect();
        let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(words);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("prefix_search", size), &dict, |b, d| {
            b.iter(|| {
                let results: Vec<_> = Transducer::new(d.clone(), Algorithm::Standard)
                    .query_ordered(black_box("test"), 1)
                    .prefix()
                    .take(10)
                    .collect();
                black_box(results);
            });
        });
    }

    group.finish();
}

// Benchmark: take() vs take_while() performance
fn bench_iteration_limits(c: &mut Criterion) {
    let identifiers = generate_code_identifiers(3000);
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms(identifiers.iter().map(|s| s.as_str()));

    let mut group = c.benchmark_group("iteration_limits");

    // Using take()
    group.bench_function("take_10", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("get"), 1)
                .prefix()
                .take(10)
                .collect();
            black_box(results);
        });
    });

    // Using take_while() with distance
    group.bench_function("take_while_distance", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("get"), 2)
                .prefix()
                .take_while(|c| c.distance <= 1)
                .collect();
            black_box(results);
        });
    });

    // Collect all (worst case)
    group.bench_function("collect_all", |b| {
        b.iter(|| {
            let results: Vec<_> = Transducer::new(dict.clone(), Algorithm::Standard)
                .query_ordered(black_box("xyz"), 1) // Rare prefix
                .prefix()
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_prefix_vs_exact,
    bench_prefix_distances,
    bench_filtering_strategies,
    bench_filter_complexity,
    bench_combined_operations,
    bench_scalability,
    bench_iteration_limits,
);
criterion_main!(benches);
