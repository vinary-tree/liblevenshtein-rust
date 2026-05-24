#![cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]

#[cfg(target_arch = "x86_64")]
use criterion::BenchmarkId;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(target_arch = "x86_64")]
use liblevenshtein::transducer::simd::find_edge_label_simd;

/// Benchmark SIMD edge lookup vs scalar across various edge counts.
///
/// This measures the raw performance of the SIMD edge lookup function
/// compared to scalar linear search, validating our threshold choices.
fn bench_edge_lookup_simd_vs_scalar(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        let mut group = c.benchmark_group("edge_lookup_simd_vs_scalar");

        // Test case 1: 4 edges (SSE4.1 threshold)
        let edges_4: Vec<(u8, usize)> = vec![(b'a', 1), (b'e', 2), (b'i', 3), (b'o', 4)];

        group.bench_function(BenchmarkId::new("SIMD", "4_edges"), |b| {
            b.iter(|| black_box(find_edge_label_simd(black_box(&edges_4), black_box(b'i'))));
        });

        group.bench_function(BenchmarkId::new("Scalar", "4_edges"), |b| {
            b.iter(|| black_box(edges_4.iter().position(|(l, _)| *l == black_box(b'i'))));
        });

        // Test case 2: 8 edges (SSE4.1 optimal)
        let edges_8: Vec<(u8, usize)> = (0..8).map(|i| (b'a' + i, i as usize)).collect();

        group.bench_function(BenchmarkId::new("SIMD", "8_edges"), |b| {
            b.iter(|| black_box(find_edge_label_simd(black_box(&edges_8), black_box(b'e'))));
        });

        group.bench_function(BenchmarkId::new("Scalar", "8_edges"), |b| {
            b.iter(|| black_box(edges_8.iter().position(|(l, _)| *l == black_box(b'e'))));
        });

        // Test case 3: 16 edges (AVX2 threshold)
        let edges_16: Vec<(u8, usize)> = (0..16).map(|i| (b'a' + i, i as usize)).collect();

        group.bench_function(BenchmarkId::new("SIMD", "16_edges"), |b| {
            b.iter(|| black_box(find_edge_label_simd(black_box(&edges_16), black_box(b'h'))));
        });

        group.bench_function(BenchmarkId::new("Scalar", "16_edges"), |b| {
            b.iter(|| black_box(edges_16.iter().position(|(l, _)| *l == black_box(b'h'))));
        });

        // Test case 4: 32 edges (AVX2 full)
        let edges_32: Vec<(u8, usize)> = (0..32).map(|i| (i as u8, i as usize)).collect();

        group.bench_function(BenchmarkId::new("SIMD", "32_edges"), |b| {
            b.iter(|| black_box(find_edge_label_simd(black_box(&edges_32), black_box(16))));
        });

        group.bench_function(BenchmarkId::new("Scalar", "32_edges"), |b| {
            b.iter(|| black_box(edges_32.iter().position(|(l, _)| *l == black_box(16))));
        });

        group.finish();
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("SIMD benchmarks require x86_64 architecture and simd feature");
    }
}

/// Benchmark edge lookup at different positions (first, middle, last).
///
/// This validates that SIMD maintains consistent performance regardless of
/// where the target label appears in the sorted edge list.
fn bench_edge_lookup_position_variance(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        let mut group = c.benchmark_group("edge_lookup_position");

        let edges_16: Vec<(u8, usize)> = (0..16).map(|i| (b'a' + i, i as usize)).collect();

        // First position (index 0)
        group.bench_function("first", |b| {
            b.iter(|| black_box(find_edge_label_simd(black_box(&edges_16), black_box(b'a'))));
        });

        // Middle position (index 8)
        group.bench_function("middle", |b| {
            b.iter(|| black_box(find_edge_label_simd(black_box(&edges_16), black_box(b'i'))));
        });

        // Last position (index 15)
        group.bench_function("last", |b| {
            b.iter(|| black_box(find_edge_label_simd(black_box(&edges_16), black_box(b'p'))));
        });

        // Not found
        group.bench_function("not_found", |b| {
            b.iter(|| black_box(find_edge_label_simd(black_box(&edges_16), black_box(b'z'))));
        });

        group.finish();
    }
}

/// Benchmark realistic mixed workload with varying edge counts.
///
/// This simulates actual dictionary traversal where nodes have different
/// numbers of edges. Most nodes have 1-5 edges, but some have 10+.
fn bench_edge_lookup_realistic_workload(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        let mut group = c.benchmark_group("edge_lookup_realistic");

        // Create a representative distribution of node edge counts
        // Based on typical English dictionary structure
        let test_cases: Vec<(Vec<(u8, usize)>, u8)> = vec![
            // 1 edge (common for leaf-adjacent nodes)
            (vec![(b'a', 1)], b'a'),
            // 2 edges (very common)
            (vec![(b'a', 1), (b'e', 2)], b'e'),
            // 3 edges (common)
            (vec![(b'a', 1), (b'e', 2), (b'i', 3)], b'i'),
            // 5 edges (common vowels + consonants)
            (
                vec![(b'a', 1), (b'e', 2), (b'i', 3), (b'o', 4), (b'u', 5)],
                b'o',
            ),
            // 8 edges (high-branching node)
            ((0..8).map(|i| (b'a' + i, i as usize)).collect(), b'e'),
            // 12 edges (rare but exists)
            ((0..12).map(|i| (b'a' + i, i as usize)).collect(), b'h'),
        ];

        group.bench_function("mixed_workload", |b| {
            b.iter(|| {
                for (edges, target) in &test_cases {
                    black_box(find_edge_label_simd(black_box(edges), black_box(*target)));
                }
            });
        });

        group.finish();
    }
}

/// Benchmark dictionary integration.
///
/// This measures the end-to-end performance improvement in actual
/// dictionary operations with SIMD edge lookup integrated.
fn bench_dawg_dictionary_integration(c: &mut Criterion) {
    use liblevenshtein::prelude::*;

    let mut group = c.benchmark_group("dictionary_integration");

    // Create a dictionary with realistic English words
    let terms = vec![
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "test",
        "testing",
        "tester",
        "tests",
        "tested",
        "hello",
        "world",
        "help",
        "helper",
        "helping",
        "program",
        "programming",
        "programmer",
        "programs",
        "computer",
        "computing",
        "computation",
        "compute",
    ];

    let dict = DoubleArrayTrie::from_terms(terms.clone());

    // Benchmark: Dictionary contains() (uses transition() internally)
    group.bench_function("contains_existing", |b| {
        b.iter(|| black_box(dict.contains(black_box("programming"))));
    });

    group.bench_function("contains_missing", |b| {
        b.iter(|| black_box(dict.contains(black_box("nonexistent"))));
    });

    // Benchmark: Multiple lookups (realistic workload)
    let test_words = vec!["test", "programming", "hello", "computer", "nonexistent"];

    group.bench_function("batch_contains", |b| {
        b.iter(|| {
            for word in &test_words {
                black_box(dict.contains(black_box(word)));
            }
        });
    });

    group.finish();
}

/// Benchmark transducer query performance with SIMD-accelerated dictionaries.
///
/// This measures the overall query-level impact of SIMD edge lookup.
/// Expected improvement: 8-12% on fuzzy queries.
fn bench_transducer_query_integration(c: &mut Criterion) {
    use liblevenshtein::prelude::*;

    let mut group = c.benchmark_group("transducer_query_integration");

    // Create dictionary and transducer
    let terms = vec![
        "test",
        "testing",
        "tester",
        "tests",
        "tested",
        "best",
        "rest",
        "west",
        "pest",
        "nest",
        "hello",
        "help",
        "helper",
        "helping",
        "helped",
        "world",
        "word",
        "work",
        "working",
        "worker",
        "program",
        "programming",
        "programmer",
        "programs",
        "compute",
        "computer",
        "computing",
        "computation",
    ];

    let dict = DoubleArrayTrie::from_terms(terms);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Query with distance 1 (typical autocorrect)
    group.bench_function("query_distance_1", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer.query(black_box("tset"), black_box(1)).collect();
            black_box(results)
        });
    });

    // Query with distance 2 (fuzzy search)
    group.bench_function("query_distance_2", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query(black_box("programing"), black_box(2))
                .collect();
            black_box(results)
        });
    });

    // Realistic mixed workload
    let test_queries = vec![
        ("test", 1),
        ("tset", 2),
        ("helo", 1),
        ("programing", 2),
        ("computr", 2),
    ];

    group.bench_function("realistic_workload", |b| {
        b.iter(|| {
            for (query, max_dist) in &test_queries {
                let results: Vec<_> = transducer
                    .query(black_box(*query), black_box(*max_dist))
                    .collect();
                black_box(results);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_edge_lookup_simd_vs_scalar,
    bench_edge_lookup_position_variance,
    bench_edge_lookup_realistic_workload,
    bench_dawg_dictionary_integration,
    bench_transducer_query_integration,
);
criterion_main!(benches);
