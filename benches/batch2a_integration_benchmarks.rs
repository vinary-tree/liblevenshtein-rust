use criterion::{black_box, criterion_group, criterion_main, Criterion};
use liblevenshtein::prelude::*;

/// Integration benchmark: End-to-end query performance with Batch 2A SIMD
///
/// This benchmark measures the actual query-level performance improvement
/// from integrating SIMD optimizations (minimum distance) into the transducer.
fn bench_query_integration(c: &mut Criterion) {
    // Create a test dictionary with common English words
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

    let dict = DoubleArrayTrie::from_terms(terms);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut group = c.benchmark_group("query_integration");

    // Test 1: Short query with exact match
    group.bench_function("exact_match", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer.query(black_box("test"), black_box(0)).collect();
            black_box(results)
        });
    });

    // Test 2: Query with distance 1 (typical autocorrect)
    group.bench_function("distance_1", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer.query(black_box("tset"), black_box(1)).collect();
            black_box(results)
        });
    });

    // Test 3: Query with distance 2 (fuzzy search)
    group.bench_function("distance_2", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer.query(black_box("tset"), black_box(2)).collect();
            black_box(results)
        });
    });

    // Test 4: Longer query with distance 2
    group.bench_function("long_query_distance_2", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query(black_box("programing"), black_box(2))
                .collect();
            black_box(results)
        });
    });

    // Test 5: Multiple queries (realistic workload)
    let test_queries = vec![
        ("test", 1),
        ("tset", 2),
        ("programing", 2),
        ("computr", 2),
        ("helo", 1),
    ];

    group.bench_function("realistic_workload", |b| {
        b.iter(|| {
            for (query, max_distance) in &test_queries {
                let results: Vec<_> = transducer
                    .query(black_box(*query), black_box(*max_distance))
                    .collect();
                black_box(results);
            }
        });
    });

    group.finish();
}

/// Benchmark: State min_distance() calls (directly measures SIMD impact)
///
/// This measures the specific function we integrated SIMD into.
fn bench_min_distance_integration(c: &mut Criterion) {
    use liblevenshtein::transducer::{Position, State};

    let mut group = c.benchmark_group("min_distance_integration");

    // Test with different state sizes
    for size in [2, 4, 6, 8] {
        let mut state = State::new();
        let query_length = 20;
        for i in 0..size {
            state.insert(Position::new(i, i % 3), Algorithm::Standard, query_length);
        }

        group.bench_function(&format!("{}_positions", size), |b| {
            b.iter(|| black_box(state.min_distance()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_query_integration,
    bench_min_distance_integration,
);
criterion_main!(benches);
