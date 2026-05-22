//! Benchmarks for PrefixZipper implementation.
//!
//! This benchmark suite measures:
//! - Throughput for varying prefix selectivity
//! - Memory allocation patterns
//! - Backend-specific performance characteristics
//! - Impact of tree depth on iteration performance
//!
//! Run with:
//! ```bash
//! cargo bench --bench prefix_zipper_benchmarks
//! ```
//!
//! For flamegraph profiling:
//! ```bash
//! RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes" \
//!   cargo flamegraph --bench prefix_zipper_benchmarks -- --bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::double_array_trie_zipper::DoubleArrayTrieZipper;
use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::dynamic_dawg_zipper::DynamicDawgZipper;
use libdictenstein::prefix_zipper::PrefixZipper;

// ============================================================================
// Test Data Generation
// ============================================================================

/// Generate test dictionary with predictable prefix distribution
fn generate_test_dict(size: usize) -> Vec<String> {
    let mut terms = Vec::with_capacity(size);

    // High selectivity prefix: "zebra" (rare start)
    for i in 0..5 {
        terms.push(format!("zebra{}", i));
    }

    // Medium selectivity prefix: "test" (common in code)
    for i in 0..50 {
        terms.push(format!("test{}", i));
        terms.push(format!("testing{}", i));
    }

    // Low selectivity prefix: "a" (very common start)
    for i in 0..200 {
        terms.push(format!("a{}", i));
        terms.push(format!("apple{}", i));
        terms.push(format!("application{}", i));
    }

    // Fill rest with random words
    for i in terms.len()..size {
        terms.push(format!("word{}", i));
    }

    terms.truncate(size);
    terms.sort();
    terms.dedup();
    terms
}

/// Generate deep tree for depth testing
fn generate_deep_tree(max_depth: usize) -> Vec<String> {
    let mut terms = Vec::new();

    // Create chain: a, aa, aaa, aaaa, ...
    for depth in 1..=max_depth {
        terms.push("a".repeat(depth));
    }

    // Create branches at each level
    for depth in 1..=max_depth {
        for suffix in ['b', 'c', 'd'] {
            let mut term = "a".repeat(depth);
            term.push(suffix);
            terms.push(term);
        }
    }

    terms
}

// ============================================================================
// Benchmark Group 1: Varying Selectivity
// ============================================================================

fn bench_selectivity_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_selectivity");

    let dict_size = 10_000;
    let terms = generate_test_dict(dict_size);
    let dict = DoubleArrayTrie::from_terms(terms.iter());

    // Test cases: (prefix, description, expected_matches)
    let test_cases = [
        ("zebra", "high_selectivity", 5),
        ("test", "medium_selectivity", 100),
        ("a", "low_selectivity", 600),
        ("", "empty_prefix", dict_size),
    ];

    for (prefix, desc, expected) in test_cases {
        group.bench_with_input(BenchmarkId::new(desc, expected), &prefix, |b, &prefix| {
            b.iter(|| {
                let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
                let count = zipper.with_prefix(prefix.as_bytes()).unwrap().count();
                black_box(count);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Benchmark Group 2: Dictionary Size Scaling
// ============================================================================

fn bench_dictionary_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("dictionary_size");

    for size in [1_000, 10_000, 100_000] {
        let terms = generate_test_dict(size);
        let dict = DoubleArrayTrie::from_terms(terms.iter());

        group.bench_with_input(
            BenchmarkId::new("medium_selectivity", size),
            &dict,
            |b, dict| {
                b.iter(|| {
                    let zipper = DoubleArrayTrieZipper::new_from_dict(dict);
                    let count = zipper.with_prefix(b"test").unwrap().count();
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark Group 3: Backend Comparison
// ============================================================================

fn bench_backend_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_comparison");

    let terms = generate_test_dict(10_000);

    // DoubleArrayTrie
    let dat = DoubleArrayTrie::from_terms(terms.iter());
    group.bench_function("DoubleArrayTrie", |b| {
        b.iter(|| {
            let zipper = DoubleArrayTrieZipper::new_from_dict(&dat);
            let count = zipper.with_prefix(b"test").unwrap().count();
            black_box(count);
        });
    });

    // DynamicDawg
    let dawg: DynamicDawg<()> = DynamicDawg::from_terms(terms.iter());
    group.bench_function("DynamicDawg", |b| {
        b.iter(|| {
            let zipper = DynamicDawgZipper::new_from_dict(&dawg);
            let count = zipper.with_prefix(b"test").unwrap().count();
            black_box(count);
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Group 4: Tree Depth Impact
// ============================================================================

fn bench_tree_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_depth");

    for max_depth in [5, 10, 15, 20] {
        let terms = generate_deep_tree(max_depth);
        let dict = DoubleArrayTrie::from_terms(terms.iter());

        group.bench_with_input(BenchmarkId::new("depth", max_depth), &dict, |b, dict| {
            b.iter(|| {
                let zipper = DoubleArrayTrieZipper::new_from_dict(dict);
                let count = zipper.with_prefix(b"a").unwrap().count();
                black_box(count);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Benchmark Group 5: Collection vs Count
// ============================================================================

fn bench_collection_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_overhead");

    let terms = generate_test_dict(10_000);
    let dict = DoubleArrayTrie::from_terms(terms.iter());

    // Just count (minimal allocation)
    group.bench_function("count_only", |b| {
        b.iter(|| {
            let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
            let count = zipper.with_prefix(b"test").unwrap().count();
            black_box(count);
        });
    });

    // Collect into Vec (allocates results)
    group.bench_function("collect_vec", |b| {
        b.iter(|| {
            let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
            let results: Vec<_> = zipper
                .with_prefix(b"test")
                .unwrap()
                .map(|(path, _)| path)
                .collect();
            black_box(results);
        });
    });

    // Collect and convert to strings (full overhead)
    group.bench_function("collect_strings", |b| {
        b.iter(|| {
            let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
            let results: Vec<String> = zipper
                .with_prefix(b"test")
                .unwrap()
                .map(|(path, _)| String::from_utf8(path).unwrap())
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Group 6: Prefix Navigation Cost
// ============================================================================

fn bench_prefix_navigation(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_navigation");

    let terms = generate_test_dict(10_000);
    let dict = DoubleArrayTrie::from_terms(terms.iter());

    // Varying prefix lengths
    for (prefix, len) in [
        ("", 0),
        ("a", 1),
        ("te", 2),
        ("tes", 3),
        ("test", 4),
        ("testi", 5),
        ("testin", 6),
        ("testing", 7),
    ] {
        group.bench_with_input(
            BenchmarkId::new("nav_length", len),
            &prefix,
            |b, &prefix| {
                b.iter(|| {
                    let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
                    // Just navigate, don't iterate
                    let result = zipper.with_prefix(prefix.as_bytes());
                    black_box(result.is_some());
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark Group 7: Iteration Only (post-navigation)
// ============================================================================

fn bench_iteration_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration_only");

    let terms = generate_test_dict(10_000);
    let dict = DoubleArrayTrie::from_terms(terms.iter());

    group.bench_function("iterate_100_results", |b| {
        b.iter(|| {
            // Clone iterator to allow repeated benchmarking
            let zipper = DoubleArrayTrieZipper::new_from_dict(&dict);
            let count = zipper.with_prefix(b"test").unwrap().count();
            black_box(count);
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_selectivity_throughput,
    bench_dictionary_size,
    bench_backend_comparison,
    bench_tree_depth,
    bench_collection_overhead,
    bench_prefix_navigation,
    bench_iteration_only,
);

criterion_main!(benches);
