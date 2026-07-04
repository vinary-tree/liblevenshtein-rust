use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::prelude::*;
use std::hint::black_box;

/// Generate test terms
fn generate_terms(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("word{:06}", i)).collect()
}

/// Benchmark different threshold values
fn bench_threshold_values(c: &mut Criterion) {
    let dict_sizes = [500, 1000, 5000];
    let _thresholds = [2, 4, 6, 8, 10, 12, 16, 32];

    for &dict_size in &dict_sizes {
        let terms = generate_terms(dict_size);
        let dict = DoubleArrayTrie::from_terms(terms.iter().map(|s| s.as_str()));

        // NOTE: We can't extract nodes directly, so we'll test with the actual contains()
        // which uses threshold=8. For true threshold testing, we'd need to modify source code.
        //
        // Instead, let's create a more realistic dictionary with varied edge counts

        let mut group = c.benchmark_group(format!("threshold_impact_{}", dict_size));
        group.throughput(Throughput::Elements(dict_size as u64));

        // Benchmark current implementation (threshold=8)
        group.bench_function(BenchmarkId::new("current_impl", "threshold_8"), |b| {
            let test_terms: Vec<_> = terms.iter().step_by(10).map(|s| s.as_str()).collect();

            b.iter(|| {
                for term in &test_terms {
                    black_box(dict.contains(term));
                }
            });
        });

        group.finish();
    }

    // Since we can't easily test different thresholds without modifying source,
    // let's instead measure the performance of linear vs binary search directly
    bench_search_strategies(c);
}

/// Benchmark linear vs binary search on different edge counts
fn bench_search_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_strategy_comparison");

    // Test with different edge counts to find crossover point
    for edge_count in [2, 4, 6, 8, 10, 12, 16, 20, 26].iter() {
        let edges: Vec<(u8, usize)> = (0..*edge_count as u8)
            .map(|i| (b'a' + i, i as usize))
            .collect();

        // Test search for middle element
        let target = b'a' + (*edge_count as u8 / 2);

        group.bench_with_input(
            BenchmarkId::new("linear_search", edge_count),
            edge_count,
            |b, _| {
                b.iter(|| {
                    let result = edges.iter().find(|(l, _)| *l == target);
                    black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_search", edge_count),
            edge_count,
            |b, _| {
                b.iter(|| {
                    let result = edges.binary_search_by_key(&target, |(l, _)| *l);
                    let _ = black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_threshold_values);
criterion_main!(benches);
