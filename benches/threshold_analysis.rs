use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use liblevenshtein::prelude::*;

/// Analyze the distribution of edge counts in DAWG nodes
fn analyze_edge_distribution(dict: &DoubleArrayTrie) -> EdgeDistribution {
    let mut distribution = EdgeDistribution::new();

    // Traverse all nodes and count edge distributions
    // Note: We need to access internals, so this will analyze via node traversal
    fn traverse_and_count(node: &impl DictionaryNode, dist: &mut EdgeDistribution) {
        if let Some(edge_count) = node.edge_count() {
            dist.record(edge_count);

            for (_, child) in node.edges() {
                traverse_and_count(&child, dist);
            }
        }
    }

    traverse_and_count(&dict.root(), &mut distribution);
    distribution
}

#[derive(Default)]
struct EdgeDistribution {
    counts: Vec<usize>,
    total_nodes: usize,
}

impl EdgeDistribution {
    fn new() -> Self {
        Self {
            counts: vec![0; 32], // Support up to 32 edges
            total_nodes: 0,
        }
    }

    fn record(&mut self, edge_count: usize) {
        if edge_count < self.counts.len() {
            self.counts[edge_count] += 1;
        }
        self.total_nodes += 1;
    }

    fn print_stats(&self) {
        println!("\n=== Edge Count Distribution ===");
        println!("Total nodes: {}", self.total_nodes);
        println!("\nEdge Count | Frequency | Percentage | Cumulative");
        println!("-----------|-----------|------------|------------");

        let mut cumulative = 0;
        for (edge_count, &freq) in self.counts.iter().enumerate() {
            if freq > 0 {
                cumulative += freq;
                let pct = (freq as f64 / self.total_nodes as f64) * 100.0;
                let cum_pct = (cumulative as f64 / self.total_nodes as f64) * 100.0;
                println!(
                    "{:10} | {:9} | {:9.2}% | {:9.2}%",
                    edge_count, freq, pct, cum_pct
                );
            }
        }

        // Calculate statistics
        let median = self.calculate_percentile(50.0);
        let p75 = self.calculate_percentile(75.0);
        let p90 = self.calculate_percentile(90.0);
        let p95 = self.calculate_percentile(95.0);
        let p99 = self.calculate_percentile(99.0);

        println!("\n=== Percentiles ===");
        println!("Median (p50): {} edges", median);
        println!("p75: {} edges", p75);
        println!("p90: {} edges", p90);
        println!("p95: {} edges", p95);
        println!("p99: {} edges", p99);
    }

    fn calculate_percentile(&self, percentile: f64) -> usize {
        let target = (self.total_nodes as f64 * percentile / 100.0) as usize;
        let mut cumulative = 0;

        for (edge_count, &freq) in self.counts.iter().enumerate() {
            cumulative += freq;
            if cumulative >= target {
                return edge_count;
            }
        }

        self.counts.len() - 1
    }
}

/// Generate terms for testing
fn generate_terms(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("word{:06}", i)).collect()
}

/// Benchmark different threshold values for adaptive search
fn bench_threshold_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_threshold");

    // Test with different dictionary sizes
    for dict_size in [100, 500, 1000, 5000].iter() {
        let terms = generate_terms(*dict_size);
        let dict = DoubleArrayTrie::from_terms(terms.clone());

        // Analyze distribution for this dictionary
        println!("\n\n=== Dictionary size: {} terms ===", dict_size);
        let dist = analyze_edge_distribution(&dict);
        dist.print_stats();

        // Note: We can't directly test different thresholds without modifying the source
        // This benchmark measures the current implementation's performance
        // We'll need to modify the threshold in source code and re-run

        group.bench_with_input(
            BenchmarkId::new("contains", dict_size),
            dict_size,
            |b, _| {
                let test_terms: Vec<_> = terms.iter().step_by(10).map(|s| s.as_str()).collect();

                b.iter(|| {
                    for term in &test_terms {
                        black_box(dict.contains(term));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("edge_lookup", dict_size),
            dict_size,
            |b, _| {
                let root = dict.root();
                b.iter(|| {
                    let node = root.transition(b'w').unwrap();
                    black_box(node);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_threshold_values);
criterion_main!(benches);
