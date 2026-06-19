use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::prelude::*;

/// Parameterized DAWG for testing different thresholds
/// We'll manually implement the search with different thresholds
#[allow(dead_code)]
fn search_with_threshold(nodes: &[DawgNode], term: &str, threshold: usize) -> bool {
    let mut node_idx = 0;

    for &byte in term.as_bytes() {
        let edges = &nodes[node_idx].edges;

        let next_idx = if edges.len() < threshold {
            // Linear search
            edges.iter().find(|(l, _)| *l == byte).map(|(_, idx)| *idx)
        } else {
            // Binary search
            edges
                .binary_search_by_key(&byte, |(l, _)| *l)
                .ok()
                .map(|pos| edges[pos].1)
        };

        match next_idx {
            Some(idx) => node_idx = idx,
            None => return false,
        }
    }

    nodes[node_idx].is_final
}

// Expose DawgNode for benchmarking (normally private)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DawgNode {
    pub edges: Vec<(u8, usize)>,
    pub is_final: bool,
}

/// Extract nodes from DoubleArrayTrie for direct testing
/// Rebuilds a benchmark-local node view so threshold variants can be compared
/// without exposing DoubleArrayTrie internals.
#[allow(dead_code)]
fn extract_nodes(dict: &DoubleArrayTrie) -> Vec<DawgNode> {
    // We can't directly access the internal arrays since they're private
    // Instead, we'll traverse and rebuild
    let mut nodes = Vec::new();
    let mut visited = std::collections::HashMap::new();

    fn visit_node(
        node: &impl DictionaryNode<Unit = u8>,
        nodes: &mut Vec<DawgNode>,
        visited: &mut std::collections::HashMap<usize, usize>,
        node_id: usize,
    ) -> usize {
        if let Some(&idx) = visited.get(&node_id) {
            return idx;
        }

        let idx = nodes.len();
        visited.insert(node_id, idx);

        // Insert the node before visiting children so shared descendants can
        // refer back to the already allocated benchmark-local index.
        nodes.push(DawgNode {
            edges: Vec::new(),
            is_final: node.is_final(),
        });

        // Collect edges
        let edges: Vec<_> = node
            .edges()
            .map(|(label, child)| {
                let child_id = node_id * 256 + label as usize; // Fake ID for uniqueness
                let child_idx = visit_node(&child, nodes, visited, child_id);
                (label, child_idx)
            })
            .collect();

        nodes[idx].edges = edges;
        idx
    }

    visit_node(&dict.root(), &mut nodes, &mut visited, 0);
    nodes
}

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
