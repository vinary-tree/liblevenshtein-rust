//! Benchmarks for PersistentARTrie (Persistent Adaptive Radix Trie)
//!
//! This benchmark suite compares PersistentARTrie against other dictionary
//! implementations to measure:
//! - Construction/insertion throughput
//! - Exact lookup throughput
//! - Levenshtein query performance
//! - Memory and disk efficiency
//!
//! Run with: cargo bench --bench persistent_artrie_benchmarks --features persistent-artrie

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use libdictenstein::{
    double_array_trie::DoubleArrayTrie, dynamic_dawg::DynamicDawg, persistent_artrie::PersistentARTrie,
    Dictionary, DictionaryNode,
};
use std::hint::black_box as bb;

/// Generate realistic dictionary terms for benchmarking
fn generate_terms(size: usize) -> Vec<String> {
    let mut terms = Vec::with_capacity(size);

    // Common English prefixes and suffixes for realistic dictionary
    let prefixes = [
        "pre", "un", "re", "in", "dis", "en", "non", "over", "mis", "sub",
        "anti", "auto", "bio", "co", "counter", "de", "ex", "hyper", "inter", "multi",
    ];
    let roots = [
        "test", "code", "data", "work", "play", "read", "write", "run", "walk", "talk",
        "think", "make", "take", "give", "find", "look", "know", "want", "seem", "feel",
    ];
    let suffixes = [
        "ing", "ed", "er", "est", "ly", "ness", "ment", "tion", "able", "ful",
        "less", "ize", "ify", "ward", "wise", "ous", "ive", "al", "ary", "ory",
    ];

    // Generate realistic word combinations
    for i in 0..size {
        let prefix_idx = i % prefixes.len();
        let root_idx = (i / prefixes.len()) % roots.len();
        let suffix_idx = (i / (prefixes.len() * roots.len())) % suffixes.len();

        // Mix of word lengths
        let word = match i % 4 {
            0 => format!("{}{}", roots[root_idx], suffixes[suffix_idx]),
            1 => format!("{}{}", prefixes[prefix_idx], roots[root_idx]),
            2 => format!("{}{}{}", prefixes[prefix_idx], roots[root_idx], suffixes[suffix_idx]),
            _ => roots[root_idx].to_string(),
        };

        terms.push(word);

        // Add some numeric suffixes for variety
        if i % 10 == 0 {
            terms.push(format!("{}{}", roots[root_idx], i));
        }
    }

    terms.sort();
    terms.dedup();
    terms
}

/// Generate query terms (mix of existing and non-existing)
fn generate_queries(terms: &[String], count: usize) -> Vec<String> {
    let mut queries = Vec::with_capacity(count);

    // Half from dictionary, half are typos
    for i in 0..count {
        if i % 2 == 0 && i / 2 < terms.len() {
            queries.push(terms[i / 2].clone());
        } else {
            // Create a "typo" by appending or modifying
            let base = &terms[i % terms.len()];
            if base.len() > 2 {
                // Single character substitution
                let mut chars: Vec<char> = base.chars().collect();
                chars[1] = 'x';
                queries.push(chars.into_iter().collect());
            } else {
                queries.push(format!("{}x", base));
            }
        }
    }

    queries
}

// ============================================================================
// Construction Benchmarks
// ============================================================================

/// Benchmark PersistentARTrie construction via insertions
fn bench_part_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("part_construction");
    group.sample_size(20); // Fewer samples due to I/O

    for size in [100, 500, 1000, 5000].iter() {
        let terms = generate_terms(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("persistent_artrie", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
                    for term in &terms {
                        let _ = dict.insert(bb(term));
                    }
                    black_box(dict)
                });
            },
        );
    }
    group.finish();
}

/// Benchmark DynamicDawg construction for comparison
fn bench_dynamic_dawg_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_dawg_construction");

    for size in [100, 500, 1000, 5000].iter() {
        let terms = generate_terms(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("dynamic_dawg", size),
            size,
            |b, _| {
                b.iter(|| {
                    let dict = DynamicDawg::<()>::default();
                    for term in &terms {
                        dict.insert(bb(term));
                    }
                    black_box(dict)
                });
            },
        );
    }
    group.finish();
}

/// Benchmark DoubleArrayTrie construction for comparison
fn bench_dat_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dat_construction");

    for size in [100, 500, 1000, 5000].iter() {
        let terms = generate_terms(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("double_array_trie", size),
            size,
            |b, _| {
                b.iter(|| {
                    let dict = DoubleArrayTrie::from_terms(bb(&terms));
                    black_box(dict)
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Lookup Benchmarks
// ============================================================================

/// Benchmark PersistentARTrie exact lookup
fn bench_part_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("part_lookup");
    group.sample_size(50);

    for size in [100, 1000, 5000].iter() {
        let terms = generate_terms(*size);
        let queries = generate_queries(&terms, 100);

        let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
        for term in &terms {
            let _ = dict.insert(term);
        }

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("persistent_artrie", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut found = 0;
                    for query in &queries {
                        if dict.contains(bb(query)) {
                            found += 1;
                        }
                    }
                    black_box(found)
                });
            },
        );
    }
    group.finish();
}

/// Benchmark DynamicDawg exact lookup for comparison
fn bench_dynamic_dawg_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_dawg_lookup");

    for size in [100, 1000, 5000].iter() {
        let terms = generate_terms(*size);
        let queries = generate_queries(&terms, 100);

        let dict = DynamicDawg::<()>::default();
        for term in &terms {
            dict.insert(term);
        }

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("dynamic_dawg", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut found = 0;
                    for query in &queries {
                        if dict.contains(bb(query)) {
                            found += 1;
                        }
                    }
                    black_box(found)
                });
            },
        );
    }
    group.finish();
}

/// Benchmark DoubleArrayTrie exact lookup for comparison
fn bench_dat_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("dat_lookup");

    for size in [100, 1000, 5000].iter() {
        let terms = generate_terms(*size);
        let queries = generate_queries(&terms, 100);

        let dict = DoubleArrayTrie::from_terms(&terms);

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("double_array_trie", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut found = 0;
                    for query in &queries {
                        if dict.contains(bb(query)) {
                            found += 1;
                        }
                    }
                    black_box(found)
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Edge Traversal Benchmarks (critical for Levenshtein automata)
// ============================================================================

/// Benchmark PersistentARTrie edge traversal
fn bench_part_edge_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("part_edge_traversal");
    group.sample_size(50);

    for size in [100, 1000, 5000].iter() {
        let terms = generate_terms(*size);

        let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
        for term in &terms {
            let _ = dict.insert(term);
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("persistent_artrie", size),
            size,
            |b, _| {
                b.iter(|| {
                    // DFS traversal counting all edges
                    let mut count = 0usize;
                    let mut stack = vec![dict.root()];
                    while let Some(node) = stack.pop() {
                        for (_, child) in node.edges() {
                            count += 1;
                            stack.push(child);
                        }
                    }
                    black_box(count)
                });
            },
        );
    }
    group.finish();
}

/// Benchmark DynamicDawg edge traversal for comparison
fn bench_dynamic_dawg_edge_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_dawg_edge_traversal");

    for size in [100, 1000, 5000].iter() {
        let terms = generate_terms(*size);

        let dict = DynamicDawg::<()>::default();
        for term in &terms {
            dict.insert(term);
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("dynamic_dawg", size),
            size,
            |b, _| {
                b.iter(|| {
                    // DFS traversal counting all edges
                    let mut count = 0usize;
                    let mut stack = vec![dict.root()];
                    while let Some(node) = stack.pop() {
                        for (_, child) in node.edges() {
                            count += 1;
                            stack.push(child);
                        }
                    }
                    black_box(count)
                });
            },
        );
    }
    group.finish();
}

/// Benchmark DoubleArrayTrie edge traversal for comparison
fn bench_dat_edge_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("dat_edge_traversal");

    for size in [100, 1000, 5000].iter() {
        let terms = generate_terms(*size);
        let dict = DoubleArrayTrie::from_terms(&terms);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("double_array_trie", size),
            size,
            |b, _| {
                b.iter(|| {
                    // DFS traversal counting all edges
                    let mut count = 0usize;
                    let mut stack = vec![dict.root()];
                    while let Some(node) = stack.pop() {
                        for (_, child) in node.edges() {
                            count += 1;
                            stack.push(child);
                        }
                    }
                    black_box(count)
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Node Transition Benchmarks (single character lookup)
// ============================================================================

/// Benchmark PersistentARTrie single transitions along known paths
fn bench_part_transitions(c: &mut Criterion) {
    let mut group = c.benchmark_group("part_transitions");
    group.sample_size(100);

    for size in [100, 1000, 5000].iter() {
        let terms = generate_terms(*size);
        let queries: Vec<_> = terms.iter().take(100).collect();

        let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
        for term in &terms {
            let _ = dict.insert(term);
        }

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("persistent_artrie", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut transitions = 0usize;
                    for query in &queries {
                        let mut node = dict.root();
                        for &byte in query.as_bytes() {
                            if let Some(next) = node.transition(bb(byte)) {
                                node = next;
                                transitions += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    black_box(transitions)
                });
            },
        );
    }
    group.finish();
}

/// Benchmark DynamicDawg single transitions for comparison
fn bench_dynamic_dawg_transitions(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_dawg_transitions");

    for size in [100, 1000, 5000].iter() {
        let terms = generate_terms(*size);
        let queries: Vec<_> = terms.iter().take(100).collect();

        let dict = DynamicDawg::<()>::default();
        for term in &terms {
            dict.insert(term);
        }

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("dynamic_dawg", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut transitions = 0usize;
                    for query in &queries {
                        let mut node = dict.root();
                        for &byte in query.as_bytes() {
                            if let Some(next) = node.transition(bb(byte)) {
                                node = next;
                                transitions += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    black_box(transitions)
                });
            },
        );
    }
    group.finish();
}

/// Benchmark DoubleArrayTrie single transitions for comparison
fn bench_dat_transitions(c: &mut Criterion) {
    let mut group = c.benchmark_group("dat_transitions");

    for size in [100, 1000, 5000].iter() {
        let terms = generate_terms(*size);
        let queries: Vec<_> = terms.iter().take(100).collect();
        let dict = DoubleArrayTrie::from_terms(&terms);

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("double_array_trie", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut transitions = 0usize;
                    for query in &queries {
                        let mut node = dict.root();
                        for &byte in query.as_bytes() {
                            if let Some(next) = node.transition(bb(byte)) {
                                node = next;
                                transitions += 1;
                            } else {
                                break;
                            }
                        }
                    }
                    black_box(transitions)
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Memory Layout Benchmarks
// ============================================================================

/// Measure memory efficiency of different dictionary sizes
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.sample_size(10);

    for size in [1000, 5000, 10000].iter() {
        let terms = generate_terms(*size);

        // PersistentARTrie
        group.bench_with_input(
            BenchmarkId::new("persistent_artrie_size", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut dict: PersistentARTrie<()> = PersistentARTrie::new();
                    for term in &terms {
                        let _ = dict.insert(term);
                    }
                    // Return dict to prevent optimization
                    black_box(dict.len())
                });
            },
        );

        // DynamicDawg
        group.bench_with_input(
            BenchmarkId::new("dynamic_dawg_size", size),
            size,
            |b, _| {
                b.iter(|| {
                    let dict = DynamicDawg::<()>::default();
                    for term in &terms {
                        dict.insert(term);
                    }
                    black_box(dict.len())
                });
            },
        );

        // DoubleArrayTrie
        group.bench_with_input(
            BenchmarkId::new("double_array_trie_size", size),
            size,
            |b, _| {
                b.iter(|| {
                    let dict = DoubleArrayTrie::from_terms(&terms);
                    black_box(dict.len())
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Disk I/O Benchmarks (requires persistent-artrie feature)
// ============================================================================

/// Benchmark PersistentARTrie with disk persistence enabled
fn bench_part_disk_io(c: &mut Criterion) {
    use std::time::Instant;
    use tempfile::tempdir;

    let mut group = c.benchmark_group("part_disk_io");
    group.sample_size(10); // Fewer samples due to I/O

    for size in [100, 500, 1000].iter() {
        let terms = generate_terms(*size);

        // Benchmark: Create + Insert + Sync
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("create_insert_sync", size),
            size,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        let dir = tempdir().unwrap();
                        let path = dir.path().join("bench.part");

                        let start = Instant::now();
                        let mut dict = PersistentARTrie::<()>::create(&path).unwrap();
                        for term in &terms {
                            let _ = dict.insert(bb(term));
                        }
                        let _ = dict.sync();
                        total += start.elapsed();
                        drop(dict);
                    }
                    total
                });
            },
        );

        // Benchmark: Recovery time
        group.bench_with_input(
            BenchmarkId::new("recovery", size),
            size,
            |b, _| {
                // Setup: create and populate dictionary
                let dir = tempdir().unwrap();
                let path = dir.path().join("bench.part");
                {
                    let mut dict = PersistentARTrie::<()>::create(&path).unwrap();
                    for term in &terms {
                        let _ = dict.insert(term);
                    }
                    let _ = dict.sync();
                }

                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        let dict = PersistentARTrie::<()>::open(&path).unwrap();
                        black_box(dict.len());
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        // Benchmark: Checkpoint
        group.bench_with_input(
            BenchmarkId::new("checkpoint", size),
            size,
            |b, _| {
                let dir = tempdir().unwrap();
                let path = dir.path().join("bench.part");
                let mut dict = PersistentARTrie::<()>::create(&path).unwrap();
                for term in &terms {
                    let _ = dict.insert(term);
                }

                b.iter(|| {
                    let _ = dict.checkpoint();
                    black_box(())
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    construction_benches,
    bench_part_construction,
    bench_dynamic_dawg_construction,
    bench_dat_construction,
);

criterion_group!(
    lookup_benches,
    bench_part_lookup,
    bench_dynamic_dawg_lookup,
    bench_dat_lookup,
);

criterion_group!(
    edge_traversal_benches,
    bench_part_edge_traversal,
    bench_dynamic_dawg_edge_traversal,
    bench_dat_edge_traversal,
);

criterion_group!(
    transition_benches,
    bench_part_transitions,
    bench_dynamic_dawg_transitions,
    bench_dat_transitions,
);

criterion_group!(
    memory_benches,
    bench_memory_efficiency,
);

criterion_group!(
    disk_io_benches,
    bench_part_disk_io,
);

criterion_main!(
    construction_benches,
    lookup_benches,
    edge_traversal_benches,
    transition_benches,
    memory_benches,
    disk_io_benches,
);
