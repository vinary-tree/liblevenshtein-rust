//! Benchmarks for SCDAWG optimization experiments.
//!
//! This benchmark suite measures baseline performance and evaluates
//! potential optimizations (bloom filter, SIMD) for SCDAWG edge lookup.
//!
//! Run with:
//!   cargo bench --bench scdawg_optimization_benchmarks
//!
//! Run with features:
//!   cargo bench --bench scdawg_optimization_benchmarks --features scdawg-bloom
//!   cargo bench --bench scdawg_optimization_benchmarks --features scdawg-simd

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::dictionary::scdawg::Scdawg;
use liblevenshtein::dictionary::substring::SubstringDictionary;
use liblevenshtein::dictionary::{Dictionary, DictionaryNode};
use std::collections::HashMap;

// ============================================================================
// Test Data Setup
// ============================================================================

/// Load dictionary from system words file with configurable size
fn load_dictionary(target_size: usize) -> Vec<String> {
    if let Ok(contents) = std::fs::read_to_string("/usr/share/dict/words") {
        contents
            .lines()
            .map(|s| s.trim().to_lowercase())
            .filter(|s| s.len() >= 3 && s.len() <= 20 && s.chars().all(|c| c.is_ascii_alphabetic()))
            .take(target_size)
            .collect()
    } else {
        generate_synthetic_dictionary(target_size)
    }
}

/// Generate synthetic dictionary for systems without /usr/share/dict/words
fn generate_synthetic_dictionary(size: usize) -> Vec<String> {
    use std::collections::HashSet;

    let base_words = [
        "algorithm", "structure", "computer", "science", "program",
        "function", "variable", "constant", "iterator", "reference",
        "pattern", "matching", "distance", "automaton", "transducer",
        "dictionary", "benchmark", "performance", "optimization", "implementation",
        "cathedral", "category", "catering", "catastrophe", "catalyst",
        "application", "approximation", "acceleration", "authentication", "authorization",
    ];

    let suffixes = ["", "s", "ed", "ing", "er", "est", "ly", "tion", "ment", "ness"];
    let prefixes = ["", "un", "re", "pre", "mis", "dis", "over", "under", "out", "sub"];

    let mut words = HashSet::new();

    for base in &base_words {
        for prefix in &prefixes {
            for suffix in &suffixes {
                let word = format!("{}{}{}", prefix, base, suffix);
                if word.len() >= 3 && word.len() <= 20 {
                    words.insert(word);
                }
                if words.len() >= size {
                    return words.into_iter().collect();
                }
            }
        }
    }

    // If still need more words, generate numbered variants
    let mut i = 0;
    while words.len() < size {
        words.insert(format!("word{:06}", i));
        i += 1;
    }

    words.into_iter().collect()
}

/// Generate a query pattern of specified length
fn generate_query(length: usize, seed: usize) -> String {
    let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
    (0..length)
        .map(|i| {
            let idx = (seed * 31 + i * 17) % chars.len();
            chars[idx]
        })
        .collect()
}

/// Generate a realistic query based on a dictionary word
fn generate_realistic_query(dict: &[String], seed: usize, target_len: usize) -> String {
    if dict.is_empty() {
        return generate_query(target_len, seed);
    }

    let word = &dict[seed % dict.len()];
    if word.len() >= target_len {
        word.chars().take(target_len).collect()
    } else {
        word.clone()
    }
}

// ============================================================================
// Edge Distribution Analysis
// ============================================================================

/// Analyze edge count distribution across all SCDAWG nodes.
///
/// This is crucial for understanding whether SIMD (which typically benefits
/// from 12+ edges) is applicable to SCDAWG nodes.
fn analyze_edge_distribution(scdawg: &Scdawg<()>) -> EdgeDistributionStats {
    let mut edge_counts: HashMap<usize, usize> = HashMap::new();
    let mut total_nodes = 0usize;
    let mut total_edges = 0usize;

    // Walk all nodes via BFS from root
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(scdawg.root());

    while let Some(node) = queue.pop_front() {
        // Create a simple unique identifier for the node
        let node_id = format!("{:?}", &node);
        if visited.contains(&node_id) {
            continue;
        }
        visited.insert(node_id);
        total_nodes += 1;

        let edge_count = node.edge_count().unwrap_or(0);
        total_edges += edge_count;
        *edge_counts.entry(edge_count).or_insert(0) += 1;

        // Queue children
        for (_, child) in node.edges() {
            queue.push_back(child);
        }
    }

    EdgeDistributionStats {
        total_nodes,
        total_edges,
        edge_counts,
        avg_edges: if total_nodes > 0 {
            total_edges as f64 / total_nodes as f64
        } else {
            0.0
        },
    }
}

#[derive(Debug)]
struct EdgeDistributionStats {
    total_nodes: usize,
    total_edges: usize,
    edge_counts: HashMap<usize, usize>,
    avg_edges: f64,
}

impl EdgeDistributionStats {
    fn print_summary(&self) {
        eprintln!("\n=== Edge Distribution Summary ===");
        eprintln!("Total nodes: {}", self.total_nodes);
        eprintln!("Total edges: {}", self.total_edges);
        eprintln!("Avg edges/node: {:.2}", self.avg_edges);
        eprintln!("\nDistribution:");

        let mut counts: Vec<_> = self.edge_counts.iter().collect();
        counts.sort_by_key(|(k, _)| *k);

        for (edge_count, node_count) in counts.iter().take(20) {
            let pct = **node_count as f64 / self.total_nodes as f64 * 100.0;
            eprintln!("  {} edges: {} nodes ({:.1}%)", edge_count, node_count, pct);
        }

        // SIMD threshold analysis
        let nodes_12_plus: usize = self.edge_counts
            .iter()
            .filter(|(&k, _)| k >= 12)
            .map(|(_, &v)| v)
            .sum();
        let pct_12_plus = nodes_12_plus as f64 / self.total_nodes as f64 * 100.0;
        eprintln!("\nNodes with 12+ edges (SIMD threshold): {} ({:.1}%)", nodes_12_plus, pct_12_plus);

        // 4 or fewer (SmallVec inline)
        let nodes_4_or_less: usize = self.edge_counts
            .iter()
            .filter(|(&k, _)| k <= 4)
            .map(|(_, &v)| v)
            .sum();
        let pct_4_or_less = nodes_4_or_less as f64 / self.total_nodes as f64 * 100.0;
        eprintln!("Nodes with ≤4 edges (SmallVec inline): {} ({:.1}%)", nodes_4_or_less, pct_4_or_less);
    }
}

// ============================================================================
// Hit/Miss Ratio Analysis
// ============================================================================

/// Measure the hit/miss ratio for edge lookups during substring search.
///
/// This determines whether a bloom filter (which helps on misses) would be beneficial.
fn measure_hit_miss_ratio(scdawg: &Scdawg<()>, patterns: &[String]) -> HitMissStats {
    let mut hits = 0usize;
    let mut misses = 0usize;

    for pattern in patterns {
        let mut current = scdawg.root();
        for &byte in pattern.as_bytes() {
            match current.transition(byte) {
                Some(next) => {
                    hits += 1;
                    current = next;
                }
                None => {
                    misses += 1;
                    break; // Pattern not found, stop here
                }
            }
        }
    }

    HitMissStats { hits, misses }
}

#[derive(Debug)]
struct HitMissStats {
    hits: usize,
    misses: usize,
}

impl HitMissStats {
    fn total(&self) -> usize {
        self.hits + self.misses
    }

    fn hit_ratio(&self) -> f64 {
        if self.total() == 0 {
            0.0
        } else {
            self.hits as f64 / self.total() as f64
        }
    }

    fn miss_ratio(&self) -> f64 {
        if self.total() == 0 {
            0.0
        } else {
            self.misses as f64 / self.total() as f64
        }
    }

    fn print_summary(&self) {
        eprintln!("\n=== Hit/Miss Ratio Summary ===");
        eprintln!("Total lookups: {}", self.total());
        eprintln!("Hits: {} ({:.1}%)", self.hits, self.hit_ratio() * 100.0);
        eprintln!("Misses: {} ({:.1}%)", self.misses, self.miss_ratio() * 100.0);
    }
}

// ============================================================================
// Baseline Benchmarks
// ============================================================================

/// Benchmark baseline substring search performance.
fn bench_substring_search_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("scdawg_substring_baseline");
    group.sample_size(100);

    for dict_size in [10_000, 50_000, 88_996] {
        let dict_words = load_dictionary(dict_size);
        let actual_size = dict_words.len();

        if actual_size < dict_size / 2 {
            eprintln!("Warning: Only loaded {} words for target {}", actual_size, dict_size);
            continue;
        }

        let scdawg = Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));

        // Print distribution stats for first run only
        if dict_size == 10_000 {
            let stats = analyze_edge_distribution(&scdawg);
            stats.print_summary();
        }

        // Generate patterns of varying lengths
        for pattern_len in [5, 10, 15, 20] {
            let patterns: Vec<String> = (0..100)
                .map(|i| generate_realistic_query(&dict_words, i * 1009, pattern_len))
                .collect();

            // Print hit/miss stats for first configuration
            if dict_size == 10_000 && pattern_len == 10 {
                let hm_stats = measure_hit_miss_ratio(&scdawg, &patterns);
                hm_stats.print_summary();
            }

            let id = format!("d{}_p{}", actual_size, pattern_len);
            group.throughput(Throughput::Elements(patterns.len() as u64));

            group.bench_function(BenchmarkId::new("find_substring", &id), |b| {
                b.iter(|| {
                    let mut found = 0usize;
                    for pattern in &patterns {
                        if black_box(scdawg.contains_substring(pattern)) {
                            found += 1;
                        }
                    }
                    found
                })
            });
        }
    }

    group.finish();
}

/// Benchmark isolated edge lookup (get_edge) microbenchmark.
///
/// This directly measures the performance of edge lookup, which is the
/// target for bloom filter and SIMD optimization.
fn bench_edge_lookup_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("scdawg_edge_lookup_baseline");
    group.sample_size(200);

    let dict_words = load_dictionary(50_000);
    let scdawg = Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));

    // Get all possible labels (a-z)
    let labels: Vec<u8> = (b'a'..=b'z').collect();

    // Benchmark edge lookup from root (high fanout)
    let root = scdawg.root();
    group.bench_function("root_edge_lookup", |b| {
        b.iter(|| {
            let mut found = 0usize;
            for &label in &labels {
                if black_box(root.transition(label)).is_some() {
                    found += 1;
                }
            }
            found
        })
    });

    // Generate random patterns and measure edge lookups along paths
    let patterns: Vec<String> = (0..100)
        .map(|i| generate_realistic_query(&dict_words, i * 997, 10))
        .collect();

    group.bench_function("path_edge_lookups", |b| {
        b.iter(|| {
            let mut total_lookups = 0usize;
            for pattern in &patterns {
                let mut current = scdawg.root();
                for &byte in pattern.as_bytes() {
                    total_lookups += 1;
                    match current.transition(byte) {
                        Some(next) => current = next,
                        None => break,
                    }
                }
            }
            total_lookups
        })
    });

    // Benchmark misses specifically (looking for non-existent edges)
    group.bench_function("miss_edge_lookups", |b| {
        b.iter(|| {
            let mut total = 0usize;
            let root = scdawg.root();
            // Look for digits (which shouldn't exist in alpha dictionary)
            for digit in b'0'..=b'9' {
                if black_box(root.transition(digit)).is_none() {
                    total += 1;
                }
            }
            total
        })
    });

    group.finish();
}

/// Benchmark WallBreaker end-to-end performance (baseline).
fn bench_wallbreaker_baseline(c: &mut Criterion) {
    use liblevenshtein::wallbreaker::WallBreaker;

    let mut group = c.benchmark_group("wallbreaker_baseline");
    group.sample_size(50);

    let configs = [
        (10_000, 2, 20, "d10k_k2_q20"),
        (10_000, 4, 50, "d10k_k4_q50"),
        (50_000, 4, 50, "d50k_k4_q50"),
    ];

    for (dict_size, max_distance, query_len, label) in configs {
        let dict_words = load_dictionary(dict_size);
        let actual_size = dict_words.len();

        if actual_size < dict_size / 2 {
            continue;
        }

        let scdawg = Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
        let wb = WallBreaker::new(&scdawg, max_distance);

        let queries: Vec<String> = (0..10)
            .map(|i| generate_realistic_query(&dict_words, i * 7919, query_len))
            .collect();

        group.throughput(Throughput::Elements(queries.len() as u64));

        group.bench_function(BenchmarkId::new("wallbreaker", label), |b| {
            b.iter(|| {
                let mut total = 0usize;
                for query in &queries {
                    total += black_box(wb.query(query).count());
                }
                total
            })
        });
    }

    group.finish();
}

/// Print detailed edge distribution analysis (run as a "benchmark").
fn bench_edge_distribution_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_distribution_analysis");
    group.sample_size(10); // Just run once to print stats

    for dict_size in [10_000, 50_000, 88_996] {
        let dict_words = load_dictionary(dict_size);
        let actual_size = dict_words.len();

        if actual_size < dict_size / 2 {
            continue;
        }

        let scdawg = Scdawg::<()>::from_terms(dict_words.iter().map(|s| s.as_str()));
        let stats = analyze_edge_distribution(&scdawg);

        eprintln!("\n\n========================================");
        eprintln!("EDGE DISTRIBUTION FOR {} WORDS", actual_size);
        stats.print_summary();

        // Also measure hit/miss for realistic queries
        let patterns: Vec<String> = (0..1000)
            .map(|i| generate_realistic_query(&dict_words, i * 997, 10))
            .collect();
        let hm_stats = measure_hit_miss_ratio(&scdawg, &patterns);
        hm_stats.print_summary();

        // Random patterns (more misses expected)
        let random_patterns: Vec<String> = (0..1000)
            .map(|i| generate_query(10, i * 1009))
            .collect();
        eprintln!("\nHit/Miss for RANDOM patterns (len 10):");
        let random_hm_stats = measure_hit_miss_ratio(&scdawg, &random_patterns);
        random_hm_stats.print_summary();

        let id = format!("d{}", actual_size);
        group.bench_function(BenchmarkId::new("analysis", &id), |b| {
            b.iter(|| {
                // Just iterate over nodes to have something to measure
                black_box(scdawg.term_count())
            })
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    baseline_benches,
    bench_substring_search_baseline,
    bench_edge_lookup_baseline,
    bench_wallbreaker_baseline,
);

criterion_group!(
    analysis_benches,
    bench_edge_distribution_analysis,
);

criterion_main!(baseline_benches, analysis_benches);
