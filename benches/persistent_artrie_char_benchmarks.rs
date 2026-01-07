//! Benchmarks for PersistentARTrieChar (Persistent Adaptive Radix Trie - Character Level)
//!
//! This benchmark suite evaluates PersistentARTrieChar performance for:
//! - Unicode term construction/insertion throughput
//! - Exact lookup throughput with Unicode terms
//! - Edge traversal at character level
//! - Optimistic read performance
//! - Disk I/O with Unicode data
//!
//! Run with: cargo bench --bench persistent_artrie_char_benchmarks --features persistent-artrie

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use libdictenstein::{
    persistent_artrie_char::PersistentARTrieChar, DictionaryNode,
};
#[cfg(feature = "persistent-artrie")]
use libdictenstein::persistent_artrie_char::DiskBackedCharTrieInner;
use std::hint::black_box as bb;

/// Generate realistic Unicode dictionary terms for benchmarking
fn generate_unicode_terms(size: usize) -> Vec<String> {
    let mut terms = Vec::with_capacity(size);

    // English prefixes and suffixes
    let en_prefixes = ["pre", "un", "re", "dis", "over", "anti", "auto"];
    let en_roots = ["test", "code", "data", "work", "play", "read", "write"];
    let en_suffixes = ["ing", "ed", "er", "ly", "ness", "ment", "tion"];

    // Greek letters (common in technical text)
    let greek = ["αλφα", "βητα", "γαμμα", "δελτα", "σιγμα", "ωμεγα"];

    // Japanese words (hiragana/katakana mix)
    let japanese = ["にほん", "コンピュータ", "データ", "プログラム", "テスト"];

    // Chinese words
    let chinese = ["数据", "程序", "测试", "代码", "编程", "算法"];

    // Emoji sequences
    let emoji = ["🚀", "⭐", "🎉", "💡", "🔥", "✨"];

    for i in 0..size {
        let term = match i % 6 {
            0 => {
                // English word combinations
                let prefix_idx = i % en_prefixes.len();
                let root_idx = (i / en_prefixes.len()) % en_roots.len();
                let suffix_idx = (i / (en_prefixes.len() * en_roots.len())) % en_suffixes.len();
                format!(
                    "{}{}{}",
                    en_prefixes[prefix_idx], en_roots[root_idx], en_suffixes[suffix_idx]
                )
            }
            1 => {
                // Greek terms
                let idx = i % greek.len();
                format!("{}{}", greek[idx], i)
            }
            2 => {
                // Japanese terms
                let idx = i % japanese.len();
                format!("{}{}", japanese[idx], i)
            }
            3 => {
                // Chinese terms
                let idx = i % chinese.len();
                format!("{}{}", chinese[idx], i)
            }
            4 => {
                // Mixed script
                let en_idx = i % en_roots.len();
                let emoji_idx = i % emoji.len();
                format!("{}{}{}", en_roots[en_idx], emoji[emoji_idx], i)
            }
            _ => {
                // Plain ASCII for baseline comparison
                format!("term{:06}", i)
            }
        };
        terms.push(term);
    }

    terms.sort();
    terms.dedup();
    terms
}

/// Generate query terms (mix of existing and non-existing)
fn generate_queries(terms: &[String], count: usize) -> Vec<String> {
    let mut queries = Vec::with_capacity(count);

    for i in 0..count {
        if i % 2 == 0 && i / 2 < terms.len() {
            queries.push(terms[i / 2].clone());
        } else {
            // Create a "typo" by appending a character
            let base = &terms[i % terms.len()];
            queries.push(format!("{}x", base));
        }
    }

    queries
}

// ============================================================================
// Construction Benchmarks
// ============================================================================

/// Benchmark PersistentARTrieChar construction via insertions
fn bench_char_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_construction");
    group.sample_size(20);

    for size in [100, 500, 1000, 5000].iter() {
        let terms = generate_unicode_terms(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("persistent_artrie_char", size),
            size,
            |b, _| {
                b.iter(|| {
                    let dict: PersistentARTrieChar<()> = PersistentARTrieChar::new();
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

/// Benchmark construction with pure ASCII (for comparison)
fn bench_char_construction_ascii(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_construction_ascii");
    group.sample_size(20);

    for size in [100, 500, 1000, 5000].iter() {
        // Generate ASCII-only terms
        let terms: Vec<String> = (0..*size).map(|i| format!("term{:06}", i)).collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("persistent_artrie_char_ascii", size),
            size,
            |b, _| {
                b.iter(|| {
                    let dict: PersistentARTrieChar<()> = PersistentARTrieChar::new();
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

// ============================================================================
// Lookup Benchmarks
// ============================================================================

/// Benchmark PersistentARTrieChar exact lookup with Unicode
fn bench_char_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_lookup");
    group.sample_size(50);

    for size in [100, 1000, 5000].iter() {
        let terms = generate_unicode_terms(*size);
        let queries = generate_queries(&terms, 100);

        let dict: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        for term in &terms {
            dict.insert(term);
        }

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("persistent_artrie_char", size),
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

/// Benchmark lookup with CJK characters (multibyte)
fn bench_char_lookup_cjk(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_lookup_cjk");
    group.sample_size(50);

    // Generate CJK-heavy terms
    let chinese_chars: Vec<char> = "数据结构算法程序代码测试编程开发".chars().collect();
    let terms: Vec<String> = (0..1000)
        .map(|i| {
            let mut s = String::new();
            for j in 0..5 {
                s.push(chinese_chars[(i + j) % chinese_chars.len()]);
            }
            s
        })
        .collect();

    let queries: Vec<String> = terms.iter().take(100).cloned().collect();

    let dict: PersistentARTrieChar<()> = PersistentARTrieChar::new();
    for term in &terms {
        dict.insert(term);
    }

    group.throughput(Throughput::Elements(100));
    group.bench_function("cjk_lookup", |b| {
        b.iter(|| {
            let mut found = 0;
            for query in &queries {
                if dict.contains(bb(query)) {
                    found += 1;
                }
            }
            black_box(found)
        });
    });
    group.finish();
}

// ============================================================================
// Edge Traversal Benchmarks (critical for Levenshtein automata)
// ============================================================================

/// Benchmark PersistentARTrieChar edge traversal
fn bench_char_edge_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_edge_traversal");
    group.sample_size(50);

    for size in [100, 1000, 5000].iter() {
        let terms = generate_unicode_terms(*size);

        let dict: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        for term in &terms {
            dict.insert(term);
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("persistent_artrie_char", size),
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
// Node Transition Benchmarks (character-level lookup)
// ============================================================================

/// Benchmark PersistentARTrieChar single character transitions
fn bench_char_transitions(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_transitions");
    group.sample_size(100);

    for size in [100, 1000, 5000].iter() {
        let terms = generate_unicode_terms(*size);
        let queries: Vec<_> = terms.iter().take(100).collect();

        let dict: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        for term in &terms {
            dict.insert(term);
        }

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("persistent_artrie_char", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut transitions = 0usize;
                    for query in &queries {
                        let mut node = dict.root();
                        for ch in query.chars() {
                            if let Some(next) = node.transition(bb(ch)) {
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

/// Benchmark transitions with emoji (4-byte UTF-8 / supplementary plane)
fn bench_char_transitions_emoji(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_transitions_emoji");
    group.sample_size(50);

    // Generate emoji sequences
    let emojis = ["🚀", "🎉", "💡", "🔥", "⭐", "✨", "🎊", "🎯", "🏆", "💻"];
    let terms: Vec<String> = (0..500)
        .map(|i| {
            let mut s = String::new();
            for j in 0..4 {
                s.push_str(emojis[(i + j) % emojis.len()]);
            }
            s
        })
        .collect();

    let queries: Vec<_> = terms.iter().take(50).collect();

    let dict: PersistentARTrieChar<()> = PersistentARTrieChar::new();
    for term in &terms {
        dict.insert(term);
    }

    group.throughput(Throughput::Elements(50));
    group.bench_function("emoji_transitions", |b| {
        b.iter(|| {
            let mut transitions = 0usize;
            for query in &queries {
                let mut node = dict.root();
                for ch in query.chars() {
                    if let Some(next) = node.transition(bb(ch)) {
                        node = next;
                        transitions += 1;
                    } else {
                        break;
                    }
                }
            }
            black_box(transitions)
        });
    });
    group.finish();
}

// ============================================================================
// Iterator Benchmarks
// ============================================================================

/// Benchmark term iteration
fn bench_char_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_iteration");
    group.sample_size(30);

    for size in [100, 500, 1000].iter() {
        let terms = generate_unicode_terms(*size);

        let dict: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        for term in &terms {
            dict.insert(term);
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("iter", size),
            size,
            |b, _| {
                b.iter(|| {
                    let count = dict.iter().count();
                    black_box(count)
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Optimistic Read Benchmarks (Phase C7 concurrency feature)
// ============================================================================

/// Benchmark optimistic contains operations
#[cfg(feature = "persistent-artrie")]
fn bench_char_optimistic_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_optimistic_reads");
    group.sample_size(50);

    for size in [100, 1000, 5000].iter() {
        let terms = generate_unicode_terms(*size);
        let queries = generate_queries(&terms, 100);

        let dict: PersistentARTrieChar<()> = PersistentARTrieChar::new();
        for term in &terms {
            dict.insert(term);
        }

        // Regular contains
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("regular_contains", size),
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
// Memory Efficiency Benchmarks
// ============================================================================

/// Measure memory efficiency with Unicode terms
fn bench_char_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_memory_efficiency");
    group.sample_size(10);

    for size in [1000, 5000, 10000].iter() {
        let terms = generate_unicode_terms(*size);

        group.bench_with_input(
            BenchmarkId::new("persistent_artrie_char_size", size),
            size,
            |b, _| {
                b.iter(|| {
                    let dict: PersistentARTrieChar<()> = PersistentARTrieChar::new();
                    for term in &terms {
                        dict.insert(term);
                    }
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

/// Benchmark DiskBackedCharTrieInner with disk persistence enabled
#[cfg(feature = "persistent-artrie")]
fn bench_char_disk_io(c: &mut Criterion) {
    use std::time::Instant;
    use tempfile::tempdir;

    let mut group = c.benchmark_group("char_disk_io");
    group.sample_size(10);

    for size in [100, 500, 1000].iter() {
        let terms = generate_unicode_terms(*size);

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
                        let path = dir.path().join("bench.chartrie");

                        let start = Instant::now();
                        let mut dict =
                            DiskBackedCharTrieInner::<()>::create(&path).expect("create dict");
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
                let path = dir.path().join("bench.chartrie");
                {
                    let mut dict = DiskBackedCharTrieInner::<()>::create(&path).expect("create dict");
                    for term in &terms {
                        let _ = dict.insert(term);
                    }
                    let _ = dict.sync();
                }

                b.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        let dict = DiskBackedCharTrieInner::<()>::open(&path).expect("open dict");
                        black_box(dict.len);
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
                let path = dir.path().join("bench.chartrie");
                let mut dict = DiskBackedCharTrieInner::<()>::create(&path).expect("create dict");
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

/// Benchmark disk I/O with CJK-heavy data
#[cfg(feature = "persistent-artrie")]
fn bench_char_disk_io_cjk(c: &mut Criterion) {
    use std::time::Instant;
    use tempfile::tempdir;

    let mut group = c.benchmark_group("char_disk_io_cjk");
    group.sample_size(10);

    // Generate CJK-heavy terms
    let chinese_chars: Vec<char> = "数据结构算法程序代码测试编程开发系统网络".chars().collect();
    let terms: Vec<String> = (0..500)
        .map(|i| {
            let mut s = String::new();
            for j in 0..6 {
                s.push(chinese_chars[(i + j) % chinese_chars.len()]);
            }
            s
        })
        .collect();

    group.throughput(Throughput::Elements(500));
    group.bench_function("cjk_create_insert_sync", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let dir = tempdir().unwrap();
                let path = dir.path().join("bench.chartrie");

                let start = Instant::now();
                let mut dict = DiskBackedCharTrieInner::<()>::create(&path).expect("create dict");
                for term in &terms {
                    let _ = dict.insert(bb(term));
                }
                let _ = dict.sync();
                total += start.elapsed();
                drop(dict);
            }
            total
        });
    });
    group.finish();
}

// ============================================================================
// Atomic Operations Benchmarks
// ============================================================================

/// Benchmark atomic increment operations
#[cfg(feature = "persistent-artrie")]
fn bench_char_atomic_ops(c: &mut Criterion) {
    use tempfile::tempdir;

    let mut group = c.benchmark_group("char_atomic_ops");
    group.sample_size(30);

    // Test increment performance
    let terms: Vec<String> = (0..100).map(|i| format!("counter_{}", i)).collect();

    group.throughput(Throughput::Elements(100));
    group.bench_function("increment", |b| {
        let dir = tempdir().unwrap();
        let path = dir.path().join("atomic.chartrie");
        let mut dict = DiskBackedCharTrieInner::<i64>::create(&path).expect("create dict");

        // Pre-populate
        for term in &terms {
            let _ = dict.increment(term, 0);
        }

        b.iter(|| {
            for term in &terms {
                let _ = dict.increment(bb(term), 1);
            }
            black_box(())
        });
    });

    group.bench_function("upsert", |b| {
        let dir = tempdir().unwrap();
        let path = dir.path().join("upsert.chartrie");
        let mut dict = DiskBackedCharTrieInner::<i64>::create(&path).expect("create dict");

        b.iter(|| {
            for (i, term) in terms.iter().enumerate() {
                let _ = dict.upsert(bb(term), i as i64);
            }
            black_box(())
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    construction_benches,
    bench_char_construction,
    bench_char_construction_ascii,
);

criterion_group!(
    lookup_benches,
    bench_char_lookup,
    bench_char_lookup_cjk,
);

criterion_group!(
    edge_traversal_benches,
    bench_char_edge_traversal,
);

criterion_group!(
    transition_benches,
    bench_char_transitions,
    bench_char_transitions_emoji,
);

criterion_group!(
    iteration_benches,
    bench_char_iteration,
);

criterion_group!(
    memory_benches,
    bench_char_memory_efficiency,
);

#[cfg(feature = "persistent-artrie")]
criterion_group!(
    optimistic_benches,
    bench_char_optimistic_reads,
);

#[cfg(feature = "persistent-artrie")]
criterion_group!(
    disk_io_benches,
    bench_char_disk_io,
    bench_char_disk_io_cjk,
);

#[cfg(feature = "persistent-artrie")]
criterion_group!(
    atomic_benches,
    bench_char_atomic_ops,
);

#[cfg(feature = "persistent-artrie")]
criterion_main!(
    construction_benches,
    lookup_benches,
    edge_traversal_benches,
    transition_benches,
    iteration_benches,
    memory_benches,
    optimistic_benches,
    disk_io_benches,
    atomic_benches,
);

#[cfg(not(feature = "persistent-artrie"))]
criterion_main!(
    construction_benches,
    lookup_benches,
    edge_traversal_benches,
    transition_benches,
    iteration_benches,
    memory_benches,
);
