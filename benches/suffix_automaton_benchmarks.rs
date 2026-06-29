//! Benchmarks for SuffixAutomaton construction and query performance.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::suffix_automaton::SuffixAutomaton;
use liblevenshtein::transducer::{Algorithm, Transducer};

/// Generate test corpus of varying sizes
fn generate_corpus(size: usize) -> Vec<String> {
    (0..size)
        .map(|i| {
            // Create texts of varying lengths (20-100 characters)
            let len = 20 + (i % 80);
            let base_text = "The quick brown fox jumps over the lazy dog. ";
            let repeat = (len / base_text.len()) + 1;
            format!("{}_{}", base_text.repeat(repeat), i)[..len].to_string()
        })
        .collect()
}

/// Benchmark: SuffixAutomaton construction from texts
fn bench_construction_varying_corpus_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("suffix_automaton_construction");

    for size in [10, 50, 100, 500].iter() {
        let texts = generate_corpus(*size);
        let total_chars: usize = texts.iter().map(|t| t.len()).sum();

        group.throughput(Throughput::Bytes(total_chars as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let dict: SuffixAutomaton<()> =
                    SuffixAutomaton::from_texts(black_box(texts.clone()));
                black_box(dict);
            });
        });
    }
    group.finish();
}

/// Benchmark: SuffixAutomaton incremental insertion
fn bench_incremental_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("suffix_automaton_incremental_insertion");

    for size in [10, 50, 100, 500].iter() {
        let texts = generate_corpus(*size);
        let total_chars: usize = texts.iter().map(|t| t.len()).sum();

        group.throughput(Throughput::Bytes(total_chars as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let dict: SuffixAutomaton<()> = SuffixAutomaton::new();
                for text in black_box(&texts) {
                    dict.insert(text);
                }
                black_box(dict);
            });
        });
    }
    group.finish();
}

/// Benchmark: Substring query performance with varying corpus sizes
fn bench_query_varying_corpus_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("suffix_automaton_query_corpus_size");

    for size in [10, 50, 100, 500].iter() {
        let texts = generate_corpus(*size);
        let dict: SuffixAutomaton<()> = SuffixAutomaton::from_texts(texts.clone());
        let transducer = Transducer::new(dict, Algorithm::Standard);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query(black_box("quick brown"), black_box(1))
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Substring query performance with varying distances
fn bench_query_varying_distance(c: &mut Criterion) {
    let texts = generate_corpus(100);
    let dict: SuffixAutomaton<()> = SuffixAutomaton::from_texts(texts);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("suffix_automaton_query_distance");

    for distance in [0, 1, 2, 3].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(distance), distance, |b, &d| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query(black_box("quick brown fox"), black_box(d))
                    .collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Substring query performance with varying query lengths
fn bench_query_varying_query_length(c: &mut Criterion) {
    let text =
        "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
    let dict: SuffixAutomaton<()> = SuffixAutomaton::from_text(text);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("suffix_automaton_query_length");

    let queries = [
        ("fox", 3),
        ("brown fox", 9),
        ("quick brown fox jumps", 21),
        ("The quick brown fox jumps over", 30),
    ];

    for (query, len) in queries.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(len), len, |b, _| {
            b.iter(|| {
                let results: Vec<_> = transducer.query(black_box(query), black_box(1)).collect();
                black_box(results);
            });
        });
    }
    group.finish();
}

/// Benchmark: Ordered query performance
fn bench_ordered_query(c: &mut Criterion) {
    let texts = generate_corpus(100);
    let dict: SuffixAutomaton<()> = SuffixAutomaton::from_texts(texts);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("suffix_automaton_ordered_query");

    group.bench_function("ordered_query_distance_2", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query_ordered(black_box("quick brown"), black_box(2))
                .take(10) // Take first 10 results
                .collect();
            black_box(results);
        });
    });

    group.bench_function("standard_query_distance_2", |b| {
        b.iter(|| {
            let results: Vec<_> = transducer
                .query(black_box("quick brown"), black_box(2))
                .take(10)
                .collect();
            black_box(results);
        });
    });

    group.finish();
}

/// Benchmark: Code search use case
fn bench_code_search(c: &mut Criterion) {
    let code_samples = vec![
        r#"fn calculate_total(items: &[Item]) -> f64 { items.iter().map(|i| i.price).sum() }"#,
        r#"async fn fetch_data(url: &str) -> Result<String, Error> { reqwest::get(url).await?.text().await }"#,
        r#"impl Display for MyType { fn fmt(&self, f: &mut Formatter) -> fmt::Result { write!(f, "{}", self.value) } }"#,
        r#"#[derive(Debug, Clone, Serialize)] struct User { id: u64, name: String, email: String }"#,
        r#"match result { Ok(value) => println!("Success: {}", value), Err(e) => eprintln!("Error: {}", e) }"#,
    ];

    let dict: SuffixAutomaton<()> = SuffixAutomaton::from_texts(code_samples);
    let transducer = Transducer::new(dict, Algorithm::Standard);
    let mut group = c.benchmark_group("suffix_automaton_code_search");

    let searches = [
        ("calculate", 0),  // Exact match
        ("calculte", 1),   // 1 typo
        ("async fn", 0),   // Multi-word exact
        ("async func", 2), // Multi-word with typos
        ("derive", 0),     // Attribute search
    ];

    for (query, distance) in searches.iter() {
        let label = format!("{}_{}", query.replace(' ', "_"), distance);
        group.bench_function(&label, |b| {
            b.iter(|| {
                let results: Vec<_> = transducer
                    .query(black_box(query), black_box(*distance))
                    .collect();
                black_box(results);
            });
        });
    }

    group.finish();
}

/// Benchmark: Dynamic updates
fn bench_dynamic_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("suffix_automaton_dynamic");

    group.bench_function("insert_single", |b| {
        let dict: SuffixAutomaton<()> = SuffixAutomaton::new();
        b.iter(|| {
            dict.insert(black_box("The quick brown fox jumps over the lazy dog"));
        });
    });

    group.bench_function("insert_remove_cycle", |b| {
        let dict: SuffixAutomaton<()> = SuffixAutomaton::new();
        let text = "The quick brown fox jumps over the lazy dog";
        b.iter(|| {
            dict.insert(black_box(text));
            dict.remove(black_box(text));
        });
    });

    group.bench_function("compaction", |b| {
        let dict: SuffixAutomaton<()> = SuffixAutomaton::new();
        // Build up some data
        for i in 0..100 {
            dict.insert(&format!("text number {}", i));
        }
        // Remove half
        for i in 0..50 {
            dict.remove(&format!("text number {}", i));
        }
        b.iter(|| {
            dict.compact();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_construction_varying_corpus_size,
    bench_incremental_insertion,
    bench_query_varying_corpus_size,
    bench_query_varying_distance,
    bench_query_varying_query_length,
    bench_ordered_query,
    bench_code_search,
    bench_dynamic_updates,
);
criterion_main!(benches);
