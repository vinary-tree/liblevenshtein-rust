//! Benchmarks for LLev parsing and phonetic rule compilation.
//!
//! This benchmark suite measures:
//! - LLev file parsing time
//! - RuleSetChar construction time
//! - NFA compilation time
//! - End-to-end cold-start latency
//!
//! Run with:
//! ```bash
//! taskset -c 0 cargo bench --bench phonetic_compilation_benchmarks --features phonetic-rules
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

#[cfg(feature = "phonetic-rules")]
mod benchmarks {
    use super::*;
    use liblevenshtein::phonetic::llev::lexer::{Lexer, Token};
    use liblevenshtein::phonetic::llev::{load_file, parse_str};
    use liblevenshtein::phonetic::RuleSetChar;

    /// Rule files to benchmark
    const RULE_FILES: &[(&str, &str)] = &[
        ("zompist", "data/rules/english/zompist.llev"),
        ("homophones", "data/rules/english/homophones.llev"),
        ("text_speak", "data/rules/english/text_speak.llev"),
    ];

    /// Benchmark LLev file parsing
    pub fn bench_llev_parsing(c: &mut Criterion) {
        let mut group = c.benchmark_group("llev_parsing");
        group.sample_size(100);

        for (name, path) in RULE_FILES {
            // Read file content for throughput calculation
            let content = std::fs::read_to_string(path).expect("Failed to read rule file");
            let content_len = content.len() as u64;

            group.throughput(Throughput::Bytes(content_len));

            group.bench_with_input(BenchmarkId::new("load_file", name), path, |b, path| {
                b.iter(|| {
                    let file = load_file(black_box(*path)).expect("Parse failed");
                    black_box(file)
                });
            });
        }

        group.finish();
    }

    /// Benchmark RuleSetChar construction from parsed LLev file
    pub fn bench_ruleset_construction(c: &mut Criterion) {
        let mut group = c.benchmark_group("ruleset_construction");
        group.sample_size(100);

        for (name, path) in RULE_FILES {
            let llev_file = load_file(path).expect("Failed to parse");
            let rule_count = llev_file.rules.len();

            group.throughput(Throughput::Elements(rule_count as u64));

            group.bench_with_input(
                BenchmarkId::new("from_llev", name),
                &llev_file,
                |b, file| {
                    b.iter(|| {
                        let ruleset =
                            RuleSetChar::from_llev(black_box(file)).expect("Conversion failed");
                        black_box(ruleset)
                    });
                },
            );
        }

        group.finish();
    }

    /// Benchmark end-to-end cold start (parse + compile)
    pub fn bench_cold_start(c: &mut Criterion) {
        let mut group = c.benchmark_group("cold_start");
        group.sample_size(50); // Fewer samples for slower benchmark

        for (name, path) in RULE_FILES {
            group.bench_with_input(BenchmarkId::new("end_to_end", name), path, |b, path| {
                b.iter(|| {
                    let file = load_file(black_box(*path)).expect("Parse failed");
                    let ruleset = RuleSetChar::from_llev(&file).expect("Conversion failed");
                    black_box(ruleset)
                });
            });
        }

        group.finish();
    }

    /// Benchmark many small parses (simulating high-frequency pattern compilation)
    pub fn bench_small_parses(c: &mut Criterion) {
        let mut group = c.benchmark_group("small_parses");
        group.sample_size(100);

        // Generate simple rule strings
        let rules: Vec<String> = (0..100)
            .map(|i| format!("rule{} -> replacement{};", i, i))
            .collect();

        let total_bytes: u64 = rules.iter().map(|r| r.len() as u64).sum();
        group.throughput(Throughput::Bytes(total_bytes));

        group.bench_function("parse_100_rules", |b| {
            b.iter(|| {
                for rule in &rules {
                    let _ = black_box(parse_str(black_box(rule)));
                }
            });
        });

        group.finish();
    }

    /// Benchmark lexer token generation (isolate tokenization overhead)
    pub fn bench_lexer_throughput(c: &mut Criterion) {
        let mut group = c.benchmark_group("lexer_throughput");
        group.sample_size(100);

        for (name, path) in RULE_FILES {
            let content = std::fs::read_to_string(path).expect("Failed to read");
            let content_len = content.len() as u64;

            group.throughput(Throughput::Bytes(content_len));

            group.bench_with_input(
                BenchmarkId::new("tokenize", name),
                &content,
                |b, content| {
                    b.iter(|| {
                        let mut lexer = Lexer::new(black_box(content));
                        let mut count = 0usize;
                        loop {
                            match lexer.next_token() {
                                Ok(Token::Eof) => break,
                                Ok(_) => count += 1,
                                Err(_) => break,
                            }
                        }
                        black_box(count)
                    });
                },
            );
        }

        group.finish();
    }

    criterion_group!(
        benches,
        bench_llev_parsing,
        bench_ruleset_construction,
        bench_cold_start,
        bench_small_parses,
        bench_lexer_throughput,
    );
}

#[cfg(feature = "phonetic-rules")]
criterion_main!(benchmarks::benches);

#[cfg(not(feature = "phonetic-rules"))]
fn main() {
    eprintln!("This benchmark requires the 'phonetic-rules' feature.");
    eprintln!(
        "Run with: cargo bench --bench phonetic_compilation_benchmarks --features phonetic-rules"
    );
}
