//! Benchmarks comparing sequential vs parallel phonetic grep.
//!
//! This benchmark suite measures the performance of:
//! 1. Sequential vs parallel intra-document scanning (`scan` vs `scan_parallel`)
//! 2. Sequential vs parallel inter-document scanning (`scan_documents_parallel`)
//!
//! Run with: cargo bench --bench phonetic_grep_parallel_benchmarks --features parallel-grep

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liblevenshtein::phonetic::grep_online::PhoneticGrepOnline;
use liblevenshtein::phonetic::types::{ContextChar, PhoneChar, RewriteRuleChar};

/// Helper to create a simple rule for testing.
fn make_rule(pattern: &str, replacement: &str, context: ContextChar) -> RewriteRuleChar {
    fn char_to_phone(c: char) -> PhoneChar {
        let lower = c.to_ascii_lowercase();
        if "aeiou".contains(lower) {
            PhoneChar::Vowel(c)
        } else {
            PhoneChar::Consonant(c)
        }
    }

    RewriteRuleChar {
        rule_id: 0,
        rule_name: format!("{} -> {}", pattern, replacement),
        pattern: pattern.chars().map(char_to_phone).collect(),
        replacement: replacement.chars().map(char_to_phone).collect(),
        context,
        weight: 1.0,
    }
}

/// Create standard phonetic rules for benchmarking.
fn standard_rules() -> Vec<RewriteRuleChar> {
    vec![
        make_rule("ph", "f", ContextChar::Anywhere),
        make_rule("oo", "u", ContextChar::Anywhere),
        make_rule("ee", "i", ContextChar::Anywhere),
        make_rule("ck", "k", ContextChar::Anywhere),
        make_rule("gh", "f", ContextChar::Final),
    ]
}

/// Generate a document of approximately the given size with realistic text.
fn generate_document(target_size: usize) -> String {
    let words = [
        "phone", "fone", "food", "good", "book", "look", "check", "quick", "tough", "enough",
        "the", "and", "to", "a", "in", "that", "is", "was", "for", "on", "are", "with", "they",
        "be", "at", "one", "have", "this", "from", "or", "had", "by", "word", "but", "not", "what",
        "all", "were", "we", "when", "your", "can", "said",
    ];

    let mut doc = String::with_capacity(target_size);
    let mut word_idx = 0;

    while doc.len() < target_size {
        if !doc.is_empty() {
            doc.push(' ');
        }
        doc.push_str(words[word_idx % words.len()]);
        word_idx += 1;
    }

    doc.truncate(target_size);
    doc
}

/// Benchmark intra-document parallelism: scan() vs scan_parallel()
fn bench_intra_document_parallelism(c: &mut Criterion) {
    let mut group = c.benchmark_group("intra_document_parallelism");

    let rules = standard_rules();
    let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);

    // Test different document sizes
    let sizes = [
        ("1KB", 1024),
        ("10KB", 10 * 1024),
        ("100KB", 100 * 1024),
        ("1MB", 1024 * 1024),
    ];

    for (name, size) in sizes {
        let doc = generate_document(size);

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("sequential", name), &doc, |b, doc| {
            b.iter(|| black_box(grep.scan(doc)))
        });

        #[cfg(feature = "parallel-grep")]
        group.bench_with_input(BenchmarkId::new("parallel", name), &doc, |b, doc| {
            b.iter(|| black_box(grep.scan_parallel(doc)))
        });
    }

    group.finish();
}

/// Benchmark inter-document parallelism: sequential loop vs scan_documents_parallel()
#[cfg(feature = "parallel-grep")]
fn bench_inter_document_parallelism(c: &mut Criterion) {
    let mut group = c.benchmark_group("inter_document_parallelism");

    let rules = standard_rules();
    let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);

    // Test with different numbers of documents
    let doc_counts = [10, 50, 100, 500];
    let doc_size = 1024; // 1KB per document

    for count in doc_counts {
        let documents: Vec<(usize, String)> = (0..count)
            .map(|i| (i, generate_document(doc_size)))
            .collect();

        let total_bytes = count * doc_size;
        group.throughput(Throughput::Bytes(total_bytes as u64));

        // Sequential: loop over documents
        group.bench_with_input(
            BenchmarkId::new("sequential_loop", count),
            &documents,
            |b, docs| {
                b.iter(|| {
                    let results: Vec<_> = docs
                        .iter()
                        .map(|(id, text)| (*id, grep.scan(text)))
                        .collect();
                    black_box(results)
                })
            },
        );

        // Parallel: scan_documents_parallel
        let doc_refs: Vec<(usize, &str)> = documents
            .iter()
            .map(|(id, text)| (*id, text.as_str()))
            .collect();

        group.bench_with_input(BenchmarkId::new("parallel", count), &doc_refs, |b, docs| {
            b.iter(|| black_box(grep.scan_documents_parallel(docs.clone())))
        });
    }

    group.finish();
}

/// Benchmark nested parallelism: inter + intra document parallelism
#[cfg(feature = "parallel-grep")]
fn bench_nested_parallelism(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_parallelism");

    let rules = standard_rules();
    let grep = PhoneticGrepOnline::with_rules("phone", rules, 1);

    // Large documents where nested parallelism might help
    let doc_counts = [4, 8, 16];
    let doc_size = 100 * 1024; // 100KB per document

    for count in doc_counts {
        let documents: Vec<(usize, String)> = (0..count)
            .map(|i| (i, generate_document(doc_size)))
            .collect();

        let total_bytes = count * doc_size;
        group.throughput(Throughput::Bytes(total_bytes as u64));

        let doc_refs: Vec<(usize, &str)> = documents
            .iter()
            .map(|(id, text)| (*id, text.as_str()))
            .collect();

        // Inter-document parallel only
        group.bench_with_input(
            BenchmarkId::new("inter_only", count),
            &doc_refs,
            |b, docs| b.iter(|| black_box(grep.scan_documents_parallel(docs.clone()))),
        );

        // Nested: inter + intra document parallelism
        group.bench_with_input(BenchmarkId::new("nested", count), &doc_refs, |b, docs| {
            b.iter(|| black_box(grep.scan_documents_parallel_nested(docs.clone())))
        });
    }

    group.finish();
}

/// Benchmark filtering vs scanning all documents
#[cfg(feature = "parallel-grep")]
fn bench_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_filtering");

    let rules = standard_rules();
    let grep = PhoneticGrepOnline::with_rules("phone", rules, 0);

    // Create documents with varying match rates
    let doc_count = 100;
    let doc_size = 1024;

    // 10% of documents have "phone" (will match), rest have "hello"
    let documents: Vec<(usize, String)> = (0..doc_count)
        .map(|i| {
            let content = if i % 10 == 0 {
                "phone".to_string()
            } else {
                generate_document(doc_size)
            };
            (i, content)
        })
        .collect();

    let doc_refs: Vec<(usize, &str)> = documents
        .iter()
        .map(|(id, text)| (*id, text.as_str()))
        .collect();

    // Scan all documents
    group.bench_with_input(
        BenchmarkId::new("scan_all", doc_count),
        &doc_refs,
        |b, docs| b.iter(|| black_box(grep.scan_documents_parallel(docs.clone()))),
    );

    // Filter documents (only return those with matches)
    group.bench_with_input(
        BenchmarkId::new("filter_only", doc_count),
        &doc_refs,
        |b, docs| b.iter(|| black_box(grep.filter_documents_parallel(docs.clone()))),
    );

    // Count matches (lightweight)
    group.bench_with_input(
        BenchmarkId::new("count_only", doc_count),
        &doc_refs,
        |b, docs| b.iter(|| black_box(grep.count_documents_parallel(docs.clone()))),
    );

    group.finish();
}

#[cfg(feature = "parallel-grep")]
criterion_group!(
    benches,
    bench_intra_document_parallelism,
    bench_inter_document_parallelism,
    bench_nested_parallelism,
    bench_filtering,
);

#[cfg(not(feature = "parallel-grep"))]
criterion_group!(benches, bench_intra_document_parallelism,);

criterion_main!(benches);
