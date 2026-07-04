//! Corpus-based benchmarks for liblevenshtein-rust.
//!
//! These benchmarks measure construction time, query performance, and memory
//! usage using standard corpora (Norvig's big.txt and spelling error datasets).
//!
//! ## Running Benchmarks
//!
//! Download corpora first:
//!
//! ```bash
//! ./scripts/download_corpora.sh
//! ```
//!
//! Run benchmarks:
//!
//! ```bash
//! cargo bench --bench corpus_benchmarks
//! ```
//!
//! ## Benchmark Groups
//!
//! 1. **Construction**: Build dictionaries from big.txt subsets (1K-500K words)
//! 2. **Realistic Queries**: Query performance with frequency-stratified workload
//! 3. **Validation Queries**: Performance on real spelling errors (Holbrook/Aspell)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::corpus::{BigTxtCorpus, MittonCorpus, QueryWorkload};
use liblevenshtein::prelude::*;
use std::collections::BTreeSet;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;

const SEED: u64 = 42;

fn fallback_code_identifiers() -> Vec<String> {
    [
        "query_ordered",
        "query_with_distance",
        "standard_distance",
        "transposition_distance",
        "merge_and_split_distance",
        "DoubleArrayTrie",
        "DynamicDawg",
        "PathMapDictionary",
        "ValueFilteredQueryIterator",
        "CharacteristicVector",
        "OperationSet",
        "SubstitutionPolicy",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn collect_rust_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

fn extract_rust_identifiers(source: &str, identifiers: &mut BTreeSet<String>) {
    for token in source.split(|c: char| !(c == '_' || c.is_ascii_alphanumeric())) {
        let Some(first) = token.chars().next() else {
            continue;
        };

        if token.len() >= 3 && (first == '_' || first.is_ascii_alphabetic()) {
            identifiers.insert(token.to_string());
        }
    }
}

fn load_code_identifier_workload(root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    collect_rust_files(root, &mut files);

    let mut identifiers = BTreeSet::new();
    for file in files {
        if let Ok(source) = fs::read_to_string(file) {
            extract_rust_identifiers(&source, &mut identifiers);
        }
    }

    let identifiers: Vec<_> = identifiers.into_iter().take(10_000).collect();
    if identifiers.is_empty() {
        fallback_code_identifiers()
    } else {
        identifiers
    }
}

fn medical_term_workload() -> Vec<String> {
    [
        "acetaminophen",
        "amoxicillin",
        "bronchoscopy",
        "cardiomyopathy",
        "cholecystectomy",
        "diagnosis",
        "electrocardiogram",
        "hematocrit",
        "hypertension",
        "immunotherapy",
        "metformin",
        "nephrolithiasis",
        "ophthalmology",
        "pharmacy",
        "rhinorrhea",
        "sinusitis",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn dna_sequence_workload() -> Vec<String> {
    [
        "ACGTACGTACGT",
        "ACGTACGTTCGT",
        "GATTACAGATTA",
        "GATTACTGATTA",
        "TTGACAAGCTTA",
        "TTGACATGCTTA",
        "CGTATCGATCGA",
        "CGTATGGATCGA",
        "AACCGGTTAACC",
        "AACCGGCTAACC",
        "GGGAAATTTCCC",
        "GGGAAAATTCCC",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn workload_queries(name: &str) -> Vec<(&'static str, usize)> {
    match name {
        "code_identifiers" => vec![
            ("query_ordred", 1),
            ("standrd_distance", 1),
            ("ValueFilteredQueryItertor", 2),
            ("CharacteristicVctor", 2),
        ],
        "medical_terms" => vec![
            ("acetominophen", 2),
            ("amoxycillin", 1),
            ("cardiomyopthy", 1),
            ("opthamology", 2),
        ],
        "dna_sequences" => vec![
            ("ACGTACGTACGA", 1),
            ("GATTACTGATTA", 1),
            ("TTGACATGCTTA", 1),
            ("GGGAAAATTCCA", 1),
        ],
        _ => Vec::new(),
    }
}

fn load_big_txt() -> Option<BigTxtCorpus> {
    let path = "data/corpora/big.txt";
    if Path::new(path).exists() {
        BigTxtCorpus::load(path).ok()
    } else {
        eprintln!(
            "Warning: {} not found. Run: ./scripts/download_corpora.sh",
            path
        );
        None
    }
}

fn load_holbrook() -> Option<MittonCorpus> {
    let path = "data/corpora/holbrook.dat";
    if Path::new(path).exists() {
        MittonCorpus::load(path).ok()
    } else {
        eprintln!("Warning: {} not found", path);
        None
    }
}

fn construction_benchmarks(c: &mut Criterion) {
    let Some(corpus) = load_big_txt() else {
        eprintln!("Skipping construction benchmarks: big.txt not found");
        return;
    };

    let mut words = corpus.words_sorted();
    words.truncate(500_000); // Limit to 500K words

    let mut group = c.benchmark_group("corpus_construction");

    // Reduced sample size and increased measurement time to handle slow DoubleArrayTrie
    // construction at large scales (100K+ words can take 60+ seconds per iteration).
    // This prevents Criterion panics from insufficient samples with zero variance.
    group.sample_size(10); // Criterion minimum
    group.measurement_time(Duration::from_secs(120)); // Allow slow constructions to complete

    for size in [1_000, 10_000, 32_000, 100_000, 500_000] {
        if size > words.len() {
            continue;
        }

        let subset: Vec<_> = words[..size].to_vec();

        group.throughput(Throughput::Elements(size as u64));

        // DoubleArrayTrie construction has O(n²) or worse complexity and becomes
        // impractically slow at 100K+ words (60+ seconds per iteration). Skip these
        // sizes to prevent benchmark hangs while still measuring other backends.
        if size < 100_000 {
            group.bench_with_input(
                BenchmarkId::new("DoubleArrayTrie", size),
                &subset,
                |b, words| {
                    b.iter(|| {
                        let dict = DoubleArrayTrie::from_terms(black_box(words.iter().copied()));
                        black_box(dict);
                    });
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new("DynamicDawg", size),
            &subset,
            |b, words| {
                b.iter(|| {
                    let dict: DynamicDawg =
                        DynamicDawg::from_terms(black_box(words.iter().copied()));
                    black_box(dict);
                });
            },
        );
    }

    group.finish();
}

fn realistic_query_benchmarks(c: &mut Criterion) {
    let Some(corpus) = load_big_txt() else {
        eprintln!("Skipping query benchmarks: big.txt not found");
        return;
    };

    // Build dictionary from first 32K words (typical dictionary size)
    let mut words = corpus.words_sorted();
    words.truncate(32_000);

    let dict = DoubleArrayTrie::from_terms(words.iter().copied());
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Generate realistic query workload (Zipfian distribution)
    let workload = QueryWorkload::from_frequencies(&corpus.frequencies, corpus.total, 1000, SEED);

    let mut group = c.benchmark_group("corpus_realistic_queries");
    group.sample_size(100);

    for distance in [1, 2, 3] {
        group.bench_with_input(
            BenchmarkId::new("distance", distance),
            &distance,
            |b, &dist| {
                let queries = workload.query_strings();
                let mut idx = 0;

                b.iter(|| {
                    let query = queries[idx % queries.len()];
                    let results: Vec<_> = transducer.query(black_box(query), dist).collect();
                    idx = (idx + 1) % queries.len();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn validation_query_benchmarks(c: &mut Criterion) {
    let Some(holbrook) = load_holbrook() else {
        eprintln!("Skipping validation benchmarks: holbrook.dat not found");
        return;
    };

    // Build dictionary from Holbrook correct words
    let words = holbrook.correct_words_sorted();
    let dict = DoubleArrayTrie::from_terms(words.iter().copied());
    let transducer = Transducer::new(dict, Algorithm::Standard);

    // Collect all misspellings
    let misspellings: Vec<_> = holbrook
        .all_errors()
        .iter()
        .take(100) // Limit to 100 for benchmarking
        .map(|(misspelling, _, _)| misspelling.to_string())
        .collect();

    let mut group = c.benchmark_group("corpus_validation_queries");
    group.sample_size(50);

    for distance in [1, 2, 3] {
        group.bench_with_input(
            BenchmarkId::new("holbrook_distance", distance),
            &distance,
            |b, &dist| {
                let mut idx = 0;

                b.iter(|| {
                    let query = &misspellings[idx % misspellings.len()];
                    let results: Vec<_> = transducer.query(black_box(query), dist).collect();
                    idx = (idx + 1) % misspellings.len();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

fn backend_comparison_realistic_workload(c: &mut Criterion) {
    let Some(corpus) = load_big_txt() else {
        eprintln!("Skipping backend comparison: big.txt not found");
        return;
    };

    // Build dictionaries from 32K words
    let mut words = corpus.words_sorted();
    words.truncate(32_000);

    let dat = DoubleArrayTrie::from_terms(words.iter().copied());
    let dawg: DynamicDawg = DynamicDawg::from_terms(words.iter().copied());

    let transducer_dat = Transducer::new(dat, Algorithm::Standard);
    let transducer_dawg = Transducer::new(dawg, Algorithm::Standard);

    // Generate realistic queries
    let workload = QueryWorkload::from_frequencies(&corpus.frequencies, corpus.total, 100, SEED);
    let queries = workload.query_strings();

    let mut group = c.benchmark_group("corpus_backend_comparison");
    group.sample_size(100);

    let distance = 2; // Typical spelling correction distance

    group.bench_function("DoubleArrayTrie", |b| {
        let mut idx = 0;
        b.iter(|| {
            let query = queries[idx % queries.len()];
            let results: Vec<_> = transducer_dat.query(black_box(query), distance).collect();
            idx = (idx + 1) % queries.len();
            black_box(results);
        });
    });

    group.bench_function("DynamicDawg", |b| {
        let mut idx = 0;
        b.iter(|| {
            let query = queries[idx % queries.len()];
            let results: Vec<_> = transducer_dawg.query(black_box(query), distance).collect();
            idx = (idx + 1) % queries.len();
            black_box(results);
        });
    });

    group.finish();
}

fn domain_specific_workload_benchmarks(c: &mut Criterion) {
    let workloads = [
        (
            "code_identifiers",
            load_code_identifier_workload(Path::new("src")),
        ),
        ("medical_terms", medical_term_workload()),
        ("dna_sequences", dna_sequence_workload()),
    ];

    let mut group = c.benchmark_group("domain_specific_workloads");
    group.sample_size(100);

    for (name, terms) in workloads {
        let dict = DoubleArrayTrie::from_terms(terms.iter().map(String::as_str));
        let transducer = Transducer::new(dict, Algorithm::Standard);
        let queries = workload_queries(name);

        group.bench_with_input(
            BenchmarkId::new(name, terms.len()),
            &queries,
            |b, queries| {
                let mut idx = 0;

                b.iter(|| {
                    let (query, distance) = queries[idx % queries.len()];
                    let results: Vec<_> = transducer.query(black_box(query), distance).collect();
                    idx = (idx + 1) % queries.len();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    construction_benchmarks,
    realistic_query_benchmarks,
    validation_query_benchmarks,
    backend_comparison_realistic_workload,
    domain_specific_workload_benchmarks
);
criterion_main!(benches);
