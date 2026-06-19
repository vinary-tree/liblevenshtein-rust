use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::prelude::{Algorithm, Transducer};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[cfg(feature = "phonetic-rules")]
use liblevenshtein::dictionary::phonetic_normalized::PhoneticNormalizedDictionary;
#[cfg(all(feature = "phonetic-rules", feature = "embedded-rules"))]
use liblevenshtein::phonetic::language::rules_for_language;
#[cfg(feature = "phonetic-rules")]
use liblevenshtein::phonetic::llev::{load_file_with_includes, parse_str, RuleSetChar};
#[cfg(feature = "phonetic-rules")]
use liblevenshtein::phonetic::nfa::{compile as compile_nfa, ProductAutomatonChar};
#[cfg(feature = "phonetic-rules")]
use liblevenshtein::phonetic::regex::parse as parse_regex;
#[cfg(feature = "phonetic-rules")]
use liblevenshtein::phonetic::types::{PhoneChar, RewriteRuleChar};

struct CountingAllocator;

static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone)]
struct Options {
    samples: usize,
    warmups: usize,
    workload: Workload,
    birkbeck_dir: Option<PathBuf>,
    mitton_corpus_paths: Vec<PathBuf>,
    text_corpus_paths: Vec<PathBuf>,
    openslr_lexicon_paths: Vec<PathBuf>,
    cmudict_path: Option<PathBuf>,
    corpus_limit: usize,
    max_distance: usize,
    recall_k: usize,
    phonetic_dialect: String,
    phonetic_rules_file: Option<PathBuf>,
    phonetic_rule_extensions: Vec<PathBuf>,
    phonetic_rule_extension_order: RuleExtensionOrder,
    phonetic_target_files: Vec<PathBuf>,
    diagnostic_limit: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Workload {
    All,
    LevUnordered,
    LevOrdered,
    PhoneticNormalized,
    PhoneticRegexProduct,
    PhoneticRegexProductScan,
    BirkbeckFawthrop,
    MittonSpelling,
    TextCorpusLev,
    OpenSlrLexicon,
    CmudictPhonetic,
    CmudictPhoneticDiagnostic,
    PhoneticTargetedRules,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RuleExtensionOrder {
    BeforePrimary,
    AfterPrimary,
}

impl RuleExtensionOrder {
    fn parse(value: &str) -> Self {
        match value {
            "before" | "before-primary" | "prepend" => Self::BeforePrimary,
            "after" | "after-primary" | "append" => Self::AfterPrimary,
            other => panic!(
                "unknown --phonetic-rules-extension-order {other:?}; expected before or after"
            ),
        }
    }

    #[cfg(feature = "phonetic-rules")]
    fn as_str(self) -> &'static str {
        match self {
            Self::BeforePrimary => "before",
            Self::AfterPrimary => "after",
        }
    }
}

struct MeasureOutcome {
    result_count: usize,
    expected_count: usize,
    matched_count: usize,
    recall_at_k: f64,
    reciprocal_rank: f64,
    case_index: usize,
}

impl MeasureOutcome {
    fn synthetic(result_count: usize) -> Self {
        Self {
            result_count,
            expected_count: 0,
            matched_count: 0,
            recall_at_k: 0.0,
            reciprocal_rank: 0.0,
            case_index: 0,
        }
    }
}

impl Options {
    fn parse() -> Self {
        let mut opts = Self {
            samples: 36,
            warmups: 3,
            workload: Workload::All,
            birkbeck_dir: None,
            mitton_corpus_paths: Vec::new(),
            text_corpus_paths: Vec::new(),
            openslr_lexicon_paths: Vec::new(),
            cmudict_path: None,
            corpus_limit: 512,
            max_distance: 2,
            recall_k: 5,
            phonetic_dialect: "zompist-default".to_string(),
            phonetic_rules_file: None,
            phonetic_rule_extensions: Vec::new(),
            phonetic_rule_extension_order: RuleExtensionOrder::AfterPrimary,
            phonetic_target_files: Vec::new(),
            diagnostic_limit: 30,
        };

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--samples" => {
                    let value = args
                        .next()
                        .expect("--samples requires a positive integer value");
                    opts.samples = value.parse().expect("--samples must be a positive integer");
                }
                "--warmups" => {
                    let value = args
                        .next()
                        .expect("--warmups requires a positive integer value");
                    opts.warmups = value.parse().expect("--warmups must be a positive integer");
                }
                "--workload" => {
                    let value = args.next().expect("--workload requires a value");
                    opts.workload = match value.as_str() {
                        "all" => Workload::All,
                        "lev-unordered" => Workload::LevUnordered,
                        "lev-ordered" => Workload::LevOrdered,
                        "phonetic-normalized" => Workload::PhoneticNormalized,
                        "phonetic-regex-product" => Workload::PhoneticRegexProduct,
                        "phonetic-regex-product-scan" => Workload::PhoneticRegexProductScan,
                        "birkbeck-fawthrop" => Workload::BirkbeckFawthrop,
                        "mitton-spelling" => Workload::MittonSpelling,
                        "text-corpus-lev" | "pizza-chili-text" => Workload::TextCorpusLev,
                        "openslr-lexicon" | "wfst-lexicon" => Workload::OpenSlrLexicon,
                        "cmudict-phonetic" => Workload::CmudictPhonetic,
                        "cmudict-phonetic-diagnostic" => Workload::CmudictPhoneticDiagnostic,
                        "phonetic-targeted-rules" => Workload::PhoneticTargetedRules,
                        other => panic!(
                            "unknown workload {other:?}; expected all, lev-unordered, lev-ordered, phonetic-normalized, phonetic-regex-product, phonetic-regex-product-scan, birkbeck-fawthrop, mitton-spelling, text-corpus-lev, openslr-lexicon, cmudict-phonetic, cmudict-phonetic-diagnostic, or phonetic-targeted-rules"
                        ),
                    };
                }
                "--birkbeck-dir" => {
                    let value = args.next().expect("--birkbeck-dir requires a path");
                    opts.birkbeck_dir = Some(PathBuf::from(value));
                }
                "--mitton-corpus" => {
                    let value = args.next().expect("--mitton-corpus requires a path");
                    opts.mitton_corpus_paths.push(PathBuf::from(value));
                }
                "--text-corpus" => {
                    let value = args.next().expect("--text-corpus requires a path");
                    opts.text_corpus_paths.push(PathBuf::from(value));
                }
                "--openslr-lexicon" | "--wfst-lexicon" => {
                    let value = args.next().expect("--openslr-lexicon requires a path");
                    opts.openslr_lexicon_paths.push(PathBuf::from(value));
                }
                "--cmudict" => {
                    let value = args.next().expect("--cmudict requires a path");
                    opts.cmudict_path = Some(PathBuf::from(value));
                }
                "--corpus-limit" => {
                    let value = args
                        .next()
                        .expect("--corpus-limit requires a positive integer value");
                    opts.corpus_limit = value
                        .parse()
                        .expect("--corpus-limit must be a positive integer");
                }
                "--max-distance" => {
                    let value = args
                        .next()
                        .expect("--max-distance requires a non-negative integer value");
                    opts.max_distance = value
                        .parse()
                        .expect("--max-distance must be a non-negative integer");
                }
                "--recall-k" => {
                    let value = args
                        .next()
                        .expect("--recall-k requires a positive integer value");
                    opts.recall_k = value
                        .parse()
                        .expect("--recall-k must be a positive integer");
                    assert!(opts.recall_k > 0, "--recall-k must be positive");
                }
                "--phonetic-dialect" => {
                    opts.phonetic_dialect =
                        args.next().expect("--phonetic-dialect requires a value");
                }
                "--phonetic-rules-file" => {
                    let value = args.next().expect("--phonetic-rules-file requires a path");
                    opts.phonetic_rules_file = Some(PathBuf::from(value));
                }
                "--phonetic-rules-extension" => {
                    let value = args
                        .next()
                        .expect("--phonetic-rules-extension requires a path");
                    opts.phonetic_rule_extensions.push(PathBuf::from(value));
                }
                "--phonetic-rules-extension-order" => {
                    let value = args
                        .next()
                        .expect("--phonetic-rules-extension-order requires before or after");
                    opts.phonetic_rule_extension_order = RuleExtensionOrder::parse(&value);
                }
                "--phonetic-target-file" => {
                    let value = args.next().expect("--phonetic-target-file requires a path");
                    opts.phonetic_target_files.push(PathBuf::from(value));
                }
                "--diagnostic-limit" => {
                    let value = args
                        .next()
                        .expect("--diagnostic-limit requires a positive integer value");
                    opts.diagnostic_limit = value
                        .parse()
                        .expect("--diagnostic-limit must be a positive integer");
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --release --example scientific_eval -- [--samples N] [--warmups N] [--workload all|lev-unordered|lev-ordered|phonetic-normalized|phonetic-regex-product|phonetic-regex-product-scan|birkbeck-fawthrop|mitton-spelling|text-corpus-lev|openslr-lexicon|cmudict-phonetic|cmudict-phonetic-diagnostic|phonetic-targeted-rules] [--birkbeck-dir DIR] [--mitton-corpus PATH ...] [--text-corpus PATH ...] [--openslr-lexicon PATH ...] [--cmudict PATH] [--corpus-limit N] [--max-distance N] [--recall-k N] [--phonetic-dialect zompist-default|en-us|en-gb|...] [--phonetic-rules-file PATH] [--phonetic-rules-extension PATH ...] [--phonetic-rules-extension-order before|after] [--phonetic-target-file PATH ...] [--diagnostic-limit N]"
                    );
                    std::process::exit(0);
                }
                other => panic!("unknown argument {other:?}"),
            }
        }

        opts
    }
}

fn reset_counters() {
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    DEALLOCATED_BYTES.store(0, Ordering::Relaxed);
    ALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);
}

fn counters() -> (usize, usize, usize) {
    let allocated = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let deallocated = DEALLOCATED_BYTES.load(Ordering::Relaxed);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    (
        allocated,
        allocations,
        allocated.saturating_sub(deallocated),
    )
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}

#[cfg(feature = "phonetic-rules")]
fn json_string_array(values: &[String]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(value));
        out.push('"');
    }
    out.push(']');
    out
}

#[cfg(feature = "phonetic-rules")]
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0; b.len() + 1];

    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b.len()]
}

fn create_dictionary(size: usize) -> DoubleArrayTrie {
    let words: Vec<String> = (0..size)
        .map(|i| match i % 5 {
            0 => format!("test{i}"),
            1 => format!("best{i}"),
            2 => format!("rest{i}"),
            3 => format!("word{i}"),
            _ => format!("term{i}"),
        })
        .collect();
    DoubleArrayTrie::from_terms(words)
}

fn measure<F>(workload: &str, samples: usize, warmups: usize, f: F)
where
    F: FnMut(usize) -> MeasureOutcome,
{
    measure_with_phonetic_dialect(workload, samples, warmups, "none", f);
}

fn measure_with_phonetic_dialect<F>(
    workload: &str,
    samples: usize,
    warmups: usize,
    phonetic_dialect: &str,
    mut f: F,
) where
    F: FnMut(usize) -> MeasureOutcome,
{
    let phonetic_dialect = json_escape(phonetic_dialect);
    for sample in 0..samples + warmups {
        let warmup = sample < warmups;
        reset_counters();

        let started = Instant::now();
        let outcome = black_box(f(sample));
        let elapsed = started.elapsed();

        let (allocated_bytes, allocation_count, live_bytes) = counters();
        println!(
            "{{\"workload\":\"{workload}\",\"phonetic_dialect\":\"{phonetic_dialect}\",\"sample\":{sample},\"warmup\":{warmup},\"elapsed_us\":{},\"allocated_bytes\":{allocated_bytes},\"allocation_count\":{allocation_count},\"live_bytes\":{live_bytes},\"result_count\":{},\"expected_count\":{},\"matched_count\":{},\"recall_at_k\":{},\"reciprocal_rank\":{},\"case_index\":{}}}",
            elapsed.as_nanos() as f64 / 1000.0,
            outcome.result_count,
            outcome.expected_count,
            outcome.matched_count,
            outcome.recall_at_k,
            outcome.reciprocal_rank,
            outcome.case_index
        );
    }
}

fn run_lev_unordered(samples: usize, warmups: usize) {
    let dict = create_dictionary(1_000);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    measure("lev_unordered_1k_d2", samples, warmups, |_| {
        MeasureOutcome::synthetic(
            transducer
                .query(black_box("test500"), black_box(2))
                .collect::<Vec<_>>()
                .len(),
        )
    });
}

fn run_lev_ordered(samples: usize, warmups: usize) {
    let dict = create_dictionary(1_000);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    measure("lev_ordered_1k_d2", samples, warmups, |_| {
        MeasureOutcome::synthetic(
            transducer
                .query_ordered(black_box("test500"), black_box(2))
                .collect::<Vec<_>>()
                .len(),
        )
    });
}

#[derive(Clone)]
struct SpellingCase {
    correct: String,
    misspelling: String,
}

#[cfg(feature = "phonetic-rules")]
#[derive(Clone)]
struct HomophoneCase {
    query: String,
    expected: Vec<String>,
}

#[cfg(feature = "phonetic-rules")]
#[derive(Clone)]
struct TargetedRuleCase {
    query: String,
    expected: String,
}

fn normalize_ascii_word(token: &str) -> Option<String> {
    let trimmed = token.trim_matches(|c: char| !c.is_ascii_alphabetic());
    if trimmed.len() < 2 || !trimmed.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    Some(trimmed.to_ascii_lowercase())
}

fn read_birkbeck_pairs(path: &Path, out: &mut Vec<SpellingCase>) {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read Birkbeck file {}: {err}", path.display()));

    for line in text.lines() {
        let mut columns = line.split_whitespace();
        let Some(correct) = columns.next().and_then(normalize_ascii_word) else {
            continue;
        };
        let Some(misspelling) = columns.next().and_then(normalize_ascii_word) else {
            continue;
        };
        if correct != misspelling {
            out.push(SpellingCase {
                correct,
                misspelling,
            });
        }
    }
}

fn load_birkbeck_fawthrop(dir: &Path, limit: usize) -> Vec<SpellingCase> {
    let mut cases = Vec::new();
    read_birkbeck_pairs(&dir.join("FAWTHROP1DAT.643"), &mut cases);
    read_birkbeck_pairs(&dir.join("FAWTHROP2DAT.643"), &mut cases);
    cases.sort_by(|a, b| {
        a.misspelling
            .cmp(&b.misspelling)
            .then_with(|| a.correct.cmp(&b.correct))
    });
    cases.dedup_by(|a, b| a.correct == b.correct && a.misspelling == b.misspelling);
    cases.truncate(limit);
    assert!(
        !cases.is_empty(),
        "Birkbeck Fawthrop loader found no spelling pairs in {}",
        dir.display()
    );
    cases
}

fn run_birkbeck_fawthrop(
    dir: &Path,
    samples: usize,
    warmups: usize,
    limit: usize,
    max_distance: usize,
    recall_k: usize,
) {
    let cases = load_birkbeck_fawthrop(dir, limit);
    let mut terms: Vec<String> = cases.iter().map(|case| case.correct.clone()).collect();
    terms.sort();
    terms.dedup();

    let dict = DoubleArrayTrie::from_terms(terms);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    measure("birkbeck_fawthrop_ordered", samples, warmups, |sample| {
        let case_index = sample % cases.len();
        let case = &cases[case_index];
        let candidates: Vec<_> = transducer
            .query_ordered(black_box(&case.misspelling), black_box(max_distance))
            .collect();
        let rank = candidates
            .iter()
            .position(|candidate| candidate.term == case.correct)
            .map(|idx| idx + 1);
        let matched_count = usize::from(rank.is_some_and(|idx| idx <= recall_k));

        MeasureOutcome {
            result_count: candidates.len(),
            expected_count: 1,
            matched_count,
            recall_at_k: matched_count as f64,
            reciprocal_rank: rank.map_or(0.0, |idx| 1.0 / idx as f64),
            case_index,
        }
    });
}

fn read_mitton_pairs(path: &Path, out: &mut Vec<SpellingCase>) {
    let file = File::open(path)
        .unwrap_or_else(|err| panic!("failed to open Mitton corpus {}: {err}", path.display()));
    let reader = BufReader::new(file);
    let mut current_correct = None;

    for line in reader.lines() {
        let line = line
            .unwrap_or_else(|err| panic!("failed to read Mitton corpus {}: {err}", path.display()));
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(correct) = trimmed.strip_prefix('$') {
            current_correct = normalize_ascii_word(correct);
            continue;
        }

        let Some(correct) = current_correct.as_ref() else {
            continue;
        };
        let Some(misspelling) = trimmed
            .split_whitespace()
            .next()
            .and_then(normalize_ascii_word)
        else {
            continue;
        };
        if correct != &misspelling {
            out.push(SpellingCase {
                correct: correct.clone(),
                misspelling,
            });
        }
    }
}

fn load_mitton_spelling(paths: &[PathBuf], limit: usize) -> Vec<SpellingCase> {
    assert!(
        !paths.is_empty(),
        "mitton-spelling requires at least one --mitton-corpus PATH"
    );

    let mut cases = Vec::new();
    for path in paths {
        read_mitton_pairs(path, &mut cases);
    }
    cases.sort_by(|a, b| {
        a.misspelling
            .cmp(&b.misspelling)
            .then_with(|| a.correct.cmp(&b.correct))
    });
    cases.dedup_by(|a, b| a.correct == b.correct && a.misspelling == b.misspelling);
    cases.truncate(limit);
    assert!(
        !cases.is_empty(),
        "Mitton loader found no spelling pairs in {}",
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    cases
}

fn run_mitton_spelling(
    paths: &[PathBuf],
    samples: usize,
    warmups: usize,
    limit: usize,
    max_distance: usize,
    recall_k: usize,
) {
    let cases = load_mitton_spelling(paths, limit);
    let mut terms: Vec<String> = cases.iter().map(|case| case.correct.clone()).collect();
    terms.sort();
    terms.dedup();

    let dict = DoubleArrayTrie::from_terms(terms);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    measure("mitton_spelling_ordered", samples, warmups, |sample| {
        let case_index = sample % cases.len();
        let case = &cases[case_index];
        let candidates: Vec<_> = transducer
            .query_ordered(black_box(&case.misspelling), black_box(max_distance))
            .collect();
        let rank = candidates
            .iter()
            .position(|candidate| candidate.term == case.correct)
            .map(|idx| idx + 1);
        let matched_count = usize::from(rank.is_some_and(|idx| idx <= recall_k));

        MeasureOutcome {
            result_count: candidates.len(),
            expected_count: 1,
            matched_count,
            recall_at_k: matched_count as f64,
            reciprocal_rank: rank.map_or(0.0, |idx| 1.0 / idx as f64),
            case_index,
        }
    });
}

fn record_ascii_tokens(line: &str, frequencies: &mut HashMap<String, usize>) {
    let mut token = String::new();

    for ch in line.chars() {
        if ch.is_ascii_alphabetic() {
            token.push(ch.to_ascii_lowercase());
        } else if token.len() >= 2 {
            *frequencies.entry(std::mem::take(&mut token)).or_insert(0) += 1;
        } else {
            token.clear();
        }
    }

    if token.len() >= 2 {
        *frequencies.entry(token).or_insert(0) += 1;
    }
}

fn load_text_corpus_terms(paths: &[PathBuf], limit: usize) -> Vec<String> {
    assert!(
        !paths.is_empty(),
        "text-corpus-lev requires at least one --text-corpus PATH"
    );

    let mut frequencies = HashMap::new();
    for path in paths {
        let file = File::open(path)
            .unwrap_or_else(|err| panic!("failed to open text corpus {}: {err}", path.display()));
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.unwrap_or_else(|err| {
                panic!("failed to read text corpus {}: {err}", path.display())
            });
            record_ascii_tokens(&line, &mut frequencies);
        }
    }

    let mut terms: Vec<(String, usize)> = frequencies.into_iter().collect();
    terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    terms.truncate(limit);

    let mut terms: Vec<String> = terms.into_iter().map(|(term, _)| term).collect();
    terms.sort();
    assert!(
        !terms.is_empty(),
        "text corpus loader found no ASCII word terms in {}",
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    terms
}

fn load_openslr_lexicon_terms(paths: &[PathBuf], limit: usize) -> Vec<String> {
    assert!(
        !paths.is_empty(),
        "openslr-lexicon requires at least one --openslr-lexicon PATH"
    );

    let mut terms = HashSet::new();
    for path in paths {
        let file = File::open(path)
            .unwrap_or_else(|err| panic!("failed to open lexicon {}: {err}", path.display()));
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line
                .unwrap_or_else(|err| panic!("failed to read lexicon {}: {err}", path.display()));
            let trimmed = line.split('#').next().unwrap_or("").trim();
            if trimmed.is_empty() || trimmed.starts_with(';') || trimmed.starts_with('\\') {
                continue;
            }

            let Some(raw_word) = trimmed.split_whitespace().next() else {
                continue;
            };
            if raw_word.starts_with('<') || raw_word.starts_with('[') {
                continue;
            }
            if let Some(word) = normalize_ascii_word(raw_word) {
                terms.insert(word);
            }
        }
    }

    let mut terms: Vec<String> = terms.into_iter().collect();
    terms.sort();
    terms.truncate(limit);
    assert!(
        !terms.is_empty(),
        "OpenSLR lexicon loader found no ASCII lexicon terms in {}",
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    terms
}

fn deterministic_query(term: &str, sample: usize, max_distance: usize) -> String {
    if max_distance == 0 {
        return term.to_string();
    }

    let mut chars: Vec<char> = term.chars().collect();
    if chars.len() < 2 {
        return term.to_string();
    }

    match sample % 4 {
        0 => {
            chars.remove(sample % chars.len());
            chars.into_iter().collect()
        }
        1 => {
            let idx = sample % chars.len();
            chars[idx] = if chars[idx] == 'a' { 'e' } else { 'a' };
            chars.into_iter().collect()
        }
        2 => {
            let idx = sample % (chars.len() - 1);
            chars.swap(idx, idx + 1);
            chars.into_iter().collect()
        }
        _ => {
            let idx = sample % (chars.len() + 1);
            chars.insert(idx, 'x');
            chars.into_iter().collect()
        }
    }
}

fn run_dictionary_terms_ordered(
    workload: &str,
    terms: Vec<String>,
    samples: usize,
    warmups: usize,
    max_distance: usize,
    recall_k: usize,
) {
    let cases = terms.clone();
    let dict = DoubleArrayTrie::from_terms(terms);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    measure(workload, samples, warmups, |sample| {
        let case_index = sample % cases.len();
        let expected = &cases[case_index];
        let query = deterministic_query(expected, sample, max_distance);
        let candidates: Vec<_> = transducer
            .query_ordered(black_box(&query), black_box(max_distance))
            .collect();
        let rank = candidates
            .iter()
            .position(|candidate| candidate.term == *expected)
            .map(|idx| idx + 1);
        let matched_count = usize::from(rank.is_some_and(|idx| idx <= recall_k));

        MeasureOutcome {
            result_count: candidates.len(),
            expected_count: 1,
            matched_count,
            recall_at_k: matched_count as f64,
            reciprocal_rank: rank.map_or(0.0, |idx| 1.0 / idx as f64),
            case_index,
        }
    });
}

fn run_text_corpus_lev(
    paths: &[PathBuf],
    samples: usize,
    warmups: usize,
    limit: usize,
    max_distance: usize,
    recall_k: usize,
) {
    let terms = load_text_corpus_terms(paths, limit);
    run_dictionary_terms_ordered(
        "text_corpus_lev_ordered",
        terms,
        samples,
        warmups,
        max_distance,
        recall_k,
    );
}

fn run_openslr_lexicon(
    paths: &[PathBuf],
    samples: usize,
    warmups: usize,
    limit: usize,
    max_distance: usize,
    recall_k: usize,
) {
    let terms = load_openslr_lexicon_terms(paths, limit);
    run_dictionary_terms_ordered(
        "openslr_lexicon_lev_ordered",
        terms,
        samples,
        warmups,
        max_distance,
        recall_k,
    );
}

#[cfg(feature = "phonetic-rules")]
fn extended_words(count: usize) -> Vec<String> {
    const BASE: &[&str] = &[
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for", "not", "on",
        "with", "he", "as", "you", "do", "at", "this", "phone", "fone", "elephant", "elefant",
        "knight", "night", "through", "thru", "colour", "color",
    ];

    (0..count)
        .map(|i| {
            let base = BASE[i % BASE.len()];
            if i < BASE.len() {
                base.to_string()
            } else {
                format!("{}{}", base, i / BASE.len())
            }
        })
        .collect()
}

#[cfg(feature = "phonetic-rules")]
fn phonetic_dictionary_from_dialect(
    terms: &[String],
    dialect: &str,
) -> PhoneticNormalizedDictionary<()> {
    match dialect {
        "default" | "zompist" | "zompist-default" => {
            PhoneticNormalizedDictionary::<()>::from_terms(terms)
        }
        other => phonetic_dictionary_from_embedded_dialect(terms, other),
    }
}

#[cfg(feature = "phonetic-rules")]
fn phonetic_dictionary_from_config(
    terms: &[String],
    dialect: &str,
    rules_file: Option<&Path>,
    rule_extensions: &[PathBuf],
    rule_extension_order: RuleExtensionOrder,
) -> PhoneticNormalizedDictionary<()> {
    if let Some(rules_file) = rules_file {
        let rules = rules_from_llev_files(rules_file, rule_extensions, rule_extension_order);
        PhoneticNormalizedDictionary::<()>::from_terms_with_rules(terms, rules)
    } else {
        assert!(
            rule_extensions.is_empty(),
            "--phonetic-rules-extension requires --phonetic-rules-file"
        );
        phonetic_dictionary_from_dialect(terms, dialect)
    }
}

#[cfg(feature = "phonetic-rules")]
fn rules_from_llev_files(
    primary: &Path,
    extensions: &[PathBuf],
    extension_order: RuleExtensionOrder,
) -> Vec<liblevenshtein::phonetic::types::RewriteRuleChar> {
    let mut include_paths = Vec::new();
    if let Some(parent) = primary.parent() {
        include_paths.push(parent.to_path_buf());
        if let Some(grandparent) = parent.parent() {
            include_paths.push(grandparent.to_path_buf());
        }
    }

    let mut file = load_file_with_includes(primary, &include_paths).unwrap_or_else(|err| {
        panic!(
            "failed to load LLev rules file {}: {err}",
            primary.display()
        )
    });

    let mut extension_files = Vec::new();
    for extension in extensions {
        let text = fs::read_to_string(extension).unwrap_or_else(|err| {
            panic!(
                "failed to read LLev extension file {}: {err}",
                extension.display()
            )
        });
        let extension_file = parse_str(&text).unwrap_or_else(|err| {
            panic!(
                "failed to parse LLev extension file {}: {err}",
                extension.display()
            )
        });
        extension_files.push(extension_file);
    }

    match extension_order {
        RuleExtensionOrder::AfterPrimary => {
            for extension_file in extension_files {
                file.merge(extension_file);
            }
        }
        RuleExtensionOrder::BeforePrimary => {
            for mut extension_file in extension_files.into_iter().rev() {
                extension_file.merge(file);
                file = extension_file;
            }
        }
    }

    let ruleset = RuleSetChar::from_llev(&file)
        .unwrap_or_else(|err| panic!("failed to convert LLev rule set: {err}"));
    ruleset.rules
}

#[cfg(feature = "phonetic-rules")]
fn phonetic_label(
    dialect: &str,
    rules_file: Option<&Path>,
    rule_extensions: &[PathBuf],
    rule_extension_order: RuleExtensionOrder,
) -> String {
    if let Some(rules_file) = rules_file {
        let mut label = format!("llev:{}", rules_file.display());
        for extension in rule_extensions {
            label.push_str("+ext:");
            label.push_str(&extension.display().to_string());
        }
        if !rule_extensions.is_empty() {
            label.push_str("+ext_order:");
            label.push_str(rule_extension_order.as_str());
        }
        label
    } else {
        dialect.to_string()
    }
}

#[cfg(all(feature = "phonetic-rules", feature = "embedded-rules"))]
fn phonetic_dictionary_from_embedded_dialect(
    terms: &[String],
    dialect: &str,
) -> PhoneticNormalizedDictionary<()> {
    let rules = rules_for_language(dialect)
        .unwrap_or_else(|| panic!("unsupported liblevenshtein phonetic dialect {dialect:?}"));
    PhoneticNormalizedDictionary::<()>::from_terms_with_rules(terms, rules)
}

#[cfg(all(feature = "phonetic-rules", not(feature = "embedded-rules")))]
fn phonetic_dictionary_from_embedded_dialect(
    _terms: &[String],
    dialect: &str,
) -> PhoneticNormalizedDictionary<()> {
    panic!(
        "liblevenshtein phonetic dialect {dialect:?} requires building with --features embedded-rules"
    );
}

#[cfg(feature = "phonetic-rules")]
fn run_phonetic_normalized(
    samples: usize,
    warmups: usize,
    phonetic_dialect: &str,
    phonetic_rules_file: Option<&Path>,
    phonetic_rule_extensions: &[PathBuf],
    phonetic_rule_extension_order: RuleExtensionOrder,
) {
    let words = extended_words(10_000);
    let dict = phonetic_dictionary_from_config(
        &words,
        phonetic_dialect,
        phonetic_rules_file,
        phonetic_rule_extensions,
        phonetic_rule_extension_order,
    );
    let label = phonetic_label(
        phonetic_dialect,
        phonetic_rules_file,
        phonetic_rule_extensions,
        phonetic_rule_extension_order,
    );

    measure_with_phonetic_dialect(
        "phonetic_normalized_10k_phone_d2",
        samples,
        warmups,
        &label,
        |_| MeasureOutcome::synthetic(dict.query(black_box("phone"), black_box(2)).len()),
    );
}

#[cfg(feature = "phonetic-rules")]
fn run_phonetic_regex_product(
    samples: usize,
    warmups: usize,
    limit: usize,
    max_distance: usize,
    phonetic_dialect: &str,
    phonetic_rules_file: Option<&Path>,
    phonetic_rule_extensions: &[PathBuf],
    phonetic_rule_extension_order: RuleExtensionOrder,
    use_scan_control: bool,
) {
    let words = extended_words(limit.max(30));
    let dict = phonetic_dictionary_from_config(
        &words,
        phonetic_dialect,
        phonetic_rules_file,
        phonetic_rule_extensions,
        phonetic_rule_extension_order,
    );
    let label = phonetic_label(
        phonetic_dialect,
        phonetic_rules_file,
        phonetic_rule_extensions,
        phonetic_rule_extension_order,
    );

    let pattern_terms = ["phone", "colour", "knight", "through", "elephant"];
    let products: Vec<ProductAutomatonChar> = pattern_terms
        .iter()
        .map(|term| {
            let pattern = dict.normalize(term);
            let ast = parse_regex(&pattern).unwrap_or_else(|err| {
                panic!("failed to parse normalized pattern {pattern}: {err}")
            });
            let nfa = compile_nfa(&ast).unwrap_or_else(|err| {
                panic!("failed to compile normalized pattern {pattern}: {err}")
            });
            ProductAutomatonChar::with_algorithm(nfa, max_distance as u8, dict.algorithm())
        })
        .collect();

    measure_with_phonetic_dialect(
        if use_scan_control {
            "phonetic_regex_product_scan_precompiled"
        } else {
            "phonetic_regex_product_precompiled"
        },
        samples,
        warmups,
        &label,
        |sample| {
            let product = &products[sample % products.len()];
            if use_scan_control {
                MeasureOutcome::synthetic(phonetic_regex_product_scan_count(
                    &dict,
                    black_box(product),
                ))
            } else {
                MeasureOutcome::synthetic(dict.query_with_product(black_box(product)).len())
            }
        },
    );
}

#[cfg(feature = "phonetic-rules")]
fn phonetic_regex_product_scan_count(
    dict: &PhoneticNormalizedDictionary<()>,
    product: &ProductAutomatonChar,
) -> usize {
    dict.iter_normalized()
        .filter_map(|(normalized, originals)| {
            product
                .min_distance(&normalized)
                .map(|_| originals.iter().filter(|term| !term.is_empty()).count())
        })
        .sum()
}

#[cfg(feature = "phonetic-rules")]
fn cmudict_base_word(raw: &str) -> Option<String> {
    let base = raw.split_once('(').map_or(raw, |(base, _)| base);
    normalize_ascii_word(base)
}

#[cfg(feature = "phonetic-rules")]
fn load_cmudict_homophones(path: &Path, limit: usize) -> Vec<HomophoneCase> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read CMUdict file {}: {err}", path.display()));
    let mut by_pronunciation: HashMap<String, HashSet<String>> = HashMap::new();

    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with(";;;") {
            continue;
        }

        let mut parts = line.split_whitespace();
        let Some(word) = parts.next().and_then(cmudict_base_word) else {
            continue;
        };
        let pronunciation = parts.collect::<Vec<_>>().join(" ");
        if pronunciation.is_empty() {
            continue;
        }
        by_pronunciation
            .entry(pronunciation)
            .or_default()
            .insert(word);
    }

    let mut groups: Vec<Vec<String>> = by_pronunciation
        .into_values()
        .filter_map(|set| {
            if set.len() < 2 {
                return None;
            }
            let mut group: Vec<String> = set.into_iter().collect();
            group.sort();
            Some(group)
        })
        .collect();
    groups.sort_by(|a, b| a[0].cmp(&b[0]).then_with(|| a.len().cmp(&b.len())));

    let mut cases = Vec::new();
    for group in groups {
        for query in &group {
            let expected: Vec<String> = group
                .iter()
                .filter(|term| *term != query)
                .cloned()
                .collect();
            if !expected.is_empty() {
                cases.push(HomophoneCase {
                    query: query.clone(),
                    expected,
                });
            }
            if cases.len() >= limit {
                return cases;
            }
        }
    }

    assert!(
        !cases.is_empty(),
        "CMUdict loader found no homophone groups in {}",
        path.display()
    );
    cases
}

#[cfg(feature = "phonetic-rules")]
fn phones_to_ascii_word(phones: &[PhoneChar]) -> Option<String> {
    let mut word = String::new();
    for phone in phones {
        for ch in phone.chars() {
            if !ch.is_ascii_alphabetic() {
                return None;
            }
            word.push(ch.to_ascii_lowercase());
        }
    }

    if word.len() < 2 {
        None
    } else {
        Some(word)
    }
}

#[cfg(feature = "phonetic-rules")]
fn targeted_case_from_rule(rule: &RewriteRuleChar) -> Option<TargetedRuleCase> {
    let query = phones_to_ascii_word(&rule.pattern)?;
    let expected = phones_to_ascii_word(&rule.replacement)?;
    if query == expected {
        return None;
    }
    Some(TargetedRuleCase { query, expected })
}

#[cfg(feature = "phonetic-rules")]
fn load_targeted_rule_cases(paths: &[PathBuf], limit: usize) -> Vec<TargetedRuleCase> {
    assert!(
        !paths.is_empty(),
        "phonetic-targeted-rules requires --phonetic-target-file or --phonetic-rules-extension"
    );

    let mut cases = Vec::new();
    let mut seen = HashSet::new();

    for path in paths {
        let text = fs::read_to_string(path).unwrap_or_else(|err| {
            panic!("failed to read LLev target file {}: {err}", path.display())
        });
        let file = parse_str(&text).unwrap_or_else(|err| {
            panic!("failed to parse LLev target file {}: {err}", path.display())
        });
        let ruleset = RuleSetChar::from_llev(&file).unwrap_or_else(|err| {
            panic!(
                "failed to convert LLev target file {}: {err}",
                path.display()
            )
        });

        for rule in &ruleset.rules {
            let Some(case) = targeted_case_from_rule(rule) else {
                continue;
            };
            if seen.insert((case.query.clone(), case.expected.clone())) {
                cases.push(case);
                if cases.len() >= limit {
                    return cases;
                }
            }
        }
    }

    assert!(
        !cases.is_empty(),
        "no usable ASCII rewrite target cases found in {}",
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    cases
}

#[cfg(feature = "phonetic-rules")]
fn run_cmudict_phonetic(
    path: &Path,
    samples: usize,
    warmups: usize,
    limit: usize,
    max_distance: usize,
    recall_k: usize,
    phonetic_dialect: &str,
    phonetic_rules_file: Option<&Path>,
    phonetic_rule_extensions: &[PathBuf],
    phonetic_rule_extension_order: RuleExtensionOrder,
) {
    let cases = load_cmudict_homophones(path, limit);
    let mut terms = HashSet::new();
    for case in &cases {
        terms.insert(case.query.clone());
        terms.extend(case.expected.iter().cloned());
    }
    let mut terms: Vec<String> = terms.into_iter().collect();
    terms.sort();

    let dict = phonetic_dictionary_from_config(
        &terms,
        phonetic_dialect,
        phonetic_rules_file,
        phonetic_rule_extensions,
        phonetic_rule_extension_order,
    );
    let label = phonetic_label(
        phonetic_dialect,
        phonetic_rules_file,
        phonetic_rule_extensions,
        phonetic_rule_extension_order,
    );

    measure_with_phonetic_dialect(
        "cmudict_phonetic_homophones",
        samples,
        warmups,
        &label,
        |sample| {
            let case_index = sample % cases.len();
            let case = &cases[case_index];
            let candidates = dict.query(black_box(&case.query), black_box(max_distance));
            let expected: HashSet<&str> = case.expected.iter().map(String::as_str).collect();
            let mut matched = HashSet::new();
            let mut first_rank = None;

            for (idx, candidate) in candidates.iter().take(recall_k).enumerate() {
                if expected.contains(candidate.term.as_str()) {
                    matched.insert(candidate.term.as_str());
                    first_rank.get_or_insert(idx + 1);
                }
            }

            MeasureOutcome {
                result_count: candidates.len(),
                expected_count: expected.len(),
                matched_count: matched.len(),
                recall_at_k: matched.len() as f64 / expected.len() as f64,
                reciprocal_rank: first_rank.map_or(0.0, |idx| 1.0 / idx as f64),
                case_index,
            }
        },
    );
}

#[cfg(feature = "phonetic-rules")]
fn run_phonetic_targeted_rules(
    samples: usize,
    warmups: usize,
    limit: usize,
    max_distance: usize,
    recall_k: usize,
    phonetic_dialect: &str,
    phonetic_rules_file: Option<&Path>,
    phonetic_rule_extensions: &[PathBuf],
    phonetic_rule_extension_order: RuleExtensionOrder,
    phonetic_target_files: &[PathBuf],
) {
    let target_files = if phonetic_target_files.is_empty() {
        phonetic_rule_extensions
    } else {
        phonetic_target_files
    };
    let cases = load_targeted_rule_cases(target_files, limit);

    let mut terms = HashSet::new();
    for case in &cases {
        terms.insert(case.query.clone());
        terms.insert(case.expected.clone());
    }
    let mut terms: Vec<String> = terms.into_iter().collect();
    terms.sort();

    let dict = phonetic_dictionary_from_config(
        &terms,
        phonetic_dialect,
        phonetic_rules_file,
        phonetic_rule_extensions,
        phonetic_rule_extension_order,
    );
    let mut label = phonetic_label(
        phonetic_dialect,
        phonetic_rules_file,
        phonetic_rule_extensions,
        phonetic_rule_extension_order,
    );
    label.push_str("+targets:");
    label.push_str(
        &target_files
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("+"),
    );

    measure_with_phonetic_dialect(
        "phonetic_targeted_rule_pairs",
        samples,
        warmups,
        &label,
        |sample| {
            let case_index = sample % cases.len();
            let case = &cases[case_index];
            let candidates = dict.query(black_box(&case.query), black_box(max_distance));
            let rank = candidates
                .iter()
                .position(|candidate| candidate.term == case.expected)
                .map(|idx| idx + 1);
            let matched_count = usize::from(rank.is_some_and(|idx| idx <= recall_k));

            MeasureOutcome {
                result_count: candidates.len(),
                expected_count: 1,
                matched_count,
                recall_at_k: matched_count as f64,
                reciprocal_rank: rank.map_or(0.0, |idx| 1.0 / idx as f64),
                case_index,
            }
        },
    );
}

#[cfg(feature = "phonetic-rules")]
fn run_cmudict_phonetic_diagnostic(
    path: &Path,
    limit: usize,
    diagnostic_limit: usize,
    max_distance: usize,
    recall_k: usize,
    phonetic_dialect: &str,
    phonetic_rules_file: Option<&Path>,
    phonetic_rule_extensions: &[PathBuf],
    phonetic_rule_extension_order: RuleExtensionOrder,
) {
    let cases = load_cmudict_homophones(path, limit.max(diagnostic_limit));
    let mut terms = HashSet::new();
    for case in &cases {
        terms.insert(case.query.clone());
        terms.extend(case.expected.iter().cloned());
    }
    let mut terms: Vec<String> = terms.into_iter().collect();
    terms.sort();

    let dict = phonetic_dictionary_from_config(
        &terms,
        phonetic_dialect,
        phonetic_rules_file,
        phonetic_rule_extensions,
        phonetic_rule_extension_order,
    );
    let label = phonetic_label(
        phonetic_dialect,
        phonetic_rules_file,
        phonetic_rule_extensions,
        phonetic_rule_extension_order,
    );
    let escaped_label = json_escape(&label);
    let rule_count = dict.rules().len();
    let normalized_count = dict.normalized_count();

    for (case_index, case) in cases.iter().take(diagnostic_limit).enumerate() {
        reset_counters();
        let started = Instant::now();
        let candidates = dict.query(black_box(&case.query), black_box(max_distance));
        let elapsed = started.elapsed();
        let (allocated_bytes, allocation_count, live_bytes) = counters();

        let query_normalized = dict.normalize(&case.query);
        let top_terms: Vec<String> = candidates
            .iter()
            .take(recall_k)
            .map(|candidate| candidate.term.clone())
            .collect();
        let full_result_terms: HashSet<&str> = candidates
            .iter()
            .map(|candidate| candidate.term.as_str())
            .collect();

        for expected in &case.expected {
            let expected_normalized = dict.normalize(expected);
            let normalized_distance = levenshtein_distance(&query_normalized, &expected_normalized);
            let matched_rank = candidates
                .iter()
                .position(|candidate| candidate.term == *expected)
                .map(|idx| idx + 1);
            let matched_top_k = matched_rank.is_some_and(|rank| rank <= recall_k);
            let in_full_results = full_result_terms.contains(expected.as_str());
            let root_cause = if matched_top_k {
                "retrieved"
            } else if in_full_results {
                "ranking_or_limit"
            } else if normalized_distance <= max_distance {
                "normalized_index_or_query_bug"
            } else {
                "coverage_or_oracle_gap"
            };

            println!(
                "{{\"workload\":\"cmudict_phonetic_diagnostic\",\"phonetic_dialect\":\"{escaped_label}\",\"case_index\":{case_index},\"query\":\"{}\",\"expected\":\"{}\",\"query_normalized\":\"{}\",\"expected_normalized\":\"{}\",\"normalized_distance\":{normalized_distance},\"max_distance\":{max_distance},\"recall_k\":{recall_k},\"matched_top_k\":{matched_top_k},\"matched_rank\":{},\"in_full_results\":{in_full_results},\"root_cause\":\"{root_cause}\",\"top_terms\":{},\"candidate_count\":{},\"rule_count\":{rule_count},\"term_count\":{},\"normalized_count\":{normalized_count},\"elapsed_us\":{},\"allocated_bytes\":{allocated_bytes},\"allocation_count\":{allocation_count},\"live_bytes\":{live_bytes}}}",
                json_escape(&case.query),
                json_escape(expected),
                json_escape(&query_normalized),
                json_escape(&expected_normalized),
                matched_rank.map_or_else(|| "null".to_string(), |rank| rank.to_string()),
                json_string_array(&top_terms),
                candidates.len(),
                terms.len(),
                elapsed.as_nanos() as f64 / 1000.0
            );
        }
    }
}

#[cfg(not(feature = "phonetic-rules"))]
fn run_phonetic_normalized(
    _samples: usize,
    _warmups: usize,
    _phonetic_dialect: &str,
    _phonetic_rules_file: Option<&Path>,
    _phonetic_rule_extensions: &[PathBuf],
    _phonetic_rule_extension_order: RuleExtensionOrder,
) {
    panic!("phonetic-normalized workload requires --features phonetic-rules");
}

#[cfg(not(feature = "phonetic-rules"))]
fn run_phonetic_regex_product(
    _samples: usize,
    _warmups: usize,
    _limit: usize,
    _max_distance: usize,
    _phonetic_dialect: &str,
    _phonetic_rules_file: Option<&Path>,
    _phonetic_rule_extensions: &[PathBuf],
    _phonetic_rule_extension_order: RuleExtensionOrder,
    _use_scan_control: bool,
) {
    panic!("phonetic-regex-product workload requires --features phonetic-rules");
}

#[cfg(not(feature = "phonetic-rules"))]
fn run_cmudict_phonetic(
    _path: &Path,
    _samples: usize,
    _warmups: usize,
    _limit: usize,
    _max_distance: usize,
    _recall_k: usize,
    _phonetic_dialect: &str,
    _phonetic_rules_file: Option<&Path>,
    _phonetic_rule_extensions: &[PathBuf],
    _phonetic_rule_extension_order: RuleExtensionOrder,
) {
    panic!("cmudict-phonetic workload requires --features phonetic-rules");
}

#[cfg(not(feature = "phonetic-rules"))]
fn run_phonetic_targeted_rules(
    _samples: usize,
    _warmups: usize,
    _limit: usize,
    _max_distance: usize,
    _recall_k: usize,
    _phonetic_dialect: &str,
    _phonetic_rules_file: Option<&Path>,
    _phonetic_rule_extensions: &[PathBuf],
    _phonetic_rule_extension_order: RuleExtensionOrder,
    _phonetic_target_files: &[PathBuf],
) {
    panic!("phonetic-targeted-rules workload requires --features phonetic-rules");
}

#[cfg(not(feature = "phonetic-rules"))]
fn run_cmudict_phonetic_diagnostic(
    _path: &Path,
    _limit: usize,
    _diagnostic_limit: usize,
    _max_distance: usize,
    _recall_k: usize,
    _phonetic_dialect: &str,
    _phonetic_rules_file: Option<&Path>,
    _phonetic_rule_extensions: &[PathBuf],
    _phonetic_rule_extension_order: RuleExtensionOrder,
) {
    panic!("cmudict-phonetic-diagnostic workload requires --features phonetic-rules");
}

fn main() {
    let opts = Options::parse();

    if matches!(opts.workload, Workload::All | Workload::LevUnordered) {
        run_lev_unordered(opts.samples, opts.warmups);
    }
    if matches!(opts.workload, Workload::All | Workload::LevOrdered) {
        run_lev_ordered(opts.samples, opts.warmups);
    }
    if matches!(opts.workload, Workload::All | Workload::PhoneticNormalized) {
        run_phonetic_normalized(
            opts.samples,
            opts.warmups,
            &opts.phonetic_dialect,
            opts.phonetic_rules_file.as_deref(),
            &opts.phonetic_rule_extensions,
            opts.phonetic_rule_extension_order,
        );
    }
    if matches!(opts.workload, Workload::PhoneticRegexProductScan) {
        run_phonetic_regex_product(
            opts.samples,
            opts.warmups,
            opts.corpus_limit,
            opts.max_distance,
            &opts.phonetic_dialect,
            opts.phonetic_rules_file.as_deref(),
            &opts.phonetic_rule_extensions,
            opts.phonetic_rule_extension_order,
            true,
        );
    }
    if matches!(
        opts.workload,
        Workload::All | Workload::PhoneticRegexProduct
    ) {
        run_phonetic_regex_product(
            opts.samples,
            opts.warmups,
            opts.corpus_limit,
            opts.max_distance,
            &opts.phonetic_dialect,
            opts.phonetic_rules_file.as_deref(),
            &opts.phonetic_rule_extensions,
            opts.phonetic_rule_extension_order,
            false,
        );
    }
    if matches!(opts.workload, Workload::BirkbeckFawthrop)
        || (matches!(opts.workload, Workload::All) && opts.birkbeck_dir.is_some())
    {
        let dir = opts
            .birkbeck_dir
            .as_deref()
            .expect("birkbeck-fawthrop requires --birkbeck-dir DIR");
        run_birkbeck_fawthrop(
            dir,
            opts.samples,
            opts.warmups,
            opts.corpus_limit,
            opts.max_distance,
            opts.recall_k,
        );
    }
    if matches!(opts.workload, Workload::MittonSpelling)
        || (matches!(opts.workload, Workload::All) && !opts.mitton_corpus_paths.is_empty())
    {
        run_mitton_spelling(
            &opts.mitton_corpus_paths,
            opts.samples,
            opts.warmups,
            opts.corpus_limit,
            opts.max_distance,
            opts.recall_k,
        );
    }
    if matches!(opts.workload, Workload::TextCorpusLev)
        || (matches!(opts.workload, Workload::All) && !opts.text_corpus_paths.is_empty())
    {
        run_text_corpus_lev(
            &opts.text_corpus_paths,
            opts.samples,
            opts.warmups,
            opts.corpus_limit,
            opts.max_distance,
            opts.recall_k,
        );
    }
    if matches!(opts.workload, Workload::OpenSlrLexicon)
        || (matches!(opts.workload, Workload::All) && !opts.openslr_lexicon_paths.is_empty())
    {
        run_openslr_lexicon(
            &opts.openslr_lexicon_paths,
            opts.samples,
            opts.warmups,
            opts.corpus_limit,
            opts.max_distance,
            opts.recall_k,
        );
    }
    if matches!(opts.workload, Workload::CmudictPhonetic)
        || (matches!(opts.workload, Workload::All) && opts.cmudict_path.is_some())
    {
        let path = opts
            .cmudict_path
            .as_deref()
            .expect("cmudict-phonetic requires --cmudict PATH");
        run_cmudict_phonetic(
            path,
            opts.samples,
            opts.warmups,
            opts.corpus_limit,
            opts.max_distance,
            opts.recall_k,
            &opts.phonetic_dialect,
            opts.phonetic_rules_file.as_deref(),
            &opts.phonetic_rule_extensions,
            opts.phonetic_rule_extension_order,
        );
    }
    if matches!(opts.workload, Workload::PhoneticTargetedRules) {
        run_phonetic_targeted_rules(
            opts.samples,
            opts.warmups,
            opts.corpus_limit,
            opts.max_distance,
            opts.recall_k,
            &opts.phonetic_dialect,
            opts.phonetic_rules_file.as_deref(),
            &opts.phonetic_rule_extensions,
            opts.phonetic_rule_extension_order,
            &opts.phonetic_target_files,
        );
    }
    if matches!(opts.workload, Workload::CmudictPhoneticDiagnostic) {
        let path = opts
            .cmudict_path
            .as_deref()
            .expect("cmudict-phonetic-diagnostic requires --cmudict PATH");
        run_cmudict_phonetic_diagnostic(
            path,
            opts.corpus_limit,
            opts.diagnostic_limit,
            opts.max_distance,
            opts.recall_k,
            &opts.phonetic_dialect,
            opts.phonetic_rules_file.as_deref(),
            &opts.phonetic_rule_extensions,
            opts.phonetic_rule_extension_order,
        );
    }
}
