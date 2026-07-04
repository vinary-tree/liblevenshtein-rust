//! Benchmarks comparing PhoneticNormalizedDictionary implementations:
//! - Old: BK-tree + HashMap for fuzzy queries
//! - New: FuzzyMultiMap (Levenshtein automaton + DAWG)
//!
//! Run with:
//!   cargo bench --bench bktree_vs_fuzzymap_benchmarks --features "phonetic-rules"
//!
//! To test with SIMD-accelerated BK-tree distance:
//!   cargo bench --bench bktree_vs_fuzzymap_benchmarks --features "phonetic-rules,simd"

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::{HashMap, HashSet};
use std::hint::black_box;

// ============================================================================
// OLD IMPLEMENTATION: BK-tree + HashMap (for comparison)
// ============================================================================

/// Levenshtein distance computation (used by BK-tree)
/// Uses SIMD-accelerated version on x86_64 targets.
#[cfg(target_arch = "x86_64")]
fn levenshtein_distance(a: &str, b: &str) -> usize {
    liblevenshtein::distance::standard_distance(a, b)
}

#[cfg(not(target_arch = "x86_64"))]
fn levenshtein_distance(a: &str, b: &str) -> usize {
    levenshtein_distance_scalar(a, b)
}

/// Scalar Levenshtein distance implementation (fallback)
#[cfg(not(target_arch = "x86_64"))]
fn levenshtein_distance_scalar(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for (j, item) in prev.iter_mut().enumerate().take(n + 1) {
        *item = j;
    }

    for i in 1..=m {
        curr[0] = i;

        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

struct BKNode {
    value: String,
    children: HashMap<usize, Box<BKNode>>,
}

struct BKTree {
    root: Option<Box<BKNode>>,
    size: usize,
}

impl BKTree {
    fn new() -> Self {
        Self {
            root: None,
            size: 0,
        }
    }

    fn insert(&mut self, value: String) {
        if self.root.is_none() {
            self.root = Some(Box::new(BKNode {
                value,
                children: HashMap::new(),
            }));
            self.size = 1;
            return;
        }

        let mut current = self.root.as_mut().expect("root exists");
        loop {
            let dist = levenshtein_distance(&value, &current.value);
            if dist == 0 {
                return;
            }

            if let std::collections::hash_map::Entry::Vacant(e) = current.children.entry(dist) {
                e.insert(Box::new(BKNode {
                    value,
                    children: HashMap::new(),
                }));
                self.size += 1;
                return;
            } else {
                current = current.children.get_mut(&dist).expect("key exists");
            }
        }
    }

    fn query(&self, query: &str, max_distance: usize) -> Vec<(String, usize)> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.query_recursive(root, query, max_distance, &mut results);
        }
        results
    }

    fn query_recursive(
        &self,
        node: &BKNode,
        query: &str,
        max_distance: usize,
        results: &mut Vec<(String, usize)>,
    ) {
        let dist = levenshtein_distance(query, &node.value);

        if dist <= max_distance {
            results.push((node.value.clone(), dist));
        }

        let min_child_dist = dist.saturating_sub(max_distance);
        let max_child_dist = dist + max_distance;

        for (&child_dist, child) in &node.children {
            if child_dist >= min_child_dist && child_dist <= max_child_dist {
                self.query_recursive(child, query, max_distance, results);
            }
        }
    }
}

struct OldPhoneticNormalizedDict {
    normalized_index: HashMap<String, HashSet<String>>,
    bk_tree: BKTree,
}

impl OldPhoneticNormalizedDict {
    fn from_normalized_pairs(pairs: impl Iterator<Item = (String, String)>) -> Self {
        let mut normalized_index = HashMap::<String, HashSet<String>>::new();
        let mut bk_tree = BKTree::new();

        for (original, normalized) in pairs {
            if !normalized_index.contains_key(&normalized) {
                bk_tree.insert(normalized.clone());
            }
            normalized_index
                .entry(normalized)
                .or_default()
                .insert(original);
        }

        Self {
            normalized_index,
            bk_tree,
        }
    }

    fn query(&self, normalized_query: &str, max_distance: usize) -> Vec<(String, usize, String)> {
        if max_distance == 0 {
            if let Some(originals) = self.normalized_index.get(normalized_query) {
                return originals
                    .iter()
                    .map(|term| (term.clone(), 0, normalized_query.to_string()))
                    .collect();
            }
            return Vec::new();
        }

        let bk_results = self.bk_tree.query(normalized_query, max_distance);
        let mut results = Vec::new();

        for (normalized_form, dist) in bk_results {
            if let Some(originals) = self.normalized_index.get(&normalized_form) {
                for term in originals {
                    results.push((term.clone(), dist, normalized_form.clone()));
                }
            }
        }

        results.sort_by_key(|(_, d, _)| *d);
        results
    }
}

// ============================================================================
// NEW IMPLEMENTATION: FuzzyMultiMap
// ============================================================================

use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use liblevenshtein::cache::multimap::FuzzyMultiMap;
use liblevenshtein::transducer::Algorithm;

struct NewPhoneticNormalizedDict {
    normalized_multimap: FuzzyMultiMap<HashSet<String>, DynamicDawgChar<HashSet<String>>>,
}

impl NewPhoneticNormalizedDict {
    fn from_normalized_pairs(pairs: impl Iterator<Item = (String, String)>) -> Self {
        // No bloom filter - it adds construction overhead and doesn't help positive queries
        let dict = DynamicDawgChar::<HashSet<String>>::new();

        for (original, normalized) in pairs {
            dict.update_or_insert(&normalized, HashSet::from([original.clone()]), |set| {
                set.insert(original.clone());
            });
        }

        let normalized_multimap = FuzzyMultiMap::new(dict, Algorithm::Standard);
        Self {
            normalized_multimap,
        }
    }

    fn query(&self, normalized_query: &str, max_distance: usize) -> Vec<(String, usize, String)> {
        let fuzzy_results = self
            .normalized_multimap
            .query_with_distance(normalized_query, max_distance);

        let mut results: Vec<(String, usize, String)> = fuzzy_results
            .into_iter()
            .flat_map(|(normalized_form, distance, originals)| {
                originals
                    .into_iter()
                    .filter(|t| !t.is_empty())
                    .map(move |term| (term, distance, normalized_form.clone()))
            })
            .collect();

        results.sort_by_key(|(_, d, _)| *d);
        results
    }
}

// ============================================================================
// TEST DATA GENERATION
// ============================================================================

use std::fs::File;
use std::io::{BufRead, BufReader};

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .replace("ph", "f")
        .replace("ough", "o")
        .replace("ight", "ite")
        .replace("tion", "shun")
        .replace("kn", "n")
        .replace("wr", "r")
        .replace("gh", "")
        .chars()
        .filter(|c| c.is_alphabetic())
        .collect()
}

/// Load dictionary words from system dictionary file.
/// Falls back to embedded words if system dictionary is unavailable.
fn load_dictionary_words() -> Vec<String> {
    let dict_paths = [
        "/usr/share/dict/words",
        "/usr/share/dict/american-english",
        "/usr/share/dict/british-english",
    ];

    for path in dict_paths {
        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            let words: Vec<String> = reader
                .lines()
                .map_while(Result::ok)
                .filter(|w| w.len() >= 3 && w.len() <= 15) // Filter reasonable word lengths
                .filter(|w| w.chars().all(|c| c.is_ascii_alphabetic())) // ASCII only
                .collect();

            if !words.is_empty() {
                return words;
            }
        }
    }

    // Fallback: embedded word list (subset of common English words)
    vec![
        "the",
        "and",
        "that",
        "have",
        "for",
        "not",
        "with",
        "you",
        "this",
        "but",
        "his",
        "from",
        "they",
        "say",
        "her",
        "she",
        "will",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "out",
        "about",
        "who",
        "get",
        "which",
        "make",
        "can",
        "like",
        "time",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "phone",
        "elephant",
        "knight",
        "psychology",
        "pneumonia",
        "through",
        "though",
        "thought",
        "enough",
        "cough",
        "rough",
        "tough",
        "bought",
        "brought",
        "caught",
        "daughter",
        "laughter",
        "slaughter",
        "nation",
        "station",
        "action",
        "fiction",
        "section",
        "mention",
        "attention",
        "write",
        "wrong",
        "wrist",
        "wrap",
        "wrestle",
        "wreck",
        "wrench",
        "knife",
        "know",
        "knee",
        "knock",
        "knit",
        "knot",
        "knowledge",
        "photograph",
        "telephone",
        "microphone",
        "saxophone",
        "symphony",
        "pharmacy",
        "phantom",
        "phase",
        "phenomenon",
        "philosophy",
        "physical",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Global dictionary cache to avoid repeated file I/O.
static DICTIONARY_WORDS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

fn generate_test_data(count: usize) -> Vec<(String, String)> {
    let words = DICTIONARY_WORDS.get_or_init(load_dictionary_words);

    // Use deterministic sampling for reproducible benchmarks
    let step = words.len().max(1) / count.max(1);
    let step = step.max(1);

    let mut pairs = Vec::with_capacity(count);

    for i in 0..count {
        let idx = (i * step) % words.len();
        let word = &words[idx];
        let normalized = normalize(word);
        pairs.push((word.clone(), normalized));
    }

    pairs
}

/// Generate realistic query variations (common misspellings/typos)
fn generate_queries(count: usize) -> Vec<String> {
    let words = DICTIONARY_WORDS.get_or_init(load_dictionary_words);

    // Sample words and apply phonetic normalization (simulating user queries)
    let step = words.len().max(1) / count.max(1);
    let step = step.max(1);

    (0..count)
        .map(|i| {
            let idx = (i * step + words.len() / 3) % words.len(); // Offset from test data
            normalize(&words[idx])
        })
        .collect()
}

// ============================================================================
// BENCHMARKS
// ============================================================================

fn benchmark_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");

    for size in [100, 1_000, 10_000, 50_000] {
        let data: Vec<_> = generate_test_data(size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("bk_tree", size), &data, |b, data| {
            b.iter(|| {
                let dict = OldPhoneticNormalizedDict::from_normalized_pairs(
                    data.iter().map(|(o, n)| (o.clone(), n.clone())),
                );
                black_box(dict)
            });
        });

        group.bench_with_input(
            BenchmarkId::new("fuzzy_multimap", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let dict = NewPhoneticNormalizedDict::from_normalized_pairs(
                        data.iter().map(|(o, n)| (o.clone(), n.clone())),
                    );
                    black_box(dict)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_query_distance_0(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_d0");

    for size in [100, 1_000, 10_000, 50_000] {
        let data = generate_test_data(size);
        let queries = generate_queries(100);

        let old_dict = OldPhoneticNormalizedDict::from_normalized_pairs(
            data.iter().map(|(o, n)| (o.clone(), n.clone())),
        );
        let new_dict = NewPhoneticNormalizedDict::from_normalized_pairs(
            data.iter().map(|(o, n)| (o.clone(), n.clone())),
        );

        group.throughput(Throughput::Elements(queries.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("bk_tree", size),
            &(&old_dict, &queries),
            |b, (dict, queries)| {
                b.iter(|| {
                    for q in queries.iter() {
                        black_box(dict.query(q, 0));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fuzzy_multimap", size),
            &(&new_dict, &queries),
            |b, (dict, queries)| {
                b.iter(|| {
                    for q in queries.iter() {
                        black_box(dict.query(q, 0));
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_query_distance_1(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_d1");

    for size in [100, 1_000, 10_000, 50_000] {
        let data = generate_test_data(size);
        let queries = generate_queries(100);

        let old_dict = OldPhoneticNormalizedDict::from_normalized_pairs(
            data.iter().map(|(o, n)| (o.clone(), n.clone())),
        );
        let new_dict = NewPhoneticNormalizedDict::from_normalized_pairs(
            data.iter().map(|(o, n)| (o.clone(), n.clone())),
        );

        group.throughput(Throughput::Elements(queries.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("bk_tree", size),
            &(&old_dict, &queries),
            |b, (dict, queries)| {
                b.iter(|| {
                    for q in queries.iter() {
                        black_box(dict.query(q, 1));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fuzzy_multimap", size),
            &(&new_dict, &queries),
            |b, (dict, queries)| {
                b.iter(|| {
                    for q in queries.iter() {
                        black_box(dict.query(q, 1));
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_query_distance_2(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_d2");

    for size in [100, 1_000, 10_000, 50_000] {
        let data = generate_test_data(size);
        let queries = generate_queries(100);

        let old_dict = OldPhoneticNormalizedDict::from_normalized_pairs(
            data.iter().map(|(o, n)| (o.clone(), n.clone())),
        );
        let new_dict = NewPhoneticNormalizedDict::from_normalized_pairs(
            data.iter().map(|(o, n)| (o.clone(), n.clone())),
        );

        group.throughput(Throughput::Elements(queries.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("bk_tree", size),
            &(&old_dict, &queries),
            |b, (dict, queries)| {
                b.iter(|| {
                    for q in queries.iter() {
                        black_box(dict.query(q, 2));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fuzzy_multimap", size),
            &(&new_dict, &queries),
            |b, (dict, queries)| {
                b.iter(|| {
                    for q in queries.iter() {
                        black_box(dict.query(q, 2));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_construction,
    benchmark_query_distance_0,
    benchmark_query_distance_1,
    benchmark_query_distance_2,
);
criterion_main!(benches);
