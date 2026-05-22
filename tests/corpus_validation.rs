//! Corpus validation tests for liblevenshtein-rust.
//!
//! These tests validate the spell correction algorithm against standard
//! corpora including Birkbeck, Holbrook, Aspell, and Wikipedia error datasets.
//!
//! ## Running Tests
//!
//! These tests require downloading corpora first:
//!
//! ```bash
//! ./scripts/download_corpora.sh
//! unzip data/corpora/birkbeck.zip -d data/corpora/birkbeck/
//! ```
//!
//! Run tests with:
//!
//! ```bash
//! cargo test --test corpus_validation -- --ignored --test-threads=1
//! ```
//!
//! ## Success Criteria
//!
//! | Corpus | Metric | Target | Distance | Achieved |
//! |--------|--------|--------|----------|----------|
//! | Holbrook | Recall | >85% | ≤2 | 86.6% ✓ |
//! | Holbrook | Recall | 100% | ≤3 | 100% ✓ |
//! | Aspell | Coverage | >85% | ≤2 | 100% ✓ |
//! | Wikipedia | Coverage | >90% | ≤2 | 100% ✓ |

use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::corpus::MittonCorpus;
use liblevenshtein::prelude::*;
use std::collections::HashSet;
use std::path::Path;

/// Statistics for corpus validation.
#[derive(Debug, Clone)]
struct ValidationStats {
    total_errors: usize,
    found_at_distance: [usize; 4], // Found at distance 0, 1, 2, 3
    not_found: usize,
}

impl ValidationStats {
    fn new() -> Self {
        Self {
            total_errors: 0,
            found_at_distance: [0; 4],
            not_found: 0,
        }
    }

    fn record(&mut self, found: bool, distance: usize) {
        self.total_errors += 1;
        if found {
            if distance <= 3 {
                self.found_at_distance[distance] += 1;
            }
        } else {
            self.not_found += 1;
        }
    }

    fn recall_at_distance(&self, max_distance: usize) -> f64 {
        let found: usize = self.found_at_distance[..=max_distance.min(3)].iter().sum();
        found as f64 / self.total_errors as f64
    }

    fn summary(&self) -> String {
        format!(
            "Total: {}, Found@0: {}, Found@1: {}, Found@2: {}, Found@3: {}, Not found: {} | \
             Recall@1: {:.1}%, Recall@2: {:.1}%, Recall@3: {:.1}%",
            self.total_errors,
            self.found_at_distance[0],
            self.found_at_distance[1],
            self.found_at_distance[2],
            self.found_at_distance[3],
            self.not_found,
            self.recall_at_distance(1) * 100.0,
            self.recall_at_distance(2) * 100.0,
            self.recall_at_distance(3) * 100.0,
        )
    }
}

/// Calculate naive Levenshtein distance for validation.
fn naive_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<_> = a.chars().collect();
    let b_chars: Vec<_> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            dp[i][j] = (dp[i - 1][j] + 1) // deletion
                .min(dp[i][j - 1] + 1) // insertion
                .min(dp[i - 1][j - 1] + cost); // substitution
        }
    }

    dp[m][n]
}

/// Build dictionary from corpus correct words.
fn build_dictionary_from_corpus(corpus: &MittonCorpus) -> DoubleArrayTrie {
    let words = corpus.correct_words_sorted();
    DoubleArrayTrie::from_terms(words)
}

#[test]
#[ignore]
fn test_holbrook_recall() {
    let corpus_path = "data/corpora/holbrook.dat";
    if !Path::new(corpus_path).exists() {
        eprintln!("Skipping test: {} not found", corpus_path);
        eprintln!("Run: ./scripts/download_corpora.sh");
        return;
    }

    let corpus = MittonCorpus::load(corpus_path).expect("Failed to load Holbrook corpus");
    let dict = build_dictionary_from_corpus(&corpus);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut stats = ValidationStats::new();

    for (correct, misspellings) in &corpus.errors {
        for (misspelling, frequency) in misspellings {
            let actual_distance = naive_levenshtein(misspelling, correct);

            // Skip errors beyond supported distance (algorithm supports up to 3)
            if actual_distance > 3 {
                continue;
            }

            // Query with actual distance
            let results: HashSet<_> = transducer.query(misspelling, actual_distance).collect();

            let found = results.contains(correct);
            stats.record(found, actual_distance);

            // Debug failures at distance 2 (high-frequency ones)
            if !found && actual_distance == 2 && *frequency > 1 {
                eprintln!(
                    "Failed@2: '{}' -> '{}' (freq={})",
                    misspelling, correct, frequency
                );
            }
        }
    }

    println!("\nHolbrook Validation Results:");
    println!("{}", stats.summary());

    // Target: >85% recall at distance ≤2 (baseline performance)
    // Note: 100% recall at distance ≤3
    let recall_at_2 = stats.recall_at_distance(2);
    assert!(
        recall_at_2 >= 0.85,
        "Holbrook recall at distance ≤2 is {:.2}% (target: ≥85%)",
        recall_at_2 * 100.0
    );
}

#[test]
#[ignore]
fn test_aspell_coverage() {
    let corpus_path = "data/corpora/aspell.dat";
    if !Path::new(corpus_path).exists() {
        eprintln!("Skipping test: {} not found", corpus_path);
        eprintln!("Run: ./scripts/download_corpora.sh");
        return;
    }

    let corpus = MittonCorpus::load(corpus_path).expect("Failed to load Aspell corpus");
    let dict = build_dictionary_from_corpus(&corpus);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut stats = ValidationStats::new();

    for (correct, misspellings) in &corpus.errors {
        for (misspelling, _frequency) in misspellings {
            let actual_distance = naive_levenshtein(misspelling, correct);

            // Skip errors beyond supported distance
            if actual_distance > 3 {
                continue;
            }

            let results: HashSet<_> = transducer.query(misspelling, actual_distance).collect();

            let found = results.contains(correct);
            stats.record(found, actual_distance);
        }
    }

    println!("\nAspell Validation Results:");
    println!("{}", stats.summary());

    // Target: >85% coverage at distance ≤2
    let coverage = stats.recall_at_distance(2);
    assert!(
        coverage >= 0.85,
        "Aspell coverage at distance ≤2 is {:.2}% (target: ≥85%)",
        coverage * 100.0
    );
}

#[test]
#[ignore]
fn test_wikipedia_coverage() {
    let corpus_path = "data/corpora/wikipedia.dat";
    if !Path::new(corpus_path).exists() {
        eprintln!("Skipping test: {} not found", corpus_path);
        eprintln!("Run: ./scripts/download_corpora.sh");
        return;
    }

    let corpus = MittonCorpus::load(corpus_path).expect("Failed to load Wikipedia corpus");
    let dict = build_dictionary_from_corpus(&corpus);
    let transducer = Transducer::new(dict, Algorithm::Standard);

    let mut stats = ValidationStats::new();

    for (correct, misspellings) in &corpus.errors {
        for (misspelling, _frequency) in misspellings {
            let actual_distance = naive_levenshtein(misspelling, correct);

            // Skip errors beyond supported distance
            if actual_distance > 3 {
                continue;
            }

            let results: HashSet<_> = transducer.query(misspelling, actual_distance).collect();

            let found = results.contains(correct);
            stats.record(found, actual_distance);
        }
    }

    println!("\nWikipedia Validation Results:");
    println!("{}", stats.summary());

    // Target: >90% coverage at distance ≤2
    let coverage = stats.recall_at_distance(2);
    assert!(
        coverage >= 0.90,
        "Wikipedia coverage at distance ≤2 is {:.2}% (target: ≥90%)",
        coverage * 100.0
    );
}

#[test]
#[ignore]
fn test_algorithm_consistency_across_corpora() {
    // This test ensures that Standard and Transposition algorithms produce
    // consistent results across all corpora

    let corpora = vec![
        ("holbrook", "data/corpora/holbrook.dat"),
        ("aspell", "data/corpora/aspell.dat"),
        ("wikipedia", "data/corpora/wikipedia.dat"),
    ];

    for (name, path) in corpora {
        if !Path::new(path).exists() {
            eprintln!("Skipping {}: {} not found", name, path);
            continue;
        }

        let corpus = MittonCorpus::load(path).unwrap();
        let dict = build_dictionary_from_corpus(&corpus);

        let transducer_std = Transducer::new(dict.clone(), Algorithm::Standard);
        let transducer_trans = Transducer::new(dict, Algorithm::Transposition);

        let mut mismatches = 0;

        for (correct, misspellings) in corpus.errors.iter().take(10) {
            for (misspelling, _) in misspellings.iter().take(5) {
                let results_std: HashSet<_> = transducer_std.query(misspelling, 2).collect();

                let results_trans: HashSet<_> = transducer_trans.query(misspelling, 2).collect();

                // Standard should be a subset of Transposition (or equal)
                if !results_std.is_subset(&results_trans) {
                    eprintln!(
                        "{}: '{}' -> '{}': Standard results not subset of Transposition",
                        name, misspelling, correct
                    );
                    mismatches += 1;
                }
            }
        }

        assert_eq!(
            mismatches, 0,
            "Algorithm consistency check failed for {}: {} mismatches",
            name, mismatches
        );
    }
}

#[test]
#[ignore]
fn test_cross_corpus_correct_words_distinct() {
    // Sanity check: ensure corpora have different characteristics

    let holbrook = MittonCorpus::load("data/corpora/holbrook.dat");
    let aspell = MittonCorpus::load("data/corpora/aspell.dat");
    let wikipedia = MittonCorpus::load("data/corpora/wikipedia.dat");

    if let (Ok(h), Ok(a), Ok(w)) = (holbrook, aspell, wikipedia) {
        println!("Holbrook: {} correct words", h.num_correct_words());
        println!("Aspell: {} correct words", a.num_correct_words());
        println!("Wikipedia: {} correct words", w.num_correct_words());

        // Verify they have different word counts (sanity check)
        assert_ne!(h.num_correct_words(), a.num_correct_words());
        assert_ne!(h.num_correct_words(), w.num_correct_words());
    }
}
