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
use libdictenstein::dynamic_dawg::DynamicDawg;
use liblevenshtein::corpus::MittonCorpus;
use liblevenshtein::distance::{
    affine_gap_distance, damerau_levenshtein_distance, hamming_distance, indel_distance,
    transposition_distance,
};
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
use liblevenshtein::transducer::{
    AffineGapParams, LogFrequencyScorer, OperationSet, SubsequenceQueryIterator,
};
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

    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate().take(n + 1) {
        *cell = j;
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

/// Report how often unrestricted Damerau separates from OSA on the complete
/// Birkbeck spelling-error archive. This is an evidence-producing corpus gate,
/// so it is ignored during the default unit suite and run explicitly in Phase 6.
#[cfg(feature = "grep-archives")]
#[test]
#[ignore]
fn birkbeck_true_damerau_divergence_report() {
    let corpus = MittonCorpus::load_birkbeck_zip("data/corpora/birkbeck.zip")
        .expect("the checked-in Birkbeck archive must parse");
    let corrections = corpus.correct_words_sorted();
    let dictionary = DoubleArrayTrie::<()>::from_terms(corrections.iter().copied());
    for &correct in &corrections {
        assert!(
            dictionary.contains(correct),
            "large DoubleArrayTrie inventory lost exact term {correct:?}"
        );
    }
    let transducer = Transducer::new(dictionary, Algorithm::DamerauLevenshtein);

    let mut pairs = 0usize;
    let mut divergent = 0usize;
    let mut true_damerau_better = 0usize;
    let mut true_damerau_worse = 0usize;
    let mut automaton_checked = 0usize;
    let mut separator_examples = Vec::new();

    for (correct, misspellings) in &corpus.errors {
        for (misspelling, frequency) in misspellings {
            pairs = pairs.saturating_add(*frequency);
            let osa = transposition_distance(misspelling, correct);
            let damerau = damerau_levenshtein_distance(misspelling, correct);
            if damerau <= 3 {
                automaton_checked = automaton_checked.saturating_add(1);
                let found = transducer
                    .query_with_distance(misspelling, damerau)
                    .any(|candidate| candidate.term == *correct && candidate.distance == damerau);
                if !found {
                    let candidates: Vec<_> = transducer
                        .query_with_distance(misspelling, damerau)
                        .map(|candidate| (candidate.term, candidate.distance))
                        .collect();
                    let exact_candidates: Vec<_> = transducer.query(correct, 0).collect();
                    panic!(
                        "true-Damerau automaton missed {misspelling:?} -> {correct:?} at exact distance {damerau}; candidates={candidates:?}; exact_dictionary_query={exact_candidates:?}"
                    );
                }
            }
            if osa != damerau {
                divergent = divergent.saturating_add(*frequency);
                if damerau < osa {
                    true_damerau_better = true_damerau_better.saturating_add(*frequency);
                } else {
                    true_damerau_worse = true_damerau_worse.saturating_add(*frequency);
                }
                separator_examples.push((misspelling, correct, osa, damerau));
            }
        }
    }

    // `MittonCorpus` intentionally exposes a HashMap, whose iteration order is
    // randomized. Sort before selecting examples so the evidence report is
    // byte-reproducible across processes and platforms.
    separator_examples.sort_unstable();
    separator_examples.truncate(12);

    let rate = if pairs == 0 {
        0.0
    } else {
        divergent as f64 / pairs as f64
    };
    println!(
        "Birkbeck OSA/true-DL: pairs={pairs}, automaton_checked={automaton_checked}, \
         divergent={divergent}, rate={:.6}%, \
         true_better={true_damerau_better}, true_worse={true_damerau_worse}, \
         examples={separator_examples:?}",
        100.0 * rate,
    );

    assert!(pairs > 0, "the archive must contribute data pairs");
    assert_eq!(
        true_damerau_worse, 0,
        "unrestricted Damerau cannot exceed OSA"
    );
    assert_eq!(divergent, true_damerau_better);
}

/// Check the affine automaton against the independent Gotoh DP on every
/// explicitly corrected pair in the complete Birkbeck archive whose exact cost
/// fits the frozen corpus budget.
///
/// The non-degenerate parameters make a one-character gap cost two while a
/// substitution costs one. This exercises layer changes on real spelling data
/// without letting rare, very distant pairs create an unbounded corpus run.
#[cfg(feature = "grep-archives")]
#[test]
#[ignore]
fn birkbeck_affine_gap_reference_gate() {
    const MAX_COST: usize = 3;

    let corpus = MittonCorpus::load_birkbeck_zip("data/corpora/birkbeck.zip")
        .expect("the checked-in Birkbeck archive must parse");
    let params = AffineGapParams::new(1.0, 1.0, 1.0)
        .expect("the frozen integer parameters must be exactly scalable");

    // HashMap iteration is randomized. A stable order keeps both the report and
    // any minimized failure reproducible across processes and platforms.
    let mut corrections: Vec<_> = corpus.errors.keys().map(String::as_str).collect();
    corrections.sort_unstable();

    let mut observations = 0usize;
    let mut source_pairs = 0usize;
    let mut eligible_pairs = 0usize;
    let mut eligible_observations = 0usize;
    for correct in corrections {
        // The corpus obligation is per pair, while randomized multi-term tests
        // separately establish whole-result-set equality. A singleton trie
        // avoids repeatedly enumerating unrelated Birkbeck candidates and
        // keeps the exhaustive evidence gate proportional to the source data.
        let transducer = Transducer::new(
            DynamicDawg::<()>::from_terms(vec![correct]),
            Algorithm::Standard,
        );
        let mut misspellings: Vec<_> = corpus.errors[correct]
            .iter()
            .map(|(misspelling, frequency)| (misspelling.as_str(), *frequency))
            .collect();
        misspellings.sort_unstable();

        for (misspelling, frequency) in misspellings {
            source_pairs = source_pairs.saturating_add(1);
            observations = observations.saturating_add(frequency);
            let distance = affine_gap_distance(misspelling, correct, params)
                .expect("the bounded Birkbeck strings and costs must not overflow");
            if distance > MAX_COST {
                continue;
            }

            eligible_pairs = eligible_pairs.saturating_add(1);
            eligible_observations = eligible_observations.saturating_add(frequency);
            let found = transducer
                .query_affine_scaled(misspelling, MAX_COST, params)
                .any(|candidate| candidate.term == correct && candidate.distance == distance);
            assert!(
                found,
                "affine automaton missed {misspelling:?} -> {correct:?} at exact cost {distance}"
            );
        }
    }

    println!(
        "Birkbeck affine gap: source_pairs={}, observations={observations}, \
         eligible_pairs={eligible_pairs}, eligible_observations={eligible_observations}, \
         params=(open=1, extend=1, substitution=1), max_cost={MAX_COST}",
        source_pairs,
    );
    assert!(observations > 0, "the archive must contribute observations");
    assert!(
        eligible_pairs > 0,
        "the frozen budget must exercise at least one corrected pair"
    );
}

/// Check each Class-A preset against its independent string reference on every
/// explicitly corrected pair in the complete Birkbeck archive.
///
/// This is intentionally a string-oracle corpus gate. The Phase-0 structural
/// benchmark rejected specialized dictionary walkers, so no backend matrix is
/// applicable to these presets.
#[cfg(feature = "grep-archives")]
#[test]
#[ignore]
fn birkbeck_class_a_preset_reference_gate() {
    let corpus = MittonCorpus::load_birkbeck_zip("data/corpora/birkbeck.zip")
        .expect("the checked-in Birkbeck archive must parse");
    let hamming = GeneralizedAutomaton::try_with_operations(u8::MAX, OperationSet::hamming())
        .expect("the built-in Hamming preset must validate");
    let indel = GeneralizedAutomaton::try_with_operations(u8::MAX, OperationSet::indel())
        .expect("the built-in indel preset must validate");
    let bounded_skip =
        GeneralizedAutomaton::try_with_operations(u8::MAX, OperationSet::bounded_skip())
            .expect("the built-in bounded-skip preset must validate");

    let subsequence_reference = |word: &str, input: &str| {
        let mut expected = input.chars();
        let mut next = expected.next();
        for unit in word.chars() {
            if next == Some(unit) {
                next = expected.next();
            }
        }
        next.is_none()
            .then(|| word.chars().count() - input.chars().count())
    };

    let mut corrections: Vec<_> = corpus.errors.keys().map(String::as_str).collect();
    corrections.sort_unstable();
    let mut pairs = 0usize;
    let mut observations = 0usize;
    let mut hamming_defined = 0usize;
    let mut subsequences = 0usize;

    for correct in corrections {
        let mut misspellings: Vec<_> = corpus.errors[correct]
            .iter()
            .map(|(misspelling, frequency)| (misspelling.as_str(), *frequency))
            .collect();
        misspellings.sort_unstable();
        for (misspelling, frequency) in misspellings {
            pairs = pairs.saturating_add(1);
            observations = observations.saturating_add(frequency);

            let hamming_reference = hamming_distance(correct, misspelling);
            hamming_defined += usize::from(hamming_reference.is_some());
            assert_eq!(
                hamming
                    .scaled_distance(correct, misspelling)
                    .expect("bounded corpus alignment must be evaluable"),
                hamming_reference,
                "Hamming mismatch for {correct:?} -> {misspelling:?}",
            );

            let indel_reference = indel_distance(correct, misspelling);
            assert_eq!(
                indel
                    .scaled_distance(correct, misspelling)
                    .expect("bounded corpus alignment must be evaluable"),
                Some(indel_reference),
                "indel mismatch for {correct:?} -> {misspelling:?}",
            );

            let skip_reference = subsequence_reference(correct, misspelling);
            subsequences += usize::from(skip_reference.is_some());
            assert_eq!(
                bounded_skip
                    .scaled_distance(correct, misspelling)
                    .expect("bounded corpus alignment must be evaluable"),
                skip_reference,
                "bounded-skip mismatch for {correct:?} -> {misspelling:?}",
            );
        }
    }

    println!(
        "Birkbeck Class A: pairs={pairs}, observations={observations}, \
         hamming_defined={hamming_defined}, bounded_skip_subsequences={subsequences}",
    );
    assert!(pairs > 0, "the archive must contribute corrected pairs");
    assert!(hamming_defined > 0, "the corpus must exercise Hamming");
    assert!(subsequences > 0, "the corpus must exercise bounded skip");
}

/// Exercise the Phase-9 subsequence and ranked-value surfaces on the complete
/// Birkbeck archive. Every corrected pair that satisfies the independent flat
/// subsequence predicate is checked. Ranked ordering and value preservation are
/// additionally checked on a deterministic prefix of 128 distinct misspellings
/// against one mapped dictionary containing every Birkbeck correction.
#[cfg(feature = "grep-archives")]
#[test]
#[ignore]
fn birkbeck_phase9_downstream_surface_gate() {
    const RANKED_QUERY_LIMIT: usize = 128;

    let corpus = MittonCorpus::load_birkbeck_zip("data/corpora/birkbeck.zip")
        .expect("the checked-in Birkbeck archive must parse");
    let is_subsequence = |query: &[u8], candidate: &[u8]| {
        let mut matched = 0usize;
        for &unit in candidate {
            matched += usize::from(matched < query.len() && unit == query[matched]);
        }
        matched == query.len()
    };

    let mut corrections: Vec<_> = corpus.errors.keys().map(String::as_str).collect();
    corrections.sort_unstable();
    let mut source_pairs = 0usize;
    let mut observations = 0usize;
    let mut subsequence_pairs = 0usize;
    let mut ranked_queries = Vec::new();
    let mut valued_terms = Vec::with_capacity(corrections.len());

    for correct in corrections {
        let dictionary = DynamicDawg::<()>::from_terms([correct]);
        let mut misspellings: Vec<_> = corpus.errors[correct]
            .iter()
            .map(|(misspelling, frequency)| (misspelling.as_str(), *frequency))
            .collect();
        misspellings.sort_unstable();
        let total_frequency = misspellings
            .iter()
            .map(|(_, frequency)| *frequency as u64)
            .sum::<u64>();
        valued_terms.push((correct.to_owned(), total_frequency));

        for (misspelling, frequency) in misspellings {
            source_pairs = source_pairs.saturating_add(1);
            observations = observations.saturating_add(frequency);
            if ranked_queries.len() < RANKED_QUERY_LIMIT {
                ranked_queries.push(misspelling.to_owned());
            }
            if !is_subsequence(misspelling.as_bytes(), correct.as_bytes()) {
                continue;
            }

            subsequence_pairs = subsequence_pairs.saturating_add(1);
            let found = SubsequenceQueryIterator::from_dictionary(
                &dictionary,
                misspelling.as_bytes().to_vec(),
            )
            .any(|candidate| candidate.units == correct.as_bytes());
            assert!(
                found,
                "subsequence DFS missed Birkbeck pair {misspelling:?} -> {correct:?}"
            );
        }
    }

    let transducer = Transducer::new(
        DynamicDawg::<u64>::from_terms_with_values(valued_terms),
        Algorithm::Standard,
    );
    let mut ranked_checked = 0usize;
    for query in ranked_queries {
        let ranked: Vec<_> = transducer
            .query_suggestions(&query, 2, LogFrequencyScorer)
            .collect();
        let mut projected: Vec<_> = ranked
            .iter()
            .map(|item| (item.term.clone(), item.distance, item.value))
            .collect();
        let mut plain: Vec<_> = transducer.query_values(&query, 2).collect();
        projected.sort_unstable();
        plain.sort_unstable();
        assert_eq!(projected, plain, "ranked multiset mismatch for {query:?}");
        assert!(ranked.windows(2).all(|pair| {
            pair[0].distance < pair[1].distance
                || (pair[0].distance == pair[1].distance
                    && (pair[0].confidence > pair[1].confidence
                        || (pair[0].confidence == pair[1].confidence
                            && pair[0].term <= pair[1].term)))
        }));
        ranked_checked = ranked_checked.saturating_add(1);
    }

    println!(
        "Birkbeck Phase 9: source_pairs={source_pairs}, observations={observations}, \
         subsequence_pairs={subsequence_pairs}, ranked_queries={ranked_checked}",
    );
    assert!(
        source_pairs > 0,
        "the archive must contribute corrected pairs"
    );
    assert!(
        subsequence_pairs > 0,
        "the archive must exercise subsequence acceptance"
    );
    assert_eq!(ranked_checked, RANKED_QUERY_LIMIT);
}
