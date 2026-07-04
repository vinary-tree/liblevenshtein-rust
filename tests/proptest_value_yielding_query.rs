//! Property-based tests for the value-yielding query (`Transducer::query_values`, G9).
//!
//! `ValueYieldingQueryIterator` shares its BFS traversal verbatim with the
//! already-validated `query` / `query_with_distance` / value-filtered paths
//! (`tests/transducer_comprehensive.rs`, `tests/proptest_priority_query.rs`).
//! The behavior that is *new* to G9 is reading each final node's value during
//! traversal (and skipping finals whose value is `None`). These properties pin
//! exactly that surface against the real dictionary, which is also the
//! invariant the TLA+ model `docs/verification/tla/ValueYieldingQuery.tla`
//! checks (`ValueCorrectness`, `Soundness`, `NoValuelessYielded`, `Completeness`).
//!
//! All tests use `DoubleArrayTrie` (always compiled) so they run under the
//! default `cargo test`.

#[path = "common/ascii_strategies.rs"]
mod ascii_strategies;

use ascii_strategies::{ascii_word_strategy, small_dict_strategy};
use libdictenstein::double_array_trie::DoubleArrayTrieBuilder;
use liblevenshtein::distance::standard_distance;
use liblevenshtein::prelude::*;
use proptest::prelude::*;
use std::collections::HashSet;

/// Build a `DoubleArrayTrie<usize>` mapping each word to its enumeration index.
/// `from_terms_with_values` always stores `Some(value)`, so EVERY final node has
/// a value: the None-skip branch never fires and the `(term, distance)` set must
/// match `query_with_distance` exactly.
fn build_values_dict(words: &[String]) -> DoubleArrayTrie<usize> {
    let pairs: Vec<(String, usize)> = words
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, w)| (w, i))
        .collect();
    DoubleArrayTrie::from_terms_with_values(pairs)
}

/// Brute-force oracle: the set of UNIQUE dictionary terms within `max` edits.
fn linear_scan_terms(words: &[String], query: &str, max: usize) -> HashSet<String> {
    words
        .iter()
        .filter(|w| standard_distance(query, w) <= max)
        .cloned()
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// VALUE PARITY (the only genuinely new surface of G9): the value read during
    /// traversal equals a fresh dictionary lookup for the same term.
    #[test]
    fn prop_value_yielding_value_parity(
        words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max in 0usize..=2,
    ) {
        let transducer = Transducer::new(build_values_dict(&words), Algorithm::Standard);
        let dict = transducer.dictionary();
        for (term, _dist, value) in transducer.query_values(&query, max) {
            prop_assert_eq!(Some(value), dict.get_value(&term), "value mismatch for {:?}", term);
        }
    }

    /// SET PARITY: with every term valued, `query_values` yields exactly the same
    /// `(term, distance)` pairs as `query_with_distance` (no skipping).
    #[test]
    fn prop_value_yielding_set_parity_vs_query(
        words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max in 0usize..=2,
    ) {
        let transducer = Transducer::new(build_values_dict(&words), Algorithm::Standard);
        let yielded: HashSet<(String, usize)> =
            transducer.query_values(&query, max).map(|(t, d, _)| (t, d)).collect();
        let plain: HashSet<(String, usize)> =
            transducer.query_with_distance(&query, max).map(|c| (c.term, c.distance)).collect();
        prop_assert_eq!(yielded, plain);
    }

    /// SOUNDNESS: every yielded distance is within the threshold.
    #[test]
    fn prop_value_yielding_soundness(
        words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max in 0usize..=2,
    ) {
        let transducer = Transducer::new(build_values_dict(&words), Algorithm::Standard);
        for (_t, d, _v) in transducer.query_values(&query, max) {
            prop_assert!(d <= max, "distance {d} exceeds max {max}");
        }
    }

    /// COMPLETENESS: every unique dictionary term within range is yielded.
    #[test]
    fn prop_value_yielding_completeness(
        words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max in 0usize..=2,
    ) {
        let transducer = Transducer::new(build_values_dict(&words), Algorithm::Standard);
        let yielded: HashSet<String> =
            transducer.query_values(&query, max).map(|(t, _, _)| t).collect();
        for t in linear_scan_terms(&words, &query, max) {
            prop_assert!(yielded.contains(&t), "in-range term {t:?} not yielded");
        }
    }

    /// DEDUP: no term is yielded more than once.
    #[test]
    fn prop_value_yielding_dedup(
        words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max in 0usize..=2,
    ) {
        let transducer = Transducer::new(build_values_dict(&words), Algorithm::Standard);
        let yielded: Vec<String> =
            transducer.query_values(&query, max).map(|(t, _, _)| t).collect();
        let unique: HashSet<&String> = yielded.iter().collect();
        prop_assert_eq!(unique.len(), yielded.len(), "duplicate term yielded: {:?}", yielded);
    }

    /// DISTANCE EXACT: each yielded distance equals the standard edit distance.
    #[test]
    fn prop_value_yielding_distance_exact(
        words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max in 0usize..=2,
    ) {
        let transducer = Transducer::new(build_values_dict(&words), Algorithm::Standard);
        for (t, d, _v) in transducer.query_values(&query, max) {
            prop_assert_eq!(d, standard_distance(&query, &t), "distance mismatch for {:?}", t);
        }
    }

    /// MIXED SKIP: with ~half the finals valueless, `query_values` yields exactly
    /// the in-range terms that HAVE a value, and never a valueless one. This is
    /// the proptest-strength counterpart of the deterministic skip unit test.
    #[test]
    fn prop_value_yielding_mixed_skip(
        words in small_dict_strategy(),
        query in ascii_word_strategy(),
        max in 0usize..=2,
    ) {
        // Deterministic value assignment: a term is valued iff its sorted-unique
        // index is even; the rest are inserted with `None`.
        let mut uniq: Vec<String> = words.iter().cloned().collect::<HashSet<_>>().into_iter().collect();
        uniq.sort();
        let mut builder = DoubleArrayTrieBuilder::<usize>::new();
        for (i, w) in uniq.iter().enumerate() {
            builder.insert_with_value(w, if i % 2 == 0 { Some(i) } else { None });
        }
        let transducer = Transducer::new(builder.build(), Algorithm::Standard);
        let dict = transducer.dictionary();

        let yielded: HashSet<(String, usize)> =
            transducer.query_values(&query, max).map(|(t, d, _)| (t, d)).collect();

        // No valueless term may be yielded.
        for (t, _d) in &yielded {
            prop_assert!(dict.get_value(t).is_some(), "valueless term yielded: {t:?}");
        }

        // Restricted set parity: the `query_with_distance` terms that have a value.
        let expected: HashSet<(String, usize)> = transducer
            .query_with_distance(&query, max)
            .filter(|c| dict.get_value(&c.term).is_some())
            .map(|c| (c.term, c.distance))
            .collect();
        prop_assert_eq!(yielded, expected);
    }
}
