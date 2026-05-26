//! Property-based tests for the fuzzy query iterators (`src/transducer/`).
//!
//! These relate to the TLA+ `PriorityQuery` model (`docs/verification/tla/`),
//! which describes an A* search with an *admissible* heuristic, an optimal first
//! result, and strict distance ordering.
//!
//! IMPORTANT (verified against the source): the Rust `PriorityQueryIterator`
//! deliberately uses an INADMISSIBLE heuristic — `h = query_len - max_consumed`
//! (`src/transducer/priority_query.rs:173`), which overestimates remaining cost
//! whenever the unconsumed query would match for free. Its documented contract
//! is therefore only "Distance-first (approximate lex)" for *fast first-k
//! results* (module header table), NOT optimal ordering. So we test it for the
//! property it actually guarantees: a sound and complete result SET.
//!
//! The strict-ordering / optimality guarantee (the TLA+ `OptimalityProperty` /
//! `ResultsOrdered`) is provided by the separate `OrderedQueryIterator`
//! (`Transducer::query_ordered`, "Strict distance + lexicographic"), which we
//! test for ordering and first-result optimality.

use liblevenshtein::distance::standard_distance;
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::{PriorityCandidate, PriorityQueryIterator};
use proptest::prelude::*;
use std::collections::HashSet;

fn arb_word() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-c]{1,6}").unwrap()
}

fn arb_dict() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(arb_word(), 1..=15)
}

/// Brute-force oracle: dictionary terms within `max_distance` of `query`.
fn linear_scan(words: &[String], query: &str, max_distance: usize) -> HashSet<String> {
    words
        .iter()
        .filter(|w| standard_distance(query, w) <= max_distance)
        .cloned()
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// PriorityQuery (approximate A*): the result SET is sound and complete
    /// (equals the linear-scan oracle) and every reported distance is exact.
    /// Ordering is intentionally NOT asserted — the heuristic is inadmissible.
    #[test]
    fn priority_query_result_set_correct(
        words in arb_dict(),
        query in arb_word(),
        max_distance in 0usize..=3,
    ) {
        let dict = DynamicDawg::<()>::from_terms(words.iter().cloned());
        let got: Vec<PriorityCandidate> =
            PriorityQueryIterator::new(dict.root(), &query, max_distance, Algorithm::Standard)
                .collect();

        let oracle = linear_scan(&words, &query, max_distance);
        let got_terms: HashSet<String> = got.iter().map(|c| c.term.clone()).collect();
        prop_assert_eq!(&got_terms, &oracle, "PriorityQuery result set != linear scan");
        prop_assert_eq!(got_terms.len(), got.len(), "duplicate terms emitted");
        for c in &got {
            prop_assert!(c.distance <= max_distance);
            prop_assert_eq!(c.distance, standard_distance(&query, &c.term));
        }
    }

    /// OrderedQuery: result SET equals the oracle (sound + complete) with exact
    /// distances.
    #[test]
    fn ordered_query_result_set_correct(
        words in arb_dict(),
        query in arb_word(),
        max_distance in 0usize..=3,
    ) {
        let dict = DoubleArrayTrie::from_terms(words.iter().cloned());
        let transducer = Transducer::new(dict, Algorithm::Standard);
        let got: Vec<_> = transducer.query_ordered(&query, max_distance).collect();

        let oracle = linear_scan(&words, &query, max_distance);
        let got_terms: HashSet<String> = got.iter().map(|c| c.term.clone()).collect();
        prop_assert_eq!(&got_terms, &oracle, "OrderedQuery result set != linear scan");
        for c in &got {
            prop_assert_eq!(c.distance, standard_distance(&query, &c.term));
        }
    }

    /// OrderedQuery ordering guarantee: ascending edit distance, then
    /// lexicographic within a distance.
    #[test]
    fn ordered_query_strictly_ordered(
        words in arb_dict(),
        query in arb_word(),
        max_distance in 0usize..=3,
    ) {
        let dict = DoubleArrayTrie::from_terms(words.iter().cloned());
        let transducer = Transducer::new(dict, Algorithm::Standard);
        let got: Vec<_> = transducer.query_ordered(&query, max_distance).collect();
        for w in got.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            prop_assert!(
                a.distance < b.distance || (a.distance == b.distance && a.term <= b.term),
                "ordering violated: ({}, {:?}) then ({}, {:?})",
                a.distance, a.term, b.distance, b.term
            );
        }
    }

    /// OrderedQuery optimality: when any match exists, the first result is
    /// distance-minimal (the admissible-A* property the TLA+ model captures).
    #[test]
    fn ordered_query_first_is_optimal(
        words in arb_dict(),
        query in arb_word(),
        max_distance in 0usize..=3,
    ) {
        let dict = DoubleArrayTrie::from_terms(words.iter().cloned());
        let transducer = Transducer::new(dict, Algorithm::Standard);
        let oracle = linear_scan(&words, &query, max_distance);
        let mut it = transducer.query_ordered(&query, max_distance);
        match it.next() {
            Some(first) => {
                let min_d = oracle
                    .iter()
                    .map(|t| standard_distance(&query, t))
                    .min()
                    .expect("oracle non-empty since OrderedQuery returned a match");
                prop_assert_eq!(first.distance, min_d, "first ordered result not minimal");
            }
            None => prop_assert!(oracle.is_empty(), "OrderedQuery empty but oracle has matches"),
        }
    }
}
