use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::distance::{damerau_levenshtein_distance, transposition_distance};
use liblevenshtein::prelude::*;
use proptest::prelude::*;
use std::collections::BTreeMap;

fn word() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::sample::select(vec!['a', 'b', 'c', 'd']), 0..=7)
        .prop_map(|characters| characters.into_iter().collect())
}

fn dictionary() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(word(), 1..=14)
}

fn reference_matches(words: &[String], query: &str, budget: usize) -> BTreeMap<String, usize> {
    words
        .iter()
        .filter_map(|word| {
            let distance = damerau_levenshtein_distance(query, word);
            (distance <= budget).then(|| (word.clone(), distance))
        })
        .collect()
}

fn automaton_matches(words: &[String], query: &str, budget: usize) -> BTreeMap<String, usize> {
    let dictionary = DoubleArrayTrie::from_terms(words);
    Transducer::new(dictionary, Algorithm::DamerauLevenshtein)
        .query_with_distance(query, budget)
        .map(|candidate| (candidate.term, candidate.distance))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    #[test]
    fn exact_set_and_distance_equality_at_k_le_three(
        words in dictionary(),
        query in word(),
        budget in 0usize..=3,
    ) {
        let expected = reference_matches(&words, &query, budget);
        let actual = automaton_matches(&words, &query, budget);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn unrestricted_distance_never_exceeds_osa(source in word(), target in word()) {
        prop_assert!(
            damerau_levenshtein_distance(&source, &target)
                <= transposition_distance(&source, &target)
        );
    }

    #[test]
    fn short_pairs_need_no_interleaved_transposition(source in word(), target in word()) {
        if source.chars().count() <= 2 && target.chars().count() <= 2 {
            prop_assert_eq!(
                damerau_levenshtein_distance(&source, &target),
                transposition_distance(&source, &target)
            );
        }
    }

    #[test]
    fn budget_monotonicity_preserves_reported_distances(
        words in dictionary(),
        query in word(),
        budget in 0usize..3,
    ) {
        let lower = automaton_matches(&words, &query, budget);
        let upper = automaton_matches(&words, &query, budget + 1);

        for (term, distance) in lower {
            prop_assert_eq!(upper.get(&term), Some(&distance));
        }
    }

    #[test]
    fn traversal_order_is_deterministic(
        words in dictionary(),
        query in word(),
        budget in 0usize..=3,
    ) {
        let dictionary = DoubleArrayTrie::from_terms(&words);
        let transducer = Transducer::new(dictionary, Algorithm::DamerauLevenshtein);
        let first: Vec<_> = transducer.query_with_distance(&query, budget).collect();
        let second: Vec<_> = transducer.query_with_distance(&query, budget).collect();
        prop_assert_eq!(first, second);
    }

    #[test]
    fn legacy_osa_automaton_remains_reference_exact_and_order_deterministic(
        words in dictionary(),
        query in word(),
        budget in 0usize..=3,
    ) {
        let expected: BTreeMap<_, _> = words
            .iter()
            .filter_map(|word| {
                let distance = transposition_distance(&query, word);
                (distance <= budget).then(|| (word.clone(), distance))
            })
            .collect();
        let dictionary = DoubleArrayTrie::from_terms(&words);
        let transducer = Transducer::new(dictionary, Algorithm::Transposition);
        let first: Vec<_> = transducer.query_with_distance(&query, budget).collect();
        let second: Vec<_> = transducer.query_with_distance(&query, budget).collect();
        prop_assert_eq!(&first, &second);
        let actual: BTreeMap<_, _> = first
            .into_iter()
            .map(|candidate| (candidate.term, candidate.distance))
            .collect();
        prop_assert_eq!(actual, expected);
    }
}
