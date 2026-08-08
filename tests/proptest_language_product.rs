//! Differential and algebraic tests for the generic language product.

use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::dynamic_dawg::DynamicDawgU64;
#[cfg(all(feature = "phonetic-rules", feature = "perf-instrumentation"))]
use libdictenstein::{Dictionary, DictionaryNode};
use liblevenshtein::distance::standard_distance;
#[cfg(feature = "phonetic-rules")]
use liblevenshtein::phonetic::nfa::{
    compile, compile_bytes, ProductAutomaton, ProductAutomatonChar,
};
#[cfg(feature = "phonetic-rules")]
use liblevenshtein::phonetic::regex::error::ParseErrorKind;
#[cfg(feature = "phonetic-rules")]
use liblevenshtein::phonetic::regex::{parse, parse_bytes};
#[cfg(all(feature = "phonetic-rules", feature = "perf-instrumentation"))]
use liblevenshtein::transducer::language::LanguageQueryIterator;
use liblevenshtein::transducer::language::{Frontier, LanguageProduct, SmallDfa};
use liblevenshtein::transducer::{Algorithm, Transducer};
use proptest::prelude::*;
use std::collections::BTreeSet;

fn literal_dfa<U: Clone + Eq>(literal: &[U]) -> SmallDfa<U> {
    let mut dfa = SmallDfa::new();
    let mut previous = 0;
    for (index, unit) in literal.iter().enumerate() {
        let next = dfa.add_state(index + 1 == literal.len()).unwrap();
        dfa.add_transition(previous, unit.clone(), next).unwrap();
        previous = next;
    }
    if literal.is_empty() {
        dfa.set_accepting(0, true).unwrap();
    }
    dfa
}

fn snapshot<S: Clone>(frontier: &Frontier<S>) -> Vec<Option<S>> {
    (0..frontier.len())
        .map(|level| frontier.level(level).cloned())
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(600))]

    #[test]
    fn small_dfa_literal_agrees_with_reference_distance(
        pattern in prop::collection::vec(0u64..=3, 0..=6),
        input in prop::collection::vec(0u64..=3, 0..=6),
        budget in 0u8..=3,
    ) {
        let pattern_string: String = pattern.iter().map(|value| char::from(b'a' + *value as u8)).collect();
        let input_string: String = input.iter().map(|value| char::from(b'a' + *value as u8)).collect();
        let expected = standard_distance(&pattern_string, &input_string);
        let product = LanguageProduct::new(literal_dfa(&pattern), budget);
        prop_assert_eq!(
            product.distance_to_language(input),
            (expected <= usize::from(budget)).then_some(expected as u8),
        );
    }

    #[test]
    fn frontier_is_bounded_and_merge_commutes_with_step(
        literal in prop::collection::vec(0u8..=3, 0..=6),
        left_prefix in prop::collection::vec(0u8..=3, 0..=5),
        right_prefix in prop::collection::vec(0u8..=3, 0..=5),
        unit in 0u8..=3,
        budget in 0u8..=3,
    ) {
        let product = LanguageProduct::new(literal_dfa(&literal), budget);
        let mut left = product.initial_frontier();
        for value in left_prefix {
            left = product.step(&left, &value);
            prop_assert_eq!(left.len(), usize::from(budget) + 1);
        }
        let mut right = product.initial_frontier();
        for value in right_prefix {
            right = product.step(&right, &value);
            prop_assert_eq!(right.len(), usize::from(budget) + 1);
        }

        let step_of_union = product.step(&product.merge(&left, &right), &unit);
        let union_of_steps = product.merge(
            &product.step(&left, &unit),
            &product.step(&right, &unit),
        );
        prop_assert_eq!(snapshot(&step_of_union), snapshot(&union_of_steps));
    }
}

#[test]
fn u64_dictionary_language_query_is_backend_generic() {
    let dictionary = DynamicDawgU64::<()>::new();
    for term in [&[10u64, 20][..], &[10, 30], &[99], &[]] {
        dictionary.insert_sequence(term);
    }
    let transducer = Transducer::new(dictionary, Algorithm::Standard);
    let matches: BTreeSet<_> = transducer
        .query_language(literal_dfa(&[10u64, 20]), 1)
        .map(|matched| (matched.units, matched.distance))
        .collect();
    assert_eq!(
        matches,
        BTreeSet::from([(vec![10, 20], 0), (vec![10, 30], 1),])
    );
}

#[test]
fn query_language_covers_empty_unicode_and_exact_budget_boundaries() {
    let dictionary = DoubleArrayTrieChar::from_terms(["", "café", "cafe\u{301}", "茶"]);
    let transducer = Transducer::new(dictionary, Algorithm::Standard);

    let empty: Vec<_> = transducer
        .query_language(literal_dfa::<char>(&[]), 0)
        .map(|matched| matched.units.into_iter().collect::<String>())
        .collect();
    assert_eq!(empty, vec![""]);

    let matches: BTreeSet<_> = transducer
        .query_language(literal_dfa(&['c', 'a', 'f', 'é']), 1)
        .map(|matched| {
            (
                matched.units.into_iter().collect::<String>(),
                matched.distance,
            )
        })
        .collect();
    assert!(matches.contains(&("café".to_string(), 0)));
    // Combining sequences are character sequences, not normalization aliases.
    assert!(!matches.iter().any(|(term, _)| term == "cafe\u{301}"));
}

#[cfg(feature = "phonetic-rules")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn nfa_language_product_agrees_with_legacy_literal_product(
        pattern in prop::string::string_regex("[a-c]{1,6}").unwrap(),
        input in prop::string::string_regex("[a-c]{0,6}").unwrap(),
        budget in 0u8..=3,
    ) {
        let regex = parse(&pattern).unwrap();
        let nfa = compile(&regex).unwrap();
        let legacy = ProductAutomatonChar::new(nfa.clone(), budget);
        let generic = LanguageProduct::new(nfa, budget);
        prop_assert_eq!(
            generic.distance_to_language(input.chars()),
            legacy.min_distance(&input),
        );
    }

    #[test]
    fn byte_nfa_language_product_agrees_with_reference_and_compatibility_wrapper(
        pattern in prop::string::string_regex("[a-c]{1,6}").unwrap(),
        input in prop::string::string_regex("[a-c]{0,6}").unwrap(),
        budget in 0u8..=3,
    ) {
        let nfa = compile_bytes(&parse_bytes(pattern.as_bytes()).unwrap()).unwrap();
        let compatibility = ProductAutomaton::new(nfa.clone(), budget);
        let generic = LanguageProduct::new(nfa, budget);
        let expected = standard_distance(&pattern, &input);
        let expected = (expected <= usize::from(budget)).then_some(expected as u8);
        prop_assert_eq!(generic.distance_to_language(input.bytes()), expected);
        prop_assert_eq!(compatibility.min_distance(input.as_bytes()), expected);
    }
}

#[cfg(feature = "phonetic-rules")]
#[test]
fn regex_query_equals_brute_force_language_oracle() {
    let terms = ["", "a", "b", "ab", "ac", "abc", "bc", "cab", "zz"];
    let dictionary = DoubleArrayTrieChar::from_terms(terms);
    let transducer = Transducer::new(dictionary, Algorithm::Standard);
    let observed: BTreeSet<_> = transducer
        .query_regex("a(b|c)", 1)
        .unwrap()
        .map(|matched| {
            (
                matched.units.into_iter().collect::<String>(),
                matched.distance,
            )
        })
        .collect();

    let expected: BTreeSet<_> = terms
        .into_iter()
        .filter_map(|term| {
            let distance = standard_distance(term, "ab").min(standard_distance(term, "ac"));
            (distance <= 1).then(|| (term.to_string(), distance as u8))
        })
        .collect();
    assert_eq!(observed, expected);
}

#[cfg(feature = "phonetic-rules")]
#[test]
fn query_regex_rejects_automata_above_the_resource_ceiling() {
    let dictionary = DoubleArrayTrieChar::from_terms(["a"]);
    let transducer = Transducer::new(dictionary, Algorithm::Standard);
    // A Thompson literal uses roughly two NFA states per scalar. This remains
    // below the parser's input-size ceiling while exceeding the traversal cap.
    for pattern in ["a".repeat(2_050), "a{1000000}".to_string()] {
        let error = match transducer.query_regex(&pattern, 1) {
            Ok(_) => panic!("oversized regular-language product must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error.kind,
            ParseErrorKind::PatternTooComplex { size, max }
                if size > max && max == liblevenshtein::transducer::language::LANGUAGE_PRODUCT_MAX_STATES
        ));
    }
}

#[cfg(all(feature = "phonetic-rules", feature = "perf-instrumentation"))]
#[test]
fn frontier_query_measurably_avoids_the_full_dictionary_scan() {
    fn edge_count<N: DictionaryNode>(node: N) -> usize {
        node.edges().map(|(_, child)| 1 + edge_count(child)).sum()
    }

    let mut terms: Vec<String> = (0..5_000).map(|i| format!("z{i:04}")).collect();
    terms.extend([
        "phone".into(),
        "fone".into(),
        "phones".into(),
        "stone".into(),
    ]);
    let dictionary = DoubleArrayTrieChar::from_terms(&terms);
    let full_scan_edges = edge_count(dictionary.root());
    let nfa = compile(&parse("(ph|f)one").unwrap()).unwrap();
    let mut query =
        LanguageQueryIterator::from_dictionary(&dictionary, LanguageProduct::new(nfa, 1));
    let matches: BTreeSet<String> = query
        .by_ref()
        .map(|matched| matched.units.into_iter().collect())
        .collect();
    let stats = query.stats();

    assert_eq!(
        matches,
        BTreeSet::from(["fone".into(), "phone".into(), "phones".into()])
    );
    assert!(
        stats.edges_enumerated * 10 < full_scan_edges,
        "frontier edges={} full-scan edges={full_scan_edges}",
        stats.edges_enumerated,
    );
}
