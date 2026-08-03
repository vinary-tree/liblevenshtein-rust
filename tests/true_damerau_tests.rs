use libdictenstein::double_array_trie::DoubleArrayTrie;
use liblevenshtein::distance::{
    damerau_levenshtein_distance, damerau_levenshtein_distance_bounded, standard_distance,
    transposition_distance,
};
use liblevenshtein::prelude::*;
use std::collections::BTreeMap;

#[test]
fn lowrance_wagner_separating_examples() {
    assert_eq!(damerau_levenshtein_distance("CA", "ABC"), 2);
    assert_eq!(transposition_distance("CA", "ABC"), 3);
    assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
    assert_eq!(damerau_levenshtein_distance("a cat", "an act"), 2);
}

#[test]
fn reference_handles_empty_unicode_and_thresholds() {
    assert_eq!(damerau_levenshtein_distance("", "café"), 4);
    assert_eq!(damerau_levenshtein_distance("新古", "古新"), 1);
    assert_eq!(
        damerau_levenshtein_distance_bounded("CA", "ABC", 2),
        Some(2)
    );
    assert_eq!(damerau_levenshtein_distance_bounded("CA", "ABC", 1), None);
}

#[test]
fn unrestricted_distance_is_never_worse_than_osa_or_standard_on_examples() {
    let words = ["", "a", "ab", "ba", "CA", "ABC", "abcd", "badc", "éa", "aé"];
    for source in words {
        for target in words {
            let damerau = damerau_levenshtein_distance(source, target);
            assert!(damerau <= transposition_distance(source, target));
            assert!(damerau <= standard_distance(source, target));
        }
    }
}

#[test]
fn automaton_finds_the_edit_script_separating_case_with_exact_distance() {
    let terms = ["AC", "ABC", "CAB", "CA", "CBA", "XYZ"];
    let dictionary = DoubleArrayTrie::from_terms(terms);
    let transducer = Transducer::new(dictionary, Algorithm::DamerauLevenshtein);

    let actual: BTreeMap<_, _> = transducer
        .query_with_distance("CA", 2)
        .map(|candidate| (candidate.term, candidate.distance))
        .collect();
    let expected: BTreeMap<_, _> = terms
        .into_iter()
        .filter_map(|term| {
            let distance = damerau_levenshtein_distance("CA", term);
            (distance <= 2).then(|| (term.to_owned(), distance))
        })
        .collect();

    assert_eq!(actual, expected);
    assert_eq!(actual.get("ABC"), Some(&2));
}

#[test]
fn automaton_preserves_standard_paths_at_budget_three() {
    let dictionary = DoubleArrayTrie::from_terms(["scarcely"]);
    let transducer = Transducer::new(dictionary, Algorithm::DamerauLevenshtein);
    let actual: Vec<_> = transducer
        .query_with_distance("scacally", 3)
        .map(|candidate| (candidate.term, candidate.distance))
        .collect();

    assert_eq!(damerau_levenshtein_distance("scacally", "scarcely"), 3);
    assert_eq!(actual, vec![("scarcely".to_owned(), 3)]);
}

#[test]
fn osa_result_order_and_distances_remain_frozen() {
    let dictionary = DoubleArrayTrie::from_terms(["AC", "ABC", "CAB", "CA", "CBA", "XYZ"]);
    let transducer = Transducer::new(dictionary, Algorithm::Transposition);
    let actual: Vec<_> = transducer
        .query_with_distance("CA", 2)
        .map(|candidate| (candidate.term, candidate.distance))
        .collect();

    assert_eq!(
        actual,
        vec![
            ("AC".to_owned(), 1),
            ("CA".to_owned(), 0),
            ("CAB".to_owned(), 1),
            ("CBA".to_owned(), 1),
        ]
    );
}

#[test]
fn selector_round_trips_through_text_surfaces() {
    let algorithm = Algorithm::DamerauLevenshtein;
    assert_eq!(algorithm, Algorithm::DamerauLevenshtein);
    assert_eq!(algorithm.to_string(), "damerau-levenshtein");
    assert_eq!("damerau".parse::<Algorithm>().unwrap(), algorithm);
}

#[cfg(feature = "ffi")]
#[test]
fn selector_round_trips_through_ffi_conversion() {
    let algorithm: Algorithm = liblevenshtein::ffi::LlevAlgorithm::DamerauLevenshtein.into();
    assert_eq!(algorithm, Algorithm::DamerauLevenshtein);
}
