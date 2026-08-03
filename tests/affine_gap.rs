use libdictenstein::double_array_trie::{DoubleArrayTrie, DoubleArrayTrieChar};
use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgChar, DynamicDawgU64};
use libdictenstein::suffix_automaton::{SuffixAutomaton, SuffixAutomatonChar};
use liblevenshtein::cost::CostScale;
use liblevenshtein::distance::{affine_gap_distance, affine_gap_distance_units};
use liblevenshtein::transducer::{
    AffineGapParams, Algorithm, Candidate, SubstitutionSet, SubstitutionSetChar, Transducer,
};
use liblevenshtein::transducer::{OwnedRestricted, Restricted, RestrictedChar};
use proptest::prelude::*;
use std::collections::BTreeMap;

fn scaled(open: usize, extend: usize, substitution: usize) -> AffineGapParams {
    AffineGapParams::from_scaled(
        CostScale::new(1).expect("unit scale"),
        open,
        extend,
        substitution,
    )
}

fn word() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::sample::select(vec!['a', 'b', 'c']), 0..=7)
        .prop_map(|units| units.into_iter().collect())
}

fn dictionary() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(word(), 1..=12)
}

fn reference_matches(
    words: &[String],
    query: &str,
    budget: usize,
    params: AffineGapParams,
) -> BTreeMap<String, usize> {
    words
        .iter()
        .filter_map(|word| {
            let distance = affine_gap_distance(query, word, params)?;
            (distance <= budget).then(|| (word.clone(), distance))
        })
        .collect()
}

fn automaton_matches(
    words: &[String],
    query: &str,
    budget: usize,
    params: AffineGapParams,
) -> BTreeMap<String, usize> {
    Transducer::new(DoubleArrayTrie::from_terms(words), Algorithm::Standard)
        .query_affine_scaled(query, budget, params)
        .map(|candidate| (candidate.term, candidate.distance))
        .collect()
}

#[test]
fn gotoh_examples_and_public_decimal_surface() {
    let dictionary = DoubleArrayTrie::from_terms(["a", "abcd", "kitten", "sitting"]);
    let transducer = Transducer::new(dictionary, Algorithm::Standard);
    let params = AffineGapParams::new(3.0, 2.0, 10.0).expect("exact integers");
    let result = transducer
        .query_affine("a", 9.0, params)
        .expect("exact budget")
        .find(|candidate| candidate.term == "abcd")
        .expect("gap run is in range");
    assert_eq!(result.distance, 9.0);
    assert_eq!(result.scaled_distance, 9);

    let lev = scaled(0, 1, 1);
    assert_eq!(affine_gap_distance("kitten", "sitting", lev), Some(3));
}

#[test]
fn forward_b5_fuses_query_skip_with_current_dictionary_edge() {
    let words = [String::from("a")];
    let params = scaled(0, 1, 1);

    let matches = automaton_matches(&words, "ba", 1, params);

    assert_eq!(matches.get("a"), Some(&1));
    assert_eq!(matches, reference_matches(&words, "ba", 1, params));
}

#[test]
fn byte_char_u64_and_policy_surfaces_share_the_kernel() {
    let params = scaled(2, 1, 2);
    let terms = ["a", "abcd", "é", "e"];

    let bytes: BTreeMap<_, _> = Transducer::new(
        DynamicDawg::<()>::from_terms(terms),
        Algorithm::MergeAndSplit,
    )
    .query_affine_scaled("a", 5, params)
    .map(|candidate| (candidate.term, candidate.distance))
    .collect();
    assert_eq!(bytes.get("abcd"), Some(&5));

    let chars: BTreeMap<_, _> = Transducer::new(
        DoubleArrayTrieChar::from_terms(terms),
        Algorithm::DamerauLevenshtein,
    )
    .query_affine_scaled("a", 5, params)
    .map(|candidate| (candidate.term, candidate.distance))
    .collect();
    assert_eq!(chars.get("abcd"), Some(&5));

    let u64_dictionary = DynamicDawgU64::<()>::new();
    u64_dictionary.insert_sequence(&[1, 2, 3, 4]);
    let units: Vec<_> = Transducer::new(u64_dictionary, Algorithm::Standard)
        .query_units_affine_scaled(&[1], 5, params)
        .collect();
    assert_eq!(units[0].term, [1, 2, 3, 4]);
    assert_eq!(units[0].distance, 5);

    let mut byte_set = SubstitutionSet::new();
    byte_set.allow('e', 'x');
    let restricted = Transducer::with_policy(
        DoubleArrayTrie::from_terms(["e"]),
        Algorithm::Standard,
        Restricted::new(&byte_set),
    );
    assert_eq!(
        restricted
            .query_affine_scaled("x", 0, params)
            .map(|candidate: Candidate| candidate.term)
            .collect::<Vec<_>>(),
        ["e"]
    );

    let owned = Transducer::with_policy(
        DynamicDawg::<()>::from_terms(["e"]),
        Algorithm::Standard,
        OwnedRestricted::new(byte_set),
    );
    assert_eq!(owned.query_affine_scaled("x", 0, params).count(), 1);

    let mut char_set = SubstitutionSetChar::new();
    char_set.allow('é', 'e');
    let char_policy = Transducer::with_policy(
        DynamicDawgChar::<()>::from_terms(["é"]),
        Algorithm::Standard,
        RestrictedChar::new(&char_set),
    );
    assert_eq!(char_policy.query_affine_scaled("e", 0, params).count(), 1);
}

#[test]
fn suffix_backends_preserve_affine_suffix_completion_semantics() {
    let params = scaled(2, 1, 2);
    let byte_suffix = Transducer::new(
        SuffixAutomaton::<()>::from_text("abcd"),
        Algorithm::Standard,
    );
    assert!(byte_suffix
        .query_affine_scaled("bcd", 0, params)
        .any(|candidate| candidate.term == "bcd"));

    let char_suffix = Transducer::new(
        SuffixAutomatonChar::<()>::from_text("café"),
        Algorithm::Standard,
    );
    assert!(char_suffix
        .query_affine_scaled("afé", 0, params)
        .any(|candidate| candidate.term == "afé"));
}

#[cfg(feature = "pathmap-backend")]
#[test]
fn pathmap_snapshot_uses_the_affine_kernel() {
    use libdictenstein::pathmap::{PathMapDictionary, PathMapDictionaryChar};

    let params = scaled(2, 1, 2);
    let byte_dictionary: PathMapDictionary<()> = PathMapDictionary::from_terms(["a", "abcd"]);
    let byte = Transducer::new(byte_dictionary.snapshot(), Algorithm::Standard);
    assert_eq!(
        byte.query_affine_scaled("a", 5, params)
            .find(|candidate| candidate.term == "abcd")
            .map(|candidate| candidate.distance),
        Some(5)
    );

    let char_dictionary: PathMapDictionaryChar<()> =
        PathMapDictionaryChar::from_terms(["é", "été"]);
    let chars = Transducer::new(char_dictionary.snapshot(), Algorithm::Standard);
    assert_eq!(
        chars
            .query_affine_scaled("é", 4, params)
            .find(|candidate| candidate.term == "été")
            .map(|candidate| candidate.distance),
        Some(4)
    );
}

#[cfg(feature = "persistent-artrie")]
#[test]
fn persistent_artrie_uses_the_affine_kernel() {
    use libdictenstein::persistent_artrie::PersistentARTrie;

    let directory = tempfile::tempdir().expect("temporary persistent dictionary");
    let dictionary: PersistentARTrie<()> =
        PersistentARTrie::create(directory.path().join("affine.artrie"))
            .expect("create persistent dictionary");
    assert!(dictionary.insert("a"));
    assert!(dictionary.insert("abcd"));

    let transducer = Transducer::new(dictionary, Algorithm::Standard);
    assert_eq!(
        transducer
            .query_affine_scaled("a", 5, scaled(2, 1, 2))
            .find(|candidate| candidate.term == "abcd")
            .map(|candidate| candidate.distance),
        Some(5)
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    #[test]
    fn automaton_is_set_and_cost_equal_to_gotoh(
        words in dictionary(),
        query in word(),
        open in 0usize..=4,
        extend in 1usize..=4,
        substitution in 0usize..=5,
        budget in 0usize..=16,
    ) {
        let params = scaled(open, extend, substitution);
        prop_assert_eq!(
            automaton_matches(&words, &query, budget, params),
            reference_matches(&words, &query, budget, params),
        );
    }

    #[test]
    fn distance_is_symmetric_and_has_identity(
        source in word(),
        target in word(),
        open in 0usize..=4,
        extend in 1usize..=4,
        substitution in 1usize..=5,
    ) {
        let params = scaled(open, extend, substitution);
        let forward = affine_gap_distance(&source, &target, params);
        let reverse = affine_gap_distance(&target, &source, params);
        prop_assert_eq!(forward, reverse);
        prop_assert_eq!(forward == Some(0), source == target);
    }

    #[test]
    fn budget_monotonicity_preserves_exact_costs(
        words in dictionary(),
        query in word(),
        open in 0usize..=3,
        extend in 1usize..=3,
        substitution in 1usize..=4,
        budget in 0usize..15,
    ) {
        let params = scaled(open, extend, substitution);
        let lower = automaton_matches(&words, &query, budget, params);
        let upper = automaton_matches(&words, &query, budget + 1, params);
        for (term, distance) in lower {
            prop_assert_eq!(upper.get(&term), Some(&distance));
        }
    }

    #[test]
    fn traversal_order_is_deterministic(
        words in dictionary(),
        query in word(),
        open in 0usize..=3,
        extend in 1usize..=3,
        substitution in 0usize..=4,
        budget in 0usize..=12,
    ) {
        let params = scaled(open, extend, substitution);
        let transducer = Transducer::new(DoubleArrayTrie::from_terms(&words), Algorithm::Standard);
        let first: Vec<_> = transducer.query_affine_scaled(&query, budget, params).collect();
        let second: Vec<_> = transducer.query_affine_scaled(&query, budget, params).collect();
        prop_assert_eq!(first, second);
    }

    #[test]
    fn unit_reference_is_generic(
        source in prop::collection::vec(0u64..4, 0..=6),
        target in prop::collection::vec(0u64..4, 0..=6),
    ) {
        let params = scaled(2, 1, 3);
        prop_assert_eq!(
            affine_gap_distance_units(&source, &target, params),
            affine_gap_distance_units(&target, &source, params),
        );
    }
}
