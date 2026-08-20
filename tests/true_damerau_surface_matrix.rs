//! Phase 6 integration matrix for unrestricted Damerau–Levenshtein.
//!
//! The transition kernel is generic over dictionary units and substitution
//! policy, while the public crate exposes several result iterators. These tests
//! keep one Lowrance–Wagner oracle at the center and drive every applicable
//! combination through the same separating edit script.

use libdictenstein::double_array_trie::{
    DoubleArrayTrie, DoubleArrayTrieChar, DoubleArrayTrieZipper,
};
use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgChar, DynamicDawgU64};
use libdictenstein::suffix_automaton::{SuffixAutomaton, SuffixAutomatonChar};
use libdictenstein::{
    CharUnit, Dictionary, DictionaryNode, MappedDictionary, MappedDictionaryNode,
};
use liblevenshtein::distance::damerau_levenshtein_distance;
use liblevenshtein::prelude::*;
use liblevenshtein::transducer::substitution_policy::{
    OwnedRestricted, Restricted, RestrictedChar, SubstitutionPolicyFor, Unrestricted,
};
use liblevenshtein::transducer::{
    PriorityQueryIterator, SubstitutionSet, SubstitutionSetChar, UnitCandidate, ZipperQueryIterator,
};
use std::collections::{BTreeMap, BTreeSet, HashSet};

const QUERY: &str = "CA";
const BUDGET: usize = 2;
const TERMS: [&str; 7] = ["", "AC", "ABC", "CA", "CAB", "CBA", "XYZ"];

fn expected() -> BTreeMap<String, usize> {
    TERMS
        .iter()
        .filter_map(|term| {
            let distance = damerau_levenshtein_distance(QUERY, term);
            (distance <= BUDGET).then(|| ((*term).to_owned(), distance))
        })
        .collect()
}

fn candidate_map<I>(iter: I) -> BTreeMap<String, usize>
where
    I: IntoIterator<Item = Candidate>,
{
    iter.into_iter()
        .map(|candidate| (candidate.term, candidate.distance))
        .collect()
}

fn assert_string_backend<D, F>(factory: F)
where
    D: Dictionary,
    D::Node: DictionaryNode,
    <D::Node as DictionaryNode>::Unit: CharUnit,
    Unrestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
    F: Fn() -> D,
{
    let expected = expected();

    let plain: BTreeSet<_> = Transducer::new(factory(), Algorithm::DamerauLevenshtein)
        .query(QUERY, BUDGET)
        .collect();
    assert_eq!(plain, expected.keys().cloned().collect());

    let candidates = candidate_map(
        Transducer::new(factory(), Algorithm::DamerauLevenshtein)
            .query_with_distance(QUERY, BUDGET),
    );
    assert_eq!(candidates, expected);

    let units: BTreeMap<_, _> = Transducer::new(factory(), Algorithm::DamerauLevenshtein)
        .query_units_with_distance(&<D::Node as DictionaryNode>::Unit::from_str(QUERY), BUDGET)
        .map(|candidate| {
            (
                <D::Node as DictionaryNode>::Unit::to_string(&candidate.term),
                candidate.distance,
            )
        })
        .collect();
    assert_eq!(units, expected);

    let ordered: Vec<_> = Transducer::new(factory(), Algorithm::DamerauLevenshtein)
        .query_ordered(QUERY, BUDGET)
        .collect();
    assert!(ordered
        .windows(2)
        .all(|pair| pair[0].distance <= pair[1].distance));
    let ordered_map: BTreeMap<_, _> = ordered
        .into_iter()
        .map(|candidate| (candidate.term, candidate.distance))
        .collect();
    assert_eq!(ordered_map, expected);

    let ranked: Vec<_> = Transducer::new(factory(), Algorithm::DamerauLevenshtein)
        .query_ranked(QUERY, BUDGET)
        .collect();
    let ranked_map: BTreeMap<_, _> = ranked
        .into_iter()
        .map(|candidate| (candidate.term, candidate.distance))
        .collect();
    assert_eq!(ranked_map, expected);

    let priority: BTreeMap<_, _> = PriorityQueryIterator::new(
        factory().root(),
        QUERY,
        BUDGET,
        Algorithm::DamerauLevenshtein,
    )
    .map(|candidate| (candidate.term, candidate.distance))
    .collect();
    assert_eq!(priority, expected);
}

#[test]
fn byte_and_char_prefix_backends_share_exact_true_damerau_semantics() {
    assert_string_backend(|| DoubleArrayTrie::from_terms(TERMS));
    assert_string_backend(|| DynamicDawg::<()>::from_terms(TERMS));
    assert_string_backend(|| DoubleArrayTrieChar::from_terms(TERMS));
    assert_string_backend(|| DynamicDawgChar::<()>::from_terms(TERMS));
}

#[cfg(feature = "pathmap-backend")]
#[test]
fn pathmap_byte_and_char_backends_share_exact_true_damerau_semantics() {
    use libdictenstein::pathmap::{PathMapDictionary, PathMapDictionaryChar};

    assert_string_backend(|| PathMapDictionary::<()>::from_terms(TERMS));
    assert_string_backend(|| PathMapDictionaryChar::<()>::from_terms(TERMS));
}

#[cfg(feature = "persistent-artrie")]
#[test]
fn persistent_byte_and_char_backends_share_exact_true_damerau_semantics() {
    use libdictenstein::persistent_artrie::char::PersistentARTrieChar;
    use libdictenstein::persistent_artrie::PersistentARTrie;

    fn byte_dictionary() -> PersistentARTrie<()> {
        let temporary = Box::leak(Box::new(
            tempfile::tempdir().expect("temporary persistent true-Damerau dictionary"),
        ));
        let dictionary = PersistentARTrie::create(temporary.path().join("damerau.artrie"))
            .expect("create persistent true-Damerau dictionary");
        for term in TERMS {
            dictionary.insert(term);
        }
        dictionary
    }

    fn char_dictionary() -> PersistentARTrieChar<()> {
        PersistentARTrieChar::try_from_iter(TERMS)
            .expect("construct persistent true-Damerau character dictionary")
    }

    assert_string_backend(byte_dictionary);
    assert_string_backend(char_dictionary);
}

#[test]
fn substring_backends_use_their_documented_prefix_distance_with_true_damerau() {
    let byte_results: Vec<_> = Transducer::new(
        SuffixAutomaton::<()>::from_text("--ABC--"),
        Algorithm::DamerauLevenshtein,
    )
    .query_with_distance(QUERY, BUDGET)
    .collect();
    // Suffix dictionaries intentionally use prefix/subsequence completion:
    // unmatched query suffixes are free once a represented substring has
    // matched. The whole-string CA→ABC separator is therefore masked by the
    // cheaper C/CA-prefix path and is not an applicable reference-DP equality.
    assert!(byte_results
        .iter()
        .any(|candidate| candidate.term == "ABC" && candidate.distance == 1));

    let char_results: Vec<_> = Transducer::new(
        SuffixAutomatonChar::<()>::from_text("λABC茶"),
        Algorithm::DamerauLevenshtein,
    )
    .query_with_distance(QUERY, BUDGET)
    .collect();
    assert!(char_results
        .iter()
        .any(|candidate| candidate.term == "ABC" && candidate.distance == 1));
}

#[test]
fn byte_policy_matrix_keeps_the_history_carrying_variant() {
    let mut substitutions = SubstitutionSet::new();
    substitutions.allow('x', 'y');

    let borrowed = Transducer::with_policy(
        DoubleArrayTrie::from_terms(["AC", "ABC", "CA"]),
        Algorithm::DamerauLevenshtein,
        Restricted::new(&substitutions),
    );
    assert_eq!(
        borrowed
            .query_with_distance(QUERY, BUDGET)
            .find(|candidate| candidate.term == "ABC")
            .map(|candidate| candidate.distance),
        Some(2)
    );

    let owned = Transducer::with_policy(
        DynamicDawg::<()>::from_terms(["AC", "ABC", "CA"]),
        Algorithm::DamerauLevenshtein,
        OwnedRestricted::new(substitutions),
    );
    assert_eq!(
        owned
            .query_with_distance(QUERY, BUDGET)
            .find(|candidate| candidate.term == "ABC")
            .map(|candidate| candidate.distance),
        Some(2)
    );
}

#[test]
fn character_policy_matrix_preserves_unicode_and_true_damerau() {
    let mut substitutions = SubstitutionSetChar::new();
    substitutions.allow('é', 'e');
    let transducer = Transducer::with_policy(
        DynamicDawgChar::<()>::from_terms(["ABC", "CA", "éCA"]),
        Algorithm::DamerauLevenshtein,
        RestrictedChar::new(&substitutions),
    );

    assert_eq!(
        transducer
            .query_with_distance(QUERY, BUDGET)
            .find(|candidate| candidate.term == "ABC")
            .map(|candidate| candidate.distance),
        Some(2)
    );
    assert!(transducer.query("eCA", 0).any(|term| term == "éCA"));
}

#[test]
fn mapped_result_surfaces_preserve_distances_values_and_filters() {
    fn assert_mapped<D>(dictionary: D)
    where
        D: MappedDictionary<Value = u16>,
        D::Node: MappedDictionaryNode<Value = u16> + DictionaryNode,
        <D::Node as DictionaryNode>::Unit: CharUnit,
        Unrestricted: SubstitutionPolicyFor<<D::Node as DictionaryNode>::Unit>,
    {
        let transducer = Transducer::new(dictionary, Algorithm::DamerauLevenshtein);
        let values: BTreeMap<_, _> = transducer
            .query_values(QUERY, BUDGET)
            .map(|(term, distance, value)| (term, (distance, value)))
            .collect();
        assert_eq!(values.get("ABC"), Some(&(2, 7)));

        let filtered = candidate_map(transducer.query_filtered(QUERY, BUDGET, |value| *value == 7));
        assert_eq!(filtered, BTreeMap::from([("ABC".to_owned(), 2)]));

        let allowed = HashSet::from([7u16]);
        let selected = candidate_map(transducer.query_by_value_set(QUERY, BUDGET, &allowed));
        assert_eq!(selected, BTreeMap::from([("ABC".to_owned(), 2)]));
    }

    let entries = [("AC", 3u16), ("ABC", 7), ("CA", 11), ("XYZ", 13)];
    assert_mapped(DoubleArrayTrie::from_terms_with_values(entries));
    assert_mapped(DynamicDawg::from_terms_with_values(entries));
}

#[test]
fn u64_units_and_values_preserve_true_damerau_semantics() {
    let dictionary = DynamicDawgU64::<()>::new();
    dictionary.insert_sequence(&[1, 2, 3]);
    dictionary.insert_sequence(&[3, 1]);
    dictionary.insert_sequence(&[1, 3]);
    let transducer = Transducer::new(dictionary, Algorithm::DamerauLevenshtein);

    let results: BTreeMap<_, _> = transducer
        .query_units_with_distance(&[3, 1], 2)
        .map(|candidate: UnitCandidate<u64>| (candidate.term, candidate.distance))
        .collect();
    assert_eq!(results.get(&vec![1, 2, 3]), Some(&2));

    let valued = DynamicDawgU64::<u64>::new();
    valued.insert_sequence_with_value(&[1, 2, 3], 29u64);
    let valued_transducer = Transducer::new(valued, Algorithm::DamerauLevenshtein);
    let results: Vec<_> = valued_transducer.query_units_values(&[3, 1], 2).collect();
    assert_eq!(results, vec![(vec![1, 2, 3], 2, 29)]);
}

#[cfg(feature = "persistent-artrie")]
#[test]
fn persistent_u64_backend_preserves_true_damerau_semantics() {
    use libdictenstein::persistent_artrie::PersistentARTrieU64Compact;

    let dictionary: PersistentARTrieU64Compact<()> = PersistentARTrieU64Compact::new();
    dictionary.insert_sequence(&[1, 2, 3]);
    let transducer = Transducer::new(dictionary, Algorithm::DamerauLevenshtein);
    let results: Vec<_> = transducer.query_units_with_distance(&[3, 1], 2).collect();
    assert!(results
        .iter()
        .any(|candidate| candidate.term == [1, 2, 3] && candidate.distance == 2));
}

#[test]
fn zipper_surface_preserves_true_damerau_semantics() {
    use liblevenshtein::transducer::transition::{initial_state, transition_state};

    let mut state = initial_state(QUERY.len(), BUDGET, Algorithm::DamerauLevenshtein);
    for unit in b"ABC" {
        state = transition_state(
            &state,
            Unrestricted,
            *unit,
            QUERY.as_bytes(),
            BUDGET,
            Algorithm::DamerauLevenshtein,
            false,
        )
        .expect("the non-pooled transition remains viable through ABC");
    }
    assert_eq!(
        state.infer_distance(QUERY.len()),
        Some(2),
        "non-pooled state: {:?}",
        state.positions()
    );

    let mut pool = liblevenshtein::transducer::StatePool::new();
    let direct = liblevenshtein::transducer::AutomatonZipper::new(
        QUERY.as_bytes(),
        BUDGET,
        Algorithm::DamerauLevenshtein,
    )
    .transition(b'A', &mut pool)
    .and_then(|zipper| zipper.transition(b'B', &mut pool))
    .and_then(|zipper| zipper.transition(b'C', &mut pool))
    .expect("the direct automaton zipper remains viable through ABC");
    assert_eq!(
        direct.infer_distance(QUERY.len()),
        Some(2),
        "pooled zipper state: {:?}",
        direct.state().positions()
    );

    let dictionary = DoubleArrayTrie::from_terms(["AC", "ABC", "CA"]);
    let automaton = liblevenshtein::transducer::AutomatonZipper::new(
        QUERY.as_bytes(),
        BUDGET,
        Algorithm::DamerauLevenshtein,
    );
    let mut intersection = liblevenshtein::transducer::IntersectionZipper::new(
        DoubleArrayTrieZipper::new_from_dict(&dictionary),
        automaton,
    );
    let mut intersection_pool = liblevenshtein::transducer::StatePool::new();
    for wanted in b"ABC" {
        let next = {
            intersection
                .children(&mut intersection_pool)
                .find_map(|(label, child)| (label == *wanted).then_some(child))
                .expect("dictionary×automaton zipper has the ABC path")
        };
        intersection = next;
    }
    assert_eq!(intersection.distance(), Some(2));

    let zipper = DoubleArrayTrieZipper::new_from_dict(&dictionary);
    let results = candidate_map(ZipperQueryIterator::new(
        zipper,
        QUERY,
        BUDGET,
        Algorithm::DamerauLevenshtein,
    ));
    assert_eq!(results.get("ABC"), Some(&2), "zipper results: {results:?}");
}

#[test]
fn repeated_calls_are_order_deterministic_on_every_iterator_surface() {
    let dictionary = DoubleArrayTrie::from_terms(TERMS);
    let transducer = Transducer::new(dictionary, Algorithm::DamerauLevenshtein);

    let plain_a: Vec<_> = transducer.query_with_distance(QUERY, BUDGET).collect();
    let plain_b: Vec<_> = transducer.query_with_distance(QUERY, BUDGET).collect();
    assert_eq!(plain_a, plain_b);

    let ordered_a: Vec<_> = transducer.query_ordered(QUERY, BUDGET).collect();
    let ordered_b: Vec<_> = transducer.query_ordered(QUERY, BUDGET).collect();
    assert_eq!(ordered_a, ordered_b);
}
