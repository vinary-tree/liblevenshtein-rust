//! Executable invariants for the Phase-9 downstream surfaces.

use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgU64};
use libdictenstein::Dictionary;
use liblevenshtein::filter::{HybridMatcher, NgramIndex};
use liblevenshtein::transducer::language::{balance_lower_bound, is_dyck_word, DyckCorrector};
use liblevenshtein::transducer::{
    Algorithm, AllowedPrefixes, CandidateF64, ContextualCost, ContextualQueryIterator, EditContext,
    LogFrequencyScorer, MatchMode, MatchModeError, NoPruning, OperationCostsF64, PrefixPruner,
    PrefixQueryIterator, QueryIteratorF64, SubsequenceQueryIterator, Transducer,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;
use std::collections::{BTreeMap, BTreeSet};

fn phase9_config() -> ProptestConfig {
    let mut config = ProptestConfig::with_failure_persistence(FileFailurePersistence::Direct(
        "tests/proptest_phase9_downstream.proptest-regressions",
    ));
    config.cases = 2_000;
    config
}

fn text(max: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(prop::sample::select(vec!['a', 'b', 'c']), 0..=max)
        .prop_map(|units| units.into_iter().collect())
}

fn is_subsequence(query: &[u8], candidate: &[u8]) -> bool {
    let mut matched = 0usize;
    for &unit in candidate {
        matched += usize::from(matched < query.len() && unit == query[matched]);
    }
    matched == query.len()
}

fn reference_levenshtein<U: Eq>(left: &[U], right: &[U]) -> usize {
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    for (row, left_unit) in left.iter().enumerate() {
        let mut current = vec![0; right.len() + 1];
        current[0] = row + 1;
        for column in 1..=right.len() {
            current[column] = previous[column]
                .saturating_add(1)
                .min(current[column - 1].saturating_add(1))
                .min(previous[column - 1] + usize::from(left_unit != &right[column - 1]));
        }
        previous = current;
    }
    previous[right.len()]
}

fn generate_balanced(
    kinds: usize,
    max_len: usize,
    prefix: &mut Vec<u64>,
    stack: &mut Vec<usize>,
    output: &mut Vec<Vec<u64>>,
) {
    if stack.is_empty() {
        output.push(prefix.clone());
    }
    if prefix.len() == max_len {
        return;
    }
    if let Some(&kind) = stack.last() {
        prefix.push((kinds + kind) as u64);
        stack.pop();
        generate_balanced(kinds, max_len, prefix, stack, output);
        stack.push(kind);
        prefix.pop();
    }
    if prefix.len() < max_len {
        for kind in 0..kinds {
            prefix.push(kind as u64);
            stack.push(kind);
            generate_balanced(kinds, max_len, prefix, stack, output);
            stack.pop();
            prefix.pop();
        }
    }
}

fn reverse_and_swap(tokens: &[u64], kinds: usize) -> Vec<u64> {
    tokens
        .iter()
        .rev()
        .map(|&token| {
            if token < kinds as u64 {
                token + kinds as u64
            } else {
                token - kinds as u64
            }
        })
        .collect()
}

fn reverse_kind_names(tokens: &[u64], kinds: usize) -> Vec<u64> {
    tokens
        .iter()
        .map(|&token| {
            let is_close = token >= kinds as u64;
            let kind = token as usize % kinds;
            let renamed = kinds - 1 - kind;
            (renamed + usize::from(is_close) * kinds) as u64
        })
        .collect()
}

#[derive(Debug)]
struct CountingPruner {
    enters: usize,
    leaves: usize,
    salt: usize,
}

impl PrefixPruner<u8> for CountingPruner {
    fn enter(&mut self, unit: u8, depth: usize) -> bool {
        self.enters += 1;
        !(unit as usize + depth + self.salt).is_multiple_of(4)
    }

    fn leave(&mut self, _unit: u8, _depth: usize) {
        self.leaves += 1;
    }
}

#[derive(Clone, Copy, Debug)]
struct InvalidDynamicCosts;

impl ContextualCost<u8> for InvalidDynamicCosts {
    fn substitution_cost(
        &self,
        _context: &EditContext<'_, u8>,
        _query: u8,
        _dictionary: u8,
    ) -> Option<f64> {
        Some(-1.0)
    }

    fn insertion_cost(&self, _context: &EditContext<'_, u8>, _dictionary: u8) -> Option<f64> {
        Some(f64::NAN)
    }

    fn deletion_cost(&self, _context: &EditContext<'_, u8>, _query: u8) -> Option<f64> {
        Some(1.0)
    }

    fn min_nonzero_cost(&self) -> f64 {
        1.0
    }
}

proptest! {
    #![proptest_config(phase9_config())]

    #[test]
    fn subsequence_dfs_equals_flat_reference(
        terms in prop::collection::vec(text(7), 0..20),
        query in text(5),
    ) {
        let dictionary = DoubleArrayTrie::from_terms(terms.clone());
        let actual: BTreeSet<_> = SubsequenceQueryIterator::from_dictionary(
            &dictionary,
            query.as_bytes().to_vec(),
        )
        .map(|item| item.units)
        .collect();
        let expected: BTreeSet<_> = terms
            .iter()
            .map(|term| term.as_bytes().to_vec())
            .filter(|term| is_subsequence(query.as_bytes(), term))
            .collect();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn allowed_prefix_pruning_is_exact_set_intersection(
        terms in prop::collection::vec(text(7), 0..20),
        query in text(5),
        selector in any::<u64>(),
    ) {
        let dictionary = DoubleArrayTrie::from_terms(terms.clone());
        let allowed: BTreeSet<Vec<u8>> = terms
            .iter()
            .enumerate()
            .filter(|(index, _)| selector.rotate_left((*index % 64) as u32) & 1 == 1)
            .map(|(_, term)| term.as_bytes().to_vec())
            .collect();
        let pruner = AllowedPrefixes::new(allowed.iter());
        let actual: BTreeSet<_> = SubsequenceQueryIterator::with_pruner(
            dictionary.root(),
            query.as_bytes().to_vec(),
            pruner,
        )
        .map(|item| item.units)
        .collect();
        let expected: BTreeSet<_> = allowed
            .into_iter()
            .filter(|term| is_subsequence(query.as_bytes(), term))
            .collect();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn every_prefix_enter_has_exactly_one_leave_even_when_rejected(
        terms in prop::collection::vec(text(7), 0..20),
        query in text(5),
        salt in any::<u8>(),
    ) {
        let dictionary = DoubleArrayTrie::from_terms(terms);
        let mut traversal = SubsequenceQueryIterator::with_pruner(
            dictionary.root(),
            query.into_bytes(),
            CountingPruner { enters: 0, leaves: 0, salt: salt as usize },
        );
        let _ = traversal.by_ref().count();
        let pruner = traversal.into_pruner();
        prop_assert_eq!(pruner.enters, pruner.leaves);
    }

    #[test]
    fn ranked_values_preserve_multiset_and_two_level_order(
        entries in prop::collection::vec((text(6), 0_u64..10_000), 0..18),
        query in text(5),
        budget in 0_usize..=3,
    ) {
        let unique: BTreeMap<_, _> = entries.into_iter().collect();
        let dictionary = DoubleArrayTrie::from_terms_with_values(
            unique.iter().map(|(term, value)| (term.as_str(), *value)),
        );
        let transducer = Transducer::new(dictionary, Algorithm::Standard);
        let ranked: Vec<_> = transducer
            .query_suggestions(&query, budget, LogFrequencyScorer)
            .collect();
        let mut ranked_multiset: Vec<_> = ranked
            .iter()
            .map(|item| (item.term.clone(), item.distance, item.value))
            .collect();
        let mut plain: Vec<_> = transducer.query_values(&query, budget).collect();
        ranked_multiset.sort();
        plain.sort();
        prop_assert_eq!(ranked_multiset, plain);
        for pair in ranked.windows(2) {
            prop_assert!(pair[0].distance <= pair[1].distance);
            if pair[0].distance == pair[1].distance {
                prop_assert!(pair[0].confidence >= pair[1].confidence);
                if pair[0].confidence == pair[1].confidence {
                    prop_assert!(pair[0].term <= pair[1].term);
                }
            }
        }
    }

    #[test]
    fn match_mode_range_equals_ordered_candidate_filter(
        terms in prop::collection::vec(text(7), 0..20),
        query in text(5),
        first in 0_usize..=4,
        second in 0_usize..=4,
    ) {
        let min_distance = first.min(second);
        let max_distance = first.max(second);
        let transducer = Transducer::new(
            DoubleArrayTrie::from_terms(terms),
            Algorithm::Standard,
        );
        let actual: Vec<_> = transducer
            .query_mode(
                &query,
                MatchMode::Range {
                    min_distance,
                    max_distance,
                },
            )
            .expect("an ordered match-mode range with sorted bounds must be valid")
            .collect();
        let expected: Vec<_> = transducer
            .query_ordered(&query, max_distance)
            .filter(|candidate| candidate.distance >= min_distance)
            .collect();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn prefix_query_no_pruning_equals_breadth_first_query_set(
        terms in prop::collection::vec(text(7), 0..20),
        query in text(5),
        budget in 0_usize..=3,
        algorithm in prop::sample::select(vec![
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
            Algorithm::DamerauLevenshtein,
        ]),
    ) {
        let transducer = Transducer::new(
            DoubleArrayTrie::from_terms(terms),
            algorithm,
        );
        let actual: BTreeMap<_, _> = transducer
            .query_units_with_pruner(query.as_bytes(), budget, NoPruning)
            .map(|candidate| (candidate.units, candidate.distance))
            .collect();
        let expected: BTreeMap<_, _> = transducer
            .query_units_with_distance(query.as_bytes(), budget)
            .map(|candidate| (candidate.term, candidate.distance))
            .collect();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn bracket_projection_is_admissible_against_bruteforce_dyck(
        kinds in 1_usize..=2,
        input in prop::collection::vec(0_u64..4, 0..=5),
    ) {
        let input: Vec<u64> = input.into_iter().map(|token| token % (2 * kinds) as u64).collect();
        let lower = balance_lower_bound(&input, kinds).unwrap();
        let mut language = Vec::new();
        generate_balanced(
            kinds,
            input.len().saturating_mul(2),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut language,
        );
        let exact = language
            .iter()
            .map(|word| reference_levenshtein(&input, word))
            .min()
            .unwrap();
        prop_assert!(lower <= exact, "lower={lower}, exact={exact}, input={input:?}, kinds={kinds}");
    }

    #[test]
    fn exact_multi_kind_dyck_correction_matches_bounded_bruteforce(
        kinds in 1_usize..=2,
        input in prop::collection::vec(0_u64..4, 0..=4),
    ) {
        let input: Vec<u64> = input.into_iter().map(|token| token % (2 * kinds) as u64).collect();
        let correction = DyckCorrector::new(kinds).correct(&input).unwrap();
        let mut language = Vec::new();
        generate_balanced(
            kinds,
            input.len().saturating_mul(2),
            &mut Vec::new(),
            &mut Vec::new(),
            &mut language,
        );
        let brute_force = language
            .iter()
            .map(|word| reference_levenshtein(&input, word))
            .min()
            .unwrap();

        prop_assert_eq!(correction.distance, brute_force);
        prop_assert_eq!(correction.replay(&input), Some(correction.corrected.clone()));
        prop_assert_eq!(is_dyck_word(&correction.corrected, kinds), Ok(true));
        prop_assert_eq!(
            correction.edits.iter().map(|edit| edit.cost()).sum::<usize>(),
            correction.distance
        );
        prop_assert!(balance_lower_bound(&input, kinds).unwrap() <= correction.distance);
    }

    #[test]
    fn exact_dyck_correction_obeys_language_and_edit_invariants(
        kinds in 1_usize..=3,
        input in prop::collection::vec(0_u64..6, 0..=6),
        suffix in prop::collection::vec(0_u64..6, 0..=4),
    ) {
        let alphabet = (2 * kinds) as u64;
        let input: Vec<u64> = input.into_iter().map(|token| token % alphabet).collect();
        let suffix: Vec<u64> = suffix.into_iter().map(|token| token % alphabet).collect();
        let corrector = DyckCorrector::new(kinds);
        let correction = corrector.correct(&input).unwrap();

        prop_assert_eq!(is_dyck_word(&correction.corrected, kinds), Ok(true));
        prop_assert_eq!(correction.replay(&input), Some(correction.corrected.clone()));
        prop_assert_eq!(
            correction.edits.iter().map(|edit| edit.cost()).sum::<usize>(),
            correction.distance,
        );
        prop_assert_eq!(
            reference_levenshtein(&input, &correction.corrected),
            correction.distance,
        );
        prop_assert!(correction.distance <= input.len());
        prop_assert!(correction.corrected.len() <= input.len().saturating_mul(2));

        // A valid result is a fixed point, and deterministic tie-breaking makes
        // repeated correction byte-for-byte and witness-for-witness stable.
        let fixed = corrector.correct(&correction.corrected).unwrap();
        prop_assert_eq!(fixed.distance, 0);
        prop_assert_eq!(fixed.corrected, correction.corrected.clone());
        prop_assert_eq!(corrector.correct(&input).unwrap(), correction.clone());

        // Dyck languages and unit Levenshtein distance are invariant under
        // reverse+open/close swap and under a permutation of bracket kinds.
        let reversed = reverse_and_swap(&input, kinds);
        prop_assert_eq!(corrector.correct(&reversed).unwrap().distance, correction.distance);
        let renamed = reverse_kind_names(&input, kinds);
        prop_assert_eq!(corrector.correct(&renamed).unwrap().distance, correction.distance);

        // Concatenating independently corrected words supplies a valid witness
        // for the concatenated source, hence this exact-distance upper bound.
        let suffix_correction = corrector.correct(&suffix).unwrap();
        let mut concatenated = input.clone();
        concatenated.extend_from_slice(&suffix);
        let concatenated_distance = corrector.correct(&concatenated).unwrap().distance;
        prop_assert!(
            concatenated_distance <= correction.distance + suffix_correction.distance,
        );
    }

    #[test]
    fn context_free_adapter_matches_weighted_iterator(
        terms in prop::collection::vec(text(6), 0..18),
        query in text(5),
        budget in 0_usize..=3,
    ) {
        let dictionary = DoubleArrayTrieChar::from_terms(terms);
        let costs = OperationCostsF64::standard();
        let contextual: BTreeMap<_, _> = ContextualQueryIterator::from_dictionary(
            &dictionary,
            query.chars().collect(),
            budget as f64,
            costs,
        )
        .unwrap()
        .map(|candidate| (candidate.units.iter().collect::<String>(), candidate.distance))
        .collect();
        let weighted: BTreeMap<_, _> = QueryIteratorF64::<_, CandidateF64>::new(
            dictionary.root(),
            query,
            budget as f64,
            Algorithm::Standard,
            costs,
        )
        .map(|candidate| (candidate.term, candidate.distance))
        .collect();
        prop_assert_eq!(contextual, weighted);
    }

    #[test]
    fn contextual_realignment_guard_is_symmetric_and_zero_slack_is_exact(
        left in 0_usize..100,
        right in 0_usize..100,
        minimum in 1_usize..20,
        slack in 0_usize..500,
    ) {
        let forward = left.abs_diff(right).saturating_mul(minimum) <= slack;
        let reverse = right.abs_diff(left).saturating_mul(minimum) <= slack;
        prop_assert_eq!(forward, reverse);
        if left.abs_diff(right).saturating_mul(minimum) == 0 {
            prop_assert_eq!(left, right);
        }
    }

}

#[test]
fn subsequence_surface_is_unit_generic_for_u64_tokens() {
    let dictionary: DynamicDawgU64<()> = DynamicDawgU64::new();
    dictionary.insert_sequence(&[10, 20, 30]);
    dictionary.insert_sequence(&[10, 99, 20, 30]);
    dictionary.insert_sequence(&[30, 20, 10]);
    let actual: BTreeSet<_> =
        SubsequenceQueryIterator::from_dictionary(&dictionary, vec![10, 20, 30])
            .map(|item| item.units)
            .collect();
    assert_eq!(
        actual,
        BTreeSet::from([vec![10, 20, 30], vec![10, 99, 20, 30]])
    );
}

#[test]
fn ranked_values_are_backend_independent_and_support_u64_units() {
    let entries = vec![
        ("cat", 1u64),
        ("bat", 1_000),
        ("cot", 100),
        ("cut", 10),
        ("dog", 10_000),
    ];
    let dat = Transducer::new(
        DoubleArrayTrie::from_terms_with_values(entries.clone()),
        Algorithm::Standard,
    );
    let dawg = Transducer::new(
        DynamicDawg::from_terms_with_values(entries),
        Algorithm::Standard,
    );
    let project = |items: Vec<liblevenshtein::transducer::Suggestion<u64>>| {
        items
            .into_iter()
            .map(|item| {
                (
                    item.term,
                    item.distance,
                    item.value,
                    item.confidence.to_bits(),
                )
            })
            .collect::<Vec<_>>()
    };
    let dat_results = project(
        dat.query_suggestions("cat", 3, LogFrequencyScorer)
            .collect(),
    );
    let dawg_results = project(
        dawg.query_suggestions("cat", 3, LogFrequencyScorer)
            .collect(),
    );
    assert_eq!(dat_results, dawg_results);
    assert_eq!(&dat_results[0].0, "cat");
    assert_eq!(dat_results[0].1, 0);
    assert_eq!(&dat_results[1].0, "bat");

    let token_dictionary = DynamicDawgU64::<u64>::new();
    token_dictionary.insert_sequence_with_value(&[10, 20], 1);
    token_dictionary.insert_sequence_with_value(&[10, 30], 100);
    token_dictionary.insert_sequence_with_value(&[40, 50], 10_000);
    let token_transducer = Transducer::new(token_dictionary, Algorithm::Standard);
    let token_results: Vec<_> = token_transducer
        .query_unit_suggestions(&[10, 20], 2, LogFrequencyScorer)
        .collect();
    assert_eq!(token_results[0].distance, 0);
    assert_eq!(token_results[0].value, 1);
    assert!(token_results
        .windows(2)
        .all(|pair| pair[0].distance <= pair[1].distance));
}

#[test]
fn source_filter_adapters_prune_during_dictionary_dfs() {
    let terms = ["apple", "apply", "banana", "grape", "maple", "orange"];
    let dictionary = DoubleArrayTrie::from_terms(terms);
    let transducer = Transducer::new(dictionary.clone(), Algorithm::Standard);

    let ngrams = NgramIndex::from_iter(2, terms.into_iter().map(str::to_owned));
    let ngram_candidates = ngrams.find_candidates("apple", 1);
    let expected: BTreeSet<_> = ngrams
        .find_candidates("apple", 1)
        .into_iter()
        .filter(|term| reference_levenshtein(b"apple", term.as_bytes()) <= 1)
        .map(|term| term.as_bytes().to_vec())
        .collect();
    let mut ngram_walk = transducer.query_with_pruner("apple", 1, ngrams.prefix_pruner("apple", 1));
    let actual: BTreeSet<_> = ngram_walk.by_ref().map(|item| item.units).collect();
    assert_eq!(actual, expected);
    assert_eq!(
        ngram_walk.pruner().len(),
        AllowedPrefixes::new(ngram_candidates.iter().map(|term| term.as_bytes())).len()
    );
    assert!(ngram_walk.stats().externally_pruned_subtrees > 0);

    let hybrid = HybridMatcher::ngram_only(terms.into_iter().map(str::to_owned), 2);
    let expected: BTreeSet<_> = hybrid
        .filter_candidates("apple", 1)
        .into_iter()
        .filter(|term| reference_levenshtein(b"apple", term.as_bytes()) <= 1)
        .map(|term| term.as_bytes().to_vec())
        .collect();
    let mut hybrid_walk =
        transducer.query_with_pruner("apple", 1, hybrid.prefix_pruner("apple", 1));
    let actual: BTreeSet<_> = hybrid_walk.by_ref().map(|item| item.units).collect();
    assert_eq!(actual, expected);
    assert!(hybrid_walk.stats().externally_pruned_subtrees > 0);
}

#[test]
fn prefix_query_unwinds_balanced_events_when_stopped_early() {
    let dictionary = DoubleArrayTrie::from_terms(["apple", "apply", "banana"]);
    let mut query = PrefixQueryIterator::with_policy_and_pruner(
        dictionary.root(),
        b"apple".to_vec(),
        2,
        Algorithm::Standard,
        liblevenshtein::transducer::Unrestricted,
        CountingPruner {
            enters: 0,
            leaves: 0,
            salt: 0,
        },
        false,
    );
    let _ = query.next();
    let pruner = query.into_pruner();
    assert_eq!(pruner.enters, pruner.leaves);
}

#[test]
fn contextual_dynamic_costs_fail_closed_and_are_counted() {
    let dictionary = DoubleArrayTrie::from_terms(["a"]);
    let mut query = ContextualQueryIterator::from_dictionary(
        &dictionary,
        b"a".to_vec(),
        1.0,
        InvalidDynamicCosts,
    )
    .unwrap();
    assert_eq!(query.by_ref().count(), 0);
    assert!(query.stats().invalid_costs_rejected > 0);
    assert_eq!(query.stats().subtrees_pruned, 1);
}

#[test]
fn match_mode_rejects_an_inverted_public_range() {
    let transducer = Transducer::new(DoubleArrayTrie::from_terms(["cat"]), Algorithm::Standard);
    assert!(matches!(
        transducer.query_mode(
            "cat",
            MatchMode::Range {
                min_distance: 2,
                max_distance: 1,
            },
        ),
        Err(MatchModeError::InvalidRange {
            min_distance: 2,
            max_distance: 1,
        })
    ));
}
