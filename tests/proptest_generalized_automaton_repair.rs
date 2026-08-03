//! Differential and property tests for operation-driven generalized acceptance.

use liblevenshtein::distance::standard_distance;
use liblevenshtein::transducer::generalized::{GeneralizedAutomaton, GeneralizedAutomatonError};
use liblevenshtein::transducer::{
    OperationSet, OperationSetBuilder, OperationSetValidationError, OperationType, SubstitutionSet,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

fn regression_config(cases: u32) -> ProptestConfig {
    let mut config = ProptestConfig::with_failure_persistence(FileFailurePersistence::WithSource(
        "proptest-regressions",
    ));
    config.cases = cases;
    config
}

fn hamming_operations() -> OperationSet {
    OperationSetBuilder::new()
        .with_match()
        .with_substitution()
        .build()
}

fn indel_operations() -> OperationSet {
    OperationSetBuilder::new()
        .with_match()
        .with_insertion()
        .with_deletion()
        .build()
}

fn bounded_skip_operations() -> OperationSet {
    OperationSetBuilder::new()
        .with_match()
        .with_deletion()
        .build()
}

fn hamming_reference(left: &str, right: &str) -> Option<usize> {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    (left.len() == right.len()).then(|| {
        left.iter()
            .zip(&right)
            .filter(|(left, right)| left != right)
            .count()
    })
}

fn lcs_length(left: &str, right: &str) -> usize {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    let mut previous = vec![0_usize; right.len() + 1];
    let mut current = previous.clone();
    for left_char in left {
        for (column, right_char) in right.iter().enumerate() {
            current[column + 1] = if left_char == *right_char {
                previous[column] + 1
            } else {
                current[column].max(previous[column + 1])
            };
        }
        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }
    previous[right.len()]
}

fn indel_reference(left: &str, right: &str) -> usize {
    left.chars().count() + right.chars().count() - 2 * lcs_length(left, right)
}

fn bounded_skip_reference(word: &str, input: &str) -> Option<usize> {
    let mut input_chars = input.chars();
    let mut expected = input_chars.next();
    for candidate in word.chars() {
        if expected == Some(candidate) {
            expected = input_chars.next();
        }
    }
    (expected.is_none()).then(|| word.chars().count() - input.chars().count())
}

fn arb_text() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-d]{0,12}").expect("valid property-test regex")
}

proptest! {
    #![proptest_config(regression_config(2_000))]

    #[test]
    fn standard_operations_equal_reference_dp(word in arb_text(), input in arb_text(), budget in 0_u8..=5) {
        let automaton = GeneralizedAutomaton::new(budget);
        prop_assert_eq!(automaton.accepts(&word, &input), standard_distance(&word, &input) <= usize::from(budget));
    }

    #[test]
    fn hamming_operations_reject_unequal_lengths_and_equal_reference(word in arb_text(), input in arb_text(), budget in 0_u8..=5) {
        let automaton = GeneralizedAutomaton::with_operations(budget, hamming_operations());
        let expected = hamming_reference(&word, &input).is_some_and(|cost| cost <= usize::from(budget));
        prop_assert_eq!(automaton.accepts(&word, &input), expected);
    }

    #[test]
    fn indel_operations_equal_lcs_identity(word in arb_text(), input in arb_text(), budget in 0_u8..=5) {
        let automaton = GeneralizedAutomaton::with_operations(budget, indel_operations());
        prop_assert_eq!(automaton.accepts(&word, &input), indel_reference(&word, &input) <= usize::from(budget));
    }

    #[test]
    fn bounded_skip_operations_equal_subsequence_reference(word in arb_text(), input in arb_text(), budget in 0_u8..=5) {
        let automaton = GeneralizedAutomaton::with_operations(budget, bounded_skip_operations());
        let expected = bounded_skip_reference(&word, &input).is_some_and(|cost| cost <= usize::from(budget));
        prop_assert_eq!(automaton.accepts(&word, &input), expected);
    }

    #[test]
    fn acceptance_is_monotone_in_integer_budget(word in arb_text(), input in arb_text(), budget in 0_u8..5) {
        let operations = OperationSetBuilder::new()
            .with_standard_ops()
            .with_operation(OperationType::new(1, 1, 0.15, "cheap_substitution"))
            .build();
        let lower = GeneralizedAutomaton::with_operations(budget, operations.clone());
        let upper = GeneralizedAutomaton::with_operations(budget + 1, operations);
        if lower.accepts(&word, &input) {
            prop_assert!(upper.accepts(&word, &input));
        }
    }
}

#[test]
fn fractional_weights_accumulate_and_are_not_free() {
    let operations = OperationSetBuilder::new()
        .with_match()
        .with_operation(OperationType::new(1, 1, 0.15, "cheap_substitution"))
        .build();

    let strict = GeneralizedAutomaton::with_operations(0, operations.clone());
    assert!(!strict.accepts("a", "b"));

    let budget_one = GeneralizedAutomaton::with_operations(1, operations);
    assert!(budget_one.accepts("aaaaaa", "bbbbbb")); // 6 × 0.15 = 0.90
    assert!(!budget_one.accepts("aaaaaaa", "bbbbbbb")); // 7 × 0.15 = 1.05
    assert_eq!(budget_one.cost_scale().unwrap().denominator(), 20);
}

#[test]
fn hamming_indel_and_bounded_skip_examples_pin_empty_side_semantics() {
    let hamming = GeneralizedAutomaton::with_operations(1, hamming_operations());
    assert!(hamming.accepts("abcd", "abxd"));
    assert!(!hamming.accepts("abcd", "abc"));
    assert!(!hamming.accepts("", "a"));

    let indel = GeneralizedAutomaton::with_operations(1, indel_operations());
    assert!(!indel.accepts("a", "b"));
    let indel_two = GeneralizedAutomaton::with_operations(2, indel_operations());
    assert!(indel_two.accepts("a", "b"));

    let skip = GeneralizedAutomaton::with_operations(2, bounded_skip_operations());
    assert!(skip.accepts("abcd", "ad"));
    assert!(!skip.accepts("ad", "abcd"));
}

#[test]
fn restricted_unicode_operations_count_scalars_not_utf8_bytes() {
    let mut restriction = SubstitutionSet::new();
    restriction.allow_str("é", "e");
    let operations = OperationSetBuilder::new()
        .with_match()
        .with_operation(OperationType::with_restriction(
            1,
            1,
            0.5,
            restriction,
            "accent_fold",
        ))
        .build();
    let automaton = GeneralizedAutomaton::with_operations(1, operations);

    assert!(automaton.accepts("café", "cafe"));
    assert_eq!(automaton.scaled_distance("é", "e"), Ok(Some(1)));
    assert_eq!(automaton.cost_scale().unwrap().denominator(), 2);
}

#[test]
fn non_finite_weights_are_reported_and_bool_api_fails_closed() {
    let operations = OperationSetBuilder::new()
        .with_match()
        .with_operation(OperationType::new(1, 1, f64::INFINITY, "invalid"))
        .build();
    let automaton = GeneralizedAutomaton::with_operations(1, operations.clone());
    assert!(matches!(
        automaton.try_accepts("a", "b"),
        Err(GeneralizedAutomatonError::OperationSet(
            OperationSetValidationError::InvalidWeight { .. }
        ))
    ));
    assert!(!automaton.accepts("a", "b"));
    assert!(matches!(
        GeneralizedAutomaton::try_with_operations(1, operations),
        Err(GeneralizedAutomatonError::OperationSet(
            OperationSetValidationError::InvalidWeight { .. }
        ))
    ));
}
