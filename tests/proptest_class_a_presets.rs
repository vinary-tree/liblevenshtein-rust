//! Three-way and metric-law properties for the alignment-expressible presets.

use liblevenshtein::distance::{
    hamming_distance, indel_distance, indel_distance_bounded, standard_distance,
};
use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
use liblevenshtein::transducer::{
    OperationSet, OperationSetBuilder, OperationSetValidationError, OperationType,
    MAX_OPERATION_SET_TOTAL_CONSUMPTION,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

fn config() -> ProptestConfig {
    let mut config = ProptestConfig::with_failure_persistence(FileFailurePersistence::WithSource(
        "proptest-regressions",
    ));
    config.cases = 2_000;
    config
}

fn alphabet() -> impl Strategy<Value = char> {
    prop::sample::select(vec!['a', 'b', 'c', 'é', '\u{301}'])
}

fn text(maximum: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(alphabet(), 0..=maximum).prop_map(|units| units.into_iter().collect())
}

fn equal_length_triple() -> impl Strategy<Value = (String, String, String)> {
    (0usize..=8).prop_flat_map(|length| {
        (
            prop::collection::vec(alphabet(), length),
            prop::collection::vec(alphabet(), length),
            prop::collection::vec(alphabet(), length),
        )
            .prop_map(|(a, b, c)| {
                (
                    a.into_iter().collect(),
                    b.into_iter().collect(),
                    c.into_iter().collect(),
                )
            })
    })
}

fn manual_hamming() -> OperationSet {
    OperationSetBuilder::new()
        .with_match()
        .with_substitution()
        .build()
}

fn manual_indel() -> OperationSet {
    OperationSetBuilder::new()
        .with_match()
        .with_insertion()
        .with_deletion()
        .build()
}

fn manual_bounded_skip() -> OperationSet {
    OperationSetBuilder::new()
        .with_match()
        .with_deletion()
        .build()
}

fn exact_generalized(word: &str, input: &str, operations: OperationSet) -> Option<usize> {
    GeneralizedAutomaton::try_with_operations(32, operations)
        .expect("generated preset must validate")
        .scaled_distance(word, input)
        .expect("small generated alignment must be evaluable")
}

fn lcs_length(left: &str, right: &str) -> usize {
    let left: Vec<_> = left.chars().collect();
    let right: Vec<_> = right.chars().collect();
    let mut previous = vec![0usize; right.len() + 1];
    let mut current = previous.clone();
    for left_unit in left {
        for (column, right_unit) in right.iter().enumerate() {
            current[column + 1] = if left_unit == *right_unit {
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
    let mut input_units = input.chars();
    let mut expected = input_units.next();
    for unit in word.chars() {
        if expected == Some(unit) {
            expected = input_units.next();
        }
    }
    expected
        .is_none()
        .then(|| word.chars().count() - input.chars().count())
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn hamming_preset_manual_grid_and_reference_agree(word in text(10), input in text(10)) {
        let reference = hamming_distance(&word, &input);
        let preset = exact_generalized(&word, &input, OperationSet::hamming());
        let explicit = exact_generalized(&word, &input, manual_hamming());
        prop_assert_eq!(preset, reference);
        prop_assert_eq!(explicit, reference);
    }

    #[test]
    fn indel_preset_manual_grid_and_lcs_reference_agree(word in text(10), input in text(10)) {
        let reference = indel_reference(&word, &input);
        let preset = exact_generalized(&word, &input, OperationSet::indel());
        let explicit = exact_generalized(&word, &input, manual_indel());
        prop_assert_eq!(preset, Some(reference));
        prop_assert_eq!(explicit, Some(reference));
        prop_assert_eq!(indel_distance(&word, &input), reference);
    }

    #[test]
    fn bounded_skip_preset_manual_grid_and_subsequence_reference_agree(word in text(10), input in text(10)) {
        let reference = bounded_skip_reference(&word, &input);
        let preset = exact_generalized(&word, &input, OperationSet::bounded_skip());
        let explicit = exact_generalized(&word, &input, manual_bounded_skip());
        prop_assert_eq!(preset, reference);
        prop_assert_eq!(explicit, reference);
    }

    #[test]
    fn hamming_is_a_metric_on_each_fixed_length_space((a, b, c) in equal_length_triple()) {
        let ab = hamming_distance(&a, &b).expect("equal generated lengths");
        let ba = hamming_distance(&b, &a).expect("equal generated lengths");
        let ac = hamming_distance(&a, &c).expect("equal generated lengths");
        let bc = hamming_distance(&b, &c).expect("equal generated lengths");
        prop_assert_eq!(ab, ba);
        prop_assert_eq!(ab == 0, a == b);
        prop_assert!(ac <= ab + bc);
    }

    #[test]
    fn indel_metric_axioms_hold(a in text(8), b in text(8), c in text(8)) {
        let ab = indel_distance(&a, &b);
        let ba = indel_distance(&b, &a);
        let ac = indel_distance(&a, &c);
        let bc = indel_distance(&b, &c);
        prop_assert_eq!(ab, ba);
        prop_assert_eq!(ab == 0, a == b);
        prop_assert!(ac <= ab + bc);
    }

    #[test]
    fn bounded_indel_returns_the_exact_thresholded_result(a in text(12), b in text(12), budget in 0usize..=12) {
        let exact = indel_distance(&a, &b);
        prop_assert_eq!(
            indel_distance_bounded(&a, &b, budget),
            (exact <= budget).then_some(exact),
        );
    }

    #[test]
    fn inter_metric_ordering_is_preserved(a in text(10), b in text(10)) {
        let levenshtein = standard_distance(&a, &b);
        let indel = indel_distance(&a, &b);
        prop_assert!(levenshtein <= indel);
        prop_assert!(indel <= levenshtein.saturating_mul(2));
        if let Some(hamming) = hamming_distance(&a, &b) {
            prop_assert!(levenshtein <= hamming);
        }
    }

    #[test]
    fn indel_preserves_length_and_parity_invariants(a in text(12), b in text(12)) {
        let left = a.chars().count();
        let right = b.chars().count();
        let distance = indel_distance(&a, &b);
        prop_assert!(left.abs_diff(right) <= distance);
        prop_assert_eq!(distance % 2, left.abs_diff(right) % 2);
        prop_assert!(distance <= left.saturating_add(right));
    }

    #[test]
    fn valid_generated_operation_sets_respect_the_aggregate_guard(
        consumptions in prop::collection::vec((0usize..=8, 0usize..=8), 0..=32),
    ) {
        let mut builder = OperationSetBuilder::new();
        let mut total = 0usize;
        for (source, target) in consumptions {
            if source == 0 && target == 0 {
                continue;
            }
            total += source + target;
            builder = builder.with_operation(OperationType::new(
                source,
                target,
                1.0,
                "generated-positive",
            ));
        }
        prop_assert!(total <= MAX_OPERATION_SET_TOTAL_CONSUMPTION);
        prop_assert_eq!(builder.build().validate(), Ok(()));
    }

    #[test]
    fn generated_nonprogressing_operations_fail_closed(weight in 1u16..=1_000) {
        let operations = OperationSetBuilder::new()
            .with_operation(OperationType::new(0, 0, f64::from(weight), "generated-cycle"))
            .build();
        let rejected = matches!(
            operations.validate(),
            Err(OperationSetValidationError::NoProgress { .. })
        );
        prop_assert!(rejected, "non-progressing operation passed validation");
    }
}
