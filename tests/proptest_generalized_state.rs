//! Properties for the bounded streaming generalized-state compatibility API.

use liblevenshtein::cost::CostScale;
use liblevenshtein::transducer::generalized::{
    CharacteristicVector, GeneralizedState, GeneralizedStateError, GeneralizedTransitionInput,
};
use liblevenshtein::transducer::{OperationSetBuilder, OperationType};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

fn regression_config() -> ProptestConfig {
    let mut config = ProptestConfig::with_failure_persistence(FileFailurePersistence::WithSource(
        "proptest-regressions",
    ));
    config.cases = 2_000;
    config
}

proptest! {
    #![proptest_config(regression_config())]

    #[test]
    fn applicable_operation_order_preserves_the_least_exact_cost(
        left_numerator in 1_u8..=9,
        right_numerator in 1_u8..=9,
        reverse in any::<bool>(),
    ) {
        let left = f64::from(left_numerator) / 10.0;
        let right = f64::from(right_numerator) / 10.0;
        let mut builder = OperationSetBuilder::new();
        if reverse {
            builder = builder
                .with_operation(OperationType::new(1, 1, right, "right"))
                .with_operation(OperationType::new(1, 1, left, "left"));
        } else {
            builder = builder
                .with_operation(OperationType::new(1, 1, left, "left"))
                .with_operation(OperationType::new(1, 1, right, "right"));
        }
        let operations = builder.build();
        let scale = CostScale::for_operations(&operations)
            .expect("generated finite decimal weights have an exact scale");
        let expected = scale
            .to_scaled(left.min(right))
            .expect("minimum weight belongs to the common scale");

        let state = GeneralizedState::initial(1);
        let vector = CharacteristicVector::new('b', "$a");
        let input = GeneralizedTransitionInput::new(
            &operations,
            &vector,
            "a",
            None,
            "$a",
            'b',
            1,
        );
        let next = state
            .try_transition(input)
            .expect("generated operation set is representable")
            .expect("at least one sub-unit substitution fits budget one");
        let costs = next.positions().map(|position| position.errors()).collect::<Vec<_>>();

        prop_assert_eq!(next.cost_scale(), scale);
        prop_assert_eq!(costs, vec![expected]);
    }

    #[test]
    fn unrepresentable_arity_is_reported_before_transition(
        consume_x in 3_usize..=16,
        consume_y in 1_usize..=16,
    ) {
        let operations = OperationSetBuilder::new()
            .with_operation(OperationType::new(consume_x, consume_y, 1.0, "wide"))
            .build();
        let state = GeneralizedState::initial(1);
        let vector = CharacteristicVector::new('b', "$a");
        let input = GeneralizedTransitionInput::new(
            &operations,
            &vector,
            "a",
            None,
            "$a",
            'b',
            1,
        );

        prop_assert_eq!(
            state.try_transition(input),
            Err(GeneralizedStateError::UnsupportedOperationArity {
                name: "wide".into(),
                consume_x,
                consume_y,
            })
        );
    }
}

#[test]
fn hamming_stream_does_not_invent_a_deletion_to_reach_a_later_match() {
    let operations = OperationSetBuilder::new()
        .with_match()
        .with_substitution()
        .build();
    let state = GeneralizedState::initial(2);
    let vector = CharacteristicVector::new('a', "$$ba");
    let input = GeneralizedTransitionInput::new(&operations, &vector, "ba", None, "$$ba", 'a', 1);
    let next = state
        .try_transition(input)
        .expect("Hamming costs are representable")
        .expect("substitution remains available");

    assert!(next.positions().any(|position| position.offset() == 0));
    assert!(next.positions().all(|position| position.offset() != 1));
}
