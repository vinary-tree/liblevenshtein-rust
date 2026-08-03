//! Refinement properties for Unicode operation restrictions.

use liblevenshtein::transducer::{OperationType, SubstitutionSet};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

fn regression_config() -> ProptestConfig {
    let mut config = ProptestConfig::with_failure_persistence(FileFailurePersistence::WithSource(
        "proptest-regressions",
    ));
    config.cases = 2_000;
    config
}

fn nonempty_unicode() -> impl Strategy<Value = String> {
    prop::collection::vec(any::<char>(), 1..=4).prop_map(|chars| chars.into_iter().collect())
}

proptest! {
    #![proptest_config(regression_config())]

    #[test]
    fn valid_utf8_compatibility_apis_refine_scalar_slice_semantics(
        source in nonempty_unicode(),
        target in nonempty_unicode(),
    ) {
        let mut restriction = SubstitutionSet::new();
        restriction.allow_str(&source, &target);
        let operation = OperationType::with_restriction(
            source.chars().count(),
            target.chars().count(),
            0.5,
            restriction,
            "generated_unicode_restriction",
        );
        let first_target = target
            .chars()
            .next()
            .expect("the strategy generates non-empty targets");

        prop_assert!(operation.can_apply(source.as_bytes(), target.as_bytes()));
        prop_assert!(operation.can_apply_str(&source, &target));
        prop_assert!(operation.applies_to_slices(&source, &target));
        prop_assert!(operation.can_apply_to_source(source.as_bytes()));
        prop_assert!(operation.matches_first_target_char(source.as_bytes(), first_target));
    }
}

#[test]
fn invalid_utf8_retains_the_raw_byte_compatibility_contract() {
    let operation = OperationType::new(2, 1, 0.5, "raw_bytes");
    assert!(operation.can_apply(&[0xff, 0xfe], &[0xfd]));
    assert!(operation.can_apply_to_source(&[0xff, 0xfe]));
}
