//! Properties for exact empty-side operation rates.

use liblevenshtein::transducer::{EmptySideRate, OperationSetBuilder, OperationType};
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
    fn deletion_rate_is_exact_and_operation_order_independent(
        specifications in prop::collection::vec((1_usize..=8, 1_u8..=20), 1..=12),
        scalar_count in 0_usize..=128,
        budget in 0_u8..=16,
    ) {
        let build = |items: &[(usize, u8)]| {
            let mut builder = OperationSetBuilder::new();
            for &(consumed, tenths) in items {
                builder = builder.with_operation(OperationType::new(
                    consumed,
                    0,
                    f64::from(tenths) / 10.0,
                    "generated_deletion",
                ));
            }
            builder.build()
        };

        let forward = build(&specifications);
        let mut reversed = specifications.clone();
        reversed.reverse();
        let reverse = build(&reversed);
        let rate = forward.rho_del().expect("generated decimal weights are exact");

        prop_assert_eq!(rate, reverse.rho_del().expect("reordering preserves the scale"));
        let (numerator, denominator) = rate.ratio().expect("a deletion was generated");
        let reference = (numerator as u128) * (scalar_count as u128)
            <= u128::from(budget) * (denominator as u128);
        prop_assert_eq!(rate.fits_budget(scalar_count, budget), reference);

        let maximum = rate.max_consumable(budget);
        prop_assert!(rate.fits_budget(maximum, budget));
        if maximum < usize::MAX {
            prop_assert!(!rate.fits_budget(maximum + 1, budget));
        }
    }

    #[test]
    fn balanced_operations_leave_both_empty_side_rates_infinite(
        arity in 1_usize..=16,
        tenths in 0_u8..=20,
        budget in 0_u8..=16,
    ) {
        let operations = OperationSetBuilder::new()
            .with_operation(OperationType::new(
                arity,
                arity,
                f64::from(tenths) / 10.0,
                "balanced",
            ))
            .build();

        prop_assert_eq!(operations.rho_del(), Ok(EmptySideRate::Infinite));
        prop_assert_eq!(operations.rho_ins(), Ok(EmptySideRate::Infinite));
        prop_assert!(operations.rho_del().unwrap().fits_budget(0, budget));
        prop_assert!(!operations.rho_del().unwrap().fits_budget(1, budget));
    }
}
