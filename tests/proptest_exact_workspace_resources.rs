//! Executable correspondence for the exact-workspace logical storage algebra.

use std::mem::size_of;

use liblevenshtein::time_series::elastic::{ElasticKernel, ElasticTransducer, QueryPlanStorage};
use liblevenshtein::time_series::{
    DtwConfig, IncompleteReason, OperationOutcome, PageBudget, QuantizationConfig, ResourceKind,
    ResourceLimits,
};
use proptest::prelude::*;

fn dtw_workspace_storage(config: &DtwConfig, query_len: usize) -> (usize, usize) {
    let plan = config
        .query_plan_storage(query_len)
        .expect("small DTW plan storage is representable");
    let width = config
        .column_len(query_len)
        .expect("small DTW frontier width is representable");
    let frontier = width
        .checked_mul(2 * (size_of::<f64>() + size_of::<usize>()))
        .expect("small exact frontier is representable");
    let retained = plan
        .retained_bytes()
        .checked_add(frontier)
        .expect("small workspace retained bytes are representable");
    let peak = plan.construction_peak_bytes().max(retained);
    (retained, peak)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn query_plan_formula_is_exact_and_normalized(
        elements in 0_usize..=10_000,
        retained_per_element in 0_usize..=256,
        transient_per_element in 0_usize..=256,
    ) {
        let storage = QueryPlanStorage::checked_per_element(
            elements,
            retained_per_element,
            transient_per_element,
        ).expect("bounded generated products fit usize");
        let expected_retained = elements * retained_per_element;
        let expected_peak = expected_retained + elements * transient_per_element;
        prop_assert_eq!(storage.retained_bytes(), expected_retained);
        prop_assert_eq!(storage.construction_peak_bytes(), expected_peak);

        let normalized = QueryPlanStorage::new(expected_peak, expected_retained);
        prop_assert_eq!(normalized.retained_bytes(), expected_peak);
        prop_assert_eq!(normalized.construction_peak_bytes(), expected_peak);
    }

    #[test]
    fn exact_workspace_preflight_and_reuse_match_declared_peak(
        query_len in 1_usize..=32,
        collision_members in 1_usize..=8,
    ) {
        let config = DtwConfig::new(query_len);
        let query = vec![0.0; query_len];
        let mut index: ElasticTransducer<DtwConfig, u64> = ElasticTransducer::new(
            QuantizationConfig::for_u8(-1.0, 1.0),
            config,
        );
        for stable_id in 0..collision_members as u64 {
            prop_assert!(index.insert(stable_id, &query));
        }
        let (retained, peak) = dtw_workspace_storage(index.kernel(), query_len);
        prop_assert!(retained <= peak);

        let below = ResourceLimits {
            max_scratch_bytes: peak - 1,
            ..ResourceLimits::default()
        };
        let rejected_at_exact_boundary = matches!(
            index.search_knn_bounded(&query, collision_members, below)
                .expect("finite query is valid"),
            OperationOutcome::Incomplete {
                reason: IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit,
                    requested,
                },
                partial: None,
                continuation: None,
                ..
            } if limit == peak - 1 && requested == peak
        );
        prop_assert!(rejected_at_exact_boundary);

        let workspace_exact = ResourceLimits {
            max_scratch_bytes: peak,
            ..ResourceLimits::default()
        };
        let scorer_only = index.search_knn_bounded(&query, collision_members, workspace_exact)
            .expect("finite query is valid");
        let output_bytes = collision_members
            .checked_mul(size_of::<(u64, f64)>())
            .expect("small generated output is representable");
        let finalization_peak = retained
            .checked_add(output_bytes)
            .expect("small generated finalization peak is representable");
        if finalization_peak > peak {
            let rejected_before_output_commit = matches!(
                scorer_only,
                OperationOutcome::Incomplete {
                    reason: IncompleteReason::BudgetExceeded {
                        resource: ResourceKind::ScratchBytes,
                        limit,
                        requested,
                    },
                    partial: None,
                    continuation: None,
                    ..
                } if limit == peak && requested == finalization_peak
            );
            prop_assert!(rejected_before_output_commit);
        }

        let finalization_exact = ResourceLimits {
            max_scratch_bytes: finalization_peak.max(peak),
            ..ResourceLimits::default()
        };
        let outcome = index.search_knn_bounded(&query, collision_members, finalization_exact)
            .expect("finite query is valid");
        let OperationOutcome::Complete { value, usage } = outcome else {
            prop_assert!(false, "declared finalization peak must admit exact materialization");
            return Ok(());
        };
        prop_assert_eq!(value.len(), collision_members);
        prop_assert_eq!(usage.scratch_bytes, finalization_peak.max(peak));

        // Range construction first admits the same workspace, then must
        // preflight its retained bytes plus the first later arena state.
        let range = index.search_range_bounded(
            &query,
            0.0,
            workspace_exact,
            PageBudget::default(),
        ).expect("finite range query and cutoff are valid");
        let later_state_rejected_before_commit = matches!(
            range,
            OperationOutcome::Incomplete {
                reason: IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit,
                    requested,
                },
                ..
            } if limit == peak && requested > retained
        );
        prop_assert!(later_state_rejected_before_commit);
    }
}

#[test]
fn query_plan_size_overflow_is_tagged_before_construction() {
    assert!(matches!(
        QueryPlanStorage::checked_per_element(usize::MAX, 2, 0),
        Err(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::ScratchBytes,
        })
    ));
    assert!(matches!(
        QueryPlanStorage::checked_per_element(usize::MAX, 1, 1),
        Err(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::ScratchBytes,
        })
    ));
}
