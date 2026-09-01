//! Correspondence and resumption properties for the shared temporal
//! automaton × dictionary product.

use liblevenshtein::time_series::elastic::{BoundedRangeOutcome, Cost, ElasticKernel};
use liblevenshtein::time_series::{
    ErpConfig, ErpTransducer, FrechetConfig, FrechetTransducer, IncompleteReason, MsmConfig,
    MsmTransducer, OperationOutcome, PageBudget, QuantizationConfig, ResourceLimits, TwedConfig,
    TwedTransducer,
};
use proptest::prelude::*;

fn exhaust<K>(
    mut outcome: BoundedRangeOutcome<'_, K, usize>,
    page: PageBudget,
) -> Vec<(usize, Cost<K>)>
where
    K: ElasticKernel,
{
    loop {
        match outcome {
            OperationOutcome::Complete { value, .. } => return value,
            OperationOutcome::Incomplete {
                continuation: Some(next),
                reason: IncompleteReason::BudgetExceeded { .. },
                ..
            } => {
                let stats = next.retained_product_state_stats();
                if stats.frames == 0 {
                    assert_eq!(stats.states, 0, "exact-scan mode retains no product arena");
                } else {
                    assert!(stats.states >= stats.frames);
                }
                outcome = next.resume(page);
            }
            OperationOutcome::Incomplete {
                continuation: Some(_),
                reason,
                ..
            } => panic!("a resumable generated query used a non-budget reason: {reason:?}"),
            OperationOutcome::Incomplete {
                continuation: None,
                reason,
                ..
            } => panic!("a generated bounded product terminated: {reason:?}"),
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn every_shared_temporal_product_equals_its_unpaged_exact_search(
        raw_series in prop::collection::vec(
            prop::collection::vec(-12i16..=12, 0..9),
            0..18,
        ),
        raw_query in prop::collection::vec(-12i16..=12, 0..9),
        cutoff in 0u16..=100,
    ) {
        let series: Vec<Vec<f64>> = raw_series
            .into_iter()
            .map(|candidate| candidate.into_iter().map(f64::from).collect())
            .collect();
        let query: Vec<f64> = raw_query.into_iter().map(f64::from).collect();
        let cutoff = f64::from(cutoff);
        let quantizer = QuantizationConfig::for_u8(-12.0, 12.0);
        let page = PageBudget {
            max_work_units: 1_000_000,
            max_results: 1,
        };

        let msm = MsmTransducer::from_series(
            quantizer.clone(),
            MsmConfig::try_new(1.0).expect("positive MSM operation cost is valid"),
            &series,
        );
        let msm_paged = exhaust(
            msm.search_range_bounded(&query, cutoff, ResourceLimits::default(), page)
                .expect("generated MSM query is finite"),
            page,
        );
        prop_assert_eq!(msm_paged, msm.search_range(&query, cutoff));

        let erp = ErpTransducer::from_series(quantizer.clone(), ErpConfig::new(0.0), &series);
        let erp_paged = exhaust(
            erp.search_range_bounded(&query, cutoff, ResourceLimits::default(), page)
                .expect("generated ERP query is finite"),
            page,
        );
        prop_assert_eq!(erp_paged, erp.search_range(&query, cutoff));

        let twed = TwedTransducer::from_series(
            quantizer.clone(),
            TwedConfig::new(0.5, 1.0),
            &series,
        );
        let twed_paged = exhaust(
            twed.search_range_bounded(&query, cutoff, ResourceLimits::default(), page)
                .expect("generated unit-grid TWED query is finite"),
            page,
        );
        prop_assert_eq!(twed_paged, twed.search_range(&query, cutoff));

        let frechet =
            FrechetTransducer::from_series(quantizer, FrechetConfig::new(), &series);
        let frechet_paged = exhaust(
            frechet
                .search_range_bounded(&query, cutoff, ResourceLimits::default(), page)
                .expect("generated Frechet query is finite"),
            page,
        );
        prop_assert_eq!(frechet_paged, frechet.search_range(&query, cutoff));
    }
}
