//! Integration and property tests for the public `ElasticKernel` seam.
//!
//! The production MSM adapter is covered by the unchanged MSM suites. This
//! file intentionally supplies a second, tiny kernel so accidental MSM-shaped
//! assumptions in the generic walker cannot hide behind the type alias.

use liblevenshtein::cost::{CostMonoid, WeightedCost};
use liblevenshtein::time_series::elastic::interval::interval_dist;
use liblevenshtein::time_series::elastic::MetricElasticKernel;
use liblevenshtein::time_series::elastic::{
    Cost, ElasticKernel, ElasticTransducer, QueryPlanStorage,
};
use liblevenshtein::time_series::{
    AuditedMetricTimeSeriesIndex, DtwConfig, ErpConfig, FrechetConfig, IncompleteReason,
    MetricErpTransducer, MetricFrechetTransducer, MetricMsmConfig, MetricMsmKernel,
    MetricTwedConfig, MsmKernel, QuantizationConfig, TwedConfig,
};
use proptest::prelude::*;

/// Pointwise L1 distance restricted to equal-length series.
#[derive(Clone, Copy, Debug, Default)]
struct PointwiseL1;

impl ElasticKernel for PointwiseL1 {
    const IS_METRIC: bool = true;

    type Monoid = WeightedCost;
    type Carry = ();
    type QueryPlan = ();

    fn query_plan_storage(&self, _query_len: usize) -> Result<QueryPlanStorage, IncompleteReason> {
        Ok(QueryPlanStorage::EMPTY)
    }

    fn column_len(&self, _query_len: usize) -> Option<usize> {
        Some(1)
    }

    fn final_row(&self, _query_len: usize) -> usize {
        0
    }

    fn step_column(
        &self,
        previous: &[Cost<Self>],
        query: &[f64],
        interval: (f64, f64),
        _carry: Option<Self::Carry>,
        depth: usize,
        _plan: &Self::QueryPlan,
        column: &mut Vec<Cost<Self>>,
    ) -> (Cost<Self>, Self::Carry) {
        let cost = match depth.checked_sub(1).and_then(|index| query.get(index)) {
            Some(value) => {
                let prefix = if depth == 1 {
                    WeightedCost::ZERO
                } else {
                    previous.first().copied().unwrap_or(WeightedCost::TOP)
                };
                WeightedCost::combine(prefix, interval_dist(*value, interval.0, interval.1))
            }
            None => WeightedCost::TOP,
        };
        column.clear();
        column.push(cost);
        (cost, ())
    }

    fn exact_with_cutoff(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: Cost<Self>,
    ) -> Option<Cost<Self>> {
        let exact = if query.len() == candidate.len()
            && query.iter().chain(candidate).all(|value| value.is_finite())
        {
            query
                .iter()
                .zip(candidate)
                .map(|(left, right)| (left - right).abs())
                .fold(WeightedCost::ZERO, WeightedCost::combine)
        } else {
            WeightedCost::TOP
        };
        WeightedCost::within(exact, cutoff).then_some(exact)
    }

    fn candidate_lower_bound(
        &self,
        query: &[f64],
        candidate: &[f64],
        _plan: &Self::QueryPlan,
    ) -> Cost<Self> {
        if query.len() == candidate.len() {
            WeightedCost::ZERO
        } else {
            WeightedCost::TOP
        }
    }

    fn try_plan(&self, _query: &[f64]) -> Result<Self::QueryPlan, IncompleteReason> {
        Ok(())
    }

    fn empty_pair_cost(&self) -> Cost<Self> {
        WeightedCost::ZERO
    }

    fn empty_vs_nonempty_cost(&self, _nonempty: &[f64]) -> Cost<Self> {
        WeightedCost::TOP
    }
}

impl MetricElasticKernel for PointwiseL1 {}

fn assert_metric_kernel<K: MetricElasticKernel>() {}

fn assert_audited_metric_index<I: AuditedMetricTimeSeriesIndex>() {}

#[test]
fn metric_status_is_queryable_and_triangle_dependent_code_is_type_gated() {
    const {
        assert!(!MsmKernel::IS_METRIC);
        assert!(MetricMsmKernel::IS_METRIC);
        assert!(!ErpConfig::IS_METRIC);
        assert!(!FrechetConfig::IS_METRIC);
        assert!(MetricTwedConfig::IS_METRIC);
        assert!(!DtwConfig::IS_METRIC);
        assert!(!TwedConfig::IS_METRIC);
    }
    let _ = MetricMsmConfig::try_new(1.0).unwrap();
    assert_metric_kernel::<MetricMsmKernel>();
    assert_metric_kernel::<MetricTwedConfig>();
    assert_metric_kernel::<PointwiseL1>();
    assert_audited_metric_index::<ElasticTransducer<MetricMsmKernel>>();
    assert_audited_metric_index::<ElasticTransducer<MetricTwedConfig>>();
    assert_audited_metric_index::<MetricErpTransducer>();
    assert_audited_metric_index::<MetricFrechetTransducer>();
}

fn brute_force(series: &[Vec<f64>], query: &[f64], cutoff: f64) -> Vec<(usize, f64)> {
    let kernel = PointwiseL1;
    let mut results: Vec<_> = series
        .iter()
        .enumerate()
        .filter_map(|(id, candidate)| {
            kernel
                .exact_with_cutoff(query, candidate, cutoff)
                .map(|distance| (id, distance))
        })
        .collect();
    results.sort_by(|left, right| left.1.total_cmp(&right.1));
    results
}

#[test]
fn custom_kernel_range_and_knn_match_brute_force() {
    let series = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.0, 2.5, 3.0],
        vec![9.0, 9.0, 9.0],
        vec![1.0, 2.0],
    ];
    let query = [1.0, 2.0, 3.5];
    let index = ElasticTransducer::<PointwiseL1>::from_series(
        QuantizationConfig::for_u8(0.0, 10.0),
        PointwiseL1,
        &series,
    );

    assert_eq!(
        index.search_range(&query, 1.0),
        brute_force(&series, &query, 1.0)
    );

    let mut expected = brute_force(&series, &query, f64::INFINITY);
    expected.truncate(2);
    assert_eq!(index.search_knn(&query, 2, f64::INFINITY), expected);

    let (observed, stats) = index.search_knn_with_stats(&query, 2, f64::INFINITY);
    assert_eq!(observed, expected);
    assert!(stats.accounting_is_consistent());
    assert!(stats.visited_nodes > 0);
    assert_eq!(
        stats.prefix_pruned + stats.columns_built,
        stats.visited_edges
    );
    assert_eq!(
        stats.candidate_bound_pruned + stats.exact_evaluations,
        stats.candidates_considered
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    #[test]
    fn custom_kernel_range_is_exact_and_monotone(
        series in prop::collection::vec(
            prop::collection::vec(0.0f64..20.0, 0..7),
            0..18,
        ),
        query in prop::collection::vec(0.0f64..20.0, 0..7),
        cutoff in 0.0f64..30.0,
        extra in 0.0f64..10.0,
    ) {
        let index = ElasticTransducer::<PointwiseL1>::from_series(
            QuantizationConfig::for_u8(0.0, 20.0),
            PointwiseL1,
            &series,
        );

        let at_cutoff = index.search_range(&query, cutoff);
        let expected = brute_force(&series, &query, cutoff);
        prop_assert_eq!(&at_cutoff, &expected);

        let wider = index.search_range(&query, cutoff + extra);
        let wider_ids: std::collections::HashSet<_> =
            wider.iter().map(|(id, _)| *id).collect();
        prop_assert!(at_cutoff.iter().all(|(id, _)| wider_ids.contains(id)));
        prop_assert!(at_cutoff.windows(2).all(|pair| pair[0].1 <= pair[1].1));
    }

    #[test]
    fn observed_knn_is_result_transparent_and_accounted(
        series in prop::collection::vec(
            prop::collection::vec(0.0f64..20.0, 1..7),
            0..18,
        ),
        query in prop::collection::vec(0.0f64..20.0, 1..7),
        k in 0usize..20,
    ) {
        let index = ElasticTransducer::<PointwiseL1>::from_series(
            QuantizationConfig::for_u8(0.0, 20.0),
            PointwiseL1,
            &series,
        );
        let expected = index.search_knn(&query, k, f64::INFINITY);
        let (observed, stats) = index.search_knn_with_stats(&query, k, f64::INFINITY);

        prop_assert_eq!(observed, expected);
        prop_assert!(stats.accounting_is_consistent());
        prop_assert_eq!(
            stats.prefix_pruned.saturating_add(stats.columns_built),
            stats.visited_edges
        );
        prop_assert_eq!(
            stats.candidate_bound_pruned.saturating_add(stats.exact_evaluations),
            stats.candidates_considered
        );
    }
}
