//! Move-Split-Merge implementation of [`super::elastic::ElasticKernel`].

use super::elastic::{Cost, ElasticKernel, MetricElasticKernel};
use super::lower_bounds::length_lb;
use super::msm::{series_values_are_finite, MsmConfig};
use super::msm_interval::{interval_column_len, step_interval_column_into_with_bound};
use crate::cost::{CostMonoid, WeightedCost};

/// Elastic-kernel adapter for the existing exact MSM implementation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MsmKernel {
    config: MsmConfig,
}

impl MsmKernel {
    /// Construct a normalized MSM kernel.
    #[inline]
    pub fn new(config: MsmConfig) -> Self {
        Self {
            config: config.normalized(),
        }
    }

    /// Effective MSM configuration.
    #[inline]
    pub fn config(&self) -> &MsmConfig {
        &self.config
    }
}

impl From<MsmConfig> for MsmKernel {
    #[inline]
    fn from(config: MsmConfig) -> Self {
        Self::new(config)
    }
}

impl ElasticKernel for MsmKernel {
    const IS_METRIC: bool = true;

    type Monoid = WeightedCost;
    type Carry = (f64, f64);
    type QueryPlan = ();

    #[inline]
    fn normalized(self) -> Self {
        Self::new(self.config)
    }

    #[inline]
    fn supports_interval_query(&self, query: &[f64]) -> bool {
        series_values_are_finite(query)
    }

    #[inline]
    fn column_len(&self, query_len: usize) -> Option<usize> {
        interval_column_len(query_len)
    }

    #[inline]
    fn final_row(&self, query_len: usize) -> usize {
        query_len
    }

    fn step_column(
        &self,
        previous: &[Cost<Self>],
        query: &[f64],
        current_interval: (f64, f64),
        previous_carry: Option<Self::Carry>,
        _depth: usize,
        _plan: &Self::QueryPlan,
        column: &mut Vec<Cost<Self>>,
    ) -> (Cost<Self>, Self::Carry) {
        let lower_bound = step_interval_column_into_with_bound(
            previous,
            query,
            current_interval,
            previous_carry,
            self.config.split_merge_cost(),
            column,
        );
        (lower_bound, current_interval)
    }

    #[inline]
    fn exact_with_cutoff(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: Cost<Self>,
    ) -> Option<Cost<Self>> {
        self.config.distance_with_cutoff(query, candidate, cutoff)
    }

    #[inline]
    fn candidate_lower_bound(
        &self,
        query: &[f64],
        candidate: &[f64],
        _plan: &Self::QueryPlan,
    ) -> Cost<Self> {
        if series_values_are_finite(query) && series_values_are_finite(candidate) {
            length_lb(query, candidate, self.config.split_merge_cost())
        } else {
            WeightedCost::TOP
        }
    }

    #[inline]
    fn plan(&self, _query: &[f64]) -> Self::QueryPlan {}

    #[inline]
    fn empty_pair_cost(&self) -> Cost<Self> {
        WeightedCost::ZERO
    }

    #[inline]
    fn empty_vs_nonempty_cost(&self, _nonempty: &[f64]) -> Cost<Self> {
        WeightedCost::TOP
    }
}

impl MetricElasticKernel for MsmKernel {}
