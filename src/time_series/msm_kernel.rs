//! Move-Split-Merge implementation of [`super::elastic::ElasticKernel`].

use super::bounded::IncompleteReason;
use super::elastic::sparse::{charge_work, NeighborSeedRows};
use super::elastic::{
    Cost, ElasticKernel, MetricElasticKernel, PointFrontierStep, QueryPlanStorage,
};
use super::lower_bounds::length_lb;
use super::msm::{series_values_are_finite, MetricMsmConfig, MsmConfig};
use super::msm_interval::{
    c_func_merge_lb, c_func_split_lb, interval_column_len, step_interval_column_into_with_bound,
};
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
    const IS_METRIC: bool = false;

    type Monoid = WeightedCost;
    type Carry = (f64, f64);
    type QueryPlan = ();

    #[inline]
    fn query_plan_storage(&self, _query_len: usize) -> Result<QueryPlanStorage, IncompleteReason> {
        Ok(QueryPlanStorage::EMPTY)
    }

    #[inline]
    fn canonical_carry_key(&self, carry: Self::Carry) -> Option<[u64; 2]> {
        crate::time_series::elastic::canonical_f64_pair_state_key(carry.0, carry.1)
    }

    #[inline]
    fn normalized(self) -> Self {
        Self::new(self.config)
    }

    #[inline]
    fn supports_interval_query(&self, query: &[f64]) -> bool {
        series_values_are_finite(query)
    }

    #[inline]
    fn alignment_is_structurally_possible(&self, query_len: usize, candidate_len: usize) -> bool {
        query_len == candidate_len || (query_len > 0 && candidate_len > 0)
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

    fn step_point_frontier(
        &self,
        previous: &[Cost<Self>],
        previous_active: &[usize],
        query: &[f64],
        target: f64,
        previous_carry: Option<Self::Carry>,
        depth: usize,
        _plan: &Self::QueryPlan,
        cutoff: Cost<Self>,
        max_work: usize,
        column: &mut [Cost<Self>],
        active: &mut Vec<usize>,
    ) -> Option<PointFrontierStep<Cost<Self>, Self::Carry>> {
        let expected = query.len().checked_add(1)?;
        if column.len() != expected || query.is_empty() || depth == 0 {
            return Some(PointFrontierStep::Advanced {
                lower_bound: WeightedCost::TOP,
                carry: (target, target),
                work: 0,
            });
        }
        let split_merge = self.config.split_merge_cost();
        let last_row = query.len();
        let mut lower_bound = WeightedCost::TOP;
        let mut work = 0usize;

        if depth == 1 {
            if let Err(requested) = charge_work(&mut work, max_work) {
                return Some(PointFrontierStep::WorkLimitExceeded {
                    completed: work,
                    requested,
                });
            }
            let first = (query[0] - target).abs();
            if WeightedCost::within(first, cutoff) {
                column[1] = first;
                active.push(1);
                lower_bound = first;
            }
            for row in 2..=last_row {
                if let Err(requested) = charge_work(&mut work, max_work) {
                    return Some(PointFrontierStep::WorkLimitExceeded {
                        completed: work,
                        requested,
                    });
                }
                let cost = WeightedCost::combine(
                    column[row - 1],
                    c_func_merge_lb(query[row - 1], query[row - 2], target, target, split_merge),
                );
                if !WeightedCost::within(cost, cutoff) {
                    break;
                }
                column[row] = cost;
                active.push(row);
                lower_bound = lower_bound.min(cost);
            }
            return Some(PointFrontierStep::Advanced {
                lower_bound,
                carry: (target, target),
                work,
            });
        }

        let previous_target = previous_carry.map_or(target, |interval| interval.0);
        let mut seeds = NeighborSeedRows::new(previous_active, last_row);
        let mut next_seed = seeds.next();
        while let Some(start) = next_seed {
            let mut row = start.max(1);
            loop {
                while next_seed.is_some_and(|seed| seed <= row) {
                    next_seed = seeds.next();
                }
                if let Err(requested) = charge_work(&mut work, max_work) {
                    return Some(PointFrontierStep::WorkLimitExceeded {
                        completed: work,
                        requested,
                    });
                }
                let cost = if row == 1 {
                    WeightedCost::combine(
                        previous[1],
                        c_func_split_lb(
                            target,
                            target,
                            query[0],
                            previous_target,
                            previous_target,
                            split_merge,
                        ),
                    )
                } else {
                    let moved =
                        WeightedCost::combine(previous[row - 1], (query[row - 1] - target).abs());
                    let merged = WeightedCost::combine(
                        column[row - 1],
                        c_func_merge_lb(
                            query[row - 1],
                            query[row - 2],
                            target,
                            target,
                            split_merge,
                        ),
                    );
                    let split = WeightedCost::combine(
                        previous[row],
                        c_func_split_lb(
                            target,
                            target,
                            query[row - 1],
                            previous_target,
                            previous_target,
                            split_merge,
                        ),
                    );
                    moved.min(merged).min(split)
                };
                if WeightedCost::within(cost, cutoff) {
                    column[row] = cost;
                    active.push(row);
                    lower_bound = lower_bound.min(cost);
                    if row < last_row {
                        row += 1;
                        continue;
                    }
                }
                break;
            }
        }
        Some(PointFrontierStep::Advanced {
            lower_bound,
            carry: (target, target),
            work,
        })
    }

    fn step_interval_frontier(
        &self,
        previous: &[Cost<Self>],
        previous_active: &[usize],
        query: &[f64],
        target: (f64, f64),
        previous_carry: Option<Self::Carry>,
        depth: usize,
        _plan: &Self::QueryPlan,
        cutoff: Cost<Self>,
        max_work: usize,
        column: &mut [Cost<Self>],
        active: &mut Vec<usize>,
    ) -> Option<PointFrontierStep<Cost<Self>, Self::Carry>> {
        let expected = query.len().checked_add(1)?;
        if column.len() != expected || query.is_empty() || depth == 0 {
            return Some(PointFrontierStep::Advanced {
                lower_bound: WeightedCost::TOP,
                carry: target,
                work: 0,
            });
        }
        let split_merge = self.config.split_merge_cost();
        let last_row = query.len();
        let mut lower_bound = WeightedCost::TOP;
        let mut work = 0usize;

        if depth == 1 {
            if let Err(requested) = charge_work(&mut work, max_work) {
                return Some(PointFrontierStep::WorkLimitExceeded {
                    completed: work,
                    requested,
                });
            }
            let first = super::msm_interval::interval_dist(query[0], target.0, target.1);
            if WeightedCost::within(first, cutoff) {
                column[1] = first;
                active.push(1);
                lower_bound = first;
            }
            for row in 2..=last_row {
                if let Err(requested) = charge_work(&mut work, max_work) {
                    return Some(PointFrontierStep::WorkLimitExceeded {
                        completed: work,
                        requested,
                    });
                }
                let cost = WeightedCost::combine(
                    column[row - 1],
                    c_func_merge_lb(
                        query[row - 1],
                        query[row - 2],
                        target.0,
                        target.1,
                        split_merge,
                    ),
                );
                if !WeightedCost::within(cost, cutoff) {
                    break;
                }
                column[row] = cost;
                active.push(row);
                lower_bound = lower_bound.min(cost);
            }
            return Some(PointFrontierStep::Advanced {
                lower_bound,
                carry: target,
                work,
            });
        }

        let previous_target = previous_carry.unwrap_or(target);
        let mut seeds = NeighborSeedRows::new(previous_active, last_row);
        let mut next_seed = seeds.next();
        while let Some(start) = next_seed {
            let mut row = start.max(1);
            loop {
                while next_seed.is_some_and(|seed| seed <= row) {
                    next_seed = seeds.next();
                }
                if let Err(requested) = charge_work(&mut work, max_work) {
                    return Some(PointFrontierStep::WorkLimitExceeded {
                        completed: work,
                        requested,
                    });
                }
                let cost = if row == 1 {
                    WeightedCost::combine(
                        previous[1],
                        c_func_split_lb(
                            target.0,
                            target.1,
                            query[0],
                            previous_target.0,
                            previous_target.1,
                            split_merge,
                        ),
                    )
                } else {
                    let moved = WeightedCost::combine(
                        previous[row - 1],
                        super::msm_interval::interval_dist(query[row - 1], target.0, target.1),
                    );
                    let merged = WeightedCost::combine(
                        column[row - 1],
                        c_func_merge_lb(
                            query[row - 1],
                            query[row - 2],
                            target.0,
                            target.1,
                            split_merge,
                        ),
                    );
                    let split = WeightedCost::combine(
                        previous[row],
                        c_func_split_lb(
                            target.0,
                            target.1,
                            query[row - 1],
                            previous_target.0,
                            previous_target.1,
                            split_merge,
                        ),
                    );
                    moved.min(merged).min(split)
                };
                if WeightedCost::within(cost, cutoff) {
                    column[row] = cost;
                    active.push(row);
                    lower_bound = lower_bound.min(cost);
                    if row < last_row {
                        row += 1;
                        continue;
                    }
                }
                break;
            }
        }
        Some(PointFrontierStep::Advanced {
            lower_bound,
            carry: target,
            work,
        })
    }

    #[inline]
    fn vertical_epsilon_extension(
        &self,
        query: &[f64],
        target: (f64, f64),
        row: usize,
        column: &[Cost<Self>],
        _plan: &Self::QueryPlan,
    ) -> Option<Cost<Self>> {
        if row < 2 {
            return None;
        }
        let predecessor = *column.get(row - 1)?;
        Some(WeightedCost::combine(
            predecessor,
            c_func_merge_lb(
                *query.get(row - 1)?,
                *query.get(row - 2)?,
                target.0,
                target.1,
                self.config.split_merge_cost(),
            ),
        ))
    }

    #[inline]
    fn carry_interval(&self, carry: Self::Carry) -> Option<(f64, f64)> {
        Some(carry)
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
    fn try_plan(&self, _query: &[f64]) -> Result<Self::QueryPlan, IncompleteReason> {
        Ok(())
    }

    #[inline]
    fn empty_pair_cost(&self) -> Cost<Self> {
        WeightedCost::ZERO
    }

    #[inline]
    fn empty_vs_nonempty_cost(&self, _nonempty: &[f64]) -> Cost<Self> {
        WeightedCost::TOP
    }
}

/// Metric-domain MSM kernel. Unlike [`MsmKernel`], construction proves `c > 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricMsmKernel {
    inner: MsmKernel,
}

impl MetricMsmKernel {
    /// Construct the kernel from a validated metric-domain configuration.
    #[inline]
    pub fn new(config: MetricMsmConfig) -> Self {
        Self {
            inner: MsmKernel::new(*config.as_config()),
        }
    }

    /// Borrow the underlying validated raw configuration.
    #[inline]
    pub fn config(&self) -> &MsmConfig {
        self.inner.config()
    }
}

impl From<MetricMsmConfig> for MetricMsmKernel {
    #[inline]
    fn from(config: MetricMsmConfig) -> Self {
        Self::new(config)
    }
}

impl ElasticKernel for MetricMsmKernel {
    const IS_METRIC: bool = true;

    type Monoid = <MsmKernel as ElasticKernel>::Monoid;
    type Carry = <MsmKernel as ElasticKernel>::Carry;
    type QueryPlan = <MsmKernel as ElasticKernel>::QueryPlan;

    #[inline]
    fn query_plan_storage(&self, query_len: usize) -> Result<QueryPlanStorage, IncompleteReason> {
        self.inner.query_plan_storage(query_len)
    }

    #[inline]
    fn canonical_carry_key(&self, carry: Self::Carry) -> Option<[u64; 2]> {
        self.inner.canonical_carry_key(carry)
    }

    #[inline]
    fn supports_interval_query(&self, query: &[f64]) -> bool {
        !query.is_empty() && self.inner.supports_interval_query(query)
    }

    #[inline]
    fn alignment_is_structurally_possible(&self, query_len: usize, candidate_len: usize) -> bool {
        query_len > 0 && candidate_len > 0
    }

    #[inline]
    fn column_len(&self, query_len: usize) -> Option<usize> {
        self.inner.column_len(query_len)
    }

    #[inline]
    fn final_row(&self, query_len: usize) -> usize {
        self.inner.final_row(query_len)
    }

    fn step_column(
        &self,
        previous: &[Cost<Self>],
        query: &[f64],
        current_interval: (f64, f64),
        previous_carry: Option<Self::Carry>,
        depth: usize,
        plan: &Self::QueryPlan,
        column: &mut Vec<Cost<Self>>,
    ) -> (Cost<Self>, Self::Carry) {
        self.inner.step_column(
            previous,
            query,
            current_interval,
            previous_carry,
            depth,
            plan,
            column,
        )
    }

    #[inline]
    fn step_point_frontier(
        &self,
        previous: &[Cost<Self>],
        previous_active: &[usize],
        query: &[f64],
        target: f64,
        previous_carry: Option<Self::Carry>,
        depth: usize,
        plan: &Self::QueryPlan,
        cutoff: Cost<Self>,
        max_work: usize,
        column: &mut [Cost<Self>],
        active: &mut Vec<usize>,
    ) -> Option<PointFrontierStep<Cost<Self>, Self::Carry>> {
        self.inner.step_point_frontier(
            previous,
            previous_active,
            query,
            target,
            previous_carry,
            depth,
            plan,
            cutoff,
            max_work,
            column,
            active,
        )
    }

    #[inline]
    fn step_interval_frontier(
        &self,
        previous: &[Cost<Self>],
        previous_active: &[usize],
        query: &[f64],
        target: (f64, f64),
        previous_carry: Option<Self::Carry>,
        depth: usize,
        plan: &Self::QueryPlan,
        cutoff: Cost<Self>,
        max_work: usize,
        column: &mut [Cost<Self>],
        active: &mut Vec<usize>,
    ) -> Option<PointFrontierStep<Cost<Self>, Self::Carry>> {
        self.inner.step_interval_frontier(
            previous,
            previous_active,
            query,
            target,
            previous_carry,
            depth,
            plan,
            cutoff,
            max_work,
            column,
            active,
        )
    }

    #[inline]
    fn vertical_epsilon_extension(
        &self,
        query: &[f64],
        target: (f64, f64),
        row: usize,
        column: &[Cost<Self>],
        plan: &Self::QueryPlan,
    ) -> Option<Cost<Self>> {
        self.inner
            .vertical_epsilon_extension(query, target, row, column, plan)
    }

    #[inline]
    fn carry_interval(&self, carry: Self::Carry) -> Option<(f64, f64)> {
        self.inner.carry_interval(carry)
    }

    #[inline]
    fn prefix_lower_bound(
        &self,
        query: &[f64],
        current_interval: (f64, f64),
        previous_carry: Option<Self::Carry>,
        depth: usize,
        plan: &Self::QueryPlan,
    ) -> Cost<Self> {
        self.inner
            .prefix_lower_bound(query, current_interval, previous_carry, depth, plan)
    }

    #[inline]
    fn exact_with_cutoff(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: Cost<Self>,
    ) -> Option<Cost<Self>> {
        if query.is_empty() || candidate.is_empty() {
            None
        } else {
            self.inner.exact_with_cutoff(query, candidate, cutoff)
        }
    }

    #[inline]
    fn candidate_lower_bound(
        &self,
        query: &[f64],
        candidate: &[f64],
        plan: &Self::QueryPlan,
    ) -> Cost<Self> {
        self.inner.candidate_lower_bound(query, candidate, plan)
    }

    #[inline]
    fn try_plan(&self, query: &[f64]) -> Result<Self::QueryPlan, IncompleteReason> {
        self.inner.try_plan(query)
    }

    #[inline]
    fn empty_pair_cost(&self) -> Cost<Self> {
        WeightedCost::TOP
    }

    #[inline]
    fn empty_vs_nonempty_cost(&self, _nonempty: &[f64]) -> Cost<Self> {
        WeightedCost::TOP
    }
}

impl MetricElasticKernel for MetricMsmKernel {}
