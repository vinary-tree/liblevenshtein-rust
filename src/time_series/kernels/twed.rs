//! Time Warp Edit Distance (TWED) for unit-spaced scalar time series.
//!
//! TWED edits *segments*, rather than isolated samples. With the conventional
//! sentinels `x[0] = y[0] = 0` and timestamps `t_i = i`, deleting a sample
//! costs its change from the preceding sample plus stiffness `nu` and gap
//! penalty `lambda`. Matching two samples also compares both preceding
//! samples and penalizes temporal displacement.
//!
//! [`TwedConfig`] represents the complete non-negative parameter family,
//! including the useful but non-metric `nu = lambda = 0` degeneracy.
//! [`MetricTwedConfig`] validates the strict `nu > 0` premise used by
//! Marteau's metric proof and is the only TWED type implementing
//! [`crate::time_series::elastic::MetricElasticKernel`].
//!
//! The recurrence and metric proof are due to Marteau, *Time Warp Edit
//! Distance with Stiffness Adjustment for Time Series Matching*, IEEE TPAMI
//! 31(2), 2009, DOI
//! [10.1109/TPAMI.2008.76](https://doi.org/10.1109/TPAMI.2008.76).

use super::super::elastic::interval::{interval_dist, interval_gap};
use super::super::elastic::sparse::{charge_work, NeighborSeedRows};
use super::super::elastic::{
    Cost, ElasticKernel, ElasticTransducer, MetricElasticKernel, PointFrontierStep,
    QueryPlanStorage,
};
use crate::cost::{CostMonoid, WeightedCost};
use crate::time_series::bounded::IncompleteReason;

const DEFAULT_NU: f64 = 0.001;
const DEFAULT_LAMBDA: f64 = 1.0;
const ORIGIN_INTERVAL: (f64, f64) = (0.0, 0.0);

#[inline]
fn normalize_nonnegative(value: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}

#[inline]
fn series_is_finite(series: &[f64]) -> bool {
    series.iter().all(|value| value.is_finite())
}

#[inline]
fn add3(a: f64, b: f64, c: f64) -> f64 {
    WeightedCost::combine(WeightedCost::combine(a, b), c)
}

#[inline]
fn segment_cost(current: f64, previous: f64, nu: f64, lambda: f64) -> f64 {
    add3((current - previous).abs(), nu, lambda)
}

#[inline]
fn match_cost(
    x_current: f64,
    x_previous: f64,
    y_current: f64,
    y_previous: f64,
    x_index: usize,
    y_index: usize,
    nu: f64,
) -> f64 {
    let displacement = x_index.abs_diff(y_index) as f64;
    add3(
        (x_current - y_current).abs(),
        (x_previous - y_previous).abs(),
        2.0 * nu * displacement,
    )
}

#[inline]
fn boundary_cost(series: &[f64], nu: f64, lambda: f64) -> f64 {
    let mut previous = 0.0;
    series.iter().fold(WeightedCost::ZERO, |cost, current| {
        let next = WeightedCost::combine(cost, segment_cost(*current, previous, nu, lambda));
        previous = *current;
        next
    })
}

/// Configuration for the complete non-negative TWED parameter family.
///
/// Public fields make serialized configuration explicit. Every computation
/// normalizes negative or non-finite fields to zero, so directly constructed
/// invalid values cannot introduce NaN into the walker. Because this type also
/// represents `nu = 0`, it is not a compile-time metric witness; use
/// [`MetricTwedConfig::try_new`] when triangle-dependent indexing is required.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwedConfig {
    /// Stiffness applied to each unit timestamp difference.
    pub nu: f64,
    /// Constant penalty applied to target or query deletion.
    pub lambda: f64,
}

impl Default for TwedConfig {
    fn default() -> Self {
        Self::new(DEFAULT_NU, DEFAULT_LAMBDA)
    }
}

impl TwedConfig {
    /// Construct TWED with non-negative stiffness and gap penalty.
    ///
    /// Negative and non-finite inputs normalize to zero. This intentionally
    /// permits the documented zero-parameter degeneracy.
    #[inline]
    pub fn new(nu: f64, lambda: f64) -> Self {
        Self {
            nu: normalize_nonnegative(nu),
            lambda: normalize_nonnegative(lambda),
        }
    }

    /// Effective finite, non-negative stiffness.
    #[inline]
    pub fn stiffness(&self) -> f64 {
        normalize_nonnegative(self.nu)
    }

    /// Effective finite, non-negative gap penalty.
    #[inline]
    pub fn gap_penalty(&self) -> f64 {
        normalize_nonnegative(self.lambda)
    }

    /// Whether this instance satisfies the premise of Marteau's metric proof.
    #[inline]
    pub fn is_metric_configuration(&self) -> bool {
        self.stiffness() > 0.0
    }

    /// Return a configuration whose public fields are normalized.
    #[inline]
    pub fn normalized(self) -> Self {
        Self::new(self.nu, self.lambda)
    }

    /// Convert to a compile-time metric witness when `nu > 0`.
    #[inline]
    pub fn metric(self) -> Result<MetricTwedConfig, MetricTwedConfigError> {
        MetricTwedConfig::try_new(self.nu, self.lambda)
    }

    /// Compute exact TWED in `O(mn)` time and `O(min(m,n))` space.
    pub fn distance(&self, x: &[f64], y: &[f64]) -> f64 {
        self.distance_with_cutoff(x, y, f64::INFINITY)
            .unwrap_or(f64::INFINITY)
    }

    /// Compute exact TWED when it is at most `max_cost`.
    ///
    /// Every edit cost is non-negative. Consequently a completed row whose
    /// minimum exceeds the cutoff cannot lead to a surviving final path.
    pub fn distance_with_cutoff(&self, x: &[f64], y: &[f64], max_cost: f64) -> Option<f64> {
        if max_cost.is_nan() || max_cost < 0.0 || !series_is_finite(x) || !series_is_finite(y) {
            return None;
        }

        // TWED is symmetric; allocate the shorter series as the stored row.
        if y.len() > x.len() {
            return self.distance_with_cutoff(y, x, max_cost);
        }

        let nu = self.stiffness();
        let lambda = self.gap_penalty();
        let row_len = y.len().checked_add(1)?;
        let mut previous = vec![WeightedCost::TOP; row_len];
        let mut current = vec![WeightedCost::TOP; row_len];
        previous[0] = WeightedCost::ZERO;

        let mut y_previous = 0.0;
        for (column_index, y_current) in y.iter().enumerate() {
            let column = column_index + 1;
            previous[column] = WeightedCost::combine(
                previous[column - 1],
                segment_cost(*y_current, y_previous, nu, lambda),
            );
            y_previous = *y_current;
        }

        let mut x_previous = 0.0;
        for (row_index, x_current) in x.iter().enumerate() {
            let row = row_index + 1;
            let delete_x_cost = segment_cost(*x_current, x_previous, nu, lambda);
            current[0] = WeightedCost::combine(previous[0], delete_x_cost);
            let mut row_min = current[0];
            let mut y_previous = 0.0;

            for (column_index, y_current) in y.iter().enumerate() {
                let column = column_index + 1;
                let match_pair = WeightedCost::combine(
                    previous[column - 1],
                    match_cost(
                        *x_current, x_previous, *y_current, y_previous, row, column, nu,
                    ),
                );
                let delete_x = WeightedCost::combine(previous[column], delete_x_cost);
                let delete_y = WeightedCost::combine(
                    current[column - 1],
                    segment_cost(*y_current, y_previous, nu, lambda),
                );
                current[column] = match_pair.min(delete_x).min(delete_y);
                row_min = row_min.min(current[column]);
                y_previous = *y_current;
            }

            if !WeightedCost::within(row_min, max_cost) {
                return None;
            }
            std::mem::swap(&mut previous, &mut current);
            x_previous = *x_current;
        }

        let exact = previous[y.len()];
        WeightedCost::within(exact, max_cost).then_some(exact)
    }

    /// Length-imbalance lower bound for one candidate pair.
    #[inline]
    pub fn candidate_lower_bound(&self, x: &[f64], y: &[f64]) -> f64 {
        twed_length_lower_bound(x.len(), y.len(), self.gap_penalty())
    }
}

/// Validation error for [`MetricTwedConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MetricTwedConfigError {
    /// Stiffness must be finite and strictly positive.
    #[error("TWED metric stiffness nu must be finite and strictly positive")]
    NonPositiveStiffness,
    /// Gap penalty must be finite and non-negative.
    #[error("TWED metric gap penalty lambda must be finite and non-negative")]
    InvalidGapPenalty,
}

/// Validated metric TWED configuration.
///
/// This wrapper makes the primary-source condition `nu > 0, lambda >= 0`
/// unrepresentable by construction and therefore safely implements
/// [`MetricElasticKernel`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricTwedConfig(TwedConfig);

impl Default for MetricTwedConfig {
    fn default() -> Self {
        // These constants satisfy the invariant by construction.
        Self(TwedConfig::new(DEFAULT_NU, DEFAULT_LAMBDA))
    }
}

impl MetricTwedConfig {
    /// Validate and construct metric TWED.
    pub fn try_new(nu: f64, lambda: f64) -> Result<Self, MetricTwedConfigError> {
        if !nu.is_finite() || nu <= 0.0 {
            return Err(MetricTwedConfigError::NonPositiveStiffness);
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(MetricTwedConfigError::InvalidGapPenalty);
        }
        Ok(Self(TwedConfig { nu, lambda }))
    }

    /// Validated strictly positive stiffness.
    #[inline]
    pub fn stiffness(&self) -> f64 {
        self.0.nu
    }

    /// Validated non-negative gap penalty.
    #[inline]
    pub fn gap_penalty(&self) -> f64 {
        self.0.lambda
    }

    /// Borrow the underlying full-family configuration.
    #[inline]
    pub fn as_config(&self) -> &TwedConfig {
        &self.0
    }

    /// Compute exact metric TWED.
    #[inline]
    pub fn distance(&self, x: &[f64], y: &[f64]) -> f64 {
        self.0.distance(x, y)
    }

    /// Compute exact metric TWED when it is at most `max_cost`.
    #[inline]
    pub fn distance_with_cutoff(&self, x: &[f64], y: &[f64], max_cost: f64) -> Option<f64> {
        self.0.distance_with_cutoff(x, y, max_cost)
    }
}

/// TWED kernel name for APIs that distinguish configuration from policy.
pub type TwedKernel = TwedConfig;

/// Explicit name for index-displacement TWED on a unit sampling grid.
pub type UnitGridTwedConfig = TwedConfig;

/// Explicit metric-domain name for index-displacement TWED on a unit grid.
pub type MetricUnitGridTwedConfig = MetricTwedConfig;

/// Explicit kernel name for full-family unit-grid TWED.
pub type UnitGridTwedKernel = TwedConfig;

/// Explicit kernel name for metric unit-grid TWED.
pub type MetricUnitGridTwedKernel = MetricTwedConfig;

/// Exact full-family TWED index backed by the generic elastic trie walker.
pub type TwedTransducer<V = usize> = ElasticTransducer<TwedKernel, V>;

/// Exact full-family unit-grid TWED index.
pub type UnitGridTwedTransducer<V = usize> = TwedTransducer<V>;

/// Validated metric TWED kernel name.
pub type MetricTwedKernel = MetricTwedConfig;

/// Exact metric-only TWED index.
pub type MetricTwedTransducer<V = usize> = ElasticTransducer<MetricTwedKernel, V>;

/// Exact metric unit-grid TWED index.
pub type MetricUnitGridTwedTransducer<V = usize> = MetricTwedTransducer<V>;

/// Lower bound from the number of unavoidable length-changing edits.
///
/// Any path between lengths `m` and `n` contains at least `|m-n|` deletions,
/// each of which pays `lambda` in addition to non-negative segment and
/// timestamp costs.
#[inline]
pub fn twed_length_lower_bound(m: usize, n: usize, lambda: f64) -> f64 {
    let lambda = normalize_nonnegative(lambda);
    WeightedCost::combine(0.0, m.abs_diff(n) as f64 * lambda)
}

trait TwedPolicy: Clone + std::fmt::Debug + Send + Sync + 'static {
    const METRIC: bool;

    fn config(&self) -> TwedConfig;

    fn normalize_policy(self) -> Self;
}

impl TwedPolicy for TwedConfig {
    const METRIC: bool = false;

    #[inline]
    fn config(&self) -> TwedConfig {
        self.normalized()
    }

    #[inline]
    fn normalize_policy(self) -> Self {
        self.normalized()
    }
}

impl TwedPolicy for MetricTwedConfig {
    const METRIC: bool = true;

    #[inline]
    fn config(&self) -> TwedConfig {
        self.0
    }

    #[inline]
    fn normalize_policy(self) -> Self {
        self
    }
}

impl<T: TwedPolicy> ElasticKernel for T {
    const IS_METRIC: bool = T::METRIC;

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
        self.normalize_policy()
    }

    #[inline]
    fn supports_interval_query(&self, query: &[f64]) -> bool {
        series_is_finite(query)
    }

    #[inline]
    fn column_len(&self, query_len: usize) -> Option<usize> {
        query_len.checked_add(1)
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
        depth: usize,
        _plan: &Self::QueryPlan,
        column: &mut Vec<Cost<Self>>,
    ) -> (Cost<Self>, Self::Carry) {
        let Some(column_len) = query.len().checked_add(1) else {
            column.clear();
            return (WeightedCost::TOP, current_interval);
        };
        column.resize(column_len, WeightedCost::TOP);

        let config = self.config();
        let nu = config.stiffness();
        let lambda = config.gap_penalty();
        let previous_interval = previous_carry.unwrap_or(ORIGIN_INTERVAL);
        let target_delete_cost = add3(
            interval_gap(current_interval, previous_interval),
            nu,
            lambda,
        );

        let root_boundary = |row: usize| boundary_cost(&query[..row], nu, lambda);
        let previous_cell = |row: usize| {
            if depth == 1 {
                root_boundary(row)
            } else {
                previous.get(row).copied().unwrap_or(WeightedCost::TOP)
            }
        };

        column[0] = WeightedCost::combine(previous_cell(0), target_delete_cost);
        let mut lower_bound = column[0];
        for (row_index, query_current) in query.iter().enumerate() {
            let row = row_index + 1;
            let query_previous = row_index
                .checked_sub(1)
                .and_then(|index| query.get(index))
                .copied()
                .unwrap_or(0.0);
            let query_delete_cost = segment_cost(*query_current, query_previous, nu, lambda);
            let displacement = row.abs_diff(depth) as f64;
            let match_lower_bound = add3(
                interval_dist(*query_current, current_interval.0, current_interval.1),
                interval_dist(query_previous, previous_interval.0, previous_interval.1),
                2.0 * nu * displacement,
            );

            let match_pair = WeightedCost::combine(previous_cell(row - 1), match_lower_bound);
            let delete_query = WeightedCost::combine(column[row - 1], query_delete_cost);
            let delete_target = WeightedCost::combine(previous_cell(row), target_delete_cost);
            column[row] = match_pair.min(delete_query).min(delete_target);
            lower_bound = lower_bound.min(column[row]);
        }

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
        if column.len() != expected || depth == 0 {
            return Some(PointFrontierStep::Advanced {
                lower_bound: WeightedCost::TOP,
                carry: (target, target),
                work: 0,
            });
        }
        let config = self.config();
        let nu = config.stiffness();
        let lambda = config.gap_penalty();
        let target_previous = previous_carry.map_or(0.0, |interval| interval.0);
        let target_delete = segment_cost(target, target_previous, nu, lambda);
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
            if WeightedCost::within(target_delete, cutoff) {
                column[0] = target_delete;
                active.push(0);
                lower_bound = target_delete;
            }
            let mut root_previous = WeightedCost::ZERO;
            for row in 1..=last_row {
                if let Err(requested) = charge_work(&mut work, max_work) {
                    return Some(PointFrontierStep::WorkLimitExceeded {
                        completed: work,
                        requested,
                    });
                }
                let query_current = query[row - 1];
                let query_previous = row
                    .checked_sub(2)
                    .and_then(|index| query.get(index))
                    .copied()
                    .unwrap_or(0.0);
                let query_delete = segment_cost(query_current, query_previous, nu, lambda);
                let root_current = WeightedCost::combine(root_previous, query_delete);
                let pair = WeightedCost::combine(
                    root_previous,
                    match_cost(query_current, query_previous, target, 0.0, row, 1, nu),
                );
                let delete_query = WeightedCost::combine(column[row - 1], query_delete);
                let delete_target = WeightedCost::combine(root_current, target_delete);
                let cost = pair.min(delete_query).min(delete_target);
                if WeightedCost::within(cost, cutoff) {
                    column[row] = cost;
                    active.push(row);
                    lower_bound = lower_bound.min(cost);
                }
                root_previous = root_current;
            }
            return Some(PointFrontierStep::Advanced {
                lower_bound,
                carry: (target, target),
                work,
            });
        }

        let mut seeds = NeighborSeedRows::new(previous_active, last_row);
        let mut next_seed = seeds.next();
        while let Some(start) = next_seed {
            let mut row = start;
            loop {
                while next_seed == Some(row) {
                    next_seed = seeds.next();
                }
                if let Err(requested) = charge_work(&mut work, max_work) {
                    return Some(PointFrontierStep::WorkLimitExceeded {
                        completed: work,
                        requested,
                    });
                }
                let cost = if row == 0 {
                    WeightedCost::combine(previous[0], target_delete)
                } else {
                    let query_current = query[row - 1];
                    let query_previous = row
                        .checked_sub(2)
                        .and_then(|index| query.get(index))
                        .copied()
                        .unwrap_or(0.0);
                    let query_delete = segment_cost(query_current, query_previous, nu, lambda);
                    let pair = WeightedCost::combine(
                        previous[row - 1],
                        match_cost(
                            query_current,
                            query_previous,
                            target,
                            target_previous,
                            row,
                            depth,
                            nu,
                        ),
                    );
                    let delete_query = WeightedCost::combine(column[row - 1], query_delete);
                    let delete_target = WeightedCost::combine(previous[row], target_delete);
                    pair.min(delete_query).min(delete_target)
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
        if column.len() != expected || depth == 0 {
            return Some(PointFrontierStep::Advanced {
                lower_bound: WeightedCost::TOP,
                carry: target,
                work: 0,
            });
        }
        let config = self.config();
        let nu = config.stiffness();
        let lambda = config.gap_penalty();
        let target_previous = previous_carry.unwrap_or(ORIGIN_INTERVAL);
        let target_delete = add3(interval_gap(target, target_previous), nu, lambda);
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
            if WeightedCost::within(target_delete, cutoff) {
                column[0] = target_delete;
                active.push(0);
                lower_bound = target_delete;
            }
            let mut root_previous = WeightedCost::ZERO;
            for row in 1..=last_row {
                if let Err(requested) = charge_work(&mut work, max_work) {
                    return Some(PointFrontierStep::WorkLimitExceeded {
                        completed: work,
                        requested,
                    });
                }
                let query_current = query[row - 1];
                let query_previous = row
                    .checked_sub(2)
                    .and_then(|index| query.get(index))
                    .copied()
                    .unwrap_or(0.0);
                let query_delete = segment_cost(query_current, query_previous, nu, lambda);
                let root_current = WeightedCost::combine(root_previous, query_delete);
                let displacement = row.abs_diff(1) as f64;
                let pair_cost = add3(
                    interval_dist(query_current, target.0, target.1),
                    interval_dist(query_previous, 0.0, 0.0),
                    2.0 * nu * displacement,
                );
                let pair = WeightedCost::combine(root_previous, pair_cost);
                let delete_query = WeightedCost::combine(column[row - 1], query_delete);
                let delete_target = WeightedCost::combine(root_current, target_delete);
                let cost = pair.min(delete_query).min(delete_target);
                if WeightedCost::within(cost, cutoff) {
                    column[row] = cost;
                    active.push(row);
                    lower_bound = lower_bound.min(cost);
                }
                root_previous = root_current;
            }
            return Some(PointFrontierStep::Advanced {
                lower_bound,
                carry: target,
                work,
            });
        }

        let mut seeds = NeighborSeedRows::new(previous_active, last_row);
        let mut next_seed = seeds.next();
        while let Some(start) = next_seed {
            let mut row = start;
            loop {
                while next_seed == Some(row) {
                    next_seed = seeds.next();
                }
                if let Err(requested) = charge_work(&mut work, max_work) {
                    return Some(PointFrontierStep::WorkLimitExceeded {
                        completed: work,
                        requested,
                    });
                }
                let cost = if row == 0 {
                    WeightedCost::combine(previous[0], target_delete)
                } else {
                    let query_current = query[row - 1];
                    let query_previous = row
                        .checked_sub(2)
                        .and_then(|index| query.get(index))
                        .copied()
                        .unwrap_or(0.0);
                    let query_delete = segment_cost(query_current, query_previous, nu, lambda);
                    let displacement = row.abs_diff(depth) as f64;
                    let pair_cost = add3(
                        interval_dist(query_current, target.0, target.1),
                        interval_dist(query_previous, target_previous.0, target_previous.1),
                        2.0 * nu * displacement,
                    );
                    let pair = WeightedCost::combine(previous[row - 1], pair_cost);
                    let delete_query = WeightedCost::combine(column[row - 1], query_delete);
                    let delete_target = WeightedCost::combine(previous[row], target_delete);
                    pair.min(delete_query).min(delete_target)
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
        _target: (f64, f64),
        row: usize,
        column: &[Cost<Self>],
        _plan: &Self::QueryPlan,
    ) -> Option<Cost<Self>> {
        let predecessor = *column.get(row.checked_sub(1)?)?;
        let query_current = *query.get(row - 1)?;
        let query_previous = row
            .checked_sub(2)
            .and_then(|index| query.get(index))
            .copied()
            .unwrap_or(0.0);
        let config = self.config();
        Some(WeightedCost::combine(
            predecessor,
            segment_cost(
                query_current,
                query_previous,
                config.stiffness(),
                config.gap_penalty(),
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
        self.config().distance_with_cutoff(query, candidate, cutoff)
    }

    #[inline]
    fn candidate_lower_bound(
        &self,
        query: &[f64],
        candidate: &[f64],
        _plan: &Self::QueryPlan,
    ) -> Cost<Self> {
        self.config().candidate_lower_bound(query, candidate)
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
    fn empty_vs_nonempty_cost(&self, nonempty: &[f64]) -> Cost<Self> {
        if series_is_finite(nonempty) {
            let config = self.config();
            boundary_cost(nonempty, config.stiffness(), config.gap_penalty())
        } else {
            WeightedCost::TOP
        }
    }
}

impl MetricElasticKernel for MetricTwedConfig {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn reference_distance(x: &[f64], y: &[f64], config: TwedConfig) -> f64 {
        if !series_is_finite(x) || !series_is_finite(y) {
            return f64::INFINITY;
        }
        let nu = config.stiffness();
        let lambda = config.gap_penalty();
        let mut matrix = vec![vec![WeightedCost::TOP; y.len() + 1]; x.len() + 1];
        matrix[0][0] = WeightedCost::ZERO;
        for i in 1..=x.len() {
            let previous = i
                .checked_sub(2)
                .and_then(|index| x.get(index))
                .copied()
                .unwrap_or(0.0);
            matrix[i][0] = WeightedCost::combine(
                matrix[i - 1][0],
                segment_cost(x[i - 1], previous, nu, lambda),
            );
        }
        for j in 1..=y.len() {
            let previous = j
                .checked_sub(2)
                .and_then(|index| y.get(index))
                .copied()
                .unwrap_or(0.0);
            matrix[0][j] = WeightedCost::combine(
                matrix[0][j - 1],
                segment_cost(y[j - 1], previous, nu, lambda),
            );
        }
        for i in 1..=x.len() {
            let x_previous = i
                .checked_sub(2)
                .and_then(|index| x.get(index))
                .copied()
                .unwrap_or(0.0);
            for j in 1..=y.len() {
                let y_previous = j
                    .checked_sub(2)
                    .and_then(|index| y.get(index))
                    .copied()
                    .unwrap_or(0.0);
                matrix[i][j] = WeightedCost::combine(
                    matrix[i - 1][j - 1],
                    match_cost(x[i - 1], x_previous, y[j - 1], y_previous, i, j, nu),
                )
                .min(WeightedCost::combine(
                    matrix[i - 1][j],
                    segment_cost(x[i - 1], x_previous, nu, lambda),
                ))
                .min(WeightedCost::combine(
                    matrix[i][j - 1],
                    segment_cost(y[j - 1], y_previous, nu, lambda),
                ));
            }
        }
        matrix[x.len()][y.len()]
    }

    fn scalar_column(query: &[f64], target: &[f64], config: TwedConfig) -> Vec<f64> {
        let nu = config.stiffness();
        let lambda = config.gap_penalty();
        let mut previous = Vec::with_capacity(query.len() + 1);
        previous.push(0.0);
        for row in 1..=query.len() {
            let query_previous = row
                .checked_sub(2)
                .and_then(|index| query.get(index))
                .copied()
                .unwrap_or(0.0);
            previous.push(WeightedCost::combine(
                previous[row - 1],
                segment_cost(query[row - 1], query_previous, nu, lambda),
            ));
        }
        for (target_index, target_current) in target.iter().enumerate() {
            let depth = target_index + 1;
            let target_previous = target_index
                .checked_sub(1)
                .and_then(|index| target.get(index))
                .copied()
                .unwrap_or(0.0);
            let target_delete = segment_cost(*target_current, target_previous, nu, lambda);
            let mut column = vec![WeightedCost::TOP; query.len() + 1];
            column[0] = WeightedCost::combine(previous[0], target_delete);
            for row in 1..=query.len() {
                let query_previous = row
                    .checked_sub(2)
                    .and_then(|index| query.get(index))
                    .copied()
                    .unwrap_or(0.0);
                column[row] = WeightedCost::combine(
                    previous[row - 1],
                    match_cost(
                        query[row - 1],
                        query_previous,
                        *target_current,
                        target_previous,
                        row,
                        depth,
                        nu,
                    ),
                )
                .min(WeightedCost::combine(
                    column[row - 1],
                    segment_cost(query[row - 1], query_previous, nu, lambda),
                ))
                .min(WeightedCost::combine(previous[row], target_delete));
            }
            previous = column;
        }
        previous
    }

    #[test]
    fn recurrence_branches_boundaries_and_cutoff_are_explicit() {
        let config = TwedConfig::new(1.0, 2.0);
        assert_eq!(config.distance(&[], &[]), 0.0);
        assert_eq!(config.distance(&[3.0], &[]), 6.0);
        assert_eq!(config.distance(&[], &[3.0]), 6.0);
        assert_eq!(config.distance(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        assert_eq!(config.distance(&[1.0], &[2.0]), 1.0);
        assert_eq!(config.distance_with_cutoff(&[1.0], &[2.0], 1.0), Some(1.0));
        assert_eq!(config.distance_with_cutoff(&[1.0], &[2.0], 0.5), None);
        assert_eq!(config.distance_with_cutoff(&[f64::NAN], &[2.0], 4.0), None);
    }

    #[test]
    fn zero_parameter_degeneracy_is_not_mislabelled_metric() {
        let degenerate = TwedConfig::new(0.0, 0.0);
        assert_eq!(degenerate.distance(&[1.0], &[1.0, 1.0]), 0.0);
        assert!(!degenerate.is_metric_configuration());
        assert_eq!(
            degenerate.metric(),
            Err(MetricTwedConfigError::NonPositiveStiffness)
        );
        assert!(MetricTwedConfig::try_new(0.001, 0.0).is_ok());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2_000))]

        #[test]
        fn optimized_dp_matches_independent_full_matrix(
            x in prop::collection::vec(-20i16..=20, 0..9),
            y in prop::collection::vec(-20i16..=20, 0..9),
            raw_nu in 0u8..=5,
            raw_lambda in 0u8..=5,
            cutoff in 0u16..=160,
        ) {
            let x: Vec<f64> = x.into_iter().map(f64::from).collect();
            let y: Vec<f64> = y.into_iter().map(f64::from).collect();
            let config = TwedConfig::new(f64::from(raw_nu), f64::from(raw_lambda));
            let actual = config.distance(&x, &y);
            prop_assert_eq!(actual, reference_distance(&x, &y, config));
            let cutoff = f64::from(cutoff);
            let expected = WeightedCost::within(actual, cutoff).then_some(actual);
            prop_assert_eq!(config.distance_with_cutoff(&x, &y, cutoff), expected);
        }

        #[test]
        fn interval_columns_are_admissible_and_point_bins_are_exact(
            query in prop::collection::vec(-12i16..=12, 1..8),
            target in prop::collection::vec(-12i16..=12, 1..8),
            radii in prop::collection::vec(0u8..=4, 1..8),
            raw_nu in 0u8..=4,
            raw_lambda in 0u8..=4,
        ) {
            let query: Vec<f64> = query.into_iter().map(f64::from).collect();
            let target: Vec<f64> = target.into_iter().map(f64::from).collect();
            let config = TwedConfig::new(f64::from(raw_nu), f64::from(raw_lambda));
            let mut relaxed_previous = vec![WeightedCost::TOP; query.len() + 1];
            let mut point_previous = vec![WeightedCost::TOP; query.len() + 1];
            let mut relaxed_carry = None;
            let mut point_carry = None;

            for (target_index, value) in target.iter().enumerate() {
                let depth = target_index + 1;
                let radius = f64::from(radii[target_index % radii.len()]);
                let mut relaxed = Vec::new();
                let mut point = Vec::new();
                let (_, next_relaxed_carry) = config.step_column(
                    &relaxed_previous,
                    &query,
                    (*value - radius, *value + radius),
                    relaxed_carry,
                    depth,
                    &(),
                    &mut relaxed,
                );
                let (_, next_point_carry) = config.step_column(
                    &point_previous,
                    &query,
                    (*value, *value),
                    point_carry,
                    depth,
                    &(),
                    &mut point,
                );
                let exact = scalar_column(&query, &target[..depth], config);
                for row in 0..=query.len() {
                    prop_assert!(relaxed[row] <= exact[row] + WeightedCost::EPSILON);
                    prop_assert_eq!(point[row], exact[row]);
                }
                relaxed_previous = relaxed;
                point_previous = point;
                relaxed_carry = Some(next_relaxed_carry);
                point_carry = Some(next_point_carry);
            }
        }

        #[test]
        fn local_interval_leaves_are_exact_and_separable(
            x_current in -10i16..=10,
            x_previous in -10i16..=10,
            current_start in -10i16..=10,
            current_width in 0u8..=5,
            previous_start in -10i16..=10,
            previous_width in 0u8..=5,
            raw_nu in 0u8..=4,
            raw_lambda in 0u8..=4,
            i in 1usize..10,
            j in 1usize..10,
        ) {
            let x_current = f64::from(x_current);
            let x_previous = f64::from(x_previous);
            let current_end = current_start + i16::from(current_width);
            let previous_end = previous_start + i16::from(previous_width);
            let current_lo = f64::from(current_start);
            let current_hi = f64::from(current_end);
            let previous_lo = f64::from(previous_start);
            let previous_hi = f64::from(previous_end);
            let nu = f64::from(raw_nu);
            let lambda = f64::from(raw_lambda);

            let closed_match = add3(
                interval_dist(x_current, current_lo, current_hi),
                interval_dist(x_previous, previous_lo, previous_hi),
                2.0 * nu * i.abs_diff(j) as f64,
            );
            let mut brute_match = f64::INFINITY;
            let mut brute_delete = f64::INFINITY;
            for current in current_start..=current_end {
                for previous in previous_start..=previous_end {
                    brute_match = brute_match.min(match_cost(
                        x_current,
                        x_previous,
                        f64::from(current),
                        f64::from(previous),
                        i,
                        j,
                        nu,
                    ));
                    brute_delete = brute_delete.min(segment_cost(
                        f64::from(current),
                        f64::from(previous),
                        nu,
                        lambda,
                    ));
                }
            }
            prop_assert_eq!(closed_match, brute_match);
            prop_assert_eq!(
                add3(interval_gap((current_lo, current_hi), (previous_lo, previous_hi)), nu, lambda),
                brute_delete,
            );
        }

        #[test]
        fn metric_axioms_and_length_bound_hold_for_validated_configuration(
            x in prop::collection::vec(-8i16..=8, 0..7),
            y in prop::collection::vec(-8i16..=8, 0..7),
            z in prop::collection::vec(-8i16..=8, 0..7),
            raw_nu in 1u8..=4,
            raw_lambda in 0u8..=4,
        ) {
            let x_f: Vec<f64> = x.iter().copied().map(f64::from).collect();
            let y_f: Vec<f64> = y.iter().copied().map(f64::from).collect();
            let z_f: Vec<f64> = z.iter().copied().map(f64::from).collect();
            let metric = MetricTwedConfig::try_new(
                f64::from(raw_nu),
                f64::from(raw_lambda),
            ).unwrap();
            let d_xy = metric.distance(&x_f, &y_f);
            let d_yx = metric.distance(&y_f, &x_f);
            let d_xz = metric.distance(&x_f, &z_f);
            let d_yz = metric.distance(&y_f, &z_f);

            prop_assert!(d_xy >= 0.0);
            prop_assert_eq!(d_xy, d_yx);
            prop_assert_eq!(d_xy == 0.0, x == y);
            prop_assert!(d_xz <= d_xy + d_yz + WeightedCost::EPSILON);
            prop_assert!(
                twed_length_lower_bound(x.len(), y.len(), metric.gap_penalty())
                    <= d_xy + WeightedCost::EPSILON
            );
        }
    }
}
