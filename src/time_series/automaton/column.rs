use std::marker::PhantomData;
use std::mem::size_of;

use crate::cost::CostMonoid;
use crate::time_series::bounded::{
    IncompleteReason, Operand, ResourceKind, ResourceUsage, TemporalValidationError,
};
use crate::time_series::elastic::{ElasticKernel, PointFrontierStep};

use super::erp::OnlineAutomatonLimits;
use super::{OnlineStepOutcome, TemporalAutomatonError};

/// Exact observation of one committed target prefix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElasticOnlineObservation {
    /// Number of target samples committed by this machine.
    pub consumed_target_len: usize,
    /// Number of finite DP positions not pruned by the construction cutoff.
    pub active_positions: usize,
    /// Exact final-row cost when it is finite and within the cutoff.
    pub distance_within_cutoff: Option<f64>,
    /// Smallest finite live column cost.
    pub minimum_active_cost: Option<f64>,
}

/// Fixed-query, two-generation online automaton for an elastic kernel.
///
/// The machine constructs one reachable DP generation per target sample and
/// immediately reclaims the preceding scratch generation. It retains the
/// fixed query and two query-width columns, but never a target sample or target
/// prefix. Its memory is therefore independent of stream length.
///
/// A finite cutoff is required because the legacy [`ElasticKernel`] transition
/// contract uses positive infinity both for unreachable cells and for
/// floating-point overflow. With a finite cutoff either condition is safely
/// outside the accepted language; an unbounded production automaton would
/// need a tagged kernel arithmetic result to distinguish them.
pub struct ElasticOnlineAutomaton<K>
where
    K: ElasticKernel,
    K::Monoid: CostMonoid<Cost = f64>,
{
    kernel: K,
    query: Box<[f64]>,
    plan: K::QueryPlan,
    cutoff: f64,
    limits: OnlineAutomatonLimits,
    current: Vec<f64>,
    next: Vec<f64>,
    current_active: Vec<usize>,
    next_active: Vec<usize>,
    carry: Option<K::Carry>,
    final_row: usize,
    consumed_target_len: usize,
    initial_distance: Option<f64>,
    scratch_bytes: usize,
    _cost: PhantomData<K::Monoid>,
}

impl<K> std::fmt::Debug for ElasticOnlineAutomaton<K>
where
    K: ElasticKernel,
    K::Monoid: CostMonoid<Cost = f64>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ElasticOnlineAutomaton")
            .field("kernel", &self.kernel)
            .field("query_len", &self.query.len())
            .field("cutoff", &self.cutoff)
            .field("column_width", &self.current.len())
            .field("active_positions", &self.current_active.len())
            .field("consumed_target_len", &self.consumed_target_len)
            .field("scratch_bytes", &self.scratch_bytes)
            .finish_non_exhaustive()
    }
}

impl<K> ElasticOnlineAutomaton<K>
where
    K: ElasticKernel,
    K::Monoid: CostMonoid<Cost = f64>,
{
    /// Construct a bounded online machine for one fixed finite query.
    pub fn new(
        query: &[f64],
        kernel: K,
        cutoff: f64,
        limits: OnlineAutomatonLimits,
    ) -> Result<Self, TemporalAutomatonError> {
        if query.len() > limits.max_query_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: Operand::Query,
                len: query.len(),
                limit: limits.max_query_len,
            }
            .into());
        }
        if let Some(index) = query.iter().position(|sample| !sample.is_finite()) {
            return Err(TemporalValidationError::NonFiniteSample {
                operand: Operand::Query,
                index,
            }
            .into());
        }
        let kernel = kernel.normalized();
        if !cutoff.is_finite() || !kernel.cutoff_is_valid(cutoff) {
            return Err(TemporalValidationError::InvalidCutoff.into());
        }
        let column_width =
            kernel
                .column_len(query.len())
                .ok_or(TemporalValidationError::InvalidConfiguration(
                    "kernel has no online column state for this query domain",
                ))?;
        if column_width == 0 || column_width > limits.max_frontier_positions {
            return Err(TemporalAutomatonError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::QueueEntries,
                    limit: limits.max_frontier_positions,
                    requested: column_width,
                },
            ));
        }
        let scratch_bytes = query
            .len()
            .checked_mul(size_of::<f64>())
            .and_then(|bytes| {
                column_width
                    .checked_mul(size_of::<f64>())
                    .and_then(|columns| columns.checked_mul(2))
                    .and_then(|columns| bytes.checked_add(columns))
            })
            .and_then(|bytes| {
                column_width
                    .checked_mul(size_of::<usize>())
                    .and_then(|indices| indices.checked_mul(2))
                    .and_then(|indices| bytes.checked_add(indices))
            })
            .ok_or(TemporalAutomatonError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        if scratch_bytes > limits.max_scratch_bytes {
            return Err(TemporalAutomatonError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested: scratch_bytes,
                },
            ));
        }

        let mut query_storage = Vec::new();
        reserve_exact(&mut query_storage, query.len(), limits.max_scratch_bytes)?;
        query_storage.extend_from_slice(query);
        let plan = kernel.plan(&query_storage);
        let mut current = Vec::new();
        reserve_exact(&mut current, column_width, limits.max_scratch_bytes)?;
        current.resize(column_width, K::Monoid::TOP);
        let mut next = Vec::new();
        reserve_exact(&mut next, column_width, limits.max_scratch_bytes)?;
        next.resize(column_width, K::Monoid::TOP);
        let mut current_active = Vec::new();
        reserve_exact(&mut current_active, column_width, limits.max_scratch_bytes)?;
        let mut next_active = Vec::new();
        reserve_exact(&mut next_active, column_width, limits.max_scratch_bytes)?;
        let initial = if query.is_empty() {
            kernel.empty_pair_cost()
        } else {
            kernel.empty_vs_nonempty_cost(query)
        };
        let initial_distance = valid_within(initial, cutoff).then_some(canonical_zero(initial));

        Ok(Self {
            final_row: kernel.final_row(query.len()),
            kernel,
            query: query_storage.into_boxed_slice(),
            plan,
            cutoff,
            limits,
            current,
            next,
            current_active,
            next_active,
            carry: None,
            consumed_target_len: 0,
            initial_distance,
            scratch_bytes,
            _cost: PhantomData,
        })
    }

    /// Observe the current committed prefix without consuming input.
    pub fn observation(&self) -> ElasticOnlineObservation {
        if self.consumed_target_len == 0 {
            return ElasticOnlineObservation {
                consumed_target_len: 0,
                active_positions: usize::from(self.initial_distance.is_some()),
                distance_within_cutoff: self.initial_distance,
                minimum_active_cost: self.initial_distance,
            };
        }
        let distance_within_cutoff = self
            .current
            .get(self.final_row)
            .copied()
            .filter(|cost| valid_within(*cost, self.cutoff))
            .map(canonical_zero);
        let minimum_active_cost = self
            .current_active
            .iter()
            .filter_map(|row| self.current.get(*row).copied())
            .filter(|cost| cost.is_finite())
            .min_by(f64::total_cmp)
            .map(canonical_zero);
        ElasticOnlineObservation {
            consumed_target_len: self.consumed_target_len,
            active_positions: self.current_active.len(),
            distance_within_cutoff,
            minimum_active_cost,
        }
    }

    /// Consume one finite target sample transactionally.
    pub fn advance(
        &mut self,
        target: f64,
    ) -> Result<OnlineStepOutcome<ElasticOnlineObservation>, TemporalValidationError> {
        if !target.is_finite() {
            return Err(TemporalValidationError::NonFiniteSample {
                operand: Operand::Candidate,
                index: self.consumed_target_len,
            });
        }
        let next_depth = match self.consumed_target_len.checked_add(1) {
            Some(depth) => depth,
            None => {
                return Ok(self.incomplete(
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::SeriesLength,
                    },
                    0,
                ));
            }
        };
        for row in self.next_active.drain(..) {
            if let Some(cost) = self.next.get_mut(row) {
                *cost = K::Monoid::TOP;
            }
        }
        let sparse = self.kernel.step_point_frontier(
            &self.current,
            &self.current_active,
            &self.query,
            target,
            self.carry,
            next_depth,
            &self.plan,
            self.cutoff,
            self.limits.max_step_work_units,
            &mut self.next,
            &mut self.next_active,
        );
        let (lower_bound, carry, actual_work) = match sparse {
            Some(PointFrontierStep::Advanced {
                lower_bound,
                carry,
                work,
            }) => (lower_bound, carry, work),
            Some(PointFrontierStep::WorkLimitExceeded {
                completed,
                requested,
            }) => {
                return Ok(self.incomplete(
                    IncompleteReason::BudgetExceeded {
                        resource: ResourceKind::WorkUnits,
                        limit: self.limits.max_step_work_units,
                        requested,
                    },
                    completed,
                ));
            }
            None => {
                let work = self.current.len();
                if work > self.limits.max_step_work_units {
                    return Ok(self.incomplete(
                        IncompleteReason::BudgetExceeded {
                            resource: ResourceKind::WorkUnits,
                            limit: self.limits.max_step_work_units,
                            requested: work,
                        },
                        0,
                    ));
                }
                let (lower_bound, carry) = self.kernel.step_column(
                    &self.current,
                    &self.query,
                    (target, target),
                    self.carry,
                    next_depth,
                    &self.plan,
                    &mut self.next,
                );
                self.next_active.extend(
                    self.next
                        .iter()
                        .enumerate()
                        .filter_map(|(row, cost)| cost.is_finite().then_some(row)),
                );
                (lower_bound, carry, work)
            }
        };
        debug_assert!(actual_work <= self.limits.max_step_work_units);
        debug_assert!(
            self.next_active.windows(2).all(|pair| pair[0] < pair[1]),
            "sparse kernels report sorted unique active rows"
        );
        if lower_bound.is_nan() || lower_bound < 0.0 {
            return Ok(self.incomplete(IncompleteReason::NumericOverflow, actual_work));
        }
        for row in &self.next_active {
            let cost = self
                .next
                .get_mut(*row)
                .expect("a kernel-reported active row is within its fixed column");
            if cost.is_nan() || *cost < 0.0 || *cost == f64::NEG_INFINITY {
                return Ok(self.incomplete(IncompleteReason::NumericOverflow, actual_work));
            }
            if !valid_within(*cost, self.cutoff) {
                *cost = K::Monoid::TOP;
            } else {
                *cost = canonical_zero(*cost);
            }
        }
        self.next_active
            .retain(|row| self.next.get(*row).is_some_and(|cost| cost.is_finite()));

        std::mem::swap(&mut self.current, &mut self.next);
        std::mem::swap(&mut self.current_active, &mut self.next_active);
        self.carry = Some(carry);
        self.consumed_target_len = next_depth;
        Ok(OnlineStepOutcome::Advanced {
            value: self.observation(),
            usage: self.step_usage(actual_work),
        })
    }

    /// Fixed logical bytes retained by this stream machine.
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }

    fn incomplete(
        &mut self,
        reason: IncompleteReason,
        completed_work: usize,
    ) -> OnlineStepOutcome<ElasticOnlineObservation> {
        self.next.fill(K::Monoid::TOP);
        self.next_active.clear();
        OnlineStepOutcome::Incomplete {
            reason,
            usage: self.step_usage(completed_work),
        }
    }

    fn step_usage(&self, work: usize) -> ResourceUsage {
        ResourceUsage {
            dp_cells: work,
            work_units: work,
            scratch_bytes: self.scratch_bytes,
            queue_entries: self.current_active.len(),
            ..ResourceUsage::default()
        }
    }
}

#[inline]
fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

#[inline]
fn valid_within(value: f64, cutoff: f64) -> bool {
    value.is_finite() && value >= 0.0 && value <= cutoff
}

fn reserve_exact<T>(
    storage: &mut Vec<T>,
    additional: usize,
    limit: usize,
) -> Result<(), TemporalAutomatonError> {
    storage.try_reserve_exact(additional).map_err(|_| {
        TemporalAutomatonError::Resource(IncompleteReason::BudgetExceeded {
            resource: ResourceKind::ScratchBytes,
            limit,
            requested: additional.saturating_mul(size_of::<T>()),
        })
    })
}
