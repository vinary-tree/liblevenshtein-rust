use std::mem::size_of;

use crate::cost::CostMonoid;
use crate::time_series::bounded::{
    IncompleteReason, Operand, ResourceKind, ResourceUsage, TemporalValidationError,
};
use crate::time_series::elastic::{Cost, ElasticKernel, PointFrontierStep, QueryPlanStorage};

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

/// Reusable two-generation exact point frontier for one fixed query.
///
/// This is the single execution spine shared by online stream machines and
/// bounded full-precision survivor verification. Query-plan metadata and both
/// cost/active-row generations are allocated once, then reset in place between
/// candidates. No target prefix is retained.
pub(crate) struct ExactPointWorkspace<K: ElasticKernel> {
    pub(crate) plan: K::QueryPlan,
    pub(crate) current: Vec<Cost<K>>,
    pub(crate) next: Vec<Cost<K>>,
    pub(crate) current_active: Vec<usize>,
    pub(crate) next_active: Vec<usize>,
    carry: Option<K::Carry>,
    final_row: usize,
    consumed_target_len: usize,
    retained_bytes: usize,
    construction_peak_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExactPointAdvance<C> {
    pub(crate) lower_bound: C,
    pub(crate) work: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExactPointFailure {
    pub(crate) reason: IncompleteReason,
    pub(crate) completed_work: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ExactPointDecision<C> {
    WithinCutoff(C),
    AboveCutoff,
    NoFiniteAlignment,
}

impl<K: ElasticKernel> ExactPointWorkspace<K> {
    /// Exact logical plan/frontier storage, excluding the caller-owned query.
    pub(crate) fn storage(
        kernel: &K,
        query_len: usize,
    ) -> Result<QueryPlanStorage, IncompleteReason> {
        let column_width = match kernel.column_len(query_len) {
            Some(width) => width,
            None if query_len == 0 => 0,
            None => {
                return Err(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                });
            }
        };
        let cost_bytes = column_width
            .checked_mul(size_of::<Cost<K>>())
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        let active_bytes = column_width
            .checked_mul(size_of::<usize>())
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        let frontier_bytes =
            cost_bytes
                .checked_add(active_bytes)
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                })?;
        let plan = kernel.query_plan_storage(query_len)?;
        let retained_bytes = plan.retained_bytes().checked_add(frontier_bytes).ok_or(
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            },
        )?;
        Ok(QueryPlanStorage::new(
            retained_bytes,
            plan.construction_peak_bytes().max(retained_bytes),
        ))
    }

    pub(crate) fn try_new(
        kernel: &K,
        query: &[f64],
        max_scratch_bytes: usize,
    ) -> Result<Self, IncompleteReason> {
        let storage = Self::storage(kernel, query.len())?;
        if storage.construction_peak_bytes() > max_scratch_bytes {
            return Err(IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: max_scratch_bytes,
                requested: storage.construction_peak_bytes(),
            });
        }
        // Construct the plan before allocating the reusable frontier: the
        // declared construction peak is therefore exact for this order.
        let plan = kernel.try_plan(query)?;
        let column_width = match kernel.column_len(query.len()) {
            Some(width) => width,
            None if query.is_empty() => 0,
            None => {
                return Err(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                });
            }
        };
        let mut current = Vec::new();
        try_reserve_exact(&mut current, column_width)?;
        current.resize(column_width, K::Monoid::TOP);
        let mut next = Vec::new();
        try_reserve_exact(&mut next, column_width)?;
        next.resize(column_width, K::Monoid::TOP);
        let mut current_active = Vec::new();
        try_reserve_exact(&mut current_active, column_width)?;
        let mut next_active = Vec::new();
        try_reserve_exact(&mut next_active, column_width)?;

        Ok(Self {
            plan,
            current,
            next,
            current_active,
            next_active,
            carry: None,
            final_row: kernel.final_row(query.len()),
            consumed_target_len: 0,
            retained_bytes: storage.retained_bytes(),
            construction_peak_bytes: storage.construction_peak_bytes(),
        })
    }

    #[inline]
    pub(crate) fn plan(&self) -> &K::QueryPlan {
        &self.plan
    }

    #[inline]
    pub(crate) fn current(&self) -> &[Cost<K>] {
        &self.current
    }

    #[inline]
    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    #[inline]
    pub(crate) fn construction_peak_bytes(&self) -> usize {
        self.construction_peak_bytes
    }

    /// Reset to the empty target without reallocating any generation.
    pub(crate) fn reset(&mut self) {
        self.current.fill(K::Monoid::TOP);
        self.next.fill(K::Monoid::TOP);
        self.current_active.clear();
        self.next_active.clear();
        self.carry = None;
        self.consumed_target_len = 0;
    }

    /// Advance one point transactionally through the kernel's exact frontier.
    pub(crate) fn advance(
        &mut self,
        kernel: &K,
        query: &[f64],
        target: f64,
        cutoff: Cost<K>,
        max_work: usize,
    ) -> Result<ExactPointAdvance<Cost<K>>, ExactPointFailure> {
        let next_depth = self
            .consumed_target_len
            .checked_add(1)
            .ok_or(ExactPointFailure {
                reason: IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::SeriesLength,
                },
                completed_work: 0,
            })?;
        self.clear_next_generation();
        let sparse = kernel.step_point_frontier(
            &self.current,
            &self.current_active,
            query,
            target,
            self.carry,
            next_depth,
            &self.plan,
            cutoff,
            max_work,
            &mut self.next,
            &mut self.next_active,
        );
        let (lower_bound, carry, work) = match sparse {
            Some(PointFrontierStep::Advanced {
                lower_bound,
                carry,
                work,
            }) => (lower_bound, carry, work),
            Some(PointFrontierStep::WorkLimitExceeded {
                completed,
                requested,
            }) => {
                self.discard_next_generation();
                return Err(ExactPointFailure {
                    reason: IncompleteReason::BudgetExceeded {
                        resource: ResourceKind::WorkUnits,
                        limit: max_work,
                        requested,
                    },
                    completed_work: completed,
                });
            }
            None => {
                let work = self.current.len();
                if work > max_work {
                    return Err(ExactPointFailure {
                        reason: IncompleteReason::BudgetExceeded {
                            resource: ResourceKind::WorkUnits,
                            limit: max_work,
                            requested: work,
                        },
                        completed_work: 0,
                    });
                }
                let (lower_bound, carry) = kernel.step_column(
                    &self.current,
                    query,
                    (target, target),
                    self.carry,
                    next_depth,
                    &self.plan,
                    &mut self.next,
                );
                self.next_active
                    .extend(self.next.iter().enumerate().filter_map(|(row, cost)| {
                        (K::Monoid::compare(*cost, K::Monoid::TOP) == std::cmp::Ordering::Less)
                            .then_some(row)
                    }));
                (lower_bound, carry, work)
            }
        };
        debug_assert!(work <= max_work);
        debug_assert!(
            self.next_active.windows(2).all(|pair| pair[0] < pair[1]),
            "sparse kernels report sorted unique active rows"
        );
        if !lawful_cost::<K>(lower_bound) {
            self.discard_next_generation();
            return Err(ExactPointFailure {
                reason: IncompleteReason::NumericOverflow,
                completed_work: work,
            });
        }
        for index in 0..self.next_active.len() {
            let row = self.next_active[index];
            let Some(cost) = self.next.get_mut(row) else {
                self.discard_next_generation();
                return Err(ExactPointFailure {
                    reason: IncompleteReason::InvalidStoredData,
                    completed_work: work,
                });
            };
            if !lawful_cost::<K>(*cost) {
                self.discard_next_generation();
                return Err(ExactPointFailure {
                    reason: IncompleteReason::NumericOverflow,
                    completed_work: work,
                });
            }
            if !K::Monoid::within(*cost, cutoff) {
                *cost = K::Monoid::TOP;
            }
        }
        self.next_active.retain(|row| {
            self.next.get(*row).is_some_and(|cost| {
                K::Monoid::compare(*cost, K::Monoid::TOP) == std::cmp::Ordering::Less
            })
        });
        std::mem::swap(&mut self.current, &mut self.next);
        std::mem::swap(&mut self.current_active, &mut self.next_active);
        self.carry = Some(carry);
        self.consumed_target_len = next_depth;
        Ok(ExactPointAdvance { lower_bound, work })
    }

    /// Exact cutoff score using the same online transitions and one allocation.
    pub(crate) fn score_candidate(
        &mut self,
        kernel: &K,
        query: &[f64],
        candidate: &[f64],
        cutoff: Cost<K>,
        max_step_work: usize,
    ) -> Result<ExactPointDecision<Cost<K>>, IncompleteReason> {
        self.reset();
        let structurally_possible =
            kernel.alignment_is_structurally_possible(query.len(), candidate.len());
        if !structurally_possible {
            return Ok(ExactPointDecision::NoFiniteAlignment);
        }
        if query.is_empty() || candidate.is_empty() {
            let cost = if query.is_empty() && candidate.is_empty() {
                kernel.empty_pair_cost()
            } else {
                kernel.empty_vs_nonempty_cost(if query.is_empty() { candidate } else { query })
            };
            return classify_exact::<K>(cost, cutoff, structurally_possible);
        }
        for target in candidate {
            self.advance(kernel, query, *target, cutoff, max_step_work)
                .map_err(|failure| failure.reason)?;
            if self.current_active.is_empty() {
                return if K::Monoid::compare(cutoff, K::Monoid::TOP) == std::cmp::Ordering::Less {
                    Ok(ExactPointDecision::AboveCutoff)
                } else {
                    Err(IncompleteReason::NumericOverflow)
                };
            }
        }
        let cost = self
            .current
            .get(self.final_row)
            .copied()
            .unwrap_or(K::Monoid::TOP);
        classify_exact::<K>(cost, cutoff, structurally_possible)
    }

    fn clear_next_generation(&mut self) {
        for row in self.next_active.drain(..) {
            if let Some(cost) = self.next.get_mut(row) {
                *cost = K::Monoid::TOP;
            }
        }
    }

    pub(crate) fn discard_next_generation(&mut self) {
        self.next.fill(K::Monoid::TOP);
        self.next_active.clear();
    }
}

#[inline]
fn lawful_cost<K: ElasticKernel>(cost: Cost<K>) -> bool {
    K::Monoid::compare(cost, K::Monoid::ZERO) != std::cmp::Ordering::Less
        && K::Monoid::compare(cost, K::Monoid::TOP) != std::cmp::Ordering::Greater
}

fn classify_exact<K: ElasticKernel>(
    cost: Cost<K>,
    cutoff: Cost<K>,
    structurally_possible: bool,
) -> Result<ExactPointDecision<Cost<K>>, IncompleteReason> {
    if !lawful_cost::<K>(cost) {
        return Err(IncompleteReason::NumericOverflow);
    }
    match K::Monoid::compare(cost, K::Monoid::TOP) {
        std::cmp::Ordering::Less if K::Monoid::within(cost, cutoff) => {
            Ok(ExactPointDecision::WithinCutoff(cost))
        }
        std::cmp::Ordering::Less => Ok(ExactPointDecision::AboveCutoff),
        std::cmp::Ordering::Equal
            if !structurally_possible
                || K::Monoid::compare(cutoff, K::Monoid::TOP) == std::cmp::Ordering::Less =>
        {
            Ok(if structurally_possible {
                ExactPointDecision::AboveCutoff
            } else {
                ExactPointDecision::NoFiniteAlignment
            })
        }
        std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => {
            Err(IncompleteReason::NumericOverflow)
        }
    }
}

/// Fixed-query, two-generation online automaton for an elastic kernel.
///
/// The machine constructs one reachable DP generation per target sample and
/// immediately reclaims the preceding scratch generation. It retains the
/// fixed query, two query-width columns, and one fixed-size kernel carry. The
/// carry may contain the immediately preceding target value or interval, but
/// the machine never retains an unbounded target prefix. Its memory is
/// therefore independent of stream length.
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
    workspace: ExactPointWorkspace<K>,
    cutoff: f64,
    limits: OnlineAutomatonLimits,
    initial_distance: Option<f64>,
    scratch_bytes: usize,
    scratch_peak_bytes: usize,
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
            .field("column_width", &self.workspace.current.len())
            .field("active_positions", &self.workspace.current_active.len())
            .field("consumed_target_len", &self.workspace.consumed_target_len)
            .field("scratch_bytes", &self.scratch_bytes)
            .field("scratch_peak_bytes", &self.scratch_peak_bytes)
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
        let query_bytes =
            query
                .len()
                .checked_mul(size_of::<f64>())
                .ok_or(TemporalAutomatonError::Resource(
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    },
                ))?;
        let workspace_storage = ExactPointWorkspace::<K>::storage(&kernel, query.len())
            .map_err(TemporalAutomatonError::Resource)?;
        let scratch_bytes = query_bytes
            .checked_add(workspace_storage.retained_bytes())
            .ok_or(TemporalAutomatonError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        let scratch_peak_bytes = query_bytes
            .checked_add(workspace_storage.construction_peak_bytes())
            .ok_or(TemporalAutomatonError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        if scratch_peak_bytes > limits.max_scratch_bytes {
            return Err(TemporalAutomatonError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested: scratch_peak_bytes,
                },
            ));
        }

        let mut query_storage = Vec::new();
        try_reserve_exact(&mut query_storage, query.len())
            .map_err(TemporalAutomatonError::Resource)?;
        query_storage.extend_from_slice(query);
        let workspace = ExactPointWorkspace::try_new(
            &kernel,
            &query_storage,
            limits.max_scratch_bytes - query_bytes,
        )
        .map_err(TemporalAutomatonError::Resource)?;
        let initial = if query.is_empty() {
            kernel.empty_pair_cost()
        } else {
            kernel.empty_vs_nonempty_cost(query)
        };
        let initial_distance = valid_within(initial, cutoff).then_some(canonical_zero(initial));

        Ok(Self {
            kernel,
            query: query_storage.into_boxed_slice(),
            workspace,
            cutoff,
            limits,
            initial_distance,
            scratch_bytes,
            scratch_peak_bytes,
        })
    }

    /// Observe the current committed prefix without consuming input.
    pub fn observation(&self) -> ElasticOnlineObservation {
        if self.workspace.consumed_target_len == 0 {
            return ElasticOnlineObservation {
                consumed_target_len: 0,
                active_positions: usize::from(self.initial_distance.is_some()),
                distance_within_cutoff: self.initial_distance,
                minimum_active_cost: self.initial_distance,
            };
        }
        let distance_within_cutoff = self
            .workspace
            .current
            .get(self.workspace.final_row)
            .copied()
            .filter(|cost| valid_within(*cost, self.cutoff))
            .map(canonical_zero);
        let minimum_active_cost = self
            .workspace
            .current_active
            .iter()
            .filter_map(|row| self.workspace.current.get(*row).copied())
            .filter(|cost| cost.is_finite())
            .min_by(f64::total_cmp)
            .map(canonical_zero);
        ElasticOnlineObservation {
            consumed_target_len: self.workspace.consumed_target_len,
            active_positions: self.workspace.current_active.len(),
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
                index: self.workspace.consumed_target_len,
            });
        }
        let actual_work = match self.workspace.advance(
            &self.kernel,
            &self.query,
            target,
            self.cutoff,
            self.limits.max_step_work_units,
        ) {
            Ok(advanced) => {
                debug_assert!(lawful_cost::<K>(advanced.lower_bound));
                advanced.work
            }
            Err(failure) => {
                return Ok(self.incomplete(failure.reason, failure.completed_work));
            }
        };
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
        self.workspace.discard_next_generation();
        OnlineStepOutcome::Incomplete {
            reason,
            usage: self.step_usage(completed_work),
        }
    }

    fn step_usage(&self, work: usize) -> ResourceUsage {
        ResourceUsage {
            dp_cells: work,
            work_units: work,
            scratch_bytes: self.scratch_peak_bytes,
            queue_entries: self.workspace.current_active.len(),
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

fn try_reserve_exact<T>(storage: &mut Vec<T>, additional: usize) -> Result<(), IncompleteReason> {
    storage
        .try_reserve_exact(additional)
        .map_err(|_| IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested: additional.saturating_mul(size_of::<T>()),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::kernels::{DtwConfig, ErpConfig, FrechetConfig, TwedConfig};
    use crate::time_series::msm::MsmConfig;
    use crate::time_series::msm_kernel::MsmKernel;

    fn assert_reuse_is_allocation_stable<K>(kernel: K)
    where
        K: ElasticKernel,
        K::Monoid: CostMonoid<Cost = f64>,
    {
        let query = [0.0, 1.0, 2.0, 3.0];
        let candidates: &[&[f64]] = &[
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            &[0.5],
            &[],
            &[3.0, 2.0, 1.0, 0.0, -1.0],
        ];
        let mut workspace =
            ExactPointWorkspace::try_new(&kernel, &query, usize::MAX).expect("small workspace");
        let mut cost_pointers = [workspace.current.as_ptr(), workspace.next.as_ptr()];
        cost_pointers.sort_unstable();
        let mut active_pointers = [
            workspace.current_active.as_ptr(),
            workspace.next_active.as_ptr(),
        ];
        active_pointers.sort_unstable();
        let capacities = (
            workspace.current.capacity(),
            workspace.next.capacity(),
            workspace.current_active.capacity(),
            workspace.next_active.capacity(),
        );
        let retained = workspace.retained_bytes();

        for candidate in candidates {
            // Poison every generation so reset correctness cannot depend on a
            // preceding candidate's length or frontier shape.
            workspace.current.fill(37.0);
            workspace.next.fill(41.0);
            workspace.current_active.clear();
            workspace.next_active.clear();
            workspace.current_active.extend(0..workspace.current.len());
            workspace.next_active.extend(0..workspace.next.len());

            let expected = kernel.exact_with_cutoff(&query, candidate, 10_000.0);
            let step_work = workspace.current.len().max(1);
            let actual = workspace
                .score_candidate(&kernel, &query, candidate, 10_000.0, step_work)
                .expect("finite small score");
            match (actual, expected) {
                (ExactPointDecision::WithinCutoff(actual), Some(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
                (ExactPointDecision::AboveCutoff | ExactPointDecision::NoFiniteAlignment, None) => {
                }
                mismatch => panic!("workspace/legacy mismatch: {mismatch:?}"),
            }
            let mut actual_cost_pointers = [workspace.current.as_ptr(), workspace.next.as_ptr()];
            actual_cost_pointers.sort_unstable();
            let mut actual_active_pointers = [
                workspace.current_active.as_ptr(),
                workspace.next_active.as_ptr(),
            ];
            actual_active_pointers.sort_unstable();
            assert_eq!(actual_cost_pointers, cost_pointers);
            assert_eq!(actual_active_pointers, active_pointers);
            assert_eq!(
                (
                    workspace.current.capacity(),
                    workspace.next.capacity(),
                    workspace.current_active.capacity(),
                    workspace.next_active.capacity(),
                ),
                capacities
            );
            assert_eq!(workspace.retained_bytes(), retained);
        }
    }

    #[test]
    fn every_builtin_reuses_fixed_frontier_allocations_across_candidate_shapes() {
        assert_reuse_is_allocation_stable(MsmKernel::new(MsmConfig::try_new(1.0).unwrap()));
        assert_reuse_is_allocation_stable(ErpConfig::new(0.0));
        assert_reuse_is_allocation_stable(TwedConfig::new(0.5, 1.0));
        assert_reuse_is_allocation_stable(FrechetConfig::new());
        assert_reuse_is_allocation_stable(DtwConfig::new(8));
    }

    #[test]
    fn failed_step_is_transactional_and_retry_matches_a_fresh_workspace() {
        let kernel = FrechetConfig::new();
        let query = [0.0, 1.0, 2.0, 3.0];
        let candidate = [0.0, 1.5, 3.0];
        let mut retried = ExactPointWorkspace::try_new(&kernel, &query, usize::MAX).unwrap();
        let failure = retried
            .advance(&kernel, &query, candidate[0], 100.0, 1)
            .expect_err("the second recurrence row exceeds the one-unit ceiling");
        assert!(matches!(
            failure.reason,
            IncompleteReason::BudgetExceeded {
                resource: ResourceKind::WorkUnits,
                limit: 1,
                requested: 2,
            }
        ));
        assert_eq!(retried.consumed_target_len, 0);

        let width = retried.current.len();
        let retried_score = retried
            .score_candidate(&kernel, &query, &candidate, 100.0, width)
            .unwrap();
        let mut fresh = ExactPointWorkspace::try_new(&kernel, &query, usize::MAX).unwrap();
        let fresh_score = fresh
            .score_candidate(&kernel, &query, &candidate, 100.0, width)
            .unwrap();
        assert_eq!(retried_score, fresh_score);
    }

    #[test]
    fn online_step_usage_reports_every_retained_and_consumed_resource() {
        let mut machine = ElasticOnlineAutomaton::new(
            &[0.0, 1.0],
            FrechetConfig::new(),
            100.0,
            OnlineAutomatonLimits::default(),
        )
        .unwrap();
        let OnlineStepOutcome::Advanced { value, usage } = machine.advance(0.5).unwrap() else {
            panic!("small finite transition must advance");
        };
        assert!(usage.dp_cells > 0);
        assert_eq!(usage.work_units, usage.dp_cells);
        assert_eq!(usage.scratch_bytes, machine.scratch_peak_bytes);
        assert!(usage.scratch_bytes >= machine.scratch_bytes());
        assert_eq!(usage.queue_entries, value.active_positions);
        assert!(usage.queue_entries > 0);
    }
}
