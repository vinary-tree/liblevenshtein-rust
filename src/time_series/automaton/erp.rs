use std::mem::size_of;

use super::arena::{TemporalArenaLimits, TemporalStateArena, TemporalStateId};
use super::cache::BoundedTransitionCache;
use super::state::TemporalPosition;
use super::{OnlineStepOutcome, TemporalAutomatonError};
use crate::time_series::bounded::{
    IncompleteReason, Operand, ResourceKind, ResourceLimits, ResourceUsage, TemporalValidationError,
};
use crate::time_series::kernels::ErpConfig;

/// Per-machine and per-transition limits for a fixed-query online automaton.
///
/// Work is limited **per consumed target unit**, not cumulatively over the
/// session. A cumulative ceiling would inevitably stop every infinite stream
/// even when retained memory is stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OnlineAutomatonLimits {
    /// Maximum samples copied into the fixed query plan.
    pub max_query_len: usize,
    /// Maximum canonical positions retained in either live generation.
    pub max_frontier_positions: usize,
    /// Maximum logical transition candidates and canonicalization cells
    /// inspected for one target unit.
    pub max_step_work_units: usize,
    /// Maximum bytes retained by the query plan, two frontier generations, and
    /// generation-stamped scratch arrays.
    pub max_scratch_bytes: usize,
}

impl Default for OnlineAutomatonLimits {
    fn default() -> Self {
        Self {
            max_query_len: 1_000_000,
            max_frontier_positions: 1_000_001,
            max_step_work_units: 100_000_000,
            max_scratch_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Exact observation after one target prefix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErpOnlineObservation {
    /// Number of target samples committed by this machine.
    pub consumed_target_len: usize,
    /// Number of non-subsumed weighted positions in the live frontier.
    pub active_positions: usize,
    /// Exact ERP distance for the current target prefix when it satisfies the
    /// inclusive construction cutoff.
    pub distance_within_cutoff: Option<f64>,
    /// Smallest accumulated path cost before final query deletions.
    pub minimum_active_cost: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScalarInterval {
    low: f64,
    high: f64,
}

impl ScalarInterval {
    fn point(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self {
            low: value,
            high: value,
        })
    }

    fn from_bounds(low: f64, high: f64) -> Option<Self> {
        (!low.is_nan() && !high.is_nan() && low <= high).then_some(Self { low, high })
    }

    #[inline]
    fn distance(self, value: f64) -> f64 {
        if value < self.low {
            self.low - value
        } else if value > self.high {
            value - self.high
        } else {
            0.0
        }
    }
}

#[inline]
fn minimum_option(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ErpFrontierTransition {
    pub(crate) target: Option<TemporalStateId>,
    pub(crate) work_units: usize,
}

/// Query-local compact-state engine used by dictionary-product traversal.
///
/// The embedded online worker supplies the same two-generation transition
/// implementation used by [`ErpOnlineAutomaton`]. Before each dictionary edge,
/// its current generation is populated from one interned state ID; equal
/// targets are collision-checked and interned once.
pub(crate) struct ErpFrontierMachine {
    worker: ErpOnlineAutomaton,
    arena: TemporalStateArena<()>,
    cache: BoundedTransitionCache<u8>,
    seed: TemporalStateId,
    arena_position_limit: usize,
}

impl ErpFrontierMachine {
    pub(crate) fn new(
        query: &[f64],
        config: ErpConfig,
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<Self, TemporalAutomatonError> {
        let frontier_limit = query
            .len()
            .checked_add(1)
            .ok_or(TemporalAutomatonError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::QueueEntries,
                },
            ))?
            .min(limits.max_queue_entries.max(1));
        let worker = ErpOnlineAutomaton::new(
            query,
            config,
            cutoff,
            OnlineAutomatonLimits {
                max_query_len: limits.max_series_len,
                max_frontier_positions: frontier_limit,
                max_step_work_units: limits.max_work_units,
                max_scratch_bytes: limits.max_scratch_bytes,
            },
        )?;
        let remaining_scratch = limits
            .max_scratch_bytes
            .checked_sub(worker.scratch_bytes)
            .ok_or(TemporalAutomatonError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested: worker.scratch_bytes,
                },
            ))?;
        // State storage is semantically required; the cache is only a work
        // optimization. Give the arena three quarters of the remaining
        // logical budget and keep the cache rigorously bounded by the rest.
        let arena_bytes = remaining_scratch.saturating_mul(3) / 4;
        let cache_bytes = remaining_scratch.saturating_sub(arena_bytes);
        let bytes_per_position = size_of::<TemporalPosition>().max(1);
        let bytes_per_state = size_of::<Vec<TemporalPosition>>()
            .saturating_add(size_of::<u64>())
            .saturating_add(size_of::<TemporalStateId>())
            .max(1);
        let minimum_arena_bytes = bytes_per_state
            .max(bytes_per_position)
            .checked_mul(2)
            .ok_or(TemporalAutomatonError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        if arena_bytes < minimum_arena_bytes {
            return Err(TemporalAutomatonError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested: worker.scratch_bytes.saturating_add(minimum_arena_bytes),
                },
            ));
        }
        let state_bytes = arena_bytes / 2;
        let position_bytes = arena_bytes.saturating_sub(state_bytes);
        let arena_limits = TemporalArenaLimits {
            max_states: limits
                .max_trie_nodes
                .min(state_bytes / bytes_per_state)
                .min(u32::MAX as usize),
            max_positions: position_bytes / bytes_per_position,
        };
        let cache_entry_bytes = size_of::<(TemporalStateId, u8)>()
            .saturating_add(size_of::<Option<TemporalStateId>>())
            .saturating_mul(3)
            .max(1);
        let mut arena = TemporalStateArena::new(arena_limits);
        let seed = arena
            .intern((), worker.current.clone())
            .map_err(TemporalAutomatonError::Resource)?;
        Ok(Self {
            worker,
            arena,
            cache: BoundedTransitionCache::new(
                limits.max_trie_edges.min(cache_bytes / cache_entry_bytes),
            ),
            seed,
            arena_position_limit: arena_limits.max_positions,
        })
    }

    #[inline]
    pub(crate) fn seed(&self) -> TemporalStateId {
        self.seed
    }

    pub(crate) fn transition(
        &mut self,
        source: TemporalStateId,
        label: u8,
        bounds: (f64, f64),
    ) -> Result<ErpFrontierTransition, IncompleteReason> {
        if let Some(target) = self.cache.get(source, label) {
            return Ok(ErpFrontierTransition {
                target,
                work_units: 0,
            });
        }
        let interval = ScalarInterval::from_bounds(bounds.0, bounds.1)
            .ok_or(IncompleteReason::InvalidStoredData)?;
        self.worker.current.clear();
        self.worker
            .current
            .extend_from_slice(&self.arena.get(source).positions);
        let outcome = self.worker.advance_interval(interval);
        let (work_units, advanced) = match outcome {
            OnlineStepOutcome::Advanced { usage, .. } => (usage.work_units, true),
            OnlineStepOutcome::Incomplete { reason, .. } => return Err(reason),
        };
        debug_assert!(advanced);

        let target = if self.worker.current.is_empty() {
            None
        } else {
            let mut compact = Vec::new();
            compact
                .try_reserve_exact(self.worker.current.len())
                .map_err(|_| IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: self.arena_position_limit,
                    requested: self
                        .arena
                        .position_count()
                        .saturating_add(self.worker.current.len()),
                })?;
            compact.extend_from_slice(&self.worker.current);
            Some(self.arena.intern((), compact)?)
        };
        // Cache allocation is never a correctness dependency. If its bounded
        // reservation fails, this exact transition remains usable and may be
        // recomputed later.
        let _ = self.cache.insert(source, label, target);
        Ok(ErpFrontierTransition { target, work_units })
    }

    pub(crate) fn lower_bound(&self, state: TemporalStateId) -> f64 {
        self.arena
            .get(state)
            .positions
            .iter()
            .map(|position| position.cost.get())
            .min_by(f64::total_cmp)
            .unwrap_or(f64::INFINITY)
    }

    pub(crate) fn final_cost(
        &self,
        state: TemporalStateId,
    ) -> Result<Option<f64>, IncompleteReason> {
        self.worker
            .final_distance_for_positions(&self.arena.get(state).positions)
    }

    pub(crate) fn transition_work_bound(
        &self,
        state: TemporalStateId,
    ) -> Result<usize, IncompleteReason> {
        self.worker
            .transition_work_bound_for(&self.arena.get(state).positions)
    }

    pub(crate) fn retained_counts(&self) -> (usize, usize, usize) {
        (
            self.arena.state_count(),
            self.arena.position_count(),
            self.cache.len(),
        )
    }

    pub(crate) fn retained_scratch_bytes(&self) -> Result<usize, IncompleteReason> {
        let (states, positions, cached_transitions) = self.retained_counts();
        let state_bytes = states
            .checked_mul(
                size_of::<Vec<TemporalPosition>>()
                    .saturating_add(size_of::<u64>())
                    .saturating_add(size_of::<TemporalStateId>()),
            )
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        let position_bytes = positions.checked_mul(size_of::<TemporalPosition>()).ok_or(
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            },
        )?;
        let cache_bytes = cached_transitions
            .checked_mul(
                size_of::<(TemporalStateId, u8)>()
                    .saturating_add(size_of::<Option<TemporalStateId>>())
                    .saturating_mul(3),
            )
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        self.worker
            .scratch_bytes
            .checked_add(state_bytes)
            .and_then(|bytes| bytes.checked_add(position_bytes))
            .and_then(|bytes| bytes.checked_add(cache_bytes))
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })
    }
}

/// Fixed-query ERP automaton for an online target stream of unknown length.
///
/// The machine retains the query, its deletion costs, two canonical frontier
/// generations, and query-sized generation-stamped scratch. It never stores a
/// consumed target sample. `advance` is transactional: a resource or numeric
/// failure leaves the semantic prefix and current frontier unchanged.
#[derive(Debug)]
pub struct ErpOnlineAutomaton {
    query: Box<[f64]>,
    deletion_costs: Box<[f64]>,
    deletion_suffix_costs: Box<[f64]>,
    gap: f64,
    cutoff: f64,
    limits: OnlineAutomatonLimits,
    current: Vec<TemporalPosition>,
    next: Vec<TemporalPosition>,
    scratch_costs: Box<[f64]>,
    scratch_generation: Box<[u32]>,
    generation: u32,
    consumed_target_len: usize,
    scratch_bytes: usize,
}

impl ErpOnlineAutomaton {
    /// Construct a stable online ERP machine for `query`.
    pub fn new(
        query: &[f64],
        config: ErpConfig,
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
        if let Some(index) = query.iter().position(|value| !value.is_finite()) {
            return Err(TemporalValidationError::NonFiniteSample {
                operand: Operand::Query,
                index,
            }
            .into());
        }
        if cutoff.is_nan() || cutoff < 0.0 || cutoff == f64::NEG_INFINITY {
            return Err(TemporalValidationError::InvalidCutoff.into());
        }
        if limits.max_frontier_positions == 0 {
            return Err(TemporalValidationError::InvalidConfiguration(
                "online frontier limit must be positive",
            )
            .into());
        }

        let slot_count = query.len().checked_add(1).ok_or({
            TemporalAutomatonError::Resource(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })
        })?;
        let scratch_bytes = retained_bytes(query.len(), slot_count).ok_or({
            TemporalAutomatonError::Resource(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })
        })?;
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
        try_reserve_exact(&mut query_storage, query.len(), limits.max_scratch_bytes)?;
        query_storage.extend_from_slice(query);

        let gap = config.normalized().gap_value();
        let mut deletion_storage = Vec::new();
        try_reserve_exact(&mut deletion_storage, query.len(), limits.max_scratch_bytes)?;
        for value in query {
            let cost = (*value - gap).abs();
            if !cost.is_finite() {
                return Err(TemporalAutomatonError::Resource(
                    IncompleteReason::NumericOverflow,
                ));
            }
            deletion_storage.push(cost);
        }
        let mut suffix_storage = Vec::new();
        try_reserve_exact(&mut suffix_storage, slot_count, limits.max_scratch_bytes)?;
        suffix_storage.resize(slot_count, 0.0);
        for query_index in (0..query.len()).rev() {
            let suffix = suffix_storage[query_index + 1] + deletion_storage[query_index];
            if suffix.is_finite() {
                suffix_storage[query_index] = suffix;
            } else if cutoff.is_finite() {
                suffix_storage[query_index] = f64::INFINITY;
            } else {
                return Err(TemporalAutomatonError::Resource(
                    IncompleteReason::NumericOverflow,
                ));
            }
        }

        let mut current = Vec::new();
        try_reserve_exact(
            &mut current,
            slot_count.min(limits.max_frontier_positions),
            limits.max_scratch_bytes,
        )?;
        current.push(
            TemporalPosition::new(0, 0.0).expect("zero-index zero-cost ERP root is canonical"),
        );
        let mut next = Vec::new();
        try_reserve_exact(
            &mut next,
            slot_count.min(limits.max_frontier_positions),
            limits.max_scratch_bytes,
        )?;

        let mut scratch_costs = Vec::new();
        try_reserve_exact(&mut scratch_costs, slot_count, limits.max_scratch_bytes)?;
        scratch_costs.resize(slot_count, 0.0);
        let mut scratch_generation = Vec::new();
        try_reserve_exact(
            &mut scratch_generation,
            slot_count,
            limits.max_scratch_bytes,
        )?;
        scratch_generation.resize(slot_count, 0);

        let machine = Self {
            query: query_storage.into_boxed_slice(),
            deletion_costs: deletion_storage.into_boxed_slice(),
            deletion_suffix_costs: suffix_storage.into_boxed_slice(),
            gap,
            cutoff,
            limits,
            current,
            next,
            scratch_costs: scratch_costs.into_boxed_slice(),
            scratch_generation: scratch_generation.into_boxed_slice(),
            generation: 0,
            consumed_target_len: 0,
            scratch_bytes,
        };
        machine
            .final_distance_for_positions(&machine.current)
            .map_err(TemporalAutomatonError::Resource)?;
        Ok(machine)
    }

    /// Observation for the already committed prefix without consuming input.
    pub fn observation(&self) -> ErpOnlineObservation {
        let distance_within_cutoff = self
            .final_distance_within_cutoff()
            .expect("committed ERP frontier has a representable final suffix");
        ErpOnlineObservation {
            consumed_target_len: self.consumed_target_len,
            active_positions: self.current.len(),
            distance_within_cutoff,
            minimum_active_cost: self
                .current
                .iter()
                .map(|position| position.cost.get())
                .min_by(f64::total_cmp),
        }
    }

    /// Consume one finite target sample.
    pub fn advance(
        &mut self,
        target: f64,
    ) -> Result<OnlineStepOutcome<ErpOnlineObservation>, TemporalValidationError> {
        let interval =
            ScalarInterval::point(target).ok_or(TemporalValidationError::NonFiniteSample {
                operand: Operand::Candidate,
                index: self.consumed_target_len,
            })?;
        Ok(self.advance_interval(interval))
    }

    fn advance_interval(
        &mut self,
        target: ScalarInterval,
    ) -> OnlineStepOutcome<ErpOnlineObservation> {
        let Some(next_consumed) = self.consumed_target_len.checked_add(1) else {
            return self.incomplete(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::SeriesLength,
                },
                0,
            );
        };
        self.next.clear();
        self.start_scratch_generation();
        let mut work = 0usize;

        if self.current.is_empty() {
            self.consumed_target_len = next_consumed;
            return OnlineStepOutcome::Advanced {
                value: self.observation(),
                usage: self.step_usage(work),
            };
        }

        let insertion_cost = target.distance(self.gap);
        if !insertion_cost.is_finite() {
            return self.incomplete(IncompleteReason::NumericOverflow, work);
        }

        let sparse_candidate_work = match self.sparse_candidate_work_bound(&self.current) {
            Ok(bound) => bound,
            Err(reason) => return self.incomplete(reason, work),
        };
        let dense_candidate_work = match self.query.len().checked_add(1) {
            Some(bound) => bound,
            None => {
                return self.incomplete(
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::WorkUnits,
                    },
                    work,
                );
            }
        };

        if sparse_candidate_work <= dense_candidate_work {
            // Very narrow cutoffs often leave one position. Expanding only
            // that position can stop as soon as its non-negative suffix is
            // over cutoff, avoiding a full query scan.
            for position_index in 0..self.current.len() {
                let position = self.current[position_index];
                if let Err(reason) = self.charge_work(&mut work, 1) {
                    return self.incomplete(reason, work);
                }
                match self.checked_path_add(position.cost.get(), insertion_cost) {
                    Ok(Some(cost)) => self.record_candidate(position.query_index(), cost),
                    Ok(None) => {}
                    Err(reason) => return self.incomplete(reason, work),
                }

                let mut deletion_mass = 0.0;
                for substitution_index in position.query_index()..self.query.len() {
                    if let Err(reason) = self.charge_work(&mut work, 1) {
                        return self.incomplete(reason, work);
                    }
                    let base = match self.checked_path_add(position.cost.get(), deletion_mass) {
                        Ok(Some(cost)) => cost,
                        Ok(None) => break,
                        Err(reason) => return self.incomplete(reason, work),
                    };
                    let candidate = match self
                        .checked_path_add(base, target.distance(self.query[substitution_index]))
                    {
                        Ok(value) => value,
                        Err(reason) => return self.incomplete(reason, work),
                    };
                    if let Some(cost) = candidate {
                        self.record_candidate(substitution_index + 1, cost);
                    }
                    deletion_mass = match self.checked_unbounded_add(
                        deletion_mass,
                        self.deletion_costs[substitution_index],
                    ) {
                        Ok(cost) => cost,
                        Err(reason) => return self.incomplete(reason, work),
                    };
                }
            }
        } else {
            // Reconstruct the deletion closure from left to right, then
            // evaluate one ordinary ERP column. This is equivalent to the
            // sparse expansion but linear for broad frontiers.
            let mut position_cursor = 0usize;
            let mut previous_closure: Option<f64> = None;
            let mut next_left: Option<f64> = None;
            for query_index in 0..=self.query.len() {
                if let Err(reason) = self.charge_work(&mut work, 1) {
                    return self.incomplete(reason, work);
                }
                let previous_left = previous_closure;
                let exact = self.current.get(position_cursor).and_then(|position| {
                    (position.query_index() == query_index).then_some(position.cost.get())
                });
                if exact.is_some() {
                    position_cursor += 1;
                }

                let deletion_closure = if query_index == 0 {
                    None
                } else {
                    match previous_left {
                        Some(cost) => match self
                            .checked_path_add(cost, self.deletion_costs[query_index - 1])
                        {
                            Ok(value) => value,
                            Err(reason) => return self.incomplete(reason, work),
                        },
                        None => None,
                    }
                };
                let current_closure = minimum_option(exact, deletion_closure);
                let insert = match current_closure {
                    Some(cost) => match self.checked_path_add(cost, insertion_cost) {
                        Ok(value) => value,
                        Err(reason) => return self.incomplete(reason, work),
                    },
                    None => None,
                };
                let substitute = if query_index == 0 {
                    None
                } else {
                    match previous_left {
                        Some(cost) => match self
                            .checked_path_add(cost, target.distance(self.query[query_index - 1]))
                        {
                            Ok(value) => value,
                            Err(reason) => return self.incomplete(reason, work),
                        },
                        None => None,
                    }
                };
                let delete = if query_index == 0 {
                    None
                } else {
                    match next_left {
                        Some(cost) => match self
                            .checked_path_add(cost, self.deletion_costs[query_index - 1])
                        {
                            Ok(value) => value,
                            Err(reason) => return self.incomplete(reason, work),
                        },
                        None => None,
                    }
                };
                let candidate = minimum_option(minimum_option(insert, substitute), delete);
                if let Some(cost) = candidate {
                    self.record_candidate(query_index, cost);
                }
                previous_closure = current_closure;
                next_left = candidate;
            }
            debug_assert_eq!(position_cursor, self.current.len());
        }

        let mut epsilon_best: Option<f64> = None;
        for query_index in 0..=self.query.len() {
            if let Err(reason) = self.charge_work(&mut work, 1) {
                return self.incomplete(reason, work);
            }
            if query_index > 0 {
                epsilon_best = match epsilon_best {
                    Some(cost) => {
                        match self.checked_path_add(cost, self.deletion_costs[query_index - 1]) {
                            Ok(value) => value,
                            Err(reason) => return self.incomplete(reason, work),
                        }
                    }
                    None => None,
                };
            }

            if self.scratch_generation[query_index] != self.generation {
                continue;
            }
            let candidate = self.scratch_costs[query_index];
            let dominated = epsilon_best.is_some_and(|reachable| reachable <= candidate);
            if !dominated {
                let requested = self.next.len().saturating_add(1);
                if requested > self.limits.max_frontier_positions {
                    return self.incomplete(
                        IncompleteReason::BudgetExceeded {
                            resource: ResourceKind::QueueEntries,
                            limit: self.limits.max_frontier_positions,
                            requested,
                        },
                        work,
                    );
                }
                self.next.push(
                    TemporalPosition::new(query_index, candidate)
                        .expect("validated finite candidate and checked query index"),
                );
            }
            epsilon_best = Some(match epsilon_best {
                Some(reachable) => reachable.min(candidate),
                None => candidate,
            });
        }

        if let Err(reason) = self.final_distance_for_positions(&self.next) {
            return self.incomplete(reason, work);
        }

        std::mem::swap(&mut self.current, &mut self.next);
        self.next.clear();
        self.consumed_target_len = next_consumed;
        OnlineStepOutcome::Advanced {
            value: self.observation(),
            usage: self.step_usage(work),
        }
    }

    fn start_scratch_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.scratch_generation.fill(0);
            self.generation = 1;
        }
    }

    #[inline]
    fn record_candidate(&mut self, query_index: usize, cost: f64) {
        debug_assert!(cost.is_finite() && cost >= 0.0 && cost <= self.cutoff);
        if self.scratch_generation[query_index] == self.generation {
            if cost < self.scratch_costs[query_index] {
                self.scratch_costs[query_index] = cost;
            }
        } else {
            self.scratch_generation[query_index] = self.generation;
            self.scratch_costs[query_index] = cost;
        }
    }

    #[inline]
    fn checked_path_add(&self, left: f64, right: f64) -> Result<Option<f64>, IncompleteReason> {
        let sum = left + right;
        if sum.is_finite() {
            Ok((sum <= self.cutoff).then_some(sum))
        } else if self.cutoff.is_finite() {
            Ok(None)
        } else {
            Err(IncompleteReason::NumericOverflow)
        }
    }

    fn sparse_candidate_work_bound(
        &self,
        positions: &[TemporalPosition],
    ) -> Result<usize, IncompleteReason> {
        positions.iter().try_fold(0usize, |work, position| {
            let substitutions = self
                .query
                .len()
                .checked_sub(position.query_index())
                .ok_or(IncompleteReason::InvalidStoredData)?;
            work.checked_add(substitutions.saturating_add(1)).ok_or(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::WorkUnits,
                },
            )
        })
    }

    fn transition_work_bound_for(
        &self,
        positions: &[TemporalPosition],
    ) -> Result<usize, IncompleteReason> {
        if positions.is_empty() {
            return Ok(0);
        }
        let width =
            self.query
                .len()
                .checked_add(1)
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::WorkUnits,
                })?;
        let candidate_work = self.sparse_candidate_work_bound(positions)?.min(width);
        candidate_work
            .checked_add(width)
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::WorkUnits,
            })
    }

    #[inline]
    fn checked_unbounded_add(&self, left: f64, right: f64) -> Result<f64, IncompleteReason> {
        let sum = left + right;
        if sum.is_finite() {
            Ok(sum)
        } else if self.cutoff.is_finite() {
            Ok(f64::INFINITY)
        } else {
            Err(IncompleteReason::NumericOverflow)
        }
    }

    #[inline]
    fn charge_work(&self, work: &mut usize, amount: usize) -> Result<(), IncompleteReason> {
        let requested = work
            .checked_add(amount)
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::WorkUnits,
            })?;
        if requested > self.limits.max_step_work_units {
            return Err(IncompleteReason::BudgetExceeded {
                resource: ResourceKind::WorkUnits,
                limit: self.limits.max_step_work_units,
                requested,
            });
        }
        *work = requested;
        Ok(())
    }

    fn final_distance_within_cutoff(&self) -> Result<Option<f64>, IncompleteReason> {
        self.final_distance_for_positions(&self.current)
    }

    fn final_distance_for_positions(
        &self,
        positions: &[TemporalPosition],
    ) -> Result<Option<f64>, IncompleteReason> {
        let mut best: Option<f64> = None;
        for position in positions {
            let cost = position.cost.get()
                + self
                    .deletion_suffix_costs
                    .get(position.query_index())
                    .copied()
                    .ok_or(IncompleteReason::InvalidStoredData)?;
            if !cost.is_finite() {
                if self.cutoff.is_finite() {
                    continue;
                }
                return Err(IncompleteReason::NumericOverflow);
            }
            if cost <= self.cutoff {
                best = Some(best.map_or(cost, |existing| existing.min(cost)));
            }
        }
        Ok(best)
    }

    fn incomplete(
        &mut self,
        reason: IncompleteReason,
        work: usize,
    ) -> OnlineStepOutcome<ErpOnlineObservation> {
        self.next.clear();
        OnlineStepOutcome::Incomplete {
            reason,
            usage: self.step_usage(work),
        }
    }

    fn step_usage(&self, work: usize) -> ResourceUsage {
        ResourceUsage {
            dp_cells: work,
            work_units: work,
            scratch_bytes: self.scratch_bytes,
            queue_entries: self.current.len().max(self.next.len()),
            ..ResourceUsage::default()
        }
    }
}

fn retained_bytes(query_len: usize, slot_count: usize) -> Option<usize> {
    query_len
        .checked_mul(size_of::<f64>())?
        .checked_mul(2)?
        .checked_add(slot_count.checked_mul(size_of::<f64>())?)?
        .checked_add(
            slot_count
                .checked_mul(size_of::<TemporalPosition>())?
                .checked_mul(2)?,
        )?
        .checked_add(slot_count.checked_mul(size_of::<f64>())?)?
        .checked_add(slot_count.checked_mul(size_of::<u32>())?)
}

fn try_reserve_exact<T>(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transactional_budget_stop_does_not_consume_target() {
        let limits = OnlineAutomatonLimits {
            max_step_work_units: 0,
            ..OnlineAutomatonLimits::default()
        };
        let mut machine = ErpOnlineAutomaton::new(&[1.0, 2.0], ErpConfig::default(), 10.0, limits)
            .expect("small query fits fixed machine limits");
        let before = machine.observation();
        let outcome = machine.advance(1.0).expect("target sample is finite");
        assert!(matches!(outcome, OnlineStepOutcome::Incomplete { .. }));
        assert_eq!(machine.observation(), before);
    }

    #[test]
    fn unknown_length_target_is_not_buffered() {
        let mut machine = ErpOnlineAutomaton::new(
            &[0.0, 1.0, 2.0],
            ErpConfig::default(),
            f64::INFINITY,
            OnlineAutomatonLimits::default(),
        )
        .expect("small query fits fixed machine limits");
        let scratch_bytes = machine.scratch_bytes;
        for index in 0..100_000usize {
            let target = (index % 3) as f64;
            assert!(machine
                .advance(target)
                .expect("target is finite")
                .advanced());
            assert_eq!(machine.scratch_bytes, scratch_bytes);
        }
        assert_eq!(machine.observation().consumed_target_len, 100_000);
    }

    #[test]
    fn initial_observation_is_query_to_empty_distance() {
        let machine = ErpOnlineAutomaton::new(
            &[1.0, -2.0],
            ErpConfig::default(),
            3.0,
            OnlineAutomatonLimits::default(),
        )
        .expect("small query fits fixed machine limits");
        assert_eq!(machine.observation().distance_within_cutoff, Some(3.0));
    }

    #[test]
    fn abstract_extreme_bins_accept_infinite_bounds_but_not_nan() {
        let low = ScalarInterval::from_bounds(f64::NEG_INFINITY, 1.0)
            .expect("lower extreme quantizer bin is a lawful interval");
        let high = ScalarInterval::from_bounds(1.0, f64::INFINITY)
            .expect("upper extreme quantizer bin is a lawful interval");
        assert_eq!(low.distance(0.0), 0.0);
        assert_eq!(low.distance(2.0), 1.0);
        assert_eq!(high.distance(0.0), 1.0);
        assert_eq!(high.distance(2.0), 0.0);
        assert!(ScalarInterval::from_bounds(f64::NAN, 1.0).is_none());
        assert!(ScalarInterval::from_bounds(1.0, f64::NAN).is_none());
        assert!(ScalarInterval::from_bounds(2.0, 1.0).is_none());
    }
}
