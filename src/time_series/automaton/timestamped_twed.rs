use std::mem::size_of;

use crate::cost::{CostMonoid, WeightedCost};
use crate::time_series::bounded::{
    IncompleteReason, Operand, ResourceKind, ResourceUsage, TemporalValidationError,
};
use crate::time_series::elastic::sparse::{charge_work, NeighborSeedRows};
use crate::time_series::timestamped_twed::{
    delete_cost, match_cost, MetricTimestampedTwedConfig, TimestampUnit, TimestampedSeries,
    TimestampedTwedError,
};

use super::erp::OnlineAutomatonLimits;
use super::OnlineStepOutcome;

/// Exact observation of one committed explicit-timestamp target prefix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampedTwedOnlineObservation {
    /// Number of target points committed by the machine.
    pub consumed_target_len: usize,
    /// Number of finite DP cells still viable under the construction cutoff.
    pub active_positions: usize,
    /// Exact complete-prefix distance when finite and within the cutoff.
    pub distance_within_cutoff: Option<f64>,
    /// Smallest finite live column cost.
    pub minimum_active_cost: Option<f64>,
}

/// Fixed-query online metric TWED over explicit physical timestamps.
///
/// The machine retains the validated query, two query-width columns, and the
/// immediately preceding target point. It never retains the target prefix, so
/// memory is independent of the number of committed target points. Every
/// transition is iterative and transactional: invalid timestamps or bounded
/// resource failures leave the preceding prefix unchanged.
pub struct TimestampedTwedOnlineAutomaton {
    query: TimestampedSeries,
    config: MetricTimestampedTwedConfig,
    cutoff: f64,
    limits: OnlineAutomatonLimits,
    target_unit: TimestampUnit,
    target_origin: f64,
    current: Vec<f64>,
    next: Vec<f64>,
    current_active: Vec<usize>,
    next_active: Vec<usize>,
    previous_target: Option<(f64, f64)>,
    consumed_target_len: usize,
    scratch_bytes: usize,
}

impl std::fmt::Debug for TimestampedTwedOnlineAutomaton {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TimestampedTwedOnlineAutomaton")
            .field("query_len", &self.query.values().len())
            .field("target_unit", &self.target_unit)
            .field("target_origin", &self.target_origin)
            .field("cutoff", &self.cutoff)
            .field("column_width", &self.current.len())
            .field("active_positions", &self.current_active.len())
            .field("consumed_target_len", &self.consumed_target_len)
            .field("scratch_bytes", &self.scratch_bytes)
            .finish_non_exhaustive()
    }
}

impl TimestampedTwedOnlineAutomaton {
    /// Construct an online machine for one owned, validated query.
    ///
    /// `target_unit` and `target_origin` describe the complete target stream;
    /// they must exactly match the query's canonical physical frame. A finite
    /// cutoff is required so unreachable cells and binary64 overflow are both
    /// safely outside the accepted language.
    pub fn new(
        query: TimestampedSeries,
        target_unit: TimestampUnit,
        target_origin: f64,
        config: MetricTimestampedTwedConfig,
        cutoff: f64,
        limits: OnlineAutomatonLimits,
    ) -> Result<Self, TimestampedTwedError> {
        if target_unit != query.unit() {
            return Err(TimestampedTwedError::MixedUnits);
        }
        if !target_origin.is_finite() {
            return Err(TimestampedTwedError::NonFiniteTimestamp { index: None });
        }
        if target_origin != query.origin() {
            return Err(TimestampedTwedError::MixedOrigins);
        }
        if !cutoff.is_finite() || cutoff < 0.0 {
            return Err(TemporalValidationError::InvalidCutoff.into());
        }
        let query_len = query.values().len();
        if query_len > limits.max_query_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: Operand::Query,
                len: query_len,
                limit: limits.max_query_len,
            }
            .into());
        }
        let column_width = query_len.checked_add(1).ok_or({
            TimestampedTwedError::Resource(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::QueueEntries,
            })
        })?;
        if column_width > limits.max_frontier_positions {
            return Err(TimestampedTwedError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::QueueEntries,
                    limit: limits.max_frontier_positions,
                    requested: column_width,
                },
            ));
        }
        let scratch_bytes = query_len
            .checked_mul(2)
            .and_then(|slots| slots.checked_mul(size_of::<f64>()))
            .and_then(|query_bytes| {
                column_width
                    .checked_mul(2)
                    .and_then(|slots| slots.checked_mul(size_of::<f64>()))
                    .and_then(|column_bytes| query_bytes.checked_add(column_bytes))
            })
            .and_then(|bytes| {
                column_width
                    .checked_mul(2)
                    .and_then(|slots| slots.checked_mul(size_of::<usize>()))
                    .and_then(|active_bytes| bytes.checked_add(active_bytes))
            })
            .ok_or({
                TimestampedTwedError::Resource(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                })
            })?;
        if scratch_bytes > limits.max_scratch_bytes {
            return Err(TimestampedTwedError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested: scratch_bytes,
                },
            ));
        }

        let mut current = Vec::new();
        reserve_column(&mut current, column_width, scratch_bytes)?;
        current.resize(column_width, WeightedCost::TOP);
        current[0] = WeightedCost::ZERO;
        for row in 1..column_width {
            let query_index = row - 1;
            let previous_value = query_index
                .checked_sub(1)
                .and_then(|index| query.values().get(index))
                .copied()
                .unwrap_or(0.0);
            let previous_time = query_index
                .checked_sub(1)
                .and_then(|index| query.timestamps().get(index))
                .copied()
                .unwrap_or(query.origin());
            let local = delete_cost(
                query.values()[query_index],
                previous_value,
                query.timestamps()[query_index],
                previous_time,
                &config,
            );
            current[row] = WeightedCost::combine(current[row - 1], local);
            if !valid_within(current[row], cutoff) {
                current[row] = WeightedCost::TOP;
            }
        }
        let mut next = Vec::new();
        reserve_column(&mut next, column_width, scratch_bytes)?;
        next.resize(column_width, WeightedCost::TOP);
        let mut current_active = Vec::new();
        reserve_indices(&mut current_active, column_width, scratch_bytes)?;
        current_active.extend(
            current
                .iter()
                .enumerate()
                .filter_map(|(row, cost)| cost.is_finite().then_some(row)),
        );
        let mut next_active = Vec::new();
        reserve_indices(&mut next_active, column_width, scratch_bytes)?;

        Ok(Self {
            query,
            config,
            cutoff,
            limits,
            target_unit,
            target_origin,
            current,
            next,
            current_active,
            next_active,
            previous_target: None,
            consumed_target_len: 0,
            scratch_bytes,
        })
    }

    /// Observe the current committed target prefix.
    pub fn observation(&self) -> TimestampedTwedOnlineObservation {
        let distance_within_cutoff = (self.consumed_target_len > 0)
            .then(|| self.current.last().copied())
            .flatten()
            .filter(|cost| valid_within(*cost, self.cutoff))
            .map(canonical_zero);
        let minimum_active_cost = self
            .current_active
            .iter()
            .filter_map(|row| self.current.get(*row).copied())
            .min_by(f64::total_cmp)
            .map(canonical_zero);
        TimestampedTwedOnlineObservation {
            consumed_target_len: self.consumed_target_len,
            active_positions: self.current_active.len(),
            distance_within_cutoff,
            minimum_active_cost,
        }
    }

    /// Consume one finite target point with a strictly increasing timestamp.
    pub fn advance(
        &mut self,
        value: f64,
        timestamp: f64,
    ) -> Result<OnlineStepOutcome<TimestampedTwedOnlineObservation>, TimestampedTwedError> {
        if !value.is_finite() {
            return Err(TemporalValidationError::NonFiniteSample {
                operand: Operand::Candidate,
                index: self.consumed_target_len,
            }
            .into());
        }
        if !timestamp.is_finite() {
            return Err(TimestampedTwedError::NonFiniteTimestamp {
                index: Some(self.consumed_target_len),
            });
        }
        match self.previous_target {
            Some((_, previous_time)) if timestamp <= previous_time => {
                return Err(TimestampedTwedError::NonMonotoneTimestamp {
                    index: self.consumed_target_len,
                });
            }
            None if timestamp < self.target_origin => {
                return Err(TimestampedTwedError::TimestampBeforeOrigin);
            }
            Some(_) | None => {}
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
        if self.current_active.is_empty() {
            if self.limits.max_step_work_units == 0 {
                return Ok(self.incomplete(
                    IncompleteReason::BudgetExceeded {
                        resource: ResourceKind::WorkUnits,
                        limit: 0,
                        requested: 1,
                    },
                    0,
                ));
            }
            self.previous_target = Some((value, timestamp));
            self.consumed_target_len = next_depth;
            return Ok(OnlineStepOutcome::Advanced {
                value: self.observation(),
                usage: self.step_usage(1),
            });
        }

        while let Some(row) = self.next_active.pop() {
            self.next[row] = WeightedCost::TOP;
        }
        let (previous_value, previous_time) =
            self.previous_target.unwrap_or((0.0, self.target_origin));
        let target_delete = delete_cost(
            value,
            previous_value,
            timestamp,
            previous_time,
            &self.config,
        );
        let last_row = self.current.len() - 1;
        let mut work = 0usize;
        let mut seeds = NeighborSeedRows::new(&self.current_active, last_row);
        let mut next_seed = seeds.next();
        while let Some(start) = next_seed {
            let mut row = start;
            loop {
                while next_seed == Some(row) {
                    next_seed = seeds.next();
                }
                if let Err(requested) = charge_work(&mut work, self.limits.max_step_work_units) {
                    return Ok(self.incomplete(
                        IncompleteReason::BudgetExceeded {
                            resource: ResourceKind::WorkUnits,
                            limit: self.limits.max_step_work_units,
                            requested,
                        },
                        work,
                    ));
                }
                let cost = if row == 0 {
                    WeightedCost::combine(self.current[0], target_delete)
                } else {
                    let query_index = row - 1;
                    let query_previous_value = query_index
                        .checked_sub(1)
                        .and_then(|index| self.query.values().get(index))
                        .copied()
                        .unwrap_or(0.0);
                    let query_previous_time = query_index
                        .checked_sub(1)
                        .and_then(|index| self.query.timestamps().get(index))
                        .copied()
                        .unwrap_or(self.query.origin());
                    let query_delete = delete_cost(
                        self.query.values()[query_index],
                        query_previous_value,
                        self.query.timestamps()[query_index],
                        query_previous_time,
                        &self.config,
                    );
                    let pair = WeightedCost::combine(
                        self.current[row - 1],
                        match_cost(
                            self.query.values()[query_index],
                            query_previous_value,
                            self.query.timestamps()[query_index],
                            query_previous_time,
                            value,
                            previous_value,
                            timestamp,
                            previous_time,
                            &self.config,
                        ),
                    );
                    let delete_query = WeightedCost::combine(self.next[row - 1], query_delete);
                    let delete_target = WeightedCost::combine(self.current[row], target_delete);
                    pair.min(delete_query).min(delete_target)
                };
                if cost.is_nan() || cost < 0.0 || cost == f64::NEG_INFINITY {
                    return Ok(self.incomplete(IncompleteReason::NumericOverflow, work));
                }
                if valid_within(cost, self.cutoff) {
                    self.next[row] = canonical_zero(cost);
                    self.next_active.push(row);
                    if row < last_row {
                        row += 1;
                        continue;
                    }
                }
                break;
            }
        }
        debug_assert!(self.next_active.windows(2).all(|pair| pair[0] < pair[1]));

        std::mem::swap(&mut self.current, &mut self.next);
        std::mem::swap(&mut self.current_active, &mut self.next_active);
        self.previous_target = Some((value, timestamp));
        self.consumed_target_len = next_depth;
        Ok(OnlineStepOutcome::Advanced {
            value: self.observation(),
            usage: self.step_usage(work),
        })
    }

    /// Fixed logical bytes retained by this machine.
    #[inline]
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }

    fn incomplete(
        &mut self,
        reason: IncompleteReason,
        completed_work: usize,
    ) -> OnlineStepOutcome<TimestampedTwedOnlineObservation> {
        self.next.fill(WeightedCost::TOP);
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
            queue_entries: self.current.iter().filter(|cost| cost.is_finite()).count(),
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

fn reserve_column(
    storage: &mut Vec<f64>,
    width: usize,
    requested_bytes: usize,
) -> Result<(), TimestampedTwedError> {
    storage.try_reserve_exact(width).map_err(|_| {
        TimestampedTwedError::Resource(IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested: requested_bytes,
        })
    })
}

fn reserve_indices(
    storage: &mut Vec<usize>,
    width: usize,
    requested_bytes: usize,
) -> Result<(), TimestampedTwedError> {
    storage.try_reserve_exact(width).map_err(|_| {
        TimestampedTwedError::Resource(IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested: requested_bytes,
        })
    })
}
