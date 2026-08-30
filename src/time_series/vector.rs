//! Typed vector samples and metric temporal paths.
//!
//! Vector samples are never flattened into scalar sequences. A sealed ground
//! metric compares corresponding points, and discrete Frechet then aligns
//! whole points along monotone paths. Consecutive identical points are
//! canonicalized so the exposed path space is the true metric quotient rather
//! than the raw pseudometric domain.

use std::mem::size_of;
use thiserror::Error;

use super::automaton::{OnlineAutomatonLimits, OnlineStepOutcome};
use super::bounded::{
    ExactDecision, IncompleteReason, NoWitness, OperationOutcome, ResourceKind, ResourceLedger,
    ResourceLimits, ResourceUsage, TemporalValidationError,
};
use super::elastic::sparse::{charge_work, NeighborSeedRows};
use crate::cost::{BottleneckCost, CostMonoid, WeightedCost};

mod sealed {
    pub trait Sealed {}
}

/// Audited metric used to compare two equal-dimensional vector samples.
///
/// The trait is sealed: exact temporal metric claims are available only for
/// implementations whose nonnegativity, identity, symmetry, and triangle laws
/// are part of this crate's test and review surface.
pub trait GroundMetric: sealed::Sealed + Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Compute a finite nonnegative point distance, or positive infinity on
    /// floating-point overflow.
    fn distance(&self, left: &VectorSample, right: &VectorSample) -> f64;
}

/// Manhattan (`L1`) ground metric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct L1GroundMetric;

impl sealed::Sealed for L1GroundMetric {}

impl GroundMetric for L1GroundMetric {
    fn distance(&self, left: &VectorSample, right: &VectorSample) -> f64 {
        point_pairs(left, right)
            .map(|(a, b)| (a - b).abs())
            .fold(WeightedCost::ZERO, WeightedCost::combine)
    }
}

/// Euclidean (`L2`) ground metric, accumulated with stable `hypot` steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct L2GroundMetric;

impl sealed::Sealed for L2GroundMetric {}

impl GroundMetric for L2GroundMetric {
    fn distance(&self, left: &VectorSample, right: &VectorSample) -> f64 {
        point_pairs(left, right)
            .map(|(a, b)| a - b)
            .fold(0.0_f64, f64::hypot)
    }
}

/// Chebyshev (`L-infinity`) ground metric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LinfGroundMetric;

impl sealed::Sealed for LinfGroundMetric {}

impl GroundMetric for LinfGroundMetric {
    fn distance(&self, left: &VectorSample, right: &VectorSample) -> f64 {
        point_pairs(left, right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }
}

/// Failure to construct or compare typed vector temporal values.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum VectorMetricError {
    /// A point had zero coordinates.
    #[error("vector samples must have positive dimension")]
    ZeroDimension,
    /// A coordinate was NaN or infinite.
    #[error("coordinate {index} is not finite")]
    NonFiniteCoordinate {
        /// Zero-based coordinate index.
        index: usize,
    },
    /// The path contained no samples.
    #[error("a discrete Frechet metric path must be nonempty")]
    EmptyPath,
    /// Samples within or across paths had different dimensions.
    #[error("vector dimensions differ: expected {expected}, observed {observed}")]
    DimensionMismatch {
        /// Required coordinate count.
        expected: usize,
        /// Observed coordinate count.
        observed: usize,
    },
    /// Fixed online state or scratch allocation exceeded an explicit ceiling.
    #[error("vector temporal automaton resource construction failed: {0:?}")]
    Resource(IncompleteReason),
    /// Shared temporal request validation failed.
    #[error(transparent)]
    InvalidRequest(#[from] TemporalValidationError),
}

/// Exact observation of one committed vector-target prefix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorFrechetOnlineObservation {
    /// Number of target points consumed by the fixed-query machine.
    pub consumed_target_len: usize,
    /// Non-subsumed recurrence rows within the construction cutoff.
    pub active_positions: usize,
    /// Exact complete-prefix distance when it lies within the cutoff.
    pub distance_within_cutoff: Option<f64>,
    /// Smallest finite bottleneck value in the live generation.
    pub minimum_active_cost: Option<f64>,
}

/// Owned, finite, positive-dimensional vector sample.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorSample {
    coordinates: Box<[f64]>,
}

impl VectorSample {
    /// Validate and copy one vector sample.
    pub fn try_new(coordinates: &[f64], limits: ResourceLimits) -> Result<Self, VectorMetricError> {
        if coordinates.is_empty() {
            return Err(VectorMetricError::ZeroDimension);
        }
        if coordinates.len() > limits.max_dimension {
            return Err(TemporalValidationError::DimensionTooLarge {
                dimension: coordinates.len(),
                limit: limits.max_dimension,
            }
            .into());
        }
        if let Some(index) = coordinates
            .iter()
            .position(|coordinate| !coordinate.is_finite())
        {
            return Err(VectorMetricError::NonFiniteCoordinate { index });
        }
        Ok(Self {
            coordinates: coordinates.into(),
        })
    }

    /// Return the fixed coordinate count.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }

    /// Borrow the finite coordinates.
    #[inline]
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }
}

/// Canonical nonempty vector path modulo consecutive identical stutters.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorFrechetPath {
    samples: Box<[VectorSample]>,
    dimension: usize,
}

impl VectorFrechetPath {
    /// Validate a path and collapse every maximal run of identical points.
    pub fn try_new(
        samples: Vec<VectorSample>,
        limits: ResourceLimits,
    ) -> Result<Self, VectorMetricError> {
        if samples.is_empty() {
            return Err(VectorMetricError::EmptyPath);
        }
        if samples.len() > limits.max_series_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: super::bounded::Operand::Query,
                len: samples.len(),
                limit: limits.max_series_len,
            }
            .into());
        }
        let dimension = samples[0].dimension();
        let mut canonical = Vec::with_capacity(samples.len());
        for sample in samples {
            if sample.dimension() != dimension {
                return Err(VectorMetricError::DimensionMismatch {
                    expected: dimension,
                    observed: sample.dimension(),
                });
            }
            if canonical.last().is_none_or(|previous| previous != &sample) {
                canonical.push(sample);
            }
        }
        Ok(Self {
            samples: canonical.into_boxed_slice(),
            dimension,
        })
    }

    /// Validate rows directly without flattening channel boundaries.
    pub fn try_from_rows(
        rows: &[&[f64]],
        limits: ResourceLimits,
    ) -> Result<Self, VectorMetricError> {
        let mut samples = Vec::with_capacity(rows.len());
        for row in rows {
            samples.push(VectorSample::try_new(row, limits)?);
        }
        Self::try_new(samples, limits)
    }

    /// Return the fixed coordinate count of every point.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Borrow the stutter-free vector samples.
    #[inline]
    pub fn samples(&self) -> &[VectorSample] {
        &self.samples
    }
}

/// Discrete Frechet metric over vector paths and an audited ground metric.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorFrechetMetric<M: GroundMetric> {
    ground: M,
}

impl<M: GroundMetric> VectorFrechetMetric<M> {
    /// Construct a vector-path metric from an audited ground metric.
    #[inline]
    pub fn new(ground: M) -> Self {
        Self { ground }
    }

    /// Borrow the point metric.
    #[inline]
    pub fn ground_metric(&self) -> &M {
        &self.ground
    }

    /// Compute exact vector discrete Frechet under deterministic hard limits.
    pub fn distance_bounded(
        &self,
        left: &VectorFrechetPath,
        right: &VectorFrechetPath,
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, VectorMetricError> {
        if left.dimension != right.dimension {
            return Err(VectorMetricError::DimensionMismatch {
                expected: left.dimension,
                observed: right.dimension,
            });
        }
        if cutoff.is_nan() || cutoff < 0.0 {
            return Err(TemporalValidationError::InvalidCutoff.into());
        }
        if left.samples.len() > limits.max_series_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: super::bounded::Operand::Query,
                len: left.samples.len(),
                limit: limits.max_series_len,
            }
            .into());
        }
        if right.samples.len() > limits.max_series_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: super::bounded::Operand::Candidate,
                len: right.samples.len(),
                limit: limits.max_series_len,
            }
            .into());
        }

        let mut ledger = ResourceLedger::new(limits);
        let cells = left.samples.len().checked_mul(right.samples.len());
        let coordinate_work = cells.and_then(|count| count.checked_mul(left.dimension));
        let scratch = left
            .samples
            .len()
            .min(right.samples.len())
            .checked_mul(2)
            .and_then(|slots| slots.checked_mul(std::mem::size_of::<f64>()));
        let (Some(cells), Some(coordinate_work), Some(scratch)) = (cells, coordinate_work, scratch)
        else {
            return Ok(incomplete(
                ledger,
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                },
            ));
        };
        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::DpCells, cells),
            (ResourceKind::WorkUnits, coordinate_work),
            (ResourceKind::ScratchBytes, scratch),
        ]) {
            return Ok(incomplete(ledger, reason));
        }

        let exact = vector_frechet_with_cutoff(&self.ground, left, right, cutoff);
        let usage = ledger.usage();
        Ok(match exact {
            Some(distance) if distance.is_finite() => OperationOutcome::Complete {
                value: ExactDecision::WithinCutoff {
                    distance,
                    witness: NoWitness,
                },
                usage,
            },
            Some(_) | None if cutoff.is_finite() => OperationOutcome::Complete {
                value: ExactDecision::AboveCutoff,
                usage,
            },
            Some(_) | None => OperationOutcome::Incomplete {
                partial: None,
                reason: IncompleteReason::NumericOverflow,
                continuation: None,
                usage,
            },
        })
    }
}

fn vector_frechet_with_cutoff<M: GroundMetric>(
    ground: &M,
    left: &VectorFrechetPath,
    right: &VectorFrechetPath,
    cutoff: f64,
) -> Option<f64> {
    if right.samples.len() > left.samples.len() {
        return vector_frechet_with_cutoff(ground, right, left, cutoff);
    }
    let mut previous = vec![BottleneckCost::TOP; right.samples.len()];
    let mut current = vec![BottleneckCost::TOP; right.samples.len()];
    previous[0] = ground.distance(&left.samples[0], &right.samples[0]);
    for column in 1..right.samples.len() {
        previous[column] = BottleneckCost::combine(
            previous[column - 1],
            ground.distance(&left.samples[0], &right.samples[column]),
        );
    }
    for left_sample in &left.samples[1..] {
        current[0] =
            BottleneckCost::combine(previous[0], ground.distance(left_sample, &right.samples[0]));
        let mut row_min = current[0];
        for column in 1..right.samples.len() {
            let predecessor = previous[column - 1]
                .min(previous[column])
                .min(current[column - 1]);
            current[column] = BottleneckCost::combine(
                predecessor,
                ground.distance(left_sample, &right.samples[column]),
            );
            row_min = row_min.min(current[column]);
        }
        if !BottleneckCost::within(row_min, cutoff) {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let exact = previous[right.samples.len() - 1];
    BottleneckCost::within(exact, cutoff).then_some(exact)
}

/// Fixed-query, stack-safe online vector discrete-Fréchet automaton.
///
/// Whole vector points are labels: coordinates are never flattened and an
/// alignment transition cannot cross a channel boundary. The machine keeps
/// two query-width generations and sorted active-row IDs, consumes an unknown
/// number of target points, and retains no target prefix.
pub struct VectorFrechetOnlineAutomaton<M: GroundMetric> {
    query: VectorFrechetPath,
    ground: M,
    cutoff: f64,
    limits: OnlineAutomatonLimits,
    current: Vec<f64>,
    next: Vec<f64>,
    current_active: Vec<usize>,
    next_active: Vec<usize>,
    consumed_target_len: usize,
    scratch_bytes: usize,
}

impl<M: GroundMetric> std::fmt::Debug for VectorFrechetOnlineAutomaton<M> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VectorFrechetOnlineAutomaton")
            .field("query_len", &self.query.samples.len())
            .field("dimension", &self.query.dimension)
            .field("cutoff", &self.cutoff)
            .field("active_positions", &self.current_active.len())
            .field("consumed_target_len", &self.consumed_target_len)
            .field("scratch_bytes", &self.scratch_bytes)
            .finish_non_exhaustive()
    }
}

impl<M: GroundMetric> VectorFrechetOnlineAutomaton<M> {
    /// Construct a bounded online machine for one canonical fixed query.
    pub fn new(
        query: VectorFrechetPath,
        ground: M,
        cutoff: f64,
        limits: OnlineAutomatonLimits,
    ) -> Result<Self, VectorMetricError> {
        if !cutoff.is_finite() || cutoff < 0.0 {
            return Err(TemporalValidationError::InvalidCutoff.into());
        }
        if query.samples.len() > limits.max_query_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: super::bounded::Operand::Query,
                len: query.samples.len(),
                limit: limits.max_query_len,
            }
            .into());
        }
        let width = query.samples.len();
        if width > limits.max_frontier_positions {
            return Err(VectorMetricError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::QueueEntries,
                    limit: limits.max_frontier_positions,
                    requested: width,
                },
            ));
        }
        let query_bytes = width
            .checked_mul(query.dimension)
            .and_then(|coordinates| coordinates.checked_mul(size_of::<f64>()));
        let generation_bytes = width
            .checked_mul(2)
            .and_then(|slots| slots.checked_mul(size_of::<f64>()))
            .and_then(|costs| {
                width
                    .checked_mul(2)
                    .and_then(|slots| slots.checked_mul(size_of::<usize>()))
                    .and_then(|active| costs.checked_add(active))
            });
        let scratch_bytes = query_bytes
            .and_then(|query| generation_bytes.and_then(|state| query.checked_add(state)))
            .ok_or(VectorMetricError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        if scratch_bytes > limits.max_scratch_bytes {
            return Err(VectorMetricError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested: scratch_bytes,
                },
            ));
        }

        let mut current = Vec::new();
        reserve_vector_storage(&mut current, width, scratch_bytes)?;
        current.resize(width, BottleneckCost::TOP);
        let mut next = Vec::new();
        reserve_vector_storage(&mut next, width, scratch_bytes)?;
        next.resize(width, BottleneckCost::TOP);
        let mut current_active = Vec::new();
        reserve_vector_indices(&mut current_active, width, scratch_bytes)?;
        let mut next_active = Vec::new();
        reserve_vector_indices(&mut next_active, width, scratch_bytes)?;

        Ok(Self {
            query,
            ground,
            cutoff,
            limits,
            current,
            next,
            current_active,
            next_active,
            consumed_target_len: 0,
            scratch_bytes,
        })
    }

    /// Observe the already committed target prefix.
    pub fn observation(&self) -> VectorFrechetOnlineObservation {
        let distance_within_cutoff = (self.consumed_target_len > 0)
            .then(|| self.current.last().copied())
            .flatten()
            .filter(|cost| valid_online_cost(*cost, self.cutoff));
        let minimum_active_cost = self
            .current_active
            .iter()
            .filter_map(|row| self.current.get(*row).copied())
            .min_by(f64::total_cmp);
        VectorFrechetOnlineObservation {
            consumed_target_len: self.consumed_target_len,
            active_positions: self.current_active.len(),
            distance_within_cutoff,
            minimum_active_cost,
        }
    }

    /// Consume one finite point of the fixed vector dimension.
    pub fn advance(
        &mut self,
        target: &VectorSample,
    ) -> Result<OnlineStepOutcome<VectorFrechetOnlineObservation>, VectorMetricError> {
        if target.dimension() != self.query.dimension {
            return Err(VectorMetricError::DimensionMismatch {
                expected: self.query.dimension,
                observed: target.dimension(),
            });
        }
        let next_depth =
            self.consumed_target_len
                .checked_add(1)
                .ok_or(VectorMetricError::Resource(
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::SeriesLength,
                    },
                ))?;

        while let Some(row) = self.next_active.pop() {
            self.next[row] = BottleneckCost::TOP;
        }
        let mut work = 0usize;
        if next_depth == 1 {
            for row in 0..self.query.samples.len() {
                if let Err(requested) = charge_vector_work(
                    &mut work,
                    self.query.dimension,
                    self.limits.max_step_work_units,
                ) {
                    return Ok(self.incomplete_work(requested, work));
                }
                let predecessor = if row == 0 {
                    BottleneckCost::ZERO
                } else {
                    self.next[row - 1]
                };
                let cost = BottleneckCost::combine(
                    predecessor,
                    self.ground.distance(&self.query.samples[row], target),
                );
                if !valid_online_cost(cost, self.cutoff) {
                    break;
                }
                self.next[row] = cost;
                self.next_active.push(row);
            }
        } else if self.current_active.is_empty() {
            if charge_work(&mut work, self.limits.max_step_work_units).is_err() {
                return Ok(self.incomplete_work(1, 0));
            }
        } else {
            let last_row = self.query.samples.len() - 1;
            let mut seeds = NeighborSeedRows::new(&self.current_active, last_row);
            let mut next_seed = seeds.next();
            while let Some(start) = next_seed {
                let mut row = start;
                loop {
                    while next_seed == Some(row) {
                        next_seed = seeds.next();
                    }
                    if let Err(requested) = charge_vector_work(
                        &mut work,
                        self.query.dimension,
                        self.limits.max_step_work_units,
                    ) {
                        return Ok(self.incomplete_work(requested, work));
                    }
                    let diagonal = row
                        .checked_sub(1)
                        .and_then(|index| self.current.get(index))
                        .copied()
                        .unwrap_or(BottleneckCost::TOP);
                    let same = self.current[row];
                    let vertical = row
                        .checked_sub(1)
                        .and_then(|index| self.next.get(index))
                        .copied()
                        .unwrap_or(BottleneckCost::TOP);
                    let cost = BottleneckCost::combine(
                        diagonal.min(same).min(vertical),
                        self.ground.distance(&self.query.samples[row], target),
                    );
                    if !cost.is_finite() && cost != BottleneckCost::TOP {
                        return Ok(self.incomplete_numeric(work));
                    }
                    if valid_online_cost(cost, self.cutoff) {
                        self.next[row] = cost;
                        self.next_active.push(row);
                        if row < last_row {
                            row += 1;
                            continue;
                        }
                    }
                    break;
                }
            }
        }

        std::mem::swap(&mut self.current, &mut self.next);
        std::mem::swap(&mut self.current_active, &mut self.next_active);
        self.consumed_target_len = next_depth;
        Ok(OnlineStepOutcome::Advanced {
            value: self.observation(),
            usage: self.step_usage(work),
        })
    }

    /// Fixed retained logical bytes; independent of target-prefix length.
    #[inline]
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }

    fn incomplete_work(
        &mut self,
        requested: usize,
        completed: usize,
    ) -> OnlineStepOutcome<VectorFrechetOnlineObservation> {
        self.clear_speculative();
        OnlineStepOutcome::Incomplete {
            reason: IncompleteReason::BudgetExceeded {
                resource: ResourceKind::WorkUnits,
                limit: self.limits.max_step_work_units,
                requested,
            },
            usage: self.step_usage(completed),
        }
    }

    fn incomplete_numeric(
        &mut self,
        completed: usize,
    ) -> OnlineStepOutcome<VectorFrechetOnlineObservation> {
        self.clear_speculative();
        OnlineStepOutcome::Incomplete {
            reason: IncompleteReason::NumericOverflow,
            usage: self.step_usage(completed),
        }
    }

    fn clear_speculative(&mut self) {
        while let Some(row) = self.next_active.pop() {
            self.next[row] = BottleneckCost::TOP;
        }
    }

    fn step_usage(&self, work: usize) -> ResourceUsage {
        ResourceUsage {
            dp_cells: work.checked_div(self.query.dimension).unwrap_or(0),
            work_units: work,
            scratch_bytes: self.scratch_bytes,
            queue_entries: self.current_active.len(),
            ..ResourceUsage::default()
        }
    }
}

#[inline]
fn charge_vector_work(work: &mut usize, amount: usize, limit: usize) -> Result<(), usize> {
    let requested = work.checked_add(amount).unwrap_or(usize::MAX);
    if requested > limit {
        Err(requested)
    } else {
        *work = requested;
        Ok(())
    }
}

#[inline]
fn valid_online_cost(cost: f64, cutoff: f64) -> bool {
    cost.is_finite() && cost >= 0.0 && cost <= cutoff
}

fn reserve_vector_storage(
    storage: &mut Vec<f64>,
    width: usize,
    requested: usize,
) -> Result<(), VectorMetricError> {
    storage.try_reserve_exact(width).map_err(|_| {
        VectorMetricError::Resource(IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        })
    })
}

fn reserve_vector_indices(
    storage: &mut Vec<usize>,
    width: usize,
    requested: usize,
) -> Result<(), VectorMetricError> {
    storage.try_reserve_exact(width).map_err(|_| {
        VectorMetricError::Resource(IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        })
    })
}

fn point_pairs<'a>(
    left: &'a VectorSample,
    right: &'a VectorSample,
) -> impl Iterator<Item = (f64, f64)> + 'a {
    debug_assert_eq!(left.dimension(), right.dimension());
    left.coordinates
        .iter()
        .copied()
        .zip(right.coordinates.iter().copied())
}

fn incomplete(ledger: ResourceLedger, reason: IncompleteReason) -> OperationOutcome<ExactDecision> {
    OperationOutcome::Incomplete {
        partial: None,
        reason,
        continuation: None,
        usage: ledger.usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact(outcome: OperationOutcome<ExactDecision>) -> f64 {
        match outcome {
            OperationOutcome::Complete {
                value: ExactDecision::WithinCutoff { distance, .. },
                ..
            } => distance,
            other => panic!("expected exact completion, got {other:?}"),
        }
    }

    #[test]
    fn audited_ground_metrics_have_expected_values() {
        let limits = ResourceLimits::default();
        let left = VectorSample::try_new(&[0.0, 0.0], limits).unwrap();
        let right = VectorSample::try_new(&[3.0, 4.0], limits).unwrap();
        assert_eq!(L1GroundMetric.distance(&left, &right), 7.0);
        assert_eq!(L2GroundMetric.distance(&left, &right), 5.0);
        assert_eq!(LinfGroundMetric.distance(&left, &right), 4.0);
    }

    #[test]
    fn vector_frechet_uses_points_and_stutter_quotient() {
        let limits = ResourceLimits::default();
        let left =
            VectorFrechetPath::try_from_rows(&[&[0.0, 0.0], &[0.0, 0.0], &[1.0, 1.0]], limits)
                .unwrap();
        let right = VectorFrechetPath::try_from_rows(&[&[0.0, 0.0], &[1.0, 1.0]], limits).unwrap();
        assert_eq!(left, right);
        let distance = VectorFrechetMetric::new(L2GroundMetric)
            .distance_bounded(&left, &right, f64::INFINITY, limits)
            .unwrap();
        assert_eq!(exact(distance), 0.0);
    }

    #[test]
    fn dimension_mismatch_is_never_flattened() {
        let limits = ResourceLimits::default();
        let left = VectorFrechetPath::try_from_rows(&[&[0.0, 1.0]], limits).unwrap();
        let right = VectorFrechetPath::try_from_rows(&[&[0.0, 1.0, 2.0]], limits).unwrap();
        assert!(matches!(
            VectorFrechetMetric::new(L1GroundMetric).distance_bounded(
                &left,
                &right,
                f64::INFINITY,
                limits
            ),
            Err(VectorMetricError::DimensionMismatch { .. })
        ));
    }
}
