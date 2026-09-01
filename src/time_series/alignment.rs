//! Replayable, explicitly bounded temporal alignment certificates.
//!
//! Witness extraction is an opt-in batch operation, separate from the stable
//! online automata. It materializes a full trace matrix only after reserving
//! all DP, work, scratch, and maximum witness storage against hard limits.
//! Traceback is an iterative monotone walk and therefore stack-safe.

use std::fmt;
use std::mem::size_of;

use super::bounded::{
    ExactDecision, IncompleteReason, Operand, OperationOutcome, ResourceKind, ResourceLedger,
    ResourceLimits, TemporalValidationError,
};
use super::kernels::{DtwConfig, ErpConfig, FrechetConfig, MetricTwedConfig, TwedConfig};
use super::msm::MsmConfig;
use super::timestamped_twed::{
    delete_cost as timestamped_delete_cost, match_cost as timestamped_match_cost,
    MetricTimestampedTwedConfig, TimestampedSeries, TimestampedTwedError,
};

/// Stable schema version of newly emitted temporal alignment witnesses.
pub const TEMPORAL_ALIGNMENT_WITNESS_VERSION: u16 = 1;

/// Stable kernel discriminator carried by a replayable temporal witness.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TemporalAlignmentKind {
    /// Edit distance with Real Penalty.
    Erp = 1,
    /// Scalar TWED on an implicit unit-spaced grid.
    UnitGridTwed = 2,
    /// Symmetrically banded, explicitly non-metric DTW.
    BandedDtw = 3,
    /// Scalar discrete Fréchet coupling.
    DiscreteFrechet = 4,
    /// Metric TWED over explicit physical timestamps.
    TimestampedTwed = 5,
}

/// One monotone operation in a generic temporal alignment.
///
/// The numeric representation is stable for certificate serialization. Each
/// operation records the destination endpoints after it commits.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TemporalAlignmentOperation {
    /// Consume one sample from each operand.
    Align = 1,
    /// Consume one query sample while retaining the candidate endpoint.
    AdvanceQuery = 2,
    /// Consume one candidate sample while retaining the query endpoint.
    AdvanceCandidate = 3,
}

/// Serialized operation record in a temporal alignment certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TemporalAlignmentStep {
    operation: TemporalAlignmentOperation,
    query_endpoint: Option<u64>,
    candidate_endpoint: Option<u64>,
    local_cost_bits: u64,
}

impl TemporalAlignmentStep {
    /// Reconstruct an untrusted serialized step for validation by replay.
    #[inline]
    pub const fn from_raw_parts(
        operation: TemporalAlignmentOperation,
        query_endpoint: Option<u64>,
        candidate_endpoint: Option<u64>,
        local_cost_bits: u64,
    ) -> Self {
        Self {
            operation,
            query_endpoint,
            candidate_endpoint,
            local_cost_bits,
        }
    }

    /// Operation tag.
    #[inline]
    pub const fn operation(&self) -> TemporalAlignmentOperation {
        self.operation
    }

    /// Zero-based query endpoint after this operation, or `None` at its empty prefix.
    #[inline]
    pub const fn query_endpoint(&self) -> Option<u64> {
        self.query_endpoint
    }

    /// Zero-based candidate endpoint after this operation, or `None` at its empty prefix.
    #[inline]
    pub const fn candidate_endpoint(&self) -> Option<u64> {
        self.candidate_endpoint
    }

    /// Exact IEEE-754 bits of the operation's recomputed local cost.
    #[inline]
    pub const fn local_cost_bits(&self) -> u64 {
        self.local_cost_bits
    }

    /// Decode the recorded local cost.
    #[inline]
    pub fn local_cost(&self) -> f64 {
        f64::from_bits(self.local_cost_bits)
    }
}

/// Versioned, deterministic temporal alignment certificate.
///
/// `from_parts` intentionally accepts untrusted data. A certificate becomes
/// authoritative only after the kernel-specific replay method validates its
/// version, kernel tag, operations, endpoints, local costs, and final endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalAlignmentWitness {
    version: u16,
    kind: TemporalAlignmentKind,
    steps: Vec<TemporalAlignmentStep>,
}

impl TemporalAlignmentWitness {
    /// Reconstruct an untrusted serialized witness for fail-closed replay.
    #[inline]
    pub fn from_parts(
        version: u16,
        kind: TemporalAlignmentKind,
        steps: Vec<TemporalAlignmentStep>,
    ) -> Self {
        Self {
            version,
            kind,
            steps,
        }
    }

    /// Stable schema version.
    #[inline]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Kernel whose recurrence produced this witness.
    #[inline]
    pub const fn kind(&self) -> TemporalAlignmentKind {
        self.kind
    }

    /// Ordered forward replay steps.
    #[inline]
    pub fn steps(&self) -> &[TemporalAlignmentStep] {
        &self.steps
    }

    /// Number of replay operations.
    #[inline]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the certificate contains no operations.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Reason a generic temporal certificate failed replay validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TemporalWitnessReplayError {
    /// The serialized schema version is unsupported.
    UnsupportedVersion {
        /// Version found in the untrusted witness.
        found: u16,
    },
    /// The witness was replayed against a different kernel family.
    KernelMismatch {
        /// Kernel required by the replay entry point.
        expected: TemporalAlignmentKind,
        /// Kernel tag carried by the witness.
        found: TemporalAlignmentKind,
    },
    /// Configuration or operands are outside the kernel's lawful domain.
    InvalidDomain,
    /// The operation cannot legally advance from the current alignment cell.
    MalformedOperation {
        /// First invalid operation.
        step_index: usize,
    },
    /// Recorded endpoints differ from the endpoints implied by the operation path.
    MalformedEndpoint {
        /// First invalid operation, or `steps.len()` for an incomplete path.
        step_index: usize,
    },
    /// The stored local cost does not equal the cost recomputed from exact inputs.
    LocalCostMismatch {
        /// First operation with a forged or stale cost.
        step_index: usize,
    },
    /// Recomputed finite local costs overflowed binary64 accumulation.
    NumericOverflow,
}

impl fmt::Display for TemporalWitnessReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { found } => {
                write!(formatter, "unsupported temporal witness version {found}")
            }
            Self::KernelMismatch { expected, found } => {
                write!(
                    formatter,
                    "temporal witness kernel {found:?} is not {expected:?}"
                )
            }
            Self::InvalidDomain => formatter.write_str("temporal witness domain is invalid"),
            Self::MalformedOperation { step_index } => {
                write!(
                    formatter,
                    "malformed temporal operation at step {step_index}"
                )
            }
            Self::MalformedEndpoint { step_index } => {
                write!(
                    formatter,
                    "malformed temporal endpoint at step {step_index}"
                )
            }
            Self::LocalCostMismatch { step_index } => {
                write!(
                    formatter,
                    "temporal local-cost mismatch at step {step_index}"
                )
            }
            Self::NumericOverflow => formatter.write_str("temporal witness replay overflowed"),
        }
    }
}

impl std::error::Error for TemporalWitnessReplayError {}

/// Stable schema version of newly emitted MSM alignment witnesses.
pub const MSM_ALIGNMENT_WITNESS_VERSION: u16 = 1;

/// One operation in a forward MSM alignment path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MsmAlignmentStep {
    /// Consume one sample from each operand and pay their absolute difference.
    Move,
    /// Consume one query sample while holding the current candidate sample.
    Merge,
    /// Consume one candidate sample while holding the current query sample.
    Split,
}

/// Compact, replayable certificate for one exact MSM distance.
///
/// The certificate intentionally stores no caller-supplied costs. [`Self::replay`]
/// recomputes every local operation cost from the original operands and
/// normalized MSM configuration, rejects a malformed path, and returns the
/// resulting total only when both operands are consumed exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MsmAlignmentWitness {
    version: u16,
    steps: Vec<MsmAlignmentStep>,
}

impl Default for MsmAlignmentWitness {
    fn default() -> Self {
        Self {
            version: MSM_ALIGNMENT_WITNESS_VERSION,
            steps: Vec::new(),
        }
    }
}

impl MsmAlignmentWitness {
    /// Reconstruct an untrusted serialized MSM witness for validation by replay.
    #[inline]
    pub fn from_parts(version: u16, steps: Vec<MsmAlignmentStep>) -> Self {
        Self { version, steps }
    }

    /// Stable schema version.
    #[inline]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Ordered operation tags in forward replay order.
    #[inline]
    pub fn steps(&self) -> &[MsmAlignmentStep] {
        &self.steps
    }

    /// Number of certified operations.
    #[inline]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether this is the unique empty/empty certificate.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Replay the certificate against exact operands and configuration.
    pub fn replay(
        &self,
        query: &[f64],
        candidate: &[f64],
        config: &MsmConfig,
    ) -> Result<f64, MsmWitnessReplayError> {
        if self.version != MSM_ALIGNMENT_WITNESS_VERSION {
            return Err(MsmWitnessReplayError::UnsupportedVersion {
                found: self.version,
            });
        }
        if !config.c.is_finite()
            || config.c < 0.0
            || query
                .iter()
                .chain(candidate)
                .any(|value| !value.is_finite())
        {
            return Err(MsmWitnessReplayError::InvalidDomain);
        }
        if query.is_empty() || candidate.is_empty() {
            return if query.is_empty() && candidate.is_empty() && self.steps.is_empty() {
                Ok(0.0)
            } else {
                Err(MsmWitnessReplayError::MalformedPath { step_index: 0 })
            };
        }

        let mut query_index = 0_usize;
        let mut candidate_index = 0_usize;
        let mut cost = 0.0;
        for (step_index, step) in self.steps.iter().copied().enumerate() {
            let edge = match step {
                MsmAlignmentStep::Move if step_index == 0 => (query[0] - candidate[0]).abs(),
                MsmAlignmentStep::Move => {
                    query_index = query_index
                        .checked_add(1)
                        .ok_or(MsmWitnessReplayError::MalformedPath { step_index })?;
                    candidate_index = candidate_index
                        .checked_add(1)
                        .ok_or(MsmWitnessReplayError::MalformedPath { step_index })?;
                    let (Some(query_value), Some(candidate_value)) =
                        (query.get(query_index), candidate.get(candidate_index))
                    else {
                        return Err(MsmWitnessReplayError::MalformedPath { step_index });
                    };
                    (*query_value - *candidate_value).abs()
                }
                MsmAlignmentStep::Merge if step_index > 0 => {
                    let previous_query = query_index;
                    query_index = query_index
                        .checked_add(1)
                        .ok_or(MsmWitnessReplayError::MalformedPath { step_index })?;
                    let (Some(query_value), Some(previous_value), Some(candidate_value)) = (
                        query.get(query_index),
                        query.get(previous_query),
                        candidate.get(candidate_index),
                    ) else {
                        return Err(MsmWitnessReplayError::MalformedPath { step_index });
                    };
                    config.c_func(*query_value, *previous_value, *candidate_value)
                }
                MsmAlignmentStep::Split if step_index > 0 => {
                    let previous_candidate = candidate_index;
                    candidate_index = candidate_index
                        .checked_add(1)
                        .ok_or(MsmWitnessReplayError::MalformedPath { step_index })?;
                    let (Some(candidate_value), Some(query_value), Some(previous_value)) = (
                        candidate.get(candidate_index),
                        query.get(query_index),
                        candidate.get(previous_candidate),
                    ) else {
                        return Err(MsmWitnessReplayError::MalformedPath { step_index });
                    };
                    config.c_func(*candidate_value, *query_value, *previous_value)
                }
                _ => return Err(MsmWitnessReplayError::MalformedPath { step_index }),
            };
            cost += edge;
            if !cost.is_finite() {
                return Err(MsmWitnessReplayError::NumericOverflow);
            }
        }
        if self.steps.is_empty()
            || query_index.checked_add(1) != Some(query.len())
            || candidate_index.checked_add(1) != Some(candidate.len())
        {
            return Err(MsmWitnessReplayError::MalformedPath {
                step_index: self.steps.len(),
            });
        }
        Ok(cost)
    }
}

/// Reason an MSM certificate could not be replayed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsmWitnessReplayError {
    /// The serialized schema version is unsupported.
    UnsupportedVersion {
        /// Version found in the untrusted witness.
        found: u16,
    },
    /// Configuration or operand samples are outside MSM's finite raw domain.
    InvalidDomain,
    /// An operation did not monotonically consume a valid operand position.
    MalformedPath {
        /// First invalid operation, or `steps.len()` for an incomplete path.
        step_index: usize,
    },
    /// Recomputed finite edge costs overflowed binary64 accumulation.
    NumericOverflow,
}

impl fmt::Display for MsmWitnessReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion { found } => {
                write!(formatter, "unsupported MSM witness version {found}")
            }
            Self::InvalidDomain => formatter.write_str("MSM witness domain is invalid"),
            Self::MalformedPath { step_index } => {
                write!(formatter, "malformed MSM witness at step {step_index}")
            }
            Self::NumericOverflow => formatter.write_str("MSM witness replay overflowed"),
        }
    }
}

impl std::error::Error for MsmWitnessReplayError {}

impl MsmConfig {
    /// Compute an exact cutoff decision with a replayable MSM alignment.
    ///
    /// Unlike stable score-only and online surfaces, witness extraction must
    /// retain predecessor information. This method therefore preflights a full
    /// `(m+1) × (n+1)` trace matrix and a worst-case `m+n-1` operation vector.
    /// It never falls back to an unbounded allocation.
    pub fn distance_with_alignment_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision<MsmAlignmentWitness>>, TemporalValidationError> {
        if !self.c.is_finite() || self.c < 0.0 {
            return Err(TemporalValidationError::InvalidConfiguration(
                "MSM split/merge cost must be finite and nonnegative",
            ));
        }
        if cutoff.is_nan() || cutoff < 0.0 || (cutoff.is_infinite() && cutoff.is_sign_negative()) {
            return Err(TemporalValidationError::InvalidCutoff);
        }
        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, query)?;
        ledger.validate_finite_series(Operand::Candidate, candidate)?;

        if query.is_empty() || candidate.is_empty() {
            let value = if query.is_empty() && candidate.is_empty() {
                ExactDecision::WithinCutoff {
                    distance: 0.0,
                    witness: MsmAlignmentWitness::default(),
                }
            } else {
                ExactDecision::NoFiniteAlignment
            };
            return Ok(OperationOutcome::Complete {
                value,
                usage: ledger.usage(),
            });
        }

        let Some(rows) = query.len().checked_add(1) else {
            return Ok(arithmetic_incomplete(ledger, ResourceKind::DpCells));
        };
        let Some(columns) = candidate.len().checked_add(1) else {
            return Ok(arithmetic_incomplete(ledger, ResourceKind::DpCells));
        };
        let Some(matrix_cells) = rows.checked_mul(columns) else {
            return Ok(arithmetic_incomplete(ledger, ResourceKind::DpCells));
        };
        let Some(matrix_bytes) = matrix_cells.checked_mul(size_of::<f64>()) else {
            return Ok(arithmetic_incomplete(ledger, ResourceKind::ScratchBytes));
        };
        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::DpCells, matrix_cells),
            (ResourceKind::WorkUnits, matrix_cells),
        ]) {
            return Ok(incomplete(ledger, reason));
        }
        if let Err(reason) = ledger.observe_peak(ResourceKind::ScratchBytes, matrix_bytes) {
            return Ok(incomplete(ledger, reason));
        }

        let mut matrix = Vec::new();
        if matrix.try_reserve_exact(matrix_cells).is_err() {
            return Ok(incomplete(
                ledger,
                IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: matrix_bytes,
                },
            ));
        }
        matrix.resize(matrix_cells, f64::INFINITY);
        let cell = |row: usize, column: usize| row * columns + column;
        matrix[cell(1, 1)] = (query[0] - candidate[0]).abs();
        for row in 2..=query.len() {
            matrix[cell(row, 1)] = matrix[cell(row - 1, 1)]
                + self.c_func(query[row - 1], query[row - 2], candidate[0]);
        }
        for column in 2..=candidate.len() {
            matrix[cell(1, column)] = matrix[cell(1, column - 1)]
                + self.c_func(candidate[column - 1], query[0], candidate[column - 2]);
        }
        for row in 2..=query.len() {
            for column in 2..=candidate.len() {
                let moved = matrix[cell(row - 1, column - 1)]
                    + (query[row - 1] - candidate[column - 1]).abs();
                let merged = matrix[cell(row - 1, column)]
                    + self.c_func(query[row - 1], query[row - 2], candidate[column - 1]);
                let split = matrix[cell(row, column - 1)]
                    + self.c_func(candidate[column - 1], query[row - 1], candidate[column - 2]);
                matrix[cell(row, column)] = moved.min(merged).min(split);
            }
        }
        let distance = matrix[cell(query.len(), candidate.len())];
        if !distance.is_finite() {
            return Ok(incomplete(ledger, IncompleteReason::NumericOverflow));
        }
        if distance > cutoff {
            return Ok(OperationOutcome::Complete {
                value: ExactDecision::AboveCutoff,
                usage: ledger.usage(),
            });
        }

        let Some(max_steps) = query
            .len()
            .checked_add(candidate.len())
            .and_then(|sum| sum.checked_sub(1))
        else {
            return Ok(arithmetic_incomplete(ledger, ResourceKind::WitnessBytes));
        };
        let Some(witness_bytes) = max_steps
            .checked_mul(size_of::<MsmAlignmentStep>())
            .and_then(|bytes| bytes.checked_add(size_of::<MsmAlignmentWitness>()))
        else {
            return Ok(arithmetic_incomplete(ledger, ResourceKind::WitnessBytes));
        };
        if let Err(reason) = ledger.observe_peak(ResourceKind::WitnessBytes, witness_bytes) {
            return Ok(incomplete(ledger, reason));
        }
        let mut reversed = Vec::new();
        if reversed.try_reserve_exact(max_steps).is_err() {
            return Ok(incomplete(
                ledger,
                IncompleteReason::AllocationFailed {
                    resource: ResourceKind::WitnessBytes,
                    requested: witness_bytes,
                },
            ));
        }
        let mut row = query.len();
        let mut column = candidate.len();
        while row > 1 || column > 1 {
            let current = matrix[cell(row, column)];
            if row > 1 && column > 1 {
                let moved = matrix[cell(row - 1, column - 1)]
                    + (query[row - 1] - candidate[column - 1]).abs();
                if current.to_bits() == moved.to_bits() {
                    reversed.push(MsmAlignmentStep::Move);
                    row -= 1;
                    column -= 1;
                    continue;
                }
            }
            if row > 1 {
                let merged = matrix[cell(row - 1, column)]
                    + self.c_func(query[row - 1], query[row - 2], candidate[column - 1]);
                if current.to_bits() == merged.to_bits() {
                    reversed.push(MsmAlignmentStep::Merge);
                    row -= 1;
                    continue;
                }
            }
            if column > 1 {
                let split = matrix[cell(row, column - 1)]
                    + self.c_func(candidate[column - 1], query[row - 1], candidate[column - 2]);
                if current.to_bits() == split.to_bits() {
                    reversed.push(MsmAlignmentStep::Split);
                    column -= 1;
                    continue;
                }
            }
            return Ok(incomplete(ledger, IncompleteReason::InvalidStoredData));
        }
        reversed.push(MsmAlignmentStep::Move);
        reversed.reverse();
        let witness = MsmAlignmentWitness {
            version: MSM_ALIGNMENT_WITNESS_VERSION,
            steps: reversed,
        };
        if witness
            .replay(query, candidate, self)
            .ok()
            .map(f64::to_bits)
            != Some(distance.to_bits())
        {
            return Ok(incomplete(ledger, IncompleteReason::InvalidStoredData));
        }
        Ok(OperationOutcome::Complete {
            value: ExactDecision::WithinCutoff { distance, witness },
            usage: ledger.usage(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Accumulation {
    Additive,
    Bottleneck,
}

impl Accumulation {
    #[inline]
    fn extend(self, predecessor: f64, local: f64) -> f64 {
        match self {
            Self::Additive => {
                if predecessor == f64::INFINITY || local == f64::INFINITY {
                    f64::INFINITY
                } else {
                    predecessor + local
                }
            }
            Self::Bottleneck => {
                if predecessor == f64::INFINITY || local == f64::INFINITY {
                    f64::INFINITY
                } else {
                    predecessor.max(local)
                }
            }
        }
    }
}

trait TemporalAlignmentModel {
    fn kind(&self) -> TemporalAlignmentKind;

    fn accumulation(&self) -> Accumulation;

    fn path_exists(&self, query_len: usize, candidate_len: usize) -> bool;

    fn trace_band(&self) -> Option<usize> {
        None
    }

    fn local_cost(
        &self,
        operation: TemporalAlignmentOperation,
        query_consumed: usize,
        candidate_consumed: usize,
    ) -> Option<f64>;

    fn reported_cost(&self, accumulated: f64) -> f64 {
        accumulated
    }
}

struct ErpAlignmentModel<'a> {
    query: &'a [f64],
    candidate: &'a [f64],
    gap: f64,
}

impl TemporalAlignmentModel for ErpAlignmentModel<'_> {
    fn kind(&self) -> TemporalAlignmentKind {
        TemporalAlignmentKind::Erp
    }

    fn accumulation(&self) -> Accumulation {
        Accumulation::Additive
    }

    fn path_exists(&self, _query_len: usize, _candidate_len: usize) -> bool {
        true
    }

    fn local_cost(
        &self,
        operation: TemporalAlignmentOperation,
        query_consumed: usize,
        candidate_consumed: usize,
    ) -> Option<f64> {
        match operation {
            TemporalAlignmentOperation::Align => Some(
                (self.query.get(query_consumed.checked_sub(1)?)?
                    - self.candidate.get(candidate_consumed.checked_sub(1)?)?)
                .abs(),
            ),
            TemporalAlignmentOperation::AdvanceQuery => {
                Some((self.query.get(query_consumed.checked_sub(1)?)? - self.gap).abs())
            }
            TemporalAlignmentOperation::AdvanceCandidate => {
                Some((self.candidate.get(candidate_consumed.checked_sub(1)?)? - self.gap).abs())
            }
        }
    }
}

struct UnitGridTwedAlignmentModel<'a> {
    query: &'a [f64],
    candidate: &'a [f64],
    stiffness: f64,
    gap: f64,
}

impl UnitGridTwedAlignmentModel<'_> {
    #[inline]
    fn segment(&self, series: &[f64], consumed: usize) -> Option<f64> {
        let index = consumed.checked_sub(1)?;
        let current = *series.get(index)?;
        let previous = index
            .checked_sub(1)
            .and_then(|previous| series.get(previous).copied())
            .unwrap_or(0.0);
        Some((current - previous).abs() + self.stiffness + self.gap)
    }
}

impl TemporalAlignmentModel for UnitGridTwedAlignmentModel<'_> {
    fn kind(&self) -> TemporalAlignmentKind {
        TemporalAlignmentKind::UnitGridTwed
    }

    fn accumulation(&self) -> Accumulation {
        Accumulation::Additive
    }

    fn path_exists(&self, _query_len: usize, _candidate_len: usize) -> bool {
        true
    }

    fn local_cost(
        &self,
        operation: TemporalAlignmentOperation,
        query_consumed: usize,
        candidate_consumed: usize,
    ) -> Option<f64> {
        match operation {
            TemporalAlignmentOperation::AdvanceQuery => self.segment(self.query, query_consumed),
            TemporalAlignmentOperation::AdvanceCandidate => {
                self.segment(self.candidate, candidate_consumed)
            }
            TemporalAlignmentOperation::Align => {
                let query_index = query_consumed.checked_sub(1)?;
                let candidate_index = candidate_consumed.checked_sub(1)?;
                let query_current = *self.query.get(query_index)?;
                let candidate_current = *self.candidate.get(candidate_index)?;
                let query_previous = query_index
                    .checked_sub(1)
                    .and_then(|index| self.query.get(index).copied())
                    .unwrap_or(0.0);
                let candidate_previous = candidate_index
                    .checked_sub(1)
                    .and_then(|index| self.candidate.get(index).copied())
                    .unwrap_or(0.0);
                let displacement = query_consumed.abs_diff(candidate_consumed) as f64;
                Some(
                    (query_current - candidate_current).abs()
                        + (query_previous - candidate_previous).abs()
                        + 2.0 * self.stiffness * displacement,
                )
            }
        }
    }
}

struct PointCouplingAlignmentModel<'a> {
    query: &'a [f64],
    candidate: &'a [f64],
    kind: TemporalAlignmentKind,
    band: Option<usize>,
    accumulation: Accumulation,
}

impl TemporalAlignmentModel for PointCouplingAlignmentModel<'_> {
    fn kind(&self) -> TemporalAlignmentKind {
        self.kind
    }

    fn accumulation(&self) -> Accumulation {
        self.accumulation
    }

    fn path_exists(&self, query_len: usize, candidate_len: usize) -> bool {
        match (query_len, candidate_len) {
            (0, 0) => true,
            (0, _) | (_, 0) => false,
            _ => self
                .band
                .is_none_or(|band| query_len.abs_diff(candidate_len) <= band),
        }
    }

    fn trace_band(&self) -> Option<usize> {
        self.band
    }

    fn local_cost(
        &self,
        _operation: TemporalAlignmentOperation,
        query_consumed: usize,
        candidate_consumed: usize,
    ) -> Option<f64> {
        let query = *self.query.get(query_consumed.checked_sub(1)?)?;
        let candidate = *self.candidate.get(candidate_consumed.checked_sub(1)?)?;
        let deviation = query - candidate;
        Some(if self.kind == TemporalAlignmentKind::BandedDtw {
            deviation * deviation
        } else {
            deviation.abs()
        })
    }

    fn reported_cost(&self, accumulated: f64) -> f64 {
        if self.kind == TemporalAlignmentKind::BandedDtw {
            accumulated.sqrt()
        } else {
            accumulated
        }
    }
}

struct TimestampedTwedAlignmentModel<'a> {
    query: &'a TimestampedSeries,
    candidate: &'a TimestampedSeries,
    config: &'a MetricTimestampedTwedConfig,
}

impl TimestampedTwedAlignmentModel<'_> {
    #[inline]
    fn segment(&self, series: &TimestampedSeries, consumed: usize) -> Option<f64> {
        let index = consumed.checked_sub(1)?;
        let value = *series.values().get(index)?;
        let time = *series.timestamps().get(index)?;
        let previous_value = index
            .checked_sub(1)
            .and_then(|previous| series.values().get(previous).copied())
            .unwrap_or(0.0);
        let previous_time = index
            .checked_sub(1)
            .and_then(|previous| series.timestamps().get(previous).copied())
            .unwrap_or(series.origin());
        Some(timestamped_delete_cost(
            value,
            previous_value,
            time,
            previous_time,
            self.config,
        ))
    }
}

impl TemporalAlignmentModel for TimestampedTwedAlignmentModel<'_> {
    fn kind(&self) -> TemporalAlignmentKind {
        TemporalAlignmentKind::TimestampedTwed
    }

    fn accumulation(&self) -> Accumulation {
        Accumulation::Additive
    }

    fn path_exists(&self, query_len: usize, candidate_len: usize) -> bool {
        query_len > 0 && candidate_len > 0
    }

    fn local_cost(
        &self,
        operation: TemporalAlignmentOperation,
        query_consumed: usize,
        candidate_consumed: usize,
    ) -> Option<f64> {
        match operation {
            TemporalAlignmentOperation::AdvanceQuery => self.segment(self.query, query_consumed),
            TemporalAlignmentOperation::AdvanceCandidate => {
                self.segment(self.candidate, candidate_consumed)
            }
            TemporalAlignmentOperation::Align => {
                let query_index = query_consumed.checked_sub(1)?;
                let candidate_index = candidate_consumed.checked_sub(1)?;
                let query_value = *self.query.values().get(query_index)?;
                let candidate_value = *self.candidate.values().get(candidate_index)?;
                let query_time = *self.query.timestamps().get(query_index)?;
                let candidate_time = *self.candidate.timestamps().get(candidate_index)?;
                let query_previous_value = query_index
                    .checked_sub(1)
                    .and_then(|index| self.query.values().get(index).copied())
                    .unwrap_or(0.0);
                let candidate_previous_value = candidate_index
                    .checked_sub(1)
                    .and_then(|index| self.candidate.values().get(index).copied())
                    .unwrap_or(0.0);
                let query_previous_time = query_index
                    .checked_sub(1)
                    .and_then(|index| self.query.timestamps().get(index).copied())
                    .unwrap_or(self.query.origin());
                let candidate_previous_time = candidate_index
                    .checked_sub(1)
                    .and_then(|index| self.candidate.timestamps().get(index).copied())
                    .unwrap_or(self.candidate.origin());
                Some(timestamped_match_cost(
                    query_value,
                    query_previous_value,
                    query_time,
                    query_previous_time,
                    candidate_value,
                    candidate_previous_value,
                    candidate_time,
                    candidate_previous_time,
                    self.config,
                ))
            }
        }
    }
}

#[derive(Clone, Copy)]
struct TraceRow {
    start: usize,
    len: usize,
    offset: usize,
}

struct TraceGrid {
    rows: Vec<TraceRow>,
    codes: Vec<u8>,
}

impl TraceGrid {
    fn specification(
        query_len: usize,
        candidate_len: usize,
        band: Option<usize>,
    ) -> Option<(usize, usize)> {
        let row_count = query_len.checked_add(1)?;
        let metadata_bytes = row_count.checked_mul(size_of::<TraceRow>())?;
        let cells = if let Some(band) = band {
            let mut cells = 1_usize;
            for row in 1..=query_len {
                let start = row.saturating_sub(band).max(1);
                let end = row.saturating_add(band).min(candidate_len);
                if start <= end {
                    cells = cells.checked_add(end.checked_sub(start)?.checked_add(1)?)?;
                }
            }
            cells
        } else {
            row_count.checked_mul(candidate_len.checked_add(1)?)?
        };
        Some((cells, metadata_bytes))
    }

    fn try_new(
        query_len: usize,
        candidate_len: usize,
        band: Option<usize>,
        cells: usize,
    ) -> Result<Self, ()> {
        let row_count = query_len.checked_add(1).ok_or(())?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(row_count).map_err(|_| ())?;
        let mut offset = 0_usize;
        for row in 0..=query_len {
            let (start, len) = if let Some(band) = band {
                if row == 0 {
                    (0, 1)
                } else {
                    let start = row.saturating_sub(band).max(1);
                    let end = row.saturating_add(band).min(candidate_len);
                    (start, end.checked_sub(start).map_or(0, |width| width + 1))
                }
            } else {
                (0, candidate_len.checked_add(1).ok_or(())?)
            };
            rows.push(TraceRow { start, len, offset });
            offset = offset.checked_add(len).ok_or(())?;
        }
        if offset != cells {
            return Err(());
        }
        let mut codes = Vec::new();
        codes.try_reserve_exact(cells).map_err(|_| ())?;
        codes.resize(cells, 0);
        Ok(Self { rows, codes })
    }

    #[inline]
    fn bounds(&self, row: usize) -> Option<(usize, usize)> {
        let trace = *self.rows.get(row)?;
        trace
            .len
            .checked_sub(1)
            .and_then(|width| trace.start.checked_add(width))
            .map(|end| (trace.start, end))
    }

    #[inline]
    fn index(&self, row: usize, column: usize) -> Option<usize> {
        let trace = *self.rows.get(row)?;
        let relative = column.checked_sub(trace.start)?;
        (relative < trace.len).then(|| trace.offset + relative)
    }

    #[inline]
    fn set(&mut self, row: usize, column: usize, operation: TemporalAlignmentOperation) {
        let index = self
            .index(row, column)
            .expect("a computed trace cell belongs to its validated layout");
        self.codes[index] = operation as u8;
    }

    #[inline]
    fn get(&self, row: usize, column: usize) -> Option<TemporalAlignmentOperation> {
        match *self.codes.get(self.index(row, column)?)? {
            value if value == TemporalAlignmentOperation::Align as u8 => {
                Some(TemporalAlignmentOperation::Align)
            }
            value if value == TemporalAlignmentOperation::AdvanceQuery as u8 => {
                Some(TemporalAlignmentOperation::AdvanceQuery)
            }
            value if value == TemporalAlignmentOperation::AdvanceCandidate as u8 => {
                Some(TemporalAlignmentOperation::AdvanceCandidate)
            }
            _ => None,
        }
    }
}

#[inline]
fn consider_transition(
    best: &mut f64,
    selected: &mut Option<TemporalAlignmentOperation>,
    operation: TemporalAlignmentOperation,
    candidate: f64,
) {
    if candidate < *best {
        *best = candidate;
        *selected = Some(operation);
    }
}

fn extract_alignment<M: TemporalAlignmentModel>(
    model: &M,
    query_len: usize,
    candidate_len: usize,
    cutoff: f64,
    mut ledger: ResourceLedger,
) -> OperationOutcome<ExactDecision<TemporalAlignmentWitness>> {
    if !model.path_exists(query_len, candidate_len) {
        return OperationOutcome::Complete {
            value: ExactDecision::NoFiniteAlignment,
            usage: ledger.usage(),
        };
    }
    let Some((trace_cells, metadata_bytes)) =
        TraceGrid::specification(query_len, candidate_len, model.trace_band())
    else {
        return alignment_arithmetic_incomplete(ledger, ResourceKind::DpCells);
    };
    let Some(columns) = candidate_len.checked_add(1) else {
        return alignment_arithmetic_incomplete(ledger, ResourceKind::DpCells);
    };
    let Some(work) = trace_cells.checked_mul(3) else {
        return alignment_arithmetic_incomplete(ledger, ResourceKind::WorkUnits);
    };
    let Some(row_bytes) = columns
        .checked_mul(2)
        .and_then(|slots| slots.checked_mul(size_of::<f64>()))
    else {
        return alignment_arithmetic_incomplete(ledger, ResourceKind::ScratchBytes);
    };
    let Some(scratch_bytes) = trace_cells
        .checked_add(metadata_bytes)
        .and_then(|bytes| bytes.checked_add(row_bytes))
    else {
        return alignment_arithmetic_incomplete(ledger, ResourceKind::ScratchBytes);
    };
    if let Err(reason) = ledger.charge_many(&[
        (ResourceKind::DpCells, trace_cells),
        (ResourceKind::WorkUnits, work),
    ]) {
        return alignment_incomplete(ledger, reason);
    }
    if let Err(reason) = ledger.observe_peak(ResourceKind::ScratchBytes, scratch_bytes) {
        return alignment_incomplete(ledger, reason);
    }

    let mut trace =
        match TraceGrid::try_new(query_len, candidate_len, model.trace_band(), trace_cells) {
            Ok(trace) => trace,
            Err(()) => {
                return alignment_incomplete(
                    ledger,
                    IncompleteReason::AllocationFailed {
                        resource: ResourceKind::ScratchBytes,
                        requested: scratch_bytes,
                    },
                )
            }
        };
    let mut previous = Vec::new();
    if previous.try_reserve_exact(columns).is_err() {
        return alignment_incomplete(
            ledger,
            IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested: scratch_bytes,
            },
        );
    }
    previous.resize(columns, f64::INFINITY);
    let mut current = Vec::new();
    if current.try_reserve_exact(columns).is_err() {
        return alignment_incomplete(
            ledger,
            IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested: scratch_bytes,
            },
        );
    }
    current.resize(columns, f64::INFINITY);
    previous[0] = 0.0;

    if let Some((start, end)) = trace.bounds(0) {
        for column in start.max(1)..=end {
            if let Some(local) =
                model.local_cost(TemporalAlignmentOperation::AdvanceCandidate, 0, column)
            {
                let cost = model.accumulation().extend(previous[column - 1], local);
                if cost.is_finite() {
                    previous[column] = cost;
                    trace.set(0, column, TemporalAlignmentOperation::AdvanceCandidate);
                }
            }
        }
    }

    for row in 1..=query_len {
        let Some((start, end)) = trace.bounds(row) else {
            std::mem::swap(&mut previous, &mut current);
            continue;
        };
        if model.trace_band().is_some() {
            // A packed band advances by at most one column per row. Clear its
            // live cells and one sentinel on either side; this prevents stale
            // two-row values without turning narrow-band DTW back into
            // quadratic row clearing.
            let clear_start = start.saturating_sub(1);
            let clear_end = end.saturating_add(1).min(candidate_len);
            current[clear_start..=clear_end].fill(f64::INFINITY);
        } else {
            current.fill(f64::INFINITY);
        }
        for column in start..=end {
            let mut best = f64::INFINITY;
            let mut selected = None;
            // This priority is part of the stable certificate contract.
            if row > 0 && column > 0 {
                if let Some(local) =
                    model.local_cost(TemporalAlignmentOperation::Align, row, column)
                {
                    consider_transition(
                        &mut best,
                        &mut selected,
                        TemporalAlignmentOperation::Align,
                        model.accumulation().extend(previous[column - 1], local),
                    );
                }
            }
            if let Some(local) =
                model.local_cost(TemporalAlignmentOperation::AdvanceQuery, row, column)
            {
                consider_transition(
                    &mut best,
                    &mut selected,
                    TemporalAlignmentOperation::AdvanceQuery,
                    model.accumulation().extend(previous[column], local),
                );
            }
            if column > 0 {
                if let Some(local) =
                    model.local_cost(TemporalAlignmentOperation::AdvanceCandidate, row, column)
                {
                    consider_transition(
                        &mut best,
                        &mut selected,
                        TemporalAlignmentOperation::AdvanceCandidate,
                        model.accumulation().extend(current[column - 1], local),
                    );
                }
            }
            if let Some(operation) = selected.filter(|_| best.is_finite()) {
                current[column] = best;
                trace.set(row, column, operation);
            }
        }
        std::mem::swap(&mut previous, &mut current);
    }

    let accumulated = previous[candidate_len];
    if !accumulated.is_finite() {
        return if cutoff.is_finite() {
            OperationOutcome::Complete {
                value: ExactDecision::AboveCutoff,
                usage: ledger.usage(),
            }
        } else {
            alignment_incomplete(ledger, IncompleteReason::NumericOverflow)
        };
    }
    let distance = model.reported_cost(accumulated);
    if !distance.is_finite() {
        return alignment_incomplete(ledger, IncompleteReason::NumericOverflow);
    }
    if distance > cutoff {
        return OperationOutcome::Complete {
            value: ExactDecision::AboveCutoff,
            usage: ledger.usage(),
        };
    }

    let Some(max_steps) = query_len.checked_add(candidate_len) else {
        return alignment_arithmetic_incomplete(ledger, ResourceKind::WitnessBytes);
    };
    let Some(witness_bytes) = max_steps
        .checked_mul(size_of::<TemporalAlignmentStep>())
        .and_then(|bytes| bytes.checked_add(size_of::<TemporalAlignmentWitness>()))
    else {
        return alignment_arithmetic_incomplete(ledger, ResourceKind::WitnessBytes);
    };
    if let Err(reason) = ledger.observe_peak(ResourceKind::WitnessBytes, witness_bytes) {
        return alignment_incomplete(ledger, reason);
    }
    let mut reversed = Vec::new();
    if reversed.try_reserve_exact(max_steps).is_err() {
        return alignment_incomplete(
            ledger,
            IncompleteReason::AllocationFailed {
                resource: ResourceKind::WitnessBytes,
                requested: witness_bytes,
            },
        );
    }
    let mut row = query_len;
    let mut column = candidate_len;
    while row > 0 || column > 0 {
        let Some(operation) = trace.get(row, column) else {
            return alignment_incomplete(ledger, IncompleteReason::InvalidStoredData);
        };
        let Some(local) = model.local_cost(operation, row, column) else {
            return alignment_incomplete(ledger, IncompleteReason::InvalidStoredData);
        };
        let query_endpoint = match row.checked_sub(1).map(u64::try_from).transpose() {
            Ok(endpoint) => endpoint,
            Err(_) => return alignment_arithmetic_incomplete(ledger, ResourceKind::WitnessBytes),
        };
        let candidate_endpoint = match column.checked_sub(1).map(u64::try_from).transpose() {
            Ok(endpoint) => endpoint,
            Err(_) => return alignment_arithmetic_incomplete(ledger, ResourceKind::WitnessBytes),
        };
        reversed.push(TemporalAlignmentStep::from_raw_parts(
            operation,
            query_endpoint,
            candidate_endpoint,
            local.to_bits(),
        ));
        match operation {
            TemporalAlignmentOperation::Align => {
                row -= 1;
                column -= 1;
            }
            TemporalAlignmentOperation::AdvanceQuery => row -= 1,
            TemporalAlignmentOperation::AdvanceCandidate => column -= 1,
        }
    }
    reversed.reverse();
    let witness = TemporalAlignmentWitness::from_parts(
        TEMPORAL_ALIGNMENT_WITNESS_VERSION,
        model.kind(),
        reversed,
    );
    if replay_alignment(model, &witness, query_len, candidate_len)
        .ok()
        .map(f64::to_bits)
        != Some(distance.to_bits())
    {
        return alignment_incomplete(ledger, IncompleteReason::InvalidStoredData);
    }
    OperationOutcome::Complete {
        value: ExactDecision::WithinCutoff { distance, witness },
        usage: ledger.usage(),
    }
}

fn endpoint(consumed: usize) -> Result<Option<u64>, TemporalWitnessReplayError> {
    consumed
        .checked_sub(1)
        .map(u64::try_from)
        .transpose()
        .map_err(|_| TemporalWitnessReplayError::InvalidDomain)
}

fn replay_alignment<M: TemporalAlignmentModel>(
    model: &M,
    witness: &TemporalAlignmentWitness,
    query_len: usize,
    candidate_len: usize,
) -> Result<f64, TemporalWitnessReplayError> {
    if witness.version != TEMPORAL_ALIGNMENT_WITNESS_VERSION {
        return Err(TemporalWitnessReplayError::UnsupportedVersion {
            found: witness.version,
        });
    }
    if witness.kind != model.kind() {
        return Err(TemporalWitnessReplayError::KernelMismatch {
            expected: model.kind(),
            found: witness.kind,
        });
    }
    if !model.path_exists(query_len, candidate_len) {
        return Err(TemporalWitnessReplayError::InvalidDomain);
    }
    let mut query_consumed = 0_usize;
    let mut candidate_consumed = 0_usize;
    let mut accumulated = 0.0;
    for (step_index, step) in witness.steps.iter().enumerate() {
        match step.operation {
            TemporalAlignmentOperation::Align => {
                query_consumed = query_consumed
                    .checked_add(1)
                    .ok_or(TemporalWitnessReplayError::MalformedOperation { step_index })?;
                candidate_consumed = candidate_consumed
                    .checked_add(1)
                    .ok_or(TemporalWitnessReplayError::MalformedOperation { step_index })?;
            }
            TemporalAlignmentOperation::AdvanceQuery => {
                query_consumed = query_consumed
                    .checked_add(1)
                    .ok_or(TemporalWitnessReplayError::MalformedOperation { step_index })?;
            }
            TemporalAlignmentOperation::AdvanceCandidate => {
                candidate_consumed = candidate_consumed
                    .checked_add(1)
                    .ok_or(TemporalWitnessReplayError::MalformedOperation { step_index })?;
            }
        }
        if query_consumed > query_len || candidate_consumed > candidate_len {
            return Err(TemporalWitnessReplayError::MalformedOperation { step_index });
        }
        if step.query_endpoint != endpoint(query_consumed)?
            || step.candidate_endpoint != endpoint(candidate_consumed)?
        {
            return Err(TemporalWitnessReplayError::MalformedEndpoint { step_index });
        }
        let Some(local) = model.local_cost(step.operation, query_consumed, candidate_consumed)
        else {
            return Err(TemporalWitnessReplayError::MalformedOperation { step_index });
        };
        if !local.is_finite() {
            return Err(TemporalWitnessReplayError::NumericOverflow);
        }
        if step.local_cost_bits != local.to_bits() {
            return Err(TemporalWitnessReplayError::LocalCostMismatch { step_index });
        }
        accumulated = model.accumulation().extend(accumulated, local);
        if !accumulated.is_finite() {
            return Err(TemporalWitnessReplayError::NumericOverflow);
        }
    }
    if query_consumed != query_len || candidate_consumed != candidate_len {
        return Err(TemporalWitnessReplayError::MalformedEndpoint {
            step_index: witness.steps.len(),
        });
    }
    let distance = model.reported_cost(accumulated);
    distance
        .is_finite()
        .then_some(distance)
        .ok_or(TemporalWitnessReplayError::NumericOverflow)
}

fn validate_cutoff(cutoff: f64) -> Result<(), TemporalValidationError> {
    if cutoff.is_nan() || cutoff < 0.0 || (cutoff.is_infinite() && cutoff.is_sign_negative()) {
        Err(TemporalValidationError::InvalidCutoff)
    } else {
        Ok(())
    }
}

fn finite_series(series: &[f64]) -> bool {
    series.iter().all(|value| value.is_finite())
}

impl TemporalAlignmentWitness {
    /// Replay and validate this witness as scalar ERP.
    pub fn replay_erp(
        &self,
        query: &[f64],
        candidate: &[f64],
        config: &ErpConfig,
    ) -> Result<f64, TemporalWitnessReplayError> {
        if !finite_series(query) || !finite_series(candidate) {
            return Err(TemporalWitnessReplayError::InvalidDomain);
        }
        replay_alignment(
            &ErpAlignmentModel {
                query,
                candidate,
                gap: config.gap_value(),
            },
            self,
            query.len(),
            candidate.len(),
        )
    }

    /// Replay and validate this witness as scalar unit-grid TWED.
    pub fn replay_unit_grid_twed(
        &self,
        query: &[f64],
        candidate: &[f64],
        config: &TwedConfig,
    ) -> Result<f64, TemporalWitnessReplayError> {
        if !finite_series(query) || !finite_series(candidate) {
            return Err(TemporalWitnessReplayError::InvalidDomain);
        }
        replay_alignment(
            &UnitGridTwedAlignmentModel {
                query,
                candidate,
                stiffness: config.stiffness(),
                gap: config.gap_penalty(),
            },
            self,
            query.len(),
            candidate.len(),
        )
    }

    /// Replay and validate this witness as root-distance banded DTW.
    pub fn replay_banded_dtw(
        &self,
        query: &[f64],
        candidate: &[f64],
        config: &DtwConfig,
    ) -> Result<f64, TemporalWitnessReplayError> {
        if !finite_series(query) || !finite_series(candidate) {
            return Err(TemporalWitnessReplayError::InvalidDomain);
        }
        replay_alignment(
            &PointCouplingAlignmentModel {
                query,
                candidate,
                kind: TemporalAlignmentKind::BandedDtw,
                band: Some(config.band),
                accumulation: Accumulation::Additive,
            },
            self,
            query.len(),
            candidate.len(),
        )
    }

    /// Replay and validate this witness as scalar discrete Fréchet.
    pub fn replay_discrete_frechet(
        &self,
        query: &[f64],
        candidate: &[f64],
        _config: &FrechetConfig,
    ) -> Result<f64, TemporalWitnessReplayError> {
        if !finite_series(query) || !finite_series(candidate) {
            return Err(TemporalWitnessReplayError::InvalidDomain);
        }
        replay_alignment(
            &PointCouplingAlignmentModel {
                query,
                candidate,
                kind: TemporalAlignmentKind::DiscreteFrechet,
                band: None,
                accumulation: Accumulation::Bottleneck,
            },
            self,
            query.len(),
            candidate.len(),
        )
    }

    /// Replay and validate this witness as physical-time TWED.
    pub fn replay_timestamped_twed(
        &self,
        query: &TimestampedSeries,
        candidate: &TimestampedSeries,
        config: &MetricTimestampedTwedConfig,
    ) -> Result<f64, TemporalWitnessReplayError> {
        if query.unit() != candidate.unit() || query.origin() != candidate.origin() {
            return Err(TemporalWitnessReplayError::InvalidDomain);
        }
        replay_alignment(
            &TimestampedTwedAlignmentModel {
                query,
                candidate,
                config,
            },
            self,
            query.values().len(),
            candidate.values().len(),
        )
    }
}

impl ErpConfig {
    /// Compute an exact cutoff decision with a deterministic ERP witness.
    pub fn distance_with_alignment_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision<TemporalAlignmentWitness>>, TemporalValidationError>
    {
        validate_cutoff(cutoff)?;
        let ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, query)?;
        ledger.validate_finite_series(Operand::Candidate, candidate)?;
        Ok(extract_alignment(
            &ErpAlignmentModel {
                query,
                candidate,
                gap: self.gap_value(),
            },
            query.len(),
            candidate.len(),
            cutoff,
            ledger,
        ))
    }
}

impl TwedConfig {
    /// Compute exact unit-grid TWED with a deterministic replayable witness.
    pub fn distance_with_alignment_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision<TemporalAlignmentWitness>>, TemporalValidationError>
    {
        validate_cutoff(cutoff)?;
        let ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, query)?;
        ledger.validate_finite_series(Operand::Candidate, candidate)?;
        Ok(extract_alignment(
            &UnitGridTwedAlignmentModel {
                query,
                candidate,
                stiffness: self.stiffness(),
                gap: self.gap_penalty(),
            },
            query.len(),
            candidate.len(),
            cutoff,
            ledger,
        ))
    }
}

impl MetricTwedConfig {
    /// Compute exact metric unit-grid TWED with a replayable witness.
    #[inline]
    pub fn distance_with_alignment_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision<TemporalAlignmentWitness>>, TemporalValidationError>
    {
        self.as_config()
            .distance_with_alignment_bounded(query, candidate, cutoff, limits)
    }
}

impl DtwConfig {
    /// Compute exact root-distance banded DTW with a replayable witness.
    pub fn distance_with_alignment_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision<TemporalAlignmentWitness>>, TemporalValidationError>
    {
        validate_cutoff(cutoff)?;
        let ledger = ResourceLedger::new(limits);
        if self.band > limits.max_band_width {
            return Ok(alignment_incomplete(
                ledger,
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::BandWidth,
                    limit: limits.max_band_width,
                    requested: self.band,
                },
            ));
        }
        ledger.validate_finite_series(Operand::Query, query)?;
        ledger.validate_finite_series(Operand::Candidate, candidate)?;
        Ok(extract_alignment(
            &PointCouplingAlignmentModel {
                query,
                candidate,
                kind: TemporalAlignmentKind::BandedDtw,
                band: Some(self.band),
                accumulation: Accumulation::Additive,
            },
            query.len(),
            candidate.len(),
            cutoff,
            ledger,
        ))
    }
}

impl FrechetConfig {
    /// Compute exact scalar discrete Fréchet with a replayable coupling witness.
    pub fn distance_with_alignment_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision<TemporalAlignmentWitness>>, TemporalValidationError>
    {
        validate_cutoff(cutoff)?;
        let ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, query)?;
        ledger.validate_finite_series(Operand::Candidate, candidate)?;
        Ok(extract_alignment(
            &PointCouplingAlignmentModel {
                query,
                candidate,
                kind: TemporalAlignmentKind::DiscreteFrechet,
                band: None,
                accumulation: Accumulation::Bottleneck,
            },
            query.len(),
            candidate.len(),
            cutoff,
            ledger,
        ))
    }
}

impl MetricTimestampedTwedConfig {
    /// Compute exact physical-time TWED with a deterministic replayable witness.
    pub fn distance_with_alignment_bounded(
        &self,
        query: &TimestampedSeries,
        candidate: &TimestampedSeries,
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision<TemporalAlignmentWitness>>, TimestampedTwedError>
    {
        if query.unit() != candidate.unit() {
            return Err(TimestampedTwedError::MixedUnits);
        }
        if query.origin() != candidate.origin() {
            return Err(TimestampedTwedError::MixedOrigins);
        }
        validate_cutoff(cutoff)?;
        let ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, query.values())?;
        ledger.validate_finite_series(Operand::Candidate, candidate.values())?;
        Ok(extract_alignment(
            &TimestampedTwedAlignmentModel {
                query,
                candidate,
                config: self,
            },
            query.values().len(),
            candidate.values().len(),
            cutoff,
            ledger,
        ))
    }
}

fn alignment_incomplete(
    ledger: ResourceLedger,
    reason: IncompleteReason,
) -> OperationOutcome<ExactDecision<TemporalAlignmentWitness>> {
    OperationOutcome::Incomplete {
        partial: None,
        reason,
        continuation: None,
        usage: ledger.usage(),
    }
}

fn alignment_arithmetic_incomplete(
    ledger: ResourceLedger,
    resource: ResourceKind,
) -> OperationOutcome<ExactDecision<TemporalAlignmentWitness>> {
    alignment_incomplete(ledger, IncompleteReason::ArithmeticOverflow { resource })
}

fn incomplete(
    ledger: ResourceLedger,
    reason: IncompleteReason,
) -> OperationOutcome<ExactDecision<MsmAlignmentWitness>> {
    OperationOutcome::Incomplete {
        partial: None,
        reason,
        continuation: None,
        usage: ledger.usage(),
    }
}

fn arithmetic_incomplete(
    ledger: ResourceLedger,
    resource: ResourceKind,
) -> OperationOutcome<ExactDecision<MsmAlignmentWitness>> {
    incomplete(ledger, IncompleteReason::ArithmeticOverflow { resource })
}
