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
use super::msm::MsmConfig;

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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MsmAlignmentWitness {
    steps: Vec<MsmAlignmentStep>,
}

impl MsmAlignmentWitness {
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
        let witness = MsmAlignmentWitness { steps: reversed };
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
