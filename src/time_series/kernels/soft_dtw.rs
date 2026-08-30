//! Analysis-only Soft-DTW for finite scalar sequences.
//!
//! This is the stabilized log-sum-exp recurrence of Cuturi and Blondel,
//! *Soft-DTW: a Differentiable Loss Function for Time-Series*, ICML 2017,
//! [arXiv:1703.01541](https://arxiv.org/abs/1703.01541). It is intentionally
//! not an [`super::super::elastic::ElasticKernel`]: Soft-DTW aggregates every
//! alignment path, so min-cost antichain subsumption and exact absence-proof
//! trie pruning do not preserve its value.

use thiserror::Error;

use crate::time_series::bounded::{
    IncompleteReason, Operand, OperationOutcome, ResourceKind, ResourceLedger, ResourceLimits,
    TemporalValidationError,
};

/// Validated positive smoothing parameter for analysis-only Soft-DTW.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftDtwConfig {
    gamma: f64,
}

/// Invalid Soft-DTW analysis configuration.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum SoftDtwConfigError {
    /// `gamma` must be finite and strictly positive.
    #[error("Soft-DTW smoothing gamma must be finite and strictly positive")]
    InvalidGamma,
}

/// Exact outcome of a bounded Soft-DTW analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
pub enum SoftDtwAnalysis {
    /// Finite soft alignment value. Soft-DTW is a loss, not a metric distance,
    /// and this value may be negative.
    Finite {
        /// Complete stabilized log-sum-exp recurrence value.
        value: f64,
    },
    /// One operand was empty while the other was not, so no monotone alignment
    /// path exists.
    NoFiniteAlignment,
}

impl SoftDtwConfig {
    /// Validate a strictly positive finite smoothing parameter.
    pub fn try_new(gamma: f64) -> Result<Self, SoftDtwConfigError> {
        if gamma.is_finite() && gamma > 0.0 {
            Ok(Self { gamma })
        } else {
            Err(SoftDtwConfigError::InvalidGamma)
        }
    }

    /// Positive smoothing parameter.
    #[inline]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Evaluate the complete soft alignment partition under hard limits.
    ///
    /// The implementation is iterative and stores two rows on the shorter
    /// operand. It never applies a row cutoff or path subsumption: either the
    /// whole `m × n` recurrence completes, or the result is tagged incomplete.
    pub fn analyze_bounded(
        &self,
        left: &[f64],
        right: &[f64],
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<SoftDtwAnalysis>, TemporalValidationError> {
        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, left)?;
        ledger.validate_finite_series(Operand::Candidate, right)?;

        if left.is_empty() || right.is_empty() {
            return Ok(OperationOutcome::Complete {
                value: if left.is_empty() && right.is_empty() {
                    SoftDtwAnalysis::Finite { value: 0.0 }
                } else {
                    SoftDtwAnalysis::NoFiniteAlignment
                },
                usage: ledger.usage(),
            });
        }

        let cells = match left.len().checked_mul(right.len()) {
            Some(cells) => cells,
            None => return Ok(incomplete(ledger, ResourceKind::DpCells)),
        };
        let width = left.len().min(right.len()).checked_add(1);
        let scratch_bytes = width
            .and_then(|width| width.checked_mul(2))
            .and_then(|slots| slots.checked_mul(std::mem::size_of::<f64>()));
        let Some(scratch_bytes) = scratch_bytes else {
            return Ok(incomplete(ledger, ResourceKind::ScratchBytes));
        };
        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::DpCells, cells),
            (ResourceKind::WorkUnits, cells),
            (ResourceKind::ScratchBytes, scratch_bytes),
        ]) {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason,
                continuation: None,
                usage: ledger.usage(),
            });
        }

        let (rows, columns) = if right.len() <= left.len() {
            (left, right)
        } else {
            (right, left)
        };
        let width = columns.len() + 1;
        let mut previous = Vec::new();
        if previous.try_reserve_exact(width).is_err() {
            return Ok(allocation_incomplete(ledger, scratch_bytes));
        }
        previous.resize(width, f64::INFINITY);
        previous[0] = 0.0;
        let mut current = Vec::new();
        if current.try_reserve_exact(width).is_err() {
            return Ok(allocation_incomplete(ledger, scratch_bytes));
        }
        current.resize(width, f64::INFINITY);

        for left_value in rows {
            current[0] = f64::INFINITY;
            for (column_index, right_value) in columns.iter().enumerate() {
                let column = column_index + 1;
                let delta = *left_value - *right_value;
                let local = delta * delta;
                let predecessor = stable_soft_min(
                    previous[column - 1],
                    previous[column],
                    current[column - 1],
                    self.gamma,
                );
                current[column] = local + predecessor;
            }
            std::mem::swap(&mut previous, &mut current);
        }

        let value = previous[columns.len()];
        if !value.is_finite() {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason: IncompleteReason::NumericOverflow,
                continuation: None,
                usage: ledger.usage(),
            });
        }
        Ok(OperationOutcome::Complete {
            value: SoftDtwAnalysis::Finite { value },
            usage: ledger.usage(),
        })
    }
}

#[inline]
fn stable_soft_min(left: f64, diagonal: f64, above: f64, gamma: f64) -> f64 {
    let minimum = left.min(diagonal).min(above);
    if !minimum.is_finite() {
        return f64::INFINITY;
    }
    let partition = ((minimum - left) / gamma).exp()
        + ((minimum - diagonal) / gamma).exp()
        + ((minimum - above) / gamma).exp();
    minimum - gamma * partition.ln()
}

fn incomplete(ledger: ResourceLedger, resource: ResourceKind) -> OperationOutcome<SoftDtwAnalysis> {
    OperationOutcome::Incomplete {
        partial: None,
        reason: IncompleteReason::ArithmeticOverflow { resource },
        continuation: None,
        usage: ledger.usage(),
    }
}

fn allocation_incomplete(
    ledger: ResourceLedger,
    requested: usize,
) -> OperationOutcome<SoftDtwAnalysis> {
    OperationOutcome::Incomplete {
        partial: None,
        reason: IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        },
        continuation: None,
        usage: ledger.usage(),
    }
}
