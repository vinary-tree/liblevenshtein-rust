//! Analysis-only Soft-DTW for finite scalar sequences.
//!
//! This is the stabilized log-sum-exp recurrence of Cuturi and Blondel,
//! *Soft-DTW: a Differentiable Loss Function for Time-Series*, ICML 2017,
//! [arXiv:1703.01541](https://arxiv.org/abs/1703.01541). It is intentionally
//! not an [`super::super::elastic::ElasticKernel`]: Soft-DTW aggregates every
//! alignment path, so min-cost antichain subsumption and exact absence-proof
//! trie pruning do not preserve its value.
//!
//! ```compile_fail
//! use liblevenshtein::time_series::elastic::MetricElasticKernel;
//! use liblevenshtein::time_series::SoftDtwConfig;
//!
//! fn exact_metric_index<K: MetricElasticKernel>() {}
//! exact_metric_index::<SoftDtwConfig>();
//! ```

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

/// Complete Soft-DTW value and gradients for two scalar operands.
///
/// Gradient evaluation retains the full forward and adjoint matrices because
/// every forward cell can influence the reverse sweep. This batch-only type is
/// intentionally separate from the constant-retention score surface.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftDtwGradientAnalysis {
    /// Complete Soft-DTW discrepancy.
    pub value: f64,
    /// Derivative with respect to each sample of the left operand.
    pub left_gradient: Vec<f64>,
    /// Derivative with respect to each sample of the right operand.
    pub right_gradient: Vec<f64>,
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

    /// Evaluate the discrepancy and both operand gradients under hard limits.
    ///
    /// The reverse recurrence is Algorithm 2 of Cuturi and Blondel. It is an
    /// iterative reverse row-major sweep, so stack use is constant. Unlike the
    /// score-only method, differentiability intrinsically requires
    /// $`\mathcal{O}(mn)`$ retained adjoint state; the complete allocation is
    /// checked before either matrix is created.
    pub fn analyze_with_gradient_bounded(
        &self,
        left: &[f64],
        right: &[f64],
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<SoftDtwGradientAnalysis>, TemporalValidationError> {
        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, left)?;
        ledger.validate_finite_series(Operand::Candidate, right)?;
        if left.is_empty() || right.is_empty() {
            return Err(TemporalValidationError::EmptyMetricSeries);
        }

        let matrix_rows = match left.len().checked_add(2) {
            Some(rows) => rows,
            None => {
                return Ok(gradient_arithmetic_incomplete(
                    ledger,
                    ResourceKind::DpCells,
                ))
            }
        };
        let matrix_columns = match right.len().checked_add(2) {
            Some(columns) => columns,
            None => {
                return Ok(gradient_arithmetic_incomplete(
                    ledger,
                    ResourceKind::DpCells,
                ))
            }
        };
        let matrix_cells = match matrix_rows.checked_mul(matrix_columns) {
            Some(cells) => cells,
            None => {
                return Ok(gradient_arithmetic_incomplete(
                    ledger,
                    ResourceKind::DpCells,
                ))
            }
        };
        let recurrence_cells = match left.len().checked_mul(right.len()) {
            Some(cells) => cells,
            None => {
                return Ok(gradient_arithmetic_incomplete(
                    ledger,
                    ResourceKind::DpCells,
                ))
            }
        };
        let work = match recurrence_cells.checked_mul(2) {
            Some(work) => work,
            None => {
                return Ok(gradient_arithmetic_incomplete(
                    ledger,
                    ResourceKind::WorkUnits,
                ))
            }
        };
        let gradient_slots = match left.len().checked_add(right.len()) {
            Some(slots) => slots,
            None => {
                return Ok(gradient_arithmetic_incomplete(
                    ledger,
                    ResourceKind::ScratchBytes,
                ));
            }
        };
        let scratch_slots = match matrix_cells
            .checked_mul(2)
            .and_then(|slots| slots.checked_add(gradient_slots))
        {
            Some(slots) => slots,
            None => {
                return Ok(gradient_arithmetic_incomplete(
                    ledger,
                    ResourceKind::ScratchBytes,
                ));
            }
        };
        let scratch_bytes = match scratch_slots.checked_mul(std::mem::size_of::<f64>()) {
            Some(bytes) => bytes,
            None => {
                return Ok(gradient_arithmetic_incomplete(
                    ledger,
                    ResourceKind::ScratchBytes,
                ));
            }
        };
        if let Err(reason) = ledger.charge_many(&[
            (ResourceKind::DpCells, recurrence_cells),
            (ResourceKind::WorkUnits, work),
        ]) {
            return Ok(gradient_incomplete(ledger, reason));
        }
        if let Err(reason) = ledger.observe_peak(ResourceKind::ScratchBytes, scratch_bytes) {
            return Ok(gradient_incomplete(ledger, reason));
        }

        let mut forward = Vec::new();
        if forward.try_reserve_exact(matrix_cells).is_err() {
            return Ok(gradient_allocation_incomplete(ledger, scratch_bytes));
        }
        forward.resize(matrix_cells, f64::INFINITY);
        let mut adjoint = Vec::new();
        if adjoint.try_reserve_exact(matrix_cells).is_err() {
            return Ok(gradient_allocation_incomplete(ledger, scratch_bytes));
        }
        adjoint.resize(matrix_cells, 0.0);
        let mut left_gradient = Vec::new();
        if left_gradient.try_reserve_exact(left.len()).is_err() {
            return Ok(gradient_allocation_incomplete(ledger, scratch_bytes));
        }
        left_gradient.resize(left.len(), 0.0);
        let mut right_gradient = Vec::new();
        if right_gradient.try_reserve_exact(right.len()).is_err() {
            return Ok(gradient_allocation_incomplete(ledger, scratch_bytes));
        }
        right_gradient.resize(right.len(), 0.0);

        let cell = |row: usize, column: usize| row * matrix_columns + column;
        forward[cell(0, 0)] = 0.0;
        for row in 1..=left.len() {
            for column in 1..=right.len() {
                let delta = left[row - 1] - right[column - 1];
                forward[cell(row, column)] = delta * delta
                    + stable_soft_min(
                        forward[cell(row - 1, column)],
                        forward[cell(row - 1, column - 1)],
                        forward[cell(row, column - 1)],
                        self.gamma,
                    );
            }
        }
        let value = forward[cell(left.len(), right.len())];
        if !value.is_finite() {
            return Ok(gradient_incomplete(
                ledger,
                IncompleteReason::NumericOverflow,
            ));
        }

        // Algorithm 2's terminal sentinel makes the unique virtual successor
        // contribute one to the final real cell. The remaining bottom/right
        // border is negative infinity, making its exponential weights zero.
        for row in 0..matrix_rows {
            forward[cell(row, matrix_columns - 1)] = f64::NEG_INFINITY;
        }
        for column in 0..matrix_columns {
            forward[cell(matrix_rows - 1, column)] = f64::NEG_INFINITY;
        }
        forward[cell(matrix_rows - 1, matrix_columns - 1)] = value;
        adjoint[cell(matrix_rows - 1, matrix_columns - 1)] = 1.0;

        for row in (1..=left.len()).rev() {
            for column in (1..=right.len()).rev() {
                let current = forward[cell(row, column)];
                let down_local = if row < left.len() {
                    let delta = left[row] - right[column - 1];
                    delta * delta
                } else {
                    0.0
                };
                let right_local = if column < right.len() {
                    let delta = left[row - 1] - right[column];
                    delta * delta
                } else {
                    0.0
                };
                let diagonal_local = if row < left.len() && column < right.len() {
                    let delta = left[row] - right[column];
                    delta * delta
                } else {
                    0.0
                };
                let down = reverse_weight(
                    forward[cell(row + 1, column)],
                    current,
                    down_local,
                    self.gamma,
                );
                let right_weight = reverse_weight(
                    forward[cell(row, column + 1)],
                    current,
                    right_local,
                    self.gamma,
                );
                let diagonal = reverse_weight(
                    forward[cell(row + 1, column + 1)],
                    current,
                    diagonal_local,
                    self.gamma,
                );
                let alignment = adjoint[cell(row + 1, column)] * down
                    + adjoint[cell(row, column + 1)] * right_weight
                    + adjoint[cell(row + 1, column + 1)] * diagonal;
                if !alignment.is_finite() {
                    return Ok(gradient_incomplete(
                        ledger,
                        IncompleteReason::NumericOverflow,
                    ));
                }
                adjoint[cell(row, column)] = alignment;
                let derivative = 2.0 * alignment * (left[row - 1] - right[column - 1]);
                left_gradient[row - 1] += derivative;
                right_gradient[column - 1] -= derivative;
            }
        }
        if left_gradient
            .iter()
            .chain(&right_gradient)
            .any(|gradient| !gradient.is_finite())
        {
            return Ok(gradient_incomplete(
                ledger,
                IncompleteReason::NumericOverflow,
            ));
        }
        Ok(OperationOutcome::Complete {
            value: SoftDtwGradientAnalysis {
                value,
                left_gradient,
                right_gradient,
            },
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

#[inline]
fn reverse_weight(successor: f64, current: f64, local: f64, gamma: f64) -> f64 {
    if successor == f64::NEG_INFINITY {
        0.0
    } else {
        ((successor - current - local) / gamma).exp()
    }
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

fn gradient_arithmetic_incomplete(
    ledger: ResourceLedger,
    resource: ResourceKind,
) -> OperationOutcome<SoftDtwGradientAnalysis> {
    gradient_incomplete(ledger, IncompleteReason::ArithmeticOverflow { resource })
}

fn gradient_allocation_incomplete(
    ledger: ResourceLedger,
    requested: usize,
) -> OperationOutcome<SoftDtwGradientAnalysis> {
    gradient_incomplete(
        ledger,
        IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested,
        },
    )
}

fn gradient_incomplete(
    ledger: ResourceLedger,
    reason: IncompleteReason,
) -> OperationOutcome<SoftDtwGradientAnalysis> {
    OperationOutcome::Incomplete {
        partial: None,
        reason,
        continuation: None,
        usage: ledger.usage(),
    }
}
