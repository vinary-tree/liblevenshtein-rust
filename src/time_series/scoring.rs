//! Canonical fail-closed scalar scoring adapters.
//!
//! Every adapter drives the same two-generation sparse online transition
//! engine used by dictionary products. Batch scoring therefore gains tagged
//! validation, checked allocation, deterministic resource ceilings, and
//! stack-safe execution without maintaining a second recurrence.

use crate::cost::CostMonoid;

use super::automaton::{
    ElasticOnlineAutomaton, OnlineAutomatonLimits, OnlineStepOutcome, TemporalAutomatonError,
};
use super::bounded::{
    ExactDecision, IncompleteReason, NoWitness, Operand, OperationOutcome, ResourceKind,
    ResourceLedger, ResourceLimits, TemporalValidationError,
};
use super::elastic::ElasticKernel;
use super::kernels::{DtwConfig, ErpConfig, FrechetConfig, MetricTwedConfig, TwedConfig};

fn bounded_native_distance<K>(
    kernel: K,
    query: &[f64],
    candidate: &[f64],
    cutoff: f64,
    limits: ResourceLimits,
) -> Result<OperationOutcome<ExactDecision>, TemporalValidationError>
where
    K: ElasticKernel,
    K::Monoid: CostMonoid<Cost = f64>,
{
    if cutoff.is_nan() || cutoff < 0.0 || (cutoff.is_infinite() && cutoff.is_sign_negative()) {
        return Err(TemporalValidationError::InvalidCutoff);
    }
    let kernel = kernel.normalized();
    let mut ledger = ResourceLedger::new(limits);
    ledger.validate_finite_series(Operand::Query, query)?;
    ledger.validate_finite_series(Operand::Candidate, candidate)?;

    // Structural reachability is a semantic result, not the numeric TOP
    // sentinel. Resolve it before empty-side and frontier arithmetic so both a
    // finite cutoff and an unbounded cutoff return the same exact tag.
    if !kernel.alignment_is_structurally_possible(query.len(), candidate.len()) {
        return Ok(OperationOutcome::Complete {
            value: ExactDecision::NoFiniteAlignment,
            usage: ledger.usage(),
        });
    }

    if query.is_empty() || candidate.is_empty() {
        let distance = if query.is_empty() && candidate.is_empty() {
            kernel.empty_pair_cost()
        } else {
            kernel.empty_vs_nonempty_cost(if query.is_empty() { candidate } else { query })
        };
        let value = if !distance.is_finite() {
            ExactDecision::NoFiniteAlignment
        } else if distance <= cutoff {
            ExactDecision::WithinCutoff {
                distance,
                witness: NoWitness,
            }
        } else {
            ExactDecision::AboveCutoff
        };
        return Ok(OperationOutcome::Complete {
            value,
            usage: ledger.usage(),
        });
    }

    let Some(dp_cells) = query.len().checked_add(1).and_then(|rows| {
        candidate
            .len()
            .checked_add(1)
            .and_then(|columns| rows.checked_mul(columns))
    }) else {
        return Ok(incomplete(
            ledger,
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::DpCells,
            },
        ));
    };
    if let Err(reason) = ledger.charge_many(&[
        (ResourceKind::DpCells, dp_cells),
        (ResourceKind::WorkUnits, dp_cells),
    ]) {
        return Ok(incomplete(ledger, reason));
    }

    let construction_cutoff = if cutoff.is_finite() { cutoff } else { f64::MAX };
    let online_limits = OnlineAutomatonLimits {
        max_query_len: limits.max_series_len,
        max_frontier_positions: limits.max_queue_entries,
        max_step_work_units: limits.max_work_units,
        max_scratch_bytes: limits.max_scratch_bytes,
    };
    let mut machine =
        match ElasticOnlineAutomaton::new(query, kernel, construction_cutoff, online_limits) {
            Ok(machine) => machine,
            Err(TemporalAutomatonError::Validation(error)) => return Err(error),
            Err(TemporalAutomatonError::Resource(reason)) => return Ok(incomplete(ledger, reason)),
        };
    if let Err(reason) = ledger.observe_peak(ResourceKind::ScratchBytes, machine.scratch_bytes()) {
        return Ok(incomplete(ledger, reason));
    }
    for sample in candidate {
        if let OnlineStepOutcome::Incomplete { reason, .. } = machine.advance(*sample)? {
            return Ok(incomplete(ledger, reason));
        }
    }
    let value = match machine.observation().distance_within_cutoff {
        Some(distance) if distance.is_finite() && distance <= cutoff => {
            ExactDecision::WithinCutoff {
                distance,
                witness: NoWitness,
            }
        }
        Some(_) | None if cutoff.is_finite() => ExactDecision::AboveCutoff,
        Some(_) | None => {
            return Ok(incomplete(ledger, IncompleteReason::NumericOverflow));
        }
    };
    Ok(OperationOutcome::Complete {
        value,
        usage: ledger.usage(),
    })
}

fn incomplete(ledger: ResourceLedger, reason: IncompleteReason) -> OperationOutcome<ExactDecision> {
    OperationOutcome::Incomplete {
        partial: None,
        reason,
        continuation: None,
        usage: ledger.usage(),
    }
}

impl ErpConfig {
    /// Exact tagged ERP cutoff decision using the sparse online recurrence.
    pub fn distance_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, TemporalValidationError> {
        bounded_native_distance(*self, query, candidate, cutoff, limits)
    }
}

impl TwedConfig {
    /// Exact tagged unit-grid TWED cutoff decision.
    pub fn distance_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, TemporalValidationError> {
        bounded_native_distance(*self, query, candidate, cutoff, limits)
    }
}

impl MetricTwedConfig {
    /// Exact tagged metric unit-grid TWED cutoff decision.
    pub fn distance_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, TemporalValidationError> {
        bounded_native_distance(*self.as_config(), query, candidate, cutoff, limits)
    }
}

impl FrechetConfig {
    /// Exact tagged scalar discrete-Fréchet cutoff decision.
    pub fn distance_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, TemporalValidationError> {
        bounded_native_distance(*self, query, candidate, cutoff, limits)
    }
}

impl DtwConfig {
    /// Exact tagged root-distance decision for explicitly banded DTW.
    pub fn distance_bounded(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: f64,
        limits: ResourceLimits,
    ) -> Result<OperationOutcome<ExactDecision>, TemporalValidationError> {
        if cutoff.is_nan() || cutoff < 0.0 || (cutoff.is_infinite() && cutoff.is_sign_negative()) {
            return Err(TemporalValidationError::InvalidCutoff);
        }
        if self.band > limits.max_band_width {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason: IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::BandWidth,
                    limit: limits.max_band_width,
                    requested: self.band,
                },
                continuation: None,
                usage: Default::default(),
            });
        }
        if !query.is_empty()
            && !candidate.is_empty()
            && query.len().abs_diff(candidate.len()) > self.band
        {
            let ledger = ResourceLedger::new(limits);
            ledger.validate_finite_series(Operand::Query, query)?;
            ledger.validate_finite_series(Operand::Candidate, candidate)?;
            return Ok(OperationOutcome::Complete {
                value: ExactDecision::NoFiniteAlignment,
                usage: ledger.usage(),
            });
        }
        let squared_cutoff = if cutoff.is_finite() {
            cutoff * cutoff
        } else {
            f64::INFINITY
        };
        let outcome = bounded_native_distance(*self, query, candidate, squared_cutoff, limits)?;
        Ok(match outcome {
            OperationOutcome::Complete {
                value: ExactDecision::WithinCutoff { distance, witness },
                usage,
            } => OperationOutcome::Complete {
                value: ExactDecision::WithinCutoff {
                    distance: distance.sqrt(),
                    witness,
                },
                usage,
            },
            OperationOutcome::Complete { value, usage } => {
                OperationOutcome::Complete { value, usage }
            }
            OperationOutcome::Incomplete {
                partial,
                reason,
                continuation,
                usage,
            } => OperationOutcome::Incomplete {
                partial,
                reason,
                continuation,
                usage,
            },
        })
    }
}
