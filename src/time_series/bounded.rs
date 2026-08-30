//! Fail-closed resource accounting and tagged outcomes for temporal operations.
//!
//! The legacy time-series APIs predate production resource contracts and use
//! `Option`, empty vectors, or infinity for several unrelated states.  This
//! module is the canonical boundary for new APIs: successful exhaustion is
//! impossible, all counters use preview-then-commit checked arithmetic, and a
//! complete empty result is structurally different from an incomplete one.

use std::fmt;

use thiserror::Error;

/// Operand named by a validation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// Query, or left-hand series for a scalar comparison.
    Query,
    /// Indexed candidate, or right-hand series for a scalar comparison.
    Candidate,
}

impl fmt::Display for Operand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Query => "query",
            Self::Candidate => "candidate",
        })
    }
}

/// Resource whose exact ceiling stopped an operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResourceKind {
    /// Number of samples in one input series.
    SeriesLength,
    /// Number of coordinates in one vector-valued sample.
    Dimension,
    /// Width of a constrained dynamic-programming band.
    BandWidth,
    /// Dynamic-programming cells evaluated or reserved.
    DpCells,
    /// Kernel- or traversal-defined units of computational work.
    WorkUnits,
    /// Peak bytes of temporary working storage.
    ScratchBytes,
    /// Trie nodes expanded during an indexed query.
    TrieNodes,
    /// Trie edges inspected during an indexed query.
    TrieEdges,
    /// Full-precision indexed candidates inspected.
    Candidates,
    /// Exact results retained by the operation.
    Results,
    /// Peak number of pending traversal queue entries.
    QueueEntries,
    /// Peak bytes retained by replayable alignment witnesses.
    WitnessBytes,
    /// Peak bytes retained by a resumable continuation.
    ContinuationBytes,
    /// Bytes read from or written to an exact persistent snapshot.
    SnapshotBytes,
}

/// Hard ceilings retained for the lifetime of one bounded operation/session.
///
/// Defaults are deliberately finite. Applications should normally lower them
/// to match a preregistered workload rather than treating them as quotas to
/// consume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Maximum samples in either operand.
    pub max_series_len: usize,
    /// Maximum coordinates in a vector-valued sample.
    pub max_dimension: usize,
    /// Maximum admissible dynamic-programming band width.
    pub max_band_width: usize,
    /// Maximum cumulative dynamic-programming cell charge.
    pub max_dp_cells: usize,
    /// Maximum cumulative logical work charge.
    pub max_work_units: usize,
    /// Maximum peak temporary-storage size in bytes.
    pub max_scratch_bytes: usize,
    /// Maximum cumulative trie-node visits.
    pub max_trie_nodes: usize,
    /// Maximum cumulative trie-edge visits.
    pub max_trie_edges: usize,
    /// Maximum cumulative full-precision candidate inspections.
    pub max_candidates: usize,
    /// Maximum exact results retained by the operation.
    pub max_results: usize,
    /// Maximum peak traversal queue length.
    pub max_queue_entries: usize,
    /// Maximum peak alignment-witness size in bytes.
    pub max_witness_bytes: usize,
    /// Maximum peak continuation size in bytes.
    pub max_continuation_bytes: usize,
    /// Maximum persistent-snapshot I/O size in bytes.
    pub max_snapshot_bytes: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_series_len: 1_000_000,
            max_dimension: 65_536,
            max_band_width: 1_000_000,
            max_dp_cells: 100_000_000,
            max_work_units: 200_000_000,
            max_scratch_bytes: 512 * 1024 * 1024,
            max_trie_nodes: 10_000_000,
            max_trie_edges: 20_000_000,
            max_candidates: 1_000_000,
            max_results: 100_000,
            max_queue_entries: 1_000_000,
            max_witness_bytes: 64 * 1024 * 1024,
            max_continuation_bytes: 64 * 1024 * 1024,
            max_snapshot_bytes: 1024 * 1024 * 1024,
        }
    }
}

impl ResourceLimits {
    #[inline]
    pub(crate) fn ceiling(self, resource: ResourceKind) -> usize {
        match resource {
            ResourceKind::SeriesLength => self.max_series_len,
            ResourceKind::Dimension => self.max_dimension,
            ResourceKind::BandWidth => self.max_band_width,
            ResourceKind::DpCells => self.max_dp_cells,
            ResourceKind::WorkUnits => self.max_work_units,
            ResourceKind::ScratchBytes => self.max_scratch_bytes,
            ResourceKind::TrieNodes => self.max_trie_nodes,
            ResourceKind::TrieEdges => self.max_trie_edges,
            ResourceKind::Candidates => self.max_candidates,
            ResourceKind::Results => self.max_results,
            ResourceKind::QueueEntries => self.max_queue_entries,
            ResourceKind::WitnessBytes => self.max_witness_bytes,
            ResourceKind::ContinuationBytes => self.max_continuation_bytes,
            ResourceKind::SnapshotBytes => self.max_snapshot_bytes,
        }
    }
}

/// Per-call slice for a resumable operation.
///
/// Session-wide hard ceilings remain in [`ResourceLimits`]. Reaching either
/// page ceiling pauses before the next externally visible unit of work and
/// returns a continuation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageBudget {
    /// Maximum logical work charged by one call.
    pub max_work_units: usize,
    /// Maximum newly accepted results produced by one call.
    pub max_results: usize,
}

impl Default for PageBudget {
    fn default() -> Self {
        Self {
            max_work_units: 100_000,
            max_results: 1_000,
        }
    }
}

/// Deterministic charged usage. A charge may represent work already performed
/// or capacity reserved before a non-resumable operation begins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceUsage {
    /// Cumulative dynamic-programming cells charged.
    pub dp_cells: usize,
    /// Cumulative logical work units charged.
    pub work_units: usize,
    /// Peak temporary-storage size observed in bytes.
    pub scratch_bytes: usize,
    /// Cumulative trie nodes visited.
    pub trie_nodes: usize,
    /// Cumulative trie edges visited.
    pub trie_edges: usize,
    /// Cumulative full-precision candidates inspected.
    pub candidates: usize,
    /// Cumulative exact results retained.
    pub results: usize,
    /// Peak traversal queue length observed.
    pub queue_entries: usize,
    /// Peak alignment-witness size observed in bytes.
    pub witness_bytes: usize,
    /// Peak continuation size observed in bytes.
    pub continuation_bytes: usize,
    /// Cumulative persistent-snapshot bytes charged.
    pub snapshot_bytes: usize,
}

impl ResourceUsage {
    #[inline]
    fn value(self, resource: ResourceKind) -> usize {
        match resource {
            ResourceKind::SeriesLength | ResourceKind::Dimension | ResourceKind::BandWidth => 0,
            ResourceKind::DpCells => self.dp_cells,
            ResourceKind::WorkUnits => self.work_units,
            ResourceKind::ScratchBytes => self.scratch_bytes,
            ResourceKind::TrieNodes => self.trie_nodes,
            ResourceKind::TrieEdges => self.trie_edges,
            ResourceKind::Candidates => self.candidates,
            ResourceKind::Results => self.results,
            ResourceKind::QueueEntries => self.queue_entries,
            ResourceKind::WitnessBytes => self.witness_bytes,
            ResourceKind::ContinuationBytes => self.continuation_bytes,
            ResourceKind::SnapshotBytes => self.snapshot_bytes,
        }
    }

    #[inline]
    fn set(&mut self, resource: ResourceKind, value: usize) {
        match resource {
            ResourceKind::SeriesLength | ResourceKind::Dimension | ResourceKind::BandWidth => {}
            ResourceKind::DpCells => self.dp_cells = value,
            ResourceKind::WorkUnits => self.work_units = value,
            ResourceKind::ScratchBytes => self.scratch_bytes = value,
            ResourceKind::TrieNodes => self.trie_nodes = value,
            ResourceKind::TrieEdges => self.trie_edges = value,
            ResourceKind::Candidates => self.candidates = value,
            ResourceKind::Results => self.results = value,
            ResourceKind::QueueEntries => self.queue_entries = value,
            ResourceKind::WitnessBytes => self.witness_bytes = value,
            ResourceKind::ContinuationBytes => self.continuation_bytes = value,
            ResourceKind::SnapshotBytes => self.snapshot_bytes = value,
        }
    }
}

/// Reason a valid request did not complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum IncompleteReason {
    /// A checked charge would exceed its configured ceiling.
    BudgetExceeded {
        /// Resource whose ceiling would be exceeded.
        resource: ResourceKind,
        /// Configured inclusive ceiling.
        limit: usize,
        /// Total charge that would have resulted.
        requested: usize,
    },
    /// Computing a resource charge overflowed `usize`.
    ArithmeticOverflow {
        /// Resource whose charge could not be represented.
        resource: ResourceKind,
    },
    /// Floating-point arithmetic could not represent a finite exact result.
    NumericOverflow,
    /// The allocator could not reserve a validated, bounded working buffer.
    AllocationFailed {
        /// Resource class for the requested allocation.
        resource: ResourceKind,
        /// Requested logical size in bytes or entries, depending on `resource`.
        requested: usize,
    },
    /// Indexed state violated an invariant required by exact verification.
    InvalidStoredData,
    /// The caller explicitly cancelled the operation.
    Cancelled,
    /// The requested bounded operation is not implemented by this kernel.
    Unsupported,
}

/// Validation failures are request errors, never empty results.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TemporalValidationError {
    /// An operand exceeded the configured sample-count ceiling.
    #[error("{operand} length {len} exceeds configured limit {limit}")]
    SeriesTooLong {
        /// Operand that exceeded the limit.
        operand: Operand,
        /// Observed sample count.
        len: usize,
        /// Configured inclusive sample-count ceiling.
        limit: usize,
    },
    /// An operand contained a NaN or infinite sample.
    #[error("{operand} sample {index} is not finite")]
    NonFiniteSample {
        /// Operand containing the invalid sample.
        operand: Operand,
        /// Zero-based index of the invalid sample.
        index: usize,
    },
    /// A cutoff was negative or NaN.
    #[error("cutoff must be finite and nonnegative, or positive infinity")]
    InvalidCutoff,
    /// A kernel parameter violated its documented domain.
    #[error("invalid temporal configuration: {0}")]
    InvalidConfiguration(&'static str),
    /// A metric-only API received an empty series outside its domain.
    #[error("empty series is outside this metric domain")]
    EmptyMetricSeries,
    /// A vector-valued sample exceeded the configured dimension limit.
    #[error("dimension {dimension} exceeds configured limit {limit}")]
    DimensionTooLarge {
        /// Observed coordinate count.
        dimension: usize,
        /// Configured inclusive coordinate-count ceiling.
        limit: usize,
    },
    /// Corresponding vector-valued samples had different dimensions.
    #[error("series dimensions do not agree")]
    DimensionMismatch,
}

/// Result of a bounded operation.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub enum OperationOutcome<T, C = ()> {
    /// The operation exhausted its exact search space successfully.
    Complete {
        /// Complete exact value, including a possibly empty result collection.
        value: T,
        /// Final cumulative and peak resource accounting.
        usage: ResourceUsage,
    },
    /// The operation stopped without proving exhaustive completion.
    Incomplete {
        /// Exact subset or intermediate value available so far, if exposed.
        partial: Option<T>,
        /// Fail-closed reason the operation stopped or paused.
        reason: IncompleteReason,
        /// State with which the same operation may be resumed, when resumable.
        continuation: Option<C>,
        /// Cumulative and peak resource accounting at the stop point.
        usage: ResourceUsage,
    },
}

impl<T, C> OperationOutcome<T, C> {
    /// Return cumulative and peak resource accounting for either outcome.
    #[inline]
    pub fn usage(&self) -> ResourceUsage {
        match self {
            Self::Complete { usage, .. } | Self::Incomplete { usage, .. } => *usage,
        }
    }

    /// Return whether the operation proved exhaustive completion.
    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete { .. })
    }
}

/// Zero-sized marker used by exact score-only calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoWitness;

/// Exact cutoff decision. `AboveCutoff` is not an error, while
/// `NoFiniteAlignment` records a mathematically unreachable constrained path.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub enum ExactDecision<W = NoWitness> {
    /// A finite exact distance satisfied the inclusive cutoff.
    ///
    /// `witness` is replayable when a witness-producing scorer is used.
    WithinCutoff {
        /// Exact finite distance between the two operands.
        distance: f64,
        /// Optional proof object that replays this exact distance.
        witness: W,
    },
    /// Every finite exact alignment costs more than the cutoff.
    AboveCutoff,
    /// The constrained alignment domain contains no finite alignment.
    NoFiniteAlignment,
}

/// Checked preview-then-commit resource ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLedger {
    limits: ResourceLimits,
    usage: ResourceUsage,
}

impl ResourceLedger {
    /// Create an empty ledger governed by `limits`.
    #[inline]
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            limits,
            usage: ResourceUsage::default(),
        }
    }

    /// Return the immutable ceilings governing this ledger.
    #[inline]
    pub fn limits(&self) -> ResourceLimits {
        self.limits
    }

    /// Return the charges committed so far.
    #[inline]
    pub fn usage(&self) -> ResourceUsage {
        self.usage
    }

    /// Charge a cumulative resource without partially mutating on failure.
    pub fn charge(
        &mut self,
        resource: ResourceKind,
        amount: usize,
    ) -> Result<(), IncompleteReason> {
        debug_assert!(!matches!(
            resource,
            ResourceKind::SeriesLength | ResourceKind::Dimension | ResourceKind::BandWidth
        ));
        let current = self.usage.value(resource);
        let Some(requested) = current.checked_add(amount) else {
            return Err(IncompleteReason::ArithmeticOverflow { resource });
        };
        let limit = self.limits.ceiling(resource);
        if requested > limit {
            return Err(IncompleteReason::BudgetExceeded {
                resource,
                limit,
                requested,
            });
        }
        self.usage.set(resource, requested);
        Ok(())
    }

    /// Atomically preview and commit a related set of charges.
    pub fn charge_many(
        &mut self,
        charges: &[(ResourceKind, usize)],
    ) -> Result<(), IncompleteReason> {
        let mut preview = *self;
        for &(resource, amount) in charges {
            preview.charge(resource, amount)?;
        }
        self.usage = preview.usage;
        Ok(())
    }

    /// Observe a retained/peak resource without cumulative double-counting.
    pub fn observe_peak(
        &mut self,
        resource: ResourceKind,
        amount: usize,
    ) -> Result<(), IncompleteReason> {
        debug_assert!(matches!(
            resource,
            ResourceKind::ScratchBytes
                | ResourceKind::QueueEntries
                | ResourceKind::WitnessBytes
                | ResourceKind::ContinuationBytes
                | ResourceKind::SnapshotBytes
        ));
        let limit = self.limits.ceiling(resource);
        if amount > limit {
            return Err(IncompleteReason::BudgetExceeded {
                resource,
                limit,
                requested: amount,
            });
        }
        if amount > self.usage.value(resource) {
            self.usage.set(resource, amount);
        }
        Ok(())
    }

    /// Validate one operand's sample count against its hard ceiling.
    pub fn validate_series_len(
        &self,
        operand: Operand,
        len: usize,
    ) -> Result<(), TemporalValidationError> {
        let limit = self.limits.max_series_len;
        if len > limit {
            return Err(TemporalValidationError::SeriesTooLong {
                operand,
                len,
                limit,
            });
        }
        Ok(())
    }

    /// Validate one scalar series' length and finite-value invariant.
    pub fn validate_finite_series(
        &self,
        operand: Operand,
        series: &[f64],
    ) -> Result<(), TemporalValidationError> {
        self.validate_series_len(operand, series.len())?;
        if let Some(index) = series.iter().position(|sample| !sample.is_finite()) {
            return Err(TemporalValidationError::NonFiniteSample { operand, index });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_limit_is_accepted_and_one_past_is_atomic() {
        let limits = ResourceLimits {
            max_work_units: 7,
            ..ResourceLimits::default()
        };
        let mut ledger = ResourceLedger::new(limits);
        ledger.charge(ResourceKind::WorkUnits, 7).unwrap();
        let before = ledger.usage();
        assert_eq!(
            ledger.charge(ResourceKind::WorkUnits, 1),
            Err(IncompleteReason::BudgetExceeded {
                resource: ResourceKind::WorkUnits,
                limit: 7,
                requested: 8,
            })
        );
        assert_eq!(ledger.usage(), before);
    }

    #[test]
    fn arithmetic_overflow_does_not_mutate_usage() {
        let limits = ResourceLimits {
            max_work_units: usize::MAX,
            ..ResourceLimits::default()
        };
        let mut ledger = ResourceLedger::new(limits);
        ledger.charge(ResourceKind::WorkUnits, usize::MAX).unwrap();
        let before = ledger.usage();
        assert_eq!(
            ledger.charge(ResourceKind::WorkUnits, 1),
            Err(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::WorkUnits,
            })
        );
        assert_eq!(ledger.usage(), before);
    }

    #[test]
    fn related_charges_commit_atomically() {
        let limits = ResourceLimits {
            max_dp_cells: 10,
            max_work_units: 4,
            ..ResourceLimits::default()
        };
        let mut ledger = ResourceLedger::new(limits);
        assert!(ledger
            .charge_many(&[(ResourceKind::DpCells, 10), (ResourceKind::WorkUnits, 5),])
            .is_err());
        assert_eq!(ledger.usage(), ResourceUsage::default());
    }

    #[test]
    fn complete_empty_and_incomplete_empty_are_distinct() {
        let complete: OperationOutcome<Vec<u8>> = OperationOutcome::Complete {
            value: Vec::new(),
            usage: ResourceUsage::default(),
        };
        let incomplete: OperationOutcome<Vec<u8>> = OperationOutcome::Incomplete {
            partial: Some(Vec::new()),
            reason: IncompleteReason::Cancelled,
            continuation: None,
            usage: ResourceUsage::default(),
        };
        assert!(complete.is_complete());
        assert!(!incomplete.is_complete());
        assert_ne!(complete, incomplete);
    }
}
