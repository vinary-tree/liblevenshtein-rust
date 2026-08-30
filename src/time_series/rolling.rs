//! Stable bounded rolling windows for resumable exact temporal retrieval.

use super::automaton::TemporalAutomatonError;
use super::bounded::{
    IncompleteReason, Operand, PageBudget, ResourceKind, ResourceLimits, ResourceUsage,
    TemporalValidationError,
};
use super::elastic::{BoundedRangeOutcome, Cost, ElasticKernel, ElasticTransducer};

/// Owned preregistered query window emitted from an unknown-length stream.
#[derive(Clone, Debug, PartialEq)]
pub struct RollingWindowSnapshot {
    window_id: u64,
    start_offset: u64,
    end_offset: u64,
    values: Box<[f64]>,
}

impl RollingWindowSnapshot {
    /// Monotone zero-based ID of this emitted window.
    #[inline]
    pub fn window_id(&self) -> u64 {
        self.window_id
    }

    /// Inclusive stream offset of the first sample in the window.
    #[inline]
    pub fn start_offset(&self) -> u64 {
        self.start_offset
    }

    /// Exclusive stream offset immediately after the window.
    #[inline]
    pub fn end_offset(&self) -> u64 {
        self.end_offset
    }

    /// Exact chronological samples in the fixed-width window.
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Start or resume exact range retrieval for this immutable window.
    ///
    /// The returned continuation owns its query copy and immutably borrows the
    /// index revision, so paging cannot silently switch dictionaries or
    /// windows. An incomplete empty partial is never evidence of absence.
    pub fn search_range_bounded<'a, K, V>(
        &self,
        index: &'a ElasticTransducer<K, V>,
        cutoff: Cost<K>,
        limits: ResourceLimits,
        page: PageBudget,
    ) -> Result<BoundedRangeOutcome<'a, K, V>, TemporalValidationError>
    where
        K: ElasticKernel,
        V: Eq + std::hash::Hash + Clone,
    {
        index.search_range_bounded(&self.values, cutoff, limits, page)
    }
}

/// Result of consuming one rolling-stream sample.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub enum RollingWindowStep {
    /// The sample was committed; `snapshot` is present exactly at a configured
    /// emission boundary.
    Advanced {
        /// Newly emitted immutable window, if any.
        snapshot: Option<RollingWindowSnapshot>,
        /// Per-step bounded work and retained/snapshot bytes.
        usage: ResourceUsage,
    },
    /// The sample was not committed.
    Incomplete {
        /// Fail-closed resource reason.
        reason: IncompleteReason,
        /// Resources observed before the transactional stop.
        usage: ResourceUsage,
    },
}

/// Fixed-capacity circular rolling-query machine.
///
/// The machine emits the first snapshot when `window_len` samples are present
/// and subsequent snapshots every `stride` samples. It retains exactly one
/// circular window regardless of stream length. Snapshot allocation is
/// prevalidated and occurs before committing the triggering sample.
#[derive(Debug)]
pub struct BoundedRollingWindow {
    storage: Vec<f64>,
    window_len: usize,
    stride: usize,
    write_index: usize,
    retained_len: usize,
    consumed: u64,
    emitted: u64,
    since_emit: usize,
    scratch_bytes: usize,
}

impl BoundedRollingWindow {
    /// Construct a positive-width, positive-stride rolling machine.
    pub fn new(
        window_len: usize,
        stride: usize,
        limits: ResourceLimits,
    ) -> Result<Self, TemporalAutomatonError> {
        if window_len == 0 {
            return Err(TemporalValidationError::InvalidConfiguration(
                "rolling window length must be positive",
            )
            .into());
        }
        if stride == 0 {
            return Err(TemporalValidationError::InvalidConfiguration(
                "rolling window stride must be positive",
            )
            .into());
        }
        if window_len > limits.max_series_len {
            return Err(TemporalValidationError::SeriesTooLong {
                operand: Operand::Query,
                len: window_len,
                limit: limits.max_series_len,
            }
            .into());
        }
        let scratch_bytes = window_len.checked_mul(std::mem::size_of::<f64>()).ok_or(
            TemporalAutomatonError::Resource(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            }),
        )?;
        if scratch_bytes > limits.max_scratch_bytes {
            return Err(TemporalAutomatonError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested: scratch_bytes,
                },
            ));
        }
        if scratch_bytes > limits.max_snapshot_bytes {
            return Err(TemporalAutomatonError::Resource(
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::SnapshotBytes,
                    limit: limits.max_snapshot_bytes,
                    requested: scratch_bytes,
                },
            ));
        }
        let mut storage = Vec::new();
        storage.try_reserve_exact(window_len).map_err(|_| {
            TemporalAutomatonError::Resource(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested: scratch_bytes,
            })
        })?;
        storage.resize(window_len, 0.0);
        Ok(Self {
            storage,
            window_len,
            stride,
            write_index: 0,
            retained_len: 0,
            consumed: 0,
            emitted: 0,
            since_emit: 0,
            scratch_bytes,
        })
    }

    /// Consume one finite stream sample transactionally.
    pub fn advance(&mut self, sample: f64) -> Result<RollingWindowStep, TemporalValidationError> {
        if !sample.is_finite() {
            return Err(TemporalValidationError::NonFiniteSample {
                operand: Operand::Candidate,
                index: usize::try_from(self.consumed).unwrap_or(usize::MAX),
            });
        }
        let next_consumed = match self.consumed.checked_add(1) {
            Some(consumed) => consumed,
            None => {
                return Ok(RollingWindowStep::Incomplete {
                    reason: IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::SeriesLength,
                    },
                    usage: self.usage(0, 0),
                });
            }
        };
        let next_retained = self.retained_len.checked_add(1).ok_or(
            TemporalValidationError::InvalidConfiguration(
                "rolling retained length overflowed its fixed window",
            ),
        )?;
        let next_since_emit = self.since_emit.checked_add(1);
        let first_emit = next_retained == self.window_len && self.emitted == 0;
        let repeated_emit = self.emitted > 0 && next_since_emit == Some(self.stride);
        let will_emit = first_emit || repeated_emit;
        let next_emitted = if will_emit {
            match self.emitted.checked_add(1) {
                Some(emitted) => emitted,
                None => {
                    return Ok(RollingWindowStep::Incomplete {
                        reason: IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::Results,
                        },
                        usage: self.usage(0, 0),
                    });
                }
            }
        } else {
            self.emitted
        };
        let committed_since_emit = if will_emit {
            0
        } else if self.emitted > 0 {
            match next_since_emit {
                Some(since) => since,
                None => {
                    return Ok(RollingWindowStep::Incomplete {
                        reason: IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::SeriesLength,
                        },
                        usage: self.usage(0, 0),
                    });
                }
            }
        } else {
            0
        };

        let snapshot = if will_emit {
            let mut values = Vec::new();
            if values.try_reserve_exact(self.window_len).is_err() {
                return Ok(RollingWindowStep::Incomplete {
                    reason: IncompleteReason::AllocationFailed {
                        resource: ResourceKind::SnapshotBytes,
                        requested: self.scratch_bytes,
                    },
                    usage: self.usage(0, 0),
                });
            }
            if self.retained_len < self.window_len {
                values.extend_from_slice(&self.storage[..self.retained_len]);
            } else {
                for offset in 1..self.window_len {
                    let remaining = self.window_len - self.write_index;
                    let index = if offset >= remaining {
                        offset - remaining
                    } else {
                        self.write_index + offset
                    };
                    values.push(self.storage[index]);
                }
            }
            values.push(sample);
            debug_assert_eq!(values.len(), self.window_len);
            let window_width = u64::try_from(self.window_len).map_err(|_| {
                TemporalValidationError::InvalidConfiguration(
                    "rolling window length is not representable as a stream offset",
                )
            })?;
            Some(RollingWindowSnapshot {
                window_id: self.emitted,
                start_offset: next_consumed - window_width,
                end_offset: next_consumed,
                values: values.into_boxed_slice(),
            })
        } else {
            None
        };

        self.storage[self.write_index] = sample;
        self.write_index = (self.write_index + 1) % self.window_len;
        self.retained_len = next_retained.min(self.window_len);
        self.consumed = next_consumed;
        self.emitted = next_emitted;
        self.since_emit = committed_since_emit;
        let usage = self.usage(usize::from(will_emit), 1);
        Ok(RollingWindowStep::Advanced { snapshot, usage })
    }

    /// Fixed retained logical bytes, independent of `consumed_samples`.
    #[inline]
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }

    /// Total committed stream samples.
    #[inline]
    pub fn consumed_samples(&self) -> u64 {
        self.consumed
    }

    fn usage(&self, emitted: usize, work: usize) -> ResourceUsage {
        ResourceUsage {
            work_units: work,
            scratch_bytes: self.scratch_bytes,
            snapshot_bytes: emitted.saturating_mul(self.scratch_bytes),
            queue_entries: self.retained_len,
            ..ResourceUsage::default()
        }
    }
}
