//! Exact, bounded dictionary products for physical-timestamp TWED.
//!
//! Each `u64` edge denotes one whole `(value interval, time interval, unit)`
//! label. The query automaton is constructed only along dictionary edges that
//! are actually observed. Its interval-relaxed DP residuals are interned by
//! exact bitwise equality and referenced from the iterative DFS by compact
//! identifiers. Quantization is used only to prove admissible subtree lower
//! bounds: every surviving collision-bucket member is verified from its
//! retained full-precision [`TimestampedSeries`].
//!
//! A [`OperationOutcome::Complete`](super::OperationOutcome::Complete) result
//! means that the immutable captured dictionary revision was exhausted. An
//! incomplete empty partial is never evidence of absence.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem::size_of;

use libdictenstein::dynamic_dawg::{DynamicDawgU64, DynamicDawgU64Node};
use libdictenstein::{DictionaryNode, DictionaryTraversalRoot, SnapshotNodeIdentity};
use smallvec::SmallVec;
use thiserror::Error;

use crate::transducer::dictionary_traversal::{DfsNodeEdges, TraversalCursor, TraversalSession};

use super::bounded::{
    IncompleteReason, Operand, OperationOutcome, PageBudget, ResourceKind, ResourceLedger,
    ResourceLimits, ResourceUsage, TemporalValidationError,
};
use super::timestamped_twed::{
    delete_cost, match_cost, MetricTimestampedTwedConfig, TimestampUnit, TimestampedScalarBox,
    TimestampedSeries, TimestampedTwedError,
};

const LABEL_BITS: u32 = 31;
const LABEL_MASK: u64 = (1_u64 << LABEL_BITS) - 1;
const MAX_BINS: u32 = 1_u32 << LABEL_BITS;

/// A fixed, typed quantizer for physical-timestamp dictionary labels.
///
/// The high two token bits encode the canonical unit, the next 31 encode the
/// value bin, and the low 31 encode the time bin. Thus dictionary equality is
/// equality of the entire typed label, never equality of a scalar projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampedTwedQuantizer {
    unit: TimestampUnit,
    origin: f64,
    value_domain: (f64, f64),
    time_domain: (f64, f64),
    value_bins: u32,
    time_bins: u32,
}

impl TimestampedTwedQuantizer {
    /// Construct a finite fixed-grid quantizer.
    ///
    /// Both domains must have strictly positive width, the time domain must
    /// begin no earlier than `origin`, and each bin count must lie in
    /// $`[1,2^{31}]`$.
    pub fn try_new(
        unit: TimestampUnit,
        origin: f64,
        value_domain: (f64, f64),
        time_domain: (f64, f64),
        value_bins: u32,
        time_bins: u32,
    ) -> Result<Self, TimestampedTwedIndexError> {
        if !origin.is_finite() {
            return Err(TimestampedTwedIndexError::InvalidQuantizer(
                "origin must be finite",
            ));
        }
        if !finite_positive_width(value_domain) {
            return Err(TimestampedTwedIndexError::InvalidQuantizer(
                "value domain must be finite and have positive width",
            ));
        }
        if !finite_positive_width(time_domain) || time_domain.0 < origin {
            return Err(TimestampedTwedIndexError::InvalidQuantizer(
                "time domain must be finite, have positive width, and follow the origin",
            ));
        }
        if value_bins == 0 || time_bins == 0 || value_bins > MAX_BINS || time_bins > MAX_BINS {
            return Err(TimestampedTwedIndexError::InvalidQuantizer(
                "bin counts must lie in [1, 2^31]",
            ));
        }
        Ok(Self {
            unit,
            origin,
            value_domain,
            time_domain,
            value_bins,
            time_bins,
        })
    }

    /// Return the canonical unit carried by every label.
    #[inline]
    pub fn unit(self) -> TimestampUnit {
        self.unit
    }

    /// Return the shared physical origin.
    #[inline]
    pub fn origin(self) -> f64 {
        self.origin
    }

    /// Return the finite scalar-value domain.
    #[inline]
    pub fn value_domain(self) -> (f64, f64) {
        self.value_domain
    }

    /// Return the finite timestamp domain.
    #[inline]
    pub fn time_domain(self) -> (f64, f64) {
        self.time_domain
    }

    /// Encode one concrete `(value,time,unit)` point as a whole typed label.
    pub fn encode(&self, value: f64, time: f64) -> Result<u64, TimestampedTwedIndexError> {
        let value_bin = bin_index(value, self.value_domain, self.value_bins)
            .ok_or(TimestampedTwedIndexError::PointOutsideQuantizerDomain)?;
        let time_bin = bin_index(time, self.time_domain, self.time_bins)
            .ok_or(TimestampedTwedIndexError::PointOutsideQuantizerDomain)?;
        Ok(
            (unit_tag(self.unit) << 62)
                | (u64::from(value_bin) << LABEL_BITS)
                | u64::from(time_bin),
        )
    }

    /// Decode one token into its conservative closed scalar/time box.
    pub fn decode(&self, token: u64) -> Result<TimestampedScalarBox, TimestampedTwedIndexError> {
        if token >> 62 != unit_tag(self.unit) {
            return Err(TimestampedTwedIndexError::InvalidStoredToken);
        }
        let value_bin = ((token >> LABEL_BITS) & LABEL_MASK) as u32;
        let time_bin = (token & LABEL_MASK) as u32;
        if value_bin >= self.value_bins || time_bin >= self.time_bins {
            return Err(TimestampedTwedIndexError::InvalidStoredToken);
        }
        let value = bin_interval(self.value_domain, self.value_bins, value_bin);
        let time = bin_interval(self.time_domain, self.time_bins, time_bin);
        TimestampedScalarBox::try_new(value, time, self.unit).map_err(Into::into)
    }
}

/// Hard product-specific ceilings in addition to the common resource ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimestampedTwedProductLimits {
    /// Common cumulative and peak ceilings.
    pub resources: ResourceLimits,
    /// Maximum exact query-local residuals retained in the append-only arena.
    pub max_product_states: usize,
    /// Maximum canonical recurrence positions retained across all residuals.
    pub max_product_positions: usize,
    /// Maximum observed `(state,label)` transitions retained as an optimization.
    pub max_transition_cache_entries: usize,
}

impl Default for TimestampedTwedProductLimits {
    fn default() -> Self {
        Self {
            resources: ResourceLimits::default(),
            max_product_states: 1_000_000,
            max_product_positions: 8_000_000,
            max_transition_cache_entries: 2_000_000,
        }
    }
}

/// Construction, insertion, or request-validation failure.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TimestampedTwedIndexError {
    /// The typed quantizer configuration is not meaningful.
    #[error("invalid timestamped TWED quantizer: {0}")]
    InvalidQuantizer(&'static str),
    /// A value or timestamp lies outside the fixed quantizer domain.
    #[error("point lies outside the timestamped TWED quantizer domain")]
    PointOutsideQuantizerDomain,
    /// A series does not use the index's canonical unit.
    #[error("series timestamp unit differs from the index unit")]
    MixedUnits,
    /// A series does not use the index's immutable physical origin.
    #[error("series timestamp origin differs from the index origin")]
    MixedOrigins,
    /// A cutoff was NaN or negative.
    #[error("cutoff must be nonnegative and not NaN")]
    InvalidCutoff,
    /// A token reachable from the captured dictionary violates the encoding.
    #[error("captured dictionary contains an invalid typed token")]
    InvalidStoredToken,
    /// A stable episode identifier could not be incremented.
    #[error("stable episode identifier space is exhausted")]
    EpisodeIdOverflow,
    /// A bounded allocation failed while constructing or inserting data.
    #[error("resource construction failed: {0:?}")]
    Resource(IncompleteReason),
    /// Explicit-timestamp series validation failed.
    #[error(transparent)]
    Timestamped(#[from] TimestampedTwedError),
    /// Shared temporal request validation failed.
    #[error(transparent)]
    Validation(#[from] TemporalValidationError),
}

/// One exact member of a completed or partial range result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampedTwedRangeMatch<'a, V> {
    /// Stable insertion-order identifier used for deterministic ties.
    pub episode_id: u64,
    /// Caller metadata retained at full precision.
    pub value: &'a V,
    /// Original physical-timestamp series used by exact verification.
    pub series: &'a TimestampedSeries,
    /// Exact physical-timestamp TWED distance.
    pub distance: f64,
}

/// Retained-state diagnostics for the lazy dictionary product.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimestampedTwedProductStats {
    /// Live explicit DFS frames.
    pub frames: usize,
    /// Distinct exact residuals in the query-local arena.
    pub states: usize,
    /// Canonical recurrence positions retained by those residuals.
    pub positions: usize,
    /// Observed transitions retained in the bounded cache.
    pub cached_transitions: usize,
    /// Exact residuals that reused an earlier compact identifier.
    pub reused_states: usize,
}

struct StoredEpisode<V> {
    episode_id: u64,
    value: V,
    series: TimestampedSeries,
}

/// Ephemeral exact index over typed physical-timestamp sequences.
///
/// `DynamicDawgU64` stores quantized paths and a bucket identifier at each
/// terminal. Buckets retain every colliding original; quantization therefore
/// affects pruning work, never membership or returned distances.
pub struct TimestampedTwedIndex<V> {
    dictionary: DynamicDawgU64<usize>,
    buckets: Vec<Vec<StoredEpisode<V>>>,
    quantizer: TimestampedTwedQuantizer,
    config: MetricTimestampedTwedConfig,
    next_episode_id: u64,
    len: usize,
}

impl<V> TimestampedTwedIndex<V> {
    /// Construct an empty ephemeral exact index.
    pub fn new(quantizer: TimestampedTwedQuantizer, config: MetricTimestampedTwedConfig) -> Self {
        Self {
            dictionary: DynamicDawgU64::new(),
            buckets: Vec::new(),
            quantizer,
            config,
            next_episode_id: 0,
            len: 0,
        }
    }

    /// Return the number of retained full-precision episodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether no full-precision episodes are retained.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the typed quantizer defining dictionary labels.
    #[inline]
    pub fn quantizer(&self) -> TimestampedTwedQuantizer {
        self.quantizer
    }

    /// Insert one full-precision episode and return its stable identifier.
    ///
    /// This is a construction API, not a bounded query. Callers should build
    /// the index before admitting queries and enforce their own ingestion
    /// memory policy. Query-side allocations are fallible and fail closed.
    pub fn insert(
        &mut self,
        value: V,
        series: TimestampedSeries,
    ) -> Result<u64, TimestampedTwedIndexError> {
        self.validate_series_identity(&series)?;
        let token_bytes = series.values().len().checked_mul(size_of::<u64>()).ok_or(
            TimestampedTwedIndexError::Resource(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            }),
        )?;
        let mut tokens = Vec::new();
        tokens
            .try_reserve_exact(series.values().len())
            .map_err(|_| {
                TimestampedTwedIndexError::Resource(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: token_bytes,
                })
            })?;
        for (&sample, &timestamp) in series.values().iter().zip(series.timestamps()) {
            tokens.push(self.quantizer.encode(sample, timestamp)?);
        }

        let episode_id = self.next_episode_id;
        let next_episode_id = episode_id
            .checked_add(1)
            .ok_or(TimestampedTwedIndexError::EpisodeIdOverflow)?;
        let next_len = self
            .len
            .checked_add(1)
            .ok_or(TimestampedTwedIndexError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::Candidates,
                },
            ))?;

        if let Some(bucket_id) = self.dictionary.get_sequence_value(&tokens) {
            let bucket =
                self.buckets
                    .get_mut(bucket_id)
                    .ok_or(TimestampedTwedIndexError::Resource(
                        IncompleteReason::InvalidStoredData,
                    ))?;
            bucket.try_reserve(1).map_err(|_| {
                TimestampedTwedIndexError::Resource(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: bucket.len().saturating_add(1),
                })
            })?;
            bucket.push(StoredEpisode {
                episode_id,
                value,
                series,
            });
        } else {
            self.buckets.try_reserve(1).map_err(|_| {
                TimestampedTwedIndexError::Resource(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: self.buckets.len().saturating_add(1),
                })
            })?;
            let bucket_id = self.buckets.len();
            let mut bucket = Vec::new();
            bucket.try_reserve_exact(1).map_err(|_| {
                TimestampedTwedIndexError::Resource(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: 1,
                })
            })?;
            bucket.push(StoredEpisode {
                episode_id,
                value,
                series,
            });
            self.buckets.push(bucket);
            let inserted = self
                .dictionary
                .insert_sequence_with_value(&tokens, bucket_id);
            if !inserted {
                return Err(TimestampedTwedIndexError::Resource(
                    IncompleteReason::InvalidStoredData,
                ));
            }
        }
        self.next_episode_id = next_episode_id;
        self.len = next_len;
        Ok(episode_id)
    }

    /// Start a strict, resumable exact range query.
    pub fn search_range_bounded<'a>(
        &'a self,
        query: &'a TimestampedSeries,
        cutoff: f64,
        limits: TimestampedTwedProductLimits,
        page: PageBudget,
    ) -> Result<TimestampedTwedRangeOutcome<'a, V>, TimestampedTwedIndexError> {
        self.validate_series_identity(query)?;
        ResourceLedger::new(limits.resources)
            .validate_series_len(Operand::Query, query.values().len())?;
        if cutoff.is_nan() || cutoff < 0.0 {
            return Err(TimestampedTwedIndexError::InvalidCutoff);
        }
        let continuation = TimestampedTwedRangeContinuation::start(self, query, cutoff, limits);
        match continuation {
            Ok(continuation) => Ok(continuation.resume(page)),
            Err((reason, ledger)) => Ok(OperationOutcome::Incomplete {
                partial: None,
                reason,
                continuation: None,
                usage: ledger.usage(),
            }),
        }
    }

    /// Exact bounded `k`-nearest-neighbour search over physical timestamps.
    ///
    /// This strict surface performs a deterministic full-precision scan with
    /// one reusable pair of DP generations and a size-`k` max heap. It is the
    /// fail-closed complement of the lazy range product: no approximate index
    /// can prove completeness, and exceeding any candidate, cell, work,
    /// scratch, or result ceiling returns `Incomplete` rather than a truncated
    /// nearest-neighbour claim.
    pub fn search_knn_bounded<'a>(
        &'a self,
        query: &'a TimestampedSeries,
        k: usize,
        limits: ResourceLimits,
    ) -> Result<TimestampedTwedKnnOutcome<'a, V>, TimestampedTwedIndexError> {
        self.validate_series_identity(query)?;
        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_series_len(Operand::Query, query.values().len())?;
        if k == 0 || self.is_empty() {
            return Ok(OperationOutcome::Complete {
                value: Vec::new(),
                usage: ledger.usage(),
            });
        }

        let capacity = k.min(self.len);
        if let Err(reason) = ledger.charge(ResourceKind::Results, capacity) {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason,
                continuation: None,
                usage: ledger.usage(),
            });
        }
        let width =
            query
                .values()
                .len()
                .checked_add(1)
                .ok_or(TimestampedTwedIndexError::Resource(
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    },
                ))?;
        let scratch_bytes = width
            .checked_mul(size_of::<f64>())
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or(TimestampedTwedIndexError::Resource(
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            ))?;
        if let Err(reason) = ledger.observe_peak(ResourceKind::ScratchBytes, scratch_bytes) {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason,
                continuation: None,
                usage: ledger.usage(),
            });
        }
        let mut previous = match fallible_f64_buffer(width, ResourceKind::ScratchBytes) {
            Ok(buffer) => buffer,
            Err(reason) => {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason,
                    continuation: None,
                    usage: ledger.usage(),
                });
            }
        };
        let mut current = match fallible_f64_buffer(width, ResourceKind::ScratchBytes) {
            Ok(buffer) => buffer,
            Err(reason) => {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason,
                    continuation: None,
                    usage: ledger.usage(),
                });
            }
        };
        // This vector is both the bounded max-heap during scanning and the
        // eventual public output buffer. Keeping the heap operations local
        // avoids an infallible wrapper-to-output collection at finalization.
        let mut best = Vec::new();
        if best.try_reserve_exact(capacity).is_err() {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason: IncompleteReason::AllocationFailed {
                    resource: ResourceKind::Results,
                    requested: capacity,
                },
                continuation: None,
                usage: ledger.usage(),
            });
        }

        for candidate in self.buckets.iter().flatten() {
            let candidate_len = candidate.series.values().len();
            if candidate_len > limits.max_series_len {
                return Ok(timestamped_knn_incomplete(
                    best,
                    IncompleteReason::BudgetExceeded {
                        resource: ResourceKind::SeriesLength,
                        limit: limits.max_series_len,
                        requested: candidate_len,
                    },
                    ledger,
                ));
            }
            let cells = match exact_cells(query.values().len(), candidate_len) {
                Ok(cells) => cells,
                Err(reason) => return Ok(timestamped_knn_incomplete(best, reason, ledger)),
            };
            let work = match exact_work(query.values().len(), candidate_len) {
                Ok(work) => work,
                Err(reason) => return Ok(timestamped_knn_incomplete(best, reason, ledger)),
            };
            if let Err(reason) = ledger.charge_many(&[
                (ResourceKind::Candidates, 1),
                (ResourceKind::DpCells, cells),
                (ResourceKind::WorkUnits, work),
            ]) {
                return Ok(timestamped_knn_incomplete(best, reason, ledger));
            }
            let cutoff = if best.len() == capacity {
                best.first().map_or(f64::INFINITY, |entry| entry.distance)
            } else {
                f64::INFINITY
            };
            let exact = match TimestampedTwedRangeContinuation::<V>::score_exact(
                &self.config,
                query,
                cutoff,
                &candidate.series,
                &mut previous,
                &mut current,
            ) {
                Ok(exact) => exact,
                Err(reason) => return Ok(timestamped_knn_incomplete(best, reason, ledger)),
            };
            let Some(distance) = exact else {
                continue;
            };
            let entry = TimestampedTwedRangeMatch {
                episode_id: candidate.episode_id,
                value: &candidate.value,
                series: &candidate.series,
                distance,
            };
            if best.len() < capacity {
                timestamped_knn_heap_push(&mut best, entry);
            } else if best
                .first()
                .is_some_and(|worst| timestamped_match_order(&entry, worst).is_lt())
            {
                best[0] = entry;
                timestamped_knn_heap_sift_down(&mut best, 0);
            }
        }

        Ok(OperationOutcome::Complete {
            value: sorted_timestamped_knn(best),
            usage: ledger.usage(),
        })
    }

    fn validate_series_identity(
        &self,
        series: &TimestampedSeries,
    ) -> Result<(), TimestampedTwedIndexError> {
        if series.unit() != self.quantizer.unit {
            return Err(TimestampedTwedIndexError::MixedUnits);
        }
        if series.origin().to_bits() != self.quantizer.origin.to_bits() {
            return Err(TimestampedTwedIndexError::MixedOrigins);
        }
        Ok(())
    }
}

/// Tagged result of a bounded physical-timestamp TWED range query.
pub type TimestampedTwedRangeOutcome<'a, V> = OperationOutcome<
    Vec<TimestampedTwedRangeMatch<'a, V>>,
    TimestampedTwedRangeContinuation<'a, V>,
>;

/// Tagged result of a bounded exact physical-timestamp TWED kNN scan.
pub type TimestampedTwedKnnOutcome<'a, V> = OperationOutcome<Vec<TimestampedTwedRangeMatch<'a, V>>>;

fn timestamped_match_order<V>(
    left: &TimestampedTwedRangeMatch<'_, V>,
    right: &TimestampedTwedRangeMatch<'_, V>,
) -> Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.episode_id.cmp(&right.episode_id))
}

fn timestamped_knn_heap_push<'a, V>(
    heap: &mut Vec<TimestampedTwedRangeMatch<'a, V>>,
    entry: TimestampedTwedRangeMatch<'a, V>,
) {
    heap.push(entry);
    let mut child = heap.len() - 1;
    while child > 0 {
        let parent = (child - 1) / 2;
        if !timestamped_match_order(&heap[parent], &heap[child]).is_lt() {
            break;
        }
        heap.swap(parent, child);
        child = parent;
    }
}

fn timestamped_knn_heap_sift_down<V>(
    heap: &mut [TimestampedTwedRangeMatch<'_, V>],
    mut parent: usize,
) {
    loop {
        let Some(left) = parent.checked_mul(2).and_then(|index| index.checked_add(1)) else {
            return;
        };
        if left >= heap.len() {
            return;
        }
        let right = left + 1;
        let larger_child =
            if right < heap.len() && timestamped_match_order(&heap[left], &heap[right]).is_lt() {
                right
            } else {
                left
            };
        if !timestamped_match_order(&heap[parent], &heap[larger_child]).is_lt() {
            return;
        }
        heap.swap(parent, larger_child);
        parent = larger_child;
    }
}

fn sorted_timestamped_knn<V>(
    mut best: Vec<TimestampedTwedRangeMatch<'_, V>>,
) -> Vec<TimestampedTwedRangeMatch<'_, V>> {
    best.sort_unstable_by(timestamped_match_order);
    best
}

fn timestamped_knn_incomplete<V>(
    best: Vec<TimestampedTwedRangeMatch<'_, V>>,
    reason: IncompleteReason,
    ledger: ResourceLedger,
) -> TimestampedTwedKnnOutcome<'_, V> {
    OperationOutcome::Incomplete {
        partial: Some(sorted_timestamped_knn(best)),
        reason,
        continuation: None,
        usage: ledger.usage(),
    }
}

#[derive(Clone, Copy)]
struct ProductPosition {
    row: u32,
    cost: f64,
}

struct ProductState {
    previous_token: Option<u64>,
    position_start: u32,
    position_end: u32,
    final_cost: f64,
}

struct ProductStateView<'a> {
    previous_token: Option<u64>,
    positions: &'a [ProductPosition],
    final_cost: f64,
}

struct ProductStateArena {
    max_states: usize,
    max_positions: usize,
    reused_states: usize,
    collision_heap_capacity_count: usize,
    states: Vec<ProductState>,
    positions: Vec<ProductPosition>,
    fingerprints: HashMap<u64, SmallVec<[usize; 2]>>,
}

impl ProductStateArena {
    fn new(max_states: usize, max_positions: usize) -> Self {
        Self {
            max_states,
            max_positions,
            reused_states: 0,
            collision_heap_capacity_count: 0,
            states: Vec::new(),
            positions: Vec::new(),
            fingerprints: HashMap::new(),
        }
    }

    fn get(&self, id: usize) -> Option<ProductStateView<'_>> {
        let state = self.states.get(id)?;
        let start = usize::try_from(state.position_start).ok()?;
        let end = usize::try_from(state.position_end).ok()?;
        Some(ProductStateView {
            previous_token: state.previous_token,
            positions: self.positions.get(start..end)?,
            final_cost: state.final_cost,
        })
    }

    fn exact_equal(
        state: ProductStateView<'_>,
        previous_token: Option<u64>,
        positions: &[ProductPosition],
        final_cost: f64,
    ) -> bool {
        state.previous_token == previous_token
            && canonical_bits(state.final_cost) == canonical_bits(final_cost)
            && state.positions.len() == positions.len()
            && state.positions.iter().zip(positions).all(|(left, right)| {
                left.row == right.row && canonical_bits(left.cost) == canonical_bits(right.cost)
            })
    }

    fn fingerprint(
        previous_token: Option<u64>,
        positions: &[ProductPosition],
        final_cost: f64,
    ) -> u64 {
        let mut hasher = DeterministicHasher::default();
        previous_token.hash(&mut hasher);
        canonical_bits(final_cost).hash(&mut hasher);
        positions.len().hash(&mut hasher);
        for position in positions {
            position.row.hash(&mut hasher);
            canonical_bits(position.cost).hash(&mut hasher);
        }
        hasher.finish()
    }

    fn intern(
        &mut self,
        previous_token: Option<u64>,
        positions: &[ProductPosition],
        final_cost: f64,
    ) -> Result<usize, IncompleteReason> {
        let fingerprint = Self::fingerprint(previous_token, positions, final_cost);
        self.intern_at_fingerprint(previous_token, positions, final_cost, fingerprint)
    }

    #[cfg(test)]
    fn intern_with_fingerprint(
        &mut self,
        previous_token: Option<u64>,
        positions: &[ProductPosition],
        final_cost: f64,
        fingerprint: u64,
    ) -> Result<usize, IncompleteReason> {
        self.intern_at_fingerprint(previous_token, positions, final_cost, fingerprint)
    }

    fn intern_at_fingerprint(
        &mut self,
        previous_token: Option<u64>,
        positions: &[ProductPosition],
        final_cost: f64,
        fingerprint: u64,
    ) -> Result<usize, IncompleteReason> {
        if positions.windows(2).any(|pair| pair[0].row >= pair[1].row)
            || positions.iter().any(|position| !position.cost.is_finite())
            || final_cost.is_nan()
        {
            return Err(IncompleteReason::InvalidStoredData);
        }
        if let Some(candidates) = self.fingerprints.get(&fingerprint) {
            for &candidate in candidates {
                let state = self
                    .get(candidate)
                    .ok_or(IncompleteReason::InvalidStoredData)?;
                if Self::exact_equal(state, previous_token, positions, final_cost) {
                    self.reused_states = self.reused_states.saturating_add(1);
                    return Ok(candidate);
                }
            }
        }

        let requested =
            self.states
                .len()
                .checked_add(1)
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::QueueEntries,
                })?;
        if requested > self.max_states {
            return Err(IncompleteReason::BudgetExceeded {
                resource: ResourceKind::QueueEntries,
                limit: self.max_states,
                requested,
            });
        }
        let requested_positions = self.positions.len().checked_add(positions.len()).ok_or(
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            },
        )?;
        let position_limit = self.max_positions.min(u32::MAX as usize);
        if requested_positions > position_limit {
            return Err(IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: position_limit,
                requested: requested_positions,
            });
        }
        self.states
            .try_reserve(1)
            .map_err(|_| IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested,
            })?;
        let bytes = positions
            .len()
            .checked_mul(size_of::<ProductPosition>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        self.positions
            .try_reserve_exact(positions.len())
            .map_err(|_| IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested: bytes,
            })?;
        self.fingerprints
            .try_reserve(1)
            .map_err(|_| IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested,
            })?;
        let mut collision_capacity_before = 0;
        if let Some(bucket) = self.fingerprints.get_mut(&fingerprint) {
            collision_capacity_before = if bucket.spilled() {
                bucket.capacity()
            } else {
                0
            };
            bucket
                .try_reserve(1)
                .map_err(|_| IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: bucket.len().saturating_add(1),
                })?;
        }

        let id = self.states.len();
        let position_start = u32::try_from(self.positions.len()).map_err(|_| {
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            }
        })?;
        self.positions.extend_from_slice(positions);
        let position_end = u32::try_from(self.positions.len()).map_err(|_| {
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            }
        })?;
        self.states.push(ProductState {
            previous_token,
            position_start,
            position_end,
            final_cost,
        });
        let bucket = self.fingerprints.entry(fingerprint).or_default();
        bucket.push(id);
        let collision_capacity_after = if bucket.spilled() {
            bucket.capacity()
        } else {
            0
        };
        self.collision_heap_capacity_count = self
            .collision_heap_capacity_count
            .checked_sub(collision_capacity_before)
            .and_then(|capacity| capacity.checked_add(collision_capacity_after))
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        Ok(id)
    }

    fn retained_bytes(&self) -> Option<usize> {
        let headers = self
            .states
            .capacity()
            .checked_mul(size_of::<ProductState>())?;
        let positions = self
            .positions
            .capacity()
            .checked_mul(size_of::<ProductPosition>())?;
        let map = self
            .fingerprints
            .capacity()
            .checked_mul(size_of::<(u64, SmallVec<[usize; 2]>)>())?;
        let ids = self
            .collision_heap_capacity_count
            .checked_mul(size_of::<usize>())?;
        headers
            .checked_add(positions)?
            .checked_add(map)?
            .checked_add(ids)
    }
}

struct ProductFrame {
    state: usize,
    final_bucket: Option<usize>,
    next_candidate: usize,
    edges: DfsNodeEdges<DynamicDawgU64Node<usize>>,
}

impl ProductFrame {
    fn open(
        traversal: &mut TraversalSession<DynamicDawgU64Node<usize>>,
        cursor: TraversalCursor<<DynamicDawgU64Node<usize> as DictionaryNode>::SnapshotCursor>,
        state: usize,
    ) -> Result<Self, IncompleteReason> {
        let final_value = traversal.final_value_at_cursor(cursor, None);
        let edges = traversal.open_dfs_node(cursor);
        let final_bucket = if edges.is_final() {
            Some(final_value.ok_or(IncompleteReason::InvalidStoredData)?)
        } else {
            if final_value.is_some() {
                return Err(IncompleteReason::InvalidStoredData);
            }
            None
        };
        Ok(Self {
            state,
            final_bucket,
            next_candidate: 0,
            edges,
        })
    }
}

#[derive(Default)]
struct DeterministicHasher(u64);

impl Hasher for DeterministicHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = if self.0 == 0 {
            0xcbf2_9ce4_8422_2325
        } else {
            self.0
        };
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = hash;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// In-memory state for a strict exact range query.
///
/// The captured root and the immutable borrow of the index bind every page to
/// one dictionary revision. All traversal is iterative; target depth consumes
/// heap-bounded frames rather than call-stack frames.
pub struct TimestampedTwedRangeContinuation<'a, V> {
    index: &'a TimestampedTwedIndex<V>,
    query: &'a TimestampedSeries,
    cutoff: f64,
    captured_root: DynamicDawgU64Node<usize>,
    captured_terms: usize,
    traversal: TraversalSession<DynamicDawgU64Node<usize>>,
    stack: Vec<ProductFrame>,
    states: ProductStateArena,
    transitions: HashMap<(usize, u64), Option<usize>>,
    max_transition_cache_entries: usize,
    source_frontier: Vec<ProductPosition>,
    next_frontier: Vec<ProductPosition>,
    canonical_frontier: Vec<ProductPosition>,
    scheduled_rows: Vec<u32>,
    exact_previous: Vec<f64>,
    exact_current: Vec<f64>,
    results: Vec<TimestampedTwedRangeMatch<'a, V>>,
    ledger: ResourceLedger,
    terminal: Option<IncompleteReason>,
}

impl<V> std::fmt::Debug for TimestampedTwedRangeContinuation<'_, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TimestampedTwedRangeContinuation")
            .field("query_len", &self.query.values().len())
            .field("captured_terms", &self.captured_terms)
            .field("stats", &self.retained_product_state_stats())
            .field("result_count", &self.results.len())
            .field("usage", &self.ledger.usage())
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}

impl<'a, V> TimestampedTwedRangeContinuation<'a, V> {
    #[expect(
        clippy::result_large_err,
        reason = "the cold start failure retains its exact ledger inline so reporting a resource failure never requires another allocation"
    )]
    fn start(
        index: &'a TimestampedTwedIndex<V>,
        query: &'a TimestampedSeries,
        cutoff: f64,
        limits: TimestampedTwedProductLimits,
    ) -> Result<Self, (IncompleteReason, ResourceLedger)> {
        let mut ledger = ResourceLedger::new(limits.resources);
        if limits.max_product_states == 0 {
            return Err((
                IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::QueueEntries,
                    limit: 0,
                    requested: 1,
                },
                ledger,
            ));
        }
        let width = match query.values().len().checked_add(1) {
            Some(width) => width,
            None => {
                return Err((
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::DpCells,
                    },
                    ledger,
                ));
            }
        };
        let scratch_bytes = match width
            .checked_mul(size_of::<f64>())
            .and_then(|bytes| bytes.checked_mul(2))
        {
            Some(bytes) => bytes,
            None => {
                return Err((
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    },
                    ledger,
                ));
            }
        };
        if let Err(reason) = ledger.observe_peak(ResourceKind::ScratchBytes, scratch_bytes) {
            return Err((reason, ledger));
        }

        let mut root_final_cost = 0.0;
        let mut previous_value = 0.0;
        let mut previous_time = query.origin();
        for row in 1..width {
            let value = query.values()[row - 1];
            let time = query.timestamps()[row - 1];
            root_final_cost +=
                delete_cost(value, previous_value, time, previous_time, &index.config);
            if !root_final_cost.is_finite() {
                return Err((IncompleteReason::NumericOverflow, ledger));
            }
            previous_value = value;
            previous_time = time;
        }

        let root_final_cost = if root_final_cost <= cutoff {
            root_final_cost
        } else {
            f64::INFINITY
        };
        let root_position = [ProductPosition { row: 0, cost: 0.0 }];
        let mut states =
            ProductStateArena::new(limits.max_product_states, limits.max_product_positions);
        let root_state = match states.intern(None, &root_position, root_final_cost) {
            Ok(id) => id,
            Err(reason) => return Err((reason, ledger)),
        };
        let (captured_root, captured_terms) = index.dictionary.root_with_term_count();
        let (mut traversal, root_cursor) =
            TraversalSession::capture_mapped(DictionaryTraversalRoot::owned(captured_root.clone()));
        if !traversal.supports_efficient_dfs_edge_paging() {
            return Err((IncompleteReason::Unsupported, ledger));
        }
        let root_frame = match ProductFrame::open(&mut traversal, root_cursor, root_state) {
            Ok(frame) => frame,
            Err(reason) => return Err((reason, ledger)),
        };
        if let Err(reason) = ledger.charge(ResourceKind::TrieNodes, 1) {
            return Err((reason, ledger));
        }
        let mut stack = Vec::new();
        if stack.try_reserve_exact(1).is_err() {
            return Err((
                IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested: size_of::<ProductFrame>(),
                },
                ledger,
            ));
        }
        stack.push(root_frame);

        let exact_previous = match fallible_f64_buffer(width, ResourceKind::ScratchBytes) {
            Ok(buffer) => buffer,
            Err(reason) => return Err((reason, ledger)),
        };
        let exact_current = match fallible_f64_buffer(width, ResourceKind::ScratchBytes) {
            Ok(buffer) => buffer,
            Err(reason) => return Err((reason, ledger)),
        };
        let mut session = Self {
            index,
            query,
            cutoff,
            captured_root,
            captured_terms,
            traversal,
            stack,
            states,
            transitions: HashMap::new(),
            max_transition_cache_entries: limits.max_transition_cache_entries,
            source_frontier: Vec::new(),
            next_frontier: Vec::new(),
            canonical_frontier: Vec::new(),
            scheduled_rows: Vec::new(),
            exact_previous,
            exact_current,
            results: Vec::new(),
            ledger,
            terminal: None,
        };
        if let Err(reason) = session.observe_state_peaks() {
            return Err((reason, session.ledger));
        }
        Ok(session)
    }

    /// Exact members found so far. This is not complete until `resume`
    /// returns [`OperationOutcome::Complete`].
    pub fn exact_partial(&self) -> &[TimestampedTwedRangeMatch<'a, V>] {
        &self.results
    }

    /// Return the atomically captured dictionary term count.
    #[inline]
    pub fn captured_term_count(&self) -> usize {
        self.captured_terms
    }

    /// Return the opaque root identity for the captured revision, when the
    /// backend supplies one.
    #[inline]
    pub fn captured_revision_identity(&self) -> Option<SnapshotNodeIdentity> {
        self.captured_root.snapshot_node_identity()
    }

    /// Return cumulative and peak query usage.
    #[inline]
    pub fn usage(&self) -> ResourceUsage {
        self.ledger.usage()
    }

    /// Return the current bounded product footprint.
    pub fn retained_product_state_stats(&self) -> TimestampedTwedProductStats {
        TimestampedTwedProductStats {
            frames: self.stack.len(),
            states: self.states.states.len(),
            positions: self.states.positions.len(),
            cached_transitions: self.transitions.len(),
            reused_states: self.states.reused_states,
        }
    }

    /// Cancel without converting an exact partial set into an absence claim.
    pub fn cancel(mut self) -> TimestampedTwedRangeOutcome<'a, V> {
        let partial = Some(self.sorted_results());
        OperationOutcome::Incomplete {
            partial,
            reason: IncompleteReason::Cancelled,
            continuation: None,
            usage: self.ledger.usage(),
        }
    }

    /// Resume this immutable-revision query for one bounded page.
    pub fn resume(mut self, page: PageBudget) -> TimestampedTwedRangeOutcome<'a, V> {
        let step = self.advance(page);
        match step {
            OperationOutcome::Complete { usage, .. } => OperationOutcome::Complete {
                value: self.sorted_results(),
                usage,
            },
            OperationOutcome::Incomplete {
                reason,
                continuation,
                usage,
                ..
            } => {
                if continuation.is_some() {
                    OperationOutcome::Incomplete {
                        partial: None,
                        reason,
                        continuation: Some(self),
                        usage,
                    }
                } else {
                    let partial = Some(self.sorted_results());
                    OperationOutcome::Incomplete {
                        partial,
                        reason,
                        continuation: None,
                        usage,
                    }
                }
            }
        }
    }

    fn sorted_results(&mut self) -> Vec<TimestampedTwedRangeMatch<'a, V>> {
        self.results.sort_unstable_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.episode_id.cmp(&right.episode_id))
        });
        std::mem::take(&mut self.results)
    }

    fn paused(
        &self,
        resource: ResourceKind,
        limit: usize,
        requested: usize,
    ) -> OperationOutcome<(), ()> {
        OperationOutcome::Incomplete {
            partial: None,
            reason: IncompleteReason::BudgetExceeded {
                resource,
                limit,
                requested,
            },
            continuation: Some(()),
            usage: self.ledger.usage(),
        }
    }

    fn terminate(&mut self, reason: IncompleteReason) -> OperationOutcome<(), ()> {
        self.terminal = Some(reason);
        OperationOutcome::Incomplete {
            partial: None,
            reason,
            continuation: None,
            usage: self.ledger.usage(),
        }
    }

    fn advance(&mut self, page: PageBudget) -> OperationOutcome<(), ()> {
        if let Some(reason) = self.terminal {
            return self.terminate(reason);
        }
        let mut page_work = 0_usize;
        let mut page_results = 0_usize;

        loop {
            let Some(frame) = self.stack.last() else {
                return OperationOutcome::Complete {
                    value: (),
                    usage: self.ledger.usage(),
                };
            };

            if let Some(bucket_id) = frame.final_bucket {
                let candidate_slot = frame.next_candidate;
                let Some(bucket) = self.index.buckets.get(bucket_id) else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                if candidate_slot < bucket.len() {
                    let candidate = &bucket[candidate_slot];
                    if candidate.series.values().len() > self.ledger.limits().max_series_len {
                        return self.terminate(IncompleteReason::BudgetExceeded {
                            resource: ResourceKind::SeriesLength,
                            limit: self.ledger.limits().max_series_len,
                            requested: candidate.series.values().len(),
                        });
                    }
                    if page_results >= page.max_results {
                        return self.paused(
                            ResourceKind::Results,
                            page.max_results,
                            page_results.saturating_add(1),
                        );
                    }
                    let score_work = match exact_work(
                        self.query.values().len(),
                        candidate.series.values().len(),
                    ) {
                        Ok(work) => work,
                        Err(reason) => return self.terminate(reason),
                    };
                    let requested_page = match page_work.checked_add(score_work) {
                        Some(requested) => requested,
                        None => {
                            return self.terminate(IncompleteReason::ArithmeticOverflow {
                                resource: ResourceKind::WorkUnits,
                            });
                        }
                    };
                    if requested_page > page.max_work_units {
                        return self.paused(
                            ResourceKind::WorkUnits,
                            page.max_work_units,
                            requested_page,
                        );
                    }
                    let cells = match exact_cells(
                        self.query.values().len(),
                        candidate.series.values().len(),
                    ) {
                        Ok(cells) => cells,
                        Err(reason) => return self.terminate(reason),
                    };
                    if let Err(reason) = self.ledger.charge_many(&[
                        (ResourceKind::Candidates, 1),
                        (ResourceKind::DpCells, cells),
                        (ResourceKind::WorkUnits, score_work),
                    ]) {
                        return self.terminate(reason);
                    }
                    let exact = match Self::score_exact(
                        &self.index.config,
                        self.query,
                        self.cutoff,
                        &candidate.series,
                        &mut self.exact_previous,
                        &mut self.exact_current,
                    ) {
                        Ok(exact) => exact,
                        Err(reason) => return self.terminate(reason),
                    };
                    self.stack
                        .last_mut()
                        .expect("candidate frame remains live across exact scoring")
                        .next_candidate += 1;
                    page_work = requested_page;
                    if let Some(distance) = exact {
                        if self.results.try_reserve(1).is_err() {
                            return self.terminate(IncompleteReason::AllocationFailed {
                                resource: ResourceKind::ContinuationBytes,
                                requested: self.results.len().saturating_add(1),
                            });
                        }
                        if let Err(reason) = self.ledger.charge(ResourceKind::Results, 1) {
                            return self.terminate(reason);
                        }
                        self.results.push(TimestampedTwedRangeMatch {
                            episode_id: candidate.episode_id,
                            value: &candidate.value,
                            series: &candidate.series,
                            distance,
                        });
                        page_results += 1;
                    }
                    if let Err(reason) = self.observe_state_peaks() {
                        return self.terminate(reason);
                    }
                    continue;
                }
                self.stack
                    .last_mut()
                    .expect("exhausted candidate frame remains live")
                    .final_bucket = None;
                continue;
            }

            if frame.edges.remaining() == 0 {
                self.stack.pop();
                continue;
            }

            let transition_work = match self.query.values().len().checked_add(1) {
                Some(work) => work,
                None => {
                    return self.terminate(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::WorkUnits,
                    });
                }
            };
            let requested_page = match page_work.checked_add(transition_work) {
                Some(requested) => requested,
                None => {
                    return self.terminate(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::WorkUnits,
                    });
                }
            };
            if requested_page > page.max_work_units {
                return self.paused(ResourceKind::WorkUnits, page.max_work_units, requested_page);
            }

            let state_id = frame.state;
            let mut charged = self.ledger;
            if let Err(reason) = charged.charge_many(&[
                (ResourceKind::TrieEdges, 1),
                (ResourceKind::DpCells, transition_work),
                (ResourceKind::WorkUnits, transition_work),
            ]) {
                return self.terminate(reason);
            }
            let edge = {
                let (traversal, stack) = (&mut self.traversal, &mut self.stack);
                let Some(frame) = stack.last_mut() else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                traversal.next_dfs_edge(&mut frame.edges)
            };
            let Some((token, child)) = edge else {
                return self.terminate(IncompleteReason::InvalidStoredData);
            };
            self.ledger = charged;
            let next_state = match self.transition(state_id, token) {
                Ok(next) => next,
                Err(reason) => return self.terminate(reason),
            };
            page_work = requested_page;
            if let Some(next_state) = next_state {
                let mut charged = self.ledger;
                if let Err(reason) = charged.charge(ResourceKind::TrieNodes, 1) {
                    return self.terminate(reason);
                }
                if self.stack.len() == self.stack.capacity()
                    && self.stack.try_reserve_exact(1).is_err()
                {
                    return self.terminate(IncompleteReason::AllocationFailed {
                        resource: ResourceKind::ContinuationBytes,
                        requested: self.stack.len().saturating_add(1),
                    });
                }
                let child_frame = match ProductFrame::open(&mut self.traversal, child, next_state) {
                    Ok(frame) => frame,
                    Err(reason) => return self.terminate(reason),
                };
                self.ledger = charged;
                self.stack.push(child_frame);
                if let Err(reason) = self.observe_state_peaks() {
                    return self.terminate(reason);
                }
            }
        }
    }

    fn transition(
        &mut self,
        state_id: usize,
        token: u64,
    ) -> Result<Option<usize>, IncompleteReason> {
        if let Some(cached) = self.transitions.get(&(state_id, token)) {
            return Ok(*cached);
        }
        let current = self
            .index
            .quantizer
            .decode(token)
            .map_err(|_| IncompleteReason::InvalidStoredData)?;
        let state = self
            .states
            .get(state_id)
            .ok_or(IncompleteReason::InvalidStoredData)?;
        let previous_token = state.previous_token;
        let state_final_cost = state.final_cost;
        reconstruct_frontier(
            &self.index.config,
            self.query,
            state.positions,
            state_final_cost,
            self.cutoff,
            &mut self.source_frontier,
        )?;
        let previous = match previous_token {
            Some(previous) => self
                .index
                .quantizer
                .decode(previous)
                .map_err(|_| IncompleteReason::InvalidStoredData)?,
            None => TimestampedScalarBox::point(
                0.0,
                self.index.quantizer.origin,
                self.index.quantizer.unit,
            )
            .map_err(|_| IncompleteReason::InvalidStoredData)?,
        };
        let candidate_delete = self
            .index
            .config
            .interval_delete_lower_bound(current, previous)
            .map_err(|_| IncompleteReason::InvalidStoredData)?;
        if !candidate_delete.is_finite() {
            return Err(IncompleteReason::NumericOverflow);
        }
        build_scheduled_rows(
            &self.source_frontier,
            self.query.values().len().saturating_add(1),
            &mut self.scheduled_rows,
        )?;
        step_sparse_interval_frontier(
            &self.index.config,
            self.query,
            current,
            previous,
            candidate_delete,
            self.cutoff,
            &self.source_frontier,
            &self.scheduled_rows,
            &mut self.next_frontier,
        )?;

        let next = if self.next_frontier.is_empty() {
            None
        } else {
            canonicalize_frontier(
                &self.index.config,
                self.query,
                &self.next_frontier,
                &mut self.canonical_frontier,
            )?;
            let final_row = u32::try_from(self.query.values().len()).map_err(|_| {
                IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                }
            })?;
            let final_cost = self
                .next_frontier
                .last()
                .filter(|position| position.row == final_row)
                .map_or(f64::INFINITY, |position| position.cost);
            Some(
                self.states
                    .intern(Some(token), &self.canonical_frontier, final_cost)?,
            )
        };
        if self.transitions.len() < self.max_transition_cache_entries
            && self.transitions.try_reserve(1).is_ok()
        {
            self.transitions.insert((state_id, token), next);
        }
        Ok(next)
    }

    fn score_exact(
        config: &MetricTimestampedTwedConfig,
        query: &TimestampedSeries,
        cutoff: f64,
        candidate: &TimestampedSeries,
        exact_previous: &mut Vec<f64>,
        exact_current: &mut Vec<f64>,
    ) -> Result<Option<f64>, IncompleteReason> {
        let width = query.values().len() + 1;
        exact_previous[..width].fill(f64::INFINITY);
        exact_current[..width].fill(f64::INFINITY);
        exact_previous[0] = 0.0;
        let mut query_previous_value = 0.0;
        let mut query_previous_time = query.origin();
        for row in 1..width {
            let value = query.values()[row - 1];
            let time = query.timestamps()[row - 1];
            exact_previous[row] = exact_previous[row - 1]
                + delete_cost(
                    value,
                    query_previous_value,
                    time,
                    query_previous_time,
                    config,
                );
            query_previous_value = value;
            query_previous_time = time;
        }

        let mut candidate_previous_value = 0.0;
        let mut candidate_previous_time = candidate.origin();
        for (&candidate_value, &candidate_time) in
            candidate.values().iter().zip(candidate.timestamps())
        {
            let candidate_delete = delete_cost(
                candidate_value,
                candidate_previous_value,
                candidate_time,
                candidate_previous_time,
                config,
            );
            exact_current[0] = exact_previous[0] + candidate_delete;
            let mut column_min = exact_current[0];
            let mut query_previous_value = 0.0;
            let mut query_previous_time = query.origin();
            for row in 1..width {
                let query_value = query.values()[row - 1];
                let query_time = query.timestamps()[row - 1];
                let pair = exact_previous[row - 1]
                    + match_cost(
                        query_value,
                        query_previous_value,
                        query_time,
                        query_previous_time,
                        candidate_value,
                        candidate_previous_value,
                        candidate_time,
                        candidate_previous_time,
                        config,
                    );
                let delete_query = exact_current[row - 1]
                    + delete_cost(
                        query_value,
                        query_previous_value,
                        query_time,
                        query_previous_time,
                        config,
                    );
                let delete_candidate = exact_previous[row] + candidate_delete;
                let cost = pair.min(delete_query).min(delete_candidate);
                if !cost.is_finite() {
                    return Err(IncompleteReason::NumericOverflow);
                }
                exact_current[row] = cost;
                column_min = column_min.min(cost);
                query_previous_value = query_value;
                query_previous_time = query_time;
            }
            if column_min > cutoff {
                return Ok(None);
            }
            std::mem::swap(exact_previous, exact_current);
            candidate_previous_value = candidate_value;
            candidate_previous_time = candidate_time;
        }
        let exact = exact_previous[width - 1];
        if !exact.is_finite() {
            return Err(IncompleteReason::NumericOverflow);
        }
        Ok((exact <= cutoff).then_some(exact))
    }

    fn observe_state_peaks(&mut self) -> Result<(), IncompleteReason> {
        // A paged cursor represents unvisited fanout in dictionary metadata;
        // it is not retained queue memory. Every live DFS frame owns at most
        // one fixed inline page already accounted for in the frame header.
        let queue_entries = self.stack.len();
        self.ledger
            .observe_peak(ResourceKind::QueueEntries, queue_entries)?;

        let frame_headers = self
            .stack
            .capacity()
            .checked_mul(size_of::<ProductFrame>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        let state_bytes =
            self.states
                .retained_bytes()
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ContinuationBytes,
                })?;
        let cache_bytes = self
            .transitions
            .capacity()
            .checked_mul(size_of::<((usize, u64), Option<usize>)>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        let result_bytes = self
            .results
            .capacity()
            .checked_mul(size_of::<TimestampedTwedRangeMatch<'a, V>>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        let sparse_positions = self
            .source_frontier
            .capacity()
            .checked_add(self.next_frontier.capacity())
            .and_then(|positions| positions.checked_add(self.canonical_frontier.capacity()))
            .and_then(|positions| positions.checked_mul(size_of::<ProductPosition>()))
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        let scheduled_rows = self
            .scheduled_rows
            .capacity()
            .checked_mul(size_of::<u32>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        let dense_exact = self
            .exact_previous
            .capacity()
            .checked_add(self.exact_current.capacity())
            .and_then(|slots| slots.checked_mul(size_of::<f64>()))
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        let query_scratch = sparse_positions
            .checked_add(scheduled_rows)
            .and_then(|bytes| bytes.checked_add(dense_exact))
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        self.ledger
            .observe_peak(ResourceKind::ScratchBytes, query_scratch)?;
        let continuation = frame_headers
            .checked_add(state_bytes)
            .and_then(|bytes| bytes.checked_add(cache_bytes))
            .and_then(|bytes| bytes.checked_add(result_bytes))
            .and_then(|bytes| bytes.checked_add(query_scratch))
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        self.ledger
            .observe_peak(ResourceKind::ContinuationBytes, continuation)
    }
}

fn reserve_position_slot(
    positions: &mut Vec<ProductPosition>,
    maximum_len: usize,
) -> Result<(), IncompleteReason> {
    if positions.len() < positions.capacity() {
        return Ok(());
    }
    let requested = positions
        .len()
        .checked_add(1)
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::ScratchBytes,
        })?;
    if requested > maximum_len {
        return Err(IncompleteReason::InvalidStoredData);
    }
    let target = positions
        .capacity()
        .max(4)
        .checked_mul(2)
        .unwrap_or(maximum_len)
        .min(maximum_len)
        .max(requested);
    positions
        .try_reserve_exact(target.saturating_sub(positions.len()))
        .map_err(|_| IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested: target.saturating_mul(size_of::<ProductPosition>()),
        })
}

fn push_position(
    positions: &mut Vec<ProductPosition>,
    maximum_len: usize,
    position: ProductPosition,
) -> Result<(), IncompleteReason> {
    reserve_position_slot(positions, maximum_len)?;
    positions.push(position);
    Ok(())
}

fn reserve_row_slot(rows: &mut Vec<u32>, maximum_len: usize) -> Result<(), IncompleteReason> {
    if rows.len() < rows.capacity() {
        return Ok(());
    }
    let requested = rows
        .len()
        .checked_add(1)
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::ScratchBytes,
        })?;
    if requested > maximum_len {
        return Err(IncompleteReason::InvalidStoredData);
    }
    let target = rows
        .capacity()
        .max(4)
        .checked_mul(2)
        .unwrap_or(maximum_len)
        .min(maximum_len)
        .max(requested);
    rows.try_reserve_exact(target.saturating_sub(rows.len()))
        .map_err(|_| IncompleteReason::AllocationFailed {
            resource: ResourceKind::ScratchBytes,
            requested: target.saturating_mul(size_of::<u32>()),
        })
}

#[inline]
fn query_delete_cost_at(
    config: &MetricTimestampedTwedConfig,
    query: &TimestampedSeries,
    row: usize,
) -> Result<f64, IncompleteReason> {
    if row == 0 || row > query.values().len() {
        return Err(IncompleteReason::InvalidStoredData);
    }
    let previous_value = if row == 1 {
        0.0
    } else {
        query.values()[row - 2]
    };
    let previous_time = if row == 1 {
        query.origin()
    } else {
        query.timestamps()[row - 2]
    };
    let cost = delete_cost(
        query.values()[row - 1],
        previous_value,
        query.timestamps()[row - 1],
        previous_time,
        config,
    );
    if cost.is_finite() {
        Ok(cost)
    } else {
        Err(IncompleteReason::NumericOverflow)
    }
}

/// Reconstruct exactly the live dense column denoted by one canonical sparse
/// residual. Every omitted row is an explicit same-column query deletion with
/// bit-identical binary64 cost; no approximate comparison establishes the
/// simulation.
fn reconstruct_frontier(
    config: &MetricTimestampedTwedConfig,
    query: &TimestampedSeries,
    canonical: &[ProductPosition],
    final_cost: f64,
    cutoff: f64,
    output: &mut Vec<ProductPosition>,
) -> Result<(), IncompleteReason> {
    output.clear();
    let width =
        query
            .values()
            .len()
            .checked_add(1)
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::DpCells,
            })?;
    for (index, base) in canonical.iter().copied().enumerate() {
        let row = usize::try_from(base.row).map_err(|_| IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::DpCells,
        })?;
        if row >= width
            || !base.cost.is_finite()
            || base.cost > cutoff
            || output
                .last()
                .is_some_and(|previous| previous.row >= base.row)
        {
            return Err(IncompleteReason::InvalidStoredData);
        }
        push_position(output, width, base)?;
        let stop = canonical
            .get(index + 1)
            .map_or(width, |next| next.row as usize);
        for vertical_row in row.saturating_add(1)..stop {
            let previous_cost = output
                .last()
                .ok_or(IncompleteReason::InvalidStoredData)?
                .cost;
            let cost = previous_cost + query_delete_cost_at(config, query, vertical_row)?;
            if !cost.is_finite() {
                return Err(IncompleteReason::NumericOverflow);
            }
            if cost > cutoff {
                break;
            }
            push_position(
                output,
                width,
                ProductPosition {
                    row: u32::try_from(vertical_row).map_err(|_| {
                        IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::DpCells,
                        }
                    })?,
                    cost,
                },
            )?;
        }
    }
    let final_row =
        u32::try_from(query.values().len()).map_err(|_| IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::DpCells,
        })?;
    let reconstructed_final = output
        .last()
        .filter(|position| position.row == final_row)
        .map_or(f64::INFINITY, |position| position.cost);
    if canonical_bits(reconstructed_final) != canonical_bits(final_cost) {
        return Err(IncompleteReason::InvalidStoredData);
    }
    Ok(())
}

fn build_scheduled_rows(
    source: &[ProductPosition],
    width: usize,
    rows: &mut Vec<u32>,
) -> Result<(), IncompleteReason> {
    rows.clear();
    for position in source {
        let row =
            usize::try_from(position.row).map_err(|_| IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::DpCells,
            })?;
        if row >= width {
            return Err(IncompleteReason::InvalidStoredData);
        }
        for scheduled in [row, row.saturating_add(1)] {
            if scheduled >= width {
                continue;
            }
            let scheduled =
                u32::try_from(scheduled).map_err(|_| IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                })?;
            if rows.last().copied() != Some(scheduled) {
                reserve_row_slot(rows, width)?;
                rows.push(scheduled);
            }
        }
    }
    Ok(())
}

#[inline]
fn frontier_cost(frontier: &[ProductPosition], row: usize) -> f64 {
    let Ok(row) = u32::try_from(row) else {
        return f64::INFINITY;
    };
    frontier
        .binary_search_by_key(&row, |position| position.row)
        .ok()
        .map_or(f64::INFINITY, |index| frontier[index].cost)
}

#[allow(clippy::too_many_arguments)]
fn step_sparse_interval_frontier(
    config: &MetricTimestampedTwedConfig,
    query: &TimestampedSeries,
    current: TimestampedScalarBox,
    previous: TimestampedScalarBox,
    candidate_delete: f64,
    cutoff: f64,
    source: &[ProductPosition],
    scheduled_rows: &[u32],
    output: &mut Vec<ProductPosition>,
) -> Result<(), IncompleteReason> {
    output.clear();
    let width =
        query
            .values()
            .len()
            .checked_add(1)
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::DpCells,
            })?;
    let mut scheduled_index = 0_usize;
    let mut vertical_row = None::<usize>;
    let mut last_evaluated = None::<usize>;

    while scheduled_index < scheduled_rows.len() || vertical_row.is_some() {
        let scheduled = scheduled_rows.get(scheduled_index).map(|row| *row as usize);
        let row = match (scheduled, vertical_row) {
            (Some(scheduled), Some(vertical)) => scheduled.min(vertical),
            (Some(scheduled), None) => scheduled,
            (None, Some(vertical)) => vertical,
            (None, None) => break,
        };
        while scheduled_rows
            .get(scheduled_index)
            .is_some_and(|scheduled| *scheduled as usize == row)
        {
            scheduled_index += 1;
        }
        if vertical_row == Some(row) {
            vertical_row = None;
        }
        if last_evaluated == Some(row) {
            continue;
        }
        last_evaluated = Some(row);

        let cost = if row == 0 {
            frontier_cost(source, 0) + candidate_delete
        } else {
            let query_value = query.values()[row - 1];
            let query_time = query.timestamps()[row - 1];
            let query_previous_value = if row == 1 {
                0.0
            } else {
                query.values()[row - 2]
            };
            let query_previous_time = if row == 1 {
                query.origin()
            } else {
                query.timestamps()[row - 2]
            };
            let pair = frontier_cost(source, row - 1)
                + config
                    .interval_match_lower_bound(
                        query_value,
                        query_previous_value,
                        query_time,
                        query_previous_time,
                        query.unit(),
                        current,
                        previous,
                    )
                    .map_err(|_| IncompleteReason::InvalidStoredData)?;
            let delete_query = output
                .last()
                .filter(|position| position.row as usize + 1 == row)
                .map_or(f64::INFINITY, |position| {
                    position.cost
                        + delete_cost(
                            query_value,
                            query_previous_value,
                            query_time,
                            query_previous_time,
                            config,
                        )
                });
            let delete_candidate = frontier_cost(source, row) + candidate_delete;
            pair.min(delete_query).min(delete_candidate)
        };
        if cost.is_nan() {
            return Err(IncompleteReason::NumericOverflow);
        }
        if cost.is_finite() && cost <= cutoff {
            push_position(
                output,
                width,
                ProductPosition {
                    row: u32::try_from(row).map_err(|_| IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::DpCells,
                    })?,
                    cost,
                },
            )?;
            if row + 1 < width {
                vertical_row = Some(row + 1);
            }
        }
    }
    Ok(())
}

fn canonicalize_frontier(
    config: &MetricTimestampedTwedConfig,
    query: &TimestampedSeries,
    active: &[ProductPosition],
    output: &mut Vec<ProductPosition>,
) -> Result<(), IncompleteReason> {
    output.clear();
    let width =
        query
            .values()
            .len()
            .checked_add(1)
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::DpCells,
            })?;
    for (index, position) in active.iter().copied().enumerate() {
        let dominated = index.checked_sub(1).is_some_and(|previous_index| {
            let previous = active[previous_index];
            if previous.row.checked_add(1) != Some(position.row) {
                return false;
            }
            let Ok(vertical) = query_delete_cost_at(config, query, position.row as usize) else {
                return false;
            };
            exact_vertical_subsumes(previous.cost, vertical, position.cost)
        });
        if !dominated {
            push_position(output, width, position)?;
        }
    }
    if active.is_empty() != output.is_empty() {
        return Err(IncompleteReason::InvalidStoredData);
    }
    Ok(())
}

#[inline]
fn exact_vertical_subsumes(previous_cost: f64, vertical_cost: f64, current_cost: f64) -> bool {
    canonical_bits(previous_cost + vertical_cost) == canonical_bits(current_cost)
}

fn finite_positive_width(domain: (f64, f64)) -> bool {
    domain.0.is_finite()
        && domain.1.is_finite()
        && domain.0 < domain.1
        && (domain.1 - domain.0).is_finite()
}

fn unit_tag(unit: TimestampUnit) -> u64 {
    match unit {
        TimestampUnit::Seconds => 0,
        TimestampUnit::Milliseconds => 1,
        TimestampUnit::Microseconds => 2,
        TimestampUnit::Nanoseconds => 3,
    }
}

fn bin_index(value: f64, domain: (f64, f64), bins: u32) -> Option<u32> {
    if !value.is_finite() || value < domain.0 || value > domain.1 {
        return None;
    }
    if value == domain.1 {
        return Some(bins - 1);
    }
    let scaled = (value - domain.0) / (domain.1 - domain.0) * f64::from(bins);
    let bin = scaled.floor();
    if bin < 0.0 || bin >= f64::from(bins) {
        None
    } else {
        Some(bin as u32)
    }
}

fn bin_interval(domain: (f64, f64), bins: u32, bin: u32) -> (f64, f64) {
    let width = (domain.1 - domain.0) / f64::from(bins);
    let lower = if bin == 0 {
        domain.0
    } else {
        (domain.0 + width * f64::from(bin)).next_down()
    };
    let upper = if bin + 1 == bins {
        domain.1
    } else {
        (domain.0 + width * f64::from(bin + 1)).next_up()
    };
    (lower, upper)
}

fn canonical_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

fn fallible_f64_buffer(len: usize, resource: ResourceKind) -> Result<Vec<f64>, IncompleteReason> {
    let bytes = len
        .checked_mul(size_of::<f64>())
        .ok_or(IncompleteReason::ArithmeticOverflow { resource })?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(len)
        .map_err(|_| IncompleteReason::AllocationFailed {
            resource,
            requested: bytes,
        })?;
    buffer.resize(len, f64::INFINITY);
    Ok(buffer)
}

fn exact_work(query_len: usize, candidate_len: usize) -> Result<usize, IncompleteReason> {
    query_len
        .checked_mul(candidate_len)
        .and_then(|inner| inner.checked_add(query_len))
        .and_then(|work| work.checked_add(candidate_len))
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::WorkUnits,
        })
}

fn exact_cells(query_len: usize, candidate_len: usize) -> Result<usize, IncompleteReason> {
    query_len
        .checked_add(1)
        .and_then(|rows| {
            candidate_len
                .checked_add(1)
                .and_then(|columns| rows.checked_mul(columns))
        })
        .ok_or(IncompleteReason::ArithmeticOverflow {
            resource: ResourceKind::DpCells,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamped_knn_push_preserves_the_total_max_heap_order() {
        let series = TimestampedSeries::try_new(
            &[0.0],
            &[0.0],
            TimestampUnit::Seconds,
            ResourceLimits::default(),
        )
        .unwrap();
        let value = ();
        let mut heap = Vec::new();
        // This insertion order distinguishes the correct zero-based parent
        // `(child - 1) / 2` from the superficially plausible one-based formula
        // `child / 2`: the latter leaves value 4 below parent value 3.
        for (episode_id, distance) in [1.0, 4.0, 5.0, 6.0, 2.0, 3.0, 7.0].into_iter().enumerate() {
            timestamped_knn_heap_push(
                &mut heap,
                TimestampedTwedRangeMatch {
                    episode_id: episode_id as u64,
                    value: &value,
                    series: &series,
                    distance,
                },
            );
            for child in 1..heap.len() {
                let parent = (child - 1) / 2;
                let independent_order = heap[parent]
                    .distance
                    .total_cmp(&heap[child].distance)
                    .then_with(|| heap[parent].episode_id.cmp(&heap[child].episode_id));
                assert!(!independent_order.is_lt());
            }
        }
        assert_eq!((heap[0].distance, heap[0].episode_id), (7.0, 6));

        heap[0].distance = -1.0;
        timestamped_knn_heap_sift_down(&mut heap, 0);
        for child in 1..heap.len() {
            let parent = (child - 1) / 2;
            let independent_order = heap[parent]
                .distance
                .total_cmp(&heap[child].distance)
                .then_with(|| heap[parent].episode_id.cmp(&heap[child].episode_id));
            assert!(!independent_order.is_lt());
        }

        // Force sift-down through index 2 when it has a left child at 5 and no
        // right child at 6. The right-bound guard must short-circuit before
        // indexing that absent child.
        let mut even_heap = Vec::new();
        for (episode_id, distance) in [100.0, 80.0, 90.0, 70.0, 60.0, 85.0]
            .into_iter()
            .enumerate()
        {
            even_heap.push(TimestampedTwedRangeMatch {
                episode_id: episode_id as u64,
                value: &value,
                series: &series,
                distance,
            });
        }
        even_heap[0].distance = -1.0;
        timestamped_knn_heap_sift_down(&mut even_heap, 0);
        for child in 1..even_heap.len() {
            let parent = (child - 1) / 2;
            let independent_order = even_heap[parent]
                .distance
                .total_cmp(&even_heap[child].distance)
                .then_with(|| {
                    even_heap[parent]
                        .episode_id
                        .cmp(&even_heap[child].episode_id)
                });
            assert!(!independent_order.is_lt());
        }
    }

    #[test]
    fn exact_equality_defeats_forced_fingerprint_bucket_collisions() {
        let left_positions = [
            ProductPosition { row: 0, cost: 0.0 },
            ProductPosition { row: 1, cost: 1.0 },
        ];
        let right_positions = [
            ProductPosition { row: 0, cost: 0.0 },
            ProductPosition { row: 1, cost: 2.0 },
        ];
        let mut arena = ProductStateArena::new(8, 16);
        let left = arena
            .intern_with_fingerprint(Some(1), &left_positions, 1.0, 0)
            .unwrap();
        let right = arena
            .intern_with_fingerprint(Some(2), &right_positions, 2.0, 0)
            .unwrap();
        let reused = arena
            .intern_with_fingerprint(Some(1), &left_positions, 1.0, 0)
            .unwrap();

        assert_ne!(left, right);
        assert_eq!(reused, left);
        assert_eq!(arena.reused_states, 1);
        assert!(!ProductStateArena::exact_equal(
            arena.get(left).unwrap(),
            Some(2),
            &right_positions,
            2.0,
        ));
    }

    #[test]
    fn typed_token_carries_unit_value_and_time() {
        let quantizer = TimestampedTwedQuantizer::try_new(
            TimestampUnit::Milliseconds,
            0.0,
            (-2.0, 2.0),
            (0.0, 8.0),
            4,
            8,
        )
        .unwrap();
        let token = quantizer.encode(0.25, 3.25).unwrap();
        assert_eq!(token >> 62, 1);
        let interval = quantizer.decode(token).unwrap();
        assert!(interval.value_interval().0 <= 0.25);
        assert!(interval.value_interval().1 >= 0.25);
        assert!(interval.time_interval().0 <= 3.25);
        assert!(interval.time_interval().1 >= 3.25);
    }

    #[test]
    fn exact_vertical_subsumption_is_bitwise_and_omission_is_detectable() {
        assert!(exact_vertical_subsumes(1.0, 2.0, 3.0));
        assert!(!exact_vertical_subsumes(1.0, 2.0, 3.0_f64.next_up()));
        assert!(exact_vertical_subsumes(-0.0, 0.0, 0.0));
    }
}
