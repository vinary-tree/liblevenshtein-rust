//! Generic exact elastic-distance retrieval over a trie of quantized series.
//!
//! [`ElasticTransducer`] indexes reference time series as quantized byte
//! sequences in a [`DynamicDawg`]. A kernel supplies the interval-relaxed
//! dynamic-programming transition and exact scorer; the walker supplies all
//! storage, prefix sharing, pruning, and result ordering.
//!
//! # Guarantee
//!
//! [`ElasticTransducer::search_range`] returns *exactly* the set
//! `{ id : D(query, reference_id) ≤ τ }` — no false negatives and no false
//! positives — and [`ElasticTransducer::search_knn`] returns exactly the `k`
//! nearest finite-distance references under the kernel.
//!
//! # How it stays exact while pruning
//!
//! Descending one trie edge consumes one target element, known only up to its
//! quantization bin `[lo, hi]`. Each trie node at depth `d` therefore carries
//! a relaxed DP column for the query against the first `d` target elements.
//! Two consequences make the walk exact:
//!
//! * **Sound pruning (no false negatives).** The minimum live cell of a node's
//!   column lower-bounds the true distance of *every* reference reachable below
//!   it (K1), and lawful extensions cannot reduce accumulated cost (K2). A
//!   subtree whose bound exceeds `τ` or the running k-th best is safely skipped.
//! * **Exact verification (no false positives).** At each final node whose
//!   column lower bound is within threshold, the candidate is re-scored against
//!   the stored **full-precision** original with
//!   [`ElasticKernel::exact_with_cutoff`]; only genuine matches are emitted
//!   (K3), after an optional admissible candidate bound (K4).

mod snapshot;
pub use snapshot::{
    ElasticSnapshot, ElasticSnapshotError, ElasticSnapshotIdentity, ElasticSnapshotKernel,
    ElasticSnapshotMetadata,
};

use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap, VecDeque};

use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgNode};
use libdictenstein::{Dictionary, DictionaryNode, MappedDictionaryNode};

use super::{Cost, ElasticKernel, PointFrontierStep};
use crate::cost::CostMonoid;
use crate::time_series::automaton::{ErpFrontierMachine, TemporalAutomatonError, TemporalStateId};
use crate::time_series::bounded::{
    IncompleteReason, Operand, OperationOutcome, PageBudget, ResourceKind, ResourceLedger,
    ResourceLimits, ResourceUsage, TemporalValidationError,
};
use crate::time_series::encoding::QuantizationConfig;
use crate::time_series::kernels::ErpConfig;
use crate::time_series::msm::MsmConfig;
use crate::time_series::msm_kernel::MsmKernel;

const DEFAULT_RESULT_BUFFER_CAPACITY: usize = 64;

type BucketLocation = (usize, usize);

#[inline]
fn take_sequence(counter: &mut usize) -> Option<usize> {
    let sequence = *counter;
    *counter = sequence.checked_add(1)?;
    Some(sequence)
}

#[inline]
fn next_sequence(counter: &mut usize) -> Option<usize> {
    *counter = counter.checked_add(1)?;
    Some(*counter)
}

#[inline]
fn byte_unit_index(unit: u8) -> usize {
    usize::from(unit)
}

#[inline]
fn byte_unit_bin(unit: u8) -> u32 {
    u32::from(unit)
}

#[derive(Debug)]
struct StoredSeries {
    series: Vec<f64>,
    bucket_location: BucketLocation,
}

struct KnnQueueNode<K: ElasticKernel, N> {
    lower_bound: Cost<K>,
    sequence: usize,
    depth: usize,
    node: N,
    column: Vec<Cost<K>>,
    carry: Option<K::Carry>,
}

impl<K: ElasticKernel, N> PartialEq for KnnQueueNode<K, N> {
    fn eq(&self, other: &Self) -> bool {
        K::Monoid::compare(self.lower_bound, other.lower_bound) == Ordering::Equal
            && self.sequence == other.sequence
    }
}

impl<K: ElasticKernel, N> Eq for KnnQueueNode<K, N> {}

impl<K: ElasticKernel, N> PartialOrd for KnnQueueNode<K, N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: ElasticKernel, N> Ord for KnnQueueNode<K, N> {
    fn cmp(&self, other: &Self) -> Ordering {
        K::Monoid::compare(other.lower_bound, self.lower_bound)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

#[derive(Clone)]
struct KnnBestResult<K: ElasticKernel, V> {
    distance: Cost<K>,
    sequence: usize,
    value: V,
}

/// Observational counters for one exact elastic k-nearest-neighbour search.
///
/// These counters do not participate in traversal decisions. They expose the
/// pruning economics needed by reproducible experiments while
/// [`ElasticTransducer::search_knn`] keeps its original result-only API.
/// `visited_nodes` counts expanded queue nodes (including the root), whereas
/// `queued_subtrees_pruned` counts nodes discarded after a tighter k-th-best
/// cutoff makes the queue minimum inadmissible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElasticSearchStats {
    /// Priority-queue nodes expanded by the best-first traversal.
    pub visited_nodes: usize,
    /// Outgoing trie edges inspected at expanded nodes.
    pub visited_edges: usize,
    /// Edges rejected by a constant-time prefix lower bound.
    pub prefix_pruned: usize,
    /// Interval DP columns constructed after the prefix gate.
    pub columns_built: usize,
    /// Constructed columns rejected by their subtree lower bound.
    pub column_pruned: usize,
    /// Already-queued subtrees rejected after the k-th-best cutoff tightened.
    pub queued_subtrees_pruned: usize,
    /// Full-precision candidate records considered at admitted final nodes.
    pub candidates_considered: usize,
    /// Candidates rejected by their kernel-specific K4 lower bound.
    pub candidate_bound_pruned: usize,
    /// Exact dynamic-programming evaluations attempted after all lower bounds.
    pub exact_evaluations: usize,
    /// Exact evaluations that returned no value within the current cutoff.
    pub cutoff_abandoned: usize,
}

impl ElasticSearchStats {
    /// Whether the two accounting partitions are internally consistent.
    ///
    /// Every visited edge is either prefix-pruned or receives one column, and
    /// every considered candidate is either candidate-bound-pruned or exactly
    /// evaluated. The remaining inequalities express subset relationships.
    #[must_use]
    pub fn accounting_is_consistent(&self) -> bool {
        self.prefix_pruned
            .checked_add(self.columns_built)
            .is_some_and(|total| total == self.visited_edges)
            && self
                .candidate_bound_pruned
                .checked_add(self.exact_evaluations)
                .is_some_and(|total| total == self.candidates_considered)
            && self.column_pruned <= self.columns_built
            && self.cutoff_abandoned <= self.exact_evaluations
    }
}

impl<K: ElasticKernel, V> PartialEq for KnnBestResult<K, V> {
    fn eq(&self, other: &Self) -> bool {
        K::Monoid::compare(self.distance, other.distance) == Ordering::Equal
            && self.sequence == other.sequence
    }
}

impl<K: ElasticKernel, V> Eq for KnnBestResult<K, V> {}

impl<K: ElasticKernel, V> PartialOrd for KnnBestResult<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: ElasticKernel, V> Ord for KnnBestResult<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        K::Monoid::compare(self.distance, other.distance)
            .then_with(|| self.sequence.cmp(&other.sequence))
    }
}

/// An exact elastic-distance index over quantized reference series.
///
/// `V` is the identifier associated with each reference (default `usize`).
/// Multiple references that quantize to the same byte key are all retained and
/// individually verified, so quantization collisions never silently drop a
/// result.
#[derive(Debug)]
pub struct ElasticTransducer<K: ElasticKernel, V: Eq + std::hash::Hash + Clone = usize> {
    /// Prefix-sharing trie over the u8-quantized reference sequences. The
    /// stored value is the final bucket id for all references that quantize to
    /// that byte key.
    dawg: DynamicDawg<usize>,
    /// Quantization configuration (defines the per-bin intervals).
    quant: QuantizationConfig,
    /// Kernel supplying relaxed columns and exact verification.
    kernel: K,
    /// Precomputed quantization-bin intervals for hot trie-edge traversal.
    bin_bounds: Vec<(f64, f64)>,
    /// Final bucket id → all reference ids sharing that quantized key.
    buckets: Vec<Vec<V>>,
    /// Reference id → original series and bucket slot (for exact verification and O(1) upserts).
    originals: HashMap<V, StoredSeries>,
}

struct RangeWalkContext<'a, K: ElasticKernel, V> {
    query: &'a [f64],
    plan: &'a K::QueryPlan,
    tau: Cost<K>,
    out: &'a mut Vec<(V, Cost<K>)>,
    column_width: usize,
}

/// One explicit DFS frame for stack-safe range traversal.
struct RangeFrame<K: ElasticKernel, N> {
    depth: usize,
    carry: Option<K::Carry>,
    column: Vec<Cost<K>>,
    final_bucket: Option<usize>,
    next_candidate: usize,
    edges: VecDeque<(u8, N)>,
}

/// Explicit bounded-traversal frame containing only a compact product-state
/// identifier and dictionary cursor state.
struct BoundedRangeFrame<N> {
    depth: usize,
    state: TemporalStateId,
    final_bucket: Option<usize>,
    next_candidate: usize,
    edges: VecDeque<(u8, N)>,
}

/// One live kernel state in the on-the-fly dictionary product.
struct RangeProductPosition<K: ElasticKernel> {
    row: u32,
    cost: Cost<K>,
}

struct RangeProductState<K: ElasticKernel> {
    carry: Option<K::Carry>,
    positions: Vec<RangeProductPosition<K>>,
}

/// Retained-state diagnostics for one on-the-fly dictionary product.
///
/// A trie traversal owns exactly one product state per live explicit DFS
/// frame. `column_cells` counts the kernel cells retained by those states; it
/// is independent of the number of target symbols consumed before the current
/// point in a streaming traversal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElasticProductStateStats {
    /// Live explicit dictionary-traversal frames.
    pub frames: usize,
    /// Live query-automaton states addressed by compact IDs.
    pub states: usize,
    /// Kernel cells retained by all live product states.
    pub column_cells: usize,
}

enum RangeSessionMode<K: ElasticKernel> {
    Trie {
        stack: Vec<BoundedRangeFrame<DynamicDawgNode<usize>>>,
        states: Vec<RangeProductState<K>>,
        column_width: usize,
        previous: Vec<Cost<K>>,
        next: Vec<Cost<K>>,
        previous_active: Vec<usize>,
        next_active: Vec<usize>,
    },
    Scan {
        bucket: usize,
        slot: usize,
    },
    Done,
}

/// Explicit stack frame for the lazy ERP automaton/dictionary product.
///
/// Unlike [`RangeFrame`], this frame owns no dynamic-programming column. The
/// query-local automaton arena owns each canonical antichain once and the DFS
/// stack refers to it by a compact, collision-checked ID.
struct ErpAutomatonRangeFrame {
    depth: usize,
    state: TemporalStateId,
    final_bucket: Option<usize>,
    next_candidate: usize,
    edges: VecDeque<(u8, DynamicDawgNode<usize>)>,
}

/// Resumable exact ERP range traversal using an on-demand automaton product.
///
/// Each dictionary edge constructs only the reachable canonical ERP frontier.
/// Stack depth is represented by an explicit `Vec`, so traversal never uses
/// the call stack. Quantization collisions remain candidates until every
/// full-precision original has been independently verified.
pub struct ErpAutomatonRangeContinuation<'a, V>
where
    V: Eq + std::hash::Hash + Clone,
{
    index: &'a ElasticTransducer<ErpConfig, V>,
    query: Box<[f64]>,
    tau: f64,
    machine: ErpFrontierMachine,
    stack: Vec<ErpAutomatonRangeFrame>,
    results: Vec<(V, f64)>,
    pending_match: Option<(V, f64)>,
    ledger: ResourceLedger,
    terminal: Option<IncompleteReason>,
    done: bool,
}

impl<V> std::fmt::Debug for ErpAutomatonRangeContinuation<'_, V>
where
    V: Eq + std::hash::Hash + Clone,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (states, positions, cached_transitions) = self.machine.retained_counts();
        formatter
            .debug_struct("ErpAutomatonRangeContinuation")
            .field("query_len", &self.query.len())
            .field("stack_depth", &self.stack.len())
            .field("canonical_states", &states)
            .field("canonical_positions", &positions)
            .field("cached_transitions", &cached_transitions)
            .field("result_count", &self.results.len())
            .field("usage", &self.ledger.usage())
            .field("terminal", &self.terminal)
            .field("done", &self.done)
            .finish()
    }
}

/// In-memory continuation for one exact bounded range query.
///
/// The continuation immutably borrows its index, so Rust prevents mutation of
/// the indexed snapshot between pages. Partial results are exact members found
/// so far but do not establish absence until [`OperationOutcome::Complete`].
pub struct RangeContinuation<'a, K, V>
where
    K: ElasticKernel,
    V: Eq + std::hash::Hash + Clone,
{
    index: &'a ElasticTransducer<K, V>,
    query: Box<[f64]>,
    plan: K::QueryPlan,
    tau: Cost<K>,
    mode: RangeSessionMode<K>,
    results: Vec<(V, Cost<K>)>,
    pending_match: Option<(V, Cost<K>)>,
    ledger: ResourceLedger,
    terminal: Option<IncompleteReason>,
}

/// Exact identifier/distance pairs returned by elastic range and kNN queries.
pub type ExactRangeResults<K, V> = Vec<(V, Cost<K>)>;

/// Tagged outcome of a bounded, resumable exact elastic range query.
pub type BoundedRangeOutcome<'a, K, V> =
    OperationOutcome<ExactRangeResults<K, V>, RangeContinuation<'a, K, V>>;

/// Tagged outcome of a bounded non-resumable exact elastic search.
pub type ExactSearchOutcome<K, V> = OperationOutcome<ExactRangeResults<K, V>>;

/// Tagged outcome of the specialized canonical ERP automaton product.
pub type ErpAutomatonRangeOutcome<'a, V> =
    OperationOutcome<Vec<(V, f64)>, ErpAutomatonRangeContinuation<'a, V>>;

impl<K, V> std::fmt::Debug for RangeContinuation<'_, K, V>
where
    K: ElasticKernel,
    V: Eq + std::hash::Hash + Clone,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RangeContinuation")
            .field("query_len", &self.query.len())
            .field("result_count", &self.results.len())
            .field("usage", &self.ledger.usage())
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
enum CandidateSource {
    Trie,
    Scan,
}

impl<'a, K, V> RangeContinuation<'a, K, V>
where
    K: ElasticKernel,
    V: Eq + std::hash::Hash + Clone,
{
    /// Exact matches discovered so far. This is an exact subset, but it is
    /// complete only after [`Self::resume`] returns `Complete`.
    pub fn exact_partial(&self) -> &[(V, Cost<K>)] {
        &self.results
    }

    /// Cumulative charged/peak usage for this query session.
    pub fn usage(&self) -> ResourceUsage {
        self.ledger.usage()
    }

    /// Return the live dictionary × kernel product-state footprint.
    ///
    /// During interval-trie traversal `frames == states`: the state arena is a
    /// stack arena whose last state is reclaimed with its DFS frame. Exact
    /// scan fallback and completed sessions retain no product states.
    pub fn retained_product_state_stats(&self) -> ElasticProductStateStats {
        match &self.mode {
            RangeSessionMode::Trie {
                stack,
                states,
                column_width: _,
                previous: _,
                next: _,
                previous_active: _,
                next_active: _,
            } => ElasticProductStateStats {
                frames: stack.len(),
                states: states.len(),
                column_cells: states
                    .iter()
                    .map(|state| state.positions.len())
                    .fold(0usize, usize::saturating_add),
            },
            RangeSessionMode::Scan { .. } | RangeSessionMode::Done => {
                ElasticProductStateStats::default()
            }
        }
    }

    /// Resume this exact query for one bounded page of work.
    pub fn resume(mut self, page: PageBudget) -> OperationOutcome<Vec<(V, Cost<K>)>, Self> {
        let step = self.advance(page);
        match step {
            OperationOutcome::Complete { usage, .. } => OperationOutcome::Complete {
                value: std::mem::take(&mut self.results),
                usage,
            },
            OperationOutcome::Incomplete {
                reason,
                continuation,
                usage,
                ..
            } => {
                let partial = Some(ElasticTransducer::<K, V>::finish_range_results(
                    self.results.clone(),
                ));
                OperationOutcome::Incomplete {
                    partial,
                    reason,
                    continuation: continuation.map(|()| self),
                    usage,
                }
            }
        }
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

    fn page_allows(
        &self,
        used: usize,
        amount: usize,
        limit: usize,
        resource: ResourceKind,
    ) -> Result<(), IncompleteReason> {
        let Some(requested) = used.checked_add(amount) else {
            return Err(IncompleteReason::ArithmeticOverflow { resource });
        };
        if requested > limit {
            return Err(IncompleteReason::BudgetExceeded {
                resource,
                limit,
                requested,
            });
        }
        Ok(())
    }

    fn observe_state_peaks(&mut self) -> Result<(), IncompleteReason> {
        let (queue_entries, automaton_bytes, edge_bytes) = match &self.mode {
            RangeSessionMode::Trie {
                stack,
                states,
                previous,
                next,
                previous_active,
                next_active,
                ..
            } => {
                let mut position_bytes = 0usize;
                let mut edge_bytes = 0usize;
                for state in states {
                    position_bytes = position_bytes
                        .checked_add(
                            state
                                .positions
                                .capacity()
                                .checked_mul(std::mem::size_of::<RangeProductPosition<K>>())
                                .ok_or(IncompleteReason::ArithmeticOverflow {
                                    resource: ResourceKind::ScratchBytes,
                                })?,
                        )
                        .ok_or(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::ScratchBytes,
                        })?;
                }
                let dense_bytes = previous
                    .capacity()
                    .checked_add(next.capacity())
                    .and_then(|cells| cells.checked_mul(std::mem::size_of::<Cost<K>>()))
                    .ok_or(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    })?;
                let active_bytes = previous_active
                    .capacity()
                    .checked_add(next_active.capacity())
                    .and_then(|cells| cells.checked_mul(std::mem::size_of::<usize>()))
                    .ok_or(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    })?;
                let automaton_bytes = position_bytes
                    .checked_add(dense_bytes)
                    .and_then(|bytes| bytes.checked_add(active_bytes))
                    .ok_or(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    })?;
                for frame in stack {
                    edge_bytes = edge_bytes
                        .checked_add(
                            frame
                                .edges
                                .capacity()
                                .checked_mul(std::mem::size_of::<(u8, DynamicDawgNode<usize>)>())
                                .ok_or(IncompleteReason::ArithmeticOverflow {
                                    resource: ResourceKind::ContinuationBytes,
                                })?,
                        )
                        .ok_or(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::ContinuationBytes,
                        })?;
                }
                debug_assert_eq!(stack.len(), states.len());
                (stack.len(), automaton_bytes, edge_bytes)
            }
            RangeSessionMode::Scan { .. } | RangeSessionMode::Done => (0, 0, 0),
        };

        self.ledger
            .observe_peak(ResourceKind::QueueEntries, queue_entries)?;
        self.ledger
            .observe_peak(ResourceKind::ScratchBytes, automaton_bytes)?;

        let query_bytes = self
            .query
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        let result_bytes = self
            .results
            .capacity()
            .checked_mul(std::mem::size_of::<(V, Cost<K>)>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        let frame_bytes = match &self.mode {
            RangeSessionMode::Trie { stack, .. } => stack
                .capacity()
                .checked_mul(std::mem::size_of::<BoundedRangeFrame<DynamicDawgNode<usize>>>())
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ContinuationBytes,
                })?,
            RangeSessionMode::Scan { .. } | RangeSessionMode::Done => 0,
        };
        let state_header_bytes = match &self.mode {
            RangeSessionMode::Trie { states, .. } => states
                .capacity()
                .checked_mul(std::mem::size_of::<RangeProductState<K>>())
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ContinuationBytes,
                })?,
            RangeSessionMode::Scan { .. } | RangeSessionMode::Done => 0,
        };
        let continuation_bytes = query_bytes
            .checked_add(automaton_bytes)
            .and_then(|bytes| bytes.checked_add(edge_bytes))
            .and_then(|bytes| bytes.checked_add(frame_bytes))
            .and_then(|bytes| bytes.checked_add(state_header_bytes))
            .and_then(|bytes| bytes.checked_add(result_bytes))
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        self.ledger
            .observe_peak(ResourceKind::ContinuationBytes, continuation_bytes)
    }

    fn advance(&mut self, page: PageBudget) -> OperationOutcome<(), ()> {
        if let Some(reason) = self.terminal {
            return self.terminate(reason);
        }

        let mut page_work = 0usize;
        let mut page_results = 0usize;
        loop {
            if self.pending_match.is_some() {
                if let Err(reason) =
                    self.page_allows(page_results, 1, page.max_results, ResourceKind::Results)
                {
                    return match reason {
                        IncompleteReason::BudgetExceeded {
                            resource,
                            limit,
                            requested,
                        } => self.paused(resource, limit, requested),
                        other => self.terminate(other),
                    };
                }
                if let Err(reason) = self.ledger.charge(ResourceKind::Results, 1) {
                    return self.terminate(reason);
                }
                self.results.push(
                    self.pending_match
                        .take()
                        .expect("pending match checked above"),
                );
                page_results += 1;
                if let Err(reason) = self.observe_state_peaks() {
                    return self.terminate(reason);
                }
                continue;
            }

            let next_candidate = match &mut self.mode {
                RangeSessionMode::Trie { stack, .. } => {
                    let Some(frame) = stack.last_mut() else {
                        self.mode = RangeSessionMode::Done;
                        continue;
                    };
                    if let Some(bucket_id) = frame.final_bucket {
                        match self.index.buckets.get(bucket_id) {
                            Some(ids) => match ids.get(frame.next_candidate) {
                                Some(id) => Some((id.clone(), CandidateSource::Trie)),
                                None => {
                                    frame.final_bucket = None;
                                    None
                                }
                            },
                            None => return self.terminate(IncompleteReason::InvalidStoredData),
                        }
                    } else {
                        None
                    }
                }
                RangeSessionMode::Scan { bucket, slot } => loop {
                    let Some(ids) = self.index.buckets.get(*bucket) else {
                        self.mode = RangeSessionMode::Done;
                        break None;
                    };
                    if let Some(id) = ids.get(*slot) {
                        break Some((id.clone(), CandidateSource::Scan));
                    }
                    *bucket = match bucket.checked_add(1) {
                        Some(next) => next,
                        None => {
                            return self.terminate(IncompleteReason::ArithmeticOverflow {
                                resource: ResourceKind::Candidates,
                            });
                        }
                    };
                    *slot = 0;
                },
                RangeSessionMode::Done => {
                    self.results = ElasticTransducer::<K, V>::finish_range_results(std::mem::take(
                        &mut self.results,
                    ));
                    return OperationOutcome::Complete {
                        value: (),
                        usage: self.ledger.usage(),
                    };
                }
            };

            if let Some((id, source)) = next_candidate {
                let Some(stored) = self.index.originals.get(&id) else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                if stored.series.len() > self.ledger.limits().max_series_len
                    || stored.series.iter().any(|sample| !sample.is_finite())
                {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                }
                let candidate_work = match self
                    .query
                    .len()
                    .max(1)
                    .checked_mul(stored.series.len().max(1))
                {
                    Some(work) => work,
                    None => {
                        return self.terminate(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::DpCells,
                        });
                    }
                };
                if let Err(reason) = self.page_allows(
                    page_work,
                    candidate_work,
                    page.max_work_units,
                    ResourceKind::WorkUnits,
                ) {
                    return match reason {
                        IncompleteReason::BudgetExceeded {
                            resource,
                            limit,
                            requested,
                        } => self.paused(resource, limit, requested),
                        other => self.terminate(other),
                    };
                }
                if let Err(reason) = self.ledger.charge_many(&[
                    (ResourceKind::Candidates, 1),
                    (ResourceKind::DpCells, candidate_work),
                    (ResourceKind::WorkUnits, candidate_work),
                ]) {
                    return self.terminate(reason);
                }
                page_work += candidate_work;

                match (&mut self.mode, source) {
                    (RangeSessionMode::Trie { stack, .. }, CandidateSource::Trie) => {
                        let Some(frame) = stack.last_mut() else {
                            return self.terminate(IncompleteReason::InvalidStoredData);
                        };
                        frame.next_candidate = match frame.next_candidate.checked_add(1) {
                            Some(next) => next,
                            None => {
                                return self.terminate(IncompleteReason::ArithmeticOverflow {
                                    resource: ResourceKind::Candidates,
                                });
                            }
                        };
                    }
                    (RangeSessionMode::Scan { slot, .. }, CandidateSource::Scan) => {
                        *slot = match slot.checked_add(1) {
                            Some(next) => next,
                            None => {
                                return self.terminate(IncompleteReason::ArithmeticOverflow {
                                    resource: ResourceKind::Candidates,
                                });
                            }
                        };
                    }
                    _ => return self.terminate(IncompleteReason::InvalidStoredData),
                }

                let candidate_bound = self.index.kernel.candidate_lower_bound(
                    &self.query,
                    &stored.series,
                    &self.plan,
                );
                if !K::Monoid::within(candidate_bound, self.tau) {
                    continue;
                }
                let Some(exact) =
                    self.index
                        .kernel
                        .exact_with_cutoff(&self.query, &stored.series, self.tau)
                else {
                    continue;
                };
                if K::Monoid::compare(exact, K::Monoid::TOP) != Ordering::Less {
                    return self.terminate(IncompleteReason::NumericOverflow);
                }
                if !K::Monoid::within(exact, self.tau) {
                    continue;
                }
                if page_results >= page.max_results {
                    self.pending_match = Some((id, exact));
                    let requested = page_results.saturating_add(1);
                    return self.paused(ResourceKind::Results, page.max_results, requested);
                }
                if let Err(reason) = self.ledger.charge(ResourceKind::Results, 1) {
                    return self.terminate(reason);
                }
                self.results.push((id, exact));
                page_results += 1;
                if let Err(reason) = self.observe_state_peaks() {
                    return self.terminate(reason);
                }
                continue;
            }

            let edge_plan = match &self.mode {
                RangeSessionMode::Trie {
                    stack,
                    states,
                    column_width,
                    ..
                } => {
                    let Some(frame) = stack.last() else {
                        self.mode = RangeSessionMode::Done;
                        continue;
                    };
                    let Some((unit, child)) = frame.edges.front() else {
                        let RangeSessionMode::Trie { stack, states, .. } = &mut self.mode else {
                            unreachable!();
                        };
                        let popped = stack.pop().expect("last frame was observed above");
                        let expected = states
                            .len()
                            .checked_sub(1)
                            .and_then(|index| u32::try_from(index).ok())
                            .map(TemporalStateId);
                        debug_assert_eq!(expected, Some(popped.state));
                        states.pop();
                        continue;
                    };
                    let Some(child_depth) = frame.depth.checked_add(1) else {
                        return self.terminate(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::TrieNodes,
                        });
                    };
                    let Some(state) = states.get(frame.state.index()) else {
                        return self.terminate(IncompleteReason::InvalidStoredData);
                    };
                    let interval = self.index.bin_bounds_for(*unit);
                    let prefix_lower_bound = self.index.kernel.prefix_lower_bound(
                        &self.query,
                        interval,
                        state.carry,
                        child_depth,
                        &self.plan,
                    );
                    let build_column = K::Monoid::within(prefix_lower_bound, self.tau);
                    let work = if build_column {
                        let closure_and_step = if state.positions.is_empty() {
                            Some(*column_width)
                        } else {
                            column_width.checked_mul(2)
                        };
                        closure_and_step.and_then(|work| work.checked_add(1))
                    } else {
                        Some(1)
                    };
                    let Some(work) = work else {
                        return self.terminate(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::WorkUnits,
                        });
                    };
                    Some((
                        *unit,
                        child.clone(),
                        child_depth,
                        interval,
                        frame.state,
                        build_column,
                        work,
                    ))
                }
                RangeSessionMode::Scan { .. } => continue,
                RangeSessionMode::Done => continue,
            };

            let Some((unit, child, child_depth, interval, source, build_column, edge_work)) =
                edge_plan
            else {
                continue;
            };
            if let Err(reason) = self.page_allows(
                page_work,
                edge_work,
                page.max_work_units,
                ResourceKind::WorkUnits,
            ) {
                return match reason {
                    IncompleteReason::BudgetExceeded {
                        resource,
                        limit,
                        requested,
                    } => self.paused(resource, limit, requested),
                    other => self.terminate(other),
                };
            }
            if let Err(reason) = self.ledger.charge_many(&[
                (ResourceKind::TrieEdges, 1),
                (ResourceKind::WorkUnits, edge_work),
            ]) {
                return self.terminate(reason);
            }
            page_work += edge_work;

            let RangeSessionMode::Trie {
                stack,
                states,
                column_width,
                previous,
                next,
                previous_active,
                next_active,
            } = &mut self.mode
            else {
                return self.terminate(IncompleteReason::InvalidStoredData);
            };
            let Some(frame) = stack.last_mut() else {
                return self.terminate(IncompleteReason::InvalidStoredData);
            };
            if frame.edges.front().map(|(label, _)| *label) != Some(unit) {
                return self.terminate(IncompleteReason::InvalidStoredData);
            }
            frame.edges.pop_front();
            if !build_column {
                continue;
            }
            let Some(source_state) = states.get(source.index()) else {
                return self.terminate(IncompleteReason::InvalidStoredData);
            };
            while let Some(row) = previous_active.pop() {
                let Some(cost) = previous.get_mut(row) else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                *cost = K::Monoid::TOP;
            }
            for position in &source_state.positions {
                let row = usize::try_from(position.row).map_err(|_| {
                    IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::DpCells,
                    }
                });
                let row = match row {
                    Ok(row) if row < *column_width => row,
                    Ok(_) => return self.terminate(IncompleteReason::InvalidStoredData),
                    Err(reason) => return self.terminate(reason),
                };
                previous[row] = position.cost;
                previous_active.push(row);
            }
            // Reconstruct the epsilon closure represented by each retained
            // antichain position only when the next consuming transition
            // demands it. The scan stops at the next retained representative
            // or as soon as the non-negative closure leaves the cutoff.
            let held_target = source_state
                .carry
                .and_then(|carry| self.index.kernel.carry_interval(carry))
                .unwrap_or((0.0, 0.0));
            previous_active.clear();
            for (position_index, position) in source_state.positions.iter().enumerate() {
                let start = position.row as usize;
                previous_active.push(start);
                let stop = source_state
                    .positions
                    .get(position_index + 1)
                    .map_or(*column_width, |next_position| next_position.row as usize);
                for row in start.saturating_add(1)..stop {
                    let Some(vertical) = self.index.kernel.vertical_epsilon_extension(
                        &self.query,
                        held_target,
                        row,
                        previous,
                        &self.plan,
                    ) else {
                        break;
                    };
                    if !K::Monoid::within(vertical, self.tau) {
                        break;
                    }
                    previous[row] = vertical;
                    previous_active.push(row);
                }
            }
            debug_assert!(previous_active.windows(2).all(|pair| pair[0] < pair[1]));
            while let Some(row) = next_active.pop() {
                let Some(cost) = next.get_mut(row) else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                *cost = K::Monoid::TOP;
            }

            let sparse = self.index.kernel.step_interval_frontier(
                previous,
                previous_active,
                &self.query,
                interval,
                source_state.carry,
                child_depth,
                &self.plan,
                self.tau,
                *column_width,
                next,
                next_active,
            );
            let (lower_bound, carry) = match sparse {
                Some(PointFrontierStep::Advanced {
                    lower_bound,
                    carry,
                    work,
                }) => {
                    debug_assert!(work <= *column_width);
                    (lower_bound, carry)
                }
                Some(PointFrontierStep::WorkLimitExceeded {
                    completed: _,
                    requested,
                }) => {
                    let limit = *column_width;
                    return self.terminate(IncompleteReason::BudgetExceeded {
                        resource: ResourceKind::WorkUnits,
                        limit,
                        requested,
                    });
                }
                None => {
                    let (lower_bound, carry) = self.index.kernel.step_column(
                        previous,
                        &self.query,
                        interval,
                        source_state.carry,
                        child_depth,
                        &self.plan,
                        next,
                    );
                    next_active.extend(next.iter().enumerate().filter_map(|(row, cost)| {
                        K::Monoid::within(*cost, self.tau).then_some(row)
                    }));
                    (lower_bound, carry)
                }
            };
            if !K::Monoid::within(lower_bound, self.tau) {
                continue;
            }
            debug_assert!(next_active.windows(2).all(|pair| pair[0] < pair[1]));
            let final_row = self.index.kernel.final_row(self.query.len());
            let relaxed_admits = next
                .get(final_row)
                .is_some_and(|cost| K::Monoid::within(*cost, self.tau));

            // A row is removed only when the kernel reconstructs an explicit
            // immediate zero-input vertical extension with exactly the same
            // cost. The retained predecessor is therefore a formal residual
            // simulation witness; approximate floating-point comparisons are
            // never used for state identity or dominance.
            let canonical_position_count = next_active
                .iter()
                .filter(|row| {
                    let Some(cost) = next.get(**row).copied() else {
                        return true;
                    };
                    self.index
                        .kernel
                        .vertical_epsilon_extension(&self.query, interval, **row, next, &self.plan)
                        .is_none_or(|vertical| {
                            K::Monoid::compare(vertical, cost) != Ordering::Equal
                        })
                })
                .count();

            let existing_position_bytes = states.iter().try_fold(0usize, |bytes, state| {
                state
                    .positions
                    .capacity()
                    .checked_mul(std::mem::size_of::<RangeProductPosition<K>>())
                    .and_then(|state_bytes| bytes.checked_add(state_bytes))
                    .ok_or(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    })
            });
            let existing_position_bytes = match existing_position_bytes {
                Ok(bytes) => bytes,
                Err(reason) => return self.terminate(reason),
            };
            let requested_position_bytes = match canonical_position_count
                .checked_mul(std::mem::size_of::<RangeProductPosition<K>>())
                .and_then(|bytes| bytes.checked_add(existing_position_bytes))
            {
                Some(bytes) => bytes,
                None => {
                    return self.terminate(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    });
                }
            };
            let fixed_scratch_bytes = match column_width
                .checked_mul(std::mem::size_of::<Cost<K>>())
                .and_then(|bytes| bytes.checked_mul(2))
                .and_then(|bytes| {
                    column_width
                        .checked_mul(std::mem::size_of::<usize>())
                        .and_then(|active_bytes| active_bytes.checked_mul(2))
                        .and_then(|active_bytes| bytes.checked_add(active_bytes))
                }) {
                Some(bytes) => bytes,
                None => {
                    return self.terminate(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    });
                }
            };
            let prospective_scratch_bytes =
                match requested_position_bytes.checked_add(fixed_scratch_bytes) {
                    Some(bytes) => bytes,
                    None => {
                        return self.terminate(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::ScratchBytes,
                        });
                    }
                };
            if let Err(reason) = self
                .ledger
                .observe_peak(ResourceKind::ScratchBytes, prospective_scratch_bytes)
            {
                return self.terminate(reason);
            }
            let mut positions = Vec::new();
            if positions
                .try_reserve_exact(canonical_position_count)
                .is_err()
            {
                return self.terminate(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: prospective_scratch_bytes,
                });
            }
            for row in next_active.iter().copied() {
                let raw_row = match u32::try_from(row) {
                    Ok(row) => row,
                    Err(_) => {
                        return self.terminate(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::DpCells,
                        });
                    }
                };
                let Some(cost) = next.get(row).copied() else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                let dominated = self
                    .index
                    .kernel
                    .vertical_epsilon_extension(&self.query, interval, row, next, &self.plan)
                    .is_some_and(|vertical| K::Monoid::compare(vertical, cost) == Ordering::Equal);
                if K::Monoid::within(cost, self.tau) && !dominated {
                    positions.push(RangeProductPosition { row: raw_row, cost });
                }
            }
            debug_assert_eq!(positions.len(), canonical_position_count);
            if let Err(reason) = self.ledger.charge(ResourceKind::TrieNodes, 1) {
                return self.terminate(reason);
            }
            let final_bucket = (child.is_final() && relaxed_admits)
                .then(|| child.value_at_final())
                .flatten();
            let edge_count = child.edge_count().unwrap_or(usize::from(u8::MAX) + 1);
            let mut edges = VecDeque::new();
            if edges.try_reserve_exact(edge_count).is_err() {
                let requested =
                    edge_count.saturating_mul(std::mem::size_of::<(u8, DynamicDawgNode<usize>)>());
                return self.terminate(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested,
                });
            }
            child.for_each_edge(|label, grandchild| edges.push_back((label, grandchild)));
            edges
                .make_contiguous()
                .sort_unstable_by_key(|(label, _)| *label);
            let raw_state = match u32::try_from(states.len()) {
                Ok(state) => state,
                Err(_) => {
                    return self.terminate(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::QueueEntries,
                    });
                }
            };
            let state = TemporalStateId(raw_state);
            if states.try_reserve(1).is_err() || stack.try_reserve(1).is_err() {
                let requested = states.len().saturating_add(stack.len()).saturating_add(2);
                return self.terminate(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested,
                });
            }
            states.push(RangeProductState {
                carry: Some(carry),
                positions,
            });
            stack.push(BoundedRangeFrame {
                depth: child_depth,
                state,
                final_bucket,
                next_candidate: 0,
                edges,
            });
            if let Err(reason) = self.observe_state_peaks() {
                return self.terminate(reason);
            }
        }
    }
}

impl<'a, V> ErpAutomatonRangeContinuation<'a, V>
where
    V: Eq + std::hash::Hash + Clone,
{
    /// Exact matches discovered so far. They prove membership individually,
    /// but prove completeness only after [`Self::resume`] returns `Complete`.
    pub fn exact_partial(&self) -> &[(V, f64)] {
        &self.results
    }

    /// Cumulative charges and retained-state peaks for this query.
    pub fn usage(&self) -> ResourceUsage {
        self.ledger.usage()
    }

    /// Current canonical-state, canonical-position, and transition-cache
    /// counts. These counters are observational and never affect ordering.
    pub fn retained_counts(&self) -> (usize, usize, usize) {
        self.machine.retained_counts()
    }

    /// Resume one bounded page of exact dictionary-product traversal.
    pub fn resume(mut self, page: PageBudget) -> OperationOutcome<Vec<(V, f64)>, Self> {
        let step = self.advance(page);
        match step {
            OperationOutcome::Complete { usage, .. } => OperationOutcome::Complete {
                value: std::mem::take(&mut self.results),
                usage,
            },
            OperationOutcome::Incomplete {
                reason,
                continuation,
                usage,
                ..
            } => {
                let partial = Some(ElasticTransducer::<ErpConfig, V>::finish_range_results(
                    self.results.clone(),
                ));
                OperationOutcome::Incomplete {
                    partial,
                    reason,
                    continuation: continuation.map(|()| self),
                    usage,
                }
            }
        }
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

    fn page_allows(
        &self,
        used: usize,
        amount: usize,
        limit: usize,
        resource: ResourceKind,
    ) -> Result<(), IncompleteReason> {
        let requested = used
            .checked_add(amount)
            .ok_or(IncompleteReason::ArithmeticOverflow { resource })?;
        if requested > limit {
            return Err(IncompleteReason::BudgetExceeded {
                resource,
                limit,
                requested,
            });
        }
        Ok(())
    }

    fn observe_state_peaks(&mut self) -> Result<(), IncompleteReason> {
        self.ledger
            .observe_peak(ResourceKind::QueueEntries, self.stack.len())?;
        let machine_bytes = self.machine.retained_scratch_bytes()?;
        self.ledger
            .observe_peak(ResourceKind::ScratchBytes, machine_bytes)?;

        let edge_count = self.stack.iter().try_fold(0usize, |total, frame| {
            total
                .checked_add(frame.edges.len())
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ContinuationBytes,
                })
        })?;
        let edge_bytes = edge_count
            .checked_mul(std::mem::size_of::<(u8, DynamicDawgNode<usize>)>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        let stack_bytes = self
            .stack
            .len()
            .checked_mul(std::mem::size_of::<ErpAutomatonRangeFrame>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        let query_bytes = self
            .query
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        let result_bytes = self
            .results
            .len()
            .checked_mul(std::mem::size_of::<(V, f64)>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        let continuation_bytes = machine_bytes
            .checked_add(edge_bytes)
            .and_then(|bytes| bytes.checked_add(stack_bytes))
            .and_then(|bytes| bytes.checked_add(query_bytes))
            .and_then(|bytes| bytes.checked_add(result_bytes))
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        self.ledger
            .observe_peak(ResourceKind::ContinuationBytes, continuation_bytes)
    }

    fn advance(&mut self, page: PageBudget) -> OperationOutcome<(), ()> {
        if let Some(reason) = self.terminal {
            return self.terminate(reason);
        }

        let mut page_work = 0usize;
        let mut page_results = 0usize;
        loop {
            if let Some(pending) = self.pending_match.take() {
                if let Err(reason) =
                    self.page_allows(page_results, 1, page.max_results, ResourceKind::Results)
                {
                    self.pending_match = Some(pending);
                    return match reason {
                        IncompleteReason::BudgetExceeded {
                            resource,
                            limit,
                            requested,
                        } => self.paused(resource, limit, requested),
                        other => self.terminate(other),
                    };
                }
                if let Err(reason) = self.ledger.charge(ResourceKind::Results, 1) {
                    self.pending_match = Some(pending);
                    return self.terminate(reason);
                }
                self.results.push(pending);
                page_results += 1;
                if let Err(reason) = self.observe_state_peaks() {
                    return self.terminate(reason);
                }
                continue;
            }

            if self.done {
                self.results = ElasticTransducer::<ErpConfig, V>::finish_range_results(
                    std::mem::take(&mut self.results),
                );
                return OperationOutcome::Complete {
                    value: (),
                    usage: self.ledger.usage(),
                };
            }

            let next_candidate = self.stack.last_mut().and_then(|frame| {
                frame.final_bucket.and_then(|bucket_id| {
                    self.index
                        .buckets
                        .get(bucket_id)
                        .and_then(|ids| ids.get(frame.next_candidate))
                        .cloned()
                })
            });
            if let Some(id) = next_candidate {
                let Some(stored) = self.index.originals.get(&id) else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                if stored.series.len() > self.ledger.limits().max_series_len {
                    return self.terminate(IncompleteReason::BudgetExceeded {
                        resource: ResourceKind::SeriesLength,
                        limit: self.ledger.limits().max_series_len,
                        requested: stored.series.len(),
                    });
                }
                if stored.series.iter().any(|sample| !sample.is_finite()) {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                }
                let candidate_work = match self.query.len().checked_add(1).and_then(|left| {
                    stored
                        .series
                        .len()
                        .checked_add(1)
                        .and_then(|right| left.checked_mul(right))
                }) {
                    Some(work) => work,
                    None => {
                        return self.terminate(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::DpCells,
                        });
                    }
                };
                if let Err(reason) = self.page_allows(
                    page_work,
                    candidate_work,
                    page.max_work_units,
                    ResourceKind::WorkUnits,
                ) {
                    return match reason {
                        IncompleteReason::BudgetExceeded {
                            resource,
                            limit,
                            requested,
                        } => self.paused(resource, limit, requested),
                        other => self.terminate(other),
                    };
                }
                if let Err(reason) = self.ledger.charge_many(&[
                    (ResourceKind::Candidates, 1),
                    (ResourceKind::DpCells, candidate_work),
                    (ResourceKind::WorkUnits, candidate_work),
                ]) {
                    return self.terminate(reason);
                }
                page_work += candidate_work;
                let Some(frame) = self.stack.last_mut() else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                frame.next_candidate = match frame.next_candidate.checked_add(1) {
                    Some(next) => next,
                    None => {
                        return self.terminate(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::Candidates,
                        });
                    }
                };

                if self
                    .index
                    .kernel
                    .candidate_lower_bound(&self.query, &stored.series)
                    > self.tau
                {
                    continue;
                }
                let Some(exact) =
                    self.index
                        .kernel
                        .distance_with_cutoff(&self.query, &stored.series, self.tau)
                else {
                    continue;
                };
                if !exact.is_finite() {
                    return self.terminate(IncompleteReason::NumericOverflow);
                }
                if exact > self.tau {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                }
                if page_results >= page.max_results {
                    self.pending_match = Some((id, exact));
                    return self.paused(
                        ResourceKind::Results,
                        page.max_results,
                        page_results.saturating_add(1),
                    );
                }
                if let Err(reason) = self.ledger.charge(ResourceKind::Results, 1) {
                    return self.terminate(reason);
                }
                self.results.push((id, exact));
                page_results += 1;
                if let Err(reason) = self.observe_state_peaks() {
                    return self.terminate(reason);
                }
                continue;
            }

            if let Some(frame) = self.stack.last_mut() {
                frame.final_bucket = None;
            }

            let Some(frame) = self.stack.last() else {
                self.done = true;
                continue;
            };
            let Some((unit, child)) = frame.edges.front().cloned() else {
                self.stack.pop();
                if let Err(reason) = self.observe_state_peaks() {
                    return self.terminate(reason);
                }
                continue;
            };
            let child_depth = match frame.depth.checked_add(1) {
                Some(depth) => depth,
                None => {
                    return self.terminate(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::SeriesLength,
                    });
                }
            };
            if child_depth > self.ledger.limits().max_series_len {
                return self.terminate(IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::SeriesLength,
                    limit: self.ledger.limits().max_series_len,
                    requested: child_depth,
                });
            }
            let edge_work = match self.machine.transition_work_bound(frame.state) {
                Ok(work) => work,
                Err(reason) => return self.terminate(reason),
            };
            if let Err(reason) = self.page_allows(
                page_work,
                edge_work,
                page.max_work_units,
                ResourceKind::WorkUnits,
            ) {
                return match reason {
                    IncompleteReason::BudgetExceeded {
                        resource,
                        limit,
                        requested,
                    } => self.paused(resource, limit, requested),
                    other => self.terminate(other),
                };
            }
            if let Err(reason) = self.ledger.charge_many(&[
                (ResourceKind::TrieEdges, 1),
                (ResourceKind::DpCells, edge_work),
                (ResourceKind::WorkUnits, edge_work),
            ]) {
                return self.terminate(reason);
            }
            page_work += edge_work;
            let source = frame.state;
            let Some(frame) = self.stack.last_mut() else {
                return self.terminate(IncompleteReason::InvalidStoredData);
            };
            if frame.edges.front().map(|(label, _)| *label) != Some(unit) {
                return self.terminate(IncompleteReason::InvalidStoredData);
            }
            frame.edges.pop_front();

            let transition =
                match self
                    .machine
                    .transition(source, unit, self.index.bin_bounds_for(unit))
                {
                    Ok(transition) => transition,
                    Err(reason) => return self.terminate(reason),
                };
            debug_assert!(transition.work_units <= edge_work);
            let Some(target) = transition.target else {
                continue;
            };
            if self.machine.lower_bound(target) > self.tau {
                return self.terminate(IncompleteReason::InvalidStoredData);
            }
            if let Err(reason) = self.ledger.charge(ResourceKind::TrieNodes, 1) {
                return self.terminate(reason);
            }
            let final_cost = match self.machine.final_cost(target) {
                Ok(cost) => cost,
                Err(reason) => return self.terminate(reason),
            };
            let final_bucket = if child.is_final() && final_cost.is_some() {
                child.value_at_final()
            } else {
                None
            };
            let mut edges = Vec::new();
            child.for_each_edge(|label, grandchild| edges.push((label, grandchild)));
            edges.sort_by_key(|(label, _)| *label);
            self.stack.push(ErpAutomatonRangeFrame {
                depth: child_depth,
                state: target,
                final_bucket,
                next_candidate: 0,
                edges: edges.into(),
            });
            if let Err(reason) = self.observe_state_peaks() {
                return self.terminate(reason);
            }
        }
    }
}

impl<K, V> ElasticTransducer<K, V>
where
    K: ElasticKernel,
    V: Eq + std::hash::Hash + Clone,
{
    /// Create an empty transducer.
    ///
    /// Uses a byte-quantized trie, matching [`crate::time_series::TimeSeriesIndex`] and
    /// [`crate::time_series::HybridSearchIndex`]. Quantizers wider than 256 bins are
    /// coarsened to a byte-compatible 256-bin config over the same value range.
    pub fn new<C>(quant: QuantizationConfig, kernel: C) -> Self
    where
        C: Into<K>,
    {
        let quant = quant.into_u8_compatible();
        let kernel = kernel.into().normalized();
        let bin_bounds = (0..quant.num_bins)
            .map(|bin| quant.bin_bounds(bin))
            .collect();
        Self {
            dawg: DynamicDawg::new(),
            quant,
            kernel,
            bin_bounds,
            buckets: Vec::new(),
            originals: HashMap::new(),
        }
    }

    /// Insert a reference series under identifier `value`.
    ///
    /// Returns `true` if `value` was not previously present.
    pub fn insert(&mut self, value: V, series: &[f64]) -> bool {
        let key = self.quant.encode_u8(series);
        let bucket_id = match self.dawg.get_bytes_value(&key) {
            Some(existing) => existing,
            None => {
                let bucket_id = self.buckets.len();
                self.buckets.push(Vec::with_capacity(1));
                let inserted = self.dawg.insert_bytes_with_value(&key, bucket_id);
                debug_assert!(
                    inserted,
                    "new elastic bucket key should insert exactly once"
                );
                bucket_id
            }
        };
        let is_new = !self.originals.contains_key(&value);

        let bucket_location = match self
            .originals
            .get(&value)
            .map(|stored| stored.bucket_location)
        {
            Some((old_bucket_id, old_slot)) if old_bucket_id == bucket_id => (bucket_id, old_slot),
            Some((old_bucket_id, old_slot)) => {
                if self.remove_from_bucket(&value, old_bucket_id, old_slot) {
                    self.release_empty_bucket_storage(old_bucket_id);
                }
                self.push_to_bucket(value.clone(), bucket_id)
            }
            None => self.push_to_bucket(value.clone(), bucket_id),
        };

        self.originals.insert(
            value,
            StoredSeries {
                series: series.to_vec(),
                bucket_location,
            },
        );
        is_new
    }

    fn push_to_bucket(&mut self, value: V, bucket_id: usize) -> BucketLocation {
        let slot = self.buckets[bucket_id].len();
        self.buckets[bucket_id].push(value);
        (bucket_id, slot)
    }

    fn remove_from_bucket(&mut self, value: &V, bucket_id: usize, slot: usize) -> bool {
        let (moved, became_empty) = {
            let bucket = &mut self.buckets[bucket_id];
            debug_assert!(bucket.get(slot).is_some_and(|stored| stored == value));
            let removed = bucket.swap_remove(slot);
            debug_assert!(&removed == value);
            (bucket.get(slot).cloned(), bucket.is_empty())
        };
        if let Some(moved) = moved {
            if let Some(stored) = self.originals.get_mut(&moved) {
                stored.bucket_location = (bucket_id, slot);
            }
        }
        became_empty
    }

    fn release_empty_bucket_storage(&mut self, bucket_id: usize) {
        debug_assert!(self.buckets[bucket_id].is_empty());
        self.buckets[bucket_id] = Vec::new();
    }

    /// Number of indexed references.
    #[inline]
    pub fn len(&self) -> usize {
        self.originals.len()
    }

    /// Whether the index holds no references.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.originals.is_empty()
    }

    /// The quantization configuration.
    #[inline]
    pub fn quant_config(&self) -> &QuantizationConfig {
        &self.quant
    }

    /// Elastic kernel and its normalized configuration.
    #[inline]
    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    /// Retrieve the original series for a reference id, if present.
    #[inline]
    pub fn get_original(&self, value: &V) -> Option<&[f64]> {
        self.originals
            .get(value)
            .map(|stored| stored.series.as_slice())
    }

    /// Remove a reference id from the index.
    ///
    /// This drops the stored full-precision series and removes the id from its
    /// quantized candidate bucket. Empty buckets remain addressable from the
    /// DAWG because this index uses arbitrary byte keys and `DynamicDawg` only
    /// exposes string-key deletion, but their backing storage is released and
    /// they produce no candidates during exact search.
    ///
    /// # Returns
    ///
    /// `true` if the reference id was present and removed, `false` otherwise.
    pub fn remove(&mut self, value: V) -> bool {
        let Some(stored) = self.originals.remove(&value) else {
            return false;
        };

        let (bucket_id, slot) = stored.bucket_location;
        if self.remove_from_bucket(&value, bucket_id, slot) {
            self.release_empty_bucket_storage(bucket_id);
        }
        true
    }

    /// Exact range search, sorted by ascending kernel cost.
    ///
    /// One prefix-amortized trie traversal with admissible interval pruning,
    /// followed by exact full-precision verification at survivors. Returns no
    /// false negatives and no false positives.
    pub fn search_range(&self, query: &[f64], tau: Cost<K>) -> Vec<(V, Cost<K>)> {
        let m = query.len();
        if self.is_empty() {
            return Vec::new();
        }
        let plan = self.kernel.plan(query);
        if m == 0 || !self.kernel.supports_interval_query(query) {
            return self.scan_range(query, &plan, tau);
        }
        let root = self.dawg.root();
        let Some(column_width) = self.kernel.column_len(m) else {
            return Vec::new();
        };
        let mut out: Vec<(V, Cost<K>)> =
            Vec::with_capacity(self.len().min(DEFAULT_RESULT_BUFFER_CAPACITY));
        let mut ctx = RangeWalkContext {
            query,
            plan: &plan,
            tau,
            out: &mut out,
            column_width,
        };
        self.walk_range_iterative(root, 0, None, vec![K::Monoid::TOP; column_width], &mut ctx);

        Self::finish_range_results(out)
    }

    /// Start an exact, bounded, resumable range query.
    ///
    /// A complete empty vector proves that no indexed series is within the
    /// cutoff. An incomplete empty vector is only an exact subset and carries
    /// a continuation when another page can make progress.
    pub fn search_range_bounded(
        &self,
        query: &[f64],
        tau: Cost<K>,
        limits: ResourceLimits,
        page: PageBudget,
    ) -> Result<BoundedRangeOutcome<'_, K, V>, TemporalValidationError> {
        if !self.kernel.cutoff_is_valid(tau) {
            return Err(TemporalValidationError::InvalidCutoff);
        }
        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, query)?;
        let plan = self.kernel.plan(query);

        let (mode, terminal) = if self.is_empty() {
            (RangeSessionMode::Done, None)
        } else if query.is_empty() || !self.kernel.supports_interval_query(query) {
            (RangeSessionMode::Scan { bucket: 0, slot: 0 }, None)
        } else if let Some(column_width) = self.kernel.column_len(query.len()) {
            let root = self.dawg.root();
            let final_bucket = root.value_at_final().filter(|_| root.is_final());
            let mut edges = VecDeque::new();
            let edge_count = root.edge_count().unwrap_or(usize::from(u8::MAX) + 1);
            let mut terminal = None;
            if edges.try_reserve_exact(edge_count).is_err() {
                terminal = Some(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested: edge_count
                        .saturating_mul(std::mem::size_of::<(u8, DynamicDawgNode<usize>)>()),
                });
            }
            if terminal.is_none() {
                root.for_each_edge(|unit, child| edges.push_back((unit, child)));
                edges
                    .make_contiguous()
                    .sort_unstable_by_key(|(label, _)| *label);
            }
            let frame = BoundedRangeFrame {
                depth: 0,
                state: TemporalStateId(0),
                final_bucket,
                next_candidate: 0,
                edges,
            };
            if terminal.is_none() {
                terminal = ledger.charge(ResourceKind::TrieNodes, 1).err();
            }
            if column_width > u32::MAX as usize {
                terminal = Some(IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::DpCells,
                    limit: u32::MAX as usize,
                    requested: column_width,
                });
            }
            let scratch_requested = column_width
                .checked_mul(std::mem::size_of::<Cost<K>>())
                .and_then(|bytes| bytes.checked_mul(2))
                .and_then(|bytes| {
                    column_width
                        .checked_mul(std::mem::size_of::<usize>())
                        .and_then(|active| active.checked_mul(2))
                        .and_then(|active| bytes.checked_add(active))
                });
            let scratch_requested = match scratch_requested {
                Some(bytes) => bytes,
                None => {
                    terminal = Some(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    });
                    0
                }
            };
            if scratch_requested > limits.max_scratch_bytes {
                terminal = Some(IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested: scratch_requested,
                });
            }

            let mut previous = Vec::new();
            let mut next = Vec::new();
            let mut previous_active = Vec::new();
            let mut next_active = Vec::new();
            if terminal.is_none()
                && (previous.try_reserve_exact(column_width).is_err()
                    || next.try_reserve_exact(column_width).is_err()
                    || previous_active.try_reserve_exact(column_width).is_err()
                    || next_active.try_reserve_exact(column_width).is_err())
            {
                terminal = Some(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: scratch_requested,
                });
            }
            if terminal.is_none() {
                previous.resize(column_width, K::Monoid::TOP);
                next.resize(column_width, K::Monoid::TOP);
            }
            let mut stack = Vec::new();
            let mut states = Vec::new();
            if stack.try_reserve_exact(1).is_err() || states.try_reserve_exact(1).is_err() {
                terminal = Some(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested: std::mem::size_of::<BoundedRangeFrame<DynamicDawgNode<usize>>>()
                        .saturating_add(std::mem::size_of::<RangeProductState<K>>()),
                });
            }
            if terminal.is_none() {
                stack.push(frame);
                states.push(RangeProductState {
                    carry: None,
                    positions: Vec::new(),
                });
            }
            (
                RangeSessionMode::Trie {
                    stack,
                    states,
                    column_width,
                    previous,
                    next,
                    previous_active,
                    next_active,
                },
                terminal,
            )
        } else {
            (
                RangeSessionMode::Done,
                Some(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::DpCells,
                }),
            )
        };

        let mut continuation = RangeContinuation {
            index: self,
            query: query.into(),
            plan,
            tau,
            mode,
            results: Vec::with_capacity(
                self.len()
                    .min(DEFAULT_RESULT_BUFFER_CAPACITY)
                    .min(limits.max_results),
            ),
            pending_match: None,
            ledger,
            terminal,
        };
        if continuation.terminal.is_none() {
            if let Err(reason) = continuation.observe_state_peaks() {
                continuation.terminal = Some(reason);
            }
        }
        Ok(continuation.resume(page))
    }

    /// Exact k-nearest-neighbour search with hard, fail-closed limits.
    ///
    /// This strict adapter intentionally uses a deterministic full-precision
    /// scan. The legacy best-first convenience API retains a dense DP column
    /// in each queue entry and is therefore unsuitable as release evidence
    /// under adversarial branching. This bounded path retains only `O(k)`
    /// results plus the exact scorer's two-row scratch, charges every candidate
    /// and worst-case DP cell before evaluation, and never maps exhaustion to a
    /// complete empty vector.
    pub fn search_knn_bounded(
        &self,
        query: &[f64],
        k: usize,
        limits: ResourceLimits,
    ) -> Result<ExactSearchOutcome<K, V>, TemporalValidationError> {
        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, query)?;
        if k == 0 || self.is_empty() {
            return Ok(OperationOutcome::Complete {
                value: Vec::new(),
                usage: ledger.usage(),
            });
        }

        let capacity = k.min(self.len());
        if let Err(reason) = ledger.charge(ResourceKind::Results, capacity) {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason,
                continuation: None,
                usage: ledger.usage(),
            });
        }
        let mut best = BinaryHeap::new();
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

        let plan = self.kernel.plan(query);
        let mut cutoff = K::Monoid::TOP;
        let mut sequence = 0_usize;
        for id in self.ids_in_bucket_order() {
            let Some(stored) = self.originals.get(id) else {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason: IncompleteReason::InvalidStoredData,
                    continuation: None,
                    usage: ledger.usage(),
                });
            };
            if stored.series.len() > limits.max_series_len
                || stored.series.iter().any(|sample| !sample.is_finite())
            {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason: IncompleteReason::InvalidStoredData,
                    continuation: None,
                    usage: ledger.usage(),
                });
            }
            let Some(dp_cells) = query.len().checked_add(1).and_then(|rows| {
                stored
                    .series
                    .len()
                    .checked_add(1)
                    .and_then(|cols| rows.checked_mul(cols))
            }) else {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason: IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::DpCells,
                    },
                    continuation: None,
                    usage: ledger.usage(),
                });
            };
            if let Err(reason) = ledger.charge_many(&[
                (ResourceKind::Candidates, 1),
                (ResourceKind::DpCells, dp_cells),
                (ResourceKind::WorkUnits, dp_cells),
            ]) {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason,
                    continuation: None,
                    usage: ledger.usage(),
                });
            }
            let Some(scratch_bytes) = query
                .len()
                .max(stored.series.len())
                .checked_add(1)
                .and_then(|width| width.checked_mul(2))
                .and_then(|cells| cells.checked_mul(std::mem::size_of::<Cost<K>>()))
            else {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason: IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    },
                    continuation: None,
                    usage: ledger.usage(),
                });
            };
            if let Err(reason) = ledger.observe_peak(ResourceKind::ScratchBytes, scratch_bytes) {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason,
                    continuation: None,
                    usage: ledger.usage(),
                });
            }

            let lower_bound = self
                .kernel
                .candidate_lower_bound(query, &stored.series, &plan);
            if !K::Monoid::within(lower_bound, cutoff) {
                continue;
            }
            let Some(exact) = self.kernel.exact_with_cutoff(query, &stored.series, cutoff) else {
                continue;
            };
            let Some(candidate_sequence) = take_sequence(&mut sequence) else {
                return Ok(OperationOutcome::Incomplete {
                    partial: None,
                    reason: IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::Results,
                    },
                    continuation: None,
                    usage: ledger.usage(),
                });
            };
            if Self::push_knn_result(&mut best, id.clone(), exact, k, candidate_sequence) {
                cutoff = Self::knn_cutoff(&best, k);
            }
        }
        Ok(OperationOutcome::Complete {
            value: Self::finish_knn_results(best),
            usage: ledger.usage(),
        })
    }

    /// Deterministic exact fallback for queries outside interval traversal's
    /// domain, including the kernel-specific empty-series boundary.
    fn scan_range(&self, query: &[f64], plan: &K::QueryPlan, tau: Cost<K>) -> Vec<(V, Cost<K>)> {
        let mut out = Vec::with_capacity(self.len().min(DEFAULT_RESULT_BUFFER_CAPACITY));
        for id in self.ids_in_bucket_order() {
            let Some(stored) = self.originals.get(id) else {
                continue;
            };
            let lower_bound = self
                .kernel
                .candidate_lower_bound(query, &stored.series, plan);
            if !K::Monoid::within(lower_bound, tau) {
                continue;
            }
            if let Some(exact) = self
                .kernel
                .exact_with_cutoff(query, &stored.series, tau)
                .filter(|cost| K::Monoid::within(*cost, tau))
            {
                out.push((id.clone(), exact));
            }
        }
        Self::finish_range_results(out)
    }

    /// Iterate every indexed reference id in deterministic bucket/slot order.
    ///
    /// `buckets` is a `Vec<Vec<V>>` whose slot order is fixed by insertion, so
    /// this yields a reproducible sequence across process runs — unlike
    /// iterating the `originals` `HashMap`, whose default `RandomState` seeds a
    /// different iteration order per process. Every inserted id lives in exactly
    /// one bucket slot (see `push_to_bucket`/`remove_from_bucket`), so this
    /// visits each id exactly once, matching `originals.keys()` as a set.
    fn ids_in_bucket_order(&self) -> impl Iterator<Item = &V> {
        self.buckets.iter().flatten()
    }

    #[inline]
    fn bin_bounds_for(&self, unit: u8) -> (f64, f64) {
        self.bin_bounds
            .get(byte_unit_index(unit))
            .copied()
            .unwrap_or_else(|| self.quant.bin_bounds(byte_unit_bin(unit)))
    }

    /// Iterative interval-pruned trie walk used by [`Self::search_range`].
    ///
    /// `node` sits at depth `depth` (target elements consumed); `col` is its
    /// interval DP column; `last_interval` is the bin interval of the element
    /// consumed to reach `node` (`None` at the root). Generic over the node type
    /// so the same logic applies to any byte-unit dictionary backend.
    fn walk_range_iterative<N>(
        &self,
        root: N,
        depth: usize,
        carry: Option<K::Carry>,
        column: Vec<Cost<K>>,
        ctx: &mut RangeWalkContext<'_, K, V>,
    ) where
        N: DictionaryNode<Unit = u8> + MappedDictionaryNode<Value = usize>,
    {
        let final_row = self.kernel.final_row(ctx.query.len());

        let make_frame = |node: N, depth: usize, carry: Option<K::Carry>, column: Vec<Cost<K>>| {
            // A root final represents an empty reference and has no real DP
            // column. Every non-root final is admitted only by its terminal row.
            let relaxed_admits = depth == 0
                || column
                    .get(final_row)
                    .is_some_and(|cost| K::Monoid::within(*cost, ctx.tau));
            let final_bucket = (node.is_final() && relaxed_admits)
                .then(|| node.value_at_final())
                .flatten();
            let mut edges = VecDeque::new();
            node.for_each_edge(|unit, child| edges.push_back((unit, child)));
            RangeFrame::<K, N> {
                depth,
                carry,
                column,
                final_bucket,
                next_candidate: 0,
                edges,
            }
        };

        let mut stack = vec![make_frame(root, depth, carry, column)];
        while !stack.is_empty() {
            let next_id = {
                let frame = stack.last_mut().expect("nonempty checked above");
                let next = frame
                    .final_bucket
                    .and_then(|bucket_id| self.buckets.get(bucket_id))
                    .and_then(|ids| ids.get(frame.next_candidate))
                    .cloned();
                if next.is_some() {
                    frame.next_candidate = frame.next_candidate.saturating_add(1);
                } else {
                    frame.final_bucket = None;
                }
                next
            };
            if let Some(id) = next_id {
                if let Some(stored) = self.originals.get(&id) {
                    let candidate_bound =
                        self.kernel
                            .candidate_lower_bound(ctx.query, &stored.series, ctx.plan);
                    if K::Monoid::within(candidate_bound, ctx.tau) {
                        if let Some(exact) = self
                            .kernel
                            .exact_with_cutoff(ctx.query, &stored.series, ctx.tau)
                            .filter(|cost| K::Monoid::within(*cost, ctx.tau))
                        {
                            ctx.out.push((id, exact));
                        }
                    }
                }
                continue;
            }

            let child_frame = {
                let frame = stack.last_mut().expect("nonempty checked above");
                let Some((unit, child)) = frame.edges.pop_front() else {
                    stack.pop();
                    continue;
                };
                let Some(child_depth) = frame.depth.checked_add(1) else {
                    continue;
                };
                let interval = self.bin_bounds_for(unit);
                let prefix_lower_bound = self.kernel.prefix_lower_bound(
                    ctx.query,
                    interval,
                    frame.carry,
                    child_depth,
                    ctx.plan,
                );
                if !K::Monoid::within(prefix_lower_bound, ctx.tau) {
                    continue;
                }
                let mut child_column = Vec::with_capacity(ctx.column_width);
                let (child_lower_bound, child_carry) = self.kernel.step_column(
                    &frame.column,
                    ctx.query,
                    interval,
                    frame.carry,
                    child_depth,
                    ctx.plan,
                    &mut child_column,
                );
                K::Monoid::within(child_lower_bound, ctx.tau).then_some((
                    child,
                    child_depth,
                    Some(child_carry),
                    child_column,
                ))
            };
            if let Some((child, child_depth, child_carry, child_column)) = child_frame {
                stack.push(make_frame(child, child_depth, child_carry, child_column));
            }
        }
    }

    /// Exact k-nearest-neighbor search by kernel distance.
    ///
    /// Uses a single best-first trie traversal keyed by the admissible
    /// interval-column lower bound. Once `k` exact candidates have been found,
    /// every queued subtree whose lower bound exceeds the current kth exact
    /// distance is safely pruned. `initial_threshold` is retained for API
    /// compatibility; exactness and result ordering do not depend on it.
    pub fn search_knn(
        &self,
        query: &[f64],
        k: usize,
        initial_threshold: Cost<K>,
    ) -> Vec<(V, Cost<K>)> {
        self.search_knn_with_stats(query, k, initial_threshold).0
    }

    /// Exact k-nearest-neighbour search with observational pruning counters.
    ///
    /// Results are byte-for-byte identical to [`Self::search_knn`]. The
    /// accompanying counters partition visited edges and final candidates, so
    /// callers can compare pruning policies without inferring work from wall
    /// time. Counter overflow stops incrementing the affected field at
    /// `usize::MAX`; it never changes a search decision or result.
    pub fn search_knn_with_stats(
        &self,
        query: &[f64],
        k: usize,
        initial_threshold: Cost<K>,
    ) -> (Vec<(V, Cost<K>)>, ElasticSearchStats) {
        let _ = initial_threshold;
        let mut stats = ElasticSearchStats::default();
        if k == 0 || self.is_empty() {
            return (Vec::new(), stats);
        }
        let query_plan = self.kernel.plan(query);
        if query.is_empty() || !self.kernel.supports_interval_query(query) {
            return self.scan_knn_with_stats(query, &query_plan, k);
        }

        let final_row = self.kernel.final_row(query.len());
        let Some(column_width) = self.kernel.column_len(query.len()) else {
            return (Vec::new(), stats);
        };
        let mut best: BinaryHeap<KnnBestResult<K, V>> =
            BinaryHeap::with_capacity(k.min(self.len()));
        let mut kth_distance = K::Monoid::TOP;
        let mut sequence = 0usize;
        let mut result_sequence = 0usize;
        let mut queue = BinaryHeap::with_capacity(1);
        queue.push(KnnQueueNode::<K, _> {
            lower_bound: K::Monoid::ZERO,
            sequence,
            depth: 0,
            node: self.dawg.root(),
            column: vec![K::Monoid::TOP; column_width],
            carry: None,
        });

        while let Some(current) = queue.pop() {
            if best.len() >= k && !K::Monoid::within(current.lower_bound, kth_distance) {
                stats.queued_subtrees_pruned = stats
                    .queued_subtrees_pruned
                    .saturating_add(queue.len().saturating_add(1));
                break;
            }
            stats.visited_nodes = stats.visited_nodes.saturating_add(1);

            if current.node.is_final()
                && (current.depth == 0
                    || current
                        .column
                        .get(final_row)
                        .is_some_and(|cost| K::Monoid::within(*cost, kth_distance)))
            {
                if let Some(bucket_id) = current.node.value_at_final() {
                    let ids = &self.buckets[bucket_id];
                    for id in ids {
                        if let Some(stored) = self.originals.get(id) {
                            stats.candidates_considered =
                                stats.candidates_considered.saturating_add(1);
                            let candidate_bound = self.kernel.candidate_lower_bound(
                                query,
                                &stored.series,
                                &query_plan,
                            );
                            if !K::Monoid::within(candidate_bound, kth_distance) {
                                stats.candidate_bound_pruned =
                                    stats.candidate_bound_pruned.saturating_add(1);
                                continue;
                            }
                            stats.exact_evaluations = stats.exact_evaluations.saturating_add(1);
                            let Some(exact) =
                                self.kernel
                                    .exact_with_cutoff(query, &stored.series, kth_distance)
                            else {
                                stats.cutoff_abandoned = stats.cutoff_abandoned.saturating_add(1);
                                continue;
                            };
                            let Some(candidate_sequence) = take_sequence(&mut result_sequence)
                            else {
                                continue;
                            };
                            if Self::push_knn_result(
                                &mut best,
                                id.clone(),
                                exact,
                                k,
                                candidate_sequence,
                            ) {
                                kth_distance = Self::knn_cutoff(&best, k);
                            }
                        }
                    }
                }
            }

            current.node.for_each_edge(|unit, child| {
                stats.visited_edges = stats.visited_edges.saturating_add(1);
                let interval = self.bin_bounds_for(unit);
                let Some(child_depth) = current.depth.checked_add(1) else {
                    return;
                };
                let prefix_lower_bound = self.kernel.prefix_lower_bound(
                    query,
                    interval,
                    current.carry,
                    child_depth,
                    &query_plan,
                );
                if best.len() >= k && !K::Monoid::within(prefix_lower_bound, kth_distance) {
                    stats.prefix_pruned = stats.prefix_pruned.saturating_add(1);
                    return;
                }
                let mut child_column = Vec::with_capacity(column_width);
                stats.columns_built = stats.columns_built.saturating_add(1);
                let (lower_bound, carry) = self.kernel.step_column(
                    &current.column,
                    query,
                    interval,
                    current.carry,
                    child_depth,
                    &query_plan,
                    &mut child_column,
                );
                if best.len() < k || K::Monoid::within(lower_bound, kth_distance) {
                    let Some(child_sequence) = next_sequence(&mut sequence) else {
                        return;
                    };
                    queue.push(KnnQueueNode {
                        lower_bound,
                        sequence: child_sequence,
                        depth: child_depth,
                        node: child,
                        column: child_column,
                        carry: Some(carry),
                    });
                } else {
                    stats.column_pruned = stats.column_pruned.saturating_add(1);
                }
            });
        }

        debug_assert!(stats.accounting_is_consistent());
        (Self::finish_knn_results(best), stats)
    }

    fn scan_knn_with_stats(
        &self,
        query: &[f64],
        plan: &K::QueryPlan,
        k: usize,
    ) -> (Vec<(V, Cost<K>)>, ElasticSearchStats) {
        let mut best: BinaryHeap<KnnBestResult<K, V>> =
            BinaryHeap::with_capacity(k.min(self.len()));
        let mut cutoff = K::Monoid::TOP;
        let mut sequence = 0usize;
        let mut stats = ElasticSearchStats::default();

        for id in self.ids_in_bucket_order() {
            let Some(stored) = self.originals.get(id) else {
                continue;
            };
            stats.candidates_considered = stats.candidates_considered.saturating_add(1);
            let lower_bound = self
                .kernel
                .candidate_lower_bound(query, &stored.series, plan);
            if !K::Monoid::within(lower_bound, cutoff) {
                stats.candidate_bound_pruned = stats.candidate_bound_pruned.saturating_add(1);
                continue;
            }
            stats.exact_evaluations = stats.exact_evaluations.saturating_add(1);
            let Some(exact) = self.kernel.exact_with_cutoff(query, &stored.series, cutoff) else {
                stats.cutoff_abandoned = stats.cutoff_abandoned.saturating_add(1);
                continue;
            };
            let Some(candidate_sequence) = take_sequence(&mut sequence) else {
                break;
            };
            if Self::push_knn_result(&mut best, id.clone(), exact, k, candidate_sequence) {
                cutoff = Self::knn_cutoff(&best, k);
            }
        }

        debug_assert!(stats.accounting_is_consistent());
        (Self::finish_knn_results(best), stats)
    }

    fn push_knn_result(
        best: &mut BinaryHeap<KnnBestResult<K, V>>,
        id: V,
        distance: Cost<K>,
        k: usize,
        sequence: usize,
    ) -> bool {
        if k == 0 || K::Monoid::compare(distance, K::Monoid::TOP) != Ordering::Less {
            return false;
        }

        if best.len() == k {
            let Some(worst) = best.peek() else {
                return false;
            };
            if K::Monoid::compare(distance, worst.distance) != Ordering::Less {
                return false;
            }
            best.pop();
        }

        best.push(KnnBestResult {
            distance,
            sequence,
            value: id,
        });
        true
    }

    fn knn_cutoff(best: &BinaryHeap<KnnBestResult<K, V>>, k: usize) -> Cost<K> {
        if best.len() >= k {
            best.peek().map_or(K::Monoid::TOP, |entry| entry.distance)
        } else {
            K::Monoid::TOP
        }
    }

    fn finish_range_results(results: Vec<(V, Cost<K>)>) -> Vec<(V, Cost<K>)> {
        let mut best_by_id: HashMap<V, (Cost<K>, usize)> = HashMap::with_capacity(results.len());
        for (sequence, (value, distance)) in results.into_iter().enumerate() {
            match best_by_id.entry(value) {
                Entry::Vacant(entry) => {
                    entry.insert((distance, sequence));
                }
                Entry::Occupied(mut entry) => {
                    let (best_distance, best_sequence) = entry.get_mut();
                    if K::Monoid::compare(distance, *best_distance) == Ordering::Less {
                        *best_distance = distance;
                        *best_sequence = sequence;
                    }
                }
            }
        }

        let mut results: Vec<(V, Cost<K>, usize)> = best_by_id
            .into_iter()
            .map(|(value, (distance, sequence))| (value, distance, sequence))
            .collect();
        results.sort_by(|a, b| K::Monoid::compare(a.1, b.1).then_with(|| a.2.cmp(&b.2)));
        results
            .into_iter()
            .map(|(value, distance, _)| (value, distance))
            .collect()
    }

    fn finish_knn_results(best: BinaryHeap<KnnBestResult<K, V>>) -> Vec<(V, Cost<K>)> {
        let mut best = best.into_vec();
        best.sort_by(|a, b| {
            K::Monoid::compare(a.distance, b.distance).then_with(|| a.sequence.cmp(&b.sequence))
        });
        best.into_iter()
            .map(|entry| (entry.value, entry.distance))
            .collect()
    }
}

impl<K> ElasticTransducer<K, usize>
where
    K: ElasticKernel,
{
    /// Build a transducer from a slice of reference series, assigning each the
    /// id equal to its index. Convenience mirror of
    /// [`crate::time_series::TimeSeriesIndex::from_series`].
    pub fn from_series<C>(quant: QuantizationConfig, kernel: C, series: &[Vec<f64>]) -> Self
    where
        C: Into<K>,
    {
        let mut idx = Self::new(quant, kernel);
        idx.originals.reserve(series.len());
        idx.buckets.reserve(series.len());
        for (i, s) in series.iter().enumerate() {
            idx.insert(i, s);
        }
        idx
    }
}

impl<V> ElasticTransducer<ErpConfig, V>
where
    V: Eq + std::hash::Hash + Clone,
{
    /// Start a bounded exact ERP range query using the lazy automaton product.
    ///
    /// The dictionary traversal constructs only reachable canonical ERP
    /// antichains. DFS frames retain compact state IDs rather than one dense DP
    /// column per trie depth. Every quantization-collision bucket member is
    /// still scored against its stored full-precision series before emission.
    pub fn search_range_automaton_bounded(
        &self,
        query: &[f64],
        tau: f64,
        limits: ResourceLimits,
        page: PageBudget,
    ) -> Result<ErpAutomatonRangeOutcome<'_, V>, TemporalAutomatonError> {
        let mut ledger = ResourceLedger::new(limits);
        let machine = ErpFrontierMachine::new(query, self.kernel, tau, limits)?;
        let mut stack = Vec::new();
        let mut terminal = None;
        let done = self.is_empty();

        if !done {
            let root = self.dawg.root();
            let seed = machine.seed();
            let final_cost = machine
                .final_cost(seed)
                .map_err(TemporalAutomatonError::Resource)?;
            let final_bucket = if root.is_final() && final_cost.is_some() {
                root.value_at_final()
            } else {
                None
            };
            let mut edges = Vec::new();
            root.for_each_edge(|label, child| edges.push((label, child)));
            edges.sort_by_key(|(label, _)| *label);
            stack.push(ErpAutomatonRangeFrame {
                depth: 0,
                state: seed,
                final_bucket,
                next_candidate: 0,
                edges: edges.into(),
            });
            terminal = ledger.charge(ResourceKind::TrieNodes, 1).err();
        }

        let result_capacity = self
            .len()
            .min(DEFAULT_RESULT_BUFFER_CAPACITY)
            .min(limits.max_results);
        let mut continuation = ErpAutomatonRangeContinuation {
            index: self,
            query: query.into(),
            tau,
            machine,
            stack,
            results: Vec::with_capacity(result_capacity),
            pending_match: None,
            ledger,
            terminal,
            done,
        };
        if continuation.terminal.is_none() {
            if let Err(reason) = continuation.observe_state_peaks() {
                continuation.terminal = Some(reason);
            }
        }
        Ok(continuation.resume(page))
    }
}

impl<V> ElasticTransducer<MsmKernel, V>
where
    V: Eq + std::hash::Hash + Clone,
{
    /// Effective MSM configuration for the compatibility
    /// [`crate::time_series::MsmTransducer`] specialization.
    #[inline]
    pub fn msm_config(&self) -> &MsmConfig {
        self.kernel.config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::msm_interval::interval_column_len;
    use crate::time_series::MsmTransducer;
    use proptest::prelude::*;

    /// Brute-force reference set: every series within `tau`, sorted ascending.
    fn brute_range(
        series: &[Vec<f64>],
        query: &[f64],
        msm: &MsmConfig,
        tau: f64,
    ) -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = series
            .iter()
            .enumerate()
            .map(|(i, s)| (i, msm.distance(query, s)))
            .filter(|(_, d)| *d <= tau + 1e-9)
            .collect();
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        v
    }

    #[test]
    fn range_matches_brute_force_small() {
        let series = vec![
            vec![10.0, 20.0, 30.0],
            vec![11.0, 21.0, 29.0],
            vec![50.0, 60.0, 70.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![90.0, 10.0, 50.0],
        ];
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let idx = MsmTransducer::from_series(quant.clone(), msm, &series);

        let query = vec![12.0, 22.0, 31.0];
        for &tau in &[0.0, 2.0, 5.0, 10.0, 50.0, 1000.0] {
            let got = idx.search_range(&query, tau);
            let want = brute_range(&series, &query, &msm, tau);
            let got_ids: Vec<usize> = got.iter().map(|(i, _)| *i).collect();
            let want_ids: Vec<usize> = want.iter().map(|(i, _)| *i).collect();
            assert_eq!(got_ids, want_ids, "tau={tau}: id sets differ");
            for ((gi, gd), (wi, wd)) in got.iter().zip(want.iter()) {
                assert_eq!(gi, wi);
                assert!((gd - wd).abs() < 1e-9, "distance mismatch at id {gi}");
            }
        }
    }

    #[test]
    fn msm_transducer_empty_and_nonfinite_query_order_is_deterministic() {
        // Several references at *equal* MSM distance (every non-empty reference
        // is at +∞ for an empty or non-finite query). The result order must
        // come from the deterministic `buckets` store, never the randomized
        // `originals` HashMap. Independent index instances seed independent
        // HashMap `RandomState`s, so a HashMap-iterating path would order these
        // ties differently between instances; a bucket-iterating path is stable.
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);

        let build = || {
            let mut idx: MsmTransducer<usize> = MsmTransducer::new(quant.clone(), msm);
            for id in 0..16usize {
                let base = id as f64;
                idx.insert(id, &[1.0 + base, 2.0 + base, 3.0 + base]);
            }
            idx
        };

        let empty_query: Vec<f64> = Vec::new();
        let nonfinite_query = [f64::NAN, 1.0, 2.0];

        let reference = build();
        let empty_expected = reference.search_range(&empty_query, f64::INFINITY);
        let nonfinite_expected = reference.search_range(&nonfinite_query, f64::INFINITY);

        // Membership sanity: every id present, all at distance +∞.
        assert_eq!(empty_expected.len(), 16);
        assert!(empty_expected.iter().all(|(_, d)| d.is_infinite()));
        assert_eq!(nonfinite_expected.len(), 16);
        assert!(nonfinite_expected.iter().all(|(_, d)| d.is_infinite()));

        // Determinism across independent instances (fresh RandomState each time).
        for _ in 0..8 {
            let idx = build();
            assert_eq!(
                idx.search_range(&empty_query, f64::INFINITY),
                empty_expected
            );
            assert_eq!(
                idx.search_range(&nonfinite_query, f64::INFINITY),
                nonfinite_expected
            );
        }
    }

    #[test]
    fn constructor_normalizes_invalid_msm_cost() {
        let quant = QuantizationConfig::for_u8(0.0, 10.0);
        let negative: MsmTransducer<usize> =
            MsmTransducer::new(quant.clone(), MsmConfig { c: -1.0 });
        let infinite: MsmTransducer<usize> =
            MsmTransducer::new(quant, MsmConfig { c: f64::INFINITY });

        assert_eq!(negative.msm_config().c, 0.0);
        assert_eq!(*infinite.msm_config(), MsmConfig::default());
    }

    #[test]
    fn bin_bounds_for_uses_explicit_byte_unit_conversions() {
        let quant = QuantizationConfig::for_u8(0.0, 255.0);
        let zero_bounds = quant.bin_bounds(0);
        let max_bounds = quant.bin_bounds(u32::from(u8::MAX));
        let idx: MsmTransducer<usize> = MsmTransducer::new(quant, MsmConfig::new(1.0));

        assert_eq!(byte_unit_index(0), 0);
        assert_eq!(byte_unit_bin(0), 0);
        assert_eq!(byte_unit_index(u8::MAX), usize::from(u8::MAX));
        assert_eq!(byte_unit_bin(u8::MAX), u32::from(u8::MAX));
        assert_eq!(idx.bin_bounds_for(0), zero_bounds);
        assert_eq!(idx.bin_bounds_for(u8::MAX), max_bounds);
    }

    #[test]
    fn knn_matches_brute_force_small() {
        let series = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.5, 3.5],
            vec![5.0, 6.0, 7.0],
            vec![1.1, 2.1, 2.9],
            vec![9.0, 8.0, 7.0],
        ];
        let quant = QuantizationConfig::for_u8(0.0, 10.0);
        let msm = MsmConfig::new(0.5);
        let idx = MsmTransducer::from_series(quant.clone(), msm, &series);
        let query = vec![1.2, 2.2, 3.1];

        for k in 1..=series.len() {
            let got = idx.search_knn(&query, k, 0.5);
            let mut want = brute_range(&series, &query, &msm, f64::INFINITY);
            want.truncate(k);
            let got_d: Vec<f64> = got.iter().map(|(_, d)| *d).collect();
            let want_d: Vec<f64> = want.iter().map(|(_, d)| *d).collect();
            assert_eq!(got_d.len(), want_d.len(), "k={k} count");
            for (g, w) in got_d.iter().zip(want_d.iter()) {
                assert!((g - w).abs() < 1e-9, "k={k}: {g} != {w}");
            }
        }
    }

    #[test]
    fn knn_sequence_helpers_do_not_wrap_at_usize_boundary() {
        let mut result_sequence = usize::MAX - 1;
        assert_eq!(take_sequence(&mut result_sequence), Some(usize::MAX - 1));
        assert_eq!(result_sequence, usize::MAX);
        assert_eq!(take_sequence(&mut result_sequence), None);
        assert_eq!(result_sequence, usize::MAX);

        let mut queue_sequence = usize::MAX - 1;
        assert_eq!(next_sequence(&mut queue_sequence), Some(usize::MAX));
        assert_eq!(queue_sequence, usize::MAX);
        assert_eq!(next_sequence(&mut queue_sequence), None);
        assert_eq!(queue_sequence, usize::MAX);
    }

    #[test]
    fn traversal_column_width_uses_checked_interval_width() {
        assert_eq!(interval_column_len(0), Some(1));
        assert_eq!(interval_column_len(8), Some(9));
        assert_eq!(interval_column_len(usize::MAX), None);
    }

    #[test]
    fn knn_result_heap_maintains_bounded_top_k() {
        let mut best = BinaryHeap::new();
        let mut sequence = 0usize;

        assert!(MsmTransducer::push_knn_result(
            &mut best, 1usize, 5.0, 2, sequence
        ));
        sequence += 1;
        assert!(MsmTransducer::push_knn_result(
            &mut best, 2usize, 3.0, 2, sequence
        ));
        sequence += 1;
        assert!(MsmTransducer::push_knn_result(
            &mut best, 3usize, 4.0, 2, sequence
        ));
        sequence += 1;

        assert_eq!(
            MsmTransducer::finish_knn_results(best.clone()),
            vec![(2, 3.0), (3, 4.0)]
        );

        assert!(!MsmTransducer::push_knn_result(
            &mut best, 2usize, 7.0, 2, sequence
        ));
        sequence += 1;
        assert_eq!(
            MsmTransducer::finish_knn_results(best.clone()),
            vec![(2, 3.0), (3, 4.0)]
        );

        assert!(MsmTransducer::push_knn_result(
            &mut best, 3usize, 2.0, 2, sequence
        ));
        sequence += 1;
        assert_eq!(
            MsmTransducer::finish_knn_results(best.clone()),
            vec![(3, 2.0), (2, 3.0)]
        );

        assert!(!MsmTransducer::push_knn_result(
            &mut best, 4usize, 9.0, 2, sequence
        ));
        assert_eq!(
            MsmTransducer::finish_knn_results(best),
            vec![(3, 2.0), (2, 3.0)]
        );
    }

    #[test]
    fn range_result_finisher_keeps_best_distance_per_id_without_reordering_ties() {
        let results = vec![(7usize, 4.0), (9, 2.0), (7, 1.5), (11, 1.5), (9, 3.0)];

        assert_eq!(
            MsmTransducer::finish_range_results(results),
            vec![(7, 1.5), (11, 1.5), (9, 2.0)]
        );
    }

    #[test]
    fn collisions_are_all_recovered() {
        // Two distinct ids whose series quantize identically must both be
        // returned (no silent drop on key collision).
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);
        idx.insert(7, &[10.0, 20.0, 30.0]);
        idx.insert(9, &[10.2, 20.2, 30.2]); // same bins as id 7 at 256 bins over [0,100]
        let got = idx.search_range(&[10.1, 20.1, 30.1], 5.0);
        let ids: std::collections::HashSet<u32> = got.iter().map(|(v, _)| *v).collect();
        assert!(
            ids.contains(&7) && ids.contains(&9),
            "both colliding ids must appear: {ids:?}"
        );
    }

    #[test]
    fn reinserting_same_id_in_same_bucket_updates_original_without_duplicate_membership() {
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);

        assert!(idx.insert(7, &[10.0, 20.0, 30.0]));
        assert!(!idx.insert(7, &[10.01, 20.01, 30.01]));

        assert_eq!(idx.len(), 1);
        assert_eq!(idx.buckets.iter().map(Vec::len).sum::<usize>(), 1);
        let stored = idx.originals.get(&7).unwrap();
        assert_eq!(
            idx.buckets[stored.bucket_location.0][stored.bucket_location.1],
            7
        );
        assert_eq!(idx.get_original(&7), Some(&[10.01, 20.01, 30.01][..]));

        let got = idx.search_range(&[10.01, 20.01, 30.01], 0.0);
        assert_eq!(got, vec![(7, 0.0)]);
    }

    #[test]
    fn reinserting_same_id_in_different_bucket_relocates_membership() {
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);

        assert!(idx.insert(7, &[10.0]));
        assert!(idx.insert(9, &[10.01]));
        assert!(!idx.insert(7, &[90.0]));

        assert_eq!(idx.len(), 2);
        assert_eq!(idx.buckets.iter().map(Vec::len).sum::<usize>(), 2);
        assert_eq!(idx.get_original(&7), Some(&[90.0][..]));
        assert_eq!(idx.get_original(&9), Some(&[10.01][..]));

        let old_bucket = idx.quant.encode_u8(&[10.01]);
        let old_bucket_id = idx.dawg.get_bytes_value(&old_bucket).unwrap();
        assert_eq!(idx.buckets[old_bucket_id], vec![9]);
        assert_eq!(
            idx.originals.get(&9).map(|stored| stored.bucket_location),
            Some((old_bucket_id, 0))
        );

        assert_eq!(idx.search_range(&[10.01], 0.0), vec![(9, 0.0)]);
        assert_eq!(idx.search_range(&[90.0], 0.0), vec![(7, 0.0)]);
    }

    #[test]
    fn remove_value_clears_membership_and_releases_empty_bucket_storage() {
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);

        assert!(idx.insert(7, &[10.0, 20.0]));
        let bucket_id = idx.originals.get(&7).unwrap().bucket_location.0;
        assert_eq!(idx.search_range(&[10.0, 20.0], 0.0), vec![(7, 0.0)]);

        assert!(idx.remove(7));
        assert!(!idx.remove(7));

        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
        assert_eq!(idx.get_original(&7), None);
        assert!(idx.buckets[bucket_id].is_empty());
        assert_eq!(idx.buckets[bucket_id].capacity(), 0);
        assert!(idx.search_range(&[10.0, 20.0], 0.0).is_empty());
        assert!(idx.search_knn(&[10.0, 20.0], 1, 1.0).is_empty());
    }

    #[test]
    fn remove_value_preserves_swapped_bucket_location() {
        let quant = QuantizationConfig::for_u8(0.0, 100.0);
        let msm = MsmConfig::new(1.0);
        let mut idx: MsmTransducer<u32> = MsmTransducer::new(quant, msm);

        assert!(idx.insert(7, &[10.0]));
        assert!(idx.insert(9, &[10.01]));

        let shared_bucket = idx.originals.get(&7).unwrap().bucket_location.0;
        assert_eq!(
            idx.originals.get(&9).unwrap().bucket_location.0,
            shared_bucket
        );

        assert!(idx.remove(7));

        assert_eq!(idx.buckets[shared_bucket], vec![9]);
        assert_eq!(
            idx.originals.get(&9).map(|stored| stored.bucket_location),
            Some((shared_bucket, 0))
        );
        assert_eq!(idx.search_range(&[10.01], 0.0), vec![(9, 0.0)]);

        assert!(!idx.insert(9, &[90.0]));
        assert!(idx.buckets[shared_bucket].is_empty());
        assert_eq!(idx.buckets[shared_bucket].capacity(), 0);
        assert!(idx.search_range(&[10.01], 0.0).is_empty());
        assert_eq!(idx.search_range(&[90.0], 0.0), vec![(9, 0.0)]);
    }

    #[test]
    fn large_quantizer_is_coarsened_without_losing_exactness() {
        let series = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.2, 2.2, 3.2],
            vec![9.0, 9.0, 9.0],
        ];
        let quant = QuantizationConfig::for_u16(0.0, 10.0);
        let msm = MsmConfig::new(1.0);
        let idx = MsmTransducer::from_series(quant, msm, &series);
        let query = vec![1.1, 2.1, 3.1];

        assert_eq!(idx.quant_config().num_bins, 256);
        assert_eq!(
            idx.search_range(&query, 1.0),
            brute_range(&series, &query, &msm, 1.0)
        );
    }

    proptest! {
        // The exactness gate (A6): range search equals brute force for random
        // reference sets, queries, c, and thresholds.
        #[test]
        fn prop_range_exact(
            series in prop::collection::vec(
                prop::collection::vec(0.0f64..100.0, 1..7),
                1..15,
            ),
            query in prop::collection::vec(0.0f64..100.0, 1..7),
            c in 0.1f64..3.0,
            tau in 0.0f64..40.0,
        ) {
            let quant = QuantizationConfig::for_u8(0.0, 100.0);
            let msm = MsmConfig::new(c);
            let idx = MsmTransducer::from_series(quant, msm, &series);

            let got = idx.search_range(&query, tau);
            let want = brute_range(&series, &query, &msm, tau);

            // Same set of ids (dedup brute by best-distance id, since duplicate
            // identical series get distinct indices but identical distance).
            let got_ids: std::collections::HashSet<usize> = got.iter().map(|(i, _)| *i).collect();
            let want_ids: std::collections::HashSet<usize> = want.iter().map(|(i, _)| *i).collect();
            prop_assert_eq!(got_ids, want_ids);

            // Every emitted distance is the exact MSM distance.
            for (i, d) in &got {
                let exact = msm.distance(&query, &series[*i]);
                prop_assert!((d - exact).abs() < 1e-9);
            }
        }

        // k-NN exactness: returned distances equal the k smallest brute-force
        // distances (compared as a multiset of distances, robust to ties).
        #[test]
        fn prop_knn_exact(
            series in prop::collection::vec(
                prop::collection::vec(0.0f64..50.0, 1..6),
                1..12,
            ),
            query in prop::collection::vec(0.0f64..50.0, 1..6),
            c in 0.2f64..2.0,
            k in 1usize..6,
        ) {
            let quant = QuantizationConfig::for_u8(0.0, 50.0);
            let msm = MsmConfig::new(c);
            let idx = MsmTransducer::from_series(quant, msm, &series);

            let got = idx.search_knn(&query, k, 1.0);
            let mut want = brute_range(&series, &query, &msm, f64::INFINITY);
            want.truncate(k);

            prop_assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(want.iter()) {
                // Distances must match position-for-position (both sorted asc).
                prop_assert!((g.1 - w.1).abs() < 1e-9, "got {} want {}", g.1, w.1);
            }
        }
    }
}
