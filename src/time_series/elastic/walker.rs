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
//! $`\{\mathit{id}:D(\mathit{query},\mathit{reference}_{\mathit{id}})\le\tau\}`$ — no false negatives and no false
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
//!   subtree whose bound exceeds $`\tau`$ or the running k-th best is safely skipped.
//! * **Exact verification (no false positives).** At each final node whose
//!   column lower bound is within threshold, the stored **full-precision**
//!   original is re-scored through the reusable exact point-frontier workspace;
//!   only genuine matches are emitted (K3), after an optional admissible
//!   candidate bound (K4). The workspace shares the online transition engine
//!   and performs no per-candidate allocation.

use std::fmt;

/// SHA-256 identity of a complete canonical elastic snapshot payload.
///
/// The identity remains part of range certificates even when the optional
/// persistent snapshot backend is not compiled: `None` identifies an
/// in-memory index and `Some` binds evidence to one verified snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ElasticSnapshotIdentity(pub [u8; 32]);

impl fmt::Display for ElasticSnapshotIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[cfg(feature = "persistent-artrie")]
mod snapshot;
#[cfg(feature = "persistent-artrie")]
pub use snapshot::{
    ElasticSnapshot, ElasticSnapshotError, ElasticSnapshotKernel, ElasticSnapshotLimits,
    ElasticSnapshotMetadata, SnapshotPersistentDictionary,
};

use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use libdictenstein::dynamic_dawg::DynamicDawg;
#[cfg(feature = "persistent-artrie")]
use libdictenstein::persistent_artrie::{PersistentARTrie, PersistentARTrieNode};
use libdictenstein::{Dictionary, DictionaryNode, DictionaryTraversalRoot, MappedDictionaryNode};
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use thiserror::Error;

use super::{Cost, ElasticKernel, PointFrontierStep};
use crate::cost::CostMonoid;
use crate::time_series::automaton::{
    BoundedTransitionCache, ErpFrontierMachine, ExactPointDecision, ExactPointWorkspace,
    TemporalAutomatonError, TemporalStateId,
};
use crate::time_series::bounded::{
    IncompleteReason, Operand, OperationOutcome, PageBudget, ResourceKind, ResourceLedger,
    ResourceLimits, ResourceUsage, TemporalValidationError,
};
use crate::time_series::encoding::QuantizationConfig;
use crate::time_series::kernels::ErpConfig;
use crate::time_series::msm::MsmConfig;
use crate::time_series::msm_kernel::MsmKernel;
use crate::transducer::dictionary_traversal::{DfsNodeEdges, TraversalCursor, TraversalSession};

const DEFAULT_RESULT_BUFFER_CAPACITY: usize = 64;

/// Narrow dictionary boundary used by exact elastic traversal.
///
/// Search depends only on an immutable root whose labels are quantized bytes
/// and whose terminal values identify collision buckets.  The default backend
/// is [`DynamicDawg`]; complete snapshots load the same lazy product walker over
/// a byte-keyed [`PersistentARTrie`] without projecting the dictionary into a
/// second in-memory graph.
pub trait ElasticDictionaryBackend: std::fmt::Debug {
    /// Backend-native quantized edge label. Current scalar elastic indexes bind
    /// this to `u8`; typed timestamp/vector profiles can implement the same
    /// boundary with whole labels instead of flattening channels into bytes.
    type Label: libdictenstein::CharUnit;

    /// Immutable node handle for one captured dictionary revision.
    type Node: DictionaryNode<Unit = Self::Label> + MappedDictionaryNode<Value = usize>;

    /// Capture the root of one immutable revision.
    fn elastic_root(&self) -> Self::Node;

    /// Look up a collision-bucket id by its exact quantized byte key.
    fn elastic_bucket(&self, key: &[Self::Label]) -> Option<usize>;

    /// Exact number of terminal keys in the captured dictionary revision.
    fn elastic_len(&self) -> Option<usize>;
}

/// Dictionary mutation boundary with a no-visible-change-on-error contract.
///
/// An implementation may return `Err` only if lookup, traversal, and terminal
/// cardinality remain exactly as they were before the call. Backends that can
/// publish a write before reporting a later durability error must not implement
/// this trait. In particular, complete persistent snapshot dictionaries are
/// deliberately search-only at the type boundary.
pub trait ElasticMutableDictionaryBackend: ElasticDictionaryBackend {
    /// Fallibly insert one absent exact quantized key without updating an
    /// existing terminal. `Ok(false)` and `Err(_)` must leave it unchanged.
    fn elastic_try_insert_bucket(
        &mut self,
        key: &[Self::Label],
        bucket: usize,
    ) -> Result<bool, ElasticMutationError>;
}

impl ElasticDictionaryBackend for DynamicDawg<usize> {
    type Label = u8;
    type Node = <Self as Dictionary>::Node;

    fn elastic_root(&self) -> Self::Node {
        self.root()
    }

    fn elastic_bucket(&self, key: &[u8]) -> Option<usize> {
        self.get_bytes_value(key)
    }

    fn elastic_len(&self) -> Option<usize> {
        Dictionary::len(self)
    }
}

impl ElasticMutableDictionaryBackend for DynamicDawg<usize> {
    fn elastic_try_insert_bucket(
        &mut self,
        key: &[u8],
        bucket: usize,
    ) -> Result<bool, ElasticMutationError> {
        Ok(self.update_or_insert_bytes(key, bucket, |_| {}))
    }
}

#[cfg(feature = "persistent-artrie")]
impl ElasticDictionaryBackend for PersistentARTrie<usize> {
    type Label = u8;
    type Node = PersistentARTrieNode<usize>;

    fn elastic_root(&self) -> Self::Node {
        self.root()
    }

    fn elastic_bucket(&self, key: &[u8]) -> Option<usize> {
        self.get_value_bytes(key)
    }

    fn elastic_len(&self) -> Option<usize> {
        Dictionary::len(self)
    }
}

/// Fail-closed error from a transactional elastic-index mutation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElasticMutationError {
    /// A bounded heap reservation failed before state was changed.
    AllocationFailed {
        /// Requested byte count when it can be represented.
        requested: usize,
    },
    /// The dictionary backend rejected its durable mutation.
    Dictionary(String),
    /// A missing key unexpectedly became present during the mutation boundary.
    DictionaryConflict,
    /// Existing dictionary, bucket, and original-series state disagreed.
    InvalidState,
}

impl fmt::Display for ElasticMutationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllocationFailed { requested } => {
                write!(
                    formatter,
                    "elastic mutation allocation of {requested} bytes failed"
                )
            }
            Self::Dictionary(error) => {
                write!(formatter, "elastic dictionary mutation failed: {error}")
            }
            Self::DictionaryConflict => {
                formatter.write_str("elastic dictionary changed during transactional mutation")
            }
            Self::InvalidState => formatter
                .write_str("elastic dictionary, collision buckets, and originals are inconsistent"),
        }
    }
}

impl std::error::Error for ElasticMutationError {}

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

/// Hard evidence/work ceilings for an optional exact range certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElasticCertificateLimits {
    /// Common validation, cumulative-work, and peak-retention ceilings.
    pub resources: ResourceLimits,
    /// Maximum evidence records retained.
    pub max_records: usize,
    /// Maximum total quantized-path bytes retained across all records.
    pub max_path_bytes: usize,
    /// Maximum charged transition/candidate dynamic-programming work.
    pub max_work_units: usize,
}

impl Default for ElasticCertificateLimits {
    fn default() -> Self {
        let resources = ResourceLimits::default();
        Self {
            resources,
            max_records: resources.max_results,
            max_path_bytes: resources.max_witness_bytes,
            max_work_units: resources.max_work_units,
        }
    }
}

/// One independently checkable K1--K4 decision in an exact range traversal.
#[derive(Clone, Debug, PartialEq)]
pub enum ElasticRangeEvidence<C> {
    /// K1 prefix bound rejected this edge and its complete descendant language.
    PrefixPruned {
        /// Quantized dictionary prefix whose descendant language was rejected.
        quantized_path: Vec<u8>,
        /// Certified K1 lower bound for every descendant of `quantized_path`.
        lower_bound: C,
    },
    /// K2 interval-column bound rejected the complete child subtree.
    SubtreePruned {
        /// Quantized child prefix whose complete subtree was rejected.
        quantized_path: Vec<u8>,
        /// Certified K2 lower bound for every terminal below the child prefix.
        lower_bound: C,
    },
    /// The accepting path's terminal interval row rejected this collision bucket.
    TerminalPruned {
        /// Quantized accepting key whose collision bucket was rejected.
        quantized_path: Vec<u8>,
        /// Certified terminal-row lower bound for the complete collision bucket.
        lower_bound: C,
    },
    /// K4 rejected one full-precision collision member before exact scoring.
    CandidatePruned {
        /// Quantized key whose collision bucket contains the candidate.
        quantized_path: Vec<u8>,
        /// Stable identity of the rejected full-precision candidate.
        stable_id: u64,
        /// Certified K4 candidate-specific lower bound.
        candidate_bound: C,
    },
    /// K3 exact verification result; `survived` is exactly `exact <= cutoff`.
    ExactCandidate {
        /// Quantized key whose collision bucket contains the candidate.
        quantized_path: Vec<u8>,
        /// Stable identity of the exactly evaluated full-precision candidate.
        stable_id: u64,
        /// Certified K4 lower bound evaluated before exact scoring.
        candidate_bound: C,
        /// Exact K3 score when it lies within the active cutoff.
        exact: Option<C>,
        /// Whether the exact candidate belongs to the requested closed range.
        survived: bool,
    },
}

/// Deterministic evidence stream for one exact range query.
#[derive(Clone, Debug, PartialEq)]
pub struct ElasticRangeCertificate<C> {
    /// Complete-snapshot identity when the searched index came from a snapshot.
    pub snapshot_identity: Option<ElasticSnapshotIdentity>,
    /// Exact IEEE-754 query words, in channel/sample order.
    pub query_bits: Vec<u64>,
    /// Exact range cutoff.
    pub cutoff: C,
    /// Canonically ordered K1--K4 decisions.
    pub evidence: Vec<ElasticRangeEvidence<C>>,
    /// Charged transition/candidate dynamic-programming work.
    pub work_units: usize,
    /// Total quantized-path bytes retained in `evidence`.
    pub path_bytes: usize,
    /// Total logical certificate storage charged to the witness ceiling.
    ///
    /// This includes the query bit pattern, every evidence-record header, and
    /// every owned quantized path. It deliberately excludes allocator slack,
    /// which never changes the serializable proof object.
    pub witness_bytes: usize,
}

/// Fail-closed certificate construction error.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ElasticCertificateError {
    /// The query or cutoff is outside the kernel's documented domain.
    #[error(transparent)]
    Validation(#[from] TemporalValidationError),
    /// This kernel/backend pair cannot produce complete replayable evidence.
    #[error("this kernel/backend pair does not support complete replayable evidence")]
    Unsupported,
    /// The captured dictionary, bucket table, or original table is inconsistent.
    #[error("the captured dictionary, collision buckets, and originals are inconsistent")]
    InvalidStoredData,
    /// Exact arithmetic produced a non-finite value inside a finite domain.
    #[error("exact certificate arithmetic produced a non-finite value")]
    NumericOverflow,
    /// A declared evidence/work ceiling was exceeded.
    #[error("{resource:?} budget exceeded: limit {limit}, requested {requested}")]
    BudgetExceeded {
        /// Resource whose declared ceiling would be exceeded.
        resource: ResourceKind,
        /// Declared inclusive ceiling for the resource.
        limit: usize,
        /// Resource amount required by the attempted operation.
        requested: usize,
    },
    /// Checked accounting overflowed.
    #[error("checked {resource:?} certificate accounting overflowed")]
    ArithmeticOverflow {
        /// Resource whose checked accounting overflowed.
        resource: ResourceKind,
    },
    /// A fallible certificate allocation failed.
    #[error("failed to allocate {requested} units of {resource:?} for the certificate")]
    AllocationFailed {
        /// Resource whose backing allocation failed.
        resource: ResourceKind,
        /// Requested allocation size in the resource's documented unit.
        requested: usize,
    },
}

impl From<IncompleteReason> for ElasticCertificateError {
    fn from(reason: IncompleteReason) -> Self {
        match reason {
            IncompleteReason::BudgetExceeded {
                resource,
                limit,
                requested,
            } => Self::BudgetExceeded {
                resource,
                limit,
                requested,
            },
            IncompleteReason::ArithmeticOverflow { resource } => {
                Self::ArithmeticOverflow { resource }
            }
            IncompleteReason::AllocationFailed {
                resource,
                requested,
            } => Self::AllocationFailed {
                resource,
                requested,
            },
            IncompleteReason::NumericOverflow => Self::NumericOverflow,
            IncompleteReason::InvalidStoredData => Self::InvalidStoredData,
            IncompleteReason::Unsupported | IncompleteReason::Cancelled => Self::Unsupported,
        }
    }
}

struct CertificateBuilder<C> {
    limits: ElasticCertificateLimits,
    evidence: Vec<ElasticRangeEvidence<C>>,
    work_units: usize,
    path_bytes: usize,
    witness_bytes: usize,
}

impl<C> CertificateBuilder<C> {
    fn new(
        limits: ElasticCertificateLimits,
        base_witness_bytes: usize,
    ) -> Result<Self, ElasticCertificateError> {
        if base_witness_bytes > limits.resources.max_witness_bytes {
            return Err(ElasticCertificateError::BudgetExceeded {
                resource: ResourceKind::WitnessBytes,
                limit: limits.resources.max_witness_bytes,
                requested: base_witness_bytes,
            });
        }
        let mut evidence = Vec::new();
        let record_bytes = std::mem::size_of::<ElasticRangeEvidence<C>>().max(1);
        let witness_record_capacity = limits
            .resources
            .max_witness_bytes
            .saturating_sub(base_witness_bytes)
            / record_bytes;
        let initial = limits
            .max_records
            .min(limits.resources.max_results)
            .min(witness_record_capacity)
            .min(DEFAULT_RESULT_BUFFER_CAPACITY);
        evidence.try_reserve_exact(initial).map_err(|_| {
            ElasticCertificateError::AllocationFailed {
                resource: ResourceKind::WitnessBytes,
                requested: initial.saturating_mul(std::mem::size_of::<ElasticRangeEvidence<C>>()),
            }
        })?;
        Ok(Self {
            limits,
            evidence,
            work_units: 0,
            path_bytes: 0,
            witness_bytes: base_witness_bytes,
        })
    }

    fn charge_work(&mut self, amount: usize) -> Result<(), ElasticCertificateError> {
        let requested = self.work_units.checked_add(amount).ok_or(
            ElasticCertificateError::ArithmeticOverflow {
                resource: ResourceKind::WorkUnits,
            },
        )?;
        let limit = self
            .limits
            .max_work_units
            .min(self.limits.resources.max_work_units);
        if requested > limit {
            return Err(ElasticCertificateError::BudgetExceeded {
                resource: ResourceKind::WorkUnits,
                limit,
                requested,
            });
        }
        self.work_units = requested;
        Ok(())
    }

    fn record<F>(&mut self, path: &[u8], make: F) -> Result<(), ElasticCertificateError>
    where
        F: FnOnce(Vec<u8>) -> ElasticRangeEvidence<C>,
    {
        let requested_records = self.evidence.len().checked_add(1).ok_or(
            ElasticCertificateError::ArithmeticOverflow {
                resource: ResourceKind::Results,
            },
        )?;
        let record_limit = self
            .limits
            .max_records
            .min(self.limits.resources.max_results);
        if requested_records > record_limit {
            return Err(ElasticCertificateError::BudgetExceeded {
                resource: ResourceKind::Results,
                limit: record_limit,
                requested: requested_records,
            });
        }
        let requested_path_bytes = self.path_bytes.checked_add(path.len()).ok_or(
            ElasticCertificateError::ArithmeticOverflow {
                resource: ResourceKind::WitnessBytes,
            },
        )?;
        if requested_path_bytes > self.limits.max_path_bytes {
            return Err(ElasticCertificateError::BudgetExceeded {
                resource: ResourceKind::WitnessBytes,
                limit: self.limits.max_path_bytes,
                requested: requested_path_bytes,
            });
        }
        let record_bytes = std::mem::size_of::<ElasticRangeEvidence<C>>();
        let requested_witness_bytes = self
            .witness_bytes
            .checked_add(record_bytes)
            .and_then(|bytes| bytes.checked_add(path.len()))
            .ok_or(ElasticCertificateError::ArithmeticOverflow {
                resource: ResourceKind::WitnessBytes,
            })?;
        if requested_witness_bytes > self.limits.resources.max_witness_bytes {
            return Err(ElasticCertificateError::BudgetExceeded {
                resource: ResourceKind::WitnessBytes,
                limit: self.limits.resources.max_witness_bytes,
                requested: requested_witness_bytes,
            });
        }
        self.evidence.try_reserve_exact(1).map_err(|_| {
            ElasticCertificateError::AllocationFailed {
                resource: ResourceKind::WitnessBytes,
                requested: requested_records
                    .saturating_mul(std::mem::size_of::<ElasticRangeEvidence<C>>()),
            }
        })?;
        let mut owned_path = Vec::new();
        owned_path.try_reserve_exact(path.len()).map_err(|_| {
            ElasticCertificateError::AllocationFailed {
                resource: ResourceKind::WitnessBytes,
                requested: requested_path_bytes,
            }
        })?;
        owned_path.extend_from_slice(path);
        self.evidence.push(make(owned_path));
        self.path_bytes = requested_path_bytes;
        self.witness_bytes = requested_witness_bytes;
        Ok(())
    }
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
pub struct ElasticTransducer<
    K: ElasticKernel,
    V: Eq + std::hash::Hash + Clone = usize,
    D: ElasticDictionaryBackend = DynamicDawg<usize>,
> {
    /// Prefix-sharing trie over the u8-quantized reference sequences. The
    /// stored value is the final bucket id for all references that quantize to
    /// that byte key.
    dawg: D,
    /// Quantization configuration (defines the per-bin intervals).
    quant: QuantizationConfig,
    /// Kernel supplying relaxed columns and exact verification.
    kernel: K,
    /// Precomputed quantization-bin intervals for hot trie-edge traversal.
    bin_bounds: Vec<(f64, f64)>,
    /// Final bucket id → all reference ids sharing that quantized key.
    buckets: Vec<Vec<V>>,
    /// Reference ID to original series and bucket slot for exact verification and
    /// $`\mathcal{O}(1)`$ upserts.
    originals: HashMap<V, StoredSeries>,
    /// Verified identity of the complete snapshot that produced this index.
    snapshot_identity: Option<ElasticSnapshotIdentity>,
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

/// One explicit DFS frame for certificate construction.
///
/// As in the ordinary bounded product, the frame owns no DP column. A compact
/// exact residual ID addresses the query-local canonical-state arena; the
/// shared path zipper below the stack supplies replayable path evidence.
struct CertifiedRangeFrame<N: DictionaryNode> {
    depth: usize,
    state: TemporalStateId,
    candidate_bucket: Option<usize>,
    candidates: Vec<u64>,
    next_candidate: usize,
    edges: DfsNodeEdges<N>,
}

/// Explicit bounded-traversal frame containing only a compact product-state
/// identifier and dictionary cursor state.
struct BoundedRangeFrame<N: DictionaryNode> {
    depth: usize,
    state: TemporalStateId,
    final_bucket: Option<usize>,
    next_candidate: usize,
    edges: DfsNodeEdges<N>,
}

impl<N> BoundedRangeFrame<N>
where
    N: DictionaryNode<Unit = u8> + MappedDictionaryNode<Value = usize>,
{
    fn open(
        traversal: &mut TraversalSession<N>,
        cursor: TraversalCursor<N::SnapshotCursor>,
        depth: usize,
        state: TemporalStateId,
        terminal_admitted: bool,
    ) -> Result<Self, IncompleteReason> {
        let final_value = traversal.final_value_at_cursor(cursor, None);
        let edges = traversal.open_dfs_node(cursor);
        let final_bucket = if edges.is_final() {
            let bucket = final_value.ok_or(IncompleteReason::InvalidStoredData)?;
            terminal_admitted.then_some(bucket)
        } else {
            if final_value.is_some() {
                return Err(IncompleteReason::InvalidStoredData);
            }
            None
        };
        Ok(Self {
            depth,
            state,
            final_bucket,
            next_candidate: 0,
            edges,
        })
    }
}

/// One live kernel state in the on-the-fly dictionary product.
struct RangeProductPosition<K: ElasticKernel> {
    row: u32,
    cost: Cost<K>,
}

struct RangeProductState<K: ElasticKernel> {
    depth: usize,
    carry: Option<K::Carry>,
    positions: Vec<RangeProductPosition<K>>,
    final_cost: Cost<K>,
}

struct RangeProductStateArena<K: ElasticKernel> {
    max_states: usize,
    max_positions: usize,
    position_count: usize,
    position_capacity_count: usize,
    collision_heap_capacity_count: usize,
    reused_states: usize,
    states: Vec<RangeProductState<K>>,
    fingerprints: FxHashMap<u64, SmallVec<[TemporalStateId; 2]>>,
}

impl<K: ElasticKernel> RangeProductStateArena<K> {
    fn new(max_states: usize, max_positions: usize) -> Self {
        Self {
            max_states,
            max_positions,
            position_count: 0,
            position_capacity_count: 0,
            collision_heap_capacity_count: 0,
            reused_states: 0,
            states: Vec::new(),
            fingerprints: FxHashMap::default(),
        }
    }

    fn fingerprint(kernel: &K, state: &RangeProductState<K>) -> Option<u64> {
        let mut hasher = FxHasher::default();
        state.depth.hash(&mut hasher);
        match state.carry {
            Some(carry) => {
                1_u8.hash(&mut hasher);
                kernel.canonical_carry_key(carry)?.hash(&mut hasher);
            }
            None => 0_u8.hash(&mut hasher),
        }
        K::Monoid::canonical_state_key(state.final_cost)?.hash(&mut hasher);
        state.positions.len().hash(&mut hasher);
        for position in &state.positions {
            position.row.hash(&mut hasher);
            K::Monoid::canonical_state_key(position.cost)?.hash(&mut hasher);
        }
        Some(hasher.finish())
    }

    fn exactly_equal(
        kernel: &K,
        left: &RangeProductState<K>,
        right: &RangeProductState<K>,
    ) -> bool {
        if left.depth != right.depth || left.positions.len() != right.positions.len() {
            return false;
        }
        let carries_equal = match (left.carry, right.carry) {
            (None, None) => true,
            (Some(left), Some(right)) => {
                kernel.canonical_carry_key(left) == kernel.canonical_carry_key(right)
            }
            (None, Some(_)) | (Some(_), None) => false,
        };
        carries_equal
            && K::Monoid::canonical_state_key(left.final_cost)
                == K::Monoid::canonical_state_key(right.final_cost)
            && left
                .positions
                .iter()
                .zip(&right.positions)
                .all(|(left, right)| {
                    left.row == right.row
                        && K::Monoid::canonical_state_key(left.cost)
                            == K::Monoid::canonical_state_key(right.cost)
                })
    }

    fn intern(
        &mut self,
        kernel: &K,
        state: RangeProductState<K>,
    ) -> Result<TemporalStateId, IncompleteReason> {
        let fingerprint = Self::fingerprint(kernel, &state);
        if let Some(fingerprint) = fingerprint {
            if let Some(candidates) = self.fingerprints.get(&fingerprint) {
                for candidate in candidates {
                    let existing = self
                        .states
                        .get(candidate.index())
                        .ok_or(IncompleteReason::InvalidStoredData)?;
                    if Self::exactly_equal(kernel, existing, &state) {
                        self.reused_states = self.reused_states.saturating_add(1);
                        return Ok(*candidate);
                    }
                }
            }
        }

        let requested_states =
            self.states
                .len()
                .checked_add(1)
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::QueueEntries,
                })?;
        if requested_states > self.max_states || requested_states > u32::MAX as usize {
            return Err(IncompleteReason::BudgetExceeded {
                resource: ResourceKind::QueueEntries,
                limit: self.max_states.min(u32::MAX as usize),
                requested: requested_states,
            });
        }
        let requested_positions = self
            .position_count
            .checked_add(state.positions.len())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        if requested_positions > self.max_positions {
            return Err(IncompleteReason::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: self.max_positions,
                requested: requested_positions,
            });
        }

        self.states
            .try_reserve(1)
            .map_err(|_| IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested: requested_states,
            })?;
        let mut collision_capacity_before = 0;
        if let Some(fingerprint) = fingerprint {
            self.fingerprints
                .try_reserve(1)
                .map_err(|_| IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: requested_states,
                })?;
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
        }

        let id = TemporalStateId(u32::try_from(self.states.len()).map_err(|_| {
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::QueueEntries,
            }
        })?);
        let position_capacity = state.positions.capacity();
        self.states.push(state);
        if let Some(fingerprint) = fingerprint {
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
        }
        self.position_count = requested_positions;
        self.position_capacity_count = self
            .position_capacity_count
            .checked_add(position_capacity)
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        Ok(id)
    }

    #[inline]
    fn get(&self, id: TemporalStateId) -> Option<&RangeProductState<K>> {
        self.states.get(id.index())
    }

    fn retained_bytes(&self) -> Option<usize> {
        let headers = self
            .states
            .capacity()
            .checked_mul(std::mem::size_of::<RangeProductState<K>>())?;
        let positions = self
            .position_capacity_count
            .checked_mul(std::mem::size_of::<RangeProductPosition<K>>())?;
        let fingerprints = self
            .fingerprints
            .capacity()
            .checked_mul(std::mem::size_of::<(u64, SmallVec<[TemporalStateId; 2]>)>())?;
        let collision_ids = self
            .collision_heap_capacity_count
            .checked_mul(std::mem::size_of::<TemporalStateId>())?;
        headers
            .checked_add(positions)?
            .checked_add(fingerprints)?
            .checked_add(collision_ids)
    }
}

/// Retained-state diagnostics for one on-the-fly dictionary product.
///
/// The query-local arena retains every distinct canonical state reached so far,
/// including states no longer on the explicit DFS path, because cached product
/// transitions may reuse them later. `column_cells` counts cells in that full
/// retained arena; it is independent of raw target-prefix length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElasticProductStateStats {
    /// Live explicit dictionary-traversal frames.
    pub frames: usize,
    /// Live query-automaton states addressed by compact IDs.
    pub states: usize,
    /// Kernel cells retained by all live product states.
    pub column_cells: usize,
    /// Complete observed transitions retained in the bounded cache.
    pub cached_transitions: usize,
    /// Equal canonical residuals that reused an existing state ID.
    pub reused_states: usize,
}

#[expect(
    clippy::large_enum_variant,
    reason = "the hot trie session stays inline so page resumption adds no heap indirection or allocation"
)]
enum RangeSessionMode<K: ElasticKernel, N: DictionaryNode> {
    Trie {
        traversal: TraversalSession<N>,
        stack: Vec<BoundedRangeFrame<N>>,
        states: RangeProductStateArena<K>,
        cache: BoundedTransitionCache<u8>,
        column_width: usize,
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
struct ErpAutomatonRangeFrame<N: DictionaryNode> {
    depth: usize,
    state: TemporalStateId,
    final_bucket: Option<usize>,
    next_candidate: usize,
    edges: DfsNodeEdges<N>,
}

impl<N> ErpAutomatonRangeFrame<N>
where
    N: DictionaryNode<Unit = u8> + MappedDictionaryNode<Value = usize>,
{
    fn open(
        traversal: &mut TraversalSession<N>,
        cursor: TraversalCursor<N::SnapshotCursor>,
        depth: usize,
        state: TemporalStateId,
        terminal_admitted: bool,
    ) -> Result<Self, IncompleteReason> {
        let final_value = traversal.final_value_at_cursor(cursor, None);
        let edges = traversal.open_dfs_node(cursor);
        let final_bucket = if edges.is_final() {
            let bucket = final_value.ok_or(IncompleteReason::InvalidStoredData)?;
            terminal_admitted.then_some(bucket)
        } else {
            if final_value.is_some() {
                return Err(IncompleteReason::InvalidStoredData);
            }
            None
        };
        Ok(Self {
            depth,
            state,
            final_bucket,
            next_candidate: 0,
            edges,
        })
    }
}

/// Resumable exact ERP range traversal using an on-demand automaton product.
///
/// Each dictionary edge constructs only the reachable canonical ERP frontier.
/// Stack depth is represented by an explicit `Vec`, so traversal never uses
/// the call stack. Quantization collisions remain candidates until every
/// full-precision original has been independently verified.
pub struct ErpAutomatonRangeContinuation<'a, V, D: ElasticDictionaryBackend = DynamicDawg<usize>>
where
    V: Eq + std::hash::Hash + Clone,
{
    index: &'a ElasticTransducer<ErpConfig, V, D>,
    query: Vec<f64>,
    tau: f64,
    machine: ErpFrontierMachine,
    traversal: TraversalSession<D::Node>,
    stack: Vec<ErpAutomatonRangeFrame<D::Node>>,
    results: Vec<(V, f64)>,
    pending_match: Option<(V, f64)>,
    ledger: ResourceLedger,
    terminal: Option<IncompleteReason>,
    done: bool,
}

impl<V, D> std::fmt::Debug for ErpAutomatonRangeContinuation<'_, V, D>
where
    V: Eq + std::hash::Hash + Clone,
    D: ElasticDictionaryBackend<Label = u8>,
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
/// the indexed snapshot between pages. Results discovered so far are borrowed
/// through [`RangeContinuation::exact_partial`]; they do not establish absence
/// until [`OperationOutcome::Complete`]. A paused outcome does not duplicate
/// this potentially large vector into its optional `partial` field.
pub struct RangeContinuation<'a, K, V, D: ElasticDictionaryBackend = DynamicDawg<usize>>
where
    K: ElasticKernel,
    V: Eq + std::hash::Hash + Clone,
{
    index: &'a ElasticTransducer<K, V, D>,
    query: Vec<f64>,
    workspace: Option<ExactPointWorkspace<K>>,
    tau: Cost<K>,
    mode: RangeSessionMode<K, D::Node>,
    results: Vec<(V, Cost<K>)>,
    pending_match: Option<(V, Cost<K>)>,
    ledger: ResourceLedger,
    terminal: Option<IncompleteReason>,
}

/// Exact identifier/distance pairs returned by elastic range and kNN queries.
pub type ExactRangeResults<K, V> = Vec<(V, Cost<K>)>;

/// Exact results paired with their complete deterministic range certificate.
pub type CertifiedRangeResults<K> = (Vec<(u64, Cost<K>)>, ElasticRangeCertificate<Cost<K>>);

/// Tagged outcome of a bounded, resumable exact elastic range query.
pub type BoundedRangeOutcome<'a, K, V, D = DynamicDawg<usize>> =
    OperationOutcome<ExactRangeResults<K, V>, RangeContinuation<'a, K, V, D>>;

/// Tagged outcome of a bounded non-resumable exact elastic search.
pub type ExactSearchOutcome<K, V> = OperationOutcome<ExactRangeResults<K, V>>;

/// Tagged outcome of the specialized canonical ERP automaton product.
pub type ErpAutomatonRangeOutcome<'a, V, D = DynamicDawg<usize>> =
    OperationOutcome<Vec<(V, f64)>, ErpAutomatonRangeContinuation<'a, V, D>>;

impl<K, V, D> std::fmt::Debug for RangeContinuation<'_, K, V, D>
where
    K: ElasticKernel,
    V: Eq + std::hash::Hash + Clone,
    D: ElasticDictionaryBackend<Label = u8>,
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

impl<'a, K, V, D> RangeContinuation<'a, K, V, D>
where
    K: ElasticKernel,
    V: Eq + std::hash::Hash + Clone,
    D: ElasticDictionaryBackend<Label = u8>,
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

    /// Cancel this query explicitly without converting its exact partial set
    /// into evidence of exhaustive absence.
    pub fn cancel(mut self) -> OperationOutcome<Vec<(V, Cost<K>)>, Self> {
        match self.take_finished_results() {
            Ok(partial) => OperationOutcome::Incomplete {
                partial: Some(partial),
                reason: IncompleteReason::Cancelled,
                continuation: None,
                usage: self.ledger.usage(),
            },
            Err(reason) => OperationOutcome::Incomplete {
                partial: None,
                reason,
                continuation: None,
                usage: self.ledger.usage(),
            },
        }
    }

    /// Return the live dictionary × kernel product-state footprint.
    ///
    /// `frames` counts the explicit DFS path. `states` counts distinct exact
    /// residuals retained in the bounded query-local interner, so equal states
    /// reached by different dictionary prefixes share one compact ID. Exact
    /// scan fallback and completed sessions retain no product states.
    pub fn retained_product_state_stats(&self) -> ElasticProductStateStats {
        match &self.mode {
            RangeSessionMode::Trie {
                stack,
                states,
                cache,
                ..
            } => ElasticProductStateStats {
                frames: stack.len(),
                states: states.states.len(),
                column_cells: states.position_count,
                cached_transitions: cache.len(),
                reused_states: states.reused_states,
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
                if continuation.is_some() {
                    // The continuation already owns this exact subset. Do not
                    // duplicate and re-sort an ever-growing result vector on
                    // every page; callers can borrow it through
                    // `exact_partial()` before resuming.
                    OperationOutcome::Incomplete {
                        partial: None,
                        reason,
                        continuation: Some(self),
                        usage,
                    }
                } else {
                    match self.take_finished_results() {
                        Ok(partial) => OperationOutcome::Incomplete {
                            partial: Some(partial),
                            reason,
                            continuation: None,
                            usage,
                        },
                        Err(finalization_reason) => OperationOutcome::Incomplete {
                            partial: None,
                            reason: finalization_reason,
                            continuation: None,
                            usage,
                        },
                    }
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

    fn reserve_result_slot(&mut self) -> Result<(), IncompleteReason> {
        if self.results.len() < self.results.capacity() {
            return Ok(());
        }
        let requested =
            self.results
                .len()
                .checked_add(1)
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::Results,
                })?;
        self.results
            .try_reserve_exact(1)
            .map_err(|_| IncompleteReason::AllocationFailed {
                resource: ResourceKind::Results,
                requested,
            })
    }

    fn take_finished_results(&mut self) -> Result<Vec<(V, Cost<K>)>, IncompleteReason> {
        let retained_scratch = self
            .workspace
            .as_ref()
            .map_or(0, ExactPointWorkspace::retained_bytes);
        ElasticTransducer::<K, V, D>::try_finish_bounded_range_results(
            std::mem::take(&mut self.results),
            retained_scratch,
            &mut self.ledger,
        )
    }

    fn observe_state_peaks(&mut self) -> Result<(), IncompleteReason> {
        let (queue_entries, automaton_bytes, edge_bytes) = match &self.mode {
            RangeSessionMode::Trie {
                stack,
                states,
                cache,
                ..
            } => {
                let state_bytes =
                    states
                        .retained_bytes()
                        .ok_or(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::ScratchBytes,
                        })?;
                let cache_bytes =
                    cache
                        .retained_bytes()
                        .ok_or(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::ScratchBytes,
                        })?;
                let workspace_bytes = self
                    .workspace
                    .as_ref()
                    .ok_or(IncompleteReason::InvalidStoredData)?
                    .retained_bytes();
                let automaton_bytes = state_bytes
                    .checked_add(cache_bytes)
                    .and_then(|bytes| bytes.checked_add(workspace_bytes))
                    .ok_or(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    })?;
                // Each frame owns only one fixed-capacity inline edge page;
                // that storage is already included in the frame header below.
                (stack.len(), automaton_bytes, 0)
            }
            RangeSessionMode::Scan { .. } | RangeSessionMode::Done => (
                0,
                self.workspace
                    .as_ref()
                    .map_or(0, ExactPointWorkspace::retained_bytes),
                0,
            ),
        };

        let construction_peak = self
            .workspace
            .as_ref()
            .map_or(0, ExactPointWorkspace::construction_peak_bytes);

        self.ledger
            .observe_peak(ResourceKind::QueueEntries, queue_entries)?;
        self.ledger.observe_peak(
            ResourceKind::ScratchBytes,
            automaton_bytes.max(construction_peak),
        )?;

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
                .checked_mul(std::mem::size_of::<BoundedRangeFrame<D::Node>>())
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::ContinuationBytes,
                })?,
            RangeSessionMode::Scan { .. } | RangeSessionMode::Done => 0,
        };
        let continuation_bytes = query_bytes
            .checked_add(automaton_bytes)
            .and_then(|bytes| bytes.checked_add(edge_bytes))
            .and_then(|bytes| bytes.checked_add(frame_bytes))
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
                if let Err(reason) = self.reserve_result_slot() {
                    return self.terminate(reason);
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
                    self.results = match self.take_finished_results() {
                        Ok(results) => results,
                        Err(reason) => return self.terminate(reason),
                    };
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
                let candidate_work = match self.query.len().checked_add(1).and_then(|rows| {
                    stored
                        .series
                        .len()
                        .checked_add(1)
                        .and_then(|columns| rows.checked_mul(columns))
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

                let Some(workspace) = self.workspace.as_mut() else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                let candidate_bound = self.index.kernel.candidate_lower_bound(
                    &self.query,
                    &stored.series,
                    workspace.plan(),
                );
                if !K::Monoid::within(candidate_bound, self.tau) {
                    continue;
                }
                let step_work = workspace.current().len().max(1);
                let exact = match workspace.score_candidate(
                    &self.index.kernel,
                    &self.query,
                    &stored.series,
                    self.tau,
                    step_work,
                ) {
                    Ok(ExactPointDecision::WithinCutoff(exact)) => exact,
                    Ok(ExactPointDecision::AboveCutoff | ExactPointDecision::NoFiniteAlignment) => {
                        continue
                    }
                    Err(reason) => return self.terminate(reason),
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
                if let Err(reason) = self.reserve_result_slot() {
                    return self.terminate(reason);
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

            let Some(workspace) = self.workspace.as_ref() else {
                return self.terminate(IncompleteReason::InvalidStoredData);
            };
            let plan = workspace.plan();
            let edge_plan = match &mut self.mode {
                RangeSessionMode::Trie {
                    traversal,
                    stack,
                    states,
                    cache,
                    column_width,
                    ..
                } => {
                    let Some(frame) = stack.last_mut() else {
                        self.mode = RangeSessionMode::Done;
                        continue;
                    };
                    let Some((unit, _)) = traversal.peek_dfs_edge(&mut frame.edges) else {
                        stack.pop().expect("last frame was observed above");
                        continue;
                    };
                    let Some(child_depth) = frame.depth.checked_add(1) else {
                        return self.terminate(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::TrieNodes,
                        });
                    };
                    let Some(state) = states.get(frame.state) else {
                        return self.terminate(IncompleteReason::InvalidStoredData);
                    };
                    debug_assert_eq!(state.depth, frame.depth);
                    let interval = self.index.bin_bounds_for(unit);
                    let cached = cache.get(frame.state, unit);
                    let (build_column, work) = if cached.is_some() {
                        (true, Some(1))
                    } else {
                        let prefix_lower_bound = self.index.kernel.prefix_lower_bound(
                            &self.query,
                            interval,
                            state.carry,
                            child_depth,
                            plan,
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
                        (build_column, work)
                    };
                    let Some(work) = work else {
                        return self.terminate(IncompleteReason::ArithmeticOverflow {
                            resource: ResourceKind::WorkUnits,
                        });
                    };
                    Some((
                        unit,
                        child_depth,
                        interval,
                        frame.state,
                        build_column,
                        work,
                        cached,
                    ))
                }
                RangeSessionMode::Scan { .. } => continue,
                RangeSessionMode::Done => continue,
            };

            let Some((unit, child_depth, interval, source, build_column, edge_work, cached)) =
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
                traversal,
                stack,
                states,
                cache,
                column_width,
            } = &mut self.mode
            else {
                return self.terminate(IncompleteReason::InvalidStoredData);
            };
            let Some(workspace) = self.workspace.as_mut() else {
                return self.terminate(IncompleteReason::InvalidStoredData);
            };
            let workspace_retained_bytes = workspace.retained_bytes();
            let plan = &workspace.plan;
            let previous = &mut workspace.current;
            let next = &mut workspace.next;
            let previous_active = &mut workspace.current_active;
            let next_active = &mut workspace.next_active;
            let Some(frame) = stack.last_mut() else {
                return self.terminate(IncompleteReason::InvalidStoredData);
            };
            let Some((consumed_unit, child)) = traversal.next_dfs_edge(&mut frame.edges) else {
                return self.terminate(IncompleteReason::InvalidStoredData);
            };
            if consumed_unit != unit {
                return self.terminate(IncompleteReason::InvalidStoredData);
            }

            if let Some(cached_target) = cached {
                let Some(state) = cached_target else {
                    continue;
                };
                let Some(target_state) = states.get(state) else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                if target_state.depth != child_depth {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                }
                let relaxed_admits = K::Monoid::within(target_state.final_cost, self.tau);
                if stack.try_reserve(1).is_err() {
                    let requested = stack.len().saturating_add(1);
                    return self.terminate(IncompleteReason::AllocationFailed {
                        resource: ResourceKind::ContinuationBytes,
                        requested,
                    });
                }
                if let Err(reason) = self.ledger.charge(ResourceKind::TrieNodes, 1) {
                    return self.terminate(reason);
                }
                let frame = match BoundedRangeFrame::open(
                    traversal,
                    child,
                    child_depth,
                    state,
                    relaxed_admits,
                ) {
                    Ok(frame) => frame,
                    Err(reason) => return self.terminate(reason),
                };
                stack.push(frame);
                if let Err(reason) = self.observe_state_peaks() {
                    return self.terminate(reason);
                }
                continue;
            }

            if !build_column {
                let _ = cache.insert(source, unit, None);
                continue;
            }
            let source_carry = {
                let Some(source_state) = states.get(source) else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                while let Some(row) = previous_active.pop() {
                    let Some(cost) = previous.get_mut(row) else {
                        return self.terminate(IncompleteReason::InvalidStoredData);
                    };
                    *cost = K::Monoid::TOP;
                }
                for position in &source_state.positions {
                    let row = match usize::try_from(position.row) {
                        Ok(row) if row < *column_width => row,
                        Ok(_) => return self.terminate(IncompleteReason::InvalidStoredData),
                        Err(_) => {
                            return self.terminate(IncompleteReason::ArithmeticOverflow {
                                resource: ResourceKind::DpCells,
                            });
                        }
                    };
                    previous[row] = position.cost;
                    previous_active.push(row);
                }
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
                            plan,
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
                source_state.carry
            };
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
                source_carry,
                child_depth,
                plan,
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
                        source_carry,
                        child_depth,
                        plan,
                        next,
                    );
                    next_active.extend(next.iter().enumerate().filter_map(|(row, cost)| {
                        K::Monoid::within(*cost, self.tau).then_some(row)
                    }));
                    (lower_bound, carry)
                }
            };
            if !K::Monoid::within(lower_bound, self.tau) {
                let _ = cache.insert(source, unit, None);
                continue;
            }
            debug_assert!(next_active.windows(2).all(|pair| pair[0] < pair[1]));
            let final_row = self.index.kernel.final_row(self.query.len());
            let final_cost = next.get(final_row).copied().unwrap_or(K::Monoid::TOP);
            let relaxed_admits = K::Monoid::within(final_cost, self.tau);

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
                        .vertical_epsilon_extension(&self.query, interval, **row, next, plan)
                        .is_none_or(|vertical| {
                            K::Monoid::compare(vertical, cost) != Ordering::Equal
                        })
                })
                .count();

            let existing_state_bytes = match states.retained_bytes() {
                Some(bytes) => bytes,
                None => {
                    return self.terminate(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    });
                }
            };
            let requested_state_bytes = match canonical_position_count
                .checked_mul(std::mem::size_of::<RangeProductPosition<K>>())
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<RangeProductState<K>>()))
                .and_then(|bytes| bytes.checked_add(existing_state_bytes))
            {
                Some(bytes) => bytes,
                None => {
                    return self.terminate(IncompleteReason::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    });
                }
            };
            let prospective_scratch_bytes =
                match requested_state_bytes.checked_add(workspace_retained_bytes) {
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
                    .vertical_epsilon_extension(&self.query, interval, row, next, plan)
                    .is_some_and(|vertical| K::Monoid::compare(vertical, cost) == Ordering::Equal);
                if K::Monoid::within(cost, self.tau) && !dominated {
                    positions.push(RangeProductPosition { row: raw_row, cost });
                }
            }
            debug_assert_eq!(positions.len(), canonical_position_count);
            if stack.try_reserve(1).is_err() {
                let requested = stack.len().saturating_add(1);
                return self.terminate(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested,
                });
            }
            let state = match states.intern(
                &self.index.kernel,
                RangeProductState {
                    depth: child_depth,
                    carry: Some(carry),
                    positions,
                    final_cost,
                },
            ) {
                Ok(state) => state,
                Err(reason) => return self.terminate(reason),
            };
            let _ = cache.insert(source, unit, Some(state));
            if let Err(reason) = self.ledger.charge(ResourceKind::TrieNodes, 1) {
                return self.terminate(reason);
            }
            let frame =
                match BoundedRangeFrame::open(traversal, child, child_depth, state, relaxed_admits)
                {
                    Ok(frame) => frame,
                    Err(reason) => return self.terminate(reason),
                };
            stack.push(frame);
            if let Err(reason) = self.observe_state_peaks() {
                return self.terminate(reason);
            }
        }
    }
}

impl<'a, V, D> ErpAutomatonRangeContinuation<'a, V, D>
where
    V: Eq + std::hash::Hash + Clone,
    D: ElasticDictionaryBackend<Label = u8>,
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

    /// Cancel this query explicitly without claiming its partial exact set is
    /// a complete range result.
    pub fn cancel(mut self) -> OperationOutcome<Vec<(V, f64)>, Self> {
        match self.take_finished_results() {
            Ok(partial) => OperationOutcome::Incomplete {
                partial: Some(partial),
                reason: IncompleteReason::Cancelled,
                continuation: None,
                usage: self.ledger.usage(),
            },
            Err(reason) => OperationOutcome::Incomplete {
                partial: None,
                reason,
                continuation: None,
                usage: self.ledger.usage(),
            },
        }
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
                if continuation.is_some() {
                    OperationOutcome::Incomplete {
                        partial: None,
                        reason,
                        continuation: Some(self),
                        usage,
                    }
                } else {
                    match self.take_finished_results() {
                        Ok(partial) => OperationOutcome::Incomplete {
                            partial: Some(partial),
                            reason,
                            continuation: None,
                            usage,
                        },
                        Err(finalization_reason) => OperationOutcome::Incomplete {
                            partial: None,
                            reason: finalization_reason,
                            continuation: None,
                            usage,
                        },
                    }
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

    fn reserve_result_slot(&mut self) -> Result<(), IncompleteReason> {
        if self.results.len() < self.results.capacity() {
            return Ok(());
        }
        let requested =
            self.results
                .len()
                .checked_add(1)
                .ok_or(IncompleteReason::ArithmeticOverflow {
                    resource: ResourceKind::Results,
                })?;
        self.results
            .try_reserve_exact(1)
            .map_err(|_| IncompleteReason::AllocationFailed {
                resource: ResourceKind::Results,
                requested,
            })
    }

    fn take_finished_results(&mut self) -> Result<Vec<(V, f64)>, IncompleteReason> {
        let retained_scratch = self.machine.retained_scratch_bytes()?;
        ElasticTransducer::<ErpConfig, V, D>::try_finish_bounded_range_results(
            std::mem::take(&mut self.results),
            retained_scratch,
            &mut self.ledger,
        )
    }

    fn observe_state_peaks(&mut self) -> Result<(), IncompleteReason> {
        self.ledger
            .observe_peak(ResourceKind::QueueEntries, self.stack.len())?;
        let machine_bytes = self.machine.retained_scratch_bytes()?;
        self.ledger
            .observe_peak(ResourceKind::ScratchBytes, machine_bytes)?;

        let stack_bytes = self
            .stack
            .capacity()
            .checked_mul(std::mem::size_of::<ErpAutomatonRangeFrame<D::Node>>())
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
            .capacity()
            .checked_mul(std::mem::size_of::<(V, f64)>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ContinuationBytes,
            })?;
        let continuation_bytes = machine_bytes
            .checked_add(stack_bytes)
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
                if let Err(reason) = self.reserve_result_slot() {
                    self.pending_match = Some(pending);
                    return self.terminate(reason);
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
                self.results = match self.take_finished_results() {
                    Ok(results) => results,
                    Err(reason) => return self.terminate(reason),
                };
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
                let candidate_work = match self
                    .query
                    .len()
                    .checked_add(1)
                    .and_then(|width| width.checked_mul(2))
                    .and_then(|per_step| per_step.checked_mul(stored.series.len()))
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
                let exact = match self.machine.score_candidate(&stored.series) {
                    Ok(Some(exact)) => exact,
                    Ok(None) => continue,
                    Err(reason) => return self.terminate(reason),
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
                if let Err(reason) = self.reserve_result_slot() {
                    return self.terminate(reason);
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

            let (unit, child_depth, source) = {
                let Some(frame) = self.stack.last_mut() else {
                    self.done = true;
                    continue;
                };
                let Some((unit, _)) = self.traversal.peek_dfs_edge(&mut frame.edges) else {
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
                (unit, child_depth, frame.state)
            };
            if child_depth > self.ledger.limits().max_series_len {
                return self.terminate(IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::SeriesLength,
                    limit: self.ledger.limits().max_series_len,
                    requested: child_depth,
                });
            }
            let edge_work = match self.machine.transition_work_bound(source) {
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
            let child = {
                let Some(frame) = self.stack.last_mut() else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                let Some((consumed_unit, child)) = self.traversal.next_dfs_edge(&mut frame.edges)
                else {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                };
                if consumed_unit != unit {
                    return self.terminate(IncompleteReason::InvalidStoredData);
                }
                child
            };

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
            if self.stack.len() == self.stack.capacity() && self.stack.try_reserve_exact(1).is_err()
            {
                return self.terminate(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested: self.stack.len().saturating_add(1),
                });
            }
            let frame = match ErpAutomatonRangeFrame::open(
                &mut self.traversal,
                child,
                child_depth,
                target,
                final_cost.is_some(),
            ) {
                Ok(frame) => frame,
                Err(reason) => return self.terminate(reason),
            };
            self.stack.push(frame);
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
    /// Create an empty transducer backed by the default in-memory byte DAWG.
    pub fn new<C>(quant: QuantizationConfig, kernel: C) -> Self
    where
        C: Into<K>,
    {
        Self::with_dictionary(quant, kernel, DynamicDawg::new())
    }
}

impl<K, V, D> ElasticTransducer<K, V, D>
where
    K: ElasticKernel,
    V: Eq + std::hash::Hash + Clone,
    D: ElasticDictionaryBackend<Label = u8>,
{
    fn with_dictionary<C>(quant: QuantizationConfig, kernel: C, dawg: D) -> Self
    where
        C: Into<K>,
    {
        let quant = quant.into_u8_compatible();
        let kernel = kernel.into().normalized();
        let bin_bounds = (0..quant.num_bins)
            .map(|bin| quant.bin_bounds(bin))
            .collect();
        Self {
            dawg,
            quant,
            kernel,
            bin_bounds,
            buckets: Vec::new(),
            originals: HashMap::new(),
            snapshot_identity: None,
        }
    }

    /// Insert a reference series under identifier `value`.
    ///
    /// Returns `true` if `value` was not previously present. For a fail-closed
    /// result from a fallible persistent backend, use [`Self::try_insert`]. A
    /// failed call through this compatibility surface returns `false` and
    /// leaves all logical state and the verified snapshot identity unchanged.
    pub fn insert(&mut self, value: V, series: &[f64]) -> bool
    where
        D: ElasticMutableDictionaryBackend,
    {
        self.try_insert(value, series).unwrap_or(false)
    }

    /// Transactionally insert or replace one full-precision reference series.
    ///
    /// All fallible heap reservations and value clones happen before the
    /// dictionary mutation. If the backend rejects that mutation, the
    /// dictionary, collision buckets, originals, and snapshot identity remain
    /// observably unchanged. Once a missing dictionary key is durably inserted,
    /// the remaining commit uses only capacity reserved in the prepare phase.
    ///
    /// Returns `Ok(true)` iff `value` was not previously present.
    pub fn try_insert(&mut self, value: V, series: &[f64]) -> Result<bool, ElasticMutationError>
    where
        D: ElasticMutableDictionaryBackend,
    {
        let mut key = Vec::new();
        key.try_reserve_exact(series.len()).map_err(|_| {
            ElasticMutationError::AllocationFailed {
                requested: series.len(),
            }
        })?;
        key.extend(series.iter().map(|sample| self.quant.quantize_u8(*sample)));

        let mut replacement_series = Vec::new();
        replacement_series
            .try_reserve_exact(series.len())
            .map_err(|_| ElasticMutationError::AllocationFailed {
                requested: series.len().saturating_mul(std::mem::size_of::<f64>()),
            })?;
        replacement_series.extend_from_slice(series);

        let old_location = self
            .originals
            .get(&value)
            .map(|stored| stored.bucket_location);
        let is_new = old_location.is_none();
        if is_new {
            self.originals
                .try_reserve(1)
                .map_err(|_| ElasticMutationError::AllocationFailed {
                    requested: self
                        .originals
                        .len()
                        .saturating_add(1)
                        .saturating_mul(std::mem::size_of::<(V, StoredSeries)>()),
                })?;
        }

        // Clone every identifier that the commit can need before any state is
        // changed. This includes the element swap_remove will relocate.
        let displaced = match old_location {
            Some((old_bucket, old_slot)) => {
                let bucket = self
                    .buckets
                    .get(old_bucket)
                    .ok_or(ElasticMutationError::InvalidState)?;
                if bucket.get(old_slot) != Some(&value) {
                    return Err(ElasticMutationError::InvalidState);
                }
                if old_slot + 1 < bucket.len() {
                    let last_slot = bucket.len() - 1;
                    let displaced = bucket[last_slot].clone();
                    if self
                        .originals
                        .get(&displaced)
                        .map(|stored| stored.bucket_location)
                        != Some((old_bucket, last_slot))
                    {
                        return Err(ElasticMutationError::InvalidState);
                    }
                    Some(displaced)
                } else {
                    None
                }
            }
            None => None,
        };

        let existing_bucket = self.dawg.elastic_bucket(&key);
        let mut prepared_bucket = None;
        let bucket_id = match existing_bucket {
            Some(bucket_id) => {
                let bucket = self
                    .buckets
                    .get_mut(bucket_id)
                    .ok_or(ElasticMutationError::InvalidState)?;
                if old_location.is_none_or(|(old_bucket, _)| old_bucket != bucket_id) {
                    bucket.try_reserve_exact(1).map_err(|_| {
                        ElasticMutationError::AllocationFailed {
                            requested: bucket
                                .len()
                                .saturating_add(1)
                                .saturating_mul(std::mem::size_of::<V>()),
                        }
                    })?;
                }
                bucket_id
            }
            None => {
                let bucket_id = self.buckets.len();
                self.buckets.try_reserve_exact(1).map_err(|_| {
                    ElasticMutationError::AllocationFailed {
                        requested: self
                            .buckets
                            .len()
                            .saturating_add(1)
                            .saturating_mul(std::mem::size_of::<Vec<V>>()),
                    }
                })?;
                let mut bucket = Vec::new();
                bucket.try_reserve_exact(1).map_err(|_| {
                    ElasticMutationError::AllocationFailed {
                        requested: std::mem::size_of::<V>(),
                    }
                })?;
                bucket.push(value.clone());
                prepared_bucket = Some(bucket);
                bucket_id
            }
        };

        let target_member = if existing_bucket.is_some()
            && old_location.is_none_or(|(old_bucket, _)| old_bucket != bucket_id)
        {
            Some(value.clone())
        } else {
            None
        };

        // This is the only fallible external mutation. Every subsequent write
        // consumes capacity and clones prepared above.
        if existing_bucket.is_none() && !self.dawg.elastic_try_insert_bucket(&key, bucket_id)? {
            return Err(ElasticMutationError::DictionaryConflict);
        }

        if let Some(bucket) = prepared_bucket {
            self.buckets.push(bucket);
        }

        if let Some((old_bucket, old_slot)) = old_location {
            if old_bucket != bucket_id {
                let became_empty = {
                    let bucket = &mut self.buckets[old_bucket];
                    let removed = bucket.swap_remove(old_slot);
                    debug_assert!(removed == value);
                    bucket.is_empty()
                };
                if let Some(displaced) = displaced {
                    self.originals
                        .get_mut(&displaced)
                        .expect("displaced member validated during prepare")
                        .bucket_location = (old_bucket, old_slot);
                }
                if became_empty {
                    self.release_empty_bucket_storage(old_bucket);
                }
            }
        }

        let bucket_location = match old_location {
            Some((old_bucket, old_slot)) if old_bucket == bucket_id => (bucket_id, old_slot),
            _ if existing_bucket.is_none() => (bucket_id, 0),
            _ => {
                let bucket = &mut self.buckets[bucket_id];
                let slot = bucket.len();
                bucket.push(target_member.expect("prepared target member"));
                (bucket_id, slot)
            }
        };
        self.originals.insert(
            value,
            StoredSeries {
                series: replacement_series,
                bucket_location,
            },
        );

        // A successful content mutation detaches the live index from its
        // immutable verified generation. Error paths above retain the binding.
        self.snapshot_identity = None;
        Ok(is_new)
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

    /// Verified complete-snapshot identity, or `None` for live/unsealed state.
    ///
    /// Any successful content mutation clears this binding, so evidence cannot
    /// silently retain a stale generation identity. Failed transactional
    /// mutations leave it unchanged.
    #[inline]
    pub fn snapshot_identity(&self) -> Option<ElasticSnapshotIdentity> {
        self.snapshot_identity
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
        self.snapshot_identity = None;

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
        let root = self.dawg.elastic_root();
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
    /// cutoff. A paused outcome carries its exact subset only once, inside the
    /// continuation, where [`RangeContinuation::exact_partial`] can borrow it.
    pub fn search_range_bounded(
        &self,
        query: &[f64],
        tau: Cost<K>,
        limits: ResourceLimits,
        page: PageBudget,
    ) -> Result<BoundedRangeOutcome<'_, K, V, D>, TemporalValidationError> {
        if !self.kernel.cutoff_is_valid(tau) {
            return Err(TemporalValidationError::InvalidCutoff);
        }
        let mut ledger = ResourceLedger::new(limits);
        ledger.validate_finite_series(Operand::Query, query)?;

        let mut query_storage = Vec::new();
        if query_storage.try_reserve_exact(query.len()).is_err() {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason: IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested: query.len().saturating_mul(std::mem::size_of::<f64>()),
                },
                continuation: None,
                usage: ledger.usage(),
            });
        }
        query_storage.extend_from_slice(query);

        let (mut workspace, mut construction_terminal) = if self.is_empty() {
            (None, None)
        } else {
            match ExactPointWorkspace::try_new(&self.kernel, query, limits.max_scratch_bytes) {
                Ok(workspace) => {
                    let terminal = ledger
                        .observe_peak(
                            ResourceKind::ScratchBytes,
                            workspace.construction_peak_bytes(),
                        )
                        .err();
                    (Some(workspace), terminal)
                }
                Err(reason) => (None, Some(reason)),
            }
        };

        let (mode, mode_terminal) = if construction_terminal.is_some() {
            (RangeSessionMode::Done, construction_terminal)
        } else if self.is_empty() {
            (RangeSessionMode::Done, None)
        } else if query.is_empty() || !self.kernel.supports_interval_query(query) {
            (RangeSessionMode::Scan { bucket: 0, slot: 0 }, None)
        } else if let Some(column_width) = self.kernel.column_len(query.len()) {
            let root = self.dawg.elastic_root();
            let (mut traversal, root_cursor) =
                TraversalSession::capture_mapped(DictionaryTraversalRoot::owned(root));
            let mut terminal = (!traversal.supports_efficient_dfs_edge_paging())
                .then_some(IncompleteReason::Unsupported);
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
            let workspace_retained = workspace
                .as_ref()
                .map_or(0, ExactPointWorkspace::retained_bytes);
            let remaining_scratch = limits.max_scratch_bytes.saturating_sub(workspace_retained);
            let arena_bytes = remaining_scratch.saturating_mul(3) / 4;
            let cache_bytes = remaining_scratch.saturating_sub(arena_bytes);
            let arena_header_bytes = arena_bytes / 2;
            let arena_position_bytes = arena_bytes.saturating_sub(arena_header_bytes);
            let bytes_per_state = std::mem::size_of::<RangeProductState<K>>()
                .saturating_add(std::mem::size_of::<(u64, SmallVec<[TemporalStateId; 2]>)>())
                .max(1);
            let bytes_per_position = std::mem::size_of::<RangeProductPosition<K>>().max(1);
            let bytes_per_cache_entry =
                std::mem::size_of::<((TemporalStateId, u8), Option<TemporalStateId>)>()
                    .saturating_add(std::mem::size_of::<(TemporalStateId, u8)>())
                    .max(1);
            let max_states = limits
                .max_trie_nodes
                .min(arena_header_bytes / bytes_per_state)
                .min(u32::MAX as usize);
            let max_positions = arena_position_bytes / bytes_per_position;
            let max_cache_entries = limits
                .max_trie_edges
                .min(cache_bytes / bytes_per_cache_entry);
            let mut stack = Vec::new();
            let mut states = RangeProductStateArena::new(max_states, max_positions);
            let cache = BoundedTransitionCache::new(max_cache_entries);
            if terminal.is_none() && max_states == 0 {
                terminal = Some(IncompleteReason::BudgetExceeded {
                    resource: ResourceKind::ScratchBytes,
                    limit: limits.max_scratch_bytes,
                    requested: workspace_retained.saturating_add(bytes_per_state),
                });
            }
            if terminal.is_none() && stack.try_reserve_exact(1).is_err() {
                terminal = Some(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested: std::mem::size_of::<BoundedRangeFrame<D::Node>>(),
                });
            }
            if terminal.is_none() {
                match states.intern(
                    &self.kernel,
                    RangeProductState {
                        depth: 0,
                        carry: None,
                        positions: Vec::new(),
                        final_cost: self.kernel.empty_vs_nonempty_cost(query),
                    },
                ) {
                    Ok(root_state) => {
                        let terminal_admitted = states
                            .get(root_state)
                            .is_some_and(|state| K::Monoid::within(state.final_cost, tau));
                        match BoundedRangeFrame::open(
                            &mut traversal,
                            root_cursor,
                            0,
                            root_state,
                            terminal_admitted,
                        ) {
                            Ok(frame) => stack.push(frame),
                            Err(reason) => terminal = Some(reason),
                        }
                    }
                    Err(reason) => terminal = Some(reason),
                }
            }
            (
                RangeSessionMode::Trie {
                    traversal,
                    stack,
                    states,
                    cache,
                    column_width,
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

        construction_terminal = mode_terminal;
        let result_capacity = self
            .len()
            .min(DEFAULT_RESULT_BUFFER_CAPACITY)
            .min(limits.max_results);
        let mut results = Vec::new();
        if construction_terminal.is_none() && results.try_reserve_exact(result_capacity).is_err() {
            construction_terminal = Some(IncompleteReason::AllocationFailed {
                resource: ResourceKind::Results,
                requested: result_capacity,
            });
        }
        let mut continuation = RangeContinuation {
            index: self,
            query: query_storage,
            workspace: workspace.take(),
            tau,
            mode,
            results,
            pending_match: None,
            ledger,
            terminal: construction_terminal,
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

        let mut workspace =
            match ExactPointWorkspace::try_new(&self.kernel, query, limits.max_scratch_bytes) {
                Ok(workspace) => workspace,
                Err(reason) => {
                    return Ok(OperationOutcome::Incomplete {
                        partial: None,
                        reason,
                        continuation: None,
                        usage: ledger.usage(),
                    });
                }
            };
        if let Err(reason) = ledger.observe_peak(
            ResourceKind::ScratchBytes,
            workspace.construction_peak_bytes(),
        ) {
            return Ok(OperationOutcome::Incomplete {
                partial: None,
                reason,
                continuation: None,
                usage: ledger.usage(),
            });
        }
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
            let lower_bound =
                self.kernel
                    .candidate_lower_bound(query, &stored.series, workspace.plan());
            if !K::Monoid::within(lower_bound, cutoff) {
                continue;
            }
            let step_work = workspace.current().len().max(1);
            let exact = match workspace.score_candidate(
                &self.kernel,
                query,
                &stored.series,
                cutoff,
                step_work,
            ) {
                Ok(ExactPointDecision::WithinCutoff(exact)) => exact,
                Ok(ExactPointDecision::AboveCutoff | ExactPointDecision::NoFiniteAlignment) => {
                    continue
                }
                Err(reason) => {
                    return Ok(OperationOutcome::Incomplete {
                        partial: None,
                        reason,
                        continuation: None,
                        usage: ledger.usage(),
                    });
                }
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
        match Self::try_finish_bounded_knn_results(best, workspace.retained_bytes(), &mut ledger) {
            Ok(value) => Ok(OperationOutcome::Complete {
                value,
                usage: ledger.usage(),
            }),
            Err(reason) => Ok(OperationOutcome::Incomplete {
                partial: None,
                reason,
                continuation: None,
                usage: ledger.usage(),
            }),
        }
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
    /// one bucket slot (see transactional insertion and `remove_from_bucket`), so this
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
            node: self.dawg.elastic_root(),
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

    /// Finish a bounded range result with one explicit permutation workspace.
    ///
    /// Bounded traversal visits each stored episode id exactly once: the
    /// private `originals` map owns unique ids and every live bucket entry is
    /// tied to that id's single `bucket_location`. Consequently the defensive
    /// duplicate coalescing required by the legacy convenience walker is not
    /// needed here. Original vector positions are the encounter sequence and
    /// therefore a total tie key. The single `(old, destination)` permutation
    /// is sorted without hidden allocation, charged together with the live
    /// scorer state, and then applied to the existing payload vector by swaps.
    fn try_finish_bounded_range_results(
        mut results: Vec<(V, Cost<K>)>,
        retained_scratch: usize,
        ledger: &mut ResourceLedger,
    ) -> Result<Vec<(V, Cost<K>)>, IncompleteReason> {
        let len = results.len();
        let permutation_bytes = len
            .checked_mul(std::mem::size_of::<(usize, usize)>())
            .ok_or(IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            })?;
        let scratch_peak = retained_scratch.checked_add(permutation_bytes).ok_or(
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            },
        )?;
        ledger.observe_peak(ResourceKind::ScratchBytes, scratch_peak)?;

        let mut permutation = Vec::new();
        permutation
            .try_reserve_exact(len)
            .map_err(|_| IncompleteReason::AllocationFailed {
                resource: ResourceKind::ScratchBytes,
                requested: permutation_bytes,
            })?;
        permutation.extend((0..len).map(|old| (old, 0_usize)));
        permutation.sort_unstable_by(|left, right| {
            K::Monoid::compare(results[left.0].1, results[right.0].1)
                .then_with(|| left.0.cmp(&right.0))
        });
        for (destination, entry) in permutation.iter_mut().enumerate() {
            entry.1 = destination;
        }
        permutation.sort_unstable_by_key(|entry| entry.0);
        for position in 0..len {
            let mut swaps = 0_usize;
            while permutation[position].1 != position {
                if swaps == len {
                    return Err(IncompleteReason::InvalidStoredData);
                }
                let destination = permutation[position].1;
                results.swap(position, destination);
                permutation.swap(position, destination);
                swaps += 1;
            }
        }
        Ok(results)
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

    fn try_finish_bounded_knn_results(
        best: BinaryHeap<KnnBestResult<K, V>>,
        retained_scratch: usize,
        ledger: &mut ResourceLedger,
    ) -> Result<Vec<(V, Cost<K>)>, IncompleteReason> {
        let mut best = best.into_vec();
        best.sort_unstable_by(|left, right| {
            K::Monoid::compare(left.distance, right.distance)
                .then_with(|| left.sequence.cmp(&right.sequence))
        });
        let len = best.len();
        let output_bytes = len.checked_mul(std::mem::size_of::<(V, Cost<K>)>()).ok_or(
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            },
        )?;
        let scratch_peak = retained_scratch.checked_add(output_bytes).ok_or(
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            },
        )?;
        ledger.observe_peak(ResourceKind::ScratchBytes, scratch_peak)?;

        let mut finished = Vec::new();
        finished
            .try_reserve_exact(len)
            .map_err(|_| IncompleteReason::AllocationFailed {
                resource: ResourceKind::Results,
                requested: len,
            })?;
        finished.extend(best.into_iter().map(|entry| (entry.value, entry.distance)));
        Ok(finished)
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

impl<K, D> ElasticTransducer<K, u64, D>
where
    K: ElasticKernel,
    Cost<K>: PartialEq,
    D: ElasticDictionaryBackend<Label = u8>,
{
    /// Exact range search with deterministic, bounded K1--K4 evidence.
    ///
    /// Failure to retain the complete evidence stream fails closed: no partial
    /// certificate is returned. Edges and collision members are ordered by byte
    /// label and stable id, so the evidence is invariant under hash seeding and
    /// remains byte-for-byte equal after complete-snapshot reload.
    pub fn search_range_with_certificate(
        &self,
        query: &[f64],
        tau: Cost<K>,
        limits: ElasticCertificateLimits,
    ) -> Result<CertifiedRangeResults<K>, ElasticCertificateError> {
        if !self.kernel.cutoff_is_valid(tau) {
            return Err(TemporalValidationError::InvalidCutoff.into());
        }
        let mut ledger = ResourceLedger::new(limits.resources);
        ledger.validate_finite_series(Operand::Query, query)?;
        if query.is_empty() || !self.kernel.supports_interval_query(query) {
            return Err(ElasticCertificateError::Unsupported);
        }
        let column_width = self
            .kernel
            .column_len(query.len())
            .ok_or(ElasticCertificateError::Unsupported)?;
        if column_width > u32::MAX as usize {
            return Err(ElasticCertificateError::BudgetExceeded {
                resource: ResourceKind::DpCells,
                limit: u32::MAX as usize,
                requested: column_width,
            });
        }
        let query_bits_bytes = query.len().checked_mul(std::mem::size_of::<u64>()).ok_or(
            ElasticCertificateError::ArithmeticOverflow {
                resource: ResourceKind::WitnessBytes,
            },
        )?;
        if query_bits_bytes > limits.resources.max_witness_bytes {
            return Err(ElasticCertificateError::BudgetExceeded {
                resource: ResourceKind::WitnessBytes,
                limit: limits.resources.max_witness_bytes,
                requested: query_bits_bytes,
            });
        }
        let mut query_bits = Vec::new();
        query_bits.try_reserve_exact(query.len()).map_err(|_| {
            ElasticCertificateError::AllocationFailed {
                resource: ResourceKind::WitnessBytes,
                requested: query.len().saturating_mul(std::mem::size_of::<u64>()),
            }
        })?;
        query_bits.extend(query.iter().map(|value| value.to_bits()));
        let mut builder = CertificateBuilder::new(limits, query_bits_bytes)?;
        // The same fallible plan and two exact frontier generations drive both
        // interval-product transitions and K3 full-precision survivor checks.
        // Candidate verification resets them in place and never allocates.
        let mut workspace =
            ExactPointWorkspace::try_new(&self.kernel, query, limits.resources.max_scratch_bytes)?;
        ledger.observe_peak(
            ResourceKind::ScratchBytes,
            workspace.construction_peak_bytes(),
        )?;
        let fixed_scratch_bytes = workspace.retained_bytes();

        let remaining_scratch = limits
            .resources
            .max_scratch_bytes
            .saturating_sub(fixed_scratch_bytes);
        let bytes_per_state = std::mem::size_of::<RangeProductState<K>>()
            .saturating_add(std::mem::size_of::<(u64, SmallVec<[TemporalStateId; 2]>)>())
            .max(1);
        let bytes_per_position = std::mem::size_of::<RangeProductPosition<K>>().max(1);
        let max_states = limits
            .resources
            .max_trie_nodes
            .min((remaining_scratch / 2) / bytes_per_state)
            .min(u32::MAX as usize);
        let max_positions =
            (remaining_scratch.saturating_sub(remaining_scratch / 2)) / bytes_per_position;
        if max_states == 0 {
            return Err(ElasticCertificateError::BudgetExceeded {
                resource: ResourceKind::ScratchBytes,
                limit: limits.resources.max_scratch_bytes,
                requested: fixed_scratch_bytes.saturating_add(bytes_per_state),
            });
        }
        let mut states = RangeProductStateArena::new(max_states, max_positions);
        let root_state = states.intern(
            &self.kernel,
            RangeProductState {
                depth: 0,
                carry: None,
                positions: Vec::new(),
                final_cost: self.kernel.empty_vs_nonempty_cost(query),
            },
        )?;

        let root = self.dawg.elastic_root();
        let (mut traversal, root_cursor) =
            TraversalSession::capture_mapped(DictionaryTraversalRoot::owned(root));
        if !traversal.supports_efficient_dfs_edge_paging() {
            return Err(ElasticCertificateError::Unsupported);
        }
        ledger.charge(ResourceKind::TrieNodes, 1)?;
        let root_final_value = traversal.final_value_at_cursor(root_cursor, None);
        let root_edges = traversal.open_dfs_node(root_cursor);
        let mut root_candidates = Vec::new();
        let mut root_candidate_bucket = None;
        if root_edges.is_final() {
            let bucket_id = root_final_value.ok_or(ElasticCertificateError::InvalidStoredData)?;
            let root_admitted = states
                .get(root_state)
                .is_some_and(|state| K::Monoid::within(state.final_cost, tau));
            if root_admitted {
                let bucket = self
                    .buckets
                    .get(bucket_id)
                    .ok_or(ElasticCertificateError::InvalidStoredData)?;
                if bucket.is_empty() {
                    return Err(ElasticCertificateError::InvalidStoredData);
                }
                if bucket.len() > limits.resources.max_queue_entries {
                    return Err(ElasticCertificateError::BudgetExceeded {
                        resource: ResourceKind::QueueEntries,
                        limit: limits.resources.max_queue_entries,
                        requested: bucket.len(),
                    });
                }
                root_candidates
                    .try_reserve_exact(bucket.len())
                    .map_err(|_| ElasticCertificateError::AllocationFailed {
                        resource: ResourceKind::ScratchBytes,
                        requested: bucket.len().saturating_mul(std::mem::size_of::<u64>()),
                    })?;
                root_candidates.extend(bucket.iter().copied());
                root_candidates.sort_unstable();
                if root_candidates.windows(2).any(|pair| pair[0] == pair[1]) {
                    return Err(ElasticCertificateError::InvalidStoredData);
                }
                root_candidate_bucket = Some(bucket_id);
            } else {
                let terminal_bound = states
                    .get(root_state)
                    .map(|state| state.final_cost)
                    .ok_or(ElasticCertificateError::InvalidStoredData)?;
                builder.record(&[], |quantized_path| ElasticRangeEvidence::TerminalPruned {
                    quantized_path,
                    lower_bound: terminal_bound,
                })?;
            }
        } else if root_final_value.is_some() {
            return Err(ElasticCertificateError::InvalidStoredData);
        }
        let mut stack: Vec<CertifiedRangeFrame<D::Node>> = Vec::new();
        stack
            .try_reserve_exact(1)
            .map_err(|_| ElasticCertificateError::AllocationFailed {
                resource: ResourceKind::ContinuationBytes,
                requested: std::mem::size_of::<CertifiedRangeFrame<D::Node>>(),
            })?;
        stack.push(CertifiedRangeFrame {
            depth: 0,
            state: root_state,
            candidate_bucket: root_candidate_bucket,
            candidates: root_candidates,
            next_candidate: 0,
            edges: root_edges,
        });
        let mut path = Vec::<u8>::new();
        let mut results = Vec::new();
        let initial_result_capacity = self
            .len()
            .min(DEFAULT_RESULT_BUFFER_CAPACITY)
            .min(limits.resources.max_results);
        results.try_reserve(initial_result_capacity).map_err(|_| {
            ElasticCertificateError::AllocationFailed {
                resource: ResourceKind::Results,
                requested: initial_result_capacity
                    .saturating_mul(std::mem::size_of::<(u64, Cost<K>)>()),
            }
        })?;

        let initial_state_bytes =
            states
                .retained_bytes()
                .ok_or(ElasticCertificateError::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                })?;
        ledger.observe_peak(
            ResourceKind::ScratchBytes,
            fixed_scratch_bytes.checked_add(initial_state_bytes).ok_or(
                ElasticCertificateError::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                },
            )?,
        )?;
        ledger.observe_peak(
            ResourceKind::QueueEntries,
            stack
                .len()
                .saturating_add(stack.last().map_or(0, |frame| frame.candidates.len())),
        )?;

        while !stack.is_empty() {
            let candidate = {
                let frame = stack.last_mut().expect("nonempty checked");
                let candidate = frame
                    .candidates
                    .get(frame.next_candidate)
                    .copied()
                    .zip(frame.candidate_bucket);
                if candidate.is_some() {
                    frame.next_candidate = frame.next_candidate.saturating_add(1);
                } else if !frame.candidates.is_empty() {
                    // Drop the collision view before traversing children; no
                    // exhausted bucket allocation is retained down the path.
                    frame.candidates = Vec::new();
                    frame.candidate_bucket = None;
                    frame.next_candidate = 0;
                }
                candidate
            };
            if let Some((stable_id, bucket_id)) = candidate {
                let Some(stored) = self.originals.get(&stable_id) else {
                    return Err(ElasticCertificateError::InvalidStoredData);
                };
                let (stored_bucket, stored_slot) = stored.bucket_location;
                if stored_bucket != bucket_id
                    || self
                        .buckets
                        .get(stored_bucket)
                        .and_then(|bucket| bucket.get(stored_slot))
                        != Some(&stable_id)
                    || stored.series.len() != path.len()
                    || stored
                        .series
                        .iter()
                        .zip(&path)
                        .any(|(sample, unit)| self.quant.quantize(*sample) as u8 != *unit)
                {
                    return Err(ElasticCertificateError::InvalidStoredData);
                }
                if stored.series.len() > limits.resources.max_series_len
                    || stored.series.iter().any(|sample| !sample.is_finite())
                {
                    return Err(ElasticCertificateError::InvalidStoredData);
                }
                let candidate_work = query
                    .len()
                    .checked_add(1)
                    .and_then(|rows| {
                        stored
                            .series
                            .len()
                            .checked_add(1)
                            .and_then(|columns| rows.checked_mul(columns))
                    })
                    .ok_or(ElasticCertificateError::ArithmeticOverflow {
                        resource: ResourceKind::WorkUnits,
                    })?;
                builder.charge_work(candidate_work)?;
                ledger.charge_many(&[
                    (ResourceKind::Candidates, 1),
                    (ResourceKind::DpCells, candidate_work),
                    (ResourceKind::WorkUnits, candidate_work),
                ])?;
                let candidate_bound =
                    self.kernel
                        .candidate_lower_bound(query, &stored.series, workspace.plan());
                if !K::Monoid::within(candidate_bound, tau) {
                    builder.record(&path, |quantized_path| {
                        ElasticRangeEvidence::CandidatePruned {
                            quantized_path,
                            stable_id,
                            candidate_bound,
                        }
                    })?;
                    continue;
                }
                let step_work = workspace.current().len().max(1);
                let exact = match workspace.score_candidate(
                    &self.kernel,
                    query,
                    &stored.series,
                    tau,
                    step_work,
                )? {
                    ExactPointDecision::WithinCutoff(exact) => Some(exact),
                    ExactPointDecision::AboveCutoff | ExactPointDecision::NoFiniteAlignment => None,
                };
                let survived = exact.is_some();
                builder.record(&path, |quantized_path| {
                    ElasticRangeEvidence::ExactCandidate {
                        quantized_path,
                        stable_id,
                        candidate_bound,
                        exact,
                        survived,
                    }
                })?;
                if let Some(exact) = exact.filter(|cost| K::Monoid::within(*cost, tau)) {
                    let requested_results = results.len().checked_add(1).ok_or(
                        ElasticCertificateError::ArithmeticOverflow {
                            resource: ResourceKind::Results,
                        },
                    )?;
                    if requested_results > limits.resources.max_results {
                        return Err(ElasticCertificateError::BudgetExceeded {
                            resource: ResourceKind::Results,
                            limit: limits.resources.max_results,
                            requested: requested_results,
                        });
                    }
                    ledger.charge(ResourceKind::Results, 1)?;
                    results.try_reserve_exact(1).map_err(|_| {
                        ElasticCertificateError::AllocationFailed {
                            resource: ResourceKind::Results,
                            requested: requested_results,
                        }
                    })?;
                    results.push((stable_id, exact));
                }
                continue;
            }

            if stack.last().expect("nonempty checked").edges.remaining() == 0 {
                let depth = stack.last().expect("nonempty checked").depth;
                stack.pop();
                if depth > 0 && path.pop().is_none() {
                    return Err(ElasticCertificateError::InvalidStoredData);
                }
                continue;
            }
            let (unit, child_depth, source) = {
                let frame = stack.last_mut().expect("nonempty checked");
                let Some((unit, _)) = traversal.peek_dfs_edge(&mut frame.edges) else {
                    return Err(ElasticCertificateError::InvalidStoredData);
                };
                let child_depth = frame.depth.checked_add(1).ok_or(
                    ElasticCertificateError::ArithmeticOverflow {
                        resource: ResourceKind::TrieNodes,
                    },
                )?;
                (unit, child_depth, frame.state)
            };
            let requested_path =
                path.len()
                    .checked_add(1)
                    .ok_or(ElasticCertificateError::ArithmeticOverflow {
                        resource: ResourceKind::ContinuationBytes,
                    })?;
            if requested_path > limits.resources.max_series_len {
                return Err(ElasticCertificateError::BudgetExceeded {
                    resource: ResourceKind::SeriesLength,
                    limit: limits.resources.max_series_len,
                    requested: requested_path,
                });
            }
            if requested_path > limits.resources.max_continuation_bytes {
                return Err(ElasticCertificateError::BudgetExceeded {
                    resource: ResourceKind::ContinuationBytes,
                    limit: limits.resources.max_continuation_bytes,
                    requested: requested_path,
                });
            }
            if path.len() == path.capacity() {
                path.try_reserve_exact(1).map_err(|_| {
                    ElasticCertificateError::AllocationFailed {
                        resource: ResourceKind::ContinuationBytes,
                        requested: requested_path,
                    }
                })?;
            }
            let interval = self.bin_bounds_for(unit);
            let plan = &workspace.plan;
            let previous = &mut workspace.current;
            let next = &mut workspace.next;
            let previous_active = &mut workspace.current_active;
            let next_active = &mut workspace.next_active;
            let source_state = states
                .get(source)
                .ok_or(ElasticCertificateError::InvalidStoredData)?;
            if source_state.depth.checked_add(1) != Some(child_depth) {
                return Err(ElasticCertificateError::InvalidStoredData);
            }
            let prefix_bound = self.kernel.prefix_lower_bound(
                query,
                interval,
                source_state.carry,
                child_depth,
                plan,
            );
            let build_column = K::Monoid::within(prefix_bound, tau);
            let transition_bound = if source_state.positions.is_empty() {
                column_width
            } else {
                column_width
                    .checked_mul(2)
                    .ok_or(ElasticCertificateError::ArithmeticOverflow {
                        resource: ResourceKind::WorkUnits,
                    })?
            };
            let edge_work = (if build_column { transition_bound } else { 0 })
                .checked_add(1)
                .ok_or(ElasticCertificateError::ArithmeticOverflow {
                    resource: ResourceKind::WorkUnits,
                })?;
            builder.charge_work(edge_work)?;
            ledger.charge_many(&[
                (ResourceKind::TrieEdges, 1),
                (ResourceKind::WorkUnits, edge_work),
                (ResourceKind::DpCells, transition_bound),
            ])?;
            let child = {
                let frame = stack.last_mut().expect("nonempty checked");
                let Some((consumed_unit, child)) = traversal.next_dfs_edge(&mut frame.edges) else {
                    return Err(ElasticCertificateError::InvalidStoredData);
                };
                if consumed_unit != unit {
                    return Err(ElasticCertificateError::InvalidStoredData);
                }
                path.push(unit);
                if !K::Monoid::within(prefix_bound, tau) {
                    builder.record(&path, |quantized_path| ElasticRangeEvidence::PrefixPruned {
                        quantized_path,
                        lower_bound: prefix_bound,
                    })?;
                    path.pop();
                    None
                } else {
                    Some(child)
                }
            };
            let Some(child) = child else {
                continue;
            };

            // Reconstruct only the exact dense generation demanded by this
            // transition from the canonical sparse residual.
            while let Some(row) = previous_active.pop() {
                *previous
                    .get_mut(row)
                    .ok_or(ElasticCertificateError::InvalidStoredData)? = K::Monoid::TOP;
            }
            let source_state = states
                .get(source)
                .ok_or(ElasticCertificateError::InvalidStoredData)?;
            for position in &source_state.positions {
                let row = usize::try_from(position.row).map_err(|_| {
                    ElasticCertificateError::ArithmeticOverflow {
                        resource: ResourceKind::DpCells,
                    }
                })?;
                *previous
                    .get_mut(row)
                    .ok_or(ElasticCertificateError::InvalidStoredData)? = position.cost;
            }
            let held_target = source_state
                .carry
                .and_then(|carry| self.kernel.carry_interval(carry))
                .unwrap_or((0.0, 0.0));
            for (position_index, position) in source_state.positions.iter().enumerate() {
                let start = position.row as usize;
                previous_active.push(start);
                let stop = source_state
                    .positions
                    .get(position_index + 1)
                    .map_or(column_width, |next_position| next_position.row as usize);
                for row in start.saturating_add(1)..stop {
                    let Some(vertical) = self.kernel.vertical_epsilon_extension(
                        query,
                        held_target,
                        row,
                        previous,
                        plan,
                    ) else {
                        break;
                    };
                    if !K::Monoid::within(vertical, tau) {
                        break;
                    }
                    previous[row] = vertical;
                    previous_active.push(row);
                }
            }
            while let Some(row) = next_active.pop() {
                *next
                    .get_mut(row)
                    .ok_or(ElasticCertificateError::InvalidStoredData)? = K::Monoid::TOP;
            }

            let source_carry = source_state.carry;
            let sparse = self.kernel.step_interval_frontier(
                previous,
                previous_active,
                query,
                interval,
                source_carry,
                child_depth,
                plan,
                tau,
                transition_bound,
                next,
                next_active,
            );
            let (lower_bound, carry) =
                match sparse {
                    Some(PointFrontierStep::Advanced {
                        lower_bound,
                        carry,
                        work,
                    }) => {
                        if work > transition_bound {
                            return Err(ElasticCertificateError::InvalidStoredData);
                        }
                        (lower_bound, carry)
                    }
                    Some(PointFrontierStep::WorkLimitExceeded { requested, .. }) => {
                        return Err(ElasticCertificateError::BudgetExceeded {
                            resource: ResourceKind::WorkUnits,
                            limit: transition_bound,
                            requested,
                        });
                    }
                    None => {
                        let (lower_bound, carry) = self.kernel.step_column(
                            previous,
                            query,
                            interval,
                            source_carry,
                            child_depth,
                            plan,
                            next,
                        );
                        next_active.extend(next.iter().enumerate().filter_map(|(row, cost)| {
                            K::Monoid::within(*cost, tau).then_some(row)
                        }));
                        (lower_bound, carry)
                    }
                };
            if !K::Monoid::within(lower_bound, tau) {
                builder.record(&path, |quantized_path| {
                    ElasticRangeEvidence::SubtreePruned {
                        quantized_path,
                        lower_bound,
                    }
                })?;
                path.pop();
                continue;
            }

            let final_row = self.kernel.final_row(query.len());
            let final_cost = next.get(final_row).copied().unwrap_or(K::Monoid::TOP);
            let canonical_position_count = next_active
                .iter()
                .filter(|row| {
                    let Some(cost) = next.get(**row).copied() else {
                        return true;
                    };
                    self.kernel
                        .vertical_epsilon_extension(query, interval, **row, next, plan)
                        .is_none_or(|vertical| {
                            K::Monoid::compare(vertical, cost) != Ordering::Equal
                        })
                })
                .count();
            let current_state_bytes =
                states
                    .retained_bytes()
                    .ok_or(ElasticCertificateError::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    })?;
            let prospective_state_bytes = current_state_bytes
                .checked_add(std::mem::size_of::<RangeProductState<K>>())
                .and_then(|bytes| {
                    canonical_position_count
                        .checked_mul(std::mem::size_of::<RangeProductPosition<K>>())
                        .and_then(|positions| bytes.checked_add(positions))
                })
                .ok_or(ElasticCertificateError::ArithmeticOverflow {
                    resource: ResourceKind::ScratchBytes,
                })?;
            ledger.observe_peak(
                ResourceKind::ScratchBytes,
                fixed_scratch_bytes
                    .checked_add(prospective_state_bytes)
                    .ok_or(ElasticCertificateError::ArithmeticOverflow {
                        resource: ResourceKind::ScratchBytes,
                    })?,
            )?;
            let mut positions = Vec::new();
            positions
                .try_reserve_exact(canonical_position_count)
                .map_err(|_| ElasticCertificateError::AllocationFailed {
                    resource: ResourceKind::ScratchBytes,
                    requested: prospective_state_bytes,
                })?;
            for row in next_active.iter().copied() {
                let cost = next
                    .get(row)
                    .copied()
                    .ok_or(ElasticCertificateError::InvalidStoredData)?;
                let dominated = self
                    .kernel
                    .vertical_epsilon_extension(query, interval, row, next, plan)
                    .is_some_and(|vertical| K::Monoid::compare(vertical, cost) == Ordering::Equal);
                if K::Monoid::within(cost, tau) && !dominated {
                    positions.push(RangeProductPosition {
                        row: u32::try_from(row).map_err(|_| {
                            ElasticCertificateError::ArithmeticOverflow {
                                resource: ResourceKind::DpCells,
                            }
                        })?,
                        cost,
                    });
                }
            }
            if positions.len() != canonical_position_count {
                return Err(ElasticCertificateError::InvalidStoredData);
            }
            let state = states.intern(
                &self.kernel,
                RangeProductState {
                    depth: child_depth,
                    carry: Some(carry),
                    positions,
                    final_cost,
                },
            )?;
            ledger.charge(ResourceKind::TrieNodes, 1)?;

            let mut candidates = Vec::new();
            let mut candidate_bucket = None;
            let final_value = traversal.final_value_at_cursor(child, None);
            let edges = traversal.open_dfs_node(child);
            if edges.is_final() {
                let bucket_id = final_value.ok_or(ElasticCertificateError::InvalidStoredData)?;
                let terminal_bound = states
                    .get(state)
                    .map(|state| state.final_cost)
                    .ok_or(ElasticCertificateError::InvalidStoredData)?;
                if K::Monoid::within(terminal_bound, tau) {
                    if let Some(bucket) = self.buckets.get(bucket_id) {
                        if bucket.is_empty() {
                            return Err(ElasticCertificateError::InvalidStoredData);
                        }
                        if bucket.len() > limits.resources.max_queue_entries {
                            return Err(ElasticCertificateError::BudgetExceeded {
                                resource: ResourceKind::QueueEntries,
                                limit: limits.resources.max_queue_entries,
                                requested: bucket.len(),
                            });
                        }
                        candidates.try_reserve_exact(bucket.len()).map_err(|_| {
                            ElasticCertificateError::AllocationFailed {
                                resource: ResourceKind::ScratchBytes,
                                requested: bucket.len().saturating_mul(std::mem::size_of::<u64>()),
                            }
                        })?;
                        candidates.extend(bucket.iter().copied());
                        candidates.sort_unstable();
                        if candidates.windows(2).any(|pair| pair[0] == pair[1]) {
                            return Err(ElasticCertificateError::InvalidStoredData);
                        }
                        candidate_bucket = Some(bucket_id);
                    } else {
                        return Err(ElasticCertificateError::InvalidStoredData);
                    }
                } else {
                    builder.record(&path, |quantized_path| {
                        ElasticRangeEvidence::TerminalPruned {
                            quantized_path,
                            lower_bound: terminal_bound,
                        }
                    })?;
                }
            } else if final_value.is_some() {
                return Err(ElasticCertificateError::InvalidStoredData);
            }

            let requested_frames =
                stack
                    .len()
                    .checked_add(1)
                    .ok_or(ElasticCertificateError::ArithmeticOverflow {
                        resource: ResourceKind::ContinuationBytes,
                    })?;
            let frame_bytes = requested_frames
                .checked_mul(std::mem::size_of::<CertifiedRangeFrame<D::Node>>())
                .ok_or(ElasticCertificateError::ArithmeticOverflow {
                    resource: ResourceKind::ContinuationBytes,
                })?;
            let new_candidate_bytes = candidates
                .capacity()
                .checked_mul(std::mem::size_of::<u64>())
                .ok_or(ElasticCertificateError::ArithmeticOverflow {
                    resource: ResourceKind::ContinuationBytes,
                })?;
            // DFS exhausts and releases a frame's accepting bucket before it
            // visits any outgoing edge, so only the prospective child can own
            // collision-candidate storage at this point.
            let candidate_bytes = new_candidate_bytes;
            let retained_product_bytes =
                states
                    .retained_bytes()
                    .ok_or(ElasticCertificateError::ArithmeticOverflow {
                        resource: ResourceKind::ContinuationBytes,
                    })?;
            let retained_column_bytes = frame_bytes
                .checked_add(candidate_bytes)
                .and_then(|bytes| bytes.checked_add(path.capacity()))
                .and_then(|bytes| bytes.checked_add(fixed_scratch_bytes))
                .and_then(|bytes| bytes.checked_add(retained_product_bytes))
                .ok_or(ElasticCertificateError::ArithmeticOverflow {
                    resource: ResourceKind::ContinuationBytes,
                })?;
            if retained_column_bytes > limits.resources.max_continuation_bytes {
                return Err(ElasticCertificateError::BudgetExceeded {
                    resource: ResourceKind::ContinuationBytes,
                    limit: limits.resources.max_continuation_bytes,
                    requested: retained_column_bytes,
                });
            }
            stack
                .try_reserve_exact(1)
                .map_err(|_| ElasticCertificateError::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested: requested_frames,
                })?;
            stack.push(CertifiedRangeFrame {
                depth: child_depth,
                state,
                candidate_bucket,
                candidates,
                next_candidate: 0,
                edges,
            });
            ledger.observe_peak(
                ResourceKind::QueueEntries,
                stack
                    .len()
                    .checked_add(stack.last().map_or(0, |frame| frame.candidates.len()))
                    .ok_or(ElasticCertificateError::ArithmeticOverflow {
                        resource: ResourceKind::QueueEntries,
                    })?,
            )?;
        }

        results.sort_unstable_by(|left, right| {
            K::Monoid::compare(left.1, right.1).then_with(|| left.0.cmp(&right.0))
        });
        Ok((
            results,
            ElasticRangeCertificate {
                snapshot_identity: self.snapshot_identity,
                query_bits,
                cutoff: tau,
                evidence: builder.evidence,
                work_units: builder.work_units,
                path_bytes: builder.path_bytes,
                witness_bytes: builder.witness_bytes,
            },
        ))
    }

    /// Recompute and compare every certificate decision against an expected
    /// query and cutoff. Any mutated bound, path, survivor, id, ordering, query,
    /// cutoff, or snapshot identity fails verification.
    pub fn verify_range_certificate(
        &self,
        query: &[f64],
        tau: Cost<K>,
        certificate: &ElasticRangeCertificate<Cost<K>>,
        limits: ElasticCertificateLimits,
    ) -> Result<bool, ElasticCertificateError> {
        if certificate.snapshot_identity != self.snapshot_identity
            || certificate.cutoff != tau
            || certificate.query_bits.len() != query.len()
            || !certificate
                .query_bits
                .iter()
                .zip(query)
                .all(|(bits, value)| *bits == value.to_bits())
        {
            return Ok(false);
        }
        let (_, expected) = self.search_range_with_certificate(query, tau, limits)?;
        Ok(&expected == certificate)
    }
}

impl<V, D> ElasticTransducer<ErpConfig, V, D>
where
    V: Eq + std::hash::Hash + Clone,
    D: ElasticDictionaryBackend<Label = u8>,
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
    ) -> Result<ErpAutomatonRangeOutcome<'_, V, D>, TemporalAutomatonError> {
        let mut ledger = ResourceLedger::new(limits);
        let machine = ErpFrontierMachine::new(query, self.kernel, tau, limits)?;
        let mut query_storage = Vec::new();
        query_storage.try_reserve_exact(query.len()).map_err(|_| {
            TemporalAutomatonError::Resource(IncompleteReason::AllocationFailed {
                resource: ResourceKind::ContinuationBytes,
                requested: query.len().saturating_mul(std::mem::size_of::<f64>()),
            })
        })?;
        query_storage.extend_from_slice(query);
        let mut stack = Vec::new();
        let mut terminal = None;
        let done = self.is_empty();
        let root = self.dawg.elastic_root();
        let (mut traversal, root_cursor) =
            TraversalSession::capture_mapped(DictionaryTraversalRoot::owned(root));

        if !done {
            if !traversal.supports_efficient_dfs_edge_paging() {
                terminal = Some(IncompleteReason::Unsupported);
            }
            let seed = machine.seed();
            let final_cost = machine
                .final_cost(seed)
                .map_err(TemporalAutomatonError::Resource)?;
            if terminal.is_none() && stack.try_reserve_exact(1).is_err() {
                terminal = Some(IncompleteReason::AllocationFailed {
                    resource: ResourceKind::ContinuationBytes,
                    requested: std::mem::size_of::<ErpAutomatonRangeFrame<D::Node>>(),
                });
            }
            if terminal.is_none() {
                match ErpAutomatonRangeFrame::open(
                    &mut traversal,
                    root_cursor,
                    0,
                    seed,
                    final_cost.is_some(),
                ) {
                    Ok(frame) => stack.push(frame),
                    Err(reason) => terminal = Some(reason),
                }
            }
            if terminal.is_none() {
                terminal = ledger.charge(ResourceKind::TrieNodes, 1).err();
            }
        }

        let result_capacity = self
            .len()
            .min(DEFAULT_RESULT_BUFFER_CAPACITY)
            .min(limits.max_results);
        let mut results = Vec::new();
        if terminal.is_none() && results.try_reserve_exact(result_capacity).is_err() {
            terminal = Some(IncompleteReason::AllocationFailed {
                resource: ResourceKind::Results,
                requested: result_capacity,
            });
        }
        let mut continuation = ErpAutomatonRangeContinuation {
            index: self,
            query: query_storage,
            tau,
            machine,
            traversal,
            stack,
            results,
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

impl<V, D> ElasticTransducer<MsmKernel, V, D>
where
    V: Eq + std::hash::Hash + Clone,
    D: ElasticDictionaryBackend<Label = u8>,
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
    use crate::time_series::kernels::ErpConfig;
    use crate::time_series::msm_interval::interval_column_len;
    use crate::time_series::MsmTransducer;
    use proptest::prelude::*;

    #[derive(Debug, Default)]
    struct RejectingDictionary {
        inner: DynamicDawg<usize>,
    }

    impl ElasticDictionaryBackend for RejectingDictionary {
        type Label = u8;
        type Node = <DynamicDawg<usize> as Dictionary>::Node;

        fn elastic_root(&self) -> Self::Node {
            self.inner.root()
        }

        fn elastic_bucket(&self, key: &[u8]) -> Option<usize> {
            self.inner.get_bytes_value(key)
        }

        fn elastic_len(&self) -> Option<usize> {
            Dictionary::len(&self.inner)
        }
    }

    impl ElasticMutableDictionaryBackend for RejectingDictionary {
        fn elastic_try_insert_bucket(
            &mut self,
            _key: &[u8],
            _bucket: usize,
        ) -> Result<bool, ElasticMutationError> {
            Err(ElasticMutationError::Dictionary(
                "injected durable write failure".to_owned(),
            ))
        }
    }

    fn erp_product_state(cost: f64) -> RangeProductState<ErpConfig> {
        RangeProductState {
            depth: 1,
            carry: Some(()),
            positions: vec![RangeProductPosition { row: 1, cost }],
            final_cost: cost,
        }
    }

    #[test]
    fn exact_residual_interner_canonicalizes_signed_zero_and_reuses_one_id() {
        let kernel = ErpConfig::new(0.0);
        let mut arena = RangeProductStateArena::new(8, 8);
        let negative_zero = arena
            .intern(&kernel, erp_product_state(-0.0))
            .expect("first lawful residual fits");
        let positive_zero = arena
            .intern(&kernel, erp_product_state(0.0))
            .expect("equal lawful residual reuses the arena entry");

        assert_eq!(negative_zero, positive_zero);
        assert_eq!(arena.states.len(), 1);
        assert_eq!(arena.reused_states, 1);
    }

    #[test]
    fn fingerprint_collision_never_merges_unequal_exact_residuals() {
        let kernel = ErpConfig::new(0.0);
        let mut arena = RangeProductStateArena::new(8, 8);
        let first = arena
            .intern(&kernel, erp_product_state(1.0))
            .expect("first lawful residual fits");
        let second_state = erp_product_state(2.0);
        let second_fingerprint = RangeProductStateArena::fingerprint(&kernel, &second_state)
            .expect("weighted costs expose exact canonical keys");
        arena
            .fingerprints
            .entry(second_fingerprint)
            .or_default()
            .push(first);

        let second = arena
            .intern(&kernel, second_state)
            .expect("collision bucket is exact-checked before insertion");
        assert_ne!(first, second);
        assert_eq!(arena.states.len(), 2);
    }

    #[test]
    fn residual_arena_limit_rejects_before_inserting_a_distinct_state() {
        let kernel = ErpConfig::new(0.0);
        let mut arena = RangeProductStateArena::new(1, 8);
        let first = arena
            .intern(&kernel, erp_product_state(1.0))
            .expect("first residual fits the exact state limit");
        let error = arena
            .intern(&kernel, erp_product_state(2.0))
            .expect_err("second distinct residual crosses the state limit");

        assert!(matches!(
            error,
            IncompleteReason::BudgetExceeded {
                resource: ResourceKind::QueueEntries,
                limit: 1,
                requested: 2,
            }
        ));
        assert_eq!(arena.states.len(), 1);
        assert_eq!(arena.states[0].positions[0].cost, 1.0);
        assert_eq!(first, TemporalStateId(0));
    }

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
            .filter(|(_, d)| *d <= tau)
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
    fn dictionary_failure_leaves_all_index_components_and_identity_unchanged() {
        let mut index: ElasticTransducer<MsmKernel, u64, RejectingDictionary> =
            ElasticTransducer::with_dictionary(
                QuantizationConfig::for_u8(0.0, 100.0),
                MsmConfig::new(1.0),
                RejectingDictionary::default(),
            );
        let identity = ElasticSnapshotIdentity([0x5a; 32]);
        index.snapshot_identity = Some(identity);

        assert!(matches!(
            index.try_insert(7, &[10.0, 20.0]),
            Err(ElasticMutationError::Dictionary(message))
                if message == "injected durable write failure"
        ));
        assert_eq!(index.snapshot_identity(), Some(identity));
        assert_eq!(index.len(), 0);
        assert_eq!(index.dawg.elastic_len(), Some(0));
        assert!(index.buckets.is_empty());
        assert!(index.originals.is_empty());

        // The infallible compatibility surface also remains fail-closed; it
        // exposes no error detail, so callers that need evidence use try_insert.
        assert!(!index.insert(7, &[10.0, 20.0]));
        assert_eq!(index.snapshot_identity(), Some(identity));
        assert_eq!(index.len(), 0);
        assert_eq!(index.dawg.elastic_len(), Some(0));
        assert!(index.buckets.is_empty());
        assert!(index.originals.is_empty());
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
