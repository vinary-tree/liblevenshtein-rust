//! Generic exact retrieval for elastic time-series distances.
//!
//! An *elastic kernel* supplies one dynamic-programming (DP) column transition
//! and an exact candidate scorer. [`ElasticTransducer`] supplies the shared
//! quantized trie, prefix-amortized column storage, range traversal, and
//! best-first `k`-nearest-neighbour traversal. This division keeps the trie
//! algorithm independent of whether paths accumulate costs with addition
//! (MSM, ERP, TWED, and DTW) or with a bottleneck maximum (discrete Fréchet).
//!
//! # Soundness contract
//!
//! Implementations must discharge four obligations. Let `B(p, i)` denote the
//! value in row `i` of the interval column for trie prefix `p`, and let `D` be
//! the exact distance.
//!
//! - **K1 — interval admissibility:** every concrete series represented below
//!   `p` has a DP column whose row `i` is at least `B(p, i)`.
//! - **K2 — inflation:** every lawful step cost is no smaller than the monoid
//!   identity. Together with monotonicity, extending a path cannot lower its
//!   accumulated cost.
//! - **K3 — exact survivors:** the exact point-frontier recurrence used by
//!   bounded verification and [`ElasticKernel::exact_with_cutoff`] used by
//!   unbounded compatibility calls agree on the exact `D(query, candidate)`
//!   whenever that value is within the cutoff.
//! - **K4 — bound coherence:** every value returned by
//!   [`ElasticKernel::candidate_lower_bound`] is no greater than `D`.
//!
//! K1 and K2 justify subtree pruning by a column minimum. K3 removes false
//! positives at leaves. K4 is an optional second leaf-level filter; returning
//! the monoid identity is always sound. No obligation requires the triangle
//! inequality, so non-metric kernels such as banded DTW remain admissible.
//!
//! The corresponding cost-algebra laws are documented by
//! [`crate::cost::CostMonoid`].

use crate::cost::CostMonoid;
use crate::time_series::bounded::{IncompleteReason, ResourceKind};
use std::fmt::Debug;

pub mod interval;
pub(crate) mod sparse;

/// Internal tagged result of one exact sparse point transition.
#[doc(hidden)]
pub enum PointFrontierStep<C, Carry> {
    /// The next generation was constructed within its hard work ceiling.
    Advanced {
        lower_bound: C,
        carry: Carry,
        work: usize,
    },
    /// Construction stopped before evaluating the requested row.
    WorkLimitExceeded { completed: usize, requested: usize },
}
mod walker;

pub use walker::{
    BoundedRangeOutcome, CertifiedRangeResults, ElasticCertificateError, ElasticCertificateLimits,
    ElasticDictionaryBackend, ElasticMutableDictionaryBackend, ElasticMutationError,
    ElasticProductStateStats, ElasticRangeCertificate, ElasticRangeEvidence, ElasticSearchStats,
    ElasticSnapshotIdentity, ElasticTransducer, ErpAutomatonRangeContinuation,
    ErpAutomatonRangeOutcome, ExactRangeResults, ExactSearchOutcome, RangeContinuation,
};
#[cfg(feature = "persistent-artrie")]
pub use walker::{
    ElasticSnapshot, ElasticSnapshotError, ElasticSnapshotKernel, ElasticSnapshotLimits,
    ElasticSnapshotMetadata, SnapshotPersistentDictionary,
};

/// Cost carrier selected by an elastic kernel.
pub type Cost<K> = <<K as ElasticKernel>::Monoid as CostMonoid>::Cost;

/// Logical storage owned while constructing and retaining one query plan.
///
/// `retained_bytes` remains live for the query lifetime. `construction_peak_bytes`
/// includes retained storage plus transient plan-building scratch. Allocator
/// bookkeeping and capacity rounding are intentionally outside this logical
/// resource ABI, consistently with the other temporal resource ledgers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueryPlanStorage {
    retained_bytes: usize,
    construction_peak_bytes: usize,
}

impl QueryPlanStorage {
    /// A query plan that owns no heap storage.
    pub const EMPTY: Self = Self {
        retained_bytes: 0,
        construction_peak_bytes: 0,
    };

    /// Describe an exact logical retained size and construction peak.
    ///
    /// The peak is normalized upward to the retained size because retained
    /// storage necessarily exists at the end of construction.
    pub const fn new(retained_bytes: usize, construction_peak_bytes: usize) -> Self {
        Self {
            retained_bytes,
            construction_peak_bytes: if construction_peak_bytes < retained_bytes {
                retained_bytes
            } else {
                construction_peak_bytes
            },
        }
    }

    /// Checked storage for `elements` fixed-width retained and transient units.
    pub fn checked_per_element(
        elements: usize,
        retained_bytes_per_element: usize,
        transient_bytes_per_element: usize,
    ) -> Result<Self, IncompleteReason> {
        let retained_bytes = elements.checked_mul(retained_bytes_per_element).ok_or(
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            },
        )?;
        let transient_bytes = elements.checked_mul(transient_bytes_per_element).ok_or(
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            },
        )?;
        let construction_peak_bytes = retained_bytes.checked_add(transient_bytes).ok_or(
            IncompleteReason::ArithmeticOverflow {
                resource: ResourceKind::ScratchBytes,
            },
        )?;
        Ok(Self::new(retained_bytes, construction_peak_bytes))
    }

    /// Logical bytes retained for the complete query lifetime.
    #[inline]
    pub const fn retained_bytes(self) -> usize {
        self.retained_bytes
    }

    /// Peak logical plan-owned bytes during construction.
    #[inline]
    pub const fn construction_peak_bytes(self) -> usize {
        self.construction_peak_bytes
    }
}

#[inline]
pub(crate) fn canonical_f64_state_key(value: f64) -> Option<u64> {
    (!value.is_nan()).then(|| if value == 0.0 { 0 } else { value.to_bits() })
}

#[inline]
pub(crate) fn canonical_f64_pair_state_key(left: f64, right: f64) -> Option<[u64; 2]> {
    Some([
        canonical_f64_state_key(left)?,
        canonical_f64_state_key(right)?,
    ])
}

/// Dynamic-programming and exact-scoring policy for one elastic distance.
///
/// The trait has no object-safe requirement: the walker is monomorphized once
/// per kernel, keeping cost combination and comparison statically dispatched.
pub trait ElasticKernel: Clone + Debug + Send + Sync + 'static {
    /// Whether the exact distance satisfies the metric axioms on its
    /// documented quotient domain.
    ///
    /// This is a machine-queryable labelling contract, not permission to use
    /// triangle-inequality pruning. Such structures must additionally require
    /// [`MetricElasticKernel`], which DTW deliberately does not implement.
    const IS_METRIC: bool;

    /// Ordered path-cost algebra.
    type Monoid: CostMonoid;

    /// Prefix state not represented in a DP column.
    ///
    /// MSM and TWED carry the preceding target interval, DTW carries its
    /// cumulative prefix LB_Keogh value, and ERP and discrete Fréchet use
    /// `()`.
    type Carry: Copy + Debug + Send + Sync;

    /// Query metadata computed once and shared by every visited trie edge.
    ///
    /// DTW uses lower/upper envelopes; kernels without preprocessing use `()`.
    type QueryPlan: Default + Debug + Send + Sync;

    /// Exact logical plan storage for a query of `query_len` samples.
    ///
    /// Implementations must include all retained query metadata and all
    /// transient plan-building storage, and the declaration must correspond to
    /// the allocations performed by [`Self::try_plan`]. Checked overflow is a
    /// tagged resource failure; bounded callers preflight this value before
    /// constructing the plan.
    fn query_plan_storage(&self, query_len: usize) -> Result<QueryPlanStorage, IncompleteReason>;

    /// Exact canonical key for continuation context retained in a product state.
    ///
    /// Equal keys must imply identical future transition behavior. Returning
    /// `None` disables interning and transition caching for the affected state
    /// without changing exact results. Approximate floating equality is never
    /// a lawful key.
    #[doc(hidden)]
    #[inline]
    fn canonical_carry_key(&self, _carry: Self::Carry) -> Option<[u64; 2]> {
        None
    }

    /// Normalize public configuration values at the construction boundary.
    #[inline]
    fn normalized(self) -> Self {
        self
    }

    /// Whether `cutoff` belongs to the kernel's lawful nonnegative cost domain.
    ///
    /// The default accepts values ordered between the monoid identity and top.
    /// Floating-point monoids order NaN outside that closed interval.
    #[inline]
    fn cutoff_is_valid(&self, cutoff: Cost<Self>) -> bool {
        Self::Monoid::compare(cutoff, Self::Monoid::ZERO) != std::cmp::Ordering::Less
            && Self::Monoid::compare(cutoff, Self::Monoid::TOP) != std::cmp::Ordering::Greater
    }

    /// Whether interval traversal is defined for this query.
    ///
    /// The default accepts finite real-valued samples. A rejected query uses
    /// deterministic exact scanning for range search and yields no finite kNN
    /// candidates, avoiding NaN-contaminated priority queues.
    #[inline]
    fn supports_interval_query(&self, query: &[f64]) -> bool {
        query.iter().all(|value| value.is_finite())
    }

    /// Whether the recurrence admits any alignment for these operand lengths,
    /// independently of sample values and finite path cost.
    ///
    /// A bounded scorer consults this predicate only when its cutoff is
    /// [`Self::Monoid::TOP`]. A structurally reachable final state that instead
    /// evaluates to `TOP` is numeric overflow, not evidence of absence.
    /// Kernels with forbidden empty operands or constrained bands override the
    /// default.
    #[doc(hidden)]
    #[inline]
    fn alignment_is_structurally_possible(&self, _query_len: usize, _candidate_len: usize) -> bool {
        true
    }

    /// Required DP column length for a query of `query_len` samples.
    fn column_len(&self, query_len: usize) -> Option<usize>;

    /// Row containing the distance for a complete target series.
    fn final_row(&self, query_len: usize) -> usize;

    /// Advance one interval-relaxed DP column.
    ///
    /// `previous_carry` is `None` at the first target element. The return value
    /// is `(column_lower_bound, carry_for_the_consumed_interval)`.
    #[allow(clippy::too_many_arguments)]
    fn step_column(
        &self,
        previous: &[Cost<Self>],
        query: &[f64],
        current_interval: (f64, f64),
        previous_carry: Option<Self::Carry>,
        depth: usize,
        plan: &Self::QueryPlan,
        column: &mut Vec<Cost<Self>>,
    ) -> (Cost<Self>, Self::Carry);

    /// Construct one exact point-label frontier from the rows reachable in the
    /// preceding generation.
    ///
    /// Returning `None` selects the dense-column fallback. A specialized
    /// implementation writes only finite in-cutoff rows into `column`, appends
    /// their strictly increasing row indices to `active`, and returns the
    /// minimum live cost plus the exact number of recurrence rows evaluated.
    /// It must charge each row before evaluation and return
    /// [`PointFrontierStep::WorkLimitExceeded`] without evaluating a row that
    /// would cross `max_work`.
    /// Query-sized scratch remains allocated once by the caller; no target
    /// prefix is retained.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn step_point_frontier(
        &self,
        _previous: &[Cost<Self>],
        _previous_active: &[usize],
        _query: &[f64],
        _target: f64,
        _previous_carry: Option<Self::Carry>,
        _depth: usize,
        _plan: &Self::QueryPlan,
        _cutoff: Cost<Self>,
        _max_work: usize,
        _column: &mut [Cost<Self>],
        _active: &mut Vec<usize>,
    ) -> Option<PointFrontierStep<Cost<Self>, Self::Carry>> {
        None
    }

    /// Construct one interval-relaxed sparse frontier for a dictionary edge.
    ///
    /// The contract is the interval analogue of
    /// [`Self::step_point_frontier`]. Returning `None` selects the legacy dense
    /// column transition; a specialized implementation schedules only rows
    /// seeded by the previous frontier and its immediate successors, then
    /// performs the kernel's vertical closure. Each row is charged before it
    /// is evaluated.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn step_interval_frontier(
        &self,
        _previous: &[Cost<Self>],
        _previous_active: &[usize],
        _query: &[f64],
        _target: (f64, f64),
        _previous_carry: Option<Self::Carry>,
        _depth: usize,
        _plan: &Self::QueryPlan,
        _cutoff: Cost<Self>,
        _max_work: usize,
        _column: &mut [Cost<Self>],
        _active: &mut Vec<usize>,
    ) -> Option<PointFrontierStep<Cost<Self>, Self::Carry>> {
        None
    }

    /// Recompute the immediate vertical epsilon extension ending at `row`.
    ///
    /// The dictionary product removes a row only when this value compares
    /// exactly equal to the row's recurrence cost. Returning `None` disables
    /// cross-row subsumption. Implementations must return only a concrete
    /// zero-target-consumption transition in the kernel's formal model.
    #[doc(hidden)]
    fn vertical_epsilon_extension(
        &self,
        _query: &[f64],
        _target: (f64, f64),
        _row: usize,
        _column: &[Cost<Self>],
        _plan: &Self::QueryPlan,
    ) -> Option<Cost<Self>> {
        None
    }

    /// Recover the target interval held by a canonical frontier context.
    ///
    /// Kernels whose vertical epsilon cost depends on the held target label
    /// return it here so a compact state can lazily reconstruct only the
    /// closure rows demanded by the next transition. Context-free vertical
    /// costs may return any fixed finite interval.
    #[doc(hidden)]
    fn carry_interval(&self, _carry: Self::Carry) -> Option<(f64, f64)> {
        None
    }

    /// Optional constant-time lower bound evaluated before a child DP column.
    ///
    /// The default identity bound preserves existing kernels. Banded DTW uses
    /// the cumulative prefix LB_Keogh value carried by the parent, allowing the
    /// walker to reject a child before paying for its banded column. A kernel
    /// that overrides this method must return a K1-admissible subtree bound and
    /// must ensure the carry returned by [`Self::step_column`] advances the same
    /// prefix state.
    #[inline]
    fn prefix_lower_bound(
        &self,
        _query: &[f64],
        _current_interval: (f64, f64),
        _previous_carry: Option<Self::Carry>,
        _depth: usize,
        _plan: &Self::QueryPlan,
    ) -> Cost<Self> {
        Self::Monoid::ZERO
    }

    /// Unbounded compatibility scorer for one complete candidate.
    ///
    /// Strict bounded operations instead reuse their fallibly allocated exact
    /// point-frontier workspace. Implementations must keep this scorer in exact
    /// correspondence with [`Self::step_point_frontier`]. `None` means the
    /// distance exceeds cutoff or no alignment exists; bounded callers use the
    /// structural predicate and tagged arithmetic path rather than relying on
    /// this intentionally compact legacy result.
    fn exact_with_cutoff(
        &self,
        query: &[f64],
        candidate: &[f64],
        cutoff: Cost<Self>,
    ) -> Option<Cost<Self>>;

    /// Admissible candidate-level lower bound.
    fn candidate_lower_bound(
        &self,
        query: &[f64],
        candidate: &[f64],
        plan: &Self::QueryPlan,
    ) -> Cost<Self>;

    /// Fallibly build immutable query metadata once per search.
    fn try_plan(&self, query: &[f64]) -> Result<Self::QueryPlan, IncompleteReason>;

    /// Build query metadata for an explicitly unbounded compatibility surface.
    ///
    /// Strict production adapters call [`Self::try_plan`] and preserve its
    /// tagged failure. This convenience method panics rather than silently
    /// substituting an unsafe default plan after allocation failure.
    #[inline]
    fn plan(&self, query: &[f64]) -> Self::QueryPlan {
        self.try_plan(query)
            .unwrap_or_else(|error| panic!("elastic query-plan construction failed: {error:?}"))
    }

    /// Exact distance between two empty series.
    fn empty_pair_cost(&self) -> Cost<Self>;

    /// Exact empty/nonempty distance for the supplied nonempty series.
    ///
    /// The sequence argument is necessary for value-dependent gap models such
    /// as ERP; a nullary constant would make the abstraction MSM-specific.
    fn empty_vs_nonempty_cost(&self, nonempty: &[f64]) -> Cost<Self>;
}

/// Compile-time gate for structures whose soundness uses the triangle
/// inequality.
///
/// Implement this marker only after the kernel's metric or quotient-metric
/// proof has been reviewed. Generic lower-bound traversal needs no metric
/// axiom and therefore remains parameterized by [`ElasticKernel`] alone.
///
/// ```compile_fail
/// use liblevenshtein::time_series::elastic::MetricElasticKernel;
/// use liblevenshtein::time_series::DtwConfig;
///
/// fn triangle_dependent_index<K: MetricElasticKernel>() {}
/// triangle_dependent_index::<DtwConfig>();
/// ```
pub trait MetricElasticKernel: ElasticKernel {}
