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
//! - **K3 — exact survivors:** [`ElasticKernel::exact_with_cutoff`] returns the
//!   exact `D(query, candidate)` whenever that value is within the cutoff.
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
use std::fmt::Debug;

pub mod interval;
mod walker;

pub use walker::{ElasticSearchStats, ElasticTransducer};

/// Cost carrier selected by an elastic kernel.
pub type Cost<K> = <<K as ElasticKernel>::Monoid as CostMonoid>::Cost;

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

    /// Normalize public configuration values at the construction boundary.
    #[inline]
    fn normalized(self) -> Self {
        self
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

    /// Exact candidate score, returning `None` exactly when it exceeds cutoff
    /// (or is outside the kernel's lawful domain).
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

    /// Build immutable query metadata once per search.
    fn plan(&self, query: &[f64]) -> Self::QueryPlan;

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
