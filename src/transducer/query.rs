//! Lazy query iterators for approximate string matching.

use super::dictionary_traversal::{
    CursorNativePath, ParentArenaPath, PathFrontier, ResultPathStrategy, TraversalCursor,
    TraversalSession,
};
use super::packed_dfa::ExactLabelDfaRow;
use super::packed_special::{
    PackedMergeSplitMachine, PackedOsaMachine, PackedSpecialMachine, SpecialKernel,
};
use super::packed_standard::PackedStandardMachine;
use super::query_result::QueryResult;
use super::transition::{
    initial_state_affine, AffineTransitionSettings, FinishMode, TransitionSettings,
    UnitCostFrontier, UnitCostMachine,
};
use super::variants::AffineGapParams;
use super::{Algorithm, StatePool, SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted};
use libdictenstein::{CharUnit, DictionaryNode, DictionaryTraversalRoot};
use std::collections::VecDeque;
use std::marker::PhantomData;

#[cfg(feature = "perf-instrumentation")]
use rustc_hash::FxHashSet;

/// Query result containing term and distance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The matching term
    pub term: String,
    /// Edit distance from query
    pub distance: usize,
}

/// Affine-gap result with both presentation and exact fixed-point costs.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineCandidate {
    /// Matching dictionary term.
    pub term: String,
    /// Cost converted through the parameter set's exact scale.
    pub distance: f64,
    /// Exact integer cost used by pruning and ordering comparisons.
    pub scaled_distance: usize,
}

/// Iterator returned by [`Transducer::query_affine`](crate::transducer::Transducer::query_affine).
pub struct AffineQueryIterator<N: DictionaryNode, P: SubstitutionPolicy = Unrestricted> {
    inner: QueryIterator<N, Candidate, P>,
    params: AffineGapParams,
}

impl<N: DictionaryNode, P: SubstitutionPolicy> AffineQueryIterator<N, P> {
    pub(crate) fn new(inner: QueryIterator<N, Candidate, P>, params: AffineGapParams) -> Self {
        Self { inner, params }
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>> Iterator
    for AffineQueryIterator<N, P>
{
    type Item = AffineCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|candidate| AffineCandidate {
            distance: self.params.unscale_cost(candidate.distance),
            scaled_distance: candidate.distance,
            term: candidate.term,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    std::iter::FusedIterator for AffineQueryIterator<N, P>
{
}

/// Query result containing the matched term as a raw unit sequence and its distance.
///
/// The units-native analogue of [`Candidate`]: `term` is the matched dictionary key
/// as a `Vec<U>` (e.g. a `Vec<u64>` token-id sequence) with **no** `String`
/// reconstruction, so it is lossless for alphabets whose `to_string` is not
/// round-trippable (notably `u64` token sequences). See [`QueryIterator`] and
/// [`Transducer::query_units_with_distance`](crate::transducer::Transducer::query_units_with_distance).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitCandidate<U: CharUnit> {
    /// The matching term as its unit sequence.
    pub term: Vec<U>,
    /// Edit distance from the query.
    pub distance: usize,
}

struct QueryStep<F> {
    frontier: F,
    #[cfg(feature = "perf-instrumentation")]
    position_count: usize,
    #[cfg(feature = "perf-instrumentation")]
    storage_bytes: usize,
}

trait QueryKernel<U, P>
where
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
{
    type Frontier;

    // Keep the transition inputs flat: this is a hot interface implemented by
    // distinct kernels, and aggregating them would obscure their independent
    // lifetimes without reducing call-site work.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        frontier: &Self::Frontier,
        state_pool: &mut StatePool,
        policy: &P,
        label: U,
        query: &[U],
        max_distance: usize,
        substring_mode: bool,
    ) -> Option<QueryStep<Self::Frontier>>;

    #[allow(clippy::too_many_arguments)]
    fn expand<N, F>(
        &mut self,
        frontier: &Self::Frontier,
        traversal: &mut TraversalSession<N>,
        position: TraversalCursor<N::SnapshotCursor>,
        state_pool: &mut StatePool,
        policy: &P,
        query: &[U],
        max_distance: usize,
        substring_mode: bool,
        visitor: F,
    ) -> bool
    where
        N: DictionaryNode<Unit = U>,
        F: FnMut(U, TraversalCursor<N::SnapshotCursor>, QueryStep<Self::Frontier>),
    {
        traversal.filter_map_edges_and_finality(
            position,
            |label| {
                crate::causal_perf::record_edges_enumerated(1);
                self.step(
                    frontier,
                    state_pool,
                    policy,
                    label,
                    query,
                    max_distance,
                    substring_mode,
                )
            },
            visitor,
        )
    }

    fn finish_distance(
        &self,
        frontier: &Self::Frontier,
        query_length: usize,
        substring_mode: bool,
    ) -> Option<usize>;

    fn unit_frontier(&self, _frontier: &Self::Frontier) -> Option<UnitCostFrontier> {
        None
    }
}

struct UnitCostQueryKernel<U: CharUnit> {
    algorithm: Algorithm,
    transitions: UnitCostMachine<U>,
}

/// Concrete packed-machine interface selected once when a query starts.
///
/// This private trait is statically dispatched. It shares the ordinary query
/// scheduler across packed algorithms without retaining the `UnitCostMachine`
/// representation branch in every edge and finality check.
trait PackedQueryMachine<U, P>
where
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
{
    fn step_prepared(&mut self, source: u64, policy: &P, label: U, query: &[U]) -> Option<u64>;

    fn prepare_source_row(&self, source: u64) -> Option<ExactLabelDfaRow>;
    #[cfg(feature = "perf-instrumentation")]
    fn source_row_label_is_class_zero(&self, label: U) -> bool;

    fn step_prepared_source_row(
        &mut self,
        source: &mut ExactLabelDfaRow,
        policy: &P,
        label: U,
        query: &[U],
    ) -> Option<u64>;

    fn complete_distance(&self, frontier: u64) -> Option<usize>;

    fn min_distance(&self, frontier: u64) -> Option<usize>;

    fn record_attempt();

    fn record_dead();

    #[cfg(feature = "perf-instrumentation")]
    fn active_len(&self, frontier: u64) -> usize;
}

impl<U, P> PackedQueryMachine<U, P> for PackedStandardMachine<U>
where
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
{
    #[inline(always)]
    fn step_prepared(&mut self, source: u64, policy: &P, label: U, query: &[U]) -> Option<u64> {
        self.step_prepared(source, policy, label, query)
    }

    #[inline(always)]
    fn prepare_source_row(&self, source: u64) -> Option<ExactLabelDfaRow> {
        self.prepare_source_row(source)
    }

    #[cfg(feature = "perf-instrumentation")]
    #[inline(always)]
    fn source_row_label_is_class_zero(&self, label: U) -> bool {
        self.source_row_label_is_class_zero(label)
    }

    #[inline(always)]
    fn step_prepared_source_row(
        &mut self,
        source: &mut ExactLabelDfaRow,
        policy: &P,
        label: U,
        query: &[U],
    ) -> Option<u64> {
        self.step_prepared_source_row(source, policy, label, query)
    }

    #[inline(always)]
    fn complete_distance(&self, frontier: u64) -> Option<usize> {
        self.complete_distance(frontier)
    }

    #[inline(always)]
    fn min_distance(&self, frontier: u64) -> Option<usize> {
        self.min_distance(frontier)
    }

    #[inline(always)]
    fn record_attempt() {
        crate::causal_perf::record_packed_standard_transition_attempts(1);
    }

    #[inline(always)]
    fn record_dead() {
        crate::causal_perf::record_packed_standard_transition_dead(1);
    }

    #[cfg(feature = "perf-instrumentation")]
    #[inline(always)]
    fn active_len(&self, frontier: u64) -> usize {
        self.active_len(frontier)
    }
}

impl<U, P, K> PackedQueryMachine<U, P> for PackedSpecialMachine<U, K>
where
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    K: SpecialKernel,
{
    #[inline(always)]
    fn step_prepared(&mut self, source: u64, _policy: &P, label: U, _query: &[U]) -> Option<u64> {
        self.step_prepared(source, label)
    }

    #[inline(always)]
    fn prepare_source_row(&self, source: u64) -> Option<ExactLabelDfaRow> {
        self.prepare_source_row(source)
    }

    #[cfg(feature = "perf-instrumentation")]
    #[inline(always)]
    fn source_row_label_is_class_zero(&self, label: U) -> bool {
        self.source_row_label_is_class_zero(label)
    }

    #[inline(always)]
    fn step_prepared_source_row(
        &mut self,
        source: &mut ExactLabelDfaRow,
        _policy: &P,
        label: U,
        _query: &[U],
    ) -> Option<u64> {
        self.step_prepared_source_row(source, label)
    }

    #[inline(always)]
    fn complete_distance(&self, frontier: u64) -> Option<usize> {
        self.complete_distance(frontier)
    }

    #[inline(always)]
    fn min_distance(&self, frontier: u64) -> Option<usize> {
        self.min_distance(frontier)
    }

    #[inline(always)]
    fn record_attempt() {
        match K::ALGORITHM {
            Algorithm::Transposition => {
                crate::causal_perf::record_packed_osa_transition_attempts(1);
            }
            Algorithm::MergeAndSplit => {
                crate::causal_perf::record_packed_merge_split_transition_attempts(1);
            }
            Algorithm::Standard | Algorithm::DamerauLevenshtein => {
                unreachable!("a packed special kernel has a continuation algorithm")
            }
        }
    }

    #[inline(always)]
    fn record_dead() {
        match K::ALGORITHM {
            Algorithm::Transposition => crate::causal_perf::record_packed_osa_transition_dead(1),
            Algorithm::MergeAndSplit => {
                crate::causal_perf::record_packed_merge_split_transition_dead(1);
            }
            Algorithm::Standard | Algorithm::DamerauLevenshtein => {
                unreachable!("a packed special kernel has a continuation algorithm")
            }
        }
    }

    #[cfg(feature = "perf-instrumentation")]
    #[inline(always)]
    fn active_len(&self, frontier: u64) -> usize {
        self.active_len(frontier)
    }
}

struct PackedQueryKernel<M> {
    transitions: M,
}

impl<U, P, M> QueryKernel<U, P> for PackedQueryKernel<M>
where
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    M: PackedQueryMachine<U, P>,
{
    type Frontier = UnitCostFrontier;

    #[inline(always)]
    fn step(
        &mut self,
        frontier: &Self::Frontier,
        _state_pool: &mut StatePool,
        policy: &P,
        label: U,
        query: &[U],
        _max_distance: usize,
        _substring_mode: bool,
    ) -> Option<QueryStep<Self::Frontier>> {
        crate::causal_perf::record_transition_attempts(1);
        M::record_attempt();
        let target = self
            .transitions
            .step_prepared(frontier.0, policy, label, query)
            .map(UnitCostFrontier);
        if target.is_none() {
            M::record_dead();
        }
        target.map(|frontier| QueryStep {
            #[cfg(feature = "perf-instrumentation")]
            position_count: self.transitions.active_len(frontier.0),
            #[cfg(feature = "perf-instrumentation")]
            storage_bytes: std::mem::size_of::<UnitCostFrontier>(),
            frontier,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn expand<N, F>(
        &mut self,
        frontier: &Self::Frontier,
        traversal: &mut TraversalSession<N>,
        position: TraversalCursor<N::SnapshotCursor>,
        _state_pool: &mut StatePool,
        policy: &P,
        query: &[U],
        _max_distance: usize,
        _substring_mode: bool,
        visitor: F,
    ) -> bool
    where
        N: DictionaryNode<Unit = U>,
        F: FnMut(U, TraversalCursor<N::SnapshotCursor>, QueryStep<Self::Frontier>),
    {
        if let Some(mut source) = self.transitions.prepare_source_row(frontier.0) {
            crate::causal_perf::record_packed_dfa_source_rows_prepared(1);
            #[cfg(feature = "perf-instrumentation")]
            let mut class_zero_seen = false;
            let transitions = &mut self.transitions;
            return traversal.filter_map_edges_and_finality(
                position,
                |label| {
                    #[cfg(feature = "perf-instrumentation")]
                    if transitions.source_row_label_is_class_zero(label) {
                        crate::causal_perf::record_packed_dfa_class_zero_probes(1);
                        if class_zero_seen {
                            crate::causal_perf::record_packed_dfa_class_zero_reusable_probes(1);
                        }
                        class_zero_seen = true;
                    }
                    crate::causal_perf::record_edges_enumerated(1);
                    crate::causal_perf::record_transition_attempts(1);
                    M::record_attempt();
                    let target = transitions
                        .step_prepared_source_row(&mut source, policy, label, query)
                        .map(UnitCostFrontier);
                    if target.is_none() {
                        M::record_dead();
                    }
                    target.map(|frontier| QueryStep {
                        #[cfg(feature = "perf-instrumentation")]
                        position_count: transitions.active_len(frontier.0),
                        #[cfg(feature = "perf-instrumentation")]
                        storage_bytes: std::mem::size_of::<UnitCostFrontier>(),
                        frontier,
                    })
                },
                visitor,
            );
        }

        let transitions = &mut self.transitions;
        traversal.filter_map_edges_and_finality(
            position,
            |label| {
                crate::causal_perf::record_edges_enumerated(1);
                crate::causal_perf::record_transition_attempts(1);
                M::record_attempt();
                let target = transitions
                    .step_prepared(frontier.0, policy, label, query)
                    .map(UnitCostFrontier);
                if target.is_none() {
                    M::record_dead();
                }
                target.map(|frontier| QueryStep {
                    #[cfg(feature = "perf-instrumentation")]
                    position_count: transitions.active_len(frontier.0),
                    #[cfg(feature = "perf-instrumentation")]
                    storage_bytes: std::mem::size_of::<UnitCostFrontier>(),
                    frontier,
                })
            },
            visitor,
        )
    }

    #[inline(always)]
    fn finish_distance(
        &self,
        frontier: &Self::Frontier,
        _query_length: usize,
        substring_mode: bool,
    ) -> Option<usize> {
        if substring_mode {
            self.transitions.min_distance(frontier.0)
        } else {
            self.transitions.complete_distance(frontier.0)
        }
    }

    #[inline(always)]
    fn unit_frontier(&self, frontier: &Self::Frontier) -> Option<UnitCostFrontier> {
        Some(*frontier)
    }
}

impl<U, P> QueryKernel<U, P> for UnitCostQueryKernel<U>
where
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
{
    type Frontier = UnitCostFrontier;

    #[inline]
    fn step(
        &mut self,
        frontier: &Self::Frontier,
        state_pool: &mut StatePool,
        policy: &P,
        label: U,
        query: &[U],
        max_distance: usize,
        substring_mode: bool,
    ) -> Option<QueryStep<Self::Frontier>> {
        self.transitions
            .step(
                *frontier,
                state_pool,
                policy,
                label,
                query,
                TransitionSettings::new(max_distance, self.algorithm, substring_mode),
            )
            .map(|frontier| QueryStep {
                #[cfg(feature = "perf-instrumentation")]
                position_count: self.transitions.active_len(frontier),
                #[cfg(feature = "perf-instrumentation")]
                storage_bytes: self.transitions.frontier_storage_bytes(frontier),
                frontier,
            })
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn expand<N, F>(
        &mut self,
        frontier: &Self::Frontier,
        traversal: &mut TraversalSession<N>,
        position: TraversalCursor<N::SnapshotCursor>,
        state_pool: &mut StatePool,
        policy: &P,
        query: &[U],
        max_distance: usize,
        substring_mode: bool,
        visitor: F,
    ) -> bool
    where
        N: DictionaryNode<Unit = U>,
        F: FnMut(U, TraversalCursor<N::SnapshotCursor>, QueryStep<Self::Frontier>),
    {
        let settings = TransitionSettings::new(max_distance, self.algorithm, substring_mode);
        let mut row = self
            .transitions
            .prepare_row(*frontier, state_pool, policy, query, settings);
        traversal.filter_map_edges_and_finality(
            position,
            |label| {
                crate::causal_perf::record_edges_enumerated(1);
                row.step(label).map(|frontier| QueryStep {
                    #[cfg(feature = "perf-instrumentation")]
                    position_count: row.active_len(frontier),
                    #[cfg(feature = "perf-instrumentation")]
                    storage_bytes: row.frontier_storage_bytes(frontier),
                    frontier,
                })
            },
            visitor,
        )
    }

    #[inline]
    fn finish_distance(
        &self,
        frontier: &Self::Frontier,
        query_length: usize,
        substring_mode: bool,
    ) -> Option<usize> {
        self.transitions.finish_distance(
            *frontier,
            if substring_mode {
                FinishMode::Substring
            } else {
                FinishMode::Complete
            },
            query_length,
        )
    }

    fn unit_frontier(&self, frontier: &Self::Frontier) -> Option<UnitCostFrontier> {
        Some(*frontier)
    }
}

struct AffineQueryKernel<U: CharUnit> {
    params: AffineGapParams,
    transitions: UnitCostMachine<U>,
}

impl<U, P> QueryKernel<U, P> for AffineQueryKernel<U>
where
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
{
    type Frontier = UnitCostFrontier;

    #[inline]
    fn step(
        &mut self,
        frontier: &Self::Frontier,
        state_pool: &mut StatePool,
        policy: &P,
        label: U,
        query: &[U],
        max_distance: usize,
        substring_mode: bool,
    ) -> Option<QueryStep<Self::Frontier>> {
        self.transitions
            .transition_affine_generated(
                *frontier,
                state_pool,
                policy,
                label,
                query,
                AffineTransitionSettings::new(max_distance, self.params, substring_mode),
            )
            .map(|frontier| QueryStep {
                #[cfg(feature = "perf-instrumentation")]
                position_count: self.transitions.active_len(frontier),
                #[cfg(feature = "perf-instrumentation")]
                storage_bytes: std::mem::size_of::<UnitCostFrontier>(),
                frontier,
            })
    }

    #[inline]
    fn finish_distance(
        &self,
        frontier: &Self::Frontier,
        query_length: usize,
        substring_mode: bool,
    ) -> Option<usize> {
        self.transitions.finish_affine_distance(
            *frontier,
            query_length,
            self.params,
            substring_mode,
        )
    }

    #[inline(always)]
    fn unit_frontier(&self, frontier: &Self::Frontier) -> Option<UnitCostFrontier> {
        Some(*frontier)
    }
}

/// Lazy iterator over query matches with configurable result type.
///
/// This iterator can return either:
/// - `String`: Just the matching terms (set `R = String`)
/// - `Candidate`: Terms with their edit distances (set `R = Candidate`)
/// - Custom types: Implement `QueryResult` trait
///
/// The result type is determined by the generic parameter `R`, which defaults
/// to `String` for backward compatibility.
///
/// # Type Parameter
///
/// - `N`: Dictionary node type
/// - `R`: Result type (defaults to `String`). Must implement `QueryResult`.
///
/// # Performance
///
/// Uses StatePool to eliminate State cloning overhead during traversal.
/// The pool is created per-query and reuses State allocations across
/// all transitions, reducing memory allocation by 6-10% of runtime.
///
/// Distance is computed once from automaton states (zero overhead), then
/// converted to the result type via `QueryResult::from_match()` which is
/// inlined and monomorphized at compile time (zero-cost abstraction).
///
/// # Examples
///
/// ```
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::transducer::{QueryIterator, Candidate};
///
/// let dict = DoubleArrayTrie::from_terms(vec!["test"]);
///
/// // Returns String (default)
/// let iter: QueryIterator<_, String> = QueryIterator::new(
///     dict.root(), "test".to_string(), 1, Algorithm::Standard
/// );
/// for term in iter {
///     println!("{}", term);
/// }
///
/// // Returns Candidate (term + distance)
/// let iter: QueryIterator<_, Candidate> = QueryIterator::new(
///     dict.root(), "test".to_string(), 1, Algorithm::Standard
/// );
/// for candidate in iter {
///     println!("{}: {}", candidate.term, candidate.distance);
/// }
/// ```
pub struct QueryIterator<N: DictionaryNode, R = String, P: SubstitutionPolicy = Unrestricted> {
    inner: QueryIteratorInner<N, R, P>,
}

enum QueryIteratorInner<N: DictionaryNode, R, P: SubstitutionPolicy> {
    PackedStandard(
        PathQueryIteratorCore<
            N,
            R,
            P,
            PackedQueryKernel<PackedStandardMachine<N::Unit>>,
            UnitCostFrontier,
        >,
    ),
    PackedOsa(
        PathQueryIteratorCore<
            N,
            R,
            P,
            PackedQueryKernel<PackedOsaMachine<N::Unit>>,
            UnitCostFrontier,
        >,
    ),
    PackedMergeSplit(
        PathQueryIteratorCore<
            N,
            R,
            P,
            PackedQueryKernel<PackedMergeSplitMachine<N::Unit>>,
            UnitCostFrontier,
        >,
    ),
    Unit(PathQueryIteratorCore<N, R, P, UnitCostQueryKernel<N::Unit>, UnitCostFrontier>),
    Affine(PathQueryIteratorCore<N, R, P, AffineQueryKernel<N::Unit>, UnitCostFrontier>),
}

enum PathQueryIteratorCore<N, R, P, K, F: 'static>
where
    N: DictionaryNode,
    P: SubstitutionPolicy,
{
    Parent(QueryIteratorCore<N, R, P, K, F, ParentArenaPath>),
    Cursor(QueryIteratorCore<N, R, P, K, F, CursorNativePath>),
}

struct QueryIteratorCore<N, R, P, K, F: 'static, S>
where
    N: DictionaryNode,
    P: SubstitutionPolicy,
    S: ResultPathStrategy<N>,
{
    pending: VecDeque<PathFrontier<S::Trace, F>>,
    traversal: TraversalSession<N>,
    query: Vec<N::Unit>,
    max_distance: usize,
    policy: P, // Substitution policy for matching
    path_storage: S::Storage,
    finished: bool,
    state_pool: StatePool, // Pool for State allocation reuse
    kernel: K,
    #[cfg(feature = "perf-instrumentation")]
    generated_products: FxHashSet<(
        super::dictionary_traversal::TraversalProductIdentity,
        UnitCostFrontier,
    )>,
    substring_mode: bool, // Enable substring matching (for suffix automata)
    _result_type: PhantomData<R>, // Zero-sized marker for result type
    _path_strategy: PhantomData<fn() -> S>,
}

impl<N: DictionaryNode, R: QueryResult<N::Unit>> QueryIterator<N, R, Unrestricted> {
    /// Create a new query iterator with unrestricted policy (standard Levenshtein)
    pub fn new(root: N, query: String, max_distance: usize, algorithm: Algorithm) -> Self {
        Self::with_substring_mode(root, query, max_distance, algorithm, false)
    }

    /// Create a new query iterator with substring matching mode and unrestricted policy
    pub fn with_substring_mode(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        substring_mode: bool,
    ) -> Self {
        Self::with_policy_and_substring(
            root,
            query,
            max_distance,
            algorithm,
            Unrestricted,
            substring_mode,
        )
    }
}

impl<
        N: DictionaryNode,
        R: QueryResult<N::Unit>,
        P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    > QueryIterator<N, R, P>
{
    /// Create a new query iterator with custom substitution policy
    pub fn with_policy(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
    ) -> Self {
        Self::with_policy_and_substring(root, query, max_distance, algorithm, policy, false)
    }

    /// Create a new query iterator with custom policy and substring matching mode
    pub fn with_policy_and_substring(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        // `from_str` derives units from the query string. For `u8`/`char` alphabets
        // this is the term itself; for a `u64` token alphabet it byte-packs the UTF-8
        // (lossy) — such callers must use [`with_units`](Self::with_units) instead.
        Self::with_traversal_root_and_policy(
            DictionaryTraversalRoot::owned(root),
            query,
            max_distance,
            algorithm,
            policy,
            substring_mode,
        )
    }

    pub(crate) fn with_traversal_root_and_policy(
        root: DictionaryTraversalRoot<N>,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        let query_units = N::Unit::from_str(&query);
        Self::with_traversal_root_and_units(
            root,
            query_units,
            max_distance,
            algorithm,
            policy,
            substring_mode,
        )
    }

    /// Create a query iterator from a pre-computed unit sequence, bypassing the
    /// `&str` → units conversion.
    ///
    /// This is the units-native entry point required for alphabets whose
    /// `CharUnit::from_str` is not the identity — notably a `u64` token-id
    /// dictionary built via `insert_sequence(&[u64])`, where the `&str` path would
    /// byte-pack the query and never match. The automaton core is fully
    /// unit-generic, so this differs from [`with_policy_and_substring`](Self::with_policy_and_substring)
    /// only in skipping `from_str`.
    pub fn with_units(
        root: N,
        query_units: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        Self::with_traversal_root_and_units(
            DictionaryTraversalRoot::owned(root),
            query_units,
            max_distance,
            algorithm,
            policy,
            substring_mode,
        )
    }

    pub(crate) fn with_traversal_root_and_units(
        root: DictionaryTraversalRoot<N>,
        query_units: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        let (transitions, frontier) = UnitCostMachine::seeded::<P>(
            &query_units,
            TransitionSettings::new(max_distance, algorithm, substring_mode),
        );
        let (traversal, root) = TraversalSession::capture(root);
        let inner = if static_packed_query_dispatch_disabled() {
            QueryIteratorInner::Unit(PathQueryIteratorCore::new(
                root,
                traversal,
                frontier,
                query_units,
                max_distance,
                policy,
                substring_mode,
                UnitCostQueryKernel {
                    algorithm,
                    transitions,
                },
            ))
        } else {
            match transitions {
                UnitCostMachine::PackedStandard(transitions) => {
                    QueryIteratorInner::PackedStandard(PathQueryIteratorCore::new(
                        root,
                        traversal,
                        frontier,
                        query_units,
                        max_distance,
                        policy,
                        substring_mode,
                        PackedQueryKernel { transitions },
                    ))
                }
                UnitCostMachine::PackedOsa(transitions) => {
                    QueryIteratorInner::PackedOsa(PathQueryIteratorCore::new(
                        root,
                        traversal,
                        frontier,
                        query_units,
                        max_distance,
                        policy,
                        substring_mode,
                        PackedQueryKernel { transitions },
                    ))
                }
                UnitCostMachine::PackedMergeSplit(transitions) => {
                    QueryIteratorInner::PackedMergeSplit(PathQueryIteratorCore::new(
                        root,
                        traversal,
                        frontier,
                        query_units,
                        max_distance,
                        policy,
                        substring_mode,
                        PackedQueryKernel { transitions },
                    ))
                }
                UnitCostMachine::Positional(transitions) => {
                    QueryIteratorInner::Unit(PathQueryIteratorCore::new(
                        root,
                        traversal,
                        frontier,
                        query_units,
                        max_distance,
                        policy,
                        substring_mode,
                        UnitCostQueryKernel {
                            algorithm,
                            transitions: UnitCostMachine::Positional(transitions),
                        },
                    ))
                }
            }
        };
        Self { inner }
    }

    /// Create a scaled affine-gap iterator from a query string.
    pub fn with_affine_policy_and_substring(
        root: N,
        query: String,
        max_cost: usize,
        params: AffineGapParams,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        let query_units = N::Unit::from_str(&query);
        Self::with_affine_traversal_root_and_units(
            DictionaryTraversalRoot::owned(root),
            query_units,
            max_cost,
            params,
            policy,
            substring_mode,
        )
    }

    pub(crate) fn with_affine_traversal_root_and_substring(
        root: DictionaryTraversalRoot<N>,
        query: String,
        max_cost: usize,
        params: AffineGapParams,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        Self::with_affine_traversal_root_and_units(
            root,
            N::Unit::from_str(&query),
            max_cost,
            params,
            policy,
            substring_mode,
        )
    }

    /// Create a scaled affine-gap iterator from a native unit sequence.
    pub fn with_affine_units(
        root: N,
        query_units: Vec<N::Unit>,
        max_cost: usize,
        params: AffineGapParams,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        Self::with_affine_traversal_root_and_units(
            DictionaryTraversalRoot::owned(root),
            query_units,
            max_cost,
            params,
            policy,
            substring_mode,
        )
    }

    pub(crate) fn with_affine_traversal_root_and_units(
        root: DictionaryTraversalRoot<N>,
        query_units: Vec<N::Unit>,
        max_cost: usize,
        params: AffineGapParams,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        let initial = initial_state_affine(query_units.len(), max_cost, params);
        let settings = AffineTransitionSettings::new(max_cost, params, substring_mode);
        let (transitions, initial) =
            UnitCostMachine::seeded_affine(query_units.len(), &initial, settings);
        let (traversal, root) = TraversalSession::capture(root);
        Self {
            inner: QueryIteratorInner::Affine(PathQueryIteratorCore::new(
                root,
                traversal,
                initial,
                query_units,
                max_cost,
                policy,
                substring_mode,
                AffineQueryKernel {
                    params,
                    transitions,
                },
            )),
        }
    }
}

impl<N, R, P, K, F: 'static> PathQueryIteratorCore<N, R, P, K, F>
where
    N: DictionaryNode,
    R: QueryResult<N::Unit>,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    K: QueryKernel<N::Unit, P, Frontier = F>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        frontier: F,
        query: Vec<N::Unit>,
        max_distance: usize,
        policy: P,
        substring_mode: bool,
        kernel: K,
    ) -> Self {
        if traversal.supports_cursor_key_units() {
            Self::Cursor(QueryIteratorCore::new(
                root,
                traversal,
                frontier,
                query,
                max_distance,
                policy,
                substring_mode,
                kernel,
            ))
        } else {
            Self::Parent(QueryIteratorCore::new(
                root,
                traversal,
                frontier,
                query,
                max_distance,
                policy,
                substring_mode,
                kernel,
            ))
        }
    }

    #[inline]
    fn next_match(&mut self) -> Option<R> {
        match self {
            Self::Parent(core) => (!core.finished).then(|| core.advance()).flatten(),
            Self::Cursor(core) => (!core.finished).then(|| core.advance()).flatten(),
        }
    }
}

impl<N, R, P, K, F: 'static, S> QueryIteratorCore<N, R, P, K, F, S>
where
    N: DictionaryNode,
    R: QueryResult<N::Unit>,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    K: QueryKernel<N::Unit, P, Frontier = F>,
    S: ResultPathStrategy<N>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        frontier: F,
        query: Vec<N::Unit>,
        max_distance: usize,
        policy: P,
        substring_mode: bool,
        kernel: K,
    ) -> Self {
        let (mut pending, path_storage) = S::acquire_queue();
        pending.push_back(PathFrontier::new(S::root(root), frontier));
        Self {
            pending,
            traversal,
            query,
            max_distance,
            policy,
            path_storage,
            finished: false,
            state_pool: StatePool::new(),
            kernel,
            #[cfg(feature = "perf-instrumentation")]
            generated_products: FxHashSet::default(),
            substring_mode,
            _result_type: PhantomData,
            _path_strategy: PhantomData,
        }
    }

    /// Advance to the next match
    #[inline]
    fn advance(&mut self) -> Option<R> {
        while let Some(intersection) = self.pending.pop_front() {
            crate::causal_perf::record_dictionary_intersections(1);
            crate::causal_perf::record_final_checks(1);
            let is_final = self.queue_children_and_finality(&intersection);
            // Check if this is a final match
            if is_final {
                let distance = self
                    .kernel
                    .finish_distance(
                        &intersection.frontier,
                        self.query.len(),
                        self.substring_mode,
                    )
                    .unwrap_or(usize::MAX);

                if distance <= self.max_distance {
                    let units = S::materialize_units(
                        &intersection.trace,
                        &self.traversal,
                        &self.path_storage,
                    );
                    if !self.traversal.accepts_final_units(&units) {
                        continue;
                    }

                    // Convert (units, distance) to result type R
                    // This is zero-cost: QueryResult::from_match is inlined
                    // and monomorphized at compile time. String/Candidate apply
                    // CharUnit::to_string; Vec<U>/UnitCandidate keep the units verbatim.
                    crate::causal_perf::record_matches_materialized(1);
                    return Some(R::from_match(&units, distance));
                }
            }
        }

        self.finished = true;
        None
    }

    /// Queue child intersections for exploration
    #[inline]
    fn queue_children_and_finality(&mut self, intersection: &PathFrontier<S::Trace, F>) -> bool {
        if let Some(_state) = self.kernel.unit_frontier(&intersection.frontier) {
            crate::causal_perf::record_generated_product_expansions(1);
            #[cfg(feature = "perf-instrumentation")]
            if let Some(node) = self
                .traversal
                .product_identity(S::position(&intersection.trace))
            {
                crate::causal_perf::record_generated_product_identity_expansions(1);
                if self.generated_products.insert((node, _state)) {
                    crate::causal_perf::record_generated_product_unique_expansions(1);
                } else {
                    crate::causal_perf::record_generated_product_repeated_expansions(1);
                }
            }
        }
        let mut expansion = S::begin_expansion(&intersection.trace);
        let max_distance = self.max_distance;
        let substring_mode = self.substring_mode;
        let query = &self.query;
        let policy = &self.policy;
        let kernel = &mut self.kernel;
        let state_pool = &mut self.state_pool;
        let pending = &mut self.pending;
        let path_storage = &mut self.path_storage;

        kernel.expand(
            &intersection.frontier,
            &mut self.traversal,
            S::position(&intersection.trace),
            state_pool,
            policy,
            query,
            max_distance,
            substring_mode,
            |label, child_position, step| {
                crate::causal_perf::record_transition_accepted(1);
                #[cfg(feature = "perf-instrumentation")]
                {
                    crate::causal_perf::record_state_positions_enqueued(step.position_count as u64);
                    crate::causal_perf::record_state_bytes_enqueued(step.storage_bytes as u64);
                }
                pending.push_back(PathFrontier::new(
                    S::child_trace(
                        &intersection.trace,
                        &mut expansion,
                        label,
                        child_position,
                        path_storage,
                    ),
                    step.frontier,
                ));
                crate::causal_perf::record_pending_queue_size(pending.len());
            },
        )
    }
}

impl<N, R, P, K, F: 'static, S> Drop for QueryIteratorCore<N, R, P, K, F, S>
where
    N: DictionaryNode,
    P: SubstitutionPolicy,
    S: ResultPathStrategy<N>,
{
    fn drop(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        let path_storage = std::mem::take(&mut self.path_storage);
        S::release_queue(pending, path_storage);
    }
}

impl<
        N: DictionaryNode,
        R: QueryResult<N::Unit>,
        P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    > Iterator for QueryIterator<N, R, P>
{
    type Item = R;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            QueryIteratorInner::PackedStandard(core) => core.next_match(),
            QueryIteratorInner::PackedOsa(core) => core.next_match(),
            QueryIteratorInner::PackedMergeSplit(core) => core.next_match(),
            QueryIteratorInner::Unit(core) => core.next_match(),
            QueryIteratorInner::Affine(core) => core.next_match(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<
        N: DictionaryNode,
        R: QueryResult<N::Unit>,
        P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    > std::iter::FusedIterator for QueryIterator<N, R, P>
{
}

#[inline]
fn static_packed_query_dispatch_disabled() -> bool {
    #[cfg(feature = "benchmark-controls")]
    {
        use std::sync::OnceLock;
        static DISABLED: OnceLock<bool> = OnceLock::new();
        *DISABLED.get_or_init(|| {
            std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_STATIC_PACKED_DISPATCH").is_some()
        })
    }
    #[cfg(not(feature = "benchmark-controls"))]
    {
        false
    }
}

// Type aliases for ergonomic use
/// Type alias for query iterator that returns just term strings.
///
/// Equivalent to `QueryIterator<N, String>`.
pub type StringQueryIterator<N> = QueryIterator<N, String>;

/// Type alias for query iterator that returns Candidate structs (term + distance).
///
/// Equivalent to `QueryIterator<N, Candidate>`.
pub type CandidateIterator<N> = QueryIterator<N, Candidate>;

/// Type alias for a query iterator that returns matched terms as raw unit
/// sequences (`Vec<N::Unit>`) — the units-native, lossless counterpart of
/// [`StringQueryIterator`] (e.g. `Vec<u64>` token-id sequences).
pub type UnitQueryIterator<N> = QueryIterator<N, Vec<<N as DictionaryNode>::Unit>>;

/// Type alias for a query iterator that returns [`UnitCandidate`] structs
/// (unit-sequence term + distance) — the units-native counterpart of
/// [`CandidateIterator`].
pub type UnitCandidateIterator<N> = QueryIterator<N, UnitCandidate<<N as DictionaryNode>::Unit>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::eviction::Noop;
    use crate::transducer::dictionary_traversal::{CursorPathTrace, ParentPathTrace, PathFrontier};
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::Dictionary;
    use libdictenstein::SnapshotTraversalCursor;

    #[test]
    fn unit_cost_queue_entries_remain_cache_compact() {
        assert!(
            std::mem::size_of::<
                PathFrontier<ParentPathTrace<char, SnapshotTraversalCursor>, UnitCostFrontier>,
            >() <= 32,
            "unit-cost traversal queue entries must not inherit affine State storage"
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            std::mem::size_of::<
                PathFrontier<CursorPathTrace<SnapshotTraversalCursor>, UnitCostFrontier>,
            >(),
            16,
            "cursor-native unit-cost entries contain only cursor and frontier"
        );
    }

    #[test]
    fn test_query_exact_match() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let query: QueryIterator<_, String> =
            QueryIterator::new(dict.root(), "test".to_string(), 0, Algorithm::Standard);

        let result: Vec<_> = query.collect();
        assert_eq!(result, vec!["test"]);
    }

    #[test]
    fn cursor_native_results_are_relative_to_a_descendant_root() {
        let dict = DoubleArrayTrie::from_terms(["car", "cat"]);
        let subtree = dict.root().transition(b'c').expect("c subtree");
        let results: Vec<String> =
            QueryIterator::new(subtree, "at".to_owned(), 0, Algorithm::Standard).collect();
        assert_eq!(results, vec!["at"]);
    }

    #[test]
    fn transparent_wrappers_preserve_cursor_native_key_capability() {
        let wrapped = Noop::new(Noop::new(DoubleArrayTrie::from_terms(["car", "cat"])));
        let root = wrapped.root();
        assert!(root.supports_snapshot_cursor_key_units());
        let results: Vec<String> =
            QueryIterator::new(root, "cat".to_owned(), 0, Algorithm::Standard).collect();
        assert_eq!(results, vec!["cat"]);
    }

    #[test]
    fn test_query_with_distance() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "testing"]);
        let query = QueryIterator::new(dict.root(), "test".to_string(), 1, Algorithm::Standard);

        let results: Vec<_> = query.collect();
        assert!(results.contains(&"test".to_string()));
        assert!(results.contains(&"best".to_string()));
        assert!(results.contains(&"rest".to_string()));
        // "testing" should not match with distance 1
    }

    #[test]
    fn test_candidate_iterator() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best"]);
        let query = CandidateIterator::new(dict.root(), "test".to_string(), 1, Algorithm::Standard);

        let candidates: Vec<_> = query.collect();
        assert!(candidates
            .iter()
            .any(|c| c.term == "test" && c.distance == 0));
        assert!(candidates
            .iter()
            .any(|c| c.term == "best" && c.distance == 1));
    }

    #[test]
    fn test_empty_query() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let query = QueryIterator::new(dict.root(), "".to_string(), 0, Algorithm::Standard);

        let results: Vec<_> = query.collect();
        // Empty query with distance 0 should match nothing unless dict has empty string
        assert!(results.is_empty() || results.contains(&"".to_string()));
    }

    #[test]
    fn test_query_reconstructs_term_past_initial_path_capacity() {
        let long_term = "a".repeat(96);
        let dict = DoubleArrayTrie::from_terms(vec![long_term.as_str()]);
        let query: QueryIterator<_, String> =
            QueryIterator::new(dict.root(), long_term.clone(), 0, Algorithm::Standard);

        let results: Vec<_> = query.collect();
        assert_eq!(results, vec![long_term]);
    }
}
