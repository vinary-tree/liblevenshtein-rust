//! Lazy query iterators for approximate string matching.

use super::dictionary_traversal::TraversalSession;
use super::query_result::QueryResult;
use super::state::State;
use super::transition::{
    initial_state_affine, AffineTransitionSettings, FinishMode, TransitionSettings,
    UnitCostFrontier, UnitCostMachine,
};
use super::variants::{AffineGapParams, AffineV};
#[cfg(feature = "perf-instrumentation")]
use super::Position;
use super::{Algorithm, StatePool, SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted};
use libdictenstein::{CharUnit, DictionaryNode, DictionaryTraversalRoot, SnapshotTraversalCursor};
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
        self.inner.size_hint()
    }
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

const NO_PATH: usize = usize::MAX;

struct QueryPathNode<U: CharUnit> {
    label: U,
    depth: usize,
    parent: usize,
}

struct QueryIntersection<U: CharUnit, F> {
    label: Option<U>,
    position: SnapshotTraversalCursor,
    state: F,
    parent: usize,
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
    type Frontier = State;

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
            .transition_affine(
                frontier,
                state_pool,
                policy,
                label,
                query,
                AffineTransitionSettings::new(max_distance, self.params, substring_mode),
            )
            .map(|frontier| QueryStep {
                #[cfg(feature = "perf-instrumentation")]
                position_count: frontier.len(),
                #[cfg(feature = "perf-instrumentation")]
                storage_bytes: frontier
                    .len()
                    .saturating_mul(std::mem::size_of::<Position>()),
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
        if substring_mode {
            frontier.min_distance()
        } else {
            frontier.infer_distance_with::<AffineV>(query_length, self.params)
        }
    }
}

impl<U: CharUnit, F> QueryIntersection<U, F> {
    #[inline]
    fn new(position: SnapshotTraversalCursor, state: F) -> Self {
        Self {
            label: None,
            position,
            state,
            parent: NO_PATH,
        }
    }

    #[inline]
    fn with_parent(label: U, position: SnapshotTraversalCursor, state: F, parent: usize) -> Self {
        Self {
            label: Some(label),
            position,
            state,
            parent,
        }
    }

    /// Reconstruct the matched term as its raw unit sequence (root → this node).
    ///
    /// Result types convert this via [`QueryResult::from_match`]; `String`/`Candidate`
    /// apply `CharUnit::to_string`, while `Vec<U>`/`UnitCandidate` keep the units
    /// verbatim (lossless for `u64` token sequences).
    fn units(&self, path_arena: &[QueryPathNode<U>]) -> Vec<U> {
        let parent_depth = if self.parent == NO_PATH {
            0
        } else {
            path_arena[self.parent].depth
        };
        let capacity = parent_depth + usize::from(self.label.is_some());
        let mut units = Vec::with_capacity(capacity);

        if let Some(label) = self.label {
            units.push(label);
        }

        let mut current = self.parent;
        while current != NO_PATH {
            let node = &path_arena[current];
            units.push(node.label);
            current = node.parent;
        }

        units.reverse();
        units
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
    Unit(QueryIteratorCore<N, R, P, UnitCostQueryKernel<N::Unit>, UnitCostFrontier>),
    Affine(QueryIteratorCore<N, R, P, AffineQueryKernel<N::Unit>, State>),
}

struct QueryIteratorCore<N, R, P, K, F>
where
    N: DictionaryNode,
    P: SubstitutionPolicy,
{
    pending: VecDeque<QueryIntersection<N::Unit, F>>,
    traversal: TraversalSession<N>,
    query: Vec<N::Unit>,
    max_distance: usize,
    policy: P, // Substitution policy for matching
    path_arena: Vec<QueryPathNode<N::Unit>>,
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
        Self {
            inner: QueryIteratorInner::Unit(QueryIteratorCore::new(
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
            )),
        }
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
        Self::with_affine_units(root, query_units, max_cost, params, policy, substring_mode)
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
        let initial = initial_state_affine(query_units.len(), max_cost, params);
        let transitions = UnitCostMachine::unseeded_positional(query_units.len(), max_cost);
        let (traversal, root) = TraversalSession::capture(DictionaryTraversalRoot::owned(root));
        Self {
            inner: QueryIteratorInner::Affine(QueryIteratorCore::new(
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

impl<N, R, P, K, F> QueryIteratorCore<N, R, P, K, F>
where
    N: DictionaryNode,
    R: QueryResult<N::Unit>,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    K: QueryKernel<N::Unit, P, Frontier = F>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        root: SnapshotTraversalCursor,
        traversal: TraversalSession<N>,
        frontier: F,
        query: Vec<N::Unit>,
        max_distance: usize,
        policy: P,
        substring_mode: bool,
        kernel: K,
    ) -> Self {
        let mut pending = VecDeque::new();
        pending.push_back(QueryIntersection::new(root, frontier));
        Self {
            pending,
            traversal,
            query,
            max_distance,
            policy,
            path_arena: Vec::with_capacity(64),
            finished: false,
            state_pool: StatePool::new(),
            kernel,
            #[cfg(feature = "perf-instrumentation")]
            generated_products: FxHashSet::default(),
            substring_mode,
            _result_type: PhantomData,
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
                    .finish_distance(&intersection.state, self.query.len(), self.substring_mode)
                    .unwrap_or(usize::MAX);

                if distance <= self.max_distance {
                    let units = intersection.units(&self.path_arena);

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
    fn queue_children_and_finality(
        &mut self,
        intersection: &QueryIntersection<N::Unit, F>,
    ) -> bool {
        if let Some(_state) = self.kernel.unit_frontier(&intersection.state) {
            crate::causal_perf::record_generated_product_expansions(1);
            #[cfg(feature = "perf-instrumentation")]
            if let Some(node) = self.traversal.product_identity(intersection.position) {
                crate::causal_perf::record_generated_product_identity_expansions(1);
                if self.generated_products.insert((node, _state)) {
                    crate::causal_perf::record_generated_product_unique_expansions(1);
                } else {
                    crate::causal_perf::record_generated_product_repeated_expansions(1);
                }
            }
        }
        let mut child_parent_path = None;
        let max_distance = self.max_distance;
        let substring_mode = self.substring_mode;
        let query = &self.query;
        let policy = &self.policy;
        let kernel = &mut self.kernel;
        let state_pool = &mut self.state_pool;
        let pending = &mut self.pending;
        let path_arena = &mut self.path_arena;
        let current_label = intersection.label;
        let current_parent = intersection.parent;

        self.traversal.filter_map_edges_and_finality(
            intersection.position,
            |label| {
                crate::causal_perf::record_edges_enumerated(1);
                kernel.step(
                    &intersection.state,
                    state_pool,
                    policy,
                    label,
                    query,
                    max_distance,
                    substring_mode,
                )
            },
            |label, child_position, step| {
                crate::causal_perf::record_transition_accepted(1);
                #[cfg(feature = "perf-instrumentation")]
                {
                    crate::causal_perf::record_state_positions_enqueued(step.position_count as u64);
                    crate::causal_perf::record_state_bytes_enqueued(step.storage_bytes as u64);
                }
                let parent_path = match child_parent_path {
                    Some(path) => path,
                    None => {
                        let path = match current_label {
                            Some(label) => {
                                let depth = if current_parent == NO_PATH {
                                    1
                                } else {
                                    path_arena[current_parent].depth.saturating_add(1)
                                };
                                let index = path_arena.len();
                                path_arena.push(QueryPathNode {
                                    label,
                                    depth,
                                    parent: current_parent,
                                });
                                index
                            }
                            None => NO_PATH,
                        };
                        child_parent_path = Some(path);
                        path
                    }
                };

                pending.push_back(QueryIntersection::with_parent(
                    label,
                    child_position,
                    step.frontier,
                    parent_path,
                ));
                crate::causal_perf::record_pending_queue_size(pending.len());
            },
        )
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
            QueryIteratorInner::Unit(core) => {
                if core.finished {
                    None
                } else {
                    core.advance()
                }
            }
            QueryIteratorInner::Affine(core) => {
                if core.finished {
                    None
                } else {
                    core.advance()
                }
            }
        }
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
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::Dictionary;

    #[test]
    fn unit_cost_queue_entries_remain_cache_compact() {
        assert!(
            std::mem::size_of::<QueryIntersection<char, UnitCostFrontier>>() <= 32,
            "unit-cost traversal queue entries must not inherit affine State storage"
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
