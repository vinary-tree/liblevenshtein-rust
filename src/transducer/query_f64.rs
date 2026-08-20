//! Lazy query iterators for float-weighted approximate string matching.
//!
//! This module provides `QueryIteratorF64`, which performs fuzzy queries
//! using configurable float costs per operation type.
//!
//! # Differences from Integer Query Iterator
//!
//! | Aspect | Integer (`query.rs`) | Float (`query_f64.rs`) |
//! |--------|----------------------|------------------------|
//! | Distance type | `usize` | `f64` |
//! | Threshold | `max_distance: usize` | `max_cost: f64` |
//! | Costs | Hardcoded `+1` | `OperationCostsF64` |
//! | Result | `Candidate { distance: usize }` | `CandidateF64 { distance: f64 }` |

use super::transition_f64::{initial_state_f64, CachedF64Transitions, TransitionSettingsF64};
use super::{
    Algorithm, OperationCostsF64, StateF64, StatePoolF64, SubstitutionPolicy,
    SubstitutionPolicyFor, Unrestricted,
};
use crate::transducer::dictionary_traversal::{
    CursorNativePath, ParentArenaPath, PathFrontier, ResultPathStrategy, TraversalCursor,
    TraversalSession,
};
use libdictenstein::{CharUnit, DictionaryNode, DictionaryTraversalRoot};
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Query result containing term and float distance.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateF64 {
    /// The matching term
    pub term: String,
    /// Edit distance from query (float)
    pub distance: f64,
}

/// Float-weighted query result containing the matched term as a raw unit sequence
/// and its float distance.
///
/// The units-native analogue of [`CandidateF64`]: `term` is the matched key as a
/// `Vec<U>` (e.g. a `Vec<u64>` token-id sequence) with no `String` reconstruction,
/// so it is lossless for `u64` token alphabets. Returned by
/// [`Transducer::query_units_weighted`](crate::transducer::Transducer::query_units_weighted).
#[derive(Debug, Clone, PartialEq)]
pub struct UnitCandidateF64<U: CharUnit> {
    /// The matching term as its unit sequence.
    pub term: Vec<U>,
    /// Weighted edit distance from the query (float).
    pub distance: f64,
}

/// Trait for converting a match (matched units + float distance) into a result type.
///
/// Similar to [`QueryResult`](super::query_result::QueryResult) but for float
/// distances. `String`/`CandidateF64` reconstruct the term text via
/// `CharUnit::to_string`; `Vec<U>`/`UnitCandidateF64` keep the units verbatim.
pub trait QueryResultF64<U: CharUnit>: Sized {
    /// Convert a match into the result type.
    fn from_match(units: &[U], distance: f64) -> Self;
}

/// Implementation for `String`: reconstructs the term text, ignoring distance
/// (lossy byte-unpack for `u64` token sequences — prefer `Vec<u64>`/`UnitCandidateF64`).
impl<U: CharUnit> QueryResultF64<U> for String {
    #[inline]
    fn from_match(units: &[U], _distance: f64) -> Self {
        U::to_string(units)
    }
}

/// Implementation for `CandidateF64`: reconstructs the term text plus float distance.
impl<U: CharUnit> QueryResultF64<U> for CandidateF64 {
    #[inline]
    fn from_match(units: &[U], distance: f64) -> Self {
        CandidateF64 {
            term: U::to_string(units),
            distance,
        }
    }
}

/// Implementation for `Vec<U>`: the matched term as its raw unit sequence.
impl<U: CharUnit> QueryResultF64<U> for Vec<U> {
    #[inline]
    fn from_match(units: &[U], _distance: f64) -> Self {
        units.to_vec()
    }
}

/// Implementation for [`UnitCandidateF64`]: units-native term plus float distance.
impl<U: CharUnit> QueryResultF64<U> for UnitCandidateF64<U> {
    #[inline]
    fn from_match(units: &[U], distance: f64) -> Self {
        UnitCandidateF64 {
            term: units.to_vec(),
            distance,
        }
    }
}

/// Lazy iterator over float-weighted query matches.
///
/// This iterator uses configurable float costs for edit operations and
/// returns results with float distances.
///
/// # Type Parameters
///
/// - `N`: Dictionary node type
/// - `R`: Result type (defaults to `String`). Must implement `QueryResultF64`.
/// - `P`: Substitution policy (defaults to `Unrestricted`)
///
/// # Performance
///
/// Uses `StatePoolF64` to eliminate `StateF64` cloning overhead during traversal.
/// The pool is created per-query and reuses `StateF64` allocations across
/// all transitions.
///
/// # Examples
///
/// ```
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::transducer::{QueryIteratorF64, CandidateF64, OperationCostsF64};
///
/// let dict = DoubleArrayTrie::from_terms(vec!["test", "best"]);
///
/// // Standard costs (all 1.0)
/// let costs = OperationCostsF64::standard();
///
/// // Returns CandidateF64 (term + float distance)
/// let iter: QueryIteratorF64<_, CandidateF64> = QueryIteratorF64::new(
///     dict.root(), "test".to_string(), 2.0, Algorithm::Standard, costs
/// );
/// for candidate in iter {
///     println!("{}: {:.2}", candidate.term, candidate.distance);
/// }
/// ```
pub struct QueryIteratorF64<N: DictionaryNode, R = String, P: SubstitutionPolicy = Unrestricted> {
    inner: PathQueryIteratorF64<N, R, P>,
}

enum PathQueryIteratorF64<N: DictionaryNode, R, P: SubstitutionPolicy> {
    Parent(QueryIteratorF64Core<N, R, P, ParentArenaPath>),
    Cursor(QueryIteratorF64Core<N, R, P, CursorNativePath>),
}

struct QueryIteratorF64Core<N, R, P, S>
where
    N: DictionaryNode,
    P: SubstitutionPolicy,
    S: ResultPathStrategy<N>,
{
    pending: VecDeque<PathFrontier<S::Trace, StateF64>>,
    traversal: TraversalSession<N>,
    query: Vec<N::Unit>,
    max_cost: f64,
    algorithm: Algorithm,
    costs: OperationCostsF64,
    policy: P,
    path_storage: S::Storage,
    finished: bool,
    state_pool: StatePoolF64,
    /// Shared weighted transition cache; queued states are epsilon-closed.
    unit_transitions: CachedF64Transitions<N::Unit>,
    substring_mode: bool,
    _result_type: PhantomData<R>,
    _path_strategy: PhantomData<fn() -> S>,
}

impl<N: DictionaryNode, R: QueryResultF64<N::Unit>> QueryIteratorF64<N, R, Unrestricted> {
    /// Create a new float-weighted query iterator with unrestricted policy.
    pub fn new(
        root: N,
        query: String,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
    ) -> Self {
        Self::with_substring_mode(root, query, max_cost, algorithm, costs, false)
    }

    /// Create a new float-weighted query iterator with substring matching mode.
    pub fn with_substring_mode(
        root: N,
        query: String,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
        substring_mode: bool,
    ) -> Self {
        Self::with_policy_and_substring(
            root,
            query,
            max_cost,
            algorithm,
            costs,
            Unrestricted,
            substring_mode,
        )
    }
}

impl<
        N: DictionaryNode,
        R: QueryResultF64<N::Unit>,
        P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    > QueryIteratorF64<N, R, P>
{
    /// Create a new float-weighted query iterator with custom substitution policy.
    pub fn with_policy(
        root: N,
        query: String,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
        policy: P,
    ) -> Self {
        Self::with_policy_and_substring(root, query, max_cost, algorithm, costs, policy, false)
    }

    /// Create a new float-weighted query iterator with custom policy and substring mode.
    pub fn with_policy_and_substring(
        root: N,
        query: String,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        // See `QueryIterator::with_policy_and_substring`: `from_str` byte-packs a
        // `u64` token query (lossy) — such callers must use [`with_units`](Self::with_units).
        let query_units = N::Unit::from_str(&query);
        Self::with_units(
            root,
            query_units,
            max_cost,
            algorithm,
            costs,
            policy,
            substring_mode,
        )
    }

    /// Create a float-weighted query iterator from a pre-computed unit sequence,
    /// bypassing the `&str` → units conversion (required for `u64` token dictionaries;
    /// see [`QueryIterator::with_units`](super::query::QueryIterator::with_units)).
    #[allow(clippy::too_many_arguments)]
    pub fn with_units(
        root: N,
        query_units: Vec<N::Unit>,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        Self::with_traversal_root_and_units(
            DictionaryTraversalRoot::owned(root),
            query_units,
            max_cost,
            algorithm,
            costs,
            policy,
            substring_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_traversal_root_and_units(
        root: DictionaryTraversalRoot<N>,
        query_units: Vec<N::Unit>,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        let initial = initial_state_f64(query_units.len(), max_cost, algorithm, &costs);
        let unit_transitions = CachedF64Transitions::new(query_units.len(), max_cost, &costs);

        let (traversal, root) = TraversalSession::capture(root);
        Self {
            inner: PathQueryIteratorF64::new(
                root,
                traversal,
                initial,
                query_units,
                max_cost,
                algorithm,
                costs,
                policy,
                unit_transitions,
                substring_mode,
            ),
        }
    }
}

impl<N, R, P> PathQueryIteratorF64<N, R, P>
where
    N: DictionaryNode,
    R: QueryResultF64<N::Unit>,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        initial: StateF64,
        query: Vec<N::Unit>,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
        policy: P,
        unit_transitions: CachedF64Transitions<N::Unit>,
        substring_mode: bool,
    ) -> Self {
        if traversal.supports_cursor_key_units() {
            Self::Cursor(QueryIteratorF64Core::new(
                root,
                traversal,
                initial,
                query,
                max_cost,
                algorithm,
                costs,
                policy,
                unit_transitions,
                substring_mode,
            ))
        } else {
            Self::Parent(QueryIteratorF64Core::new(
                root,
                traversal,
                initial,
                query,
                max_cost,
                algorithm,
                costs,
                policy,
                unit_transitions,
                substring_mode,
            ))
        }
    }

    #[inline]
    fn next_match(&mut self) -> Option<R> {
        match self {
            Self::Parent(core) => core.next_match(),
            Self::Cursor(core) => core.next_match(),
        }
    }
}

impl<N, R, P, S> QueryIteratorF64Core<N, R, P, S>
where
    N: DictionaryNode,
    R: QueryResultF64<N::Unit>,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    S: ResultPathStrategy<N>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        initial: StateF64,
        query: Vec<N::Unit>,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
        policy: P,
        unit_transitions: CachedF64Transitions<N::Unit>,
        substring_mode: bool,
    ) -> Self {
        let (mut pending, path_storage) = S::acquire_queue();
        pending.push_back(PathFrontier::new(S::root(root), initial));
        Self {
            pending,
            traversal,
            query,
            max_cost,
            algorithm,
            costs,
            policy,
            path_storage,
            finished: false,
            state_pool: StatePoolF64::new(),
            unit_transitions,
            substring_mode,
            _result_type: PhantomData,
            _path_strategy: PhantomData,
        }
    }

    #[inline]
    fn next_match(&mut self) -> Option<R> {
        if self.finished {
            None
        } else {
            self.advance()
        }
    }

    /// Advance to the next match
    fn advance(&mut self) -> Option<R> {
        while let Some(intersection) = self.pending.pop_front() {
            let is_final = self.queue_children_and_finality(&intersection);
            // Check if this is a final match
            if is_final {
                // Infer the distance based on matching mode
                let distance = if self.substring_mode {
                    &intersection
                        .frontier
                        .min_distance()
                        .unwrap_or(f64::INFINITY)
                } else {
                    &intersection
                        .frontier
                        .infer_distance(self.query.len(), self.costs.deletion)
                        .unwrap_or(f64::INFINITY)
                };

                if *distance <= self.max_cost + 1e-9 {
                    let units = S::materialize_units(
                        &intersection.trace,
                        &self.traversal,
                        &self.path_storage,
                    );
                    if !self.traversal.accepts_final_units(&units) {
                        continue;
                    }
                    return Some(R::from_match(&units, *distance));
                }
            }
        }

        self.finished = true;
        None
    }

    /// Queue child intersections for exploration
    fn queue_children_and_finality(
        &mut self,
        intersection: &PathFrontier<S::Trace, StateF64>,
    ) -> bool {
        let mut expansion = S::begin_expansion(&intersection.trace);
        let query = &self.query;
        let policy = &self.policy;
        let costs = &self.costs;
        let max_cost = self.max_cost;
        let algorithm = self.algorithm;
        let substring_mode = self.substring_mode;
        let unit_transitions = &mut self.unit_transitions;
        let state_pool = &mut self.state_pool;
        let path_storage = &mut self.path_storage;
        let pending = &mut self.pending;

        self.traversal.filter_map_edges_and_finality(
            S::position(&intersection.trace),
            |label| {
                unit_transitions.transition(
                    &intersection.frontier,
                    state_pool,
                    policy,
                    label,
                    query,
                    TransitionSettingsF64::new(max_cost, algorithm, costs, substring_mode),
                )
            },
            |label, child_position, next_state| {
                pending.push_back(PathFrontier::new(
                    S::child_trace(
                        &intersection.trace,
                        &mut expansion,
                        label,
                        child_position,
                        path_storage,
                    ),
                    next_state,
                ));
            },
        )
    }
}

impl<N, R, P, S> Drop for QueryIteratorF64Core<N, R, P, S>
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
        R: QueryResultF64<N::Unit>,
        P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    > Iterator for QueryIteratorF64<N, R, P>
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_match()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<
        N: DictionaryNode,
        R: QueryResultF64<N::Unit>,
        P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    > std::iter::FusedIterator for QueryIteratorF64<N, R, P>
{
}

/// Type alias for float-weighted query iterator that returns just term strings.
pub type StringQueryIteratorF64<N> = QueryIteratorF64<N, String>;

/// Type alias for float-weighted query iterator that returns CandidateF64 structs.
pub type CandidateIteratorF64<N> = QueryIteratorF64<N, CandidateF64>;

/// Type alias for a float-weighted query iterator returning matched terms as raw
/// unit sequences (`Vec<N::Unit>`) — the units-native, lossless counterpart of
/// [`StringQueryIteratorF64`].
pub type UnitQueryIteratorF64<N> = QueryIteratorF64<N, Vec<<N as DictionaryNode>::Unit>>;

/// Type alias for a float-weighted query iterator returning [`UnitCandidateF64`]
/// structs (unit-sequence term + float distance) — the units-native counterpart of
/// [`CandidateIteratorF64`].
pub type UnitCandidateIteratorF64<N> =
    QueryIteratorF64<N, UnitCandidateF64<<N as DictionaryNode>::Unit>>;

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::Dictionary;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_query_exact_match() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let costs = OperationCostsF64::standard();
        let query: QueryIteratorF64<_, String> = QueryIteratorF64::new(
            dict.root(),
            "test".to_string(),
            0.0,
            Algorithm::Standard,
            costs,
        );

        let result: Vec<_> = query.collect();
        assert_eq!(result, vec!["test"]);
    }

    #[test]
    fn test_query_with_distance() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "testing"]);
        let costs = OperationCostsF64::standard();
        let query = QueryIteratorF64::new(
            dict.root(),
            "test".to_string(),
            1.0,
            Algorithm::Standard,
            costs,
        );

        let results: Vec<_> = query.collect();
        assert!(results.contains(&"test".to_string()));
        assert!(results.contains(&"best".to_string()));
        assert!(results.contains(&"rest".to_string()));
    }

    #[test]
    fn test_shared_prefix_branching_reconstructs_terms() {
        let dict = DoubleArrayTrie::from_terms(vec!["cart", "care", "cars"]);
        let costs = OperationCostsF64::standard();
        let query: QueryIteratorF64<_, String> = QueryIteratorF64::new(
            dict.root(),
            "car".to_string(),
            1.0,
            Algorithm::Standard,
            costs,
        );

        let mut results: Vec<_> = query.collect();
        results.sort();

        assert_eq!(results, vec!["care", "cars", "cart"]);
    }

    #[test]
    fn test_candidate_iterator_f64() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best"]);
        let costs = OperationCostsF64::standard();
        let query = CandidateIteratorF64::new(
            dict.root(),
            "test".to_string(),
            1.0,
            Algorithm::Standard,
            costs,
        );

        let candidates: Vec<_> = query.collect();
        assert!(candidates
            .iter()
            .any(|c| c.term == "test" && c.distance.abs() < EPSILON));
        assert!(candidates
            .iter()
            .any(|c| c.term == "best" && (c.distance - 1.0).abs() < EPSILON));
    }

    #[test]
    fn test_custom_costs() {
        let dict = DoubleArrayTrie::from_terms(vec!["abc", "axc"]);
        // Make substitution expensive (2.0)
        let costs = OperationCostsF64::custom(2.0, 1.0, 1.0, 1.0, 1.0, 1.0);

        let query = CandidateIteratorF64::new(
            dict.root(),
            "abc".to_string(),
            1.5, // Only allows insertion/deletion, not substitution
            Algorithm::Standard,
            costs,
        );

        let candidates: Vec<_> = query.collect();
        // "abc" should match exactly (distance 0)
        assert!(candidates.iter().any(|c| c.term == "abc"));
        // "axc" requires substitution (cost 2.0) which exceeds threshold
        assert!(!candidates.iter().any(|c| c.term == "axc"));
    }

    #[test]
    fn test_typo_friendly_transposition() {
        let dict = DoubleArrayTrie::from_terms(vec!["the", "teh"]);
        let costs = OperationCostsF64::typo_friendly();

        // Transposition cost is 0.5 in typo_friendly, so threshold of 1.0 captures it
        let query = CandidateIteratorF64::new(
            dict.root(),
            "the".to_string(),
            1.0, // Threshold covers transposition cost of 0.5
            Algorithm::Transposition,
            costs,
        );

        let candidates: Vec<_> = query.collect();
        // "the" should match exactly (distance 0)
        assert!(candidates
            .iter()
            .any(|c| c.term == "the" && c.distance.abs() < EPSILON));
        // "teh" requires transposition (cost 0.5 in typo_friendly)
        assert!(candidates
            .iter()
            .any(|c| c.term == "teh" && (c.distance - 0.5).abs() < EPSILON));
    }

    #[test]
    fn test_standard_costs_equivalent_to_integer() {
        let dict = DoubleArrayTrie::from_terms(vec!["cat", "car", "bat", "bar"]);
        let costs = OperationCostsF64::standard();

        let query = CandidateIteratorF64::new(
            dict.root(),
            "cat".to_string(),
            1.0,
            Algorithm::Standard,
            costs,
        );

        let candidates: Vec<_> = query.collect();

        // With standard costs, should behave same as integer Levenshtein
        // cat -> cat: distance 0
        // cat -> car: distance 1 (substitute t->r)
        // cat -> bat: distance 1 (substitute c->b)
        // cat -> bar: distance 2 (too far)

        assert!(candidates.iter().any(|c| c.term == "cat"));
        assert!(candidates.iter().any(|c| c.term == "car"));
        assert!(candidates.iter().any(|c| c.term == "bat"));
        assert!(!candidates.iter().any(|c| c.term == "bar"));
    }

    #[test]
    fn test_empty_query() {
        let dict = DoubleArrayTrie::from_terms(vec!["a", "ab"]);
        let costs = OperationCostsF64::standard();

        let query =
            CandidateIteratorF64::new(dict.root(), "".to_string(), 1.0, Algorithm::Standard, costs);

        let candidates: Vec<_> = query.collect();
        // Empty query with max_cost 1.0 should match single-char words
        assert!(candidates.iter().any(|c| c.term == "a"));
        // "ab" requires 2 insertions, exceeds threshold
        assert!(!candidates.iter().any(|c| c.term == "ab"));
    }
}
