//! Priority queue-based query iterator using A* search.
//!
//! This module provides an alternative to `OrderedQueryIterator` that uses
//! A* search with heuristics for potentially faster result production.
//!
//! # Comparison with OrderedQueryIterator
//!
//! | Feature | OrderedQueryIterator | PriorityQueryIterator |
//! |---------|---------------------|-----------------------|
//! | Algorithm | BFS with distance buckets | A* with priority queue |
//! | Ordering | Strict distance + lexicographic | Distance-first (approximate lex) |
//! | Memory | O(states per level) | O(all pending states) |
//! | Best for | Exact ordering requirements | Fast first-k results |
//!
//! # A* Heuristic
//!
//! The heuristic estimates remaining cost as:
//! - Characters remaining in query that must be consumed
//! - Minimum possible operations to reach a final state
//!
//! ```text
//! h(state) = max(0, query_len - max_consumed_chars)
//! f(state) = g(state) + h(state)
//! ```
//!
//! # Example
//!
//! ```rust
//! use libdictenstein::{dynamic_dawg::DynamicDawg, Dictionary};
//! use liblevenshtein::transducer::{Algorithm, PriorityQueryIterator};
//!
//! let dict = DynamicDawg::<()>::new();
//! for term in ["apple", "apply", "banana"] {
//!     dict.insert(term);
//! }
//! let iter = PriorityQueryIterator::new(dict.root(), "aple", 2, Algorithm::Standard);
//!
//! let candidates: Vec<_> = iter.take(3).collect();
//! assert!(candidates.iter().all(|candidate| candidate.distance <= 2));
//! assert!(candidates.iter().any(|candidate| candidate.term == "apple"));
//! ```

use super::transition::{
    with_prepared_unit_cost_row, FinishMode, PreparedUnitCostRow, TransitionSettings,
    UnitCostFrontier, UnitCostMachine,
};
use super::{Algorithm, StatePool, SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted};
use crate::transducer::dictionary_traversal::{
    compare_parent_paths, materialize_parent_path, new_path_arena, push_parent_path, ContextHeap,
    ParentPathKey, ParentPathNode, TraversalCursor, TraversalSession,
};
use libdictenstein::{CharUnit, DictionaryNode, DictionaryTraversalRoot};
use std::cmp::Ordering;
#[cfg(feature = "benchmark-controls")]
use std::collections::BinaryHeap;

/// Dictionary node paired with the current Levenshtein automaton state.
struct PriorityIntersection<N: DictionaryNode> {
    position: TraversalCursor<N::SnapshotCursor>,
    state: UnitCostFrontier,
}

impl<N: DictionaryNode> PriorityIntersection<N> {
    #[inline]
    fn new(position: TraversalCursor<N::SnapshotCursor>, state: UnitCostFrontier) -> Self {
        Self { position, state }
    }
}

/// Entry in the priority queue for A* search.
struct SearchEntry<N: DictionaryNode, K> {
    /// Current dictionary node and automaton state.
    intersection: PriorityIntersection<N>,
    /// Constant-size key into the query-local parent-path arena.
    path: K,
    /// Actual cost so far (minimum errors in state)
    g_cost: usize,
    /// f-cost = g-cost + heuristic, used for priority ordering
    f_cost: usize,
}

impl<N: DictionaryNode, K> SearchEntry<N, K> {
    fn new(intersection: PriorityIntersection<N>, path: K, g_cost: usize, h_cost: usize) -> Self {
        Self {
            intersection,
            path,
            g_cost,
            f_cost: g_cost.saturating_add(h_cost),
        }
    }
}

/// Preserve the historical `BinaryHeap<SearchEntry>` ordering exactly while
/// comparing compact parent-path keys against shared arena storage.
#[inline]
fn compare_arena_search_entries<N: DictionaryNode>(
    paths: &[ParentPathNode<N::Unit>],
    left: &SearchEntry<N, ArenaPath<N::Unit>>,
    right: &SearchEntry<N, ArenaPath<N::Unit>>,
) -> Ordering {
    match right.f_cost.cmp(&left.f_cost) {
        Ordering::Equal => match right.g_cost.cmp(&left.g_cost) {
            Ordering::Equal => match left.path.first.cmp(&right.path.first) {
                Ordering::Equal => compare_parent_paths(paths, left.path.key, right.path.key),
                ordering => ordering,
            },
            ordering => ordering,
        },
        ordering => ordering,
    }
}

trait PriorityPathStrategy<N: DictionaryNode>: 'static {
    type Path;
    type Queue;
    type Storage;

    fn new_queue() -> Self::Queue;
    fn new_storage() -> Self::Storage;
    fn root() -> Self::Path;
    fn child(storage: &mut Self::Storage, parent: &Self::Path, label: N::Unit) -> Self::Path;
    fn push(queue: &mut Self::Queue, storage: &Self::Storage, entry: SearchEntry<N, Self::Path>);
    fn pop(queue: &mut Self::Queue, storage: &Self::Storage) -> Option<SearchEntry<N, Self::Path>>;
    fn with_units<R>(
        storage: &Self::Storage,
        path: &Self::Path,
        operation: impl FnOnce(&[N::Unit]) -> R,
    ) -> R;
}

struct ArenaPriorityPath;

#[derive(Clone, Copy)]
struct ArenaPath<U: CharUnit> {
    key: ParentPathKey,
    /// Resolves comparisons across root branches without walking either
    /// parent chain; equal first units retain the exact arena fallback.
    first: Option<U>,
}

impl<N: DictionaryNode> PriorityPathStrategy<N> for ArenaPriorityPath {
    type Path = ArenaPath<N::Unit>;
    type Queue = ContextHeap<SearchEntry<N, Self::Path>>;
    type Storage = Vec<ParentPathNode<N::Unit>>;

    fn new_queue() -> Self::Queue {
        ContextHeap::with_capacity(64)
    }

    fn new_storage() -> Self::Storage {
        new_path_arena()
    }

    fn root() -> Self::Path {
        ArenaPath {
            key: ParentPathKey::ROOT,
            first: None,
        }
    }

    fn child(storage: &mut Self::Storage, parent: &Self::Path, label: N::Unit) -> Self::Path {
        ArenaPath {
            key: push_parent_path(storage, parent.key, label),
            first: parent.first.or(Some(label)),
        }
    }

    fn push(queue: &mut Self::Queue, storage: &Self::Storage, entry: SearchEntry<N, Self::Path>) {
        queue.push_by(entry, |left, right| {
            compare_arena_search_entries(storage, left, right)
        });
    }

    fn pop(queue: &mut Self::Queue, storage: &Self::Storage) -> Option<SearchEntry<N, Self::Path>> {
        queue.pop_by(|left, right| compare_arena_search_entries(storage, left, right))
    }

    fn with_units<R>(
        storage: &Self::Storage,
        path: &Self::Path,
        operation: impl FnOnce(&[N::Unit]) -> R,
    ) -> R {
        let units = materialize_parent_path(storage, path.key);
        operation(&units)
    }
}

#[cfg(feature = "benchmark-controls")]
struct LegacyPriorityPath;

#[cfg(feature = "benchmark-controls")]
struct LegacySearchEntry<N: DictionaryNode>(SearchEntry<N, Vec<N::Unit>>);

#[cfg(feature = "benchmark-controls")]
impl<N: DictionaryNode> Ord for LegacySearchEntry<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        let left = &self.0;
        let right = &other.0;
        match right.f_cost.cmp(&left.f_cost) {
            Ordering::Equal => match right.g_cost.cmp(&left.g_cost) {
                Ordering::Equal => left.path.cmp(&right.path),
                ordering => ordering,
            },
            ordering => ordering,
        }
    }
}

#[cfg(feature = "benchmark-controls")]
impl<N: DictionaryNode> PartialOrd for LegacySearchEntry<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "benchmark-controls")]
impl<N: DictionaryNode> PartialEq for LegacySearchEntry<N> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

#[cfg(feature = "benchmark-controls")]
impl<N: DictionaryNode> Eq for LegacySearchEntry<N> {}

#[cfg(feature = "benchmark-controls")]
impl<N: DictionaryNode> PriorityPathStrategy<N> for LegacyPriorityPath {
    type Path = Vec<N::Unit>;
    type Queue = BinaryHeap<LegacySearchEntry<N>>;
    type Storage = ();

    fn new_queue() -> Self::Queue {
        BinaryHeap::with_capacity(64)
    }

    fn new_storage() -> Self::Storage {}

    fn root() -> Self::Path {
        Vec::new()
    }

    fn child(_storage: &mut Self::Storage, parent: &Self::Path, label: N::Unit) -> Self::Path {
        let mut child = parent.clone();
        child.push(label);
        child
    }

    fn push(queue: &mut Self::Queue, _storage: &Self::Storage, entry: SearchEntry<N, Self::Path>) {
        queue.push(LegacySearchEntry(entry));
    }

    fn pop(
        queue: &mut Self::Queue,
        _storage: &Self::Storage,
    ) -> Option<SearchEntry<N, Self::Path>> {
        queue.pop().map(|entry| entry.0)
    }

    fn with_units<R>(
        _storage: &Self::Storage,
        path: &Self::Path,
        operation: impl FnOnce(&[N::Unit]) -> R,
    ) -> R {
        operation(path)
    }
}

/// Priority queue-based query result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriorityCandidate {
    /// The matching term
    pub term: String,
    /// Edit distance from query
    pub distance: usize,
}

/// A* search iterator for approximate string matching.
///
/// Uses a priority queue with f-cost (g + h) ordering to efficiently
/// find matches in approximately distance order.
///
/// # Type Parameters
///
/// * `N` - Dictionary node type
pub struct PriorityQueryIterator<N: DictionaryNode, P: SubstitutionPolicy = Unrestricted> {
    inner: PriorityQueryMode<N, P>,
}

enum PriorityQueryMode<N: DictionaryNode, P: SubstitutionPolicy> {
    Arena(PriorityQueryCore<N, P, ArenaPriorityPath>),
    #[cfg(feature = "benchmark-controls")]
    Legacy(PriorityQueryCore<N, P, LegacyPriorityPath>),
}

struct PriorityQueryCore<N, P, S>
where
    N: DictionaryNode,
    P: SubstitutionPolicy,
    S: PriorityPathStrategy<N>,
{
    /// Priority queue ordered by f-cost
    queue: S::Queue,
    /// Statically selected parent-path representation and storage.
    path_storage: S::Storage,
    /// Retained snapshot owner and cursor traversal backend.
    traversal: TraversalSession<N>,
    /// Query units (bytes or chars)
    query: Vec<N::Unit>,
    /// Query length for heuristic computation
    query_len: usize,
    /// Maximum distance to explore
    max_distance: usize,
    /// Levenshtein algorithm
    algorithm: Algorithm,
    /// Substitution policy for matching
    policy: P,
    /// State pool for allocation reuse
    state_pool: StatePool,
    /// Shared cached transition kernel; queued states are epsilon-closed.
    unit_transitions: UnitCostMachine<N::Unit>,
}

impl<N: DictionaryNode> PriorityQueryIterator<N, Unrestricted> {
    /// Create a new priority query iterator.
    ///
    /// # Arguments
    ///
    /// * `root` - Root node of the dictionary
    /// * `query` - Query string
    /// * `max_distance` - Maximum edit distance to consider
    /// * `algorithm` - Levenshtein algorithm variant
    ///
    /// # Example
    ///
    /// ```rust
    /// use libdictenstein::{dynamic_dawg::DynamicDawg, Dictionary};
    /// use liblevenshtein::transducer::{Algorithm, PriorityQueryIterator};
    ///
    /// let dict = DynamicDawg::<()>::new();
    /// dict.insert("test");
    /// dict.insert("best");
    /// let iter = PriorityQueryIterator::new(dict.root(), "test", 2, Algorithm::Standard);
    /// let results: Vec<_> = iter.collect();
    /// assert_eq!(results[0].term, "test");
    /// assert_eq!(results[0].distance, 0);
    /// ```
    pub fn new(root: N, query: &str, max_distance: usize, algorithm: Algorithm) -> Self {
        Self::with_policy(root, query, max_distance, algorithm, Unrestricted)
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    PriorityQueryIterator<N, P>
{
    /// Create a new priority query iterator with a custom substitution policy.
    pub fn with_policy(
        root: N,
        query: &str,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
    ) -> Self {
        #[cfg(feature = "benchmark-controls")]
        if legacy_priority_paths_requested() {
            return Self {
                inner: PriorityQueryMode::Legacy(PriorityQueryCore::new(
                    root,
                    query,
                    max_distance,
                    algorithm,
                    policy,
                )),
            };
        }

        Self {
            inner: PriorityQueryMode::Arena(PriorityQueryCore::new(
                root,
                query,
                max_distance,
                algorithm,
                policy,
            )),
        }
    }
}

impl<N, P, S> PriorityQueryCore<N, P, S>
where
    N: DictionaryNode,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    S: PriorityPathStrategy<N>,
{
    fn new(root: N, query: &str, max_distance: usize, algorithm: Algorithm, policy: P) -> Self {
        let query_units = N::Unit::from_str(query);
        let query_len = query_units.len();
        let settings = TransitionSettings::new(max_distance, algorithm, false);
        let (unit_transitions, initial) = UnitCostMachine::seeded::<P>(&query_units, settings);

        let mut queue = S::new_queue();
        let path_storage = S::new_storage();

        let (traversal, root) = TraversalSession::capture(DictionaryTraversalRoot::owned(root));

        // Initialize with root node
        let root_intersection = PriorityIntersection::new(root, initial);
        let g_cost = 0; // No errors yet at root
        let h_cost = query_len; // Must consume all query characters

        S::push(
            &mut queue,
            &path_storage,
            SearchEntry::new(root_intersection, S::root(), g_cost, h_cost),
        );

        Self {
            queue,
            path_storage,
            traversal,
            query: query_units,
            query_len,
            max_distance,
            algorithm,
            policy,
            state_pool: StatePool::new(),
            unit_transitions,
        }
    }

    /// Get the next match from the priority queue.
    fn advance(&mut self) -> Option<PriorityCandidate> {
        while let Some(entry) = S::pop(&mut self.queue, &self.path_storage) {
            let is_final = self.expand_children_and_finality(&entry);
            // Check if this is a final match
            if is_final {
                let distance = self
                    .unit_transitions
                    .finish_distance(
                        entry.intersection.state,
                        FinishMode::Complete,
                        self.query_len,
                    )
                    .unwrap_or(usize::MAX);

                if distance <= self.max_distance {
                    let candidate = S::with_units(&self.path_storage, &entry.path, |term_units| {
                        self.traversal
                            .accepts_final_units(term_units)
                            .then(|| PriorityCandidate {
                                term: N::Unit::to_string(term_units),
                                distance,
                            })
                    });
                    if candidate.is_some() {
                        return candidate;
                    }
                }
            }
        }

        None
    }

    /// Expand children of a search entry into the priority queue.
    #[inline]
    fn expand_children_and_finality(&mut self, entry: &SearchEntry<N, S::Path>) -> bool {
        let query_len = self.query_len;
        let max_distance = self.max_distance;
        let algorithm = self.algorithm;
        let query = &self.query;
        let policy = &self.policy;
        let state_pool = &mut self.state_pool;
        let unit_transitions = &mut self.unit_transitions;
        let queue = &mut self.queue;
        let path_storage = &mut self.path_storage;
        let settings = TransitionSettings::new(max_distance, algorithm, false);
        with_prepared_unit_cost_row!(
            unit_transitions,
            entry.intersection.state,
            state_pool,
            policy,
            query,
            settings,
            |row| self.traversal.filter_map_edges_and_finality(
                entry.intersection.position,
                |label| {
                    let next_state = row.step(label)?;
                    let g_cost = row.min_distance(next_state).unwrap_or(0);
                    if g_cost > max_distance {
                        return None;
                    }
                    let max_consumed = row.max_consumed(next_state);
                    let h_cost = query_len.saturating_sub(max_consumed);
                    Some((next_state, g_cost, h_cost))
                },
                |label, child_position, (next_state, g_cost, h_cost)| {
                    let child_intersection = PriorityIntersection::new(child_position, next_state);
                    let child_path = S::child(path_storage, &entry.path, label);
                    S::push(
                        queue,
                        path_storage,
                        SearchEntry::new(child_intersection, child_path, g_cost, h_cost),
                    );
                },
            )
        )
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>> Iterator
    for PriorityQueryIterator<N, P>
{
    type Item = PriorityCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            PriorityQueryMode::Arena(core) => core.advance(),
            #[cfg(feature = "benchmark-controls")]
            PriorityQueryMode::Legacy(core) => core.advance(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    std::iter::FusedIterator for PriorityQueryIterator<N, P>
{
}

#[cfg(feature = "benchmark-controls")]
fn legacy_priority_paths_requested() -> bool {
    use std::sync::OnceLock;
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_USE_CLONED_PRIORITY_PATHS").is_some()
    })
}

/// Convenience function to create a priority query iterator.
pub fn priority_query<N: DictionaryNode>(
    root: N,
    query: &str,
    max_distance: usize,
    algorithm: Algorithm,
) -> PriorityQueryIterator<N> {
    PriorityQueryIterator::new(root, query, max_distance, algorithm)
}

/// Convenience function to create a priority query iterator with a custom policy.
pub fn priority_query_with_policy<N, P>(
    root: N,
    query: &str,
    max_distance: usize,
    algorithm: Algorithm,
    policy: P,
) -> PriorityQueryIterator<N, P>
where
    N: DictionaryNode,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
{
    PriorityQueryIterator::with_policy(root, query, max_distance, algorithm, policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::{OwnedRestricted, SubstitutionSet};
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::Dictionary;

    fn test_dict() -> libdictenstein::dynamic_dawg::DynamicDawg {
        let dawg = libdictenstein::dynamic_dawg::DynamicDawg::new();
        for term in ["apple", "apply", "appeal", "banana", "test", "best", "rest"] {
            dawg.insert(term);
        }
        dawg
    }

    #[test]
    fn test_exact_match() {
        let dict = test_dict();
        let mut iter = PriorityQueryIterator::new(dict.root(), "apple", 2, Algorithm::Standard);

        let first = iter.next();
        assert!(first.is_some());
        let candidate = first.expect("test fixture: first candidate exists (asserted above)");
        assert_eq!(candidate.term, "apple");
        assert_eq!(candidate.distance, 0);
    }

    #[test]
    fn test_close_matches() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "aple", 2, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Should find "apple" and "apply" within distance 2
        let terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();
        assert!(
            terms.contains(&"apple"),
            "Should contain 'apple': {:?}",
            terms
        );
    }

    #[test]
    fn test_distance_ordering() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "test", 2, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Check that results are roughly ordered by distance
        // (A* doesn't guarantee strict ordering, but close matches should come first)
        if results.len() >= 2 {
            let first_dist = results[0].distance;
            // First result should be the exact match or very close
            assert!(
                first_dist <= 1,
                "First result distance should be <= 1, got {}",
                first_dist
            );
        }
    }

    #[test]
    fn test_max_distance_respected() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "xyz", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // All results should be within max_distance
        for candidate in &results {
            assert!(
                candidate.distance <= 1,
                "Distance {} exceeds max 1 for term {}",
                candidate.distance,
                candidate.term
            );
        }
    }

    #[test]
    fn test_empty_query() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "", 3, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Empty query matches short terms within distance 3
        // All 3-letter terms should match
        let terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();
        // May or may not have results depending on dictionary
        // Just ensure no panic
        let _ = terms;
    }

    #[test]
    fn test_shared_prefix_branching_uses_cached_term_units() {
        let dict = DoubleArrayTrie::from_terms(vec!["cart", "care", "cars"]);
        let iter = PriorityQueryIterator::new(dict.root(), "car", 1, Algorithm::Standard);

        let mut results: Vec<_> = iter.into_iter().map(|candidate| candidate.term).collect();
        results.sort();

        assert_eq!(results, vec!["care", "cars", "cart"]);
    }

    #[cfg(feature = "benchmark-controls")]
    #[test]
    fn arena_paths_preserve_legacy_priority_order_exactly() {
        let dictionary = DoubleArrayTrie::from_terms([
            "a",
            "aa",
            "ab",
            "aba",
            "abb",
            "b",
            "ba",
            "care",
            "cars",
            "cart",
            "long-shared-prefix-a",
            "long-shared-prefix-b",
        ]);
        for (query, distance) in [("a", 2), ("car", 2), ("long-shared-prefix", 2)] {
            let mut arena = PriorityQueryCore::<_, _, ArenaPriorityPath>::new(
                dictionary.root(),
                query,
                distance,
                Algorithm::Standard,
                Unrestricted,
            );
            let mut legacy = PriorityQueryCore::<_, _, LegacyPriorityPath>::new(
                dictionary.root(),
                query,
                distance,
                Algorithm::Standard,
                Unrestricted,
            );
            let arena_results: Vec<_> = std::iter::from_fn(|| arena.advance()).collect();
            let legacy_results: Vec<_> = std::iter::from_fn(|| legacy.advance()).collect();
            assert_eq!(arena_results, legacy_results, "query={query:?}");
        }
    }

    #[test]
    fn test_transposition() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "tset", 2, Algorithm::Transposition);

        let results: Vec<_> = iter.collect();

        // "tset" with transposition should find "test" at distance 1
        let test_result = results.iter().find(|c| c.term == "test");
        assert!(
            test_result.is_some(),
            "Should find 'test' for 'tset' with transposition"
        );

        if let Some(candidate) = test_result {
            assert_eq!(candidate.distance, 1, "Transposition should be distance 1");
        }
    }

    #[test]
    fn test_take_early() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "test", 2, Algorithm::Standard);

        // Take only first 2 results
        let results: Vec<_> = iter.take(2).collect();
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_no_matches() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "zzzzzzzzz", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();
        assert!(
            results.is_empty(),
            "Should find no matches for 'zzzzzzzzz' within distance 1"
        );
    }

    #[test]
    fn test_custom_policy() {
        let dict = test_dict();
        let mut substitutions = SubstitutionSet::new();
        substitutions.allow('a', 'o');
        let policy = OwnedRestricted::new(substitutions);

        let iter = PriorityQueryIterator::with_policy(
            dict.root(),
            "opple",
            0,
            Algorithm::Standard,
            policy,
        );
        let results: Vec<_> = iter.collect();

        assert!(
            results.iter().any(|candidate| candidate.term == "apple"),
            "custom policy should allow 'opple' to match 'apple' at distance 0: {:?}",
            results
        );
    }
}
