//! Ordered query iterators that return results by distance, then lexicographically.
//!
//! This module provides iterators that yield spelling candidates in a specific order:
//! 1. Primary: Ascending edit distance (0, 1, 2, ...)
//! 2. Secondary: Lexicographic (alphabetical)
//!
//! This ordering enables efficient "top-k" queries and take-while patterns.

use super::transition::{
    with_prepared_unit_cost_row, FinishMode, PreparedUnitCostRow, TransitionSettings,
    UnitCostFrontier, UnitCostMachine,
};
use super::{Algorithm, StatePool, SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted};
use crate::transducer::dictionary_traversal::{
    CursorNativePath, ParentArenaPath, PathFrontier, ResultPathStrategy, TraversalCursor,
    TraversalSession,
};
use libdictenstein::{CharUnit, DictionaryNode, DictionaryTraversalRoot};
use std::collections::VecDeque;
#[cfg(feature = "benchmark-controls")]
use std::sync::OnceLock;

/// Query result containing term and distance.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OrderedCandidate {
    /// Edit distance from query (primary sort key)
    pub distance: usize,
    /// The matching term (secondary sort key - lexicographic)
    pub term: String,
}

struct OrderedIntersection<E> {
    path: E,
    children_queued: bool,
    is_final: bool,
}

impl<E> OrderedIntersection<E> {
    #[inline]
    fn new(path: E) -> Self {
        Self {
            path,
            children_queued: false,
            is_final: false,
        }
    }
}

/// Lazy iterator that returns candidates in distance-first, lexicographic order.
///
/// This iterator yields all distance=0 matches first (exact matches), then all
/// distance=1 matches (alphabetically), then distance=2, etc. This ordering
/// enables efficient "top-k" queries using `take(n)` and distance-bounded
/// queries using `take_while`.
///
/// # Ordering Guarantees
///
/// 1. **Primary:** Results are ordered by ascending edit distance
/// 2. **Secondary:** Within each distance, results are lexicographically ordered
///
/// # Performance
///
/// - Explores the search space in distance layers (BFS-like)
/// - Uses StatePool for allocation reuse
/// - Leverages pre-sorted DAWG edges for lexicographic ordering
/// - Truly lazy - can stop early with `take(n)` or `take_while`
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::prelude::*;
///
/// let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "testing"]);
/// let transducer = Transducer::new(dict, Algorithm::Standard);
///
/// // Get first 3 closest matches
/// for candidate in transducer.query_ordered("tset", 2).take(3) {
///     println!("{}: {}", candidate.term, candidate.distance);
/// }
/// // Output (in order):
/// // test: 0
/// // best: 1
/// // rest: 1
///
/// // Get all matches within distance 1
/// for candidate in transducer.query_ordered("tset", 2).take_while(|c| c.distance <= 1) {
///     println!("{}", candidate.term);
/// }
/// ```
pub struct OrderedQueryIterator<N: DictionaryNode, P: SubstitutionPolicy = Unrestricted> {
    inner: PathOrderedQueryIterator<N, P>,
}

enum PathOrderedQueryIterator<N: DictionaryNode, P: SubstitutionPolicy> {
    Parent(OrderedQueryCore<N, P, ParentArenaPath>),
    Cursor(OrderedQueryCore<N, P, CursorNativePath>),
}

type OrderedPending<T> = Vec<VecDeque<OrderedIntersection<PathFrontier<T, UnitCostFrontier>>>>;

struct OrderedQueryCore<N, P, S>
where
    N: DictionaryNode,
    P: SubstitutionPolicy,
    S: ResultPathStrategy<N>,
{
    /// Pending intersections grouped by minimum distance
    pending_by_distance: OrderedPending<S::Trace>,
    /// Retained snapshot owner and cursor traversal backend.
    traversal: TraversalSession<N>,
    /// Current distance level being explored
    current_distance: usize,
    /// Maximum distance to explore
    max_distance: usize,
    /// Query units (bytes or chars)
    query: Vec<N::Unit>,
    /// Levenshtein algorithm
    algorithm: Algorithm,
    /// Substitution policy
    policy: P,
    /// State pool for allocation reuse
    state_pool: StatePool,
    /// Shared cached transition kernel; queued states are epsilon-closed.
    unit_transitions: UnitCostMachine<N::Unit>,
    /// Whether the deferred root frontier has been encoded into the selected
    /// query-lifetime transition representation.
    activated: bool,
    /// Statically selected result-path storage (parent arena or no storage).
    path_storage: S::Storage,
    /// Substring matching mode (for suffix automata)
    substring_mode: bool,
    /// Sorted buffer for current distance level (ensures lexicographic ordering)
    sorted_buffer: Vec<OrderedCandidate>,
}

impl<N: DictionaryNode> OrderedQueryIterator<N, Unrestricted> {
    /// Create a new ordered query iterator with unrestricted policy
    pub fn new(root: N, query: String, max_distance: usize, algorithm: Algorithm) -> Self {
        Self::with_substring_mode(root, query, max_distance, algorithm, false)
    }

    /// Create a new ordered query iterator with substring matching mode and unrestricted policy
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

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    OrderedQueryIterator<N, P>
{
    /// Create a new ordered query iterator with custom substitution policy
    pub fn with_policy(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
    ) -> Self {
        Self::with_policy_and_substring(root, query, max_distance, algorithm, policy, false)
    }

    /// Create a new ordered query iterator with custom policy and substring matching mode
    pub fn with_policy_and_substring(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        Self::with_traversal_root_and_policy_and_substring(
            DictionaryTraversalRoot::owned(root),
            query,
            max_distance,
            algorithm,
            policy,
            substring_mode,
        )
    }

    pub(crate) fn with_traversal_root_and_policy_and_substring(
        root: DictionaryTraversalRoot<N>,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        let query_units = N::Unit::from_str(&query);
        // Machine selection is deferred until the first poll. An ordinary
        // ordered query may therefore select the same packed representation as
        // unordered traversal, while `.prefix()` before polling can directly
        // select the legacy positional representation without constructing and
        // discarding a packed machine.
        let unit_transitions =
            UnitCostMachine::unseeded_positional(query_units.len(), max_distance);
        let (traversal, root) = TraversalSession::capture(root);
        Self {
            inner: PathOrderedQueryIterator::new(
                root,
                traversal,
                query_units,
                max_distance,
                algorithm,
                policy,
                unit_transitions,
                substring_mode,
            ),
        }
    }

    #[cfg(test)]
    fn activate(&mut self, require_positional: bool) {
        self.inner.activate(require_positional);
    }

    #[cfg(test)]
    fn unit_transitions(&self) -> &UnitCostMachine<N::Unit> {
        match &self.inner {
            PathOrderedQueryIterator::Parent(core) => &core.unit_transitions,
            PathOrderedQueryIterator::Cursor(core) => &core.unit_transitions,
        }
    }

    #[inline]
    fn advance(&mut self) -> Option<OrderedCandidate> {
        self.inner.advance()
    }
}

impl<N, P> PathOrderedQueryIterator<N, P>
where
    N: DictionaryNode,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
        unit_transitions: UnitCostMachine<N::Unit>,
        substring_mode: bool,
    ) -> Self {
        if traversal.supports_cursor_key_units() {
            Self::Cursor(OrderedQueryCore::new(
                root,
                traversal,
                query,
                max_distance,
                algorithm,
                policy,
                unit_transitions,
                substring_mode,
            ))
        } else {
            Self::Parent(OrderedQueryCore::new(
                root,
                traversal,
                query,
                max_distance,
                algorithm,
                policy,
                unit_transitions,
                substring_mode,
            ))
        }
    }

    #[cfg(test)]
    #[inline]
    fn activate(&mut self, require_positional: bool) {
        match self {
            Self::Parent(core) => core.activate(require_positional),
            Self::Cursor(core) => core.activate(require_positional),
        }
    }

    #[inline]
    fn enable_prefix(&mut self) {
        match self {
            Self::Parent(core) => core.enable_prefix(),
            Self::Cursor(core) => core.enable_prefix(),
        }
    }

    #[inline]
    fn advance(&mut self) -> Option<OrderedCandidate> {
        match self {
            Self::Parent(core) => core.advance(),
            Self::Cursor(core) => core.advance(),
        }
    }

    #[inline]
    fn advance_prefix(&mut self) -> Option<OrderedCandidate> {
        match self {
            Self::Parent(core) => core.advance_prefix(),
            Self::Cursor(core) => core.advance_prefix(),
        }
    }
}

impl<N, P, S> OrderedQueryCore<N, P, S>
where
    N: DictionaryNode,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    S: ResultPathStrategy<N>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
        unit_transitions: UnitCostMachine<N::Unit>,
        substring_mode: bool,
    ) -> Self {
        let (mut pending_by_distance, path_storage) =
            S::acquire_buckets(max_distance.saturating_add(1));
        pending_by_distance[0].push_back(OrderedIntersection::new(PathFrontier::new(
            S::root(root),
            UnitCostFrontier(0),
        )));
        Self {
            pending_by_distance,
            traversal,
            current_distance: 0,
            max_distance,
            query,
            algorithm,
            policy,
            state_pool: StatePool::new(),
            unit_transitions,
            activated: false,
            path_storage,
            substring_mode,
            sorted_buffer: Vec::with_capacity(64),
        }
    }

    /// Select one transition representation for the lifetime of an unpolled
    /// iterator and encode its deferred root frontier.
    fn activate(&mut self, require_positional: bool) {
        if self.activated {
            return;
        }
        let settings =
            TransitionSettings::new(self.max_distance, self.algorithm, self.substring_mode);
        let (unit_transitions, initial) =
            if require_positional || force_positional_ordered_enabled() {
                UnitCostMachine::seeded_positional(self.query.len(), settings)
            } else {
                UnitCostMachine::seeded::<P>(&self.query, settings)
            };
        let root = self.pending_by_distance[0]
            .front_mut()
            .expect("deferred ordered iterator lost its root intersection");
        root.path.frontier = initial;
        self.unit_transitions = unit_transitions;
        self.activated = true;
    }

    /// Convert every queued frontier to the canonical positional antichain
    /// under prefix settings while preserving bucket membership and FIFO order.
    ///
    /// This cold path supports the unusual but valid sequence `next();
    /// prefix()`. Ordinary ordered traversal never materializes packed states.
    fn convert_active_to_prefix_positional(&mut self) {
        debug_assert!(self.activated);
        let settings = TransitionSettings::new(self.max_distance, self.algorithm, true);
        let frontiers = self
            .pending_by_distance
            .iter()
            .flat_map(|bucket| bucket.iter().map(|intersection| intersection.path.frontier))
            .collect::<Vec<_>>();
        let (unit_transitions, mapping) =
            self.unit_transitions
                .reencode_as_positional(self.query.len(), settings, frontiers);
        for intersection in self
            .pending_by_distance
            .iter_mut()
            .flat_map(|bucket| bucket.iter_mut())
        {
            intersection.path.frontier = *mapping
                .get(&intersection.path.frontier)
                .expect("ordered prefix conversion omitted a queued frontier");
        }
        self.unit_transitions = unit_transitions;
    }

    /// Advance to the next match in order
    #[inline]
    fn advance(&mut self) -> Option<OrderedCandidate> {
        self.activate(false);

        // First, check if we have buffered results to yield
        if let Some(result) = self.sorted_buffer.pop() {
            return Some(result);
        }

        // Buffer is exhausted, need to collect next distance level
        while self.current_distance <= self.max_distance {
            // Clear buffer and reset index for new distance level
            self.sorted_buffer.clear();

            // Collect ALL results at the current distance level
            while let Some(mut intersection) =
                self.pending_by_distance[self.current_distance].pop_front()
            {
                crate::causal_perf::record_dictionary_intersections(1);
                crate::causal_perf::record_final_checks(1);
                // Expand each dictionary cursor once. A final whose completed
                // distance belongs to a later layer retains only this compact
                // intersection when requeued; its children are not duplicated.
                let is_final = self.queue_children_and_finality(&mut intersection);
                if is_final {
                    let distance = self
                        .unit_transitions
                        .finish_distance(
                            intersection.path.frontier,
                            if self.substring_mode {
                                FinishMode::Substring
                            } else {
                                FinishMode::Complete
                            },
                            self.query.len(),
                        )
                        .unwrap_or(usize::MAX);

                    if distance <= self.max_distance {
                        if distance == self.current_distance {
                            // Distance matches current level - add to buffer
                            let units = S::materialize_units(
                                &intersection.path.trace,
                                &self.traversal,
                                &self.path_storage,
                            );
                            if !self.traversal.accepts_final_units(&units) {
                                continue;
                            }
                            let term = N::Unit::to_string(&units);
                            crate::causal_perf::record_matches_materialized(1);
                            self.sorted_buffer.push(OrderedCandidate { distance, term });
                        } else if distance > self.current_distance {
                            // Actual distance is higher than bucket - requeue to correct bucket
                            // This can happen when min_dist underestimates final distance
                            self.pending_by_distance[distance].push_back(intersection);
                        }
                        // If distance < current_distance, skip (already passed that level)
                    }
                }
            }

            // If we collected any results at this distance, sort them and return first
            if !self.sorted_buffer.is_empty() {
                self.sort_buffer_for_pop();
                return self.sorted_buffer.pop();
            }

            // No results at this distance, move to next
            self.current_distance += 1;
        }

        None
    }

    /// Queue children and return finality through the generic fused node seam.
    /// Prefix traversal always expands every popped node, unlike the ordered
    /// distance scheduler's conditional `distance < current_distance` branch.
    #[inline]
    fn queue_children_and_finality(
        &mut self,
        intersection: &mut OrderedIntersection<PathFrontier<S::Trace, UnitCostFrontier>>,
    ) -> bool {
        if intersection.children_queued {
            return intersection.is_final;
        }
        intersection.children_queued = true;
        let mut expansion = S::begin_expansion(&intersection.path.trace);
        let state_pool = &mut self.state_pool;
        let unit_transitions = &mut self.unit_transitions;
        let policy = &self.policy;
        let query = &self.query;
        let max_distance = self.max_distance;
        let algorithm = self.algorithm;
        let substring_mode = self.substring_mode;
        let path_storage = &mut self.path_storage;
        let pending = &mut self.pending_by_distance;
        let settings = TransitionSettings::new(max_distance, algorithm, substring_mode);
        intersection.is_final = with_prepared_unit_cost_row!(
            unit_transitions,
            intersection.path.frontier,
            state_pool,
            policy,
            query,
            settings,
            |row| self.traversal.filter_map_edges_and_finality(
                S::position(&intersection.path.trace),
                |label| {
                    crate::causal_perf::record_edges_enumerated(1);
                    let next_state = row.step(label)?;
                    row.min_distance(next_state)
                        .filter(|&min_dist| min_dist <= max_distance)
                        .map(|min_dist| {
                            #[cfg(feature = "perf-instrumentation")]
                            {
                                (
                                    next_state,
                                    min_dist,
                                    row.active_len(next_state),
                                    row.frontier_storage_bytes(next_state),
                                )
                            }
                            #[cfg(not(feature = "perf-instrumentation"))]
                            {
                                (next_state, min_dist, 0usize, 0usize)
                            }
                        })
                },
                |label, child_position, (next_state, min_dist, position_count, storage_bytes)| {
                    crate::causal_perf::record_transition_accepted(1);
                    #[cfg(feature = "perf-instrumentation")]
                    {
                        crate::causal_perf::record_state_positions_enqueued(position_count as u64);
                        crate::causal_perf::record_state_bytes_enqueued(storage_bytes as u64);
                    }
                    #[cfg(not(feature = "perf-instrumentation"))]
                    let _ = (position_count, storage_bytes);
                    pending[min_dist].push_back(OrderedIntersection::new(PathFrontier::new(
                        S::child_trace(
                            &intersection.path.trace,
                            &mut expansion,
                            label,
                            child_position,
                            path_storage,
                        ),
                        next_state,
                    )));
                    #[cfg(feature = "perf-instrumentation")]
                    crate::causal_perf::record_pending_queue_size(
                        pending.iter().map(VecDeque::len).sum(),
                    );
                },
            )
        );
        intersection.is_final
    }

    #[inline]
    fn sort_buffer_for_pop(&mut self) {
        // Store terms in reverse lexicographic order so pop() yields owned candidates
        // in ascending lexicographic order without cloning their String payloads.
        if self.sorted_buffer.len() <= 10 {
            // For small buffers, insertion sort is faster due to better cache locality
            for i in 1..self.sorted_buffer.len() {
                let mut j = i;
                while j > 0 && self.sorted_buffer[j].term > self.sorted_buffer[j - 1].term {
                    self.sorted_buffer.swap(j, j - 1);
                    j -= 1;
                }
            }
        } else {
            // For larger buffers, use unstable sort (faster, doesn't preserve order of equal elements)
            self.sorted_buffer
                .sort_unstable_by(|a, b| b.term.cmp(&a.term));
        }
    }

    fn enable_prefix(&mut self) {
        self.substring_mode = true;
        if self.activated {
            self.convert_active_to_prefix_positional();
        } else {
            self.activate(true);
        }
    }

    #[inline]
    fn advance_prefix(&mut self) -> Option<OrderedCandidate> {
        if let Some(result) = self.sorted_buffer.pop() {
            return Some(result);
        }

        let query_len = self.query.len();
        while self.current_distance <= self.max_distance {
            self.sorted_buffer.clear();
            while let Some(mut intersection) =
                self.pending_by_distance[self.current_distance].pop_front()
            {
                crate::causal_perf::record_dictionary_intersections(1);
                crate::causal_perf::record_final_checks(1);
                let is_final = self.queue_children_and_finality(&mut intersection);
                let matched_distance = if is_final {
                    self.unit_transitions
                        .finish_distance(intersection.path.frontier, FinishMode::Prefix, query_len)
                        .filter(|&distance| {
                            distance <= self.max_distance && distance == self.current_distance
                        })
                } else {
                    None
                };

                if let Some(distance) = matched_distance {
                    let units = S::materialize_units(
                        &intersection.path.trace,
                        &self.traversal,
                        &self.path_storage,
                    );
                    if !self.traversal.accepts_final_units(&units) {
                        continue;
                    }
                    let term = N::Unit::to_string(&units);
                    crate::causal_perf::record_matches_materialized(1);
                    self.sorted_buffer.push(OrderedCandidate { distance, term });
                }
            }

            if !self.sorted_buffer.is_empty() {
                self.sort_buffer_for_pop();
                return self.sorted_buffer.pop();
            }
            self.current_distance += 1;
        }
        None
    }
}

impl<N, P> OrderedQueryIterator<N, P>
where
    N: DictionaryNode,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
{
    /// Add a filter predicate to this iterator.
    ///
    /// Returns a new iterator that only yields candidates matching the predicate.
    /// The filter is applied during traversal, allowing early termination.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Filter to only identifiers starting with lowercase
    /// query.filter(|candidate| {
    ///     candidate.term.chars().next()
    ///         .map(|c| c.is_lowercase())
    ///         .unwrap_or(false)
    /// })
    /// ```
    pub fn filter<F>(self, predicate: F) -> FilteredOrderedQueryIterator<N, P, F>
    where
        F: Fn(&OrderedCandidate) -> bool,
    {
        FilteredOrderedQueryIterator {
            inner: self,
            predicate,
        }
    }

    /// Switch to prefix matching mode.
    ///
    /// In prefix mode, dictionary terms that START with something approximately
    /// equal to the query are matched, allowing terms to be longer than the query.
    ///
    /// This is essential for autocomplete/code completion where users type partial
    /// identifiers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Query: "tes"
    /// // Matches: "test" (d=0), "testing" (d=0), "tester" (d=0), "best" (d=1)
    /// query.prefix()
    /// ```
    pub fn prefix(mut self) -> PrefixOrderedQueryIterator<N, P> {
        self.inner.enable_prefix();
        PrefixOrderedQueryIterator { inner: self }
    }
}

#[cfg(feature = "benchmark-controls")]
fn force_positional_ordered_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_FORCE_POSITIONAL_ORDERED").is_some()
    })
}

#[cfg(not(feature = "benchmark-controls"))]
#[inline(always)]
const fn force_positional_ordered_enabled() -> bool {
    false
}

impl<N, P, S> Drop for OrderedQueryCore<N, P, S>
where
    N: DictionaryNode,
    P: SubstitutionPolicy,
    S: ResultPathStrategy<N>,
{
    fn drop(&mut self) {
        let pending_by_distance = std::mem::take(&mut self.pending_by_distance);
        let path_storage = std::mem::take(&mut self.path_storage);
        S::release_buckets(pending_by_distance, path_storage);
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>> Iterator
    for OrderedQueryIterator<N, P>
{
    type Item = OrderedCandidate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.advance()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    std::iter::FusedIterator for OrderedQueryIterator<N, P>
{
}

/// Filtered ordered query iterator.
///
/// Wraps an OrderedQueryIterator and applies a filter predicate to results.
/// Only candidates matching the predicate are yielded.
pub struct FilteredOrderedQueryIterator<N: DictionaryNode, P: SubstitutionPolicy, F>
where
    F: Fn(&OrderedCandidate) -> bool,
{
    inner: OrderedQueryIterator<N, P>,
    predicate: F,
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>, F> Iterator
    for FilteredOrderedQueryIterator<N, P, F>
where
    F: Fn(&OrderedCandidate) -> bool,
{
    type Item = OrderedCandidate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Keep advancing until we find a match or exhaust the iterator
        loop {
            let candidate = self.inner.next()?;
            if (self.predicate)(&candidate) {
                return Some(candidate);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>, F>
    std::iter::FusedIterator for FilteredOrderedQueryIterator<N, P, F>
where
    F: Fn(&OrderedCandidate) -> bool,
{
}

/// Prefix ordered query iterator.
///
/// Performs approximate prefix matching where dictionary terms that START with
/// something approximately equal to the query are matched. Terms can be longer
/// than the query.
///
/// Essential for autocomplete and code completion.
pub struct PrefixOrderedQueryIterator<N: DictionaryNode, P: SubstitutionPolicy = Unrestricted> {
    inner: OrderedQueryIterator<N, P>,
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    PrefixOrderedQueryIterator<N, P>
{
    /// Advance to the next prefix match in order
    #[inline]
    fn advance_prefix(&mut self) -> Option<OrderedCandidate> {
        self.inner.inner.advance_prefix()
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>> Iterator
    for PrefixOrderedQueryIterator<N, P>
{
    type Item = OrderedCandidate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.advance_prefix()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    std::iter::FusedIterator for PrefixOrderedQueryIterator<N, P>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
    use libdictenstein::dynamic_dawg::DynamicDawgU64;
    use libdictenstein::Dictionary;

    #[test]
    fn test_ordered_exact_match() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 0, Algorithm::Standard);

        let results: Vec<_> = query.collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "test");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_ordered_buffers_reuse_after_partial_drop_preserves_order() {
        let dict = DoubleArrayTrie::from_terms(vec!["cat", "bat", "car", "dog"]);
        let mut partial =
            OrderedQueryIterator::new(dict.root(), "cat".to_string(), 3, Algorithm::Standard);
        assert_eq!(
            partial.next().map(|candidate| candidate.term),
            Some("cat".to_owned())
        );
        drop(partial);

        let reused =
            OrderedQueryIterator::new(dict.root(), "cat".to_string(), 1, Algorithm::Standard)
                .collect::<Vec<_>>();
        assert_eq!(
            reused,
            vec![
                OrderedCandidate {
                    distance: 0,
                    term: "cat".to_owned(),
                },
                OrderedCandidate {
                    distance: 1,
                    term: "bat".to_owned(),
                },
                OrderedCandidate {
                    distance: 1,
                    term: "car".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn test_ordered_query_reconstructs_term_past_initial_path_capacity() {
        let long_term = "a".repeat(96);
        let dict = DoubleArrayTrie::from_terms(vec![long_term.as_str()]);
        let query =
            OrderedQueryIterator::new(dict.root(), long_term.clone(), 0, Algorithm::Standard);

        let results: Vec<_> = query.collect();
        assert_eq!(
            results,
            vec![OrderedCandidate {
                distance: 0,
                term: long_term,
            }]
        );
    }

    #[test]
    fn test_ordered_distance_first() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "test",    // distance 0
            "best",    // distance 1
            "rest",    // distance 1
            "testing", // distance 3
            "nest",    // distance 1
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 3, Algorithm::Standard);

        let results: Vec<_> = query.collect();

        // Verify distance ordering
        for i in 1..results.len() {
            assert!(
                results[i - 1].distance <= results[i].distance,
                "Distance ordering violated: {} (d={}) should come before {} (d={})",
                results[i - 1].term,
                results[i - 1].distance,
                results[i].term,
                results[i].distance
            );
        }

        // Verify exact match comes first
        assert_eq!(results[0].term, "test");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_ordered_lexicographic_within_distance() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "test", "best", "fest", "nest", "rest", "west", "zest",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 1, Algorithm::Standard);

        let results: Vec<_> = query.collect();

        // Group by distance
        let mut by_distance: Vec<Vec<String>> = vec![Vec::new(); 2];
        for candidate in results {
            by_distance[candidate.distance].push(candidate.term);
        }

        // Verify distance 0
        assert_eq!(by_distance[0], vec!["test"]);

        // Verify distance 1 is lexicographically sorted
        let dist1 = &by_distance[1];
        for i in 1..dist1.len() {
            assert!(
                dist1[i - 1] <= dist1[i],
                "Lexicographic ordering violated: {} should come before {}",
                dist1[i - 1],
                dist1[i]
            );
        }
    }

    #[test]
    fn test_ordered_large_distance_bucket_lexicographic() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "cat", "pat", "nat", "mat", "lat", "kat", "jat", "hat", "gat", "fat", "eat", "dat",
            "bat",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "cat".to_string(), 1, Algorithm::Standard);

        let terms: Vec<_> = query.map(|candidate| candidate.term).collect();
        assert_eq!(
            terms,
            vec![
                "cat", "bat", "dat", "eat", "fat", "gat", "hat", "jat", "kat", "lat", "mat", "nat",
                "pat",
            ]
        );
    }

    #[test]
    fn test_ordered_take() {
        let dict =
            DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "nest", "testing", "resting"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 3, Algorithm::Standard);

        // Take only first 3 results
        let results: Vec<_> = query.take(3).collect();
        assert_eq!(results.len(), 3);

        // First should be exact match
        assert_eq!(results[0].distance, 0);
        assert_eq!(results[0].term, "test");

        // Next two should be distance 1
        assert_eq!(results[1].distance, 1);
        assert_eq!(results[2].distance, 1);
    }

    #[test]
    fn test_ordered_take_while() {
        let dict =
            DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "nest", "testing", "resting"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 3, Algorithm::Standard);

        // Take while distance <= 1
        let results: Vec<_> = query.take_while(|c| c.distance <= 1).collect();

        // All results should have distance <= 1
        for candidate in &results {
            assert!(candidate.distance <= 1);
        }

        // Should include exact match
        assert!(results.iter().any(|c| c.term == "test" && c.distance == 0));

        // Should not include distance 3 results
        assert!(!results.iter().any(|c| c.term == "testing"));
        assert!(!results.iter().any(|c| c.term == "resting"));
    }

    #[test]
    fn test_ordered_empty_query() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "xyz".to_string(), 0, Algorithm::Standard);

        let results: Vec<_> = query.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_ordered_consistency_with_unordered() {
        // Verify ordered iterator returns same results as unordered, just in different order
        use crate::transducer::query::QueryIterator;

        let dict =
            DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "nest", "fest", "testing"]);

        let ordered =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 2, Algorithm::Standard);

        let unordered: QueryIterator<_, String> =
            QueryIterator::new(dict.root(), "test".to_string(), 2, Algorithm::Standard);

        let mut ordered_terms: Vec<_> = ordered.map(|c| c.term).collect();
        let mut unordered_terms: Vec<_> = unordered.collect();

        ordered_terms.sort();
        unordered_terms.sort();

        assert_eq!(ordered_terms, unordered_terms);
    }

    #[test]
    fn test_filtered_query() {
        let dict =
            DoubleArrayTrie::from_terms(vec!["test", "Test", "TEST", "best", "Best", "rest"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 1, Algorithm::Standard);

        // Filter to only lowercase terms
        let results: Vec<_> = query
            .filter(|c| c.term.chars().all(|ch| ch.is_lowercase()))
            .collect();

        // Should only include lowercase results
        for candidate in &results {
            assert!(candidate.term.chars().all(|ch| ch.is_lowercase()));
        }

        // Should include lowercase matches
        assert!(results.iter().any(|c| c.term == "test"));
        assert!(results.iter().any(|c| c.term == "best"));
        assert!(results.iter().any(|c| c.term == "rest"));

        // Should NOT include uppercase matches
        assert!(!results.iter().any(|c| c.term == "Test"));
        assert!(!results.iter().any(|c| c.term == "TEST"));
        assert!(!results.iter().any(|c| c.term == "Best"));
    }

    #[test]
    fn test_filtered_query_with_distance() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "best", "rest", "nest"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 3, Algorithm::Standard);

        // Filter to terms with exactly 4 characters
        let results: Vec<_> = query.filter(|c| c.term.len() == 4).collect();

        // All results should have exactly 4 characters
        for candidate in &results {
            assert_eq!(candidate.term.len(), 4);
        }

        // Should include 4-letter matches
        assert!(results.iter().any(|c| c.term == "test"));
        assert!(results.iter().any(|c| c.term == "best"));
        assert!(results.iter().any(|c| c.term == "rest"));
        assert!(results.iter().any(|c| c.term == "nest"));

        // Should NOT include longer terms
        assert!(!results.iter().any(|c| c.term == "testing"));
    }

    #[test]
    fn test_filtered_query_maintains_order() {
        let dict =
            DoubleArrayTrie::from_terms(vec!["a", "aa", "aaa", "ab", "abc", "b", "ba", "baa"]);

        let query = OrderedQueryIterator::new(dict.root(), "a".to_string(), 2, Algorithm::Standard);

        // Filter to terms starting with 'a'
        let results: Vec<_> = query.filter(|c| c.term.starts_with('a')).collect();

        // Verify ordering is maintained (distance-first, then lexicographic)
        for i in 1..results.len() {
            assert!(
                results[i - 1].distance <= results[i].distance,
                "Distance ordering violated"
            );

            if results[i - 1].distance == results[i].distance {
                assert!(
                    results[i - 1].term <= results[i].term,
                    "Lexicographic ordering violated within distance level"
                );
            }
        }
    }

    #[test]
    fn test_filtered_query_with_take() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "test", "testing", "tester", "best", "rest", "nest", "fest",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 2, Algorithm::Standard);

        // Filter to terms ending with 'st' and take first 3
        let results: Vec<_> = query.filter(|c| c.term.ends_with("st")).take(3).collect();

        assert_eq!(results.len(), 3);

        // All should end with 'st'
        for candidate in &results {
            assert!(candidate.term.ends_with("st"));
        }

        // Should be ordered by distance
        assert!(results[0].distance <= results[1].distance);
        assert!(results[1].distance <= results[2].distance);
    }

    #[test]
    fn test_prefix_exact_match() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tester", "tested"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 0, Algorithm::Standard);

        let results: Vec<_> = query.prefix().collect();

        // Should match all terms starting with "test" exactly
        assert!(
            results.len() >= 4,
            "Expected at least 4 results, got {}",
            results.len()
        );
        assert!(results.iter().any(|c| c.term == "test" && c.distance == 0));
        assert!(results
            .iter()
            .any(|c| c.term == "testing" && c.distance == 0));
        assert!(results
            .iter()
            .any(|c| c.term == "tester" && c.distance == 0));
        assert!(results
            .iter()
            .any(|c| c.term == "tested" && c.distance == 0));
    }

    #[test]
    fn test_prefix_with_errors() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "best", "resting", "rest"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "tes".to_string(), 1, Algorithm::Standard);

        let results: Vec<_> = query.prefix().collect();

        // Should match:
        // - "test", "testing" with d=0 (exact prefix match)
        // - "best", "rest", "resting" with d=1 (one error in prefix)
        assert!(results.iter().any(|c| c.term == "test" && c.distance == 0));
        assert!(results
            .iter()
            .any(|c| c.term == "testing" && c.distance == 0));
        assert!(results.iter().any(|c| c.term == "best" && c.distance == 1));
        assert!(results.iter().any(|c| c.term == "rest" && c.distance == 1));
    }

    #[test]
    fn test_prefix_ordering() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "test", "testing", "tester", "best", "resting", "rest",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 2, Algorithm::Standard);

        let results: Vec<_> = query.prefix().collect();

        // Verify distance-first ordering
        for i in 1..results.len() {
            assert!(
                results[i - 1].distance <= results[i].distance,
                "Distance ordering violated: {} (d={}) should come before {} (d={})",
                results[i - 1].term,
                results[i - 1].distance,
                results[i].term,
                results[i].distance
            );
        }

        // First results should be distance=0
        let first_distance = results[0].distance;
        assert_eq!(first_distance, 0, "First result should have distance 0");
    }

    #[test]
    fn test_prefix_large_distance_bucket_lexicographic() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "prez", "prea", "preq", "preb", "prex", "prec", "pred", "pree", "pref", "preg", "preh",
            "prei", "pre",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "pre".to_string(), 0, Algorithm::Standard);

        let terms: Vec<_> = query.prefix().map(|candidate| candidate.term).collect();
        assert_eq!(
            terms,
            vec![
                "pre", "prea", "preb", "prec", "pred", "pree", "pref", "preg", "preh", "prei",
                "preq", "prex", "prez",
            ]
        );
    }

    #[test]
    fn test_prefix_vs_exact() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tester"]);

        // Exact matching
        let exact_query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 0, Algorithm::Standard);

        let exact_results: Vec<_> = exact_query.collect();

        // Prefix matching
        let prefix_query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 0, Algorithm::Standard);

        let prefix_results: Vec<_> = prefix_query.prefix().collect();

        // Exact should only match "test"
        assert_eq!(exact_results.len(), 1);
        assert_eq!(exact_results[0].term, "test");

        // Prefix should match all terms starting with "test"
        assert!(prefix_results.len() >= 3);
        assert!(prefix_results.iter().any(|c| c.term == "test"));
        assert!(prefix_results.iter().any(|c| c.term == "testing"));
        assert!(prefix_results.iter().any(|c| c.term == "tester"));
    }

    #[test]
    fn test_prefix_autocomplete_scenario() {
        // Simulating code completion
        let dict = DoubleArrayTrie::from_terms(vec![
            "getValue",
            "getVariable",
            "getValue2",
            "setValue",
            "setVariable",
            "removeValue",
            "hasValue",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "getVal".to_string(), 1, Algorithm::Standard);

        let results: Vec<_> = query.prefix().take(5).collect();

        // Should prioritize exact prefix matches
        // Results should be ordered by distance, then alphabetically
        for candidate in &results {
            println!("{}: {}", candidate.term, candidate.distance);
        }

        // Should include getValue family with low distance
        assert!(results.iter().any(|c| c.term.starts_with("getValue")));
    }

    #[test]
    fn test_prefix_with_filter() {
        // Combining prefix matching with filtering
        let dict = DoubleArrayTrie::from_terms(vec![
            "TestCase",
            "testMethod",
            "testHelper",
            "bestPractice",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 1, Algorithm::Standard);

        // Prefix match + filter for lowercase
        let results: Vec<_> = query
            .prefix()
            .filter(|c| {
                c.term
                    .chars()
                    .next()
                    .expect("test fixture: candidate term is non-empty")
                    .is_lowercase()
            })
            .collect();

        // Should only include lowercase-starting matches
        for candidate in &results {
            assert!(candidate
                .term
                .chars()
                .next()
                .expect("test fixture: candidate term is non-empty")
                .is_lowercase());
        }

        assert!(results.iter().any(|c| c.term == "testMethod"));
        assert!(results.iter().any(|c| c.term == "testHelper"));
        assert!(!results.iter().any(|c| c.term == "TestCase"));
    }

    #[test]
    fn ordinary_ordered_queries_activate_the_shared_packed_machines() {
        let dict = DoubleArrayTrie::from_terms(vec!["ab", "ba", "aab", "abb"]);

        let mut standard =
            OrderedQueryIterator::new(dict.root(), "ab".to_owned(), 2, Algorithm::Standard);
        standard.activate(false);
        assert!(matches!(
            standard.unit_transitions(),
            UnitCostMachine::PackedStandard(_)
        ));

        let mut osa =
            OrderedQueryIterator::new(dict.root(), "ab".to_owned(), 2, Algorithm::Transposition);
        osa.activate(false);
        assert!(matches!(
            osa.unit_transitions(),
            UnitCostMachine::PackedOsa(_)
        ));

        let mut merge_split =
            OrderedQueryIterator::new(dict.root(), "ab".to_owned(), 2, Algorithm::MergeAndSplit);
        merge_split.activate(false);
        assert!(matches!(
            merge_split.unit_transitions(),
            UnitCostMachine::PackedMergeSplit(_)
        ));
    }

    #[test]
    fn fresh_prefix_activates_positional_without_building_a_packed_machine() {
        let dict = DoubleArrayTrie::from_terms(vec!["pre", "prefix", "present"]);
        let prefix =
            OrderedQueryIterator::new(dict.root(), "pre".to_owned(), 1, Algorithm::Standard)
                .prefix();

        assert!(matches!(
            prefix.inner.unit_transitions(),
            UnitCostMachine::Positional(_)
        ));
    }

    #[test]
    fn partial_packed_to_prefix_conversion_matches_positional_continuation() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "bat", "bats", "car", "cat", "cater", "cats", "cut", "scat",
        ]);

        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
        ] {
            let mut packed = OrderedQueryIterator::new(dict.root(), "cat".to_owned(), 2, algorithm);
            let packed_first = packed.next();
            assert!(matches!(
                (packed.unit_transitions(), algorithm),
                (UnitCostMachine::PackedStandard(_), Algorithm::Standard)
                    | (UnitCostMachine::PackedOsa(_), Algorithm::Transposition)
                    | (
                        UnitCostMachine::PackedMergeSplit(_),
                        Algorithm::MergeAndSplit
                    )
            ));
            let packed_remainder = packed.prefix().collect::<Vec<_>>();

            let mut positional =
                OrderedQueryIterator::new(dict.root(), "cat".to_owned(), 2, algorithm);
            positional.activate(true);
            let positional_first = positional.next();
            let positional_remainder = positional.prefix().collect::<Vec<_>>();

            assert_eq!(packed_first, positional_first, "algorithm={algorithm:?}");
            assert_eq!(
                packed_remainder, positional_remainder,
                "algorithm={algorithm:?}"
            );
        }
    }

    #[test]
    fn packed_ordered_execution_is_unit_generic_for_char_and_u64_domains() {
        let char_dict = DynamicDawgChar::<()>::from_terms(["café", "cafe", "cafés", "safe"]);
        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
        ] {
            let mut query =
                OrderedQueryIterator::new(char_dict.root(), "café".to_owned(), 2, algorithm);
            query.activate(false);
            assert!(!matches!(
                query.unit_transitions(),
                UnitCostMachine::Positional(_)
            ));
            let results = query.collect::<Vec<_>>();
            assert!(results
                .iter()
                .any(|candidate| candidate.term == "café" && candidate.distance == 0));
        }

        let u64_dict = DynamicDawgU64::<()>::from_terms(["alpha", "alpah", "alfa", "beta"]);
        for algorithm in [
            Algorithm::Standard,
            Algorithm::Transposition,
            Algorithm::MergeAndSplit,
        ] {
            let mut query =
                OrderedQueryIterator::new(u64_dict.root(), "alpha".to_owned(), 2, algorithm);
            query.activate(false);
            assert!(!matches!(
                query.unit_transitions(),
                UnitCostMachine::Positional(_)
            ));
            let results = query.collect::<Vec<_>>();
            assert!(results
                .iter()
                .any(|candidate| candidate.term == "alpha" && candidate.distance == 0));
        }
    }
}
