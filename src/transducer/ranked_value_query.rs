//! Lazy value-aware queries ordered by distance and derived confidence.

use super::suggestion::{Suggestion, SuggestionScorer};
use super::transition::{
    with_prepared_unit_cost_row, FinishMode, PreparedUnitCostRow, TransitionSettings,
    UnitCostFrontier, UnitCostMachine,
};
use super::{Algorithm, StatePool, SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted};
use crate::transducer::dictionary_traversal::{
    CursorNativePath, MappedValueSource, ParentArenaPath, PathFrontier, ResultPathStrategy,
    TraversalCursor, TraversalSession,
};
use libdictenstein::value::DictionaryValue;
use libdictenstein::{CharUnit, DictionaryTraversalRoot, MappedDictionaryNode};
use std::collections::VecDeque;

struct RankedIntersection<N: MappedDictionaryNode, E> {
    path: E,
    children_queued: bool,
    final_source: Option<MappedValueSource<N>>,
}

impl<N: MappedDictionaryNode, E> RankedIntersection<N, E> {
    fn new(path: E) -> Self {
        Self {
            path,
            children_queued: false,
            final_source: None,
        }
    }
}

/// A lazy mapped-dictionary iterator with a strict two-level order.
///
/// Results are ordered by increasing edit distance. Only the current distance
/// layer is materialized, scored, and sorted; within it confidence decreases,
/// with the term as a deterministic ascending tie-break. Thus `.take(k)` never
/// materializes result strings or values from a later distance layer.
pub struct RankedValueQueryIterator<N, S, P: SubstitutionPolicy = Unrestricted>
where
    N: MappedDictionaryNode,
    S: SuggestionScorer<N::Value>,
{
    inner: PathRankedValueQueryIterator<N, S, P>,
}

enum PathRankedValueQueryIterator<N, C, P>
where
    N: MappedDictionaryNode,
    C: SuggestionScorer<N::Value>,
    P: SubstitutionPolicy,
{
    Parent(RankedValueQueryCore<N, C, P, ParentArenaPath>),
    Cursor(RankedValueQueryCore<N, C, P, CursorNativePath>),
}

type RankedPending<N, T> = Vec<VecDeque<RankedIntersection<N, PathFrontier<T, UnitCostFrontier>>>>;

fn queue_ranked_children_with_prepared_row<N, K, R>(
    intersection: &mut RankedIntersection<N, PathFrontier<K::Trace, UnitCostFrontier>>,
    traversal: &mut TraversalSession<N>,
    row: &mut R,
    max_distance: usize,
    path_storage: &mut K::Storage,
    pending: &mut RankedPending<N, K::Trace>,
) where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    K: ResultPathStrategy<N>,
    R: PreparedUnitCostRow<N::Unit>,
{
    let mut expansion = K::begin_expansion(&intersection.path.trace);
    intersection.final_source = traversal.filter_map_edges_and_final_source(
        K::position(&intersection.path.trace),
        |label| {
            let state = row.step(label)?;
            row.min_distance(state)
                .filter(|distance| *distance <= max_distance)
                .map(|distance| (state, distance))
        },
        |label, child_position, (state, distance)| {
            pending[distance].push_back(RankedIntersection::new(PathFrontier::new(
                K::child_trace(
                    &intersection.path.trace,
                    &mut expansion,
                    label,
                    child_position,
                    path_storage,
                ),
                state,
            )));
        },
    );
}

struct RankedValueQueryCore<N, C, P, K>
where
    N: MappedDictionaryNode,
    C: SuggestionScorer<N::Value>,
    P: SubstitutionPolicy,
    K: ResultPathStrategy<N>,
{
    pending_by_distance: RankedPending<N, K::Trace>,
    traversal: TraversalSession<N>,
    current_distance: usize,
    max_distance: usize,
    query: Vec<N::Unit>,
    algorithm: Algorithm,
    scorer: C,
    policy: P,
    state_pool: StatePool,
    unit_transitions: UnitCostMachine<N::Unit>,
    path_storage: K::Storage,
    sorted_buffer: Vec<Suggestion<N::Value>>,
}

impl<N, S> RankedValueQueryIterator<N, S, Unrestricted>
where
    N: MappedDictionaryNode,
    S: SuggestionScorer<N::Value>,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
{
    /// Construct from a text query and an unrestricted substitution policy.
    pub fn new(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        scorer: S,
    ) -> Self {
        Self::with_policy(root, query, max_distance, algorithm, scorer, Unrestricted)
    }

    #[cfg(feature = "bindings-core")]
    pub(crate) fn with_traversal_root(
        root: DictionaryTraversalRoot<N>,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        scorer: S,
    ) -> Self {
        Self::with_units_root(
            root,
            N::Unit::from_str(&query),
            max_distance,
            algorithm,
            scorer,
            Unrestricted,
        )
    }
}

impl<N, S, P> RankedValueQueryIterator<N, S, P>
where
    N: MappedDictionaryNode,
    S: SuggestionScorer<N::Value>,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
{
    /// Construct from a text query and explicit substitution policy.
    pub fn with_policy(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        scorer: S,
        policy: P,
    ) -> Self {
        Self::with_policy_traversal_root(
            DictionaryTraversalRoot::owned(root),
            query,
            max_distance,
            algorithm,
            scorer,
            policy,
        )
    }

    pub(crate) fn with_policy_traversal_root(
        root: DictionaryTraversalRoot<N>,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        scorer: S,
        policy: P,
    ) -> Self {
        Self::with_units_root(
            root,
            N::Unit::from_str(&query),
            max_distance,
            algorithm,
            scorer,
            policy,
        )
    }

    /// Construct from native query units, including non-text token alphabets.
    pub fn with_units(
        root: N,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        scorer: S,
        policy: P,
    ) -> Self {
        Self::with_units_traversal_root(
            DictionaryTraversalRoot::owned(root),
            query,
            max_distance,
            algorithm,
            scorer,
            policy,
        )
    }

    pub(crate) fn with_units_traversal_root(
        root: DictionaryTraversalRoot<N>,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        scorer: S,
        policy: P,
    ) -> Self {
        Self::with_units_root(root, query, max_distance, algorithm, scorer, policy)
    }

    fn with_units_root(
        root: DictionaryTraversalRoot<N>,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        scorer: S,
        policy: P,
    ) -> Self {
        let settings = TransitionSettings::new(max_distance, algorithm, false);
        let (unit_transitions, initial) = UnitCostMachine::seeded::<P>(&query, settings);
        let (traversal, root) = TraversalSession::capture_mapped(root);
        Self {
            inner: PathRankedValueQueryIterator::new(
                root,
                traversal,
                query,
                max_distance,
                algorithm,
                scorer,
                policy,
                unit_transitions,
                initial,
            ),
        }
    }

    /// Current result distance layer. Exposed for streaming instrumentation.
    pub fn current_distance(&self) -> usize {
        self.inner.current_distance()
    }

    /// Number of materialized results in the current layer.
    pub fn buffered_results(&self) -> usize {
        self.inner.buffered_results()
    }
}

impl<N, C, P> PathRankedValueQueryIterator<N, C, P>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    C: SuggestionScorer<N::Value>,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        scorer: C,
        policy: P,
        unit_transitions: UnitCostMachine<N::Unit>,
        initial: UnitCostFrontier,
    ) -> Self {
        if traversal.supports_cursor_key_units() {
            Self::Cursor(RankedValueQueryCore::new(
                root,
                traversal,
                query,
                max_distance,
                algorithm,
                scorer,
                policy,
                unit_transitions,
                initial,
            ))
        } else {
            Self::Parent(RankedValueQueryCore::new(
                root,
                traversal,
                query,
                max_distance,
                algorithm,
                scorer,
                policy,
                unit_transitions,
                initial,
            ))
        }
    }

    fn current_distance(&self) -> usize {
        match self {
            Self::Parent(core) => core.current_distance,
            Self::Cursor(core) => core.current_distance,
        }
    }

    fn buffered_results(&self) -> usize {
        match self {
            Self::Parent(core) => core.sorted_buffer.len(),
            Self::Cursor(core) => core.sorted_buffer.len(),
        }
    }

    fn advance(&mut self) -> Option<Suggestion<N::Value>> {
        match self {
            Self::Parent(core) => core.advance(),
            Self::Cursor(core) => core.advance(),
        }
    }
}

impl<N, C, P, K> RankedValueQueryCore<N, C, P, K>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    C: SuggestionScorer<N::Value>,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    K: ResultPathStrategy<N>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        scorer: C,
        policy: P,
        unit_transitions: UnitCostMachine<N::Unit>,
        initial: UnitCostFrontier,
    ) -> Self {
        let (mut pending_by_distance, path_storage) =
            K::cold_buckets(max_distance.saturating_add(1));
        pending_by_distance[0].push_back(RankedIntersection::new(PathFrontier::new(
            K::root(root),
            initial,
        )));
        Self {
            pending_by_distance,
            traversal,
            current_distance: 0,
            max_distance,
            query,
            algorithm,
            scorer,
            policy,
            state_pool: StatePool::new(),
            unit_transitions,
            path_storage,
            sorted_buffer: Vec::with_capacity(64),
        }
    }

    fn queue_children_and_finality(
        &mut self,
        intersection: &mut RankedIntersection<N, PathFrontier<K::Trace, UnitCostFrontier>>,
    ) -> bool {
        if intersection.children_queued {
            return intersection.final_source.is_some();
        }
        intersection.children_queued = true;
        let state_pool = &mut self.state_pool;
        let unit_transitions = &mut self.unit_transitions;
        let policy = &self.policy;
        let query = &self.query;
        let max_distance = self.max_distance;
        let algorithm = self.algorithm;
        let path_storage = &mut self.path_storage;
        let pending = &mut self.pending_by_distance;
        let settings = TransitionSettings::new(max_distance, algorithm, false);
        with_prepared_unit_cost_row!(
            unit_transitions,
            intersection.path.frontier,
            state_pool,
            policy,
            query,
            settings,
            |row| queue_ranked_children_with_prepared_row::<N, K, _>(
                intersection,
                &mut self.traversal,
                &mut row,
                max_distance,
                path_storage,
                pending,
            )
        );
        intersection.final_source.is_some()
    }

    fn normalized_confidence(&self, term: &str, distance: usize, value: &N::Value) -> f64 {
        let score = self.scorer.confidence(term, distance, value);
        if score.is_finite() {
            score
        } else {
            f64::NEG_INFINITY
        }
    }

    fn sort_current_layer(&mut self) {
        self.sorted_buffer.sort_unstable_by(|left, right| {
            left.confidence
                .total_cmp(&right.confidence)
                .then_with(|| right.term.cmp(&left.term))
        });
    }

    fn advance(&mut self) -> Option<Suggestion<N::Value>>
    where
        N::Value: DictionaryValue,
    {
        if let Some(suggestion) = self.sorted_buffer.pop() {
            return Some(suggestion);
        }

        while self.current_distance <= self.max_distance {
            while let Some(mut intersection) =
                self.pending_by_distance[self.current_distance].pop_front()
            {
                let is_final = self.queue_children_and_finality(&mut intersection);
                if is_final {
                    let distance = self
                        .unit_transitions
                        .finish_distance(
                            intersection.path.frontier,
                            FinishMode::Complete,
                            self.query.len(),
                        )
                        .unwrap_or(usize::MAX);
                    if distance <= self.max_distance {
                        if distance > self.current_distance {
                            self.pending_by_distance[distance].push_back(intersection);
                            continue;
                        }
                        if distance == self.current_distance {
                            let final_source = intersection
                                .final_source
                                .take()
                                .expect("a final ranked intersection retains its value source");
                            let final_units = self.traversal.requires_final_units().then(|| {
                                K::materialize_units(
                                    &intersection.path.trace,
                                    &self.traversal,
                                    &self.path_storage,
                                )
                            });
                            if final_units
                                .as_deref()
                                .is_some_and(|units| !self.traversal.accepts_final_units(units))
                            {
                                continue;
                            }
                            if let Some(value) = self
                                .traversal
                                .resolve_final_value(final_source, final_units.as_deref())
                            {
                                let units = final_units.unwrap_or_else(|| {
                                    K::materialize_units(
                                        &intersection.path.trace,
                                        &self.traversal,
                                        &self.path_storage,
                                    )
                                });
                                let term = N::Unit::to_string(&units);
                                let confidence =
                                    self.normalized_confidence(&term, distance, &value);
                                self.sorted_buffer.push(Suggestion {
                                    term,
                                    distance,
                                    value,
                                    confidence,
                                });
                            }
                        }
                    }
                }
            }

            if !self.sorted_buffer.is_empty() {
                self.sort_current_layer();
                return self.sorted_buffer.pop();
            }
            self.current_distance = self.current_distance.saturating_add(1);
        }
        None
    }
}

impl<N, S, P> Iterator for RankedValueQueryIterator<N, S, P>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    S: SuggestionScorer<N::Value>,
    P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
{
    type Item = Suggestion<N::Value>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.advance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::LogFrequencyScorer;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::Dictionary;

    #[test]
    fn distance_then_confidence_order_is_lazy_and_deterministic() {
        let dictionary = DoubleArrayTrie::from_terms_with_values([
            ("cat", 1u64),
            ("bat", 100u64),
            ("cot", 10u64),
            ("dog", 1_000u64),
        ]);

        let mut query = RankedValueQueryIterator::new(
            dictionary.root(),
            "cat".to_owned(),
            3,
            Algorithm::Standard,
            LogFrequencyScorer,
        );
        let first = query.next().unwrap();
        assert_eq!((first.term.as_str(), first.distance), ("cat", 0));
        assert_eq!(query.current_distance(), 0);
        let rest: Vec<_> = query.collect();
        assert_eq!(rest[0].term, "bat");
        assert_eq!(rest[1].term, "cot");
        assert!(rest
            .windows(2)
            .all(|pair| pair[0].distance <= pair[1].distance));
    }
}
