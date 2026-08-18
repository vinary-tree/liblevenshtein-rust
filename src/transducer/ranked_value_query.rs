//! Lazy value-aware queries ordered by distance and derived confidence.

use super::suggestion::{Suggestion, SuggestionScorer};
use super::transition::{FinishMode, TransitionSettings, UnitCostFrontier, UnitCostMachine};
use super::{Algorithm, StatePool, SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted};
use crate::transducer::dictionary_traversal::{MappedValueSource, TraversalSession};
use libdictenstein::value::DictionaryValue;
use libdictenstein::{
    CharUnit, DictionaryTraversalRoot, MappedDictionaryNode, SnapshotTraversalCursor,
};
use std::collections::VecDeque;

const NO_PATH: usize = usize::MAX;

struct RankedPathNode<U: CharUnit> {
    label: U,
    depth: usize,
    parent: usize,
}

struct RankedIntersection<N: MappedDictionaryNode> {
    label: Option<N::Unit>,
    position: SnapshotTraversalCursor,
    state: UnitCostFrontier,
    parent: usize,
    children_queued: bool,
    final_source: Option<MappedValueSource<N>>,
}

impl<N: MappedDictionaryNode> RankedIntersection<N> {
    fn root(position: SnapshotTraversalCursor, state: UnitCostFrontier) -> Self {
        Self {
            label: None,
            position,
            state,
            parent: NO_PATH,
            children_queued: false,
            final_source: None,
        }
    }

    fn child(
        label: N::Unit,
        position: SnapshotTraversalCursor,
        state: UnitCostFrontier,
        parent: usize,
    ) -> Self {
        Self {
            label: Some(label),
            position,
            state,
            parent,
            children_queued: false,
            final_source: None,
        }
    }

    fn term(&self, paths: &[RankedPathNode<N::Unit>]) -> String {
        let mut units = Vec::with_capacity(if self.parent == NO_PATH {
            usize::from(self.label.is_some())
        } else {
            paths[self.parent].depth + usize::from(self.label.is_some())
        });
        if let Some(label) = self.label {
            units.push(label);
        }
        let mut cursor = self.parent;
        while cursor != NO_PATH {
            units.push(paths[cursor].label);
            cursor = paths[cursor].parent;
        }
        units.reverse();
        N::Unit::to_string(&units)
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
    pending_by_distance: Vec<VecDeque<RankedIntersection<N>>>,
    traversal: TraversalSession<N>,
    current_distance: usize,
    max_distance: usize,
    query: Vec<N::Unit>,
    algorithm: Algorithm,
    scorer: S,
    policy: P,
    state_pool: StatePool,
    unit_transitions: UnitCostMachine<N::Unit>,
    path_arena: Vec<RankedPathNode<N::Unit>>,
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
        Self::with_units(
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
        Self::with_units_root(
            DictionaryTraversalRoot::owned(root),
            query,
            max_distance,
            algorithm,
            scorer,
            policy,
        )
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
        let mut pending_by_distance = (0..=max_distance)
            .map(|_| VecDeque::with_capacity(32))
            .collect::<Vec<_>>();
        let (traversal, root) = TraversalSession::capture_mapped(root);
        pending_by_distance[0].push_back(RankedIntersection::root(root, initial));
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
            path_arena: Vec::with_capacity(64),
            sorted_buffer: Vec::with_capacity(64),
        }
    }

    /// Current result distance layer. Exposed for streaming instrumentation.
    pub fn current_distance(&self) -> usize {
        self.current_distance
    }

    /// Number of materialized results in the current layer.
    pub fn buffered_results(&self) -> usize {
        self.sorted_buffer.len()
    }

    fn queue_children_and_finality(&mut self, intersection: &mut RankedIntersection<N>) -> bool {
        if intersection.children_queued {
            return intersection.final_source.is_some();
        }
        intersection.children_queued = true;
        let mut parent_path = None;
        let state_pool = &mut self.state_pool;
        let unit_transitions = &mut self.unit_transitions;
        let policy = &self.policy;
        let query = &self.query;
        let max_distance = self.max_distance;
        let algorithm = self.algorithm;
        let paths = &mut self.path_arena;
        let pending = &mut self.pending_by_distance;
        intersection.final_source = self.traversal.filter_map_edges_and_final_source(
            intersection.position,
            |label| {
                let state = unit_transitions.step(
                    intersection.state,
                    state_pool,
                    policy,
                    label,
                    query,
                    TransitionSettings::new(max_distance, algorithm, false),
                )?;
                unit_transitions
                    .min_distance(state)
                    .filter(|distance| *distance <= max_distance)
                    .map(|distance| (state, distance))
            },
            |label, child_position, (state, distance)| {
                let parent = *parent_path.get_or_insert_with(|| match intersection.label {
                    Some(current) => {
                        let depth = if intersection.parent == NO_PATH {
                            1
                        } else {
                            paths[intersection.parent].depth.saturating_add(1)
                        };
                        let index = paths.len();
                        paths.push(RankedPathNode {
                            label: current,
                            depth,
                            parent: intersection.parent,
                        });
                        index
                    }
                    None => NO_PATH,
                });
                pending[distance].push_back(RankedIntersection::child(
                    label,
                    child_position,
                    state,
                    parent,
                ));
            },
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
                        .finish_distance(intersection.state, FinishMode::Complete, self.query.len())
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
                            if let Some(value) = self.traversal.resolve_final_value(final_source) {
                                let term = intersection.term(&self.path_arena);
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
        self.advance()
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
