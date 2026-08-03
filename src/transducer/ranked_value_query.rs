//! Lazy value-aware queries ordered by distance and derived confidence.

use super::suggestion::{Suggestion, SuggestionScorer};
use super::transition::{initial_state, transition_state_pooled_ref, TransitionSettings};
use super::{Algorithm, State, StatePool, SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted};
use libdictenstein::value::DictionaryValue;
use libdictenstein::{CharUnit, MappedDictionaryNode};
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

const NO_PATH: usize = usize::MAX;

struct RankedPathNode<U: CharUnit> {
    label: U,
    depth: usize,
    parent: usize,
}

struct RankedIntersection<N: MappedDictionaryNode> {
    label: Option<N::Unit>,
    node: N,
    state: State,
    parent: usize,
    children_queued: bool,
}

impl<N: MappedDictionaryNode> RankedIntersection<N> {
    fn root(node: N, state: State) -> Self {
        Self {
            label: None,
            node,
            state,
            parent: NO_PATH,
            children_queued: false,
        }
    }

    fn child(label: N::Unit, node: N, state: State, parent: usize) -> Self {
        Self {
            label: Some(label),
            node,
            state,
            parent,
            children_queued: false,
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
    current_distance: usize,
    max_distance: usize,
    query: Vec<N::Unit>,
    algorithm: Algorithm,
    scorer: S,
    policy: P,
    state_pool: StatePool,
    path_arena: Vec<RankedPathNode<N::Unit>>,
    seen: FxHashSet<String>,
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
        let initial = initial_state(query.len(), max_distance, algorithm);
        let mut pending_by_distance = (0..=max_distance)
            .map(|_| VecDeque::with_capacity(32))
            .collect::<Vec<_>>();
        pending_by_distance[0].push_back(RankedIntersection::root(root, initial));
        Self {
            pending_by_distance,
            current_distance: 0,
            max_distance,
            query,
            algorithm,
            scorer,
            policy,
            state_pool: StatePool::new(),
            path_arena: Vec::with_capacity(64),
            seen: FxHashSet::default(),
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

    fn push_path(&mut self, label: N::Unit, parent: usize) -> usize {
        let depth = if parent == NO_PATH {
            1
        } else {
            self.path_arena[parent].depth.saturating_add(1)
        };
        let index = self.path_arena.len();
        self.path_arena.push(RankedPathNode {
            label,
            depth,
            parent,
        });
        index
    }

    fn queue_children(&mut self, intersection: &mut RankedIntersection<N>) {
        if intersection.children_queued {
            return;
        }
        intersection.children_queued = true;
        let mut parent_path = None;
        for (label, child) in intersection.node.edges() {
            if let Some(state) = transition_state_pooled_ref(
                &intersection.state,
                &mut self.state_pool,
                &self.policy,
                label,
                &self.query,
                TransitionSettings::new(self.max_distance, self.algorithm, false),
            ) {
                if let Some(distance) = state
                    .min_distance()
                    .filter(|distance| *distance <= self.max_distance)
                {
                    let parent = *parent_path.get_or_insert_with(|| match intersection.label {
                        Some(current) => self.push_path(current, intersection.parent),
                        None => NO_PATH,
                    });
                    self.pending_by_distance[distance]
                        .push_back(RankedIntersection::child(label, child, state, parent));
                } else {
                    self.state_pool.release(state);
                }
            }
        }
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
                if intersection.node.is_final() {
                    let distance = intersection
                        .state
                        .infer_distance(self.query.len())
                        .unwrap_or(usize::MAX);
                    if distance <= self.max_distance {
                        if distance > self.current_distance {
                            self.queue_children(&mut intersection);
                            self.pending_by_distance[distance].push_back(intersection);
                            continue;
                        }
                        if distance == self.current_distance {
                            if let Some(value) = intersection.node.value() {
                                let term = intersection.term(&self.path_arena);
                                if self.seen.insert(term.clone()) {
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
                self.queue_children(&mut intersection);
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
