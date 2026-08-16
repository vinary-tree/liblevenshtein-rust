//! Prefix-pruned fuzzy matching with explicit depth-first traversal.
//!
//! [`QueryIterator`](super::QueryIterator) is breadth-first. A stateful
//! [`PrefixPruner`] is a balanced enter/leave visitor, so sharing one visitor
//! across that breadth-first frontier would mix the states of sibling paths.
//! This iterator deliberately uses an explicit DFS stack: every accepted
//! `enter(unit, depth)` remains active until that exact subtree is exhausted,
//! then receives one matching `leave(unit, depth)`.

use super::transition::{
    initial_state, CachedUnitTransitions, GeneratedStateId, TransitionSettings,
};
use super::{
    Algorithm, NoPruning, PrefixPruner, StatePool, SubstitutionPolicy, SubstitutionPolicyFor,
    Unrestricted,
};
use libdictenstein::{CharUnit, Dictionary, DictionaryNode};

/// Work counters for a prefix-pruned fuzzy DFS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixQueryStats {
    /// Nodes whose finality or outgoing edges were inspected.
    pub nodes_visited: usize,
    /// Outgoing dictionary edges considered.
    pub edges_enumerated: usize,
    /// Whole subtrees rejected by the caller-supplied prefix pruner.
    pub externally_pruned_subtrees: usize,
    /// Whole subtrees rejected by the Levenshtein automaton state transition.
    pub automaton_pruned_subtrees: usize,
}

/// One fuzzy dictionary match produced by [`PrefixQueryIterator`].
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixQueryMatch<U: CharUnit> {
    /// Raw dictionary units in root-to-leaf order.
    pub units: Vec<U>,
    /// Unit-cost edit distance from the query.
    pub distance: usize,
    /// Optional score reported by the prefix pruner.
    pub score: Option<f64>,
}

struct Frame<N: DictionaryNode> {
    edges: std::vec::IntoIter<(N::Unit, N)>,
    state: GeneratedStateId,
    entered_by: Option<N::Unit>,
    is_final: bool,
    final_checked: bool,
}

impl<N: DictionaryNode> Frame<N> {
    fn new(node: N, state: GeneratedStateId, entered_by: Option<N::Unit>) -> Self {
        let mut edges = Vec::with_capacity(node.edge_count().unwrap_or(0));
        let is_final = node.visit_edges_and_finality(|label, child| edges.push((label, child)));
        Self {
            edges: edges.into_iter(),
            state,
            entered_by,
            is_final,
            final_checked: false,
        }
    }
}

/// Lazy fuzzy-query DFS with a balanced caller-supplied prefix pruner.
///
/// The automaton state remains path-local on the DFS stack. Memory is
/// proportional to the current dictionary depth, rather than to the width of
/// a breadth-first frontier. Result order is dictionary DFS order and is not
/// the distance-first order of [`OrderedQueryIterator`](super::OrderedQueryIterator).
pub struct PrefixQueryIterator<N, S = Unrestricted, P = NoPruning>
where
    N: DictionaryNode,
    S: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    P: PrefixPruner<N::Unit>,
{
    query: Vec<N::Unit>,
    max_distance: usize,
    algorithm: Algorithm,
    substitution_policy: S,
    prefix_pruner: Option<P>,
    substring_mode: bool,
    stack: Vec<Frame<N>>,
    prefix: Vec<N::Unit>,
    state_pool: StatePool,
    unit_transitions: CachedUnitTransitions<N::Unit>,
    stats: PrefixQueryStats,
}

impl<N> PrefixQueryIterator<N, Unrestricted, NoPruning>
where
    N: DictionaryNode,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
{
    /// Construct an unrestricted fuzzy DFS without external pruning.
    pub fn new(root: N, query: Vec<N::Unit>, max_distance: usize, algorithm: Algorithm) -> Self {
        Self::with_policy_and_pruner(
            root,
            query,
            max_distance,
            algorithm,
            Unrestricted,
            NoPruning,
            false,
        )
    }

    /// Construct from a compatible dictionary without external pruning.
    pub fn from_dictionary<D>(
        dictionary: &D,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Self
    where
        D: Dictionary<Node = N>,
    {
        Self::with_policy_and_pruner(
            dictionary.root(),
            query,
            max_distance,
            algorithm,
            Unrestricted,
            NoPruning,
            dictionary.is_suffix_based(),
        )
    }
}

impl<N, S, P> PrefixQueryIterator<N, S, P>
where
    N: DictionaryNode,
    S: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    P: PrefixPruner<N::Unit>,
{
    /// Construct a fuzzy DFS with explicit substitution and prefix policies.
    #[allow(clippy::too_many_arguments)]
    pub fn with_policy_and_pruner(
        root: N,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        substitution_policy: S,
        prefix_pruner: P,
        substring_mode: bool,
    ) -> Self {
        let query_length = query.len();
        let initial = initial_state(query_length, max_distance, algorithm);
        let settings = TransitionSettings::new(max_distance, algorithm, substring_mode);
        let mut unit_transitions = CachedUnitTransitions::new(query_length, max_distance);
        let initial = unit_transitions.seed_generated_state(&initial, settings);
        Self {
            query,
            max_distance,
            algorithm,
            substitution_policy,
            prefix_pruner: Some(prefix_pruner),
            substring_mode,
            stack: vec![Frame::new(root, initial, None)],
            prefix: Vec::new(),
            state_pool: StatePool::new(),
            unit_transitions,
            stats: PrefixQueryStats {
                nodes_visited: 1,
                ..PrefixQueryStats::default()
            },
        }
    }

    /// Snapshot traversal counters.
    pub fn stats(&self) -> PrefixQueryStats {
        self.stats
    }

    /// Borrow the prefix pruner, for example to inspect scorer counters.
    pub fn pruner(&self) -> &P {
        self.prefix_pruner
            .as_ref()
            .expect("the prefix pruner is present until the iterator is consumed")
    }

    /// Consume the iterator and recover a balanced prefix pruner.
    ///
    /// If iteration stopped early, outstanding DFS frames are unwound first,
    /// so every successful `enter` has received its matching `leave`.
    pub fn into_pruner(mut self) -> P {
        self.unwind();
        self.prefix_pruner
            .take()
            .expect("the prefix pruner is present until it is extracted")
    }

    fn pruner_mut(&mut self) -> &mut P {
        self.prefix_pruner
            .as_mut()
            .expect("the prefix pruner is present while iteration is active")
    }

    fn unwind(&mut self) {
        while let Some(frame) = self.stack.pop() {
            if let Some(unit) = frame.entered_by {
                let depth = self.prefix.len();
                let popped = self.prefix.pop();
                debug_assert_eq!(popped, Some(unit));
                if let Some(pruner) = self.prefix_pruner.as_mut() {
                    pruner.leave(unit, depth);
                }
            }
        }
    }
}

impl<N, S, P> Iterator for PrefixQueryIterator<N, S, P>
where
    N: DictionaryNode,
    S: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    P: PrefixPruner<N::Unit>,
{
    type Item = PrefixQueryMatch<N::Unit>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.stack.last()?;

            let unchecked_final = {
                let frame = self
                    .stack
                    .last_mut()
                    .expect("the DFS stack was observed non-empty");
                if frame.final_checked {
                    None
                } else {
                    frame.final_checked = true;
                    let state = self.unit_transitions.generated_frontier_state(frame.state);
                    Some((
                        frame.is_final,
                        if self.substring_mode {
                            state.min_distance()
                        } else {
                            state.infer_distance(self.query.len())
                        },
                    ))
                }
            };

            if let Some((true, Some(distance))) = unchecked_final {
                if distance <= self.max_distance {
                    let prefix = self.prefix.clone();
                    if self.pruner_mut().permits_accept(&prefix) {
                        let score = self.pruner_mut().accept(&prefix);
                        return Some(PrefixQueryMatch {
                            units: prefix,
                            distance,
                            score,
                        });
                    }
                }
            }

            let edge = self
                .stack
                .last_mut()
                .expect("the DFS stack was observed non-empty")
                .edges
                .next();
            if let Some((unit, child)) = edge {
                self.stats.edges_enumerated = self.stats.edges_enumerated.saturating_add(1);
                let depth = self.prefix.len().saturating_add(1);
                if !self.pruner_mut().enter(unit, depth) {
                    self.stats.externally_pruned_subtrees =
                        self.stats.externally_pruned_subtrees.saturating_add(1);
                    self.pruner_mut().leave(unit, depth);
                    continue;
                }

                let settings =
                    TransitionSettings::new(self.max_distance, self.algorithm, self.substring_mode);
                let next_state = {
                    let parent_state = self
                        .stack
                        .last()
                        .expect("the DFS parent remains on the stack")
                        .state;
                    self.unit_transitions.transition_generated(
                        parent_state,
                        &mut self.state_pool,
                        &self.substitution_policy,
                        unit,
                        &self.query,
                        settings,
                    )
                };

                if let Some(state) = next_state {
                    self.prefix.push(unit);
                    self.stack.push(Frame::new(child, state, Some(unit)));
                    self.stats.nodes_visited = self.stats.nodes_visited.saturating_add(1);
                } else {
                    self.stats.automaton_pruned_subtrees =
                        self.stats.automaton_pruned_subtrees.saturating_add(1);
                    self.pruner_mut().leave(unit, depth);
                }
                continue;
            }

            let frame = self
                .stack
                .pop()
                .expect("the DFS stack was observed non-empty");
            if let Some(unit) = frame.entered_by {
                let depth = self.prefix.len();
                let popped = self.prefix.pop();
                debug_assert_eq!(popped, Some(unit));
                self.pruner_mut().leave(unit, depth);
            }
        }
    }
}

impl<N, S, P> Drop for PrefixQueryIterator<N, S, P>
where
    N: DictionaryNode,
    S: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>,
    P: PrefixPruner<N::Unit>,
{
    fn drop(&mut self) {
        self.unwind();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::AllowedPrefixes;
    use libdictenstein::double_array_trie::DoubleArrayTrie;

    #[test]
    fn fuzzy_dfs_matches_and_prunes_inside_the_dictionary_walk() {
        let dictionary = DoubleArrayTrie::from_terms(["apple", "apply", "ample", "banana"]);
        let allowed = AllowedPrefixes::new([b"apple".as_slice(), b"ample".as_slice()]);
        let mut query = PrefixQueryIterator::with_policy_and_pruner(
            dictionary.root(),
            b"apple".to_vec(),
            1,
            Algorithm::Standard,
            Unrestricted,
            allowed,
            false,
        );
        let matches: Vec<_> = query.by_ref().collect();
        assert_eq!(
            matches
                .iter()
                .map(|item| item.units.as_slice())
                .collect::<Vec<_>>(),
            [b"ample".as_slice(), b"apple".as_slice()]
        );
        assert!(query.stats().externally_pruned_subtrees > 0);
    }
}
