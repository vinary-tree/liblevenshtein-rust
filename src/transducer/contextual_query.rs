//! Opt-in trie traversal for context-dependent edit costs.

use super::{ContextualCost, EditContext};
use crate::transducer::dictionary_traversal::{
    acquire_traversal_buffers_with_capacity, release_traversal_buffers, TraversalCursor,
    TraversalSession,
};
use libdictenstein::{CharUnit, Dictionary, DictionaryNode, DictionaryTraversalRoot};
use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{self, Display};

/// Construction error for [`ContextualQueryIterator`].
#[derive(Clone, Debug, PartialEq)]
pub enum ContextualQueryError {
    /// Threshold is negative or non-finite.
    InvalidThreshold(f64),
    /// The cost model did not provide a finite, strictly positive lower bound.
    InvalidMinimumCost(f64),
}

impl Display for ContextualQueryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidThreshold(value) => {
                write!(
                    formatter,
                    "contextual query threshold must be finite and non-negative, got {value}"
                )
            }
            Self::InvalidMinimumCost(value) => write!(
                formatter,
                "ContextualCost::min_nonzero_cost must be finite and strictly positive, got {value}"
            ),
        }
    }
}

impl Error for ContextualQueryError {}

/// One accepted contextual-cost match.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextualCandidate<U: CharUnit> {
    /// Raw matched dictionary units.
    pub units: Vec<U>,
    /// Exact contextual DP cost.
    pub distance: f64,
}

/// Work and validation counters for a contextual traversal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContextualQueryStats {
    /// Trie nodes removed from the pending queue.
    pub nodes_visited: usize,
    /// Outgoing edges considered.
    pub edges_enumerated: usize,
    /// Edges rejected because every DP cell exceeded the threshold.
    pub subtrees_pruned: usize,
    /// Non-finite or negative dynamic costs treated as forbidden.
    pub invalid_costs_rejected: usize,
}

const NO_PATH: usize = usize::MAX;

struct ContextualPathNode<U: CharUnit> {
    label: U,
    parent: usize,
    depth: usize,
}

struct Pending<U: CharUnit, C: Copy> {
    position: TraversalCursor<C>,
    label: Option<U>,
    parent: usize,
    column: Vec<f64>,
}

/// Lazy dictionary traversal driven by context-dependent Levenshtein columns.
///
/// This is separate from `QueryIteratorF64`: contextual decisions vary by query
/// position and prefix, defeating characteristic-vector reuse. Space is bounded
/// by queued trie nodes times one `query.len() + 1` column.
pub struct ContextualQueryIterator<N, C>
where
    N: DictionaryNode,
    C: ContextualCost<N::Unit>,
{
    query: Vec<N::Unit>,
    max_cost: f64,
    costs: C,
    pending: VecDeque<Pending<N::Unit, N::SnapshotCursor>>,
    traversal: TraversalSession<N>,
    path_arena: Vec<ContextualPathNode<N::Unit>>,
    prefix_scratch: Vec<N::Unit>,
    column_pool: Vec<Vec<f64>>,
    stats: ContextualQueryStats,
}

impl<N, C> ContextualQueryIterator<N, C>
where
    N: DictionaryNode,
    N::Unit: CharUnit + std::hash::Hash,
    C: ContextualCost<N::Unit>,
{
    /// Construct from a root node and native query units.
    pub fn try_new(
        root: N,
        query: Vec<N::Unit>,
        max_cost: f64,
        costs: C,
    ) -> Result<Self, ContextualQueryError> {
        Self::try_from_traversal_root(DictionaryTraversalRoot::owned(root), query, max_cost, costs)
    }

    fn try_from_traversal_root(
        root: DictionaryTraversalRoot<N>,
        query: Vec<N::Unit>,
        max_cost: f64,
        costs: C,
    ) -> Result<Self, ContextualQueryError> {
        if !max_cost.is_finite() || max_cost < 0.0 {
            return Err(ContextualQueryError::InvalidThreshold(max_cost));
        }
        let minimum = costs.min_nonzero_cost();
        if !minimum.is_finite() || minimum <= 0.0 {
            return Err(ContextualQueryError::InvalidMinimumCost(minimum));
        }

        let mut initial = vec![f64::INFINITY; query.len() + 1];
        initial[0] = 0.0;
        for index in 1..=query.len() {
            let context = EditContext::new(&query, index - 1, &[], None);
            initial[index] = costs
                .deletion_cost(&context, query[index - 1])
                .and_then(valid_cost)
                .map_or(f64::INFINITY, |cost| initial[index - 1] + cost);
        }
        let (traversal, root) = TraversalSession::capture(root);
        let (mut pending, path_arena) = acquire_traversal_buffers_with_capacity(1, 64);
        pending.push_back(Pending {
            position: root,
            label: None,
            parent: NO_PATH,
            column: initial,
        });
        Ok(Self {
            query,
            max_cost,
            costs,
            pending,
            traversal,
            path_arena,
            prefix_scratch: Vec::with_capacity(64),
            column_pool: Vec::with_capacity(32),
            stats: ContextualQueryStats::default(),
        })
    }

    /// Construct from any compatible dictionary.
    pub fn from_dictionary<D>(
        dictionary: &D,
        query: Vec<N::Unit>,
        max_cost: f64,
        costs: C,
    ) -> Result<Self, ContextualQueryError>
    where
        D: Dictionary<Node = N>,
    {
        Self::try_from_traversal_root(dictionary.traversal_root(), query, max_cost, costs)
    }

    /// Snapshot traversal and fail-closed validation counters.
    pub fn stats(&self) -> ContextualQueryStats {
        self.stats
    }

    fn materialize_prefix_into(
        entry: &Pending<N::Unit, N::SnapshotCursor>,
        path_arena: &[ContextualPathNode<N::Unit>],
        output: &mut Vec<N::Unit>,
    ) {
        output.clear();
        if let Some(label) = entry.label {
            output.push(label);
        }
        let mut current = entry.parent;
        while current != NO_PATH {
            let node = &path_arena[current];
            output.push(node.label);
            current = node.parent;
        }
        output.reverse();
    }
}

impl<N, C> Drop for ContextualQueryIterator<N, C>
where
    N: DictionaryNode,
    C: ContextualCost<N::Unit>,
{
    fn drop(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        let path_arena = std::mem::take(&mut self.path_arena);
        release_traversal_buffers(pending, path_arena);
    }
}

fn valid_cost(cost: f64) -> Option<f64> {
    (cost.is_finite() && cost >= 0.0).then_some(cost)
}

fn cost_or_infinity(stats: &mut ContextualQueryStats, cost: Option<f64>) -> f64 {
    match cost.and_then(valid_cost) {
        Some(cost) => cost,
        None => {
            stats.invalid_costs_rejected = stats.invalid_costs_rejected.saturating_add(1);
            f64::INFINITY
        }
    }
}

fn child_column<U, C>(
    query: &[U],
    costs: &C,
    stats: &mut ContextualQueryStats,
    parent_column: &[f64],
    prefix: &[U],
    unit: U,
    mut current: Vec<f64>,
) -> Vec<f64>
where
    U: CharUnit,
    C: ContextualCost<U>,
{
    current.clear();
    current.resize(query.len() + 1, f64::INFINITY);
    let insertion_context = EditContext::new(query, 0, prefix, Some(unit));
    current[0] =
        parent_column[0] + cost_or_infinity(stats, costs.insertion_cost(&insertion_context, unit));

    for index in 1..=query.len() {
        let query_unit = query[index - 1];
        let (insertion_cost, deletion_cost, substitution_cost) = {
            let context = EditContext::new(query, index - 1, prefix, Some(unit));
            (
                costs.insertion_cost(&context, unit),
                costs.deletion_cost(&context, query_unit),
                costs.substitution_cost(&context, query_unit, unit),
            )
        };
        let insertion = parent_column[index] + cost_or_infinity(stats, insertion_cost);
        let deletion = current[index - 1] + cost_or_infinity(stats, deletion_cost);
        let substitution = parent_column[index - 1] + cost_or_infinity(stats, substitution_cost);
        current[index] = insertion.min(deletion).min(substitution);
    }
    current
}

impl<N, C> Iterator for ContextualQueryIterator<N, C>
where
    N: DictionaryNode,
    N::Unit: CharUnit + std::hash::Hash,
    C: ContextualCost<N::Unit>,
{
    type Item = ContextualCandidate<N::Unit>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(pending) = self.pending.pop_front() {
            self.stats.nodes_visited = self.stats.nodes_visited.saturating_add(1);
            Self::materialize_prefix_into(&pending, &self.path_arena, &mut self.prefix_scratch);
            let mut child_parent = None;
            let query = &self.query;
            let costs = &self.costs;
            let stats = &mut self.stats;
            let queue = &mut self.pending;
            let paths = &mut self.path_arena;
            let prefix = &self.prefix_scratch;
            let parent_column = &pending.column;
            let column_pool = &mut self.column_pool;
            let max_cost = self.max_cost;
            let is_final = self.traversal.filter_map_edges_and_finality(
                pending.position,
                |unit| {
                    stats.edges_enumerated = stats.edges_enumerated.saturating_add(1);
                    let reusable = column_pool.pop().unwrap_or_default();
                    let column =
                        child_column(query, costs, stats, parent_column, prefix, unit, reusable);
                    if column.iter().any(|cost| *cost <= max_cost) {
                        Some(column)
                    } else {
                        stats.subtrees_pruned = stats.subtrees_pruned.saturating_add(1);
                        column_pool.push(column);
                        None
                    }
                },
                |unit, child_position, column| {
                    let parent = *child_parent.get_or_insert_with(|| match pending.label {
                        Some(label) => {
                            let depth = if pending.parent == NO_PATH {
                                1
                            } else {
                                paths[pending.parent].depth.saturating_add(1)
                            };
                            let index = paths.len();
                            paths.push(ContextualPathNode {
                                label,
                                parent: pending.parent,
                                depth,
                            });
                            index
                        }
                        None => NO_PATH,
                    });
                    queue.push_back(Pending {
                        position: child_position,
                        label: Some(unit),
                        parent,
                        column,
                    });
                },
            );
            let distance = pending.column[self.query.len()];
            self.column_pool.push(pending.column);
            let accepted = is_final && distance <= self.max_cost;

            if accepted {
                if !self.traversal.accepts_final_units(&self.prefix_scratch) {
                    continue;
                }
                return Some(ContextualCandidate {
                    units: self.prefix_scratch.clone(),
                    distance,
                });
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<N, C> std::iter::FusedIterator for ContextualQueryIterator<N, C>
where
    N: DictionaryNode,
    N::Unit: CharUnit + std::hash::Hash,
    C: ContextualCost<N::Unit>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::{EnglishSoftC, PositionalSilentE};
    use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;

    #[test]
    fn reference_surfaces_change_only_the_documented_context() {
        let dictionary = DoubleArrayTrieChar::from_terms(["sit", "cat", "rat", "rat"]);
        let soft: Vec<_> = ContextualQueryIterator::from_dictionary(
            &dictionary,
            "cit".chars().collect(),
            0.25,
            EnglishSoftC::default(),
        )
        .unwrap()
        .collect();
        assert_eq!(soft.len(), 1);
        assert_eq!(soft[0].units.iter().collect::<String>(), "sit");
        assert_eq!(soft[0].distance, 0.25);

        let silent: Vec<_> = ContextualQueryIterator::from_dictionary(
            &dictionary,
            "rate".chars().collect(),
            0.25,
            PositionalSilentE::default(),
        )
        .unwrap()
        .collect();
        assert_eq!(silent.len(), 1);
        assert_eq!(silent[0].units.iter().collect::<String>(), "rat");
    }
}
