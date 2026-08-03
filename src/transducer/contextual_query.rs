//! Opt-in trie traversal for context-dependent edit costs.

use super::{ContextualCost, EditContext};
use libdictenstein::{CharUnit, Dictionary, DictionaryNode};
use rustc_hash::FxHashSet;
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

struct Pending<N: DictionaryNode> {
    node: N,
    prefix: Vec<N::Unit>,
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
    pending: VecDeque<Pending<N>>,
    seen: FxHashSet<Vec<N::Unit>>,
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
        let mut pending = VecDeque::with_capacity(1);
        pending.push_back(Pending {
            node: root,
            prefix: Vec::new(),
            column: initial,
        });
        Ok(Self {
            query,
            max_cost,
            costs,
            pending,
            seen: FxHashSet::default(),
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
        Self::try_new(dictionary.root(), query, max_cost, costs)
    }

    /// Snapshot traversal and fail-closed validation counters.
    pub fn stats(&self) -> ContextualQueryStats {
        self.stats
    }

    fn cost_or_infinity(&mut self, cost: Option<f64>) -> f64 {
        match cost.and_then(valid_cost) {
            Some(cost) => cost,
            None => {
                self.stats.invalid_costs_rejected =
                    self.stats.invalid_costs_rejected.saturating_add(1);
                f64::INFINITY
            }
        }
    }

    fn child_column(&mut self, pending: &Pending<N>, unit: N::Unit) -> Vec<f64> {
        let mut current = vec![f64::INFINITY; self.query.len() + 1];
        let insertion_context = EditContext::new(&self.query, 0, &pending.prefix, Some(unit));
        current[0] = pending.column[0]
            + self.cost_or_infinity(self.costs.insertion_cost(&insertion_context, unit));

        for index in 1..=self.query.len() {
            let query_unit = self.query[index - 1];
            let (insertion_cost, deletion_cost, substitution_cost) = {
                let context = EditContext::new(&self.query, index - 1, &pending.prefix, Some(unit));
                (
                    self.costs.insertion_cost(&context, unit),
                    self.costs.deletion_cost(&context, query_unit),
                    self.costs.substitution_cost(&context, query_unit, unit),
                )
            };
            let insertion = pending.column[index] + self.cost_or_infinity(insertion_cost);
            let deletion = current[index - 1] + self.cost_or_infinity(deletion_cost);
            let substitution = pending.column[index - 1] + self.cost_or_infinity(substitution_cost);
            current[index] = insertion.min(deletion).min(substitution);
        }
        current
    }
}

fn valid_cost(cost: f64) -> Option<f64> {
    (cost.is_finite() && cost >= 0.0).then_some(cost)
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
            let accepted = pending.node.is_final()
                && pending.column[self.query.len()] <= self.max_cost
                && self.seen.insert(pending.prefix.clone());

            for (unit, child) in pending.node.edges() {
                self.stats.edges_enumerated = self.stats.edges_enumerated.saturating_add(1);
                let column = self.child_column(&pending, unit);
                if column.iter().any(|cost| *cost <= self.max_cost) {
                    let mut prefix = pending.prefix.clone();
                    prefix.push(unit);
                    self.pending.push_back(Pending {
                        node: child,
                        prefix,
                        column,
                    });
                } else {
                    self.stats.subtrees_pruned = self.stats.subtrees_pruned.saturating_add(1);
                }
            }

            if accepted {
                return Some(ContextualCandidate {
                    units: pending.prefix,
                    distance: pending.column[self.query.len()],
                });
            }
        }
        None
    }
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
