//! Iterative dictionary × language-product traversal.

use super::{Frontier, LanguageAutomaton, LanguageProduct};
use libdictenstein::{CharUnit, Dictionary, DictionaryNode};
use std::collections::VecDeque;

const NO_PARENT: usize = usize::MAX;

/// Actual traversal work performed by [`LanguageQueryIterator`].
///
/// Counters are updated when the `perf-instrumentation` feature is enabled and
/// remain zero otherwise, keeping the production iterator's hot path unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LanguageQueryStats {
    /// Dictionary nodes removed from the work queue.
    pub nodes_visited: usize,
    /// Outgoing dictionary edges enumerated before frontier pruning.
    pub edges_enumerated: usize,
}

#[derive(Clone, Debug)]
struct PathUnit<U> {
    unit: U,
    parent: usize,
    depth: usize,
}

#[derive(Clone, Debug)]
struct Pending<N: DictionaryNode, S> {
    node: N,
    label: Option<N::Unit>,
    parent: usize,
    frontier: Frontier<S>,
}

/// One accepting dictionary node reached by a language-product traversal.
#[derive(Clone, Debug)]
pub struct LanguageMatch<N: DictionaryNode> {
    /// Raw root-to-node dictionary units.
    pub units: Vec<N::Unit>,
    /// Exact minimum distance to the language.
    pub distance: u8,
    /// The accepting dictionary node, retained for mapped-value access.
    pub node: N,
}

/// Lazy, iterative dictionary × language-product iterator.
///
/// The queue owns cloned dictionary nodes and bounded frontiers. No recursion is
/// used, so dictionary depth cannot overflow the process stack.
pub struct LanguageQueryIterator<N, L>
where
    N: DictionaryNode,
    L: LanguageAutomaton<N::Unit>,
{
    product: LanguageProduct<N::Unit, L>,
    pending: VecDeque<Pending<N, L::StateSet>>,
    path: Vec<PathUnit<N::Unit>>,
    stats: LanguageQueryStats,
}

impl<N, L> LanguageQueryIterator<N, L>
where
    N: DictionaryNode,
    L: LanguageAutomaton<N::Unit>,
{
    /// Start at `root` using `product`.
    pub fn new(root: N, product: LanguageProduct<N::Unit, L>) -> Self {
        let frontier = product.initial_frontier();
        let mut pending = VecDeque::with_capacity(1);
        pending.push_back(Pending {
            node: root,
            label: None,
            parent: NO_PARENT,
            frontier,
        });
        Self {
            product,
            pending,
            path: Vec::new(),
            stats: LanguageQueryStats::default(),
        }
    }

    /// Construct from any dictionary with a matching unit type.
    pub fn from_dictionary<D>(dictionary: &D, product: LanguageProduct<N::Unit, L>) -> Self
    where
        D: Dictionary<Node = N>,
    {
        Self::new(dictionary.root(), product)
    }

    /// Borrow the language product driving the traversal.
    pub fn product(&self) -> &LanguageProduct<N::Unit, L> {
        &self.product
    }

    /// Snapshot traversal counters.
    pub fn stats(&self) -> LanguageQueryStats {
        self.stats
    }

    fn materialize(&self, label: Option<N::Unit>, parent: usize) -> Vec<N::Unit> {
        let mut result = Vec::with_capacity(if parent == NO_PARENT {
            usize::from(label.is_some())
        } else {
            self.path[parent].depth + usize::from(label.is_some())
        });
        if let Some(unit) = label {
            result.push(unit);
        }
        let mut cursor = parent;
        while cursor != NO_PARENT {
            result.push(self.path[cursor].unit);
            cursor = self.path[cursor].parent;
        }
        result.reverse();
        result
    }
}

impl<N, L> Iterator for LanguageQueryIterator<N, L>
where
    N: DictionaryNode,
    N::Unit: CharUnit,
    L: LanguageAutomaton<N::Unit>,
{
    type Item = LanguageMatch<N>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.pending.pop_front() {
            #[cfg(feature = "perf-instrumentation")]
            {
                self.stats.nodes_visited = self.stats.nodes_visited.saturating_add(1);
            }
            let mut child_parent = None;
            let product = &mut self.product;
            let pending = &mut self.pending;
            let path = &mut self.path;
            #[cfg(feature = "perf-instrumentation")]
            let stats = &mut self.stats;
            let is_final = entry.node.filter_map_edges_and_finality(
                |unit| {
                    #[cfg(feature = "perf-instrumentation")]
                    {
                        stats.edges_enumerated = stats.edges_enumerated.saturating_add(1);
                    }
                    let frontier = product.step(&entry.frontier, &unit);
                    (!frontier.is_empty()).then_some(frontier)
                },
                |unit, child, frontier| {
                    let parent = *child_parent.get_or_insert_with(|| match entry.label {
                        Some(label) => {
                            let depth = if entry.parent == NO_PARENT {
                                1
                            } else {
                                path[entry.parent].depth.saturating_add(1)
                            };
                            let index = path.len();
                            path.push(PathUnit {
                                unit: label,
                                parent: entry.parent,
                                depth,
                            });
                            index
                        }
                        None => entry.parent,
                    });
                    pending.push_back(Pending {
                        node: child,
                        label: Some(unit),
                        parent,
                        frontier,
                    });
                },
            );
            let distance = is_final
                .then(|| self.product.min_accepting_distance(&entry.frontier))
                .flatten();

            if let Some(distance) = distance {
                return Some(LanguageMatch {
                    units: self.materialize(entry.label, entry.parent),
                    distance,
                    node: entry.node,
                });
            }
        }
        None
    }
}
