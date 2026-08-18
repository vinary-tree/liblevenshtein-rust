//! Iterative dictionary × language-product traversal.

use super::{Frontier, LanguageAutomaton, LanguageProduct};
use crate::transducer::dictionary_traversal::{DeferredNodeSource, TraversalSession};
#[cfg(feature = "bindings-phonetic")]
use libdictenstein::MappedDictionaryNode;
use libdictenstein::{
    CharUnit, Dictionary, DictionaryNode, DictionaryTraversalRoot, SnapshotTraversalCursor,
};
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
struct Pending<U: CharUnit, S> {
    position: SnapshotTraversalCursor,
    label: Option<U>,
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

/// One accepting dictionary value reached by a mapped language-product walk.
#[cfg(feature = "bindings-phonetic")]
pub(crate) struct MappedLanguageMatch<N: MappedDictionaryNode> {
    pub(crate) units: Vec<N::Unit>,
    pub(crate) distance: u8,
    pub(crate) value: Option<N::Value>,
}

struct PendingLanguageMatch<N: DictionaryNode> {
    units: Vec<N::Unit>,
    distance: u8,
    source: DeferredNodeSource<N>,
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
    pending: VecDeque<Pending<N::Unit, L::StateSet>>,
    traversal: TraversalSession<N>,
    path: Vec<PathUnit<N::Unit>>,
    stats: LanguageQueryStats,
}

/// Mapped result adapter over the same language-product traversal engine.
///
/// It permits compact flat-graph traversal because an accepting node escapes
/// only as its resolved value, never as a backend-owned child handle.
#[cfg(feature = "bindings-phonetic")]
pub(crate) struct MappedLanguageQueryIterator<N, L>
where
    N: MappedDictionaryNode,
    L: LanguageAutomaton<N::Unit>,
{
    inner: LanguageQueryIterator<N, L>,
}

impl<N, L> LanguageQueryIterator<N, L>
where
    N: DictionaryNode,
    L: LanguageAutomaton<N::Unit>,
{
    /// Start at `root` using `product`.
    pub fn new(root: N, product: LanguageProduct<N::Unit, L>) -> Self {
        Self::from_traversal_root(DictionaryTraversalRoot::owned(root), product)
    }

    fn from_traversal_root(
        root: DictionaryTraversalRoot<N>,
        product: LanguageProduct<N::Unit, L>,
    ) -> Self {
        let (traversal, root) = TraversalSession::capture_nodes(root);
        Self::from_session(traversal, root, product)
    }

    fn from_session(
        traversal: TraversalSession<N>,
        root: SnapshotTraversalCursor,
        product: LanguageProduct<N::Unit, L>,
    ) -> Self {
        let frontier = product.initial_frontier();
        let mut pending = VecDeque::with_capacity(1);
        pending.push_back(Pending {
            position: root,
            label: None,
            parent: NO_PARENT,
            frontier,
        });
        Self {
            product,
            pending,
            traversal,
            path: Vec::new(),
            stats: LanguageQueryStats::default(),
        }
    }

    /// Construct from any dictionary with a matching unit type.
    pub fn from_dictionary<D>(dictionary: &D, product: LanguageProduct<N::Unit, L>) -> Self
    where
        D: Dictionary<Node = N>,
    {
        // Node-returning matches need backend-native cursors rather than the
        // value-less flat projection, so avoid constructing a graph that the
        // deferred-node capture would immediately discard.
        Self::from_traversal_root(DictionaryTraversalRoot::owned(dictionary.root()), product)
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

    fn next_source(&mut self) -> Option<PendingLanguageMatch<N>>
    where
        N::Unit: CharUnit,
    {
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
            let final_source = self.traversal.filter_map_edges_and_final_source(
                entry.position,
                |unit| {
                    #[cfg(feature = "perf-instrumentation")]
                    {
                        stats.edges_enumerated = stats.edges_enumerated.saturating_add(1);
                    }
                    let frontier = product.step(&entry.frontier, &unit);
                    (!frontier.is_empty()).then_some(frontier)
                },
                |unit, child_position, frontier| {
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
                        position: child_position,
                        label: Some(unit),
                        parent,
                        frontier,
                    });
                },
            );
            let distance = final_source
                .as_ref()
                .and_then(|_| self.product.min_accepting_distance(&entry.frontier));

            if let Some(distance) = distance {
                return Some(PendingLanguageMatch {
                    units: self.materialize(entry.label, entry.parent),
                    distance,
                    source: final_source.expect("an accepting final retains its node source"),
                });
            }
        }
        None
    }
}

#[cfg(feature = "bindings-phonetic")]
impl<N, L> MappedLanguageQueryIterator<N, L>
where
    N: MappedDictionaryNode,
    N::Unit: CharUnit,
    L: LanguageAutomaton<N::Unit>,
{
    pub(crate) fn from_traversal_root(
        root: DictionaryTraversalRoot<N>,
        product: LanguageProduct<N::Unit, L>,
    ) -> Self {
        let (traversal, root) = TraversalSession::capture_mapped(root);
        Self {
            inner: LanguageQueryIterator::from_session(traversal, root, product),
        }
    }
}

#[cfg(feature = "bindings-phonetic")]
impl<N, L> Iterator for MappedLanguageQueryIterator<N, L>
where
    N: MappedDictionaryNode,
    N::Unit: CharUnit,
    L: LanguageAutomaton<N::Unit>,
{
    type Item = MappedLanguageMatch<N>;

    fn next(&mut self) -> Option<Self::Item> {
        let match_source = self.inner.next_source()?;
        let value = self
            .inner
            .traversal
            .resolve_final_value(match_source.source);
        Some(MappedLanguageMatch {
            units: match_source.units,
            distance: match_source.distance,
            value,
        })
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
        let match_source = self.next_source()?;
        let node = self.traversal.resolve_node(match_source.source);
        Some(LanguageMatch {
            units: match_source.units,
            distance: match_source.distance,
            node,
        })
    }
}
