//! Iterative dictionary × language-product traversal.

use super::{Frontier, LanguageAutomaton, LanguageProduct};
use crate::transducer::dictionary_traversal::{
    CursorNativePath, DeferredNodeSource, ParentArenaPath, PathFrontier, ResultPathStrategy,
    TraversalCursor, TraversalSession,
};
#[cfg(any(feature = "bindings-phonetic", feature = "phonetic-rules"))]
use libdictenstein::MappedDictionaryNode;
use libdictenstein::{CharUnit, Dictionary, DictionaryNode, DictionaryTraversalRoot};
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

/// Compact query-local name for one canonical language-product frontier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct LanguageFrontierId(usize);

/// Collision-checked state interner and observed-edge cache.
///
/// The dictionary queue carries only `LanguageFrontierId`; complete bit-set
/// frontiers live once in this arena. No subset graph is constructed eagerly.
struct LanguageFrontierArena<U, S> {
    frontiers: Vec<Frontier<S>>,
    by_fingerprint: FxHashMap<u64, SmallVec<[LanguageFrontierId; 1]>>,
    transitions: FxHashMap<(LanguageFrontierId, U), Option<LanguageFrontierId>>,
}

impl<U, S> LanguageFrontierArena<U, S>
where
    U: Copy + Eq + Hash,
    S: Eq + Hash,
{
    fn new(initial: Frontier<S>) -> Self {
        let fingerprint = Self::fingerprint(&initial);
        let mut by_fingerprint = FxHashMap::default();
        by_fingerprint.insert(fingerprint, SmallVec::from_slice(&[LanguageFrontierId(0)]));
        Self {
            frontiers: vec![initial],
            by_fingerprint,
            transitions: FxHashMap::default(),
        }
    }

    #[inline]
    fn fingerprint(frontier: &Frontier<S>) -> u64 {
        let mut hasher = FxHasher::default();
        frontier.hash(&mut hasher);
        hasher.finish()
    }

    fn intern(&mut self, frontier: Frontier<S>) -> LanguageFrontierId {
        let fingerprint = Self::fingerprint(&frontier);
        if let Some(states) = self.by_fingerprint.get(&fingerprint) {
            if let Some(&id) = states.iter().find(|&&id| self.frontiers[id.0] == frontier) {
                return id;
            }
        }

        // The fingerprint narrows the lookup only; exact frontier equality is
        // required before reusing an ID.
        let id = LanguageFrontierId(self.frontiers.len());
        self.frontiers.push(frontier);
        self.by_fingerprint.entry(fingerprint).or_default().push(id);
        id
    }

    #[inline]
    fn step<L>(
        &mut self,
        product: &LanguageProduct<U, L>,
        source: LanguageFrontierId,
        unit: U,
    ) -> Option<LanguageFrontierId>
    where
        L: LanguageAutomaton<U, StateSet = S>,
    {
        if let Some(target) = self.transitions.get(&(source, unit)) {
            return *target;
        }
        let next = product.step(&self.frontiers[source.0], &unit);
        let target = (!next.is_empty()).then(|| self.intern(next));
        self.transitions.insert((source, unit), target);
        target
    }

    #[inline(always)]
    fn frontier(&self, id: LanguageFrontierId) -> &Frontier<S> {
        &self.frontiers[id.0]
    }
}

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
#[cfg(any(feature = "bindings-phonetic", feature = "phonetic-rules"))]
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
/// The queue owns dictionary path traces and one machine-word frontier ID.
/// Complete canonical frontiers live once in the query-local arena. No
/// recursion is used, so dictionary depth cannot overflow the process stack.
pub struct LanguageQueryIterator<N, L>
where
    N: DictionaryNode,
    L: LanguageAutomaton<N::Unit>,
{
    inner: PathLanguageQuery<N, L>,
}

enum PathLanguageQuery<N, L>
where
    N: DictionaryNode,
    L: LanguageAutomaton<N::Unit>,
{
    Parent(LanguageQueryCore<N, L, ParentArenaPath>),
    Cursor(LanguageQueryCore<N, L, CursorNativePath>),
}

struct LanguageQueryCore<N, L, K>
where
    N: DictionaryNode,
    L: LanguageAutomaton<N::Unit>,
    K: ResultPathStrategy<N>,
{
    product: LanguageProduct<N::Unit, L>,
    frontiers: LanguageFrontierArena<N::Unit, L::StateSet>,
    pending: VecDeque<PathFrontier<K::Trace, LanguageFrontierId>>,
    traversal: TraversalSession<N>,
    path_storage: K::Storage,
    stats: LanguageQueryStats,
}

/// Mapped result adapter over the same language-product traversal engine.
///
/// It permits compact flat-graph traversal because an accepting node escapes
/// only as its resolved value, never as a backend-owned child handle.
#[cfg(any(feature = "bindings-phonetic", feature = "phonetic-rules"))]
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
        root: TraversalCursor<N::SnapshotCursor>,
        product: LanguageProduct<N::Unit, L>,
    ) -> Self {
        Self {
            inner: PathLanguageQuery::new(root, traversal, product),
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
        self.inner.product()
    }

    /// Snapshot traversal counters.
    pub fn stats(&self) -> LanguageQueryStats {
        self.inner.stats()
    }

    #[cfg(any(feature = "bindings-phonetic", feature = "phonetic-rules"))]
    fn traversal(&self) -> &TraversalSession<N> {
        self.inner.traversal()
    }

    fn next_source(&mut self) -> Option<PendingLanguageMatch<N>>
    where
        N::Unit: CharUnit,
    {
        self.inner.next_source()
    }

    fn resolve_node(&self, source: DeferredNodeSource<N>) -> N {
        self.inner.traversal().resolve_node(source)
    }
}

impl<N, L> PathLanguageQuery<N, L>
where
    N: DictionaryNode,
    N::Unit: CharUnit,
    L: LanguageAutomaton<N::Unit>,
{
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        product: LanguageProduct<N::Unit, L>,
    ) -> Self {
        if traversal.supports_cursor_key_units() {
            Self::Cursor(LanguageQueryCore::new(root, traversal, product))
        } else {
            Self::Parent(LanguageQueryCore::new(root, traversal, product))
        }
    }

    fn product(&self) -> &LanguageProduct<N::Unit, L> {
        match self {
            Self::Parent(core) => &core.product,
            Self::Cursor(core) => &core.product,
        }
    }

    fn stats(&self) -> LanguageQueryStats {
        match self {
            Self::Parent(core) => core.stats,
            Self::Cursor(core) => core.stats,
        }
    }

    fn traversal(&self) -> &TraversalSession<N> {
        match self {
            Self::Parent(core) => &core.traversal,
            Self::Cursor(core) => &core.traversal,
        }
    }

    fn next_source(&mut self) -> Option<PendingLanguageMatch<N>> {
        match self {
            Self::Parent(core) => core.next_source(),
            Self::Cursor(core) => core.next_source(),
        }
    }
}

impl<N, L, K> LanguageQueryCore<N, L, K>
where
    N: DictionaryNode,
    N::Unit: CharUnit,
    L: LanguageAutomaton<N::Unit>,
    K: ResultPathStrategy<N>,
{
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        product: LanguageProduct<N::Unit, L>,
    ) -> Self {
        let frontier = product.initial_frontier();
        let frontiers = LanguageFrontierArena::new(frontier);
        let (mut pending, path_storage) = K::cold_queue();
        pending.push_back(PathFrontier::new(K::root(root), LanguageFrontierId(0)));
        Self {
            product,
            frontiers,
            pending,
            traversal,
            path_storage,
            stats: LanguageQueryStats::default(),
        }
    }

    fn next_source(&mut self) -> Option<PendingLanguageMatch<N>> {
        while let Some(entry) = self.pending.pop_front() {
            #[cfg(feature = "perf-instrumentation")]
            {
                self.stats.nodes_visited = self.stats.nodes_visited.saturating_add(1);
            }
            let mut expansion = K::begin_expansion(&entry.trace);
            let product = &self.product;
            let frontiers = &mut self.frontiers;
            let pending = &mut self.pending;
            let path_storage = &mut self.path_storage;
            #[cfg(feature = "perf-instrumentation")]
            let stats = &mut self.stats;
            let final_source = self.traversal.filter_map_edges_and_final_source(
                K::position(&entry.trace),
                |unit| {
                    #[cfg(feature = "perf-instrumentation")]
                    {
                        stats.edges_enumerated = stats.edges_enumerated.saturating_add(1);
                    }
                    frontiers.step(product, entry.frontier, unit)
                },
                |unit, child_position, frontier| {
                    pending.push_back(PathFrontier::new(
                        K::child_trace(
                            &entry.trace,
                            &mut expansion,
                            unit,
                            child_position,
                            path_storage,
                        ),
                        frontier,
                    ));
                },
            );
            let distance = final_source.as_ref().and_then(|_| {
                self.product
                    .min_accepting_distance(self.frontiers.frontier(entry.frontier))
            });

            if let Some(distance) = distance {
                let units = K::materialize_units(&entry.trace, &self.traversal, &self.path_storage);
                if !self.traversal.accepts_final_units(&units) {
                    continue;
                }
                return Some(PendingLanguageMatch {
                    units,
                    distance,
                    source: final_source.expect("an accepting final retains its node source"),
                });
            }
        }
        None
    }
}

#[cfg(any(feature = "bindings-phonetic", feature = "phonetic-rules"))]
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

#[cfg(any(feature = "bindings-phonetic", feature = "phonetic-rules"))]
impl<N, L> Iterator for MappedLanguageQueryIterator<N, L>
where
    N: MappedDictionaryNode,
    N::Unit: CharUnit,
    L: LanguageAutomaton<N::Unit>,
{
    type Item = MappedLanguageMatch<N>;

    fn next(&mut self) -> Option<Self::Item> {
        let match_source = self.inner.next_source()?;
        let requires_units = self.inner.traversal().requires_final_units();
        let value = self.inner.traversal().resolve_final_value(
            match_source.source,
            requires_units.then_some(match_source.units.as_slice()),
        );
        Some(MappedLanguageMatch {
            units: match_source.units,
            distance: match_source.distance,
            value,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[cfg(any(feature = "bindings-phonetic", feature = "phonetic-rules"))]
impl<N, L> std::iter::FusedIterator for MappedLanguageQueryIterator<N, L>
where
    N: MappedDictionaryNode,
    N::Unit: CharUnit,
    L: LanguageAutomaton<N::Unit>,
{
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
        let node = self.resolve_node(match_source.source);
        Some(LanguageMatch {
            units: match_source.units,
            distance: match_source.distance,
            node,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<N, L> std::iter::FusedIterator for LanguageQueryIterator<N, L>
where
    N: DictionaryNode,
    N::Unit: CharUnit,
    L: LanguageAutomaton<N::Unit>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::language::SmallDfa;

    #[test]
    fn language_frontiers_are_interned_once_and_edges_are_constructed_on_demand() {
        let mut language = SmallDfa::new();
        let accepting = language.add_state(true).expect("small fixture state");
        language
            .add_transition(0, b'a', accepting)
            .expect("small fixture edge");
        let product = LanguageProduct::new(language, 1);
        let mut arena = LanguageFrontierArena::new(product.initial_frontier());

        assert_eq!(arena.frontiers.len(), 1);
        assert!(arena.transitions.is_empty());
        let first = arena
            .step(&product, LanguageFrontierId(0), b'a')
            .expect("matching product edge remains live");
        let states_after_first = arena.frontiers.len();
        assert_eq!(arena.transitions.len(), 1);

        let repeated = arena
            .step(&product, LanguageFrontierId(0), b'a')
            .expect("cached matching product edge remains live");
        assert_eq!(repeated, first);
        assert_eq!(arena.frontiers.len(), states_after_first);
        assert_eq!(arena.transitions.len(), 1);
        assert_eq!(
            std::mem::size_of::<LanguageFrontierId>(),
            std::mem::size_of::<usize>()
        );
    }
}
