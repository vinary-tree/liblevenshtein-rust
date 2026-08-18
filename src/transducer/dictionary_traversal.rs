//! Shared snapshot-cursor traversal with an owned-node compatibility arena.

use libdictenstein::{
    DictionaryNode, DictionaryTraversalRoot, MappedDictionaryNode, SnapshotTraversalCursor,
    SnapshotTraversalGraph,
};
use std::sync::Arc;

#[cfg(feature = "perf-instrumentation")]
use libdictenstein::SnapshotNodeIdentity;

struct OwnedTraversalArena<N> {
    nodes: Vec<Option<N>>,
    free: Vec<usize>,
}

impl<N> OwnedTraversalArena<N> {
    fn new(root: N) -> Self {
        Self {
            nodes: vec![Some(root)],
            free: Vec::new(),
        }
    }

    #[inline]
    fn take(&mut self, cursor: SnapshotTraversalCursor) -> N {
        let index = cursor.get() - 1;
        let node = self.nodes[index]
            .take()
            .expect("an owned traversal cursor is expanded exactly once");
        self.free.push(index);
        node
    }

    #[inline]
    fn insert(&mut self, node: N) -> SnapshotTraversalCursor {
        let index = match self.free.pop() {
            Some(index) => {
                debug_assert!(self.nodes[index].is_none());
                self.nodes[index] = Some(node);
                index
            }
            None => {
                self.nodes.push(Some(node));
                self.nodes.len() - 1
            }
        };
        SnapshotTraversalCursor::new(index + 1).expect("an owned traversal arena index is non-zero")
    }
}

enum TraversalMode<N: DictionaryNode> {
    /// Compatibility arena. Each queued cursor owns exactly one node slot; an
    /// expanded node is taken from its slot and dropped after its children have
    /// been appended, so no clone is needed merely to satisfy borrow rules.
    Owned(OwnedTraversalArena<N>),
    /// One immutable owner plus either a concrete flat graph or backend-native
    /// cursors (the latter is used by a mutated DynamicDawg revision).
    Captured {
        owner: N,
        graph: Option<Arc<SnapshotTraversalGraph<N::Unit>>>,
    },
}

/// Query-start dictionary revision whose queued locations are always one
/// pointer-sized cursor, regardless of backend representation.
pub(crate) struct TraversalSession<N: DictionaryNode> {
    mode: TraversalMode<N>,
}

/// Deferred owned-node source produced only for a final dictionary node.
///
/// Owned fallbacks retain that one expanded node until the caller establishes
/// that its automaton state is in range. Captured backends retain only a copy
/// cursor and materialize an owned handle lazily through the session owner.
pub(crate) enum DeferredNodeSource<N> {
    Owned(N),
    Captured(SnapshotTraversalCursor),
}

pub(crate) type MappedValueSource<N> = DeferredNodeSource<N>;

impl<N: DictionaryNode> TraversalSession<N> {
    /// Capture a traversal root and return its uniform root cursor.
    pub(crate) fn capture(root: DictionaryTraversalRoot<N>) -> (Self, SnapshotTraversalCursor) {
        let (owner, graph) = root.into_parts();
        if cursor_traversal_disabled() {
            return Self::owned(owner);
        }

        if let Some(graph) = graph {
            let root = graph.root_cursor();
            return (
                Self {
                    mode: TraversalMode::Captured {
                        owner,
                        graph: Some(graph),
                    },
                },
                root,
            );
        }

        if let Some(root) = owner.snapshot_root_cursor() {
            return (
                Self {
                    mode: TraversalMode::Captured { owner, graph: None },
                },
                root,
            );
        }

        Self::owned(owner)
    }

    fn owned(root: N) -> (Self, SnapshotTraversalCursor) {
        let root_cursor = SnapshotTraversalCursor::new(1).expect("one is non-zero");
        (
            Self {
                mode: TraversalMode::Owned(OwnedTraversalArena::new(root)),
            },
            root_cursor,
        )
    }

    /// Capture a revision whose accepting cursors may need to escape later as
    /// owned node handles. Value-less flat graphs are intentionally ignored.
    pub(crate) fn capture_nodes(
        root: DictionaryTraversalRoot<N>,
    ) -> (Self, SnapshotTraversalCursor) {
        let (owner, _) = root.into_parts();
        if !cursor_traversal_disabled() && owner.supports_snapshot_cursor_nodes() {
            if let Some(cursor) = owner.snapshot_root_cursor() {
                return (
                    Self {
                        mode: TraversalMode::Captured { owner, graph: None },
                    },
                    cursor,
                );
            }
        }
        Self::owned(owner)
    }

    /// Read finality and project outgoing edges without changing the queued
    /// cursor representation.
    #[inline(always)]
    pub(crate) fn filter_map_edges_and_finality<T, P, F>(
        &mut self,
        cursor: SnapshotTraversalCursor,
        project: P,
        visitor: F,
    ) -> bool
    where
        P: FnMut(N::Unit) -> Option<T>,
        F: FnMut(N::Unit, SnapshotTraversalCursor, T),
    {
        match &mut self.mode {
            TraversalMode::Owned(arena) => {
                let node = arena.take(cursor);
                let mut visitor = visitor;
                node.filter_map_edges_and_finality(project, |label, child, value| {
                    let child_cursor = arena.insert(child);
                    visitor(label, child_cursor, value);
                })
            }
            TraversalMode::Captured {
                owner: _,
                graph: Some(graph),
            } => {
                // SAFETY: captured cursors are the validated root or targets
                // read from this exact immutable graph.
                let edges = unsafe { graph.edges_and_finality_unchecked(cursor) };
                let is_final = edges.is_final();
                let mut project = project;
                let mut visitor = visitor;
                for edge in edges.edges() {
                    let label = edge.label();
                    if let Some(value) = project(label) {
                        visitor(label, edge.target_cursor(), value);
                    }
                }
                is_final
            }
            TraversalMode::Captured { owner, graph: None } => {
                // SAFETY: `capture` obtains the root cursor from `owner`; all
                // descendants are produced by this same retained revision and
                // cursors never escape the session.
                unsafe {
                    owner
                        .filter_map_snapshot_cursor_edges_and_finality(cursor, project, visitor)
                        .expect(
                            "a backend that returns a root cursor must support cursor traversal",
                        )
                }
            }
        }
    }

    /// Expand one node and retain its owned representation only when final.
    #[inline(always)]
    pub(crate) fn filter_map_edges_and_final_source<T, P, F>(
        &mut self,
        cursor: SnapshotTraversalCursor,
        project: P,
        visitor: F,
    ) -> Option<DeferredNodeSource<N>>
    where
        P: FnMut(N::Unit) -> Option<T>,
        F: FnMut(N::Unit, SnapshotTraversalCursor, T),
    {
        match &mut self.mode {
            TraversalMode::Owned(arena) => {
                let node = arena.take(cursor);
                let mut visitor = visitor;
                let is_final =
                    node.filter_map_edges_and_finality(project, |label, child, value| {
                        let child_cursor = arena.insert(child);
                        visitor(label, child_cursor, value);
                    });
                is_final.then_some(DeferredNodeSource::Owned(node))
            }
            TraversalMode::Captured { owner, graph: None } => {
                // SAFETY: a captured session retains the exact immutable owner
                // from which this cursor and all descendants originated.
                let is_final = unsafe {
                    owner
                        .filter_map_snapshot_cursor_edges_and_finality(cursor, project, visitor)
                        .expect("a native cursor backend supports cursor traversal")
                };
                is_final.then_some(DeferredNodeSource::Captured(cursor))
            }
            TraversalMode::Captured {
                owner: _,
                graph: Some(graph),
            } => {
                // SAFETY: captured cursors are the validated root or targets
                // read from this exact immutable graph.
                let edges = unsafe { graph.edges_and_finality_unchecked(cursor) };
                let is_final = edges.is_final();
                let mut project = project;
                let mut visitor = visitor;
                for edge in edges.edges() {
                    let label = edge.label();
                    if let Some(value) = project(label) {
                        visitor(label, edge.target_cursor(), value);
                    }
                }
                is_final.then_some(DeferredNodeSource::Captured(cursor))
            }
        }
    }

    /// Resolve a deferred accepting node into an owned handle.
    #[inline]
    pub(crate) fn resolve_node(&self, source: DeferredNodeSource<N>) -> N {
        match source {
            DeferredNodeSource::Owned(node) => node,
            DeferredNodeSource::Captured(cursor) => match &self.mode {
                TraversalMode::Captured { owner, graph: None } => {
                    // SAFETY: the source was created by this exact session and
                    // the retained owner still covers the cursor allocation.
                    unsafe { owner.snapshot_cursor_node(cursor) }
                        .expect("capture_nodes validated cursor node materialization")
                }
                TraversalMode::Captured { graph: Some(_), .. } => {
                    unreachable!("flat graph captures cannot materialize owned nodes")
                }
                TraversalMode::Owned(_) => {
                    unreachable!("captured sources do not belong to owned sessions")
                }
            },
        }
    }

    /// Release one queued owned fallback cursor that will never be expanded.
    #[inline]
    pub(crate) fn discard_unexpanded(&mut self, cursor: SnapshotTraversalCursor) {
        if let TraversalMode::Owned(arena) = &mut self.mode {
            drop(arena.take(cursor));
        }
    }

    #[cfg(feature = "perf-instrumentation")]
    pub(crate) fn product_identity(
        &self,
        cursor: SnapshotTraversalCursor,
    ) -> Option<TraversalProductIdentity> {
        match &self.mode {
            TraversalMode::Owned(arena) => arena.nodes[cursor.get() - 1]
                .as_ref()
                .and_then(DictionaryNode::snapshot_node_identity)
                .map(TraversalProductIdentity::Node),
            TraversalMode::Captured { .. } => Some(TraversalProductIdentity::Cursor(cursor)),
        }
    }

    #[cfg(test)]
    fn is_flat(&self) -> bool {
        matches!(self.mode, TraversalMode::Captured { graph: Some(_), .. })
    }

    #[cfg(test)]
    fn owned_slot_count(&self) -> Option<usize> {
        match &self.mode {
            TraversalMode::Owned(arena) => Some(arena.nodes.len()),
            TraversalMode::Captured { .. } => None,
        }
    }

    #[cfg(test)]
    fn owned_live_slot_count(&self) -> Option<usize> {
        match &self.mode {
            TraversalMode::Owned(arena) => {
                Some(arena.nodes.iter().filter(|node| node.is_some()).count())
            }
            TraversalMode::Captured { .. } => None,
        }
    }
}

impl<N: MappedDictionaryNode> TraversalSession<N> {
    /// Capture a mapped revision, preferring its compact immutable graph when
    /// the retained owner can resolve graph-local value cursors.
    pub(crate) fn capture_mapped(
        root: DictionaryTraversalRoot<N>,
    ) -> (Self, SnapshotTraversalCursor) {
        let (owner, graph) = root.into_parts();
        if !cursor_traversal_disabled() && owner.supports_snapshot_graph_values() {
            if let Some(graph) = graph {
                let root = graph.root_cursor();
                return (
                    Self {
                        mode: TraversalMode::Captured {
                            owner,
                            graph: Some(graph),
                        },
                    },
                    root,
                );
            }
        }
        if !cursor_traversal_disabled() && owner.supports_snapshot_cursor_values() {
            if let Some(cursor) = owner.snapshot_root_cursor() {
                return (
                    Self {
                        mode: TraversalMode::Captured { owner, graph: None },
                    },
                    cursor,
                );
            }
        }
        Self::owned(owner)
    }

    /// Resolve a final node's value after the automaton has accepted it.
    #[inline]
    pub(crate) fn resolve_final_value(&self, source: MappedValueSource<N>) -> Option<N::Value> {
        match source {
            MappedValueSource::Owned(node) => node.value_at_final(),
            MappedValueSource::Captured(cursor) => match &self.mode {
                TraversalMode::Captured { owner, graph: None } => {
                    // SAFETY: the source was created by this exact session and
                    // has not escaped the retained revision.
                    unsafe { owner.snapshot_cursor_value(cursor) }
                        .expect("capture_mapped validated cursor value access")
                }
                TraversalMode::Captured {
                    owner,
                    graph: Some(graph),
                } => {
                    // SAFETY: the graph and owner were captured together and
                    // the source cursor was produced by this exact graph.
                    unsafe { owner.snapshot_graph_cursor_value(graph, cursor) }
                        .expect("capture_mapped validated graph value access")
                }
                _ => unreachable!("captured value sources belong to a native mapped session"),
            },
        }
    }
}

#[cfg(feature = "perf-instrumentation")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum TraversalProductIdentity {
    Node(SnapshotNodeIdentity),
    Cursor(SnapshotTraversalCursor),
}

fn cursor_traversal_disabled() -> bool {
    use std::sync::OnceLock;
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_SNAPSHOT_CURSORS").is_some()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
    use libdictenstein::Dictionary;

    #[test]
    fn dynamic_dawg_uses_flat_capture_and_dat_uses_owned_arena() {
        let dynamic = DynamicDawgChar::<()>::from_sorted_terms(["a", "b"]);
        let (session, _) = TraversalSession::capture(dynamic.traversal_root());
        assert!(session.is_flat());

        let dat = DoubleArrayTrie::<()>::from_terms(["a", "b"]);
        let (session, _) = TraversalSession::capture(dat.traversal_root());
        assert!(!session.is_flat());
    }

    #[test]
    fn captured_session_keeps_its_revision_alive_across_mutation() {
        let dynamic = DynamicDawgChar::<()>::from_sorted_terms(["cat", "dog"]);
        let (mut session, root) = TraversalSession::capture(dynamic.traversal_root());
        dynamic.insert("cow");

        let mut labels = Vec::new();
        let is_final = session.filter_map_edges_and_finality(
            root,
            |_label| Some(()),
            |label, _, ()| labels.push(label),
        );
        assert!(!is_final);
        assert_eq!(labels, ['c', 'd']);
    }

    #[test]
    fn owned_arena_reuses_consumed_slots() {
        let dictionary = DoubleArrayTrie::<()>::from_terms(["abc"]);
        let (mut session, mut cursor) = TraversalSession::capture(dictionary.traversal_root());

        for expected in *b"abc" {
            let mut child = None;
            session.filter_map_edges_and_finality(
                cursor,
                |_label| Some(()),
                |label, next, ()| {
                    assert_eq!(label, expected);
                    child = Some(next);
                },
            );
            cursor = child.expect("the linear dictionary has one child");
        }

        assert_eq!(session.owned_slot_count(), Some(1));
    }

    #[test]
    fn discarded_owned_cursors_release_pruned_siblings_immediately() {
        let terms = (b'a'..=b'z')
            .map(|label| char::from(label).to_string())
            .collect::<Vec<_>>();
        let dictionary = DoubleArrayTrie::<()>::from_terms(terms);
        let (mut session, root) = TraversalSession::capture(dictionary.traversal_root());
        let mut children = Vec::new();
        session.filter_map_edges_and_finality(root, Some, |_label, child, _| children.push(child));
        assert_eq!(session.owned_live_slot_count(), Some(children.len()));

        for child in children {
            session.discard_unexpanded(child);
        }
        assert_eq!(session.owned_live_slot_count(), Some(0));
    }

    #[test]
    fn mapped_native_cursor_keeps_old_revision_and_resolves_value_lazily() {
        let dynamic = DynamicDawgChar::from_sorted_terms_with_values([("cat", 7_u64), ("dog", 8)]);
        let root_node = dynamic.root();
        let (mut session, mut position) =
            TraversalSession::capture_mapped(DictionaryTraversalRoot::owned(root_node));
        dynamic.insert_with_value("cow", 9);

        for expected in ['c', 'a', 't'] {
            let mut child = None;
            let final_source = session.filter_map_edges_and_final_source(
                position,
                |label| (label == expected).then_some(()),
                |_label, next, ()| child = Some(next),
            );
            assert!(final_source.is_none());
            position = child.expect("the captured old revision contains cat");
        }

        let final_source = session
            .filter_map_edges_and_final_source(position, |_| None::<()>, |_, _, _| {})
            .expect("cat is final in the captured revision");
        assert_eq!(session.resolve_final_value(final_source), Some(7));
    }
}
