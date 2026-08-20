//! Monomorphized helpers for transparent dictionary-node decorators.

use libdictenstein::{DictionaryNode, DictionaryTraversalRoot};

/// Preserve an inner dictionary's captured traversal projection while changing
/// only the retained root-node decorator.
///
/// A snapshot graph stores units and backend-local cursors, not concrete node
/// handles, so it remains valid when the owner is wrapped transparently. The
/// unit equality bound prevents accidental projection across alphabets.
#[inline]
pub(crate) fn map_transparent_traversal_root<N, W, F>(
    root: DictionaryTraversalRoot<N>,
    wrap: F,
) -> DictionaryTraversalRoot<W>
where
    N: DictionaryNode,
    W: DictionaryNode<
        Unit = N::Unit,
        SnapshotCursor = N::SnapshotCursor,
        SnapshotGraphValueHandle = N::SnapshotGraphValueHandle,
    >,
    F: FnOnce(N) -> W,
{
    let (graph, node) = root.into_parts().into_projection_and_root();
    if transparent_traversal_forwarding_disabled() {
        return DictionaryTraversalRoot::owned(wrap(node));
    }
    match graph {
        Some(graph) => DictionaryTraversalRoot::captured(wrap(node), graph),
        None => DictionaryTraversalRoot::owned(wrap(node)),
    }
}

/// Same-binary causal control for transparent-wrapper cursor forwarding.
#[inline]
pub(crate) fn transparent_traversal_forwarding_disabled() -> bool {
    #[cfg(feature = "benchmark-controls")]
    {
        use std::sync::OnceLock;
        static DISABLED: OnceLock<bool> = OnceLock::new();
        *DISABLED.get_or_init(|| {
            std::env::var_os("LIBLEVENSHTEIN_CAUSAL_DISABLE_WRAPPER_TRAVERSAL_FORWARDING").is_some()
        })
    }
    #[cfg(not(feature = "benchmark-controls"))]
    {
        false
    }
}

/// Implement every dictionary-node seam for a transparent newtype decorator.
///
/// The generated methods forward snapshot cursors, compact graph values,
/// identity, fused finality/edge visits, and predicate-first edge projection.
/// Each invocation supplies only the wrapper-specific child constructor;
/// monomorphization removes the helper boundary from traversal hot loops.
macro_rules! impl_dictionary_node_adapter {
    (
        [$($generic:tt)*],
        $node:ty,
        [$($bounds:tt)*],
        |$owner:ident, $child:ident| $wrap:expr,
        |$requires_owner:ident| $requires_final_units:expr,
        |$accepts_owner:ident, $units:ident| $accepts_final_units:expr
    ) => {
        impl<$($generic)*> libdictenstein::DictionaryNode for $node
        where
            $($bounds)*
        {
            type Unit = N::Unit;
            type SnapshotCursor = N::SnapshotCursor;
            type SnapshotGraphValueHandle = N::SnapshotGraphValueHandle;

            #[inline]
            fn requires_final_units(&self) -> bool {
                let $requires_owner = self;
                $requires_final_units
            }

            #[inline]
            fn accepts_final_units(&self, units: &[Self::Unit]) -> bool {
                let $accepts_owner = self;
                let $units = units;
                $accepts_final_units
            }

            #[inline]
            fn snapshot_node_identity(&self) -> Option<libdictenstein::SnapshotNodeIdentity> {
                if $crate::dictionary::node_adapter::transparent_traversal_forwarding_disabled() {
                    return None;
                }
                self.inner.snapshot_node_identity()
            }

            #[inline]
            fn snapshot_root_cursor(&self) -> Option<Self::SnapshotCursor> {
                if $crate::dictionary::node_adapter::transparent_traversal_forwarding_disabled() {
                    return None;
                }
                self.inner.snapshot_root_cursor()
            }

            #[inline]
            fn contains_snapshot_cursor(
                &self,
                cursor: Self::SnapshotCursor,
            ) -> bool {
                if $crate::dictionary::node_adapter::transparent_traversal_forwarding_disabled() {
                    return false;
                }
                self.inner.contains_snapshot_cursor(cursor)
            }

            #[inline]
            fn supports_snapshot_cursor_nodes(&self) -> bool {
                if $crate::dictionary::node_adapter::transparent_traversal_forwarding_disabled() {
                    return false;
                }
                self.inner.supports_snapshot_cursor_nodes()
            }

            #[inline]
            fn supports_snapshot_cursor_key_units(&self) -> bool {
                if $crate::dictionary::node_adapter::transparent_traversal_forwarding_disabled() {
                    return false;
                }
                self.inner.supports_snapshot_cursor_key_units()
            }

            #[inline]
            unsafe fn snapshot_cursor_key_units(
                &self,
                cursor: Self::SnapshotCursor,
            ) -> Option<Vec<Self::Unit>> {
                // SAFETY: this topology-preserving wrapper retains the exact
                // inner revision, root, labels, and cursor identity.
                unsafe { self.inner.snapshot_cursor_key_units(cursor) }
            }

            #[inline]
            unsafe fn snapshot_cursor_node(
                &self,
                cursor: Self::SnapshotCursor,
            ) -> Option<Self> {
                let $owner = self;
                // SAFETY: this transparent wrapper retains the exact inner
                // revision and forwards the caller's cursor unchanged.
                unsafe { self.inner.snapshot_cursor_node(cursor) }.map(|$child| $wrap)
            }

            #[inline]
            unsafe fn filter_map_snapshot_cursor_edges_and_finality<T, Project, Visitor>(
                &self,
                cursor: Self::SnapshotCursor,
                project: Project,
                visitor: Visitor,
            ) -> Option<bool>
            where
                Project: FnMut(Self::Unit) -> Option<T>,
                Visitor: FnMut(Self::Unit, Self::SnapshotCursor, T),
            {
                // SAFETY: cursor identity and retained revision are unchanged.
                unsafe {
                    self.inner
                        .filter_map_snapshot_cursor_edges_and_finality(cursor, project, visitor)
                }
            }

            #[inline]
            unsafe fn snapshot_cursor_is_final(
                &self,
                cursor: Self::SnapshotCursor,
            ) -> Option<bool> {
                // SAFETY: cursor identity and retained revision are unchanged.
                unsafe { self.inner.snapshot_cursor_is_final(cursor) }
            }

            #[inline]
            unsafe fn snapshot_cursor_transition(
                &self,
                cursor: Self::SnapshotCursor,
                label: Self::Unit,
            ) -> Option<Option<Self::SnapshotCursor>> {
                // SAFETY: cursor identity and retained revision are unchanged.
                unsafe { self.inner.snapshot_cursor_transition(cursor, label) }
            }

            #[inline]
            fn supports_efficient_snapshot_cursor_edge_paging(&self) -> bool {
                if $crate::dictionary::node_adapter::transparent_traversal_forwarding_disabled() {
                    return false;
                }
                self.inner
                    .supports_efficient_snapshot_cursor_edge_paging()
            }

            #[inline]
            unsafe fn visit_snapshot_cursor_edge_page<Visitor>(
                &self,
                cursor: Self::SnapshotCursor,
                start: usize,
                capacity: usize,
                visitor: Visitor,
            ) -> Option<(bool, usize)>
            where
                Visitor: FnMut(Self::Unit, Self::SnapshotCursor),
            {
                // SAFETY: cursor identity and retained revision are unchanged.
                unsafe {
                    self.inner
                        .visit_snapshot_cursor_edge_page(cursor, start, capacity, visitor)
                }
            }

            #[inline]
            fn is_final(&self) -> bool {
                self.inner.is_final()
            }

            #[inline]
            fn transition(&self, label: Self::Unit) -> Option<Self> {
                let $owner = self;
                self.inner.transition(label).map(|$child| $wrap)
            }

            #[inline]
            fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
                let $owner = self;
                Box::new(
                    self.inner
                        .edges()
                        .map(move |(label, $child)| (label, $wrap)),
                )
            }

            #[inline]
            fn for_each_edge<Visitor>(&self, mut visitor: Visitor)
            where
                Visitor: FnMut(Self::Unit, Self),
            {
                let $owner = self;
                self.inner
                    .for_each_edge(|label, $child| visitor(label, $wrap));
            }

            #[inline]
            fn visit_edges_and_finality<Visitor>(&self, mut visitor: Visitor) -> bool
            where
                Visitor: FnMut(Self::Unit, Self),
            {
                let $owner = self;
                self.inner
                    .visit_edges_and_finality(|label, $child| visitor(label, $wrap))
            }

            #[inline]
            fn filter_map_edges<T, Project, Visitor>(&self, project: Project, mut visitor: Visitor)
            where
                Project: FnMut(Self::Unit) -> Option<T>,
                Visitor: FnMut(Self::Unit, Self, T),
            {
                let $owner = self;
                self.inner
                    .filter_map_edges(project, |label, $child, value| visitor(label, $wrap, value));
            }

            #[inline]
            fn filter_map_edges_and_finality<T, Project, Visitor>(
                &self,
                project: Project,
                mut visitor: Visitor,
            ) -> bool
            where
                Project: FnMut(Self::Unit) -> Option<T>,
                Visitor: FnMut(Self::Unit, Self, T),
            {
                let $owner = self;
                self.inner
                    .filter_map_edges_and_finality(project, |label, $child, value| {
                        visitor(label, $wrap, value)
                    })
            }

            #[inline]
            fn supports_efficient_edge_paging(&self) -> bool {
                self.inner.supports_efficient_edge_paging()
            }

            #[inline]
            fn visit_edge_page_and_finality<Visitor>(
                &self,
                start: usize,
                capacity: usize,
                mut visitor: Visitor,
            ) -> (bool, usize)
            where
                Visitor: FnMut(Self::Unit, Self),
            {
                let $owner = self;
                self.inner.visit_edge_page_and_finality(
                    start,
                    capacity,
                    |label, $child| visitor(label, $wrap),
                )
            }

            #[inline]
            fn visit_edge_page<Visitor>(
                &self,
                start: usize,
                capacity: usize,
                mut visitor: Visitor,
            ) -> usize
            where
                Visitor: FnMut(Self::Unit, Self),
            {
                let $owner = self;
                self.inner.visit_edge_page(
                    start,
                    capacity,
                    |label, $child| visitor(label, $wrap),
                )
            }

            #[inline]
            fn has_edge(&self, label: Self::Unit) -> bool {
                self.inner.has_edge(label)
            }

            #[inline]
            fn edge_count(&self) -> Option<usize> {
                self.inner.edge_count()
            }
        }
    };
}

macro_rules! impl_transparent_dictionary_node {
    ($wrapper:ident, |$owner:ident, $child:ident| $wrap:expr) => {
        $crate::dictionary::node_adapter::impl_dictionary_node_adapter!(
            [N],
            $wrapper<N>,
            [N: libdictenstein::DictionaryNode],
            |$owner, $child| $wrap,
            |node| node.inner.requires_final_units(),
            |node, units| node.inner.accepts_final_units(units)
        );
    };
}

/// Implement all topology/cursor seams while supplying semantic final-key
/// visibility explicitly. This keeps semantic decorators on the same
/// monomorphized traversal substrate as transparent wrappers.
macro_rules! impl_semantic_dictionary_node {
    (
        $wrapper:ident,
        |$owner:ident, $child:ident| $wrap:expr,
        |$requires_owner:ident| $requires_final_units:expr,
        |$accepts_owner:ident, $units:ident| $accepts_final_units:expr
    ) => {
        $crate::dictionary::node_adapter::impl_dictionary_node_adapter!(
            [N],
            $wrapper<N>,
            [N: libdictenstein::DictionaryNode],
            |$owner, $child| $wrap,
            |$requires_owner| $requires_final_units,
            |$accepts_owner, $units| $accepts_final_units
        );
    };
}

/// Generic form of [`impl_semantic_dictionary_node`] for decorators carrying
/// additional statically dispatched policy parameters.
macro_rules! impl_semantic_dictionary_node_generic {
    (
        [$($generic:tt)*],
        $node:ty,
        [$($bounds:tt)*],
        |$owner:ident, $child:ident| $wrap:expr,
        |$requires_owner:ident| $requires_final_units:expr,
        |$accepts_owner:ident, $units:ident| $accepts_final_units:expr
    ) => {
        $crate::dictionary::node_adapter::impl_dictionary_node_adapter!(
            [$($generic)*],
            $node,
            [$($bounds)*],
            |$owner, $child| $wrap,
            |$requires_owner| $requires_final_units,
            |$accepts_owner, $units| $accepts_final_units
        );
    };
}

/// Add mapped-value forwarding to a node already implemented by
/// [`impl_transparent_dictionary_node`]. Kept separate so wrappers that alter
/// value semantics (for example TTL or lazy initialization) cannot opt in by
/// accident.
macro_rules! impl_transparent_mapped_dictionary_node {
    ($wrapper:ident) => {
        impl<N> libdictenstein::MappedDictionaryNode for $wrapper<N>
        where
            N: libdictenstein::MappedDictionaryNode,
        {
            type Value = N::Value;

            #[inline]
            fn value(&self) -> Option<Self::Value> {
                self.inner.value()
            }

            #[inline]
            fn value_at_final(&self) -> Option<Self::Value> {
                self.inner.value_at_final()
            }

            #[inline]
            fn value_at_final_with_units(&self, units: &[Self::Unit]) -> Option<Self::Value> {
                self.inner.value_at_final_with_units(units)
            }

            #[inline]
            fn supports_snapshot_cursor_values(&self) -> bool {
                if $crate::dictionary::node_adapter::transparent_traversal_forwarding_disabled() {
                    return false;
                }
                self.inner.supports_snapshot_cursor_values()
            }

            #[inline]
            fn supports_snapshot_graph_values(&self) -> bool {
                if $crate::dictionary::node_adapter::transparent_traversal_forwarding_disabled() {
                    return false;
                }
                self.inner.supports_snapshot_graph_values()
            }

            #[inline]
            fn snapshot_traversal_graph(
                &self,
            ) -> Option<
                std::sync::Arc<
                    libdictenstein::SnapshotTraversalGraph<
                        Self::Unit,
                        Self::SnapshotGraphValueHandle,
                    >,
                >,
            > {
                if $crate::dictionary::node_adapter::transparent_traversal_forwarding_disabled() {
                    return None;
                }
                self.inner.snapshot_traversal_graph()
            }

            #[inline]
            unsafe fn snapshot_cursor_value(
                &self,
                cursor: Self::SnapshotCursor,
            ) -> Option<Option<Self::Value>> {
                // SAFETY: the transparent wrapper forwards the exact cursor
                // to the exact retained inner revision.
                unsafe { self.inner.snapshot_cursor_value(cursor) }
            }

            #[inline]
            unsafe fn snapshot_cursor_value_with_units(
                &self,
                cursor: Self::SnapshotCursor,
                units: &[Self::Unit],
            ) -> Option<Option<Self::Value>> {
                // SAFETY: the transparent wrapper forwards the exact cursor
                // and root-relative key to the retained inner revision.
                unsafe { self.inner.snapshot_cursor_value_with_units(cursor, units) }
            }

            #[inline]
            unsafe fn snapshot_graph_cursor_value(
                &self,
                graph: &libdictenstein::SnapshotTraversalGraph<
                    Self::Unit,
                    Self::SnapshotGraphValueHandle,
                >,
                cursor: libdictenstein::SnapshotTraversalCursor,
            ) -> Option<Option<Self::Value>> {
                // SAFETY: the graph, cursor, and retained inner owner remain
                // paired exactly as captured by the underlying dictionary.
                unsafe { self.inner.snapshot_graph_cursor_value(graph, cursor) }
            }

            #[inline]
            unsafe fn snapshot_graph_cursor_value_with_units(
                &self,
                graph: &libdictenstein::SnapshotTraversalGraph<
                    Self::Unit,
                    Self::SnapshotGraphValueHandle,
                >,
                cursor: libdictenstein::SnapshotTraversalCursor,
                units: &[Self::Unit],
            ) -> Option<Option<Self::Value>> {
                // SAFETY: the transparent wrapper preserves the graph, cursor,
                // owner, and exact root-relative key as one captured revision.
                unsafe {
                    self.inner
                        .snapshot_graph_cursor_value_with_units(graph, cursor, units)
                }
            }
        }
    };
}

pub(crate) use impl_dictionary_node_adapter;
pub(crate) use impl_semantic_dictionary_node;
pub(crate) use impl_semantic_dictionary_node_generic;
pub(crate) use impl_transparent_dictionary_node;
pub(crate) use impl_transparent_mapped_dictionary_node;

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
    use libdictenstein::{Dictionary, MappedDictionaryNode};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct Calls {
        boxed: AtomicUsize,
        direct: AtomicUsize,
        fused: AtomicUsize,
        paged: AtomicUsize,
    }

    #[derive(Clone)]
    struct ProbeNode {
        final_node: bool,
        calls: Arc<Calls>,
    }

    #[derive(Clone)]
    struct TransparentProbe<N> {
        inner: N,
    }

    impl_transparent_dictionary_node!(TransparentProbe, |_owner, child| TransparentProbe {
        inner: child,
    });
    impl_transparent_mapped_dictionary_node!(TransparentProbe);

    impl DictionaryNode for ProbeNode {
        type Unit = u8;
        type SnapshotCursor = libdictenstein::SnapshotTraversalCursor;
        type SnapshotGraphValueHandle = libdictenstein::SnapshotTraversalCursor;

        fn is_final(&self) -> bool {
            self.final_node
        }

        fn transition(&self, _label: Self::Unit) -> Option<Self> {
            None
        }

        fn edges(&self) -> Box<dyn Iterator<Item = (Self::Unit, Self)> + '_> {
            self.calls.boxed.fetch_add(1, Ordering::Relaxed);
            Box::new(std::iter::empty())
        }

        fn for_each_edge<F>(&self, mut visitor: F)
        where
            F: FnMut(Self::Unit, Self),
        {
            self.calls.direct.fetch_add(1, Ordering::Relaxed);
            visitor(7, self.clone());
        }

        fn visit_edges_and_finality<F>(&self, mut visitor: F) -> bool
        where
            F: FnMut(Self::Unit, Self),
        {
            self.calls.fused.fetch_add(1, Ordering::Relaxed);
            visitor(11, self.clone());
            self.final_node
        }

        fn supports_efficient_edge_paging(&self) -> bool {
            true
        }

        fn visit_edge_page_and_finality<F>(
            &self,
            start: usize,
            capacity: usize,
            mut visitor: F,
        ) -> (bool, usize)
        where
            F: FnMut(Self::Unit, Self),
        {
            self.calls.paged.fetch_add(1, Ordering::Relaxed);
            if start == 0 && capacity > 0 {
                visitor(13, self.clone());
            }
            (self.final_node, 1)
        }
    }

    #[test]
    fn wrapped_visitation_preserves_the_inner_hot_seams() {
        let calls = Arc::new(Calls::default());
        let node = ProbeNode {
            final_node: true,
            calls: Arc::clone(&calls),
        };

        let node = TransparentProbe { inner: node };
        let mut labels = Vec::new();
        node.for_each_edge(|label, _| labels.push(label));
        assert_eq!(labels, [7]);
        assert_eq!(calls.direct.load(Ordering::Relaxed), 1);
        assert_eq!(calls.boxed.load(Ordering::Relaxed), 0);

        labels.clear();
        let final_node = node.visit_edges_and_finality(|label, _| labels.push(label));
        assert!(final_node);
        assert_eq!(labels, [11]);
        assert_eq!(calls.fused.load(Ordering::Relaxed), 1);
        assert_eq!(calls.boxed.load(Ordering::Relaxed), 0);

        labels.clear();
        assert!(node.supports_efficient_edge_paging());
        let (final_node, total) = node.visit_edge_page_and_finality(0, 1, |label, child| {
            labels.push(label);
            assert!(child.inner.final_node);
        });
        assert!(final_node);
        assert_eq!(total, 1);
        assert_eq!(labels, [13]);
        assert_eq!(calls.paged.load(Ordering::Relaxed), 1);
        assert_eq!(calls.boxed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn mapped_root_preserves_graph_and_value_cursor_pairing() {
        let dictionary =
            DynamicDawgChar::from_sorted_terms_with_values([("cat", 7_u64), ("dog", 8)]);
        let root = map_transparent_traversal_root(dictionary.traversal_root(), |inner| {
            TransparentProbe { inner }
        });
        let (graph, owner) = root.into_parts().into_projection_and_root();
        let graph = graph.expect("transparent mapping retains the compact projection");
        assert!(owner.supports_snapshot_graph_values());

        let mut cursor = graph.root_cursor();
        for expected in ['c', 'a', 't'] {
            cursor = graph
                .edges_and_finality(cursor)
                .edges()
                .iter()
                .find(|edge| edge.label() == expected)
                .expect("the captured graph contains cat")
                .target_cursor();
        }

        // SAFETY: graph and cursor were obtained from this exact retained owner.
        let value = unsafe { owner.snapshot_graph_cursor_value(&graph, cursor) };
        assert_eq!(value, Some(Some(7)));
    }
}
