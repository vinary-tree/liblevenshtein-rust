//! Monomorphized helpers for transparent dictionary-node decorators.

use libdictenstein::DictionaryNode;

/// Visit an inner node's edges while wrapping each child without allocating
/// an intermediate collection or boxed iterator.
#[inline]
pub(crate) fn for_each_wrapped_edge<N, W, Wrap, Visitor>(
    inner: &N,
    mut wrap: Wrap,
    mut visitor: Visitor,
) where
    N: DictionaryNode,
    Wrap: FnMut(N) -> W,
    Visitor: FnMut(N::Unit, W),
{
    inner.for_each_edge(|label, child| visitor(label, wrap(child)));
}

/// Read finality and visit wrapped children through the inner node's fused
/// seam. Boundary-aware inner nodes therefore retain their single-operation
/// implementation through arbitrarily stacked decorators.
#[inline]
pub(crate) fn visit_wrapped_edges_and_finality<N, W, Wrap, Visitor>(
    inner: &N,
    mut wrap: Wrap,
    mut visitor: Visitor,
) -> bool
where
    N: DictionaryNode,
    Wrap: FnMut(N) -> W,
    Visitor: FnMut(N::Unit, W),
{
    inner.visit_edges_and_finality(|label, child| visitor(label, wrap(child)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Default)]
    struct Calls {
        boxed: AtomicUsize,
        direct: AtomicUsize,
        fused: AtomicUsize,
    }

    #[derive(Clone)]
    struct ProbeNode {
        final_node: bool,
        calls: Arc<Calls>,
    }

    impl DictionaryNode for ProbeNode {
        type Unit = u8;

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
    }

    #[test]
    fn wrapped_visitation_preserves_the_inner_hot_seams() {
        let calls = Arc::new(Calls::default());
        let node = ProbeNode {
            final_node: true,
            calls: Arc::clone(&calls),
        };

        let mut labels = Vec::new();
        for_each_wrapped_edge(&node, |child| child, |label, _| labels.push(label));
        assert_eq!(labels, [7]);
        assert_eq!(calls.direct.load(Ordering::Relaxed), 1);
        assert_eq!(calls.boxed.load(Ordering::Relaxed), 0);

        labels.clear();
        let final_node =
            visit_wrapped_edges_and_finality(&node, |child| child, |label, _| labels.push(label));
        assert!(final_node);
        assert_eq!(labels, [11]);
        assert_eq!(calls.fused.load(Ordering::Relaxed), 1);
        assert_eq!(calls.boxed.load(Ordering::Relaxed), 0);
    }
}
