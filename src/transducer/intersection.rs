//! Intersection of dictionary traversal and automaton state.

use super::state::State;
use libdictenstein::{CharUnit, DictionaryNode};

/// Lightweight representation of path history.
///
/// Used to reconstruct the term path without storing full Intersection data.
/// This eliminates Arc overhead from dictionary node cloning in parent chains.
///
/// # Memory Efficiency
///
/// PathNode is ~24 bytes (label + depth + padding + pointer) vs ~50+ bytes for full Intersection.
/// For queries exploring 1000 paths, this saves ~26KB per query.
///
/// # Performance
///
/// Eliminates Arc::clone operations in parent chains, reducing query overhead
/// by an estimated 15-25% for DAWG dictionaries.
///
/// Depth is cached (O(1) access) vs recursive calculation (O(depth)).
///
/// # Type Parameters
///
/// - `U`: Character unit type ([`u8`] for bytes, [`char`] for Unicode characters)
#[derive(Clone)]
pub struct PathNode<U: CharUnit> {
    /// Edge label from parent
    label: U,
    /// Cached depth from root (enables O(1) depth queries and Vec preallocation)
    depth: usize,
    /// Parent in the path
    parent: Option<Box<PathNode<U>>>,
}

impl<U: CharUnit> PathNode<U> {
    /// Create a new path node
    #[inline(always)]
    pub fn new(label: U, parent: Option<Box<PathNode<U>>>) -> Self {
        let depth = parent
            .as_ref()
            .map_or(1, |parent| parent.depth.saturating_add(1));
        Self {
            label,
            depth,
            parent,
        }
    }

    /// Collect labels into vector (for term reconstruction)
    ///
    /// Uses iterative approach to avoid stack overflow on deep paths.
    /// Previous recursive implementation could overflow for paths > ~1000 levels.
    pub fn collect_labels(&self, labels: &mut Vec<U>) {
        // Iteratively walk the parent chain
        let mut current = Some(self);
        while let Some(node) = current {
            labels.push(node.label);
            current = node.parent.as_deref();
        }
    }

    /// Get the depth (number of labels in the path)
    #[inline(always)]
    pub fn depth(&self) -> usize {
        self.depth
    }
}

/// Intersection of dictionary node and Levenshtein automaton state.
///
/// Represents a point in the simultaneous traversal of both the dictionary
/// graph and the Levenshtein automaton. Each intersection tracks:
/// - The current dictionary node
/// - The current automaton state (positions)
/// - The edge label from the parent (for path reconstruction)
/// - A lightweight parent path (for backtracking)
///
/// # Performance Optimization
///
/// Uses PathNode for parent chain instead of full Intersection to eliminate
/// Arc cloning overhead. The parent chain only needs labels for path
/// reconstruction, not the full dictionary node.
pub struct Intersection<N: DictionaryNode> {
    /// Edge label from parent (character unit)
    pub label: Option<N::Unit>,

    /// Current dictionary node
    pub node: N,

    /// Current automaton state
    pub state: State,

    /// Parent path (for path reconstruction) - lightweight, no node cloning
    pub parent: Option<Box<PathNode<N::Unit>>>,
}

impl<N: DictionaryNode> Intersection<N> {
    /// Create a new intersection (root)
    pub fn new(node: N, state: State) -> Self {
        Self {
            label: None,
            node,
            state,
            parent: None,
        }
    }

    /// Create a child intersection with a parent path
    #[inline]
    pub fn with_parent(
        label: N::Unit,
        node: N,
        state: State,
        parent: Option<Box<PathNode<N::Unit>>>,
    ) -> Self {
        Self {
            label: Some(label),
            node,
            state,
            parent,
        }
    }

    /// Reconstruct the term (path) from root to this intersection
    pub fn term(&self) -> String {
        let mut units = Vec::with_capacity(self.depth());

        // Collect current label
        if let Some(label) = self.label {
            units.push(label);
        }

        // Collect parent labels
        if let Some(parent) = &self.parent {
            parent.collect_labels(&mut units);
        }

        units.reverse();
        N::Unit::to_string(&units)
    }

    /// Get the depth (length of path from root)
    pub fn depth(&self) -> usize {
        match &self.parent {
            Some(parent) => 1 + parent.depth(),
            None => {
                if self.label.is_some() {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Check if this intersection represents a complete match
    #[inline(always)]
    pub fn is_final(&self) -> bool {
        self.node.is_final()
    }

    /// Get the minimum distance at this intersection
    #[inline(always)]
    pub fn min_distance(&self) -> Option<usize> {
        self.state.min_distance()
    }
}

// Manual Clone implementation - clones PathNode parent (lightweight)
impl<N: DictionaryNode> Clone for Intersection<N> {
    fn clone(&self) -> Self {
        Self {
            label: self.label,
            node: self.node.clone(),
            state: self.state.clone(),
            // Clone PathNode parent (cheap - no dictionary node cloning)
            parent: self.parent.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::Position;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::Dictionary;

    #[test]
    fn test_intersection_creation() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let root = dict.root();
        let state = State::single(Position::new(0, 0));

        let intersection = Intersection::new(root, state);
        assert_eq!(intersection.depth(), 0);
        assert_eq!(intersection.term(), "");
    }

    #[test]
    fn test_intersection_path_reconstruction() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let root = dict.root();

        // Build path: t -> e -> s using PathNode
        let t_node = root
            .transition(b't')
            .expect("test fixture: 't' exists in dictionary root");
        let _i2 = Intersection::with_parent(
            b't',
            t_node.clone(),
            State::new(),
            None, // Root parent
        );

        let e_node = t_node
            .transition(b'e')
            .expect("test fixture: 'e' exists at t-node");
        let _i3 = Intersection::with_parent(
            b'e',
            e_node.clone(),
            State::new(),
            Some(Box::new(PathNode::new(b't', None))), // t -> root
        );

        let s_node = e_node
            .transition(b's')
            .expect("test fixture: 's' exists at e-node");
        let i4 = Intersection::with_parent(
            b's',
            s_node,
            State::new(),
            Some(Box::new(PathNode::new(
                b'e',
                Some(Box::new(PathNode::new(b't', None))),
            ))), // e -> t -> root
        );

        assert_eq!(i4.term(), "tes");
        assert_eq!(i4.depth(), 3);
    }

    #[test]
    fn test_path_node_depth_exceeds_u16_boundary() {
        let parent = Box::new(PathNode {
            label: b'a',
            depth: usize::from(u16::MAX),
            parent: None,
        });

        let node = PathNode::new(b'b', Some(parent));

        assert_eq!(node.depth(), usize::from(u16::MAX) + 1);
    }

    #[test]
    fn test_path_node_depth_saturates_at_usize_max() {
        let parent = Box::new(PathNode {
            label: b'a',
            depth: usize::MAX,
            parent: None,
        });

        let node = PathNode::new(b'b', Some(parent));

        assert_eq!(node.depth(), usize::MAX);
    }
}
