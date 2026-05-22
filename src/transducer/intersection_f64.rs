//! Intersection of dictionary traversal and float-weighted automaton state.
//!
//! This module provides `IntersectionF64`, which tracks the simultaneous
//! traversal of a dictionary graph and a float-weighted Levenshtein automaton.

use super::intersection::PathNode;
use super::state_f64::StateF64;
use libdictenstein::{CharUnit, DictionaryNode};

/// Intersection of dictionary node and float-weighted automaton state.
///
/// Represents a point in the simultaneous traversal of both the dictionary
/// graph and the float-weighted Levenshtein automaton. Each intersection tracks:
/// - The current dictionary node
/// - The current automaton state (float-weighted positions)
/// - The edge label from the parent (for path reconstruction)
/// - A lightweight parent path (for backtracking)
///
/// # Performance Optimization
///
/// Uses PathNode for parent chain (same as integer version) to eliminate
/// Arc cloning overhead. The parent chain only needs labels for path
/// reconstruction, not the full dictionary node.
pub struct IntersectionF64<N: DictionaryNode> {
    /// Edge label from parent (character unit)
    pub label: Option<N::Unit>,

    /// Current dictionary node
    pub node: N,

    /// Current automaton state (float-weighted)
    pub state: StateF64,

    /// Parent path (for path reconstruction) - lightweight, no node cloning
    pub parent: Option<Box<PathNode<N::Unit>>>,
}

impl<N: DictionaryNode> IntersectionF64<N> {
    /// Create a new intersection (root)
    pub fn new(node: N, state: StateF64) -> Self {
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
        state: StateF64,
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
        let mut units = Vec::new();

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

    /// Get the minimum distance at this intersection (float)
    #[inline(always)]
    pub fn min_distance(&self) -> Option<f64> {
        self.state.min_distance()
    }
}

// Manual Clone implementation - clones PathNode parent (lightweight)
impl<N: DictionaryNode> Clone for IntersectionF64<N> {
    fn clone(&self) -> Self {
        Self {
            label: self.label,
            node: self.node.clone(),
            state: self.state.clone(),
            parent: self.parent.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::PositionF64;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::Dictionary;

    #[test]
    fn test_intersection_creation() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let root = dict.root();
        let state = StateF64::single(PositionF64::new(0, 0.0));

        let intersection = IntersectionF64::new(root, state);
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
        let e_node = t_node
            .transition(b'e')
            .expect("test fixture: 'e' exists at t-node");
        let s_node = e_node
            .transition(b's')
            .expect("test fixture: 's' exists at e-node");

        let i4 = IntersectionF64::with_parent(
            b's',
            s_node,
            StateF64::new(),
            Some(Box::new(PathNode::new(
                b'e',
                Some(Box::new(PathNode::new(b't', None))),
            ))),
        );

        assert_eq!(i4.term(), "tes");
        assert_eq!(i4.depth(), 3);
    }
}
