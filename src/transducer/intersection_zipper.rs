//! Zipper for the intersection of dictionary and automaton traversal.
//!
//! This module provides a zipper that composes dictionary navigation with
//! automaton state tracking, enabling efficient fuzzy string matching.

use crate::transducer::{AutomatonZipper, StatePool};
use libdictenstein::zipper::DictZipper;
use std::sync::Arc;

#[derive(Clone)]
struct ZipperPathNode {
    label: u8,
    depth: usize,
    parent: Option<Arc<ZipperPathNode>>,
}

impl ZipperPathNode {
    #[inline]
    fn new(label: u8, parent: Option<Arc<ZipperPathNode>>) -> Self {
        let depth = parent
            .as_ref()
            .map_or(1, |parent| parent.depth.saturating_add(1));
        Self {
            label,
            depth,
            parent,
        }
    }

    fn collect_labels(&self, labels: &mut Vec<u8>) {
        let mut current = Some(self);
        while let Some(node) = current {
            labels.push(node.label);
            current = node.parent.as_deref();
        }
    }

    #[inline]
    fn depth(&self) -> usize {
        self.depth
    }
}

/// Zipper for traversing the intersection of dictionary and Levenshtein automaton.
///
/// An `IntersectionZipper` combines a dictionary zipper (tracking position in the
/// dictionary graph) with an automaton zipper (tracking Levenshtein state). This
/// composition enables efficient fuzzy string matching by simultaneously navigating
/// both structures.
///
/// # Type Parameters
///
/// * `D` - Dictionary zipper type implementing `DictZipper`
///
/// # Design
///
/// The intersection zipper maintains:
/// - Dictionary position via `DictZipper`
/// - Automaton state via `AutomatonZipper`
/// - Shared parent path chain (for term reconstruction)
///
/// Navigation (via `children()`) produces child zippers only when both:
/// 1. The dictionary has an edge with a given label
/// 2. The automaton can transition on that label (produces viable state)
///
/// # Memory Efficiency
///
/// - Uses lightweight shared path nodes instead of cloning full parent chains
/// - Dictionary zipper is generic (can be Copy for index-based or Clone for Arc-based)
/// - AutomatonZipper uses Arc for query sharing
///
/// # Examples
///
/// ```rust
/// # // Note: This example requires the 'pathmap-backend' feature
/// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
/// use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
/// use liblevenshtein::transducer::{AutomatonZipper, Algorithm, StatePool};
/// use liblevenshtein::transducer::intersection_zipper::IntersectionZipper;
///
/// let dict = PathMapDictionary::<()>::new();
/// dict.insert("cat");
/// dict.insert("dog");
///
/// // Create dictionary zipper
/// let dict_zipper = PathMapZipper::new_from_dict(&dict);
///
/// // Create automaton zipper for query "cat" with max distance 1
/// let auto_zipper = AutomatonZipper::new("cat".as_bytes(), 1, Algorithm::Standard);
///
/// // Create intersection zipper
/// let intersection = IntersectionZipper::new(dict_zipper, auto_zipper);
///
/// // Navigate through children
/// let mut pool = StatePool::new();
/// for (label, child) in intersection.children(&mut pool) {
///     println!("Edge: {}", label as char);
///     if child.is_match() {
///         println!("  Match found: {}", child.term());
///     }
/// }
/// ```
#[derive(Clone)]
pub struct IntersectionZipper<D>
where
    D: DictZipper<Unit = u8>,
{
    /// Dictionary zipper at current position
    dict: D,

    /// Automaton zipper at current state
    automaton: AutomatonZipper,

    /// Parent path for term reconstruction (shared by sibling zippers)
    parent: Option<Arc<ZipperPathNode>>,
}

impl<D> IntersectionZipper<D>
where
    D: DictZipper<Unit = u8>,
{
    /// Create a new intersection zipper at the root.
    ///
    /// # Arguments
    ///
    /// * `dict` - Dictionary zipper at root position
    /// * `automaton` - Automaton zipper at initial state
    ///
    /// # Examples
    ///
    /// ```rust
    /// # // Note: This example requires the 'pathmap-backend' feature
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    /// use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm};
    /// use liblevenshtein::transducer::intersection_zipper::IntersectionZipper;
    ///
    /// let dict = PathMapDictionary::<()>::new();
    /// let dict_zipper = PathMapZipper::new_from_dict(&dict);
    /// let auto_zipper = AutomatonZipper::new("test".as_bytes(), 1, Algorithm::Standard);
    ///
    /// let intersection = IntersectionZipper::new(dict_zipper, auto_zipper);
    /// ```
    pub fn new(dict: D, automaton: AutomatonZipper) -> Self {
        IntersectionZipper {
            dict,
            automaton,
            parent: None,
        }
    }

    /// Create an intersection zipper with a parent path.
    ///
    /// Used internally when creating child zippers.
    fn with_parent(
        dict: D,
        automaton: AutomatonZipper,
        parent: Option<Arc<ZipperPathNode>>,
    ) -> Self {
        IntersectionZipper {
            dict,
            automaton,
            parent,
        }
    }

    /// Check if the current position represents a match.
    ///
    /// A match occurs when:
    /// 1. The dictionary position is final (marks end of a term)
    /// 2. The automaton state can accept with distance ≤ max_distance
    ///
    /// # Returns
    ///
    /// `true` if this is a valid match, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # // Note: This example requires the 'pathmap-backend' feature
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    /// use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm, StatePool};
    /// use liblevenshtein::transducer::intersection_zipper::IntersectionZipper;
    ///
    /// let dict = PathMapDictionary::<()>::new();
    /// dict.insert("cat");
    ///
    /// let dict_zipper = PathMapZipper::new_from_dict(&dict);
    /// let auto_zipper = AutomatonZipper::new("cat".as_bytes(), 1, Algorithm::Standard);
    /// let mut intersection = IntersectionZipper::new(dict_zipper, auto_zipper);
    ///
    /// // Not a match at root
    /// assert!(!intersection.is_match());
    ///
    /// // Navigate through "cat"
    /// let mut pool = StatePool::new();
    /// let children: Vec<_> = intersection.children(&mut pool).collect();
    /// for (label, child) in children {
    ///     if label == b'c' {
    ///         intersection = child;
    ///         break;
    ///     }
    /// }
    /// // Still not a match after just 'c'
    /// assert!(!intersection.is_match());
    /// ```
    #[inline]
    pub fn is_match(&self) -> bool {
        if !self.dict.is_final() {
            return false;
        }

        // For final dictionary nodes, check if automaton accepts
        let term_length = self.depth();
        self.automaton
            .infer_distance(term_length)
            .map(|dist| dist <= self.automaton.max_distance())
            .unwrap_or(false)
    }

    /// Get the edit distance if this is a match.
    ///
    /// # Returns
    ///
    /// `Some(distance)` if this is a match within max_distance, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use libdictenstein::pathmap::PathMapDictionary;
    /// use libdictenstein::pathmap::zipper::PathMapZipper;
    /// use liblevenshtein::transducer::{
    ///     Algorithm, AutomatonZipper, IntersectionZipper, StatePool,
    /// };
    ///
    /// let dictionary = PathMapDictionary::<()>::new();
    /// dictionary.insert("cat");
    /// let dictionary = PathMapZipper::new_from_dict(&dictionary);
    /// let automaton = AutomatonZipper::new(b"bat", 1, Algorithm::Standard);
    /// let mut intersection = IntersectionZipper::new(dictionary, automaton);
    /// let mut pool = StatePool::new();
    ///
    /// for &expected in b"cat" {
    ///     let children: Vec<_> = intersection.children(&mut pool).collect();
    ///     intersection = children
    ///         .into_iter()
    ///         .find(|(label, _)| *label == expected)
    ///         .expect("the dictionary contains the requested path")
    ///         .1;
    /// }
    /// assert_eq!(intersection.distance(), Some(1));
    /// ```
    #[inline]
    pub fn distance(&self) -> Option<usize> {
        if !self.dict.is_final() {
            return None;
        }

        let term_length = self.depth();
        let dist = self.automaton.infer_distance(term_length)?;

        if dist <= self.automaton.max_distance() {
            Some(dist)
        } else {
            None
        }
    }

    /// Get the depth (term length) from root to current position.
    ///
    /// # Returns
    ///
    /// The number of edges traversed from root.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use libdictenstein::pathmap::PathMapDictionary;
    /// use libdictenstein::pathmap::zipper::PathMapZipper;
    /// use liblevenshtein::transducer::{
    ///     Algorithm, AutomatonZipper, IntersectionZipper, StatePool,
    /// };
    ///
    /// let dictionary = PathMapDictionary::<()>::new();
    /// dictionary.insert("cat");
    /// let dictionary = PathMapZipper::new_from_dict(&dictionary);
    /// let automaton = AutomatonZipper::new(b"cat", 0, Algorithm::Standard);
    /// let mut intersection = IntersectionZipper::new(dictionary, automaton);
    /// assert_eq!(intersection.depth(), 0);
    ///
    /// let mut pool = StatePool::new();
    /// for &expected in b"cat" {
    ///     let children: Vec<_> = intersection.children(&mut pool).collect();
    ///     intersection = children
    ///         .into_iter()
    ///         .find(|(label, _)| *label == expected)
    ///         .expect("the dictionary contains the requested path")
    ///         .1;
    /// }
    /// assert_eq!(intersection.depth(), 3);
    /// ```
    #[inline]
    pub fn depth(&self) -> usize {
        self.parent.as_ref().map(|p| p.depth()).unwrap_or(0)
    }

    /// Reconstruct the term (path) from root to current position.
    ///
    /// # Returns
    ///
    /// The complete term as a String.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use libdictenstein::pathmap::PathMapDictionary;
    /// use libdictenstein::pathmap::zipper::PathMapZipper;
    /// use liblevenshtein::transducer::{
    ///     Algorithm, AutomatonZipper, IntersectionZipper, StatePool,
    /// };
    ///
    /// let dictionary = PathMapDictionary::<()>::new();
    /// dictionary.insert("cat");
    /// let dictionary = PathMapZipper::new_from_dict(&dictionary);
    /// let automaton = AutomatonZipper::new(b"cat", 0, Algorithm::Standard);
    /// let mut intersection = IntersectionZipper::new(dictionary, automaton);
    /// let mut pool = StatePool::new();
    ///
    /// for &expected in b"cat" {
    ///     let children: Vec<_> = intersection.children(&mut pool).collect();
    ///     intersection = children
    ///         .into_iter()
    ///         .find(|(label, _)| *label == expected)
    ///         .expect("the dictionary contains the requested path")
    ///         .1;
    /// }
    /// assert_eq!(intersection.term(), "cat");
    /// ```
    pub fn term(&self) -> String {
        let mut units = Vec::with_capacity(self.depth());

        // Collect labels from parent chain
        if let Some(parent) = &self.parent {
            parent.collect_labels(&mut units);
        }

        units.reverse();
        String::from_utf8_lossy(&units).into_owned()
    }

    /// Navigate to child intersections.
    ///
    /// Produces an iterator over `(label, child_zipper)` pairs for all edges where:
    /// 1. The dictionary has an edge with that label
    /// 2. The automaton can transition on that label
    ///
    /// Uses the provided `StatePool` for efficient state allocation.
    ///
    /// # Arguments
    ///
    /// * `pool` - StatePool for automaton state reuse
    ///
    /// # Returns
    ///
    /// Iterator over (label, child_intersection) pairs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # // Note: This example requires the 'pathmap-backend' feature
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    /// use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
    /// use liblevenshtein::transducer::{AutomatonZipper, Algorithm, StatePool};
    /// use liblevenshtein::transducer::intersection_zipper::IntersectionZipper;
    ///
    /// let dict = PathMapDictionary::<()>::new();
    /// dict.insert("cat");
    /// dict.insert("car");
    ///
    /// let dict_zipper = PathMapZipper::new_from_dict(&dict);
    /// let auto_zipper = AutomatonZipper::new("cat".as_bytes(), 1, Algorithm::Standard);
    /// let intersection = IntersectionZipper::new(dict_zipper, auto_zipper);
    ///
    /// let mut pool = StatePool::new();
    /// let children: Vec<_> = intersection.children(&mut pool).collect();
    ///
    /// // Should have one child: 'c'
    /// assert_eq!(children.len(), 1);
    /// assert_eq!(children[0].0, b'c');
    /// ```
    pub fn children<'a>(
        &'a self,
        pool: &'a mut StatePool,
    ) -> impl Iterator<Item = (u8, Self)> + 'a {
        let parent_for_children = self.parent.clone();
        let automaton = &self.automaton;

        self.dict.children().filter_map(move |(label, dict_child)| {
            // Try to transition automaton with this label
            automaton.transition(label, pool).map(|auto_child| {
                // Create parent node for child
                let new_parent = Some(Arc::new(ZipperPathNode::new(
                    label,
                    parent_for_children.clone(),
                )));

                let child = IntersectionZipper::with_parent(dict_child, auto_child, new_parent);

                (label, child)
            })
        })
    }

    /// Check if the automaton state is viable (has positions).
    ///
    /// A non-viable state means no further matches are possible from this position.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use libdictenstein::pathmap::PathMapDictionary;
    /// use libdictenstein::pathmap::zipper::PathMapZipper;
    /// use liblevenshtein::transducer::{
    ///     Algorithm, AutomatonZipper, IntersectionZipper, StatePool,
    /// };
    ///
    /// let dictionary = PathMapDictionary::<()>::new();
    /// dictionary.insert("cat");
    /// let dictionary = PathMapZipper::new_from_dict(&dictionary);
    /// let automaton = AutomatonZipper::new(b"cat", 0, Algorithm::Standard);
    /// let intersection = IntersectionZipper::new(dictionary, automaton);
    /// assert!(intersection.is_viable());
    ///
    /// // `children` prunes failed automaton transitions, so every yielded
    /// // dictionary/automaton intersection remains viable.
    /// let mut pool = StatePool::new();
    /// let children: Vec<_> = intersection.children(&mut pool).collect();
    /// assert_eq!(children.len(), 1);
    /// assert!(children.iter().all(|(_, child)| child.is_viable()));
    /// ```
    pub fn is_viable(&self) -> bool {
        self.automaton.is_viable()
    }

    /// Get a reference to the dictionary zipper.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use libdictenstein::pathmap::PathMapDictionary;
    /// use libdictenstein::pathmap::zipper::PathMapZipper;
    /// use libdictenstein::DictZipper;
    /// use liblevenshtein::transducer::{
    ///     Algorithm, AutomatonZipper, IntersectionZipper, StatePool,
    /// };
    ///
    /// let dictionary = PathMapDictionary::<()>::new();
    /// dictionary.insert("cat");
    /// let dictionary = PathMapZipper::new_from_dict(&dictionary);
    /// let automaton = AutomatonZipper::new(b"cat", 0, Algorithm::Standard);
    /// let mut intersection = IntersectionZipper::new(dictionary, automaton);
    /// assert!(!intersection.dict_zipper().is_final());
    ///
    /// let mut pool = StatePool::new();
    /// for &expected in b"cat" {
    ///     let children: Vec<_> = intersection.children(&mut pool).collect();
    ///     intersection = children
    ///         .into_iter()
    ///         .find(|(label, _)| *label == expected)
    ///         .expect("the dictionary contains the requested path")
    ///         .1;
    /// }
    /// assert!(intersection.dict_zipper().is_final());
    /// ```
    pub fn dict_zipper(&self) -> &D {
        &self.dict
    }

    /// Get a reference to the automaton zipper.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use libdictenstein::pathmap::PathMapDictionary;
    /// use libdictenstein::pathmap::zipper::PathMapZipper;
    /// use liblevenshtein::transducer::{
    ///     Algorithm, AutomatonZipper, IntersectionZipper,
    /// };
    ///
    /// let dictionary = PathMapDictionary::<()>::new();
    /// let dictionary = PathMapZipper::new_from_dict(&dictionary);
    /// let automaton = AutomatonZipper::new(b"needle", 2, Algorithm::Standard);
    /// let intersection = IntersectionZipper::new(dictionary, automaton);
    ///
    /// assert_eq!(intersection.automaton_zipper().query(), b"needle");
    /// assert_eq!(intersection.automaton_zipper().max_distance(), 2);
    /// ```
    pub fn automaton_zipper(&self) -> &AutomatonZipper {
        &self.automaton
    }
}

#[cfg(test)]
mod path_node_tests {
    use super::*;

    #[test]
    fn test_zipper_path_node_depth_exceeds_u16_boundary() {
        let parent = Arc::new(ZipperPathNode {
            label: b'a',
            depth: usize::from(u16::MAX),
            parent: None,
        });

        let node = ZipperPathNode::new(b'b', Some(parent));

        assert_eq!(node.depth(), usize::from(u16::MAX) + 1);
    }

    #[test]
    fn test_zipper_path_node_depth_saturates_at_usize_max() {
        let parent = Arc::new(ZipperPathNode {
            label: b'a',
            depth: usize::MAX,
            parent: None,
        });

        let node = ZipperPathNode::new(b'b', Some(parent));

        assert_eq!(node.depth(), usize::MAX);
    }
}

#[cfg(all(test, feature = "pathmap-backend"))]
mod tests {
    use super::*;
    use crate::distance::standard_distance;
    use crate::transducer::Algorithm;
    use libdictenstein::pathmap::zipper::PathMapZipper;
    use libdictenstein::pathmap::PathMapDictionary;

    #[test]
    fn test_new_creates_root() {
        let dict = PathMapDictionary::<()>::new();
        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let auto_zipper = AutomatonZipper::new(b"test", 1, Algorithm::Standard);

        let intersection = IntersectionZipper::new(dict_zipper, auto_zipper);

        assert_eq!(intersection.depth(), 0);
        assert!(!intersection.is_match());
        assert!(intersection.is_viable());
    }

    #[test]
    fn test_exact_match() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let auto_zipper = AutomatonZipper::new(b"cat", 0, Algorithm::Standard);
        let mut intersection = IntersectionZipper::new(dict_zipper, auto_zipper);

        let mut pool = StatePool::new();

        // Navigate through "cat"
        for expected in b"cat" {
            let children: Vec<_> = intersection.children(&mut pool).collect();
            let mut found = false;
            for (label, child) in children {
                if &label == expected {
                    intersection = child;
                    found = true;
                    break;
                }
            }
            assert!(found, "Should find edge for '{}'", *expected as char);
        }

        // Should be a match with distance 0
        assert!(intersection.is_match());
        assert_eq!(intersection.distance(), Some(0));
        assert_eq!(intersection.term(), "cat");
        assert_eq!(intersection.depth(), 3);
    }

    #[test]
    fn test_fuzzy_match() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        // Query "bat" with max distance 1 (one substitution)
        let auto_zipper = AutomatonZipper::new(b"bat", 1, Algorithm::Standard);
        let mut intersection = IntersectionZipper::new(dict_zipper, auto_zipper);

        let mut pool = StatePool::new();

        // Navigate through "cat"
        for expected in b"cat" {
            let children: Vec<_> = intersection.children(&mut pool).collect();
            let mut found = false;
            for (label, child) in children {
                if &label == expected {
                    intersection = child;
                    found = true;
                    break;
                }
            }
            assert!(found, "Should find edge for '{}'", *expected as char);
        }

        // Should be a match with distance 1 (b->c substitution)
        assert!(intersection.is_match());
        assert_eq!(intersection.distance(), Some(1));
        assert_eq!(intersection.term(), "cat");
    }

    #[test]
    fn test_no_match_exceeds_distance() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        // Query "dog" with max distance 1 (would need 3 substitutions)
        let auto_zipper = AutomatonZipper::new(b"dog", 1, Algorithm::Standard);
        let mut intersection = IntersectionZipper::new(dict_zipper, auto_zipper);

        let mut pool = StatePool::new();

        // Navigate through "cat"
        for expected in b"cat" {
            let children: Vec<_> = intersection.children(&mut pool).collect();
            let mut found = false;
            for (label, child) in children {
                if &label == expected {
                    intersection = child;
                    found = true;
                    break;
                }
            }
            if !found {
                // May not be able to navigate due to distance constraints
                return;
            }
        }

        // If we got here, should not be a match (distance too high)
        assert!(!intersection.is_match());
        assert_eq!(intersection.distance(), None);
    }

    #[test]
    fn test_children_iteration() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");
        dict.insert("car");
        dict.insert("dog");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let auto_zipper = AutomatonZipper::new(b"cat", 1, Algorithm::Standard);
        let intersection = IntersectionZipper::new(dict_zipper, auto_zipper);

        let mut pool = StatePool::new();
        let children: Vec<_> = intersection.children(&mut pool).collect();

        // Should have 2 children: 'c' and 'd'
        assert_eq!(children.len(), 2);

        let labels: Vec<u8> = children.iter().map(|(label, _)| *label).collect();
        assert!(labels.contains(&b'c'));
        assert!(labels.contains(&b'd'));
    }

    #[test]
    fn test_multiple_matches() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");
        dict.insert("ca");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let auto_zipper = AutomatonZipper::new(b"cat", 1, Algorithm::Standard);
        let mut intersection = IntersectionZipper::new(dict_zipper, auto_zipper);

        let mut pool = StatePool::new();

        // Navigate to 'c'
        let children: Vec<_> = intersection.children(&mut pool).collect();
        for (label, child) in children {
            if label == b'c' {
                intersection = child;
                break;
            }
        }

        // Navigate to 'a'
        let children: Vec<_> = intersection.children(&mut pool).collect();
        for (label, child) in children {
            if label == b'a' {
                intersection = child;
                break;
            }
        }

        // At "ca", the unmatched final 't' in query "cat" costs one deletion.
        // Pin the zipper result to the independent full-string oracle.
        assert!(intersection.is_match());
        assert_eq!(
            intersection.distance(),
            Some(standard_distance("cat", "ca"))
        );
        assert_eq!(intersection.term(), "ca");

        // Continue to 't'
        let children: Vec<_> = intersection.children(&mut pool).collect();
        for (label, child) in children {
            if label == b't' {
                intersection = child;
                break;
            }
        }

        // At "cat" - should be a match with distance 0
        assert!(intersection.is_match());
        assert_eq!(intersection.distance(), Some(0));
        assert_eq!(intersection.term(), "cat");
    }

    #[test]
    fn test_term_reconstruction() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("hello");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let auto_zipper = AutomatonZipper::new(b"hello", 0, Algorithm::Standard);
        let mut intersection = IntersectionZipper::new(dict_zipper, auto_zipper);

        let mut pool = StatePool::new();

        // Navigate through "hello"
        for expected in b"hello" {
            let children: Vec<_> = intersection.children(&mut pool).collect();
            for (label, child) in children {
                if label == *expected {
                    intersection = child;
                    break;
                }
            }
        }

        assert_eq!(intersection.term(), "hello");
        assert_eq!(intersection.depth(), 5);
    }

    #[test]
    fn test_shared_prefix_branching_reconstructs_terms() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cart");
        dict.insert("care");
        dict.insert("cars");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let auto_zipper = AutomatonZipper::new(b"car", 1, Algorithm::Standard);
        let mut frontier = vec![IntersectionZipper::new(dict_zipper, auto_zipper)];
        let mut pool = StatePool::new();
        let mut matches = Vec::new();

        while let Some(intersection) = frontier.pop() {
            if intersection.is_match() {
                matches.push(intersection.term());
            }

            frontier.extend(
                intersection
                    .children(&mut pool)
                    .map(|(_, child)| child)
                    .collect::<Vec<_>>(),
            );
        }

        matches.sort();
        assert_eq!(matches, vec!["care", "cars", "cart"]);
    }

    #[test]
    fn test_empty_dictionary() {
        let dict = PathMapDictionary::<()>::new();

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let auto_zipper = AutomatonZipper::new(b"test", 1, Algorithm::Standard);
        let intersection = IntersectionZipper::new(dict_zipper, auto_zipper);

        let mut pool = StatePool::new();
        let children: Vec<_> = intersection.children(&mut pool).collect();

        assert_eq!(children.len(), 0);
        assert!(!intersection.is_match());
    }

    #[test]
    fn test_clone_independence() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let auto_zipper = AutomatonZipper::new(b"cat", 1, Algorithm::Standard);
        let intersection = IntersectionZipper::new(dict_zipper, auto_zipper);

        let clone1 = intersection.clone();
        let clone2 = intersection.clone();

        let mut pool = StatePool::new();

        // Navigate clone1
        let mut _z1 = clone1;
        let children: Vec<_> = _z1.children(&mut pool).collect();
        for (label, child) in children {
            if label == b'c' {
                _z1 = child;
                break;
            }
        }

        // clone2 should still be at root
        assert_eq!(clone2.depth(), 0);
    }
}
