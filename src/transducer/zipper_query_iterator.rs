//! Query iterator using zipper-based traversal.
//!
//! This module provides an iterator that uses IntersectionZipper for BFS
//! traversal of fuzzy matches. It serves as a proof-of-concept for the
//! zipper architecture and enables performance comparison with the existing
//! node-based QueryIterator.

use crate::transducer::transition::CachedUnitTransitions;
use crate::transducer::{Algorithm, AutomatonZipper, Candidate, IntersectionZipper, StatePool};
use libdictenstein::zipper::DictZipper;
use std::collections::VecDeque;

/// Query iterator using zipper-based BFS traversal.
///
/// This iterator composes dictionary and automaton zippers to traverse
/// fuzzy matches in breadth-first order. It maintains a queue of
/// IntersectionZipper states and explores them level-by-level.
///
/// # Type Parameters
///
/// - `D`: Dictionary zipper type (must have `Unit = u8`)
///
/// # Performance
///
/// - **Memory**: O(branching_factor × max_distance) for the BFS queue
/// - **Time**: O(dictionary_size) worst case, pruned by max_distance
///
/// # Examples
///
/// ```ignore
/// # // Note: This example requires the 'pathmap-backend' feature
/// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
/// use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
/// use liblevenshtein::transducer::{Algorithm, ZipperQueryIterator};
///
/// let dict = PathMapDictionary::<()>::new();
/// dict.insert("cat");
/// dict.insert("dog");
/// dict.insert("car");
///
/// let dict_zipper = PathMapZipper::new_from_dict(&dict);
///
/// let iter = ZipperQueryIterator::new(
///     dict_zipper,
///     "cat",
///     1,
///     Algorithm::Standard
/// );
///
/// let results: Vec<_> = iter.collect();
/// assert!(results.iter().any(|c| c.term == "cat"));
/// assert!(results.iter().any(|c| c.term == "car"));
/// ```
pub struct ZipperQueryIterator<D>
where
    D: DictZipper<Unit = u8>,
{
    /// Queue of intersections to explore (BFS)
    queue: VecDeque<IntersectionZipper<D>>,

    /// State pool for efficient allocation reuse
    pool: StatePool,

    /// Query-local characteristic vectors shared by every zipper branch.
    unit_transitions: CachedUnitTransitions<u8>,
}

impl<D> ZipperQueryIterator<D>
where
    D: DictZipper<Unit = u8>,
{
    /// Create a new zipper-based query iterator.
    ///
    /// # Arguments
    ///
    /// * `dict_zipper` - Dictionary zipper at root position
    /// * `query` - Query string to match against
    /// * `max_distance` - Maximum edit distance threshold
    /// * `algorithm` - Levenshtein algorithm variant
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # // Note: This example requires the 'pathmap-backend' feature
    /// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    /// use liblevenshtein::dictionary::pathmap_zipper::PathMapZipper;
    /// use liblevenshtein::transducer::{Algorithm, ZipperQueryIterator};
    ///
    /// let dict = PathMapDictionary::<()>::new();
    /// dict.insert("test");
    ///
    /// let dict_zipper = PathMapZipper::new_from_dict(&dict);
    /// let iter = ZipperQueryIterator::new(
    ///     dict_zipper,
    ///     "test",
    ///     1,
    ///     Algorithm::Standard
    /// );
    ///
    /// let results: Vec<_> = iter.collect();
    /// assert_eq!(results.len(), 1);
    /// assert_eq!(results[0].term, "test");
    /// ```
    pub fn new(dict_zipper: D, query: &str, max_distance: usize, algorithm: Algorithm) -> Self {
        let automaton = AutomatonZipper::new(query.as_bytes(), max_distance, algorithm);
        let intersection = IntersectionZipper::new(dict_zipper, automaton);

        let mut queue = VecDeque::with_capacity(64);
        queue.push_back(intersection);

        ZipperQueryIterator {
            queue,
            pool: StatePool::new(),
            unit_transitions: CachedUnitTransitions::new(query.len(), max_distance),
        }
    }
}

impl<D> Iterator for ZipperQueryIterator<D>
where
    D: DictZipper<Unit = u8>,
{
    type Item = Candidate;

    fn next(&mut self) -> Option<Self::Item> {
        // BFS loop: explore states until we find a match
        while let Some(intersection) = self.queue.pop_front() {
            // `distance` already performs the dictionary finality check. Calling
            // `is_match` first would repeat that observation for every match.
            if let Some(distance) = intersection.distance() {
                let candidate = Candidate {
                    term: intersection.term(),
                    distance,
                };

                self.queue_viable_children(&intersection);

                return Some(candidate);
            }

            // Not a match - add viable children to queue
            self.queue_viable_children(&intersection);
        }

        None
    }
}

impl<D> ZipperQueryIterator<D>
where
    D: DictZipper<Unit = u8>,
{
    fn queue_viable_children(&mut self, intersection: &IntersectionZipper<D>) {
        for (_label, child) in
            intersection.children_cached(&mut self.pool, &mut self.unit_transitions)
        {
            if child.is_viable() {
                self.queue.push_back(child);
            }
        }
    }
}

#[cfg(all(test, feature = "pathmap-backend"))]
mod tests {
    use super::*;
    use crate::distance::standard_distance;
    use libdictenstein::pathmap::zipper::PathMapZipper;
    use libdictenstein::pathmap::PathMapDictionary;

    #[test]
    fn test_exact_match() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "cat", 0, Algorithm::Standard);

        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "cat");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_fuzzy_match() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");
        dict.insert("car");
        dict.insert("dog");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "cat", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Should find "cat" (distance 0) and "car" (distance 1)
        assert!(results.iter().any(|c| c.term == "cat" && c.distance == 0));
        assert!(results.iter().any(|c| c.term == "car" && c.distance == 1));

        // Should NOT find "dog" (distance 3)
        assert!(!results.iter().any(|c| c.term == "dog"));
    }

    #[test]
    fn test_multiple_distances() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");
        dict.insert("at");
        dict.insert("ca");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "cat", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Should find all three terms
        assert_eq!(results.len(), 3);

        // "cat" should have distance 0
        assert!(results.iter().any(|c| c.term == "cat" && c.distance == 0));

        // Both shorter terms require one full-string deletion. Compare to the
        // independent oracle so prefix-only scoring cannot return unnoticed.
        for term in ["at", "ca"] {
            let expected = standard_distance("cat", term);
            assert!(
                results
                    .iter()
                    .any(|candidate| candidate.term == term && candidate.distance == expected),
                "missing oracle distance for {term}"
            );
        }
    }

    #[test]
    fn test_empty_dictionary() {
        let dict = PathMapDictionary::<()>::new();

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "cat", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_no_matches_within_distance() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("cat");
        dict.insert("car");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        // Query "dog" with max distance 1 - should not match "cat" or "car"
        let iter = ZipperQueryIterator::new(dict_zipper, "dog", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_transposition_algorithm() {
        let dict = PathMapDictionary::<()>::new();
        dict.insert("abcd");
        dict.insert("bacd");

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "abcd", 1, Algorithm::Transposition);

        let results: Vec<_> = iter.collect();

        // Should find "abcd" (exact) and "bacd" (transposition of first two chars)
        assert!(results.iter().any(|c| c.term == "abcd"));
        assert!(results.iter().any(|c| c.term == "bacd"));
    }

    #[test]
    fn test_larger_dictionary() {
        let dict = PathMapDictionary::<()>::new();
        let words = vec![
            "cat", "car", "card", "care", "careful", "dog", "door", "dot", "test", "testing",
            "tester",
        ];
        for word in &words {
            dict.insert(word);
        }

        let dict_zipper = PathMapZipper::new_from_dict(&dict);
        let iter = ZipperQueryIterator::new(dict_zipper, "car", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Should find "car", "cat", "card", "care"
        assert!(results.iter().any(|c| c.term == "car"));
        assert!(results.iter().any(|c| c.term == "cat"));
        assert!(results.iter().any(|c| c.term == "card"));
        assert!(results.iter().any(|c| c.term == "care"));

        // Should have at least 4 results
        assert!(results.len() >= 4);
    }
}
