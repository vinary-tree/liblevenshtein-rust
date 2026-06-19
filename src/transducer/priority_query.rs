//! Priority queue-based query iterator using A* search.
//!
//! This module provides an alternative to `OrderedQueryIterator` that uses
//! A* search with heuristics for potentially faster result production.
//!
//! # Comparison with OrderedQueryIterator
//!
//! | Feature | OrderedQueryIterator | PriorityQueryIterator |
//! |---------|---------------------|-----------------------|
//! | Algorithm | BFS with distance buckets | A* with priority queue |
//! | Ordering | Strict distance + lexicographic | Distance-first (approximate lex) |
//! | Memory | O(states per level) | O(all pending states) |
//! | Best for | Exact ordering requirements | Fast first-k results |
//!
//! # A* Heuristic
//!
//! The heuristic estimates remaining cost as:
//! - Characters remaining in query that must be consumed
//! - Minimum possible operations to reach a final state
//!
//! ```text
//! h(state) = max(0, query_len - max_consumed_chars)
//! f(state) = g(state) + h(state)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::transducer::PriorityQueryIterator;
//!
//! let dict = DynamicDawg::from_iter(["apple", "apply", "banana"]);
//! let iter = PriorityQueryIterator::new(dict.root(), "aple", 2, Algorithm::Standard);
//!
//! for candidate in iter.take(3) {
//!     println!("{}: {}", candidate.term, candidate.distance);
//! }
//! ```

use super::transition::{initial_state, transition_state_pooled};
use super::{Algorithm, Intersection, PathNode, State, StatePool, Unrestricted};
use libdictenstein::{CharUnit, DictionaryNode};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Entry in the priority queue for A* search.
struct SearchEntry<N: DictionaryNode> {
    /// Current intersection (state + node + path)
    intersection: Box<Intersection<N>>,
    /// Cached path units for deterministic heap tie-breaking.
    term_units: Vec<N::Unit>,
    /// Actual cost so far (minimum errors in state)
    g_cost: usize,
    /// f-cost = g-cost + heuristic, used for priority ordering
    f_cost: usize,
}

impl<N: DictionaryNode> SearchEntry<N> {
    fn new(
        intersection: Box<Intersection<N>>,
        term_units: Vec<N::Unit>,
        g_cost: usize,
        h_cost: usize,
    ) -> Self {
        Self {
            intersection,
            term_units,
            g_cost,
            f_cost: g_cost.saturating_add(h_cost),
        }
    }
}

// Implement ordering for min-heap (lower f-cost = higher priority)
impl<N: DictionaryNode> Ord for SearchEntry<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: lower f-cost has higher priority (reverse for max-heap → min-heap)
        // Secondary: lower g-cost has higher priority (prefer actual progress)
        // Tertiary: lexicographic on term for determinism
        match other.f_cost.cmp(&self.f_cost) {
            Ordering::Equal => match other.g_cost.cmp(&self.g_cost) {
                Ordering::Equal => self.term_units.cmp(&other.term_units),
                ord => ord,
            },
            ord => ord,
        }
    }
}

impl<N: DictionaryNode> PartialOrd for SearchEntry<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<N: DictionaryNode> PartialEq for SearchEntry<N> {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost == other.f_cost
            && self.g_cost == other.g_cost
            && self.term_units == other.term_units
    }
}

impl<N: DictionaryNode> Eq for SearchEntry<N> {}

/// Priority queue-based query result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriorityCandidate {
    /// The matching term
    pub term: String,
    /// Edit distance from query
    pub distance: usize,
}

/// A* search iterator for approximate string matching.
///
/// Uses a priority queue with f-cost (g + h) ordering to efficiently
/// find matches in approximately distance order.
///
/// # Type Parameters
///
/// * `N` - Dictionary node type
pub struct PriorityQueryIterator<N: DictionaryNode> {
    /// Priority queue ordered by f-cost
    queue: BinaryHeap<SearchEntry<N>>,
    /// Query units (bytes or chars)
    query: Vec<N::Unit>,
    /// Query length for heuristic computation
    query_len: usize,
    /// Maximum distance to explore
    max_distance: usize,
    /// Levenshtein algorithm
    algorithm: Algorithm,
    /// State pool for allocation reuse
    state_pool: StatePool,
}

impl<N: DictionaryNode> PriorityQueryIterator<N> {
    /// Create a new priority query iterator.
    ///
    /// # Arguments
    ///
    /// * `root` - Root node of the dictionary
    /// * `query` - Query string
    /// * `max_distance` - Maximum edit distance to consider
    /// * `algorithm` - Levenshtein algorithm variant
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let iter = PriorityQueryIterator::new(dict.root(), "test", 2, Algorithm::Standard);
    /// ```
    pub fn new(root: N, query: &str, max_distance: usize, algorithm: Algorithm) -> Self {
        let query_units = N::Unit::from_str(query);
        let query_len = query_units.len();
        let initial = initial_state(query_len, max_distance, algorithm);

        let mut queue = BinaryHeap::with_capacity(64);

        // Initialize with root node
        let root_intersection = Box::new(Intersection::new(root, initial));
        let g_cost = 0; // No errors yet at root
        let h_cost = query_len; // Must consume all query characters

        queue.push(SearchEntry::new(
            root_intersection,
            Vec::new(),
            g_cost,
            h_cost,
        ));

        Self {
            queue,
            query: query_units,
            query_len,
            max_distance,
            algorithm,
            state_pool: StatePool::new(),
        }
    }

    /// Compute the heuristic cost for a state.
    ///
    /// The heuristic is the minimum number of operations needed to reach
    /// a final state, which is the number of unconsumed query characters.
    #[inline]
    fn heuristic(&self, state: &State) -> usize {
        // Find the maximum term_index across all positions
        // This represents how much of the query has been consumed
        let max_consumed = state
            .positions()
            .iter()
            .map(|p| p.term_index)
            .max()
            .unwrap_or(0);

        self.query_len.saturating_sub(max_consumed)
    }

    /// Get the next match from the priority queue.
    fn advance(&mut self) -> Option<PriorityCandidate> {
        while let Some(entry) = self.queue.pop() {
            // Check if this is a final match
            if entry.intersection.is_final() {
                let distance = entry
                    .intersection
                    .state
                    .infer_distance(self.query_len)
                    .unwrap_or(usize::MAX);

                if distance <= self.max_distance {
                    // Found a valid match - queue children before returning
                    self.expand_children(&entry);

                    return Some(PriorityCandidate {
                        term: N::Unit::to_string(&entry.term_units),
                        distance,
                    });
                }
            }

            // Expand children regardless of finality
            self.expand_children(&entry);
        }

        None
    }

    /// Expand children of a search entry into the priority queue.
    #[inline]
    fn expand_children(&mut self, entry: &SearchEntry<N>) {
        for (label, child_node) in entry.intersection.node.edges() {
            if let Some(next_state) = transition_state_pooled(
                &entry.intersection.state,
                &mut self.state_pool,
                Unrestricted, // Use unrestricted policy
                label,
                &self.query,
                self.max_distance,
                self.algorithm,
                false, // Not prefix mode
            ) {
                // Compute costs for child
                let g_cost = next_state.min_distance().unwrap_or(0);

                // Prune if already over max distance
                if g_cost > self.max_distance {
                    continue;
                }

                // Heuristic: remaining query characters that haven't been consumed
                // This is used for priority ordering, not for pruning
                let h_cost = self.heuristic(&next_state);

                // Create parent path node from current intersection's label
                let parent_path = entry.intersection.label.map(|current_label| {
                    Box::new(PathNode::new(
                        current_label,
                        entry.intersection.parent.clone(),
                    ))
                });

                // Create child intersection
                let child_intersection = Box::new(Intersection::with_parent(
                    label,
                    child_node,
                    next_state,
                    parent_path,
                ));

                let mut child_term_units = entry.term_units.clone();
                child_term_units.push(label);

                self.queue.push(SearchEntry::new(
                    child_intersection,
                    child_term_units,
                    g_cost,
                    h_cost,
                ));
            }
        }
    }
}

impl<N: DictionaryNode> Iterator for PriorityQueryIterator<N> {
    type Item = PriorityCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        self.advance()
    }
}

/// Convenience function to create a priority query iterator.
pub fn priority_query<N: DictionaryNode>(
    root: N,
    query: &str,
    max_distance: usize,
    algorithm: Algorithm,
) -> PriorityQueryIterator<N> {
    PriorityQueryIterator::new(root, query, max_distance, algorithm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::Dictionary;

    fn test_dict() -> libdictenstein::dynamic_dawg::DynamicDawg {
        let dawg = libdictenstein::dynamic_dawg::DynamicDawg::new();
        for term in ["apple", "apply", "appeal", "banana", "test", "best", "rest"] {
            dawg.insert(term);
        }
        dawg
    }

    #[test]
    fn test_exact_match() {
        let dict = test_dict();
        let mut iter = PriorityQueryIterator::new(dict.root(), "apple", 2, Algorithm::Standard);

        let first = iter.next();
        assert!(first.is_some());
        let candidate = first.expect("test fixture: first candidate exists (asserted above)");
        assert_eq!(candidate.term, "apple");
        assert_eq!(candidate.distance, 0);
    }

    #[test]
    fn test_close_matches() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "aple", 2, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Should find "apple" and "apply" within distance 2
        let terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();
        assert!(
            terms.contains(&"apple"),
            "Should contain 'apple': {:?}",
            terms
        );
    }

    #[test]
    fn test_distance_ordering() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "test", 2, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Check that results are roughly ordered by distance
        // (A* doesn't guarantee strict ordering, but close matches should come first)
        if results.len() >= 2 {
            let first_dist = results[0].distance;
            // First result should be the exact match or very close
            assert!(
                first_dist <= 1,
                "First result distance should be <= 1, got {}",
                first_dist
            );
        }
    }

    #[test]
    fn test_max_distance_respected() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "xyz", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // All results should be within max_distance
        for candidate in &results {
            assert!(
                candidate.distance <= 1,
                "Distance {} exceeds max 1 for term {}",
                candidate.distance,
                candidate.term
            );
        }
    }

    #[test]
    fn test_empty_query() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "", 3, Algorithm::Standard);

        let results: Vec<_> = iter.collect();

        // Empty query matches short terms within distance 3
        // All 3-letter terms should match
        let terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();
        // May or may not have results depending on dictionary
        // Just ensure no panic
        let _ = terms;
    }

    #[test]
    fn test_transposition() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "tset", 2, Algorithm::Transposition);

        let results: Vec<_> = iter.collect();

        // "tset" with transposition should find "test" at distance 1
        let test_result = results.iter().find(|c| c.term == "test");
        assert!(
            test_result.is_some(),
            "Should find 'test' for 'tset' with transposition"
        );

        if let Some(candidate) = test_result {
            assert_eq!(candidate.distance, 1, "Transposition should be distance 1");
        }
    }

    #[test]
    fn test_take_early() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "test", 2, Algorithm::Standard);

        // Take only first 2 results
        let results: Vec<_> = iter.take(2).collect();
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_no_matches() {
        let dict = test_dict();
        let iter = PriorityQueryIterator::new(dict.root(), "zzzzzzzzz", 1, Algorithm::Standard);

        let results: Vec<_> = iter.collect();
        assert!(
            results.is_empty(),
            "Should find no matches for 'zzzzzzzzz' within distance 1"
        );
    }
}
