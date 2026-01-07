//! Ordered query iterators that return results by distance, then lexicographically.
//!
//! This module provides iterators that yield spelling candidates in a specific order:
//! 1. Primary: Ascending edit distance (0, 1, 2, ...)
//! 2. Secondary: Lexicographic (alphabetical)
//!
//! This ordering enables efficient "top-k" queries and take-while patterns.

use super::transition::{initial_state, transition_state_pooled};
use super::{Algorithm, Intersection, PathNode, StatePool, SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted};
use libdictenstein::{CharUnit, DictionaryNode};
use std::collections::VecDeque;

/// Query result containing term and distance.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OrderedCandidate {
    /// Edit distance from query (primary sort key)
    pub distance: usize,
    /// The matching term (secondary sort key - lexicographic)
    pub term: String,
}

/// Lazy iterator that returns candidates in distance-first, lexicographic order.
///
/// This iterator yields all distance=0 matches first (exact matches), then all
/// distance=1 matches (alphabetically), then distance=2, etc. This ordering
/// enables efficient "top-k" queries using `take(n)` and distance-bounded
/// queries using `take_while`.
///
/// # Ordering Guarantees
///
/// 1. **Primary:** Results are ordered by ascending edit distance
/// 2. **Secondary:** Within each distance, results are lexicographically ordered
///
/// # Performance
///
/// - Explores the search space in distance layers (BFS-like)
/// - Uses StatePool for allocation reuse
/// - Leverages pre-sorted DAWG edges for lexicographic ordering
/// - Truly lazy - can stop early with `take(n)` or `take_while`
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::prelude::*;
///
/// let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "testing"]);
/// let transducer = Transducer::new(dict, Algorithm::Standard);
///
/// // Get first 3 closest matches
/// for candidate in transducer.query_ordered("tset", 2).take(3) {
///     println!("{}: {}", candidate.term, candidate.distance);
/// }
/// // Output (in order):
/// // test: 0
/// // best: 1
/// // rest: 1
///
/// // Get all matches within distance 1
/// for candidate in transducer.query_ordered("tset", 2).take_while(|c| c.distance <= 1) {
///     println!("{}", candidate.term);
/// }
/// ```
pub struct OrderedQueryIterator<N: DictionaryNode, P: SubstitutionPolicy = Unrestricted> {
    /// Pending intersections grouped by minimum distance
    pending_by_distance: Vec<VecDeque<Box<Intersection<N>>>>,
    /// Current distance level being explored
    current_distance: usize,
    /// Maximum distance to explore
    max_distance: usize,
    /// Query units (bytes or chars)
    query: Vec<N::Unit>,
    /// Levenshtein algorithm
    algorithm: Algorithm,
    /// Substitution policy
    policy: P,
    /// State pool for allocation reuse
    state_pool: StatePool,
    /// Substring matching mode (for suffix automata)
    substring_mode: bool,
    /// Sorted buffer for current distance level (ensures lexicographic ordering)
    sorted_buffer: Vec<OrderedCandidate>,
    /// Index into sorted_buffer for next result
    buffer_index: usize,
}

impl<N: DictionaryNode> OrderedQueryIterator<N, Unrestricted> {
    /// Create a new ordered query iterator with unrestricted policy
    pub fn new(root: N, query: String, max_distance: usize, algorithm: Algorithm) -> Self {
        Self::with_substring_mode(root, query, max_distance, algorithm, false)
    }

    /// Create a new ordered query iterator with substring matching mode and unrestricted policy
    pub fn with_substring_mode(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        substring_mode: bool,
    ) -> Self {
        Self::with_policy_and_substring(root, query, max_distance, algorithm, Unrestricted, substring_mode)
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>> OrderedQueryIterator<N, P> {
    /// Create a new ordered query iterator with custom substitution policy
    pub fn with_policy(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
    ) -> Self {
        Self::with_policy_and_substring(root, query, max_distance, algorithm, policy, false)
    }

    /// Create a new ordered query iterator with custom policy and substring matching mode
    pub fn with_policy_and_substring(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        let query_units = N::Unit::from_str(&query);
        let initial = initial_state(query_units.len(), max_distance, algorithm);

        // Create buckets for each distance level (0..=max_distance)
        // Pre-allocate capacity to reduce reallocations during traversal
        let mut pending_by_distance: Vec<VecDeque<_>> = (0..=max_distance)
            .map(|_| VecDeque::with_capacity(32))
            .collect();

        // Start with root at distance 0 - it will be checked for finality in advance()
        pending_by_distance[0].push_back(Box::new(Intersection::new(root, initial)));

        Self {
            pending_by_distance,
            current_distance: 0,
            max_distance,
            query: query_units,
            algorithm,
            policy,
            state_pool: StatePool::new(),
            substring_mode,
            sorted_buffer: Vec::with_capacity(64), // Heuristic: typical max results per distance
            buffer_index: 0,
        }
    }

    /// Advance to the next match in order
    #[inline]
    fn advance(&mut self) -> Option<OrderedCandidate> {
        // First, check if we have buffered results to yield
        if self.buffer_index < self.sorted_buffer.len() {
            let result = self.sorted_buffer[self.buffer_index].clone();
            self.buffer_index += 1;
            return Some(result);
        }

        // Buffer is exhausted, need to collect next distance level
        while self.current_distance <= self.max_distance {
            // Clear buffer and reset index for new distance level
            self.sorted_buffer.clear();
            self.buffer_index = 0;

            // Collect ALL results at the current distance level
            while let Some(intersection) =
                self.pending_by_distance[self.current_distance].pop_front()
            {
                // Check if this is a final match
                let is_final = intersection.is_final();
                if is_final {
                    // Compute distance based on matching mode
                    let distance = if self.substring_mode {
                        // Substring mode: don't penalize unmatched query suffix
                        intersection.state.min_distance().unwrap_or(usize::MAX)
                    } else {
                        // Standard mode: penalize remaining query characters
                        intersection
                            .state
                            .infer_distance(self.query.len())
                            .unwrap_or(usize::MAX)
                    };

                    if distance <= self.max_distance {
                        if distance == self.current_distance {
                            // Distance matches current level - add to buffer
                            let term = intersection.term();
                            self.sorted_buffer.push(OrderedCandidate { distance, term });

                            // Queue children for further exploration
                            self.queue_children(&intersection);
                        } else if distance > self.current_distance {
                            // Actual distance is higher than bucket - requeue to correct bucket
                            // This can happen when min_dist underestimates final distance
                            self.pending_by_distance[distance].push_back(intersection);
                        }
                        // If distance < current_distance, skip (already passed that level)
                    } else {
                        // Distance exceeds max_distance, but still queue children
                        self.queue_children(&intersection);
                    }
                } else {
                    // Not final, queue children for further exploration
                    self.queue_children(&intersection);
                }
            }

            // If we collected any results at this distance, sort them and return first
            if !self.sorted_buffer.is_empty() {
                // Adaptive sorting: insertion sort for small n, unstable sort for larger n
                // Threshold of 10 is empirically good for sorting algorithms crossover
                if self.sorted_buffer.len() <= 10 {
                    // For small buffers, insertion sort is faster due to better cache locality
                    for i in 1..self.sorted_buffer.len() {
                        let mut j = i;
                        while j > 0 && self.sorted_buffer[j].term < self.sorted_buffer[j - 1].term {
                            self.sorted_buffer.swap(j, j - 1);
                            j -= 1;
                        }
                    }
                } else {
                    // For larger buffers, use unstable sort (faster, doesn't preserve order of equal elements)
                    self.sorted_buffer
                        .sort_unstable_by(|a, b| a.term.cmp(&b.term));
                }

                // Return first result from buffer
                self.buffer_index = 1;
                return Some(self.sorted_buffer[0].clone());
            }

            // No results at this distance, move to next
            self.current_distance += 1;
        }

        None
    }

    /// Queue child intersections into appropriate distance buckets
    #[inline]
    fn queue_children(&mut self, intersection: &Intersection<N>) {
        // Edges are iterated in sorted order (lexicographic) thanks to DAWG construction
        for (label, child_node) in intersection.node.edges() {
            if let Some(next_state) = transition_state_pooled(
                &intersection.state,
                &mut self.state_pool,
                self.policy, // Use the iterator's policy parameter
                label,
                &self.query,
                self.max_distance,
                self.algorithm,
                self.substring_mode, // Use prefix_mode=true only for substring matching
            ) {
                // Determine minimum possible distance from this state
                if let Some(min_dist) = next_state.min_distance() {
                    if min_dist <= self.max_distance {
                        // Create lightweight PathNode for parent chain
                        let parent_path = intersection.label.map(|current_label| {
                            Box::new(PathNode::new(current_label, intersection.parent.clone()))
                        });

                        let child = Box::new(Intersection::with_parent(
                            label,
                            child_node,
                            next_state,
                            parent_path,
                        ));

                        // Add to the appropriate distance bucket
                        self.pending_by_distance[min_dist].push_back(child);
                    }
                }
            }
        }
    }

    /// Add a filter predicate to this iterator.
    ///
    /// Returns a new iterator that only yields candidates matching the predicate.
    /// The filter is applied during traversal, allowing early termination.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Filter to only identifiers starting with lowercase
    /// query.filter(|candidate| {
    ///     candidate.term.chars().next()
    ///         .map(|c| c.is_lowercase())
    ///         .unwrap_or(false)
    /// })
    /// ```
    pub fn filter<F>(self, predicate: F) -> FilteredOrderedQueryIterator<N, P, F>
    where
        F: Fn(&OrderedCandidate) -> bool,
    {
        FilteredOrderedQueryIterator {
            inner: self,
            predicate,
        }
    }

    /// Switch to prefix matching mode.
    ///
    /// In prefix mode, dictionary terms that START with something approximately
    /// equal to the query are matched, allowing terms to be longer than the query.
    ///
    /// This is essential for autocomplete/code completion where users type partial
    /// identifiers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Query: "tes"
    /// // Matches: "test" (d=0), "testing" (d=0), "tester" (d=0), "best" (d=1)
    /// query.prefix()
    /// ```
    pub fn prefix(mut self) -> PrefixOrderedQueryIterator<N, P> {
        // Enable substring mode for prefix matching
        // This allows matching terms that start with the query without penalizing the unmatched suffix
        self.substring_mode = true;
        PrefixOrderedQueryIterator { inner: self }
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>> Iterator for OrderedQueryIterator<N, P> {
    type Item = OrderedCandidate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.advance()
    }
}

/// Filtered ordered query iterator.
///
/// Wraps an OrderedQueryIterator and applies a filter predicate to results.
/// Only candidates matching the predicate are yielded.
pub struct FilteredOrderedQueryIterator<N: DictionaryNode, P: SubstitutionPolicy, F>
where
    F: Fn(&OrderedCandidate) -> bool,
{
    inner: OrderedQueryIterator<N, P>,
    predicate: F,
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>, F> Iterator for FilteredOrderedQueryIterator<N, P, F>
where
    F: Fn(&OrderedCandidate) -> bool,
{
    type Item = OrderedCandidate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Keep advancing until we find a match or exhaust the iterator
        loop {
            let candidate = self.inner.next()?;
            if (self.predicate)(&candidate) {
                return Some(candidate);
            }
        }
    }
}

/// Prefix ordered query iterator.
///
/// Performs approximate prefix matching where dictionary terms that START with
/// something approximately equal to the query are matched. Terms can be longer
/// than the query.
///
/// Essential for autocomplete and code completion.
pub struct PrefixOrderedQueryIterator<N: DictionaryNode, P: SubstitutionPolicy = Unrestricted> {
    inner: OrderedQueryIterator<N, P>,
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>> PrefixOrderedQueryIterator<N, P> {
    /// Advance to the next prefix match in order
    #[inline]
    fn advance_prefix(&mut self) -> Option<OrderedCandidate> {
        let query_len = self.inner.query.len();

        // Explore distance levels in ascending order
        while self.inner.current_distance <= self.inner.max_distance {
            // Try to get next intersection from current distance level
            if let Some(intersection) =
                self.inner.pending_by_distance[self.inner.current_distance].pop_front()
            {
                // Check if this is a complete word (final node) that matches our prefix
                let should_return = if intersection.is_final() {
                    // For prefix matching: check if we've consumed the entire query
                    if let Some(distance) = intersection.state.infer_prefix_distance(query_len) {
                        distance <= self.inner.max_distance
                            && distance == self.inner.current_distance
                    } else {
                        false
                    }
                } else {
                    false
                };

                // Always queue children for further exploration
                self.inner.queue_children(&intersection);

                // Return the result if it's a complete word matching our prefix
                if should_return {
                    let term = intersection.term();
                    let distance = intersection.state.infer_prefix_distance(query_len).unwrap();
                    return Some(OrderedCandidate { distance, term });
                }
            } else {
                // Current distance level exhausted, move to next
                self.inner.current_distance += 1;
            }
        }

        None
    }
}

impl<N: DictionaryNode, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>> Iterator for PrefixOrderedQueryIterator<N, P> {
    type Item = OrderedCandidate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.advance_prefix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::Dictionary;

    #[test]
    fn test_ordered_exact_match() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 0, Algorithm::Standard);

        let results: Vec<_> = query.collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "test");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_ordered_distance_first() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "test",    // distance 0
            "best",    // distance 1
            "rest",    // distance 1
            "testing", // distance 3
            "nest",    // distance 1
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 3, Algorithm::Standard);

        let results: Vec<_> = query.collect();

        // Verify distance ordering
        for i in 1..results.len() {
            assert!(
                results[i - 1].distance <= results[i].distance,
                "Distance ordering violated: {} (d={}) should come before {} (d={})",
                results[i - 1].term,
                results[i - 1].distance,
                results[i].term,
                results[i].distance
            );
        }

        // Verify exact match comes first
        assert_eq!(results[0].term, "test");
        assert_eq!(results[0].distance, 0);
    }

    #[test]
    fn test_ordered_lexicographic_within_distance() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "test", "best", "fest", "nest", "rest", "west", "zest",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 1, Algorithm::Standard);

        let results: Vec<_> = query.collect();

        // Group by distance
        let mut by_distance: Vec<Vec<String>> = vec![Vec::new(); 2];
        for candidate in results {
            by_distance[candidate.distance].push(candidate.term);
        }

        // Verify distance 0
        assert_eq!(by_distance[0], vec!["test"]);

        // Verify distance 1 is lexicographically sorted
        let dist1 = &by_distance[1];
        for i in 1..dist1.len() {
            assert!(
                dist1[i - 1] <= dist1[i],
                "Lexicographic ordering violated: {} should come before {}",
                dist1[i - 1],
                dist1[i]
            );
        }
    }

    #[test]
    fn test_ordered_take() {
        let dict =
            DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "nest", "testing", "resting"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 3, Algorithm::Standard);

        // Take only first 3 results
        let results: Vec<_> = query.take(3).collect();
        assert_eq!(results.len(), 3);

        // First should be exact match
        assert_eq!(results[0].distance, 0);
        assert_eq!(results[0].term, "test");

        // Next two should be distance 1
        assert_eq!(results[1].distance, 1);
        assert_eq!(results[2].distance, 1);
    }

    #[test]
    fn test_ordered_take_while() {
        let dict =
            DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "nest", "testing", "resting"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 3, Algorithm::Standard);

        // Take while distance <= 1
        let results: Vec<_> = query.take_while(|c| c.distance <= 1).collect();

        // All results should have distance <= 1
        for candidate in &results {
            assert!(candidate.distance <= 1);
        }

        // Should include exact match
        assert!(results.iter().any(|c| c.term == "test" && c.distance == 0));

        // Should not include distance 3 results
        assert!(!results.iter().any(|c| c.term == "testing"));
        assert!(!results.iter().any(|c| c.term == "resting"));
    }

    #[test]
    fn test_ordered_empty_query() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "xyz".to_string(), 0, Algorithm::Standard);

        let results: Vec<_> = query.collect();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_ordered_consistency_with_unordered() {
        // Verify ordered iterator returns same results as unordered, just in different order
        use crate::transducer::query::QueryIterator;

        let dict =
            DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "nest", "fest", "testing"]);

        let ordered =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 2, Algorithm::Standard);

        let unordered: QueryIterator<_, String> =
            QueryIterator::new(dict.root(), "test".to_string(), 2, Algorithm::Standard);

        let mut ordered_terms: Vec<_> = ordered.map(|c| c.term).collect();
        let mut unordered_terms: Vec<_> = unordered.collect();

        ordered_terms.sort();
        unordered_terms.sort();

        assert_eq!(ordered_terms, unordered_terms);
    }

    #[test]
    fn test_filtered_query() {
        let dict =
            DoubleArrayTrie::from_terms(vec!["test", "Test", "TEST", "best", "Best", "rest"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 1, Algorithm::Standard);

        // Filter to only lowercase terms
        let results: Vec<_> = query
            .filter(|c| c.term.chars().all(|ch| ch.is_lowercase()))
            .collect();

        // Should only include lowercase results
        for candidate in &results {
            assert!(candidate.term.chars().all(|ch| ch.is_lowercase()));
        }

        // Should include lowercase matches
        assert!(results.iter().any(|c| c.term == "test"));
        assert!(results.iter().any(|c| c.term == "best"));
        assert!(results.iter().any(|c| c.term == "rest"));

        // Should NOT include uppercase matches
        assert!(!results.iter().any(|c| c.term == "Test"));
        assert!(!results.iter().any(|c| c.term == "TEST"));
        assert!(!results.iter().any(|c| c.term == "Best"));
    }

    #[test]
    fn test_filtered_query_with_distance() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "best", "rest", "nest"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 3, Algorithm::Standard);

        // Filter to terms with exactly 4 characters
        let results: Vec<_> = query.filter(|c| c.term.len() == 4).collect();

        // All results should have exactly 4 characters
        for candidate in &results {
            assert_eq!(candidate.term.len(), 4);
        }

        // Should include 4-letter matches
        assert!(results.iter().any(|c| c.term == "test"));
        assert!(results.iter().any(|c| c.term == "best"));
        assert!(results.iter().any(|c| c.term == "rest"));
        assert!(results.iter().any(|c| c.term == "nest"));

        // Should NOT include longer terms
        assert!(!results.iter().any(|c| c.term == "testing"));
    }

    #[test]
    fn test_filtered_query_maintains_order() {
        let dict =
            DoubleArrayTrie::from_terms(vec!["a", "aa", "aaa", "ab", "abc", "b", "ba", "baa"]);

        let query = OrderedQueryIterator::new(dict.root(), "a".to_string(), 2, Algorithm::Standard);

        // Filter to terms starting with 'a'
        let results: Vec<_> = query.filter(|c| c.term.starts_with('a')).collect();

        // Verify ordering is maintained (distance-first, then lexicographic)
        for i in 1..results.len() {
            assert!(
                results[i - 1].distance <= results[i].distance,
                "Distance ordering violated"
            );

            if results[i - 1].distance == results[i].distance {
                assert!(
                    results[i - 1].term <= results[i].term,
                    "Lexicographic ordering violated within distance level"
                );
            }
        }
    }

    #[test]
    fn test_filtered_query_with_take() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "test", "testing", "tester", "best", "rest", "nest", "fest",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 2, Algorithm::Standard);

        // Filter to terms ending with 'st' and take first 3
        let results: Vec<_> = query.filter(|c| c.term.ends_with("st")).take(3).collect();

        assert_eq!(results.len(), 3);

        // All should end with 'st'
        for candidate in &results {
            assert!(candidate.term.ends_with("st"));
        }

        // Should be ordered by distance
        assert!(results[0].distance <= results[1].distance);
        assert!(results[1].distance <= results[2].distance);
    }

    #[test]
    fn test_prefix_exact_match() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tester", "tested"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 0, Algorithm::Standard);

        let results: Vec<_> = query.prefix().collect();

        // Should match all terms starting with "test" exactly
        assert!(
            results.len() >= 4,
            "Expected at least 4 results, got {}",
            results.len()
        );
        assert!(results.iter().any(|c| c.term == "test" && c.distance == 0));
        assert!(results
            .iter()
            .any(|c| c.term == "testing" && c.distance == 0));
        assert!(results
            .iter()
            .any(|c| c.term == "tester" && c.distance == 0));
        assert!(results
            .iter()
            .any(|c| c.term == "tested" && c.distance == 0));
    }

    #[test]
    fn test_prefix_with_errors() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "best", "resting", "rest"]);

        let query =
            OrderedQueryIterator::new(dict.root(), "tes".to_string(), 1, Algorithm::Standard);

        let results: Vec<_> = query.prefix().collect();

        // Should match:
        // - "test", "testing" with d=0 (exact prefix match)
        // - "best", "rest", "resting" with d=1 (one error in prefix)
        assert!(results.iter().any(|c| c.term == "test" && c.distance == 0));
        assert!(results
            .iter()
            .any(|c| c.term == "testing" && c.distance == 0));
        assert!(results.iter().any(|c| c.term == "best" && c.distance == 1));
        assert!(results.iter().any(|c| c.term == "rest" && c.distance == 1));
    }

    #[test]
    fn test_prefix_ordering() {
        let dict = DoubleArrayTrie::from_terms(vec![
            "test", "testing", "tester", "best", "resting", "rest",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 2, Algorithm::Standard);

        let results: Vec<_> = query.prefix().collect();

        // Verify distance-first ordering
        for i in 1..results.len() {
            assert!(
                results[i - 1].distance <= results[i].distance,
                "Distance ordering violated: {} (d={}) should come before {} (d={})",
                results[i - 1].term,
                results[i - 1].distance,
                results[i].term,
                results[i].distance
            );
        }

        // First results should be distance=0
        let first_distance = results[0].distance;
        assert_eq!(first_distance, 0, "First result should have distance 0");
    }

    #[test]
    fn test_prefix_vs_exact() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tester"]);

        // Exact matching
        let exact_query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 0, Algorithm::Standard);

        let exact_results: Vec<_> = exact_query.collect();

        // Prefix matching
        let prefix_query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 0, Algorithm::Standard);

        let prefix_results: Vec<_> = prefix_query.prefix().collect();

        // Exact should only match "test"
        assert_eq!(exact_results.len(), 1);
        assert_eq!(exact_results[0].term, "test");

        // Prefix should match all terms starting with "test"
        assert!(prefix_results.len() >= 3);
        assert!(prefix_results.iter().any(|c| c.term == "test"));
        assert!(prefix_results.iter().any(|c| c.term == "testing"));
        assert!(prefix_results.iter().any(|c| c.term == "tester"));
    }

    #[test]
    fn test_prefix_autocomplete_scenario() {
        // Simulating code completion
        let dict = DoubleArrayTrie::from_terms(vec![
            "getValue",
            "getVariable",
            "getValue2",
            "setValue",
            "setVariable",
            "removeValue",
            "hasValue",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "getVal".to_string(), 1, Algorithm::Standard);

        let results: Vec<_> = query.prefix().take(5).collect();

        // Should prioritize exact prefix matches
        // Results should be ordered by distance, then alphabetically
        for candidate in &results {
            println!("{}: {}", candidate.term, candidate.distance);
        }

        // Should include getValue family with low distance
        assert!(results.iter().any(|c| c.term.starts_with("getValue")));
    }

    #[test]
    fn test_prefix_with_filter() {
        // Combining prefix matching with filtering
        let dict = DoubleArrayTrie::from_terms(vec![
            "TestCase",
            "testMethod",
            "testHelper",
            "bestPractice",
        ]);

        let query =
            OrderedQueryIterator::new(dict.root(), "test".to_string(), 1, Algorithm::Standard);

        // Prefix match + filter for lowercase
        let results: Vec<_> = query
            .prefix()
            .filter(|c| c.term.chars().next().unwrap().is_lowercase())
            .collect();

        // Should only include lowercase-starting matches
        for candidate in &results {
            assert!(candidate.term.chars().next().unwrap().is_lowercase());
        }

        assert!(results.iter().any(|c| c.term == "testMethod"));
        assert!(results.iter().any(|c| c.term == "testHelper"));
        assert!(!results.iter().any(|c| c.term == "TestCase"));
    }
}
