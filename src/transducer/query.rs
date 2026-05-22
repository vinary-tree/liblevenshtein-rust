//! Lazy query iterators for approximate string matching.

use super::query_result::QueryResult;
use super::transition::{initial_state, transition_state_pooled};
use super::{
    Algorithm, Intersection, StatePool, SubstitutionPolicy, SubstitutionPolicyFor, Unrestricted,
};
use libdictenstein::{CharUnit, DictionaryNode};
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Query result containing term and distance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The matching term
    pub term: String,
    /// Edit distance from query
    pub distance: usize,
}

/// Lazy iterator over query matches with configurable result type.
///
/// This iterator can return either:
/// - `String`: Just the matching terms (set `R = String`)
/// - `Candidate`: Terms with their edit distances (set `R = Candidate`)
/// - Custom types: Implement `QueryResult` trait
///
/// The result type is determined by the generic parameter `R`, which defaults
/// to `String` for backward compatibility.
///
/// # Type Parameter
///
/// - `N`: Dictionary node type
/// - `R`: Result type (defaults to `String`). Must implement `QueryResult`.
///
/// # Performance
///
/// Uses StatePool to eliminate State cloning overhead during traversal.
/// The pool is created per-query and reuses State allocations across
/// all transitions, reducing memory allocation by 6-10% of runtime.
///
/// Distance is computed once from automaton states (zero overhead), then
/// converted to the result type via `QueryResult::from_match()` which is
/// inlined and monomorphized at compile time (zero-cost abstraction).
///
/// # Examples
///
/// ```
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::transducer::{QueryIterator, Candidate};
///
/// let dict = DoubleArrayTrie::from_terms(vec!["test"]);
///
/// // Returns String (default)
/// let iter: QueryIterator<_, String> = QueryIterator::new(
///     dict.root(), "test".to_string(), 1, Algorithm::Standard
/// );
/// for term in iter {
///     println!("{}", term);
/// }
///
/// // Returns Candidate (term + distance)
/// let iter: QueryIterator<_, Candidate> = QueryIterator::new(
///     dict.root(), "test".to_string(), 1, Algorithm::Standard
/// );
/// for candidate in iter {
///     println!("{}: {}", candidate.term, candidate.distance);
/// }
/// ```
pub struct QueryIterator<
    N: DictionaryNode,
    R: QueryResult = String,
    P: SubstitutionPolicy = Unrestricted,
> {
    pending: VecDeque<Box<Intersection<N>>>,
    query: Vec<N::Unit>,
    max_distance: usize,
    algorithm: Algorithm,
    policy: P, // Substitution policy for matching
    finished: bool,
    state_pool: StatePool,        // Pool for State allocation reuse
    substring_mode: bool,         // Enable substring matching (for suffix automata)
    _result_type: PhantomData<R>, // Zero-sized marker for result type
}

impl<N: DictionaryNode, R: QueryResult> QueryIterator<N, R, Unrestricted> {
    /// Create a new query iterator with unrestricted policy (standard Levenshtein)
    pub fn new(root: N, query: String, max_distance: usize, algorithm: Algorithm) -> Self {
        Self::with_substring_mode(root, query, max_distance, algorithm, false)
    }

    /// Create a new query iterator with substring matching mode and unrestricted policy
    pub fn with_substring_mode(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        substring_mode: bool,
    ) -> Self {
        Self::with_policy_and_substring(
            root,
            query,
            max_distance,
            algorithm,
            Unrestricted,
            substring_mode,
        )
    }
}

impl<N: DictionaryNode, R: QueryResult, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    QueryIterator<N, R, P>
{
    /// Create a new query iterator with custom substitution policy
    pub fn with_policy(
        root: N,
        query: String,
        max_distance: usize,
        algorithm: Algorithm,
        policy: P,
    ) -> Self {
        Self::with_policy_and_substring(root, query, max_distance, algorithm, policy, false)
    }

    /// Create a new query iterator with custom policy and substring matching mode
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

        let mut pending = VecDeque::new();

        // Always add root to pending queue - it will be checked for finality in advance()
        // and its children will be queued normally
        pending.push_back(Box::new(Intersection::new(root, initial)));

        Self {
            pending,
            query: query_units,
            max_distance,
            algorithm,
            policy,
            finished: false,
            state_pool: StatePool::new(), // Create pool for this query
            substring_mode,
            _result_type: PhantomData, // Zero-sized, no runtime cost
        }
    }

    /// Advance to the next match
    fn advance(&mut self) -> Option<R> {
        while let Some(intersection) = self.pending.pop_front() {
            // Check if this is a final match
            if intersection.is_final() {
                // Infer the distance based on matching mode
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
                    let term = intersection.term();

                    // Queue children for further exploration
                    self.queue_children(&intersection);

                    // Convert (term, distance) to result type R
                    // This is zero-cost: QueryResult::from_match is inlined
                    // and monomorphized at compile time
                    return Some(R::from_match(term, distance));
                } else {
                    // Even if this final node is too far, we must still explore its children
                    // Example: dict ["z", "za"], query "za" (dist=0)
                    // At 'z': is_final=true, distance=1 (too far), but we must explore 'a'
                    self.queue_children(&intersection);
                }
            } else {
                // Not final: queue children to continue exploring
                self.queue_children(&intersection);
            }
        }

        self.finished = true;
        None
    }

    /// Queue child intersections for exploration
    fn queue_children(&mut self, intersection: &Intersection<N>) {
        for (label, child_node) in intersection.node.edges() {
            if let Some(next_state) = transition_state_pooled(
                &intersection.state,
                &mut self.state_pool, // Use pool for State allocation reuse
                self.policy,          // Use the iterator's policy parameter
                label,
                &self.query,
                self.max_distance,
                self.algorithm,
                self.substring_mode, // Use prefix_mode=true only for substring matching
            ) {
                // ✅ Create lightweight PathNode (no Arc clone!)
                // Only stores label and parent chain - dictionary node not needed in parent
                let parent_path = intersection.label.map(|current_label| {
                    Box::new(super::intersection::PathNode::new(
                        current_label,
                        intersection.parent.clone(), // Clone PathNode chain (cheap)
                    ))
                });

                let child = Box::new(Intersection::with_parent(
                    label,
                    child_node,
                    next_state,
                    parent_path, // ← Lightweight path, no node cloning!
                ));

                self.pending.push_back(child);
            }
        }
    }
}

impl<N: DictionaryNode, R: QueryResult, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    Iterator for QueryIterator<N, R, P>
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            None
        } else {
            self.advance()
        }
    }
}

// Type aliases for ergonomic use
/// Type alias for query iterator that returns just term strings.
///
/// Equivalent to `QueryIterator<N, String>`.
pub type StringQueryIterator<N> = QueryIterator<N, String>;

/// Type alias for query iterator that returns Candidate structs (term + distance).
///
/// Equivalent to `QueryIterator<N, Candidate>`.
pub type CandidateIterator<N> = QueryIterator<N, Candidate>;

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use libdictenstein::Dictionary;

    #[test]
    fn test_query_exact_match() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let query: QueryIterator<_, String> =
            QueryIterator::new(dict.root(), "test".to_string(), 0, Algorithm::Standard);

        let result: Vec<_> = query.collect();
        assert_eq!(result, vec!["test"]);
    }

    #[test]
    fn test_query_with_distance() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "testing"]);
        let query = QueryIterator::new(dict.root(), "test".to_string(), 1, Algorithm::Standard);

        let results: Vec<_> = query.collect();
        assert!(results.contains(&"test".to_string()));
        assert!(results.contains(&"best".to_string()));
        assert!(results.contains(&"rest".to_string()));
        // "testing" should not match with distance 1
    }

    #[test]
    fn test_candidate_iterator() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best"]);
        let query = CandidateIterator::new(dict.root(), "test".to_string(), 1, Algorithm::Standard);

        let candidates: Vec<_> = query.collect();
        assert!(candidates
            .iter()
            .any(|c| c.term == "test" && c.distance == 0));
        assert!(candidates
            .iter()
            .any(|c| c.term == "best" && c.distance == 1));
    }

    #[test]
    fn test_empty_query() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let query = QueryIterator::new(dict.root(), "".to_string(), 0, Algorithm::Standard);

        let results: Vec<_> = query.collect();
        // Empty query with distance 0 should match nothing unless dict has empty string
        assert!(results.is_empty() || results.contains(&"".to_string()));
    }
}
