//! Lazy query iterators for float-weighted approximate string matching.
//!
//! This module provides `QueryIteratorF64`, which performs fuzzy queries
//! using configurable float costs per operation type.
//!
//! # Differences from Integer Query Iterator
//!
//! | Aspect | Integer (`query.rs`) | Float (`query_f64.rs`) |
//! |--------|----------------------|------------------------|
//! | Distance type | `usize` | `f64` |
//! | Threshold | `max_distance: usize` | `max_cost: f64` |
//! | Costs | Hardcoded `+1` | `OperationCostsF64` |
//! | Result | `Candidate { distance: usize }` | `CandidateF64 { distance: f64 }` |

use super::intersection::PathNode;
use super::intersection_f64::IntersectionF64;
use super::transition_f64::{initial_state_f64, transition_state_pooled_f64};
use super::{
    Algorithm, OperationCostsF64, StatePoolF64, SubstitutionPolicy, SubstitutionPolicyFor,
    Unrestricted,
};
use crate::dictionary::{CharUnit, DictionaryNode};
use std::collections::VecDeque;
use std::marker::PhantomData;

/// Query result containing term and float distance.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateF64 {
    /// The matching term
    pub term: String,
    /// Edit distance from query (float)
    pub distance: f64,
}

/// Trait for converting a match (term + float distance) into a result type.
///
/// Similar to `QueryResult` but for float distances.
pub trait QueryResultF64: Sized {
    /// Convert a match into the result type.
    fn from_match(term: String, distance: f64) -> Self;
}

/// Implementation for String: returns just the term, ignoring distance.
impl QueryResultF64 for String {
    #[inline]
    fn from_match(term: String, _distance: f64) -> Self {
        term
    }
}

/// Implementation for CandidateF64: returns both term and distance.
impl QueryResultF64 for CandidateF64 {
    #[inline]
    fn from_match(term: String, distance: f64) -> Self {
        CandidateF64 { term, distance }
    }
}

/// Lazy iterator over float-weighted query matches.
///
/// This iterator uses configurable float costs for edit operations and
/// returns results with float distances.
///
/// # Type Parameters
///
/// - `N`: Dictionary node type
/// - `R`: Result type (defaults to `String`). Must implement `QueryResultF64`.
/// - `P`: Substitution policy (defaults to `Unrestricted`)
///
/// # Performance
///
/// Uses `StatePoolF64` to eliminate `StateF64` cloning overhead during traversal.
/// The pool is created per-query and reuses `StateF64` allocations across
/// all transitions.
///
/// # Examples
///
/// ```
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::transducer::{QueryIteratorF64, CandidateF64, OperationCostsF64};
///
/// let dict = DoubleArrayTrie::from_terms(vec!["test", "best"]);
///
/// // Standard costs (all 1.0)
/// let costs = OperationCostsF64::standard();
///
/// // Returns CandidateF64 (term + float distance)
/// let iter: QueryIteratorF64<_, CandidateF64> = QueryIteratorF64::new(
///     dict.root(), "test".to_string(), 2.0, Algorithm::Standard, costs
/// );
/// for candidate in iter {
///     println!("{}: {:.2}", candidate.term, candidate.distance);
/// }
/// ```
pub struct QueryIteratorF64<
    N: DictionaryNode,
    R: QueryResultF64 = String,
    P: SubstitutionPolicy = Unrestricted,
> {
    pending: VecDeque<Box<IntersectionF64<N>>>,
    query: Vec<N::Unit>,
    max_cost: f64,
    algorithm: Algorithm,
    costs: OperationCostsF64,
    policy: P,
    finished: bool,
    state_pool: StatePoolF64,
    substring_mode: bool,
    _result_type: PhantomData<R>,
}

impl<N: DictionaryNode, R: QueryResultF64> QueryIteratorF64<N, R, Unrestricted> {
    /// Create a new float-weighted query iterator with unrestricted policy.
    pub fn new(
        root: N,
        query: String,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
    ) -> Self {
        Self::with_substring_mode(root, query, max_cost, algorithm, costs, false)
    }

    /// Create a new float-weighted query iterator with substring matching mode.
    pub fn with_substring_mode(
        root: N,
        query: String,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
        substring_mode: bool,
    ) -> Self {
        Self::with_policy_and_substring(
            root,
            query,
            max_cost,
            algorithm,
            costs,
            Unrestricted,
            substring_mode,
        )
    }
}

impl<N: DictionaryNode, R: QueryResultF64, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    QueryIteratorF64<N, R, P>
{
    /// Create a new float-weighted query iterator with custom substitution policy.
    pub fn with_policy(
        root: N,
        query: String,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
        policy: P,
    ) -> Self {
        Self::with_policy_and_substring(root, query, max_cost, algorithm, costs, policy, false)
    }

    /// Create a new float-weighted query iterator with custom policy and substring mode.
    pub fn with_policy_and_substring(
        root: N,
        query: String,
        max_cost: f64,
        algorithm: Algorithm,
        costs: OperationCostsF64,
        policy: P,
        substring_mode: bool,
    ) -> Self {
        let query_units = N::Unit::from_str(&query);
        let initial = initial_state_f64(query_units.len(), max_cost, algorithm, &costs);

        let mut pending = VecDeque::new();
        pending.push_back(Box::new(IntersectionF64::new(root, initial)));

        Self {
            pending,
            query: query_units,
            max_cost,
            algorithm,
            costs,
            policy,
            finished: false,
            state_pool: StatePoolF64::new(),
            substring_mode,
            _result_type: PhantomData,
        }
    }

    /// Advance to the next match
    fn advance(&mut self) -> Option<R> {
        while let Some(intersection) = self.pending.pop_front() {
            // Check if this is a final match
            if intersection.is_final() {
                // Infer the distance based on matching mode
                let distance = if self.substring_mode {
                    intersection.state.min_distance().unwrap_or(f64::INFINITY)
                } else {
                    intersection
                        .state
                        .infer_distance(self.query.len())
                        .unwrap_or(f64::INFINITY)
                };

                if distance <= self.max_cost + 1e-9 {
                    let term = intersection.term();

                    // Queue children for further exploration
                    self.queue_children(&intersection);

                    return Some(R::from_match(term, distance));
                } else {
                    // Even if too far, explore children
                    self.queue_children(&intersection);
                }
            } else {
                self.queue_children(&intersection);
            }
        }

        self.finished = true;
        None
    }

    /// Queue child intersections for exploration
    fn queue_children(&mut self, intersection: &IntersectionF64<N>) {
        for (label, child_node) in intersection.node.edges() {
            if let Some(next_state) = transition_state_pooled_f64(
                &intersection.state,
                &mut self.state_pool,
                self.policy,
                label,
                &self.query,
                self.max_cost,
                self.algorithm,
                &self.costs,
                self.substring_mode,
            ) {
                // Create lightweight PathNode
                let parent_path = intersection.label.map(|current_label| {
                    Box::new(PathNode::new(current_label, intersection.parent.clone()))
                });

                let child = Box::new(IntersectionF64::with_parent(
                    label,
                    child_node,
                    next_state,
                    parent_path,
                ));

                self.pending.push_back(child);
            }
        }
    }
}

impl<N: DictionaryNode, R: QueryResultF64, P: SubstitutionPolicy + SubstitutionPolicyFor<N::Unit>>
    Iterator for QueryIteratorF64<N, R, P>
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

/// Type alias for float-weighted query iterator that returns just term strings.
pub type StringQueryIteratorF64<N> = QueryIteratorF64<N, String>;

/// Type alias for float-weighted query iterator that returns CandidateF64 structs.
pub type CandidateIteratorF64<N> = QueryIteratorF64<N, CandidateF64>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::double_array_trie::DoubleArrayTrie;
    use crate::dictionary::Dictionary;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_query_exact_match() {
        let dict = DoubleArrayTrie::from_terms(vec!["test"]);
        let costs = OperationCostsF64::standard();
        let query: QueryIteratorF64<_, String> =
            QueryIteratorF64::new(dict.root(), "test".to_string(), 0.0, Algorithm::Standard, costs);

        let result: Vec<_> = query.collect();
        assert_eq!(result, vec!["test"]);
    }

    #[test]
    fn test_query_with_distance() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best", "rest", "testing"]);
        let costs = OperationCostsF64::standard();
        let query = QueryIteratorF64::new(
            dict.root(),
            "test".to_string(),
            1.0,
            Algorithm::Standard,
            costs,
        );

        let results: Vec<_> = query.collect();
        assert!(results.contains(&"test".to_string()));
        assert!(results.contains(&"best".to_string()));
        assert!(results.contains(&"rest".to_string()));
    }

    #[test]
    fn test_candidate_iterator_f64() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "best"]);
        let costs = OperationCostsF64::standard();
        let query = CandidateIteratorF64::new(
            dict.root(),
            "test".to_string(),
            1.0,
            Algorithm::Standard,
            costs,
        );

        let candidates: Vec<_> = query.collect();
        assert!(candidates
            .iter()
            .any(|c| c.term == "test" && c.distance.abs() < EPSILON));
        assert!(candidates
            .iter()
            .any(|c| c.term == "best" && (c.distance - 1.0).abs() < EPSILON));
    }

    #[test]
    fn test_custom_costs() {
        let dict = DoubleArrayTrie::from_terms(vec!["abc", "axc"]);
        // Make substitution expensive (2.0)
        let costs = OperationCostsF64::custom(2.0, 1.0, 1.0, 1.0, 1.0, 1.0);

        let query = CandidateIteratorF64::new(
            dict.root(),
            "abc".to_string(),
            1.5, // Only allows insertion/deletion, not substitution
            Algorithm::Standard,
            costs,
        );

        let candidates: Vec<_> = query.collect();
        // "abc" should match exactly (distance 0)
        assert!(candidates.iter().any(|c| c.term == "abc"));
        // "axc" requires substitution (cost 2.0) which exceeds threshold
        assert!(!candidates.iter().any(|c| c.term == "axc"));
    }

    #[test]
    fn test_typo_friendly_transposition() {
        let dict = DoubleArrayTrie::from_terms(vec!["the", "teh"]);
        let costs = OperationCostsF64::typo_friendly();

        // Transposition cost is 0.5 in typo_friendly, so threshold of 1.0 captures it
        let query = CandidateIteratorF64::new(
            dict.root(),
            "the".to_string(),
            1.0, // Threshold covers transposition cost of 0.5
            Algorithm::Transposition,
            costs,
        );

        let candidates: Vec<_> = query.collect();
        // "the" should match exactly (distance 0)
        assert!(candidates.iter().any(|c| c.term == "the" && c.distance.abs() < EPSILON));
        // "teh" requires transposition (cost 0.5 in typo_friendly)
        assert!(candidates.iter().any(|c| c.term == "teh" && (c.distance - 0.5).abs() < EPSILON));
    }

    #[test]
    fn test_standard_costs_equivalent_to_integer() {
        let dict = DoubleArrayTrie::from_terms(vec!["cat", "car", "bat", "bar"]);
        let costs = OperationCostsF64::standard();

        let query = CandidateIteratorF64::new(
            dict.root(),
            "cat".to_string(),
            1.0,
            Algorithm::Standard,
            costs,
        );

        let candidates: Vec<_> = query.collect();

        // With standard costs, should behave same as integer Levenshtein
        // cat -> cat: distance 0
        // cat -> car: distance 1 (substitute t->r)
        // cat -> bat: distance 1 (substitute c->b)
        // cat -> bar: distance 2 (too far)

        assert!(candidates.iter().any(|c| c.term == "cat"));
        assert!(candidates.iter().any(|c| c.term == "car"));
        assert!(candidates.iter().any(|c| c.term == "bat"));
        assert!(!candidates.iter().any(|c| c.term == "bar"));
    }

    #[test]
    fn test_empty_query() {
        let dict = DoubleArrayTrie::from_terms(vec!["a", "ab"]);
        let costs = OperationCostsF64::standard();

        let query = CandidateIteratorF64::new(
            dict.root(),
            "".to_string(),
            1.0,
            Algorithm::Standard,
            costs,
        );

        let candidates: Vec<_> = query.collect();
        // Empty query with max_cost 1.0 should match single-char words
        assert!(candidates.iter().any(|c| c.term == "a"));
        // "ab" requires 2 insertions, exceeds threshold
        assert!(!candidates.iter().any(|c| c.term == "ab"));
    }
}
