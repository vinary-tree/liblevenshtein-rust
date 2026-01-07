//! Value-filtered query iterators for context-aware fuzzy matching.
//!
//! This module implements query iterators that filter candidates based on their
//! associated values during result collection. The filter is evaluated before
//! materializing term strings, which can improve performance when many results
//! match the distance threshold but few match the value filter.

use libdictenstein::value::DictionaryValue;
use libdictenstein::{CharUnit, MappedDictionaryNode};
use crate::transducer::intersection::PathNode;
use crate::transducer::transition::{initial_state, transition_state_pooled};
use crate::transducer::{Algorithm, Candidate, Intersection, StatePool, SubstitutionPolicyFor, Unrestricted};
use std::collections::{HashSet, VecDeque};

/// Iterator that yields candidates filtered by their associated values.
///
/// This iterator evaluates the value predicate during result collection, before
/// materializing term strings. This can improve performance when the selectivity
/// is high (>50% of candidates pass the filter) by avoiding string allocations.
///
/// # Type Parameters
///
/// - `N`: The dictionary node type (must support value access)
/// - `F`: The filter predicate function
///
/// # Performance Characteristics
///
/// - **Graph traversal**: Same as unfiltered queries (no pruning)
/// - **Predicate evaluation**: Checked on every final node encountered
/// - **String materialization**: Only for candidates that pass the filter
///
/// **When to use**:
/// - High selectivity (>50% of candidates pass filter): saves string allocations
/// - Set-based filters: use `query_by_value_set()` for optimized HashSet lookups
///
/// **When NOT to use**:
/// - Low selectivity (<50%): predicate overhead exceeds materialization savings
/// - Simple filters: post-filtering with `.filter()` is often faster due to lazy evaluation
///
/// # Comparison to Post-Filtering
///
/// ```text
/// | Approach         | Traversal | Predicate Calls    | String Allocations |
/// |------------------|-----------|--------------------|--------------------|
/// | Value-filtered   | Full      | All final nodes    | Only matches       |
/// | Post-filtered    | Full      | Only consumed      | All finals (lazy)  |
/// ```
///
/// For most use cases, post-filtering is recommended:
/// ```rust,ignore
/// transducer.query("term", 2)
///     .filter(|term| dict.get_value(term) == Some(target_scope))
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
///
/// // Dictionary with scope IDs
/// let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values(vec![
///     ("println", 1),   // std scope
///     ("my_func", 2),   // local scope
/// ]);
///
/// let transducer = Transducer::new(dict, Algorithm::Standard);
///
/// // Query with scope filter (only local scope)
/// let matches: Vec<_> = transducer
///     .query_filtered("my", 2, |scope_id| *scope_id == 2)
///     .collect();
/// ```
pub struct ValueFilteredQueryIterator<N, F>
where
    N: MappedDictionaryNode,
    F: Fn(&N::Value) -> bool,
{
    /// Query units (bytes or chars)
    query: Vec<N::Unit>,
    /// Maximum edit distance
    max_distance: usize,
    /// Algorithm (Standard or Transposition)
    algorithm: Algorithm,
    /// Value filter predicate
    filter: F,
    /// Queue of pending intersections to explore
    pending: VecDeque<Box<Intersection<N>>>,
    /// Set of seen terms (for deduplication)
    seen: HashSet<String>,
    /// State pool for efficient state allocation reuse
    state_pool: StatePool,
    /// Iterator finished flag
    finished: bool,
}

impl<N, F> ValueFilteredQueryIterator<N, F>
where
    N: MappedDictionaryNode,
    F: Fn(&N::Value) -> bool,
{
    /// Create a new value-filtered query iterator.
    ///
    /// # Parameters
    ///
    /// - `root`: Root node of the dictionary
    /// - `term`: Query term to match against
    /// - `max_distance`: Maximum edit distance
    /// - `algorithm`: Levenshtein algorithm variant
    /// - `filter`: Predicate to test values (return true to include)
    pub fn new(
        root: N,
        term: String,
        max_distance: usize,
        algorithm: Algorithm,
        filter: F,
    ) -> Self {
        let query_units = N::Unit::from_str(&term);
        let initial = initial_state(query_units.len(), max_distance, algorithm);

        let mut pending = VecDeque::new();
        pending.push_back(Box::new(Intersection::new(root, initial)));

        Self {
            query: query_units,
            max_distance,
            algorithm,
            filter,
            pending,
            seen: HashSet::new(),
            state_pool: StatePool::new(),
            finished: false,
        }
    }
}

impl<N, F> Iterator for ValueFilteredQueryIterator<N, F>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    F: Fn(&N::Value) -> bool,
{
    type Item = Candidate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        while let Some(intersection) = self.pending.pop_front() {
            // Check if this is a final node
            if intersection.is_final() {
                // Infer distance (standard mode - penalize remaining query characters)
                let distance = intersection
                    .state
                    .infer_distance(self.query.len())
                    .unwrap_or(usize::MAX);

                if distance <= self.max_distance {
                    // CRITICAL: Check value filter BEFORE materializing term
                    if let Some(value) = intersection.node.value() {
                        if !(self.filter)(&value) {
                            // Value doesn't match filter - queue children but skip this match
                            self.queue_children(&intersection);
                            continue;
                        }
                    }

                    // Materialize term string
                    let term = intersection.term();

                    // Deduplicate
                    if self.seen.insert(term.clone()) {
                        // Queue children for further exploration
                        self.queue_children(&intersection);

                        // Return the candidate
                        return Some(Candidate { term, distance });
                    }
                } else {
                    // Even if this final node is too far, still explore its children
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
}

impl<N, F> ValueFilteredQueryIterator<N, F>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    F: Fn(&N::Value) -> bool,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
{
    /// Queue child intersections for exploration
    #[inline]
    fn queue_children(&mut self, intersection: &Intersection<N>) {
        for (label, child_node) in intersection.node.edges() {
            if let Some(next_state) = transition_state_pooled(
                &intersection.state,
                &mut self.state_pool,
                Unrestricted, // Default policy: allow all substitutions
                label,
                &self.query,
                self.max_distance,
                self.algorithm,
                false, // standard mode (not substring mode)
            ) {
                // Create lightweight PathNode
                let parent_path = intersection.label.map(|current_label| {
                    Box::new(PathNode::new(current_label, intersection.parent.clone()))
                });

                let child = Box::new(Intersection::with_parent(
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

/// Iterator that yields candidates filtered by value set membership.
///
/// Optimized for checking if a value is in a set (e.g., hierarchical scope visibility).
/// Uses efficient HashSet lookups for the predicate check.
///
/// # Performance
///
/// - HashSet membership: O(1) average case
/// - No overhead scaling with set size
/// - Beneficial for high-selectivity filters (>50% match rate)
///
/// For low-selectivity filters, consider post-filtering instead.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashSet;
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
///
/// let dict: PathMapDictionary<u32> = /* ... */;
/// let transducer = Transducer::new(dict, Algorithm::Standard);
///
/// // Query for terms in scopes 1, 2, or 3 (hierarchical scope visibility)
/// let visible_scopes: HashSet<u32> = [1, 2, 3].iter().cloned().collect();
/// let matches: Vec<_> = transducer
///     .query_by_value_set("my_func", 2, &visible_scopes)
///     .collect();
/// ```
pub struct ValueSetFilteredQueryIterator<N, V>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Eq + std::hash::Hash,
{
    /// Query units (bytes or chars)
    query: Vec<N::Unit>,
    /// Maximum edit distance
    max_distance: usize,
    /// Algorithm (Standard or Transposition)
    algorithm: Algorithm,
    /// Value set for membership testing
    value_set: HashSet<V>,
    /// Queue of pending intersections to explore
    pending: VecDeque<Box<Intersection<N>>>,
    /// Set of seen terms (for deduplication)
    seen: HashSet<String>,
    /// State pool for efficient state allocation reuse
    state_pool: StatePool,
    /// Iterator finished flag
    finished: bool,
}

impl<N, V> ValueSetFilteredQueryIterator<N, V>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Eq + std::hash::Hash + Clone,
{
    /// Create a new value-set-filtered query iterator.
    ///
    /// Only returns candidates whose values are in the provided set.
    pub fn new(
        root: N,
        term: String,
        max_distance: usize,
        algorithm: Algorithm,
        value_set: HashSet<V>,
    ) -> Self {
        let query_units = N::Unit::from_str(&term);
        let initial = initial_state(query_units.len(), max_distance, algorithm);

        let mut pending = VecDeque::new();
        pending.push_back(Box::new(Intersection::new(root, initial)));

        Self {
            query: query_units,
            max_distance,
            algorithm,
            value_set,
            pending,
            seen: HashSet::new(),
            state_pool: StatePool::new(),
            finished: false,
        }
    }
}

impl<N, V> Iterator for ValueSetFilteredQueryIterator<N, V>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Eq + std::hash::Hash,
{
    type Item = Candidate;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        while let Some(intersection) = self.pending.pop_front() {
            // Check if this is a final node
            if intersection.is_final() {
                // Infer distance (standard mode - penalize remaining query characters)
                let distance = intersection
                    .state
                    .infer_distance(self.query.len())
                    .unwrap_or(usize::MAX);

                if distance <= self.max_distance {
                    // CRITICAL: Check value set membership BEFORE materializing term
                    if let Some(value) = intersection.node.value() {
                        if !self.value_set.contains(&value) {
                            // Value not in set - queue children but skip this match
                            self.queue_children(&intersection);
                            continue;
                        }
                    }

                    // Materialize term string
                    let term = intersection.term();

                    // Deduplicate
                    if self.seen.insert(term.clone()) {
                        // Queue children for further exploration
                        self.queue_children(&intersection);

                        // Return the candidate
                        return Some(Candidate { term, distance });
                    }
                } else {
                    // Even if this final node is too far, still explore its children
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
}

impl<N, V> ValueSetFilteredQueryIterator<N, V>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Eq + std::hash::Hash,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
{
    /// Queue child intersections for exploration
    #[inline]
    fn queue_children(&mut self, intersection: &Intersection<N>) {
        for (label, child_node) in intersection.node.edges() {
            if let Some(next_state) = transition_state_pooled(
                &intersection.state,
                &mut self.state_pool,
                Unrestricted, // Default policy: allow all substitutions
                label,
                &self.query,
                self.max_distance,
                self.algorithm,
                false, // standard mode (not substring mode)
            ) {
                // Create lightweight PathNode
                let parent_path = intersection.label.map(|current_label| {
                    Box::new(PathNode::new(current_label, intersection.parent.clone()))
                });

                let child = Box::new(Intersection::with_parent(
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

#[cfg(test)]
#[cfg(feature = "pathmap-backend")]
mod tests {
    use super::*;
    use libdictenstein::{pathmap::PathMapDictionary, Dictionary};
    use crate::transducer::Transducer;

    #[test]
    fn test_value_filtered_query_basic() {
        // Create dictionary with scope IDs
        let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values(vec![
            ("test", 1),
            ("tests", 1),
            ("testing", 2),
            ("tester", 1),
        ]);

        let transducer = Transducer::new(dict, Algorithm::Standard);

        // Query with filter for scope 1 only
        let root = transducer.dictionary().root();
        let iter = ValueFilteredQueryIterator::new(
            root,
            "test".to_string(),
            1,
            Algorithm::Standard,
            |scope_id| *scope_id == 1,
        );

        let mut results: Vec<_> = iter.collect();
        results.sort_by(|a, b| a.term.cmp(&b.term));

        // Should find "test" (distance 0) and "tests" (distance 1 - insert 's')
        // "testing" has wrong scope (2), "tester" is too far (distance 2)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].term, "test");
        assert_eq!(results[0].distance, 0);
        assert_eq!(results[1].term, "tests");
        assert_eq!(results[1].distance, 1);
    }

    #[test]
    fn test_value_set_filtered_query() {
        // Create dictionary with scope IDs
        let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values(vec![
            ("println", 1),   // std
            ("format", 1),    // std
            ("func", 2),      // local - exact match
            ("funcs", 2),     // local - distance 1
            ("my_func", 2),   // local - distance 3, out of range
            ("your_func", 3), // other - wrong scope
        ]);

        let transducer = Transducer::new(dict, Algorithm::Standard);

        // Query for scopes 1 and 2 (std + local, not other)
        let root = transducer.dictionary().root();
        let visible_scopes: HashSet<u32> = [1, 2].iter().cloned().collect();
        let iter = ValueSetFilteredQueryIterator::new(
            root,
            "func".to_string(),
            2,
            Algorithm::Standard,
            visible_scopes,
        );

        let mut results: Vec<_> = iter.collect();
        results.sort_by(|a, b| a.term.cmp(&b.term));

        // Should find "func" (scope 2, distance 0) and "funcs" (scope 2, distance 1 - insert 's')
        // "my_func" is out of range (distance 3), "your_func" is wrong scope (3)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].term, "func");
        assert_eq!(results[1].term, "funcs");
    }

    #[test]
    fn test_value_filtered_query_empty_result() {
        let dict: PathMapDictionary<u32> =
            PathMapDictionary::from_terms_with_values(vec![("test", 1), ("testing", 1)]);

        let transducer = Transducer::new(dict, Algorithm::Standard);

        // Query with filter for scope 2 (none exist)
        let root = transducer.dictionary().root();
        let iter = ValueFilteredQueryIterator::new(
            root,
            "test".to_string(),
            1,
            Algorithm::Standard,
            |scope_id| *scope_id == 2,
        );

        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 0);
    }
}
