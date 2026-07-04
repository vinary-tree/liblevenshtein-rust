//! Value-filtered query iterators for context-aware fuzzy matching.
//!
//! This module implements query iterators that filter candidates based on their
//! associated values during result collection. The filter is evaluated before
//! materializing term strings, which can improve performance when many results
//! match the distance threshold but few match the value filter.

use crate::transducer::transition::{
    initial_state, transition_state_pooled_ref, TransitionSettings,
};
use crate::transducer::{
    Algorithm, Candidate, State, StatePool, SubstitutionPolicyFor, Unrestricted,
};
use libdictenstein::value::DictionaryValue;
use libdictenstein::{CharUnit, MappedDictionaryNode};
use rustc_hash::FxHashSet;
use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

type SeenTerms = FxHashSet<Box<str>>;

const NO_PATH: usize = usize::MAX;

struct ValueQueryPathNode<U: CharUnit> {
    label: U,
    depth: usize,
    parent: usize,
}

struct ValueQueryIntersection<N: MappedDictionaryNode> {
    label: Option<N::Unit>,
    node: N,
    state: State,
    parent: usize,
}

impl<N: MappedDictionaryNode> ValueQueryIntersection<N> {
    #[inline]
    fn new(node: N, state: State) -> Self {
        Self {
            label: None,
            node,
            state,
            parent: NO_PATH,
        }
    }

    #[inline]
    fn with_parent(label: N::Unit, node: N, state: State, parent: usize) -> Self {
        Self {
            label: Some(label),
            node,
            state,
            parent,
        }
    }

    fn term(&self, path_arena: &[ValueQueryPathNode<N::Unit>]) -> String {
        let parent_depth = if self.parent == NO_PATH {
            0
        } else {
            path_arena[self.parent].depth
        };
        let capacity = parent_depth + usize::from(self.label.is_some());
        let mut units = Vec::with_capacity(capacity);

        if let Some(label) = self.label {
            units.push(label);
        }

        let mut current = self.parent;
        while current != NO_PATH {
            let node = &path_arena[current];
            units.push(node.label);
            current = node.parent;
        }

        units.reverse();
        N::Unit::to_string(&units)
    }

    #[inline(always)]
    fn is_final(&self) -> bool {
        self.node.is_final()
    }
}

#[inline]
fn push_path_node<U: CharUnit>(
    path_arena: &mut Vec<ValueQueryPathNode<U>>,
    label: U,
    parent: usize,
) -> usize {
    let depth = if parent == NO_PATH {
        1
    } else {
        path_arena[parent].depth.saturating_add(1)
    };
    let index = path_arena.len();
    path_arena.push(ValueQueryPathNode {
        label,
        depth,
        parent,
    });
    index
}

#[inline]
fn mark_seen_term(seen: &mut SeenTerms, term: &str) -> bool {
    if seen.contains(term) {
        return false;
    }
    seen.insert(term.to_owned().into_boxed_str());
    true
}

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
    pending: VecDeque<ValueQueryIntersection<N>>,
    /// Arena of shared parent path nodes used to reconstruct yielded terms.
    path_arena: Vec<ValueQueryPathNode<N::Unit>>,
    /// Set of seen terms (for deduplication)
    seen: SeenTerms,
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

        let mut pending = VecDeque::with_capacity(1);
        pending.push_back(ValueQueryIntersection::new(root, initial));

        Self {
            query: query_units,
            max_distance,
            algorithm,
            filter,
            pending,
            path_arena: Vec::with_capacity(64),
            seen: SeenTerms::default(),
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
                    // CRITICAL: Check value filter BEFORE materializing term.
                    let Some(value) = intersection.node.value() else {
                        self.queue_children(&intersection);
                        continue;
                    };

                    if !(self.filter)(&value) {
                        // Value doesn't match filter - queue children but skip this match.
                        self.queue_children(&intersection);
                        continue;
                    }

                    // Materialize term string
                    let term = intersection.term(&self.path_arena);

                    // Deduplicate
                    if mark_seen_term(&mut self.seen, &term) {
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
    fn queue_children(&mut self, intersection: &ValueQueryIntersection<N>) {
        let mut child_parent_path = None;

        for (label, child_node) in intersection.node.edges() {
            if let Some(next_state) = transition_state_pooled_ref(
                &intersection.state,
                &mut self.state_pool,
                &Unrestricted, // Default policy: allow all substitutions
                label,
                &self.query,
                TransitionSettings::new(
                    self.max_distance,
                    self.algorithm,
                    false, // standard mode (not substring mode)
                ),
            ) {
                let parent_path = match child_parent_path {
                    Some(path) => path,
                    None => {
                        let path = match intersection.label {
                            Some(current_label) => push_path_node(
                                &mut self.path_arena,
                                current_label,
                                intersection.parent,
                            ),
                            None => NO_PATH,
                        };
                        child_parent_path = Some(path);
                        path
                    }
                };

                let child =
                    ValueQueryIntersection::with_parent(label, child_node, next_state, parent_path);

                self.pending.push_back(child);
            }
        }
    }
}

/// Iterator that yields `(term, distance, value)` for every match within the
/// edit-distance threshold, reading each final node's value *during* traversal
/// so the caller needs no second dictionary lookup. Final nodes whose value is
/// `None` are skipped (they terminate no stored entry).
pub struct ValueYieldingQueryIterator<N>
where
    N: MappedDictionaryNode,
{
    /// Query units (bytes or chars)
    query: Vec<N::Unit>,
    /// Maximum edit distance
    max_distance: usize,
    /// Algorithm (Standard or Transposition)
    algorithm: Algorithm,
    /// Queue of pending intersections to explore
    pending: VecDeque<ValueQueryIntersection<N>>,
    /// Arena of shared parent path nodes used to reconstruct yielded terms.
    path_arena: Vec<ValueQueryPathNode<N::Unit>>,
    /// Set of seen terms (for deduplication)
    seen: SeenTerms,
    /// State pool for efficient state allocation reuse
    state_pool: StatePool,
    /// Iterator finished flag
    finished: bool,
}

impl<N> ValueYieldingQueryIterator<N>
where
    N: MappedDictionaryNode,
{
    /// Create a new value-yielding query iterator.
    pub fn new(root: N, term: String, max_distance: usize, algorithm: Algorithm) -> Self {
        let query_units = N::Unit::from_str(&term);
        let initial = initial_state(query_units.len(), max_distance, algorithm);

        let mut pending = VecDeque::with_capacity(1);
        pending.push_back(ValueQueryIntersection::new(root, initial));

        Self {
            query: query_units,
            max_distance,
            algorithm,
            pending,
            path_arena: Vec::with_capacity(64),
            seen: SeenTerms::default(),
            state_pool: StatePool::new(),
            finished: false,
        }
    }
}

impl<N> Iterator for ValueYieldingQueryIterator<N>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
{
    type Item = (String, usize, N::Value);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        while let Some(intersection) = self.pending.pop_front() {
            if intersection.is_final() {
                let distance = intersection
                    .state
                    .infer_distance(self.query.len())
                    .unwrap_or(usize::MAX);

                if distance <= self.max_distance {
                    let term = intersection.term(&self.path_arena);
                    // Deduplicate by term; children are queued exactly once,
                    // on the first visit of each term (matching the
                    // value-filtered iterator's discipline).
                    if mark_seen_term(&mut self.seen, &term) {
                        self.queue_children(&intersection);
                        // Read the value during traversal — no second lookup.
                        if let Some(value) = intersection.node.value() {
                            return Some((term, distance, value));
                        }
                        // Final but valueless: keep exploring.
                    }
                } else {
                    // Too far, but its children may still be in range.
                    self.queue_children(&intersection);
                }
            } else {
                self.queue_children(&intersection);
            }
        }

        self.finished = true;
        None
    }
}

impl<N> ValueYieldingQueryIterator<N>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
{
    /// Queue child intersections for exploration (identical traversal to the
    /// value-filtered iterator).
    #[inline]
    fn queue_children(&mut self, intersection: &ValueQueryIntersection<N>) {
        let mut child_parent_path = None;

        for (label, child_node) in intersection.node.edges() {
            if let Some(next_state) = transition_state_pooled_ref(
                &intersection.state,
                &mut self.state_pool,
                &Unrestricted,
                label,
                &self.query,
                TransitionSettings::new(self.max_distance, self.algorithm, false),
            ) {
                let parent_path = match child_parent_path {
                    Some(path) => path,
                    None => {
                        let path = match intersection.label {
                            Some(current_label) => push_path_node(
                                &mut self.path_arena,
                                current_label,
                                intersection.parent,
                            ),
                            None => NO_PATH,
                        };
                        child_parent_path = Some(path);
                        path
                    }
                };

                let child =
                    ValueQueryIntersection::with_parent(label, child_node, next_state, parent_path);

                self.pending.push_back(child);
            }
        }
    }
}

enum ValueSetMembership<'a, V> {
    Borrowed(&'a HashSet<V>),
    Owned(HashSet<V>),
}

impl<V> ValueSetMembership<'_, V>
where
    V: Eq + Hash,
{
    #[inline]
    fn contains(&self, value: &V) -> bool {
        match self {
            Self::Borrowed(set) => set.contains(value),
            Self::Owned(set) => set.contains(value),
        }
    }
}

/// Iterator that yields candidates filtered by value set membership.
///
/// Optimized for checking if a value is in a set (e.g., hierarchical scope visibility).
/// Uses efficient HashSet lookups for the predicate check and can either borrow
/// an existing set or own one supplied directly to [`Self::new`].
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
pub struct ValueSetFilteredQueryIterator<'a, N, V>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Eq + Hash,
{
    /// Query units (bytes or chars)
    query: Vec<N::Unit>,
    /// Maximum edit distance
    max_distance: usize,
    /// Algorithm (Standard or Transposition)
    algorithm: Algorithm,
    /// Value set for membership testing
    value_set: ValueSetMembership<'a, V>,
    /// Queue of pending intersections to explore
    pending: VecDeque<ValueQueryIntersection<N>>,
    /// Arena of shared parent path nodes used to reconstruct yielded terms.
    path_arena: Vec<ValueQueryPathNode<N::Unit>>,
    /// Set of seen terms (for deduplication)
    seen: SeenTerms,
    /// State pool for efficient state allocation reuse
    state_pool: StatePool,
    /// Iterator finished flag
    finished: bool,
}

impl<'a, N, V> ValueSetFilteredQueryIterator<'a, N, V>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Eq + Hash,
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
        Self::from_membership(
            root,
            term,
            max_distance,
            algorithm,
            ValueSetMembership::Owned(value_set),
        )
    }

    /// Create a new value-set-filtered query iterator borrowing the provided set.
    ///
    /// This avoids cloning large scope/value sets for one-shot queries.
    pub fn new_borrowed(
        root: N,
        term: String,
        max_distance: usize,
        algorithm: Algorithm,
        value_set: &'a HashSet<V>,
    ) -> Self {
        Self::from_membership(
            root,
            term,
            max_distance,
            algorithm,
            ValueSetMembership::Borrowed(value_set),
        )
    }

    fn from_membership(
        root: N,
        term: String,
        max_distance: usize,
        algorithm: Algorithm,
        value_set: ValueSetMembership<'a, V>,
    ) -> Self {
        let query_units = N::Unit::from_str(&term);
        let initial = initial_state(query_units.len(), max_distance, algorithm);

        let mut pending = VecDeque::with_capacity(1);
        pending.push_back(ValueQueryIntersection::new(root, initial));

        Self {
            query: query_units,
            max_distance,
            algorithm,
            value_set,
            pending,
            path_arena: Vec::with_capacity(64),
            seen: SeenTerms::default(),
            state_pool: StatePool::new(),
            finished: false,
        }
    }
}

impl<N, V> Iterator for ValueSetFilteredQueryIterator<'_, N, V>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Eq + Hash,
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
                    // CRITICAL: Check value set membership BEFORE materializing term.
                    let Some(value) = intersection.node.value() else {
                        self.queue_children(&intersection);
                        continue;
                    };

                    if !self.value_set.contains(&value) {
                        // Value not in set - queue children but skip this match.
                        self.queue_children(&intersection);
                        continue;
                    }

                    // Materialize term string
                    let term = intersection.term(&self.path_arena);

                    // Deduplicate
                    if mark_seen_term(&mut self.seen, &term) {
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

impl<N, V> ValueSetFilteredQueryIterator<'_, N, V>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Eq + Hash,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
{
    /// Queue child intersections for exploration
    #[inline]
    fn queue_children(&mut self, intersection: &ValueQueryIntersection<N>) {
        let mut child_parent_path = None;

        for (label, child_node) in intersection.node.edges() {
            if let Some(next_state) = transition_state_pooled_ref(
                &intersection.state,
                &mut self.state_pool,
                &Unrestricted, // Default policy: allow all substitutions
                label,
                &self.query,
                TransitionSettings::new(
                    self.max_distance,
                    self.algorithm,
                    false, // standard mode (not substring mode)
                ),
            ) {
                let parent_path = match child_parent_path {
                    Some(path) => path,
                    None => {
                        let path = match intersection.label {
                            Some(current_label) => push_path_node(
                                &mut self.path_arena,
                                current_label,
                                intersection.parent,
                            ),
                            None => NO_PATH,
                        };
                        child_parent_path = Some(path);
                        path
                    }
                };

                let child =
                    ValueQueryIntersection::with_parent(label, child_node, next_state, parent_path);

                self.pending.push_back(child);
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "pathmap-backend")]
mod tests {
    use super::*;
    use crate::transducer::Transducer;
    use libdictenstein::{pathmap::PathMapDictionary, Dictionary};

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
    fn test_value_set_filtered_query_borrowed_set() {
        let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values(vec![
            ("alpha", 1),
            ("alpine", 2),
            ("alt", 3),
        ]);
        let transducer = Transducer::new(dict, Algorithm::Standard);
        let root = transducer.dictionary().root();
        let visible_scopes: HashSet<u32> = [1, 2].iter().cloned().collect();
        let iter = ValueSetFilteredQueryIterator::new_borrowed(
            root,
            "alp".to_string(),
            3,
            Algorithm::Standard,
            &visible_scopes,
        );

        let mut results: Vec<_> = iter.collect();
        results.sort_by(|a, b| a.term.cmp(&b.term));

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].term, "alpha");
        assert_eq!(results[1].term, "alpine");
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

/// Tests for the value-yielding query iterator (`Transducer::query_values` /
/// [`ValueYieldingQueryIterator`]).
///
/// Unlike the sibling [`ValueFilteredQueryIterator`] tests above (which need the
/// `pathmap-backend` feature), these use [`DoubleArrayTrie`], which is always
/// compiled — so they run under the default `cargo test`. They pin the behavior
/// that is *new* relative to the value-filtered iterator: yielding
/// `(term, distance, value)` with the value read during traversal, and skipping
/// final nodes whose value is `None`.
#[cfg(test)]
mod value_yielding_tests {
    use super::{mark_seen_term, SeenTerms};
    use crate::transducer::{Algorithm, Candidate, Transducer};
    use libdictenstein::double_array_trie::{DoubleArrayTrie, DoubleArrayTrieBuilder};
    use std::collections::HashSet;

    fn sorted(mut v: Vec<(String, usize, u32)>) -> Vec<(String, usize, u32)> {
        v.sort();
        v
    }

    #[test]
    fn mark_seen_term_deduplicates_with_borrowed_lookup() {
        let mut seen = SeenTerms::default();

        assert!(mark_seen_term(&mut seen, "test"));
        assert!(!mark_seen_term(&mut seen, "test"));
        assert!(seen.contains("test"));
        assert_eq!(seen.len(), 1);
    }

    #[test]
    fn test_value_yielding_query_basic() {
        let dict: DoubleArrayTrie<u32> = DoubleArrayTrie::from_terms_with_values(vec![
            ("test", 1u32),
            ("tests", 1),
            ("testing", 2),
            ("tester", 1),
        ]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results = sorted(transducer.query_values("test", 1).collect());

        // "test" (dist 0) and "tests" (dist 1, insert 's'); "tester" (dist 2) and
        // "testing" (dist 3) are out of range. Values come back UNFILTERED — the
        // distinction from the value-filtered iterator.
        assert_eq!(
            results,
            vec![("test".to_string(), 0, 1u32), ("tests".to_string(), 1, 1)]
        );
    }

    #[test]
    fn test_value_yielding_query_distance_zero() {
        let dict: DoubleArrayTrie<u32> =
            DoubleArrayTrie::from_terms_with_values(vec![("test", 7u32), ("tests", 8)]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results = sorted(transducer.query_values("test", 0).collect());
        assert_eq!(results, vec![("test".to_string(), 0, 7u32)]);
    }

    #[test]
    fn test_value_query_iterators_reconstruct_term_past_initial_path_capacity() {
        let long_term = "a".repeat(96);
        let dict: DoubleArrayTrie<u32> =
            DoubleArrayTrie::from_terms_with_values(vec![(long_term.as_str(), 7u32)]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let filtered: Vec<_> = transducer
            .query_filtered(&long_term, 0, |value: &u32| *value == 7)
            .collect();
        assert_eq!(
            filtered,
            vec![Candidate {
                term: long_term.clone(),
                distance: 0,
            }]
        );

        let yielded = sorted(transducer.query_values(&long_term, 0).collect());
        assert_eq!(yielded, vec![(long_term.clone(), 0, 7u32)]);

        let visible_values: HashSet<u32> = HashSet::from([7]);
        let by_value_set: Vec<_> = transducer
            .query_by_value_set(&long_term, 0, &visible_values)
            .collect();
        assert_eq!(
            by_value_set,
            vec![Candidate {
                term: long_term,
                distance: 0,
            }]
        );
    }

    #[test]
    fn test_value_yielding_query_dedup() {
        // "abc" is reachable from query "abd" by substitution, "ab" by deletion;
        // the BFS may revisit a term via multiple states. Assert each term once.
        let dict: DoubleArrayTrie<u32> =
            DoubleArrayTrie::from_terms_with_values(vec![("ab", 7u32), ("abc", 8), ("abd", 9)]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<(String, usize, u32)> = transducer.query_values("abc", 2).collect();
        let unique: HashSet<&String> = results.iter().map(|(t, _, _)| t).collect();
        assert_eq!(
            unique.len(),
            results.len(),
            "no term may be yielded twice: {results:?}"
        );
    }

    #[test]
    fn test_value_yielding_query_empty_result() {
        let dict: DoubleArrayTrie<u32> =
            DoubleArrayTrie::from_terms_with_values(vec![("test", 1u32), ("testing", 1)]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let results: Vec<(String, usize, u32)> = transducer.query_values("zzzz", 1).collect();
        assert!(
            results.is_empty(),
            "no match within distance 1: {results:?}"
        );
    }

    #[test]
    fn test_value_yielding_skips_valueless_final() {
        // "a" is a FINAL node with NO value (inserted via insert_with_value(_, None));
        // "ab" is final WITH a value. `query_values` must skip "a" (valueless) yet
        // still descend through it to reach "ab" — proving children are queued
        // before the None short-circuit.
        let mut builder = DoubleArrayTrieBuilder::<u32>::new();
        builder.insert_with_value("a", None);
        builder.insert_with_value("ab", Some(100));
        let dict = builder.build();
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let yielded = sorted(transducer.query_values("a", 1).collect());
        assert_eq!(
            yielded,
            vec![("ab".to_string(), 1, 100u32)],
            "valueless final 'a' must be skipped; 'ab' must still be reached"
        );

        // Contrast: the plain query yields BOTH "a" and "ab" (it does not read values).
        let plain: HashSet<String> = transducer.query("a", 1).collect();
        assert!(
            plain.contains("a") && plain.contains("ab"),
            "plain query should yield both finals: {plain:?}"
        );
        // query_values set == plain set minus the valueless term.
        let valued: HashSet<String> = yielded.iter().map(|(t, _, _)| t.clone()).collect();
        let mut diff: Vec<String> = plain.difference(&valued).cloned().collect();
        diff.sort();
        assert_eq!(diff, vec!["a".to_string()]);
    }

    #[test]
    fn test_value_filtered_skips_valueless_final_but_descends() {
        let mut builder = DoubleArrayTrieBuilder::<u32>::new();
        builder.insert_with_value("a", None);
        builder.insert_with_value("ab", Some(100));
        builder.insert_with_value("ac", Some(200));
        let transducer = Transducer::new(builder.build(), Algorithm::Standard);

        let mut yielded: Vec<_> = transducer
            .query_filtered("a", 1, |value: &u32| *value == 100)
            .map(|candidate| (candidate.term, candidate.distance))
            .collect();
        yielded.sort();

        assert_eq!(yielded, vec![("ab".to_string(), 1)]);
    }

    #[test]
    fn test_value_set_filtered_skips_valueless_final_but_descends() {
        let mut builder = DoubleArrayTrieBuilder::<u32>::new();
        builder.insert_with_value("a", None);
        builder.insert_with_value("ab", Some(100));
        builder.insert_with_value("ac", Some(200));
        let transducer = Transducer::new(builder.build(), Algorithm::Standard);
        let visible_values: HashSet<u32> = HashSet::from([100]);

        let mut yielded: Vec<_> = transducer
            .query_by_value_set("a", 1, &visible_values)
            .map(|candidate| (candidate.term, candidate.distance))
            .collect();
        yielded.sort();

        assert_eq!(yielded, vec![("ab".to_string(), 1)]);
    }

    #[test]
    fn test_value_iterators_shared_prefix_branching_reconstruct_terms() {
        let dict: DoubleArrayTrie<u32> =
            DoubleArrayTrie::from_terms_with_values(vec![("cart", 1u32), ("care", 2), ("cars", 3)]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let mut filtered: Vec<_> = transducer
            .query_filtered("car", 1, |value: &u32| *value >= 1)
            .map(|candidate| candidate.term)
            .collect();
        filtered.sort();
        assert_eq!(filtered, vec!["care", "cars", "cart"]);

        let mut yielded: Vec<_> = transducer.query_values("car", 1).collect();
        yielded.sort();
        assert_eq!(
            yielded,
            vec![
                ("care".to_string(), 1, 2),
                ("cars".to_string(), 1, 3),
                ("cart".to_string(), 1, 1),
            ]
        );

        let visible_values: HashSet<u32> = HashSet::from([1, 2, 3]);
        let mut by_value_set: Vec<_> = transducer
            .query_by_value_set("car", 1, &visible_values)
            .map(|candidate| candidate.term)
            .collect();
        by_value_set.sort();
        assert_eq!(by_value_set, vec!["care", "cars", "cart"]);
    }

    #[test]
    fn test_value_yielding_matches_get_value() {
        let dict: DoubleArrayTrie<u32> = DoubleArrayTrie::from_terms_with_values(vec![
            ("test", 11u32),
            ("tests", 22),
            ("tester", 33),
        ]);
        let transducer = Transducer::new(dict, Algorithm::Standard);

        let mut count = 0;
        for (term, _dist, value) in transducer.query_values("test", 2) {
            assert_eq!(
                Some(value),
                transducer.dictionary().get_value(&term),
                "yielded value for {term:?} must equal a fresh dictionary lookup"
            );
            count += 1;
        }
        assert!(count > 0, "expected at least one match to validate");
    }
}
