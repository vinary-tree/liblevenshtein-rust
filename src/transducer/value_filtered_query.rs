//! Value-filtered query iterators for context-aware fuzzy matching.
//!
//! This module implements query iterators that filter candidates based on their
//! associated values during result collection. The filter is evaluated before
//! materializing term strings, which can improve performance when many results
//! match the distance threshold but few match the value filter.

use crate::transducer::dictionary_traversal::{
    CursorNativePath, MappedValueSource, ParentArenaPath, PathFrontier, ResultPathStrategy,
    TraversalCursor, TraversalSession,
};
use crate::transducer::transition::{
    with_prepared_unit_cost_row, FinishMode, PreparedUnitCostRow, TransitionSettings,
    UnitCostFrontier, UnitCostMachine,
};
use crate::transducer::{Algorithm, Candidate, StatePool, SubstitutionPolicyFor, Unrestricted};
use libdictenstein::value::DictionaryValue;
use libdictenstein::{CharUnit, DictionaryTraversalRoot, MappedDictionaryNode};
use std::collections::{HashSet, VecDeque};
use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "perf-instrumentation")]
use crate::transducer::dictionary_traversal::TraversalProductIdentity;
#[cfg(feature = "perf-instrumentation")]
use rustc_hash::FxHashSet;

/// The emitted term type for [`ValueYieldingQueryIterator`]: either the
/// reconstructed text (`String`) or the raw unit sequence (`Vec<U>`, lossless for
/// `u64` token dictionaries whose `to_string` is a lossy byte-unpack).
pub trait ValueTerm<U: CharUnit>: Sized {
    /// Build the term from the matched unit sequence.
    fn from_units(units: &[U]) -> Self;
}

impl<U: CharUnit> ValueTerm<U> for String {
    #[inline]
    fn from_units(units: &[U]) -> Self {
        U::to_string(units)
    }
}

impl<U: CharUnit> ValueTerm<U> for Vec<U> {
    #[inline]
    fn from_units(units: &[U]) -> Self {
        units.to_vec()
    }
}

fn seeded_transitions<U: CharUnit>(
    query: &[U],
    max_distance: usize,
    algorithm: Algorithm,
) -> (UnitCostMachine<U>, UnitCostFrontier) {
    let settings = TransitionSettings::new(max_distance, algorithm, false);
    UnitCostMachine::seeded::<Unrestricted>(query, settings)
}

struct ValueTransitionSettings<'a, U> {
    query: &'a [U],
    max_distance: usize,
    algorithm: Algorithm,
}

struct ValueTransitionStep {
    state: UnitCostFrontier,
    #[cfg(feature = "perf-instrumentation")]
    position_count: usize,
    #[cfg(feature = "perf-instrumentation")]
    storage_bytes: usize,
}

#[inline]
fn queue_value_children_with_prepared_row<N, S, R>(
    intersection: &PathFrontier<S::Trace, UnitCostFrontier>,
    traversal: &mut TraversalSession<N>,
    pending: &mut VecDeque<PathFrontier<S::Trace, UnitCostFrontier>>,
    path_storage: &mut S::Storage,
    row: &mut R,
) -> Option<MappedValueSource<N>>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
    S: ResultPathStrategy<N>,
    R: PreparedUnitCostRow<N::Unit>,
{
    let mut expansion = S::begin_expansion(&intersection.trace);
    traversal.filter_map_edges_and_final_source(
        S::position(&intersection.trace),
        |label| {
            crate::causal_perf::record_edges_enumerated(1);
            row.step(label).map(|state| ValueTransitionStep {
                state,
                #[cfg(feature = "perf-instrumentation")]
                position_count: row.active_len(state),
                #[cfg(feature = "perf-instrumentation")]
                storage_bytes: row.frontier_storage_bytes(state),
            })
        },
        |label, child_position, step| {
            crate::causal_perf::record_transition_accepted(1);
            #[cfg(feature = "perf-instrumentation")]
            {
                crate::causal_perf::record_state_positions_enqueued(step.position_count as u64);
                crate::causal_perf::record_state_bytes_enqueued(step.storage_bytes as u64);
            }
            pending.push_back(PathFrontier::new(
                S::child_trace(
                    &intersection.trace,
                    &mut expansion,
                    label,
                    child_position,
                    path_storage,
                ),
                step.state,
            ));
            crate::causal_perf::record_pending_queue_size(pending.len());
        },
    )
}

#[inline]
fn queue_value_children_and_finality<N, S>(
    intersection: &PathFrontier<S::Trace, UnitCostFrontier>,
    traversal: &mut TraversalSession<N>,
    pending: &mut VecDeque<PathFrontier<S::Trace, UnitCostFrontier>>,
    path_storage: &mut S::Storage,
    state_pool: &mut StatePool,
    unit_transitions: &mut UnitCostMachine<N::Unit>,
    settings: ValueTransitionSettings<'_, N::Unit>,
) -> Option<MappedValueSource<N>>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
    S: ResultPathStrategy<N>,
{
    let transition_settings =
        TransitionSettings::new(settings.max_distance, settings.algorithm, false);
    let policy = Unrestricted;
    with_prepared_unit_cost_row!(
        unit_transitions,
        intersection.frontier,
        state_pool,
        &policy,
        settings.query,
        transition_settings,
        |row| queue_value_children_with_prepared_row::<N, S, _>(
            intersection,
            traversal,
            pending,
            path_storage,
            &mut row,
        )
    )
}

enum PathValueTraversal<N: MappedDictionaryNode> {
    Parent(ValueTraversalCore<N, ParentArenaPath>),
    Cursor(ValueTraversalCore<N, CursorNativePath>),
}

struct ValueTraversalCore<N, S>
where
    N: MappedDictionaryNode,
    S: ResultPathStrategy<N>,
{
    query: Vec<N::Unit>,
    max_distance: usize,
    algorithm: Algorithm,
    pending: VecDeque<PathFrontier<S::Trace, UnitCostFrontier>>,
    traversal: TraversalSession<N>,
    path_storage: S::Storage,
    state_pool: StatePool,
    unit_transitions: UnitCostMachine<N::Unit>,
    finished: bool,
    #[cfg(feature = "perf-instrumentation")]
    generated_products: FxHashSet<(TraversalProductIdentity, UnitCostFrontier)>,
}

impl<N> PathValueTraversal<N>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
{
    fn new(
        root: DictionaryTraversalRoot<N>,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Self {
        let (unit_transitions, initial) = seeded_transitions(&query, max_distance, algorithm);
        let (traversal, root) = TraversalSession::capture_mapped(root);
        if traversal.supports_cursor_key_units() {
            Self::Cursor(ValueTraversalCore::new(
                root,
                traversal,
                query,
                max_distance,
                algorithm,
                unit_transitions,
                initial,
            ))
        } else {
            Self::Parent(ValueTraversalCore::new(
                root,
                traversal,
                query,
                max_distance,
                algorithm,
                unit_transitions,
                initial,
            ))
        }
    }

    #[inline]
    fn advance<R, A, B>(&mut self, accept: A, build: B) -> Option<R>
    where
        A: Fn(&N::Value) -> bool,
        B: Fn(Vec<N::Unit>, usize, N::Value) -> R,
    {
        match self {
            Self::Parent(core) => core.advance(accept, build),
            Self::Cursor(core) => core.advance(accept, build),
        }
    }
}

impl<N, S> ValueTraversalCore<N, S>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
    S: ResultPathStrategy<N>,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        root: TraversalCursor<N::SnapshotCursor>,
        traversal: TraversalSession<N>,
        query: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
        unit_transitions: UnitCostMachine<N::Unit>,
        initial: UnitCostFrontier,
    ) -> Self {
        let (mut pending, path_storage) = S::acquire_queue();
        pending.push_back(PathFrontier::new(S::root(root), initial));
        Self {
            query,
            max_distance,
            algorithm,
            pending,
            traversal,
            path_storage,
            state_pool: StatePool::new(),
            unit_transitions,
            finished: false,
            #[cfg(feature = "perf-instrumentation")]
            generated_products: FxHashSet::default(),
        }
    }

    #[inline]
    fn advance<R, A, B>(&mut self, accept: A, build: B) -> Option<R>
    where
        A: Fn(&N::Value) -> bool,
        B: Fn(Vec<N::Unit>, usize, N::Value) -> R,
    {
        if self.finished {
            return None;
        }

        while let Some(intersection) = self.pending.pop_front() {
            crate::causal_perf::record_dictionary_intersections(1);
            crate::causal_perf::record_final_checks(1);
            crate::causal_perf::record_generated_product_expansions(1);
            #[cfg(feature = "perf-instrumentation")]
            if let Some(node) = self
                .traversal
                .product_identity(S::position(&intersection.trace))
            {
                crate::causal_perf::record_generated_product_identity_expansions(1);
                if self
                    .generated_products
                    .insert((node, intersection.frontier))
                {
                    crate::causal_perf::record_generated_product_unique_expansions(1);
                } else {
                    crate::causal_perf::record_generated_product_repeated_expansions(1);
                }
            }

            let final_source = queue_value_children_and_finality::<N, S>(
                &intersection,
                &mut self.traversal,
                &mut self.pending,
                &mut self.path_storage,
                &mut self.state_pool,
                &mut self.unit_transitions,
                ValueTransitionSettings {
                    query: &self.query,
                    max_distance: self.max_distance,
                    algorithm: self.algorithm,
                },
            );
            let Some(final_source) = final_source else {
                continue;
            };
            let distance = self
                .unit_transitions
                .finish_distance(
                    intersection.frontier,
                    FinishMode::Complete,
                    self.query.len(),
                )
                .unwrap_or(usize::MAX);
            if distance > self.max_distance {
                continue;
            }

            let mut final_units = self.traversal.requires_final_units().then(|| {
                S::materialize_units(&intersection.trace, &self.traversal, &self.path_storage)
            });
            if final_units
                .as_deref()
                .is_some_and(|units| !self.traversal.accepts_final_units(units))
            {
                continue;
            }
            let Some(value) = self
                .traversal
                .resolve_final_value(final_source, final_units.as_deref())
            else {
                continue;
            };
            if !accept(&value) {
                continue;
            }
            let units = final_units.take().unwrap_or_else(|| {
                S::materialize_units(&intersection.trace, &self.traversal, &self.path_storage)
            });
            crate::causal_perf::record_matches_materialized(1);
            return Some(build(units, distance, value));
        }

        self.finished = true;
        None
    }
}

impl<N, S> Drop for ValueTraversalCore<N, S>
where
    N: MappedDictionaryNode,
    S: ResultPathStrategy<N>,
{
    fn drop(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        let path_storage = std::mem::take(&mut self.path_storage);
        S::release_queue(pending, path_storage);
    }
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
/// ```rust
/// # #[cfg(feature = "pathmap-backend")]
/// # {
/// use liblevenshtein::dictionary::MappedDictionary;
/// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
/// use liblevenshtein::prelude::*;
///
/// let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values([
///     ("println", 1),
///     ("my_func", 2),
/// ]);
/// let transducer = Transducer::standard(dict);
/// let target_scope = 2;
/// let matches: Vec<_> = transducer
///     .query_terms("my", 5)
///     .filter(|term| transducer.dictionary().get_value(term) == Some(target_scope))
///     .collect();
/// assert_eq!(matches, ["my_func"]);
/// # }
/// ```
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "pathmap-backend")]
/// # {
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
/// # }
/// ```
pub struct ValueFilteredQueryIterator<N, F>
where
    N: MappedDictionaryNode,
    F: Fn(&N::Value) -> bool,
{
    /// Value filter predicate
    filter: F,
    inner: PathValueTraversal<N>,
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
        Self::with_traversal_root(
            DictionaryTraversalRoot::owned(root),
            term,
            max_distance,
            algorithm,
            filter,
        )
    }

    pub(crate) fn with_traversal_root(
        root: DictionaryTraversalRoot<N>,
        term: String,
        max_distance: usize,
        algorithm: Algorithm,
        filter: F,
    ) -> Self {
        let query_units = N::Unit::from_str(&term);
        Self {
            filter,
            inner: PathValueTraversal::new(root, query_units, max_distance, algorithm),
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
        let filter = &self.filter;
        self.inner.advance(
            |value| filter(value),
            |units, distance, _value| Candidate {
                term: N::Unit::to_string(&units),
                distance,
            },
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<N, F> std::iter::FusedIterator for ValueFilteredQueryIterator<N, F>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    F: Fn(&N::Value) -> bool,
{
}

/// Iterator that yields `(term, distance, value)` for every match within the
/// edit-distance threshold, reading each final node's value *during* traversal
/// so the caller needs no second dictionary lookup. Final nodes whose value is
/// `None` are skipped (they terminate no stored entry).
pub struct ValueYieldingQueryIterator<N, T = String>
where
    N: MappedDictionaryNode,
{
    inner: PathValueTraversal<N>,
    /// Selects text versus native-unit results without runtime storage.
    _term_type: PhantomData<T>,
}

impl<N> ValueYieldingQueryIterator<N, String>
where
    N: MappedDictionaryNode,
{
    /// Create a new value-yielding query iterator over a string query.
    ///
    /// Yields `(term: String, distance, value)`. For a `u64` token dictionary use
    /// [`with_unit_query`](ValueYieldingQueryIterator::with_unit_query) instead —
    /// `from_str` byte-packs the query.
    pub fn new(root: N, term: String, max_distance: usize, algorithm: Algorithm) -> Self {
        let query_units = N::Unit::from_str(&term);
        Self::with_units(
            DictionaryTraversalRoot::owned(root),
            query_units,
            max_distance,
            algorithm,
        )
    }

    pub(crate) fn with_traversal_root(
        root: DictionaryTraversalRoot<N>,
        term: String,
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Self {
        let query_units = N::Unit::from_str(&term);
        Self::with_units(root, query_units, max_distance, algorithm)
    }
}

impl<N> ValueYieldingQueryIterator<N, Vec<N::Unit>>
where
    N: MappedDictionaryNode,
{
    /// Create a value-yielding query iterator over a raw unit sequence, yielding
    /// `(term: Vec<Unit>, distance, value)` — the units-native, lossless variant
    /// required for `u64` token-id dictionaries.
    pub fn with_unit_query(
        root: N,
        query_units: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Self {
        Self::with_units(
            DictionaryTraversalRoot::owned(root),
            query_units,
            max_distance,
            algorithm,
        )
    }

    pub(crate) fn with_unit_query_traversal_root(
        root: DictionaryTraversalRoot<N>,
        query_units: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Self {
        Self::with_units(root, query_units, max_distance, algorithm)
    }
}

impl<N, T> ValueYieldingQueryIterator<N, T>
where
    N: MappedDictionaryNode,
{
    /// Shared constructor from a pre-computed unit query. The emitted term type `T`
    /// (`String` or `Vec<Unit>`) is chosen by the caller / return type.
    fn with_units(
        root: DictionaryTraversalRoot<N>,
        query_units: Vec<N::Unit>,
        max_distance: usize,
        algorithm: Algorithm,
    ) -> Self {
        Self {
            inner: PathValueTraversal::new(root, query_units, max_distance, algorithm),
            _term_type: PhantomData,
        }
    }
}

impl<N, T> Iterator for ValueYieldingQueryIterator<N, T>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    T: ValueTerm<N::Unit>,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
{
    type Item = (T, usize, N::Value);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.advance(
            |_| true,
            |units, distance, value| (T::from_units(&units), distance, value),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<N, T> std::iter::FusedIterator for ValueYieldingQueryIterator<N, T>
where
    N: MappedDictionaryNode,
    N::Value: DictionaryValue,
    T: ValueTerm<N::Unit>,
    Unrestricted: SubstitutionPolicyFor<N::Unit>,
{
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
/// ```rust
/// # #[cfg(feature = "pathmap-backend")]
/// # {
/// use std::collections::HashSet;
/// use liblevenshtein::prelude::*;
/// use liblevenshtein::dictionary::pathmap::PathMapDictionary;
///
/// let dict: PathMapDictionary<u32> = PathMapDictionary::from_terms_with_values([
///     ("println", 1),
///     ("my_func", 2),
///     ("private_helper", 9),
/// ]);
/// let transducer = Transducer::new(dict, Algorithm::Standard);
///
/// // Query for terms in scopes 1, 2, or 3 (hierarchical scope visibility)
/// let visible_scopes: HashSet<u32> = [1, 2, 3].iter().cloned().collect();
/// let matches: Vec<_> = transducer
///     .query_by_value_set("my_func", 2, &visible_scopes)
///     .collect();
/// assert!(matches.iter().any(|candidate| candidate.term == "my_func"));
/// assert!(matches.iter().all(|candidate| candidate.term != "private_helper"));
/// # }
/// ```
pub struct ValueSetFilteredQueryIterator<'a, N, V>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Eq + Hash,
{
    /// Value set for membership testing
    value_set: ValueSetMembership<'a, V>,
    inner: PathValueTraversal<N>,
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
            DictionaryTraversalRoot::owned(root),
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
            DictionaryTraversalRoot::owned(root),
            term,
            max_distance,
            algorithm,
            ValueSetMembership::Borrowed(value_set),
        )
    }

    pub(crate) fn new_borrowed_traversal_root(
        root: DictionaryTraversalRoot<N>,
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
        root: DictionaryTraversalRoot<N>,
        term: String,
        max_distance: usize,
        algorithm: Algorithm,
        value_set: ValueSetMembership<'a, V>,
    ) -> Self {
        let query_units = N::Unit::from_str(&term);
        Self {
            value_set,
            inner: PathValueTraversal::new(root, query_units, max_distance, algorithm),
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
        let value_set = &self.value_set;
        self.inner.advance(
            |value| value_set.contains(value),
            |units, distance, _value| Candidate {
                term: N::Unit::to_string(&units),
                distance,
            },
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<N, V> std::iter::FusedIterator for ValueSetFilteredQueryIterator<'_, N, V>
where
    N: MappedDictionaryNode<Value = V>,
    V: DictionaryValue + Eq + Hash,
{
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
    use crate::transducer::dictionary_traversal::{CursorPathTrace, ParentPathTrace, PathFrontier};
    use crate::transducer::transition::UnitCostFrontier;
    use crate::transducer::{Algorithm, Candidate, Transducer};
    use libdictenstein::double_array_trie::{DoubleArrayTrie, DoubleArrayTrieBuilder};
    use libdictenstein::SnapshotTraversalCursor;
    use std::collections::HashSet;

    fn sorted(mut v: Vec<(String, usize, u32)>) -> Vec<(String, usize, u32)> {
        v.sort();
        v
    }

    #[test]
    fn queued_value_intersection_is_compact() {
        assert!(
            std::mem::size_of::<
                PathFrontier<ParentPathTrace<char, SnapshotTraversalCursor>, UnitCostFrontier>,
            >() <= 32
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            std::mem::size_of::<
                PathFrontier<CursorPathTrace<SnapshotTraversalCursor>, UnitCostFrontier>,
            >(),
            16
        );
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
        // deterministic dictionary traversal still has one path per term.
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
