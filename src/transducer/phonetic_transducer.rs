//! Phonetic transducer combining NFA-based phonetic matching with dictionary lookups.
//!
//! This module provides [`PhoneticTransducer`] which integrates phonetic NFA patterns
//! with dictionaries for combined phonetic + edit distance matching (fuzzy regex).
//!
//! # Architecture
//!
//! The phonetic transducer composes two automata:
//!
//! 1. **Phonetic NFA**: Handles sound-based variations (`ph ↔ f`, `c → s / _[ei]`)
//! 2. **Dictionary traversal**: Efficiently explores the dictionary
//!
//! The result is a fuzzy regex that finds dictionary terms matching a phonetic
//! pattern within an edit distance threshold.
//!
//! # Examples
//!
//! ```rust
//! use libdictenstein::double_array_trie::DoubleArrayTrieChar;
//! use liblevenshtein::transducer::PhoneticTransducerChar;
//! use liblevenshtein::phonetic::nfa::compile;
//! use liblevenshtein::phonetic::regex::parse;
//!
//! // Build dictionary
//! let dict = DoubleArrayTrieChar::from_terms(["phone", "phones", "fone", "elephant"]);
//!
//! // Build phonetic NFA for pattern "(ph|f)one"
//! let regex = parse("(ph|f)one").expect("the documented regex is valid");
//! let pattern = compile(&regex).expect("the documented regex compiles");
//!
//! // Create phonetic transducer
//! let transducer = PhoneticTransducerChar::new(dict, pattern, 1);
//!
//! // Exact alternatives in the pattern are retained by the product traversal.
//! let terms: Vec<_> = transducer
//!     .query_sorted("fone")
//!     .into_iter()
//!     .map(|candidate| candidate.term)
//!     .collect();
//! assert!(terms.contains(&"fone".to_string()));
//! assert!(terms.contains(&"phone".to_string()));
//! assert!(!terms.contains(&"elephant".to_string()));
//! ```

#[cfg(feature = "phonetic-rules")]
use crate::phonetic::nfa::product::ProductAutomatonChar;
#[cfg(feature = "phonetic-rules")]
use crate::phonetic::nfa::{NFAChar, NFA};
#[cfg(feature = "phonetic-rules")]
use crate::transducer::articulatory_costs::ArticulatoryCosts;
#[cfg(feature = "phonetic-rules")]
use crate::transducer::dictionary_traversal::{TraversalCursor, TraversalSession};
#[cfg(feature = "phonetic-rules")]
use crate::transducer::language::{
    LanguageProduct, LanguageQueryIterator, MappedLanguageQueryIterator,
};
#[cfg(feature = "phonetic-rules")]
use crate::transducer::Algorithm;
use libdictenstein::{Dictionary, DictionaryNode};
#[cfg(feature = "phonetic-rules")]
use libdictenstein::{MappedDictionary, MappedDictionaryNode};

use std::{
    cmp::Ordering,
    collections::VecDeque,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

#[cfg(feature = "phonetic-rules")]
const PHONETIC_NO_PATH: usize = usize::MAX;

#[cfg(feature = "phonetic-rules")]
struct PhoneticPathNode<U: Copy> {
    label: U,
    parent: usize,
    depth: usize,
}

#[cfg(feature = "phonetic-rules")]
struct PhoneticTraversal<C: Copy> {
    position: TraversalCursor<C>,
    label: Option<char>,
    parent: usize,
    depth: usize,
}

#[cfg(feature = "phonetic-rules")]
impl<C: Copy> PhoneticTraversal<C> {
    #[inline]
    fn root(position: TraversalCursor<C>) -> Self {
        Self {
            position,
            label: None,
            parent: PHONETIC_NO_PATH,
            depth: 0,
        }
    }

    #[inline]
    fn child(position: TraversalCursor<C>, label: char, parent: usize, depth: usize) -> Self {
        Self {
            position,
            label: Some(label),
            parent,
            depth,
        }
    }
}

#[cfg(feature = "phonetic-rules")]
fn collect_path_units<U: Copy>(
    label: Option<U>,
    parent: usize,
    arena: &[PhoneticPathNode<U>],
) -> Vec<U> {
    let parent_depth = if parent == PHONETIC_NO_PATH {
        0
    } else {
        arena[parent].depth
    };
    let capacity = parent_depth + usize::from(label.is_some());
    let mut units = Vec::with_capacity(capacity);

    if let Some(label) = label {
        units.push(label);
    }

    let mut current = parent;
    while current != PHONETIC_NO_PATH {
        let node = &arena[current];
        units.push(node.label);
        current = node.parent;
    }

    units.reverse();
    units
}

/// State retained only by the fractional articulatory full-dictionary scan.
///
/// Unit-cost phonetic queries use the incremental language-product
/// intersection and therefore never construct this queue, traversal session,
/// or parent-path arena. Mapped and unmapped scans share the same storage and
/// release policy; they differ only in how an accepting terminal is resolved.
#[cfg(feature = "phonetic-rules")]
struct ArticulatoryScan<N>
where
    N: DictionaryNode<Unit = char>,
{
    product: ProductAutomatonChar,
    queue: VecDeque<PhoneticTraversal<N::SnapshotCursor>>,
    traversal: TraversalSession<N>,
    path_arena: Vec<PhoneticPathNode<char>>,
    max_depth: usize,
}

#[cfg(feature = "phonetic-rules")]
impl<N> ArticulatoryScan<N>
where
    N: DictionaryNode<Unit = char>,
{
    fn from_root(
        root: libdictenstein::DictionaryTraversalRoot<N>,
        product: ProductAutomatonChar,
    ) -> Self {
        let (traversal, root) = TraversalSession::capture(root);
        Self::from_session(traversal, root, product)
    }

    fn from_session(
        traversal: TraversalSession<N>,
        root: TraversalCursor<N::SnapshotCursor>,
        product: ProductAutomatonChar,
    ) -> Self {
        let mut queue = VecDeque::with_capacity(1);
        queue.push_back(PhoneticTraversal::root(root));
        Self {
            product,
            queue,
            traversal,
            path_arena: Vec::with_capacity(64),
            max_depth: 100,
        }
    }
}

#[cfg(feature = "phonetic-rules")]
impl<N> ArticulatoryScan<N>
where
    N: MappedDictionaryNode + DictionaryNode<Unit = char>,
{
    fn from_mapped_root(
        root: libdictenstein::DictionaryTraversalRoot<N>,
        product: ProductAutomatonChar,
    ) -> Self {
        let (traversal, root) = TraversalSession::capture_mapped(root);
        Self::from_session(traversal, root, product)
    }
}

// ============================================================================
// Phonetic Candidate
// ============================================================================

/// A candidate result from phonetic transducer query.
#[derive(Debug, Clone)]
pub struct PhoneticCandidate {
    /// The matching term from the dictionary
    pub term: String,
    /// Edit distance from the query to this term (integer operation count,
    /// independent of articulatory weighting).
    pub edit_distance: u8,
    /// Articulatory adjustment to the raw edit distance.
    ///
    /// When the transducer is built with
    /// [`PhoneticTransducerChar::with_articulatory_costs`] this is the *signed*
    /// articulatory component `min_cost − edit_distance`, where `min_cost` is the
    /// articulatory-weighted alignment cost
    /// ([`ProductAutomatonChar::min_cost`](crate::phonetic::nfa::product::ProductAutomatonChar::min_cost)).
    /// Because a phonetically near substitution costs less than a full edit
    /// (`substitution_cost ∈ [0, base]`), this value is `≤ 0`: it is the
    /// *discount* a sound-alike alignment earns over a plain edit. It is exactly
    /// `0.0` for an exact match and for the default (non-articulatory) query
    /// path, so `total_cost == edit_distance` there.
    pub phonetic_cost: f64,
    /// Combined total cost (`edit_distance + phonetic_cost`).
    ///
    /// Equals the articulatory-weighted alignment cost `min_cost` when
    /// articulatory costs are configured (`≤ edit_distance`), and
    /// `edit_distance as f64` otherwise. This is the field candidates are ranked
    /// by (lower is better), so a sound-alike match outranks a same-edit-distance
    /// but phonetically distant one.
    pub total_cost: f64,
}

impl PhoneticCandidate {
    /// Create a new phonetic candidate.
    pub fn new(term: String, edit_distance: u8, phonetic_cost: f64) -> Self {
        let total_cost = edit_distance as f64 + phonetic_cost;
        Self {
            term,
            edit_distance,
            phonetic_cost,
            total_cost,
        }
    }
}

impl PartialEq for PhoneticCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.term == other.term
            && self.edit_distance == other.edit_distance
            && self.phonetic_cost.total_cmp(&other.phonetic_cost) == Ordering::Equal
            && self.total_cost.total_cmp(&other.total_cost) == Ordering::Equal
    }
}

impl Eq for PhoneticCandidate {}

impl Hash for PhoneticCandidate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.term.hash(state);
        self.edit_distance.hash(state);
        self.phonetic_cost.to_bits().hash(state);
        self.total_cost.to_bits().hash(state);
    }
}

impl PartialOrd for PhoneticCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PhoneticCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by total_cost (lower is better), then by term alphabetically
        self.total_cost
            .total_cmp(&other.total_cost)
            .then_with(|| self.term.cmp(&other.term))
            .then_with(|| self.edit_distance.cmp(&other.edit_distance))
            .then_with(|| self.phonetic_cost.total_cmp(&other.phonetic_cost))
    }
}

/// A value-returning candidate from a character-level phonetic query.
///
/// Identical to [`PhoneticCandidate`] but additionally carries the dictionary's
/// stored `value` at the matched term (e.g. a term-id), so a caller such as a
/// lexical corrector can emit `(value, total_cost)` in one pass, with no second
/// string lookup. `edit_distance` / `phonetic_cost` / `total_cost` carry exactly
/// the same meaning and ranking as [`PhoneticCandidate`].
///
/// `V` is left free of trait bounds here; ordering helpers such as
/// [`PhoneticTransducerChar::query_values_sorted`] rank by `total_cost` and so
/// never require `V: Ord`.
#[cfg(feature = "phonetic-rules")]
#[derive(Debug, Clone)]
pub struct PhoneticValueCandidate<V> {
    /// The matching term from the dictionary.
    pub term: String,
    /// Edit distance from the query to this term (integer operation count).
    pub edit_distance: u8,
    /// Articulatory adjustment to the raw edit distance; see
    /// [`PhoneticCandidate::phonetic_cost`]. `≤ 0` under an articulatory cost
    /// model, exactly `0.0` on the default (integer) path.
    pub phonetic_cost: f64,
    /// Combined total cost (`edit_distance + phonetic_cost`); the ranking key.
    pub total_cost: f64,
    /// The dictionary value stored at the matched term (e.g. a term-id).
    pub value: V,
}

#[cfg(feature = "phonetic-rules")]
impl<V> PhoneticValueCandidate<V> {
    /// Create a new value-returning phonetic candidate, deriving `total_cost`.
    pub fn new(term: String, edit_distance: u8, phonetic_cost: f64, value: V) -> Self {
        let total_cost = f64::from(edit_distance) + phonetic_cost;
        Self {
            term,
            edit_distance,
            phonetic_cost,
            total_cost,
            value,
        }
    }
}

#[cfg(feature = "phonetic-rules")]
#[inline]
fn phonetic_child_depth(depth: usize) -> Option<usize> {
    depth.checked_add(1)
}

// ============================================================================
// Character-level Phonetic Transducer
// ============================================================================

/// Phonetic transducer combining NFA pattern matching with dictionary lookups.
///
/// This transducer performs fuzzy regex queries by:
/// 1. Using a phonetic NFA to match sound-based variations
/// 2. Allowing additional edit distance for typos
/// 3. Efficiently traversing the dictionary
#[cfg(feature = "phonetic-rules")]
#[derive(Debug, Clone)]
pub struct PhoneticTransducerChar<D: Dictionary> {
    /// The dictionary to search
    dictionary: D,
    /// The phonetic NFA pattern
    nfa: NFAChar,
    /// Maximum allowed edit distance
    max_distance: u8,
    /// Reserved phonetic weight (default: `0.0`).
    ///
    /// A flat, pattern-agnostic weight knob, distinct from `articulatory_costs`.
    /// Currently stored and threaded to the query iterator but NOT applied:
    /// every emitted [`PhoneticCandidate`] takes its (non-zero) phonetic cost
    /// from `articulatory_costs`, not from this field. Clamped to `>= 0.0` at
    /// construction. Tracked for a future flat-weighting feature.
    phonetic_weight: f64,
    /// Optional articulatory (feature-distance) cost model.
    ///
    /// When `Some`, the query builds its product automaton
    /// ([`ProductAutomatonChar::with_articulatory_costs`]) with these costs and
    /// reports each candidate's articulatory-weighted `total_cost` /
    /// `phonetic_cost`. When `None` (the default), the query is a pure integer
    /// edit-distance search and every candidate has `phonetic_cost == 0.0`.
    articulatory_costs: Option<ArticulatoryCosts>,
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> PhoneticTransducerChar<D>
where
    D::Node: DictionaryNode<Unit = char>,
{
    /// Create a new phonetic transducer.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The dictionary to search
    /// * `nfa` - The phonetic NFA pattern
    /// * `max_distance` - Maximum edit distance allowed
    pub fn new(dictionary: D, nfa: NFAChar, max_distance: u8) -> Self {
        Self {
            dictionary,
            nfa,
            max_distance,
            phonetic_weight: 0.0,
            articulatory_costs: None,
        }
    }

    /// Create a phonetic transducer that scores matches with an **articulatory
    /// (feature-distance) cost model**.
    ///
    /// This is the "full articulatory" query path. Instead of a pure integer
    /// edit-distance search, each candidate's `total_cost` is the
    /// articulatory-weighted alignment cost
    /// ([`ProductAutomatonChar::min_cost`](crate::phonetic::nfa::product::ProductAutomatonChar::min_cost)):
    /// a substitution between phonetically near symbols (e.g. a voiced/voiceless
    /// pair such as `p`↔`b`) costs a fraction of a full edit, so sound-alike
    /// terms rank ahead of same-edit-distance but phonetically distant ones. The
    /// integer `edit_distance` is still reported (from
    /// [`min_distance`](crate::phonetic::nfa::product::ProductAutomatonChar::min_distance)),
    /// and `phonetic_cost = total_cost − edit_distance ≤ 0` is the articulatory
    /// discount.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - The dictionary to search
    /// * `nfa` - The phonetic NFA pattern
    /// * `max_distance` - Maximum edit distance allowed (admission is still gated
    ///   by the integer edit distance; the articulatory cost only *ranks* the
    ///   admitted candidates)
    /// * `articulatory_costs` - The feature-distance cost model
    pub fn with_articulatory_costs(
        dictionary: D,
        nfa: NFAChar,
        max_distance: u8,
        articulatory_costs: ArticulatoryCosts,
    ) -> Self {
        Self {
            dictionary,
            nfa,
            max_distance,
            phonetic_weight: 0.0,
            articulatory_costs: Some(articulatory_costs),
        }
    }

    /// Create a phonetic transducer with a custom phonetic weight.
    ///
    /// Reserved: the phonetic weight is currently stored (and threaded to the
    /// query iterator) but NOT applied to matching cost or ranking. Every
    /// emitted [`PhoneticCandidate`] reports `phonetic_cost == 0.0`, so
    /// `total_cost == edit_distance`. The value is clamped to `>= 0.0` so a
    /// future implementation cannot violate monotone-cost pruning. Tracked for a
    /// future weighted-phonetic-matching feature.
    pub fn with_phonetic_weight(
        dictionary: D,
        nfa: NFAChar,
        max_distance: u8,
        phonetic_weight: f64,
    ) -> Self {
        Self {
            dictionary,
            nfa,
            max_distance,
            phonetic_weight: phonetic_weight.max(0.0),
            articulatory_costs: None,
        }
    }

    /// Query for dictionary terms matching the phonetic pattern.
    ///
    /// Returns an iterator over [`PhoneticCandidate`] results in dictionary
    /// traversal order. Use [`query_sorted`](Self::query_sorted) when total-cost
    /// ordering is required.
    pub fn query(&self, input: &str) -> PhoneticQueryIteratorChar<'_, D> {
        PhoneticQueryIteratorChar::new(
            &self.dictionary,
            &self.nfa,
            input,
            self.max_distance,
            self.phonetic_weight,
            self.articulatory_costs,
        )
    }

    /// Construct the historical unit-cost iterator layout for a same-binary
    /// causal benchmark.
    ///
    /// This surface exists only with `benchmark-controls`. It recreates the
    /// pre-mode-split field, construction, dispatch, and drop order while
    /// delegating results to the same incremental language-product engine.
    /// Production builds expose neither this type nor this constructor.
    #[cfg(feature = "benchmark-controls")]
    #[doc(hidden)]
    pub fn query_legacy_unit_cost_retention_control(
        &self,
        input: &str,
    ) -> LegacyPhoneticQueryIteratorChar<'_, D> {
        assert!(
            self.articulatory_costs.is_none(),
            "the legacy retention control models only unit-cost phonetic queries"
        );
        LegacyPhoneticQueryIteratorChar::new(
            &self.dictionary,
            &self.nfa,
            input,
            self.max_distance,
            self.phonetic_weight,
        )
    }

    /// Query and collect all results, sorted by total cost.
    pub fn query_sorted(&self, input: &str) -> Vec<PhoneticCandidate> {
        let mut results: Vec<_> = self.query(input).collect();
        results.sort();
        results
    }

    /// Get the underlying dictionary.
    #[inline]
    pub fn dictionary(&self) -> &D {
        &self.dictionary
    }

    /// Get the phonetic NFA.
    #[inline]
    pub fn nfa(&self) -> &NFAChar {
        &self.nfa
    }

    /// Get the maximum distance.
    #[inline]
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Extract the dictionary, consuming the transducer.
    pub fn into_dictionary(self) -> D {
        self.dictionary
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D> PhoneticTransducerChar<D>
where
    D: MappedDictionary,
    D::Node: MappedDictionaryNode<Value = D::Value> + DictionaryNode<Unit = char>,
{
    /// Query for dictionary terms matching the phonetic pattern, returning each
    /// match's stored dictionary value (e.g. a term-id) alongside its costs.
    ///
    /// This is the value-returning counterpart of [`query`](Self::query): it
    /// yields [`PhoneticValueCandidate`] (a [`PhoneticCandidate`] plus the
    /// `value` at the matched term), so a lexical corrector can emit
    /// `(term_id, total_cost)` in a single pass with no string round-trip.
    /// Articulatory weighting (if the transducer was built with
    /// [`with_articulatory_costs`](Self::with_articulatory_costs)) applies here
    /// exactly as it does for [`query`](Self::query).
    pub fn query_values(&self, input: &str) -> PhoneticValueQueryIteratorChar<'_, D> {
        PhoneticValueQueryIteratorChar::new(
            &self.dictionary,
            &self.nfa,
            input,
            self.max_distance,
            self.articulatory_costs,
        )
    }

    /// Query and collect value-returning results, sorted by total cost.
    ///
    /// Ranks by `total_cost` (then term); requires no `Ord` bound on the value
    /// type.
    pub fn query_values_sorted(&self, input: &str) -> Vec<PhoneticValueCandidate<D::Value>> {
        let mut results: Vec<_> = self.query_values(input).collect();
        results.sort_by(|a, b| {
            a.total_cost
                .total_cmp(&b.total_cost)
                .then_with(|| a.term.cmp(&b.term))
        });
        results
    }
}

// ============================================================================
// Query Iterator (Character-level)
// ============================================================================

/// Iterator over phonetic query results.
#[cfg(feature = "phonetic-rules")]
enum PhoneticQueryModeChar<N>
where
    N: DictionaryNode<Unit = char>,
{
    Incremental(LanguageQueryIterator<N, NFAChar>),
    Articulatory(ArticulatoryScan<N>),
}

/// Historical character-phonetic iterator retained solely as an exact
/// same-binary experimental control.
///
/// Field order is intentional: before query-lifetime mode selection, the
/// dormant product was constructed first and dropped before the active
/// language iterator, followed by the unused scan state. Keeping this as a
/// distinct type means the treatment retains the actual production iterator
/// layout rather than an inline optional control payload.
#[cfg(all(feature = "phonetic-rules", feature = "benchmark-controls"))]
#[doc(hidden)]
pub struct LegacyPhoneticQueryIteratorChar<'a, D: Dictionary>
where
    D::Node: DictionaryNode<Unit = char>,
{
    _product: ProductAutomatonChar,
    language_query: Option<LanguageQueryIterator<D::Node, NFAChar>>,
    _queue: VecDeque<PhoneticTraversal<<D::Node as DictionaryNode>::SnapshotCursor>>,
    _traversal: Option<TraversalSession<D::Node>>,
    _path_arena: Vec<PhoneticPathNode<char>>,
    _dictionary: PhantomData<&'a D>,
    _max_depth: usize,
    _phonetic_weight: f64,
}

#[cfg(all(feature = "phonetic-rules", feature = "benchmark-controls"))]
impl<'a, D: Dictionary> LegacyPhoneticQueryIteratorChar<'a, D>
where
    D::Node: DictionaryNode<Unit = char>,
{
    fn new(
        dictionary: &'a D,
        nfa: &NFAChar,
        _input: &str,
        max_distance: u8,
        phonetic_weight: f64,
    ) -> Self {
        let product = ProductAutomatonChar::new(nfa.clone(), max_distance);
        let language_query = product.supports_incremental_trie_intersection().then(|| {
            LanguageQueryIterator::from_dictionary(
                dictionary,
                LanguageProduct::new(nfa.clone(), max_distance),
            )
        });
        let (traversal, root) = if language_query.is_none() {
            let (session, root) = TraversalSession::capture(dictionary.traversal_root());
            (Some(session), Some(root))
        } else {
            (None, None)
        };
        let mut queue = VecDeque::with_capacity(usize::from(root.is_some()));
        if let Some(root) = root {
            queue.push_back(PhoneticTraversal::root(root));
        }

        Self {
            _product: product,
            language_query,
            _queue: queue,
            _traversal: traversal,
            _path_arena: Vec::with_capacity(64),
            _dictionary: PhantomData,
            _max_depth: 100,
            _phonetic_weight: phonetic_weight,
        }
    }
}

#[cfg(all(feature = "phonetic-rules", feature = "benchmark-controls"))]
impl<D: Dictionary> Iterator for LegacyPhoneticQueryIteratorChar<'_, D>
where
    D::Node: DictionaryNode<Unit = char>,
{
    type Item = PhoneticCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        let matched = self.language_query.as_mut()?.next()?;
        Some(PhoneticCandidate::new(
            matched.units.into_iter().collect(),
            matched.distance,
            0.0,
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[cfg(all(feature = "phonetic-rules", feature = "benchmark-controls"))]
impl<D: Dictionary> std::iter::FusedIterator for LegacyPhoneticQueryIteratorChar<'_, D> where
    D::Node: DictionaryNode<Unit = char>
{
}

/// Iterator over phonetic query results.
#[cfg(feature = "phonetic-rules")]
pub struct PhoneticQueryIteratorChar<'a, D: Dictionary>
where
    D::Node: DictionaryNode<Unit = char>,
{
    /// Query-lifetime dispatch keeps mutually exclusive traversal machinery in
    /// distinct variants rather than retaining unused queues and automata.
    mode: PhoneticQueryModeChar<D::Node>,
    /// Keeps the iterator lifetime tied to the dictionary that produced its nodes.
    _dictionary: PhantomData<&'a D>,
    /// Phonetic weight
    _phonetic_weight: f64,
}

#[cfg(feature = "phonetic-rules")]
impl<'a, D: Dictionary> PhoneticQueryIteratorChar<'a, D>
where
    D::Node: DictionaryNode<Unit = char>,
{
    fn new(
        dictionary: &'a D,
        nfa: &NFAChar,
        _input: &str,
        max_distance: u8,
        phonetic_weight: f64,
        articulatory_costs: Option<ArticulatoryCosts>,
    ) -> Self {
        let mode = match articulatory_costs {
            Some(costs) => {
                let product = ProductAutomatonChar::with_articulatory_costs(
                    nfa.clone(),
                    f64::from(max_distance),
                    Algorithm::Standard,
                    costs,
                );
                PhoneticQueryModeChar::Articulatory(ArticulatoryScan::from_root(
                    dictionary.traversal_root(),
                    product,
                ))
            }
            None => PhoneticQueryModeChar::Incremental(LanguageQueryIterator::from_dictionary(
                dictionary,
                LanguageProduct::new(nfa.clone(), max_distance),
            )),
        };

        Self {
            mode,
            _dictionary: PhantomData,
            _phonetic_weight: phonetic_weight,
        }
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> Iterator for PhoneticQueryIteratorChar<'_, D>
where
    D::Node: DictionaryNode<Unit = char>,
{
    type Item = PhoneticCandidate;

    fn next(&mut self) -> Option<Self::Item> {
        let scan = match &mut self.mode {
            PhoneticQueryModeChar::Incremental(query) => {
                let matched = query.next()?;
                return Some(PhoneticCandidate::new(
                    matched.units.into_iter().collect(),
                    matched.distance,
                    0.0,
                ));
            }
            PhoneticQueryModeChar::Articulatory(scan) => scan,
        };

        while let Some(entry) = scan.queue.pop_front() {
            // Depth limit to prevent infinite exploration
            if entry.depth > scan.max_depth {
                scan.traversal.discard_unexpanded(entry.position);
                continue;
            }

            // Inspect finality and enqueue children as one logical dictionary
            // operation whenever this node is expandable. This lets
            // boundary-backed dictionaries fuse their lock/callback work.
            let child_depth = phonetic_child_depth(entry.depth)
                .filter(|child_depth| *child_depth <= scan.max_depth);
            let mut child_parent = None;
            let paths = &mut scan.path_arena;
            let queue = &mut scan.queue;
            let is_final = scan.traversal.filter_map_edges_and_finality(
                entry.position,
                |_| child_depth,
                |c, child_position, child_depth| {
                    let parent = *child_parent.get_or_insert_with(|| match entry.label {
                        Some(label) => {
                            let depth = if entry.parent == PHONETIC_NO_PATH {
                                1
                            } else {
                                paths[entry.parent].depth.saturating_add(1)
                            };
                            let index = paths.len();
                            paths.push(PhoneticPathNode {
                                label,
                                parent: entry.parent,
                                depth,
                            });
                            index
                        }
                        None => entry.parent,
                    });
                    queue.push_back(PhoneticTraversal::child(
                        child_position,
                        c,
                        parent,
                        child_depth,
                    ));
                },
            );

            // Determine whether this node yields a candidate after its children
            // are enqueued, so extensions of a matched prefix are never skipped.
            let candidate = if is_final {
                // Check if the product automaton accepts this path
                let path: String = collect_path_units(entry.label, entry.parent, paths)
                    .into_iter()
                    .collect();
                let visible = !scan.traversal.requires_final_units() || {
                    let units: Vec<_> = path.chars().collect();
                    scan.traversal.accepts_final_units(&units)
                };
                visible
                    .then(|| scan.product.min_distance(&path))
                    .flatten()
                    .map(|distance| {
                        // The full scan exists only for articulatory scoring, so
                        // `min_cost − edit_distance` is the signed discount.
                        let total = scan
                            .product
                            .min_cost(&path)
                            .unwrap_or_else(|| f64::from(distance));
                        PhoneticCandidate::new(path, distance, total - f64::from(distance))
                    })
            } else {
                None
            };

            if let Some(candidate) = candidate {
                return Some(candidate);
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> std::iter::FusedIterator for PhoneticQueryIteratorChar<'_, D> where
    D::Node: DictionaryNode<Unit = char>
{
}

// ============================================================================
// Value-returning Query Iterator (Character-level)
// ============================================================================

/// Iterator over value-returning phonetic query results (character-level).
///
/// Like [`PhoneticQueryIteratorChar`] but yields [`PhoneticValueCandidate`],
/// reading the dictionary's stored value at each matched term. Requires the
/// dictionary to be a [`MappedDictionary`] whose nodes are
/// [`MappedDictionaryNode`]s (e.g. a term → term-id vocabulary trie).
#[cfg(feature = "phonetic-rules")]
enum PhoneticValueQueryModeChar<N>
where
    N: MappedDictionaryNode + DictionaryNode<Unit = char>,
{
    Incremental(MappedLanguageQueryIterator<N, NFAChar>),
    Articulatory(ArticulatoryScan<N>),
}

/// Iterator over value-returning character-level phonetic query results.
///
/// The iterator selects either incremental language-product traversal or the
/// fractional articulatory full scan when it is created, retaining storage for
/// only that execution mode.
#[cfg(feature = "phonetic-rules")]
pub struct PhoneticValueQueryIteratorChar<'a, D: MappedDictionary>
where
    D::Node: MappedDictionaryNode<Value = D::Value> + DictionaryNode<Unit = char>,
{
    /// Query-lifetime dispatch for the mutually exclusive incremental and
    /// articulatory traversal engines.
    mode: PhoneticValueQueryModeChar<D::Node>,
    /// Keeps the iterator lifetime tied to the dictionary that produced its nodes.
    _dictionary: PhantomData<&'a D>,
}

#[cfg(feature = "phonetic-rules")]
impl<'a, D: MappedDictionary> PhoneticValueQueryIteratorChar<'a, D>
where
    D::Node: MappedDictionaryNode<Value = D::Value> + DictionaryNode<Unit = char>,
{
    fn new(
        dictionary: &'a D,
        nfa: &NFAChar,
        _input: &str,
        max_distance: u8,
        articulatory_costs: Option<ArticulatoryCosts>,
    ) -> Self {
        let mode = match articulatory_costs {
            Some(costs) => {
                let product = ProductAutomatonChar::with_articulatory_costs(
                    nfa.clone(),
                    f64::from(max_distance),
                    Algorithm::Standard,
                    costs,
                );
                PhoneticValueQueryModeChar::Articulatory(ArticulatoryScan::from_mapped_root(
                    dictionary.traversal_root(),
                    product,
                ))
            }
            None => PhoneticValueQueryModeChar::Incremental(
                MappedLanguageQueryIterator::from_traversal_root(
                    dictionary.traversal_root(),
                    LanguageProduct::new(nfa.clone(), max_distance),
                ),
            ),
        };

        Self {
            mode,
            _dictionary: PhantomData,
        }
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: MappedDictionary> Iterator for PhoneticValueQueryIteratorChar<'_, D>
where
    D::Node: MappedDictionaryNode<Value = D::Value> + DictionaryNode<Unit = char>,
{
    type Item = PhoneticValueCandidate<D::Value>;

    fn next(&mut self) -> Option<Self::Item> {
        let scan = match &mut self.mode {
            PhoneticValueQueryModeChar::Incremental(query) => {
                for matched in query {
                    if let Some(value) = matched.value {
                        return Some(PhoneticValueCandidate::new(
                            matched.units.into_iter().collect(),
                            matched.distance,
                            0.0,
                            value,
                        ));
                    }
                }
                return None;
            }
            PhoneticValueQueryModeChar::Articulatory(scan) => scan,
        };

        while let Some(entry) = scan.queue.pop_front() {
            if entry.depth > scan.max_depth {
                scan.traversal.discard_unexpanded(entry.position);
                continue;
            }

            let child_depth = phonetic_child_depth(entry.depth)
                .filter(|child_depth| *child_depth <= scan.max_depth);
            let mut child_parent = None;
            let paths = &mut scan.path_arena;
            let queue = &mut scan.queue;
            let final_source = scan.traversal.filter_map_edges_and_final_source(
                entry.position,
                |_| child_depth,
                |c, child_position, child_depth| {
                    let parent = *child_parent.get_or_insert_with(|| match entry.label {
                        Some(label) => {
                            let depth = if entry.parent == PHONETIC_NO_PATH {
                                1
                            } else {
                                paths[entry.parent].depth.saturating_add(1)
                            };
                            let index = paths.len();
                            paths.push(PhoneticPathNode {
                                label,
                                parent: entry.parent,
                                depth,
                            });
                            index
                        }
                        None => entry.parent,
                    });
                    queue.push_back(PhoneticTraversal::child(
                        child_position,
                        c,
                        parent,
                        child_depth,
                    ));
                },
            );

            // A candidate requires a stored value at this terminal and a
            // phonetic/edit match. Children have already been enqueued.
            let candidate = if let Some(final_source) = final_source {
                let path: String = collect_path_units(entry.label, entry.parent, paths)
                    .into_iter()
                    .collect();
                if let Some(distance) = scan.product.min_distance(&path) {
                    let final_units = scan
                        .traversal
                        .requires_final_units()
                        .then(|| path.chars().collect::<Vec<_>>());
                    if final_units
                        .as_deref()
                        .is_some_and(|units| !scan.traversal.accepts_final_units(units))
                    {
                        None
                    } else {
                        scan.traversal
                            .resolve_final_value(final_source, final_units.as_deref())
                            .map(|value| {
                                let total = scan
                                    .product
                                    .min_cost(&path)
                                    .unwrap_or_else(|| f64::from(distance));
                                PhoneticValueCandidate::new(
                                    path,
                                    distance,
                                    total - f64::from(distance),
                                    value,
                                )
                            })
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(candidate) = candidate {
                return Some(candidate);
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: MappedDictionary> std::iter::FusedIterator for PhoneticValueQueryIteratorChar<'_, D> where
    D::Node: MappedDictionaryNode<Value = D::Value> + DictionaryNode<Unit = char>
{
}

// ============================================================================
// Byte-level Phonetic Transducer
// ============================================================================

/// Byte-level phonetic transducer.
#[cfg(feature = "phonetic-rules")]
#[derive(Debug, Clone)]
pub struct PhoneticTransducer<D: Dictionary> {
    /// The dictionary to search
    dictionary: D,
    /// The phonetic NFA pattern
    nfa: NFA,
    /// Maximum allowed edit distance
    max_distance: u8,
    /// Reserved phonetic weight (default: `0.0`).
    ///
    /// Currently stored and threaded to the query iterator but NOT applied:
    /// every emitted [`PhoneticCandidateByte`] has `phonetic_cost == 0.0`.
    /// Clamped to `>= 0.0` at construction. Tracked for a future
    /// weighted-phonetic-matching feature.
    phonetic_weight: f64,
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> PhoneticTransducer<D>
where
    D::Node: DictionaryNode<Unit = u8>,
{
    /// Create a new byte-level phonetic transducer.
    pub fn new(dictionary: D, nfa: NFA, max_distance: u8) -> Self {
        Self {
            dictionary,
            nfa,
            max_distance,
            phonetic_weight: 0.0,
        }
    }

    /// Create with a custom phonetic weight.
    ///
    /// Reserved: the phonetic weight is currently stored (and threaded to the
    /// query iterator) but NOT applied to matching cost or ranking. Every
    /// emitted [`PhoneticCandidateByte`] reports `phonetic_cost == 0.0`, so
    /// `total_cost == edit_distance`. The value is clamped to `>= 0.0` to
    /// preserve the future monotone-cost-pruning invariant. Tracked for a future
    /// weighted-phonetic-matching feature.
    pub fn with_phonetic_weight(
        dictionary: D,
        nfa: NFA,
        max_distance: u8,
        phonetic_weight: f64,
    ) -> Self {
        Self {
            dictionary,
            nfa,
            max_distance,
            phonetic_weight: phonetic_weight.max(0.0),
        }
    }

    /// Query for dictionary terms matching the phonetic pattern.
    pub fn query(&self, input: &[u8]) -> PhoneticQueryIterator<'_, D> {
        PhoneticQueryIterator::new(
            &self.dictionary,
            &self.nfa,
            input,
            self.max_distance,
            self.phonetic_weight,
        )
    }

    /// Query and collect all results, sorted by total cost.
    pub fn query_sorted(&self, input: &[u8]) -> Vec<PhoneticCandidateByte> {
        let mut results: Vec<_> = self.query(input).collect();
        results.sort();
        results
    }

    /// Get the underlying dictionary.
    #[inline]
    pub fn dictionary(&self) -> &D {
        &self.dictionary
    }

    /// Get the phonetic NFA.
    #[inline]
    pub fn nfa(&self) -> &NFA {
        &self.nfa
    }

    /// Get the maximum distance.
    #[inline]
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Extract the dictionary.
    pub fn into_dictionary(self) -> D {
        self.dictionary
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D> PhoneticTransducer<D>
where
    D: MappedDictionary,
    D::Node: MappedDictionaryNode<Value = D::Value> + DictionaryNode<Unit = u8>,
{
    /// Query for dictionary terms matching the phonetic pattern, returning each
    /// match's stored dictionary value (e.g. a term-id) alongside its costs.
    ///
    /// The byte counterpart of [`PhoneticTransducerChar::query_values`]. Yields
    /// [`PhoneticValueCandidateByte`]; `phonetic_cost` is always `0.0` for the
    /// byte path.
    pub fn query_values(&self, input: &[u8]) -> PhoneticValueQueryIterator<'_, D> {
        PhoneticValueQueryIterator::new(&self.dictionary, &self.nfa, input, self.max_distance)
    }

    /// Query and collect value-returning results, sorted by total cost.
    pub fn query_values_sorted(&self, input: &[u8]) -> Vec<PhoneticValueCandidateByte<D::Value>> {
        let mut results: Vec<_> = self.query_values(input).collect();
        results.sort_by(|a, b| {
            a.total_cost
                .total_cmp(&b.total_cost)
                .then_with(|| a.term.cmp(&b.term))
        });
        results
    }
}

/// Byte-level phonetic candidate.
#[derive(Debug, Clone)]
pub struct PhoneticCandidateByte {
    /// The matching term from the dictionary
    pub term: Vec<u8>,
    /// Edit distance from the query to this term
    pub edit_distance: u8,
    /// Phonetic transformation cost — always `0.0` for the byte path.
    ///
    /// Articulatory (feature-distance) costs are defined over phonemes
    /// (`char`s), and the byte NFA's transition label carries no "expected byte"
    /// to weight a substitution against, so a per-substitution articulatory cost
    /// is not expressible at byte granularity. Byte-level phonetic matching is
    /// therefore integer edit distance only; use
    /// [`PhoneticTransducerChar::with_articulatory_costs`] for the weighted
    /// (articulatory) path.
    pub phonetic_cost: f64,
    /// Combined total cost (`edit_distance + phonetic_cost`).
    ///
    /// Because `phonetic_cost` is always `0.0` for bytes, this equals
    /// `edit_distance as f64`.
    pub total_cost: f64,
}

impl PartialEq for PhoneticCandidateByte {
    fn eq(&self, other: &Self) -> bool {
        self.term == other.term
            && self.edit_distance == other.edit_distance
            && self.phonetic_cost.total_cmp(&other.phonetic_cost) == Ordering::Equal
            && self.total_cost.total_cmp(&other.total_cost) == Ordering::Equal
    }
}

impl Eq for PhoneticCandidateByte {}

impl Hash for PhoneticCandidateByte {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.term.hash(state);
        self.edit_distance.hash(state);
        self.phonetic_cost.to_bits().hash(state);
        self.total_cost.to_bits().hash(state);
    }
}

impl PhoneticCandidateByte {
    /// Create a new phonetic candidate.
    pub fn new(term: Vec<u8>, edit_distance: u8, phonetic_cost: f64) -> Self {
        let total_cost = edit_distance as f64 + phonetic_cost;
        Self {
            term,
            edit_distance,
            phonetic_cost,
            total_cost,
        }
    }
}

impl PartialOrd for PhoneticCandidateByte {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PhoneticCandidateByte {
    fn cmp(&self, other: &Self) -> Ordering {
        self.total_cost
            .total_cmp(&other.total_cost)
            .then_with(|| self.term.cmp(&other.term))
            .then_with(|| self.edit_distance.cmp(&other.edit_distance))
            .then_with(|| self.phonetic_cost.total_cmp(&other.phonetic_cost))
    }
}

/// A value-returning candidate from a byte-level phonetic query.
///
/// The byte analogue of [`PhoneticValueCandidate`]: it carries the dictionary's
/// stored `value` at the matched term alongside the byte term. `phonetic_cost`
/// is always `0.0` for the byte path (see [`PhoneticCandidateByte::phonetic_cost`]),
/// so `total_cost == edit_distance`.
#[cfg(feature = "phonetic-rules")]
#[derive(Debug, Clone)]
pub struct PhoneticValueCandidateByte<V> {
    /// The matching term from the dictionary.
    pub term: Vec<u8>,
    /// Edit distance from the query to this term.
    pub edit_distance: u8,
    /// Phonetic transformation cost — always `0.0` for the byte path.
    pub phonetic_cost: f64,
    /// Combined total cost (`edit_distance + phonetic_cost`); the ranking key.
    pub total_cost: f64,
    /// The dictionary value stored at the matched term (e.g. a term-id).
    pub value: V,
}

#[cfg(feature = "phonetic-rules")]
impl<V> PhoneticValueCandidateByte<V> {
    /// Create a new value-returning byte phonetic candidate, deriving `total_cost`.
    pub fn new(term: Vec<u8>, edit_distance: u8, phonetic_cost: f64, value: V) -> Self {
        let total_cost = f64::from(edit_distance) + phonetic_cost;
        Self {
            term,
            edit_distance,
            phonetic_cost,
            total_cost,
            value,
        }
    }
}

/// Iterator over byte-level phonetic query results.
#[cfg(feature = "phonetic-rules")]
pub struct PhoneticQueryIterator<'a, D: Dictionary>
where
    D::Node: DictionaryNode<Unit = u8>,
{
    /// Frontier-pruned generic language query.
    language_query: LanguageQueryIterator<D::Node, NFA>,
    /// Keeps the iterator lifetime tied to the dictionary that produced its nodes.
    _dictionary: PhantomData<&'a D>,
    /// Phonetic weight
    _phonetic_weight: f64,
}

#[cfg(feature = "phonetic-rules")]
impl<'a, D: Dictionary> PhoneticQueryIterator<'a, D>
where
    D::Node: DictionaryNode<Unit = u8>,
{
    fn new(
        dictionary: &'a D,
        nfa: &NFA,
        _input: &[u8],
        max_distance: u8,
        phonetic_weight: f64,
    ) -> Self {
        Self {
            language_query: LanguageQueryIterator::from_dictionary(
                dictionary,
                LanguageProduct::new(nfa.clone(), max_distance),
            ),
            _dictionary: PhantomData,
            _phonetic_weight: phonetic_weight,
        }
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> Iterator for PhoneticQueryIterator<'_, D>
where
    D::Node: DictionaryNode<Unit = u8>,
{
    type Item = PhoneticCandidateByte;

    fn next(&mut self) -> Option<Self::Item> {
        self.language_query
            .next()
            .map(|matched| PhoneticCandidateByte::new(matched.units, matched.distance, 0.0))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: Dictionary> std::iter::FusedIterator for PhoneticQueryIterator<'_, D> where
    D::Node: DictionaryNode<Unit = u8>
{
}

// ============================================================================
// Value-returning Query Iterator (Byte-level)
// ============================================================================

/// Iterator over value-returning byte-level phonetic query results.
///
/// The byte analogue of [`PhoneticValueQueryIteratorChar`]: yields
/// [`PhoneticValueCandidateByte`], reading the dictionary's stored value at each
/// matched term. Requires the dictionary to be a [`MappedDictionary`] whose
/// nodes are [`MappedDictionaryNode`]s. `phonetic_cost` is `0.0` (byte paths are
/// integer edit distance only).
#[cfg(feature = "phonetic-rules")]
pub struct PhoneticValueQueryIterator<'a, D: MappedDictionary>
where
    D::Node: MappedDictionaryNode<Value = D::Value> + DictionaryNode<Unit = u8>,
{
    /// Frontier-pruned generic language query.
    language_query: MappedLanguageQueryIterator<D::Node, NFA>,
    /// Keeps the iterator lifetime tied to the dictionary that produced its nodes.
    _dictionary: PhantomData<&'a D>,
}

#[cfg(feature = "phonetic-rules")]
impl<'a, D: MappedDictionary> PhoneticValueQueryIterator<'a, D>
where
    D::Node: MappedDictionaryNode<Value = D::Value> + DictionaryNode<Unit = u8>,
{
    fn new(dictionary: &'a D, nfa: &NFA, _input: &[u8], max_distance: u8) -> Self {
        Self {
            language_query: MappedLanguageQueryIterator::from_traversal_root(
                dictionary.traversal_root(),
                LanguageProduct::new(nfa.clone(), max_distance),
            ),
            _dictionary: PhantomData,
        }
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: MappedDictionary> Iterator for PhoneticValueQueryIterator<'_, D>
where
    D::Node: MappedDictionaryNode<Value = D::Value> + DictionaryNode<Unit = u8>,
{
    type Item = PhoneticValueCandidateByte<D::Value>;

    fn next(&mut self) -> Option<Self::Item> {
        for matched in &mut self.language_query {
            if let Some(value) = matched.value {
                return Some(PhoneticValueCandidateByte::new(
                    matched.units,
                    matched.distance,
                    0.0,
                    value,
                ));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[cfg(feature = "phonetic-rules")]
impl<D: MappedDictionary> std::iter::FusedIterator for PhoneticValueQueryIterator<'_, D> where
    D::Node: MappedDictionaryNode<Value = D::Value> + DictionaryNode<Unit = u8>
{
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "phonetic-rules")]
mod tests {
    use super::*;
    use crate::phonetic::nfa::compiler::{compile, compile_bytes};
    use crate::phonetic::regex::{parse, parse_bytes};
    use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
    use libdictenstein::double_array_trie::DoubleArrayTrie;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn candidate_hash<T: Hash>(candidate: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        candidate.hash(&mut hasher);
        hasher.finish()
    }

    #[cfg(feature = "phonetic-rules")]
    #[test]
    fn test_phonetic_child_depth_rejects_overflow() {
        assert_eq!(phonetic_child_depth(0), Some(1));
        assert_eq!(phonetic_child_depth(usize::MAX), None);
    }

    #[test]
    fn character_query_retains_only_its_selected_execution_mode() {
        let nfa = compile(&parse("phone").expect("parse")).expect("compile");
        let incremental = PhoneticTransducerChar::new(
            DoubleArrayTrieChar::from_terms(["phone", "fone"]),
            nfa.clone(),
            1,
        );
        assert!(matches!(
            incremental.query("phone").mode,
            PhoneticQueryModeChar::Incremental(_)
        ));

        let articulatory = PhoneticTransducerChar::with_articulatory_costs(
            DoubleArrayTrieChar::from_terms(["phone", "fone"]),
            nfa,
            1,
            ArticulatoryCosts::default(),
        );
        let query = articulatory.query("phone");
        let PhoneticQueryModeChar::Articulatory(scan) = query.mode else {
            panic!("articulatory scoring requires the full-scan mode");
        };
        assert_eq!(scan.queue.len(), 1);
        assert!(scan.path_arena.is_empty());
    }

    #[cfg(feature = "benchmark-controls")]
    #[test]
    fn legacy_phonetic_retention_control_is_exact_and_layout_distinct() {
        let nfa = compile(&parse("phone").expect("parse")).expect("compile");
        let transducer = PhoneticTransducerChar::new(
            DoubleArrayTrieChar::from_terms(["phone", "phones", "fone", "stone"]),
            nfa,
            2,
        );

        let current = transducer.query("phone");
        let legacy = transducer.query_legacy_unit_cost_retention_control("phone");
        assert!(legacy.language_query.is_some());
        assert!(legacy._queue.is_empty());
        assert_eq!(legacy._queue.capacity(), 0);
        assert!(legacy._traversal.is_none());
        assert!(legacy._path_arena.is_empty());
        assert_eq!(legacy._path_arena.capacity(), 64);
        assert_eq!(legacy._max_depth, 100);
        assert!(std::mem::size_of_val(&legacy) > std::mem::size_of_val(&current));

        let current: Vec<_> = current.collect();
        let legacy: Vec<_> = legacy.collect();
        assert_eq!(legacy, current);
    }

    #[test]
    fn test_phonetic_candidate_ordering() {
        let c1 = PhoneticCandidate::new("apple".to_string(), 0, 0.0);
        let c2 = PhoneticCandidate::new("apply".to_string(), 1, 0.0);
        let c3 = PhoneticCandidate::new("banana".to_string(), 0, 0.0);

        assert!(c1 < c2); // 0.0 < 1.0
        assert!(c1 < c3); // same cost, alphabetically
    }

    #[test]
    fn test_phonetic_candidate_eq_ord_hash_contract() {
        let phonetic_path = PhoneticCandidate::new("same".to_string(), 0, 1.0);
        let edit_path = PhoneticCandidate::new("same".to_string(), 1, 0.0);

        assert_eq!(phonetic_path.total_cost, edit_path.total_cost);
        assert_ne!(phonetic_path, edit_path);
        assert_ne!(phonetic_path.cmp(&edit_path), Ordering::Equal);

        let positive_zero = PhoneticCandidate {
            term: "zero".to_string(),
            edit_distance: 0,
            phonetic_cost: 0.0,
            total_cost: 0.0,
        };
        let negative_zero = PhoneticCandidate {
            term: "zero".to_string(),
            edit_distance: 0,
            phonetic_cost: -0.0,
            total_cost: -0.0,
        };

        assert_ne!(positive_zero, negative_zero);
        assert_ne!(positive_zero.cmp(&negative_zero), Ordering::Equal);

        let same_candidate = phonetic_path.clone();
        assert_eq!(phonetic_path, same_candidate);
        assert_eq!(
            candidate_hash(&phonetic_path),
            candidate_hash(&same_candidate)
        );
    }

    #[test]
    fn test_phonetic_candidate_byte_eq_ord_hash_contract() {
        let phonetic_path = PhoneticCandidateByte::new(b"same".to_vec(), 0, 1.0);
        let edit_path = PhoneticCandidateByte::new(b"same".to_vec(), 1, 0.0);

        assert_eq!(phonetic_path.total_cost, edit_path.total_cost);
        assert_ne!(phonetic_path, edit_path);
        assert_ne!(phonetic_path.cmp(&edit_path), Ordering::Equal);

        let positive_zero = PhoneticCandidateByte {
            term: b"zero".to_vec(),
            edit_distance: 0,
            phonetic_cost: 0.0,
            total_cost: 0.0,
        };
        let negative_zero = PhoneticCandidateByte {
            term: b"zero".to_vec(),
            edit_distance: 0,
            phonetic_cost: -0.0,
            total_cost: -0.0,
        };

        assert_ne!(positive_zero, negative_zero);
        assert_ne!(positive_zero.cmp(&negative_zero), Ordering::Equal);

        let same_candidate = phonetic_path.clone();
        assert_eq!(phonetic_path, same_candidate);
        assert_eq!(
            candidate_hash(&phonetic_path),
            candidate_hash(&same_candidate)
        );
    }

    #[test]
    fn test_phonetic_transducer_basic() {
        let dict = DoubleArrayTrieChar::from_terms(["phone", "fone", "bone", "tone"]);
        let nfa = compile(&parse("(ph|f)one").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa, 1);

        let results: Vec<_> = transducer.query("phone").collect();
        let terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();

        // Should find "phone" and "fone" (exact pattern matches)
        // May also find "bone" and "tone" within distance 1
        assert!(terms.contains(&"phone") || terms.contains(&"fone"));
    }

    #[test]
    fn test_phonetic_transducer_sorted() {
        let dict = DoubleArrayTrieChar::from_terms(["test", "best", "rest", "nest"]);
        let nfa = compile(&parse("test").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa, 1);

        let results = transducer.query_sorted("test");

        // First result should be exact match
        if !results.is_empty() {
            assert_eq!(results[0].term, "test");
            assert_eq!(results[0].edit_distance, 0);
        }
    }

    #[test]
    fn test_phonetic_transducer_no_match() {
        let dict = DoubleArrayTrieChar::from_terms(["xyz", "abc", "def"]);
        let nfa = compile(&parse("phone").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa, 1);

        let results: Vec<_> = transducer.query("phone").collect();

        // No matches - "phone" is too far from all dictionary terms
        assert!(results.is_empty());
    }

    #[test]
    fn test_phonetic_transducer_alternation() {
        let dict = DoubleArrayTrieChar::from_terms(["cat", "kat", "bat", "hat"]);
        let nfa = compile(&parse("(c|k)at").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa, 0);

        let results: Vec<_> = transducer.query("cat").collect();
        let terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();

        // Both "cat" and "kat" match the pattern exactly
        assert!(terms.contains(&"cat"));
        assert!(terms.contains(&"kat"));
    }

    #[test]
    fn test_byte_phonetic_transducer_shared_prefix_branching() {
        let dict = DoubleArrayTrie::from_terms(["phone", "fone", "phony", "bone"]);
        let nfa = compile_bytes(&parse_bytes(b"(ph|f)one").expect("parse")).expect("compile bytes");

        let transducer = PhoneticTransducer::new(dict, nfa, 1);
        let mut terms: Vec<_> = transducer
            .query(b"phone")
            .map(|candidate| candidate.term)
            .collect();
        terms.sort();

        assert!(terms.contains(&b"fone".to_vec()));
        assert!(terms.contains(&b"phone".to_vec()));
    }

    #[test]
    fn test_phonetic_transducer_with_distance() {
        let dict = DoubleArrayTrieChar::from_terms(["phone", "phones", "phoned"]);
        let nfa = compile(&parse("phone").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa, 1);

        let results = transducer.query_sorted("phone");
        let terms: Vec<_> = results.iter().map(|c| c.term.as_str()).collect();

        // Should find "phone" (exact) and possibly "phones"/"phoned" (distance 1)
        assert!(terms.contains(&"phone"));
    }

    #[test]
    fn test_phonetic_transducer_accessors() {
        let dict = DoubleArrayTrieChar::from_terms(["test"]);
        let nfa = compile(&parse("test").expect("parse")).expect("compile");

        let transducer = PhoneticTransducerChar::new(dict, nfa.clone(), 2);

        assert_eq!(transducer.max_distance(), 2);
        assert!(!transducer.dictionary().is_empty());

        // Test into_dictionary
        let _recovered_dict = transducer.into_dictionary();
    }

    /// F2: a non-zero phonetic weight is stored but INERT. Pin the documented
    /// contract so the docs and code cannot silently drift apart: every emitted
    /// candidate must report `phonetic_cost == 0.0` and, consequently,
    /// `total_cost == edit_distance`.
    #[test]
    fn test_phonetic_weight_leaves_candidate_cost_zero() {
        let dict = DoubleArrayTrieChar::from_terms(["phone", "phones", "phoned"]);
        let nfa = compile(&parse("phone").expect("parse")).expect("compile");

        // Weight 7.0 is arbitrary and non-zero; it must not change any cost.
        let transducer = PhoneticTransducerChar::with_phonetic_weight(dict, nfa, 1, 7.0);
        let results: Vec<_> = transducer.query("phone").collect();
        assert!(!results.is_empty(), "expected at least the exact match");
        for candidate in &results {
            assert_eq!(
                candidate.phonetic_cost, 0.0,
                "phonetic_cost must be 0.0 for {:?}",
                candidate.term
            );
            assert_eq!(
                candidate.total_cost,
                f64::from(candidate.edit_distance),
                "total_cost must equal edit_distance for {:?}",
                candidate.term
            );
        }
    }

    /// F2 (byte-level): mirror of the char contract for `PhoneticCandidateByte`.
    #[test]
    fn test_byte_phonetic_weight_leaves_candidate_cost_zero() {
        let dict = DoubleArrayTrie::from_terms(["phone", "phones", "bone"]);
        let nfa = compile_bytes(&parse_bytes(b"phone").expect("parse")).expect("compile bytes");

        let transducer = PhoneticTransducer::with_phonetic_weight(dict, nfa, 1, 4.0);
        let results: Vec<_> = transducer.query(b"phone").collect();
        assert!(!results.is_empty(), "expected at least the exact match");
        for candidate in &results {
            assert_eq!(candidate.phonetic_cost, 0.0);
            assert_eq!(candidate.total_cost, f64::from(candidate.edit_distance));
        }
    }
}
