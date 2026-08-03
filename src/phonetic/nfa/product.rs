//! Product Automaton: NFA × Levenshtein for fuzzy regex matching.
//!
//! This module implements the product automaton that combines phonetic NFAs
//! with Levenshtein automata for fuzzy regular expression matching.
//!
//! # Fuzzy Regular Expressions
//!
//! A fuzzy regex matches strings that are:
//! - Phonetically similar (via NFA rewrites: `ph → f`, `c → s / _[ei]`)
//! - Within edit distance (via Levenshtein: insertions, deletions, substitutions)
//!
//! # Product Automaton
//!
//! The product automaton `NFA × Lev` has:
//! - **States**: `(nfa_state_set, input_position, edit_distance)`
//! - **Transitions**: Combined NFA and Levenshtein transitions
//! - **Acceptance**: NFA accepts AND edit distance ≤ max
//!
//! # Examples
//!
//! ```ignore
//! use liblevenshtein::phonetic::nfa::product::ProductAutomatonChar;
//! use liblevenshtein::phonetic::nfa::compile;
//! use liblevenshtein::phonetic::regex::parse;
//!
//! // Compile phonetic pattern: matches "phone" or "fone"
//! let nfa = compile(&parse("(ph|f)one").expect("doc: regex parse must succeed")).expect("doc: nfa compile must succeed");
//!
//! // Create product automaton with max edit distance 2
//! let product = ProductAutomatonChar::new(nfa, 2);
//!
//! // Test fuzzy matching
//! assert!(product.accepts("phone"));   // exact NFA match
//! assert!(product.accepts("fone"));    // NFA alternate
//! assert!(product.accepts("phones"));  // NFA match + insertion
//! assert!(product.accepts("fon"));     // NFA alternate + deletion
//! assert!(!product.accepts("xyz"));    // too far from any NFA match
//! ```

use super::state_set::StateSet;
use super::types::StateId;
use super::{NFAChar, NFA};
use crate::transducer::articulatory_costs::ArticulatoryCosts;
use crate::transducer::language::LanguageProduct;
use crate::transducer::Algorithm;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

const PRODUCT_COST_KEY_SCALE: f64 = 1_000_000.0;
const PRODUCT_SEARCH_CAPACITY_LIMIT: usize = 4096;
const I64_MIN_F64_BOUND: f64 = -9_223_372_036_854_775_808.0;
const I64_MAX_F64_BOUND: f64 = 9_223_372_036_854_775_808.0;
const F64_EXPONENT_MASK: u64 = 0x7ff;
const F64_MANTISSA_BITS: u32 = 52;
const F64_MANTISSA_MASK: u64 = (1u64 << F64_MANTISSA_BITS) - 1;
const F64_EXPONENT_BIAS: i32 = 1023;

#[inline]
fn assert_product_algorithm_supported(algorithm: Algorithm) {
    assert!(
        algorithm != Algorithm::DamerauLevenshtein,
        "Algorithm::DamerauLevenshtein is history-dependent and is not supported by the phonetic NFA product; use Algorithm::Transposition for optimal string alignment"
    );
}

#[inline]
fn product_search_capacity(input_len: usize, max_errors: u8) -> usize {
    input_len
        .saturating_add(1)
        .saturating_mul(usize::from(max_errors).saturating_add(1))
        .clamp(1, PRODUCT_SEARCH_CAPACITY_LIMIT)
}

#[inline]
fn deletion_closure_capacity(seed_count: usize, max_errors: u8) -> usize {
    if seed_count == 0 {
        0
    } else {
        seed_count
            .saturating_mul(usize::from(max_errors).saturating_add(1))
            .min(PRODUCT_SEARCH_CAPACITY_LIMIT)
    }
}

#[inline]
fn final_search_capacity(remaining_errors: u8) -> usize {
    usize::from(remaining_errors)
        .saturating_add(1)
        .min(PRODUCT_SEARCH_CAPACITY_LIMIT)
}

#[inline]
fn state_set_to_sorted_vec(states: StateSet) -> Vec<StateId> {
    let mut states_vec: Vec<StateId> = Vec::with_capacity(states.len());
    states_vec.extend(states.iter());
    states_vec.sort_unstable();
    states_vec
}

#[inline]
fn enqueue_product_search_state(
    queue: &mut VecDeque<(usize, Vec<StateId>, u8)>,
    visited: &mut FxHashSet<(usize, Vec<StateId>, u8)>,
    pos: usize,
    states: Vec<StateId>,
    errors: u8,
) {
    if visited.insert((pos, states.clone(), errors)) {
        queue.push_back((pos, states, errors));
    }
}

#[inline]
fn enqueue_final_search_state(
    queue: &mut VecDeque<(Vec<StateId>, u8)>,
    visited: &mut FxHashSet<Vec<StateId>>,
    states: Vec<StateId>,
    distance: u8,
) {
    if visited.insert(states.clone()) {
        queue.push_back((states, distance));
    }
}

#[inline]
fn max_distance_budget_len(max_distance: u8) -> usize {
    usize::from(max_distance)
}

#[inline]
fn finite_integral_f64_to_i64(value: f64) -> i64 {
    debug_assert!(value.is_finite());
    debug_assert_eq!(value.round(), value);
    debug_assert!(value > I64_MIN_F64_BOUND);
    debug_assert!(value < I64_MAX_F64_BOUND);

    if value == 0.0 {
        return 0;
    }

    let bits = value.to_bits();
    let is_negative = (bits >> 63) != 0;
    let exponent_bits =
        u16::try_from((bits >> F64_MANTISSA_BITS) & F64_EXPONENT_MASK).expect("masked exponent");

    if exponent_bits == 0 {
        return 0;
    }

    let exponent = i32::from(exponent_bits) - F64_EXPONENT_BIAS;
    if exponent < 0 {
        return 0;
    }

    let significand = (1u64 << F64_MANTISSA_BITS) | (bits & F64_MANTISSA_MASK);
    let magnitude = if exponent >= i32::try_from(F64_MANTISSA_BITS).expect("mantissa bits") {
        let shift =
            u32::try_from(exponent - i32::try_from(F64_MANTISSA_BITS).expect("mantissa bits"))
                .expect("nonnegative shift");
        u128::from(significand)
            .checked_shl(shift)
            .unwrap_or(u128::MAX)
    } else {
        let shift =
            u32::try_from(i32::try_from(F64_MANTISSA_BITS).expect("mantissa bits") - exponent)
                .expect("nonnegative shift");
        u128::from(significand >> shift)
    };

    if is_negative {
        if magnitude == (1u128 << 63) {
            i64::MIN
        } else {
            -i64::try_from(magnitude).expect("bounded negative magnitude")
        }
    } else {
        i64::try_from(magnitude).expect("bounded positive magnitude")
    }
}

#[inline]
fn finite_positive_ceil_to_u8(value: f64) -> u8 {
    debug_assert!(value.is_finite());
    debug_assert!(value > 0.0);
    debug_assert!(value < f64::from(u8::MAX));

    let mut distance = 1u8;
    while distance < u8::MAX && f64::from(distance) < value {
        distance += 1;
    }
    distance
}

#[inline]
fn product_cost_key(cost: f64) -> i64 {
    if cost.is_nan() {
        return i64::MIN;
    }
    if cost.is_infinite() {
        return if cost.is_sign_positive() {
            i64::MAX
        } else {
            i64::MIN + 1
        };
    }

    let scaled = (cost * PRODUCT_COST_KEY_SCALE).round();
    if scaled >= I64_MAX_F64_BOUND {
        i64::MAX
    } else if scaled <= I64_MIN_F64_BOUND {
        i64::MIN
    } else {
        finite_integral_f64_to_i64(scaled)
    }
}

#[inline]
fn accumulated_cost_to_u8_distance(cost: f64) -> u8 {
    if cost.is_nan() {
        return u8::MAX;
    }
    if cost <= 0.0 {
        return 0;
    }
    if cost >= f64::from(u8::MAX) {
        return u8::MAX;
    }

    finite_positive_ceil_to_u8(cost)
}

#[inline]
fn max_cost_to_u8_distance(max_cost: f64) -> u8 {
    if max_cost.is_nan() || max_cost <= 0.0 {
        return 0;
    }
    if max_cost >= f64::from(u8::MAX) {
        return u8::MAX;
    }

    finite_positive_ceil_to_u8(max_cost)
}

/// Character-level product automaton: NFA × Levenshtein.
///
/// Computes the intersection of a phonetic NFA with a Levenshtein automaton
/// to enable fuzzy pattern matching.
///
/// # Articulatory Cost Integration
///
/// When `articulatory_costs` is provided, substitution operations use
/// phonetically-informed costs based on articulatory features of IPA characters.
/// For example, `p↔b` (voicing only) costs less than `p↔k` (different place).
/// This improves phonetic correction quality for residual errors not covered
/// by explicit NFA rules.
#[derive(Debug, Clone)]
pub struct ProductAutomatonChar {
    /// The phonetic NFA
    nfa: NFAChar,
    /// Maximum accumulated cost (f64 to support fractional articulatory costs)
    max_cost: f64,
    /// Reserved phonetic weight (default: `0.0`).
    ///
    /// Currently stored but NOT applied to matching cost or ranking: no
    /// `accepts`/`min_distance`/`transition` path reads it. Tracked for a
    /// future weighted-phonetic-matching feature. Invariant: clamped to
    /// `>= 0.0` at construction so a future implementation cannot violate
    /// monotone-cost pruning (edge costs must be non-negative).
    phonetic_weight: f64,
    /// Levenshtein algorithm variant (standard, transposition, merge-and-split)
    algorithm: Algorithm,
    /// Optional articulatory costs for phonetically-informed substitutions.
    /// When `Some`, substitution cost depends on articulatory distance between
    /// input and pattern characters. When `None`, fixed cost (1.0) is used.
    articulatory_costs: Option<ArticulatoryCosts>,
}

/// A state in the product automaton.
///
/// Represents a configuration during fuzzy matching:
/// - Which NFA states are active (after epsilon closure)
/// - Accumulated cost of edit operations
///
/// # Note on Float Costs
///
/// The `accumulated_cost` field uses `f64` to support fractional articulatory
/// costs (e.g., `p↔b` might cost 0.1 while `p↔k` costs 0.6). For comparison
/// and hashing, costs are mapped to a fixed micro-cost key so equality remains
/// transitive and consistent with `Hash`.
#[derive(Debug, Clone)]
pub struct ProductStateChar {
    /// Set of active NFA states (after epsilon closure)
    pub nfa_states: Vec<StateId>,
    /// Accumulated cost of edit operations (supports fractional articulatory costs)
    pub accumulated_cost: f64,
}

impl ProductStateChar {
    /// Create a new product state.
    pub fn new(nfa_states: FxHashSet<StateId>, accumulated_cost: f64) -> Self {
        let mut states: Vec<StateId> = nfa_states.into_iter().collect();
        states.sort_unstable(); // Canonical ordering for equality comparison
        Self {
            nfa_states: states,
            accumulated_cost,
        }
    }

    fn from_state_ids(nfa_states: &[StateId], accumulated_cost: f64) -> Self {
        let is_canonical = nfa_states.windows(2).all(|pair| pair[0] < pair[1]);
        if is_canonical {
            return Self {
                nfa_states: nfa_states.to_vec(),
                accumulated_cost,
            };
        }

        let mut states = nfa_states.to_vec();
        states.sort_unstable();
        states.dedup();
        Self {
            nfa_states: states,
            accumulated_cost,
        }
    }

    /// Create a new product state with integer edit distance (backwards compatibility).
    pub fn with_edit_distance(nfa_states: FxHashSet<StateId>, edit_distance: u8) -> Self {
        Self::new(nfa_states, edit_distance as f64)
    }

    /// Get the edit distance as an integer (rounds up for fractional costs).
    pub fn edit_distance(&self) -> u8 {
        accumulated_cost_to_u8_distance(self.accumulated_cost)
    }
}

impl PartialEq for ProductStateChar {
    fn eq(&self, other: &Self) -> bool {
        self.nfa_states == other.nfa_states
            && product_cost_key(self.accumulated_cost) == product_cost_key(other.accumulated_cost)
    }
}

impl Eq for ProductStateChar {}

impl std::hash::Hash for ProductStateChar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.nfa_states.hash(state);
        // Hash the cost with fixed precision to avoid floating-point issues
        product_cost_key(self.accumulated_cost).hash(state);
    }
}

impl ProductAutomatonChar {
    fn empty_pattern_distance(input: &str) -> usize {
        input.chars().count()
    }

    /// Create a new product automaton with default settings.
    ///
    /// # Arguments
    ///
    /// * `nfa` - The phonetic NFA pattern
    /// * `max_distance` - Maximum edit distance allowed
    pub fn new(nfa: NFAChar, max_distance: u8) -> Self {
        Self {
            nfa,
            max_cost: max_distance as f64,
            phonetic_weight: 0.0,
            algorithm: Algorithm::Standard,
            articulatory_costs: None,
        }
    }

    /// Create a product automaton with a specific algorithm.
    ///
    /// # Arguments
    ///
    /// * `nfa` - The phonetic NFA pattern
    /// * `max_distance` - Maximum edit distance allowed
    /// * `algorithm` - The Levenshtein algorithm variant to use
    pub fn with_algorithm(nfa: NFAChar, max_distance: u8, algorithm: Algorithm) -> Self {
        assert_product_algorithm_supported(algorithm);
        Self {
            nfa,
            max_cost: max_distance as f64,
            phonetic_weight: 0.0,
            algorithm,
            articulatory_costs: None,
        }
    }

    /// Create a product automaton with a phonetic weight.
    ///
    /// Reserved: the phonetic weight is currently stored but NOT applied to
    /// matching cost or ranking; no `accepts`/`min_distance`/`transition` path
    /// reads it, and phonetic-transducer candidates always report
    /// `phonetic_cost == 0.0`. Tracked for a future weighted-phonetic-matching
    /// feature. The value is clamped to `>= 0.0` so a future implementation
    /// cannot violate monotone-cost pruning (edge costs must be non-negative).
    pub fn with_phonetic_weight(nfa: NFAChar, max_distance: u8, phonetic_weight: f64) -> Self {
        Self {
            nfa,
            max_cost: max_distance as f64,
            phonetic_weight: phonetic_weight.max(0.0),
            algorithm: Algorithm::Standard,
            articulatory_costs: None,
        }
    }

    /// Create a product automaton with both algorithm and phonetic weight.
    ///
    /// Reserved: as with [`Self::with_phonetic_weight`], the phonetic weight is
    /// stored but NOT applied to matching cost or ranking (`phonetic_cost` is
    /// always `0.0`). It is clamped to `>= 0.0` to preserve the future
    /// monotone-cost-pruning invariant. Tracked for a future
    /// weighted-phonetic-matching feature.
    pub fn with_algorithm_and_weight(
        nfa: NFAChar,
        max_distance: u8,
        algorithm: Algorithm,
        phonetic_weight: f64,
    ) -> Self {
        assert_product_algorithm_supported(algorithm);
        Self {
            nfa,
            max_cost: max_distance as f64,
            phonetic_weight: phonetic_weight.max(0.0),
            algorithm,
            articulatory_costs: None,
        }
    }

    /// Create a product automaton with articulatory costs for phonetically-informed
    /// substitution weighting.
    ///
    /// # Arguments
    ///
    /// * `nfa` - The phonetic NFA pattern
    /// * `max_cost` - Maximum accumulated cost allowed (sum of all edit operations)
    /// * `algorithm` - The Levenshtein algorithm variant to use
    /// * `articulatory_costs` - Articulatory cost configuration for substitutions
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::phonetic::nfa::product::ProductAutomatonChar;
    /// use liblevenshtein::transducer::{Algorithm, ArticulatoryCosts};
    ///
    /// let costs = ArticulatoryCosts::default();
    /// let product = ProductAutomatonChar::with_articulatory_costs(
    ///     nfa,
    ///     2.0,  // max accumulated cost
    ///     Algorithm::Standard,
    ///     costs,
    /// );
    /// ```
    pub fn with_articulatory_costs(
        nfa: NFAChar,
        max_cost: f64,
        algorithm: Algorithm,
        articulatory_costs: ArticulatoryCosts,
    ) -> Self {
        assert_product_algorithm_supported(algorithm);
        // F7: reject cost configurations that would break monotone-cost pruning
        // (negative/NaN/infinite operation costs, or a non-zero match cost).
        // `ArticulatoryCosts::is_valid()` transitively checks its base
        // `OperationCostsF64::is_valid()`. Debug-only so valid callers pay
        // nothing in release builds.
        debug_assert!(
            articulatory_costs.is_valid(),
            "articulatory costs must be finite, non-negative, with match_cost == 0.0: {articulatory_costs}"
        );
        Self {
            nfa,
            max_cost,
            phonetic_weight: 0.0,
            algorithm,
            articulatory_costs: Some(articulatory_costs),
        }
    }

    /// Get the maximum cost threshold.
    pub fn max_cost(&self) -> f64 {
        self.max_cost
    }

    /// Get the maximum edit distance (integer, for backwards compatibility).
    pub fn max_distance(&self) -> u8 {
        max_cost_to_u8_distance(self.max_cost)
    }

    /// Get the articulatory costs, if configured.
    pub fn articulatory_costs(&self) -> Option<&ArticulatoryCosts> {
        self.articulatory_costs.as_ref()
    }

    /// Get the (reserved) phonetic weight.
    ///
    /// Returns the stored, clamped (`>= 0.0`) weight. Note this value is NOT
    /// applied to matching cost or ranking in the current implementation.
    pub fn phonetic_weight(&self) -> f64 {
        self.phonetic_weight
    }

    /// Get the algorithm variant.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Borrow the language NFA used by this product.
    ///
    /// Exposed within the crate so dictionary intersections can delegate to the
    /// generic language-product walker without cloning product internals.
    pub(crate) fn nfa(&self) -> &NFAChar {
        &self.nfa
    }

    /// Returns true when this product automaton can be advanced incrementally
    /// over a trie frontier without falling back to full-string matching.
    ///
    /// The incremental API currently mirrors the standard integer-cost
    /// Levenshtein semantics used by [`Self::min_distance`]. Algorithms that
    /// need adjacent input lookahead, such as transposition and merge/split,
    /// keep using the exact full-string matcher.
    pub fn supports_incremental_trie_intersection(&self) -> bool {
        self.algorithm == Algorithm::Standard && self.articulatory_costs.is_none()
    }

    /// Get the initial state of the product automaton.
    ///
    /// The initial state is the epsilon closure of the NFA start state
    /// with 0 accumulated cost.
    pub fn initial_state(&self) -> ProductStateChar {
        let initial_closure: FxHashSet<StateId> =
            self.nfa.epsilon_closure_single(self.nfa.start()).into();
        ProductStateChar::new(initial_closure, 0.0)
    }

    /// Get the initial frontier for incremental trie/product traversal.
    ///
    /// This includes deletion closure from the regex side, matching the cases
    /// where the pattern can advance before any dictionary character is
    /// consumed.
    pub fn initial_frontier(&self) -> Vec<ProductStateChar> {
        self.deletion_closure(vec![self.initial_state()])
    }

    /// Advance an incremental product frontier by one dictionary character.
    ///
    /// The returned frontier contains all product states reachable after
    /// consuming `c`, including deletion closure from the regex side. An empty
    /// frontier means no completion below the current trie edge can match within
    /// the configured distance bound.
    pub fn transition_frontier(
        &self,
        frontier: &[ProductStateChar],
        c: char,
    ) -> Vec<ProductStateChar> {
        let mut next = Vec::with_capacity(frontier.len().saturating_mul(3));

        for state in frontier {
            if state.accumulated_cost > self.max_cost {
                continue;
            }

            let match_states = self.nfa_step_ids(&state.nfa_states, c);
            if !match_states.is_empty() {
                next.push(ProductStateChar::new(match_states, state.accumulated_cost));
            }

            if state.accumulated_cost < self.max_cost {
                let next_cost = state.accumulated_cost + 1.0;

                let subst_states = self.nfa_advance_ids(&state.nfa_states);
                if !subst_states.is_empty() && next_cost <= self.max_cost {
                    next.push(ProductStateChar::new(subst_states, next_cost));
                }

                if next_cost <= self.max_cost {
                    next.push(ProductStateChar::from_state_ids(
                        &state.nfa_states,
                        next_cost,
                    ));
                }
            }
        }

        self.deletion_closure(next)
    }

    /// Minimum accepting distance represented by an incremental frontier.
    pub fn min_accepting_distance(&self, frontier: &[ProductStateChar]) -> Option<u8> {
        frontier
            .iter()
            .filter(|state| self.is_accepting(state))
            .map(ProductStateChar::edit_distance)
            .min()
    }

    fn deletion_closure(&self, seeds: Vec<ProductStateChar>) -> Vec<ProductStateChar> {
        let seed_count = seeds.len();
        let capacity = deletion_closure_capacity(seed_count, self.max_distance());
        let mut visited: FxHashSet<ProductStateChar> = FxHashSet::default();
        visited.reserve(capacity);
        let mut queue = VecDeque::with_capacity(capacity);
        queue.extend(seeds);
        let mut closure = Vec::with_capacity(capacity);

        while let Some(state) = queue.pop_front() {
            if state.accumulated_cost > self.max_cost {
                continue;
            }
            if !visited.insert(state.clone()) {
                continue;
            }

            if state.accumulated_cost < self.max_cost {
                let deleted_states = self.nfa_advance_ids(&state.nfa_states);
                let next_cost = state.accumulated_cost + 1.0;
                if !deleted_states.is_empty() && next_cost <= self.max_cost {
                    queue.push_back(ProductStateChar::new(deleted_states, next_cost));
                }
            }

            closure.push(state);
        }

        closure
    }

    /// Check if a product state is accepting.
    ///
    /// A state is accepting if:
    /// 1. At least one NFA state is final
    /// 2. Accumulated cost is within the maximum
    pub fn is_accepting(&self, state: &ProductStateChar) -> bool {
        if state.accumulated_cost > self.max_cost {
            return false;
        }

        // Check if any NFA state is final
        state.nfa_states.iter().any(|&s| self.nfa.is_final(s))
    }

    /// Compute the substitution cost between input char and pattern char.
    ///
    /// If articulatory costs are configured, uses phonetically-informed costs.
    /// Otherwise, uses fixed cost (1.0).
    #[inline]
    fn substitution_cost(&self, input_char: char, pattern_char: Option<char>) -> f64 {
        match (&self.articulatory_costs, pattern_char) {
            (Some(costs), Some(pc)) => costs.substitution_cost(input_char, pc),
            _ => 1.0, // Fixed cost when no articulatory costs or no specific pattern char
        }
    }

    /// Get the insertion cost (currently fixed at 1.0).
    #[inline]
    fn insertion_cost(&self) -> f64 {
        match &self.articulatory_costs {
            Some(costs) => costs.insertion_cost(),
            None => 1.0,
        }
    }

    /// Compute successor states after consuming a character.
    ///
    /// This implements the product transition function:
    /// - For each active NFA state, try matching the input character
    /// - Also consider edit operations (insertion, deletion, substitution)
    ///
    /// # Articulatory Costs
    ///
    /// When articulatory costs are configured, substitution operations use
    /// phonetically-informed costs based on IPA features. For example:
    /// - `p↔b` (voicing only) might cost 0.1
    /// - `p↔k` (different place) might cost 0.6
    ///
    /// # Arguments
    ///
    /// * `state` - Current product state
    /// * `c` - Input character to process
    ///
    /// # Returns
    ///
    /// Set of successor states reachable via matching or edit operations.
    pub fn transition(&self, state: &ProductStateChar, c: char) -> Vec<ProductStateChar> {
        let mut successors = Vec::with_capacity(if state.accumulated_cost < self.max_cost {
            3
        } else {
            1
        });

        // 1. NFA Match: consume character in NFA (no cost)
        let match_states = self.nfa_step_ids(&state.nfa_states, c);
        if !match_states.is_empty() {
            successors.push(ProductStateChar::new(match_states, state.accumulated_cost));
        }

        // Edit operations only if we have budget
        if state.accumulated_cost < self.max_cost {
            // 2. Substitution: NFA doesn't match, consume with articulatory cost
            // For each active state, try all transitions with character-specific cost
            for &nfa_state in &state.nfa_states {
                for trans in self.nfa.transitions_from(nfa_state) {
                    if trans.label.consumes_input() {
                        // Compute articulatory cost based on pattern character
                        let pattern_char = trans.label.expected_char();
                        let sub_cost = self.substitution_cost(c, pattern_char);
                        let new_cost = state.accumulated_cost + sub_cost;

                        // Only add if within budget
                        if new_cost <= self.max_cost {
                            let closure = self.nfa.epsilon_closure_single(trans.to);
                            let closure_set: FxHashSet<StateId> = closure.into();
                            if !closure_set.is_empty() {
                                let subst_state = ProductStateChar::new(closure_set, new_cost);
                                if !successors.contains(&subst_state) {
                                    successors.push(subst_state);
                                }
                            }
                        }
                    }
                }
            }

            // 3. Insertion: input has extra char, NFA stays in place
            let ins_cost = state.accumulated_cost + self.insertion_cost();
            if ins_cost <= self.max_cost {
                successors.push(ProductStateChar::from_state_ids(
                    &state.nfa_states,
                    ins_cost,
                ));
            }

            // 4. Deletion: NFA pattern has extra char, advance NFA without consuming input
            // This is handled differently - we advance NFA states via epsilon transitions
            // But we need to also advance via character transitions (pretend we saw the char)
            // Actually, deletion in Levenshtein means the pattern has something the input doesn't
            // So we advance NFA and don't consume input - this is like "free" NFA transitions
            // We'll handle this in the accepts() method by trying to advance NFA with budget
        }

        successors
    }

    fn nfa_step_ids(&self, states: &[StateId], c: char) -> FxHashSet<StateId> {
        self.nfa_step_state_set(states.iter().copied(), c).into()
    }

    fn nfa_step_vec(&self, states: &[StateId], c: char) -> Vec<StateId> {
        state_set_to_sorted_vec(self.nfa_step_state_set(states.iter().copied(), c))
    }

    fn nfa_step_state_set<I>(&self, states: I, c: char) -> StateSet
    where
        I: IntoIterator<Item = StateId>,
    {
        let mut next_states = StateSet::with_capacity(self.nfa.num_states());

        for state in states {
            for trans in self.nfa.transitions_from(state) {
                if trans.label.matches(c) && trans.label.consumes_input() {
                    next_states.insert(trans.to);
                }
            }
        }

        self.nfa.epsilon_closure(&next_states)
    }

    /// Check if the input string is accepted by the fuzzy regex.
    ///
    /// Uses BFS to explore the product state space, pruning states
    /// that exceed the cost budget.
    ///
    /// # Note
    ///
    /// This method uses integer error tracking for efficiency. For full
    /// articulatory cost support, use the `transition()` method in a
    /// custom traversal loop.
    ///
    /// # Arguments
    ///
    /// * `input` - Input string to match
    ///
    /// # Returns
    ///
    /// `true` if input matches the NFA pattern within cost threshold.
    pub fn accepts(&self, input: &str) -> bool {
        // Early exit: empty pattern
        if self.nfa.is_empty() {
            return Self::empty_pattern_distance(input)
                <= max_distance_budget_len(self.max_distance());
        }

        let input_chars: Vec<char> = input.chars().collect();
        let n = input_chars.len();
        let max_errors = self.max_distance(); // Use integer approximation

        // BFS state: (nfa_states, input_position, edit_distance)
        let initial_closure =
            state_set_to_sorted_vec(self.nfa.epsilon_closure_single(self.nfa.start()));

        // Use dynamic programming / BFS
        // State: (position in input, set of NFA states, edit distance used)
        let search_capacity = product_search_capacity(n, max_errors);
        let mut visited: FxHashSet<(usize, Vec<StateId>, u8)> = FxHashSet::default();
        visited.reserve(search_capacity);
        let mut queue: VecDeque<(usize, Vec<StateId>, u8)> =
            VecDeque::with_capacity(search_capacity);

        enqueue_product_search_state(&mut queue, &mut visited, 0, initial_closure, 0);

        while let Some((pos, nfa_states, errors)) = queue.pop_front() {
            // Prune if over budget
            if errors > max_errors {
                continue;
            }

            // Check acceptance: at end of input and NFA accepts
            if pos == n {
                // Check if we can reach final state with remaining budget
                if self.can_reach_final_vec(&nfa_states, errors, max_errors) {
                    return true;
                }
                continue;
            }

            let c = input_chars[pos];

            // 1. Match: NFA consumes character, no error
            let match_states = self.nfa_step_vec(&nfa_states, c);
            if !match_states.is_empty() {
                enqueue_product_search_state(
                    &mut queue,
                    &mut visited,
                    pos + 1,
                    match_states,
                    errors,
                );
            }

            if errors < max_errors {
                let advanced_states = self.nfa_advance_vec(&nfa_states);

                // 2. Substitution: NFA advances (any transition), +1 error
                if !advanced_states.is_empty() {
                    enqueue_product_search_state(
                        &mut queue,
                        &mut visited,
                        pos + 1,
                        advanced_states.clone(),
                        errors + 1,
                    );
                }

                // 3. Insertion: Stay in NFA, consume input char, +1 error
                enqueue_product_search_state(
                    &mut queue,
                    &mut visited,
                    pos + 1,
                    nfa_states.clone(),
                    errors + 1,
                );

                // 4. Deletion: Advance NFA without consuming input, +1 error
                if !advanced_states.is_empty() {
                    enqueue_product_search_state(
                        &mut queue,
                        &mut visited,
                        pos,
                        advanced_states.clone(),
                        errors + 1,
                    );
                }

                // 5. Transposition: swap adjacent characters (e.g., "ab" → "ba")
                if self.algorithm.supports_transposition() && pos + 1 < n {
                    let next_c = input_chars[pos + 1];
                    let trans_states = self.nfa_step_transposed_vec(&nfa_states, c, next_c);
                    if !trans_states.is_empty() {
                        enqueue_product_search_state(
                            &mut queue,
                            &mut visited,
                            pos + 2,
                            trans_states,
                            errors + 1,
                        );
                    }
                }

                // 6. Merge: two adjacent input chars → one NFA edge, cost 1.
                //    Generic Merge-and-Split: merge is character-agnostic by
                //    definition (any two adjacent symbols merge to one), so the
                //    symbols are not validated. See `nfa_step_merged_from_advance_vec`.
                if self.algorithm.supports_merge_split() && pos + 1 < n {
                    let next_c = input_chars[pos + 1];
                    if !advanced_states.is_empty() {
                        let merge_states =
                            self.nfa_step_merged_from_advance_vec(&advanced_states, c, next_c);
                        if !merge_states.is_empty() {
                            enqueue_product_search_state(
                                &mut queue,
                                &mut visited,
                                pos + 2,
                                merge_states,
                                errors + 1,
                            );
                        }
                    }
                }

                // 7. Split: one input char → two NFA edges, cost 1.
                //    Generic Merge-and-Split: split is character-agnostic by
                //    definition (any symbol splits to two), so it is not validated
                //    against either edge. See `nfa_step_split_from_advance_vec`.
                if self.algorithm.supports_merge_split() && !advanced_states.is_empty() {
                    let split_states = self.nfa_step_split_from_advance_vec(&advanced_states, c);
                    if !split_states.is_empty() {
                        enqueue_product_search_state(
                            &mut queue,
                            &mut visited,
                            pos + 1,
                            split_states,
                            errors + 1,
                        );
                    }
                }
            }
        }

        false
    }

    fn nfa_advance_ids(&self, states: &[StateId]) -> FxHashSet<StateId> {
        self.nfa_advance_state_set(states.iter().copied()).into()
    }

    fn nfa_advance_vec(&self, states: &[StateId]) -> Vec<StateId> {
        state_set_to_sorted_vec(self.nfa_advance_state_set(states.iter().copied()))
    }

    fn nfa_advance_state_set<I>(&self, states: I) -> StateSet
    where
        I: IntoIterator<Item = StateId>,
    {
        let mut next_states = StateSet::with_capacity(self.nfa.num_states());

        for state in states {
            for trans in self.nfa.transitions_from(state) {
                if trans.label.consumes_input() {
                    next_states.insert(trans.to);
                }
            }
        }

        self.nfa.epsilon_closure(&next_states)
    }

    /// Step NFA with transposed characters.
    ///
    /// For transposition, we match c2 first, then c1 (swapped order).
    /// This handles cases like "ab" → "ba" where the input has the characters
    /// in transposed order relative to the pattern.
    fn nfa_step_transposed_vec(&self, states: &[StateId], c1: char, c2: char) -> Vec<StateId> {
        let after_c2 = self.nfa_step_vec(states, c2);
        self.nfa_step_vec(&after_c2, c1)
    }

    /// Merge step for `Algorithm::MergeAndSplit`.
    ///
    /// In the Merge-and-Split edit metric, `merge` collapses two adjacent input
    /// characters into one at cost 1 as an *unconstrained structural operation*:
    /// the metric imposes no relation on the specific characters (any two adjacent
    /// input chars may merge onto any single pattern edge). `_c1`/`_c2` are
    /// therefore correctly ignored — the operation is character-agnostic *by
    /// definition* — and the pair collapses onto the one edge already taken by
    /// `advanced_states` (via `nfa_advance`). This is *exact* for the metric, which
    /// is deliberately more permissive than plain Levenshtein (the raison d'être of
    /// merge/split), and never misses a match. (Transposition differs: the swap
    /// `c1c2`↔`c2c1` is unambiguous, so [`Self::nfa_step_transposed_vec`] validates
    /// its characters.) A character-*constrained* variant — e.g. an OCR/phonetic
    /// confusion table such as `rn`↔`m` — would require a merge-relation map and is
    /// a separate feature, not part of this generic algorithm.
    fn nfa_step_merged_from_advance_vec(
        &self,
        advanced_states: &[StateId],
        _c1: char,
        _c2: char,
    ) -> Vec<StateId> {
        advanced_states.to_vec()
    }

    /// Split step for `Algorithm::MergeAndSplit`.
    ///
    /// `split` covers one input character with two consuming NFA edges at cost 1,
    /// the dual of `merge`. Like merge it is an *unconstrained structural operation*
    /// of the generic Merge-and-Split metric: any single input char may split across
    /// any two pattern edges, so `_c` is correctly ignored (character-agnostic by
    /// definition). `advanced_states` is the first `nfa_advance`; this applies the
    /// second. Exact for the metric; never misses a match.
    fn nfa_step_split_from_advance_vec(
        &self,
        advanced_states: &[StateId],
        _c: char,
    ) -> Vec<StateId> {
        self.nfa_advance_vec(advanced_states)
    }

    /// Check if we can reach a final state with remaining error budget.
    fn can_reach_final_vec(&self, states: &[StateId], current_errors: u8, max_errors: u8) -> bool {
        // If any current state is final, we're done
        if states.iter().any(|&s| self.nfa.is_final(s)) {
            return true;
        }

        // Otherwise, try to reach final with remaining budget
        let remaining = max_errors.saturating_sub(current_errors);
        if remaining == 0 {
            return false;
        }

        // BFS to find path to final within remaining budget
        let search_capacity = final_search_capacity(remaining);
        let mut visited: FxHashSet<Vec<StateId>> = FxHashSet::default();
        visited.reserve(search_capacity);
        let mut queue: VecDeque<(Vec<StateId>, u8)> = VecDeque::with_capacity(search_capacity);
        enqueue_final_search_state(&mut queue, &mut visited, states.to_vec(), 0);

        while let Some((current, dist)) = queue.pop_front() {
            if dist > remaining {
                continue;
            }

            if current.iter().any(|&s| self.nfa.is_final(s)) {
                return true;
            }

            // Advance NFA (deletion from pattern)
            let next = self.nfa_advance_vec(&current);
            if !next.is_empty() {
                enqueue_final_search_state(&mut queue, &mut visited, next, dist + 1);
            }
        }

        false
    }

    /// Get the minimum edit distance to any accepting state.
    ///
    /// # Arguments
    ///
    /// * `input` - Input string to match
    ///
    /// # Returns
    ///
    /// Minimum edit distance, or `None` if no match within max_distance.
    pub fn min_distance(&self, input: &str) -> Option<u8> {
        if self.nfa.is_empty() {
            let distance = Self::empty_pattern_distance(input);
            return if distance == 0 {
                Some(0)
            } else if distance <= max_distance_budget_len(self.max_distance()) {
                u8::try_from(distance).ok()
            } else {
                None
            };
        }

        let input_chars: Vec<char> = input.chars().collect();
        let n = input_chars.len();
        let max_errors = self.max_distance();

        let initial_closure =
            state_set_to_sorted_vec(self.nfa.epsilon_closure_single(self.nfa.start()));

        let mut min_dist: Option<u8> = None;
        let search_capacity = product_search_capacity(n, max_errors);
        let mut visited: FxHashSet<(usize, Vec<StateId>, u8)> = FxHashSet::default();
        visited.reserve(search_capacity);
        let mut queue: VecDeque<(usize, Vec<StateId>, u8)> =
            VecDeque::with_capacity(search_capacity);

        enqueue_product_search_state(&mut queue, &mut visited, 0, initial_closure, 0);

        while let Some((pos, nfa_states, errors)) = queue.pop_front() {
            if errors > max_errors {
                continue;
            }

            // Skip if we already found a better solution
            if let Some(min) = min_dist {
                if errors >= min {
                    continue;
                }
            }

            if pos == n {
                if let Some(final_dist) = self.distance_to_final_vec(&nfa_states, errors) {
                    match min_dist {
                        None => min_dist = Some(final_dist),
                        Some(current) if final_dist < current => min_dist = Some(final_dist),
                        _ => {}
                    }
                }
                continue;
            }

            let c = input_chars[pos];

            // 1. Match
            let match_states = self.nfa_step_vec(&nfa_states, c);
            if !match_states.is_empty() {
                enqueue_product_search_state(
                    &mut queue,
                    &mut visited,
                    pos + 1,
                    match_states,
                    errors,
                );
            }

            if errors < max_errors {
                let advanced_states = self.nfa_advance_vec(&nfa_states);

                // 2. Substitution
                if !advanced_states.is_empty() {
                    enqueue_product_search_state(
                        &mut queue,
                        &mut visited,
                        pos + 1,
                        advanced_states.clone(),
                        errors + 1,
                    );
                }

                // 3. Insertion
                enqueue_product_search_state(
                    &mut queue,
                    &mut visited,
                    pos + 1,
                    nfa_states.clone(),
                    errors + 1,
                );

                // 4. Deletion
                if !advanced_states.is_empty() {
                    enqueue_product_search_state(
                        &mut queue,
                        &mut visited,
                        pos,
                        advanced_states.clone(),
                        errors + 1,
                    );
                }

                // 5. Transposition: swap adjacent characters (e.g., "ab" → "ba")
                if self.algorithm.supports_transposition() && pos + 1 < n {
                    let next_c = input_chars[pos + 1];
                    let trans_states = self.nfa_step_transposed_vec(&nfa_states, c, next_c);
                    if !trans_states.is_empty() {
                        enqueue_product_search_state(
                            &mut queue,
                            &mut visited,
                            pos + 2,
                            trans_states,
                            errors + 1,
                        );
                    }
                }

                // 6. Merge: two adjacent input chars → one NFA edge, cost 1.
                //    Generic Merge-and-Split: merge is character-agnostic by
                //    definition (any two adjacent symbols merge to one), so the
                //    symbols are not validated. See `nfa_step_merged_from_advance_vec`.
                if self.algorithm.supports_merge_split() && pos + 1 < n {
                    let next_c = input_chars[pos + 1];
                    if !advanced_states.is_empty() {
                        let merge_states =
                            self.nfa_step_merged_from_advance_vec(&advanced_states, c, next_c);
                        if !merge_states.is_empty() {
                            enqueue_product_search_state(
                                &mut queue,
                                &mut visited,
                                pos + 2,
                                merge_states,
                                errors + 1,
                            );
                        }
                    }
                }

                // 7. Split: one input char → two NFA edges, cost 1.
                //    Generic Merge-and-Split: split is character-agnostic by
                //    definition (any symbol splits to two), so it is not validated
                //    against either edge. See `nfa_step_split_from_advance_vec`.
                if self.algorithm.supports_merge_split() && !advanced_states.is_empty() {
                    let split_states = self.nfa_step_split_from_advance_vec(&advanced_states, c);
                    if !split_states.is_empty() {
                        enqueue_product_search_state(
                            &mut queue,
                            &mut visited,
                            pos + 1,
                            split_states,
                            errors + 1,
                        );
                    }
                }
            }
        }

        min_dist
    }

    /// Compute distance to reach final state from current states.
    fn distance_to_final_vec(&self, states: &[StateId], base_dist: u8) -> Option<u8> {
        if states.iter().any(|&s| self.nfa.is_final(s)) {
            return Some(base_dist);
        }

        let remaining = self.max_distance().saturating_sub(base_dist);
        if remaining == 0 {
            return None;
        }

        let search_capacity = final_search_capacity(remaining);
        let mut visited: FxHashSet<Vec<StateId>> = FxHashSet::default();
        visited.reserve(search_capacity);
        let mut queue: VecDeque<(Vec<StateId>, u8)> = VecDeque::with_capacity(search_capacity);
        enqueue_final_search_state(&mut queue, &mut visited, states.to_vec(), 0);

        while let Some((current, dist)) = queue.pop_front() {
            if dist > remaining {
                continue;
            }

            if current.iter().any(|&s| self.nfa.is_final(s)) {
                return Some(base_dist + dist);
            }

            let next = self.nfa_advance_vec(&current);
            if !next.is_empty() {
                enqueue_final_search_state(&mut queue, &mut visited, next, dist + 1);
            }
        }

        None
    }

    /// Deletion cost (skip a pattern symbol). Fixed `1.0` unless articulatory
    /// costs are configured; mirrors [`insertion_cost`](Self::insertion_cost).
    #[inline]
    fn deletion_cost(&self) -> f64 {
        match &self.articulatory_costs {
            Some(costs) => costs.deletion_cost(),
            None => 1.0,
        }
    }

    /// Transposition cost (swap two adjacent input symbols). Fixed `1.0` unless
    /// articulatory costs are configured.
    #[inline]
    fn transposition_cost(&self) -> f64 {
        match &self.articulatory_costs {
            Some(costs) => costs.transposition_cost(),
            None => 1.0,
        }
    }

    /// Merge cost (two input symbols collapse onto one pattern edge). Fixed
    /// `1.0` unless articulatory costs are configured.
    #[inline]
    fn merge_cost(&self) -> f64 {
        match &self.articulatory_costs {
            Some(costs) => costs.merge_cost(),
            None => 1.0,
        }
    }

    /// Split cost (one input symbol splits across two pattern edges). Fixed
    /// `1.0` unless articulatory costs are configured.
    #[inline]
    fn split_cost(&self) -> f64 {
        match &self.articulatory_costs {
            Some(costs) => costs.split_cost(),
            None => 1.0,
        }
    }

    /// Minimum *articulatory-weighted* cost to accept `input`, or `None` if the
    /// least-cost alignment exceeds `max_cost`.
    ///
    /// This is the fractional twin of [`min_distance`](Self::min_distance): it
    /// runs the same seven product operations (match, substitution, insertion,
    /// deletion, transposition, merge, split) but accumulates a real-valued
    /// cost, where a substitution is charged its *articulatory* (feature)
    /// distance when [`ArticulatoryCosts`] are configured — so a phonetically
    /// near substitution (e.g. a voiced/voiceless pair) costs a fraction of a
    /// full edit. With no articulatory costs configured every operation costs
    /// `1.0` and the result equals `min_distance` cast to `f64`.
    ///
    /// # Algorithm
    ///
    /// A uniform-cost (Dijkstra) search over product states `(position, NFA
    /// state set)`. Every edit edge carries a non-negative cost, so the first
    /// accepting configuration popped from the min-priority frontier is
    /// provably optimal. `deletion` advances the NFA past a pattern symbol
    /// *without* consuming input at any position, which subsumes the "reach a
    /// final state through trailing deletions" case that `min_distance` handles
    /// with a separate `distance_to_final_vec` walk. The frontier is ordered by
    /// `product_cost_key`, a monotone `f64 -> i64` map, which also keys the
    /// closed set.
    ///
    /// The returned cost is `<= min_distance(input)` — substitutions are only
    /// ever discounted, never inflated, relative to the unit-cost model.
    pub fn min_cost(&self, input: &str) -> Option<f64> {
        // Local min-priority frontier node, ordered solely by `cost_key`.
        struct CostFrontierNode {
            cost_key: i64,
            cost: f64,
            pos: usize,
            states: Vec<StateId>,
        }
        impl PartialEq for CostFrontierNode {
            fn eq(&self, other: &Self) -> bool {
                self.cost_key == other.cost_key
            }
        }
        impl Eq for CostFrontierNode {}
        impl PartialOrd for CostFrontierNode {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for CostFrontierNode {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.cost_key.cmp(&other.cost_key)
            }
        }

        let ins_cost = self.insertion_cost();
        let del_cost = self.deletion_cost();
        // Small tolerance so an alignment whose cost lands exactly on the budget
        // is not rejected by floating-point rounding.
        let budget = self.max_cost + 1e-9;

        if self.nfa.is_empty() {
            // Empty pattern: the only alignment inserts every input character.
            let cost = input.chars().count() as f64 * ins_cost;
            return (cost <= budget).then_some(cost);
        }

        let input_chars: Vec<char> = input.chars().collect();
        let n = input_chars.len();

        let initial = state_set_to_sorted_vec(self.nfa.epsilon_closure_single(self.nfa.start()));

        let mut frontier: BinaryHeap<Reverse<CostFrontierNode>> = BinaryHeap::new();
        // Dijkstra closed set: least cost-key already settled per (pos, states).
        let mut best: FxHashMap<(usize, Vec<StateId>), i64> = FxHashMap::default();

        frontier.push(Reverse(CostFrontierNode {
            cost_key: product_cost_key(0.0),
            cost: 0.0,
            pos: 0,
            states: initial,
        }));

        while let Some(Reverse(node)) = frontier.pop() {
            let CostFrontierNode {
                cost, pos, states, ..
            } = node;
            if cost > budget {
                continue;
            }

            // Skip if this configuration was already settled at an equal-or-lower cost.
            let cost_key = product_cost_key(cost);
            let key = (pos, states.clone());
            match best.get(&key) {
                Some(&settled) if settled <= cost_key => continue,
                _ => {
                    best.insert(key, cost_key);
                }
            }

            // Accept: all input consumed AND some NFA state is final. Dijkstra
            // guarantees this first accepting pop is the global minimum.
            if pos == n && states.iter().any(|&s| self.nfa.is_final(s)) {
                return Some(cost);
            }

            // Deletion: advance the NFA past one pattern symbol, input position
            // unchanged. Available at every position (including `pos == n`),
            // which is how trailing pattern symbols reach a final state.
            let advanced = self.nfa_advance_vec(&states);
            if !advanced.is_empty() {
                let next_cost = cost + del_cost;
                if next_cost <= budget {
                    frontier.push(Reverse(CostFrontierNode {
                        cost_key: product_cost_key(next_cost),
                        cost: next_cost,
                        pos,
                        states: advanced.clone(),
                    }));
                }
            }

            if pos < n {
                let c = input_chars[pos];

                // 1. Match: consume input against a matching edge (no cost).
                let matched = self.nfa_step_vec(&states, c);
                if !matched.is_empty() {
                    frontier.push(Reverse(CostFrontierNode {
                        cost_key,
                        cost,
                        pos: pos + 1,
                        states: matched,
                    }));
                }

                // 2. Substitution: consume input against a *non*-matching edge,
                //    charged the articulatory distance to that edge's symbol.
                for &nfa_state in &states {
                    for trans in self.nfa.transitions_from(nfa_state) {
                        if trans.label.consumes_input() {
                            let sub_cost = self.substitution_cost(c, trans.label.expected_char());
                            let next_cost = cost + sub_cost;
                            if next_cost <= budget {
                                let closure = state_set_to_sorted_vec(
                                    self.nfa.epsilon_closure_single(trans.to),
                                );
                                if !closure.is_empty() {
                                    frontier.push(Reverse(CostFrontierNode {
                                        cost_key: product_cost_key(next_cost),
                                        cost: next_cost,
                                        pos: pos + 1,
                                        states: closure,
                                    }));
                                }
                            }
                        }
                    }
                }

                // 3. Insertion: consume input with no NFA move.
                let next_cost = cost + ins_cost;
                if next_cost <= budget {
                    frontier.push(Reverse(CostFrontierNode {
                        cost_key: product_cost_key(next_cost),
                        cost: next_cost,
                        pos: pos + 1,
                        states: states.clone(),
                    }));
                }

                // 4. Transposition: swap two adjacent input symbols.
                if self.algorithm.supports_transposition() && pos + 1 < n {
                    let transposed = self.nfa_step_transposed_vec(&states, c, input_chars[pos + 1]);
                    if !transposed.is_empty() {
                        let next_cost = cost + self.transposition_cost();
                        if next_cost <= budget {
                            frontier.push(Reverse(CostFrontierNode {
                                cost_key: product_cost_key(next_cost),
                                cost: next_cost,
                                pos: pos + 2,
                                states: transposed,
                            }));
                        }
                    }
                }

                // 5/6. Merge & Split (generic Merge-and-Split), both off the
                //      NFA-advanced state set.
                if self.algorithm.supports_merge_split() && !advanced.is_empty() {
                    // Merge: two input symbols collapse onto one pattern edge.
                    if pos + 1 < n {
                        let merged = self.nfa_step_merged_from_advance_vec(
                            &advanced,
                            c,
                            input_chars[pos + 1],
                        );
                        if !merged.is_empty() {
                            let next_cost = cost + self.merge_cost();
                            if next_cost <= budget {
                                frontier.push(Reverse(CostFrontierNode {
                                    cost_key: product_cost_key(next_cost),
                                    cost: next_cost,
                                    pos: pos + 2,
                                    states: merged,
                                }));
                            }
                        }
                    }
                    // Split: one input symbol splits across two pattern edges.
                    let split = self.nfa_step_split_from_advance_vec(&advanced, c);
                    if !split.is_empty() {
                        let next_cost = cost + self.split_cost();
                        if next_cost <= budget {
                            frontier.push(Reverse(CostFrontierNode {
                                cost_key: product_cost_key(next_cost),
                                cost: next_cost,
                                pos: pos + 1,
                                states: split,
                            }));
                        }
                    }
                }
            }
        }

        None
    }
}

// ============================================================================
// Byte-level Product Automaton
// ============================================================================

/// Byte-level product automaton.
#[derive(Debug, Clone)]
pub struct ProductAutomaton {
    /// The phonetic NFA
    nfa: NFA,
    /// Maximum edit distance
    max_distance: u8,
    /// Reserved phonetic weight (default: `0.0`).
    ///
    /// Currently stored but NOT applied to matching cost or ranking: no
    /// `accepts`/`min_distance` path reads it. Tracked for a future
    /// weighted-phonetic-matching feature. Invariant: clamped to `>= 0.0` at
    /// construction to preserve monotone-cost pruning (edge costs must be
    /// non-negative).
    phonetic_weight: f64,
    /// Levenshtein algorithm variant
    algorithm: Algorithm,
}

/// A state in the byte-level product automaton.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProductState {
    /// Set of active NFA states
    pub nfa_states: Vec<StateId>,
    /// Number of edit operations used
    pub edit_distance: u8,
}

impl ProductState {
    /// Create a new product state.
    pub fn new(nfa_states: FxHashSet<StateId>, edit_distance: u8) -> Self {
        let mut states: Vec<StateId> = nfa_states.into_iter().collect();
        states.sort_unstable();
        Self {
            nfa_states: states,
            edit_distance,
        }
    }

    /// Get the edit distance.
    #[inline]
    pub fn edit_distance(&self) -> u8 {
        self.edit_distance
    }
}

impl ProductAutomaton {
    /// Create a new product automaton.
    pub fn new(nfa: NFA, max_distance: u8) -> Self {
        Self {
            nfa,
            max_distance,
            phonetic_weight: 0.0,
            algorithm: Algorithm::Standard,
        }
    }

    /// Create a product automaton with a specific algorithm.
    pub fn with_algorithm(nfa: NFA, max_distance: u8, algorithm: Algorithm) -> Self {
        assert_product_algorithm_supported(algorithm);
        Self {
            nfa,
            max_distance,
            phonetic_weight: 0.0,
            algorithm,
        }
    }

    /// Create with phonetic weight.
    ///
    /// Reserved: the phonetic weight is currently stored but NOT applied to
    /// matching cost or ranking (no `accepts`/`min_distance` path reads it, and
    /// phonetic-transducer candidates always report `phonetic_cost == 0.0`).
    /// Tracked for a future weighted-phonetic-matching feature. The value is
    /// clamped to `>= 0.0` so a future implementation cannot violate
    /// monotone-cost pruning (edge costs must be non-negative).
    pub fn with_phonetic_weight(nfa: NFA, max_distance: u8, phonetic_weight: f64) -> Self {
        Self {
            nfa,
            max_distance,
            phonetic_weight: phonetic_weight.max(0.0),
            algorithm: Algorithm::Standard,
        }
    }

    /// Create a product automaton with both algorithm and phonetic weight.
    ///
    /// Reserved: as with [`Self::with_phonetic_weight`], the phonetic weight is
    /// stored but NOT applied to matching cost or ranking (`phonetic_cost` is
    /// always `0.0`). It is clamped to `>= 0.0` to preserve the future
    /// monotone-cost-pruning invariant. Tracked for a future
    /// weighted-phonetic-matching feature.
    pub fn with_algorithm_and_weight(
        nfa: NFA,
        max_distance: u8,
        algorithm: Algorithm,
        phonetic_weight: f64,
    ) -> Self {
        assert_product_algorithm_supported(algorithm);
        Self {
            nfa,
            max_distance,
            phonetic_weight: phonetic_weight.max(0.0),
            algorithm,
        }
    }

    /// Get max distance.
    pub fn max_distance(&self) -> u8 {
        self.max_distance
    }

    /// Get the (reserved) phonetic weight stored in this product automaton.
    ///
    /// Returns the stored, clamped (`>= 0.0`) weight. Note this value is NOT
    /// applied to substitutions, matching cost, or ranking in the current
    /// implementation.
    pub fn phonetic_weight(&self) -> f64 {
        self.phonetic_weight
    }

    /// Get the algorithm variant.
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get the initial state.
    pub fn initial_state(&self) -> ProductState {
        let initial_closure: FxHashSet<StateId> =
            self.nfa.epsilon_closure_single(self.nfa.start()).into();
        ProductState::new(initial_closure, 0)
    }

    /// Check if accepting.
    pub fn is_accepting(&self, state: &ProductState) -> bool {
        if state.edit_distance() > self.max_distance() {
            return false;
        }
        state.nfa_states.iter().any(|&s| self.nfa.is_final(s))
    }

    fn nfa_step_vec(&self, states: &[StateId], b: u8) -> Vec<StateId> {
        state_set_to_sorted_vec(self.nfa_step_state_set(states.iter().copied(), b))
    }

    fn nfa_step_state_set<I>(&self, states: I, b: u8) -> StateSet
    where
        I: IntoIterator<Item = StateId>,
    {
        let mut next_states = StateSet::with_capacity(self.nfa.num_states());

        for state in states {
            for trans in self.nfa.transitions_from(state) {
                if trans.label.matches(b) && trans.label.consumes_input() {
                    next_states.insert(trans.to);
                }
            }
        }

        self.nfa.epsilon_closure(&next_states)
    }

    fn nfa_advance_vec(&self, states: &[StateId]) -> Vec<StateId> {
        state_set_to_sorted_vec(self.nfa_advance_state_set(states.iter().copied()))
    }

    fn nfa_advance_state_set<I>(&self, states: I) -> StateSet
    where
        I: IntoIterator<Item = StateId>,
    {
        let mut next_states = StateSet::with_capacity(self.nfa.num_states());

        for state in states {
            for trans in self.nfa.transitions_from(state) {
                if trans.label.consumes_input() {
                    next_states.insert(trans.to);
                }
            }
        }

        self.nfa.epsilon_closure(&next_states)
    }

    /// Step NFA with transposed bytes.
    fn nfa_step_transposed_vec(&self, states: &[StateId], b1: u8, b2: u8) -> Vec<StateId> {
        let after_b2 = self.nfa_step_vec(states, b2);
        self.nfa_step_vec(&after_b2, b1)
    }

    /// Merge step for `Algorithm::MergeAndSplit` (byte-level analogue of the char
    /// version).
    ///
    /// Two adjacent input bytes merge into one at cost 1 as the generic metric's
    /// *unconstrained structural operation*: any two adjacent bytes may merge onto
    /// any single pattern edge, so `_b1`/`_b2` are correctly ignored (byte-agnostic
    /// by definition). Exact for the metric; never misses a match. (A byte-constrained
    /// variant would need a merge-relation map — a separate feature.)
    fn nfa_step_merged_from_advance_vec(
        &self,
        advanced_states: &[StateId],
        _b1: u8,
        _b2: u8,
    ) -> Vec<StateId> {
        advanced_states.to_vec()
    }

    /// Split step for `Algorithm::MergeAndSplit` (byte-level analogue of the char
    /// version).
    ///
    /// One input byte splits across two consuming NFA edges at cost 1 — the generic
    /// metric's unconstrained structural dual of merge; `_b` is correctly ignored
    /// (byte-agnostic by definition). Exact for the metric; never misses a match.
    fn nfa_step_split_from_advance_vec(&self, advanced_states: &[StateId], _b: u8) -> Vec<StateId> {
        self.nfa_advance_vec(advanced_states)
    }

    /// Check if input is accepted.
    pub fn accepts(&self, input: &[u8]) -> bool {
        self.min_distance(input).is_some()
    }

    /// Get minimum distance.
    pub fn min_distance(&self, input: &[u8]) -> Option<u8> {
        if self.nfa.is_empty() {
            return if input.is_empty() {
                Some(0)
            } else if input.len() <= max_distance_budget_len(self.max_distance()) {
                u8::try_from(input.len()).ok()
            } else {
                None
            };
        }

        if self.algorithm == Algorithm::Standard {
            return LanguageProduct::new(self.nfa.clone(), self.max_distance)
                .distance_to_language(input.iter().copied());
        }

        let n = input.len();
        let max_errors = self.max_distance();
        let initial_closure =
            state_set_to_sorted_vec(self.nfa.epsilon_closure_single(self.nfa.start()));

        let mut min_dist: Option<u8> = None;
        let search_capacity = product_search_capacity(n, max_errors);
        let mut visited: FxHashSet<(usize, Vec<StateId>, u8)> = FxHashSet::default();
        visited.reserve(search_capacity);
        let mut queue: VecDeque<(usize, Vec<StateId>, u8)> =
            VecDeque::with_capacity(search_capacity);

        enqueue_product_search_state(&mut queue, &mut visited, 0, initial_closure, 0);

        while let Some((pos, nfa_states, errors)) = queue.pop_front() {
            if errors > max_errors {
                continue;
            }

            if let Some(min) = min_dist {
                if errors >= min {
                    continue;
                }
            }

            if pos == n {
                if let Some(final_dist) = self.distance_to_final_vec(&nfa_states, errors) {
                    match min_dist {
                        None => min_dist = Some(final_dist),
                        Some(current) if final_dist < current => min_dist = Some(final_dist),
                        _ => {}
                    }
                }
                continue;
            }

            let b = input[pos];

            // 1. Match
            let match_states = self.nfa_step_vec(&nfa_states, b);
            if !match_states.is_empty() {
                enqueue_product_search_state(
                    &mut queue,
                    &mut visited,
                    pos + 1,
                    match_states,
                    errors,
                );
            }

            if errors < max_errors {
                let advanced_states = self.nfa_advance_vec(&nfa_states);

                // 2. Substitution
                if !advanced_states.is_empty() {
                    enqueue_product_search_state(
                        &mut queue,
                        &mut visited,
                        pos + 1,
                        advanced_states.clone(),
                        errors + 1,
                    );
                }

                // 3. Insertion
                enqueue_product_search_state(
                    &mut queue,
                    &mut visited,
                    pos + 1,
                    nfa_states.clone(),
                    errors + 1,
                );

                // 4. Deletion
                if !advanced_states.is_empty() {
                    enqueue_product_search_state(
                        &mut queue,
                        &mut visited,
                        pos,
                        advanced_states.clone(),
                        errors + 1,
                    );
                }

                // 5. Transposition
                if self.algorithm.supports_transposition() && pos + 1 < n {
                    let next_b = input[pos + 1];
                    let trans_states = self.nfa_step_transposed_vec(&nfa_states, b, next_b);
                    if !trans_states.is_empty() {
                        enqueue_product_search_state(
                            &mut queue,
                            &mut visited,
                            pos + 2,
                            trans_states,
                            errors + 1,
                        );
                    }
                }

                // 6. Merge: two adjacent input bytes → one NFA edge, cost 1.
                //    Generic Merge-and-Split: merge is byte-agnostic by definition
                //    (bytes not validated); see `nfa_step_merged_from_advance_vec`.
                if self.algorithm.supports_merge_split() && pos + 1 < n {
                    let next_b = input[pos + 1];
                    if !advanced_states.is_empty() {
                        let merge_states =
                            self.nfa_step_merged_from_advance_vec(&advanced_states, b, next_b);
                        if !merge_states.is_empty() {
                            enqueue_product_search_state(
                                &mut queue,
                                &mut visited,
                                pos + 2,
                                merge_states,
                                errors + 1,
                            );
                        }
                    }
                }

                // 7. Split: one input byte → two NFA edges, cost 1.
                //    Generic Merge-and-Split: split is byte-agnostic by definition
                //    (byte not validated); see `nfa_step_split_from_advance_vec`.
                if self.algorithm.supports_merge_split() && !advanced_states.is_empty() {
                    let split_states = self.nfa_step_split_from_advance_vec(&advanced_states, b);
                    if !split_states.is_empty() {
                        enqueue_product_search_state(
                            &mut queue,
                            &mut visited,
                            pos + 1,
                            split_states,
                            errors + 1,
                        );
                    }
                }
            }
        }

        min_dist
    }

    /// Compute distance to final.
    fn distance_to_final_vec(&self, states: &[StateId], base_dist: u8) -> Option<u8> {
        if states.iter().any(|&s| self.nfa.is_final(s)) {
            return Some(base_dist);
        }

        let remaining = self.max_distance().saturating_sub(base_dist);
        if remaining == 0 {
            return None;
        }

        let search_capacity = final_search_capacity(remaining);
        let mut visited: FxHashSet<Vec<StateId>> = FxHashSet::default();
        visited.reserve(search_capacity);
        let mut queue: VecDeque<(Vec<StateId>, u8)> = VecDeque::with_capacity(search_capacity);
        enqueue_final_search_state(&mut queue, &mut visited, states.to_vec(), 0);

        while let Some((current, dist)) = queue.pop_front() {
            if dist > remaining {
                continue;
            }

            if current.iter().any(|&s| self.nfa.is_final(s)) {
                return Some(base_dist + dist);
            }

            let next = self.nfa_advance_vec(&current);
            if !next.is_empty() {
                enqueue_final_search_state(&mut queue, &mut visited, next, dist + 1);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::nfa::compiler::{compile, compile_bytes};
    use crate::phonetic::regex::{parse, parse_bytes};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn product_state_hash(state: &ProductStateChar) -> u64 {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_product_search_capacity_accounts_for_error_layers() {
        assert_eq!(product_search_capacity(4, 0), 5);
        assert_eq!(product_search_capacity(4, 2), 15);
        assert_eq!(
            product_search_capacity(usize::MAX, u8::MAX),
            PRODUCT_SEARCH_CAPACITY_LIMIT
        );
    }

    #[test]
    fn test_deletion_and_final_search_capacity_are_bounded() {
        assert_eq!(deletion_closure_capacity(0, u8::MAX), 0);
        assert_eq!(deletion_closure_capacity(2, 3), 8);
        assert_eq!(
            deletion_closure_capacity(usize::MAX, u8::MAX),
            PRODUCT_SEARCH_CAPACITY_LIMIT
        );
        assert_eq!(final_search_capacity(0), 1);
        assert_eq!(final_search_capacity(u8::MAX), usize::from(u8::MAX) + 1);
    }

    #[test]
    fn test_product_cost_distance_narrowing_is_explicitly_bounded() {
        assert_eq!(finite_integral_f64_to_i64(-1.0), -1);
        assert_eq!(finite_integral_f64_to_i64(-0.0), 0);
        assert_eq!(finite_integral_f64_to_i64(0.0), 0);
        assert_eq!(finite_integral_f64_to_i64(1.0), 1);
        assert_eq!(
            finite_integral_f64_to_i64(4_503_599_627_370_496.0),
            4_503_599_627_370_496
        );

        assert_eq!(finite_positive_ceil_to_u8(0.1), 1);
        assert_eq!(finite_positive_ceil_to_u8(1.0), 1);
        assert_eq!(finite_positive_ceil_to_u8(1.1), 2);
        assert_eq!(finite_positive_ceil_to_u8(254.0), 254);
        assert_eq!(finite_positive_ceil_to_u8(254.1), u8::MAX);

        assert_eq!(product_cost_key(f64::NEG_INFINITY), i64::MIN + 1);
        assert_eq!(product_cost_key(-f64::MAX), i64::MIN);
        assert_eq!(product_cost_key(-0.000_001), -1);
        assert_eq!(product_cost_key(0.0), 0);
        assert_eq!(product_cost_key(0.000_001), 1);
        assert_eq!(product_cost_key(f64::MAX), i64::MAX);
        assert_eq!(product_cost_key(f64::INFINITY), i64::MAX);
        assert_eq!(product_cost_key(f64::NAN), i64::MIN);

        assert_eq!(accumulated_cost_to_u8_distance(-1.0), 0);
        assert_eq!(accumulated_cost_to_u8_distance(0.0), 0);
        assert_eq!(accumulated_cost_to_u8_distance(0.1), 1);
        assert_eq!(accumulated_cost_to_u8_distance(254.0), 254);
        assert_eq!(accumulated_cost_to_u8_distance(254.1), u8::MAX);
        assert_eq!(accumulated_cost_to_u8_distance(255.0), u8::MAX);
        assert_eq!(accumulated_cost_to_u8_distance(f64::INFINITY), u8::MAX);
        assert_eq!(accumulated_cost_to_u8_distance(f64::NAN), u8::MAX);

        assert_eq!(max_cost_to_u8_distance(-1.0), 0);
        assert_eq!(max_cost_to_u8_distance(f64::NAN), 0);
        assert_eq!(max_cost_to_u8_distance(0.1), 1);
        assert_eq!(max_cost_to_u8_distance(254.0), 254);
        assert_eq!(max_cost_to_u8_distance(254.1), u8::MAX);
        assert_eq!(max_cost_to_u8_distance(300.0), u8::MAX);
        assert_eq!(max_cost_to_u8_distance(f64::INFINITY), u8::MAX);
    }

    #[test]
    fn test_max_distance_budget_len_widens_u8_boundaries() {
        assert_eq!(max_distance_budget_len(0), 0);
        assert_eq!(max_distance_budget_len(u8::MAX), usize::from(u8::MAX));

        let zero_char = ProductAutomatonChar::new(NFAChar::new(), 0);
        assert!(zero_char.accepts(""));
        assert!(!zero_char.accepts("a"));
        assert_eq!(zero_char.min_distance("a"), None);

        let max_char = ProductAutomatonChar::new(NFAChar::new(), u8::MAX);
        let max_scalar_input = "a".repeat(usize::from(u8::MAX));
        let over_scalar_input = "a".repeat(usize::from(u8::MAX) + 1);
        assert!(max_char.accepts(&max_scalar_input));
        assert_eq!(max_char.min_distance(&max_scalar_input), Some(u8::MAX));
        assert!(!max_char.accepts(&over_scalar_input));
        assert_eq!(max_char.min_distance(&over_scalar_input), None);

        let max_byte = ProductAutomaton::new(NFA::new(), u8::MAX);
        let max_byte_input = vec![b'a'; usize::from(u8::MAX)];
        let over_byte_input = vec![b'a'; usize::from(u8::MAX) + 1];
        assert!(max_byte.accepts(&max_byte_input));
        assert_eq!(max_byte.min_distance(&max_byte_input), Some(u8::MAX));
        assert!(!max_byte.accepts(&over_byte_input));
        assert_eq!(max_byte.min_distance(&over_byte_input), None);
    }

    // ============================================================================
    // ProductAutomatonChar Tests
    // ============================================================================

    #[test]
    fn test_product_exact_match() {
        let nfa = compile(&parse("phone").expect("test: parse phone")).expect("test: compile nfa");
        let product = ProductAutomatonChar::new(nfa, 2);

        assert!(product.accepts("phone"));
        assert!(!product.accepts("xyz"));
    }

    #[test]
    fn test_product_alternation() {
        let nfa = compile(&parse("ph|f").expect("test: parse ph|f")).expect("test: compile nfa");
        let product = ProductAutomatonChar::new(nfa, 0);

        assert!(product.accepts("ph"));
        assert!(product.accepts("f"));
        assert!(!product.accepts("g"));
    }

    #[test]
    fn test_product_with_edit_distance() {
        let nfa = compile(&parse("phone").expect("test: parse phone")).expect("test: compile nfa");
        let product = ProductAutomatonChar::new(nfa, 2);

        // Exact match
        assert!(product.accepts("phone"));

        // One insertion
        assert!(product.accepts("phones")); // +s

        // One deletion
        assert!(product.accepts("phon")); // -e

        // One substitution
        assert!(product.accepts("phome")); // n→m

        // Two edits
        assert!(product.accepts("phon")); // -e, -e... wait, that's just one
        assert!(product.accepts("fone")); // ph→f is 2 edits (delete h, substitute p→f)
    }

    #[test]
    fn test_product_phonetic_pattern() {
        // Pattern: (ph|f)one - matches "phone" or "fone"
        let nfa = compile(&parse("(ph|f)one").expect("test: parse (ph|f)one"))
            .expect("test: compile nfa");
        let product = ProductAutomatonChar::new(nfa, 1);

        assert!(product.accepts("phone"));
        assert!(product.accepts("fone"));
        assert!(product.accepts("phones")); // +s (insertion)
        assert!(product.accepts("fones")); // +s (insertion)
                                           // "bone" is within distance 1 of "fone" (b→f substitution)
        assert!(product.accepts("bone"));

        // With max_distance=0, only exact matches
        let product_exact = ProductAutomatonChar::new(
            compile(&parse("(ph|f)one").expect("test: parse (ph|f)one"))
                .expect("test: compile nfa"),
            0,
        );
        assert!(product_exact.accepts("phone"));
        assert!(product_exact.accepts("fone"));
        assert!(!product_exact.accepts("bone")); // b doesn't match exactly
    }

    #[test]
    fn test_product_star() {
        let nfa = compile(&parse("a*").expect("test: parse a*")).expect("test: compile nfa");
        let product = ProductAutomatonChar::new(nfa, 1);

        assert!(product.accepts("")); // 0 a's
        assert!(product.accepts("a")); // 1 a
        assert!(product.accepts("aa")); // 2 a's
        assert!(product.accepts("b")); // 1 edit (substitute a for b or insert b)
        assert!(product.accepts("ab")); // 1 edit
    }

    #[test]
    fn test_product_min_distance() {
        let nfa = compile(&parse("phone").expect("test: parse phone")).expect("test: compile nfa");
        let product = ProductAutomatonChar::new(nfa, 3);

        assert_eq!(product.min_distance("phone"), Some(0));
        assert_eq!(product.min_distance("phon"), Some(1));
        assert_eq!(product.min_distance("phones"), Some(1));
        assert_eq!(product.min_distance("phome"), Some(1));
    }

    #[test]
    fn test_empty_char_product_counts_unicode_scalars_not_bytes() {
        let product = ProductAutomatonChar::new(NFAChar::new(), 1);

        assert!(product.accepts("é"));
        assert_eq!(product.min_distance("é"), Some(1));
        assert!(!product.accepts("éé"));
        assert_eq!(product.min_distance("éé"), None);

        let product = ProductAutomatonChar::new(NFAChar::new(), 2);
        assert!(product.accepts("éé"));
        assert_eq!(product.min_distance("éé"), Some(2));
    }

    #[test]
    fn test_incremental_frontier_matches_min_distance() {
        let nfa = compile(&parse("ab").expect("test: parse ab")).expect("test: compile nfa");
        let product = ProductAutomatonChar::new(nfa, 2);

        for input in ["", "a", "ab", "abc", "ba", "zz"] {
            let mut frontier = product.initial_frontier();
            for ch in input.chars() {
                frontier = product.transition_frontier(&frontier, ch);
            }

            assert_eq!(
                product.min_accepting_distance(&frontier),
                product.min_distance(input),
                "incremental frontier mismatch for {input:?}"
            );
        }
    }

    #[test]
    fn test_product_state_char_eq_matches_hash_key() {
        let states: FxHashSet<StateId> = [0, 1].into_iter().collect();

        let same_bucket = ProductStateChar::new(states.clone(), 0.500_000_499_90);
        let same_bucket_roundoff = ProductStateChar::new(states.clone(), 0.500_000_499_91);
        assert_eq!(same_bucket, same_bucket_roundoff);
        assert_eq!(
            product_state_hash(&same_bucket),
            product_state_hash(&same_bucket_roundoff)
        );

        let adjacent_bucket = ProductStateChar::new(states, 0.500_000_500_40);
        assert_ne!(same_bucket, adjacent_bucket);
    }

    #[test]
    fn test_product_char_class() {
        let nfa =
            compile(&parse("[aeiou]+").expect("test: parse [aeiou]+")).expect("test: compile nfa");
        let product = ProductAutomatonChar::new(nfa, 1);

        assert!(product.accepts("a"));
        assert!(product.accepts("aeiou"));
        assert!(product.accepts("b")); // 1 subst (b→a)
        assert!(product.accepts("ab")); // 1 error (b→a subst or b deletion)
                                        // Empty string IS within distance 1 of a vowel (insert any vowel)
        assert!(product.accepts(""));

        // With max_distance=0, only exact matches
        let product_exact = ProductAutomatonChar::new(
            compile(&parse("[aeiou]+").expect("test: parse [aeiou]+")).expect("test: compile nfa"),
            0,
        );
        assert!(product_exact.accepts("a"));
        assert!(product_exact.accepts("aeiou"));
        assert!(!product_exact.accepts("")); // empty doesn't match [aeiou]+
        assert!(!product_exact.accepts("b")); // b doesn't match any vowel
    }

    #[test]
    fn test_product_over_budget() {
        let nfa = compile(&parse("abc").expect("test: parse abc")).expect("test: compile nfa");
        let product = ProductAutomatonChar::new(nfa, 1);

        assert!(product.accepts("abc")); // exact
        assert!(product.accepts("ab")); // 1 deletion
        assert!(product.accepts("abcd")); // 1 insertion
        assert!(!product.accepts("xyz")); // 3 substitutions > budget
    }

    // ============================================================================
    // ProductAutomaton (byte-level) Tests
    // ============================================================================

    #[test]
    fn test_product_bytes_exact() {
        let nfa = compile_bytes(&parse_bytes(b"phone").expect("test: parse phone bytes"))
            .expect("test: compile nfa bytes");
        let product = ProductAutomaton::new(nfa, 2);

        assert!(product.accepts(b"phone"));
        assert!(!product.accepts(b"xyz"));
    }

    #[test]
    fn test_product_bytes_with_edits() {
        let nfa = compile_bytes(&parse_bytes(b"abc").expect("test: parse abc bytes"))
            .expect("test: compile nfa bytes");
        let product = ProductAutomaton::new(nfa, 1);

        assert!(product.accepts(b"abc"));
        assert!(product.accepts(b"ab"));
        assert!(product.accepts(b"abcd"));
        assert!(!product.accepts(b"xyz"));
    }

    #[test]
    fn test_product_bytes_min_distance() {
        let nfa = compile_bytes(&parse_bytes(b"phone").expect("test: parse phone bytes"))
            .expect("test: compile nfa bytes");
        let product = ProductAutomaton::new(nfa, 3);

        assert_eq!(product.min_distance(b"phone"), Some(0));
        assert_eq!(product.min_distance(b"phon"), Some(1));
    }

    // ============================================================================
    // Algorithm-specific Tests (Transposition, Merge/Split)
    // ============================================================================

    #[test]
    fn test_transposition_accepts() {
        // With standard algorithm, "ab" does NOT match "ba" with distance 1
        // (requires 2 substitutions: a→b, b→a)
        let nfa = compile(&parse("ab").expect("test: parse ab")).expect("test: compile nfa");
        let standard = ProductAutomatonChar::new(nfa.clone(), 1);
        assert!(!standard.accepts("ba")); // distance 2 with standard

        // With transposition algorithm, "ab" DOES match "ba" with distance 1
        let transposition =
            ProductAutomatonChar::with_algorithm(nfa.clone(), 1, Algorithm::Transposition);
        assert!(transposition.accepts("ba")); // distance 1 with transposition

        // Merge-and-Split has (1, 2) and (2, 1) structural operations, but no
        // (2, 2) transposition. Its capability flags must not silently enable
        // the transposition-only product construction.
        let merge_split = ProductAutomatonChar::with_algorithm(nfa, 1, Algorithm::MergeAndSplit);
        assert!(!merge_split.accepts("ba"));
    }

    #[test]
    #[should_panic(expected = "is history-dependent and is not supported")]
    fn char_product_rejects_true_damerau_instead_of_silently_computing_osa() {
        let nfa = compile(&parse("CA").expect("test: parse CA")).expect("test: compile nfa");
        let _ = ProductAutomatonChar::with_algorithm(nfa, 2, Algorithm::DamerauLevenshtein);
    }

    #[test]
    #[should_panic(expected = "is history-dependent and is not supported")]
    fn byte_product_rejects_true_damerau_instead_of_silently_computing_osa() {
        let nfa = compile_bytes(&parse_bytes(b"CA").expect("test: parse CA bytes"))
            .expect("test: compile byte nfa");
        let _ = ProductAutomaton::with_algorithm(nfa, 2, Algorithm::DamerauLevenshtein);
    }

    #[test]
    fn test_transposition_min_distance() {
        let nfa = compile(&parse("ab").expect("test: parse ab")).expect("test: compile nfa");

        // Standard: "ba" is distance 2 from "ab"
        let standard = ProductAutomatonChar::new(nfa.clone(), 2);
        assert_eq!(standard.min_distance("ba"), Some(2));

        // Transposition: "ba" is distance 1 from "ab"
        let transposition = ProductAutomatonChar::with_algorithm(nfa, 2, Algorithm::Transposition);
        assert_eq!(transposition.min_distance("ba"), Some(1));
    }

    #[test]
    fn test_transposition_longer_string() {
        // "hte" is "the" with h and t transposed
        let nfa = compile(&parse("the").expect("test: parse the")).expect("test: compile nfa");

        let standard = ProductAutomatonChar::new(nfa.clone(), 1);
        // With standard, "hte" requires 2 substitutions
        assert!(!standard.accepts("hte"));

        let transposition = ProductAutomatonChar::with_algorithm(nfa, 1, Algorithm::Transposition);
        // With transposition, "hte" is just 1 transposition away
        assert!(transposition.accepts("hte"));
    }

    #[test]
    fn test_merge_split_accepts() {
        // With merge/split, we can match strings where chars are merged or split
        let nfa = compile(&parse("abc").expect("test: parse abc")).expect("test: compile nfa");

        // Standard algorithm
        let standard = ProductAutomatonChar::new(nfa.clone(), 1);
        // "abcd" is 1 insertion (matches with standard)
        assert!(standard.accepts("abcd"));
        // "ab" is 1 deletion (matches with standard)
        assert!(standard.accepts("ab"));

        // With merge-and-split, we have additional operations
        let merge_split =
            ProductAutomatonChar::with_algorithm(nfa.clone(), 1, Algorithm::MergeAndSplit);
        assert!(merge_split.accepts("abcd")); // Still works
        assert!(merge_split.accepts("ab")); // Still works
    }

    #[test]
    fn test_merge_split_min_distance() {
        let nfa = compile(&parse("abc").expect("test: parse abc")).expect("test: compile nfa");

        let standard = ProductAutomatonChar::new(nfa.clone(), 3);
        let merge_split = ProductAutomatonChar::with_algorithm(nfa, 3, Algorithm::MergeAndSplit);

        // Both should find exact match
        assert_eq!(standard.min_distance("abc"), Some(0));
        assert_eq!(merge_split.min_distance("abc"), Some(0));

        // Both should find single-edit matches
        assert_eq!(standard.min_distance("ab"), Some(1));
        assert_eq!(merge_split.min_distance("ab"), Some(1));
    }

    #[test]
    fn test_algorithm_getter() {
        let nfa = compile(&parse("test").expect("test: parse test")).expect("test: compile nfa");

        let standard = ProductAutomatonChar::new(nfa.clone(), 1);
        assert_eq!(standard.algorithm(), Algorithm::Standard);

        let transposition =
            ProductAutomatonChar::with_algorithm(nfa.clone(), 1, Algorithm::Transposition);
        assert_eq!(transposition.algorithm(), Algorithm::Transposition);

        let merge_split = ProductAutomatonChar::with_algorithm(nfa, 1, Algorithm::MergeAndSplit);
        assert_eq!(merge_split.algorithm(), Algorithm::MergeAndSplit);
    }

    #[test]
    fn test_byte_level_transposition() {
        let nfa = compile_bytes(&parse_bytes(b"ab").expect("test: parse ab bytes"))
            .expect("test: compile nfa bytes");

        let standard = ProductAutomaton::new(nfa.clone(), 1);
        assert!(!standard.accepts(b"ba")); // distance 2 with standard

        let transposition =
            ProductAutomaton::with_algorithm(nfa.clone(), 1, Algorithm::Transposition);
        assert!(transposition.accepts(b"ba")); // distance 1 with transposition

        let merge_split = ProductAutomaton::with_algorithm(nfa, 1, Algorithm::MergeAndSplit);
        assert!(!merge_split.accepts(b"ba"));
    }

    /// Document the ACTUAL accepted language of `Algorithm::MergeAndSplit`.
    ///
    /// Merge/split are character-agnostic structural operations of the generic
    /// Merge-and-Split edit metric (any two adjacent symbols merge to one; any
    /// symbol splits to two, cost 1), so they accept strings the `Standard`
    /// algorithm rejects at the same budget — the correct, deliberately more
    /// permissive generic-metric distance — while never MISSING anything
    /// `Standard` accepts.
    #[test]
    fn test_merge_split_generic_unconstrained_semantics() {
        let nfa = compile(&parse("abc").expect("test: parse abc")).expect("test: compile nfa");
        let standard = ProductAutomatonChar::new(nfa.clone(), 1);
        let merge_split =
            ProductAutomatonChar::with_algorithm(nfa.clone(), 1, Algorithm::MergeAndSplit);

        // Generic SPLIT (character-agnostic by definition): "zc" is Levenshtein
        // distance 2 from "abc", but its Merge-and-Split distance is 1 — split the
        // single input 'z' across the pattern's "a" and "b" edges, then match 'c'.
        // This is the correct generic-metric distance (merge/split are cheaper ops,
        // deliberately more permissive than Levenshtein), not an over-approximation.
        assert!(!standard.accepts("zc"));
        assert!(merge_split.accepts("zc"));
        assert_eq!(merge_split.min_distance("zc"), Some(1));

        // Generic MERGE (character-agnostic by definition): "zzbc" is Levenshtein
        // distance 2 from "abc", but its Merge-and-Split distance is 1 — merge the
        // adjacent "zz" onto the single "a" edge, then match "bc". Correct for the
        // generic metric.
        assert!(!standard.accepts("zzbc"));
        assert!(merge_split.accepts("zzbc"));

        // Does NOT miss: everything the Standard algorithm accepts within the
        // same budget is still accepted (exact match + ordinary single edits).
        assert!(merge_split.accepts("abc")); // exact
        assert!(merge_split.accepts("ab")); // standard deletion
        assert!(merge_split.accepts("abcd")); // standard insertion
        assert_eq!(merge_split.min_distance("abc"), Some(0));
    }

    /// F2: pin the documented reality that `phonetic_weight` is stored but
    /// inert — it is clamped to `>= 0.0` and never changes matching results.
    #[test]
    fn test_phonetic_weight_is_clamped_and_inert() {
        let nfa = compile(&parse("phone").expect("test: parse phone")).expect("test: compile nfa");

        // Negative / NaN weights are clamped to >= 0.0 (F2 pruning invariant).
        let neg = ProductAutomatonChar::with_phonetic_weight(nfa.clone(), 2, -3.5);
        assert_eq!(neg.phonetic_weight(), 0.0);
        let nan = ProductAutomatonChar::with_phonetic_weight(nfa.clone(), 2, f64::NAN);
        assert_eq!(nan.phonetic_weight(), 0.0);

        // A positive weight is stored verbatim...
        let weighted = ProductAutomatonChar::with_algorithm_and_weight(
            nfa.clone(),
            2,
            Algorithm::Standard,
            5.0,
        );
        assert_eq!(weighted.phonetic_weight(), 5.0);

        // ...but does NOT affect matching: accepts/min_distance are identical to
        // the zero-weight baseline for every probe.
        let baseline = ProductAutomatonChar::new(nfa, 2);
        for probe in ["phone", "phon", "phones", "phome", "xyz"] {
            assert_eq!(weighted.accepts(probe), baseline.accepts(probe));
            assert_eq!(weighted.min_distance(probe), baseline.min_distance(probe));
        }

        // Byte-level automaton mirrors the same clamp + inert contract.
        let bnfa = compile_bytes(&parse_bytes(b"phone").expect("test: parse phone bytes"))
            .expect("test: compile nfa bytes");
        let bneg = ProductAutomaton::with_phonetic_weight(bnfa.clone(), 2, -1.0);
        assert_eq!(bneg.phonetic_weight(), 0.0);
        let bweighted =
            ProductAutomaton::with_algorithm_and_weight(bnfa.clone(), 2, Algorithm::Standard, 2.0);
        assert_eq!(bweighted.phonetic_weight(), 2.0);
        let bbaseline = ProductAutomaton::new(bnfa, 2);
        let byte_probes: [&[u8]; 4] = [b"phone", b"phon", b"phones", b"xyz"];
        for probe in byte_probes {
            assert_eq!(bweighted.accepts(probe), bbaseline.accepts(probe));
            assert_eq!(bweighted.min_distance(probe), bbaseline.min_distance(probe));
        }
    }

    // ============================================================================
    // Articulatory Cost Tests
    // ============================================================================

    #[cfg(feature = "phonetic-rules")]
    mod articulatory_tests {
        use super::*;
        use crate::transducer::ArticulatoryCosts;

        /// Test that articulatory costs constructor works properly.
        #[test]
        fn test_articulatory_costs_constructor() {
            let nfa =
                compile(&parse("test").expect("test: parse test")).expect("test: compile nfa");
            let costs = ArticulatoryCosts::default();

            let product =
                ProductAutomatonChar::with_articulatory_costs(nfa, 2.0, Algorithm::Standard, costs);

            assert!(product.articulatory_costs().is_some());
            assert!((product.max_cost() - 2.0).abs() < 1e-9);
            assert_eq!(product.algorithm(), Algorithm::Standard);
        }

        /// Test that substitution costs reflect articulatory distance.
        /// Similar sounds (voicing pairs) should cost less than distant sounds.
        #[test]
        fn test_substitution_cost_varies_by_phonetic_similarity() {
            let nfa = compile(&parse("p").expect("test: parse p")).expect("test: compile nfa");
            let costs = ArticulatoryCosts::default();

            let product =
                ProductAutomatonChar::with_articulatory_costs(nfa, 2.0, Algorithm::Standard, costs);

            // p→b (voicing only) should cost less than p→k (different place)
            let pb_cost = product.substitution_cost('b', Some('p'));
            let pk_cost = product.substitution_cost('k', Some('p'));

            assert!(
                pb_cost < pk_cost,
                "p→b ({}) should be cheaper than p→k ({})",
                pb_cost,
                pk_cost
            );

            // p→p should be free
            let pp_cost = product.substitution_cost('p', Some('p'));
            assert!(pp_cost < 0.01, "p→p should be nearly free, got {}", pp_cost);
        }

        /// Test that the transition() method uses articulatory costs for substitutions.
        #[test]
        fn test_transition_uses_articulatory_costs() {
            // Simple pattern that matches 'p'
            let nfa = compile(&parse("p").expect("test: parse p")).expect("test: compile nfa");
            let costs = ArticulatoryCosts::default();

            let product =
                ProductAutomatonChar::with_articulatory_costs(nfa, 2.0, Algorithm::Standard, costs);

            let initial = product.initial_state();

            // Transition with exact match 'p' - no cost
            let match_successors = product.transition(&initial, 'p');
            let match_state = match_successors
                .iter()
                .find(|s| s.accumulated_cost < 0.01)
                .expect("should find match state with zero cost");
            assert!(
                match_state.accumulated_cost < 0.01,
                "Exact match should have near-zero cost, got {}",
                match_state.accumulated_cost
            );

            // Transition with 'b' - should have low articulatory cost (voicing pair)
            let b_successors = product.transition(&initial, 'b');
            let b_subst_state = b_successors
                .iter()
                .find(|s| s.accumulated_cost > 0.01 && s.accumulated_cost < 0.5)
                .expect("should find substitution state with low cost for 'b'");

            // Transition with 'k' - should have higher articulatory cost
            let k_successors = product.transition(&initial, 'k');
            let k_subst_state = k_successors
                .iter()
                .find(|s| s.accumulated_cost > 0.3)
                .expect("should find substitution state with higher cost for 'k'");

            assert!(
                b_subst_state.accumulated_cost < k_subst_state.accumulated_cost,
                "p→b ({}) should be cheaper than p→k ({})",
                b_subst_state.accumulated_cost,
                k_subst_state.accumulated_cost
            );
        }

        /// Test accumulated cost across multiple transitions.
        #[test]
        fn test_accumulated_cost_tracking() {
            let nfa = compile(&parse("ab").expect("test: parse ab")).expect("test: compile nfa");
            let costs = ArticulatoryCosts::default();

            let product = ProductAutomatonChar::with_articulatory_costs(
                nfa,
                3.0, // Allow up to 3.0 total cost
                Algorithm::Standard,
                costs,
            );

            let initial = product.initial_state();
            assert!(
                initial.accumulated_cost.abs() < 1e-9,
                "Initial cost should be 0"
            );

            // Transition with 'a' (exact match)
            let after_a = product.transition(&initial, 'a');
            let match_a = after_a
                .iter()
                .find(|s| s.accumulated_cost < 0.01)
                .expect("should find exact match for 'a'");

            // Transition with 'd' instead of 'b' (substitution)
            let after_d = product.transition(match_a, 'd');
            // Should have accumulated some cost from the substitution
            let subst_state = after_d
                .iter()
                .find(|s| s.accumulated_cost > 0.1)
                .expect("should find state with accumulated substitution cost");

            assert!(
                subst_state.accumulated_cost > 0.1,
                "Accumulated cost should reflect substitution, got {}",
                subst_state.accumulated_cost
            );
        }

        /// Test that without articulatory costs, fixed cost (1.0) is used.
        #[test]
        fn test_fixed_cost_without_articulatory() {
            let nfa = compile(&parse("p").expect("test: parse p")).expect("test: compile nfa");

            // No articulatory costs
            let product = ProductAutomatonChar::new(nfa, 2);

            // Without articulatory costs, all substitutions should cost 1.0
            let initial = product.initial_state();

            let b_successors = product.transition(&initial, 'b');
            let k_successors = product.transition(&initial, 'k');

            // Find substitution states (with cost = 1.0)
            let b_subst = b_successors
                .iter()
                .find(|s| (s.accumulated_cost - 1.0).abs() < 0.01);
            let k_subst = k_successors
                .iter()
                .find(|s| (s.accumulated_cost - 1.0).abs() < 0.01);

            assert!(
                b_subst.is_some(),
                "Should find substitution with cost 1.0 for 'b'"
            );
            assert!(
                k_subst.is_some(),
                "Should find substitution with cost 1.0 for 'k'"
            );
        }

        /// Test that edit_distance() rounds up fractional costs.
        #[test]
        fn test_edit_distance_rounds_up() {
            use rustc_hash::FxHashSet;

            let states: FxHashSet<StateId> = vec![0].into_iter().collect();

            // 0.3 cost should round up to 1
            let state1 = ProductStateChar::new(states.clone(), 0.3);
            assert_eq!(state1.edit_distance(), 1);

            // 1.7 cost should round up to 2
            let state2 = ProductStateChar::new(states.clone(), 1.7);
            assert_eq!(state2.edit_distance(), 2);

            // 0.0 cost should be 0
            let state3 = ProductStateChar::new(states, 0.0);
            assert_eq!(state3.edit_distance(), 0);
        }

        /// Test articulatory costs with IPA characters.
        #[test]
        fn test_ipa_articulatory_costs() {
            // Pattern with IPA voiceless bilabial stop
            let nfa = compile(&parse("p").expect("test: parse p")).expect("test: compile nfa");
            let costs = ArticulatoryCosts::default();

            let product =
                ProductAutomatonChar::with_articulatory_costs(nfa, 2.0, Algorithm::Standard, costs);

            // Test with IPA characters (if supported)
            // ʃ (voiceless postalveolar fricative) vs s (voiceless alveolar fricative)
            // Both are voiceless fricatives, but different place
            let sh_cost = product.substitution_cost('ʃ', Some('s'));
            let sh_p_cost = product.substitution_cost('ʃ', Some('p'));

            // ʃ→s should be cheaper than ʃ→p (fricative vs stop)
            assert!(
                sh_cost < sh_p_cost,
                "ʃ→s ({}) should be cheaper than ʃ→p ({})",
                sh_cost,
                sh_p_cost
            );
        }

        #[test]
        fn test_min_cost_exact_match_is_zero() {
            let nfa = compile(&parse("pat").expect("parse")).expect("compile");
            let product = ProductAutomatonChar::with_articulatory_costs(
                nfa,
                2.0,
                Algorithm::Standard,
                ArticulatoryCosts::default(),
            );
            assert_eq!(product.min_cost("pat"), Some(0.0));
        }

        #[test]
        fn test_min_cost_equals_min_distance_without_articulatory() {
            // With no articulatory costs every operation costs 1.0, so the
            // uniform-cost search must agree with the integer edit distance.
            let nfa = compile(&parse("pat").expect("parse")).expect("compile");
            let product = ProductAutomatonChar::new(nfa, 3);
            for input in ["pat", "bat", "pot", "pats", "at", "ppat"] {
                let cost = product.min_cost(input);
                let dist = product.min_distance(input).map(f64::from);
                assert_eq!(
                    cost, dist,
                    "min_cost must equal min_distance (as f64) without articulatory costs, for {input:?}"
                );
            }
        }

        #[test]
        fn test_min_cost_discounts_soundalike_substitution() {
            // Pattern "pat"; align dictionary terms "bat" (b→p, near) and "cat"
            // (c→p, far). Both are one integer edit, but the articulatory cost of
            // the near pair must be strictly smaller, and both below a full edit.
            let nfa = compile(&parse("pat").expect("parse")).expect("compile");
            let product = ProductAutomatonChar::with_articulatory_costs(
                nfa,
                2.0,
                Algorithm::Standard,
                ArticulatoryCosts::default(),
            );

            let bat = product.min_cost("bat").expect("bat within budget");
            let cat = product.min_cost("cat").expect("cat within budget");

            assert_eq!(product.min_distance("bat"), Some(1));
            assert_eq!(product.min_distance("cat"), Some(1));
            assert!(
                bat < 1.0,
                "near substitution discounted below a full edit: {bat}"
            );
            assert!(
                bat < cat,
                "near pair (b→p = {bat}) must cost less than far pair (c→p = {cat})"
            );
            // min_cost never exceeds min_distance (substitutions only discount).
            assert!(bat <= 1.0 && cat <= 1.0);
        }

        #[test]
        fn test_min_cost_respects_budget() {
            // A term needing more than the cost budget yields None.
            let nfa = compile(&parse("pat").expect("parse")).expect("compile");
            let product = ProductAutomatonChar::with_articulatory_costs(
                nfa,
                1.0,
                Algorithm::Standard,
                ArticulatoryCosts::default(),
            );
            // "xyzzy" is far more than 1.0 in articulatory cost from "pat".
            assert_eq!(product.min_cost("xyzzy"), None);
        }

        /// Test that max_cost threshold is respected.
        #[test]
        fn test_max_cost_threshold() {
            let nfa = compile(&parse("abc").expect("test: parse abc")).expect("test: compile nfa");
            let costs = ArticulatoryCosts::default();

            // Very low max_cost - should prune expensive substitutions
            let product = ProductAutomatonChar::with_articulatory_costs(
                nfa,
                0.5, // Only allow 0.5 total cost
                Algorithm::Standard,
                costs,
            );

            let initial = product.initial_state();

            // Transition with 'z' - expensive substitution should be pruned
            let z_successors = product.transition(&initial, 'z');

            // All successor states should have cost <= 0.5
            for state in &z_successors {
                assert!(
                    state.accumulated_cost <= 0.5 + 1e-9,
                    "State cost {} exceeds max_cost 0.5",
                    state.accumulated_cost
                );
            }
        }

        /// Test is_accepting with articulatory costs.
        #[test]
        fn test_is_accepting_with_articulatory_costs() {
            let nfa = compile(&parse("p").expect("test: parse p")).expect("test: compile nfa");
            let costs = ArticulatoryCosts::default();

            let product = ProductAutomatonChar::with_articulatory_costs(
                nfa,
                1.0, // max cost 1.0
                Algorithm::Standard,
                costs,
            );

            let initial = product.initial_state();

            // After matching 'p', should accept
            let after_p = product.transition(&initial, 'p');
            let match_state = after_p
                .iter()
                .find(|s| s.accumulated_cost < 0.01)
                .expect("should find match state");
            assert!(
                product.is_accepting(match_state),
                "Should accept after matching 'p'"
            );

            // After substituting with similar sound 'b', might still accept if cost < 1.0
            let after_b = product.transition(&initial, 'b');
            let subst_state = after_b
                .iter()
                .find(|s| s.accumulated_cost > 0.01 && s.accumulated_cost <= 1.0);
            if let Some(state) = subst_state {
                assert!(
                    product.is_accepting(state),
                    "Should accept similar substitution within cost budget"
                );
            }
        }
    }
}
