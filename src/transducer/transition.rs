//! State transition logic for Levenshtein automata.

use super::variant::{with_variant, AutomatonVariant, TransitionCtx, VariantSpec};
use super::variants::{AffineGapParams, AffineV};
use super::{
    Algorithm, Position, PositionKind, State, StatePool, SubstitutionPolicy, SubstitutionPolicyFor,
};
use libdictenstein::CharUnit;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Configuration shared by pooled state transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransitionSettings {
    /// Maximum edit distance.
    pub max_distance: usize,
    /// Edit algorithm variant.
    pub algorithm: Algorithm,
    /// Whether positions beyond the query length are free matches.
    pub prefix_mode: bool,
}

impl TransitionSettings {
    /// Create transition settings.
    pub const fn new(max_distance: usize, algorithm: Algorithm, prefix_mode: bool) -> Self {
        Self {
            max_distance,
            algorithm,
            prefix_mode,
        }
    }
}

/// Configuration shared by pooled affine-gap transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AffineTransitionSettings {
    /// Maximum exact scaled cost.
    pub max_cost: usize,
    /// Exact affine-gap parameters.
    pub params: AffineGapParams,
    /// Whether positions beyond the query length are free matches.
    pub prefix_mode: bool,
}

impl AffineTransitionSettings {
    /// Create affine transition settings.
    pub(crate) const fn new(max_cost: usize, params: AffineGapParams, prefix_mode: bool) -> Self {
        Self {
            max_cost,
            params,
            prefix_mode,
        }
    }
}

/// Compute the characteristic vector for a position in the query.
///
/// The characteristic vector indicates which characters in a window
/// of the query term can be consumed without error when matching the
/// dictionary character.
///
/// # Arguments
/// * `policy` - Substitution policy for character matching
/// * `dict_unit` - Character unit from the dictionary edge
/// * `query` - Query term units (bytes or chars)
/// * `window_size` - Size of the window (typically max_distance + 1)
/// * `offset` - Base offset in query
/// # Returns
/// Slice of booleans indicating positions where characters can match without error.
/// This includes both exact matches AND policy-allowed substitutions.
/// Uses stack storage for the common max-distance ≤ 7 case and grows only when
/// callers request a larger edit window.
///
/// # Semantics
///
/// For unrestricted policies, only exact character matches are allowed (traditional Levenshtein).
/// For restricted policies, the policy determines if a substitution should be cost-free.
/// This treats policy-allowed substitutions as "equivalences" rather than edits.
///
/// # Performance
///
#[inline]
fn characteristic_vector<'a, U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
    policy: &P,
    dict_unit: U,
    query: &[U],
    window_size: usize,
    offset: usize,
    output: &'a mut SmallVec<[bool; 8]>,
) -> &'a [bool] {
    output.clear();
    output.reserve(window_size);

    // The characteristic vector shows which query positions can consume dict_unit without error.
    // This includes:
    // 1. Exact character matches (query[i] == dict_unit)
    // 2. Policy-allowed substitutions (policy.is_allowed(...))
    //
    // For Unrestricted policy: is_allowed always returns false (no zero-cost substitutions)
    // For Restricted policy: is_allowed checks the substitution set

    for i in 0..window_size {
        let matched = if let Some(query_unit) = checked_query_index(offset, i)
            .and_then(|query_idx| query.get(query_idx))
            .copied()
        {
            query_unit == dict_unit || is_substitution_allowed(policy, dict_unit, query_unit)
        } else {
            false
        };
        output.push(matched);
    }

    output.as_slice()
}

/// Helper function to check if a substitution is allowed.
///
/// This function uses the `SubstitutionPolicyFor` marker trait to safely
/// check substitutions for both byte-level (u8) and character-level (char) operations.
///
/// # Type Safety
///
/// The `P: SubstitutionPolicyFor<U>` bound ensures compile-time type safety:
/// - `Restricted<'a>` can only be used with `U = u8`
/// - `RestrictedChar<'a>` can only be used with `U = char`
/// - `Unrestricted` works with both `u8` and `char`
///
/// # Zero-Cost Abstraction
///
/// For `Unrestricted` policy, this function compiles to a constant `false` and
/// is optimized away entirely by the compiler through inlining and constant propagation.
#[inline(always)]
fn is_substitution_allowed<U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
    policy: &P,
    dict_unit: U,
    query_unit: U,
) -> bool {
    // Safe! No transmute needed - uses the correct trait method for the unit type
    policy.is_allowed_for(dict_unit, query_unit)
}

#[inline(always)]
fn checked_query_index(offset: usize, window_index: usize) -> Option<usize> {
    offset.checked_add(window_index)
}

#[inline(always)]
fn checked_window_end(start: usize, limit: usize, len: usize) -> Option<usize> {
    let end = start.checked_add(limit).unwrap_or(len).min(len);
    (start <= end).then_some(end)
}

#[inline(always)]
fn checked_successor_or_max(value: usize) -> usize {
    match value.checked_add(1) {
        Some(next) => next,
        None => usize::MAX,
    }
}

#[inline(always)]
pub(crate) fn transition_window_size(max_distance: usize, query_length: usize) -> usize {
    let distance_window = checked_successor_or_max(max_distance);
    let query_window = checked_successor_or_max(query_length).max(1);
    distance_window.min(query_window)
}

/// Query-local cache of complete label/query equivalence vectors.
///
/// Values include a false suffix as wide as any unit-cost transition window.
/// A transition can therefore borrow an offset window directly, including at
/// the end of the query, without rebuilding a `SmallVec` for every state
/// position. The cache is unit- and substitution-policy-generic.
pub(crate) struct CharacteristicCache<U: CharUnit> {
    vectors: FxHashMap<U, Box<[bool]>>,
    query_length: usize,
    padding: usize,
}

/// Query-local unit-transition engine shared by every iterator surface.
///
/// It owns the lazy characteristic cache and centralizes the epsilon-closed
/// queued-state invariant. Concrete unit, substitution-policy, and automaton
/// variant types are still monomorphized; this abstraction introduces no
/// trait objects or indirect calls in the transition loop.
pub(crate) struct CachedUnitTransitions<U: CharUnit> {
    cache: CharacteristicCache<U>,
    max_distance: usize,
}

impl<U: CharUnit> CachedUnitTransitions<U> {
    pub(crate) fn new(query_length: usize, max_distance: usize) -> Self {
        Self {
            cache: CharacteristicCache::new(query_length, max_distance),
            max_distance,
        }
    }

    #[inline]
    pub(crate) fn transition<P>(
        &mut self,
        state: &State,
        pool: &mut StatePool,
        policy: &P,
        dict_unit: U,
        query: &[U],
        settings: TransitionSettings,
    ) -> Option<State>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        debug_assert_eq!(settings.max_distance, self.max_distance);
        let matches = self.cache.matches_for(policy, dict_unit, query);
        transition_epsilon_closed_state_pooled_cached(state, pool, matches, query.len(), settings)
    }

    /// Transition an affine-gap state through the same query-local
    /// characteristic cache and epsilon-closed queue invariant.
    #[inline]
    pub(crate) fn transition_affine<P>(
        &mut self,
        state: &State,
        pool: &mut StatePool,
        policy: &P,
        dict_unit: U,
        query: &[U],
        settings: AffineTransitionSettings,
    ) -> Option<State>
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        debug_assert_eq!(settings.max_cost, self.max_distance);

        // An affine query gap can inspect every remaining query unit when its
        // extension cost is zero. Grow the false suffix once before borrowing
        // the label vector so every legal window remains a direct slice.
        self.cache.ensure_padding(query.len().saturating_add(1));
        let matches = self.cache.matches_for(policy, dict_unit, query);
        let ctx = TransitionCtx::new(
            query.len(),
            settings.max_cost,
            settings.prefix_mode,
            settings.params,
        );
        let characteristics = CachedCharacteristics::new(matches, query.len());
        transition_epsilon_closed_state_pooled_with::<AffineV, _>(
            state,
            pool,
            &characteristics,
            &ctx,
        )
    }
}

impl<U: CharUnit> CharacteristicCache<U> {
    pub(crate) fn new(query_length: usize, max_distance: usize) -> Self {
        Self {
            vectors: FxHashMap::default(),
            query_length,
            padding: transition_window_size(max_distance, query_length),
        }
    }

    #[inline]
    pub(crate) fn matches_for<P>(&mut self, policy: &P, dict_unit: U, query: &[U]) -> &[bool]
    where
        P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    {
        debug_assert_eq!(query.len(), self.query_length);
        let matches = self.vectors.entry(dict_unit).or_insert_with(|| {
            let mut matches = Vec::with_capacity(query.len().saturating_add(self.padding));
            matches.extend(query.iter().copied().map(|query_unit| {
                query_unit == dict_unit || policy.is_allowed_for(dict_unit, query_unit)
            }));
            matches.resize(query.len().saturating_add(self.padding), false);
            matches.into_boxed_slice()
        });
        matches
    }

    /// Ensure cached vectors contain at least `padding` false units after the
    /// query. Existing vectors are extended in place only when a specialized
    /// automaton requires a wider look-ahead window.
    pub(crate) fn ensure_padding(&mut self, padding: usize) {
        if padding <= self.padding {
            return;
        }

        let length = self.query_length.saturating_add(padding);
        for matches in self.vectors.values_mut() {
            let mut extended = std::mem::take(matches).into_vec();
            extended.resize(length, false);
            *matches = extended.into_boxed_slice();
        }
        self.padding = padding;
    }
}

#[inline(always)]
fn remaining_error_window(max_distance: usize, num_errors: usize) -> usize {
    match max_distance.checked_sub(num_errors) {
        Some(remaining) => checked_successor_or_max(remaining),
        None => 0,
    }
}

#[inline(always)]
fn checked_position_with_offsets(
    term_index: usize,
    term_offset: usize,
    num_errors: usize,
    error_offset: usize,
) -> Option<Position> {
    let term_index = term_index.checked_add(term_offset)?;
    let num_errors = num_errors.checked_add(error_offset)?;
    Some(Position::new(term_index, num_errors))
}

#[inline(always)]
fn checked_special_position_with_offsets(
    term_index: usize,
    term_offset: usize,
    num_errors: usize,
    error_offset: usize,
    kind: PositionKind,
) -> Option<Position> {
    let term_index = term_index.checked_add(term_offset)?;
    let num_errors = num_errors.checked_add(error_offset)?;
    Some(Position::with_kind(term_index, num_errors, kind, 0))
}

#[inline(always)]
fn checked_deletion_position(
    term_index: usize,
    num_errors: usize,
    match_offset: usize,
) -> Option<Position> {
    let term_offset = match_offset.checked_add(1)?;
    checked_position_with_offsets(term_index, term_offset, num_errors, match_offset)
}

#[inline(always)]
fn push_checked_position(next: &mut SmallVec<[Position; 4]>, position: Option<Position>) {
    if let Some(position) = position {
        next.push(position);
    }
}

#[inline(always)]
fn has_query_units(term_index: usize, count: usize, query_length: usize) -> bool {
    term_index
        .checked_add(count)
        .is_some_and(|end| end <= query_length)
}

/// Transition a position given a characteristic vector.
///
/// Computes all possible next positions from the current position
/// after consuming a dictionary character, considering the query
/// term through the characteristic vector.
///
/// # Standard Algorithm
///
/// From position `(i, e)` (term_index=i, num_errors=e), we can reach:
/// - `(i+1, e)` if `query[i]` matches `dict_char` (no error)
/// - `(i+1, e+1)` via substitution (if different)
/// - `(i, e+1)` via insertion (dictionary char, no query advance)
/// - `(i+1, e+1)` via deletion (skip query char)
///
/// The characteristic vector tells us whether query[offset+k] matches
/// for k in [0..window_size).
///
/// # Prefix Mode
///
/// When `prefix_mode` is true, characters beyond query_length are treated
/// as free matches (no errors added), enabling autocomplete/prefix matching.
#[inline]
pub fn transition_position(
    position: &Position,
    characteristic_vector: &[bool],
    query_length: usize,
    max_distance: usize,
    algorithm: Algorithm,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    algorithm.assert_supported_max_distance(max_distance);
    match algorithm {
        Algorithm::Standard => transition_standard(
            position,
            characteristic_vector,
            query_length,
            max_distance,
            prefix_mode,
        ),
        Algorithm::Transposition => transition_transposition(
            position,
            characteristic_vector,
            query_length,
            max_distance,
            prefix_mode,
        ),
        Algorithm::MergeAndSplit => transition_merge_split(
            position,
            characteristic_vector,
            query_length,
            max_distance,
            prefix_mode,
        ),
        Algorithm::DamerauLevenshtein => transition_damerau(
            position,
            characteristic_vector,
            query_length,
            max_distance,
            prefix_mode,
        ),
    }
}

/// Standard algorithm transition (insert, delete, substitute)
///
/// Returns SmallVec to avoid heap allocations for small result sets.
/// Most transitions produce 2-3 positions, so we stack-allocate up to 4.
///
/// # Prefix Matching Support
///
/// Find the index of the first true value in characteristic_vector[start..start+limit]
/// Returns None if no true value is found within the range.
///
/// This corresponds to the `index_of` function in the C++ implementation.
#[inline]
fn index_of_match(cv: &[bool], start: usize, limit: usize) -> Option<usize> {
    let end = checked_window_end(start, limit, cv.len())?;
    cv.get(start..end)?.iter().position(|&matched| matched)
}

/// Standard Levenshtein position transition function.
///
/// This implementation follows the C++/Java logic exactly, including the
/// multi-character deletion optimization via `index_of`.
///
/// When `prefix_mode` is true and term_index >= query_length, additional dictionary
/// characters are treated as free matches, keeping the position "stuck" at query_length
/// with the same error count.
#[inline]
pub(crate) fn transition_standard(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    transition_standard_into(
        position,
        cv,
        query_length,
        max_distance,
        prefix_mode,
        &mut next,
    );
    next
}

/// Fill a caller-owned buffer with Standard Levenshtein successors.
///
/// Keeping ownership at the caller avoids constructing and then assigning a
/// temporary `SmallVec` inside [`AutomatonVariant::successors`].
#[inline(always)]
pub(crate) fn transition_standard_into(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
    next: &mut SmallVec<[Position; 4]>,
) {
    debug_assert!(next.is_empty());
    let i = position.term_index;
    let e = position.num_errors;
    let h = 0; // cv is offset-adjusted to start at position i, so h = 0
    let w = cv.len();

    // Prefix matching: if enabled and we've consumed the full query, treat any character as a free match
    if prefix_mode && i >= query_length {
        next.push(Position::new(i, e));
        return;
    }

    // Case 1: e < max_distance (can still add errors)
    if e < max_distance {
        // Subcase 1a: At least 2 characters remain in query (h + 2 <= w)
        if h + 2 <= w {
            let a = remaining_error_window(max_distance, e);
            let b = w - h;
            let k = a.min(b);

            match index_of_match(cv, h, k) {
                Some(0) => {
                    // Immediate match at cv[h]
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
                }
                Some(j) => {
                    // Match found at cv[h + j]
                    // Return: insertion, substitution, and multi-character deletion
                    push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                    push_checked_position(next, checked_deletion_position(i, e, j));
                }
                None => {
                    // No match found in range
                    push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                }
            }
        }
        // Subcase 1b: Exactly 1 character remains (h + 1 == w)
        else if h + 1 == w {
            if cv[h] {
                // Match at last position
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            } else {
                // No match at last position
                push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
            }
        }
        // Subcase 1c: Past the end of query (h >= w)
        else {
            // Only insertion is possible
            push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
        }
    }
    // Case 2: e == max_distance (at max errors, only exact matches allowed)
    else if e == max_distance && h < w && cv[h] {
        push_checked_position(next, checked_position_with_offsets(i, 1, max_distance, 0));
    }
}

/// Unrestricted Damerau–Levenshtein position transition.
///
/// Normal positions retain every Standard successor and may start a
/// Lowrance–Wagner macro transition. Pending positions either resolve their
/// deferred query endpoint or consume another intervening dictionary unit.
#[inline]
pub(crate) fn transition_damerau(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    transition_damerau_into(
        position,
        cv,
        query_length,
        max_distance,
        prefix_mode,
        &mut next,
    );
    next
}

/// Fill a caller-owned buffer with unrestricted Damerau successors.
#[inline(always)]
pub(crate) fn transition_damerau_into(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
    next: &mut SmallVec<[Position; 4]>,
) {
    debug_assert!(next.is_empty());

    match position.kind() {
        PositionKind::Normal => {
            transition_standard_into(position, cv, query_length, max_distance, prefix_mode, next);

            let Some(remaining_budget) = max_distance.checked_sub(position.num_errors) else {
                return;
            };
            let max_delta = remaining_budget
                .min(cv.len().saturating_sub(1))
                .min(usize::from(u8::MAX));

            for (delta, &matches) in cv.iter().enumerate().take(max_delta + 1).skip(1) {
                if !matches {
                    continue;
                }
                let Some(num_errors) = position.num_errors.checked_add(delta) else {
                    continue;
                };
                let Ok(delta) = u8::try_from(delta) else {
                    continue;
                };
                next.push(Position::new_damerau_pending(
                    position.term_index,
                    num_errors,
                    delta,
                ));
            }
        }
        PositionKind::DamerauPending => {
            let delta = usize::from(position.aux());

            if cv.first().copied().unwrap_or(false) {
                let term_offset = delta.checked_add(1);
                if let Some(term_index) =
                    term_offset.and_then(|offset| position.term_index.checked_add(offset))
                {
                    next.push(Position::new(term_index, position.num_errors));
                }
            }

            if let Some(num_errors) = position.num_errors.checked_add(1) {
                if num_errors <= max_distance {
                    next.push(Position::new_damerau_pending(
                        position.term_index,
                        num_errors,
                        position.aux(),
                    ));
                }
            }
        }
        _ => {}
    }
}

/// Transposition algorithm transition (adds swap of adjacent chars)
///
/// This implements the transposition Levenshtein distance which allows
/// swapping of two adjacent characters as a single edit operation.
#[inline]
pub(crate) fn transition_transposition(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    transition_transposition_into(
        position,
        cv,
        query_length,
        max_distance,
        prefix_mode,
        &mut next,
    );
    next
}

/// Fill a caller-owned buffer with OSA successors.
#[inline(always)]
pub(crate) fn transition_transposition_into(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
    next: &mut SmallVec<[Position; 4]>,
) {
    debug_assert!(next.is_empty());
    let i = position.term_index;
    let e = position.num_errors;
    let t = position.is_special(); // Transposition flag
    let h = 0; // cv is offset-adjusted
    let w = cv.len();

    // Prefix matching
    if prefix_mode && i >= query_length {
        next.push(Position::new(i, e));
        return;
    }

    // Case 1: e == 0 (no errors yet)
    if e == 0 && max_distance > 0 {
        if h + 2 <= w {
            let a = remaining_error_window(max_distance, 0);
            let b = w - h;
            let k = a.min(b);

            match index_of_match(cv, h, k) {
                Some(0) => {
                    // Immediate match
                    push_checked_position(next, checked_position_with_offsets(i, 1, 0, 0));
                }
                Some(1) => {
                    // Match at next position - potential transposition
                    next.push(Position::new(i, 1)); // insertion
                    next.push(Position::with_kind(i, 1, PositionKind::OsaTransposing, 0));
                    push_checked_position(next, checked_position_with_offsets(i, 1, 1, 0));
                    push_checked_position(next, checked_position_with_offsets(i, 2, 1, 0));
                }
                Some(j) => {
                    // Match found at position j > 1
                    next.push(Position::new(i, 1)); // insertion
                    push_checked_position(next, checked_position_with_offsets(i, 1, 1, 0));
                    push_checked_position(next, checked_deletion_position(i, 0, j));
                }
                None => {
                    // No match found
                    next.push(Position::new(i, 1)); // insertion
                    push_checked_position(next, checked_position_with_offsets(i, 1, 1, 0));
                }
            }
        } else if h + 1 == w {
            if cv[h] {
                push_checked_position(next, checked_position_with_offsets(i, 1, 0, 0));
            } else {
                next.push(Position::new(i, 1));
                push_checked_position(next, checked_position_with_offsets(i, 1, 1, 0));
            }
        } else {
            next.push(Position::new(i, 1));
        }
    }
    // Case 2: 1 <= e < max_distance
    else if e >= 1 && e < max_distance {
        if h + 2 <= w {
            if !t {
                // Not in transposition state
                let a = remaining_error_window(max_distance, e);
                let b = w - h;
                let k = a.min(b);

                match index_of_match(cv, h, k) {
                    Some(0) => {
                        push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
                    }
                    Some(1) => {
                        push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                        push_checked_position(
                            next,
                            checked_special_position_with_offsets(
                                i,
                                0,
                                e,
                                1,
                                PositionKind::OsaTransposing,
                            ),
                        );
                        push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                        push_checked_position(next, checked_position_with_offsets(i, 2, e, 1));
                    }
                    Some(j) => {
                        push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                        push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                        push_checked_position(next, checked_deletion_position(i, e, j));
                    }
                    None => {
                        push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                        push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                    }
                }
            } else {
                // In transposition state (is_special == true)
                if cv[h] {
                    // Complete the transposition by matching
                    push_checked_position(next, checked_position_with_offsets(i, 2, e, 0));
                }
                // else: no valid transitions from failed transposition
            }
        } else if h + 1 == w {
            if cv[h] {
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            } else {
                push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
            }
        } else {
            push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
        }
    }
    // Case 3: e == max_distance (at max errors)
    else if e == max_distance {
        if h < w && !t {
            if cv[h] {
                push_checked_position(next, checked_position_with_offsets(i, 1, max_distance, 0));
            }
            // else: no transitions at max distance without match
        } else if h + 2 <= w && t && cv[h] {
            // Complete transposition at max distance
            push_checked_position(next, checked_position_with_offsets(i, 2, max_distance, 0));
        }
    }
}

/// Merge and split algorithm transition
///
/// This implements merge-and-split operations:
/// - Merge: Two query characters combine into one dictionary character
/// - Split: One query character expands into two dictionary characters
#[inline]
pub(crate) fn transition_merge_split(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    transition_merge_split_into(
        position,
        cv,
        query_length,
        max_distance,
        prefix_mode,
        &mut next,
    );
    next
}

/// Fill a caller-owned buffer with merge/split successors.
#[inline(always)]
pub(crate) fn transition_merge_split_into(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
    next: &mut SmallVec<[Position; 4]>,
) {
    debug_assert!(next.is_empty());
    let i = position.term_index;
    let e = position.num_errors;
    let s = position.is_special(); // Special flag for merge/split
    let h = 0; // cv is offset-adjusted
    let w = cv.len();

    // Prefix matching
    if prefix_mode && i >= query_length {
        next.push(Position::new(i, e));
        return;
    }

    // Case 1: e == 0 (no errors yet)
    if e == 0 && max_distance > 0 {
        if h + 2 <= w {
            if cv[h] {
                // Immediate match
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            } else {
                // No match - add error operations including merge/split
                push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                if i < query_length {
                    push_checked_position(
                        next,
                        checked_special_position_with_offsets(i, 0, e, 1, PositionKind::Splitting),
                    );
                }
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                // Merge operation: skip 2 query chars (only if we have 2 chars available)
                if has_query_units(i, 2, query_length) {
                    push_checked_position(next, checked_position_with_offsets(i, 2, e, 1));
                }
            }
        } else if h + 1 == w {
            if cv[h] {
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            } else {
                push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                if i < query_length {
                    push_checked_position(
                        next,
                        checked_special_position_with_offsets(i, 0, e, 1, PositionKind::Splitting),
                    );
                }
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
            }
        } else {
            push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
        }
    }
    // Case 2: e < max_distance (can still add errors)
    else if e < max_distance {
        if h + 2 <= w {
            if !s {
                // Not in special state
                if cv[h] {
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
                } else {
                    push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                    // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                    if i < query_length {
                        push_checked_position(
                            next,
                            checked_special_position_with_offsets(
                                i,
                                0,
                                e,
                                1,
                                PositionKind::Splitting,
                            ),
                        );
                    }
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                    // Merge operation: skip 2 query chars (only if we have 2 chars available)
                    if has_query_units(i, 2, query_length) {
                        push_checked_position(next, checked_position_with_offsets(i, 2, e, 1));
                    }
                }
            } else {
                // In special state (completing split)
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            }
        } else if h + 1 == w {
            if !s {
                if cv[h] {
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
                } else {
                    push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
                    // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                    if i < query_length {
                        push_checked_position(
                            next,
                            checked_special_position_with_offsets(
                                i,
                                0,
                                e,
                                1,
                                PositionKind::Splitting,
                            ),
                        );
                    }
                    push_checked_position(next, checked_position_with_offsets(i, 1, e, 1));
                }
            } else {
                // Special state at boundary
                push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
            }
        } else {
            push_checked_position(next, checked_position_with_offsets(i, 0, e, 1));
        }
    }
    // Case 3: e == max_distance (at max errors)
    else if e == max_distance && h < w {
        if !s {
            if cv[h] {
                push_checked_position(next, checked_position_with_offsets(i, 1, max_distance, 0));
            }
            // else: no transitions at max distance without match
        } else {
            // Special state: can advance even at max distance (completing split)
            push_checked_position(next, checked_position_with_offsets(i, 1, e, 0));
        }
    }
}

/// Compute epsilon closure: add positions reachable by deletion (skipping query chars)
/// without consuming dictionary characters.
///
/// Optimized version that modifies the state in-place to avoid cloning.
#[inline]
fn epsilon_closure_mut<V: AutomatonVariant>(state: &mut State, ctx: &TransitionCtx<V::Params>) {
    // Pre-allocate with typical size to avoid reallocation
    let mut to_process: SmallVec<[Position; 8]> = SmallVec::with_capacity(8);

    // Start with current positions
    for pos in state.positions() {
        to_process.push(*pos);
    }

    let mut processed = 0;
    let mut epsilon = SmallVec::<[Position; 4]>::new();
    while processed < to_process.len() {
        let position = to_process[processed];
        processed += 1;

        epsilon.clear();
        V::epsilon_successors(position, ctx, &mut epsilon);
        for deleted in epsilon.iter().copied() {
            // Try to insert - State.insert handles deduplication efficiently
            // Only add to to_process if it was actually inserted (new position)
            if state.insert_with::<V>(deleted, ctx) {
                to_process.push(deleted);
            }
        }
    }
}

/// Wrapper that creates a new state and applies epsilon closure
fn epsilon_closure<V: AutomatonVariant>(state: &State, ctx: &TransitionCtx<V::Params>) -> State {
    let mut result = state.clone();
    epsilon_closure_mut::<V>(&mut result, ctx);
    result
}

/// Compute epsilon closure from source into target state (pool-friendly).
///
/// This function copies positions from the source state into the target state
/// and then applies epsilon closure in-place. The target state is cleared first.
///
/// # Performance
///
/// This is optimized for use with StatePool:
/// - Target state's Vec allocation is reused
/// - Position is Copy, so no clone overhead
/// - Eliminates one State::clone compared to epsilon_closure()
#[inline]
fn epsilon_closure_into<V: AutomatonVariant>(
    source: &State,
    target: &mut State,
    ctx: &TransitionCtx<V::Params>,
) {
    // Copy positions from source to target
    target.copy_from(source);

    // Apply epsilon closure in-place
    epsilon_closure_mut::<V>(target, ctx);
}

#[inline]
fn transition_zero_distance_into<
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    V: AutomatonVariant,
>(
    state: &State,
    target: &mut State,
    policy: &P,
    dict_unit: U,
    query: &[U],
    ctx: &TransitionCtx<V::Params>,
) {
    let query_length = ctx.query_length;

    for position in state.positions() {
        if ctx.prefix_mode && position.term_index >= query_length {
            target.insert_with::<V>(Position::new(position.term_index, position.num_errors), ctx);
            continue;
        }

        if position.num_errors != 0 {
            continue;
        }

        let Some(&query_unit) = query.get(position.term_index) else {
            continue;
        };

        if query_unit == dict_unit || policy.is_allowed_for(dict_unit, query_unit) {
            if let Some(next_position) =
                checked_position_with_offsets(position.term_index, 1, position.num_errors, 0)
            {
                target.insert_with::<V>(next_position, ctx);
            }
        }
    }
}

#[inline]
fn supports_zero_distance_fast_path(state: &State) -> bool {
    state
        .positions()
        .iter()
        .all(|position| !position.is_special() && position.num_errors == 0)
}

/// Transition an entire state given a dictionary character unit.
///
/// Computes the next state by transitioning all positions in the
/// current state and merging the results.
pub fn transition_state<U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
    state: &State,
    policy: P,
    dict_unit: U,
    query: &[U],
    max_distance: usize,
    algorithm: Algorithm,
    prefix_mode: bool,
) -> Option<State> {
    algorithm.assert_supported_max_distance(max_distance);
    let ctx = TransitionCtx::unit(query.len(), max_distance, prefix_mode);
    with_variant!(VariantSpec::from(algorithm), |V| {
        transition_state_inner::<U, P, V>(state, &policy, dict_unit, query, &ctx)
    })
}

#[inline]
fn transition_state_inner<
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    V: AutomatonVariant,
>(
    state: &State,
    policy: &P,
    dict_unit: U,
    query: &[U],
    ctx: &TransitionCtx<V::Params>,
) -> Option<State> {
    crate::causal_perf::record_transition_attempts(1);
    if ctx.max_distance == 0
        && V::supports_zero_distance_fast_path(ctx)
        && supports_zero_distance_fast_path(state)
    {
        let mut next_state = State::new();
        transition_zero_distance_into::<U, P, V>(
            state,
            &mut next_state,
            policy,
            dict_unit,
            query,
            ctx,
        );
        return (!next_state.is_empty()).then_some(next_state);
    }

    // First, compute epsilon closure to handle deletions
    let expanded_state = epsilon_closure::<V>(state, ctx);

    let mut next_state = State::new();
    let mut cv = SmallVec::<[bool; 8]>::new();
    let mut next_positions = SmallVec::<[Position; 4]>::new();

    for position in expanded_state.positions() {
        let offset = position.term_index;
        let window_size = V::skip_window(position, ctx);
        let cv = characteristic_vector(policy, dict_unit, query, window_size, offset, &mut cv);

        next_positions.clear();
        V::successors(*position, cv, ctx, &mut next_positions);
        for next_pos in next_positions.iter().copied() {
            next_state.insert_with::<V>(next_pos, ctx);
        }
    }

    if next_state.is_empty() {
        None
    } else {
        Some(next_state)
    }
}

/// Transition a state using a StatePool for allocation reuse (optimized).
///
/// This is the pool-aware version of `transition_state` that eliminates
/// State cloning overhead by reusing allocations from the pool.
///
/// # Performance
///
/// Compared to `transition_state`:
/// - Eliminates Vec allocation for expanded_state (reuses from pool)
/// - Eliminates Vec allocation for next_state (reuses from pool)
/// - Reduces State::clone overhead by ~6-10% of total runtime
///
/// # Arguments
///
/// * `state` - Current automaton state
/// * `pool` - State pool for allocation reuse
/// * `policy` - Substitution policy for character matching
/// * `dict_unit` - Dictionary character unit to transition on
/// * `query` - Query term units (bytes or chars)
/// * `settings` - Maximum distance, algorithm, and prefix-mode configuration
///
/// # Returns
///
/// The next state after transitioning, or None if no valid transitions exist.
/// The returned state is acquired from the pool (caller should release it when done).
#[inline]
pub fn transition_state_pooled<U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
    state: &State,
    pool: &mut StatePool,
    policy: P,
    dict_unit: U,
    query: &[U],
    settings: TransitionSettings,
) -> Option<State> {
    transition_state_pooled_ref(state, pool, &policy, dict_unit, query, settings)
}

/// Transition a state using a borrowed policy and a StatePool for allocation reuse.
///
/// This preserves the public by-value `transition_state_pooled` API while letting
/// internal iterators avoid cloning non-ZST policies once per explored edge.
#[inline]
pub(crate) fn transition_state_pooled_ref<
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
>(
    state: &State,
    pool: &mut StatePool,
    policy: &P,
    dict_unit: U,
    query: &[U],
    settings: TransitionSettings,
) -> Option<State> {
    settings
        .algorithm
        .assert_supported_max_distance(settings.max_distance);
    let ctx = TransitionCtx::unit(query.len(), settings.max_distance, settings.prefix_mode);
    with_variant!(VariantSpec::from(settings.algorithm), |V| {
        transition_state_pooled_inner::<U, P, V>(state, pool, policy, dict_unit, query, &ctx)
    })
}

/// Transition from a state that is already epsilon-closed and close the
/// accepted result before returning it.
///
/// This is the query-iterator fast path. Its caller maintains the invariant
/// that the initial state and every queued successor are epsilon-closed for
/// the same transition context. A dictionary node can therefore share one
/// closure across all of its outgoing labels instead of copying and closing
/// the source independently for every edge.
#[inline]
pub(crate) fn transition_epsilon_closed_state_pooled_cached(
    state: &State,
    pool: &mut StatePool,
    matches: &[bool],
    query_length: usize,
    settings: TransitionSettings,
) -> Option<State> {
    settings
        .algorithm
        .assert_supported_max_distance(settings.max_distance);
    let ctx = TransitionCtx::unit(query_length, settings.max_distance, settings.prefix_mode);
    let characteristics = CachedCharacteristics {
        matches,
        query_length,
    };
    with_variant!(VariantSpec::from(settings.algorithm), |V| {
        transition_epsilon_closed_state_pooled_with::<V, _>(state, pool, &characteristics, &ctx)
    })
}

pub(crate) trait CharacteristicProvider {
    fn query_length(&self) -> usize;

    fn window<'a>(
        &'a self,
        offset: usize,
        window_size: usize,
        scratch: &'a mut SmallVec<[bool; 8]>,
    ) -> &'a [bool];
}

pub(crate) struct OnDemandCharacteristics<'a, U: CharUnit, P> {
    policy: &'a P,
    dict_unit: U,
    query: &'a [U],
}

impl<'a, U: CharUnit, P> OnDemandCharacteristics<'a, U, P> {
    pub(crate) const fn new(policy: &'a P, dict_unit: U, query: &'a [U]) -> Self {
        Self {
            policy,
            dict_unit,
            query,
        }
    }
}

impl<U, P> CharacteristicProvider for OnDemandCharacteristics<'_, U, P>
where
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
{
    #[inline(always)]
    fn query_length(&self) -> usize {
        self.query.len()
    }

    #[inline(always)]
    fn window<'a>(
        &'a self,
        offset: usize,
        window_size: usize,
        scratch: &'a mut SmallVec<[bool; 8]>,
    ) -> &'a [bool] {
        characteristic_vector(
            self.policy,
            self.dict_unit,
            self.query,
            window_size,
            offset,
            scratch,
        )
    }
}

pub(crate) struct CachedCharacteristics<'a> {
    matches: &'a [bool],
    query_length: usize,
}

impl<'a> CachedCharacteristics<'a> {
    pub(crate) const fn new(matches: &'a [bool], query_length: usize) -> Self {
        Self {
            matches,
            query_length,
        }
    }
}

impl CharacteristicProvider for CachedCharacteristics<'_> {
    #[inline(always)]
    fn query_length(&self) -> usize {
        self.query_length
    }

    #[inline(always)]
    fn window<'a>(
        &'a self,
        offset: usize,
        window_size: usize,
        scratch: &'a mut SmallVec<[bool; 8]>,
    ) -> &'a [bool] {
        let start = offset.min(self.query_length);
        if let Some(end) = start.checked_add(window_size) {
            if let Some(window) = self.matches.get(start..end) {
                return window;
            }
        }

        // Defensive fallback for a caller whose requested window exceeds the
        // cache's construction bound. Public distance validation normally
        // makes this unreachable, but retaining it keeps extreme inputs safe.
        scratch.clear();
        scratch.resize(window_size, false);
        scratch.as_slice()
    }
}

#[inline]
fn transition_state_pooled_inner<
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
    V: AutomatonVariant,
>(
    state: &State,
    pool: &mut StatePool,
    policy: &P,
    dict_unit: U,
    query: &[U],
    ctx: &TransitionCtx<V::Params>,
) -> Option<State> {
    let characteristics = OnDemandCharacteristics {
        policy,
        dict_unit,
        query,
    };
    transition_state_pooled_with::<V, _>(state, pool, &characteristics, ctx)
}

#[inline]
fn transition_state_pooled_with<V: AutomatonVariant, C: CharacteristicProvider>(
    state: &State,
    pool: &mut StatePool,
    characteristics: &C,
    ctx: &TransitionCtx<V::Params>,
) -> Option<State> {
    crate::causal_perf::record_transition_attempts(1);
    if ctx.max_distance == 0
        && V::supports_zero_distance_fast_path(ctx)
        && supports_zero_distance_fast_path(state)
    {
        let mut next_state = pool.acquire();
        transition_zero_distance_with_characteristics::<V, C>(
            state,
            &mut next_state,
            characteristics,
            ctx,
        );

        if next_state.is_empty() {
            pool.release(next_state);
            return None;
        }

        return Some(next_state);
    }

    // Acquire a state from pool for epsilon closure (reuses allocation!)
    let mut expanded_state = pool.acquire();

    // Compute epsilon closure into the pooled state (no clone!)
    crate::causal_perf::record_epsilon_input_positions(state.len() as u64);
    epsilon_closure_into::<V>(state, &mut expanded_state, ctx);
    crate::causal_perf::record_epsilon_output_positions(expanded_state.len() as u64);

    // Acquire another state from pool for next state
    let mut next_state = pool.acquire();
    transition_epsilon_closed_state_into::<V, C>(
        &expanded_state,
        &mut next_state,
        characteristics,
        ctx,
    );

    // Return expanded_state to pool (no longer needed)
    pool.release(expanded_state);

    if next_state.is_empty() {
        // Return empty state to pool
        pool.release(next_state);
        None
    } else {
        Some(next_state)
    }
}

/// Pooled transition kernel for the epsilon-closed queued-state invariant.
///
/// The source closure is label-independent, so the iterator stores it once.
/// The non-empty output is closed in place before it is queued. Both this
/// path and the compatibility path above share the same monomorphized
/// per-position transition kernel.
#[inline]
fn transition_epsilon_closed_state_pooled_with<V: AutomatonVariant, C: CharacteristicProvider>(
    state: &State,
    pool: &mut StatePool,
    characteristics: &C,
    ctx: &TransitionCtx<V::Params>,
) -> Option<State> {
    crate::causal_perf::record_transition_attempts(1);
    let mut next_state = pool.acquire();

    if ctx.max_distance == 0
        && V::supports_zero_distance_fast_path(ctx)
        && supports_zero_distance_fast_path(state)
    {
        transition_zero_distance_with_characteristics::<V, C>(
            state,
            &mut next_state,
            characteristics,
            ctx,
        );
    } else {
        transition_epsilon_closed_state_into::<V, C>(state, &mut next_state, characteristics, ctx);
    }

    if next_state.is_empty() {
        pool.release(next_state);
        return None;
    }

    // Establish the invariant consumed by the next dictionary node. Closing
    // here performs the work once per accepted state rather than once per
    // outgoing edge of that state.
    crate::causal_perf::record_epsilon_input_positions(next_state.len() as u64);
    epsilon_closure_mut::<V>(&mut next_state, ctx);
    crate::causal_perf::record_epsilon_output_positions(next_state.len() as u64);

    Some(next_state)
}

/// Consume one dictionary unit from an epsilon-closed source state.
///
/// Keeping this as the single generic inner loop avoids duplicating the hot
/// algorithm logic between the public compatibility path and the queued-state
/// fast path. `V` and `C` remain statically dispatched and monomorphized.
#[inline]
fn transition_epsilon_closed_state_into<V: AutomatonVariant, C: CharacteristicProvider>(
    state: &State,
    next_state: &mut State,
    characteristics: &C,
    ctx: &TransitionCtx<V::Params>,
) {
    let mut cv = SmallVec::<[bool; 8]>::new();
    let mut next_positions = SmallVec::<[Position; 4]>::new();

    for position in state.positions() {
        let offset = position.term_index;
        let window_size = V::skip_window(position, ctx);
        crate::causal_perf::record_characteristic_vectors(1);
        crate::causal_perf::record_characteristic_units(window_size as u64);
        let cv = characteristics.window(offset, window_size, &mut cv);

        next_positions.clear();
        V::successors(*position, cv, ctx, &mut next_positions);
        crate::causal_perf::record_successor_candidates(next_positions.len() as u64);
        for next_pos in next_positions.iter().copied() {
            next_state.insert_with::<V>(next_pos, ctx);
        }
    }
}

#[inline]
fn transition_zero_distance_with_characteristics<V: AutomatonVariant, C: CharacteristicProvider>(
    state: &State,
    target: &mut State,
    characteristics: &C,
    ctx: &TransitionCtx<V::Params>,
) {
    let query_length = characteristics.query_length();
    let mut scratch = SmallVec::<[bool; 8]>::new();

    for position in state.positions() {
        if ctx.prefix_mode && position.term_index >= query_length {
            target.insert_with::<V>(Position::new(position.term_index, position.num_errors), ctx);
            continue;
        }
        if position.num_errors != 0 {
            continue;
        }
        if characteristics
            .window(position.term_index, 1, &mut scratch)
            .first()
            .copied()
            .unwrap_or(false)
        {
            if let Some(next_position) =
                checked_position_with_offsets(position.term_index, 1, position.num_errors, 0)
            {
                target.insert_with::<V>(next_position, ctx);
            }
        }
    }
}

/// Create the initial state for a query.
///
/// The initial state contains positions representing all possible
/// ways to start matching (including initial errors via deletions/insertions).
pub fn initial_state(query_length: usize, max_distance: usize, algorithm: Algorithm) -> State {
    algorithm.assert_supported_max_distance(max_distance);
    let ctx = TransitionCtx::unit(query_length, max_distance, false);
    with_variant!(VariantSpec::from(algorithm), |V| {
        initial_state_with::<V>(&ctx)
    })
}

#[inline]
fn initial_state_with<V: AutomatonVariant>(ctx: &TransitionCtx<V::Params>) -> State {
    let mut state = State::new();

    // Start at position (0, 0) - no errors, beginning of query
    state.insert_with::<V>(Position::new(0, 0), ctx);

    // Seed every legal query-prefix gap. For unit-cost variants this is the
    // historical `(i, i)` prefix; affine uses the same fixpoint shape with its
    // layer-aware open/extension costs.
    epsilon_closure_mut::<V>(&mut state, ctx);

    state
}

/// Create the initial state for an exact scaled affine-gap query.
pub(crate) fn initial_state_affine(
    query_length: usize,
    max_cost: usize,
    params: AffineGapParams,
) -> State {
    // Keep the runtime seam's marker executable even though parameters require
    // the typed path below rather than the unit-parameter dispatch macro.
    let _spec = VariantSpec::AffineGap;
    let ctx = TransitionCtx::new(query_length, max_cost, false, params);
    initial_state_with::<AffineV>(&ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::{Restricted, SubstitutionSet, Unrestricted};

    #[test]
    fn test_characteristic_vector() {
        let query = b"test";
        let policy = Unrestricted;
        let mut buffer = SmallVec::<[bool; 8]>::new();

        let cv = characteristic_vector(&policy, b't', query, 3, 0, &mut buffer);
        assert_eq!(cv, &[true, false, false]);

        let cv = characteristic_vector(&policy, b'e', query, 3, 0, &mut buffer);
        assert_eq!(cv, &[false, true, false]);

        let cv = characteristic_vector(&policy, b's', query, 3, 1, &mut buffer);
        assert_eq!(cv, &[false, true, false]);
    }

    #[test]
    fn test_characteristic_vector_supports_large_windows() {
        let query = b"abcdefghijklmnop";
        let policy = Unrestricted;
        let mut buffer = SmallVec::<[bool; 8]>::new();

        let cv = characteristic_vector(&policy, b'j', query, 12, 0, &mut buffer);

        assert_eq!(cv.len(), 12);
        assert!(cv[9]);
    }

    #[test]
    fn characteristic_cache_extends_existing_vectors_for_specialized_windows() {
        let query = b"test";
        let policy = Unrestricted;
        let mut cache = CharacteristicCache::new(query.len(), 1);

        let before = cache.matches_for(&policy, b't', query).to_vec();
        assert_eq!(before.len(), query.len() + 2);

        cache.ensure_padding(8);
        let after = cache.matches_for(&policy, b't', query);
        assert_eq!(after.len(), query.len() + 8);
        assert_eq!(&after[..query.len()], &[true, false, false, true]);
        assert!(after[query.len()..].iter().all(|matched| !matched));
    }

    #[test]
    fn checked_transition_arithmetic_helpers_handle_boundaries() {
        assert_eq!(checked_query_index(usize::MAX, 1), None);
        assert_eq!(checked_query_index(usize::MAX - 1, 1), Some(usize::MAX));

        assert_eq!(checked_window_end(1, usize::MAX, 4), Some(4));
        assert_eq!(checked_window_end(5, 1, 4), None);

        assert_eq!(transition_window_size(usize::MAX, 0), 1);
        assert_eq!(transition_window_size(2, usize::MAX), 3);

        assert_eq!(remaining_error_window(5, 2), 4);
        assert_eq!(remaining_error_window(usize::MAX, 0), usize::MAX);
        assert_eq!(remaining_error_window(2, 5), 0);

        assert_eq!(checked_position_with_offsets(usize::MAX, 1, 0, 0), None);
        assert_eq!(checked_position_with_offsets(0, 1, usize::MAX, 1), None);
        assert_eq!(checked_deletion_position(usize::MAX, 0, 0), None);
        assert_eq!(checked_deletion_position(0, 0, usize::MAX), None);
        assert!(!has_query_units(usize::MAX, 1, usize::MAX));
    }

    #[test]
    fn damerau_budget_ceiling_accepts_the_largest_representable_delta() {
        let state = initial_state(
            0,
            Algorithm::MAX_DAMERAU_DISTANCE,
            Algorithm::DamerauLevenshtein,
        );
        assert_eq!(state.len(), 1);
    }

    #[test]
    #[should_panic(expected = "Algorithm::DamerauLevenshtein supports max_distance <= 255")]
    fn damerau_budget_ceiling_rejects_incomplete_semantics() {
        let _ = transition_position(
            &Position::new(0, 0),
            &[false; 257],
            257,
            Algorithm::MAX_DAMERAU_DISTANCE + 1,
            Algorithm::DamerauLevenshtein,
            false,
        );
    }

    #[test]
    fn characteristic_vector_overflowing_offset_is_unmatched() {
        let query = b"a";
        let policy = Unrestricted;
        let mut buffer = SmallVec::<[bool; 8]>::new();

        let cv = characteristic_vector(&policy, b'a', query, 2, usize::MAX, &mut buffer);

        assert_eq!(cv, &[false, false]);
    }

    #[test]
    fn transition_standard_drops_overflowing_successor() {
        let pos = Position::new(usize::MAX, 0);
        let cv = vec![true];

        let next = transition_standard(&pos, &cv, usize::MAX, 1, false);

        assert!(next.is_empty());
    }

    #[test]
    fn transition_merge_split_rejects_overflowing_two_unit_check() {
        assert!(!has_query_units(usize::MAX - 1, 2, usize::MAX));

        let pos = Position::new(usize::MAX - 1, 0);
        let cv = vec![false, false];
        let next = transition_merge_split(&pos, &cv, usize::MAX, 1, false);

        assert!(!next.contains(&Position::new(0, 1)));
        assert!(next.contains(&Position::new(usize::MAX, 1)));
        assert!(next.contains(&Position::new(usize::MAX - 1, 1)));
    }

    #[test]
    fn test_transition_standard_match() {
        let pos = Position::new(0, 0);
        let cv = vec![true, false, false]; // Matches at position 0
        let query_length = 4; // e.g., "test"
        let next = transition_standard(&pos, &cv, query_length, 2, false);

        // Should advance with no error on match
        assert!(next.contains(&Position::new(1, 0)));
    }

    #[test]
    fn test_transition_standard_operations() {
        let pos = Position::new(1, 0);
        let cv = vec![false, false, true]; // No match at position 1
        let query_length = 4; // e.g., "test"
        let next = transition_standard(&pos, &cv, query_length, 2, false);

        // Should include:
        // - Substitution: (2, 1)
        // - Insertion: (1, 1)
        // - Deletion: (2, 1)
        assert!(next.len() >= 2);
        assert!(next.contains(&Position::new(1, 1))); // Insertion
    }

    #[test]
    fn test_initial_state() {
        let state = initial_state(5, 2, Algorithm::Standard);

        // With Standard subsumption, (0,0) subsumes both (1,1) and (2,2)
        // because |0-1|=1 <= (1-0)=1 and |0-2|=2 <= (2-0)=2
        // So only (0,0) remains in the initial state.
        //
        // This is correct: (0,0) can reach everything that (1,1) and (2,2)
        // can reach, so keeping only (0,0) is sufficient and more efficient.
        assert_eq!(state.len(), 1);
        assert!(state.positions().contains(&Position::new(0, 0)));
    }

    #[test]
    fn test_transition_state() {
        let query = b"test";
        let max_distance = 2;
        let mut state = State::new();
        state.insert(Position::new(0, 0), Algorithm::Standard, max_distance);
        let policy = Unrestricted;

        let next = transition_state(
            &state,
            policy,
            b't',
            query,
            max_distance,
            Algorithm::Standard,
            false,
        );
        assert!(next.is_some());

        let next_state = next.expect("test fixture: transition produces Some (asserted above)");
        // Should have advanced after matching 't'
        assert!(next_state.positions().iter().any(|p| p.term_index > 0));
    }

    #[test]
    fn test_zero_distance_transition_preserves_restricted_substitution() {
        let query = b"kat";
        let state = initial_state(query.len(), 0, Algorithm::Standard);

        let mut substitutions = SubstitutionSet::new();
        substitutions.allow('c', 'k');
        let policy = Restricted::new(&substitutions);

        let next = transition_state(&state, policy, b'c', query, 0, Algorithm::Standard, false)
            .expect("restricted c->k substitution should advance at distance zero");

        assert!(next.positions().contains(&Position::new(1, 0)));

        let mut pool = StatePool::new();
        let pooled = transition_state_pooled(
            &state,
            &mut pool,
            policy,
            b'c',
            query,
            TransitionSettings::new(0, Algorithm::Standard, false),
        )
        .expect("pooled restricted c->k substitution should advance at distance zero");

        assert_eq!(pooled.positions(), next.positions());

        let mut pool = StatePool::new();
        let borrowed = transition_state_pooled_ref(
            &state,
            &mut pool,
            &policy,
            b'c',
            query,
            TransitionSettings::new(0, Algorithm::Standard, false),
        )
        .expect("borrowed-policy pooled restricted c->k substitution should advance");

        assert_eq!(borrowed.positions(), next.positions());
    }

    #[test]
    fn test_zero_distance_transition_rejects_unmatched_unit() {
        let query = b"kat";
        let state = initial_state(query.len(), 0, Algorithm::Standard);
        let policy = Unrestricted;

        assert!(
            transition_state(&state, policy, b'c', query, 0, Algorithm::Standard, false).is_none()
        );
    }

    #[test]
    fn test_zero_distance_transition_preserves_prefix_mode_after_query() {
        let query = b"cat";
        let state = State::single(Position::new(query.len(), 0));
        let policy = Unrestricted;

        let next = transition_state(&state, policy, b's', query, 0, Algorithm::Standard, true)
            .expect("prefix mode should keep accepting after query is consumed");

        assert_eq!(next.positions(), &[Position::new(query.len(), 0)]);

        assert!(
            transition_state(&state, policy, b's', query, 0, Algorithm::Standard, false).is_none()
        );
    }

    #[test]
    fn test_zero_distance_transition_falls_back_for_special_positions() {
        let query = b"a";
        let state = State::single(Position::new_osa_transposing(0, 0));
        let policy = Unrestricted;

        let next = transition_state(
            &state,
            policy,
            b'x',
            query,
            0,
            Algorithm::MergeAndSplit,
            false,
        )
        .expect("special merge/split state should use the general transition path");

        assert_eq!(next.positions(), &[Position::new(1, 0)]);
    }
}
