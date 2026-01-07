//! State transition logic for Levenshtein automata.

use super::{Algorithm, Position, State, StatePool, SubstitutionPolicy, SubstitutionPolicyFor};
use libdictenstein::CharUnit;
use smallvec::SmallVec;

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
/// * `buffer` - Stack-allocated buffer to write results into
///
/// # Returns
/// Slice of booleans indicating positions where characters can match without error.
/// This includes both exact matches AND policy-allowed substitutions.
/// Uses stack-allocated array (max 8 elements) to avoid heap allocations.
///
/// # Semantics
///
/// For unrestricted policies, only exact character matches are allowed (traditional Levenshtein).
/// For restricted policies, the policy determines if a substitution should be cost-free.
/// This treats policy-allowed substitutions as "equivalences" rather than edits.
///
/// # Performance
///
/// The substitution policy check is monomorphized, so the `Unrestricted`
/// policy compiles to identical code as the pre-generic implementation
/// (zero overhead).
#[inline]
fn characteristic_vector<'a, U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
    policy: P,
    dict_unit: U,
    query: &[U],
    window_size: usize,
    offset: usize,
    buffer: &'a mut [bool; 8],
) -> &'a [bool] {
    // Most queries use max_distance ≤ 7, so window_size ≤ 8
    let len = window_size.min(8);

    // The characteristic vector shows which query positions can consume dict_unit without error.
    // This includes:
    // 1. Exact character matches (query[i] == dict_unit)
    // 2. Policy-allowed substitutions (policy.is_allowed(...))
    //
    // For Unrestricted policy: is_allowed always returns false (no zero-cost substitutions)
    // For Restricted policy: is_allowed checks the substitution set

    for (i, item) in buffer.iter_mut().enumerate().take(len) {
        let query_idx = offset + i;
        if query_idx < query.len() {
            let query_unit = query[query_idx];
            *item = query_unit == dict_unit
                || is_substitution_allowed(&policy, dict_unit, query_unit);
        } else {
            *item = false;
        }
    }
    &buffer[..len]
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

/// Transition a position given a characteristic vector.
///
/// Computes all possible next positions from the current position
/// after consuming a dictionary character, considering the query
/// term through the characteristic vector.
///
/// # Standard Algorithm
///
/// From position `(i, e)` (term_index=i, num_errors=e), we can reach:
/// - `(i+1, e)` if query[i] matches dict_char (no error)
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
    (0..limit).find(|&j| cv.get(start + j).copied().unwrap_or(false))
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
fn transition_standard(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    let i = position.term_index;
    let e = position.num_errors;
    let h = 0; // cv is offset-adjusted to start at position i, so h = 0
    let w = cv.len();

    // Prefix matching: if enabled and we've consumed the full query, treat any character as a free match
    if prefix_mode && i >= query_length {
        next.push(Position::new(i, e));
        return next;
    }

    // Case 1: e < max_distance (can still add errors)
    if e < max_distance {
        // Subcase 1a: At least 2 characters remain in query (h + 2 <= w)
        if h + 2 <= w {
            let a = max_distance.saturating_sub(e).saturating_add(1);
            let b = w - h;
            let k = a.min(b);

            match index_of_match(cv, h, k) {
                Some(0) => {
                    // Immediate match at cv[h]
                    next.push(Position::new(i + 1, e));
                }
                Some(j) => {
                    // Match found at cv[h + j]
                    // Return: insertion, substitution, and multi-character deletion
                    next.push(Position::new(i, e + 1)); // insertion
                    next.push(Position::new(i + 1, e + 1)); // substitution
                    next.push(Position::new(i + j + 1, e + j)); // deletion (skip j chars)
                }
                None => {
                    // No match found in range
                    next.push(Position::new(i, e + 1)); // insertion
                    next.push(Position::new(i + 1, e + 1)); // substitution
                }
            }
        }
        // Subcase 1b: Exactly 1 character remains (h + 1 == w)
        else if h + 1 == w {
            if cv[h] {
                // Match at last position
                next.push(Position::new(i + 1, e));
            } else {
                // No match at last position
                next.push(Position::new(i, e + 1)); // insertion
                next.push(Position::new(i + 1, e + 1)); // substitution
            }
        }
        // Subcase 1c: Past the end of query (h >= w)
        else {
            // Only insertion is possible
            next.push(Position::new(i, e + 1));
        }
    }
    // Case 2: e == max_distance (at max errors, only exact matches allowed)
    else if e == max_distance && h < w && cv[h] {
        next.push(Position::new(i + 1, max_distance));
    }

    next
}

/// Transposition algorithm transition (adds swap of adjacent chars)
///
/// This implements the transposition Levenshtein distance which allows
/// swapping of two adjacent characters as a single edit operation.
#[inline]
fn transition_transposition(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    let i = position.term_index;
    let e = position.num_errors;
    let t = position.is_special; // Transposition flag
    let h = 0; // cv is offset-adjusted
    let w = cv.len();

    // Prefix matching
    if prefix_mode && i >= query_length {
        next.push(Position::new(i, e));
        return next;
    }

    // Case 1: e == 0 (no errors yet)
    if e == 0 && max_distance > 0 {
        if h + 2 <= w {
            let a = max_distance.saturating_add(1);
            let b = w - h;
            let k = a.min(b);

            match index_of_match(cv, h, k) {
                Some(0) => {
                    // Immediate match
                    next.push(Position::new(i + 1, 0));
                }
                Some(1) => {
                    // Match at next position - potential transposition
                    next.push(Position::new(i, 1)); // insertion
                    next.push(Position::new_special(i, 1)); // transposition start
                    next.push(Position::new(i + 1, 1)); // substitution
                    next.push(Position::new(i + 2, 1)); // special transposition
                }
                Some(j) => {
                    // Match found at position j > 1
                    next.push(Position::new(i, 1)); // insertion
                    next.push(Position::new(i + 1, 1)); // substitution
                    next.push(Position::new(i + j + 1, j)); // multi-char deletion
                }
                None => {
                    // No match found
                    next.push(Position::new(i, 1)); // insertion
                    next.push(Position::new(i + 1, 1)); // substitution
                }
            }
        } else if h + 1 == w {
            if cv[h] {
                next.push(Position::new(i + 1, 0));
            } else {
                next.push(Position::new(i, 1));
                next.push(Position::new(i + 1, 1));
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
                let a = max_distance.saturating_sub(e).saturating_add(1);
                let b = w - h;
                let k = a.min(b);

                match index_of_match(cv, h, k) {
                    Some(0) => {
                        next.push(Position::new(i + 1, e));
                    }
                    Some(1) => {
                        next.push(Position::new(i, e + 1));
                        next.push(Position::new_special(i, e + 1));
                        next.push(Position::new(i + 1, e + 1));
                        next.push(Position::new(i + 2, e + 1));
                    }
                    Some(j) => {
                        next.push(Position::new(i, e + 1));
                        next.push(Position::new(i + 1, e + 1));
                        next.push(Position::new(i + j + 1, e + j));
                    }
                    None => {
                        next.push(Position::new(i, e + 1));
                        next.push(Position::new(i + 1, e + 1));
                    }
                }
            } else {
                // In transposition state (is_special == true)
                if cv[h] {
                    // Complete the transposition by matching
                    next.push(Position::new(i + 2, e));
                }
                // else: no valid transitions from failed transposition
            }
        } else if h + 1 == w {
            if cv[h] {
                next.push(Position::new(i + 1, e));
            } else {
                next.push(Position::new(i, e + 1));
                next.push(Position::new(i + 1, e + 1));
            }
        } else {
            next.push(Position::new(i, e + 1));
        }
    }
    // Case 3: e == max_distance (at max errors)
    else if e == max_distance {
        if h < w && !t {
            if cv[h] {
                next.push(Position::new(i + 1, max_distance));
            }
            // else: no transitions at max distance without match
        } else if h + 2 <= w && t && cv[h] {
            // Complete transposition at max distance
            next.push(Position::new(i + 2, max_distance));
        }
    }

    next
}

/// Merge and split algorithm transition
///
/// This implements merge-and-split operations:
/// - Merge: Two query characters combine into one dictionary character
/// - Split: One query character expands into two dictionary characters
#[inline]
fn transition_merge_split(
    position: &Position,
    cv: &[bool],
    query_length: usize,
    max_distance: usize,
    prefix_mode: bool,
) -> SmallVec<[Position; 4]> {
    let mut next = SmallVec::new();
    let i = position.term_index;
    let e = position.num_errors;
    let s = position.is_special; // Special flag for merge/split
    let h = 0; // cv is offset-adjusted
    let w = cv.len();

    // Prefix matching
    if prefix_mode && i >= query_length {
        next.push(Position::new(i, e));
        return next;
    }

    // Case 1: e == 0 (no errors yet)
    if e == 0 && max_distance > 0 {
        if h + 2 <= w {
            if cv[h] {
                // Immediate match
                next.push(Position::new(i + 1, e));
            } else {
                // No match - add error operations including merge/split
                next.push(Position::new(i, e + 1)); // insertion
                                                    // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                if i < query_length {
                    next.push(Position::new_special(i, e + 1)); // split start
                }
                next.push(Position::new(i + 1, e + 1)); // substitution
                                                        // Merge operation: skip 2 query chars (only if we have 2 chars available)
                if i + 2 <= query_length {
                    next.push(Position::new(i + 2, e + 1));
                }
            }
        } else if h + 1 == w {
            if cv[h] {
                next.push(Position::new(i + 1, e));
            } else {
                next.push(Position::new(i, e + 1));
                // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                if i < query_length {
                    next.push(Position::new_special(i, e + 1));
                }
                next.push(Position::new(i + 1, e + 1));
            }
        } else {
            next.push(Position::new(i, e + 1));
        }
    }
    // Case 2: e < max_distance (can still add errors)
    else if e < max_distance {
        if h + 2 <= w {
            if !s {
                // Not in special state
                if cv[h] {
                    next.push(Position::new(i + 1, e));
                } else {
                    next.push(Position::new(i, e + 1));
                    // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                    if i < query_length {
                        next.push(Position::new_special(i, e + 1));
                    }
                    next.push(Position::new(i + 1, e + 1));
                    // Merge operation: skip 2 query chars (only if we have 2 chars available)
                    if i + 2 <= query_length {
                        next.push(Position::new(i + 2, e + 1));
                    }
                }
            } else {
                // In special state (completing split)
                next.push(Position::new(i + 1, e));
            }
        } else if h + 1 == w {
            if !s {
                if cv[h] {
                    next.push(Position::new(i + 1, e));
                } else {
                    next.push(Position::new(i, e + 1));
                    // Split operation: one query char becomes two dict chars (only if we have 1 char available)
                    if i < query_length {
                        next.push(Position::new_special(i, e + 1));
                    }
                    next.push(Position::new(i + 1, e + 1));
                }
            } else {
                // Special state at boundary
                next.push(Position::new(i + 1, e));
            }
        } else {
            next.push(Position::new(i, e + 1));
        }
    }
    // Case 3: e == max_distance (at max errors)
    else if e == max_distance && h < w {
        if !s {
            if cv[h] {
                next.push(Position::new(i + 1, max_distance));
            }
            // else: no transitions at max distance without match
        } else {
            // Special state: can advance even at max distance (completing split)
            next.push(Position::new(i + 1, e));
        }
    }

    next
}

/// Compute epsilon closure: add positions reachable by deletion (skipping query chars)
/// without consuming dictionary characters.
///
/// Optimized version that modifies the state in-place to avoid cloning.
#[inline]
fn epsilon_closure_mut(
    state: &mut State,
    query_length: usize,
    max_distance: usize,
    algorithm: Algorithm,
) {
    // Pre-allocate with typical size to avoid reallocation
    let mut to_process: SmallVec<[Position; 8]> = SmallVec::with_capacity(8);

    // Start with current positions
    for pos in state.positions() {
        to_process.push(*pos);
    }

    let mut processed = 0;
    while processed < to_process.len() {
        let position = &to_process[processed];
        processed += 1;

        // Can we delete a query character (skip to next position)?
        if position.num_errors < max_distance && position.term_index < query_length {
            let deleted = Position::new(position.term_index + 1, position.num_errors + 1);

            // Try to insert - State.insert handles deduplication efficiently
            // Only add to to_process if it was actually inserted (new position)
            let len_before = state.len();
            state.insert(deleted, algorithm, query_length);
            if state.len() > len_before {
                to_process.push(deleted);
            }
        }
    }
}

/// Wrapper that creates a new state and applies epsilon closure
fn epsilon_closure(
    state: &State,
    query_length: usize,
    max_distance: usize,
    algorithm: Algorithm,
) -> State {
    let mut result = state.clone();
    epsilon_closure_mut(&mut result, query_length, max_distance, algorithm);
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
fn epsilon_closure_into(
    source: &State,
    target: &mut State,
    query_length: usize,
    max_distance: usize,
    algorithm: Algorithm,
) {
    // Copy positions from source to target
    target.copy_from(source);

    // Apply epsilon closure in-place
    epsilon_closure_mut(target, query_length, max_distance, algorithm);
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
    let window_size = max_distance.saturating_add(1);
    let query_length = query.len();

    // First, compute epsilon closure to handle deletions
    let expanded_state = epsilon_closure(state, query_length, max_distance, algorithm);

    let mut next_state = State::new();

    // Stack-allocated buffer for characteristic vector (avoids heap allocations)
    let mut cv_buffer = [false; 8];

    for position in expanded_state.positions() {
        let offset = position.term_index;
        let cv = characteristic_vector(policy, dict_unit, query, window_size, offset, &mut cv_buffer);

        let next_positions = transition_position(
            position,
            cv,
            query_length,
            max_distance,
            algorithm,
            prefix_mode,
        );

        for next_pos in next_positions {
            next_state.insert(next_pos, algorithm, query_length);
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
/// * `max_distance` - Maximum edit distance
/// * `algorithm` - Algorithm variant (Standard, Transposition, MergeAndSplit)
/// * `prefix_mode` - Enable prefix matching (characters beyond query treated as free matches)
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
    max_distance: usize,
    algorithm: Algorithm,
    prefix_mode: bool,
) -> Option<State> {
    let window_size = max_distance.saturating_add(1);
    let query_length = query.len();

    // Acquire a state from pool for epsilon closure (reuses allocation!)
    let mut expanded_state = pool.acquire();

    // Compute epsilon closure into the pooled state (no clone!)
    epsilon_closure_into(
        state,
        &mut expanded_state,
        query_length,
        max_distance,
        algorithm,
    );

    // Acquire another state from pool for next state
    let mut next_state = pool.acquire();

    // Stack-allocated buffer for characteristic vector
    let mut cv_buffer = [false; 8];

    for position in expanded_state.positions() {
        let offset = position.term_index;
        let cv = characteristic_vector(policy, dict_unit, query, window_size, offset, &mut cv_buffer);

        let next_positions = transition_position(
            position,
            cv,
            query_length,
            max_distance,
            algorithm,
            prefix_mode,
        );

        for next_pos in next_positions {
            next_state.insert(next_pos, algorithm, query_length);
        }
    }

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

/// Create the initial state for a query.
///
/// The initial state contains positions representing all possible
/// ways to start matching (including initial errors via deletions/insertions).
pub fn initial_state(query_length: usize, max_distance: usize, algorithm: Algorithm) -> State {
    let mut state = State::new();

    // Start at position (0, 0) - no errors, beginning of query
    state.insert(Position::new(0, 0), algorithm, query_length);

    // Also add positions for initial deletions (skipping query chars)
    for i in 1..=max_distance.min(query_length) {
        state.insert(Position::new(i, i), algorithm, query_length);
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::Unrestricted;

    #[test]
    fn test_characteristic_vector() {
        let query = b"test";
        let mut buffer = [false; 8];
        let policy = Unrestricted;

        let cv = characteristic_vector(policy, b't', query, 3, 0, &mut buffer);
        assert_eq!(cv, &[true, false, false]);

        let cv = characteristic_vector(policy, b'e', query, 3, 0, &mut buffer);
        assert_eq!(cv, &[false, true, false]);

        let cv = characteristic_vector(policy, b's', query, 3, 1, &mut buffer);
        assert_eq!(cv, &[false, true, false]);
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

        let next_state = next.unwrap();
        // Should have advanced after matching 't'
        assert!(next_state.positions().iter().any(|p| p.term_index > 0));
    }
}
