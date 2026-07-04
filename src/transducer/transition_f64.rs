//! State transition logic for float-weighted Levenshtein automata.
//!
//! This module provides transition functions for automata with configurable
//! float costs per operation type (operation-level weighting).
//!
//! # Operation-Level vs Transition-Level Weighting
//!
//! This module implements **operation-level** weighting where costs are fixed
//! per operation TYPE (e.g., all substitutions cost 1.5). For **transition-level**
//! weighting where costs depend on actual values (like MSM's `|x_i - y_j|`),
//! see the `time_series` module.
//!
//! # Differences from Integer Transitions
//!
//! | Aspect | Integer (`transition.rs`) | Float (`transition_f64.rs`) |
//! |--------|---------------------------|------------------------------|
//! | Cost type | `usize` | `f64` |
//! | Cost source | Hardcoded `+1` | `OperationCostsF64` |
//! | Threshold | `max_distance: usize` | `max_cost: f64` |
//! | Comparison | `e < max_distance` | `cost < max_cost` (with epsilon) |

use super::{
    Algorithm, OperationCostsF64, PositionF64, StateF64, StatePoolF64, SubstitutionPolicy,
    SubstitutionPolicyFor,
};
use crate::numeric::{nonnegative_ceil_to_usize, nonnegative_floor_to_usize_saturating};
use libdictenstein::CharUnit;
use smallvec::SmallVec;

/// Epsilon for float comparisons in cost thresholds.
const COST_EPSILON: f64 = 1e-9;

/// Configuration shared by float-weighted state transitions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionSettingsF64<'a> {
    /// Maximum edit cost.
    pub max_cost: f64,
    /// Edit algorithm variant.
    pub algorithm: Algorithm,
    /// Per-operation cost table.
    pub costs: &'a OperationCostsF64,
    /// Whether positions beyond the query length are free matches.
    pub prefix_mode: bool,
}

impl<'a> TransitionSettingsF64<'a> {
    /// Create float-weighted transition settings.
    pub const fn new(
        max_cost: f64,
        algorithm: Algorithm,
        costs: &'a OperationCostsF64,
        prefix_mode: bool,
    ) -> Self {
        Self {
            max_cost,
            algorithm,
            costs,
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
/// This is identical to the integer version since character matching
/// is independent of cost model.
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

    for i in 0..window_size {
        let matched = if let Some(query_unit) = checked_query_index(offset, i)
            .and_then(|query_idx| query.get(query_idx))
            .copied()
        {
            query_unit == dict_unit || policy.is_allowed_for(dict_unit, query_unit)
        } else {
            false
        };
        output.push(matched);
    }

    output.as_slice()
}

/// Find the index of the first true value in `cv[start..start+limit]`.
#[inline]
fn index_of_match(cv: &[bool], start: usize, limit: usize) -> Option<usize> {
    let end = checked_window_end(start, limit, cv.len())?;
    cv.get(start..end)?.iter().position(|&matched| matched)
}

/// Transition a position given a characteristic vector (float-weighted).
///
/// Computes all possible next positions from the current position
/// after consuming a dictionary character, considering the query
/// term through the characteristic vector.
///
/// # Cost Model
///
/// Unlike the integer version which uses hardcoded `+1` costs, this version
/// uses configurable costs from `OperationCostsF64`:
/// - Match: `costs.match_cost` (typically 0.0)
/// - Substitution: `costs.substitution`
/// - Insertion: `costs.insertion`
/// - Deletion: `costs.deletion`
/// - Transposition: `costs.transposition`
/// - Split: `costs.split`
/// - Merge: `costs.merge`
#[inline]
pub fn transition_position_f64(
    position: &PositionF64,
    characteristic_vector: &[bool],
    query_length: usize,
    max_cost: f64,
    algorithm: Algorithm,
    costs: &OperationCostsF64,
    prefix_mode: bool,
) -> SmallVec<[PositionF64; 4]> {
    match algorithm {
        Algorithm::Standard => transition_standard_f64(
            position,
            characteristic_vector,
            query_length,
            max_cost,
            costs,
            prefix_mode,
        ),
        Algorithm::Transposition => transition_transposition_f64(
            position,
            characteristic_vector,
            query_length,
            max_cost,
            costs,
            prefix_mode,
        ),
        Algorithm::MergeAndSplit => transition_merge_split_f64(
            position,
            characteristic_vector,
            query_length,
            max_cost,
            costs,
            prefix_mode,
        ),
    }
}

/// Check if cost exceeds threshold (with epsilon tolerance).
#[inline]
fn exceeds_threshold(cost: f64, max_cost: f64) -> bool {
    cost > max_cost + COST_EPSILON
}

/// Check if cost is at or near threshold.
#[inline]
fn at_threshold(cost: f64, max_cost: f64) -> bool {
    (cost - max_cost).abs() < COST_EPSILON
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
fn query_window_size(query_length: usize) -> usize {
    checked_successor_or_max(query_length).max(1)
}

#[inline(always)]
fn transition_window_size_f64(max_cost: f64, min_cost: f64, query_length: usize) -> usize {
    let query_window = query_window_size(query_length);
    if min_cost > 0.0 {
        let cost_window = checked_successor_or_max(nonnegative_ceil_to_usize(max_cost / min_cost));
        cost_window.min(query_window)
    } else {
        query_window
    }
}

#[inline(always)]
fn checked_position_with_offset(
    term_index: usize,
    term_offset: usize,
    accumulated_cost: f64,
) -> Option<PositionF64> {
    let term_index = term_index.checked_add(term_offset)?;
    Some(PositionF64::new(term_index, accumulated_cost))
}

#[inline(always)]
fn checked_special_position_with_offset(
    term_index: usize,
    term_offset: usize,
    accumulated_cost: f64,
) -> Option<PositionF64> {
    let term_index = term_index.checked_add(term_offset)?;
    Some(PositionF64::new_special(term_index, accumulated_cost))
}

#[inline(always)]
fn checked_deletion_position(
    term_index: usize,
    accumulated_cost: f64,
    match_offset: usize,
) -> Option<PositionF64> {
    let term_offset = match_offset.checked_add(1)?;
    checked_position_with_offset(term_index, term_offset, accumulated_cost)
}

#[inline(always)]
fn push_checked_position(next: &mut SmallVec<[PositionF64; 4]>, position: Option<PositionF64>) {
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

/// Compute the window size based on remaining cost budget.
///
/// For float costs, we need to determine how many positions ahead
/// we can potentially reach with the remaining cost budget.
#[inline]
fn compute_window_limit(remaining_cost: f64, deletion_cost: f64) -> usize {
    if deletion_cost <= 0.0 {
        // Zero-cost deletions could reach infinitely far (shouldn't happen in practice)
        8 // Use max window size
    } else {
        // How many deletions can we afford?
        let max_deletions = nonnegative_floor_to_usize_saturating(remaining_cost / deletion_cost);
        checked_successor_or_max(max_deletions).min(8) // +1 for the match position, capped at 8
    }
}

/// Standard algorithm transition with float costs (insert, delete, substitute).
#[inline]
fn transition_standard_f64(
    position: &PositionF64,
    cv: &[bool],
    query_length: usize,
    max_cost: f64,
    costs: &OperationCostsF64,
    prefix_mode: bool,
) -> SmallVec<[PositionF64; 4]> {
    let mut next = SmallVec::new();
    let i = position.term_index;
    let e = position.accumulated_cost;
    let h = 0; // cv is offset-adjusted
    let w = cv.len();

    // Prefix matching: if enabled and we've consumed the full query
    if prefix_mode && i >= query_length {
        next.push(PositionF64::new(i, e));
        return next;
    }

    let remaining_cost = max_cost - e;

    // Case 1: Have cost budget remaining
    if remaining_cost > COST_EPSILON {
        // Subcase 1a: At least 2 characters remain in query
        if h + 2 <= w {
            let k = compute_window_limit(remaining_cost, costs.deletion);
            let k = k.min(w - h);

            match index_of_match(cv, h, k) {
                Some(0) => {
                    // Immediate match at cv[h]
                    push_checked_position(
                        &mut next,
                        checked_position_with_offset(i, 1, e + costs.match_cost),
                    );
                }
                Some(j) => {
                    // Match found at cv[h + j]
                    let ins_cost = e + costs.insertion;
                    let sub_cost = e + costs.substitution;
                    let del_cost = e + (j as f64) * costs.deletion;

                    if !exceeds_threshold(ins_cost, max_cost) {
                        next.push(PositionF64::new(i, ins_cost)); // insertion
                    }
                    if !exceeds_threshold(sub_cost, max_cost) {
                        push_checked_position(
                            &mut next,
                            checked_position_with_offset(i, 1, sub_cost),
                        );
                    }
                    if !exceeds_threshold(del_cost, max_cost) {
                        push_checked_position(&mut next, checked_deletion_position(i, del_cost, j));
                    }
                }
                None => {
                    // No match found in range
                    let ins_cost = e + costs.insertion;
                    let sub_cost = e + costs.substitution;

                    if !exceeds_threshold(ins_cost, max_cost) {
                        next.push(PositionF64::new(i, ins_cost)); // insertion
                    }
                    if !exceeds_threshold(sub_cost, max_cost) {
                        push_checked_position(
                            &mut next,
                            checked_position_with_offset(i, 1, sub_cost),
                        );
                    }
                }
            }
        }
        // Subcase 1b: Exactly 1 character remains
        else if h + 1 == w {
            if cv[h] {
                push_checked_position(
                    &mut next,
                    checked_position_with_offset(i, 1, e + costs.match_cost),
                );
            } else {
                let ins_cost = e + costs.insertion;
                let sub_cost = e + costs.substitution;

                if !exceeds_threshold(ins_cost, max_cost) {
                    next.push(PositionF64::new(i, ins_cost));
                }
                if !exceeds_threshold(sub_cost, max_cost) {
                    push_checked_position(&mut next, checked_position_with_offset(i, 1, sub_cost));
                }
            }
        }
        // Subcase 1c: Past the end of query
        else {
            let ins_cost = e + costs.insertion;
            if !exceeds_threshold(ins_cost, max_cost) {
                next.push(PositionF64::new(i, ins_cost));
            }
        }
    }
    // Case 2: At max cost - only exact matches allowed
    else if at_threshold(e, max_cost) && h < w && cv[h] {
        push_checked_position(&mut next, checked_position_with_offset(i, 1, max_cost));
    }

    next
}

/// Transposition algorithm transition with float costs.
#[inline]
fn transition_transposition_f64(
    position: &PositionF64,
    cv: &[bool],
    query_length: usize,
    max_cost: f64,
    costs: &OperationCostsF64,
    prefix_mode: bool,
) -> SmallVec<[PositionF64; 4]> {
    let mut next = SmallVec::new();
    let i = position.term_index;
    let e = position.accumulated_cost;
    let t = position.is_special; // Transposition flag
    let h = 0;
    let w = cv.len();

    if prefix_mode && i >= query_length {
        next.push(PositionF64::new(i, e));
        return next;
    }

    let remaining_cost = max_cost - e;

    // Case 1: No errors yet (e == 0)
    if e.abs() < COST_EPSILON && remaining_cost > COST_EPSILON {
        if h + 2 <= w {
            let k = compute_window_limit(remaining_cost, costs.deletion);
            let k = k.min(w - h);

            match index_of_match(cv, h, k) {
                Some(0) => {
                    // Immediate match
                    push_checked_position(&mut next, checked_position_with_offset(i, 1, 0.0));
                }
                Some(1) => {
                    // Match at next position - potential transposition
                    let ins_cost = costs.insertion;
                    let trans_cost = costs.transposition;
                    let sub_cost = costs.substitution;
                    let del_cost = costs.deletion;

                    if !exceeds_threshold(ins_cost, max_cost) {
                        next.push(PositionF64::new(i, ins_cost));
                    }
                    if !exceeds_threshold(trans_cost, max_cost) {
                        next.push(PositionF64::new_special(i, trans_cost));
                    }
                    if !exceeds_threshold(sub_cost, max_cost) {
                        push_checked_position(
                            &mut next,
                            checked_position_with_offset(i, 1, sub_cost),
                        );
                    }
                    if !exceeds_threshold(del_cost, max_cost) {
                        push_checked_position(
                            &mut next,
                            checked_position_with_offset(i, 2, del_cost),
                        );
                    }
                }
                Some(j) => {
                    let ins_cost = costs.insertion;
                    let sub_cost = costs.substitution;
                    let del_cost = (j as f64) * costs.deletion;

                    if !exceeds_threshold(ins_cost, max_cost) {
                        next.push(PositionF64::new(i, ins_cost));
                    }
                    if !exceeds_threshold(sub_cost, max_cost) {
                        push_checked_position(
                            &mut next,
                            checked_position_with_offset(i, 1, sub_cost),
                        );
                    }
                    if !exceeds_threshold(del_cost, max_cost) {
                        push_checked_position(&mut next, checked_deletion_position(i, del_cost, j));
                    }
                }
                None => {
                    let ins_cost = costs.insertion;
                    let sub_cost = costs.substitution;

                    if !exceeds_threshold(ins_cost, max_cost) {
                        next.push(PositionF64::new(i, ins_cost));
                    }
                    if !exceeds_threshold(sub_cost, max_cost) {
                        push_checked_position(
                            &mut next,
                            checked_position_with_offset(i, 1, sub_cost),
                        );
                    }
                }
            }
        } else if h + 1 == w {
            if cv[h] {
                push_checked_position(&mut next, checked_position_with_offset(i, 1, 0.0));
            } else {
                let ins_cost = costs.insertion;
                let sub_cost = costs.substitution;

                if !exceeds_threshold(ins_cost, max_cost) {
                    next.push(PositionF64::new(i, ins_cost));
                }
                if !exceeds_threshold(sub_cost, max_cost) {
                    push_checked_position(&mut next, checked_position_with_offset(i, 1, sub_cost));
                }
            }
        } else {
            let ins_cost = costs.insertion;
            if !exceeds_threshold(ins_cost, max_cost) {
                next.push(PositionF64::new(i, ins_cost));
            }
        }
    }
    // Case 2: Have some errors but not at threshold
    else if remaining_cost > COST_EPSILON {
        if h + 2 <= w {
            if !t {
                // Not in transposition state
                let k = compute_window_limit(remaining_cost, costs.deletion);
                let k = k.min(w - h);

                match index_of_match(cv, h, k) {
                    Some(0) => {
                        push_checked_position(&mut next, checked_position_with_offset(i, 1, e));
                    }
                    Some(1) => {
                        let ins_cost = e + costs.insertion;
                        let trans_cost = e + costs.transposition;
                        let sub_cost = e + costs.substitution;
                        let del_cost = e + costs.deletion;

                        if !exceeds_threshold(ins_cost, max_cost) {
                            next.push(PositionF64::new(i, ins_cost));
                        }
                        if !exceeds_threshold(trans_cost, max_cost) {
                            push_checked_position(
                                &mut next,
                                checked_special_position_with_offset(i, 0, trans_cost),
                            );
                        }
                        if !exceeds_threshold(sub_cost, max_cost) {
                            push_checked_position(
                                &mut next,
                                checked_position_with_offset(i, 1, sub_cost),
                            );
                        }
                        if !exceeds_threshold(del_cost, max_cost) {
                            push_checked_position(
                                &mut next,
                                checked_position_with_offset(i, 2, del_cost),
                            );
                        }
                    }
                    Some(j) => {
                        let ins_cost = e + costs.insertion;
                        let sub_cost = e + costs.substitution;
                        let del_cost = e + (j as f64) * costs.deletion;

                        if !exceeds_threshold(ins_cost, max_cost) {
                            next.push(PositionF64::new(i, ins_cost));
                        }
                        if !exceeds_threshold(sub_cost, max_cost) {
                            push_checked_position(
                                &mut next,
                                checked_position_with_offset(i, 1, sub_cost),
                            );
                        }
                        if !exceeds_threshold(del_cost, max_cost) {
                            push_checked_position(
                                &mut next,
                                checked_deletion_position(i, del_cost, j),
                            );
                        }
                    }
                    None => {
                        let ins_cost = e + costs.insertion;
                        let sub_cost = e + costs.substitution;

                        if !exceeds_threshold(ins_cost, max_cost) {
                            next.push(PositionF64::new(i, ins_cost));
                        }
                        if !exceeds_threshold(sub_cost, max_cost) {
                            push_checked_position(
                                &mut next,
                                checked_position_with_offset(i, 1, sub_cost),
                            );
                        }
                    }
                }
            } else {
                // In transposition state - complete it
                if cv[h] {
                    push_checked_position(&mut next, checked_position_with_offset(i, 2, e));
                }
            }
        } else if h + 1 == w {
            if cv[h] {
                push_checked_position(&mut next, checked_position_with_offset(i, 1, e));
            } else {
                let ins_cost = e + costs.insertion;
                let sub_cost = e + costs.substitution;

                if !exceeds_threshold(ins_cost, max_cost) {
                    next.push(PositionF64::new(i, ins_cost));
                }
                if !exceeds_threshold(sub_cost, max_cost) {
                    push_checked_position(&mut next, checked_position_with_offset(i, 1, sub_cost));
                }
            }
        } else {
            let ins_cost = e + costs.insertion;
            if !exceeds_threshold(ins_cost, max_cost) {
                next.push(PositionF64::new(i, ins_cost));
            }
        }
    }
    // Case 3: At max cost
    else if at_threshold(e, max_cost) {
        if h < w && !t {
            if cv[h] {
                push_checked_position(&mut next, checked_position_with_offset(i, 1, max_cost));
            }
        } else if h + 2 <= w && t && cv[h] {
            push_checked_position(&mut next, checked_position_with_offset(i, 2, max_cost));
        }
    }

    next
}

/// Merge and split algorithm transition with float costs.
#[inline]
fn transition_merge_split_f64(
    position: &PositionF64,
    cv: &[bool],
    query_length: usize,
    max_cost: f64,
    costs: &OperationCostsF64,
    prefix_mode: bool,
) -> SmallVec<[PositionF64; 4]> {
    let mut next = SmallVec::new();
    let i = position.term_index;
    let e = position.accumulated_cost;
    let s = position.is_special; // Special flag for merge/split
    let h = 0;
    let w = cv.len();

    if prefix_mode && i >= query_length {
        next.push(PositionF64::new(i, e));
        return next;
    }

    let remaining_cost = max_cost - e;

    // Case 1: No errors yet (e == 0)
    if e.abs() < COST_EPSILON && remaining_cost > COST_EPSILON {
        if h + 2 <= w {
            if cv[h] {
                // Immediate match
                push_checked_position(&mut next, checked_position_with_offset(i, 1, 0.0));
            } else {
                // No match - add error operations
                let ins_cost = costs.insertion;
                let split_cost = costs.split;
                let sub_cost = costs.substitution;
                let merge_cost = costs.merge;

                if !exceeds_threshold(ins_cost, max_cost) {
                    next.push(PositionF64::new(i, ins_cost));
                }
                // Split: one query char becomes two dict chars
                if i < query_length && !exceeds_threshold(split_cost, max_cost) {
                    next.push(PositionF64::new_special(i, split_cost));
                }
                if !exceeds_threshold(sub_cost, max_cost) {
                    push_checked_position(&mut next, checked_position_with_offset(i, 1, sub_cost));
                }
                // Merge: two query chars become one dict char
                if has_query_units(i, 2, query_length) && !exceeds_threshold(merge_cost, max_cost) {
                    push_checked_position(
                        &mut next,
                        checked_position_with_offset(i, 2, merge_cost),
                    );
                }
            }
        } else if h + 1 == w {
            if cv[h] {
                push_checked_position(&mut next, checked_position_with_offset(i, 1, 0.0));
            } else {
                let ins_cost = costs.insertion;
                let split_cost = costs.split;
                let sub_cost = costs.substitution;

                if !exceeds_threshold(ins_cost, max_cost) {
                    next.push(PositionF64::new(i, ins_cost));
                }
                if i < query_length && !exceeds_threshold(split_cost, max_cost) {
                    next.push(PositionF64::new_special(i, split_cost));
                }
                if !exceeds_threshold(sub_cost, max_cost) {
                    push_checked_position(&mut next, checked_position_with_offset(i, 1, sub_cost));
                }
            }
        } else {
            let ins_cost = costs.insertion;
            if !exceeds_threshold(ins_cost, max_cost) {
                next.push(PositionF64::new(i, ins_cost));
            }
        }
    }
    // Case 2: Have some errors but not at threshold
    else if remaining_cost > COST_EPSILON {
        if h + 2 <= w {
            if !s {
                // Not in special state
                if cv[h] {
                    push_checked_position(&mut next, checked_position_with_offset(i, 1, e));
                } else {
                    let ins_cost = e + costs.insertion;
                    let split_cost = e + costs.split;
                    let sub_cost = e + costs.substitution;
                    let merge_cost = e + costs.merge;

                    if !exceeds_threshold(ins_cost, max_cost) {
                        next.push(PositionF64::new(i, ins_cost));
                    }
                    if i < query_length && !exceeds_threshold(split_cost, max_cost) {
                        next.push(PositionF64::new_special(i, split_cost));
                    }
                    if !exceeds_threshold(sub_cost, max_cost) {
                        push_checked_position(
                            &mut next,
                            checked_position_with_offset(i, 1, sub_cost),
                        );
                    }
                    if has_query_units(i, 2, query_length)
                        && !exceeds_threshold(merge_cost, max_cost)
                    {
                        push_checked_position(
                            &mut next,
                            checked_position_with_offset(i, 2, merge_cost),
                        );
                    }
                }
            } else {
                // In special state (completing split)
                push_checked_position(&mut next, checked_position_with_offset(i, 1, e));
            }
        } else if h + 1 == w {
            if !s {
                if cv[h] {
                    push_checked_position(&mut next, checked_position_with_offset(i, 1, e));
                } else {
                    let ins_cost = e + costs.insertion;
                    let split_cost = e + costs.split;
                    let sub_cost = e + costs.substitution;

                    if !exceeds_threshold(ins_cost, max_cost) {
                        next.push(PositionF64::new(i, ins_cost));
                    }
                    if i < query_length && !exceeds_threshold(split_cost, max_cost) {
                        next.push(PositionF64::new_special(i, split_cost));
                    }
                    if !exceeds_threshold(sub_cost, max_cost) {
                        push_checked_position(
                            &mut next,
                            checked_position_with_offset(i, 1, sub_cost),
                        );
                    }
                }
            } else {
                push_checked_position(&mut next, checked_position_with_offset(i, 1, e));
            }
        } else {
            let ins_cost = e + costs.insertion;
            if !exceeds_threshold(ins_cost, max_cost) {
                next.push(PositionF64::new(i, ins_cost));
            }
        }
    }
    // Case 3: At max cost
    else if at_threshold(e, max_cost) && h < w {
        if !s {
            if cv[h] {
                push_checked_position(&mut next, checked_position_with_offset(i, 1, max_cost));
            }
        } else {
            // Special state: can advance even at max cost
            push_checked_position(&mut next, checked_position_with_offset(i, 1, e));
        }
    }

    next
}

/// Compute epsilon closure: add positions reachable by deletion.
///
/// For float costs, each deletion adds `costs.deletion` to the accumulated cost.
#[inline]
fn epsilon_closure_mut_f64(
    state: &mut StateF64,
    query_length: usize,
    max_cost: f64,
    algorithm: Algorithm,
    costs: &OperationCostsF64,
) {
    let mut to_process: SmallVec<[PositionF64; 8]> = SmallVec::with_capacity(8);

    for pos in state.positions() {
        to_process.push(*pos);
    }

    let mut processed = 0;
    while processed < to_process.len() {
        let position = &to_process[processed];
        processed += 1;

        let new_cost = position.accumulated_cost + costs.deletion;
        if !exceeds_threshold(new_cost, max_cost) && position.term_index < query_length {
            let Some(deleted) = checked_position_with_offset(position.term_index, 1, new_cost)
            else {
                continue;
            };

            let len_before = state.len();
            state.insert(
                deleted,
                algorithm,
                query_length,
                costs.insertion.max(costs.deletion),
            );
            if state.len() > len_before {
                to_process.push(deleted);
            }
        }
    }
}

/// Compute epsilon closure from source into target state (pool-friendly).
#[inline]
fn epsilon_closure_into_f64(
    source: &StateF64,
    target: &mut StateF64,
    query_length: usize,
    max_cost: f64,
    algorithm: Algorithm,
    costs: &OperationCostsF64,
) {
    target.copy_from(source);
    epsilon_closure_mut_f64(target, query_length, max_cost, algorithm, costs);
}

/// Transition an entire state given a dictionary character unit (float-weighted).
///
/// This is the main entry point for transitioning states with float costs.
pub fn transition_state_f64<U: CharUnit, P: SubstitutionPolicy + SubstitutionPolicyFor<U>>(
    state: &StateF64,
    policy: P,
    dict_unit: U,
    query: &[U],
    settings: TransitionSettingsF64<'_>,
) -> Option<StateF64> {
    let min_cost = settings.costs.min_nonzero_cost();
    let window_size = transition_window_size_f64(settings.max_cost, min_cost, query.len());
    let query_length = query.len();

    // First, expand state with epsilon closure (deletions)
    let mut expanded_state = state.clone();
    epsilon_closure_mut_f64(
        &mut expanded_state,
        query_length,
        settings.max_cost,
        settings.algorithm,
        settings.costs,
    );

    let mut next_state = StateF64::new();
    let mut cv = SmallVec::<[bool; 8]>::new();

    for position in expanded_state.positions() {
        let offset = position.term_index;
        let cv = characteristic_vector(&policy, dict_unit, query, window_size, offset, &mut cv);

        let next_positions = transition_position_f64(
            position,
            cv,
            query_length,
            settings.max_cost,
            settings.algorithm,
            settings.costs,
            settings.prefix_mode,
        );

        for next_pos in next_positions {
            next_state.insert(
                next_pos,
                settings.algorithm,
                query_length,
                settings.costs.insertion.max(settings.costs.deletion),
            );
        }
    }

    if next_state.is_empty() {
        None
    } else {
        Some(next_state)
    }
}

/// Transition a state using a StatePoolF64 for allocation reuse (optimized).
///
/// This is the pool-aware version that eliminates State cloning overhead
/// by reusing allocations from the pool.
#[inline]
pub fn transition_state_pooled_f64<
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
>(
    state: &StateF64,
    pool: &mut StatePoolF64,
    policy: P,
    dict_unit: U,
    query: &[U],
    settings: TransitionSettingsF64<'_>,
) -> Option<StateF64> {
    transition_state_pooled_f64_ref(state, pool, &policy, dict_unit, query, settings)
}

/// Transition a state using a borrowed policy and a StatePoolF64.
///
/// This keeps the public by-value API stable while allowing internal query
/// iterators to avoid cloning non-ZST policies for every explored edge.
#[inline]
pub(crate) fn transition_state_pooled_f64_ref<
    U: CharUnit,
    P: SubstitutionPolicy + SubstitutionPolicyFor<U>,
>(
    state: &StateF64,
    pool: &mut StatePoolF64,
    policy: &P,
    dict_unit: U,
    query: &[U],
    settings: TransitionSettingsF64<'_>,
) -> Option<StateF64> {
    let min_cost = settings.costs.min_nonzero_cost();
    let window_size = transition_window_size_f64(settings.max_cost, min_cost, query.len());
    let query_length = query.len();

    // Acquire state from pool for epsilon closure
    let mut expanded_state = pool.acquire();
    epsilon_closure_into_f64(
        state,
        &mut expanded_state,
        query_length,
        settings.max_cost,
        settings.algorithm,
        settings.costs,
    );

    // Acquire another state for next state
    let mut next_state = pool.acquire();
    let mut cv = SmallVec::<[bool; 8]>::new();

    for position in expanded_state.positions() {
        let offset = position.term_index;
        let cv = characteristic_vector(policy, dict_unit, query, window_size, offset, &mut cv);

        let next_positions = transition_position_f64(
            position,
            cv,
            query_length,
            settings.max_cost,
            settings.algorithm,
            settings.costs,
            settings.prefix_mode,
        );

        for next_pos in next_positions {
            next_state.insert(
                next_pos,
                settings.algorithm,
                query_length,
                settings.costs.insertion.max(settings.costs.deletion),
            );
        }
    }

    // Return expanded state to pool
    pool.release(expanded_state);

    if next_state.is_empty() {
        pool.release(next_state);
        None
    } else {
        Some(next_state)
    }
}

/// Create the initial state for a float-weighted query.
///
/// Similar to integer version but uses float costs for initial deletions.
pub fn initial_state_f64(
    query_length: usize,
    max_cost: f64,
    algorithm: Algorithm,
    costs: &OperationCostsF64,
) -> StateF64 {
    let mut state = StateF64::new();

    // Start at position (0, 0.0)
    let max_index_op_cost = costs.insertion.max(costs.deletion);
    state.insert(
        PositionF64::new(0, 0.0),
        algorithm,
        query_length,
        max_index_op_cost,
    );

    // Add positions for initial deletions
    let mut cost = costs.deletion;
    let mut i = 1;
    while cost <= max_cost + COST_EPSILON && i <= query_length {
        state.insert(
            PositionF64::new(i, cost),
            algorithm,
            query_length,
            max_index_op_cost,
        );
        let Some(next_i) = i.checked_add(1) else {
            break;
        };
        i = next_i;
        cost += costs.deletion;
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transducer::Unrestricted;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_characteristic_vector() {
        let query = b"test";
        let policy = Unrestricted;
        let mut buffer = SmallVec::<[bool; 8]>::new();

        let cv = characteristic_vector(&policy, b't', query, 3, 0, &mut buffer);
        assert_eq!(cv, &[true, false, false]);

        let cv = characteristic_vector(&policy, b'e', query, 3, 0, &mut buffer);
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
    fn checked_float_transition_arithmetic_helpers_handle_boundaries() {
        assert_eq!(checked_query_index(usize::MAX, 1), None);
        assert_eq!(checked_window_end(1, usize::MAX, 4), Some(4));
        assert_eq!(checked_window_end(5, 1, 4), None);
        assert_eq!(checked_successor_or_max(usize::MAX), usize::MAX);
        assert_eq!(checked_successor_or_max(1), 2);

        let exact_large = 4_503_599_627_370_496.0;
        let expected_large = usize::try_from(4_503_599_627_370_496_u128).unwrap_or(usize::MAX);

        assert_eq!(nonnegative_floor_to_usize_saturating(f64::NAN), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(-1.0), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(-0.0), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(0.0), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(0.1), 0);
        assert_eq!(nonnegative_floor_to_usize_saturating(2.0), 2);
        assert_eq!(nonnegative_floor_to_usize_saturating(2.9), 2);
        assert_eq!(nonnegative_floor_to_usize_saturating(exact_large), expected_large);
        assert_eq!(nonnegative_floor_to_usize_saturating(f64::INFINITY), usize::MAX);
        assert_eq!(nonnegative_floor_to_usize_saturating(f64::MAX), usize::MAX);
        assert_eq!(nonnegative_ceil_to_usize(f64::NAN), 0);
        assert_eq!(nonnegative_ceil_to_usize(-1.0), 0);
        assert_eq!(nonnegative_ceil_to_usize(-0.0), 0);
        assert_eq!(nonnegative_ceil_to_usize(0.0), 0);
        assert_eq!(nonnegative_ceil_to_usize(0.1), 1);
        assert_eq!(nonnegative_ceil_to_usize(2.0), 2);
        assert_eq!(nonnegative_ceil_to_usize(2.1), 3);
        assert_eq!(nonnegative_ceil_to_usize(exact_large), expected_large);
        assert_eq!(nonnegative_ceil_to_usize(f64::INFINITY), usize::MAX);
        assert_eq!(nonnegative_ceil_to_usize(f64::MAX), usize::MAX);

        assert_eq!(
            transition_window_size_f64(f64::INFINITY, f64::MIN_POSITIVE, usize::MAX),
            usize::MAX
        );
        assert_eq!(transition_window_size_f64(2.0, 1.0, usize::MAX), 3);
        assert_eq!(checked_position_with_offset(usize::MAX, 1, 0.0), None);
        assert_eq!(checked_deletion_position(0, 1.0, usize::MAX), None);
        assert!(!has_query_units(usize::MAX - 1, 2, usize::MAX));
    }

    #[test]
    fn characteristic_vector_overflowing_offset_is_unmatched() {
        let query = b"x";
        let policy = Unrestricted;
        let mut buffer = SmallVec::<[bool; 8]>::new();

        let cv = characteristic_vector(&policy, b'x', query, 2, usize::MAX, &mut buffer);

        assert_eq!(cv, &[false, false]);
    }

    #[test]
    fn index_of_match_caps_overflowing_limit() {
        assert_eq!(index_of_match(&[false, true, true], 1, usize::MAX), Some(0));
        assert_eq!(index_of_match(&[false, false], usize::MAX, 1), None);
    }

    #[test]
    fn test_transition_standard_match() {
        let pos = PositionF64::new(0, 0.0);
        let cv = vec![true, false, false];
        let costs = OperationCostsF64::standard();
        let next = transition_standard_f64(&pos, &cv, 4, 2.0, &costs, false);

        // Should advance with no error on match
        assert!(next
            .iter()
            .any(|p| p.term_index == 1 && p.accumulated_cost.abs() < EPSILON));
    }

    #[test]
    fn test_transition_standard_operations() {
        let pos = PositionF64::new(1, 0.0);
        let cv = vec![false, false, true];
        let costs = OperationCostsF64::standard();
        let next = transition_standard_f64(&pos, &cv, 4, 2.0, &costs, false);

        // Should include insertion at (1, 1.0)
        assert!(next
            .iter()
            .any(|p| p.term_index == 1 && (p.accumulated_cost - 1.0).abs() < EPSILON));
    }

    #[test]
    fn test_transition_with_custom_costs() {
        let pos = PositionF64::new(0, 0.0);
        let cv = vec![false, false, false]; // No matches
        let costs = OperationCostsF64::custom(1.5, 0.5, 0.8, 0.3, 2.0, 2.0);
        let next = transition_standard_f64(&pos, &cv, 4, 3.0, &costs, false);

        // Should have insertion at (0, 0.5) and substitution at (1, 1.5)
        let has_insertion = next
            .iter()
            .any(|p| p.term_index == 0 && (p.accumulated_cost - 0.5).abs() < EPSILON);
        let has_substitution = next
            .iter()
            .any(|p| p.term_index == 1 && (p.accumulated_cost - 1.5).abs() < EPSILON);
        assert!(has_insertion, "Should have insertion at cost 0.5");
        assert!(has_substitution, "Should have substitution at cost 1.5");
    }

    #[test]
    fn transition_standard_f64_drops_overflowing_successor() {
        let pos = PositionF64::new(usize::MAX, 0.0);
        let costs = OperationCostsF64::standard();

        let next = transition_standard_f64(&pos, &[true], usize::MAX, 1.0, &costs, false);

        assert!(next.is_empty());
    }

    #[test]
    fn transition_merge_split_f64_rejects_overflowing_two_unit_check() {
        let pos = PositionF64::new(usize::MAX - 1, 0.0);
        let costs = OperationCostsF64::standard();

        let next =
            transition_merge_split_f64(&pos, &[false, false], usize::MAX, 2.0, &costs, false);

        assert!(next.iter().any(|p| p.term_index == usize::MAX - 1));
        assert!(next.iter().any(|p| p.term_index == usize::MAX));
        assert!(!next.iter().any(|p| p.term_index < usize::MAX - 1));
    }

    #[test]
    fn test_compute_window_limit_saturates_extreme_float_budget() {
        assert_eq!(compute_window_limit(f64::MAX, f64::MIN_POSITIVE), 8);
        assert_eq!(compute_window_limit(f64::INFINITY, f64::MIN_POSITIVE), 8);
    }

    #[test]
    fn test_initial_state() {
        let costs = OperationCostsF64::standard();
        let state = initial_state_f64(5, 2.0, Algorithm::Standard, &costs);

        // With Standard subsumption, (0,0.0) subsumes positions reachable via deletions
        // similar to integer version
        assert!(!state.is_empty());
        assert!(state
            .positions()
            .iter()
            .any(|p| p.term_index == 0 && p.accumulated_cost.abs() < EPSILON));
    }

    #[test]
    fn test_transition_state() {
        let query = b"test";
        let max_cost = 2.0;
        let costs = OperationCostsF64::standard();
        let state = initial_state_f64(query.len(), max_cost, Algorithm::Standard, &costs);
        let policy = Unrestricted;

        let next = transition_state_f64(
            &state,
            policy,
            b't',
            query,
            TransitionSettingsF64::new(max_cost, Algorithm::Standard, &costs, false),
        );
        assert!(next.is_some());

        let next_state = next.expect("test fixture: transition produces Some (asserted above)");
        // Should have advanced after matching 't'
        assert!(next_state.positions().iter().any(|p| p.term_index > 0));
    }

    #[test]
    fn test_typo_friendly_transposition() {
        let costs = OperationCostsF64::typo_friendly();
        assert!((costs.transposition - 0.5).abs() < EPSILON);
        assert!((costs.substitution - 1.2).abs() < EPSILON);
    }

    #[test]
    fn test_merge_split_transitions() {
        let pos = PositionF64::new(0, 0.0);
        let cv = vec![false, false, false]; // No matches
        let costs = OperationCostsF64::standard();
        let query_length = 4;
        let next = transition_merge_split_f64(&pos, &cv, query_length, 2.0, &costs, false);

        // Should have insertion, split (special), substitution, and merge
        let has_insertion = next.iter().any(|p| p.term_index == 0 && !p.is_special);
        let has_split = next.iter().any(|p| p.term_index == 0 && p.is_special);
        let has_substitution = next.iter().any(|p| p.term_index == 1 && !p.is_special);
        let has_merge = next.iter().any(|p| p.term_index == 2 && !p.is_special);

        assert!(has_insertion, "Should have insertion");
        assert!(has_split, "Should have split (special position)");
        assert!(has_substitution, "Should have substitution");
        assert!(has_merge, "Should have merge");
    }

    #[test]
    fn test_pooled_transition() {
        let query = b"test";
        let max_cost = 2.0;
        let costs = OperationCostsF64::standard();
        let state = initial_state_f64(query.len(), max_cost, Algorithm::Standard, &costs);
        let policy = Unrestricted;
        let mut pool = StatePoolF64::new();

        let next = transition_state_pooled_f64(
            &state,
            &mut pool,
            policy,
            b't',
            query,
            TransitionSettingsF64::new(max_cost, Algorithm::Standard, &costs, false),
        );
        assert!(next.is_some());

        // Check pool stats
        assert!(pool.total_reuses() > 0);

        let mut pool = StatePoolF64::new();
        let borrowed = transition_state_pooled_f64_ref(
            &state,
            &mut pool,
            &policy,
            b't',
            query,
            TransitionSettingsF64::new(max_cost, Algorithm::Standard, &costs, false),
        );

        assert_eq!(
            borrowed.expect("borrowed pooled transition should produce a state"),
            next.expect("public pooled transition should produce a state")
        );
    }
}
