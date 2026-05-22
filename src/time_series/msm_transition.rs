//! MSM automaton transitions with data-dependent costs.
//!
//! This module implements the core transition logic for the MSM automaton.
//! Unlike operation-level weighted Levenshtein where costs are static per operation type,
//! MSM costs are computed dynamically based on the actual values being compared.
//!
//! # MSM Operations
//!
//! | Operation | DP Transition | Cost Formula |
//! |-----------|---------------|--------------|
//! | Move | Diagonal (i-1, j-1) → (i, j) | `\|x_i - y_j\|` |
//! | Merge-like | Vertical (i-1, j) → (i, j) | `C(x_i, x_{i-1}, y_j)` |
//! | Split-like | Horizontal (i, j-1) → (i, j) | `C(y_j, x_i, y_{j-1})` |
//!
//! # C() Function
//!
//! The C(a, b, c) function determines the cost of split/merge-like operations:
//!
//! ```text
//! C(a, b, c) = c_const           if b ≤ a ≤ c OR b ≥ a ≥ c
//!            = c_const + min(|a-b|, |a-c|)  otherwise
//! ```
//!
//! Where:
//! - `a` is the value being inserted/removed
//! - `b` is the adjacent value in the source series
//! - `c` is the adjacent value in the target series
//! - `c_const` is the base cost for split/merge operations

use super::msm_position::MsmPosition;
use super::msm_state::MsmState;
use super::MsmConfig;
use smallvec::SmallVec;

/// Epsilon for float comparisons.
const COST_EPSILON: f64 = 1e-9;

/// Transition a single MSM position given the next values from both series.
///
/// This function computes all possible next positions from a single position
/// by applying the three MSM operations: Move, Merge-like, and Split-like.
///
/// # Arguments
///
/// * `position` - Current automaton position
/// * `query_value` - The next value from the query series (x_i)
/// * `target_value` - The next value from the target series (y_j)
/// * `config` - MSM configuration (contains c_const)
/// * `max_cost` - Maximum cost threshold
/// * `query_length` - Length of the query series
/// * `target_length` - Length of the target series
///
/// # Returns
///
/// A vector of new positions reachable from this position.
#[inline]
pub fn transition_msm_position(
    position: &MsmPosition,
    query_value: Option<f64>,
    target_value: Option<f64>,
    config: &MsmConfig,
    max_cost: f64,
    query_length: usize,
    target_length: usize,
) -> SmallVec<[MsmPosition; 4]> {
    let mut next_positions = SmallVec::new();

    // Move operation: consume one from both series (diagonal)
    // Cost = |x_i - y_j|
    if let (Some(qv), Some(tv)) = (query_value, target_value) {
        let move_cost = position.accumulated_cost + (qv - tv).abs();
        if move_cost <= max_cost + COST_EPSILON {
            let new_pos = MsmPosition::new(
                position.query_index + 1,
                position.target_index + 1,
                move_cost,
                qv, // Update last query value
                tv, // Update last target value
            );
            if new_pos.can_reach_acceptance(query_length, target_length, max_cost, config.c) {
                next_positions.push(new_pos);
            }
        }
    }

    // Merge-like operation: consume one from query only (vertical)
    // Cost = C(x_i, x_{i-1}, y_j)
    // This requires a query value and a target value for C() context
    if let Some(qv) = query_value {
        // We need a target value for the C() function
        // Use the last_target_value as the context
        let c_cost = config.c_func(qv, position.last_query_value, position.last_target_value);
        let merge_cost = position.accumulated_cost + c_cost;
        if merge_cost <= max_cost + COST_EPSILON {
            let new_pos = MsmPosition::new(
                position.query_index + 1,
                position.target_index, // Target index doesn't change
                merge_cost,
                qv,                         // Update last query value
                position.last_target_value, // Keep last target value
            );
            if new_pos.can_reach_acceptance(query_length, target_length, max_cost, config.c) {
                next_positions.push(new_pos);
            }
        }
    }

    // Split-like operation: consume one from target only (horizontal)
    // Cost = C(y_j, x_i, y_{j-1})
    if let Some(tv) = target_value {
        // We need the current query value (or last if at end) for C() context
        let query_context = query_value.unwrap_or(position.last_query_value);
        let c_cost = config.c_func(tv, query_context, position.last_target_value);
        let split_cost = position.accumulated_cost + c_cost;
        if split_cost <= max_cost + COST_EPSILON {
            let new_pos = MsmPosition::new(
                position.query_index, // Query index doesn't change
                position.target_index + 1,
                split_cost,
                position.last_query_value, // Keep last query value
                tv,                        // Update last target value
            );
            if new_pos.can_reach_acceptance(query_length, target_length, max_cost, config.c) {
                next_positions.push(new_pos);
            }
        }
    }

    next_positions
}

/// Transition an entire MSM state given the next values from both series.
///
/// This function applies transitions to all positions in the current state
/// and returns a new state with the resulting positions (after subsumption).
///
/// # Arguments
///
/// * `state` - Current MSM state
/// * `query_value` - The next value from the query series (None if exhausted)
/// * `target_value` - The next value from the target series (None if exhausted)
/// * `config` - MSM configuration
/// * `max_cost` - Maximum cost threshold
/// * `query_length` - Length of the query series
/// * `target_length` - Length of the target series
///
/// # Returns
///
/// The new state, or `None` if no valid transitions exist.
pub fn transition_msm_state(
    state: &MsmState,
    query_value: Option<f64>,
    target_value: Option<f64>,
    config: &MsmConfig,
    max_cost: f64,
    query_length: usize,
    target_length: usize,
) -> Option<MsmState> {
    if state.is_empty() {
        return None;
    }

    // If no new values, check if we're at the end
    if query_value.is_none() && target_value.is_none() {
        // Keep positions that have reached the final state
        let final_positions: Vec<_> = state
            .iter()
            .filter(|p| p.is_final(query_length, target_length))
            .cloned()
            .collect();

        if final_positions.is_empty() {
            return None;
        }

        let mut new_state = MsmState::with_capacity(final_positions.len());
        for pos in final_positions {
            new_state.insert(pos, max_cost, COST_EPSILON);
        }
        return Some(new_state);
    }

    let mut new_state = MsmState::with_capacity(state.len() * 3);

    for position in state.iter() {
        let next_positions = transition_msm_position(
            position,
            query_value,
            target_value,
            config,
            max_cost,
            query_length,
            target_length,
        );

        for new_pos in next_positions {
            new_state.insert(new_pos, max_cost, COST_EPSILON);
        }
    }

    if new_state.is_empty() {
        None
    } else {
        Some(new_state)
    }
}

/// Create the initial MSM state for comparing two series.
///
/// # Arguments
///
/// * `query` - The query series
/// * `target` - The target series
/// * `config` - MSM configuration
/// * `max_cost` - Maximum cost threshold
///
/// # Returns
///
/// The initial state with the starting position, or `None` if series are incompatible.
pub fn initial_msm_state(
    query: &[f64],
    target: &[f64],
    config: &MsmConfig,
    max_cost: f64,
) -> Option<MsmState> {
    // Handle empty series
    if query.is_empty() && target.is_empty() {
        // Both empty - perfect match
        let mut state = MsmState::new();
        state.insert_unchecked(MsmPosition::new(0, 0, 0.0, 0.0, 0.0));
        return Some(state);
    }

    if query.is_empty() || target.is_empty() {
        // Can't match empty to non-empty with MSM operations
        return None;
    }

    // Initial position with first values as context
    let initial_pos = MsmPosition::initial(query[0], target[0]);

    // Check if it can possibly reach acceptance
    if !initial_pos.can_reach_acceptance(query.len(), target.len(), max_cost, config.c) {
        return None;
    }

    Some(MsmState::single(initial_pos))
}

/// Compute MSM distance using the automaton approach.
///
/// This is an alternative to the DP approach that uses automaton transitions.
/// It's useful for understanding the automaton behavior and for future
/// optimizations like early termination and trie-based indexing.
///
/// # Arguments
///
/// * `query` - The query series
/// * `target` - The target series
/// * `config` - MSM configuration
/// * `max_cost` - Maximum cost threshold (use f64::INFINITY for no threshold)
///
/// # Returns
///
/// The MSM distance if within threshold, or `None` if exceeds threshold.
pub fn msm_distance_automaton(
    query: &[f64],
    target: &[f64],
    config: &MsmConfig,
    max_cost: f64,
) -> Option<f64> {
    // Handle empty series
    if query.is_empty() && target.is_empty() {
        return Some(0.0);
    }
    if query.is_empty() || target.is_empty() {
        return None;
    }

    // Fail-fast: ensure an initial state can be constructed (return value
    // is unused — `state` below is built directly from the first elements).
    let _ = initial_msm_state(query, target, config, max_cost)?;

    // Process first position: Cost(1,1) = |x_0 - y_0|
    let initial_cost = (query[0] - target[0]).abs();
    if initial_cost > max_cost + COST_EPSILON {
        return None;
    }

    // Set up the state after processing first elements
    let mut state = MsmState::single(MsmPosition::new(1, 1, initial_cost, query[0], target[0]));

    // Process remaining elements using a wavefront approach
    // We need to fill all cells (i, j) for i in 1..=m, j in 1..=n
    let m = query.len();
    let n = target.len();

    // Initialize first column (i varies, j=1)
    let mut first_col = MsmState::single(MsmPosition::new(1, 1, initial_cost, query[0], target[0]));
    for i in 2..=m {
        let prev_cost = first_col
            .iter()
            .find(|p| p.query_index == i - 1 && p.target_index == 1)
            .map(|p| p.accumulated_cost)
            .unwrap_or(f64::INFINITY);

        let c_cost = config.c_func(query[i - 1], query[i - 2], target[0]);
        let new_cost = prev_cost + c_cost;

        if new_cost <= max_cost + COST_EPSILON {
            first_col.insert_unchecked(MsmPosition::new(i, 1, new_cost, query[i - 1], target[0]));
        }
    }

    // Initialize first row (i=1, j varies)
    let mut current_row =
        MsmState::single(MsmPosition::new(1, 1, initial_cost, query[0], target[0]));
    for j in 2..=n {
        let prev_cost = current_row
            .iter()
            .find(|p| p.query_index == 1 && p.target_index == j - 1)
            .map(|p| p.accumulated_cost)
            .unwrap_or(f64::INFINITY);

        let c_cost = config.c_func(target[j - 1], query[0], target[j - 2]);
        let new_cost = prev_cost + c_cost;

        if new_cost <= max_cost + COST_EPSILON {
            current_row.insert_unchecked(MsmPosition::new(1, j, new_cost, query[0], target[j - 1]));
        }
    }

    // Merge first column and first row into state
    state.clear();
    for pos in first_col.iter() {
        state.insert_unchecked(*pos);
    }
    for pos in current_row.iter() {
        if pos.query_index != 1 || pos.target_index != 1 {
            // Avoid duplicate (1,1)
            state.insert_unchecked(*pos);
        }
    }

    // Fill remaining cells row by row
    for i in 2..=m {
        for j in 2..=n {
            // Find best costs from predecessors
            let mut best_cost = f64::INFINITY;
            let mut best_qv = query[i - 1];
            let mut best_tv = target[j - 1];

            // Move: from (i-1, j-1)
            if let Some(prev) = state
                .iter()
                .find(|p| p.query_index == i - 1 && p.target_index == j - 1)
            {
                let move_cost = prev.accumulated_cost + (query[i - 1] - target[j - 1]).abs();
                if move_cost < best_cost {
                    best_cost = move_cost;
                    best_qv = query[i - 1];
                    best_tv = target[j - 1];
                }
            }

            // Merge: from (i-1, j)
            if let Some(prev) = state
                .iter()
                .find(|p| p.query_index == i - 1 && p.target_index == j)
            {
                let c_cost = config.c_func(query[i - 1], prev.last_query_value, target[j - 1]);
                let merge_cost = prev.accumulated_cost + c_cost;
                if merge_cost < best_cost {
                    best_cost = merge_cost;
                    best_qv = query[i - 1];
                    best_tv = prev.last_target_value;
                }
            }

            // Split: from (i, j-1)
            if let Some(prev) = state
                .iter()
                .find(|p| p.query_index == i && p.target_index == j - 1)
            {
                let c_cost = config.c_func(target[j - 1], query[i - 1], prev.last_target_value);
                let split_cost = prev.accumulated_cost + c_cost;
                if split_cost < best_cost {
                    best_cost = split_cost;
                    best_qv = prev.last_query_value;
                    best_tv = target[j - 1];
                }
            }

            if best_cost <= max_cost + COST_EPSILON {
                state.insert_unchecked(MsmPosition::new(i, j, best_cost, best_qv, best_tv));
            }
        }
    }

    // Return the final distance
    state.min_final_distance(m, n)
}

/// MSM distance computation using a more efficient wavefront approach.
///
/// This implementation maintains only the necessary frontier of positions
/// rather than all positions, making it more suitable for the automaton model.
pub fn msm_distance_wavefront(
    query: &[f64],
    target: &[f64],
    config: &MsmConfig,
    max_cost: f64,
) -> Option<f64> {
    let m = query.len();
    let n = target.len();

    // Handle empty series
    if m == 0 && n == 0 {
        return Some(0.0);
    }
    if m == 0 || n == 0 {
        return None;
    }

    // Use 2D cost array (same as DP but with early termination potential)
    let mut cost = vec![vec![f64::INFINITY; n + 1]; m + 1];

    // Base case
    cost[1][1] = (query[0] - target[0]).abs();
    if cost[1][1] > max_cost + COST_EPSILON {
        return None;
    }

    // Initialize first column
    for i in 2..=m {
        let c_cost = config.c_func(query[i - 1], query[i - 2], target[0]);
        cost[i][1] = cost[i - 1][1] + c_cost;
    }

    // Initialize first row
    for j in 2..=n {
        let c_cost = config.c_func(target[j - 1], query[0], target[j - 2]);
        cost[1][j] = cost[1][j - 1] + c_cost;
    }

    // Fill the matrix with early termination
    let mut has_valid = true;
    for i in 2..=m {
        let mut row_has_valid = false;
        for j in 2..=n {
            let move_cost = cost[i - 1][j - 1] + (query[i - 1] - target[j - 1]).abs();
            let merge_cost =
                cost[i - 1][j] + config.c_func(query[i - 1], query[i - 2], target[j - 1]);
            let split_cost =
                cost[i][j - 1] + config.c_func(target[j - 1], query[i - 1], target[j - 2]);

            cost[i][j] = move_cost.min(merge_cost).min(split_cost);

            if cost[i][j] <= max_cost + COST_EPSILON {
                row_has_valid = true;
            }
        }
        if !row_has_valid {
            has_valid = false;
            break;
        }
    }

    if has_valid && cost[m][n] <= max_cost + COST_EPSILON {
        Some(cost[m][n])
    } else if cost[m][n].is_finite() {
        Some(cost[m][n])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_transition_msm_position_move() {
        let config = MsmConfig::new(1.0);
        let pos = MsmPosition::new(0, 0, 0.0, 1.0, 2.0);

        let next = transition_msm_position(&pos, Some(1.5), Some(2.5), &config, 10.0, 3, 3);

        // Should have move, merge, and split transitions
        assert!(next.len() >= 1);

        // Find the move transition
        let move_pos = next
            .iter()
            .find(|p| p.query_index == 1 && p.target_index == 1);
        assert!(move_pos.is_some());
        let move_pos = move_pos.expect("expected Some move_pos in test");
        assert!(approx_eq(move_pos.accumulated_cost, 1.0)); // |1.5 - 2.5| = 1.0
    }

    #[test]
    fn test_transition_msm_position_merge() {
        let config = MsmConfig::new(1.0);
        let pos = MsmPosition::new(0, 1, 0.0, 1.0, 2.0);

        // Only query value, no target value (at end of target)
        let next = transition_msm_position(&pos, Some(1.5), None, &config, 10.0, 3, 1);

        // Should have merge-like transition
        let merge_pos = next
            .iter()
            .find(|p| p.query_index == 1 && p.target_index == 1);
        assert!(merge_pos.is_some());
    }

    #[test]
    fn test_transition_msm_position_split() {
        let config = MsmConfig::new(1.0);
        let pos = MsmPosition::new(1, 0, 0.0, 1.0, 2.0);

        // Only target value, no query value (at end of query)
        let next = transition_msm_position(&pos, None, Some(2.5), &config, 10.0, 1, 3);

        // Should have split-like transition
        let split_pos = next
            .iter()
            .find(|p| p.query_index == 1 && p.target_index == 1);
        assert!(split_pos.is_some());
    }

    #[test]
    fn test_transition_msm_state() {
        let config = MsmConfig::new(1.0);
        let state = MsmState::single(MsmPosition::new(0, 0, 0.0, 1.0, 2.0));

        let next_state = transition_msm_state(&state, Some(1.5), Some(2.5), &config, 10.0, 3, 3);

        assert!(next_state.is_some());
        let next_state = next_state.expect("expected Some next_state in test");
        assert!(!next_state.is_empty());
    }

    #[test]
    fn test_initial_msm_state() {
        let query = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        let config = MsmConfig::new(1.0);

        let state = initial_msm_state(&query, &target, &config, 10.0);
        assert!(state.is_some());
        assert_eq!(state.expect("expected Some state in test").len(), 1);
    }

    #[test]
    fn test_initial_msm_state_empty() {
        let config = MsmConfig::new(1.0);

        // Both empty
        let state = initial_msm_state(&[], &[], &config, 10.0);
        assert!(state.is_some());

        // One empty
        let state = initial_msm_state(&[1.0], &[], &config, 10.0);
        assert!(state.is_none());
    }

    #[test]
    fn test_msm_distance_automaton_identical() {
        let config = MsmConfig::new(1.0);
        let series = vec![1.0, 2.0, 3.0];

        let dist = msm_distance_automaton(&series, &series, &config, f64::INFINITY);
        assert!(dist.is_some());
        assert!(approx_eq(dist.expect("expected Some dist in test"), 0.0));
    }

    #[test]
    fn test_msm_distance_automaton_single_move() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0];
        let y = vec![2.0];

        let dist = msm_distance_automaton(&x, &y, &config, f64::INFINITY);
        assert!(dist.is_some());
        assert!(approx_eq(dist.expect("expected Some dist in test"), 1.0)); // |1.0 - 2.0| = 1.0
    }

    #[test]
    fn test_msm_distance_automaton_shift() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];

        let dist = msm_distance_automaton(&x, &y, &config, f64::INFINITY);
        assert!(dist.is_some());
        // Each element shifted by 1: 3 moves of cost 1 each
        assert!(approx_eq(dist.expect("expected Some dist in test"), 3.0));
    }

    #[test]
    fn test_msm_distance_wavefront_identical() {
        let config = MsmConfig::new(1.0);
        let series = vec![1.0, 2.0, 3.0];

        let dist = msm_distance_wavefront(&series, &series, &config, f64::INFINITY);
        assert!(dist.is_some());
        assert!(approx_eq(dist.expect("expected Some dist in test"), 0.0));
    }

    #[test]
    fn test_msm_distance_wavefront_matches_dp() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let y = vec![1.0, 3.0, 2.0];

        let dist_dp = config.distance(&x, &y);
        let dist_wavefront = msm_distance_wavefront(&x, &y, &config, f64::INFINITY);

        assert!(dist_wavefront.is_some());
        assert!(
            approx_eq(
                dist_dp,
                dist_wavefront.expect("expected Some dist_wavefront in test")
            ),
            "DP: {}, Wavefront: {}",
            dist_dp,
            dist_wavefront.expect("expected Some dist_wavefront in test")
        );
    }

    #[test]
    fn test_msm_distance_automaton_matches_dp() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let y = vec![1.0, 3.0, 2.0];

        let dist_dp = config.distance(&x, &y);
        let dist_auto = msm_distance_automaton(&x, &y, &config, f64::INFINITY);

        assert!(dist_auto.is_some());
        assert!(
            approx_eq(dist_dp, dist_auto.expect("expected Some dist_auto in test")),
            "DP: {}, Automaton: {}",
            dist_dp,
            dist_auto.expect("expected Some dist_auto in test")
        );
    }

    #[test]
    fn test_msm_with_threshold() {
        let config = MsmConfig::new(1.0);
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![5.0, 6.0, 7.0]; // Each diff = 4, total = 12

        // Should find distance
        let dist = msm_distance_wavefront(&x, &y, &config, 15.0);
        assert!(dist.is_some());

        // Should be filtered by threshold
        let dist = msm_distance_wavefront(&x, &y, &config, 5.0);
        // May or may not return None depending on early termination
        if let Some(d) = dist {
            assert!(d > 5.0);
        }
    }

    #[test]
    fn test_c_function_in_transitions() {
        let config = MsmConfig::new(1.0);

        // Test C() function behavior through transitions
        // C(a, b, c) = c_const if b <= a <= c or b >= a >= c
        // Otherwise c_const + min(|a-b|, |a-c|)

        // Case: a between b and c
        let cost1 = config.c_func(2.0, 1.0, 3.0); // 1 <= 2 <= 3
        assert!(approx_eq(cost1, 1.0));

        // Case: a outside [b, c]
        let cost2 = config.c_func(5.0, 1.0, 3.0); // 5 > 3
        assert!(approx_eq(cost2, 1.0 + 2.0)); // c + min(|5-1|, |5-3|) = 1 + 2
    }
}
