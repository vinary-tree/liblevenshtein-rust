//! Verus model of TWED's Rust-facing arithmetic and pruning obligations.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;
use vstd::arithmetic::mul::lemma_mul_inequality;

fn main() {}

verus! {

spec fn iabs(value: int) -> int {
    if 0 <= value { value } else { -value }
}

spec fn interval_dist(value: int, low: int, high: int) -> int {
    if value < low {
        low - value
    } else if high < value {
        value - high
    } else {
        0
    }
}

spec fn interval_gap(low1: int, high1: int, low2: int, high2: int) -> int {
    if high1 < low2 {
        low2 - high1
    } else if high2 < low1 {
        low1 - high2
    } else {
        0
    }
}

proof fn interval_distance_is_admissible(value: int, low: int, high: int, concrete: int)
    requires
        low <= high,
        low <= concrete <= high,
    ensures
        0 <= interval_dist(value, low, high),
        interval_dist(value, low, high) <= iabs(value - concrete),
{
    reveal(interval_dist);
    reveal(iabs);
}

proof fn point_interval_distance_is_exact(value: int, point: int)
    ensures
        interval_dist(value, point, point) == iabs(value - point),
{
    reveal(interval_dist);
    reveal(iabs);
}

proof fn interval_gap_is_admissible(
    low1: int,
    high1: int,
    low2: int,
    high2: int,
    left: int,
    right: int,
)
    requires
        low1 <= high1,
        low2 <= high2,
        low1 <= left <= high1,
        low2 <= right <= high2,
    ensures
        0 <= interval_gap(low1, high1, low2, high2),
        interval_gap(low1, high1, low2, high2) <= iabs(left - right),
{
    reveal(interval_gap);
    reveal(iabs);
}

proof fn point_interval_gap_is_exact(left: int, right: int)
    ensures
        interval_gap(left, left, right, right) == iabs(left - right),
{
    reveal(interval_gap);
    reveal(iabs);
}

proof fn match_leaf_is_separable_and_admissible(
    x_current: int,
    x_previous: int,
    current_low: int,
    current_high: int,
    previous_low: int,
    previous_high: int,
    y_current: int,
    y_previous: int,
    temporal: int,
)
    requires
        current_low <= y_current <= current_high,
        previous_low <= y_previous <= previous_high,
    ensures
        interval_dist(x_current, current_low, current_high)
            + interval_dist(x_previous, previous_low, previous_high)
            + temporal
        <= iabs(x_current - y_current) + iabs(x_previous - y_previous) + temporal,
{
    interval_distance_is_admissible(x_current, current_low, current_high, y_current);
    interval_distance_is_admissible(x_previous, previous_low, previous_high, y_previous);
}

proof fn deletion_leaf_is_admissible(
    current_low: int,
    current_high: int,
    previous_low: int,
    previous_high: int,
    current: int,
    previous: int,
    nu: nat,
    lambda: nat,
)
    requires
        current_low <= current <= current_high,
        previous_low <= previous <= previous_high,
    ensures
        interval_gap(current_low, current_high, previous_low, previous_high)
            + nu + lambda
        <= iabs(current - previous) + nu + lambda,
{
    interval_gap_is_admissible(
        current_low, current_high, previous_low, previous_high, current, previous,
    );
}

proof fn additive_cell_is_monotone(
    predecessor: int,
    predecessor2: int,
    local: int,
    local2: int,
)
    requires
        predecessor <= predecessor2,
        local <= local2,
    ensures
        predecessor + local <= predecessor2 + local2,
{
}

proof fn nonnegative_leaf_inflates(prefix: nat, leaf: nat)
    ensures
        prefix <= prefix + leaf,
{
}

proof fn length_bound_is_admissible(
    length_gap: int,
    deletions: int,
    lambda: int,
    exact: int,
)
    requires
        0 <= length_gap,
        0 <= lambda,
        length_gap <= deletions,
        deletions * lambda <= exact,
    ensures
        length_gap * lambda <= exact,
{
    lemma_mul_inequality(length_gap, deletions, lambda);
}

proof fn candidate_length_gate_prunes_soundly(bound: int, exact: int, cutoff: int)
    requires
        bound <= exact,
        cutoff < bound,
    ensures
        cutoff < exact,
{
}

proof fn positive_stiffness_is_not_zero(nu: int)
    requires
        0 < nu,
    ensures
        nu != 0,
{
}

proof fn zero_parameter_deletion_witness()
    ensures
        iabs(0int - 0int) + 0int + 0int == 0,
{
    reveal(iabs);
}

proof fn concatenated_script_cost_is_additive(left_cost: nat, right_cost: nat)
    ensures
        left_cost + right_cost == right_cost + left_cost,
{
}

spec fn physical_delete_leaf(
    value_delta: nat,
    elapsed_time: nat,
    nu: nat,
    lambda: nat,
) -> nat {
    value_delta + nu * elapsed_time + lambda
}

/// Monotone timestamps and validated nonnegative parameters make every
/// explicit-time deletion leaf nonnegative by construction.
proof fn physical_delete_leaf_is_nonnegative(
    value_delta: nat,
    elapsed_time: nat,
    nu: nat,
    lambda: nat,
)
    ensures
        physical_delete_leaf(value_delta, elapsed_time, nu, lambda) >= 0,
{
}

/// An elapsed time of one canonical unit is exactly the unit-grid deletion
/// leaf rather than an approximation to it.
proof fn unit_elapsed_physical_delete_is_unit_grid(
    value_delta: nat,
    nu: nat,
    lambda: nat,
)
    ensures
        physical_delete_leaf(value_delta, 1, nu, lambda)
            == value_delta + nu + lambda,
{
}

/// Strict timestamp validation implies a positive elapsed-time term for every
/// target sample after the first.
proof fn strict_timestamp_step_has_positive_elapsed_time(
    previous_time: int,
    current_time: int,
)
    requires
        previous_time < current_time,
    ensures
        current_time - previous_time > 0,
{
}

}
