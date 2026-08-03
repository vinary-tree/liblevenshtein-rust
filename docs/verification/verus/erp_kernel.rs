//! Verus model of ERP's Rust-facing arithmetic and pruning obligations.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;

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

proof fn degenerate_interval_is_exact(value: int, point: int)
    ensures
        interval_dist(value, point, point) == iabs(value - point),
{
    reveal(interval_dist);
    reveal(iabs);
}

proof fn reverse_absolute_difference(x: int, y: int, gap: int)
    ensures
        iabs(iabs(x - gap) - iabs(y - gap)) <= iabs(x - y),
{
    reveal(iabs);
}

proof fn deletion_of_gap_has_zero_cost(gap: int)
    ensures
        iabs(gap - gap) == 0,
{
    reveal(iabs);
}

proof fn zero_deletion_cost_identifies_gap(value: int, gap: int)
    requires
        iabs(value - gap) == 0,
    ensures
        value == gap,
{
    reveal(iabs);
}

proof fn candidate_bound_prunes_soundly(
    left_mass: int,
    right_mass: int,
    exact: int,
    cutoff: int,
)
    requires
        iabs(left_mass - right_mass) <= exact,
        cutoff < iabs(left_mass - right_mass),
    ensures
        cutoff < exact,
{
}

proof fn row_minimum_cutoff_is_sound(row_minimum: int, exact: int, cutoff: int)
    requires
        row_minimum <= exact,
        cutoff < row_minimum,
    ensures
        cutoff < exact,
{
}

proof fn nonnegative_step_inflates(prefix: nat, step: nat)
    ensures
        prefix <= prefix + step,
{
}

}
