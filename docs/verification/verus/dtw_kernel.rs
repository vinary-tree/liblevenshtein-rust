//! Verus model of banded DTW's Rust-facing interval and pruning obligations.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;
use vstd::arithmetic::mul::lemma_mul_nonnegative;

fn main() {}

verus! {

spec fn iabs(value: int) -> int {
    if 0 <= value { value } else { -value }
}

spec fn imin(left: int, right: int) -> int {
    if left <= right { left } else { right }
}

spec fn min3(a: int, b: int, c: int) -> int {
    imin(a, imin(b, c))
}

spec fn sq(value: int) -> int {
    value * value
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

spec fn dtw_step(north: int, west: int, diagonal: int, local: int) -> int {
    min3(north, west, diagonal) + local
}

proof fn square_is_monotone_on_nonnegative_inputs(a: int, b: int)
    requires
        0 <= a,
        a <= b,
    ensures
        sq(a) <= sq(b),
{
    reveal(sq);
    assert(0 <= b - a);
    assert(0 <= b + a);
    lemma_mul_nonnegative(b - a, b + a);
    assert(b * b - a * a == (b - a) * (b + a)) by (nonlinear_arith);
}

proof fn squared_interval_cost_is_admissible(
    value: int,
    low: int,
    high: int,
    concrete: int,
)
    requires
        low <= high,
        low <= concrete <= high,
    ensures
        0 <= interval_dist(value, low, high),
        sq(interval_dist(value, low, high)) <= sq(value - concrete),
{
    reveal(interval_dist);
    reveal(sq);
    if value < low {
        assert(0 <= low - value);
        assert(low - value <= concrete - value);
        assert(0 <= concrete - value);
        assert(0 <= (concrete - value) - (low - value));
        square_is_monotone_on_nonnegative_inputs(low - value, concrete - value);
        assert((concrete - value) * (concrete - value)
            == (value - concrete) * (value - concrete)) by (nonlinear_arith);
    } else if high < value {
        assert(0 <= value - high);
        assert(value - high <= value - concrete);
        assert(0 <= value - concrete);
        assert(0 <= (value - concrete) - (value - high));
        square_is_monotone_on_nonnegative_inputs(value - high, value - concrete);
    } else {
        assert(0 <= sq(value - concrete)) by (nonlinear_arith);
    }
}

proof fn degenerate_interval_square_is_exact(value: int, point: int)
    ensures
        sq(interval_dist(value, point, point)) == sq(value - point),
{
    reveal(interval_dist);
    reveal(sq);
    if value < point {
        assert((point - value) * (point - value)
            == (value - point) * (value - point)) by (nonlinear_arith);
    } else if point < value {
    } else {
        assert(value == point);
    }
}

proof fn interval_dtw_step_is_admissible(
    n: int, w: int, d: int, local: int,
    n2: int, w2: int, d2: int, local2: int,
)
    requires
        n <= n2,
        w <= w2,
        d <= d2,
        local <= local2,
    ensures
        dtw_step(n, w, d, local) <= dtw_step(n2, w2, d2, local2),
{
    reveal(dtw_step);
    reveal(min3);
    reveal(imin);
}

proof fn additive_accumulation_is_nonnegative(prefix: int, local: int)
    requires
        0 <= prefix,
        0 <= local,
    ensures
        0 <= prefix + local,
        prefix <= prefix + local,
{
}

proof fn prefix_keogh_step_is_admissible(
    prefix_bound: int,
    exact_prefix: int,
    interval_local: int,
    exact_local: int,
)
    requires
        prefix_bound <= exact_prefix,
        interval_local <= exact_local,
    ensures
        prefix_bound + interval_local <= exact_prefix + exact_local,
{
}

proof fn prefix_first_gate_prunes_soundly(bound: int, exact: int, cutoff: int)
    requires
        bound <= exact,
        cutoff < bound,
    ensures
        cutoff < exact,
{
}

proof fn excessive_length_gap_is_outside_band(
    query_len: int,
    target_len: int,
    band: int,
)
    requires
        0 <= query_len,
        0 <= target_len,
        0 <= band,
        band < iabs(query_len - target_len),
    ensures
        !(iabs(query_len - target_len) <= band),
{
    reveal(iabs);
}

proof fn squared_local_cost_is_symmetric(x: int, y: int)
    ensures
        sq(x - y) == sq(y - x),
        0 <= sq(x - y),
{
    reveal(sq);
    assert((x - y) * (x - y) == (y - x) * (y - x)) by (nonlinear_arith);
    assert(0 <= (x - y) * (x - y)) by (nonlinear_arith);
}

proof fn band_one_triangle_counterexample_squared_witness()
    ensures
        sq(0int - 1int) == 1,
        sq(1int - 1int) + sq(1int - 1int) == 0,
        sq(0int - 1int) + sq(0int - 1int) == 2,
        1int + 0int < 2int,
{
    reveal(sq);
}

}
