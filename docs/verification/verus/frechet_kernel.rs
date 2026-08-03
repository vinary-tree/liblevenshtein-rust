//! Verus model of discrete Fréchet's Rust-facing kernel obligations.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;

fn main() {}

verus! {

spec fn iabs(value: int) -> int {
    if 0 <= value { value } else { -value }
}

spec fn imin(left: int, right: int) -> int {
    if left <= right { left } else { right }
}

spec fn imax(left: int, right: int) -> int {
    if left <= right { right } else { left }
}

spec fn min3(a: int, b: int, c: int) -> int {
    imin(a, imin(b, c))
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

spec fn frechet_step(north: int, west: int, diagonal: int, link: int) -> int {
    imax(min3(north, west, diagonal), link)
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

proof fn bottleneck_inflates(prefix: int, link: int)
    ensures
        prefix <= imax(prefix, link),
{
    reveal(imax);
}

proof fn frechet_step_is_monotone(
    n: int, w: int, d: int, link: int,
    n2: int, w2: int, d2: int, link2: int,
)
    requires
        n <= n2,
        w <= w2,
        d <= d2,
        link <= link2,
    ensures
        frechet_step(n, w, d, link) <= frechet_step(n2, w2, d2, link2),
{
    reveal(frechet_step);
    reveal(min3);
    reveal(imin);
    reveal(imax);
}

proof fn endpoint_bound_is_admissible(first: int, last: int, exact: int)
    requires
        first <= exact,
        last <= exact,
    ensures
        imax(first, last) <= exact,
{
    reveal(imax);
}

proof fn combined_candidate_bound_is_admissible(
    endpoint: int,
    hausdorff: int,
    exact: int,
)
    requires
        endpoint <= exact,
        hausdorff <= exact,
    ensures
        imax(endpoint, hausdorff) <= exact,
{
    reveal(imax);
}

proof fn candidate_bound_prunes_soundly(bound: int, exact: int, cutoff: int)
    requires
        bound <= exact,
        cutoff < bound,
    ensures
        cutoff < exact,
{
}

proof fn bottleneck_triangle_composition_step(
    prefix_xy: int,
    prefix_yz: int,
    prefix_xz: int,
    link_xy: int,
    link_yz: int,
    link_xz: int,
)
    requires
        0 <= prefix_xy,
        0 <= prefix_yz,
        0 <= link_xy,
        0 <= link_yz,
        prefix_xz <= prefix_xy + prefix_yz,
        link_xz <= link_xy + link_yz,
    ensures
        imax(prefix_xz, link_xz)
            <= imax(prefix_xy, link_xy) + imax(prefix_yz, link_yz),
{
    reveal(imax);
}

proof fn zero_link_identifies_points(x: int, y: int)
    requires
        iabs(x - y) == 0,
    ensures
        x == y,
{
    reveal(iabs);
}

}
