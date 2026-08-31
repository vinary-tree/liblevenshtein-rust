//! Verus model of fixed-channel vector-kernel obligations.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;
use vstd::arithmetic::mul::{
    lemma_mul_inequality, lemma_mul_is_commutative, lemma_mul_is_distributive_add,
    lemma_mul_nonzero,
};

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

/// One coordinate of K1. Positive fixed coefficients can be summed after
/// applying this fact independently to every typed channel.
proof fn weighted_interval_coordinate_is_admissible(
    value: int,
    low: int,
    high: int,
    concrete: int,
    coefficient: int,
)
    requires
        low <= high,
        low <= concrete <= high,
        0 <= coefficient,
    ensures
        0 <= interval_dist(value, low, high),
        coefficient * interval_dist(value, low, high)
            <= coefficient * iabs(value - concrete),
{
    reveal(interval_dist);
    reveal(iabs);
    if value < low {
        assert(interval_dist(value, low, high) == low - value);
        assert(iabs(value - concrete) == concrete - value);
    } else if high < value {
        assert(interval_dist(value, low, high) == value - high);
        assert(iabs(value - concrete) == value - concrete);
    }
    assert(interval_dist(value, low, high) <= iabs(value - concrete));
    lemma_mul_inequality(
        interval_dist(value, low, high),
        iabs(value - concrete),
        coefficient,
    );
}

/// Refining a coordinate interval can only strengthen its K1 lower bound.
proof fn interval_refinement_is_monotone(
    value: int,
    coarse_low: int,
    coarse_high: int,
    fine_low: int,
    fine_high: int,
)
    requires
        coarse_low <= fine_low,
        fine_low <= fine_high,
        fine_high <= coarse_high,
    ensures
        interval_dist(value, coarse_low, coarse_high)
            <= interval_dist(value, fine_low, fine_high),
{
    reveal(interval_dist);
}

proof fn positive_weighted_sum_preserves_identity(
    left_1: int,
    left_2: int,
    weight_1: int,
    weight_2: int,
)
    requires
        0 < weight_1,
        0 < weight_2,
        0 <= left_1,
        0 <= left_2,
        weight_1 * left_1 + weight_2 * left_2 == 0,
    ensures
        left_1 == 0,
        left_2 == 0,
{
    lemma_mul_inequality(0, left_1, weight_1);
    lemma_mul_inequality(0, left_2, weight_2);
    lemma_mul_is_commutative(0, weight_1);
    lemma_mul_is_commutative(left_1, weight_1);
    lemma_mul_is_commutative(0, weight_2);
    lemma_mul_is_commutative(left_2, weight_2);
    assert(0 <= weight_1 * left_1);
    assert(0 <= weight_2 * left_2);
    assert(weight_1 * left_1 == 0);
    assert(weight_2 * left_2 == 0);
    lemma_mul_nonzero(weight_1, left_1);
    lemma_mul_nonzero(weight_2, left_2);
}

/// Coordinate triangle inequalities lift through a fixed positive sum.
proof fn positive_weighted_sum_preserves_triangle(
    xy_1: int,
    yz_1: int,
    xz_1: int,
    xy_2: int,
    yz_2: int,
    xz_2: int,
    weight_1: int,
    weight_2: int,
)
    requires
        0 <= xy_1,
        0 <= yz_1,
        0 <= xz_1,
        0 <= xy_2,
        0 <= yz_2,
        0 <= xz_2,
        0 <= weight_1,
        0 <= weight_2,
        xz_1 <= xy_1 + yz_1,
        xz_2 <= xy_2 + yz_2,
    ensures
        weight_1 * xz_1 + weight_2 * xz_2
            <= (weight_1 * xy_1 + weight_2 * xy_2)
                + (weight_1 * yz_1 + weight_2 * yz_2),
{
    lemma_mul_inequality(xz_1, xy_1 + yz_1, weight_1);
    lemma_mul_inequality(xz_2, xy_2 + yz_2, weight_2);
    lemma_mul_is_commutative(xz_1, weight_1);
    lemma_mul_is_commutative(xy_1 + yz_1, weight_1);
    lemma_mul_is_commutative(xz_2, weight_2);
    lemma_mul_is_commutative(xy_2 + yz_2, weight_2);
    lemma_mul_is_distributive_add(weight_1, xy_1, yz_1);
    lemma_mul_is_distributive_add(weight_2, xy_2, yz_2);
}

proof fn k2_additive_inflation(prefix: nat, local: nat)
    ensures
        prefix <= prefix + local,
{
}

proof fn k2_bottleneck_inflation(prefix: nat, local: nat)
    ensures
        prefix <= if prefix <= local { local } else { prefix },
{
}

proof fn k3_exact_survivor_is_sound(exact: nat, reported: nat, cutoff: nat)
    requires
        reported == exact,
        reported <= cutoff,
    ensures
        exact <= cutoff,
{
}

proof fn k4_candidate_bound_prunes_soundly(bound: nat, exact: nat, cutoff: nat)
    requires
        bound <= exact,
        cutoff < bound,
    ensures
        cutoff < exact,
{
}

/// Missing-channel renormalization is pair-dependent: the same visible
/// channel discrepancy is doubled when one of two unit weights is absent.
proof fn pair_renormalization_changes_distance(discrepancy: nat)
    requires
        0 < discrepancy,
    ensures
        discrepancy != 2 * discrepancy,
{
}

}
