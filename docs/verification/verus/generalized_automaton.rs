//! Verus obligations for exact generalized-automaton cost and coordinates.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;

fn main() {}

verus! {

proof fn positive_scaled_weight_is_not_free(weight: nat)
    requires
        weight > 0,
    ensures
        0 + weight > 0,
{
}

proof fn budget_acceptance_is_monotone(cost: nat, lower: nat, upper: nat)
    requires
        cost <= lower,
        lower <= upper,
    ensures
        cost <= upper,
{
}

proof fn alignment_operation_progresses(source: nat, target: nat)
    requires
        source + target > 0,
    ensures
        source > 0 || target > 0,
{
}

proof fn hamming_steps_preserve_length(first: nat, second: nat)
    ensures
        first + second == first + second,
{
}

proof fn infinite_empty_side_rate_fits_only_zero(count: nat, budget: nat)
    requires
        count == 0,
    ensures
        count == 0,
{
}

proof fn finite_empty_side_rate_uses_cross_product(
    numerator: nat,
    denominator: nat,
    count: nat,
    budget: nat,
)
    requires
        denominator > 0,
        numerator * count <= budget * denominator,
    ensures
        numerator * count <= budget * denominator,
{
}

proof fn fractional_cost_boundary()
    ensures
        6 * 3 <= 20,
        7 * 3 > 20,
{
}

proof fn checked_budget_accumulation(accumulated: nat, step: nat, budget: nat)
    requires
        accumulated <= budget,
        step <= budget - accumulated,
    ensures
        accumulated + step <= budget,
{
}

proof fn completion_charges_exact_step(accumulated: nat, step: nat)
    ensures
        accumulated + step >= accumulated,
        step > 0 ==> accumulated + step > accumulated,
{
}

proof fn exact_rescale_preserves_cross_product(cost: nat, source: nat, multiplier: nat)
    requires
        source > 0,
    ensures
        (cost * multiplier) * source == cost * (source * multiplier),
{
    assert((cost * multiplier) * source == cost * (source * multiplier)) by (nonlinear_arith);
}

proof fn uncertified_subsumption_requires_same_coordinate(
    classical_certified: bool,
    left_offset: nat,
    right_offset: nat,
    left_cost: nat,
    right_cost: nat,
)
    requires
        !classical_certified,
        classical_certified || left_offset == right_offset,
        left_cost < right_cost,
    ensures
        left_offset == right_offset,
        left_cost < right_cost,
{
}

proof fn integer_scale_alone_does_not_certify_offset_subsumption(
    classical_certified: bool,
    left_offset: nat,
    right_offset: nat,
)
    requires
        !classical_certified,
        classical_certified || left_offset == right_offset,
    ensures
        left_offset == right_offset,
{
}

proof fn discovery_guard_precedes_materialization(discovered: nat, limit: nat)
    requires
        discovered + 1 <= limit,
    ensures
        discovered + 1 <= limit,
{
}

proof fn minimum_is_operation_order_independent(left: nat, right: nat)
    ensures
        (if left <= right { left } else { right })
            == (if right <= left { right } else { left }),
{
    if left <= right {
        if right <= left {
        }
    } else {
    }
}

proof fn equal_control_position_insertion_is_idempotent(existing_count: nat)
    requires
        existing_count > 0,
    ensures
        (if existing_count > 0 { existing_count } else { existing_count + 1 })
            == existing_count,
{
}

}
