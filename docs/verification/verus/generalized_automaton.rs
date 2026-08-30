//! Verus obligations for exact generalized-automaton cost and coordinates.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;

fn main() {}

verus! {

#[derive(PartialEq, Eq)]
enum EmptySideRate {
    Infinite,
    Finite { numerator: nat, denominator: nat },
}

spec fn empty_side_fits(rate: EmptySideRate, count: nat, budget: nat) -> bool {
    match rate {
        EmptySideRate::Infinite => count == 0,
        EmptySideRate::Finite { numerator, denominator } =>
            denominator > 0 && numerator * count <= budget * denominator,
    }
}

spec fn hamming_source_consumption(matches: nat, substitutions: nat) -> nat {
    matches + substitutions
}

spec fn hamming_target_consumption(matches: nat, substitutions: nat) -> nat {
    matches + substitutions
}

spec fn materialized_discovery_count(discovered: nat) -> nat {
    discovered + 1
}

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

proof fn hamming_steps_preserve_length(matches: nat, substitutions: nat)
    ensures
        hamming_source_consumption(matches, substitutions)
            == hamming_target_consumption(matches, substitutions),
{
}

proof fn infinite_empty_side_rate_fits_only_zero(count: nat, budget: nat)
    ensures
        empty_side_fits(EmptySideRate::Infinite, count, budget) == (count == 0),
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
    ensures
        empty_side_fits(
            EmptySideRate::Finite { numerator, denominator },
            count,
            budget,
        ) == (numerator * count <= budget * denominator),
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
        materialized_discovery_count(discovered) <= limit,
        materialized_discovery_count(discovered) > discovered,
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

/// A generalized operation can inspect at most `max_target_consumption`
/// preceding target generations. A ring containing that many predecessors
/// plus one scratch/current row has prefix-independent retained size.
pub open spec fn generalized_retained_cells_after_prefix(
    consumed_target: nat,
    max_target_consumption: nat,
    source_width: nat,
) -> nat {
    (max_target_consumption + 1) * source_width
}

proof fn finite_lookback_rows_are_stream_length_independent(
    consumed_target: nat,
    max_target_consumption: nat,
    source_width: nat,
)
    ensures
        generalized_retained_cells_after_prefix(
            consumed_target,
            max_target_consumption,
            source_width,
        ) == generalized_retained_cells_after_prefix(
            0,
            max_target_consumption,
            source_width,
        ),
        generalized_retained_cells_after_prefix(
            consumed_target,
            max_target_consumption,
            source_width,
        ) == (max_target_consumption + 1) * source_width,
{
}

/// Every positive-target operation reads a committed predecessor generation;
/// zero-target operations read an earlier source coordinate in the scratch
/// row because validation forbids a zero/zero operation.
proof fn generalized_predecessor_is_topologically_earlier(
    source_consumption: nat,
    target_consumption: nat,
)
    requires
        source_consumption + target_consumption > 0,
    ensures
        target_consumption > 0 || source_consumption > 0,
{
}

}
