//! Verus obligations for the streaming unrestricted-Damerau variant.
//!
//! Verified directly by `verus`; Cargo does not compile this file.

use vstd::prelude::*;

fn main() {}

verus! {

spec fn entry_error(error: nat, delta: nat) -> nat {
    error + delta
}

spec fn macro_cost(delta: nat, between: nat) -> int {
    delta as int + between as int
}

spec fn lowrance_wagner_term(delta: nat, between: nat) -> int {
    (delta as int - 1) + between as int + 1
}

spec fn pending_subsumes(
    lhs_index: nat,
    lhs_error: nat,
    lhs_delta: nat,
    rhs_index: nat,
    rhs_error: nat,
    rhs_delta: nat,
) -> bool {
    lhs_error <= rhs_error
        && lhs_index == rhs_index
        && lhs_delta == rhs_delta
}

spec fn extend_pending(error: nat, delta: nat, steps: nat) -> (nat, nat) {
    (error + steps, delta)
}

proof fn entry_is_budget_bounded(error: nat, delta: nat, budget: nat)
    requires
        1 <= delta,
        delta < 256,
        error + delta <= budget,
    ensures
        entry_error(error, delta) <= budget,
        1 <= delta < 256,
{
}

proof fn macro_charge_is_lowrance_wagner(delta: nat, between: nat)
    requires
        1 <= delta,
    ensures
        macro_cost(delta, between) == lowrance_wagner_term(delta, between),
{
}

proof fn extension_preserves_delta_and_charges_every_step(
    error: nat,
    delta: nat,
    steps: nat,
    budget: nat,
)
    requires
        steps > 0,
        error + steps <= budget,
        1 <= delta < 256,
    ensures
        extend_pending(error, delta, steps).0 <= budget,
        extend_pending(error, delta, steps).0 > error,
        extend_pending(error, delta, steps).1 == delta,
{
}

proof fn pending_subsumption_exposes_all_required_equalities(
    lhs_index: nat,
    lhs_error: nat,
    lhs_delta: nat,
    rhs_index: nat,
    rhs_error: nat,
    rhs_delta: nat,
)
    requires
        pending_subsumes(
            lhs_index,
            lhs_error,
            lhs_delta,
            rhs_index,
            rhs_error,
            rhs_delta,
        ),
    ensures
        lhs_error <= rhs_error,
        lhs_index == rhs_index,
        lhs_delta == rhs_delta,
{
}

proof fn unequal_pending_delta_is_incomparable(
    index: nat,
    lhs_error: nat,
    lhs_delta: nat,
    rhs_error: nat,
    rhs_delta: nat,
)
    requires
        lhs_delta != rhs_delta,
    ensures
        !pending_subsumes(
            index,
            lhs_error,
            lhs_delta,
            index,
            rhs_error,
            rhs_delta,
        ),
{
}

proof fn resolved_index_is_strictly_forward(index: nat, delta: nat)
    requires
        1 <= delta,
    ensures
        index < index + delta + 1,
{
}

}
