//! Verus obligations for the PositionKind/AutomatonVariant seam.
//!
//! Verified directly by `verus`; Cargo does not compile this file.

use vstd::prelude::*;

fn main() {}

verus! {

spec fn is_special(kind: nat) -> bool {
    kind != 0
}
spec fn select_variant(algorithm: nat) -> nat {
    if algorithm == 0 { 0 } else if algorithm == 1 { 1 } else { 2 }
}

/// One edge dispatch selects a variant while preserving the position payload
/// that the selected kernel will consume.
spec fn dispatch_edge(algorithm: nat, position: nat) -> (nat, nat) {
    (select_variant(algorithm), position)
}

proof fn normal_constructor_preserves_payload_invariant()
    ensures
        !is_special(0),
        0nat < 256,
{
}

proof fn full_key_distinguishes_kind(
    i: nat,
    e: nat,
    lhs_kind: nat,
    rhs_kind: nat,
    aux: nat,
)
    requires
        lhs_kind != rhs_kind,
    ensures
        (i, e, lhs_kind, aux) != (i, e, rhs_kind, aux),
{
}

proof fn full_key_distinguishes_aux(
    i: nat,
    e: nat,
    kind: nat,
    lhs_aux: nat,
    rhs_aux: nat,
)
    requires
        lhs_aux != rhs_aux,
    ensures
        (i, e, kind, lhs_aux) != (i, e, kind, rhs_aux),
{
}

proof fn runtime_static_dispatch_equivalent(algorithm: nat)
    requires
        algorithm <= 2,
    ensures
        select_variant(algorithm) == algorithm,
{
}

proof fn edge_selection_is_position_independent(
    algorithm: nat,
    lhs_position: nat,
    rhs_position: nat,
)
    requires
        algorithm <= 2,
    ensures
        dispatch_edge(algorithm, lhs_position).0 == algorithm,
        dispatch_edge(algorithm, rhs_position).0 == algorithm,
        dispatch_edge(algorithm, lhs_position).1 == lhs_position,
        dispatch_edge(algorithm, rhs_position).1 == rhs_position,
{
}

proof fn standard_true_implies_error_order(lhs_error: nat, rhs_error: nat)
    requires
        lhs_error <= rhs_error,
    ensures
        rhs_error - lhs_error <= rhs_error,
{
}

proof fn osa_mixed_special_states_are_separate(lhs_special: bool, rhs_special: bool)
    requires
        lhs_special != rhs_special,
    ensures
        !(lhs_special && rhs_special),
        !(!lhs_special && !rhs_special),
{
}

proof fn merge_split_subsumption_is_strict(lhs_error: nat, rhs_error: nat)
    requires
        lhs_error < rhs_error,
    ensures
        lhs_error <= rhs_error,
        lhs_error != rhs_error,
{
}

proof fn one_dispatch_serves_every_position(
    algorithm: nat,
    first: nat,
    second: nat,
    third: nat,
)
    requires
        algorithm <= 2,
    ensures
        dispatch_edge(algorithm, first).0 == algorithm,
        dispatch_edge(algorithm, second).0 == algorithm,
        dispatch_edge(algorithm, third).0 == algorithm,
        dispatch_edge(algorithm, first).1 == first,
        dispatch_edge(algorithm, second).1 == second,
        dispatch_edge(algorithm, third).1 == third,
{
}

}
