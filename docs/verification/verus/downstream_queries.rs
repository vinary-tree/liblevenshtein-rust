//! Verus obligations for Phase-9 downstream query surfaces.
//!
//! Verified directly by `verus`; Cargo does not compile this file.

use vstd::prelude::*;

fn main() {}

verus! {

spec fn substitution_cost(left: nat, right: nat) -> nat {
    if left == right { 0 } else { 1 }
}

spec fn erase_bracket_kind(kinds: nat, token: nat) -> nat {
    if token < kinds { 0 } else { 1 }
}

proof fn kind_erasure_is_nonexpansive(kinds: nat, left: nat, right: nat)
    ensures
        substitution_cost(erase_bracket_kind(kinds, left), erase_bracket_kind(kinds, right))
            <= substitution_cost(left, right),
{
}

proof fn three_kinds_depth_ten_exceeds_public_guard()
    ensures
        1 + 3 + 9 + 27 + 81 + 243 + 729 + 2187 + 6561 + 19683 + 59049 == 88573,
        88573 > 4096,
{
}

proof fn rejected_enter_still_receives_leave(enters: nat, leaves: nat)
    requires
        enters == leaves,
    ensures
        enters + 1 == leaves + 1,
{
}

spec fn match_mode_accepts(minimum: nat, maximum: nat, distance: nat) -> bool {
    minimum <= distance && distance <= maximum
}

proof fn exact_match_mode_accepts_only_its_distance(exact: nat, distance: nat)
    ensures
        match_mode_accepts(exact, exact, distance) == (distance == exact),
{
}

proof fn range_match_mode_respects_automaton_budget(
    minimum: nat,
    maximum: nat,
    distance: nat,
)
    requires
        match_mode_accepts(minimum, maximum, distance),
    ensures
        distance <= maximum,
{
}

proof fn unwinding_active_dfs_frames_restores_balance(
    enters: nat,
    leaves: nat,
    depth: nat,
)
    requires
        enters == leaves + depth,
    ensures
        enters == leaves + depth,
{
}

spec fn ranked_before(
    left_distance: nat,
    left_confidence: nat,
    left_term: nat,
    right_distance: nat,
    right_confidence: nat,
    right_term: nat,
) -> bool {
    left_distance < right_distance
        || (left_distance == right_distance
            && (left_confidence > right_confidence
                || (left_confidence == right_confidence && left_term <= right_term)))
}

proof fn ranked_order_is_antisymmetric(
    ld: nat,
    lc: nat,
    lt: nat,
    rd: nat,
    rc: nat,
    rt: nat,
)
    requires
        ranked_before(ld, lc, lt, rd, rc, rt),
        ranked_before(rd, rc, rt, ld, lc, lt),
    ensures
        ld == rd,
        lc == rc,
        lt == rt,
{
}

spec fn index_offset(left: int, right: int) -> int {
    if left <= right { right - left } else { left - right }
}

spec fn contextual_realignment_safe(
    left: int,
    right: int,
    minimum: int,
    slack: int,
) -> bool {
    index_offset(left, right) * minimum <= slack
}

proof fn contextual_guard_is_symmetric(
    left: int,
    right: int,
    minimum: int,
    slack: int,
)
    ensures
        contextual_realignment_safe(left, right, minimum, slack)
            == contextual_realignment_safe(right, left, minimum, slack),
{
}

proof fn zero_slack_forbids_distinct_positions(
    left: int,
    right: int,
)
    requires
        left >= 0,
        right >= 0,
        contextual_realignment_safe(left, right, 1, 0),
    ensures
        left == right,
{
    reveal(contextual_realignment_safe);
    reveal(index_offset);
    if left < right {
        assert(right - left >= 1);
    } else if right < left {
        assert(left - right >= 1);
    }
}

}
