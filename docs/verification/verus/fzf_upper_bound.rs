//! Verus model of the capacity-sensitive FuzzyMatchV2 recurrence bound.

use vstd::prelude::*;

fn main() {}

verus! {

pub open spec fn maximum(left: int, right: int) -> int {
    if left >= right { left } else { right }
}

pub open spec fn feasible_term(feasible: bool, completed: int, term: int) -> int {
    if feasible { term } else { completed }
}

pub open spec fn bound(
    completed: int,
    unstarted: int,
    active: int,
    query_len: int,
    active_remaining: int,
    capacity: int,
    beta: int,
) -> int {
    maximum(
        completed,
        maximum(
            feasible_term(query_len <= capacity, completed, unstarted),
            feasible_term(
                active_remaining <= capacity,
                completed,
                active + active_remaining * beta,
            ),
        ),
    )
}

proof fn completed_projection_is_bounded(
    completed: int, unstarted: int, active: int,
    query_len: int, active_remaining: int, capacity: int, beta: int,
)
    ensures completed <= bound(
        completed, unstarted, active, query_len, active_remaining, capacity, beta,
    ),
{}

proof fn gap_projection_is_bounded(
    completed: int, unstarted: int, active: int, child_score: int,
    query_len: int, active_remaining: int, capacity: int, beta: int,
)
    requires
        0 <= beta,
        active_remaining <= capacity,
        child_score <= active,
    ensures child_score + active_remaining * beta <= bound(
        completed, unstarted, active, query_len, active_remaining, capacity, beta,
    ),
{
    assert(child_score + active_remaining * beta
        <= active + active_remaining * beta);
}

proof fn match_projection_is_bounded(
    completed: int, unstarted: int, active: int, child_score: int,
    query_len: int, child_remaining: int, capacity: int, beta: int,
)
    requires
        0 <= beta,
        child_remaining + 1 <= capacity,
        child_score <= active + beta,
    ensures child_score + child_remaining * beta <= bound(
        completed, unstarted, active, query_len, child_remaining + 1, capacity, beta,
    ),
{
    assert(child_score + child_remaining * beta
        <= active + beta + child_remaining * beta);
    assert(active + beta + child_remaining * beta
        == active + (child_remaining + 1) * beta) by (nonlinear_arith);
}

proof fn newly_started_projection_is_bounded(
    completed: int, unstarted: int, active: int, child_projection: int,
    query_len: int, active_remaining: int, capacity: int, beta: int,
)
    requires
        query_len <= capacity,
        child_projection <= unstarted,
    ensures child_projection <= bound(
        completed, unstarted, active, query_len, active_remaining, capacity, beta,
    ),
{}

proof fn pruning_derived_projection_is_sound(score: int, upper: int, cutoff: int)
    requires score <= upper, upper < cutoff,
    ensures score < cutoff,
{}

proof fn arctic_transition_deltas_telescope(
    initial: int, middle: int, final_score: int,
)
    ensures initial + (middle - initial) + (final_score - middle) == final_score,
{}

}
