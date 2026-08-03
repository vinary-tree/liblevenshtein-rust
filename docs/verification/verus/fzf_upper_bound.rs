//! Verus model of the fzf local-alignment branch-and-bound invariants.

use vstd::prelude::*;

fn main() {}

verus! {

pub open spec fn maximum(left: int, right: int) -> int {
    if left >= right { left } else { right }
}

proof fn unstarted_or_active_bound_is_sound(exact: int, unstarted: int, active: int)
    requires exact <= unstarted || exact <= active,
    ensures exact <= maximum(unstarted, active),
{}

proof fn pruning_below_cutoff_cannot_remove_a_top_k_candidate(
    exact: int, unstarted: int, active: int, cutoff: int,
)
    requires
        exact <= unstarted || exact <= active,
        maximum(unstarted, active) < cutoff,
    ensures exact < cutoff,
{}

proof fn arctic_transition_deltas_telescope(
    initial: int, middle: int, final_score: int,
)
    ensures initial + (middle - initial) + (final_score - middle) == final_score,
{}

proof fn active_only_counterexample()
    ensures 10 < 20, 20 <= maximum(20, 10),
{}

}
