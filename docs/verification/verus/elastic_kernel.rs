//! Verus model of the generic elastic-walker pruning obligations.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;
use vstd::math::max;

fn main() {}

verus! {

proof fn k2_additive_inflation(accumulated: nat, step: nat)
    ensures
        accumulated <= accumulated + step,
{
}

proof fn k2_bottleneck_inflation(accumulated: nat, step: nat)
    ensures
        accumulated <= if accumulated <= step { step } else { accumulated },
{
}

proof fn k1_subtree_prune_sound(node_bound: nat, exact: nat, cutoff: nat)
    requires
        node_bound <= exact,
        cutoff < node_bound,
    ensures
        cutoff < exact,
{
}

proof fn k4_candidate_prune_sound(candidate_bound: nat, exact: nat, cutoff: nat)
    requires
        candidate_bound <= exact,
        cutoff < candidate_bound,
    ensures
        cutoff < exact,
{
}

proof fn k3_exact_rescore_no_false_positive(reported: nat, exact: nat, cutoff: nat)
    requires
        reported == exact,
        reported <= cutoff,
    ensures
        exact <= cutoff,
{
}

proof fn best_first_cutoff_sound(
    popped_bound: nat,
    queued_bound: nat,
    exact: nat,
    kth: nat,
)
    requires
        popped_bound <= queued_bound,
        queued_bound <= exact,
        kth < popped_bound,
    ensures
        kth < exact,
{
}

proof fn interval_gap_is_symmetric(
    a_low: int,
    a_high: int,
    b_low: int,
    b_high: int,
)
    requires
        a_low <= a_high,
        b_low <= b_high,
    ensures
        max(0int, max(a_low - b_high, b_low - a_high))
            == max(0int, max(b_low - a_high, a_low - b_high)),
{
}

proof fn degenerate_interval_gap_is_point_distance(a: int, b: int)
    ensures
        max(0int, max(a - b, b - a)) == if a <= b { b - a } else { a - b },
{
}

proof fn prefix_prune_step_preserves_edge_partition(
    edges: nat,
    prefix_pruned: nat,
    columns_built: nat,
)
    requires
        edges == prefix_pruned + columns_built,
    ensures
        edges + 1 == (prefix_pruned + 1) + columns_built,
{
}

proof fn column_step_preserves_edge_partition(
    edges: nat,
    prefix_pruned: nat,
    columns_built: nat,
)
    requires
        edges == prefix_pruned + columns_built,
    ensures
        edges + 1 == prefix_pruned + (columns_built + 1),
{
}

proof fn candidate_bound_step_preserves_partition(
    candidates: nat,
    candidate_bound_pruned: nat,
    exact_evaluations: nat,
)
    requires
        candidates == candidate_bound_pruned + exact_evaluations,
    ensures
        candidates + 1 == (candidate_bound_pruned + 1) + exact_evaluations,
{
}

proof fn exact_step_preserves_candidate_partition(
    candidates: nat,
    candidate_bound_pruned: nat,
    exact_evaluations: nat,
)
    requires
        candidates == candidate_bound_pruned + exact_evaluations,
    ensures
        candidates + 1 == candidate_bound_pruned + (exact_evaluations + 1),
{
}

proof fn observational_subsets_are_bounded(
    columns_built: nat,
    column_pruned: nat,
    exact_evaluations: nat,
    cutoff_abandoned: nat,
)
    requires
        column_pruned <= columns_built,
        cutoff_abandoned <= exact_evaluations,
    ensures
        column_pruned + (columns_built - column_pruned) == columns_built,
        cutoff_abandoned + (exact_evaluations - cutoff_abandoned)
            == exact_evaluations,
{
}

}
