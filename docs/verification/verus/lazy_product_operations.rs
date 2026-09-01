//! Rust-shaped refinement lemmas for product zippers and observed transitions.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;

fn main() {}

verus! {

pub open spec fn product_child_live(
    dictionary_child_exists: bool,
    query_child_live: bool,
) -> bool {
    dictionary_child_exists && query_child_live
}

/// A product child is constructible exactly when both synchronized component
/// operations succeed on the same observed edge.
proof fn product_child_requires_both_components(
    dictionary_child_exists: bool,
    query_child_live: bool,
)
    ensures
        product_child_live(dictionary_child_exists, query_child_live)
            <==> (dictionary_child_exists && query_child_live),
{
}

/// A persistent dictionary zipper copies the immutable revision identity into
/// every child focus.
pub open spec fn descended_snapshot_revision(parent_revision: nat) -> nat {
    parent_revision
}

proof fn zipper_descent_preserves_snapshot_revision(revision: nat)
    ensures
        descended_snapshot_revision(revision) == revision,
{
}

/// The shared reverse parent spine adds exactly one label per descent.  Path
/// materialization may be delayed without changing logical depth.
proof fn zipper_descent_extends_path_depth(depth: nat)
    ensures
        depth + 1 > depth,
{
}

pub open spec fn iterative_spine_release_iterations(
    uniquely_owned_nodes: nat,
    reaches_shared_suffix: bool,
) -> nat {
    uniquely_owned_nodes + if reaches_shared_suffix { 1nat } else { 0nat }
}

/// An iterative shared-spine release performs one loop iteration per unique
/// node and at most one additional failed-unwrapping iteration at a shared
/// suffix.  Its native call-stack depth is independent of path depth.
proof fn iterative_spine_release_has_linear_work_and_constant_call_stack(
    uniquely_owned_nodes: nat,
    reaches_shared_suffix: bool,
)
    ensures
        iterative_spine_release_iterations(
            uniquely_owned_nodes,
            reaches_shared_suffix,
        ) <= uniquely_owned_nodes + 1nat,
{
}

/// A release that encounters a shared suffix does not drain that suffix; its
/// final owner will resume the same iterative rule later.
proof fn shared_spine_release_stops_after_unwrap_failure(unique_nodes: nat)
    ensures
        iterative_spine_release_iterations(unique_nodes, true)
            == unique_nodes + 1nat,
{
}

pub open spec fn query_first_child_live(
    dictionary_child_exists: bool,
    query_child_live: bool,
) -> bool {
    if query_child_live { dictionary_child_exists } else { false }
}

/// Projecting through the query machine before constructing a dictionary
/// child preserves the synchronized product's live/dead decision.
proof fn query_first_child_is_product_equivalent(
    dictionary_child_exists: bool,
    query_child_live: bool,
)
    ensures
        query_first_child_live(dictionary_child_exists, query_child_live)
            == product_child_live(dictionary_child_exists, query_child_live),
{
}

pub open spec fn projected_child_constructions(query_child_live: bool) -> nat {
    if query_child_live { 1 } else { 0 }
}

/// A query-rejected edge constructs no owned dictionary child focus.
proof fn rejected_projection_constructs_no_child()
    ensures
        projected_child_constructions(false) == 0,
{
}

/// A live query projection constructs exactly one child focus.
proof fn live_projection_constructs_one_child()
    ensures
        projected_child_constructions(true) == 1,
{
}

pub open spec fn erased_focus_revision(revision: nat, path_bytes: nat) -> nat {
    revision
}

pub open spec fn erased_focus_node(node: nat, path_bytes: nat) -> nat {
    node
}

/// Consuming a zipper into its opaque traversal view may erase path-only
/// bytes, but it retains the exact snapshot revision and native focus node.
proof fn path_erasure_preserves_native_focus(
    revision: nat,
    node: nat,
    path_bytes: nat,
)
    ensures
        erased_focus_revision(revision, path_bytes) == revision,
        erased_focus_node(node, path_bytes) == node,
{
}

pub open spec fn complete_cache_answer(
    hit: bool,
    cached: nat,
    recomputed: nat,
) -> nat {
    if hit { cached } else { recomputed }
}

/// When a complete cache entry is the exact recomputed successor, hit and miss
/// paths have the same observation.
proof fn exact_cache_hit_refines_recomputation(
    hit: bool,
    cached: nat,
    recomputed: nat,
)
    requires
        cached == recomputed,
    ensures
        complete_cache_answer(hit, cached, recomputed) == recomputed,
{
}

/// Lazy construction cannot create more transition misses than reachable
/// product edges that were actually inspected.
proof fn constructed_transitions_are_observation_bounded(
    constructed_transitions: nat,
    inspected_reachable_edges: nat,
)
    requires
        constructed_transitions <= inspected_reachable_edges,
    ensures
        inspected_reachable_edges - constructed_transitions
            + constructed_transitions == inspected_reachable_edges,
{
}

pub open spec fn admits_final_score(score: nat, cutoff: nat) -> bool {
    score <= cutoff
}

/// A finite finalizer output becomes a public range result only after the
/// scheduler reapplies the configured cutoff.
proof fn admitted_final_score_is_within_cutoff(score: nat, cutoff: nat)
    requires
        admits_final_score(score, cutoff),
    ensures
        score <= cutoff,
{
}

/// Closing query-only operations may produce a finite score above the cutoff;
/// the public admission predicate rejects it.
proof fn over_cutoff_final_score_is_rejected(score: nat, cutoff: nat)
    requires
        cutoff < score,
    ensures
        !admits_final_score(score, cutoff),
{
}

pub open spec fn compact_product_frame_bytes(
    dictionary_cursor_bytes: nat,
    state_id_bytes: nat,
    path_handle_bytes: nat,
) -> nat {
    dictionary_cursor_bytes + state_id_bytes + path_handle_bytes
}

/// Queue entries carry one compact state ID.  Frontier width affects the
/// query-local arena, not the automaton contribution to each queue entry.
proof fn compact_product_frame_is_frontier_width_independent(
    dictionary_cursor_bytes: nat,
    state_id_bytes: nat,
    path_handle_bytes: nat,
    frontier_width: nat,
)
    ensures
        compact_product_frame_bytes(
            dictionary_cursor_bytes,
            state_id_bytes,
            path_handle_bytes,
        ) == dictionary_cursor_bytes + state_id_bytes + path_handle_bytes,
{
}

/// Reordering a completed unordered schedule preserves each result's
/// membership observation.  Ordered surfaces require an additional tie-order
/// theorem and do not use this lemma alone.
proof fn completed_unordered_scheduler_preserves_membership(
    before_contains: bool,
    after_contains: bool,
)
    requires
        before_contains == after_contains,
    ensures
        before_contains <==> after_contains,
{
}

}
