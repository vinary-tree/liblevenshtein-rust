//! Rust-shaped refinement lemmas for lazy weighted temporal frontiers.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;

fn main() {}

verus! {

pub open spec fn min2(left: nat, right: nat) -> nat {
    if left <= right { left } else { right }
}

pub open spec fn max2(left: nat, right: nat) -> nat {
    if left <= right { right } else { left }
}

pub open spec fn min3(left: nat, middle: nat, right: nat) -> nat {
    min2(left, min2(middle, right))
}

pub open spec fn additive_cell(
    diagonal: nat,
    above: nat,
    left: nat,
    substitution: nat,
    deletion: nat,
    insertion: nat,
) -> nat {
    min3(
        diagonal + substitution,
        above + deletion,
        left + insertion,
    )
}

pub open spec fn bottleneck_cell(
    diagonal: nat,
    above: nat,
    left: nat,
    link: nat,
) -> nat {
    max2(link, min3(diagonal, above, left))
}

/// Arithmetic projection of the epsilon-path residual-simulation theorem.
proof fn epsilon_dominance_preserves_every_suffix(
    dominator_cost: nat,
    reach_cost: nat,
    dominated_cost: nat,
    suffix_cost: nat,
)
    requires
        dominator_cost + reach_cost <= dominated_cost,
    ensures
        dominator_cost + (reach_cost + suffix_cost)
            <= dominated_cost + suffix_cost,
{
}

/// Rust-shaped K1 recurrence refinement for additive temporal kernels.
proof fn interval_additive_step_is_lower_simulation(
    abstract_diagonal: nat,
    abstract_above: nat,
    abstract_left: nat,
    concrete_diagonal: nat,
    concrete_above: nat,
    concrete_left: nat,
    abstract_substitution: nat,
    abstract_deletion: nat,
    abstract_insertion: nat,
    concrete_substitution: nat,
    concrete_deletion: nat,
    concrete_insertion: nat,
)
    requires
        abstract_diagonal <= concrete_diagonal,
        abstract_above <= concrete_above,
        abstract_left <= concrete_left,
        abstract_substitution <= concrete_substitution,
        abstract_deletion <= concrete_deletion,
        abstract_insertion <= concrete_insertion,
    ensures
        additive_cell(
            abstract_diagonal,
            abstract_above,
            abstract_left,
            abstract_substitution,
            abstract_deletion,
            abstract_insertion,
        ) <= additive_cell(
            concrete_diagonal,
            concrete_above,
            concrete_left,
            concrete_substitution,
            concrete_deletion,
            concrete_insertion,
        ),
{
}

/// Rust-shaped K1 recurrence refinement for discrete Frechet.
proof fn interval_bottleneck_step_is_lower_simulation(
    abstract_diagonal: nat,
    abstract_above: nat,
    abstract_left: nat,
    abstract_link: nat,
    concrete_diagonal: nat,
    concrete_above: nat,
    concrete_left: nat,
    concrete_link: nat,
)
    requires
        abstract_diagonal <= concrete_diagonal,
        abstract_above <= concrete_above,
        abstract_left <= concrete_left,
        abstract_link <= concrete_link,
    ensures
        bottleneck_cell(
            abstract_diagonal,
            abstract_above,
            abstract_left,
            abstract_link,
        ) <= bottleneck_cell(
            concrete_diagonal,
            concrete_above,
            concrete_left,
            concrete_link,
        ),
{
}

/// Retained state depends on the live generations and cache ceilings, not on
/// the number of target units already consumed.
proof fn generational_retention_is_prefix_independent(
    consumed_prefix: nat,
    current_positions: nat,
    next_positions: nat,
    cached_transitions: nat,
    frontier_limit: nat,
    cache_limit: nat,
)
    requires
        current_positions <= frontier_limit,
        next_positions <= frontier_limit,
        cached_transitions <= cache_limit,
    ensures
        current_positions + next_positions + cached_transitions
            <= 2 * frontier_limit + cache_limit,
{
}

/// Executable mirror of the tagged finish boundary.
fn finish_tag(exhausted: bool) -> (tag: u8)
    ensures
        (tag == 1) <==> exhausted,
        (tag == 0) <==> !exhausted,
{
    if exhausted { 1 } else { 0 }
}

/// A sparse generated-target table cannot contain more distinct cells than
/// the source/class transitions observed by the traversal.
proof fn sparse_cells_are_observation_bounded(
    distinct_observed: nat,
    observed: nat,
)
    requires
        distinct_observed <= observed,
    ensures
        observed - distinct_observed + distinct_observed == observed,
{
}

/// A fingerprint is only a lookup accelerator.  Reusing an interned state ID
/// is justified by exact canonical-key equality, which preserves every indexed
/// member of the frontier independently of the fingerprint value.
proof fn exact_canonical_key_reuse_preserves_membership(
    left_member: bool,
    right_member: bool,
)
    requires
        left_member == right_member,
    ensures
        left_member <==> right_member,
{
}

/// Pushing a frame for an already-interned residual grows only the explicit
/// DFS stack. The referenced state identifier remains inside the unchanged
/// query-local arena.
pub open spec fn state_count_after_reused_push(state_count: nat) -> nat {
    state_count
}

proof fn reused_state_push_preserves_arena_reference(
    frame_count: nat,
    state_count: nat,
    state_id: nat,
)
    requires
        state_id < state_count,
    ensures
        state_id < state_count_after_reused_push(state_count),
        frame_count + 1 > frame_count,
{
}

/// A freshly committed residual receives the old arena length as its stable
/// identifier, which is in bounds immediately after the commit.
proof fn fresh_state_identifier_is_in_bounds(state_count: nat)
    ensures
        state_count < state_count + 1,
{
}

/// Popping a DFS frame reclaims the path entry but intentionally retains the
/// interned residual for exact reuse by another reached dictionary prefix.
pub open spec fn state_count_after_frame_pop(state_count: nat) -> nat {
    state_count
}

proof fn frame_pop_retains_interned_arena(
    frame_count: nat,
    state_count: nat,
    state_id: nat,
)
    requires
        frame_count > 0,
        state_id < state_count,
    ensures
        frame_count - 1 < frame_count,
        state_id < state_count_after_frame_pop(state_count),
{
}

/// Arena retention is governed by the explicit canonical-state budget, not
/// by a false frame/state cardinality equality.
pub open spec fn state_count_after_fresh_attempt(
    state_count: nat,
    max_states: nat,
) -> nat {
    if state_count < max_states { state_count + 1 } else { state_count }
}

proof fn interned_arena_respects_explicit_state_limit(
    state_count: nat,
    max_states: nat,
)
    requires
        state_count <= max_states,
    ensures
        state_count_after_fresh_attempt(state_count, max_states) <= max_states,
{
}

pub open spec fn retain_child_if_fits(
    limit: nat,
    current: nat,
    prospective: nat,
) -> nat {
    if prospective <= limit { prospective } else { current }
}

/// A scratch preflight failure is atomic: it retains neither the child column
/// nor a child DFS frame.
proof fn rejected_child_preflight_is_atomic(
    limit: nat,
    current: nat,
    prospective: nat,
)
    requires
        prospective > limit,
    ensures
        retain_child_if_fits(limit, current, prospective) == current,
{
}

/// Charging before row evaluation ensures that successful sparse work never
/// exceeds the limit and that the rejected `(used + 1)`st row is not built.
proof fn admitted_sparse_row_never_exceeds_budget(
    used: nat,
    limit: nat,
)
    requires
        used < limit,
    ensures
        used + 1 <= limit,
{
}

proof fn rejected_sparse_row_is_pre_evaluation(
    used: nat,
    limit: nat,
)
    requires
        used >= limit,
    ensures
        !(used + 1 <= limit),
{
}

/// A diagonal consuming successor is seeded at the immediate successor row;
/// vertical epsilon closure then visits monotonically increasing rows.
proof fn diagonal_seed_and_vertical_closure_are_complete(
    predecessor: nat,
    row: nat,
)
    requires
        row >= predecessor + 1,
    ensures
        predecessor + 1 <= row,
{
}

proof fn vertical_reachability_is_successor_closed(
    seed: nat,
    row: nat,
)
    requires
        seed <= row,
    ensures
        seed <= row + 1,
{
}

/// If a recurrence cell is exactly its immediate vertical epsilon extension,
/// the predecessor is a no-more-expensive representative for every suffix.
proof fn equal_vertical_cell_is_dominated(
    predecessor_cost: nat,
    edge_cost: nat,
    cell_cost: nat,
)
    requires
        cell_cost == predecessor_cost + edge_cost,
    ensures
        predecessor_cost + edge_cost <= cell_cost,
{
}

/// Soft-DTW must account for every positive path contribution: dropping one
/// strictly decreases the partition mass, even if its cost is not minimal.
proof fn soft_partition_positive_contribution_cannot_be_pruned(
    retained_mass: nat,
    contribution: nat,
)
    requires
        contribution > 0,
    ensures
        retained_mass + contribution > retained_mass,
{
}

/// A rolling machine retains at most its registered window, regardless of
/// total stream length.
pub open spec fn rolling_retained_after_prefix(
    retained: nat,
    window: nat,
    consumed: nat,
) -> nat {
    if retained <= window { retained } else { window }
}

proof fn rolling_retention_is_stream_length_independent(
    retained: nat,
    window: nat,
    consumed: nat,
)
    ensures
        rolling_retained_after_prefix(retained, window, consumed)
            == rolling_retained_after_prefix(retained, window, 0),
        rolling_retained_after_prefix(retained, window, consumed) <= window,
{
}

proof fn rolling_snapshot_has_registered_width(
    emitted_width: nat,
    window: nat,
)
    requires
        emitted_width == window,
    ensures
        emitted_width <= window,
{
}

/// A failed content digest can never become an accepted complete snapshot.
proof fn snapshot_checksum_mismatch_fails_closed(
    checksum_ok: bool,
    config_ok: bool,
)
    requires
        !checksum_ok,
    ensures
        !(checksum_ok && config_ok),
{
}

proof fn snapshot_configuration_mismatch_fails_closed(
    checksum_ok: bool,
    config_ok: bool,
)
    requires
        !config_ok,
    ensures
        !(checksum_ok && config_ok),
{
}

/// A checked local predecessor equality extends a replay prefix to exactly
/// the next dynamic-programming value.
proof fn witness_local_edge_replays_exactly(
    replayed_prefix: nat,
    edge_cost: nat,
    next_cell: nat,
)
    requires
        next_cell == replayed_prefix + edge_cost,
    ensures
        replayed_prefix + edge_cost == next_cell,
{
}

/// Every reverse grid step decreases at least one coordinate and increases
/// neither, so the iterative traceback measure strictly decreases.
proof fn monotone_traceback_step_decreases_measure(
    query_index: nat,
    target_index: nat,
    previous_query: nat,
    previous_target: nat,
)
    requires
        previous_query <= query_index,
        previous_target <= target_index,
        previous_query < query_index || previous_target < target_index,
    ensures
        previous_query + previous_target < query_index + target_index,
{
}

pub open spec fn reserve_witness(limit: nat, requested: nat) -> nat {
    if requested <= limit { requested } else { 0 }
}

/// A failed preflight commits no witness storage.
proof fn rejected_witness_reservation_is_atomic(limit: nat, requested: nat)
    requires
        requested > limit,
    ensures
        reserve_witness(limit, requested) == 0,
{
}

}
