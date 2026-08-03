//! Verus model of the Rust language-product safety and algebraic invariants.
//!
//! This file is verified directly with `verus`; it is not compiled by Cargo.

use vstd::prelude::*;
use vstd::assert_isets_equal;

fn main() {}

verus! {

pub uninterp spec fn edge(source: u32, unit: u64, target: u32) -> bool;

/// Relational image of a language-state set under one input unit.
pub open spec fn step_set(states: ISet<u32>, unit: u64) -> ISet<u32> {
    ISet::new(|target: u32| exists|source: u32|
        states.contains(source) && edge(source, unit, target))
}

/// A level accepts when it contains at least one final language state.
pub open spec fn level_accepts(states: ISet<u32>, finals: ISet<u32>) -> bool {
    exists|state: u32| states.contains(state) && finals.contains(state)
}

/// Clearing states already present at a cheaper level preserves whether either
/// level accepts. This is the two-level induction step used by canonicalize.
proof fn canonicalization_preserves_acceptance(
    cheaper: ISet<u32>,
    dearer: ISet<u32>,
    finals: ISet<u32>,
)
    ensures
        level_accepts(cheaper, finals) || level_accepts(dearer, finals)
        <==>
        level_accepts(cheaper, finals)
            || level_accepts(dearer.difference(cheaper), finals),
{
    if level_accepts(dearer, finals) {
        let witness = choose|state: u32|
            dearer.contains(state) && finals.contains(state);
        if cheaper.contains(witness) {
            assert(level_accepts(cheaper, finals));
        } else {
            assert(dearer.difference(cheaper).contains(witness));
            assert(level_accepts(dearer.difference(cheaper), finals));
        }
    }
    if level_accepts(dearer.difference(cheaper), finals) {
        let witness = choose|state: u32|
            dearer.difference(cheaper).contains(state) && finals.contains(state);
        assert(dearer.contains(witness));
        assert(level_accepts(dearer, finals));
    }
}

/// Relational image distributes over set union. This is the frontier merge law
/// at one exact cost; pointwise application proves the full frontier theorem.
proof fn step_distributes_over_union(left: ISet<u32>, right: ISet<u32>, unit: u64)
    ensures
        step_set(left.union(right), unit)
            == step_set(left, unit).union(step_set(right, unit)),
{
    assert_isets_equal!(
        step_set(left.union(right), unit),
        step_set(left, unit).union(step_set(right, unit)),
        target => {
            if step_set(left.union(right), unit).contains(target) {
                let source = choose|source: u32|
                    left.union(right).contains(source) && edge(source, unit, target);
                if left.contains(source) {
                    assert(step_set(left, unit).contains(target));
                } else {
                    assert(right.contains(source));
                    assert(step_set(right, unit).contains(target));
                }
            }
            if step_set(left, unit).union(step_set(right, unit)).contains(target) {
                if step_set(left, unit).contains(target) {
                    let source = choose|source: u32|
                        left.contains(source) && edge(source, unit, target);
                    assert(left.union(right).contains(source));
                    assert(step_set(left.union(right), unit).contains(target));
                } else {
                    let source = choose|source: u32|
                        right.contains(source) && edge(source, unit, target);
                    assert(left.union(right).contains(source));
                    assert(step_set(left.union(right), unit).contains(target));
                }
            }
        }
    );
}

/// Executable mirror of the Rust `level + 1` branch. Preconditions are exactly
/// the guards in `step` and `deletion_closure`.
fn checked_next_level(level: usize, max_distance: u8) -> (next: usize)
    requires
        level < max_distance as usize,
    ensures
        next == level + 1,
        next <= max_distance as usize,
        next < 256,
{
    level + 1
}

/// A `u8` edit budget always allocates a nonempty frontier representable in
/// `usize`, and the level count is bounded by 256.
proof fn frontier_level_bound(max_distance: u8)
    ensures
        1 <= max_distance as nat + 1,
        max_distance as nat + 1 <= 256,
{
}

/// Executable mirror of the small-DFA bit operation. The constructor enforces
/// the same `< 31` state bound, so this shift cannot overflow or enter the
/// reserved sink bit.
fn checked_state_bit(state: u32) -> (bit: u32)
    requires
        state < 31,
    ensures
        bit == 1u32 << state,
{
    1u32 << state
}

}
