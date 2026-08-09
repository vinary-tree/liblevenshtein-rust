//! Verus model of the two-pass batch-arena fill (FV obligation #8).
//!
//! This file is verified directly with `verus`; it is not compiled by Cargo.
//!
//! `fill_batch` in src/ffi/index.rs builds a batch in two passes: pass one
//! appends each match term's bytes to a contiguous `byte_arena` and records
//! the term's start offset in `offsets`; pass two, run only AFTER the arena
//! has stopped growing, fixes up each descriptor's `term_data` to point at
//! `byte_arena.as_ptr().add(offsets[i])`. The realloc-safety of that fixup
//! rests entirely on the offset arithmetic being sound: every recorded offset
//! must still address the term's bytes in the FINAL arena, the windows must
//! not overlap, and slicing the arena at a term's window must recover exactly
//! that term. Raw pointers are outside Verus (they carry the
//! `ffi-leased-batch-aliasing` contract row and Miri coverage instead); this
//! model proves the pure sequence logic the fixup depends on.
//!
//! Laws (registry: docs/verification/ABI_INVARIANTS.tsv):
//!   LLEV-ARENA-1  offsets are the prefix sums: window i starts where window
//!                 i-1 ends (contiguous, gap-free packing);
//!   LLEV-ARENA-2  every window lies within the final arena (the fixup index
//!                 is always in bounds — the property realloc could break if
//!                 fixup ran before the arena stabilized);
//!   LLEV-ARENA-3  slicing the final arena at a term's window recovers the
//!                 term exactly (content fidelity), and distinct windows are
//!                 disjoint.
//!
//! Executable mirror: tests/ffi_arena_stability.rs (forces multiple arena
//! reallocations within one fill and derefs every descriptor post-fixup).

use vstd::prelude::*;

fn main() {}

verus! {

/// The start offset of term `i`: the total length of all terms before it.
/// This is exactly what pass one records in `offsets`.
pub open spec fn offset(terms: Seq<Seq<u8>>, i: int) -> nat
    decreases i,
{
    if i <= 0 {
        0
    } else {
        offset(terms, i - 1) + terms[i - 1].len()
    }
}

/// The arena after pass one: every term's bytes concatenated in order.
pub open spec fn arena(terms: Seq<Seq<u8>>) -> Seq<u8>
    decreases terms.len(),
{
    if terms.len() == 0 {
        Seq::empty()
    } else {
        arena(terms.drop_last()) + terms.last()
    }
}

/// The arena length equals the offset just past the last term.
proof fn arena_len_is_final_offset(terms: Seq<Seq<u8>>)
    ensures
        arena(terms).len() == offset(terms, terms.len() as int),
    decreases terms.len(),
{
    if terms.len() == 0 {
    } else {
        arena_len_is_final_offset(terms.drop_last());
        // offset(terms, len) = offset(terms, len-1) + terms[len-1].len(),
        // and offset agrees on the shared prefix with drop_last.
        offset_stable_under_drop_last(terms, terms.len() as int - 1);
    }
}

/// `offset` depends only on the terms before index `i`, so dropping the last
/// term does not change any offset at or below `len - 1`.
proof fn offset_stable_under_drop_last(terms: Seq<Seq<u8>>, i: int)
    requires
        0 <= i <= terms.len() - 1,
    ensures
        offset(terms, i) == offset(terms.drop_last(), i),
    decreases i,
{
    if i <= 0 {
    } else {
        offset_stable_under_drop_last(terms, i - 1);
        assert(terms[i - 1] == terms.drop_last()[i - 1]);
    }
}

/// LLEV-ARENA-1: windows are contiguous — window i ends exactly where window
/// i+1 begins.
proof fn windows_are_contiguous(terms: Seq<Seq<u8>>, i: int)
    requires
        0 <= i < terms.len(),
    ensures
        offset(terms, i) + terms[i].len() == offset(terms, i + 1),
{
    // Directly from the definition of offset at i+1.
}

/// `offset` is monotone nondecreasing.
proof fn offset_monotone(terms: Seq<Seq<u8>>, i: int, j: int)
    requires
        0 <= i <= j <= terms.len(),
    ensures
        offset(terms, i) <= offset(terms, j),
    decreases j - i,
{
    if i == j {
    } else {
        offset_monotone(terms, i, j - 1);
        // offset(terms, j) = offset(terms, j-1) + terms[j-1].len() >= offset(terms, j-1).
    }
}

/// LLEV-ARENA-2: every term's window lies within the final arena — the fixup
/// index `offset(i)` plus the term length never exceeds the arena length, so
/// the second-pass pointer arithmetic is always in bounds.
proof fn window_within_arena(terms: Seq<Seq<u8>>, i: int)
    requires
        0 <= i < terms.len(),
    ensures
        offset(terms, i) + terms[i].len() <= arena(terms).len(),
{
    windows_are_contiguous(terms, i);
    offset_monotone(terms, i + 1, terms.len() as int);
    arena_len_is_final_offset(terms);
}

/// Slicing the arena at term `i`'s window recovers exactly term `i`. Proved by
/// induction from the last term inward: the last window is the arena's tail,
/// and every earlier window lives entirely in `arena(drop_last)`.
proof fn window_recovers_term(terms: Seq<Seq<u8>>, i: int)
    requires
        0 <= i < terms.len(),
    ensures
        arena(terms).subrange(
            offset(terms, i) as int,
            (offset(terms, i) + terms[i].len()) as int,
        ) == terms[i],
    decreases terms.len(),
{
    let n = terms.len() as int;
    arena_len_is_final_offset(terms);
    if i == n - 1 {
        // The last window is the appended tail: arena = arena(drop_last) ++ last.
        arena_len_is_final_offset(terms.drop_last());
        offset_stable_under_drop_last(terms, n - 1);
        let head = arena(terms.drop_last());
        assert(arena(terms) == head + terms.last());
        assert(offset(terms, n - 1) == head.len());
        assert(terms[n - 1] == terms.last());
        // subrange(head.len(), head.len() + last.len()) of (head ++ last) == last.
        assert(arena(terms).subrange(head.len() as int, (head.len() + terms.last().len()) as int)
            =~= terms.last());
    } else {
        // Term i is entirely inside arena(drop_last); recurse there.
        let shorter = terms.drop_last();
        window_recovers_term(shorter, i);
        offset_stable_under_drop_last(terms, i);
        offset_stable_under_drop_last(terms, i + 1);
        windows_are_contiguous(shorter, i);
        offset_monotone(shorter, i + 1, shorter.len() as int);
        arena_len_is_final_offset(shorter);
        assert(terms[i] == shorter[i]);
        let head = arena(shorter);
        assert(arena(terms) == head + terms.last());
        // The window is within head, and prefix subranges of (head ++ tail)
        // agree with subranges of head.
        assert(offset(terms, i) + terms[i].len() <= head.len()) by {
            window_within_arena(shorter, i);
        }
        assert(arena(terms).subrange(offset(terms, i) as int, (offset(terms, i) + terms[i].len()) as int)
            =~= head.subrange(offset(shorter, i) as int, (offset(shorter, i) + shorter[i].len()) as int));
    }
}

/// LLEV-ARENA-3 (disjointness): distinct term windows do not overlap — a
/// direct corollary of contiguity and monotonicity, so no fixup writes alias.
proof fn windows_disjoint(terms: Seq<Seq<u8>>, i: int, j: int)
    requires
        0 <= i < j < terms.len(),
    ensures
        offset(terms, i) + terms[i].len() <= offset(terms, j),
{
    windows_are_contiguous(terms, i);
    offset_monotone(terms, i + 1, j);
}

}
