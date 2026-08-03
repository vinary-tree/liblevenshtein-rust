//! Verus model of the representation-preserving Phase 11 subsumption fallback.

use vstd::prelude::*;

fn main() {}

verus! {

pub open spec fn abs_diff(left: int, right: int) -> int {
    if left <= right { right - left } else { left - right }
}

pub open spec fn shared_unit(
    mode: int,
    li: int,
    lc: int,
    ls: bool,
    ri: int,
    rc: int,
    rs: bool,
    query_length: int,
    index_step: int,
) -> bool {
    lc <= rc &&
    if mode == 0 {
        abs_diff(li, ri) * index_step <= rc - lc
    } else if mode == 1 {
        if ls && rs {
            li == ri
        } else if ls || rs {
            false
        } else {
            abs_diff(li, ri) * index_step <= rc - lc
        }
    } else {
        ls == rs
            && li <= query_length
            && !(ls && li >= query_length && ri < query_length)
            && lc < rc
            && li == ri
    }
}

pub open spec fn legacy_unit(
    mode: int,
    li: int,
    lc: int,
    ls: bool,
    ri: int,
    rc: int,
    rs: bool,
    query_length: int,
    index_step: int,
) -> bool {
    if mode == 0 {
        lc <= rc && abs_diff(li, ri) * index_step <= rc - lc
    } else if mode == 1 {
        if ls && rs {
            lc <= rc && li == ri
        } else if ls || rs {
            false
        } else {
            lc <= rc && abs_diff(li, ri) * index_step <= rc - lc
        }
    } else {
        lc <= rc
            && ls == rs
            && li <= query_length
            && !(ls && li >= query_length && ri < query_length)
            && lc < rc
            && li == ri
    }
}

proof fn shared_unit_is_legacy(
    mode: int,
    li: int,
    lc: int,
    ls: bool,
    ri: int,
    rc: int,
    rs: bool,
    query_length: int,
    index_step: int,
)
    requires
        mode <= 2,
        mode >= 0,
        li >= 0,
        lc >= 0,
        ri >= 0,
        rc >= 0,
        query_length >= 0,
        index_step >= 0,
    ensures
        shared_unit(mode, li, lc, ls, ri, rc, rs, query_length, index_step)
            == legacy_unit(mode, li, lc, ls, ri, rc, rs, query_length, index_step),
{
    if mode == 0 {
    } else if mode == 1 {
        if ls && rs {
        } else if ls || rs {
        }
    }
}

proof fn mixed_transposition_states_are_incomparable(
    li: int,
    lc: int,
    ri: int,
    rc: int,
    query_length: int,
    index_step: int,
)
    requires
        li >= 0,
        lc >= 0,
        ri >= 0,
        rc >= 0,
        query_length >= 0,
        index_step >= 0,
    ensures
        !shared_unit(1, li, lc, true, ri, rc, false, query_length, index_step),
        !shared_unit(1, li, lc, false, ri, rc, true, query_length, index_step),
{
}

}
