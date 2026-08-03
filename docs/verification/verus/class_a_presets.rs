//! Verus obligations for Class-A preset and validation arithmetic.
//!
//! This file is verified directly with `verus`; Cargo does not compile it.

use vstd::prelude::*;

fn main() {}

verus! {

spec fn indel_cost(inserted: nat, deleted: nat) -> nat {
    inserted + deleted
}

spec fn alignment_consumption(kept: nat, inserted: nat, deleted: nat) -> (nat, nat) {
    (kept + deleted, kept + inserted)
}

proof fn hamming_coordinate_triangle(
    left_right: nat,
    left_middle: nat,
    middle_right: nat,
)
    requires
        left_right <= 1,
        left_middle <= 1,
        middle_right <= 1,
        left_right > 0 ==> left_middle > 0 || middle_right > 0,
    ensures
        left_right <= left_middle + middle_right,
{
}

proof fn hamming_sum_triangle(
    prefix_left_right: nat,
    prefix_left_middle: nat,
    prefix_middle_right: nat,
    cell_left_right: nat,
    cell_left_middle: nat,
    cell_middle_right: nat,
)
    requires
        prefix_left_right <= prefix_left_middle + prefix_middle_right,
        cell_left_right <= cell_left_middle + cell_middle_right,
    ensures
        prefix_left_right + cell_left_right
            <= (prefix_left_middle + cell_left_middle)
                + (prefix_middle_right + cell_middle_right),
{
}

proof fn reversing_indel_counts_preserves_cost(
    kept: nat,
    inserted: nat,
    deleted: nat,
)
    ensures
        indel_cost(inserted, deleted) == indel_cost(deleted, inserted),
        alignment_consumption(kept, inserted, deleted).0
            == alignment_consumption(kept, deleted, inserted).1,
        alignment_consumption(kept, inserted, deleted).1
            == alignment_consumption(kept, deleted, inserted).0,
{
}

proof fn indel_length_lower_bounds(
    kept: nat,
    inserted: nat,
    deleted: nat,
)
    ensures
        kept + deleted <= kept + inserted + inserted + deleted,
        kept + inserted <= kept + deleted + inserted + deleted,
{
}

proof fn concatenating_indel_scripts_adds_cost(
    first_inserted: nat,
    first_deleted: nat,
    second_inserted: nat,
    second_deleted: nat,
)
    ensures
        (first_inserted + second_inserted) + (first_deleted + second_deleted)
            == (first_inserted + first_deleted) + (second_inserted + second_deleted),
{
}

proof fn bounded_skip_has_exact_directional_cost(matched: nat, skipped: nat)
    ensures
        matched <= matched + skipped,
        (matched + skipped) - matched == skipped,
{
}

proof fn validation_prefix_stays_bounded(prefix: nat, suffix: nat, limit: nat)
    requires
        prefix + suffix <= limit,
    ensures
        prefix <= limit,
{
}

proof fn progressing_operation_advances(source: nat, target: nat)
    requires
        source + target > 0,
    ensures
        source > 0 || target > 0,
{
}

proof fn checked_aggregate_update_stays_bounded(
    aggregate: nat,
    source: nat,
    target: nat,
    limit: nat,
)
    requires
        aggregate <= limit,
        source + target <= limit - aggregate,
    ensures
        aggregate + source + target <= limit,
{
}

spec fn empty_side_indel_distance(length: nat) -> nat {
    length
}

proof fn affordable_empty_side_is_exact(length: nat, budget: nat)
    ensures
        (empty_side_indel_distance(length) <= budget) == (length <= budget),
{
}

}
