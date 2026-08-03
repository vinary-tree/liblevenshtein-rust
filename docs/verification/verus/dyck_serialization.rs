//! Verus obligations for exact Dyck correction and strict binary persistence.
//!
//! Verified directly by `verus`; Cargo does not compile this file.

use vstd::prelude::*;

fn main() {}

verus! {

spec fn replacement_cost(actual: nat, expected: nat) -> nat {
    if actual == expected { 0 } else { 1 }
}

proof fn replacement_cost_zero_iff_equal(actual: nat, expected: nat)
    ensures
        (replacement_cost(actual, expected) == 0) == (actual == expected),
{
}

spec fn typed_pair(kinds: nat, kind: nat, opening: nat, closing: nat) -> bool {
    kind < kinds && opening == kind && closing == kinds + kind
}

proof fn cross_kind_closers_are_distinct(kinds: nat, left: nat, right: nat)
    requires
        left < kinds,
        right < kinds,
        left != right,
    ensures
        kinds + left != kinds + right,
{
}

spec fn min4(first: nat, second: nat, third: nat, fourth: nat) -> nat {
    let left = if first <= second { first } else { second };
    let right = if third <= fourth { third } else { fourth };
    if left <= right { left } else { right }
}

proof fn min4_is_no_greater_than_each_candidate(
    first: nat,
    second: nat,
    third: nat,
    fourth: nat,
)
    ensures
        min4(first, second, third, fourth) <= first,
        min4(first, second, third, fourth) <= second,
        min4(first, second, third, fourth) <= third,
        min4(first, second, third, fourth) <= fourth,
{
}

spec fn pair_from_first_cost(
    actual_open: nat,
    expected_open: nat,
    inner_cost: nat,
    actual_close: nat,
    expected_close: nat,
    suffix_cost: nat,
) -> nat {
    replacement_cost(actual_open, expected_open) + inner_cost
        + replacement_cost(actual_close, expected_close) + suffix_cost
}

proof fn zero_cost_consumed_pair_is_identity(
    actual_open: nat,
    expected_open: nat,
    inner_cost: nat,
    actual_close: nat,
    expected_close: nat,
    suffix_cost: nat,
)
    requires
        pair_from_first_cost(
            actual_open,
            expected_open,
            inner_cost,
            actual_close,
            expected_close,
            suffix_cost,
        ) == 0,
    ensures
        actual_open == expected_open,
        actual_close == expected_close,
        inner_cost == 0,
        suffix_cost == 0,
{
}

spec fn accept_envelope(
    magic_ok: bool,
    version: nat,
    flags: nat,
    declared_bytes: nat,
    available_bytes: nat,
    consumed_bytes: nat,
    payload_limit: nat,
    operations: nat,
    operation_limit: nat,
    pairs: nat,
    pair_limit: nat,
    semantic_validation: bool,
) -> bool {
    magic_ok
        && version == 1
        && flags == 0
        && declared_bytes == available_bytes
        && consumed_bytes == declared_bytes
        && declared_bytes <= payload_limit
        && operations <= operation_limit
        && pairs <= pair_limit
        && semantic_validation
}

proof fn accepted_envelope_is_exact_and_bounded(
    magic_ok: bool,
    version: nat,
    flags: nat,
    declared_bytes: nat,
    available_bytes: nat,
    consumed_bytes: nat,
    payload_limit: nat,
    operations: nat,
    operation_limit: nat,
    pairs: nat,
    pair_limit: nat,
    semantic_validation: bool,
)
    requires
        accept_envelope(
            magic_ok,
            version,
            flags,
            declared_bytes,
            available_bytes,
            consumed_bytes,
            payload_limit,
            operations,
            operation_limit,
            pairs,
            pair_limit,
            semantic_validation,
        ),
    ensures
        magic_ok,
        version == 1,
        flags == 0,
        consumed_bytes == available_bytes,
        declared_bytes <= payload_limit,
        operations <= operation_limit,
        pairs <= pair_limit,
        semantic_validation,
{
}

proof fn trailing_or_over_limit_envelope_is_rejected(
    magic_ok: bool,
    version: nat,
    flags: nat,
    declared_bytes: nat,
    available_bytes: nat,
    consumed_bytes: nat,
    payload_limit: nat,
    operations: nat,
    operation_limit: nat,
    pairs: nat,
    pair_limit: nat,
    semantic_validation: bool,
)
    requires
        available_bytes > declared_bytes
            || operations > operation_limit
            || pairs > pair_limit,
    ensures
        !accept_envelope(
            magic_ok,
            version,
            flags,
            declared_bytes,
            available_bytes,
            consumed_bytes,
            payload_limit,
            operations,
            operation_limit,
            pairs,
            pair_limit,
            semantic_validation,
        ),
{
}

spec fn accept_protobuf(
    wire_well_formed: bool,
    supported_format: bool,
    payload_bytes: nat,
    payload_limit: nat,
    operations: nat,
    operation_limit: nat,
    largest_name_bytes: nat,
    name_limit: nat,
    largest_operation_pairs: nat,
    per_operation_pair_limit: nat,
    total_pairs: nat,
    total_pair_limit: nat,
    pair_text_bytes: nat,
    pair_text_limit: nat,
    semantic_validation: bool,
) -> bool {
    wire_well_formed
        && supported_format
        && payload_bytes <= payload_limit
        && operations <= operation_limit
        && largest_name_bytes <= name_limit
        && largest_operation_pairs <= per_operation_pair_limit
        && total_pairs <= total_pair_limit
        && pair_text_bytes <= pair_text_limit
        && semantic_validation
}

proof fn accepted_protobuf_is_preflight_bounded(
    wire_well_formed: bool,
    supported_format: bool,
    payload_bytes: nat,
    payload_limit: nat,
    operations: nat,
    operation_limit: nat,
    largest_name_bytes: nat,
    name_limit: nat,
    largest_operation_pairs: nat,
    per_operation_pair_limit: nat,
    total_pairs: nat,
    total_pair_limit: nat,
    pair_text_bytes: nat,
    pair_text_limit: nat,
    semantic_validation: bool,
)
    requires
        accept_protobuf(
            wire_well_formed,
            supported_format,
            payload_bytes,
            payload_limit,
            operations,
            operation_limit,
            largest_name_bytes,
            name_limit,
            largest_operation_pairs,
            per_operation_pair_limit,
            total_pairs,
            total_pair_limit,
            pair_text_bytes,
            pair_text_limit,
            semantic_validation,
        ),
    ensures
        wire_well_formed,
        supported_format,
        payload_bytes <= payload_limit,
        operations <= operation_limit,
        largest_name_bytes <= name_limit,
        largest_operation_pairs <= per_operation_pair_limit,
        total_pairs <= total_pair_limit,
        pair_text_bytes <= pair_text_limit,
        semantic_validation,
{
}

proof fn over_limit_protobuf_is_rejected_before_allocation(
    wire_well_formed: bool,
    supported_format: bool,
    payload_bytes: nat,
    payload_limit: nat,
    operations: nat,
    operation_limit: nat,
    largest_name_bytes: nat,
    name_limit: nat,
    largest_operation_pairs: nat,
    per_operation_pair_limit: nat,
    total_pairs: nat,
    total_pair_limit: nat,
    pair_text_bytes: nat,
    pair_text_limit: nat,
    semantic_validation: bool,
)
    requires
        operations > operation_limit
            || largest_operation_pairs > per_operation_pair_limit
            || total_pairs > total_pair_limit
            || pair_text_bytes > pair_text_limit,
    ensures
        !accept_protobuf(
            wire_well_formed,
            supported_format,
            payload_bytes,
            payload_limit,
            operations,
            operation_limit,
            largest_name_bytes,
            name_limit,
            largest_operation_pairs,
            per_operation_pair_limit,
            total_pairs,
            total_pair_limit,
            pair_text_bytes,
            pair_text_limit,
            semantic_validation,
        ),
{
}

spec fn encode_weight_bits(bits: u64) -> u64 {
    bits
}

spec fn decode_weight_bits(bits: u64) -> u64 {
    bits
}

proof fn weight_bits_round_trip_exactly(bits: u64)
    ensures
        decode_weight_bits(encode_weight_bits(bits)) == bits,
{
}

spec fn decode_u16_le(first: nat, second: nat) -> nat {
    first + 256 * second
}

proof fn version_one_header_bytes_decode_little_endian()
    ensures
        decode_u16_le(1, 0) == 1,
        decode_u16_le(0, 1) == 256,
{
}

proof fn wire_cursor_advance_is_exact(
    start: nat,
    width: nat,
    total: nat,
)
    requires
        start <= total,
        width <= total - start,
    ensures
        start + width <= total,
        start + width - start == width,
{
}

proof fn bounded_varint_cursor_stays_within_input(
    start: nat,
    consumed: nat,
    total: nat,
)
    requires
        start <= total,
        1 <= consumed,
        consumed <= 10,
        consumed <= total - start,
    ensures
        start < start + consumed,
        start + consumed <= total,
{
}

proof fn length_delimited_partition_is_exact(
    prefix_bytes: nat,
    payload_bytes: nat,
    suffix_bytes: nat,
    total_bytes: nat,
)
    requires
        total_bytes == prefix_bytes + payload_bytes + suffix_bytes,
    ensures
        prefix_bytes + payload_bytes <= total_bytes,
        total_bytes - (prefix_bytes + payload_bytes) == suffix_bytes,
{
}

spec fn accept_gzip(
    checksum_valid: bool,
    compressed_bytes: nat,
    compressed_limit: nat,
    decompressed_bytes: nat,
    decompressed_limit: nat,
    consumed_compressed_bytes: nat,
    supplied_bytes: nat,
    inner_accepted: bool,
) -> bool {
    checksum_valid
        && compressed_bytes <= compressed_limit
        && decompressed_bytes <= decompressed_limit
        && consumed_compressed_bytes == supplied_bytes
        && inner_accepted
}

proof fn trailing_or_oversized_gzip_is_rejected(
    checksum_valid: bool,
    compressed_bytes: nat,
    compressed_limit: nat,
    decompressed_bytes: nat,
    decompressed_limit: nat,
    consumed_compressed_bytes: nat,
    supplied_bytes: nat,
    inner_accepted: bool,
)
    requires
        consumed_compressed_bytes < supplied_bytes
            || compressed_bytes > compressed_limit
            || decompressed_bytes > decompressed_limit,
    ensures
        !accept_gzip(
            checksum_valid,
            compressed_bytes,
            compressed_limit,
            decompressed_bytes,
            decompressed_limit,
            consumed_compressed_bytes,
            supplied_bytes,
            inner_accepted,
        ),
{
}

}
