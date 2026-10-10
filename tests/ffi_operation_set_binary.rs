#![cfg(all(feature = "ffi", feature = "serialization"))]

use liblevenshtein::ffi::{
    llev_decoded_operation_set_free, llev_decoded_operation_set_len,
    llev_decoded_operation_set_operation_at, llev_decoded_operation_set_restriction_at,
    llev_decoded_operation_set_serialize, llev_operation_set_deserialize,
    llev_operation_set_serialize, llev_owned_bytes_free, LlevDecodedOperationSet,
    LlevGeneralizedOperation, LlevGeneralizedRestriction, LlevOperationSetLimits, LlevOwnedBytes,
    LlevSerializedOperationView, LlevSerializedRestrictionView, LlevStatus,
};
use liblevenshtein::transducer::{
    OperationApplicability, OperationSet, OperationType, SubstitutionSet,
};

fn limits() -> LlevOperationSetLimits {
    LlevOperationSetLimits {
        max_payload_bytes: 4096,
        max_operations: 8,
        max_operation_name_bytes: 64,
        max_restriction_pairs_per_operation: 8,
        max_total_restriction_pairs: 8,
        max_restriction_text_bytes: 64,
    }
}

#[cfg(all(feature = "protobuf", feature = "compression"))]
#[test]
fn rc5_operation_payloads_decode_with_current_native_bridge() {
    let fixtures: [&[u8]; 4] = [
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/operation-binary-v1.bin"),
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/operation-protobuf-v1.bin"),
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/operation-gzip-binary-v1.bin"),
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/operation-gzip-protobuf-v1.bin"),
    ];
    for (index, bytes) in fixtures.into_iter().enumerate() {
        let mut decoded: *mut LlevDecodedOperationSet = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_operation_set_deserialize(
                    (index + 1) as u32,
                    bytes.as_ptr(),
                    bytes.len(),
                    &limits(),
                    &mut decoded,
                )
            },
            LlevStatus::Ok,
            "RC.5 operation format {} did not decode",
            index + 1
        );
        let mut count = 0;
        assert_eq!(
            unsafe { llev_decoded_operation_set_len(decoded, &mut count) },
            LlevStatus::Ok
        );
        assert_eq!(count, 3);
        let mut byte = LlevSerializedRestrictionView::default();
        assert_eq!(
            unsafe { llev_decoded_operation_set_restriction_at(decoded, 1, 0, &mut byte) },
            LlevStatus::Ok
        );
        assert_eq!(
            (byte.kind, byte.source_byte, byte.target_byte),
            (1, 0xff, 0x80)
        );
        let mut text = LlevSerializedRestrictionView::default();
        assert_eq!(
            unsafe { llev_decoded_operation_set_restriction_at(decoded, 2, 0, &mut text) },
            LlevStatus::Ok
        );
        assert_eq!(text.kind, 2);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(text.source_data, text.source_len) },
            b"ph"
        );
        let mut reencoded = LlevOwnedBytes::default();
        assert_eq!(
            unsafe {
                llev_decoded_operation_set_serialize(
                    decoded,
                    (index + 1) as u32,
                    &limits(),
                    &mut reencoded,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(
            unsafe { std::slice::from_raw_parts(reencoded.data, reencoded.len) },
            bytes
        );
        unsafe {
            llev_decoded_operation_set_free(decoded);
            llev_owned_bytes_free(&mut reencoded);
        }
    }
}

fn native_set() -> OperationSet {
    let mut set = OperationSet::new();
    set.add(OperationType::with_owned_applicability(
        1,
        1,
        0.0,
        OperationApplicability::Equal,
        "match",
    ));
    let mut restrictions = SubstitutionSet::new();
    restrictions.allow_str("ph", "f");
    set.add(OperationType::with_owned_applicability(
        2,
        1,
        0.25,
        OperationApplicability::Listed(restrictions),
        "digraph",
    ));
    set
}

fn borrowed_operations() -> (
    [LlevGeneralizedOperation; 2],
    [LlevGeneralizedRestriction; 1],
) {
    let pairs = [LlevGeneralizedRestriction {
        source_data: b"ph".as_ptr().cast(),
        source_len: 2,
        target_data: b"f".as_ptr().cast(),
        target_len: 1,
    }];
    let operations = [
        LlevGeneralizedOperation {
            consume_source: 1,
            consume_target: 1,
            weight: 0.0,
            name_data: b"match".as_ptr().cast(),
            name_len: 5,
            applicability: 1,
            reserved: 0,
            restrictions: std::ptr::null(),
            restriction_count: 0,
        },
        LlevGeneralizedOperation {
            consume_source: 2,
            consume_target: 1,
            weight: 0.25,
            name_data: b"digraph".as_ptr().cast(),
            name_len: 7,
            applicability: 3,
            reserved: 0,
            restrictions: std::ptr::null(),
            restriction_count: 1,
        },
    ];
    (operations, pairs)
}

#[cfg(all(feature = "protobuf", feature = "compression"))]
#[test]
fn all_formats_match_native_bytes_and_round_trip() {
    let native = native_set();
    let (mut operations, pairs) = borrowed_operations();
    operations[1].restrictions = pairs.as_ptr();
    for format_id in 1..=4 {
        let expected = match format_id {
            1 => native.to_binary().unwrap(),
            2 => native.to_protobuf().unwrap(),
            3 => native.to_binary_gzip().unwrap(),
            4 => native.to_protobuf_gzip().unwrap(),
            _ => unreachable!(),
        };
        let mut encoded = LlevOwnedBytes::default();
        assert_eq!(
            unsafe {
                llev_operation_set_serialize(
                    format_id,
                    operations.as_ptr(),
                    operations.len(),
                    &limits(),
                    &mut encoded,
                )
            },
            LlevStatus::Ok
        );
        assert_eq!(
            unsafe { std::slice::from_raw_parts(encoded.data, encoded.len) },
            expected
        );
        let mut decoded: *mut LlevDecodedOperationSet = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_operation_set_deserialize(
                    format_id,
                    expected.as_ptr(),
                    expected.len(),
                    &limits(),
                    &mut decoded,
                )
            },
            LlevStatus::Ok
        );
        let mut count = 0;
        assert_eq!(
            unsafe { llev_decoded_operation_set_len(decoded, &mut count) },
            LlevStatus::Ok
        );
        assert_eq!(count, 2);
        let mut op = LlevSerializedOperationView::default();
        assert_eq!(
            unsafe { llev_decoded_operation_set_operation_at(decoded, 1, &mut op) },
            LlevStatus::Ok
        );
        assert_eq!(op.restriction_count, 1);
        assert_eq!(op.applicability, 3);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(op.name_data, op.name_len) },
            b"digraph"
        );
        let mut pair = LlevSerializedRestrictionView::default();
        assert_eq!(
            unsafe { llev_decoded_operation_set_restriction_at(decoded, 1, 0, &mut pair) },
            LlevStatus::Ok
        );
        assert_eq!(pair.kind, 2);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(pair.source_data, pair.source_len) },
            b"ph"
        );
        let mut reencoded = LlevOwnedBytes::default();
        assert_eq!(
            unsafe {
                llev_decoded_operation_set_serialize(decoded, format_id, &limits(), &mut reencoded)
            },
            LlevStatus::Ok
        );
        assert_eq!(
            unsafe { std::slice::from_raw_parts(reencoded.data, reencoded.len) },
            expected
        );
        unsafe {
            llev_decoded_operation_set_free(decoded);
            llev_owned_bytes_free(&mut encoded);
            llev_owned_bytes_free(&mut reencoded);
        }
    }
}

#[test]
fn byte_restrictions_survive_snapshot_and_bad_inputs_fail_without_publication() {
    let mut set = OperationSet::new();
    let mut restrictions = SubstitutionSet::new();
    restrictions.allow_byte(0xff, 0x80);
    set.add(OperationType::with_owned_applicability(
        1,
        1,
        1.0,
        OperationApplicability::Listed(restrictions),
        "byte",
    ));
    let expected = set.to_binary().unwrap();
    let mut decoded: *mut LlevDecodedOperationSet = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_operation_set_deserialize(
                1,
                expected.as_ptr(),
                expected.len(),
                &limits(),
                &mut decoded,
            )
        },
        LlevStatus::Ok
    );
    let mut pair = LlevSerializedRestrictionView::default();
    assert_eq!(
        unsafe { llev_decoded_operation_set_restriction_at(decoded, 0, 0, &mut pair) },
        LlevStatus::Ok
    );
    assert_eq!(
        (pair.kind, pair.source_byte, pair.target_byte),
        (1, 0xff, 0x80)
    );
    let mut reencoded = LlevOwnedBytes::default();
    assert_eq!(
        unsafe { llev_decoded_operation_set_serialize(decoded, 1, &limits(), &mut reencoded) },
        LlevStatus::Ok
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(reencoded.data, reencoded.len) },
        expected
    );
    unsafe {
        llev_decoded_operation_set_free(decoded);
        llev_owned_bytes_free(&mut reencoded);
    }

    let mut wrong_version = expected.clone();
    wrong_version[8] = 2;
    for bad in [wrong_version, expected[..expected.len() - 1].to_vec()] {
        assert_eq!(
            unsafe {
                llev_operation_set_deserialize(1, bad.as_ptr(), bad.len(), &limits(), &mut decoded)
            },
            LlevStatus::InvalidArgument
        );
        assert!(decoded.is_null());
    }
    let mut tiny = limits();
    tiny.max_payload_bytes = 8;
    assert_eq!(
        unsafe {
            llev_operation_set_deserialize(
                1,
                expected.as_ptr(),
                expected.len(),
                &tiny,
                &mut decoded,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert!(decoded.is_null());
}
