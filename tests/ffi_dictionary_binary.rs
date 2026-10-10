#![cfg(all(feature = "ffi", feature = "serialization"))]

#[cfg(all(feature = "protobuf", feature = "compression"))]
use libdictenstein::serialization::{
    DatProtobufSerializer, GzipSerializer, OptimizedProtobufSerializer, ProtobufSerializer,
};
use libdictenstein::{
    double_array_trie::DoubleArrayTrie,
    serialization::{BincodeSerializer, DictionarySerializer},
};
use liblevenshtein::ffi::{
    llev_decoded_dictionary_term_at, llev_decoded_dictionary_terms_free,
    llev_decoded_dictionary_terms_len, llev_dictionary_deserialize, llev_dictionary_serialize,
    llev_owned_bytes_free, LlevDecodedDictionaryTerms, LlevDictionaryLimits, LlevOwnedBytes,
    LlevStatus, LlevUtf8Slice,
};

fn limits() -> LlevDictionaryLimits {
    LlevDictionaryLimits {
        max_terms: 4,
        max_term_bytes: 16,
        max_total_term_bytes: 32,
        max_payload_bytes: 256,
    }
}

#[cfg(all(feature = "protobuf", feature = "compression"))]
#[test]
fn rc5_native_dictionary_payloads_decode_with_current_bridge() {
    let fixtures: [&[u8]; 7] = [
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/dictionary-bincode-v1.bin"),
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/dictionary-protobuf-v1.bin"),
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/dictionary-protobuf-v2.bin"),
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/dictionary-gzip-bincode-v1.bin"),
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/dictionary-gzip-protobuf-v1.bin"),
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/dictionary-gzip-protobuf-v2.bin"),
        include_bytes!("../bindings/julia/Liblevenshtein/test/fixtures/serialization_rc5/dictionary-dat-v1.bin"),
    ];
    for (index, bytes) in fixtures.into_iter().enumerate() {
        let mut decoded: *mut LlevDecodedDictionaryTerms = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_dictionary_deserialize(
                    (index + 1) as u32,
                    bytes.as_ptr(),
                    bytes.len(),
                    &limits(),
                    &mut decoded,
                )
            },
            LlevStatus::Ok,
            "RC.5 format {} did not decode",
            index + 1
        );
        let mut count = 0;
        assert_eq!(
            unsafe { llev_decoded_dictionary_terms_len(decoded, &mut count) },
            LlevStatus::Ok
        );
        let mut terms = Vec::new();
        for term_index in 0..count {
            let mut view = LlevUtf8Slice {
                data: std::ptr::null(),
                len: 0,
            };
            assert_eq!(
                unsafe { llev_decoded_dictionary_term_at(decoded, term_index, &mut view) },
                LlevStatus::Ok
            );
            let bytes = if view.len == 0 {
                &[][..]
            } else {
                unsafe { std::slice::from_raw_parts(view.data, view.len) }
            };
            terms.push(std::str::from_utf8(bytes).unwrap().to_owned());
        }
        assert_eq!(terms, ["", "cab", "café"]);
        unsafe { llev_decoded_dictionary_terms_free(decoded) };
    }
}

#[test]
fn bincode_bridge_matches_native_wire_and_retains_decoded_terms() {
    let source = ["cab", "ab", "café"];
    let slices: Vec<_> = source
        .iter()
        .map(|term| LlevUtf8Slice {
            data: term.as_ptr(),
            len: term.len(),
        })
        .collect();
    let mut bytes = LlevOwnedBytes::default();
    let status = unsafe {
        llev_dictionary_serialize(1, slices.as_ptr(), slices.len(), &limits(), &mut bytes)
    };
    assert_eq!(status, LlevStatus::Ok);
    let encoded = unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) }.to_vec();
    let native = DoubleArrayTrie::from_terms(source);
    let mut native_bytes = Vec::new();
    BincodeSerializer::serialize(&native, &mut native_bytes).unwrap();
    assert_eq!(encoded, native_bytes);

    let mut decoded: *mut LlevDecodedDictionaryTerms = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_dictionary_deserialize(1, encoded.as_ptr(), encoded.len(), &limits(), &mut decoded)
        },
        LlevStatus::Ok
    );
    unsafe { llev_owned_bytes_free(&mut bytes) };
    let mut count = 0;
    assert_eq!(
        unsafe { llev_decoded_dictionary_terms_len(decoded, &mut count) },
        LlevStatus::Ok
    );
    let mut roundtrip = Vec::new();
    for index in 0..count {
        let mut view = LlevUtf8Slice {
            data: std::ptr::null(),
            len: 0,
        };
        assert_eq!(
            unsafe { llev_decoded_dictionary_term_at(decoded, index, &mut view) },
            LlevStatus::Ok
        );
        roundtrip.push(
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(view.data, view.len))
            }
            .to_owned(),
        );
    }
    assert_eq!(roundtrip, ["ab", "cab", "café"]);
    unsafe { llev_decoded_dictionary_terms_free(decoded) };
}

#[test]
fn bincode_bridge_rejects_malformed_and_over_budget_inputs_without_outputs() {
    let source = ["hello"];
    let slices = [LlevUtf8Slice {
        data: source[0].as_ptr(),
        len: source[0].len(),
    }];
    let mut bytes = LlevOwnedBytes::default();
    assert_eq!(
        unsafe { llev_dictionary_serialize(99, slices.as_ptr(), 1, &limits(), &mut bytes) },
        LlevStatus::InvalidArgument
    );
    assert!(bytes.data.is_null());
    let mut tiny = limits();
    tiny.max_payload_bytes = 8;
    let bounded = unsafe { llev_dictionary_serialize(1, slices.as_ptr(), 1, &tiny, &mut bytes) };
    assert_eq!(
        bounded,
        LlevStatus::LimitExceeded,
        "serialized length={}",
        bytes.len
    );
    assert!(bytes.data.is_null());

    let mut decoded: *mut LlevDecodedDictionaryTerms = std::ptr::null_mut();
    let mut payload = Vec::new();
    payload.extend_from_slice(&1_u64.to_le_bytes());
    payload.extend_from_slice(&5_u64.to_le_bytes());
    payload.extend_from_slice(b"hello");
    for bad in [
        payload[..payload.len() - 1].to_vec(),
        [payload.as_slice(), b"!"].concat(),
        {
            let mut invalid = payload.clone();
            *invalid.last_mut().unwrap() = 0xff;
            invalid
        },
    ] {
        assert_eq!(
            unsafe {
                llev_dictionary_deserialize(1, bad.as_ptr(), bad.len(), &limits(), &mut decoded)
            },
            LlevStatus::InvalidArgument
        );
        assert!(decoded.is_null());
    }
    let mut one_term_only = limits();
    one_term_only.max_term_bytes = 4;
    assert_eq!(
        unsafe {
            llev_dictionary_deserialize(
                1,
                payload.as_ptr(),
                payload.len(),
                &one_term_only,
                &mut decoded,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert!(decoded.is_null());
}

#[cfg(all(feature = "protobuf", feature = "compression"))]
#[test]
fn all_general_dictionary_formats_match_native_bytes_and_round_trip() {
    let source = ["", "cab", "ab", "café", "cab"];
    let slices: Vec<_> = source
        .iter()
        .map(|term| LlevUtf8Slice {
            data: term.as_ptr(),
            len: term.len(),
        })
        .collect();
    let dictionary = DoubleArrayTrie::from_terms(source);
    let mut ceilings = limits();
    ceilings.max_terms = source.len();
    ceilings.max_payload_bytes = 2048;
    for format_id in 1..=7 {
        let mut bytes = LlevOwnedBytes::default();
        assert_eq!(
            unsafe {
                llev_dictionary_serialize(
                    format_id,
                    slices.as_ptr(),
                    slices.len(),
                    &ceilings,
                    &mut bytes,
                )
            },
            LlevStatus::Ok,
            "format {format_id}"
        );
        let encoded = unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) }.to_vec();
        let mut native = Vec::new();
        match format_id {
            1 => BincodeSerializer::serialize(&dictionary, &mut native),
            2 => ProtobufSerializer::serialize(&dictionary, &mut native),
            3 => OptimizedProtobufSerializer::serialize(&dictionary, &mut native),
            4 => GzipSerializer::<BincodeSerializer>::serialize(&dictionary, &mut native),
            5 => GzipSerializer::<ProtobufSerializer>::serialize(&dictionary, &mut native),
            6 => GzipSerializer::<OptimizedProtobufSerializer>::serialize(&dictionary, &mut native),
            7 => DatProtobufSerializer::serialize_dat(&dictionary, &mut native),
            _ => unreachable!(),
        }
        .unwrap();
        assert_eq!(encoded, native, "native byte parity for format {format_id}");

        let mut decoded = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_dictionary_deserialize(
                    format_id,
                    encoded.as_ptr(),
                    encoded.len(),
                    &ceilings,
                    &mut decoded,
                )
            },
            LlevStatus::Ok,
            "decode format {format_id}"
        );
        let mut len = 0;
        assert_eq!(
            unsafe { llev_decoded_dictionary_terms_len(decoded, &mut len) },
            LlevStatus::Ok
        );
        let mut terms = Vec::new();
        for index in 0..len {
            let mut view = LlevUtf8Slice {
                data: std::ptr::null(),
                len: 0,
            };
            assert_eq!(
                unsafe { llev_decoded_dictionary_term_at(decoded, index, &mut view) },
                LlevStatus::Ok
            );
            let bytes = if view.len == 0 {
                &[][..]
            } else {
                unsafe { std::slice::from_raw_parts(view.data, view.len) }
            };
            terms.push(std::str::from_utf8(bytes).unwrap().to_owned());
        }
        assert_eq!(terms, ["", "ab", "cab", "café"], "format {format_id}");
        unsafe { llev_decoded_dictionary_terms_free(decoded) };

        if (4..=6).contains(&format_id) {
            let mut corrupt = encoded.clone();
            corrupt.push(0xff);
            let mut output = std::ptr::null_mut();
            assert_eq!(
                unsafe {
                    llev_dictionary_deserialize(
                        format_id,
                        corrupt.as_ptr(),
                        corrupt.len(),
                        &ceilings,
                        &mut output,
                    )
                },
                LlevStatus::InvalidArgument
            );
            assert!(output.is_null());
        }
        unsafe { llev_owned_bytes_free(&mut bytes) };
    }
}

#[cfg(feature = "protobuf")]
#[test]
fn protobuf_graph_preflight_rejects_cycles_and_inconsistent_counts() {
    // V1: nodes=[0], final=[0], edge=(0,'a',0), size=1.
    let cyclic = [
        0x0a, 0x01, 0x00, 0x12, 0x01, 0x00, 0x1a, 0x02, 0x10, b'a', 0x28, 0x01,
    ];
    let mut decoded = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_dictionary_deserialize(2, cyclic.as_ptr(), cyclic.len(), &limits(), &mut decoded)
        },
        LlevStatus::InvalidArgument
    );
    assert!(decoded.is_null());

    // V1: root has two distinct final children, but declares one term.
    let count_mismatch = [
        0x0a, 0x03, 0x00, 0x01, 0x02, 0x12, 0x02, 0x01, 0x02, 0x1a, 0x04, 0x10, b'a', 0x18, 0x01,
        0x1a, 0x04, 0x10, b'b', 0x18, 0x02, 0x28, 0x01,
    ];
    assert_eq!(
        unsafe {
            llev_dictionary_deserialize(
                2,
                count_mismatch.as_ptr(),
                count_mismatch.len(),
                &limits(),
                &mut decoded,
            )
        },
        LlevStatus::InvalidArgument
    );
    assert!(decoded.is_null());

    // DAT protobuf: LDT1 followed by a declared three-byte term with no bytes.
    let truncated_dat = [
        0x22, 0x08, b'L', b'D', b'T', b'1', 0x03, 0x00, 0x00, 0x00, 0x30, 0x01,
    ];
    assert_eq!(
        unsafe {
            llev_dictionary_deserialize(
                7,
                truncated_dat.as_ptr(),
                truncated_dat.len(),
                &limits(),
                &mut decoded,
            )
        },
        LlevStatus::InvalidArgument
    );
    assert!(decoded.is_null());
}
