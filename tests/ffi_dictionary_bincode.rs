#![cfg(all(feature = "ffi", feature = "serialization"))]

use libdictenstein::{
    double_array_trie::DoubleArrayTrie,
    serialization::{BincodeSerializer, DictionarySerializer},
};
use liblevenshtein::ffi::{
    llev_decoded_bincode_term_at, llev_decoded_bincode_terms_free, llev_decoded_bincode_terms_len,
    llev_dictionary_bincode_deserialize, llev_dictionary_bincode_serialize, llev_owned_bytes_free,
    LlevDecodedBincodeTerms, LlevDictionaryBincodeLimits, LlevOwnedBytes, LlevStatus,
    LlevUtf8Slice,
};

fn limits() -> LlevDictionaryBincodeLimits {
    LlevDictionaryBincodeLimits {
        max_terms: 4,
        max_term_bytes: 16,
        max_total_term_bytes: 32,
        max_payload_bytes: 256,
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
        llev_dictionary_bincode_serialize(1, slices.as_ptr(), slices.len(), &limits(), &mut bytes)
    };
    assert_eq!(status, LlevStatus::Ok);
    let encoded = unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) }.to_vec();
    let native = DoubleArrayTrie::from_terms(source);
    let mut native_bytes = Vec::new();
    BincodeSerializer::serialize(&native, &mut native_bytes).unwrap();
    assert_eq!(encoded, native_bytes);

    let mut decoded: *mut LlevDecodedBincodeTerms = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            llev_dictionary_bincode_deserialize(
                1,
                encoded.as_ptr(),
                encoded.len(),
                &limits(),
                &mut decoded,
            )
        },
        LlevStatus::Ok
    );
    unsafe { llev_owned_bytes_free(&mut bytes) };
    let mut count = 0;
    assert_eq!(
        unsafe { llev_decoded_bincode_terms_len(decoded, &mut count) },
        LlevStatus::Ok
    );
    let mut roundtrip = Vec::new();
    for index in 0..count {
        let mut view = LlevUtf8Slice {
            data: std::ptr::null(),
            len: 0,
        };
        assert_eq!(
            unsafe { llev_decoded_bincode_term_at(decoded, index, &mut view) },
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
    unsafe { llev_decoded_bincode_terms_free(decoded) };
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
        unsafe { llev_dictionary_bincode_serialize(2, slices.as_ptr(), 1, &limits(), &mut bytes) },
        LlevStatus::InvalidArgument
    );
    assert!(bytes.data.is_null());
    let mut tiny = limits();
    tiny.max_payload_bytes = 8;
    let bounded =
        unsafe { llev_dictionary_bincode_serialize(1, slices.as_ptr(), 1, &tiny, &mut bytes) };
    assert_eq!(
        bounded,
        LlevStatus::LimitExceeded,
        "serialized length={}",
        bytes.len
    );
    assert!(bytes.data.is_null());

    let mut decoded: *mut LlevDecodedBincodeTerms = std::ptr::null_mut();
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
                llev_dictionary_bincode_deserialize(
                    1,
                    bad.as_ptr(),
                    bad.len(),
                    &limits(),
                    &mut decoded,
                )
            },
            LlevStatus::InvalidArgument
        );
        assert!(decoded.is_null());
    }
    let mut one_term_only = limits();
    one_term_only.max_term_bytes = 4;
    assert_eq!(
        unsafe {
            llev_dictionary_bincode_deserialize(
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
