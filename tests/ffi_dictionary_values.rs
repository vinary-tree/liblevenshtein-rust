#![cfg(all(feature = "ffi", feature = "serialization"))]

use libdictenstein::{
    double_array_trie::{DoubleArrayTrie, DoubleArrayTrieChar},
    serialization::BincodeSerializer,
};
use liblevenshtein::ffi::{
    llev_decoded_value_entries_free, llev_decoded_value_entries_len, llev_decoded_value_entry_at,
    llev_owned_bytes_free, llev_valued_dictionary_deserialize, llev_valued_dictionary_serialize,
    LlevDecodedValueEntries, LlevOwnedBytes, LlevStatus, LlevValueEntryInput, LlevValueEntryView,
    LlevValueLimits,
};

fn limits() -> LlevValueLimits {
    LlevValueLimits {
        max_entries: 4,
        max_term_bytes: 16,
        max_total_term_bytes: 32,
        max_value_bytes: 16,
        max_total_value_bytes: 32,
        max_payload_bytes: 256,
    }
}

#[test]
fn valued_bincode_matches_native_byte_and_unicode_serializers() {
    let terms = ["cab", "café"];
    let values = [vec![0, 255], Vec::new()];
    for unit_domain in 1..=2 {
        for value_kind in 1..=2 {
            let inputs = [
                LlevValueEntryInput {
                    term_data: terms[0].as_ptr(),
                    term_len: terms[0].len(),
                    value_data: values[0].as_ptr(),
                    value_len: values[0].len(),
                    value_u64: 7,
                },
                LlevValueEntryInput {
                    term_data: terms[1].as_ptr(),
                    term_len: terms[1].len(),
                    value_data: values[1].as_ptr(),
                    value_len: values[1].len(),
                    value_u64: 9,
                },
            ];
            let mut bytes = LlevOwnedBytes::default();
            assert_eq!(
                unsafe {
                    llev_valued_dictionary_serialize(
                        unit_domain,
                        value_kind,
                        inputs.as_ptr(),
                        inputs.len(),
                        &limits(),
                        &mut bytes,
                    )
                },
                LlevStatus::Ok
            );
            let encoded = unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) };
            let mut expected = Vec::new();
            match (unit_domain, value_kind) {
                (1, 1) => BincodeSerializer::serialize_with_values(
                    &DoubleArrayTrie::<u64>::from_terms_with_values([("cab", 7), ("café", 9)]),
                    &mut expected,
                ),
                (2, 1) => BincodeSerializer::serialize_with_values_char(
                    &DoubleArrayTrieChar::<u64>::from_terms_with_values([("cab", 7), ("café", 9)]),
                    &mut expected,
                ),
                (1, 2) => BincodeSerializer::serialize_with_values(
                    &DoubleArrayTrie::<Vec<u8>>::from_terms_with_values([
                        ("cab", vec![0, 255]),
                        ("café", vec![]),
                    ]),
                    &mut expected,
                ),
                (2, 2) => BincodeSerializer::serialize_with_values_char(
                    &DoubleArrayTrieChar::<Vec<u8>>::from_terms_with_values([
                        ("cab", vec![0, 255]),
                        ("café", vec![]),
                    ]),
                    &mut expected,
                ),
                _ => unreachable!(),
            }
            .unwrap();
            assert_eq!(encoded, expected);

            let mut decoded: *mut LlevDecodedValueEntries = std::ptr::null_mut();
            assert_eq!(
                unsafe {
                    llev_valued_dictionary_deserialize(
                        unit_domain,
                        value_kind,
                        encoded.as_ptr(),
                        encoded.len(),
                        &limits(),
                        &mut decoded,
                    )
                },
                LlevStatus::Ok
            );
            let mut count = 0;
            assert_eq!(
                unsafe { llev_decoded_value_entries_len(decoded, &mut count) },
                LlevStatus::Ok
            );
            assert_eq!(count, 2);
            for index in 0..count {
                let mut view = LlevValueEntryView::default();
                assert_eq!(
                    unsafe { llev_decoded_value_entry_at(decoded, index, &mut view) },
                    LlevStatus::Ok
                );
                assert_eq!(view.value_kind, value_kind);
                assert_eq!(
                    unsafe { std::slice::from_raw_parts(view.term_data, view.term_len) },
                    terms[index].as_bytes()
                );
                if value_kind == 1 {
                    assert_eq!(view.value_u64, [7, 9][index]);
                } else {
                    let actual = if view.value_len == 0 {
                        &[][..]
                    } else {
                        unsafe { std::slice::from_raw_parts(view.value_data, view.value_len) }
                    };
                    assert_eq!(actual, values[index]);
                }
            }
            unsafe {
                llev_decoded_value_entries_free(decoded);
                llev_owned_bytes_free(&mut bytes);
            }
        }
    }
}

#[test]
fn valued_bincode_rejects_corruption_and_resource_overruns() {
    let dictionary =
        DoubleArrayTrie::<Vec<u8>>::from_terms_with_values([("a", vec![0, 255]), ("b", vec![])]);
    let mut valid = Vec::new();
    BincodeSerializer::serialize_with_values(&dictionary, &mut valid).unwrap();
    let mut decoded: *mut LlevDecodedValueEntries = std::ptr::null_mut();
    for bad in [
        valid[..valid.len() - 1].to_vec(),
        [valid.as_slice(), b"!"].concat(),
    ] {
        assert_eq!(
            unsafe {
                llev_valued_dictionary_deserialize(
                    1,
                    2,
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
    let mut oversized_term = valid.clone();
    oversized_term[8] = 0xff;
    assert_eq!(
        unsafe {
            llev_valued_dictionary_deserialize(
                1,
                2,
                oversized_term.as_ptr(),
                oversized_term.len(),
                &limits(),
                &mut decoded,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert!(decoded.is_null());
    let mut too_small = limits();
    too_small.max_value_bytes = 1;
    too_small.max_total_value_bytes = 1;
    assert_eq!(
        unsafe {
            llev_valued_dictionary_deserialize(
                1,
                2,
                valid.as_ptr(),
                valid.len(),
                &too_small,
                &mut decoded,
            )
        },
        LlevStatus::LimitExceeded
    );
    assert!(decoded.is_null());
}
