#![cfg(all(feature = "ffi", feature = "serialization"))]

#[cfg(feature = "protobuf")]
use libdictenstein::serialization::SuffixAutomatonProtobufSerializer;
use libdictenstein::{serialization::BincodeSerializer, suffix_automaton::SuffixAutomaton};
use liblevenshtein::ffi::{
    llev_decoded_dictionary_term_at, llev_decoded_dictionary_terms_free,
    llev_decoded_dictionary_terms_len, llev_owned_bytes_free, llev_suffix_source_deserialize,
    llev_suffix_source_serialize, LlevDecodedDictionaryTerms, LlevDictionaryLimits, LlevOwnedBytes,
    LlevStatus, LlevUtf8Slice,
};

fn limits() -> LlevDictionaryLimits {
    LlevDictionaryLimits {
        max_terms: 4,
        max_term_bytes: 16,
        max_total_term_bytes: 32,
        max_payload_bytes: 512,
    }
}

#[test]
fn suffix_source_formats_match_native_bytes_and_preserve_order() {
    let source = ["cab", "café", "cab"];
    let automaton = SuffixAutomaton::from_texts(source);
    let slices: Vec<_> = source
        .iter()
        .map(|text| LlevUtf8Slice {
            data: text.as_ptr(),
            len: text.len(),
        })
        .collect();
    for format_id in 1..=if cfg!(feature = "protobuf") { 2 } else { 1 } {
        let mut bytes = LlevOwnedBytes::default();
        assert_eq!(
            unsafe {
                llev_suffix_source_serialize(
                    format_id,
                    slices.as_ptr(),
                    slices.len(),
                    &limits(),
                    &mut bytes,
                )
            },
            LlevStatus::Ok
        );
        let encoded = unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) }.to_vec();
        let mut native = Vec::new();
        match format_id {
            1 => BincodeSerializer::serialize_suffix_automaton(&automaton, &mut native),
            #[cfg(feature = "protobuf")]
            2 => SuffixAutomatonProtobufSerializer::serialize_suffix_automaton(
                &automaton,
                &mut native,
            ),
            _ => unreachable!(),
        }
        .unwrap();
        assert_eq!(encoded, native, "format {format_id}");

        let mut decoded: *mut LlevDecodedDictionaryTerms = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                llev_suffix_source_deserialize(
                    format_id,
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
            unsafe { llev_decoded_dictionary_terms_len(decoded, &mut count) },
            LlevStatus::Ok
        );
        let mut texts = Vec::new();
        for index in 0..count {
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
            texts.push(std::str::from_utf8(bytes).unwrap().to_owned());
        }
        assert_eq!(texts, automaton.source_texts());
        unsafe { llev_decoded_dictionary_terms_free(decoded) };
    }
}

#[test]
fn suffix_source_preflight_rejects_corruption_and_wrong_versions() {
    let mut decoded = std::ptr::null_mut();
    let truncated = [1_u8, 0, 0, 0];
    assert_eq!(
        unsafe {
            llev_suffix_source_deserialize(
                1,
                truncated.as_ptr(),
                truncated.len(),
                &limits(),
                &mut decoded,
            )
        },
        LlevStatus::InvalidArgument
    );
    assert!(decoded.is_null());
    assert_eq!(
        unsafe {
            llev_suffix_source_deserialize(
                99,
                truncated.as_ptr(),
                truncated.len(),
                &limits(),
                &mut decoded,
            )
        },
        LlevStatus::InvalidArgument
    );
    assert!(decoded.is_null());

    #[cfg(feature = "protobuf")]
    {
        // One UTF-8 source text with a declared string count of two.
        let mismatched = [0x0a, 0x01, b'a', 0x10, 0x02];
        assert_eq!(
            unsafe {
                llev_suffix_source_deserialize(
                    2,
                    mismatched.as_ptr(),
                    mismatched.len(),
                    &limits(),
                    &mut decoded,
                )
            },
            LlevStatus::InvalidArgument
        );
        assert!(decoded.is_null());
    }
}
