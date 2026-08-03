#![cfg(feature = "compression")]

use flate2::write::GzEncoder;
use flate2::Compression;
use liblevenshtein::transducer::{
    OperationSet, OperationSetBinaryLimits, OperationSetBuilder, OperationSetGzipError,
    OperationType, SubstitutionSet,
};
use proptest::prelude::*;
use std::io::Write;

fn gzip(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(bytes).expect("test gzip writes");
    encoder.finish().expect("test gzip finishes")
}

fn restricted_set(pairs: &[(u8, u8)]) -> OperationSet {
    let mut restriction = SubstitutionSet::new();
    for &(source, target) in pairs {
        restriction.allow_byte(source, target);
    }
    OperationSetBuilder::new()
        .with_standard_ops()
        .with_operation(OperationType::with_owned_restriction(
            1,
            1,
            0.5,
            restriction,
            "runtime_bytes".to_owned(),
        ))
        .build()
}

#[test]
fn bincode_gzip_is_an_exact_outer_wrapper() {
    let operations = restricted_set(&[(0xff, 0x00), (b'a', b'b')]);
    let raw = operations.to_binary().expect("bincode encodes");
    let compressed = operations.to_binary_gzip().expect("gzip encodes");

    assert_eq!(gzip(&raw), compressed);
    assert_eq!(
        OperationSet::from_binary_gzip(&compressed).expect("gzip round trip"),
        operations
    );
}

#[cfg(feature = "protobuf")]
#[test]
fn protobuf_gzip_is_an_exact_outer_wrapper() {
    let operations = restricted_set(&[(0xff, 0x00), (b'a', b'b')]);
    let raw = operations.to_protobuf().expect("protobuf encodes");
    let compressed = operations.to_protobuf_gzip().expect("gzip encodes");

    assert_eq!(gzip(&raw), compressed);
    assert_eq!(
        OperationSet::from_protobuf_gzip(&compressed).expect("gzip round trip"),
        operations
    );
}

#[test]
fn gzip_rejects_trailing_members_junk_and_checksum_corruption() {
    let operations = OperationSet::standard();
    let compressed = operations.to_binary_gzip().expect("gzip encodes");

    let mut trailing_junk = compressed.clone();
    trailing_junk.extend_from_slice(b"junk");
    assert!(matches!(
        OperationSet::from_binary_gzip(&trailing_junk),
        Err(OperationSetGzipError::TrailingCompressedData { .. })
    ));

    let mut concatenated = compressed.clone();
    concatenated.extend_from_slice(&compressed);
    assert!(matches!(
        OperationSet::from_binary_gzip(&concatenated),
        Err(OperationSetGzipError::TrailingCompressedData { .. })
    ));

    let mut corrupt = compressed;
    let last = corrupt.last_mut().expect("gzip has checksum trailer");
    *last ^= 0xff;
    assert!(matches!(
        OperationSet::from_binary_gzip(&corrupt),
        Err(OperationSetGzipError::Gzip(_))
    ));
}

#[test]
fn gzip_enforces_decompressed_limit_before_inner_decode() {
    let compressed = gzip(&[0_u8; 22]);
    let limits = OperationSetBinaryLimits {
        max_payload_bytes: 1,
        ..OperationSetBinaryLimits::default()
    };
    assert!(matches!(
        OperationSet::from_binary_gzip_with_limits(&compressed, limits),
        Err(OperationSetGzipError::DecompressedPayloadTooLarge {
            observed: 22,
            limit: 21,
        })
    ));
}

#[test]
fn gzip_reduces_a_repetitive_operation_table_but_remains_optional() {
    let mut operations = OperationSet::new();
    for index in 0..256 {
        operations.add(OperationType::new_owned(
            1,
            1,
            1.0,
            format!("repetitive_operation_family_{index:03}"),
        ));
    }

    let raw = operations.to_binary().expect("bincode encodes");
    let compressed = operations.to_binary_gzip().expect("gzip encodes");
    assert!(
        compressed.len() < raw.len(),
        "synthetic repetitive data should demonstrate gzip's possible size benefit"
    );
    assert_eq!(
        OperationSet::from_binary_gzip(&compressed).expect("gzip round trip"),
        operations
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn gzip_roundtrip_corresponds_to_the_uncompressed_decoder(
        pairs in prop::collection::btree_set((any::<u8>(), any::<u8>()), 0..64),
    ) {
        let pairs = pairs.into_iter().collect::<Vec<_>>();
        let operations = restricted_set(&pairs);
        let compressed = operations.to_binary_gzip().expect("generated set compresses");
        let restored = OperationSet::from_binary_gzip(&compressed)
            .expect("generated gzip decodes");
        prop_assert_eq!(restored, operations);
    }

    #[test]
    fn arbitrary_gzip_input_never_panics_or_bypasses_inner_limits(
        bytes in prop::collection::vec(any::<u8>(), 0..=512),
        payload_limit in 0_usize..=512,
        operation_limit in 0_usize..=16,
    ) {
        let limits = OperationSetBinaryLimits {
            max_payload_bytes: payload_limit,
            max_operations: operation_limit,
            max_operation_name_bytes: 64,
            max_restriction_pairs_per_operation: 64,
            max_total_restriction_pairs: 64,
            max_restriction_text_bytes: 256,
        };
        let decoded = std::panic::catch_unwind(|| {
            OperationSet::from_binary_gzip_with_limits(&bytes, limits)
        });
        prop_assert!(decoded.is_ok(), "gzip decoder panicked on {bytes:?}");
        if let Ok(Ok(operation_set)) = decoded {
            prop_assert!(operation_set.len() <= operation_limit);
            prop_assert!(operation_set.validate().is_ok());
        }
    }
}
