#![cfg(feature = "serialization")]

use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
use liblevenshtein::transducer::{
    OperationApplicability, OperationSet, OperationSetBinaryError, OperationSetBinaryLimits,
    OperationSetBuilder, OperationType, SubstitutionSet, MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES,
    OPERATION_SET_BINARY_MAGIC, OPERATION_SET_BINARY_VERSION,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;
use serde::Serialize;

fn operation_set_config() -> ProptestConfig {
    let mut config = ProptestConfig::with_failure_persistence(FileFailurePersistence::Direct(
        "tests/operation_set_serialization.proptest-regressions",
    ));
    config.cases = 512;
    config
}

fn rich_operation_set(restrictions: SubstitutionSet) -> OperationSet {
    OperationSetBuilder::new()
        .with_standard_ops()
        .with_transposition()
        .with_operation(OperationType::with_restriction(
            2,
            1,
            0.125,
            restrictions,
            "runtime_digraph",
        ))
        .build()
}

fn envelope(payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(20 + payload.len());
    bytes.extend_from_slice(&OPERATION_SET_BINARY_MAGIC);
    bytes.extend_from_slice(&OPERATION_SET_BINARY_VERSION.to_le_bytes());
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

#[derive(Serialize)]
struct RawOperation {
    consume_x: u64,
    consume_y: u64,
    weight: f64,
    applicability: RawApplicability,
    name: String,
}

#[derive(Serialize)]
enum RawApplicability {
    Any,
    Equal,
    AdjacentTranspose,
    Listed(Vec<RawSubstitutionPair>),
}

#[derive(Serialize)]
enum RawSubstitutionPair {
    Bytes { source: u8, target: u8 },
    Strings { source: Box<str>, target: Box<str> },
}

#[test]
fn binary_roundtrip_preserves_full_runtime_configuration_and_execution() {
    let mut restrictions = SubstitutionSet::new();
    restrictions.allow_str("ph", "f");
    restrictions.allow_str("αβ", "γ");
    let original = rich_operation_set(restrictions);

    let encoded = original.to_binary().expect("valid operation set encodes");
    let restored = OperationSet::from_binary(&encoded).expect("binary envelope decodes");
    assert_eq!(restored, original);
    assert_eq!(restored.operations()[5].name(), "runtime_digraph");
    assert!(restored.operations()[5].can_apply_str("ph", "f"));
    assert!(restored.operations()[5].can_apply_str("αβ", "γ"));

    for budget in 0..=3 {
        let before = GeneralizedAutomaton::try_with_operations(budget, original.clone())
            .expect("original operation set is executable");
        let after = GeneralizedAutomaton::try_with_operations(budget, restored.clone())
            .expect("restored operation set is executable");
        for (source, target) in [
            ("phone", "fone"),
            ("αβx", "γx"),
            ("test", "tset"),
            ("kitten", "sitting"),
            ("", "abc"),
        ] {
            assert_eq!(
                before.scaled_distance(source, target),
                after.scaled_distance(source, target),
                "execution changed for {source:?} -> {target:?} at budget {budget}"
            );
        }
    }
}

#[test]
fn canonical_restriction_order_produces_deterministic_bytes() {
    let mut forward = SubstitutionSet::new();
    forward.allow_str("ph", "f");
    forward.allow_str("ch", "k");
    forward.allow_str("th", "t");
    forward.allow_str("sh", "s");
    forward.allow_str("αβ", "γ");

    let mut reverse = SubstitutionSet::new();
    reverse.allow_str("αβ", "γ");
    reverse.allow_str("sh", "s");
    reverse.allow_str("th", "t");
    reverse.allow_str("ch", "k");
    reverse.allow_str("ph", "f");

    assert_eq!(
        rich_operation_set(forward)
            .to_binary()
            .expect("first set encodes"),
        rich_operation_set(reverse)
            .to_binary()
            .expect("second set encodes")
    );
}

#[test]
fn private_wire_preserves_the_version_one_layout() {
    let mut restriction = SubstitutionSet::new();
    restriction.allow_byte(b'a', b'b');
    restriction.allow_str("α", "β");
    let operation_set = OperationSetBuilder::new()
        .with_operation(OperationType::new(1, 1, 0.5, "replace"))
        .with_operation(OperationType::with_applicability(
            1,
            1,
            0.0,
            OperationApplicability::Equal,
            "equal",
        ))
        .with_operation(OperationType::with_applicability(
            2,
            2,
            1.0,
            OperationApplicability::AdjacentTranspose,
            "transpose",
        ))
        .with_operation(OperationType::with_restriction(
            1,
            1,
            0.25,
            restriction,
            "listed",
        ))
        .build();
    let expected_payload = bincode::serde::encode_to_vec(
        vec![
            RawOperation {
                consume_x: 1,
                consume_y: 1,
                weight: 0.5,
                applicability: RawApplicability::Any,
                name: "replace".to_owned(),
            },
            RawOperation {
                consume_x: 1,
                consume_y: 1,
                weight: 0.0,
                applicability: RawApplicability::Equal,
                name: "equal".to_owned(),
            },
            RawOperation {
                consume_x: 2,
                consume_y: 2,
                weight: 1.0,
                applicability: RawApplicability::AdjacentTranspose,
                name: "transpose".to_owned(),
            },
            RawOperation {
                consume_x: 1,
                consume_y: 1,
                weight: 0.25,
                applicability: RawApplicability::Listed(vec![
                    RawSubstitutionPair::Bytes {
                        source: b'a',
                        target: b'b',
                    },
                    RawSubstitutionPair::Strings {
                        source: "α".into(),
                        target: "β".into(),
                    },
                ]),
                name: "listed".to_owned(),
            },
        ],
        bincode::config::legacy(),
    )
    .expect("version-one fixture encodes");

    assert_eq!(
        operation_set.to_binary().expect("operation set encodes"),
        envelope(&expected_payload)
    );
}

#[test]
fn binary_roundtrip_preserves_f64_bits_and_non_utf8_restrictions() {
    let weights = [f64::MIN_POSITIVE, 0.1, 1.0 / 3.0, f64::MAX];
    let mut builder = OperationSetBuilder::new().with_standard_ops();
    for (index, weight) in weights.into_iter().enumerate() {
        builder = builder.with_operation(OperationType::new_owned(
            1,
            1,
            weight,
            format!("weighted_{index}"),
        ));
    }

    let mut restriction = SubstitutionSet::new();
    restriction.allow_byte(0xff, 0x00);
    builder = builder.with_operation(OperationType::with_restriction(
        1,
        1,
        0.25,
        restriction,
        "raw_bytes",
    ));

    let operations = builder.build();
    let bytes = operations.to_binary().expect("binary model encodes");
    let restored = OperationSet::from_binary(&bytes).expect("binary model decodes");

    for (&expected, operation) in weights.iter().zip(&restored.operations()[4..8]) {
        assert_eq!(operation.weight().to_bits(), expected.to_bits());
    }
    assert!(restored.operations()[8].can_apply(&[0xff], &[0x00]));
}

#[test]
fn envelope_rejects_corruption_versions_lengths_and_resource_claims() {
    let encoded = OperationSet::standard()
        .to_binary()
        .expect("standard set encodes");

    assert!(matches!(
        OperationSet::from_binary(&encoded[..10]),
        Err(OperationSetBinaryError::TruncatedHeader { .. })
    ));

    let mut bad_magic = encoded.clone();
    bad_magic[0] ^= 0xff;
    assert!(matches!(
        OperationSet::from_binary(&bad_magic),
        Err(OperationSetBinaryError::InvalidMagic)
    ));

    let mut bad_version = encoded.clone();
    bad_version[8..10].copy_from_slice(&2_u16.to_le_bytes());
    assert!(matches!(
        OperationSet::from_binary(&bad_version),
        Err(OperationSetBinaryError::UnsupportedVersion(2))
    ));

    let mut bad_flags = encoded.clone();
    bad_flags[10..12].copy_from_slice(&1_u16.to_le_bytes());
    assert!(matches!(
        OperationSet::from_binary(&bad_flags),
        Err(OperationSetBinaryError::UnsupportedFlags(1))
    ));

    let mut oversized = encoded.clone();
    oversized[12..20]
        .copy_from_slice(&((MAX_OPERATION_SET_BINARY_PAYLOAD_BYTES as u64) + 1).to_le_bytes());
    assert!(matches!(
        OperationSet::from_binary(&oversized),
        Err(OperationSetBinaryError::PayloadTooLarge { .. })
    ));

    let mut appended = encoded;
    appended.push(0);
    assert!(matches!(
        OperationSet::from_binary(&appended),
        Err(OperationSetBinaryError::LengthMismatch { .. })
    ));
}

#[test]
fn caller_selected_limits_reject_each_bounded_resource() {
    let encoded = OperationSet::standard()
        .to_binary()
        .expect("standard encodes");
    let limits = OperationSetBinaryLimits {
        max_operations: 3,
        ..OperationSetBinaryLimits::default()
    };
    assert!(matches!(
        OperationSet::from_binary_with_limits(&encoded, limits),
        Err(OperationSetBinaryError::ResourceLimit {
            resource: "operation count",
            ..
        })
    ));

    let mut restrictions = SubstitutionSet::new();
    restrictions.allow_str("ph", "f");
    restrictions.allow_str("αβ", "γ");
    let encoded = rich_operation_set(restrictions)
        .to_binary()
        .expect("rich set encodes");

    let limits = OperationSetBinaryLimits {
        max_operation_name_bytes: 4,
        ..OperationSetBinaryLimits::default()
    };
    assert!(matches!(
        OperationSet::from_binary_with_limits(&encoded, limits),
        Err(OperationSetBinaryError::ResourceLimit {
            resource: "operation-name bytes",
            ..
        })
    ));

    let limits = OperationSetBinaryLimits {
        max_restriction_pairs_per_operation: 1,
        ..OperationSetBinaryLimits::default()
    };
    assert!(matches!(
        OperationSet::from_binary_with_limits(&encoded, limits),
        Err(OperationSetBinaryError::ResourceLimit {
            resource: "restriction pairs per operation",
            ..
        })
    ));

    let limits = OperationSetBinaryLimits {
        max_total_restriction_pairs: 1,
        ..OperationSetBinaryLimits::default()
    };
    assert!(matches!(
        OperationSet::from_binary_with_limits(&encoded, limits),
        Err(OperationSetBinaryError::ResourceLimit {
            resource: "total restriction pairs",
            ..
        })
    ));

    let limits = OperationSetBinaryLimits {
        max_restriction_text_bytes: 1,
        ..OperationSetBinaryLimits::default()
    };
    assert!(matches!(
        OperationSet::from_binary_with_limits(&encoded, limits),
        Err(OperationSetBinaryError::ResourceLimit {
            resource: "restriction text bytes",
            ..
        })
    ));

    let limits = OperationSetBinaryLimits {
        max_payload_bytes: encoded.len() - 21,
        ..OperationSetBinaryLimits::default()
    };
    assert!(matches!(
        OperationSet::from_binary_with_limits(&encoded, limits),
        Err(OperationSetBinaryError::PayloadTooLarge { .. })
    ));
}

#[test]
fn declared_operation_count_is_rejected_before_vector_allocation() {
    let payload = ((liblevenshtein::transducer::MAX_OPERATION_SET_TOTAL_CONSUMPTION as u64) + 1)
        .to_le_bytes();
    assert!(matches!(
        OperationSet::from_binary(&envelope(&payload)),
        Err(OperationSetBinaryError::Decode(_))
    ));
}

#[test]
fn decoder_rejects_semantically_invalid_raw_payloads() {
    for raw in [
        RawOperation {
            consume_x: 0,
            consume_y: 0,
            weight: 1.0,
            applicability: RawApplicability::Any,
            name: "stuck".to_owned(),
        },
        RawOperation {
            consume_x: 1,
            consume_y: 1,
            weight: f64::INFINITY,
            applicability: RawApplicability::Any,
            name: "infinite".to_owned(),
        },
        RawOperation {
            consume_x: 1,
            consume_y: 1,
            weight: 1.0,
            applicability: RawApplicability::Any,
            name: String::new(),
        },
    ] {
        let payload = bincode::serde::encode_to_vec(vec![raw], bincode::config::legacy())
            .expect("raw test wire value encodes");
        assert!(matches!(
            OperationSet::from_binary(&envelope(&payload)),
            Err(OperationSetBinaryError::Decode(_))
        ));
    }

    let payload = bincode::serde::encode_to_vec(
        vec![RawOperation {
            consume_x: 1,
            consume_y: 1,
            weight: 1.0,
            applicability: RawApplicability::Listed(vec![RawSubstitutionPair::Strings {
                source: "ph".into(),
                target: "f".into(),
            }]),
            name: "wrong_dimension".to_owned(),
        }],
        bincode::config::legacy(),
    )
    .expect("raw test wire value encodes");
    assert!(matches!(
        OperationSet::from_binary(&envelope(&payload)),
        Err(OperationSetBinaryError::Decode(_))
    ));
}

proptest! {
    #![proptest_config(operation_set_config())]

    #[test]
    fn binary_roundtrip_is_deterministic_and_execution_equivalent(
        pairs in prop::collection::btree_set((any::<u8>(), any::<u8>()), 0..64),
        source in "[a-z]{0,12}",
        target in "[a-z]{0,12}",
        budget in 0_u8..=4,
    ) {
        let mut restrictions = SubstitutionSet::new();
        for &(left, right) in &pairs {
            restrictions.allow_byte(left, right);
        }
        let operations = OperationSetBuilder::new()
            .with_standard_ops()
            .with_operation(OperationType::with_restriction(
                1,
                1,
                0.5,
                restrictions,
                "byte_equivalence",
            ))
            .build();
        let bytes = operations.to_binary().expect("generated set encodes");
        let restored = OperationSet::from_binary(&bytes).expect("generated set decodes");

        prop_assert_eq!(&restored, &operations);
        prop_assert_eq!(restored.to_binary().expect("restored set re-encodes"), bytes);

        let before = GeneralizedAutomaton::try_with_operations(budget, operations)
            .expect("generated source automaton builds");
        let after = GeneralizedAutomaton::try_with_operations(budget, restored)
            .expect("generated restored automaton builds");
        prop_assert_eq!(
            before.scaled_distance(&source, &target),
            after.scaled_distance(&source, &target)
        );
    }

    #[test]
    fn arbitrary_binary_input_never_panics_or_bypasses_limits(
        bytes in prop::collection::vec(any::<u8>(), 0..=512),
        operation_limit in 0_usize..=16,
        pair_limit in 0_usize..=64,
        text_limit in 0_usize..=256,
    ) {
        let limits = OperationSetBinaryLimits {
            max_payload_bytes: 512,
            max_operations: operation_limit,
            max_operation_name_bytes: 64,
            max_restriction_pairs_per_operation: pair_limit,
            max_total_restriction_pairs: pair_limit,
            max_restriction_text_bytes: text_limit,
        };
        let decoded = std::panic::catch_unwind(|| {
            OperationSet::from_binary_with_limits(&bytes, limits)
        });
        prop_assert!(decoded.is_ok(), "decoder panicked on {bytes:?}");
        if let Ok(Ok(operation_set)) = decoded {
            prop_assert!(operation_set.len() <= operation_limit);
            prop_assert!(operation_set.validate().is_ok());
            for operation in operation_set.operations() {
                prop_assert!(operation.name().len() <= 64);
                if let OperationApplicability::Listed(restriction) = operation.applicability() {
                    prop_assert!(restriction.len() <= pair_limit);
                }
            }
        }
    }
}
