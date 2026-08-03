#![cfg(feature = "protobuf")]

use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
use liblevenshtein::transducer::{
    OperationApplicability, OperationSet, OperationSetBinaryLimits, OperationSetBuilder,
    OperationSetProtobufError, OperationType, SubstitutionSet,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

fn protobuf_config() -> ProptestConfig {
    let mut config = ProptestConfig::with_failure_persistence(FileFailurePersistence::Direct(
        "tests/operation_set_protobuf.proptest-regressions",
    ));
    config.cases = 512;
    config
}

fn rich_operation_set(restrictions: SubstitutionSet) -> OperationSet {
    OperationSetBuilder::new()
        .with_standard_ops()
        .with_transposition()
        .with_operation(OperationType::with_owned_restriction(
            2,
            1,
            0.125,
            restrictions,
            "runtime_digraph".to_owned(),
        ))
        .build()
}

fn push_varint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn varint_field(tag: u32, value: u64) -> Vec<u8> {
    let mut output = Vec::new();
    push_varint(u64::from(tag) << 3, &mut output);
    push_varint(value, &mut output);
    output
}

fn fixed64_field(tag: u32, value: u64) -> Vec<u8> {
    let mut output = Vec::new();
    push_varint((u64::from(tag) << 3) | 1, &mut output);
    output.extend_from_slice(&value.to_le_bytes());
    output
}

fn bytes_field(tag: u32, value: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    push_varint((u64::from(tag) << 3) | 2, &mut output);
    push_varint(value.len() as u64, &mut output);
    output.extend_from_slice(value);
    output
}

fn operation_message(
    consume_x: u64,
    consume_y: u64,
    weight_bits: u64,
    applicability: i32,
    restrictions: &[Vec<u8>],
    name: &[u8],
) -> Vec<u8> {
    let mut operation = Vec::new();
    operation.extend(varint_field(1, consume_x));
    operation.extend(varint_field(2, consume_y));
    operation.extend(fixed64_field(3, weight_bits));
    operation.extend(varint_field(4, applicability as u64));
    for restriction in restrictions {
        operation.extend(bytes_field(5, restriction));
    }
    operation.extend(bytes_field(6, name));
    operation
}

fn container(operations: &[Vec<u8>]) -> Vec<u8> {
    let mut operation_set = Vec::new();
    for operation in operations {
        operation_set.extend(bytes_field(1, operation));
    }
    bytes_field(1, &operation_set)
}

fn byte_pair(source: u64, target: u64) -> Vec<u8> {
    let mut pair = Vec::new();
    let mut bytes = varint_field(1, source);
    bytes.extend(varint_field(2, target));
    pair.extend(bytes_field(1, &bytes));
    pair
}

fn string_pair(source: &[u8], target: &[u8]) -> Vec<u8> {
    let mut pair = Vec::new();
    let mut strings = bytes_field(1, source);
    strings.extend(bytes_field(2, target));
    pair.extend(bytes_field(2, &strings));
    pair
}

#[test]
fn protobuf_roundtrip_preserves_configuration_bits_and_execution() {
    let mut restrictions = SubstitutionSet::new();
    restrictions.allow_str("ph", "f");
    restrictions.allow_str("αβ", "γ");

    let mut original = rich_operation_set(restrictions);
    let mut raw_restriction = SubstitutionSet::new();
    raw_restriction.allow_byte(0xff, 0x00);
    original.add(OperationType::with_owned_restriction(
        1,
        1,
        0.25,
        raw_restriction,
        "raw_bytes".to_owned(),
    ));
    let executable_original = original.clone();
    original.add(OperationType::with_owned_applicability(
        1,
        1,
        f64::MIN_POSITIVE,
        OperationApplicability::Any,
        "runtime_any".to_owned(),
    ));
    original.add(OperationType::with_owned_applicability(
        1,
        1,
        -0.0,
        OperationApplicability::Equal,
        "negative_zero_match".to_owned(),
    ));

    let encoded = original.to_protobuf().expect("valid operation set encodes");
    let restored = OperationSet::from_protobuf(&encoded).expect("protobuf decodes");

    assert_eq!(restored, original);
    assert_eq!(
        restored
            .operations()
            .last()
            .expect("last operation")
            .weight()
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    assert!(restored.operations()[5].can_apply_str("ph", "f"));
    assert!(restored.operations()[5].can_apply_str("αβ", "γ"));
    assert!(restored.operations()[6].can_apply(&[0xff], &[0x00]));

    let executable_restored = OperationSet::from_protobuf(
        &executable_original
            .to_protobuf()
            .expect("executable operation set encodes"),
    )
    .expect("executable operation set decodes");

    for budget in 0..=3 {
        let before = GeneralizedAutomaton::try_with_operations(budget, executable_original.clone())
            .expect("source operation set is executable");
        let after = GeneralizedAutomaton::try_with_operations(budget, executable_restored.clone())
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
fn canonical_restriction_order_produces_deterministic_protobuf() {
    let pairs = [
        ("ph", "f"),
        ("ch", "k"),
        ("th", "t"),
        ("sh", "s"),
        ("αβ", "γ"),
    ];
    let mut forward = SubstitutionSet::new();
    for (source, target) in pairs {
        forward.allow_str(source, target);
    }
    let mut reverse = SubstitutionSet::new();
    for (source, target) in pairs.into_iter().rev() {
        reverse.allow_str(source, target);
    }

    assert_eq!(
        rich_operation_set(forward)
            .to_protobuf()
            .expect("first set encodes"),
        rich_operation_set(reverse)
            .to_protobuf()
            .expect("second set encodes")
    );
}

#[test]
fn protobuf_preflight_rejects_counts_before_decoded_vectors_are_built() {
    let empty_operation = Vec::new();
    let encoded = container(&[empty_operation.clone(), empty_operation]);
    let limits = OperationSetBinaryLimits {
        max_operations: 1,
        ..OperationSetBinaryLimits::default()
    };
    assert!(matches!(
        OperationSet::from_protobuf_with_limits(&encoded, limits),
        Err(OperationSetProtobufError::ResourceLimit {
            resource: "operation count",
            observed: 2,
            limit: 1,
        })
    ));

    let encoded = container(&[operation_message(
        1,
        1,
        1.0_f64.to_bits(),
        4,
        &[Vec::new(), Vec::new()],
        b"listed",
    )]);
    let limits = OperationSetBinaryLimits {
        max_restriction_pairs_per_operation: 1,
        ..OperationSetBinaryLimits::default()
    };
    assert!(matches!(
        OperationSet::from_protobuf_with_limits(&encoded, limits),
        Err(OperationSetProtobufError::ResourceLimit {
            resource: "restriction pairs per operation",
            observed: 2,
            limit: 1,
        })
    ));

    let encoded = container(&[operation_message(
        2,
        1,
        1.0_f64.to_bits(),
        4,
        &[string_pair(b"phonetic", b"f")],
        b"listed",
    )]);
    let limits = OperationSetBinaryLimits {
        max_restriction_text_bytes: 4,
        ..OperationSetBinaryLimits::default()
    };
    assert!(matches!(
        OperationSet::from_protobuf_with_limits(&encoded, limits),
        Err(OperationSetProtobufError::ResourceLimit {
            resource: "restriction text bytes",
            ..
        })
    ));
}

#[test]
fn protobuf_rejects_malformed_wire_and_missing_schema_version() {
    assert!(matches!(
        OperationSet::from_protobuf(&[]),
        Err(OperationSetProtobufError::UnsupportedFormat)
    ));
    assert!(matches!(
        OperationSet::from_protobuf(&[0x0a, 0x80]),
        Err(OperationSetProtobufError::MalformedWire(_))
    ));
    assert!(matches!(
        OperationSet::from_protobuf(&[0x0b]),
        Err(OperationSetProtobufError::MalformedWire(_))
    ));
}

#[test]
fn protobuf_rejects_invalid_discriminators_weights_and_hidden_restrictions() {
    let unspecified = container(&[operation_message(
        1,
        1,
        1.0_f64.to_bits(),
        0,
        &[],
        b"unspecified",
    )]);
    assert!(matches!(
        OperationSet::from_protobuf(&unspecified),
        Err(OperationSetProtobufError::InvalidField {
            field: "applicability",
            ..
        })
    ));

    let infinite = container(&[operation_message(
        1,
        1,
        f64::INFINITY.to_bits(),
        1,
        &[],
        b"infinite",
    )]);
    assert!(matches!(
        OperationSet::from_protobuf(&infinite),
        Err(OperationSetProtobufError::InvalidField {
            field: "weight_bits",
            ..
        })
    ));

    let hidden = container(&[operation_message(
        1,
        1,
        1.0_f64.to_bits(),
        1,
        &[byte_pair(b'a' as u64, b'b' as u64)],
        b"not_listed",
    )]);
    assert!(matches!(
        OperationSet::from_protobuf(&hidden),
        Err(OperationSetProtobufError::InvalidField {
            field: "restriction",
            ..
        })
    ));
}

#[test]
fn protobuf_rejects_invalid_pair_payloads_and_semantic_arities() {
    let oversized_byte = container(&[operation_message(
        1,
        1,
        1.0_f64.to_bits(),
        4,
        &[byte_pair(256, 0)],
        b"oversized_byte",
    )]);
    assert!(matches!(
        OperationSet::from_protobuf(&oversized_byte),
        Err(OperationSetProtobufError::InvalidField {
            field: "restriction.bytes.source",
            ..
        })
    ));

    let missing_oneof = container(&[operation_message(
        1,
        1,
        1.0_f64.to_bits(),
        4,
        &[Vec::new()],
        b"missing_pair",
    )]);
    assert!(matches!(
        OperationSet::from_protobuf(&missing_oneof),
        Err(OperationSetProtobufError::InvalidField {
            field: "restriction.pair",
            ..
        })
    ));

    let empty_string = container(&[operation_message(
        1,
        1,
        1.0_f64.to_bits(),
        4,
        &[string_pair(b"", b"x")],
        b"empty_string",
    )]);
    assert!(matches!(
        OperationSet::from_protobuf(&empty_string),
        Err(OperationSetProtobufError::InvalidField {
            field: "restriction.strings",
            ..
        })
    ));

    let wrong_arity = container(&[operation_message(
        1,
        1,
        1.0_f64.to_bits(),
        4,
        &[string_pair(b"ph", b"f")],
        b"wrong_arity",
    )]);
    assert!(matches!(
        OperationSet::from_protobuf(&wrong_arity),
        Err(OperationSetProtobufError::Validation(_))
    ));
}

#[test]
fn protobuf_unknown_fields_are_forward_compatible_but_not_reemitted() {
    let original = OperationSet::standard();
    let canonical = original.to_protobuf().expect("standard set encodes");
    let mut with_unknown = canonical.clone();
    with_unknown.extend(varint_field(127, 42));

    let restored = OperationSet::from_protobuf(&with_unknown).expect("unknown field is skipped");
    assert_eq!(restored, original);
    assert_eq!(
        restored.to_protobuf().expect("restored set re-encodes"),
        canonical
    );
}

proptest! {
    #![proptest_config(protobuf_config())]

    #[test]
    fn protobuf_roundtrip_is_deterministic_and_execution_equivalent(
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
            .with_operation(OperationType::with_owned_restriction(
                1,
                1,
                0.5,
                restrictions,
                "byte_equivalence".to_owned(),
            ))
            .build();
        let bytes = operations.to_protobuf().expect("generated set encodes");
        let restored = OperationSet::from_protobuf(&bytes).expect("generated set decodes");

        prop_assert_eq!(&restored, &operations);
        prop_assert_eq!(restored.to_protobuf().expect("restored set re-encodes"), bytes);

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
    fn arbitrary_protobuf_input_never_panics_or_bypasses_limits(
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
            OperationSet::from_protobuf_with_limits(&bytes, limits)
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
