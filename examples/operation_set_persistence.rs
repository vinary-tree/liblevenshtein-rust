//! Round-trip a complete generalized operation set through both supported
//! binary persistence formats.
//!
//! Run with:
//! `cargo run --example operation_set_persistence --features protobuf`

use liblevenshtein::transducer::generalized::GeneralizedAutomaton;
use liblevenshtein::transducer::{
    OperationSet, OperationSetBuilder, OperationType, SubstitutionSet,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut digraphs = SubstitutionSet::new();
    digraphs.allow_str("ph", "f");
    digraphs.allow_str("αβ", "γ");

    let operations = OperationSetBuilder::new()
        .with_standard_ops()
        .with_transposition()
        .with_operation(OperationType::with_owned_restriction(
            2,
            1,
            0.125,
            digraphs,
            "runtime_digraph".to_owned(),
        ))
        .build();

    // Compile-gate the generalized-operation construction shown in
    // docs/design/generalized-operations.md as well as the persistence calls.
    let automaton = GeneralizedAutomaton::try_with_operations(1, operations.clone())?;
    let scale = automaton.cost_scale()?;
    assert_eq!(scale.denominator(), 8);
    assert_eq!(automaton.scaled_distance("phone", "fone")?, Some(1));
    assert_eq!(scale.from_scaled(1), 0.125);

    let bincode_bytes = operations.to_binary()?;
    let protobuf_bytes = operations.to_protobuf()?;

    let from_bincode = OperationSet::from_binary(&bincode_bytes)?;
    let from_protobuf = OperationSet::from_protobuf(&protobuf_bytes)?;
    assert_eq!(from_bincode, operations);
    assert_eq!(from_protobuf, operations);
    assert!(from_protobuf.operations()[5].can_apply_str("ph", "f"));

    println!("bincode: {} bytes", bincode_bytes.len());
    println!("protobuf: {} bytes", protobuf_bytes.len());
    Ok(())
}
