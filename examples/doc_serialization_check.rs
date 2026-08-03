//! Compile-check for every serialization pattern shown in the documentation.
//!
//! This example exists to keep `docs/user-guide/serialization.md` and the per-backend
//! implementation guides **honest**: every snippet they teach is reproduced here, so a
//! `cargo check --example doc_serialization_check --all-features` fails the moment a
//! documented API drifts out of existence.
//!
//! Run with:
//! ```sh
//! cargo check --example doc_serialization_check --all-features
//! ```

use libdictenstein::double_array_trie::DoubleArrayTrie;
use libdictenstein::dynamic_dawg::{DynamicDawg, DynamicDawgChar};
use libdictenstein::serialization::{
    bincode_compat, BincodeSerializer, DictionarySerializer, SerializationError,
};
use liblevenshtein::transducer::Algorithm;
use liblevenshtein::transducer::{OperationSet, OperationSetBinaryLimits, OperationSetBuilder};

/// `BincodeSerializer` round-trip through a byte buffer
/// (`Vec<u8>: Write`, `&[u8]: Read`) — the pattern used by the DoubleArrayTrie,
/// DynamicDawg and PathMapDictionary implementation guides.
fn bincode_roundtrip() -> Result<(), SerializationError> {
    let dict = DoubleArrayTrie::from_terms(vec!["save", "load"]);

    let mut bytes = Vec::new();
    BincodeSerializer::serialize(&dict, &mut bytes)?;

    let loaded: DoubleArrayTrie = BincodeSerializer::deserialize(&bytes[..])?;
    assert!(loaded.contains("save"));
    Ok(())
}

/// A `&D` receiver must be passed straight through (no extra `&`) — the
/// `save_dictionary` / `save_to_redis` pattern.
fn serialize_via_reference(dict: &DoubleArrayTrie) -> Result<Vec<u8>, SerializationError> {
    let mut bytes = Vec::new();
    BincodeSerializer::serialize(dict, &mut bytes)?;
    Ok(bytes)
}

/// `DynamicDawg` (byte-level) deep-copy via serialization.
fn dynamic_dawg_deep_copy() -> Result<(), SerializationError> {
    let dict1: DynamicDawg = DynamicDawg::from_terms(vec!["test"]);

    let mut bytes = Vec::new();
    BincodeSerializer::serialize(&dict1, &mut bytes)?;
    let dict2: DynamicDawg = BincodeSerializer::deserialize(&bytes[..])?;

    assert!(dict2.contains("test"));
    Ok(())
}

/// `DynamicDawgChar` is `Unit = char`, so the `DictionarySerializer` trait (bounded on
/// `Unit = u8`) does NOT apply. It derives serde, so it round-trips through the
/// `bincode_compat` shim — exactly what `dynamic-dawg-char.md` now teaches.
fn dynamic_dawg_char_deep_copy() -> Result<(), Box<dyn std::error::Error>> {
    let dict1: DynamicDawgChar = DynamicDawgChar::from_terms(vec!["café", "新しい"]);

    let bytes = bincode_compat::serialize(&dict1)?;
    let dict2: DynamicDawgChar = bincode_compat::deserialize(&bytes)?;

    assert!(dict2.contains("café"));
    Ok(())
}

/// The "Custom Serialization" section: raw serde through the bincode-2.x shim.
fn custom_serde_bytes() -> Result<(), Box<dyn std::error::Error>> {
    let dict = DoubleArrayTrie::from_terms(vec!["test"]);
    let bytes: Vec<u8> = bincode_compat::serialize(&dict)?;
    let _dict: DoubleArrayTrie = bincode_compat::deserialize(&bytes)?;
    Ok(())
}

/// Public algorithm selectors, including unrestricted Damerau, retain their
/// identity through the same serde-compatible binary format.
fn algorithm_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = bincode_compat::serialize(&Algorithm::DamerauLevenshtein)?;
    let restored: Algorithm = bincode_compat::deserialize(&bytes)?;
    assert_eq!(restored, Algorithm::DamerauLevenshtein);
    Ok(())
}

/// The "Version Your Dictionaries" best practice: embed a backend that derives serde
/// and stream it out with the bincode-1.x-shaped `serialize_into`.
fn versioned_dictionary() -> Result<(), Box<dyn std::error::Error>> {
    #[derive(serde::Serialize, serde::Deserialize)]
    struct VersionedDictionary {
        version: String,
        dict: DoubleArrayTrie,
    }

    let dict = DoubleArrayTrie::from_terms(vec!["test"]);
    let versioned = VersionedDictionary {
        version: "1.0.0".to_string(),
        dict,
    };

    // `serialize_into` takes `&mut W`.
    let mut buf: Vec<u8> = Vec::new();
    bincode_compat::serialize_into(&mut buf, &versioned)?;

    let _restored: VersionedDictionary = bincode_compat::deserialize(&buf)?;
    Ok(())
}

/// Gzip wraps any `DictionarySerializer`: `GzipSerializer::<BincodeSerializer>`.
#[cfg(feature = "compression")]
fn gzip_roundtrip() -> Result<(), SerializationError> {
    use libdictenstein::serialization::GzipSerializer;

    let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);

    let mut gz = Vec::new();
    GzipSerializer::<BincodeSerializer>::serialize(&dict, &mut gz)?;

    let loaded: DoubleArrayTrie = GzipSerializer::<BincodeSerializer>::deserialize(&gz[..])?;
    assert!(loaded.contains("test"));
    Ok(())
}

/// The Protocol Buffers section uses the portable general serializer.
#[cfg(feature = "protobuf")]
fn protobuf_roundtrip() -> Result<(), SerializationError> {
    use libdictenstein::serialization::ProtobufSerializer;

    let dict = DoubleArrayTrie::from_terms(vec!["test", "tested", "testing"]);
    let mut bytes = Vec::new();
    ProtobufSerializer::serialize(&dict, &mut bytes)?;
    let loaded: DoubleArrayTrie = ProtobufSerializer::deserialize(&bytes[..])?;
    assert!(loaded.contains("tested"));
    Ok(())
}

/// The stable, bounded binary envelope used by generalized edit operations.
fn operation_set_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let operations = OperationSetBuilder::new().with_standard_ops().build();
    let bytes = operations.to_binary()?;

    let limits = OperationSetBinaryLimits {
        max_operations: 16,
        ..OperationSetBinaryLimits::default()
    };
    let restored = OperationSet::from_binary_with_limits(&bytes, limits)?;
    assert_eq!(restored, operations);
    Ok(())
}

/// Portable OperationSet schema and its bounded pre-allocation decoder.
#[cfg(feature = "protobuf")]
fn operation_set_protobuf_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let operations = OperationSetBuilder::new().with_standard_ops().build();
    let bytes = operations.to_protobuf()?;
    let restored =
        OperationSet::from_protobuf_with_limits(&bytes, OperationSetBinaryLimits::default())?;
    assert_eq!(restored, operations);
    Ok(())
}

/// Gzip remains an outer wrapper around either OperationSet binary format.
#[cfg(feature = "compression")]
fn operation_set_gzip_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
    let operations = OperationSetBuilder::new().with_standard_ops().build();
    let bincode = operations.to_binary_gzip()?;
    assert_eq!(OperationSet::from_binary_gzip(&bincode)?, operations);

    #[cfg(feature = "protobuf")]
    {
        let protobuf = operations.to_protobuf_gzip()?;
        assert_eq!(OperationSet::from_protobuf_gzip(&protobuf)?, operations);
    }
    Ok(())
}

/// `PathMapDictionary` implements NO serde traits — it must go through
/// `BincodeSerializer` (which encodes terms via the `Dictionary` trait).
#[cfg(feature = "pathmap-backend")]
fn pathmap_roundtrip() -> Result<(), SerializationError> {
    use libdictenstein::pathmap::PathMapDictionary;
    // `PathMapDictionary::contains` is a `Dictionary` *trait* method (the other backends
    // also expose an inherent one), so the trait must be in scope here.
    use libdictenstein::Dictionary;

    let dict1: PathMapDictionary = PathMapDictionary::from_terms(vec!["test"]);

    let mut bytes = Vec::new();
    BincodeSerializer::serialize(&dict1, &mut bytes)?;
    let dict2: PathMapDictionary = BincodeSerializer::deserialize(&bytes[..])?;

    assert!(dict2.contains("test"));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bincode_roundtrip()?;
    let dict = DoubleArrayTrie::from_terms(vec!["test"]);
    serialize_via_reference(&dict)?;
    dynamic_dawg_deep_copy()?;
    dynamic_dawg_char_deep_copy()?;
    custom_serde_bytes()?;
    algorithm_roundtrip()?;
    versioned_dictionary()?;

    #[cfg(feature = "compression")]
    gzip_roundtrip()?;

    #[cfg(feature = "protobuf")]
    protobuf_roundtrip()?;

    operation_set_roundtrip()?;

    #[cfg(feature = "protobuf")]
    operation_set_protobuf_roundtrip()?;

    #[cfg(feature = "compression")]
    operation_set_gzip_roundtrip()?;

    #[cfg(feature = "pathmap-backend")]
    pathmap_roundtrip()?;

    println!("all documented serialization patterns compile and round-trip");
    Ok(())
}
