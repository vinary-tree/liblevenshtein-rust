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
    bincode_compat, BincodeSerializer, DictionarySerializer, JsonSerializer, PlainTextSerializer,
    SerializationError,
};

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

/// The other text/binary serializers share the same `DictionarySerializer` trait.
fn json_and_plaintext() -> Result<(), SerializationError> {
    let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);

    let mut json = Vec::new();
    JsonSerializer::serialize(&dict, &mut json)?;
    let _from_json: DoubleArrayTrie = JsonSerializer::deserialize(&json[..])?;

    let mut text = Vec::new();
    PlainTextSerializer::serialize(&dict, &mut text)?;
    let _from_text: DoubleArrayTrie = PlainTextSerializer::deserialize(&text[..])?;
    Ok(())
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
    json_and_plaintext()?;
    dynamic_dawg_deep_copy()?;
    dynamic_dawg_char_deep_copy()?;
    custom_serde_bytes()?;
    versioned_dictionary()?;

    #[cfg(feature = "compression")]
    gzip_roundtrip()?;

    #[cfg(feature = "pathmap-backend")]
    pathmap_roundtrip()?;

    println!("all documented serialization patterns compile and round-trip");
    Ok(())
}
