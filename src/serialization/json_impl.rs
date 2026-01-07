//! JSON serializer for human-readable format.

use libdictenstein::{Dictionary, DictionaryNode};
use std::io::{Read, Write};

use super::{extract_terms, DictionaryFromTerms, DictionarySerializer, SerializationError};

/// JSON serializer for human-readable format.
///
/// This serializer uses JSON for easy debugging and manual inspection.
/// It's less efficient than bincode but useful for development.
pub struct JsonSerializer;

impl DictionarySerializer for JsonSerializer {
    fn serialize<D, W>(dict: &D, mut writer: W) -> Result<(), SerializationError>
    where
        D: Dictionary,
        D::Node: DictionaryNode<Unit = u8>,
        W: Write,
    {
        let terms = extract_terms(dict);
        serde_json::to_writer_pretty(&mut writer, &terms)?;
        Ok(())
    }

    fn deserialize<D, R>(mut reader: R) -> Result<D, SerializationError>
    where
        D: DictionaryFromTerms,
        R: Read,
    {
        let terms: Vec<String> = serde_json::from_reader(&mut reader)?;
        Ok(D::from_terms(terms))
    }
}
