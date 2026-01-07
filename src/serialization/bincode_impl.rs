//! Bincode serializer for compact binary format.

use libdictenstein::{Dictionary, DictionaryNode};
use std::io::{Read, Write};

use super::{extract_terms, DictionaryFromTerms, DictionarySerializer, SerializationError};

use libdictenstein::suffix_automaton::SuffixAutomaton;

/// Bincode serializer for compact binary format.
///
/// This serializer uses bincode for fast, space-efficient serialization.
/// It's ideal for production use where storage space and load time matter.
pub struct BincodeSerializer;

impl BincodeSerializer {
    /// Serialize a suffix automaton using its source texts.
    ///
    /// This is more efficient than using the generic `serialize()` method,
    /// which would extract all substrings.
    pub fn serialize_suffix_automaton<W>(
        automaton: &SuffixAutomaton,
        mut writer: W,
    ) -> Result<(), SerializationError>
    where
        W: Write,
    {
        let texts = automaton.source_texts();
        bincode::serialize_into(&mut writer, &texts)?;
        Ok(())
    }

    /// Deserialize a suffix automaton from source texts.
    pub fn deserialize_suffix_automaton<R>(
        mut reader: R,
    ) -> Result<SuffixAutomaton, SerializationError>
    where
        R: Read,
    {
        let texts: Vec<String> = bincode::deserialize_from(&mut reader)?;
        Ok(SuffixAutomaton::from_texts(texts))
    }
}

impl DictionarySerializer for BincodeSerializer {
    fn serialize<D, W>(dict: &D, mut writer: W) -> Result<(), SerializationError>
    where
        D: Dictionary,
        D::Node: DictionaryNode<Unit = u8>,
        W: Write,
    {
        let terms = extract_terms(dict);
        bincode::serialize_into(&mut writer, &terms)?;
        Ok(())
    }

    fn deserialize<D, R>(mut reader: R) -> Result<D, SerializationError>
    where
        D: DictionaryFromTerms,
        R: Read,
    {
        let terms: Vec<String> = bincode::deserialize_from(&mut reader)?;
        Ok(D::from_terms(terms))
    }
}
