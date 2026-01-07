//! Dictionary serialization support.
//!
//! This module provides serialization and deserialization of dictionaries
//! using various formats (bincode, JSON, protobuf) with optional compression.
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::serialization::{BincodeSerializer, DictionarySerializer};
//! use std::fs::File;
//!
//! // Create and populate dictionary
//! let dict = DoubleArrayTrie::from_terms(vec!["test", "testing"]);
//!
//! // Serialize to file
//! let file = File::create("dict.bin")?;
//! BincodeSerializer::serialize(&dict, file)?;
//!
//! // Deserialize from file
//! let file = File::open("dict.bin")?;
//! let loaded_dict: DoubleArrayTrie = BincodeSerializer::deserialize(file)?;
//! ```

use libdictenstein::{Dictionary, DictionaryNode};
use std::io::{Read, Write};

// Serializer implementations
mod bincode_impl;
mod json_impl;
mod plaintext_impl;

#[cfg(feature = "protobuf")]
pub mod protobuf_impl;

#[cfg(feature = "compression")]
mod compression_impl;

// Re-exports
pub use self::bincode_impl::BincodeSerializer;
pub use self::json_impl::JsonSerializer;
pub use self::plaintext_impl::PlainTextSerializer;

#[cfg(feature = "protobuf")]
pub use self::protobuf_impl::{
    DatProtobufSerializer, OptimizedProtobufSerializer, ProtobufSerializer,
    SuffixAutomatonProtobufSerializer,
};

#[cfg(feature = "compression")]
pub use self::compression_impl::GzipSerializer;

/// Trait for serializing and deserializing dictionaries.
pub trait DictionarySerializer {
    /// Serialize a dictionary to a writer.
    ///
    /// # Arguments
    ///
    /// * `dict` - The dictionary to serialize
    /// * `writer` - Where to write the serialized data
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails or writing fails.
    fn serialize<D, W>(dict: &D, writer: W) -> Result<(), SerializationError>
    where
        D: Dictionary,
        D::Node: DictionaryNode<Unit = u8>,
        W: Write;

    /// Deserialize a dictionary from a reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - Where to read the serialized data from
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails or reading fails.
    fn deserialize<D, R>(reader: R) -> Result<D, SerializationError>
    where
        D: DictionaryFromTerms,
        R: Read;
}

/// Trait for dictionaries that can be constructed from a list of terms.
///
/// This is used by the serialization system to reconstruct dictionaries
/// after deserialization.
pub trait DictionaryFromTerms: Sized {
    /// Create a dictionary from an iterator of terms.
    fn from_terms<I: IntoIterator<Item = String>>(terms: I) -> Self;
}

/// Errors that can occur during serialization/deserialization.
#[derive(Debug, thiserror::Error)]
pub enum SerializationError {
    /// Error during bincode serialization
    #[error("Bincode error")]
    Bincode(#[from] bincode::Error),
    /// Error during JSON serialization
    #[error("JSON error")]
    Json(#[from] serde_json::Error),
    /// Error during protobuf serialization
    #[cfg(feature = "protobuf")]
    #[error("Protobuf error")]
    Protobuf(#[from] prost::DecodeError),
    /// I/O error
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    /// Dictionary iteration error
    #[error("Dictionary error: {0}")]
    DictionaryError(String),
}

/// Helper to extract all terms from a dictionary.
///
/// This performs a depth-first traversal of the dictionary trie
/// to collect all valid terms.
///
/// **Note**: For suffix automata, use `extract_suffix_automaton_texts()` instead,
/// as this function would extract all possible substrings rather than source texts.
pub fn extract_terms<D>(dict: &D) -> Vec<String>
where
    D: Dictionary,
    D::Node: DictionaryNode<Unit = u8>,
{
    // Pre-allocate with estimated capacity
    let est_size = dict.len().unwrap_or(100);
    let mut terms = Vec::with_capacity(est_size);
    let mut current_term = Vec::with_capacity(32); // Most words < 32 bytes

    fn dfs<N: DictionaryNode<Unit = u8>>(
        node: &N,
        current_term: &mut Vec<u8>,
        terms: &mut Vec<String>,
    ) {
        if node.is_final() {
            // SAFETY: Dictionary implementations maintain the invariant that
            // all terms are valid UTF-8. We avoid the clone by using
            // from_utf8_unchecked, which is safe because:
            // 1. Dictionaries are constructed from valid UTF-8 strings
            // 2. We only traverse edges that were part of valid UTF-8 terms
            // 3. The byte sequence is validated during dictionary construction
            //
            // Fallback: If somehow invalid UTF-8 is encountered (shouldn't happen),
            // we use from_utf8_lossy which replaces invalid sequences with �
            match std::str::from_utf8(current_term) {
                Ok(s) => terms.push(s.to_string()),
                Err(_) => {
                    // Defensive: shouldn't happen with proper dictionary implementations
                    terms.push(String::from_utf8_lossy(current_term).into_owned());
                }
            }
        }

        // Explore all edges
        for (byte, child) in node.edges() {
            current_term.push(byte);
            dfs(&child, current_term, terms);
            current_term.pop();
        }
    }

    let root = dict.root();
    dfs(&root, &mut current_term, &mut terms);

    terms
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::double_array_trie::DoubleArrayTrie;

    #[test]
    fn test_bincode_roundtrip() {
        let dict = DoubleArrayTrie::from_terms(vec!["hello", "world", "test"]);
        let mut buffer = Vec::new();

        BincodeSerializer::serialize(&dict, &mut buffer).unwrap();
        let loaded: DoubleArrayTrie = BincodeSerializer::deserialize(&buffer[..]).unwrap();

        assert!(loaded.contains("hello"));
        assert!(loaded.contains("world"));
        assert!(loaded.contains("test"));
        assert!(!loaded.contains("missing"));
    }

    #[test]
    fn test_json_roundtrip() {
        let dict = DoubleArrayTrie::from_terms(vec!["alpha", "beta", "gamma"]);
        let mut buffer = Vec::new();

        JsonSerializer::serialize(&dict, &mut buffer).unwrap();
        let loaded: DoubleArrayTrie = JsonSerializer::deserialize(&buffer[..]).unwrap();

        assert!(loaded.contains("alpha"));
        assert!(loaded.contains("beta"));
        assert!(loaded.contains("gamma"));
        assert!(!loaded.contains("delta"));
    }

    #[test]
    fn test_extract_terms() {
        let dict = DoubleArrayTrie::from_terms(vec!["apple", "apply", "application"]);
        let terms = extract_terms(&dict);

        assert_eq!(terms.len(), 3);
        assert!(terms.contains(&"apple".to_string()));
        assert!(terms.contains(&"apply".to_string()));
        assert!(terms.contains(&"application".to_string()));
    }

    #[test]
    fn test_suffix_automaton_serialization() {
        use libdictenstein::suffix_automaton::SuffixAutomaton;

        let texts = vec!["hello world".to_string(), "test string".to_string()];
        let dict = SuffixAutomaton::from_texts(texts.clone());

        // Test bincode serialization using specialized methods
        let mut buffer = Vec::new();
        BincodeSerializer::serialize_suffix_automaton(&dict, &mut buffer).unwrap();

        let loaded = BincodeSerializer::deserialize_suffix_automaton(&buffer[..]).unwrap();

        // Verify the loaded automaton works
        assert!(loaded.contains("hello"));
        assert!(loaded.contains("world"));
        assert!(loaded.contains("test"));
        assert!(loaded.contains("string"));
        assert!(!loaded.contains("missing"));

        // Verify source texts are preserved
        let sources = loaded.source_texts();
        assert_eq!(sources.len(), 2);
        assert!(sources.contains(&"hello world".to_string()));
        assert!(sources.contains(&"test string".to_string()));
    }

    #[cfg(feature = "protobuf")]
    #[test]
    fn test_suffix_automaton_protobuf_serialization() {
        use libdictenstein::suffix_automaton::SuffixAutomaton;
        use crate::serialization::SuffixAutomatonProtobufSerializer;

        let texts = vec!["hello world".to_string(), "test string".to_string()];
        let dict = SuffixAutomaton::from_texts(texts.clone());

        // Test protobuf serialization
        let mut buffer = Vec::new();
        SuffixAutomatonProtobufSerializer::serialize_suffix_automaton(&dict, &mut buffer).unwrap();

        let loaded =
            SuffixAutomatonProtobufSerializer::deserialize_suffix_automaton(&buffer[..]).unwrap();

        // Verify the loaded automaton works
        assert!(loaded.contains("hello"));
        assert!(loaded.contains("world"));
        assert!(loaded.contains("test"));
        assert!(loaded.contains("string"));
        assert!(!loaded.contains("missing"));

        // Verify source texts are preserved
        let sources = loaded.source_texts();
        assert_eq!(sources.len(), 2);
        assert!(sources.contains(&"hello world".to_string()));
        assert!(sources.contains(&"test string".to_string()));
    }

    #[cfg(feature = "protobuf")]
    #[test]
    fn test_dat_protobuf_serialization() {
        use crate::serialization::DatProtobufSerializer;

        let dict = DoubleArrayTrie::from_terms(vec!["apple", "apply", "application"]);

        // Test protobuf serialization
        let mut buffer = Vec::new();
        DatProtobufSerializer::serialize_dat(&dict, &mut buffer).unwrap();

        let loaded = DatProtobufSerializer::deserialize_dat(&buffer[..]).unwrap();

        // Verify the loaded dictionary works
        assert!(loaded.contains("apple"));
        assert!(loaded.contains("apply"));
        assert!(loaded.contains("application"));
        assert!(!loaded.contains("app"));
        assert!(!loaded.contains("banana"));
    }

    #[cfg(feature = "protobuf")]
    #[test]
    fn test_protobuf_roundtrip() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);
        let mut buffer = Vec::new();

        ProtobufSerializer::serialize(&dict, &mut buffer).unwrap();
        let loaded: DoubleArrayTrie = ProtobufSerializer::deserialize(&buffer[..]).unwrap();

        assert!(loaded.contains("test"));
        assert!(loaded.contains("testing"));
        assert!(loaded.contains("tested"));
        assert!(!loaded.contains("tester"));
    }

    #[cfg(feature = "protobuf")]
    #[test]
    fn test_optimized_protobuf_roundtrip() {
        let dict = DoubleArrayTrie::from_terms(vec!["alpha", "beta", "gamma"]);
        let mut buffer = Vec::new();

        OptimizedProtobufSerializer::serialize(&dict, &mut buffer).unwrap();
        let loaded: DoubleArrayTrie =
            OptimizedProtobufSerializer::deserialize(&buffer[..]).unwrap();

        assert!(loaded.contains("alpha"));
        assert!(loaded.contains("beta"));
        assert!(loaded.contains("gamma"));
        assert!(!loaded.contains("delta"));
    }

    #[cfg(feature = "protobuf")]
    #[test]
    fn test_protobuf_format_comparison() {
        // Compare serialization sizes for different protobuf formats
        let dict = DoubleArrayTrie::from_terms(vec![
            "test",
            "testing",
            "tested",
            "tester",
            "tests",
            "apple",
            "apply",
            "application",
            "applicable",
        ]);

        // Standard ProtobufSerializer (V1 format)
        let mut buf_v1 = Vec::new();
        ProtobufSerializer::serialize(&dict, &mut buf_v1).unwrap();

        // OptimizedProtobufSerializer (V2 format)
        let mut buf_v2 = Vec::new();
        OptimizedProtobufSerializer::serialize(&dict, &mut buf_v2).unwrap();

        // DatProtobufSerializer (term extraction)
        let mut buf_dat = Vec::new();
        DatProtobufSerializer::serialize_dat(&dict, &mut buf_dat).unwrap();

        // V2 should be smaller than V1 (delta encoding + packed format)
        assert!(
            buf_v2.len() < buf_v1.len(),
            "V2 ({} bytes) should be smaller than V1 ({} bytes)",
            buf_v2.len(),
            buf_v1.len()
        );

        // DAT format should be competitive
        println!("Protobuf V1 size: {} bytes", buf_v1.len());
        println!("Protobuf V2 size: {} bytes", buf_v2.len());
        println!("DAT format size: {} bytes", buf_dat.len());

        // Verify all formats deserialize correctly
        let loaded_v1: DoubleArrayTrie = ProtobufSerializer::deserialize(&buf_v1[..]).unwrap();
        let loaded_v2: DoubleArrayTrie =
            OptimizedProtobufSerializer::deserialize(&buf_v2[..]).unwrap();
        let loaded_dat = DatProtobufSerializer::deserialize_dat(&buf_dat[..]).unwrap();

        for term in ["test", "testing", "apple", "application"] {
            assert!(loaded_v1.contains(term));
            assert!(loaded_v2.contains(term));
            assert!(loaded_dat.contains(term));
        }
    }
}
