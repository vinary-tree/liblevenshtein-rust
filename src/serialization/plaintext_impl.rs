//! Plain text serialization for dictionaries.
//!
//! Serializes dictionaries as newline-delimited UTF-8 text files (one term per line).
//! This is the simplest and most human-readable format, ideal for:
//! - Manual editing
//! - Version control
//! - Debugging
//! - Cross-language compatibility
//!
//! # Format
//!
//! Each term is written on its own line, encoded as UTF-8:
//! ```text
//! apple
//! banana
//! cherry
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::prelude::*;
//! use liblevenshtein::serialization::{PlainTextSerializer, DictionarySerializer};
//! use std::fs::File;
//!
//! let dict = DoubleArrayTrie::from_terms(vec!["apple", "banana", "cherry"]);
//!
//! // Serialize to file
//! let file = File::create("terms.txt")?;
//! PlainTextSerializer::serialize(&dict, file)?;
//!
//! // Deserialize from file
//! let file = File::open("terms.txt")?;
//! let loaded: DoubleArrayTrie = PlainTextSerializer::deserialize(file)?;
//! ```

use super::{extract_terms, DictionaryFromTerms, DictionarySerializer, SerializationError};
use libdictenstein::{Dictionary, DictionaryNode};
use std::io::{BufRead, BufReader, Write};

/// Plain text serializer using newline-delimited UTF-8.
pub struct PlainTextSerializer;

impl DictionarySerializer for PlainTextSerializer {
    fn serialize<D, W>(dict: &D, mut writer: W) -> Result<(), SerializationError>
    where
        D: Dictionary,
        D::Node: DictionaryNode<Unit = u8>,
        W: Write,
    {
        // Extract all terms from the dictionary
        let terms = extract_terms(dict);

        // Write each term on its own line
        for term in terms {
            writeln!(writer, "{}", term)?;
        }

        Ok(())
    }

    fn deserialize<D, R>(reader: R) -> Result<D, SerializationError>
    where
        D: DictionaryFromTerms,
        R: std::io::Read,
    {
        let buf_reader = BufReader::new(reader);
        let mut terms = Vec::new();

        // Read each line as a term
        for line in buf_reader.lines() {
            let term = line?;
            // Skip empty lines
            if !term.is_empty() {
                terms.push(term);
            }
        }

        Ok(D::from_terms(terms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libdictenstein::double_array_trie::DoubleArrayTrie;

    #[test]
    fn test_plaintext_roundtrip() {
        let dict = DoubleArrayTrie::from_terms(vec!["apple", "banana", "cherry"]);
        let mut buffer = Vec::new();

        PlainTextSerializer::serialize(&dict, &mut buffer).unwrap();
        let loaded: DoubleArrayTrie = PlainTextSerializer::deserialize(&buffer[..]).unwrap();

        assert!(loaded.contains("apple"));
        assert!(loaded.contains("banana"));
        assert!(loaded.contains("cherry"));
        assert!(!loaded.contains("date"));
    }

    #[test]
    fn test_plaintext_format() {
        let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tested"]);
        let mut buffer = Vec::new();

        PlainTextSerializer::serialize(&dict, &mut buffer).unwrap();
        let text = String::from_utf8(buffer).unwrap();

        // Should have three lines (order may vary due to dictionary traversal)
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(text.contains("test\n"));
        assert!(text.contains("testing\n"));
        assert!(text.contains("tested\n"));
    }

    #[test]
    fn test_plaintext_empty_lines_skipped() {
        let input = "apple\n\nbanana\n\ncherry\n".as_bytes();
        let loaded: DoubleArrayTrie = PlainTextSerializer::deserialize(input).unwrap();

        assert_eq!(loaded.len(), Some(3));
        assert!(loaded.contains("apple"));
        assert!(loaded.contains("banana"));
        assert!(loaded.contains("cherry"));
    }

    #[test]
    fn test_plaintext_utf8() {
        let dict = DoubleArrayTrie::from_terms(vec!["café", "naïve", "日本語"]);
        let mut buffer = Vec::new();

        PlainTextSerializer::serialize(&dict, &mut buffer).unwrap();
        let loaded: DoubleArrayTrie = PlainTextSerializer::deserialize(&buffer[..]).unwrap();

        assert!(loaded.contains("café"));
        assert!(loaded.contains("naïve"));
        assert!(loaded.contains("日本語"));
    }
}
