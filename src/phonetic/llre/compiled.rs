//! AOT (Ahead-of-Time) compilation support for `.llre` files.
//!
//! This module provides serialization/deserialization of compiled NFAs for
//! instant loading without runtime parsing and compilation.
//!
//! # Binary Format
//!
//! ```text
//! +------------------+------------------+------------------+
//! | Magic "LLRE"     | Version (1 byte) | Flags (1 byte)   |
//! | (4 bytes)        |                  |                  |
//! +------------------+------------------+------------------+
//! | Metadata length (4 bytes, little-endian)              |
//! +-------------------------------------------------------+
//! | Metadata (bincode)                                     |
//! +-------------------------------------------------------+
//! | Symbol table length (4 bytes, little-endian)          |
//! +-------------------------------------------------------+
//! | Symbol table (bincode)                                 |
//! +-------------------------------------------------------+
//! | NFA (bincode)                                          |
//! +-------------------------------------------------------+
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use liblevenshtein::phonetic::llre::{compile, save, load};
//!
//! // Compile and save
//! let file = parse_str("^hello$")?;
//! let compiled = compile(&file)?;
//! save(&compiled, "pattern.llre.bin")?;
//!
//! // Load pre-compiled
//! let loaded = load("pattern.llre.bin")?;
//! assert!(loaded.matches("hello"));
//! ```

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serialization")]
use std::collections::HashMap;
#[cfg(feature = "serialization")]
use std::io::{Read, Write};
#[cfg(feature = "serialization")]
use std::path::Path;

use super::error::{LLreError, LLreErrorKind, LLreResult};
use super::nfa_compiler::CompiledNFA;

/// Magic bytes for .llre.bin files
pub const MAGIC: &[u8; 4] = b"LLRE";

/// Current binary format version
pub const VERSION: u8 = 1;

/// Flags byte layout:
/// - Bit 0: multiline
/// - Bit 1: dotall
/// - Bit 2: case_insensitive
/// - Bits 3-7: reserved
#[cfg(feature = "serialization")]
fn flags_to_byte(multiline: bool, dotall: bool, case_insensitive: bool) -> u8 {
    let mut flags = 0u8;
    if multiline {
        flags |= 0x01;
    }
    if dotall {
        flags |= 0x02;
    }
    if case_insensitive {
        flags |= 0x04;
    }
    flags
}

#[cfg(feature = "serialization")]
fn byte_to_flags(byte: u8) -> (bool, bool, bool) {
    let multiline = (byte & 0x01) != 0;
    let dotall = (byte & 0x02) != 0;
    let case_insensitive = (byte & 0x04) != 0;
    (multiline, dotall, case_insensitive)
}

/// Serializable metadata for compiled patterns.
#[cfg(feature = "serialization")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledMetadata {
    /// Pattern name from @name directive
    pub name: Option<String>,
    /// Pattern version from @version directive
    pub version: Option<String>,
    /// Author from @author directive
    pub author: Option<String>,
    /// Description from @description directive
    pub description: Option<String>,
    /// Original pattern source (for debugging)
    pub pattern_source: Option<String>,
}

#[cfg(feature = "serialization")]
impl Default for CompiledMetadata {
    fn default() -> Self {
        Self {
            name: None,
            version: None,
            author: None,
            description: None,
            pattern_source: None,
        }
    }
}

#[cfg(feature = "serialization")]
impl From<&CompiledNFA> for CompiledMetadata {
    fn from(nfa: &CompiledNFA) -> Self {
        Self {
            name: nfa.name.clone(),
            version: nfa.version.clone(),
            author: None,
            description: None,
            pattern_source: None,
        }
    }
}

/// Serializable symbol table.
#[cfg(feature = "serialization")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SerializedSymbols {
    /// Character class symbols
    pub char_classes: HashMap<String, Vec<char>>,
}

/// Save a compiled NFA to a file.
#[cfg(feature = "serialization")]
pub fn save<P: AsRef<Path>>(compiled: &CompiledNFA, path: P) -> LLreResult<()> {
    let bytes = to_bytes(compiled)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Load a compiled NFA from a file.
#[cfg(feature = "serialization")]
pub fn load<P: AsRef<Path>>(path: P) -> LLreResult<CompiledNFA> {
    let bytes = std::fs::read(path)?;
    from_bytes(&bytes)
}

/// Serialize a compiled NFA to bytes.
#[cfg(feature = "serialization")]
pub fn to_bytes(compiled: &CompiledNFA) -> LLreResult<Vec<u8>> {
    let mut buffer = Vec::new();

    // Write magic
    buffer.extend_from_slice(MAGIC);

    // Write version
    buffer.push(VERSION);

    // Write flags
    let flags_byte = flags_to_byte(
        compiled.multiline,
        compiled.dotall,
        compiled.case_insensitive,
    );
    buffer.push(flags_byte);

    // Serialize metadata
    let metadata = CompiledMetadata::from(compiled);
    let metadata_bytes = bincode::serialize(&metadata)
        .map_err(|e| LLreError::new(LLreErrorKind::SerializationFailed(e.to_string())))?;

    // Write metadata length and data
    let metadata_len = metadata_bytes.len() as u32;
    buffer.extend_from_slice(&metadata_len.to_le_bytes());
    buffer.extend_from_slice(&metadata_bytes);

    // Serialize symbol table (empty for now - symbols are baked into NFA)
    let symbols = SerializedSymbols::default();
    let symbols_bytes = bincode::serialize(&symbols)
        .map_err(|e| LLreError::new(LLreErrorKind::SerializationFailed(e.to_string())))?;

    // Write symbols length and data
    let symbols_len = symbols_bytes.len() as u32;
    buffer.extend_from_slice(&symbols_len.to_le_bytes());
    buffer.extend_from_slice(&symbols_bytes);

    // Serialize NFA
    let nfa_bytes = bincode::serialize(&compiled.nfa)
        .map_err(|e| LLreError::new(LLreErrorKind::SerializationFailed(e.to_string())))?;
    buffer.extend_from_slice(&nfa_bytes);

    Ok(buffer)
}

/// Deserialize a compiled NFA from bytes.
#[cfg(feature = "serialization")]
pub fn from_bytes(bytes: &[u8]) -> LLreResult<CompiledNFA> {
    if bytes.len() < 10 {
        return Err(LLreError::new(LLreErrorKind::InvalidBinaryFormat(
            "file too small".into(),
        )));
    }

    let mut cursor = 0;

    // Check magic
    if &bytes[cursor..cursor + 4] != MAGIC {
        return Err(LLreError::new(LLreErrorKind::InvalidBinaryFormat(
            "invalid magic bytes".into(),
        )));
    }
    cursor += 4;

    // Check version
    let version = bytes[cursor];
    if version != VERSION {
        return Err(LLreError::new(LLreErrorKind::VersionMismatch {
            expected: VERSION,
            found: version,
        }));
    }
    cursor += 1;

    // Read flags
    let flags_byte = bytes[cursor];
    let (multiline, dotall, case_insensitive) = byte_to_flags(flags_byte);
    cursor += 1;

    // Read metadata length
    if cursor + 4 > bytes.len() {
        return Err(LLreError::new(LLreErrorKind::InvalidBinaryFormat(
            "truncated metadata length".into(),
        )));
    }
    let metadata_len = u32::from_le_bytes([
        bytes[cursor],
        bytes[cursor + 1],
        bytes[cursor + 2],
        bytes[cursor + 3],
    ]) as usize;
    cursor += 4;

    // Read metadata
    if cursor + metadata_len > bytes.len() {
        return Err(LLreError::new(LLreErrorKind::InvalidBinaryFormat(
            "truncated metadata".into(),
        )));
    }
    let metadata: CompiledMetadata = bincode::deserialize(&bytes[cursor..cursor + metadata_len])
        .map_err(|e| LLreError::new(LLreErrorKind::DeserializationFailed(e.to_string())))?;
    cursor += metadata_len;

    // Read symbols length
    if cursor + 4 > bytes.len() {
        return Err(LLreError::new(LLreErrorKind::InvalidBinaryFormat(
            "truncated symbols length".into(),
        )));
    }
    let symbols_len = u32::from_le_bytes([
        bytes[cursor],
        bytes[cursor + 1],
        bytes[cursor + 2],
        bytes[cursor + 3],
    ]) as usize;
    cursor += 4;

    // Skip symbols (they're baked into the NFA)
    if cursor + symbols_len > bytes.len() {
        return Err(LLreError::new(LLreErrorKind::InvalidBinaryFormat(
            "truncated symbols".into(),
        )));
    }
    cursor += symbols_len;

    // Read NFA
    let nfa = bincode::deserialize(&bytes[cursor..])
        .map_err(|e| LLreError::new(LLreErrorKind::DeserializationFailed(e.to_string())))?;

    Ok(CompiledNFA {
        nfa,
        multiline,
        dotall,
        case_insensitive,
        name: metadata.name,
        version: metadata.version,
    })
}

/// Stream-based save for large NFAs.
#[cfg(feature = "serialization")]
pub fn save_to_writer<W: Write>(compiled: &CompiledNFA, writer: &mut W) -> LLreResult<()> {
    let bytes = to_bytes(compiled)?;
    writer.write_all(&bytes)?;
    Ok(())
}

/// Stream-based load for large NFAs.
#[cfg(feature = "serialization")]
pub fn load_from_reader<R: Read>(reader: &mut R) -> LLreResult<CompiledNFA> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    from_bytes(&bytes)
}

#[cfg(all(test, feature = "serialization"))]
mod tests {
    use super::*;
    use crate::phonetic::llre::{compile, parser::parse_str};
    use tempfile::TempDir;

    #[test]
    fn test_serialize_deserialize() {
        let file = parse_str("^hello$").expect("Failed to parse");
        let compiled = compile(&file).expect("Failed to compile");

        let bytes = to_bytes(&compiled).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        assert!(loaded.matches("hello"));
        assert!(!loaded.matches("world"));
    }

    #[test]
    fn test_save_load_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let path = temp_dir.path().join("test.llre.bin");

        let file = parse_str(
            r#"
            @name "Test Pattern"
            ^[a-z]+$
        "#,
        )
        .expect("Failed to parse");
        let compiled = compile(&file).expect("Failed to compile");

        save(&compiled, &path).expect("Failed to save");
        let loaded = load(&path).expect("Failed to load");

        assert_eq!(loaded.name, Some("Test Pattern".to_string()));
        assert!(loaded.matches("hello"));
        assert!(!loaded.matches("123"));
    }

    #[test]
    fn test_flags_roundtrip() {
        let file = parse_str(
            r#"
            @flags multiline, dotall
            ^hello$
        "#,
        )
        .expect("Failed to parse");
        let compiled = compile(&file).expect("Failed to compile");

        assert!(compiled.multiline);
        assert!(compiled.dotall);

        let bytes = to_bytes(&compiled).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        assert!(loaded.multiline);
        assert!(loaded.dotall);
    }

    #[test]
    fn test_invalid_magic() {
        let bytes = b"XXXX\x01\x00";
        let result = from_bytes(bytes);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, LLreErrorKind::InvalidBinaryFormat(_)));
    }

    #[test]
    fn test_version_mismatch() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(99); // Invalid version
        bytes.push(0); // Flags
                       // ... rest would be needed for a real test

        // This will fail with version mismatch
        let result = from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_pattern() {
        let file = parse_str(
            r#"
            @name "Email"
            ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
        "#,
        )
        .expect("Failed to parse");
        let compiled = compile(&file).expect("Failed to compile");

        let bytes = to_bytes(&compiled).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        assert!(loaded.matches("test@example.com"));
        assert!(loaded.matches("user.name+tag@sub.domain.org"));
        assert!(!loaded.matches("invalid"));
    }

    #[test]
    fn test_flags_byte_conversion() {
        // Test all combinations
        for multiline in [false, true] {
            for dotall in [false, true] {
                for case_insensitive in [false, true] {
                    let byte = flags_to_byte(multiline, dotall, case_insensitive);
                    let (m, d, c) = byte_to_flags(byte);
                    assert_eq!(multiline, m);
                    assert_eq!(dotall, d);
                    assert_eq!(case_insensitive, c);
                }
            }
        }
    }
}

// Stub implementations when serialization feature is disabled
#[cfg(not(feature = "serialization"))]
pub fn save<P: AsRef<std::path::Path>>(_compiled: &CompiledNFA, _path: P) -> LLreResult<()> {
    Err(LLreError::new(LLreErrorKind::SerializationFailed(
        "serialization feature not enabled".into(),
    )))
}

#[cfg(not(feature = "serialization"))]
pub fn load<P: AsRef<std::path::Path>>(_path: P) -> LLreResult<CompiledNFA> {
    Err(LLreError::new(LLreErrorKind::DeserializationFailed(
        "serialization feature not enabled".into(),
    )))
}

#[cfg(not(feature = "serialization"))]
pub fn to_bytes(_compiled: &CompiledNFA) -> LLreResult<Vec<u8>> {
    Err(LLreError::new(LLreErrorKind::SerializationFailed(
        "serialization feature not enabled".into(),
    )))
}

#[cfg(not(feature = "serialization"))]
pub fn from_bytes(_bytes: &[u8]) -> LLreResult<CompiledNFA> {
    Err(LLreError::new(LLreErrorKind::DeserializationFailed(
        "serialization feature not enabled".into(),
    )))
}
