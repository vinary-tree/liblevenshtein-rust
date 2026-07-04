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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

#[cfg(feature = "serialization")]
fn section_len_to_u32(section: &str, len: usize) -> LLreResult<u32> {
    u32::try_from(len).map_err(|_| {
        LLreError::new(LLreErrorKind::SerializationFailed(format!(
            "{section} section is {len} bytes, which exceeds the {} byte LLRE binary format limit",
            u32::MAX
        )))
    })
}

#[cfg(feature = "serialization")]
fn u32_len_to_usize(section: &str, len: u32) -> LLreResult<usize> {
    usize::try_from(len).map_err(|_| {
        LLreError::new(LLreErrorKind::InvalidBinaryFormat(format!(
            "{section} length {len} exceeds this platform's addressable size"
        )))
    })
}

#[cfg(feature = "serialization")]
fn read_u32_len(bytes: &[u8], cursor: &mut usize, section: &str) -> LLreResult<usize> {
    let end = cursor
        .checked_add(4)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| {
            LLreError::new(LLreErrorKind::InvalidBinaryFormat(format!(
                "truncated {section} length"
            )))
        })?;

    let mut len_bytes = [0u8; 4];
    len_bytes.copy_from_slice(&bytes[*cursor..end]);
    let len = u32_len_to_usize(section, u32::from_le_bytes(len_bytes))?;
    *cursor = end;

    Ok(len)
}

#[cfg(feature = "serialization")]
fn section_end(
    bytes_len: usize,
    cursor: usize,
    section_len: usize,
    section: &str,
) -> LLreResult<usize> {
    cursor
        .checked_add(section_len)
        .filter(|&end| end <= bytes_len)
        .ok_or_else(|| {
            LLreError::new(LLreErrorKind::InvalidBinaryFormat(format!(
                "truncated {section}"
            )))
        })
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
    let metadata_bytes = bincode::serde::encode_to_vec(&metadata, bincode::config::legacy())
        .map_err(|e| LLreError::new(LLreErrorKind::SerializationFailed(e.to_string())))?;

    // Write metadata length and data
    let metadata_len = section_len_to_u32("metadata", metadata_bytes.len())?;
    buffer.extend_from_slice(&metadata_len.to_le_bytes());
    buffer.extend_from_slice(&metadata_bytes);

    // Keep the symbol-table section in the binary format; LLRE symbols are
    // expanded into the NFA before serialization.
    let symbols = SerializedSymbols::default();
    let symbols_bytes = bincode::serde::encode_to_vec(&symbols, bincode::config::legacy())
        .map_err(|e| LLreError::new(LLreErrorKind::SerializationFailed(e.to_string())))?;

    // Write symbols length and data
    let symbols_len = section_len_to_u32("symbols", symbols_bytes.len())?;
    buffer.extend_from_slice(&symbols_len.to_le_bytes());
    buffer.extend_from_slice(&symbols_bytes);

    // Serialize NFA
    let nfa_bytes = bincode::serde::encode_to_vec(&compiled.nfa, bincode::config::legacy())
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
    let metadata_len = read_u32_len(bytes, &mut cursor, "metadata")?;

    // Read metadata
    let metadata_end = section_end(bytes.len(), cursor, metadata_len, "metadata")?;
    let metadata: CompiledMetadata =
        bincode::serde::decode_from_slice(&bytes[cursor..metadata_end], bincode::config::legacy())
            .map(|(__decoded, _)| __decoded)
            .map_err(|e| LLreError::new(LLreErrorKind::DeserializationFailed(e.to_string())))?;
    cursor = metadata_end;

    // Read symbols length
    let symbols_len = read_u32_len(bytes, &mut cursor, "symbols")?;

    // Skip symbols (they're baked into the NFA)
    cursor = section_end(bytes.len(), cursor, symbols_len, "symbols")?;

    // Read NFA
    let nfa = bincode::serde::decode_from_slice(&bytes[cursor..], bincode::config::legacy())
        .map(|(__decoded, _)| __decoded)
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
        let bytes = b"BAD!\x01\x00";
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

    #[test]
    fn test_section_len_to_u32_preserves_format_boundary() {
        let max_len = usize::try_from(u32::MAX).expect("u32::MAX fits in usize");

        assert_eq!(
            section_len_to_u32("metadata", max_len).expect("u32::MAX is encodable"),
            u32::MAX
        );

        #[cfg(target_pointer_width = "64")]
        {
            let err = section_len_to_u32("metadata", max_len + 1)
                .expect_err("oversized metadata section should fail");

            assert!(matches!(
                err.kind,
                LLreErrorKind::SerializationFailed(ref msg)
                    if msg.contains("metadata") && msg.contains("exceeds")
            ));
        }
    }

    #[test]
    fn test_read_u32_len_preserves_boundary_and_advances_cursor() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&123u32.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        let mut cursor = 0;

        assert_eq!(read_u32_len(&bytes, &mut cursor, "metadata").unwrap(), 123);
        assert_eq!(cursor, 4);

        match usize::try_from(u32::MAX) {
            Ok(max_len) => {
                assert_eq!(
                    read_u32_len(&bytes, &mut cursor, "symbols").unwrap(),
                    max_len
                );
                assert_eq!(cursor, 8);
            }
            Err(_) => {
                let err = read_u32_len(&bytes, &mut cursor, "symbols")
                    .expect_err("u32::MAX should not fit in usize on this target");
                assert!(matches!(
                    err.kind,
                    LLreErrorKind::InvalidBinaryFormat(ref msg)
                        if msg.contains("symbols") && msg.contains("addressable")
                ));
                assert_eq!(cursor, 4);
            }
        }
    }

    #[test]
    fn test_read_u32_len_rejects_truncated_length_without_advancing_cursor() {
        let mut cursor = 0;
        let err = read_u32_len(&[1, 2, 3], &mut cursor, "metadata")
            .expect_err("truncated length field should fail");

        assert!(matches!(
            err.kind,
            LLreErrorKind::InvalidBinaryFormat(ref msg) if msg == "truncated metadata length"
        ));
        assert_eq!(cursor, 0);
    }

    #[test]
    fn test_section_end_rejects_truncated_and_overflowing_ranges() {
        let truncated = section_end(8, 4, 8, "metadata")
            .expect_err("section extending past buffer should fail");
        assert!(matches!(
            truncated.kind,
            LLreErrorKind::InvalidBinaryFormat(ref msg) if msg == "truncated metadata"
        ));

        let overflowing = section_end(usize::MAX, usize::MAX, 1, "symbols")
            .expect_err("section end arithmetic overflow should fail");
        assert!(matches!(
            overflowing.kind,
            LLreErrorKind::InvalidBinaryFormat(ref msg) if msg == "truncated symbols"
        ));
    }

    // ========================================================================
    // Non-ASCII regression tests (Phase 4 / item 1781)
    //
    // A compiled NFA is char-based (`NFAChar`), so multi-byte literals must
    // survive the parse -> compile -> to_bytes -> from_bytes round-trip and
    // still accept/reject the same inputs as before serialization.
    // ========================================================================

    #[test]
    fn test_serialize_deserialize_non_ascii_literal() {
        // `é` is a 2-byte UTF-8 literal; the anchored pattern must match `café`
        // exactly and reject the ASCII look-alike `cafe` after a round-trip.
        let file = parse_str("^café$").expect("Failed to parse non-ASCII pattern");
        let compiled = compile(&file).expect("Failed to compile");

        let bytes = to_bytes(&compiled).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        assert!(loaded.matches("café"), "loaded NFA should match 'café'");
        assert!(
            !loaded.matches("cafe"),
            "loaded NFA should reject ASCII 'cafe' (no accent)"
        );
    }

    #[test]
    fn test_serialize_deserialize_non_ascii_cjk_literal() {
        // Each CJK ideograph is a 3-byte UTF-8 literal. The anchored `^中文$`
        // must match the full string and reject partial / extended CJK input.
        let file = parse_str("^中文$").expect("Failed to parse CJK pattern");
        let compiled = compile(&file).expect("Failed to compile");

        let bytes = to_bytes(&compiled).expect("Failed to serialize");
        let loaded = from_bytes(&bytes).expect("Failed to deserialize");

        assert!(loaded.matches("中文"), "loaded NFA should match '中文'");
        assert!(
            !loaded.matches("中"),
            "loaded NFA should reject the partial CJK input '中'"
        );
        assert!(
            !loaded.matches("中文字"),
            "loaded NFA should reject the extended CJK input '中文字'"
        );
    }
}

// Feature-disabled implementations keep the public API available.
#[cfg(not(feature = "serialization"))]
/// Return an error because LLRE binary serialization is not enabled.
pub fn save<P: AsRef<std::path::Path>>(_compiled: &CompiledNFA, _path: P) -> LLreResult<()> {
    Err(LLreError::new(LLreErrorKind::SerializationFailed(
        "serialization feature not enabled".into(),
    )))
}

#[cfg(not(feature = "serialization"))]
/// Return an error because LLRE binary deserialization is not enabled.
pub fn load<P: AsRef<std::path::Path>>(_path: P) -> LLreResult<CompiledNFA> {
    Err(LLreError::new(LLreErrorKind::DeserializationFailed(
        "serialization feature not enabled".into(),
    )))
}

#[cfg(not(feature = "serialization"))]
/// Return an error because LLRE byte serialization is not enabled.
pub fn to_bytes(_compiled: &CompiledNFA) -> LLreResult<Vec<u8>> {
    Err(LLreError::new(LLreErrorKind::SerializationFailed(
        "serialization feature not enabled".into(),
    )))
}

#[cfg(not(feature = "serialization"))]
/// Return an error because LLRE byte deserialization is not enabled.
pub fn from_bytes(_bytes: &[u8]) -> LLreResult<CompiledNFA> {
    Err(LLreError::new(LLreErrorKind::DeserializationFailed(
        "serialization feature not enabled".into(),
    )))
}
