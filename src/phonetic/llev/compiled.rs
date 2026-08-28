//! Binary serialization for compiled rule sets.
//!
//! This module provides AOT (ahead-of-time) compilation for phonetic rule sets.
//! Rules can be compiled to a binary format for faster loading at runtime.
//!
//! # File Format
//!
//! The compiled format uses bincode for efficient binary serialization with
//! a version header for forward compatibility:
//!
//! ```text
//! +----------------+------------------+
//! | Magic (4 bytes)| Version (1 byte) |
//! +----------------+------------------+
//! | Serialized RuleSet/RuleSetChar   |
//! +-----------------------------------+
//! ```
//!
//! Magic bytes: `LLEV` (0x4C4C4556)
//! Version: 1 (current)
//!
//! # Usage
//!
//! ```rust,no_run
//! use liblevenshtein::phonetic::llev::{RuleSetChar, parse_str, compiled};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse and convert rules
//! let file = parse_str("ph -> f; gh -> ;")?;
//! let ruleset = RuleSetChar::from_llev(&file)?;
//!
//! // Save to binary
//! compiled::save_char(&ruleset, "rules.llev.bin")?;
//!
//! // Load from binary (faster)
//! let loaded = compiled::load_char("rules.llev.bin")?;
//! assert_eq!(loaded.apply("phone"), "fone");
//! # Ok(())
//! # }
//! ```
//!
//! # Feature Flag
//!
//! This module requires the `serialization` feature:
//!
//! ```toml
//! [dependencies]
//! liblevenshtein = { version = "...", features = ["serialization"] }
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::error::{LLevError, LLevErrorKind, LLevResult};
use super::ruleset::{RuleSet, RuleSetChar};

/// Magic bytes identifying a compiled LLev file.
const MAGIC: [u8; 4] = *b"LLEV";

/// Current file format version.
const VERSION: u8 = 1;

/// Header size in bytes.
const HEADER_SIZE: usize = 5;

// ============================================================================
// Byte-level Operations
// ============================================================================

/// Save a byte-level rule set to a binary file.
///
/// # Arguments
///
/// * `ruleset` - The rule set to save
/// * `path` - Output file path
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be created
/// - Serialization fails
///
/// # Example
///
/// ```rust,no_run
/// use liblevenshtein::phonetic::llev::{parse_str, save, RuleSet};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let file = parse_str("ph -> f;")?;
/// let ruleset = RuleSet::from_llev(&file)?;
/// save(&ruleset, "rules.llev.bin")?;
/// # Ok(())
/// # }
/// ```
pub fn save<P: AsRef<Path>>(ruleset: &RuleSet, path: P) -> LLevResult<()> {
    let file = File::create(path.as_ref()).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to create file: {}",
            e
        )))
    })?;
    let mut writer = BufWriter::new(file);

    // Write header
    writer.write_all(&MAGIC).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to write magic bytes: {}",
            e
        )))
    })?;
    writer.write_all(&[VERSION]).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to write version: {}",
            e
        )))
    })?;

    // Serialize rule set
    let encoded =
        bincode::serde::encode_to_vec(ruleset, bincode::config::legacy()).map_err(|e| {
            LLevError::new(LLevErrorKind::IoError(format!(
                "Failed to serialize ruleset: {}",
                e
            )))
        })?;
    writer.write_all(&encoded).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to write ruleset: {}",
            e
        )))
    })?;

    writer.flush().map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to flush writer: {}",
            e
        )))
    })?;

    Ok(())
}

/// Load a byte-level rule set from a binary file.
///
/// # Arguments
///
/// * `path` - Input file path
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - The file is not a valid compiled LLev file
/// - The file version is not supported
/// - Deserialization fails
///
/// # Example
///
/// ```rust,no_run
/// use liblevenshtein::phonetic::llev::load;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ruleset = load("rules.llev.bin")?;
/// assert!(!ruleset.is_empty());
/// # Ok(())
/// # }
/// ```
pub fn load<P: AsRef<Path>>(path: P) -> LLevResult<RuleSet> {
    let file = File::open(path.as_ref()).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to open file: {}",
            e
        )))
    })?;
    let mut reader = BufReader::new(file);

    // Read and verify header
    let mut header = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to read header: {}",
            e
        )))
    })?;

    // Verify magic bytes
    if header[0..4] != MAGIC {
        return Err(LLevError::new(LLevErrorKind::InvalidFormat(
            "Invalid magic bytes: not a compiled LLev file".into(),
        )));
    }

    // Check version
    let version = header[4];
    if version != VERSION {
        return Err(LLevError::new(LLevErrorKind::InvalidFormat(format!(
            "Unsupported version: {} (expected {})",
            version, VERSION
        ))));
    }

    // Read and deserialize rule set
    let mut data = Vec::new();
    reader.read_to_end(&mut data).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to read ruleset data: {}",
            e
        )))
    })?;

    bincode::serde::decode_from_slice(&data, bincode::config::legacy())
        .map(|(__decoded, _)| __decoded)
        .map_err(|e| {
            LLevError::new(LLevErrorKind::IoError(format!(
                "Failed to deserialize ruleset: {}",
                e
            )))
        })
}

// ============================================================================
// Character-level Operations
// ============================================================================

/// Save a character-level rule set to a binary file.
///
/// # Arguments
///
/// * `ruleset` - The rule set to save
/// * `path` - Output file path
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be created
/// - Serialization fails
///
/// # Example
///
/// ```rust,no_run
/// use liblevenshtein::phonetic::llev::{parse_str, save_char, RuleSetChar};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let file = parse_str("ph -> f;")?;
/// let ruleset = RuleSetChar::from_llev(&file)?;
/// save_char(&ruleset, "rules.llev.bin")?;
/// # Ok(())
/// # }
/// ```
pub fn save_char<P: AsRef<Path>>(ruleset: &RuleSetChar, path: P) -> LLevResult<()> {
    let file = File::create(path.as_ref()).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to create file: {}",
            e
        )))
    })?;
    let mut writer = BufWriter::new(file);

    // Write header
    writer.write_all(&MAGIC).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to write magic bytes: {}",
            e
        )))
    })?;
    writer.write_all(&[VERSION]).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to write version: {}",
            e
        )))
    })?;

    // Serialize rule set
    let encoded =
        bincode::serde::encode_to_vec(ruleset, bincode::config::legacy()).map_err(|e| {
            LLevError::new(LLevErrorKind::IoError(format!(
                "Failed to serialize ruleset: {}",
                e
            )))
        })?;
    writer.write_all(&encoded).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to write ruleset: {}",
            e
        )))
    })?;

    writer.flush().map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to flush writer: {}",
            e
        )))
    })?;

    Ok(())
}

/// Load a character-level rule set from a binary file.
///
/// # Arguments
///
/// * `path` - Input file path
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - The file is not a valid compiled LLev file
/// - The file version is not supported
/// - Deserialization fails
///
/// # Example
///
/// ```rust,no_run
/// use liblevenshtein::phonetic::llev::load_char;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ruleset = load_char("rules.llev.bin")?;
/// assert_eq!(ruleset.apply("phone"), "fone");
/// # Ok(())
/// # }
/// ```
pub fn load_char<P: AsRef<Path>>(path: P) -> LLevResult<RuleSetChar> {
    let file = File::open(path.as_ref()).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to open file: {}",
            e
        )))
    })?;
    let mut reader = BufReader::new(file);

    // Read and verify header
    let mut header = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to read header: {}",
            e
        )))
    })?;

    // Verify magic bytes
    if header[0..4] != MAGIC {
        return Err(LLevError::new(LLevErrorKind::InvalidFormat(
            "Invalid magic bytes: not a compiled LLev file".into(),
        )));
    }

    // Check version
    let version = header[4];
    if version != VERSION {
        return Err(LLevError::new(LLevErrorKind::InvalidFormat(format!(
            "Unsupported version: {} (expected {})",
            version, VERSION
        ))));
    }

    // Read and deserialize rule set
    let mut data = Vec::new();
    reader.read_to_end(&mut data).map_err(|e| {
        LLevError::new(LLevErrorKind::IoError(format!(
            "Failed to read ruleset data: {}",
            e
        )))
    })?;

    bincode::serde::decode_from_slice(&data, bincode::config::legacy())
        .map(|(__decoded, _)| __decoded)
        .map_err(|e| {
            LLevError::new(LLevErrorKind::IoError(format!(
                "Failed to deserialize ruleset: {}",
                e
            )))
        })
}

// ============================================================================
// Byte Vector Operations
// ============================================================================

/// Serialize a byte-level rule set to a byte vector.
///
/// Useful for embedding compiled rules or network transmission.
///
/// # Example
///
/// ```rust
/// use liblevenshtein::phonetic::llev::{
///     from_bytes, parse_str, to_bytes, RuleSet,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let file = parse_str("ph -> f;")?;
/// let ruleset = RuleSet::from_llev(&file)?;
/// let data = to_bytes(&ruleset)?;
/// // ... store or transmit data ...
/// let loaded = from_bytes(&data)?;
/// assert_eq!(loaded.len(), ruleset.len());
/// # Ok(())
/// # }
/// ```
pub fn to_bytes(ruleset: &RuleSet) -> LLevResult<Vec<u8>> {
    let mut data = Vec::with_capacity(HEADER_SIZE + 1024);

    // Write header
    data.extend_from_slice(&MAGIC);
    data.push(VERSION);

    // Serialize rule set
    let encoded =
        bincode::serde::encode_to_vec(ruleset, bincode::config::legacy()).map_err(|e| {
            LLevError::new(LLevErrorKind::IoError(format!(
                "Failed to serialize ruleset: {}",
                e
            )))
        })?;
    data.extend_from_slice(&encoded);

    Ok(data)
}

/// Deserialize a byte-level rule set from a byte slice.
///
/// # Errors
///
/// Returns an error if the data is not a valid compiled ruleset.
pub fn from_bytes(data: &[u8]) -> LLevResult<RuleSet> {
    if data.len() < HEADER_SIZE {
        return Err(LLevError::new(LLevErrorKind::InvalidFormat(
            "Data too short to be a valid compiled ruleset".into(),
        )));
    }

    // Verify magic bytes
    if data[0..4] != MAGIC {
        return Err(LLevError::new(LLevErrorKind::InvalidFormat(
            "Invalid magic bytes: not a compiled LLev file".into(),
        )));
    }

    // Check version
    let version = data[4];
    if version != VERSION {
        return Err(LLevError::new(LLevErrorKind::InvalidFormat(format!(
            "Unsupported version: {} (expected {})",
            version, VERSION
        ))));
    }

    // Deserialize rule set
    bincode::serde::decode_from_slice(&data[HEADER_SIZE..], bincode::config::legacy())
        .map(|(__decoded, _)| __decoded)
        .map_err(|e| {
            LLevError::new(LLevErrorKind::IoError(format!(
                "Failed to deserialize ruleset: {}",
                e
            )))
        })
}

/// Serialize a character-level rule set to a byte vector.
///
/// Useful for embedding compiled rules or network transmission.
pub fn to_bytes_char(ruleset: &RuleSetChar) -> LLevResult<Vec<u8>> {
    let mut data = Vec::with_capacity(HEADER_SIZE + 1024);

    // Write header
    data.extend_from_slice(&MAGIC);
    data.push(VERSION);

    // Serialize rule set
    let encoded =
        bincode::serde::encode_to_vec(ruleset, bincode::config::legacy()).map_err(|e| {
            LLevError::new(LLevErrorKind::IoError(format!(
                "Failed to serialize ruleset: {}",
                e
            )))
        })?;
    data.extend_from_slice(&encoded);

    Ok(data)
}

/// Deserialize a character-level rule set from a byte slice.
///
/// # Errors
///
/// Returns an error if the data is not a valid compiled ruleset.
pub fn from_bytes_char(data: &[u8]) -> LLevResult<RuleSetChar> {
    if data.len() < HEADER_SIZE {
        return Err(LLevError::new(LLevErrorKind::InvalidFormat(
            "Data too short to be a valid compiled ruleset".into(),
        )));
    }

    // Verify magic bytes
    if data[0..4] != MAGIC {
        return Err(LLevError::new(LLevErrorKind::InvalidFormat(
            "Invalid magic bytes: not a compiled LLev file".into(),
        )));
    }

    // Check version
    let version = data[4];
    if version != VERSION {
        return Err(LLevError::new(LLevErrorKind::InvalidFormat(format!(
            "Unsupported version: {} (expected {})",
            version, VERSION
        ))));
    }

    // Deserialize rule set
    bincode::serde::decode_from_slice(&data[HEADER_SIZE..], bincode::config::legacy())
        .map(|(__decoded, _)| __decoded)
        .map_err(|e| {
            LLevError::new(LLevErrorKind::IoError(format!(
                "Failed to deserialize ruleset: {}",
                e
            )))
        })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phonetic::llev::parser::parse_str;
    use tempfile::TempDir;

    #[test]
    fn test_save_load_roundtrip() {
        let file = parse_str("ph -> f; gh -> ;").expect("parse failed");
        let ruleset = RuleSet::from_llev(&file).expect("conversion failed");

        let dir = TempDir::new().expect("failed to create scratch dir");
        let path = dir.path().join("test.llev.bin");

        save(&ruleset, &path).expect("save failed");
        let loaded = load(&path).expect("load failed");

        assert_eq!(loaded.rules.len(), 2);
    }

    #[test]
    fn test_save_load_char_roundtrip() {
        let file = parse_str("ph -> f; gh -> ;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        let dir = TempDir::new().expect("failed to create scratch dir");
        let path = dir.path().join("test.llev.bin");

        save_char(&ruleset, &path).expect("save failed");
        let loaded = load_char(&path).expect("load failed");

        assert_eq!(loaded.rules.len(), 2);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let file = parse_str("ph -> f; gh -> ; c -> s / _[ei];").expect("parse failed");
        let ruleset = RuleSet::from_llev(&file).expect("conversion failed");

        let data = to_bytes(&ruleset).expect("to_bytes failed");
        let loaded = from_bytes(&data).expect("from_bytes failed");

        assert_eq!(loaded.rules.len(), 3);
    }

    #[test]
    fn test_bytes_char_roundtrip() {
        let file = parse_str("ph -> f; gh -> ; c -> s / _[ei];").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        let data = to_bytes_char(&ruleset).expect("to_bytes_char failed");
        let loaded = from_bytes_char(&data).expect("from_bytes_char failed");

        assert_eq!(loaded.rules.len(), 3);
    }

    #[test]
    fn test_invalid_magic() {
        let data = vec![0x00, 0x00, 0x00, 0x00, 0x01, 0x00];
        let result = from_bytes(&data);

        assert!(result.is_err());
        match result.unwrap_err().kind {
            LLevErrorKind::InvalidFormat(msg) => {
                assert!(msg.contains("magic"));
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_invalid_version() {
        let data = vec![b'L', b'L', b'E', b'V', 0xFF, 0x00];
        let result = from_bytes(&data);

        assert!(result.is_err());
        match result.unwrap_err().kind {
            LLevErrorKind::InvalidFormat(msg) => {
                assert!(msg.contains("version"));
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_data_too_short() {
        let data = vec![b'L', b'L', b'E'];
        let result = from_bytes(&data);

        assert!(result.is_err());
        match result.unwrap_err().kind {
            LLevErrorKind::InvalidFormat(msg) => {
                assert!(msg.contains("too short"));
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_metadata_preserved() {
        let file = parse_str(
            r#"
            @name "Test Rules"
            @version "1.0"
            ph -> f;
            "#,
        )
        .expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        let data = to_bytes_char(&ruleset).expect("to_bytes_char failed");
        let loaded = from_bytes_char(&data).expect("from_bytes_char failed");

        assert_eq!(loaded.name, Some("Test Rules".to_string()));
        assert_eq!(loaded.version, Some("1.0".to_string()));
    }

    #[test]
    fn test_context_preserved() {
        let file = parse_str("c -> s / _[ei];").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");

        let data = to_bytes_char(&ruleset).expect("to_bytes_char failed");
        let loaded = from_bytes_char(&data).expect("from_bytes_char failed");

        assert_eq!(loaded.rules.len(), 1);
        match &loaded.rules[0].context {
            crate::phonetic::types::ContextChar::BeforeVowel(chars) => {
                assert_eq!(chars.len(), 2);
                assert!(chars.contains(&'e'));
                assert!(chars.contains(&'i'));
            }
            _ => panic!("Expected BeforeVowel context"),
        }
    }

    #[test]
    fn test_file_not_found() {
        let result = load("/nonexistent/path/to/file.llev.bin");

        assert!(result.is_err());
        match result.unwrap_err().kind {
            LLevErrorKind::IoError(msg) => {
                assert!(msg.contains("Failed to open"));
            }
            _ => panic!("Expected IoError"),
        }
    }

    // ========================================================================
    // Non-ASCII regression tests (Phase 4 / item 1781)
    //
    // A char-level rule set (`RuleSetChar`) must preserve rules whose pattern or
    // replacement contains multi-byte UTF-8 characters across the bincode
    // round-trip, keeping both the rule count and the transformation behavior.
    // ========================================================================

    #[test]
    fn test_bytes_char_roundtrip_non_ascii_rule() {
        // The rule pattern `é` is a 2-byte UTF-8 literal. Round-tripping the
        // ruleset must preserve the rule and its rewriting behavior.
        let file = parse_str("é -> e;").expect("parse failed");
        let ruleset = RuleSetChar::from_llev(&file).expect("conversion failed");
        assert_eq!(ruleset.rules.len(), 1, "expected exactly one parsed rule");

        // Capture pre-serialization behavior to compare after the round-trip.
        let before_accent = ruleset.apply("é");
        let before_word = ruleset.apply("café");

        let data = to_bytes_char(&ruleset).expect("to_bytes_char failed");
        let loaded = from_bytes_char(&data).expect("from_bytes_char failed");

        // Rule count and behavior must both survive serialization.
        assert_eq!(loaded.rules.len(), ruleset.rules.len());
        assert_eq!(loaded.apply("é"), before_accent);
        assert_eq!(loaded.apply("café"), before_word);
        // The accented character must actually be rewritten to its ASCII form.
        assert_eq!(loaded.apply("é"), "e");
        assert_eq!(loaded.apply("café"), "cafe");
    }
}
