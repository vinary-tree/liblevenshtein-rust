//! Node Serialization for Persistent ART (Character-Level)
//!
//! This module provides binary serialization and deserialization for char ART nodes.
//! The format is designed to be:
//! - **Compact**: Minimize disk space usage
//! - **Fast**: Efficient encoding/decoding with minimal allocations
//! - **Versioned**: Support future format evolution
//! - **Unicode-aware**: Proper handling of 4-byte character keys
//!
//! # Serialization Format
//!
//! All nodes share a common header followed by type-specific data:
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────┐
//! │ SerializedCharNodeHeader (16 bytes)                                │
//! ├───────────┬───────────┬───────────┬───────────┬────────────────────┤
//! │ magic[4]  │ version   │ node_type │ flags     │ reserved[2]        │
//! │ "ARC\0"   │ u8        │ u8        │ u8        │ [u8; 2]            │
//! ├───────────┴───────────┴───────────┴───────────┴────────────────────┤
//! │ num_children: u16     │ prefix_len: u8        │ _padding: u8       │
//! ├───────────────────────┴───────────────────────┴────────────────────┤
//! │ data_size: u32 (size of type-specific data)                        │
//! └────────────────────────────────────────────────────────────────────┘
//! │ CharCompressedPrefix (24 bytes, if prefix_len > 0)                 │
//! └────────────────────────────────────────────────────────────────────┘
//! │ Type-specific data (variable size)                                 │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Type-Specific Layouts
//!
//! ## CharNode4
//! ```text
//! │ keys: [u32; 4]        │ 16 bytes                                   │
//! │ children: [u64; 4]    │ 32 bytes (SwizzledPtr as u64)              │
//! │ value_ptr: u64        │ 8 bytes                                    │
//! Total: 56 bytes + header
//! ```
//!
//! ## CharNode16
//! ```text
//! │ keys: [u32; 16]       │ 64 bytes                                   │
//! │ children: [u64; 16]   │ 128 bytes (SwizzledPtr as u64)             │
//! │ value_ptr: u64        │ 8 bytes                                    │
//! Total: 200 bytes + header
//! ```
//!
//! ## CharNode48
//! ```text
//! │ keys: [u32; 48]       │ 192 bytes (sorted for binary search)       │
//! │ children: [u64; 48]   │ 384 bytes (SwizzledPtr as u64)             │
//! │ value_ptr: u64        │ 8 bytes                                    │
//! Total: 584 bytes + header
//! ```
//!
//! ## CharBucket
//! ```text
//! │ num_entries: u32      │ 4 bytes                                    │
//! │ value_ptr: u64        │ 8 bytes                                    │
//! │ entries: [(u32, u64)] │ 12 bytes × num_entries                     │
//! Total: 12 + 12*num_entries bytes + header
//! ```

use std::io::{Read, Write};

use crate::dictionary::persistent_artrie::error::{PersistentARTrieError, Result};
use crate::dictionary::persistent_artrie::swizzled_ptr::SwizzledPtr;

use super::nodes::{
    CharBucket, CharCompressedPrefix, CharNode, CharNode16, CharNode4, CharNode48,
    CharNodeHeader, CHAR_MAX_PREFIX_LEN, flags,
};

/// Helper to convert io::Error to PersistentARTrieError for serialization operations
fn io_err(e: std::io::Error) -> PersistentARTrieError {
    PersistentARTrieError::io_error("char serialization", "<buffer>", e)
}

/// Magic bytes identifying a char ART node in the serialized format
pub const CHAR_NODE_MAGIC: [u8; 4] = *b"ARC\0"; // ART + Char

/// Current serialization format version for char nodes
pub const CHAR_FORMAT_VERSION: u8 = 2;

/// Serialized header size in bytes
pub const CHAR_SERIALIZED_HEADER_SIZE: usize = 16;

/// Char node type discriminants for serialization
pub mod char_node_types {
    pub const CHARNODE4: u8 = 104;
    pub const CHARNODE16: u8 = 116;
    pub const CHARNODE48: u8 = 148;
    pub const CHARBUCKET: u8 = 101;
}

/// Serialized char node header (fixed 16 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SerializedCharNodeHeader {
    /// Magic bytes "ARC\0"
    pub magic: [u8; 4],
    /// Format version
    pub version: u8,
    /// Node type (104, 116, 148, 101)
    pub node_type: u8,
    /// Node flags (is_final, is_dirty, is_leaf)
    pub flags: u8,
    /// Reserved for future use
    pub reserved: u8,
    /// Number of children
    pub num_children: u16,
    /// Compressed prefix length (0-6 chars)
    pub prefix_len: u8,
    /// Padding for alignment
    pub _padding: u8,
    /// Size of the type-specific data following this header
    pub data_size: u32,
}

impl SerializedCharNodeHeader {
    /// Create a header from a CharNodeHeader
    pub fn from_node_header(header: &CharNodeHeader, data_size: u32) -> Self {
        Self {
            magic: CHAR_NODE_MAGIC,
            version: CHAR_FORMAT_VERSION,
            node_type: header.node_type,
            flags: header.flags,
            reserved: 0,
            num_children: header.num_children,
            prefix_len: header.prefix_len,
            _padding: 0,
            data_size,
        }
    }

    /// Convert to a CharNodeHeader
    pub fn to_node_header(&self) -> CharNodeHeader {
        CharNodeHeader {
            node_type: self.node_type,
            prefix_len: self.prefix_len,
            flags: self.flags,
            _padding: 0,
            num_children: self.num_children,
            _padding2: [0; 2],
            version: 0, // Version is runtime-only
        }
    }

    /// Validate the header
    pub fn validate(&self) -> Result<()> {
        if self.magic != CHAR_NODE_MAGIC {
            return Err(PersistentARTrieError::InvalidMagic {
                expected: u64::from_le_bytes([
                    CHAR_NODE_MAGIC[0],
                    CHAR_NODE_MAGIC[1],
                    CHAR_NODE_MAGIC[2],
                    CHAR_NODE_MAGIC[3],
                    0,
                    0,
                    0,
                    0,
                ]),
                found: u64::from_le_bytes([
                    self.magic[0],
                    self.magic[1],
                    self.magic[2],
                    self.magic[3],
                    0,
                    0,
                    0,
                    0,
                ]),
            });
        }
        if self.version > CHAR_FORMAT_VERSION {
            return Err(PersistentARTrieError::UnsupportedVersion {
                max_supported: CHAR_FORMAT_VERSION as u32,
                found: self.version as u32,
            });
        }
        match self.node_type {
            char_node_types::CHARNODE4
            | char_node_types::CHARNODE16
            | char_node_types::CHARNODE48
            | char_node_types::CHARBUCKET => {}
            _ => {
                return Err(PersistentARTrieError::corrupted(format!(
                    "invalid char node type: {}",
                    self.node_type
                )));
            }
        }
        if self.prefix_len as usize > CHAR_MAX_PREFIX_LEN {
            return Err(PersistentARTrieError::corrupted(format!(
                "prefix length {} exceeds maximum {}",
                self.prefix_len, CHAR_MAX_PREFIX_LEN
            )));
        }
        Ok(())
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> [u8; CHAR_SERIALIZED_HEADER_SIZE] {
        let mut bytes = [0u8; CHAR_SERIALIZED_HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4] = self.version;
        bytes[5] = self.node_type;
        bytes[6] = self.flags;
        bytes[7] = self.reserved;
        bytes[8..10].copy_from_slice(&self.num_children.to_le_bytes());
        bytes[10] = self.prefix_len;
        bytes[11] = self._padding;
        bytes[12..16].copy_from_slice(&self.data_size.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8; CHAR_SERIALIZED_HEADER_SIZE]) -> Self {
        Self {
            magic: [bytes[0], bytes[1], bytes[2], bytes[3]],
            version: bytes[4],
            node_type: bytes[5],
            flags: bytes[6],
            reserved: bytes[7],
            num_children: u16::from_le_bytes([bytes[8], bytes[9]]),
            prefix_len: bytes[10],
            _padding: bytes[11],
            data_size: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        }
    }
}

/// Calculate the serialized size of a char node
pub fn char_serialized_size(node: &CharNode) -> usize {
    CHAR_SERIALIZED_HEADER_SIZE + char_prefix_size(node) + char_node_data_size(node)
}

fn char_prefix_size(node: &CharNode) -> usize {
    if node.header().prefix_len > 0 {
        CHAR_MAX_PREFIX_LEN * 4 // 6 chars × 4 bytes = 24 bytes
    } else {
        0
    }
}

fn char_node_data_size(node: &CharNode) -> usize {
    match node {
        // CharNode4: 4 keys × 4 + 4 children × 8 + value_ptr × 8 = 56
        CharNode::N4(_) => 4 * 4 + 4 * 8 + 8,
        // CharNode16: 16 keys × 4 + 16 children × 8 + value_ptr × 8 = 200
        CharNode::N16(_) => 16 * 4 + 16 * 8 + 8,
        // CharNode48: 48 keys × 4 + 48 children × 8 + value_ptr × 8 = 584
        CharNode::N48(_) => 48 * 4 + 48 * 8 + 8,
        // CharBucket: num_entries × 4 + value_ptr × 8 + entries × (4 + 8) = 12 + 12n
        CharNode::Bucket(n) => 4 + 8 + n.entries.len() * 12,
    }
}

/// Serialize a CharNode to a writer
pub fn serialize_char_node<W: Write>(node: &CharNode, writer: &mut W) -> Result<usize> {
    let data_size = char_prefix_size(node) + char_node_data_size(node);
    let header = SerializedCharNodeHeader::from_node_header(node.header(), data_size as u32);

    // Write header
    writer.write_all(&header.to_bytes()).map_err(io_err)?;

    // Write prefix if present
    if node.header().prefix_len > 0 {
        let prefix = node.prefix();
        for &c in &prefix.chars {
            writer.write_all(&c.to_le_bytes()).map_err(io_err)?;
        }
    }

    // Write type-specific data
    match node {
        CharNode::N4(n) => serialize_charnode4(n, writer)?,
        CharNode::N16(n) => serialize_charnode16(n, writer)?,
        CharNode::N48(n) => serialize_charnode48(n, writer)?,
        CharNode::Bucket(n) => serialize_charbucket(n, writer)?,
    }

    Ok(CHAR_SERIALIZED_HEADER_SIZE + data_size)
}

fn serialize_charnode4<W: Write>(node: &CharNode4, writer: &mut W) -> Result<()> {
    // Write keys (4 × u32)
    for key in &node.keys {
        writer.write_all(&key.to_le_bytes()).map_err(io_err)?;
    }

    // Write children as u64
    for child in &node.children {
        let raw = child.to_raw();
        writer.write_all(&raw.to_le_bytes()).map_err(io_err)?;
    }

    // Write value_ptr
    let value_raw = node.value_ptr.to_raw();
    writer.write_all(&value_raw.to_le_bytes()).map_err(io_err)?;

    Ok(())
}

fn serialize_charnode16<W: Write>(node: &CharNode16, writer: &mut W) -> Result<()> {
    // Write keys (16 × u32)
    for key in &node.keys {
        writer.write_all(&key.to_le_bytes()).map_err(io_err)?;
    }

    // Write children as u64
    for child in &node.children {
        let raw = child.to_raw();
        writer.write_all(&raw.to_le_bytes()).map_err(io_err)?;
    }

    // Write value_ptr
    let value_raw = node.value_ptr.to_raw();
    writer.write_all(&value_raw.to_le_bytes()).map_err(io_err)?;

    Ok(())
}

fn serialize_charnode48<W: Write>(node: &CharNode48, writer: &mut W) -> Result<()> {
    // Write keys (48 × u32, sorted)
    for key in &node.keys {
        writer.write_all(&key.to_le_bytes()).map_err(io_err)?;
    }

    // Write children as u64
    for child in &node.children {
        let raw = child.to_raw();
        writer.write_all(&raw.to_le_bytes()).map_err(io_err)?;
    }

    // Write value_ptr
    let value_raw = node.value_ptr.to_raw();
    writer.write_all(&value_raw.to_le_bytes()).map_err(io_err)?;

    Ok(())
}

fn serialize_charbucket<W: Write>(node: &CharBucket, writer: &mut W) -> Result<()> {
    // Write number of entries
    let num_entries = node.entries.len() as u32;
    writer
        .write_all(&num_entries.to_le_bytes())
        .map_err(io_err)?;

    // Write value_ptr
    let value_raw = node.value_ptr.to_raw();
    writer.write_all(&value_raw.to_le_bytes()).map_err(io_err)?;

    // Write entries as (key: u32, child: u64) pairs
    // Sort entries for deterministic serialization
    let mut entries: Vec<_> = node.entries.iter().collect();
    entries.sort_by_key(|&(k, _)| *k);

    for (&key, child) in entries {
        writer.write_all(&key.to_le_bytes()).map_err(io_err)?;
        let child_raw = child.to_raw();
        writer.write_all(&child_raw.to_le_bytes()).map_err(io_err)?;
    }

    Ok(())
}

/// Deserialize a CharNode from a reader
pub fn deserialize_char_node<R: Read>(reader: &mut R) -> Result<CharNode> {
    // Read and validate header
    let mut header_bytes = [0u8; CHAR_SERIALIZED_HEADER_SIZE];
    reader.read_exact(&mut header_bytes).map_err(io_err)?;
    let header = SerializedCharNodeHeader::from_bytes(&header_bytes);
    header.validate()?;

    // Read prefix if present
    let prefix = if header.prefix_len > 0 {
        let mut chars = [0u32; CHAR_MAX_PREFIX_LEN];
        for c in &mut chars {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes).map_err(io_err)?;
            *c = u32::from_le_bytes(bytes);
        }
        CharCompressedPrefix { chars }
    } else {
        CharCompressedPrefix::empty()
    };

    // Deserialize type-specific data
    match header.node_type {
        char_node_types::CHARNODE4 => deserialize_charnode4(reader, &header, prefix),
        char_node_types::CHARNODE16 => deserialize_charnode16(reader, &header, prefix),
        char_node_types::CHARNODE48 => deserialize_charnode48(reader, &header, prefix),
        char_node_types::CHARBUCKET => deserialize_charbucket(reader, &header, prefix),
        _ => Err(PersistentARTrieError::corrupted(format!(
            "invalid char node type: {}",
            header.node_type
        ))),
    }
}

fn deserialize_charnode4<R: Read>(
    reader: &mut R,
    header: &SerializedCharNodeHeader,
    prefix: CharCompressedPrefix,
) -> Result<CharNode> {
    let mut node = CharNode4::new();
    node.header = header.to_node_header();
    node.prefix = prefix;

    // Read keys
    for key in &mut node.keys {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes).map_err(io_err)?;
        *key = u32::from_le_bytes(bytes);
    }

    // Read children
    for child in &mut node.children {
        let mut raw_bytes = [0u8; 8];
        reader.read_exact(&mut raw_bytes).map_err(io_err)?;
        *child = SwizzledPtr::from_raw(u64::from_le_bytes(raw_bytes));
    }

    // Read value_ptr
    let mut value_bytes = [0u8; 8];
    reader.read_exact(&mut value_bytes).map_err(io_err)?;
    node.value_ptr = SwizzledPtr::from_raw(u64::from_le_bytes(value_bytes));

    Ok(CharNode::N4(Box::new(node)))
}

fn deserialize_charnode16<R: Read>(
    reader: &mut R,
    header: &SerializedCharNodeHeader,
    prefix: CharCompressedPrefix,
) -> Result<CharNode> {
    let mut node = CharNode16::new();
    node.header = header.to_node_header();
    node.prefix = prefix;

    // Read keys
    for key in &mut node.keys {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes).map_err(io_err)?;
        *key = u32::from_le_bytes(bytes);
    }

    // Read children
    for child in &mut node.children {
        let mut raw_bytes = [0u8; 8];
        reader.read_exact(&mut raw_bytes).map_err(io_err)?;
        *child = SwizzledPtr::from_raw(u64::from_le_bytes(raw_bytes));
    }

    // Read value_ptr
    let mut value_bytes = [0u8; 8];
    reader.read_exact(&mut value_bytes).map_err(io_err)?;
    node.value_ptr = SwizzledPtr::from_raw(u64::from_le_bytes(value_bytes));

    Ok(CharNode::N16(Box::new(node)))
}

fn deserialize_charnode48<R: Read>(
    reader: &mut R,
    header: &SerializedCharNodeHeader,
    prefix: CharCompressedPrefix,
) -> Result<CharNode> {
    let mut node = CharNode48::new();
    node.header = header.to_node_header();
    node.prefix = prefix;

    // Read keys
    for key in &mut node.keys {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes).map_err(io_err)?;
        *key = u32::from_le_bytes(bytes);
    }

    // Read children
    for child in &mut node.children {
        let mut raw_bytes = [0u8; 8];
        reader.read_exact(&mut raw_bytes).map_err(io_err)?;
        *child = SwizzledPtr::from_raw(u64::from_le_bytes(raw_bytes));
    }

    // Read value_ptr
    let mut value_bytes = [0u8; 8];
    reader.read_exact(&mut value_bytes).map_err(io_err)?;
    node.value_ptr = SwizzledPtr::from_raw(u64::from_le_bytes(value_bytes));

    Ok(CharNode::N48(Box::new(node)))
}

fn deserialize_charbucket<R: Read>(
    reader: &mut R,
    header: &SerializedCharNodeHeader,
    prefix: CharCompressedPrefix,
) -> Result<CharNode> {
    let mut node = CharBucket::new();
    node.header = header.to_node_header();
    node.prefix = prefix;

    // Read number of entries
    let mut num_entries_bytes = [0u8; 4];
    reader.read_exact(&mut num_entries_bytes).map_err(io_err)?;
    let num_entries = u32::from_le_bytes(num_entries_bytes) as usize;

    // Read value_ptr
    let mut value_bytes = [0u8; 8];
    reader.read_exact(&mut value_bytes).map_err(io_err)?;
    node.value_ptr = SwizzledPtr::from_raw(u64::from_le_bytes(value_bytes));

    // Read entries
    for _ in 0..num_entries {
        let mut key_bytes = [0u8; 4];
        reader.read_exact(&mut key_bytes).map_err(io_err)?;
        let key = u32::from_le_bytes(key_bytes);

        let mut child_bytes = [0u8; 8];
        reader.read_exact(&mut child_bytes).map_err(io_err)?;
        let child = SwizzledPtr::from_raw(u64::from_le_bytes(child_bytes));

        node.entries.insert(key, child);
    }

    Ok(CharNode::Bucket(Box::new(node)))
}

/// Serialize a CharNode to a byte vector
pub fn char_to_bytes(node: &CharNode) -> Result<Vec<u8>> {
    let mut buffer = Vec::with_capacity(char_serialized_size(node));
    serialize_char_node(node, &mut buffer)?;
    Ok(buffer)
}

/// Deserialize a CharNode from a byte slice
pub fn char_from_bytes(bytes: &[u8]) -> Result<CharNode> {
    let mut reader = std::io::Cursor::new(bytes);
    deserialize_char_node(&mut reader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::persistent_artrie::NodeType;
    use crate::dictionary::persistent_artrie_char::nodes::CharArtNode;

    #[test]
    fn test_header_roundtrip() {
        let header = SerializedCharNodeHeader {
            magic: CHAR_NODE_MAGIC,
            version: CHAR_FORMAT_VERSION,
            node_type: char_node_types::CHARNODE4,
            flags: flags::IS_FINAL,
            reserved: 0,
            num_children: 3,
            prefix_len: 5,
            _padding: 0,
            data_size: 100,
        };

        let bytes = header.to_bytes();
        let restored = SerializedCharNodeHeader::from_bytes(&bytes);

        assert_eq!(restored.magic, CHAR_NODE_MAGIC);
        assert_eq!(restored.version, CHAR_FORMAT_VERSION);
        assert_eq!(restored.node_type, char_node_types::CHARNODE4);
        assert_eq!(restored.flags, flags::IS_FINAL);
        assert_eq!(restored.num_children, 3);
        assert_eq!(restored.prefix_len, 5);
        assert_eq!(restored.data_size, 100);
    }

    #[test]
    fn test_header_validation() {
        let mut header = SerializedCharNodeHeader {
            magic: CHAR_NODE_MAGIC,
            version: CHAR_FORMAT_VERSION,
            node_type: char_node_types::CHARNODE4,
            flags: 0,
            reserved: 0,
            num_children: 0,
            prefix_len: 0,
            _padding: 0,
            data_size: 0,
        };

        // Valid header
        assert!(header.validate().is_ok());

        // Invalid magic
        header.magic = *b"BAD\0";
        assert!(matches!(
            header.validate(),
            Err(PersistentARTrieError::InvalidMagic { .. })
        ));
        header.magic = CHAR_NODE_MAGIC;

        // Future version
        header.version = 255;
        assert!(matches!(
            header.validate(),
            Err(PersistentARTrieError::UnsupportedVersion { .. })
        ));
        header.version = CHAR_FORMAT_VERSION;

        // Invalid node type
        header.node_type = 99;
        assert!(matches!(
            header.validate(),
            Err(PersistentARTrieError::CorruptedFile { .. })
        ));
        header.node_type = char_node_types::CHARNODE4;

        // Invalid prefix length
        header.prefix_len = 10;
        assert!(matches!(
            header.validate(),
            Err(PersistentARTrieError::CorruptedFile { .. })
        ));
    }

    #[test]
    fn test_charnode4_roundtrip() {
        let mut node4 = CharNode4::new();
        let prefix_chars: Vec<u32> = "test".chars().map(|c| c as u32).collect();
        node4.prefix = CharCompressedPrefix::from_chars(&prefix_chars);
        node4.header.prefix_len = 4;
        node4.header.set_final(true);

        // Add some children
        node4
            .add_child('a' as u32, SwizzledPtr::on_disk(100, 0, NodeType::CharNode4))
            .expect("add child a");
        node4
            .add_child('b' as u32, SwizzledPtr::on_disk(200, 0, NodeType::CharNode16))
            .expect("add child b");

        let node = CharNode::N4(Box::new(node4));
        let bytes = char_to_bytes(&node).expect("serialize");
        let restored = char_from_bytes(&bytes).expect("deserialize");

        assert!(matches!(restored, CharNode::N4(_)));
        assert_eq!(restored.header().prefix_len, 4);
        assert!(restored.header().is_final());
        assert_eq!(restored.header().num_children, 2);
        assert!(restored.find_child('a' as u32).is_some());
        assert!(restored.find_child('b' as u32).is_some());
        assert!(restored.find_child('c' as u32).is_none());
    }

    #[test]
    fn test_charnode16_roundtrip() {
        let mut node16 = CharNode16::new();
        let prefix_chars: Vec<u32> = "prefix".chars().map(|c| c as u32).collect();
        node16.prefix = CharCompressedPrefix::from_chars(&prefix_chars);
        node16.header.prefix_len = 6;

        // Add some children
        for i in 0..8 {
            node16
                .add_child('a' as u32 + i, SwizzledPtr::on_disk(i as u32, 0, NodeType::CharNode4))
                .expect("add child");
        }

        let node = CharNode::N16(Box::new(node16));
        let bytes = char_to_bytes(&node).expect("serialize");
        let restored = char_from_bytes(&bytes).expect("deserialize");

        assert!(matches!(restored, CharNode::N16(_)));
        assert_eq!(restored.header().prefix_len, 6);
        assert_eq!(restored.header().num_children, 8);

        for i in 0..8 {
            assert!(restored.find_child('a' as u32 + i).is_some());
        }
    }

    #[test]
    fn test_charnode48_roundtrip() {
        let mut node48 = CharNode48::new();

        // Add children at various Unicode code points
        let keys: Vec<u32> = "αβγδεζηθ".chars().map(|c| c as u32).collect();
        for (i, &key) in keys.iter().enumerate() {
            node48
                .add_child(key, SwizzledPtr::on_disk(i as u32, 0, NodeType::CharNode4))
                .expect("add child");
        }

        let node = CharNode::N48(Box::new(node48));
        let bytes = char_to_bytes(&node).expect("serialize");
        let restored = char_from_bytes(&bytes).expect("deserialize");

        assert!(matches!(restored, CharNode::N48(_)));
        assert_eq!(restored.header().num_children, 8);

        for &key in &keys {
            assert!(
                restored.find_child(key).is_some(),
                "should find key {}",
                char::from_u32(key).unwrap_or('?')
            );
        }
    }

    #[test]
    fn test_charbucket_roundtrip() {
        let mut bucket = CharBucket::new();

        // Add many children (Unicode + emoji)
        let keys: Vec<u32> = "日本語中文한글🎉🎊🎋🎌🎍🎎🎏🎐🎑🎒🎓"
            .chars()
            .map(|c| c as u32)
            .collect();

        for (i, &key) in keys.iter().enumerate() {
            bucket
                .add_child(key, SwizzledPtr::on_disk(i as u32, 0, NodeType::CharNode4))
                .expect("add child");
        }

        bucket.header.set_final(true);

        let node = CharNode::Bucket(Box::new(bucket));
        let bytes = char_to_bytes(&node).expect("serialize");
        let restored = char_from_bytes(&bytes).expect("deserialize");

        assert!(matches!(restored, CharNode::Bucket(_)));
        assert!(restored.header().is_final());
        assert_eq!(restored.header().num_children, keys.len() as u16);

        for &key in &keys {
            assert!(
                restored.find_child(key).is_some(),
                "should find key {}",
                char::from_u32(key).unwrap_or('?')
            );
        }
    }

    #[test]
    fn test_empty_node_roundtrip() {
        // Test that empty nodes serialize and deserialize correctly
        for create_node in [
            || CharNode::N4(Box::new(CharNode4::new())),
            || CharNode::N16(Box::new(CharNode16::new())),
            || CharNode::N48(Box::new(CharNode48::new())),
            || CharNode::Bucket(Box::new(CharBucket::new())),
        ] {
            let node = create_node();
            let bytes = char_to_bytes(&node).expect("serialize");
            let restored = char_from_bytes(&bytes).expect("deserialize");
            assert_eq!(restored.header().num_children, 0);
        }
    }

    #[test]
    fn test_serialized_size_calculation() {
        // CharNode4 without prefix: 16 header + 0 prefix + 56 data
        let node4 = CharNode::N4(Box::new(CharNode4::new()));
        assert_eq!(char_serialized_size(&node4), 16 + 0 + 56);

        // CharNode4 with prefix: 16 header + 24 prefix + 56 data
        let mut node4_with_prefix = CharNode4::new();
        let prefix: Vec<u32> = "test".chars().map(|c| c as u32).collect();
        node4_with_prefix.prefix = CharCompressedPrefix::from_chars(&prefix);
        node4_with_prefix.header.prefix_len = 4;
        let node4_p = CharNode::N4(Box::new(node4_with_prefix));
        assert_eq!(char_serialized_size(&node4_p), 16 + 24 + 56);

        // CharNode16 without prefix: 16 + 0 + 200
        let node16 = CharNode::N16(Box::new(CharNode16::new()));
        assert_eq!(char_serialized_size(&node16), 16 + 0 + 200);

        // CharNode48 without prefix: 16 + 0 + 584
        let node48 = CharNode::N48(Box::new(CharNode48::new()));
        assert_eq!(char_serialized_size(&node48), 16 + 0 + 584);

        // CharBucket with 5 entries: 16 + 0 + (4 + 8 + 5*12)
        let mut bucket = CharBucket::new();
        for i in 0..5 {
            bucket
                .add_child(i, SwizzledPtr::on_disk(i as u32, 0, NodeType::CharNode4))
                .expect("add");
        }
        let bucket_node = CharNode::Bucket(Box::new(bucket));
        assert_eq!(char_serialized_size(&bucket_node), 16 + 0 + (4 + 8 + 5 * 12));
    }

    #[test]
    fn test_unicode_prefix_roundtrip() {
        let mut node = CharNode4::new();
        let prefix: Vec<u32> = "日本🎉".chars().map(|c| c as u32).collect();
        node.prefix = CharCompressedPrefix::from_chars(&prefix);
        node.header.prefix_len = 3;

        let char_node = CharNode::N4(Box::new(node));
        let bytes = char_to_bytes(&char_node).expect("serialize");
        let restored = char_from_bytes(&bytes).expect("deserialize");

        assert_eq!(restored.header().prefix_len, 3);
        let restored_chars = restored.prefix().to_chars(3);
        assert_eq!(restored_chars, vec!['日', '本', '🎉']);
    }

    #[test]
    fn test_value_ptr_roundtrip() {
        let mut node = CharNode4::new();
        node.value_ptr = SwizzledPtr::on_disk(999, 123, NodeType::Bucket);
        node.header.set_final(true);

        let char_node = CharNode::N4(Box::new(node));
        let bytes = char_to_bytes(&char_node).expect("serialize");
        let restored = char_from_bytes(&bytes).expect("deserialize");

        if let CharNode::N4(n) = restored {
            let loc = n.value_ptr.disk_location().expect("should have disk location");
            assert_eq!(loc.block_id, 999);
            assert_eq!(loc.offset, 123);
        } else {
            panic!("Expected CharNode::N4");
        }
    }
}
