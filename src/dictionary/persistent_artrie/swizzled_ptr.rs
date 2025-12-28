//! Swizzled pointer implementation for transparent memory/disk addressing.
//!
//! A swizzled pointer uses a single 64-bit value to represent either:
//! - A memory pointer (when the node is loaded in RAM)
//! - A disk reference (block_id + offset when the node is on disk)
//!
//! The MSB (bit 63) discriminates between the two states:
//! - MSB = 1: Memory pointer (remaining 63 bits are the address)
//! - MSB = 0: Disk reference (encoded block_id + offset + flags)
//!
//! This design enables lazy loading: start with disk references, swizzle to
//! memory pointers on first access, and the transition is atomic.

use std::sync::atomic::{AtomicU64, Ordering};

use super::error::SwizzleError;

/// The swizzle flag is the MSB (bit 63).
/// When set, the pointer is in memory; when clear, it's a disk reference.
const SWIZZLE_FLAG: u64 = 1 << 63;

/// Mask to extract the memory address (clear the MSB).
const PTR_MASK: u64 = !SWIZZLE_FLAG;

/// Bit layout for disk references (when MSB = 0):
/// - Bits 62-40: Block ID (23 bits = 8M blocks)
/// - Bits 39-18: Offset within block (22 bits = 4MB offset)
/// - Bits 17-0: Flags including node type (18 bits)
const BLOCK_ID_SHIFT: u32 = 40;
const OFFSET_SHIFT: u32 = 18;
const BLOCK_ID_BITS: u32 = 23;
const OFFSET_BITS: u32 = 22;
const FLAGS_BITS: u32 = 18;

const BLOCK_ID_MASK: u64 = (1 << BLOCK_ID_BITS) - 1; // 0x7FFFFF (23 bits)
const OFFSET_MASK: u64 = (1 << OFFSET_BITS) - 1; // 0x3FFFFF (22 bits)
const FLAGS_MASK: u64 = (1 << FLAGS_BITS) - 1; // 0x3FFFF (18 bits)

/// Maximum block ID (8M - 1).
pub const MAX_BLOCK_ID: u32 = (1 << BLOCK_ID_BITS) - 1;

/// Maximum offset within a block (4MB - 1).
pub const MAX_OFFSET: u32 = (1 << OFFSET_BITS) - 1;

/// Node type identifiers stored in the flags field.
///
/// # Byte-Level Nodes (0-99)
///
/// These are used by `PersistentARTrie` with u8 keys:
/// - `Node4`: 1-4 children, linear scan
/// - `Node16`: 5-16 children, SSE4.1 SIMD
/// - `Node48`: 17-48 children, indexed lookup
/// - `Node256`: 49-256 children, direct array
/// - `Bucket`: Leaf bucket with strings
///
/// # Char-Level Nodes (100-199)
///
/// These are used by `PersistentARTrieChar` with u32 keys:
/// - `CharNode4`: 1-4 children, linear scan
/// - `CharNode16`: 5-16 children, AVX2 SIMD (8×u32)
/// - `CharNode48`: 17-48 children, binary search
/// - `CharBucket`: >48 children, HashMap-like
///
/// Note: CharNode256 is impossible for u32 keys (would require 4GB array).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NodeType {
    // === Byte-Level Nodes (0-99) ===

    /// Node with 1-4 children, linear scan lookup (byte-level).
    Node4 = 4,
    /// Node with 5-16 children, SIMD lookup (byte-level).
    Node16 = 16,
    /// Node with 17-48 children, indexed lookup (byte-level).
    Node48 = 48,
    /// Node with 49-256 children, direct array lookup (byte-level).
    Node256 = 0, // Use 0 since 256 doesn't fit in u8 nicely
    /// Leaf bucket containing multiple strings (byte-level).
    Bucket = 1,

    // === Char-Level Nodes (100-199) ===

    /// Char node with 1-4 children, linear scan lookup (char-level).
    CharNode4 = 104,
    /// Char node with 5-16 children, AVX2 SIMD lookup (char-level).
    CharNode16 = 116,
    /// Char node with 17-48 children, binary search lookup (char-level).
    CharNode48 = 148,
    /// Char bucket with >48 children, HashMap-like (char-level).
    CharBucket = 101,
}

impl NodeType {
    /// Convert from u8, returning None for invalid values.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            // Byte-level nodes
            4 => Some(NodeType::Node4),
            16 => Some(NodeType::Node16),
            48 => Some(NodeType::Node48),
            0 => Some(NodeType::Node256),
            1 => Some(NodeType::Bucket),
            // Char-level nodes
            104 => Some(NodeType::CharNode4),
            116 => Some(NodeType::CharNode16),
            148 => Some(NodeType::CharNode48),
            101 => Some(NodeType::CharBucket),
            _ => None,
        }
    }

    /// Check if this is a byte-level node type.
    #[inline]
    pub fn is_byte_level(&self) -> bool {
        matches!(
            self,
            NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 | NodeType::Bucket
        )
    }

    /// Check if this is a char-level node type.
    #[inline]
    pub fn is_char_level(&self) -> bool {
        matches!(
            self,
            NodeType::CharNode4 | NodeType::CharNode16 | NodeType::CharNode48 | NodeType::CharBucket
        )
    }
}


/// A swizzled pointer that can represent either a memory address or a disk location.
///
/// This is the core mechanism for lazy loading: nodes start as disk references
/// and are swizzled to memory pointers when first accessed.
///
/// # Thread Safety
///
/// All operations use atomic instructions with appropriate memory ordering.
/// Multiple threads can safely race to swizzle the same pointer; only one
/// will succeed, and all will observe the same final state.
#[derive(Debug)]
#[repr(transparent)]
pub struct SwizzledPtr(AtomicU64);

impl SwizzledPtr {
    /// Create a null pointer (neither in memory nor on disk).
    pub const fn null() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Create a new unswizzled (on-disk) pointer.
    ///
    /// # Arguments
    ///
    /// * `block_id` - The block containing the node (max 8M - 1)
    /// * `offset` - Offset within the block (max 4MB - 1)
    /// * `node_type` - The type of node at this location
    ///
    /// # Panics
    ///
    /// Panics in debug mode if block_id or offset exceed their maximum values.
    pub fn on_disk(block_id: u32, offset: u32, node_type: NodeType) -> Self {
        debug_assert!(
            block_id <= MAX_BLOCK_ID,
            "block_id {} exceeds maximum {}",
            block_id,
            MAX_BLOCK_ID
        );
        debug_assert!(
            offset <= MAX_OFFSET,
            "offset {} exceeds maximum {}",
            offset,
            MAX_OFFSET
        );

        let encoded = ((block_id as u64 & BLOCK_ID_MASK) << BLOCK_ID_SHIFT)
            | ((offset as u64 & OFFSET_MASK) << OFFSET_SHIFT)
            | (node_type as u64 & FLAGS_MASK);

        debug_assert!(
            encoded & SWIZZLE_FLAG == 0,
            "disk reference must not have swizzle flag set"
        );

        Self(AtomicU64::new(encoded))
    }

    /// Create a new swizzled (in-memory) pointer.
    ///
    /// # Safety
    ///
    /// The pointer must be valid and have bit 63 clear (which is true for
    /// all user-space pointers on modern 64-bit systems).
    ///
    /// # Panics
    ///
    /// Panics if the pointer has bit 63 set (which would conflict with the swizzle flag).
    pub fn in_memory<T>(ptr: *const T) -> Self {
        let addr = ptr as u64;
        assert!(
            addr & SWIZZLE_FLAG == 0,
            "pointer address has bit 63 set, which conflicts with swizzle flag"
        );
        Self(AtomicU64::new(addr | SWIZZLE_FLAG))
    }

    /// Check if this pointer is null.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.load(Ordering::Acquire) == 0
    }

    /// Check if this pointer is swizzled (pointing to memory).
    #[inline]
    pub fn is_swizzled(&self) -> bool {
        self.0.load(Ordering::Acquire) & SWIZZLE_FLAG != 0
    }

    /// Check if this pointer is unswizzled (pointing to disk).
    #[inline]
    pub fn is_on_disk(&self) -> bool {
        let val = self.0.load(Ordering::Acquire);
        val != 0 && val & SWIZZLE_FLAG == 0
    }

    /// Get the memory pointer (fast path for swizzled pointers).
    ///
    /// # Safety
    ///
    /// The caller must ensure that `is_swizzled()` returns true before calling this.
    /// The returned pointer is valid as long as the node is not evicted.
    #[inline]
    pub unsafe fn as_ptr_unchecked<T>(&self) -> *const T {
        let val = self.0.load(Ordering::Acquire);
        debug_assert!(val & SWIZZLE_FLAG != 0, "pointer is not swizzled");
        (val & PTR_MASK) as *const T
    }

    /// Get the memory pointer, returning None if not swizzled.
    #[inline]
    pub fn as_ptr<T>(&self) -> Option<*const T> {
        let val = self.0.load(Ordering::Acquire);
        if val & SWIZZLE_FLAG != 0 {
            Some((val & PTR_MASK) as *const T)
        } else {
            None
        }
    }

    /// Decode the disk location from an unswizzled pointer.
    ///
    /// Returns None if the pointer is swizzled (in memory) or null.
    pub fn disk_location(&self) -> Option<DiskLocation> {
        let val = self.0.load(Ordering::Acquire);
        if val == 0 || val & SWIZZLE_FLAG != 0 {
            return None;
        }

        let block_id = ((val >> BLOCK_ID_SHIFT) & BLOCK_ID_MASK) as u32;
        let offset = ((val >> OFFSET_SHIFT) & OFFSET_MASK) as u32;
        let type_byte = (val & FLAGS_MASK) as u8;
        let node_type = NodeType::from_u8(type_byte)?;

        Some(DiskLocation {
            block_id,
            offset,
            node_type,
        })
    }

    /// Atomically swizzle: replace a disk reference with a memory pointer.
    ///
    /// This operation is atomic and thread-safe. If multiple threads race to
    /// swizzle the same pointer, only one will succeed.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The memory pointer to swizzle to
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the swizzle succeeded
    /// - `Err(AlreadySwizzled)` if the pointer was already in memory
    /// - `Err(RaceCondition)` if another thread swizzled first
    pub fn swizzle<T>(&self, ptr: *const T) -> Result<(), SwizzleError> {
        let old = self.0.load(Ordering::Acquire);

        // Already swizzled?
        if old & SWIZZLE_FLAG != 0 {
            return Err(SwizzleError::AlreadySwizzled);
        }

        // Null pointer?
        if old == 0 {
            return Err(SwizzleError::AlreadyUnswizzled);
        }

        let addr = ptr as u64;
        assert!(
            addr & SWIZZLE_FLAG == 0,
            "pointer address has bit 63 set"
        );

        let new = addr | SWIZZLE_FLAG;

        self.0
            .compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| SwizzleError::RaceCondition)
    }

    /// Atomically unswizzle: replace a memory pointer with a disk reference.
    ///
    /// This is used during eviction to convert an in-memory node back to
    /// a disk reference.
    ///
    /// # Arguments
    ///
    /// * `block_id` - The block where the node is stored
    /// * `offset` - Offset within the block
    /// * `node_type` - The type of node
    ///
    /// # Returns
    ///
    /// - `Ok(old_ptr)` with the previous memory pointer if successful
    /// - `Err(AlreadyUnswizzled)` if the pointer was already on disk
    /// - `Err(RaceCondition)` if another thread modified the pointer
    pub fn unswizzle<T>(
        &self,
        block_id: u32,
        offset: u32,
        node_type: NodeType,
    ) -> Result<*const T, SwizzleError> {
        if block_id > MAX_BLOCK_ID {
            return Err(SwizzleError::BlockIdOverflow { block_id });
        }
        if offset > MAX_OFFSET {
            return Err(SwizzleError::OffsetOverflow { offset });
        }

        let old = self.0.load(Ordering::Acquire);

        // Not swizzled?
        if old & SWIZZLE_FLAG == 0 {
            return Err(SwizzleError::AlreadyUnswizzled);
        }

        let new = ((block_id as u64 & BLOCK_ID_MASK) << BLOCK_ID_SHIFT)
            | ((offset as u64 & OFFSET_MASK) << OFFSET_SHIFT)
            | (node_type as u64 & FLAGS_MASK);

        self.0
            .compare_exchange(old, new, Ordering::AcqRel, Ordering::Acquire)
            .map(|v| (v & PTR_MASK) as *const T)
            .map_err(|_| SwizzleError::RaceCondition)
    }

    /// Get the raw u64 value for serialization.
    ///
    /// This returns the internal representation which can be stored
    /// and later restored with `from_raw`.
    pub fn to_raw(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    /// Create a SwizzledPtr from a raw u64 value.
    ///
    /// This is the inverse of `to_raw` and is used for deserialization.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` was produced by a previous
    /// call to `to_raw` and represents a valid pointer state.
    pub fn from_raw(raw: u64) -> Self {
        Self(AtomicU64::new(raw))
    }
}

impl Clone for SwizzledPtr {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(self.0.load(Ordering::Acquire)))
    }
}

impl Default for SwizzledPtr {
    fn default() -> Self {
        Self::null()
    }
}

/// Decoded disk location from an unswizzled pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskLocation {
    /// Block ID (0 to 8M - 1).
    pub block_id: u32,
    /// Offset within the block (0 to 4MB - 1).
    pub offset: u32,
    /// Type of node at this location.
    pub node_type: NodeType,
}

impl DiskLocation {
    /// Calculate the absolute byte offset in the file.
    ///
    /// # Arguments
    ///
    /// * `block_size` - Size of each block in bytes
    pub fn file_offset(&self, block_size: usize) -> u64 {
        (self.block_id as u64 * block_size as u64) + self.offset as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_pointer() {
        let ptr = SwizzledPtr::null();
        assert!(ptr.is_null());
        assert!(!ptr.is_swizzled());
        assert!(!ptr.is_on_disk());
    }

    #[test]
    fn test_disk_reference() {
        let ptr = SwizzledPtr::on_disk(1234, 5678, NodeType::Node16);
        assert!(!ptr.is_null());
        assert!(!ptr.is_swizzled());
        assert!(ptr.is_on_disk());

        let loc = ptr.disk_location().expect("should have disk location");
        assert_eq!(loc.block_id, 1234);
        assert_eq!(loc.offset, 5678);
        assert_eq!(loc.node_type, NodeType::Node16);
    }

    #[test]
    fn test_memory_pointer() {
        let data: u64 = 42;
        let ptr = SwizzledPtr::in_memory(&data);
        assert!(!ptr.is_null());
        assert!(ptr.is_swizzled());
        assert!(!ptr.is_on_disk());

        let retrieved: *const u64 = ptr.as_ptr().expect("should have memory pointer");
        assert_eq!(unsafe { *retrieved }, 42);
    }

    #[test]
    fn test_swizzle() {
        let ptr = SwizzledPtr::on_disk(100, 200, NodeType::Node4);
        assert!(ptr.is_on_disk());

        let data: u64 = 12345;
        ptr.swizzle(&data).expect("swizzle should succeed");

        assert!(ptr.is_swizzled());
        let retrieved: *const u64 = ptr.as_ptr().expect("should have memory pointer");
        assert_eq!(unsafe { *retrieved }, 12345);
    }

    #[test]
    fn test_double_swizzle_fails() {
        let ptr = SwizzledPtr::on_disk(100, 200, NodeType::Node4);
        let data: u64 = 42;
        ptr.swizzle(&data).expect("first swizzle should succeed");

        let result = ptr.swizzle(&data);
        assert_eq!(result, Err(SwizzleError::AlreadySwizzled));
    }

    #[test]
    fn test_unswizzle() {
        let data: u64 = 42;
        let ptr = SwizzledPtr::in_memory(&data);
        assert!(ptr.is_swizzled());

        let old_ptr: *const u64 = ptr
            .unswizzle(500, 600, NodeType::Bucket)
            .expect("unswizzle should succeed");

        assert!(ptr.is_on_disk());
        assert_eq!(unsafe { *old_ptr }, 42);

        let loc = ptr.disk_location().expect("should have disk location");
        assert_eq!(loc.block_id, 500);
        assert_eq!(loc.offset, 600);
        assert_eq!(loc.node_type, NodeType::Bucket);
    }

    #[test]
    fn test_max_values() {
        let ptr = SwizzledPtr::on_disk(MAX_BLOCK_ID, MAX_OFFSET, NodeType::Node256);
        let loc = ptr.disk_location().expect("should have disk location");
        assert_eq!(loc.block_id, MAX_BLOCK_ID);
        assert_eq!(loc.offset, MAX_OFFSET);
    }

    #[test]
    fn test_file_offset_calculation() {
        let loc = DiskLocation {
            block_id: 10,
            offset: 1024,
            node_type: NodeType::Node16,
        };

        // With 256KB blocks
        let block_size = 256 * 1024;
        let expected = 10 * block_size as u64 + 1024;
        assert_eq!(loc.file_offset(block_size), expected);
    }

    #[test]
    fn test_node_type_roundtrip() {
        for node_type in [
            NodeType::Node4,
            NodeType::Node16,
            NodeType::Node48,
            NodeType::Node256,
            NodeType::Bucket,
        ] {
            let ptr = SwizzledPtr::on_disk(123, 456, node_type);
            let loc = ptr.disk_location().expect("should decode");
            assert_eq!(loc.node_type, node_type);
        }
    }
}
