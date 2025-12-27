//! Disk Manager for Persistent Adaptive Radix Trie
//!
//! This module provides memory-mapped file I/O abstraction for the persistent
//! dictionary implementation. It manages:
//!
//! - Memory-mapped file via `memmap2` crate
//! - Block allocation with 256KB block size (optimal for NVMe)
//! - Free list management for block reuse
//! - File header with metadata (magic, version, root pointer)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    File Layout                               │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Block 0: File Header (256KB)                                 │
//! │   - Magic number (8 bytes)                                   │
//! │   - Version (4 bytes)                                        │
//! │   - Flags (4 bytes)                                          │
//! │   - Root pointer (8 bytes)                                   │
//! │   - Block count (8 bytes)                                    │
//! │   - Free list head (8 bytes)                                 │
//! │   - Entry count (8 bytes)                                    │
//! │   - Checksum (8 bytes)                                       │
//! │   - Reserved (256KB - 56 bytes)                              │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Block 1: Data Block (256KB)                                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Block 2: Data Block (256KB)                                  │
//! ├─────────────────────────────────────────────────────────────┤
//! │ ...                                                          │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Block Size Rationale
//!
//! 256KB blocks are chosen for:
//! - Optimal NVMe I/O granularity (typical 4KB-256KB sweet spot)
//! - Reduced metadata overhead vs smaller pages
//! - Good cache locality for sequential access
//! - Alignment with memory-mapped page boundaries

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write as IoWrite};
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[cfg(feature = "persistent-artrie")]
use memmap2::{MmapMut, MmapOptions};

#[cfg(feature = "parking_lot")]
use parking_lot::RwLock;
#[cfg(not(feature = "parking_lot"))]
use std::sync::RwLock;

use super::error::{PersistentARTrieError, Result};

/// Block size: 256KB (262144 bytes)
/// Optimal for NVMe I/O and memory-mapped pages
pub const BLOCK_SIZE: usize = 256 * 1024;

/// Maximum supported block count (24-bit block ID = 16M blocks = 4TB max file size)
pub const MAX_BLOCK_COUNT: u32 = 1 << 24;

/// Magic number identifying a valid PART file
/// "PART" in ASCII + version nibbles
pub const MAGIC_NUMBER: u64 = 0x5041_5254_0001_0000; // "PART" + v1.0

/// Current file format version
pub const FORMAT_VERSION: u32 = 1;

/// File header structure (stored at block 0)
///
/// Total size: 64 bytes (cacheline aligned)
/// Remaining space in block 0 is reserved for future use
#[repr(C, align(64))]
#[derive(Debug)]
pub struct FileHeader {
    /// Magic number for file identification
    pub magic: u64,
    /// File format version
    pub version: u32,
    /// Flags (reserved for future use)
    pub flags: u32,
    /// Root node pointer (swizzled format)
    pub root_ptr: AtomicU64,
    /// Total number of allocated blocks
    pub block_count: AtomicU32,
    /// Padding for alignment
    _pad1: u32,
    /// Head of free block list (0 = no free blocks)
    pub free_list_head: AtomicU64,
    /// Total number of entries in the dictionary
    pub entry_count: AtomicU64,
    /// CRC-64 checksum of header (excluding this field)
    pub checksum: u64,
}

impl FileHeader {
    /// Create a new file header with default values
    pub fn new() -> Self {
        Self {
            magic: MAGIC_NUMBER,
            version: FORMAT_VERSION,
            flags: 0,
            root_ptr: AtomicU64::new(0),
            block_count: AtomicU32::new(1), // Block 0 is header
            _pad1: 0,
            free_list_head: AtomicU64::new(0),
            entry_count: AtomicU64::new(0),
            checksum: 0,
        }
    }

    /// Validate the header magic and version
    pub fn validate(&self) -> Result<()> {
        if self.magic != MAGIC_NUMBER {
            return Err(PersistentARTrieError::InvalidMagic {
                expected: MAGIC_NUMBER,
                found: self.magic,
            });
        }
        if self.version > FORMAT_VERSION {
            return Err(PersistentARTrieError::UnsupportedVersion {
                max_supported: FORMAT_VERSION,
                found: self.version,
            });
        }
        Ok(())
    }

    /// Compute CRC-64 checksum of header fields (excluding checksum field)
    pub fn compute_checksum(&self) -> u64 {
        // Simple FNV-1a hash for checksum (not cryptographic)
        let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
        let prime: u64 = 0x100000001b3; // FNV prime

        // Hash each field
        for byte in self.magic.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }
        for byte in self.version.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }
        for byte in self.flags.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }
        for byte in self.root_ptr.load(Ordering::SeqCst).to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }
        for byte in self.block_count.load(Ordering::SeqCst).to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }
        for byte in self.free_list_head.load(Ordering::SeqCst).to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }
        for byte in self.entry_count.load(Ordering::SeqCst).to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(prime);
        }

        hash
    }

    /// Update the checksum field
    pub fn update_checksum(&mut self) {
        self.checksum = self.compute_checksum();
    }

    /// Verify the checksum
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.compute_checksum()
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        bytes[0..8].copy_from_slice(&self.magic.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.version.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.flags.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.root_ptr.load(Ordering::SeqCst).to_le_bytes());
        bytes[24..28].copy_from_slice(&self.block_count.load(Ordering::SeqCst).to_le_bytes());
        bytes[28..32].copy_from_slice(&0u32.to_le_bytes()); // padding
        bytes[32..40].copy_from_slice(&self.free_list_head.load(Ordering::SeqCst).to_le_bytes());
        bytes[40..48].copy_from_slice(&self.entry_count.load(Ordering::SeqCst).to_le_bytes());
        bytes[48..56].copy_from_slice(&self.checksum.to_le_bytes());
        // bytes[56..64] remain zero (reserved)
        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        Self {
            magic: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            version: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            flags: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            root_ptr: AtomicU64::new(u64::from_le_bytes(bytes[16..24].try_into().unwrap())),
            block_count: AtomicU32::new(u32::from_le_bytes(bytes[24..28].try_into().unwrap())),
            _pad1: 0,
            free_list_head: AtomicU64::new(u64::from_le_bytes(bytes[32..40].try_into().unwrap())),
            entry_count: AtomicU64::new(u64::from_le_bytes(bytes[40..48].try_into().unwrap())),
            checksum: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
        }
    }
}

impl Default for FileHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Free block list entry
///
/// When a block is freed, it's added to the free list.
/// The first 8 bytes of a free block contain the next free block ID.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FreeBlockEntry {
    /// Next free block ID (0 = end of list)
    pub next: u64,
}

/// Disk manager for persistent storage
///
/// Provides memory-mapped I/O with block allocation and free list management.
/// Thread-safe for concurrent read access; writes require external synchronization.
pub struct DiskManager {
    /// The underlying file
    file: File,
    /// Memory-mapped region (optional, for read-heavy workloads)
    #[cfg(feature = "persistent-artrie")]
    mmap: Option<RwLock<MmapMut>>,
    /// Current file size in bytes
    file_size: AtomicU64,
    /// Path to the file (for error messages)
    path: String,
}

impl DiskManager {
    /// Create a new disk manager, creating the file if it doesn't exist
    ///
    /// # Arguments
    /// * `path` - Path to the data file
    ///
    /// # Returns
    /// * `Ok(DiskManager)` - Successfully opened/created file
    /// * `Err(PersistentARTrieError)` - I/O or format error
    #[cfg(feature = "persistent-artrie")]
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Create or open the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .map_err(|e| PersistentARTrieError::IoError {
                operation: "create file".to_string(),
                path: path_str.clone(),
                source: e,
            })?;

        let metadata = file.metadata().map_err(|e| PersistentARTrieError::IoError {
            operation: "get metadata".to_string(),
            path: path_str.clone(),
            source: e,
        })?;

        let file_size = metadata.len();

        // If file is empty, initialize with header block
        if file_size == 0 {
            Self::initialize_file(&file, &path_str)?;
        }

        let file_size = file
            .metadata()
            .map_err(|e| PersistentARTrieError::IoError {
                operation: "get metadata after init".to_string(),
                path: path_str.clone(),
                source: e,
            })?
            .len();

        // Create memory map
        let mmap = if file_size > 0 {
            let mmap = unsafe {
                MmapOptions::new()
                    .len(file_size as usize)
                    .map_mut(&file)
                    .map_err(|e| PersistentARTrieError::MmapError {
                        operation: "create mmap".to_string(),
                        source: e,
                    })?
            };
            Some(RwLock::new(mmap))
        } else {
            None
        };

        Ok(Self {
            file,
            mmap,
            file_size: AtomicU64::new(file_size),
            path: path_str,
        })
    }

    /// Open an existing disk manager (file must exist)
    #[cfg(feature = "persistent-artrie")]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| PersistentARTrieError::IoError {
                operation: "open file".to_string(),
                path: path_str.clone(),
                source: e,
            })?;

        let file_size = file
            .metadata()
            .map_err(|e| PersistentARTrieError::IoError {
                operation: "get metadata".to_string(),
                path: path_str.clone(),
                source: e,
            })?
            .len();

        if file_size < BLOCK_SIZE as u64 {
            return Err(PersistentARTrieError::CorruptedFile {
                reason: "File too small to contain header block".to_string(),
            });
        }

        // Create memory map
        let mmap = unsafe {
            MmapOptions::new()
                .len(file_size as usize)
                .map_mut(&file)
                .map_err(|e| PersistentARTrieError::MmapError {
                    operation: "create mmap".to_string(),
                    source: e,
                })?
        };

        let manager = Self {
            file,
            mmap: Some(RwLock::new(mmap)),
            file_size: AtomicU64::new(file_size),
            path: path_str,
        };

        // Validate header
        let header = manager.read_header()?;
        header.validate()?;

        if !header.verify_checksum() {
            return Err(PersistentARTrieError::ChecksumMismatch {
                block_id: 0,
                expected: header.compute_checksum(),
                found: header.checksum,
            });
        }

        Ok(manager)
    }

    /// Initialize a new file with header block
    fn initialize_file(file: &File, path: &str) -> Result<()> {
        // Allocate header block (block 0)
        file.set_len(BLOCK_SIZE as u64)
            .map_err(|e| PersistentARTrieError::IoError {
                operation: "set initial file length".to_string(),
                path: path.to_string(),
                source: e,
            })?;

        // Write header
        let mut header = FileHeader::new();
        header.update_checksum();

        let mut file_writer = file;
        file_writer
            .seek(SeekFrom::Start(0))
            .map_err(|e| PersistentARTrieError::IoError {
                operation: "seek to header".to_string(),
                path: path.to_string(),
                source: e,
            })?;

        file_writer
            .write_all(&header.to_bytes())
            .map_err(|e| PersistentARTrieError::IoError {
                operation: "write header".to_string(),
                path: path.to_string(),
                source: e,
            })?;

        file_writer.sync_all().map_err(|e| PersistentARTrieError::IoError {
            operation: "sync after header write".to_string(),
            path: path.to_string(),
            source: e,
        })?;

        Ok(())
    }

    /// Read the file header
    #[cfg(feature = "persistent-artrie")]
    pub fn read_header(&self) -> Result<FileHeader> {
        let mmap_guard = self.mmap.as_ref().ok_or_else(|| {
            PersistentARTrieError::CorruptedFile {
                reason: "No memory map available".to_string(),
            }
        })?;

        #[cfg(feature = "parking_lot")]
        let mmap = mmap_guard.read();
        #[cfg(not(feature = "parking_lot"))]
        let mmap = mmap_guard.read().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "mmap read lock".to_string(),
            }
        })?;

        if mmap.len() < 64 {
            return Err(PersistentARTrieError::CorruptedFile {
                reason: "File too small for header".to_string(),
            });
        }

        let bytes: [u8; 64] = mmap[0..64].try_into().map_err(|_| {
            PersistentARTrieError::CorruptedFile {
                reason: "Failed to read header bytes".to_string(),
            }
        })?;

        Ok(FileHeader::from_bytes(&bytes))
    }

    /// Write the file header
    #[cfg(feature = "persistent-artrie")]
    pub fn write_header(&self, header: &FileHeader) -> Result<()> {
        let mmap_guard = self.mmap.as_ref().ok_or_else(|| {
            PersistentARTrieError::CorruptedFile {
                reason: "No memory map available".to_string(),
            }
        })?;

        #[cfg(feature = "parking_lot")]
        let mut mmap = mmap_guard.write();
        #[cfg(not(feature = "parking_lot"))]
        let mut mmap = mmap_guard.write().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "mmap write lock".to_string(),
            }
        })?;

        let bytes = header.to_bytes();
        mmap[0..64].copy_from_slice(&bytes);

        Ok(())
    }

    /// Allocate a new block
    ///
    /// First checks the free list, then extends the file if needed.
    ///
    /// # Returns
    /// * `Ok(block_id)` - The ID of the allocated block
    /// * `Err(PersistentARTrieError)` - Allocation failed
    #[cfg(feature = "persistent-artrie")]
    pub fn allocate_block(&self) -> Result<u32> {
        // Try to get a block from the free list first
        let header = self.read_header()?;
        let free_head = header.free_list_head.load(Ordering::SeqCst);

        if free_head != 0 {
            // Pop from free list
            let block_id = (free_head >> 40) as u32; // Extract block ID from swizzled format

            // Read the next pointer from the free block
            let next = self.read_free_block_next(block_id)?;

            // Update free list head
            header.free_list_head.store(next, Ordering::SeqCst);
            self.write_header(&header)?;

            return Ok(block_id);
        }

        // No free blocks, extend file
        let block_count = header.block_count.load(Ordering::SeqCst);

        if block_count >= MAX_BLOCK_COUNT {
            return Err(PersistentARTrieError::OutOfSpace {
                current_blocks: block_count,
                max_blocks: MAX_BLOCK_COUNT,
            });
        }

        let new_block_id = block_count;
        let new_file_size = (block_count as u64 + 1) * BLOCK_SIZE as u64;

        // Extend file
        self.file
            .set_len(new_file_size)
            .map_err(|e| PersistentARTrieError::IoError {
                operation: "extend file".to_string(),
                path: self.path.clone(),
                source: e,
            })?;

        // Remap the file
        self.remap(new_file_size)?;

        // Update header
        header
            .block_count
            .store(block_count + 1, Ordering::SeqCst);
        self.file_size.store(new_file_size, Ordering::SeqCst);

        // Update header with new checksum
        let mut updated_header = self.read_header()?;
        updated_header
            .block_count
            .store(block_count + 1, Ordering::SeqCst);
        updated_header.checksum = updated_header.compute_checksum();
        self.write_header(&updated_header)?;

        Ok(new_block_id)
    }

    /// Free a block, adding it to the free list
    #[cfg(feature = "persistent-artrie")]
    pub fn free_block(&self, block_id: u32) -> Result<()> {
        if block_id == 0 {
            return Err(PersistentARTrieError::InvalidBlockId {
                block_id,
                reason: "Cannot free header block".to_string(),
            });
        }

        let header = self.read_header()?;
        let block_count = header.block_count.load(Ordering::SeqCst);

        if block_id >= block_count {
            return Err(PersistentARTrieError::InvalidBlockId {
                block_id,
                reason: format!("Block ID {} >= block count {}", block_id, block_count),
            });
        }

        // Get current free list head
        let old_head = header.free_list_head.load(Ordering::SeqCst);

        // Write old head to the freed block
        self.write_free_block_next(block_id, old_head)?;

        // Update free list head to point to this block
        // Encode block_id in swizzled format (block_id << 40)
        let new_head = (block_id as u64) << 40;
        header.free_list_head.store(new_head, Ordering::SeqCst);

        // Update checksum and write header
        let mut updated_header = self.read_header()?;
        updated_header.free_list_head.store(new_head, Ordering::SeqCst);
        updated_header.checksum = updated_header.compute_checksum();
        self.write_header(&updated_header)?;

        Ok(())
    }

    /// Read the next pointer from a free block
    #[cfg(feature = "persistent-artrie")]
    fn read_free_block_next(&self, block_id: u32) -> Result<u64> {
        let offset = block_id as usize * BLOCK_SIZE;

        let mmap_guard = self.mmap.as_ref().ok_or_else(|| {
            PersistentARTrieError::CorruptedFile {
                reason: "No memory map available".to_string(),
            }
        })?;

        #[cfg(feature = "parking_lot")]
        let mmap = mmap_guard.read();
        #[cfg(not(feature = "parking_lot"))]
        let mmap = mmap_guard.read().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "mmap read lock".to_string(),
            }
        })?;

        if offset + 8 > mmap.len() {
            return Err(PersistentARTrieError::InvalidBlockId {
                block_id,
                reason: "Block offset exceeds file size".to_string(),
            });
        }

        let bytes: [u8; 8] = mmap[offset..offset + 8].try_into().map_err(|_| {
            PersistentARTrieError::CorruptedFile {
                reason: "Failed to read free block next pointer".to_string(),
            }
        })?;

        Ok(u64::from_le_bytes(bytes))
    }

    /// Write the next pointer to a free block
    #[cfg(feature = "persistent-artrie")]
    fn write_free_block_next(&self, block_id: u32, next: u64) -> Result<()> {
        let offset = block_id as usize * BLOCK_SIZE;

        let mmap_guard = self.mmap.as_ref().ok_or_else(|| {
            PersistentARTrieError::CorruptedFile {
                reason: "No memory map available".to_string(),
            }
        })?;

        #[cfg(feature = "parking_lot")]
        let mut mmap = mmap_guard.write();
        #[cfg(not(feature = "parking_lot"))]
        let mut mmap = mmap_guard.write().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "mmap write lock".to_string(),
            }
        })?;

        if offset + 8 > mmap.len() {
            return Err(PersistentARTrieError::InvalidBlockId {
                block_id,
                reason: "Block offset exceeds file size".to_string(),
            });
        }

        mmap[offset..offset + 8].copy_from_slice(&next.to_le_bytes());

        Ok(())
    }

    /// Remap the file after extending
    #[cfg(feature = "persistent-artrie")]
    fn remap(&self, new_size: u64) -> Result<()> {
        let mmap_guard = self.mmap.as_ref().ok_or_else(|| {
            PersistentARTrieError::CorruptedFile {
                reason: "No memory map available".to_string(),
            }
        })?;

        #[cfg(feature = "parking_lot")]
        let mut mmap = mmap_guard.write();
        #[cfg(not(feature = "parking_lot"))]
        let mut mmap = mmap_guard.write().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "mmap write lock".to_string(),
            }
        })?;

        // Create new mmap
        let new_mmap = unsafe {
            MmapOptions::new()
                .len(new_size as usize)
                .map_mut(&self.file)
                .map_err(|e| PersistentARTrieError::MmapError {
                    operation: "remap after extend".to_string(),
                    source: e,
                })?
        };

        *mmap = new_mmap;
        Ok(())
    }

    /// Read a block into a buffer
    ///
    /// # Arguments
    /// * `block_id` - The block to read
    /// * `buffer` - Buffer to read into (must be BLOCK_SIZE bytes)
    #[cfg(feature = "persistent-artrie")]
    pub fn read_block(&self, block_id: u32, buffer: &mut [u8; BLOCK_SIZE]) -> Result<()> {
        let offset = block_id as usize * BLOCK_SIZE;

        let mmap_guard = self.mmap.as_ref().ok_or_else(|| {
            PersistentARTrieError::CorruptedFile {
                reason: "No memory map available".to_string(),
            }
        })?;

        #[cfg(feature = "parking_lot")]
        let mmap = mmap_guard.read();
        #[cfg(not(feature = "parking_lot"))]
        let mmap = mmap_guard.read().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "mmap read lock".to_string(),
            }
        })?;

        if offset + BLOCK_SIZE > mmap.len() {
            return Err(PersistentARTrieError::InvalidBlockId {
                block_id,
                reason: format!(
                    "Block offset {} + size {} exceeds file size {}",
                    offset,
                    BLOCK_SIZE,
                    mmap.len()
                ),
            });
        }

        buffer.copy_from_slice(&mmap[offset..offset + BLOCK_SIZE]);
        Ok(())
    }

    /// Write a block from a buffer
    ///
    /// # Arguments
    /// * `block_id` - The block to write
    /// * `buffer` - Buffer to write from (must be BLOCK_SIZE bytes)
    #[cfg(feature = "persistent-artrie")]
    pub fn write_block(&self, block_id: u32, buffer: &[u8; BLOCK_SIZE]) -> Result<()> {
        let offset = block_id as usize * BLOCK_SIZE;

        let mmap_guard = self.mmap.as_ref().ok_or_else(|| {
            PersistentARTrieError::CorruptedFile {
                reason: "No memory map available".to_string(),
            }
        })?;

        #[cfg(feature = "parking_lot")]
        let mut mmap = mmap_guard.write();
        #[cfg(not(feature = "parking_lot"))]
        let mut mmap = mmap_guard.write().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "mmap write lock".to_string(),
            }
        })?;

        if offset + BLOCK_SIZE > mmap.len() {
            return Err(PersistentARTrieError::InvalidBlockId {
                block_id,
                reason: format!(
                    "Block offset {} + size {} exceeds file size {}",
                    offset,
                    BLOCK_SIZE,
                    mmap.len()
                ),
            });
        }

        mmap[offset..offset + BLOCK_SIZE].copy_from_slice(buffer);
        Ok(())
    }

    /// Read a slice of bytes from a block
    ///
    /// # Arguments
    /// * `block_id` - The block to read from
    /// * `offset_in_block` - Offset within the block
    /// * `buffer` - Buffer to read into
    #[cfg(feature = "persistent-artrie")]
    pub fn read_bytes(&self, block_id: u32, offset_in_block: usize, buffer: &mut [u8]) -> Result<()> {
        let file_offset = block_id as usize * BLOCK_SIZE + offset_in_block;

        let mmap_guard = self.mmap.as_ref().ok_or_else(|| {
            PersistentARTrieError::CorruptedFile {
                reason: "No memory map available".to_string(),
            }
        })?;

        #[cfg(feature = "parking_lot")]
        let mmap = mmap_guard.read();
        #[cfg(not(feature = "parking_lot"))]
        let mmap = mmap_guard.read().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "mmap read lock".to_string(),
            }
        })?;

        let end_offset = file_offset + buffer.len();
        if end_offset > mmap.len() {
            return Err(PersistentARTrieError::InvalidBlockId {
                block_id,
                reason: format!(
                    "Read range [{}, {}) exceeds file size {}",
                    file_offset, end_offset, mmap.len()
                ),
            });
        }

        buffer.copy_from_slice(&mmap[file_offset..end_offset]);
        Ok(())
    }

    /// Write a slice of bytes to a block
    ///
    /// # Arguments
    /// * `block_id` - The block to write to
    /// * `offset_in_block` - Offset within the block
    /// * `buffer` - Buffer to write from
    #[cfg(feature = "persistent-artrie")]
    pub fn write_bytes(&self, block_id: u32, offset_in_block: usize, buffer: &[u8]) -> Result<()> {
        let file_offset = block_id as usize * BLOCK_SIZE + offset_in_block;

        let mmap_guard = self.mmap.as_ref().ok_or_else(|| {
            PersistentARTrieError::CorruptedFile {
                reason: "No memory map available".to_string(),
            }
        })?;

        #[cfg(feature = "parking_lot")]
        let mut mmap = mmap_guard.write();
        #[cfg(not(feature = "parking_lot"))]
        let mut mmap = mmap_guard.write().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "mmap write lock".to_string(),
            }
        })?;

        let end_offset = file_offset + buffer.len();
        if end_offset > mmap.len() {
            return Err(PersistentARTrieError::InvalidBlockId {
                block_id,
                reason: format!(
                    "Write range [{}, {}) exceeds file size {}",
                    file_offset, end_offset, mmap.len()
                ),
            });
        }

        mmap[file_offset..end_offset].copy_from_slice(buffer);
        Ok(())
    }

    /// Flush all changes to disk
    #[cfg(feature = "persistent-artrie")]
    pub fn sync(&self) -> Result<()> {
        if let Some(mmap_guard) = &self.mmap {
            #[cfg(feature = "parking_lot")]
            let mmap = mmap_guard.read();
            #[cfg(not(feature = "parking_lot"))]
            let mmap = mmap_guard.read().map_err(|_| {
                PersistentARTrieError::LockPoisoned {
                    resource: "mmap read lock".to_string(),
                }
            })?;

            mmap.flush().map_err(|e| PersistentARTrieError::MmapError {
                operation: "flush mmap".to_string(),
                source: e,
            })?;
        }

        self.file.sync_all().map_err(|e| PersistentARTrieError::IoError {
            operation: "sync file".to_string(),
            path: self.path.clone(),
            source: e,
        })?;

        Ok(())
    }

    /// Get the current file size in bytes
    pub fn file_size(&self) -> u64 {
        self.file_size.load(Ordering::SeqCst)
    }

    /// Get the current block count
    #[cfg(feature = "persistent-artrie")]
    pub fn block_count(&self) -> Result<u32> {
        let header = self.read_header()?;
        Ok(header.block_count.load(Ordering::SeqCst))
    }

    /// Get the entry count
    #[cfg(feature = "persistent-artrie")]
    pub fn entry_count(&self) -> Result<u64> {
        let header = self.read_header()?;
        Ok(header.entry_count.load(Ordering::SeqCst))
    }

    /// Update the entry count
    #[cfg(feature = "persistent-artrie")]
    pub fn set_entry_count(&self, count: u64) -> Result<()> {
        let header = self.read_header()?;
        header.entry_count.store(count, Ordering::SeqCst);

        let mut updated_header = header;
        updated_header.checksum = updated_header.compute_checksum();
        self.write_header(&updated_header)?;

        Ok(())
    }

    /// Get the root pointer
    #[cfg(feature = "persistent-artrie")]
    pub fn root_ptr(&self) -> Result<u64> {
        let header = self.read_header()?;
        Ok(header.root_ptr.load(Ordering::SeqCst))
    }

    /// Set the root pointer
    #[cfg(feature = "persistent-artrie")]
    pub fn set_root_ptr(&self, ptr: u64) -> Result<()> {
        let header = self.read_header()?;
        header.root_ptr.store(ptr, Ordering::SeqCst);

        let mut updated_header = header;
        updated_header.checksum = updated_header.compute_checksum();
        self.write_header(&updated_header)?;

        Ok(())
    }

    /// Get a raw pointer to a location in the memory map
    ///
    /// # Safety
    /// The caller must ensure the returned pointer is not used after the mmap is remapped
    /// or the DiskManager is dropped.
    #[cfg(feature = "persistent-artrie")]
    pub unsafe fn raw_ptr(&self, block_id: u32, offset_in_block: usize) -> Result<*const u8> {
        let file_offset = block_id as usize * BLOCK_SIZE + offset_in_block;

        let mmap_guard = self.mmap.as_ref().ok_or_else(|| {
            PersistentARTrieError::CorruptedFile {
                reason: "No memory map available".to_string(),
            }
        })?;

        #[cfg(feature = "parking_lot")]
        let mmap = mmap_guard.read();
        #[cfg(not(feature = "parking_lot"))]
        let mmap = mmap_guard.read().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "mmap read lock".to_string(),
            }
        })?;

        if file_offset >= mmap.len() {
            return Err(PersistentARTrieError::InvalidBlockId {
                block_id,
                reason: format!(
                    "Offset {} exceeds file size {}",
                    file_offset,
                    mmap.len()
                ),
            });
        }

        Ok(mmap.as_ptr().add(file_offset))
    }
}

#[cfg(all(test, feature = "persistent-artrie"))]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_file_header_serialization() {
        let header = FileHeader::new();
        let bytes = header.to_bytes();
        let restored = FileHeader::from_bytes(&bytes);

        assert_eq!(restored.magic, MAGIC_NUMBER);
        assert_eq!(restored.version, FORMAT_VERSION);
        assert_eq!(restored.flags, 0);
        assert_eq!(restored.root_ptr.load(Ordering::SeqCst), 0);
        assert_eq!(restored.block_count.load(Ordering::SeqCst), 1);
        assert_eq!(restored.free_list_head.load(Ordering::SeqCst), 0);
        assert_eq!(restored.entry_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_header_checksum() {
        let mut header = FileHeader::new();
        header.update_checksum();

        assert!(header.verify_checksum());

        // Modify a field and verify checksum fails
        header.entry_count.store(42, Ordering::SeqCst);
        assert!(!header.verify_checksum());

        // Update checksum and verify it passes again
        header.update_checksum();
        assert!(header.verify_checksum());
    }

    #[test]
    fn test_create_and_open() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test.part");

        // Create new file
        {
            let dm = DiskManager::create(&path).expect("Failed to create DiskManager");
            assert_eq!(dm.file_size(), BLOCK_SIZE as u64);

            let header = dm.read_header().expect("Failed to read header");
            assert_eq!(header.magic, MAGIC_NUMBER);
            assert_eq!(header.block_count.load(Ordering::SeqCst), 1);
        }

        // Open existing file
        {
            let dm = DiskManager::open(&path).expect("Failed to open DiskManager");
            let header = dm.read_header().expect("Failed to read header");
            assert_eq!(header.magic, MAGIC_NUMBER);
        }
    }

    #[test]
    fn test_allocate_blocks() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_alloc.part");

        let dm = DiskManager::create(&path).expect("Failed to create DiskManager");

        // Allocate several blocks
        let block1 = dm.allocate_block().expect("Failed to allocate block 1");
        let block2 = dm.allocate_block().expect("Failed to allocate block 2");
        let block3 = dm.allocate_block().expect("Failed to allocate block 3");

        assert_eq!(block1, 1);
        assert_eq!(block2, 2);
        assert_eq!(block3, 3);

        assert_eq!(dm.block_count().expect("Failed to get block count"), 4);
        assert_eq!(dm.file_size(), 4 * BLOCK_SIZE as u64);
    }

    #[test]
    fn test_free_list() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_free.part");

        let dm = DiskManager::create(&path).expect("Failed to create DiskManager");

        // Allocate blocks
        let block1 = dm.allocate_block().expect("alloc 1");
        let block2 = dm.allocate_block().expect("alloc 2");
        let block3 = dm.allocate_block().expect("alloc 3");

        assert_eq!(block1, 1);
        assert_eq!(block2, 2);
        assert_eq!(block3, 3);

        // Free block 2
        dm.free_block(block2).expect("free block 2");

        // Next allocation should reuse block 2
        let block4 = dm.allocate_block().expect("alloc 4");
        assert_eq!(block4, 2);

        // Now allocate a new block
        let block5 = dm.allocate_block().expect("alloc 5");
        assert_eq!(block5, 4);
    }

    #[test]
    fn test_read_write_block() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_rw.part");

        let dm = DiskManager::create(&path).expect("Failed to create DiskManager");
        let block_id = dm.allocate_block().expect("Failed to allocate block");

        // Write test data
        let mut write_buf = [0u8; BLOCK_SIZE];
        write_buf[0] = 0xDE;
        write_buf[1] = 0xAD;
        write_buf[2] = 0xBE;
        write_buf[3] = 0xEF;
        write_buf[BLOCK_SIZE - 1] = 0xFF;

        dm.write_block(block_id, &write_buf)
            .expect("Failed to write block");

        // Read back
        let mut read_buf = [0u8; BLOCK_SIZE];
        dm.read_block(block_id, &mut read_buf)
            .expect("Failed to read block");

        assert_eq!(read_buf[0], 0xDE);
        assert_eq!(read_buf[1], 0xAD);
        assert_eq!(read_buf[2], 0xBE);
        assert_eq!(read_buf[3], 0xEF);
        assert_eq!(read_buf[BLOCK_SIZE - 1], 0xFF);
    }

    #[test]
    fn test_read_write_bytes() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_bytes.part");

        let dm = DiskManager::create(&path).expect("Failed to create DiskManager");
        let block_id = dm.allocate_block().expect("Failed to allocate block");

        // Write at offset
        let data = b"Hello, World!";
        dm.write_bytes(block_id, 100, data)
            .expect("Failed to write bytes");

        // Read back
        let mut read_buf = [0u8; 13];
        dm.read_bytes(block_id, 100, &mut read_buf)
            .expect("Failed to read bytes");

        assert_eq!(&read_buf, data);
    }

    #[test]
    fn test_root_ptr() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_root.part");

        let dm = DiskManager::create(&path).expect("Failed to create DiskManager");

        // Initially zero
        assert_eq!(dm.root_ptr().expect("root_ptr"), 0);

        // Set root pointer
        dm.set_root_ptr(0x123456789ABCDEF0)
            .expect("set_root_ptr");

        assert_eq!(
            dm.root_ptr().expect("root_ptr after set"),
            0x123456789ABCDEF0
        );

        // Sync and reopen
        dm.sync().expect("sync");
        drop(dm);

        let dm2 = DiskManager::open(&path).expect("reopen");
        assert_eq!(
            dm2.root_ptr().expect("root_ptr after reopen"),
            0x123456789ABCDEF0
        );
    }

    #[test]
    fn test_entry_count() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_entry.part");

        let dm = DiskManager::create(&path).expect("Failed to create DiskManager");

        assert_eq!(dm.entry_count().expect("entry_count"), 0);

        dm.set_entry_count(12345).expect("set_entry_count");
        assert_eq!(dm.entry_count().expect("entry_count after set"), 12345);
    }

    #[test]
    fn test_cannot_free_header_block() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_no_free_header.part");

        let dm = DiskManager::create(&path).expect("Failed to create DiskManager");

        let result = dm.free_block(0);
        assert!(result.is_err());

        if let Err(PersistentARTrieError::InvalidBlockId { block_id, reason }) = result {
            assert_eq!(block_id, 0);
            assert!(reason.contains("header"));
        } else {
            panic!("Expected InvalidBlockId error");
        }
    }

    #[test]
    fn test_invalid_block_id() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test_invalid.part");

        let dm = DiskManager::create(&path).expect("Failed to create DiskManager");

        // Try to read a block that doesn't exist
        let mut buf = [0u8; BLOCK_SIZE];
        let result = dm.read_block(999, &mut buf);
        assert!(result.is_err());
    }
}
