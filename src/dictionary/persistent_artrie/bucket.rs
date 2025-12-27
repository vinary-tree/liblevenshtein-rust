//! B-trie String Buckets for Leaf Storage
//!
//! This module implements B-trie style string buckets for efficient leaf storage
//! in the Persistent Adaptive Radix Trie. Each bucket is an 8KB page that stores
//! multiple string suffixes sharing a common prefix (determined by the trie path).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    StringBucket (8KB)                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │  BucketHeader (32 bytes)                                    │
//! │  - magic, version, flags                                    │
//! │  - entry_count, data_offset, free_space                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Directory (variable, sorted by suffix)                     │
//! │  - StringEntry[0]: suffix_offset, suffix_len, value_offset  │
//! │  - StringEntry[1]: ...                                      │
//! │  - ...                                                      │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Free Space                                                 │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Data Area (grows downward from end of page)                │
//! │  - String suffixes and associated values                    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **Sorted Directory**: Binary search for O(log n) lookups within bucket
//! - **Compacted Storage**: Suffixes stored contiguously to maximize cache locality
//! - **B-trie Splits**: When full, bucket splits into sibling buckets
//! - **Variable-Length Values**: Support for arbitrary value types
//!
//! # References
//!
//! - [B-tries for disk-based string management](https://link.springer.com/article/10.1007/s00778-008-0094-1)
//!   (Askitis & Zobel, VLDBJ 2009)

use super::{BUCKET_PAGE_SIZE, MAX_BUCKET_ENTRIES};

/// Magic bytes identifying a string bucket: "PARTBKT\0"
pub const BUCKET_MAGIC: u64 = 0x0054_4B42_5452_4150;

/// Current bucket format version
pub const BUCKET_VERSION: u16 = 1;

/// Header size in bytes
pub const HEADER_SIZE: usize = 32;

/// Size of a directory entry in bytes
pub const ENTRY_SIZE: usize = 8;

/// Maximum data area size (page size - header)
pub const MAX_DATA_SIZE: usize = BUCKET_PAGE_SIZE - HEADER_SIZE;

/// Minimum free space before considering compaction (256 bytes)
pub const MIN_FREE_SPACE: usize = 256;

/// Split threshold as percentage (75% full triggers split)
pub const SPLIT_THRESHOLD: f32 = 0.75;

/// Bucket header flags
pub mod flags {
    /// Bucket has been compacted
    pub const COMPACTED: u16 = 0x0001;
    /// Bucket is marked for split
    pub const NEEDS_SPLIT: u16 = 0x0002;
    /// Bucket contains values (not just keys)
    pub const HAS_VALUES: u16 = 0x0004;
    /// Bucket is a leaf (no child buckets)
    pub const IS_LEAF: u16 = 0x0008;
}

/// Header for a string bucket (32 bytes)
///
/// The header contains metadata about the bucket including the number of entries,
/// where the data area starts, and how much free space remains.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BucketHeader {
    /// Magic bytes for validation: "PARTBKT\0"
    pub magic: u64,
    /// Format version for forward compatibility
    pub version: u16,
    /// Bucket flags (see `flags` module)
    pub flags: u16,
    /// Number of entries in the directory
    pub entry_count: u16,
    /// Reserved for alignment
    pub reserved: u16,
    /// Offset where data area begins (grows downward from page end)
    pub data_start: u32,
    /// Total free space in bytes
    pub free_space: u32,
    /// Checksum for integrity verification (CRC32 of data)
    pub checksum: u32,
}

impl BucketHeader {
    /// Create a new empty bucket header
    pub fn new() -> Self {
        Self {
            magic: BUCKET_MAGIC,
            version: BUCKET_VERSION,
            flags: flags::IS_LEAF,
            entry_count: 0,
            reserved: 0,
            data_start: BUCKET_PAGE_SIZE as u32,
            free_space: MAX_DATA_SIZE as u32,
            checksum: 0,
        }
    }

    /// Create a header with specific flags
    pub fn with_flags(flags: u16) -> Self {
        let mut header = Self::new();
        header.flags = flags | flags::IS_LEAF;
        header
    }

    /// Check if this bucket has values (not just keys)
    #[inline]
    pub fn has_values(&self) -> bool {
        self.flags & flags::HAS_VALUES != 0
    }

    /// Check if this bucket needs to be split
    #[inline]
    pub fn needs_split(&self) -> bool {
        self.flags & flags::NEEDS_SPLIT != 0
    }

    /// Check if this bucket is a leaf
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.flags & flags::IS_LEAF != 0
    }

    /// Check if the bucket is at or above the split threshold
    #[inline]
    pub fn should_split(&self) -> bool {
        let used = MAX_DATA_SIZE as u32 - self.free_space;
        let threshold = (MAX_DATA_SIZE as f32 * SPLIT_THRESHOLD) as u32;
        used >= threshold || self.entry_count as usize >= MAX_BUCKET_ENTRIES
    }

    /// Validate the header magic and version
    pub fn validate(&self) -> Result<(), BucketError> {
        if self.magic != BUCKET_MAGIC {
            return Err(BucketError::InvalidMagic {
                expected: BUCKET_MAGIC,
                found: self.magic,
            });
        }
        if self.version > BUCKET_VERSION {
            return Err(BucketError::UnsupportedVersion {
                max_supported: BUCKET_VERSION,
                found: self.version,
            });
        }
        Ok(())
    }

    /// Calculate the directory end offset
    #[inline]
    pub fn directory_end(&self) -> usize {
        HEADER_SIZE + (self.entry_count as usize * ENTRY_SIZE)
    }

    /// Calculate available space for new entries
    #[inline]
    pub fn available_space(&self) -> usize {
        if self.data_start as usize <= self.directory_end() {
            0
        } else {
            self.data_start as usize - self.directory_end()
        }
    }
}

impl Default for BucketHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// A directory entry pointing to a string suffix in the data area
///
/// Entries are stored sorted by suffix to enable binary search.
/// The entry contains offsets into the data area, not the actual data.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringEntry {
    /// Offset of suffix data from start of page
    pub suffix_offset: u16,
    /// Length of suffix in bytes
    pub suffix_len: u16,
    /// Offset of value data from start of page (0 if no value)
    pub value_offset: u16,
    /// Length of value in bytes (0 if no value)
    pub value_len: u16,
}

impl StringEntry {
    /// Create a new entry for a key-only string (no value)
    pub fn key_only(suffix_offset: u16, suffix_len: u16) -> Self {
        Self {
            suffix_offset,
            suffix_len,
            value_offset: 0,
            value_len: 0,
        }
    }

    /// Create a new entry for a key-value pair
    pub fn with_value(
        suffix_offset: u16,
        suffix_len: u16,
        value_offset: u16,
        value_len: u16,
    ) -> Self {
        Self {
            suffix_offset,
            suffix_len,
            value_offset,
            value_len,
        }
    }

    /// Check if this entry has an associated value
    #[inline]
    pub fn has_value(&self) -> bool {
        self.value_len > 0
    }

    /// Total size of this entry's data (suffix + value)
    #[inline]
    pub fn data_size(&self) -> usize {
        self.suffix_len as usize + self.value_len as usize
    }

    /// Serialize entry to bytes
    pub fn to_bytes(&self) -> [u8; ENTRY_SIZE] {
        let mut bytes = [0u8; ENTRY_SIZE];
        bytes[0..2].copy_from_slice(&self.suffix_offset.to_le_bytes());
        bytes[2..4].copy_from_slice(&self.suffix_len.to_le_bytes());
        bytes[4..6].copy_from_slice(&self.value_offset.to_le_bytes());
        bytes[6..8].copy_from_slice(&self.value_len.to_le_bytes());
        bytes
    }

    /// Deserialize entry from bytes
    pub fn from_bytes(bytes: &[u8; ENTRY_SIZE]) -> Self {
        Self {
            suffix_offset: u16::from_le_bytes([bytes[0], bytes[1]]),
            suffix_len: u16::from_le_bytes([bytes[2], bytes[3]]),
            value_offset: u16::from_le_bytes([bytes[4], bytes[5]]),
            value_len: u16::from_le_bytes([bytes[6], bytes[7]]),
        }
    }
}

/// An 8KB string bucket page
///
/// This is the in-memory representation of a bucket. The raw page data
/// is stored in `data` and accessed through the directory entries.
#[derive(Clone)]
pub struct StringBucket {
    /// The raw page data (8KB)
    data: Box<[u8; BUCKET_PAGE_SIZE]>,
}

impl StringBucket {
    /// Create a new empty bucket
    pub fn new() -> Self {
        let mut data = Box::new([0u8; BUCKET_PAGE_SIZE]);
        let header = BucketHeader::new();
        Self::write_header(&mut data, &header);
        Self { data }
    }

    /// Create a bucket with values enabled
    pub fn with_values() -> Self {
        let mut data = Box::new([0u8; BUCKET_PAGE_SIZE]);
        let header = BucketHeader::with_flags(flags::HAS_VALUES);
        Self::write_header(&mut data, &header);
        Self { data }
    }

    /// Create a bucket from raw page data
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BucketError> {
        if bytes.len() != BUCKET_PAGE_SIZE {
            return Err(BucketError::InvalidSize {
                expected: BUCKET_PAGE_SIZE,
                found: bytes.len(),
            });
        }

        let mut data = Box::new([0u8; BUCKET_PAGE_SIZE]);
        data.copy_from_slice(bytes);

        let bucket = Self { data };
        bucket.header().validate()?;
        Ok(bucket)
    }

    /// Get the raw page data
    pub fn as_bytes(&self) -> &[u8; BUCKET_PAGE_SIZE] {
        &self.data
    }

    /// Get a mutable reference to the raw page data
    pub fn as_bytes_mut(&mut self) -> &mut [u8; BUCKET_PAGE_SIZE] {
        &mut self.data
    }

    /// Read the bucket header
    pub fn header(&self) -> BucketHeader {
        Self::read_header(&self.data)
    }

    /// Update the bucket header
    pub fn set_header(&mut self, header: &BucketHeader) {
        Self::write_header(&mut self.data, header);
    }

    /// Number of entries in the bucket
    #[inline]
    pub fn len(&self) -> usize {
        self.header().entry_count as usize
    }

    /// Check if the bucket is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if the bucket is full (no more entries can be added)
    pub fn is_full(&self) -> bool {
        let header = self.header();
        header.entry_count as usize >= MAX_BUCKET_ENTRIES
            || header.available_space() < ENTRY_SIZE + MIN_FREE_SPACE
    }

    /// Get the directory entry at the given index
    pub fn get_entry(&self, index: usize) -> Option<StringEntry> {
        let header = self.header();
        if index >= header.entry_count as usize {
            return None;
        }

        let offset = HEADER_SIZE + (index * ENTRY_SIZE);
        let bytes: [u8; ENTRY_SIZE] = self.data[offset..offset + ENTRY_SIZE]
            .try_into()
            .expect("slice length matches ENTRY_SIZE");
        Some(StringEntry::from_bytes(&bytes))
    }

    /// Get the suffix bytes for an entry
    pub fn get_suffix(&self, entry: &StringEntry) -> &[u8] {
        let start = entry.suffix_offset as usize;
        let end = start + entry.suffix_len as usize;
        &self.data[start..end]
    }

    /// Get the value bytes for an entry (if present)
    pub fn get_value(&self, entry: &StringEntry) -> Option<&[u8]> {
        if entry.value_len == 0 {
            return None;
        }
        let start = entry.value_offset as usize;
        let end = start + entry.value_len as usize;
        Some(&self.data[start..end])
    }

    /// Binary search for a suffix in the directory
    ///
    /// Returns `Ok(index)` if found, `Err(index)` where the suffix should be inserted.
    pub fn search(&self, suffix: &[u8]) -> Result<usize, usize> {
        let header = self.header();
        let count = header.entry_count as usize;

        if count == 0 {
            return Err(0);
        }

        let mut left = 0;
        let mut right = count;

        while left < right {
            let mid = left + (right - left) / 2;
            let entry = self.get_entry(mid).expect("valid index");
            let entry_suffix = self.get_suffix(&entry);

            match entry_suffix.cmp(suffix) {
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }

        Err(left)
    }

    /// Check if the bucket contains the given suffix
    pub fn contains(&self, suffix: &[u8]) -> bool {
        self.search(suffix).is_ok()
    }

    /// Insert a key-only entry (no value)
    ///
    /// Returns `Ok(true)` if inserted, `Ok(false)` if already exists.
    pub fn insert_key(&mut self, suffix: &[u8]) -> Result<bool, BucketError> {
        self.insert_impl(suffix, None)
    }

    /// Insert a key-value pair
    ///
    /// Returns `Ok(true)` if inserted, `Ok(false)` if already exists (value updated).
    pub fn insert(&mut self, suffix: &[u8], value: &[u8]) -> Result<bool, BucketError> {
        self.insert_impl(suffix, Some(value))
    }

    /// Internal insert implementation
    fn insert_impl(&mut self, suffix: &[u8], value: Option<&[u8]>) -> Result<bool, BucketError> {
        let suffix_len = suffix.len();
        let value_len = value.map_or(0, |v| v.len());
        let total_data_size = suffix_len + value_len;

        // Check size limits
        if suffix_len > u16::MAX as usize || value_len > u16::MAX as usize {
            return Err(BucketError::DataTooLarge {
                max: u16::MAX as usize,
                found: suffix_len.max(value_len),
            });
        }

        let mut header = self.header();

        // Check for existing entry
        match self.search(suffix) {
            Ok(index) => {
                // Entry exists - update value if provided
                if let Some(new_value) = value {
                    self.update_value_at(index, new_value)?;
                }
                return Ok(false);
            }
            Err(insert_pos) => {
                // Check space requirements
                let space_needed = ENTRY_SIZE + total_data_size;
                if header.available_space() < space_needed {
                    return Err(BucketError::InsufficientSpace {
                        needed: space_needed,
                        available: header.available_space(),
                    });
                }

                if header.entry_count as usize >= MAX_BUCKET_ENTRIES {
                    return Err(BucketError::BucketFull);
                }

                // Allocate space in data area (grows downward)
                let new_data_start = header.data_start as usize - total_data_size;

                // Write suffix
                self.data[new_data_start..new_data_start + suffix_len].copy_from_slice(suffix);

                // Write value if present
                let value_offset = if let Some(v) = value {
                    let offset = new_data_start + suffix_len;
                    self.data[offset..offset + value_len].copy_from_slice(v);
                    offset as u16
                } else {
                    0
                };

                // Create directory entry
                let entry = StringEntry {
                    suffix_offset: new_data_start as u16,
                    suffix_len: suffix_len as u16,
                    value_offset,
                    value_len: value_len as u16,
                };

                // Insert entry at correct position (shift existing entries)
                self.insert_entry_at(insert_pos, &entry);

                // Update header
                header.entry_count += 1;
                header.data_start = new_data_start as u32;
                header.free_space -= space_needed as u32;

                if value.is_some() {
                    header.flags |= flags::HAS_VALUES;
                }

                self.set_header(&header);
                Ok(true)
            }
        }
    }

    /// Update the value for an entry at the given index
    fn update_value_at(&mut self, index: usize, new_value: &[u8]) -> Result<(), BucketError> {
        let entry = self.get_entry(index).ok_or(BucketError::InvalidIndex)?;

        // If new value fits in existing space, update in place
        if new_value.len() <= entry.value_len as usize {
            let offset = entry.value_offset as usize;
            self.data[offset..offset + new_value.len()].copy_from_slice(new_value);

            // Update entry with new length
            let new_entry = StringEntry {
                value_len: new_value.len() as u16,
                ..entry
            };
            self.write_entry_at(index, &new_entry);
            return Ok(());
        }

        // Otherwise, need to allocate new space
        // This is a simplified implementation - a production version would
        // track fragmentation and compact when needed
        let mut header = self.header();
        let space_needed = new_value.len();

        if header.available_space() < space_needed {
            return Err(BucketError::InsufficientSpace {
                needed: space_needed,
                available: header.available_space(),
            });
        }

        let new_offset = header.data_start as usize - space_needed;
        self.data[new_offset..new_offset + new_value.len()].copy_from_slice(new_value);

        let new_entry = StringEntry {
            value_offset: new_offset as u16,
            value_len: new_value.len() as u16,
            ..entry
        };
        self.write_entry_at(index, &new_entry);

        header.data_start = new_offset as u32;
        header.free_space -= space_needed as u32;
        self.set_header(&header);

        Ok(())
    }

    /// Insert an entry at the given position in the directory
    fn insert_entry_at(&mut self, index: usize, entry: &StringEntry) {
        let header = self.header();
        let count = header.entry_count as usize;

        // Shift existing entries to make room
        if index < count {
            let src_start = HEADER_SIZE + (index * ENTRY_SIZE);
            let src_end = HEADER_SIZE + (count * ENTRY_SIZE);
            let dst_start = src_start + ENTRY_SIZE;

            // Move entries one position forward
            self.data.copy_within(src_start..src_end, dst_start);
        }

        // Write the new entry
        self.write_entry_at(index, entry);
    }

    /// Write an entry at the given position in the directory
    fn write_entry_at(&mut self, index: usize, entry: &StringEntry) {
        let offset = HEADER_SIZE + (index * ENTRY_SIZE);
        let bytes = entry.to_bytes();
        self.data[offset..offset + ENTRY_SIZE].copy_from_slice(&bytes);
    }

    /// Remove an entry by suffix
    ///
    /// Returns the removed entry if found.
    pub fn remove(&mut self, suffix: &[u8]) -> Option<StringEntry> {
        match self.search(suffix) {
            Ok(index) => {
                let entry = self.get_entry(index).expect("valid index after search");

                // Shift entries to fill the gap
                let mut header = self.header();
                let count = header.entry_count as usize;

                if index + 1 < count {
                    let src_start = HEADER_SIZE + ((index + 1) * ENTRY_SIZE);
                    let src_end = HEADER_SIZE + (count * ENTRY_SIZE);
                    let dst_start = HEADER_SIZE + (index * ENTRY_SIZE);
                    self.data.copy_within(src_start..src_end, dst_start);
                }

                // Update header
                header.entry_count -= 1;
                // Note: we don't reclaim data space here - compaction handles that
                self.set_header(&header);

                Some(entry)
            }
            Err(_) => None,
        }
    }

    /// Iterate over all entries in sorted order
    pub fn iter(&self) -> BucketIterator<'_> {
        BucketIterator {
            bucket: self,
            index: 0,
            count: self.header().entry_count as usize,
        }
    }

    /// Get the split point for this bucket (median entry)
    pub fn split_point(&self) -> Option<usize> {
        let count = self.len();
        if count < 2 {
            return None;
        }
        Some(count / 2)
    }

    /// Split this bucket at the median into two new buckets
    ///
    /// Returns `(left_bucket, right_bucket, split_key)` where:
    /// - `left_bucket` contains entries [0, split_point)
    /// - `right_bucket` contains entries [split_point, count)
    /// - `split_key` is the first key in the right bucket (for routing)
    ///
    /// Returns `None` if the bucket has fewer than 2 entries.
    pub fn split(&self) -> Option<SplitResult> {
        let split_point = self.split_point()?;
        let count = self.len();
        let has_values = self.header().has_values();

        // Create new buckets
        let mut left = if has_values {
            StringBucket::with_values()
        } else {
            StringBucket::new()
        };

        let mut right = if has_values {
            StringBucket::with_values()
        } else {
            StringBucket::new()
        };

        // Track the split key
        let mut split_key = Vec::new();

        // Copy entries to left bucket (0..split_point)
        for i in 0..split_point {
            let entry = self.get_entry(i).expect("valid index");
            let suffix = self.get_suffix(&entry);
            let value = self.get_value(&entry);

            if let Some(v) = value {
                left.insert(suffix, v).expect("left bucket has space");
            } else {
                left.insert_key(suffix).expect("left bucket has space");
            }
        }

        // Copy entries to right bucket (split_point..count)
        for i in split_point..count {
            let entry = self.get_entry(i).expect("valid index");
            let suffix = self.get_suffix(&entry);
            let value = self.get_value(&entry);

            // Capture the split key (first key in right bucket)
            if i == split_point {
                split_key = suffix.to_vec();
            }

            if let Some(v) = value {
                right.insert(suffix, v).expect("right bucket has space");
            } else {
                right.insert_key(suffix).expect("right bucket has space");
            }
        }

        Some(SplitResult {
            left,
            right,
            split_key,
        })
    }

    /// Split this bucket into multiple buckets based on the first byte of each suffix
    ///
    /// This is used when a bucket should be converted into an ART node with
    /// child buckets. Returns a map from first-byte to child bucket.
    ///
    /// Entries with empty suffixes are collected into a special "final" list.
    pub fn split_by_first_byte(&self) -> SplitByByteResult {
        let has_values = self.header().has_values();
        let mut buckets: std::collections::BTreeMap<u8, StringBucket> =
            std::collections::BTreeMap::new();
        let mut finals: Vec<(Vec<u8>, Option<Vec<u8>>)> = Vec::new();

        for i in 0..self.len() {
            let entry = self.get_entry(i).expect("valid index");
            let suffix = self.get_suffix(&entry);
            let value = self.get_value(&entry);

            if suffix.is_empty() {
                // Empty suffix means this is a final state for the parent prefix
                finals.push((Vec::new(), value.map(|v| v.to_vec())));
            } else {
                // Get or create bucket for this first byte
                let first_byte = suffix[0];
                let remaining = &suffix[1..];

                let bucket = buckets.entry(first_byte).or_insert_with(|| {
                    if has_values {
                        StringBucket::with_values()
                    } else {
                        StringBucket::new()
                    }
                });

                if let Some(v) = value {
                    bucket.insert(remaining, v).expect("child bucket has space");
                } else {
                    bucket.insert_key(remaining).expect("child bucket has space");
                }
            }
        }

        SplitByByteResult { buckets, finals }
    }

    /// Compact the bucket to reclaim fragmented space
    ///
    /// This rebuilds the data area contiguously, eliminating gaps from
    /// deleted entries or updated values.
    pub fn compact(&mut self) {
        let count = self.len();
        if count == 0 {
            return;
        }

        let has_values = self.header().has_values();

        // Collect all entries with their data
        let entries: Vec<_> = (0..count)
            .map(|i| {
                let entry = self.get_entry(i).expect("valid index");
                let suffix = self.get_suffix(&entry).to_vec();
                let value = self.get_value(&entry).map(|v| v.to_vec());
                (suffix, value)
            })
            .collect();

        // Reset bucket
        *self = if has_values {
            StringBucket::with_values()
        } else {
            StringBucket::new()
        };

        // Re-insert all entries
        for (suffix, value) in entries {
            if let Some(v) = value {
                self.insert(&suffix, &v).expect("re-insert succeeds");
            } else {
                self.insert_key(&suffix).expect("re-insert succeeds");
            }
        }

        // Mark as compacted
        let mut header = self.header();
        header.flags |= flags::COMPACTED;
        self.set_header(&header);
    }

    /// Merge another bucket into this one
    ///
    /// The other bucket's entries are inserted into this bucket.
    /// Returns an error if there isn't enough space.
    pub fn merge(&mut self, other: &StringBucket) -> Result<(), BucketError> {
        for i in 0..other.len() {
            let entry = other.get_entry(i).expect("valid index");
            let suffix = other.get_suffix(&entry);
            let value = other.get_value(&entry);

            if let Some(v) = value {
                self.insert(suffix, v)?;
            } else {
                self.insert_key(suffix)?;
            }
        }
        Ok(())
    }

    /// Read header from raw data
    fn read_header(data: &[u8; BUCKET_PAGE_SIZE]) -> BucketHeader {
        BucketHeader {
            magic: u64::from_le_bytes(data[0..8].try_into().expect("slice length is 8")),
            version: u16::from_le_bytes(data[8..10].try_into().expect("slice length is 2")),
            flags: u16::from_le_bytes(data[10..12].try_into().expect("slice length is 2")),
            entry_count: u16::from_le_bytes(data[12..14].try_into().expect("slice length is 2")),
            reserved: u16::from_le_bytes(data[14..16].try_into().expect("slice length is 2")),
            data_start: u32::from_le_bytes(data[16..20].try_into().expect("slice length is 4")),
            free_space: u32::from_le_bytes(data[20..24].try_into().expect("slice length is 4")),
            checksum: u32::from_le_bytes(data[24..28].try_into().expect("slice length is 4")),
        }
    }

    /// Write header to raw data
    fn write_header(data: &mut [u8; BUCKET_PAGE_SIZE], header: &BucketHeader) {
        data[0..8].copy_from_slice(&header.magic.to_le_bytes());
        data[8..10].copy_from_slice(&header.version.to_le_bytes());
        data[10..12].copy_from_slice(&header.flags.to_le_bytes());
        data[12..14].copy_from_slice(&header.entry_count.to_le_bytes());
        data[14..16].copy_from_slice(&header.reserved.to_le_bytes());
        data[16..20].copy_from_slice(&header.data_start.to_le_bytes());
        data[20..24].copy_from_slice(&header.free_space.to_le_bytes());
        data[24..28].copy_from_slice(&header.checksum.to_le_bytes());
    }
}

impl Default for StringBucket {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StringBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = self.header();
        f.debug_struct("StringBucket")
            .field("entry_count", &header.entry_count)
            .field("data_start", &header.data_start)
            .field("free_space", &header.free_space)
            .field("has_values", &header.has_values())
            .finish()
    }
}

/// Iterator over bucket entries
pub struct BucketIterator<'a> {
    bucket: &'a StringBucket,
    index: usize,
    count: usize,
}

impl<'a> Iterator for BucketIterator<'a> {
    type Item = (StringEntry, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.count {
            return None;
        }

        let entry = self.bucket.get_entry(self.index)?;
        let suffix = self.bucket.get_suffix(&entry);
        self.index += 1;
        Some((entry, suffix))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BucketIterator<'_> {}

/// Result of splitting a bucket at the median
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// Left bucket containing entries [0, split_point)
    pub left: StringBucket,
    /// Right bucket containing entries [split_point, count)
    pub right: StringBucket,
    /// The first key in the right bucket (routing key)
    pub split_key: Vec<u8>,
}

/// Result of splitting a bucket by first byte
#[derive(Debug)]
pub struct SplitByByteResult {
    /// Map from first byte to child bucket
    pub buckets: std::collections::BTreeMap<u8, StringBucket>,
    /// Entries with empty suffixes (represent final states)
    /// Each element is (empty_vec, optional_value)
    pub finals: Vec<(Vec<u8>, Option<Vec<u8>>)>,
}

/// Errors that can occur during bucket operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BucketError {
    /// Invalid magic bytes in header
    InvalidMagic {
        /// Expected magic value
        expected: u64,
        /// Found magic value
        found: u64,
    },
    /// Unsupported bucket format version
    UnsupportedVersion {
        /// Maximum supported version
        max_supported: u16,
        /// Version found in bucket
        found: u16,
    },
    /// Invalid bucket size
    InvalidSize {
        /// Expected size
        expected: usize,
        /// Found size
        found: usize,
    },
    /// Not enough space for operation
    InsufficientSpace {
        /// Space needed
        needed: usize,
        /// Space available
        available: usize,
    },
    /// Bucket has reached maximum entries
    BucketFull,
    /// Data too large for bucket
    DataTooLarge {
        /// Maximum allowed size
        max: usize,
        /// Actual size
        found: usize,
    },
    /// Invalid entry index
    InvalidIndex,
    /// Bucket data corruption detected
    Corrupted(String),
}

impl std::fmt::Display for BucketError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BucketError::InvalidMagic { expected, found } => {
                write!(
                    f,
                    "invalid bucket magic: expected 0x{:016x}, found 0x{:016x}",
                    expected, found
                )
            }
            BucketError::UnsupportedVersion {
                max_supported,
                found,
            } => {
                write!(
                    f,
                    "unsupported bucket version: max supported {}, found {}",
                    max_supported, found
                )
            }
            BucketError::InvalidSize { expected, found } => {
                write!(f, "invalid bucket size: expected {}, found {}", expected, found)
            }
            BucketError::InsufficientSpace { needed, available } => {
                write!(
                    f,
                    "insufficient space: need {} bytes, have {} available",
                    needed, available
                )
            }
            BucketError::BucketFull => write!(f, "bucket is full"),
            BucketError::DataTooLarge { max, found } => {
                write!(f, "data too large: max {} bytes, found {}", max, found)
            }
            BucketError::InvalidIndex => write!(f, "invalid entry index"),
            BucketError::Corrupted(msg) => write!(f, "bucket corrupted: {}", msg),
        }
    }
}

impl std::error::Error for BucketError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_bucket() {
        let bucket = StringBucket::new();
        let header = bucket.header();

        assert_eq!(header.magic, BUCKET_MAGIC);
        assert_eq!(header.version, BUCKET_VERSION);
        assert_eq!(header.entry_count, 0);
        assert!(bucket.is_empty());
        assert!(!bucket.is_full());
    }

    #[test]
    fn test_insert_and_search() {
        let mut bucket = StringBucket::new();

        // Insert some entries
        assert!(bucket.insert_key(b"apple").unwrap());
        assert!(bucket.insert_key(b"banana").unwrap());
        assert!(bucket.insert_key(b"cherry").unwrap());

        assert_eq!(bucket.len(), 3);

        // Search for entries
        assert!(bucket.contains(b"apple"));
        assert!(bucket.contains(b"banana"));
        assert!(bucket.contains(b"cherry"));
        assert!(!bucket.contains(b"date"));

        // Verify sorted order
        let entries: Vec<_> = bucket.iter().map(|(_, s)| s.to_vec()).collect();
        assert_eq!(entries, vec![b"apple".to_vec(), b"banana".to_vec(), b"cherry".to_vec()]);
    }

    #[test]
    fn test_insert_with_values() {
        let mut bucket = StringBucket::with_values();

        bucket.insert(b"key1", b"value1").unwrap();
        bucket.insert(b"key2", b"value2").unwrap();

        let header = bucket.header();
        assert!(header.has_values());

        // Retrieve values
        match bucket.search(b"key1") {
            Ok(idx) => {
                let entry = bucket.get_entry(idx).unwrap();
                let value = bucket.get_value(&entry).unwrap();
                assert_eq!(value, b"value1");
            }
            Err(_) => panic!("key1 not found"),
        }
    }

    #[test]
    fn test_duplicate_insert() {
        let mut bucket = StringBucket::new();

        assert!(bucket.insert_key(b"test").unwrap()); // First insert returns true
        assert!(!bucket.insert_key(b"test").unwrap()); // Duplicate returns false

        assert_eq!(bucket.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut bucket = StringBucket::new();

        bucket.insert_key(b"apple").unwrap();
        bucket.insert_key(b"banana").unwrap();
        bucket.insert_key(b"cherry").unwrap();

        assert_eq!(bucket.len(), 3);

        // Remove middle entry
        let removed = bucket.remove(b"banana");
        assert!(removed.is_some());
        assert_eq!(bucket.len(), 2);

        // Verify remaining entries are still in order
        assert!(bucket.contains(b"apple"));
        assert!(!bucket.contains(b"banana"));
        assert!(bucket.contains(b"cherry"));

        let entries: Vec<_> = bucket.iter().map(|(_, s)| s.to_vec()).collect();
        assert_eq!(entries, vec![b"apple".to_vec(), b"cherry".to_vec()]);
    }

    #[test]
    fn test_remove_not_found() {
        let mut bucket = StringBucket::new();
        bucket.insert_key(b"apple").unwrap();

        let removed = bucket.remove(b"banana");
        assert!(removed.is_none());
        assert_eq!(bucket.len(), 1);
    }

    #[test]
    fn test_sorted_insertion() {
        let mut bucket = StringBucket::new();

        // Insert in reverse order
        bucket.insert_key(b"zebra").unwrap();
        bucket.insert_key(b"apple").unwrap();
        bucket.insert_key(b"mango").unwrap();

        // Verify sorted order is maintained
        let entries: Vec<_> = bucket.iter().map(|(_, s)| s.to_vec()).collect();
        assert_eq!(entries, vec![b"apple".to_vec(), b"mango".to_vec(), b"zebra".to_vec()]);
    }

    #[test]
    fn test_binary_search() {
        let mut bucket = StringBucket::new();

        for i in 0..50 {
            let key = format!("key{:03}", i);
            bucket.insert_key(key.as_bytes()).unwrap();
        }

        assert_eq!(bucket.len(), 50);

        // Test binary search works correctly
        assert_eq!(bucket.search(b"key000"), Ok(0));
        assert_eq!(bucket.search(b"key025"), Ok(25));
        assert_eq!(bucket.search(b"key049"), Ok(49));

        // Test insertion points for missing keys
        assert_eq!(bucket.search(b"key000a"), Err(1));
        assert_eq!(bucket.search(b"aaa"), Err(0)); // Before all
        assert_eq!(bucket.search(b"zzz"), Err(50)); // After all
    }

    #[test]
    fn test_bucket_serialization() {
        let mut bucket = StringBucket::new();

        bucket.insert_key(b"hello").unwrap();
        bucket.insert_key(b"world").unwrap();

        // Serialize
        let bytes = bucket.as_bytes().clone();

        // Deserialize
        let bucket2 = StringBucket::from_bytes(&bytes).unwrap();

        assert_eq!(bucket2.len(), 2);
        assert!(bucket2.contains(b"hello"));
        assert!(bucket2.contains(b"world"));
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = [0u8; BUCKET_PAGE_SIZE];
        data[0..8].copy_from_slice(&0u64.to_le_bytes()); // Wrong magic

        let result = StringBucket::from_bytes(&data);
        assert!(matches!(result, Err(BucketError::InvalidMagic { .. })));
    }

    #[test]
    fn test_header_validation() {
        let header = BucketHeader::new();
        assert!(header.validate().is_ok());

        let invalid_header = BucketHeader {
            magic: 0,
            ..header
        };
        assert!(matches!(
            invalid_header.validate(),
            Err(BucketError::InvalidMagic { .. })
        ));
    }

    #[test]
    fn test_split_threshold() {
        let mut bucket = StringBucket::new();

        // Fill bucket until it should split
        let mut i = 0;
        while !bucket.header().should_split() && i < MAX_BUCKET_ENTRIES {
            let key = format!("key{:06}", i);
            if bucket.insert_key(key.as_bytes()).is_err() {
                break;
            }
            i += 1;
        }

        // Should eventually hit the threshold
        assert!(bucket.header().should_split() || bucket.is_full());
    }

    #[test]
    fn test_entry_serialization() {
        let entry = StringEntry {
            suffix_offset: 1234,
            suffix_len: 56,
            value_offset: 7890,
            value_len: 12,
        };

        let bytes = entry.to_bytes();
        let restored = StringEntry::from_bytes(&bytes);

        assert_eq!(entry, restored);
    }

    #[test]
    fn test_empty_suffix() {
        let mut bucket = StringBucket::new();

        // Insert empty suffix (represents a word that ends at this trie node)
        assert!(bucket.insert_key(b"").unwrap());
        assert!(bucket.contains(b""));
        assert_eq!(bucket.len(), 1);
    }

    #[test]
    fn test_update_value() {
        let mut bucket = StringBucket::with_values();

        bucket.insert(b"key", b"old_value").unwrap();

        // Update with new value
        bucket.insert(b"key", b"new").unwrap();

        match bucket.search(b"key") {
            Ok(idx) => {
                let entry = bucket.get_entry(idx).unwrap();
                let value = bucket.get_value(&entry).unwrap();
                assert_eq!(value, b"new");
            }
            Err(_) => panic!("key not found"),
        }
    }

    #[test]
    fn test_split_empty_bucket() {
        let bucket = StringBucket::new();
        assert!(bucket.split().is_none());
    }

    #[test]
    fn test_split_single_entry() {
        let mut bucket = StringBucket::new();
        bucket.insert_key(b"test").unwrap();
        assert!(bucket.split().is_none());
    }

    #[test]
    fn test_split_basic() {
        let mut bucket = StringBucket::new();

        // Insert 10 entries
        for i in 0..10 {
            let key = format!("key{:02}", i);
            bucket.insert_key(key.as_bytes()).unwrap();
        }

        let result = bucket.split().expect("should split");

        // Left bucket should have entries 0-4
        assert_eq!(result.left.len(), 5);
        assert!(result.left.contains(b"key00"));
        assert!(result.left.contains(b"key04"));
        assert!(!result.left.contains(b"key05"));

        // Right bucket should have entries 5-9
        assert_eq!(result.right.len(), 5);
        assert!(result.right.contains(b"key05"));
        assert!(result.right.contains(b"key09"));
        assert!(!result.right.contains(b"key04"));

        // Split key should be the first key in right bucket
        assert_eq!(result.split_key, b"key05");
    }

    #[test]
    fn test_split_with_values() {
        let mut bucket = StringBucket::with_values();

        bucket.insert(b"a", b"val_a").unwrap();
        bucket.insert(b"b", b"val_b").unwrap();
        bucket.insert(b"c", b"val_c").unwrap();
        bucket.insert(b"d", b"val_d").unwrap();

        let result = bucket.split().expect("should split");

        // Both buckets should preserve has_values flag
        assert!(result.left.header().has_values());
        assert!(result.right.header().has_values());

        // Check values are preserved
        if let Ok(idx) = result.left.search(b"a") {
            let entry = result.left.get_entry(idx).unwrap();
            assert_eq!(result.left.get_value(&entry), Some(b"val_a".as_slice()));
        }
    }

    #[test]
    fn test_split_by_first_byte() {
        let mut bucket = StringBucket::new();

        bucket.insert_key(b"apple").unwrap();
        bucket.insert_key(b"apricot").unwrap();
        bucket.insert_key(b"banana").unwrap();
        bucket.insert_key(b"berry").unwrap();
        bucket.insert_key(b"cherry").unwrap();
        bucket.insert_key(b"").unwrap(); // Empty suffix (final state)

        let result = bucket.split_by_first_byte();

        // Should have 3 child buckets (a, b, c)
        assert_eq!(result.buckets.len(), 3);

        // 'a' bucket should have "pple" and "pricot"
        let a_bucket = result.buckets.get(&b'a').expect("'a' bucket exists");
        assert_eq!(a_bucket.len(), 2);
        assert!(a_bucket.contains(b"pple"));
        assert!(a_bucket.contains(b"pricot"));

        // 'b' bucket should have "anana" and "erry"
        let b_bucket = result.buckets.get(&b'b').expect("'b' bucket exists");
        assert_eq!(b_bucket.len(), 2);
        assert!(b_bucket.contains(b"anana"));
        assert!(b_bucket.contains(b"erry"));

        // 'c' bucket should have "herry"
        let c_bucket = result.buckets.get(&b'c').expect("'c' bucket exists");
        assert_eq!(c_bucket.len(), 1);
        assert!(c_bucket.contains(b"herry"));

        // Empty suffix should be in finals
        assert_eq!(result.finals.len(), 1);
        assert_eq!(result.finals[0].0, Vec::<u8>::new());
    }

    #[test]
    fn test_compact() {
        let mut bucket = StringBucket::new();

        // Insert and remove to create fragmentation
        bucket.insert_key(b"aaa").unwrap();
        bucket.insert_key(b"bbb").unwrap();
        bucket.insert_key(b"ccc").unwrap();
        bucket.remove(b"bbb");

        let before_compact = bucket.header().free_space;

        bucket.compact();

        // After compaction, should have reclaimed space
        let after_compact = bucket.header().free_space;
        assert!(after_compact >= before_compact);

        // Data should still be accessible
        assert!(bucket.contains(b"aaa"));
        assert!(!bucket.contains(b"bbb"));
        assert!(bucket.contains(b"ccc"));

        // Compacted flag should be set
        assert!(bucket.header().flags & flags::COMPACTED != 0);
    }

    #[test]
    fn test_merge_buckets() {
        let mut bucket1 = StringBucket::new();
        let mut bucket2 = StringBucket::new();

        bucket1.insert_key(b"apple").unwrap();
        bucket1.insert_key(b"cherry").unwrap();

        bucket2.insert_key(b"banana").unwrap();
        bucket2.insert_key(b"date").unwrap();

        bucket1.merge(&bucket2).unwrap();

        assert_eq!(bucket1.len(), 4);
        assert!(bucket1.contains(b"apple"));
        assert!(bucket1.contains(b"banana"));
        assert!(bucket1.contains(b"cherry"));
        assert!(bucket1.contains(b"date"));

        // Should be sorted
        let entries: Vec<_> = bucket1.iter().map(|(_, s)| s.to_vec()).collect();
        assert_eq!(
            entries,
            vec![
                b"apple".to_vec(),
                b"banana".to_vec(),
                b"cherry".to_vec(),
                b"date".to_vec()
            ]
        );
    }

    #[test]
    fn test_split_and_merge_roundtrip() {
        let mut original = StringBucket::new();

        for i in 0..20 {
            let key = format!("key{:02}", i);
            original.insert_key(key.as_bytes()).unwrap();
        }

        let original_entries: Vec<_> = original.iter().map(|(_, s)| s.to_vec()).collect();

        // Split
        let result = original.split().expect("should split");

        // Merge back
        let mut merged = result.left;
        merged.merge(&result.right).unwrap();

        // Should have all original entries
        let merged_entries: Vec<_> = merged.iter().map(|(_, s)| s.to_vec()).collect();
        assert_eq!(original_entries, merged_entries);
    }
}
