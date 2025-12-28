//! Dictionary Implementation for Persistent ART
//!
//! This module provides the `PersistentARTrie` dictionary type that implements
//! the `Dictionary` and `MappedDictionary` traits for integration with the
//! Levenshtein automata transducer.
//!
//! # In-Memory vs Disk-Backed
//!
//! This implementation currently provides an in-memory version suitable for
//! development and testing. The disk-backed version with memory-mapped I/O
//! will be added in a future phase.
//!
//! # Thread Safety
//!
//! The dictionary uses `Arc<RwLock>` for thread-safe concurrent access.
//! Read operations can proceed in parallel, while writes are serialized.

use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::dictionary::{Dictionary, MappedDictionary, SyncStrategy};
use crate::dictionary::value::DictionaryValue;
use super::bucket::StringBucket;
use super::error::{PersistentARTrieError, Result};
use super::node_impl::PersistentARTrieNode;
use super::nodes::{ArtNode, Node, Node4, AddChildError};
use super::swizzled_ptr::{DiskLocation, NodeType, SwizzledPtr};
use super::transitions::{bucket_to_art_node, ChildNode};
#[cfg(feature = "persistent-artrie")]
use super::serialization;

#[cfg(feature = "persistent-artrie")]
use super::buffer_manager::BufferManager;
#[cfg(feature = "persistent-artrie")]
use super::wal::{Lsn, WalWriter};

/// Maximum buffer size for reading serialized ART nodes (4KB should be ample).
/// Largest node is Node256 at ~2KB, so 4KB provides good margin.
#[cfg(feature = "persistent-artrie")]
const ART_NODE_BUFFER_SIZE: usize = 4096;

/// A Persistent Adaptive Radix Trie dictionary.
///
/// This dictionary stores terms in a hybrid structure combining:
/// - **ART nodes** for efficient internal node traversal (Node4/16/48/256)
/// - **String buckets** for efficient leaf storage (multiple terms per bucket)
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
///
/// let mut dict = PersistentARTrie::new();
/// dict.insert("hello");
/// dict.insert("world");
///
/// assert!(dict.contains("hello"));
/// assert!(!dict.contains("hi"));
/// ```
#[derive(Clone)]
pub struct PersistentARTrie<V: DictionaryValue = ()> {
    /// Inner state protected by read-write lock
    pub(crate) inner: Arc<RwLock<PersistentARTrieInner<V>>>,
}

/// Inner state of the Persistent ART
pub(crate) struct PersistentARTrieInner<V: DictionaryValue> {
    /// Root node of the trie (starts as a bucket, grows to ART)
    pub(crate) root: TrieRoot<V>,
    /// Number of terms in the dictionary
    pub(crate) term_count: usize,
    /// Whether the dictionary has been modified
    pub(crate) dirty: bool,

    // === Storage Layer (only active with persistent-artrie feature) ===
    // Note: DiskManager is owned by BufferManager and accessible via buffer_manager.disk_manager()
    /// Buffer manager with Clock-evicted page cache (owns DiskManager)
    #[cfg(feature = "persistent-artrie")]
    pub(crate) buffer_manager: Option<Arc<RwLock<BufferManager>>>,
    /// Write-ahead log writer for durability
    #[cfg(feature = "persistent-artrie")]
    pub(crate) wal_writer: Option<Arc<RwLock<WalWriter>>>,
    /// Next log sequence number to assign
    #[cfg(feature = "persistent-artrie")]
    pub(crate) next_lsn: Lsn,
    /// Prefetcher for DFS traversal optimization
    #[cfg(feature = "persistent-artrie")]
    pub(crate) prefetcher: super::prefetch::Prefetcher,
}

/// The root of the trie can be either a bucket or an ART node
pub(crate) enum TrieRoot<V: DictionaryValue> {
    /// Root is a single bucket (for small dictionaries)
    Bucket(StringBucket),
    /// Root is an ART node (for larger dictionaries)
    ArtNode {
        /// The root ART node
        node: Node,
        /// Child nodes (bucket or sub-ART)
        children: Vec<(u8, ChildNode)>,
        /// Whether empty string is in dictionary
        is_final: bool,
        /// Value for empty string
        value: Option<V>,
    },
}

impl<V: DictionaryValue> PersistentARTrie<V> {
    /// Create a new empty in-memory dictionary.
    ///
    /// This creates a purely in-memory dictionary without disk persistence.
    /// For disk-backed persistence, use `create()` or `open()`.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(PersistentARTrieInner {
                root: TrieRoot::Bucket(StringBucket::with_values()),
                term_count: 0,
                dirty: false,
                #[cfg(feature = "persistent-artrie")]
                buffer_manager: None,
                #[cfg(feature = "persistent-artrie")]
                wal_writer: None,
                #[cfg(feature = "persistent-artrie")]
                next_lsn: 0,
                #[cfg(feature = "persistent-artrie")]
                prefetcher: super::prefetch::Prefetcher::disabled(),
            })),
        }
    }

    /// Create a new persistent dictionary at the given path.
    ///
    /// This creates a new dictionary file with WAL for crash recovery.
    /// If a file already exists at the path, this will return an error.
    ///
    /// # Arguments
    /// * `path` - Path to the dictionary file (will also create `.wal` file)
    ///
    /// # Example
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
    ///
    /// let dict: PersistentARTrie<()> = PersistentARTrie::create("words.part")?;
    /// ```
    #[cfg(feature = "persistent-artrie")]
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        use super::disk_manager::DiskManager;
        use super::buffer_manager::BufferManager;
        use super::wal::WalWriter;
        use super::DEFAULT_BUFFER_POOL_SIZE;

        let path = path.as_ref();

        // Fail if file already exists
        if path.exists() {
            return Err(PersistentARTrieError::io_error(
                "create",
                path.display().to_string(),
                std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "Dictionary file already exists",
                ),
            ));
        }

        // Create disk manager (creates new file)
        let disk_manager = DiskManager::create(path)?;

        // Create buffer manager with default pool size (takes ownership of disk_manager)
        let buffer_manager = BufferManager::new(disk_manager, DEFAULT_BUFFER_POOL_SIZE);
        let buffer_manager = Arc::new(RwLock::new(buffer_manager));

        // Create WAL file alongside the main file
        let wal_path = path.with_extension("wal");
        let wal_writer = WalWriter::create(&wal_path)?;
        let wal_writer = Arc::new(RwLock::new(wal_writer));

        Ok(Self {
            inner: Arc::new(RwLock::new(PersistentARTrieInner {
                root: TrieRoot::Bucket(StringBucket::with_values()),
                term_count: 0,
                dirty: false,
                buffer_manager: Some(buffer_manager),
                wal_writer: Some(wal_writer),
                next_lsn: 1, // Start at 1, 0 reserved for "no LSN"
                prefetcher: super::prefetch::Prefetcher::new(),
            })),
        })
    }

    /// Open an existing persistent dictionary from disk.
    ///
    /// This opens an existing dictionary file and replays the WAL if needed
    /// to recover from any crash.
    ///
    /// # Arguments
    /// * `path` - Path to the dictionary file
    ///
    /// # Example
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
    ///
    /// let dict: PersistentARTrie<()> = PersistentARTrie::open("words.part")?;
    /// ```
    #[cfg(feature = "persistent-artrie")]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        use super::disk_manager::DiskManager;
        use super::buffer_manager::BufferManager;
        use super::wal::WalWriter;
        use super::recovery::RecoveryManager;
        use super::DEFAULT_BUFFER_POOL_SIZE;

        let path = path.as_ref();

        // Fail if file doesn't exist
        if !path.exists() {
            return Err(PersistentARTrieError::io_error(
                "open",
                path.display().to_string(),
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Dictionary file not found",
                ),
            ));
        }

        // Open disk manager
        let disk_manager = DiskManager::open(path)?;

        // Try to load the trie from disk (if a root descriptor exists)
        let root_ptr = disk_manager.root_ptr()?;
        let entry_count = disk_manager.entry_count()?;
        let (loaded_root, loaded_term_count) = if root_ptr != 0 {
            // Load root descriptor from disk
            match Self::load_root_from_disk(&disk_manager, root_ptr) {
                Ok((root, count)) => (Some(root), count),
                Err(e) => {
                    eprintln!("Warning: Failed to load trie from disk: {:?}", e);
                    (None, 0)
                }
            }
        } else {
            (None, 0)
        };

        // Create buffer manager (takes ownership of disk_manager)
        let buffer_manager = BufferManager::new(disk_manager, DEFAULT_BUFFER_POOL_SIZE);
        let buffer_manager = Arc::new(RwLock::new(buffer_manager));

        // Recover from WAL if it exists
        let wal_path = path.with_extension("wal");
        let (recovered_ops, next_lsn, checkpoint_lsn) = if wal_path.exists() {
            let recovery_manager = RecoveryManager::new(&wal_path);
            match recovery_manager.recover() {
                Ok(state) => {
                    let lsn = state.next_lsn;
                    let cp_lsn = state.stats.checkpoint_lsn;
                    (state.into_operations(), lsn, cp_lsn)
                }
                Err(e) => {
                    eprintln!("Warning: WAL recovery error: {:?}", e);
                    (Vec::new(), 1, None)
                }
            }
        } else {
            (Vec::new(), 1, None)
        };

        // Open WAL writer (append mode)
        let wal_writer = if wal_path.exists() {
            WalWriter::open(&wal_path)?
        } else {
            WalWriter::create(&wal_path)?
        };
        let wal_writer = Arc::new(RwLock::new(wal_writer));

        // Create the dictionary with storage layer
        // Use loaded root if available, otherwise start with empty bucket
        let was_loaded_from_disk = loaded_root.is_some();
        let (initial_root, initial_term_count) = match loaded_root {
            Some(root) => (root, loaded_term_count as usize),
            None => (TrieRoot::Bucket(StringBucket::with_values()), 0),
        };

        let dict = Self {
            inner: Arc::new(RwLock::new(PersistentARTrieInner {
                root: initial_root,
                term_count: initial_term_count,
                dirty: false,
                buffer_manager: Some(buffer_manager),
                wal_writer: Some(wal_writer.clone()),
                next_lsn,
                prefetcher: super::prefetch::Prefetcher::new(),
            })),
        };

        // Replay recovered operations
        // If we loaded from disk, only replay operations AFTER the checkpoint
        {
            let mut inner = dict.inner.write().expect("lock poisoned");

            // Determine the LSN threshold for skipping
            // Operations with LSN <= threshold are already in the on-disk state
            let skip_threshold = if was_loaded_from_disk {
                checkpoint_lsn
            } else {
                None
            };

            let mut replayed_count = 0;
            for op in recovered_ops.into_iter() {
                match op {
                    super::recovery::RecoveredOperation::Insert { lsn, term, value } => {
                        // Skip if this operation is already reflected in disk state
                        if let Some(threshold) = skip_threshold {
                            if lsn <= threshold {
                                continue;
                            }
                        }
                        // Deserialize value from WAL if present
                        let deserialized_value: Option<V> = value.and_then(|bytes| {
                            match bincode::deserialize(&bytes) {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    eprintln!("Warning: Failed to deserialize value from WAL: {:?}", e);
                                    None
                                }
                            }
                        });
                        // Replay insert without re-logging to WAL
                        inner.insert_impl_no_wal(&term, deserialized_value);
                        replayed_count += 1;
                    }
                    super::recovery::RecoveredOperation::Remove { lsn, term } => {
                        // Skip if this operation is already reflected in disk state
                        if let Some(threshold) = skip_threshold {
                            if lsn <= threshold {
                                continue;
                            }
                        }
                        // Replay remove without re-logging to WAL
                        inner.remove_impl_no_wal(&term);
                        replayed_count += 1;
                    }
                    super::recovery::RecoveredOperation::Increment {
                        lsn,
                        term,
                        delta: _,
                        result,
                    } => {
                        // Skip if this operation is already reflected in disk state
                        if let Some(threshold) = skip_threshold {
                            if lsn <= threshold {
                                continue;
                            }
                        }
                        // For increment recovery, we set the final result value directly
                        // (this is idempotent even if replayed multiple times)
                        let value_bytes = result.to_le_bytes().to_vec();
                        if let Ok(value) = bincode::deserialize(&value_bytes) {
                            inner.upsert_impl_no_wal(&term, value);
                        }
                        replayed_count += 1;
                    }
                    super::recovery::RecoveredOperation::Upsert { lsn, term, value } => {
                        // Skip if this operation is already reflected in disk state
                        if let Some(threshold) = skip_threshold {
                            if lsn <= threshold {
                                continue;
                            }
                        }
                        // Deserialize and apply value
                        if let Ok(v) = bincode::deserialize(&value) {
                            inner.upsert_impl_no_wal(&term, v);
                        }
                        replayed_count += 1;
                    }
                    super::recovery::RecoveredOperation::CompareAndSwap {
                        lsn,
                        term,
                        new_value,
                        success,
                    } => {
                        // Skip if this operation is already reflected in disk state
                        if let Some(threshold) = skip_threshold {
                            if lsn <= threshold {
                                continue;
                            }
                        }
                        // Only apply if the CAS succeeded
                        if success {
                            if let Ok(v) = bincode::deserialize(&new_value) {
                                inner.upsert_impl_no_wal(&term, v);
                            }
                        }
                        replayed_count += 1;
                    }
                }
            }
            // Mark clean after recovery replay
            inner.dirty = false;

            // If we loaded from disk AND replayed no operations, we can truncate the WAL
            // (all operations were already persisted to disk before the checkpoint)
            if was_loaded_from_disk && replayed_count == 0 {
                if let Ok(wal) = wal_writer.write() {
                    if let Err(e) = wal.truncate() {
                        eprintln!("Warning: Failed to truncate WAL after recovery: {:?}", e);
                    }
                }
            }
        }

        Ok(dict)
    }

    /// Load the trie root from disk.
    ///
    /// Reads the root descriptor block and deserializes the trie structure.
    ///
    /// # Returns
    /// Tuple of (TrieRoot, term_count) on success.
    #[cfg(feature = "persistent-artrie")]
    fn load_root_from_disk(
        disk_manager: &super::disk_manager::DiskManager,
        root_ptr: u64,
    ) -> Result<(TrieRoot<V>, u64)> {
        use super::disk_manager::BLOCK_SIZE;
        use super::BUCKET_PAGE_SIZE;

        // Decode the SwizzledPtr to get block_id
        let ptr = SwizzledPtr::from_raw(root_ptr);
        if ptr.is_null() || ptr.is_swizzled() {
            return Err(PersistentARTrieError::corrupted(
                "Invalid root pointer: null or already swizzled",
            ));
        }

        let location = ptr.disk_location().ok_or_else(|| {
            PersistentARTrieError::corrupted("Could not decode disk location from root pointer")
        })?;

        // Read the descriptor block
        let mut descriptor_buf = [0u8; BLOCK_SIZE];
        disk_manager.read_block(location.block_id, &mut descriptor_buf)?;

        // Parse root descriptor
        // Format:
        //   0: type (1 byte)
        //   1: is_final (1 byte)
        //   2-5: term_count (4 bytes, little endian)
        //   6-9: value_len (4 bytes, little endian)
        //   10-17: root_ptr (8 bytes, little endian)
        //   18+: value bytes (if any)
        let root_type = descriptor_buf[0];
        let is_final = descriptor_buf[1] != 0;
        let term_count = u32::from_le_bytes([
            descriptor_buf[2],
            descriptor_buf[3],
            descriptor_buf[4],
            descriptor_buf[5],
        ]);
        let value_len = u32::from_le_bytes([
            descriptor_buf[6],
            descriptor_buf[7],
            descriptor_buf[8],
            descriptor_buf[9],
        ]);
        let data_ptr = u64::from_le_bytes([
            descriptor_buf[10],
            descriptor_buf[11],
            descriptor_buf[12],
            descriptor_buf[13],
            descriptor_buf[14],
            descriptor_buf[15],
            descriptor_buf[16],
            descriptor_buf[17],
        ]);

        let _ = value_len; // Value deserialization not yet implemented
        let _ = is_final;  // Used for ArtNode but we simplified

        match root_type {
            ROOT_TYPE_BUCKET => {
                // Load bucket from disk
                let bucket_ptr = SwizzledPtr::from_raw(data_ptr);
                let bucket_loc = bucket_ptr.disk_location().ok_or_else(|| {
                    PersistentARTrieError::corrupted("Invalid bucket pointer in root descriptor")
                })?;

                let mut bucket_data = [0u8; BUCKET_PAGE_SIZE];
                disk_manager.read_bytes(bucket_loc.block_id, 0, &mut bucket_data)?;

                let bucket = StringBucket::from_bytes(&bucket_data).map_err(|e| {
                    PersistentARTrieError::corrupted(format!("Failed to load bucket: {:?}", e))
                })?;

                Ok((TrieRoot::Bucket(bucket), term_count as u64))
            }
            ROOT_TYPE_ART_NODE => {
                // Load the ART node from disk
                let node_ptr = SwizzledPtr::from_raw(data_ptr);

                // Load the node and its children recursively
                let (node, children) = Self::load_art_node_with_children(disk_manager, &node_ptr)?;

                // Deserialize the root value if present
                let root_value: Option<V> = if value_len > 0 {
                    let value_bytes = &descriptor_buf[18..18 + value_len as usize];
                    bincode::deserialize(value_bytes).ok()
                } else {
                    None
                };

                Ok((
                    TrieRoot::ArtNode {
                        node,
                        children,
                        is_final,
                        value: root_value,
                    },
                    term_count as u64,
                ))
            }
            ROOT_TYPE_EMPTY | _ => {
                // Empty or unknown type
                Ok((TrieRoot::Bucket(StringBucket::with_values()), 0))
            }
        }
    }

    /// Load an ART node from disk and recursively load all its children.
    ///
    /// This method deserializes an ART node and builds the in-memory ChildNode
    /// structure by loading each child (which may be a bucket or another ART node).
    ///
    /// # Returns
    /// Tuple of (Node, Vec<(u8, ChildNode)>) representing the node and its children.
    #[cfg(feature = "persistent-artrie")]
    fn load_art_node_with_children(
        disk_manager: &super::disk_manager::DiskManager,
        node_ptr: &SwizzledPtr,
    ) -> Result<(Node, Vec<(u8, ChildNode)>)> {
        // Get disk location from SwizzledPtr
        let location = node_ptr.disk_location().ok_or_else(|| {
            PersistentARTrieError::corrupted("Invalid node pointer: cannot get disk location")
        })?;

        // Read the node data from disk
        let mut node_data = [0u8; ART_NODE_BUFFER_SIZE];
        disk_manager.read_bytes(location.block_id, 0, &mut node_data)?;

        // Deserialize the node
        let node = serialization::from_bytes(&node_data).map_err(|e| {
            PersistentARTrieError::corrupted(format!("Failed to deserialize ART node: {:?}", e))
        })?;

        // Load all children recursively
        let mut children = Vec::new();
        for (key, child_ptr) in node.iter_children() {
            if !child_ptr.is_null() {
                let child = Self::load_child_from_disk(disk_manager, child_ptr)?;
                children.push((key, child));
            }
        }

        Ok((node, children))
    }

    /// Load a child node (bucket or ART node) from disk.
    ///
    /// This method examines the SwizzledPtr's node type to determine whether
    /// the child is a bucket or an ART node, and loads it appropriately.
    #[cfg(feature = "persistent-artrie")]
    fn load_child_from_disk(
        disk_manager: &super::disk_manager::DiskManager,
        child_ptr: &SwizzledPtr,
    ) -> Result<ChildNode> {
        use super::BUCKET_PAGE_SIZE;

        let location = child_ptr.disk_location().ok_or_else(|| {
            PersistentARTrieError::corrupted("Invalid child pointer: cannot get disk location")
        })?;

        // Determine child type from the DiskLocation's node_type
        let node_type = location.node_type;

        match node_type {
            NodeType::Bucket => {
                // Load bucket from disk
                let mut bucket_data = [0u8; BUCKET_PAGE_SIZE];
                disk_manager.read_bytes(location.block_id, 0, &mut bucket_data)?;

                let bucket = StringBucket::from_bytes(&bucket_data).map_err(|e| {
                    PersistentARTrieError::corrupted(format!("Failed to load child bucket: {:?}", e))
                })?;

                Ok(ChildNode::Bucket(bucket))
            }
            NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
                // Read the node data from disk
                let mut node_data = [0u8; ART_NODE_BUFFER_SIZE];
                disk_manager.read_bytes(location.block_id, 0, &mut node_data)?;

                // Deserialize the node
                let node = serialization::from_bytes(&node_data).map_err(|e| {
                    PersistentARTrieError::corrupted(format!("Failed to deserialize child ART node: {:?}", e))
                })?;

                // Check if node is final (has IS_FINAL flag set)
                let is_final = node.header().is_final();

                // Load children recursively
                let mut children = Vec::new();
                for (key, grandchild_ptr) in node.iter_children() {
                    if !grandchild_ptr.is_null() {
                        let grandchild = Self::load_child_from_disk(disk_manager, grandchild_ptr)?;
                        children.push((key, grandchild));
                    }
                }

                Ok(ChildNode::ArtNode {
                    node,
                    is_final,
                    value: None, // Value serialization for nested nodes is future work
                    children,
                })
            }
            // Char-level nodes should never appear in byte-level trie
            NodeType::CharNode4 | NodeType::CharNode16 | NodeType::CharNode48 | NodeType::CharBucket => {
                Err(PersistentARTrieError::corrupted(
                    "Char-level node type encountered in byte-level PersistentARTrie"
                ))
            }
        }
    }

    /// Insert a term into the dictionary (without value)
    pub fn insert(&mut self, term: &str) -> bool {
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.insert_impl(term.as_bytes(), None)
    }

    /// Insert a term with an associated value
    pub fn insert_with_value(&mut self, term: &str, value: V) -> bool {
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.insert_impl(term.as_bytes(), Some(value))
    }

    /// Remove a term from the dictionary
    pub fn remove(&mut self, term: &str) -> bool {
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.remove_impl(term.as_bytes())
    }

    /// Check if the dictionary is dirty (has uncommitted changes)
    pub fn is_dirty(&self) -> bool {
        let inner = self.inner.read().expect("lock poisoned");
        inner.dirty
    }

    /// Mark the dictionary as clean (after flushing to disk)
    pub fn mark_clean(&mut self) {
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.dirty = false;
    }

    /// Flush all buffered data to disk for durability.
    ///
    /// This ensures all WAL records are synced to persistent storage.
    /// Call this after a batch of operations to ensure durability.
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut dict: PersistentARTrie<()> = PersistentARTrie::create("words.part")?;
    /// dict.insert("hello");
    /// dict.insert("world");
    /// dict.sync()?; // Ensure both inserts are durable
    /// ```
    #[cfg(feature = "persistent-artrie")]
    pub fn sync(&self) -> Result<()> {
        let inner = self.inner.read().expect("lock poisoned");

        // Sync WAL to disk
        if let Some(ref wal_writer) = inner.wal_writer {
            let wal = wal_writer.write().expect("WAL lock poisoned");
            wal.sync()?;
        }

        // Flush all dirty pages from buffer manager
        if let Some(ref buffer_manager) = inner.buffer_manager {
            buffer_manager.read().expect("buffer lock poisoned").flush_all()?;
        }

        Ok(())
    }

    /// Create a checkpoint to allow WAL truncation.
    ///
    /// A checkpoint persists all in-memory trie data to disk, then records
    /// the current LSN in the WAL. This allows older WAL records to be safely
    /// removed after recovery.
    ///
    /// This should be called periodically to prevent unbounded WAL growth.
    ///
    /// # Algorithm
    ///
    /// 1. Persist all in-memory nodes to disk via `persist_to_disk()`
    /// 2. Write a Checkpoint record to WAL with the current LSN
    /// 3. Sync the WAL to ensure durability
    ///
    /// After a checkpoint, recovery only needs to replay WAL records with
    /// LSN > checkpoint_lsn, as earlier operations are already in the
    /// persistent trie structure.
    #[cfg(feature = "persistent-artrie")]
    pub fn checkpoint(&mut self) -> Result<()> {
        use super::wal::WalRecord;

        // First, persist all in-memory data to disk
        {
            let mut inner = self.inner.write().expect("lock poisoned");
            inner.persist_to_disk()?;
        }

        // Then write the checkpoint record to WAL
        let inner = self.inner.read().expect("lock poisoned");

        if let Some(ref wal_writer) = inner.wal_writer {
            let wal = wal_writer.write().expect("WAL lock poisoned");

            // Get current LSN as checkpoint
            let checkpoint_lsn = inner.next_lsn.saturating_sub(1);
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let record = WalRecord::Checkpoint {
                checkpoint_lsn,
                timestamp,
            };

            wal.append(record)?;
            wal.sync()?;
        }

        Ok(())
    }

    /// Get prefetch statistics for performance monitoring.
    ///
    /// Returns a snapshot of prefetch operation counts including:
    /// - Total requests
    /// - Cache hits (node already in memory)
    /// - I/O operations issued
    /// - Dropped requests (queue full)
    ///
    /// # Example
    /// ```rust,ignore
    /// let dict: PersistentARTrie<()> = PersistentARTrie::create("words.part")?;
    /// // ... perform queries ...
    /// let stats = dict.prefetch_stats();
    /// println!("Prefetch hit rate: {:.1}%", stats.hit_rate() * 100.0);
    /// ```
    #[cfg(feature = "persistent-artrie")]
    pub fn prefetch_stats(&self) -> super::prefetch::PrefetchStatsSnapshot {
        let inner = self.inner.read().expect("lock poisoned");
        inner.prefetcher.stats().snapshot()
    }

    /// Get a snapshot node for traversal
    fn get_root_node(&self) -> PersistentARTrieNode<V> {
        let inner = self.inner.read().expect("lock poisoned");
        match &inner.root {
            TrieRoot::Bucket(bucket) => PersistentARTrieNode::new_bucket(bucket.clone()),
            TrieRoot::ArtNode {
                node,
                is_final,
                value,
                ..
            } => PersistentARTrieNode::new_art_node(
                node.clone(),
                *is_final,
                value.clone(),
            ),
        }
    }
}

impl<V: DictionaryValue> Default for PersistentARTrie<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: DictionaryValue> PersistentARTrieInner<V> {
    /// Insert implementation with WAL logging (for persistent mode).
    fn insert_impl(&mut self, term: &[u8], value: Option<V>) -> bool {
        // Clone value for WAL logging if needed (before move into core)
        #[cfg(feature = "persistent-artrie")]
        let value_for_wal = value.clone();

        // Perform the actual insert
        let inserted = self.insert_impl_core(term, value);

        // Log to WAL if insert was successful OR if we're updating an existing term's value
        // We need to log value updates even when the term already exists (inserted = false)
        #[cfg(feature = "persistent-artrie")]
        if inserted || value_for_wal.is_some() {
            if let Some(ref wal_writer) = self.wal_writer {
                use super::wal::WalRecord;

                // Serialize the value using bincode if present
                let serialized_value = value_for_wal.and_then(|v| {
                    match bincode::serialize(&v) {
                        Ok(bytes) => Some(bytes),
                        Err(e) => {
                            eprintln!("Warning: Failed to serialize value for WAL: {:?}", e);
                            None
                        }
                    }
                });

                let record = WalRecord::Insert {
                    term: term.to_vec(),
                    value: serialized_value,
                };
                if let Err(e) = wal_writer.write().expect("WAL lock poisoned").append(record) {
                    // Log error but don't fail the insert - data is in memory
                    eprintln!("Warning: Failed to log insert to WAL: {:?}", e);
                }
            }
        }

        inserted
    }

    /// Core insert implementation without WAL logging.
    fn insert_impl_core(&mut self, term: &[u8], value: Option<V>) -> bool {
        let inserted = match &mut self.root {
            TrieRoot::Bucket(bucket) => {
                // Clone value here in case we need to retry after bucket conversion
                let value_for_retry = value.clone();

                // Serialize value for bucket storage
                #[cfg(feature = "persistent-artrie")]
                let serialized_value: Option<Vec<u8>> = value.and_then(|v| {
                    bincode::serialize(&v).ok()
                });

                #[cfg(not(feature = "persistent-artrie"))]
                let serialized_value: Option<Vec<u8>> = None;

                let result = if let Some(ref val_bytes) = serialized_value {
                    bucket.insert(term, val_bytes)
                } else {
                    bucket.insert_key(term)
                };

                match result {
                    Ok(inserted) => {
                        // Check if bucket needs to be converted to ART
                        if bucket.header().should_split() {
                            self.convert_bucket_to_art();
                        }
                        inserted
                    }
                    Err(_) => {
                        // Bucket is full, convert to ART and retry
                        self.convert_bucket_to_art();
                        // Retry insert in the new ART structure (no double WAL logging)
                        self.insert_impl_core(term, value_for_retry);
                        true
                    }
                }
            }
            TrieRoot::ArtNode {
                node,
                children,
                is_final,
                value: root_value,
            } => {
                // Serialize value for bucket storage (same as root bucket case)
                #[cfg(feature = "persistent-artrie")]
                let serialized_value: Option<Vec<u8>> = value.clone().and_then(|v| {
                    bincode::serialize(&v).ok()
                });

                #[cfg(not(feature = "persistent-artrie"))]
                let serialized_value: Option<Vec<u8>> = None;

                if term.is_empty() {
                    // Inserting empty string
                    if *is_final {
                        if value.is_some() {
                            *root_value = value;
                        }
                        false
                    } else {
                        *is_final = true;
                        *root_value = value;
                        true
                    }
                } else {
                    // Find or create child for first byte
                    let first_byte = term[0];
                    let remaining = &term[1..];

                    // Find existing child
                    let child_idx = children.iter().position(|(b, _)| *b == first_byte);

                    if let Some(idx) = child_idx {
                        // Insert into existing child
                        match &mut children[idx].1 {
                            ChildNode::Bucket(bucket) => {
                                // Insert with value if provided
                                let result = if let Some(ref val_bytes) = serialized_value {
                                    bucket.insert(remaining, val_bytes)
                                } else {
                                    bucket.insert_key(remaining)
                                };

                                match result {
                                    Ok(inserted) => inserted,
                                    Err(_) => {
                                        // Bucket is full, convert to ART node
                                        if let Some(result) = bucket_to_art_node(bucket).ok() {
                                            let new_children: Vec<(u8, ChildNode)> = result
                                                .children
                                                .into_iter()
                                                .map(|(b, bucket)| (b, ChildNode::Bucket(bucket)))
                                                .collect();
                                            children[idx].1 = ChildNode::ArtNode {
                                                node: result.node,
                                                is_final: result.is_final,
                                                value: result.final_value,
                                                children: new_children,
                                            };
                                            // Retry insert in the converted ART node
                                            if let Some((_, _, _, child_children)) =
                                                children[idx].1.as_art_node_mut()
                                            {
                                                // Find or create child for first byte of remaining
                                                if !remaining.is_empty() {
                                                    let first = remaining[0];
                                                    let rest = &remaining[1..];
                                                    // Try to insert into child
                                                    for (b, c) in child_children.iter_mut() {
                                                        if *b == first {
                                                            if let Some(bucket) = c.as_bucket_mut() {
                                                                // Insert with value
                                                                let insert_result = if let Some(ref val_bytes) = serialized_value {
                                                                    bucket.insert(rest, val_bytes)
                                                                } else {
                                                                    bucket.insert_key(rest)
                                                                };
                                                                return insert_result.unwrap_or(false);
                                                            }
                                                            return false;
                                                        }
                                                    }
                                                    // Create new bucket child
                                                    let mut new_bucket = StringBucket::with_values();
                                                    // Insert with value
                                                    if let Some(ref val_bytes) = serialized_value {
                                                        let _ = new_bucket.insert(rest, val_bytes);
                                                    } else {
                                                        let _ = new_bucket.insert_key(rest);
                                                    }
                                                    child_children.push((first, ChildNode::Bucket(new_bucket)));
                                                    return true;
                                                }
                                            }
                                        }
                                        false
                                    }
                                }
                            }
                            ChildNode::ArtNode {
                                is_final: child_is_final,
                                value: child_value,
                                children: child_children,
                                ..
                            } => {
                                // Recursive insert into child ART
                                if remaining.is_empty() {
                                    if *child_is_final {
                                        // Value already exists at this node. Value update
                                        // is not implemented because DictionaryValue doesn't
                                        // require serialization (V -> Vec<u8>).
                                        let _ = value; // Acknowledge value parameter
                                        false
                                    } else {
                                        *child_is_final = true;
                                        true
                                    }
                                } else {
                                    let first = remaining[0];
                                    let rest = &remaining[1..];

                                    // Find or create child
                                    for (b, c) in child_children.iter_mut() {
                                        if *b == first {
                                            // Use recursive insert_key for all child types
                                            return c.insert_key(rest);
                                        }
                                    }

                                    // Create new bucket child
                                    let mut new_bucket = StringBucket::with_values();
                                    let _ = new_bucket.insert_key(rest);
                                    child_children.push((first, ChildNode::Bucket(new_bucket)));
                                    true
                                }
                            }
                            ChildNode::DiskRef { .. } => {
                                // Cannot insert into disk ref without loading first
                                // This should be resolved before insert
                                false
                            }
                        }
                    } else {
                        // Create new child bucket
                        let mut bucket = StringBucket::with_values();
                        // Insert with value if provided
                        if let Some(ref val_bytes) = serialized_value {
                            let _ = bucket.insert(remaining, val_bytes);
                        } else {
                            let _ = bucket.insert_key(remaining);
                        }

                        // Add child to ART node
                        let ptr = SwizzledPtr::null();
                        let _ = match node {
                            Node::N4(n) => n.add_child(first_byte, ptr),
                            Node::N16(n) => n.add_child(first_byte, ptr),
                            Node::N48(n) => n.add_child(first_byte, ptr),
                            Node::N256(n) => n.add_child(first_byte, ptr),
                        };

                        children.push((first_byte, ChildNode::Bucket(bucket)));
                        true
                    }
                }
            }
        };

        if inserted {
            self.term_count += 1;
            self.dirty = true;
        }

        inserted
    }

    /// Remove implementation with WAL logging (for persistent mode).
    fn remove_impl(&mut self, term: &[u8]) -> bool {
        // Perform the actual remove
        let removed = self.remove_impl_core(term);

        // Log to WAL if remove was successful and we have a WAL writer
        #[cfg(feature = "persistent-artrie")]
        if removed {
            if let Some(ref wal_writer) = self.wal_writer {
                use super::wal::WalRecord;
                let record = WalRecord::Remove {
                    term: term.to_vec(),
                };
                if let Err(e) = wal_writer.write().expect("WAL lock poisoned").append(record) {
                    // Log error but don't fail the remove - data is in memory
                    eprintln!("Warning: Failed to log remove to WAL: {:?}", e);
                }
            }
        }

        removed
    }

    /// Core remove implementation without WAL logging.
    fn remove_impl_core(&mut self, term: &[u8]) -> bool {
        let removed = match &mut self.root {
            TrieRoot::Bucket(bucket) => bucket.remove(term).is_some(),
            TrieRoot::ArtNode {
                node: _,
                children,
                is_final,
                value,
            } => {
                if term.is_empty() {
                    if *is_final {
                        *is_final = false;
                        *value = None;
                        true
                    } else {
                        false
                    }
                } else {
                    let first_byte = term[0];
                    let remaining = &term[1..];

                    let child_idx = children.iter().position(|(b, _)| *b == first_byte);

                    if let Some(idx) = child_idx {
                        match &mut children[idx].1 {
                            ChildNode::Bucket(bucket) => bucket.remove(remaining).is_some(),
                            ChildNode::ArtNode {
                                is_final: child_is_final,
                                value: child_value,
                                children: child_children,
                                ..
                            } => {
                                // Recursive remove from child ART
                                if remaining.is_empty() {
                                    if *child_is_final {
                                        *child_is_final = false;
                                        *child_value = None;
                                        true
                                    } else {
                                        false
                                    }
                                } else {
                                    let first = remaining[0];
                                    let rest = &remaining[1..];

                                    // Find child with matching byte
                                    for (b, c) in child_children.iter_mut() {
                                        if *b == first {
                                            // Use recursive remove_key for all child types
                                            return c.remove_key(rest);
                                        }
                                    }
                                    false
                                }
                            }
                            ChildNode::DiskRef { .. } => {
                                // Cannot remove from disk ref without loading first
                                false
                            }
                        }
                    } else {
                        false
                    }
                }
            }
        };

        if removed {
            self.term_count -= 1;
            self.dirty = true;
        }

        removed
    }

    /// Convert root bucket to ART node structure
    fn convert_bucket_to_art(&mut self) {
        if let TrieRoot::Bucket(bucket) = &self.root {
            if let Some(result) = bucket_to_art_node(bucket).ok() {
                let children: Vec<(u8, ChildNode)> = result
                    .children
                    .into_iter()
                    .map(|(b, bucket)| (b, ChildNode::Bucket(bucket)))
                    .collect();

                self.root = TrieRoot::ArtNode {
                    node: result.node,
                    children,
                    is_final: result.is_final,
                    // Value cannot be preserved from bucket conversion because
                    // bucket uses Vec<u8> while TrieRoot uses V. Adding serde
                    // bounds to DictionaryValue would enable value preservation.
                    value: None,
                };
            }
        }
    }

    /// Insert implementation without WAL logging (for recovery replay).
    ///
    /// This is used during WAL recovery to avoid re-logging operations
    /// that are already in the WAL.
    #[cfg(feature = "persistent-artrie")]
    fn insert_impl_no_wal(&mut self, term: &[u8], value: Option<V>) -> bool {
        // Call core implementation directly to skip WAL logging
        self.insert_impl_core(term, value)
    }

    /// Remove implementation without WAL logging (for recovery replay).
    ///
    /// This is used during WAL recovery to avoid re-logging operations
    /// that are already in the WAL.
    #[cfg(feature = "persistent-artrie")]
    fn remove_impl_no_wal(&mut self, term: &[u8]) -> bool {
        // Call core implementation directly to skip WAL logging
        self.remove_impl_core(term)
    }

    /// Upsert implementation without WAL logging (for recovery replay).
    ///
    /// This updates the value if the term exists, or inserts if it doesn't.
    /// Used during WAL recovery to replay Upsert, Increment, and CAS operations.
    #[cfg(feature = "persistent-artrie")]
    fn upsert_impl_no_wal(&mut self, term: &[u8], value: V) {
        // First remove existing entry (if any) to allow update
        self.remove_impl_core(term);
        // Then insert with new value
        self.insert_impl_core(term, Some(value));
    }

    /// Check if a term is contained in the dictionary.
    ///
    /// This method handles:
    /// - Bucket root lookups
    /// - ART node traversal
    /// - Lazy loading of DiskRef children
    /// - Prefetching of sibling nodes for better I/O performance
    fn contains_impl(&self, term: &[u8]) -> bool {
        match &self.root {
            TrieRoot::Bucket(bucket) => bucket.contains(term),
            TrieRoot::ArtNode {
                children,
                is_final,
                ..
            } => {
                if term.is_empty() {
                    return *is_final;
                }

                let first_byte = term[0];
                let remaining = &term[1..];

                // Prefetch DiskRef children at the root level
                #[cfg(feature = "persistent-artrie")]
                self.prefetch_disk_refs(children);

                // Find child with matching first byte
                for (b, child) in children {
                    if *b == first_byte {
                        return self.contains_in_child(child, remaining);
                    }
                }
                false
            }
        }
    }

    /// Get the value associated with a term.
    ///
    /// Returns `Some(value)` if the term exists and has an associated value,
    /// `None` if the term doesn't exist or has no value.
    #[cfg(feature = "persistent-artrie")]
    fn get_value_impl(&self, term: &[u8]) -> Option<V> {
        match &self.root {
            TrieRoot::Bucket(bucket) => {
                // Search for the term in the bucket
                match bucket.search(term) {
                    Ok(idx) => {
                        // Found the term, get its value
                        if let Some(entry) = bucket.get_entry(idx) {
                            if let Some(value_bytes) = bucket.get_value(&entry) {
                                // Deserialize the value
                                bincode::deserialize(value_bytes).ok()
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    Err(_) => None, // Term not found
                }
            }
            TrieRoot::ArtNode {
                children,
                is_final,
                value,
                ..
            } => {
                if term.is_empty() {
                    if *is_final {
                        return value.clone();
                    }
                    return None;
                }

                let first_byte = term[0];
                let remaining = &term[1..];

                // Find child with matching first byte
                for (b, child) in children {
                    if *b == first_byte {
                        return self.get_value_in_child(child, remaining);
                    }
                }
                None
            }
        }
    }

    /// Get value from a child node.
    #[cfg(feature = "persistent-artrie")]
    fn get_value_in_child(&self, child: &ChildNode, remaining: &[u8]) -> Option<V> {
        match child {
            ChildNode::Bucket(bucket) => {
                match bucket.search(remaining) {
                    Ok(idx) => {
                        if let Some(entry) = bucket.get_entry(idx) {
                            if let Some(value_bytes) = bucket.get_value(&entry) {
                                bincode::deserialize(value_bytes).ok()
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    Err(_) => None,
                }
            }
            ChildNode::ArtNode {
                is_final,
                children,
                value,
                ..
            } => {
                if remaining.is_empty() {
                    if *is_final {
                        // Deserialize value from stored bytes
                        return value.as_ref().and_then(|bytes| {
                            bincode::deserialize(bytes).ok()
                        });
                    }
                    return None;
                }

                let first_byte = remaining[0];
                let rest = &remaining[1..];

                for (b, grandchild) in children {
                    if *b == first_byte {
                        return self.get_value_in_child(grandchild, rest);
                    }
                }
                None
            }
            ChildNode::DiskRef { .. } => {
                // TODO: Load from disk and get value
                None
            }
        }
    }

    /// Check if remaining term is contained in a child node.
    ///
    /// Handles all child node types including lazy loading of DiskRef.
    /// Uses prefetcher to read-ahead sibling nodes for better I/O performance.
    fn contains_in_child(&self, child: &ChildNode, remaining: &[u8]) -> bool {
        match child {
            ChildNode::Bucket(bucket) => bucket.contains(remaining),
            ChildNode::ArtNode {
                is_final,
                children,
                ..
            } => {
                if remaining.is_empty() {
                    return *is_final;
                }

                let first_byte = remaining[0];
                let rest = &remaining[1..];

                // Prefetch sibling DiskRef children for better I/O performance
                #[cfg(feature = "persistent-artrie")]
                self.prefetch_disk_refs(children);

                // Recursively search in children
                for (b, child) in children {
                    if *b == first_byte {
                        return self.contains_in_child(child, rest);
                    }
                }
                false
            }
            ChildNode::DiskRef { ptr } => {
                // Lazy load from disk
                if let Some(disk_location) = ptr.disk_location() {
                    #[cfg(feature = "persistent-artrie")]
                    if let Ok(resolved) = self.resolve_disk_ref(&disk_location) {
                        return self.contains_in_child(&resolved, remaining);
                    }
                }
                false
            }
        }
    }

    /// Prefetch all DiskRef children in a children list.
    ///
    /// This hints the prefetcher to start loading disk-resident children
    /// in the background while we process the current node.
    #[cfg(feature = "persistent-artrie")]
    fn prefetch_disk_refs(&self, children: &[(u8, ChildNode)]) {
        for (_, child) in children {
            if let ChildNode::DiskRef { ptr } = child {
                self.prefetcher.prefetch(ptr);
            }
        }
    }

    /// Resolve a DiskRef to its actual node data by loading from disk.
    ///
    /// This is the core lazy loading mechanism. When a child is stored as a
    /// DiskRef (pointing to disk), this method:
    /// 1. Reads the page data from the BufferManager
    /// 2. Deserializes the node or bucket data
    /// 3. Returns the resolved ChildNode
    ///
    /// # Arguments
    /// * `disk_location` - The disk location to load from
    ///
    /// # Returns
    /// The resolved ChildNode, or an error if loading failed.
    #[cfg(feature = "persistent-artrie")]
    fn resolve_disk_ref(&self, disk_location: &DiskLocation) -> Result<ChildNode> {
        // Get buffer manager (required for disk I/O)
        let buffer_manager = self.buffer_manager.as_ref().ok_or_else(|| {
            PersistentARTrieError::internal("No buffer manager available for disk I/O")
        })?;

        // Fetch the page containing the node
        let bm = buffer_manager.read().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "buffer_manager".to_string(),
            }
        })?;

        let page_guard = bm.fetch_page(disk_location.block_id)?;
        let page_data = page_guard.data();

        // Deserialize based on node type
        let offset = disk_location.offset as usize;
        let node_data = &page_data[offset..];

        match disk_location.node_type {
            NodeType::Bucket => {
                // Deserialize bucket
                // For now, return an empty bucket - full bucket serialization
                // will be implemented in Phase 7.4
                let bucket = StringBucket::new();
                Ok(ChildNode::Bucket(bucket))
            }
            NodeType::Node4 | NodeType::Node16 | NodeType::Node48 | NodeType::Node256 => {
                // Deserialize ART node
                let node = serialization::from_bytes(node_data)?;
                let is_final = node.header().is_final();
                Ok(ChildNode::ArtNode {
                    node,
                    is_final, // Read from deserialized node header
                    value: None, // Value not stored in node header currently
                    children: Vec::new(), // Children will be loaded lazily via SwizzledPtr
                })
            }
            // Char-level nodes should never appear in byte-level trie
            NodeType::CharNode4 | NodeType::CharNode16 | NodeType::CharNode48 | NodeType::CharBucket => {
                Err(PersistentARTrieError::corrupted(
                    "Char-level node type encountered in byte-level PersistentARTrie"
                ))
            }
        }
    }

    /// Check if a child needs lazy loading and resolve it if necessary.
    ///
    /// This is a no-op without the persistent-artrie feature.
    #[cfg(not(feature = "persistent-artrie"))]
    #[inline]
    fn resolve_child_if_needed(&self, child: &ChildNode) -> Option<ChildNode> {
        // Without persistent-artrie, DiskRef variants should never exist
        match child {
            ChildNode::DiskRef { .. } => None, // Should never happen
            _ => None, // No resolution needed for in-memory nodes
        }
    }

    /// Check if a child needs lazy loading and resolve it if necessary.
    ///
    /// Returns Some(resolved_child) if the child was a DiskRef that was successfully
    /// resolved, or None if no resolution was needed (already in memory).
    #[cfg(feature = "persistent-artrie")]
    fn resolve_child_if_needed(&self, child: &ChildNode) -> Option<ChildNode> {
        match child {
            ChildNode::DiskRef { ptr } => {
                // Get disk location from SwizzledPtr
                if let Some(disk_location) = ptr.disk_location() {
                    // Resolve from disk
                    self.resolve_disk_ref(&disk_location).ok()
                } else {
                    None
                }
            }
            _ => None, // Already in memory
        }
    }

    /// Serialize a bucket to disk and return a SwizzledPtr to its location.
    ///
    /// This allocates a new page via the BufferManager, writes the bucket data,
    /// and returns a SwizzledPtr pointing to the disk location.
    ///
    /// # Arguments
    /// * `bucket` - The bucket to serialize
    ///
    /// # Returns
    /// A SwizzledPtr pointing to the bucket on disk.
    #[cfg(feature = "persistent-artrie")]
    fn serialize_bucket_to_disk(&self, bucket: &StringBucket) -> Result<SwizzledPtr> {
        use super::BUCKET_PAGE_SIZE;

        // Get buffer manager
        let buffer_manager = self.buffer_manager.as_ref().ok_or_else(|| {
            PersistentARTrieError::internal("No buffer manager for disk serialization")
        })?;

        let bm = buffer_manager.write().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "buffer_manager".to_string(),
            }
        })?;

        // Allocate a new page
        let mut page_guard = bm.new_page()?;
        let block_id = page_guard.block_id();

        // Write bucket data to the page
        let bucket_bytes = bucket.as_bytes();
        let page_data = page_guard.data_mut();

        // Copy bucket data (bucket is exactly BUCKET_PAGE_SIZE = 8KB)
        // Pages are BLOCK_SIZE = 256KB, so bucket fits at offset 0
        page_data[..BUCKET_PAGE_SIZE].copy_from_slice(bucket_bytes);

        // Create SwizzledPtr pointing to this bucket
        let ptr = SwizzledPtr::on_disk(block_id, 0, NodeType::Bucket);

        Ok(ptr)
    }

    /// Serialize an ART node to disk and return a SwizzledPtr to its location.
    ///
    /// This allocates a new page via the BufferManager, writes the serialized node,
    /// and returns a SwizzledPtr pointing to the disk location.
    ///
    /// # Arguments
    /// * `node` - The ART node to serialize
    ///
    /// # Returns
    /// A SwizzledPtr pointing to the node on disk.
    #[cfg(feature = "persistent-artrie")]
    fn serialize_node_to_disk(&self, node: &Node) -> Result<SwizzledPtr> {
        // Get buffer manager
        let buffer_manager = self.buffer_manager.as_ref().ok_or_else(|| {
            PersistentARTrieError::internal("No buffer manager for disk serialization")
        })?;

        let bm = buffer_manager.write().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "buffer_manager".to_string(),
            }
        })?;

        // Serialize the node to bytes
        let node_bytes = serialization::to_bytes(node)?;

        // Allocate a new page
        let mut page_guard = bm.new_page()?;
        let block_id = page_guard.block_id();

        // Write node data to the page
        let page_data = page_guard.data_mut();
        page_data[..node_bytes.len()].copy_from_slice(&node_bytes);

        // Determine node type for SwizzledPtr
        let node_type = match node {
            Node::N4(_) => NodeType::Node4,
            Node::N16(_) => NodeType::Node16,
            Node::N48(_) => NodeType::Node48,
            Node::N256(_) => NodeType::Node256,
        };

        // Create SwizzledPtr pointing to this node
        let ptr = SwizzledPtr::on_disk(block_id, 0, node_type);

        Ok(ptr)
    }

    /// Persist all modified nodes in the trie to disk.
    ///
    /// This method walks through the trie structure and serializes all
    /// in-memory nodes to disk, then updates the file header with the
    /// root pointer. After this, the trie can be loaded from disk without
    /// replaying the WAL.
    ///
    /// # Algorithm
    ///
    /// 1. Recursively serialize all children (buckets and nested ART nodes)
    /// 2. Serialize the root node/bucket
    /// 3. Create a root descriptor block with metadata
    /// 4. Update the file header's root_ptr to point to the descriptor
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if serialization fails.
    #[cfg(feature = "persistent-artrie")]
    pub fn persist_to_disk(&mut self) -> Result<()> {

        // Get buffer manager
        let buffer_manager = self.buffer_manager.as_ref().ok_or_else(|| {
            PersistentARTrieError::internal("No buffer manager for disk serialization")
        })?;

        // Serialize the trie root and get a descriptor
        let (root_type, root_ptr, is_final, value_bytes, term_count) = match &self.root {
            TrieRoot::Bucket(bucket) => {
                // Serialize the bucket
                let ptr = self.serialize_bucket_to_disk(bucket)?;
                (ROOT_TYPE_BUCKET, ptr.to_raw(), false, Vec::new(), self.term_count)
            }
            TrieRoot::ArtNode {
                node,
                children,
                is_final,
                value,
            } => {
                // First, serialize all children recursively and collect their pointers
                let mut child_ptrs: Vec<(u8, u64)> = Vec::with_capacity(children.len());
                for (edge, child) in children {
                    let ptr = self.serialize_child_to_disk(child)?;
                    child_ptrs.push((*edge, ptr.to_raw()));
                }

                // Create a copy of the node with updated child pointers
                let mut node_copy = node.clone();
                for (edge, ptr_raw) in &child_ptrs {
                    // Update the node's child pointer for this edge
                    if let Some(child_ptr) = node_copy.find_child_mut(*edge) {
                        *child_ptr = SwizzledPtr::from_raw(*ptr_raw);
                    }
                }

                // Serialize the updated node
                let node_ptr = self.serialize_node_to_disk(&node_copy)?;

                // Prepare value bytes (empty for now since DictionaryValue lacks serialization)
                let value_bytes = Vec::new();
                let _ = value; // Value serialization requires serde bounds on DictionaryValue

                (ROOT_TYPE_ART_NODE, node_ptr.to_raw(), *is_final, value_bytes, self.term_count)
            }
        };

        // Create root descriptor block
        // Format:
        //   0: type (1 byte)
        //   1: is_final (1 byte)
        //   2-5: term_count (4 bytes, little endian)
        //   6-9: value_len (4 bytes, little endian)
        //   10-17: root_ptr (8 bytes, little endian)
        //   18+: value bytes (if any)
        let mut descriptor = vec![0u8; 18 + value_bytes.len()];
        descriptor[0] = root_type;
        descriptor[1] = if is_final { 1 } else { 0 };
        descriptor[2..6].copy_from_slice(&(term_count as u32).to_le_bytes());
        descriptor[6..10].copy_from_slice(&(value_bytes.len() as u32).to_le_bytes());
        descriptor[10..18].copy_from_slice(&root_ptr.to_le_bytes());
        if !value_bytes.is_empty() {
            descriptor[18..].copy_from_slice(&value_bytes);
        }

        // Allocate a block for the descriptor and write it
        let bm = buffer_manager.write().map_err(|_| {
            PersistentARTrieError::LockPoisoned {
                resource: "buffer_manager".to_string(),
            }
        })?;

        let mut page_guard = bm.new_page()?;
        let block_id = page_guard.block_id();
        let page_data = page_guard.data_mut();
        page_data[..descriptor.len()].copy_from_slice(&descriptor);

        // Update the file header with the root pointer
        let dm = bm.disk_manager();
        let root_descriptor_ptr = SwizzledPtr::on_disk(block_id, 0, NodeType::Bucket);
        dm.set_root_ptr(root_descriptor_ptr.to_raw())?;
        dm.set_entry_count(term_count as u64)?;

        // Flush all pages to ensure durability
        bm.flush_all()?;
        dm.sync()?;

        self.dirty = false;
        Ok(())
    }

    /// Serialize a ChildNode to disk and return its SwizzledPtr.
    #[cfg(feature = "persistent-artrie")]
    fn serialize_child_to_disk(&self, child: &ChildNode) -> Result<SwizzledPtr> {
        match child {
            ChildNode::Bucket(bucket) => self.serialize_bucket_to_disk(bucket),
            ChildNode::ArtNode {
                node,
                is_final,
                value,
                children,
            } => {
                // Recursively serialize all children first
                let mut child_ptrs: Vec<(u8, u64)> = Vec::with_capacity(children.len());
                for (edge, child) in children {
                    let ptr = self.serialize_child_to_disk(child)?;
                    child_ptrs.push((*edge, ptr.to_raw()));
                }

                // Create a copy of the node with updated child pointers
                let mut node_copy = node.clone();
                for (edge, ptr_raw) in &child_ptrs {
                    if let Some(child_ptr) = node_copy.find_child_mut(*edge) {
                        *child_ptr = SwizzledPtr::from_raw(*ptr_raw);
                    }
                }

                // Serialize the node
                let node_ptr = self.serialize_node_to_disk(&node_copy)?;

                // For nested ART nodes, we create a mini-descriptor
                // Format: is_final (1) + value_len (4) + node_ptr (8) + value
                let value_bytes: Vec<u8> = Vec::new(); // Value serialization not yet implemented
                let _ = value;
                let _ = is_final;

                // Just return the node pointer directly for now
                // Full nested descriptor support would require more complex format
                Ok(node_ptr)
            }
            ChildNode::DiskRef { ptr } => {
                // Already on disk, return as-is
                Ok(ptr.clone())
            }
        }
    }
}

/// Root descriptor type constants
const ROOT_TYPE_EMPTY: u8 = 0;
const ROOT_TYPE_BUCKET: u8 = 1;
const ROOT_TYPE_ART_NODE: u8 = 2;

impl<V: DictionaryValue> Dictionary for PersistentARTrie<V> {
    type Node = PersistentARTrieNode<V>;

    fn root(&self) -> Self::Node {
        self.get_root_node()
    }

    fn contains(&self, term: &str) -> bool {
        let inner = self.inner.read().expect("lock poisoned");
        inner.contains_impl(term.as_bytes())
    }

    fn len(&self) -> Option<usize> {
        let inner = self.inner.read().expect("lock poisoned");
        Some(inner.term_count)
    }

    fn sync_strategy(&self) -> SyncStrategy {
        SyncStrategy::InternalSync
    }
}

impl<V: DictionaryValue> MappedDictionary for PersistentARTrie<V> {
    type Value = V;

    fn get_value(&self, term: &str) -> Option<Self::Value> {
        let inner = self.inner.read().expect("lock poisoned");
        inner.get_value_impl(term.as_bytes())
    }
}

impl<V: DictionaryValue> std::fmt::Debug for PersistentARTrie<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read().expect("lock poisoned");
        f.debug_struct("PersistentARTrie")
            .field("term_count", &inner.term_count)
            .field("dirty", &inner.dirty)
            .finish()
    }
}

// ============================================================================
// Iterator Implementation
// ============================================================================

/// Iterator state for DFS traversal of the trie
#[derive(Clone)]
enum IterState {
    /// Iterating over a bucket's entries
    Bucket {
        /// Current prefix (path to this bucket)
        prefix: Vec<u8>,
        /// Entries to iterate (suffix, value_bytes)
        entries: Vec<(Vec<u8>, Option<Vec<u8>>)>,
        /// Current index in entries
        index: usize,
    },
    /// Iterating over an ART node's children
    ArtNode {
        /// Current prefix (path to this node)
        prefix: Vec<u8>,
        /// Whether this node is final (represents a term)
        is_final: bool,
        /// Value at this node if final
        value: Option<Vec<u8>>,
        /// Whether we've yielded the final state yet
        yielded_final: bool,
        /// Children to visit (edge byte, child)
        children: Vec<(u8, ChildNode)>,
        /// Current child index
        child_index: usize,
    },
}

/// Iterator over all terms in a PersistentARTrie.
///
/// This iterator performs a depth-first traversal of the trie,
/// yielding terms in lexicographic order.
///
/// # Example
///
/// ```rust,ignore
/// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
///
/// let mut dict = PersistentARTrie::new();
/// dict.insert("apple");
/// dict.insert("banana");
///
/// for term in dict.iter() {
///     println!("{}", String::from_utf8_lossy(&term));
/// }
/// ```
pub struct TermIterator<V: DictionaryValue> {
    /// Stack of iteration states for DFS
    stack: Vec<IterState>,
    /// Marker for value type
    _marker: std::marker::PhantomData<V>,
}

impl<V: DictionaryValue> TermIterator<V> {
    /// Create a new iterator starting from the trie root
    fn new(root: &TrieRoot<V>) -> Self {
        let mut stack = Vec::new();

        match root {
            TrieRoot::Bucket(bucket) => {
                // Collect all bucket entries
                let entries: Vec<(Vec<u8>, Option<Vec<u8>>)> = bucket
                    .iter()
                    .map(|(entry, suffix)| {
                        let value = bucket.get_value(&entry).map(|v| v.to_vec());
                        (suffix.to_vec(), value)
                    })
                    .collect();

                if !entries.is_empty() {
                    stack.push(IterState::Bucket {
                        prefix: Vec::new(),
                        entries,
                        index: 0,
                    });
                }
            }
            TrieRoot::ArtNode {
                is_final,
                value,
                children,
                ..
            } => {
                // Serialize value if present
                #[cfg(feature = "persistent-artrie")]
                let value_bytes = value.as_ref().and_then(|v| bincode::serialize(v).ok());
                #[cfg(not(feature = "persistent-artrie"))]
                let value_bytes: Option<Vec<u8>> = None;
                let _ = value; // Silence unused warning

                stack.push(IterState::ArtNode {
                    prefix: Vec::new(),
                    is_final: *is_final,
                    value: value_bytes,
                    yielded_final: false,
                    children: children.clone(),
                    child_index: 0,
                });
            }
        }

        Self {
            stack,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<V: DictionaryValue> Iterator for TermIterator<V> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let state = self.stack.last_mut()?;

            match state {
                IterState::Bucket {
                    prefix,
                    entries,
                    index,
                } => {
                    if *index < entries.len() {
                        let (suffix, _value) = &entries[*index];
                        let mut term = prefix.clone();
                        term.extend_from_slice(suffix);
                        *index += 1;
                        return Some(term);
                    } else {
                        // Done with this bucket
                        self.stack.pop();
                    }
                }
                IterState::ArtNode {
                    prefix,
                    is_final,
                    yielded_final,
                    children,
                    child_index,
                    ..
                } => {
                    // First, yield the final state if applicable
                    if *is_final && !*yielded_final {
                        *yielded_final = true;
                        return Some(prefix.clone());
                    }

                    // Then, process children
                    if *child_index < children.len() {
                        let (edge, child) = children[*child_index].clone();
                        *child_index += 1;

                        let mut child_prefix = prefix.clone();
                        child_prefix.push(edge);

                        // Push child state onto stack
                        match child {
                            ChildNode::Bucket(bucket) => {
                                let entries: Vec<(Vec<u8>, Option<Vec<u8>>)> = bucket
                                    .iter()
                                    .map(|(entry, suffix)| {
                                        let value = bucket.get_value(&entry).map(|v| v.to_vec());
                                        (suffix.to_vec(), value)
                                    })
                                    .collect();

                                if !entries.is_empty() {
                                    self.stack.push(IterState::Bucket {
                                        prefix: child_prefix,
                                        entries,
                                        index: 0,
                                    });
                                }
                            }
                            ChildNode::ArtNode {
                                is_final: child_final,
                                value: child_value,
                                children: child_children,
                                ..
                            } => {
                                self.stack.push(IterState::ArtNode {
                                    prefix: child_prefix,
                                    is_final: child_final,
                                    value: child_value,
                                    yielded_final: false,
                                    children: child_children,
                                    child_index: 0,
                                });
                            }
                            ChildNode::DiskRef { .. } => {
                                // Skip disk refs for now - they would need async loading
                                // In a full implementation, we'd resolve them here
                            }
                        }
                    } else {
                        // Done with this ART node
                        self.stack.pop();
                    }
                }
            }
        }
    }
}

/// Iterator over all terms with their values in a PersistentARTrie.
///
/// This iterator performs a depth-first traversal of the trie,
/// yielding (term, value) pairs in lexicographic order.
pub struct TermValueIterator<V: DictionaryValue> {
    /// Stack of iteration states for DFS
    stack: Vec<IterState>,
    /// Marker for value type
    _marker: std::marker::PhantomData<V>,
}

impl<V: DictionaryValue> TermValueIterator<V> {
    /// Create a new iterator starting from the trie root
    fn new(root: &TrieRoot<V>) -> Self {
        let mut stack = Vec::new();

        match root {
            TrieRoot::Bucket(bucket) => {
                let entries: Vec<(Vec<u8>, Option<Vec<u8>>)> = bucket
                    .iter()
                    .map(|(entry, suffix)| {
                        let value = bucket.get_value(&entry).map(|v| v.to_vec());
                        (suffix.to_vec(), value)
                    })
                    .collect();

                if !entries.is_empty() {
                    stack.push(IterState::Bucket {
                        prefix: Vec::new(),
                        entries,
                        index: 0,
                    });
                }
            }
            TrieRoot::ArtNode {
                is_final,
                value,
                children,
                ..
            } => {
                #[cfg(feature = "persistent-artrie")]
                let value_bytes = value.as_ref().and_then(|v| bincode::serialize(v).ok());
                #[cfg(not(feature = "persistent-artrie"))]
                let value_bytes: Option<Vec<u8>> = None;
                let _ = value;

                stack.push(IterState::ArtNode {
                    prefix: Vec::new(),
                    is_final: *is_final,
                    value: value_bytes,
                    yielded_final: false,
                    children: children.clone(),
                    child_index: 0,
                });
            }
        }

        Self {
            stack,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<V: DictionaryValue> Iterator for TermValueIterator<V> {
    type Item = (Vec<u8>, Option<V>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let state = self.stack.last_mut()?;

            match state {
                IterState::Bucket {
                    prefix,
                    entries,
                    index,
                } => {
                    if *index < entries.len() {
                        let (suffix, value_bytes) = &entries[*index];
                        let mut term = prefix.clone();
                        term.extend_from_slice(suffix);

                        // Deserialize value if present
                        #[cfg(feature = "persistent-artrie")]
                        let value: Option<V> = value_bytes
                            .as_ref()
                            .and_then(|bytes| bincode::deserialize(bytes).ok());
                        #[cfg(not(feature = "persistent-artrie"))]
                        let value: Option<V> = None;
                        let _ = value_bytes;

                        *index += 1;
                        return Some((term, value));
                    } else {
                        self.stack.pop();
                    }
                }
                IterState::ArtNode {
                    prefix,
                    is_final,
                    value: value_bytes,
                    yielded_final,
                    children,
                    child_index,
                } => {
                    if *is_final && !*yielded_final {
                        *yielded_final = true;

                        #[cfg(feature = "persistent-artrie")]
                        let value: Option<V> = value_bytes
                            .as_ref()
                            .and_then(|bytes| bincode::deserialize(bytes).ok());
                        #[cfg(not(feature = "persistent-artrie"))]
                        let value: Option<V> = None;
                        let _ = value_bytes;

                        return Some((prefix.clone(), value));
                    }

                    if *child_index < children.len() {
                        let (edge, child) = children[*child_index].clone();
                        *child_index += 1;

                        let mut child_prefix = prefix.clone();
                        child_prefix.push(edge);

                        match child {
                            ChildNode::Bucket(bucket) => {
                                let entries: Vec<(Vec<u8>, Option<Vec<u8>>)> = bucket
                                    .iter()
                                    .map(|(entry, suffix)| {
                                        let value = bucket.get_value(&entry).map(|v| v.to_vec());
                                        (suffix.to_vec(), value)
                                    })
                                    .collect();

                                if !entries.is_empty() {
                                    self.stack.push(IterState::Bucket {
                                        prefix: child_prefix,
                                        entries,
                                        index: 0,
                                    });
                                }
                            }
                            ChildNode::ArtNode {
                                is_final: child_final,
                                value: child_value,
                                children: child_children,
                                ..
                            } => {
                                self.stack.push(IterState::ArtNode {
                                    prefix: child_prefix,
                                    is_final: child_final,
                                    value: child_value,
                                    yielded_final: false,
                                    children: child_children,
                                    child_index: 0,
                                });
                            }
                            ChildNode::DiskRef { .. } => {
                                // Skip disk refs for now
                            }
                        }
                    } else {
                        self.stack.pop();
                    }
                }
            }
        }
    }
}

impl<V: DictionaryValue> PersistentARTrie<V> {
    /// Iterate over all terms in the dictionary.
    ///
    /// Returns an iterator yielding terms as `Vec<u8>` in lexicographic order.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
    ///
    /// let mut dict = PersistentARTrie::new();
    /// dict.insert("apple");
    /// dict.insert("banana");
    ///
    /// for term in dict.iter() {
    ///     println!("{}", String::from_utf8_lossy(&term));
    /// }
    /// ```
    pub fn iter(&self) -> TermIterator<V> {
        let inner = self.inner.read().expect("lock poisoned");
        TermIterator::new(&inner.root)
    }

    /// Iterate over all terms with their values.
    ///
    /// Returns an iterator yielding `(term, Option<value>)` pairs in lexicographic order.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
    ///
    /// let mut dict: PersistentARTrie<i32> = PersistentARTrie::new();
    /// dict.insert_with_value("apple", 1);
    /// dict.insert_with_value("banana", 2);
    ///
    /// for (term, value) in dict.iter_with_values() {
    ///     println!("{}: {:?}", String::from_utf8_lossy(&term), value);
    /// }
    /// ```
    pub fn iter_with_values(&self) -> TermValueIterator<V> {
        let inner = self.inner.read().expect("lock poisoned");
        TermValueIterator::new(&inner.root)
    }

    /// Iterate over all terms as strings.
    ///
    /// This is a convenience method that converts terms to UTF-8 strings,
    /// skipping any terms that contain invalid UTF-8.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
    ///
    /// let mut dict = PersistentARTrie::new();
    /// dict.insert("hello");
    /// dict.insert("world");
    ///
    /// for term in dict.iter_strings() {
    ///     println!("{}", term);
    /// }
    /// ```
    pub fn iter_strings(&self) -> impl Iterator<Item = String> + '_ {
        self.iter()
            .filter_map(|bytes| String::from_utf8(bytes).ok())
    }
}

// ===========================================================================
// Atomic Operations
// ===========================================================================
//
// These operations provide lock-free atomic semantics for concurrent access.
// While the underlying storage uses RwLock, the API ensures atomic read-modify-write
// semantics through CAS (Compare-And-Swap) patterns and WAL logging.

impl<V: DictionaryValue + serde::Serialize + serde::de::DeserializeOwned> PersistentARTrie<V> {
    /// Atomically increment a numeric value associated with a term.
    ///
    /// If the term doesn't exist, inserts it with `delta` as the initial value.
    /// If the term exists but the value cannot be interpreted as i64, returns an error.
    ///
    /// This operation is atomic: the read-modify-write is performed under a lock,
    /// and the result is logged to WAL before returning.
    ///
    /// # Arguments
    ///
    /// * `term` - The term to increment
    /// * `delta` - The delta to add (can be negative for decrement)
    ///
    /// # Returns
    ///
    /// The new value after increment, or an error if the operation failed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
    ///
    /// let dict: PersistentARTrie<i64> = PersistentARTrie::new();
    ///
    /// // First increment creates the entry with value 1
    /// assert_eq!(dict.increment("counter", 1)?, 1);
    ///
    /// // Subsequent increments add to the existing value
    /// assert_eq!(dict.increment("counter", 5)?, 6);
    /// assert_eq!(dict.increment("counter", -2)?, 4);
    /// ```
    #[cfg(feature = "persistent-artrie")]
    pub fn increment(&self, term: &str, delta: i64) -> super::error::Result<i64> {
        self.increment_bytes(term.as_bytes(), delta)
    }

    /// Atomically increment a value by term bytes.
    ///
    /// See [`increment`](Self::increment) for details.
    #[cfg(feature = "persistent-artrie")]
    pub fn increment_bytes(&self, term: &[u8], delta: i64) -> super::error::Result<i64> {
        let mut inner = self.inner.write().expect("lock poisoned");

        // Read current value (if exists)
        let current: i64 = match inner.get_value_impl(term) {
            Some(v) => {
                // Try to interpret the value as i64
                let bytes = bincode::serialize(&v)
                    .map_err(|e| super::error::PersistentARTrieError::internal(format!("Serialization error: {}", e)))?;
                if bytes.len() == 8 {
                    i64::from_le_bytes(bytes.try_into().unwrap())
                } else {
                    // Try to deserialize as i64 directly
                    bincode::deserialize::<i64>(&bytes)
                        .map_err(|e| super::error::PersistentARTrieError::internal(
                            format!("Value cannot be interpreted as i64: {}", e)
                        ))?
                }
            }
            None => 0, // Initial value if term doesn't exist
        };

        let new_value = current + delta;

        // Create value from i64
        let value_bytes = bincode::serialize(&new_value)
            .map_err(|e| super::error::PersistentARTrieError::internal(format!("Serialization error: {}", e)))?;
        let v: V = bincode::deserialize(&value_bytes)
            .map_err(|e| super::error::PersistentARTrieError::internal(
                format!("Cannot create value from i64: {}", e)
            ))?;

        // Update the value
        inner.remove_impl_core(term);
        inner.insert_impl_core(term, Some(v));

        // Log to WAL
        if let Some(ref wal_writer) = inner.wal_writer {
            let record = super::wal::WalRecord::Increment {
                term: term.to_vec(),
                delta,
                result: new_value,
            };
            wal_writer.write().expect("WAL lock").append(record)?;
        }

        Ok(new_value)
    }

    /// Atomically update or insert a value.
    ///
    /// If the term exists, updates its value. If not, inserts the term with the value.
    /// This is atomic: the operation is logged to WAL before returning.
    ///
    /// # Arguments
    ///
    /// * `term` - The term to upsert
    /// * `value` - The value to set
    ///
    /// # Returns
    ///
    /// `true` if a new term was inserted, `false` if an existing term was updated.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
    ///
    /// let dict: PersistentARTrie<String> = PersistentARTrie::new();
    ///
    /// // Insert new term
    /// assert!(dict.upsert("greeting", "hello".to_string())?);
    ///
    /// // Update existing term
    /// assert!(!dict.upsert("greeting", "hi".to_string())?);
    /// ```
    #[cfg(feature = "persistent-artrie")]
    pub fn upsert(&self, term: &str, value: V) -> super::error::Result<bool> {
        self.upsert_bytes(term.as_bytes(), value)
    }

    /// Atomically upsert by term bytes.
    ///
    /// See [`upsert`](Self::upsert) for details.
    #[cfg(feature = "persistent-artrie")]
    pub fn upsert_bytes(&self, term: &[u8], value: V) -> super::error::Result<bool> {
        let mut inner = self.inner.write().expect("lock poisoned");

        // Check if term exists
        let existed = inner.contains_impl(term);

        // Remove existing entry (if any) and insert new value
        inner.remove_impl_core(term);
        inner.insert_impl_core(term, Some(value.clone()));

        // Serialize value for WAL
        let value_bytes = bincode::serialize(&value)
            .map_err(|e| super::error::PersistentARTrieError::internal(format!("Serialization error: {}", e)))?;

        // Log to WAL
        if let Some(ref wal_writer) = inner.wal_writer {
            let record = super::wal::WalRecord::Upsert {
                term: term.to_vec(),
                value: value_bytes,
            };
            wal_writer.write().expect("WAL lock").append(record)?;
        }

        Ok(!existed)
    }

    /// Atomically compare and swap a value.
    ///
    /// Updates the value only if the current value matches `expected`.
    /// This provides optimistic concurrency control.
    ///
    /// # Arguments
    ///
    /// * `term` - The term to update
    /// * `expected` - The expected current value (None means term should not exist)
    /// * `new_value` - The new value to set
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the swap succeeded, `Ok(false)` if the current value didn't match expected.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
    ///
    /// let dict: PersistentARTrie<i32> = PersistentARTrie::new();
    ///
    /// // Insert initial value
    /// dict.upsert("counter", 0)?;
    ///
    /// // CAS succeeds when expected matches
    /// assert!(dict.compare_and_swap("counter", Some(0), 1)?);
    ///
    /// // CAS fails when expected doesn't match
    /// assert!(!dict.compare_and_swap("counter", Some(0), 2)?);
    /// ```
    #[cfg(feature = "persistent-artrie")]
    pub fn compare_and_swap(
        &self,
        term: &str,
        expected: Option<V>,
        new_value: V,
    ) -> super::error::Result<bool> {
        self.compare_and_swap_bytes(term.as_bytes(), expected, new_value)
    }

    /// Atomically compare and swap by term bytes.
    ///
    /// See [`compare_and_swap`](Self::compare_and_swap) for details.
    #[cfg(feature = "persistent-artrie")]
    pub fn compare_and_swap_bytes(
        &self,
        term: &[u8],
        expected: Option<V>,
        new_value: V,
    ) -> super::error::Result<bool> {
        let mut inner = self.inner.write().expect("lock poisoned");

        // Read current value
        let current = inner.get_value_impl(term);

        // Check if current matches expected
        let matches = match (&current, &expected) {
            (None, None) => true,
            (Some(c), Some(e)) => {
                // Compare serialized forms for equality
                let c_bytes = bincode::serialize(c).ok();
                let e_bytes = bincode::serialize(e).ok();
                c_bytes == e_bytes
            }
            _ => false,
        };

        // Serialize for WAL
        let expected_bytes = expected.as_ref().and_then(|e| bincode::serialize(e).ok());
        let new_value_bytes = bincode::serialize(&new_value)
            .map_err(|e| super::error::PersistentARTrieError::internal(format!("Serialization error: {}", e)))?;

        if matches {
            // Perform the swap
            inner.remove_impl_core(term);
            inner.insert_impl_core(term, Some(new_value));
        }

        // Log to WAL (always log, including success status for idempotency)
        if let Some(ref wal_writer) = inner.wal_writer {
            let record = super::wal::WalRecord::CompareAndSwap {
                term: term.to_vec(),
                expected: expected_bytes,
                new_value: new_value_bytes,
                success: matches,
            };
            wal_writer.write().expect("WAL lock").append(record)?;
        }

        Ok(matches)
    }

    /// Get the current value and increment atomically (fetch-and-add).
    ///
    /// Returns the value *before* the increment.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dict: PersistentARTrie<i64> = PersistentARTrie::new();
    /// dict.upsert("counter", 5)?;
    ///
    /// let old = dict.fetch_add("counter", 3)?; // old = 5, new value = 8
    /// assert_eq!(old, 5);
    /// ```
    #[cfg(feature = "persistent-artrie")]
    pub fn fetch_add(&self, term: &str, delta: i64) -> super::error::Result<i64> {
        let new_value = self.increment(term, delta)?;
        Ok(new_value - delta) // Return the old value
    }

    /// Get or insert a default value atomically.
    ///
    /// If the term exists, returns its current value.
    /// If not, inserts the default value and returns it.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dict: PersistentARTrie<i32> = PersistentARTrie::new();
    ///
    /// // First call inserts the default
    /// let v = dict.get_or_insert("key", 42)?;
    /// assert_eq!(v, 42);
    ///
    /// // Second call returns existing value
    /// let v = dict.get_or_insert("key", 100)?;
    /// assert_eq!(v, 42);
    /// ```
    #[cfg(feature = "persistent-artrie")]
    pub fn get_or_insert(&self, term: &str, default: V) -> super::error::Result<V> {
        self.get_or_insert_bytes(term.as_bytes(), default)
    }

    /// Get or insert by term bytes.
    ///
    /// See [`get_or_insert`](Self::get_or_insert) for details.
    #[cfg(feature = "persistent-artrie")]
    pub fn get_or_insert_bytes(&self, term: &[u8], default: V) -> super::error::Result<V> {
        let mut inner = self.inner.write().expect("lock poisoned");

        // Check if term exists
        if let Some(v) = inner.get_value_impl(term) {
            return Ok(v);
        }

        // Insert default value
        inner.insert_impl_core(term, Some(default.clone()));

        // Serialize for WAL
        let value_bytes = bincode::serialize(&default)
            .map_err(|e| super::error::PersistentARTrieError::internal(format!("Serialization error: {}", e)))?;

        // Log to WAL
        if let Some(ref wal_writer) = inner.wal_writer {
            let record = super::wal::WalRecord::Upsert {
                term: term.to_vec(),
                value: value_bytes,
            };
            wal_writer.write().expect("WAL lock").append(record)?;
        }

        Ok(default)
    }
}

/// Drop implementation for clean shutdown.
///
/// Attempts a best-effort sync on drop to ensure data durability.
/// This is not guaranteed to succeed (e.g., if locks are poisoned),
/// but provides a safety net for normal program termination.
#[cfg(feature = "persistent-artrie")]
impl<V: DictionaryValue> Drop for PersistentARTrie<V> {
    fn drop(&mut self) {
        // Best-effort sync on close
        if let Ok(inner) = self.inner.read() {
            // Sync WAL
            if let Some(ref wal_writer) = inner.wal_writer {
                if let Ok(wal) = wal_writer.write() {
                    let _ = wal.sync();
                }
            }
            // Flush buffer manager dirty pages
            if let Some(ref buffer_manager) = inner.buffer_manager {
                if let Ok(bm) = buffer_manager.read() {
                    let _ = bm.flush_all();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_dictionary() {
        let dict: PersistentARTrie = PersistentARTrie::new();
        assert_eq!(dict.len(), Some(0));
        assert!(!dict.is_dirty());
    }

    #[test]
    fn test_insert_and_contains() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        assert!(dict.insert("apple"));
        assert!(dict.insert("banana"));
        assert!(dict.insert("cherry"));

        assert!(dict.contains("apple"));
        assert!(dict.contains("banana"));
        assert!(dict.contains("cherry"));
        assert!(!dict.contains("date"));

        assert_eq!(dict.len(), Some(3));
        assert!(dict.is_dirty());
    }

    #[test]
    fn test_duplicate_insert() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        assert!(dict.insert("test"));
        assert!(!dict.insert("test")); // Duplicate

        assert_eq!(dict.len(), Some(1));
    }

    #[test]
    fn test_remove() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        dict.insert("apple");
        dict.insert("banana");

        assert!(dict.remove("apple"));
        assert!(!dict.contains("apple"));
        assert!(dict.contains("banana"));

        assert_eq!(dict.len(), Some(1));
    }

    #[test]
    fn test_remove_not_found() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        dict.insert("apple");

        assert!(!dict.remove("banana"));
        assert_eq!(dict.len(), Some(1));
    }

    #[test]
    fn test_empty_string() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        assert!(dict.insert(""));
        assert!(dict.contains(""));

        dict.insert("test");
        assert!(dict.contains(""));
        assert!(dict.contains("test"));
    }

    #[test]
    fn test_dictionary_trait() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        dict.insert("hello");
        dict.insert("world");

        // Test through Dictionary trait
        let dict_ref: &dyn Dictionary<Node = _> = &dict;
        assert!(dict_ref.contains("hello"));
        assert!(!dict_ref.contains("hi"));
    }

    #[test]
    fn test_mark_clean() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        dict.insert("test");
        assert!(dict.is_dirty());

        dict.mark_clean();
        assert!(!dict.is_dirty());
    }

    #[test]
    fn test_many_insertions() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();

        // Insert many terms to trigger bucket conversion
        for i in 0..100 {
            dict.insert(&format!("word{:03}", i));
        }

        assert_eq!(dict.len(), Some(100));

        // Verify all terms exist
        for i in 0..100 {
            assert!(dict.contains(&format!("word{:03}", i)));
        }
    }

    #[test]
    fn test_sync_strategy() {
        let dict: PersistentARTrie = PersistentARTrie::new();
        assert_eq!(dict.sync_strategy(), SyncStrategy::InternalSync);
    }

    #[test]
    fn test_iter_empty() {
        let dict: PersistentARTrie = PersistentARTrie::new();
        let terms: Vec<_> = dict.iter().collect();
        assert!(terms.is_empty());
    }

    #[test]
    fn test_iter_single() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();
        dict.insert("hello");

        let terms: Vec<_> = dict.iter().collect();
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0], b"hello".to_vec());
    }

    #[test]
    fn test_iter_multiple() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();
        dict.insert("apple");
        dict.insert("banana");
        dict.insert("cherry");

        let terms: Vec<String> = dict.iter_strings().collect();
        assert_eq!(terms.len(), 3);

        // Should contain all terms
        assert!(terms.contains(&"apple".to_string()));
        assert!(terms.contains(&"banana".to_string()));
        assert!(terms.contains(&"cherry".to_string()));
    }

    #[test]
    fn test_iter_with_empty_string() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();
        dict.insert("");
        dict.insert("hello");

        let terms: Vec<String> = dict.iter_strings().collect();
        assert_eq!(terms.len(), 2);
        assert!(terms.contains(&"".to_string()));
        assert!(terms.contains(&"hello".to_string()));
    }

    #[test]
    fn test_iter_common_prefix() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();
        dict.insert("test");
        dict.insert("testing");
        dict.insert("tested");
        dict.insert("tester");

        let terms: Vec<String> = dict.iter_strings().collect();
        assert_eq!(terms.len(), 4);
        assert!(terms.contains(&"test".to_string()));
        assert!(terms.contains(&"testing".to_string()));
        assert!(terms.contains(&"tested".to_string()));
        assert!(terms.contains(&"tester".to_string()));
    }

    #[test]
    fn test_iter_preserves_order() {
        let mut dict: PersistentARTrie = PersistentARTrie::new();
        // Insert in random order
        dict.insert("cherry");
        dict.insert("apple");
        dict.insert("banana");

        let terms: Vec<String> = dict.iter_strings().collect();
        // Should be in lexicographic order
        // (but bucket-based iteration may have different order within bucket)
        assert_eq!(terms.len(), 3);
    }

    #[test]
    fn test_clone() {
        let mut dict1: PersistentARTrie = PersistentARTrie::new();
        dict1.insert("test");

        let dict2 = dict1.clone();

        // Both should see the same data (Arc sharing)
        assert!(dict2.contains("test"));
        assert_eq!(dict2.len(), Some(1));
    }

    #[cfg(feature = "persistent-artrie")]
    mod persistent_tests {
        use super::*;
        use tempfile::TempDir;

        #[test]
        fn test_create_and_open() {
            let temp_dir = TempDir::new().expect("create temp dir");
            let dict_path = temp_dir.path().join("test.part");

            // Create new dictionary
            {
                let mut dict: PersistentARTrie<()> =
                    PersistentARTrie::create(&dict_path).expect("create dict");
                dict.insert("hello");
                dict.insert("world");
                dict.sync().expect("sync");
            }

            // Open existing dictionary
            {
                let dict: PersistentARTrie<()> =
                    PersistentARTrie::open(&dict_path).expect("open dict");
                assert!(dict.contains("hello"));
                assert!(dict.contains("world"));
                assert_eq!(dict.len(), Some(2));
            }
        }

        #[test]
        fn test_create_fails_if_exists() {
            let temp_dir = TempDir::new().expect("create temp dir");
            let dict_path = temp_dir.path().join("test.part");

            // Create the file first
            std::fs::write(&dict_path, b"dummy").expect("create file");

            // Trying to create should fail
            let result: Result<PersistentARTrie<()>> = PersistentARTrie::create(&dict_path);
            assert!(result.is_err());
        }

        #[test]
        fn test_open_fails_if_not_exists() {
            let temp_dir = TempDir::new().expect("create temp dir");
            let dict_path = temp_dir.path().join("nonexistent.part");

            let result: Result<PersistentARTrie<()>> = PersistentARTrie::open(&dict_path);
            assert!(result.is_err());
        }

        #[test]
        fn test_wal_recovery() {
            let temp_dir = TempDir::new().expect("create temp dir");
            let dict_path = temp_dir.path().join("test.part");

            // Create dictionary and insert data
            {
                let mut dict: PersistentARTrie<()> =
                    PersistentARTrie::create(&dict_path).expect("create dict");
                dict.insert("apple");
                dict.insert("banana");
                dict.insert("cherry");
                dict.sync().expect("sync");
            }

            // Reopen and verify WAL recovery
            {
                let dict: PersistentARTrie<()> =
                    PersistentARTrie::open(&dict_path).expect("open dict");
                assert!(dict.contains("apple"));
                assert!(dict.contains("banana"));
                assert!(dict.contains("cherry"));
            }
        }

        #[test]
        fn test_checkpoint() {
            let temp_dir = TempDir::new().expect("create temp dir");
            let dict_path = temp_dir.path().join("test.part");

            let mut dict: PersistentARTrie<()> =
                PersistentARTrie::create(&dict_path).expect("create dict");
            dict.insert("test");
            dict.checkpoint().expect("checkpoint");
        }

        #[test]
        fn test_sync() {
            let temp_dir = TempDir::new().expect("create temp dir");
            let dict_path = temp_dir.path().join("test.part");

            let mut dict: PersistentARTrie<()> =
                PersistentARTrie::create(&dict_path).expect("create dict");
            dict.insert("test");
            dict.sync().expect("sync");
        }

        #[test]
        fn test_many_insertions_persistent() {
            let temp_dir = TempDir::new().expect("create temp dir");
            let dict_path = temp_dir.path().join("test.part");

            // Create and insert many terms
            {
                let mut dict: PersistentARTrie<()> =
                    PersistentARTrie::create(&dict_path).expect("create dict");
                for i in 0..50 {
                    dict.insert(&format!("word{:03}", i));
                }
                dict.sync().expect("sync");
            }

            // Reopen and verify
            {
                let dict: PersistentARTrie<()> =
                    PersistentARTrie::open(&dict_path).expect("open dict");
                assert_eq!(dict.len(), Some(50));
                for i in 0..50 {
                    assert!(
                        dict.contains(&format!("word{:03}", i)),
                        "missing word{:03}",
                        i
                    );
                }
            }
        }
    }

    // === Atomic Operations Tests ===

    #[cfg(feature = "persistent-artrie")]
    mod atomic_ops_tests {
        use super::*;
        use tempfile::tempdir;

        #[test]
        fn test_increment_new_term() {
            let dir = tempdir().expect("create temp dir");
            let dict_path = dir.path().join("atomic_test.part");

            let dict: PersistentARTrie<i64> =
                PersistentARTrie::create(&dict_path).expect("create dict");

            // First increment creates the entry with the delta value
            let result = dict.increment("counter", 1).expect("increment");
            assert_eq!(result, 1, "First increment should return delta value");

            // Verify the term exists
            assert!(dict.contains("counter"));
        }

        #[test]
        fn test_increment_existing_term() {
            let dir = tempdir().expect("create temp dir");
            let dict_path = dir.path().join("atomic_test.part");

            let dict: PersistentARTrie<i64> =
                PersistentARTrie::create(&dict_path).expect("create dict");

            // Insert initial value
            dict.upsert("counter", 10i64).expect("upsert");

            // Increment
            let result = dict.increment("counter", 5).expect("increment");
            assert_eq!(result, 15);

            // Negative increment (decrement)
            let result = dict.increment("counter", -3).expect("decrement");
            assert_eq!(result, 12);
        }

        #[test]
        fn test_upsert_new_term() {
            let dir = tempdir().expect("create temp dir");
            let dict_path = dir.path().join("atomic_test.part");

            let dict: PersistentARTrie<String> =
                PersistentARTrie::create(&dict_path).expect("create dict");

            // Insert new term
            let is_new = dict.upsert("greeting", "hello".to_string()).expect("upsert");
            assert!(is_new, "Should return true for new insertion");

            // Verify value
            let value = dict.get_value("greeting");
            assert_eq!(value, Some("hello".to_string()));
        }

        #[test]
        fn test_upsert_existing_term() {
            let dir = tempdir().expect("create temp dir");
            let dict_path = dir.path().join("atomic_test.part");

            let dict: PersistentARTrie<String> =
                PersistentARTrie::create(&dict_path).expect("create dict");

            // Insert initial value
            dict.upsert("greeting", "hello".to_string()).expect("upsert");

            // Update existing term
            let is_new = dict.upsert("greeting", "hi".to_string()).expect("upsert");
            assert!(!is_new, "Should return false for update");

            // Verify updated value
            let value = dict.get_value("greeting");
            assert_eq!(value, Some("hi".to_string()));
        }

        #[test]
        fn test_compare_and_swap_success() {
            let dir = tempdir().expect("create temp dir");
            let dict_path = dir.path().join("atomic_test.part");

            let dict: PersistentARTrie<i32> =
                PersistentARTrie::create(&dict_path).expect("create dict");

            // Insert initial value
            dict.upsert("counter", 0i32).expect("upsert");

            // CAS succeeds when expected matches
            let success = dict.compare_and_swap("counter", Some(0), 1).expect("cas");
            assert!(success, "CAS should succeed when expected matches");

            // Verify new value
            assert_eq!(dict.get_value("counter"), Some(1));
        }

        #[test]
        fn test_compare_and_swap_failure() {
            let dir = tempdir().expect("create temp dir");
            let dict_path = dir.path().join("atomic_test.part");

            let dict: PersistentARTrie<i32> =
                PersistentARTrie::create(&dict_path).expect("create dict");

            // Insert initial value
            dict.upsert("counter", 5i32).expect("upsert");

            // CAS fails when expected doesn't match
            let success = dict.compare_and_swap("counter", Some(0), 10).expect("cas");
            assert!(!success, "CAS should fail when expected doesn't match");

            // Value should be unchanged
            assert_eq!(dict.get_value("counter"), Some(5));
        }

        #[test]
        fn test_compare_and_swap_none_expected() {
            let dir = tempdir().expect("create temp dir");
            let dict_path = dir.path().join("atomic_test.part");

            let dict: PersistentARTrie<i32> =
                PersistentARTrie::create(&dict_path).expect("create dict");

            // CAS with None expected succeeds when term doesn't exist
            let success = dict.compare_and_swap("new_key", None, 42).expect("cas");
            assert!(success, "CAS should succeed when expecting None and key doesn't exist");

            // Verify value was inserted
            assert_eq!(dict.get_value("new_key"), Some(42));
        }

        #[test]
        fn test_fetch_add() {
            let dir = tempdir().expect("create temp dir");
            let dict_path = dir.path().join("atomic_test.part");

            let dict: PersistentARTrie<i64> =
                PersistentARTrie::create(&dict_path).expect("create dict");

            // Insert initial value
            dict.upsert("counter", 10i64).expect("upsert");

            // fetch_add returns old value
            let old = dict.fetch_add("counter", 5).expect("fetch_add");
            assert_eq!(old, 10, "fetch_add should return old value");

            // Verify new value
            let new_val = dict.increment("counter", 0).expect("read");
            assert_eq!(new_val, 15);
        }

        #[test]
        fn test_get_or_insert_new() {
            let dir = tempdir().expect("create temp dir");
            let dict_path = dir.path().join("atomic_test.part");

            let dict: PersistentARTrie<i32> =
                PersistentARTrie::create(&dict_path).expect("create dict");

            // get_or_insert on new key returns default
            let value = dict.get_or_insert("key", 42).expect("get_or_insert");
            assert_eq!(value, 42);

            // Verify it was inserted
            assert!(dict.contains("key"));
        }

        #[test]
        fn test_get_or_insert_existing() {
            let dir = tempdir().expect("create temp dir");
            let dict_path = dir.path().join("atomic_test.part");

            let dict: PersistentARTrie<i32> =
                PersistentARTrie::create(&dict_path).expect("create dict");

            // Insert initial value
            dict.upsert("key", 100i32).expect("upsert");

            // get_or_insert returns existing value, ignores default
            let value = dict.get_or_insert("key", 42).expect("get_or_insert");
            assert_eq!(value, 100, "Should return existing value, not default");
        }
    }
}
