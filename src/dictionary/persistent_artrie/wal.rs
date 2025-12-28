//! Write-Ahead Log (WAL) for crash recovery.
//!
//! This module implements a redo-only WAL for the PersistentARTrie. The WAL
//! ensures durability by logging operations before they are applied to the
//! main data structure.
//!
//! # Design
//!
//! The WAL uses a redo-only approach:
//! - Operations are logged before being applied
//! - On crash recovery, the log is replayed from the last checkpoint
//! - Periodic checkpoints truncate the log
//!
//! # Record Format
//!
//! Each record has the following layout:
//! ```text
//! +----------+----------+----------+----------+----------+
//! | CRC32    | Length   | LSN      | Type     | Payload  |
//! | (4 bytes)| (4 bytes)| (8 bytes)| (1 byte) | (varies) |
//! +----------+----------+----------+----------+----------+
//! ```
//!
//! # Group Commit
//!
//! For performance, multiple operations can be batched into a single fsync:
//! - Writers append to the log buffer
//! - A background thread (or explicit flush) fsyncs periodically
//! - Writers wait for their LSN to be durable before returning
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::persistent_artrie::wal::{Wal, WalRecord, WalRecordType};
//!
//! let wal = Wal::create("data.wal")?;
//!
//! // Log an insert operation
//! let lsn = wal.append(WalRecord::Insert {
//!     term: "hello".as_bytes().to_vec(),
//!     value: None,
//! })?;
//!
//! // Ensure durability
//! wal.sync()?;
//!
//! // On recovery
//! let wal = Wal::open("data.wal")?;
//! for record in wal.iter() {
//!     // Replay the operation
//! }
//! ```

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Log Sequence Number - monotonically increasing identifier for log records.
pub type Lsn = u64;

/// CRC32 for record integrity verification.
fn crc32(data: &[u8]) -> u32 {
    // Simple CRC32 implementation (IEEE polynomial)
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// WAL record types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalRecordType {
    /// Insert a term (with optional value)
    Insert = 1,
    /// Remove a term
    Remove = 2,
    /// Checkpoint marker
    Checkpoint = 3,
    /// Begin transaction (for future use)
    BeginTx = 4,
    /// Commit transaction (for future use)
    CommitTx = 5,
    /// Abort transaction (for future use)
    AbortTx = 6,
    /// Atomic increment operation
    Increment = 7,
    /// Atomic upsert operation (update if exists, insert if not)
    Upsert = 8,
    /// Atomic compare-and-swap operation
    CompareAndSwap = 9,
}

impl TryFrom<u8> for WalRecordType {
    type Error = WalError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(WalRecordType::Insert),
            2 => Ok(WalRecordType::Remove),
            3 => Ok(WalRecordType::Checkpoint),
            4 => Ok(WalRecordType::BeginTx),
            5 => Ok(WalRecordType::CommitTx),
            6 => Ok(WalRecordType::AbortTx),
            7 => Ok(WalRecordType::Increment),
            8 => Ok(WalRecordType::Upsert),
            9 => Ok(WalRecordType::CompareAndSwap),
            _ => Err(WalError::InvalidRecordType(value)),
        }
    }
}

/// A WAL record containing an operation to replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalRecord {
    /// Insert a term with optional serialized value
    Insert {
        /// The term to insert (UTF-8 bytes)
        term: Vec<u8>,
        /// Optional serialized value
        value: Option<Vec<u8>>,
    },
    /// Remove a term
    Remove {
        /// The term to remove
        term: Vec<u8>,
    },
    /// Checkpoint marker with metadata
    Checkpoint {
        /// LSN up to which data is durable in the main file
        checkpoint_lsn: Lsn,
        /// Timestamp of checkpoint
        timestamp: u64,
    },
    /// Begin transaction
    BeginTx {
        /// Transaction ID
        tx_id: u64,
    },
    /// Commit transaction
    CommitTx {
        /// Transaction ID
        tx_id: u64,
    },
    /// Abort transaction
    AbortTx {
        /// Transaction ID
        tx_id: u64,
    },
    /// Atomic increment operation
    ///
    /// Increments the value associated with a term by `delta`.
    /// If the term doesn't exist, inserts with `delta` as the initial value.
    Increment {
        /// The term to increment
        term: Vec<u8>,
        /// The delta to add (can be negative)
        delta: i64,
        /// The resulting value after increment
        result: i64,
    },
    /// Atomic upsert operation
    ///
    /// Updates the value if the term exists, otherwise inserts a new term.
    Upsert {
        /// The term to upsert
        term: Vec<u8>,
        /// The new serialized value
        value: Vec<u8>,
    },
    /// Atomic compare-and-swap operation
    ///
    /// Updates the value only if the current value matches `expected`.
    CompareAndSwap {
        /// The term to update
        term: Vec<u8>,
        /// The expected current value (None means term should not exist)
        expected: Option<Vec<u8>>,
        /// The new value to set
        new_value: Vec<u8>,
        /// Whether the swap succeeded
        success: bool,
    },
}

impl WalRecord {
    /// Get the record type.
    pub fn record_type(&self) -> WalRecordType {
        match self {
            WalRecord::Insert { .. } => WalRecordType::Insert,
            WalRecord::Remove { .. } => WalRecordType::Remove,
            WalRecord::Checkpoint { .. } => WalRecordType::Checkpoint,
            WalRecord::BeginTx { .. } => WalRecordType::BeginTx,
            WalRecord::CommitTx { .. } => WalRecordType::CommitTx,
            WalRecord::AbortTx { .. } => WalRecordType::AbortTx,
            WalRecord::Increment { .. } => WalRecordType::Increment,
            WalRecord::Upsert { .. } => WalRecordType::Upsert,
            WalRecord::CompareAndSwap { .. } => WalRecordType::CompareAndSwap,
        }
    }

    /// Serialize the record payload to bytes.
    pub fn serialize_payload(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        match self {
            WalRecord::Insert { term, value } => {
                // Term length (4 bytes) + term + has_value (1 byte) + [value_length + value]
                buf.extend_from_slice(&(term.len() as u32).to_le_bytes());
                buf.extend_from_slice(term);
                if let Some(v) = value {
                    buf.push(1); // has_value = true
                    buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                    buf.extend_from_slice(v);
                } else {
                    buf.push(0); // has_value = false
                }
            }
            WalRecord::Remove { term } => {
                buf.extend_from_slice(&(term.len() as u32).to_le_bytes());
                buf.extend_from_slice(term);
            }
            WalRecord::Checkpoint {
                checkpoint_lsn,
                timestamp,
            } => {
                buf.extend_from_slice(&checkpoint_lsn.to_le_bytes());
                buf.extend_from_slice(&timestamp.to_le_bytes());
            }
            WalRecord::BeginTx { tx_id }
            | WalRecord::CommitTx { tx_id }
            | WalRecord::AbortTx { tx_id } => {
                buf.extend_from_slice(&tx_id.to_le_bytes());
            }
            WalRecord::Increment {
                term,
                delta,
                result,
            } => {
                // Term length (4 bytes) + term + delta (8 bytes) + result (8 bytes)
                buf.extend_from_slice(&(term.len() as u32).to_le_bytes());
                buf.extend_from_slice(term);
                buf.extend_from_slice(&delta.to_le_bytes());
                buf.extend_from_slice(&result.to_le_bytes());
            }
            WalRecord::Upsert { term, value } => {
                // Term length (4 bytes) + term + value length (4 bytes) + value
                buf.extend_from_slice(&(term.len() as u32).to_le_bytes());
                buf.extend_from_slice(term);
                buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
                buf.extend_from_slice(value);
            }
            WalRecord::CompareAndSwap {
                term,
                expected,
                new_value,
                success,
            } => {
                // Term length + term + has_expected (1 byte) + [expected_len + expected] + new_value_len + new_value + success (1 byte)
                buf.extend_from_slice(&(term.len() as u32).to_le_bytes());
                buf.extend_from_slice(term);
                if let Some(exp) = expected {
                    buf.push(1); // has_expected = true
                    buf.extend_from_slice(&(exp.len() as u32).to_le_bytes());
                    buf.extend_from_slice(exp);
                } else {
                    buf.push(0); // has_expected = false
                }
                buf.extend_from_slice(&(new_value.len() as u32).to_le_bytes());
                buf.extend_from_slice(new_value);
                buf.push(if *success { 1 } else { 0 });
            }
        }

        buf
    }

    /// Deserialize a record from type and payload.
    pub fn deserialize(record_type: WalRecordType, payload: &[u8]) -> Result<Self, WalError> {
        match record_type {
            WalRecordType::Insert => {
                if payload.len() < 5 {
                    return Err(WalError::CorruptedRecord("Insert payload too short".into()));
                }
                let term_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                if payload.len() < 4 + term_len + 1 {
                    return Err(WalError::CorruptedRecord("Insert term truncated".into()));
                }
                let term = payload[4..4 + term_len].to_vec();
                let has_value = payload[4 + term_len] != 0;
                let value = if has_value {
                    let value_offset = 4 + term_len + 1;
                    if payload.len() < value_offset + 4 {
                        return Err(WalError::CorruptedRecord("Insert value length truncated".into()));
                    }
                    let value_len = u32::from_le_bytes(
                        payload[value_offset..value_offset + 4].try_into().unwrap(),
                    ) as usize;
                    if payload.len() < value_offset + 4 + value_len {
                        return Err(WalError::CorruptedRecord("Insert value truncated".into()));
                    }
                    Some(payload[value_offset + 4..value_offset + 4 + value_len].to_vec())
                } else {
                    None
                };
                Ok(WalRecord::Insert { term, value })
            }
            WalRecordType::Remove => {
                if payload.len() < 4 {
                    return Err(WalError::CorruptedRecord("Remove payload too short".into()));
                }
                let term_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                if payload.len() < 4 + term_len {
                    return Err(WalError::CorruptedRecord("Remove term truncated".into()));
                }
                let term = payload[4..4 + term_len].to_vec();
                Ok(WalRecord::Remove { term })
            }
            WalRecordType::Checkpoint => {
                if payload.len() < 16 {
                    return Err(WalError::CorruptedRecord("Checkpoint payload too short".into()));
                }
                let checkpoint_lsn = u64::from_le_bytes(payload[0..8].try_into().unwrap());
                let timestamp = u64::from_le_bytes(payload[8..16].try_into().unwrap());
                Ok(WalRecord::Checkpoint {
                    checkpoint_lsn,
                    timestamp,
                })
            }
            WalRecordType::BeginTx => {
                if payload.len() < 8 {
                    return Err(WalError::CorruptedRecord("BeginTx payload too short".into()));
                }
                let tx_id = u64::from_le_bytes(payload[0..8].try_into().unwrap());
                Ok(WalRecord::BeginTx { tx_id })
            }
            WalRecordType::CommitTx => {
                if payload.len() < 8 {
                    return Err(WalError::CorruptedRecord("CommitTx payload too short".into()));
                }
                let tx_id = u64::from_le_bytes(payload[0..8].try_into().unwrap());
                Ok(WalRecord::CommitTx { tx_id })
            }
            WalRecordType::AbortTx => {
                if payload.len() < 8 {
                    return Err(WalError::CorruptedRecord("AbortTx payload too short".into()));
                }
                let tx_id = u64::from_le_bytes(payload[0..8].try_into().unwrap());
                Ok(WalRecord::AbortTx { tx_id })
            }
            WalRecordType::Increment => {
                // term_len (4) + term + delta (8) + result (8)
                if payload.len() < 4 {
                    return Err(WalError::CorruptedRecord("Increment payload too short".into()));
                }
                let term_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                if payload.len() < 4 + term_len + 16 {
                    return Err(WalError::CorruptedRecord("Increment payload truncated".into()));
                }
                let term = payload[4..4 + term_len].to_vec();
                let delta_offset = 4 + term_len;
                let delta = i64::from_le_bytes(
                    payload[delta_offset..delta_offset + 8].try_into().unwrap(),
                );
                let result = i64::from_le_bytes(
                    payload[delta_offset + 8..delta_offset + 16]
                        .try_into()
                        .unwrap(),
                );
                Ok(WalRecord::Increment {
                    term,
                    delta,
                    result,
                })
            }
            WalRecordType::Upsert => {
                // term_len (4) + term + value_len (4) + value
                if payload.len() < 4 {
                    return Err(WalError::CorruptedRecord("Upsert payload too short".into()));
                }
                let term_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                if payload.len() < 4 + term_len + 4 {
                    return Err(WalError::CorruptedRecord("Upsert term truncated".into()));
                }
                let term = payload[4..4 + term_len].to_vec();
                let value_len_offset = 4 + term_len;
                let value_len = u32::from_le_bytes(
                    payload[value_len_offset..value_len_offset + 4]
                        .try_into()
                        .unwrap(),
                ) as usize;
                if payload.len() < value_len_offset + 4 + value_len {
                    return Err(WalError::CorruptedRecord("Upsert value truncated".into()));
                }
                let value = payload[value_len_offset + 4..value_len_offset + 4 + value_len].to_vec();
                Ok(WalRecord::Upsert { term, value })
            }
            WalRecordType::CompareAndSwap => {
                // term_len + term + has_expected (1) + [expected_len + expected] + new_value_len + new_value + success (1)
                if payload.len() < 4 {
                    return Err(WalError::CorruptedRecord("CAS payload too short".into()));
                }
                let term_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                if payload.len() < 4 + term_len + 1 {
                    return Err(WalError::CorruptedRecord("CAS term truncated".into()));
                }
                let term = payload[4..4 + term_len].to_vec();
                let mut offset = 4 + term_len;

                let has_expected = payload[offset] != 0;
                offset += 1;

                let expected = if has_expected {
                    if payload.len() < offset + 4 {
                        return Err(WalError::CorruptedRecord("CAS expected length truncated".into()));
                    }
                    let exp_len = u32::from_le_bytes(
                        payload[offset..offset + 4].try_into().unwrap(),
                    ) as usize;
                    offset += 4;
                    if payload.len() < offset + exp_len {
                        return Err(WalError::CorruptedRecord("CAS expected truncated".into()));
                    }
                    let exp = payload[offset..offset + exp_len].to_vec();
                    offset += exp_len;
                    Some(exp)
                } else {
                    None
                };

                if payload.len() < offset + 4 {
                    return Err(WalError::CorruptedRecord("CAS new_value length truncated".into()));
                }
                let new_value_len = u32::from_le_bytes(
                    payload[offset..offset + 4].try_into().unwrap(),
                ) as usize;
                offset += 4;
                if payload.len() < offset + new_value_len + 1 {
                    return Err(WalError::CorruptedRecord("CAS new_value truncated".into()));
                }
                let new_value = payload[offset..offset + new_value_len].to_vec();
                offset += new_value_len;

                let success = payload[offset] != 0;

                Ok(WalRecord::CompareAndSwap {
                    term,
                    expected,
                    new_value,
                    success,
                })
            }
        }
    }
}

/// WAL error types.
#[derive(Debug)]
pub enum WalError {
    /// I/O error
    Io(io::Error),
    /// Invalid record type byte
    InvalidRecordType(u8),
    /// Corrupted record (CRC mismatch or invalid format)
    CorruptedRecord(String),
    /// Unexpected end of file
    UnexpectedEof,
    /// WAL file already exists
    AlreadyExists,
    /// WAL file not found
    NotFound,
}

impl std::fmt::Display for WalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WalError::Io(e) => write!(f, "WAL I/O error: {}", e),
            WalError::InvalidRecordType(t) => write!(f, "Invalid WAL record type: {}", t),
            WalError::CorruptedRecord(msg) => write!(f, "Corrupted WAL record: {}", msg),
            WalError::UnexpectedEof => write!(f, "Unexpected end of WAL file"),
            WalError::AlreadyExists => write!(f, "WAL file already exists"),
            WalError::NotFound => write!(f, "WAL file not found"),
        }
    }
}

impl std::error::Error for WalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WalError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for WalError {
    fn from(err: io::Error) -> Self {
        WalError::Io(err)
    }
}

/// WAL file header.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WalHeader {
    /// Magic number: "PARTWAL\0"
    pub magic: [u8; 8],
    /// Version number
    pub version: u32,
    /// Last checkpoint LSN
    pub checkpoint_lsn: Lsn,
    /// Reserved for future use
    pub reserved: [u8; 44],
}

impl WalHeader {
    /// Magic number for WAL files.
    pub const MAGIC: [u8; 8] = *b"PARTWAL\0";
    /// Current version.
    pub const VERSION: u32 = 1;
    /// Header size in bytes.
    pub const SIZE: usize = 64;

    /// Create a new header.
    pub fn new() -> Self {
        WalHeader {
            magic: Self::MAGIC,
            version: Self::VERSION,
            checkpoint_lsn: 0,
            reserved: [0; 44],
        }
    }

    /// Serialize header to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..20].copy_from_slice(&self.checkpoint_lsn.to_le_bytes());
        buf[20..64].copy_from_slice(&self.reserved);
        buf
    }

    /// Deserialize header from bytes.
    pub fn from_bytes(buf: &[u8; Self::SIZE]) -> Result<Self, WalError> {
        let magic: [u8; 8] = buf[0..8].try_into().unwrap();
        if magic != Self::MAGIC {
            return Err(WalError::CorruptedRecord("Invalid WAL magic number".into()));
        }
        let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        if version != Self::VERSION {
            return Err(WalError::CorruptedRecord(format!(
                "Unsupported WAL version: {}",
                version
            )));
        }
        let checkpoint_lsn = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        let reserved: [u8; 44] = buf[20..64].try_into().unwrap();

        Ok(WalHeader {
            magic,
            version,
            checkpoint_lsn,
            reserved,
        })
    }
}

impl Default for WalHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Write-Ahead Log writer.
///
/// Handles appending records to the log with optional group commit.
pub struct WalWriter {
    /// Path to the WAL file
    path: PathBuf,
    /// File handle
    file: Mutex<BufWriter<File>>,
    /// Current LSN (next LSN to assign)
    next_lsn: AtomicU64,
    /// Last synced LSN
    synced_lsn: AtomicU64,
    /// Header (cached)
    header: Mutex<WalHeader>,
}

impl WalWriter {
    /// Record header size: CRC32 (4) + Length (4) + LSN (8) + Type (1) = 17 bytes
    const RECORD_HEADER_SIZE: usize = 17;

    /// Create a new WAL file.
    pub fn create(path: impl AsRef<Path>) -> Result<Self, WalError> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            return Err(WalError::AlreadyExists);
        }

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&path)?;

        let mut writer = BufWriter::new(file);

        // Write header
        let header = WalHeader::new();
        writer.write_all(&header.to_bytes())?;
        writer.flush()?;

        Ok(WalWriter {
            path,
            file: Mutex::new(writer),
            next_lsn: AtomicU64::new(1), // LSN 0 reserved for "no LSN"
            synced_lsn: AtomicU64::new(0),
            header: Mutex::new(header),
        })
    }

    /// Open an existing WAL file for appending.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WalError> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(WalError::NotFound);
        }

        let file = OpenOptions::new().read(true).write(true).open(&path)?;

        // Read header
        let mut reader = BufReader::new(&file);
        let mut header_buf = [0u8; WalHeader::SIZE];
        reader.read_exact(&mut header_buf)?;
        let header = WalHeader::from_bytes(&header_buf)?;

        // Find the last LSN by scanning the log
        let mut last_lsn: Lsn = 0;
        let mut reader = WalReader::new(path.clone())?;
        while let Some(result) = reader.next_record() {
            if let Ok((lsn, _)) = result {
                last_lsn = lsn;
            }
        }

        // Seek to end for appending
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mut writer = BufWriter::new(file);
        writer.seek(SeekFrom::End(0))?;

        Ok(WalWriter {
            path,
            file: Mutex::new(writer),
            next_lsn: AtomicU64::new(last_lsn + 1),
            synced_lsn: AtomicU64::new(last_lsn),
            header: Mutex::new(header),
        })
    }

    /// Append a record to the WAL.
    ///
    /// Returns the LSN assigned to the record.
    pub fn append(&self, record: WalRecord) -> Result<Lsn, WalError> {
        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);
        let payload = record.serialize_payload();
        let record_type = record.record_type() as u8;

        // Build the record buffer
        let total_len = Self::RECORD_HEADER_SIZE + payload.len();
        let mut buf = Vec::with_capacity(total_len);

        // Placeholder for CRC (will be computed after)
        buf.extend_from_slice(&[0u8; 4]);
        // Length (including header)
        buf.extend_from_slice(&(total_len as u32).to_le_bytes());
        // LSN
        buf.extend_from_slice(&lsn.to_le_bytes());
        // Type
        buf.push(record_type);
        // Payload
        buf.extend_from_slice(&payload);

        // Compute CRC over everything except the CRC field itself
        let crc = crc32(&buf[4..]);
        buf[0..4].copy_from_slice(&crc.to_le_bytes());

        // Write to file
        let mut file = self.file.lock().expect("WAL lock poisoned");
        file.write_all(&buf)?;

        Ok(lsn)
    }

    /// Sync (fsync) the WAL to disk.
    ///
    /// Returns the highest LSN that is now durable.
    pub fn sync(&self) -> Result<Lsn, WalError> {
        let mut file = self.file.lock().expect("WAL lock poisoned");
        file.flush()?;
        file.get_ref().sync_all()?;

        let current_lsn = self.next_lsn.load(Ordering::SeqCst) - 1;
        self.synced_lsn.store(current_lsn, Ordering::SeqCst);

        Ok(current_lsn)
    }

    /// Get the current (next) LSN.
    pub fn current_lsn(&self) -> Lsn {
        self.next_lsn.load(Ordering::SeqCst)
    }

    /// Get the last synced LSN.
    pub fn synced_lsn(&self) -> Lsn {
        self.synced_lsn.load(Ordering::SeqCst)
    }

    /// Get the path to the WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write a checkpoint record and update the header.
    pub fn checkpoint(&self, checkpoint_lsn: Lsn) -> Result<Lsn, WalError> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let record = WalRecord::Checkpoint {
            checkpoint_lsn,
            timestamp,
        };

        let lsn = self.append(record)?;
        self.sync()?;

        // Update header
        let mut header = self.header.lock().expect("header lock poisoned");
        header.checkpoint_lsn = checkpoint_lsn;

        // Write updated header
        let mut file = self.file.lock().expect("WAL lock poisoned");
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header.to_bytes())?;
        file.flush()?;
        file.get_ref().sync_all()?;

        // Seek back to end
        file.seek(SeekFrom::End(0))?;

        Ok(lsn)
    }

    /// Get the last checkpoint LSN.
    pub fn checkpoint_lsn(&self) -> Lsn {
        let header = self.header.lock().expect("header lock poisoned");
        header.checkpoint_lsn
    }

    /// Truncate the WAL file, removing all records.
    ///
    /// This is typically called after successful recovery to prevent
    /// re-replaying the same operations on subsequent opens. The header
    /// is preserved but all records are removed.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or a `WalError` on failure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // After recovery completes successfully
    /// wal_writer.truncate()?;
    /// ```
    pub fn truncate(&self) -> Result<(), WalError> {
        let mut file = self.file.lock().expect("WAL lock poisoned");

        // Flush any pending writes
        file.flush()?;

        // Get the underlying file and truncate to header size only
        let inner_file = file.get_mut();
        inner_file.set_len(WalHeader::SIZE as u64)?;

        // Seek to end of header for subsequent writes
        file.seek(SeekFrom::Start(WalHeader::SIZE as u64))?;

        // Reset LSN counters
        self.next_lsn.store(1, Ordering::SeqCst);
        self.synced_lsn.store(0, Ordering::SeqCst);

        // Reset checkpoint LSN in header
        {
            let mut header = self.header.lock().expect("header lock poisoned");
            header.checkpoint_lsn = 0;

            // Write updated header to disk
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header.to_bytes())?;
            file.flush()?;
            file.get_ref().sync_all()?;

            // Seek back to end of header for subsequent writes
            file.seek(SeekFrom::Start(WalHeader::SIZE as u64))?;
        }

        Ok(())
    }
}

/// WAL reader for recovery.
pub struct WalReader {
    reader: BufReader<File>,
    #[allow(dead_code)]
    path: PathBuf,
}

impl WalReader {
    /// Open a WAL file for reading.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, WalError> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);

        // Skip header
        reader.seek(SeekFrom::Start(WalHeader::SIZE as u64))?;

        Ok(WalReader { reader, path })
    }

    /// Read the next record from the WAL.
    ///
    /// Returns `None` at end of file, `Some(Err(...))` on error.
    pub fn next_record(&mut self) -> Option<Result<(Lsn, WalRecord), WalError>> {
        // Read header: CRC (4) + Length (4) + LSN (8) + Type (1)
        let mut header_buf = [0u8; WalWriter::RECORD_HEADER_SIZE];
        match self.reader.read_exact(&mut header_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(WalError::Io(e))),
        }

        let stored_crc = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
        let length = u32::from_le_bytes(header_buf[4..8].try_into().unwrap()) as usize;
        let lsn = u64::from_le_bytes(header_buf[8..16].try_into().unwrap());
        let record_type_byte = header_buf[16];

        // Validate length
        if length < WalWriter::RECORD_HEADER_SIZE {
            return Some(Err(WalError::CorruptedRecord(
                "Record length too small".into(),
            )));
        }

        let payload_len = length - WalWriter::RECORD_HEADER_SIZE;

        // Read payload
        let mut payload = vec![0u8; payload_len];
        if !payload.is_empty() {
            match self.reader.read_exact(&mut payload) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    return Some(Err(WalError::UnexpectedEof))
                }
                Err(e) => return Some(Err(WalError::Io(e))),
            }
        }

        // Verify CRC
        let mut crc_data = Vec::with_capacity(length - 4);
        crc_data.extend_from_slice(&header_buf[4..]);
        crc_data.extend_from_slice(&payload);
        let computed_crc = crc32(&crc_data);

        if stored_crc != computed_crc {
            return Some(Err(WalError::CorruptedRecord(format!(
                "CRC mismatch: stored={:#x}, computed={:#x}",
                stored_crc, computed_crc
            ))));
        }

        // Parse record type
        let record_type = match WalRecordType::try_from(record_type_byte) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        // Deserialize record
        match WalRecord::deserialize(record_type, &payload) {
            Ok(record) => Some(Ok((lsn, record))),
            Err(e) => Some(Err(e)),
        }
    }

    /// Get an iterator over all records.
    pub fn iter(self) -> WalRecordIterator {
        WalRecordIterator { reader: self }
    }

    /// Read the header from the WAL file.
    pub fn read_header(path: impl AsRef<Path>) -> Result<WalHeader, WalError> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);
        let mut header_buf = [0u8; WalHeader::SIZE];
        reader.read_exact(&mut header_buf)?;
        WalHeader::from_bytes(&header_buf)
    }
}

/// Iterator over WAL records.
pub struct WalRecordIterator {
    reader: WalReader,
}

impl Iterator for WalRecordIterator {
    type Item = Result<(Lsn, WalRecord), WalError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.next_record()
    }
}

/// Group commit coordinator.
///
/// Batches multiple WAL writes into a single fsync for better performance.
pub struct GroupCommit {
    wal: Arc<WalWriter>,
    /// Pending LSNs waiting for sync
    pending: Mutex<Vec<(Lsn, std::sync::mpsc::Sender<Result<(), WalError>>)>>,
    /// Sync interval in milliseconds
    #[allow(dead_code)]
    sync_interval_ms: u64,
}

impl GroupCommit {
    /// Create a new group commit coordinator.
    pub fn new(wal: Arc<WalWriter>, sync_interval_ms: u64) -> Self {
        GroupCommit {
            wal,
            pending: Mutex::new(Vec::new()),
            sync_interval_ms,
        }
    }

    /// Append a record and wait for it to be durable.
    pub fn append_sync(&self, record: WalRecord) -> Result<Lsn, WalError> {
        let lsn = self.wal.append(record)?;

        // For simplicity, sync immediately
        // In production, we'd batch and use the sync_interval
        self.wal.sync()?;

        Ok(lsn)
    }

    /// Get the underlying WAL writer.
    pub fn wal(&self) -> &WalWriter {
        &self.wal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_crc32() {
        let data = b"hello world";
        let crc = crc32(data);
        assert_eq!(crc, 0x0D4A1185); // Known CRC32 value
    }

    #[test]
    fn test_wal_record_serialize_deserialize() {
        let record = WalRecord::Insert {
            term: b"hello".to_vec(),
            value: Some(b"world".to_vec()),
        };
        let payload = record.serialize_payload();
        let deserialized =
            WalRecord::deserialize(WalRecordType::Insert, &payload).expect("deserialize failed");

        assert_eq!(record, deserialized);
    }

    #[test]
    fn test_wal_record_remove() {
        let record = WalRecord::Remove {
            term: b"goodbye".to_vec(),
        };
        let payload = record.serialize_payload();
        let deserialized =
            WalRecord::deserialize(WalRecordType::Remove, &payload).expect("deserialize failed");

        assert_eq!(record, deserialized);
    }

    #[test]
    fn test_wal_create_and_append() {
        let dir = tempdir().expect("create temp dir");
        let wal_path = dir.path().join("test.wal");

        let wal = WalWriter::create(&wal_path).expect("create WAL");

        let lsn1 = wal
            .append(WalRecord::Insert {
                term: b"hello".to_vec(),
                value: None,
            })
            .expect("append");

        let lsn2 = wal
            .append(WalRecord::Insert {
                term: b"world".to_vec(),
                value: Some(b"value".to_vec()),
            })
            .expect("append");

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);

        wal.sync().expect("sync");
    }

    #[test]
    fn test_wal_read_records() {
        let dir = tempdir().expect("create temp dir");
        let wal_path = dir.path().join("test.wal");

        // Write records
        {
            let wal = WalWriter::create(&wal_path).expect("create WAL");
            wal.append(WalRecord::Insert {
                term: b"hello".to_vec(),
                value: None,
            })
            .expect("append");
            wal.append(WalRecord::Remove {
                term: b"world".to_vec(),
            })
            .expect("append");
            wal.sync().expect("sync");
        }

        // Read records
        let reader = WalReader::new(&wal_path).expect("open WAL");
        let records: Vec<_> = reader.iter().collect();

        assert_eq!(records.len(), 2);

        let (lsn1, rec1) = records[0].as_ref().expect("record 1");
        assert_eq!(*lsn1, 1);
        assert!(matches!(rec1, WalRecord::Insert { term, .. } if term == b"hello"));

        let (lsn2, rec2) = records[1].as_ref().expect("record 2");
        assert_eq!(*lsn2, 2);
        assert!(matches!(rec2, WalRecord::Remove { term } if term == b"world"));
    }

    #[test]
    fn test_wal_checkpoint() {
        let dir = tempdir().expect("create temp dir");
        let wal_path = dir.path().join("test.wal");

        {
            let wal = WalWriter::create(&wal_path).expect("create WAL");
            wal.append(WalRecord::Insert {
                term: b"test".to_vec(),
                value: None,
            })
            .expect("append");
            wal.checkpoint(1).expect("checkpoint");
        }

        // Verify checkpoint LSN is persisted
        let header = WalReader::read_header(&wal_path).expect("read header");
        assert_eq!(header.checkpoint_lsn, 1);
    }

    #[test]
    fn test_wal_reopen() {
        let dir = tempdir().expect("create temp dir");
        let wal_path = dir.path().join("test.wal");

        // Create and write
        {
            let wal = WalWriter::create(&wal_path).expect("create WAL");
            wal.append(WalRecord::Insert {
                term: b"first".to_vec(),
                value: None,
            })
            .expect("append");
            wal.sync().expect("sync");
        }

        // Reopen and append more
        {
            let wal = WalWriter::open(&wal_path).expect("open WAL");
            assert_eq!(wal.current_lsn(), 2); // Next LSN should be 2
            wal.append(WalRecord::Insert {
                term: b"second".to_vec(),
                value: None,
            })
            .expect("append");
            wal.sync().expect("sync");
        }

        // Verify all records
        let reader = WalReader::new(&wal_path).expect("open WAL");
        let records: Vec<_> = reader.iter().collect();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_wal_truncate() {
        let dir = tempdir().expect("create temp dir");
        let wal_path = dir.path().join("test.wal");

        // Create WAL and write some records
        {
            let wal = WalWriter::create(&wal_path).expect("create WAL");
            wal.append(WalRecord::Insert {
                term: b"first".to_vec(),
                value: None,
            })
            .expect("append");
            wal.append(WalRecord::Insert {
                term: b"second".to_vec(),
                value: None,
            })
            .expect("append");
            wal.checkpoint(2).expect("checkpoint");
            wal.sync().expect("sync");

            // Verify records exist before truncate
            assert_eq!(wal.current_lsn(), 4); // 2 inserts + 1 checkpoint = LSN 3, next is 4

            // Truncate the WAL
            wal.truncate().expect("truncate");

            // Verify LSN is reset
            assert_eq!(wal.current_lsn(), 1);
            assert_eq!(wal.synced_lsn(), 0);
            assert_eq!(wal.checkpoint_lsn(), 0);
        }

        // Verify WAL is empty after truncate
        let reader = WalReader::new(&wal_path).expect("open WAL");
        let records: Vec<_> = reader.iter().collect();
        assert_eq!(records.len(), 0, "WAL should be empty after truncate");

        // Verify we can append new records after truncate
        {
            let wal = WalWriter::open(&wal_path).expect("open WAL");
            assert_eq!(wal.current_lsn(), 1); // Should start fresh

            let lsn = wal
                .append(WalRecord::Insert {
                    term: b"new_record".to_vec(),
                    value: None,
                })
                .expect("append after truncate");
            assert_eq!(lsn, 1);
            wal.sync().expect("sync");
        }

        // Verify new record is readable
        let reader = WalReader::new(&wal_path).expect("open WAL");
        let records: Vec<_> = reader.iter().collect();
        assert_eq!(records.len(), 1);
        let (lsn, rec) = records[0].as_ref().expect("record");
        assert_eq!(*lsn, 1);
        assert!(matches!(rec, WalRecord::Insert { term, .. } if term == b"new_record"));
    }
}
