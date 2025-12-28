//! Error types for Persistent Adaptive Radix Trie
//!
//! This module defines all error types that can occur during persistent
//! dictionary operations, including:
//!
//! - I/O errors (file operations, memory mapping)
//! - Format errors (invalid magic, unsupported version)
//! - Corruption errors (checksum mismatch, invalid structure)
//! - Concurrency errors (lock poisoning, swizzle failures)
//! - Resource errors (out of space, buffer pool exhausted)

use std::io;
use thiserror::Error;

/// Result type alias for persistent ARTrie operations
pub type Result<T> = std::result::Result<T, PersistentARTrieError>;

/// Errors that can occur during persistent ARTrie operations
#[derive(Error, Debug)]
pub enum PersistentARTrieError {
    /// I/O error during file operations
    #[error("I/O error during {operation} on '{path}': {source}")]
    IoError {
        /// What operation was being performed
        operation: String,
        /// Path to the file (if applicable)
        path: String,
        /// Underlying I/O error
        #[source]
        source: io::Error,
    },

    /// Memory mapping error
    #[error("Memory map error during {operation}: {source}")]
    MmapError {
        /// What operation was being performed
        operation: String,
        /// Underlying I/O error
        #[source]
        source: io::Error,
    },

    /// Invalid magic number in file header
    #[error("Invalid magic number: expected 0x{expected:016X}, found 0x{found:016X}")]
    InvalidMagic {
        /// Expected magic number
        expected: u64,
        /// Actual magic number found
        found: u64,
    },

    /// Unsupported file format version
    #[error("Unsupported file version: max supported {max_supported}, found {found}")]
    UnsupportedVersion {
        /// Maximum version we can read
        max_supported: u32,
        /// Version found in file
        found: u32,
    },

    /// File corruption detected
    #[error("Corrupted file: {reason}")]
    CorruptedFile {
        /// Description of corruption
        reason: String,
    },

    /// Checksum verification failed
    #[error("Checksum mismatch in block {block_id}: expected 0x{expected:016X}, found 0x{found:016X}")]
    ChecksumMismatch {
        /// Block that failed verification
        block_id: u32,
        /// Expected checksum
        expected: u64,
        /// Actual checksum found
        found: u64,
    },

    /// Invalid block ID
    #[error("Invalid block ID {block_id}: {reason}")]
    InvalidBlockId {
        /// The invalid block ID
        block_id: u32,
        /// Why it's invalid
        reason: String,
    },

    /// Out of disk space or block count limit reached
    #[error("Out of space: {current_blocks}/{max_blocks} blocks used")]
    OutOfSpace {
        /// Current number of blocks
        current_blocks: u32,
        /// Maximum number of blocks
        max_blocks: u32,
    },

    /// Buffer pool exhausted (all pages pinned)
    #[error("Buffer pool exhausted: {pinned_pages}/{total_pages} pages pinned")]
    BufferPoolExhausted {
        /// Number of pinned pages
        pinned_pages: usize,
        /// Total pages in pool
        total_pages: usize,
    },

    /// Lock was poisoned (panic occurred while holding lock)
    #[error("Lock poisoned for {resource}")]
    LockPoisoned {
        /// What resource the lock protected
        resource: String,
    },

    /// Swizzle operation failed
    #[error("Swizzle error: {0}")]
    SwizzleError(#[from] SwizzleError),

    /// Node type mismatch
    #[error("Node type mismatch: expected {expected}, found {found}")]
    NodeTypeMismatch {
        /// Expected node type
        expected: String,
        /// Actual node type found
        found: String,
    },

    /// Key too long
    #[error("Key too long: {length} bytes (max {max_length})")]
    KeyTooLong {
        /// Actual key length
        length: usize,
        /// Maximum allowed length
        max_length: usize,
    },

    /// Value too large
    #[error("Value too large: {size} bytes (max {max_size})")]
    ValueTooLarge {
        /// Actual value size
        size: usize,
        /// Maximum allowed size
        max_size: usize,
    },

    /// Bucket overflow (too many entries for bucket)
    #[error("Bucket overflow in block {block_id}: {entries} entries")]
    BucketOverflow {
        /// Block containing the bucket
        block_id: u32,
        /// Number of entries
        entries: usize,
    },

    /// WAL (Write-Ahead Log) error
    #[error("WAL error: {reason}")]
    WalError {
        /// Description of WAL error
        reason: String,
    },

    /// Recovery error during startup
    #[error("Recovery error: {reason}")]
    RecoveryError {
        /// Description of recovery error
        reason: String,
    },

    /// Operation not supported in read-only mode
    #[error("Operation not supported in read-only mode: {operation}")]
    ReadOnlyMode {
        /// The operation that was attempted
        operation: String,
    },

    /// Concurrent modification detected (for optimistic locking)
    #[error("Concurrent modification detected on block {block_id}")]
    ConcurrentModification {
        /// Block that was modified
        block_id: u32,
    },

    /// Path compression prefix mismatch
    #[error("Prefix mismatch at depth {depth}: expected '{expected}', found '{found}'")]
    PrefixMismatch {
        /// Depth in the trie where mismatch occurred
        depth: usize,
        /// Expected prefix (hex encoded)
        expected: String,
        /// Found prefix (hex encoded)
        found: String,
    },

    /// Internal error (should not happen, indicates bug)
    #[error("Internal error: {message}")]
    InternalError {
        /// Description of the error
        message: String,
    },
}

/// Errors specific to pointer swizzling operations
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwizzleError {
    /// Attempted to swizzle a pointer that's already swizzled (in memory)
    #[error("Pointer is already swizzled (in memory)")]
    AlreadySwizzled,

    /// Attempted to unswizzle a pointer that's not swizzled (still on disk)
    #[error("Pointer is not swizzled (on disk)")]
    AlreadyUnswizzled,

    /// Concurrent swizzle/unswizzle operation modified the pointer
    #[error("Concurrent modification detected (race condition)")]
    RaceCondition,

    /// Null pointer encountered (cannot swizzle null)
    #[error("Cannot swizzle null pointer")]
    NullPointer,

    /// Block ID exceeds maximum allowed value
    #[error("Block ID overflow: {block_id} exceeds maximum")]
    BlockIdOverflow {
        /// The block ID that was too large
        block_id: u32,
    },

    /// Offset exceeds maximum allowed value
    #[error("Offset overflow: {offset} exceeds maximum")]
    OffsetOverflow {
        /// The offset that was too large
        offset: u32,
    },

    /// Invalid node type encoding
    #[error("Invalid node type: {value}")]
    InvalidNodeType {
        /// Raw node type value
        value: u8,
    },
}

impl PersistentARTrieError {
    /// Create an I/O error with context
    pub fn io_error(operation: impl Into<String>, path: impl Into<String>, source: io::Error) -> Self {
        Self::IoError {
            operation: operation.into(),
            path: path.into(),
            source,
        }
    }

    /// Create a corruption error
    pub fn corrupted(reason: impl Into<String>) -> Self {
        Self::CorruptedFile {
            reason: reason.into(),
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
        }
    }

    /// Check if this error is recoverable (can retry operation)
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::ConcurrentModification { .. }
                | Self::SwizzleError(SwizzleError::RaceCondition)
        )
    }

    /// Check if this error indicates corruption (file should be repaired/rebuilt)
    pub fn is_corruption(&self) -> bool {
        matches!(
            self,
            Self::CorruptedFile { .. }
                | Self::ChecksumMismatch { .. }
                | Self::InvalidMagic { .. }
        )
    }

    /// Check if this error is transient (e.g., out of buffer space)
    pub fn is_transient(&self) -> bool {
        matches!(self, Self::BufferPoolExhausted { .. })
    }
}

#[cfg(feature = "persistent-artrie")]
impl From<super::wal::WalError> for PersistentARTrieError {
    fn from(err: super::wal::WalError) -> Self {
        use super::wal::WalError;
        match err {
            WalError::Io(e) => Self::IoError {
                operation: "WAL operation".to_string(),
                path: String::new(),
                source: e,
            },
            WalError::InvalidRecordType(t) => Self::CorruptedFile {
                reason: format!("Invalid WAL record type: {}", t),
            },
            WalError::CorruptedRecord(msg) => Self::CorruptedFile {
                reason: format!("Corrupted WAL record: {}", msg),
            },
            WalError::UnexpectedEof => Self::CorruptedFile {
                reason: "Unexpected end of WAL file".to_string(),
            },
            WalError::AlreadyExists => Self::InternalError {
                message: "WAL file already exists".to_string(),
            },
            WalError::NotFound => Self::IoError {
                operation: "WAL open".to_string(),
                path: String::new(),
                source: io::Error::new(io::ErrorKind::NotFound, "WAL file not found"),
            },
        }
    }
}

#[cfg(feature = "persistent-artrie")]
impl From<super::recovery::RecoveryError> for PersistentARTrieError {
    fn from(err: super::recovery::RecoveryError) -> Self {
        Self::InternalError {
            message: format!("Recovery error: {}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = PersistentARTrieError::InvalidMagic {
            expected: 0x5041525400010000,
            found: 0x0000000000000000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid magic"));
        assert!(msg.contains("5041525400010000"));
    }

    #[test]
    fn test_swizzle_error_display() {
        let err = SwizzleError::AlreadySwizzled;
        assert_eq!(format!("{}", err), "Pointer is already swizzled (in memory)");

        let err = SwizzleError::BlockIdOverflow { block_id: 42 };
        assert!(format!("{}", err).contains("42"));

        let err = SwizzleError::OffsetOverflow { offset: 1024 };
        assert!(format!("{}", err).contains("1024"));
    }

    #[test]
    fn test_is_recoverable() {
        let recoverable = PersistentARTrieError::ConcurrentModification { block_id: 1 };
        assert!(recoverable.is_recoverable());

        let not_recoverable = PersistentARTrieError::CorruptedFile {
            reason: "test".to_string(),
        };
        assert!(!not_recoverable.is_recoverable());
    }

    #[test]
    fn test_is_corruption() {
        let corruption = PersistentARTrieError::CorruptedFile {
            reason: "test".to_string(),
        };
        assert!(corruption.is_corruption());

        let checksum = PersistentARTrieError::ChecksumMismatch {
            block_id: 0,
            expected: 123,
            found: 456,
        };
        assert!(checksum.is_corruption());

        let not_corruption = PersistentARTrieError::OutOfSpace {
            current_blocks: 100,
            max_blocks: 100,
        };
        assert!(!not_corruption.is_corruption());
    }

    #[test]
    fn test_swizzle_error_conversion() {
        let swizzle_err = SwizzleError::NullPointer;
        let artrie_err: PersistentARTrieError = swizzle_err.into();

        match artrie_err {
            PersistentARTrieError::SwizzleError(SwizzleError::NullPointer) => {}
            _ => panic!("Expected SwizzleError variant"),
        }
    }

    #[test]
    fn test_io_error_helper() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err = PersistentARTrieError::io_error("open", "/path/to/file", io_err);

        match err {
            PersistentARTrieError::IoError {
                operation,
                path,
                source,
            } => {
                assert_eq!(operation, "open");
                assert_eq!(path, "/path/to/file");
                assert_eq!(source.kind(), io::ErrorKind::NotFound);
            }
            _ => panic!("Expected IoError variant"),
        }
    }
}
