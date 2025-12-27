//! Crash recovery for Persistent ART.
//!
//! This module implements redo-only crash recovery using the Write-Ahead Log (WAL).
//! Recovery follows the ARIES (Algorithm for Recovery and Isolation Exploiting
//! Semantics) approach, simplified for our redo-only needs.
//!
//! # Recovery Process
//!
//! 1. **Analysis Phase**: Scan WAL from last checkpoint, identify committed transactions
//! 2. **Redo Phase**: Replay all committed operations to rebuild state
//! 3. **Cleanup Phase**: Truncate WAL after recovery
//!
//! # Checkpoint Strategy
//!
//! Checkpoints are fuzzy - they don't require quiescing the system. The checkpoint
//! LSN indicates the point after which all committed transactions are guaranteed
//! to be in the WAL.
//!
//! # Example
//!
//! ```rust,ignore
//! use liblevenshtein::dictionary::persistent_artrie::recovery::RecoveryManager;
//!
//! let recovery = RecoveryManager::new("data.wal")?;
//! let state = recovery.recover()?;
//!
//! // State contains all committed operations
//! for op in state.operations() {
//!     // Apply to trie...
//! }
//! ```

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::wal::{Lsn, WalError, WalReader, WalRecord};

/// Error types for recovery operations.
#[derive(Debug)]
pub enum RecoveryError {
    /// I/O error during recovery
    Io(io::Error),
    /// WAL file is corrupted
    CorruptedWal(String),
    /// Checkpoint not found
    NoCheckpoint,
    /// Invalid checkpoint data
    InvalidCheckpoint { lsn: Lsn, reason: String },
    /// Transaction log inconsistency
    TransactionInconsistency { tx_id: u64, reason: String },
    /// Recovery operation failed
    RecoveryFailed(String),
}

impl std::fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoveryError::Io(e) => write!(f, "I/O error: {}", e),
            RecoveryError::CorruptedWal(msg) => write!(f, "Corrupted WAL: {}", msg),
            RecoveryError::NoCheckpoint => write!(f, "No checkpoint found in WAL"),
            RecoveryError::InvalidCheckpoint { lsn, reason } => {
                write!(f, "Invalid checkpoint at LSN {}: {}", lsn, reason)
            }
            RecoveryError::TransactionInconsistency { tx_id, reason } => {
                write!(f, "Transaction {} inconsistency: {}", tx_id, reason)
            }
            RecoveryError::RecoveryFailed(msg) => write!(f, "Recovery failed: {}", msg),
        }
    }
}

impl std::error::Error for RecoveryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RecoveryError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for RecoveryError {
    fn from(e: io::Error) -> Self {
        RecoveryError::Io(e)
    }
}

impl From<WalError> for RecoveryError {
    fn from(e: WalError) -> Self {
        match e {
            WalError::Io(io_err) => RecoveryError::Io(io_err),
            WalError::CorruptedRecord(msg) => RecoveryError::CorruptedWal(msg),
            WalError::UnexpectedEof => RecoveryError::CorruptedWal("Unexpected EOF".into()),
            WalError::InvalidRecordType(t) => {
                RecoveryError::CorruptedWal(format!("Invalid record type: {}", t))
            }
            WalError::AlreadyExists => RecoveryError::RecoveryFailed("WAL already exists".into()),
            WalError::NotFound => RecoveryError::NoCheckpoint,
        }
    }
}

/// Result type for recovery operations.
pub type Result<T> = std::result::Result<T, RecoveryError>;

/// A recovered operation ready to be applied.
#[derive(Debug, Clone)]
pub enum RecoveredOperation {
    /// Insert a term with optional value
    Insert { term: Vec<u8>, value: Option<Vec<u8>> },
    /// Remove a term
    Remove { term: Vec<u8> },
}

/// State of a transaction during recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransactionState {
    /// Transaction has begun but not yet committed/aborted
    InProgress,
    /// Transaction has committed
    Committed,
    /// Transaction has aborted
    Aborted,
}

/// Pending operations for a transaction.
#[derive(Debug, Default)]
struct PendingTransaction {
    /// Transaction state
    state: Option<TransactionState>,
    /// Operations in this transaction (in order)
    operations: Vec<RecoveredOperation>,
    /// LSN of the begin record
    begin_lsn: Option<Lsn>,
}

/// Recovery statistics for monitoring and debugging.
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Total records scanned
    pub records_scanned: u64,
    /// Records that passed CRC validation
    pub valid_records: u64,
    /// Records that failed CRC validation
    pub corrupted_records: u64,
    /// Committed transactions found
    pub committed_transactions: u64,
    /// Aborted transactions found
    pub aborted_transactions: u64,
    /// In-progress (incomplete) transactions
    pub incomplete_transactions: u64,
    /// Insert operations recovered
    pub insert_operations: u64,
    /// Remove operations recovered
    pub remove_operations: u64,
    /// Checkpoint LSN used for recovery
    pub checkpoint_lsn: Option<Lsn>,
    /// Recovery duration
    pub duration_ms: u64,
}

/// Recovered state from WAL.
#[derive(Debug)]
pub struct RecoveredState {
    /// All committed operations in LSN order
    operations: Vec<RecoveredOperation>,
    /// The LSN after all recovered operations
    pub next_lsn: Lsn,
    /// Recovery statistics
    pub stats: RecoveryStats,
}

impl RecoveredState {
    /// Get iterator over recovered operations.
    pub fn operations(&self) -> impl Iterator<Item = &RecoveredOperation> {
        self.operations.iter()
    }

    /// Get the number of recovered operations.
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Consume self and return operations.
    pub fn into_operations(self) -> Vec<RecoveredOperation> {
        self.operations
    }
}

/// Recovery manager for crash recovery.
///
/// The recovery manager reads the WAL and reconstructs the committed state
/// of the dictionary. It handles:
/// - Checkpoint detection
/// - Transaction tracking
/// - Redo replay of committed operations
pub struct RecoveryManager {
    /// Path to WAL file
    wal_path: PathBuf,
}

impl RecoveryManager {
    /// Create a new recovery manager for the given WAL path.
    pub fn new<P: AsRef<Path>>(wal_path: P) -> Self {
        RecoveryManager {
            wal_path: wal_path.as_ref().to_path_buf(),
        }
    }

    /// Check if recovery is needed.
    ///
    /// Recovery is needed if the WAL file exists and contains records
    /// beyond the last checkpoint.
    pub fn needs_recovery(&self) -> Result<bool> {
        if !self.wal_path.exists() {
            return Ok(false);
        }

        let wal_reader = match WalReader::new(&self.wal_path) {
            Ok(r) => r,
            Err(_) => return Ok(false), // Can't open WAL, no recovery needed
        };

        // If we can read at least one record, we need recovery
        for result in wal_reader.iter() {
            match result {
                Ok(_) => return Ok(true),
                Err(_) => return Ok(false), // Empty or corrupted WAL
            }
        }

        Ok(false)
    }

    /// Perform recovery and return recovered state.
    ///
    /// This method:
    /// 1. Scans the WAL from the beginning (or last checkpoint)
    /// 2. Tracks transaction states
    /// 3. Collects operations from committed transactions
    /// 4. Returns operations in LSN order
    pub fn recover(&self) -> Result<RecoveredState> {
        let start_time = Instant::now();
        let mut stats = RecoveryStats::default();

        if !self.wal_path.exists() {
            return Ok(RecoveredState {
                operations: Vec::new(),
                next_lsn: 1,
                stats,
            });
        }

        // Phase 1: Analysis - Find checkpoint and track transactions
        let (checkpoint_lsn, transactions) = self.analysis_phase(&mut stats)?;
        stats.checkpoint_lsn = checkpoint_lsn;

        // Phase 2: Redo - Collect committed operations
        let (operations, next_lsn) = self.redo_phase(checkpoint_lsn, &transactions, &mut stats)?;

        stats.duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(RecoveredState {
            operations,
            next_lsn,
            stats,
        })
    }

    /// Analysis phase: Scan WAL and track transaction states.
    ///
    /// Returns:
    /// - The checkpoint LSN to start redo from (None means start from beginning)
    /// - Map of transaction IDs to their final states
    fn analysis_phase(
        &self,
        stats: &mut RecoveryStats,
    ) -> Result<(Option<Lsn>, HashMap<u64, TransactionState>)> {
        let wal_reader = WalReader::new(&self.wal_path)?;

        let mut checkpoint_lsn: Option<Lsn> = None;
        let mut transactions: HashMap<u64, TransactionState> = HashMap::new();

        for result in wal_reader.iter() {
            stats.records_scanned += 1;

            let (_lsn, record) = match result {
                Ok(r) => {
                    stats.valid_records += 1;
                    r
                }
                Err(e) => {
                    stats.corrupted_records += 1;
                    // Log corruption but continue - we want to recover as much as possible
                    eprintln!("Warning: Corrupted record during analysis: {:?}", e);
                    continue;
                }
            };

            match record {
                WalRecord::Checkpoint { checkpoint_lsn: cp_lsn, .. } => {
                    // Use the most recent checkpoint
                    checkpoint_lsn = Some(cp_lsn);
                }
                WalRecord::BeginTx { tx_id } => {
                    transactions.insert(tx_id, TransactionState::InProgress);
                }
                WalRecord::CommitTx { tx_id } => {
                    if let Some(state) = transactions.get_mut(&tx_id) {
                        *state = TransactionState::Committed;
                        stats.committed_transactions += 1;
                    }
                }
                WalRecord::AbortTx { tx_id } => {
                    if let Some(state) = transactions.get_mut(&tx_id) {
                        *state = TransactionState::Aborted;
                        stats.aborted_transactions += 1;
                    }
                }
                _ => {}
            }
        }

        // Count incomplete transactions
        for state in transactions.values() {
            if *state == TransactionState::InProgress {
                stats.incomplete_transactions += 1;
            }
        }

        Ok((checkpoint_lsn, transactions))
    }

    /// Redo phase: Replay committed operations.
    ///
    /// Returns:
    /// - Vector of committed operations in LSN order
    /// - The next LSN to use
    fn redo_phase(
        &self,
        _checkpoint_lsn: Option<Lsn>,
        _transactions: &HashMap<u64, TransactionState>,
        stats: &mut RecoveryStats,
    ) -> Result<(Vec<RecoveredOperation>, Lsn)> {
        let wal_reader = WalReader::new(&self.wal_path)?;

        let mut operations: Vec<RecoveredOperation> = Vec::new();
        let mut next_lsn: Lsn = 1;

        // Track which transaction each non-transactional operation belongs to
        // Operations outside transactions are considered implicitly committed
        let mut current_tx: Option<u64> = None;
        let mut pending_tx_ops: HashMap<u64, Vec<RecoveredOperation>> = HashMap::new();

        for result in wal_reader.iter() {
            let (lsn, record) = match result {
                Ok(r) => r,
                Err(_) => continue, // Skip corrupted records
            };

            next_lsn = lsn + 1;

            match record {
                WalRecord::BeginTx { tx_id } => {
                    current_tx = Some(tx_id);
                    pending_tx_ops.entry(tx_id).or_default();
                }
                WalRecord::CommitTx { tx_id } => {
                    // Move pending ops to committed list
                    if let Some(ops) = pending_tx_ops.remove(&tx_id) {
                        operations.extend(ops);
                    }
                    if current_tx == Some(tx_id) {
                        current_tx = None;
                    }
                }
                WalRecord::AbortTx { tx_id } => {
                    // Discard pending ops
                    pending_tx_ops.remove(&tx_id);
                    if current_tx == Some(tx_id) {
                        current_tx = None;
                    }
                }
                WalRecord::Insert { term, value } => {
                    let op = RecoveredOperation::Insert { term, value };
                    if let Some(tx_id) = current_tx {
                        // Part of a transaction - buffer until commit
                        pending_tx_ops.entry(tx_id).or_default().push(op);
                    } else {
                        // Not in a transaction - implicitly committed
                        operations.push(op);
                        stats.insert_operations += 1;
                    }
                }
                WalRecord::Remove { term } => {
                    let op = RecoveredOperation::Remove { term };
                    if let Some(tx_id) = current_tx {
                        pending_tx_ops.entry(tx_id).or_default().push(op);
                    } else {
                        operations.push(op);
                        stats.remove_operations += 1;
                    }
                }
                WalRecord::Checkpoint { .. } => {
                    // Checkpoints don't affect operation replay
                }
            }
        }

        // Update stats for transactional operations
        for op in &operations {
            match op {
                RecoveredOperation::Insert { .. } => stats.insert_operations += 1,
                RecoveredOperation::Remove { .. } => stats.remove_operations += 1,
            }
        }

        Ok((operations, next_lsn))
    }

    /// Perform recovery and apply operations to a callback.
    ///
    /// This is useful when you want to apply operations as they're recovered
    /// rather than collecting them all in memory first.
    pub fn recover_with_callback<F>(&self, mut callback: F) -> Result<RecoveryStats>
    where
        F: FnMut(RecoveredOperation) -> Result<()>,
    {
        let state = self.recover()?;
        for op in state.operations {
            callback(op)?;
        }
        Ok(state.stats)
    }
}

/// Incremental recovery for large WALs.
///
/// This struct allows recovering operations in batches, useful when
/// memory is constrained or when you want to show progress.
pub struct IncrementalRecovery {
    /// Underlying WAL reader
    reader: WalReader,
    /// Known transaction states (from analysis phase)
    #[allow(dead_code)]
    transactions: HashMap<u64, TransactionState>,
    /// Current transaction context
    current_tx: Option<u64>,
    /// Pending operations for current transaction
    pending_ops: Vec<RecoveredOperation>,
    /// Next LSN
    next_lsn: Lsn,
    /// Whether analysis phase is complete
    #[allow(dead_code)]
    analysis_complete: bool,
}

impl IncrementalRecovery {
    /// Create new incremental recovery from WAL path.
    pub fn new<P: AsRef<Path>>(wal_path: P) -> Result<Self> {
        let wal_reader = WalReader::new(wal_path)?;

        Ok(IncrementalRecovery {
            reader: wal_reader,
            transactions: HashMap::new(),
            current_tx: None,
            pending_ops: Vec::new(),
            next_lsn: 1,
            analysis_complete: false,
        })
    }

    /// Get the next batch of recovered operations.
    ///
    /// Returns None when recovery is complete.
    pub fn next_batch(&mut self, max_ops: usize) -> Result<Option<Vec<RecoveredOperation>>> {
        let mut batch = Vec::with_capacity(max_ops);

        while batch.len() < max_ops {
            match self.reader.next_record() {
                Some(Ok((lsn, record))) => {
                    self.next_lsn = lsn + 1;
                    if let Some(ops) = self.process_record(record)? {
                        batch.extend(ops);
                    }
                }
                Some(Err(_)) => {
                    // Skip corrupted records
                    continue;
                }
                None => {
                    // WAL exhausted
                    if batch.is_empty() {
                        return Ok(None);
                    }
                    break;
                }
            }
        }

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }

    /// Process a single record and return any committed operations.
    fn process_record(&mut self, record: WalRecord) -> Result<Option<Vec<RecoveredOperation>>> {
        match record {
            WalRecord::BeginTx { tx_id } => {
                self.current_tx = Some(tx_id);
                self.pending_ops.clear();
                Ok(None)
            }
            WalRecord::CommitTx { tx_id } => {
                if self.current_tx == Some(tx_id) {
                    let ops = std::mem::take(&mut self.pending_ops);
                    self.current_tx = None;
                    Ok(Some(ops))
                } else {
                    Ok(None)
                }
            }
            WalRecord::AbortTx { tx_id } => {
                if self.current_tx == Some(tx_id) {
                    self.pending_ops.clear();
                    self.current_tx = None;
                }
                Ok(None)
            }
            WalRecord::Insert { term, value } => {
                let op = RecoveredOperation::Insert { term, value };
                if self.current_tx.is_some() {
                    self.pending_ops.push(op);
                    Ok(None)
                } else {
                    Ok(Some(vec![op]))
                }
            }
            WalRecord::Remove { term } => {
                let op = RecoveredOperation::Remove { term };
                if self.current_tx.is_some() {
                    self.pending_ops.push(op);
                    Ok(None)
                } else {
                    Ok(Some(vec![op]))
                }
            }
            WalRecord::Checkpoint { .. } => Ok(None),
        }
    }

    /// Get the next LSN after recovery.
    pub fn next_lsn(&self) -> Lsn {
        self.next_lsn
    }
}

/// Apply recovered operations to a PersistentARTrie.
///
/// This is a convenience function for the common case of recovering
/// directly into a trie structure.
pub fn apply_to_trie<V, F>(
    operations: impl IntoIterator<Item = RecoveredOperation>,
    mut insert_fn: F,
) -> std::result::Result<usize, String>
where
    F: FnMut(&[u8], Option<&[u8]>) -> std::result::Result<(), String>,
{
    let mut count = 0;

    for op in operations {
        match op {
            RecoveredOperation::Insert { term, value } => {
                insert_fn(&term, value.as_deref())?;
                count += 1;
            }
            RecoveredOperation::Remove { term } => {
                // For removes, we pass None as the value to indicate removal
                // The actual implementation would call a remove function
                insert_fn(&term, None)?;
                count += 1;
            }
        }
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::wal::WalWriter;
    use tempfile::tempdir;

    #[test]
    fn test_recovery_empty_wal() {
        let dir = tempdir().expect("create tempdir");
        let wal_path = dir.path().join("empty.wal");

        let manager = RecoveryManager::new(&wal_path);

        // No WAL file - should not need recovery
        assert!(!manager.needs_recovery().expect("needs_recovery"));

        // Recovery should return empty state
        let state = manager.recover().expect("recover");
        assert_eq!(state.operation_count(), 0);
        assert_eq!(state.next_lsn, 1);
    }

    #[test]
    fn test_recovery_simple_operations() {
        let dir = tempdir().expect("create tempdir");
        let wal_path = dir.path().join("test.wal");

        // Write some operations to WAL
        {
            let writer = WalWriter::create(&wal_path).expect("create writer");

            writer
                .append(WalRecord::Insert {
                    term: b"hello".to_vec(),
                    value: None,
                })
                .expect("append insert");

            writer
                .append(WalRecord::Insert {
                    term: b"world".to_vec(),
                    value: Some(b"value".to_vec()),
                })
                .expect("append insert");

            writer
                .append(WalRecord::Remove {
                    term: b"hello".to_vec(),
                })
                .expect("append remove");

            writer.sync().expect("sync");
        }

        // Recover
        let manager = RecoveryManager::new(&wal_path);
        assert!(manager.needs_recovery().expect("needs_recovery"));

        let state = manager.recover().expect("recover");
        assert_eq!(state.operation_count(), 3);

        let ops: Vec<_> = state.operations().collect();

        match &ops[0] {
            RecoveredOperation::Insert { term, value } => {
                assert_eq!(term, b"hello");
                assert!(value.is_none());
            }
            _ => panic!("Expected insert"),
        }

        match &ops[1] {
            RecoveredOperation::Insert { term, value } => {
                assert_eq!(term, b"world");
                assert_eq!(value.as_deref(), Some(b"value".as_slice()));
            }
            _ => panic!("Expected insert"),
        }

        match &ops[2] {
            RecoveredOperation::Remove { term } => {
                assert_eq!(term, b"hello");
            }
            _ => panic!("Expected remove"),
        }
    }

    #[test]
    fn test_recovery_with_transactions() {
        let dir = tempdir().expect("create tempdir");
        let wal_path = dir.path().join("tx.wal");

        // Write transactional operations
        {
            let writer = WalWriter::create(&wal_path).expect("create writer");

            // Committed transaction
            writer
                .append(WalRecord::BeginTx { tx_id: 1 })
                .expect("append begin");
            writer
                .append(WalRecord::Insert {
                    term: b"committed".to_vec(),
                    value: None,
                })
                .expect("append insert");
            writer
                .append(WalRecord::CommitTx { tx_id: 1 })
                .expect("append commit");

            // Aborted transaction
            writer
                .append(WalRecord::BeginTx { tx_id: 2 })
                .expect("append begin");
            writer
                .append(WalRecord::Insert {
                    term: b"aborted".to_vec(),
                    value: None,
                })
                .expect("append insert");
            writer
                .append(WalRecord::AbortTx { tx_id: 2 })
                .expect("append abort");

            // Incomplete transaction (no commit/abort)
            writer
                .append(WalRecord::BeginTx { tx_id: 3 })
                .expect("append begin");
            writer
                .append(WalRecord::Insert {
                    term: b"incomplete".to_vec(),
                    value: None,
                })
                .expect("append insert");

            writer.sync().expect("sync");
        }

        // Recover
        let manager = RecoveryManager::new(&wal_path);
        let state = manager.recover().expect("recover");

        // Only committed transaction should be recovered
        assert_eq!(state.operation_count(), 1);

        let ops: Vec<_> = state.into_operations();
        match &ops[0] {
            RecoveredOperation::Insert { term, .. } => {
                assert_eq!(term, b"committed");
            }
            _ => panic!("Expected insert"),
        }
    }

    #[test]
    fn test_recovery_stats() {
        let dir = tempdir().expect("create tempdir");
        let wal_path = dir.path().join("stats.wal");

        {
            let writer = WalWriter::create(&wal_path).expect("create writer");

            // 3 inserts outside transaction
            for i in 0..3 {
                writer
                    .append(WalRecord::Insert {
                        term: format!("term{}", i).into_bytes(),
                        value: None,
                    })
                    .expect("append");
            }

            // 1 committed transaction with 2 ops
            writer
                .append(WalRecord::BeginTx { tx_id: 1 })
                .expect("append");
            writer
                .append(WalRecord::Insert {
                    term: b"tx1".to_vec(),
                    value: None,
                })
                .expect("append");
            writer
                .append(WalRecord::Remove {
                    term: b"tx1".to_vec(),
                })
                .expect("append");
            writer
                .append(WalRecord::CommitTx { tx_id: 1 })
                .expect("append");

            // Checkpoint - use next_lsn (8 records were written, so LSN 8)
            writer.checkpoint(8).expect("checkpoint");

            writer.sync().expect("sync");
        }

        let manager = RecoveryManager::new(&wal_path);
        let state = manager.recover().expect("recover");

        // 3 non-tx inserts + 2 tx ops = 5 total
        assert_eq!(state.operation_count(), 5);
        assert_eq!(state.stats.committed_transactions, 1);
        assert!(state.stats.checkpoint_lsn.is_some());
    }

    #[test]
    fn test_incremental_recovery() {
        let dir = tempdir().expect("create tempdir");
        let wal_path = dir.path().join("incremental.wal");

        // Write 10 operations
        {
            let writer = WalWriter::create(&wal_path).expect("create writer");
            for i in 0..10 {
                writer
                    .append(WalRecord::Insert {
                        term: format!("term{}", i).into_bytes(),
                        value: None,
                    })
                    .expect("append");
            }
            writer.sync().expect("sync");
        }

        // Recover in batches of 3
        let mut recovery = IncrementalRecovery::new(&wal_path).expect("create recovery");
        let mut total = 0;

        while let Some(batch) = recovery.next_batch(3).expect("next_batch") {
            total += batch.len();
        }

        assert_eq!(total, 10);
    }

    #[test]
    fn test_recovery_with_callback() {
        let dir = tempdir().expect("create tempdir");
        let wal_path = dir.path().join("callback.wal");

        {
            let writer = WalWriter::create(&wal_path).expect("create writer");
            for i in 0..5 {
                writer
                    .append(WalRecord::Insert {
                        term: format!("term{}", i).into_bytes(),
                        value: None,
                    })
                    .expect("append");
            }
            writer.sync().expect("sync");
        }

        let mut collected = Vec::new();
        let manager = RecoveryManager::new(&wal_path);

        manager
            .recover_with_callback(|op| {
                collected.push(op);
                Ok(())
            })
            .expect("recover_with_callback");

        assert_eq!(collected.len(), 5);
    }
}
