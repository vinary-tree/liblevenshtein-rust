//! Error handling tests for PersistentARTrie.
//!
//! These tests verify graceful error handling:
//! - Corrupted WAL with various patterns
//! - Partial writes / simulated crashes
//! - File permission errors
//! - Invalid file formats
//! - Edge cases in recovery
//!
//! # Test Strategy
//!
//! We test both the error detection (does it fail?) and graceful
//! degradation (does it recover what it can?).

#![cfg(feature = "persistent-artrie")]

use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
use liblevenshtein::dictionary::Dictionary;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use tempfile::TempDir;

// =============================================================================
// Test: WAL Corruption Patterns
// =============================================================================

#[test]
fn test_corrupted_wal_truncated() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("truncated_wal.part");
    // WAL uses .with_extension("wal"), so foo.part -> foo.wal
    let wal_path = dict_path.with_extension("wal");

    // Create dictionary and insert some data
    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");
        for i in 0..10 {
            let _ = dict.insert(&format!("term{:03}", i));
        }
        dict.sync().expect("sync");
    }

    // Verify WAL exists before corruption
    assert!(wal_path.exists(), "WAL should exist after sync");

    // Corrupt the WAL by truncating it mid-record
    {
        let file = OpenOptions::new()
            .write(true)
            .open(&wal_path)
            .expect("open wal");

        let len = file.metadata().expect("metadata").len();
        if len > 50 {
            file.set_len(len - 20).expect("truncate");
        }
    }

    // Try to open - should either succeed with partial recovery or fail gracefully
    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::open(&dict_path);

    // Either we recover successfully or we get a recoverable error
    match result {
        Ok(dict) => {
            // Partial recovery is acceptable
            // We may have fewer terms than inserted
            assert!(dict.len().unwrap_or(0) <= 10);
        }
        Err(e) => {
            // Error should be a recovery error, not a panic
            let msg = format!("{:?}", e);
            assert!(
                msg.contains("Recovery")
                    || msg.contains("WAL")
                    || msg.contains("truncat")
                    || msg.contains("incomplete"),
                "Unexpected error: {}",
                msg
            );
        }
    }
}

#[test]
fn test_corrupted_wal_garbage_bytes() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("garbage_wal.part");
    let wal_path = dict_path.with_extension("wal");

    // Create dictionary
    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");
        for i in 0..5 {
            let _ = dict.insert(&format!("term{:03}", i));
        }
        dict.sync().expect("sync");
    }

    assert!(wal_path.exists(), "WAL should exist after sync");

    // Append garbage to WAL
    {
        let mut file = OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .expect("open wal");

        // Write random garbage that doesn't look like a valid record
        let garbage = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0xFF, 0xFF];
        file.write_all(&garbage).expect("write garbage");
    }

    // Recovery should handle garbage gracefully
    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::open(&dict_path);

    match result {
        Ok(dict) => {
            // Should recover valid records before the garbage
            // Recovery may have succeeded with some or all terms
            let _ = dict.contains("term000"); // Check doesn't panic
            let _ = dict.len(); // Get length doesn't panic
        }
        Err(e) => {
            // Error is acceptable - corrupted data
            let msg = format!("{:?}", e);
            assert!(
                msg.contains("Recovery") || msg.contains("parse") || msg.contains("invalid"),
                "Unexpected error: {}",
                msg
            );
        }
    }
}

#[test]
fn test_corrupted_wal_header() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("header_wal.part");
    let wal_path = dict_path.with_extension("wal");

    // Create dictionary
    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");
        let _ = dict.insert("test");
        dict.sync().expect("sync");
    }

    assert!(wal_path.exists(), "WAL should exist after sync");

    // Corrupt the WAL header (first bytes)
    {
        let mut file = OpenOptions::new()
            .write(true)
            .open(&wal_path)
            .expect("open wal");

        // Overwrite the magic bytes
        file.seek(SeekFrom::Start(0)).expect("seek");
        file.write_all(&[0x00, 0x00, 0x00, 0x00]).expect("write");
    }

    // Should fail to open or recover with error
    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::open(&dict_path);

    match result {
        Ok(_) => {
            // Acceptable if we fall back to empty state
        }
        Err(e) => {
            let msg = format!("{:?}", e);
            assert!(
                msg.contains("header") || msg.contains("magic") || msg.contains("Recovery"),
                "Unexpected error: {}",
                msg
            );
        }
    }
}

#[test]
fn test_corrupted_wal_checksum() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("checksum_wal.part");
    let wal_path = dict_path.with_extension("wal");

    // Create dictionary with checkpoint
    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");
        for i in 0..10 {
            let _ = dict.insert(&format!("term{:03}", i));
        }
        dict.checkpoint().expect("checkpoint");
    }

    assert!(wal_path.exists(), "WAL should exist after checkpoint");

    // Flip some bits in the middle of the WAL
    {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&wal_path)
            .expect("open wal");

        let len = file.metadata().expect("metadata").len();
        if len > 100 {
            file.seek(SeekFrom::Start(len / 2)).expect("seek");
            let mut byte = [0u8; 1];
            file.read_exact(&mut byte).expect("read");
            byte[0] ^= 0xFF; // Flip all bits
            file.seek(SeekFrom::Start(len / 2)).expect("seek back");
            file.write_all(&byte).expect("write");
        }
    }

    // Recovery should detect corruption
    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::open(&dict_path);

    // Either fails or recovers partially
    match result {
        Ok(dict) => {
            // Partial recovery is acceptable
            assert!(dict.len().unwrap_or(0) <= 10);
        }
        Err(e) => {
            let msg = format!("{:?}", e);
            assert!(
                msg.contains("checksum")
                    || msg.contains("corrupt")
                    || msg.contains("invalid")
                    || msg.contains("Recovery"),
                "Unexpected error: {}",
                msg
            );
        }
    }
}

// =============================================================================
// Test: File System Errors
// =============================================================================

#[test]
fn test_create_in_nonexistent_directory() {
    let path = "/nonexistent/directory/path/that/does/not/exist/dict.part";

    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::create(path);

    assert!(result.is_err(), "Should fail for nonexistent directory");

    let err = result.unwrap_err();
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("No such file")
            || msg.contains("not found")
            || msg.contains("directory")
            || msg.contains("Io"),
        "Unexpected error: {}",
        msg
    );
}

#[test]
fn test_open_nonexistent_file() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let path = temp_dir.path().join("does_not_exist.part");

    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::open(&path);

    assert!(result.is_err(), "Should fail for nonexistent file");

    let err = result.unwrap_err();
    let msg = format!("{:?}", err);
    assert!(
        msg.contains("not found") || msg.contains("No such") || msg.contains("Io"),
        "Unexpected error: {}",
        msg
    );
}

#[test]
fn test_create_existing_file() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("existing.part");

    // Create first dictionary
    {
        let _dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");
    }

    // Try to create again - should fail
    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::create(&dict_path);

    assert!(result.is_err(), "Should fail for existing file");
}

#[test]
#[cfg(unix)]
fn test_readonly_directory() {
    use std::os::unix::fs::PermissionsExt;

    let temp_dir = TempDir::new().expect("create temp dir");
    let readonly_dir = temp_dir.path().join("readonly");
    fs::create_dir(&readonly_dir).expect("create dir");

    // Make directory read-only
    let mut perms = fs::metadata(&readonly_dir).expect("metadata").permissions();
    perms.set_mode(0o555); // r-xr-xr-x
    fs::set_permissions(&readonly_dir, perms.clone()).expect("set perms");

    let dict_path = readonly_dir.join("dict.part");
    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::create(&dict_path);

    // Restore permissions before checking result
    perms.set_mode(0o755);
    let _ = fs::set_permissions(&readonly_dir, perms);

    assert!(result.is_err(), "Should fail for read-only directory");
}

// =============================================================================
// Test: Empty/Minimal Files
// =============================================================================

#[test]
fn test_empty_wal_file() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("empty_wal.part");
    let wal_path = dict_path.with_extension("wal");

    // Create dictionary
    {
        let _dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");
    }

    // Replace WAL with empty file
    {
        let _ = File::create(&wal_path).expect("create empty file");
    }

    // Should handle empty WAL gracefully
    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::open(&dict_path);

    match result {
        Ok(dict) => {
            // Empty dictionary is acceptable
            assert_eq!(dict.len().unwrap_or(0), 0);
        }
        Err(e) => {
            // Error about empty/invalid WAL is acceptable
            let msg = format!("{:?}", e);
            assert!(
                msg.contains("empty")
                    || msg.contains("header")
                    || msg.contains("too short")
                    || msg.contains("Recovery")
                    || msg.contains("IoError")
                    || msg.contains("UnexpectedEof"),
                "Unexpected error: {}",
                msg
            );
        }
    }
}

#[test]
fn test_zero_filled_wal() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("zero_wal.part");
    let wal_path = dict_path.with_extension("wal");

    // Create dictionary
    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");
        let _ = dict.insert("test");
        dict.sync().expect("sync");
    }

    assert!(wal_path.exists(), "WAL should exist after sync");

    // Replace WAL content with zeros
    {
        let mut file = OpenOptions::new()
            .write(true)
            .open(&wal_path)
            .expect("open wal");

        let zeros = vec![0u8; 1024];
        file.write_all(&zeros).expect("write zeros");
    }

    // Should handle zero-filled WAL
    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::open(&dict_path);

    // Either error or empty recovery
    match result {
        Ok(dict) => {
            assert!(dict.len().unwrap_or(0) <= 1);
        }
        Err(_) => {
            // Error is acceptable
        }
    }
}

// =============================================================================
// Test: Recovery Edge Cases
// =============================================================================

#[test]
fn test_recovery_with_only_removes() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("only_removes.part");

    // Create dictionary, insert, remove, and checkpoint
    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");

        for i in 0..5 {
            let _ = dict.insert(&format!("term{}", i));
        }
        for i in 0..5 {
            let _ = dict.remove(&format!("term{}", i));
        }

        dict.sync().expect("sync");
    }

    // Reopen
    let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path).expect("open");

    // Should have 0 terms after removes
    assert_eq!(dict.len().unwrap_or(0), 0);
}

#[test]
fn test_recovery_interleaved_insert_remove() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("interleaved.part");

    // Create with interleaved operations
    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");

        let _ = dict.insert("keep1");
        let _ = dict.insert("remove1");
        let _ = dict.remove("remove1");
        let _ = dict.insert("keep2");
        let _ = dict.insert("remove2");
        let _ = dict.remove("remove2");
        let _ = dict.insert("keep3");

        dict.sync().expect("sync");
    }

    // Reopen and verify
    let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path).expect("open");

    assert!(dict.contains("keep1"));
    assert!(dict.contains("keep2"));
    assert!(dict.contains("keep3"));
    assert!(!dict.contains("remove1"));
    assert!(!dict.contains("remove2"));
}

#[test]
fn test_recovery_duplicate_inserts() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("duplicates.part");

    // Insert same term multiple times
    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");

        // First insert succeeds
        assert!(dict.insert("term"));
        // Subsequent inserts should fail (term exists)
        assert!(!dict.insert("term"));
        assert!(!dict.insert("term"));

        dict.sync().expect("sync");
    }

    // Reopen
    let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path).expect("open");

    assert!(dict.contains("term"));
    assert_eq!(dict.len().unwrap_or(0), 1);
}

#[test]
fn test_recovery_remove_nonexistent() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("remove_nonexistent.part");

    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");

        let _ = dict.insert("exists");
        // Try to remove something that doesn't exist
        assert!(!dict.remove("does_not_exist"));

        dict.sync().expect("sync");
    }

    // Reopen
    let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path).expect("open");

    assert!(dict.contains("exists"));
    assert!(!dict.contains("does_not_exist"));
}

// =============================================================================
// Test: Checkpoint Edge Cases
// =============================================================================

#[test]
fn test_checkpoint_empty_dictionary() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("empty_checkpoint.part");

    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");

        // Checkpoint empty dictionary
        dict.checkpoint().expect("checkpoint");
    }

    // Reopen
    let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path).expect("open");

    assert_eq!(dict.len().unwrap_or(0), 0);
}

#[test]
fn test_multiple_checkpoints_empty() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("multi_empty_checkpoint.part");

    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");

        // Multiple checkpoints on empty dictionary
        for _ in 0..5 {
            dict.checkpoint().expect("checkpoint");
        }
    }

    // Reopen
    let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path).expect("open");

    assert_eq!(dict.len().unwrap_or(0), 0);
}

#[test]
fn test_checkpoint_after_all_removed() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("all_removed_checkpoint.part");

    {
        let mut dict: PersistentARTrie<()> =
            PersistentARTrie::create(&dict_path).expect("create dict");

        // Insert and remove all
        for i in 0..5 {
            let _ = dict.insert(&format!("term{}", i));
        }
        for i in 0..5 {
            let _ = dict.remove(&format!("term{}", i));
        }

        // Checkpoint when dictionary is empty
        dict.checkpoint().expect("checkpoint");
    }

    // Reopen
    let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path).expect("open");

    assert_eq!(dict.len().unwrap_or(0), 0);
}

// =============================================================================
// Test: Invalid Data Types
// =============================================================================

#[test]
fn test_open_wrong_file_type() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("wrong_type.part");

    // Create a text file instead of a dictionary
    {
        let mut file = File::create(&dict_path).expect("create file");
        file.write_all(b"This is not a dictionary file\n")
            .expect("write");
    }

    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::open(&dict_path);

    assert!(result.is_err(), "Should fail for wrong file type");
}

#[test]
fn test_moderately_large_term() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("large_term.part");

    let mut dict: PersistentARTrie<()> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    // Try to insert a moderately large term (1KB - safe for stack)
    // Note: Very large terms (64KB+) can cause stack overflow due to
    // recursive bucket implementation. This is a known limitation.
    let large_term: String = "x".repeat(1_000);
    let result = dict.insert(&large_term);

    // Should succeed for 1KB terms
    assert!(result);
    assert!(dict.contains(&large_term));
}

// =============================================================================
// Test: Sync Without Writes
// =============================================================================

#[test]
fn test_sync_no_changes() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("sync_no_changes.part");

    let dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path).expect("create dict");

    // Sync without any writes - should be no-op
    dict.sync().expect("sync");
    dict.sync().expect("sync again");
}

#[test]
fn test_checkpoint_no_changes() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("checkpoint_no_changes.part");

    let mut dict: PersistentARTrie<()> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    // Insert, checkpoint, then checkpoint again without changes
    let _ = dict.insert("term");
    dict.checkpoint().expect("checkpoint 1");
    dict.checkpoint().expect("checkpoint 2"); // No changes since last checkpoint
}

// =============================================================================
// Test: Error Message Quality
// =============================================================================

#[test]
fn test_error_messages_are_descriptive() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let nonexistent = temp_dir.path().join("nonexistent.part");

    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::open(&nonexistent);

    assert!(result.is_err());
    let err = result.unwrap_err();

    // Error should be convertible to string
    let msg = format!("{}", err);
    assert!(!msg.is_empty(), "Error message should not be empty");

    // Debug format should work too
    let debug_msg = format!("{:?}", err);
    assert!(!debug_msg.is_empty(), "Debug message should not be empty");
}
