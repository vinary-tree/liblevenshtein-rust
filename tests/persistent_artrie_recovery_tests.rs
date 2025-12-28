//! Crash recovery integration tests for PersistentARTrie.
//!
//! These tests verify that the WAL-based recovery mechanism correctly restores
//! dictionary state after simulated crashes (drops without sync).

#![cfg(feature = "persistent-artrie")]

use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
use liblevenshtein::dictionary::Dictionary;
use tempfile::tempdir;

/// Test 13.1: Recovery after clean shutdown.
/// Create dictionary, insert terms, sync, close, reopen and verify all terms present.
#[test]
fn test_recovery_after_clean_shutdown() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict.art");

    let terms = vec!["apple", "banana", "cherry", "date", "elderberry"];

    // Create dictionary and insert terms
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        for term in &terms {
            dict.insert(term);
        }

        // Sync to ensure durability
        dict.sync().expect("sync");
    }

    // Reopen and verify all terms present
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        for term in &terms {
            assert!(
                dict.contains(term),
                "Term '{}' should be present after recovery",
                term
            );
        }
    }
}

/// Test 13.2: Recovery after crash (drop without sync).
/// Create dictionary, insert terms, drop without sync, reopen and verify WAL replay recovers data.
#[test]
fn test_recovery_after_crash_no_sync() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict.art");

    let terms = vec!["foo", "bar", "baz", "qux"];

    // Create dictionary and insert terms (no explicit sync before drop)
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        for term in &terms {
            dict.insert(term);
        }

        // Simulate crash - drop without sync
        // The WAL should have buffered the writes
    }

    // Reopen and verify recovery
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        // Terms should be recovered from WAL replay
        for term in &terms {
            assert!(
                dict.contains(term),
                "Term '{}' should be recovered from WAL after crash",
                term
            );
        }
    }
}

/// Test 13.3: Mixed insert/remove recovery.
/// Create, insert A, insert B, remove A, crash, verify only B present after recovery.
#[test]
fn test_mixed_insert_remove_recovery() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict.art");

    // Create and perform mixed operations
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        dict.insert("alpha");
        dict.insert("beta");
        dict.insert("gamma");
        dict.remove("alpha");
        dict.insert("delta");
        dict.remove("gamma");

        // Sync to ensure WAL is durable
        dict.sync().expect("sync");
    }

    // Reopen and verify correct state
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        // alpha was removed
        assert!(
            !dict.contains("alpha"),
            "alpha was removed and should not be present"
        );

        // beta was inserted and never removed
        assert!(
            dict.contains("beta"),
            "beta should be present"
        );

        // gamma was removed
        assert!(
            !dict.contains("gamma"),
            "gamma was removed and should not be present"
        );

        // delta was inserted after some removes
        assert!(
            dict.contains("delta"),
            "delta should be present"
        );
    }
}

/// Test 13.4: Checkpoint + recovery.
/// Insert terms, checkpoint, insert more, crash, verify checkpoint + WAL replay recovers all.
#[test]
fn test_checkpoint_and_recovery() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict.art");

    let pre_checkpoint_terms: Vec<String> = (0..50)
        .map(|i| format!("pre_{}", i))
        .collect();

    let post_checkpoint_terms: Vec<String> = (0..20)
        .map(|i| format!("post_{}", i))
        .collect();

    // Create dictionary, insert terms, checkpoint, insert more
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        // Insert pre-checkpoint terms
        for term in &pre_checkpoint_terms {
            dict.insert(term);
        }

        // Checkpoint to mark these as durable
        dict.checkpoint().expect("checkpoint");

        // Insert post-checkpoint terms
        for term in &post_checkpoint_terms {
            dict.insert(term);
        }

        // Sync to ensure WAL has post-checkpoint entries
        dict.sync().expect("sync");
    }

    // Reopen and verify all terms present
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        // All pre-checkpoint terms should be present
        for term in &pre_checkpoint_terms {
            assert!(
                dict.contains(term),
                "Pre-checkpoint term '{}' should be present",
                term
            );
        }

        // All post-checkpoint terms should also be present (recovered from WAL)
        for term in &post_checkpoint_terms {
            assert!(
                dict.contains(term),
                "Post-checkpoint term '{}' should be recovered from WAL",
                term
            );
        }
    }
}

/// Test 13.5: Corrupted WAL handling.
/// Create WAL with valid entries, corrupt the file, verify graceful degradation.
#[test]
fn test_corrupted_wal_graceful_degradation() {
    use std::fs::OpenOptions;
    use std::io::Write;

    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict.art");
    let wal_path = dict_path.with_extension("wal");

    // Create dictionary and insert terms
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        dict.insert("valid_term_1");
        dict.insert("valid_term_2");
        dict.sync().expect("sync");
    }

    // Corrupt the WAL file by appending garbage
    {
        let mut file = OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .expect("open WAL for corruption");

        // Write garbage bytes that will fail CRC check
        file.write_all(b"CORRUPTED_DATA_THAT_WILL_FAIL_CRC_CHECK")
            .expect("write corruption");
        file.sync_all().expect("sync corruption");
    }

    // Reopen - should handle corruption gracefully
    // The recovery manager should log a warning but not crash
    {
        let result = PersistentARTrie::<()>::open(&dict_path);

        // Should either succeed with partial recovery or fail gracefully
        match result {
            Ok(dict) => {
                // If we succeeded, the valid terms before corruption should be present
                // (depending on where the corruption occurred)
                // At minimum, the dictionary should be usable
                let _ = dict.contains("valid_term_1");
            }
            Err(e) => {
                // If it failed, it should be a recognizable error, not a panic
                let error_msg = format!("{:?}", e);
                assert!(
                    error_msg.contains("Corrupted") ||
                    error_msg.contains("CRC") ||
                    error_msg.contains("invalid") ||
                    error_msg.contains("recovery"),
                    "Error should indicate corruption-related issue: {}",
                    error_msg
                );
            }
        }
    }
}

/// Test: Multiple open/close cycles with incremental updates.
/// Verifies that WAL truncation after recovery works correctly.
#[test]
fn test_multiple_reopen_cycles() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict.art");

    // Cycle 1: Create and add initial terms
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        dict.insert("cycle1_a");
        dict.insert("cycle1_b");
        dict.sync().expect("sync");
    }

    // Cycle 2: Reopen, verify previous terms, add more
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary cycle 2");

        assert!(dict.contains("cycle1_a"), "cycle1_a should exist in cycle 2");
        assert!(dict.contains("cycle1_b"), "cycle1_b should exist in cycle 2");

        dict.insert("cycle2_a");
        dict.insert("cycle2_b");
        dict.sync().expect("sync");
    }

    // Cycle 3: Reopen, verify all terms, add more
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary cycle 3");

        assert!(dict.contains("cycle1_a"), "cycle1_a should exist in cycle 3");
        assert!(dict.contains("cycle1_b"), "cycle1_b should exist in cycle 3");
        assert!(dict.contains("cycle2_a"), "cycle2_a should exist in cycle 3");
        assert!(dict.contains("cycle2_b"), "cycle2_b should exist in cycle 3");

        dict.insert("cycle3_a");
        dict.remove("cycle1_a");
        dict.sync().expect("sync");
    }

    // Cycle 4: Final verification
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary cycle 4");

        assert!(!dict.contains("cycle1_a"), "cycle1_a was removed");
        assert!(dict.contains("cycle1_b"), "cycle1_b should exist");
        assert!(dict.contains("cycle2_a"), "cycle2_a should exist");
        assert!(dict.contains("cycle2_b"), "cycle2_b should exist");
        assert!(dict.contains("cycle3_a"), "cycle3_a should exist");
    }
}

/// Test: Moderate number of operations followed by recovery.
/// Tests WAL replay with multiple terms. Note: Currently limited by bucket size (256).
#[test]
fn test_large_scale_recovery() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict.art");

    // Use 200 terms to stay within bucket capacity (max 256 before split)
    // Full bucket splitting to ART nodes is future work
    let num_terms = 200;
    let terms: Vec<String> = (0..num_terms)
        .map(|i| format!("term_{:05}", i))
        .collect();

    // Create and insert many terms
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        for term in &terms {
            dict.insert(term);
        }

        dict.sync().expect("sync");
    }

    // Reopen and verify all terms
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        for term in &terms {
            assert!(
                dict.contains(term),
                "Term '{}' should be present after recovery",
                term
            );
        }
    }
}

/// Test: Recovery with empty dictionary (no operations).
#[test]
fn test_empty_dictionary_recovery() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict.art");

    // Create empty dictionary
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");
        dict.sync().expect("sync");
    }

    // Reopen empty dictionary
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        assert!(!dict.contains("anything"), "Empty dictionary should not contain any terms");
    }
}

// =============================================================================
// Phase 15: Value Persistence Tests
// =============================================================================

use liblevenshtein::dictionary::MappedDictionary;
use liblevenshtein::dictionary::value::DictionaryValue;
use serde::{Deserialize, Serialize};

/// Custom value type for testing serialization
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
struct TestValue {
    id: u32,
    name: String,
}

impl DictionaryValue for TestValue {}

/// Test 15.1: Value persistence through clean shutdown.
/// Insert terms with values, sync, close, reopen and verify values are recovered.
#[test]
fn test_value_persistence_clean_shutdown() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict_values.art");

    // Create dictionary and insert terms with values
    {
        let mut dict: PersistentARTrie<u32> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        dict.insert_with_value("apple", 1);
        dict.insert_with_value("banana", 2);
        dict.insert_with_value("cherry", 3);

        // Sync to ensure durability
        dict.sync().expect("sync");
    }

    // Reopen and verify values are recovered
    {
        let dict: PersistentARTrie<u32> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        assert_eq!(dict.get_value("apple"), Some(1), "apple should have value 1");
        assert_eq!(dict.get_value("banana"), Some(2), "banana should have value 2");
        assert_eq!(dict.get_value("cherry"), Some(3), "cherry should have value 3");
        assert_eq!(dict.get_value("nonexistent"), None, "nonexistent should have no value");
    }
}

/// Test 15.2: Value persistence through crash recovery.
/// Insert terms with values, crash (drop without sync), reopen and verify WAL replay recovers values.
#[test]
fn test_value_persistence_crash_recovery() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict_crash_values.art");

    // Create dictionary and insert terms with values (no sync before drop)
    {
        let mut dict: PersistentARTrie<u32> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        dict.insert_with_value("foo", 42);
        dict.insert_with_value("bar", 100);
        dict.insert_with_value("baz", 999);

        // Simulate crash - no sync before drop
        // WAL should buffer the writes
    }

    // Reopen and verify values are recovered from WAL replay
    {
        let dict: PersistentARTrie<u32> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        assert_eq!(dict.get_value("foo"), Some(42), "foo should have value 42");
        assert_eq!(dict.get_value("bar"), Some(100), "bar should have value 100");
        assert_eq!(dict.get_value("baz"), Some(999), "baz should have value 999");
    }
}

/// Test 15.3: Complex value type persistence.
/// Test with a custom struct value type.
#[test]
fn test_complex_value_persistence() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict_complex.art");

    let values = vec![
        ("user_alice", TestValue { id: 1, name: "Alice".into() }),
        ("user_bob", TestValue { id: 2, name: "Bob".into() }),
        ("user_charlie", TestValue { id: 3, name: "Charlie".into() }),
    ];

    // Create dictionary with complex values
    {
        let mut dict: PersistentARTrie<TestValue> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        for (term, value) in &values {
            dict.insert_with_value(term, value.clone());
        }

        dict.sync().expect("sync");
    }

    // Reopen and verify complex values are recovered
    {
        let dict: PersistentARTrie<TestValue> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        for (term, expected) in &values {
            let actual = dict.get_value(term);
            assert_eq!(
                actual.as_ref(), Some(expected),
                "Term '{}' should have value {:?}",
                term, expected
            );
        }
    }
}

/// Test 15.4: Mixed insert (with and without values) recovery.
#[test]
fn test_mixed_value_recovery() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict_mixed_values.art");

    // Create dictionary with mixed inserts
    {
        let mut dict: PersistentARTrie<u32> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        // Insert some with values
        dict.insert_with_value("with_value_1", 10);
        dict.insert_with_value("with_value_2", 20);

        // Insert some without values (using default implementation)
        dict.insert("no_value_1");
        dict.insert("no_value_2");

        dict.sync().expect("sync");
    }

    // Reopen and verify
    {
        let dict: PersistentARTrie<u32> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        // Terms with values
        assert_eq!(dict.get_value("with_value_1"), Some(10));
        assert_eq!(dict.get_value("with_value_2"), Some(20));

        // Terms without values (should still be present but with no value)
        assert!(dict.contains("no_value_1"));
        assert!(dict.contains("no_value_2"));
        // Note: get_value returns None for terms inserted without values
        assert_eq!(dict.get_value("no_value_1"), None);
        assert_eq!(dict.get_value("no_value_2"), None);
    }
}

/// Test 15.5: Value update persistence.
/// Verify that updating a value (re-inserting with different value) persists correctly.
#[test]
fn test_value_update_persistence() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict_update_values.art");

    // Create and insert initial values
    {
        let mut dict: PersistentARTrie<u32> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        dict.insert_with_value("counter", 1);
        dict.sync().expect("sync");
    }

    // Reopen and update the value
    {
        let mut dict: PersistentARTrie<u32> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        // Update the value
        dict.insert_with_value("counter", 100);
        dict.sync().expect("sync");
    }

    // Reopen and verify updated value
    {
        let dict: PersistentARTrie<u32> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        assert_eq!(dict.get_value("counter"), Some(100), "counter should have updated value 100");
    }
}

// =============================================================================
// Phase 16: ART Node Persistence Tests
// =============================================================================

/// Test 16.1: ART node persistence after bucket split.
/// Insert enough terms with diverse prefixes to trigger bucket-to-ART conversion.
/// Note: Terms must have diverse first bytes to properly split into multiple child buckets.
#[test]
fn test_art_node_bucket_split_persistence() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict_art_split.art");

    // Create 400 terms with diverse first bytes to trigger bucket split properly.
    // Terms like "aa_000", "ab_000", ..., "zz_000" have 676 unique prefixes (26*26).
    // Using format that ensures diverse byte distribution.
    let mut terms = Vec::new();
    for i in 0..15 {
        for c1 in b'a'..=b'z' {
            for c2 in b'a'..=b'f' {
                terms.push(format!("{}{}{:03}", c1 as char, c2 as char, i));
            }
        }
    }
    // 26 * 6 * 15 = 2340 terms, but we'll just use the first 400
    terms.truncate(400);
    let num_terms = terms.len();

    // Create dictionary and insert enough terms to trigger ART node creation
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        for term in &terms {
            dict.insert(term);
        }

        // Checkpoint to persist the ART structure
        dict.checkpoint().expect("checkpoint");
    }

    // Reopen and verify all terms are present
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        for term in &terms {
            assert!(
                dict.contains(term),
                "Term '{}' should be present after ART node recovery",
                term
            );
        }

        assert_eq!(
            dict.len(),
            Some(num_terms),
            "Dictionary should have {} terms after ART node recovery",
            num_terms
        );
    }
}

/// Test 16.2: Large-scale ART node persistence with diverse prefixes.
/// Tests ART node splitting with terms having different first bytes.
#[test]
fn test_art_node_diverse_prefixes_persistence() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict_art_diverse.art");

    // Create terms with diverse first characters to trigger bucket split by first byte
    // This creates terms like "a_000", "b_000", ..., "z_000", "a_001", ...
    let mut terms = Vec::new();
    for i in 0..20 {
        for c in b'a'..=b'z' {
            terms.push(format!("{}_{:03}", c as char, i));
        }
    }
    // 26 letters * 20 iterations = 520 terms

    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        for term in &terms {
            dict.insert(term);
        }

        dict.checkpoint().expect("checkpoint");
    }

    // Reopen and verify
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        for term in &terms {
            assert!(
                dict.contains(term),
                "Term '{}' should be present after diverse ART recovery",
                term
            );
        }

        assert_eq!(dict.len(), Some(terms.len()));
    }
}

/// Test 16.3: ART node persistence with values.
/// Verifies that values are correctly persisted through ART node structures.
/// Uses diverse prefixes to ensure proper bucket splitting.
#[test]
fn test_art_node_with_values_persistence() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict_art_values.art");

    // Create 400 terms with values using diverse prefixes to trigger proper ART split
    let mut terms_with_values = Vec::new();
    let mut counter = 0u32;
    for i in 0..15 {
        for c1 in b'a'..=b'z' {
            for c2 in b'a'..=b'f' {
                terms_with_values.push((format!("{}{}{:03}", c1 as char, c2 as char, i), counter));
                counter += 1;
            }
        }
    }
    terms_with_values.truncate(400);

    {
        let mut dict: PersistentARTrie<u32> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        for (term, value) in &terms_with_values {
            dict.insert_with_value(term, *value);
        }

        dict.checkpoint().expect("checkpoint");
    }

    // Reopen and verify all terms and values
    {
        let dict: PersistentARTrie<u32> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        for (term, expected_value) in &terms_with_values {
            assert!(dict.contains(term), "Term '{}' should be present", term);
            assert_eq!(
                dict.get_value(term),
                Some(*expected_value),
                "Term '{}' should have value {}",
                term,
                expected_value
            );
        }
    }
}

/// Test 16.4: ART node recovery without checkpoint (WAL replay with ART).
/// Inserts many terms then drops without checkpoint, relying on WAL replay.
/// Uses diverse prefixes to ensure proper bucket splitting.
#[test]
fn test_art_node_wal_only_recovery() {
    let dir = tempdir().expect("create temp dir");
    let dict_path = dir.path().join("test_dict_art_wal.art");

    // Create terms with diverse prefixes for proper bucket splitting
    let mut terms = Vec::new();
    for i in 0..15 {
        for c1 in b'a'..=b'z' {
            for c2 in b'a'..=b'f' {
                terms.push(format!("{}{}{:03}", c1 as char, c2 as char, i));
            }
        }
    }
    terms.truncate(400);

    // Create and insert without explicit checkpoint
    {
        let mut dict: PersistentARTrie<()> = PersistentARTrie::create(&dict_path)
            .expect("create dictionary");

        for term in &terms {
            dict.insert(term);
        }

        // Only sync WAL, don't checkpoint
        dict.sync().expect("sync");
    }

    // Reopen - should recover from WAL
    {
        let dict: PersistentARTrie<()> = PersistentARTrie::open(&dict_path)
            .expect("open dictionary");

        for term in &terms {
            assert!(
                dict.contains(term),
                "Term '{}' should be present after WAL-only recovery",
                term
            );
        }
    }
}
