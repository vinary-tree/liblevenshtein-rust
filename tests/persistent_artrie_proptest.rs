//! Property-based tests for PersistentARTrie using proptest
//!
//! These tests verify invariants and discover edge cases for the
//! persistent ART implementation.
//!
//! Run with: cargo test --features persistent-artrie --test persistent_artrie_proptest

#![cfg(feature = "persistent-artrie")]

use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
use liblevenshtein::dictionary::{Dictionary, MappedDictionary};
use proptest::prelude::*;
use std::collections::{HashMap, HashSet};
use tempfile::TempDir;

// Strategy for generating simple ASCII terms
fn ascii_term_strategy() -> impl Strategy<Value = String> {
    // Lowercase ASCII letters, 1-20 characters
    "[a-z]{1,20}"
}

// Strategy for generating terms with diverse first bytes (for bucket splitting)
fn diverse_term_strategy() -> impl Strategy<Value = String> {
    // First char from a-z, rest from a-z0-9, 2-15 chars total
    prop::string::string_regex("[a-z][a-z0-9]{1,14}")
        .expect("valid regex")
}

// Strategy for generating byte sequences (for fuzzing)
fn byte_term_strategy() -> impl Strategy<Value = Vec<u8>> {
    // Any printable ASCII bytes to avoid null bytes
    prop::collection::vec(32u8..127, 1..=30)
}

// Strategy for generating small dictionaries with values
fn small_dict_with_values_strategy() -> impl Strategy<Value = Vec<(String, i32)>> {
    prop::collection::vec((ascii_term_strategy(), any::<i32>()), 1..=50)
}

// Strategy for generating medium-sized dictionaries
fn medium_dict_strategy() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(diverse_term_strategy(), 10..=200)
}

// Strategy for operations (insert or remove)
#[derive(Debug, Clone)]
enum Operation {
    Insert(String, i32),
    Remove(String),
}

fn operation_strategy() -> impl Strategy<Value = Operation> {
    prop_oneof![
        // 70% inserts
        (ascii_term_strategy(), any::<i32>()).prop_map(|(t, v)| Operation::Insert(t, v)),
        // 30% removes
        ascii_term_strategy().prop_map(Operation::Remove),
    ]
}

fn operations_strategy() -> impl Strategy<Value = Vec<Operation>> {
    prop::collection::vec(operation_strategy(), 10..=100)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: All inserted items are retrievable
    #[test]
    fn prop_inserted_items_retrievable(
        terms in prop::collection::vec(ascii_term_strategy(), 1..=50)
    ) {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        {
            let mut dict = PersistentARTrie::<i32>::create(&path)
                .expect("create dict");

            // Deduplicate terms for testing
            let unique_terms: HashSet<_> = terms.iter().cloned().collect();

            // Insert all terms with values
            for (i, term) in unique_terms.iter().enumerate() {
                let _ = dict.insert_with_value(term, i as i32);
            }

            // Verify all terms are retrievable
            for (i, term) in unique_terms.iter().enumerate() {
                prop_assert!(
                    dict.contains(term),
                    "Term '{}' should be in dictionary after insert",
                    term
                );
                prop_assert_eq!(
                    dict.get_value(term),
                    Some(i as i32),
                    "Term '{}' should have value {}",
                    term,
                    i
                );
            }
        }
    }

    /// Property: Removed items are not retrievable
    #[test]
    fn prop_removed_items_not_retrievable(
        terms in prop::collection::vec(ascii_term_strategy(), 1..=30)
    ) {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        {
            let mut dict = PersistentARTrie::<i32>::create(&path)
                .expect("create dict");

            // Deduplicate terms
            let unique_terms: Vec<_> = terms.iter().cloned().collect::<HashSet<_>>().into_iter().collect();

            // Insert all terms
            for (i, term) in unique_terms.iter().enumerate() {
                let _ = dict.insert_with_value(term, i as i32);
            }

            // Remove half the terms
            let to_remove: Vec<_> = unique_terms.iter().take(unique_terms.len() / 2).cloned().collect();
            for term in &to_remove {
                let _ = dict.remove(term);
            }

            // Verify removed terms are not retrievable
            for term in &to_remove {
                prop_assert!(
                    !dict.contains(term),
                    "Term '{}' should NOT be in dictionary after remove",
                    term
                );
                prop_assert_eq!(
                    dict.get_value(term),
                    None,
                    "Term '{}' should NOT have a value after remove",
                    term
                );
            }

            // Verify remaining terms are still retrievable
            for (i, term) in unique_terms.iter().enumerate().skip(unique_terms.len() / 2) {
                prop_assert!(
                    dict.contains(term),
                    "Term '{}' should still be in dictionary after other removes",
                    term
                );
                prop_assert_eq!(
                    dict.get_value(term),
                    Some(i as i32),
                    "Term '{}' should still have value {}",
                    term,
                    i
                );
            }
        }
    }

    /// Property: len() matches actual count of items
    #[test]
    fn prop_len_matches_count(
        operations in operations_strategy()
    ) {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        {
            let mut dict = PersistentARTrie::<i32>::create(&path)
                .expect("create dict");

            // Track expected state manually
            let mut expected: HashMap<String, i32> = HashMap::new();

            for op in operations {
                match op {
                    Operation::Insert(term, value) => {
                        expected.insert(term.clone(), value);
                        let _ = dict.insert_with_value(&term, value);
                    }
                    Operation::Remove(term) => {
                        expected.remove(&term);
                        let _ = dict.remove(&term);
                    }
                }
            }

            // Verify length matches expected
            prop_assert_eq!(
                dict.len(),
                Some(expected.len()),
                "len() should match expected count"
            );

            // Double-check by iterating
            for (term, value) in &expected {
                prop_assert!(
                    dict.contains(term),
                    "Term '{}' should be in dictionary",
                    term
                );
                prop_assert_eq!(
                    dict.get_value(term),
                    Some(*value),
                    "Term '{}' should have value {}",
                    term,
                    value
                );
            }
        }
    }

    /// Property: Round-trip through checkpoint preserves data
    #[test]
    fn prop_checkpoint_roundtrip(
        terms in small_dict_with_values_strategy()
    ) {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        // Deduplicate and track expected state
        let expected: HashMap<String, i32> = terms.into_iter().collect();

        // Phase 1: Insert and checkpoint
        {
            let mut dict = PersistentARTrie::<i32>::create(&path)
                .expect("create dict");

            for (term, value) in &expected {
                let _ = dict.insert_with_value(term, *value);
            }

            dict.checkpoint().expect("checkpoint should succeed");
            dict.sync().expect("sync should succeed");
        }

        // Phase 2: Reopen and verify
        {
            let dict = PersistentARTrie::<i32>::open(&path)
                .expect("reopen dict");

            prop_assert_eq!(
                dict.len(),
                Some(expected.len()),
                "len() should match after reopen"
            );

            for (term, value) in &expected {
                prop_assert!(
                    dict.contains(term),
                    "Term '{}' should exist after checkpoint+reopen",
                    term
                );
                prop_assert_eq!(
                    dict.get_value(term),
                    Some(*value),
                    "Term '{}' should have value {} after checkpoint+reopen",
                    term,
                    value
                );
            }
        }
    }

    /// Property: WAL recovery preserves data without checkpoint
    #[test]
    fn prop_wal_recovery_roundtrip(
        terms in small_dict_with_values_strategy()
    ) {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        // Deduplicate and track expected state
        let expected: HashMap<String, i32> = terms.into_iter().collect();

        // Phase 1: Insert and sync (no checkpoint)
        {
            let mut dict = PersistentARTrie::<i32>::create(&path)
                .expect("create dict");

            for (term, value) in &expected {
                let _ = dict.insert_with_value(term, *value);
            }

            dict.sync().expect("sync should succeed");
        }

        // Phase 2: Reopen and verify (recovery from WAL)
        {
            let dict = PersistentARTrie::<i32>::open(&path)
                .expect("reopen dict");

            prop_assert_eq!(
                dict.len(),
                Some(expected.len()),
                "len() should match after WAL recovery"
            );

            for (term, value) in &expected {
                prop_assert!(
                    dict.contains(term),
                    "Term '{}' should exist after WAL recovery",
                    term
                );
                prop_assert_eq!(
                    dict.get_value(term),
                    Some(*value),
                    "Term '{}' should have value {} after WAL recovery",
                    term,
                    value
                );
            }
        }
    }

    /// Property: Byte sequence fuzzing - arbitrary bytes are handled correctly
    #[test]
    fn prop_byte_sequence_fuzzing(
        byte_terms in prop::collection::vec(byte_term_strategy(), 1..=30)
    ) {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        // Deduplicate byte sequences
        let unique_terms: Vec<_> = byte_terms.into_iter().collect::<HashSet<_>>().into_iter().collect();

        {
            let mut dict = PersistentARTrie::<i32>::create(&path)
                .expect("create dict");

            // Insert all byte sequences as UTF-8 strings (lossy conversion)
            let mut expected: HashMap<String, i32> = HashMap::new();
            for (i, bytes) in unique_terms.iter().enumerate() {
                let term = String::from_utf8_lossy(bytes).to_string();
                expected.insert(term.clone(), i as i32);
                let _ = dict.insert_with_value(&term, i as i32);
            }

            // Verify all terms are retrievable
            for (term, value) in &expected {
                prop_assert!(
                    dict.contains(term),
                    "Byte-derived term should be in dictionary"
                );
                prop_assert_eq!(
                    dict.get_value(term),
                    Some(*value),
                    "Byte-derived term should have correct value"
                );
            }
        }
    }

    /// Property: Large-scale operations with diverse prefixes (triggers ART splitting)
    #[test]
    fn prop_large_scale_diverse_prefixes(
        terms in medium_dict_strategy()
    ) {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        // Deduplicate
        let unique_terms: HashSet<_> = terms.iter().cloned().collect();

        {
            let mut dict = PersistentARTrie::<i32>::create(&path)
                .expect("create dict");

            // Insert all terms
            for (i, term) in unique_terms.iter().enumerate() {
                let _ = dict.insert_with_value(term, i as i32);
            }

            prop_assert_eq!(
                dict.len(),
                Some(unique_terms.len()),
                "len() should match unique term count"
            );

            // Verify all terms are retrievable
            for (i, term) in unique_terms.iter().enumerate() {
                prop_assert!(
                    dict.contains(term),
                    "Term '{}' should be in dictionary",
                    term
                );
                prop_assert_eq!(
                    dict.get_value(term),
                    Some(i as i32),
                    "Term '{}' should have correct value",
                    term
                );
            }
        }
    }

    /// Property: Insert-remove-reinsert preserves final state
    #[test]
    fn prop_insert_remove_reinsert(
        terms in prop::collection::vec(ascii_term_strategy(), 1..=20)
    ) {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        let unique_terms: Vec<_> = terms.into_iter().collect::<HashSet<_>>().into_iter().collect();

        {
            let mut dict = PersistentARTrie::<i32>::create(&path)
                .expect("create dict");

            // Phase 1: Insert with value 1
            for term in &unique_terms {
                let _ = dict.insert_with_value(term, 1);
            }

            // Phase 2: Remove all
            for term in &unique_terms {
                let _ = dict.remove(term);
            }

            // Should be empty
            prop_assert_eq!(dict.len(), Some(0), "Dictionary should be empty after removes");

            // Phase 3: Reinsert with value 2
            for term in &unique_terms {
                let _ = dict.insert_with_value(term, 2);
            }

            // All terms should have value 2
            for term in &unique_terms {
                prop_assert!(
                    dict.contains(term),
                    "Term '{}' should exist after reinsert",
                    term
                );
                prop_assert_eq!(
                    dict.get_value(term),
                    Some(2),
                    "Term '{}' should have value 2 after reinsert",
                    term
                );
            }
        }
    }

    /// Property: Multiple checkpoints maintain consistency
    #[test]
    fn prop_multiple_checkpoints(
        batches in prop::collection::vec(
            prop::collection::vec(ascii_term_strategy(), 1..=10),
            1..=5
        )
    ) {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        let mut all_terms: HashSet<String> = HashSet::new();

        {
            let mut dict = PersistentARTrie::<i32>::create(&path)
                .expect("create dict");

            for (batch_idx, batch) in batches.iter().enumerate() {
                // Insert batch
                for term in batch {
                    all_terms.insert(term.clone());
                    let _ = dict.insert_with_value(term, batch_idx as i32);
                }

                // Checkpoint after each batch
                dict.checkpoint().expect("checkpoint should succeed");

                // Verify all terms so far
                prop_assert_eq!(
                    dict.len(),
                    Some(all_terms.len()),
                    "len() should match after batch {} checkpoint",
                    batch_idx
                );
            }
        }

        // Reopen and verify final state
        {
            let dict = PersistentARTrie::<i32>::open(&path)
                .expect("reopen dict");

            prop_assert_eq!(
                dict.len(),
                Some(all_terms.len()),
                "len() should match after reopen"
            );

            for term in &all_terms {
                prop_assert!(
                    dict.contains(term),
                    "Term '{}' should exist after multiple checkpoints",
                    term
                );
            }
        }
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    /// Regression test: Empty dictionary should have len() = 0
    #[test]
    fn regression_empty_dict_len() {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        let dict = PersistentARTrie::<i32>::create(&path).expect("create dict");
        assert_eq!(dict.len(), Some(0));
    }

    /// Regression test: Single term insert/contains/remove cycle
    #[test]
    fn regression_single_term_lifecycle() {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        let mut dict = PersistentARTrie::<i32>::create(&path).expect("create dict");

        assert!(!dict.contains("test"));
        let _ = dict.insert_with_value("test", 42);
        assert!(dict.contains("test"));
        assert_eq!(dict.get_value("test"), Some(42));
        let _ = dict.remove("test");
        assert!(!dict.contains("test"));
        assert_eq!(dict.get_value("test"), None);
    }

    /// Regression test: Terms with common prefix
    #[test]
    fn regression_common_prefix_terms() {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        let mut dict = PersistentARTrie::<i32>::create(&path).expect("create dict");

        let _ = dict.insert_with_value("test", 1);
        let _ = dict.insert_with_value("testing", 2);
        let _ = dict.insert_with_value("tester", 3);
        let _ = dict.insert_with_value("tested", 4);

        assert_eq!(dict.len(), Some(4));
        assert_eq!(dict.get_value("test"), Some(1));
        assert_eq!(dict.get_value("testing"), Some(2));
        assert_eq!(dict.get_value("tester"), Some(3));
        assert_eq!(dict.get_value("tested"), Some(4));
    }

    /// Regression test: Value update (insert same key twice)
    #[test]
    fn regression_value_update() {
        let temp_dir = TempDir::new().expect("temp dir");
        let path = temp_dir.path().join("test_dict");

        let mut dict = PersistentARTrie::<i32>::create(&path).expect("create dict");

        let _ = dict.insert_with_value("key", 1);
        assert_eq!(dict.get_value("key"), Some(1));

        let _ = dict.insert_with_value("key", 2);
        assert_eq!(dict.get_value("key"), Some(2));

        // Length should still be 1 (same key, updated value)
        assert_eq!(dict.len(), Some(1));
    }
}
