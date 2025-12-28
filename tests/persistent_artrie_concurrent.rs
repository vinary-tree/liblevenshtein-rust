//! Concurrent access tests for PersistentARTrie.
//!
//! These tests verify thread-safety of the PersistentARTrie implementation:
//! - Multiple concurrent readers
//! - Single writer with multiple readers
//! - Concurrent transducer queries
//! - Reader during checkpoint operations
//!
//! # Architecture Notes
//!
//! PersistentARTrie uses `Arc<RwLock>` internally for thread-safety:
//! - Clone creates a shallow copy (same underlying data)
//! - Multiple clones can be passed to different threads
//! - RwLock ensures read/write safety
//!
//! # Known Limitations
//!
//! - Bucket capacity is 256 entries per bucket
//! - Tests stay within safe capacity limits
//! - Write operations are serialized by RwLock

#![cfg(feature = "persistent-artrie")]

use liblevenshtein::dictionary::persistent_artrie::PersistentARTrie;
use liblevenshtein::dictionary::Dictionary;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

/// Number of reader threads for concurrent tests.
const NUM_READERS: usize = 8;

/// Number of operations per thread.
const OPS_PER_THREAD: usize = 100;

/// Generate diverse terms for concurrent tests.
fn generate_terms(count: usize, prefix: &str) -> Vec<String> {
    (0..count).map(|i| format!("{}{:05}", prefix, i)).collect()
}

// =============================================================================
// Test: Multiple Concurrent Readers
// =============================================================================

#[test]
fn test_concurrent_readers() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("concurrent_readers.part");

    // Create and populate dictionary
    let mut dict: PersistentARTrie<i32> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    // Insert test terms with values
    let terms: Vec<String> = generate_terms(100, "term");
    for (i, term) in terms.iter().enumerate() {
        let _ = dict.insert_with_value(term, i as i32);
    }
    dict.sync().expect("sync");

    // Use a barrier to synchronize thread starts
    let barrier = Arc::new(Barrier::new(NUM_READERS));
    let terms_arc = Arc::new(terms);
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_READERS)
        .map(|_| {
            let dict_clone = dict.clone();
            let barrier_clone = barrier.clone();
            let terms_clone = terms_arc.clone();
            let success = success_count.clone();

            thread::spawn(move || {
                // Wait for all threads to be ready
                barrier_clone.wait();

                // Perform concurrent reads
                let mut local_success = 0;
                for term in terms_clone.iter() {
                    if dict_clone.contains(term) {
                        local_success += 1;
                    }
                }

                success.fetch_add(local_success, Ordering::SeqCst);
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("thread join");
    }

    // Each reader should find all 100 terms
    let total = success_count.load(Ordering::SeqCst);
    assert_eq!(
        total,
        NUM_READERS * 100,
        "All readers should find all terms"
    );
}

// =============================================================================
// Test: Single Writer with Multiple Readers
// =============================================================================

#[test]
fn test_single_writer_multiple_readers() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("writer_readers.part");

    // Create dictionary with initial terms
    let mut dict: PersistentARTrie<i32> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    // Insert some initial terms
    let initial_terms: Vec<String> = generate_terms(50, "init");
    for (i, term) in initial_terms.iter().enumerate() {
        let _ = dict.insert_with_value(term, i as i32);
    }
    dict.sync().expect("sync");

    let stop_flag = Arc::new(AtomicBool::new(false));
    let read_count = Arc::new(AtomicUsize::new(0));
    let terms_arc = Arc::new(initial_terms.clone());

    // Spawn reader threads
    let reader_handles: Vec<_> = (0..NUM_READERS)
        .map(|_| {
            let dict_clone = dict.clone();
            let stop = stop_flag.clone();
            let count = read_count.clone();
            let terms = terms_arc.clone();

            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    for term in terms.iter() {
                        // Read operations should succeed even during writes
                        let _ = dict_clone.contains(term);
                        count.fetch_add(1, Ordering::Relaxed);
                    }
                    thread::yield_now();
                }
            })
        })
        .collect();

    // Writer thread: insert new terms
    let new_terms: Vec<String> = generate_terms(50, "new_");
    let mut writer_dict = dict.clone();
    let writer_handle = thread::spawn(move || {
        for (i, term) in new_terms.iter().enumerate() {
            let _ = writer_dict.insert_with_value(term, (i + 1000) as i32);
            thread::sleep(Duration::from_micros(100));
        }
    });

    // Let it run for a short time
    thread::sleep(Duration::from_millis(100));

    // Stop readers and wait for writer
    stop_flag.store(true, Ordering::SeqCst);
    writer_handle.join().expect("writer join");

    for handle in reader_handles {
        handle.join().expect("reader join");
    }

    // Verify reads occurred
    let reads = read_count.load(Ordering::SeqCst);
    assert!(reads > 0, "Readers should have performed reads");

    // Verify all terms are present after writes complete
    for term in initial_terms.iter() {
        assert!(dict.contains(term), "Initial term should exist: {}", term);
    }
}

// =============================================================================
// Test: Concurrent Reads During Checkpoint
// =============================================================================

#[test]
fn test_reader_during_checkpoint() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("checkpoint_readers.part");

    // Create and populate dictionary
    let mut dict: PersistentARTrie<i32> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    let terms: Vec<String> = generate_terms(100, "chkp");
    for (i, term) in terms.iter().enumerate() {
        let _ = dict.insert_with_value(term, i as i32);
    }
    dict.sync().expect("sync");

    let stop_flag = Arc::new(AtomicBool::new(false));
    let read_errors = Arc::new(AtomicUsize::new(0));
    let terms_arc = Arc::new(terms.clone());

    // Spawn reader threads that continuously read during checkpoint
    let reader_handles: Vec<_> = (0..NUM_READERS)
        .map(|_| {
            let dict_clone = dict.clone();
            let stop = stop_flag.clone();
            let errors = read_errors.clone();
            let terms = terms_arc.clone();

            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    for term in terms.iter() {
                        if !dict_clone.contains(term) {
                            // Term should always be found (snapshot isolation)
                            errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
        })
        .collect();

    // Perform checkpoints while readers are active
    for _ in 0..3 {
        dict.checkpoint().expect("checkpoint");
        thread::sleep(Duration::from_millis(10));
    }

    // Stop readers
    stop_flag.store(true, Ordering::SeqCst);

    for handle in reader_handles {
        handle.join().expect("reader join");
    }

    // No read errors should occur
    let errors = read_errors.load(Ordering::SeqCst);
    assert_eq!(
        errors, 0,
        "No read errors should occur during checkpoints"
    );
}

// =============================================================================
// Test: Concurrent Value Lookups
// =============================================================================

#[test]
fn test_concurrent_value_lookups() {
    use liblevenshtein::dictionary::MappedDictionary;

    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("value_lookups.part");

    // Create and populate dictionary with values
    let mut dict: PersistentARTrie<i32> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    let terms: Vec<(String, i32)> = (0..100)
        .map(|i| (format!("value{:05}", i), i * 10))
        .collect();

    for (term, value) in &terms {
        let _ = dict.insert_with_value(term, *value);
    }
    dict.sync().expect("sync");

    let barrier = Arc::new(Barrier::new(NUM_READERS));
    let terms_arc = Arc::new(terms);
    let value_mismatches = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_READERS)
        .map(|_| {
            let dict_clone = dict.clone();
            let barrier_clone = barrier.clone();
            let terms_clone = terms_arc.clone();
            let mismatches = value_mismatches.clone();

            thread::spawn(move || {
                barrier_clone.wait();

                for (term, expected) in terms_clone.iter() {
                    if let Some(actual) = dict_clone.get_value(term) {
                        if actual != *expected {
                            mismatches.fetch_add(1, Ordering::Relaxed);
                        }
                    } else {
                        mismatches.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread join");
    }

    let mismatches = value_mismatches.load(Ordering::SeqCst);
    assert_eq!(mismatches, 0, "All values should match expected");
}

// =============================================================================
// Test: Writer Contention
// =============================================================================

#[test]
fn test_writer_contention() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("writer_contention.part");

    let dict: PersistentARTrie<i32> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    let barrier = Arc::new(Barrier::new(4));
    let successful_inserts = Arc::new(AtomicUsize::new(0));

    // Spawn 4 writer threads, each trying to insert different terms
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let mut dict_clone = dict.clone();
            let barrier_clone = barrier.clone();
            let inserts = successful_inserts.clone();

            thread::spawn(move || {
                barrier_clone.wait();

                // Each thread inserts terms with unique prefix
                let prefix = format!("t{}_", thread_id);
                for i in 0..25 {
                    let term = format!("{}{:03}", prefix, i);
                    if dict_clone.insert_with_value(&term, (thread_id * 100 + i) as i32) {
                        inserts.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread join");
    }

    // All 100 inserts should succeed (4 threads × 25 terms)
    let total = successful_inserts.load(Ordering::SeqCst);
    assert_eq!(total, 100, "All inserts should succeed");

    // Verify all terms are present
    for thread_id in 0..4 {
        for i in 0..25 {
            let term = format!("t{}_{:03}", thread_id, i);
            assert!(dict.contains(&term), "Term should exist: {}", term);
        }
    }
}

// =============================================================================
// Test: Read-Write Interleaving
// =============================================================================

#[test]
fn test_read_write_interleaving() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("interleaving.part");

    // Pre-populate with some terms
    let mut dict: PersistentARTrie<i32> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    let initial_terms: Vec<String> = generate_terms(50, "pre_");
    for (i, term) in initial_terms.iter().enumerate() {
        let _ = dict.insert_with_value(term, i as i32);
    }

    let operations = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));

    // Spawn interleaved reader/writer threads
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let mut dict_clone = dict.clone();
            let ops = operations.clone();
            let stop = stop_flag.clone();
            let terms = initial_terms.clone();

            thread::spawn(move || {
                let mut local_ops = 0;

                while !stop.load(Ordering::Relaxed) && local_ops < OPS_PER_THREAD {
                    // Alternate between reads and writes
                    if local_ops % 2 == 0 {
                        // Read operation
                        let term = &terms[local_ops % terms.len()];
                        let _ = dict_clone.contains(term);
                    } else {
                        // Write operation
                        let term = format!("new_t{}_{:04}", thread_id, local_ops);
                        let _ = dict_clone.insert_with_value(&term, local_ops as i32);
                    }

                    local_ops += 1;
                    ops.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // Let threads run
    thread::sleep(Duration::from_millis(50));
    stop_flag.store(true, Ordering::SeqCst);

    for handle in handles {
        handle.join().expect("thread join");
    }

    // Should have completed many operations
    let total_ops = operations.load(Ordering::SeqCst);
    assert!(total_ops > 0, "Operations should have been performed");

    // Original terms should still exist
    for term in &initial_terms {
        assert!(dict.contains(term), "Original term should exist: {}", term);
    }
}

// =============================================================================
// Test: Stress - Many Short-Lived Threads
// =============================================================================

#[test]
fn test_many_short_lived_threads() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("short_lived.part");

    let mut dict: PersistentARTrie<i32> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    // Pre-populate
    for i in 0..50 {
        let term = format!("base{:03}", i);
        let _ = dict.insert_with_value(&term, i);
    }
    dict.sync().expect("sync");

    let success_count = Arc::new(AtomicUsize::new(0));

    // Spawn many short-lived threads
    let handles: Vec<_> = (0..50)
        .map(|i| {
            let dict_clone = dict.clone();
            let success = success_count.clone();

            thread::spawn(move || {
                // Each thread does a few operations then exits
                let term = format!("base{:03}", i % 50);
                if dict_clone.contains(&term) {
                    success.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread join");
    }

    let successes = success_count.load(Ordering::SeqCst);
    assert_eq!(successes, 50, "All lookups should succeed");
}

// =============================================================================
// Test: Concurrent Opens of Same Dictionary (should fail)
// =============================================================================

#[test]
fn test_concurrent_opens_same_path() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("same_path.part");

    // Create the dictionary
    let _dict: PersistentARTrie<()> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    // Try to create another dictionary at the same path - should fail
    let result: Result<PersistentARTrie<()>, _> = PersistentARTrie::create(&dict_path);
    assert!(
        result.is_err(),
        "Creating another dictionary at same path should fail"
    );
}

// =============================================================================
// Test: Clone Shares State
// =============================================================================

#[test]
fn test_clone_shares_state() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("shared_state.part");

    let mut dict: PersistentARTrie<i32> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    // Insert via original handle
    let _ = dict.insert_with_value("hello", 42);

    // Clone should see the insert
    let dict_clone = dict.clone();
    assert!(
        dict_clone.contains("hello"),
        "Clone should see original insert"
    );

    // Insert via clone
    let mut dict_clone2 = dict.clone();
    let _ = dict_clone2.insert_with_value("world", 100);

    // Original should see clone's insert
    assert!(dict.contains("world"), "Original should see clone's insert");
}

// =============================================================================
// Test: Sync From Multiple Threads
// =============================================================================

#[test]
fn test_sync_from_multiple_threads() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let dict_path = temp_dir.path().join("multi_sync.part");

    let mut dict: PersistentARTrie<i32> =
        PersistentARTrie::create(&dict_path).expect("create dict");

    // Insert some data
    for i in 0..50 {
        let _ = dict.insert_with_value(&format!("sync{:03}", i), i);
    }

    let barrier = Arc::new(Barrier::new(4));
    let sync_errors = Arc::new(AtomicUsize::new(0));

    // Multiple threads calling sync concurrently
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let dict_clone = dict.clone();
            let barrier_clone = barrier.clone();
            let errors = sync_errors.clone();

            thread::spawn(move || {
                barrier_clone.wait();

                // Multiple sync calls should be safe
                for _ in 0..5 {
                    if dict_clone.sync().is_err() {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread join");
    }

    let errors = sync_errors.load(Ordering::SeqCst);
    assert_eq!(errors, 0, "All sync calls should succeed");
}

// =============================================================================
// Test: Transducer Queries (if available)
// =============================================================================

// Note: Concurrent transducer tests require the transducer module.
// This is a placeholder for when that integration is needed.
#[test]
#[ignore = "Transducer concurrent tests require transducer module integration"]
fn test_concurrent_transducer_queries() {
    // TODO: Add transducer concurrent query tests when needed
    // This would test multiple threads querying with Levenshtein automata
}
