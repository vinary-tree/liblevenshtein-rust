//! Stress tests for Contextual Completion Engine
//!
//! These tests verify correctness and stability under extreme conditions:
//! - Large numbers of contexts (10K+)
//! - Large dictionaries (100K+ terms)
//! - Deep hierarchies (100+ levels)
//! - Many checkpoints (1K+)
//! - Long-running operations

#[cfg(feature = "pathmap-backend")]
mod stress_tests {
    use liblevenshtein::contextual::DynamicContextualCompletionEngine;

    /// Test with a large number of root contexts
    #[test]
    #[ignore] // Run with: cargo test --features pathmap-backend stress -- --ignored
    fn test_many_contexts() {
        let engine = DynamicContextualCompletionEngine::new();

        // Create 10,000 root contexts
        for i in 0..10_000 {
            let ctx = engine.create_root_context(i);
            assert_eq!(ctx, i);
            assert!(engine.context_exists(ctx));
        }

        // Verify they all exist
        for i in 0..10_000 {
            assert!(engine.context_exists(i));
        }

        // Add a term to each
        for i in 0..10_000 {
            engine.finalize_direct(i, &format!("term_{}", i)).unwrap();
        }

        // Query from random contexts
        for i in (0..10_000).step_by(100) {
            let results = engine.complete(i, &format!("term_{}", i), 0);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].term, format!("term_{}", i));
        }
    }

    /// Test with a large dictionary
    #[test]
    #[ignore]
    fn test_large_dictionary() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        // Add 100,000 terms
        println!("Adding 100,000 terms...");
        for i in 0..100_000 {
            let term = format!("identifier_{:06}", i);
            engine.finalize_direct(ctx, &term).unwrap();

            if i % 10_000 == 0 {
                println!("  Added {} terms", i);
            }
        }

        // Add some matching terms
        engine.finalize_direct(ctx, "helper").unwrap();
        engine.finalize_direct(ctx, "helper_fn").unwrap();
        engine.finalize_direct(ctx, "global_helper").unwrap();

        // Query should still be fast
        println!("Querying...");
        let results = engine.complete(ctx, "help", 2);
        assert!(results.len() >= 3);

        // Verify specific terms are found
        let helper_found = results.iter().any(|c| c.term == "helper");
        assert!(helper_found, "Should find 'helper'");
    }

    /// Test with deep hierarchy
    #[test]
    #[ignore]
    fn test_deep_hierarchy() {
        let engine = DynamicContextualCompletionEngine::new();

        // Create a hierarchy 1000 levels deep
        println!("Creating 1000-level hierarchy...");
        let root = engine.create_root_context(0);
        let mut current = root;

        for i in 1..1_000 {
            current = engine.create_child_context(i, current).unwrap();

            if i % 100 == 0 {
                println!("  Created {} levels", i);
            }
        }

        // Add terms at various levels
        engine.finalize_direct(0, "root_term").unwrap();
        engine.finalize_direct(500, "mid_term").unwrap();
        engine.finalize_direct(999, "leaf_term").unwrap();

        // Query from leaf should see all ancestor terms
        println!("Querying from leaf...");
        let visible = engine.get_visible_contexts(999);
        assert_eq!(visible.len(), 1000); // Should see all ancestors

        // Query should find root terms
        let results = engine.complete(999, "root", 2);
        assert!(results.iter().any(|c| c.term == "root_term"));
    }

    /// Test with many checkpoints
    #[test]
    #[ignore]
    fn test_many_checkpoints() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        // Create 10,000 checkpoints
        println!("Creating 10,000 checkpoints...");
        for i in 0..10_000 {
            engine.insert_char(ctx, 'a').unwrap();
            engine.checkpoint(ctx).unwrap();

            if i % 1_000 == 0 {
                println!("  Created {} checkpoints", i);
            }
        }

        assert_eq!(engine.checkpoint_count(ctx), 10_000);

        // Undo 5,000 times
        println!("Undoing 5,000 times...");
        for i in 0..5_000 {
            engine.undo(ctx).unwrap();

            if i % 1_000 == 0 {
                println!("  Undone {} times", i);
            }
        }

        // Should have 5,000 characters left
        let draft = engine.get_draft(ctx).unwrap();
        assert_eq!(draft.len(), 5_000);
    }

    /// Test with wide fan-out
    #[test]
    #[ignore]
    fn test_wide_fanout() {
        let engine = DynamicContextualCompletionEngine::new();
        let root = engine.create_root_context(0);

        // Create 1,000 children of root
        println!("Creating 1,000 children...");
        let mut children = vec![];
        for i in 1..=1_000 {
            let child = engine.create_child_context(i, root).unwrap();
            children.push(child);

            if i % 100 == 0 {
                println!("  Created {} children", i);
            }
        }

        // Add a term to root
        engine.finalize_direct(root, "global_var").unwrap();

        // All children should see the root term
        println!("Verifying visibility...");
        for (idx, &child) in children.iter().enumerate() {
            // Check that the child can see the root context
            let visible = engine.get_visible_contexts(child);
            assert!(
                visible.contains(&root),
                "Child {} should have root in visible contexts",
                idx
            );

            // Query for the exact term
            let results = engine.complete(child, "global_var", 0);
            assert!(
                results.iter().any(|c| c.term == "global_var"),
                "Child {} should see global_var",
                idx
            );

            if idx % 100 == 0 {
                println!("  Verified {} children", idx);
            }
        }
    }

    /// Test long-running drafts with many edits
    #[test]
    #[ignore]
    fn test_long_draft_session() {
        let engine = DynamicContextualCompletionEngine::new();
        let ctx = engine.create_root_context(0);

        // Simulate a long editing session with 100,000 operations
        println!("Simulating 100,000 edit operations...");
        for i in 0..100_000 {
            match i % 4 {
                0 => {
                    // Insert
                    engine.insert_char(ctx, 'a').unwrap();
                }
                1 => {
                    // Checkpoint
                    engine.checkpoint(ctx).unwrap();
                }
                2 => {
                    // Delete
                    engine.delete_char(ctx);
                }
                3 => {
                    // Undo (may fail if no checkpoints)
                    let _ = engine.undo(ctx);
                }
                unexpected => panic!("i % 4 produced unexpected remainder {unexpected}"),
            }

            if i % 10_000 == 0 {
                println!("  Completed {} operations", i);
            }
        }

        // Engine should still be functional
        let _ = engine.get_draft(ctx);
        engine.clear_draft(ctx).unwrap();
        engine.clear_checkpoints(ctx).unwrap();

        // Can still insert after all that
        engine.insert_str(ctx, "still_works").unwrap();
        let draft = engine.get_draft(ctx).unwrap();
        assert_eq!(draft, "still_works");
    }

    /// Test removal of large subtrees
    #[test]
    #[ignore]
    fn test_large_subtree_removal() {
        let engine = DynamicContextualCompletionEngine::new();
        let root = engine.create_root_context(0);

        // Create a balanced tree: root -> 10 children -> 100 grandchildren each
        println!("Creating tree with 1,010 nodes...");
        let mut all_nodes = vec![root];

        for i in 1..=10 {
            let child = engine.create_child_context(i, root).unwrap();
            all_nodes.push(child);

            for j in 1..=100 {
                let id = i * 1000 + j;
                let grandchild = engine.create_child_context(id, child).unwrap();
                all_nodes.push(grandchild);
            }
        }

        println!("Created {} nodes", all_nodes.len());
        assert_eq!(all_nodes.len(), 1011); // root + 10 children + 1000 grandchildren

        // Remove one child (should remove it and its 100 descendants)
        println!("Removing subtree...");
        assert!(engine.remove_context(1));

        // Verify removal
        assert!(!engine.context_exists(1));
        for j in 1..=100 {
            assert!(!engine.context_exists(1000 + j));
        }

        // Other subtrees should still exist
        assert!(engine.context_exists(2));
        assert!(engine.context_exists(2001));
    }

    /// Test concurrent stress (multiple threads)
    #[test]
    #[ignore]
    fn test_concurrent_stress() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(DynamicContextualCompletionEngine::new());
        let root = engine.create_root_context(0);

        // Populate dictionary
        for i in 0..1_000 {
            engine
                .finalize_direct(root, &format!("term_{}", i))
                .unwrap();
        }

        println!("Running concurrent stress test with 16 threads...");
        let mut handles = vec![];

        for thread_id in 0..16 {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                let ctx = engine_clone.create_root_context(1000 + thread_id);

                // Each thread performs 1,000 operations
                for i in 0..1_000 {
                    match i % 3 {
                        0 => {
                            // Insert
                            engine_clone
                                .insert_str(ctx, &format!("draft_{}", i))
                                .unwrap();
                            engine_clone.clear_draft(ctx).unwrap();
                        }
                        1 => {
                            // Query
                            let results = engine_clone.complete(root, "term", 2);
                            assert!(!results.is_empty());
                        }
                        2 => {
                            // Finalize
                            engine_clone
                                .finalize_direct(ctx, &format!("finalized_{}", i))
                                .unwrap();
                        }
                        unexpected => panic!("i % 3 produced unexpected remainder {unexpected}"),
                    }
                }

                println!("  Thread {} completed", thread_id);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        println!("Concurrent stress test completed successfully");
    }

    /// Test memory stability (no leaks during rapid alloc/dealloc)
    #[test]
    #[ignore]
    fn test_memory_stability() {
        println!("Testing memory stability with rapid create/destroy cycles...");

        for cycle in 0..100 {
            let engine = DynamicContextualCompletionEngine::new();

            // Create many contexts
            for i in 0..1_000 {
                engine.create_root_context(i);
            }

            // Add drafts
            for i in 0..1_000 {
                engine.insert_str(i, "test_data").unwrap();
            }

            // Create checkpoints
            for i in 0..1_000 {
                engine.checkpoint(i).unwrap();
            }

            // Remove half the contexts
            for i in (0..1_000).step_by(2) {
                engine.remove_context(i);
            }

            // Engine is dropped here
            if cycle % 10 == 0 {
                println!("  Completed {} cycles", cycle);
            }
        }

        println!("Memory stability test completed");
    }
}
