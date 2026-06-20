//! # Contextual Completion Engine Example
//!
//! This example demonstrates the `ContextualCompletionEngine` for incremental
//! code completion with draft management, checkpoints, and hierarchical scopes.
//!
//! ## Features Demonstrated
//!
//! - **Character-level insertion**: Build identifiers incrementally
//! - **Draft management**: In-progress text per context
//! - **Checkpoints**: Create undo points and restore previous states
//! - **Hierarchical contexts**: Child scopes see parent scope identifiers
//! - **Query fusion**: Search both drafts and finalized terms
//! - **Fuzzy matching**: Efficient automaton-based Levenshtein distance
//!
//! ## Example Scenario
//!
//! Simulates typing in a code editor with multiple nested scopes:
//!
//! ```text
//! // Global scope (id=0)
//! let global_var = 1;
//! let global_helper = 2;
//!
//! fn process_data() {  // Function scope (id=1)
//!     let result = 3;
//!
//!     {  // Block scope (id=2)
//!         let local_var = 5;
//!
//!         // User types "hel" here...
//!         // Should see: global_helper (finalized), hello (draft if typing it)
//!     }
//! }
//! ```
//!
//! ## Running this Example
//!
//! ```bash
//! cargo run --example contextual_completion --features pathmap-backend
//! ```

use liblevenshtein::contextual::DynamicContextualCompletionEngine;
use liblevenshtein::transducer::Algorithm;

fn main() {
    println!("Contextual Completion Engine Demo");
    println!("==================================\n");

    // Create the engine with Standard Levenshtein distance
    let engine = DynamicContextualCompletionEngine::with_algorithm(Algorithm::Standard);

    // Create hierarchical context tree: global -> function -> block
    let global = engine.create_root_context(0);
    let function = engine.create_child_context(1, global).unwrap();
    let block = engine.create_child_context(2, function).unwrap();

    println!("Created context hierarchy:");
    println!("  global (0) -> function (1) -> block (2)\n");

    // Scenario 1: Populate global scope with finalized terms
    println!("Scenario 1: Adding global identifiers");
    println!("--------------------------------------");
    engine.finalize_direct(global, "global_var").unwrap();
    engine.finalize_direct(global, "global_helper").unwrap();
    engine.finalize_direct(global, "helper_fn").unwrap();

    println!("Finalized: global_var, global_helper, helper_fn");
    println!();

    // Scenario 2: Add function-scoped identifiers
    println!("Scenario 2: Adding function-scoped identifiers");
    println!("-----------------------------------------------");
    engine.finalize_direct(function, "result").unwrap();
    engine.finalize_direct(function, "process").unwrap();

    println!("Finalized: result, process");
    println!();

    // Scenario 3: Incremental typing with draft
    println!("Scenario 3: Typing 'hello' incrementally in block scope");
    println!("--------------------------------------------------------");

    // Type 'h'
    engine.insert_char(block, 'h').unwrap();
    println!("Typed: 'h'");
    println!("Current draft: {:?}", engine.get_draft(block).unwrap());

    // Type 'e'
    engine.insert_char(block, 'e').unwrap();
    println!("Typed: 'he'");
    println!("Current draft: {:?}", engine.get_draft(block).unwrap());

    // Query for completions (should match global_helper)
    let completions = engine.complete(block, "hel", 2);
    println!("\nCompletions for 'hel' (max distance 2):");
    for comp in &completions {
        println!(
            "  - {} (distance: {}, draft: {})",
            comp.term, comp.distance, comp.is_draft
        );
    }
    println!();

    // Scenario 4: Checkpoint and undo
    println!("Scenario 4: Using checkpoints for undo");
    println!("---------------------------------------");

    // Finish typing 'hello'
    engine.insert_char(block, 'l').unwrap();
    println!("Typed: 'hel'");

    // Create checkpoint
    engine.checkpoint(block).unwrap();
    println!("Created checkpoint at 'hel'");

    // Type more characters
    engine.insert_char(block, 'l').unwrap();
    engine.insert_char(block, 'o').unwrap();
    println!("Typed: 'hello'");
    println!("Current draft: {:?}", engine.get_draft(block).unwrap());

    // Undo to checkpoint
    engine.undo(block).unwrap();
    println!("Undid to checkpoint");
    println!(
        "Current draft after undo: {:?}",
        engine.get_draft(block).unwrap()
    );
    println!();

    // Scenario 5: Finalize draft
    println!("Scenario 5: Finalizing draft");
    println!("-----------------------------");

    // Type 'lo' again to complete 'hello'
    engine.insert_char(block, 'l').unwrap();
    engine.insert_char(block, 'o').unwrap();

    let term = engine.finalize(block).unwrap();
    println!("Finalized draft: '{}'", term);
    println!("Draft cleared: {}", !engine.has_draft(block));
    println!();

    // Scenario 6: Hierarchical visibility
    println!("Scenario 6: Hierarchical scope visibility");
    println!("------------------------------------------");

    // Query from block scope (should see global, function, and block terms)
    let block_completions = engine.complete(block, "help", 2);
    println!("Completions from block scope (query: 'help', distance ≤ 2):");
    if block_completions.is_empty() {
        println!("  (no matches)");
    } else {
        for comp in &block_completions {
            println!(
                "  - {} (distance: {}, contexts: {:?})",
                comp.term, comp.distance, comp.contexts
            );
        }
    }

    // Query from global scope (should NOT see function/block terms)
    let global_completions = engine.complete(global, "hello", 1);
    println!("\nCompletions from global scope (query: 'hello', distance ≤ 1):");
    if global_completions.is_empty() {
        println!("  (no matches - 'hello' is in block scope, not visible from global)");
    } else {
        for comp in &global_completions {
            println!(
                "  - {} (distance: {}, contexts: {:?})",
                comp.term, comp.distance, comp.contexts
            );
        }
    }

    let block_completions2 = engine.complete(block, "hello", 1);
    println!("\nCompletions from block scope (query: 'hello', distance ≤ 1):");
    if block_completions2.is_empty() {
        println!("  (no matches)");
    } else {
        for comp in &block_completions2 {
            println!(
                "  - {} (distance: {}, contexts: {:?})",
                comp.term, comp.distance, comp.contexts
            );
        }
    }
    println!("\nScope rule: children see parent terms, but parents don't see children");

    // Scenario 7: Draft override
    println!("Scenario 7: Draft overrides finalized");
    println!("--------------------------------------");

    // We have 'hello' finalized in block scope
    // Now create a draft with the same name
    engine.insert_str(block, "hello_world").unwrap();

    let completions = engine.complete(block, "hello", 2);
    println!("Completions for 'hello' (with draft 'hello_world'):");
    for comp in &completions {
        println!(
            "  - {} (distance: {}, draft: {})",
            comp.term, comp.distance, comp.is_draft
        );
    }

    // The draft version should appear first due to sorting
    println!("\nDrafts are deduplicated - only one 'hello*' appears");

    // Clean up
    engine.discard(block).unwrap();
    println!("Discarded draft");
    println!();

    // Summary
    println!("\nSummary");
    println!("=======");
    println!("Terms in dictionary:");
    println!("  Global scope (0): global_var, global_helper, helper_fn");
    println!("  Function scope (1): result, process");
    println!("  Block scope (2): hello");
    println!("\nVisible from block scope (2): all 6 terms");
    println!("Visible from function scope (1): 5 terms (not 'hello')");
    println!("Visible from global scope (0): 3 terms (only global)");

    println!("\n\nPerformance Details");
    println!("=================");
    println!("- Draft matching: O(n*m) naive Levenshtein (small n)");
    println!("- Finalized matching: O(k) automaton-based (k = matches)");
    println!("- Context visibility: O(depth) tree traversal");
    println!("- Checkpoint storage: 8 bytes per checkpoint");
    println!("- Thread-safe: Arc<Mutex> for drafts, Arc<RwLock> for dictionary");
}
