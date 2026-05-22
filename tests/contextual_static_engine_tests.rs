//! Tests for StaticContextualCompletionEngine.
//!
//! These tests cover:
//! - Basic completion functionality
//! - Draft buffer lifecycle (insert, delete, finalize)
//! - Context hierarchy creation
//! - Algorithm variants

#![cfg(feature = "pathmap-backend")]

use libdictenstein::double_array_trie::DoubleArrayTrieBuilder;
use liblevenshtein::contextual::{ContextId, StaticContextualCompletionEngine};
use liblevenshtein::transducer::Algorithm;

// ============================================================================
// Basic Construction Tests
// ============================================================================

#[test]
fn test_engine_construction() {
    let mut builder = DoubleArrayTrieBuilder::new();
    builder.insert_with_value("test", Some(vec![0]));
    let dict = builder.build();

    let _engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
}

#[test]
fn test_engine_with_empty_dictionary() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Should handle empty dictionary gracefully
    let results = engine.complete(ctx, "anything", 2);
    assert!(results.is_ok());
    assert!(results.unwrap().is_empty());
}

// ============================================================================
// Basic Completion Tests
// ============================================================================

#[test]
fn test_basic_completion_from_static_dict() {
    let mut builder = DoubleArrayTrieBuilder::new();
    // Add terms with context 0 visibility
    builder.insert_with_value("hello", Some(vec![0]));
    builder.insert_with_value("help", Some(vec![0]));
    builder.insert_with_value("world", Some(vec![0]));
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Query should find matches from static dictionary
    // "hel" -> "help" = 1 edit (insert 'p')
    // "hel" -> "hello" = 2 edits (insert 'l' and 'o')
    let results = engine.complete(ctx, "hel", 2).expect("complete failed");
    let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();

    assert!(terms.contains(&"hello"), "Should find 'hello'");
    assert!(terms.contains(&"help"), "Should find 'help'");
}

#[test]
fn test_completion_with_finalized_terms() {
    let mut builder = DoubleArrayTrieBuilder::new();
    builder.insert_with_value("static_term", Some(vec![0]));
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Finalize a new term using the proper workflow
    engine
        .insert_str(ctx, "dynamic_term")
        .expect("insert_str failed");
    engine.finalize(ctx).expect("finalize failed");

    // Should find finalized term
    // "dynamic" -> "dynamic_term" = 5 edits (insert '_term')
    let results = engine.complete(ctx, "dynamic", 5).expect("complete failed");
    let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();

    assert!(
        terms.contains(&"dynamic_term"),
        "Should find finalized term"
    );
}

#[test]
fn test_exact_match_completion() {
    let mut builder = DoubleArrayTrieBuilder::new();
    builder.insert_with_value("exact", Some(vec![0]));
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Exact match with distance 0
    let results = engine.complete(ctx, "exact", 0).expect("complete failed");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].term, "exact");
}

// ============================================================================
// Draft Buffer Lifecycle Tests
// ============================================================================

#[test]
fn test_draft_insert_chars() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Insert characters one by one
    engine.insert_char(ctx, 't').expect("insert failed");
    engine.insert_char(ctx, 'e').expect("insert failed");
    engine.insert_char(ctx, 's').expect("insert failed");
    engine.insert_char(ctx, 't').expect("insert failed");

    let draft = engine.get_draft(ctx).expect("get_draft failed");
    assert_eq!(draft, "test");
}

#[test]
fn test_draft_insert_string() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    engine.insert_str(ctx, "hello").expect("insert_str failed");

    let draft = engine.get_draft(ctx).expect("get_draft failed");
    assert_eq!(draft, "hello");
}

#[test]
fn test_draft_delete_char() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    engine.insert_str(ctx, "test").expect("insert_str failed");

    // Delete last character
    engine.delete_char(ctx).expect("delete_char failed");
    let draft = engine.get_draft(ctx).expect("get_draft failed");
    assert_eq!(draft, "tes");

    // Delete another
    engine.delete_char(ctx).expect("delete_char failed");
    let draft = engine.get_draft(ctx).expect("get_draft failed");
    assert_eq!(draft, "te");
}

#[test]
fn test_draft_clear() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    engine
        .insert_str(ctx, "content")
        .expect("insert_str failed");
    engine.clear_draft(ctx).expect("clear_draft failed");

    let draft = engine.get_draft(ctx).expect("get_draft failed");
    assert_eq!(draft, "");
}

#[test]
fn test_draft_finalize() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Build draft and finalize
    engine
        .insert_str(ctx, "newterm")
        .expect("insert_str failed");
    engine.finalize(ctx).expect("finalize failed");

    // Draft should be cleared after finalize
    let draft = engine.get_draft(ctx).expect("get_draft failed");
    assert_eq!(draft, "");

    // Term should now be searchable
    let results = engine.complete(ctx, "newterm", 0).expect("complete failed");
    assert!(
        results.iter().any(|c| c.term == "newterm"),
        "Finalized term should be searchable"
    );
}

// ============================================================================
// Context Hierarchy Tests
// ============================================================================

#[test]
fn test_create_root_context() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);

    let ctx1 = engine
        .create_root_context(0)
        .expect("create_root_context failed");
    let ctx2 = engine
        .create_root_context(1)
        .expect("create_root_context failed");

    assert_eq!(ctx1, 0);
    assert_eq!(ctx2, 1);
}

#[test]
fn test_create_child_context() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);

    let parent = engine
        .create_root_context(0)
        .expect("create_root_context failed");
    // API is create_child_context(child_id, parent_id)
    let child = engine
        .create_child_context(1, parent)
        .expect("create_child failed");

    assert_eq!(child, 1);
}

#[test]
fn test_deep_context_hierarchy() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);

    // Create a chain: root -> child -> grandchild
    let root = engine
        .create_root_context(0)
        .expect("create_root_context failed");
    let child = engine
        .create_child_context(1, root)
        .expect("create_child failed");
    let grandchild = engine
        .create_child_context(2, child)
        .expect("create_grandchild failed");

    // All contexts should be valid
    assert_eq!(root, 0);
    assert_eq!(child, 1);
    assert_eq!(grandchild, 2);
}

#[test]
fn test_context_visibility_inheritance() {
    let mut builder = DoubleArrayTrieBuilder::new();
    builder.insert_with_value("root_term", Some(vec![0]));
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);

    let root = engine
        .create_root_context(0)
        .expect("create_root_context failed");
    let child = engine
        .create_child_context(1, root)
        .expect("create_child failed");

    // Child should be able to find terms visible in root
    // "root" -> "root_term" = 5 edits (insert '_term')
    let results = engine.complete(child, "root", 5).expect("complete failed");
    assert!(
        results.iter().any(|c| c.term == "root_term"),
        "Child should see root's terms"
    );
}

// ============================================================================
// Algorithm Variant Tests
// ============================================================================

#[test]
fn test_transposition_algorithm() {
    let mut builder = DoubleArrayTrieBuilder::new();
    builder.insert_with_value("test", Some(vec![0]));
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Transposition);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // "tset" is "test" with 's' and 'e' transposed
    let results = engine.complete(ctx, "tset", 1).expect("complete failed");
    assert!(
        results.iter().any(|c| c.term == "test"),
        "Transposition algorithm should find 'test' from 'tset'"
    );
}

#[test]
fn test_merge_and_split_algorithm() {
    let mut builder = DoubleArrayTrieBuilder::new();
    builder.insert_with_value("hello", Some(vec![0]));
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::MergeAndSplit);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Should find hello with standard edits
    let results = engine.complete(ctx, "helo", 1).expect("complete failed");
    assert!(
        results.iter().any(|c| c.term == "hello"),
        "MergeAndSplit should find 'hello' from 'helo'"
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_create_child_of_nonexistent_parent() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);

    // Try to create child of non-existent parent (child_id, parent_id)
    let result = engine.create_child_context(1, 999);
    assert!(
        result.is_err(),
        "Should fail to create child of non-existent parent"
    );
}

#[test]
fn test_finalize_empty_draft() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Finalize with empty draft
    let result = engine.finalize(ctx);
    assert!(result.is_ok(), "Finalize should succeed with empty draft");
    assert_eq!(
        result.unwrap(),
        "",
        "Finalized empty draft should return empty string"
    );
}

#[test]
fn test_multiple_finalize_operations() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Finalize multiple terms
    engine.insert_str(ctx, "term1").expect("insert failed");
    engine.finalize(ctx).expect("finalize failed");

    engine.insert_str(ctx, "term2").expect("insert failed");
    engine.finalize(ctx).expect("finalize failed");

    engine.insert_str(ctx, "term3").expect("insert failed");
    engine.finalize(ctx).expect("finalize failed");

    // All terms should be searchable
    let results = engine.complete(ctx, "term", 1).expect("complete failed");
    let terms: Vec<&str> = results.iter().map(|c| c.term.as_str()).collect();

    assert!(terms.contains(&"term1"));
    assert!(terms.contains(&"term2"));
    assert!(terms.contains(&"term3"));
}

#[test]
fn test_draft_in_completion_results() {
    let builder = DoubleArrayTrieBuilder::<Vec<ContextId>>::new();
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Insert draft without finalizing
    engine.insert_str(ctx, "draft_term").expect("insert failed");

    // Draft should appear in completion results
    // "draft" -> "draft_term" = 5 edits (insert '_term')
    let results = engine.complete(ctx, "draft", 5).expect("complete failed");
    let draft_result = results.iter().find(|c| c.term == "draft_term");

    assert!(draft_result.is_some(), "Draft should appear in completion");
    assert!(
        draft_result.unwrap().is_draft,
        "Draft should be marked as draft"
    );
}

#[test]
fn test_clone_engine() {
    let mut builder = DoubleArrayTrieBuilder::new();
    builder.insert_with_value("test", Some(vec![0]));
    let dict = builder.build();

    let engine =
        StaticContextualCompletionEngine::with_double_array_trie(dict, Algorithm::Standard);
    let ctx = engine
        .create_root_context(0)
        .expect("create_root_context failed");

    // Clone the engine
    let engine_clone = engine.clone();

    // Both should share state (Arc-based cloning)
    engine.insert_str(ctx, "shared").expect("insert failed");

    let draft_original = engine.get_draft(ctx).expect("get_draft failed");
    let draft_clone = engine_clone.get_draft(ctx).expect("get_draft failed");

    assert_eq!(
        draft_original, draft_clone,
        "Cloned engine should share state"
    );
}
