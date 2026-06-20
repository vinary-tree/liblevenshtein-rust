# Query Fusion Completeness - Formal Proof Documentation

**Status**: Proof sketch documented; no checked contextual Rocq module exists yet
**Rocq target module**: `rocq/liblevenshtein/ContextualCompletion/Query.v`
**Date**: 2025-01-21
**Authors**: Formal Verification Team

---

## Overview

This theorem establishes that the `complete()` function correctly combines draft and finalized completions, returning the union of all matching terms visible from a context, with duplicates removed and drafts taking priority.

### Why This Matters

**User Impact**: This is the **core completion correctness theorem**. Without it:
- **Missing suggestions**: Terms visible in scope not returned
- **Wrong ordering**: Draft terms ranked below finalized (should override)
- **Duplicates**: Same term appears multiple times
- **Stale results**: Finalized term shown when draft exists

**Dependencies**: Combines all previous theorems:
- **Theorem 1**: Context visibility determines which scopes to query
- **Theorem 2**: Draft buffers provide valid UTF-8 terms
- **Theorem 7**: Finalized terms from dictionary

### Real-World Example

```rust
let engine = DynamicContextualCompletionEngine::new();
let global = engine.create_root_context(0);
let local = engine.create_child_context(1, global);

// Finalize "result" in global scope
engine.finalize_direct(global, "result");

// Type "res" in local scope draft
engine.insert_str(local, "res");

// Query from local context
let completions = engine.complete(local, "re", 1);

// Expected: ["res" (draft), "result" (finalized)]
// - "res" from local draft buffer ✓
// - "result" from global dictionary ✓
// - Both match query "re" with distance ≤ 1 ✓
// - No duplicates ✓
// - Draft ranks higher (appears first after sort) ✓
```

---

## Definitions

```coq
(* Result set - completions with their sources *)
Record Completion : Type := {
  term : string;
  distance : nat;
  source_context : ContextId;
  is_draft : bool
}.

(* Query fusion - union of draft + finalized results *)
Definition complete (tree : ContextTree) (drafts : DraftMap)
  (dict : Dictionary) (ctx : ContextId) (query : string) (max_dist : nat)
  : list Completion :=
  let visible := visible_contexts tree ctx in
  let draft_results := complete_drafts drafts visible query max_dist in
  let finalized_results := complete_finalized dict visible query max_dist in
  deduplicate (draft_results ++ finalized_results).

(* Deduplication - keep first occurrence (draft priority) *)
Fixpoint deduplicate (results : list Completion) : list Completion :=
  match results with
  | [] => []
  | r :: rest => r :: deduplicate (filter (fun x => x.(term) <> r.(term)) rest)
  end.
```

---

## Theorem Statement

```coq
Theorem query_fusion_correctness :
  forall (tree : ContextTree) (drafts : DraftMap) (dict : Dictionary)
         (ctx : ContextId) (query : string) (max_dist : nat),
    well_formed tree ->
    valid_drafts drafts ->
    valid_dictionary dict ->
    let results := complete tree drafts dict ctx query max_dist in
    (* Soundness: All results match query within distance *)
    (forall c, In c results ->
      levenshtein_distance query c.(term) <= max_dist) /\
    (* Completeness: All visible matching terms returned *)
    (forall term,
      (visible_and_matches tree drafts dict ctx term query max_dist) ->
      (exists c, In c results /\ c.(term) = term)) /\
    (* No duplicates *)
    (forall i j, i < j < length results ->
      nth i results default_completion .(term) <>
      nth j results default_completion .(term)) /\
    (* Draft priority: Drafts appear before finalized for same term *)
    (forall term,
      (exists draft_ctx, In term (draft_terms drafts draft_ctx)) ->
      (forall c, In c results /\ c.(term) = term -> c.(is_draft) = true)).
```

---

## Proof Sketch

**Strategy**: Set theory on result sets + deduplication properties.

**Main Steps**:

1. **Soundness**: By construction - both `complete_drafts` and `complete_finalized` filter by distance
2. **Completeness**: Union of draft + finalized covers all visible terms
3. **No Duplicates**: `deduplicate` removes all but first occurrence
4. **Draft Priority**: Draft results prepended to finalized, dedup keeps first

---

## Implementation Correspondence

**Source**: `src/contextual/engine.rs:1058-1087`

```rust
pub fn complete(&self, context: ContextId, query: &str, max_distance: usize)
  -> Vec<Completion> {
    let mut results = Vec::new();

    // Query drafts (visible contexts)
    results.extend(self.complete_drafts(context, query, max_distance));

    // Query finalized terms (dictionary)
    results.extend(self.complete_finalized(context, query, max_distance));

    // Deduplicate (draft overrides finalized)
    let mut seen = HashSet::new();
    results.retain(|c| seen.insert(c.term.clone()));

    results.sort();
    results
}
```

**Correspondence**:
- `visible_contexts` → Theorem 1 (Context Visibility)
- `complete_drafts` → Queries all visible draft buffers
- `complete_finalized` → Queries dictionary with visibility filter
- `HashSet dedup` → Keeps first occurrence (drafts come first)
- `sort()` → Orders by (distance, term)

---

## Test Coverage

**Unit Tests**: `src/contextual/engine.rs:1800-2000` (#[cfg(test)])

```rust
#[test]
fn test_complete_fusion() {
    let engine = Engine::new();
    let root = engine.create_root_context(0);

    engine.insert_str(root, "draft_term");
    engine.finalize_direct(root, "final_term");

    let results = engine.complete(root, "term", 2);
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|c| c.term == "draft_term" && c.is_draft));
    assert!(results.iter().any(|c| c.term == "final_term" && !c.is_draft));
}
```

---

## Related Theorems

**Depends on**:
- Theorem 1 (Context Visibility)
- Theorem 2 (Draft Consistency)

**Required by**:
- rholang-language-server completion handler

---

**Last Updated**: 2025-01-21
**Review trigger**: Reconcile this page when a checked contextual Rocq module is added.
