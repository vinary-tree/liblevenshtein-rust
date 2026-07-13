# Hierarchical Visibility Soundness - Formal Proof Documentation

**Status**: Proof sketch documented; no checked contextual Rocq module exists yet
**Rocq target module**: `rocq/liblevenshtein/ContextualCompletion/Visibility.v`
**Date**: 2025-01-21
**Authors**: Formal Verification Team

---

## Overview

This theorem establishes that hierarchical context visibility correctly implements lexical scoping: child contexts can see symbols from parent contexts (outer scopes), but parent contexts **cannot** see symbols from child contexts (inner scopes).

### Why This Matters

**User Impact**: This is the **scope isolation correctness theorem**. Without it:
- **Privacy violations**: Inner scope symbols leak to outer scopes
- **Name pollution**: Local variables visible globally
- **Wrong completions**: Irrelevant symbols suggested
- **Broken encapsulation**: Function-local definitions visible at module level

**Dependencies**: Built on:
- **Theorem 1**: Context visibility (ancestor chain traversal)
- Used by:
  - **Theorem 4**: Query fusion (which contexts to query)
  - LSP scope detection (symbol resolution)

### Real-World Example

```rust
let engine = DynamicContextualCompletionEngine::new();

// Create scope hierarchy: global → module → function
let global = engine.create_root_context(0);
let module = engine.create_child_context(1, global).unwrap();
let function = engine.create_child_context(2, module).unwrap();

// Define symbols at different scopes
engine.finalize_direct(global, "global_var").unwrap();
engine.finalize_direct(module, "module_var").unwrap();
engine.finalize_direct(function, "local_var").unwrap();

// Query from function (child) - should see ALL ancestors
let results = engine.complete_finalized(function, "var", 10);
assert!(results.iter().any(|c| c.term == "global_var"));   // ✓ Sees global
assert!(results.iter().any(|c| c.term == "module_var"));   // ✓ Sees module
assert!(results.iter().any(|c| c.term == "local_var"));    // ✓ Sees self

// Query from global (parent) - should NOT see descendants
let results = engine.complete_finalized(global, "var", 10);
assert!(results.iter().any(|c| c.term == "global_var"));   // ✓ Sees self
assert!(!results.iter().any(|c| c.term == "module_var"));  // ✗ Does not see module
assert!(!results.iter().any(|c| c.term == "local_var"));   // ✗ Does not see function

// VIOLATION EXAMPLE (if theorem was false):
// Without this theorem, "local_var" could leak to global scope!
// This would break encapsulation and create security issues.
```

---

## Definitions

```coq
(* Ancestor relationship (from Theorem 1) *)
Fixpoint is_ancestor (tree : ContextTree) (ancestor child : ContextId) : bool :=
  match parent tree child with
  | None => false                                   (* child is root *)
  | Some p => if p = ancestor then true            (* direct parent *)
              else is_ancestor tree ancestor p      (* recursive check *)
  end.

(* Descendant relationship (inverse of ancestor) *)
Definition is_descendant (tree : ContextTree) (descendant ancestor : ContextId) : bool :=
  is_ancestor tree ancestor descendant.

(* Visibility predicate: ctx1 visible from ctx2 *)
Definition visible_from (tree : ContextTree) (ctx1 ctx2 : ContextId) : Prop :=
  ctx1 = ctx2 \/                              (* Same context *)
  is_ancestor tree ctx1 ctx2 = true.          (* ctx1 is ancestor of ctx2 *)

(* Invisible predicate: ctx1 NOT visible from ctx2 *)
Definition invisible_from (tree : ContextTree) (ctx1 ctx2 : ContextId) : Prop :=
  ctx1 <> ctx2 /\
  is_ancestor tree ctx1 ctx2 = false.
```

---

## Theorem Statement

```coq
Theorem hierarchical_visibility_soundness :
  forall (tree : ContextTree) (parent child sibling : ContextId),
    well_formed tree ->
    is_parent tree parent child ->
    is_parent tree parent sibling ->
    child <> sibling ->
    (* Part 1: Children see parents (upward visibility) *)
    visible_from tree parent child /\
    (* Part 2: Parents don't see children (no downward visibility) *)
    invisible_from tree child parent /\
    (* Part 3: Siblings don't see each other (no lateral visibility) *)
    invisible_from tree sibling child /\
    (* Part 4: Visibility is transitive upward *)
    (forall ancestor, is_ancestor tree ancestor child = true ->
                      visible_from tree ancestor child) /\
    (* Part 5: Invisibility is transitive downward *)
    (forall descendant, is_descendant tree descendant parent = true ->
                        descendant <> child ->
                        invisible_from tree descendant parent).
```

---

## Proof Sketch

**Strategy**: Prove visibility properties by case analysis on context relationships.

### Part 1: Children See Parents (Upward Visibility)

**Claim**: If `child` is a child of `parent`, then `parent` is visible from `child`.

**Proof**:
1. By definition: `is_parent tree parent child` means `parent(child) = Some(parent)`
2. By definition of `visible_from`: `visible_from tree parent child` iff `parent = child` OR `is_ancestor tree parent child = true`
3. Since `child <> parent` (well-formedness: no self-loops)
4. We need to show: `is_ancestor tree parent child = true`
5. By definition of `is_ancestor`:
   ```
   is_ancestor tree parent child
   = match parent(child) with
     | Some p => if p = parent then true else ...
   ```
6. Since `parent(child) = Some(parent)`, the `if` condition succeeds
7. Therefore: `is_ancestor tree parent child = true`
8. Therefore: `visible_from tree parent child` ✓ `$\blacksquare$`

### Part 2: Parents Don't See Children (No Downward Visibility)

**Claim**: If `child` is a child of `parent`, then `child` is NOT visible from `parent`.

**Proof**:
1. By definition: `invisible_from tree child parent` iff `child <> parent` AND `is_ancestor tree child parent = false`
2. **Case 1**: `child = parent`
   - Contradicts well-formedness (no cycles)
   - Therefore `child <> parent` ✓
3. **Case 2**: Show `is_ancestor tree child parent = false`
   - Assume for contradiction: `is_ancestor tree child parent = true`
   - Then there exists a path: `parent → ... → child → ... → parent` (cycle!)
   - This contradicts well-formedness (trees are acyclic)
   - Therefore `is_ancestor tree child parent = false` ✓
4. Therefore: `invisible_from tree child parent` ✓ `$\blacksquare$`

### Part 3: Siblings Don't See Each Other (No Lateral Visibility)

**Claim**: If `sibling` and `child` are distinct children of `parent`, then `sibling` is NOT visible from `child`.

**Proof**:
1. By definition: `invisible_from tree sibling child` iff `sibling <> child` AND `is_ancestor tree sibling child = false`
2. Given: `sibling <> child` (by assumption) ✓
3. **Show**: `is_ancestor tree sibling child = false`
   - By definition: `is_ancestor tree sibling child` iff `sibling` is on the path from `child` to root
   - Path from `child` to root: `child → parent → ... → root`
   - Path from `sibling` to root: `sibling → parent → ... → root`
   - These paths **diverge** at `parent` (sibling is NOT on child's path)
   - Therefore: `is_ancestor tree sibling child = false` ✓
4. Therefore: `invisible_from tree sibling child` ✓ `$\blacksquare$`

### Part 4: Visibility is Transitive Upward

**Claim**: If `ancestor` is an ancestor of `child`, then `ancestor` is visible from `child`.

**Proof**:
1. Given: `is_ancestor tree ancestor child = true`
2. By definition of `visible_from`: `visible_from tree ancestor child` iff `ancestor = child` OR `is_ancestor tree ancestor child = true`
3. Since we have `is_ancestor tree ancestor child = true`, the second disjunct holds
4. Therefore: `visible_from tree ancestor child` ✓ `$\blacksquare$`

### Part 5: Invisibility is Transitive Downward

**Claim**: If `descendant` is a descendant of `parent` (but not equal to `child`), then `descendant` is NOT visible from `parent`.

**Proof**:
1. Given: `is_descendant tree descendant parent = true` AND `descendant <> child`
2. By definition: `is_descendant tree descendant parent` = `is_ancestor tree parent descendant`
3. So we have: `is_ancestor tree parent descendant = true`
4. **Show**: `invisible_from tree descendant parent`
   - Need: `descendant <> parent` (given by well-formedness: ancestors `$\ne$` descendants)
   - Need: `is_ancestor tree descendant parent = false`
   - Assume for contradiction: `is_ancestor tree descendant parent = true`
   - Then: `is_ancestor tree parent descendant = true` AND `is_ancestor tree descendant parent = true`
   - This creates a cycle: `parent → ... → descendant → ... → parent`
   - Contradicts well-formedness (acyclic tree)
   - Therefore: `is_ancestor tree descendant parent = false` ✓
5. Therefore: `invisible_from tree descendant parent` ✓ `$\blacksquare$`

---

## Key Lemmas

**Lemma 1** (Acyclic Tree):
```coq
Lemma tree_acyclic :
  forall (tree : ContextTree) (ctx : ContextId),
    well_formed tree ->
    is_ancestor tree ctx ctx = false.
```

**Proof**: By contradiction. If `is_ancestor tree ctx ctx = true`, then there exists a path `ctx → ... → ctx`, violating well-formedness.

**Lemma 2** (Ancestor Asymmetry):
```coq
Lemma ancestor_asymmetry :
  forall (tree : ContextTree) (ctx1 ctx2 : ContextId),
    well_formed tree ->
    is_ancestor tree ctx1 ctx2 = true ->
    is_ancestor tree ctx2 ctx1 = false.
```

**Proof**: Follows from Lemma 1 (acyclic) + transitivity. If both were true, we'd have a cycle.

**Lemma 3** (Path Divergence):
```coq
Lemma sibling_paths_diverge :
  forall (tree : ContextTree) (parent child1 child2 : ContextId),
    well_formed tree ->
    is_parent tree parent child1 ->
    is_parent tree parent child2 ->
    child1 <> child2 ->
    ~(is_ancestor tree child1 child2) /\ ~(is_ancestor tree child2 child1).
```

**Proof**: Paths from siblings to root diverge at their common parent, so neither sibling is on the other's path.

**Lemma 4** (Visible Contexts Monotonicity):
```coq
Lemma visible_contexts_subset :
  forall (tree : ContextTree) (parent child : ContextId),
    well_formed tree ->
    is_parent tree parent child ->
    exists (visible_child visible_parent : list ContextId),
      visible_child = visible_contexts tree child /\
      visible_parent = visible_contexts tree parent /\
      (forall ctx, In ctx visible_parent -> In ctx visible_child).
```

**Proof**: Child's visible contexts = `[child] ++ parent's visible contexts`. Therefore parent's visible set is a strict subset.

---

## Implementation Correspondence

**Source**: `src/contextual/context_tree.rs:335-349`, `src/contextual/engine.rs:1782-1802`

### is_descendant Implementation

```rust
// src/contextual/context_tree.rs:335-349
pub fn is_descendant(&self, child_id: ContextId, ancestor_id: ContextId) -> bool {
    if child_id == ancestor_id {
        return false;  // Strict descendant (not reflexive)
    }

    let mut current = self.parent(child_id);
    while let Some(parent_id) = current {
        if parent_id == ancestor_id {
            return true;  // Found ancestor in parent chain
        }
        current = self.parent(parent_id);  // Continue up the tree
    }

    false  // Reached root without finding ancestor
}
```

**Correspondence**:
- **Line 336-338**: Implements `child <> ancestor` check (non-reflexive)
- **Line 340**: Starts at parent of `child_id` (upward traversal)
- **Line 341-345**: Follows parent chain (implements `is_ancestor` recursion)
- **Line 348**: Returns `false` if ancestor not found (not in parent chain)

### Visibility Enforcement in complete_finalized

```rust
// src/contextual/engine.rs (simplified from lines 1090-1120)
pub fn complete_finalized(&self, context: ContextId, query: &str, max_distance: usize)
    -> Vec<Completion>
{
    let mut results = Vec::new();

    // Get visible contexts (self + ancestors)
    let visible = self.context_tree.visible_contexts(context);

    for ctx in visible {
        // Query dictionary for terms finalized in this context
        if let Some(terms) = self.finalized_symbols.get(&ctx) {
            for term in terms {
                let distance = Self::levenshtein_distance(query, term);
                if distance <= max_distance {
                    results.push(Completion {
                        term: term.clone(),
                        distance,
                        source_context: ctx,
                        is_draft: false,
                    });
                }
            }
        }
    }

    results
}
```

**Correspondence**:
- **Upward Visibility**: `visible_contexts(context)` includes `context` + all ancestors (Part 1 + Part 4)
- **No Downward Visibility**: `visible_contexts(parent)` does NOT include children (Part 2)
- **No Lateral Visibility**: Sibling contexts not in each other's `visible_contexts` (Part 3)

### Test: Hierarchical Visibility

```rust
// src/contextual/engine.rs:1782-1802
#[test]
fn test_complete_hierarchical_visibility() {
    let engine = DynamicContextualCompletionEngine::new();
    let global = engine.create_root_context(0);
    let func = engine.create_child_context(1, global).unwrap();

    // Global term
    engine.finalize_direct(global, "global_var").unwrap();

    // Local term
    engine.finalize_direct(func, "local_var").unwrap();

    // Query from func - should see both (PART 1: child sees parent)
    let results = engine.complete_finalized(func, "var", 10);
    assert!(results.iter().any(|c| c.term == "global_var"));   // ✓
    assert!(results.iter().any(|c| c.term == "local_var"));    // ✓

    // Query from global - should NOT see local_var (PART 2: parent doesn't see child)
    let results = engine.complete_finalized(global, "var", 10);
    assert!(results.iter().any(|c| c.term == "global_var"));   // ✓
    assert!(!results.iter().any(|c| c.term == "local_var"));   // ✓ Invisible!
}
```

**Correspondence**:
- **Lines 1794-1796**: Tests Part 1 (child sees parent) + Part 4 (transitive upward)
- **Lines 1798-1801**: Tests Part 2 (parent doesn't see child) + Part 5 (transitive downward)

---

## Test Coverage

**Unit Tests**: `src/contextual/context_tree.rs:505-535` (#[cfg(test)])

```rust
#[test]
fn test_is_descendant() {
    let mut tree = ContextTree::new();
    let root = tree.create_root(1);
    let child = tree.create_child(2, root).unwrap();
    let grandchild = tree.create_child(3, child).unwrap();

    // Descendant relationships
    assert!(tree.is_descendant(child, root));         // Direct child
    assert!(tree.is_descendant(grandchild, root));    // Transitive
    assert!(tree.is_descendant(grandchild, child));   // Direct child

    // NOT descendant (upward, lateral, or self)
    assert!(!tree.is_descendant(root, child));        // Parent NOT descendant of child
    assert!(!tree.is_descendant(child, grandchild));  // Parent NOT descendant of grandchild
    assert!(!tree.is_descendant(child, child));       // NOT self-descendant
}
```

**Integration Tests**: `src/contextual/engine.rs:1782-1802`

See test above (`test_complete_hierarchical_visibility`).

**Candidate Property-Based Tests**:

Using `proptest` crate:

```rust
proptest! {
    // Asymmetry: If A is ancestor of B, B is NOT ancestor of A
    #[test]
    fn prop_ancestor_asymmetry(tree: ContextTree, ctx1: ContextId, ctx2: ContextId) {
        if tree.is_descendant(ctx1, ctx2) {
            prop_assert!(!tree.is_descendant(ctx2, ctx1));
        }
    }

    // Transitivity: If A is ancestor of B and B is ancestor of C, then A is ancestor of C
    #[test]
    fn prop_ancestor_transitivity(
        tree: ContextTree,
        ctx1: ContextId,
        ctx2: ContextId,
        ctx3: ContextId
    ) {
        if tree.is_descendant(ctx2, ctx1) && tree.is_descendant(ctx3, ctx2) {
            prop_assert!(tree.is_descendant(ctx3, ctx1));
        }
    }

    // Visibility completeness: visible_contexts includes all ancestors
    #[test]
    fn prop_visible_includes_ancestors(tree: ContextTree, ctx: ContextId) {
        let visible = tree.visible_contexts(ctx);
        for ancestor in tree.ancestors(ctx) {
            prop_assert!(visible.contains(&ancestor));
        }
    }

    // Visibility soundness: visible_contexts excludes all non-ancestors
    #[test]
    fn prop_visible_excludes_non_ancestors(tree: ContextTree, ctx: ContextId) {
        let visible = tree.visible_contexts(ctx);
        for other in tree.all_contexts() {
            if !tree.is_descendant(ctx, other) && other != ctx {
                prop_assert!(!visible.contains(&other));
            }
        }
    }
}
```

---

## Related Theorems

**Depends on**:
- **Theorem 1** (Context Visibility) - `visible_contexts()` implementation

**Required by**:
- **Theorem 4** (Query Fusion) - Which contexts to query
- rholang-language-server scope detection
- LSP goto-definition (symbol resolution)

**Complements**:
- **Theorem 7** (Finalization Atomicity) - Ensures finalized symbols respect visibility

---

## Real-World Impact

### Without This Theorem

**Scenario**: User writing nested function in Rholang

```rholang
contract outer(@x) = {
  new temp in {           // temp is local to outer
    contract inner(@y) = {
      new result in {     // result is local to inner
        // ... computation ...
      }
    }
  }
}

// At module level, user types "re" and expects:
// - "rho" (builtin)
// - "receive" (builtin)
//
// WITHOUT Theorem 6, might also see:
// - "result" (leaked from inner!)  ← PRIVACY VIOLATION
```

**Consequences**:
1. **Privacy violation**: Inner scope symbols visible globally
2. **Noise**: Completion list polluted with irrelevant suggestions
3. **Security risk**: Internal variable names exposed
4. **Cognitive load**: User confused by unexpected suggestions

### With This Theorem (Correct Behavior)

```rholang
// At module level, "re" completes to:
completions = ["rho", "receive"]  // ✓ Only global symbols

// Inside "inner", "re" completes to:
completions = ["result", "rho", "receive"]  // ✓ Local + global

// Inside "outer" (but outside "inner"), "re" completes to:
completions = ["rho", "receive"]  // ✓ "result" is invisible (from inner)
```

---

## Promotion Criteria and Optimization Notes

### Rocq Formalization

**File**: `rocq/liblevenshtein/ContextualCompletion/Visibility.v`

**Proof Structure**:
```coq
(* Imports *)
Require Import Coq.Lists.List.
Require Import ContextTree.

(* Main theorem *)
Theorem hierarchical_visibility_soundness : (* ... *).
Proof.
  intros tree parent child sibling Hwf Hparent1 Hparent2 Hneq.
  split. (* Part 1: Children see parents *)
  - unfold visible_from. right. apply parent_is_ancestor. assumption.
  split. (* Part 2: Parents don't see children *)
  - apply descendant_not_ancestor_of_parent. assumption.
  split. (* Part 3: Siblings don't see each other *)
  - apply sibling_not_ancestor. assumption.
  split. (* Part 4: Transitivity upward *)
  - intros ancestor Hanc. unfold visible_from. right. assumption.
  (* Part 5: Transitivity downward *)
  - intros descendant Hdesc Hneq'. apply descendant_not_visible_from_ancestor.
    + assumption.
    + assumption.
Qed.
```

### Performance Considerations

**Current Implementation**: `$\mathcal{O}(\text{depth})$` for `is_descendant` check (follows parent chain)

**Optimization candidate** (if needed for large trees):
- **Interval Labeling**: Assign each node `[left, right]` interval during DFS
- **Descendant Check**: `ctx1` is descendant of `ctx2` iff `left2 < left1 < right1 < right2`
- **Time**: `$\mathcal{O}(1)$` check vs `$\mathcal{O}(\text{depth})$`
- **Space**: 2 integers per node (16 bytes overhead)

**When to optimize**:
- Tree depth > 100 (rare in LSP use cases)
- Frequent `is_descendant` queries (currently rare)

---

**Last Updated**: 2025-01-21
**Review trigger**: Reconcile this page when a checked contextual Rocq module is added.
