# Context Tree Visibility - Formal Proof Documentation

**Status**: Proof sketch documented; no checked contextual Rocq module exists yet
**Rocq target module**: `rocq/liblevenshtein/ContextualCompletion/Visibility.v`
**Date**: 2025-01-21
**Authors**: Formal Verification Team

---

## Table of Contents

1. [Overview](#overview)
2. [Definitions](#definitions)
3. [Theorem Statement](#theorem-statement)
4. [Proof Sketch](#proof-sketch)
5. [Key Lemmas](#key-lemmas)
6. [Implementation Correspondence](#implementation-correspondence)
7. [Test Coverage](#test-coverage)
8. [Promotion Criteria](#promotion-criteria)

---

## Overview

This theorem establishes the correctness of the `visible_contexts()` function, which is foundational for hierarchical symbol visibility in contextual completion. It ensures that for any context in the tree, we can correctly compute all ancestors (scopes) visible from that context.

### Why This Matters

**User Impact**: Without correct context visibility, code completion would:
- **Miss symbols** from outer scopes (false negatives)
- **Show symbols** from unrelated scopes (false positives)
- **Order results incorrectly** (inner scope symbols should rank higher)

**Downstream Dependencies**: This theorem is a prerequisite for:
- **Theorem 6** (Hierarchical Visibility Soundness) - Builds on this foundation
- **Theorem 4** (Query Fusion Completeness) - Relies on correct visibility chain
- **rholang-language-server scope detection** - Maps Rholang scopes to contexts

### Real-World Example

```rust
// Code editor with nested scopes
let global = tree.create_root(0);              // Global scope
let module = tree.create_child(1, global);     // Module scope
let function = tree.create_child(2, module);   // Function scope

// When cursor is in function scope, user should see symbols from:
let visible = tree.visible_contexts(function);
// Returns: [function, module, global]  ← Order matters!
//          ^innermost  ^parent  ^root
```

**Correctness Properties**:
1. **Soundness**: All returned contexts are ancestors (or self)
2. **Completeness**: All ancestors are returned
3. **Ordering**: Innermost → outermost (for completion ranking)
4. **Termination**: Always terminates (no cycles)

---

## Definitions

### Context Tree Structure

```coq
(* Context identifier - natural number *)
Definition ContextId := nat.

(* Context tree node - stores parent pointer *)
Record ContextNode : Type := {
  id : ContextId;
  parent : option ContextId  (* None for root *)
}.

(* Context tree - map from ID to parent *)
Definition ContextTree := list (ContextId * option ContextId).

(* Well-formedness invariant *)
Definition well_formed (tree : ContextTree) : Prop :=
  (* 1. All parent references are valid *)
  (forall id p, In (id, Some p) tree -> exists p_node, In (p, p_node) tree) /\
  (* 2. No cycles (all parent chains terminate) *)
  (forall id, exists n, parent_chain_length tree id = Some n) /\
  (* 3. Each ID appears at most once *)
  (forall id p1 p2, In (id, p1) tree -> In (id, p2) tree -> p1 = p2).
```

### Visibility Relation

```coq
(* Ancestor relation: a is an ancestor of d if d's parent chain reaches a *)
Fixpoint is_ancestor (tree : ContextTree) (a d : ContextId) : bool :=
  match lookup tree d with
  | None => false
  | Some None => a =? d           (* d is root, check if a = d *)
  | Some (Some p) =>
      if a =? d then true          (* d is its own ancestor *)
      else is_ancestor tree a p    (* Recursively check parent *)
  end.

(* Visible contexts: all ancestors including self *)
Fixpoint visible_contexts (tree : ContextTree) (id : ContextId)
  : list ContextId :=
  match lookup tree id with
  | None => []                     (* Context doesn't exist *)
  | Some None => [id]              (* Root context *)
  | Some (Some p) => id :: visible_contexts tree p  (* Prepend self *)
  end.
```

### Ordering Property

```coq
(* Contexts are ordered from innermost to outermost *)
Definition correct_ordering (result : list ContextId) (tree : ContextTree)
  (id : ContextId) : Prop :=
  match result with
  | [] => ~ In id tree   (* Empty only if context doesn't exist *)
  | first :: rest =>
      first = id /\      (* First element is the query context *)
      (forall i j, i < j < length result ->
        (* Each element is parent of previous *)
        parent tree (nth i result 0) = Some (nth j result 0))
  end.
```

---

## Theorem Statement

### Informal Statement

**Context Visibility Correctness**: For any context `c` in a well-formed context tree `T`, the function `visible_contexts(c)` returns exactly the set of all ancestor contexts (including `c` itself), ordered from innermost (self) to outermost (root).

### Formal Statement

```coq
Theorem visible_contexts_correct :
  forall (tree : ContextTree) (id : ContextId),
    well_formed tree ->
    In id tree ->
    let result := visible_contexts tree id in
    (* Soundness: All returned contexts are ancestors *)
    (forall ctx, In ctx result -> is_ancestor tree ctx id = true) /\
    (* Completeness: All ancestors are returned *)
    (forall ctx, is_ancestor tree ctx id = true -> In ctx result) /\
    (* Correct ordering: innermost to outermost *)
    correct_ordering result tree id.
```

**English**: If the tree is well-formed and the context exists, then:
1. Every context in the result is an ancestor (soundness)
2. Every ancestor appears in the result (completeness)
3. The result is ordered correctly: `[id, parent, grandparent, ..., root]`

---

## Proof Sketch

### Strategy

**Structural induction** on the parent chain from `id` to root, leveraging the well-formedness invariant to guarantee termination.

### Main Steps

1. **Base Case** (id is root):
   ```coq
   (* If id has no parent, visible_contexts returns [id] *)
   - Result: [id]
   - Soundness: id is its own ancestor ✓
   - Completeness: id is the only ancestor ✓
   - Ordering: Single element is trivially ordered ✓
   ```

2. **Inductive Case** (id has parent p):
   ```coq
   (* Assume visible_contexts(p) is correct by IH *)
   - IH: visible_contexts(tree, p) satisfies all three properties
   - Goal: Show visible_contexts(tree, id) = id :: visible_contexts(tree, p)
           satisfies all three properties

   (* Soundness *)
   - id is in result: id is ancestor of id ✓ (reflexive)
   - Rest of result = IH result: Ancestors of p are ancestors of id ✓ (transitive)

   (* Completeness *)
   - id is returned as first element ✓
   - Any ancestor a of id:
     - If a = id: Already in result as first element ✓
     - If a ≠ id: a is ancestor of p (by definition) → in IH result ✓

   (* Ordering *)
   - First element is id ✓
   - Rest follows ordering of IH (parent chain of p) ✓
   - id's parent is p (first element of rest) ✓
   ```

3. **Termination**:
   ```coq
   (* Well-formedness guarantees no cycles *)
   - parent_chain_length(tree, id) = Some n  (from well_formed)
   - Induction proceeds along finite parent chain
   - Must terminate at root (parent = None) ✓
   ```

### Detailed Proof

```coq
Proof.
  intros tree id Hwf Hin.
  unfold visible_contexts.

  (* Induction on parent chain length *)
  remember (parent_chain_length tree id) as n eqn:Heqn.
  destruct n as [n|]; [| contradiction]. (* Must have finite chain *)

  generalize dependent id.
  induction n as [|n' IH]; intros id Hin Heqn.

  - (* Base case: n = 0, id is root *)
    assert (parent tree id = None) by (apply parent_chain_zero; auto).
    simpl. rewrite H.
    split; [| split].
    + (* Soundness *)
      intros ctx Hctx. simpl in Hctx.
      destruct Hctx as [Hid | []]; subst.
      apply is_ancestor_reflexive.
    + (* Completeness *)
      intros ctx Hanc.
      apply is_ancestor_root_only in Hanc; auto.
      left. auto.
    + (* Ordering *)
      apply ordering_single_element.

  - (* Inductive case: n = S n', id has parent *)
    destruct (parent tree id) as [p|] eqn:Hparent.
    + (* id has parent p *)
      assert (parent_chain_length tree p = Some n') as Hp.
      { apply parent_chain_successor with id; auto. }

      (* Apply IH to parent *)
      specialize (IH p).
      assert (In p tree) as Hinp by (apply parent_in_tree with id; auto).
      specialize (IH Hinp Hp).
      destruct IH as [IHsound [IHcomp IHord]].

      simpl. rewrite Hparent.
      split; [| split].

      * (* Soundness: id :: visible_contexts(p) are all ancestors *)
        intros ctx [Hid | Hrest].
        -- subst. apply is_ancestor_reflexive.
        -- apply IHsound in Hrest.
           apply is_ancestor_transitive with p; auto.
           apply is_ancestor_parent; auto.

      * (* Completeness: All ancestors appear in result *)
        intros ctx Hanc.
        destruct (id =? ctx) eqn:Heq.
        -- left. apply Nat.eqb_eq. auto.
        -- right. apply IHcomp.
           apply is_ancestor_via_parent; auto.
           ++ apply Nat.eqb_neq. auto.
           ++ auto.

      * (* Ordering: id first, then parent chain *)
        apply ordering_cons; auto.
        -- reflexivity.
        -- apply IHord.

    + (* id has no parent (root), but n > 0 - contradiction *)
      exfalso.
      apply parent_chain_positive_has_parent in Heqn; auto.
      congruence.
Qed.
```

---

## Key Lemmas

### Auxiliary Theorems

**Lemma 1: Ancestor Reflexivity**
```coq
Lemma is_ancestor_reflexive :
  forall tree id, In id tree -> is_ancestor tree id id = true.
```
*Proof*: Direct from definition - every context is its own ancestor.

**Lemma 2: Ancestor Transitivity**
```coq
Lemma is_ancestor_transitive :
  forall tree a b c,
    is_ancestor tree a b = true ->
    is_ancestor tree b c = true ->
    is_ancestor tree a c = true.
```
*Proof*: Induction on parent chain from `c` to `b`, then `b` to `a`.

**Lemma 3: Parent is Ancestor**
```coq
Lemma is_ancestor_parent :
  forall tree id p,
    parent tree id = Some p ->
    is_ancestor tree p id = true.
```
*Proof*: Direct from definition of `is_ancestor` - parent is in parent chain.

**Lemma 4: Parent Chain Finite**
```coq
Lemma parent_chain_length_exists :
  forall tree id,
    well_formed tree ->
    In id tree ->
    exists n, parent_chain_length tree id = Some n.
```
*Proof*: From well-formedness (no cycles), parent chain must terminate at root.

**Lemma 5: Ordering Preservation**
```coq
Lemma ordering_cons :
  forall id p rest tree,
    parent tree id = Some p ->
    correct_ordering rest tree p ->
    correct_ordering (id :: rest) tree id.
```
*Proof*: Show first element is `id`, and rest follows ordering of parent chain.

---

## Implementation Correspondence

### Source Files

**Primary Implementation**:
- **File**: `src/contextual/context_tree.rs`
- **Lines**: 244-258
- **Function**: `ContextTree::visible_contexts()`

```rust
pub fn visible_contexts(&self, id: ContextId) -> Vec<ContextId> {
    let mut result = Vec::new();
    let mut current = Some(id);

    while let Some(ctx_id) = current {
        if self.nodes.contains_key(&ctx_id) {
            result.push(ctx_id);              // Prepend current context
            current = self.parent(ctx_id);    // Move to parent
        } else {
            break;  // Context doesn't exist
        }
    }

    result
}
```

**Correspondence to Formal Specification**:

| Formal Construct | Rust Implementation | Notes |
|------------------|---------------------|-------|
| `visible_contexts tree id` | `self.visible_contexts(id)` | Same signature |
| Base case (no parent) | `self.parent(ctx_id)` returns `None` → loop exits | Correct ✓ |
| Recursive case | While loop with `current = parent` | Iterative equivalent ✓ |
| Result ordering | `push()` appends in traversal order | Innermost→outermost ✓ |
| Termination | `nodes.contains_key()` check + parent chain finite | Well-formedness ensures termination ✓ |
| Non-existent context | Early `break` if context not in tree | Returns empty `Vec` ✓ |

**Well-Formedness Maintenance**:

The Rust implementation maintains well-formedness through:
1. **No cycles**: `create_child()` only links existing parents (lines 139-145)
2. **Valid parents**: Parent must exist before creating child (line 140 check)
3. **Unique IDs**: `HashMap` enforces unique keys

```rust
pub fn create_child(&mut self, id: ContextId, parent_id: ContextId)
    -> Result<ContextId> {
    if !self.nodes.contains_key(&parent_id) {
        return Err(ContextError::ContextNotFound(parent_id));  // ← Enforces valid parent
    }
    self.nodes.insert(id, Some(parent_id));
    Ok(id)
}
```

### Performance Characteristics

**Time Complexity**: `$\mathcal{O}(d)$` where d = depth of context in tree
- **Best case**: `$\mathcal{O}(1)$` - root context
- **Worst case**: `$\mathcal{O}(n)$` - deeply nested context (n = total contexts)
- **Typical**: `$\mathcal{O}(\log  n)$` - balanced tree

**Space Complexity**: `$\mathcal{O}(d)$` for result vector
- **In-place**: No additional heap allocation beyond result
- **No recursion**: Iterative implementation avoids stack overhead

**Benchmarks** (from implementation comments):
- Tree with 50 contexts: < 1µs per query
- Deep nesting (depth 20): ~500ns per query
- Allocation overhead: ~48 bytes per result vector

---

## Test Coverage

### Unit Tests

**Location**: `src/contextual/context_tree.rs` (lines 330-450, #[cfg(test)])

**Test 1: Basic Visibility**
```rust
#[test]
fn test_visible_contexts_single() {
    let mut tree = ContextTree::new();
    let root = tree.create_root(1);

    assert_eq!(tree.visible_contexts(root), vec![root]);
    // ✓ Tests: Root returns only itself
}
```

**Test 2: Parent-Child Chain**
```rust
#[test]
fn test_visible_contexts_chain() {
    let mut tree = ContextTree::new();
    let global = tree.create_root(1);
    let module = tree.create_child(2, global).unwrap();
    let function = tree.create_child(3, module).unwrap();

    assert_eq!(
        tree.visible_contexts(function),
        vec![function, module, global]
    );
    // ✓ Tests: Soundness, completeness, ordering
}
```

**Test 3: Non-Existent Context**
```rust
#[test]
fn test_visible_contexts_nonexistent() {
    let tree = ContextTree::new();

    assert_eq!(tree.visible_contexts(999), vec![]);
    // ✓ Tests: Handles missing context gracefully
}
```

**Test 4: Sibling Contexts**
```rust
#[test]
fn test_visible_contexts_siblings() {
    let mut tree = ContextTree::new();
    let root = tree.create_root(1);
    let child1 = tree.create_child(2, root).unwrap();
    let child2 = tree.create_child(3, root).unwrap();

    // Siblings don't see each other
    assert_eq!(tree.visible_contexts(child1), vec![child1, root]);
    assert_eq!(tree.visible_contexts(child2), vec![child2, root]);
    // ✓ Tests: Isolation (not in this theorem, but good sanity check)
}
```

### Integration Tests

**rholang-language-server Integration**:

**Location**: `/home/dylon/Workspace/f1r3fly.io/rholang-language-server/tests/test_completion.rs`

**Test: Nested Scope Priority**
```rust
#[test]
fn test_nested_scope_priority_inner() {
    // Rholang code with 3 nested scopes
    let code = r#"
        new result1 in {
            new result2 in {
                new result3 in {
                    r  // ← Cursor here
                }
            }
        }
    "#;

    // Completion query for "r" should return:
    // [result3, result2, result1] ← innermost to outermost

    // This test DEPENDS on Theorem 1 (Context Visibility)
    // Without correct visible_contexts(), results would be:
    // - Wrong order: [result1, result2, result3] ✗
    // - Missing symbols: [result3] only ✗
    // - Extra symbols: [result1, result2, result3, unrelated_var] ✗
}
```

### Candidate Property-Based Tests

**Property 1: Soundness**
```rust
proptest! {
    #[test]
    fn visible_contexts_are_ancestors(
        tree in arbitrary_context_tree(),
        id in arbitrary_context_id()
    ) {
        let visible = tree.visible_contexts(id);
        for ctx in visible {
            assert!(is_ancestor(&tree, ctx, id));
        }
    }
}
```

**Property 2: Completeness**
```rust
proptest! {
    #[test]
    fn all_ancestors_visible(
        tree in arbitrary_context_tree(),
        id in arbitrary_context_id()
    ) {
        let visible = tree.visible_contexts(id);
        for ctx in all_contexts(&tree) {
            if is_ancestor(&tree, ctx, id) {
                assert!(visible.contains(&ctx));
            }
        }
    }
}
```

**Property 3: Ordering**
```rust
proptest! {
    #[test]
    fn visible_contexts_ordered_innermost_to_root(
        tree in arbitrary_context_tree(),
        id in arbitrary_context_id()
    ) {
        let visible = tree.visible_contexts(id);
        if !visible.is_empty() {
            // First element is query context
            assert_eq!(visible[0], id);

            // Each subsequent element is parent of previous
            for i in 0..visible.len()-1 {
                assert_eq!(tree.parent(visible[i]), Some(visible[i+1]));
            }
        }
    }
}
```

---

## Promotion Criteria

### Rocq Formalization

**Target module contents**:
1. Create `rocq/liblevenshtein/ContextualCompletion/Core.v`
   - Define `ContextId`, `ContextTree`, `well_formed` types
   - Prove well-formedness lemmas

2. Create `rocq/liblevenshtein/ContextualCompletion/Visibility.v`
   - Formalize `visible_contexts` function
   - Prove Theorem 1 (this document)
   - Prove auxiliary lemmas

3. Extract to verified implementation
   - Generate OCaml code from Coq
   - Transpile to Rust
   - Compare with existing implementation

### Property-Test Specification Coverage

**Priority**: High (validates theorem without full Coq formalization)

**Candidate coverage**:
1. Add `proptest` dependency to `Cargo.toml`
2. Implement `arbitrary_context_tree()` generator
3. Add 3 property tests from "Property-Based Tests" section above
4. Run in CI (10,000 cases per test)

### Performance Optimization

**Current**: `$\mathcal{O}(d)$` time, `$\mathcal{O}(d)$` space (d = depth)

**Potential Optimization**: Cache visibility results
- **Idea**: Store `Vec<ContextId>` in each context node
- **Trade-off**: 2x memory for `$\mathcal{O}(1)$` lookup
- **When useful**: Deep trees (d > 10) with frequent queries
- **Invalidation**: Must rebuild on tree modification

**Decision**: Keep the current `$\mathcal{O}(d)$` implementation unless profiling shows that cached visibility justifies the memory trade-off.

### Cycle Detection

**Current**: Assumes well-formedness (no cycles)

**Optional extension**: Defensive cycle detection
```rust
pub fn has_cycle(&self, id: ContextId) -> bool {
    let mut visited = HashSet::new();
    let mut current = Some(id);

    while let Some(ctx) = current {
        if !visited.insert(ctx) {
            return true;  // Cycle detected
        }
        current = self.parent(ctx);
    }
    false
}
```

**Use case**: Debug mode assertions, malformed tree recovery

---

## Related Theorems

**Depends on**: None (foundational theorem)

**Required by**:
- **Theorem 6** (Hierarchical Visibility Soundness) - Uses `visible_contexts()` to define visibility rules
- **Theorem 4** (Query Fusion Completeness) - Queries all visible contexts for symbols

**See also**:
- `docs/formal-verification/proofs/07_dictionary_backends/01_trie_reachability.md` - Dictionary correctness assumed by this theorem

---

## Changelog

### 2025-01-21 - Initial Documentation

**Created**:
- Full formal specification
- Proof sketch with induction strategy
- Implementation correspondence analysis
- Test coverage mapping

**Status**: Proof-sketch documentation complete; no checked contextual Rocq module exists yet.

---

## References

**Implementation**:
- `src/contextual/context_tree.rs:244-258` - `visible_contexts()` implementation
- `src/contextual/context_tree.rs:330-450` - Unit tests

**Theory**:
- `docs/algorithms/07-contextual-completion/README.md` - Algorithm overview
- Tree data structures: [Wikipedia - Tree traversal](https://en.wikipedia.org/wiki/Tree_traversal)

**Formal Methods**:
- [Software Foundations - Induction](https://softwarefoundations.cis.upenn.edu/lf-current/Induction.html)
- [CPDT - Inductive Types](http://adam.chlipala.net/cpdt/html/InductiveTypes.html)

---

**Last Updated**: 2025-01-21
**Review trigger**: Reconcile this page when a checked contextual Rocq module is added.
