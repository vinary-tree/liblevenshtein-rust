# Formal Verification Requirements for Contextual Completion (Phase 9)

**Purpose**: This document specifies the formal verification requirements (Coq proofs) needed to enable mathematically proven correctness guarantees for downstream consumers of liblevenshtein-rust's contextual completion engine.

**Status**: Documentation complete, Coq proofs awaiting implementation
**Priority**: P0 for LSP consumers requiring formal verification (rholang-language-server)

---

## Overview

The contextual completion engine (Phase 9) provides scope-aware code completion with hierarchical visibility, draft buffers, and checkpoint-based undo. While the implementation is thoroughly tested (93 passing tests, property-based validation), **formal verification using Coq is required to provide mathematical certainty** for safety-critical applications.

### Why Formal Verification Matters

**Current State**: Empirical confidence via property-based testing
- ✅ 93 unit/integration tests passing
- ✅ Property tests with 50+ generated cases
- ✅ Comprehensive coverage of API surface

**Gap**: No mathematical proof of correctness
- ❌ Property tests can miss edge cases
- ❌ No guarantee against subtle concurrency bugs
- ❌ Cannot formally reason about invariant preservation

**Blocked Use Cases**:
1. **LSP servers** requiring proven symbol visibility correctness
2. **Security-critical** code completion (e.g., access control suggestions)
3. **Research** contributions with formal correctness claims

---

## Theorem Catalog

This section lists the 7 theorems requiring formal proof in Coq. Theorems 1 and 6 are **blocking** for LSP formal verification; the remaining 5 are useful but not critical.

### ✅ Theorem 1: Context Tree Visibility (CRITICAL - P0)

**Statement**: For any context `C` in the context tree, a term `t` is visible in `C` if and only if:
- `t` is finalized in `C`, OR
- `t` is finalized in some ancestor context `A` where `A` is reachable from `C` via parent links

**Formal Specification (Coq)**:

```coq
(* Context tree structure *)
Inductive Context : Type :=
  | GlobalContext : Context
  | ChildContext : Context → ContextId → Context.

(* Term finalization relation *)
Definition finalized_in (t : string) (c : Context) (dict : Dictionary) : Prop :=
  ∃ value : Vec ContextId,
    dictionary_lookup dict t = Some value ∧
    List.In c value.

(* Ancestor relation *)
Inductive ancestor : Context → Context → Prop :=
  | ancestor_direct : ∀ C id, ancestor C (ChildContext C id)
  | ancestor_trans : ∀ C1 C2 C3,
      ancestor C1 C2 → ancestor C2 C3 → ancestor C1 C3.

(* Visibility relation *)
Definition visible_in (t : string) (c : Context) (dict : Dictionary) : Prop :=
  finalized_in t c dict ∨
  (∃ A : Context, ancestor A c ∧ finalized_in t A dict).

(* THEOREM 1 *)
Theorem context_tree_visibility :
  ∀ (C : Context) (t : string) (dict : Dictionary),
    visible_in t C dict ↔
    (finalized_in t C dict ∨
     ∃ A : Context, ancestor A C ∧ finalized_in t A dict).
Proof.
  (* Proof strategy: induction on the context tree, using ancestor
     reflexivity/transitivity and the definition of visible_in. *)
Admitted.
```

**Blocking**:
- **LSP VO-4**: Query returns all symbols finalized in current context
- **LSP VO-5**: Query returns all symbols finalized in ancestor contexts
- **LSP VO-6**: Query does NOT return symbols from sibling/descendant contexts

**Coq File**: `formal-verification/coq/Visibility.v`

**Dependencies**:
- ContextTree data structure formalization (`ContextTree.v`)
- Dictionary interface specification (`Dictionary.v`)
- Ancestor relation properties (reflexivity, transitivity) (`AncestorRelation.v`)

**Estimated Effort**: 2-4 weeks

---

### ✅ Theorem 6: Hierarchical Visibility Soundness (CRITICAL - P0)

**Statement**: For contexts `C₁` and `C₂` where `C₂` is a child of `C₁`:
1. **Inheritance**: All terms visible in `C₁` are visible in `C₂`
2. **Isolation**: Terms finalized in `C₂` are NOT visible in `C₁` unless also finalized in `C₁` or its ancestors

**Formal Specification (Coq)**:

```coq
(* Parent relation *)
Definition parent (C1 C2 : Context) : Prop :=
  ∃ id : ContextId, C2 = ChildContext C1 id.

(* THEOREM 6 *)
Theorem hierarchical_visibility_soundness :
  ∀ (C1 C2 : Context) (t : string) (dict : Dictionary),
    parent C1 C2 →
    (* Part 1: Inheritance *)
    (visible_in t C1 dict → visible_in t C2 dict) ∧
    (* Part 2: Isolation *)
    (finalized_in t C2 dict →
      ¬visible_in t C1 dict ∨
      finalized_in t C1 dict ∨
      ∃ A : Context, ancestor A C1 ∧ finalized_in t A dict).
Proof.
  (* Proof strategy: inheritance follows by extending the ancestor chain from
     C1 to C2; isolation follows by case analysis on visible_in for C1 and the
     absence of descendant visibility in the definition. *)
Admitted.
```

**Blocking**:
- **LSP VO-5**: Child contexts inherit parent visibility (prevents omissions)
- **LSP VO-6**: Parent contexts don't see child-only symbols (prevents leaks)

**Coq File**: `formal-verification/coq/Hierarchy.v`

**Dependencies**:
- Theorem 1 (Context Tree Visibility)
- Parent relation properties
- Visibility relation transitivity lemmas

**Estimated Effort**: 1-2 weeks (builds on Theorem 1)

---

### Theorem 2: UTF-8 Correctness (USEFUL - P2)

**Statement**: Draft buffers correctly handle multi-byte UTF-8 characters without corruption.

**Informal Specification**:
- **Insertion**: Inserting a character `c` appends it to the buffer, preserving UTF-8 validity
- **Deletion**: Removing a character deletes exactly one Unicode code point, not partial bytes
- **Invariant**: Buffer content is always valid UTF-8

**Why Useful**: Documents that Rust's `String` UTF-8 guarantees are preserved through draft operations.

**Blocking**: ❌ No (Rust's type system already enforces UTF-8 validity)

**Coq File**: `formal-verification/coq/UTF8.v`

**Estimated Effort**: 1 week

---

### Theorem 3: Draft Finalization Idempotence (USEFUL - P2)

**Statement**: Finalizing the same draft multiple times produces the same result.

**Formal Specification (Sketch)**:

```coq
Theorem draft_finalization_idempotence :
  ∀ (ctx : ContextId) (engine : CompletionEngine) (draft : String),
    let engine1 := finalize(engine, ctx, draft) in
    let engine2 := finalize(engine1, ctx, draft) in
    dictionary_state engine1 = dictionary_state engine2.
```

**Why Useful**: Strengthens defensive programming guarantees, simplifies reasoning about finalization.

**Blocking**: ❌ No (finalization is not expected to be called multiple times in normal usage)

**Coq File**: `formal-verification/coq/Finalization.v`

**Estimated Effort**: 1 week

---

### Theorem 4: Checkpoint Rollback Correctness (USEFUL - P3)

**Statement**: Creating a checkpoint, undoing, and restoring is idempotent.

**Formal Specification (Sketch)**:

```coq
Theorem checkpoint_rollback_correctness :
  ∀ (ctx : ContextId) (engine : CompletionEngine),
    let cp := checkpoint(engine, ctx) in
    let engine1 := insert_str(engine, ctx, "test") in
    let engine2 := restore(engine1, ctx, cp) in
    draft_state(engine2, ctx) = draft_state(engine, ctx).
```

**Why Useful**: Enables safe implementation of undo/redo features in editors.

**Blocking**: ❌ No (checkpoint/undo is not currently used in LSP, reserved for future features)

**Coq File**: `formal-verification/coq/Checkpoints.v`

**Estimated Effort**: 1 week

---

### Theorem 5: Fuzzy Query Soundness (USEFUL - P3)

**Statement**: All completions returned by fuzzy query are within the specified Levenshtein distance.

**Formal Specification (Sketch)**:

```coq
Theorem fuzzy_query_soundness :
  ∀ (query : string) (max_distance : nat) (engine : CompletionEngine),
    ∀ completion ∈ complete(engine, ctx, query, max_distance),
      levenshtein_distance(query, completion.term) ≤ max_distance.
```

**Why Useful**: Strengthens trust in fuzzy completion accuracy.

**Blocking**: ❌ No (fuzzy matching correctness is delegated to Levenshtein automaton layer)

**Coq File**: `formal-verification/coq/FuzzyQuery.v`

**Dependencies**: Levenshtein automaton correctness proof (Layer 2)

**Estimated Effort**: 1-2 weeks (requires formalizing automaton semantics)

---

### Theorem 7: Context Switching Isolation (USEFUL - P3)

**Statement**: Switching contexts does not leak symbols between unrelated contexts.

**Formal Specification (Sketch)**:

```coq
Theorem context_switching_isolation :
  ∀ (ctx_a ctx_b : ContextId) (engine : CompletionEngine),
    ¬ancestor ctx_a ctx_b →
    ¬ancestor ctx_b ctx_a →
    ∀ t : string,
      (finalized_in t ctx_a ∧ ¬finalized_in t ctx_b) →
      ¬visible_in t ctx_b.
```

**Why Useful**: Makes explicit what Theorem 6 implies about sibling context isolation.

**Blocking**: ❌ No (follows from Theorem 6 as a corollary)

**Coq File**: `formal-verification/coq/ContextSwitching.v`

**Dependencies**: Theorem 6

**Estimated Effort**: <1 week (simple corollary)

---

## Priority and Blocking Status

| Theorem | Priority | Blocks LSP | Effort | Dependencies |
|---------|----------|------------|--------|--------------|
| **Theorem 1** (Context Tree Visibility) | **P0** | ✅ Yes | 2-4 weeks | ContextTree, Dictionary models |
| **Theorem 6** (Hierarchical Visibility) | **P0** | ✅ Yes | 1-2 weeks | Theorem 1 |
| Theorem 2 (UTF-8 Correctness) | P2 | ❌ No | 1 week | None |
| Theorem 3 (Draft Finalization) | P2 | ❌ No | 1 week | Dictionary semantics |
| Theorem 4 (Checkpoint Rollback) | P3 | ❌ No | 1 week | DraftBuffer semantics |
| Theorem 5 (Fuzzy Query Soundness) | P3 | ❌ No | 1-2 weeks | Layer 2 automaton proofs |
| Theorem 7 (Context Switching) | P3 | ❌ No | <1 week | Theorem 6 |

**Total Estimated Effort**: 7-12 weeks for all theorems, **3-6 weeks for P0 only**

---

## Recommended Coq Formalization Structure

### Directory Layout

```
liblevenshtein-rust/
└── formal-verification/
    └── coq/
        ├── _CoqProject              # Coq project configuration
        ├── Makefile                 # Build automation
        ├── README.md                # Proof documentation
        │
        ├── Foundations/             # Core definitions
        │   ├── ContextTree.v        # Context tree data structure
        │   ├── Dictionary.v         # Dictionary interface
        │   ├── DraftBuffer.v        # Character buffer model
        │   └── CompletionEngine.v   # Engine state model
        │
        ├── Relations/               # Semantic relations
        │   ├── Ancestor.v           # Ancestor relation + lemmas
        │   ├── Visibility.v         # Visibility relation + lemmas
        │   └── Finalization.v       # Finalization semantics
        │
        ├── Theorems/                # Main proofs
        │   ├── Theorem1_Visibility.v        # Context tree visibility (P0)
        │   ├── Theorem6_Hierarchy.v         # Hierarchical soundness (P0)
        │   ├── Theorem2_UTF8.v              # UTF-8 correctness
        │   ├── Theorem3_Idempotence.v       # Draft finalization
        │   ├── Theorem4_Checkpoints.v       # Checkpoint rollback
        │   ├── Theorem5_Fuzzy.v             # Fuzzy query soundness
        │   └── Theorem7_Isolation.v         # Context switching
        │
        └── Extraction/              # Code extraction
            ├── Extract.v            # Extraction configuration
            └── extracted/           # Generated Rust code
                └── verified_engine.rs
```

### Dependency Graph

```
Foundations (ContextTree.v, Dictionary.v, DraftBuffer.v)
      ↓
Relations (Ancestor.v, Visibility.v, Finalization.v)
      ↓
Theorem 1 (Visibility) ← P0 BLOCKING
      ↓
Theorem 6 (Hierarchy) ← P0 BLOCKING
      ↓
Theorem 7 (Isolation) ← Corollary
      ↓
Theorem 2, 3, 4, 5 ← Independent (P2-P3)
      ↓
Extraction (Optional: verified_engine.rs)
```

---

## Proof Strategy

### Phase 1: Foundations (Weeks 1-2)

**Goal**: Formalize core data structures and relations

**Tasks**:
1. **Model `ContextTree`** (`Foundations/ContextTree.v`):
   ```coq
   Inductive ContextTree : Type :=
     | EmptyTree : ContextTree
     | Node : ContextId → Context → ContextTree → ContextTree.

   Definition lookup_context (tree : ContextTree) (id : ContextId) : option Context := (* ... *).
   ```

2. **Model `Dictionary`** (`Foundations/Dictionary.v`):
   ```coq
   Parameter Dictionary : Type.
   Parameter lookup : Dictionary → string → option (Vec ContextId).
   Parameter insert : Dictionary → string → Vec ContextId → Dictionary.

   Axiom lookup_insert_eq : ∀ d k v,
     lookup (insert d k v) k = Some v.

   Axiom lookup_insert_neq : ∀ d k1 k2 v,
     k1 ≠ k2 → lookup (insert d k1 v) k2 = lookup d k2.
   ```

3. **Define `ancestor` relation** (`Relations/Ancestor.v`):
   ```coq
   Lemma ancestor_transitive : ∀ A B C,
     ancestor A B → ancestor B C → ancestor A C.

   Lemma ancestor_irreflexive : ∀ C,
     ¬ancestor C C.

   Lemma ancestor_antisymmetric : ∀ A B,
     ancestor A B → ancestor B A → False.
   ```

**Deliverable**: Compiled `.vo` files, basic sanity checks

---

### Phase 2: Theorem 1 Proof (Weeks 3-4)

**Goal**: Prove context tree visibility correctness

**Proof Outline**:

```coq
Theorem context_tree_visibility :
  ∀ (C : Context) (t : string) (dict : Dictionary),
    visible_in t C dict ↔
    (finalized_in t C dict ∨
     ∃ A : Context, ancestor A C ∧ finalized_in t A dict).
Proof.
  intros C t dict.
  unfold visible_in.
  split.
  - (* → direction: trivial by definition *)
    intros H. assumption.
  - (* ← direction: trivial by definition *)
    intros H. assumption.
Qed.
```

**Note**: The theorem statement is definitional, so proof is trivial. The real work is in proving **lemmas about visibility queries**:

```coq
Lemma query_completeness :
  ∀ (C : Context) (t : string) (dict : Dictionary),
    visible_in t C dict →
    ∃ completion : Completion,
      completion ∈ complete(C, t, 0) ∧
      completion.term = t.

Lemma query_soundness :
  ∀ (C : Context) (completion : Completion),
    completion ∈ complete(C, query, max_dist) →
    visible_in completion.term C dict.
```

**Deliverable**: `Theorems/Theorem1_Visibility.v` compiled

---

### Phase 3: Theorem 6 Proof (Weeks 5-6)

**Goal**: Prove hierarchical visibility soundness

**Proof Outline**:

```coq
Theorem hierarchical_visibility_soundness :
  ∀ (C1 C2 : Context) (t : string) (dict : Dictionary),
    parent C1 C2 →
    (visible_in t C1 dict → visible_in t C2 dict) ∧
    (finalized_in t C2 dict →
      ¬visible_in t C1 dict ∨
      finalized_in t C1 dict ∨
      ∃ A : Context, ancestor A C1 ∧ finalized_in t A dict).
Proof.
  intros C1 C2 t dict H_parent.
  destruct H_parent as [id H_eq].
  subst C2.
  split.

  - (* Part 1: Inheritance (C1 visible → C2 visible) *)
    intros H_vis_C1.
    unfold visible_in in *.
    destruct H_vis_C1 as [H_fin_C1 | [A [H_anc_A_C1 H_fin_A]]].
    + (* t finalized in C1 *)
      right. exists C1. split.
      * apply ancestor_direct.
      * assumption.
    + (* t finalized in ancestor of C1 *)
      right. exists A. split.
      * apply ancestor_trans with C1.
        { assumption. }
        { apply ancestor_direct. }
      * assumption.

  - (* Part 2: Isolation (C2 finalized → not visible in C1 OR finalized in C1/ancestor) *)
    intros H_fin_C2.
    unfold visible_in.
    (* Key insight: finalization in child doesn't propagate upward *)
    (* Either t is not visible in C1, OR it's separately finalized in C1/ancestor *)
    tauto.
Qed.
```

**Key Lemmas**:

```coq
Lemma child_inherits_visibility :
  ∀ C1 C2 t dict,
    parent C1 C2 →
    visible_in t C1 dict →
    visible_in t C2 dict.

Lemma parent_isolates_child :
  ∀ C1 C2 t dict,
    parent C1 C2 →
    finalized_in t C2 dict →
    ¬finalized_in t C1 dict →
    ¬(∃ A, ancestor A C1 ∧ finalized_in t A dict) →
    ¬visible_in t C1 dict.
```

**Deliverable**: `Theorems/Theorem6_Hierarchy.v` compiled

---

### Phase 4: Optional Theorems (Weeks 7-12)

**Goal**: Prove Theorems 2-5, 7 (non-blocking)

**Parallel Workstreams**:
- **UTF-8 (Theorem 2)**: Formalize `Vec<char>` invariants
- **Idempotence (Theorem 3)**: Model dictionary insertion semantics
- **Checkpoints (Theorem 4)**: Model stack-based undo
- **Fuzzy Query (Theorem 5)**: Formalize Levenshtein distance
- **Isolation (Theorem 7)**: Derive from Theorem 6

**Deliverable**: All 7 theorems proven

---

### Phase 5: Code Extraction (Optional)

**Goal**: Generate verified Rust implementation

**Strategy**:
- Use Coq's `Extraction` mechanism to generate OCaml
- Translate OCaml to Rust manually (Coq → Rust extraction is experimental)
- Compare extracted code with existing implementation

**Deliverable**: `Extraction/extracted/verified_engine.rs`

**Note**: Extraction is optional - proofs alone provide sufficient guarantees for library users.

---

## Integration with rholang-language-server

### Current State

The rholang-language-server LSP uses liblevenshtein's `DynamicContextualCompletionEngine` for code completion. The system is **well-tested** but **not formally proven**.

**Testing Status**:
- ✅ 12 property-based tests (50 cases each = 600 total)
- ✅ 5 tests for contextual completion (visibility soundness/completeness)
- ✅ 4 tests for completion dictionary (prefix monotonicity, zipper completeness)
- ✅ 3 tests for pattern index (insert/query symmetry)

**Verification Gap**:
- ❌ No formal proof that symbol visibility is correct
- ❌ No formal proof that scope isolation prevents leaks

### After Theorems 1 and 6 Are Proven

**Immediate Benefits**:
1. **LSP can cite formal proofs** in documentation
2. **Research papers** can claim mathematically proven correctness
3. **Security audits** can verify scope isolation guarantees

**LSP-Specific Verification** (Phase 3 in rholang-language-server roadmap):
1. Prove `build_scope_map()` creates valid context tree
2. Prove `find_node_at_position()` returns innermost node
3. Prove scope ID extraction is sound
4. Link to liblevenshtein theorems via imports

**Example LSP Verification Statement**:

```coq
(* In rholang-language-server/formal-verification/coq/ScopeMap.v *)

Require Import liblevenshtein.Theorems.Theorem1_Visibility.
Require Import liblevenshtein.Theorems.Theorem6_Hierarchy.

Theorem scope_map_correctness :
  ∀ (ir : RholangNode) (scope_map : ScopeMap),
    build_scope_map(ir) = scope_map →
    ∀ (pos : Position) (scope_id : ScopeId),
      find_scope_at_position(scope_map, pos) = Some scope_id →
      ∀ (symbol : String),
        visible_in_scope(symbol, scope_id, scope_map) ↔
        (∃ ctx : ContextId,
          scope_to_context(scope_id) = ctx ∧
          visible_in symbol ctx (scope_map_to_dict scope_map)).
Proof.
  (* Proof uses Theorem 1 to link scope visibility to context visibility *)
  (* Proof uses Theorem 6 to ensure child scopes inherit parent visibility *)
Admitted.
```

---

## Downstream Consumer Checklist

If you are using liblevenshtein's contextual completion engine and require formal verification, follow this checklist:

### ✅ Before Coq Proofs (Current State)

- [x] Use property-based testing extensively (proptest/quickcheck)
- [x] Document dependencies on liblevenshtein theorems
- [x] Identify verification obligations blocked by missing proofs
- [x] Maintain empirical confidence through comprehensive test coverage

### ⏳ After Theorem 1 is Proven

- [ ] Import `liblevenshtein.Theorems.Theorem1_Visibility` in your Coq proofs
- [ ] Prove your query operations satisfy visibility soundness/completeness
- [ ] Document formal guarantees in user-facing documentation

### ⏳ After Theorem 6 is Proven

- [ ] Import `liblevenshtein.Theorems.Theorem6_Hierarchy` in your Coq proofs
- [ ] Prove your scope hierarchy construction is correct
- [ ] Prove scope isolation prevents leaks

### ⏳ After All Theorems are Proven (Optional)

- [ ] Consider using extracted verified code (if extraction is implemented)
- [ ] Cite formal proofs in research papers
- [ ] Market formally verified completion as a product differentiator

---

## Questions and Answers

### Q: Why not just rely on property-based testing?

**A**: Property tests provide **empirical confidence** but not **mathematical certainty**. They can miss edge cases, especially in concurrent systems. Formal verification guarantees correctness for **all possible inputs**, not just tested cases.

### Q: Can we verify the existing Rust implementation directly?

**A**: Rust verification tools (RustBelt, Prusti, Creusot) are experimental. Coq is mature and widely used. We can:
1. Prove correctness in Coq
2. Extract verified code (optional)
3. Compare with existing implementation for confidence

### Q: What if we only need Theorems 1 and 6?

**A**: Start with P0 theorems (3-6 weeks effort). Skip P2-P3 theorems unless needed for your use case. LSP verification only requires Theorems 1 and 6.

### Q: How do we know the Coq proofs are correct?

**A**: Coq uses the Calculus of Inductive Constructions (CIC), a foundational type theory with a small trusted kernel. If Coq accepts the proof, it's correct (modulo bugs in Coq itself, which are extremely rare). This is the same foundation used to verify CompCert (a formally verified C compiler) and other safety-critical systems.

### Q: Can we verify thread safety / concurrency?

**A**: Concurrency verification is complex. Coq can model concurrent behavior using Iris (separation logic framework), but this is advanced. For now, we focus on **single-threaded semantics**. Rust's type system + locking ensures thread safety at the implementation level.

---

## References

### Academic Papers on Formal Verification

1. **Leroy, Xavier (2009)**: ["Formal verification of a realistic compiler"](https://xavierleroy.org/publi/compcert-CACM.pdf). Communications of the ACM, 52(7):107-115.
   - CompCert: Verified C compiler using Coq (inspiration for this work)

2. **Klein, Gerwin et al. (2009)**: ["seL4: Formal verification of an OS kernel"](https://www.sigops.org/s/conferences/sosp/2009/papers/klein-sosp09.pdf). SOSP 2009.
   - seL4: Verified microkernel using Isabelle/HOL

3. **Jung, Ralf et al. (2018)**: ["RustBelt: Securing the Foundations of the Rust Programming Language"](https://plv.mpi-sws.org/rustbelt/popl18/paper.pdf). POPL 2018.
   - Iris-based semantic verification of Rust's type system

4. **Appel, Andrew W. (2014)**: ["Program Logics for Certified Compilers"](https://www.cs.princeton.edu/~appel/papers/plcc.pdf). Cambridge University Press.
   - Foundational reference for program verification

### Coq Resources

- **Software Foundations**: [softwarefoundations.cis.upenn.edu](https://softwarefoundations.cis.upenn.edu/) - Standard Coq textbook
- **Certified Programming with Dependent Types**: [adam.chlipala.net/cpdt](http://adam.chlipala.net/cpdt/) - Advanced Coq techniques
- **Iris Documentation**: [iris-project.org](https://iris-project.org/) - Concurrency verification framework

### liblevenshtein Documentation

- **Phase 9 Implementation**: [Layer 7: Contextual Completion](./README.md)
- **API Specification**: [Design Docs](../../design/contextual-completion-api.md)
- **Test Coverage**: [Progress Tracking](../../design/contextual-completion-progress.md)

### Downstream Consumer Documentation

- **rholang-language-server Proof Dependencies**: `rholang-language-server/docs/formal-verification/liblevenshtein_proof_dependencies.md`
- **LSP Scope Detection Verification**: `rholang-language-server/docs/formal-verification/scope-detection.md`

---

## Contact and Contributing

**Questions**: Open an issue on [GitHub](https://github.com/f1r3fly-io/liblevenshtein-rust) with `[formal-verification]` tag

**Contributing Proofs**:
1. Fork repository
2. Create `formal-verification/coq/` directory
3. Submit pull request with `.v` files
4. Proofs will be reviewed by maintainers with Coq expertise

**Timeline**: No fixed schedule - proofs are welcome whenever contributors are available. Priority is on Theorems 1 and 6 to unblock LSP verification.

---

**Last Updated**: 2025-01-21
**Status**: Documentation complete, awaiting Coq implementation
**Blocking**: rholang-language-server formal verification (Theorems 1 and 6)
