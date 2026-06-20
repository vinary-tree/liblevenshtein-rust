# 06 · Contextual Completion Engine

**What you'll learn.** How to drive an IDE-style **incremental** completion engine: build
identifiers one keystroke at a time, manage an in-progress **draft** per scope, create
**checkpoints** you can **undo** to, and arrange scopes into a **hierarchy** where child
scopes see their parents' identifiers but not vice-versa. This sits one layer above the
fuzzy maps of [tutorial 05](../05-values/README.md): the engine fuses *draft* text and
*finalized* terms into a single fuzzy query.

---

## The concept

### What the engine adds over a bare dictionary

`DynamicContextualCompletionEngine` (feature `pathmap-backend`) models the lifecycle of a
symbol *as you type it*. The key vocabulary:

- **Context** — a lexical scope with an integer id, arranged in a tree (global → function
  → block). A context sees its own terms plus every **ancestor's**.
- **Draft** — the in-progress, not-yet-committed identifier for a context, built up
  character by character (`insert_char`) or in bulk (`insert_str`).
- **Finalized term** — a committed identifier, added directly (`finalize_direct`) or by
  promoting the current draft (`finalize`).
- **Checkpoint / undo** — a saved draft position you can roll back to, for editor undo.
- **Query fusion** — `complete` searches *both* the draft and the finalized terms of all
  visible scopes in one call, returning candidates flagged with `is_draft` and the set of
  `contexts` they came from.

> Terms defined. A **scope** is a region of code over which a name is visible.
> "Hierarchical visibility" means a name declared in an outer scope is usable in inner
> scopes (a block sees its function's locals; the function does *not* see the block's).

### How a completion is answered

`complete(context, query, k)` runs two matchers and merges them: a small naïve
Levenshtein pass over the (short) **draft**, and the automaton-based fuzzy match over the
**finalized** dictionary, restricted to terms visible from `context` by walking the
context tree to the root. Drafts are deduplicated against finalized terms and, being the
text the user is actively typing, sort to the front.

### Why incremental state matters

A real editor doesn't re-tokenize the world on every keystroke — it maintains *typing
state*. Modeling drafts, checkpoints, and scope visibility explicitly lets the engine
answer "what can I complete *right here, right now*?" cheaply and correctly, including
undo, without rebuilding any dictionary.

![Context scope tree: nested global → function → block contexts, with arrows showing that each child can see identifiers finalized in its ancestors while parents cannot see descendants.](../../diagrams/contextual/context-scope-tree.svg)

---

## Walking through `examples/contextual_completion.rs`

### 1 · Build a context hierarchy

Create a root context, then nest children under it. `create_child_context` returns a
`Result`, so handle the failure case explicitly:

```rust
use liblevenshtein::contextual::DynamicContextualCompletionEngine;
use liblevenshtein::transducer::Algorithm;

let engine = DynamicContextualCompletionEngine::with_algorithm(Algorithm::Standard);

let global   = engine.create_root_context(0);
let function = engine.create_child_context(1, global).expect("create child failed");
let block    = engine.create_child_context(2, function).expect("create child failed");
```

### 2 · Finalize identifiers into scopes

Commit known names directly into the scope where they're declared:

```rust
engine.finalize_direct(global, "global_var").expect("finalize failed");
engine.finalize_direct(global, "global_helper").expect("finalize failed");
engine.finalize_direct(function, "result").expect("finalize failed");
engine.finalize_direct(function, "process").expect("finalize failed");
```

### 3 · Type a draft incrementally and complete against it

Each `insert_char` extends the block scope's draft; `complete` fuzzily matches the query
against *both* the draft and all visible finalized terms (here it should surface
`global_helper` from an ancestor scope):

```rust
engine.insert_char(block, 'h').expect("insert failed");
engine.insert_char(block, 'e').expect("insert failed");
println!("draft = {:?}", engine.get_draft(block).expect("no draft"));   // "he"

for comp in engine.complete(block, "hel", 2) {
    println!("  {} (distance: {}, draft: {})", comp.term, comp.distance, comp.is_draft);
}
```

### 4 · Checkpoint, type more, then undo

A checkpoint saves the current draft position; later edits can be rolled back to it —
exactly an editor's undo:

```rust
engine.insert_char(block, 'l').expect("insert failed");   // draft = "hel"
engine.checkpoint(block).expect("checkpoint failed");     // save here

engine.insert_char(block, 'l').expect("insert failed");
engine.insert_char(block, 'o').expect("insert failed");   // draft = "hello"

engine.undo(block).expect("undo failed");                 // back to "hel"
assert_eq!(engine.get_draft(block).expect("no draft"), "hel");
```

### 5 · Promote a draft and observe hierarchical visibility

`finalize` turns the current draft into a committed term and clears the draft. Afterward,
the block scope sees *all* ancestors' terms, while the global scope sees none of its
descendants':

```rust
engine.insert_char(block, 'l').expect("insert failed");
engine.insert_char(block, 'o').expect("insert failed");
let term = engine.finalize(block).expect("finalize failed");   // "hello"
assert!(!engine.has_draft(block));

// Visible from block (sees global + function + block):
let from_block = engine.complete(block, "help", 2);
// NOT visible from global (cannot see block's "hello"):
let from_global = engine.complete(global, "hello", 1);          // empty
```

The example finishes by showing a draft *overriding* a finalized term of the same name
(only one `hello*` appears, the draft first) and `discard` clearing an abandoned draft.

![Draft / checkpoint lifecycle: the state machine of a draft — empty → typing (insert_char) → checkpointed → undo/redo → finalized (promote) or discarded — and how each transition affects what complete() returns.](../../diagrams/contextual/draft-checkpoint-lifecycle.svg)

---

## Run it

This example requires the `pathmap-backend` feature:

```bash
cargo run --example contextual_completion --features pathmap-backend
```

> **crates.io note.** `pathmap-backend` uses a git dependency and is unavailable from a
> plain `crates.io` install — build from source to enable it.

For a related, value-driven take on lexical scopes see
`examples/hierarchical_scope_completion.rs` (also `pathmap-backend`).

---

## Key takeaways

- **`DynamicContextualCompletionEngine`** models typing state: contexts (scopes), drafts,
  checkpoints/undo, and hierarchical visibility.
- **`complete(ctx, q, k)`** fuses a naïve match over the short *draft* with the
  automaton-based match over *finalized* terms visible from `ctx`, flagging each result
  with `is_draft` and its `contexts`.
- Child scopes see ancestors' identifiers; **parents never see descendants'** — the rule
  that makes scope-correct completion possible from one engine.

---

[← 05 · Values & Fuzzy Maps](../05-values/README.md) · Next: [07 · Performance →](../07-performance/README.md)

[← Documentation Index](../../README.md)
