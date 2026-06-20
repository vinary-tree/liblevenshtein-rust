# Contextual Completion — Implementation

**Internal components of the contextual completion engine (Layer 7).**

The contextual completion layer provides scope-aware, hierarchical code
completion. These documents describe its constituent components: the engine that
fuses draft and finalized queries, the lexical-scope context tree, the
in-memory draft buffer, and the checkpoint stack that enables time-travel
undo/redo. For consumer-facing scenarios, see the
[use-cases](../use-cases/README.md) directory.

## Components

| Document | Purpose |
|----------|---------|
| [completion-engine.md](completion-engine.md) | `ContextualCompletionEngine` — fuses drafts with finalized terms and manages the context tree. |
| [context-tree.md](context-tree.md) | `ContextTree` — the lexical-scope hierarchy and its visibility rules. |
| [draft-buffer.md](draft-buffer.md) | `DraftBuffer` — in-memory storage of work-in-progress (unfinalized) terms. |
| [checkpoint-system.md](checkpoint-system.md) | `CheckpointStack` — checkpoint/restore for time-travel undo/redo of completion state. |

**Status: Living reference.**

[← Documentation Index](../../../README.md)
