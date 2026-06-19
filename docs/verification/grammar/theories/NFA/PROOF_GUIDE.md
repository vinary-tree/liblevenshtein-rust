# Proof Maintenance Guide for NFA/Phonetic Verification

**Status**: current source files compile without proof escape hatches.
**Verification command**: run targeted `rocq c` commands under `systemd-run`
with `MemoryMax` and `MemorySwapMax=0`.

## Current Contract Shape

The NFA layer separates executable definitions from evidence-premise contracts:

- `Completeness.v` proves edit-sequence and acceptance contracts when an
  executable acceptance witness is supplied.
- `Soundness.v` proves edit-witness contracts when an explicit edit witness or
  traced path carries the operation evidence.
- `Soundness.v` now uses `extract_edit_sequence_full` for position-only paths
  and `PathEntry` traces when exact operation membership is required.
- `Operations.v` proves the bounded diagonal and well-formedness properties for
  the Phase 1 phonetic operation set.
- `Automaton.v`, `Transitions.v`, and the complexity files keep the executable
  NFA state-transition model and its structural invariants.

This shape is intentional: plain `AutomatonPath = list Position` does not store
which operation produced each edge, so exact operation-membership proofs use
traced paths.

## No-Escape Audit

Run this from the repository root after proof edits:

```bash
rg -n "Admitted\\.|admit\\.|Axiom|Parameter|Conjecture|Hypothesis" \
  docs/verification/grammar/theories/NFA \
  rocq/liblevenshtein \
  -g '*.v'
```

Expected result: no matches.

## Focused Compile Order

For the NFA theory files, refresh dependencies in this order when `.vo` files
are stale:

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  rocq c -Q docs/verification/core/theories Liblevenshtein.Core \
         -Q docs/verification/grammar/theories Liblevenshtein.Grammar.Verification \
         docs/verification/grammar/theories/NFA/Types.v

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  rocq c -Q docs/verification/core/theories Liblevenshtein.Core \
         -Q docs/verification/grammar/theories Liblevenshtein.Grammar.Verification \
         docs/verification/grammar/theories/NFA/Operations.v

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  rocq c -Q docs/verification/core/theories Liblevenshtein.Core \
         -Q docs/verification/grammar/theories Liblevenshtein.Grammar.Verification \
         docs/verification/grammar/theories/NFA/Automaton.v

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  rocq c -Q docs/verification/core/theories Liblevenshtein.Core \
         -Q docs/verification/grammar/theories Liblevenshtein.Grammar.Verification \
         docs/verification/grammar/theories/NFA/Transitions.v

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  rocq c -Q docs/verification/core/theories Liblevenshtein.Core \
         -Q docs/verification/grammar/theories Liblevenshtein.Grammar.Verification \
         docs/verification/grammar/theories/NFA/Completeness.v

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  rocq c -Q docs/verification/core/theories Liblevenshtein.Core \
         -Q docs/verification/grammar/theories Liblevenshtein.Grammar.Verification \
         docs/verification/grammar/theories/NFA/Soundness.v
```

## Development Rules

- Prefer executable definitions plus small lemmas over broad theorem statements
  that only restate a premise.
- If a theorem needs operation membership, use `PathEntry` traces or add an
  explicit witness carrying the operation.
- Keep source comments descriptive rather than aspirational: describe the
  abstraction currently modeled by the file.
- Re-run the no-escape audit and focused compile slice before committing.
