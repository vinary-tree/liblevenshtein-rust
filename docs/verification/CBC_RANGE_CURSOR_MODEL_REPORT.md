# CBC bounded range cursor model report

[RangeCursorOwnership.tla](tla/RangeCursorOwnership.tla) represents three
distinct original occurrences. Two share one physical dictionary node; their
identities remain separate. A candidate moves from the unseen cursor suffix
to active scoring, then to sound exclusion or a pending exact result, and
finally into the emitted set. `Pause` and `Resume` preserve ownership in idle
or pending-result phases. This is a finite model of the lifecycle described
in [CBC section 4](../theory/certified-metric-automata.md#4-a-session-invariant-that-covers-the-whole-search).

## Reproduction and results

From the repository root, run TLC separately for each configuration so its
time-named state directories do not collide:

```sh
tlc -config docs/verification/tla/RangeCursorOwnership.cfg docs/verification/tla/RangeCursorOwnership.tla
tlc -config docs/verification/tla/RangeCursorOwnershipFair.cfg docs/verification/tla/RangeCursorOwnership.tla
tlc -config docs/verification/tla/RangeCursorOwnershipNoPrivate.cfg docs/verification/tla/RangeCursorOwnership.tla
tlc -config docs/verification/tla/RangeCursorOwnershipSkipCollision.cfg docs/verification/tla/RangeCursorOwnership.tla
tlc -config docs/verification/tla/RangeCursorOwnershipEarlyFinish.cfg docs/verification/tla/RangeCursorOwnership.tla
```

On 2026-09-29 with TLC 2.19, the clean safety and strong-fairness models each
explored 16 distinct states and passed. The fairness configuration checks
eventual completion when pick, classify, publish, resume, and finish actions
are strongly fair. Without that environment/scheduler premise, the model
does not claim termination: a client could remain paused or repeatedly pause.

| Mutant configuration | Expected violation | Observed TLC result |
|---|---|---|
| `NoPrivate` | The first picked original disappears from coverage | `Coverage` violated after `Pick`, exit 12 |
| `SkipCollision` | Node-level dedup skips the second original at a shared node | `Coverage` violated after `SkipCollision`, exit 12 |
| `EarlyFinish` | Completion ignores the last exact match waiting for a result slot | `NoFalseCompletion` violated in `waiting` with `pendingMatch = "b"`, exit 12 |

The model proves only its finite state space. Its `Eligible` set stands for an
authoritative exact scoring decision; it does not prove a numeric lower bound,
the Rust scorer, physical node enumeration, output ordering, resource charges,
allocation failure, or executable correspondence. The companion
[Rocq ownership proof](temporal_automata/theories/CertifiedSearchSession.v)
establishes generic finite-step invariants and exact membership under those
explicit classification premises.
