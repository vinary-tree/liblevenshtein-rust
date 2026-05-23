# Formal Verification Gates

This repo has completed proof islands and known incomplete proof trees. Use the
manifest-driven gates here as the source of truth rather than older status docs.
The remaining proof closure order is tracked in `PROOF_COMPLETION_PLAN.md`.

## Commands

```bash
# Cheap audit: counts all active proof gaps and checks trusted metadata.
scripts/verify-formal.sh audit

# Trusted-file gate: no Admitted. and no unallowlisted assumptions in trusted files.
scripts/verify-formal.sh trusted

# Build trusted Coq files through cgroup memory caps.
scripts/verify-formal.sh coq-trusted

# Build one Coq file through a selected memory profile.
scripts/verify-formal.sh coq-file exceptional docs/verification/core/theories/Automaton/Completeness.v

# Run bounded TLA+ checks through JVM and cgroup caps.
scripts/verify-formal.sh tla
```

## CI

`.github/workflows/formal-verification.yml` runs the trusted-scope audit on PRs
and pushes that touch verification files. Full Coq/TLC execution is
`workflow_dispatch` only and requires a self-hosted Linux runner so
`systemd-run` can enforce the memory caps.

## Resource Profiles

All Rocq/TLC execution goes through `systemd-run --user --scope` by default.
If user-scoped systemd is unavailable, the runner falls back to `prlimit --as`
with a conservative virtual-address ceiling so Rocq can start without running
unbounded. The runner refuses fully uncapped proof execution unless
`FORMAL_VERIFY_ALLOW_UNCAPPED=1` is explicitly set.

| Profile | MemoryMax | CPUQuota | Use |
|---|---:|---:|---|
| `light` | 8G | 400% | small lemmas and metadata checks |
| `standard` | 32G | 800% | ordinary Coq files and small TLC models |
| `heavy` | 96G | 1200% | large proof projects |
| `exceptional` | 128G | 1200% | serial-only memory-heavy proofs |

## Policy

- No trusted file may contain active `Admitted.`.
- Axioms, parameters, conjectures, and trusted-file hypotheses must be listed
  in `ASSUMPTIONS.tsv` with a citation before they are allowed in trusted scope.
- Broad algorithm-correctness axioms are not acceptable as final closure.
  Decompose them into narrow lemmas, cited mathematical assumptions, or proven
  local facts.
- Debug and legacy files are audited but do not support public correctness
  claims until promoted in `FORMAL_VERIFICATION_MANIFEST.tsv`.
