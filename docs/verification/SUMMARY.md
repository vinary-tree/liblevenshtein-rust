# Verification Summary

**Project**: liblevenshtein formal verification and scientific validation

## Current Scope

The verification tree contains active Rocq, TLA, documentation, and empirical
validation artifacts for:

- core edit-distance and automaton contracts,
- phonetic rewrite and NFA behavior,
- grammar-correction lattice and pipeline contracts,
- interval-relaxed MSM trie search,
- product-automaton model checking,
- Rust property tests and scientific benchmark gates.

## Active Proof Areas

| Area | Primary location | Current evidence |
|------|------------------|------------------|
| Core distance and automata | `docs/verification/core/theories/` | maintained Rocq modules for distance, traces, automata, composition, lower bounds, and triangle support |
| Phonetic verification | `docs/verification/phonetic/` and `rocq/liblevenshtein/` | split/composition/cost-accounting proof files plus Rust phonetic integration tests |
| Grammar verification | `docs/verification/grammar/theories/` | checked core, layer, composition, and NFA proof slices with no active proof-escape terms in `.v` sources |
| TLA product automaton | `docs/verification/tla/` | bounded witness model checked by TLC |
| MSM interval search | `docs/verification/msm/` | design and proof notes aligned with exact MSM-over-trie Rust tests |

## Verification Gates

Use capped commands for proof and test work. The grammar suite currently fits a
2 GiB capped single-job build:

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  make -C docs/verification/grammar -j1
```

The core suite is much larger. Current capped observations:

| Cap | Last reached file | Unit result |
|-----|-------------------|-------------|
| 2 GiB | `DPMatrix/SnocLemmas.v` | `oom-kill` during Rocq compilation |
| 4 GiB | trace layer after `Automaton/MainTheorem.v` | `oom-kill` during Rocq compilation |
| 8 GiB | `Trace/DamerauTrace.v` | `oom-kill` during Rocq compilation |

```bash
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
  make -C docs/verification/core/theories -j1
```

For Rust-side gates, prefer focused targets with `CARGO_BUILD_JOBS=1`, `-j1`,
and `-- --test-threads=1` when the target is large enough to threaten RSS.

## Audit Commands

Proof escape scan:

```bash
rg -n "^\s*(Admitted\.|admit\.|Axiom |Parameter |Conjecture |Hypothesis )" \
  docs/verification rocq/liblevenshtein -g '*.v'
```

Source stale-marker scan:

```bash
rg -n "$STALE_MARKER_PATTERN" \
  src tests benches examples rocq/liblevenshtein -g '*.rs' -g '*.v'
```

## Maintenance Notes

- Treat proof files, Rust property tests, and recorded benchmark gates as the
  authoritative evidence for current status.
- Treat older phase reports as historical context unless this summary or a
  current README cites them as active evidence.
- Keep generated Rocq artifacts out of commits unless they are intentionally
  tracked.
- Clean task-owned `/tmp` scratch directories after capped verification runs.
