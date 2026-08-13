# Cross-Language Benchmark Program

Measured, reproducible performance comparison across three axes:

1. **Java vs Java** — the liblevenshtein-rust JVM binding
   (`io.vinarytree:liblevenshtein` 0.10.0, Java 22 FFM) against legacy
   [liblevenshtein-java 3.0.0](https://github.com/universal-automata/liblevenshtein-java)
   (`com.github.universal-automata:liblevenshtein:3.0.0`, Maven Central),
   both on the same JDK.
2. **JavaScript vs JavaScript** — the `@vinary-tree/liblevenshtein` Node
   facade (native N-API, WASM, and WASI backends) against the legacy
   compiled-CoffeeScript build 2.0.4, vendored byte-exactly from the
   `gh-pages` branch that served the original interactive demo, both on the
   same Node.
3. **Cross-language atlas** — the identical protocol over every language
   facade of the Rust implementation (C, C++, Python, Java, Clojure,
   JavaScript, ClojureScript, .NET, Go, Swift, Ruby, Fortran, OCaml,
   Haskell, Lua) plus the pure-Rust core, relating each binding's cost to
   the raw automaton.

This directory is self-contained: nothing in `bindings/`, `src/`, the
sibling repos, or CI is modified by the program.

## Layout

| Path | Contents |
|---|---|
| `harnesses/common/PROTOCOL.md` | **The normative spec** — CLI contract, timing loop, checksum, fairness rules. Start here. |
| `workload/` | Committed inputs: `dictionary.txt` (79,343 aspell en_US words, byte-sorted), `queries/*.txt` (8 primary sets × 1000), `queries-meta/*.jsonl` (audit trail), `provenance.json` (package versions + SHA-256s), `generate_workload.py` (deterministic regenerator, SplitMix64 seed 42). |
| `schema/` | `result.schema.json` (one JSON per cell), `environment.schema.json` (one per run). |
| `harnesses/<lang>/` | One harness per language, each implementing PROTOCOL.md. |
| `legacy/javascript/vendor/` | The vendored 2.0.4 legacy JS + `provenance.json` + license. |
| `scripts/` | `doctor.sh` (readiness), `run-one.sh` / `run-all.sh` (orchestration), `gate.py` (correctness pre-gate vs the Rust oracle), `aggregate.py` (stats + tables), `jmh_to_result.py`, `env-capture.sh`, `vendor-legacy-js.sh`. |
| `targets.tsv` | Declarative target manifest (backends, cpusets, phases). |
| `results/` | Git-ignored; timestamped run directories. |
| `.stage/` | Git-ignored; build staging (Haskell pkg-config prefix, Lua modules, compiled harness binaries). |

## Quickstart

```bash
# 0. one-time native builds (release, target-cpu=native)
( cd ../..            && RUSTFLAGS="-C target-cpu=native" cargo build --locked --release --features native-bindings-full )
( cd ../../../libdictenstein && RUSTFLAGS="-C target-cpu=native" cargo build --release --features ffi )

# 1. readiness: toolchains, artifacts, governor, smoke queries
scripts/doctor.sh

# 2. correctness gate, then the full timed matrix (resumable)
scripts/run-all.sh --results results/$(date +%Y%m%d_%H%M%S)

# 3. aggregate + render tables
scripts/aggregate.py results/<run-id>
```

Every raw log is teed into the results directory; `environment.json` pins
toolchain versions, git commits, artifact SHA-256s, governor, and cpusets.

## Methodology in one paragraph

All harnesses read the same two committed input files, assert the
dictionary's byte-sortedness, build the dictionary once (timed separately in
`construct` mode), then run identical warmup-and-sample loops where one
sample is one full pass over the query set with every cursor fully drained
(PROTOCOL.md §5–6). Correctness precedes timing: `gate.py` compares every
target's order-insensitive FNV-1a64 result checksum against the Rust oracle
across the full query-cell family and refuses timing on mismatch; each timed pass re-asserts
its match-count/byte/distance triple against the cell's untimed gate pass.
CPU pinning, the performance governor, fixed JVM heaps, JMH forks for the
Java pair, and per-cell frequency snapshots control the environment; results
report median/MAD/p10/p90 with bootstrap CIs, never bare means.

## Workload provenance

`workload/dictionary.txt` is the aspell en_US master dump
(`aspell 0.60.8.2-2`, `aspell-en 2026.02.25-1`), expanded, filtered to
`^[a-z]+$`, byte-sorted (`LC_ALL=C sort -u` equivalent): 79,343 words.
Query sets are seeded-random samples and mutations (SplitMix64, base seed
42), bucketed by *realized* edit distance verified with reference DPs —
`std-dK` under plain Levenshtein, `tr-dK` under restricted-Damerau (OSA,
matching every implementation's `transposition` algorithm). Mutants that are
themselves dictionary words stay in, labeled `real_word` in
`queries-meta/`. The empty string never occurs (legacy Java 3.0.0's
empty-string bug is designed out rather than special-cased).
`generate_workload.py --verify` detects any drift against `provenance.json`;
regeneration is byte-identical by construction.

## Fairness commitments

- Same pre-sorted dictionary for everyone; legacy builders get their
  documented `isSorted=true` fast path; sorting is excluded from
  construction timing on all sides.
- Legacy JS runs with `sort_candidates(false)` — the new stack returns
  traversal order, so result sorting would be uncompensated extra work.
- Timed loops carry an O(1) accumulator triple, not per-byte hashing;
  checksums are computed in untimed gate passes only.
- Both sides of each head-to-head share one JDK/Node, one cpuset, one heap
  configuration; deviations from language defaults are pinned in PROTOCOL.md
  §10 and echoed in each result's `notes`.
- A target that fails to build or gate lands as an explicit `failed` /
  documented-mismatch row in the report — never a silent omission.

## Deliverables

Rendered into `docs/benchmarks/cross-language/` (results + the Java and
JavaScript migration cases) and `docs/scientific-ledger/` (preregistered
hypotheses → measurements → verdicts, backed by pgmcp experiment records).
