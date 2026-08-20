# Java parity causal harness

This directory contains diagnostic-only tools for explaining the performance
gap with `liblevenshtein-java`. They do not change production algorithms.
The normative experiment lifecycle, statistical model, profiling rules, and
acceptance criteria are in
[`docs/benchmarks/optimization-and-profiling-methodology.md`](../../docs/benchmarks/optimization-and-profiling-methodology.md).

Generate and verify the deterministic anchor strata and structural shapes:

```console
python3 benchmarks/causal/generate_corpora.py --output target/benchmark-corpora/java-parity
python3 benchmarks/causal/generate_corpora.py --output target/benchmark-corpora/java-parity --verify
```

The manifest labels every cell with its explicit `unit_domain` and contains
sorted and deterministically shuffled byte, Unicode-scalar, and packed-`u64`
shapes. Matrix runners dispatch on that field rather than corpus naming.
Packed-`u64` sorted files use the
backend's actual lexicographic order—little-endian numeric values for each
eight-byte chunk—not ordinary UTF-8 lexical order. In the construction matrix,
sorted cells call `from_sorted_terms` and shuffled cells call the unordered
bulk `from_terms`; online `insert` publication is a distinct API and is
measured only when `--constructor stream` is requested explicitly.

Build timing and work-counter drivers separately. `resource-profiling` enables
the resource harness without hot-loop counters; `causal-resource-profiling`
adds both consumer and provider work counters:

```console
cargo build --release --bin causal_query_profile
cargo build --release --features resource-profiling --bin causal_resource_profile
cargo build --release --features perf-instrumentation --bin causal_query_profile
cargo build --release --features causal-resource-profiling --bin causal_resource_profile
cargo build --release --bin causal_construction_bench
```

Run the complete structural work matrix after generating the corpora:

```console
benchmarks/causal/run-counter-matrix.sh \
  target/benchmark-corpora/java-parity/manifest.json \
  target/benchmark-results/counter-matrix
benchmarks/causal/run-construction-matrix.sh \
  target/benchmark-corpora/java-parity/manifest.json \
  target/benchmark-results/construction-matrix
benchmarks/causal/run-legacy-structure-probe.sh benchmarks/cross-language/workload/dictionary.txt /tmp/liblev-java-structure
```

The construction runner applies the strict selected-CPU and complete-LLC gate
before and after every cell, continuously monitors competing benchmark
processes, and records the executable and manifest SHA-256 digests. Generated
corpora live under disk-backed `target/benchmark-corpora` rather than `/tmp`,
which is commonly a RAM-backed `tmpfs` on benchmark hosts. Each cell hashes its
dictionary and query file against the manifest before and after timing and
records an outside-timed length, membership, and deterministic-checksum proof
from an extra construction. Pass
`--resume` to retain only contract-complete cells whose corpus, run
configuration, and executable digest still match; a cell rejected before or
after timing is rerun rather than partially admitted.

Run AMD uProf, Heaptrack, or `perf stat` through the headless wrapper. The
output directory must not already exist:

```console
benchmarks/causal/profile-headless.sh uprof /tmp/causal-uprof -- target/release/causal_query_profile ARGS...
benchmarks/causal/profile-headless.sh heaptrack /tmp/causal-heap -- target/release/causal_query_profile ARGS...
benchmarks/causal/profile-headless.sh perf-stat /tmp/causal-perf -- target/release/causal_query_profile ARGS...
```

The Heaptrack path always passes `--record-only` and analyzes only with
`heaptrack_print`. It must never call `heaptrack --analyze`, `heaptrack -a`, or
`heaptrack_gui`.

Paired timing runners call `host-load-admission.py` before warmup and before
every measured pair. The gate samples host-wide `/proc/stat` counters, resolves
the selected CPU's hardware-thread and last-level-cache topology from sysfs,
and rejects direct-core, sibling, or shared-cache contention. Load on other
Threadripper CCDs is retained in the adjacent `*-host-load.jsonl` evidence but
does not waste otherwise independent cores. A rejected gate exits with status
3 before another pair is measured; the existing CSV remains auditable and must
not be presented as a complete experiment. Every load record carries `warmup`
or `pair-N` as its label. A rejected `pair-N` record proves that the runner
stopped before measuring that pair.

The bounded query-cache policy matrix is deterministic and fixes the
benchmark-only AHash key before constructing any cache. Production caches keep
independent randomized keys. The matrix compares TinyLFU+SIEVE with dense,
purpose-built FIFO, LRU, SIEVE-only, and aging exact-LFU controls across hot
hits, scan pollution, a disjoint phase, and Zipf locality. Every raw row
includes latency, allocator calls/bytes, cache quality, resident entries,
logical weight, exact checksum, execution order, and binary SHA-256:

```console
cargo run --release --features benchmark-controls --example query_cache_policy_matrix
```

For decision evidence, build once and use the topology-gated 51-replicate
runner. It writes the CSV and adjacent host-load ledger without overwriting
existing evidence:

```console
cargo build --release --features benchmark-controls --example query_cache_policy_matrix
benchmarks/causal/run-query-cache-policy-matrix-experiment.sh \
  target/release/examples/query_cache_policy_matrix \
  benchmarks/causal/evidence/YYYY-MM-DD/query-cache-policy-performance.csv
```

The complete fairness model, limitations, and interpretation rules are in
[`docs/benchmarks/query-cache-policy-matrix.md`](../../docs/benchmarks/query-cache-policy-matrix.md);
the row contract is
[`schemas/query-cache-policy-matrix.schema.json`](schemas/query-cache-policy-matrix.schema.json).
The completed 2026-08-19 evidence contains 51 replicates per arm and an
accepted pgmcp experiment 299 decision for TinyLFU+SIEVE versus aging exact
LFU on Zipf locality. The runner also supports strict `--resume`: it retains
only complete, digest-matching replicates, rebuilds its accepted host-load
ledger from committed per-replicate admission sidecars, separates rejected
gates, and refuses any run with a nonempty foreign-contention ledger. Benign
monitor diagnostics do not make the evidence unresumable.

The all-backend propagation matrix exercises every production
`DictionaryNode` family across its applicable byte, Unicode-scalar, and `u64`
domains. It emits explicit inapplicable rows, construction and query resource
metrics, and exact semantic checksums for Standard, OSA, merge-and-split, and
unrestricted Damerau–Levenshtein:

```console
cargo build --release \
  --example backend_propagation_matrix \
  --features "benchmark-controls pathmap-backend persistent-artrie"
benchmarks/causal/run-backend-propagation-matrix.sh \
  target/release/examples/backend_propagation_matrix \
  target/release/examples/backend_propagation_matrix \
  benchmarks/causal/evidence/YYYY-MM-DD/backend-propagation-matrix.csv \
  51 3 256 64 1 2
```

The complete admission, bounds, and comparison protocol is in
[`docs/benchmarks/backend-propagation-evidence.md`](../../docs/benchmarks/backend-propagation-evidence.md);
the row contract is
[`schemas/backend-propagation-matrix.schema.json`](schemas/backend-propagation-matrix.schema.json).
All runtime A/B switches are isolated behind `benchmark-controls`; ordinary
production builds do not read causal-control environment variables.

The victim-planning experiment uses the same compiled binary for both arms,
alternates arm order, fixes the benchmark-only hash seed, and compares the
allocating transactional reference with the production circular-span planner:

```console
cargo build --release --features benchmark-controls --bin causal_query_cache_profile
benchmarks/causal/run-query-cache-victim-plan-experiment.sh \
  target/release/causal_query_cache_profile /tmp/query-cache-victim-plan.csv
```

The accepted 2026-08-18 evidence is
`allocation-free-in-place-query-cache-victim-planning.csv`, with its complete
topology ledger in the adjacent `*-host-load.jsonl` file. The preserved
`invalidated-mark-array-query-cache-victim-planning.csv` is deliberately not
decision evidence: its arms used independent randomized hash keys and its
first in-place implementation mutated the SIEVE hand/reference state when an
admission was rejected. Its filename records that invalidation so it cannot be
silently mistaken for the corrected result.

The transition/traversal residual experiments use the same topology gate and
retain exact result and work signatures. The class-zero runner compares the
source-row-local packed-DFA result cache with the ordinary physical target
probe. The generic environment runner compares any single same-binary
resource-profiling control; it was used for static packed-row dispatch. The
parent-path runner is intentionally a two-binary comparison because it changes
type layout, and records both executable digests in every row:

```console
benchmarks/causal/run-class-zero-row-cache-experiment.sh \
  BINARY DICTIONARY QUERIES OUTPUT.csv
benchmarks/causal/run-resource-env-experiment.sh \
  BINARY DICTIONARY QUERIES OUTPUT.csv \
  LIBLEVENSHTEIN_CAUSAL_DISABLE_STATIC_PACKED_ROWS
benchmarks/causal/run-parent-path-compaction-experiment.sh \
  CONTROL_BINARY TREATMENT_BINARY DICTIONARY QUERIES OUTPUT.csv
```

The accepted evidence is `class-zero-row-cache.csv`,
`static-packed-source-row-dispatch.csv`, and
`compact-parent-path-metadata.csv`, each with its adjacent host-load ledger.
The latter also has two headless Heaptrack captures summarized in
`compact-parent-path-metadata-heaptrack.csv`; collection used
`heaptrack --record-only` and analysis used only `heaptrack_print`.

`validate_gate.py` checks exact-result equivalence and the construction,
native-query, resource-boundary, and batch-size counter identities. Passing
these identities validates the observations; it does not by itself authorize
production optimization. The pgmcp causal ledger is the decision record for
that gate.
