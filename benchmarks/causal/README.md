# Java parity causal harness

This directory contains diagnostic-only tools for explaining the performance
gap with `liblevenshtein-java`. They do not change production algorithms.

Generate and verify the deterministic anchor strata and structural shapes:

```console
python3 benchmarks/causal/generate_corpora.py --output /tmp/liblev-corpora
python3 benchmarks/causal/generate_corpora.py --output /tmp/liblev-corpora --verify
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
benchmarks/causal/run-counter-matrix.sh /tmp/liblev-corpora/manifest.json /tmp/liblev-counter-matrix
benchmarks/causal/run-construction-matrix.sh /tmp/liblev-corpora/manifest.json /tmp/liblev-construction-matrix
benchmarks/causal/run-legacy-structure-probe.sh benchmarks/cross-language/workload/dictionary.txt /tmp/liblev-java-structure
```

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
independent randomized keys:

```console
cargo run --release --features benchmark-controls --example query_cache_policy_matrix
```

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
