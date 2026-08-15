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

Build the work-counter drivers separately from timing binaries:

```console
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

`validate_gate.py` checks exact-result equivalence and the construction,
native-query, resource-boundary, and batch-size counter identities. Passing
these identities validates the observations; it does not by itself authorize
production optimization. The pgmcp causal ledger is the decision record for
that gate.
