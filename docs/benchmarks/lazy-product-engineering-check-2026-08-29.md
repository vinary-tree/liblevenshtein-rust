# Lazy-product engineering benchmark check — 2026-08-29

**Status:** admitted engineering check, not a release-grade cross-host result
**Source:** working tree of `codex/regresspec-temporal-complete`, based on
`e1fd13da8e5b5b9f54c512c9a054665356fbe341`
**Purpose:** detect gross regressions and choose justified hot-path work after
the compact lazy-product refactor

This record measures the maintained compact zipper and elastic-product
benchmarks after correctness, correspondence, and bounded-memory gates passed.
It is deliberately modest: one host, ten samples, 0.5/0.3-second warmup, and
1.0/0.7-second measurement windows. The results support local engineering
decisions; they are not portable latency promises.

## Environment and admission

- AMD Ryzen Threadripper PRO 5975WX, Linux 7.1.9, x86-64.
- `rustc 1.97.1` (`8bab26f4f`, LLVM 22.1.6) and `cargo 1.97.1`.
- Package tuple: `liblevenshtein = 4.0.0-rc.6`,
  `libdictenstein = 4.0.0-rc.6`; resolution used the committed `Cargo.lock`.
- Every command ran under `MemoryMax=4G` and `MemorySwapMax=0`, with Cargo
  parallelism limited to two jobs.
- The full all-features library and integration/property suites, strict Clippy,
  and the trusted formal gates passed before these timings were interpreted.

Commands:

```text
cargo bench --bench zipper_vs_node_benchmark --features pathmap-backend -j 2 -- \
  --warm-up-time 0.5 --measurement-time 1 --sample-size 10
cargo bench --bench elastic_kernel_benchmarks -j 2 -- \
  --warm-up-time 0.3 --measurement-time 0.7 --sample-size 10
```

## Compact PathMap zipper

The benchmark dictionary contains 115 fixed English terms. The table reports
Criterion point estimates; lower is better.

| Operation | Node before | Zipper before | Node after | Zipper after | After zipper / node |
|---|---:|---:|---:|---:|---:|
| query, cutoff 0 | 1.2666 µs | 2.1737 µs | 1.1417 µs | 1.1474 µs | 1.005× |
| query, cutoff 1 | 4.4932 µs | 8.1187 µs | 4.1509 µs | 4.2055 µs | 1.013× |
| query, cutoff 2 | 11.636 µs | 20.740 µs | 11.794 µs | 11.480 µs | 0.973× |
| five-query batch, cutoff 1 | 23.570 µs | 44.848 µs | 22.515 µs | 22.241 µs | 0.988× |
| iterator creation | 375.21 ns | 336.28 ns | 356.12 ns | 389.21 ns | 1.093× |

The initial compact-frontier treatment established that initialization was not
the traversal bottleneck: PathMap created every child focus and copied its path
before the automaton rejected the edge, the zipper scheduler allocated another
parent spine for live edges, and automaton representation dispatch remained in
the loop. The accepted treatment made a zipper focus consumable as an opaque
`ZipperTraversalNode`, added projection-before-child-construction to
`DictZipper`, specialized PathMap to discard path-only context after that
opaque conversion, and delegated the query to the production `QueryIterator`
core. The immutable TrieRef remains the snapshot owner; the shared parent arena
reconstructs results relative to the supplied focus.

The treatment reduced zipper query latency by 44–50% and brought it within
about 1.3% of direct-node traversal on all measured queries; cutoff 2 and the
batch were slightly faster in this run. Construction alone became 9% slower,
but end-to-end queries reached the stated within-noise architectural target.
Correctness gates additionally prove that a 256-way rejected root constructs
only the one viable child, non-root paths remain relative, snapshot isolation
survives later mutation, and an 8,192-edge query completes on a 256 KiB stack.

## Elastic online frontier and dictionary product

The most diagnostic ERP comparisons were:

| Operation | Point estimate | Comparison |
|---|---:|---:|
| 256-sample scalar two-row ERP | 209.13 µs | baseline |
| 256-sample online frontier, cutoff 8 | 13.710 µs | 15.25× faster |
| 256-sample online frontier, infinite cutoff | 895.45 µs | 4.28× slower |
| 1,000×64 convenience trie range | 2.6359 ms | baseline |
| 1,000×64 bounded automaton trie range | 1.5279 ms | 1.73× faster |

The narrow frontier validates the intended payoff from cutoff pruning. The
infinite-cutoff case shows that sparse canonicalization has overhead when no
positions die; callers and future adaptive dispatch should retain the scalar
two-row scorer for that regime. The bounded dictionary product reduced latency
by about 42% while adding explicit resource and continuation semantics.

Other point estimates provide scale checks rather than cross-implementation
comparisons: Fréchet range over 1,000×64 was 3.9349 ms, band-8 DTW range was
2.8202 ms, unit-grid TWED range was 3.1488 ms, and the TWED length lower bound
was 1.6444 ns.

## Decision and remeasurement rule

The accepted architecture remains a lazy product with compact state identity,
proved pruning, explicit iterative schedulers, and bounded retention. The
measurements justify two concrete decisions:

1. zipper-rooted and node-rooted Levenshtein queries share one production
   scheduler; backend differences are confined to captured focus operations;
2. elastic online scoring keeps the scalar fallback when a cutoff cannot make
   the sparse frontier materially narrower.

Rerun this check after changing canonicalization, transition-cache policy,
dictionary cursor representation, scheduler order, or exact-final admission.
Result-set equality and the full correspondence/property gates are mandatory
before comparing timing distributions.
