# Bounded query-cache policy matrix

## Purpose

This experiment compares the production `VersionedQueryCache` policy—TinyLFU
admission followed by SIEVE victim selection—with bounded FIFO, LRU,
SIEVE-only, and aging exact-LFU controls. It measures policy overhead and cache
quality separately so a high hit rate cannot conceal an expensive lookup path.
TinyLFU is an approximate-frequency admission policy designed to resist
one-hit scans [1]. SIEVE is a low-mutation eviction policy that uses one visited
bit per resident [2].

The topology-gated release experiment completed on 2026-08-19. The protocol,
raw results, validity checks, and pgmcp decision are reported below.

## Fairness model

All five policies receive the same immutable universe of 4,096 query strings,
the same exact one-element result allocation on each miss, the same logical
weight for each key, and the same deterministic request traces. Every policy
has a hard 128-entry limit and a hard 384-unit logical-weight limit.

The four control policies use a preallocated dense key-to-slot directory. A
hash map is not intrinsic to FIFO, LRU, SIEVE, or LFU, so making every control
perform general-purpose hash-table allocation and probing would measure an
accidental implementation choice. The controls nevertheless own an `Arc<str>`
key and `Arc<[usize]>` result for every admitted resident, matching the
production cache's key/result ownership shape. LRU uses an intrusive,
array-backed doubly linked list; only aging exact LFU performs a resident scan,
because selecting the exact minimum frequency is part of that control.

Policy construction, trace construction, initial cache warming, CSV rendering,
and process startup are outside measured intervals. Timed allocation counters
observe calls and requested bytes made through Rust's process-global system
allocator during the request trace. They are allocator-request counts, not
retained-set-size or resident-set-size estimates.

## Workloads and metrics

| Workload | Request pattern | Quality signal |
|---|---|---|
| `hot-hits` | Repeated uniform access to one warmed 128-key set | Hit latency and allocations per operation |
| `scan` | One pass over 1,280 previously unseen keys after warming | Number of original hot keys retained |
| `phase-shift` | 64 rounds over a disjoint 128-key set | First round reaching 95% new-set residency and cumulative hit rate |
| `zipf` | Fixed-seed Zipf trace over 4,096 keys; first 10% is warmup | Measured-tail hit rate and locality cost |

Every evidence row records elapsed nanoseconds, nanoseconds per operation,
allocation/deallocation calls and bytes, hit/miss counts, hit rate, resident
entries, resident logical weight, hard limits, workload-specific quality, exact
result checksum, within-process policy/workload order, and executable SHA-256.
The adjacent `*-host-load.jsonl` ledger records the selected CPU and its
complete last-level-cache topology before and after every committed replicate.
Its source of truth is the immutable per-replicate file in
`*-admissions/`; rejected gates are recorded separately in
`*-host-load-rejections.jsonl` and never enter the accepted ledger. A
continuous one-second process monitor independently rejects a run if another
benchmark or profiling harness overlaps the timed process.

The runner rotates policy order by replicate and workload order by both
replicate and policy position. This balances thermal and frequency drift
without randomizing the workload itself.

## Executable protocol

Build the release example but do not run it while another release build or
benchmark owns the host:

```console
cargo build --release --features benchmark-controls --example query_cache_policy_matrix
```

Then collect 51 topology-admitted replicates on one physical CPU:

```console
benchmarks/causal/run-query-cache-policy-matrix-experiment.sh \
  target/release/examples/query_cache_policy_matrix \
  benchmarks/causal/evidence/YYYY-MM-DD/query-cache-policy-performance.csv \
  51 3 100000 200000
```

The output and its sibling transactional evidence paths must not already
exist. The runner performs three unrecorded warmup processes, rejects a
replicate when the topology gate fails, validates 20 policy/workload rows per
replicate, checks both residency bounds, and requires identical result
checksums across policies. Only then does it atomically commit the replicate's
admission sidecar and updated CSV and rebuild the aggregate accepted ledger.
It does not delete diagnostic or rejection evidence after a failure.

An interrupted run can be continued with `--resume`. Resume is deliberately
strict: the CSV header and executable digest must match, every retained
replicate must contain exactly 20 rows in monotone order, and no nonempty
foreign-process ledger may exist. It reconstructs `*-host-load.jsonl` solely
from the committed admission sidecars; a diagnostic-only contention-monitor
log does not block recovery. A host-admission rejection does not create or
replace a replicate. Previously accepted replicates are never rerun. When
migrating pre-transactional evidence, the original mixed ledger is preserved
as `*-host-load-pre-transactional.jsonl`, rejected rows are split out, and
only admissions proving complete CSV replicates become committed sidecars.

For a fast structural check, reduce the operation counts and sample count; such
output is diagnostic and must not be presented as performance evidence:

```console
benchmarks/causal/run-query-cache-policy-matrix-experiment.sh \
  target/debug/examples/query_cache_policy_matrix /tmp/query-cache-smoke.csv \
  1 3 1000 4096
```

The row contract is machine-readable in
[`query-cache-policy-matrix.schema.json`](../../benchmarks/causal/schemas/query-cache-policy-matrix.schema.json).

## Results

The matrix contains 51 replicates per policy and workload. Values below are
medians; `MAD` is median absolute deviation. Allocation counts are calls per
request. `phase-95` is the first round at which at least 95% of the disjoint
phase's keys are resident, and `scan-retained` counts the original 128 hot keys
remaining after a 1,280-key scan.

| workload | policy | ns/op | MAD | allocations/op | hit rate | phase-95 | scan-retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| hot hits | FIFO | 4.529 | 0.051 | 0.000 | 1.000000 | — | — |
| hot hits | LRU | 7.294 | 0.082 | 0.000 | 1.000000 | — | — |
| hot hits | SIEVE | 5.145 | 0.048 | 0.000 | 1.000000 | — | — |
| hot hits | exact LFU | 6.257 | 0.033 | 0.000 | 1.000000 | — | — |
| hot hits | TinyLFU+SIEVE | 34.412 | 0.189 | 0.000 | 1.000000 | — | — |
| phase shift | FIFO | 5.305 | 0.035 | 0.047 | 0.984375 | 1 | — |
| phase shift | LRU | 7.736 | 0.092 | 0.047 | 0.984375 | 1 | — |
| phase shift | SIEVE | 5.791 | 0.061 | 0.047 | 0.984375 | 1 | — |
| phase shift | exact LFU | 236.310 | 0.319 | 1.171 | 0.609619 | 25 | — |
| phase shift | TinyLFU+SIEVE | 52.536 | 0.344 | 0.341 | 0.837402 | 19 | — |
| scan | FIFO | 48.444 | 0.485 | 3.000 | 0.000000 | — | 0 |
| scan | LRU | 48.474 | 0.681 | 3.000 | 0.000000 | — | 0 |
| scan | SIEVE | 48.099 | 0.485 | 3.000 | 0.000000 | — | 0 |
| scan | exact LFU | 596.773 | 0.884 | 3.000 | 0.000000 | — | 127 |
| scan | TinyLFU+SIEVE | 187.473 | 0.813 | 2.000 | 0.000000 | — | 128 |
| Zipf | FIFO | 29.288 | 0.148 | 1.351 | 0.549617 | — | — |
| Zipf | LRU | 29.159 | 0.147 | 1.191 | 0.603128 | — | — |
| Zipf | SIEVE | 27.275 | 0.124 | 1.153 | 0.615622 | — | — |
| Zipf | exact LFU | 199.165 | 0.147 | 1.016 | 0.661372 | — | — |
| Zipf | TinyLFU+SIEVE | 61.409 | 0.329 | 0.729 | 0.664172 | — | — |

TinyLFU+SIEVE strictly dominated aging exact LFU on the preregistered Zipf
comparison: latency fell from 199.165 to 61.409 ns/op while hit rate rose from
0.661372 to 0.664172. Pgmcp experiment 299 accepted the hypothesis with 51
samples per arm (one-sided Welch `p = 2.19e-112`, Cohen's `d = -92.76`, 95%
mean-difference confidence interval `[-138.754, -137.575]` ns/op). Both arms
departed from normality; the robustness check nevertheless had complete
separation (Cliff's delta `-1`, Mann–Whitney `p` rounded to zero).

The simple policies establish the lower latency bound. TinyLFU+SIEVE costs
34.134 ns/op more than SIEVE on Zipf but raises hit rate by 0.048550. Therefore
TinyLFU+SIEVE has lower end-to-end expected cost whenever deriving a missed
value costs more than approximately 703 ns:

```math
C_{\text{miss}} >
\frac{61.409 - 27.275}{0.664172 - 0.615622}
= 703.1\ \text{ns}.
```

That threshold is well below a dictionary/transducer query miss, while the
scan result shows why admission matters: SIEVE alone evicted all 128 hot
entries, whereas TinyLFU+SIEVE retained all of them. FIFO, LRU, or SIEVE remains
the better choice for a cache whose misses are sub-microsecond and whose access
stream lacks scans. Exact LFU is not competitive: its resident-wide minimum
search is expensive and its slowly decaying history adapts poorly to a phase
change.

## Validity and evidence

All 204 replicate/workload groups produced identical result checksums across
the five policies. No row exceeded either the 128-entry or 384-unit bound. The
historical host ledger contains 104 admitted pre/post records and one rejected
pre-run record: replicate 16 observed 12% on the selected CPU against a 10%
limit, so it stopped before timing. Resume retained replicates 1–15 and
completed 16–51 without rerunning a valid replicate. This artifact predates
the transactional sidecar/rejection-ledger split described above and remains
unchanged for provenance. The continuous monitor created no foreign contention
ledger.

The evidence artifacts are:

- [`query-cache-policy-performance.csv`](../../benchmarks/causal/evidence/2026-08-19/query-cache-policy-performance.csv), SHA-256 `b8cd28fbf0037ac8d95ea48452a8c2b1aacc8e6bab06e0ddd91609cda2fc65ff`;
- [`query-cache-policy-performance-host-load.jsonl`](../../benchmarks/causal/evidence/2026-08-19/query-cache-policy-performance-host-load.jsonl), SHA-256 `349d7b3f85f8d031252810913ea654bff85cfb0918db3058703ec576338aa12c`;
- [`query-cache-policy-primary-zipf.csv`](../../benchmarks/causal/evidence/2026-08-19/query-cache-policy-primary-zipf.csv), SHA-256 `c38e38ea0b420875c52f4d565d23fd4b17bf1c629f213ce482b420dcd4db7be7`.

## Interpretation and stop rules

The exact-LFU comparison has a registered and accepted pgmcp decision. The
four-policy matrix remains multi-objective: a policy should not be selected
from latency alone, but from scan retention, phase adaptation, Zipf hit rate,
allocator traffic, miss cost, and bounded residency. Result checksums and both
hard bounds are mandatory validity gates rather than optimization metrics.

The allocator counts are process-global. The harness is intentionally
single-threaded; adding worker threads would make unrelated allocations enter
the counters and invalidate comparisons. Dense-directory controls also mean
absolute control latency is a best-case policy floor, not a claim that an
application with arbitrary keys can avoid key lookup. The production cache's
collision-exact keyed hashing remains part of its measured cost.

## References

1. G. Einziger, R. Friedman, and B. Manes. “TinyLFU: A Highly Efficient Cache
   Admission Policy.” *ACM Transactions on Storage* 13(4), 2017.
   [doi:10.1145/3149371](https://doi.org/10.1145/3149371).
2. Y. Zhang et al. “SIEVE Is Simpler than LRU: An Efficient Turn-Key Eviction
   Algorithm for Web Caches.” *NSDI*, 2024.
   [USENIX](https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo).
