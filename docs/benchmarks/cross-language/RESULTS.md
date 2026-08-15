# Cross-Language Benchmark Results — run `phase0`

**Status: CAMPAIGN PAUSED, partially measured.** This document reports every
measurement taken so far and states precisely what has *not* been measured. It
is a faithful snapshot, not a finished atlas; §6 enumerates the gaps.

**Measured:** 424 cells, 0 schema-invalid, 0 measured under foreign contention.
**Raw data:** pgmcp data table `xlang_bench_cells`, `run_id = phase0`.
**Method:** [`methodology.md`](methodology.md); harness contract:
[`PROTOCOL.md`](../../../benchmarks/cross-language/harnesses/common/PROTOCOL.md).

Workload for every number below: the committed 79,343-word aspell en_US
dictionary, 1,000 queries per timed pass, 30 samples per cell (20 for JMH cells:
2 forks × 10 iterations). All arms passed the correctness gate — **1,455
comparisons, 0 mismatches** — before any timing was accepted.

---

## 1. What was measured

| target | backend(s) | role |
|---|---|---|
| `rust` | DynamicDawg, DoubleArrayTrie | pure core; oracle and atlas anchor |
| `jvm-vinary` | DynamicDawg, DoubleArrayTrie | Java pair, Rust-backed arm |
| `jvm-legacy` | legacy SortedDawg | Java pair, legacy arm |
| `cpp` | DynamicDawg | C++ pair, Rust-backed arm |
| `cpp-legacy` | legacy SortedDawg | C++ pair, legacy arm |
| `js-native` | DynamicDawg, DoubleArrayTrie | atlas (JS pair incomplete — see §6) |

## 2. Head-to-head pairs

Ratio convention: $`t_{\text{legacy}} / t_{\text{vinary}}`$, so **above 1 means
the Rust-backed side is faster**. Aggregated geometrically (§1.3 of the
methodology).

| pair | arm | geomean | median | range | cells won |
|---|---|---|---|---|---|
| Java | vinary `DynamicDawg` | **0.342** | 0.345 | [0.265, 0.424] | 0 / 45 |
| Java | vinary `DoubleArrayTrie` | **0.291** | 0.294 | [0.210, 0.404] | 0 / 45 |
| C++ | vinary `DynamicDawg` | **1.158** | 1.133 | [0.927, 1.702] | 38 / 45 |

The two pairs point in opposite directions, and the reason is not that the Rust
core behaves differently in the two settings — it is the same core in both. It
is that the two legacy libraries differ enormously from each other. Measured on
the same coordinates, **legacy Java is roughly 2.8–3.1× faster than legacy
C++**. Full treatment in [`java-comparison.md`](java-comparison.md) and
[`cpp-comparison.md`](cpp-comparison.md).

The JavaScript pair has **no result**: its legacy arm was never timed (§6).

## 3. Binding overhead atlas

Query cost of each binding relative to the pure Rust core on identical
coordinates, `DynamicDawg` both sides:

| binding | geomean overhead | range | cells |
|---|---|---|---|
| `cpp` (C++ facade over the C ABI) | **1.576×** | [1.291, 2.049] | 60 |
| `js-native` (N-API addon) | **1.660×** | [1.350, 2.030] | 60 |
| `jvm-vinary` (Java 22 FFM) | **1.670×** | [1.401, 2.208] | 60 |

The striking feature is how *little* these differ. Three bindings in three
unrelated host runtimes — a thin C++ RAII wrapper, a Node N-API addon, and a
JVM FFM downcall — land within 6% of each other. That is the signature of a
cost paid *below* all of them, in the shared `llev_*` C ABI, rather than in any
host-language facade.

The C++ facade is the cleanest available estimate of that floor, because it adds
the least of its own: **the ABI floor is $`\approx`$ 1.576×**, and the JVM's much-discussed
FFM layer adds only about 6% on top of it. Optimization aimed at host-language
facades would therefore be aimed at the wrong 6%.

> **The `c` target is absent from this table**, and its absence is deliberate.
> Its harness accumulated the per-pass triple through caller-supplied pointers,
> forcing a memory store per match and inflating its measured overhead by
> roughly 11% — an artifact of the instrument, not the boundary. The defect is
> fixed and the fix is gate-verified, but all 124 pre-fix `c` cells are
> quarantined pending re-measurement. The previously circulated **1.763×**
> C-ABI figure comes from those cells and should not be used; 1.576× from the
> C++ facade supersedes it.

## 4. Construction

Ten timed builds (three for `DoubleArrayTrie`, whose builds are far slower),
sorting excluded from the timed region for every arm:

| target | backend | median | MAD |
|---|---|---|---|
| `jvm-legacy` | legacy SortedDawg | **41.24 ms** | 2.911 ms |
| `rust` | DynamicDawg | 70.47 ms | 0.239 ms |
| `cpp` | DynamicDawg | 89.10 ms | 0.146 ms |
| `jvm-vinary` | DynamicDawg | 98.89 ms | 1.292 ms |
| `js-native` | DynamicDawg | 118.55 ms | 0.369 ms |
| `cpp-legacy` | legacy SortedDawg | 135.70 ms | 1.760 ms |
| `rust` | DoubleArrayTrie | **11,247.91 ms** | 3.673 ms |
| `jvm-vinary` | DoubleArrayTrie | **12,765.53 ms** | 3.268 ms |

Two results stand out. **Legacy Java builds its DAWG faster than the Rust core
does** — the one axis on which the 2016 library beats the current
implementation outright. And **`DoubleArrayTrie` construction costs roughly 160×
`DynamicDawg`**, about 11–13 seconds either way, which is a disqualifying figure
for any workload that builds dictionaries at run time.

## 5. Memory (peak RSS)

Measured by `/usr/bin/time -v` around a construct-plus-one-pass child process:

| target | backend | peak RSS |
|---|---|---|
| `rust` | DoubleArrayTrie | **17.39 MiB** |
| `rust` | DynamicDawg | 31.38 MiB |
| `cpp` | DynamicDawg | 39.20 MiB |
| `cpp-legacy` | legacy SortedDawg | 49.56 MiB |
| `js-native` | DynamicDawg | 102.24 MiB |
| `jvm-vinary` | DynamicDawg | 192.01 MiB |
| `jvm-vinary` | DoubleArrayTrie | 194.47 MiB |
| `jvm-legacy` | legacy SortedDawg | 567.37 MiB |

This **refines the `DoubleArrayTrie` verdict rather than confirming it wholesale.**
DAT is worse on query speed and catastrophically worse on construction, but it
is the *smallest* structure measured — 1.80× smaller than `DynamicDawg` in the
pure-Rust arm. It is a space-optimized structure, not a read-optimized one on
this workload. Any recommendation should say so.

The Java memory gap is the Java pair's clearest win: the Rust-backed binding
uses **2.95× less** RSS than legacy Java, which holds its DAWG on the JVM heap.

## 6. Not measured

Required by the methodology: an explicit accounting of every gap. **104
coordinates are unmeasured**, in three groups.

**Never timed — the campaign was paused mid-sweep.** 14 of 21 target × backend
legs:

| leg | consequence |
|---|---|
| `js-legacy` | **the JavaScript pair has no result at all** |
| `js-wasm`, `js-wasi` | no WASM/WASI overhead figures |
| `python`, `go`, `ruby`, `lua`, `dotnet`, `swift`, `fortran`, `ocaml`, `haskell`, `clojure`, `clojurescript` | atlas covers 3 bindings instead of 14 |

**Quarantined pending re-measurement** (retained on disk, each with a manifest):

| set | count | reason |
|---|---|---|
| `c` (all cells) | 124 | harness accumulator aliasing (§3) |
| `rust` × DoubleArrayTrie | 42 | orphaned by an interrupted batch |
| interrupted batch cells | 56 | incomplete records: no environment provenance |

**Excluded by design.** `damerau_levenshtein` is absent from every pair table:
no legacy implementation provides it, so it has no counterpart to compare
against. It *is* measured on the Rust-backed side and appears in the atlas.

## 7. Provisional figures

Two classes of number here are safe, and one is not:

- **Pair ratios (§2) are drift-safe.** Both arms of each pair were measured
  time-adjacent, so ambient load affects them equally and cancels in the ratio.
- **Construction and memory (§4, §5) are single measurements**, not ratios, and
  are correspondingly robust.
- **Atlas overheads (§3) are provisional.** They divide by a `rust` anchor
  measured hours earlier in the same sweep, during which drift sentinels moved
  by up to **10.7%** as external load fell. The 6% spread among the three
  bindings is *smaller than that drift bound*, so their ordering relative to one
  another should not be treated as established. The $`\approx`$ 1.576× floor is far larger
  than the drift and is safe. Recomputing against an end-of-run anchor is
  required before the ordering is quoted.

## 8. Reproduction

```bash
cd benchmarks/cross-language
scripts/doctor.sh --targets rust,cpp,cpp-legacy,js-native
python3 scripts/gate.py results/phase0          # must be green first
scripts/run-all.sh --results "$PWD/results/phase0"   # resume is implicit
python3 scripts/aggregate.py results/phase0
python3 scripts/pgmcp-upload.py results/phase0  # publish raw cells
```

Generated tables live in `results/phase0/tables/`: `pair_java.md`,
`pair_cpp.md`, `pair_javascript.md`, `atlas_overhead.md`, `construct.md`,
`memory.md`, `contention.md`, `not_measured.md`, `query_summary.md`.

## References

1. Daciuk, J., Mihov, S., Watson, B. W., & Watson, R. E. (2000). Incremental
   Construction of Minimal Acyclic Finite-State Automata. *Computational
   Linguistics*, 26(1), 3–16. <https://doi.org/10.1162/089120100561601>
2. Schulz, K. U., & Mihov, S. (2002). Fast string correction with Levenshtein
   automata. *IJDAR*, 5(1), 67–85. <https://doi.org/10.1007/s10032-002-0082-8>
3. Aoe, J. (1989). An efficient digital search algorithm by using a
   double-array structure. *IEEE Transactions on Software Engineering*, 15(9),
   1066–1077. <https://doi.org/10.1109/32.31365>
4. Fleming, P. J., & Wallace, J. J. (1986). How not to lie with statistics: the
   correct way to summarize benchmark results. *CACM*, 29(3), 218–221.
   <https://doi.org/10.1145/5666.5673>
