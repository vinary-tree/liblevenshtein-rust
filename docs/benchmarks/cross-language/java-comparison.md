# Java vs Java: liblevenshtein-java 3.0.0 against the Rust-backed JVM binding

**Status:** measured, complete (165 timed cells). **Run:** `phase0`.
**Verdict on the migration thesis this program set out to test:
NOT SUPPORTED on query throughput.**

This document reports the first *measured* comparison between the two Java
options for `liblevenshtein`. Every previous statement on the subject — the
archived [`JAVA_COMPARISON.md`](../../archive/performance/JAVA_COMPARISON.md),
which speculated that "Rust is likely significantly faster than Java" — was
feature analysis, not measurement. The measurement disagrees with the
speculation, and this document reports what was measured.

## 1. The two implementations

| | Legacy | Rust-backed |
|---|---|---|
| Artifact | `com.github.universal-automata:liblevenshtein:3.0.0` (Maven Central) | `io.vinarytree:liblevenshtein:0.10.0` |
| Implementation | Pure Java, Java 8 bytecode (2016) | Rust core reached through a Java 22 FFM binding (no JNI) |
| Dictionary | `SortedDawg` on the Java heap | libdictenstein `DynamicDawg` / `DoubleArrayTrie` behind the `llev_*` C ABI |
| Maintenance | Last upstream commit 2016-05-29 | Actively developed |

*DAWG* = directed acyclic word graph, the minimal acyclic automaton of a word
set, built incrementally from sorted input after Daciuk et al. [4]. *FFM* =
the JDK's Foreign Function & Memory API. *Query* here means: given a query
term and a maximum edit distance `d`, enumerate every dictionary term within
`d` edits, as a lazy cursor that the caller drains fully.

## 2. Headline result

Both implementations were run under JMH on **the same JDK 26, the same
cpuset, and identical fixed 2 GiB heaps**, over the same committed
79,343-word aspell en_US dictionary, 1,000 queries per timed pass, 2 forks ×
10 measurement iterations per cell.

Across all **45 shared cells** (3 algorithms × 3 distances × 5 query sets),
the ratio of legacy pass time to Rust-backed pass time is:

```math
\frac{t_{\text{legacy}}}{t_{\text{vinary}}}: \quad
\text{geomean } 0.342, \quad \text{median } 0.345, \quad
\text{range } [0.265,\ 0.424]
```

A ratio below 1 means the legacy library is *faster*. **Legacy Java is about
2.9× faster than the Rust-backed binding**, and it wins in every single one
of the 45 cells. The effect is uniform, not workload-specific:

| algorithm | geomean ratio | | distance | geomean ratio |
|---|---|---|---|---|
| standard | 0.325 | | d = 1 | 0.330 |
| transposition | 0.346 | | d = 2 | 0.331 |
| merge_and_split | 0.357 | | d = 3 | 0.368 |

### The preregistered hypothesis, and its refutation

Hypothesis **H-J1** (pgmcp experiment 178, criterion locked *before* any
measurement) predicted the Rust-backed binding would reach **at least 3×**
the legacy library's throughput at the deciding coordinate. Measured there:

| arm | median pass time | 95% CI of the median |
|---|---|---|
| legacy (`SortedDawg`) | **462.34 ms** | [458.21, 464.18] |
| Rust-backed (`DynamicDawg`) | **1288.32 ms** | [1281.21, 1294.23] |

Ratio 0.359 where ≥ 3.0 was required — missed by roughly an order of
magnitude, in the opposite direction, with non-overlapping confidence
intervals. **H-J1 is refuted.** It is recorded as refuted; it is not
reinterpreted after the fact.

### The deficit is not merely the binding boundary

The pure-Rust core, measured with no bindings and no JVM at all, is *also*
slower than the 2016 Java library on the same cell (standard, d = 1, hits):

| | µs per query |
|---|---|
| legacy Java 3.0.0 (pure Java) | **51.2** |
| pure Rust core (no FFI, no JVM) | 71.7 |
| Rust core via JVM FFM binding (`DynamicDawg`) | 158.4 |
| Rust core via JVM FFM binding (`DoubleArrayTrie`) | 202.7 |

So there are two independent gaps: the Rust core trails legacy Java by
≈ 1.4×, and the JVM binding adds a further ≈ 2.2× on top. Note also that
`DoubleArrayTrie`, nominally the read-optimized backend, is *slower* here
than `DynamicDawg`.

## 3. Where the migration case does hold

Query throughput is one axis of three. On the other two, and on correctness,
the picture differs sharply.

### Correctness: identical, proven

Before any timing was accepted, every target had to reproduce a Rust oracle's
result multiset exactly, compared as `(match count, Σ term bytes, Σ distance,
order-insensitive FNV-1a-64 checksum)`. Legacy Java matched on **all 45
shared cells**, bit for bit. Whatever else the two libraries differ in, they
do not differ in what they return.

One known correctness delta is *not* exercised by this workload: legacy
3.0.0 mishandles the empty string (a root-finality bug, fixed only in an
unpushed local commit). The generated workload contains no empty string, so
the gate could not have caught it, and it is documented here rather than
discovered by a user.

### Memory: a decisive Rust-backed win

Peak resident set size of a whole process that builds the dictionary and runs
one full pass (`/usr/bin/time -v`, identical 2 GiB heaps):

| | peak RSS |
|---|---|
| legacy Java `SortedDawg` | **567.4 MiB** |
| Rust-backed `DynamicDawg` | **192.0 MiB** |
| Rust-backed `DoubleArrayTrie` | 194.5 MiB |

The Rust-backed binding uses **≈ 2.95× less memory**, because the dictionary
lives in native memory rather than as Java heap objects. For a service
holding a large dictionary resident, this can matter more than per-query
latency.

### Construction: legacy is faster; one backend is disqualifying

Median dictionary build from the same pre-sorted 79,343-word list (sorting
excluded from timing on both sides; legacy takes its documented
`isSorted=true` fast path):

| | median build |
|---|---|
| legacy `SortedDawg` | **41.2 ms** |
| Rust-backed `DynamicDawg` (batch insert) | 98.9 ms |
| Rust-backed `DoubleArrayTrie` | **12,765 ms** |

Hypothesis **H-J2** allowed the Rust-backed build to cost up to 1.5× the
legacy build; at 2.4× it is **refuted** as well. The `DoubleArrayTrie` result
— **310× slower** to build than legacy, for *worse* query performance — makes
that backend unsuitable for this dictionary size on either axis.

## 4. Why: profile of the Rust core

A CPU profile of the pure Rust core (standard, d = 2, `std-d2`; `perf`,
3,102 samples) locates the cost in the automaton simulation, **not** in
dictionary synchronization:

| share | symbol |
|---|---|
| 25.2% | `transition_state_pooled_ref` |
| 24.5% | `QueryIterator::queue_children` |
| 15.7% | `characteristic_vector` |
| 13.7% | `QueryIterator::advance` |
| 7.1% | `State::copy_from` |
| 2.9% | `LockFreeDawgNode::drop` |
| 3.0% | `malloc` + `free` |
| 1.0% | `SmallVec<[(u8, Arc<LockFreeDawgNode>); 4]>::clone` |

Roughly half the time is state-transition machinery. A prior hypothesis —
that the lock-free dictionary's per-node atomic loads dominated — is
**refuted** by this profile; no atomic-heavy symbol appears near the top.

There is, however, one concrete and apparently fixable inefficiency:
`queue_children` **clones a `SmallVec` of `(byte, Arc<node>)` edges for each
visited node**, paying an atomic reference-count increment per child edge,
with `LockFreeDawgNode::drop` (2.9%) as the matching decrement. Iterating
those edges by reference rather than cloning should recover most of that
≈ 7% cluster. That alone does not close a 1.4× gap, and no optimization has
been attempted here — this document reports measurements, and the lead is
recorded for follow-up work.

## 5. Recommendation

**On the evidence, a user of liblevenshtein-java 3.0.0 whose workload
resembles this one should not migrate for query speed.** They would pay
roughly 2.9× in per-query latency. Migration is nonetheless justified when:

- **memory dominates** — the Rust-backed binding uses ≈ 3× less RSS;
- **maintenance matters** — legacy 3.0.0 has had no upstream commit since
  2016, ships a `protobuf 3.0.0-beta-3` transitive dependency, and carries
  the unfixed empty-string bug;
- **features are needed** that 3.0.0 lacks (for example the
  `damerau_levenshtein` algorithm, measured here for the Rust-backed side
  only because legacy does not implement it).

Anyone migrating for throughput should wait until the profile findings in
§4 have been acted on and re-measured. The honest summary is that the Rust
implementation's *engineering* is ahead and its *query performance on this
workload* is behind.

## 6. Threats to validity

- **One dictionary, one shape.** 79,343 lowercase-ASCII words. Results may
  differ for much larger dictionaries, Unicode-heavy sets, or long terms.
  `DoubleArrayTrie`'s construction cost in particular is size-sensitive.
- **Materializing drain.** Both sides fully materialize `(term, distance)`
  per match, the migration-realistic path. The Rust-backed binding also
  offers a zero-copy borrowed-batch API that was deliberately not used, since
  legacy has no equivalent; it may narrow the gap for callers who adopt it.
- **cpuset spans two L3 domains.** `taskset -c 2-9` covers CCD0 cores 2–7
  and CCD1 cores 8–9 on this host. Both arms share it identically, so no
  comparison is biased, but VM-target dispersion is widened — which is why
  medians with MAD and bootstrap intervals are reported rather than means.
- **Sample count.** 20 samples per cell (2 forks × 10 iterations). The
  preregistered protocol prescribes ≥ 51 replicates for the hypothesis
  arms; those deeper runs are collected separately and are what formally
  decide H-J1/H-J2. The 45-cell agreement reported here is corroborating
  breadth, not the formal test.

## 7. Reproduction

```bash
cd benchmarks/cross-language
scripts/doctor.sh --targets jvm-vinary,jvm-legacy      # readiness
scripts/gate.py results/<run> --targets rust,jvm-vinary,jvm-legacy
scripts/run-jvm-pair.sh stage       && \
scripts/run-jvm-pair.sh verify-full results/<run> && \
scripts/run-jvm-pair.sh jmh         results/<run>
scripts/aggregate.py results/<run>
```

Raw per-sample data for every cell is in the pgmcp data table
`xlang_bench_cells`; hypotheses, criteria and verdicts are pgmcp experiments
178 (H-J1) and 179 (H-J2). Method: [`PROTOCOL.md`](../../../benchmarks/cross-language/harnesses/common/PROTOCOL.md).

## References

1. V. I. Levenshtein. *Binary codes capable of correcting deletions,
   insertions, and reversals.* Soviet Physics Doklady 10(8):707–710, 1966.
2. R. A. Wagner and M. J. Fischer. *The string-to-string correction problem.*
   Journal of the ACM 21(1):168–173, 1974.
   [doi:10.1145/321796.321811](https://doi.org/10.1145/321796.321811)
3. F. J. Damerau. *A technique for computer detection and correction of
   spelling errors.* CACM 7(3):171–176, 1964.
   [doi:10.1145/363958.363994](https://doi.org/10.1145/363958.363994)
4. J. Daciuk, S. Mihov, B. W. Watson, R. E. Watson. *Incremental construction
   of minimal acyclic finite-state automata.* Computational Linguistics
   26(1):3–16, 2000.
   [doi:10.1162/089120100561601](https://doi.org/10.1162/089120100561601)
5. J. Aoe. *An efficient digital search algorithm by using a double-array
   structure.* IEEE TSE 15(9):1066–1077, 1989.
   [doi:10.1109/32.31365](https://doi.org/10.1109/32.31365)
6. K. U. Schulz and S. Mihov. *Fast string correction with Levenshtein
   automata.* IJDAR 5(1):67–85, 2002.
   [doi:10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)
7. B. Efron. *Bootstrap methods: another look at the jackknife.* Annals of
   Statistics 7(1):1–26, 1979.
   [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552)
