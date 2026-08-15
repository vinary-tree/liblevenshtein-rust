# Java vs Java: liblevenshtein-java 3.0.0 against the Rust-backed JVM binding

**Status:** phase-0 baseline and post-optimization closure measured. **Runs:**
`phase0` and `liblev-h-o7-final`. **Current verdict:** practical query parity
is achieved across the shared matrix; the preregistered 3x-throughput target
is not achieved.

This document reports the first *measured* comparison between the two Java
options for `liblevenshtein`. Every previous statement on the subject — the
archived [`JAVA_COMPARISON.md`](../../archive/performance/JAVA_COMPARISON.md),
which speculated that "Rust is likely significantly faster than Java" — was
feature analysis, not measurement. The measurement disagrees with the
speculation, and this document reports what was measured.

## 0. Post-optimization closure (2026-08-15)

The rest of this report preserves the original phase-0 baseline and causal
motivation. After the controlled H-O9 through H-O30 campaign, the unchanged
45-coordinate Java-to-Java matrix was rerun with the same dictionary, query
sets, materializing drain contract, JDK, 2 GiB heaps, cpuset, and 2-fork x
10-iteration JMH protocol. All 45 pairs reproduced the same result counts and
checksums exactly.

| post-optimization breadth result | value |
|---|---:|
| geometric mean, legacy time / Vinary time | 0.965 |
| median ratio | 0.974 |
| range | [0.766, 1.109] |
| Vinary wins | 16 / 45 |
| legacy wins | 29 / 45 |

A ratio above 1 means Vinary is faster. On geometric mean Vinary is therefore
only 3.6% slower overall, rather than phase 0's 2.9x slowdown. The result is
mixed by algorithm in a diagnostically useful way:

| algorithm | Vinary wins | legacy / Vinary geometric mean | interpretation |
|---|---:|---:|---|
| standard | 5 / 15 | 0.983 | Vinary 1.8% slower |
| transposition | 10 / 15 | 1.019 | Vinary 1.9% faster |
| merge-and-split | 1 / 15 | 0.899 | Vinary 11.2% slower |

The original anchor is now 54.543 ms for Vinary versus 51.256 ms for legacy
(`standard`, d = 1, `hits`), a 6.4% residual rather than a 3.09x deficit. At
the formal H-J1 coordinate (`transposition`, d = 2, `tr-d2`), the breadth run
measured 445.499 ms for Vinary versus 448.694 ms for legacy, making Vinary
0.7% faster. The formal 51-sample pair, pooled from three independently
compiler-guarded JVM forks per arm, measured:

| formal H-J1 arm | median pass | 95% bootstrap CI of median |
|---|---:|---:|
| legacy control | 455.861 ms | [455.223, 456.856] ms |
| Vinary treatment | 466.581 ms | [463.029, 468.978] ms |

At that depth Vinary was 2.35% slower. The preregistered one-sided Welch test
did not support a Vinary improvement (`p = 0.664`, Cohen's `d = +0.084`), and
the mandatory practical requirement—Vinary at least 3x as fast—was decisively
missed. Experiment 205 is therefore rejected. Practical matrix parity does not
retroactively satisfy that stronger target.

Construction moved beyond parity as well. The current direct Rust medians are
21.416 ms for arbitrary-order byte terms and 11.076 ms for the explicit
pre-ordered constructor, versus the historical 41.2 ms Java ordered-build
reference. The binding-owned freeze build measured 26.547 ms after H-O28,
down from 96.271 ms. These values show that the former construction deficit
was architectural path-copy/publication work, not an inherent Rust or
reclamation disadvantage.

The final compiler-guarded pure-Rust core gate (`standard`, d = 1, `hits`)
measured 42.057 ms over 51 samples, with a 95% bootstrap median interval of
[41.959, 42.396] ms. It passes the 51.2 ms core threshold by 17.9%, confirming
that the remaining JVM-level differences are boundary/runtime effects rather
than a slower native standard-Levenshtein kernel.

The current recommendation is consequently different from the phase-0 one:
query throughput is no longer a reason to reject migration for workloads
represented by this matrix. Transposition is slightly ahead, standard is at
near parity, and merge-and-split remains the clearest optimization target.
The Rust-backed implementation retains its approximately 3x peak-RSS advantage
and active-maintenance benefits. Users requiring a literal 3x query-throughput
gain over legacy Java should not migrate on that expectation; the measured
outcome is parity, not a threefold win.

> **Read §0 before quoting the historical baseline in §2.** The phase-0
> measurements showed legacy Java winning every query cell, but the completed
> optimization campaign reduced that gap to practical matrix parity. The
> original measurements remain below as an immutable baseline rather than a
> statement of current performance.

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

## 2. Phase-0 headline result (historical baseline)

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

Ratio 0.359 where $`\ge`$ 3.0 was required — missed by roughly an order of
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
$`\approx`$ 1.4×, and the JVM binding adds a further $`\approx`$ 2.2× on top. Note also that
`DoubleArrayTrie`, nominally the read-optimized backend, is *slower* here
than `DynamicDawg`.

### Calibrating against a third implementation: legacy Java is the outlier

Read alone, the table above invites the conclusion that the Rust core is
slow. A third, independently written implementation of the same algorithm
family — `liblevenshtein-cpp`, the C++ sibling of the same upstream project
— shows that this reading is wrong. Measuring it under the identical
protocol reframes both numbers:

| implementation | standard, d = 1, `hits` | transposition, d = 2, `tr-d2` |
|---|---|---|
| legacy Java 3.0.0 (pure Java) | **51.2 µs/query** | **462.3 µs/query** |
| pure Rust core (no FFI) | 71.7 µs/query | 790.0 µs/query |
| legacy C++ (`liblevenshtein-cpp`) | 143.7 µs/query | 1432.8 µs/query |

Against the C++ baseline, *both* the Rust core and legacy Java are faster,
and by wide margins:

| speedup over legacy C++ | standard, d = 1 | transposition, d = 2 |
|---|---|---|
| pure Rust core | 2.00× | 1.81× |
| legacy Java 3.0.0 | 2.80× | 3.10× |

The correct reading is therefore not "the Rust core is slow" but **"legacy
Java 3.0.0 is exceptionally fast"** — a 2016 JVM library that outruns a
compiled-C++ implementation of the same algorithm by roughly 3×. The Rust
core clears that same C++ baseline by about 2×; it is simply beaten by a
faster competitor, not by a representative one. Any claim that the Rust
implementation underperforms *implementations in general* is unsupported by
this evidence — it underperforms exactly one, unusually well-optimized peer.

This also sharpens what §4's profile is looking for. The question is not
"why is Rust slow?" (it is not) but "what does the JVM implementation do
that buys it a further $`\approx`$ 1.4–1.7× over an already-fast native
implementation?" — a materially different, and more tractable,
investigation.

## 3. Where the migration case does hold

Query throughput is one axis of three. On the other two, and on correctness,
the picture differs sharply.

### Correctness: identical, proven

Before any timing was accepted, every target had to reproduce a Rust oracle's
result multiset exactly, compared as a four-tuple: match count, summed term
byte length, summed distance, and an order-insensitive FNV-1a-64 checksum
over every returned `(term, distance)` pair. Legacy Java matched on **all 45
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

The Rust-backed binding uses **$`\approx`$ 2.95× less memory**, because the dictionary
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
$`\approx`$ 7% cluster. That alone does not close a 1.4× gap, and no optimization has
been attempted here — this document reports measurements, and the lead is
recorded for follow-up work.

## 5. Phase-0 recommendation (superseded by section 0)

**On the evidence, a user of liblevenshtein-java 3.0.0 whose workload
resembles this one should not migrate for query speed.** They would pay
roughly 2.9× in per-query latency. Migration is nonetheless justified when:

- **memory dominates** — the Rust-backed binding uses $`\approx`$ 3× less RSS;
- **maintenance matters** — legacy 3.0.0 has had no upstream commit since
  2016, ships a `protobuf 3.0.0-beta-3` transitive dependency, and carries
  the unfixed empty-string bug;
- **features are needed** that 3.0.0 lacks (for example the
  `damerau_levenshtein` algorithm, measured here for the Rust-backed side
  only because legacy does not implement it).

Anyone migrating for throughput should wait until the profile findings in
§4 have been acted on and re-measured. The honest summary is that the Rust
implementation's *engineering* is ahead and its *query performance on this
workload* is behind — behind **this specific competitor**, which §2 shows to
be an outlier rather than a typical baseline. Against the project's own C++
implementation the Rust core is roughly 2× faster; the migration case that
fails here is specifically "migrate away from liblevenshtein-java 3.0.0 for
speed", not "prefer another implementation to the Rust one".

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
  preregistered protocol prescribes $`\ge`$ 51 replicates for the hypothesis
  arms; those deeper runs are collected separately and are what formally
  decide H-J1/H-J2. The 45-cell agreement reported here is corroborating
  breadth, not the formal test.
- **Cross-arm drift on the C++ calibration.** The two Java arms were measured
  time-adjacent and are directly comparable. The `liblevenshtein-cpp` figures
  in §2 were collected in a different window of the same sweep, during which
  fixed-cell drift sentinels moved by up to 10.7% as external load on the host
  fell. That bound is an order of magnitude smaller than the 1.8–3.1× ratios
  it is used to establish, so the qualitative conclusion (legacy Java is the
  outlier; the Rust core clears the C++ baseline) is robust to it — but the
  third-decimal ratios should be taken as provisional until the anchor is
  re-measured at sweep end and the atlas ratios are recomputed against it.

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
178 (H-J1), 179 (H-J2), and 205 (the post-optimization H-O7 decision).
Method: [`PROTOCOL.md`](../../../benchmarks/cross-language/harnesses/common/PROTOCOL.md).

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
