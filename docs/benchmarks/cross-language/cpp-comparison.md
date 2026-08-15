# C++ vs C++: liblevenshtein-cpp against the Rust-backed C++ facade

**Status:** measured, complete (45 shared cells per arm, plus construction and
memory). **Run:** `phase0`.
**Verdict on the migration thesis: SUPPORTED on all three axes** — query
throughput, construction time, and peak memory — though the query margin is
narrow and the reason it is narrow is itself the most useful result here.

This is the third of three head-to-head comparisons in the cross-language
benchmark program. Unlike the Java pair, where the legacy library won decisively
on throughput (see [`java-comparison.md`](java-comparison.md)), the C++ pair
favors the Rust-backed side in every measured category.

## 1. The two implementations

| | Legacy | Rust-backed |
|---|---|---|
| Artifact | `universal-automata/liblevenshtein-cpp` @ `5c0f9e9169aa` (declared version 4.0.0) | `vinary-tree` 0.10.0, C++ facade over the `llev_*` C ABI |
| Implementation | Pure C++, built in-tree with CMake | Rust core reached through a C ABI, wrapped in an RAII C++23 header |
| Dictionary | `SortedDawg` (built from sorted input) | libdictenstein `DynamicDawg` behind the `ldict_*` ABI |
| Compiler | GCC 16.1.1, `-DCMAKE_BUILD_TYPE=Release` | GCC 16.1.1, `-O2`; Rust core `-C target-cpu=native`, release |
| External deps | protobuf 35.1.0 | none beyond the two cdylibs |

*DAWG* = directed acyclic word graph, the minimal acyclic automaton recognizing
a finite word set, built incrementally from lexicographically sorted input after
Daciuk et al. [1]. *RAII* = Resource Acquisition Is Initialization, the C++ idiom
binding a resource's lifetime to an object's scope. *Query* means: given a term
and a maximum edit distance `d`, enumerate every dictionary term within `d`
edits, draining the resulting cursor fully.

**Provenance caveat, stated up front.** The upstream C++ repository carries no
release tags, and its working tree was dirty at build time. The measured artifact
is therefore the *version under active development*, not a published release:
commit `5c0f9e9169aa`, working-tree diff SHA-256
`56860cce6bd60060…`, built 2026-08-14T13:11:41Z. This is recorded in
`.stage/cpp-legacy/provenance.json`. To measure clean `HEAD` instead, point
`XL_LEGACY_CPP_REPO` at a `git worktree add` checkout.

## 2. Headline result

Both arms ran the same committed 79,343-word aspell en_US dictionary, the same
1,000-query sets, the same cpuset, and 30 timed samples per cell. Across the
**45 shared cells** (3 algorithms × 3 distances × 5 query sets — `damerau_levenshtein`
is excluded because the legacy library does not implement it):

```math
\frac{t_{\text{legacy}}}{t_{\text{vinary}}}: \quad
\text{geomean } 1.158, \quad \text{median } 1.133, \quad
\text{range } [0.927,\ 1.702]
```

A ratio above 1 means the Rust-backed facade is *faster*. It wins **38 of 45
cells**. The margin is not uniform — it grows sharply with edit distance:

| distance | geomean speedup | | algorithm | geomean speedup |
|---|---|---|---|---|
| d = 1 | 1.022 | | merge_and_split | 1.229 |
| d = 2 | 1.115 | | transposition | 1.131 |
| d = 3 | **1.363** | | standard | 1.117 |

At `d = 1` the two are effectively tied; by `d = 3` the Rust-backed side is over
a third faster. Section 3 explains why, and the explanation is the actionable
part of this document.

### Where the legacy library wins

Seven cells go to legacy C++, and six of the seven are at `d = 1`:

| algorithm | d | query set | legacy µs/q | vinary µs/q | ratio |
|---|---|---|---|---|---|
| merge_and_split | 2 | oov | 9657.2 | 10420.8 | 0.927 |
| transposition | 1 | oov | 84.4 | 88.4 | 0.955 |
| standard | 1 | std-d3 | 106.8 | 110.2 | 0.969 |
| standard | 1 | std-d2 | 113.8 | 117.4 | 0.969 |
| standard | 1 | std-d1 | 124.6 | 127.4 | 0.978 |
| transposition | 1 | tr-d3 | 112.8 | 114.7 | 0.983 |
| standard | 1 | oov | 81.5 | 82.5 | 0.989 |

Every loss is within 7.3%, and the clustering at `d = 1` is the signature of a
fixed per-query cost that the Rust core's faster automaton has too little work
to amortize.

## 3. The decomposition: a 1.9× core, most of it spent at the boundary

The pure Rust core was measured on the identical 45 coordinates with no FFI and
no C++ wrapper. Writing $`t_{\text{legacy}}`$, $`t_{\text{facade}}`$, and
$`t_{\text{core}}`$ for the median pass time of the legacy C++ library, the
Rust-backed C++ facade, and the pure Rust core respectively, comparing all three
arms decomposes the headline exactly:

| comparison | geomean | range | what it measures |
|---|---|---|---|
| $`t_{\text{legacy}} / t_{\text{core}}`$ | **1.895×** | [1.48, 2.43] | how much faster the Rust automaton is |
| $`t_{\text{facade}} / t_{\text{core}}`$ | **1.636×** | [1.38, 2.05] | what the C ABI + RAII wrapper costs |
| $`t_{\text{legacy}} / t_{\text{facade}}`$ | **1.158×** | [0.93, 1.70] | what a migrating user actually gets |

These are consistent by construction, and the identity holds on the measured
geomeans to three decimals:

```math
\frac{1.895}{1.636} = 1.158
```

**The Rust core is 1.895× faster than the legacy C++ implementation, and the
binding boundary gives back 1.636× of it.** A user migrating today receives
1.158×; the remaining $`1.895 / 1.158 \approx 1.64`$ is currently spent crossing
the ABI rather than delivered.

This also explains the distance gradient in §2. The ABI cost is dominated by
per-match work — arena writes and `LlevMatch` descriptor construction in the
batch protocol — while the core's advantage grows with the amount of automaton
work per query. At `d = 1` few matches are returned per query and the boundary
cost is proportionally largest; at `d = 3` each query returns far more matches,
and although the boundary cost rises with them, the core's advantage rises
faster.

**Consequence.** The C++ migration pitch is currently limited by the ABI, not by
the automaton. Reducing per-match ABI cost — tracked as **H-O8** in the
optimization epic, already identified there as the highest-leverage item — would
move this pair from 1.158× toward the core's 1.895×, roughly a 64% improvement
in the migration argument without touching the automaton at all. No other change
available to this pair has comparable leverage.

## 4. Construction, memory, and correctness

### Construction: 1.52× faster

Ten timed builds per arm over the same 79,343 sorted terms, sorting excluded from
the timed region for both:

| arm | median build | MAD |
|---|---|---|
| legacy `SortedDawg` | 135.70 ms | 1.760 ms |
| vinary `DynamicDawg` | **89.10 ms** | 0.146 ms |

The Rust-backed build is **1.523× faster** and an order of magnitude more
repeatable (MAD 0.146 ms versus 1.760 ms). Note the contrast with the Java pair,
where legacy construction was the *faster* side; the result does not generalize
across legacy implementations.

### Memory: 1.26× smaller

Peak resident set size, measured by `/usr/bin/time -v` around a construct-plus-
one-pass child process:

| arm | peak RSS |
|---|---|
| legacy `SortedDawg` | 49.56 MiB |
| vinary `DynamicDawg` | **39.20 MiB** |

A 20.9% reduction. This is a far narrower margin than the Java pair's
$`\approx`$ 2.95×, which is expected: both arms here are native, so neither pays
for a managed heap.

### Correctness: identical, proven

Before any timing was accepted, both arms had to reproduce the Rust oracle's
result multiset exactly, compared as match count, summed term byte length,
summed distance, and an order-insensitive FNV-1a-64 checksum over every returned
`(term, distance)` pair. They agree bit for bit — for example at
standard/`d`=1/`hits`, both return 3,620 matches with checksum
`3bdc59281f42611a`. The legacy C++ builder also served as an independent check on
the workload's sortedness invariant: `ll::sorted_dawg` returns `nullptr` on
unsorted input, and it never did.

## 5. Recommendation

**A user of `liblevenshtein-cpp` should migrate.** Unlike the Java case, every
measured axis favors the Rust-backed facade:

- **query throughput** — 1.158× geomean, winning 38 of 45 cells, and 1.363× at
  `d = 3` where fuzzy search is most expensive;
- **construction** — 1.523× faster with an order of magnitude less variance;
- **memory** — 20.9% smaller peak RSS;
- **dependencies** — protobuf 35.1.0 disappears from the dependency graph;
- **features** — `damerau_levenshtein` is available, which the legacy library
  does not implement (and which is consequently excluded from the 45 shared
  cells above).

The honest qualifier is that the throughput win is modest *today* and negligible
at `d = 1`. A user whose workload is dominated by `d = 1` queries should migrate
for the construction, memory, dependency, and feature reasons, not for query
speed. That qualifier is expected to weaken as H-O8 progresses.

## 6. Threats to validity

- **The legacy artifact is an untagged, dirty working tree.** It is the version
  under active development, not a release. Its diff is content-addressed in the
  provenance record so the exact input is reconstructible, but a published
  release could measure differently.
- **One dictionary, one shape.** 79,343 lowercase-ASCII words. Results may differ
  for much larger dictionaries, Unicode-heavy sets, or long terms.
- **Materializing drain.** Both arms fully materialize `(term, distance)` per
  match, the migration-realistic path. The Rust-backed facade also offers a
  zero-copy borrowed-batch API that was deliberately not used, since the legacy
  library has no equivalent; adopting it would narrow the ABI gap quantified in
  §3 and is the caller-side counterpart to H-O8.
- **Cross-arm drift.** Both C++ arms were measured time-adjacent, so the pair
  ratio in §2 is directly comparable. The pure Rust core figures used for the
  §3 decomposition come from an earlier window of the same sweep, during which
  fixed-cell drift sentinels moved by up to 10.7% as external host load fell.
  That bound is well below the 1.636× and 1.895× ratios it supports, but those
  two figures should be re-derived after the sweep-end anchor re-measurement
  before being quoted externally. The 1.158× pair ratio is unaffected.
- **Backend asymmetry.** The vinary C++ facade is built against `DynamicDawg`
  only. `DoubleArrayTrie` was measured for other targets and found slower on
  both query and construction axes, so its absence here does not disadvantage
  the Rust-backed arm.

## 7. Reproduction

```bash
cd benchmarks/cross-language

# Readiness (checks the legacy checkout, CMake, protoc, and the release cdylibs)
scripts/doctor.sh --targets cpp,cpp-legacy

# Correctness gate — must be green before any timing is accepted
python3 scripts/gate.py results/phase0

# Timed cells for both arms
scripts/run-one.sh cell results/phase0 cpp        dynamic_dawg query standard 1 hits
scripts/run-one.sh cell results/phase0 cpp-legacy own          query standard 1 hits

# Aggregate; writes tables/pair_cpp.md and summary.json pair_cpp
python3 scripts/aggregate.py results/phase0
```

To measure clean upstream `HEAD` rather than the working tree:

```bash
git -C /path/to/liblevenshtein-cpp worktree add /tmp/llcpp-clean 5c0f9e9169aa
XL_LEGACY_CPP_REPO=/tmp/llcpp-clean scripts/doctor.sh --targets cpp-legacy
```

## References

1. Daciuk, J., Mihov, S., Watson, B. W., & Watson, R. E. (2000). Incremental
   Construction of Minimal Acyclic Finite-State Automata. *Computational
   Linguistics*, 26(1), 3–16. <https://doi.org/10.1162/089120100561601>
2. Schulz, K. U., & Mihov, S. (2002). Fast string correction with
   Levenshtein automata. *International Journal on Document Analysis and
   Recognition*, 5(1), 67–85. <https://doi.org/10.1007/s10032-002-0082-8>
3. Fowler, G., Noll, L. C., & Vo, K.-P. FNV Non-Cryptographic Hash Algorithm.
   IETF Internet-Draft. <https://datatracker.ietf.org/doc/html/draft-eastlake-fnv-21>
4. Protocol used by every arm:
   [`benchmarks/cross-language/harnesses/common/PROTOCOL.md`](../../../benchmarks/cross-language/harnesses/common/PROTOCOL.md)
5. Companion comparisons: [`java-comparison.md`](java-comparison.md).
