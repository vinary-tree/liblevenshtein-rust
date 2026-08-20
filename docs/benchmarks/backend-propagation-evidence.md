# All-backend optimization-propagation evidence

## Purpose

This protocol and its completed 51-replicate matrix close the no-regression
evidence gap between an optimization proved on one Dynamic DAWG workload and
the generic code inherited by every production dictionary backend. It records
both semantic equivalence and the cost of construction and query execution.

A **cell** is one backend family and one edge-unit domain. A unit domain is
byte (`u8`), Unicode scalar (`char`), or token (`u64`). Each cell always emits
five rows: construction, Standard Levenshtein, optimal string alignment (OSA),
merge-and-split, and unrestricted Damerau–Levenshtein. Unsupported cells emit
five explicit `inapplicable` records with an invariant-based reason and no
fabricated timing.

## Coverage model

The matrix compiles with `benchmark-controls`, `pathmap-backend`, and
`persistent-artrie`, so optional production families cannot silently vanish.
The 18 concrete `DictionaryNode` implementations are reached through ten
families:

| Backend family | Byte | Unicode scalar | `u64` | Construction meaning |
|---|---:|---:|---:|---|
| Dynamic DAWG | applicable | applicable | applicable | Optimized unordered bulk constructor |
| Double-array trie | applicable | applicable | inapplicable | Static BASE/CHECK placement |
| PathMap | applicable | applicable | inapplicable | Persistent PathMap publication |
| Suffix automaton | applicable | applicable | inapplicable | Suffix-link index construction |
| SCDAWG | applicable | applicable | inapplicable | Compact suffix-DAWG construction |
| Persistent suffix automaton | applicable | applicable | inapplicable | In-memory persistent suffix graph |
| Persistent suffix tree | applicable | applicable | inapplicable | Compressed persistent suffix graph |
| Persistent SCDAWG | applicable | applicable | inapplicable | Persistent compact suffix graph |
| Persistent ARTrie | applicable | applicable | applicable | Generic overlay with byte, char, or token key encoding |
| Persistent vocabulary | inapplicable | applicable | inapplicable | Unicode-scalar keys mapped to stable `u64` vocabulary IDs |

This is 30 family/domain cells and exactly 150 rows per arm and replicate. The
suffix families intentionally retain suffix-language semantics; their result
checksums are compared only against the same backend cell in the other arm,
not against a finite-term trie whose accepted language differs.

## Shared measurement kernel

Construction is backend-specific, but every query row uses one monomorphized
generic function constrained by `DictionaryNode<Unit = U>`. A small borrowed
dictionary adapter delegates `traversal_root()` to the concrete dictionary.
Consequently, measuring several algorithms against one constructed backend
does not accidentally replace its compact snapshot cursor with an owned-node
fallback.

For each query, the kernel consumes the public units-native candidate iterator,
checks the reported distance and result-length bounds, and folds every result
unit and distance into an order-sensitive checksum. Control and treatment must
agree on applicability, reason, result count, and checksum for each backend,
unit, algorithm, and stage.

The global counting allocator reports allocation/deallocation calls, requested
bytes, and peak live-byte growth during each stage. Linux runs also sample
`VmRSS` after the stage. These are complementary metrics: requested bytes
describe Rust allocator traffic, whereas resident set size (RSS) includes
allocator fragmentation, mappings, and backend storage visible to the process.

## Control and treatment

The treatment profile clears all benchmark-only legacy switches. The
`legacy-shared-kernels` profile enables the exact compatibility controls for
snapshot cursors, traversal-buffer reuse, packed Standard/OSA/merge-and-split
rows, static packed dispatch, dense generated targets, characteristic indexes,
and local state subsumption. This bundle is a causal fallback profile, not a
claim that it recreates an arbitrary historical commit.

The runner accepts distinct control and treatment binaries as well. This is
required when comparing construction changes that have no runtime switch. Both
binary SHA-256 digests are recorded in every applicable and inapplicable row.

## Bounds and validity gates

The harness fails before writing a valid row if any of these conditions is
false:

1. Every constructed dictionary reports the requested cardinality and contains
   every source term.
2. Every query candidate reports a distance within the configured budget and a
   term no longer than the corpus maximum.
3. Terms, query operations, result count, allocated bytes, and RSS remain below
   the row's explicit hard bounds.
4. Every arm emits exactly 150 rows and each family/domain cell emits exactly
   five rows.
5. Inapplicable rows contain no elapsed time, allocation measurement, result,
   or checksum.
6. Control and treatment semantic checksums and result counts match exactly.

The result bound uses the conservative finite-substring ceiling
$`\mathrm{queries}\times\mathrm{terms}\times\mathrm{maximum\_term\_units}^{2}`$,
which covers suffix-language backends without pretending that they accept only
the original term set.

## Executable protocol

Build the example once; both profiles are selected inside that exact binary.
Do not collect timings while another release build, benchmark, profiler, or
sibling last-level-cache CPU is active:

```console
cargo build --release \
  --example backend_propagation_matrix \
  --features "benchmark-controls pathmap-backend persistent-artrie"
```

Then collect 51 admitted replicates. The same binary may be supplied for both
arms when only benchmark-control fallbacks are under test:

```console
benchmarks/causal/run-backend-propagation-matrix.sh \
  target/release/examples/backend_propagation_matrix \
  target/release/examples/backend_propagation_matrix \
  benchmarks/causal/evidence/YYYY-MM-DD/backend-propagation-matrix.csv \
  51 3 256 64 1 2
```

The 256-term default is a deliberate safety calibration, not a reduced backend
inventory. At 512 generated terms, the persistent suffix-tree byte constructor
completed with modest retained RSS but produced 3,397,057,939 bytes of allocator
traffic in one stage, correctly tripping the fixed 2 GiB churn bound. At 256
terms, the same complete 150-row arm stays below every bound while preserving
all families, unit domains, algorithms, distance two, and 64 mixed exact/edited
queries. The hard bound was not raised to admit an allocator-heavy workload.

The runner performs three unrecorded processes per arm, alternates arm order,
rotates family/domain and algorithm order by replicate, invokes the strict
host-load and last-level-cache topology gate before and after each measured
process, and monitors competing benchmark processes once per second. It
refuses to overwrite evidence and writes an adjacent `*-host-load.jsonl`
ledger containing only admissions for committed control/treatment pairs. Each
pair's four accepted pre/post records are first committed to an immutable file
under `*-admissions/`; rejected gates go to
`*-host-load-rejections.jsonl`. Partial diagnostic and rejection evidence is
retained after a failure.

Every legacy switch used by this matrix is compiled only by the explicit
`benchmark-controls` feature. `perf-instrumentation` and
`resource-profiling` imply that feature for their own causal binaries; an
ordinary production build contains constant optimized branches and does not
interpret these environment variables. A structural smoke check must show
identical result signatures but different legacy/treatment work (for example,
allocation counts) before a timing matrix is admitted.

Pass `--resume` after the ordinary arguments to continue an interrupted run.
Resume requires complete 300-row control/treatment pairs in monotone replicate
order, the same arm profiles and binary digests, and no nonempty
foreign-contention ledger. It rebuilds the aggregate accepted ledger only from
the committed per-pair admission files; benign contention-monitor diagnostics
do not block recovery. It never replaces an already accepted pair. A legacy
mixed host ledger is archived as `*-host-load-pre-transactional.jsonl` while
its rejected rows are split from the admissions that prove complete pairs.

For compile and structural checks only, use `--header-only` or the example's
focused tests. Such output is diagnostic and must not be reported as timing
evidence.

After a complete run, validate every pair and produce deterministic cell,
family, and algorithm summaries with:

```console
python3 -B benchmarks/causal/analyze-backend-propagation-matrix.py \
  benchmarks/causal/evidence/YYYY-MM-DD/backend-propagation-matrix.csv \
  --output benchmarks/causal/evidence/YYYY-MM-DD/backend-propagation-matrix-analysis.json
```

The machine-readable row contract is
[`backend-propagation-matrix.schema.json`](../../benchmarks/causal/schemas/backend-propagation-matrix.schema.json).

## Completed result (2026-08-20)

The final exact binary (SHA-256
`85f08fe80f2466987c855423c11f5f1dcfdc4ba2bc62dd9e3ea36e39976d2e94`)
completed all 51 admitted pairs. The analyzer validated 15,300 rows: 150 rows
per arm and replicate, 30 explicit family/domain cells, and exact equality of
applicability, reason, result count, and checksum across every control/treatment
pair. There are 21 applicable family/domain cells, hence 84 applicable query
cells and 21 construction negative controls.

| query grouping | cells | control / treatment geomean | median | range | treatment wins |
|---|---:|---:|---:|---:|---:|
| all backends and algorithms | 84 | **1.788x** | 1.732x | [0.999x, 4.800x] | 82 / 84 |
| Standard | 21 | **2.058x** | 2.173x | [1.003x, 4.080x] | 21 / 21 |
| OSA/transposition | 21 | **1.827x** | 1.757x | [0.999x, 4.353x] | 20 / 21 |
| merge-and-split | 21 | **2.121x** | 2.132x | [1.000x, 4.800x] | 21 / 21 |
| unrestricted Damerau | 21 | **1.280x** | 1.214x | [0.999x, 2.796x] | 20 / 21 |

The two nominal control wins are persistent-suffix-tree/char OSA at 0.99942x
and unrestricted Damerau at 0.99902x. Their deterministic bootstrap median
intervals are respectively [0.97678x, 1.00694x] and [0.97543x, 1.00632x].
Both effects are below one tenth of one percent and statistically include
identity. The construction negative control independently calibrates the
same-binary protocol's noise: its 21-cell geomean is 1.00218x, median 1.00047x,
and range [0.98466x, 1.03097x]. The two query cells are therefore measurement-
equivalent, not material regressions.

The backend-family result demonstrates that the gain is genuinely propagated:

| backend family | query cells | geomean | treatment wins |
|---|---:|---:|---:|
| Dynamic DAWG | 12 | **2.723x** | 12 / 12 |
| double-array trie | 8 | **2.595x** | 8 / 8 |
| PathMap | 8 | **1.461x** | 8 / 8 |
| suffix automaton | 8 | **1.155x** | 8 / 8 |
| SCDAWG | 8 | **2.266x** | 8 / 8 |
| persistent suffix automaton | 8 | **1.398x** | 8 / 8 |
| persistent suffix tree | 8 | 1.001x | 6 / 8, two equivalent |
| persistent SCDAWG | 8 | **2.395x** | 8 / 8 |
| persistent ARTrie | 12 | **1.760x** | 12 / 12 |
| persistent vocabulary | 4 | **1.635x** | 4 / 4 |

Persistent ARTrie is the important causal check for the final capability fix.
Its byte, char, and `u64` forms all win under all four algorithms; cell medians
range from 1.099x to 2.446x. Ordinary query capture remains lazy instead of
constructing an O(dictionary) dense overlay projection for each query. A
resource producer that explicitly warms and amortizes that projection can
still advertise the native snapshot cursor afterward.

The authoritative raw CSV has SHA-256
`463a9adfd79e98ad4a94b1b32bd17f1f278522a963f7d91b5224cf51ac24a686`;
the validated
[`analysis`](../../benchmarks/causal/evidence/2026-08-19/backend-propagation-matrix-analysis.json)
has SHA-256
`5f75fbf9266eaa94c7ee0961780f90a4ffb656d335f15d36a35d921e48dce28d`.
The superseded pre-gate evidence is retained in the adjacent
`backend-propagation-pre-full-projection-gate` directory with its invalidation
reason rather than silently removed.
