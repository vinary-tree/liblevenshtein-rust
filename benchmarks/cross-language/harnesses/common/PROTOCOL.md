# Cross-Language Benchmark Protocol (normative)

Every harness in `benchmarks/cross-language/harnesses/` implements THIS
document. Where a harness and this document disagree, the harness is wrong.
"MUST" is normative. The unit of comparison is a **cell**: one
`(target × backend × mode × algorithm × max_distance × queryset)` coordinate,
emitting exactly one JSON file conforming to `schema/result.schema.json`.

Mathematical definitions of the algorithms, the SplitMix64 stream, and the
FNV-1a-64 checksum — with full literature citations — live in the program
[README](../../README.md#references); this document is the bit-level
implementation contract.

## 1. CLI contract

Every harness accepts exactly these flags (unknown flags MUST abort):

```
--mode         query | construct | memory-child | verify
--algorithm    standard | transposition | merge_and_split | damerau_levenshtein
--max-distance 1 | 2 | 3
--dictionary   <absolute path to workload/dictionary.txt>
--queries      <absolute path to a workload/queries/*.txt file>
--backend      dynamic_dawg | double_array_trie | own
--out          <absolute path of the result JSON to write>
--samples      integer, default 30
--warmup-seconds  number, default 3
--gate-limit   integer, default 200        (verify mode only)
--reps         integer, default 10         (construct mode only)
--cells        <absolute path to a cells TSV>   (batch driver; see §7)
```

`--backend own` is only valid for the two legacy targets (they build their own
DAWG). New-stack harnesses construct dictionaries through the libdictenstein
facade of their language. Result JSON goes to `--out`; ALL diagnostics go to
stderr; stdout stays empty (the runner tees both).

## 2. Startup self-test (every harness, every mode)

Before touching input files the harness MUST compute and assert the seven
checksum vectors of §6.3. A failed assertion aborts with a nonzero exit. This
catches a mis-ported FNV/wrapping implementation in milliseconds instead of
poisoning a five-minute cell.

## 3. Input loading (untimed, before everything)

1. Read `--dictionary` fully into memory; split on `\n`; drop the single
   trailing empty line; every line MUST be non-empty.
2. Assert **strict byte-ascending order**: `lines[i] < lines[i+1]` bytewise
   for all i (UTF-8 bytes; the workload is lowercase ASCII). Abort on
   violation — an unsorted dictionary silently corrupts the legacy Daciuk
   DAWG builders, so this invariant is checked *everywhere*, not just in the
   generator.
3. Read `--queries` the same way (non-empty lines; no sortedness requirement).
4. Preallocate: the term list, the query list, and the per-pass sample array
   are sized once (`Vec::with_capacity`, `ArrayList(int)`, `make([]T, 0, n)`,
   etc.). No I/O, parsing, or RNG may occur after this point in any timed
   region.

## 4. Dictionary construction and transducer

- `dynamic_dawg`: libdictenstein DynamicDawg via the language facade,
  Unicode-scalar unit domain, populated with **one batch call** where the
  facade exposes one (`putAllBytes`, `update_many`, `insert_text_batch`,
  `dynamicDawg([...])`, ...); a put-loop is only acceptable where no batch
  API exists, and the result JSON MUST say so in `notes`.
- `double_array_trie`: libdictenstein DoubleArrayTrie built from the sorted
  term list.
- `own` (legacy Java): `new TransducerBuilder().dictionary(words, /*isSorted=*/true)`
  — the flag is legitimate ONLY because §3.2 asserted sortedness.
- `own` (legacy JS): `new levenshtein.Builder().dictionary(words, /*sorted=*/true)
  .algorithm(a).include_distance(true).sort_candidates(false)`.
  `sort_candidates(false)` is pinned: the new stack returns traversal order,
  so letting the legacy builder sort would bill it for work the new side
  never performs.
- Transducer: the facade's `transducer(dictionary, algorithm)` equivalent.
  Legacy targets have no `damerau_levenshtein`; the runner never schedules
  those cells (they are recorded as `skipped_unsupported` in run state, not
  by the harness).

## 5. The pass, the triple, and the gate

A **full pass** iterates the in-memory query list in file order; for each
query it opens a cursor (`query(term, max_distance)` / legacy `transduce`),
**drains it completely**, materializes each match's term and distance, and
closes the cursor (where the facade has close/Dispose/using semantics; eager
legacy arrays need no close).

Two accumulator shapes exist:

- **Gate accumulators** (untimed contexts only): the FNV checksum of §6 plus
  the triple below.
- **The triple** (timed passes): `matches` (count), `bytes` (Σ UTF-8 byte
  length of each returned term), `distsum` (Σ distance). O(1) per match,
  uniform across languages, defeats dead-code elimination. Per-byte hashing
  is deliberately EXCLUDED from timed passes — it would tax interpreted
  languages disproportionately and misattribute cost. For this ASCII
  workload, UTF-8 byte length equals code-unit length in every language;
  implementations MUST nevertheless define it as UTF-8 byte length.

**Gate pass**: in EVERY mode the harness first performs one untimed full pass
computing checksum + triple. The checksum lands in the result JSON
(`measurements.checksum_hex`), and the triple becomes the reference
`(matches, bytes, distsum)`.

## 6. Modes

### 6.1 query

```
gate_ref = gate_pass()                    # untimed, checksum + triple
deadline = now() + warmup_seconds        # >= 3 s AND >= 2 passes
passes = 0
while now() < deadline or passes < 2:
    triple = full_pass()                 # triple only, discard time
    assert triple == gate_ref.triple
    passes += 1
est = time(last warmup pass)
S = samples                              # default 30
if S * est > 300 s:                      # wall-cap rule (deterministic):
    S = max(10, floor(300 / est))        #   10..S-1 samples, never fewer than
    status = "degraded"                  #   10 even when 10×est exceeds the cap
samples_ns = preallocated [S]
for i in 0..S-1:
    t0 = mono_ns(); triple = full_pass(); samples_ns[i] = mono_ns() - t0
    assert triple == gate_ref.triple     # nondeterminism trap: abort cell
emit JSON (samples_ns, matches_per_pass, bytes, distsum, checksum, status)
```

The wall cap arithmetic is deterministic so two runs of the same cell collect
the same sample count. `status = "failed"` (and nonzero exit) only for real
faults: assertion failure, library error, OOM.

### 6.2 construct

Input files are read and sortedness asserted as usual; the gate pass is run
once against a first construction (this also warms allocator/JIT). Then:

```
for r in 0..reps-1:                      # default 10; runner passes 3 for DAT
    free previous dictionary (close/dispose where applicable)
    t0 = mono_ns(); dict = BUILD(terms); construct_ns[r] = mono_ns() - t0
```

Timed region contains ONLY the build from the already-loaded, already-sorted
in-memory list — file I/O and sorting are excluded for every implementation
(the legacy `isSorted=true` fast path and the new-stack batch insert are the
things compared). Emit `construct.times_ns[]`.

### 6.3 memory-child

Construct once, run exactly one full pass (gate accumulators), write the
result JSON (checksum, triple, `construct_ns`), exit 0. The RUNNER wraps the
process in `/usr/bin/time -v` and merges `Maximum resident set size` into the
final JSON as `memory.max_rss_kib`. Uniform across languages: peak RSS of the
whole process, interpreter/VM included — that is what a migrating user pays.

### 6.4 verify

Gate pass over only the first `--gate-limit` (default 200) queries; emit
checksum + triple; no timing fields; used by `scripts/gate.py` to compare
every target against the Rust oracle before any timing run is accepted.

## 7. Batch driver (`--cells`)

`--cells FILE` replaces `--algorithm/--max-distance/--queries/--out` with a
TSV of `algorithm \t max_distance \t queries_path \t out_path` rows. The
harness builds the dictionary ONCE per invocation (backend fixed by
`--backend`), then executes one cell per row, writing one JSON per row. It
composes with `--mode query` (each row runs §6.1, including its per-row gate
pass) and with `--mode verify` (each row runs §6.4 — this is how the
correctness gate amortizes one expensive DAT construction across its 27
oracle-comparison cells). `construct` and `memory-child` never batch. This
amortizes expensive constructions (DAT at 79 K words) across a target's 45
query cells; a fresh transducer is created per row (transducers are cheap;
the dictionary is the expensive object).

## 8. Checksum specification (bit-level, normative)

All arithmetic is unsigned 64-bit mod 2^64. Signed-64 languages (Java,
Clojure, Lua 5.4) use native wrapping ops — bit patterns are identical and
only equality is ever compared. JS uses `BigInt` masked with
`& 0xFFFFFFFFFFFFFFFFn` after every multiply/add. Swift uses `&*`/`&+`.
Fortran uses `integer(int64)` with `ieor`/`iand`/`ishft` and wrapping
multiply. OCaml `Int64` ops wrap by definition. Haskell `Word64` wraps.
Python/Ruby mask with `& 0xFFFF_FFFF_FFFF_FFFF`.

```
FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME  = 0x00000100000001b3

fnv_update(h, byte) = ((h XOR byte) * FNV_PRIME) mod 2^64     # XOR first: FNV-1a

entry(term, distance):
    h = FNV_OFFSET
    for b in utf8_bytes(term):  h = fnv_update(h, b)
    h = fnv_update(h, 0x00)                                   # separator
    for i in 0..7:              h = fnv_update(h, (distance >> 8*i) AND 0xFF)
    return h                                                  # LE64(distance)

checksum(cell) = ( Σ entry(term_i, distance_i) ) mod 2^64     # wrapping sum:
                                                              # order-insensitive,
                                                              # multiset-sensitive
serialize: 16 lowercase hex digits, zero-padded
```

### Confirmed test vectors

Derived independently by `workload/generate_workload.py --selftest` (Python
reference), the Rust oracle harness, and the C harness — all three MUST agree
(Phase 0 checkpoint). Every harness asserts these at startup (§2):

| Vector | Value |
|---|---|
| `fnv1a64("")` | `cbf29ce484222325` |
| `fnv1a64("a")` | `af63dc4c8601ec8c` |
| `entry("cat", 1)` | `9697fa3e50464bc4` |
| `entry("cat", 0)` | `b592c1475b3595e5` |
| `entry("cot", 1)` | `b8acc5d3816bcdea` |
| `checksum{("cat",0), ("cot",1)}` | `6e3f871adca163cf` |
| `checksum{}` | `0000000000000000` |

## 9. Monotonic clocks (pinned per language)

| Language | Source |
|---|---|
| Rust | `std::time::Instant` |
| C / C++ | `clock_gettime(CLOCK_MONOTONIC)` |
| Java | `System.nanoTime()` (JMH manages timing in timed cells) |
| JavaScript / ClojureScript | `process.hrtime.bigint()` |
| Python | `time.perf_counter_ns()` |
| .NET | `System.Diagnostics.Stopwatch` |
| Go | `time.Now()` (monotonic reading) |
| Swift | `ContinuousClock` |
| Ruby | `Process.clock_gettime(Process::CLOCK_MONOTONIC, :nanosecond)` |
| Fortran | `system_clock` with `int64` count/count_rate |
| OCaml | `Unix.clock_gettime` `Monotonic` (verified against OCaml 5 unix) |
| Haskell | `System.Clock.getTime Monotonic` (package `clock`) |
| Lua 5.4 | harness-local `bench_clock.so` C shim exporting `now_ns()` via `clock_gettime(CLOCK_MONOTONIC)` — `os.clock()` is CPU time and MUST NOT be used |
| Clojure | `System/nanoTime` |

## 10. Fairness rules (summary of binding decisions)

1. One OS process per `(target × backend × mode)` invocation; legacy and new
   never share a process (JIT/GC profile isolation). The `--cells` driver
   keeps one process per target×backend for query cells — acceptable because
   all its cells belong to the same side.
2. The runner pins CPUs: single-threaded runtimes `taskset -c 2`; VM runtimes
   (JVM, Node, .NET, Go, ClojureScript-on-Node) `taskset -c 2-9`, which gives
   JIT and GC threads room without letting them roam the machine. Both sides
   of a head-to-head pair always get identical cpusets, and SMT siblings are
   left idle (asserted by doctor).

   **Documented limitation.** On this host (Threadripper PRO 5975WX, 4 CCDs
   of 8 cores) `2-9` straddles two L3 domains: cores 2–7 share CCD0's L3
   (with 0–7) and cores 8–9 share CCD1's L3 (with 8–15), verified via
   `/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`. A VM
   runtime's threads can therefore land across a CCD boundary, adding
   cross-CCD L3 variance to VM-hosted targets. This does NOT bias any
   comparison — every arm of every pair runs under the identical cpuset, and
   the single-threaded targets on `-c 2` are unaffected — but it does widen
   the VM targets' dispersion, which is why the reported statistics are
   medians with MAD and bootstrap intervals rather than means. The cpuset is
   deliberately held fixed for the whole program: switching to a true
   single-CCD set mid-program would make later cells incomparable with the
   165 Java cells already collected.
3. JVM: JDK 26 both sides, `-Xms2g -Xmx2g` both sides, default G1, JMH forks.
4. Node 26 for all four JS backends and both JS sides; no V8 flag overrides.
5. Go: `GOMAXPROCS=1`, `GOGC=100` (explicit defaults, recorded).
6. Ruby: default MRI (no `--yjit`) is the primary row; a labeled 4-cell
   `--yjit` sensitivity run is reported separately.
7. Python: CPython, GC enabled, `PYTHONHASHSEED=0`.
8. Haskell: default RTS, single-capability.
9. .NET: workstation GC, default tiering (recorded).
10. Everything measurable that deviates from a language's out-of-the-box
    defaults MUST appear in the result `notes`.

## 11. Result JSON and the runner post-fill contract

One file per cell conforming to `schema/result.schema.json` (version 1.0.0).
Division of labor:

- **The harness** fills: `schema_version`, `suite`, `timestamp_utc`,
  `target` (except `artifact.sha256`), `dictionary`
  (`file`/`term_count`/`structure`/`unit_domain`/`construct_ns`), `workload`
  (`queryset`/`file`/`query_count`), `algorithm`, `max_distance`, `mode`,
  `protocol`, `measurements` / `construct`, `status`, `notes`.
- **The runner** (`run-one.sh` merge step) post-fills: `run_id`,
  `dictionary.sha256`, `workload.sha256`, `target.artifact.sha256` (where it
  knows the artifact), `environment_ref`, `cell_snapshot` (cpuset, MHz
  before/after, 1-min loadavg), and for memory mode the whole `memory`
  object parsed from `/usr/bin/time -v`. Rationale: languages without a
  stdlib SHA-256 (C, Fortran, Lua, OCaml) must not each carry a hash
  implementation; the runner computes each digest once per run.

Schema validation runs AFTER the merge; a cell is not accepted until the
merged JSON validates. Serialization: UTF-8, LF, two-space indent or
compact — consumers parse, never diff.
