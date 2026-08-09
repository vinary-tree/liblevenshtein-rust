# Binding / ABI Performance Experiments — liblevenshtein-rust

Data-driven record for the wave-W8 performance closure of the resource-ABI
boundary. Each experiment states a hypothesis, the method (with the exact
benchmark or test), the measured result, the analysis, and the decision. Defects
live in the [findings ledger](FINDINGS_LEDGER.md); this file records optimization
*decisions* — including the deliberate decisions **not** to change a default —
so that a reader can see the evidence behind the shipped configuration.

Optimization changes land only on confirmed evidence (the data-driven rule).

Hardware for the recorded numbers: AMD Ryzen Threadripper PRO 5975WX (Zen 3,
32 cores), see `/home/dylon/.claude/hardware-specifications.md`. Runs pinned with
`taskset -c 2`; criterion 0.8, `--profile bench`, 100 samples after a 3 s warm-up.

## E1 — Batch-capacity sweep: is 256 the right default?

- **Hypothesis.** The batched cursor amortizes the FFI boundary, so drain time
  should fall as the batch capacity grows (fewer `⌈M/cap⌉` crossings), with
  diminishing returns past some capacity that justifies the `defaultMatchBatch`
  of 256 in `bindings/api.json`.
- **Method.** `benches/ffi_boundary_benchmarks.rs`, group `cursor_drain_by_batch`:
  a distance-2 query (`"term000000"`) drained to completion over a real 2000-term
  `DynamicDawgBinding` consumed through `ResourceTransducer`, at capacities
  `{1, 32, 64, 128, 256, 512, 1024}`.
  Run: `taskset -c 2 cargo bench --features binding-integration-tests --bench ffi_boundary_benchmarks -- cursor_drain_by_batch`.
- **Result** (point estimate, µs per full drain):

  | capacity | 1 | 32 | 64 | 128 | 256 | 512 | 1024 |
  |---|---|---|---|---|---|---|---|
  | time (µs) | 302.6 | 309.2 | 310.2 | 306.6 | 309.2 | 311.3 | 314.6 |

- **Analysis.** Drain time is **flat within ≈4 %** across a 1024× capacity range —
  cap 1 is even marginally fastest, and the largest capacities are marginally
  slowest (larger batch buffers to fill/clear). For a realistic fuzzy query the
  match set is small and **traversal of the automaton dominates**, not the
  boundary crossings, so batch capacity is not the drain-time bottleneck here.
  The amortization the batched design provides is real but is exercised only by
  high-fan-out queries (large `M`); its shape is pinned exactly, independently of
  wall-clock noise, by the crossing census in E2 (`⌈M/cap⌉ + 1`). Larger
  defaults (512, 1024) buy no crossing reduction for typical queries while
  enlarging the per-cursor arena/descriptor buffers.
- **Decision.** **Keep `defaultMatchBatch = 256`.** It single-pages the match set
  of essentially every realistic query (so crossings collapse to the `+1`
  terminal pull), without the over-allocation of 512/1024, and the sweep shows no
  wall-clock advantage to changing it. No change to `bindings/api.json`.

## E2 — Boundary-crossing census: does the crossing count obey the paging law?

- **Hypothesis.** A query returning `M` matches crosses the consumer↔cursor
  boundary exactly `⌈M/cap⌉ + 1` times (the `+1` is the terminal `End` pull), and
  crosses the cursor↔provider boundary exactly once for the query-start snapshot,
  independent of capacity.
- **Method.** `tests/ffi_boundary_census.rs`
  (`boundary_crossings_follow_the_paging_law_and_snapshot_is_captured_once`): a
  distance-1 query over 20 single-character terms (each within edit distance 1 of
  the query, so `M = 20` deterministically) drained at capacities
  `{1, 4, 8, 16, 20, 256}` over the metrics-instrumented provider. Emits
  `target/ffi-census/boundary_crossing_census.tsv`.
- **Result** (`next_batch_calls` = consumer↔cursor crossings; `snapshot` = per-query
  provider captures):

  | capacity | matches | next_batch_calls | `⌈M/cap⌉+1` | snapshot/query |
  |---|---|---|---|---|
  | 1 | 20 | 21 | 21 | 1 |
  | 4 | 20 | 6 | 6 | 1 |
  | 8 | 20 | 4 | 4 | 1 |
  | 16 | 20 | 3 | 3 | 1 |
  | 20 | 20 | 2 | 2 | 1 |
  | 256 | 20 | 2 | 2 | 1 |

- **Analysis.** The measured crossings equal `⌈M/cap⌉ + 1` at every capacity, and
  the count is monotonically non-increasing in capacity — the batched design
  reduces boundary crossings exactly as the model predicts. Each query captures
  its provider snapshot exactly once (`snapshot/query == 1`), confirming the
  O(1) query-start capture (VT-SNAP). The provider refcount ledger over the run
  is balanced (each per-query snapshot context is created at refcount 1, then
  `+1 retain − 2 release → 0`, destroyed exactly once — one `context_drop` per
  query), corroborating VT-LIFE without duplicating the dedicated lifecycle test.
- **Decision.** No change; the batched cursor is confirmed to amortize crossings
  as designed. The census is wired as a correctness test (asserts the law), and
  its TSV is the human-readable artifact of the crossing behavior.

## Cross-references

- Crossing-law correctness (adversarial paging, high-degree nodes):
  `tests/abi_paging_correspondence.rs`, invariants VT-PAGE-1..6.
- Arena reuse (warm batches allocate ≈0): Verus `docs/verification/verus/ffi_batch_arena.rs`
  (LLEV-ARENA-1..3) and the arena profile `bindings/c/tests/arena_profile.c` +
  `scripts/profile-ffi-arena.sh`.
- Resource-handoff O(1) in dictionary size: `benches/ffi_boundary_benchmarks.rs`
  group `resource_handoff_vs_dict_size`.
