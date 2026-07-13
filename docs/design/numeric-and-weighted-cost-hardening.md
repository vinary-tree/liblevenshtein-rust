# Numeric, Weighted-Cost, and Correctness Hardening

[← Documentation Index](../README.md)

**Status**: Implemented
**Scope**: `src/transducer`, `src/time_series`, `src/phonetic`, `src/cache`, CLI/grep, WASM/FFI

---

## Executive Summary

This document records the **durable correctness invariants** established while hardening
the crate's numeric domains, weighted-edit-distance semantics, automaton contracts, and
UTF-8/serialization boundaries. It is a design-rationale companion to the code: it states
*what* each invariant is, *why* it holds, and *how* it is verified, so that future changes
preserve them.

Two of these invariants correspond to **latent correctness bugs** that were surfaced by a
new reference-DP property oracle and fixed — the weighted automaton previously returned
wrong answers for any non-unit insertion/deletion cost. Both are covered by regression tests.

---

## Terminology

| Symbol / term | Definition |
|---|---|
| `q`, `t` | the query term and a dictionary term, of lengths `m` and `n` |
| position `(i, e)` | an automaton state having consumed `i` query characters at accumulated cost `e` |
| `term_index` (`i`) | characters consumed from the query term; invariant `$0 \le  i \le  m$` |
| `accumulated_cost` (`e`) | total weighted cost of the edit operations taken to reach a position |
| `OperationCostsF64` | per-operation weights: `match_cost` (always `0`), `substitution`, `insertion`, `deletion`, `transposition`, `split`, `merge` |
| subsumption | `p1` *subsumes* `p2` when every acceptance reachable from `p2` is reachable from `p1` at no greater cost; the subsumed position is pruned |
| admissible lower bound | a distance estimate that is guaranteed **never to exceed** the true distance (required for correct pruning / KNN) |
| characteristic vector | the per-window boolean vector marking which query positions match the current dictionary character |

---

## 1. Numeric-domain invariants (Phase 1)

**Invariant N1 — no silent wraparound or lossy narrowing.** Every integer/float conversion
on a runtime-influenced value is either *provably bounded*, *checked* (returns `Option`/errors),
or *saturating* (documented). The residual-cast audit classified every remaining `as` cast; the
only `wrapping_*` operation in the crate is Myers' bit-parallel recurrence
(`src/distance/myers.rs`), where modular carry propagation across the word is **load-bearing and
intentional** (documented inline).

**Invariant N2 — capacity arithmetic cannot overflow into a tiny allocation.** Capacity and
length computations use `checked_*`/`saturating_*`/`try_reserve_exact`; the surviving additive
capacity sites are constant (`HEADER_SIZE + 1024`, `BITSET_CAPACITY + 1`).

Verification: `cargo clippy --all-features -- -D warnings`; boundary unit tests on the shared
conversion helpers (`usize::MAX`, `i64::MAX`, `$\pm \infty$`, `NaN`, huge-finite inputs).

---

## 2. Weighted-edit-distance soundness (Phase 2)

The float-weighted transducer (`CandidateIteratorF64`, `OperationCostsF64`) supports non-unit
operation costs. Three invariants make it sound; the first two fix real defects that were masked
because the existing tests only ever used unit `insertion = deletion = 1`.

### Invariant W1 — weighted subsumption uses the worst-case realignment cost

`PositionF64::subsumes` prunes `p2 = (j, f)` under `p1 = (i, e)` only when

```
e ≤ f + ε   ∧   |i − j| · max(insertion, deletion) ≤ (f − e) + ε
```

The `$|i - j|$` term is the number of index-realignment steps `p1` must spend to cover `p2`.
Each such step costs at least one insertion or deletion, so the **worst-case** per-step cost is
`max(insertion, deletion)`. Using the maximum is *conservative and sound*: subsumption fires only
when even the most expensive realignment fits in the cost slack `$(f - e)$`, so a position leading
to the sole in-budget match is never pruned.

The naive unit-cost bound `$|i - j| \le  f - e$` is only correct when `$\max (\text{insertion}, \text{deletion}) \le  1$`;
with e.g. `insertion = deletion = 2` it over-prunes and drops real matches (worked example: with
`p1 = (2, 0)`, `p2 = (0, 2)`, budget `2`, the unit bound wrongly subsumes `p2`, but `p1` needs two
insertions at cost `2` each `= 4 > 2`). The cost is threaded through
`subsumes`/`StateF64::insert`/`StateF64::merge` from `transition_f64.rs`.

### Invariant W2 — the reported distance weights trailing edits and rejects over-advanced positions

For a final dictionary node in non-substring mode, the reported distance is

```
distance = min over accepting positions (i, e) with i ≤ m  of  e + (m − i) · deletion
```

(`StateF64::infer_distance`). Two prior defects:

1. the `$(m - i)$` trailing query characters that must be deleted to reach acceptance were charged
   at a hard-coded unit cost instead of `· deletion`; and
2. a padded look-ahead window let the transition emit a spurious position with `term_index = m + 1`
   (a substitution/match *one past the query end*); `saturating_sub` masked it (remaining `→ 0`) so
   its cheaper substitution cost was reported via the `min` instead of the true trailing-insertion
   cost.

The fix charges `$(m - i) \cdot  \text{deletion}$` **and** restricts acceptance to positions with `$i \le  m$`
(over-advanced positions are transition artifacts, never valid final states). Substring mode
(`min_distance`) is unaffected: the `prefix_mode` guard in `transition_position_f64` short-circuits
before generating such positions.

### Invariant W3 — `MsmPosition`/`PositionF64`/`GeneralizedPosition` obey the `Eq`/`Ord`/`Hash` contract

`a == b ⇔ a.cmp(b) == Equal`, and `a == b ⇒ hash(a) == hash(b)`. Float fields compare via
`total_cmp`/`to_bits`; `GeneralizedPosition::Ord` tiebreaks on `entry_char` (which the derived
`Eq`/`Hash` include for the Splitting variants). This is required by `binary_search` and any
ordered/hashed container. Note these are *dedup/ordering* contracts and are **independent** of the
`$\varepsilon$`-tolerant `msm_subsumes` pruning relation, which is unchanged.

### Verification methodology — reference-DP property oracle

`tests/weighted_subsumption_soundness.rs` cross-checks the weighted automaton against an independent
reference weighted-Levenshtein dynamic program over 1200+ deterministically-fuzzed cases (small
alphabet, lengths `0..=5`, `$\text{insertion} = \text{deletion} \in  {1, 1.5, 2, 3}$`, random substitution), plus the
exact audit counterexample. For every dictionary term whose true distance is within budget it asserts
**completeness** (the term is returned) and **exactness** (the reported distance equals the DP), and
for every returned candidate it asserts soundness. This oracle both guards W1 and *discovered* W2.

---

## 3. Automaton-semantics notes (Phase 2)

- **`Algorithm::MergeAndSplit` computes the generic (unconstrained) Merge-and-Split metric.** In this
  metric merge and split are character-agnostic structural operations *by definition* — any two adjacent
  symbols merge to one, any symbol splits to two, at cost `1` — so `product.rs` correctly does not
  validate the collapsed/split characters (`nfa_advance` consumes any edge). The result is exact for the
  metric and deliberately more permissive than plain Levenshtein (the purpose of merge/split); it never
  misses a match. A character-*constrained* variant (e.g. an OCR/phonetic confusion table such as
  `rn`↔`m`) would be a separate feature requiring a merge-relation map.
- **`phonetic_weight` is reserved, not applied.** The parameter is stored (clamped `$\ge  0$` to preserve
  monotone-cost pruning) but not yet applied to matching cost; `PhoneticCandidate::phonetic_cost` is
  always `0.0`. Docstrings were corrected to remove the earlier false "added to total cost" claim.
- **`OperationCostsF64::is_valid`** now also requires finiteness (`$+\infty$` previously passed and would
  break monotone pruning); automaton constructors `debug_assert!` validity.

---

## 4. Time-series correctness (Phase 3)

- **Invariant T1 — every pruning lower bound is admissible.** The length lower bound `$|m - n| \cdot  c$`
  and the interval-MSM bounds (`msm_interval.rs`) are true lower bounds; the interval bounds are in
  fact the *exact per-interval minima*, cross-checked by brute force (`merge_lb_is_min_over_c`,
  `split_lb_is_min_over_box`, `prop_range_exact`, `prop_knn_exact`). The non-admissible Euclidean/L1
  heuristics are structurally gated behind an opt-in `LengthOnly` default and are never used on an
  exact path. Non-finite query values are rejected before any interval call.
- **Invariant T2 — tie ordering is deterministic.** `search_empty_query` (`$\tau  = +\infty$`),
  `search_non_finite_query`, and `HybridSearchIndex::search_brute_force` iterate the deterministic
  `buckets` (via `ids_in_bucket_order`) instead of the randomized `HashMap`, so equal-distance ties
  emerge in a reproducible order across runs.
- **Invariant T3 — quantization is total.** `quantize`/`quantize_u8`/`quantize_u16` clamp
  `NaN`/`$\pm \infty$`/out-of-range to valid bins; SAX `zscore_to_symbol` maps `NaN → 0` for parity.

---

## 5. String / UTF-8 / serialization safety (Phase 4)

- CLI grep match spans snap to `char` boundaries (`grep_match_byte_span`) and are reported as byte
  columns; `grep_online` tracks byte *and* char offsets separately. `token_grep` slices via
  `get().unwrap_or_default()` (defense-in-depth).
- Length-prefixed decoders (LLRE framing, `limited_read.rs`, bincode 2.x's allocation ceiling)
  checked-convert and validate before allocating/slicing.
- Non-ASCII regression tests cover 2/3/4-byte code points across grep spans and LLRE/LLEV
  serialization round-trips.

---

## Diagram — weighted subsumption realignment (Invariant W1)

```
cost
  ▲
f ┤              ● p2 = (j, f)          slack = f − e must cover the
  │             ╱                        realignment |i − j| · max(ins, del)
  │            ╱  (realign |i−j| steps,
  │           ╱    ≤ max(ins,del) each)
e ┤   ● p1 = (i, e)
  │
  └───┬─────────┬──────────────▶ term_index
      i         j

  prune p2 (keep p1)  ⇔  e ≤ f  ∧  |i − j| · max(ins, del) ≤ (f − e)
```

---

## Verification summary

| Invariant | Guarded by |
|---|---|
| N1, N2 | clippy `-D warnings`; shared-helper boundary tests |
| W1 | `tests/weighted_subsumption_soundness.rs` (reference-DP, 1200+ cases) |
| W2 | same oracle (discovered it) + `infer_distance` doc example |
| W3 | `Ord`/`Eq` contract tests (generalized position; `PositionF64`; `MsmPosition`) |
| T1 | `prop_range_exact`, `prop_knn_exact`, interval-bound brute-force property tests |
| T2 | determinism regression tests (identical `Vec` across runs) |
| T3 | quantization/SAX boundary tests |
| Phase 4 | non-ASCII grep/serialization regression tests |

Whole-crate gate: `cargo check`/`clippy -D warnings`/`test` under `--all-features` (4890 tests),
`git diff --check`, incomplete-code marker scan, and `pgmcp bug-gate`.
