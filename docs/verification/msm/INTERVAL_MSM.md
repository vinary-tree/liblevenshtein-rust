# Interval-Relaxed MSM Trie Search — Design & Verification

This document describes the **exact Move-Split-Merge (MSM) similarity search over
a trie of quantized reference series** implemented by
[`MsmTransducer`](../../../src/time_series/msm_transducer.rs), the interval-relaxed
MSM lower bounds it relies on
([`msm_interval`](../../../src/time_series/msm_interval.rs)), and the per-bin
interval bound `QuantizationConfig::bin_bounds`
([`encoding.rs`](../../../src/time_series/encoding.rs)). It records the design so
it can be reconstructed from scratch, and maps each mathematical claim to its
machine-checked proof (Coq/Rocq) or model (TLA+).

## 1. Problem and motivation

`HybridSearchIndex::search_exact` filters candidates with a *lossy* Levenshtein
distance over the quantized byte sequences, then re-scores survivors with the
exact MSM distance. The Levenshtein pre-filter can **drop a true MSM-near
neighbor** whose quantized form is several bin-edits away (a false negative),
because edit distance over bins is not a lower bound on MSM. The integration test
`transducer_is_exact_and_supersets_hybrid`
([`tests/msm_transducer_tests.rs`](../../../tests/msm_transducer_tests.rs))
exhibits exactly this: at `tau = 100`, `HybridSearchIndex` returns `{0,1,3}` while
the exact set is `{0,1,2,3}`.

`MsmTransducer::search_range(query, τ)` returns **exactly** `{ id : MSM(query,
ref_id) ≤ τ }` — no false negatives and no false positives. For non-empty
queries it walks the trie with an MSM dynamic program evaluated on the
per-element **bin intervals**, using admissible lower bounds for pruning and
exact full-precision re-scoring at finals. Empty-query edge cases are handled
directly by `MsmConfig::distance`: empty references are returned at distance 0,
and non-empty references are excluded because the Rust MSM semantics give them
infinite distance.

## 2. The MSM recurrence

For non-empty query `x` (rows `i`, length `m`) and non-empty target `y` (columns
`j`, length `n`), the MSM DP (Stefan et al. 2012;
[`msm.rs`](../../../src/time_series/msm.rs)) is

```
cost[i][j] = min{ cost[i-1][j-1] + |x_i − y_j|,            (Move)
                  cost[i-1][j]   + C(x_i, x_{i-1}, y_j),   (Merge-like)
                  cost[i][j-1]   + C(y_j, x_i, y_{j-1}) }  (Split-like)
C(a,b,c) = c_const                       if b ≤ a ≤ c or b ≥ a ≥ c
         = c_const + min(|a−b|, |a−c|)   otherwise
```

In the transducer `x` is the full-precision query and `y` is the trie path; each
target element `y_j` is known only up to its quantization bin `[lo, hi]`.

## 3. The three admissible per-element lower bounds

Let `interval_dist(v, lo, hi) = max(0, lo−v, v−hi)` be the distance from a scalar
`v` to `[lo, hi]`. With the free (interval-valued) arguments being the target
values:

* **Move** `|x_i − y_j|`, `y_j ∈ [lo,hi]` → `interval_dist(x_i, lo, hi)` (minimized
  at `y_j = clamp(x_i, lo, hi)`).
* **Merge** `C(x_i, x_{i-1}, y_j)`, `y_j ∈ [lo,hi]` → `c_func_merge_lb`. Penalty 0
  iff some `c ∈ [lo,hi]` places `x_i` between `x_{i-1}` and `c`
  (`(a≥b ∧ hi≥a) ∨ (a≤b ∧ lo≤a)`), else `min(|a−b|, interval_dist(a,lo,hi))`.
* **Split** `C(y_j, x_i, y_{j-1})`, `y_j ∈ [a_lo,a_hi]`, `y_{j-1} ∈ [c_lo,c_hi]` →
  `c_func_split_lb`. The "between" set unions to `[min(b,c_lo), max(b,c_hi)]`; the
  penalty is the gap from `[a_lo,a_hi]` to that union (0 on overlap).

Each is the **exact minimum** over its interval box, hence the tightest
admissible lower bound. Composing them through the DP yields a per-node column
that **lower-bounds** the true MSM column for every concrete reference whose
quantization matches the trie path. (It is a relaxation — each occurrence of a
shared `y_j` is minimized independently — so the column is `≤` the true column,
not necessarily achievable; that is exactly what soundness needs.)

Extreme quantization bins extend to ±∞ (everything ≤ `min_value` → bin 0;
everything ≥ `max_value` → last bin). Every subtraction branch is reached only
with finite endpoints; ±∞ routes structurally to the penalty-0 / overlap branch.

## 4. Pruning and exactness

* **Sound pruning (no false negatives).** A node's column minimum lower-bounds the
  true MSM distance of *every* reference reachable below it (any DP path to a
  deeper final crosses the column, and later MSM costs are non-negative). A
  subtree whose bound exceeds `τ` (or the running k-th best) is safely skipped.
* **Exact verification (no false positives).** At each surviving final, the
  candidate is re-scored against the stored full-precision original with
  `MsmConfig::distance`; only genuine matches are emitted.

## 5. Machine-checked verification

### Coq/Rocq (`docs/verification/msm/theories/Indexing/`, Rocq 9.1, axiom-free)

A bin interval is modelled as `Qitv = (option Q * option Q)` (`None` = unbounded),
keeping every subtraction inside `Q`.

| File | Key results | Status |
|------|-------------|--------|
| `IntervalCost.v` | `interval_dist_le_move`, `c_func_merge_lb_le`, `c_func_split_lb_le` (**admissibility** of all three bounds); `interval_dist_tight`, `c_func_merge_lb_tight`, `c_func_split_lb_tight` (**exactness** of all three — each bound is attained by a concrete value in its interval box) | trusted, axiom-free |
| `QuantizationBounds.v` | `quantize_in_bin_bounds`: `v ∈ bin_bounds(quantize v)` (executable uniform binning; extreme bins → unbounded endpoints), replacing the former placeholder quantizer | trusted, axiom-free |
| `IntervalColumn.v` | `interval_cell_le_matrix` (**column admissibility**: `interval_cell ≤ msm_matrix_cell` by `i+j` strong induction); `lb_prune_sound_msm` (final-column no-false-negative); `column_lb_le_deeper` (**subtree** bound over deeper finals, via column-min monotonicity) | trusted, axiom-free |

`Print Assumptions lb_prune_sound_msm` / `column_lb_le_deeper` / `quantize_in_bin_bounds`
report *Closed under the global context* (zero axioms). The substrate is the
existing `Metric/Symmetry.v` `msm_matrix_cell` (orientation-neutral, index-
addressable DP cell with its four recurrence lemmas) and the `Core/CFunctionBounds.v`
`is_between` case lemmas.

**Split-bound exactness** (`c_func_split_lb_tight`) shows the bound is attained by
a concrete `(av, cv)` in the `a_iv × c_iv` box — the forced-above/below cases hit
the boundary pair, the overlap case a between-pair — so the closed form equals the
true minimum, not merely a lower bound. It is cross-validated by the executable
property test `split_lb_is_min_over_box` in `msm_interval.rs` (which brute-forces
the minimum over a fine grid and checks equality). The transducer's *soundness*
depends only on `c_func_split_lb_le` and the column admissibility built on it;
exactness additionally certifies the bound is the tightest possible.

### TLA+ (`docs/verification/tla/MsmTrieSearch.tla`, TLC-checked)

Models the non-empty trie walk over an abstracted finite distance/bound table.
The Coq admissibility theorem is assumed as a table constraint
(`ASSUME AdmissibleTable` / `MonotoneDownEdges`); TLC verifies the *operational*
consequences of the traversal: `NoFalsePositives`, `NoMissedMatches`,
`PruneSound`, and `EventuallyTerminates`. Nondeterministic node order makes these
order-independent. `Model checking completed. No error has been found.`

### Rust tests

* In-module (`msm_interval.rs`, `msm_transducer.rs`): closed-form-vs-brute-force
  for each bound, `degenerate_bins_reproduce_scalar_dp`,
  `interval_column_lower_bounds_concrete`, `prop_range_exact`, `prop_knn_exact`.
* Integration (`tests/msm_transducer_tests.rs`): public-API smoke, empty/single/
  length-mismatch, empty-query/empty-reference semantics, `k > len`, threshold
  boundary, out-of-range (±∞ bins), collisions,
  `transducer_is_exact_and_supersets_hybrid`, concurrent queries,
  `prop_range_exact_with_outliers`.
* `bin_bounds` soundness: `prop_bin_bounds_contains_quantized_value` in
  `encoding.rs` (Rust mirror of `quantize_in_bin_bounds`).

## 6. Reproduce the proofs

```sh
# Coq:
scripts/verify-formal.sh coq-file standard docs/verification/msm/theories/Indexing/IntervalCost.v
scripts/verify-formal.sh coq-file standard docs/verification/msm/theories/Indexing/QuantizationBounds.v
scripts/verify-formal.sh coq-file heavy    docs/verification/msm/theories/Indexing/IntervalColumn.v
# TLA+:
scripts/verify-formal.sh tla        # discovers MsmTrieSearch.cfg
# gates:
scripts/verify-formal.sh trusted
scripts/verify-formal.sh coq-trusted
scripts/verify-formal.sh audit-vacuous
```

Loom is intentionally not used: the transducer is immutable after construction
with no interior mutability or atomics; concurrency is exercised by the
`concurrent_queries_are_consistent` integration test (read-only sharing).
