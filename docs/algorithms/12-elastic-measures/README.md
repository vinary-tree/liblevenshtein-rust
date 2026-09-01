# Elastic measures over a prefix-shared quantized trie

This chapter explains the generic algorithm behind exact elastic-distance range
and nearest-neighbour search. It complements the design contract in
[`docs/design/elastic-kernels.md`](../../design/elastic-kernels.md) with a
worked, literate implementation view.

## 1. Problem statement

Given a database $`T=\{t_1,\ldots,t_n\}`$ of real-valued series, a query
$`q`$, and threshold $`\tau`$, range search returns

```math
R(q,\tau)=\{(i,D(q,t_i))\mid D(q,t_i)\le\tau\}.
```

Flat evaluation repeats the same DP prefix for every candidate. A trie stores
quantized target prefixes once. The walker evaluates a relaxed DP column once
per trie edge, prunes impossible subtrees, and exact-scores surviving leaves.

The search remains exact even though the trie stores bins rather than original
samples because originals are retained separately and are the only source of
reported distances.

## 2. Geometry of a quantized edge

For a scalar query sample $`v`$ and bin $`[\ell,h]`$, the exact minimum
absolute deviation is

```math
\operatorname{dist}(v,[\ell,h])=\max(0,\ell-v,v-h).
```

For two bins $`A=[\ell_A,h_A]`$ and $`B=[\ell_B,h_B]`$, the exact minimum
pairwise deviation is

```math
\operatorname{gap}(A,B)=\max(0,\ell_A-h_B,\ell_B-h_A).
```

The second identity is symmetric, non-negative, zero exactly for intersecting
closed intervals, and reduces to $`|a-b|`$ for point bins. These properties
are proved in Verus/SMT and mirrored by a 2,000-case property test.

## 3. Prefix-amortized columns

Suppose candidates `t₁ = [2, 5, 8]` and `t₂ = [2, 5, 9]` quantize to keys with
the same first two bytes. A flat DP computes the columns for `[2]` and `[2,5]`
twice. The trie walker computes each once:

```text
root
 └─ bin(2)       column C₁ computed once
     └─ bin(5)   column C₂ computed once
         ├─ bin(8) → exact-score t₁
         └─ bin(9) → exact-score t₂
```

The convenience path keeps one column per live explicit DFS frame, so its
memory is $`\mathcal{O}(mL)`$ for query length $`m`$ and maximum live trie
depth $`L`$. The strict bounded path instead stores a canonical sparse
frontier behind one `TemporalStateId` per frame and uses two query-width arrays
as shared transition scratch. Each edge transition is $`\mathcal{O}(m)`$ in
the dense fallback or $`\mathcal{O}(w)`$ for a live frontier/band width $`w`$.
Both paths are iterative and therefore process-stack safe.

## 4. Exact range algorithm

### 4.1 Preconditions

The kernel must provide K1 interval admissibility, K2 inflation, K3 exact
rescoring, and K4 candidate-bound coherence. The inclusive comparison is always
delegated to the kernel's `CostMonoid::within`.

### 4.2 Algorithm

```text
ALGORITHM ELASTIC-RANGE(index, kernel, query, cutoff)
  plan ← kernel.plan(query)
  if query cannot enter interval arithmetic then
      return EXACT-SCAN(index, kernel, query, plan, cutoff)

  width ← kernel.column_len(|query|), or return empty on overflow
  columns[0] ← [TOP; width]
  output ← empty

  PROCEDURE WALK(node, depth, carry)
    finalRow ← kernel.final_row(|query|)
    finalGate ← depth = 0 OR within(columns[depth][finalRow], cutoff)

    if node is final AND finalGate then
      for candidate in node.collisionBucket do
        leafBound ← kernel.candidate_lower_bound(query, candidate, plan)
        if within(leafBound, cutoff) then
          exact ← kernel.exact_with_cutoff(query, candidate, cutoff)
          if exact exists AND within(exact, cutoff) then
            output.append(candidate.id, exact)

    for (encodedBin, child) in node.edges do
      interval ← quantization.bounds(encodedBin)
      prefixBound ← kernel.prefix_lower_bound(
          query, interval, carry, depth + 1, plan)
      if NOT within(prefixBound, cutoff) then continue
      ensure columns[depth + 1] exists
      (bound, nextCarry) ← kernel.step_column(
          columns[depth], query, interval, carry,
          depth + 1, plan, columns[depth + 1])
      if within(bound, cutoff) then
        WALK(child, depth + 1, nextCarry)

  WALK(index.root, 0, none)
  return stableMinimumPerIdentifier(output)
```

### 4.3 Proof sketch

If a child is skipped, K1 says its bound is no greater than every concrete
column it represents. K2 says extending those paths cannot cross downward into
the cutoff. Therefore the subtree contains no result. At a visited final, K4
may reject only an out-of-range exact score. K3 makes every emitted score exact.
Together these prove both no false negatives and no false positives.

## 5. Exact kNN algorithm

Replace the fixed threshold by the largest score in a bounded result heap. The
node queue is a min-heap under `CostMonoid::compare`; the result queue is a
max-heap under the same order.

```text
ALGORITHM ELASTIC-KNN(index, kernel, query, k)
  queue ← {(ZERO, root)}
  best ← empty bounded max-heap
  cutoff ← TOP

  while queue is not empty do
    current ← queue.popMinimum()
    if |best| = k AND current.bound is outside cutoff then break
    exact-score current finals through K4 then K3
    update cutoff from best.maximum when |best| = k
    compute each child bound and enqueue only when it is within cutoff

  return best sorted by (cost, stable discovery sequence)
```

When the loop stops, the popped node was the least queued bound. Every remaining
bound is therefore greater, and K1 implies every unseen exact descendant is
greater than the current kth exact score.

## 6. MSM instantiation

MSM has three operations:

| Operation | Scalar step |
|---|---|
| Move | $`\lvert x_i-y_j\rvert`$ |
| Merge | $`C(x_i,x_{i-1},y_j)`$ |
| Split | $`C(y_j,x_i,y_{j-1})`$ |

where $`C`$ charges constant $`c`$ when its first argument lies between the
other two and otherwise adds the nearer deviation. `MsmKernel` carries the
previous target bin so the interval recurrence can lower-bound the split term.
It delegates directly to the previously tested MSM column code; extraction did
not reimplement the recurrence.

## 7. ERP instantiation

Edit distance with Real Penalty (ERP) fixes a real gap value $`g`$. Its DP
has match, delete, and insert predecessors. The boundary row and column are
running gap-mass sums, not unit edit counts.

### 7.1 Literate exact DP

**Purpose.** Compute exact ERP while retaining only the shorter matrix axis.

**Invariant.** Before processing $`x_i`$, `previous[j]` equals the exact ERP
distance between $`x_{1..i-1}`$ and $`y_{1..j}`$.

```text
ALGORITHM ERP-DISTANCE(x, y, g, cutoff)
  if y is longer than x then swap x and y
  previous[0] ← 0
  for j = 1..|y| do
    previous[j] ← previous[j-1] + |y[j]-g|

  for each x[i] do
    current[0] ← previous[0] + |x[i]-g|
    rowMinimum ← current[0]
    for j = 1..|y| do
      match  ← previous[j-1] + |x[i]-y[j]|
      delete ← previous[j]   + |x[i]-g|
      insert ← current[j-1]  + |y[j]-g|
      current[j] ← min(match, delete, insert)
      rowMinimum ← min(rowMinimum, current[j])
    if rowMinimum exceeds cutoff then return none
    swap(previous, current)
  return previous[|y|] when it is within cutoff
```

Early abandonment is sound because every path to the final cell crosses the
completed row and every later edge cost is non-negative.

### 7.2 Interval column

For target bin $`B_j`$, replace the match and insertion leaves by
$`\operatorname{dist}(x_i,B_j)`$ and
$`\operatorname{dist}(g,B_j)`$. The deletion leaf stays exact. The minimum
cell in the completed column is the subtree bound.

Point intervals $`B_j=[y_j,y_j]`$ satisfy
$`\operatorname{dist}(v,B_j)=\lvert v-y_j\rvert`$, so they reproduce the
scalar DP exactly. This is stronger than admissibility and prevents a
degenerate always-zero implementation from passing the gate.

### 7.3 Candidate potential

Let $`\Phi_g(x)=\sum_i\lvert x_i-g\rvert`$. Every alignment edit changes
$`\Phi_g`$ by at most its own ERP cost, so:

```math
\big\lvert\Phi_g(x)-\Phi_g(y)\big\rvert\le D_{\mathrm{ERP}}(x,y).
```

The walker evaluates this $`\mathcal{O}(m+n)`$ bound before the exact DP.

### 7.4 Public example

```rust
use liblevenshtein::time_series::{ErpConfig, ErpTransducer, QuantizationConfig};

let references = vec![vec![1.0, 2.0], vec![1.0, 0.0, 2.0], vec![8.0]];
let index = ErpTransducer::from_series(
    QuantizationConfig::for_u8(-10.0, 10.0),
    ErpConfig::new(0.0),
    &references,
);

// Inserting the gap value 0 costs zero, so the first two references tie.
let exact = index.search_range(&[1.0, 2.0], 0.0);
assert_eq!(exact.len(), 2);
assert!(exact.iter().all(|(_, distance)| *distance == 0.0));
```

## 8. TWED instantiation

Time Warp Edit Distance (TWED) edits adjacent segments. The crate uses
unit-spaced timestamps, a shared zero sentinel, temporal stiffness $`\nu`$,
and deletion penalty $`\lambda`$. The three local alternatives are a query
segment deletion, a segment match, and a target segment deletion.

### 8.1 Literate exact DP

**Purpose.** Compute exact unit-spaced TWED in quadratic time while retaining
only two rows.

**Invariant.** Before processing query sample $`x_i`$, `previous[j]` equals
the exact value $`D(i-1,j)`$. During the row, `current[j-1]` equals
$`D(i,j-1)`$.

```text
ALGORITHM TWED-DISTANCE(x, y, nu, lambda, cutoff)
    reject non-finite samples or a negative/NaN cutoff
    store the shorter series on the row axis
    previous[0] <- 0
    for j <- 1 through length(y)
        previous[j] <- previous[j-1]
            + ABS(y[j] - y[j-1]) + nu + lambda

    for i <- 1 through length(x)
        deleteX <- ABS(x[i] - x[i-1]) + nu + lambda
        current[0] <- previous[0] + deleteX
        rowMinimum <- current[0]

        for j <- 1 through length(y)
            deleteY <- ABS(y[j] - y[j-1]) + nu + lambda
            match <- ABS(x[i] - y[j]) + ABS(x[i-1] - y[j-1])
                     + 2 * nu * ABS(i-j)
            current[j] <- MINIMUM(
                previous[j]   + deleteX,
                previous[j-1] + match,
                current[j-1]  + deleteY)
            rowMinimum <- MINIMUM(rowMinimum, current[j])

        if rowMinimum > cutoff then return NO-RESULT
        swap(previous, current)

    return previous[length(y)] only when it is within cutoff
```

The sentinel values in the pseudocode are $`x_0=y_0=0`$. Every local cost
is non-negative, so a completed row above the cutoff cannot recover later.

### 8.2 Carry-aware interval column

The current target edge gives interval $`I_j`$; the carry gives
$`I_{j-1}`$. The exact box minima are:

```math
\begin{aligned}
\underline{\mu}(i,j)&=
\operatorname{dist}(x_i,I_j)+
\operatorname{dist}(x_{i-1},I_{j-1})+
2\nu\lvert i-j\rvert,\\
\underline{\delta}_y(j)&=
\operatorname{gap}(I_{j-1},I_j)+\nu+\lambda.
\end{aligned}
```

Each free interval variable occurs in its own absolute-value term, which makes
the match minimum separable. Singleton intervals reproduce scalar leaves.
Applying the same additive/minimum recurrence to lower-bounding predecessors
and leaves yields a lower-bounding column.

### 8.3 Candidate bound and metric domain

Every length-changing edit pays $`\lambda`$, so:

```math
\lvert m-n\rvert\lambda\le D_{\mathrm{TWED}}(x,y).
```

The unrestricted family includes $`\nu=0`$ and is not uniformly metric.
Use `MetricTwedConfig::try_new` to validate finite $`\nu>0`$ and finite
$`\lambda\ge0`$; only that wrapper implements `MetricElasticKernel`.

### 8.4 Public example

```rust
use liblevenshtein::time_series::{
    MetricTwedConfig, MetricTwedTransducer, QuantizationConfig,
};

let references = vec![vec![0.0, 1.0, 2.0], vec![0.0, 2.0, 3.0]];
let kernel = MetricTwedConfig::try_new(0.5, 1.0).unwrap();
let index = MetricTwedTransducer::from_series(
    QuantizationConfig::for_u8(0.0, 3.0),
    kernel,
    &references,
);
assert_eq!(index.search_range(&[0.0, 1.0, 2.0], 0.0), vec![(0, 0.0)]);
```

The unrestricted `TwedConfig::new(0.0, 0.0)` remains available for studying
the documented degeneracy, but generic metric-dependent code cannot accept it.

## 9. Discrete Fréchet instantiation

Discrete Fréchet minimizes the maximum point distance along an
order-preserving coupling. Path alternatives use `min`, but extending one path
uses `max`; the production kernel therefore selects `BottleneckCost`.

### 9.1 Literate exact DP

**Purpose.** Compute Eiter and Mannila's Table 1 coupling distance with two
rows.

**Invariant.** Before processing $`x_i`$, `previous[j]` is the minimum
bottleneck among all couplings from $`(1,1)`$ to $`(i-1,j)`$.

```text
ALGORITHM DISCRETE-FRECHET(x, y, cutoff)
  if both sequences are empty then return 0
  if exactly one is empty then return TOP
  if y is longer than x then swap x and y

  previous[1] ← |x[1]-y[1]|
  for j = 2..|y| do
    previous[j] ← max(previous[j-1], |x[1]-y[j]|)

  for i = 2..|x| do
    current[1] ← max(previous[1], |x[i]-y[1]|)
    rowMinimum ← current[1]
    for j = 2..|y| do
      predecessor ← min(previous[j], previous[j-1], current[j-1])
      current[j] ← max(predecessor, |x[i]-y[j]|)
      rowMinimum ← min(rowMinimum, current[j])
    if rowMinimum exceeds cutoff then return none
    swap(previous, current)
  return previous[|y|] when it is within cutoff
```

Every future path extends a completed-row cell with `max`; therefore a row
minimum above the cutoff cannot recover.

### 9.2 Interval column and tightness

For a trie edge representing $`B_j=[\ell_j,h_j]`$, replace each link leaf
with $`\operatorname{dist}(x_i,B_j)`$. The recurrence itself is unchanged.
Minimum and maximum are monotone, so the relaxed column is cellwise admissible.
Point bins recover the scalar link and the entire scalar column exactly.

### 9.3 Candidate cascade

Every coupling contains the first and last links, and every query sample is
paired with at least one candidate sample. Hence:

```math
\max\!\left(
  \lvert x_1-y_1\rvert,
  \lvert x_m-y_n\rvert,
  \max_i\min_j\lvert x_i-y_j\rvert
\right)
\le D_{\mathrm{dF}}(x,y).
```

The first two terms are constant-time after boundary checks. The third sorts
the candidate and uses binary-search nearest neighbours. The walker evaluates
their maximum before exact DP.

### 9.4 Public example

```rust
use liblevenshtein::time_series::{
    FrechetConfig, FrechetTransducer, QuantizationConfig,
};

let references = vec![
    vec![1.0, 2.0, 3.0],
    vec![1.0, 1.0, 2.0, 3.0],
    vec![8.0, 9.0],
];
let index = FrechetTransducer::from_series(
    QuantizationConfig::for_u8(-10.0, 10.0),
    FrechetConfig::new(),
    &references,
);

// Consecutive stutters do not change the represented polygonal curve.
let exact = index.search_range(&[1.0, 2.0, 3.0], 0.0);
assert_eq!(exact.len(), 2);
assert!(exact.iter().all(|(_, distance)| *distance == 0.0));
```

Raw-vector identity is modulo run-length collapse. The paper, derivations, and
formal correspondence are explained in the
[research analysis](../../research/frechet/PAPER_SUMMARY.md).

## 10. Banded DTW instantiation

Dynamic time warping (DTW) aligns samples by monotone horizontal, vertical,
and diagonal steps. This implementation requires a symmetric Sakoe–Chiba
half-width $`w`$; a cell is live exactly when $`\lvert i-j\rvert\le w`$.
It accumulates squared deviations and exposes their square root publicly.

### 10.1 Literate exact banded DP

**Purpose.** Compute exact DTW for the caller-selected band, returning `TOP`
when no pinned path reaches the endpoint.

**Invariant.** Before row $`i`$, `previous[j]` is the exact squared cost for
$`(i-1,j)`$ inside the band and `TOP` outside it. `current[j-1]` is the exact
left predecessor for the current row.

```text
ALGORITHM BANDED-DTW-SQUARED(x, y, band, cutoffSquared)
  if either series is non-finite then return none
  if both are empty then return 0
  if exactly one is empty OR abs(length(x)-length(y)) > band then return TOP
  orient y as the shorter stored row
  previous ← [TOP; |y|+1]; previous[0] ← 0

  for i from 1 through |x| do
    current ← [TOP; |y|+1]
    start ← max(1, i-band)
    end ← min(|y|, i+band)
    for j from start through end do
      local ← square(x[i]-y[j])
      current[j] ← local + min(previous[j-1], previous[j], current[j-1])
    if min(current[start..end]) exceeds cutoffSquared then return none
    swap(previous, current)

  return previous[|y|] if it is within cutoffSquared
```

The public method returns the square root. The early-row cutoff is sound
because every future path extends a current cell by non-negative local costs.

### 10.2 Monotonic-deque LB_Keogh plan

For target position $`j`$, the query envelope is the minimum and maximum of
query samples reachable within $`w`$. One increasing deque computes all
window minima and one decreasing deque computes all maxima. Each query index
enters and leaves each deque once.

```text
ALGORITHM BUILD-KEOGH-PLAN(query, band)
  minima, maxima ← empty monotonic deques of query indices
  for center from 0 through |query|-1 do
    append newly reachable right-edge indices to both deques
    remove indices left of center-band
    lower[center] ← query[minima.front]
    upper[center] ← query[maxima.front]
  build suffix minima and maxima for reachable positions beyond query tail
  return lower, upper, suffix extrema
```

This preprocessing is $`\mathcal{O}(m)`$ regardless of the numeric band
value and allocates only $`\mathcal{O}(m)`$ query metadata.

### 10.3 Prefix gate before band column

At trie depth $`j`$, let $`B_j`$ be the target bin and $`E_j`$ the query
envelope. The carry stores

```math
P_j=P_{j-1}+\operatorname{gap}(B_j,E_j)^2.
```

`prefix_lower_bound` computes $`P_j`$ in constant time. The walker compares
it with the range cutoff or current kth exact cost before growing the column
buffer. Only a surviving edge pays for at most $`2w+1`$ cells. At a final,
full-series LB_Keogh is the K4 gate before exact DP.

The prefix and column bounds are independent admissible bounds. Their maximum
is returned after the column is built; neither substitutes for exact scoring.

![The live band and the prefix-before-column cascade](../../diagrams/time-series/sakoe-chiba-band.svg)

### 10.4 Public example and metric warning

```rust
use liblevenshtein::time_series::{
    DtwConfig, DtwTransducer, QuantizationConfig,
};

let references = vec![
    vec![0.0, 1.0, 2.0],
    vec![0.0, 1.0, 1.0, 2.0],
    vec![8.0, 9.0],
];
let index = DtwTransducer::from_series(
    QuantizationConfig::for_u8(0.0, 10.0),
    DtwConfig::new(1), // required inclusive half-width
    &references,
);
let within_root_distance = index.search_range(&[0.0, 1.0, 2.0], 0.25);
assert_eq!(within_root_distance.len(), 2);
assert!(within_root_distance
    .iter()
    .all(|(_, distance)| *distance <= 0.25));
```

DTW is not a metric. `DtwConfig::IS_METRIC` is `false`, and its type cannot
cross the `MetricElasticKernel` gate required by triangle-dependent indexes.
The exact trie walker is lawful because it uses lower-bound admissibility and
inflation instead. See the [source analysis](../../research/dtw/PAPER_SUMMARY.md).

## 11. Boundary examples

| Input | MSM result | Walker behavior |
|---|---:|---|
| empty vs empty | `0` | exact fallback emits the empty candidate |
| empty vs nonempty | `TOP` | no finite result; $`+\infty`$ range preserves legacy behavior |
| finite query vs NaN candidate | `TOP` | exact scorer rejects finite cutoffs |
| NaN query | `TOP` | avoids interval heap/column arithmetic and uses deterministic scan |
| quantization collision | per-original exact scores | every identifier in the bucket is rescored |
| `k = 0` | empty | returns before heap allocation |

ERP differs on empty sides: empty/nonempty distance is the finite running sum
$`\sum_i\lvert x_i-g\rvert`$. Consequently the trie root can be a valid kNN
candidate; the generic walker exact-scores it instead of assuming MSM's `TOP`.

TWED also assigns finite empty/nonempty distance. Its boundary accumulates
adjacent-sample change, stiffness, and penalty from the zero sentinel. At
$`\nu=\lambda=0`$, unequal sequences can tie at zero; this is why only the
validated wrapper carries the static metric marker.

Discrete Fréchet follows MSM's one-empty `TOP` rule but for a different reason:
an endpoint-covering coupling cannot exist when only one side has a point.
Both-empty distance is zero, and kNN omits `TOP` candidates.

Banded DTW also uses `TOP` for exactly one empty side and for an endpoint
length gap wider than its required band. Its public thresholds and results are
root-valued; its internal DP, lower bounds, and cutoffs are squared.

## 12. Testing recipe for a new kernel

1. Write an obviously correct $`\mathcal{O}(mn)`$ scalar reference DP.
2. Prove and property-test every closed-form interval step (leaf exactness).
3. Generate point bins and assert the relaxed column equals the scalar column
   exactly (degenerate-bin exactness).
4. Generate non-degenerate bins and concrete realizations, asserting K1.
5. Compare range result sets and exact scores with brute force.
6. Compare kNN sorted distance multisets with brute force, including ties.
7. Test empty, one-element, all-identical, different-length, out-of-range,
   NaN/infinity, collision, upsert, remove, and deterministic-order cases.
8. Register formal K1–K4 artifacts and add the kernel's resource guard.

Returning an always-zero interval bound may pass admissibility but fails the
degenerate-bin exactness gate; this is why both properties are mandatory.

## 13. Complexity and operational guidance

Let $`E_v`$ be visited trie edges, $`m`$ query length, $`w`$ band width,
and $`S`$ exact-scored survivors.

| Kernel shape | Traversal time | Exact verification | Reusable-column memory |
|---|---:|---:|---:|
| full column | $`\mathcal{O}(E_v m)`$ | $`S\cdot\mathcal{O}(mn)`$ worst case | $`\mathcal{O}(mL)`$ |
| TWED full column with carry | $`\mathcal{O}(E_v m)`$ | $`S\cdot\mathcal{O}(mn)`$ worst case | $`\mathcal{O}(mL)`$ plus one interval carry per depth |
| banded | $`\mathcal{O}(E_v w)`$ | kernel-dependent | $`\mathcal{O}(wL)`$ possible |
| DTW with prefix gate | $`\mathcal{O}(E_v)+\mathcal{O}(E_c w)`$, where $`E_c`$ edges pass the prefix gate | $`S\cdot\mathcal{O}(mw)`$ worst case | current implementation retains checked query-width buffers per live depth |

Quantization controls trie sharing and lower-bound tightness, never correctness.
Coarser bins share more prefixes but weaken bounds; finer bins strengthen bounds
but fragment prefixes. Measure `visited_edges`, exact candidate evaluations,
prefix prunes, columns built, column prunes, exact candidate evaluations, and
cutoff abandons rather than relying only on wall-clock time.

## 14. References

- Stefan, Athitsos, and Das, MSM, DOI
  [10.1109/TKDE.2012.88](https://doi.org/10.1109/TKDE.2012.88).
- Sakoe and Chiba, constrained DTW, DOI
  [10.1109/TASSP.1978.1163055](https://doi.org/10.1109/TASSP.1978.1163055).
- Keogh and Ratanamahatana, exact DTW indexing and LB_Keogh, DOI
  [10.1007/s10115-004-0154-9](https://doi.org/10.1007/s10115-004-0154-9).
- Eiter and Mannila, discrete Fréchet,
  [technical report CD-TR 94/64](https://www.kr.tuwien.ac.at/staff/eiter/et-archive/files/cdtr9464.pdf).
- Fréchet, continuous curve distance, DOI
  [10.1007/BF03018603](https://doi.org/10.1007/BF03018603).
- Chen and Ng, ERP, DOI
  [10.1016/B978-012088469-8.50070-X](https://doi.org/10.1016/B978-012088469-8.50070-X),
  with the [VLDB paper](https://www.vldb.org/conf/2004/RS21P2.PDF).
- Marteau, TWED, DOI
  [10.1109/TPAMI.2008.76](https://doi.org/10.1109/TPAMI.2008.76), with the
  [revised HAL manuscript](https://data.hal.science/document/hal-00135473v5).
