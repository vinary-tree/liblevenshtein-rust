# Banded dynamic time warping and LB_Keogh: source and implementation analysis

## 1. Scope and sources

**Dynamic time warping (DTW)** aligns two ordered sample sequences while
allowing either sequence to advance, pause, or advance with the other. This
implementation is deliberately narrower than the family of algorithms called
DTW: it uses scalar squared deviations, pinned endpoints, a symmetric
Sakoe–Chiba band, and a square root at the public boundary.

The band follows Sakoe and Chiba's constrained dynamic-programming treatment
in “Dynamic Programming Algorithm Optimization for Spoken Word Recognition,”
*IEEE Transactions on Acoustics, Speech, and Signal Processing* 26(1), 1978,
DOI [10.1109/TASSP.1978.1163055](https://doi.org/10.1109/TASSP.1978.1163055).
The candidate envelope follows Keogh and Ratanamahatana's “Exact Indexing of
Dynamic Time Warping,” *Knowledge and Information Systems* 7, 358–386, 2005,
DOI [10.1007/s10115-004-0154-9](https://doi.org/10.1007/s10115-004-0154-9).

“Exact” below means exact for this explicitly banded recurrence. It does not
mean that changing the band preserves the distance, nor that the band is an
approximation parameter selected internally.

## 2. Symbols and alignment geometry

Let $`x=(x_1,\ldots,x_m)`$ be the query and
$`y=(y_1,\ldots,y_n)`$ the candidate. A **warping path** is a sequence of
grid cells beginning at $`(1,1)`$, ending at $`(m,n)`$, and taking steps
from $`\{(1,0),(0,1),(1,1)\}`$. These steps preserve order but allow one
sample to align with several samples on the other side.

The required **Sakoe–Chiba band** has inclusive half-width $`w`$:

```math
\lvert i-j\rvert\le w.
```

Cells outside the band are unreachable and carry `TOP`, the extended cost
$`+\infty`$. In particular, an endpoint is unreachable when
$`\lvert m-n\rvert>w`$. The public `DtwConfig::new(w)` constructor therefore
requires $`w`$; the API has no default or unbanded constructor.

![A symmetric Sakoe–Chiba band, its live dynamic-programming cells, and the prefix-first pruning cascade](../../diagrams/time-series/sakoe-chiba-band.svg)

## 3. Exact recurrence and units

The implementation accumulates squared local deviations:

```math
C[i,j]=(x_i-y_j)^2+min\{C[i-1,j],C[i-1,j-1],C[i,j-1]\}.
```

The boundary is $`C[0,0]=0`$; all other row-zero and column-zero cells are
`TOP`. A cell is also `TOP` when $`\lvert i-j\rvert>w`$. The native kernel
cost is $`C[m,n]`$, while the public distance is

```math
D_w(x,y)=\sqrt{C[m,n]}.
```

Keeping all internal stages in squared units is load-bearing: DP columns,
LB_Keogh, cutoffs, and heap keys remain additive and comparable without a
square root per cell. `DtwTransducer` squares a public threshold on entry and
square-roots exact results on exit. Thus a public threshold of `5.0` means a
native cutoff of `25.0`, never `5.0`.

The two-row implementation uses $`\mathcal{O}(n)`$ storage after orienting
the shorter input along the stored row. It visits only live cells. For equal
lengths this is $`\mathcal{O}(m(2w+1))`$ time and
$`\mathcal{O}(m)`$ allocated storage; the live work per trie edge is
$`\mathcal{O}(2w+1)`$.

## 4. Why the band is semantic and operational

Unbanded DTW permits arbitrarily long zero-cost stutters. For example, a
constant sample can align with any number of identical samples. This has two
consequences:

1. raw vectors do not satisfy identity of indiscernibles; and
2. a trie column minimum can remain zero through a long prefix, making the
   lower bound sound but ineffective.

The band does not repair metricity, but it bounds the live wavefront and makes
length divergence observable. The same data can therefore have finite distance
under one band and `TOP` under a narrower band. The band is part of the
distance's definition, not a tuning hint.

## 5. LB_Keogh candidate bound

For target position $`j`$, define the query envelope over every query sample
reachable through the band:

```math
L_j=\min_{\lvert i-j\rvert\le w}x_i,
\qquad
U_j=\max_{\lvert i-j\rvert\le w}x_i.
```

The unavoidable deviation of candidate sample $`y_j`$ is

```math
\delta_j=
\begin{cases}
L_j-y_j & y_j<L_j,\\
y_j-U_j & y_j>U_j,\\
0 & \text{otherwise}.
\end{cases}
```

Every valid path couples $`y_j`$ to at least one query sample inside this
envelope. Therefore $`\delta_j^2`$ is no larger than at least one local cost
paid for position $`j`$, and summing once per candidate position gives

```math
\operatorname{LB}_{\mathrm{Keogh}}^2(x,y)
=\sum_{j=1}^{n}\delta_j^2
\le C[m,n].
```

The root-valued convenience function returns the square root of this sum. The
kernel retains the squared value to avoid unit conversion inside traversal.

## 6. Linear-time envelopes

A **monotonic deque** is a double-ended queue whose stored query values remain
ordered. One increasing deque exposes each window minimum; one decreasing
deque exposes each window maximum. Every query index enters and leaves each
deque at most once, so all centered envelopes are built in
$`\mathcal{O}(m)`$ time and $`\mathcal{O}(m)`$ memory.

Positions just beyond the query tail can still be reachable when the candidate
is longer. Suffix minima and maxima answer those envelope queries in constant
time without allocating memory proportional to an attacker-selected band.
Generated tests compare the deque construction with a direct window scan over
2,000 queries and bands.

## 7. Incremental interval LB_Keogh

A trie edge denotes an unknown target value in a quantization interval
$`B_j=[\ell_j,h_j]`$. Let $`E_j=[L_j,U_j]`$ be the query envelope. The exact
minimum separation between the two closed intervals is

```math
\operatorname{gap}(B_j,E_j)
=\max(0,L_j-h_j,\ell_j-U_j).
```

The cumulative prefix bound advances in constant time:

```math
P_j=P_{j-1}+\operatorname{gap}(B_j,E_j)^2.
```

Because this bound depends only on the parent prefix, current edge, and query
plan, the generic walker evaluates it before allocating or computing the
$`\mathcal{O}(w)`$ child column. A rejected edge cannot conceal an in-range
descendant: every realization inside each bin has at least the interval gap,
and every extension retains the already accumulated non-negative prefix cost.

## 8. Metric status is a code-level contract

DTW is symmetric and non-negative, but it is not a metric. Under band one, let
$`x=[0]`$, $`y=[1]`$, and $`z=[1,1]`$. Exact costs are

```math
D_1(x,y)=1,
\qquad
D_1(y,z)=0,
\qquad
D_1(x,z)=\sqrt{2}>1+0.
```

This is an executable Rust regression and an assumption-free Rocq theorem.
`DtwConfig::IS_METRIC` is `false`, and `DtwConfig` does not implement
`MetricElasticKernel`. Consequently a triangle-inequality-dependent generic
index cannot accept DTW through the checked type boundary. The quantized trie
walker remains valid because its K1–K4 proof uses admissible lower bounds and
path inflation, not metric balls.

## 9. Empty, non-finite, cutoff, and overflow behavior

- Two empty sequences have distance zero.
- Exactly one empty sequence has `TOP`: a pinned endpoint path does not exist.
- A length difference larger than $`w`$ has `TOP`.
- NaN or infinite samples are outside the exact and interval domains.
- A negative or NaN range cutoff yields no result.
- Finite overflow during squared accumulation saturates to `TOP` through the
  cost monoid.
- k-nearest-neighbour search never emits `TOP` as a neighbour.

These are API definitions for inputs beyond the papers' ordinary finite,
nonempty setting. They make adversarial behavior deterministic, but they do
not replace deployment limits on sequence length, indexed count, and request
work.

## 10. Evidence map

| Claim | Executable evidence | Independent formal evidence |
|---|---|---|
| exact banded recurrence | 2,000 optimized-vs-full-matrix cases | Rocq recurrence monotonicity; Verus and SMT cell checks |
| interval K1 and point exactness | 2,000 cellwise interval paths plus 2,000 scalar boxes | Rocq, Verus, Z3, cvc5 |
| candidate and prefix LB_Keogh | 2,000 exact-distance comparisons | Rocq prefix-sum induction; Verus/SMT first-gate lemmas |
| symmetry and non-negativity | 2,000 generated pairs | Rocq, Verus, Z3, cvc5 |
| non-metricity | fixed band-one Rust regression | executable `NotAMetric.v`; Verus/SMT squared witness |
| exact indexed range and kNN | 4,000 generated databases plus examples | generic TLC traversal with prefix-before-column invariant |
| required band | constructor and compile-fail documentation test | endpoint band-reachability theorems |

The [verification map](../../verification/README.md), [kernel design](../../design/elastic-kernels.md),
[literate algorithm](../../algorithms/12-elastic-measures/README.md), and
[security guide](../../security/resource-exhaustion.md) connect these claims to
their exact artifacts and operational controls.

## 11. References

1. H. Sakoe and S. Chiba, “Dynamic Programming Algorithm Optimization for
   Spoken Word Recognition,” *IEEE Transactions on Acoustics, Speech, and
   Signal Processing* 26(1), 43–49, 1978. DOI:
   [10.1109/TASSP.1978.1163055](https://doi.org/10.1109/TASSP.1978.1163055).
2. E. Keogh and C. A. Ratanamahatana, “Exact Indexing of Dynamic Time
   Warping,” *Knowledge and Information Systems* 7, 358–386, 2005. DOI:
   [10.1007/s10115-004-0154-9](https://doi.org/10.1007/s10115-004-0154-9).
