# TWED source metric contract and anchored transfer

**Epic:** `metric-automata-cbc-theory`, task `cbc-06-twed-source-contract`.
**Status:** source, ideal-model, and implementation boundary contract. The sample-space product metric and anchor transfer are checked in Rocq; sequence-level metricity and binary64 correspondence remain open.

## Primary sources and scope

The accessible author reports differ at the one-empty boundary. Their metric claims must be read against their own equations:

| Source | Printed recurrence and boundary | Metric statement | Consequence here |
|---|---|---|---|
| [Marteau, 2007 submission, v5, Eq. (9), PDF p. 15](https://arxiv.org/pdf/cs/0703033v5#page=15) | Interior segment edits; $`D(0,0)=0`$; every positive-axis cell is infinite. [Figure 2](https://arxiv.org/pdf/cs/0703033v5#page=19) repeats the infinite initialization. The v5 revision is dated March 2008. | Proposition 1 claims a distance on finite series, with its proof deferred to a later report. | The Rocq forced-anchor transfer models this recurrence, restricted to finite distances between anchored series. |
| [Marteau, 2008 technical report, v5, Eq. (1) and initialization, PDF pp. 3–4](https://arxiv.org/pdf/0802.3522v5#page=3) | Same three interior edits, but $`D(i,0)`$ and $`D(0,j)`$ are cumulative sums of sample distances **without** gap penalty $`\lambda`$. | [Proposition 1](https://arxiv.org/pdf/0802.3522v5#page=4) claims metricity for $`\lambda\ge0`$ and positive stiffness. | This is not the library recurrence. Its triangle claim is false as printed for positive $`\lambda`$; a strict-origin counterexample appears below. |
| [Marteau, 2009 journal article, DOI 10.1109/TPAMI.2008.76](https://doi.org/10.1109/TPAMI.2008.76) | Published TWED study; the accessible reports above supply the exact equations audited here. | The [article abstract](https://pubmed.ncbi.nlm.nih.gov/19110495/) describes a metric useful for retrieval. | The abstract alone does not settle the boundary and domain obligations. |

The [2007 submission's definition](https://arxiv.org/pdf/cs/0703033v5#page=8) includes the empty series in its stated set `U`, while its positive-axis initialization gives an infinite empty-to-nonempty distance. Thus its Proposition 1 cannot literally define a finite real-valued metric on all of `U`. The candidate source theorem needed here concerns the common-anchor, nonempty image below. The 2008 report assumes timestamps increase *within* each series but does not require the first timestamp to follow its conventional zero predecessor; its [identity proof](https://arxiv.org/pdf/0802.3522v5#page=5) implicitly uses positivity of that first step. Neither published proposition is imported as a Rocq axiom.

For the scalar library specialization fix one origin $`t_0`$, one physical time unit, stiffness $`\nu>0`$, gap penalty $`\lambda\ge0`$, ground metric $`\rho(u,v)=|u-v|`$, and finite sample/time values. A physical series $`[(u_1,t_1),\ldots,(u_m,t_m)]`$ has $`t_0<t_1<\cdots<t_m`$. Empty physical series are allowed in the mathematical transfer; the current Rust constructor excludes them. These conditions describe ideal real arithmetic and do not assert binary64 metricity.

The source proof concerns a metric on the sample/time product. The local specialization is

```math
d((u,t),(v,s))=\rho(u,v)+\nu|t-s|.
```

With $`\nu>0`$ and scalar absolute ground distance, this is a metric: nonnegative terms are symmetric, zero forces equality in both coordinates, and the triangle inequalities add. [TwedSourceMetric.v](../verification/twed/theories/Metric/TwedSourceMetric.v) checks all five product sample metric laws. This does not prove any sequence-level metric law.

## Library recurrence and anchor map

Let $`a_0=b_0=(0,t_0)`$ be the shared sentinel. The ideal recurrence is

```math
\begin{aligned}
e_i &= d(a_i,a_{i-1})+\lambda,\\
f_j &= d(b_j,b_{j-1})+\lambda,\\
v_{ij} &= d(a_i,b_j)+d(a_{i-1},b_{j-1}),\\
D(0,0)&=0,\\
D(i,0)&=D(i-1,0)+e_i,\\
D(0,j)&=D(0,j-1)+f_j,\\
D(i,j)&=\min\{D(i-1,j)+e_i,\;D(i,j-1)+f_j,\;D(i-1,j-1)+v_{ij}\}.
\end{aligned}
```

The [Rust scorer](../../src/time_series/timestamped_twed.rs) constructs both axes with `delete_cost`, which adds $`\lambda`$, and uses two rolling vectors for the interior. Its `match_cost` compares current **and** predecessor pairs. Binary64 operation order, overflow behavior, and cutoff tags require separate machine correspondence. [ORC §11.5](ordered-residual-calculus.md#115-twed-including-explicit-physical-timestamps) states the ideal recurrence and rolling-column state. The [unit-grid kernel](../../src/time_series/kernels/twed.rs) is a separate index-time specialization.

Map each input time to $`1+t_i-t_0`$ and prepend a common anchor $`(0,1)`$ to both series. Empty input becomes the one-anchor series. The condition $`t_1>t_0`$ makes the mapped timestamps strictly increasing.

The **2007** source's unavailable positive axes force the anchors to match at zero cost. Afterward, its deletion, insertion, and match charges equal the corresponding $`e_i`$, $`f_j`$, and $`v_{ij}`$: common shifts preserve time differences. The source grid at $`(i+1,j+1)`$ equals $`D(i,j)`$. In particular, comparing an anchor-only sequence with a longer anchored sequence produces the cumulative one-empty boundary **with** $`\lambda`$ per edit. [TimestampedMetricTransfer.v](../verification/twed/theories/Metric/TimestampedMetricTransfer.v) proves recurrence-grid correspondence, physical-charge equalities, strict timestamp admission, and anchor injectivity.

The transfer represents source-infinite positive axes as absent `option` values. It proves the anchored grid is the unique grid satisfying those boundaries and the source recurrence. Its final metric-pullback theorem is **conditional** on four source-distance metric premises restricted to the common-anchor image and an explicit equality between the chosen cumulative recurrence score and the source-distance pullback. Empty physical input maps to the anchor-only source series; the empty source series is outside the premise domain, where the literal 2007 source has infinite distances. The source metric laws and the recurrence-score equality must still be mechanized and instantiated for the product metric and anchored sequences. The 2008 report's cumulative boundary cannot be substituted: it omits the gap penalty, while the library and anchored 2007 grid include it.

| Correspondence | Source / implementation fact | Verification status |
|---|---|---|
| Ground metric | $`d_\nu`$ specializes the source metric on combined value/time samples; $`\nu>0`$ distinguishes timestamps. | Product laws checked in `TwedSourceMetric.v`. |
| Fixed parameters | All three operands of a metric law use the same $`\nu>0`$ and $`\lambda\ge0`$. | Rust configuration validates finite binary64 inputs; real-to-machine refinement remains open. |
| Timestamp domain | The mapped first time must follow the common anchor, then increase strictly. | `anchor_series_has_strict_times` checked. Raw Rust permits $`t_1=t_0`$; its strict wrapper in the current working tree excludes it. |
| Unit and origin | Both operands use the same `TimestampUnit` and origin; no implicit seconds/milliseconds conversion. | Rust rejects mixed operands. A whole-container configuration invariant remains separate. |
| Boundary and interior | Rust charges $`d_\nu+\lambda`$ for each one-empty edit and compares current and predecessor samples on matches. | Anchor-grid and charge-shift lemmas checked; recurrence-score/source-grid equality and source-to-Rust loop refinement remain open. |
| Cutoff and numeric result | The ideal distance is finite on the anchored image; Rust uses binary64, `WeightedCost`, cutoffs, and complete/incomplete outcomes. | Exact-score refinement and fail-closed result theorem remain open. |

## Counterexamples to overbroad metric claims

### Origin-equal raw operands lose separation

The raw Rust constructor admits a first timestamp equal to the origin. With $`\lambda=0`$, origin zero, stiffness one, and

```math
X=((1,1)),\qquad Y=((0,0),(1,1)),
```

the cumulative recurrence assigns zero to unequal series. The first sample of $`Y`$ deletes at zero cost, and the remaining samples match at zero cost. The transfer proof checks distinctness and the complete 1-by-2 score. The [raw Rust constructor](../../src/time_series/timestamped_twed.rs) admits $`Y`$; `StrictOriginTimestampedSeries` in the current working tree excludes it, but does not by itself prove machine metricity. The Rocq generic control shows the same defect for vector-valued samples whenever the ground self-distance is zero.

### The 2008 printed boundary loses triangle for positive gap penalty

Use the 2008 report's **printed** one-empty initialization, origin zero, $`\nu=1`$, $`\lambda=3`$, scalar absolute ground distance, and strictly positive timestamps. Let $`A=[(0,1)]`$, $`B=[]`$, and $`C=[(1,1),(1,2)]`$, with each tuple ordered `(value,time)`. Its boundary equations give $`D_{08}(A,B)=1`$ and $`D_{08}(B,C)=2+1=3`$; no $`\lambda`$ appears on those axes. The first samples match for cost $`1`$; inserting the second sample of $`C`$ costs $`1+3=4`$. The other final alternatives cost $`6`$ and $`7`$, so

| $`D_{08}(A_{1:i},C_{1:j})`$ | $`j=0`$ | $`j=1`$ | $`j=2`$ |
|---|---:|---:|---:|
| $`i=0`$ | 0 | 2 | 3 |
| $`i=1`$ | 1 | 1 | 5 |

```math
D_{08}(A,C)=5>1+3=D_{08}(A,B)+D_{08}(B,C).
```

Every nonempty series in this example satisfies the strict-origin condition. This contradicts [Proposition 1 as printed in the 2008 report](https://arxiv.org/pdf/0802.3522v5#page=4) for $`\lambda=3`$ and is independent of the origin-equal separation defect. The library's cumulative boundary includes $`\lambda`$, so this is **not** a counterexample to the library recurrence. The 2008 proposition cannot be imported as a black-box proof for the library.

## Local proof obligations

| Obligation | Required construction | Failure to exclude |
|---|---|---|
| Product sample metric | Derive value/time laws from $`\rho`$ and $`\nu>0`$ | Rounded positive time cost confused with ideal zero |
| Source alignment semantics | Define valid paths on the common-anchor image with predecessor context and 2007 infinite axes | Path using unavailable boundary; infinite distance to source empty |
| Recurrence equivalence | Show path/DP equality both ways and minimum attainment | Altered anchor or missing predecessor term |
| Nonnegative and reflexive | Sum nonnegative local costs and use diagonal zero matches | Negative stiffness or gap |
| Symmetry | Reverse every edit with correct context and boundaries | Directional penalty |
| Separation | Strict timestamps after the anchor prohibit zero-cost insertion/deletion | Origin-equal zero-cost deletion |
| Triangle | Establish all nine final-edit pairs and their boundary cases for the **same** 2007 recurrence | Treating predecessor cost as memoryless; importing the 2008 counterexample's false proposition |
| Transfer closure | Instantiate `strict_origin_pullback_is_metric` with image-restricted source metric laws and recurrence-score correspondence | Source metricity and the score equality left as `Context` premises |
| Executable refinement | Relate the ideal grid to rolling binary64 columns, saturation, cutoff and incomplete outcomes | Calling rounded scores a real metric without a machine theorem |

The 2008 report gives a nine-case triangle argument, but its printed boundary differs and its positive-gap proposition has the counterexample above. [TwedPrintedBoundaryCounterexample.v](../verification/twed/theories/Metric/TwedPrintedBoundaryCounterexample.v) checks the stated 1-by-2 grid and triangle failure in exact natural arithmetic. The report is a proof guide only. A Rocq proof must reconstruct all cases for the 2007 anchored recurrence with predecessor indices and base cases. A prose statement that edits compose does not discharge this work.

[TwedSourceAlignments.v](../verification/twed/theories/Metric/TwedSourceAlignments.v)
now defines finite 2007-style paths in an exact-natural fragment. It checks
the three last-step cases, existence for finite anchored inputs, the absence
of positive-axis finite paths, and rejection of a missing predecessor.
Path-to-DP cost equality and the ideal-real sequence metric laws remain open.

## Status

- **Checked:** product sample metric, anchored recurrence uniqueness, translated charges, injective domain map, strict-wrapper admission, exact-natural finite alignment existence/decomposition, and raw-domain counterexample. The 2008 triangle counterexample is checked against the printed equations in exact natural arithmetic.
- **Open for ideal metricity:** local source TWED metric theorem on the common-anchor image under matching boundary/domain/parameter conventions. Neither published Proposition 1 discharges it unchanged.
- **Open for executable qualification:** machine-score and source-to-model correspondence for the named Rust operation.

`Print Assumptions` of the ideal-real transfer and product-metric theorems
shows Rocq Stdlib's classical Dedekind-real and functional-extensionality
axioms. They are recorded in the [assumption ledger](../verification/ASSUMPTIONS.tsv);
no project-specific TWED metric proposition is an axiom. The exact-natural
2008 boundary counterexample closes under the global context.

The common anchor is a proof representation of the existing recurrence; it requires no per-automaton snapshot or additional runtime state.
