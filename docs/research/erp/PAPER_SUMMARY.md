[← Research index](../README.md)

# Edit distance with Real Penalty: paper summary and implementation analysis

**Source:** Lei Chen and Raymond T. Ng, “On the Marriage of Lp-norms and Edit
Distance,” *Proceedings of the 30th VLDB Conference*, pp. 792–803, 2004.
[DOI 10.1016/B978-012088469-8.50070-X](https://doi.org/10.1016/B978-012088469-8.50070-X) ·
[authoritative VLDB PDF](https://www.vldb.org/conf/2004/RS21P2.PDF)

## 1. Question and contribution

Chen and Ng seek a time-series measure that combines the local time-shift
tolerance of edit-style alignment with the triangle inequality of the
$`L_1`$ norm. Their Edit distance with Real Penalty (ERP) uses real-valued
absolute deviations for matched samples and compares every unmatched sample
to one fixed real gap value $`g`$. The paper also proposes one-dimensional
lower-bound indexing and combines it with triangle-inequality pruning.

The fixed gap is the decisive choice. Dynamic time warping prices a repeated
sample relative to alignment history; ERP prices a gap relative to $`g`$.
This makes each local alignment cost an ordinary absolute distance on the
augmented alphabet of real values plus the gap representative.

## 2. Definitions

Let $`x=(x_1,\ldots,x_m)`$ and $`y=(y_1,\ldots,y_n)`$ be finite real
sequences. Define $`D[i,j]`$ as the best ERP cost between their prefixes.
The boundary conditions are running gap costs:

```math
D[0,0]=0,\qquad
D[i,0]=D[i-1,0]+\lvert x_i-g\rvert,\qquad
D[0,j]=D[0,j-1]+\lvert y_j-g\rvert.
```

The recurrence is:

```math
D[i,j]=\min\begin{cases}
D[i-1,j-1]+\lvert x_i-y_j\rvert, & \text{match},\\
D[i-1,j]+\lvert x_i-g\rvert, & \text{delete }x_i,\\
D[i,j-1]+\lvert y_j-g\rvert. & \text{insert }y_j
\end{cases}
```

It takes $`\mathcal{O}(mn)`$ time. Retaining two rows or columns reduces
working memory to $`\mathcal{O}(\min(m,n))`$.

## 3. Worked source example

The paper uses $`Q=[0]`$, $`R=[1,2]`$, and $`S=[2,3,3]`$ with $`g=0`$.
It reports:

```math
D(Q,R)=3,\qquad D(R,S)=5,\qquad D(Q,S)=8.
```

Thus $`D(Q,S)=D(Q,R)+D(R,S)`$ in this example. A second source example
changes $`Q`$ to `[3]`, illustrating that inserting a zero gap permits local
time displacement while still charging real deviation.

## 4. Metric claim and the raw-sequence qualification

The paper's Theorem 1 proves the triangle inequality by composing optimal
alignments once each underlying element/gap cost obeys the triangle inequality.
Symmetry and non-negativity are immediate from absolute value.

On a Rust API that admits empty sequences and explicitly represents samples
equal to $`g`$, identity of indiscernibles needs a qualification:

```math
D([g],[])=0.
```

More generally, inserting or deleting any occurrence of $`g`$ costs zero.
Define the **$`g`$-quotient normal form** $`N_g(x)`$ by deleting every
sample equal to $`g`$. Then ERP is a pseudometric on raw sequences and its
identity law is:

```math
D(x,y)=0\quad\Longleftrightarrow\quad N_g(x)=N_g(y).
```

This is not a contradiction in the recurrence; it is the exact algebra induced
by making $`g`$ the gap representative. Documentation and tests must not call
raw ERP a strict metric without stating the quotient.

## 5. Lower bounds used by this implementation

The paper develops a one-dimensional lower-bound strategy. The trie kernel
uses the closely related gap-mass potential

```math
\Phi_g(x)=\sum_i\lvert x_i-g\rvert.
```

For a match edit, the reverse triangle inequality gives

```math
\big\lvert\lvert x_i-g\rvert-\lvert y_j-g\rvert\big\rvert
\le \lvert x_i-y_j\rvert.
```

For insertion or deletion it holds with equality. Summing over any alignment
and applying the triangle inequality yields the admissible candidate bound:

```math
\big\lvert\Phi_g(x)-\Phi_g(y)\big\rvert\le D(x,y).
```

Length difference is not a substitute: extra samples equal to $`g`$ cost
zero, so arbitrary length mismatch can have zero ERP distance.

## 6. Quantized-trie interval relaxation

When a target sample is represented by a bin $`B=[\ell,h]`$, the exact box
minimum is

```math
\operatorname{dist}(v,B)=\min_{b\in B}\lvert v-b\rvert
=\max(0,\ell-v,v-h).
```

The interval recurrence replaces match cost by
$`\operatorname{dist}(x_i,B)`$ and insertion cost by
$`\operatorname{dist}(g,B)`$; deletion cost $`\lvert x_i-g\rvert`$ is
already exact. Point bins reproduce the scalar DP exactly. Non-point bins
lower-bound every concrete realization and therefore support no-false-negative
subtree pruning.

## 7. Implementation mapping

| Paper concept | Repository implementation |
|---|---|
| Fixed gap $`g`$ | `ErpConfig::new(g)` and normalized public field |
| Formula (5) recurrence | `ErpConfig::distance_with_cutoff` |
| Running empty-side sums | `empty_vs_nonempty_cost` and DP boundaries |
| Two-row optimization | production exact scorer |
| Independent full matrix | test-only differential oracle |
| Gap-mass lower bound | `erp_gap_mass_lower_bound` |
| Interval box minimum | `ErpConfig::step_column` via `interval_dist` |
| Indexed exact search | `ErpTransducer<V>` alias of the generic walker |

## 8. Verification and limitations

Rocq proves interval admissibility, point-bin exactness, the potential bound for
arbitrary alignment scripts, quotient-equivalence laws, and that a zero-cost
alignment implies equal quotient normal forms. Verus checks the Rust-facing
integer arithmetic; Z3 and cvc5 independently reject bounded counterexamples.
Generated Rust properties mirror these invariants over the floating-point
implementation and additionally test the triangle inequality and both sides of
quotient identity.

The proof models mathematical reals or integers. IEEE-754 addition is not
associative, and an overflowing finite sum becomes `TOP`; this implementation
therefore treats finite non-negative `f64` plus positive infinity as its lawful
runtime cost domain and rejects NaN samples.
