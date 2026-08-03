# Eiter–Mannila discrete Fréchet: paper and implementation analysis

## 1. Scope and source

Thomas Eiter and Heikki Mannila introduced the **discrete Fréchet distance**,
also called the *coupling distance*, in *Computing Discrete Fréchet Distance*,
TU Vienna Technical Report CD-TR 94/64 (25 April 1994). The report has no DOI;
the canonical source for this analysis is the
[author-hosted report](https://www.kr.tuwien.ac.at/staff/eiter/et-archive/files/cdtr9464.pdf).

The measure discretizes Maurice Fréchet's curve distance. The continuous
ancestor is described in “Sur quelques points du calcul fonctionnel,” DOI
[10.1007/BF03018603](https://doi.org/10.1007/BF03018603). This implementation
does not approximate continuous curves; it computes Eiter and Mannila's exact
discrete recurrence for one-dimensional, real-valued sample sequences.

## 2. Couplings and the leash intuition

Let `$`x=(x_1,\ldots,x_m)`$` and `$`y=(y_1,\ldots,y_n)`$` be nonempty
sequences in a metric space with point distance `$`d`$`. A **coupling** starts
at `$`(1,1)`$`, ends at `$`(m,n)`$`, and advances by one of
`$`(1,0)`$`, `$`(0,1)`$`, or `$`(1,1)`$`. Thus every point of both sequences
is visited, order is preserved, and neither participant backtracks.

The coupling's length is its longest link:

```math
\lVert L\rVert=\max_{(i,j)\in L} d(x_i,y_j).
```

Discrete Fréchet distance chooses the coupling with the shortest longest link:

```math
D_{\mathrm{dF}}(x,y)=\min_L\lVert L\rVert.
```

This is the dog-and-leash intuition made finite: both walkers may wait or move,
and the score is the shortest leash sufficient for the whole ordered walk.

## 3. Table 1 dynamic program

For scalar samples this crate uses `$`d(a,b)=\lvert a-b\rvert`$`. Eiter and
Mannila's Table 1 recurrence is:

```math
D[i,j]=\max\!\left(
  \lvert x_i-y_j\rvert,
  \min\{D[i-1,j],D[i-1,j-1],D[i,j-1]\}
\right).
```

The boundary branches are load-bearing:

```math
\begin{aligned}
D[1,1]&=\lvert x_1-y_1\rvert,\\
D[i,1]&=\max(D[i-1,1],\lvert x_i-y_1\rvert),\\
D[1,j]&=\max(D[1,j-1],\lvert x_1-y_j\rvert).
\end{aligned}
```

The report proves `$`\mathcal{O}(mn)`$` time. The Rust implementation retains
only two rows and exploits symmetry to use `$`\mathcal{O}(\min(m,n))`$` memory.
The source report supplies the recurrence rather than a numeric worked table;
the source-conformance unit test therefore executes every Table 1 branch on
small hand-checkable sequences.

## 4. Metric status and the representation quotient

The report states metricity for polygonal curves. Raw vertex vectors have a
representation degeneracy: inserting a consecutive duplicate does not change
the traversed curve and costs zero. For example:

```math
D_{\mathrm{dF}}([1,1,2],[1,2])=0.
```

Let `$`R(x)`$` collapse every maximal run of equal adjacent samples to one
sample. The executable API is consequently described as a pseudometric on raw
vectors and a metric modulo run-length collapse:

```math
D_{\mathrm{dF}}(x,y)=0\quad\Longleftrightarrow\quad R(x)=R(y).
```

This is not a contradiction with the paper: the quotient identifies raw
vertex lists that trace the same zero-length-stuttered polygonal curve.

## 5. Trie interval relaxation

A quantized trie edge represents an unknown target value
`$`y_j\in B_j=[\ell_j,h_j]`$`. The only free scalar leaf is replaced by its
exact interval minimum:

```math
\lvert x_i-y_j\rvert\rightsquigarrow
\operatorname{dist}(x_i,B_j)
=\min_{v\in B_j}\lvert x_i-v\rvert.
```

Minimum and maximum are monotone in every argument. Replacing each leaf by a
lower bound therefore lower-bounds every complete cell. A point bin is exact:

```math
\operatorname{dist}(x_i,[y_j,y_j])=\lvert x_i-y_j\rvert.
```

That equality is the tightness gate. Admissibility alone would allow a useless
always-zero relaxation.

## 6. Candidate lower bounds

The following bounds are implementation derivations from the coupling
definition; the report does not present them as indexing bounds.

Every coupling contains both endpoint links, hence:

```math
\max\!\left(
  \lvert x_1-y_1\rvert,
  \lvert x_m-y_n\rvert
\right)\le D_{\mathrm{dF}}(x,y).
```

Every `$`x_i`$` is coupled to some `$`y_j`$`, yielding the one-sided Hausdorff
bound:

```math
\max_i\min_j\lvert x_i-y_j\rvert\le D_{\mathrm{dF}}(x,y).
```

The implementation sorts `$`y`$` and evaluates nearest neighbours in
`$`\mathcal{O}((m+n)\log n)`$`; it takes the maximum of this value and the
constant-time endpoint bound. Both are formalized independently before their
combination is used as K4.

## 7. Why `BottleneckCost` is sufficient

The generic walker requires path inflation, not addition. With non-negative
link `$`w`$`:

```math
a\le\max(a,w).
```

Thus a pruned prefix cannot recover below the cutoff. The walker still compares
alternative paths by minimum; only path extension changes from `$`+`$` to
`$`\max`$`. No range or k-nearest-neighbour traversal code changes for this
kernel.

## 8. Empty and non-finite inputs

The source definition assumes nonempty polygonal curves. The Rust boundary is
explicit and total:

- two empty sequences have distance zero;
- exactly one empty sequence has `TOP` (positive infinity), because no
  endpoint-covering coupling exists;
- NaN or infinite samples are outside the exact domain and never enter
  interval queues;
- kNN reports finite candidates only, while a range cutoff of `TOP` may expose
  the explicit one-empty `TOP` score.

These choices preserve the mathematical recurrence while making API behavior
deterministic on inputs outside the paper's domain.

## 9. Evidence map

| Claim | Executable evidence | Formal evidence |
|---|---|---|
| Table 1 recurrence | independent full-matrix differential property | monotone recurrence theorem |
| interval admissibility and point exactness | generated cellwise and point-bin properties | Rocq, Verus, Z3, cvc5 |
| bottleneck path inflation | range/kNN brute-force properties | cost-monoid proof plus kernel-specific obligations |
| endpoint and Hausdorff K4 | generated lower-bound property | Rocq coverage theorem; Verus/SMT combination checks |
| raw quotient identity | generated run-collapse identity | zero-link and zero-bottleneck theorems |
| generic walker behavior | generated range and kNN databases | existing `ElasticTrieSearch.tla` model |

The detailed proof mapping is in the [verification README](../../verification/README.md),
the algorithm is presented literately in the
[elastic-measures chapter](../../algorithms/12-elastic-measures/README.md), and
operational limits are in the [resource-exhaustion guide](../../security/resource-exhaustion.md).

## 10. References

1. T. Eiter and H. Mannila, “Computing Discrete Fréchet Distance,” Technical
   Report CD-TR 94/64, TU Vienna, 1994.
   [Author-hosted report](https://www.kr.tuwien.ac.at/staff/eiter/et-archive/files/cdtr9464.pdf).
2. M. Fréchet, “Sur quelques points du calcul fonctionnel,” *Rendiconti del
   Circolo Matematico di Palermo* 22, 1–72, 1906. DOI:
   [10.1007/BF03018603](https://doi.org/10.1007/BF03018603).
