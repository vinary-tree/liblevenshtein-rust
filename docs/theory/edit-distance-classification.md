# Classifying edit distances before implementing them

This document is the decision procedure for deciding whether a proposed
distance belongs in `liblevenshtein`, which existing mechanism expresses it,
and which proof obligation actually gates dictionary pruning. It prevents the
word *distance* from hiding four materially different implementation problems.

![Decision tree for placing a proposed measure: configurations remain in OperationSet, history-dependent scripts add Position state, alternative accumulation adds a CostMonoid, non-string domains add an ElasticKernel or language product, and gain-valued scoring crosses to the WFST crates.](../diagrams/automata/metric-classification.svg)

## 1. Vocabulary and the first boundary

An **edit operation** consumes a fixed number of source and target units and
has a non-negative cost. An **alignment** is a sequence of such operations in
which every input unit is consumed exactly once. An **edit script** is more
general: a later operation may edit the output of an earlier operation.

`OperationType` represents the tuple
$`t=\langle t^x,t^y,t^w\rangle`$, where $`t^x`$ and $`t^y`$ are fixed
consumption counts and $`t^w`$ is the operation cost. Consequently an
`OperationSet` computes a minimum over alignments:

```math
d_{\mathcal O}(x,y)=
\min_{A\in\operatorname{Align}_{\mathcal O}(x,y)}
\sum_{t\in A} t^w.
```

A script distance instead minimizes over intermediate strings:

```math
d_{\mathrm{script}}(x,y)=
\min\{k\mid x=z_0\to z_1\to\cdots\to z_k=y\}.
```

Insertion, deletion, and substitution give the same minimum in both models:
re-editing an intermediate result does not help. Transposition separates them.
For example, true Damerau–Levenshtein permits
`CA -> AC -> ABC`, with distance two. Optimal string alignment (OSA) cannot
edit inside a pair that it already transposed and reports three. This is the
history boundary identified by Lowrance and Wagner's unrestricted algorithm
([Lowrance and Wagner 1975](https://doi.org/10.1145/321879.321880)).

## 2. The four implementation classes

| Class | Defining question | Mechanism | Implemented members |
|---|---|---|---|
| A — alignment configuration | Can a finite set of fixed-consumption operations express the measure? | validated `OperationSet` preset | Hamming, indel/LCS cost, bounded-skip subsequence |
| B — history-dependent script | Does the next legal cost depend on earlier operations? | typed finite state in `Position` and an `AutomatonVariant` | OSA, true Damerau, affine gap; contextual-cost surface |
| C — accumulation algebra | Do path steps combine by an operation other than addition? | `CostMonoid` with minimum selection fixed | discrete Fréchet uses bottleneck `max`; additive kernels use `+` |
| D — domain or language | Are the inputs sequences, intervals, or a regular language rather than one string pair? | `ElasticKernel` or `LanguageProduct` | MSM, ERP, TWED, banded DTW, discrete Fréchet, distance to a language |

The classes answer different questions and can compose. Discrete Fréchet is
both a sequence-domain kernel and a bottleneck accumulator. A regular-language
product can use ordinary additive Levenshtein costs. Classification chooses
seams; it does not force every measure into exactly one mathematical label.

## 3. Class A: configurations, not new automata

Hamming distance illustrates why equal length is not enough. Although both
strings have length three,
$`d_{\mathrm{Lev}}(\texttt{abc},\texttt{bca})=2`$ while
$`d_{\mathrm{Ham}}(\texttt{abc},\texttt{bca})=3`$. Levenshtein may delete the
first `a` and insert it at the end. Hamming forbids both operations and retains
only match and substitution.

The presets are:

| Preset | Allowed consumption | Semantic guard |
|---|---|---|
| Hamming | match and substitution, each $`1\to1`$ | source and target lengths must agree |
| indel/LCS cost | match, insertion, deletion | substitution is absent, so a replacement costs two |
| bounded skip | match and source deletion | target must be a subsequence of the source |

`OperationSet::validate` checks progress, aggregate consumption, finite
non-negative weights, and the configured resource ceiling. The repaired
`GeneralizedAutomaton` derives empty-side completion costs from the declared
operations instead of silently reinstating Levenshtein insertion or deletion.

## 4. Class B: finite history in the position

### 4.1 Affine gaps

An affine gap run of length $`r`$ costs $`g_o+r g_e`$. A fixed operation weight
cannot express whether a gap is opening or extending for unbounded $`r`$.
Gotoh's recurrence therefore has match, query-gap, and dictionary-gap layers
([Gotoh 1982](https://doi.org/10.1016/0022-2836(82)90398-9)). The layer is one
byte of finite history in `PositionKind`.

The pruning proof is not a general claim that affine distance is a metric. Its
shipping invariant is weaker and sufficient: every legal step has
non-negative scaled cost, so a state's accumulated cost lower-bounds every
completion. The metric-index question remains separate.

### 4.2 True Damerau–Levenshtein

The unrestricted Lowrance–Wagner transposition term is:

```math
D[i,j]=\min\left(\ldots,
D[k'-1,l-1]+(i-k'-1)+(j-l-1)+1\right).
```

Under budget $`k`$, the joint interior bound is
$`(i-k')+(j-l)\le k+1`$. The streaming automaton stores a positive endpoint
delta in `Position::aux`, prepays the transposition and query-interior
deletions, charges dictionary-interior insertions as they arrive, and resolves
at the deferred endpoint. This is finite for bounded $`k`$ without enumerating
the alphabet.

OSA and true Damerau have the same informal operation vocabulary but different
composition semantics. Merge/split is different again: it models one source
unit corresponding to two target units, such as OCR confusions. It must not be
used as a substitute for unrestricted transposition.

### 4.3 Context-aware costs

A cost depending on query left/right context also needs history, but dictionary
right context is unavailable before descent. The public contextual surface
therefore exposes only information that is actually known at the transition
boundary and requires a positive `min_nonzero_cost` for its conservative
realignment bound. Learned weights belong above this surface.

## 5. Class C: how path costs accumulate

`CostMonoid` fixes candidate choice to minimum and parameterizes only sequential
accumulation. Its laws are described in the
[cost-monoid design](../design/cost-monoid.md). Additive distance uses:

```math
c_{j+1}=c_j+w_j.
```

Discrete Fréchet uses bottleneck accumulation:

```math
c_{j+1}=\max(c_j,w_j).
```

Both inflate when $`w_j\ge0`$, which is the property a bounded trie walk needs.
A semiring interface would add distributivity, closure, and division that this
acyclic bounded dynamic program never uses. Conversely, a generic semiring
does not guarantee the total order and non-negative inflation needed here.

`CostMonoid` must therefore not acquire a caller-defined choice operator,
Kleene star, or division. A use case requiring those operations crosses to the
weighted-transducer layer.

## 6. Class D: different domains and language products

`ElasticKernel` factors a sequence recurrence into four checked obligations:

- K1: an interval-relaxed column lower-bounds every represented concrete
  column;
- K2: lawful path extension cannot reduce accumulated cost;
- K3: a surviving terminal is rescored exactly; and
- K4: a candidate-level bound lower-bounds its exact distance.

The generic walker shares dictionary prefixes, applies K1 during descent, uses
K4 before exact scoring, and emits only K3 values. The traversal-level Rocq
theorem in `WalkerSoundness.v` proves that local K1 terminal bounds plus K2
child inflation imply no false negatives for the recursive walk.

The implemented kernels occupy different mathematical domains:

| Kernel | Accumulation | Identity qualification | Indexing fact used here |
|---|---|---|---|
| MSM | additive | metric on its configured finite domain | interval column bound |
| ERP | additive | identity modulo removal of the fixed gap value | interval and gap-mass bounds |
| TWED | additive | metric only under its strict stiffness parameter gate | interval and length bounds |
| banded DTW | additive | not a metric | band reachability and LB_Keogh-style lower bounds |
| discrete Fréchet | bottleneck | identity modulo consecutive duplicate collapse | interval, endpoint, and coverage bounds |

`LanguageProduct` addresses another domain change: compute $`d(w,L)`$ against
a regular language $`L`$ by advancing a language frontier alongside cost
levels. The load-bearing algebraic law is relational-image preservation over
union, not a new string edit operation.

## 7. Metricity is not pruning admissibility

A **metric** is non-negative, symmetric, identifies equal values, and satisfies
the triangle inequality. Metric trees need those properties. This repository's
dictionary and elastic walkers use lower bounds instead. Their core implication
is:

```math
\operatorname{lowerBound}(p)>k
\Longrightarrow
\forall x\in\operatorname{descendants}(p),\ d(q,x)>k.
```

No triangle inequality appears. Banded DTW can therefore use an admissible
trie lower bound while remaining explicitly excluded from metric-only indexes.
Conversely, merely proving a distance is a metric does not prove an interval
relaxation or prefix bound sound.

The code exposes this distinction: `Algorithm::is_metric` and
`ElasticKernel::IS_METRIC` label the mathematical property, while K1--K4 and
subsumption proofs gate the actual pruning mechanisms.

## 8. The crate boundary: Dyck and fzf

Exact multi-kind Dyck correction is not a regular-language preset. A bounded
depth-$`D`$ recognizer must remember a stack word, requiring

```math
N(r,D)=\sum_{d=0}^{D}r^d=\frac{r^{D+1}-1}{r-1}\quad(r\ge2)
```

states for $`r`$ bracket kinds. Exact correction therefore lives in
`lling-llang` as a pushdown/grammar algorithm, with liblevenshtein providing a
bracket projection/lower-bound surface. The implemented interval recurrence is
kind-sensitive; a closer cannot repair against the wrong opener kind.

fzf scoring crosses for a different reason. It combines alternatives with
maximum, accumulates gains and penalties, and allows a local alignment to start
after any prefix. Its correct branch-and-bound calculation must retain that
unstarted alternative. The structural DFS hook stays here, exact scoring and
the lazy adapter live in `duallity`, and the Arctic $`(\max,+)`$ semiring lives
in `lling-llang`. See the
[crate-boundary design](../design/crate-boundary-and-prune-duality.md).

## 9. Checklist for a future measure

Use this order; later questions do not repair an earlier classification error.

1. State the exact mathematical function, including quotient or normalization
   semantics.
2. Decide whether it minimizes alignments or edit scripts.
3. If alignment-expressible, encode and validate an `OperationSet` before
   adding an automaton variant.
4. If history-dependent, identify the smallest finite continuation state and
   prove that bounded search makes it finite.
5. State path accumulation, lawful carrier values, infinity, and ordering.
6. State the input domain and the exact/relaxed recurrence.
7. Derive the subtree or candidate bound actually used by traversal.
8. Prove that bound, then translate it into differential and generated
   properties.
9. Label metricity separately and prohibit incompatible index structures in
   code.
10. If gain-valued steps, closure, division, or pushdown memory are required,
    cross to the appropriate sibling crate instead of widening the bounded
    distance abstractions.

The implementation map is intentionally explicit: Class A maps to
`OperationSet`; Class B to `PositionKind` plus `AutomatonVariant`; Class C to
`CostMonoid`; Class D to `ElasticKernel` or `LanguageProduct`; pushdown
correction and weighted-transducer algebra remain across the crate boundary.
