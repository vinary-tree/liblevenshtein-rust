# Lazy ordered-cost product automata

**Status:** theory and design contract · **Scope:** fixed-query string and
finite-series scores · **Metric boundary:** metric laws qualify instances but
do not define the execution architecture

The architecture that makes `liblevenshtein` distinctive generalizes beyond
Levenshtein distance, but not by translating every score into Levenshtein edit
operations. Each score keeps its own recurrence, cost algebra, finite carry,
ground distance, and lawful domain. What generalizes is the representation of
the recurrence's remaining behavior: construct query-specialized residual
states lazily; normalize them into exact antichain frontiers; intern them behind
compact IDs; and explore their synchronized product with a dictionary only
where real dictionary edges demand a transition.

The mathematical framework is **[Ordered Residual Calculus
(ORC)](ordered-residual-calculus.md)**. Its companion manuscript supplies
realization theorems, algebras, forward and backward calculi, categories,
worked measure-to-machine derivations, and a certified optimizer design.
**Lazy ordered-cost product automata** names the operational family developed
here. A **metric automaton** is a separately qualified instance. General DTW
can use this architecture without being a metric; an arbitrary metric need not
have an effective finite or bounded-memory residual representation.

![Every measure has residual behavior; representation proofs supply executable machines. Ordered transformations and simulations support frontier optimization, dictionary products explore observed transitions, and abstractions require exact verification. Metric qualification is a separate branch from the original measure.](../diagrams/architectures/ordered-cost-theory-layers.svg)

## 1. Which mathematics is load-bearing?

No single subject is sufficient. The smallest useful synthesis is:

| Mathematics | Exact role here | Priority |
|---|---|---|
| weighted automata and weighted residuals | assign a best cost to each consumed word or finite series and define query-specialized remaining behavior | essential |
| ordered algebra, dioids, and tropical algebra | express alternative choice and path extension for additive and bottleneck recurrences | essential |
| order theory, simulations, and antichains | prove when one live position safely makes another redundant | essential |
| automata theory | define the reachable synchronized product and its language-intersection semantics | essential |
| abstract interpretation | prove interval or box labels are admissible lower simulations of concrete labels | essential for quantized temporal retrieval |
| symbolic and register automata | describe real-valued labels and bounded continuation data without falsely claiming a finite alphabet or finite state set | essential vocabulary for temporal machines |
| coalgebra | define residual realization, exact machine morphisms, chunk equivalence, and resumption | essential semantic structure |
| metric geometry | certify nonnegativity, symmetry, identity on a domain or quotient, and triangle inequality | required only for metric-qualified instances |
| category theory and Lawvere geometry | compose exact realizations and lower simulations; quantify directed residual error and nonexpansive derivatives | constructive proof tools; ORC Sections 7–9 |
| set theory and discrete mathematics | provide ambient language for relations, orders, graphs, and finite combinatorics | foundational but too general by themselves |
| algebraic and backward calculus | derive residual transitions, propagate acceptable budgets, and calculate recursive zipper contexts | constructive synthesis and optimization tools |
| differential calculus on real parameters | differentiate smooth objectives such as Soft-DTW | separate gradient semantics; no idempotent pruning assumption |

Residual behavior is the primary semantic base; weighted automata realize an
important subclass. Order and simulation justify pruning; abstract
interpretation justifies quantization. The categories in ORC prove that these
constructions compose. Their equations can be discharged when specializing a
machine, with no runtime categorical dispatch.

## 2. Ordered path costs

Let `C` be a set of exact canonical costs with alternative choice
$`\oplus`$, sequential extension $`\otimes`$, dead value $`\top`$, path
identity $`e`$, and total comparison $`\leq`$:

```math
(C,\oplus,\otimes,\top,e,\leq).
```

An exact scalar path algebra has the following laws. Each optimization must
name the subset it needs; residual realization itself needs none of them:

1. $`\oplus`$ is associative, commutative, and idempotent;
2. $`\top`$ is the identity for alternative choice;
3. $`a\leq b`$ exactly when $`a\oplus b=a`$;
4. $`\otimes`$ is associative with identity $`e`$;
5. $`\otimes`$ distributes over finite $`\oplus`$ choices;
6. path extension is monotone in both arguments;
7. $`\top`$ is absorbing for extension; and
8. every lawful extension is nondecreasing, or the kernel supplies an
   independently proved future lower bound.

Additive edit and elastic recurrences use the tropical min-plus instance:

```math
a\oplus b=\min(a,b),\qquad
a\otimes b=a+b,\qquad e=0,\qquad\top=\infty.
```

Discrete Fréchet uses a bottleneck min-max instance:

```math
a\oplus b=\min(a,b),\qquad
a\otimes b=\max(a,b),\qquad e=0,\qquad\top=\infty.
```

The crate's `CostMonoid` exposes path extension and keeps minimum selection
fixed. This API does not by itself certify these scalar laws for every
numerical implementation. ORC C1 instead uses monotone, top-preserving **cost
transformations**: pointwise minimum is choice and function composition is
extension. Composition stays associative even when a scalar floating-point
addition is rounded at every edge. Path evaluation order remains explicit.

A public quantale API is unnecessary for the present acyclic recurrences.
Quantales are nevertheless relevant already: the extended nonnegative reals
with reversed order support Lawvere geometry, and complete lattices organize
abstract interpretation. Their relevance is not restricted to infinite paths.

Floating-point costs are not quotient values under numerical tolerance. The
canonical carrier rejects NaN and invalid infinities, normalizes negative zero,
uses exact structural equality for identity, and uses total ordering for
canonical sort. Epsilon equality is forbidden for interning, dominance, and
cutoff membership.

Canonical equality does not make floating-point addition associative, or
transfer an exact-real triangle theorem to rounded outputs. ORC Sections 3.3
and 12.3 give explicit counterexamples and separate numerical contracts.

For cutoff $`\tau`$, define the saturation map:

```math
T_\tau(c)=
\begin{cases}
c & c\leq\tau,\\
\top & c>\tau.
\end{cases}
```

The lawful transition family is **cutoff-congruent** when:

```math
T_\tau(T_\tau(a)\otimes w)=T_\tau(a\otimes w).
```

**Theorem schema CA-1 (cutoff-pruning preservation).** If extension is
monotone and cutoff-congruent, saturating every over-cutoff partial cost
preserves all outputs at or below $`\tau`$. With an admissible completion
estimate $`h(p)`$, the stronger guard $`c_p\otimes h(p)>\tau`$ is sound only
after proving $`h(p)`$ no greater than every lawful completion from $`p`$.
Negative rewards and negative cycles are immediate counterexamples.

**DC-1 (complete dependency-cut pruning).** For a recurrence dependency
graph, a live cut must intersect every unresolved accepting path. Expanding
one live vertex preserves this property when its successor enumeration
contains every semantic outgoing edge and an accepting vertex is marked
resolved only after its exact endpoint obligation is proved. If the cost at
each successor is no smaller than its predecessor, the minimum cost on a
nonempty live cut lower-bounds every unresolved accepting completion. An
inclusive cutoff $`\tau`$ therefore permits strict pruning only when every
live cost is greater than $`\tau`$. The [Rocq proof and bypass
control](../verification/temporal_automata/theories/CertifiedDependencyCut.v)
show that a pending multi-label operation can jump over the current row: a
row cost of six would falsely prune a cost-four answer at cutoff five if the
pending operation were omitted. The proof uses a complete path context as its
vertex; source recurrence enumeration, numerical monotonicity, and a concrete
storage-to-cut correspondence remain instance obligations. Lexicographic
kNN pruning also needs a certified tie-key floor.

## 3. Residuals are the generalized automaton states

Fix a finite query $`q`$, lawful parameters $`\theta`$, and cutoff $`\tau`$.
Let $`F_{q,\theta,\tau}(y)`$ be the exact score truncated to an explicit
over-cutoff value. After consuming target prefix $`u`$, define its residual:

```math
F_u(z)=F_{q,\theta,\tau}(uz).
```

Prefixes $`u`$ and $`v`$ are behaviorally equivalent when every future suffix
has the same score:

```math
u\equiv v
\quad\Longleftrightarrow\quad
\forall z,\ F_u(z)=F_v(z).
```

The ideal deterministic state is a residual-equivalence class. Its transition
is the weighted derivative:

```math
\delta([u],a)=[ua].
```

This is the quantitative analogue of language left quotients and Brzozowski
derivatives. A classic Levenshtein frontier, weighted positional frontier,
temporal DP column, sparse row/cost set, previous-target register, and timestamp
register are different concrete representations of the same kind of object:
the information from the consumed prefix that can still affect every possible
continuation.

**Theorem schema RM-1 (residual realization).** After consuming $`u`$, the
canonical derivative machine is in residual $`\partial_u F`$, and its output
is $`F(u)`$.

**Theorem schema RM-2 (deterministic residual minimality).** Every reachable
deterministic cost-output machine realizing $`F`$ maps behavior-preservingly
onto the reachable residual classes. If the residual index is finite, the
derivative machine is the minimal reachable deterministic Moore machine for
$`F`$. This is not a minimality claim for weighted NFAs or arbitrary linear
semiring representations, and exact structural interning does not claim it has
discovered the complete residual quotient.

Every measure has this residual semantics, including noncomputable measures.
Execution and compactness are additional theorems. An effective scorer supplies
an exact history-retaining realization; a proved finite residual quotient
supplies a finite Moore machine; a sufficient finite recurrence context can
supply a bounded-register machine. Metric axioms alone imply none of these
representation bounds. ORC R4 constructs a strict metric with a nonregular
radius-one ball.

For the bounded online profile, the exact residual must factor through a
representation with a prefix-independent number of retained cells and bounded
continuation context. A sound abstract filter is a distinct realization level:
the constant-zero filter works for every nonnegative measure, but proves
nothing about compact **exact** scoring. Its exact verifier needs a separate
cost and storage contract. Real-valued registers may have infinitely many
values; growing counters need growing bit length even with fixed register count.

## 4. Canonical frontiers and proved subsumption

A practical residual representation is a finite frontier:

```math
S=(\kappa,\{p_1,\ldots,p_n\}),
```

where $`\kappa`$ is all finite continuation context and a position contains a
query coordinate, recurrence phase, and accumulated cost. Let $`B_p(z)`$ be
the least completion cost from $`p`$ on suffix $`z`$. The authoritative
continuation preorder is:

```math
p\preceq q
\quad\Longleftrightarrow\quad
\forall z,\ B_p(z)\leq B_q(z).
```

If $`p\preceq q`$, then $`q`$ is redundant in a minimum-cost frontier. An
executable local rule is legal only when it implies this semantic relation. A
common sufficient proof constructs a forward simulation: $`p`$ has no worse
final output and can match every transition from $`q`$ into another related
pair. A zero-target-consumption path from $`p`$ to $`q`$ is an especially
useful kernel-specific witness.

The classical Levenshtein position/error formula is one such theorem; it is
not a generic rule for MSM, ERP, TWED, or Fréchet. If a kernel has no stronger
simulation proof, it may merge exact duplicates and nothing more.

Let $`E`$ be the **executable** reflexive, transitive dominance relation, proved
sound for $`\preceq`$. Let $`P_{q,\tau,E}/{\sim_E}`$ quotient reachable live
positions by mutual $`E`$-dominance. Define its maximum antichain width:

```math
W_E(q,\tau)=\sup\{|C|\mid C\subseteq P_{q,\tau,E}/{\sim_E}
\text{ and }C\text{ is an antichain}\}.
```

**Theorem schema AW-1 (frontier-width bound).** Every normalized reachable
frontier that removes every strict $`E`$-dominated atom and keeps one
representative per remaining equivalence class contains at most
$`W_E(q,\tau)`$ positions. An incomplete simulation cannot claim the smaller
width of the full semantic order. ORC F2–F3 prove this normal-form theorem and
give a two-atom counterexample. The bound counts atoms, not context bytes or
cost bit lengths. This bounds one frontier,
not the total number of distinct canonical states interned during a branching
dictionary search. Kernel-specific order structure may permit a tighter
linear or ordered normalization; no such complexity improvement is inherited
from the generic theorem alone.

Normalization $`N`$ must prove:

```math
\operatorname{behavior}(N(S))=\operatorname{behavior}(S),
\qquad N(N(S))=N(S),
\qquad N(S)=N(\pi(S))
```

for every predecessor-enumeration permutation $`\pi`$. The representation is
sorted, duplicate-free, deterministic, and complete for every piece of future
context. Witness-carrying states additionally use a specified canonical
tie-order; equal score alone cannot authorize changing the promised witness.

## 5. Exact and abstract query transitions

An exact point transition must correspond to the independent recurrence after
one additional target label. For an exact query automaton $`A_{q,\tau}`$:

```math
L(A_{q,\tau})=\{y\mid F_{q,\theta}(y)\leq\tau\}.
```

A quantized temporal edge instead denotes a concrete set. If abstract label
$`\hat a`$ has concretization $`\gamma(\hat a)`$, the abstract transition must
lower-simulate every concrete member:

```math
S^{\#}\;R\;S\ \Longrightarrow
\forall a\in\gamma(\hat a),\qquad
\operatorname{step}^{\#}(S^{\#},\hat a)
\;R\;
\operatorname{step}(S,a),
```

where $`R`$ guarantees the abstract cost never exceeds the corresponding
concrete cost. For an exact-state embedding $`\eta`$, singleton abstraction
is exact only under the additional obligation:

```math
\operatorname{step}^{\#}(\eta(S),[a,a])
\equiv \eta(\operatorname{step}(S,a)).
```

This is abstract interpretation: intervals or boxes form an abstract domain,
and transition soundness is a lower simulation. A Galois connection is useful
for structuring the proof, but the executable obligation is the quantified
lower-bound inequality.

A point label does not recover precision already lost in a non-singleton
abstract history. Lower simulation alone also does not imply singleton
exactness; the embedding equation is a separate proof.

An interval product recognizes a candidate superset, not the exact metric
ball. Exact scoring of every full-precision member of every quantization-
collision bucket removes false positives and is the sole authority for emitted
distance and membership.

Order abstract labels by precision:

```math
\hat a_1\sqsubseteq\hat a_2
\quad\Longleftrightarrow\quad
\gamma(\hat a_1)\subseteq\gamma(\hat a_2).
```

**Theorem schema AP-1 (candidate monotonicity under refinement).** If the
abstract transformer is precision-monotone, replacing every label by a more
precise abstraction in a completed search can remove false-positive
candidates but cannot remove an exact within-cutoff result. The theorem is not
stated for partial pages under unequal budgets because tighter bounds can
change scheduling and therefore the point at which execution pauses.

Independent interval products can lose correlations needed by adjacent-target
recurrences. Soundness survives when their concretization is a superset of all
feasible correlated points, though pruning may weaken. Timestamped TWED may
benefit from a relational domain carrying $`t_{j-1}<t_j`$ or a timestamp-
difference interval; an interval box itself cannot certify validity of stored
original timestamps.

## 6. Lazy synchronized products

Let dictionary edge $`n\xrightarrow{b}n'`$ carry label $`b`$, and let decoding
$`\chi_q`$ translate it into a query observation. The product step is:

```math
(n,S)\xrightarrow{b}
(n',\operatorname{step}(S,\chi_q(b))).
```

When both components are exact acceptors, the product's accepted language is
their intersection. With abstract interval query states, the product first
accepts a sound candidate superset and exact leaf verification restores exact
result semantics.

The Cartesian set of possible pairs is never materialized. A search session
stores dictionary cursor/path data and one machine-word `StateId`. The ID
refers into a collision-checked canonical arena, and an observed-transition
cache maps $`(\mathrm{StateId},\text{ exact observation key})`$ to a successor
ID or dead state.

The defining laziness property is:

> Every constructed query transition is demanded by an inspected outgoing
> edge of a reachable dictionary-product state. No pair is created merely
> because it could exist.

This makes construction demand-driven. It is not an output-size complexity
bound: an empty result can still require inspecting a large dictionary.
Section 11 accounts for visited edges and constructed transitions. Prefix
sharing avoids rescoring descendants from scratch, subsumption reduces live width,
exact interning shares repeated residual representations, and observation
caching shares repeated transitions. None of those optimizations changes the
score recurrence.

On a trie, a node commonly identifies one prefix. On a DAWG or another graph
with shared suffixes, two distinct prefixes can reach the same dictionary node.
Memoizing only `(`dictionary node`, `StateId`)` is therefore unsound when keys,
stable IDs, values, or witnesses depend on path identity. Such pair
deduplication requires a backend-specific suffix-congruence theorem and a
correct multiplicity/reconstruction mechanism; otherwise each pending product
item retains a compact path identity even though query transitions are shared.

**Theorem BF-1 (ordered best-first kNN stopping).** Fix one immutable index
revision $`\sigma`$, a query $`q`$, and the operation's declared, injective tie
key $`t_\sigma(x)`$ for each live candidate $`x`$. Rank a finite exact result by
$`R(x)=(d(q,x),t_\sigma(x))`$ in lexicographic order, using the same exact cost
comparator as the public result API. The tie key is a result-order attribute;
it is not a component of the path-cost monoid.

Let $`F`$ cover every unverified candidate. A queued region $`Q\in F`$ carries
an admissible cost bound $`L(Q)\le d(q,x)`$ and a tie floor
$`M(Q)\le t_\sigma(x)`$ for every $`x\in Q`$. The exact minimum tie key is the
strongest such floor, but a smaller certified floor remains sound. If $`k`$
distinct exact results have been verified and their worst rank is
$`W=(c_k,t_k)`$, then $`Q`$ may be discarded when

```math
(L(Q),M(Q))\ge_{\mathrm{lex}} W,
\quad\text{equivalently}\quad
L(Q)>c_k\ \lor\ (L(Q)=c_k\ \land\ M(Q)\ge t_k).
```

If the smallest queued lower pair meets this condition, the entire queue may
be discarded. This includes equality at $`c_k`$ only because the tie floor
rules out an earlier equal-cost candidate. An unknown floor is represented by
a value below every real tie key and cannot justify equality pruning. An empty
region is discarded independently. With fewer than $`k`$ exact results there
is no kth rank and this stopping rule does not apply.

**Proof.** For every unverified $`x\in Q`$, coordinatewise admissibility gives
$`(L(Q),M(Q))\le_{\mathrm{lex}}R(x)`$: if the cost inequality is strict the tie
coordinate is irrelevant, and at equal cost the tie inequality applies. If
some unverified $`x`$ improved the current kth result, then
$`(L(Q),M(Q))\le_{\mathrm{lex}}R(x)<_{\mathrm{lex}}W`$, contradicting the
discard condition. Candidate identity is unique, so another candidate cannot
have exactly the same pair as the already verified kth result. Keeping the
best $`k`$ verified pairs therefore yields the same ordered top $`k`$ as a
complete scan. The proof needs neither metricity nor a consistent heuristic:
admissibility and frontier coverage suffice, even when child bounds are not
monotone along dictionary edges. As the best-result heap improves, $`W`$ can
only decrease, so a discarded region never needs reopening.

For a concrete candidate with tie key $`t`$, the maximum of its region bound
and any admissible candidate-specific bound is again a cost lower bound.
Comparing that bound paired with $`t`$ can avoid exact verification. An exact
verifier that is still needed must accept the kth cost inclusively: an earlier
tie at equal cost may replace the current worst result.

The tie key is **operation-specific**. General elastic bounded kNN scans
collision buckets in $`(\text{bucket ID},\text{slot})`$ order; replacement or
removal can change slots. Timestamped TWED bounded kNN orders equal distances
by stable episode ID. The legacy elastic best-first convenience API uses
encounter sequence, which is a different ordering contract. In particular, a
best-first bounded result heap must compare its declared tie keys rather than
reuse an encounter-sequence comparator. No ordering on caller values is needed.

For the current mapped dictionaries, a nonempty elastic terminal bucket has
tie floor $`(\text{bucket ID},0)`$, and a timestamped terminal bucket has its
smallest episode ID. A region's floor is the minimum of its live terminal
floor and its children's floors. Empty buckets contribute nothing, and an
entirely empty region has a separate `Empty` tag. A partially inspected
region's observed minimum is an **upper** bound on its true minimum and must
not be cached as a floor. A query-local, revision-bound structural traversal
may memoize complete floors by physical node identity only when terminal
payloads determine the same tie-key set at every path to that node. Otherwise
the path belongs in the memo key or the floor stays unknown. DAWG state
sharing alone never licenses deduplication of candidate paths.

The bound comparison uses exact machine ordering, not an epsilon. Its
numerical premise is that the implemented abstract bound is no greater than
the implemented exact score under that ordering; an ideal-real inequality
alone does not establish this for rounded binary64 recurrences. The completed
result theorem does not assert that different schedules produce identical
resource usage or incomplete partial pages. Structural work and retained
summary memory must be charged before use; if an optional summary cannot be
completed within its budget, ordinary cost-only traversal remains sound.

Three short counterexamples fix the boundary of the rule. With $`k=1`$, a
verified rank $`(5,2)`$ and an unvisited rank $`(5,1)`$ refute stopping merely
because the cost bound equals five. A region bound $`(4,100)`$ cannot be
discarded against $`(5,1)`$: its candidate might have cost four. Finally, a
cached floor of ten becomes unsafe if a later revision inserts tie key one
into the region; node summaries must be revision-bound or conservatively
weakened. A stale **smaller** floor loses pruning opportunities but remains
safe.

The [Rocq order kernel](../verification/temporal_automata/theories/LazyProductOperations.v)
proves componentwise-to-lex lower bounds, local and whole-frontier stopping,
top-$`k`$ selection preservation, complete/unknown/empty tie-floor
composition, a sound executable prune predicate for all three summary tags,
sound elastic terminal bucket and timestamped episode-ID floors,
sound combination of region and candidate-specific cost bounds, floor safety
after candidate removal, and monotonicity after the kth rank improves. The
[finite TLA+ scheduling model](../verification/tla/LexicographicKnn.tla)
checks coverage and ordered completion across nondeterministic region splits,
summary resolution, empty regions, verification, and pruning. Its known floors
are exact for the finite corpus; unknown floors cannot authorize equality
pruning. Neither result alone discharges a Rust kernel's machine-bound or
heap-correspondence proof. The checked-in
[cost-only equality mutant](../verification/tla/LexicographicKnnCostOnlyEquality.cfg)
replaces the rank-aware pruning action and makes TLC violate `NoLostTopK`;
the [model report](../verification/CBC_LEXICOGRAPHIC_MODEL_REPORT.md) records
its concrete discarded top-two candidate and reproduction command.

## 7. Operations, cursors, and product zippers

The theory applies to operations as well as states. A query engine is correct
only if dictionary navigation, query transitions, product construction, and
result verification compose without changing their individual meanings.

### 7.1 A zipper is the focused dictionary component

A **dictionary zipper** is a persistent focus into one captured dictionary
revision. Abstractly, write:

```math
Z_D=(\sigma,n,\rho),
```

where $`\sigma`$ identifies the immutable snapshot, $`n`$ is the focused
dictionary node or backend cursor, and $`\rho`$ is the minimal context needed
for navigation or path reconstruction. A classic Huet zipper factors a tree
into a focused subtree and a one-hole context; dictionary implementations may
instead use node IDs, persistent trie references, parent spines, or backend-
native cursors, but they must satisfy the same observational navigation laws.

The dictionary operations are:

```math
\operatorname{children}:Z_D\longrightarrow
\operatorname{FiniteSeq}(A\times Z_D),
```

```math
\operatorname{descend}:Z_D\times A\longrightarrow
\operatorname{Option}(Z_D),
```

```math
\operatorname{final}:Z_D\longrightarrow\mathbb{B},
\qquad
\operatorname{value}:Z_D\longrightarrow\operatorname{Option}(V).
```

For path $`u`$ and label $`a`$, the required laws are:

1. **snapshot scope:** every descendant retains the same $`\sigma`$;
2. **descent soundness:** successful `descend(z,a)` focuses exactly path
   $`ua`$ in $`\sigma`$;
3. **child completeness:** `children(z)` enumerates every and only successful
   labelled descents, once each;
4. **final/value coherence:** finality and values are those of $`u`$ in
   $`\sigma`$;
5. **persistent cloning:** copying a zipper does not mutate or invalidate
   either focus; and
6. **path coherence:** if paths are exposed, reifying the child path appends
   exactly its edge label.

A query scheduler may consume a zipper into an **opaque traversal focus** that
retains only $`(\sigma,n)`$. Erasing $`\rho`$ is sound precisely when the opaque
view exposes no zipper-path operation, finality and descent factor through
$`(\sigma,n)`$, and the scheduler records its own root-relative parent trace.
The result path is then reconstructed from that trace. This is not permission
to drop path-sensitive dictionary semantics: a backend whose visibility or
value depends on the complete key must retain or recheck those units through
its documented final-admission operation.

These specialize the snapshot laws in
[snapshot semantics](snapshot-semantics.md). They authorize $`\mathcal{O}(1)`$
snapshot capture, immutable structural sharing, cheap focus copies, and
continuations that outlive the original dictionary handle.

**Algebraic data-type differentiation** describes zipper contexts. For a
regular recursive tree $`T\cong F(T)`$ built from a polynomial functor,
$`F'(T)`$ describes one parent frame. An arbitrary-depth subtree zipper has
shape $`T\times\operatorname{List}(F'(T))`$: a focus and its stack of
frames. A one-layer derivative alone does not encode all ancestors. This
calculus explains navigation; ORC's residual and backward budget calculi
separately derive query transitions and pruning conditions.

### 7.2 The query-operation algebra

For a query machine, distinguish the following operations rather than hiding
them behind one unconstrained transition method:

| Operation | Semantic duty | Principal optimization it may authorize |
|---|---|---|
| `seed` | represent the empty target prefix | precomputed query plan and initial closure |
| `classify` | map a concrete edge label to an exact observation class | characteristic vectors, interval-bin IDs, vector boxes |
| `generate` | construct reachable consuming successors | sparse position generation |
| `close` | add every required zero-input successor | iterative ranked worklist |
| `normalize` | preserve behavior while canonicalizing and subsuming | antichain width reduction |
| `intern` | assign one ID to exactly equal canonical states | compact queued product pairs |
| `step` | compose classify, generate, close, normalize, and intern | observed-transition cache |
| `lower_bound` | never exceed any represented exact completion | subtree pruning and priority scheduling |
| `relaxed_final` | admit every potentially exact final | candidate gate only |
| `exact_verify` | compute authoritative concrete cost and witness | exact emission and deterministic ties |

The corresponding laws are transition correspondence, classification
congruence, closure completeness and idempotence, normalization preservation,
collision-safe interning, lower-bound admissibility, and exact final
authority. In full generality classification may be state-relative,
$`\chi:S\times A\to O`$; query-wide characteristic classes are the simpler
state-independent case. Observation classes are cacheable only under:

```math
\chi(S,a)=\chi(S,b)
\Longrightarrow
\operatorname{step}(S,a)=\operatorname{step}(S,b)
```

for every lawful canonical state $`S`$. An interval label generally fails this
equality for exact concrete transitions; it instead uses the abstract lower-
simulation relation from Section 5.

A cached value is either a complete successor ID or a semantic dead marker.
Budget-exceeded and allocation-failed outcomes are not semantic transitions
and must not be cached unless every resource/configuration dependency is part
of the key. Under exact state decode, transition purity, immutable arena
entries, and complete keys, cache hits refine recomputation; eviction affects
performance only.

### 7.3 A product zipper is a focused reachable pair

A **product zipper** combines one dictionary focus with one compact query
state:

```math
Z_{D\otimes A}=(Z_D,\operatorname{StateId},\omega),
```

where $`\omega`$ is optional bounded path/witness context. Its child operation
is not an arbitrary Cartesian-product enumeration:

```math
\operatorname{child}((z,s),a)=
\begin{cases}
(\operatorname{descend}(z,a),\operatorname{step}(s,\chi(a)))
  & \text{if both successors exist and the query state is live},\\
\bot & \text{otherwise.}
\end{cases}
```

![A product zipper advances the immutable dictionary focus and compact query state on one observed edge, then either prunes the child or returns another focused reachable product pair.](../diagrams/architectures/product-zipper-operations.svg)

Historically named `IntersectionZipper` values are therefore operationally
product zippers. “Intersection” remains correct for the exact accepted
language, but “product” is the precise name for the focused state and its
transition construction.

The product zipper laws are:

1. its dictionary and query components denote the same consumed path;
2. a live child exists exactly when the dictionary edge exists and the query
   transition survives;
3. every enumerated child is reachable;
4. child enumeration is complete for all live outgoing edges;
5. final emission requires dictionary finality, exact query authority, and an
   explicit $`d\leq\tau`$ admission check after all trailing query-only
   operations have been closed;
6. cloning or suspending a product focus preserves its snapshot identity; and
7. rebuilding from a continuation is observationally equivalent to retaining
   the live focus.

Child-handle construction may be delayed until after the query transition.
Let $`q(s,a)`$ be the optional query successor and $`d(z,a)`$ the optional
dictionary descent. The ordinary and projection-first forms are:

```math
\operatorname{pair}(d(z,a),q(s,a))
\quad\text{and}\quad
q(s,a)\mathbin{\mathrm{andThen}}(s'\mapsto
d(z,a)\mathbin{\mathrm{map}}(z'\mapsto(z',s'))).
```

They are equal because both operations are pure on the same label and captured
revision. Consequently a dead query projection constructs zero owned child
foci, while every live existing edge constructs exactly one. This law is the
formal basis of `DictZipper::filter_map_children`: it is an allocation-order
optimization, not an additional pruning rule.

These laws make zippers an optimization boundary. A backend may use a borrowed
node cursor for a tight local walk, a persistent value zipper for suspension,
or a compact serializable continuation for paging, provided all three are
observationally equivalent. Path materialization can be delayed until a result
or witness needs it. Sibling enumeration can batch edge labels and reuse a
prepared query row. Parent contexts can be shared rather than cloning complete
paths. None of these representation choices may change snapshot revision,
child order where order is promised, exact key identity, or result ties.

### 7.4 Schedulers are operations over the same product

DFS, BFS, distance-layer, and best-first traversal are scheduling policies over
reachable product zippers. They are interchangeable only when the public
observation allows it:

- unordered exact range results permit any fair exhaustive schedule;
- lexicographic or ranked results require the corresponding deterministic
  priority and tie order;
- kNN requires an admissible lower-bound heap and may stop only when no queued
  bound can improve the exact retained neighbors; and
- a bounded schedule must return a continuation before exceeding its ledger,
  never silently truncate.

Thus a scheduler optimization is a refinement theorem, not merely a container
swap. Its proof relates pending frontier, emitted prefix, exact tie order,
resource ledger, and immutable snapshot at every step.

## 8. Online semantics, stability, and stack safety

An online machine fixes a finite query and consumes an unknown number of target
labels. It reports the exact or tagged bounded status of every finite prefix.
This is a coalgebraic transition system of the form:

```math
\operatorname{step}:S\times A\longrightarrow
\operatorname{Outcome}(O\times S).
```

For query length $`m`$, maximum target lookback $`r`$, maximum cells per
retained generation $`w(q,\tau)`$, and continuation context $`\kappa`$,
the rolling-recurrence construction in ORC S1 bounds **retained cells** by:

```math
M_{\mathrm{cells}}(t)\leq M_{\mathrm{query}}(m)
+\mathcal{O}((r+1)w(q,\tau)+r+|\kappa|)
```

for every consumed prefix length $`t`$ when this context is sufficient.
Dense rolling recurrences have $`w=\mathcal{O}(m)`$; sparse frontiers may
be smaller. A byte bound also requires bounded encodings of all cells,
labels, counters, and scratch. Exact unbounded counters require
$`\Theta(\log(t+1))`$ bits; fixed-width counters require checked overflow.
No universal cutoff-only width is claimed for kernels with zero-cost paths.

There are two deliberately different resource profiles:

- **stream machine:** current and next generations, bounded lookback, bounded
  scratch, and no historical state arena or unbounded transition cache;
- **search session:** bounded arena, cache, queue/stack, results, witnesses, and
  continuation because dictionary branches may revisit representations.

Each transition is transactional: validate and preflight checked work and
allocation limits, construct into scratch, then commit. A rejected step leaves
the prior state observable and unchanged. Chunking a stream cannot change the
result, and resuming a paused dictionary search must equal uninterrupted
execution over the same immutable snapshot.

All closures and traversals are iterative. Exact zero-input closure requires
a termination argument, such as a finite acyclic dependency rank, or a proved
terminating fixed-point algorithm. A bounded worklist guarantees a resource
limit, not completion: exhaustion returns an explicit incomplete outcome.
Dictionary DFS uses a bounded heap stack rather than the process call stack. Its memory can still
grow with live dictionary depth, so stack safety is not mislabeled as constant
heap memory.

This contract does not define a distance between completed infinite sequences.
It defines stable processing of every finite prefix of an unknown-length target
against one fixed finite query. Bilaterally growing exact histories require a
separate windowed or infinite-path semantics.

## 9. Metric qualification is a separate theorem

The conceptual engine is an **ordered-cost automaton**; this phrase does not
name a currently exported universal Rust trait. A sealed audited metric marker
requires proofs, on the exact documented domain or quotient, of:

```math
d(x,y)\geq0,
\qquad d(x,y)=d(y,x),
\qquad d(x,y)=0\Longleftrightarrow x\sim y,
\qquad d(x,z)\leq d(x,y)+d(y,z).
```

Here $`\sim`$ is ordinary equality for strict metrics and the declared quotient
relation for ERP or discrete Fréchet. These laws are necessary for algorithms
whose correctness uses metric geometry. They are not required for synchronized
trie traversal, an admissible interval lower bound, or exact leaf verification.

Qualification belongs to the untruncated mathematical measure. Rounded
machine outputs and cutoff-saturated costs need their own numerical theorem
before triangle-based consumers may use them; exact recurrence correspondence
alone supplies no such theorem. ORC Section 12.3 gives both failure controls.

The separation produces two important controls:

- the general banded-DTW family may pass product, online, resource, and
  recurrence-correspondence gates but has no blanket metric qualification;
  a restricted instance, such as zero-width alignment on a fixed-length
  domain with a metric ground cost, can have a separate metric proof;
- raw ERP and raw Fréchet remain pseudometrics, while their gap-value and
  consecutive-stutter quotients may receive a metric-qualified wrapper.

Fixed multichannel composition pulls component metrics back along fixed maps:

```math
D(X,Y)=\sum_{c=1}^{C}w_c\,d_c(S_cX_c,S_cY_c),
\qquad w_c\geq0.
```

Fold-local transforms $`S_c`$, channel identities, and weights are fixed for
every compared pair, with zero-weight terms omitted. This always gives a
pseudometric. It is a strict metric exactly when the positive-weight maps
jointly separate inputs modulo each component's zero-distance relation.
Positive weights cannot repair noninjective joint transforms; a zero weight
need not break a metric if the remaining channels still separate inputs.
Pair-dependent missing-channel renormalization is not covered. ORC M2–M3
prove the sum, maximum, and zero-quotient constructions.

## 10. Applicability matrix

| Family | Ordered algebra | Residual representation | Required context | Qualification |
|---|---|---|---|---|
| standard Levenshtein | min-plus integers | positional error antichain | edit variant | metric |
| weighted strings | min-plus exact costs; ordered transformations for rounded execution | weighted positional frontier | operation/continuation kind | shortest-script closure, reversible equal costs, and separation; restricted alignments need an independent triangle proof |
| MSM | min-plus real | sparse query-row frontier | preceding target point or interval | metric for lawful positive split/merge cost |
| ERP | min-plus real | sparse query-row frontier | gap configuration | metric on the gap-value quotient |
| unit-grid TWED | min-plus real | query-row frontier | preceding target point and depth | metric under lawful positive stiffness on uniform grids |
| timestamped TWED | min-plus real | timestamp-aware frontier | preceding value/time and typed units | cumulative-boundary metric transfer requires first time strictly after the origin; the committed scalar raw API does not enforce that restriction, while ORC 11.5 gives its zero-distance counterexample |
| scalar/vector discrete Fréchet | min-max | bottleneck row frontier | current point/interval | metric on the consecutive-stutter quotient when the ground metric is certified |
| banded DTW | min-plus real | band-restricted row frontier | band/depth and current label | general family nonmetric; separately proved restrictions possible |
| Soft-DTW | smooth log-sum-exp recurrence | rolling dense score rows | bounded DP history | analysis-only; idempotent antichain elimination does not apply |

The metric column concerns ideal arithmetic on the stated domains; it does not
certify rounded triangle inequalities. Metricity is neither necessary nor
sufficient for compact realization. A new score enters the generic
architecture only after its residual, transition, cutoff, context, and resource
contracts are defined and proved.

## 11. Theory-directed optimization

The theory is intentionally constructive: each semantic equivalence or order
law identifies a concrete optimization and its correctness condition.

ORC Section 13 gives sixteen explicit rewrite rules and a terminating
certificate-producing optimizer design. Section 10 specifies the input
measure interface, realization levels, intermediate representation, derivation
algorithm, and local certificate checker. These are mathematical and compiler
contracts, not a claim that a generic production compiler is already exported.

| Theorem or law | Optimization | Required control |
|---|---|---|
| residual representation | discard consumed target history | every future score factors through retained state |
| residual equivalence | exact state minimization/interning | collision-checked structural equality or a proved complete quotient |
| continuation simulation | antichain subsumption | local rule implies the suffix-quantified preorder |
| observation congruence | characteristic-class transition cache | equal classes induce exactly equal successors |
| cutoff monotonicity | dead-state and subtree pruning | no lawful completion can lower the bound beneath cutoff |
| abstract lower simulation | interval/box traversal | exact verification covers every concrete collision member |
| point-abstraction exactness | specialized point transitions | exact-state embedding and singleton labels commute with transitions |
| dictionary-relative dominance | stronger local normalization | compare only suffixes accepted at this focus; include snapshot and focus in scoped cache keys |
| product reachability | on-demand construction | transitions arise only from inspected reachable edges |
| ORC R2a | adaptive dense/sparse representation | every conversion preserves the reachable machine residual and promised witness |
| zipper navigation laws | opaque native focus, projection-before-child, shared parent arena | snapshot/path/finality observations remain equal |
| sibling independence | prepared rows and batched child labels | scratch is reset transactionally between labels |
| coalgebraic state sufficiency | current/next generations and cache reclamation | chunk partitions and long prefixes are observationally equal |
| scheduler refinement | select DFS, BFS, layers, or best-first by query | completeness, order, cutoff, and continuation obligations remain true |
| witness congruence | compact parent operation IDs and delayed replay | canonical replay produces the promised exact cost and tie key |

**Schema RS-1 (representation switching).** For a dense and sparse frontier
that each satisfies residual correspondence, an adaptive scheduler may switch
at a reachable prefix only through a conversion satisfying ORC R2a. This is a
per-state obligation, including cutoff and finite-output tags. A density
threshold chooses *when* to convert; it supplies no correctness evidence. A
conversion that runs out of its declared storage must preserve the old
committed state or return a tagged incomplete outcome. In the confirmation
experiment, force switches at every possible prefix, at no prefix, and around
each density threshold, then compare the entire machine output and any
promised witness with an independent recurrence.
The [Rocq adaptive-event theorem](../verification/temporal_automata/theories/OrderedTheoryRefinements.v)
proves the generic claim for arbitrary switch schedules. A Rust converter is
still an instance obligation because no optimization implementation has been
introduced in this campaign.

**Schema CL-1 (monotone sparse lookup).** Suppose a sparse source frontier has
strictly ascending row indices and the rows evaluated by one transition are
ascending. If each evaluated row $`r>0`$ requests source rows $`r-1`$ then
$`r`$ (and row zero requests zero), the full lookup request stream is
nondecreasing. A cursor that advances while its source row is smaller than the
request therefore returns exactly the same present cost or infinity as binary
search. Its total comparisons are bounded by a constant times the number of
requests plus the source length. The scheduled-row builder and vertical-closure
merge must establish the ascending-row premise, and the proof must include
repeated requests, absent rows, row zero, and checked row conversion. This
lemma applies to the timestamped TWED sparse transition's source lookups;
it says nothing about the cost of constructing the schedule or closing rows.
The same [Rocq file](../verification/temporal_automata/theories/OrderedTheoryRefinements.v)
proves cursor answers equal independent lookups for sorted sources and
nondecreasing requests, derives the request order from ascending evaluated
rows, proves that a scheduled/vertical minimum advances past the prior row,
and gives an amortized upper bound of source length plus request count on
cursor comparisons. The executable scheduler must still establish its
sorted-row and cursor-update correspondence with this model.

**Schema RA-1 (bound-first accounting).** Let a concrete candidate bound be
admissible for every original in its collision bucket. If the bound is strictly
above an inclusive cutoff, skipping exact DP for that original preserves the
set of completed exact results. Charge the bound's actual work before
evaluation, and preflight each subsequently admitted DP step before executing
or committing it. A failed preflight leaves candidate position, results, and
continuation at the last committed boundary. The charged-work trace and tagged
resource outcome may differ from a full-DP precharge policy at the same limit;
the theorem promises equal ordered results **when both runs complete**, plus
sound tagged incompleteness and resumability under the new policy. It does not
claim budget-observational equivalence between the two policies. An uncharged
bound or a bound that is merely correct on the bucket representative violates
the premises.
The [Rocq candidate model](../verification/temporal_automata/theories/OrderedTheoryRefinements.v)
proves completed-result equality over lists of candidates, atomic rejection
of an unaffordable phase, a paused exact phase's resumption without repeating
the bound, and charged-work ceilings. Its finite-cost carrier
does not prove the Rust binary64 lower-bound relation or whole-session ledger
correspondence.

**Schema RA-2 (private sparse-edge accounting).** Decompose one sparse
transition into a finite ordered list of logical primitives, each with a
declared work charge and deterministic state update. Charge before applying a
primitive. Keep the partially computed successor private until every primitive
finishes; a failed preflight leaves the private cursor and public predecessor
unchanged. The [Rocq private-edge model](../verification/temporal_automata/theories/OrderedTheoryRefinements.v)
proves that any sequence of affordable steps and budget pauses preserves a
prefix-of-primitives invariant, never publishes a partial successor, and, once
complete, publishes exactly the uninterrupted fold with precisely the sum of
executed charges. The Rust instance must account for allocation, checked
arithmetic, and every actual primitive; a model of abstract work units alone
cannot establish those correspondences.

Let $`E_R`$ be the number of inspected edges in the reachable live product,
$`S_R`$ the number of distinct canonical query states, $`C_R`$ the number of
distinct observed state/class transitions, $`W`$ the maximum cells per live generation,
$`H`$ the maximum live DFS depth, and $`V`$ the number of full-precision
candidates verified. A useful implementation-sensitive accounting is:

```math
T=\mathcal{O}\!\left(
E_R\,c_{\mathrm{inspect}}
+C_R\,c_{\mathrm{step}}(W,r,|\kappa|)
+V\,c_{\mathrm{exact}}
+T_{\mathrm{results}}
\right),
```

```math
M_{\mathrm{search}}=
M_{\mathrm{query}}+\mathcal{O}\!\left(
S_R((r+1)W+r+|\kappa|)+C_R c_{\mathrm{entry}}+Q+H+R+\Omega
\right),
```

where inspection includes dictionary navigation, observation construction,
cache lookup/comparison, and scheduler maintenance; step cost includes
generation, closure, normalization, hashing, and exact arena comparisons.
The result term accounts for materialization and required sorting.
$`c_{\mathrm{entry}}`$ includes the observation key and cached successor.
$`Q`$ is pending scheduler storage, $`R`$ retained results, and
$`\Omega`$ bounded witness/continuation storage. The memory expression counts
cells; convert all variable-sized encodings to bytes for an allocation claim.
This is not a claim of
dictionary-size independence: a query with weak pruning may inspect the whole
dictionary. It is an accounting that reveals whether time is spent generating
new residuals, revisiting known observations, verifying quantization
collisions, or maintaining the scheduler.

For a stream machine the stronger retained-memory equation is:

```math
M_{\mathrm{stream}}=
M_{\mathrm{query}}+\mathcal{O}((r+1)W+r+|\kappa|
+\mathrm{scratch}+C_{\mathrm{bounded}}),
```

as a cell count independent of consumed prefix length under the hypotheses of
Section 8. Byte bounds additionally depend on encodings. Search arenas and
their $`S_R`$ term do not belong in this profile.

Optimization work should therefore measure more than wall time:

- constructed and reused state IDs;
- transition-cache hits, misses, and exact collision comparisons;
- generated, closed, and subsumed positions per observed edge;
- reachable dictionary edges inspected and subtrees pruned;
- peak stream generations, search states, queue frames, and path bytes;
- abstract candidates, quantization-collision members, and exact survivors;
- exact-verifier calls and witness operations; and
- tagged budget exits, continuation bytes, allocations, and peak resident set.

A performance change is accepted only after the slow recurrence oracle and
formal-model-aligned properties remain equal, its relevant mutant remains
rejected, and benchmarks show improvement in the work dimension the theorem
predicts. Examples include packed IDs, `SmallVec` frontiers, generation-stamped
scratch, sparse/dense adaptive stepping, SIMD dominance checks, interval-table
precomputation, and dictionary-native cursor batches. None is allowed to alter
exact equality, cutoff membership, deterministic ties, or fail-closed limits.

## 12. Formal theory and executable gates

### 12.1 Normative theorem-schema registry

The identifiers below name obligations, not blanket claims that every instance
has already discharged them. The
[`FORMAL_VERIFICATION_MANIFEST.tsv`](../verification/FORMAL_VERIFICATION_MANIFEST.tsv)
is authoritative for current proof status.

| ID | Schema | Required evidence |
|---|---|---|
| CA-1 | cutoff saturation preserves every at-or-below-cutoff output | ordered-algebra proof plus over-cutoff mutant |
| DC-1 | a complete live dependency cut bounds every unresolved accepting path | checked finite-step cut theorem and bypass-row mutant; concrete successor completeness, endpoint verification, cost monotonicity, and storage correspondence remain instance gates |
| FA-1 | finite finalizer output is emitted only when it remains within cutoff | exact finalization plus scheduler-boundary admission proof and mutant |
| RM-1 | transitions realize weighted left residuals | kernel recurrence correspondence |
| RM-2 | reachable deterministic residual quotient is minimal | generic Moore-machine proof; no NFA-minimality claim |
| SP-1 | semantic dominance permits atom elimination | suffix-quantified continuation proof |
| SP-2 | executable forward simulation implies semantic dominance | kernel/variant-specific simulation |
| AN-1..3 | normalization preserves behavior, is idempotent, and is permutation-independent | formal canonicalization plus randomized predecessor order |
| AW-1 | complete normalization under the executable preorder is bounded by its quotient width | ORC F2–F3 and instance analysis; semantic width needs a complete semantic-order normalizer |
| OC-1 | transition-congruent observation quotient preserves behavior | label-class correspondence |
| CC-1 | exact observed-transition cache refines recomputation | exact state decode, complete cache key, pure transition |
| CC-2 | eviction of complete cache entries is behaviorally transparent | recomputation equivalence |
| AI-1 | abstract transitions lower-simulate all concrete paths | interval/box transformer induction |
| AI-2 | an over-cutoff abstract bound safely rejects a subtree | AI-1 plus cutoff-safe extension |
| AI-3 | singleton labels reproduce transitions from exactly embedded states | commuting embedding equation and mutant for previously lost precision |
| AP-1 | greater abstract precision cannot lose completed exact results | precision monotonicity plus complete execution |
| EV-1 | candidate product plus collision retention and verification equals brute force | product completeness and exact verifier |
| MQ-1 | exact distance realization recognizes the cutoff ball | RM-1 and exact finality |
| MQ-2 | fixed nonnegative channel sums and maxima preserve pseudometrics; joint separation gives a metric | ORC M2–M3, fixed domains/maps/weights, and a separate numerical contract |
| ZP-1..7 | zipper snapshot, path, child, finality, clone, and continuation laws | backend conformance and product-focus model |
| GR-1..3 | bounded-lookback reclamation, prefix-independent retention, and generation-tag safety | recurrence dependency and ring-buffer refinement |
| RS-1 | adaptive dense/sparse switching preserves residual behavior | Rocq event-trace theorem; concrete conversion relation, machine outputs, and failure atomicity remain instance gates |
| CL-1 | sorted sparse source lookups admit a monotone cursor | Rocq lookup equivalence and comparison bound; executable scheduled/vertical row order remains an instance gate |
| RA-1 | bound-first charging preserves completed exact results and tagged resumability | Rocq candidate/result and preflight laws; per-original machine admissibility and whole-session ledger trace remain instance gates |
| RA-2 | private sparse-edge steps publish only a fully charged successor | Rocq prefix/refinement invariant under arbitrary page limits; Rust primitive inventory, checked arithmetic, and allocation remain instance gates |
| ST-1 | arbitrary stream chunking equals one uninterrupted run | coalgebraic composition plus executable property |
| TX-1 | rejected transition preflight leaves committed state unchanged | resource refinement and fault injection |
| PS-1..5 | reachability, completeness, soundness, scheduler independence, and lazy construction | product proof plus bounded lifecycle model |
| BF-1 | admissible cost/tie lower pairs certify ordered best-first kNN stopping | Rocq order kernel and finite TLA+ model, followed by machine-bound, coverage, and heap-correspondence instance proofs |
| BO-1..2 | complete outcomes are fail-closed and resumption equals uninterrupted execution | TLA+ lifecycle plus executable pages |

### 12.2 Proof dependency

The dependencies form several branches, not one chain requiring every
optimization. The main exact branch is:

```math
\text{measure semantics}\to\text{residual correspondence}
\to\text{exact dictionary product}
\to\text{bounded resumption}.
```

An ordered transformation algebra plus recurrence correspondence supports
simulation and cutoff proofs. A proved executable simulation supports frontier
normalization. Exact observation congruence supports caching independently of
stronger subsumption. Abstract lower simulation, complete collision retention,
and exact verification support an alternative candidate-product branch. ORC
K1–K2 and H1–H2 prove how exact morphisms and lower simulations compose.

Metric qualification and streaming stability are parallel branches:

```math
\text{untruncated measure}+\text{domain/quotient laws}
\to\text{metric marker}
\to\text{triangle-dependent consumers},
```

```math
\text{bounded lookback}+\text{generation tags}+\text{transactional limits}
\to\text{bounded retained cells}
\xrightarrow{\text{encoding bounds}}\text{bounded bytes}.
```

This structure prevents one local inequality, recurrence lemma, or metric proof
from being reported as verification of the entire query implementation.

The proof program proceeds in dependency order:

1. specify the measure and observation, prove kernel recurrence correspondence,
   and discharge the algebra laws actually used by its transformations;
2. prove the seed represents the empty prefix and one transition preserves the
   residual-representation relation;
3. prove exact duplicate merging and every stronger dominance rule preserve
   continuation behavior;
4. prove canonicalization is idempotent, permutation-independent, and exact;
5. prove abstract transitions lower-simulate every concrete label and point
   abstractions reproduce exact transitions;
6. prove every queued pair is reachable and every live edge is eventually
   explored unless the outcome is explicitly incomplete;
7. prove abstract candidate enumeration plus collision retention plus exact
   leaf scoring equals brute force;
8. prove collision-safe interning, checked counters, transactional budget
   rejection, pause/resume equivalence, stack safety, and prefix-independent
   stream retention;
9. separately prove the metric axioms for each qualified domain or quotient.

Rocq carries the semantic spine. Verus connects Rust-shaped arrays, arenas,
and checked transitions to those relations. Z3 and cvc5 discharge local
arithmetic obligations and pinned mutants. TLA+ checks scheduler lifecycle,
continuations, resource outcomes, and the impossibility of reporting an
incomplete or invalid search as complete empty.

Every executable instance also has independent-oracle properties: exact score
after every prefix, arbitrary chunk equivalence, interval lower bounds, point-
interval equality, brute-force range/kNN equality, hash/permutation
determinism, collision retention, witness replay, corruption failure, and
configured resource ceilings. Optimization is accepted only as a refinement
that preserves these observations.

## 13. Property and mutation program

Formal models state the unbounded mathematical obligations; executable
properties connect them to Rust, binary64 carriers, dictionary backends, and
resource failures.

### 13.1 Generic properties

1. Every online prefix equals an independent batch scorer.
2. Every arbitrary stream chunk partition equals uninterrupted execution.
3. Normalization is idempotent and permutation-independent.
4. Pruned and unpruned frontiers have equal exact outputs.
5. Equal canonical states reuse one ID; unequal states never do.
6. Forced fingerprint collisions do not merge unequal states.
7. Observation-cache hits equal direct recomputation.
8. Cache eviction changes no completed result.
9. Interval transitions lower-bound every represented sampled point.
10. Point intervals equal exact point transitions cell by cell.
11. More precise abstractions add no candidates in completed searches.
12. Exact product results equal brute force across dictionary backends.
13. Range membership is monotone in inclusive cutoff.
14. Every finite finalizer score is rechecked against the inclusive cutoff
    before public emission.
15. Every quantization-collision member is retained and exactly verified.
16. Post-warm-up retained online capacity has zero slope with stream length.
17. Rejected budget/resource transitions leave committed state unchanged.
18. Every advertised counter, queue, arena, cache, witness, and result limit is
    respected.
19. Resumed and uninterrupted searches agree for arbitrary page partitions.
20. Every witness replays to the exact returned cost.
21. Ties and result order are stable across hash seeds and snapshot reloads.
22. Dictionary zipper children equal the set of successful labelled descents.
23. Product zipper children equal the live intersection of dictionary and
    query successors.
24. Delayed path materialization equals eager root-to-focus reconstruction.
25. DFS, BFS, and best-first completed result multisets agree where their
    public ordering contracts permit comparison.
26. Forced dense/sparse switches after every reachable prefix agree with the
    machine recurrence, including exact output tags and promised witnesses.
27. Monotone-cursor source lookups equal binary search for every sparse row
    request, including absent, repeated, zero, and last-row requests.
28. Bound-first charging records all bound work and all executed DP work;
    budget exits preserve a resumable committed boundary and completed runs
    retain exact candidate order.
29. Every sparse-edge primitive is charged before private execution; a
    rejected preflight keeps the public predecessor and private cursor, and
    arbitrary page partitions publish the same completed successor.

### 13.2 Pinned mutants and negative controls

The suite must permanently reject:

- the MSM final cutoff omission;
- the zipper true-Damerau final-admission omission for query `"a"`, key `""`,
  and cutoff zero;
- hash-only state reuse;
- epsilon-based floating canonical equality;
- deletion of a live nonsubsumed atom;
- cross-context subsumption without a carry simulation;
- an incorrect interval endpoint or omitted feasible correlation;
- broken point exactness;
- pruning a cutoff-equal state;
- work charging after evaluation;
- committing before budget/allocation preflight;
- overflow, invalidity, or incompleteness converted to complete empty;
- dropping one quantization-collision original;
- buffering the consumed target prefix;
- accepting a stale generation tag;
- nondeterministic equal-cost witness selection;
- substituting unit-grid indices for explicit physical timestamps;
- an unbounded continuous-observation cache in streaming mode;
- deduplicating a DAWG product pair while losing a distinct path/value;
- allowing an unrestricted DTW family to inherit a metric marker;
- claiming strict separation after noninjective joint channel transforms;
- applying the fixed-composition theorem to pair-renormalized channels;
- using full semantic width for a duplicate-only normalizer;
- treating a point label as a repair for an already abstracted history;
- treating rounded or cutoff-saturated outputs as metrics without a proof;
- claiming a compact exact scorer from a constant-zero candidate filter;
- converting a dense frontier to a sparse frontier by density alone, without
  preserving its continuation behavior;
- advancing a sparse lookup cursor past a later smaller request; and
- charging a candidate lower bound after evaluating it, or using a bound
  proved only for the bucket representative; and
- publishing a sparse successor after only a charged prefix of its primitives.

## 14. Causal benchmark protocol

Wall-clock time is an outcome, not an explanation. Each benchmark binds commit,
toolchain, features, seed, kernel configuration, limits, snapshot identity, and
source checksums, then records the causal quantities predicted by the theory.

### 14.1 State and observation measures

- raw atoms generated, exact duplicates removed, and atoms removed by each
  named simulation rule;
- frontier width distribution and maximum $`W`$;
- canonical states, total retained atoms, ID reuse, and fingerprint-bucket
  lengths;
- observation classes per state, cache hits/misses/dead hits/evictions, and
  allocations/bytes per entry; and
- cache size and retained-memory slope against stream length.

### 14.2 Abstract-product measures

- abstract bound versus exact cost, including absolute gaps when exact cost is
  zero;
- prefix-, cell-, and leaf-bound prune counts;
- candidate amplification, exact-verification ratio, collision-bucket sizes,
  and exact-survivor count; and
- time spent tightening the abstract domain versus exact verification saved.

### 14.3 Product, zipper, and scheduler measures

- dictionary nodes/edges visited and $`E_R/E`$;
- live successor ratio, transition misses per inspected edge, and prepared-
  sibling reuse;
- maximum worklist, explicit DFS depth, zipper clone bytes, parent-spine/path
  bytes, and delayed materializations;
- queued states later rejected by a tighter kNN cutoff; and
- time to first result, time to completion, and order-normalization cost.

### 14.4 Streaming and witness measures

- current/next capacities, retained generation count, lookback and scratch high-
  water marks, and logical bytes versus prefix length;
- p50/p95/p99 per-symbol work and maximum work charged in one step;
- tagged incomplete transitions under fixed ceilings; and
- witness bytes, checkpoint count, replay work, exact replay time, and tie-group
  size.

Workloads vary query length, cutoff, branching, depth, DAWG sharing,
observation entropy, cache-adversarial label order, repeated zero-cost values,
quantization collisions, vector dimension, timestamp irregularity, DTW band,
and exact/near-cutoff/no-match distributions. Paired controls compare compact
IDs to owned states, cache off/on, duplicate-only to proved subsumption,
coarse/refined abstractions, sparse/dense transitions, DFS/best-first, and
witness off/bounded replay. RSS is reported with logical retained capacity and
allocation counts because allocator page retention can hide the true slope.

## 15. Incremental research program

The mathematical development and implementation program have separate status.
ORC supplies proofs of the generic realization, transformation, calculus,
normalization, categorical, contextual-product, geometry, synthesis, and metric
composition results. Its worked derivations and optimizer specification are
available now. The program below describes adoption and mechanization in
production kernels, whose actual status remains in the verification manifest:

1. **Definitions and traceability.** Stabilize vocabulary; inventory every
   automaton, operation, zipper, and scheduler; map each optimization to a
   theorem, property, mutant, and benchmark.
2. **Residual core.** Mechanize ORC R1–R4 and instantiate its Standard
   Levenshtein and temporal derivations against the production interfaces.
3. **Simulation and antichains.** Prove SP-1/SP-2, AN-1..3, and instance width
   bounds; retain duplicate-only fallback everywhere else.
4. **Observation congruence.** Prove characteristic-vector factorization,
   exact cache refinement, eviction transparency, and bounded policy for
   continuous observations.
5. **Abstract domains.** Formalize interval/box concretization, point
   embeddings, precision monotonicity, and relational timestamp candidates.
6. **Stable online machines.** Prove bounded lookback, generation tags,
   reclamation, chunking, and transactional steps; audit caches and witnesses
   for hidden history growth.
7. **Products and zippers.** Prove focus/navigation laws, reachable worklists,
   scheduler independence, DAWG path sensitivity, best-first stopping, bounded
   continuation, and snapshot binding.
8. **Kernel migration.** Use Standard/weighted strings as references, followed
   by ERP, scalar Fréchet, vector Fréchet, MSM, unit-grid TWED, timestamped
   TWED, and banded DTW as the nonmetric control.
9. **Refinement optimization.** Evaluate exact interning, bounded observation
   caches, packed IDs, sparse scheduling, adaptive sparse/dense transitions,
   refined abstractions, SIMD normalization, replayable witnesses, and
   scheduler specialization one at a time.
10. **Certified synthesis and optimization.** Implement the ORC Section 10
    derivation interface and Section 13 certificate-checking optimizer,
    using exact morphisms, lower simulations, and the directed residual
    discrepancy to validate each accepted transformation. Kernel-specific
    simulation discovery and decidable symbolic quotients require their
    own instance proofs.

Every migration retains a slower formal-model-aligned or independent matrix
oracle until correspondence, mutation, resource, stack, and causal performance
gates pass.

## 16. Implementation boundary

The implementation should consolidate narrow crate-private infrastructure, not
replace specialized hot recurrences with one dynamic abstraction:

- exact canonical state arena and compact IDs;
- collision-checked fingerprints;
- observed-label transition caches;
- reusable current/next scratch and bounded worklists;
- iterative bounded product schedulers;
- tagged stream and search outcomes;
- exact leaf-verifier adapters; and
- sealed, reviewed metric qualification.

Kernel-specific machines remain monomorphized. They own query preparation,
recurrence context, exact observation keys, legal epsilon reachability,
subsumption proofs, finality, and exact scoring. This preserves both semantic
clarity and optimization freedom.

## 17. Explicit nonclaims

- Every measure admits residual semantics; effective compact execution
  requires further evidence and does not follow from metric axioms.
- A symbolic real-valued temporal machine need not be finite-state in the
  strict automata-theory sense.
- A quantized interval product is not exact language intersection until every
  candidate original has been exactly verified.
- No numerical-tolerance or intuitive dominance rule is sound by default.
- Stack-safe traversal can still need bounded heap memory proportional to live
  dictionary depth.
- Stable fixed-query prefix processing is not an infinite-sequence distance.
- Fixed register count does not imply fixed bit space for exact arithmetic.
- A one-state lower filter does not prove a bounded-memory exact verifier.
- A TWED metric theorem with one boundary convention cannot silently certify
  another; strict timestamp order within a series does not exclude a zero-cost
  first sample at the sentinel origin.
- A budget-exceeded, invalid, overflowed, or approximate result is never a
  complete empty result.
- Soft-DTW does not inherit the idempotent minimum-antichain theory.
- Named proof islands do not imply that every historical MSM surface is fully
  formally verified.

## 18. References

- M. Droste, W. Kuich, and H. Vogler, editors, *Handbook of Weighted
  Automata*, Springer, 2009.
  [doi:10.1007/978-3-642-01492-5](https://doi.org/10.1007/978-3-642-01492-5)
- J. A. Brzozowski, “Derivatives of Regular Expressions,” *Journal of the ACM*
  11(4), 1964.
  [doi:10.1145/321239.321249](https://doi.org/10.1145/321239.321249)
- L. Doyen and J.-F. Raskin, “Antichain Algorithms for Finite Automata,”
  *TACAS*, 2010.
  [doi:10.1007/978-3-642-12002-2_2](https://doi.org/10.1007/978-3-642-12002-2_2)
- P. Cousot and R. Cousot, “Abstract Interpretation: A Unified Lattice Model
  for Static Analysis of Programs by Construction or Approximation of
  Fixpoints,” *POPL*, 1977.
  [doi:10.1145/512950.512973](https://doi.org/10.1145/512950.512973)
- J. J. M. M. Rutten, “Universal Coalgebra: A Theory of Systems,”
  *Theoretical Computer Science* 249(1), 2000.
  [doi:10.1016/S0304-3975(00)00056-6](https://doi.org/10.1016/S0304-3975(00)00056-6)
- F. W. Lawvere, “Metric Spaces, Generalized Logic, and Closed Categories,”
  *Rendiconti del Seminario Matematico e Fisico di Milano* 43, 1973.
  [doi:10.1007/BF02924844](https://doi.org/10.1007/BF02924844)
- G. Huet, “Functional Pearl: The Zipper,” *Journal of Functional
  Programming* 7(5), 1997.
  [doi:10.1017/S0956796897002864](https://doi.org/10.1017/S0956796897002864)
