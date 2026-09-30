# Ordered Residual Calculus

**Status:** mathematical specification and synthesis/optimization design ·
**Audience:** researchers and implementers · **Evidence:** proofs in this
document, separately identified executable checks, and linked existing formal
artifacts; this manuscript is not a claim of end-to-end mechanization

**Ordered Residual Calculus (ORC)** studies how a measure becomes an automaton,
how that automaton can be represented, and which transformations preserve its
meaning. Its organizing construction is:

```math
\text{measure}\longrightarrow\text{residual behavior}
\longrightarrow\text{certified representation}
\longrightarrow\text{optimized execution}.
```

Every measure determines residual behavior. Computability, finite state,
bounded registers, safe pruning, and metricity are additional properties, each
with its own evidence. This makes the theory applicable to arbitrary measure
specifications without promising that every specification has an efficient
automaton. A recurrence is one way to obtain a representation, not the
definition of the measure or of its automaton.

The library's [lazy ordered-cost product automata](lazy-ordered-cost-product-automata.md)
are an operational specialization: finite minimum-cost frontiers, certified
simulations, observed transitions, and dictionary synchronization. The
calculus also describes machines outside that specialization, including
non-idempotent Soft-DTW and history-retaining realizations of opaque scorers.

## 1. Domains, observations, and claims

Let $`A`$ be an input alphabet, possibly infinite, and $`A^*`$ its finite
words. Finite real-valued series use real or machine-number labels; timestamped
series use typed value/time labels. Other objects enter through an explicit
encoding and its decoding/identity contract. Fix a query $`q`$ and parameters
$`\theta`$. A measure supplies $`F(y)=d_\theta(q,y)`$.

The general residual construction uses any output set $`O`$. Ordered-cost
results additionally use a totally ordered carrier $`C`$ with greatest value
$`\top`$. The geometric results use $`[0,\infty]`$. These are nested
assumptions, not interchangeable descriptions of every measure.

Domain validation is an observation. Invalid timestamps, malformed encodings,
NaN, and resource exhaustion are not large distances. Either work on a common
lawful continuation domain or include the domain recognizer and a distinct
invalid output in the machine. A comparison of costs never licenses merging
states with different valid continuations. Below, cost equations quantify over
the common lawful domain; total-machine equations also preserve validity.

For a finite inclusive cutoff $`\tau`$, define

```math
T_\tau(c)=\begin{cases}c&c\le\tau,\\\top&c>\tau.\end{cases}
```

Three observation functions give three different equivalences: the complete
score $`F`$, the bounded score $`T_\tau F`$, and membership
$`[F\le\tau]`$. A membership certificate alone cannot preserve reported
distances. A score certificate alone cannot preserve a canonical alignment
witness. Over-cutoff and unavailable mathematical paths may share a cost
sentinel under the finite-cutoff observation; operational failures retain
their distinct tags. An infinite cutoff disables that identification wherever
infinite scores themselves are observable.

| Claim | Required evidence | What it authorizes |
|---|---|---|
| Semantic residual | A specified function and input decomposition | A canonical mathematical machine |
| Effective realization | Computable seed, step, and output with correspondence | Execution on every finite lawful input |
| Bounded-register realization | Bounded number of retained values and effective updates | Reclaiming old generations |
| Bounded-bit realization | Bounds on encodings of every retained value | An actual storage bound |
| Finite deterministic realization | Finite effective state representation and transitions | Finite-state algorithms |
| Abstract realization | A lower simulation and complete concretization | Candidate pruning followed by exact scoring |
| Metric qualification | Domain-specific untruncated metric or quotient proof | Consumers justified by metric geometry |

A finite number of exact-real registers is a mathematical memory model, not a
Rust allocation bound. An exact natural-number counter of consumed length
$`t`$ needs $`\Theta(\log(t+1))`$ bits. A fixed-width machine counter has
bounded storage but must reject overflow unless the counter can be removed or
saturated by a semantic proof. Work counters, caches, and witnesses count too.

## 2. Residual realization and its limits

### 2.1 Universal construction

Define the left residual, also called an input derivative, by

```math
(\partial_uF)(z)=F(uz).
```

The residual machine starts in $`F`$, has reachable states
$`\{\partial_uF:u\in A^*\}`$, outputs $`o(G)=G(\varepsilon)`$, and
steps by $`\delta(G,a)=\partial_aG`$. It is deterministic, but need not be
finite or effectively representable.

**R1 — realization.** After reading $`u`$ the residual machine is in
$`\partial_uF`$ and outputs $`F(u)`$.

**Proof.** The empty prefix leaves $`F`$. If the claim holds for $`u`$,
then for every $`z`$,
$`\partial_a(\partial_uF)(z)=F(uaz)=\partial_{ua}F(z)`$.
Induction proves the state claim, and evaluation at the empty word proves the
output claim. This proof uses no order, metric, or semiring.

A concrete representation consists of a state set $`S`$, seed $`s_0`$,
transition $`\delta_S`$, output $`o_S`$, and interpretation
$`b:S\to O^{A^*}`$ satisfying

```math
b(s_0)=F,\qquad b(\delta_S(s,a))=\partial_a b(s),\qquad
o_S(s)=b(s)(\varepsilon).
```

**R2 — representation certificate.** These three equations imply that every
finite-prefix output equals the specification. The same induction as R1 gives
$`b(\delta_S^*(s_0,u))=\partial_uF`$. They are the central contract for
deriving new automata.

**R2a — adaptive representation switching.** Let two representations
$`S_D,S_P`$ satisfy R2 for the same score and output carrier. For each
permitted switch $`i\to j`$, a conversion $`c_{ij}`$ must preserve behavior
on every reachable state at which that switch can occur:

```math
b_j(c_{ij}(s))=b_i(s).
```

Then an execution that interleaves ordinary transitions with any finite number
of permitted conversions still represents the residual of its consumed word.
Induct on execution events: a transition uses R2's derivative equation, and a
conversion leaves the interpreted residual unchanged. No transition
commutation or reverse conversion is needed for the chosen schedule. Conversion
failure must leave the old state available or return a tagged incomplete
outcome. A pair of correct seeds alone cannot certify a conversion: the
intermediate states may have different encodings or finite-cutoff information.
For binary64 kernels, $`b_i`$ denotes the specified **machine** outputs,
including operation order, cutoff, invalidity, and tie/witness observation
when promised; equality of ideal real recurrences is insufficient.
The [Rocq event-trace proof](../verification/temporal_automata/theories/OrderedTheoryRefinements.v)
allows arbitrary finite interleavings of consumption and conversions under
these per-transition and per-conversion equations. A concrete converter still
has to discharge the equations for its own machine carrier.

### 2.2 Minimality and effective construction

**R3 — deterministic residual minimality.** Every reachable deterministic
Moore machine realizing $`F`$ has a surjective behavior-preserving map onto
the reachable residual machine. Thus a finite reachable Moore realization
exists exactly when the residual index is finite, as an existence statement.

**Proof.** Map a state reached by $`u`$ to $`\partial_uF`$. If two words
reach the same state, determinism gives the same output after every suffix, so
the map is well-defined. Every residual has a reaching word, proving
surjectivity. Reading one label commutes with the map by R1, and empty-suffix
output is preserved. Conversely, a finite residual set is itself a finite
Moore realization.

Turning this existence result into an algorithm needs effective states, label
classification, equality, and transitions. The theorem does not decide these
properties for an arbitrary formula. It does not establish minimal weighted
NFAs, minimal register counts, or minimal arithmetic circuits. For example,
$`F(a^n)=n`$ has infinitely many Moore residuals but a one-register additive
realization. Structural interning need not discover all behavioral equality.

For a computable score with effective input/output representations, the
history machine $`S=A^*`$, $`\delta(u,a)=ua`$, $`o(u)=F(u)`$ discharges
R2 directly. Its storage grows with the retained encoding of the prefix. A
compact realization is an improvement over this construction, not a premise
silently imposed on arbitrary measures.

### 2.3 Metricity cannot supply a finite automaton

**R4 — arbitrary-ball obstruction.** Let
$`L\subseteq A^*\setminus\{\varepsilon\}`$. Define

```math
d_L(x,y)=\begin{cases}
0&x=y,\\
1&\text{one endpoint is }\varepsilon\text{ and the other is in }L,\\
2&\text{otherwise}.
\end{cases}
```

This is a metric. Symmetry and identity are immediate. If a triangle has
three distinct vertices, its two right-hand distances sum to at least two,
while its left-hand distance is at most two; repeated vertices give equality
or a zero left side. Its radius-one ball at the fixed query $`\varepsilon`$
is $`\{\varepsilon\}\cup L`$.

For $`L=\{a^nb^n:n\ge1\}`$, prefixes $`a^i`$ and $`a^j`$ with
$`i\ne j`$ are distinguished by suffix $`b^i`$. Hence even this fixed-query,
fixed-cutoff metric has infinitely many membership residuals. An undecidable
$`L`$ also gives a noncomputable metric. These are exact boundaries on
universal synthesis, not failures of the residual construction.

The existence of a small abstract filter is a different question. For every
nonnegative $`F`$, the one-state zero filter satisfies $`0\le F`$.
It retrieves every candidate and says nothing about exact residual size.
Filter usefulness must be measured by pruning or proved precision bounds;
soundness alone does not imply useful compression.

## 3. An algebra of ordered cost transformations

### 3.1 The transformation semiring

Let $`\mathcal E(C)`$ contain the monotone functions $`f:C\to C`$ that
preserve $`\top`$. Define

```math
(f\oplus g)(c)=\min(f(c),g(c)),\qquad
(f\circ g)(c)=f(g(c)),\qquad
\mathbf 0(c)=\top,\qquad\mathbf 1(c)=c.
```

**C1 — transformation algebra.**
$`(\mathcal E(C),\oplus,\circ,\mathbf0,\mathbf1)`$ is an idempotent
semiring, generally noncommutative under composition.

**Proof.** Pointwise minimum of monotone top-preserving maps again has those
properties. The same is true of composition. Minimum supplies the commutative
idempotent additive monoid and function composition supplies the multiplicative
monoid. For a monotone map on a chain,
$`f(\min(x,y))=\min(f(x),f(y))`$: choose whichever of $`x,y`$ is smaller
and use monotonicity. This proves distribution through a minimum on the input
side; distribution on the output side is pointwise evaluation. Finally,
$`f(\top)=\top`$ and the constant nature of $`\mathbf0`$ give absorption
on both sides.

On a partially ordered carrier, replace monotonicity by preservation of the
chosen finite meets, including the empty meet. Monotonicity alone is not
enough there. Multiobjective costs consequently require their own lattice or
Pareto-frontier contract; the scalar-chain theorem cannot be reused unchanged.

Path order is explicit. If updates $`f`$ then $`g`$ occur, their combined
effect is $`g\circ f`$. Examples include exact addition, bottleneck extension,
and rounded addition:

```math
f_w(c)=c+w,\qquad f_w(c)=\max(c,w),\qquad
f_w(c)=\operatorname{fl}(c+w).
```

The rounded instance uses canonical nonnegative binary64 values plus positive
infinity and a specified round-to-nearest operation. Each individual update is
monotone and top-preserving, including overflow to infinity in that numeric
model. A runtime that exposes overflow as an error has an additional outcome
semantics and cannot replace it by a dead state without an outcome certificate.

Composing rounded functions is associative **as composition**, while replacing
$`f_b\circ f_a`$ by $`f_{a+b}`$ is generally false. Function expressions
also need not admit constant-size representations. C1 supplies laws, not a
bound on the size or evaluation time of composed functions.

### 3.2 Cutoff quotient

**C2 — safe internal saturation.** If each update obeys

```math
T_\tau\circ f\circ T_\tau=T_\tau\circ f,
```

then inserting saturation after every update and minimum preserves the final
saturated value of any finite minimum/update computation.

**Proof.** Saturation preserves minimum and is idempotent. Use the displayed
equation at every update node and minimum preservation at every choice node,
inducting over the computation DAG. A finite ranked epsilon closure is such
a DAG.

Inflation $`c\le f(c)`$ is sufficient: a value already above cutoff stays
above it; values within cutoff are unchanged before the update. Inflationary
maps are closed under minimum and composition. Merely observing an over-cutoff
final score for the current prefix is insufficient: distance from `"abc"` to
`"a"` is two, but extending the target by `"bc"` gives zero.

For $`\tau'\le\tau`$, $`T_{\tau'}T_\tau=T_{\tau'}`$. Lowering a
construction cutoff is therefore sound when internal saturation and the state
representation commute as certified. Raising it cannot recover discarded
costs. Cache reuse across cutoffs needs this conversion theorem or a key that
keeps the construction cutoff distinct.

### 3.3 Scalar algebras and numerical claims

Exact min-plus and min-max carriers recover the usual scalar semiring
specializations. They permit scalar factoring, weighted concatenation, and
matrix constructions under the relevant semiring laws. Cost transformations
provide a smaller set of executable assumptions when scalar associativity is
unavailable. The existing `CostMonoid` API need not become a public semiring API
to use these proofs.

Always distinguish: mathematical equality over reals; exact integer or
rational evaluation; a specified binary64 expression graph; and certified
enclosures of a mathematical result. A real lower-bound proof transfers to a
rounded implementation only through a numerical correspondence theorem. A
bound computed by the same monotone rounded operations may be certified
directly; unrelated formulas do not inherit that fact from their real proofs.

## 4. Residual differential calculus

Pointwise operations on behaviors give constructors for machines. Write
$`F\simeq G`$ for equality at every lawful finite input under the chosen
observation. This is a semantic equation, not an assertion that two state
encodings are identical.

**D1 — derivative laws.** For any pointwise operation $`h`$ of finite arity,

```math
\partial_\varepsilon F=F,\qquad
\partial_{uv}F=\partial_v(\partial_uF),\qquad
\partial_a h(F_1,\ldots,F_k)
=h(\partial_aF_1,\ldots,\partial_aF_k).
```

**Proof.** Evaluate both sides at $`z`$; each side applies $`h`$ to
$`F_1(az),\ldots,F_k(az)`$. The word laws follow from associativity of
concatenation.

Thus pointwise minimum, sum, maximum, fixed scaling, and output truncation
commute with derivatives wherever they are defined. Output truncation uses no
inflation hypothesis; internal pruning still requires C2. A machine for
$`h(F,G)`$ keeps the pair of component states, steps both on the same symbol,
and outputs $`h(o_F,o_G)`$. R2 follows immediately. Its retained state is the
sum of the component representations plus bounded combination context.

### 4.1 Weighted concatenation

In an exact scalar semiring with choice $`\oplus`$, multiplication
$`\otimes`$, zero $`\top`$, and identity $`e`$, define the Cauchy product

```math
(F\star G)(w)=\bigoplus_{uv=w}F(u)\otimes G(v).
```

There are only $`|w|+1`$ splits. This operation chooses an input split; it is
different from evaluating both machines on the whole input. Define left scalar
action $`(c\cdot G)(z)=c\otimes G(z)`$.

**D2 — product derivative.**

```math
\partial_a(F\star G)
=(\partial_aF)\star G\;\oplus\;F(\varepsilon)\cdot\partial_aG.
```

**Proof.** Partition the splits of $`az`$ into those with empty left word and
those with left word $`au`$. The first contributes
$`F(\varepsilon)\otimes G(az)`$; the second contributes the splits of
$`z=uv`$ to $`F(au)\otimes G(v)`$.

An executable constructor retains the current state of $`F`$ and one pair
$`(c,s_G)`$ for each split already seen. Initially the pair is
$`(F(\varepsilon),s_{G,0})`$. On a label, advance every existing
$`s_G`$, advance $`F`$, and add the new pair
$`(o_F(s'_F),s_{G,0})`$. Output
$`\bigoplus_{(c,s_G)}c\otimes o_G(s_G)`$. Induction over split positions
proves correspondence. This direct constructor may retain a growing number
of splits; merging equal $`G`$ states or proving a bounded weighted quotient
is an additional optimization, not a consequence of D2 alone.

The zero series is constantly $`\top`$. The concatenation identity is $`e`$
on the empty word and $`\top`$ elsewhere. Associativity of concatenation
follows by enumerating all three-way splits and applying scalar associativity
and finite distributivity.

For a **proper** series, $`F(\varepsilon)=\top`$, define $`F^*`$ by
choice over decompositions into nonempty factors, with empty decomposition
worth $`e`$. Every fixed word has finitely many such decompositions. Then

```math
F^*(\varepsilon)=e,\qquad
\partial_aF^*=(\partial_aF)\star F^*.
```

The proof separates the first nonempty factor. This guarded iteration does
not require an infinite-path semantics. Unguarded epsilon cycles require an
additional closure theory. Finite syntax does not guarantee that repeatedly
constructing derivatives will discover finitely many distinct states.

### 4.2 Independent and shared alignments

Pointwise sum of two optimum scores permits independent witnesses:

```math
\min_\pi c_1(\pi)+\min_\rho c_2(\rho)
\le\min_\pi(c_1(\pi)+c_2(\pi)).
```

Equality is not automatic. Two possible alignments with cost pairs
$`(0,10)`$ and $`(10,0)`$ give zero on the left and ten on the right.
The pair-of-machines constructor computes the independent expression.
Replacing it by a shared alignment changes the measure unless a common-optimum
or other equality proof is supplied.

## 5. Backward budget calculus

Let $`B\subseteq C`$ be downward closed. Define

```math
\operatorname{Pre}_f(B)=\{c:f(c)\in B\}.
```

**B1 — backward composition.** Preimages of downward-closed sets under
monotone maps are downward closed, and

```math
\operatorname{Pre}_{g\circ f}(B)
=\operatorname{Pre}_f(\operatorname{Pre}_g(B)).
```

**Proof.** If $`c\le d`$ and $`f(d)\in B`$, monotonicity gives
$`f(c)\le f(d)`$, hence $`f(c)\in B`$. Composition is elementwise:
$`g(f(c))\in B`$ exactly when $`f(c)\in g^{-1}(B)`$.

This is an adjunction: for arbitrary sets $`U,V`$,
$`f[U]\subseteq V`$ iff $`U\subseteq f^{-1}[V]`$. When restricted to
downsets, use downward closure of the direct image as the left adjoint. The
preimage construction remains exact when no greatest allowed scalar exists.

**B2 — alternatives.** On a chain, for downward-closed $`B`$,

```math
\operatorname{Pre}_{f\oplus g}(B)
=\operatorname{Pre}_f(B)\cup\operatorname{Pre}_g(B).
```

**Proof.** A minimum is one of its operands. It belongs to $`B`$ exactly
when at least one operand belongs to $`B`$. The other direction also uses
downward closure.

For exact nonnegative additive costs and $`B_\tau=[0,\tau]`$,
$`\operatorname{Pre}_{c\mapsto c+w}(B_\tau)=[0,\tau-w]`$ when
$`w\le\tau`$, and is empty otherwise. For bottleneck extension it is
$`B_\tau`$ when $`w\le\tau`$, and empty otherwise. For finite binary64
carriers, preimages can instead be computed with their exact rounded predicate;
real subtraction must not be substituted without proof.

On a finite acyclic continuation graph, assign a final vertex its allowed
terminal costs, pull each successor budget backward through its edge update,
and union alternatives in reverse topological order. B1 and B2 prove that a
vertex budget contains exactly the costs admitting some accepted continuation.
A superset approximation of that budget is safe for rejection: only costs
outside the superset can be discarded. This reverses the numeric inequality
used for lower-bound scores; confusing the two directions loses results.

On the complete carrier $`[0,\infty]`$, an equivalent scalar view uses the
completion value
$`V(s,H)=\inf_{z\in H}\llbracket s\rrbracket(z)`$ for a suffix language
$`H`$. A computable $`\ell(s,H)\le V(s,H)`$ safely rejects the entire
subtree when $`\ell(s,H)>\tau`$. Empty $`H`$ has value infinity. Finite
dictionaries attain nonempty minima; an arbitrary infinite $`H`$ need not.
No termination argument follows merely from writing this infimum.

## 6. Frontier algebra and certified elimination

An atom contains a control position, accumulated cost, and all continuation
context needed to interpret its future. Let $`B_p(z)`$ be its least total
cost after completing suffix $`z`$, including accumulated and final costs.
A finite frontier $`P`$ denotes

```math
\llbracket P\rrbracket(z)=\min_{p\in P}B_p(z),\qquad
\llbracket\varnothing\rrbracket(z)=\top.
```

The semantic dominance preorder is
$`p\preceq_{\rm sem}q\iff\forall z,\ B_p(z)\le B_q(z)`$.
An executable preorder $`\preceq_E`$ is certified when it implies this
relation. It can be much weaker; duplicate-only reduction is a legal example.

**F1 — elimination.** If $`p\preceq_E q`$ and both occur in $`P`$,
removing $`q`$ preserves $`\llbracket P\rrbracket`$.

**Proof.** For every suffix the minimum of the two contributions is the
contribution of $`p`$. Taking the minimum with the other atoms preserves
equality.

For nondeterministic atoms whose transitions already carry their accumulated
costs, a sufficient forward simulation requires: no worse empty-suffix output,
and for every consuming successor of the dominated atom, some consuming
successor of the dominating atom is related to it. Induction on suffix length
proves semantic dominance. Epsilon transitions must first be closed using a
finite ranked construction, or be covered by a proved weak simulation.

For the same control and continuation context, a smaller cost simulates a
larger cost when enabled paths are cost-independent, or have compatible
downward-closed budget guards, and every corresponding update is monotone.
Apply monotonicity along each common path and take the minimum. Comparing only
costs across different pending operations or previous-target registers does
not satisfy these hypotheses.

### 6.1 Normal forms and width

Quotient $`\preceq_E`$ by mutual simulation. For a finite input frontier,
retain the minimal quotient classes and choose the least encoded atom present
in each retained class, using a specified deterministic total tie order.
Sort the resulting representatives. Call this operation $`N_E`$.

**F2 — canonical frontier laws.** $`N_E`$ preserves behavior, is idempotent,
and is independent of enumeration order. Normalized union
$`P\sqcup_E Q=N_E(P\cup Q)`$ is associative, commutative, and idempotent
on these normalized finite frontiers.

**Proof.** Every removed class in a finite poset has a retained minimal class
below it; F1 preserves its contribution. Replacing mutually related atoms
also preserves contribution in both directions. Minimal classes and least
present representatives depend only on the input set, proving permutation
independence and idempotence. In a union, any class removed earlier is still
covered by its earlier dominator or by a smaller surviving class, by
transitivity. Choosing the least present representative is associative under
union. Therefore intermediate normalization does not change the final minimal
classes or their selected representatives.

**F3 — width.** Every such frontier has at most
$`\operatorname{width}(P_E/{\sim_E})`$ atoms, where $`P_E`$ is the
relevant reachable atom preorder. Apply the bound within its declared context
scope, then account for the context representation separately.

**Proof.** Distinct retained quotient classes are incomparable. Their number
is therefore bounded by the definition of width.

This theorem need not give a finite or useful bound. It also cannot replace
$`E`$ with a stronger semantic relation. For example, two distinct atoms
with constant behaviors zero and one have semantic width one. A duplicate-only
normalizer retains both, so its executable-order width is two. Nor does width
bound the number of different frontiers interned over a branching search.

If a transition generator $`G_a`$ realizes derivatives of frontier behavior,
then

```math
N_EG_aN_E(P)\simeq N_EG_a(P).
```

Both denote $`\partial_a\llbracket P\rrbracket`$ by F2 and R2. This
permits normalization fusion or delay when intermediate storage limits are
also respected. Structural equality of the two encodings is a stronger
property and needs its own proof.

## 7. Categories of realizations and simulations

### 7.1 Exact machines

Fix input alphabet $`A`$ and output set $`O`$. An object is a pointed
deterministic machine $`M=(S,s_0,o,\delta)`$. An exact arrow
$`h:M\to M'`$ is a function satisfying

```math
h(s_0)=s'_0,\qquad o'\circ h=o,\qquad
h(\delta(s,a))=\delta'(h(s),a).
```

**K1 — composition.** Identities satisfy these equations; substituting them
for two arrows shows their function composition does too. Thus the objects
and arrows form a category. Induction on input words proves that every arrow
preserves outputs. The map to $`O^{A^*}`$ sending a state to its behavior
is the unique unpointed coalgebra morphism to the final behavior machine;
with a fixed initial behavior it also preserves the point. R3 is its reachable
surjection.

These are coalgebras for $`S\mapsto O\times S^A`$. A representation that
has no convenient functional arrow can use a bisimulation relation: related
states have equal outputs and successors remain related. Induction still
establishes equality. This accommodates sparse/dense implementations whose
encodings do not form a simple quotient function.

### 7.2 Lower simulations

Now fix ordered outputs $`C`$. An arrow $`R:M\rightsquigarrow N`$ is a
relation containing the initial pair such that for each $`sRt`$,

```math
o_M(s)\le o_N(t),\qquad
\delta_M(s,a)\ R\ \delta_N(t,a)\quad\text{for every }a.
```

**K2 — compositional soundness.** Such an arrow implies
$`\llbracket M\rrbracket\le\llbracket N\rrbracket`$. Identity relations
are arrows and relational composition preserves the conditions. Inclusion of
relations orders each hom-set, and composition is monotone in both arguments.
Consequently this is a locally ordered category of simulations.

**Proof.** Induct on input length for the output inequality. For composition,
choose the intermediate related state; output inequalities compose by
transitivity and successor relations compose through its successor. The
initial pair composes through the intermediate initial state. Relation
inclusion is preserved by existential relational composition.

For symbolic abstraction, labels may differ. Require
$`s^\#Rs`$ and $`a\in\gamma(\hat a)`$ to imply
$`\delta^\#(s^\#,\hat a)R\delta(s,a)`$, together with the output
inequality. Along any concrete word represented by the abstract labels,
induction gives a lower bound. Fixed quantization can alternatively pull the
abstract machine back to the concrete alphabet before applying K2.

Singleton labels alone do not repair earlier lost precision. Point exactness
means that an exact state embedding $`\eta`$ satisfies
$`\delta^\#(\eta(s),[a,a])=\eta(\delta(s,a))`$, up to the declared
representation equality. It does not assert equality for every already
abstracted state.

### 7.3 Composition operators

A synchronous pair steps both states on the same input and combines outputs
with a fixed operation $`h`$. Exact arrows lift componentwise. If $`h`$ is
monotone, lower simulations lift componentwise as well: apply the component
inequalities then monotonicity. Associative and unital $`h`$ induce the usual
associativity and unit isomorphisms of these pairs, where defined.

This is distinct from weighted concatenation, from composition of two-tape
transducers over intermediate words, and from a categorical product. Their
universal properties and path choices differ. In particular, a synchronized
score sum does not enforce a common internal alignment.

## 8. Dictionary synchronization and contextual reduction

Let a deterministic dictionary focus $`z`$ denote the suffix language
$`H_z`$ of one immutable revision. Finality decides whether
$`\varepsilon\in H_z`$ and an edge $`a`$ leads to the left quotient
$`H_{z'}=a^{-1}H_z`$. Include path context in the focus whenever node identity
alone does not determine visibility or values.

The synchronized state $`(z,s)`$ steps both components on an existing edge.
Its scored behavior is $`\llbracket s\rrbracket(w)`$ when
$`w\in H_z`$, and $`\top`$ otherwise. Exact arrows and lower simulations
lift by holding $`z`$ fixed. The proof is by cases on membership and then K1
or K2. This makes a local query optimization reusable in every lawful
dictionary backend.

**H1 — restricted dominance.** Define

```math
p\preceq_H q\iff\forall w\in H,\ B_p(w)\le B_q(w).
```

Removing a dominated atom preserves the scored behavior restricted to $`H`$,
by the same pointwise minimum proof as F1. If $`H'\subseteq H`$, dominance
on $`H`$ implies dominance on $`H'`$. If $`F\le_H G`$, then
$`\partial_aF\le_{a^{-1}H}\partial_aG`$, since each tested suffix
$`w`$ has $`aw\in H`$. Notice that the derivative language
$`a^{-1}H`$ need not itself be a subset of $`H`$.

For example, behaviors with values $`B_p(a)=0,B_q(a)=1`$ and
$`B_p(b)=2,B_q(b)=0`$ are globally incomparable but $`p`$ dominates
$`q`$ when $`H=\{a\}`$. A reduction valid there is unsound under
$`H=\{b\}`$.

On a finite acyclic dictionary, exact contextual equivalence of two fixed
deterministic states can be checked recursively by comparing their outputs at
final foci and their derivatives at every outgoing edge. Memoize state pairs
and complete dictionary context; use a topological worklist. Induction on
remaining dictionary height proves the check. Its cost can approach exhaustive
evaluation, so useful applications use reusable suffix summaries or cheap
sufficient certificates. Correctness does not imply a speedup.

Contextual reduction returns a dictionary-scoped state. Its identity includes
the immutable revision and the complete suffix/visibility context on which the
proof depends. It cannot enter a query-global cache as an unrestricted state.
On a DAWG, sharing query transitions does not authorize dropping distinct
root-to-node keys, stable IDs, values, or witness paths.

**H2 — candidate/exact composition.** Suppose each stored original is
represented by an abstract path, the abstract machine lower-simulates its
concrete score, every original in each admitted collision bucket is retained,
and the verifier returns the specified exact score. Exhaustive candidate
traversal followed by the inclusive exact cutoff test returns exactly the
within-cutoff originals, provided every internal pruning guard is an
admissible lower bound on **all completions** from that product state.

**Proof.** An exact within-cutoff original has no over-cutoff admissible bound
on its path and hence is admitted; collision retention ensures it reaches the
verifier. Conversely every emitted original passes exact verification. Finite
traversal, or a completed resumable traversal, supplies exhaustive coverage.

This proves completed-result semantics. Changing bounds or work accounting
may change partial pages. Continuations must preserve the remaining work and
snapshot, and an incomplete traversal must never report complete empty.

## 9. Geometry of residual behavior

For behaviors into $`[0,\infty]`$ on a common domain, define

```math
a\mathbin{\dot{\smash{-}}}b=\inf\{r\in[0,\infty]:a\le b+r\},\qquad
\Delta(F,G)=\sup_w\bigl(F(w)\mathbin{\dot{\smash{-}}}G(w)\bigr).
```

In particular $`\infty\dot{\smash{-}}\infty=0`$; a finite right operand and
infinite left operand give infinity. This avoids undefined subtraction of
infinities. Equivalently, $`\Delta(F,G)`$ is the least uniform additive
slack making $`F\le G+r`$; such a least extended slack exists here.

**G1 — directed residual metric.**

```math
\Delta(F,F)=0,\qquad
\Delta(F,H)\le\Delta(F,G)+\Delta(G,H),\qquad
\Delta(F,G)=0\iff F\le G.
```

**Proof.** The defining scalar slack is zero exactly when its left operand is
at most its right operand. Two inequalities with slacks $`r,s`$ compose to
one with slack $`r+s`$. Apply this at every word and take suprema; an infinite
right-hand slack makes the triangle inequality immediate.

**G2 — nonexpansive derivatives and alternatives.**

```math
\Delta(\partial_aF,\partial_aG)\le\Delta(F,G),
```

```math
\Delta(\min_i F_i,\min_i G_i)\le\max_i\Delta(F_i,G_i)
\quad\text{for a nonempty finite family}.
```

**Proof.** Derivatives test only words starting with $`a`$, a subset of the
words in the original supremum. For alternatives, every
$`F_i\le G_i+r`$ with the common maximum slack $`r`$; taking minima
preserves this inequality and commutes with adding a finite common slack.
The infinite case is immediate.

Thus semantic dominance is zero directed discrepancy. Symmetrization
$`\max(\Delta(F,G),\Delta(G,F))`$ gives an extended metric on behaviors,
and an extended pseudometric on representations with duplicate behaviors.
This is a Lawvere-enriched view using $`([0,\infty],\ge,+,0)`$ as base.
It concerns differences between residual behaviors; it does not assert that
the original sequence score is metric or that min-max path costs are an
ultrametric. Suprema over finite words are useful without defining costs of
completed infinite paths.

For a sound lower approximation $`\widehat F\le F`$ and a certificate
$`\Delta(F,\widehat F)\le\epsilon`$, we obtain
$`\widehat F\le F\le\widehat F+\epsilon`$. A lower value above cutoff
rejects safely; an upper value within cutoff certifies membership. Neither
inequality alone supplies an exact emitted cost. Approximation chains add
their certified discrepancies by G1. These laws quantify abstraction quality
without making semantic equivalence or supremum computation decidable.

## 10. Measure-to-machine synthesis

### 10.1 Specification interface

The following are mathematical/internal design interfaces, not new exported
Rust APIs. A `MeasureSpec` contains a domain and encoding contract, immutable
query/parameters, a reference score semantics, an observation profile, and a
numeric model. It has one of three executable descriptions:

| Description | Meaning | Initial construction |
|---|---|---|
| Evaluator | A terminating reference computation on a complete finite input | The history realization of R2 |
| Recurrence | Boundary expressions, cell dependencies, update expressions, and enabling predicates | Ranked recurrence evaluation |
| Composition | Primitive specifications combined by named pointwise or concatenation operators | D1/D2 and the corresponding state constructors |

A semantic-only noncomputable specification still denotes a residual machine
but has no executable artifact. A terminating evaluator is an explicit premise,
not something inferred from the word “metric.” Custom measures can supply
additional representations and certificates without changing the calculus.

The recurrence description names a finite set of phases, query-coordinate
bounds, target-generation offsets, required label registers, and a finite
topological rank for same-generation dependencies. Every dependency either
uses an earlier retained target generation or a smaller current-generation
rank. Arbitrary user expressions are permitted only with declared semantics;
automatic algebraic rewriting uses primitives with available certificates.

### 10.2 A constructive rolling-recurrence theorem

**S1 — bounded generation realization.** Suppose a fixed-query recurrence has
$`k`$ phases and $`m+1`$ query coordinates per target generation, reads at
most $`r`$ preceding generations, has a finite ranked within-generation
evaluation, and uses bounded additional label/context registers. Its cell
updates are effective and its output depends only on these retained values.
Then it has a deterministic realization with

```math
O(k(m+1)(r+1))
```

retained cost cells, plus its declared query, label/context, and scratch
storage. If at most $`b`$ bounded-size alternatives are inspected per cell,
the update uses $`O(k(m+1)b)`$ primitive evaluations per target label.

**Proof.** Initialize the boundary generation. For each new label, retain the
last $`r`$ completed generations and evaluate the next generation in its
topological rank. Induction on that rank gives exactly the reference cell:
every dependency is already computed or covered by the generation invariant.
Induction on consumed length then proves every retained generation agrees with
the reference recurrence. A generation older than the maximum offset is never
read again, so reclaiming it preserves this invariant. Projecting the specified
output proves R2. Ring indices must be checked against live generation tags or
maintained by an equivalent proved rotation. Counting cells and alternatives
gives the bounds.

The arithmetic cost of each primitive and the bit sizes of retained values are
additional terms. Witness history is not included unless the observation
profile explicitly supplies a bounded witness representation. S1 is a general
construction theorem for a recurrence, independent of whether the score is
metric, min-plus, min-max, or a soft aggregation.

### 10.3 Intermediate representation and certificates

Use two typed levels. The semantic level describes score expressions and
recurrences. The realization level describes finite control, cost registers,
label/context registers, and pure transition computations. Core nodes are:

```text
Constant, Register, LabelField, QueryField
Apply(primitive, ordered_arguments)
ChooseMin(alternatives)
Guard(predicate, enabled, disabled)
ParallelAssign(destinations, expressions)
RankedSweep(rank_domain, dependencies, assignments)
Output(expression)
Synchronize(dictionary_specification, machine)
```

`Apply` preserves argument order and parenthesization. Only certified
minimum-compatible updates participate in C1 rewrites; a general `Apply`, such
as soft minimum, carries no idempotence assumption. A ranked sweep has a
query-bounded rank domain and explicit dependency edges. `ParallelAssign`
reads old registers; dependencies on newly computed cells must instead be
explicit ranked edges. Abstract machines have a distinct type annotation
carrying their concretization relation.

A `RealizationCertificate` records the seed/step/output correspondence, input
domain, numeric model, observation profile, and resource theorem actually
proved. A `RewriteCertificate` records a source expression, target expression,
named theorem, instantiated premises, equality or inequality direction, and
scope dependencies. Scope includes query, parameters, construction cutoff,
label interpretation, and dictionary revision/context where used. Hashes
identify candidate objects; exact canonical payload comparison establishes
identity. An unchecked Boolean capability flag is not a certificate.

The finite exact fragment can check transition congruence, simulations, and
residual partitions exhaustively. Symbolic fragments use the corresponding
universal proof or a checked solver proof in the declared arithmetic theory.
Bounded sampling is evidence against a claim when it finds a counterexample;
passing samples do not discharge a universal premise. A failed or unavailable
proof leaves the existing valid realization intact and produces an explicit
unproved-obligation result. This is the total behavior of the synthesis
interface, not a purported proof that the requested optimization is impossible.

### 10.4 Derivation algorithm

```text
DERIVE(specification)
    validate domain, observation, arithmetic, and immutable parameters
    construct reference behavior F
    if evaluator description:
        construct the history realization and its R2 certificate
    if recurrence description:
        verify dependency rank and maximum lookback
        construct the direct recurrence realization using S1
    if composition description:
        derive children and apply the named D1/D2 constructor
    retain every proved realization and its precise resource profile
    apply only transformations with accepted rewrite certificates
    attach independent metric/domain and numerical certificates
    return realizations, certificates, and any unproved stronger claims
```

For an arbitrary recurrence that lacks the S1 premises but has a terminating
batch evaluator, the history realization is still exact and explicitly
identified as such. It does not satisfy a requested bounded-memory profile;
the result reports that missing certificate. No automatic derivation step
silently replaces the supplied measure with Levenshtein distance or an
approximation. Abstract filtering is an additional certified artifact.

## 11. Worked derivations

All equations in this section first describe exact mathematical arithmetic.
The same recurrence can define a machine-number specification by fixing its
primitive evaluation order. Equality to real arithmetic is then a separate
numerical claim. The executable companion checks the derived rolling machines
against independently organized full-matrix or path-enumeration references.

### 11.1 Standard Levenshtein

For fixed $`q=q_1\cdots q_m`$, let the current row have
$`r_i=d_{\rm Lev}(q_{1:i},u)`$. Seed $`r_i=i`$. On target label $`a`$,

```math
r'_0=r_0+1,\qquad
r'_i=\min\{r_i+1,\ r'_{i-1}+1,\ r_{i-1}+[q_i\ne a]\}.
```

Output $`r_m`$. The three alternatives are target insertion, query deletion,
and match/substitution. Induction on $`i`$ followed by input length proves
R2. S1 gives two query-sized generations and no previous-label register.

For integer cutoff $`k`$, C2 clips every cell to
$`\{0,\ldots,k,\top\}`$, so there are at most
$`(k+2)^{m+1}`$ row encodings. This loose bound proves finiteness without
claiming the optimal state count. The equality vector
$`([q_i=a])_{i=1}^m`$ determines the entire update, proving observation
congruence; symbols absent from the query share an observation.

An atom $`(i,c)`$ contributes
$`B_{(i,c)}(z)=c+d_{\rm Lev}(q_{i+1:m},z)`$. Since the two suffix queries
differ by $`|i-j|`$ insertions/deletions, the triangle inequality implies

```math
c+|i-j|\le d\quad\Longrightarrow\quad(i,c)\preceq_{\rm sem}(j,d).
```

This is a sufficient positional simulation for standard edits. Restricted
transpositions, pending multi-symbol operations, and affine phases require
their own continuation semantics; the formula is not transplanted to them.
The residual/frontier approach is grounded in the Schulz–Mihov construction
([reference 1](#15-related-work-and-naming)).

### 11.2 Generalized operation grammars

For an operation $`t`$ consuming $`a_t`$ query labels and $`b_t`$ target
labels, with applicability predicate $`P_t`$ and nonnegative exact cost
$`w_t`$, define

```math
D[i,j]=\min_{t\text{ applicable at }(i,j)}
\{D[i-a_t,j-b_t]+w_t\}.
```

Only nonnegative predecessor coordinates participate. Set $`D[0,0]=0`$;
unreachable cells are $`\top`$. No operation consumes zero labels on both
axes. Predicates inspect only their declared slices and bounded explicit
context. The source/target orientation is bound explicitly when adapting the
existing `OperationSet` API.

Every edge increases $`i+j`$, proving finite acyclic semantics for a finite
input. At a new target column, operations with $`b_t=0`$ have $`a_t>0`$
and therefore read earlier query coordinates. Other operations read one of
the last $`r=\max_t b_t`$ generations. A window of $`r`$ target labels
supplies applicability slices. S1 constructs the online machine; its proof
also covers operations consuming more than one target label. A full-row
minimum is not automatically a safe cut for this case: transitions can jump
over that row. Pruning must cover all live retained generations and pending
dependencies, or use the exact recurrence-level C2 saturation.

The cheapest cost at the same cell and identical continuation context suffices
by monotonicity. Stronger cross-coordinate elimination needs a grammar-specific
simulation. Exact scaling of all finite configured decimal weights gives the
integer instance; scaling does not establish classical unit-edit subsumption.
See [generalized operations](../design/generalized-operations.md).

### 11.3 ERP

Let $`\rho`$ be the ground metric and $`g`$ a fixed gap value. Seed
$`r_0=0`$ and $`r_i=r_{i-1}+\rho(q_i,g)`$. On target label $`a`$,

```math
r'_0=r_0+\rho(a,g),\qquad
r'_i=\min\{r_i+\rho(a,g),\ r'_{i-1}+\rho(q_i,g),\
r_{i-1}+\rho(q_i,a)\}.
```

Output $`r_m`$. Last-operation decomposition proves recurrence correctness;
S1 derives two rows and no previous-target register. Nonnegative increments
give C2, while a target consisting of arbitrarily many gap values has zero
cost against the empty query. Thus a positive minimum charge per target label
cannot be assumed.

For scalar intervals $`I=[l,h]`$, replace target distances by

```math
\operatorname{dist}(x,I)=\max(0,l-x,x-h).
```

It is at most $`|x-a|`$ for every $`a\in I`$. Monotonicity of minimum and
addition propagates these inequalities through the cell DAG. With an exact
initial state and singleton labels the leaves and then all cells are equal.
This proves the abstract and point certificates. The metric qualification is
on the quotient deleting gap-value samples, proved in Section 12. The original
ERP recurrence is due to Chen and Ng ([reference 7](#15-related-work-and-naming)).

### 11.4 Discrete Fréchet

For a nonempty query and nonempty target, the coupling recurrence is

```math
D[i,j]=\max\bigl(\rho(q_i,y_j),
\min\{D[i-1,j],D[i-1,j-1],D[i,j-1]\}\bigr).
```

Use $`D[0,0]=0`$ and $`D[i,0]=D[0,j]=\top`$ for positive indices.
The first cell consequently equals the first link. Every coupling's final
move is one of the three predecessor cases; taking its worst link and then
the best coupling proves the recurrence. Two rows suffice, and after stepping
the current target label need not be retained for the next score transition.

Bottleneck updates are inflationary, so C2 applies without additive path
costs. Interval/box ground-distance lower bounds lift through monotone minimum
and maximum. Exact point embeddings give equality. Scalar absolute distance
and certified vector ground metrics are instances. The stutter quotient and
triangle proof appear in Section 12. An API may separately extend the domain
by assigning zero to two empty inputs and infinity to exactly one empty input;
that is an extended pseudometric convention, not a finite distance on all
raw vectors. See Eiter and Mannila ([reference 8](#15-related-work-and-naming)).

### 11.5 TWED, including explicit physical timestamps

Write query labels as $`(q_i,t_i)`$, target labels as $`(y_j,s_j)`$, and
use the shared value sentinel zero at a fixed common time origin $`t_0=s_0`$.
Time units and origin are part of the immutable configuration. Define

```math
\begin{aligned}
e_i&=\rho(q_i,q_{i-1})+\nu(t_i-t_{i-1})+\lambda,\\
f_j&=\rho(y_j,y_{j-1})+\nu(s_j-s_{j-1})+\lambda,\\
v_{i,j}&=\rho(q_i,y_j)+\rho(q_{i-1},y_{j-1})
+\nu(|t_i-s_j|+|t_{i-1}-s_{j-1}|).
\end{aligned}
```

Use $`D[0,0]=0`$, cumulative $`e_i`$/ $`f_j`$ boundary costs, and

```math
D[i,j]=\min\{D[i-1,j]+e_i,
D[i,j-1]+f_j,D[i-1,j-1]+v_{i,j}\}.
```

The preceding target value and time affect $`f_j`$ and $`v_{i,j}`$.
Retain them with the rolling column. The initial state contains the query-only
boundary column and sentinel pair. A step validates the next time, computes
$`f_j`$, sweeps query coordinates using the formula, and commits the new
pair. The row invariant and last-operation decomposition prove R2; S1 gives
two columns plus that pair. Unit-grid TWED sets $`t_i=t_0+i`$ and
$`s_j=s_0+j`$, giving displacement $`2\nu|i-j|`$ and requiring an index
counter. It is not a substitute for irregular physical timestamps.

**Boundary-sensitive qualification.** Marteau's published manuscript uses
infinite positive-axis boundaries in Equation 9 and Figure 2, whereas the
library uses cumulative deletion boundaries. These are different measures on
the same raw inputs. The source's metric theorem must not be cited as a direct
proof of a changed boundary convention.

There is an exact reduction for the cumulative convention on a suitable
domain. Require $`\nu>0`$, $`\lambda\ge0`$, a ground metric covered by
the source theorem (including scalar absolute distance), and strictly
increasing timestamps with the **first strictly after the origin**.
Prepend the common anchor $`(0,1)`$ to each sequence and shift every original
time to $`1+t_i-t_0`$, in one fixed time unit. The anchored sequences obey
strict timestamp order. The published infinite-axis recurrence must first
match the two anchors, at zero cost. Its remaining axis cells accumulate
exactly the $`e_i`$ and $`f_j`$ above; all interior cells agree after
shifting indices. Induction on the grid proves equality. Hence this
cumulative-boundary measure is an injective pullback of the source metric,
so M2 proves its metricity, including an empty input represented by its anchor.
This proves transfer for the stated domain; numerical transfer is still separate.

The broader existing constructor accepts a first timestamp **equal** to the
origin. That domain has a strict-metric counterexample when $`\lambda=0`$:
with origin zero and $`\nu=1`$,

```math
X=((1,1)),\qquad Y=((0,0),(1,1)),\qquad D(X,Y)=0,
\quad X\ne Y.
```

The first target sample can be deleted at zero cost, then the remaining pair
matches at zero cost. Both satisfy the constructor's internal strict-order
check. Consequently its current metric label is too broad for strict identity;
the certified restriction above is needed before applying this transfer
theorem. The committed scalar source has no strict-origin wrapper; a
`StrictOriginTimestampedSeries` and `distance_bounded_strict` design exists
only as an uncommitted worktree overlay. The committed raw constructor and
scorer remain available on the broader domain, so their outputs do not
acquire a metric guarantee from this theorem. The Wolfram
companion reproduces this failure and checks the anchored reduction on its
finite corpus. See [reference 9](#15-related-work-and-naming) for the source theorem.
The [Rocq transfer proof](../verification/twed/theories/Metric/TimestampedMetricTransfer.v)
proves uniqueness of the anchored infinite-axis recurrence grid, equality of
each shifted physical-time charge, injectivity of the anchor map, and the
metric pullback conditional on the source theorem's metric axioms. It also
constructs the distinct origin-equal pair and proves that its 1-by-2
cumulative recurrence grid has final cost zero. A separate validator lemma
shows that the proposed strict guard enters the transfer domain while the
origin-equal counterexample passes the raw constructor's abstract guard. The
published source metric theorem and a binary64-to-real correspondence are
separate premises; neither is re-proved by the transfer file.

**Branch audit, 2026-09-06.** After fetching all remote heads, review covered
all twelve local branches and eighteen remote branch references, plus the
remote default-branch alias. Every branch containing the timestamped module
had identical file content, Git blob
`c67191b9ba79fba0de6533b371d4de6140e3b357`. The local heads carrying it were:

| Branch | Reviewed head |
|---|---|
| `master` | `91fcac73` |
| `codex/julia-distance-family-parity` | `91fcac73` |
| `codex/family-evidence-completeness` | `b0796ad1` |
| `codex/julia-generalized-universal` | `38b8fa22` |
| `codex/rc5-relocation-correction` | `461ccceb` |
| `codex/regresspec-temporal-complete` | `0755c2e7` |
| `codex/universal-variant-semantics` | `76c5f325` |

The remaining branches had an older unit-grid TWED implementation or no TWED
module; they contained no alternative timestamped-origin fix. The accessible
feature worktrees had no uncommitted TWED changes. Commit `61f92b8a`
introduced the timestamped module; `9a35f318` added interval bounds, resource
checks, and iterative allocation handling, while retaining the origin guard
and cumulative boundaries. Those improvements are already included in the
reviewed blob and do not remove this counterexample.

The existing [metric property test](../../tests/proptest_timestamped_twed_metric.rs)
generates strictly positive timestamp increments, including the first, so it
does not sample the constructor's origin-equality case. It checks zero
self-distance but does not assert that zero distance forces equal operands.
These facts explain how that test can pass while the broader accepted domain
fails strict identity. The finding is scoped to these reviewed revisions.

Interval relaxation bounds each ground-distance term and each feasible time
term. For target deletion, the elapsed-time lower bound is minimized over the
feasible previous/current time relation; dropping strict positivity of the
elapsed gap to zero is safe for nonnegative stiffness. A rectangular
overapproximation may lose correlation but cannot omit feasible pairs. Point
exactness requires the full previous/current state to be embedded exactly.
Every valid increment is nonnegative, so C2 applies. Invalid time order is a
domain error and never a cheap path.

### 11.6 MSM

For scalar values and fixed split/merge charge $`c>0`$, define

```math
K(a,b,d)=c+\operatorname{dist}(a,[\min(b,d),\max(b,d)]).
```

This is $`c`$ when $`a`$ lies between $`b,d`$, and otherwise adds the
smaller endpoint distance. Start $`D[1,1]=|q_1-y_1|`$. The first column
accumulates $`K(q_i,q_{i-1},y_1)`$ and the first row accumulates
$`K(y_j,q_1,y_{j-1})`$. Interior cells are

```math
D[i,j]=\min\{D[i-1,j-1]+|q_i-y_j|,
D[i-1,j]+K(q_i,q_{i-1},y_j),
D[i,j-1]+K(y_j,q_i,y_{j-1})\}.
```

Before the first target label, retain a distinct unstarted state. The first
step constructs its column using the boundary rule. Later steps retain the
previous target value with the preceding column and apply the interior rule.
This specifies seed, step, and output for nonempty inputs; the explicit empty
extension is zero for both empty and infinity for exactly one empty operand.
S1 proves bounded register count after initialization. Positive increments
support C2; cross-context subsumption still needs its own theorem.

A conservative interval rule follows directly from the displayed definition:
for interval inputs $`A,B,D`$, use
$`c+\operatorname{gap}(A,\operatorname{hull}(B\cup D))`$. Every concrete
segment between a member of $`B`$ and a member of $`D`$ is contained in
that hull, so this lower-bounds every concrete $`K`$. Singleton inputs are
exact. Monotone cell propagation proves the corresponding abstract machine.
This is a derived admissible rule; it does not assert that every existing MSM
implementation uses this exact formula. Existing MSM metric and implementation
evidence is mapped in the [MSM verification documentation](../verification/msm/README.md).

### 11.7 Banded DTW and Soft-DTW

For DTW with nonnegative local discrepancy $`\ell(q_i,y_j)`$, use

```math
D[i,j]=\ell(q_i,y_j)+
\min\{D[i-1,j],D[i-1,j-1],D[i,j-1]\},
```

with origin zero and positive-axis boundary infinity. A fixed band sets cells
outside $`|i-j|\le b`$ to infinity. Retain a column and the band/depth
context; S1 proves the next-column realization. Local costs are inflationary,
but the score can identify repeated samples and need not obey triangle
inequality. Metric qualification is unnecessary for the construction.

Soft-DTW replaces minimum by

```math
\operatorname{softmin}_\gamma(x_1,\ldots,x_k)
=-\gamma\log\sum_i\exp(-x_i/\gamma),\qquad\gamma>0.
```

Use the same boundaries and rolling dependency graph, with an all-infinite
predecessor set evaluating to infinity. S1 still applies, now to a signed
score carrier. However, two equal alternatives of cost $`x`$ aggregate to
$`x-\gamma\log2`$. Multiplicity matters: minimum-frontier elimination,
idempotent deduplication of alternative paths, and C2 based on nonnegative
local costs do not follow. Exact reuse of an entire identical deterministic
state is still legal. The score-only rolling realization does not promise
bounded storage for all gradients or alignment witnesses. See Cuturi and
Blondel ([reference 10](#15-related-work-and-naming)).

### 11.8 A compositional measure: edits plus histogram discrepancy

For a fixed nonnegative rational $`\lambda`$, define

```math
D_\lambda(x,y)=d_{\rm Lev}(x,y)
+\lambda\sum_{a\in A}|\operatorname{count}_x(a)-
\operatorname{count}_y(a)|.
```

The sum has finite support even for an infinite alphabet. For a fixed query,
keep a count for each distinct query symbol and one aggregate count of target
symbols absent from the query. The latter contributes its count directly to
the discrepancy. Initialize all target counts to zero; each label increments
exactly one counter. Pair this machine with the Levenshtein row machine and
output their stated sum. D1 and the counter invariant prove R2.

For finite cutoff $`\tau`$, any prefix length
$`t>m+\lfloor\tau\rfloor`$ is permanently dead: every extension has
Levenshtein distance at least $`t-m>\tau`$. Up to that boundary all counters
are bounded, and the Levenshtein row can be saturated at $`\tau`$ by C2.
The pair therefore has a finite bounded-score realization and an effective
finite observation partition. The proof counts the histogram storage as well
as the edit row. At $`\lambda=0`$, remove the counters by D1.

Do not prune because the current histogram discrepancy exceeds the cutoff:
missing query symbols can still arrive. A safe prefix lower bound is the
maximum of zero and target excess for the histogram component, multiplied by
$`\lambda`$, plus any certified Levenshtein completion lower bound. For
example, total positive coordinate excess relative to the query cannot be
repaired by appending more labels. Metric qualification follows from M2 below:
Levenshtein is a metric and histogram distance is a pseudometric. This example
demonstrates synthesis from composition; it is not a novelty claim for the
distance formula.

## 12. Metric and quotient qualification

### 12.1 Script metrics and alignment restrictions

**M1 — shortest-script qualification.** Suppose configurations form an
undirected graph, every primitive edit has an inverse of the same nonnegative
weight, and the distance is the infimum over all finite edit scripts. It is
an extended pseudometric. If distinct configurations have a positive lower
bound on every connecting script, it separates them. If every pair is
connected it is finite-valued. A uniform positive lower bound on every
nonidentity edge is one sufficient separation condition.

**Proof.** Empty scripts give zero self-distance. Reverse scripts give
symmetry. Concatenating scripts gives triangle inequality by taking arbitrarily
close finite approximants to each infimum; infinite right sides are immediate.
The stated separation and connectivity assumptions give the remaining laws.
Zero-cost identifications require quotienting by zero distance.

A restricted alignment grammar must additionally be proved to realize this
script distance, or have a separate alignment-composition proof. Positive
symmetric costs alone are insufficient. With substitution costs
$`w(a,b)=w(b,c)=1`$, $`w(a,c)=3`$, and insertion/deletion costs ten, a
single-column substitution recurrence gives distances one, one, three.
Unrestricted scripts can substitute twice and give at most two for the last
pair. OSA supplies another pinned example: `CA` to `AC` costs one, `AC` to
`ABC` costs one, and `CA` to `ABC` costs three.

### 12.2 Composition and quotients

**M2 — fixed channel composition.** Let $`d_i`$ be a finite family of pseudometrics and
$`S_i:X\to X_i`$ fixed maps. For fixed $`w_i\ge0`$,

```math
D(x,y)=\sum_i w_i d_i(S_ix,S_iy)
```

is a pseudometric, with zero-weight terms omitted. It is a metric exactly when
the positively weighted component comparisons jointly separate points. The
same conclusion holds for a finite maximum of positively scaled components.

**Proof.** Pullback preserves nonnegativity, symmetry, and triangle inequality.
Adding the weighted inequalities, or taking a maximum and using
$`\max_i(a_i+b_i)\le\max_i a_i+\max_i b_i`$, proves triangle inequality.
A finite sum or maximum of nonnegative terms is zero exactly when each active
term is zero.

A zero weight does not necessarily destroy metricity if other components
still separate points. Conversely a constant transform can destroy separation
despite a positive weight. Pair-dependent weights or normalization are not
covered. Pointwise minimum does not preserve metricity: on three points, take
metrics with edge triples $`(1,10,10)`$ and $`(10,1,10)`$; their minimum
has triple $`(1,1,10)`$ and fails triangle inequality.

**M3 — zero-distance quotient.** For a pseudometric $`d`$, define
$`x\sim y`$ iff $`d(x,y)=0`$. This is an equivalence relation, and
$`\bar d([x],[y])=d(x,y)`$ is well-defined and separates classes.

**Proof.** Reflexivity and symmetry are metric laws; triangle proves
transitivity. Applying triangle twice to representatives at zero distance
gives $`d(x,y)\le d(x',y')`$ and the reverse inequality. Zero class
distance therefore means equal classes.

For ERP, extend the ground alphabet by a tagged gap mapped to the fixed point
$`g`$. Compose two alignments by synchronizing their middle sequence,
inserting gap/gap columns as necessary. Project the resulting triples to an
outer alignment and delete gap/gap columns. The ground triangle inequality
bounds each projected column by the two original costs, hence bounds their
sums. Symmetry follows by swapping alignment columns. Zero-cost alignments
identify equal non-gap-value symbols in the same order; deleting every sample
equal to $`g`$ gives the same sequence. Conversely equal such representatives
have a zero-cost alignment. This proves ERP's pseudometric and quotient claims.

For discrete Fréchet, synchronize two monotone couplings on their middle
sequence. Within a fixed middle index, interleave the advances of the two
outer indices; at a middle advance both original couplings supply the next
indices. This constructs a monotone coupling of the outer sequences, after
discarding repeated pairs. Every outer link has length at most the sum of the
two original bottlenecks by the ground triangle inequality. Taking the maximum
and then optimal couplings proves triangle inequality. At zero bottleneck,
every coupled pair is identical. A change in value must occur simultaneously
on both sides, so collapsing consecutive equal runs gives identical sequences.
Conversely equal run-collapsed sequences admit a zero coupling. This proves
the nonempty stutter-quotient qualification.

### 12.3 Numerical and cutoff boundary

The untruncated mathematical measure receives M1–M3 certificates. A rounded
realization needs numerical transfer before a triangle-dependent algorithm
may use its returned numbers as exact bounds. Let $`z=2^{53}+2`$. Binary64
round-to-nearest gives

```math
\operatorname{fl}(|0-z|)=2^{53}+2,
\qquad
\operatorname{fl}(\operatorname{fl}(|0-1|)+
\operatorname{fl}(|1-z|))=2^{53}.
```

Thus even rounded absolute distance violates the naive machine-arithmetic
triangle test. Likewise, $`T_1`$ applied to ordinary distances of points
zero, one, and two turns the long side into infinity while the two short sides
still sum to two. Metric labels must not be interpreted as proofs that such
rounded or cutoff-modified arithmetic obeys the exact metric axioms.

## 13. Optimization rules and selection

The optimizer works over already certified realizations. It performs a bounded
search of alternatives using the following rule catalogue. Equality rules
preserve the listed semantic observation; inequality rules create separately
typed filters or bounds. An optimization is adopted only if its resource
contract remains satisfied and its measured objective improves.

| Rule | Required certificate | Authorized transformation | Observation |
|---|---|---|---|
| O1 | R2 plus fixed parameters | Partial evaluation and constant expression sharing | Score |
| O2 | C1 for the exact update graph | Factor a common update across minimum | Score |
| O3 | C2 | Saturate unreachable-over-budget internal costs | Bounded score |
| O4 | F1 with full context | Remove a simulated atom | Score or certified bounded score |
| O5 | F2 and transition correspondence | Fuse, delay, or batch normalization | Score |
| O6 | Equal observation implies equal complete transition | Cache by state/observation class | Score |
| O7 | Exact payload equality and immutable arena entries | Intern repeated encodings | Score |
| O8 | S1 and tagged generation invariant | Replace matrix/history storage by rolling generations | Score |
| O9 | R2 in both representations and R2a at each permitted switch | Choose dense, sparse, or packed layout, including adaptive switching | Declared score and witness |
| O10 | B1/B2 on a complete continuation graph or an admissible relaxation | Propagate remaining budgets backward | Membership/pruning |
| O11 | H1, immutable suffix context | Apply dictionary-relative reduction | Context-restricted score |
| O12 | K2 and H2 | Traverse a cheaper abstract product then verify | Completed exact results |
| O13 | Pure synchronized transitions and zipper navigation laws | Reject query projection before allocating a child focus | Completed exact results |
| O14 | G1/G2 plus an explicit slack certificate | Compose or compare approximations | Bounded discrepancy |
| O15 | Exact additive homogeneity and output-offset reconstruction | Normalize common cost offsets | Score |
| O16 | Admissible priorities, frontier coverage, and the operation's tie/continuation discipline | Choose DFS, BFS, or best-first scheduling, including certified lexicographic kNN pruning | Declared completed-result order |

For O16's exact top-$`k`$ specialization, the result order is a **separate
layer** over completed candidates. On one captured revision, give each
candidate a unique tie key and rank it by the pair of its exact cost and that
key. Give each queued region both an admissible lower cost and a lower tie
key. Once $`k`$ exact results exist, a region can be removed when its lower
pair is at least the worst exact pair in lexicographic order. In particular,
equal cost bounds permit removal only if the region's tie floor cannot beat
the current kth tie. The proof is by contradiction: a better unseen candidate
would be covered by a region whose lower pair is no greater than that
candidate's pair, and hence smaller than the kth pair. The floor may be
weakened or unknown, in which case pruning decreases; a stale floor that is
too high is unsound. This transformation never changes the cost monoid or
the exact verifier's recurrence. It requires the machine implementation's
cost comparison and lower-bound relation, not merely the ideal arithmetic
versions. LOCPA BF-1 gives the full result-order and snapshot conditions.

For O15, if an exact min-plus update satisfies
$`T(v+b\mathbf1)=T(v)+b\mathbf1`$ and the output has the same property,
represent a row as $`b\mathbf1+\bar v`$, with the minimum finite component
of $`\bar v`$ zero. Substitution proves reconstruction after each step.
Keep $`b`$ for outputs and budgets; an all-infinite row has a separate
encoding. It is a dead state only if no other retained generation or pending
context can reach a finite completion. Sharing $`\bar v`$ can improve transition reuse even when absolute
costs differ. Binary64 arithmetic does not inherit this translation law.

For exact additive path graphs, a consistent potential $`h`$ with
$`h(u)\le w(u,v)+h(v)`$ gives reweighted edges
$`w'(u,v)=w(u,v)+h(v)-h(u)\ge0`$. Summing along a path telescopes to its
old cost plus $`h(\mathrm{end})-h(\mathrm{start})`$. With accounted terminal
offsets this preserves rankings and supplies a route to best-first evaluation.
An admissible but inconsistent heuristic does not justify label-setting
finality without reopening states. This is an exact-additive specialization,
not a generic operation on bottleneck or rounded costs.

The bounded search has this specified behavior:

```text
OPTIMIZE(certified_machine, objective, limits)
    seed a worklist with the input machine and an empty proof trace
    visit candidates in deterministic canonical order
    enumerate matching O1–O16 rules and their finite matches
    discharge every premise with the declared checker
    add a new candidate only with an accepted instantiated certificate
    keep exact and abstract candidates in separate result classes
    deduplicate candidate encodings using exact structural comparison
    stop before the exploration/work/storage limits are exceeded
    replay-check each selected proof trace from the original machine
    select the feasible candidate improving the declared measured objective
    return selected artifact, trace, measurements, and exploration status
```

Candidates with unchanged measurements lose to the original or the simpler
canonical representation; no optimization is adopted just because it was
generated. Stopping exploration limits optimality of the search, not validity
of an accepted candidate. The objective names workload, work/time measure,
memory ceiling, and tie rule before measurement. The theory does not promise
that globally fastest implementations or complete semantic quotients are
computable.

Canonical witnesses require more than these score equations. Either prove
that the witness order is respected under continuation and every rewrite, or
recompute the specified canonical witness from the exact original at final
verification. An arbitrary lexicographic order on partial scripts need not be
continuation-compatible. Resource outcomes similarly use a separate execution
refinement: preflight, transactional commit, honest incomplete results, and
snapshot-preserving resumption. Optimized runs may use different amounts of
work and different page boundaries while preserving their promised completed
observations.

## 14. Evidence, controls, and reproducibility

The results R1–R4, C1–C2, D1–D2, B1–B2, F1–F3, K1–K2, H1–H2, G1–G2,
S1, and M1–M3 have mathematical proofs above. The worked recurrences apply
those results with explicit boundaries and continuation registers. The named
TWED source theorem is attributed; the boundary-transfer argument and its
strict-origin restriction are given in Section 11.5.
The existing [formal verification manifest](../verification/FORMAL_VERIFICATION_MANIFEST.tsv)
describes actual mechanized artifacts; these new manuscript identifiers do not
automatically enter its trusted set.

Run the independent executable companion from the repository root:

```sh
WolframKernel -noinit -noprompt -script scripts/verify-ordered-residual-calculus.wls
scripts/doc-mathlint.sh docs/theory/ordered-residual-calculus.md docs/theory/lazy-ordered-cost-product-automata.md docs/theory/README.md
```

The [Wolfram companion](../../scripts/verify-ordered-residual-calculus.wls)
uses [`Resolve`](https://reference.wolfram.com/language/ref/Resolve.html)
over the reals to check thirteen universally quantified identities and
inequalities: directed triangle and minimum stability, min/max distribution,
additive and bottleneck budgets, interval soundness and refinement, the MSM
between-interval formula and hull bound, common-offset normalization,
consistent potentials, and weighted sum/maximum triangle preservation.
Every formula includes its hypotheses; failure, an unevaluated result, a
timeout, or an unexpected kernel message fails the run. These are symbolic
arithmetic results, not proofs about all automaton executions.

The companion also exhaustively checks the transformation semiring on all monotone
top-preserving functions of a four-element chain, tests saturation and backward
budget laws, checks frontier normalization on finite preorders, and compares
derived streaming recurrences with independently organized batch references.
It covers every prefix in its stated finite corpus, exact interval examples,
dictionary range results, chunking, contextual reduction, and the composite
histogram measure. Arithmetic is exact except in deliberate binary64 controls.
Soft-DTW is checked through its exact path-partition polynomial on integer
squared local costs: substitute $`z=\exp(-1/\gamma)`$ and apply
$`-\gamma\log`$ to recover the score. This checks multiplicity for every
positive smoothing parameter without approximate score comparisons. The
script runs in one kernel and starts no licensed subkernels. It checks its manuscript
links and named section references so examples and evidence stay navigable.

The following counterexamples are permanent executable controls:

| Invalid inference | Rejecting example |
|---|---|
| Scalar binary64 addition is associative | $`(2^{53}+1)+1\ne2^{53}+(1+1)`$ |
| Rounded metric values automatically satisfy machine triangle tests | The zero, one, $`2^{53}+2`$ example in Section 12 |
| Cutoff-saturated metric values remain a metric | Points zero, one, two with cutoff $`3/2`$: infinity exceeds one plus one |
| Every metric has a finite threshold automaton | R4 and arbitrarily many distinguishable $`a^i`$ prefixes |
| A small lower filter proves a small exact realization | The one-state zero filter for R4 |
| An incomplete simulation gives the semantic width bound | Constant-zero/constant-one atoms under duplicate-only normalization |
| Symmetric positive substitutions suffice for metricity | The one, one, three substitution example |
| Restricted swaps inherit unrestricted edit metricity | The OSA `CA`, `AC`, `ABC` triangle |
| TWED metricity ignores sentinel/origin conventions | Zero-penalty cumulative TWED gives zero for `[(1,1)]` and `[(0,0),(1,1)]` |
| Positive channel weights guarantee separation after any transform | A constant transformed channel |
| Minimum of metrics is metric | Edge triples one/ten/ten and ten/one/ten |
| Independent and shared optimum alignments agree | Alignment pairs zero/ten and ten/zero |
| Contextual dominance is globally reusable | Suffix language changing from `{a}` to `{b}` |
| Singleton labels restore a previously abstracted state | Different previous abstract/concrete accumulated costs |
| Equal soft alternatives may be discarded | $`x-\gamma\log2\ne x`$ |
| Over-cutoff current output means dead continuation | `abc` versus `a`, then suffix `bc` |

Finite checks are falsification and executable-example evidence, not universal
proofs or validation of all Rust kernels. Production changes using these rules
must additionally use the operational document's independent-oracle, witness,
collision, resource, and backend gates at the scope of the changed code.

The benchmark interpretation is likewise explicit. Measure generated atoms,
frontier width, exact state/observation reuse, visited dictionary edges,
verification calls, context/path bytes, and retained register/bit counts. For
O10–O12 include preprocessing cost and false-positive amplification. For O8
include counters, caches, and witnesses in retained memory. Report wall time
with those causal quantities on a fixed input/configuration corpus. Demand-led
construction may still inspect the entire dictionary with zero outputs.

## 15. Related work and naming

ORC is a synthesis built on established mathematics. Its descriptive name does
not assert that residuals, ordered path algebras, antichains, or coalgebras are
new. The application-specific contribution is the explicit composition of
measure derivation, numerical semantics, representation certificates,
dictionary-relative reductions, and implementation-level optimization rules.
No theorem here is labeled a first discovery: that would require a more
specific priority comparison than a name search or a shared vocabulary.

| Name | Emphasis | Assessment |
|---|---|---|
| **Ordered Residual Calculus (ORC)** | Cost order, remaining behavior, and derivation/transformation rules | Selected working name; broad and pronounceable |
| Residual Frontier Calculus | Finite frontiers and their reduction algebra | Strong name for the min-frontier specialization |
| Calculus of Cost Automata | Direct description of the subject | Accessible but less distinctive |
| Continuation Cost Calculus | Future behavior and backward budgets | Intuitive, but less visibly tied to automata |
| Lazy ordered-cost product automata | The library's execution architecture | Retained as the name of the operational specialization |

Primary foundations and their exact roles:

1. K. U. Schulz and S. Mihov, *Fast String Correction with Levenshtein
   Automata*, IJDAR 5, 67–85, 2002.
   [DOI](https://doi.org/10.1007/s10032-002-0082-8) ·
   [author-hosted technical report](https://www.cis.uni-muenchen.de/download/cis-berichte/01-127.pdf).
   Fixed-query Levenshtein states, subsumption, and dictionary search.
2. J. J. M. M. Rutten, *Automata, Power Series, and Coinduction: Taking Input
   Derivatives Seriously*, 1999.
   [CWI publication and manuscript](https://ir.cwi.nl/pub/4557).
   Behavior as a final coalgebra and compositional input derivatives.
3. R. Alur, L. D'Antoni, J. Deshmukh, M. Raghothaman, and Y. Yuan,
   *Regular Functions and Cost Register Automata*, LICS 2013.
   [author-hosted paper](https://www.cis.upenn.edu/~alur/Lics13reg.pdf).
   Deterministic register models for quantitative functions. ORC's
   data-dependent registers and general predicates require their own
   expressiveness analysis; CRA decidability results do not transfer wholesale.
4. M. Mohri, *Weighted Automata Algorithms*, in *Handbook of Weighted
   Automata*, 2009.
   [author-hosted chapter](https://cs.nyu.edu/~mohri/pub/hwa.pdf).
   Semiring computations, composition, determinization hypotheses, and
   weight normalization. ORC's Moore minimality is not weighted-NFA minimality.
5. L. Doyen and J.-F. Raskin, *Antichain Algorithms for Finite Automata*,
   TACAS 2010.
   [author-hosted paper](https://lsv.ens-paris-saclay.fr/~doyen/papers/Antichains_Algorithms_Finite_Automata.pdf).
   Simulation orders and finite antichain representations.
6. P. Cousot and R. Cousot, *Abstract Interpretation: A Unified Lattice Model
   for Static Analysis of Programs by Construction or Approximation of
   Fixpoints*, POPL 1977.
   [author's publication record](https://www.di.ens.fr/~cousot/COUSOTpapers/POPL77.shtml).
   Sound approximation, abstract domains, and adjunctions.
7. L. Chen and R. T. Ng, *On the Marriage of Lp-norms and Edit Distance*,
   VLDB 2004.
   [conference paper](https://www.vldb.org/conf/2004/RS21P2.PDF).
   ERP's ground-gap alignment and triangle inequality; raw sample APIs need
   the explicitly stated gap-value quotient.
8. T. Eiter and H. Mannila, *Computing Discrete Fréchet Distance*, Technical
   Report CD-TR 94/64, 1994.
   [author-hosted report](https://www.kr.tuwien.ac.at/staff/eiter/et-archive/files/cdtr9464.pdf).
   Monotone couplings and bottleneck recurrence.
9. P.-F. Marteau, *Time Warp Edit Distance with Stiffness Adjustment for Time
   Series Matching*, IEEE TPAMI 31(2), 306–318, 2009.
   [DOI](https://doi.org/10.1109/TPAMI.2008.76) ·
   [manuscript](https://arxiv.org/pdf/cs/0703033) ·
   [revised manuscript record](https://data.hal.science/document/hal-00135473v5).
   Timestamped segment-edit recurrence and its metric hypotheses. Equation 9
   and Figure 2 use infinite positive-axis boundaries; Section 11.5 states
   and proves the transfer to the library's cumulative-boundary convention.
10. M. Cuturi and M. Blondel, *Soft-DTW: a Differentiable Loss Function for
    Time-Series*, ICML 2017.
    [conference paper](https://proceedings.mlr.press/v70/cuturi17a.html).
    Non-idempotent soft alignment and the score/gradient distinction.
11. N. Urabe and I. Hasuo, *Quantitative Simulations by Matrices*,
    Information and Computation 252, 110–137, 2017.
    [extended paper](https://arxiv.org/abs/1810.09146).
    Categorical simulations and checked quantitative inequalities. The
    relation-based category here is the explicitly defined deterministic
    specialization, not a claim to reproduce every matrix-simulation result.
12. F. W. Lawvere, *Metric Spaces, Generalized Logic, and Closed Categories*,
    1973; reprinted 2002.
    [journal reprint](https://www.tac.mta.ca/tac/reprints/articles/1/tr1abs.html).
    Directed extended metrics and enrichment; Section 9 proves the particular
    residual discrepancy used here.
13. C. McBride, *The Derivative of a Regular Type Is Its Type of One-Hole
    Contexts*, 2001, and G. Huet, *The Zipper*, JFP 7(5), 1997.
    [McBride manuscript](https://strictlypositive.org/diff.pdf) ·
    [Huet DOI](https://doi.org/10.1017/S0956796897002864).
    Structural differentiation of data types, a separate calculus supporting
    cursor representations. For a regular tree type $`T=\mu X.F(X)`$,
    a subtree context is a finite sequence of frames from $`F'(T)`$, giving
    a subtree zipper $`T\times\operatorname{List}(F'(T))`$.
