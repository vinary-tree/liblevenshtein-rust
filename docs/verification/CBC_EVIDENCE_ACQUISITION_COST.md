# Cost of acquiring stopping evidence

**Status:** theory model for `cbc-05-acquisition`. This document defines
questions for later quantitative experiments. It makes no measured runtime
claim and does not run the planned optimization campaigns.

BF-4 answers a semantic question: given a fixed evidence object, is the
completed ordered answer forced in every feasible completion? A search
policy must still decide which evidence to acquire, in what order, and at
what cost. The information theorem supplies no such policy or runtime
optimum. In particular, a stronger summary may be semantically useful and
still take more total work than the exact verifications it avoids.

## Evidence state and admissible decisions

An evidence state $`I`$ contains the captured eligible originals, immutable
request scope, exact verified ranks, sound intervals or relational facts, and
the set $`\mathcal M(I)`$ of feasible completed score assignments. Every
acquisition action $`a`$ produces a new state $`I'`$ only if its checked
producer establishes

```math
\varnothing\ne\mathcal M(I')\subseteq\mathcal M(I).
```

The inclusion means that sound acquisition refines information; nonemptiness
rules out accepting contradictory evidence as a stopping certificate. The
producer also preserves the captured identity and arithmetic scope. An
action may instead report validation failure, resource exhaustion, or a
suspension. Those are explicit outcomes, never silent evidence refinements.

At a leaf of a decision tree, a completed answer is allowed only if it is
safe for every assignment in $`\mathcal M(I)`$ and the result has the
required full or underfull cardinality. A leaf may defer or fail according to
its declared operation contract. The tree may branch on a checked summary,
exact verification, interval comparison, cache lookup, or eligibility fact.
Branching on an unproved numeric fact does not create a valid tree.

## Charge all actions

Use a vector of primitive counts for each action, rather than a single
surrogate count:

```math
C(a)=\bigl(v,\ s,\ q,\ r,\ h,\ m,\ c,\ p,\ b\bigr).
```

Here $`v`$ counts exact candidate verifications, $`s`$ summary construction
or update, $`q`$ queue/heap operations, $`r`$ relational reasoning,
$`h`$ hashing and collision comparisons, $`m`$ materialization or witness
replay, $`c`$ layout/cache conversion, $`p`$ certificate checking, and
$`b`$ allocation or release work. Each count has a specified unit and
source-to-model accounting obligation. A plan's **executed work** sums its
action vectors. **Retained space** and **peak live space** require separate
stateful accounting; they cannot be reconstructed by summing allocation
counts. Setup and evidence discarded after a failed branch remain charged.

A declared workload model may map the vector to predicted time with
nonnegative coefficients and include memory constraints. Those coefficients
are environment-dependent measurements, not mathematical constants supplied
by BF-4. For a stochastic workload, state the distribution over requests and
evidence outcomes; expected work and p95/p99 latency are different
objectives. For an adversarial model, state the set of possible outcomes and
optimize a worst-case objective. If a certificate or summary is shared across
queries, include construction, maintenance, invalidation, and amortization
over a declared reuse horizon.

## Four claims that require different evidence

| Claim | Sufficient kind of argument | What it does not imply |
|---|---|---|
| Decision soundness | Checked evidence producer plus BF-4 and the operation's result contract | A good acquisition policy |
| Test completeness in one evidence language | An attainable counter-completion whenever that test refuses to stop | That the test is cheap to evaluate |
| Fewest exact verifications | Lower bound over a specified policy class, fixed free/charged side information, and an algorithm meeting it | Minimum total primitive work |
| Minimum elapsed time or peak bytes | Reproducible end-to-end measurements on a named platform and workload, plus a declared decision rule | Universal optimality on other workloads |

For a restricted independent-interval language, an indistinguishable pair of
feasible completions with different exact answers shows that a sound
evidence-only policy must acquire distinguishing information or defer. This
is an **information** lower bound. It does not say which probe is cheapest:
one summary may distinguish many originals at high construction cost, while
several exact verifications may be cheaper. Relational or metric constraints
can change which completions are feasible, so the bound must be restated for
the actual language.

## Negative cost control

Consider a branch with five unresolved originals. One policy verifies each
at unit primitive cost, then stops: its declared work is five units. A
second policy constructs a perfect summary at cost twenty units and performs
no exact verification: its work is twenty units. Both reach a sound terminal
decision. The summary saves five verifications but spends fifteen more work
units. This is a model counterexample to treating verification count or bound
strength as a total-work optimum. It says nothing about measured latency;
that depends on actual primitive costs and cache behavior.

The symmetric case is possible: a summary that costs two units and avoids a
hundred expensive verifications can win. The decision depends on the
distribution of encountered regions, the reuse horizon, queue effects, and
space retention. Any claimed work lower bound must specify whether a policy
may precompute or cache summaries, ask relational queries, use exact score
oracles, or defer. Allowing free unbounded side information makes a
verification-count lower bound meaningless.

## Questions for deferred experiments

For conditioned tie summaries, inherited rank certificates, compact product
frontiers, and source cursors, record at least:

1. Which exact original verifications were avoided, and at what summary,
   queue, cache, and comparison cost?
2. How much retained metadata, conversion scratch, and peak RSS was added?
3. At what query volume is one-time construction amortized, and how often
   does snapshot or parameter revision invalidate it?
4. Does a stronger bound change first-result latency, median, p95, p99, or
   completion rate under budgets?
5. Which completion or tie-equality case witnesses that a refused stopping
   test really needs more evidence under the declared language?

The [performance adoption gate](CBC_PERFORMANCE_ADOPTION_GATE.md) supplies
the later measurement and nonregression rule. These questions refine the
planned experiments; they do not authorize execution or adoption.
