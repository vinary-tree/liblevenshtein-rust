[← Documentation Index](../../README.md)

# Gph-enriched Lawvere Theories and GSLTs

This document presents Gph-enriched Lawvere theories and Graph-Structured Lambda Theories
(GSLTs) as alternatives to full OSLF when bound variables can be eliminated through
reflection or combinators. These approaches directly capture operational semantics as
graphs enriching the hom-sets.

**Target audience**: Compiler engineers implementing operational semantics

**Primary Sources**:
- Stay, M. & Meredith, L. G. "Representing operational semantics with enriched Lawvere
  theories." arXiv:1704.03080, 2017.
- Meredith, L. G. et al. "MeTTa Architecture Proposal" (hypercube/map.md)

---

## Table of Contents

1. [For Implementers: What You Need to Know](#for-implementers-what-you-need-to-know)
2. [Comparison: GSLTs Across Calculi](#comparison-gslts-across-calculi)
3. [Motivation: Why Gph-theories?](#motivation-why-gph-theories)
4. [Mathematical Prerequisites](#mathematical-prerequisites)
5. [Definition of Gph-theories](#definition-of-gph-theories)
6. [Graph-Structured Lambda Theories (GSLTs)](#graph-structured-lambda-theories-gslts)
7. [Presentation of Finitely Generated GSLTs](#presentation-of-finitely-generated-gslts)
8. [Examples](#examples)
9. [Reduction Contexts and Gas](#reduction-contexts-and-gas)
10. [RHO Calculus as a Gph-theory](#rho-calculus-as-a-gph-theory)
11. [GSLT to Rholang Compilation](#gslt-to-rholang-compilation)
12. [Generating Typed GSLTs from Untyped Ones](#generating-typed-gslts-from-untyped-ones)
13. [Application to MeTTa](#application-to-metta)

---

## For Implementers: What You Need to Know

### The 80/20 Rule for GSLTs

**Most of what you need to implement** (handles 80% of cases):

1. **Parse GSLT specification** - extract sorts, morphisms, equations, reductions
2. **Generate type-lifted symbols** - apply T(-) rules from [05-type-lifting.md](./05-type-lifting.md)
3. **Implement src/tgt tracking** - maintain source and target of each reduction
4. **Handle structural equations** - especially multiset commutativity/associativity

**Skip for initial implementation**:
- Full modal type derivation
- Behavioral predicates
- Bisimulation checking

### Data Structures

```rust
// GSLT representation
struct GSLT {
    name: String,
    generating_sorts: Vec<Sort>,     // P, N, R, etc.
    generating_morphisms: Vec<Morphism>,
    interaction: Option<Morphism>,   // The magmal operation
    structural_equations: Vec<Equation>,
    reductions: Vec<Reduction>,
}

struct Sort {
    name: String,
    is_process_sort: bool,  // P
    is_reduction_sort: bool, // R
}

struct Morphism {
    name: String,
    domain: Vec<Arity>,
    codomain: Sort,
}

enum Arity {
    Sort(Sort),
    Product(Box<Arity>, Box<Arity>),
    Exponential(Box<Arity>, Box<Arity>),
    Unit,  // 1
}

struct Reduction {
    name: String,
    inputs: Vec<(String, Arity)>,
    source_equation: String,  // src(r(...)) = ...
    target_equation: String,  // tgt(r(...)) = ...
}
```

### Implementation Checklist

- [ ] Parse GSLT specification format
- [ ] Build symbol table for sorts and morphisms
- [ ] Validate morphism arities
- [ ] Implement structural equation normalization
- [ ] Generate type-lifted morphisms
- [ ] Extract reductions as graph edges
- [ ] (Optional) Generate typing rules
- [ ] (Optional) Compile to Rholang

### Quick Reference: GSLT Components

| Component | What It Specifies | Implementation |
|-----------|-------------------|----------------|
| Generating sorts | Base types (P, N, R, ...) | Enum variants |
| Generating morphisms | Term constructors | AST node types |
| Interaction | Binary reduction trigger | Pattern match target |
| Structural equations | Equivalence classes | Normalization rules |
| Reductions | Computation steps | Reduction rules |

---

## Comparison: GSLTs Across Calculi

This table shows how the same concepts manifest across different calculi.

### Structural Overview

| Feature | Lambda | SKI | RHO | Ambient | MeTTa |
|---------|--------|-----|-----|---------|-------|
| **Generating Sorts** | P | P | P, N | P, N, M | Term, State, KB, List, MSet |
| **Reduction Sort** | R | R | R | R | R (state transitions) |
| **Binding** | Lam: (P->P)->P | None | ?: N x (N->P)->P | nu: (N->P)->P | Pattern vars |
| **Interaction** | App | App | \| (par) | \| (par) | State transition |
| **Reflection** | None | None | @, * | None | quote, eval |

### Complete GSLT Specifications

#### Lambda Calculus GSLT

```
GSLT Lambda:
  Generating Sorts: (none beyond P, R)

  Generating Morphisms:
    s, t : R -> P                    ; Graph structure
    App : P x P -> P                 ; Application
    Lam : (P -> P) -> P              ; Abstraction

  Interaction: App

  Reductions:
    Beta : (P -> P) x P -> R
    Beta(K, N) : App(Lam(K), N) ~> ev(K, N)

    Head : R x P -> R                ; In-context
    Head(r, B) : App(s(r), B) ~> App(t(r), B)
```

#### SKI Combinator GSLT

```
GSLT SKI:
  Generating Sorts: (none beyond P, R)

  Generating Morphisms:
    s, t : R -> P
    S : 1 -> P
    K : 1 -> P
    I : 1 -> P
    App : P x P -> P
    S1 : P -> P                      ; S with 1 arg
    S2 : P x P -> P                  ; S with 2 args
    K1 : P -> P                      ; K with 1 arg

  Interaction: App

  Structural Equations:
    App(S, x) = S1(x)
    App(S1(x), y) = S2(x, y)
    App(K, x) = K1(x)

  Reductions:
    Sigma3 : P x P x P -> R
    Sigma3(x, y, z) : App(S2(x, y), z) ~> App(App(x, z), App(y, z))

    Kappa : P x P -> R
    Kappa(x, y) : App(K1(x), y) ~> x

    Iota : P -> R
    Iota(x) : App(I, x) ~> x
```

#### RHO Calculus GSLT

```
GSLT RHO:
  Generating Sorts: N (names)

  Generating Morphisms:
    s, t : R -> P
    0 : 1 -> P                       ; Nil process
    | : P x P -> P                   ; Parallel
    ! : N x P -> P                   ; Output (lift)
    ? : N x (N -> P) -> P            ; Input
    * : N -> P                       ; Dereference (drop)
    @ : P -> N                       ; Quote

  Interaction: | (parallel composition)

  Structural Equations:
    |(P, 0) = P                      ; Unit
    |(P, Q) = |(Q, P)                ; Commutativity
    |(|(P, Q), R) = |(P, |(Q, R))    ; Associativity
    @(*x) = x                        ; Quote-drop identity

  Reductions:
    Comm : N x (N -> P) x P -> R     ; Communication
    Comm(x, K, Q) : |(?(x, K), !(x, Q)) ~> ev(K, @(Q))

    ParL : R x P -> R                ; In-context (left)
    ParL(r, Q) : |(s(r), Q) ~> |(t(r), Q)

    ParR : P x R -> R                ; In-context (right)
    ParR(P, r) : |(P, s(r)) ~> |(P, t(r))
```

#### Ambient Calculus GSLT (Sketch)

```
GSLT Ambient:
  Generating Sorts: N (names), M (capabilities)

  Generating Morphisms:
    s, t : R -> P
    0 : 1 -> P
    | : P x P -> P
    [] : N x P -> P                  ; Ambient bracket
    . : M x P -> P                   ; Capability prefix
    nu : (N -> P) -> P               ; Restriction
    ! : 1 -> P                       ; Replication
    in : N -> M                      ; Enter capability
    out : N -> M                     ; Exit capability
    open : N -> M                    ; Open capability

  Interaction: | (parallel)

  Reductions:
    In : N x N x P x P x P -> R
    In(n, m, Q, R, S) : |([](n, |(.(in(m), Q), R)), [](m, S))
                      ~> [](m, |([](n, |(Q, R)), S))

    Out : N x N x P x P x P -> R
    Out(n, m, Q, R, S) : [](m, |([](n, |(.(out(m), Q), R)), S))
                       ~> |([](n, |(Q, R)), [](m, S))

    Open : N x P x P -> R
    Open(n, P, Q) : |(.(open(n), P), [](n, Q)) ~> |(P, Q)
```

### Key Differences Table

| Aspect | Lambda | SKI | RHO | Ambient |
|--------|--------|-----|-----|---------|
| **# of base reductions** | 1 (beta) | 3 (S, K, I) | 1 (comm) | 3 (in, out, open) |
| **Duplication in reductions** | None | S (z appears 2x) | Comm (x appears 2x) | All (n, m appear 2x) |
| **Structural equations** | None | Partial application | Par monoid, quote-drop | Par monoid |
| **Names** | No | No | Yes (via reflection) | Yes (primitive) |
| **Mobility** | No | No | No | Yes |

---

## Motivation: Why Gph-theories?

Full OSLF (Native Type Theory) handles arbitrary binding structures via presheaves,
but this complexity is unnecessary when:

1. **Bound variables can be eliminated** via reflection (quote/dequote)
2. **Combinators suffice** for the computational needs
3. **We only need operational semantics**, not full dependent types

The key insight from Stay & Meredith (2017):

> "With nominal features eliminated as syntactic sugar via reflection, multisorted
> Lawvere theories enriched over graphs suffice to capture the operational semantics
> of the calculus."

### Comparison: Gph-theories vs OSLF

| Aspect | Gph-enriched Lawvere | OSLF/Native Types |
|--------|---------------------|-------------------|
| Binding | Via reflection/combinators | Directly via presheaves |
| Complexity | Simpler | More general |
| Types | Sorts in multi-sorted theory | Predicates on terms/behavior |
| Rewrites | Graph edges in hom | Internal graph + modalities |
| Best for | Name-free calculi | Full nominal calculi |

### GSLTs: A More Complete Picture

While basic Gph-theories capture operational semantics, **Graph-Structured Lambda
Theories (GSLTs)** extend this framework with:

1. **Lambda structure**: Cartesian closed categories for automatic handling of
   bound variables and substitution
2. **Magmal structure**: Distinguished binary interaction operator
3. **Modal types**: Derived from the interaction structure

GSLTs provide a complete framework for:
- Well-defined operational semantics
- Compilation to Rholang
- Optimal reduction strategies

---

## Mathematical Prerequisites

### Directed Multigraphs (Gph)

A **directed multigraph** G consists of:
- A set V of **vertices**
- A set E of **edges**
- Functions s, t : E → V giving **source** and **target**

```
     s       t
E ─────> V <───── E
```

Multiple edges between the same vertices are allowed (multi-graph).

### The Category Gph

**Gph** is the category where:
- **Objects** are directed multigraphs
- **Morphisms** are graph homomorphisms (preserve vertices and edges)

Gph is **cartesian closed**:
- Products G × H exist (vertex pairs, edge pairs)
- Exponentials `$G^H$` exist (graph of homomorphisms from H to G)

### Enrichment

A category **enriched over Gph** has:
- Objects as usual
- Hom(A, B) is a **graph** (not just a set)
- Vertices of Hom(A, B) = morphisms A → B
- Edges of Hom(A, B) = "rewrites" between morphisms

### The Graph Schema Category Th(Gph)

The category **Th(Gph)** has:
- Two objects: **P** (processes/states) and **R** (rewrites)
- Two parallel morphisms: **s, t : R → P** (source and target)

```
        s
    ───────>
R             P
    ───────>
        t
```

A **graph-structured theory** Th is equipped with a functor from Th(Gph) to Th,
embedding this basic graph structure.

---

## Definition of Gph-theories

### Definition: Multisorted Lawvere Theory

A **multisorted Lawvere theory** L consists of:
1. A set S of **sorts**
2. A small category L with:
   - Objects: finite products of sorts (s₁ × ... × sₙ)
   - Morphisms: term-forming operations
   - Composition: substitution

Bill Lawvere pioneered the use of categories to capture grammars modulo equations
for structural congruence. Lawvere proved that his theories correspond to finitary
monads. Todd Trimble generalized this to
[multisorted Lawvere theories](https://ncatlab.org/toddtrimble/published/multisorted+Lawvere+theories)
with a similar monadicity theorem.

### Definition: Gph-enriched Lawvere Theory

A **Gph-theory** is a Lawvere theory where:
- Hom-sets are replaced by **hom-graphs**
- Vertices = terms (morphisms)
- Edges = **reduction steps** (rewrites between terms)

```
Hom_G(A, B) = ⟨V, E, s, t⟩
```

Where:
- V = {f : A → B} (terms of appropriate type)
- E = {r : f ⇒ g} (reductions from f to g)
- s(r) = f, t(r) = g

### Composition of Reductions

Given f ⇒ f' : A → B and g : B → C, we get:

```
g ∘ f ⇒ g ∘ f' : A → C    (post-composition)
```

Similarly for pre-composition. This makes composition a graph morphism.

### Cartesian vs Cartesian Closed

**Lawvere's original theories** were cartesian categories (finite products only).
While it is possible to define bound variables and substitution in a Lawvere theory,
it is cumbersome and not particularly enlightening.

**Lambda theories** use cartesian closed categories instead, which handle bound
variables and substitution automatically via the internal hom objects.

---

## Graph-Structured Lambda Theories (GSLTs)

GSLTs are Lawvere theories equipped with extra structure and stuff that allow us to:
1. Talk about **reductions between terms**
2. **Bind variables** naturally
3. Define **modal types**

### Definition: GSLT

A **Graph-Structured Lambda Theory (GSLT)** is:

1. **Multi-sorted**: At least two generating sorts—one for processes (P) and one
   for rewrites (R)

2. **Graph-structured**: A functor from Th(Gph) → Th embedding the source/target
   maps s, t : R → P

3. **Magmal**: A distinguished binary morphism `$\odot$` : X × Y → P called the
   **interaction**, where X, Y are sorts

4. **Lambda theory**: A cartesian closed category, automatically handling bound
   variables via exponential objects

### Why "Magmal"?

The term "magmal" refers to the algebraic structure where we have a binary operation
(the interaction `$\odot )$` without requiring associativity or other laws. This is
appropriate because:

- In **lambda calculus**, beta reduction is of the form:
  ```
  ((λx.T) U) ~> T[U/x]
  ```
  whose source is an **application** of one process to another

- In **pi calculus**, the comm rule is of the form:
  ```
  for(y <- x)P | x!z ~> P[z/y]
  ```
  whose source is a **juxtaposition** (parallel composition) of two processes

### Modal Types from Interaction

The magmal structure gives rise to **modal types**. Consider the type of terms:

```
⟨⊙([], A)⟩B
```

This represents terms that, when placed into a context `$\odot ([], x)$` where x : A, may
reduce to a term of type B.

| Calculus | Interaction `$\odot$` |  Modal Type | Interpretation |
|----------|---------------|------------|----------------|
| Lambda calculus | Application | A ⇒ B | Function type |
| Pi calculus | Parallel | ◇(A ▷ B) | Possibility modal |
| Pi calculus | Parallel | □(A ▷ B) | Necessity modal |

The "possibility" modal type ◇ corresponds to Caires' rely-guarantee type A ▷ B.
If B can depend on x, these become **dependent product types**.

---

## Presentation of Finitely Generated GSLTs

A finitely generated GSLT can be presented concisely using:

### Components of a GSLT Presentation

1. **Generating sorts**: A finite set including distinguished sorts P (processes)
   and R (rewrites) from the graph structure

2. **Interaction**: A choice of distinguished binary morphism:
   ```
   ⊙ : X × Y → P
   ```
   for some pair of sorts X, Y

3. **Generating morphisms**: A finite set including:
   - **Graph structure morphisms**: s, t : R → P (built-in)
   - **Term constructors**: morphisms with codomain P
   - **Rewrite constructors**: morphisms with codomain R

4. **Equations**: A finite set of equations between morphisms

### Rewrite Constructor Constraints

A rewrite constructor r : `$\prod _{i} Z_{i} \to  R$` whose codomain does not contain a factor of R
must factor through the interaction:

```
∃! f : ∏ᵢ Zᵢ → X × Y.  ⊙ ∘ f = s ∘ r
```

This ensures that the source of any top-level rewrite is an application of the
interaction operator.

### Rewrite Shorthand Notation

We write:

```
r : A ~> B
```

as shorthand for the equations:
```
s(r(z₁, ..., zₙ)) = A
t(r(z₁, ..., zₙ)) = B
```

where A and B use the same free variables zᵢ.

### Two Flavors of Rewrite Constructors

1. **Top-level rewrites**: Codomain does not contain a factor of R
   - These are the "axiom" rewrites of the calculus
   - Source must factor through the interaction

2. **In-context rewrites**: Codomain contains a factor of R
   - These propagate rewrites through term structure
   - Enable reduction under constructors

---

## Examples

### Example 1: SKI Combinator Calculus

The simplest non-trivial Gph-theory:

```
Gph-Theory SKI:

  Sorts: T (terms)

  Constructors (vertices in Hom(1, T)):
    S : 1 → T
    K : 1 → T
    I : 1 → T

  Application (vertex in Hom(T × T, T)):
    app : T × T → T

  Notation: We write (M N) for app(M, N)

  Reductions (edges):
    σ : (((S x) y) z) ⇒ ((x z) (y z))    ; S-reduction
    κ : ((K y) z) ⇒ y                     ; K-reduction
    ι : (I z) ⇒ z                         ; I-reduction
```

The reductions `$\sigma , \kappa , \iota$` are **edges in the hom-graph** Hom(T³, T), Hom(T², T), Hom(T, T)
respectively.

### Example 2: Untyped Lambda Calculus as GSLT

Here is a complete GSLT presentation for untyped lambda calculus with head normal
form evaluation strategy:

```
GSLT Lambda_HNF:

  Generating Sorts: (none beyond P, R)

  Generating Morphisms:
    ; Graph structure (built-in)
    s, t : R → P

    ; Term constructors
    App : P × P → P           ; Application
    Lam : (P → P) → P         ; Lambda abstraction (uses exponential)

    ; Rewrite constructors
    Beta : (P → P) × P → R    ; Top-level: beta reduction
    Head : R × P → R          ; In-context: reduction in head position

  Interaction: App

  Equations:
    ; Beta reduction (top-level)
    Beta(A, B) : App(Lam(A), B) ~> ev(A, B)

    ; Head reduction (in-context)
    Head(r, B) : App(s(r), B) ~> App(t(r), B)
```

**Key observations**:

1. **Lambda abstraction** uses the exponential P → P from the cartesian closed
   structure—this automatically handles variable binding

2. **Beta reduction** uses the evaluator `ev : (P → P) × P → P` from the cartesian
   closed structure—this is function application in the internal logic

3. **Head reduction** is an in-context rule that propagates reductions through
   the application constructor, but only in head position (left of App)

4. The **interaction** is App, so the source of Beta factors through application

### Example 3: Lambda Calculus (via Combinators)

Lambda calculus can be embedded into SKI:

```
Translation [−]:
  [x]       = x
  [M N]     = ([M] [N])
  [λx.M]    = abstract(x, [M])

Where abstract is:
  abstract(x, x)   = I
  abstract(x, M)   = K M           (if x ∉ FV(M))
  abstract(x, M N) = S (abstract(x, M)) (abstract(x, N))
```

This eliminates bound variables, enabling the Gph-theory approach.

### Example 4: Multiset Rewriting

For MeTTa's multisets:

```
Gph-Theory MSet:

  Sorts: T (terms), M (multisets)

  Constructors:
    empty  : 1 → M
    insert : T × M → M

  Structural Equations (as parallel edges):
    comm : insert(x, insert(y, m)) ⇔ insert(y, insert(x, m))

  Note: ⇔ means edges in both directions (equivalence)
```

### Example 5: Pi Calculus Core

A simplified pi calculus as a Gph-theory:

```
Gph-Theory Pi_Core:

  Sorts: P (processes), N (names)

  Constructors:
    0     : 1 → P                     ; Null process
    par   : P × P → P                 ; Parallel composition
    input : N × (N → P) → P           ; Input (with binding)
    output: N × N → P                 ; Output

  Interaction: par

  Structural Equations:
    par(0, P) = P                     ; Null identity
    par(P, Q) = par(Q, P)             ; Commutativity
    par(par(P, Q), R) = par(P, par(Q, R))  ; Associativity

  Reductions:
    ; Communication (top-level)
    comm : par(output(x, v), input(x, F)) ~> ev(F, v)

    ; Parallel reduction (in-context)
    parL : par(s(r), Q) ~> par(t(r), Q)
    parR : par(P, s(r)) ~> par(P, t(r))
```

---

## Reduction Contexts and Gas

Section 6 of Stay & Meredith (2017) shows how to model:
1. **Evaluation strategy** (which redex to reduce)
2. **Resource consumption** (gas for blockchain VMs)

### Reduction Context Markers

Add a constructor R : T → T that marks the "active" position:

```
Constructors:
  R : T → T    ; Reduction context marker

Reduction Rules (consume R):
  σ_R : (((S (R x)) y) z) ⇒ ((x z) (y z))
  κ_R : ((K (R y)) z) ⇒ y
  ι_R : (I (R z)) ⇒ z
```

The marker R is **consumed** during reduction—it acts as a linear resource.

### Modeling Gas

For blockchain VMs, extend with a gas counter:

```
Sorts: T, G (gas)

Constructors:
  zero : 1 → G
  succ : G → G
  gas  : G × T → T    ; Term with gas budget

Reductions (consume gas):
  σ_g : gas(succ(g), (((S x) y) z)) ⇒ gas(g, ((x z) (y z)))
  κ_g : gas(succ(g), ((K y) z)) ⇒ gas(g, y)
  ι_g : gas(succ(g), (I z)) ⇒ gas(g, z)

  stuck : gas(zero, M) ⇒ gas(zero, M)   ; Out of gas (no progress)
```

Each reduction step consumes one unit of gas. When gas reaches zero, computation
is stuck.

### Variable Gas Costs

For more realistic resource modeling, different operations can have different costs:

```
Constructors:
  sub : G × G → G    ; Gas subtraction (saturating at zero)

Reductions (variable cost):
  σ_g : gas(g, (((S x) y) z)) ⇒ gas(sub(g, 3), ((x z) (y z)))
  κ_g : gas(g, ((K y) z)) ⇒ gas(sub(g, 2), y)
  ι_g : gas(g, (I z)) ⇒ gas(sub(g, 1), z)
```

### Relevance to MeTTa

MeTTa's "effort objects" mentioned in Meta-MeTTa can be modeled this way:

```
state_with_effort : Term × KB × MSet × MSet × Effort → State

; Reductions consume effort
query_e : state_with_effort(p, k, w, o, succ(e))
        ⇒ state_with_effort(ε, k, w ∪ matches, o, e)
```

---

## RHO Calculus as a Gph-theory

The paper provides a complete Gph-theory for RHO combinators, showing the approach
works for real concurrent calculi.

### RHO Combinator Syntax (Gph-theory version)

```
Gph-Theory RHO:

  Sorts: W (processes), N (names)

  Constructors:
    ; Process constructors
    0     : 1 → W                     ; Null process
    (|)   : W × W → W                 ; Parallel composition
    drop  : N → W                     ; Drop: ⌊x⌋
    inp   : N × (N → W) → W           ; Input (combinator form)
    out   : N × W → W                 ; Output (lift)

    ; Name constructor
    quote : W → N                     ; Quote: ⌈P⌉

  Structural Equations:
    (| 0) P = P                       ; Null identity
    P | Q = Q | P                     ; Commutativity
    (P | Q) | R = P | (Q | R)         ; Associativity
```

### Note on Eliminating Binding

In standard RHO calculus, input has a bound variable:

```
x(y).P    ; y is bound in P
```

Using combinators, this becomes:

```
inp(x, λy.P)    ; λ represented via SKI
```

The reflection mechanism `$\lceil \lfloor x\rfloor\rceil \equiv _N x$` (quote-drop identity) ensures this encoding
is faithful.

### RHO Reductions

```
Reductions:
  ; SKI reductions (for the combinator part)
  σ, κ, ι : (standard SKI rules)

  ; Communication
  ξ : out(x, Q) | inp(x, F) ⇒ F(quote(Q))
      ; Send Q on x, receive as quoted name

  ; Evaluation (drop-quote)
  ε : drop(quote(P)) ⇒ P
      ; Evaluate a quoted process
```

### Barbed Bisimilarity Preservation

The paper proves (Theorem 4):

> The Gph-theory RHO preserves barbed bisimilarity: if `$P \approx  Q$` in the standard RHO
> calculus, then `$P \approx  Q$` in the Gph-theory interpretation.

This validates that the combinator encoding faithfully represents the original
calculus.

---

## GSLT to Rholang Compilation

A key benefit of GSLTs is that they compile directly to Rholang, enabling optimal
reduction via RSpace.

### Basic Compilation Strategy

The general idea is to:
1. Send the current state on a channel (say, @0)
2. Pattern match on left-hand sides of rewrite rules
3. Replace with right-hand sides
4. Recurse

```rholang
Interpreter = ∑ for(LHS_Pattern <- @0) {
    @0!(RHS) | Interpreter
}
```

Where:
- `LHS` and `RHS` mean left- and right-hand side
- `$\sum$` indicates mutually exclusive options (a disjunction)
- The term picks one rewrite from among matching possibilities, then proceeds

### Handling Unbounded Patterns

One problem: there may be unboundedly many patterns to match against. We don't
want the interpreter term to be infinitely large.

**Solution**: Allow many internal reductions by the interpreter for each reduction
by the original GSLT, and enumerate patterns lazily:

```rholang
Interpreter = PatternGen(0)

PatternGen(n) = for(LHS_Pattern(n) <- @0) {
    @0!(RHS(n)) | Interpreter
} + PatternGen(n+1)
```

### Smart Pattern Enumeration

A smarter enumeration uses the structure of the current state to prune patterns:

```rholang
Interpreter = for(state <- @0) {
    PatternGen'(0, state) | @1!(state)
}

PatternGen'(n, state) = for(LHS_Pattern'(n, state) <- @1) {
    @0!(RHS(n)) | Interpreter
} + PatternGen'(n+1, state)
```

Here, `LHS_Pattern'(n, state)` uses the structure of `state` to enumerate only
those patterns that `state` might match—avoiding futile pattern match attempts.

### RSpace Integration

The Rholang interpreter uses **RSpace**, a highly efficient data structure for:
- Storing continuations
- Matching sends with receives
- Pattern-based lookup

GSLTs compiled to Rholang inherit these optimizations automatically.

### MORK Backend

For even higher performance, RSpace can be implemented on **MORK** (MeTTa Optimal
Reduction Kernel), which provides:
- Trie-based pattern matching
- Bloom filter optimization for negative lookups
- LRU caching for hot patterns
- PathMap integration for shared storage

See [MORK/PathMap Integration](../metta-ecosystem/04-mork-pathmap-integration.md)
for details.

---

## Generating Typed GSLTs from Untyped Ones

### The Hypercube Functor

Given an untyped GSLT, we can systematically generate a typed version using the
**hypercube functor**. This is implemented in MeTTaIL Scala.

The basic idea:
1. Each untyped constructor becomes a family of typed constructors
2. Type parameters are added systematically
3. Reduction rules lift to typed versions

### Example: Typed Lambda Calculus

From the untyped lambda calculus GSLT:

```
; Untyped
App : P × P → P
Lam : (P → P) → P
```

The hypercube functor generates:

```
; Typed (parametric)
App[A, B] : P[A → B] × P[A] → P[B]
Lam[A, B] : (P[A] → P[B]) → P[A → B]
```

### Modal Types via Interaction

The interaction operator `$\odot$` gives rise to modal types. For lambda calculus:

```
⟨App([], A)⟩B  ≈  A → B    (arrow type)
```

For pi calculus:

```
⟨par([], A)⟩B  ≈  ◇(A ▷ B)    (possibility)
□(A ▷ B)                        (necessity)
```

### Connection to OSLF

The hypercube functor provides a mechanical procedure that approximates what OSLF
does more generally:

| Hypercube Functor | OSLF |
|-------------------|------|
| Lifts constructors to typed | Derives predicates on terms |
| Systematic parameter addition | Presheaf construction |
| Modal types from interaction | Behavioral modalities |
| Compile-time checking | Full behavioral reasoning |

For cases where the hypercube functor suffices, it provides a simpler path than
full OSLF.

---

## Application to MeTTa

### Can MeTTa Use Gph-theories?

The key question is whether MeTTa's binding (pattern variables) can be eliminated
via combinators or reflection.

**Arguments for Gph-theories**:
1. MeTTa has **quoting** (terms as data)—similar to RHO's reflection
2. Pattern variables can potentially be encoded via combinators
3. The rewrite rules have clear graph structure

**Arguments for full OSLF**:
1. Unification is more complex than simple substitution
2. Pattern matching may require genuine binding
3. Knowledge base queries involve non-trivial variable scoping

### Proposed GSLT for MeTTa Core

```
GSLT MeTTa_Core:

  Generating Sorts: Term, Atom, List, MSet, KB

  Generating Morphisms:
    ; Graph structure (built-in)
    s, t : R → P

    ; Term constructors
    atom  : Atom → Term
    list  : List → Term
    mset  : MSet → Term
    quote : Term → Term          ; Quotation (reflection)

    ; List constructors
    nil   : 1 → List
    cons  : Term × List → List

    ; Multiset constructors
    empty  : 1 → MSet
    insert : Term × MSet → MSet

    ; State constructor
    state : Term × KB × MSet × MSet → State

    ; Rewrite constructors
    Query   : Term × KB → R           ; Pattern matching
    Output  : MSet × MSet → R         ; Move workspace to output
    AddAtom : Term × KB → R           ; Add to knowledge base
    RemAtom : Term × KB → R           ; Remove from knowledge base

  Interaction: (state constructor, treating it as binary)

  Structural Equations:
    ; Multiset commutativity
    insert(x, insert(y, m)) = insert(y, insert(x, m))

  Reductions:
    ; Query (simplified—actual unification more complex)
    Query(p, k) : state(p, k, w, o) ~> state(ε, k, w ∪ matches(p, k), o)

    ; Output
    Output(w, o) : state(out, k, w, o) ~> state(ε, k, ∅, o ∪ w)

    ; Add atom
    AddAtom(t, k) : state((add t), k, w, o) ~> state(ε, k ∪ {t}, w, o)

    ; Remove atom
    RemAtom(t, k) : state((rem t), k, w, o) ~> state(ε, k \ {t}, w, o)

    ; Quote-unquote (reflection)
    Reflect : quote(unquote(t)) ~> t
```

### Layered Approach

The recommended path for MeTTa:

1. **Start with Gph-theories** for core without complex binding
2. **Model unification** as graph transformations where possible
3. **Use GSLT presentation** for compilation to Rholang
4. **Fall back to OSLF** when genuine binding is needed

### Integration with Correction WFST

In the three-tier correction architecture:

| Tier | Component | GSLT Role |
|------|-----------|-----------|
| 1 | liblevenshtein | N/A (character-level) |
| 2 | MORK/PathMap | GSLT patterns for grammar |
| 3 | MeTTaIL/MeTTaTron | GSLT-based type checking |

The GSLT framework provides:
1. **Formal semantics** for correction rules
2. **Compilation target** (Rholang) for execution
3. **Type discipline** via hypercube functor

---

## Summary

### Gph-enriched Lawvere Theories

Provide:
1. **Simpler semantics** than full OSLF
2. **Direct representation** of operational semantics as graphs
3. **Support for reduction contexts** and resource modeling
4. **Proven correctness** for RHO calculus (barbed bisimilarity)

### Graph-Structured Lambda Theories (GSLTs)

Extend Gph-theories with:
1. **Lambda structure** for automatic binding handling
2. **Magmal structure** for modal type derivation
3. **GSLT presentation format** for concise specification
4. **Compilation to Rholang** for optimal execution

### When to Use Each Approach

| Approach | Use When |
|----------|----------|
| Basic Gph-theory | Simple combinatory systems (SKI) |
| GSLT | Full calculi with binding (lambda, pi) |
| GSLT + Hypercube | Need typed versions of untyped calculi |
| Full OSLF | Need behavioral predicates, bisimulation |

### Key Decision Point

> **Can MeTTa's unification patterns be given a combinator representation?**

- **If yes** → GSLTs suffice (simpler path, compiles to Rholang)
- **If no** → Full OSLF required (handles binding natively)

MeTTa's reflective nature (quoting) suggests the simpler path is viable.

---

## Related Documents

- [05-type-lifting.md](./05-type-lifting.md): Type lifting transformation rules (T(-))
- [06-inference-rules.md](./06-inference-rules.md): Inference rules derived from GSLTs
- [02-native-type-theory-oslf.md](./02-native-type-theory-oslf.md): Full OSLF for behavioral types
- [04-rho-calculus.md](./04-rho-calculus.md): RHO calculus details

---

## References

- Stay, M. & Meredith, L. G. "Representing operational semantics with enriched
  Lawvere theories." arXiv:1704.03080, 2017.
- Meredith, L. G. et al. "MeTTa Architecture Proposal." (hypercube/map.md)
- Lawvere, F. W. "Functorial Semantics of Algebraic Theories." 1963.
- Trimble, T. "Multisorted Lawvere Theories." nLab.
- Caires, L. "Types for Sessions and Pipelines." ESOP 2010.
- See [bibliography.md](../reference/bibliography.md) for complete references.

---

## Appendix: GSLT Presentation Template

For defining new GSLTs, use this template:

```
GSLT <Name>:

  Generating Sorts:
    <list of sorts beyond P, R>

  Generating Morphisms:
    ; Graph structure (built-in)
    s, t : R → P

    ; Term constructors
    <constructor> : <domain> → <codomain>
    ...

    ; Rewrite constructors
    <rewrite> : <domain> → R
    ...

  Interaction: <binary morphism on P>

  Structural Equations:
    <equation> = <equation>
    ...

  Reductions:
    ; Top-level rewrites
    <rewrite>(args) : <source> ~> <target>

    ; In-context rewrites
    <rewrite>(r, args) : <source[s(r)]> ~> <source[t(r)]>
    ...
```

This template captures the essential information for a finitely generated GSLT
and provides a standard format for documentation and implementation.
