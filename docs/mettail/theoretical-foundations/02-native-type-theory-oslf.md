[← Documentation Index](../../README.md)

# Native Type Theory (OSLF)

This document presents the OSLF (Operational Semantics via Lawvere and Fibrations)
construction for deriving type systems from operational semantics. This is the
mathematical foundation for full semantic type checking in MeTTa.

**Target audience**: Compiler engineers and type theory enthusiasts

**Prerequisites**: Familiarity with basic category theory is helpful but not required.
See [06-inference-rules.md](./06-inference-rules.md) for notation guide.

---

## Table of Contents

1. [Notation Guide (Read This First)](#notation-guide-read-this-first)
2. [The Core Idea](#the-core-idea)
$`3. [\lambda`$-Theories with Equality](#$`\lambda`$-theories-with-equality)
4. [The Presheaf Construction](#the-presheaf-construction)
5. [The Internal Language Functor](#the-internal-language-functor)
6. [Native Types](#native-types)
7. [Behavioral Types](#behavioral-types)
8. [Application to MeTTa](#application-to-metta)
9. [Worked Example: LP(T) Construction](#worked-example-lpt-construction)
10. [OSLF vs Gph-Theory Tradeoffs](#oslf-vs-gph-theory-tradeoffs)

---

## Notation Guide (Read This First)

Before diving into OSLF, ensure you understand this notation:

### Type Theory Symbols

| Symbol | Name | Meaning | Example |
|--------|------|---------|---------|
| $`\vdash`$ | Turnstile | "derives" or "proves" | $`\Gamma  \vdash  A : B`$ |
| `:` | Colon | "has type" | `x : Nat` |
| `G` | Gamma | Context (assumptions) | `x : Int, y : Bool` |
| `->` | Arrow | Function type | `Int -> Bool` |
| `x` | Times | Product type | `Int x Bool` |
| `{x:A \| phi}` | Subset type | Elements of A satisfying phi | `{n:Nat \| n > 0}` |

### Category Theory Symbols

| Symbol | Name | Meaning |
|--------|------|---------|
| `Hom(A, B)` | Hom-set | Morphisms from A to B |
| `F : C -> D` | Functor | Structure-preserving map between categories |
| `y` | Yoneda embedding | Maps objects to representable presheaves |
| `P(T)` | Presheaf category | Category of functors $`T^\text{op}`$ -> Set |
| `Omega` | Subobject classifier | "Type of propositions" |

### OSLF-Specific Notation

| Symbol | Meaning |
|--------|---------|
| `T` | A lambda-theory (the input) |
| `P(T)` | Presheaf topos over T |
| `L(E)` | Internal language of topos E |
| `LP(T)` | Native type theory = L(P(T)) |
| `F!` | Possible next step operator |
| `F*` | Necessary next step operator |
| `diamond` | Eventually (reachability) |
| `square` | Always (invariant) |

### Reading Complex Expressions

Example: $`\Gamma  \vdash  {x:A \| \varphi (x)} \text{type}`$

Read as: "In context G, the subset type {x:A \| phi(x)} is a valid type"

Example: `diamond(hasOutput)`

Read as: "Eventually reaches a state with output" (behavioral type)

---

## The Core Idea

Native Type Theory provides a systematic way to derive a type system from any
operational semantics. The key insight is a 2-functor composition:

```
                    P                     L
    λ-theory  ─────────>  presheaf topos  ─────────>  type system
       T                      P(T)                      LP(T)
```

Where:
- **T** is the operational semantics formalized as $`a \lambda`$-theory
- **P** is the presheaf construction (categorical completion)
- **L** is the internal language functor (extracts the type theory)

The resulting type system LP(T) is called **native** because:
1. Types arise directly from the syntax (no external imposition)
2. The construction preserves all structural properties
3. Behavioral reasoning emerges naturally from the internal graph

---

## $`\lambda`$-Theories with Equality

### Definition: $`\lambda`$-Theory

A **$`\lambda`$-theory** T consists of:

1. **Sorts**: A set S of type sorts (e.g., `Term`, `State`, `Name`)
2. **Operations**: Typed function symbols f : s₁ × ... × sₙ → s
3. **Equations**: Equality axioms between terms

### Example: Simple $`\lambda`$-Theory for Terms

```
Theory T_Terms:
  Sorts: T (terms)

  Operations:
    var  : 1 → T           ; Variables
    app  : T × T → T       ; Application
    lam  : T → T           ; Abstraction

  Equations:
    app(lam(M), N) = M[N]  ; β-reduction (as equation)
```

### MeTTa as $`a \lambda`$-Theory

MeTTa can be formalized with:

```
Theory T_MeTTa:
  Sorts: Term, Atom, List, MSet, State, KB, Receipt

  Operations:
    ; Term constructors
    atom    : Atom → Term
    list    : List → Term
    mset    : MSet → Term

    ; List constructors
    nil     : 1 → List
    cons    : Term × List → List

    ; Multiset constructors
    empty   : 1 → MSet
    insert  : Term × MSet → MSet

    ; State constructor
    state   : Term × KB × MSet × MSet → State

  Equations:
    ; Multiset commutativity
    insert(x, insert(y, m)) = insert(y, insert(x, m))

    ; Multiset associativity
    insert(x, insert(y, insert(z, m))) = insert(x, insert(z, insert(y, m)))
```

---

## The Presheaf Construction

### Definition: Presheaf Category

Given $`a \lambda`$-theory T viewed as a category, the **presheaf category** P(T) consists of:

- **Objects**: Functors F : $`T^\text{op}`$ → Set (contravariant functors to sets)
- **Morphisms**: Natural transformations between functors

### The Yoneda Embedding

The **Yoneda embedding** y : T → P(T) maps each sort s to its representable presheaf:

```
y(s) = Hom_T(−, s)
```

This embedding is:
- **Full and faithful** (preserves all structure)
- **Preserves limits** (products become products)

### Key Properties of P(T)

The presheaf category P(T) is a **topos**, meaning it has:

1. **All finite limits** (products, equalizers, pullbacks)
2. **Exponentials** [P, Q] for any presheaves P, Q
3. **Subobject classifier** $`\Omega`$ (the "type of propositions")
4. **Power objects** $`\Omega P`$ for predicate formation

### Computing Internal Homs

For presheaves P, Q, the internal hom [P, Q] is:

```
[P, Q](c) = Nat(y(c) × P, Q)
```

This represents "functions from P to Q" as a presheaf itself.

---

## The Internal Language Functor

### Definition: Internal Language

Every topos E has an **internal language** L(E), which is a type theory with:

- **Types** from objects of E
- **Terms** from morphisms of E
- **Propositions** from subobjects (maps to $`\Omega )`$
- **Quantifiers** from adjoints to substitution

### The Functor L

L : Topos → TypeTheory extracts:

```
L(E) = ⟨Types, Terms, Props, ⊢⟩
```

Where:
- Types = objects of E
- Terms(A) = global elements 1 → A
- Props(A) = subobjects of A (maps $`A \to  \Omega )`$
- $`\vdash  =`$ derivability from categorical structure

### Composition LP

Composing P and L gives the native type theory:

```
LP(T) = L(P(T))
```

This type theory:
- Has all sorts from T as types
- Has all operations from T as term formers
- Has predicates (types) for any property definable over terms
- Supports full higher-order reasoning

---

## Native Types

### Definition: Native Type

A **native type** in LP(T) is a predicate $`\varphi`$ : $`A \to  \Omega`$ over some object A, expressing
a property that terms of sort A may satisfy.

### Structural Types (Codespaces)

Predicates on term constructors give **structural types**:

```
; Type of lists with length ≥ 2
LongList(l) := ∃x,y,t. l = cons(x, cons(y, t))

; Type of non-empty multisets
NonEmpty(m) := ∃x,m'. m = insert(x, m')
```

### Type Formation Rules

From the internal language, we get type formation rules:

```
Γ ⊢ A type    Γ, x:A ⊢ φ(x) prop
─────────────────────────────────
      Γ ⊢ {x:A | φ(x)} type

Γ ⊢ A type    Γ ⊢ B type
─────────────────────────
   Γ ⊢ A → B type
   Γ ⊢ A × B type
```

### Substitution as Pattern Matching

Given a morphism f : B → A and a predicate $`\varphi`$ : $`A \to  \Omega ,`$ we can form the **substitution**:

```
φ[f] : B → Ω
φ[f](b) = φ(f(b))
```

This captures pattern matching: $`\varphi[\text{unify}(p, -)]`$ checks if a term matches pattern p
with predicate $`\varphi`$ on the result.

---

## Behavioral Types

The key innovation of OSLF is **behavioral types** that reason about computation.

### The Internal Graph

MeTTa's rewrite rules form a graph internal to P(T):

```
G = ⟨s, t⟩ : E → State × State
```

Where:
- E = the object of "edges" (rewrite rule applications)
- s : E → State = source (state before rewrite)
- t : E → State = target (state after rewrite)

### Step Operators

From the internal graph, we define operators on predicates:

```
; Possible next step (existential)
F!(φ) = [s]; ∃t(φ)
       = λstate. ∃e:E. s(e) = state ∧ φ(t(e))

; Necessary next step (universal)
F*(φ) = ∀t(φ[s])
       = λstate. ∀e:E. s(e) = state ⟹ φ(t(e))
```

### Reachability Modalities

Iterating step operators gives reachability:

```
; Eventually (finite iterations of F!)
◇φ = μX. φ ∨ F!(X)

; Always (finite iterations of F*)
□φ = νX. φ ∧ F*(X)
```

### Behavioral Type Examples

```
; Type of states that can reach output
CanOutput := ◇(hasOutput)

; Type of states that always terminate
Terminating := ◇(isFinal)

; Type of states that never modify KB
KBPure := □(kbUnchanged)
```

### Bisimulation as a Type

The paper shows bisimulation can be encoded as an **inductive type**:

```
Bisim := μφ. S(φ)
```

Where S is the simulation relation step. Two states are bisimilar if they satisfy
the greatest fixed point of the simulation condition.

---

## Application to MeTTa

### Step 1: Formalize MeTTa as T_MeTTa

(See [01-metta-operational-semantics.md](./01-metta-operational-semantics.md))

### Step 2: Construct P(T_MeTTa)

The presheaf category gives:
- Representable presheaves for each sort (Term, State, etc.)
- Internal homs for function types
- Power objects for predicate types

### Step 3: Extract LP(T_MeTTa)

The internal language provides:
- Dependent types over MeTTa terms
- Predicates on knowledge bases
- Behavioral modalities for reasoning about evaluation

### Example Types in LP(T_MeTTa)

```
; Query that always succeeds
Decidable(p) := □(match(p, k) ≠ ∅)

; Safe knowledge base modification
SafeAdd(t) := ∀k. consistent(k) ⟹ consistent(k ∪ {t})

; Terminating computation
Terminates(i) := ◇(w = ∅ ∧ i = ε)
```

### Refined Binding

OSLF supports **refined binding** where pattern variables have constrained types:

```
; Query pattern where $x must be a number
(parent $x:Number Bob)

; Encoded as hom type
[$x : Number, parent($x, Bob)]
```

This is represented using the internal hom $`[\varphi , \psi ]`$ for conditioned contexts.

---

## The OSLF Construction Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        T_MeTTa                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Sorts: Term, State, KB, ...                              │   │
│  │ Ops: cons, insert, state, query, chain, ...              │   │
│  │ Eqs: multiset commutativity, ...                         │   │
│  │ Graph: rewrite rules as edges                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ P (presheaf construction)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        P(T_MeTTa)                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Objects: Presheaves (generalized term spaces)            │   │
│  │ Morphisms: Natural transformations                       │   │
│  │ Internal hom: [P,Q] function spaces                      │   │
│  │ Ω: Subobject classifier (propositions)                   │   │
│  │ Internal graph: Rewrite dynamics preserved               │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ L (internal language)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       LP(T_MeTTa)                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Types: {x:A | φ(x)}, A → B, A × B, ∀x:A.B, ∃x:A.B        │   │
│  │ Props: Predicates φ : A → Ω                              │   │
│  │ Modalities: F!, F*, ◇, □ (behavioral)                    │   │
│  │ Bisimulation: μφ.S(φ) (inductive type)                   │   │
│  │ Refined binding: [φ, ψ] (conditioned contexts)           │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Worked Example: LP(T) Construction

Let's walk through the OSLF construction for a minimal theory to make the abstract
machinery concrete.

### Step 1: Define a Simple Lambda-Theory T_Simple

```
Theory T_Simple:
  Sorts: T (terms)

  Operations:
    zero : 1 -> T
    succ : T -> T
    add  : T x T -> T

  Equations:
    add(zero, x) = x
    add(succ(x), y) = succ(add(x, y))
```

This is natural numbers with addition.

### Step 2: Construct P(T_Simple)

The presheaf category P(T_Simple) has:

**Objects**: Functors F : $`T_\text{Simple}^\text{op}`$ -> Set

Key presheaves include:
- `y(T)` = Hom(-, T) - the representable presheaf for terms
- `Omega` - the subobject classifier (propositions)
- `Omega^{y(T)}` - predicates on terms

**Morphisms**: Natural transformations

**Example presheaf**: "Even numbers"

```
Even(c) = { t in Hom(c, T) | t = add(x, x) for some x }
```

This is a sub-presheaf of y(T).

### Step 3: Extract L(P(T_Simple))

The internal language gives us:

**Types**:
- `T` (terms, from the sort)
- `T -> T` (functions on terms)
- `{x:T | Even(x)}` (subset type of even numbers)
- `{x:T | x = zero}` (singleton type)

**Type formation rules**:
```
Γ ⊢ T type

Γ ⊢ A type    Γ, x:A ⊢ φ prop
----------------------------------
Γ ⊢ {x:A | φ} type
```

**Terms and their types**:
```
⊢ zero : T
⊢ succ : T → T
⊢ add : T × T → T
```

**Propositions (predicates)**:
```
Γ, x:T ⊢ x = zero prop
Γ, x:T ⊢ Even(x) prop
```

### Step 4: Add Reductions (Make it a GSLT)

Extend with a reduction relation:

```
Extended Theory:
  Sorts: T, R (reductions)

  New Operations:
    src : R -> T
    tgt : R -> T
    step : T x T -> R   (when add can reduce)

  Equations:
    src(step(x, y)) = add(x, y)
    tgt(step(zero, y)) = y
    tgt(step(succ(x), y)) = succ(add(x, y))
```

### Step 5: Behavioral Types from Reductions

Now LP(T_Simple) includes behavioral types:

```
; Type of terms that reduce to zero
ReducesToZero(t) := diamond(t = zero)

; Expanded: exists n:Nat, r1:R, ..., rn:R.
;   src(r1) = t /\ tgt(r1) = src(r2) /\ ... /\ tgt(rn) = zero

; Type of terms that are "normal" (can't reduce)
Normal(t) := not(exists r:R. src(r) = t)
```

### Summary of the Construction

```
T_Simple (lambda-theory)
    |
    | P (presheaf construction)
    v
P(T_Simple) (topos with internal graph)
    |
    | L (internal language)
    v
LP(T_Simple) (native type theory)
    - Basic types: T
    - Function types: T -> T
    - Subset types: {x:T | phi}
    - Behavioral types: diamond(phi), square(phi)
```

---

## OSLF vs Gph-Theory Tradeoffs

### When to Use Each Approach

| Criterion | Use OSLF | Use Gph-Theory |
|-----------|----------|----------------|
| **Binding** | Complex binding patterns | Binding via reflection/combinators |
| **Types needed** | Full dependent types, refinements | Structural types, modal types |
| **Reasoning** | Bisimulation, behavioral specs | Operational semantics only |
| **Implementation effort** | High | Medium |
| **Theory maturity** | Well-established | Newer, simpler |

### Feature Comparison

| Feature | OSLF | Gph-Theory | Both |
|---------|------|------------|------|
| Structural types | Yes | Yes | Yes |
| Modal types | Via modalities | Via interaction | Yes |
| Dependent types | Full | Limited | Partial |
| Behavioral predicates | Yes | No | No |
| Bisimulation | Inductive type | No | No |
| Binding | Native presheaves | Via combinators | Different |
| Compilation target | N/A | Rholang | Different |

### The Binding Decision

The key decision point:

> **Can your calculus eliminate binding via reflection or combinators?**

**Yes (use Gph-theories)**:
- RHO calculus: quote/drop eliminates binding
- SKI: Combinators eliminate lambda
- MeTTa: Quote/eval may suffice

**No (use OSLF)**:
- Lambda calculus with explicit binding
- Calculi where binding is semantic (not just syntactic)
- Systems requiring full dependent types

### Hybrid Approach

For MeTTa, the recommended approach is:

1. **Start with Gph-theories** for core operational semantics
2. **Use type lifting** ([05-type-lifting.md](./05-type-lifting.md)) for structural types
3. **Add OSLF elements** selectively where behavioral types are needed
4. **Fall back to full OSLF** only for complex refinement types

### Implementation Complexity

| Aspect | OSLF | Gph-Theory |
|--------|------|------------|
| Category theory required | Yes, significant | Minimal |
| Implementation size | Large | Medium |
| Proof effort | High | Medium |
| Runtime performance | N/A (static) | Can compile to Rholang |

---

## Summary

Native Type Theory (OSLF) provides:

1. **Systematic derivation** of types from operational semantics
2. **Structural types** via predicates on term constructors
3. **Behavioral types** via internal graph modalities
4. **Bisimulation** as an inductive type
5. **Refined binding** for constrained pattern variables

This is the full theoretical foundation for semantic type checking in MeTTa. For a
simpler approach when binding can be eliminated, see
[03-gph-enriched-lawvere.md](./03-gph-enriched-lawvere.md).

**Related documents**:
- [05-type-lifting.md](./05-type-lifting.md): Type lifting transformation rules
- [06-inference-rules.md](./06-inference-rules.md): Practical guide to inference rules

---

## References

- Williams, P. & Stay, M. "Native Type Theory." EPTCS 372, pp. 116-132, 2022.
- Jacobs, B. "Categorical Logic and Type Theory." Elsevier, 1998.
- See [bibliography.md](../reference/bibliography.md) for complete references.
