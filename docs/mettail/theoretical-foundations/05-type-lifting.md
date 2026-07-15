[← Documentation Index](../../README.md)

# Type Lifting: Deriving Types from Operational Semantics

This document explains the **type lifting transformation** - how MeTTaIL systematically
derives a typed calculus from an untyped Graph-Structured Lambda Theory (GSLT). This
transformation is central to MeTTaIL's approach to semantic type checking.

**Target audience**: Compiler engineers implementing type systems

**Prerequisites**: Read [03-gph-enriched-lawvere.md](./03-gph-enriched-lawvere.md) for
GSLT fundamentals.

---

## Table of Contents

1. [Quick Reference for Implementers](#quick-reference-for-implementers)
2. [Motivation](#motivation)
3. [The Type Lifting Transformation](#the-type-lifting-transformation)
4. [The Duplication Rule](#the-duplication-rule)
5. [Lifting Equations](#lifting-equations)
6. [Possibility Modal Types](#possibility-modal-types)
7. [Complete Worked Examples](#complete-worked-examples)
8. [Implementation Considerations](#implementation-considerations)

---

## Quick Reference for Implementers

### The Three Transformation Rules

```
T(G)     = G                         -- shapes stay shapes
T(A x B) = T(A) x T(B)               -- products lift pointwise
T(A -> B) = T(A) x (T(A) -> T(B))    -- exponentials add type param
```

### What You Need to Implement

1. **Parse GSLT specification** - extract generating sorts, morphisms, equations
2. **Apply T(-) to each morphism** - generate type-lifted symbols
3. **Check for duplications** - extend symbols where inputs appear multiple times
4. **Lift equations** - transform term equations to type equations
5. **Extract modal types** - analyze reduction sources for context types

### Data Structures

```
// Pseudocode representation
struct Sort { name: String, is_generating: bool }

struct Morphism {
    name: String,
    inputs: Vec<Arity>,
    output: Sort,
}

enum Arity {
    Sort(Sort),
    Product(Box<Arity>, Box<Arity>),
    Exponential(Box<Arity>, Box<Arity>),  // A -> B
}

struct TypeLiftedMorphism {
    original: Morphism,
    lifted_name: String,     // e.g., "!!" for "!"
    lifted_inputs: Vec<Arity>,
    lifted_output: Sort,
    extra_factors: Vec<Sort>,  // from duplication rule
}
```

---

## Lambda Theory Presentations (Behavior Framework)

Following Wells & Stay's "Behavior in Higher-Order Languages", we can view the type
lifting transformation through a more unified lens. A **lambda theory presentation** T
consists of four components:

1. **T_type**: A set of base types (our "generating shapes")
2. **T_oper**: Operation declarations of form $`\Gamma  \vdash  f(x⃗) : B`$
3. **T_prop**: Proposition declarations (typically $`p : \text{Pr}, q : \text{Pr} \vdash  p \rightsquigarrow  q`$ for rewriting)
4. **T_ent**: Entailments (rewrite rules and context rules)

### Relationship to GSLT Notation

| GSLT (Current)           | Lambda Theory (New)              | Description                        |
|--------------------------|----------------------------------|------------------------------------|
| `shapes P, N, R`         | `T_type = {Pr, Nm}`              | Base types / generating shapes     |
| `fn sym f: A -> B`       | $`T_\text{oper}: a : A \vdash  f(a) : B`$       | Operation / function symbol        |
| `rewrite r: A -> R`      | $`T_\text{prop}: p,q : \text{Pr} \vdash  p \rightsquigarrow  q`$       | Rewrite proposition                |
| `equation e`             | $`T_\text{ent}: \Gamma  \| \Phi  \vdash  t \rightsquigarrow  u`$          | Entailment / inference rule        |

### Example: RHO Calculus in Lambda Theory Format

```
T_type = {Pr, Nm}

T_oper = {
  ⊢ 0 : Pr
  p₁ : Pr, p₂ : Pr ⊢ p₁ | p₂ : Pr
  n : Nm, p : Pr ⊢ out(n, p) : Pr
  n : Nm, λx.q : [Nm → Pr] ⊢ in(n, λx.q) : Pr
  p : Pr ⊢ @p : Nm
  n : Nm ⊢ *n : Pr
}

T_prop = {(p : Pr, q : Pr ⊢ p ⇝ q)}

T_ent = {
  n : Nm, p : Pr, λx.q : [Nm → Pr] | ⊤ ⊢ out(n,p) | in(n,λx.q) ⇝ q[@p/x]
  p₁,p₂,q : Pr | (p₁ ⇝ p₂) ⊢ p₁ | q ⇝ p₂ | q
  p : Pr | ⊤ ⊢ *(@p) ⇝ p
}
```

### Benefits of This Framework

This format provides:
- **Explicit typing context** for each operation
- **Uniform treatment of rewrites** as entailments with propositions as hypotheses
- **Compositional structure** enabling automatic derivation of transition systems via RPOs
- **Clear separation** of syntax (T_oper) from dynamics (T_ent)

The type lifting transformation described below can be seen as constructing a
**fibered category** SubT → T where:
- Propositions are subobjects $`\varphi`$ ↣ $`\Gamma`$
- Entailment corresponds to inclusion
- Substitution is pullback
- Subset types $`\{x : A \mid \varphi\}`$ arise naturally

---

## Motivation

### The Goal

Given a GSLT describing an untyped calculus, we want to systematically produce a
**typed** version where:

1. Every term constructor `f` has a "type-level" counterpart (conventionally `ff`)
2. The types capture behavioral information from the rewrite rules
3. We can derive typing rules mechanically

### Connection to MeTTa

MeTTa's operational semantics (see [01-metta-operational-semantics.md](./01-metta-operational-semantics.md))
can be formalized as a GSLT. Type lifting transforms this into a typed theory that
captures MeTTa's reduction behavior at the type level.

| MeTTa Concept | Untyped | Type-Lifted |
|---------------|---------|-------------|
| Query operation | `query: Term x KB -> State` | `queryquery: T(Term) x T(KB) -> T(State)` |
| List construction | `cons: Term x List -> List` | `conscons: T(Term) x T(List) -> T(List)` |
| Multiset insertion | `insert: Term x MSet -> MSet` | `insertinsert: T(Term) x T(MSet) -> T(MSet)` |

### The Key Intuition

Every function symbol `f` gets a "shadow" function symbol (type-lifted version):

```
Original:     f: inputs -> output
Type-lifted:  ff: T(inputs) -> T(output)
```

Where `T(-)` is a transformation on arities. The type `ff(...)` represents the
**structural type** of a term built with `f(...)`.

---

## The Type Lifting Transformation

### Rule 1: Generating Shapes

```
T(G) = G    (for any generating shape G)
```

**Meaning**: Types of things of shape G are also of shape G.

**Why**: A type of a process is still a process (structurally). Types describe the
"shape" of what could be in a position.

**Examples across calculi**:

| Calculus | Original | Type-lifted |
|----------|----------|-------------|
| lambda | `App: P x P -> P` | `AppApp: P x P -> P` |
| SKI | `S: 1 -> P` | `SS: 1 -> P` |
| RHO | `*: N -> P` | `**: N -> P` |
| Ambient | `open: N -> M` | `openopen: N -> M` |
| **MeTTa** | `atom: Atom -> Term` | `atomatom: Atom -> Term` |

### Rule 2: Products

```
T(A x B) = T(A) x T(B)
```

**Meaning**: Products transform component-wise.

**Why**: The type of a pair is a pair of types.

**Examples across calculi**:

| Calculus | Original | Type-lifted |
|----------|----------|-------------|
| lambda | `App: P x P -> P` | `AppApp: P x P -> P` |
| RHO | `\|: P x P -> P` | `\|\|: P x P -> P` |
| RHO | `!: N x P -> P` | `!!: N x P -> P` |
| Ambient | `[]: N x P -> P` | `[][]: N x P -> P` |
| **MeTTa** | `cons: Term x List -> List` | `conscons: Term x List -> List` |

### Rule 3: Exponentials (The Crucial Rule)

```
T(A -> B) = T(A) x (T(A) -> T(B))
```

**Meaning**: An exponential becomes a product of:
1. The type of the bound variable `T(A)`
2. A function from types to types `T(A) -> T(B)`

**Why**: To type a binding construct, we need:
- What type the bound variable has
- How the body's type depends on the bound variable's type

This enables **dependent types** - the type of the body can depend on what's bound.

**Examples across calculi**:

| Calculus | Original | Type-lifted | Extra factor |
|----------|----------|-------------|--------------|
| lambda | `Lam: (P -> P) -> P` | `LamLam: P x (P -> P) -> P` | Type of bound variable |
| RHO | `?: N x (N -> P) -> P` | `??: N x N x (N -> P) -> P` | Type of received name |
| Ambient | `nu: (N -> P) -> P` | `nunu: N x (N -> P) -> P` | Type of restricted name |

### Detailed Example: RHO Receive

Let's trace through the transformation for RHO's receive operator:

**Original**: `?: N x (N -> P) -> P`

1. The input arity is `N x (N -> P)`
2. Apply T:
   - `T(N x (N -> P))`
   - `= T(N) x T(N -> P)` (Rule 2)
   - `= N x T(N -> P)` (Rule 1)
   - `= N x (T(N) x (T(N) -> T(P)))` (Rule 3)
   - `= N x (N x (N -> P))` (Rule 1)
   - `= N x N x (N -> P)` (associativity)

**Result**: `??: N x N x (N -> P) -> P`

**Interpretation**: The type of a receive `?(x, lambda y.body)` is `??(X, Y, lambda y.B)` where:
- `X` is the type of the channel `x`
- `Y` is the type of names the continuation expects
- `lambda y.B` is the type of the body as a function of the received name's type

### Meta-Context Interpretation

The type lifting rules can be understood through the **meta-context** formalism
(Definition 1 in Wells & Stay's paper). A **meta-context** $`\langle -\rangle : T \to  T`$ is an
endofunctor on the type category generated by:

- $`- \times  X`$ (product on the left)
- $`[X \to  -]`$ (exponential)

The type lifting transformation T(-) corresponds to:

| Input Arity A        | T(A) as Meta-Context              | Interpretation                     |
|----------------------|-----------------------------------|------------------------------------|
| Generating shape G   | `G` (identity)                    | Types stay in same shape           |
| Product A₁ × A₂      | `T(A₁) × T(A₂)`                   | Product of lifted types            |
| Arrow A₁ → A₂        | `T(A₁) × [T(A₁) → T(A₂)]`         | Type + dependent type function     |

**Key insight**: The exponential rule `T(A → B) = T(A) × [T(A) → T(B)]` captures
that typing a binder requires both:
1. The type of the bound variable
2. How the body's type depends on that type

This structure is what enables dependent types to emerge naturally from operational
semantics.

**Rewrite operations** in the meta-context formalism have the form:

```
∏⟨Γᵢ⟩ᵢ | ⋀⟨pᵢ ⇝ qᵢ⟩ᵢ ⊢ f(p⃗) ⇝ f(q⃗)
```

This says: "given contexts $`\Gamma _{i}`$ and rewrite hypotheses pᵢ ⇝ qᵢ, the operation f
applied to the sources rewrites to f applied to the targets."

---

## The Duplication Rule

The basic type lifting rules aren't quite sufficient. We need an additional rule
for handling **duplicated inputs** in reduction sources.

### The Problem

Consider RHO's comm reduction:

```
comm: N x (N -> P) x P -> R
src(comm(x, K, Q)) = |(?(x, K), !(x, Q))
```

Notice that `x` appears **twice** in the source:
- In `?(x, K)` - the channel being listened on
- In `!(x, Q)` - the channel being sent on

This duplication affects typing because the types of `!` and `?` need to "know"
they're on the same channel.

### The Rule

**Duplication Rule**: If an input `a_j: A_j` to a base reduction appears in
multiple function symbols in the source, extend each of those function symbols'
type-lifted version by `A_j`.

### RHO Example: Detailed Walkthrough

**Step 1**: Identify the base reduction

```
comm: N x (N -> P) x P -> R
```

Inputs: `x: N`, `K: N -> P`, `Q: P`

**Step 2**: Examine the source

```
src(comm(x, K, Q)) = |(?(x, K), !(x, Q))
```

**Step 3**: Find duplications

- `x` appears in `?(x, K)` - input to `?`
- `x` appears in `!(x, Q)` - input to `!`

So `x` is duplicated between `?` and `!`.

**Step 4**: Extend type-lifted symbols

Without duplication rule:
- `!: N x P -> P` becomes `!!: N x P -> P`
- `?: N x (N -> P) -> P` becomes `??: N x N x (N -> P) -> P`

With duplication rule (add `x N` for the channel):
- `!!: N x P -> P` becomes `!!: N x P x N -> P`
- `??: N x N x (N -> P) -> P` becomes `??: N x N x (N -> P) x N -> P`

**Interpretation**:
- `!!(A, B, x)` - type of send on channel `x` sending something of type `B`
- `??(A, B, lambda y.C, x)` - type of receive on channel `x` expecting type `B`

The extra `x N` ensures the types know which channel is involved.

### Other Calculi

**lambda-calculus**: No duplication in beta rule
```
src(beta(K, N)) = App(Lam(K), N)
```
Each input appears once, so no extension needed.

**SKI**: Duplication in S combinator
```
src(sigma3(x, y, z)) = App(S2(x, y), z)
tgt(sigma3(x, y, z)) = App(App(x, z), App(y, z))
```
`z` appears twice on the RHS, affecting the types.

**Ambient**: Multiple duplications
```
src(in_red(n, m, Q, R, S)) = |([](n, |(.(in(m), Q), R)), [](m, S))
```
Both `n` and `m` appear in multiple places.

---

## Lifting Equations

Equations in the untyped GSLT lift to equations in the typed GSLT.

### The Universal Pattern

If `f(...) = g(...)` is an equation in the untyped GSLT, then `ff(...) = gg(...)`
is an equation in the typed GSLT (with appropriate lifted arities).

### RHO Examples

**Parallel composition properties**:

| Untyped | Typed |
|---------|-------|
| `\|(P, Q) = \|(Q, P)` | `\|\|(T, U) = \|\|(U, T)` |
| `\|(\|(P, Q), R) = \|(P, \|(Q, R))` | `\|\|(\|\|(T, U), V) = \|\|(T, \|\|(U, V))` |
| `\|(P, 0) = P` | `\|\|(T, 00) = T` |

**Quote/unquote**:

| Untyped | Typed |
|---------|-------|
| `@(*x) = x` | `@@(**A) = A` |

### Source/Target Equations

The source and target equations lift to `srcsrc` and `tgttgt`:

**Untyped comm**:
```
src(comm(x, K, Q)) = |(?(x, K), !(x, Q))
tgt(comm(x, K, Q)) = ev(K, @(Q))
```

**Typed comm**:
```
srcsrc(commcomm(A, B, lambda y.C, x)) = ||(?(A, B, lambda y.C, x), !!(A, **(B), x))
tgttgt(commcomm(A, B, lambda y.C, x)) = C
```

Note:
- `commcomm` is the type of a comm reduction
- `srcsrc` gives the type of the source
- `tgttgt` gives the type of the target
- The target type is `C` (the body type), possibly with substitution

---

## Possibility Modal Types

The most interesting part of type lifting is the derivation of **modal types**
from reduction structure.

### The Key Insight

In a reduction source like:
```
src(comm(x, K, Q)) = |(?(x, K), !(x, Q))
```

Each **subtree** of the AST can be viewed in its surrounding **context**:

```
                |
               / \
          ?(x,K)  !(x,Q)
           /  \    /  \
          x    K  x    Q
```

Subtrees:
1. `Q` in context `|(?(x,K), !(x, []))`
2. `?(x,K)` in context `|([], !(x,Q))`
3. `!(x,Q)` in context `|(?(x,K), [])`

Each subtree's position gives rise to a **possibility modal type**.

### What Modal Types Mean

The type `A: <K[-]>B` means:

> "A term of type A, when placed in context K, **possibly reduces** to something
> of type B"

For RHO's comm rule:
- `Q: <|(?(x,K), !(x,[]))>C` means "Q, when sent on channel x, enables reduction
  to continuation body C"

### Independent vs Dependent

**Independent modal types** (`_i`): Only track the target **type**
```
ctxposs_i(T)     -- possibly reduces to type T
ctxcomm_i(...)   -- in comm context, reduces to type T
```

**Dependent modal types** (`_d`): Also track the target **term**
```
ctxposs_d(T, t)      -- possibly reduces to term t of type T
ctxcomm_d(..., t)    -- in comm context, reduces to term t
```

**When to use each**:
- Independent: When you only care about the shape of the result
- Dependent: When you need to track exactly what the result is

### RHO Modal Types

From the comm reduction, we derive:

| Modal Type | Meaning |
|------------|---------|
| `ctxrecv_i(T, U, V, W, X)` | Process type that, when juxtaposed with receive type `??(T, U, V, W)`, possibly reduces to type X |
| `ctxsend_i(T, U, V, W)` | Process type that, when juxtaposed with send type `!!(T, U, V)`, possibly reduces to type W |
| `ctxcomm_i(T, U, V, W, X)` | Process type that, when sent and received, possibly reduces to type X |
| `ctxposs_i(T)` | General possibility: process that possibly reduces to type T |

Plus dependent versions `ctxrecv_d`, `ctxsend_d`, `ctxcomm_d`, `ctxposs_d`.

### Modal Type Equations

These modal types satisfy equations relating contexts to general possibility:

```
||(ctxrecv_i(T, U, V, W, X), ??(T, U, V, W)) = ctxposs_i(X)
||(!!(T, ctxcomm_i(T, U, V, W, X), W), ??(T, U, V, W)) = ctxposs_i(X)
||(ctxsend_i(T, U, V, W), !!(T, U, V)) = ctxposs_i(W)
```

**Reading the first equation**: A process with receive-context modal type, in
parallel with a matching receive, has general possibility type.

---

## Complete Worked Examples

### Lambda-Calculus Transformation

**Untyped GSLT**:
```
App: P x P -> P
Lam: (P -> P) -> P
beta: (P -> P) x P -> R
```

**Apply transformation**:

1. `T(P x P) = P x P` -> `AppApp: P x P -> P`
2. `T((P -> P)) = P x (P -> P)` -> `LamLam: P x (P -> P) -> P`
3. `T((P -> P) x P) = T(P -> P) x T(P) = (P x (P -> P)) x P` -> `betabeta: P x (P -> P) x P -> R`

**Check for duplication in beta**:
```
src(beta(K, N)) = App(Lam(K), N)
```
- `K` appears once (in `Lam(K)`)
- `N` appears once (as second arg to `App`)

No duplication, so no extension needed.

**Typed GSLT**:
```
AppApp: P x P -> P
LamLam: P x (P -> P) -> P
betabeta: P x (P -> P) x P -> R
```

**Modal types**: From the beta reduction:
- `<App(Lam([]), N)>B` - abstraction in application context
- `<App(Lam(K), [])>B` - argument in application context

These give rise to arrow types! The context `<App(Lam([]), N)>B` is essentially
saying "a function that, applied to N, reduces to B".

### SKI Transformation

**The S combinator creates interesting duplication**:

```
sigma3: P x P x P -> R
src(sigma3(x, y, z)) = App(S2(x, y), z)
tgt(sigma3(x, y, z)) = App(App(x, z), App(y, z))
```

`z` appears **twice** in the target (in `App(x, z)` and `App(y, z)`).

This means:
- `x` has a modal type dependent on `z`
- `y` has a modal type dependent on `z` similarly

The result: S has type that looks like `(C -> B -> A) -> (C -> B) -> C -> A`.

### RHO Transformation Summary

**Type-lifted symbols** (with duplication):

```
00: 1 -> P
||: P x P -> P
!!: N x P x N -> P      -- extra N for channel
??: N x N x (N -> P) x N -> P  -- extra N for channel
**: N -> P
@@: P -> N
srcsrc: R -> P
tgttgt: R -> P
commcomm: N x N x (N -> P) x N -> R
```

**Modal types**:
```
ctxrecv_i: N x N x (N -> P) x N x P -> P
ctxrecv_d: N x N x (N -> P) x N x P x P -> P
ctxsend_i: N x P x N x P -> P
ctxsend_d: N x P x N x P x P -> P
ctxcomm_i: N x N x (N -> P) x N x P -> P
ctxcomm_d: N x N x (N -> P) x N x P x P -> P
ctxposs_i: P -> P
ctxposs_d: P x P -> P
```

### MeTTa Transformation (Sketch)

Based on the MeTTa state machine model from [01-metta-operational-semantics.md](./01-metta-operational-semantics.md):

**Untyped operations**:
```
state: Term x KB x MSet x MSet -> State
query: Term -> R   (as part of state transition)
cons: Term x List -> List
insert: Term x MSet -> MSet
```

**Type-lifted operations**:
```
statestate: Term x KB x MSet x MSet -> State   -- all generating sorts
queryquery: Term -> R
conscons: Term x List -> List
insertinsert: Term x MSet -> MSet
```

**Multiset equations lift**:
```
insert(x, insert(y, m)) = insert(y, insert(x, m))

becomes:

insertinsert(X, insertinsert(Y, M)) = insertinsert(Y, insertinsert(X, M))
```

### Comparison: Which Rules Apply Where

| Transformation | lambda | SKI | RHO | Ambient | MeTTa |
|----------------|--------|-----|-----|---------|-------|
| Rule 1 (shapes) | App, Lam | App, S, K, I, S1, S2, K1 | All | All | All |
| Rule 2 (products) | App | App | \|, !, [], etc. | \|, [], ., etc. | cons, insert, state |
| Rule 3 (exponentials) | Lam | None | ? | nu | Pattern bindings |
| Duplication | None | S combinator (z) | comm (channel x) | in, out (n, m) | Query matches |
| Modal types | head context | head context | par, comm, exec | ambient, par, in/out/open | Query, chain, transform |

---

## Implementation Considerations

### Algorithm Overview

```python
def type_lift_gslt(gslt: GSLT) -> TypedGSLT:
    result = TypedGSLT()

    # 1. Copy generating sorts
    for sort in gslt.generating_sorts:
        result.add_sort(sort)

    # 2. Transform each morphism
    for morph in gslt.morphisms:
        lifted = lift_morphism(morph)
        result.add_morphism(lifted)

    # 3. Find duplications in reductions
    duplications = find_duplications(gslt.reductions)

    # 4. Extend morphisms per duplication rule
    for (morph, extra_factors) in duplications:
        result.extend_morphism(morph, extra_factors)

    # 5. Lift equations
    for eq in gslt.equations:
        result.add_equation(lift_equation(eq))

    # 6. Extract modal types from reduction sources
    for reduction in gslt.reductions:
        modals = extract_modal_types(reduction)
        for modal in modals:
            result.add_morphism(modal)

    return result

def lift_arity(arity: Arity) -> Arity:
    """Apply T(-) to an arity"""
    match arity:
        case Sort(s):
            return Sort(s)  # Rule 1
        case Product(a, b):
            return Product(lift_arity(a), lift_arity(b))  # Rule 2
        case Exponential(a, b):
            # Rule 3: T(A -> B) = T(A) x (T(A) -> T(B))
            ta = lift_arity(a)
            tb = lift_arity(b)
            return Product(ta, Exponential(ta, tb))

def lift_morphism(morph: Morphism) -> TypeLiftedMorphism:
    lifted_inputs = [lift_arity(inp) for inp in morph.inputs]
    return TypeLiftedMorphism(
        original=morph,
        lifted_name=morph.name + morph.name,  # ff convention
        lifted_inputs=lifted_inputs,
        lifted_output=morph.output,  # sorts stay same
        extra_factors=[]
    )
```

### Finding Duplications

```python
def find_duplications(reductions: List[Reduction]) -> List[Tuple[Morphism, List[Sort]]]:
    """
    For each reduction, find which inputs appear multiple times in the source.
    Return list of (morphism, extra_factors) pairs.
    """
    result = []

    for red in reductions:
        # Parse source expression
        source_ast = parse(red.source_equation)

        # Count occurrences of each input variable
        input_counts = count_variable_occurrences(source_ast, red.inputs)

        # Find duplicated inputs
        duplicated = [inp for inp, count in input_counts.items() if count > 1]

        # For each morphism in source that uses a duplicated input
        for morph in morphisms_in_ast(source_ast):
            used_duplicates = [inp for inp in duplicated if uses_input(morph, inp)]
            if used_duplicates:
                extra_factors = [inp.sort for inp in used_duplicates]
                result.append((morph, extra_factors))

    return result
```

### Extracting Modal Types

```python
def extract_modal_types(reduction: Reduction) -> List[ModalType]:
    """
    Extract context-based modal types from reduction source.
    Each subtree position generates independent and dependent modal types.
    """
    result = []
    source_ast = parse(reduction.source_equation)

    # Walk AST, creating context for each subtree
    for subtree, context in subtrees_with_contexts(source_ast):
        # Independent modal type: tracks result type only
        independent = ModalType(
            name=f"ctx{context.name}_i",
            params=context.type_params + [subtree.type],
            result=Sort("P"),  # process type
            meaning=f"possibly reduces to type in context {context}"
        )
        result.append(independent)

        # Dependent modal type: tracks result type and term
        dependent = ModalType(
            name=f"ctx{context.name}_d",
            params=context.type_params + [subtree.type, Sort("P")],
            result=Sort("P"),
            meaning=f"possibly reduces to specific term of type in context {context}"
        )
        result.append(dependent)

    return result
```

### Performance Considerations

1. **Memoize arity transformations**: `T(A)` for the same `A` always gives the
   same result
2. **Cache AST parsing**: Source equations are parsed multiple times
3. **Incremental updates**: When GSLT changes, only recompute affected parts
4. **Lazy modal type generation**: Generate modal types on demand rather than
   upfront

### Integration with MeTTaIL

The type lifting transformation is implemented in MeTTaIL's Scala prototype. Key files:

- Theory definitions: Specify GSLTs in a structured format
- Hypercube transformation: Applies `T(-)` rules
- BNFC generation: Produces parsers from transformed GSLTs

See [01-mettail-scala-prototype.md](../implementation/01-mettail-scala-prototype.md) for details.

---

## Behavior-Preserving Encodings

When translating between calculi (e.g., encoding lambda-calculus in RHO, or
encoding MeTTa in a core calculus), we need formal criteria for when a translation
correctly preserves operational behavior.

### Definition: Encoding (Wells & Stay, Definition 27)

An **encoding** $`[[-]] : S \to  T`$ from source calculus S to target calculus T must satisfy:

1. **Preserve products**: $`[[\Gamma  \times  \Delta ]] = [[\Gamma ]] \times  [[\Delta ]]`$
2. **Preserve pullbacks**: $`[[\varphi [\sigma ]]] = [[\varphi ]][[\sigma ]]`$ (substitution commutes with encoding)
3. **Preserve abstraction**: `[[[A → B]]] = [[[A] → [[B]]]]`
4. **Map process type appropriately**: `Pr_S` maps to $`\langle \text{Pr}_T\rangle`$ for some meta-context $`\langle -\rangle`$
5. **Preserve rewriting**: $`p \rightsquigarrow  q \vdash  [[p]] \to *_\tau  [[q]]`$ (source rewrites become silent paths)
6. **Preserve bisimilarity**: $`p \approx  q \vdash  [[p]] \approx  [[q]]`$
7. **Preserve behavior**: `p ≈⃗ [[p]]` (source is behaviorally equivalent to its encoding)

### What This Means for Implementers

When implementing a translation pass:

- **Structural preservation** (1-4): The translation should be homomorphic with
  respect to products, function types, and substitution
- **Operational preservation** (5): Source reductions must produce corresponding
  (possibly multi-step) silent transitions in the target
- **Behavioral preservation** (6-7): Behaviorally equivalent source terms must
  encode to behaviorally equivalent target terms

### Example: Lambda-Calculus in RHO

The standard encoding of lambda-calculus in RHO:

```
[[x]]           = *x
[[λx.M]]        = for(u, λv. v!(u, [[M]]))  -- for fresh name u
[[M N]]         = (νu)( [[M]] | u?(λx.x, [[N]]) )
```

This encoding:
- Maps lambda terms to RHO processes
- Maps beta-reduction to comm-reduction (with extra steps)
- Preserves behavioral equivalence (bisimilarity)

### Verifying Encodings

To verify an encoding preserves behavior:

1. **Simulation**: Show that each source reduction step is simulated by the target
2. **Bisimulation**: Show that source and encoded term are bisimilar
3. **Congruence**: Verify the encoding respects contexts (see Transparency below)

---

## Scala Implementation Mapping

The theoretical concepts map to the existing Scala implementation as follows:

| Theoretical Concept              | Scala Location                                   |
|----------------------------------|--------------------------------------------------|
| Lambda theory presentation       | `BasePres` in `TheoryEnv.scala`                  |
| Base types (T_type)              | `IdCat` case class                               |
| Products (×)                     | `ProdCat(left, right)`                           |
| Exponentials (→)                 | `ArrowCat(dom, cod)`                             |
| Function declarations            | `FnDecl` in `ModuleProcessor.scala`              |
| Rewrite declarations             | `RewriteDecl` type                               |
| Type-lifting transformation      | `Hypercube.liftTypes()` method                   |
| Pipeline orchestration           | `Pipeline.scala` pass architecture               |
| Theory environment               | `TheoryEnv.scala` context management             |
| BNFC syntax rendering            | `BNFCRenderer.scala`                             |
| Binder handling                  | `DesugarBinds.scala`                             |

### Key Data Structures

```scala
// Arities (categories in the lambda theory)
sealed trait Cat
case class IdCat(name: String) extends Cat                    // T_type elements
case class ProdCat(left: Cat, right: Cat) extends Cat         // Products
case class ArrowCat(dom: Cat, cod: Cat) extends Cat           // Exponentials

// Function declarations (T_oper elements)
case class FnDecl(name: String, inputs: List[Cat], output: Cat)

// Rewrite declarations (T_ent elements)
case class RewriteDecl(name: String, source: Term, target: Term)
```

---

## Summary

The type lifting transformation:

1. **Rule 1**: `T(G) = G` - shapes stay shapes
2. **Rule 2**: `T(A x B) = T(A) x T(B)` - products lift pointwise
3. **Rule 3**: `T(A -> B) = T(A) x (T(A) -> T(B))` - exponentials add a type parameter
4. **Duplication**: Duplicated inputs in reduction sources extend type-lifted symbols
5. **Modal types**: Each AST subtree in a reduction source generates a possibility modal type

This mechanical transformation turns operational semantics into type structure.

---

## Next Steps

- [06-inference-rules.md](./06-inference-rules.md): How to read and derive typing rules
- [02-native-type-theory-oslf.md](./02-native-type-theory-oslf.md): Full OSLF construction for behavioral types

---

## References

- Wells, P. & Stay, M. "Behavior in Higher-Order Languages." 2024.
  (Primary reference for lambda theory presentations, RPO framework, and encoding criteria)
- Arkor, N. & McDermott, D. "Formal metatheory of programming languages."
- Stay, M. & Meredith, L. G. "Representing operational semantics with enriched
  Lawvere theories." arXiv:1704.03080, 2017.
- Meredith, L. G. et al. "MeTTa Architecture Proposal." (hypercube/map.md)
- Milner, R. "Deriving bisimulation congruences for reactive systems."
  (Foundation for RPO-based transition system derivation)
- See [bibliography.md](../reference/bibliography.md) for complete references.
