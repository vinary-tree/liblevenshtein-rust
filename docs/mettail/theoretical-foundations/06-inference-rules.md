[← Documentation Index](../../README.md)

# Inference Rules: A Practical Guide for Implementers

This document explains the **typing rules** that arise from type-lifted GSLTs. We
cover the notation, core rules, and how to derive and read typing judgments.

**Target audience**: Compiler engineers implementing type checkers

**Prerequisites**: Read [05-type-lifting.md](./05-type-lifting.md) for the transformation rules.

---

## Table of Contents

1. [Reading Type Theory Notation](#reading-type-theory-notation)
2. [The Typing Hypercube](#the-typing-hypercube)
3. [Core Inference Rules](#core-inference-rules)
4. [Derived Rules for Term Constructors](#derived-rules-for-term-constructors)
5. [Reduction Typing](#reduction-typing)
6. [Complete Derivation Examples](#complete-derivation-examples)
7. [Implementation Patterns](#implementation-patterns)

---

## Reading Type Theory Notation

This section is **critical** for understanding the rest of the documentation.
If you're unfamiliar with type theory notation, read this section carefully.

### The Colon (:) - "Has Type"

The most fundamental symbol:

```
A : B     means "A has type B"
```

Examples:
- `0 : Nat` - "zero has type natural number"
- `true : Bool` - "true has type boolean"
- `P : type^Process` - "P is a process type"

### The Turnstile - "Derives" or "Proves"

The turnstile `|-` (Unicode: `⊢`) means "derives" or "proves":

```
G |- A : B     means "in context G, A has type B"
```

Read as: "From the assumptions in G, we can derive that A has type B."

Examples:
- `|- 0 : 00` - "with no assumptions, the nil process has type nil-type"
- `x : N |- *x : **x` - "assuming x is a name, *x has type **x"

### Contexts (G, Gamma)

A **context** is a list of typing assumptions:

```
G = x1 : X1, x2 : X2, ..., xn : Xn
```

Think of it as a symbol table or environment.

| Context | Meaning |
|---------|---------|
| `empty` or `|-` | No assumptions (empty context) |
| `x : N` | One assumption: variable x has type N |
| `x : N, y : P` | Two assumptions: x is a name, y is a process |
| `G, z : T` | Context G extended with z : T |

### Inference Rules - Visual Structure

An **inference rule** has this structure:

```
premise1    premise2    ...    premiseN
---------------------------------------- (rule-name)
             conclusion
```

The horizontal line means "therefore" or "implies":
- **Above the line**: What we assume/require (premises)
- **Below the line**: What we can conclude
- **To the right**: Rule name for reference

Example:
```
G |- f : A -> B    G |- x : A
-------------------------------- (app)
G |- f(x) : B
```

Read: "If f has function type A -> B, and x has type A, then f(x) has type B."

### Reading a Complete Rule

Let's walk through this RHO typing rule step by step:

```
G |- A : s^N    G |- B : s^P    G |- x : A    G |- Q : B
--------------------------------------------------------- (!-intro)
G |- !(x, Q) : !!(A, B, x)
```

**Breaking it down**:

| Premise | Reading |
|---------|---------|
| `G |- A : s^N` | "In context G, A is a name type at level s" |
| `G |- B : s^P` | "In context G, B is a process type at level s" |
| `G |- x : A` | "In context G, x has type A (which is a name type)" |
| `G |- Q : B` | "In context G, Q has type B (which is a process type)" |

**Conclusion**: `G |- !(x, Q) : !!(A, B, x)` - "the send process !(x, Q) has type !!(A, B, x)"

**Intuition**: To type a send, we need the types of the channel and message.

### Metavariables

Conventions used throughout:

| Convention | Typical Meaning |
|------------|-----------------|
| Capital letters: A, B, T, X | Types or metavariables over types |
| Lowercase letters: a, t, x | Terms or term variables |
| Greek letters: G, D | Contexts |
| s, s1, s2 | Type levels (type, kind, etc.) |
| P, Q | Processes (in RHO/Ambient) |
| N | Names (in RHO/Ambient) |

### Substitution Notation

```
B[x/y]    means "B with x substituted for y"
```

Example: If B = `y + 1`, then `B[5/y] = 5 + 1`

### Quick Reference Card

| Symbol | Name | Meaning |
|--------|------|---------|
| `:` | Colon | "has type" |
| `\|-` | Turnstile | "derives" or "proves" |
| `G` | Context | List of assumptions |
| `->` | Arrow | Function type |
| `x` | Times | Product type |
| `s^T` | Superscript | Classifier at level s for shape T |
| `------` | Rule line | "therefore" |
| `[a/x]` | Substitution | Replace x with a |

---

## The Typing Hypercube

### Type Levels

In typed lambda calculi, we have a hierarchy:

```
terms  :  types  :  kinds  :  sorts  ...
```

Each level classifies the level below:
- Terms have types
- Types have kinds
- Kinds have sorts (in sufficiently powerful systems)

### The s^T Notation

In MeTTaIL, we use `s^T` to denote the **classifier** of shape `T` at level `s`:

| Notation | Meaning |
|----------|---------|
| `type^P` | The type of processes (kind of process types) |
| `kind^P` | The kind of process types |
| `type^N` | The type of names (kind of name types) |
| `type^R` | The type of reductions |

When we write `s` without a superscript, it ranges over `{type, kind}`.

### Why "Hypercube"?

The name comes from the multidimensional structure:

- **Dimension 1**: Shapes (P, N, R, M, ...)
- **Dimension 2**: Levels (type, kind, ...)
- **Dimension 3**: Variance (covariant, contravariant)

This forms a hypercube of possible type/kind combinations.

### Hypercubes Across Calculi

| Calculus | Shapes | Type Levels |
|----------|--------|-------------|
| lambda | P | type^P, kind^P |
| SKI | P | type^P, kind^P |
| RHO | P, N, R | type^P, type^N, type^R, kind^P, kind^N |
| Ambient | P, N, M, R | type^P, type^N, type^M, type^R, kind^P, ... |
| MeTTa | Term, State, KB, List, MSet | type^Term, type^State, type^KB, ... |

**Example in RHO**:
- `00 : type^P` - the nil process type is a process type
- `**(@(@0)) : type^P` - the type of a dereferenced quoted quoted nil

---

## Core Inference Rules

These rules form the foundation, independent of specific calculi.

### Axiom Rule

Base types exist at each level:

```
--------------- (axiom)
|- type^T : kind^T
```

For each generating shape T.

**Examples**:
- `|- type^P : kind^P` - process types form a kind
- `|- type^N : kind^N` - name types form a kind

### Start Rule (Variable)

Variables have their declared types:

```
G |- A : s
----------------- (start)
G, x : A |- x : A
```

**Reading**: If A is a valid type (at level s), then in a context where x : A,
we can derive x : A.

**Example in lambda**:
```
|- P : type^P
------------------ (start)
x : P |- x : P
```

### Weakening Rule

Extra assumptions don't invalidate typing:

```
G |- A : B    G |- C : s
-------------------------- (weak)
G, x : C |- A : B
```

**Reading**: If A : B in G, and C is a valid type, then A : B still holds in G
extended with x : C.

### Arrow Formation

```
G |- A : s    G |- B : s
-------------------------- (arrow-form)
G |- (A -> B) : s
```

### Arrow Introduction (Abstraction)

```
G, x : A |- t : B
------------------- (arrow-intro)
G |- \x.t : (A -> B)
```

### Arrow Elimination (Application)

```
G |- f : (A -> B)    G |- a : A
--------------------------------- (arrow-elim)
G |- f(a) : B
```

These are crucial for lambda-calculus and for continuations in RHO and Ambient.

---

## Derived Rules for Term Constructors

Each function symbol generates typing rules following a pattern.

### The Pattern

For each function symbol `f : A1 x ... x An -> B`:

1. **Type-level symbol**: `ff : T(A1) x ... x T(An) -> T(B)`
2. **Type formation rule**: How to form valid types using `ff`
3. **Term introduction rule**: How to type terms built with `f`

### Lambda-Calculus Rules

**Application**:

Type formation:
```
G |- A : s^P    G |- B : s^P
------------------------------ (AppApp-form)
G |- AppApp(A, B) : s^P
```

Term introduction:
```
G |- A : s^P    G |- B : s^P    G |- f : A    G |- x : B
--------------------------------------------------------- (App-intro)
G |- App(f, x) : AppApp(A, B)
```

**Abstraction**:

Type formation:
```
G |- A : s^P    G, x : A |- B : s^P
------------------------------------- (LamLam-form)
G |- LamLam(A, \x.B) : s^P
```

Term introduction:
```
G |- A : s^P    G, x : A |- B : s^P    G, x : A |- t : B
---------------------------------------------------------- (Lam-intro)
G |- Lam(\x.t) : LamLam(A, \x.B)
```

### SKI Rules

**Combinators** (constants):

```
-------------- (S-type)
|- S : SS

-------------- (K-type)
|- K : KK

-------------- (I-type)
|- I : II
```

**Application** is same as lambda-calculus.

### RHO Rules

**Nil process**:

```
-------------- (00-form)
|- 00 : s^P

-------------- (0-intro)
|- 0 : 00
```

**Parallel composition**:

Type formation:
```
G |- A : s^P    G |- B : s^P
------------------------------ (||-form)
G |- ||(A, B) : s^P
```

Type-level equations:
```
G |- A : s^P
--------------------- (||-unit)
G |- ||(A, 00) = A

G |- A : s^P    G |- B : s^P
------------------------------ (||-comm)
G |- ||(A, B) = ||(B, A)

G |- A : s^P    G |- B : s^P    G |- C : s^P
---------------------------------------------- (||-assoc)
G |- ||(||(A, B), C) = ||(A, ||(B, C))
```

Term introduction:
```
G |- A : s^P    G |- B : s^P    G |- P : A    G |- Q : B
--------------------------------------------------------- (|-intro)
G |- |(P, Q) : ||(A, B)
```

**Send**:

Type formation (with channel parameter):
```
G |- A : s^N    G |- B : s^P    G |- x : A
-------------------------------------------- (!!-form)
G |- !!(A, B, x) : s^P
```

Term introduction:
```
G |- A : s^N    G |- B : s^P    G |- x : A    G |- Q : B
--------------------------------------------------------- (!-intro)
G |- !(x, Q) : !!(A, B, x)
```

**Receive**:

Type formation:
```
G |- A : s1^N    G |- B : s2^N    G, y : B |- C : s3^P    G |- x : A
---------------------------------------------------------------------- (??-form)
G |- ??(A, B, \y.C, x) : s3^P
```

Term introduction:
```
G |- A : s1^N    G |- B : s2^N    G, y : B |- C : s3^P
G |- x : A    G, y : B |- Q : C
---------------------------------------------------- (?-intro)
G |- ?(x, \y.Q) : ??(A, B, \y.C, x)
```

**Quote and Dereference**:

```
G |- A : s^P
----------------- (@@-form)
G |- @@(A) : s^N

G |- A : s^P    G |- P : A
--------------------------- (@-intro)
G |- @(P) : @@(A)

G |- A : s^N
----------------- (**-form)
G |- **(A) : s^P

G |- A : s^N    G |- x : A
--------------------------- (*-intro)
G |- *(x) : **(A)
```

Quote/dereference equations at type level:
```
G |- A : s^P
------------------- (@@**-inverse)
G |- **@@(A) = A

G |- A : s^N
------------------- (@@**-inverse)
G |- @@**(A) = A
```

### MeTTa Rules (Sketch)

Based on the state machine model:

**List construction**:
```
G |- A : type^Term    G |- B : type^List    G |- t : A    G |- l : B
---------------------------------------------------------------------- (cons-intro)
G |- cons(t, l) : conscons(A, B)
```

**Multiset insertion**:
```
G |- A : type^Term    G |- B : type^MSet    G |- t : A    G |- m : B
---------------------------------------------------------------------- (insert-intro)
G |- insert(t, m) : insertinsert(A, B)
```

**State construction**:
```
G |- I : type^Term    G |- K : type^KB    G |- W : type^MSet    G |- O : type^MSet
G |- i : I    G |- k : K    G |- w : W    G |- o : O
------------------------------------------------------------------------------------ (state-intro)
G |- state(i, k, w, o) : statestate(I, K, W, O)
```

---

## Reduction Typing

Reductions have types too, describing their sources and targets.

### Source and Target Typing

```
G |- A : s^R
------------------------ (srcsrc-form)
G |- srcsrc(A) : s^P

G |- A : s^R
------------------------ (tgttgt-form)
G |- tgttgt(A) : s^P

G |- A : s^R    G |- r : A
--------------------------- (src-intro)
G |- src(r) : srcsrc(A)

G |- A : s^R    G |- r : A
--------------------------- (tgt-intro)
G |- tgt(r) : tgttgt(A)
```

### Comparing Rewrite Rules Across Calculi

**Lambda-calculus beta**:
```
src(beta(K, N)) = App(Lam(K), N)
tgt(beta(K, N)) = ev(K, N)
```

At the type level:
```
srcsrc(betabeta(A, \x.B, C)) = AppApp(LamLam(A, \x.B), C)
tgttgt(betabeta(A, \x.B, C)) = B[C/x]
```

**RHO comm**:
```
src(comm(x, K, Q)) = |(?(x, K), !(x, Q))
tgt(comm(x, K, Q)) = ev(K, @(Q))
```

At the type level:
```
srcsrc(commcomm(A, B, \y.C, x)) = ||(??(A, B, \y.C, x), !!(A, **(B), x))
tgttgt(commcomm(A, B, \y.C, x)) = C    -- (y does not appear free in C)
```

### Context Rule Types

**Parallel context** (RHO, Ambient):
```
G |- A : s^R    G |- B : s^P
----------------------------- (par1par1-form)
G |- par1par1(A, B) : s^R

G |- A : s^R    G |- B : s^P
--------------------------------------------- (par1-srcsrc)
G |- srcsrc(par1par1(A, B)) = ||(srcsrc(A), B)

G |- A : s^R    G |- B : s^P
--------------------------------------------- (par1-tgttgt)
G |- tgttgt(par1par1(A, B)) = ||(tgttgt(A), B)
```

### RPO-Derived Transition System (Behavior Framework)

Following Wells & Stay's "Behavior in Higher-Order Languages", the transition
system for a lambda theory can be derived automatically via **relative pushouts (RPOs)**
rather than explicitly specified.

#### The Key Insight

Rather than manually specifying modal types for each context (like `ctxrecv_i`,
`ctxcomm_d`, etc.), the **derived transition system** computes transitions as
**idempotent pushouts (IPOs)**:

```
Γ ⊢ t⃗ →[c] d⟨⟨r⃗⟩⟩
```

where:
- `t⃗` is the source term(s)
- `c` is the **minimal context** (label) enabling the rewrite
- `d⟨⟨r⃗⟩⟩` is the target term in context d with arguments r⃗

**The label `c` represents what the environment must provide to enable the reduction.**

#### Definition: Derived Transition (Definition 17)

For a rewrite rule `p ⇝ q` and term `t`:

```
Γ ⊢ t →[c] d⟨⟨r⃗⟩⟩
```

if there exists an IPO square:

```
        p ←— Γ'
        ↓      ↓
        t ←— c
```

where `c` is the minimal context such that `c(t)` contains a redex matching `p`.

#### Connection to Modal Types

The modal types `ctxrecv_i`, `ctxcomm_d`, etc. from [05-type-lifting.md](./05-type-lifting.md)
are **derivable** from the RPO computation:

| Modal Type | Derived From |
|------------|--------------|
| `ctxrecv_i(...)` | IPO for comm rule with receive context |
| `ctxsend_i(...)` | IPO for comm rule with send context |
| `ctxcomm_i(...)` | IPO for comm rule with sent-process context |
| `ctxposs_i(T)` | General possibility via reflexive-transitive closure |

**Implementation note**: The current explicit modal type generation in MeTTaIL is a
concrete implementation strategy for the abstract RPO derivation. Both approaches
produce equivalent typing information.

#### Why IPOs Matter

IPOs (idempotent pushouts) ensure that:
1. Labels are **minimal** - no unnecessary context information
2. Labels are **canonical** - unique up to isomorphism
3. **Bisimilarity is a congruence** - behavioral equivalence is preserved by contexts

This is proven in Theorem 20 and Theorem 22 of the Behavior paper.

---

## Transparency and Congruence

For behavioral equivalence (bisimilarity) to be preserved under all contexts,
we need conditions on the calculus structure.

### Reactive vs Transparent Contexts

**Reactive context**: A context containing a redex pattern. For example:
- `out(n, −) | in(n, λx.q)` in RHO is reactive (contains comm redex pattern)
- `App(Lam(K), −)` in lambda-calculus is reactive (contains beta redex pattern)

**Transparent context**: A non-reactive context `c` where there exists a unique
complementary context `c̄` such that for any term `t`:

```
c(t) →[c̄] d(t)
```

### Theorem: Transparency implies Congruence (Theorem 15)

If a calculus is **transparent** (all non-reactive contexts are transparent),
then weak bisimilarity is a congruence.

**The RHO calculus and lambda-calculus are both transparent.**

This means:
- If `p ≈ q` (p and q are behaviorally equivalent)
- Then `C[p] ≈ C[q]` for any context C

### IPO Uniformity (Definition 21)

A stronger condition: context `g` is **IPO uniform** if transitions factor
predictably through sublists of the context.

**Theorem 22**: If every context is either reactive or IPO uniform, weak
bisimilarity is a congruence.

### Implications for Type Checking

These conditions ensure that:
1. Type-level behavioral equivalences are preserved by all term constructors
2. Typed terms with equivalent types behave equivalently in all contexts
3. The type system soundly approximates behavioral equivalence

### Quick Reference: Congruence Conditions

| Condition | Definition | Ensures |
|-----------|------------|---------|
| Transparency | Non-reactive contexts have unique complementary labels | Weak bisimilarity congruence |
| IPO Uniformity | Transitions factor through context sublists | Strong congruence property |
| Reactive | Context contains redex pattern | Context participates in reduction |

---

## Complete Derivation Examples

### Lambda-Calculus: Typing the Identity Function

**Goal**: Derive the type of `Lam(\x.x)` (the identity function).

**Derivation tree**:

```
                                |- P : type^P
                              ---------------- (start)
    |- P : type^P               x : P |- x : P
----------------------------------------------- (Lam-intro)
|- Lam(\x.x) : LamLam(P, \x.P)
```

**Result**: `Lam(\x.x) : LamLam(P, \x.P)`

This is the identity function with type "for any type P, takes P and returns P".

### RHO: Typing a Simple Communication

**Goal**: Type the process `|(?(x, \y.*y), !(x, 0))` - a receive and send in parallel.

**Setup**:
- Assume `x : N` (x is a name)
- The receive waits for a name and dereferences it
- The send sends the nil process

**Derivation** (sketch):

1. Type the send:
   ```
   x : N |- !(x, 0) : !!(N, 00, x)
   ```

2. Type the receive:
   ```
   x : N |- ?(x, \y.*y) : ??(N, @@(00), \y.**@@(00), x)
   ```

   The continuation `\y.*y` has type `\y.**y`, which at type `@@(00)` gives `**(@@(00))`.

3. Combine with parallel:
   ```
   x : N |- |(?(x, \y.*y), !(x, 0)) : ||(??(N, @@(00), \y.**(@@(00)), x), !!(N, 00, x))
   ```

### RHO: Typing the Comm Reduction

The reduction `comm(x, \y.*y, 0)` has:
- Source: `|(?(x, \y.*y), !(x, 0))`
- Target: `*(@0)` = `*(@0)` which reduces to `0`

**Type of the reduction**:
```
commcomm(N, @@(00), \y.**(@@(00)), x)
```

**Source type** (via srcsrc):
```
||(??(N, @@(00), \y.**(@@(00)), x), !!(N, **(@@(00)), x))
```

**Target type** (via tgttgt):
```
**(@@(00)) = 00    (by the @@**-inverse equation)
```

### Building a Derivation Tree (ASCII Art)

Here's a complete derivation for typing `!(x, 0)` in context `x : N`:

```
                                                      --------------- (00-form)
                                                      |- 00 : type^P
                                                      --------------- (0-intro)
  --------------- (axiom)   --------------- (weak)    |- 0 : 00
  |- N : type^N             x : N |- N : type^N       --------------- (weak)
  --------------- (weak)    --------------- (start)   x : N |- 0 : 00
  x : N |- N : type^N       x : N |- x : N
  ------------------------------------------------------------ (!-intro)
  x : N |- !(x, 0) : !!(N, 00, x)
```

---

## Implementation Patterns

### Type Checking Algorithm

The standard approach is **bidirectional type checking**:

```rust
enum Mode {
    Check { expected: Type },  // check term against known type
    Infer,                     // infer type from term
}

fn typecheck(ctx: &Context, term: &Term, mode: Mode) -> Result<Type, TypeError> {
    match (term, mode) {
        // Variable: look up in context
        (Term::Var(name), Mode::Infer) => {
            ctx.lookup(name).ok_or(TypeError::Unbound(name.clone()))
        }

        // Application: infer function type, check argument
        (Term::App(f, arg), Mode::Infer) => {
            let fn_type = typecheck(ctx, f, Mode::Infer)?;
            match fn_type {
                Type::Arrow(a, b) => {
                    typecheck(ctx, arg, Mode::Check { expected: *a })?;
                    Ok(*b)
                }
                _ => Err(TypeError::NotAFunction(fn_type))
            }
        }

        // Abstraction: extend context, check body
        (Term::Lam(var, body), Mode::Check { expected: Type::Arrow(a, b) }) => {
            let extended = ctx.extend(var.clone(), *a.clone());
            typecheck(&extended, body, Mode::Check { expected: *b })?;
            Ok(Type::Arrow(a, b))
        }

        // Send: infer channel and message types
        (Term::Send(chan, msg), Mode::Infer) => {
            let chan_type = typecheck(ctx, chan, Mode::Infer)?;
            let msg_type = typecheck(ctx, msg, Mode::Infer)?;
            // chan_type should be a name type
            Ok(Type::SendType(Box::new(chan_type), Box::new(msg_type), chan.clone()))
        }

        // Check mode: infer and compare
        (term, Mode::Check { expected }) => {
            let inferred = typecheck(ctx, term, Mode::Infer)?;
            if types_equal(&inferred, &expected) {
                Ok(inferred)
            } else {
                Err(TypeError::Mismatch { expected, actual: inferred })
            }
        }
    }
}
```

### Context Implementation

```rust
#[derive(Clone)]
struct Context {
    bindings: Vec<(String, Type)>,
}

impl Context {
    fn empty() -> Self {
        Context { bindings: vec![] }
    }

    fn extend(&self, name: String, ty: Type) -> Self {
        let mut new = self.clone();
        new.bindings.push((name, ty));
        new
    }

    fn lookup(&self, name: &str) -> Option<Type> {
        self.bindings.iter().rev()
            .find(|(n, _)| n == name)
            .map(|(_, ty)| ty.clone())
    }
}
```

### Type Equality with Equations

Handle type-level equations (like multiset commutativity):

```rust
fn types_equal(t1: &Type, t2: &Type) -> bool {
    // Normalize both types first
    let n1 = normalize(t1);
    let n2 = normalize(t2);
    syntactic_equal(&n1, &n2)
}

fn normalize(ty: &Type) -> Type {
    match ty {
        // ||(A, 00) = A
        Type::Par(a, b) if **b == Type::Nil => normalize(a),

        // ||(00, A) = A
        Type::Par(a, b) if **a == Type::Nil => normalize(b),

        // ||(A, B) -> canonical order (for commutativity)
        Type::Par(a, b) => {
            let na = normalize(a);
            let nb = normalize(b);
            if type_ord(&na) > type_ord(&nb) {
                Type::Par(Box::new(nb), Box::new(na))
            } else {
                Type::Par(Box::new(na), Box::new(nb))
            }
        }

        // @@(**(A)) = A
        Type::Quote(inner) => match &**inner {
            Type::Deref(a) => normalize(a),
            _ => Type::Quote(Box::new(normalize(inner)))
        }

        // **(@@(A)) = A
        Type::Deref(inner) => match &**inner {
            Type::Quote(a) => normalize(a),
            _ => Type::Deref(Box::new(normalize(inner)))
        }

        // Recursively normalize other types
        _ => ty.clone()
    }
}
```

### Generating Rules from GSLT

```rust
fn generate_rules(gslt: &GSLT) -> Vec<InferenceRule> {
    let mut rules = vec![];

    // Core rules (axiom, start, weakening, arrow)
    rules.extend(core_rules());

    // Formation rules for each type constructor
    for morph in &gslt.type_morphisms {
        rules.push(generate_formation_rule(morph));
    }

    // Introduction rules for each term constructor
    for morph in &gslt.term_morphisms {
        rules.push(generate_intro_rule(morph, &gslt.type_morphisms));
    }

    // Equation rules for structural equations
    for eq in &gslt.equations {
        rules.push(generate_equation_rule(eq));
    }

    // Source/target rules for reductions
    for red in &gslt.reductions {
        rules.extend(generate_reduction_rules(red));
    }

    rules
}

fn generate_intro_rule(term_morph: &Morphism, type_morphs: &[Morphism]) -> InferenceRule {
    // Find corresponding type-level morphism
    let type_morph = type_morphs.iter()
        .find(|m| m.name == format!("{}{}", term_morph.name, term_morph.name))
        .expect("type-lifted morphism must exist");

    // Generate premises: type well-formedness + term typing
    let mut premises = vec![];

    // Type well-formedness premises
    for (i, ty) in type_morph.inputs.iter().enumerate() {
        premises.push(Judgment {
            context: Context::meta("G"),
            term: Term::meta(format!("T{}", i)),
            ty: Type::sort_at_level(ty.sort(), "s"),
        });
    }

    // Term typing premises
    for (i, (tm_input, ty_input)) in term_morph.inputs.iter().zip(type_morph.inputs.iter()).enumerate() {
        premises.push(Judgment {
            context: Context::meta("G"),
            term: Term::meta(format!("t{}", i)),
            ty: Type::meta(format!("T{}", i)),
        });
    }

    // Conclusion
    let conclusion = Judgment {
        context: Context::meta("G"),
        term: Term::app(term_morph.name.clone(), (0..term_morph.inputs.len()).map(|i| Term::meta(format!("t{}", i))).collect()),
        ty: Type::app(type_morph.name.clone(), (0..type_morph.inputs.len()).map(|i| Type::meta(format!("T{}", i))).collect()),
    };

    InferenceRule {
        name: format!("{}-intro", term_morph.name),
        premises,
        conclusion,
    }
}
```

### Performance Optimizations

1. **Memoize normalization**: Type normalization can be expensive; cache results
2. **Use hash-consing**: Share identical type structures to reduce memory
3. **Lazy unfolding**: Don't expand type synonyms until needed
4. **Incremental checking**: When context extends, don't re-check unchanged terms

```rust
struct TypeChecker {
    normalization_cache: HashMap<Type, Type>,
    checked_terms: HashMap<(TermId, Type), bool>,
}

impl TypeChecker {
    fn normalize_cached(&mut self, ty: &Type) -> Type {
        if let Some(cached) = self.normalization_cache.get(ty) {
            return cached.clone();
        }
        let result = normalize(ty);
        self.normalization_cache.insert(ty.clone(), result.clone());
        result
    }
}
```

---

## Summary

**Core rules**:
- Axiom: Base types exist
- Start: Variables have declared types
- Weakening: Extra assumptions don't hurt
- Arrow: Function types with intro/elim

**Pattern for term constructors**:
- Type-level symbol `ff` for each `f`
- Type formation rule
- Term introduction rule

**Reduction typing**:
- `src`, `tgt` at term level
- `srcsrc`, `tgttgt` at type level
- Context rules lift as expected

---

## Quick Reference: Reading Inference Rules

| Symbol | Meaning |
|--------|---------|
| `\|-` | "proves" or "derives" |
| `G` | Context (list of assumptions) |
| `:` | "has type" |
| `s^T` | Level s classifier for shape T |
| `------` | "therefore" (conclusion below) |
| `=` | Definitional equality |
| `[a/x]` | Substitution of a for x |

---

## Next Steps

- [02-native-type-theory-oslf.md](./02-native-type-theory-oslf.md): Full OSLF for behavioral types
- [04-rho-calculus.md](./04-rho-calculus.md): RHO calculus reflection and bisimulation
- [../implementation/04-implementation-roadmap.md](../implementation/04-implementation-roadmap.md): Implementation roadmap

---

## References

- Wells, P. & Stay, M. "Behavior in Higher-Order Languages." 2024.
  (Primary reference for RPO framework, transparency conditions, and bisimilarity congruence)
- Pierce, B. C. "Types and Programming Languages." MIT Press, 2002.
- Williams, P. & Stay, M. "Native Type Theory." EPTCS 372, pp. 116-132, 2022.
- Stay, M. & Meredith, L. G. "Representing operational semantics with enriched
  Lawvere theories." arXiv:1704.03080, 2017.
- Milner, R. "Deriving bisimulation congruences for reactive systems." CONCUR 2003.
  (Foundation for RPO-based transition system derivation)
- See [bibliography.md](../reference/bibliography.md) for complete references.
