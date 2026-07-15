[← Documentation Index](../../README.md)

# MeTTa Operational Semantics

This document presents the operational semantics of MeTTa as a **symmetric reflective
higher-order concurrent calculus with backchaining**. It synthesizes content from the
Meta-MeTTa paper (Meredith et al., 2023) and the rhocube papers to provide a rigorous
foundation for semantic type checking.

**Target audience**: Compiler engineers and PL implementers

---

## Table of Contents

1. [For Implementers: Key Takeaways](#for-implementers-key-takeaways)
2. [Introduction](#introduction)
3. [Formal Grammar](#formal-grammar)
4. [Structural Congruence](#structural-congruence)
5. [The State Machine Model](#the-state-machine-model)
6. [Core Reduction Rules](#core-reduction-rules)
7. [Transactional Semantics](#transactional-semantics)
8. [Unification and Pattern Matching](#unification-and-pattern-matching)
9. [Implementation Bridge](#implementation-bridge)
10. [Evaluation Semantics](#evaluation-semantics)
11. [Connection to Type Checking](#connection-to-type-checking)
12. [Comparison with Other Calculi](#comparison-with-other-calculi)
13. [Summary](#summary)

---

## For Implementers: Key Takeaways

Before diving into details, here's what you need to know:

### The 80/20 Rule

**Essential concepts** (handle 80% of use cases):
1. The **COMM rule**: Two for-comprehensions at the same channel unify and reduce
2. **Spaces as tags**: Atoms are tagged with spaces, not placed in containers
3. The four-component state: $`\langle i, k, w, o\rangle`$ (input, knowledge, workspace, output)
4. **Structural congruence**: Parallel composition is commutative and associative

**Advanced concepts** (for complete implementation):
- Transactional semantics of COMM
- Procedural reflection (`x?P`)
- Fork-join concurrency patterns
- RSpace compilation strategy

### Data Structures You'll Need

```rust
// Process syntax
enum Process {
    Zero,                           // 0 (nil process)
    Ground(GroundValue),            // Literals
    For(Term, Name, Box<Process>),  // for(t <- x)P
    Peek(Name, Box<Process>),       // x?P (reflection)
    Deref(Name),                    // *x
    Par(Box<Process>, Box<Process>),// P | Q
}

// Name syntax - names are quoted processes
enum Name {
    Quote(Box<Process>),            // @P
}

// Term syntax
enum Term {
    Atom(Atom),
    Tuple(Vec<Term>),
}

enum Atom {
    Var(Name),
    Proc(Process),
}

// Ground values
enum GroundValue {
    Bool(bool),
    String(String),
    Int(i64),
    Collection(Collection),
}

enum Collection {
    List(Vec<Term>),
    Tuple(Vec<Term>),
    Set(HashSet<Term>),
    Map(HashMap<Term, Term>),
}
```

### Key Operations to Implement

| Operation | Complexity | Description |
|-----------|------------|-------------|
| `unify(t, u)` | $`\mathcal{O}(n)`$ typical | Pattern matching with occurs check |
| $`\text{substitute}(P, \sigma )`$ | $`\mathcal{O}(P \text{size})`$ | Apply substitution to process |
| `structural_eq(P, Q)` | $`\mathcal{O}(n)`$ | Check structural congruence |
| `step(state)` | $`\mathcal{O}(\text{varies})`$ | One reduction step via COMM |
| `compile_to_rspace(P)` | $`\mathcal{O}(P \text{size})`$ | Transform to RSpace representation |

---

## Introduction

### What This Calculus Provides

The MeTTa calculus is a **symmetric reflective higher-order concurrent calculus** that
provides:

1. **Symmetric communication**: Both sides of a rendezvous read and write simultaneously
2. **Reflection**: Processes can inspect their own execution state
3. **Higher-order**: Terms can contain processes, processes can contain terms
4. **Concurrency**: Parallel composition with non-deterministic interleaving
5. **Backchaining**: Pattern matching with unification enables logic programming

### Problems Solved by This Design

#### Secret Leaking in Naive Implementations

Early MeTTa implementations stored atoms in container structures (like AtomSpace). This
approach leaks information because:
- Container membership is observable
- Traversal order reveals structure
- No principled way to scope visibility

**Solution**: Spaces as tags, not containers. Atoms are tagged with the spaces they
occupy, using the same construct for adding, removing, and querying atoms.

#### The "All Paths" Problem

Naive implementations explore all possible computation paths, leading to:
- Exponential blowup in branching factor
- No way to express "pick one" nondeterminism
- Resource exhaustion on open-ended queries

**Solution**: The COMM rule picks one matching pair nondeterministically, and
transactional semantics ensure atomic state transitions.

---

## Formal Grammar

The grammar uses these conventions:
- `[e]` denotes a space-delimited finite sequence of `e`'s
- `[e]_seq` denotes a comma-delimited finite sequence of `e`'s

### Processes

```
P, Q  ::=  0                      ; Nil process
       |   G                      ; Ground value
       |   for(t <- x)P           ; For-comprehension (listen on channel)
       |   x?P                    ; Reflection (peek at future)
       |   *x                     ; Dereference (drop quote)
       |   P | Q                  ; Parallel composition
```

### Names (Channels)

Names are quoted processes - this is the reflection mechanism:

```
x, y  ::=  @P                     ; Quote a process to get a name
```

### Terms

```
t, u    ::=  atom                 ; Atomic term
         |   (t₁ t₂ ... tₙ)       ; Tuple of terms

atom    ::=  x                    ; Variable (a name)
         |   P                    ; Process as atom
```

### Ground Values

```
G  ::=  BoolLiteral               ; true, false
    |   StringLiteral             ; "hello"
    |   IntLiteral                ; 42
    |   C                         ; Collection

C  ::=  [t₁, t₂, ..., tₙ]         ; List
    |   (t₁, t₂, ..., tₙ)         ; Tuple
    |   Set(t₁, t₂, ..., tₙ)      ; Set
    |   {k₁:v₁, k₂:v₂, ...}       ; Map
```

### Intuitive Mapping to MeTTa Language

| MeTTa Language | MeTTa Calculus |
|----------------|----------------|
| `(addAtom space term)` | `for(term <- space)0` |
| `(remAtom space term)` | `for(term <- space)0` |
| `(? space term)` | `for(term <- space)0` |

**Key insight**: `folders = tagging`. Rather than building container data structures
for spaces (like AtomSpace), we tag atoms with the spaces they occupy. This allows the
same construct to serve for adding, removing, and querying atoms.

A rewrite rule is a continuation `t ↦ { P }`. When tagged with a space `x`, written
`for(t <- x)P`, it may be thought of as added to that space.

Under this view, a **space** is equated with all the atoms and rules tagged as being in
that space:

```
space(x) = Π_i for(t_i <- x)P_i
```

This gives us two interrelated algebras:
1. **Spaces as tags**: Isomorphic to process states, enables programmatic filtering
2. **Spaces as collections**: Enables reasoning about, filtering, and traversing spaces

These algebras interrelate, meaning spaces can be nested or mutually recursively defined.

---

## Structural Congruence

Structural congruence $`(\equiv )`$ defines when two syntactically different processes are
semantically equivalent.

### Parallel Composition Equations

Parallel composition forms a **commutative monoid**:

```
P | 0 ≡ P                     ; Identity
P | Q ≡ Q | P                 ; Commutativity
P | (Q | R) ≡ (P | Q) | R     ; Associativity
```

### Alpha-Equivalence

Bound variables can be renamed consistently:

```
         occurs(t, y)
─────────────────────────────────────────────────────
for(t <- x)P ≡ for(t{z/y} <- x)(P{z/y})  if z ∉ FN(P)
```

Where `FN(P)` is the set of free names in `P`.

### Why Congruence Matters for Implementation

Structural congruence means:
1. **Storage optimization**: `P | Q` and `Q | P` can share representations
2. **Pattern matching**: Must match modulo congruence, not just syntactically
3. **Normalization**: Can choose canonical forms for efficient comparison

---

## The State Machine Model

MeTTa computation is also modeled as a state machine with four components:

```
State = ⟨i, k, w, o⟩
```

| Component | Name | Description |
|-----------|------|-------------|
| `i` | Input | The input register (term to evaluate) |
| `k` | Knowledge | The knowledge base (database of facts and rules) |
| `w` | Workspace | The working memory (intermediate computation) |
| `o` | Output | The output register (results of computation) |

### State Transitions

Computation proceeds by applying rewrite rules to transform one state into another:

```
⟨i₀, k₀, w₀, o₀⟩ ⟶ ⟨i₁, k₁, w₁, o₁⟩
```

The knowledge base `k` typically remains constant during simple queries but can be
modified by `AddAtom` and `RemAtom` operations.

---

## Core Reduction Rules

The calculus has four core reduction rules.

### COMM Rule (Communication)

The fundamental interaction rule - two for-comprehensions at the same channel unify:

```
           σ = unify(t, u)
───────────────────────────────────────────────────
for(t <- x)P | for(u <- x)Q  ⟶  P·σ̇ | Q·σ̇
```

Where `σ̇` denotes the substitution that replaces variable-to-process bindings with
variable-to-name bindings: `{P / x}̇ = {@P / x}`.

**Example**:
```
for(($x, $y) <- chan)P | for((1, 2) <- chan)Q
⟶  P{@1/$x, @2/$y} | Q
```

### PAR Rule (Parallel Execution)

Reduction can occur within parallel composition:

```
    P ⟶ P'
──────────────────
P | Q  ⟶  P' | Q
```

### EQUIV Rule (Modulo Congruence)

Reduction respects structural congruence:

```
P ≡ P'    P' ⟶ Q'    Q' ≡ Q
────────────────────────────────
         P ⟶ Q
```

### REFL Rule (Reflection)

Reflection allows a process to inspect its own future:

```
      P ⟶ P'
────────────────────────────────
x?P  ⟶  for((P') <- x)0
```

The `x?P` construct runs `P` for one step, then makes the resulting state available
at channel `x`.

### Knowledge Base Operations

These integrate with the state machine model:

**AddAtom** - extends the knowledge base:
```
⟨(add-atom t), k, w, o⟩ ⟶ ⟨ε, k ∪ {t}, w, o⟩
```

**RemAtom** - removes from the knowledge base:
```
⟨(rem-atom t), k, w, o⟩ ⟶ ⟨ε, k \ {t}, w, o⟩
```

**Query** - matches pattern against knowledge base:
```
        match(p, k) = {σ₁, σ₂, ..., σₙ}
─────────────────────────────────────────────
⟨p, k, w, o⟩ ⟶ ⟨ε, k, w ∪ {σ₁(p), ...}, o⟩
```

---

## Transactional Semantics

### COMM as Transaction

One of the most important aspects of the COMM rule is that it is **implicitly
transactional**.

In the symmetric case, both threads are reading and writing simultaneously:
- Variables in `t` substituted by terms from `u` are being **read from** `u` and
  **written to** continuation `P·σ̇`
- Variables in `u` substituted by terms from `t` are being **read from** `t` and
  **written to** continuation `Q·σ̇`

The substitution `σ̇` is the **witness of the transaction** - it records exactly what
was exchanged.

### Namespace Coordination

A given RSpace represents a collection of MeTTa spaces that share transactional
semantics. The **namespace** served by an RSpace - all channels stored in it -
participates in coordinated transactions.

This is similar to how all tables in a SQL server participate in coordinated
transactions, but the RSpace architecture **naturally composes**, allowing for:
- Hierarchy of transactional coordination
- Tree of namespaces served by a network of RSpaces
- Decentralized transactional coordination

### Implications for Implementation

1. **Atomicity**: A COMM event either completes fully or not at all
2. **Isolation**: Concurrent COMM events on different channels don't interfere
3. **Durability**: The substitution witness can be persisted for recovery
4. **Composability**: Transactions can be nested via namespace hierarchy

---

## Unification and Pattern Matching

### Syntactic Unification

Pattern matching uses **syntactic unification** with the occurs check:

```
unify(t₁, t₂) =
  | t₁ = t₂           → {}           ; Identical terms
  | t₁ = $x           → {$x ↦ t₂}    ; Variable binding (if occurs check passes)
  | t₂ = $x           → {$x ↦ t₁}    ; Variable binding (symmetric)
  | (f a...) = (g b...) → unify(f,g) ∪ unify(a..., b...)  ; Recursive
  | otherwise         → ⊥            ; Failure
```

### Substitution Application

When applying substitution $`\sigma`$ to process `P`, written $`P\cdot \sigma`$:
- Replace each free variable `x` with $`\sigma (x)`$
- Avoid capture by renaming bound variables as needed

The dotted substitution `σ̇` converts process bindings to name bindings:
```
σ̇ = {x ↦ @P | (x ↦ P) ∈ σ}
```

---

## Implementation Bridge

### Compilation to RSpace

The key innovation is using a variant of Linda tuple space where **input is not
blocking** - instead, we store continuations at keys.

```
┌─────────────────────────────────────────────────────┐
│  Process                    RSpace                  │
├─────────────────────────────────────────────────────┤
│  0                    ⟹    ∅ (empty RSpace)        │
│  for(t <- x)P         ⟹    {hash(x): [t ↦ P]}     │
│  P | Q                ⟹    ⟦P⟧ ⊕ ⟦Q⟧              │
└─────────────────────────────────────────────────────┘
```

Where:
- The key is the **hash of the channel**
- The value is a **pattern-matching lambda** from target to body
- $`\oplus`$ is RSpace parallel composition (defined below)

### RSpace Parallel Composition

Combining two RSpaces follows these rules:

**Non-overlapping keys**: Simply union the key-value pairs.

**Overlapping keys, non-unifying patterns**: Combine into a single key-value pair
where the value is a multiset of continuations.

**Overlapping keys, unifying patterns**: This triggers the COMM rule! The MORK
algorithm provides efficient parallel unification checking.

```rust
fn combine_rspace(r1: RSpace, r2: RSpace) -> RSpace {
    let mut result = HashMap::new();

    for (key, conts1) in r1 {
        if let Some(conts2) = r2.get(&key) {
            // Check for unifying patterns
            if let Some(sigma) = find_unifier(&conts1, &conts2) {
                // COMM! Reduce and recurse
                return apply_comm(conts1, conts2, sigma);
            } else {
                // No unification - merge multisets
                result.insert(key, conts1.union(conts2));
            }
        } else {
            result.insert(key, conts1);
        }
    }
    // Add remaining keys from r2
    for (key, conts) in r2 {
        if !result.contains_key(&key) {
            result.insert(key, conts);
        }
    }
    result
}
```

### Procedural Reflection Implementation

For `x?P`, the implementation:
1. Creates a **copy** of the RSpace where `P` is evaluated
2. Allows one transactional step (one COMM event) in the future of `P`
3. Makes that resulting state available at `x` in the original RSpace

There is no way to avoid the cost of copying. An alternative using namespace shifting:
```
u * P = P{ u * x / x | x ∈ FN(P) }
@P * @Q = @(P | Q)
```
But this is actually more costly than deep copy because key-value pairs still need
to be copied and then shifted.

### MORK Integration

MORK provides efficient trie-based path manipulation for the unification check in
RSpace composition. When combining RSpaces with overlapping keys:
1. MORK indexes patterns by structure
2. Parallel unification attempts are batched
3. Successful unifications trigger COMM reductions

See [04-mork-pathmap-integration.md](../metta-ecosystem/04-mork-pathmap-integration.md)
for details.

---

## Evaluation Semantics

### Evaluation Order

MeTTa uses a **non-deterministic** evaluation order for parallel composition, which is
essential for its concurrent semantics. The PAR and EQUIV rules allow reduction in any
order consistent with structural congruence.

### Useful Features

#### Replication

In the core calculus, when two terms rendezvous, they are **consumed**. Replication
leaves one in place:

```
P, Q  ::=  ... | !P               ; Replication
```

When `!for(t <- x)P` rendezvous with `for(u <- x)Q`:
```
!for(t <- x)P | for(u <- x)Q  ⟶  !for(t <- x)P | P·σ̇ | Q·σ̇
```

#### Freshness

Private channels guarantee isolated computation:

```
P, Q  ::=  ... | new x in { P }   ; Fresh name
```

The state `new x in { for(t <- x)P | for(u <- x)Q }` guarantees the rendezvous
happens in a private space.

#### Fork-Join Concurrency

An extended for-comprehension supports fork-join patterns common in human decision
processes:

```
for(
  y₁₁ <- x₁₁ & ... & yₘ₁ <- xₘ₁ ;  // Row 1: received in any order
  ...                              // All of row i before row i+1
  y₁ₙ <- x₁ₙ & ... & yₘₙ <- xₘₙ    // Row n
){ P }
```

**Example** - academic paper review:
```
for(
  true <- reviewer₁ & true <- reviewer₂ & true <- reviewer₃
){
  // Acceptance notification
  P
}
```

### Builtin Operations

MeTTa includes builtin operations for:
- Arithmetic: `+`, `-`, `*`, `/`
- Comparison: `<`, `>`, `=`, $`\le`$, $`\ge`$
- Logic: `and`, `or`, `not`
- Type checking: `get-type`, `has-type`

---

## Connection to Type Checking

### Spatial-Behavioral Types

The type system is **generated** from the operational semantics using the OSLF
algorithm. Type syntax mirrors process syntax:

```
T, U  ::=  0̸                      ; Nil type
       |   GT                     ; Ground type
       |   ⟨(TT → N)⟩T            ; For-comprehension type
       |   ⟨x?⟩T                  ; Reflection type
       |   *N                     ; Dereference type
       |   T | U                  ; Parallel type

N     ::=  @T                     ; Name type
```

### Type Inference Rules

The type system includes rules like:

**For-comprehension**:
```
t : TT, Γ ⊢ P : T    Δ ⊢ x : V
─────────────────────────────────────────
Γ, Δ ⊢ for(t <- x)P : ⟨(TT → V)⟩T
```

**Parallel composition**:
```
Γ ⊢ P : T    Δ ⊢ Q : U
──────────────────────────
Γ, Δ ⊢ P | Q : T | U
```

### OSLF Connection

The Meta-MeTTa paper explicitly recommends OSLF for deriving "a type system for MeTTa
that includes spatial and behavioral types."

MeTTa's operational semantics can be formalized as a **λ-theory with equality**:

| MeTTa Concept | λ-Theory Formalization |
|---------------|------------------------|
| Term sorts | Sorts in the theory signature |
| Constructors | Operations in the signature |
| Parallel equations | Equational axioms |
| Rewrite rules | Internal graph edges (transitions) |
| Pattern variables | Binding structure |

See [02-native-type-theory-oslf.md](./02-native-type-theory-oslf.md) for the full
OSLF construction.

---

## Comparison with Other Calculi

Understanding how MeTTa relates to other calculi clarifies implementation choices.

### State Models Compared

| Calculus | State Model | Key Feature |
|----------|-------------|-------------|
| **MeTTa** | Process soup + $`\langle i, k, w, o\rangle`$ | Spaces as tags, reflection |
| **Lambda** | Term only | Pure substitution |
| **SKI** | Term only | Combinator-based |
| **RHO** | Process soup (multiset) | Reflection via quote/drop |
| **Ambient** | Tree of nested ambients | Hierarchical mobility |

### Binding Mechanisms

| Calculus | Binding Mechanism | Variable Scope |
|----------|-------------------|----------------|
| **MeTTa** | For-comprehension (`for(t <- x)`) | Continuation scope |
| **Lambda** | Lambda abstraction ($`\lambda x.`$) | Lexical |
| **SKI** | None (combinator encoding) | N/A |
| **RHO** | Input prefix (`x(y).P`) | Continuation scope |
| **Ambient** | Restriction ($`\nu n.P`$) | Lexical |

### Interaction Patterns

| Calculus | Primary Interaction | Rule Name |
|----------|---------------------|-----------|
| **MeTTa** | Symmetric rendezvous | COMM |
| **Lambda** | Application | Beta |
| **SKI** | Application | S, K, I |
| **RHO** | Communication | Comm |
| **Ambient** | Capability exercise | In, Out, Open |

### Reduction Strategy

| Calculus | Evaluation Order | Determinism |
|----------|------------------|-------------|
| **MeTTa** | Non-deterministic | Non-deterministic |
| **Lambda** | Various (CBV, CBN, lazy) | Deterministic (given strategy) |
| **SKI** | Leftmost-outermost typical | Deterministic |
| **RHO** | Non-deterministic | Non-deterministic |
| **Ambient** | Non-deterministic | Non-deterministic |

### Implications for Implementation

**From MeTTa's perspective**:
- The knowledge base is like a persistent multiset (similar to RHO's process soup)
- Pattern matching via COMM is more powerful than beta reduction
- Non-determinism requires careful handling (breadth-first, probabilistic, etc.)

**Shared with RHO**:
- Both use multisets with commutativity/associativity equations
- Both support reflection (MeTTa: `x?P`/`@P`/`*x`, RHO: quote/drop)
- Both have non-deterministic semantics
- Both have transactional COMM rules

**Differs from Lambda/SKI**:
- MeTTa has mutable state (knowledge base modification)
- No fixed evaluation order
- Rich pattern matching vs. simple beta/combinator reduction

---

## Summary

MeTTa's operational semantics provides:

1. **Symmetric concurrent calculus** with for-comprehensions and parallel composition
2. **Spaces as tags** rather than containers, solving secret leaking
3. **COMM rule** for synchronized communication with transactional semantics
4. **Reflection** via `x?P` for introspection of computation
5. **Structural congruence** making parallel composition a commutative monoid
6. **RSpace compilation** strategy for efficient implementation
7. **OSLF-derived types** for spatial-behavioral type checking

The next documents explore how to derive semantic types from this operational
foundation:
- [02-native-type-theory-oslf.md](./02-native-type-theory-oslf.md): Full type theory derivation
- [03-gph-enriched-lawvere.md](./03-gph-enriched-lawvere.md): Simpler graph-based approach
- [05-type-lifting.md](./05-type-lifting.md): Type lifting transformation
- [06-inference-rules.md](./06-inference-rules.md): Inference rules guide

---

## References

- Meredith, L. G., et al. "Meta-MeTTa: an operational semantics for MeTTa."
  arXiv:2305.17218, 2023.
- Meredith, L. G., and Stay, M. "rhocube: A symmetric reflective higher-order
  concurrent calculus with backchaining." F1R3FLY.io, 2024.
- Meredith, L. G., and Stay, M. "Representing operational semantics with enriched
  Lawvere theories." arXiv:1704.03080, 2017.
- Williams, M., and Stay, M. "Native Type Theory." ACT 2021.
- See [bibliography.md](../reference/bibliography.md) for complete references.
