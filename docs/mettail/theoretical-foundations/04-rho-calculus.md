[← Documentation Index](../../README.md)

# The RHO Calculus

This document presents the rho-calculus (reflective higher-order calculus), the
theoretical foundation of Rholang. Understanding RHO calculus is essential for
integrating MeTTaIL with Rholang's execution model.

**Target audience**: Compiler engineers integrating with Rholang

---

## Table of Contents

1. [For Implementers: RHO in Practice](#for-implementers-rho-in-practice)
2. [Overview](#overview)
3. [Syntax](#syntax)
4. [Structural Congruence](#structural-congruence)
5. [Reduction Semantics](#reduction-semantics)
6. [The Reflection Mechanism](#the-reflection-mechanism)
7. [Bisimulation](#bisimulation)
8. [Comparison with Lambda and SKI](#comparison-with-lambda-and-ski)
9. [Connection to Type Checking](#connection-to-type-checking)

---

## For Implementers: RHO in Practice

### What Makes RHO Different

RHO is unique among process calculi because:

1. **Names come from processes** - No primitive name generation (nu)
2. **Reflection via quote/drop** - Code as data without meta-levels
3. **Simpler theory** - Binding eliminated through reflection
4. **Direct execution model** - Maps cleanly to Rholang/RSpace

### Quick Reference: RHO vs Pi-Calculus

| Feature | Pi-Calculus | RHO Calculus |
|---------|-------------|--------------|
| Name generation | `nu x.P` (primitive) | Derived via quote |
| Communication | `x(y).P \| x<z>` | `x(y).P \| x<\|Q\|>` |
| Higher-order | Needs extensions | Built-in via reflection |
| Binding | Traditional | Eliminated via @/* |

### Key Implementation Points

```rust
// RHO term representation
enum Process {
    Nil,                              // 0
    Par(Box<Process>, Box<Process>),  // P | Q
    Input(Name, String, Box<Process>),// x(y).P
    Lift(Name, Box<Process>),         // x<|P|>
    Drop(Name),                       // *x
}

enum Name {
    Quote(Box<Process>),              // @P
}

// Key operations
fn quote(p: Process) -> Name {
    Name::Quote(Box::new(p))
}

fn drop(n: &Name) -> Process {
    match n {
        Name::Quote(p) => *p.clone(),
    }
}

// Structural equality uses quote-drop identity
fn names_equal(n1: &Name, n2: &Name) -> bool {
    // @(*x) = x
    normalize_name(n1) == normalize_name(n2)
}
```

### Reduction Implementation

```rust
fn reduce(p: &Process) -> Option<Process> {
    match p {
        // COMM rule: x<|Q|> | x(y).P -> P{@Q/y}
        Process::Par(left, right) => {
            if let Some((chan_l, body_l)) = as_lift(left) {
                if let Some((chan_r, var, cont)) = as_input(right) {
                    if names_equal(&chan_l, &chan_r) {
                        return Some(substitute(cont, var, &quote(body_l)));
                    }
                }
            }
            // Try symmetric case...
            None
        }
        // Congruence: reduce under par
        Process::Par(l, r) => {
            reduce(l).map(|l2| Process::Par(Box::new(l2), r.clone()))
                .or_else(|| reduce(r).map(|r2| Process::Par(l.clone(), Box::new(r2))))
        }
        _ => None
    }
}
```

---

## Overview

The $`\rho`$-calculus (Meredith & Radestock, 2005) is a **reflective higher-order process
calculus** that:

1. Provides a **closed theory of processes** - names arise from processes themselves
2. Eliminates **higher-order features** as syntactic sugar via reflection
3. Supports **namespace-based scoping** instead of traditional binding

### Key Innovation

Unlike $`\pi`$-calculus where names are primitive, in $`\rho`$-calculus:

> **Names are quoted processes**: $`x = \lceil P \rceil`$

This reflection mechanism enables:
- Self-referential processes
- Higher-order communication without process variables
- Simpler semantic treatments (binding eliminated via quote/dequote)

---

## Syntax

### Process Syntax (P, Q)

```
P, Q ::= 0           ; Null process (inaction)
       | x(y).P      ; Input: receive on channel x, bind to y in P
       | x⟨|P|⟩      ; Lift: send quoted P on channel x
       | ⌊x⌋         ; Drop: evaluate (dequote) name x
       | P | Q       ; Parallel composition
```

### Name Syntax (x, y)

```
x, y ::= ⌈P⌉         ; Quote: the code of process P as a name
```

### Explanation of Constructs

| Construct | Read as | Description |
|-----------|---------|-------------|
| `0` | "zero" or "nil" | Does nothing, terminated process |
| `x(y).P` | "input y on x then P" | Waits for a message on channel x |
| $`x\langle\lvert P\rvert\rangle`$ | "lift P on x" | Sends the code of P on channel x |
| $`\lfloor x\rfloor`$ | "drop x" | Runs the process whose code is x |
| `P \| Q` | "P par Q" | Runs P and Q concurrently |
| $`\lceil P\rceil`$ | "quote P" | The name (code) of process P |

### Binding Structure

The input construct `x(y).P` binds the name `y` in the body `P`. Free names are:

```math
\begin{aligned}
\mathrm{FN}(0) &= \emptyset \\
\mathrm{FN}(x(y).P) &= \{x\} \cup (\mathrm{FN}(P) \setminus \{y\}) \\
\mathrm{FN}(x\langle\lvert P\rvert\rangle) &= \{x\} \cup \mathrm{FN}(P) \\
\mathrm{FN}(\lfloor x\rfloor) &= \{x\} \\
\mathrm{FN}(P \mid Q) &= \mathrm{FN}(P) \cup \mathrm{FN}(Q)
\end{aligned}
```

---

## Structural Congruence

Structural congruence ($`\equiv`$) identifies processes that differ only in structure:

### Parallel Composition Axioms

```math
\begin{aligned}
P \mid 0 &\equiv P && \text{(null is identity)} \\
P \mid Q &\equiv Q \mid P && \text{(commutativity)} \\
(P \mid Q) \mid R &\equiv P \mid (Q \mid R) && \text{(associativity)}
\end{aligned}
```

### Alpha Equivalence

```math
x(y).P \equiv x(z).P\{z/y\} \qquad \text{(rename the bound variable; } z \text{ fresh)}
```

### Name Equivalence

Names have their own equivalence $`\equiv_N`$ based on quotation:

```math
\lceil P\rceil \equiv_N \lceil Q\rceil \quad\text{iff}\quad P \equiv Q \qquad \text{(quoted processes are equal iff their bodies are congruent)}
```

The key identity is **quote-drop**:

```math
\lceil\lfloor x\rfloor\rceil \equiv_N x \qquad \text{(quote of drop is the identity on names)}
```

This means $`x = \lceil\lfloor x\rfloor\rceil`$ - every name is the code of some process (namely, its drop).

---

## Reduction Semantics

### Communication Rule (COMM)

The fundamental reduction rule is communication:

```math
\frac{x_0 \equiv_N x_1}{x_0\langle\lvert Q\rvert\rangle \mid x_1(y).P \;\to\; P\{\lceil Q\rceil / y\}}
```

**Reading**: If a lift on channel x₀ meets an input on channel x₁, and these channels
are name-equivalent, then:
- The input receives the **quoted** sender process $`\lceil Q\rceil`$
- Substituted into the continuation P

### Important: What Gets Sent

The receiver gets $`\lceil Q\rceil`$ (the **code** of Q), not Q itself. To run Q, the receiver
must drop it:

```
x(y).⌊y⌋      ; Receive code and execute it
```

### Congruence Rules

Reduction is closed under structural congruence:

```math
\frac{P \equiv P' \qquad P' \to Q' \qquad Q' \equiv Q}{P \to Q}
```

And under parallel contexts:

```math
\frac{P \to P'}{P \mid Q \to P' \mid Q}
```

---

## The Reflection Mechanism

The quote-drop mechanism is the heart of RHO calculus.

### Quote: $`\lceil P\rceil`$

**Quote** reifies a process as a name (its "code"):

```
⌈0⌉             ; Name representing null process
⌈x(y).P⌉        ; Name representing an input process
⌈P | Q⌉         ; Name representing a parallel composition
```

Quoting is **syntactic** - it captures the structure of P, not its behavior.

### Drop: $`\lfloor x\rfloor`$

**Drop** executes a name (runs its code):

```math
\lfloor\lceil P\rceil\rfloor \equiv P \qquad \text{(drop of quote recovers the process)}
```

Drop is the inverse of quote on well-formed names.

### Lift: $`x\langle\lvert P\rvert\rangle`$

**Lift** is output with implicit quotation, but susceptible to substitution:

```
x⟨|P|⟩          ; Output the code of P on channel x
```

The difference from standard output $`x\langle\lceil P\rceil\rangle`$ is that lift's body can contain free
variables that are substituted:

```math
(\lambda x.\, y\langle\lvert x\rvert\rangle)(P) = y\langle\lvert P\rvert\rangle \qquad \text{(substitution penetrates lift)}
```

### Why Reflection Matters for Type Checking

The RHO paper states:

> "Reflection provides a powerful technique for treating nominal phenomena as
> syntactic sugar, thus paving the way for simpler semantic treatments."

This validates using Gph-enriched Lawvere theories
([03-gph-enriched-lawvere.md](./03-gph-enriched-lawvere.md)) for the
MeTTaIL/Rholang integration:

- Bound variables can be eliminated via quote/drop
- Higher-order features encoded without process variables
- Simpler operational semantics suffice

---

## Bisimulation

### N-Barbed Bisimulation

The paper parameterizes bisimulation by a set N of **observable names**:

**Definition (N-Barb)**: Process P has an N-barb at name x, written P ↓_N x, if:
- P can input or output on x
- $`x \in N`$ (the name is observable)

**Definition (N-Barbed Bisimulation)**: A relation R is an N-barbed bisimulation if
whenever P R Q:

1. **Reduction closure**: If $`P \to P'`$ then $`\exists Q'.\; Q \to^{*} Q'`$ and $`P' \mathrel{R} Q'`$
2. **Barb preservation**: If P ↓_N x then Q ⇓_N x (weak barb)
3. **Symmetry**: The same conditions hold with P and Q swapped

### Why Parameterized Bisimulation?

N-parameterization enables:

1. **Scope reasoning**: Only observe names in scope
2. **Namespace security**: Verify isolation properties
3. **Behavioral types**: Types as predicates on observable names

### Example: Namespace Security

```
; Process that only communicates on namespace α
safe(α) := P where FN(P) ⊆ α ∧ P ↓_N x ⟹ x ∈ α

; Compile-time firewall (from OSLF):
sole.in(α) := νX. (in(α, N → X) | P) ∧ ¬[in(¬[α], N → P) | P]
```

This type says: "Can input on channels in $`\alpha`$ and cannot input on $`\lnot\alpha`$."

---

## Derived Constructs

### Private Names ($`\nu`$)

Unlike $`\pi`$-calculus, RHO doesn't need primitive $`\nu`$ (new name). It's derived:

```
(νx)P := P{⌈0⌉/x}     ; Use a fresh quoted process
```

More generally, use any term guaranteed to be fresh.

### Replication (!)

Replication is also derived (using recursion through reflection):

```
!P := ⌊⌈P | ⌊⌈P⌉⌋⌉⌋    ; Self-replicating via quote-drop
```

Or using a standard fixed-point encoding.

### Higher-Order Communication

Sending and receiving processes directly:

```
; Higher-order send (send process P on channel x)
ho-send(x, P) := x⟨|P|⟩

; Higher-order receive (receive and run)
ho-recv(x) := x(y).⌊y⌋

; Communication:
ho-send(x, P) | ho-recv(x) → P
```

---

## Connection to Type Checking

### RHO Calculus in Rholang

Rholang is essentially an implementation of RHO calculus with:
- Concrete syntax for processes
- System processes for I/O, storage, etc.
- Integration with RSpace (tuple space)

### Integration with MeTTaIL

Since MeTTaIL will become the next version of Rholang:

1. **MeTTaIL provides type checking** at compile time
2. **RHO calculus runtime** executes the typed processes
3. **Behavioral types** from OSLF match RHO's bisimulation semantics

### The Bridge: Quote/Drop ↔ MeTTa Quotation

MeTTa also has quotation (terms as data). The correspondence:

| RHO Calculus | MeTTa |
|--------------|-------|
| $`\lceil P\rceil`$ (quote) | `(quote P)` |
| $`\lfloor x\rfloor`$ (drop) | `(eval x)` or unquote |
| P \| Q | Parallel in knowledge base |
| x(y).P | Pattern matching with binding |

### Behavioral Types for RHO/Rholang

Using OSLF on RHO calculus gives behavioral types:

```
; Type of processes that always respond on channel x
Responsive(x) := □(◇(↓_x))

; Type of processes that never deadlock
Deadlock-free := □(0 ∨ ◇(→))

; Type of processes that terminate
Terminating := ◇(≡ 0)
```

These types can be checked at compile time by MeTTaIL and enforced at runtime by
Rholang.

---

## Comparison with Lambda and SKI

Understanding how RHO relates to simpler calculi illuminates its design choices.

### Structural Comparison

| Aspect | Lambda-Calculus | SKI | RHO |
|--------|-----------------|-----|-----|
| **Computational model** | Functions | Combinators | Processes |
| **Primary interaction** | Application | Application | Parallel communication |
| **Binding** | Lambda (lambda x.M) | None (encoded) | Input (x(y).P) |
| **State** | Stateless | Stateless | Process soup (multiset) |
| **Determinism** | Deterministic* | Deterministic | Non-deterministic |
| **Names** | Variables | Variables | Quoted processes |

*Given evaluation strategy

### Reduction Rules Comparison

| Calculus | Rule Name | Source | Target |
|----------|-----------|--------|--------|
| **Lambda** | Beta | `(lambda x.M) N` | `M[N/x]` |
| **SKI** | S | `S x y z` | `(x z) (y z)` |
| **SKI** | K | `K x y` | `x` |
| **SKI** | I | `I x` | `x` |
| **RHO** | Comm | `x<\|Q\|> \| x(y).P` | `P[@Q/y]` |

### GSLT Form Comparison

**Lambda**:
```
App: P x P -> P
Lam: (P -> P) -> P
Beta: (P -> P) x P -> R
```

**SKI**:
```
S, K, I: 1 -> P
App: P x P -> P
Sigma, Kappa, Iota: various -> R
```

**RHO**:
```
0: 1 -> P
|: P x P -> P
!: N x P -> P
?: N x (N -> P) -> P
Comm: N x (N -> P) x P -> R
```

### Key Insight: Reflection as Binding Elimination

RHO's quote/drop mechanism (`@`, `*`) serves a similar role to SKI's combinator
encoding of lambda:

| Encoding | Source | Mechanism |
|----------|--------|-----------|
| Lambda -> SKI | `lambda x.M` | `S`, `K`, `I` combinators |
| Pi -> RHO | `nu x.P` | `@P` (quote creates fresh name) |
| RHO binding | `x(y).P` | `y` substituted with `@Q` |

This is why RHO can use Gph-enriched Lawvere theories (like SKI) rather than
requiring full OSLF (like lambda with genuine binding).

### Type-Lifting Implications

When type-lifted (see [05-type-lifting.md](./05-type-lifting.md)):

| Calculus | Exponential in binding? | Duplication? | Extra type factors? |
|----------|-------------------------|--------------|---------------------|
| **Lambda** | Yes (Lam) | No | Yes (bound var type) |
| **SKI** | No | Yes (S) | Yes (for z) |
| **RHO** | Yes (?) | Yes (Comm) | Yes (both) |

RHO inherits complexity from both:
- Exponential handling from lambda (for input binder)
- Duplication handling from SKI (for shared channel)

---

## Summary

The RHO calculus provides:

1. **Closed theory of processes** - names arise from quotation
2. **Reflection mechanism** - quote/drop for code as data
3. **Simpler semantics** - binding eliminated via reflection
4. **N-barbed bisimulation** - behavioral equivalence parameterized by observables
5. **Foundation for Rholang** - the theoretical model for execution

The reflection mechanism is key to the MeTTaIL integration: it enables the simpler
Gph-enriched Lawvere approach while maintaining full expressiveness.

---

## Related Documents

- [03-gph-enriched-lawvere.md](./03-gph-enriched-lawvere.md): GSLTs and the RHO GSLT
- [05-type-lifting.md](./05-type-lifting.md): Type lifting for RHO
- [06-inference-rules.md](./06-inference-rules.md): Typing rules including RHO examples

---

## References

- Meredith, L. G. & Radestock, M. "A Reflective Higher-order Calculus."
  ENTCS 141(5), pp. 49-67, 2005. DOI: [10.1016/j.entcs.2005.05.016](https://doi.org/10.1016/j.entcs.2005.05.016).
- Sangiorgi, D. "The pi-calculus: A Theory of Mobile Processes." Cambridge, 2001.
- See [bibliography.md](../reference/bibliography.md) for complete references.
