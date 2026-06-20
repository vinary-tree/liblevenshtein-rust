[← Documentation Index](../../README.md)

# OpenCog Hyperon Architecture

This document provides a comprehensive overview of OpenCog Hyperon, the cognitive
architecture that underlies MeTTa and serves as the foundation for MeTTaIL's
semantic type checking capabilities.

**Source**: Goertzel et al. "OpenCog Hyperon: A Framework for AGI at the Human
Level and Beyond" (2023)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architectural Overview](#architectural-overview)
3. [Atomspace: The Metagraph Knowledge Store](#atomspace-the-metagraph-knowledge-store)
4. [Four Atom Meta-Types](#four-atom-meta-types)
5. [Two-Layer Type System](#two-layer-type-system)
6. [Pattern Matching as Core Operation](#pattern-matching-as-core-operation)
7. [Gradual Typing and Paraconsistent Logic](#gradual-typing-and-paraconsistent-logic)
8. [Distributed Atomspace](#distributed-atomspace)
9. [Cognitive Synergy](#cognitive-synergy)
10. [Relevance to MeTTaIL](#relevance-to-mettail)

---

## Introduction

OpenCog Hyperon is an open-source framework designed for Artificial General
Intelligence (AGI). Unlike narrow AI systems optimized for specific tasks,
Hyperon provides a flexible infrastructure where multiple AI paradigms can
cooperate and share knowledge through a unified representation.

### Design Goals

1. **Cognitive Flexibility**: Support multiple AI approaches simultaneously
2. **Knowledge Integration**: Unified representation across paradigms
3. **Scalability**: Distributed operation across compute clusters
4. **Introspection**: Self-modifying, self-improving capabilities
5. **Theoretical Foundation**: Grounded in formal semantics

---

## Architectural Overview

The Hyperon architecture consists of layered components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         AI Agents                                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │  GOFAI  │  │  Neural │  │  Evol.  │  │ Prob.   │            │
│  │  Logic  │  │  Nets   │  │  Algo.  │  │ Graphs  │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                   │
│       └────────────┴────────────┴────────────┘                  │
│                         │                                        │
│              ┌──────────▼──────────┐                            │
│              │       MeTTa         │  ← Programming Language    │
│              │  (Meta Type Talk)   │                            │
│              └──────────┬──────────┘                            │
│                         │                                        │
│              ┌──────────▼──────────┐                            │
│              │     Atomspace       │  ← Metagraph Knowledge     │
│              │   (Atomese Core)    │     Store                  │
│              └──────────┬──────────┘                            │
│                         │                                        │
│              ┌──────────▼──────────┐                            │
│              │  Distributed AS     │  ← Scalable Storage        │
│              │       (DAS)         │                            │
│              └──────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Descriptions

| Layer | Component | Purpose |
|-------|-----------|---------|
| Top | AI Agents | Domain-specific algorithms |
| Middle | MeTTa | Programming language interface |
| Core | Atomspace | Knowledge representation & manipulation |
| Base | DAS | Distributed storage & retrieval |

---

## Atomspace: The Metagraph Knowledge Store

The Atomspace is a **metagraph** data structure that stores all knowledge in
Hyperon. Unlike traditional graphs, metagraphs allow edges to connect to other
edges, enabling higher-order relationships.

### Definition: Metagraph

A metagraph M = (V, E) where:
- V is a set of vertices (Atoms)
- E ⊆ P(V ∪ E) × P(V ∪ E) is a set of hyperedges that can connect vertices or other edges

### Key Properties

1. **Hyperedges**: Edges can connect arbitrary sets of nodes or edges
2. **Typed Nodes**: Every atom has an associated type
3. **Attention Values**: Atoms carry importance/relevance weights
4. **Truth Values**: Probabilistic confidence annotations
5. **Grounding**: External function/data attachment

### Example Structure

```
Atomspace:
  (Inheritance "cat" "animal")
  (Evaluation "has-property"
    (List "cat" "furry"))
  (TypedAtom
    (: "add" (-> Number Number Number)))
```

---

## Four Atom Meta-Types

MeTTa uses exactly **four fundamental atom meta-types**. Every atom in the
system is classified as one of these:

### 1. Symbol

A named constant representing an identifier or value.

```metta
; Symbols
foo          ; Named identifier
+            ; Operator symbol
Type         ; Type name
"string"     ; String literal (also a symbol)
```

**Properties**:
- Immutable
- Hashable for efficient lookup
- Used for constructors, operators, type names

### 2. Variable

A placeholder for pattern matching and unification.

```metta
; Variables (prefixed with $)
$x           ; Simple variable
$pattern     ; Named pattern variable
$_           ; Anonymous/wildcard variable
```

**Properties**:
- Bound during pattern matching
- Scope determined by enclosing expression
- Support unification with constraints

### 3. Expression

An ordered list of atoms (can contain any meta-type).

```metta
; Expressions (parenthesized lists)
(+ 1 2)                    ; Function application
(: x Int)                  ; Type annotation
(= (f $x) (g $x $x))       ; Equality declaration
(if $cond $then $else)     ; Control flow
```

**Properties**:
- Ordered (position matters)
- Heterogeneous (mixed types allowed)
- Recursive (expressions contain expressions)

### 4. Grounded

Foreign data or functions from the host language (Rust, Python, etc.).

```metta
; Grounded atoms (opaque to MeTTa)
<Rust::HashMap>            ; External data structure
<Python::numpy.array>      ; Foreign array
<fn:add-integers>          ; External function
```

**Properties**:
- Opaque to MeTTa's pattern matcher
- Evaluated by host language
- Enable external system integration

### Meta-Type Hierarchy

```
                    Atom
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
   ┌───────┐   ┌───────────┐  ┌──────────┐
   │Symbol │   │Expression │  │ Grounded │
   └───────┘   └───────────┘  └──────────┘
       │
       ▼
   ┌───────┐
   │Variable│
   └───────┘
```

---

## Two-Layer Type System

Atomese 2 (MeTTa's underlying representation) employs a **two-layer type
architecture** designed for maximum flexibility:

### Layer 1: Generic Core

The generic core provides minimal type infrastructure:

```metta
; Type declaration syntax
(: expr TypeName)

; Examples
(: 42 Number)
(: "hello" String)
(: add (-> Number Number Number))
```

**Core Features**:
- Basic type annotations
- Gradual typing (optional annotations)
- Type variables for polymorphism
- Structural subtyping

### Layer 2: Specialized Type Systems

Domain-specific type systems built atop the core:

```metta
; Example: Linear type system for resource tracking
(: LinearType Type)
(: consume (-> (Linear $a) Unit))

; Example: Dependent types for refinement
(: Vector (-> Type Nat Type))
(: zeros (-> (n : Nat) (Vector Int n)))
```

**Specialized Systems**:
- **Linear types**: Resource tracking
- **Dependent types**: Value-indexed types
- **Behavioral types**: Process properties (OSLF)
- **Spatial types**: Namespace constraints

### Why Two Layers?

> "We designed Atomese 2 with two separate layers: a generic core, plus one or
> more specific type systems that define their own notions of 'type' and their
> own checking/inference algorithms."
>
> — OpenCog Hyperon Paper

This separation enables:
1. **Interoperability**: Different systems can share the core
2. **Experimentation**: New type systems without core changes
3. **Gradual adoption**: Add types incrementally
4. **Domain optimization**: Specialized checking per domain

---

## Pattern Matching as Core Operation

Pattern matching is the **fundamental operation** in MeTTa. All computation
proceeds via matching patterns against the Atomspace.

### Basic Pattern Matching

```metta
; Knowledge base
(parent Alice Bob)
(parent Bob Carol)

; Query pattern (returns all matches)
!(match &self (parent $x $y) ($x $y))
; Result: ((Alice Bob) (Bob Carol))
```

### Unification Semantics

Pattern matching implements **unification with occurs check**:

```
match(pattern, target) → BindingsSet

Where BindingsSet = { σ₁, σ₂, ... } and each σᵢ : Var → Atom
```

**Algorithm Properties**:
- **Soundness**: σ(pattern) = target for all σ in result
- **Completeness**: All unifiers found
- **Most general**: Returns MGU when unique

### Rewrite Rules

Computation proceeds via conditional rewriting:

```metta
; Define rewrite rule
(= (factorial 0) 1)
(= (factorial $n)
   (* $n (factorial (- $n 1))))

; Evaluate (triggers rewrites)
!(factorial 5)
; Result: 120
```

### The Seven Minimal Operations

From Meta-MeTTa, the core operations are:

| Operation | Purpose |
|-----------|---------|
| `eval` | Evaluate expression |
| `evalc` | Conditional evaluation |
| `chain` | Sequential composition |
| `function`/`return` | Functional abstraction |
| `unify` | Pattern unification |
| `cons-atom`/`decons-atom` | List construction |
| `collapse-bind`/`superpose-bind` | Nondeterminism |

---

## Gradual Typing and Paraconsistent Logic

MeTTa supports **gradual typing**, allowing typed and untyped code to coexist.

### Gradual Typing Semantics

```metta
; Fully typed
(: add (-> Int Int Int))
(= (add $x $y) (+ $x $y))

; Untyped (dynamically checked)
(= (flexible-add $x $y) (+ $x $y))

; Mixed (type at boundary)
(= (safe-add $x $y)
   (if (and (is-int $x) (is-int $y))
       (add $x $y)
       (Error "Type mismatch")))
```

### Connection to Paraconsistent Logic

Via the Curry-Howard correspondence, gradual types map to **paraconsistent
logic** — a logic that tolerates contradictions without explosion:

```
Gradual Type System ↔ Paraconsistent Logic
         │                    │
    Type errors          Contradictions
    are localized        don't propagate
```

**Implications**:
- Type errors don't crash the entire system
- Partially typed code can still execute
- Inconsistencies are contained to their scope

---

## Distributed Atomspace

The **Distributed Atomspace (DAS)** enables Hyperon to scale across machines.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Distributed Atomspace                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Node 1    │  │   Node 2    │  │   Node 3    │             │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │             │
│  │ │Local AS │ │  │ │Local AS │ │  │ │Local AS │ │             │
│  │ └────┬────┘ │  │ └────┬────┘ │  │ └────┬────┘ │             │
│  │      │      │  │      │      │  │      │      │             │
│  │ ┌────▼────┐ │  │ ┌────▼────┐ │  │ ┌────▼────┐ │             │
│  │ │ Cache   │ │  │ │ Cache   │ │  │ │ Cache   │ │             │
│  │ └────┬────┘ │  │ └────┬────┘ │  │ └────┬────┘ │             │
│  └──────┼──────┘  └──────┼──────┘  └──────┼──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                       │
│                 ┌────────▼────────┐                             │
│                 │  Query Router   │                             │
│                 └────────┬────────┘                             │
│                          │                                       │
│                 ┌────────▼────────┐                             │
│                 │   Index Layer   │                             │
│                 │ (MongoDB/Redis) │                             │
│                 └─────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### Query Distribution

Pattern queries are distributed using:

1. **Hash-based partitioning**: Atoms sharded by hash
2. **Index-based routing**: Queries directed to relevant shards
3. **Result aggregation**: Distributed joins for complex queries

---

## Cognitive Synergy

Hyperon's architecture enables **cognitive synergy** — multiple AI paradigms
cooperating through shared knowledge.

### Paradigm Integration

| Paradigm | Atomspace Representation | Interaction Mode |
|----------|-------------------------|------------------|
| Symbolic AI | Logic atoms, rules | Direct manipulation |
| Neural Networks | Weight atoms, activations | Grounded atoms |
| Evolutionary | Population atoms | Mutation operators |
| Probabilistic | Truth values, distributions | Inference atoms |

### Synergy Example

```metta
; Neural network suggests candidates
(neural-suggest "image-features" $candidates)

; Symbolic reasoning filters
(logical-filter $candidates
  (satisfies safety-constraint))

; Evolutionary search optimizes
(evolve-solution $filtered
  fitness-function
  100)  ; generations
```

---

## Relevance to MeTTaIL

The OpenCog Hyperon architecture directly informs MeTTaIL's design:

### Type System Alignment

| Hyperon Concept | MeTTaIL Mapping |
|-----------------|-----------------|
| Two-layer types | Basic sorts + OSLF predicates |
| Gradual typing | Optional behavioral annotations |
| Metagraph | Theory categories |
| Pattern matching | Predicate evaluation |

### Implementation Path

1. **Layer 1 (Current)**: Sort-based validation in MeTTaIL
2. **Layer 2 (Planned)**: OSLF behavioral predicates

### Key Insight

The two-layer type system from Atomese 2 validates MeTTaIL's approach:
- **Generic core** = MeTTaIL's sort/constructor checking
- **Specialized systems** = OSLF behavioral types

This alignment ensures MeTTaIL can serve as MeTTa's native type system while
supporting advanced behavioral reasoning for Rholang integration.

---

## Summary

OpenCog Hyperon provides:

1. **Metagraph knowledge store** (Atomspace)
2. **Four atom meta-types** (Symbol, Variable, Expression, Grounded)
3. **Two-layer type system** (generic core + specialized)
4. **Pattern matching** as the core operation
5. **Gradual typing** with paraconsistent semantics
6. **Distributed scalability** (DAS)
7. **Cognitive synergy** across AI paradigms

These foundations directly support MeTTaIL's semantic type checking goals and
provide the theoretical justification for the layered approach to behavioral
types.

---

## References

- Goertzel et al. "OpenCog Hyperon: A Framework for AGI at the Human Level and
  Beyond" (2023)
- See [bibliography.md](../reference/bibliography.md) for complete references
- See [02-hyperon-experimental.md](./02-hyperon-experimental.md) for implementation details
