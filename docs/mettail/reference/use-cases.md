[← Documentation Index](../../README.md)

# Semantic Type Checking Use Cases

This document provides concrete examples of semantic type checking for MeTTa and
Rholang, demonstrating the practical value of OSLF-based types.

**Primary Sources**:
- Williams, P. & Stay, M. "Native Type Theory" (EPTCS 372, 2022)
- Meredith, L. G. et al. "rhocube.spatial-behavioral.types.tex" (Spatial-behavioral types)

---

## Table of Contents

1. [Spatial-Behavioral Type System](#spatial-behavioral-type-system)
2. [Type Inference Rules](#type-inference-rules)
3. [Query Validation](#query-validation)
4. [Namespace Security](#namespace-security)
5. [Termination Properties](#termination-properties)
6. [Behavioral Equivalence](#behavioral-equivalence)
7. [Resource Bounds](#resource-bounds)
8. [Contract Types](#contract-types)
9. [Correction WFST Integration](#correction-wfst-integration)

---

## Spatial-Behavioral Type System

The type system presented here is **generated** from the operational semantics using
Meredith and Stay's OSLF algorithm. This section provides the formal type syntax and
inference rules.

### Type Syntax

The following grammar defines the spatial-behavioral type language:

```
Process Types:
  T, U ::= 0                      ; nil process type
         | GT                     ; ground type
         | ⟨(TT ← N)⟩T            ; for-comprehension type
         | ⟨x?⟩T                  ; reflection type
         | *N                     ; dereference type
         | T | U                  ; parallel composition type

Name Types:
  N ::= @T                        ; quoted process type

Term Types:
  TT ::= AtomT                    ; atomic term type
       | ([T])                    ; list of process types

Atom Types:
  AtomT ::= N                     ; name atom
          | T                     ; process atom

Ground Types:
  GT ::= Bool                     ; boolean
       | String                   ; string
       | Int                      ; integer
       | C                        ; collection

Collection Types:
  C ::= List(TT)                  ; homogeneous list
      | Tuple(TT, [TT]_seq)       ; heterogeneous tuple
      | Set(TT)                   ; homogeneous set
      | Map{[TT : TT]}            ; map from keys to values
```

### Type Notation Conventions

| Notation | Meaning |
|----------|---------|
| $`\Gamma  \vdash  P : T`$ | In context $`\Gamma ,`$ process P has type T |
| $`[\text{Judgment}] \vdash  \text{Judgment}`$ | Horizontal inference rule |
| $`\vdash  J`$ | Axiom (no premises) |
| $`\Gamma , \Delta`$ | Context concatenation |
| `T \| U` | Parallel composition type |
| `@T` | Quote of process type |
| `*N` | Dereference of name type |

---

## Type Inference Rules

These rules are specific to the MeTTa-calculus. The OSLF algorithm also generates
ambient typing judgments common to all systems expressible as graph-structured
lambda theories.

### Ground Type Rules

```
                                    ; Ground types are axiomatic
─────────────────────────────────── [ground-bool]
⊢ BoolLiteral : Bool

─────────────────────────────────── [ground-string]
⊢ StringLiteral : String

─────────────────────────────────── [ground-int]
⊢ IntLiteral : Int
```

### Collection Type Rules

**Lists** (homogeneous):

```
Γ₁ ⊢ t₁ : TT   ...   Γₙ ⊢ tₙ : TT
─────────────────────────────────────────────── [collection-list]
Γ₁, ..., Γₙ ⊢ [t₁, ..., tₙ] : List[TT]
```

**Tuples** (heterogeneous):

```
Γ₁ ⊢ t₁ : TT₁   ...   Γₙ ⊢ tₙ : TTₙ
────────────────────────────────────────────────────── [collection-tuple]
Γ₁, ..., Γₙ ⊢ (t₁, ..., tₙ) : Tuple(TT₁, ..., TTₙ)
```

**Sets** (homogeneous):

```
Γ₁ ⊢ t₁ : TT   ...   Γₙ ⊢ tₙ : TT
─────────────────────────────────────────────── [collection-set]
Γ₁, ..., Γₙ ⊢ Set(t₁, ..., tₙ) : Set(TT)
```

**Maps** (key-value pairs):

```
Γ₁ ⊢ k₁ : TT₁   Δ₁ ⊢ v₁ : TT₂   ...   Γₙ ⊢ kₙ : TT₂ₙ₋₁   Δₙ ⊢ vₙ : TT₂ₙ
─────────────────────────────────────────────────────────────────────────────
Γ₁, Δ₁, ..., Γₙ, Δₙ ⊢ Map{k₁:v₁, ..., kₙ:vₙ} : Map{TT₁:TT₂, ..., TT₂ₙ₋₁:TT₂ₙ}
                                                            [collection-map]
```

### Process Type Rules

**For-Comprehension** (input):

```
t : TT, Γ ⊢ P : T       Δ ⊢ x : V
───────────────────────────────────────── [for-comprehension]
Γ, Δ ⊢ for(t ← x)P : ⟨(TT ← V)⟩T
```

This types input patterns: receiving a term of type TT on channel of type V,
continuing with process of type T.

**Reflection** (peek/query):

```
Γ ⊢ P : T       Δ ⊢ x : V
───────────────────────────── [refl]
Γ, Δ ⊢ x?P : ⟨?V⟩T
```

This types reflection operations: querying channel x while continuing with P.

**Parallel Composition**:

```
Γ ⊢ P : T       Δ ⊢ Q : U
───────────────────────────── [parallel]
Γ, Δ ⊢ P | Q : T | U
```

The type of a parallel composition is the parallel composition of types.

**Dereference**:

```
Γ ⊢ x : V
──────────────── [deref]
Γ ⊢ *x : *V
```

Dereferencing a name gives the dereference type.

**Quote**:

```
Γ ⊢ P : T
────────────── [quote]
Γ ⊢ @P : @T
```

Quoting a process gives the quoted type.

### Structural Equations

These equations establish type equivalences based on structural congruence:

**Parallel Monoid Identity**:

```
Γ ⊢ P : T
──────────────────────── [ParMonoidId]
Γ ⊢ P | 0 = P : T
```

**Parallel Monoid Associativity**:

```
Γ ⊢ (P | Q) | R : T
─────────────────────────────────────── [ParMonoidAssoc]
Γ ⊢ (P | Q) | R = P | (Q | R) : T
```

**Parallel Monoid Commutativity**:

```
Γ ⊢ P | Q : T
──────────────────────── [ParMonoidComm]
Γ ⊢ P | Q = Q | P : T
```

### Lifted Redex Constructors

These rules type reduction contexts, enabling behavioral type checking:

**Communication Redex**:

```
Γ₁ ⊢ x : V   t₁ : TT₁, Γ₂ ⊢ P : T₁   t₂ : TT₂, Γ₃ ⊢ Q : T₂
────────────────────────────────────────────────────────────── [Comm]
Γ₁, Γ₂, Γ₃ ⊢ Comm(x, t₁, t₂, P, Q) : Comm(V, TT₁, TT₂, T₁, T₂)
```

**Evaluation Redex**:

```
Γ ⊢ P : T
────────────────────────────── [Eval]
Γ ⊢ eval(P) : eval(T)
```

**Parallel Left Context**:

```
Γ ⊢ R : E       Δ ⊢ P : T
────────────────────────────────── [ParL]
Γ, Δ ⊢ parL(R, P) : parL(E, T)
```

**Parallel Right Context**:

```
Γ ⊢ R : E       Δ ⊢ P : T
────────────────────────────────── [ParR]
Γ, Δ ⊢ parR(P, R) : parR(T, E)
```

### Reduction Rules from the $`\lambda`$-theory

These rules type the source and target of reductions:

**Communication Source**:

```
Γ ⊢ Comm(x, t, u, P, Q) : Comm(V, TT₁, TT₂, T, U)
─────────────────────────────────────────────────────────────── [Comm-Src]
Γ ⊢ src(Comm(x, t, u, P, Q)) = for(t ← x)P | for(u ← x)Q
                              : ⟨TT₁ ← V⟩T | ⟨TT₂ ← V⟩U
```

**Communication Target**:

```
Γ ⊢ Comm(x, t, u, P, Q) : Comm(V, TT₁, TT₂, T, U)
─────────────────────────────────────────────────────────── [Comm-Trgt]
Γ ⊢ trgt(Comm(x, t, u, P, Q)) = Pσ̇ | Qσ̇ : T | U
```

Where σ̇ is the substitution from pattern matching.

**Evaluation Source**:

```
Γ ⊢ eval(P) : eval(T)
─────────────────────────────────────────── [Eval-Src]
Γ ⊢ src(eval(P)) = *@P : *@T
```

**Evaluation Target**:

```
Γ ⊢ eval(P) : eval(U) : R
─────────────────────────────────── [Eval-Trgt]
Γ ⊢ trgt(eval(P)) = P : U
```

**Parallel Context Source**:

```
Γ ⊢ parX(R, P) : parX(E, T)
──────────────────────────────────────────────── [Par-Src]
Γ ⊢ src(parX(R, P)) = src(R) | P : src(E) | T
```

**Parallel Context Target**:

```
Γ ⊢ parX(R, P) : parX(E, T)
────────────────────────────────────────────────── [Par-Trgt]
Γ ⊢ trgt(parX(R, P)) = trgt(R) | P : trgt(E) | T
```

---

## Query Validation

### Problem

Ensure that pattern queries only match well-typed knowledge in the knowledge base.

### MeTTa Example

```metta
; Knowledge base with typed facts
(: parent (-> Person Person Bool))
(parent Alice Bob)
(parent Bob Carol)

; Query - should only match Person × Person
!(match &self (parent $x $y) ($x $y))
```

### Type Specification

```
; Predicate: query matches typed knowledge
valid-query : Pattern × KB → Prop

valid-query(p, k) := ∀σ. match(p, k) = σ ⟹ well-typed(σ(p))
```

### Implementation in MeTTaIL

```rust
// Define predicate for valid queries
let valid_query = Predicate::Forall(
    Var::new("σ"),
    Sort::Substitution,
    Box::new(Predicate::Implies(
        Box::new(Predicate::Equals(
            Term::App("match", vec![pattern.clone(), kb.clone()]),
            Term::Var(Var::new("σ")),
        )),
        Box::new(Predicate::WellTyped(
            Term::Subst(pattern.clone(), Var::new("σ")),
        )),
    )),
);

// Check the predicate
checker.check_predicate(&valid_query, &env)?;
```

### Rholang Equivalent

```rholang
// Type-annotated query contract
contract query(@pattern: Pattern, kb: Name[KB], return: Name[List[Match]]) = {
  // Type system ensures pattern matches KB type
  for(@knowledge <- kb) {
    return!(match(pattern, knowledge))
  }
}
```

### Using Spatial-Behavioral Types

With the type system above, a query can be typed as:

```
query_type : ⟨(Pattern ← @KB)⟩List[Match]
```

The for-comprehension rule ensures that:
1. The pattern type matches the knowledge base type
2. The result is properly typed as a list of matches

---

## Namespace Security

### Problem

Guarantee that a process only communicates on channels within a trusted namespace.

### RHO Calculus Type

From the OSLF paper ($`\rho\pi`$-calculus example):

```
; Type for processes that only input on namespace α
sole.in(α) := νX. (in(α, N → X) | P) ∧ ¬[in(¬[α], N → P) | P]
```

**Reading**: Can input on channels in type $`\alpha ,`$ cannot input on $`\lnot \alpha ,`$ and continues
with this property.

### Spatial-Behavioral Formulation

Using the type inference rules:

```
; For a process to be isolated to namespace α:
; All for-comprehensions must have channel type in α

Γ ⊢ for(t ← x)P : ⟨(TT ← V)⟩T
────────────────────────────────── [namespace-check]
V ∈ α  ⟹  Γ ⊢ P : isolated(α)
```

### Behavioral Implementation

```rust
// Define namespace isolation predicate
fn namespace_isolated(namespace: &Namespace) -> Predicate {
    Predicate::Always(Box::new(
        Predicate::Forall(
            Var::new("ch"),
            Sort::Channel,
            Box::new(Predicate::Implies(
                Box::new(Predicate::CanCommunicate(Var::new("ch"))),
                Box::new(Predicate::InNamespace(Var::new("ch"), namespace.clone())),
            )),
        ),
    ))
}
```

### Rholang Example

```rholang
// Annotate contract with namespace restriction
@isolated(internal)
contract worker(internal: Namespace, external: Namespace) = {
  // This should type check - communicating within allowed namespace
  new ch in {
    internal.ch!(42) | for(@x <- internal.ch) { ... }
  }

  // This should fail type check - communicating outside namespace
  // external.leak!("secret")  // ERROR: violates @isolated(internal)
}
```

### Verification Using Type Rules

```
; Checking isolation via for-comprehension rule:

internal : @Internal, Γ ⊢ ch!(42) : ⟨(Int ← @Internal)⟩T
─────────────────────────────────────────────────────────────
@Internal ∈ internal  ✓  (allowed)

external : @External, Γ ⊢ leak!("secret") : ⟨(String ← @External)⟩T
──────────────────────────────────────────────────────────────────────
@External ∉ internal  ✗  (namespace violation)
```

---

## Termination Properties

### Problem

Verify that a computation will eventually terminate.

### Behavioral Type

```
; Type of terminating computations
Terminates := ◇(NormalForm)

; Where ◇ is "eventually" (least fixed point of possibility)
◇φ := μX. φ ∨ F!(X)
```

### MeTTa Example

```metta
; This should type as terminating
(= (factorial 0) 1)
(= (factorial $n) (* $n (factorial (- $n 1))))

; Check: all recursive calls decrease toward base case
```

### Connection to Lifted Redex Types

Using the Eval rule:

```
Γ ⊢ (factorial n) : Nat
───────────────────────────────── [Eval]
Γ ⊢ eval((factorial n)) : eval(Nat)

; Termination requires:
; trgt(eval(factorial n)) = result : Nat  (reaches normal form)
```

### Implementation

```rust
// Define termination predicate
fn terminates() -> Predicate {
    Predicate::Eventually(Box::new(Predicate::NormalForm))
}

// Check termination for a term
fn check_termination(term: &Term, theory: &Theory) -> Result<(), TypeError> {
    let graph = theory.build_reduction_graph(term);

    // Check if all paths reach normal form
    if graph.all_paths_reach(&Predicate::NormalForm) {
        Ok(())
    } else {
        // Find a non-terminating path for error message
        let cycle = graph.find_cycle();
        Err(TypeError::NonTerminating { cycle })
    }
}
```

### Rholang Example

```rholang
// Annotate with termination requirement
@terminating
contract fibonacci(@n: Nat, return: Name[Nat]) = {
  match n {
    0 => return!(0)
    1 => return!(1)
    _ => {
      new a, b in {
        fibonacci!(n - 1, *a) | fibonacci!(n - 2, *b) |
        for(@x <- a; @y <- b) {
          return!(x + y)
        }
      }
    }
  }
}
```

---

## Behavioral Equivalence

### Problem

Verify that two processes are behaviorally equivalent (bisimilar).

### Type

```
; Bisimulation as inductive type
Bisim := μφ. S(φ)

; Where S is simulation step
S(φ)(P, Q) := (∀P'. P → P' ⟹ ∃Q'. Q →* Q' ∧ φ(P', Q'))
            ∧ (∀Q'. Q → Q' ⟹ ∃P'. P →* P' ∧ φ(P', Q'))
```

### Using Reduction Types for Equivalence

From the type rules:

```
; If two processes have the same type and reduction structure,
; they are candidates for bisimulation

Γ ⊢ P : T       Γ ⊢ Q : T
Γ ⊢ src(R_P) : src(E)    Γ ⊢ src(R_Q) : src(E)
Γ ⊢ trgt(R_P) : trgt(E)  Γ ⊢ trgt(R_Q) : trgt(E)
───────────────────────────────────────────────────
P ≈ Q  (bisimulation candidate)
```

### MeTTa Example

```metta
; These should be provably equivalent
(= (double $x) (+ $x $x))
(= (double' $x) (* 2 $x))

; Check: double ≈ double'
```

### Implementation

```rust
// Check bisimulation
fn check_bisimilar(p: &Process, q: &Process, theory: &Theory) -> Result<(), TypeError> {
    let relation = compute_bisimulation(p, q, theory);

    if relation.contains(&(p.clone(), q.clone())) {
        Ok(())
    } else {
        // Find distinguishing context
        let context = find_distinguishing_context(p, q, theory);
        Err(TypeError::NotBisimilar { context })
    }
}

// Compute bisimulation relation (fixed point)
fn compute_bisimulation(p: &Process, q: &Process, theory: &Theory) -> Relation {
    let mut relation = Relation::full(); // Start with everything related

    loop {
        let refined = simulation_step(&relation, theory);
        if refined == relation {
            break;
        }
        relation = refined;
    }

    relation
}
```

### Rholang Example

```rholang
// Assert equivalence
@bisimilar(optimized_server)
contract server(requests: Name[Request]) = {
  for(@req <- requests) {
    // Original implementation
    ...
  }
}

// Must be proven equivalent to server
contract optimized_server(requests: Name[Request]) = {
  for(@req <- requests) {
    // Optimized implementation
    ...
  }
}
```

---

## Resource Bounds

### Problem

Verify that a computation uses bounded resources (gas, memory, time).

### Type

From Meta-MeTTa "effort objects":

```
; Computation bounded by gas g
Bounded(g) := □(effort ≤ g)

; Where effort tracks resource consumption
```

### Gph-Theory Model

```rust
// Gas-aware reduction
struct GasReduction {
    rule: RewriteRule,
    cost: Gas,
}

impl GasReduction {
    fn apply(&self, term: &Term, available: Gas) -> Option<(Term, Gas)> {
        if available >= self.cost {
            if let Some(result) = self.rule.apply(term) {
                Some((result, available - self.cost))
            } else {
                None
            }
        } else {
            None // Out of gas
        }
    }
}
```

### Type-Level Gas Tracking

Extend the type system with gas bounds:

```
; Gas-bounded process type
T_g ::= T @ g     ; process of type T with gas bound g

; Typing rule for bounded evaluation
Γ ⊢ P : T @ g       cost(eval(P)) ≤ g
──────────────────────────────────────── [bounded-eval]
Γ ⊢ eval(P) : eval(T) @ (g - cost(eval(P)))
```

### MeTTa Example

```metta
; Bounded computation
!(with-gas 1000
  (factorial 10))

; Type: Bounded(1000) → Result
```

### Implementation

```rust
// Check resource bounds
fn check_bounded(term: &Term, gas: Gas, theory: &Theory) -> Result<(), TypeError> {
    let mut current = term.clone();
    let mut remaining = gas;

    loop {
        match theory.step_with_gas(&current, remaining) {
            StepResult::Done(result) => return Ok(()),
            StepResult::Step(next, cost) => {
                current = next;
                remaining -= cost;
            }
            StepResult::OutOfGas => {
                return Err(TypeError::ExceedsGasBound {
                    limit: gas,
                    at: current,
                });
            }
        }
    }
}
```

### Rholang Example

```rholang
// Gas-bounded contract
@gas(1000)
contract bounded_work(@input: Data, return: Name[Result]) = {
  // Must complete within 1000 gas units
  ...
}
```

---

## Contract Types

### Problem

Ensure that smart contracts satisfy their interface specifications.

### Type Specification

```
; Contract must handle all cases and return on return channel
Contract(Input, Output) :=
  ∀input: Input. ◇(Output on return channel)
```

### Interface Types Using Spatial-Behavioral System

```
; Interface as a collection of method types
Interface ::= { method₁ : MethodType₁, ..., methodₙ : MethodTypeₙ }

; Method type includes behavioral properties
MethodType ::= (InputTypes) → OutputType @ [Annotations]

; Example ERC20 interface
ERC20 = {
  balanceOf : (@Address) → Nat @ [total, pure],
  transfer  : (@Address, @Nat, Name[Bool]) → Unit @ [total],
  approve   : (@Address, @Nat, Name[Bool]) → Unit @ [total]
}
```

### Rholang Example

```rholang
// Full contract type specification
contract Token implements ERC20 {
  // Must implement all ERC20 methods
  @total  // Always returns
  @pure   // No side effects
  def balanceOf(@address: Address): Nat

  @total
  def transfer(@to: Address, @amount: Nat, return: Name[Bool]): Unit

  // ...
}
```

### Implementation

```rust
// Check contract implements interface
fn check_implements(contract: &Contract, interface: &Interface) -> Result<(), TypeError> {
    for method in &interface.methods {
        // Check method exists
        let impl_method = contract.find_method(&method.name)
            .ok_or(TypeError::MissingMethod(method.name.clone()))?;

        // Check type matches
        if impl_method.signature != method.signature {
            return Err(TypeError::SignatureMismatch {
                method: method.name.clone(),
                expected: method.signature.clone(),
                found: impl_method.signature.clone(),
            });
        }

        // Check behavioral properties
        for annotation in &method.annotations {
            check_annotation(impl_method, annotation)?;
        }
    }

    Ok(())
}
```

### Verification Properties

| Annotation | Meaning | Verification |
|------------|---------|--------------|
| `@total` | Always returns | Termination check |
| `@pure` | No side effects | Free name analysis |
| `@safe` | No exceptions | Exception-freedom check |
| `@responsive` | Eventually responds | Liveness check |

---

## Correction WFST Integration

### Semantic Validation in Tier 3

The spatial-behavioral type system integrates into the three-tier correction
architecture at Tier 3:

```
Tier 3: Semantic Type Checking
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Syntactically Valid Candidates (from Tier 2)                  │
│                     │                                           │
│                     ▼                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Type Inference Engine                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │  Ground     │  │  Process    │  │  Behavioral │     │   │
│  │  │  Types      │  │  Types      │  │  Types      │     │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │   │
│  │         └─────────────────┼─────────────────┘           │   │
│  │                           ▼                             │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  Type Judgment: Γ ⊢ candidate : T               │   │   │
│  │  └────────────────────────┬────────────────────────┘   │   │
│  └───────────────────────────┼─────────────────────────────┘   │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Semantically Valid Corrections              │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  candidate₁ : T₁  (score: 0.95)                 │    │   │
│  │  │  candidate₂ : T₂  (score: 0.87)                 │    │   │
│  │  │  ...                                             │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Type-Aware Correction Algorithm

```rust
/// Apply semantic type checking to filter correction candidates
fn semantic_filter(
    candidates: Vec<Candidate>,
    context: &TypeContext,
) -> Vec<TypedCandidate> {
    candidates
        .into_iter()
        .filter_map(|candidate| {
            // Attempt to type the candidate
            match infer_type(&candidate.term, context) {
                Ok(inferred_type) => {
                    // Check compatibility with expected type
                    if is_compatible(&inferred_type, &context.expected_type) {
                        Some(TypedCandidate {
                            candidate,
                            inferred_type,
                            confidence: compute_confidence(&inferred_type, context),
                        })
                    } else {
                        None // Type mismatch
                    }
                }
                Err(_) => None, // Type error
            }
        })
        .collect()
}
```

### Example: Code Correction with Types

```metta
; Original (with typo)
(= (factorial $n) (* $n (factoiral (- $n 1))))
                          ^^^^^^^^ typo

; Tier 1: liblevenshtein suggests
;   factoiral → factorial (edit distance 1)
;   factoiral → factorial (edit distance 1)
;   factoiral → factor (edit distance 3)

; Tier 2: All syntactically valid

; Tier 3: Type checking
;   factorial : (-> Nat Nat) ✓
;   factor    : (-> Nat List[Nat]) ✗ (type mismatch in context)

; Result: factorial selected
```

### Integration with PathMap

The type context can be stored in PathMap for efficient lookup:

```rust
impl TypeContext {
    /// Store type environment in PathMap
    fn to_pathmap(&self) -> PathMap {
        let mut pm = PathMap::new();
        for (name, typ) in &self.bindings {
            pm.insert(name.as_bytes(), typ.encode());
        }
        pm
    }

    /// Load type environment from PathMap
    fn from_pathmap(pm: &PathMap) -> Self {
        let mut ctx = TypeContext::new();
        for (name, typ_bytes) in pm.iter() {
            let name = String::from_utf8_lossy(name).to_string();
            let typ = Type::decode(typ_bytes);
            ctx.bindings.insert(name, typ);
        }
        ctx
    }
}
```

---

## Summary

These use cases demonstrate practical applications of semantic type checking:

| Use Case | Type Feature | Benefit |
|----------|--------------|---------|
| Query validation | Predicate types | Catch type errors early |
| Namespace security | Behavioral predicates | Enforce isolation |
| Termination | Reachability (◇) | Guarantee completion |
| Equivalence | Bisimulation | Verify optimizations |
| Resource bounds | Gas modeling | Predictable costs |
| Contract types | Interface specs | API compliance |
| Correction WFST | Type inference | Semantic filtering |

Each use case maps directly to OSLF constructs:
- **Predicates** for structural properties
- **Modalities** for behavioral properties
- **Fixed points** for recursive properties
- **Substitution** for pattern matching
- **Lifted redexes** for reduction reasoning

### Type Rules Summary

| Rule | Types | Application |
|------|-------|-------------|
| Ground types | Bool, String, Int | Literal values |
| Collections | List, Tuple, Set, Map | Compound data |
| For-comprehension | $`\langle(TT \leftarrow V)\rangle T`$ | Input/receive |
| Reflection  | $`\langle`$?$`V\rangleT`$ |  Queries |
| Parallel | T \| U | Concurrency |
| Quote/Deref | @T, *N | Reflection |
| Lifted redex | Comm, Eval, Par | Behavioral |

---

## References

- Williams, P. & Stay, M. "Native Type Theory." EPTCS 372, pp. 116-132, 2022.
- Meredith, L. G. et al. "rhocube.spatial-behavioral.types.tex" (Spatial-behavioral types)
- See [02-native-type-theory-oslf.md](../theoretical-foundations/02-native-type-theory-oslf.md) for type theory
- See [03-gph-enriched-lawvere.md](../theoretical-foundations/03-gph-enriched-lawvere.md) for operational semantics
- See [bibliography.md](./bibliography.md) for complete references
