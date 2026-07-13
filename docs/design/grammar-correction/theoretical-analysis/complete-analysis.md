# Theoretical Guarantees of Multi-Layer Error Correction Pipeline

[← Documentation Index](../../../README.md)

**Analysis Document v1.0**
**Date**: 2025-11-04
**Project**: liblevenshtein-rust
**Related**: grammar-correction.md

## Executive Summary

This document provides a rigorous theoretical analysis of the multi-layer error correction pipeline described in `grammar-correction.md`. We analyze three critical properties: **determinism**, **correctness**, and **optimality** for each layer and for the composed system.

**Key Findings**:
1. Determinism holds under specific tie-breaking rules but fails with feedback
2. Correctness is guaranteed for each layer individually but composition is subtle
3. Optimality fails for the sequential pipeline; joint optimization is intractable

**Recommendations**:
- Use deterministic tie-breaking for reproducibility
- Implement approximation guarantees via beam search
- Consider Pareto optimality for multi-objective scenarios
- Document complexity/quality tradeoffs explicitly

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Formal Framework](#2-formal-framework)
3. [Layer 1: Lexical Correction](#3-layer-1-lexical-correction)
4. [Layer 2: Grammar Correction](#4-layer-2-grammar-correction)
5. [Layer 3: Semantic Validation](#5-layer-3-semantic-validation)
6. [Layer 4: Semantic Repair](#6-layer-4-semantic-repair)
7. [Layer 5: Process Verification](#7-layer-5-process-verification)
8. [Compositional Analysis](#8-compositional-analysis)
9. [Practical Recommendations](#9-practical-recommendations)
10. [Open Problems](#10-open-problems)
11. [References](#11-references)

---

## 1. Introduction

### 1.1 Motivation

The multi-layer error correction pipeline composes five distinct correction mechanisms:

```
Raw Text → [Lexical] → [Grammar] → [Semantic Validation] → [Semantic Repair] → [Process Verification] → Corrected Program
```

Each layer operates independently with its own:
- **Input/Output**: Text, tokens, parse trees, ASTs, typed ASTs, session-typed processes
- **Error model**: Character edits, syntax errors, type errors, semantic bugs, protocol violations
- **Cost function**: Edit distance, parse errors, type mismatches, constraint violations, session type errors
- **Algorithm**: Levenshtein automata, BFS+parsing, type inference, SMT solving, session type checking

**Central Questions**:
1. **Determinism**: Does the pipeline produce the same output for the same input?
2. **Correctness**: Does each layer produce valid corrections?
3. **Optimality**: Does the pipeline find the minimum-cost correction overall?

These properties have profound implications for:
- **Usability**: Determinism ensures reproducibility
- **Trust**: Correctness guarantees syntactic/semantic validity
- **Quality**: Optimality provides best possible repairs

### 1.2 Definitions

**Definition 1.1 (Determinism)**:
A function f is **deterministic** if for all inputs x:
```
f(x) = y₁ and f(x) = y₂ ⟹ y₁ = y₂
```

For our pipeline, this means running the same input multiple times produces identical output.

**Definition 1.2 (Correctness)**:
A correction function f mapping inputs I to outputs O is **correct** if for all `$i \in  I$`:
```
f(i) ∈ Valid(I)
```

where Valid(I) is the set of valid programs (syntactically, semantically, or behaviorally correct depending on layer).

**Definition 1.3 (Optimality)**:
A correction function f with cost function c: `$I \times  O \to  \mathbb{R}$` is **optimal** if for all `$i \in  I$`:
```
f(i) = argmin_{o ∈ Valid(I)} c(i, o)
```

For multi-objective problems (multiple cost functions), we need **Pareto optimality**:

**Definition 1.4 (Pareto Optimality)**:
Solution x is **Pareto optimal** if there is no solution y such that:
- y is better than x in at least one objective
- y is at least as good as x in all other objectives

---

## 2. Formal Framework

### 2.1 Pipeline Structure

**Definition 2.1 (Layer)**:
A layer L is a tuple (I_L, O_L, c_L, Alg_L) where:
- I_L: Input domain
- `$O_L$`: Output domain (`$O_L \subseteq I_{L+1}$` for sequential composition)
- `$c_L$`: `$I_L \times O_L \to \mathbb{R}_+$` (cost function)
- Alg_L: I_L → 2^{O_L} (algorithm returning candidate set)

**Definition 2.2 (Sequential Composition)**:
Layers L₁ and L₂ compose sequentially as `$L_{2} \circ  L_{1}$`:
```
(L₂ ∘ L₁)(x) = { y ∈ O_L₂ | ∃z ∈ L₁(x), y ∈ L₂(z) }
```

**Definition 2.3 (Total Cost)**:
For pipeline `$L_n \circ  ... \circ  L_{1},$` the total cost is:
```
C_total(x, y) = ∑ᵢ₌₁ⁿ c_Lᵢ(xᵢ₋₁, xᵢ)
```
where `x = x₀`, `$x_{1} \in  L_{1}(x_{0})$`, ..., `$x_{n} = y \in  L_{n}(x_{n-1})$`.

### 2.2 Pipeline Instantiation

**Layer 1: Lexical Correction**
- `$I_{1} = \Sigma *$` (strings over alphabet `$\Sigma$`)
- `$O_{1} = (\Sigma *)^k$` (`k` token candidates per position)
- `c₁(s, t) = Levenshtein_distance(s, t)`
- `Alg₁` = Levenshtein automaton query

**Layer 2: Grammar Correction**
- `$I_{2} = (\Sigma *)^m$` (token sequences)
- `O₂ = ParseTree(G)` (valid parse trees for grammar `G`)
- `c₂(tokens, tree) = #syntax_repairs`
- `Alg₂` = BFS over parse states + Tree-sitter

**Layer 3: Semantic Validation**
- `I₃ = ParseTree(G)`
- `$O_{3} = {\text{tree} \in  \text{ParseTree}(G) \lvert  \text{well}_\text{typed}(\text{tree})}$`
- `c₃(tree, tree') = 0` if `well_typed(tree')`, `$\infty$` otherwise
- `Alg₃` = Type inference (Algorithm W) + filter

**Layer 4: Semantic Repair**
- `I₄ = ParseTree(G)` with type errors
- `$O_{4} = {\text{tree} \in  \text{ParseTree}(G) \lvert  \text{well}_\text{typed}(\text{tree})}$`
- `c₄(tree, tree') = #semantic_edits`
- `Alg₄` = Template-based / SMT-based / Search-based

**Layer 5: Process Verification**
- `I₅ = TypedProgram` (for process calculi)
- `$O_{5} = {\text{prog} \lvert  \text{session}_\text{type}_\text{check}(\text{prog})}$`
- `c₅(prog, prog') = #protocol_violations_fixed`
- `Alg₅` = Session type checker + repair

### 2.3 Theoretical Tools

We employ the following theoretical frameworks:

**2.3.1 Automata Theory**
- Finite automata for lexical analysis
- Pushdown automata for parsing
- Characterize decidability and complexity

**2.3.2 Type Theory**
- Simply-typed lambda calculus (STLC)
- Hindley-Milner type inference
- Progress and preservation theorems

**2.3.3 Complexity Theory**
- Time complexity: `$\mathcal{O}(f(n))$`
- Space complexity
- Decidability (halting problem, type inference, etc.)

**2.3.4 Optimization Theory**
- Dynamic programming
- Greedy algorithms
- Approximation algorithms

---

## 3. Layer 1: Lexical Correction

### 3.1 Algorithm Description

Layer 1 uses **Universal Levenshtein Automata** to find dictionary words within edit distance d of input tokens.

**Algorithm**:
```
Input: Token w, dictionary D, max distance d
Output: Set of (candidate, distance) pairs

1. Construct Levenshtein automaton A_w,d
2. For each word v ∈ D:
3.   If A_w,d accepts v:
4.     distance = Levenshtein(w, v)
5.     Yield (v, distance)
6. Return top-k candidates sorted by distance
```

### 3.2 Determinism Analysis

**Theorem 3.1 (Levenshtein Automata Determinism)**:
The Levenshtein automaton construction is deterministic: for fixed word w and distance d, the automaton A_w,d is unique.

**Proof Sketch**:
- Construction is algorithmic (Schulz-Mihov algorithm)
- No random choices in state transitions
- DFA minimization (if applied) is unique
`$\blacksquare$`

**Corollary 3.2 (Query Determinism with Tie-Breaking)**:
Lexical correction is deterministic if:
1. Dictionary D is finite and fixed
2. Tie-breaking rule is deterministic (e.g., lexicographic order)

**Counter-Example 3.3 (Non-Determinism Without Tie-Breaking)**:
```
Input: "teh", d=1, k=1 (return top-1)
Dictionary: {tea, ten, the}

All have distance 1 from "teh"
Without tie-breaking: could return any of {tea, ten, the}
```

**Resolution**: Use lexicographic ordering as tie-breaker.

### 3.3 Correctness Analysis

**Theorem 3.4 (Levenshtein Correctness)**:
The Levenshtein automaton correctly recognizes all and only strings within distance d:
```
L(A_w,d) = {v ∈ Σ* | Levenshtein(w, v) ≤ d}
```

**Proof**: By construction of automaton states and transitions [Schulz & Mihov 2002].
`$\blacksquare$`

**Corollary 3.5 (Candidate Validity)**:
All returned candidates are valid dictionary words within the specified distance.

### 3.4 Optimality Analysis

**Theorem 3.6 (Top-k Optimality)**:
If candidates are sorted by Levenshtein distance and ties broken deterministically, the algorithm returns the k closest dictionary words.

**Proof**:
- Levenshtein distance is computed exactly (Wagner-Fischer)
- Sorting ensures minimum distances are returned first
- k smallest distances are optimal
`$\blacksquare$`

**Limitations**:
- Optimality is **per-token**, not per-sentence
- No context consideration (next layers address this)

### 3.5 Complexity Analysis

**Theorem 3.7 (Lexical Complexity)**:
For word `w` with `$\lvert w\rvert = n$`, dictionary of size `m`, max distance `d`:
- **Automaton construction**: `$\mathcal{O}(n \times  d)$` states, `$\mathcal{O}(n \times  d \times  \lvert \Sigma \rvert)$` transitions
- **Query per word**: `$\mathcal{O}(\text{length} \times  d)$` amortized
- **Total query**: `$\mathcal{O}(m \times  \text{avg}_\text{length} \times  d)$`
- **Space**: `$\mathcal{O}(n \times  d)$`

**Practical Bounds**:
- Typical: `$n \approx  10, d = 2, m = 100k,$` avg_length = 7
- Construction: ~20 states, ~1000 transitions → negligible
- Query: ~14M character comparisons → ~10ms on modern CPU

### 3.6 Summary: Layer 1 Properties

| Property | Holds? | Conditions |
|----------|--------|------------|
| **Determinism** | ✓ Yes | With deterministic tie-breaking |
| **Correctness** | ✓ Yes | All candidates within distance d |
| **Optimality** | ✓ Yes | Top-k by distance (per-token) |
| **Decidability** | ✓ Yes | Finite automaton, regular language |
| **Complexity** | `$\mathcal{O}(n \times  d)$` | Linear in word length and distance |

---

## 4. Layer 2: Grammar Correction

### 4.1 Algorithm Description

Layer 2 uses **BFS over parse states** with Tree-sitter to find syntactically valid parse trees.

**Algorithm (from grammar-correction.md)**:
```
Input: Token sequence T, grammar G, max distance d
Output: Valid parse trees within distance d

1. Initialize: queue = [(initial_parse_state, [], 0)]
2. While queue not empty:
3.   (state, edits, cost) = queue.pop()
4.   If cost > d: continue
5.   If state is complete parse:
6.     Yield (parse_tree(state), cost)
7.   For each valid next symbol s (from LookaheadIterator):
8.     state' = advance_parse(state, s)
9.     If s ∈ T: cost' = cost (match)
10.    Else: cost' = cost + 1 (insert/delete/substitute)
11.    queue.append((state', edits + [s], cost'))
```

**Variant: Beam Search**
- Keep only top-k states by heuristic score
- Trade optimality for tractability

### 4.2 Determinism Analysis

**Theorem 4.1 (BFS Non-Determinism)**:
Pure BFS grammar correction is **non-deterministic** when multiple parse trees have the same cost.

**Counter-Example 4.2**:
```python
Grammar (ambiguous):
  Expr → Expr + Expr | Expr * Expr | number

Input (with error): "1 + 2 3"  (missing operator)
Correction distance: 1

Two valid corrections (both cost 1):
1. Insert '+': "1 + 2 + 3"  →  Parse tree: ((1 + 2) + 3)
2. Insert '*': "1 + 2 * 3"  →  Parse tree: (1 + (2 * 3))

BFS may return either, depending on queue order.
```

**Proposition 4.3 (Determinism via Ordering)**:
Grammar correction becomes deterministic if:
1. Grammar is unambiguous
2. Tie-breaking rule is deterministic (e.g., prefer earlier position, lexicographic)
3. BFS exploration order is deterministic (FIFO queue)

**Proof**: With fixed exploration order and deterministic tie-breaking, the first parse tree found at distance d is always the same.
`$\blacksquare$`

**Corollary 4.4 (Beam Search Non-Determinism)**:
Beam search is non-deterministic unless beam selection is deterministic (sorted by score, then tie-broken).

### 4.3 Correctness Analysis

**Definition 4.5 (Syntactic Correctness)**:
Parse tree T is syntactically correct for grammar G if `$T \in$` ParseTree(G).

**Theorem 4.6 (Grammar Correction Correctness)**:
All parse trees returned by BFS grammar correction are syntactically valid.

**Proof**:
- BFS only advances parse state using LookaheadIterator
- LookaheadIterator returns only symbols that lead to valid parse states
- Tree-sitter guarantees parse validity
- Complete parse states correspond to valid parse trees
`$\blacksquare$`

**Caveat**: Error recovery in Tree-sitter may produce parse trees with error nodes. For true correctness, filter out trees with error nodes.

**Theorem 4.7 (Bounded Correctness)**:
For max distance d, all returned parse trees are within d edits of the input.

**Proof**: By BFS termination condition (cost > d → prune).
`$\blacksquare$`

### 4.4 Optimality Analysis

**Question**: Is BFS optimal? Does it find the minimum-cost correction?

**Theorem 4.8 (BFS Optimality for Uniform Cost)**:
If all edits have cost 1, BFS finds the minimum-distance parse tree.

**Proof**:
- BFS explores states in order of increasing distance
- First complete parse at distance d is guaranteed to be optimal
- No parse tree at distance < d exists (would have been found earlier)
`$\blacksquare$`

**Theorem 4.9 (BFS Non-Optimality for Non-Uniform Cost)**:
For non-uniform edit costs, BFS does **not** guarantee optimality.

**Counter-Example 4.10**:
```
Edit costs: Insert=1, Delete=1, Substitute=2
Input: "x = y" with syntax error
Grammar: requires "x := y" (Pascal-style)

BFS explores:
1. Distance 0: No valid parse
2. Distance 1: Try insert ':', no valid parse yet
3. Distance 2: Substitute '=' → ':=' (cost 2) → VALID ✓

But optimal is:
- Delete '=', insert ':=' (cost 1 + 1 = 2) → same cost
- OR: Insert ':' then delete '=' (cost 1 + 1 = 2) → same cost

BFS may explore these in different order.
```

**Corollary 4.11**: Use **Dijkstra's algorithm** (priority queue by cost) instead of BFS for non-uniform costs.

**Theorem 4.12 (Beam Search Non-Optimality)**:
Beam search does **not** guarantee optimality even for uniform costs.

**Counter-Example 4.13**:
```
Beam width k=2
Initial state spawns 3 children with costs [1, 1, 1]
Beam keeps 2 (arbitrary choice)
Later, the pruned branch leads to optimal solution at distance 2
Kept branches lead to solution at distance 3
Result: Beam search returns distance 3 (suboptimal)
```

**Theorem 4.14 (Beam Search Approximation)**:
Beam search with beam width k is a **k-approximation** algorithm in certain cases.

**Proof Sketch**:
- At each level, keep top-k by heuristic score
- If heuristic is admissible (never overestimates), beam search is guided toward optimal
- Approximation ratio depends on heuristic quality
`$\blacksquare$`

**Practical Heuristics**:
1. **A* heuristic**: f(n) = g(n) + h(n) where:
   - g(n) = cost so far
   - h(n) = estimated cost to completion (e.g., unparsed tokens)
2. **Language model score**: Prefer likely continuations
3. **Semantic feedback**: Prefer corrections that lead to well-typed programs

### 4.5 Complexity Analysis

**Theorem 4.15 (BFS Grammar Complexity)**:
For input length n, max distance d, beam width k, parse time p:
- **Pure BFS**: `$\mathcal{O}(\lvert \text{Grammar}\rvert^d \times  n \times  p)$` worst-case (exponential in d)
- **Beam BFS**: `$\mathcal{O}(k \times  d \times  n \times  p)$` (tractable)
- **Space**: `$\mathcal{O}(k \times  \text{state}_\text{size})$`

**Practical Bounds**:
- n = 100 tokens, d = 2, k = 20, p `$\approx$` 5ms (Tree-sitter)
- Beam BFS: ~20 × 2 × 100 × 5ms = 200ms (acceptable for IDE)

**Theorem 4.16 (Parsing Decidability)**:
Context-free grammar parsing is decidable in `$\mathcal{O}(n^{3})$` worst-case (CYK, Earley).

**Proof**: Chart parsing algorithms [Earley 1970, Younger 1967].
`$\blacksquare$`

**Corollary 4.17**:
Grammar correction is decidable but potentially expensive. Beam search is necessary for real-time use.

### 4.6 Summary: Layer 2 Properties

| Property | Holds? | Conditions |
|----------|--------|------------|
| **Determinism** | ✗ No (partial) | Requires unambiguous grammar + deterministic tie-breaking |
| **Correctness** | ✓ Yes | All parse trees are syntactically valid |
| **Optimality** | ✗ No (BFS) | Only for uniform costs; beam search is approximate |
| **Decidability** | ✓ Yes | CFG parsing is decidable |
| **Complexity** | `$\mathcal{O}(k \times  d \times  n \times  p)$` | Beam search; exponential without beam |

**Key Insight**: **Optimality is sacrificed for tractability**. Beam search provides approximation guarantees.

---

## 5. Layer 3: Semantic Validation

### 5.1 Algorithm Description

Layer 3 filters parse trees by **type correctness** using Hindley-Milner type inference.

**Algorithm**:
```
Input: Parse trees from Layer 2
Output: Well-typed parse trees

1. For each parse tree T:
2.   AST = parse_tree_to_ast(T)
3.   Try:
4.     τ = type_infer(AST)  // Algorithm W
5.     If no type error:
6.       Yield (T, τ)
7.   Except TypeError:
8.     Reject T
```

### 5.2 Determinism Analysis

**Theorem 5.1 (Type Inference Determinism)**:
Algorithm W (Hindley-Milner type inference) is deterministic: for a given AST, it produces a unique principal type (up to α-renaming).

**Proof**:
- Algorithm W is a deterministic algorithm
- Unification (Robinson's algorithm) is deterministic
- Principal type is unique (Hindley-Milner theorem)
`$\blacksquare$`

**Caveat**: Type inference may involve generating fresh type variables. If fresh variable generation is non-deterministic (e.g., based on system state), types may differ syntactically but are equivalent up to renaming.

**Resolution**: Use deterministic fresh variable generation (e.g., counter-based).

**Corollary 5.2 (Validation Determinism)**:
Semantic validation is deterministic if type inference is deterministic.

### 5.3 Correctness Analysis

**Definition 5.3 (Semantic Correctness - Type Level)**:
Program P is semantically correct (at type level) if `$\emptyset  \vdash  P$` : `$\tau$` for some type `$\tau .$`

**Theorem 5.4 (Validation Correctness)**:
All programs passing Layer 3 are well-typed.

**Proof**: By construction; Algorithm W only succeeds if type derivation exists.
`$\blacksquare$`

**Theorem 5.5 (Type Safety)**:
Well-typed programs don't go wrong (Progress + Preservation).

**Proof**: Fundamental theorems of type theory [Wright & Felleisen 1994].
`$\blacksquare$`

**Implications**: Layer 3 guarantees absence of type errors (but not absence of all bugs).

### 5.4 Optimality Analysis

**Observation 5.6**:
Layer 3 is a **filter**, not an optimizer. It does not modify programs, only accepts/rejects.

**Definition 5.7 (Filtering Optimality)**:
A filter is optimal if it accepts all and only valid programs.

**Theorem 5.8 (Validation Optimality)**:
Layer 3 is optimal: it accepts exactly the well-typed programs.

**Proof**:
- Completeness: All well-typed programs are accepted (Algorithm W completeness)
- Soundness: All accepted programs are well-typed (Algorithm W soundness)
`$\blacksquare$`

**Caveat**: Optimality is with respect to **type correctness**, not overall error correction. A well-typed program may still be semantically wrong (e.g., wrong logic).

### 5.5 Complexity Analysis

**Theorem 5.9 (Type Inference Complexity)**:
Algorithm W for Hindley-Milner has:
- **Best/Average case**: `$\mathcal{O}(n \log  n)$` for program size n
- **Worst case**: Exponential (due to occurs check)
- **Practical**: Near-linear for typical programs

**Proof**: Empirical analysis [McBride 2003, Heeren 2005].
`$\blacksquare$`

**Theorem 5.10 (Type Inference Decidability)**:
Hindley-Milner type inference is decidable.

**Proof**: Algorithm W always terminates [Damas & Milner 1982].
`$\blacksquare$`

**Extension: System F**:
- System F (polymorphic λ-calculus with explicit type abstraction) has undecidable type inference
- But Hindley-Milner (implicit polymorphism) is decidable

### 5.6 Summary: Layer 3 Properties

| Property | Holds? | Conditions |
|----------|--------|------------|
| **Determinism** | ✓ Yes | With deterministic fresh variable generation |
| **Correctness** | ✓ Yes | Only well-typed programs pass |
| **Optimality** | ✓ Yes | Accepts all and only well-typed programs |
| **Decidability** | ✓ Yes | Hindley-Milner is decidable |
| **Complexity** | `$\mathcal{O}(n \log  n)$` avg | Near-linear for typical programs |

**Key Insight**: Layer 3 is a **perfect filter** for type correctness.

---

## 6. Layer 4: Semantic Repair

### 6.1 Algorithm Description

Layer 4 **repairs** type errors using multiple strategies:

1. **Template-based**: Apply predefined fix patterns
2. **SMT-based**: Solve constraints to find repairs (MaxSMT)
3. **Search-based**: Explore program edits to restore type correctness

**Algorithm (SMT-based approach)**:
```
Input: AST with type error, constraint set C
Output: Repaired AST

1. Localize error: Find minimal unsatisfiable constraint subset
2. Generate repair constraints: C' = C \ error_constraints + repair_constraints
3. Solve: model = MaxSMT(C')
4. Extract repair: AST' = apply_model(AST, model)
5. Verify: Check AST' is well-typed
6. Return AST'
```

### 6.2 Determinism Analysis

**Theorem 6.1 (Template Repair Determinism)**:
Template-based repair is deterministic if:
1. Template matching is deterministic
2. Template selection (if multiple match) is deterministic

**Counter-Example 6.2 (SMT Non-Determinism)**:
```python
Type error:
  x: Int
  y: String
  z = x + y  # Type mismatch

Repair constraints:
  C1: Change x to String → z: String
  C2: Change y to Int → z: Int

Both satisfy constraints (MaxSMT may return either).
```

**Proposition 6.3 (SMT Determinism via Preferences)**:
SMT repair becomes deterministic with:
1. Preference ordering on constraints (soft constraints with weights)
2. Deterministic SMT solver (reproducible random seed)

**Practical Issue**: Most SMT solvers have non-deterministic heuristics. Z3 provides `smt.random_seed` for reproducibility.

### 6.3 Correctness Analysis

**Definition 6.4 (Repair Correctness)**:
Repair is correct if the repaired program is well-typed.

**Theorem 6.5 (Template Correctness)**:
Template-based repair is correct if templates are manually verified.

**Proof**: If template T: Error → Valid is verified, applying T produces valid program.
`$\blacksquare$`

**Theorem 6.6 (SMT Correctness)**:
SMT-based repair is correct if:
1. Constraint encoding is sound
2. Repaired program is verified (Algorithm W check)

**Proof**:
- SMT model satisfies constraints
- Constraints encode type correctness
- Verification confirms well-typedness
`$\blacksquare$`

**Caveat**: SMT repair may produce **semantically wrong** programs (e.g., changes variable that should not be changed).

**Example 6.7 (Incorrect Semantic Repair)**:
```python
Original (intended):
  balance: Int = 100
  withdraw(amount: Int):
    balance = balance - amount

Buggy (type error):
  balance: Int = 100
  withdraw(amount: String):  # Wrong type
    balance = balance - amount

SMT repair (type-correct but semantically wrong):
  balance: String = "100"  # Changed wrong variable!
  withdraw(amount: String):
    balance = balance - amount  # Now type-correct but wrong semantics
```

**Resolution**: Use semantic constraints (e.g., "balance must be Int") in SMT encoding.

### 6.4 Optimality Analysis

**Question**: Does SMT repair find the minimum-cost repair?

**Theorem 6.8 (MaxSMT Optimality)**:
MaxSMT finds the optimal solution for the **given constraint set** (maximizes satisfied soft constraints).

**Proof**: By definition of MaxSMT optimization [Nieuwenhuis & Oliveras 2006].
`$\blacksquare$`

**Limitation**: Optimality is relative to the **constraint encoding**, not the true semantic intent.

**Theorem 6.9 (Repair Optimality is Undecidable)**:
There is no algorithm that, for arbitrary programs and type errors, computes the semantically optimal repair.

**Proof Sketch**:
- Semantic intent is not formally specified
- Even with formal specs, equivalence checking is undecidable (Rice's theorem)
`$\blacksquare$`

**Corollary 6.10**: We can only approximate semantic optimality via heuristics (e.g., prefer smaller edits, preserve variable roles).

**Theorem 6.11 (MaxSMT Complexity)**:
MaxSMT is **NP-hard** (even FNP-hard for finding optimal solution).

**Proof**: Reduction from weighted MAX-SAT [Garey & Johnson 1979].
`$\blacksquare$`

**Practical Approaches**:
1. **Timeout**: Run SMT solver for bounded time
2. **Approximation**: Use greedy heuristics for constraint relaxation
3. **User feedback**: Present multiple repairs for user selection

### 6.5 Complexity Analysis

**Theorem 6.12 (Repair Complexity)**:
- **Template-based**: `$\mathcal{O}(\text{template}_\text{count} \times  \text{pattern}_\text{match}_\text{cost})$` — polynomial
- **SMT-based**: NP-hard in general, but modern solvers are efficient for typical instances
- **Search-based**: Exponential in search depth

**Practical Bounds**:
- Templates: Fast (milliseconds) but limited coverage
- SMT: Moderate (seconds) with good coverage
- Search: Slow (minutes) but most flexible

### 6.6 Summary: Layer 4 Properties

| Property | Holds? | Conditions |
|----------|--------|------------|
| **Determinism** | ✗ No (partial) | Requires deterministic solver + preferences |
| **Correctness** | ✓ Yes (syntactic) | Repairs are well-typed (verified) |
| **Correctness** | ✗ No (semantic) | May change wrong variables |
| **Optimality** | ✗ No (semantic) | Undecidable for true semantic optimality |
| **Optimality** | ✓ Yes (constraint) | MaxSMT optimal for given constraints |
| **Decidability** | ✗ No (general) | MaxSMT is NP-hard |
| **Complexity** | NP-hard (SMT) | Practical for typical cases with timeouts |

**Key Insight**: **Semantic repair trades optimality and decidability for practical usefulness**. Heuristics and user feedback are essential.

---

## 7. Layer 5: Process Verification

### 7.1 Algorithm Description

Layer 5 checks **session types** for process calculi (e.g., Rholang based on ρ-calculus).

**Session Types**: Behavioral types that specify communication protocols.

**Example**:
```
Session type: !Int.?String.end
Meaning: Send Int, receive String, terminate

Process:
  channel!42.channel?response.0

Check: Does process conform to session type?
```

**Algorithm**:
```
Input: Process P, session type S
Output: Accept/Reject + violations

1. Extract communication traces from P
2. Check trace ⊆ language(S)
3. Check deadlock-freedom (duality of parallel sessions)
4. Return violations (if any)
```

### 7.2 Determinism Analysis

**Theorem 7.1 (Session Type Checking Determinism)**:
Session type checking is deterministic for deterministic processes.

**Proof**:
- Type checking algorithm is deterministic
- Process semantics are deterministic (for π-calculus)
`$\blacksquare$`

**Caveat**: ρ-calculus (Rholang) has non-deterministic choice (e.g., parallel composition with race conditions). Session type checking must account for all possible interleavings.

**Corollary 7.2 (Non-Deterministic Process Checking)**:
For non-deterministic processes, session type checking considers all traces. The **result** (accept/reject) is deterministic, but the **trace** is not.

### 7.3 Correctness Analysis

**Definition 7.3 (Session Type Correctness)**:
Process P is session-correct if all communication traces conform to the session type.

**Theorem 7.4 (Session Type Safety)**:
If P has session type S and type checking succeeds:
1. **Communication safety**: No type mismatch in messages
2. **Deadlock freedom**: No circular waits (if types are dual)

**Proof**: Fundamental theorem of session types [Honda et al. 1998].
`$\blacksquare$`

**Theorem 7.5 (Progress for Session Types)**:
Well-typed processes with dual session types make progress (no deadlock).

**Proof**: By duality, sends are matched with receives [Wadler 2012].
`$\blacksquare$`

### 7.4 Optimality Analysis

**Observation 7.6**:
Layer 5 is a **verifier**, not an optimizer. It checks correctness but does not
suggest repairs; repair synthesis is outside the Layer 5 contract.

**Theorem 7.7 (Verification Completeness)**:
Session type checking is complete for the session type system: all well-typed processes are accepted.

**Proof**: By soundness and completeness of session type system [Honda et al. 1998].
`$\blacksquare$`

### 7.5 Complexity Analysis

**Theorem 7.8 (Session Type Checking Complexity)**:
For process size n:
- **Linear session types**: `$\mathcal{O}(n)$` (single channel, sequential)
- **Non-linear session types**: `$\mathcal{O}(n^k)$` for k parallel sessions
- **General π-calculus**: Undecidable (name mobility leads to unbounded behavior)

**Proof**:
- Linear case: Traverse process once
- Non-linear: Track channel interactions (polynomial)
- General: Reduction from halting problem
`$\blacksquare$`

**Practical Approach**:
- Restrict to linear session types or bounded non-linearity
- Use static analysis + runtime checks

### 7.6 Decidability

**Theorem 7.9 (Decidability of Session Types)**:
Session type checking is decidable for **finite session types** and **bounded processes**.

**Proof**: Finite session types have regular structure; checking reduces to finite state verification.
`$\blacksquare$`

**Theorem 7.10 (Undecidability for General π-Calculus)**:
Behavioral equivalence checking for π-calculus is undecidable.

**Proof**: Reduction from halting problem [Palamidessi 2003].
`$\blacksquare$`

**Implication**: Layer 5 is decidable for Rholang **if session types are finite and process behavior is bounded**.

### 7.7 Summary: Layer 5 Properties

| Property | Holds? | Conditions |
|----------|--------|------------|
| **Determinism** | ✓ Yes | For deterministic processes; result is deterministic even for non-deterministic processes |
| **Correctness** | ✓ Yes | Session type safety guarantees |
| **Optimality** | N/A | Verification layer, not optimization |
| **Decidability** | ✓ Yes (restricted) | For finite session types and bounded processes |
| **Complexity** | `$\mathcal{O}(n)$` to `$\mathcal{O}(n^k)$` | Depends on session type complexity |

**Key Insight**: Layer 5 provides **strong correctness guarantees** for concurrent communication but requires **restrictions** for decidability.

---

## 8. Compositional Analysis

### 8.1 Sequential Composition

**Question**: If each layer is correct, is the composed pipeline correct?

**Theorem 8.1 (Compositional Correctness)**:
If each layer L_i is correct (produces valid outputs), the composition `$L_n \circ  ... \circ  L_1$` is correct.

**Proof**:
- L_1 produces valid tokens → input to L_2 is valid
- L_2 produces valid parse trees → input to L_3 is valid
- L_3 filters for well-typed trees → input to L_4 is valid
- L_4 repairs to well-typed programs → input to L_5 is valid
- L_5 verifies session types → output is session-correct
`$\blacksquare$`

**Caveat**: This assumes **no errors propagate**. If L_i produces output with hidden errors, later layers may not detect them.

### 8.2 Compositional Optimality

**Question**: If each layer is optimal, is the composition optimal?

**Theorem 8.2 (Non-Compositional Optimality)**:
Optimal layers do **not** compose to optimal pipeline.

**Counter-Example 8.3**:
```
Input: "prnt(x + y" (missing closing parenthesis, typo in print)

Layer 1 (Lexical):
  Optimal: "prnt" → "print" (cost 1)

Layer 2 (Grammar):
  Input: "print(x + y"
  Optimal: Insert ')' (cost 1)
  Output: "print(x + y)"
  Total cost: 1 (lexical) + 1 (grammar) = 2

But joint optimization:
  Alternative: "prnt(x + y" → "print(x + y)" (cost 2)
    - Fix typo: cost 1
    - Insert ')': cost 1
    - Total: 2 (same)

  Or: "prnt(x + y" → "prnt(x + y)" (keep typo, just fix grammar)
    - Insert ')': cost 1
    - If "prnt" is a user-defined function, this is valid!
    - Total: 1 (better!)

Sequential composition chose "print" prematurely.
```

**Theorem 8.4 (Greedy Suboptimality)**:
Sequential (greedy) composition of optimal layers is suboptimal for the global objective.

**Proof**: By counter-example above. Sequential composition commits to Layer 1 choices before seeing Layer 2 constraints.
`$\blacksquare$`

### 8.3 Joint Optimization

**Definition 8.5 (Joint Optimization)**:
Find (x₁, ..., xₙ) that minimizes C_total `$= \sum _{i} c_L_{i}(x_{i-1}, x_{i})$` subject to validity constraints at each layer.

**Theorem 8.6 (Joint Optimization Intractability)**:
Joint optimization over all layers is **intractable** (exponential search space).

**Proof**:
- Layer 1: `k^n` candidates (`k` per token, `n` tokens)
- Layer 2: ~`$\lvert \text{Grammar}\rvert^d$` parse trees per candidate
- Layers 3-5: Combinatorial explosion
- Total: Exponential in pipeline length
`$\blacksquare$`

**Practical Approximations**:

**8.3.1 Dynamic Programming (DP)**:
If layers have **optimal substructure**, DP can compute global optimum.

**Limitation**: Layers 4-5 do not have clean substructure (SMT repair, session types are global).

**8.3.2 Beam Search with Lookahead**:
Keep top-k candidates at each layer, score with heuristic considering downstream layers.

**Algorithm**:
```
1. Layer 1: Generate top-k lexical candidates per token
2. For each candidate sequence:
3.   Layer 2: Generate top-k parse trees
4.   For each parse tree:
5.     Try type check (Layer 3)
6.     If fails, estimate repair cost (Layer 4 heuristic)
7.     Estimate session type issues (Layer 5 heuristic)
8. Sort all paths by total estimated cost
9. Return top-1 path
```

**Theorem 8.7 (Beam + Lookahead Approximation)**:
Beam search with lookahead provides better approximation than sequential greedy, but no worst-case guarantee.

**Proof**: Lookahead avoids premature commitment; beam width controls quality/performance tradeoff. No formal bound without specific assumptions.
`$\blacksquare$`

### 8.4 Pareto Optimality

**Observation**: Different layers optimize **different objectives**:
- Layer 1: Lexical similarity
- Layer 2: Syntax errors
- Layer 3: Type correctness (Boolean)
- Layer 4: Semantic edits
- Layer 5: Protocol violations

**Definition 8.8 (Pareto Optimal Solution)**:
Solution x is Pareto optimal if no solution y exists such that:
- y is strictly better than x in at least one objective
- y is at least as good as x in all objectives

**Theorem 8.9 (Pareto Frontier)**:
The set of Pareto optimal solutions forms a **Pareto frontier** in objective space.

**Practical Approach**:
1. Compute multiple candidate corrections
2. Identify Pareto frontier
3. Present to user for selection (user provides utility function)

**Algorithm**:
```
1. Generate candidates via beam search
2. For each candidate c:
3.   Compute multi-objective cost: (c_lex, c_syn, c_sem, c_proc)
4. Filter to Pareto frontier:
5.   For each pair (c1, c2):
6.     If c1 dominates c2: Remove c2
7. Return Pareto set
```

### 8.5 Feedback Mechanism

**Design (from grammar-correction.md)**:
Layers 3-5 provide feedback to Layers 1-2:
- Update weights based on which lexical/grammar choices lead to well-typed programs
- Learn from correction history

**Analysis**:

**Theorem 8.10 (Feedback Breaks Determinism)**:
Feedback mechanism breaks determinism unless feedback updates are deterministic.

**Proof**:
- Feedback updates weights based on past corrections
- If past corrections depend on user choices (non-deterministic), feedback is non-deterministic
`$\blacksquare$`

**Theorem 8.11 (Feedback May Improve Approximation)**:
Feedback can improve approximation by biasing toward semantically valid corrections.

**Proof Sketch**:
- Statistical learning: Frequent patterns guide search
- If feedback is based on correct resolutions, it approximates optimal prior
`$\blacksquare$`

**Practical Implementation**:
- Use machine learning (e.g., ranking model) to learn weights
- Train on corpus of corrected programs
- Update incrementally

### 8.6 Summary: Compositional Properties

| Property | Holds? | Explanation |
|----------|--------|-------------|
| **Compositional Correctness** | ✓ Yes | Valid outputs at each layer → valid final output |
| **Compositional Optimality** | ✗ No | Greedy layer-wise optimization is globally suboptimal |
| **Joint Optimization** | Intractable | Exponential search space |
| **Beam + Lookahead** | Approximate | Better than greedy, no formal guarantee |
| **Pareto Optimality** | Feasible | Multiple objectives → Pareto frontier |
| **Feedback Determinism** | ✗ No | Unless feedback updates are deterministic |
| **Feedback Benefit** | ✓ Yes (empirical) | Improves approximation via learning |

**Key Insight**: **Global optimality is intractable; practical systems must use approximations** (beam search, Pareto, feedback).

---

## 9. Practical Recommendations

Based on the theoretical analysis, we provide concrete recommendations for implementation.

### 9.1 Determinism

**Goal**: Ensure reproducibility for testing and debugging.

**Recommendations**:
1. **Lexical (Layer 1)**: Use lexicographic tie-breaking
2. **Grammar (Layer 2)**:
   - Use deterministic BFS queue (FIFO, sorted by state ID)
   - Break ties by preferring earlier error positions
3. **Semantic (Layers 3-4)**:
   - Use deterministic fresh variable generation (counter-based)
   - Use deterministic SMT solver (set random seed)
4. **Process (Layer 5)**: Deterministic by nature
5. **Feedback**:
   - Option A: Disable feedback for deterministic mode
   - Option B: Deterministic feedback (fixed training data, no online learning)

**Testing**:
```rust
#[test]
fn test_determinism() {
    let input = "prnt(x + y";
    let result1 = pipeline.correct(input);
    let result2 = pipeline.correct(input);
    assert_eq!(result1, result2, "Pipeline must be deterministic");
}
```

### 9.2 Correctness

**Goal**: Guarantee validity at each layer.

**Recommendations**:
1. **Lexical**: Use validated dictionary (no invalid entries)
2. **Grammar**: Filter out parse trees with error nodes
3. **Semantic**: Verify well-typedness after repair (double-check)
4. **Process**: Use proven-correct session type checker
5. **Testing**: Extensive property-based testing

**Property-Based Test**:
```rust
#[quickcheck]
fn prop_correction_valid(input: String) -> bool {
    let result = pipeline.correct(&input);
    // All results must be syntactically and semantically valid
    result.iter().all(|r| is_valid_program(r))
}
```

### 9.3 Optimality

**Goal**: Provide good approximations when exact optimality is intractable.

**Recommendations**:
1. **Beam Search**:
   - Use adaptive beam width (expand when needed)
   - Typical values: k=20 for interactive, k=100 for batch
2. **Heuristics**:
   - Use A* with admissible heuristics (never overestimate)
   - Incorporate language model scores
3. **Multi-Objective**:
   - Compute Pareto frontier
   - Present multiple options to user
4. **Feedback**:
   - Train ranking model on correction corpus
   - Use online learning to adapt to user preferences

**Evaluation**:
```rust
fn evaluate_optimality() {
    // Compare against ground truth
    for (input, expected) in test_cases {
        let result = pipeline.correct(input);
        let cost_result = total_cost(&result);
        let cost_optimal = total_cost(&expected);
        let ratio = cost_result / cost_optimal;
        println!("Approximation ratio: {}", ratio);
    }
}
```

### 9.4 Performance

**Goal**: Achieve real-time performance (<500ms for IDE).

**Recommendations**:
1. **Incremental Parsing**: Use Tree-sitter incremental reparse
2. **Caching**: Cache Levenshtein automata, parse states
3. **Parallelization**: Run layers in parallel where possible
4. **Timeouts**: Bound SMT solving to 1-2 seconds
5. **Lazy Evaluation**: Generate candidates on-demand

**Performance Budget**:
```
Layer 1 (Lexical):      50ms  (10 tokens × 5ms)
Layer 2 (Grammar):     200ms  (beam search)
Layer 3 (Validation):   50ms  (type check 20 candidates)
Layer 4 (Repair):      100ms  (if needed)
Layer 5 (Verification): 50ms  (session types)
--------------------------------------------
Total:                 450ms  (within budget)
```

### 9.5 Usability

**Goal**: Provide helpful corrections to users.

**Recommendations**:
1. **Ranking**: Show corrections sorted by confidence
2. **Explanations**: Provide rationale for each correction
3. **Multiple Options**: Show Pareto frontier (top-3)
4. **Interactive Feedback**: Learn from user selections
5. **Incremental Application**: Allow users to accept/reject per-layer

**UI Example**:
```
Error detected: "prnt(x + y"

Suggested corrections:
1. print(x + y)  [Confidence: 95%]
   - Fixed typo: prnt → print
   - Added closing parenthesis
   - Type-correct: print :: a -> IO ()

2. prnt(x + y)  [Confidence: 60%]
   - Added closing parenthesis
   - Assumes 'prnt' is user-defined function
   - Type-correct if prnt :: Int -> Int -> ()

3. print(x) + y  [Confidence: 30%]
   - Fixed typo: prnt → print
   - Moved '+' outside parentheses
   - Type warning: Cannot add IO action and value
```

### 9.6 Testing Strategy

**Recommendations**:

**9.6.1 Unit Tests**:
- Test each layer independently
- Test edge cases (empty input, very long input, deeply nested structures)

**9.6.2 Integration Tests**:
- Test full pipeline on realistic examples
- Test error propagation

**9.6.3 Property-Based Tests**:
- Determinism: Same input → same output
- Correctness: Output is always valid
- Monotonicity: More errors → higher cost

**9.6.4 Benchmark Suite**:
- Real-world programs with injected errors
- Measure precision, recall, F1 score
- Compare against baselines (IDE tools, language servers)

**9.6.5 Performance Tests**:
- Measure latency per layer
- Profile hot paths
- Test with varying input sizes

### 9.7 Documentation

**Recommendations**:
1. **Theoretical Guarantees**: Document what properties hold (this document!)
2. **Complexity**: Document time/space complexity of each layer
3. **Limitations**: Document known failure modes
4. **Tradeoffs**: Document quality vs. performance tradeoffs
5. **Configuration**: Document tunable parameters (beam width, timeouts)

---

## 10. Open Problems

### 10.1 Theoretical Open Problems

**Problem 10.1 (Approximation Ratio for Beam Search)**:
Is there a beam width k such that beam search guarantees an α-approximation for some constant `$\alpha$`?

**Difficulty**: Depends on grammar structure and error distribution. May require average-case analysis.

**Problem 10.2 (Optimal Feedback Update)**:
What is the optimal feedback update rule to maximize long-term correction quality?

**Connection**: Reinforcement learning, bandit algorithms. Requires formal reward model.

**Problem 10.3 (Decidability Boundary for Session Types)**:
Characterize exactly which session type systems are decidable for which process calculi restrictions.

**Related Work**: Linear logic, behavioral types.

### 10.2 Practical Open Problems

**Problem 10.4 (Semantic Repair Ranking)**:
How to rank multiple type-correct repairs by semantic likelihood?

**Approaches**:
- Language models (GPT, CodeBERT)
- Program synthesis techniques
- User study for ground truth

**Problem 10.5 (Error Localization)**:
For cascading errors, how to identify the root cause?

**Related Work**: SHErrLoc, Mycroft, type error diagnosis.

**Problem 10.6 (Incremental Multi-Layer Correction)**:
Can we update corrections incrementally as user types?

**Challenges**: Invalidation propagation across layers.

### 10.3 Extensions

**10.3.1 Probabilistic Correction**:
Assign probabilities to corrections based on language models.

**Advantage**: Natural ranking, uncertainty quantification.

**10.3.2 Program Synthesis**:
Instead of correcting existing code, synthesize new code from specification.

**Advantage**: Potentially better quality.

**10.3.3 Multi-Modal Correction**:
Combine text-based correction with test-based repair (e.g., fix to pass tests).

**Advantage**: Semantic grounding.

---

## 11. References

### 11.1 Automata Theory and Parsing

[1] Schulz, K. U., & Mihov, S. (2002). "Fast String Correction with Levenshtein Automata." *International Journal on Document Analysis and Recognition*, 5(1), 67-85.

[2] Earley, J. (1970). "An Efficient Context-Free Parsing Algorithm." *Communications of the ACM*, 13(2), 94-102.

[3] Younger, D. H. (1967). "Recognition and Parsing of Context-Free Languages in Time n³." *Information and Control*, 10(2), 189-208.

### 11.2 Type Theory

[4] Damas, L., & Milner, R. (1982). "Principal Type-Schemes for Functional Programs." *Proceedings of the 9th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages*, 207-212.

[5] Wright, A. K., & Felleisen, M. (1994). "A Syntactic Approach to Type Soundness." *Information and Computation*, 115(1), 38-94.

[6] McBride, C. (2003). "First-Order Unification by Structural Recursion." *Journal of Functional Programming*, 13(6), 1061-1075.

### 11.3 Error Correction

[7] Wagner, R. A., & Fischer, M. J. (1974). "The String-to-String Correction Problem." *Journal of the ACM*, 21(1), 168-173.

[8] Zhang, K., & Shasha, D. (1989). "Simple Fast Algorithms for the Editing Distance Between Trees and Related Problems." *SIAM Journal on Computing*, 18(6), 1245-1262.

[9] Bille, P. (2005). "A Survey on Tree Edit Distance and Related Problems." *Theoretical Computer Science*, 337(1-3), 217-239.

### 11.4 Type Error Localization

[10] Zhang, D., Myers, A. C., Vytiniotis, D., & Peyton-Jones, S. (2015). "SHErrLoc: A Static Holistic Error Locator." *Proceedings of the 42nd Annual ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages*, 299-312.

[11] Lerner, B. S., Flower, M., Grossman, D., & Chambers, C. (2007). "Searching for Type-Error Messages." *Proceedings of the 28th ACM SIGPLAN Conference on Programming Language Design and Implementation*, 425-434.

### 11.5 Semantic Repair

[12] Nieuwenhuis, R., & Oliveras, A. (2006). "On SAT Modulo Theories and Optimization Problems." *Proceedings of the 9th International Conference on Theory and Applications of Satisfiability Testing*, 156-169.

[13] Bjørner, N., Phan, A. D., & Fleckenstein, L. (2015). "`$\nu Z -$` An Optimizing SMT Solver." *Proceedings of the 21st International Conference on Tools and Algorithms for the Construction and Analysis of Systems*, 194-199.

### 11.6 Process Calculi and Session Types

[14] Honda, K., Vasconcelos, V. T., & Kubo, M. (1998). "Language Primitives and Type Discipline for Structured Communication-Based Programming." *Proceedings of the 7th European Symposium on Programming*, 122-138.

[15] Wadler, P. (2012). "Propositions as Sessions." *Proceedings of the 17th ACM SIGPLAN International Conference on Functional Programming*, 273-286.

[16] Palamidessi, C. (2003). "Comparing the Expressive Power of the Synchronous and Asynchronous π-Calculi." *Mathematical Structures in Computer Science*, 13(5), 685-719.

### 11.7 Multi-Objective Optimization

[17] Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W. H. Freeman.

[18] Miettinen, K. (1999). *Nonlinear Multiobjective Optimization*. Springer.

---

## Appendix A: Formal Proofs

### A.1 Proof of Theorem 4.8 (BFS Optimality)

**Theorem**: If all edits have uniform cost, BFS finds minimum-distance parse tree.

**Proof**:
Let d* be the minimum distance to a valid parse tree.

1. **BFS explores by increasing distance**: At iteration k, BFS has explored all states reachable with cost `$\le  k.$`

2. **Completeness**: If a valid parse tree exists at distance d*, BFS will reach it by iteration d*.

3. **Optimality**: Suppose BFS returns a parse tree at distance d > d*. This contradicts step 1: if a parse tree exists at d* < d, BFS would have found it in iteration d* < d.

4. **First valid tree is optimal**: The first valid parse tree found has minimum distance, as no shorter path exists (would have been found earlier).

Therefore, BFS finds the minimum-distance parse tree for uniform costs.
`$\blacksquare$`

### A.2 Proof of Theorem 8.2 (Non-Compositional Optimality)

**Theorem**: Optimal layers do not compose to optimal pipeline.

**Proof by Counter-Example**:
Consider the counter-example in Section 8.2:

Input: "prnt(x + y"

**Sequential Composition (Layer-Wise Optimal)**:
- Layer 1 optimal: "prnt" → "print" (cost 1)
- Layer 2 optimal: Insert ')' (cost 1)
- Total: 2

**Joint Optimization**:
- Alternative: Keep "prnt", insert ')' (cost 1)
- If "prnt" is defined, this is valid with total cost 1 < 2

Therefore, sequential composition of optimal layers produced suboptimal result (cost 2 vs. 1).
`$\blacksquare$`

### A.3 Proof Sketch of Theorem 6.9 (Repair Undecidability)

**Theorem**: Semantic optimality of repair is undecidable.

**Proof Sketch**:
1. Assume there exists a computable function R that computes semantically optimal repairs.

2. Consider the following decision problem:
   - Given program P with type error and specification S, does P have a repair that satisfies S?

3. This reduces to the problem of program equivalence modulo specification:
   - Does there exist P' such that P' is well-typed and `$P' \equiv_S P$` (equivalent under specification S)?

4. Program equivalence is undecidable by **Rice's Theorem**:
   - Rice's Theorem: Any non-trivial property of program behavior is undecidable.
   - "Satisfies specification S" is a property of program behavior.

5. If R exists, we could decide program equivalence (contradiction).

Therefore, semantic optimal repair is undecidable in general.
`$\blacksquare$`

(Note: This does not preclude decidable approximations for restricted program classes.)

---

## Appendix B: Complexity Summary

| Layer | Algorithm | Time Complexity | Space | Decidable? |
|-------|-----------|----------------|-------|------------|
| **Layer 1** | Levenshtein Automaton | `$\mathcal{O}(n \times  d)$` | `$\mathcal{O}(n \times  d)$` | ✓ Yes |
| **Layer 2** | BFS (pure) | `$\mathcal{O}(\\lvert G\\rvert^d \times  n \times  p)$` | `$\mathcal{O}(\\lvert G\\rvert^d)$` | ✓ Yes |
| **Layer 2** | BFS (beam) | `$\mathcal{O}(k \times  d \times  n \times  p)$` | `$\mathcal{O}(k)$` | ✓ Yes |
| **Layer 3** | Type Inference (HM) | `$\mathcal{O}(n \log  n)$` avg | `$\mathcal{O}(n)$` | ✓ Yes |
| **Layer 4** | Template Repair | `$\mathcal{O}(\text{templates} \times  \text{match})$` | `$\mathcal{O}(\text{AST})$` | ✓ Yes |
| **Layer 4** | SMT Repair | NP-hard | `$\mathcal{O}(\text{constraints})$` | ✗ No (general) |
| **Layer 5** | Session Types (linear) | `$\mathcal{O}(n)$` | `$\mathcal{O}(n)$` | ✓ Yes |
| **Layer 5** | Session Types (general) | `$\mathcal{O}(n^k)$` | `$\mathcal{O}(n^k)$` | ✗ No (full π-calc) |
| **Full Pipeline** | Sequential | Sum of above | Max of above | ✓ Yes (restricted) |
| **Joint Optimization** | Exact | Exponential | Exponential | ✗ No (intractable) |

**Notation**:
- `n`: Input size (tokens, AST nodes)
- `d`: Max edit distance
- `k`: Beam width
- `p`: Parse time per state
- `$\lvert G\rvert$`: Grammar size

---

## Appendix C: Determinism Checklist

Implement these measures to ensure determinism:

- [ ] **Lexical (Layer 1)**:
  - [ ] Use deterministic tie-breaking (lexicographic)
  - [ ] Sort candidates before returning top-k
  - [ ] Use fixed dictionary (no dynamic updates)

- [ ] **Grammar (Layer 2)**:
  - [ ] Use FIFO queue for BFS (not heap with arbitrary ordering)
  - [ ] Break ties by state ID (deterministic)
  - [ ] Use deterministic Tree-sitter (reproducible parse)

- [ ] **Semantic (Layer 3)**:
  - [ ] Use counter-based fresh variable generation
  - [ ] Avoid timestamps or random IDs in types

- [ ] **Repair (Layer 4)**:
  - [ ] Set SMT solver random seed (if using Z3)
  - [ ] Use deterministic template ordering
  - [ ] Sort repairs before returning

- [ ] **Process (Layer 5)**:
  - [ ] Deterministic trace enumeration
  - [ ] Consistent session type checking order

- [ ] **Feedback**:
  - [ ] Option A: Disable feedback for deterministic mode
  - [ ] Option B: Use fixed training data (no online learning)

- [ ] **Testing**:
  - [ ] Add determinism property tests
  - [ ] Run same input 100 times, assert identical outputs

---

## Conclusion

**Summary of Findings**:

| Property | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 | Composed |
|----------|---------|---------|---------|---------|---------|----------|
| **Determinism** | ✓ | ~ | ✓ | ~ | ✓ | ~ |
| **Correctness** | ✓ | ✓ | ✓ | ✓* | ✓ | ✓* |
| **Optimality** | ✓ | ~ | ✓ | ~ | N/A | ✗ |
| **Decidability** | ✓ | ✓ | ✓ | ~ | ✓* | ✓* |

**Legend**:
- ✓: Always holds
- ~: Holds under conditions
- ✗: Does not hold
- *: Syntactic correctness only (semantic correctness harder)

**Key Takeaways**:

1. **Determinism** is achievable but requires careful engineering (tie-breaking, fixed random seeds).

2. **Correctness** is guaranteed syntactically at each layer. Semantic correctness (preserving intent) is harder and may require user feedback.

3. **Optimality** is layer-specific. Global optimality is intractable; practical systems must use approximations (beam search, Pareto optimality).

4. **Decidability** holds for restricted versions of each layer. General cases (full SMT, general π-calculus) are undecidable.

5. **Composition** preserves correctness but not optimality. Greedy layer-wise optimization is globally suboptimal.

**Recommendations**:
- Document these properties and their conditions in user-facing documentation
- Provide configuration options for determinism/performance tradeoffs
- Implement Pareto optimization for multi-objective scenarios
- Use beam search with lookahead for better approximations
- Validate repairs with extensive testing (unit, integration, property-based)

**Future Work**:
- Formalize approximation guarantees for beam search
- Develop optimal feedback update rules (RL-based)
- Extend to probabilistic correction with confidence scores
- Integrate with test-based repair for semantic grounding

---

**Document Version**: 1.0
**Last Updated**: 2025-11-04
**Maintainer**: liblevenshtein-rust project
**Related Documents**:
- `grammar-correction.md`: Pipeline design
- `hierarchical-correction.md`: Layer 1 details
