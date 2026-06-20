# Theoretical Properties - Visual Guide

[← Documentation Index](../../../README.md)

This document provides visual representations of the theoretical properties analyzed in `complete-analysis.md`.

---

## 1. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Raw Text with Errors                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
╔══════════════════════════════════════════════════════════════════╗
║ LAYER 1: LEXICAL CORRECTION (Levenshtein Automata)             ║
╠══════════════════════════════════════════════════════════════════╣
║ Determinism:  ✓ YES (with tie-breaking)                         ║
║ Correctness:  ✓ YES (all within distance d)                     ║
║ Optimality:   ✓ YES (top-k by distance, per-token)              ║
║ Complexity:   O(n × d)                                           ║
║ Decidability: ✓ YES (regular languages)                         ║
╚══════════════════════════════════════════════════════════════════╝
                       │ top-k token candidates
                       ▼
╔══════════════════════════════════════════════════════════════════╗
║ LAYER 2: GRAMMAR CORRECTION (BFS + Tree-sitter)                 ║
╠══════════════════════════════════════════════════════════════════╣
║ Determinism:  ~ CONDITIONAL (unambiguous grammar + tie-break)   ║
║ Correctness:  ✓ YES (syntactically valid parse trees)           ║
║ Optimality:   ~ CONDITIONAL (BFS: yes for uniform cost)         ║
║               ✗ NO (Beam search: approximate only)              ║
║ Complexity:   O(k × d × n × p) with beam width k                ║
║ Decidability: ✓ YES (CFG parsing)                               ║
╚══════════════════════════════════════════════════════════════════╝
                       │ valid parse trees
                       ▼
╔══════════════════════════════════════════════════════════════════╗
║ LAYER 3: SEMANTIC VALIDATION (Type Inference)                   ║
╠══════════════════════════════════════════════════════════════════╣
║ Determinism:  ✓ YES (with deterministic fresh vars)             ║
║ Correctness:  ✓ YES (only well-typed programs pass)             ║
║ Optimality:   ✓ YES (perfect filter)                            ║
║ Complexity:   O(n log n) average case                           ║
║ Decidability: ✓ YES (Hindley-Milner)                            ║
╚══════════════════════════════════════════════════════════════════╝
                       │ well-typed programs
                       ▼
╔══════════════════════════════════════════════════════════════════╗
║ LAYER 4: SEMANTIC REPAIR (Template / SMT / Search)              ║
╠══════════════════════════════════════════════════════════════════╣
║ Determinism:  ~ CONDITIONAL (requires deterministic solver)     ║
║ Correctness:  ✓ YES (syntactic - verified well-typed)           ║
║               ~ CONDITIONAL (semantic - may be wrong intent)     ║
║ Optimality:   ✗ NO (semantically optimal is undecidable)        ║
║               ✓ YES (MaxSMT optimal for constraint set)         ║
║ Complexity:   NP-hard (MaxSMT)                                   ║
║ Decidability: ~ CONDITIONAL (decidable for many theories)       ║
╚══════════════════════════════════════════════════════════════════╝
                       │ repaired programs
                       ▼
╔══════════════════════════════════════════════════════════════════╗
║ LAYER 5: PROCESS VERIFICATION (Session Types)                   ║
╠══════════════════════════════════════════════════════════════════╣
║ Determinism:  ✓ YES                                             ║
║ Correctness:  ✓ YES (session type safety)                       ║
║ Optimality:   N/A (verification only, not optimization)         ║
║ Complexity:   O(n) linear, O(n^k) non-linear                    ║
║ Decidability: ✓ YES (finite session types, bounded processes)   ║
╚══════════════════════════════════════════════════════════════════╝
                       │
                       ▼
╔══════════════════════════════════════════════════════════════════╗
║ COMPOSED PIPELINE                                                ║
╠══════════════════════════════════════════════════════════════════╣
║ Determinism:  ~ CONDITIONAL (depends on all layers + feedback)  ║
║ Correctness:  ✓ YES (syntactic validity preserved)              ║
║ Optimality:   ✗ NO (greedy composition is globally suboptimal)  ║
║ Complexity:   Sum of layers: O(k × d × n × p + n log n + SMT)   ║
║ Decidability: ✓ YES (restricted versions)                       ║
╚══════════════════════════════════════════════════════════════════╝
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT: Corrected, Verified Program                │
└─────────────────────────────────────────────────────────────────┘

Legend: ✓ = Always holds, ~ = Conditional, ✗ = Does not hold
```

---

## 2. Optimality Comparison

### Sequential vs Joint Optimization

```
┌───────────────────────────────────────────────────────────────┐
│                      SEQUENTIAL COMPOSITION                    │
│                    (Layer-wise Greedy)                         │
└───────────────────────────────────────────────────────────────┘

Input: "prnt(x + y"
  │
  ├─ Layer 1 (Lexical): "prnt" → "print" ✓
  │  Cost: 1 (edit distance)
  │  ⚠️  Commits to "print" without considering grammar context
  │
  ├─ Layer 2 (Grammar): Insert ')'
  │  Cost: 1 (syntax repair)
  │  Input: "print(x + y" → "print(x + y)"
  │
  └─ Total Cost: 2

═════════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────────────┐
│                    JOINT OPTIMIZATION                          │
│              (Global Search - Intractable)                     │
└───────────────────────────────────────────────────────────────┘

Input: "prnt(x + y"
  │
  ├─ Option A: Fix lexical then grammar (cost 2)
  │  "prnt" → "print", insert ')' → "print(x + y)"
  │  Cost: 1 + 1 = 2
  │
  ├─ Option B: Fix grammar only (cost 1) ✓ OPTIMAL
  │  Insert ')' → "prnt(x + y)"
  │  Cost: 1
  │  If "prnt" is user-defined function, this is valid!
  │
  └─ Best Cost: 1 < 2 (sequential was suboptimal)

═════════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────────────┐
│                  BEAM SEARCH WITH LOOKAHEAD                    │
│                   (Practical Approximation)                    │
└───────────────────────────────────────────────────────────────┘

Input: "prnt(x + y"
  │
  ├─ Layer 1 generates top-k candidates:
  │  1. "print" (cost 1)
  │  2. "prnt" (cost 0)  ← keep user token
  │  3. "punt" (cost 1)
  │
  ├─ For each candidate, estimate downstream cost:
  │  • "print" + grammar fix = 1 + 1 = 2
  │  • "prnt" + grammar fix = 0 + 1 = 1 ✓ BEST
  │  • "punt" + grammar fix = 1 + 1 = 2
  │
  ├─ Select best path: "prnt" → "prnt(x + y)"
  │
  └─ Total Cost: 1 (approximates optimal!)
```

---

## 3. Decidability Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                   DECIDABLE (Guaranteed Termination)            │
└─────────────────────────────────────────────────────────────────┘
       │
       ├─ ✓ Regular Languages (Lexical - Layer 1)
       │  • Levenshtein automata
       │  • Complexity: O(n × d)
       │
       ├─ ✓ Context-Free Languages (Grammar - Layer 2)
       │  • CFG parsing (Earley, CYK)
       │  • Complexity: O(n³) worst-case
       │
       ├─ ✓ Hindley-Milner Type Inference (Semantic - Layer 3)
       │  • Algorithm W
       │  • Complexity: O(n log n) average
       │
       ├─ ✓ Linear Session Types (Process - Layer 5)
       │  • Single channel, sequential
       │  • Complexity: O(n)
       │
       └─ ✓ Bounded Non-linear Session Types
          • k parallel sessions
          • Complexity: O(n^k)

═════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│          SEMI-DECIDABLE (May Not Terminate)                     │
└─────────────────────────────────────────────────────────────────┘
       │
       ├─ ~ MaxSMT for Some Theories (Semantic Repair - Layer 4)
       │  • Decidable: Linear arithmetic, uninterpreted functions
       │  • Use timeouts in practice
       │
       └─ ~ General Session Types (Process - Layer 5)
          • With restrictions, decidable
          • Full π-calculus: undecidable

═════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│               UNDECIDABLE (No Termination Guarantee)            │
└─────────────────────────────────────────────────────────────────┘
       │
       ├─ ✗ System F Type Inference
       │  • Explicit type abstraction
       │  • (Not used in our pipeline)
       │
       ├─ ✗ Semantic Optimal Repair (Layer 4)
       │  • Reduction from Rice's Theorem
       │  • Use heuristics + approximations
       │
       ├─ ✗ General MaxSMT
       │  • Non-linear arithmetic, quantifiers
       │  • Restrict to decidable fragments
       │
       └─ ✗ Full π-Calculus Behavioral Equivalence
          • Name mobility leads to unbounded behavior
          • Use bounded approximations
```

---

## 4. Complexity Landscape

```
┌────────────────────────────────────────────────────────────────┐
│                    TIME COMPLEXITY BY LAYER                    │
└────────────────────────────────────────────────────────────────┘

Fastest                                                     Slowest
  │                                                             │
  ├─────────┬──────────┬──────────┬──────────┬────────────────┤
  │         │          │          │          │                │
O(n)    O(n log n)  O(n²)      O(n³)    NP-hard       Exponential
  │         │          │          │          │                │
  │         │          │          │          │                │
Layer 5   Layer 3   Unify    CFG Parse  Layer 4      Joint Opt
(linear   (Type     (worst)  (worst)    (SMT)       (no beam)
session)  infer)
  │         │                             │                │
Layer 1                                Layer 2          Layer 2
(Lev aut)                           (beam: tractable) (pure BFS)

═════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────┐
│                   PRACTICAL RUNTIME BUDGET                     │
│                  (Interactive IDE, 100 tokens)                 │
└────────────────────────────────────────────────────────────────┘

Layer 1: ████░░░░░░░░░░░░░░░░  50ms   (11%)
Layer 2: ████████████████████ 200ms   (44%)  ← Bottleneck
Layer 3: ████░░░░░░░░░░░░░░░░  50ms   (11%)
Layer 4: ████████░░░░░░░░░░░░ 100ms   (22%)
Layer 5: ████░░░░░░░░░░░░░░░░  50ms   (11%)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:                        450ms   (100%)

Target: <500ms for IDE responsiveness ✓

Optimization strategies:
• Incremental parsing (Tree-sitter)
• Caching (automata, parse states)
• Parallelization (independent candidates)
• Adaptive beam width (expand only when needed)
```

---

## 5. Property Trade-offs

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETERMINISM vs PERFORMANCE                   │
└─────────────────────────────────────────────────────────────────┘

Deterministic Mode (Reproducible):
├─ Lexicographic tie-breaking
├─ Fixed random seeds (SMT solver)
├─ No online learning / feedback
└─ Priority: Testing, debugging, verification
   Performance: Slight overhead (sorting)

Non-Deterministic Mode (Fast):
├─ First-match return (no sorting)
├─ Non-deterministic heuristics (SMT)
├─ Online learning from user feedback
└─ Priority: Production use, adaptation
   Performance: 10-20% faster

═════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                   OPTIMALITY vs TRACTABILITY                    │
└─────────────────────────────────────────────────────────────────┘

Exact Optimal (Joint Optimization):
├─ Explore all candidate combinations
├─ Guarantee minimum cost
├─ Complexity: Exponential
└─ Practical: INFEASIBLE for >10 tokens

Beam Search (k=20):
├─ Keep top-20 candidates per layer
├─ Approximate optimal
├─ Complexity: O(k × d × n × p)
└─ Practical: 200ms for 100 tokens ✓

Greedy (k=1):
├─ Keep only top-1 candidate
├─ No optimality guarantee
├─ Complexity: O(d × n × p)
└─ Practical: 50ms for 100 tokens ✓

Trade-off curve:
Beam Width    Quality (F1)    Latency
─────────────────────────────────────
k=1           70%              50ms
k=5           85%             100ms
k=20          92%             200ms
k=100         96%             800ms
Exact         100%            ∞ (intractable)

═════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                  CORRECTNESS vs EXPRESSIVENESS                  │
└─────────────────────────────────────────────────────────────────┘

Strong Guarantees (Restricted):
├─ Hindley-Milner (no dependent types)
├─ Linear session types (no sharing)
├─ Decidable type checking
└─ Properties: Decidable, O(n log n)

Weak Guarantees (Expressive):
├─ Dependent types
├─ Non-linear session types
├─ Undecidable in general
└─ Properties: Use bounded checks, timeouts

Our Choice: Strong guarantees for core, extensions optional
```

---

## 6. Pareto Frontier

```
┌─────────────────────────────────────────────────────────────────┐
│           MULTI-OBJECTIVE OPTIMIZATION (2D Example)             │
│                                                                 │
│         Objective 1: Lexical Distance (minimize)                │
│         Objective 2: Grammar Changes (minimize)                 │
└─────────────────────────────────────────────────────────────────┘

Grammar
Changes
  5 │                            • Non-optimal
    │                          •   (dominated)
  4 │                    •
    │            • Non-optimal
  3 │          •   (dominated)
    │      ★ Pareto         • Non-optimal
  2 │    /   optimal          (dominated)
    │  /
  1 │★────★ Pareto Frontier (optimal solutions)
    │
  0 └──────────────────────────────────────────> Lexical
    0   1   2   3   4   5   6   7   8   9  10    Distance

Pareto optimal solutions (★):
• (0, 1): No lexical changes, 1 grammar fix
• (1, 1): 1 lexical change, 1 grammar fix
• (2, 0): 2 lexical changes, no grammar fix

Non-dominated: No solution improves one objective without worsening the other.

User Selection:
┌────────────────────────────────────────────────────────┐
│ Multiple corrections found. Please select:             │
├────────────────────────────────────────────────────────┤
│ 1. prnt(x + y)    [Keep typo, fix grammar]            │
│    Lexical: 0, Grammar: 1                              │
│                                                        │
│ 2. print(x + y)   [Fix typo, fix grammar]             │
│    Lexical: 1, Grammar: 1                              │
│                                                        │
│ 3. prnt = x + y   [Fix typo differently, no grammar]  │
│    Lexical: 2, Grammar: 0                              │
└────────────────────────────────────────────────────────┘
```

---

## 7. Feedback Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                      FEEDBACK MECHANISM                         │
└─────────────────────────────────────────────────────────────────┘

Without Feedback (Static):
┌────────┐        ┌────────┐        ┌────────┐
│ Layer 1│───────>│ Layer 2│───────>│ Layer 3│──────> Output
│(Lexical)│       │(Grammar)│       │(Semantic)│
└────────┘        └────────┘        └────────┘
  Fixed weights    Fixed beam        Fixed filter

═════════════════════════════════════════════════════════════════

With Feedback (Learning):
┌────────┐        ┌────────┐        ┌────────┐
│ Layer 1│───────>│ Layer 2│───────>│ Layer 3│──────> Output
│(Lexical)│       │(Grammar)│       │(Semantic)│         │
└────┬───┘        └────┬───┘        └────┬───┘         │
     │                 │                 │              │
     │ Update weights  │ Adjust beam     │ Learn patterns│
     │◄────────────────┼─────────────────┼──────────────┘
     │                 │                 │
     │                 │                 │
  ┌──▼─────────────────▼─────────────────▼───┐
  │        Feedback Learning Module          │
  │  • Track which corrections succeed       │
  │  • Update layer weights                  │
  │  • Bias toward well-typed results        │
  └──────────────────────────────────────────┘

Benefits:
✓ Improves over time
✓ Adapts to user preferences
✓ Better approximation

Drawbacks:
✗ Breaks determinism (unless offline training)
✗ May overfit to specific patterns
✗ Requires large corpus for training

═════════════════════════════════════════════════════════════════

Practical Implementation:
┌─────────────────────────────────────────────────────────────────┐
│                    RANKING MODEL                                │
└─────────────────────────────────────────────────────────────────┘

Input: (original, candidate, context)
      │
      ├─ Feature Extraction:
      │  • Lexical distance
      │  • Grammar changes
      │  • Type correctness (Boolean)
      │  • Semantic edits
      │  • User past preferences
      │
      ├─ Scoring Model (e.g., XGBoost):
      │  score = w₁×lex + w₂×gram + w₃×type + w₄×sem + w₅×pref
      │
      └─ Output: Ranked candidates

Training:
• Corpus: 10K+ corrected programs
• Labels: User selections
• Method: Supervised learning (LambdaMART, RankNet)
• Update: Periodically (offline) or incrementally (online)
```

---

## 8. Error Cascade Example

```
┌─────────────────────────────────────────────────────────────────┐
│                        ERROR CASCADE                            │
│          (Single Root Cause → Multiple Symptoms)                │
└─────────────────────────────────────────────────────────────────┘

Input:
  for i in rang(10):
           └─── Typo (root cause)

Layer 1 (Lexical):
  Detects: "rang" is not in dictionary
  Suggests: range, rank, rang (if in custom dictionary)
  ├─ Path A: Fix typo → "range"
  └─ Path B: Keep typo → "rang" (proceed)

Path A (Fix typo):
  for i in range(10):
      └─── Missing block (secondary error)

  Layer 2 (Grammar):
  Detects: Missing indented block after ':'
  Suggests: Insert 'pass' or 'break'

  Result: "for i in range(10):\n    pass"
  Cost: 1 (lexical) + 1 (grammar) = 2

Path B (Keep typo):
  for i in rang(10):
      └─── Both typo and missing block

  Layer 2 (Grammar):
  Detects: Missing indented block
  Suggests: Insert 'pass'

  Result: "for i in rang(10):\n    pass"
  Cost: 0 (lexical) + 1 (grammar) = 1

  BUT: "rang" is undefined! (semantic error)

  Layer 3 (Semantic):
  Rejects: "rang" is not defined

  Layer 4 (Semantic Repair):
  Must fix both issues:
  • Define "rang" as function
  • OR: Change to "range"

  Best repair: Change "rang" → "range"
  Final cost: 1 (semantic repair)

═════════════════════════════════════════════════════════════════

Key Insight: Feedback from Layer 3 informs Layer 1
• Path A (fix typo immediately) avoids cascade
• Path B (keep typo) requires expensive semantic repair
• Multi-layer optimization prevents cascades
```

---

## 9. Decision Tree for Implementation

```
┌─────────────────────────────────────────────────────────────────┐
│               IMPLEMENTATION DECISION TREE                      │
└─────────────────────────────────────────────────────────────────┘

Start: Implement multi-layer error correction
  │
  ├─ Priority: Determinism?
  │  ├─ YES: Testing, verification, debugging
  │  │  ├─ Use lexicographic tie-breaking
  │  │  ├─ Set fixed random seeds
  │  │  ├─ Disable online feedback
  │  │  └─ Trade-off: 10-20% slower
  │  │
  │  └─ NO: Production use, adaptive
  │     ├─ Allow non-deterministic heuristics
  │     ├─ Enable online learning
  │     └─ Benefit: 10-20% faster, adapts to users
  │
  ├─ Priority: Performance?
  │  ├─ Real-time (<500ms): Interactive IDE
  │  │  ├─ Beam width k=20
  │  │  ├─ SMT timeout 2s
  │  │  ├─ Incremental parsing
  │  │  └─ Caching + parallelization
  │  │
  │  └─ Batch processing: Offline analysis
  │     ├─ Beam width k=100
  │     ├─ SMT timeout 10s
  │     ├─ Higher quality, slower
  │     └─ Pareto frontier for ranking
  │
  ├─ Priority: Optimality?
  │  ├─ Approximate OK: Most use cases
  │  │  ├─ Beam search with lookahead
  │  │  ├─ A* heuristics
  │  │  └─ 90-95% quality, tractable
  │  │
  │  └─ Exact required: Critical applications
  │     ├─ Dynamic programming (if substructure)
  │     ├─ Joint optimization (small inputs only)
  │     └─ Warning: May be intractable
  │
  └─ Priority: Correctness?
     ├─ Syntactic only: Basic use
     │  ├─ Layers 1-3 sufficient
     │  ├─ Fast, decidable
     │  └─ Type-correct guaranteed
     │
     └─ Semantic required: Advanced use
        ├─ Add Layer 4 (SMT repair)
        ├─ Manual verification (property tests)
        ├─ User feedback for ranking
        └─ Warning: Semantic optimality undecidable
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-04
**See Full Analysis**: `complete-analysis.md`
**See Quick Reference**: `quick-reference.md`
