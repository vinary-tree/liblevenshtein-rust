# Theoretical Guarantees - Quick Reference

[← Documentation Index](../../../README.md)

**Quick reference for theoretical properties of the multi-layer error correction pipeline**

See `complete-analysis.md` for full analysis with proofs.

---

## Property Matrix

| Layer | Determinism | Correctness | Optimality | Complexity |
|-------|-------------|-------------|------------|------------|
| **1. Lexical** | ✓ (with tie-breaking) | ✓ (all candidates within distance d) | ✓ (top-k by distance) | $`\mathcal{O}(n \times  d)`$ |
| **2. Grammar** | ~ (requires unambiguous grammar) | ✓ (syntactically valid) | ~ (BFS optimal for uniform cost) | $`\mathcal{O}(k \times  d \times  n \times  p)`$ beam |
| **3. Semantic Validation** | ✓ (with deterministic fresh vars) | ✓ (only well-typed pass) | ✓ (perfect filter) | $`\mathcal{O}(n \log  n)`$ avg |
| **4. Semantic Repair** | ~ (requires deterministic solver) | ✓ (syntactic) ~ (semantic) | ✗ (undecidable) | NP-hard (SMT) |
| **5. Process Verification** | ✓ | ✓ (session type safety) | N/A (verification only) | $`\mathcal{O}(n)`$ to $`\mathcal{O}(n^k)`$ |
| **Composition** | ~ (depends on all layers) | ✓ (syntactic) | ✗ (greedy suboptimal) | Sum of layers |

**Legend**: ✓ = Yes, ~ = Conditional, ✗ = No, N/A = Not applicable

---

## Key Theorems

### Determinism

**Theorem (Lexical Determinism)**: Levenshtein automaton construction is deterministic. Query is deterministic with lexicographic tie-breaking.

**Theorem (Grammar Non-Determinism)**: BFS grammar correction is non-deterministic for ambiguous grammars or without tie-breaking rules.

**Theorem (Type Inference Determinism)**: Algorithm W (Hindley-Milner) is deterministic with deterministic fresh variable generation.

**Theorem (Feedback Breaks Determinism)**: Feedback mechanism breaks determinism unless feedback updates are deterministic.

### Correctness

**Theorem (Levenshtein Correctness)**: Levenshtein automaton correctly recognizes all and only strings within distance d.

**Theorem (Grammar Correctness)**: All parse trees returned by BFS are syntactically valid (in ParseTree(G)).

**Theorem (Type Safety)**: Well-typed programs satisfy Progress and Preservation (don't go wrong).

**Theorem (Session Type Safety)**: Well-typed processes with dual session types are deadlock-free and communication-safe.

**Theorem (Compositional Correctness)**: If each layer is correct, the composition is correct.

### Optimality

**Theorem (Top-k Optimality)**: Lexical correction returns k closest dictionary words (per-token optimal).

**Theorem (BFS Optimality)**: BFS finds minimum-distance parse tree for **uniform edit costs**.

**Theorem (BFS Non-Optimality)**: BFS does **not** guarantee optimality for **non-uniform costs**. Use Dijkstra's algorithm instead.

**Theorem (Beam Search Non-Optimality)**: Beam search does **not** guarantee optimality even for uniform costs.

**Theorem (Non-Compositional Optimality)**: Optimal layers do **not** compose to optimal pipeline. Sequential composition is globally suboptimal.

**Theorem (Repair Undecidability)**: Semantic optimal repair is **undecidable** (by reduction from Rice's Theorem).

**Theorem (Joint Optimization Intractability)**: Joint optimization over all layers is **intractable** (exponential search space).

### Complexity

**Theorem (Lexical Complexity)**: $`\mathcal{O}(n \times  d)`$ construction, $`\mathcal{O}(\lvert V\rvert)`$ query per word, $`\mathcal{O}(n \times  d)`$ space.

**Theorem (Grammar Complexity - Pure BFS)**: $`\mathcal{O}(\lvert \text{Grammar}\rvert^d \times  n \times  p)`$ worst-case (exponential in d).

**Theorem (Grammar Complexity - Beam BFS)**: $`\mathcal{O}(k \times  d \times  n \times  p)`$ (tractable for k = 20-100).

**Theorem (Type Inference Complexity)**: $`\mathcal{O}(n \log  n)`$ average case, exponential worst case (rare in practice).

**Theorem (SMT Repair Complexity)**: MaxSMT is NP-hard.

**Theorem (Session Types Complexity)**: $`\mathcal{O}(n)`$ for linear session types, $`\mathcal{O}(n^k)`$ for non-linear.

### Decidability

**Theorem (Regular Language Decidability)**: Lexical correction is decidable (finite automata).

**Theorem (CFG Parsing Decidability)**: Grammar correction is decidable ($`\mathcal{O}(n^{3})`$ worst-case parsing).

**Theorem (Type Inference Decidability)**: Hindley-Milner type inference is decidable (Algorithm W always terminates).

**Theorem (SMT Decidability)**: MaxSMT is undecidable in general but decidable for many practical theories (linear arithmetic, uninterpreted functions).

**Theorem (Session Type Decidability)**: Session type checking is decidable for finite session types and bounded processes.

**Theorem $`(\pi`$-Calculus Undecidability)**: Behavioral equivalence for full $`\pi`$-calculus is undecidable.

---

## Counter-Examples

### CE 1: Lexical Non-Determinism
```
Input: "teh", d=1, k=1
Dictionary: {tea, ten, the}
All have distance 1
Without tie-breaking: could return any of {tea, ten, the}
```

### CE 2: Grammar Ambiguity
```python
Grammar: Expr → Expr + Expr | Expr * Expr | number
Input: "1 + 2 3" (missing operator)
Corrections (both cost 1):
  - Insert '+': "1 + 2 + 3"
  - Insert '*': "1 + 2 * 3"
BFS may return either depending on exploration order
```

### CE 3: Sequential Composition Suboptimality
```
Input: "prnt(x + y" (missing ')', typo in 'print')

Sequential (layer-wise optimal):
  Layer 1: "prnt" → "print" (cost 1)
  Layer 2: Insert ')' (cost 1)
  Total: 2

Joint optimization:
  Keep "prnt", insert ')' (cost 1)
  If "prnt" is user-defined, this is valid with total cost 1 < 2
```

### CE 4: Semantic Repair Incorrectness
```python
Original intent:
  balance: Int = 100
  withdraw(amount: Int): balance = balance - amount

Buggy code:
  balance: Int = 100
  withdraw(amount: String): balance = balance - amount  # Type error

SMT repair (type-correct but semantically wrong):
  balance: String = "100"  # Changed wrong variable!
  withdraw(amount: String): balance = balance - amount
```

---

## Practical Recommendations

### To Achieve Determinism

1. **Lexical**: Use lexicographic tie-breaking, sort candidates
2. **Grammar**: Use FIFO queue, deterministic tie-breaking, fixed exploration order
3. **Semantic**: Counter-based fresh variables, deterministic SMT solver (set seed)
4. **Feedback**: Disable for deterministic mode OR use fixed training data

### To Maximize Correctness

1. **Lexical**: Validated dictionary, no invalid entries
2. **Grammar**: Filter out parse trees with error nodes
3. **Semantic**: Double-check well-typedness after repair
4. **Process**: Use proven-correct session type checker
5. **Testing**: Extensive property-based testing

### To Approximate Optimality

1. **Beam Search**: Use k=20 (interactive) or k=100 (batch), adaptive beam width
2. **Heuristics**: A* with admissible heuristics, language model scores
3. **Multi-Objective**: Compute Pareto frontier, present multiple options
4. **Feedback**: Train ranking model on correction corpus, online learning
5. **Non-Uniform Costs**: Use Dijkstra's algorithm instead of BFS

### Performance Targets

```
Layer 1 (Lexical):       50ms  (10 tokens × 5ms)
Layer 2 (Grammar):      200ms  (beam search)
Layer 3 (Validation):    50ms  (type check 20 candidates)
Layer 4 (Repair):       100ms  (if needed)
Layer 5 (Verification):  50ms  (session types)
-------------------------------------------------
Total:                  450ms  (acceptable for IDE)
```

---

## Configuration Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `lexical_max_distance` | 2 | 1-3 | Higher = more candidates, slower |
| `lexical_top_k` | 5 | 1-20 | Number of candidates per token |
| `grammar_beam_width` | 20 | 5-100 | Higher = better quality, slower |
| `grammar_max_distance` | 2 | 1-5 | Max syntax errors to fix |
| `type_inference_timeout` | 1000ms | 100-5000ms | Timeout for complex types |
| `smt_repair_timeout` | 2000ms | 500-10000ms | Timeout for SMT solver |
| `session_type_depth` | 10 | 5-50 | Max session type nesting |
| `enable_feedback` | true | bool | Enable learning from corrections |
| `deterministic_mode` | false | bool | Sacrifice learning for reproducibility |

---

## Testing Checklist

### Unit Tests
- [ ] Test each layer independently
- [ ] Test edge cases (empty, very long, deeply nested)
- [ ] Test error propagation between layers

### Property-Based Tests
- [ ] **Determinism**: Same input → same output (run 100 times)
- [ ] **Correctness**: All outputs are valid programs
- [ ] **Monotonicity**: More errors → higher cost
- [ ] **Idempotence**: Correct(Correct(x)) = Correct(x)

### Integration Tests
- [ ] Test full pipeline on realistic examples
- [ ] Test with injected errors (typos, syntax, type)
- [ ] Test feedback mechanism

### Performance Tests
- [ ] Measure latency per layer
- [ ] Profile hot paths (flamegraph)
- [ ] Test with varying input sizes (10, 100, 1000 tokens)

### Benchmark Suite
- [ ] Real-world programs with injected errors
- [ ] Measure precision, recall, F1 score
- [ ] Compare against baselines (IDE tools, language servers)

---

## Open Problems

1. **Approximation Ratio for Beam Search**: Is there a beam width k that guarantees $`\alpha`$-approximation?

2. **Optimal Feedback Update**: What feedback rule maximizes long-term correction quality?

3. **Decidability Boundary**: Exactly which session type systems are decidable for which process calculi?

4. **Semantic Repair Ranking**: How to rank type-correct repairs by semantic likelihood?

5. **Error Localization**: For cascading errors, how to identify root cause?

6. **Incremental Multi-Layer Correction**: Can we update corrections incrementally as user types?

---

## References (Abbreviated)

For full references with details, see `complete-analysis.md`.

**Automata & Parsing**: Schulz & Mihov 2002 (Levenshtein automata), Earley 1970 (parsing)

**Type Theory**: Damas & Milner 1982 (HM type inference), Wright & Felleisen 1994 (type safety)

**Edit Distance**: Wagner & Fischer 1974 (string), Zhang & Shasha 1989 (tree)

**Error Localization**: Zhang et al. 2015 (SHErrLoc), Lerner et al. 2007 (type error search)

**SMT**: Nieuwenhuis & Oliveras 2006 (MaxSMT), Bjørner et al. 2015 ($`\nu Z`$ optimizer)

**Process Calculi**: Honda et al. 1998 (session types), Wadler 2012 (propositions as sessions)

**Complexity**: Garey & Johnson 1979 (NP-completeness), Miettinen 1999 (multi-objective)

---

## Glossary

**Determinism**: Same input always produces same output.

**Correctness**: Output satisfies validity constraints (syntactic, semantic, behavioral).

**Optimality**: Output minimizes cost function (edit distance, error count, etc.).

**Decidability**: Problem has an algorithm that always terminates with correct answer.

**Pareto Optimality**: Solution that cannot be improved in one objective without worsening another.

**Beam Search**: Approximate search that keeps only top-k candidates at each step.

**Admissible Heuristic**: Heuristic that never overestimates true cost (used in A*).

**Principal Type**: Most general type for an expression (unique up to renaming).

**Session Type**: Behavioral type specifying communication protocol.

**MaxSMT**: Optimization variant of SMT that maximizes satisfied constraints.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-04
**See Full Analysis**: `complete-analysis.md`
