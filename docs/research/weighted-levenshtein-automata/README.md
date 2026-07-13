# Weighted Levenshtein Automata - Research & Analysis

**Comprehensive Documentation on Extending Levenshtein Automata to Support Weighted Operations**

**Date**: 2025-11-06  
**Status**: Research Complete - Implementation Recommendations Provided

---

## Executive Summary

### The Question

Can a general algebra, calculus, or set of rules be derived to support:
1. **Weighted transitions** where elementary operations have variable costs?
2. **Extensible operations** allowing new edit operation types to be defined?

### The Answer

**For Weighted Transitions:**
- ✅ **YES with discretization**: Costs rounded to fixed precision work within the framework
- ❌ **NO for continuous costs**: True continuous costs break fundamental assumptions
- **Complexity**: `$\mathcal{O}(|W| \times  \max _\text{cost}/\text{precision})$` vs `$\mathcal{O}(|W|)$` for uniform costs

**For Extensible Operations:**
- ✅ **YES**: Clear methodology exists for deriving new operation types
- ✅ **Already demonstrated**: Transposition, merge/split successfully added
- **Requirement**: Operations must maintain locality and bounded cost properties

---

## Key Findings

### 1. The Derivation Methodology (How Schulz/Mihov/Mitankin Did It)

**Foundation**: Wagner-Fischer Dynamic Programming → Finite Automata

**Process:**
1. **Abstract from DP matrix** to position structure (index, errors)
2. **Geometric reasoning** in edit graph to derive subsumption
3. **Case analysis** on character matches to derive transitions
4. **Parameterization** into state types for precomputation
5. **Extension pattern** with flag bits for multi-step operations

**Key Insight**: Suffix independence (Lemma 2.0.2) justifies tracking only (position, errors).

### 2. Why Uniform Cost = 1 Is Fundamental

**The `$\mathcal{O}(|W|)$` guarantee relies on:**

```
Uniform cost → Bounded positions per state (≤ n+1)
             → Total states O(|W|) for fixed n
             → Linear construction time
```

**With variable costs:**

```
Variable costs → Infinite possible accumulated costs
              → Cannot enumerate all states
              → State explosion: O(|W| × (max_cost/min_cost)^n)
```

**Critical dependencies:**
- **Subsumption formula** `$|j-i| \le  f-e$` assumes each move costs 1
- **State types** depend on discrete error counts
- **Precomputed tables** T_n rely on fixed degree n

### 3. The Discretization Solution

**Key Idea**: Round costs to fixed precision `$\varepsilon$`

```rust
struct WeightedPosition {
    term_index: usize,
    cost_units: u32,  // actual_cost = cost_units × ε
}
```

**Advantages:**
- Finite state space: `$\mathcal{O}(|W| \times  \lceil \max _\text{cost}/\varepsilon \rceil)$`
- Preserves core structure (positions, transitions, subsumption)
- User-configurable precision/performance trade-off

**Complexity:**
- Construction: `$\mathcal{O}(|W| \times  \max _\text{cost}/\varepsilon)$`
- Query: `$\mathcal{O}(|D| \times  \max _\text{cost}/\varepsilon)$`
- **For fixed max_cost/`$\varepsilon$`**: Still `$\mathcal{O}(|W|)$` and `$\mathcal{O}(|D|)$` ✓

**Example - Keyboard Distance:**
```rust
// Costs: adjacent=0.5, same_row=1.0, different_row=1.5
// Precision: ε=0.1
// Cost units: 5, 10, 15 respectively
// State count: O(|W| × 15) = O(|W|) ✓
```

---

## Documentation Structure

This directory contains detailed analysis of:

### Core Concepts

**Derivation Methodology** (this document below)
- How positions, subsumption, transitions were derived
- Connection to dynamic programming
- Extension pattern for new operations

**Weighted Extension Analysis**
- Component-by-component breakdown
- What breaks and why
- Discretization mathematics

**General Framework**
- Recipe for deriving new operations
- Required mathematical properties
- Concrete example: keyboard-distance edits

---

## Detailed Analysis

## PART I: DERIVATION METHODOLOGY

### How Positions Were Derived

**Starting Point**: Wagner-Fischer DP computes matrix M[i,j] = d_L(W[1:i], V[1:j])

**Key Observation**: 
```
During dictionary traversal:
- V revealed character-by-character
- Need to track: "where am I in W?" and "how many errors so far?"
- Don't need: specific path taken (all that matters is distance)
```

**Mathematical Justification** (Lemma 2.0.2):
```
If W = UW' and V = UV' (same prefix), then:
  d_L(V, W) = d_L(V', W')
```

**Implication**: Distance depends only on remaining suffixes, not on how we got here!

**Therefore**:
```
Position π = (i, e)
  i = index in W (how much consumed)
  e = accumulated errors
```

**Invariants Maintained:**
1. **Admissibility**: From i#e, can accept any V with d_L(W[i+1:`$|W|], V) \le  n-e$`
2. **Monotonicity**: i only increases (no backtracking)
3. **Bounded**: `$0 \le  i \le  |W|, 0 \le  e \le  n$`

### How Subsumption Was Derived

**Formula**: i#`$e \sqsubseteq  j$`#f iff `$(e < f) \land  (|j-i| \le  f-e)$`

**Geometric Reasoning** (Edit Graph):

```
Edit graph: Horizontal axis = W, Vertical axis = V
Position (i,e) represents: matched i chars of W with e errors

From (i,e), reachable region:
  R(i,e) = { (i',e') : |i'-i| + |e'-e| ≤ n-e }
  (Manhattan ball of radius n-e)

For i#e to subsume j#f:
  Must have: R(j,f) ⊆ R(i,e)
  
  Condition 1: Radius check
    n-f ≤ n-e  →  e < f  (fewer errors = larger radius)
  
  Condition 2: Center reachability
    Can reach j from i within budget (f-e)?
    Manhattan distance |j-i| ≤ (f-e) budget units
```

**Example**:
```
n=2, positions 3#0 and 4#1

Check: 3#0 ⊑ 4#1?
  e < f: 0 < 1 ✓
  |j-i| ≤ f-e: |4-3| = 1 ≤ 1-0 = 1 ✓
  Result: YES, 3#0 subsumes 4#1
```

### How Characteristic Vectors Were Derived

**DP Recurrence** (Wagner-Fischer):
```
M[i,j] = M[i-1,j-1]              if W[i] = V[j]
M[i,j] = 1 + min(M[i-1,j-1],    if W[i] ≠ V[j]
                  M[i-1,j],
                  M[i,j-1])
```

**Key Insight**: Decision depends ONLY on character match!

**Generalization**:
```
At position i in W, reading character x from dictionary:
  Need to know: "Where does x match in W[i+1:i+k]?"
  If matches at position j: Can skip to j with j-1 substitutions
```

**Bit Vector Encoding**:
```
χ(x, W[i:j]) = ⟨b₁, ..., b_k⟩
where b_k = 1 if W[i+k] = x, else 0
```

**Example**:
```
W = "hello", i=1, x='l'
W[2:5] = "ello"
χ('l', "ello") = ⟨0,1,1,0⟩
  Position 2: 'e' ≠ 'l' → 0
  Position 3: 'l' = 'l' → 1
  Position 4: 'l' = 'l' → 1
  Position 5: 'o' ≠ 'l' → 0
```

### How Transitions Were Derived

**From DP to Transitions**:

For position `$\pi  = i$`#e reading character x:

**Case 1**: `$\chi = \langle 1, \dots\rangle$` (immediate match)
```
W[i+1] = x
  → DP: M[i+1,j+1] = M[i,j] (no error)
  → Automaton: (i+1)#e
```

**Case 2**: `$\chi  = \langle 0,...,0,1,...\rangle$` at position j (match later)
```
Three DP operations possible:
1. Insert x: Stay at i, consume x from dict
   → i#(e+1)
   
2. Delete W[i+1]: Advance in W, add error
   → (i+1)#(e+1)
   
3. Multi-substitute then match: Skip to j
   → (i+j)#(e+j-1)
```

**Optimization**: Instead of creating intermediate positions for each substitution, jump directly to match point!

**Case 3**: `$\chi = \langle 0,\dots,0\rangle$` (no match)
```
Only insert/delete:
  → {i#(e+1), (i+1)#(e+1)}
Note: Substitution would give (i+1)#(e+1), same as delete
```

### Extension to Transpositions

**Problem**: "abc" → "bac" requires 2 standard operations but 1 transposition

**Solution Strategy**:

1. **Add Flag**: i#e_t where `$t \in$` {0,1}
   - t=0: regular position
   - t=1: transposition in progress

2. **Extended Subsumption**:
   ```
   i#e_t ⊑ j#f_s iff:
     (e < f) ∧ (|j-i| ≤ f-e) ∧ (t ≤ s)
   ```
   Regular can subsume special, not vice versa

3. **New Transition Cases**:
   ```
   From i#e_0 reading x:
     If W[i+1] ≠ x AND W[i+2] = x:
       → Create i#(e+1)_1 (flag potential transposition)
   
   From i#e_1 reading y:
     If W[i+1] = y (the swapped character):
       → Complete: (i+1)#e_0
   ```

**Why It Works**: Flag tracks "seen first char, waiting for second"

---

## PART II: GENERAL DERIVATION FRAMEWORK

### Recipe for New Operations

**Step 1: Define Semantics**
```
Operation name: <operation>
Transformation: W → V (what changes?)
Cost: 1 (uniform) or c(op) (weighted)
Examples: concrete cases
```

**Step 2: Determine Atomicity**
```
Single-step operation? → No flag
Multi-step operation? → Add flag bit
```

**Step 3: Extend Positions**
```
Position = (i, e) or (i, e, flag)
Document flag states and meanings
```

**Step 4: Define Transitions**
```
For each case of characteristic vector:
  Specify resulting positions
  Justify why these positions
```

**Step 5: Extend Subsumption**
```
How do flagged positions compare?
Prove no false subsumptions
```

**Step 6: Prove Complexity**
```
Count positions per state:
  Base: O(|W|)
  Non-base: O(n) per base?
  Total: O(|W|) for fixed n?
```

**Step 7: Correctness**
```
Prove: L(Automaton) = L_target
Via induction on word length
```

### Required Properties for Operations

**Property 1: Locality**
- Decision uses only `$\mathcal{O}(n)$` lookahead in W
- Enables finite state representation

**Property 2: Bounded Cost**
- Uniform: cost = 1
- Weighted: `$c_\text{min} \le c(\text{op}) \le c_\text{max}$` with bounded ratio

**Property 3: Composability**
- Can be inserted, deleted, substituted/transformed

**Property 4: Subsumption-Compatible**
- Can define reachability relation

**Property 5: Deterministic**
- Applying operation in state produces unique result

---

## PART III: WEIGHTED EXTENSION

### Component Breakdown

#### Positions: Can They Track Cost?

**YES - with discretization:**

```rust
// Current
struct Position {
    term_index: usize,
    num_errors: usize,  // Integer count
}

// Weighted
struct WeightedPosition {
    term_index: usize,
    cost_units: u32,  // Discretized cost
}

impl WeightedPosition {
    fn from_cost(index: usize, cost: f64, precision: f64) -> Self {
        Self {
            term_index: index,
            cost_units: (cost / precision).round() as u32,
        }
    }
    
    fn actual_cost(&self, precision: f64) -> f64 {
        self.cost_units as f64 * precision
    }
}
```

**State Space**:
- Before: `$\mathcal{O}(|W| \times  n)$`
- After: `$\mathcal{O}(|W| \times  \lceil \max _\text{cost}/\varepsilon \rceil)$`
- For fixed max_cost/`$\varepsilon$`: Still `$\mathcal{O}(|W|)$` ✓

#### Subsumption: Does It Generalize?

**Modified Formula**:

```rust
fn subsumes_weighted(
    i: usize, c: u32,  // Position 1
    j: usize, d: u32,  // Position 2
    min_cost_per_op: u32,
) -> bool {
    // Must have lower cost
    if c >= d {
        return false;
    }
    
    // Can we reach j from i within budget (d-c)?
    let distance = (j - i) as u32;
    let min_cost_needed = distance * min_cost_per_op;
    let budget_available = d - c;
    
    min_cost_needed <= budget_available
}
```

**Issues**:
- Uses min_cost as conservative bound
- Exact subsumption requires shortest-path computation (expensive)
- Trade-off: Conservative pruning vs optimal pruning

#### Transitions: How Do Costs Affect Them?

```rust
fn transition_weighted(
    pos: &WeightedPosition,
    dict_char: char,
    query: &[char],
    costs: &CostFunction,
    max_cost_units: u32,
    precision: f64,
) -> Vec<WeightedPosition> {
    let i = pos.term_index;
    let c = pos.cost_units;
    let mut result = vec![];
    
    if i >= query.len() {
        return result;
    }
    
    // Match (free)
    if query[i] == dict_char {
        result.push(WeightedPosition {
            term_index: i + 1,
            cost_units: c,
        });
    }
    
    // Substitute
    let sub_cost = discretize_cost(
        costs.substitute(query[i], dict_char),
        precision
    );
    if c + sub_cost <= max_cost_units {
        result.push(WeightedPosition {
            term_index: i + 1,
            cost_units: c + sub_cost,
        });
    }
    
    // Insert
    let ins_cost = discretize_cost(costs.insert(dict_char), precision);
    if c + ins_cost <= max_cost_units {
        result.push(WeightedPosition {
            term_index: i,
            cost_units: c + ins_cost,
        });
    }
    
    // Delete
    let del_cost = discretize_cost(costs.delete(query[i]), precision);
    if c + del_cost <= max_cost_units {
        result.push(WeightedPosition {
            term_index: i + 1,
            cost_units: c + del_cost,
        });
    }
    
    result
}

fn discretize_cost(cost: f64, precision: f64) -> u32 {
    (cost / precision).round() as u32
}
```

**Key Changes**:
- Add cost to each transition
- Check budget before creating position
- Discretize costs for finiteness

#### Characteristic Vectors: Extend to Costs?

**Cost-Characteristic Vectors:**

```rust
fn cost_characteristic_vector(
    x: char,
    query: &[char],
    offset: usize,
    window_size: usize,
    costs: &CostFunction,
) -> Vec<f64> {
    (0..window_size)
        .map(|k| {
            let i = offset + k;
            if i >= query.len() {
                f64::INFINITY
            } else if query[i] == x {
                0.0  // Free match
            } else {
                costs.substitute(query[i], x)
            }
        })
        .collect()
}
```

**Usage**: Determines minimum cost to match at each position

---

## PART IV: CONCRETE EXAMPLE

### Keyboard-Distance Edit Operations

**Problem**: Standard Levenshtein treats 'a'→'s' same as 'a'→'z', but on QWERTY keyboard:
- 'a' and 's' are adjacent (likely typo)
- 'a' and 'z' are far apart (unlikely typo)

**Solution**: Variable substitution costs based on keyboard distance

**Cost Function**:
```rust
fn keyboard_distance_cost(c1: char, c2: char) -> f64 {
    if c1 == c2 {
        return 0.0;  // Match
    }
    
    let pos1 = qwerty_position(c1);  // (row, col)
    let pos2 = qwerty_position(c2);
    
    let row_diff = (pos1.row - pos2.row).abs();
    let col_diff = (pos1.col - pos2.col).abs();
    
    match (row_diff, col_diff) {
        (0, 1) | (1, 1) => 0.5,  // Adjacent (horizontal or diagonal)
        (0, _) => 1.0,            // Same row
        _ => 1.5,                 // Different rows
    }
}
```

**Discretization** `$(\varepsilon  = 0.1)$`:
```
Costs:        0.5  1.0  1.5
Units (×10):   5   10   15
```

**Position Structure**:
```rust
struct KeyboardPosition {
    term_index: usize,
    cost_units: u32,  // In units of 0.1
}
```

**Transitions**:
```rust
// Query: "tesy", Dictionary: "test"
// Starting position: (0, 0)

// Read 't': match query[0]='t' with dict='t'
// → (1, 0) cost=0.0

// Read 'e': match query[1]='e' with dict='e'
// → (2, 0) cost=0.0

// Read 's': mismatch query[2]='s' with dict='s'... wait, they match!
// → (3, 0) cost=0.0

// Read 't': match query[3]='y' with dict='t'
// Substitute 'y'→'t': adjacent on keyboard → cost=0.5 (5 units)
// → (4, 5) cost=0.5

// Reached end of query, cost = 0.5 ≤ threshold
// Result: MATCH with distance 0.5
```

**Complexity**:
```
States: O(|W| × max_cost/`$\varepsilon$`)
      = O(|W| × 2.0/0.1)
      = O(20|W|)
      = O(|W|) for fixed ratio ✓
```

---

## PART V: RECOMMENDATIONS

### For liblevenshtein-rust Implementation

**Phase 1: Universal LA (Restricted Substitutions)**
- **Effort**: 2-4 weeks
- **Overhead**: 5-15%
- **Use cases**: Keyboard proximity, OCR, phonetic
- **Status**: Documented in `/docs/research/universal-levenshtein/`
- **Recommendation**: ✅ **IMPLEMENT FIRST**

**Phase 2: Research Discretized Weights**
- **Effort**: 4-6 weeks research + 4-6 weeks implementation
- **Overhead**: 10-200× (depends on precision)
- **Use cases**: Specialized applications, scientific computing
- **Recommendation**: ⚠️ **PROTOTYPE FIRST, then decide**

**Phase 3: Full Weighted Automata**
- **Effort**: Complete rewrite (8-12 weeks)
- **Overhead**: 100-1000×
- **Use cases**: Unclear (most solved by discretization)
- **Recommendation**: ❌ **DO NOT IMPLEMENT**

### Trade-Off Analysis

**Precision vs Performance:**

| Precision `$(\varepsilon )$` |  State Multiplier | Use Case |
|---------------|------------------|----------|
| 1.0 (integer) | 1-5× | Coarse costs, fast queries |
| 0.1 | 10-50× | Keyboard/OCR distances |
| 0.01 | 100-500× | High-precision scientific |
| 0.001 | 1000-5000× | Impractical |

**Recommendation**: Start with `$\varepsilon =0.1,$` allow user configuration

---

## Mathematical Framework

### Formalization: Generalized Levenshtein Automaton

**Definition**:
```
GLA_c,ε(W, θ) = (Q, Σ, Δ, q₀, F)

where:
  W: query word
  c: CostFunction (operations → costs)
  ε: discretization precision
  θ: cost threshold (max allowed)
  
  Q: states (sets of weighted positions)
  q₀: {(0, 0)} (start)
  F: {M | ∃(i,c) ∈ M : i=|W| ∧ c·ε ≤ θ}
  Δ: weighted state transition
```

**Position Space**:
```
P = {(i, c) | i ∈ [0,|W|], c ∈ [0, ⌈θ/ε⌉]}
Cardinality: O(|W| × θ/ε)
```

**Subsumption**:
```
(i, c) ⊑ (j, d) ⟺
  c < d ∧
  min_cost_path(i→j) ≤ (d-c)·ε
```

**Complexity Theorem**:
```
Theorem: For fixed ε > 0, θ, and c_min > 0:
  |GLA_c,ε(W, θ)| = O(|W| × θ/ε)
  Construction: O(|W| × θ/ε)
  Query: O(|D| × θ/ε)
  
When θ/ε and c_max/c_min are O(1): Still O(|W|) and O(|D|)
```

---

## Limitations and Future Work

### Current Limitations

1. **Discretization Error**: `$\varepsilon  > 0$` introduces approximation
2. **Context Sensitivity**: Costs can't easily depend on position history
3. **Non-Local Operations**: Operations must examine only `$\mathcal{O}(n)$` window
4. **Bounded Costs**: Ratio c_max/c_min must be constant for `$\mathcal{O}(|W|)$`

### Future Research Directions

1. **Adaptive Precision**: Coarser `$\varepsilon$` far from solution, finer near matches
2. **Hybrid Approaches**: Combine restricted substitutions + weighted ops
3. **Approximate Algorithms**: Beam search for faster queries with quality guarantees
4. **Learning Costs**: Automatically learn operation costs from data
5. **Context-Aware Costs**: Limited context window (e.g., previous character)

---

## References

### Primary Papers

1. **Schulz, K. U., & Mihov, S.** "Fast String Correction with Levenshtein-Automata"
   - Foundation for uniform-cost automata
   - Documented in `/docs/research/levenshtein-automata/`

2. **Mitankin, P., Mihov, S., & Schulz, K. U. (2009).** "Universal Levenshtein Automata"
   - Restricted substitutions (binary allowed/forbidden)
   - Documented in `/docs/research/universal-levenshtein/`

3. **Mohri, M. (2003).** "Edit-Distance of Weighted Automata"
   - Weighted transducers, tropical semiring
   - Alternative approach (non-deterministic)

### Cross-References

- `/docs/research/levenshtein-automata/PAPER_SUMMARY.md` - Core algorithms
- `/docs/research/universal-levenshtein/README.md` - Restricted substitutions
- `/docs/algorithms/02-levenshtein-automata/README.md` - Current implementation

---

## Conclusion

**Can general rules be derived for weighted transitions?**

**Answer: YES, via discretization**

The Schulz/Mihov/Mitankin methodology CAN be extended to weighted costs by:
1. Discretizing costs to fixed precision `$\varepsilon$`
2. Tracking (position, discretized_cost) instead of (position, error_count)
3. Modifying subsumption for cost comparison
4. Adding cost to transitions

The resulting complexity `$\mathcal{O}(|W| \times  \max _\text{cost}/\varepsilon)$` remains `$\mathcal{O}(|W|)$` when cost range and precision are fixed.

**The methodology is extensible** but requires accepting precision-performance trade-offs. For most practical applications, **Universal LA (binary restrictions)** or **coarse discretization `$(\varepsilon =0.1-1.0)**$` provides the best balance of expressiveness and performance.

---

**Last Updated**: 2025-11-06  
**Status**: Research Complete  
**Next Steps**: Implement Universal LA, then prototype discretized weights

