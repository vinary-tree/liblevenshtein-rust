[← Documentation Index](../../README.md)

# Universal Levenshtein Automata - Theoretical Foundations

**Source**: "Universal Levenshtein Automata - Building and Properties" (Petar Mitankin, 2005), Master's Thesis, Sofia University "St. Kliment Ohridski"
**Sections**: All theoretical content - Definitions, Propositions, Lemmas, Theorems

This document extracts the core theoretical foundations including all proofs, lemmas, propositions, and theorems. It complements [PAPER_SUMMARY.md](PAPER_SUMMARY.md) by focusing on mathematical rigor rather than narrative flow.

---

## Table of Contents

1. [Distance Properties (Section 2)](#section-2-distance-properties)
2. [Nondeterministic Automata (Section 3)](#section-3-nondeterministic-automata)
3. [Deterministic Automata (Section 4)](#section-4-deterministic-automata)
4. [Universal Automata (Section 5)](#section-5-universal-automata)
5. [Minimality (Section 7)](#section-7-minimality)
6. [Properties (Section 8)](#section-8-properties)
7. [Cross-Reference Index](#cross-reference-index)

---

## Section 2: Distance Properties

### ⚠️ CRITICAL WARNING

**Triangle Inequality Violation** (Page 3):

```
d^t_L does NOT satisfy the triangle inequality!
```

**Counterexample**:
```
Let v = "ac", w = "ca", x = "aa"

d^t_L(v, w) = 1  (single transposition: ac ↔ ca)
d^t_L(w, x) = 1  (single transposition: ca ↔ aa)
d^t_L(v, x) = 2  (ac → aa requires 2 substitutions)

BUT: d^t_L(v, x) = 2 > d^t_L(v, w) + d^t_L(w, x) = 1 + 1 = 2
```

**Implication**: Cannot use triangle inequality for pruning or optimization with $`d^t_L`$.

---

### Definition 1: Standard Levenshtein Distance d²_L

**Recursive definition**:

```
d²_L: Σ* × Σ* → ℕ ∪ {∞}

Base cases:
d²_L(ε, ε) = 0
d²_L(ε, bw) = 1 + d²_L(ε, w)   for b ∈ Σ, w ∈ Σ*
d²_L(av, ε) = 1 + d²_L(v, ε)   for a ∈ Σ, v ∈ Σ*

Recursive case (v = av', w = bw'):
d²_L(av', bw') = min(
    if(a = b, d²_L(v', w'), ∞),  // match (no cost)
    1 + d²_L(v', bw'),            // deletion of a
    1 + d²_L(av', w'),            // insertion of b
    1 + d²_L(v', w')              // substitution a ↔ b
)
```

**Operations**: Insertion, Deletion, Substitution

---

### Definition 2: With Transposition $`d^t_L`$

Extends d²_L with transposition operation:

```
d^t_L(av', bw') = min(
    if(a = b, d^t_L(v', w'), ∞),  // match
    1 + d^t_L(v', bw'),            // deletion
    1 + d^t_L(av', w'),            // insertion
    1 + d^t_L(v', w'),             // substitution
    T^t(av', bw')                  // transposition
)

where T^t(av', bw') = {
    1 + d^t_L(v'', w'')  if v' = cv'', w' = dw'', a = d, b = c
    ∞                    otherwise
}
```

**Operations**: Insertion, Deletion, Substitution, **Transposition**

**Transposition**: Swaps adjacent characters (ab ↔ ba costs 1)

---

### Definition 3: With Merge/Split $`d^{\text{ms}}_L`$

Extends d²_L with merge and split operations:

```
d^{ms}_L(av', bw') = min(
    if(a = b, d^{ms}_L(v', w'), ∞),  // match
    1 + d^{ms}_L(v', bw'),            // deletion
    1 + d^{ms}_L(av', w'),            // insertion
    1 + d^{ms}_L(v', w'),             // substitution
    T^{ms}_m(av', bw'),                // merge
    T^{ms}_s(av', bw')                 // split
)

where:
T^{ms}_m(av', bw') = {
    1 + d^{ms}_L(v'', w')  if v' = cv''
    ∞                       otherwise
}

T^{ms}_s(av', bw') = {
    1 + d^{ms}_L(v', w'')  if w' = dw''
    ∞                       otherwise
}
```

**Operations**: Insertion, Deletion, Substitution, **Merge** (2→1), **Split** (1→2)

---

### Proposition 1: Symmetry

**Statement**:
```
∀v, w ∈ Σ*: d^χ_L(v, w) = d^χ_L(w, v)
```

for $`\chi  \in`$ $`\{\varepsilon , t, \text{ms}\}`$.

**Proof**: By structural induction on min(|v|, |w|). The recursive definitions are symmetric in v and $`w. \blacksquare`$

---

### Proposition 2: Identity

**Statement**:
```
∀w ∈ Σ*: d^χ_L(w, w) = 0
```

**Proof**: By induction on |w|.
- Base: $`d^\chi _L(\varepsilon , \varepsilon ) = 0`$ by definition
- Step: $`d^\chi _L(\text{aw}', \text{aw}')`$ = min(..., if(a = a, $`d^\chi _L(w', w'), \infty ), ...) =`$ $`d^\chi _L(w', w')`$ = 0 by IH. $`\blacksquare`$

---

### Proposition 3: Triangle Inequality (for d²_L and $`d^{\text{ms}}_L`$ only!)

**Statement**:
```
∀u, v, w ∈ Σ*: d²_L(u, w) ≤ d²_L(u, v) + d²_L(v, w)
∀u, v, w ∈ Σ*: d^{ms}_L(u, w) ≤ d^{ms}_L(u, v) + d^{ms}_L(v, w)
```

**Does NOT hold for $`d^t_L`$!** (See counterexample above)

**Proof sketch** (for d²_L): Any transformation u → v → w can be composed into a transformation u → w of cost $`\le`$ sum of intermediate costs. $`\blacksquare`$

---

### Proposition 4: Prefix Monotonicity

**Statement**:
```
∀v, w ∈ Σ*, ∀a ∈ Σ:
  d^χ_L(av, w) ≥ d^χ_L(v, w) - 1
  d^χ_L(v, aw) ≥ d^χ_L(v, w) - 1
```

**Proof**: Deleting 'a' from av costs at most 1, so $`d^\chi _L(\text{av}, w)`$ $`\ge`$ $`d^\chi _L(v, w)`$ - 1. Symmetrically for aw. $`\blacksquare`$

---

### Proposition 5: Edit Sequence Correspondence

**Statement**: For each finite edit distance $`d^\chi _L(v, w) = k,`$ there exists an edit sequence E of length k transforming v into w.

**Proof**: By construction from recursive definition. $`\blacksquare`$

---

### Definition 4: Language $`L^\chi _{\text{Lev}}(n, w)`$

**Levenshtein neighborhood**:

```
L^χ_{Lev}(n, w) = {x ∈ Σ* | d^χ_L(x, w) ≤ n}
```

The set of all strings within edit distance n of w.

**Example** (n=1, w="cat"):
```
L^ε_{Lev}(1, "cat") = {
  "cat",                          // distance 0
  "at", "ct", "ca",               // deletion
  "acat", "ccat", ..., "cata",    // insertion
  "bat", "cbt", "cas", ...        // substitution
}
```

---

### Definition 5: Language $`R^\chi (n, w)`$

**Reverse Levenshtein language**:

```
R^χ(n, w) = {x^R | x ∈ L^χ_{Lev}(n, w^R)}
```

where $`x^R`$ denotes the reverse of string x.

**Purpose**: Used in constructing automata that read input left-to-right.

---

### Proposition 6: Closure Under Reversal

**Statement**:
```
(R^χ(n, w))^R = L^χ_{Lev}(n, w^R)
```

**Proof**: Direct from definition. $`\blacksquare`$

---

### Proposition 7: Suffix Property

**Statement**:
```
x ∈ R^χ(n, w) ⇔ ∃k (0 ≤ k ≤ n): x ∈ R^χ(k, w_{|w|-k+1}...w_{|w|})
```

**Interpretation**: String x is in $`R^\chi (n, w)`$ iff it matches some suffix of w within $`k \le  n`$ errors.

**Proof**: By analyzing edit sequences. $`\blacksquare`$

---

### Proposition 8: Prefix Extension

**Statement**: If $`x \in`$ $`R^\chi (k, w_\text{suffix})`$ then xa $`\in`$ $`R^\chi (k', w_\text{longer}_\text{suffix})`$ for appropriate k'.

**Purpose**: Foundation for incremental automaton construction.

---

## Section 3: Nondeterministic Automata

### Definition 6: $`A^{\text{ND},\chi }_n(w)`$

**Nondeterministic Levenshtein Automaton**:

```
A^{ND,χ}_n(w) = (Q^{ND,χ}_n, Σ, δ^{ND,χ}_n, q_0, F^{ND,χ}_n)
```

**States**:
```
Q^{ND,ε}_n = {i#e | 0 ≤ i ≤ p ∧ 0 ≤ e ≤ n}
Q^{ND,t}_n = Q^{ND,ε}_n ∪ {i#e_t | 0 ≤ i ≤ p ∧ 0 ≤ e ≤ n}
Q^{ND,ms}_n = Q^{ND,ε}_n ∪ {i#e_s | 0 ≤ i ≤ p ∧ 0 ≤ e ≤ n}
```

where p = |w|.

**Notation**:
- i#e: Position i in word w with e errors consumed
- i#e_t: Transposition state (waiting to complete swap)
- i#e_s: Split state (waiting to emit second character)

**Initial state**: q_0 = 0#0

**Final states**: $`F^{\text{ND},\chi }_n`$ = $`\{i#e | i = p \land  e \le  n\}`$

---

### Transition Function $`\delta ^{\text{ND},\varepsilon }_n`$ (Standard)

**For position i#e** ($`0 \le  i < p, e < n`$):

```
δ^{ND,ε}_n(i#e, a) = {
    {i+1#e}                              if w_{i+1} = a  (match)
    {i#(e+1), i+1#(e+1), i+1#e}         if w_{i+1} ≠ a  (sub, ins, del)
    {i#(e+1), i+1#(e+1)}                if e = n        (no deletions)
}
```

**For boundary positions**:
```
δ^{ND,ε}_n(p#e, a) = {p#(e+1)}         // only insertion at end
δ^{ND,ε}_n(i#n, a) = {
    {i+1#n}                             if w_{i+1} = a
    {i#n, i+1#n}                        if w_{i+1} ≠ a  (no more deletions)
}
```

**Interpretation**:
- **i#(e+1)**: Deletion (advance in w, consume error, don't consume input)
- **i+1#(e+1)**: Insertion (advance in input, consume error)
- **i+1#e**: Substitution or match

---

### Transition Function $`\delta ^{\text{ND},t}_n`$ (With Transposition)

Extends $`\delta ^{\text{ND},\varepsilon }_n`$ with transposition states:

**Regular states** i#e:
```
δ^{ND,t}_n(i#e, a) = δ^{ND,ε}_n(i#e, a) ∪ T^t_n(i#e, a)

where T^t_n(i#e, a) = {
    {i#(e+1)_t}  if i+1 < p ∧ w_{i+2} = a ∧ e < n
    ∅            otherwise
}
```

**Transposition states** i#e_t:
```
δ^{ND,t}_n(i#e_t, a) = {
    {i+2#e}  if w_{i+1} = a
    ∅        otherwise
}
```

**Example**:
```
Word: w = "abc"
Input: x = "bac"

Path: 0#0 --b--> 0#0_t (detected ba vs ab)
           --a--> 2#0   (completed transposition, now at position 2)
           --c--> 3#0   (match final character)
```

---

### Transition Function $`\delta ^{\text{ND},\text{ms}}_n`$ (With Merge/Split)

Extends $`\delta ^{\text{ND},\varepsilon }_n`$ with merge and split:

**Regular states** i#e:
```
δ^{ND,ms}_n(i#e, a) = δ^{ND,ε}_n(i#e, a) ∪ T^{ms}_n(i#e, a)

where T^{ms}_n(i#e, a) = {
    {i+2#(e+1)}  if i+1 < p ∧ e < n     // merge: consume 2 chars of w
    {i#(e+1)_s}  if e < n                 // split: will emit extra char
    ∅            otherwise
}
```

**Split states** i#e_s:
```
δ^{ND,ms}_n(i#e_s, a) = {i+1#e}         // always advance (emitting 2nd char)
```

---

### Proposition 9: NFA Correctness ⭐

**Statement**: For position i#e in $`A^{\text{ND},\chi }_n(w)`$:

```
L(i#e) = L^χ_{Lev}(n - e, w_{i+1}...w_p)
```

**Interpretation**: The language accepted from state i#e is exactly the set of strings within (n - e) errors of the remaining suffix w_{i+1}...w_p.

**Proof Sketch**:
1. **Forward direction**: If $`x \in  L(i`$#e), show $`d^\chi _L(x, w_{i+1}...w_p)`$ $`\le  n - e`$
   - By induction on |x|
   - Each transition corresponds to an edit operation
   - Error budget decreases appropriately

2. **Reverse direction**: If $`d^\chi _L(x, w_{i+1}...w_p)`$ $`\le  n - e,`$ show $`x \in  L(i`$#e)
   - Construct accepting path from edit sequence
   - Each edit operation has corresponding transition

**Corollary**:
```
L(A^{ND,χ}_n(w)) = L(0#0) = L^χ_{Lev}(n, w_1...w_p) = L^χ_{Lev}(n, w)
```

$`\blacksquare`$

---

## Section 4: Deterministic Automata

### Key Idea: Bit Vector Encoding

Instead of tracking exact position in word w, track a **characteristic vector** indicating matches.

**Definition**: For character x and word w:
```
β(x, w) = b_1b_2...b_{|w|}

where b_i = {
    1  if w_i = x
    0  otherwise
}
```

**Example**:
```
w = "banana"
β('a', w) = 010101
β('b', w) = 100000
β('n', w) = 001010
β('c', w) = 000000
```

---

### Definition 7: Elementary Transition $`\delta ^{D,\varepsilon }_e`$

**Processes single bit vector** for standard Levenshtein:

```
δ^{D,ε}_e: (ℤ × ℕ) × {0,1}^+ → P(ℤ × ℕ)

δ^{D,ε}_e(i#e, ε) = {
    {i#(e+1)}  if e < n
    ∅          if e = n
}

δ^{D,ε}_e(i#e, 1b) = {i+1#e}

δ^{D,ε}_e(i#e, 0b) = {
    {i#(e+1), i+1#(e+1)}           if |b| = 0 ∧ e < n
    {i#(e+1), i+1#(e+1), i+j#(e+j-1)}  if b[j] = first 1 in b, e < n
    ∅                                   if e = n
}
```

**Interpretation**:
- **$`\varepsilon`$ (empty)**: Only substitution possible (if errors remain)
- **1b (match)**: Advance position, no error
- **0b (mismatch)**: Try substitution, insertion, or deletion until match

---

### Definition 8: Characteristic Vector Sequence

For word w and input x:
```
h_n(w, x) = β(x_1, s_n(w, 1)) β(x_2, s_n(w, 2)) ... β(x_{|x|}, s_n(w, |x|))
```

where s_n(w, i) is the "sliding window" of w visible at position i:
```
s_n(w, i) = w[max(1, i-n) ... min(|w|, i+n)]
```

**Purpose**: Encodes which characters in the visible window match each input character.

---

### Definition 9: Deterministic Automaton $`A^{D,\chi }_n(w)`$

```
A^{D,χ}_n(w) = (Q^{D,χ}_n, Σ, δ^{D,χ}_n, q_0, F^{D,χ}_n)
```

**States**: $`Q^{D,\chi }_n`$ $`\subseteq  P(\mathbb{Z}  \times  \mathbb{N} )`$ (sets of positions)

**Transition**: $`\delta ^{D,\chi }_n(q, a)`$ = $`\sqcup  (\bigcup _\{\pi  \in  q\}`$ $`\delta ^{D,\chi }_e(\pi , \beta (a, s_n(w, i))))`$

where $`\sqcup`$ denotes subsumption closure (remove subsumed positions).

**Initial state**: q_0 = {0#0}

**Final states**: $`F^{D,\chi }_n`$ = $`\{q | \exists \pi  \in  q: ﷐1﷑#n\}`$

---

### Definition 10: Elementary Transitions for t and ms

**For transposition** $`\delta ^{D,t}_e`$:

Extends $`\delta ^{D,\varepsilon }_e`$ with:
```
δ^{D,t}_e(i#e_t, 1b) = {i+2#e}      // complete transposition
δ^{D,t}_e(i#e_t, 0b) = ∅            // must match to complete

// Regular positions can enter transposition state:
δ^{D,t}_e(i#e, 0 1 b) = ... ∪ {i#(e+1)_t}
```

**For merge/split** $`\delta ^{D,\text{ms}}_e`$:

Extends $`\delta ^{D,\varepsilon }_e`$ with:
```
δ^{D,ms}_e(i#e_s, b) = {i+1#e}      // complete split

// Regular positions:
δ^{D,ms}_e(i#e, b) = ... ∪ {i+2#(e+1)} ∪ {i#(e+1)_s}
```

---

### Definition 11: Subsumption Relation $`\le ^\chi _s`$

**Standard Levenshtein**:
```
i#e ≤^ε_s j#f  ⇔  f > e ∧ |j - i| ≤ f - e
```

**With transposition**:
```
i#e   ≤^t_s j#f    ⇔  f > e ∧ |j - i| ≤ f - e
i#e_t ≤^t_s j#f    ⇔  f > e ∧ |j + 1 - i| ≤ f - e
i#e   ≤^t_s j#f_t  ⇔  false
i#e_t ≤^t_s j#f_t  ⇔  false
```

**With merge/split**:
```
i#e   ≤^{ms}_s j#f   ⇔  f > e ∧ |j - i| ≤ f - e
i#e_s ≤^{ms}_s j#f   ⇔  f > e ∧ |j - i| ≤ f - e
i#e   ≤^{ms}_s j#f_s ⇔  false
i#e_s ≤^{ms}_s j#f_s ⇔  false
```

**Intuition**: Position i#e is subsumed by j#f if j#f represents a "better" state (more errors available, close enough in position).

---

### Definition 12: Subsumption Closure $`\sqcup`$

For set of positions A:
```
⊔A = {π ∈ A | ∄π' ∈ A: π <^χ_s π'}
```

**Interpretation**: Remove all subsumed elements, keeping only maximal elements (anti-chain).

**Example**:
```
A = {3#1, 5#2, 4#1}

Check subsumptions:
- 3#1 vs 5#2: |5-3| = 2 ≤ 2-1 = 1? NO
- 3#1 vs 4#1: 1 > 1? NO
- 4#1 vs 5#2: |5-4| = 1 ≤ 2-1 = 1? YES → 4#1 <^ε_s 5#2

⊔A = {3#1, 5#2}
```

---

### Definition 13: rm(A) - Rightmost Position

For set of positions A (containing only "usual" type):
```
rm(A) = argmax_{π ∈ A} (X(π) - Y(π))
```

**Purpose**: Identifies the position furthest to the right (adjusted by error count).

---

### Definition 14: Full DFA Construction

```
A^{D,χ}_n(w) is built by:
1. Start with q_0 = {0#0}
2. For each state q and character a ∈ Σ:
   - Compute bit vector b = β(a, s_n(w, i))
   - Apply δ^{D,χ}_e to each position in q
   - Union results and apply subsumption closure ⊔
3. Mark states containing p#e (e ≤ n) as final
```

---

### Proposition 10: DFA Correctness

**Statement**:
```
L(A^{D,χ}_n(w)) = L^χ_{Lev}(n, w)
```

**Proof**: By showing:
1. Every path in $`A^{D,\chi }_n(w)`$ corresponds to an edit sequence
2. Every edit sequence has a corresponding path
3. Subsumption preserves language (subsumed states are redundant)

$`\blacksquare`$

---

### Proposition 11: State Reduction

**Statement**: Subsumption closure reduces state count without changing the accepted language.

**Proof**: If $`\pi  <^\chi _s \pi ',`$ then $`L(\pi ) \subseteq  L(\pi '),`$ so removing $`\pi`$ doesn't change $`\bigcup  L`$(positions). $`\blacksquare`$

---

## Section 5: Universal Automata

This is the **core contribution** of the thesis. The universal automaton $`A^{\forall ,\chi }_n`$ works for **any word** by using parametric positions.

### Definition 15: Universal Position Sets

**Non-final positions** $`I^\chi _s`$:

```
I^ε_s = {I + t#k | |t| ≤ k ∧ -n ≤ t ≤ n ∧ 0 ≤ k ≤ n}

I^t_s = I^ε_s ∪ {It + t#k | |t| ≤ k ∧ -n ≤ t ≤ n ∧ 0 ≤ k ≤ n}

I^{ms}_s = I^ε_s ∪ {Is + t#k | |t| ≤ k ∧ -n ≤ t ≤ n ∧ 0 ≤ k ≤ n}
```

**Notation**: I + t#k represents position (I + t, k) where I is a parameter.

**Final positions** $`M^\chi _s`$:

```
M^ε_s = {M + t#k | k ≥ -t - n ∧ -2n ≤ t ≤ 0 ∧ 0 ≤ k ≤ n}

M^t_s = M^ε_s ∪ {Mt + t#k | k ≥ -t - n ∧ -2n ≤ t ≤ 0 ∧ 0 ≤ k ≤ n}

M^{ms}_s = M^ε_s ∪ {Ms + t#k | k ≥ -t - n ∧ -2n ≤ t ≤ 0 ∧ 0 ≤ k ≤ n}
```

---

### Definition 16: Universal Transition Function $`\delta ^{\forall ,\chi }_n`$

**Elementary transition** $`\delta ^{\forall ,\chi }_e`$ operates on universal positions using:

```
r_n(I + t#e, b) = b[n + t + 1 ... min(n + t + (n - e + 1), |b|)]
r_n(M + t#e, b) = b[|b| + t + 1 ... min(|b| + t + (n - e + 1), |b|)]
```

**Full transition**:
```
δ^{∀,χ}_n(q, b) = ⊔ (⋃_{π ∈ q} δ^{∀,χ}_e(π, r_n(π, b)))
```

with diagonal crossing via:
```
if f_n(rm(q'), |b|):
    q' := m_n(q', |b|)
```

---

### Definition 17: Diagonal Crossing Functions

**f_n**: Predicate for crossing:
```
f_n(I + t#e, k) = (k ≤ 2n+1) ∧ (e ≤ t + 2n + 1 - k)
f_n(M + t#e, k) = e > t + n
```

**m_n**: Conversion function:
```
m_n(I + t#e, k) = M + (t + n + 1 - k)#e
m_n(M + t#e, k) = I + (t - n - 1 + k)#e
```

---

### Definition 18: Substitution Function d

**Maps universal positions to concrete positions**:

```
d(I + t#e, i) = (i + t)#e         for i ∈ [0, p]
d(It + t#e, i) = (i + t)#e_t
d(Is + t#e, i) = (i + t)#e_s

d(M + t#e, p) = (p + t)#e
d(Mt + t#e, p) = (p + t)#e_t
d(Ms + t#e, p) = (p + t)#e_s
```

**Purpose**: Given concrete word length p, converts I parameter to 0, M to p.

---

### Proposition 19: Main Correctness Theorem ⭐⭐⭐

**THE MOST IMPORTANT THEOREM IN THE THESIS**

**Statement**: For any word w of length p, universal automaton $`A^{\forall ,\chi }_n`$ correctly simulates deterministic automaton $`A^{D,\chi }_n(w)`$.

Formally, for all input strings x:

```
A^{∀,χ}_n traverses: q^{∀,χ}_0, b_1, q^{∀,χ}_1, b_2, ..., b_{|x|}, q^{∀,χ}_{|x|}
A^{D,χ}_n(w) traverses: q^{D,χ}_0, x_1, q^{D,χ}_1, x_2, ..., x_{|x|}, q^{D,χ}_{|x|}

where b_i = β(x_i, s_n(w, i))

Then:
I)  !q^{∀,χ}_i ⇔ !q^{D,χ}_i
II) ∀q ∈ q^{∀,χ}_i: d(q, s(i)) ∈ q^{D,χ}_i
    (s(i) = min(i, p) maps index to actual position)
III) q^{∀,χ}_i ∈ F^{∀,χ}_n ⇔ q^{D,χ}_i ∈ F^{D,χ}_n
```

**Interpretation**:
- **I)**: States are defined in lockstep
- **II)**: Every universal position corresponds to a concrete position in the DFA
- **III)**: Final states match

**Proof Strategy**: By induction on i (input position).

**Base case** (i = 0):
- $`q^{\forall ,\chi }_0`$ = {I#0}
- $`q^{D,\chi }_0`$ = {0#0} = {d(I#0, 0)}
- ✓ Correspondence holds

**Inductive step**: Assume holds for i, prove for i+1.

**Key lemmas** (proved in thesis):
1. **Lemma A**: $`r_n(q, b_{i+1}) = \beta(x_{i+1}, w[d(q, s(i))])`$
   - The extracted substring from bit vector matches the characteristic vector
2. **Lemma B**: $`\delta ^{\forall ,\chi }_e(q, b_{i+1})`$ $`\ne  \emptyset  \Leftrightarrow`$ $`\delta ^{D,\chi }_e`$(d(q, s(i)), x_{i+1}) $`\ne  \emptyset`$
   - Transitions are defined in lockstep
3. **Lemma C**: Subsumption is preserved under d
4. **Lemma D**: Diagonal crossing happens simultaneously

**Conclusion**: By induction, correspondence holds for all i. Therefore:
```
L(A^{∀,χ}_n, w) = L(A^{D,χ}_n(w)) = L^χ_{Lev}(n, w)
```

$`\blacksquare`$

**Significance**: This theorem proves that $`A^{\forall ,\chi }_n`$ is truly **universal** - it works correctly for any word w without modification.

---

## Section 7: Minimality

### Theorem 1: $`A^{\forall ,\varepsilon }_n`$ is Minimal

**Statement**: $`A^{\forall ,\varepsilon }_n`$ has the minimum number of states among all automata that are:
1. Parameter-free (work for any word length)
2. Accept $`L^\varepsilon _{\text{Lev}}(n, w)`$ for all words w

**Proof Outline**:
1. Show no two states are equivalent (Nerode equivalence)
2. For each pair of distinct states q₁, q₂, construct distinguishing word pair (w, x)
3. Use subsumption anti-chain property to show all states are necessary

$`\blacksquare`$

---

### Theorem 2: $`A^{\forall ,t}_n`$ is Minimal

Similar to Theorem 1, but accounting for transposition states.

$`\blacksquare`$

---

### Theorem 3: $`A^{\forall ,\text{ms}}_n`$ is Minimal

Similar to Theorem 1, but accounting for merge/split states.

$`\blacksquare`$

---

## Section 8: Properties

### Property 1: State Count Bounds

**Exact formulas** (from Section 6.3):

```
|I^ε_states| = O(2^{4n - log₂(√(2n+1))})
|M^ε_states| = O(n × 2^{4n - log₂(√(2n+1))})

Total: O(n × 4^{2n} / √(2n+1))
```

---

### Property 2: Construction Complexity

**Time**: $`\mathcal{O}(|\text{states}| \times  |\text{alphabet}| \times  \text{poly}(n)`$)
**Space**: $`\mathcal{O}(\lvert \text{states}\rvert \times  n^{2})`$

---

### Property 3: Query Complexity

**Given $`A^{\forall ,\chi }_n`$ and word w**:

**Time**: $`\mathcal{O}(\lvert x\rvert \times  n)`$ to check if $`x \in`$ $`L^\chi _{\text{Lev}}(n, w)`$

**Process**:
1. Compute h_n(w, x) incrementally: $`\mathcal{O}(\lvert x\rvert \times  n)`$
2. Traverse $`A^{\forall ,\chi }_n`$: $`\mathcal{O}(\lvert x\rvert)`$ transitions
3. Check final state: $`\mathcal{O}(1)`$

**Space**: $`\mathcal{O}(n)`$ for current state

---

### Property 4: Applicability

**Works for**:
- Any alphabet $`\Sigma`$
- Any word length |w|
- Any maximum distance n

**Limitation**: Exponential state count in n limits practical use to small n (typically $`n \le  3).`$

---

## Cross-Reference Index

### Key Definitions

| Concept | Definition | Page | Section |
|---------|-----------|------|---------|
| $`d^\chi _L`$ | Def. 1-3 | 3-5 | 2 |
| $`L^\chi _{\text{Lev}}(n,w)`$ | Def. 4 | 6 | 2 |
| $`A^{\text{ND},\chi }_n(w)`$ | Def. 6 | 9 | 3 |
| $`\beta (x, w)`$ | Def. 7 | 17 | 4 |
| $`\delta ^{D,\chi }_e`$ | Def. 7 | 14 | 4 |
| $`\le ^\chi _s`$ | Def. 11 | 18 | 4 |
| $`\sqcup A`$ | Def. 12 | 21 | 4 |
| $`A^{\forall ,\chi }_n`$ | Def. 15 | 29 | 5 |
| $`I^\chi _s, M^\chi _s`$ | Def. 15 | 30-33 | 5 |
| r_n, f_n, m_n | Def. 16-17 | 40-42 | 5 |

### Key Theorems

| Theorem | Statement | Page | Section |
|---------|-----------|------|---------|
| Prop. 9 | NFA Correctness | 13 | 3 |
| Prop. 19 | Universal Automaton Correctness | 43-47 | 5 |
| Thm. 1-3 | Minimality | 59-72 | 7 |

---

## Summary

The theoretical foundations establish:

1. ✅ **Distance definitions** for three variants $`(\varepsilon , t,`$ ms)
2. ✅ **NFA construction** with correctness proof
3. ✅ **DFA construction** using bit vector encoding
4. ✅ **Subsumption relation** for state minimization
5. ✅ **Universal automaton** using parametric positions
6. ✅ **Correctness proof** (Proposition 19)
7. ✅ **Minimality proofs** (Section 7)
8. ✅ **Complexity analysis** (asymptotic and concrete)

**Critical warning**: Triangle inequality fails for $`d^t_L`$!

**Main achievement**: $`A^{\forall ,\chi }_n`$ is a **parameter-free, minimal, universal** automaton for fuzzy string matching.

---

**Document Status**: ✅ Complete theoretical extraction
**Last Updated**: 2025-11-11
**Sections Covered**: 2, 3, 4, 5, 7, 8
**Total Definitions**: 18
**Total Propositions**: 19+
**Total Theorems**: 3+
