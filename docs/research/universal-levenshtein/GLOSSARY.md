[← Documentation Index](../../README.md)

# Universal Levenshtein Automata - Glossary

**Document Status**: Complete notation reference
**Source**: Mitankin, P. N. (2005). *Universal Levenshtein Automata — Building and Properties*. Master's Thesis, Sofia University "St. Kliment Ohridski" (supervisor: S. Mihov). Related journal generalisation: Mitankin, Mihov & Schulz (2011), TCS 412(22):2340–2355, [doi:10.1016/j.tcs.2011.01.013](https://doi.org/10.1016/j.tcs.2011.01.013).
**Last Updated**: 2025-11-11

---

## Purpose

This glossary provides a comprehensive reference for all mathematical notation, symbols, functions, and terminology used in the Universal Levenshtein Automata thesis and documentation.

**Related Documents**:
- [PAPER_SUMMARY.md](./PAPER_SUMMARY.md) - Full chapter-by-chapter analysis
- [README.md](./README.md) - Overview and quick start
- [ALGORITHMS.md](./ALGORITHMS.md) - Implementation algorithms
- [Core Paper Glossary](../levenshtein-automata/glossary.md) - Fixed-word automata notation

---

## Quick Reference

### Most Important Symbols

| Symbol | Meaning | Page |
|--------|---------|------|
`$| \chi  |$` Metasymbol: `$\varepsilon$` (standard), t (transposition), ms (merge/split) | 3 |
`$| d^\chi _L(v, w) |$` Levenshtein distance between v and w | 3-5 |
| i#e | Fixed-word position: position i, error count e | 9 |
| I + i#e | Universal non-final position | 30 |
| M + i#e | Universal final position | 33 |
`$| A^\forall ,\chi _n |$` Universal Levenshtein automaton | 30 |
| h_n(w, x) | Bit vector encoding of word pair | 51 |
`$| \beta (x, w) |$` Characteristic vector | 17 |
`$| \le ^\chi _s |$` Subsumption relation | 18 |
`$| \sqcup A |$` Subsumption closure | 21 |

---

## Notation by Category

### 1. Metasymbols and Variants

#### `$\chi$` (Chi) - Distance Variant Metasymbol
**Definition**: `$\chi  \in$` `$\{\varepsilon , t, \text{ms}\}$`
- **`$\chi  = \varepsilon$`** (or `$\chi  = ^{2})$`: Standard Levenshtein distance
`$- **\chi  = t**$`: Levenshtein distance with transposition
`$- **\chi  =$` ms**: Levenshtein distance with merge and split

**Usage**: Throughout the thesis as a placeholder for any of the three variants

**Example**: `$d^\chi _L$` means d²_L, `$d^t_L$`, or `$d^\text{ms}_L$` depending on context

---

### 2. Distance Functions

#### d²_L : `$\Sigma * \times  \Sigma * \to  \mathbb{N}$`
**Standard Levenshtein Distance** (Page 3)

Minimum cost to transform v into w using:
- Deletion (cost 1)
- Insertion (cost 1)
- Substitution (cost 1)

**Properties**:
- d²_L(v, w) = 0 ⇔ v = w
- d²_L(v, w) = d²_L(w, v) (symmetric)

#### `$d^t_L$` : `$\Sigma * \times  \Sigma * \to  \mathbb{N}$`
**Levenshtein Distance with Transposition** (Page 4)

Standard operations plus:
- Transposition of adjacent characters (cost 1)

**⚠️ WARNING**: Does NOT satisfy triangle inequality!
- Not a proper metric
- Example: `$d^t_L(\text{abcd}, \text{abdc})$` + `$d^t_L(\text{abdc}, \text{bdac})$` < `$d^t_L(\text{abcd}, \text{bdac})$`

#### `$d^\text{ms}_L$` : `$\Sigma * \times  \Sigma * \to  \mathbb{N}$`
**Levenshtein Distance with Merge/Split** (Page 5)

Standard operations plus:
- Merge: Two characters → one character (cost 1)
- Split: One character → two characters (cost 1)

#### `$d^\chi _L$` : `$\Sigma * \times  \Sigma * \to  \mathbb{N}$`
**Generic notation** for any of the above three distances

---

### 3. Language and Sets

#### `$L^\chi _\text{Lev}(n, w)$` : `$\mathcal{P}(\Sigma *)$`
**Levenshtein Language** (Page 6)

```
L^χ_Lev(n, w) = {v | d^χ_L(v, w) ≤ n}
```

The set of all words within distance n from w.

#### `$R^\chi (n, w)$` : `$\mathcal{P}(\Sigma *)$`
**Extension Function** (Page 7)

Recursive decomposition of `$L^\chi _\text{Lev}(n, w)$`:
- For `$\chi  = \varepsilon$`: Includes insertion, deletion, substitution, match terms
- For `$\chi  = t$`: Adds transposition term
- For `$\chi  =$` ms: Adds merge and split terms

**Key Property**: `$L^\chi _\text{Lev}(n, w)$` = `$R^\chi (n, w)$`

---

### 4. Position Notation (Fixed-Word)

#### i#e
**Standard Position** (Page 9)

Compact notation for `$\langle \langle i, 0\rangle, e\rangle$`

- **i**: Position in word `$w (0 \le  i \le  |w|)$`
- **e**: Error count consumed `$(0 \le  e \le  n)$`

**Language**: L(i#e) = `$L^\chi _\text{Lev}(n - e, w_{i+1}...w_p)$`

#### i#e_t
**Transposition Position** (Page 10)

Compact notation for `$\langle \langle i, 1\rangle, e\rangle$`

Used when detecting transposition of w_{i+1} and w_{i+2}

#### i#e_s
**Merge/Split Position** (Page 10)

Compact notation for `$\langle \langle i, 2\rangle, e\rangle$`

Used when processing merge or split operations

---

### 5. Position Notation (Universal)

#### I + i#e
**Universal Non-Final Standard Position** (Page 29)

Compact notation for `$\langle \langle \lambda I.I+i, 0\rangle, e\rangle$`

- **I**: Parameter (to be substituted with 0 for word start)
- **i**: Relative offset `$(-n \le  i \le  n)$`
- **e**: Error count `$(0 \le  e \le  n)$`

**Constraint**: `$|i| \le  e$`

#### It + i#e
**Universal Non-Final Transposition Position** (Page 31)

Compact notation for `$\langle \langle \lambda I.I+i, 1\rangle, e\rangle$`

#### Is + i#e
**Universal Non-Final Split Position** (Page 32)

Compact notation for `$\langle \langle \lambda I.I+i, 2\rangle, e\rangle$`

#### M + i#e
**Universal Final Standard Position** (Page 33)

Compact notation for `$\langle \langle \lambda M.M+i, 3\rangle, e\rangle$`

- **M**: Parameter (to be substituted with |w| for word end)
- **i**: Relative offset `$(-2n \le  i \le  0)$`
- **e**: Error count `$(0 \le  e \le  n)$`

**Constraint**: `$e \ge  -i - n$`

#### Mt + i#e
**Universal Final Transposition Position** (Page 34)

Compact notation for `$\langle \langle \lambda M.M+i, 4\rangle, e\rangle$`

#### Ms + i#e
**Universal Final Split Position** (Page 35)

Compact notation for `$\langle \langle \lambda M.M+i, 5\rangle, e\rangle$`

---

### 6. Automata

#### `$A^\text{ND},\chi _n(w)$`
**Nondeterministic Levenshtein Automaton** (Page 9)

```
A^ND,χ_n(w) = ⟨Σ, Q^ND,χ_n, I^ND,χ, F^ND,χ_n*, δ^ND,χ_n⟩
```

- **Alphabet**: `$\Sigma  \cup$` `$\{\varepsilon\}$`
- **States**: Positions i#e (and i#e_t, i#e_s for t, ms)
- **Language**: `$L^\chi _\text{Lev}(n, w)$`

#### `$A^D,\chi _n(w)$`
**Deterministic Levenshtein Automaton** (Page 23)

```
A^D,χ_n(w) = ⟨Σ, Q^D,χ_n, I^D,χ, F^D,χ_n, δ^D,χ_n⟩
```

- **Alphabet**: `$\Sigma$`
- **States**: Sets of positions (anti-chains under subsumption)
- **Language**: `$L^\chi _\text{Lev}(n, w)$`

#### `$A^\forall ,\chi _n$`
**Universal Levenshtein Automaton** (Page 30)

```
A^∀,χ_n = ⟨Σ^∀_n, Q^∀,χ_n, I^∀,χ, F^∀,χ_n, δ^∀,χ_n⟩
```

- **Alphabet**: `$\Sigma^\forall_n = \{x \in \{0,1\}^+ \mid \lvert x\rvert \le 2n + 2\}$`
- **States**: Sets of universal positions
- **Language**: `$\{h_n(w, x) | ﷐0﷑ \le  n\}$`

---

### 7. State Sets

#### `$Q^\text{ND},\chi _n$`
**NFA State Set** (Page 9)

For `$\chi  = \varepsilon$`: `$\{i#e | 0 \le  i \le  p \land  0 \le  e \le  n\}$`

Plus transposition/split states for t, ms variants.

#### `$Q^D,\chi _n$`
**DFA State Set** (Page 23)

Sets of positions that are:
1. Anti-chains under `$\le ^\chi _s$` (no position subsumes another)
2. Have a common base position

#### `$I^\chi _s$`
**Universal Non-Final Position Set** (Page 30-32)

All valid I-type positions for variant `$\chi .$`

For `$\chi  = \varepsilon$`: `$\{I + t#k | |t| \le  k \land  -n \le  t \le  n \land  0 \le  k \le  n\}$`

#### `$M^\chi _s$`
**Universal Final Position Set** (Page 33-35)

All valid M-type positions for variant `$\chi .$`

For `$\chi  = \varepsilon$`: `$\{M + t#k | k \ge  -t - n \land  -2n \le  t \le  0 \land  0 \le  k \le  n\}$`

#### `$I^\chi _\text{states}$`
**Universal Non-Final States** (Page 38)

```
I^χ_states = {Q | Q ⊆ I^χ_s ∧ ∀q₁,q₂ ∈ Q (q₁ ⊀^χ_s q₂)} \ {∅}
```

Anti-chains of non-final positions.

#### `$M^\chi _\text{states}$`
**Universal Final States** (Page 38)

```
M^χ_states = {Q | Q ⊆ M^χ_s ∧ anti-chain ∧ valid constraints} \ {∅}
```

Anti-chains of final positions with additional constraints.

---

### 8. Transition Functions

#### `$\delta ^\text{ND}$`,`$\chi _n$` : `$Q^\text{ND},\chi _n \times  (\Sigma  \cup$` `$\{\varepsilon\}$`) `$\to  \mathcal{P}$`(`$Q^\text{ND},\chi _n)$`
**NFA Transition** (Page 9-10)

Standard NFA transitions with `$\varepsilon$`-transitions for insertions.

#### `$\delta ^\text{ND}$`,`$\chi _n$`* : `$Q^\text{ND},\chi _n \times  \Sigma$`* `$\to  \mathcal{P}(Q^\text{ND},\chi _n)$`
**Extended NFA Transition** (Page 11)

Handles sequences with `$\varepsilon$`-closure.

#### `$\delta ^D,\chi _e$` : `$Q^\text{ND},\chi  \times$` {0,1}* `$\to  \mathcal{P}(Q^\text{ND},\chi )$`
**Elementary Transition Function** (Page 14-16)

Given position and bit vector, compute reachable positions.

#### `$\delta ^D,\chi _n$` : `$Q^D,\chi _n \times  \Sigma  \to  Q^D,\chi _n$`
**DFA Transition** (Page 23)

```
δ^D,χ_n(M, x) = ⊔_{π∈M} δ^D,χ_e(π, x)
```

Apply elementary transitions and subsumption closure.

#### `$\delta ^\forall ,\chi _e$` : `$(I^\chi _s \cup  M^\chi _s) \times  \Sigma ^\forall _n \to  \mathcal{P}(I^\chi _s \cup  M^\chi _s)$`
**Universal Elementary Transition** (Page 46)

Extract relevant subvector, apply `$\delta ^D,\chi _e,$` convert back to universal positions.

#### `$\delta ^\forall ,\chi _n$` : `$Q^\forall ,\chi _n \times  \Sigma ^\forall _n \to  Q^\forall ,\chi _n$`
**Universal Transition** (Page 48)

```
δ^∀,χ_n(Q, x) = {
    Δ           if f_n(rm(Δ), |x|) = false
    m_n(Δ, |x|) if f_n(rm(Δ), |x|) = true
}
where Δ = ⊔_{q∈Q} δ^∀,χ_e(q, x)
```

Includes diagonal crossing check and I/M conversion.

---

### 9. Helper Functions

#### `$w[\pi ]$` : Relevant Subword
**Definition** (Page 17): For position `$\pi  = i$`#e, returns w_{i+1}...w_{i+k} where k = min(n - e + 1, p - i)

#### `$\beta (x, w)$` : Characteristic Vector
**Definition** (Page 17): `$\beta$` : `$\Sigma  \times  \Sigma * \to$` {0,1}*

```
β(x, w₁w₂...w_p) = b₁b₂...b_p where b_i = (1 if x = w_i else 0)
```

**Example**: `$\beta ('a',$` "banana") = "101010"

#### s_n(w, i) : Relevant Subword for Position
**Definition** (Page 51): Returns window around position i

```
s_n(w, i) = w_{i-n}...w_{min(|w|, i+n+1)}
```

With padding: w_{-n+1} = ... = w_0 = $

#### h_n(w, x) : Bit Vector Encoding
**Definition** (Page 51): Encodes word pair (w, x) as bit vector sequence

```
h_n(w, x₁...x_t) = β(x₁, s_n(w,1))...β(x_t, s_n(w,t))
```

**Key Property**: `$x \in$` `$L^\chi _\text{Lev}(n, w)$` `$\Leftrightarrow  h_n(w, x) \in  L(A^\forall,\chi _n)$`

#### r_n : Relevant Subvector
**Definition** (Page 39): Extracts relevant portion of bit vector for a universal position

For I + i#e: Extract from position (n + i + 1)
For M + i#e: Extract from position (k + i + 1) where k = |input|

#### m_n : I/M Conversion
**Definition** (Page 42): Converts between non-final and final positions

```
m_n(I + i#e, k) = M + (i + n + 1 - k)#e
m_n(M + i#e, k) = I + (i - n - 1 + k)#e
```

#### f_n : Diagonal Check
**Definition** (Page 43): Checks if position crossed diagonal

For I + i#e: f_n(S, k) = true if `$k \le  2n + 1 \land  e \le  i + 2n + 1 - k$`
For M + i#e: f_n(S, k) = true if e > i + n

#### rm : Right-Most Element
**Definition** (Page 45): Returns position with maximum (e - i) in a state

```
rm(A) = position with max(e - i)
```

#### ▽_a : Allowed Lengths
**Definition** (Page 47): Returns valid input lengths for a state

For {I#0}: `$\{k | n \le  k \le  2n + 2\}$`
For other states: Computed based on right-most element

---

### 10. Subsumption Relations

#### `$\le ^\chi _s$` : Subsumption Relation
**Definition** (Page 18-19): Partial order on positions

**For `$\chi  = \varepsilon **$`:
```
i#e ≤^ε_s j#f ⇔ f > e ∧ |j - i| ≤ f - e
```

**Intuition**: Position j#f subsumes i#e if j#f has enough extra errors to "cover" the position difference.

**For universal positions** (Page 36-37):
```
I + i#e ≤^χ_s I + j#f ⇔ i#e ≤^χ_s j#f
M + i#e ≤^χ_s M + j#f ⇔ i#e ≤^χ_s j#f
```

**Properties**:
- Reflexive: `$\pi  \le ^\chi _s \pi$`
- Antisymmetric: `$\pi _{1} \le ^\chi _s \pi _{2} \land  \pi _{2} \le ^\chi _s \pi _{1} \Rightarrow  \pi _{1} = \pi _{2}$`
- Transitive: `$\pi _{1} \le ^\chi _s \pi _{2} \land  \pi _{2} \le ^\chi _s \pi _{3} \Rightarrow  \pi _{1} \le ^\chi _s \pi _{3}$`

#### `$<^\chi _s$` : Strict Subsumption
**Definition**: `$\pi _{1} <^\chi _s \pi _{2} \Leftrightarrow  \pi _{1} \le ^\chi _s \pi _{2} \land  \pi _{1} \ne  \pi _{2}$`

#### `$\sqcup A$` : Subsumption Closure
**Definition** (Page 21, 47): Removes subsumed elements

```
⊔A = {π | π ∈ ⋃A ∧ ¬∃π' ∈ ⋃A (π' <^χ_s π)}
```

Returns maximal elements (anti-chain).

---

### 11. Conversion Functions

#### `$I^\chi$` : `$\mathcal{P}(Q^\text{ND},\chi ) \to  \mathcal{P}(P^\chi$`)
**Concrete to Universal (Non-Final)** (Page 44)

```
I^ε(A) = {I + (i - 1)#e | i#e ∈ A}
```

Shifts positions by -1 and adds I parameter.

#### `$M^\chi$` : `$\mathcal{P}(Q^\text{ND},\chi ) \to  \mathcal{P}(P^\chi$`)
**Concrete to Universal (Final)** (Page 44)

```
M^ε(A) = {M + i#e | i#e ∈ A}
```

Adds M parameter without shift.

#### d : (`$I^\chi _s$` `$\cup$` `$M^\chi _s) \times  \mathbb{N}  \to$` `$Q^\text{ND},\chi$`
**Universal to Concrete** (Page 52)

```
d(I + i#e, z) = (z + i)#e
d(M + i#e, z) = (z + i)#e
```

Substitutes parameter (I or M) with concrete value z.

---

### 12. Special Notation

#### ↪ : Suffix Operator
**Definition** (Page 4): x₁...xₖ ↪ t removes first t characters

```
x₁...xₖ ↪ t = {
    ε                if t ≥ k
    x_{t+1}...xₖ    otherwise
}
```

#### < : Prefix Relation
**Definition** (Page 4): c < d means c is a prefix of d

#### `$\mu z[A]$` : Minimum
**Definition** (Page 14): The least z such that property A holds

#### ! : Definedness
**Definition**: !x means x is defined (not `$\lnot$`!)

#### `$\lnot$`! : Undefinedness
**Definition**: `$\lnot$`!x means x is undefined

#### def= : Definition
**Definition**: x def= y means x is defined to equal y

---

### 13. Set Notation

#### `$\Sigma$`
**Alphabet**: Finite set of symbols

#### `$\Sigma *$`
**All words**: Set of all finite sequences over `$\Sigma$` (including empty word `$\varepsilon )$`

#### `$\Sigma ^{+}$`
**Non-empty words**: `$\Sigma *$` \ `$\{\varepsilon\}$`

#### `$\varepsilon$`
**Empty word**: Word of length 0

#### |w|
**Word length**: Number of characters in w

#### `$\mathcal{P}(A)$`
**Power set**: Set of all subsets of A

#### `$\mathbb{N}$`
**Natural numbers**: {0, 1, 2, ...}

#### `$\mathbb{N} ^{+}$`
**Positive integers**: {1, 2, 3, ...}

#### `$\mathbb{Z}$`
**Integers**: {..., -2, -1, 0, 1, 2, ...}

---

### 14. Automaton Components

#### Q
**States**: Set of states

#### I
**Initial state**: Starting state (usually {0#0} or {I#0})

#### F
**Final states**: Accepting states

#### `$\delta$`
**Transition function**: Maps (state, symbol) to states

#### L(A)
**Language of automaton**: Set of words accepted by A

#### `$L(\pi )$`
**Language of state**: Set of words accepted starting from state `$\pi$`

---

### 15. Complexity Notation

#### `$\mathcal{O}(f(n)$`)
**Big-O**: Upper bound on growth rate

**Examples in thesis**:
- States: `$\mathcal{O}(n^{2})$`
- Construction: `$\mathcal{O}(n^{2} \cdot  2^{2n})$`

---

## Cross-Reference by Section

### Section 1: Introduction
- `$d^t_L$` triangle inequality violation warning

### Section 2: Distance Definitions (Pages 3-8)
- d²_L, `$d^t_L$`, `$d^\text{ms}_L$`
- ↪, <, `$L^\chi _\text{Lev}$`, `$R^\chi$`

### Section 3: Nondeterministic Automata (Pages 8-13)
- i#e, i#e_t, i#e_s
- `$A^{\text{ND},\chi}_n(w)$`, `$\delta^{\text{ND},\chi}_n$`, `$\text{Cl}_\varepsilon$`

### Section 4: Deterministic Automata (Pages 13-28)
`$- \delta ^D,\chi _e, \beta , w[\pi ]$`
`$- \le ^\chi _s, \sqcup , A^D,\chi _n(w)$`

### Section 5: Universal Automata (Pages 28-48)
- I + i#e, M + i#e, It, Is, Mt, Ms
`$- A^\forall ,\chi _n, r_n, m_n, f_n,$` rm, ▽_a
`$- h_n, s_n, I^\chi , M^\chi , d$`

### Section 6: Building (Pages 48-59)
- Construction algorithms (see ALGORITHMS.md)

### Section 7: Minimality (Pages 59-72)
- Proofs using subsumption properties

### Section 8: Properties (Pages 72-77)
- Additional theorems and properties

---

## Usage Tips

### For Reading the Paper
1. Start with `$\chi  = \varepsilon$` (standard) before tackling transposition/merge-split
2. Understand fixed-word automata (Sections 3-4) before universal (Section 5)
3. Remember: `$d^t_L$` violates triangle inequality (affects subsumption logic)
4. Universal positions use parameters (I, M) that get substituted with concrete values (0, |w|)

### For Implementation
1. Implement subsumption `$(\le ^\chi _s, \sqcup )$` correctly - critical for correctness
2. Bit vector encoding `$(\beta , h_n)$` must be efficient
3. Diagonal crossing (f_n, m_n) handles I/M conversion
4. Test against fixed-word automata for validation

### For Debugging
1. Check subsumption closure: No position should subsume another in a state
2. Verify diagonal crossing: f_n determines when to apply m_n
3. Validate bit vectors: h_n encoding must match characteristic vectors
4. Compare with fixed-word: Use Proposition 19 correspondence

---

## Common Confusions

### 1. Position Notation
**Fixed**: i#e (concrete position i in specific word w)
**Universal**: I + i#e (parametric, i is offset from I)

### 2. I vs M
**I-type**: Non-final states (before reaching end of word)
**M-type**: Final states (at or past end of word)

Conversion happens when crossing diagonal (detected by f_n).

### 3. Subsumption Direction
i#`$e \le ^\chi _s j$`#f means j#f **subsumes** i#e (j#f recognizes more words)

In state sets, we keep only maximal elements (not subsumed by others).

### 4. Bit Vector Encoding
h_n(w, x) produces a **sequence** of bit vectors (one per character of x)
Each bit vector encodes matches with a window around that position in w

### 5. Triangle Inequality
Only `$d^t_L$` violates it! d²_L and `$d^\text{ms}_L$` may satisfy it (not proven in thesis).

---

## Quick Lookup Table

| To find... | Look for... |
|------------|-------------|
| Distance definition | Section `$2, d^\chi _L$` | 
| Fixed-word position | Section 3, i#e |
| Universal position | Section 5, I + i#e, M + i#e |
| Subsumption | Section `$4, \le ^\chi _s$` | 
| Bit encoding | Section `$5, h_n, \beta$` | 
| Conversion I↔M | Section 5, m_n |
| Diagonal check | Section 5, f_n |
| Transition function | `$\delta ^\text{ND}$`,`$\chi _n, \delta ^D,\chi _n, \delta ^\forall ,\chi _n$` | 
| Construction algorithm | Section 6, Build_Automaton |
| Correctness proof | Section 5, Proposition 19 |
| Minimality proof | Section 7 |

---

**End of Glossary**

**Last Updated**: 2025-11-11
**Notation Count**: 50+ symbols and functions
**Cross-referenced**: All sections covered
