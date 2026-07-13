[← Documentation Index](../../README.md)

# Levenshtein Automata - Glossary

**Quick Reference for Terms, Notation, and Concepts**

**Date**: 2025-11-06

**Source paper**: Schulz, K. U. & Mihov, S. (2002). *Fast string correction with Levenshtein automata*. International Journal on Document Analysis and Recognition (IJDAR) 5, 67–85. [doi:10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)

---

> **Note:** This glossary covers **theoretical algorithm concepts** from the foundational papers. For **implementation details, performance optimizations, and user-facing features**, see the [Technical Glossary](../../GLOSSARY.md) which covers 70+ additional terms including data structures, SIMD operations, caching strategies, and API features.

---

## Mathematical Notation

### Alphabet and Strings

| Symbol | Meaning | Example |
|--------|---------|---------|
| **`$\Sigma$`** | Alphabet (set of characters) | {a, b, c, ..., z} |
| **`$\varepsilon$`** | Empty string | "" |
| **`$W, V, U$`** | Words/strings | "hello", "world" |
| **`$\lvert W\rvert, w$`** | Length of word W | `$\lvert\text{"hello"}\rvert = 5$` |
| **`$W[i]$`** | Character at position i in W | "hello"[2] = 'l' |
| **`$W[i:j]$`** | Substring from i to j | "hello"[1:3] = "el" |
| **`$a, b, x, y$`** | Individual characters | 'h', 'e', 'l' |

### Distance and Languages

| Symbol | Meaning | Paper Section |
|--------|---------|---------------|
| **`$d_L(W,V)$`** | Levenshtein distance between W and V | Definition 2.0.1 |
| **`$L_{\mathrm{Lev}}(n,W)$`** | Set of all words within distance n from W | Definition 3.0.4 |
| **`$\mathrm{LEV}_n(W)$`** | Levenshtein automaton of degree n for W | Definition 4.0.28 |
| **`$L(A)$`** | Language accepted by automaton A | Section 2 |
| **`$L(\pi)$`** | Language accepted from position `$\pi$` | Proposition 4.0.31 |

### Positions and States

| Symbol | Meaning | Paper Section |
|--------|---------|---------------|
| **i#e** | Position: index i, error count e | Definition 4.0.12 |
| **i#e_t** | t-position (with transposition flag) | Definition 7.1.1 |
| **i#e_s** | s-position (with merge/split flag) | Definition 8.1.1 |
`$| **\pi , \rho ** |$` Position variables | Throughout |
| **M, N** | State variables (sets of positions) | Definition 4.0.18 |
`$| **\sqsubseteq ** |$` Subsumption relation | Definition 4.0.15 |
`$| **[\pi ]$`↑e** | Raised position | Definition 4.0.26 |
| **[M]↑e** | Raised state | Definition 4.0.26 |

### Automata Components

| Symbol | Meaning | Paper Section |
|--------|---------|---------------|
| **A, B** | Automata | Section 2 |
| **Q** | State set | Standard FSA notation |
| **q₀** | Initial state | Standard FSA notation |
| **F** | Set of final/accepting states | Standard FSA notation |
`$| **\delta ** |$` Transition function (single position) | Definition 4.0.24 |
`$| **\Delta ** |$` Transition function (state/set of positions) | Definition 4.0.28 |
`$| **\Delta _*^W** |$` Extended transition using table | Chapter 6 |

### Characteristic Vectors and Profiles

| Symbol | Meaning | Paper Section |
|--------|---------|---------------|
`$| **\chi (x,V)** |$` Characteristic vector of x in V | Definition 4.0.10 |
| **Pr(U)** | Profile of word U | Definition 4.0.6 |
| **Pr_k(U)** | k-profile of word U | Definition 4.0.8 |
`$| **\langle b_{1},...,b_k\rangle** |$` Bit-vector notation | Throughout |

### Operations

| Symbol | Meaning | Example |
|--------|---------|---------|
`$| **\sqcup ** |$` Join/union operation on states `$| M \sqcup  N |$`
`$| **\cup ** |$` Set union `$| A \cup  B |$`
`$| **\cap ** |$` Set intersection `$| A \cap  B |$`
| **×** | Cartesian product  | `$\Sigma  \times  \Sigma$` | 
`$| **\subseteq ** |$` Subset relation `$| A \subseteq  B |$`
`$| **\in ** |$` Element of `$| x \in  \Sigma  |$`

---

## Key Terms

### A

**Accepting State** - A final state in an automaton. Reaching this state indicates successful matching.

**Acyclic** - An automaton with no cycles in its state graph. LEV_n(W) is always acyclic (Theorem 4.0.32).

**Alphabet `$(\Sigma )** -$` The set of all possible characters in strings.

### B

**Base Position** - The position in a state with the minimum index and error count. For state M, the base position is i#0.

**Bit-vector** - A sequence of binary values (0 or 1), used in characteristic vectors.

### C

**Characteristic Vector `$\chi (x,V)** - A$` bit-vector `$\langle b_{1},...,b_v\rangle$` where b_j = 1 if V[j] = x, otherwise 0.
- **Purpose**: Encodes where character x appears in word V
- **Use**: Determines transition behavior in elementary transitions

**Completion** - Process of adding error positions to ensure all possible error-correcting paths are covered.

### D

**Deletion** - Edit operation: removing a character from a word.
- **Example**: "hello" → "helo" (delete 'l')
- **Cost**: 1

**Deterministic** - An automaton where each state has at most one outgoing transition for each input symbol.

**Dictionary Automaton (`$A^D$`)** - Finite state automaton representing a dictionary of valid words.

**Distance** - See Levenshtein Distance.

### E

**Edit Distance** - See Levenshtein Distance.

**Edit Operations** - Transformations allowed when computing distance:
- **Standard**: insertion, deletion, substitution
- **Extended**: transposition, merge, split

**Elementary Transition `$\delta (\pi ,x)** -$` Transition from a single position `$\pi$` under input character x.
- **Defined in**: Table 4.1 (standard), Table 7.1 (transposition), Table 8.1 (merge/split)

**Error Bound (n)** - Maximum allowed Levenshtein distance for matching.

**Error Count (e)** - Number of edit operations accumulated at a position.

### F

**Final State** - See Accepting State.

**Fixed Degree** - The error bound `n` is constant (not varying with input size).
- **Significance**: Enables `$\mathcal{O}(\lvert W\rvert)$` construction complexity

### I

**Imitation Method** - Algorithm that simulates LEV_n(W) without explicitly constructing it (Chapter 6).

**Index (i)** - Position in the input word W, ranging from 0 to |W|.

**Insertion** - Edit operation: adding a character to a word.
- **Example**: "helo" → "hello" (insert 'l')
- **Cost**: 1

### K

**k-profile** - Local profile for a subword of length k.

### L

**Levenshtein Automaton LEV_n(W)** - Deterministic finite automaton that accepts exactly L_Lev(n,W).

**Levenshtein Distance d_L(W,V)** - Minimum number of single-character edits needed to transform W into V.

**Levenshtein Language L_Lev(n,W)** - Set of all words within Levenshtein distance n from word W.

### M

**Merge Operation** - Edit operation: two characters become one.
- **Example**: "rn" in input matches "m" in dictionary
- **Cost**: 1

**Minimal Automaton** - Automaton with the fewest states accepting a given language.

### N

**Non-deterministic** - Automaton where states may have multiple transitions for the same input symbol.
- **Note**: This paper focuses on deterministic automata

### O

**OCR (Optical Character Recognition)** - Application domain for string correction.

### P

**Parallel Traversal** - Simultaneously traversing dictionary automaton and Levenshtein automaton.

**Parametric Table (T_n)** - Precomputed table describing all states and transitions for error bound n.
- **Advantage**: Enables `$\mathcal{O}(\lvert W\rvert)$` automaton construction

**Position (i#e)** - Pair of index i and error count e.
- **Interpretation**: "Matched i characters of W with e errors"
- **Range**: `$0 \le  i \le  |W|, 0 \le  e \le  n$`

**Profile Pr(U)** - Sequence encoding structural properties of word U based on character repetition patterns.

### Q

**Query Word** - The input word W for which we're finding similar dictionary entries.

### R

**Raised Position [i#e]↑k** - Position (i+k)#(e+k) obtained by "raising" i#e.

**Relevant Subword `$W[\pi ]** -$` For position `$\pi  = i$`#e, the subword W[i+1:i+k] where k = min(n-e+1, |W|-i).

### S

**s-position (i#e_s)** - Position with merge/split flag (Chapter 8).

**Split Operation** - Edit operation: one character becomes two.
- **Example**: "m" in input matches "rn" in dictionary
- **Cost**: 1

**State (M)** - Set of positions satisfying specific properties:
1. Contains base position i#0
2. No position subsumes another
3. All positions reachable from base

**Substitution** - Edit operation: replacing one character with another.
- **Example**: "hello" → "hallo" (substitute 'e' with 'a')
- **Cost**: 1

**Subsumption `$(\pi  \sqsubseteq  \rho )** -$` Position i#e subsumes j#f if:
- e < f (fewer errors)
`$- |j-i| \le  f-e ($`within error budget)
- **Meaning**: `$\pi$` is "better" than `$\rho ,$` making `$\rho$` redundant

### T

**t-position (i#e_t)** - Position with transposition flag (Chapter 7).

**Trace** - Graphical representation of an edit operation sequence.

**Transition Function** - Function describing how automaton changes state on input.
`$- **\delta (\pi ,x)**$`: Elementary transition (single position)
`$- **\Delta (M,x)**$`: State transition (set of positions)

**Transposition** - Edit operation: swapping two adjacent characters.
- **Example**: "hello" → "hlelo" (swap 'e' and 'l')
- **Cost**: 1

**Trie** - Tree-structured dictionary where paths represent words.

### U

**Union `$(\sqcup )** -$` Combining sets of positions, removing subsumed positions.

### W

**Wagner-Fischer Algorithm** - Dynamic programming algorithm for computing Levenshtein distance in `$\mathcal{O}(\lvert W\rvert\times \lvert V\rvert)$` time. See Wagner & Fischer (1974), [doi:10.1145/321796.321811](https://doi.org/10.1145/321796.321811).

**Word** - A string over alphabet `$\Sigma .$`

---

## Common Abbreviations

| Abbreviation | Meaning |
|--------------|---------|
| **FSA** | Finite State Automaton |
| **DFA** | Deterministic Finite Automaton |
| **NFA** | Non-deterministic Finite Automaton |
| **DAWG** | Directed Acyclic Word Graph |
| **OCR** | Optical Character Recognition |
| **IR** | Information Retrieval |
| **NLP** | Natural Language Processing |

---

## Formula Quick Reference

### Levenshtein Distance (Recursive Definition)

```
d_L(ε, W) = |W|
d_L(V, ε) = |V|
d_L(aV, bW) = d_L(V,W)                              if a = b
d_L(aV, bW) = 1 + min(d_L(V,W),                    if a ≠ b
                       d_L(aV,W),
                       d_L(V,bW))
```

### Subsumption

```
i#e ⊑ j#f  ⟺  (e < f) ∧ (|j-i| ≤ f-e)
```

### Language of Position

```
L({i#e}) = L_Lev(n-e, W[i+1:|W|])
```

### Elementary Transition Cases (Standard)

For position `$\pi  = i$`#e and character x:

```
χ(x,W[π]) = ⟨1,b₂,...,b_k⟩   ⟹  δ(π,x) = {(i+1)#e}
χ(x,W[π]) = ⟨0,...,0,1,...⟩  ⟹  δ(π,x) = {i#(e+1), (i+1)#(e+1), (i+j)#(e+j-1)}
χ(x,W[π]) = ⟨0,...,0⟩        ⟹  δ(π,x) = {i#(e+1), (i+1)#(e+1)}
```

---

## State Types for n=1

From Table 5.1, the parametric states for degree 1:

| State | Positions | Interpretation |
|-------|-----------|----------------|
| **A_i** | {i#0} | No errors, matched i characters |
| **B_i** | {i#0, i#1} | Base at i, one error possible |
| **C_i** | {i#0, (i+1)#1} | Base at i, error at next position |
| **D_i** | {i#0, i#1, (i+1)#1} | Base at i, errors at i and i+1 |
| **E_i** | {i#1} | One error, matched i characters |

---

## Cross-References

### Paper Sections

- **Definition 2.0.1**: Levenshtein distance (page 10)
- **Definition 3.0.4**: Levenshtein language (page 13)
- **Definition 4.0.12**: Position (page 17)
- **Definition 4.0.15**: Subsumption (page 18)
- **Definition 4.0.24**: Elementary transition (page 21)
- **Definition 4.0.28**: LEV_n(W) (page 23)
- **Theorem 4.0.32**: Main theorem (page 24)
- **Table 4.1**: Elementary transitions for standard operations (page 22)
- **Table 5.2**: Transition table T_1 (pages 29-30)
- **Table 7.1**: Elementary transitions with transpositions (page 37)
- **Table 8.1**: Elementary transitions with merge/split (page 48)

### Code Files

- **position.rs**: Position structure, subsumption, characteristic vectors
- **transition.rs**: Elementary transition implementations
- **algorithm.rs**: Algorithm enum (Standard, Transposition, MergeAndSplit)
- **builder.rs**: Automaton construction
- **query.rs**: Parallel traversal algorithm

---

## Usage Tips

### When Reading the Paper

1. **First pass**: Focus on Chapters 1-3 for intuition
2. **Second pass**: Study Chapter 4 for core algorithm
3. **Third pass**: Dive into Chapters 5-6 for implementation
4. **Extensions**: Read Chapters 7-8 as needed

### When Reading the Code

1. **Start**: Understand Position struct (position.rs)
2. **Next**: Study elementary transitions (transition.rs)
3. **Then**: Follow query algorithm (query.rs)
4. **Reference**: Use this glossary for notation lookups

### When Debugging

1. **Position values**: Check i and e are within valid ranges
2. **Subsumption**: Verify no position in state subsumes another
3. **Characteristic vectors**: Ensure correct computation for `$W[\pi ]$`
4. **Transitions**: Match against tables in paper

---

**Last Updated**: 2025-11-06
**See Also**: [README.md](./README.md) for overview
