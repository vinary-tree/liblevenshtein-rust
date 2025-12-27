# Suffix Automaton (DAWG) Theory

The **Suffix Automaton**, also called **DAWG** (Directed Acyclic Word Graph), is the minimal deterministic finite automaton that accepts exactly the suffixes of a string. More importantly for our purposes, it can be modified to accept all **substrings** of the string, forming the foundation for the SCDAWG.

## Preliminaries

### Notation

Let w be a string over alphabet Σ.

- |w| denotes the **length** of w
- w[i] denotes the **character** at position i (0-indexed)
- w[i..j] denotes the **substring** from position i to j-1 (exclusive end)
- ε denotes the **empty string** (length 0)
- w·x or wx denotes **concatenation** of w and x
- Σ* denotes the set of all strings over Σ (including ε)

### Factors and Subwords

A **factor** (or **subword**) of w is any substring w[i..j] where 0 ≤ i ≤ j ≤ |w|.

**Definition (Factor Set)**: F(w) = {w[i..j] : 0 ≤ i ≤ j ≤ |w|}

For our running example w = "abcabcab":

```
F(w) = {ε, a, b, c, ab, bc, ca, abc, bca, cab, abca, bcab, cabc,
        abcab, bcabc, cabca, abcabc, bcabca, cabcab, abcabca,
        bcabcab, abcabcab}
```

### End-Position Set

The **end-position set** of a factor x in w is the set of positions immediately after each occurrence of x:

**Definition (End-Position Set)**:
```
end-pos(x) = {i : w[i-|x|..i] = x, |x| ≤ i ≤ |w|}
```

Note: We use 1-indexed end positions (positions 1 through |w|) to match standard notation.

**Example** for w = "abcabcab":

| Factor x | Occurrences (start) | end-pos(x) |
|----------|---------------------|------------|
| ε | everywhere | {0,1,2,3,4,5,6,7,8} |
| a | 0, 3, 6 | {1, 4, 7} |
| b | 1, 4, 7 | {2, 5, 8} |
| c | 2, 5 | {3, 6} |
| ab | 0, 3, 6 | {2, 5, 8} |
| bc | 1, 4 | {3, 6} |
| abc | 0, 3 | {3, 6} |
| cab | 2, 5 | {5, 8} |
| abcab | 0, 3 | {5, 8} |
| abcabcab | 0 | {8} |

**Key Observation**: Factors "b" and "ab" have the same end-position set {2, 5, 8}. Similarly, "c", "bc", and "abc" share {3, 6}.

## Equivalence Classes

### The Right-Context Equivalence

Two factors are **equivalent** if and only if they have the same end-position set:

**Definition (Factor Equivalence)**:
```
x ≡ y  ⟺  end-pos(x) = end-pos(y)
```

This is indeed an equivalence relation (reflexive, symmetric, transitive), partitioning F(w) into equivalence classes.

**Theorem 1**: The number of equivalence classes is at most 2|w| - 1.

*Proof sketch*: Each new character can create at most 2 new equivalence classes (one for the new suffix, possibly one from splitting an existing class).

### Equivalence Classes for "abcabcab"

Grouping factors by their end-position sets:

| Class ID | end-pos | Factors | Size |
|----------|---------|---------|------|
| q₀ | {0..8} | {ε} | 1 |
| q₁ | {1,4,7} | {a} | 1 |
| q₂ | {2,5,8} | {b, ab} | 2 |
| q₃ | {3,6} | {c, bc, abc} | 3 |
| q₄ | {4,7} | {ca, bca, abca} | 3 |
| q₅ | {5,8} | {cab, bcab, abcab} | 3 |
| q₆ | {6} | {cabc, bcabc, abcabc} | 3 |
| q₇ | {7} | {cabca, bcabca, abcabca} | 3 |
| q₈ | {8} | {cabcab, bcabcab, abcabcab} | 3 |

**Total**: 9 equivalence classes for a string of length 8 (≤ 2×8-1 = 15).

### Class Structure

Each equivalence class [x] has important structural properties:

**Lemma 1 (Suffix Chain)**: If x ≡ y and |x| < |y|, then x is a suffix of y.

*Proof*: Since end-pos(x) = end-pos(y), every occurrence of y ends where some occurrence of x ends. Since y is longer, y must contain x as a suffix.

**Corollary**: Each equivalence class forms a **suffix chain** - a sequence of strings where each is a suffix of the next:
```
shortest ⊂ ... ⊂ longest
```

For class q₃ = {c, bc, abc}:
```
c ⊂ bc ⊂ abc
```
Here "c" is a suffix of "bc", and "bc" is a suffix of "abc".

### Longest and Shortest Representatives

For each equivalence class [x]:
- **longest(x)**: The longest string in the class
- **shortest(x)**: The shortest string in the class

The strings in [x] are exactly those with lengths in [|shortest(x)|, |longest(x)|] that are suffixes of longest(x).

## The Suffix Automaton

### Definition

The **Suffix Automaton** (or **DAWG**) of w is the deterministic finite automaton:

**SA(w) = (Q, Σ, δ, q₀, F)** where:
- **Q** = set of equivalence classes {[x] : x ∈ F(w)}
- **Σ** = alphabet
- **δ([x], a)** = [xa] if xa ∈ F(w), undefined otherwise
- **q₀** = [ε] (initial state)
- **F** = {[x] : longest(x) is a suffix of w} (accepting states)

### Transition Function

The transition function δ maps (state, character) to the next state:

```
δ([x], a) = [xa]
```

This works because if x₁ ≡ x₂ (same end-positions), then x₁a ≡ x₂a (appending the same character preserves the relationship).

### Graphical Representation

For w = "abcabcab", the suffix automaton:

```
           a           b           c
    q₀ --------> q₁ --------> q₂ --------> q₃
    |            |            |            |
    |     a      |     a      |     a      |
    +------------+------------+------------+
                              |
                              v
           a           b           c
    q₃ --------> q₄ --------> q₅ --------> q₆
                              |
                              v
           a           b
    q₆ --------> q₇ --------> q₈
```

ASCII Diagram (simplified adjacency):

```
States and transitions for SA("abcabcab"):

q₀ --a--> q₁
q₀ --b--> q₂
q₀ --c--> q₃

q₁ --b--> q₂
q₁ --c--> q₄  (from "a" going to "ac"? No, wait...)
```

Let me provide a more accurate representation:

```
Suffix Automaton for "abcabcab":

        +---a---+          +---a---+
        |       v          |       v
       q₀ --a-> q₁ --b-> q₂ --c-> q₃ --a-> q₄ --b-> q₅ --c-> q₆ --a-> q₇ --b-> q₈
        |       ^---b--------^---c--------^
        |       ^---a--------^------------^
        +--b----+            |
        +--c-----------------+
```

This is getting complex. Let's use a table representation:

| From | a | b | c |
|------|---|---|---|
| q₀ | q₁ | q₂ | q₃ |
| q₁ | - | q₂ | - |
| q₂ | q₄ | - | q₃ |
| q₃ | q₄ | q₅ | - |
| q₄ | - | q₅ | q₆ |
| q₅ | q₇ | - | q₆ |
| q₆ | q₇ | q₈ | - |
| q₇ | - | q₈ | - |
| q₈ | - | - | - |

Where `-` indicates no transition (character doesn't extend any factor in that class).

## Suffix Links

### Definition

The **suffix link** of a state [x] points to the state of its longest proper suffix that forms a different equivalence class:

**Definition (Suffix Link)**:
```
slink([x]) = [y] where y is the longest proper suffix of longest(x) such that [y] ≠ [x]
```

Equivalently: slink([x]) = [z] where z has length |shortest(x)| - 1.

### Intuition

The suffix link "drops" the shortest string from the equivalence class:
- State q₃ = {c, bc, abc}, shortest = "c"
- Suffix link goes to the state containing the suffix of "c" of length |c|-1 = 0
- That's ε, so slink(q₃) = q₀

For state q₄ = {ca, bca, abca}, shortest = "ca":
- Suffix link goes to the state containing "a" (suffix of "ca" of length 1)
- slink(q₄) = q₁

### Suffix Link Tree

Suffix links form a **tree** rooted at q₀:

```
                    q₀
                 /  |  \
               q₁  q₂  q₃
                |   |
               q₄  q₅
                |   |
               q₆  q₆  (Note: q₅ and q₆ both link to appropriate states)
```

Actually, let's trace the suffix links properly:

| State | Shortest | Suffix of length |shortest|-1 | Suffix Link Target |
|-------|----------|----------------------------------|-------------------|
| q₁ | a | ε | q₀ |
| q₂ | b | ε | q₀ |
| q₃ | c | ε | q₀ |
| q₄ | ca | a | q₁ |
| q₅ | cab | ab | q₂ |
| q₆ | cabc | abc | q₃ |
| q₇ | cabca | abca | q₄ |
| q₈ | cabcab | abcab | q₅ |

### Properties of Suffix Links

**Lemma 2**: Following suffix links from any state eventually reaches q₀.

**Lemma 3**: For states [x] and [y] with slink([x]) = [y]:
```
end-pos(x) ⊂ end-pos(y)  (strict subset)
```

*Proof*: Shorter strings have (weakly) more occurrences. Since [y] represents shorter strings than [x], and they're different classes, the inclusion must be strict.

**Lemma 4**: The suffix link tree has depth at most |w|.

*Proof*: Each suffix link reduces the shortest representative's length by at least 1.

## Right Contexts

The **right context** of a factor x is the set of characters that can follow x:

**Definition (Right Context)**:
```
right-context(x) = {a ∈ Σ : xa ∈ F(w)}
```

**Key Property**: All strings in an equivalence class have the **same** right context.

*Proof*: If end-pos(x) = end-pos(y), then x and y can be extended by exactly the same characters (those that appear after their shared ending positions).

This property is what makes the suffix automaton deterministic: the next state depends only on the current equivalence class, not which specific string led to it.

## Construction Complexity

**Theorem 2** (Blumer et al., 1985): The suffix automaton of a string w can be constructed in O(|w|) time and space.

The construction algorithm:
1. Process characters left-to-right
2. For each new character, create at most 2 new states
3. Update transitions and suffix links incrementally

The resulting automaton has:
- At most 2|w| - 1 states
- At most 3|w| - 4 transitions

## What's Missing?

The suffix automaton is efficient but has limitations:

1. **Non-compact edges**: Single-character transitions between all adjacent states, even when the path is deterministic.

2. **No left extensions**: The automaton only supports appending characters (right extension), not prepending (left extension).

3. **All factors vs. prime factors**: Every factor has representation, but many are redundant for substring searching.

These limitations motivate the CDAWG (compacting) and SCDAWG (adding symmetry), covered in the next documents.

## Summary

| Concept | Definition |
|---------|------------|
| Factor | Any substring w[i..j] |
| end-pos(x) | Set of positions where x ends in w |
| Equivalence | x ≡ y ⟺ end-pos(x) = end-pos(y) |
| State | Equivalence class of factors |
| Transition | δ([x], a) = [xa] |
| Suffix link | Points to longest proper suffix in different class |
| Right context | Characters that can follow a factor |

**Key insight**: The suffix automaton groups factors by their ending positions, creating a minimal DFA for substring recognition.

**Next**: [03-cdawg](03-cdawg.md) - Compacting the suffix automaton
