# Symmetric Compact DAWG (SCDAWG) Theory

The **Symmetric Compact DAWG** (SCDAWG), also called **C2S** (Compact Symmetric), extends the CDAWG with **left extension edges**, enabling bidirectional pattern navigation. This document covers the theoretical foundations of the SCDAWG as defined by Blumer et al. (1987).

## Motivation: Bidirectional Search

The CDAWG supports efficient right extension:
```
Given pattern V, navigate to V·σ (append character σ)
```

But many algorithms need **left extension**:
```
Given pattern V, navigate to σ·V (prepend character σ)
```

### WallBreaker Example

The WallBreaker algorithm (Gerdjikov et al., 2013) for fuzzy dictionary matching requires:

1. **Substring check**: Is V a substring of some dictionary word?
2. **Right extension**: From V, reach V·σ
3. **Left extension**: From V, reach σ·V

Without left extension, WallBreaker cannot efficiently grow pattern matches toward the left, limiting its applicability.

## Left Context and Right Context

### Right Context (Review)

The **right context** of factor x is the set of strings that can follow x:

**Definition**:
```
right-context(x) = {y ∈ Σ* : xy ∈ F(w)}
```

For single characters:
```
right-context₁(x) = {a ∈ Σ : xa ∈ F(w)}
```

### Left Context

The **left context** of factor x is the set of strings that can precede x:

**Definition**:
```
left-context(x) = {y ∈ Σ* : yx ∈ F(w)}
```

For single characters:
```
left-context₁(x) = {a ∈ Σ : ax ∈ F(w)}
```

### Example for "abcabcab"

| Factor x | left-context₁(x) | right-context₁(x) |
|----------|------------------|-------------------|
| a | {ε, c} | {b} |
| b | {a} | {c, $} |
| c | {b} | {a, $} |
| ab | {ε, c} | {c, $} |
| bc | {a} | {a, $} |
| ca | {b} | {b} |
| abc | {ε, c} | {a, $} |

Where ε represents the empty context (factor at string boundary).

## Prime Subwords and Implications

### Implication (imps)

For any factor x, its **implication** is the maximal string where every occurrence of x is embedded:

**Definition (Implication)**:
```
imps(x) = γxβ

where:
- γ is the longest string such that: if x occurs at position i, then γ occurs at position i-|γ|
- β is the longest string such that: if x occurs at position i, then β occurs at position i+|x|
```

In other words, imps(x) is the longest string that occurs exactly where x occurs.

### Properties of Implications

**Lemma 1**: imps(x) is unique and well-defined.

**Lemma 2**: end-pos(x) = end-pos(imps(x))

*Proof*: By definition, imps(x) occurs exactly where x occurs, so they have identical end-positions.

**Lemma 3**: |imps(x)| ≥ |x|

*Proof*: imps(x) contains x (γxβ ⊇ x).

### Example: Implications for "abcabcab"

| Factor x | Occurrences | γ | β | imps(x) |
|----------|-------------|---|---|---------|
| a | 0,3,6 | ε | b | ab |
| b | 1,4,7 | a | ε | ab |
| ab | 0,3,6 | ε | ε | ab |
| c | 2,5 | ab | ab | abcab |
| bc | 1,4 | a | a | abca |
| abc | 0,3 | ε | a | abca |
| ca | 2,5 | ab | b | abcab |

**Observation**: "a", "b", and "ab" all have imps = "ab". This is because:
- Every 'a' is followed by 'b'
- Every 'b' is preceded by 'a'
- So imps(a) = imps(b) = imps(ab) = "ab"

### Prime Subwords

A factor x is a **prime subword** (or simply **prime**) if it equals its own implication:

**Definition (Prime Subword)**:
```
x is prime ⟺ imps(x) = x
```

**Definition (Prime Set)**:
```
P(w) = {x ∈ F(w) : x is prime} = {imps(y) : y ∈ F(w)}
```

### Properties of Prime Subwords

**Lemma 4**: The prime subwords are exactly the **longest representatives** of equivalence classes in the CDAWG.

*Proof*:
- If x = longest([x]), then no extension of x shares the same end-positions
- Therefore γ = β = ε in the implication
- So imps(x) = x, making x prime

**Lemma 5**: |P(w)| ≤ |w| + 1 (same bound as CDAWG nodes)

**Lemma 6**: For any factor x, imps(x) ∈ P(w)

### Prime Subwords for "abcabcab"

| Prime Subword | Equivalence Class |
|---------------|-------------------|
| ε | {ε} |
| ab | {a, b, ab} |
| abca | {bc, abc, bca, abca} |
| abcab | {c, ca, cab, bcab, abcab} |
| abcabc | {cabc, bcabc, abcabc} |
| abcabca | {cabca, bcabca, abcabca} |
| abcabcab | {cabcab, bcabcab, abcabcab} |

Each prime subword is the longest (and maximal) representative of its class.

## SCDAWG Definition

### Formal Definition

**Definition (SCDAWG)** from Blumer et al. (1987):

The **Symmetric Compact DAWG** of string w is the structure **C2S(w) = (V, E_R, E_L)** where:

- **V = P(w)** = set of prime subwords

- **E_R** = Right extension edges:
  ```
  E_R = {(x, imps(xa)) : x ∈ P(w), a ∈ Σ, xa ∈ F(w)}
  ```
  Label: derived from transition (first character + suffix)

- **E_L** = Left extension edges:
  ```
  E_L = {(x, imps(ax)) : x ∈ P(w), a ∈ Σ, ax ∈ F(w)}
  ```
  Label: derived from transition (prefix + first character)

### Edge Labels

Edge labels in the SCDAWG are not single characters but **substrings**.

**Right Extension Edge** from x to y = imps(xa):
```
Label = a || β_y

where β_y is the right context suffix added by imps
```

More precisely: if x·a leads to imps(xa) = γ(xa)β, then the label captures the transition.

**Left Extension Edge** from x to y = imps(ax):
```
Label = γ_y || a

where γ_y is the left context prefix added by imps
```

### Visual Representation

For a prime subword P, its edges form:

```
        Left Extensions                Right Extensions
        ──────────────                 ────────────────

            ╭─ σ₁·P ←── σ₁ ──╮    ╭── σ₁ ──→ P·σ₁ ─╮
            │                 │    │                 │
     ... ←──┤       P        ├────┤        P        ├──→ ...
            │                 │    │                 │
            ╰─ σ₂·P ←── σ₂ ──╯    ╰── σ₂ ──→ P·σ₂ ─╯

```

Each prime subword has:
- Right edges for each valid right extension character
- Left edges for each valid left extension character

## The Symmetry Property

The SCDAWG is "symmetric" in a precise sense:

**Theorem 1 (Symmetry)**:
```
CDAWG(w) with left extension edges = CDAWG(w^rev) with reversed edge direction
```

Where w^rev is the reversal of w.

### Sext Links = CDAWG(w^rev) Edges

**Definition (Sext Link)**: The **shortest extension link** (sext link) from node x is the edge in CDAWG(w^rev) that corresponds to x.

**Theorem 2** (Inenaga et al., 2001):
```
Left extension edges of CDAWG(w) = Edges of CDAWG(w^rev) (with reversed direction)
```

This is a crucial insight: **we can derive left extension edges from the CDAWG of the reversed string**.

### Implications for Construction

This symmetry means:
1. Build CDAWG(w) normally
2. Left extension edges can be derived from suffix link structure
3. No need to explicitly build CDAWG(w^rev)

## Connection to Suffix Links

### Reversed Suffix Links

**Lemma 7**: If slink(x) = y in the CDAWG, then there exists a left extension edge from y to x.

*Proof sketch*:
- slink(x) = y means y is a suffix of x
- x = αy for some non-empty α
- The first character of α provides the left extension from y to x

### Building Left Extensions from Suffix Links

For each suffix link slink(x) = y:
```
x = α · y  (for some prefix α)

Left extension edge: y --first(α)--> x
```

Where first(α) is the first character of α.

**Algorithm**:
```
for each node x in CDAWG:
    if slink(x) = y exists:
        α = x[0..|x|-|y|]  // prefix dropped by suffix link
        add left_edge(y, first(α)) = x
```

## SCDAWG for "abcabcab"

### Nodes (Prime Subwords)

| Node | Prime Subword | Length |
|------|---------------|--------|
| v₀ | ε | 0 |
| v₁ | ab | 2 |
| v₂ | abca | 4 |
| v₃ | abcab | 5 |
| v₄ | abcabc | 6 |
| v₅ | abcabca | 7 |
| v₆ | abcabcab | 8 |

### Right Extension Edges

| From | Char | To | Label |
|------|------|----|-------|
| v₀ | a | v₁ | ab |
| v₀ | b | v₁ | ab |
| v₀ | c | v₃ | abcab |
| v₁ | c | v₂ | ca |
| v₂ | b | v₃ | b |
| v₃ | c | v₄ | c |
| v₄ | a | v₅ | a |
| v₅ | b | v₆ | b |

### Left Extension Edges

| From | Char | To | Label |
|------|------|----|-------|
| v₀ | a | v₁ | ab |
| v₀ | b | v₁ | ab |
| v₀ | c | v₃ | abcab |
| v₁ | c | v₂ | ca |
| v₂ | b | v₃ | b |
| v₃ | c | v₄ | c |
| v₄ | a | v₅ | a |
| v₅ | b | v₆ | b |

Note: For this particular string, right and left extensions have similar structure due to its repetitive nature.

### ASCII Diagram

```
SCDAWG for "abcabcab":

                      RIGHT EXTENSIONS (→)
                      ════════════════════

     ε ─────ab────→ ab ────ca────→ abca ────b────→ abcab
     │               │               │               │
     │               │               │               │
     └───abcab───────────────────────┘               │
                                                     │
     abcab ────c────→ abcabc ────a────→ abcabca ────b────→ abcabcab


                      LEFT EXTENSIONS (←)
                      ═══════════════════

     ε ←────ab───── ab ←────ca───── abca ←────b───── abcab
     │               │               │               │
     │               │               │               │
     └───────────────────────────────abcab───────────┘

     abcab ←────c───── abcabc ←────a───── abcabca ←────b───── abcabcab
```

## Complexity Analysis

**Theorem 3** (Blumer et al., 1987):
For string w of length n:
- SCDAWG(w) has at most **n + 1 nodes**
- SCDAWG(w) has at most **4n - 4 edges** (2n-2 right + 2n-2 left)

Space is `𝒪(n)`, same as CDAWG but with doubled edge count.

## Comparison: Left Extension vs Backward Edges

A common implementation mistake is confusing **left extension edges** with **backward edges** (reverse of forward edges):

### Backward Edges (WRONG for bidirectional search)

If we have forward edge A →c→ B:
```
Backward edge: B →c→ A
```

This just reverses the forward path. It does NOT implement left extension.

### Left Extension Edges (CORRECT)

Left extension edge from A to B with label σ:
```
A represents pattern "xyz"
B represents pattern "σxyz" (σ prepended, NOT appended)
```

This is a fundamentally different operation.

### Example

Consider pattern "ab" (node for prime "ab"):
- **Right extension** 'c': leads to pattern "abc"
- **Left extension** 'c': leads to pattern "cab"
- **Backward edge** (wrong): would try to go "back" to 'a' or 'b'

Backward edges traverse the same strings in reverse. Left extensions navigate to DIFFERENT strings with characters prepended.

## WallBreaker Requirements Satisfied

The SCDAWG satisfies all WallBreaker requirements from Gerdjikov et al. (2013):

| Requirement | Operation | SCDAWG Support |
|-------------|-----------|----------------|
| **(1a)** | Is V a substring? | Follow right edges from root; success = found |
| **(1b)** | Right extend V → V·σ | Follow right extension edge labeled with σ |
| **(1c)** | Left extend V → σ·V | Follow left extension edge labeled with σ |

All operations complete in `𝒪(∣label∣)` time, where label is the edge label length.

## Summary

| Concept | Definition |
|---------|------------|
| Left context | Characters that can precede a factor |
| Right context | Characters that can follow a factor |
| Implication imps(x) | Maximal γxβ with same occurrences as x |
| Prime subword | Factor equal to its implication |
| Right extension edge | From x to imps(xa), appending |
| Left extension edge | From x to imps(ax), prepending |
| Symmetry | Left edges = CDAWG(w^rev) edges |

**Key insight**: The SCDAWG enables bidirectional pattern growth by adding left extension edges derived from the suffix link structure.

**Next**: [05-construction](05-construction.md) - On-line algorithm to build the SCDAWG
