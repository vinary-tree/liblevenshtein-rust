# Annotated Bibliography

This document provides an annotated bibliography of the key papers that define and construct the SCDAWG and related structures.

## Primary Sources

### Blumer et al. (1987) - SCDAWG Definition

**Full Citation**:
> Blumer, A., Blumer, J., Haussler, D., McConnell, R., & Ehrenfeucht, A. (1987).
> "Complete Inverted Files for Efficient Text Retrieval and Analysis."
> *Journal of the ACM*, 34(3), 578-595.

**Key Contributions**:
- Defines the **Symmetric Compact DAWG** (SCDAWG), also called C2S
- Introduces **prime subwords** and **implications** (imps)
- Describes **IS (Inverted File) features**: freq(), locations(), find()
- Proves O(n) space complexity

**Used In This Implementation**:
- Definition of prime subwords (Section 4)
- Left extension edge definition (Section 6)
- IS features for substring occurrence counting (Section 7)

**Key Definitions from the Paper**:
```
imps(x) = γxβ  where γ, β are maximal context strings
P(S) = {imps(x) : x ∈ F(S)} = set of prime subwords
C2S = (V, E_R, E_L) where V = P(S)
```

---

### Blumer et al. (1985) - Suffix Automaton

**Full Citation**:
> Blumer, A., Blumer, J., Haussler, D., Ehrenfeucht, A., Chen, M.-T., & Seiferas, J. (1985).
> "The Smallest Automaton Recognizing the Subwords of a Text."
> *Theoretical Computer Science*, 40, 31-55.

**Key Contributions**:
- Defines the **suffix automaton** (DAWG) as the minimal DFA for substrings
- Introduces **equivalence classes** based on end-position sets
- Proves the suffix automaton has at most 2n-1 states and 3n-4 edges
- Presents O(n) construction algorithm

**Used In This Implementation**:
- Equivalence class theory (foundation for all compact structures)
- Suffix link definition and properties
- End-position set characterization

**Key Theorem**:
```
Two strings x, y belong to the same equivalence class iff end-pos(x) = end-pos(y)
```

---

### Inenaga et al. (2001) - On-line SCDAWG Construction

**Full Citation**:
> Inenaga, S., Hoshino, H., Shinohara, A., Takeda, M., & Arikawa, S. (2001).
> "On-Line Construction of Symmetric Compact Directed Acyclic Word Graphs."
> *Proceedings of the 8th International Symposium on String Processing and Information Retrieval (SPIRE)*, 96-110.

**Key Contributions**:
- **On-line O(n) algorithm** for SCDAWG construction
- **Critical insight**: sext links of CDAWG(w) = edges of CDAWG(w^rev)
- Shows left extension edges can be maintained **during** construction
- No need to build CDAWG(w^rev) explicitly

**Used In This Implementation**:
- Sext link maintenance algorithm
- Incremental left extension edge construction
- The key insight that enabled our unified construction approach

**Key Theorem (Theorem 2)**:
```
The sext link of node v in CDAWG(w) corresponds to an edge
in CDAWG(w^rev), enabling O(n) SCDAWG construction.
```

---

### Inenaga et al. (2005) - On-line CDAWG Construction

**Full Citation**:
> Inenaga, S., Hoshino, H., Shinohara, A., Takeda, M., Arikawa, S., Mauri, G., & Pavesi, G. (2005).
> "On-line construction of compact directed acyclic word graphs."
> *Discrete Applied Mathematics*, 146(2), 156-179.

**Key Contributions**:
- Detailed **on-line O(n) CDAWG construction** algorithm
- **Reference pairs** for representing implicit nodes
- **Open edges** technique for efficient extension
- **Multi-string support** with unique end markers (Section 7)
- **Node separation** and **edge redirection** operations

**Used In This Implementation**:
- Core CDAWG construction algorithm (update, split_edge, canonize)
- Reference pair representation for implicit nodes
- Multi-string dictionary support with unique terminators
- Open edge technique for O(1) extension

**Key Algorithm (Figure 17)**:
```
update(c):
    while not check_end_point(c):
        if is_implicit(active_point):
            r = split_edge()
        else:
            r = active_point.node
        create_edge(r, sink)
        update_suffix_links(r)
        active_point = canonize(suffix(active_point))
```

---

### Crochemore & Vérin (1997) - CDAWG Direct Construction

**Full Citation**:
> Crochemore, M., & Vérin, R. (1997).
> "Direct Construction of Compact Directed Acyclic Word Graphs."
> *Proceedings of the 8th Annual Symposium on Combinatorial Pattern Matching (CPM)*, 116-129.

**Key Contributions**:
- **Direct CDAWG construction** without intermediate DAWG
- Proves CDAWG has at most n+1 nodes and 2n-2 edges
- Connection between CDAWG and suffix tree

**Used In This Implementation**:
- Space complexity bounds
- Understanding of compaction process
- Relationship between CDAWG nodes and suffix tree branching points

---

### Gerdjikov et al. (2013) - WallBreaker Algorithm

**Full Citation**:
> Gerdjikov, S., Mihov, S., & Schulz, K. U. (2013).
> "A Symmetric Approach to Efficiently Computing Edit-Distance-Based Similarity of Words in a Dictionary."
> *Language Processing and Knowledge in the Web*, 70-80.

**Key Contributions**:
- **WallBreaker algorithm** for fuzzy dictionary matching
- Defines requirements for bidirectional SCDAWG operations
- Shows how to use SCDAWG for edit distance computation

**Used In This Implementation**:
- Requirements specification (Remark 1.1):
  - (1a) Substring existence check
  - (1b) Right extension: V → V·σ
  - (1c) Left extension: V → σ·V
- Validation criteria for our SCDAWG implementation

**Key Requirements**:
```
(1a) Given V ∈ Σ*, decide if V is a factor of some dictionary word
(1b) Right extension: given V is a factor, navigate to V·σ
(1c) Left extension: given V is a factor, navigate to σ·V
```

---

## Secondary Sources

### Weiner (1973) - Suffix Trees

**Full Citation**:
> Weiner, P. (1973).
> "Linear Pattern Matching Algorithms."
> *14th Annual Symposium on Switching and Automata Theory*, 1-11.

**Contribution**: First linear-time suffix tree construction algorithm.

---

### McCreight (1976) - Suffix Tree Construction

**Full Citation**:
> McCreight, E. M. (1976).
> "A Space-Economical Suffix Tree Construction Algorithm."
> *Journal of the ACM*, 23(2), 262-272.

**Contribution**: Simplified suffix tree construction with explicit suffix links.

---

### Ukkonen (1995) - On-line Suffix Tree

**Full Citation**:
> Ukkonen, E. (1995).
> "On-line Construction of Suffix Trees."
> *Algorithmica*, 14(3), 249-260.

**Contribution**: On-line suffix tree construction, inspiring similar CDAWG approaches.

---

## Relationship Between Papers

```
    Blumer et al. (1985)          Weiner (1973)
    Suffix Automaton               Suffix Tree
         │                              │
         ▼                              ▼
    ┌────────────────────────────────────────┐
    │                                        │
    │   Crochemore & Vérin (1997)           │
    │   Direct CDAWG Construction            │
    │                                        │
    └────────────────────────────────────────┘
                      │
                      ▼
    ┌────────────────────────────────────────┐
    │   Inenaga et al. (2005)               │
    │   On-line CDAWG Construction          │
    │   Multi-string Support                │
    └────────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
    Blumer et al. (1987)     Inenaga et al. (2001)
    SCDAWG Definition        On-line SCDAWG with
    IS Features              Sext Links
         │                         │
         └───────────┬─────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  OUR            │
            │  IMPLEMENTATION │
            │  (Hybrid)       │
            └─────────────────┘
                     │
                     ▼
            Gerdjikov et al. (2013)
            WallBreaker Algorithm
            (Validation Criteria)
```

## Implementation Notes

### What We Use From Each Paper

| Paper | Used For |
|-------|----------|
| Blumer (1987) | SCDAWG definition, IS features, prime subwords |
| Blumer (1985) | Equivalence class theory, suffix links |
| Inenaga (2001) | Sext link insight, left extension construction |
| Inenaga (2005) | On-line CDAWG algorithm, multi-string support |
| Crochemore (1997) | Space bounds, compaction understanding |
| Gerdjikov (2013) | WallBreaker requirements, validation |

### Our Hybrid Approach

We combine:
1. **Inenaga (2005)** on-line CDAWG construction
2. **Inenaga (2001)** sext link maintenance during construction
3. **Blumer (1987)** IS features for occurrence tracking
4. **Multi-string support** with unique end markers

This yields an SCDAWG that:
- Constructs in O(n) time
- Supports dynamic insertions
- Provides O(|pattern|) substring search
- Enables bidirectional pattern extension
- Includes IS features (freq, locations)

## Further Reading

### Surveys and Tutorials

- **Crochemore & Rytter (2003)**: "Jewels of Stringology" - comprehensive text on string algorithms
- **Gusfield (1997)**: "Algorithms on Strings, Trees, and Sequences" - classic textbook

### Related Structures

- **Suffix Arrays**: Manber & Myers (1993) - space-efficient alternative to suffix trees
- **FM-Index**: Ferragina & Manzini (2000) - compressed full-text index
- **BWT**: Burrows & Wheeler (1994) - text transform enabling compression

### Applications

- **Bioinformatics**: DNA/protein sequence alignment and searching
- **Information Retrieval**: Full-text search in document collections
- **Data Compression**: LZ-based compression using suffix structures
- **Plagiarism Detection**: Finding common substrings across documents
