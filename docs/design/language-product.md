# Language-product design

**Status:** implemented · **Feature boundary:** core API ungated; phonetic NFA adapters and `query_regex` require `phonetic-rules`

This document specifies the product that searches a dictionary for terms near a
regular language. It explains the public contract, the cost-indexed frontier,
the proof obligations, resource policy, and compatibility boundary. For a
step-by-step implementation narrative, see the [literate algorithm](../algorithms/13-language-products/README.md).

![A regular-language automaton feeds a fixed cost-indexed frontier; canonicalization removes language states already represented at a cheaper level.](../diagrams/phonetic/language-product-frontier.svg)

*The frontier has one slot per exact edit cost; NFA nondeterminism changes the
set inside a slot, not the number of slots.*

## 1. Problem and terminology

A **language automaton** recognizes a set of unit sequences. A unit is a byte,
a Unicode scalar value, a token identifier, or any other equality-comparable
symbol. A **language product** combines that recognizer with standard unit-cost
Levenshtein edits. A **frontier** is the product state after consuming one
dictionary prefix. **Canonicalization** removes a language state from a dearer
cost level when the same state is already reachable at a cheaper level.

For input sequence $`w`$, recognized language $`L`$, and ordinary
Levenshtein distance $`d`$, the required result is:

```math
d(w,L)=\min_{v\in L} d(w,v).
```

The implementation is bounded: it returns `None` when the minimum is greater
than the caller's budget $`k`$. This is distance *to a language*, not regex
matching followed by a string-distance heuristic.

## 2. Architecture

The ungated module `transducer::language` contains four pieces:

| Type | Responsibility |
|---|---|
| `LanguageAutomaton<U>` | Set operations and one-symbol transitions for a language recognizer |
| `SmallDfa<U>` | Explicit DFA for at most 31 states, using a `u32` state-set bit mask |
| `LanguageProduct<U, L>` | Standard-edit transition kernel and canonical cost frontier |
| `LanguageQueryIterator<N, L>` | Iterative dictionary product, exact frontier interning, observed-edge caching, and lazy `LanguageMatch` emission |

The phonetic feature implements `LanguageAutomaton<char>` for `NFAChar` and
`LanguageAutomaton<u8>` for `NFA`. `Transducer::query_regex` compiles a pattern
to `NFAChar`, enforces the untrusted-input state policy, and delegates to the
same generic query. Nothing in the core module depends on the phonetic parser.

The legacy byte `ProductAutomaton` remains a compatibility wrapper because its
public `with_algorithm` constructor also supports optimal string alignment and
merge-and-split semantics. Its Standard `min_distance` and `accepts` paths now
delegate to `LanguageProduct`; only the extra algorithm variants retain the
legacy search kernel. Replacing those variants with a Standard-only alias would
silently change public behavior.

## 3. LanguageAutomaton contract

For state sets $`A`$ and $`B`$, unit $`u`$, matching transition $`\delta`$,
and arbitrary consuming transition $`\alpha`$, implementations must satisfy:

```math
\delta(A\cup B,u)=\delta(A,u)\cup\delta(B,u),
\qquad
\alpha(A\cup B)=\alpha(A)\cup\alpha(B).
```

`union_into` must be set union; `subtract` must be set difference; `empty` must
be the identity for union. `initial`, `step`, and `advance` include any required
zero-width closure. `is_accepting(S)` means that at least one state in $`S`$
accepts the empty continuation. `state_count` exposes resource-policy data but
does not trigger determinization.

These are semantic requirements. The representation may be a scalar bit mask,
a fixed bit set, a sparse set, or another canonical structure.

## 4. Cost-indexed frontier

`Frontier<S>` stores exactly $`k+1`$ optional state sets. Slot `levels[e]`
contains states reachable at exact cost $`e`$ after cheaper duplicates have
been removed. Its invariant is:

```math
0\le e<f\le k \Longrightarrow S_e\cap S_f=\varnothing.
```

At one input unit, level $`e`$ contributes:

- a match at level $`e`$ through `step`;
- an insertion at level $`e+1`$ without moving the language;
- a substitution at level $`e+1`$ through `advance`.

The deletion closure repeatedly applies `advance` without consuming input,
placing its result at the next cost. All arithmetic is guarded before `level +
1`; because $`k`$ is `u8`, there are at most 256 levels.

### 4.1 Merge proof

Suppose two histories arrive at the same cost $`e`$ with state sets $`A`$ and
$`B`$. The union law gives identical future recognition whether they are kept
separately or represented by $`A\cup B`$. Edit-cost updates depend only on the
operation and level, not on which member of the state set was selected.
Therefore unioning equal-cost histories neither loses nor invents a path.

### 4.2 Cross-level dominance proof

If state $`q`$ occurs at costs $`e<f`$, every continuation available from
$`(q,f)`$ is also available from $`(q,e)`$ with total cost smaller by
$`f-e`$. Standard edit costs are non-negative, so future steps cannot reverse
that ordering. Removing $`q`$ from level $`f`$ preserves the least accepting
cost. Rocq and Verus prove the two-level induction step; property tests execute
the full frontier law.

## 5. Dictionary traversal

![The iterator advances the product before enqueuing a child and prunes an empty frontier immediately.](../diagrams/traversal/frontier-pruned-walk.svg)

*D1 is closed at the edge: an empty next frontier prevents descent into the
entire child subtree.*

`LanguageQueryIterator` uses an explicit queue rather than recursion. A pending
entry owns a dictionary path trace and one machine-word `LanguageFrontierId`.
The complete cost-indexed frontier is stored once in a query-local arena. A
collision-checked exact interner maps equal canonical frontiers to the same ID,
and an observed-edge table maps `(source ID, exact unit)` to the target ID or a
dead transition. Paths are materialized only for accepted nodes. The iterator
has no fixed depth 100 and cannot overflow the process call stack merely
because a dictionary key is deep.

Interning is an on-demand representation optimization, not eager subset
construction. The arena contains only distinct frontiers reached through real
dictionary edges. Its size is finite for a finite automaton and edit cutoff,
but the convenience iterator does not impose a caller-configurable arena or
queue ceiling. A service that treats complete empty as release evidence must
place this engine behind the bounded/tagged adapter described in
[the shared product architecture](lazy-online-products.md#6-resource-and-failure-semantics).

The dictionary graph must present a finite traversal. Tries and directed
acyclic word graphs satisfy this directly. A backend whose `edges()` relation
contains reachable cycles must provide a finite-node visitation policy; this
iterator intentionally does not merge dictionary nodes across distinct prefixes
because those prefixes produce distinct returned terms.

With `perf-instrumentation`, `LanguageQueryStats` records nodes visited and
edges enumerated. The counters are zero-cost-disabled in normal builds.

## 6. Complexity

Let $`Q`$ be the language-state set, $`k`$ the edit budget, $`E_D`$ the
dictionary edges actually explored, and $`W_Q`$ the machine words needed for a
state set. One canonical frontier occupies:

```math
\mathcal{O}(kW_Q)
```

and one pending product entry stores only an ID plus its dictionary path trace.
A cache miss costs $`\mathcal{O}(kW_Q)`$ set work plus the recognizer's
transition work; a cache hit is an ID lookup. In the no-reuse case traversal is:

```math
\mathcal{O}(E_D k W_Q).
```

Repeated `(source ID, exact unit)` observations reduce this work. The bound is
independent of product-history multiplicity, but it does not make
regular-language intersection immune to subset diversity. In the worst case,
different dictionary prefixes can still induce exponentially many distinct NFA
subsets across the traversal. A lazy subset-DFA cache is a compatible future
optimization, not a correctness requirement.

## 7. Public API and examples

The general entry point accepts any automaton whose unit matches the dictionary:

```rust
use libdictenstein::dynamic_dawg::DynamicDawgU64;
use liblevenshtein::transducer::language::SmallDfa;
use liblevenshtein::transducer::{Algorithm, Transducer};

let dictionary = DynamicDawgU64::<()>::new();
dictionary.insert_sequence(&[10, 20]);
dictionary.insert_sequence(&[10, 30]);

let mut language = SmallDfa::new();
let q1 = language.add_state(false).unwrap();
let q2 = language.add_state(true).unwrap();
language.add_transition(0, 10_u64, q1).unwrap();
language.add_transition(q1, 20_u64, q2).unwrap();

let transducer = Transducer::new(dictionary, Algorithm::Standard);
let matches: Vec<_> = transducer.query_language(language, 1).collect();
assert_eq!(matches.len(), 2);
```

With `phonetic-rules`, a character dictionary can use:

```rust
# use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;
# use liblevenshtein::transducer::{Algorithm, Transducer};
let dictionary = DoubleArrayTrieChar::from_terms(["ab", "ac", "cab"]);
let transducer = Transducer::new(dictionary, Algorithm::Standard);
let matches: Vec<_> = transducer
    .query_regex("a(b|c)", 1)
    .expect("valid bounded regular expression")
    .collect();
```

`query_language` always uses Standard unit-cost language distance. The
`Algorithm` stored in `Transducer` and its substitution policy do not alter the
product. This separation is explicit to avoid implying unsupported OSA,
merge-and-split, or articulatory semantics.

## 8. Resource and security policy

`SmallDfa` rejects the 32nd real state because bit 31 is reserved. For regexes,
`query_regex` enforces `LANGUAGE_PRODUCT_MAX_STATES = 4096` in three stages:

1. a conservative source-length ceiling rejects pathological input before a
   left-deep AST is built;
2. a saturating Thompson-state estimate expands counted repetition and group
   references before NFA construction;
3. the constructed NFA is checked again as defense in depth.

Compact inputs such as `a{1000000}` are rejected by the second stage without
allocating the expanded NFA. `LanguageProduct::new` itself is intentionally
unrestricted for trusted, programmatically constructed automata; applications
that accept custom automata must impose their own state policy. See
[resource-exhaustion guidance](../security/resource-exhaustion.md).

## 9. Verification and executable invariants

The formal evidence is deliberately redundant:

- Verus checks the Rust-facing arithmetic, canonicalization, and union laws;
- assumption-free Rocq proves relational-image distribution, acceptance
  preservation, disjointness, and the 256-level bound;
- Z3 and cvc5 independently prove four bounded counterexample queries UNSAT;
- property tests pin exact frontier-ID and observed-edge reuse, compare literal
  DFAs to the scalar reference, NFAs to the legacy product, regex queries to
  brute force, and byte/character/token paths;
- the instrumented 5,000-term regression requires more than a tenfold edge
  reduction versus a full scan.

The source-of-truth inventory is
[`FORMAL_VERIFICATION_MANIFEST.tsv`](../verification/FORMAL_VERIFICATION_MANIFEST.tsv).

## 10. References

- K. Thompson, “Programming Techniques: Regular expression search algorithm,”
  *Communications of the ACM* 11(6), 1968.
  [doi:10.1145/363347.363387](https://doi.org/10.1145/363347.363387)
- R. A. Wagner and M. J. Fischer, “The string-to-string correction problem,”
  *Journal of the ACM* 21(1), 1974.
  [doi:10.1145/321796.321811](https://doi.org/10.1145/321796.321811)
- M. Mohri, F. Pereira, and M. Riley, “Weighted finite-state transducers in
  speech recognition,” *Computer Speech & Language* 16(1), 2002.
  [doi:10.1006/csla.2001.0184](https://doi.org/10.1006/csla.2001.0184)
