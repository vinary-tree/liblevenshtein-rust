# Lazy synchronized products and stable online automata

**Status:** production architecture · **Scope:** string, regular-language, and
elastic time-series automata · **Evidence boundary:** named formal obligations
and executable correspondence tests only

`liblevenshtein` does not build a complete edit-distance automaton and then
cross it with a complete dictionary automaton. It constructs the reachable
part of their **synchronized product** on demand. A dictionary edge supplies a
label; the query machine advances on that same label; the child is explored
only if a viable query state survives. When both components are acceptors, the
language accepted by this product is their **language intersection**. Thus
*product* names the operational machine construction, while *intersection*
names the resulting accepted language.

![A dictionary edge asks for one transition of a compact canonical query state; dead transitions prune a subtree, while relaxed temporal finals undergo full-precision verification.](../diagrams/architectures/lazy-synchronized-product.svg)

This document defines that architecture, its online-memory contract, its
performance rationale, its formal obligations, and the boundaries where a
different algorithm is intentionally used.

## 1. Terms and mathematical model

A **dictionary automaton** $`D`$ is a trie, directed acyclic word graph
(DAWG), or another finite labelled graph whose accepting paths spell stored
keys. A **query automaton** $`A`$ recognizes labels within a configured cost
of one fixed query. A **frontier** is the canonical set of query positions that
remain viable after a dictionary prefix. A **continuation context** is any
extra information that changes future transitions, such as an affine-gap
layer, a pending transposition, or the previous target sample in TWED.

For two deterministic labelled transition systems, the synchronized product
has states and transitions:

```math
Q_{D\otimes A}=Q_D\times Q_A,
\qquad
(d,a)\xrightarrow{u}(\delta_D(d,u),\delta_A(a,u)).
```

The implementation never allocates $`Q_D\times Q_A`$. It starts at the root
pair and constructs only pairs reachable through real dictionary edges whose
query transition is live. If both components are acceptors, then:

```math
L(D\otimes A)=L(D)\cap L(A).
```

For an elastic time-series index, the query side initially recognizes a
relaxed interval language. A final interval state is only a candidate gate;
the reported answer comes from an exact scorer over every stored
full-precision series in the quantization-collision bucket.

Finalization and cutoff admission are deliberately separate operations. A
finalizer may close trailing query-only edits after the last dictionary edge,
so a finite exact score can still exceed the configured cutoff. Every public
range scheduler must apply $`d\leq\tau`$ to the exact final score before
emission. `ZipperQueryIterator` pins the former true-Damerau counterexample
query `"a"`, dictionary key `""`, cutoff zero: finalization yields one, and
the iterator must reject it.

## 2. Canonical frontier and compact identity

The fundamental queue entry is conceptually:

```rust
struct ProductFrame<Cursor, StateId> {
    dictionary: Cursor,
    query_state: StateId,
}
```

`StateId` is a machine-word identifier into a query-local arena. The queue
does not own or clone a DP row. A state is constructed only after an observed
edge misses the transition cache, then normalized in this order:

1. generate only reachable consuming successors;
2. take the required zero-target-consumption closure;
3. discard positions above the cutoff;
4. apply only kernel-specific, proved subsumption;
5. sort and deduplicate the exact representation;
6. collision-check and intern the canonical key.

The transition cache is keyed by $`(\mathrm{StateId},\text{ label class})`$. A label class
may be a characteristic-vector class for strings, one exact alphabet unit for
a language product, or a canonical interval-bin identity for an elastic
index. Repeated sibling edges reuse the transition without reconstructing the
frontier.

### 2.1 Why exact interning matters

Hash equality is an accelerator, not an authority. A fingerprint selects a
small collision bucket; exact canonical-state equality decides reuse. For
floating costs, negative zero is normalized and the exact ordered bit
representation participates in identity. Approximate equality is prohibited:
merging two nearly equal costs can remove the only exact survivor.

Permutation independence is part of the contract. If $`C`$ is the
canonicalization function and $`\pi`$ a permutation of generated positions,
then:

```math
C(P)=C(\pi(P)).
```

This makes state IDs deterministic with respect to transition semantics even
when a hash seed or predecessor enumeration order changes.

## 3. Subsumption is a residual-language proof

Two positions may share a control coordinate yet have different future
behavior. Therefore a generic “lower cost wins” rule is unsound unless their
continuation contexts agree or a stronger simulation theorem applies.

Let $`R(p,z)`$ be the least additional cost for position $`p`$ to consume
future target suffix $`z`$ and finish. Position $`p`$ safely dominates
position $`q`$ only if:

```math
\operatorname{cost}(p)\otimes R(p,z)
\le
\operatorname{cost}(q)\otimes R(q,z)
\quad\text{for every suffix }z.
```

A reusable sufficient condition is a zero-target-consumption path from
$`p`$ to $`q`$ with known cost. Appending any suffix run to that path proves
the inequality. Each kernel supplies its own continuation identity and
epsilon-reachability cost; the engine does not import the classic
Levenshtein positional formula into MSM, TWED, ERP, or Fréchet.

## 4. Literate dictionary-product algorithm

**Purpose.** Enumerate exact dictionary matches without materializing the
Cartesian product or cloning a query frontier into every pending node.

**Invariant.** Every pending pair contains a dictionary cursor for one real
prefix and a canonical state ID denoting exactly the viable query positions
after that prefix.

```text
ALGORITHM LAZY-PRODUCT(dictionary, machine, cutoff, limits)
  session <- captureImmutableDictionaryRevision(dictionary)
  rootId  <- machine.intern(machine.seed(cutoff))
  pending <- one frame (session.root, rootId)

  while pending is not empty do
    frame <- removeOne(pending)

    if dictionaryFinal(frame.cursor) then
      final <- machine.finalize(frame.stateId)
      if final is finite and final is within cutoff then
      for every exact original represented by this final key do
        if exactScore(original) is within cutoff then emit exact result

    for every real edge (label, child) from frame.cursor do
      class <- machine.classify(label)
      nextId <- transitionCache.get(frame.stateId, class)
      if nextId was not observed then
        generated <- machine.generateReachable(frame.stateId, class)
        closed    <- machine.epsilonClose(generated)
        canonical <- machine.provedAntichain(closed, cutoff)
        nextId   <- machine.exactIntern(canonical)
        transitionCache.put(frame.stateId, class, nextId)

      if nextId is live then
        preflight queue, state, work, and scratch ceilings
        add (child, nextId) to pending

  return COMPLETE only after pending and deferred work are exhausted
```

The implementation is iterative. Depth-first products use an explicit frame
stack and reclaim paired scratch state on pop. Breadth-first, ranked, and
best-first products use explicit queues or heaps. No dictionary key depth can
overflow the process call stack.

## 5. Stable unknown-length target streams

An **online automaton** in this library fixes a finite query and consumes target
labels one at a time. It commits one generation only after validation and
resource preflight succeed, then reuses the previous generation as scratch.
It retains no consumed target prefix.

![A fixed query, bounded cache, current frontier, and scratch frontier suffice regardless of how many target labels were previously consumed.](../diagrams/architectures/online-retention-contract.svg)

For fixed query $`q`$, live frontier $`F`$, and configured cache $`C`$, the
retained-memory contract is:

```math
M=\mathcal{O}(|q|+|F|+|C|),
```

independent of the consumed target-prefix length. The exact representation
varies:

- Levenshtein universal and parameterized machines retain one canonical
  frontier plus fixed query metadata.
- Scalar and vector elastic machines retain current/next query-width
  generations and active-position IDs.
- Generalized operation automata retain $`r+1`$ committed rows plus one
  scratch row, where $`r`$ is the maximum target consumption of any compiled
  operation; they also retain only the corresponding finite target lookback.
- Rolling precursor queries retain an explicitly bounded window and expose a
  continuation when the work budget is exhausted.

This guarantee does **not** claim constant-memory exact distance between two
histories that both grow without bound. Such a problem generally requires
retaining history or accepting a windowed/approximate contract. Nor does it
make an unbounded result collection safe: a caller that collects an infinite
output stream necessarily owns unbounded output.

## 6. Resource and failure semantics

“Stable” has two parts. First, retained automaton memory is independent of
consumed target history. Second, production adapters place explicit ceilings
on every other dimension that can legitimately grow:

- query/window length;
- frontier positions and interned states;
- transition-cache entries;
- dictionary queue/stack entries;
- work steps;
- results and collision-bucket verification;
- witness operations;
- scratch bytes and arithmetic.

Checked arithmetic precedes allocation, and fallible reservation precedes
mutation. Validation, overflow, allocation, or budget failure is a tagged
incomplete outcome. A complete empty result means only that the finite search
was exhausted and no survivor existed. Convenience iterators without these
ceilings are not release evidence for Regresspec.

## 7. Architecture audit matrix

| Surface | Product/frontier architecture | Unknown-target retention | Trusted use boundary |
|---|---|---|---|
| Standard, OSA, merge/split Levenshtein | Canonical `UnitCostFrontier` IDs; observed characteristic transitions | Fixed query; finite canonical state space | Standard is metric; OSA is nonmetric; use each documented algorithm domain |
| Unrestricted Damerau–Levenshtein | Lazy continuation positions with $`\mathcal{O}(k^2)`$ frontier envelope | Fixed query; no target prefix | Metric typed surface; representation budget enforced |
| Weighted string query | Exact-interned `GeneratedStateIdF64`; queued machine word | Fixed query; finite cutoff state space | Metric claims require validated symmetric positive costs |
| Affine-gap query | Shared canonical unit-state arena with affine configuration | Fixed query; no target prefix | Exact Gotoh-style scorer; metricity is not implied by “affine” |
| Regular-language distance | Exact-interned cost-indexed language frontier ID | Fixed finite language automaton and cutoff | Standard unit edits only; arbitrary NFA subset diversity still needs limits |
| Universal Levenshtein variants | `UniversalOnlineAutomaton` advances one characteristic vector at a time | Fixed word, canonical state, scalar counter | Reference/correspondence surface; claim only named proved variants |
| Generalized operations | Finite-lookback row ring in `GeneralizedOnlineAutomaton` | $`r+1`$ rows and $`r`$ target labels | Arbitrary operation sets are not automatically metrics |
| MSM, ERP, unit-grid TWED, scalar Fréchet | Sparse temporal frontier IDs for dictionary products; two generations online | Fixed query; no target prefix | Metric status follows typed domain; legacy MSM wavefront prohibited |
| Timestamped TWED | Exact online point automaton with strict typed timestamps | Current/next state plus previous timestamped target point | Finite, strictly monotone, common-unit timestamps only |
| Vector Fréchet | Whole vector labels; current/next query-width generations | Fixed query and dimension | Stutter-quotient metric domain only; coordinates never flattened |
| Banded DTW | Same stable elastic online engine | Fixed query and band | Exact diagnostic challenger; never a metric marker |
| Approximate quantized `TimeSeriesIndex` | Delegates to the compact production Levenshtein product | Dictionary query, not stream history | Candidate/advisory use only; never absence evidence |
| Prefix/subsequence traversal | Explicit DFS stack; scalar or compact frontier per active depth | Dictionary-depth memory, not target-history memory | Structural query surfaces; result paths are materialized only on demand |
| Context-dependent edit costs | Iterative dictionary traversal with a full query column per pending prefix | Finite dictionary only | Nonmetric model may inspect the entire prefix; excluded from metric-state sharing |
| Articulatory phonetic full scan | Finite dictionary scan and exact terminal rescoring | Bounded dictionary depth policy | Fractional context-dependent compatibility path, not a metric product claim |
| Soft-DTW | Two-row batch analysis scorer | Finite operands only | Analysis challenger; no exact antichain retrieval or absence claim |

The old `Intersection`, `IntersectionF64`, and `AutomatonZipper` values remain
manual-navigation or compatibility types. Their ownership of one full state is
appropriate for one cursor controlled by the caller; production dictionary
query queues do not use them as per-node frontier storage.

## 8. Performance design

The architecture removes work at five levels:

1. **Prefix sharing.** A dictionary prefix is traversed once rather than
   recomputed for every terminal below it.
2. **Query-first edge projection.** A dictionary backend presents a label to
   the query transition before constructing the corresponding child focus;
   an automaton-rejected edge therefore allocates no zipper path or child
   wrapper.
3. **Sparse generation.** Only query positions reachable for the observed
   label are generated.
4. **Canonical interning.** Equal residual languages share one state ID.
5. **Transition reuse.** Equal $`(\mathrm{StateId},\text{ label class})`$ observations share
   one generated target.

Prepared sibling rows reuse scratch allocations and query classification while
enumerating one dictionary node's edges. Active-position arrays are sorted and
generation-stamped, avoiding per-transition hash maps. The exact asymptotic
frontier width is kernel-specific: no constant-width claim is made for ERP or
Fréchet, while positive-stiffness TWED and bounded edit distance admit stronger
cutoff-derived bands.

`ZipperQueryIterator` consumes its initial focus into an opaque
`ZipperTraversalNode` and delegates to the same `QueryIterator` core used by
ordinary dictionary nodes. PathMap may then erase its root-relative path
buffer, because the opaque node surface exposes only descent and finality and
the scheduler's parent arena reconstructs result paths relative to the supplied
focus. The immutable `TrieRef` remains the snapshot owner. This erasure is an
optimization, not a relaxed semantic contract: projection order, successful
children, finality, snapshot identity, result order, and result paths must be
observationally identical to ordinary zipper traversal.

Optimization must preserve exact state identity, the chosen tie order, and the
formal recurrence. Benchmark improvement alone cannot justify a new
subsumption rule.

## 9. Formal and executable evidence

The formal campaign proves reusable architecture obligations separately from
kernel recurrences:

- epsilon reachability implies residual-language simulation;
- canonicalization is permutation-independent and produces an antichain;
- exact canonical-key reuse is semantics preserving;
- additive and bottleneck interval steps lower-simulate concrete steps;
- exact leaf verification prevents abstract false positives;
- exact finalizer scores are emitted only after an explicit cutoff check;
- pause/resume observations equal uninterrupted execution;
- complete empty is possible only after exhaustion;
- current/next generation retention is prefix independent;
- DFS frame/state push and pop preserve a live-path bijection;
- erasing path-only zipper context preserves native focus and snapshot identity;
- projecting the query transition before dictionary descent constructs exactly
  the live product children and no rejected child;
- rejected scratch preflight is transactional;
- generalized finite lookback contains every predecessor of the next row.

The Rust property layer then connects those theorems to implementation:

1. compare every committed prefix to an independent full-matrix or sparse-grid
   oracle;
2. enumerate small alphabets and operation sets exhaustively;
3. randomize predecessor order and hash seeds;
4. pin omission, context-removal, approximate-equality, final-admission, and
   cutoff mutants;
5. compare exact range and kNN to brute force;
6. run deep streams to demonstrate constant retained state and iterative stack
   behavior;
7. replay every emitted alignment witness to the returned exact cost;
8. corrupt snapshots and budget boundaries to require fail-closed outcomes.

The authoritative inventory is
[`FORMAL_VERIFICATION_MANIFEST.tsv`](../verification/FORMAL_VERIFICATION_MANIFEST.tsv).
It deliberately labels remaining legacy MSM and partial universal proof debt;
this campaign does not claim that every historical MSM theorem or every
automaton surface is fully verified.

## 10. References

- K. U. Schulz and S. Mihov, “Fast String Correction with
  Levenshtein-Automata,” *International Journal on Document Analysis and
  Recognition* 5(1), 2002.
  [doi:10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)
- R. A. Wagner and M. J. Fischer, “The string-to-string correction problem,”
  *Journal of the ACM* 21(1), 1974.
  [doi:10.1145/321796.321811](https://doi.org/10.1145/321796.321811)
- P. Marteau, “Time Warp Edit Distance with Stiffness Adjustment for Time
  Series Matching,” *IEEE Transactions on Pattern Analysis and Machine
  Intelligence* 31(2), 2009.
  [doi:10.1109/TPAMI.2008.76](https://doi.org/10.1109/TPAMI.2008.76)
- A. Stefan, V. Athitsos, and G. Das, “The Move-Split-Merge Metric for Time
  Series,” *IEEE Transactions on Knowledge and Data Engineering* 25(6), 2013.
  [doi:10.1109/TKDE.2012.88](https://doi.org/10.1109/TKDE.2012.88)
- O. Gotoh, “An improved algorithm for matching biological sequences,”
  *Journal of Molecular Biology* 162(3), 1982.
  [doi:10.1016/0022-2836(82)90398-9](https://doi.org/10.1016/0022-2836(82)90398-9)
