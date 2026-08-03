---
title: k-bounded true Damerau-Levenshtein correctness and resource gate
date: 2026-08-01
project: liblevenshtein-rust
kind: feature
status: complete
verdict: correctness_resource_corpus_and_repository_gates_accepted
root_epic: extending-liblevenshtein-automaton-families-4bb97598
work_item: phase-6-k-bounded-true-damerau-levenshtein-db9ccdc9
---

# True Damerau–Levenshtein correctness and resource gate

## 1. Provenance and timing

The narrative implementation plan pre-registered exact automaton/reference
set equality at budgets `$`k\le3`$`, unchanged OSA behavior, a measured state-size
distribution, and an empirical Birkbeck OSA/true-Damerau divergence report on
2026-07-25. The example and generated correctness tests were executed before
this standalone ledger file was created, so this record does not mislabel those
runs as newly pre-registered. Their acceptance rules came from the earlier plan.

The Birkbeck and state-size measurements had **not** run when this section was
first written. Their populations, aggregation rules, and reporting fields were
frozen before execution. Sections 5 and 6 preserve the results and every
protocol correction discovered during the run.

## 2. Terms and hypotheses

**OSA** means optimal string alignment, the restricted adjacent-transposition
recurrence exposed as `Algorithm::Transposition`. **True Damerau–Levenshtein**
means the unrestricted edit-script distance in which a later edit may operate on
the output of an earlier edit. A **pending position** has stored one positive
query-endpoint delta while it consumes the intervening dictionary substring.

For query endpoint separation `$`\delta\ge1`$` and `$`b\ge0`$` intervening
dictionary units, the streaming macro charges

```math
C_{\mathrm{stream}}=\delta+b
=(\delta-1)+b+1=C_{\mathrm{LW}}.
```

The three right-hand terms are query deletions, dictionary insertions, and one
transposition in the Lowrance–Wagner recurrence.

| ID | Hypothesis or reporting obligation | Frozen decision rule |
|---|---|---|
| `damerau-001-reference` | The full last-occurrence DP implements unrestricted Damerau distance. | Published separator `$`d(\text{"CA"},\text{"ABC"})=2`$`, empty/Unicode examples, OSA dominance, and metric properties pass. |
| `damerau-002-automaton` | The streaming automaton is extensionally equal to the reference at the practical budgets. | Exact `(term, distance)` map equality for every generated dictionary/query at `$`k\in\{0,1,2,3\}`$`; 2,000 cases. |
| `damerau-003-osa-compatibility` | Adding the new selector does not change OSA. | The frozen OSA result sequence and distances remain exactly equal; the existing OSA differential suite remains green. |
| `damerau-004-laws` | Entry, extension, resolution, epsilon, budget, and pending-subsumption laws match the implementation. | Rocq, Verus, Z3, cvc5, and TLC pass without assumptions; corresponding generated properties pass. |
| `damerau-005-state-growth` | Reachable frontier size follows the predicted `$`\mathcal{O}(k^2)`$` envelope and the one-position successor buffer does not spill for `$`k\le3`$`. | Report every observed maximum for `$`k=1,2,3`$`, the ratio to `$`k^2`$`, and `SmallVec::spilled()`; do not discard a workload. |
| `damerau-006-corpus` | The feature has a measurable, directionally sound distinction from OSA on real spelling errors. | Evaluate every pair in every `*DAT.643` member of the checked-in Birkbeck archive. Report pair count, divergence count/rate, true-Damerau-better count, true-Damerau-worse count, and up to 12 first traversal-order separators. Require worse count zero; no minimum divergence rate. |

## 3. Frozen state-growth protocol

For each `$`k\in\{1,2,3\}`$`, use a repeated-unit query and dictionary prefix of
length `$`2k+4`$`. Repeated units expose every characteristic-vector delta and
allow pending states to resolve or extend at every column. Record the maximum
reachable `State::len()` over the complete prefix walk. Separately transition a
normal position with a length-`$`k+1`$` all-true characteristic vector and record
the successor count and whether its inline `SmallVec<[Position; 4]>` spills.

```text
PROFILE-STATE-GROWTH:
    for k in {1, 2, 3}:
        query := repeat('a', 2*k + 4)
        state := INITIAL-STATE(query, k, TrueDamerau)
        maximum := size(state)

        for dictionary unit in repeat('a', 2*k + 4):
            state := TRANSITION-STATE(state, unit, query, k, TrueDamerau)
            maximum := max(maximum, size(state))

        successors := TRANSITION-POSITION(normal(0,0), true^(k+1), k)
        report(k, maximum, maximum/(k*k), len(successors), spilled(successors))
```

## 4. Frozen corpus protocol

`MittonCorpus::load_birkbeck_zip` inspects every data member and reads every
record format carrying an explicit correction: fixed-column pairs, comma lists,
correction sections, marked-error lists, and pair-plus-context rows. It
lowercases both fields and assigns one observation to every explicit pair.
Raw prose and numeric answer sheets without a correction in the same member are
rejected rather than guessed. Documentation members are ignored. The loader
rejects more than 10,000 members, more than 64 MiB of declared uncompressed
data, or size overflow. For every recognized pair, compute both reference
distances without an automaton or threshold shortcut.

The corpus report is descriptive: a zero divergence rate would be a valid result
and would weaken the empirical motivation without falsifying correctness. A
case where true Damerau exceeds OSA would contradict the edit-repertoire
inclusion and fails the gate.

## 5. Results

| Evidence | Population | Result | Verdict |
|---|---:|---:|---|
| exact automaton/reference map | 2,000 generated dictionaries/queries at every `$`k\in\{0,1,2,3\}`$` | exact `(term, distance)` map equality | accepted |
| budget monotonicity and order determinism | 2,000 generated dictionaries/queries per property | every budget-`$`k`$` result survives at `$`k+1`$` with the same distance; repeated traversal order is identical | accepted |
| metric laws | 2,000 cases per identity, indiscernibility, symmetry, and triangle law | all pass | accepted |
| OSA compatibility | frozen result order plus 2,000 generated dictionaries/queries | exact frozen sequence unchanged; exact reference maps and repeated-call order agree | accepted |
| transition and subsumption invariant mirrors | 2,000 cases each for macro laws, suffix-continuation dominance, and continuation separation | entry/extend/resolve/no-epsilon arithmetic and every exercised residual-language inequality pass | accepted |
| Rocq / Verus / Z3 / cvc5 / TLC | registered obligations | Rocq passes; Verus 6 verified/0 errors; five SMT negations unsatisfiable in both solvers; TLC explores 13 states at depth 5 with all invariants | accepted |
| backend/unit/policy/result matrix | 8 no-default-feature tests; 11 all-feature tests | 12 concrete byte/char/u64 backend-unit combinations, four policies, and plain/distance/ordered/priority/value/zipper surfaces pass | accepted |
| state-size distribution | `$`k=1,2,3`$` | maxima 2, 4, 7; ratios to `$`k^2`$` 2.000000, 1.000000, 0.777778 | accepted |
| one-position successors | `$`k=1,2,3`$` | lengths 2, 3, 4; no `SmallVec` spill | accepted |
| repeated-prefix Criterion profile | `$`k=1,2,3`$`; Standard, OSA, and true Damerau | true-Damerau medians 586.15 ns, 1.7176 µs, and 3.4730 µs; 1.53, 3.32, and 5.26 times the Standard medians | accepted; measured cost follows the predicted budget-dependent state growth |
| Birkbeck divergence | 42,395 explicit pairs | 136 separators (0.320793%); true better 136, true worse 0 | accepted |
| Birkbeck DAT inventory and automaton/reference | every correction plus every recognized pair with reference distance at most 3 | exact inventory retained; 37,472 exact checks; zero false negatives | accepted |
| full unit/integration/example/doc suite | all features | 5,093 tests pass, including 3,829 library tests and 268 doctests | accepted |
| minimal-feature suite | no default features | 1,783 tests pass; every test target also compiles warning-free | accepted |
| release-quality static gates | all targets and features | formatting, strict Clippy, strict rustdoc, and documented serialization examples pass | accepted |
| documentation reproducibility | 276 living documents; 59 generated SVGs | mathlint reports zero legacy constructs; every generated SVG is in sync with its source | accepted |
| pgmcp bug gate | 148 changed files | no open anchored bugs | accepted |

The deterministic first twelve separators, ordered lexicographically by
`(misspelling, correction)`, are:

```text
acknolgeing -> acknowledging       OSA 4, true 3
acquatience -> acquaintance        OSA 4, true 3
acqunatince -> acquaintance        OSA 4, true 3
amidealty -> immediately           OSA 7, true 6
amnonnia -> pneumonia              OSA 6, true 5
anaslis -> analysis                OSA 3, true 2
anayalsis -> analysis              OSA 3, true 2
angcies -> agencies                OSA 3, true 2
anoisen -> annoyance               OSA 6, true 5
anwenser -> answer                 OSA 4, true 3
appiete -> appetite                OSA 3, true 2
approxiament -> approximate        OSA 4, true 3
```

## 6. Protocol corrections and discovered defects

The first corpus run is rejected. A uniform whitespace parser incorrectly
treated headers and prose as pairs and reported 51,236 observations. Inspection
of examples exposed the error before the result was accepted. The replacement
uses member-aware explicit formats and fixture tests. This changes the
operational definition from “every source row” to “every explicit correction
pair in every inspected data member”; raw members cannot support the original
claim without an external correction oracle.

The first full-automaton corpus run used `DoubleArrayTrie` and reported missing
corrections. Focused state traces and singleton regressions preserved the paths;
distance-zero queries then showed that the large trie build had itself lost
exact input terms. The root cause was the byte DAT builder's fixed 10,000-candidate
BASE search: on exhaustion it returned an unchecked slot and could overwrite an
occupied transition during relocation. The repair searches every representable
collision-free base, beginning at the locality hint and wrapping once, or reports
address-space exhaustion. A synthetic test occupies the former complete search
window. The accepted corpus gate is restored to `DoubleArrayTrie`, first checks
every correction in its exact inventory, then completes all 37,472 budget-eligible
automaton checks without a false negative. The defect and result remain tracked in
pgmcp as
`large-birkbeck-doublearraytrie-build-loses-exact-terms-during-relocation-fcefe1`.

The frozen record asked for the “first traversal-order” separators, but
`MittonCorpus` deliberately stores a `HashMap`, so that order is randomized.
The report now sorts all separators lexicographically before taking twelve.
This reproducibility correction does not affect the population or aggregate
decision rule.

The backend matrix exposed an older zipper finishing defect. The zipper passed
dictionary path depth to `State::infer_distance`, whose parameter is query
length. It therefore undercharged short terms and overcharged long terms; in
particular, its otherwise-correct pending macro reached `ABC` but was reported
outside budget. `AutomatonZipper` now finishes against its stored query length,
and direct state, automaton zipper, dictionary intersection, and iterator tests
all report the same distance.

The first suffix-backend assertion also used the wrong oracle. A suffix
dictionary intentionally enables prefix/subsequence completion and does not
charge an unmatched query suffix after a represented substring has matched.
Consequently, whole-string `$`d(\text{"CA"},\text{"ABC"})=2`$` is not the
suffix surface's contract; a cheaper cost-1 partial path masks it. The accepted
matrix tests that documented behavior on both byte and character suffix
automata, while all whole-term prefix dictionaries remain exactly equal to the
Lowrance–Wagner oracle.

## 7. References

- Lowrance, R., and Wagner, R. A. “An extension of the string-to-string
  correction problem.” *Journal of the ACM* 22(2), 177–183 (1975).
  [DOI 10.1145/321879.321880](https://doi.org/10.1145/321879.321880).
- Boytsov, L. “Indexing methods for approximate dictionary searching:
  Comparative analysis.” *Journal of Experimental Algorithmics* 16 (2011).
  [DOI 10.1145/1963190.1963191](https://doi.org/10.1145/1963190.1963191).
