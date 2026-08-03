# Exact multi-kind Dyck correction and its projection bound

**Status:** implemented · **Cost model:** unit insertion, deletion, and
substitution · **Alphabet:** token IDs, not text

A **Dyck word** is a properly nested delimiter word. For `$`k`$` delimiter
kinds, opening kind `$`r`$` is token `$`r`$` and its only valid closer is token
`$`k+r`$`, where `$`0\le r<k`$`. Keeping the alphabet numeric avoids a text
conversion boundary and makes the API suitable for parsers and token streams.

`DyckCorrector::correct` returns:

- the exact unit-cost Levenshtein distance to `$`D_k`$`;
- one deterministic minimum-cost balanced token sequence; and
- a replayable edit witness containing keeps, deletions, substitutions, and
  insertions at original-input boundaries.

The linear-time `balance_lower_bound` remains useful as an admissible
heuristic. It is deliberately separate because erasing kinds can hide a
cross-kind mismatch.

## 1. Exact interval recurrence

Let `$`C(i,j)`$` be the minimum cost to correct the half-open input interval
`$`w[i..j)`$` to `$`D_k`$`. The base case is `$`C(i,i)=0`$`. For a non-empty
interval, the implementation enumerates the four grammar-complete edit forms
below and takes their minimum.

For replacement cost `$`\rho(a,b)=0`$` when `$`a=b`$` and `$`1`$` otherwise:

```math
\begin{aligned}
C(i,j)=\min\{&
\rho(w_i,r)+C(i+1,p)+\rho(w_p,k+r)+C(p+1,j),\\
&1+C(i,p)+\rho(w_p,k+r)+C(p+1,j),\\
&\rho(w_i,r)+1+C(i+1,j),\\
&1+C(i+1,j)
\},
\end{aligned}
```

where the first row ranges over `$`0\le r<k`$` and `$`i<p<j`$`, the second
over `$`0\le r<k`$` and `$`i\le p<j`$`, and the third over
`$`0\le r<k`$`. The rows mean:

1. consume the first token as an opener and token `$`p`$` as its closer;
2. insert a missing opener and consume token `$`p`$` as its closer;
3. consume the first token as an opener and insert its closer; and
4. delete the first token.

Every non-empty Dyck word has the first-pair decomposition
`$`r\;u\;(k+r)\;v`$` with `$`u,v\in D_k`$`. Therefore the first two forms
cover targets whose first pair consumes an existing closer, the third covers
an inserted closer, and deletion permits the alignment to begin after an
unusable input token. This is the optimal-substructure basis of the interval
program.

The correctness specification is deliberately not this recurrence.  In Rocq,
`levenshtein_alignment` independently defines the ordinary left-to-right edit
relation with insertion, deletion, keep, and substitution columns.  The proof
normalizes every such alignment whose target is in `$`D_k`$` into one of the
four reconstruction trees at no greater cost.  The converse maps every
reconstruction tree back to an ordinary alignment at exactly the same cost.
Consequently, the table minimum is extensionally equal to the standard
Levenshtein minimum over all typed Dyck targets; it is not merely optimal among
algorithm-shaped witnesses.

The fill-order premise is itself proved. `all_splits` is an executable Gallina
enumeration of every possible consumed-closer position, while
`recurrence_descriptors` enumerates the four branch families in Rust tie order.
For each non-empty interval, every descriptor has exactly one cost once its
strict subintervals have exact minima. A constructive finite-list lemma selects
the least descriptor using decidable natural-number comparison. Strong
induction on source length then establishes the strict-subinterval invariant
for every cell and yields the unconditional theorem
`interval_recurrence_is_unconditionally_exact_standard_dyck_distance`. No
classical choice principle, axiom, admitted goal, or caller-supplied optimality
premise remains.

### Literate pseudocode

```text
CORRECT-DYCK(input, kinds)
  validate the numeric delimiter alphabet and the cubic-work policy
  set C(i, i) to zero for every input boundary i

  for interval length from 1 through input length
    for every interval [i, j) of that length
      enumerate every typed endpoint pair (r, p)
      enumerate every inserted-opener pair (r, p)
      enumerate every inserted closer r
      enumerate deletion of input[i]
      store the least cost and its deterministic backpointer

  replay backpointers from [0, input length)
  return the exact cost, corrected token sequence, and edit witness
```

The time bound is `$`\mathcal{O}(kn^3)`$`; the cost and backpointer tables use
`$`\mathcal{O}(n^2)`$` memory. `DyckCorrector::with_max_work` applies a
saturating `$`k(n+1)^3`$` guard before table allocation. Invalid token IDs,
zero bracket kinds, alphabet overflow, table-size overflow, and work-policy
violations are typed errors.

## 2. Witness invariants

The reconstruction layer maintains four executable invariants:

```math
\begin{aligned}
\operatorname{replay}(w,E)&=v,\\
\sum_{e\in E}\operatorname{cost}(e)&=C(0,n),\\
v&\in D_k,\\
C(0,n)=0&\Rightarrow v=w\land w\in D_k.
\end{aligned}
```

`DyckCorrection::replay` checks original indices and source tokens, consumes
the entire input exactly once, reproduces the stored output, and recomputes
the edit cost. Cross-kind repair is observable: with two kinds, `[0, 3]`
cannot be treated as a pair; one exact answer is `[0, 2]` at cost one.

## 3. Multi-kind pushdown reference

The lling-llang bridge supplies
`PdaBuilder::balanced_bracket_kinds`. Each delimiter kind owns a distinct
stack marker, so a closer can pop only its matching opener. The builder rejects
empty or ambiguous alphabets instead of assigning one token multiple roles.
`exact_dyck_correction` exposes the root exact corrector to lling-llang
pipelines.

The construction follows the useful part of MeTTaIL's VPA design in
`../mettail-rust/prattail/src/vpa.rs`: pair identity belongs in the stack
alphabet. Its untyped `build_skip_table`, which only receives
`Call`/`Return`/`Internal`, is not a correctness reference for multi-kind
matching because that classification cannot distinguish `(]` from `()`.

No finite VPA state count bounds the nesting depth of every accepted word. A
one-state balanced-delimiter VPA can accept arbitrarily deep nesting by using
its stack. Consequently, the exact corrector has a computational work limit,
not a semantic nesting limit.

## 4. Admissible kind-erasure bound

Define projection `$`\pi`$` by mapping every opener to a single opening symbol
and every closer to a single closing symbol. It preserves length and maps
`$`D_k`$` onto `$`D_1`$`. For unit-cost Levenshtein distance to a language:

```math
d(\pi(w),D_1)\le d(w,D_k).
```

To see why, map any edit script from `$`w`$` to `$`v\in D_k`$` through
`$`\pi`$`. Insertions and deletions retain their cost, matches remain matches,
and a substitution may become a match but never becomes more expensive. The
projected script ends in `$`\pi(v)\in D_1`$`. Minimizing both sides proves the
inequality.

A scan of the projected word leaves `$`o`$` unmatched openings and `$`c`$`
unmatched closings. Its exact one-kind correction distance is:

```math
\left\lceil\frac{o}{2}\right\rceil+
\left\lceil\frac{c}{2}\right\rceil.
```

Thus `balance_lower_bound` runs in `$`\mathcal{O}(n)`$` time and constant
additional memory. It may be zero for a cross-kind mismatch, so it is a
heuristic, never the exact multi-kind answer.

## 5. Bounded regular approximation

`balanced_depth_dfa(k,D)` recognizes exactly the words in `$`D_k`$` whose
nesting depth is at most `$`D`$`. Its state is the complete stack word, giving:

```math
N(k,D)=\sum_{d=0}^{D}k^d.
```

Construction rejects `$`N(k,D)>4096`$` before allocating the transition table.
`SmallDfaStateSet` is a dynamically sized bit vector, so the public 4,096-state
policy is now the actual representation limit rather than a 32-bit-carrier
accident.

## 6. Verification and executable correspondence

- Rocq proves typed-Dyck constructor soundness, zero-cost balanced identity,
  exact-minimum existence for every source, finite branch-enumeration
  completeness, unconditional interval-fill optimality, the
  `$`|target|\le 2|source|`$` oracle cutoff, first-pair decomposition,
  cross-kind separation, and bidirectional refinement to an independently
  defined standard Levenshtein relation without axioms or admitted goals.
- Dafny and Verus check the replacement, typed-pair, recurrence-minimum, and
  zero-cost endpoint obligations.
- Z3 and cvc5 independently reject counterexamples to those arithmetic
  obligations.
- Rust property tests compare every source subinterval with brute-force
  standard Levenshtein distance over every bounded Dyck word for 2,000
  generated inputs, replay every root witness, and check that the projection
  lower bound is admissible.
- lling-llang property tests compare the distinct-marker PDA with a reference
  stack recognizer for generated multi-kind token streams.

The proof sources are indexed in
[`FORMAL_VERIFICATION_MANIFEST.tsv`](../../verification/FORMAL_VERIFICATION_MANIFEST.tsv).
