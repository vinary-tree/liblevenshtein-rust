# Rustdoc executable-example audit — 2026-08-28

## Question

How many ignored Rustdoc examples represent unavoidable non-executable prose,
and how many are stale suppressions hiding examples that already satisfy the
current public API?

## Method

The audit used the immutable `4.0.0-rc.5` corrective source graph. The
liblevenshtein checkout was paired with the exact libdictenstein,
vinary-tree-interop, and llattice source refs used by that release rather than
the potentially different branches in ordinary sibling directories. All
commands ran from repository-local storage under `target/`; no RAM-backed
temporary filesystem was used.

The control run compiled documentation with every feature and warnings denied,
then ran the existing doctest classification:

```bash
RUSTDOCFLAGS="-D warnings" cargo +1.95 doc --offline --all-features --no-deps
RUSTDOCFLAGS="-D warnings" cargo +1.95 test --offline --all-features --doc
```

For the experiment, a disposable copy changed every `rust,ignore` and `ignore`
opening fence to an ordinary `rust` fence. No example body or library source
was changed. The same all-feature doctest command then measured which examples
passed unchanged. A second run applied only the proven passing conversions to
the reviewed source.

## Results

| Stage | Passed | Failed | Ignored | Total |
|---|---:|---:|---:|---:|
| Control classification | 265 | 0 | 348 | 613 |
| Every ignored fence activated | 430 | 183 | 0 | 613 |
| Proven passing fences activated | 430 | 0 | 183 | 613 |

The experiment established that 165 of 348 suppressions, or 47.4%, were stale:
the examples compiled and executed unchanged. All 46 examples under
`src/cache/` were among the passing set. The cache examples now have a
zero-ignore policy and a minimum executable-example count.

## Interpretation

The ignored count was not a reliable proxy for examples that inherently could
not run. It mixed at least two populations:

- valid, current intended-usage examples carrying obsolete suppressions; and
- genuine documentation defects, including stale imports, missing setup,
  incomplete fragments, incorrect assertions, and examples written against
  older API shapes.

Activating only the first population improves customer-facing evidence without
changing runtime behavior. The 183 failures are retained as visible,
monotonically bounded debt rather than being relabeled wholesale. Each must be
classified before repair so pseudocode, expected compiler errors, and runnable
usage are represented honestly.

## Controls and limitations

- Every experiment used all crate features and the same Rust toolchain and
  source graph; this controls feature-gated imports and sibling API drift.
- The full activation changed fence metadata only, preserving example bodies.
- A passing doctest establishes compilation and observed execution for that
  configuration. It does not by itself prove that the example is the clearest
  or most idiomatic presentation; prose and API-quality review remain required.
- The global ignored-count ratchet prevents debt growth but does not replace
  review of the semantic quality of individual examples.

## Decision

Accept the 165 evidence-backed fence conversions. Add an automated ratchet,
run all-feature doctests in the documentation lane, and repair the remaining
183 examples in bounded subsystem batches. New public examples must be
executable by default; `ignore` is not an accepted escape hatch.

## Repair follow-through

The same immutable source graph and all-feature command were retained for each
bounded repair batch. Unlike the initial fence-only experiment, these batches
reviewed the example bodies and public semantics before changing their
classification.

| Reviewed state | Passed | Failed | Ignored | Total |
|---|---:|---:|---:|---:|
| After crate, corpus, migration, synchronization, serialization, WallBreaker, builder, and query-policy repairs | 460 | 0 | 152 | 612 |
| After phonetic-core and embedded-language repairs | 492 | 0 | 120 | 612 |
| After `.llev` and LLRE repairs | 509 | 0 | 103 | 612 |
| After universal-automata repairs | 533 | 0 | 78 | 611 |
| After generalized-alignment repairs | 541 | 0 | 70 | 611 |
| After zipper-intersection repairs | 547 | 0 | 64 | 611 |
| After helper-and-pool repairs | 551 | 0 | 60 | 611 |
| After phonetic-entry repairs | 557 | 0 | 54 | 611 |
| After NFA-foundation repairs | 563 | 0 | 48 | 611 |
| After NFA-runtime repairs | 572 | 0 | 39 | 611 |

The total can decrease when review proves that a fence is pseudocode or a
private implementation fragment rather than customer-compilable usage. Such a
fence is relabeled `text`; it is not counted as a doctest and does not consume
the ignored-example allowance.

The universal-automata batch also served as a semantic control. It found three
classes of stale documentation: missing standalone imports, examples whose
positions violated the documented invariants, and characteristic-vector bits
whose asserted indices disagreed with exact character equality. All were
corrected and executed. One advertised behavior was not repaired as prose:
`UniversalAutomaton::with_policy` accepts and discards its policy value, while
the encoder constructs vectors using exact equality. The API documentation now
states that limitation, and pgmcp item
`wire-universalautomaton-substitution-policies-into-characteristic-vector-matching-ba95ee`
tracks the required policy-aware implementation and semantic tests.

The generalized-alignment batch demonstrated why examples execute rather than
merely compile. The comprehensive English phonetic preset originally placed
both double-to-single and single-to-double restriction pairs in one operation
declared as consuming two source scalars and one target scalar. Operation-set
validation therefore rejected every reverse pair, and the Boolean acceptance
API failed closed. The preset now uses distinct `2 → 1` simplification and
`1 → 2` expansion operations, validates as a complete set, and is exercised
through a full `phone` to `fone` alignment after composition with the standard
edit operations.

The zipper-intersection batch replaced six contextless fragments with complete
product-traversal experiments. Each example constructs an owned,
snapshot-backed PathMap zipper, pairs it with an automaton zipper, and observes
the public result after actual traversal. This verifies the boundary accessors
for distance, depth, reconstructed terms, viability, dictionary finality, and
query metadata rather than proving only that isolated method names compile.

The helper-and-pool batch found two kinds of semantic drift. The hierarchical
filter examples queried `var` at edit distance one but expected matches such as
`global_var`; whole-term Levenshtein matching cannot produce those results.
They now use four equally eligible near spellings and demonstrate that the
scope predicate retains the three visible values while excluding the fourth.
The state-pool examples and prose also predated the current position storage:
they now exercise the actual insertion signatures and distinguish owned
integer-state storage from the weighted state's inline-and-spilled `SmallVec`
representation.

The phonetic-entry batch corrected six independently stale examples. The
transducer overview had paired a byte dictionary/transducer with a
character-NFA compiler; the executable version now uses the character surface
consistently and proves both phonetic alternatives while excluding an unrelated
term. Named-class extraction now checks concrete trigraph and tetragraph
members. Its sequence example no longer names an unregistered
`complex_clusters` class and instead demonstrates the documented distinction
between an existing class with no open-ended `Sequence` variants and a missing
class. Finally, the verified-rule examples compile a real selected rule set,
and the explicit rewrite-rule literal includes the current syllable condition.

The NFA-foundation batch made six core examples executable. State display and
finality, positive and negated Unicode character classes, consuming versus
epsilon transitions, and Thompson alternation/star/concatenation now have
observable acceptance or rejection assertions. The compiler overview parses
and compiles both a regular expression and a rewrite rule through their actual
public entry points. The flag example no longer uses the invalid
`(?iu:café)` spelling: it demonstrates case-insensitive NFA expansion and the
separate `(?u:NFC:...)` runtime-normalization result explicitly.

The NFA-runtime batch repaired nine examples that previously could not serve as
customer evidence. Direct construction now uses the initial state created by
`NFAChar::new` and finalizes its CSR transitions before matching. Both optimizer
surfaces compile an intentionally unoptimized regular expression, compare the
language before and after transformation, and inspect real statistics instead
of calling nonexistent helpers. Context, incremental, lazy-DFA, and memoized
examples now build complete public automata and assert their boundary, reset,
transition-cache, and result-cache behavior. The memoization prose also matches
the implementation: a matcher owns a product automaton with one fixed distance
bound, so character-level cache keys are query strings rather than redundant
`(query, distance)` pairs.
