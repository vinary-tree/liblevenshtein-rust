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
