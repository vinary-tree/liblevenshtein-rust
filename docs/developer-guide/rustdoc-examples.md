# Executable Rust API examples

Rust examples are part of the public API contract. A reader should be able to
copy an intended-usage example from Rustdoc and compile it against the release
that published the page. CI therefore treats ordinary `rust` fences as tests,
builds them with every feature enabled, and denies Rustdoc warnings.

## Fence policy

Use the narrowest truthful fence:

- `rust` for intended usage. Cargo compiles and runs the example.
- `rust,no_run` only when compilation is the contract but execution requires an
  unavailable service, credential, device, or deliberately expensive input.
- `compile_fail` when compiler rejection is the behavior being taught.
- `text` for pseudocode or intentionally incomplete fragments.
- `rust,ignore` only for pre-existing repair debt recorded by the ratchet. Do
  not use it for new examples or as a substitute for the preceding categories.

An example that needs a hidden import, setup statement, or `Result` wrapper
should use Rustdoc's hidden `#` lines. That keeps the displayed example concise
without exempting it from compilation or execution.

## Required checks

Run the same checks used by CI:

```bash
python3 scripts/check-rust-doc-examples.py
RUSTDOCFLAGS="-D warnings" cargo test --locked --all-features --doc
RUSTDOCFLAGS="-D warnings" cargo doc --locked --all-features --no-deps
```

The first command enforces two monotone constraints:

1. The global ignored-example count cannot increase. When an ignored example
   is repaired, the baseline in `scripts/check-rust-doc-examples.py` must be
   lowered in the same change.
2. The cache API has no ignored-example allowance. Its 46 examples were all
   compiled and executed during the 2026-08-28 controlled audit, so both zero
   ignored fences and at least 46 executable fences are enforced.

When developing beside unpublished family crates, check out the exact source
refs declared by `release/version.json`. A convenient sibling checkout with a
different release candidate is not equivalent evidence. Registry releases use
the same immutable source-ref graph through the checkout actions in CI.

## Repair workflow

For each ignored example:

1. Remove `ignore` in an isolated checkout and run the all-feature doctest.
2. If it passes unchanged, keep it executable and lower the ratchet.
3. If it fails, determine whether it is intended usage, non-running compilable
   usage, an expected compiler error, or pseudocode. Repair the code and fence
   according to that classification.
4. Run the full Rustdoc and doctest gates; a targeted success alone can miss
   feature interactions or duplicate module-level examples.

Do not rewrite an example merely to make the test green. Its imports, result
handling, ownership model, and asserted result must describe the supported
public API and the behavior a customer should rely on.

## Current debt

The controlled 2026-08-28 audit started with 348 ignored examples. Running all
of them as ordinary Rust found 165 that already compiled and executed without
changes; those suppressions were removed. The first repair batch then restored
the crate quick start and three file-backed corpus examples, using `no_run` for
the examples whose input files are deliberately external. A second batch
restored the deprecated-import migration, synchronization, serialization, and
WallBreaker examples with current imports, signatures, result types, and honest
`no_run` treatment for file I/O. The fluent transducer-builder batch restored
ten more standalone examples with complete construction, meaningful assertions,
and current ordered-candidate handling. The query-policy batch then corrected
restricted substitutions, ordered filtering and prefix matching, and both
post-filtered and set-filtered value queries; two deliberately incomplete
internal-control-flow fragments were honestly relabeled as `text`. The remaining
phonetic-core batch restored the low-level rewrite, ergonomic custom-rule, and
English syllable examples with current inputs, fuel, imports, and documented
heuristic limits. The embedded-language batch then executed every previously
ignored language example and replaced stale intermediate-marker assertions with
the public `RuleSetChar::apply` method's empirically verified fixed-point output.
The rule-language batch restored 17 `.llev` and LibLevenshtein Regex Expression
(LLRE) examples, distinguishing executable parsing, matching, and in-memory
serialization from compile-checked `no_run` examples that require customer-owned
files. The universal-automata batch then repaired 24 public examples across
characteristic vectors, word-pair encoding, positions, states, diagonal
crossing, subsumption, and complete-word acceptance. It also corrected the
published characteristic vectors for `banana`, replaced invalid position
fixtures, re-exported variant-state types from the natural module root, and
classified one private-helper fragment as `text`. That review exposed a real
functional limitation: `UniversalAutomaton::with_policy` currently discards its
policy value, so its documentation now states the limitation and the repair is
tracked separately instead of claiming unsupported semantics. The generalized
alignment batch restored eight examples for runtime operation sets, exact
weighted acceptance, positions, and subsumption. Its end-to-end phonetic
example exposed and fixed an invalid built-in preset: double-consonant
simplification and expansion had been mixed into one arity declaration and are
now represented by separate `2 → 1` and `1 → 2` operations. The preset and its
composition with standard operations are both validation-checked and executed.
The zipper-intersection batch then made all six accessors for distance, depth,
term reconstruction, viability, and the underlying dictionary and automaton
zippers executable. These examples now construct a real snapshot-backed
PathMap zipper and traverse a real dictionary/automaton product instead of
referring to an undefined placeholder. The remaining 64 stay explicitly
ratcheted while they are repaired subsystem by subsystem.
The method, controls, and raw result summary are recorded in the
[scientific ledger](../scientific-ledger/rustdoc-example-audit-2026-08-28.md).
