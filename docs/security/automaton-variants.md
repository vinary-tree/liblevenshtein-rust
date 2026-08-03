# Automaton-variant security

## Threat model

The variant seam processes attacker-controlled dictionary units, query units,
query lengths, edit budgets, and dictionary topology. An attacker may attempt
to cause an incorrect match, suppress a valid match through unsound pruning,
trigger integer overflow, create a state explosion, or exploit a representation
collision between unfinished operations.

The seam does not parse bytecode, load plugins, invoke the network, or execute
downstream callbacks. `AutomatonVariant` is crate-private, so untrusted code
cannot install an arbitrary pruning policy.

## Security invariants

| Risk | Control | Executable evidence |
|---|---|---|
| Continuation collision | State order includes `kind` and `aux`; private fields prevent partial mutation. | layout/order unit tests, 2,000-case order properties, Rocq full-key injectivity, SMT counterexample checks |
| Cross-language false-negative pruning | OSA and merge/split variants reject incompatible continuation kinds. | reference subsumption proptest, Rocq/Verus/SMT separation obligations, existing differential suites |
| Runtime selector drift | `with_variant!` is the single closed mapping from `VariantSpec` to concrete types. | Rocq extensional equality and TLA+ legacy/static trace equivalence |
| Offset or cost overflow | Successor arithmetic uses `checked_add`; overflow yields no successor. | existing transition overflow unit/property tests |
| Future enum confusion | `Algorithm` is `#[non_exhaustive]`; downstream matches must include a fallback. | compile-time language rule plus API tests |
| Resource amplification | Positions remain 24 bytes on 64-bit targets and `SmallVec` keeps four successors inline. | compile-time size assertion and six-suite Criterion gate; the stricter exact-byte witness reports a documented negative result rather than being treated as proof |

## Payload validation

Legacy and normal constructors emit `aux == 0`. `DamerauPending` requires a
positive delta no larger than 255; its crate-private constructor debug-checks
positivity, and public transitions reject a budget above the representable
ceiling before traversal. A malformed payload is never normalized during
comparison: normalization could make two distinct traces share a key and
reintroduce false-negative pruning.

History support is also an engine-level capability. The unit-cost dictionary
transducer implements true Damerau. Weighted positions and phonetic language
products do not. Those boundaries reject `Algorithm::DamerauLevenshtein`
instead of projecting it silently to Standard or OSA behavior.

## Review checklist

Before adding a variant:

1. Document every legal `(kind, aux)` pair.
2. Add a constructor that creates only legal pairs; do not expose mutable
   fields.
3. Prove that `subsumes(lhs, rhs)` preserves the minimum cost for every suffix.
4. Differential-test the complete emitted result set, not only the best match.
5. Generate adversarial empty, maximal-budget, maximal-index, Unicode, and
   collision cases.
6. Confirm that `skip_window` cannot under-approximate a legal transition.
7. Run the full formal manifest and the resource/performance gates.

An exact-byte code-generation comparison is diagnostic, not a substitute for
the resource gate. Phase 5's probe changed because it captured intentional
payload initialization and an ownership-boundary change in addition to selector
code. That stronger hypothesis is recorded as rejected in the
[scientific ledger](../scientific-ledger/position-kind-zero-cost-2026-08-01.md);
the script remains strict so a future change cannot silently turn a mismatch
into a pass. Its separate optimized-LLVM-IR audit isolates selector erasure by
rejecting any runtime selector, non-Standard leaf provenance, or surviving
`switch` in the constant-Standard probe.

## Operational guidance

Treat a failed formal counterexample check, differential test, or property seed
as a correctness incident. Do not widen a pruning rule to recover benchmark
performance. Disable the unsound rule, retain the minimized regression seed,
and record a root-attributed issue in the project tracker.
