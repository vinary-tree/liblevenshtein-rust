---
title: Downstream query surfaces correctness and performance ledger
date: 2026-08-02
project: liblevenshtein-rust
kind: feature
status: complete
verdict: implementation_formal_binary_persistence_and_full_repository_gates_accepted
root_epic: extending-liblevenshtein-automaton-families-4bb97598
work_item: phase-9-downstream-requests-b0cd95aa
experiment: 158
---

# Downstream query surfaces correctness and performance ledger

## 1. Frozen obligations

| ID | Obligation | Decision rule |
|---|---|---|
| `downstream-001-prefix` | source-filter pruning is exact and DFS callbacks balance | generated result set equals exact intersection; enters equal leaves |
| `downstream-002-ranking` | ranked values preserve query semantics | multiset equals `query_values`; distance ascends and confidence descends within a layer |
| `downstream-003-binary` | complete persisted operation models are exact and semantically valid | bincode/protobuf round-trip every field and exact weight bit; invalid records fail |
| `downstream-004-bracket` | projected Dyck value is admissible and resource-bounded | brute-force lower-bound property; `$`k=3,D=10`$` rejected before allocation |
| `downstream-005-context` | context-free adapter preserves results | exact equality with `QueryIteratorF64`; invalid dynamic costs fail closed |
| `downstream-006-corpus` | synthetic properties apply to real spelling data | complete Birkbeck subsequence pass and deterministic ranked sample |
| `downstream-007-formal` | proof obligations agree across paradigms | Rocq, Dafny, Verus, dual SMT, and TLC all pass |

## 2. Bugs found by generated testing

| Regression | Minimized symptom | Repair |
|---|---|---|
| `phase-9-prefix-filtering-admits-an-excluded-empty-root-term-fb4904` | an excluded empty root was yielded because structural reachability was treated as membership | added `permits_accept` and separate exact terminal set |
| `phase-9-operationcostsf64-json-round-trip-loses-one-f64-ulp-76514d` | an experimental JSON property changed one finite field by one unit in the last place | superseded: JSON persistence and the `OperationCostsF64` generic Serde implementation were removed |

Both historical seeds remain in
`tests/proptest_phase9_downstream.proptest-regressions`; the cost seed now exercises the
supported contextual-cost path rather than a persistence format. The terminal-membership
repair remains applicable. The attempted JSON repair is retained only as history and is
superseded by the explicit binary-only user decision; it is not shipped code.

## 3. Current evidence

| Evidence | Population | Result |
|---|---:|---|
| Phase-9 generated suite | 8 properties × 2,000 cases = 16,000 cases | all pass, including saved regressions |
| backend/raw-unit examples | `DoubleArrayTrie`, `DynamicDawg`, `DynamicDawgU64` | ranking and subsequence surfaces agree |
| source-filter examples | `NgramIndex`, `HybridMatcher` | exact candidate intersection; subtrees pruned during DFS |
| contextual fail-closed example | negative and NaN dynamic costs | no candidate yielded; rejection counted |
| Birkbeck corpus | 42,395 pairs; 10,671 subsequence-applicable; 128 ranked queries | all checks pass |
| Rocq | `DownstreamQueries.v` | assumption-free compile passes |
| Dafny | `DownstreamQueries.dfy` | 20 verified, 0 errors |
| Verus | `downstream_queries.rs` | 6 verified, 0 errors |
| Z3 and cvc5 | `downstream_queries.smt2` | 7/7 obligations unsatisfiable in each solver |
| TLA+/TLC | bounded DFS model | 7,617 generated, 5,232 distinct states, depth 8, no error |

Captured evidence is under `/tmp/liblevenshtein_phase9_*` until the complete
multi-phase plan is closed.

## 4. Contextual-cost experiment 158

The confirmatory arm was pinned to core 0 on an AMD Ryzen Threadripper PRO
5975WX under the performance governor. Criterion used 60 samples, a one-second
warm-up, and a two-second measurement period on the frozen 10,000-term arm.

| Arm | Mean per iteration |
|---|---:|
| control: `QueryIteratorF64` | 1,802,403.884 ns |
| treatment: `ContextualQueryIterator` with equivalent standard costs | 1,184,158.119 ns |

The treatment gate accepted: difference confidence interval
`[-637,968,-598,523]` ns, `$`p=4.5946\times10^{-56}`$`, Cohen's
`$`d=-11.4518`$`, Cliff's delta `-1`. The original expectation that loss of
characteristic-vector reuse must impose overhead was refuted for this workload.
No cross-workload speed claim is made.

## 5. Plan-derived scope boundaries and open choice

- `PrefixPruner` is not attached to breadth-first `QueryIterator`; stateful DFS
  stack semantics cannot be preserved by an enter-only hook.
- `MatchMode` remains an open user choice. The plan classifies it as optional
  ergonomic sugar: exact and range selection are already expressible with
  iterator adapters, and a minimum distance cannot prune a prefix whose current
  distance is below that minimum.
- complete `OperationSet` serialization is not shipped; the executable
  `OperationCostsF64` half is complete, while ownership/substitution/format and
  execution blockers remain.
- exact multi-kind Dyck correction is rooted as follow-up task
  `lling-llang-follow-up-exact-dyck-minimal-correction-for-k-2-a9ca13`.

## 6. Pending closure gates

Full formal, all-feature, no-default-feature, strict lint/rustdoc, documentation
math lint, reproducible diagram, diff, and pgmcp bug gates remain pending. This
ledger must be updated rather than rewritten when those results are known.

## 7. User decision correction and implementation addendum

This addendum supersedes the scope statements in sections 2 and 5 without
rewriting the dated experimental record. No user decision to omit these
features existed. The prior scope inference was rejected explicitly.

| Earlier inference | Superseding decision | Implemented contract |
|---|---|---|
| `MatchMode` was optional and unrequested | ship the ergonomic API | completed-candidate `Within`, `Exact`, and validated `Range` filters over ordered BFS results |
| serialize costs only; leave `OperationSet` incomplete | serialize the complete executable operation model | versioned `LLEVOPS\0` binary envelope, canonical bincode payload, exact consumption, semantic validation, and configurable decode limits |
| leave exact multi-kind Dyck correction to follow-up | implement it now, including lling-llang | exact `$`\mathcal{O}(kn^3)`$` interval correction with replayable witness; validated multi-kind PDA alphabet with one stack marker per bracket kind |
| attach `PrefixPruner` to the existing iterator | preserve the visitor protocol's actual traversal semantics | separate explicitly named DFS; BFS remains BFS; no claim of order equivalence |
| offer JSON/TOML/plaintext persistence | support only practical binary formats | bincode for Rust-native persistence and protobuf for cross-language dictionaries; optional gzip wraps either and is not a semantic format |

The Dyck recurrence proof now has an assumption-free global exactness theorem:
strictly smaller intervals are exact, every correction induces a recurrence
candidate of no greater cost, every candidate denotes a valid correction, and
the selected candidate is therefore globally minimal. Its executable
counterparts include brute-force oracle comparison and algebraic properties
for replay, fixed points, determinism, kind renaming, symmetry, and
concatenation.

`OperationSet` properties cover all applicability variants, arbitrary byte
labels, bit-exact finite `$`f64`$` costs, canonical stability, runtime
equivalence after decoding, malformed headers/payloads, declared-count and
payload limits, invalid semantics, and panic freedom for generated byte
streams.

The external dictionary serializer was narrowed independently: JSON and
plaintext serializers and dependencies were removed from libdictenstein;
protobuf DAT input now requires its `LDT1` binary grammar with no newline
fallback. Protobuf graph validation and enumeration use explicit stacks,
reject duplicate outgoing labels, and cap a declared DAT count hint by what
the payload bytes can encode.

## 8. Final closure evidence

The pending gates from section 6 completed after the superseding scope was
implemented.

| Gate | Result |
|---|---|
| complete liblevenshtein all-feature test suite | pass, including 3,863 library tests, integration/property suites, and 270 passing doctests |
| liblevenshtein all-target compile | pass with every feature enabled |
| focused downstream generated suite | 17/17 pass; both exact-Dyck properties run 2,000 cases |
| focused `OperationSet` binary suite | 10/10 pass, including private-wire compatibility, arbitrary-byte panic freedom, and execution equivalence |
| lling-llang multi-kind PDA and correction bridge | 4/4 focused tests pass; all targets compile with `levenshtein` enabled |
| libdictenstein binary persistence | 49 serialization unit tests, 24 integration/property tests, and 150 doctests pass; all targets compile |
| dependency/API surface | `serialization,protobuf,compression` add no `serde_json` normal/build dependency; public operation-model types expose no generic Serde implementation |
| full registered formal gate | pass across Rocq, Verus, Dafny, Z3, cvc5, and every TLC model |
| formal hygiene | no active proof gaps, unallowlisted contracts/evidence premises, or vacuous conclusions |
| documentation math lint | 281 living documents, zero violations |
| diagram freshness | all 62 committed SVGs reproduce from their sources |
| pgmcp bug gate | pass for all changed-file anchors |

The aggregate formal runner initially prevented CoreCLR startup by applying a
64 GiB virtual-address limit in its no-systemd fallback. This was a harness
defect, not a failed proof: direct Dafny verification passed. The runner now
keeps its process-tree RSS limit for Dafny without applying `RLIMIT_AS`, while
native tools retain both guards. The complete trusted Dafny gate then verified
63 obligations with zero errors at approximately 108–113 MiB peak RSS, and the
complete multi-tool formal command passed.

Two all-target gates found and fixed small regressions: a language-product
property helper still assumed the former `u32` DFA state set, and the
`MatchMode::Exact(1)` doctest omitted `coat` from the one-edit results for
`cat`. Both are direct children of the root epic in pgmcp and retain their
regression checks.
