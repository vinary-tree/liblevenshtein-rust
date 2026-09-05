# Julia family binding qualification

This document is the review ledger for Julia support across the Vinary Tree
family. It distinguishes a useful facade from complete native-feature parity,
records exact source and continuous-integration (CI) evidence, and prevents an
unimplemented capability from being mislabeled as architecturally
inapplicable. The generated
[`family-completeness-matrix.tsv`](../../bindings/conformance/family-completeness-matrix.tsv)
remains the machine-readable authority; this document explains the judgments
behind its Julia rows.

![The family gate expands every project, capability, and language tuple before it accepts implementation, conformance, benchmark, documentation, and fresh-consumer evidence.](../diagrams/bindings/family-completeness-gate.svg)

## Qualification method

A **surface** is the target-language API through which a customer invokes a
native capability. A **fresh consumer** is a newly created Julia environment
that installs the package graph rather than importing files by relative source
path. A **complete cell** has five independent forms of evidence:

1. the Julia surface exposes the whole modeled capability with natural Julia
   types and multiple dispatch;
2. executable conformance tests cover semantics, errors, ownership, lifecycle,
   concurrency, and supported unit domains;
3. representative benchmarks compare the foreign boundary with an appropriate
   native control and enforce an explicit regression budget;
4. every required documentation topic names concrete, existing evidence; and
5. a fresh consumer resolves, installs, loads, and exercises the package.

The audit reads committed `origin/master` trees so unrelated changes in local
worktrees cannot influence the result. It compares `Project.toml`, exported
Julia names, native API models, tests, benchmarks, documentation builds, and
exact-commit GitHub Actions runs. Directory existence is discovery evidence,
not capability evidence. A partial facade therefore remains unfinished even
when its current tests pass.

## Exact reviewed baselines

| Owner | Julia package | Version | Reviewed commit | Exact-commit CI result |
|---|---|---:|---|---|
| Vinary Tree Interop | `VinaryTreeInterop` | `4.0.0-rc.6` | [`c96289cf`](https://github.com/vinary-tree/vinary-tree-interop/commit/c96289cf5182ce1a10a9fefc24535613a3d0d048) | [run 33939191094](https://github.com/vinary-tree/vinary-tree-interop/actions/runs/33939191094) passed |
| llattice | `LLattice` | `0.1.0` | [`945652ea`](https://github.com/vinary-tree/llattice/commit/945652ea20a0c9b46862b9c440428d0568863b20) | [run 33927540772](https://github.com/vinary-tree/llattice/actions/runs/33927540772) passed |
| libdictenstein | `Libdictenstein` | `4.0.0-rc.6` | [`11608be`](https://github.com/vinary-tree/libdictenstein/commit/11608be) | [bindings conformance run 33944186823](https://github.com/vinary-tree/libdictenstein/actions/runs/33944186823) passed, including the Julia collections-and-algebra job; the same commit repairs and locally qualifies Documenter 1.19 strictness |
| liblevenshtein | `Liblevenshtein` | `4.0.0-rc.6` | [`6a2f002d`](https://github.com/vinary-tree/liblevenshtein-rust/commit/6a2f002d78aa21542eeaa9c2cbda9488959964f3) | [run 33919807995](https://github.com/vinary-tree/liblevenshtein-rust/actions/runs/33919807995) passed |
| lling-llang | `LlingLlang` | `4.0.0-rc.6` | [`f7a52356`](https://github.com/vinary-tree/lling-llang/commit/f7a523569aec69b2c1fcaa7ab45e12e089fa3016) | [run 33940641874](https://github.com/vinary-tree/lling-llang/actions/runs/33940641874) passed |
| duallity | `Duallity` | `4.0.0-rc.6` | [`a6b9971c`](https://github.com/vinary-tree/duallity/commit/a6b9971c4498f2441db797ddfddd1d30305a78ca) | [CI run 33925476917](https://github.com/vinary-tree/duallity/actions/runs/33925476917) and [Documenter run 33925476918](https://github.com/vinary-tree/duallity/actions/runs/33925476918) passed |

`LLattice` follows its own `0.1.x` package line; that version is not evidence of
an accidental RC.6 omission. No package was registered or published during
this qualification.

## Capability findings

| Owner | Evidence-backed Julia surface | Remaining parity work |
|---|---|---|
| Vinary Tree Interop | retained resources; dictionary snapshots, visits, compact graphs, entries, and snapshot identity; scalar weighted finite-state transducers (WFSTs); lattice values | generate the complete ABI from the C header; qualify all value and unit-domain branches; add representative boundary benchmarks; complete topic-level docs, fresh-consumer installation, and registry staging |
| llattice | host-implementable lattice interface; maximum/minimum, Boolean, optional, finite-set, and vector-content lattices | strengthen hostile-callback and concurrent lifecycle tests; compare pairwise and batched boundaries with native controls and budgets; complete package, documentation, and registry gates |
| libdictenstein | Julia `AbstractDict`; dynamic DAWG, double-array trie, SCDawg, persistent ARTrie, and persistent vocabulary; snapshots; eager algebra; retained dictionary-resource handoff | sorted-minimal DAWG, suffix automaton, and PathMap backends; lazy zipper algebra and bounded collection operations; customer-implementable providers; generated ABI; full conformance, benchmarks, self-contained native artifacts, docs, and registry gates |
| liblevenshtein | exact distance functions; standard, adjacent-transposition, full Damerau-Levenshtein, and merge-and-split automata over byte, Unicode-scalar, and unsigned-64-bit domains; bounded cursors/reducers; snapshots; a bounded TinyLFU/SIEVE result cache; a limited phonetic surface | generalized, universal, FZF, and WallBreaker automata; complete phonetic families; ranked/contextual and specialized traversal; serialization; configurable cache policies and eviction strategies; complete cross-domain conformance, performance, artifact, documentation, and registry gates |
| lling-llang | eager scalar-WFST construction, immutable resource import, lazy composition, lattice consumption, and customer-implementable scalar-WFST and semiring providers | core WFST algorithms and path search; generic vocabulary and weight domains; CFG/parsing; differentiable, pushdown/WPDS, symbolic, acoustic/ASR/CTC, correction, and language-model decoder surfaces; remaining host-implementable interfaces; performance, artifact, documentation, and registry gates |
| duallity | dictionary-resource bridge; four edit algorithms; Levenshtein plus basic universal, generalized, phonetic, and FZF selector-based WFST construction; product composition through lling-llang | WallBreaker; full phonetic rewrite/NFA pipeline; FZF configuration, statistics, scoring, and cache controls; detailed generalized/universal construction policies; representative benchmarks, self-contained artifacts, complete docs, and registry gates |

Passing tests establish the implemented slice; they do not prove the absent
slice. In particular, the lling-llang extension-provider matrix already records
37 missing Julia translations and three partial generic WFST translations.
Likewise, a duallity enum selector is not a substitute for the public Rust
configuration, statistics, scorer, rewrite, or cache APIs behind that selector.

## Packaging, performance, and documentation findings

The current packages resolve sibling packages with `Pkg.develop(path=...)`.
That is an effective source-graph conformance test, but most packages do not yet
ship a Julia artifact (`JLL`) or another reproducible resolver for their native
library. General-registry registration, immutable tag provenance, TagBot or
Registrator integration, and public registry readback remain release work. The
RC.6 preparation rule is therefore: prepare and test metadata locally, but do
not publish or create a release tag.

Benchmark maturity differs by package. Vinary Tree Interop and libdictenstein
have no Julia benchmark. liblevenshtein times two scalar distance calls with
warmup and repeated samples, but has no native control or regression budget.
llattice uses repeated samples for pairwise and batched folds, while
lling-llang uses BenchmarkTools for composition and expansion. Duallity reports
one aggregate constructor timing. These are useful smoke measurements, not yet
the controlled cross-language evidence required for a `complete` matrix cell.

Documenter builds exist in all six packages, but the required 20-topic model is
more demanding than a generated symbol list. API reference, ownership,
concurrency, security, performance, migration, compatibility, release, and
validated common/intended examples must each cite their own evidence. The
matrix generator rejects a future `complete` cell whose inherited
documentation state has no explicit evidence.

## Distribution-only repositories

The `javascript-runtime` and `liblevenshtein-npm` repositories are
distribution owners, not native algorithm owners. Their Julia cells are
inapplicable for architectural reasons:

- `javascript-runtime` exists to place the four Rust libraries inside one
  JavaScript runtime identity and one WebAssembly memory/resource table. Its
  N-API, browser WebAssembly, and WebAssembly System Interface (WASI) paths are
  JavaScript deployment mechanisms. Julia's native `ccall` packages exchange
  the existing two-word Vinary Tree resource ABI in one native address space;
  placing a second Julia facade in the JavaScript umbrella would duplicate the
  six owner packages and introduce an unnecessary JavaScript runtime boundary.
- `liblevenshtein-npm` is the unscoped npm compatibility bridge for legacy
  JavaScript consumers. It contains no independent algorithm or ABI. A Julia
  package under that repository would neither preserve a Julia legacy
  coordinate nor own a new capability, so it would be a misleading duplicate
  of `Liblevenshtein`.

The [WASM topology](wasm-topology.md) gives the complete runtime-memory and
identity argument:

![The JavaScript umbrella owns its three runtime paths and resource table; ordinary native language packages exchange the shared resource ABI outside that distribution-only boundary.](../diagrams/bindings/wasm-umbrella-deployment.svg)

These proofs concern repository ownership only. They do not excuse any missing
capability in the six native Julia packages.

## Completion boundary

Julia family parity is complete only when every applicable row is implemented
and its five evidence gates pass, the six distribution-only rows retain the
reviewed proof above, exact-current-commit CI is green, every edited worktree is
clean, and a later explicitly authorized release verifies public registry and
documentation readback. Until then, the generated matrix must preserve the
remaining work as `audit-required` or `missing`; it must never infer completion
from a binding directory alone.
