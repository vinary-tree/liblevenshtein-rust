# Family-wide binding completeness

This directory contains three distinct forms of evidence:

- [`completeness-matrix.tsv`](completeness-matrix.tsv) audits the functions of
  the liblevenshtein facades that already exist; and
- [`public-api-traceability.tsv`](public-api-traceability.tsv) maps every
  modeled facade API item to source, guide, executable tests, and a canonical
  example, distinguishing direct references from evidence that still needs
  item-level review;
- [`family-completeness-matrix.tsv`](family-completeness-matrix.tsv) prevents a
  missing project, language, capability, package, idiom, or verification lane
  from disappearing outside a repository-local model.

The first matrix answers “how much of liblevenshtein's C surface does this
facade map?” The family matrix answers the prior question: “which public
capabilities must exist naturally in every target language across the whole
Vinary Tree family, and what evidence proves each cell?” No matrix substitutes
for either of the other two.

The traceability matrix includes functions, enums, and the iterator/reducer
protocol entry points. A reasoned absence is retained with its rationale and
marked inapplicable for source, documentation, test, and example evidence. An
exposed item's guide must name the exact public symbol; its source, test, and
example corpora must directly name that symbol or its backing native operation.
A language-level test file that never touches the item remains
`audit-required`.

![Family completeness pipeline from the declared project, language, and capability axes through generated cells and evidence gates.](../../docs/diagrams/bindings/family-completeness-gate.svg)

## Vocabulary and row identity

A **capability** is a user-visible semantic unit such as a retained resource,
dictionary snapshot, lattice operation, edit automaton, semiring, or weighted
finite-state transducer (WFST). A **host idiom** is the target language's
familiar protocol for that capability—for example Java `AutoCloseable` and
`Collection`, C# `IDisposable` and `IEnumerable`, Python context managers and
iterators, or Raku `NativeCall` plus `Iterable`.

One matrix row has the stable identity
$`(p, l, c) \in P \times L \times C_p`$, where $`P`$ is the project set,
$`L`$ is the language set, and $`C_p`$ is project $`p`$'s public capability
catalog. The initial inventory contains 8 projects, 21 languages—including
native Rust and Raku—and 64 project-owned capabilities, producing
$`21 \sum_{p \in P} |C_p| = 1{,}344`$ explicit cells.

Each row also records:

- unit domains, including byte, Unicode scalar, vocabulary identifier, and
  unsigned 64-bit token domains where applicable;
- the language-native collection, ownership, error, and extension idioms;
- the canonical package manager;
- deterministic lifecycle expectations;
- conformance, benchmark, API-documentation, and fresh installed-consumer
  states; and
- source evidence or a reviewed applicability proof.

The documentation state is not a free-form checkbox. It is derived from 20
independently reviewable topic states, so an overview cannot conceal a missing
API reference and a generated reference cannot conceal missing lifecycle or
security guidance. Every topic has a neighboring evidence column. A topic can
become `complete` only when its model entry cites at least one documentation,
validated-example, or generated-reference artifact. Local evidence is resolved
relative to the owning project and must name an existing file without escaping
that project; remote evidence must use HTTPS.

| Documentation topic | Required evidence |
|---|---|
| Overview | Purpose, scope, audience, and relationship to the family. |
| Installation | Exact coordinates, platforms, prerequisites, and an installation check. |
| Quick start | The shortest validated fresh-consumer path to a useful result. |
| Common usage | Executable examples for operations most users perform. |
| Intended usage | Executable examples for the capability's designed role and important alternatives. |
| API reference | Every public symbol, signature, overload, generic, and optionality contract. |
| Semantics | Behavior, invariants, ordering, mutation, snapshots, and unsupported combinations. |
| Errors | Exceptions, results, statuses, cancellation, and containment. |
| Lifecycle | Creation, deterministic closure, finalization fallback, and use after close. |
| Ownership | Borrowing, retention, transfer, aliasing, and resource handoff. |
| Concurrency | Thread safety, reentrancy, synchronization, and cancellation guarantees. |
| Collections and iterators | Native collection, iterator, reducer, stream, and traversal behavior. |
| Snapshots | Identity, isolation, consistency, mutation visibility, and cursor lifetime. |
| Zero-copy batching | Borrowed views, leases, buffer validity, paging, and copying tradeoffs. |
| Examples | Validated common, intended, error, lifecycle, and performance examples. |
| Migration | Prior coordinates, APIs, behavior, and release-line migration. |
| Compatibility | Runtime, ABI, platform, feature, and version guarantees. |
| Performance | Complexity, allocation, marshalling, benchmarks, and sensitive usage. |
| Security | Trust boundaries, hostile-input limits, callback safety, and reporting. |
| Release | Publication ownership, immutable provenance, registry verification, and operations. |

## State meanings

| State | Meaning |
|---|---|
| `missing` | No declared facade evidence exists. This is an implementation requirement, never an implicit exemption. |
| `audit-required` | A facade or native surface exists, but capability-level API, idiom, conformance, performance, documentation, and installed-package parity have not all been proved. |
| `review-required` | A distribution-only repository may be redundant for this language, but the architectural proof has not yet been reviewed. |
| `inapplicable` | A reviewed proof explains why direct exposure would be semantically wrong or duplicative; the generator rejects this state without a real proof file. |
| `complete` | The capability is implemented idiomatically and its conformance, benchmark, documentation, and fresh-consumer evidence are all complete. |

The strict completion gate accepts only `complete` and proved `inapplicable`
cells. The ordinary generator is intentionally usable during the campaign: it
requires the Cartesian inventory to remain exhaustive while preserving every
unfinished state as visible work.

## Generation and verification

[`family-bindings.json`](family-bindings.json) is the reviewed input model.
[`generate-family-completeness-matrix.py`](../../scripts/generate-family-completeness-matrix.py)
validates project roots and evidence paths, rejects duplicate or unknown cells,
requires Rust and Raku to remain explicit languages, requires the complete
canonical documentation-topic sequence, derives the aggregate documentation
state from those topic states, and emits the TSV in a deterministic
project/capability/language order.

```sh
# Regenerate after reviewing a model or evidence change.
python3 scripts/generate-family-completeness-matrix.py

# CI-shaped freshness and structural check.
python3 scripts/generate-family-completeness-matrix.py --check

# Final epic gate; expected to fail while any real work remains.
python3 scripts/generate-family-completeness-matrix.py \
  --check --require-complete

# Documentation-only final gate; identifies unfinished topic tuples directly.
python3 scripts/generate-family-completeness-matrix.py \
  --check --require-documentation-complete

# Public-symbol traceability, then its strict final campaign gate.
python3 scripts/generate-binding-traceability.py --check
python3 scripts/generate-binding-traceability.py --check --require-complete
```

The literate generation algorithm is:

```text
for each project p in the reviewed family model:
    validate that p's root and every declared evidence path exist
    for each public capability c owned by p:
        for each required language l:
            emit exactly one cell (p, l, c)
            if an override says inapplicable:
                require a reviewed proof file
            for each required documentation topic d:
                emit and validate the state of d
            derive the aggregate documentation state from all topic states
            preserve implementation and all four evidence dimensions

reject overrides that name cells outside the Cartesian product
compare generated bytes with the committed matrix
under --require-complete, reject every unproved or unfinished cell
```

## Advancing a cell

Do not change `missing` to `complete` merely because a directory or wrapper
exists. First audit the whole capability against the native Rust semantics and
the language's intended idioms. Then add shared differential fixtures,
property and lifecycle tests, representative benchmarks, complete API and
usage documentation, package-native validation, and a clean consumer that
resolves the exact public artifact. Record those evidence paths in a
`cellOverrides` entry before changing each dimension to `complete`. Use the
`documentationTopics` object inside that override to advance topics
independently, for example
`"overview": {"state": "complete", "evidence": ["docs/README.md#overview"]}`.
The generator computes `documentation_status`; there is no writable aggregate
override, so a topic gap cannot be hidden by declaring the aggregate complete.

When direct exposure is genuinely redundant, add an architectural proof that
names the alternative owner, explains why another facade would create two
identities or conflicting lifecycle semantics, and identifies the conformance
tests that protect the delegation. Only then may the row become
`inapplicable`.
