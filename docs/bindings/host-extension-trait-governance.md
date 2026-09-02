# Host-extension trait governance

This document governs how public extension traits from `llattice` and
lling-llang become customer-implementable interfaces in other languages. Its
machine-readable result is the
[`extension-provider-matrix.tsv`](../../bindings/conformance/extension-provider-matrix.tsv).
That matrix reports observed implementation evidence; it is not a promise that
an unimplemented facade already exists.

The central rule is simple: every discovered Rust `pub trait` must have one
stable foreign-interface design classification, and every supported language
must have an explicit implementation status. Adding a trait without extending
the classification model fails the binding gate. A trait therefore cannot
silently disappear behind a broad claim such as “custom providers supported.”

![A host implementation crosses a language facade, a negotiated resource capability, and a validated dynamic Rust adapter before an algorithm can consume it.](../diagrams/bindings/three-layer-architecture.svg)

## Terms and scope

A **host** is the process or runtime supplying callbacks, such as a JVM, Julia,
Raku, CPython, or native C application. A **provider** is a host-owned
implementation made available through a negotiated `VtResource` capability.
An **adapter** validates that capability and presents it to a Rust algorithm.
A **facade** is the target-language API that owns callbacks and resources using
that language's normal lifecycle and error conventions.

The audit covers the Rust sources beneath `llattice/src` and
`lling-llang/src`. It uses the same language sequence and idiom descriptions as
[`family-bindings.json`](../../bindings/conformance/family-bindings.json). If
there are $`T`$ discovered traits and $`L`$ modeled languages, the generated
matrix contains exactly $`T \times L`$ data rows. Duplicate trait names in
different modules remain distinct because identity is the triple
`project:source-path:trait-name`.

This is deliberately narrower than general facade completeness. A language may
consume built-in WFSTs while still lacking an API through which customers can
implement a WFST. The former belongs in the family capability matrix; the
latter is what this matrix measures.

## Translation classifications

Rust generics, associated types, borrowed values, and `Self` constraints do not
cross an application binary interface (ABI) directly. Each trait is assigned
one of these implementation strategies:

| Classification | Stable foreign form | Typical use |
|---|---|---|
| `direct-capability-vtable` | A versioned callback table queried from a retained resource. Values are either immutable resources or provider-scoped generation-checked tokens. | `Lattice`, `Semiring`, and optional semiring operations. |
| `resource-cursor-protocol` | Opaque retained resources plus stable identifiers and bounded cursors or pages. | WFST, backend, dictionary, pushdown, and tree-automaton traversal. |
| `bounded-batch-protocol` | Caller-owned buffers with explicit capacity, written count, total count, and lease rules. | Mutation streams, checkpoint codecs, and operations whose per-item callback cost would dominate. |
| `high-level-operation-interface` | Coarse operations with owned inputs/results and explicit error translation. | Acoustic models, parsers, correction layers, decoders, and symbolic theories. |
| `declared-law-marker` | Capability bits plus executable law validation; a bit is never inferred merely from the presence of base operations. | Idempotence, hash coherence, zero-sum freedom, commutativity, and nonnegativity. |
| `derived-adapter` | A facade exposes the prerequisite capabilities; the library derives the extension operation without another callback table. | Iterator helpers, algebra bridges, and blanket operation traits. |
| `reviewed-rust-only-proof` | A reviewed proof explains why a sound foreign form is impossible or meaningless. | Exceptional cases only; implementation absence is not proof. |

The classification describes the intended stable shape, not current delivery.
For example, `resource-cursor-protocol` on a missing row means the design must
use that shape when implemented; it does not make the row complete.

## Status semantics

| Status | Exact meaning |
|---|---|
| `complete` | The named language level has executable source evidence for the provider surface. For C this means the public C ABI itself; for a managed language it means an idiomatic facade, not just the ability to hand-write C callbacks. |
| `abi-available` | The shared C capability exists and can technically be called, but the named language still lacks its safe, idiomatic provider facade. |
| `partial` | An evidence-backed specialization exists, but it does not represent the full generic Rust trait. The `support_scope` column names the exact boundary. |
| `missing` | No implementation evidence is registered for that language and trait. |
| `review-required` | The surface was discovered, but its design or evidence has not yet been classified sufficiently. The strict generator does not permit this state for the trait classification itself. |
| `inapplicable` | A reviewed architectural proof establishes that the provider concept does not apply. The absence of implementation, tooling, or time is never enough. |

Native Rust rows are complete because the authoritative public trait is itself
the implementation surface. This does not imply dynamic dispatch or foreign
callbacks on the native path: monomorphized algorithms retain their existing
zero-cost route.

### Sealed dynamic adapter markers

The `Sealed`, `LatticeAccess`, and `SemiringAccess` declarations in
lling-llang's dynamic adapters are not customer extension points. Their sole
purpose is to select one of two library-owned type-level access modes:
same-thread access for the sound default, or cross-thread access after the
provider's parallel-reentrancy flag has been validated. The `Sealed` traits
live in private modules, and `LatticeAccess` and `SemiringAccess` require those
private supertraits. External Rust code therefore cannot implement them, and a
foreign facade must not let callers forge them.

All four declarations are classified `reviewed-rust-only-proof`, and every
language row is `inapplicable`. Customers still implement the underlying
lattice or semiring provider through its negotiated resource vtable; the
adapter chooses the access marker after validation. Exposing the markers as
host callbacks would erase the safety distinction and permit a thread-bound
runtime object to masquerade as parallel-reentrant.

## Existing shared capabilities

The shared interop header currently supplies three relevant foundations:

1. `VtLatticeVTable` represents immutable join/meet values, equality, stable
   bytes, diagnostics, and optional bounded folds.
2. `VtSemiringVTable` represents the base algebra over provider-scoped compact
   tokens. Separate division, star, numeric, and declared-property interfaces
   prevent an oversized table and allow capability negotiation.
3. `VtWfstVTable` represents immutable snapshot-scoped state identifiers and
   caller-owned arc pages. `VT_WFST_FLAG_LAZY` states that the provider may
   expand a state on demand; it does not weaken snapshot consistency.

The current WFST table carries `double` arc weights. It completely represents
the deliberately scalar `ScalarWfstProvider`, but only a scalar-weight
specialization of the generic Rust `Wfst`, `LazyWfst`, and `StateSource`
traits. Their C, C++, Julia, and Raku rows are consequently `partial`, not
`complete`. C++ does have an idiomatic C++20 provider facade for that scalar
WFST specialization as well as the generic semiring capability; small
semiring values use allocation-free inline tokens, and resource ownership is
mutex-free. General semiring-valued WFSTs still require a future weight-domain
capability rather than an inflated claim about the scalar table.

![The consumer queries an exact interface identity, validates its version and size, and either admits the capability or releases the retained resource.](../diagrams/bindings/interface-negotiation-activity.svg)

An ABI capability is registered only when its named tokens occur in the cited
evidence. A complete or ABI-available row must cite existing files. This guards
against classifications based only on memory or intent.

## Discovery and verification algorithm

The generator is intentionally dependency-free so it can run before Cargo,
package-manager, or foreign-runtime setup. In literate pseudocode:

```text
DISCOVER-AND-RENDER(model, family-model):
    roots <- resolve relative project roots, honoring explicit environment overrides
    traits <- every uncommented `pub trait` declaration below each project's `src`
    require keys(traits) = keys(classifications)
    require every implementation names a discovered trait and modeled language

    for each implementation evidence reference:
        require its project is known and its path stays inside that project
        require the referenced file exists
    for each declared ABI capability:
        require every dotted capability token occurs in the evidence corpus

    for each trait in stable key order:
        for each family language in canonical order:
            emit native Rust evidence, registered foreign evidence, or `missing`

    compare byte-for-byte with the committed matrix when `--check` is active
```

The scanner masks nested comments, ordinary multiline strings, and raw strings
before recognizing declarations. This prevents documentation examples from
becoming phantom API traits. Module paths are part of identity, so the two
distinct lling-llang `LanguageModel` contracts cannot overwrite one another.

## Portable repository discovery

The model stores only repository-relative defaults:

| Project | Default | Optional override |
|---|---|---|
| llattice | `../llattice` | `LLATTICE_ROOT` |
| lling-llang | `../lling-llang` | `LLING_LLANG_ROOT` |
| vinary-tree-interop evidence | `../vinary-tree-interop` | `VINARY_TREE_INTEROP_ROOT` |

Machine-specific absolute paths are forbidden in the model. An integration
worktree may be selected without editing tracked files:

```bash
LLING_LLANG_ROOT=../lling-llang-feature-worktree \
  python3 scripts/generate-extension-provider-matrix.py --check
```

This separation matters for release reproducibility: CI and a fresh consumer
use canonical sibling names, while local integration can select another
checkout explicitly.

## Lifecycle, concurrency, and security requirements

Every eventual facade represented by a complete row must uphold the shared
resource rules:

- retain callback owners for at least as long as native code may call them;
- expose deterministic close or disposal in the host language, with finalizers
  only as leak containment;
- translate host exceptions into bounded status and diagnostic values without
  unwinding across the C boundary;
- validate table size, version, interface identity, reserved fields, counts,
  identifiers, and provider-scoped value tokens before use;
- invoke no foreign callback while holding an unrelated internal lock;
- serialize only a provider that declines parallel reentrancy, and keep that
  gate scoped to the provider rather than the process;
- use bounded batches or cursor pages where callback frequency would otherwise
  dominate useful work; and
- preserve native monomorphized specializations whenever dynamic dispatch has
  a material cost.

Law-bearing semiring flags require property tests over representative values.
They are declarations used to select algorithms, so a false flag can violate
termination or correctness even when the memory boundary remains safe.

## Maintainer workflow

After changing a public trait or adding a provider facade:

1. Update
   [`extension-provider-model.json`](../../bindings/conformance/extension-provider-model.json)
   with the trait's translation class and evidence-backed language statuses.
2. Regenerate the matrix:

   ```bash
   python3 scripts/generate-extension-provider-matrix.py
   ```

3. Run the focused tests and the binding gate:

   ```bash
   python3 -m unittest scripts.tests.test_generate_extension_provider_matrix
   python3 scripts/check-bindings.py --check
   ```

4. Run each affected host's lifecycle, hostile-provider, law, and performance
   suites. A matrix row advances to complete only after those sources are part
   of its evidence.

The generator is also invoked by `scripts/check-bindings.py`. A stale matrix,
an unclassified new trait, a removed trait still present in the model, a
missing evidence file, an unknown language, an escaped path, or an unevidenced
ABI capability therefore fails the same gate as the rest of the binding
architecture.
