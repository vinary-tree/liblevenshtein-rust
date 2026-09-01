# Publishing package documentation

This document defines how a language binding becomes **documented in public**.
Building a package, uploading it, and receiving an HTTP-success response are
three different facts. A release satisfies this policy only when a consumer can
find a versioned guide and a complete generated application programming
interface (API) reference from the package's natural ecosystem.

The machine authority is
[`release/package-documentation.json`](../../release/package-documentation.json).
[`scripts/check-package-documentation.py`](../../scripts/check-package-documentation.py)
validates its local evidence, and its `--public` mode reads the unauthenticated
registry and documentation bytes. The strict `--require-complete` mode remains
red while a publishable package is still pending or any released package lacks
either required destination. An explicitly non-published numeric-only release
candidate is reported but does not consume the future final-version coordinate.

![The release operator validates immutable source, publishes one package, reads the public registry bytes, and records evidence only after a clean consumer succeeds.](../diagrams/bindings/release-operator-flow.svg)

## Terms and evidence levels

A **package surface** is one coordinate through which one or more languages
consume the same artifact. JavaScript, TypeScript, and ClojureScript therefore
share an npm surface; Java consumes the Maven artifact's Java API, while Kotlin
and Scala additionally require language-specific usage and reference material.
A **package guide** teaches installation, common and intended use, lifecycle,
errors, compatibility, and security. An **API reference** enumerates and
describes every public declaration. A **readback** is an unauthenticated fetch
from the public service, not a local build directory or workflow artifact.

The destination states are deliberately non-overlapping:

| State | Evidence |
|---|---|
| `verified` | The exact public URL was fetched, its version and identity markers matched, and the observation date is recorded. |
| `build-only` | Immutable-source automation produces the documentation artifact, but no public service serves it yet. |
| `missing` | The released package has no qualifying destination. The accompanying reason records the observed defect. |
| `deferred` | The package itself is an explicitly non-published candidate because the ecosystem cannot represent the release-candidate version. |

`build-only` is progress, not completion. Similarly, a registry page that
contains only a name, version, or rock specification is package metadata, not a
guide and not an API reference.

## Canonical destinations

The service follows the language rather than a preference invented by the
project:

| Language or runtime | Package guide | Generated API reference | Publication mechanism |
|---|---|---|---|
| Rust | docs.rs crate landing page | rustdoc on docs.rs | docs.rs builds the immutable crates.io source archive. |
| C and C++ | Versioned native release guide | Doxygen | Protected documentation deployment consumes the exact release tag and headers. |
| Python | Versioned PyPI description | pdoc long-form site | The wheel/sdist README and pdoc source must identify the same release. |
| Java | Maven Central metadata and JVM guide | Javadoc on javadoc.io | Maven Central receives sources and Javadoc JARs in the signed deployment. |
| Kotlin and Scala | JVM guide with native idioms | Dokka and Scaladoc | Generated references accompany the same Maven coordinate; they do not imply separate runtime libraries. |
| Clojure | Clojars metadata | cljdoc guide and namespace API | cljdoc reads the immutable Clojars artifact and source tag. |
| JavaScript and TypeScript | npm README | TypeDoc | The release job builds TypeDoc from the published declaration surface, then a protected documentation job deploys it. |
| ClojureScript | npm README and namespace guide | Generated namespace/API reference | The npm artifact and its exact singleton runtime remain the distribution authority. |
| C# and F# | NuGet README | DocFX | DocFX consumes the same public declarations and compiler XML shipped in the NuGet archive. |
| Go | pkg.go.dev overview | pkg.go.dev declarations | The public module tag is the immutable source authority. |
| Swift | Swift Package Index package page | DocC | Swift Package Index builds the tagged SwiftPM package. |
| Ruby | RubyGems guide link | RDoc/RubyDoc.info | The gem metadata must expose a documentation URI and the generated reference must resolve for the exact gem version. |
| Lua | LuaRocks guide | Versioned Lua/C-module API reference | The rock page must route to a rendered guide and generated public entry-point reference. |
| OCaml | ocaml.org package guide | odoc | The upstream opam-repository merge triggers the canonical build. |
| Haskell | Hackage package guide | Haddock | Hackage builds the numeric final release; RC candidates do not consume that coordinate. |
| Fortran | fpm package guide | FORD | The numeric final release publishes after the RC embargo. |
| Raku | fez/zef metadata and Pod6 guide | Rendered Pod6 API | Future NativeCall packages publish from immutable source with fez provenance. |
| Julia | General registry package page and Documenter landing page | Documenter.jl | Registrator/TagBot identify the immutable tag; Documenter deploys doctested version directories. |

The Raku and Julia rows are mandatory target architecture. They are not claims
that packages already exist; the family completeness matrix keeps every absent
project/language capability visible until its binding campaign supplies source,
tests, benchmarks, documentation, and a fresh installed consumer.

## Release invariant

For a released package $`p`$ at version $`v`$, let $`R(p,v)`$ be its registry
bytes, $`G(p,v)`$ its guide, $`A(p,v)`$ its API reference, and $`S(p,v)`$ the
immutable source revision. Completion requires:

```math
\operatorname{version}(R)=\operatorname{version}(G)=
\operatorname{version}(A)=v
\quad\land\quad
\operatorname{source}(R)=\operatorname{source}(G)=
\operatorname{source}(A)=S(p,v).
```

The equality is checked with service-appropriate evidence: exact URL segments,
registry metadata, generated source links, artifact digests where exposed, and
representative public-symbol markers. A floating `latest` page may help users
discover a release, but it cannot prove this invariant.

## Literate verification algorithm

The algorithm fails closed. It never upgrades a missing destination merely
because another destination exists.

```text
procedure VerifyPackageDocumentation(manifest M, public_readback):
    assert M.component and M.version equal the release authority
    for each package surface p in M:
        assert every source-evidence path exists in the owning repository
        if p is released:
            fetch its unauthenticated registry readback when public_readback
            assert the exact registry version and coordinate markers
        otherwise:
            require an explicit package-publication proof

        require exactly one package-guide destination
        require exactly one API-reference destination
        for each verified destination d:
            require an HTTPS URL and an ISO observation date
            fetch d.readback_url when public_readback
            assert every identity, version, and public-symbol marker
        for each build-only destination d:
            require existing generator and workflow evidence
        preserve every missing or deferred destination as an incomplete state

    under strict completion, reject pending publication and every non-verified
    destination of a released package
end procedure
```

Run the source and public gates independently:

```sh
python3 scripts/check-package-documentation.py
python3 scripts/check-package-documentation.py --public
python3 scripts/check-package-documentation.py --public --require-complete
```

The final command is the release acceptance gate. It is expected to fail while
the manifest truthfully contains released `missing` or `build-only` entries.

## Reproducible references and immutable version history

Three generators feed one versioned site. Doxygen reads the public C and C++
headers and emits both Hypertext Markup Language (HTML) and Extensible Markup
Language (XML); the XML inventory proves that every modeled C function and
public native type appears. pdoc renders the Python facade only after an
Abstract Syntax Tree (AST) check proves that every exported class and public
method has explanatory text. TypeDoc renders the JavaScript and TypeScript
declaration surface. All three receive the canonical version and immutable
source tag from [`release/version.json`](../../release/version.json).

The local construction uses repository storage under `target/`; it does not
consume memory-backed system temporary storage:

```sh
uv sync \
  --project bindings/python \
  --group documentation \
  --frozen \
  --no-install-project
npm ci --prefix bindings/javascript --ignore-scripts
cargo build --locked --features native-bindings-full
PATH="$PWD/bindings/python/.venv/bin:$PATH" \
LIBLEVENSHTEIN_LIBRARY="$PWD/target/debug/libliblevenshtein.so" \
  python3 scripts/build-package-documentation.py --surface all
python3 scripts/package-documentation-site.py build
python3 scripts/package-documentation-site.py assemble \
  --archives target/package-documentation-artifacts
```

The archive builder normalizes member order, timestamps, owners, groups, and
permission modes, so identical source produces identical gzip and tar bytes.
Its manifest records the SHA-256 digest of every served file. The assembler
rejects absolute paths, parent traversal, links, special files, duplicate
members, unknown surfaces, unlisted files, missing files, and digest changes
before extracting a version. A release asset is therefore both the immutable
history record and the input from which the complete site can be reconstructed.

The protected workflow implements the preservation algorithm literally:

```text
procedure PublishVersionedReferences(exact_tag T, archive A):
    assert T equals release.version.publication.sourceTag
    build native, Python, and JavaScript references from T
    assert A is reproducible and every public declaration is represented

    if release(T) already contains an asset named like A:
        download the existing asset and require byte equality
    else:
        append A to release(T)

    download every version archive from every release
    authenticate and safely extract every archive into its version directory
    sort the versions by semantic-version precedence
    deploy the entire reconstructed tree through the protected Pages environment
end procedure
```

This design never uses the current branch as evidence for an older tag and
never replaces a historical version directory with a newer build. The RC5
manifest consequently remains `missing` for native Doxygen and Python pdoc:
the exact `v4.0.0-rc.6` source does not contain this later automation, and an
unpublished local build cannot retroactively satisfy public evidence.

## RC5 evidence and confirmed gaps

The 2026-08-29 unauthenticated audit proves the Rust, Java, Clojure, and Go API
services and the versioned source guides named as `verified` in the manifest.
It also confirmed several failures that an HTTP-only inventory would miss:

- npm's RC5 version exists and its tarball contains `README.md`, but the public
  package metadata retained an empty `readme` field;
- both Ruby gems exist, but liblevenshtein has no `documentation_uri` and its
  exact RubyDoc.info page returns 404;
- the LuaRocks page exposes the rock specification and archives but none of the
  binding guide's public API or examples;
- NuGet ships compiler XML, but there is no browsable versioned DocFX site;
- the Swift tag exists, but no verified Swift Package Index DocC page exists;
- the Python project page is public, but no versioned pdoc reference exists;
- the OCaml candidate is attached to the GitHub prerelease, while the upstream
  opam package URL still returns 404; and
- Hackage and fpm remain intentional package-publication candidates because
  their numeric-only `4.0.0` coordinates must remain available for the final
  release.

These are evidence-backed work items, not speculative defects. The manifest
retains them until a protected deployment and subsequent public readback prove
the opposite.

## Security and publication authority

Documentation deployment is an external mutation. It uses the same immutable
tag checks and protected-environment review as package publication. A workflow
artifact may cross from a build job to a deployment job, but credentials never
cross in the other direction. Generated pages must not execute package code
from an untrusted pull request, import a mutable sibling checkout, or derive a
source link from the current default branch.

Every generated site must preserve the complete binding trust model: borrowed
lease duration, deterministic close behavior, finalizer limitations, callback
fault containment, concurrency and reentrancy, hostile-input ceilings, and the
exact package/runtime identity. Examples are compiled or executed before
deployment; decorative snippets do not count as validation evidence.

## Maintainer completion sequence

1. Generate the guide and API reference from the exact release source.
2. Fail the build on undocumented public symbols and broken internal links.
3. Publish the package through its protected registry environment.
4. Publish or trigger the ecosystem documentation service.
5. Fetch the registry page, guide, API index, deep symbol links, assets, and
   source links without credentials.
6. Exercise the documented quick start from a clean installed consumer.
7. Record immutable URLs, observation time, workflow run, and digest evidence.
8. Change the manifest state to `verified` only after all prior steps pass.

No step may be inferred from the success of another.
