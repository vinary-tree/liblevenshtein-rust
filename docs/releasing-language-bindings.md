# Releasing language bindings

Bindings are released as a dependency graph, not as one monolithic artifact.
The shared interop packages publish first, producer projects such as
libdictenstein publish next, and consumers such as liblevenshtein publish after
cross-project conformance succeeds.

![Publish-order DAG: vinary-tree-interop first, libdictenstein second, liblevenshtein third, the WFST siblings next, then the npm umbrella runtime and finally the project npm facades — every edge an exact-version pin, never a moving branch.](diagrams/bindings/registry-topology.svg)

## Release order

For a liblevenshtein release:

1. Publish `vinary-tree-interop` version required by `bindings/api.json` to
   crates.io, PyPI, Maven Central, npm, NuGet, Hackage, fpm, opam, Go, and the
   C/C++ archive channel. Push the immutable `interop-vX.Y.Z` tag; the dedicated
   `release-interop.yml` workflow owns these coordinates.
2. Publish or verify the compatible libdictenstein bindings. Their package owns
   dictionary construction, CRUD, persistence, and concrete dictionary types.
3. Build liblevenshtein against those public coordinates and the stable
   `vt.dictionary.v1` ABI.
4. Run the cross-project handoff fixture: construct/mutate the dictionary with
   libdictenstein, pass its retained resource to liblevenshtein, and verify one
   long-lived query-start snapshot plus a fresh query.
5. Publish liblevenshtein's native packages and language facades.
6. Publish the JavaScript umbrella runtime before its lightweight project
   facades.

The ordinary `vX.Y.Z` project workflow validates a local copy of the common
ABI but never republishes it. This separation is required: the ABI has its own
version and may be consumed by several project releases. `libdictenstein`,
`lling-llang`, and `duallity` likewise publish from their own repositories and
their own tags.

No job derives coordinates from an author name, email address, or checkout
path. The organization is exactly `vinary-tree`; the author is Dylon Edwards
`<dylon.devo@gmail.com>`.

## Pin coherence preconditions

Every edge in the DAG above is an exact-version pin, so a release is legal
only when the pins agree everywhere they are written down. Before pushing
**any** tag in the sequence, all of the following must hold simultaneously:

1. **Model ↔ crate agreement.** `bindings/api.json` `packageVersion` equals
   the crate version in `Cargo.toml`, and the interop version the model
   implies is the one `vinary-tree-interop/Cargo.toml` carries.
   `python3 scripts/generate-bindings.py --check` proves the generated
   constants, headers, and conformance fixtures are byte-stable against the
   model; `python3 scripts/check-bindings.py` proves symbol parity,
   coordinates, the umbrella identity guard, and the facade completeness
   matrix.
2. **Sibling pins name real, existing tags.** Every version in
   `bindings/related-projects.json` and every checkout input in
   `release.yml` refers to a tag that exists and contains the ABI surface
   being pinned. The dev-sibling checkout action is a CI convenience only;
   no release job may build against a moving sibling branch.
3. **Interop is public before anything that depends on it.** The exact
   `vinary-tree-interop` version pinned by this release exists on every
   registry a facade will resolve it from (crates.io, PyPI, Maven Central,
   npm, NuGet, Hackage, fpm, opam, Go, the C/C++ archive) — pushed via
   `interop-vX.Y.Z` and `release-interop.yml`, never republished by a
   project workflow.
4. **Producer before consumer, proven by handoff.** The pinned
   libdictenstein artifacts are public and the cross-project handoff
   fixture (construct/mutate with libdictenstein, hand the retained
   resource to liblevenshtein, verify one long-lived query-start snapshot
   plus a fresh query) passes against exactly those coordinates — not
   against a local sibling checkout.
5. **Umbrella before facades.** The npm umbrella version the project
   facades pin exists on npm before any facade publishes, and each facade's
   runtime-identity guard matches the umbrella it pins.
6. **No known pin inconsistencies outstanding.** The findings ledger's
   version-pin entry (`LLEV-B9` in
   [docs/bindings/FINDINGS_LEDGER.md](bindings/FINDINGS_LEDGER.md))
   enumerates the currently known divergences — e.g. a sibling version
   pinned before its tag exists, or a crates.io version behind the local
   tree. Every such entry must be resolved (or explicitly re-scoped to a
   later release) before the first tag of the sequence is pushed; pin
   divergences discovered *during* a release abort it.

A release that cannot satisfy one of these preconditions is not "mostly
ready" — the failure mode each one guards against is a public artifact
permanently built against the wrong bytes (published tags and registry
versions are never moved; see "Versioning and rollback").

## Registry coordinates

| Ecosystem | Shared interop | Libdictenstein producer | Liblevenshtein consumer |
|---|---|---|---|
| crates.io | `vinary-tree-interop` | `libdictenstein` | `liblevenshtein` |
| PyPI | `vinary-tree-interop` | `vinary-tree-libdictenstein` | `vinary-tree-liblevenshtein` |
| Maven Central | `io.vinarytree:vinary-tree-interop` | `io.vinarytree:libdictenstein` | `io.vinarytree:liblevenshtein` |
| Clojars | Java dependency from Central | `io.vinarytree/libdictenstein-clojure` | `io.vinarytree/liblevenshtein-clojure` |
| npm | `@vinary-tree/interop` | `@vinary-tree/libdictenstein` over the umbrella | `@vinary-tree/liblevenshtein` over the umbrella |
| C/C++ | `vinary-tree-interopConfig.cmake` | `libdictensteinConfig.cmake` and `libdictenstein.pc` | `liblevenshteinConfig.cmake` and `liblevenshtein.pc` |
| .NET | `VinaryTree.Interop` | `VinaryTree.Libdictenstein` | `VinaryTree.Liblevenshtein` |
| Go | common module subdirectory | `github.com/vinary-tree/libdictenstein/bindings/go` | `github.com/vinary-tree/liblevenshtein-rust/bindings/go` |
| Swift | `VinaryTreeInterop` | `Libdictenstein` | `Liblevenshtein` |
| Ruby | shared resource types supplied by each project package | `vinary-tree-libdictenstein` | `vinary-tree-liblevenshtein` |
| Fortran | `vinary-tree-interop` | `vinary-tree-libdictenstein` | `vinary-tree-liblevenshtein` |
| OCaml | `vinary-tree-interop` | `vinary-tree-libdictenstein` | `vinary-tree-liblevenshtein` |
| Haskell | `vinary-tree-interop` | `vinary-tree-libdictenstein` | `vinary-tree-liblevenshtein` |
| Lua | common C ABI headers | `vinary-tree-libdictenstein` | `vinary-tree-liblevenshtein` |

Maven Central is the public Java repository of record. JFrog Artifactory can
proxy those coordinates; it is not a distinct public package coordinate. Java
packages use `io.vinarytree.*` because Java identifiers cannot contain
a hyphen, while Maven groups retain `io.vinarytree`.

## Registry credentials

Protected GitHub environments scope credentials to their publish jobs:

- crates.io uses `CARGO_REGISTRY_TOKEN`;
- PyPI and npm use OIDC trusted publishing;
- Maven Central uses Central Portal credentials and signing keys;
- Clojars uses a scoped deploy token;
- NuGet uses `NUGET_API_KEY` and RubyGems uses `RUBYGEMS_API_KEY`;
- Hackage uses `HACKAGE_USERNAME` and `HACKAGE_PASSWORD`;
- fpm uses `FPM_REGISTRY_TOKEN`, LuaRocks uses `LUAROCKS_API_KEY`, and the
  opam-repository pull-request job uses `OPAM_GITHUB_TOKEN`.

Create namespace ownership for `vinary-tree` before the first release. Keep
interop and project credentials separate so a project release cannot replace
the shared ABI package accidentally.

## Exact release sequence

For the first release of an ABI revision:

1. push `interop-vX.Y.Z` in `liblevenshtein-rust` and wait for every common
   registry job plus the opam pull request;
2. release `libdictenstein`, which owns dictionary CRUD and persistence;
3. release `lling-llang` and `duallity` when their exact dependency versions
   are available;
4. push the liblevenshtein `vX.Y.Z` tag;
5. publish the related npm umbrella, then the project npm facades.

Every checkout action accepts exact tag inputs. No release job builds against a
moving sibling branch.

## Native artifacts

Release archives are built for:

- `x86_64-unknown-linux-gnu`;
- `aarch64-unknown-linux-gnu`;
- `aarch64-apple-darwin`;
- `x86_64-pc-windows-msvc`.

Each archive contains the interop and project headers, shared and static
liblevenshtein libraries, CMake config packages, and `pkg-config` metadata. It
must be relocatable and must not use `target-cpu=native`.

Dynamic CMake consumers use:

```cmake
find_package(vinary-tree-interop 0.1 CONFIG REQUIRED)
find_package(liblevenshtein 0.10 CONFIG REQUIRED)
target_link_libraries(app PRIVATE liblevenshtein::liblevenshtein)
```

Static consumers select `-DLIBLEVENSHTEIN_LINKAGE=STATIC` or link the explicit
`liblevenshtein::static` target. The package propagates platform system
libraries. Other native facades may use the installed shared package or link
statically, but package documentation must say which mode is used; loaders must
not silently select an arbitrary system library.

## Managed artifacts

Python wheels bundle liblevenshtein's native library and depend on the matching
`vinary-tree-interop` Python package for provider/resource types. Wheel tests
construct a custom provider and exercise the long-lived cursor fixture.
Libdictenstein wheels first run producer-only CRUD and persistence tests, then
the liblevenshtein matrix installs both packages and runs the real retained
resource handoff. This ordering lets a producer release precede its consumer
without weakening cross-project conformance.

The JVM artifact targets Java 22 bytecode and uses FFM. It bundles native
libraries for all required release targets and depends exactly on
`io.vinarytree:vinary-tree-interop`. Test on JDK 25 LTS and the current
JDK. Maven source classifiers contain Java sources only; platform libraries
exist solely in the runtime JAR. Clojure is built only after the exact staged
JVM and interop artifacts are installed locally, then published to Clojars.
Both Leiningen and Clojure CLI tests authorize FFM native access explicitly.

The npm project artifact is a typed facade. It pins both
`@vinary-tree/interop` and `@vinary-tree/vinary-tree` and supplies native,
browser-WASM, WASI, TypeScript, and ClojureScript entry points. The payloads and
actual resource table live in the umbrella runtime; facade packages verify the
same runtime identity before accepting a resource. Its Node prebuilds
statically contain all four related Rust components, so Node consumers do not
install a Vinary Tree shared library or configure a loader path. Browser and
WASI artifacts are separate explicit exports; a WASI host must preopen every
persistent-ARTrie directory it wants to expose. Release staging strips profiling
symbols from native prebuilds and debug custom sections from WASI while leaving
the libraries' optimized code unchanged.

## Pre-publication gates

`scripts/check-bindings.py` rejects a release unless:

- generated constants and modeled C symbols are exact;
- publishable liblevenshtein Tier 1 facades contain no dictionary-owned CRUD;
- public coordinates use only the `vinary-tree` organization;
- all language feature aliases remain opt-in;
- FFM, package dependencies, exports, and registry jobs are present;
- the required OS/architecture target triples appear in CI;
- snapshot, batch lease, reducer, and provider-concurrency contracts have
  executable tests.

The binding test suite must also pass:

- direct Rust and resource-ABI example tests;
- randomized insert/remove/update/clear/compact/checkpoint histories against
  one partially consumed cursor;
- the C ABI leased-batch and reducer tests;
- C++ custom-provider and installed-CMake smoke tests;
- Python custom-provider tests;
- JVM FFM, Clojure, TypeScript, and ClojureScript facade tests;
- Babashka reader/namespace contracts for the Clojure facades;
- .NET 8 LTS and .NET 10, Go 1.25 and 1.26, Swift on current macOS ARM,
  Ruby 3.3 through 3.5, GCC 15, AOCC Flang 5.1, and LLVM Flang 22,
  OCaml 5.2 and 5.4,
  GHC 9.6 and 9.14, and Lua 5.4 integration tests;
- required native architecture jobs and best-effort BSD jobs.

Tier 2 and Tier 3 packages are publishable because they consume retained shared
resources rather than exposing the retired liblevenshtein-owned dictionary API,
and their CI lanes run the same query-start snapshot fixture as Tier 1.

## Versioning and rollback

The interop ABI version and project API revision are separate. A project may
release without changing the ABI. An incompatible resource layout or ownership
rule receives a new interface ID/version and can coexist with the old one.

Never move a published Git tag, Go submodule tag, SwiftPM tag, or registry
version. If publication fails after some artifacts are public, fix the pipeline
and release a new patch version. Do not overwrite a public interop package:
other independently released projects may already depend on it.
