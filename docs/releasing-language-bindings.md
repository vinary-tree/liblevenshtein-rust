# Releasing the Vinary Tree language bindings

This guide defines the release architecture, version policy, publication order,
validation gates, credentials, and recovery procedure for the Vinary Tree
family. It is normative for the `4.0.0-rc.1` release train.

The central rule is **one artifact owner, one repository, one release
workflow**. Project repositories may test an exact dependency, but they never
rebuild or republish an artifact owned by another repository.

![Colored publish-order graph showing the standalone interop ABI, Rust projects, standalone JavaScript runtime, scoped facades, and final unscoped compatibility facade.](diagrams/bindings/registry-topology.svg)

## Terms and invariants

An **artifact owner** is the sole repository permitted to publish a package
coordinate. A **facade** is a language-natural public API which delegates to a
Rust implementation or shared runtime. A **release train** is the set of exact
versions intended to interoperate. A **candidate** is a build that is validated
but deliberately not uploaded. A **dist-tag** is npm's human-readable pointer
to a package version; npm publishes to `latest` unless another tag is supplied,
which is why every RC command explicitly supplies `--tag next`.

The release has five invariants:

1. Every publish-time dependency edge is exact; no branch, range, or workspace
   fallback may enter a registry artifact.
2. `release/version.json` is the version authority in each owning repository.
   Its local `scripts/sync-release-version.py` materializes registry-specific
   spellings.
3. `scripts/check-release-train.py` must accept all seven owners together before
   the first package is uploaded.
4. Hackage and fpm receive buildable numeric `4.0.0` candidates during the RC,
   but those candidates are not published because their package versions cannot
   distinguish the RC from the future final release.
5. `liblevenshtein@latest` remains `2.0.4` throughout the RC. Version-4 bytes
   first enter npm under `next`. Because npm assigns `latest` to the first
   publication of a new package even when that publication uses another tag,
   each new scoped coordinate is reserved by an inert `0.0.0` bootstrap. After
   the real RC passes installed-artifact smoke tests, its scoped `latest`
   pointer is moved from `0.0.0` to `4.0.0-rc.1` and the `bootstrap` tag is
   removed. The legacy unscoped coordinate is never moved during the RC.

Cargo documents why a published crate version is immutable and recommends a
dry run before upload in the [Cargo publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html).
npm's [dist-tag guide](https://docs.npmjs.com/adding-dist-tags-to-packages/)
explains the explicit tag required to avoid changing `latest`.

## Repository and coordinate ownership

| Owner repository | Responsibility | Public coordinates in this train |
|---|---|---|
| `vinary-tree-interop` | Shared retained-resource ABI, headers, and language resource types | `vinary-tree-interop`, `@vinary-tree/interop`, `VinaryTree.Interop`, `io.vinarytree:vinary-tree-interop`, and matching PyPI, Go, Swift, OCaml, Haskell, Fortran, CMake, and pkg-config packages |
| `libdictenstein` | Dictionary construction, mutation, persistence, and collection facades | `libdictenstein` and its project-specific language packages |
| `liblevenshtein-rust` | Levenshtein automata, matching, and query facades | `liblevenshtein` and its project-specific language packages |
| `lling-llang` | Weighted finite-state transducer toolkit and project facade | `lling-llang`, `@vinary-tree/lling-llang`, and native package metadata |
| `duallity` | Bridges between Levenshtein automata and weighted transducers | `duallity`, `@vinary-tree/duallity`, and native package metadata |
| `javascript-runtime` | One native/WASM/WASI runtime and resource table for every JavaScript facade | `@vinary-tree/vinary-tree` |
| `liblevenshtein-npm` | Compatibility ownership of the legacy unscoped npm name | `liblevenshtein` only; it delegates exactly to `@vinary-tree/liblevenshtein` |

The repositories are siblings, not nested packages. In particular,
`vinary-tree-interop` and `javascript-runtime` are not owned by
`liblevenshtein-rust`. The scoped liblevenshtein JavaScript facade remains in
`liblevenshtein-rust`; the unscoped compatibility facade is intentionally a
separate repository.

The compatibility package is a package-name bridge, not an emulation of the
legacy JavaScript API. Version 4 consumers use the new Rust-backed facade API.
Its ESM, CommonJS, TypeScript, ClojureScript, browser-WASM, and WASI exports are
thin re-exports of the exact scoped package.

## Two-phase, fail-closed workflow protocol

A **validation dispatch** builds and tests an immutable tag, stages every
candidate artifact, and creates or refreshes the checksummed GitHub
prerelease. A **registry dispatch** repeats those gates but permits exactly one
named publication job to enter its protected environment. These are separate
operations because a source tag is evidence to validate, not authorization to
mutate every configured registry.

Every manual release must run against the tag itself. A branch dispatch fails
in the contract job, even when its files happen to declare the expected
version. The `registry` choice is also fail-closed: `validate-only` enables no
registry uploader, while each other value enables one and only one uploader.

| Owner | Workflow | npm coordinate |
|---|---|---|
| `vinary-tree-interop` | `release.yml` | `@vinary-tree/interop` |
| `libdictenstein` | `release-bindings.yml` | `@vinary-tree/libdictenstein` |
| `liblevenshtein-rust` | `release.yml` | `@vinary-tree/liblevenshtein` |
| `lling-llang` | `release-bindings.yml` | `@vinary-tree/lling-llang` |
| `duallity` | `release-bindings.yml` | `@vinary-tree/duallity` |
| `javascript-runtime` | `release.yml` | `@vinary-tree/vinary-tree` |
| `liblevenshtein-npm` | `release.yml` | legacy `liblevenshtein` |

For example, validate one owner and then publish only its npm artifact:

```bash
gh workflow run release.yml \
  --repo vinary-tree/liblevenshtein-rust \
  --ref v4.0.0-rc.1 \
  -f registry=validate-only

gh workflow run release.yml \
  --repo vinary-tree/liblevenshtein-rust \
  --ref v4.0.0-rc.1 \
  -f registry=npm
```

Repositories whose workflow is named `release-bindings.yml` use that filename
in the same command. The protected `npm` environment supplies the human review
boundary; npm's trusted publisher supplies short-lived OpenID Connect (OIDC)
credentials and provenance. Neither operation uses the workstation's
long-lived npm login.

The workflow state transition is:

```text
immutable tag
  -> validate-only gates
  -> checksummed GitHub prerelease
  -> approve one protected registry environment
  -> publish one staged coordinate
  -> resolve and smoke-test the public bytes
```

If any transition fails, keep the immutable tag and version unchanged, repair
the workflow on the next release-candidate commit, and mint the next candidate.
Never widen a failed run to an `all registries` operation.

![Sequence diagram showing the release operator, immutable source, validation graph, protected environment, package registry, fresh consumer, and evidence store from tag creation through public-byte proof.](diagrams/bindings/release-operator-flow.svg)

The protected environment is an authorization boundary, not a test substitute:
the test graph has already accepted the bytes before a human permits one
external mutation. The fresh consumer then proves what the registry serves,
not what the build workspace happened to contain.

## Operator command protocol

This section is the executable command sheet. Replace values in uppercase; do
not copy a run identifier or environment identifier from an earlier release.

### 1. Establish and record the immutable source

Run the repository's sync and validation commands, inspect the diff, and commit
every intended change before creating the tag. A release tag is annotated so it
records an explicit release event rather than merely naming a commit.

```bash
RELEASE_VERSION="4.0.0-rc.1"
RELEASE_TAG="v${RELEASE_VERSION}"

python3 scripts/sync-release-version.py
git diff --check
git status --short
git tag -a "${RELEASE_TAG}" -m "Release ${RELEASE_TAG}"
git push origin "refs/tags/${RELEASE_TAG}"
```

Do not force-move a release tag after it is pushed. If the tag graph exposes a
defect, correct the source and mint the next candidate. Before the first tag,
the same commit must already have passed ordinary branch CI and the local
release gates; a tag push is the immutable replay, not the first experiment.

### 2. Observe validation rather than assuming it

Tag pushes stage artifacts and a checksummed GitHub prerelease but authorize no
package registry. Record the run URL and wait for its conclusion:

```bash
OWNER_REPOSITORY="vinary-tree/libdictenstein"
WORKFLOW_FILE="release-bindings.yml"

gh run list \
  --repo "${OWNER_REPOSITORY}" \
  --workflow "${WORKFLOW_FILE}" \
  --limit 10

gh run watch RUN_ID --repo "${OWNER_REPOSITORY}" --exit-status
```

Use `release.yml` for interop, liblevenshtein-rust, javascript-runtime, and
liblevenshtein-npm; use `release-bindings.yml` for libdictenstein,
lling-llang, and duallity. A safe replay uses `registry=validate-only` against
the tag itself.

### 3. Dispatch exactly one registry

The registry input is the capability selector. It must name one registry and
the ref must be the immutable tag:

```bash
gh workflow run "${WORKFLOW_FILE}" \
  --repo "${OWNER_REPOSITORY}" \
  --ref "${RELEASE_TAG}" \
  -f registry=crates-io

PUBLISH_RUN_ID="RUN_ID_RETURNED_BY_GITHUB"
gh run view "${PUBLISH_RUN_ID}" --repo "${OWNER_REPOSITORY}"
```

The build and package jobs rerun before the protected publication job. Never
approve the environment merely because an older validation run was green;
confirm that the current run is at the expected tag and commit.

### 4. Review a pending protected environment

GitHub exposes the exact environment awaiting approval. Read it first, then
approve only that environment for that run:

```bash
gh api \
  "repos/${OWNER_REPOSITORY}/actions/runs/${PUBLISH_RUN_ID}/pending_deployments"

ENVIRONMENT_ID="ID_FROM_THE_RESPONSE"
gh api --method POST \
  "repos/${OWNER_REPOSITORY}/actions/runs/${PUBLISH_RUN_ID}/pending_deployments" \
  -F "environment_ids[]=${ENVIRONMENT_ID}" \
  -f state=approved \
  -f comment="Approved after exact-tag validation and artifact review."

gh run watch "${PUBLISH_RUN_ID}" \
  --repo "${OWNER_REPOSITORY}" \
  --exit-status
```

The review API is documented by GitHub's
[workflow-run deployments reference](https://docs.github.com/en/rest/actions/workflow-runs#review-pending-deployments-for-a-workflow-run).
The approval does not provide a reusable registry secret: npm and PyPI obtain
short-lived OpenID Connect (OIDC) credentials from their configured trusted
publishers.

### 5. Prove the public bytes before releasing a dependent

A successful upload step is necessary but insufficient. Wait for registry
indexing, resolve the exact coordinate, verify registry metadata and digest,
and install it into a fresh consumer. Record the command output and the GitHub
run URL in the release ledger.

For a Rust crate:

```bash
CRATE_NAME="libdictenstein"
cargo info "${CRATE_NAME}@${RELEASE_VERSION}"
```

Then create a temporary consumer whose manifest pins
`=${RELEASE_VERSION}`, run `cargo check --locked`, and remove the temporary
directory. Downstream publication starts only after that registry-shaped
consumer passes.

For an npm package:

```bash
NPM_PACKAGE="@vinary-tree/libdictenstein"
npm view "${NPM_PACKAGE}@${RELEASE_VERSION}" \
  version dist.integrity dist.shasum --json

SMOKE_DIRECTORY="$(mktemp -d /tmp/vinary-tree-npm-smoke.XXXXXX)"
trap 'rm -rf -- "${SMOKE_DIRECTORY}"' EXIT
npm install --prefix "${SMOKE_DIRECTORY}" \
  "${NPM_PACKAGE}@${RELEASE_VERSION}"
(
  cd "${SMOKE_DIRECTORY}"
  node -e "require('${NPM_PACKAGE}')"
  node --input-type=module -e "await import('${NPM_PACKAGE}')"
)
```

The smoke program must import the installed package—not a repository-relative
path—and exercise its public construction, query or traversal, iteration,
snapshot where applicable, and deterministic-close contracts. Remove the exact
temporary directory after its evidence is captured.

### 6. Normalize a newly scoped npm coordinate

npm may attach `latest` to a package's first published version even when the
publication command supplied another tag. Each Vinary Tree scoped name was
therefore reserved with inert `0.0.0` bytes. Only after the RC's installed-byte
smoke passes, use an interactive web-authenticated session to replace that
bootstrap default:

```bash
npm login --auth-type=web
npm whoami

NPM_PACKAGE="@vinary-tree/libdictenstein"
npm dist-tag add "${NPM_PACKAGE}@${RELEASE_VERSION}" latest --auth-type=web
npm dist-tag rm "${NPM_PACKAGE}" bootstrap --auth-type=web
npm deprecate "${NPM_PACKAGE}@0.0.0" \
  "Bootstrap-only placeholder; use ${NPM_PACKAGE}@${RELEASE_VERSION} or newer." \
  --auth-type=web

npm view "${NPM_PACKAGE}" versions dist-tags --json
npm view "${NPM_PACKAGE}@0.0.0" deprecated --json
```

Repeat this read-modify-read protocol for `@vinary-tree/interop`,
`@vinary-tree/vinary-tree`, `@vinary-tree/libdictenstein`,
`@vinary-tree/liblevenshtein`, `@vinary-tree/lling-llang`, and
`@vinary-tree/duallity`. The postcondition is
`latest = next = 4.0.0-rc.1`, no `bootstrap` tag, and an explicit deprecation
message on `0.0.0`.

Do not create or store a token that bypasses two-factor authentication (2FA)
for this task. GitHub Actions uses npm
[trusted publishing](https://docs.npmjs.com/trusted-publishers/) and
[package provenance](https://docs.npmjs.com/generating-provenance-statements/);
the local web login is limited to post-publication metadata changes that npm
does not yet authorize through the trusted-publisher workflow.

The legacy unscoped package is deliberately different:

```bash
npm view liblevenshtein dist-tags --json
```

During the RC, it must report `latest = 2.0.4` and
`next = 4.0.0-rc.1`. Never apply the scoped-package `latest` command to this
coordinate.

## Version function

Let `M`, `m`, and `p` denote the major, minor, and patch components, and let `r`
denote the release-candidate ordinal. For this train, `$`M = 4`$`, `$`m = 0`$`,
`$`p = 0`$`, and `$`r = 1`$`. Define the canonical version `$`v`$` and numeric
base `$`b`$` as follows:

```math
v = M.m.p\text{-rc}.r = \text{4.0.0-rc.1}, \qquad b = M.m.p = \text{4.0.0}.
```

Each registry renderer `$`R_e`$` maps `$`v`$` into the syntax accepted by
ecosystem `$`e`$`:

```math
R_e(v) =
\begin{cases}
\text{4.0.0-rc.1} & e \in \{\text{Cargo,npm,Maven,Clojars,NuGet,Swift,CMake,C++}\},\\
\text{4.0.0rc1} & e = \text{PyPI},\\
\text{4.0.0rc1-1} & e = \text{LuaRocks},\\
\text{4.0.0.rc.1} & e = \text{RubyGems},\\
\text{4.0.0\textasciitilde rc1} & e = \text{opam},\\
\text{v4.0.0-rc.1} & e = \text{Go tag},\\
\text{4.0.0} & e \in \{\text{Hackage,fpm candidates}\}.
\end{cases}
```

| Ecosystem | RC spelling | Publication policy |
|---|---|---|
| Cargo, npm, Maven, Clojars, NuGet, Swift, CMake, pkg-config | `4.0.0-rc.1` | Publish after its owner passes all gates |
| PyPI | `4.0.0rc1` | Publish after wheel tests |
| RubyGems | `4.0.0.rc.1` | Publish after native-resource inspection |
| opam | `4.0.0~rc1` | Submit an opam-repository pull request |
| Go | module path ending in `/v4`; tag `v4.0.0-rc.1` | Create the immutable subdirectory tag after dependencies resolve |
| LuaRocks | `4.0.0rc1-1` | Publish linted rockspec metadata; rockspec format 1.0 permits one hyphen only, before the numeric revision |
| Hackage | `4.0.0` with `x-release-candidate: rc.1` | Build candidate only; do not upload |
| fpm | `4.0.0` | Build candidate only; do not upload |

`llattice` is intentionally outside the synchronized major-version train and
remains pinned to `0.1.0`.

## Release graph and order

Source tags for every repository may be prepared before registry publication,
because cross-project conformance tests consume exact source tags. Registry
publication still follows dependency order:

1. Publish `vinary-tree-interop` from its standalone repository.
2. Publish `libdictenstein` after the interop coordinates resolve.
3. Publish `liblevenshtein` after libdictenstein and interop resolve and the
   retained-resource handoff fixture passes.
4. Publish `lling-llang`, then `duallity`, because their optional integrations
   pin the preceding Rust crates exactly.
5. Publish `@vinary-tree/vinary-tree` from `javascript-runtime` after all four
   Rust components are available.
6. Publish the four scoped npm project facades with `--tag next`.
7. Publish `liblevenshtein@4.0.0-rc.1` from `liblevenshtein-npm`, also with
   `--tag next`, after `@vinary-tree/liblevenshtein@4.0.0-rc.1` resolves.

The libdictenstein workflow uses liblevenshtein source for a cross-project
consumer test. That is a **validation dependency**, not a reason to invert the
registry order: dictionary packages remain independently usable and are
published before the matching consumer.

### Literate release algorithm

The algorithm first proves that every coordinate describes the same train,
then publishes one owner at a time. The registry observation after each upload
is part of the algorithm: downstream publication never relies on an upload
command merely returning success.

```text
procedure RELEASE_4_RC_1(owners):
    # Establish one immutable source state for every owner.
    for owner in owners:
        require owner.worktree_is_clean
        owner.sync_release_version(check = true)
        owner.run_project_gates()

    require check_release_train(owners)
    require npm_dist_tag("liblevenshtein", "latest") = "2.0.4"

    for owner in owners:
        owner.dispatch(tag = "v4.0.0-rc.1", registry = "validate-only")
        require owner.github_prerelease.has_valid_checksums

    for owner in [interop, libdictenstein, liblevenshtein,
                  lling_llang, duallity, javascript_runtime,
                  scoped_facades, legacy_npm_facade]:
        owner.dispatch(tag = "v4.0.0-rc.1", registry = owner.next_registry)
        wait_until_every_published_coordinate_resolves(owner)
        rerun_owner_smoke_tests_against_registry_bytes(owner)

    for package in new_scoped_npm_coordinates:
        require npm_dist_tag(package, "next") = "4.0.0-rc.1"
        npm_set_dist_tag(package, "latest", "4.0.0-rc.1")
        npm_remove_dist_tag(package, "bootstrap")
        require npm_dist_tag(package, "latest") = "4.0.0-rc.1"

    require npm_dist_tag("liblevenshtein", "latest") = "2.0.4"
    record_checksums_and_release_evidence()
```

The scoped retarget above replaces npm's unusable bootstrap default only after
the RC is proven. It is not the legacy package's final-release promotion:
moving unscoped `liblevenshtein@latest` remains a separate final-release
decision.

## Preparing a release checkout

Place the repositories as siblings. The local checker accepts environment
overrides, which is useful when lling-llang or duallity must remain isolated in
release worktrees:

```bash
export VINARY_TREE_INTEROP_ROOT=../vinary-tree-interop
export VINARY_TREE_JAVASCRIPT_RUNTIME_ROOT=../javascript-runtime
export LIBDICTENSTEIN_ROOT=../libdictenstein
export LLING_LLANG_ROOT=../lling-llang
export DUALLITY_ROOT=../duallity
export LIBLEVENSHTEIN_NPM_ROOT=../liblevenshtein-npm

python3 scripts/check-release-train.py
```

Keep Cargo dependencies declared with both `path` and an exact `version` in
the reviewed source. Cargo's supported multiple-location dependency form uses
the path locally and removes it from the normalized registry manifest. Do not
rewrite `Cargo.toml` or `Cargo.lock` inside the publication job: doing so makes
the checkout dirty and causes `cargo publish` to reject the release unless an
unsafe override is added. Instead, check out exact sibling tags where local
path discovery is required and publish the unchanged source:

```bash
cargo publish --dry-run --locked
git diff --exit-code -- Cargo.toml Cargo.lock
cargo publish --locked
```

Cargo documents this normalization under
[multiple dependency locations](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#multiple-locations).
The public versions on the registry remain prerequisites for package
verification and downstream consumers; exact source checkouts do not replace
the dependency-order gate. Python wheel isolation, Gradle staging, and npm
packing provide equivalent registry-shaped checks for their ecosystems.

For every owning repository:

```bash
python3 scripts/sync-release-version.py
git diff --exit-code
```

The first command is deliberately idempotent. A diff after the second
invocation means either a coordinate escaped the model or the generator is not
stable; both are release blockers.

## Artifact-specific gates

### Rust and native C/C++

Run locked tests, all-feature tests, Clippy, documentation, and package dry
runs. Platform feature minimums required by dependencies such as Gxhash belong
in target-scoped `.cargo/config.toml` entries. Workflow-level `RUSTFLAGS` must
not override that matrix, and release binaries must not embed
`target-cpu=native`. Sanitizer jobs are the exception: their environment flags
must explicitly carry both the sanitizer instrumentation and the target's
portable baseline.

Native archives contain only the project that owns them. They depend on the
separately installed `vinary-tree-interop` CMake/pkg-config package:

```cmake
find_package(vinary-tree-interop 4.0 CONFIG REQUIRED)
find_package(liblevenshtein 4.0 CONFIG REQUIRED)
target_link_libraries(app PRIVATE liblevenshtein::liblevenshtein)
```

Required native targets are Linux x86-64, Linux ARM64, macOS x86-64, macOS
ARM64, Windows x86-64, and Windows ARM64 where the owning artifact supports a
native payload. Each archive is tested after relocation with both shared and
static linkage when offered.

### Python, JVM, Clojure, .NET, and Ruby

Python wheels use the PyPI spelling `4.0.0rc1`, bundle only their project native
library, and depend exactly on `vinary-tree-interop==4.0.0rc1`.

The JVM artifact targets the documented Java level, uses Foreign Function &
Memory resource lifetimes, bundles its project native resources, and depends
on `io.vinarytree:vinary-tree-interop:4.0.0-rc.1`. The Clojure facade is staged
only after the exact JVM artifact has been installed into the test repository.
Maven Central and Clojars are separate coordinates and separate credentials.

.NET packages depend on `VinaryTree.Interop` rather than embedding its source.
Ruby packages inspect every platform payload before `gem push`. Every managed
facade must pass its collection-idiom, resource-lifetime, snapshot-consistency,
property, and leak tests before packaging.

### JavaScript, TypeScript, and ClojureScript

`javascript-runtime` owns six native prebuilds:

- `linux-x64` and `linux-arm64`;
- `darwin-x64` and `darwin-arm64`;
- `win32-x64` and `win32-arm64`.

It also owns browser-WASM and WASI payloads. Project facades contain no native
payload and pin both `@vinary-tree/interop` and
`@vinary-tree/vinary-tree` exactly. Runtime identity checks prevent resources
from crossing different runtime instances.

The scoped facade publish command is intentionally explicit:

```bash
npm publish --access public --provenance --tag next ./dist/*.tgz
```

For every newly created scoped coordinate, verify the installed RC and then
replace npm's mandatory first-publication default and retire the bootstrap tag:

Use the interactive, read-modify-read procedure in
[Operator command protocol §6](#6-normalize-a-newly-scoped-npm-coordinate).
It deliberately includes `--auth-type=web` and verifies the tags and
deprecation after mutation.

The scoped postcondition is `latest = next = 4.0.0-rc.1`, with no `bootstrap`
tag. The immutable `0.0.0` audit artifact remains explicitly deprecated.

Before and after publishing the legacy name, verify its stable tag:

```bash
npm view liblevenshtein dist-tags --json
npm view liblevenshtein@4.0.0-rc.1 version
```

The required postcondition is `latest = 2.0.4` and `next = 4.0.0-rc.1`.

### Haskell and Fortran candidates

Hackage and fpm cannot safely publish this RC under `4.0.0`, because that
numeric coordinate is reserved for the final release. Their workflow lanes
still build source distributions, validate manifests, and archive candidates
as GitHub release evidence. They contain no registry credentials and no upload
step. The final `4.0.0` train may publish those already-tested shapes after
regenerating checksums from the final source commit.

## Registry read-back matrix

The public-byte gate is language-specific but follows one invariant: resolve
the exact public coordinate into a clean consumer and exercise the installed
API. A metadata listing alone does not prove that the package installs or that
its native payload loads.

| Registry | Resolution proof | Minimum fresh-consumer proof |
|---|---|---|
| crates.io | `cargo info NAME@4.0.0-rc.1` | Temporary crate with `NAME = "=4.0.0-rc.1"`; `cargo check --locked`; run a construction/query smoke where the crate exposes behavior |
| npm | `npm view NAME@4.0.0-rc.1 version dist.integrity dist.shasum --json` | Install the exact version into `mktemp -d`; test CommonJS and ESM entry points, runtime identity, iteration, and deterministic closure |
| PyPI | `python -m pip download --no-deps NAME==4.0.0rc1` | New virtual environment; install only downloaded wheels and exact dependencies; import, construct, iterate, snapshot, close |
| Maven Central | Resolve `GROUP:ARTIFACT:4.0.0-rc.1` from Central in an empty Gradle/Maven cache | Compile and run the Java collection and try-with-resources fixtures against the resolved JAR and extracted native library |
| Clojars | Resolve `[GROUP/ARTIFACT "4.0.0-rc.1"]` from Clojars | Run the idiomatic Clojure collection, snapshot, and resource-lifetime fixtures without a local Maven override |
| NuGet | Query the exact package version from nuget.org | Empty `dotnet new` project; add exact package; run collection, enumeration, snapshot, and `IDisposable` fixtures |
| RubyGems | `gem fetch NAME -v 4.0.0.rc.1` | Install to an isolated gem home; require the gem, traverse data, and close native resources |
| Go proxy | `go list -m MODULE@v4.0.0-rc.1` | New module with the exact `/v4` requirement; `go test` the ownership and iteration fixture |
| LuaRocks | Inspect/download `NAME 4.0.0rc1-1` from the configured server | Isolated tree; load the module and run resource and traversal fixtures |
| opam | Inspect the submitted `opam-repository` pull request and source checksum | Fresh switch; pin the candidate metadata, build, and execute its examples before merge |
| GitHub release | Verify `SHA256SUMS` against every downloaded asset | Relocate each native SDK archive and build both shared and static sample consumers where supported |

Hackage and fpm are absent from the upload rows for this RC: their numeric-only
`4.0.0` candidates are build evidence, not public registry versions. Store the
read-back commands, resolved digests, smoke outcomes, and run URLs in the
versioned release ledger.

## Credentials and protected environments

| Destination | Authentication | Recommended GitHub environment |
|---|---|---|
| crates.io | `CARGO_REGISTRY_TOKEN` | `crates-io` |
| PyPI | OIDC trusted publishing | `pypi` |
| npm | OIDC trusted publishing with provenance; account and organization require 2FA | `npm` |
| Maven Central | Central Portal credentials plus GPG public key, private key, and passphrase | `maven-central` |
| Clojars | username and scoped deploy token | `clojars` |
| NuGet | `NUGET_API_KEY` | `nuget` |
| RubyGems | `RUBYGEMS_API_KEY` | `rubygems` |
| LuaRocks | `LUAROCKS_API_KEY` | `luarocks` |
| opam repository | GitHub token authorized to push a fork and open a pull request | `opam` |

Hackage and fpm secrets are intentionally unnecessary for this RC. Apply
required-reviewer protection to every publish environment. OIDC jobs receive
`id-token: write` only in the individual job that uploads that owner's
artifact; build jobs remain read-only.

For npm, the publisher must have access to the `vinary-tree` organization and
the legacy `liblevenshtein` package. Register each repository/workflow as a
trusted publisher before pushing its tag. A local `npm login` is useful for
manual inspection but is not a substitute for CI trusted-publisher setup.

## Pre-publication checklist

- [ ] Every worktree intended for release is clean and committed.
- [ ] Every `release/version.json` says `4.0.0-rc.1` and every sync script is
      idempotent.
- [ ] `python3 scripts/check-release-train.py` passes across the seven owners.
- [ ] Generated APIs, binding documentation, ABI invariants, and completeness
      matrices are current.
- [ ] Locked Rust tests pass with default and all features; Gxhash target flags
      are supplied by platform-specific Cargo configuration.
- [ ] Native archives are relocation-tested and contain no foreign owner's
      package.
- [ ] All managed-language and JavaScript package tests pass against assembled
      artifacts.
- [ ] All workflow YAML parses and all package dry runs succeed.
- [ ] A `validate-only` dispatch at each immutable tag completes before any
      registry dispatch.
- [ ] Every registry dispatch names exactly one target and runs against
      `refs/tags/v4.0.0-rc.1`, never a branch.
- [ ] Registry namespaces, trusted publishers, signing keys, and protected
      environments exist.
- [ ] Every new scoped npm package has `latest = next = 4.0.0-rc.1`, has no
      `bootstrap` tag, and deprecates its immutable `0.0.0` reservation.
- [ ] `npm view liblevenshtein dist-tags --json` reports `latest: 2.0.4`.
- [ ] Hackage and fpm jobs are visibly candidate-only.

## Failure, rollback, and evidence

Registry versions and Git tags are immutable. If an upload succeeds in only
part of the graph, stop immediately, record the coordinates that became
public, and resume only after those exact bytes resolve. If published bytes are
wrong, release a new candidate such as `4.0.0-rc.2`; never overwrite, retag, or
silently rebuild `rc.1`.

If a facade is broken but its underlying runtime is correct, fix and republish
only that facade under a new candidate. If the shared ABI is wrong, advance the
entire dependent train. This is the operational benefit of single ownership:
the affected cut is explicit rather than hidden inside duplicated bundles.

Each GitHub release retains:

- source and native archives;
- package candidates and registry staging bundles;
- `SHA256SUMS` over every attached artifact;
- workflow run links and test summaries;
- the exact `release/version.json` used by its owner.

Publication itself is never a test. The release is complete only when a clean
consumer can resolve each public coordinate, run a representative query, close
its resource normally, and reproduce the expected checksum evidence.
