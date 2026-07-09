# Branch Archive: Retired Experiment Branches

**Created**: 2026-07-09
**Purpose**: Resolve every branch name cited in this repository's scientific
ledgers to a durable, permanent Git reference.

---

## Why this document exists

The scientific ledgers in `docs/research/` and `docs/optimization/` cite the
branch on which each experiment was performed — this is what makes a recorded
result reproducible rather than merely asserted. Branch names, however, are
mutable local pointers: once a branch is deleted, its name no longer resolves,
and a citation such as `**Branch**: feat/wallbreaker-freq-split` becomes a
dangling reference.

On 2026-07-09 the repository's 19 stray feature branches were retired. This
table preserves the mapping from every cited branch name to the immutable
commit SHA it pointed at, together with the durable reference through which
that commit remains reachable. **No commit referenced by any ledger was lost.**

---

## Resolution table

A commit is *reachable* — and therefore permanently safe from garbage
collection — if it is an ancestor of `master`, or if an annotated tag points at
it. Every retired branch satisfies one of these two conditions.

| Retired branch | Commit | Reachable via | Content status |
|---|---|---|---|
| `feat/wallbreaker-freq-split` | `3d372ef` | tag `experiment/wallbreaker-freq-split` | **Unique code**, never merged |
| `feat/wallbreaker-substring-opt` | `4a62248` | tag `experiment/wallbreaker-substring-opt` | **Unique docs**, superseded |
| `perf/phonetic-normalized-opt-5` | `9a57cca` | tag `archive/phonetic-normalized-opt-5` | Patch-equivalent to `master` |
| `release-rebuild-v0.9.0` | `66de57c` | tag `archive/release-rebuild-v0.9.0` | Patch-equivalent to `master` |
| `feat/wallbreaker-benchmarks` | `7543b75` | ancestor of `master` | Merged |
| `feat/wallbreaker-simd` | `5acf623` | ancestor of `master` | Merged |
| `dylon/ci-sibling-fix` | `1968089` | ancestor of `master` | Merged |
| `fix-nodup-definition` | `198b525` | ancestor of `master` | Merged |
| `opt/baseline` | `845a288` | ancestor of `master` | Merged |
| `opt/llev-h1-intern-class-names` | `1ec7f71` | ancestor of `master` | Merged |
| `opt/llev-h2-named-class-lookup` | `5080be7` | ancestor of `master` | Merged |
| `opt/llev-h4-smallvec-charclass` | `3db880e` | ancestor of `master` | Merged |
| `perf/phonetic-normalized-benchmarks` | `dd1438c` | ancestor of `master` | Merged |
| `perf/phonetic-normalized-opt-1` | `62db70d` | ancestor of `master` | Merged |
| `perf/phonetic-normalized-opt-2` | `ea54db5` | ancestor of `master` | Merged |
| `perf/phonetic-normalized-opt-3` | `50ceebb` | ancestor of `master` | Merged |
| `perf/phonetic-normalized-opt-4` | `3418d30` | ancestor of `master` | Merged |
| `perf/phonetic-normalized-opt-6-bktree` | `316f874` | ancestor of `master` | Merged |
| `proof-multirule-axiom` | `9fa2fc9` | ancestor of `master` | Merged |

Surviving branches: `master` and `release` (the latter carries one intentional
commit, `32c9613`, disabling PathMap in `Cargo.toml`).

---

## The two tags that carry unique content

Only two retired branches held anything absent from `master`. Both are now
annotated tags whose messages record the full hypothesis, verdict, and
measurements; read them with `git show <tag>`.

### `experiment/wallbreaker-freq-split`

The sole surviving copy of `FrequencyPatternSplitter` and
`FrequencyWallBreaker` (≈786 insertions across
`src/wallbreaker/pattern_splitter.rs`, `src/wallbreaker/mod.rs`, and
`benches/wallbreaker_benchmarks.rs`). It is the evidence behind **Experiment 3**
of the [WallBreaker scientific ledger](wallbreaker/scientific-ledger.md), whose
verdict was `❌ REJECTED (Overall regression)`.

```bash
git show experiment/wallbreaker-freq-split                              # hypothesis, results, analysis
git show experiment/wallbreaker-freq-split:src/wallbreaker/pattern_splitter.rs
git log --oneline master..experiment/wallbreaker-freq-split             # the two unmerged commits
```

This code must not be deleted: it is the reproducible basis of a documented
negative result. It forked 170 commits behind `master` and predates the
migration of the dictionary module to the `libdictenstein` crate (`cd82727`),
so any attempt to rebase it onto `master` will conflict in `src/dictionary/`.

### `experiment/wallbreaker-substring-opt`

Documents the Phase 2 suffix-link substring-search optimization as rejected on
the grounds that the then-current SCDAWG was a prefix DAWG rather than a true
suffix automaton. **This conclusion was subsequently overturned**: `master`
implemented precisely that conversion in **Experiment 5**, "True SCDAWG
Implementation (`𝒪(∣pattern∣)` Substring Search)". The tag is retained as the
historical record of the intermediate negative result.

Its source edits touch `src/dictionary/scdawg.rs` and
`src/dictionary/scdawg_char.rs`, neither of which exists on `master` any longer
(migrated to `libdictenstein` in `cd82727`). They must not be cherry-picked.

---

## Recovering a retired branch

Tags are immutable pointers; recreating a working branch from one is a single
command.

```bash
git switch -c feat/wallbreaker-freq-split experiment/wallbreaker-freq-split
```

To confirm nothing has been orphaned:

```bash
git fsck --unreachable --no-reflogs | grep commit    # expect no output
```
