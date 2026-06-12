# PathMap Integration Rework — TrieRef Nodes + Zero-Plumbing Entry Points

**Status:** implemented (libdictenstein + liblevenshtein-rust), tests green, 0.3
portability confirmed. Benchmarks (H1–H7) pending a quiet measurement window;
see [`docs/benchmarks/pathmap-trieref-rework.md`](../benchmarks/pathmap-trieref-rework.md).

This document is written so the rework can be reconstructed from scratch.

## 1. Motivation — the PathMap developer's feedback

The PathMap developer reviewed liblevenshtein's PathMap integration:

> 1. *"The pathmap integration is based on PathMap instead of the zippers so
>    it'd require quite a bit of plumbing still to test it out"* — as a fuzzy
>    query engine for MORK.
> 2. *"I believe `PathMapNode` should be replaceable by `TrieRef` btw."*

Two concrete problems followed from this:

- **Ownership friction (plumbing).** MORK holds a bare `PathMap<V>`
  (`Space.btm`) and queries it through read zippers. Our integration demanded
  ownership via `Arc<RwLock<PathMap<V>>>` inside `PathMapDictionary`, so MORK
  would have had to copy its entire trie (or restructure around our lock) just
  to try us.
- **Path-replay pathology.** The old `PathMapNode` stored
  `{ Arc<RwLock<PathMap>>, Arc<Vec<u8>> path }` and, on **every** operation,
  acquired a read lock and called `read_zipper_at_path(path)` — re-walking the
  whole path from the root. Walking a term of length `n` cost `𝒪(n²)` byte-steps
  plus `n` lock round-trips; `edges()` additionally scanned all 256 possible
  child bytes and re-validated each survivor with another lock + replay. This is
  why the PathMap backend benched markedly slower than `DynamicDawg`.

## 2. The primitive — `TrieRef`

`TrieRef` (pathmap ≥ 0.2.2) is a cheap, lock-free, `Clone`/`Send`/`Sync`
value-type handle on a trie **node**:

- `TrieRefOwned<V>` — owns its focus node by refcount (clone = a bump); no
  lifetime parameter.
- `TrieRefBorrowed<'a, V>` — a `Copy` borrow of a node in a live map for `'a`.

Both expose `path_exists`, `is_val`, `child_count`, `child_mask` (via `Zipper`),
`val` (via `ZipperValues`), and `trie_ref_at_path` (via `ZipperReadOnlySubtries`)
which descends **from the focus** in `𝒪(1)` per byte — no root replay, no lock.

### Source-verified facts (pathmap 0.2.2 ⋂ 0.3.0)

- `TrieRefOwned<V>` / `TrieRefBorrowed<'a, V>` are `Send + Sync` for
  `V: Send + Sync` — `trait TrieNode: … + Send + Sync`, so `Arc<dyn TrieNode>`
  (the node refcount) is `Send + Sync`. `DictionaryValue: Clone + Default +
  Send + Sync + Unpin + 'static` satisfies every pathmap `V` bound.
- Portable root construction (identical in 0.2.2 and 0.3.0):
  `map.into_read_zipper(&[]).trie_ref_at_path(&[])` → `TrieRefOwned`
  (`ReadZipperOwned::TrieRefT = TrieRefOwned`); `map.trie_ref_at_path(&[])` →
  `TrieRefBorrowed` directly.
- `ByteMask::iter()` is a word-skipping iterator yielding `u8` (no 256-way scan).
- `trie_ref_at_path` on a non-existent path **never panics**: a dangling
  remainder ≤ 48 bytes (`MAX_NODE_KEY_BYTES`) is stored as a node key with
  `path_exists() == false`; a longer remainder yields an invalid-sentinel ref
  (all ops return false/empty/`None`).
- `PathMap::clone()` is `𝒪(1)` (root refcount bump); all writes go through
  `make_mut`, which copies shared nodes — a snapshot is never observed
  mid-mutation.
- **No `'static`/`Send` obligation anywhere in the `Transducer` /
  `QueryIterator` / `Intersection` stack** — a borrowed dictionary works
  end-to-end. (Proven by `tests/pathmap_snapshot_tests.rs::transducer_over_borrowed_ref_no_static_bound`.)

## 3. Design

### 3.1 The sealed adapter trait (`libdictenstein/src/pathmap/core.rs`)

A small **sealed** trait insulates the rest of the crate from pathmap's lifetime
plumbing and from 0.2/0.3 API drift:

```rust
pub trait TrieRefLike<V>: Clone + Send + Sync + sealed::Sealed {
    fn path_exists(&self) -> bool;
    fn is_val(&self) -> bool;
    fn val_cloned(&self) -> Option<V>;
    fn child_mask(&self) -> ByteMask;
    fn child_count(&self) -> usize;
    fn descend_bytes(&self, bytes: &[u8]) -> Self;   // from FOCUS — no lock, no replay
}
// impls: TrieRefOwned<V>, TrieRefBorrowed<'a, V>
```

Because it is sealed, no downstream crate can implement it, and we own every
point of contact with pathmap's read-only subtrie API (the only externally
non-implementable supertrait, `ZipperReadOnlySubtries`, is consumed only inside
these impls and the `from_read_zipper` constructors).

### 3.2 The nodes (the developer's exact suggestion)

```rust
pub struct TrieRefNode<V, R: TrieRefLike<V> = TrieRefOwned<V>>     { r: R, .. }
pub struct TrieRefNodeChar<V, R: TrieRefLike<V> = TrieRefOwned<V>> { r: R, .. }
```

- `TrieRefNode`: `Unit = u8`. `transition(b)` = `r.descend_bytes(&[b])` then
  `path_exists()`. `edges()` = `child_mask().iter()` → descend each (the mask
  proves existence, so **no re-validation**). `edge_count()` = `child_count()`.
- `TrieRefNodeChar`: `Unit = char`. `transition` encodes UTF-8 then descends.
  `edges()` decodes UTF-8 by walking continuation bytes **locally from the
  focus** (read the focus child mask for lead bytes, then for each multi-byte
  lead descend the partial and read its child mask for `0b10xx_xxxx`
  continuations) — never replaying the path from the root.

`PathMapNode<V>` / `PathMapNodeChar<V>` are now type aliases of
`TrieRefNode<V, TrieRefOwned<V>>` / `TrieRefNodeChar<V, TrieRefOwned<V>>` (fields
were private — no downstream breakage).

### 3.3 Snapshot `root()` + `snapshot()`

`PathMapDictionary` / `PathMapDictionaryChar` keep their mutable API behind
`Arc<RwLock>`, but `root()` now locks **once**, does `map.read().clone()` (an
`𝒪(1)` CoW snapshot) and returns a `TrieRefNode`. Queries then run **lock-free**
over a consistent snapshot. A new `snapshot()` method returns a
`PathMapSnapshot`/`PathMapSnapshotChar`.

**Snapshot-isolation semantics (documented, non-breaking).** A node/zipper binds
to the trie at creation; in-flight traversals no longer observe concurrent
mutations. This *replaces* the old torn-traversal hazard (a fresh lock per
operation over a live, mutating map) with proper snapshot isolation — aligned
with PathMap's persistent CoW model. Audited: no existing test depended on live
visibility (the concurrent test takes fresh roots and joins before asserting).

### 3.4 Zero-plumbing, MORK-facing dictionaries (`pathmap/snapshot.rs`)

| Type | Root handle | Construct | Lifetime |
|···|···|···|···|
| `PathMapSnapshot<V>` | `TrieRefOwned` | `𝒪(1)` CoW bump | owned |
| `PathMapRef<'a, V>` | `TrieRefBorrowed` | zero-copy borrow | `'a` |
| `PathMapSnapshotChar<V>` / `PathMapRefChar<'a, V>` | … | … | … |

Constructors: `from_map`, `from_map_ref` (owned only), `from_trie_ref`,
`from_read_zipper`. MORK usage becomes:

```rust
let dict = PathMapRef::from_map(&space.btm);                            // zero-copy borrow
let dict = PathMapSnapshot::from_map_ref(&space.btm);                   // 𝒪(1) CoW snapshot
let dict = PathMapRef::from_trie_ref(space.btm.trie_ref_at_path(pfx));  // subtrie-scoped
Transducer::new(dict, Algorithm::Standard).query("fooo", 1);
```

See [`examples/mork_fuzzy_query.rs`](../../examples/mork_fuzzy_query.rs).

### 3.5 The zipper (`pathmap/zipper.rs`)

`PathMapZipper<V>` is now an alias of a generic `TrieRefZipper<V, R>` holding the
focus handle `r` plus a `path` buffer (kept only for `DictZipper::path`).
`descend`/`children` are lock-free focus descents (mask `iter()`, owned clones).
A borrowed alias `PathMapZipperRef<'a, V>` and `from_map`/`from_map_ref`/
`from_trie_ref` constructors are added.

## 4. Whole-crate module reorganization (no shims)

At the user's direction, **all** dictionary families were reorganized from flat
top-level modules into directory submodules, and the integration was migrated
without leaving any backward-compatibility shims (an initial shim-based approach
was explicitly rejected — see §6):

```
src/<family>/
  mod.rs        re-exports the family's public types
  ascii.rs      u8/byte base       char.rs     UTF-8 (char)
  u64.rs        (dynamic_dawg only) core(.rs|/) shared substrate
  zipper.rs / char_zipper.rs / u64_zipper.rs
  (pathmap also: core.rs = TrieRef substrate, snapshot.rs)
```

Families: `pathmap`, `dynamic_dawg`, `double_array_trie`, `suffix_automaton`,
`scdawg`, `persistent_artrie` (with `char/ core/ vocab/`). Every intra-crate
reference was rewritten to the real path (`super::sub::` within a family,
`crate::family::sub::` across modules); the six downstream consumers
(`liblevenshtein-rust`, `duallity`, `latex-corrector`, `lling-llang`,
`libgrammstein`, `pgmcp`) were updated to the new paths. No aliases remain.

## 5. Dependency & portability

`pathmap = { version = ">=0.2.2, <0.4", optional = true }` in both crates:
publishable (resolves to 0.2.2 on crates.io today; 0.3.0 is local-only) and
accepts `[patch.crates-io] pathmap = { path = "../PathMap" }` for local 0.3
experiments. The code targets only the verified 0.2.2 ⋂ 0.3.0 API intersection.

**0.3 portability — confirmed.** With the patch applied, both crates compile
against local PathMap 0.3.0 with **0 API errors** (0.3 transitively requires
`gxhash`, which needs AES+SSE2; both crates' `.cargo/config.toml` already set
`-C target-feature=+aes,+sse2` / `target-cpu=native`, so no extra flags are
needed for normal in-crate builds). Patch removed after the check.

## 6. Rejected alternatives

- **Backward-compatibility module-path shims** (`pub use family::sub as
  old_flat_name;`). Initially used to make the reorg transparent to the six
  consumers, but **rejected by the user** — the goal is a clean module layout,
  not a layer that keeps the old flat paths alive. Replaced by updating every
  real reference instead.
- **Keeping the live `Arc<RwLock<PathMap>>` per-operation lock model.** Rejected:
  it is the source of both the plumbing friction and the `𝒪(n²)`-per-walk
  path-replay cost. Snapshot isolation is strictly better-defined.

## 7. Validation

- `cargo test --features pathmap-backend` (libdictenstein): 484 lib + 136
  doctests + integration suites, **0 failures**; `--features persistent-artrie
  --lib`: 1705 pass.
- `tests/pathmap_snapshot_tests.rs` (liblevenshtein-rust): 6 end-to-end
  `Transducer` tests over owned snapshots **and borrowed `PathMapRef`** (the
  no-hidden-`'static` proof), subtrie read-zipper roots, transducer-level
  snapshot isolation, and Unicode — all pass.
- `cargo clippy --features pathmap-backend`: clean (no warnings introduced).
- `cargo run --example mork_fuzzy_query --features pathmap-backend`: runs;
  demonstrates the three entry points + CoW isolation + subtrie scoping.

## 8. Files

**libdictenstein** — `src/pathmap/{mod,ascii,char,zipper,core,snapshot}.rs`
(core.rs + snapshot.rs new; ascii/char/zipper reworked from the former
`pathmap.rs`/`pathmap_char.rs`/`pathmap_zipper.rs`); `src/lib.rs`, `Cargo.toml`,
`CHANGELOG.md`; plus the whole-crate family reorganization.

**liblevenshtein-rust** — `tests/pathmap_snapshot_tests.rs`,
`examples/mork_fuzzy_query.rs`, `benches/pathmap_node_ops_benchmark.rs`,
`docs/design/pathmap-trieref-rework.md`,
`docs/benchmarks/pathmap-trieref-rework.md`, `Cargo.toml`, `CHANGELOG.md`.
