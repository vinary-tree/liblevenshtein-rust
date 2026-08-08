# Dictionary Thread Safety

Every dictionary backend in liblevenshtein (via `libdictenstein`) is `Send + Sync`
and cheap to share across threads — a clone is an `Arc` bump, not a deep copy. What
differs between backends is **how concurrent reads and writes are coordinated**: every
dictionary read is lock-free or wait-free (a reader never blocks), and writers publish new
state by an atomic pointer swap or CAS rather than by excluding readers.

![Concurrency model: every dictionary backend grouped by its lock-free read mechanism — immutable arrays (DoubleArrayTrie, wait-free), ArcSwap RCU with lock-free CAS writes (the in-memory dynamic DAWG and automaton backends DynamicDawg/DynamicDawgU64/SuffixAutomaton/Scdawg, plus PathMapDictionary and BijectiveMap), persistent copy-on-write snapshots (PathMapSnapshot/PathMapRef), and the disk-persisted CAS/ArcSwap overlay family (the Persistent* backends).](../diagrams/concurrency/concurrency-model.svg)

*Concurrency model — dictionary reads grouped by lock-free read mechanism.*

## The `SyncStrategy` contract

Each backend reports a `SyncStrategy` from `dictionary.sync_strategy()`, telling the
caller what coordination it provides:

| `SyncStrategy` | Meaning |
|---|---|
| `Persistent` | An immutable / structural-sharing snapshot. **Reads need no synchronization;** writes (if any) publish a new version atomically. |
| `InternalSync` | The backend is internally synchronized for concurrent access — atomic operations or lock-free structures. **Reads are lock-free.** |
| `ExternalSync` | The backend uses interior mutability guarded by an internal lock (a `parking_lot::RwLock`). **Reads take a shared lock** and block only against an active writer. |

> The enum is a coarse hint, not a precise read-cost model. For example
> `DoubleArrayTrie` reports `ExternalSync` (the trait default) yet is immutable after
> build, so its reads never block. The grouping below is by **actual read behaviour**.

## Per-backend concurrency model

| Backend | Read path | Reads block? | Writes | Kind |
|---|---|:---:|---|---|
| **`DoubleArrayTrie(Char)`** | immutable `base`/`check` arrays | No | build-time only (`&mut self`) | static |
| **`DynamicDawgU64`** | immutable root revision retained by `Arc` | No | path-copy + root `compare_exchange` (`&self`) | dynamic |
| **`PathMapSnapshot(Char)` / `PathMapRef(Char)`** | persistent copy-on-write snapshot | No | — (immutable view) | static snapshot |
| **`PersistentARTrie(Char / U64)`** | lock-free CAS overlay over a memory-mapped trie | No | lock-free CAS (`&self`) | dynamic · disk |
| **`PersistentScdawg(Char)`** | `ArcSwap` graph load | No | `ArcSwap` publish (`&self`) | dynamic · disk |
| **`PersistentSuffixAutomaton(Char)`** | `ArcSwap` graph load | No | `ArcSwap` publish (`&self`) | dynamic · disk |
| **`PersistentSuffixTree(Char)`** | `ArcSwap` graph load | No | `ArcSwap` publish (`&self`) | dynamic · disk |
| **`PersistentVocabARTrie`** | lock-free overlay | No | lock-free overlay (`&self`) | dynamic · disk |
| **`DynamicDawg(Char)`** | immutable root revision retained by `Arc` (`LockFreeDawg` core) | No | path-copy + root `compare_exchange` (`&self`) | dynamic |
| **`SuffixAutomaton(Char)`** | `Arc<ArcSwap<…>>` load (`LockFreeSuffixAutomaton`) | No | `ArcSwap` publish (`&self`) | dynamic |
| **`Scdawg(Char)`** | `Arc<ArcSwap<…>>` load (`LockFreeScdawg`) | No | `ArcSwap` publish (`&self`) | dynamic |
| **`PathMapDictionary(Char)`** | `Arc<ArcSwap<PathMapState>>` load, then lock-free traversal | No | `ArcSwap` publish (`&self`) | dynamic |
| **`BijectiveMap`** | forward `DynamicDawgChar` (lock-free) + reverse `Arc<ArcSwap<HashMap>>` | No | `ArcSwap` publish (`&self`) | dynamic |

### Every backend reads lock-free

A reader never blocks on any dictionary backend. Every in-memory backend loads its state
through an `arc-swap` guard (or reads immutable arrays), and writers publish new state by
an atomic pointer swap or CAS rather than by excluding readers. Grouped by mechanism:

- **`DoubleArrayTrie(Char)`** — immutable after build; reads touch read-only arrays
  (wait-free).
- **`DynamicDawg(Char)`**, **`DynamicDawgU64`** — the `LockFreeDawg` core: reads retain
  one immutable root revision, while writers path-copy the changed route and
  `compare_exchange` a replacement `GraphVersion`. (`DynamicDawg` uses `u8` edge
  labels; `DynamicDawgU64` a wider `u64` label.)
- **`SuffixAutomaton(Char)`**, **`Scdawg(Char)`** — the `LockFreeSuffixAutomaton` /
  `LockFreeScdawg` cores: reads load an `Arc<ArcSwap<…>>` graph snapshot; writes publish
  a new graph by atomic swap.
- **`PathMapDictionary(Char)`** — reads load an `Arc<ArcSwap<PathMapState>>` snapshot and
  walk it lock-free; writes publish a new state by atomic swap.
- **`BijectiveMap`** — a lock-free forward `DynamicDawgChar` plus a reverse
  `Arc<ArcSwap<HashMap>>`; both directions publish by atomic swap.
- **PathMap snapshots** (`PathMapSnapshot(Char)`, `PathMapRef(Char)`) — a persistent,
  copy-on-write view that holds no lock.
- **The disk-persisted `Persistent*` family** — `PersistentARTrie(Char/U64)`,
  `PersistentScdawg(Char)`, `PersistentSuffixAutomaton(Char)`,
  `PersistentSuffixTree(Char)`, `PersistentVocabARTrie` — all read through a lock-free
  CAS / `ArcSwap` overlay over memory-mapped storage (durable *and* lock-free).

> **Historical note.** Earlier releases guarded the three in-memory dynamic backends
> (`DynamicDawg`, `SuffixAutomaton`, `Scdawg`), `PathMapDictionary`, and `BijectiveMap`
> with a `parking_lot::RwLock`, so a writer briefly excluded all readers. They have since
> adopted the same lock-free reader model as `DynamicDawgU64` and the `Persistent*`
> family: every dictionary backend now reports `SyncStrategy::InternalSync` (or
> `Persistent`), and no dictionary read blocks on a writer. The `parking_lot` /
> `std::sync` `RwLock` distinction now applies only to internal coordination inside the
> disk-backed `persistent_artrie` engine, never to a dictionary read path.

## The two read paths

Every dictionary backend now uses the lock-free `ArcSwap` read path on the left. The
`parking_lot::RwLock` path on the right is the model the in-memory backends used in
earlier releases — and the one the disk-backed `persistent_artrie` engine still uses for
internal coordination. The contrast is what motivated the migration:

![Two read paths contrasted: the wait-free ArcSwap path (reader atomic-loads an Arc snapshot and never blocks; a writer builds a new Arc and atomically swaps it) versus the parking_lot RwLock path (reader acquires a shared read guard that may block on a writer; a writer takes the exclusive write guard).](../diagrams/concurrency/arcswap-vs-rwlock.svg)

*Lock-free `ArcSwap` reads (now used by every dictionary backend) vs. the `parking_lot`
`RwLock` reads used by the in-memory backends in earlier releases and still used for
internal coordination inside the disk-backed `persistent_artrie` engine.*

## Using a dictionary across threads

Any backend can be shared and queried concurrently — clone the `Transducer` (an `Arc`
bump) and move clones into threads:

```rust
use liblevenshtein::prelude::*;
use std::thread;

let dict = DoubleArrayTrie::from_terms(vec!["test", "testing", "tester"]);
let transducer = Transducer::new(dict, Algorithm::Standard);

let handles: Vec<_> = ["tset", "tesing", "testr"]
    .into_iter()
    .map(|q| {
        let t = transducer.clone(); // cheap: Arc clone
        thread::spawn(move || t.query(q, 2).collect::<Vec<_>>())
    })
    .collect();

for h in handles {
    for term in h.join().expect("query thread") {
        println!("{term}");
    }
}
```

On every backend those queries never block one another: `DoubleArrayTrie` reads immutable
arrays, and each dynamic backend (`DynamicDawg`, `DynamicDawgU64`, `SuffixAutomaton`,
`Scdawg`, `PathMapDictionary`, `BijectiveMap`) and `Persistent*` type reads through a
lock-free `ArcSwap` / CAS snapshot. A concurrent writer publishes new state by an atomic
swap and never excludes readers.

### Concurrent reads with writes

Dynamic backends accept writes through `&self` (interior mutability), so a shared
handle can be mutated while others query:

```rust
let dict = DynamicDawg::from_terms(vec!["alpha", "beta"]);
let writer = dict.clone();
thread::spawn(move || { writer.insert("gamma"); }); // lock-free: publishes via atomic swap

// Other threads keep querying without ever blocking; they observe the update as soon as
// the writer's atomic swap completes. Reads never wait on a writer.
for term in dict.query("gama", 1) { println!("{term}"); }
```

### Measured behaviour (PathMapDictionary, lock-free `ArcSwap`)

`PathMapDictionary` holds its state in an `Arc<ArcSwap<…>>`, so reads are **lock-free**
and a write publishes a new state by an atomic pointer swap — readers never block:

```rust
pub struct PathMapDictionary<V: DictionaryValue = ()> {
    state: Arc<ArcSwap<PathMapState<V>>>,
}
```

A reader loads the current state through an `arc-swap` guard (no lock taken); a writer
builds the next state from a persistent, structurally shared copy of the trie and swaps it
in with a single atomic store. This is the same copy-on-write discipline exposed by the
`PathMapSnapshot` / `PathMapRef` views.

> **Historical note.** Earlier releases wrapped the upstream PathMap in `Arc<RwLock<…>>`.
> Under that design the concurrency tests (`tests/concurrency_test.rs`) measured **~3.82×
> read throughput on 8 threads** — readers did not block one another and queries proceeded
> during interleaved writes, the expected profile of a reader–writer lock with a short
> write critical section. Moving to `ArcSwap` removes the read lock entirely, so readers no
> longer contend even momentarily.

## Choosing for concurrency

- **Many readers, heavy concurrent writes → lock-free reads:** any in-memory dynamic
  backend — `DynamicDawg`, `DynamicDawgU64`, or `PathMapDictionary` — or a `Persistent*`
  type (durable). Readers never block on a writer.
- **Static dictionary → wait-free:** `DoubleArrayTrie` (immutable after build).
- **General dynamic use:** `DynamicDawg` (byte) or `DynamicDawgChar` (Unicode) — lock-free
  reads with `compare_exchange` writes; `DynamicDawgU64` trades a wider `u64` edge label
  for the same guarantees.
- **Substring search under concurrency:** `SuffixAutomaton` / `Scdawg` (lock-free) in
  memory, or `PersistentSuffixAutomaton` / `PersistentScdawg` (lock-free, durable) on disk.

## Related Documentation

- [Backends](backends.md) — dictionary backend comparison
- [Architecture (concurrency)](../developer-guide/architecture.md#thread-safety) — the intra-crate concurrency design
- [Getting Started](getting-started.md) — backend selection table
- [GLOSSARY → RwLock](../GLOSSARY.md) — terminology

---

[← Documentation Index](../README.md)
