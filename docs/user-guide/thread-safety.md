# Dictionary Thread Safety

Every dictionary backend in liblevenshtein (via `libdictenstein`) is `Send + Sync`
and cheap to share across threads — a clone is an `Arc` bump, not a deep copy. What
differs between backends is **how concurrent reads and writes are coordinated**: some
reads are lock-free (a reader never blocks), others take a `parking_lot` reader–writer
lock (a reader blocks only while a writer holds the exclusive lock).

![Concurrency model: every dictionary backend grouped by how reads are coordinated — lock-free/wait-free reads (immutable arrays, ArcSwap RCU, persistent copy-on-write snapshots, and the disk-persisted CAS/ArcSwap overlay family) versus parking_lot RwLock readers (the in-memory dynamic DAWG/automaton backends, PathMapDictionary, and BijectiveMap).](../diagrams/concurrency/concurrency-model.svg)

*Concurrency model — dictionary reads grouped by synchronization strategy.*

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
| **`DynamicDawgU64`** | per-node `ArcSwap<EdgeList>` load | No | lock-free `compare_exchange` (`&self`) | dynamic |
| **`PathMapSnapshot(Char)` / `PathMapRef(Char)`** | persistent copy-on-write snapshot | No | — (immutable view) | static snapshot |
| **`PersistentARTrie(Char / U64)`** | lock-free CAS overlay over a memory-mapped trie | No | lock-free CAS (`&self`) | dynamic · disk |
| **`PersistentScdawg(Char)`** | `ArcSwap` graph load | No | `ArcSwap` publish (`&self`) | dynamic · disk |
| **`PersistentSuffixAutomaton(Char)`** | `ArcSwap` graph load | No | `ArcSwap` publish (`&self`) | dynamic · disk |
| **`PersistentSuffixTree(Char)`** | `ArcSwap` graph load | No | `ArcSwap` publish (`&self`) | dynamic · disk |
| **`PersistentVocabARTrie`** | lock-free overlay | No | lock-free overlay (`&self`) | dynamic · disk |
| **`DynamicDawg(Char)`** | `Arc<RwLock<DawgCore>>` → `.read()` | **Yes** | `.write()` (`&self`) | dynamic |
| **`SuffixAutomaton(Char)`** | `Arc<RwLock<…>>` → `.read()` | **Yes** | `.write()` (`&self`) | dynamic |
| **`Scdawg(Char)`** | `Arc<RwLock<…>>` → `.read()` | **Yes** | `.write()` (`&self`) | dynamic |
| **`PathMapDictionary(Char)`** | brief `.read()` to grab an `𝒪(1)` snapshot, then lock-free traversal | momentarily | `.write()` (`&self`) | dynamic |
| **`BijectiveMap`** | two `RwLock`s (forward `DynamicDawgChar` + reverse `HashMap`) | **Yes** | both `.write()` (`&self`) | dynamic |

### Lock-free / wait-free reads

A reader never blocks on these — choose them when many threads query under heavy
concurrent writes:

- **`DoubleArrayTrie(Char)`** — immutable after build; reads touch read-only arrays.
- **`DynamicDawgU64`** — per-node `ArcSwap` RCU; reads `load()` a snapshot, writes
  `compare_exchange` a new node, all lock-free.
- **PathMap snapshots** (`PathMapSnapshot(Char)`, `PathMapRef(Char)`) — a persistent,
  copy-on-write view that holds no lock.
- **The disk-persisted `Persistent*` family** — `PersistentARTrie(Char/U64)`,
  `PersistentScdawg(Char)`, `PersistentSuffixAutomaton(Char)`,
  `PersistentSuffixTree(Char)`, `PersistentVocabARTrie` — all read through a lock-free
  CAS / `ArcSwap` overlay over memory-mapped storage (durable *and* lock-free).

### `RwLock` readers

These guard their state with a `parking_lot::RwLock` (the default;
`std::sync::RwLock` is the WASM/no-`parking_lot` fallback). Multiple readers run
concurrently; a writer briefly excludes all readers while it holds the write lock:

- **`DynamicDawg(Char)`**, **`SuffixAutomaton(Char)`**, **`Scdawg(Char)`** — the
  in-memory dynamic DAWG / automaton backends.
- **`PathMapDictionary(Char)`** — takes the lock only to clone an `𝒪(1)` copy-on-write
  snapshot, then walks the snapshot lock-free; readers are blocked only for that
  snapshot grab.
- **`BijectiveMap`** — holds two locks (a forward `DynamicDawgChar` and a reverse
  `Arc<RwLock<HashMap>>`) to keep both directions consistent.

> **Intended direction.** The three in-memory RwLock backends (`DynamicDawg`,
> `SuffixAutomaton`, `Scdawg`) are slated to adopt the lock-free reader model already
> shipping on `DynamicDawgU64` and the `Persistent*` family. Until then, their reads
> take a shared lock — accurate as of the current release.

## The two read paths

The contrast between the lock-free and lock-guarded read paths:

![Two read paths contrasted: the wait-free ArcSwap path (reader atomic-loads an Arc snapshot and never blocks; a writer builds a new Arc and atomically swaps it) versus the parking_lot RwLock path (reader acquires a shared read guard that may block on a writer; a writer takes the exclusive write guard).](../diagrams/concurrency/arcswap-vs-rwlock.svg)

*Wait-free `ArcSwap` reads (used by `DynamicDawgU64` and the `Persistent*` family) vs.
`parking_lot` `RwLock` reads (used by the in-memory dynamic DAWG/automaton backends).*

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

On a lock-free backend (`DoubleArrayTrie`, `DynamicDawgU64`, a `Persistent*` type)
those queries never block one another. On an `ExternalSync` backend
(`DynamicDawg`, `SuffixAutomaton`, `Scdawg`) the readers share a read lock and only
stall while a concurrent writer holds the write lock.

### Concurrent reads with writes

Dynamic backends accept writes through `&self` (interior mutability), so a shared
handle can be mutated while others query:

```rust
let dict = DynamicDawg::from_terms(vec!["alpha", "beta"]);
let writer = dict.clone();
thread::spawn(move || { writer.insert("gamma"); }); // takes the write lock briefly

// Other threads keep querying; on DynamicDawg they observe the update once the
// write lock is released. On DynamicDawgU64 the swap is lock-free and immediate.
for term in dict.query("gama", 1) { println!("{term}"); }
```

### Measured behaviour (PathMapDictionary, RwLock)

`PathMapDictionary` wraps the upstream PathMap in `Arc<RwLock<…>>`:

```rust
pub struct PathMapDictionary {
    map: Arc<RwLock<PathMap<()>>>,
    term_count: Arc<RwLock<usize>>,
}
```

Its concurrency tests (`tests/concurrency_test.rs`) measure **~3.82× read throughput
on 8 threads** (readers do not block one another), with queries proceeding during
interleaved writes — the expected profile of a reader–writer lock whose write
critical section is short.

## Choosing for concurrency

- **Many readers, heavy concurrent writes → lock-free reads:** `DynamicDawgU64` (in
  memory) or a `Persistent*` type (durable). Writers never block readers.
- **Static dictionary → wait-free:** `DoubleArrayTrie` (immutable after build).
- **General dynamic use:** `DynamicDawg` is fine — its `RwLock` write section is short;
  readers stall only momentarily during a write.
- **Substring search under concurrency:** `SuffixAutomaton` / `Scdawg` (RwLock) in
  memory, or `PersistentSuffixAutomaton` / `PersistentScdawg` (lock-free) on disk.

## Related Documentation

- [Backends](backends.md) — dictionary backend comparison
- [Architecture (concurrency)](../developer-guide/architecture.md#thread-safety) — the intra-crate concurrency design
- [Getting Started](getting-started.md) — backend selection table
- [GLOSSARY → RwLock](../GLOSSARY.md) — terminology

---

[← Documentation Index](../README.md)
