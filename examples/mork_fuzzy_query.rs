//! Zero-plumbing fuzzy queries over a bare `PathMap`, MORK-style.
//!
//! MORK's `Space` holds a bare `PathMap<V>` (`space.btm`) of byte-encoded
//! s-expression paths and reads it exclusively through zippers. The PathMap
//! developer noted that wiring liblevenshtein in "would require quite a bit of
//! plumbing" because the old integration demanded ownership via
//! `Arc<RwLock<PathMap>>`. The TrieRef rework removes that: you can fuzzy-query
//! a map MORK already holds **without copying it and without a lock**, through
//! three entry points:
//!
//!   1. [`PathMapRef::from_map`]`(&space.btm)`        — zero-copy borrow
//!   2. [`PathMapSnapshot::from_map_ref`]`(&space.btm)` — `𝒪(1)` copy-on-write snapshot
//!   3. [`PathMapRef::from_trie_ref`]`(space.btm.trie_ref_at_path(prefix))`
//!      — fuzzy search scoped to a subtrie
//!
//! Run with:
//! ```text
//! cargo run --example mork_fuzzy_query --features pathmap-backend
//! ```

use libdictenstein::pathmap::{PathMapRef, PathMapSnapshot};
use liblevenshtein::prelude::*;
use pathmap::PathMap;

/// A stand-in for MORK's `Space`: a bare `PathMap` of byte paths.
///
/// Here a path is `"<namespace>/<symbol>"`. MORK's real encoding tags each
/// s-expression node with an arity byte (`0x00..=0x3F`) and each symbol with a
/// size byte (`0xC1..=0xFF`) — but that structure is opaque to the entry points
/// below, which fuzzy-match over whatever byte suffix the chosen subtrie root
/// exposes. A printable `/` separator just keeps this example legible.
struct Space {
    btm: PathMap<()>,
}

impl Space {
    fn new() -> Self {
        Space {
            btm: PathMap::new(),
        }
    }

    fn add_symbol(&mut self, namespace: &str, symbol: &str) {
        let path = format!("{namespace}/{symbol}");
        self.btm.insert(path.as_bytes(), ());
    }
}

fn run<D>(label: &str, dict: D, query: &str, max_distance: usize)
where
    D: Dictionary,
{
    let transducer = Transducer::new(dict, Algorithm::Standard);
    print!("  {label}: query \"{query}\" (d≤{max_distance}) -> ");
    let mut hits: Vec<String> = transducer
        .query_with_distance(query, max_distance)
        .map(|c| format!("{} (d{})", c.term, c.distance))
        .collect();
    hits.sort();
    println!(
        "{}",
        if hits.is_empty() {
            "(none)".into()
        } else {
            hits.join(", ")
        }
    );
}

fn main() {
    let mut space = Space::new();
    for (ns, sym) in [
        ("concept", "vector"),
        ("concept", "vectors"),
        ("concept", "factor"),
        ("concept", "sector"),
        ("relation", "tensor"),
    ] {
        space.add_symbol(ns, sym);
    }

    println!("== 1. Zero-copy borrow of the live map (PathMapRef::from_map) ==");
    // Borrows `space.btm` directly; no copy, no lock. Paths include the
    // namespace, so we fuzzy-match the full encoded path.
    run(
        "borrowed",
        PathMapRef::from_map(&space.btm),
        "concept/vecto",
        1,
    );

    println!("\n== 2. 𝒪(1) copy-on-write snapshot (PathMapSnapshot::from_map_ref) ==");
    let snapshot = PathMapSnapshot::from_map_ref(&space.btm);
    // Mutate the space AFTER snapshotting:
    space.add_symbol("concept", "verctor"); // a fresh near-duplicate of "vector"
    run("snapshot (pre-mutation)", snapshot, "concept/vector", 1);
    run(
        "fresh borrow (post-mutation)",
        PathMapRef::from_map(&space.btm),
        "concept/vector",
        1,
    );
    // The snapshot does not see "verctor"; the fresh borrow does — that is the
    // copy-on-write isolation in action.

    println!("\n== 3. Subtrie-scoped fuzzy symbol search (from_trie_ref) ==");
    // Root the search PAST the "concept/" prefix, so the query is the bare
    // symbol and "relation/tensor" is excluded by construction.
    let concept = PathMapRef::from_trie_ref(space.btm.trie_ref_at_path(b"concept/"));
    run("concept subtrie", concept, "vector", 1);
}
