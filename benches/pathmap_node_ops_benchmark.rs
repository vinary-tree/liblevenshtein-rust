//! Micro-benchmarks for the TrieRef-based PathMap node operations.
//!
//! Seeds the hypothesis ledger for the TrieRef rework:
//! - **H1** `transition()` is `𝒪(1)` from the focus (flat across depth), not
//!   `𝒪(depth)` as the old root-replay node was.
//! - **H2** `edges().count()` scales with fanout via word-skipping mask
//!   iteration, with no per-child lock/replay/re-validation.
//! - **H5** `root()` snapshot cost is sub-microsecond.
//! - **H6** char-level `edges()` decodes mixed-width UTF-8 by local descent.
//!
//! Run (taskset-pinned, quiet system):
//! ```text
//! taskset -c 2 cargo bench --bench pathmap_node_ops_benchmark --features pathmap-backend
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use libdictenstein::pathmap::{PathMapDictionary, PathMapDictionaryChar};
use libdictenstein::{Dictionary, DictionaryNode};
use std::hint::black_box;

/// H1: cost of a single `transition()` at increasing depth. With TrieRef the
/// focus descends in `𝒪(1)` per byte, so this should be flat across depth.
fn transition_at_depth(c: &mut Criterion) {
    let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(vec!["a".repeat(64)]);
    let mut group = c.benchmark_group("transition_at_depth");
    for depth in [1usize, 5, 10, 20, 40] {
        // Pre-descend to `depth` (untimed setup).
        let mut node = dict.root();
        for _ in 0..depth {
            node = node.transition(b'a').expect("a-chain");
        }
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, _| {
            b.iter(|| black_box(node.transition(black_box(b'a'))));
        });
    }
    group.finish();
}

/// H1 (corrected regime): `transition()` at depth in a **branching** trie.
///
/// [`transition_at_depth`] above seeds the focus from a single-`a` chain, which
/// pathmap path-compresses into ~1 node — so the *old* path-replay node was
/// `𝒪(1)` there too, masking its `𝒪(depth)` behaviour (both ended up flat). Here
/// a **comb** forces a real branch node at every level: alongside the `a`-spine,
/// each prefix `"a"*i` also owns an `"a"*i + 'b'` leaf, so `"a"*i` has two
/// children and cannot be compressed. Descending `"a"*depth` therefore traverses
/// `depth` *distinct* nodes — the regime where the old node replays
/// `read_zipper_at_path(𝒪(depth))` per transition while the TrieRef node stays
/// `𝒪(1)` from its focus. Run against the frozen tree, this is the experiment
/// that actually tests H1's `𝒪(depth)→𝒪(1)` claim.
fn transition_at_depth_branching(c: &mut Criterion) {
    let mut group = c.benchmark_group("transition_at_depth_branching");
    for depth in [1usize, 5, 10, 20, 40] {
        // 'b' sibling at every spine level => every "a"*i is a 2-child branch;
        // the spine runs to depth+2 so the depth node keeps an 'a' child to enter.
        let mut terms: Vec<String> = Vec::with_capacity(depth + 3);
        for i in 0..=depth + 1 {
            let mut leaf = "a".repeat(i);
            leaf.push('b');
            terms.push(leaf);
        }
        terms.push("a".repeat(depth + 2));
        let dict: PathMapDictionary<()> =
            PathMapDictionary::from_terms(terms.iter().map(String::as_str).collect::<Vec<_>>());
        // Pre-descend to `depth` (untimed setup).
        let mut node = dict.root();
        for _ in 0..depth {
            node = node.transition(b'a').expect("a-spine");
        }
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, _| {
            b.iter(|| black_box(node.transition(black_box(b'a'))));
        });
    }
    group.finish();
}

/// H2 (corrected regime): `edges()` at a **deep** node, fanout held fixed at 8.
///
/// [`edges_at_fanout`] below measures at the *root*, where the old node had no
/// base path to replay per child — so its `≥3×` margin shrank to ~2.4× at high
/// fanout (both backends pay `𝒪(fanout)` for the enumeration itself). But the
/// old `edges()` re-walks the base path **once per child**
/// (`read_zipper_at_path(base_path)` inside the child loop), so at a *deep* node
/// its cost is `𝒪(fanout × depth)` while the TrieRef node is `𝒪(fanout)`
/// regardless of depth. A comb (a `z` branch at every spine level) defeats
/// compression; the measured deep node carries 8 children.
fn edges_at_depth_branching(c: &mut Criterion) {
    let mut group = c.benchmark_group("edges_at_depth_branching");
    for depth in [1usize, 8, 32] {
        let spine = "a".repeat(depth);
        let mut terms: Vec<String> = Vec::with_capacity(8 + depth);
        for i in 0..8u8 {
            terms.push(format!("{spine}{}", (b'a' + i) as char)); // 8 children on the deep node
        }
        for i in 0..depth {
            terms.push(format!("{}z", "a".repeat(i))); // 'z' branch at every spine level
        }
        let dict: PathMapDictionary<()> =
            PathMapDictionary::from_terms(terms.iter().map(String::as_str).collect::<Vec<_>>());
        let mut node = dict.root();
        for _ in 0..depth {
            node = node.transition(b'a').expect("a-spine");
        }
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, _| {
            b.iter(|| black_box(node.edges().count()));
        });
    }
    group.finish();
}

/// H6 (corrected regime): char `edges()` at a **deep** node, width held fixed.
///
/// [`char_edges_mixed`] measures at the root, where the old char node had no base
/// path to replay — so its gap to the local-descent TrieRef node was only ~2.25×.
/// The hypothesis is about a *deep* node: the old `edges()` re-walks the base
/// path once per child (`𝒪(width × depth)`), the TrieRef node descends
/// continuation bytes locally from its focus (`𝒪(width)`). A CJK comb (a `犬`
/// branch at every prefix level) defeats compression so the deep node sits
/// `depth` real nodes below the root; width is held at the 8 mixed-width
/// suffixes (ASCII / 2-byte / 3-byte CJK / 4-byte emoji).
fn char_edges_at_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_edges_at_depth");
    let suffixes = ["a", "b", "é", "ñ", "中", "文", "🎉", "🚀"];
    for depth in [1usize, 8, 32] {
        let prefix: String = "語".repeat(depth);
        let mut terms: Vec<String> = Vec::with_capacity(suffixes.len() + depth);
        for s in suffixes {
            terms.push(format!("{prefix}{s}"));
        }
        for i in 0..depth {
            terms.push(format!("{}犬", "語".repeat(i))); // branch at every prefix level
        }
        let dict: PathMapDictionaryChar<()> = PathMapDictionaryChar::from_terms(terms);
        // Descend to the deep node (untimed setup).
        let mut node = dict.root();
        for ch in prefix.chars() {
            node = node.transition(ch).expect("prefix char");
        }
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, _| {
            b.iter(|| black_box(node.edges().count()));
        });
    }
    group.finish();
}

/// H2: cost of enumerating all edges at a given fanout (mask `iter()`).
fn edges_at_fanout(c: &mut Criterion) {
    let mut group = c.benchmark_group("edges_count_at_fanout");
    for fanout in [2usize, 8, 26] {
        let terms: Vec<String> = (0..fanout)
            .map(|i| ((b'a' + i as u8) as char).to_string())
            .collect();
        let dict: PathMapDictionary<()> = PathMapDictionary::from_terms(terms);
        let root = dict.root();
        group.bench_with_input(BenchmarkId::from_parameter(fanout), &fanout, |b, _| {
            b.iter(|| black_box(root.edges().count()));
        });
    }
    group.finish();
}

/// H5: cost of taking a root (an `𝒪(1)` copy-on-write snapshot).
fn root_cost(c: &mut Criterion) {
    let dict: PathMapDictionary<()> =
        PathMapDictionary::from_terms((0..1000).map(|i| format!("term{i}")));
    c.bench_function("root_snapshot", |b| b.iter(|| black_box(dict.root())));
}

/// H6: char-level `edges()` over mixed-width UTF-8 children (ASCII, 2-byte,
/// 3-byte CJK, 4-byte emoji), decoded by local continuation-byte descent.
fn char_edges_mixed(c: &mut Criterion) {
    let dict: PathMapDictionaryChar<()> =
        PathMapDictionaryChar::from_terms(vec!["a", "b", "é", "ñ", "中", "文", "🎉", "🚀"]);
    let root = dict.root();
    c.bench_function("char_edges_mixed_width", |b| {
        b.iter(|| black_box(root.edges().count()))
    });
}

criterion_group!(
    benches,
    transition_at_depth,
    transition_at_depth_branching,
    edges_at_depth_branching,
    char_edges_at_depth,
    edges_at_fanout,
    root_cost,
    char_edges_mixed
);
criterion_main!(benches);
