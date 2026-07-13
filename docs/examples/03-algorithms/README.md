# 03 · Algorithms & Result Ordering

**What you'll learn.** The three edit *algorithms* the transducer can simulate
(`Standard`, `Transposition`, `MergeAndSplit`), and — the focus of this tutorial — how
to get results back in a useful **order**. `query` walks the trie depth-first and yields
matches in traversal order; `query_ordered` returns them **distance-first, then
lexicographically**, which is exactly what autocomplete and "top-`$k$` nearest neighbors"
want. You'll see how laziness makes `.take(k)` and `.take_while(…)` genuinely cheap.

---

## The concept

### Edit operation sets (the `Algorithm` enum)

An `Algorithm` selects *which edit operations* count as a single unit of distance. The
transducer simulates the corresponding automaton; everything else (dictionary, query
API) is unchanged:

| `Algorithm` | Operations counted as one edit | Example it uniquely catches |
|---|---|---|
| **`Standard`** | insertion, deletion, substitution | `"aple"` → `"apple"` |
| **`Transposition`** | the above **+ adjacent swap** | `"tset"` → `"test"` at `$k = 1$` |
| **`MergeAndSplit`** | the above **+ merge two→one / split one→two** | `"rn"` ↔ `"m"` OCR confusions |

> Terms defined. A **transposition** swaps two adjacent characters (`ab` → `ba`) and
> charges *one* edit instead of the two a `Standard` substitution-pair would need. A
> **merge** collapses two query characters into one candidate character; a **split**
> expands one into two — together they model the classic OCR `rn`/`m` and `cl`/`d`
> confusions.

![Operation sets: a Venn-style diagram showing Standard at the core, Transposition adding the adjacent-swap operation, and MergeAndSplit adding the merge and split operations on top.](../../diagrams/automata/operation-sets.svg)

### Two ways to consume results

Both iterators yield the same *set* of matches within distance `k`; they differ only in
**order** and in what they yield:

- **`query(W, k)` → terms**, in **depth-first traversal order** (whatever order the trie
  is walked). Cheapest when you intend to consume *all* matches.
- **`query_ordered(W, k)` → `{ term, distance }`**, sorted **by distance ascending, then
  alphabetically** within each distance band. This is a *distance-stratified breadth
  exploration*: it surfaces the closest matches first.

Crucially, `query_ordered` is **lazy**: it only computes as many results as you actually
consume. So `query_ordered(W, k).take(5)` does the work for ~5 results, not for the
whole near-match frontier — the basis of an efficient top-`$k$`.

### Why ordering matters

For spell-check and autocomplete you want the *best* suggestion first and often only the
first few. Distance-first ordering means "rank by quality"; the lexicographic tiebreak
makes the output deterministic and comes *for free* from the trie's sorted edges, so no
separate sort pass is needed.

![Query-iterator hierarchy: the family of result iterators (unordered QueryIterator, distance-with-term, and the ordered/priority iterator) and how each is produced from a Transducer.](../../diagrams/traversal/query-iterator-hierarchy.svg)

---

## Walking through `examples/ordered_query_demo.rs`

### 1 · Build a dictionary and get ordered results

`query_ordered` yields candidates already grouped by distance and alphabetized within a
group — note how `enumerate()` gives you a natural rank:

```rust
use liblevenshtein::prelude::*;

let words = vec![
    "test", "tests", "tested", "testing", "tester", "best", "rest",
    "nest", "west", "taste", "text", "tent", "temp", "team",
];
let dict = DoubleArrayTrie::from_terms(words.iter().copied());
let transducer = Transducer::new(dict, Algorithm::Standard);

for (rank, candidate) in transducer.query_ordered("tset", 3).enumerate() {
    println!("{:<3} {:<8} {}", rank + 1, candidate.distance, candidate.term);
}
// distance 0 group first, then 1, then 2, then 3 — alphabetical within each.
```

### 2 · Top-`k` nearest neighbors for free

Because the iterator is lazy, `.take(5)` computes only what it needs — an efficient
top-`$k$` with no manual heap:

```rust
for candidate in transducer.query_ordered("tset", 3).take(5) {
    println!("  {} (distance: {})", candidate.term, candidate.distance);
}
// Only ~5 results are ever computed.
```

### 3 · Distance-bounded queries with early stop

Since results arrive in non-decreasing distance order, `take_while` can **stop the whole
traversal** the moment the distance threshold is crossed — you never pay for the farther
bands:

```rust
let close: Vec<_> = transducer
    .query_ordered("tset", 3)
    .take_while(|c| c.distance <= 1)     // stops as soon as distance hits 2
    .collect();
```

### 4 · Ordered vs unordered — same set, different order

A direct comparison makes the contract explicit: identical results, but `query_ordered`
ranks them while `query` returns trie-traversal order:

```rust
let ordered: Vec<_> = transducer
    .query_ordered("test", 1)
    .map(|c| format!("{}(d={})", c.term, c.distance))
    .collect();

let unordered: Vec<_> = transducer.query("test", 1).collect();   // traversal order, terms only
```

For an autocomplete box this is the whole game: `query_ordered(prefix, k).take(n)` gives
the `$n$` best corrections, best-first, computing only what it shows.

---

## Run it

No features required:

```bash
cargo run --example ordered_query_demo
```

The output walks six scenarios: full ordered listing, top-5, distance-bounded,
an autocomplete simulation, an ordered-vs-unordered comparison, and a per-distance
histogram. A companion micro-benchmark lives in `examples/ordered_query_benchmark.rs`.

---

## Key takeaways

- **`Algorithm`** picks the operation set: `$\texttt{Standard} \subset \texttt{Transposition} \subset \texttt{MergeAndSplit}$`.
- **`query`** yields terms in traversal order; **`query_ordered`** yields
  `{ term, distance }` distance-first, then alphabetical.
- `query_ordered` is **lazy** — `.take(k)` and `.take_while(…)` make top-`$k$` and
  distance-bounded queries cheap, with the lexicographic tiebreak free from the trie.

---

[← 02 · Dictionaries](../02-dictionaries/README.md) · Next: [04 · Queries & Unicode →](../04-queries/README.md)

[← Documentation Index](../../README.md)
