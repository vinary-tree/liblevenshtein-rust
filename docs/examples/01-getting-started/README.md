# 01 · Getting Started: Your First Spell Checker

**What you'll learn.** How to stand up a working fuzzy spell checker in roughly ten
lines of Rust: build a dictionary from a word list, wrap it in a `Transducer`, and
ask for every term within a small edit distance of a (possibly misspelled) query.
By the end you will understand the three objects that every `liblevenshtein` program
is built from — a **dictionary**, an **algorithm**, and a **transducer** — and the
difference between the `query` and `query_with_distance` result iterators.

---

## The concept

### What is approximate string matching?

**Approximate string matching** (a.k.a. *fuzzy matching*) answers the question
*"which dictionary words are close to what the user typed?"* "Close" is made precise
by the **Levenshtein (edit) distance** $`d(W, s)`$: the minimum number of single-character
**insertions**, **deletions**, and **substitutions** that turn the query $`W`$ into a
candidate string $`s`$. For example $`d(\texttt{aple}, \texttt{apple}) = 1`$ (insert one `p`), and
$`d(\texttt{wrld}, \texttt{world}) = 2`$ (insert `o`, then `l`… or insert `o` and substitute — either
way, two edits).

### How does `liblevenshtein` do it without scanning the whole dictionary?

A naïve checker computes $`d(W, s)`$ against *every* entry $`s`$, costing
$`\mathcal{O}(\lvert D\rvert \cdot \lvert W\rvert \cdot \lvert s\rvert)`$ — you re-pay the query-length factor $`\lvert W\rvert`$ once per word. Instead,
`liblevenshtein` represents the query $`W`$ and the error bound $`k`$ as a **Levenshtein
automaton**: the set of still-viable $`\langle \text{position}, \text{errors}\rangle`$ states that together accept
*exactly* the strings within distance $`k`$ of $`W`$. It then walks that automaton
**in lock-step** with the dictionary trie — advancing both one symbol at a time and
**pruning** a branch the instant no automaton state survives. The automaton is
simulated and determinized lazily: only state/class transitions reached during
the walk are stored, rather than compiling an eager standalone table. Per-query
setup is $`\mathcal{O}(\lvert W\rvert)`$; the first computation of a transition costs
$`\mathcal{O}(k)`$ (a constant for fixed $`k`$), while a repeated transition is a
table lookup. The total work tracks the explored near-match frontier, not
$`\lvert D\rvert`$.

> Terms defined: $`W`$ = the query string, $`\lvert W\rvert`$ = its length, $`s`$ = a candidate from the
> dictionary, $`D`$ = the dictionary, $`\lvert D\rvert`$ = its number of edges, $`k`$ = the maximum edit
> distance (error bound), and a **position** $`\langle i, e\rangle`$ = an automaton state meaning
> "$`i`$ characters of $`W`$ consumed, $`e`$ edits spent, with $`e \le k`$".

### Why three objects?

The library cleanly separates *what you search* (the **dictionary** — here a static,
read-only `DoubleArrayTrie`, a trie packed into two integer arrays for $`\mathcal{O}(1)`$-per-edge
lookups), *how edits are counted* (the **`Algorithm`** — `Standard` counts insert /
delete / substitute; `Transposition` additionally counts an adjacent swap as one edit),
and *the engine that runs one against the other* (the **`Transducer`**, which yields
matches). Swapping any one of the three leaves the rest untouched.

![End-to-end spell-check pipeline: a misspelled query is turned into a Levenshtein automaton, intersected with the dictionary trie, and the surviving terms are returned as ranked suggestions.](../../diagrams/traversal/end-to-end-spellcheck.svg)

---

## Walking through `examples/spell_checker.rs`

### 1 · Build a dictionary and a transducer

A `DoubleArrayTrie` is the right backend for a *static* word list: fastest reads, but
treat it as read-only once built. We wrap it in a `Transducer` together with
`Algorithm::Standard`.

```rust
use liblevenshtein::prelude::*;

let dictionary_words = vec![
    "apple", "application", "apply", "banana", "band",
    "can", "candy", "cat", "dog", "test", "testing", "tested",
];

let dict = DoubleArrayTrie::from_terms(dictionary_words);
let transducer = Transducer::new(dict, Algorithm::Standard);
```

`from_terms` accepts anything iterable of `&str`/`String`; `Transducer::new` takes
ownership of the dictionary and the chosen algorithm.

### 2 · Query for terms, ignoring the distance

`transducer.query(W, k)` returns an iterator over just the matching **terms** within
distance $`k`$. Collecting it materializes the suggestions:

```rust
let matches: Vec<_> = transducer.query("aple", 1).collect();
// → ["apple"]   (one insertion of 'p')

for term in transducer.query("tset", 1) {   // transposition of "test"
    println!("  - {}", term);
}
```

With `Algorithm::Standard`, `"tset"` → `"test"` costs two edits (delete `s`, insert `s`),
so at $`k = 1`$ it is *not* matched — which motivates the algorithm comparison below.

### 3 · Query *with* the edit distance attached

When you want to *rank* suggestions (best first) or display how far each is from the
query, use `query_with_distance`. Each item is a candidate carrying both `.term` and
`.distance`:

```rust
for candidate in transducer.query_with_distance("tast", 2) {
    println!("  - {} (distance: {})", candidate.term, candidate.distance);
}
// e.g. → test (distance: 1), tested (distance: 2), ...
```

### 4 · Swap the algorithm to count transpositions

The *only* change needed to treat an adjacent character swap as a single edit is the
`Algorithm` argument — the dictionary and query API are identical:

```rust
let dict = DoubleArrayTrie::from_terms(vec!["test", "set", "reset"]);

let standard = Transducer::new(dict.clone(), Algorithm::Standard);
let standard_hits: Vec<_> = standard.query("tset", 1).collect();      // []  — needs 2 edits

let transposition = Transducer::new(dict, Algorithm::Transposition);
let trans_hits: Vec<_> = transposition.query("tset", 1).collect();    // ["test"] — 1 swap
```

This is the core idea you will reuse everywhere: a query is a lazy *simulation* of a
parameterized automaton over reduced position-sets, walked lock-step with the dictionary.

![Query flow: build one automaton from the query, then intersect it with the dictionary in a single depth-first traversal, pruning subtrees as soon as the automaton has no surviving state.](../../diagrams/traversal/query-flow.svg)

---

## Run it

This example needs no feature flags:

```bash
cargo run --example spell_checker
```

You should see suggestions for each typo, a "Query with Distances" section, and an
"Algorithm Comparison" showing `Transposition` recovering `"test"` from `"tset"` at
$`k = 1`$ where `Standard` cannot.

---

## Key takeaways

- A `liblevenshtein` program is **dictionary + algorithm + transducer**.
- `query(W, k)` yields **terms**; `query_with_distance(W, k)` yields **`{ term, distance }`**.
- Cost scales with the **number of near matches**, not the dictionary size, because the
  automaton is simulated lock-step with the trie and prunes dead branches immediately.
- `Algorithm::Standard` vs `Algorithm::Transposition` changes which edits count as one —
  nothing else.

---

Next: [02 · Dictionaries →](../02-dictionaries/README.md)

[← Documentation Index](../../README.md)
