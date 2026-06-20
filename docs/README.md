# Documentation Index

Complete documentation for **liblevenshtein-rust v0.9.1** — Levenshtein and
related (phonetic, time-series) automata for error-tolerant search over strings
and byte arrays, with several trie/DAWG dictionaries, fuzzy maps, and fuzzy
caches.

**Last Updated:** 2026-06-19  ·  **Version:** 0.9.1

![Documentation map: the nine sections of this documentation set.](diagrams/architectures/documentation-map.svg)

---

## How this documentation is organized

The set is grouped into nine sections, from first-contact tutorials through deep
theory, formal proofs, and the project's scientific research record. The
[main README](../README.md) is the canonical, quality-reference overview; this
index maps everything else.

### Document conventions

- **Math** is written in Unicode and wrapped in backticks — every token,
  including single variables (`` `k` ``, `` `χ` ``, `` `𝒪(∣W∣)` ``). The bar /
  "such that" symbol is Unicode `∣` (U+2223), never an ASCII `|`. (Markdown table
  delimiters and code-fence pipes are left as plain `|`.)
- **Diagrams** live under [`diagrams/`](diagrams/README.md) as text source + a committed
  SVG, fully coloured per a [shared legend](diagrams/README.md); docs embed the SVG.
- **Citations** link to DOIs where one exists.

### Living vs. Historical — the rule that bounds edits

> A document is **LIVING** if it describes the *current* behaviour, API, theory,
> or architecture of liblevenshtein v0.9.1 — something you would consult to *use
> or extend the library today*. A document is **HISTORICAL** if it is a dated
> record of *how we got here*: a scientific ledger, hypothesis log, experiment
> record, phase/session/completion report, or benchmark dump. Per the project's
> append-only scientific-method practice, **HISTORICAL documents are never
> rewritten or "cleaned up"** — only indexed and cross-linked. When in doubt, a
> doc whose path carries a date, phase, hypothesis, or session name is HISTORICAL.

---

## 1 · Getting Started

- [Main README](../README.md) — overview, installation, quick start, the full feature tour.
- [User Guide → Getting Started](user-guide/getting-started.md) — first dictionary and query.
- [CHANGELOG](../CHANGELOG.md) — version history (current: 0.9.1).
- [Examples & Tutorials](examples/README.md) — the numbered tutorial series
  ([01](examples/01-getting-started/README.md) … [08](examples/08-real-world/README.md)) and a map to the runnable `examples/*.rs`.

## 2 · Concepts & Theory

- [Lazy vs. Eager Automata](concepts/LAZY_VS_EAGER_AUTOMATA.md) — the central idea: a query *lazily simulates* a parameterized Levenshtein automaton, it is **not** a precompiled universal DFA.
- [Levenshtein-automata theory](research/levenshtein-automata/README.md) — the Schulz–Mihov method, glossary, and code-to-paper mapping (theory home; also cross-linked from the glossary).
- [Algorithm layer 02 — Levenshtein automata](algorithms/02-levenshtein-automata/README.md) — the position/subsumption model, with diagrams.
- [Theory](theory/) — disk-trie and SCDAWG theory pointers (backend internals now live in `libdictenstein`).
- Specialized theory: [universal automata](research/universal-levenshtein/README.md) · [weighted automata](research/weighted-levenshtein-automata/README.md) · [bimachines](research/bimachines/README.md).

## 3 · Architecture

- [Architecture Overview](architecture/overview.md) — the **inter-crate** view: liblevenshtein ↔ libdictenstein ↔ optional duallity (WFST) ↔ the `.llev`/`.llre` DSL layer.
- [Developer Guide → Architecture](developer-guide/architecture.md) — the **intra-crate** module design and traits.
- [Design specifications](design/README.md) — feature-level designs (dynamic DAWG, suffix automaton, contextual completion, protobuf serialization, grammar correction, …).
- [Algorithm Reference layers 01–09](algorithms/README.md) — the layered architecture, bottom-up.
- Diagrams: [crate boundary](diagrams/architectures/crate-boundary.svg) · [component stack](diagrams/architectures/component-stack.svg) · [C4 context](diagrams/architectures/c4-context.svg) / [container](diagrams/architectures/c4-container.svg) · [feature-flag DAG](diagrams/architectures/feature-flag-dag.svg) · [module dependencies](diagrams/architectures/module-dependency.svg).

## 4 · User Guide

- [User Guide](user-guide/README.md) — getting-started, [algorithms](user-guide/algorithms.md), [backends](user-guide/backends.md), [serialization](user-guide/serialization.md), [features](user-guide/features.md), [code completion](user-guide/code-completion.md), [thread safety](user-guide/thread-safety.md), [prefix zipper](user-guide/prefix-zipper-usage.md).
- [Guides](guides/README.md) — [articulatory distance](guides/articulatory-distance.md), [compositional phonetic + Levenshtein](guides/compositional-phonetic-levenshtein.md), [phonetic-rules developer guide](guides/phonetic-rules-developer-guide.md), [hierarchical scope completion](guides/HIERARCHICAL_SCOPE_COMPLETION.md), [restricted substitutions](guides/RESTRICTED_SUBSTITUTIONS_GUIDE.md), [grammar correction](guides/grammar-correction/README.md).
- [Phonetic extraction](phonetic-extraction/README.md) — Soundex, Metaphone, NYSIIS, Caverphone, Cologne, Daitch–Mokotoff, Beider–Morse.
- [DSL grammar reference](grammar/README.md) — the `.llev`, `.llre`, and regex EBNF grammars with prose.
- [LLRE reference](llre/README.md).

## 5 · Developer Guide

- [Developer Guide](developer-guide/README.md) — [building](developer-guide/building.md), [contributing](developer-guide/contributing.md), [performance](developer-guide/performance.md), [publishing](developer-guide/publishing.md).
- [Security & threat model](SECURITY.md) — untrusted-input surfaces (grep archive/document extraction, FFI/WASM boundaries, serialization, `.llre` ReDoS-resistance).
- [Migration](migration/README.md) — terminology and version-migration notes (including the libdictenstein extraction).
- [Development logs](development/README.md) — phase/session implementation logs *(historical)*.

## 6 · Algorithm Reference

- [Algorithm Reference](algorithms/README.md) — the [documentation index](algorithms/DOCUMENTATION_INDEX.md) and the nine layered READMEs: [01 dictionary](algorithms/01-dictionary-layer/README.md) · [02 Levenshtein automata](algorithms/02-levenshtein-automata/README.md) · [03 intersection traversal](algorithms/03-intersection-traversal/README.md) · [04 distance](algorithms/04-distance-calculation/README.md) · [05 SIMD](algorithms/05-simd-optimization/README.md) · [06 zipper navigation](algorithms/06-zipper-navigation/README.md) · [07 contextual completion](algorithms/07-contextual-completion/README.md) · [08 caching](algorithms/08-caching-layer/README.md) · [09 value storage](algorithms/09-value-storage/README.md).

## 7 · Formal Verification

- [Verification](verification/README.md) — the formal-proof artifacts. **`FORMAL_VERIFICATION_MANIFEST.tsv` is the declared source of truth** for trusted/partial/legacy status; see [INDEX](verification/INDEX.md) and [README_FORMAL_GATES](verification/README_FORMAL_GATES.md). Holds the Rocq (`.v`) theories (core, articulatory, msm, phonetic, wallbreaker, grammar, llre, myers, product) and TLA+ specs ([`tla/`](verification/tla/README.md)).
- [Formal-verification writeups](formal-verification/README.md) — the parallel markdown proof exposition and findings (defers to the manifest for canonical status).

## 8 · Research & Scientific Ledgers — *historical, append-only (indexed, not edited)*

The project keeps an append-only scientific record. These are preserved as
written; they are indexed and cross-linked but never rewritten.

- [Research](research/README.md) — per-topic investigations: levenshtein-automata, universal-levenshtein, weighted-levenshtein-automata, [wallbreaker](research/wallbreaker/README.md), [artrie](research/artrie/README.md), simd-optimization, comparative-analysis, bimachines, eviction-wrapper, grammar-correction, phonetic-corrections, batch-processing.
- [Scientific ledgers](scientific-ledger/README.md) — the canonical ledger home (automata/WFST evaluation, MSM automata evaluation).
- [Optimization journals](optimization/README.md) and [optimization results](optimizations/README.md) — hypothesis ledgers (H1/H2…), per-topic experiment logs.
- [Universal](universal/README.md) · [Generalized](generalized/README.md) — phase records for the universal & generalized automaton work.
- [MeTTaIL](mettail/README.md) — semantic type-checking for MeTTa (a large self-contained subtree: theoretical foundations, correction-WFST, simplification, implementation, ecosystem).
- [Time-series analysis](time_series/README.md) · [Integration](integration/) (MORK, PathMap) · [Benchmarks](benchmarks/README.md) · [Analysis](analysis/) · [Bug reports](bug-reports/README.md) · [Completion reports](completion-reports/README.md) · [Implementation status](implementation-status/README.md) · [Archive](archive/README.md).

## 9 · Diagrams

- [Diagram suite & style guide](diagrams/README.md) — 49 fully-coloured diagrams (source + committed SVG) built from the pgmcp diagramming catalog (PlantUML, Graphviz, D2, Structurizr, Pikchr, Asymptote), with a [shared colour legend](diagrams/_legend/color-legend.svg) and a reproducible [render pipeline](diagrams/render.sh).

---

## Glossary

- [Technical Glossary](GLOSSARY.md) — implementation, performance, and user-facing terms.
- [Levenshtein-automata theory glossary](research/levenshtein-automata/glossary.md) — Position, Subsumption, Characteristic Vector, and other theoretical terms.

---

**Navigation:** [← Main README](../README.md) · [User Guide](user-guide/README.md) · [Developer Guide](developer-guide/README.md) · [Algorithm Reference](algorithms/README.md) · [Diagrams](diagrams/README.md)
