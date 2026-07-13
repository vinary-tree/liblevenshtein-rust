# Archived: SCDAWG deep-dive chapters (superseded by libdictenstein)

**Status: Archived — historical reference, not maintained.**

The canonical, maintained SCDAWG theory now lives in the **libdictenstein** crate. For how SCDAWG
integrates with liblevenshtein, see the live pointer [`docs/theory/scdawg/README.md`](../../../theory/scdawg/README.md).
These chapters are preserved verbatim from before the 2026-07-12 documentation trim (they use the
project's earlier notation and are not re-conformed):

1. [01-introduction](01-introduction.md) — problem motivation: why substring indices
2. [02-suffix-automaton](02-suffix-automaton.md) — equivalence classes, suffix links, end-positions
3. [03-cdawg](03-cdawg.md) — Compact DAWG: compaction and primary/secondary edges
4. [04-scdawg](04-scdawg.md) — Symmetric Compact DAWG: left extensions and prime subwords
5. [05-construction](05-construction.md) — on-line construction with sext links
6. [06-operations](06-operations.md) — substring search, bidirectional extension, IS features
7. [07-references](07-references.md) — annotated bibliography

See [`../README.md`](../README.md) for the archival rationale.
