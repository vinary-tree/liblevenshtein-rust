# Archived theory deep-dives (superseded by libdictenstein)

**Status: Archived — preserved reference, not maintained.**

These are the original in-repo deep-dive chapters for two data structures that were extracted to
the **[libdictenstein](https://github.com/vinary-tree/libdictenstein)** crate, which now owns their
canonical, maintained theory documentation. They were moved here (verbatim) during the 2026-07-12
documentation campaign when the live [`docs/theory/`](../../theory/) pages were trimmed to concise
**integration pointers**.

They are kept for historical reference; the authoritative, up-to-date treatment is in libdictenstein.

- [`scdawg/`](scdawg/) — Symmetric Compact DAWG: suffix-automaton → CDAWG → SCDAWG, on-line
  construction, operations, references. Canonical version:
  [`libdictenstein/docs/theory/scdawg/`](../../../../libdictenstein/docs/theory/scdawg/README.md).
- [`disk-tries/`](disk-tries/) — B-trie, Adaptive Radix Tree, persistent ART, buffer management,
  the hybrid PART design, benchmark results. Canonical version:
  [`libdictenstein/docs/theory/disk-tries/`](../../../../libdictenstein/docs/theory/disk-tries/README.md).

Live integration pointers: [`docs/theory/scdawg/README.md`](../../theory/scdawg/README.md) ·
[`docs/theory/disk-tries/README.md`](../../theory/disk-tries/README.md).
