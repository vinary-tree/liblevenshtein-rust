# SCDAWG — integration with liblevenshtein

> **Source of truth for the theory:** the complete, pedagogical treatment of the
> **Symmetric Compact Directed Acyclic Word Graph (SCDAWG)** — construction, complexity
> proofs, and operations — lives in the **[libdictenstein](https://github.com/f1r3fly-io/libdictenstein)**
> crate that owns the data structure:
> [`libdictenstein/docs/theory/scdawg/`](../../../../libdictenstein/docs/theory/scdawg/README.md).
> This page covers only **how liblevenshtein uses it**. The earlier in-repo deep-dive chapters
> duplicated that treatment and are preserved under
> [`docs/archive/theory/scdawg/`](../../archive/theory/README.md).

## What it is (in one paragraph)

An **SCDAWG** (also *C2S*, Compact Symmetric) is the most space-efficient index that supports
substring queries over a fixed text in `$\mathcal{O}(\lvert P\rvert)$` time for a pattern `$P$`,
**plus bidirectional navigation**: from the locus of a substring `$V$` it can extend to the right
(`$V\sigma$`) *or* to the left (`$\sigma V$`) by one symbol `$\sigma$`, and enumerate every
occurrence. It refines the suffix automaton / CDAWG of Blumer et al. [[1]](#references) with reverse
(left-extension) edges (Inenaga et al. [[2]](#references)). For a text of length `$n$` it has at
most `$n+1$` states and `$4n-4$` transitions in `$\mathcal{O}(n)$` space, all queries running in
`$\mathcal{O}(m)$` for a pattern of length `$m$`.

## Why liblevenshtein needs it

Two liblevenshtein capabilities rest on the SCDAWG, exposed through the `Scdawg` (byte / `u8`) and
`ScdawgChar` (Unicode scalar / `u32`) dictionary backends in the companion `libdictenstein` crate:

- **Substring / infix fuzzy search.** Unlike a prefix trie, the SCDAWG indexes *all* substrings, so
  a query can match anywhere inside a dictionary term.
- **WallBreaker (large error bounds).** The [WallBreaker](../../../README.md#wallbreaker-large-error-bounds)
  filter splits a long pattern into `$k+1$` (Standard) or `$2k+1$` (Transposition / MergeAndSplit)
  disjoint pieces; by the pigeonhole principle at least one piece survives error-free, is located
  exactly in `$\mathcal{O}(\lvert \text{piece}\rvert)$`, and is then **grown left and right** into a
  candidate. That left-and-right growth is exactly the SCDAWG's bidirectional-extension property —
  no other index in the toolbox provides it in both directions.

```text
                    ┌──────────────────────────────────────────────┐
   pattern  P  ───▶ │  split into k+1 disjoint pieces               │
                    └───────────────┬──────────────────────────────┘
                                    │  exact locate (pigeonhole survivor)
                                    ▼
                    ┌──────────────────────────────────────────────┐
        Scdawg ───▶ │  locus of piece  →  extend ← and →  →  candidate
                    └───────────────┬──────────────────────────────┘
                                    │  verify  d(P, cand) ≤ k
                                    ▼
                                 results
```

The bidirectional-growth soundness (the `$k+1$` / `$2k+1$` piece counts) is machine-checked,
admit-free, in `docs/verification/wallbreaker/.../WallBreakerPigeonhole.v`.

## Further reading

- **Full SCDAWG theory (canonical source):** [`libdictenstein/docs/theory/scdawg/`](../../../../libdictenstein/docs/theory/scdawg/README.md).
- WallBreaker in liblevenshtein: [`research/wallbreaker/`](../../research/wallbreaker/README.md).
- Archived in-repo deep-dive chapters (superseded by libdictenstein): [`docs/archive/theory/scdawg/`](../../archive/theory/README.md).

## References

1. A. Blumer, J. Blumer, D. Haussler, R. McConnell, and A. Ehrenfeucht. "Complete inverted files for efficient text retrieval and analysis." *Journal of the ACM*, 34(3):578–595, 1987. [doi:10.1145/28869.28873](https://doi.org/10.1145/28869.28873)
2. S. Inenaga, H. Hoshino, A. Shinohara, M. Takeda, S. Arikawa, G. Mauri, and G. Pavesi. "On-line construction of compact directed acyclic word graphs." *Discrete Applied Mathematics*, 146(2):156–179, 2005. [doi:10.1016/j.dam.2004.04.012](https://doi.org/10.1016/j.dam.2004.04.012)
