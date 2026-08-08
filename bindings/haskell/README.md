# Haskell binding (Tier 3)

The Haskell package consumes a retained `DictionaryResource` from the separate
`vinary-tree-libdictenstein` package. `nextBatch` reads each leased native batch
directly and releases it before returning immutable Haskell values; `next` and
`foldBatches` provide incremental traversal without materializing the query.

The Hackage package is `vinary-tree-liblevenshtein`.
