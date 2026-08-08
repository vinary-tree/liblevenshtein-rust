# Vinary Tree liblevenshtein for Fortran

The Fortran 2018 module exposes all distance families, Unicode/raw-byte/u64
streaming queries, retained dictionary construction, and phonetic automata. A
`query_iterator` leases one bounded native batch and returns one owned
`levenshtein_match` at a time. Finalizers are a safety net; call `close` when
deterministic release matters.

The fpm package is named `vinary-tree-liblevenshtein` and depends on the shared
`vinary-tree-interop` module. Link with `-lliblevenshtein`; CMake installations
let applications choose shared or static linkage.
