# OCaml binding (Tier 3)

The OCaml 5 binding consumes retained `Vinary_tree_interop.resource` values
published by the separate `vinary_tree_libdictenstein` package. Native query
cursors are lazy and snapshot-stable; `to_seq` pulls one match at a time and
`fold_batches` crosses the FFI once per bounded batch.

The opam package is `vinary_tree_liblevenshtein`.
