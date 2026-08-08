# Vinary Tree JavaScript runtime

`@vinary-tree/vinary-tree` is the single-instance JavaScript runtime for the
related Vinary Tree libraries. It statically combines `libdictenstein`,
`liblevenshtein`, `lling-llang`, and `duallity` so resource handles can cross
project boundaries without copying or loading multiple native runtimes.

Use the package root for Node's native N-API backend, `@vinary-tree/vinary-tree/wasm`
for browsers, or `@vinary-tree/vinary-tree/wasi` for Node/WASI applications that
need WASI filesystem access. The project-specific packages provide the
idiomatic JavaScript, TypeScript, and ClojureScript facades.

Query cursors stream bounded batches and retain query-start snapshot semantics:
mutating or closing a dictionary after creating a cursor does not change that
cursor's remaining results.

The package is licensed under Apache-2.0. Source and release documentation are
available from the [Vinary Tree liblevenshtein repository](https://github.com/vinary-tree/liblevenshtein-rust).
