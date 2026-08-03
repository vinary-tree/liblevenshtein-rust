# Migrating to the split CLI in 0.10

Version 0.10 separates the reusable Rust library from the command application.
The split is intentionally breaking: there are no deprecated compatibility
modules in the library.

## Library consumers

Continue to depend on:

```toml
[dependencies]
liblevenshtein = "0.10"
```

The `cli`, `grep-compression`, `grep-archives`, `grep-documents`, and
`grep-full` features no longer exist. Application-only packages—including
`clap`, `rustyline`, archive codecs, XML/office parsers, PDF extraction, and
OCR bindings—are absent from the library's normal dependency graph.

The reusable in-memory phonetic matchers remain available under
`liblevenshtein::phonetic`; `parallel-grep` still enables their Rayon-backed
parallel paths.

## Command users

Version 0.10.0 is not published yet. For now, build the sibling
[`liblevenshtein-rust-cli`](https://github.com/vinary-tree/liblevenshtein-rust-cli)
checkout. After the coordinated library-first release, install the new package:

```bash
cargo install liblevenshtein-cli
```

The executable is still named `liblevenshtein`, so scripts invoking it do not
need a command-name change. Source builds move to the
`liblevenshtein-rust-cli` repository. Its pure-Rust compression, archive, and
document support is enabled by default; OCR remains opt-in.

## Removed library API

`MittonCorpus::load_birkbeck_zip` was a ZIP fixture loader rather than a
matching primitive and has been removed with the archive dependency.

## Documentation ownership

Documentation follows the code and dependency boundary:

| Subject | Owner after 0.10 |
|---|---|
| Reusable automata, edit distance, phonetic engines, dictionary integration, serialization APIs, WASM, and FFI | `liblevenshtein-rust/docs` |
| Executable syntax, dictionary commands, REPL commands, grep ingestion, archives, compression, document extraction, OCR, OS packages, and CLI security | `liblevenshtein-rust-cli/docs` |
| Historical CLI/REPL implementation completion report | `liblevenshtein-rust-cli/docs/completion-reports` |
| Grep ingestion diagram | `liblevenshtein-rust-cli/docs/diagrams` |

Library pages retain short cross-links where a feature has both a reusable API
and an application operation. The application repository owns the operational
examples because only it can keep them synchronized with the executable parser.
