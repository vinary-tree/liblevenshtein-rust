# Architecture

How liblevenshtein is structured, at two levels:

- **[Architecture Overview](overview.md)** — the *inter-crate* view: how
  liblevenshtein relates to [`libdictenstein`](overview.md#1--the-crate-boundary)
  (dictionary backends), the optional `duallity` WFST integration, and the
  `.llev`/`.llre` DSL layer, with the crate-boundary diagram and the 0.9.0
  extraction story.
- **[Developer Guide → Architecture](../developer-guide/architecture.md)** — the
  *intra-crate* view: the `src/` module map, core traits, design principles,
  concurrency model, and the diagrams for each.

See also the [Algorithm Reference](../algorithms/README.md) for the layered (01–09)
architecture, bottom-up, and the [Diagrams](../diagrams/README.md) suite.

---

[← Documentation Index](../README.md)
