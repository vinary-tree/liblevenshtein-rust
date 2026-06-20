# Concepts

**Foundational concepts and mental models for understanding liblevenshtein-rust.**

This directory collects conceptual explainers that clarify *how to think about*
the library rather than how to call a specific API. They establish vocabulary
and intuitions that the rest of the documentation builds on. Start here if a
term used elsewhere (for example, "lazy" versus "eager" automata) is unfamiliar.

## Documents

| Document | Purpose |
|----------|---------|
| [LAZY_VS_EAGER_AUTOMATA.md](LAZY_VS_EAGER_AUTOMATA.md) | Explains the distinction between lazy (on-demand, simulated) and eager (precomputed) Levenshtein automata, when each applies, and the performance trade-offs. |

### Key idea

A query is a lazy *simulation* of a parameterized Levenshtein automaton walked
lock-step with the dictionary; positions track `(term_index, num_errors, is_special)`
and are pruned online by subsumption. Understanding lazy-vs-eager evaluation is
the prerequisite for reasoning about the query iterators in
[Layer 3](../algorithms/03-intersection-traversal/README.md) and the automata in
[Layer 2](../algorithms/02-levenshtein-automata/README.md).

**Status: Living reference.**

[← Documentation Index](../README.md)
