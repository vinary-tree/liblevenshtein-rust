# Migration Guides

**Guides for migrating across API and terminology changes in liblevenshtein-rust.**

This directory documents breaking or renaming changes between versions so that
existing callers can update their code with minimal friction. Each guide maps
old names and patterns to their current equivalents and explains the rationale
for the change.

## Guides

| Document | Purpose |
|----------|---------|
| [LAZY_EAGER_TERMINOLOGY.md](LAZY_EAGER_TERMINOLOGY.md) | Migration guide for the lazy/eager automaton terminology rename: old → new names and how to update call sites. |

For the underlying conceptual distinction, see
[../concepts/LAZY_VS_EAGER_AUTOMATA.md](../concepts/LAZY_VS_EAGER_AUTOMATA.md).

**Status: Living reference.**

[← Documentation Index](../README.md)
