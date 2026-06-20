# Guides

**Task-oriented how-to guides for liblevenshtein-rust features.**

These guides walk through using specific capabilities of the library end-to-end,
with worked examples and configuration recipes. They complement the
[User Guide](../user-guide/) (broad onboarding) and the
[Algorithm Documentation](../algorithms/) (theory and internals) by focusing on
concrete tasks: phonetically-informed correction, restricted substitution sets,
and hierarchical scope-aware completion.

## Guides

| Document | Purpose |
|----------|---------|
| [articulatory-distance.md](articulatory-distance.md) | Phonetically-informed substitution costs derived from articulatory features, so acoustically-near errors cost less than `1`. |
| [compositional-phonetic-levenshtein.md](compositional-phonetic-levenshtein.md) | Compose phonetic NFAs with Levenshtein automata for spelling correction that respects pronunciation. |
| [HIERARCHICAL_SCOPE_COMPLETION.md](HIERARCHICAL_SCOPE_COMPLETION.md) | Build hierarchical, lexical-scope-aware completion using the contextual completion layer. |
| [phonetic-rules-developer-guide.md](phonetic-rules-developer-guide.md) | Author, register, and apply phonetic rewrite rules as a developer. |
| [RESTRICTED_SUBSTITUTIONS_GUIDE.md](RESTRICTED_SUBSTITUTIONS_GUIDE.md) | Constrain which character substitutions an automaton permits via a substitution policy. |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| [grammar-correction/](grammar-correction/) | Guides for implementing grammar-correction guarantees on top of the correction stack. |

**Status: Living reference.**

[← Documentation Index](../README.md)
