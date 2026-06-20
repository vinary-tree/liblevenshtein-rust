# DSL Grammar Reference

liblevenshtein ships two small domain-specific languages for phonetic matching,
plus the phonetic-regex sublanguage they share. Each is specified as an
[EBNF](#ebnf-notation) grammar in this directory and compiled ahead of time by
the `src/phonetic/` runtime.

| File | Language | Extension | Purpose | Implemented by |
|---|---|---|---|---|
| [`llev.ebnf`](llev.ebnf) | LLev | `.llev` | phonetic **rewrite-rule** sets | `src/phonetic/llev/{lexer,parser}.rs` |
| [`llre.ebnf`](llre.ebnf) | LLRE | `.llre` | a single phonetic **regex pattern** + metadata | `src/phonetic/llre/{parser,loader,compiled}.rs` |
| [`regex.ebnf`](regex.ebnf) | phonetic regex | — | the regex sublanguage both reuse | `src/phonetic/regex/{lexer,parser}.rs` |

## EBNF notation

The grammars use ISO Extended Backus–Naur Form: `=` defines a production, `,`
concatenates, `|` alternates, `{ … }` means zero-or-more repetition, `[ … ]`
means optional, `( … )` groups, and `"…"` is a terminal. `IDENTIFIER`, `STRING`,
and `NEWLINE` are lexical tokens produced by the lexer.

## 1 · LLev — phonetic rewrite rules (`.llev`)

A `.llev` file is a sequence of directives and rule definitions
(`llev.ebnf`):

```ebnf
llev_file       = { directive | rule_definition | NEWLINE } ;
rule_definition = [ metadata_block ] , rewrite_rule , [ terminator ] ;
rewrite_rule    = pattern , "->" , replacement , [ context ] , [ weight ] ;
```

A **rewrite rule** reads `pattern -> replacement / context`: rewrite `pattern`
to `replacement` when it occurs in `context` (where `#` denotes a word
boundary and `_` marks the rewrite site). The optional `metadata_block`
(`[ id: …, name: …, weight: …, group: … ]`) tags the rule. Directives
(`@name`, `@version`, `@author`, `@include`, `@define`) provide file metadata and
macros.

**Worked example** (from `examples/phonetic_spellcheck/rules/homophones.llev`):

```llev
@version "1.0.0";

// spelled-out letter name → letter, only as a whole word ( #_# )
[id: 300, name: "oh to o", weight: 0.0, group: letter_homophones]
oh -> o / #_# ;

// normalise the "to / too / two" family to "to"
[id: 310, name: "too to to", weight: 0.0, group: homophones]
too -> to / #_# ;
```

Applied to a term, the rule-set rewrites it to a canonical phonetic form; the
`PhoneticNormalizedDictionary` then fuzzy-matches against those normalised forms,
so a query for `"too"` collides with the canonical `"to"`. Rules ship for 53
languages under [`data/rules/`](../../data/rules/).

### Compilation pipeline

![.llev compilation: source → lexer → AST → ruleset → compiled rules applied by apply_rules_seq.](../diagrams/phonetic/llev-compilation.svg)

The loader compiles a `.llev` file lexer → AST → ruleset → compiled form, after
which `apply_rules_seq` applies the rules to a term. See the
[phonetic-rules developer guide](../guides/phonetic-rules-developer-guide.md) for
authoring details.

## 2 · LLRE — a phonetic regex file (`.llre`)

A `.llre` file holds optional directives followed by a single regex pattern
(`llre.ebnf`):

```ebnf
llre_file = { directive | NEWLINE | comment } , pattern , [ NEWLINE ] , [ comment ] ;
directive = "@" , directive_name , directive_value , NEWLINE ;
comment   = "#" , { ANY_CHAR - NEWLINE } , NEWLINE ;
pattern   = regex ;   (* the regex sublanguage — see §3 *)
```

Directives (`@name`, `@version`, `@flags`, `@import`, …) configure the pattern;
`@import` pulls in shared definitions resolved by the loader. The pattern itself
is the phonetic-regex sublanguage of §3.

### Compilation pipeline

![.llre compilation: source → lexer → parser → AST → symbol expander → NFA compiler → NFA.](../diagrams/phonetic/llre-compilation.svg)

A `.llre` pattern is compiled source → lexer → parser → AST → symbol-expander →
NFA-compiler → NFA. Because the result is an NFA simulated in linear time
(Thompson/Glushkov construction), matching is **ReDoS-resistant by construction** —
see [Security](../SECURITY.md#2--the-llre--regex-dsl--redos-resistant-by-construction).

## 3 · The phonetic-regex sublanguage

Both DSLs build on a regex grammar that extends ordinary regular expressions with
phonetic features (`regex.ebnf`):

```ebnf
regex        = alternation ;
rewrite_rule = pattern , "->" , replacement , [ context ] , [ weight ] ;
pattern      = alternation ;
replacement  = alternation | (* empty — deletion *) ;
```

In addition to the usual operators (alternation `|`, concatenation, repetition),
it provides phonetic character classes and feature predicates used by the rewrite
rules above. An empty `replacement` expresses deletion. The full production set
is in [`regex.ebnf`](regex.ebnf).

## See also

- [Guides → Phonetic-rules developer guide](../guides/phonetic-rules-developer-guide.md)
- [Guides → Compositional phonetic + Levenshtein](../guides/compositional-phonetic-levenshtein.md)
- [LLRE reference](../llre/README.md)
- [Architecture Overview → phonetic DSL layer](../architecture/overview.md#4--the-phonetic-dsl-layer)
- [GLOSSARY → `.llev`, `.llre`, Thompson Construction](../GLOSSARY.md#terminology-added-since-2025-phonetics--time-series--automaton-variants--dsls--verification)

---

[← Documentation Index](../README.md)
