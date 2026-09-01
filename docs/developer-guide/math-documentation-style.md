# Mathematical documentation style and enforcement

Liblevenshtein uses GitHub’s MathJax extension for mathematical prose. The
delimiter order is part of the syntax, not a cosmetic convention: dollar signs
surround a single backtick span. Correct inline mathematics therefore has the
form $`d(x,y)\le k`$. A display formula uses a fenced block whose info string is
`math`:

```math
D(x,y)=\min_{\pi\in\mathcal{A}(x,y)} C(\pi).
```

Here $`x`$ and $`y`$ are operands, $`\mathcal{A}(x,y)`$ is their set of lawful
alignments, $`\pi`$ is one alignment, and $`C(\pi)`$ is its cost. Defining the
symbols next to the example keeps the notation readable without requiring
prior knowledge of a particular distance kernel.

## Why delimiter order matters

GitHub first parses CommonMark and then sends recognized mathematics to
MathJax. A dollar-delimited backtick span survives both stages. Bare dollar
math can lose Markdown escapes before MathJax sees it, while a code span that
contains the dollars renders inert monospace text. The following source
examples show the accepted and rejected byte arrangements without asking the
Markdown renderer to interpret them:

```text
accepted inline:       $`d(x,y)\le k`$
rejected bare dollar:  $d(x,y)\le k$
rejected transposed:   `$d(x,y)\le k$`
rejected stray opener: `$`d(x,y)\le k`$
rejected stray closer: $`d(x,y)\le k`$`
```

A literal dollar sign belongs in an ordinary code span, as in `$`. Currency,
regular-expression anchors, shell variables, and fenced source examples are
not mathematical prose. Mathematical symbols must use LaTeX commands inside a
math span: write $`x\le y`$, not a Unicode relation in prose or code styling.
The U+00B5 MICRO SIGN in a duration such as `10 µs` remains a unit symbol and
is deliberately not treated as Greek $`\mu`$.

## Repository classification

The gate distinguishes three Markdown classes:

| Class | Registry or rule | Enforcement |
|---|---|---|
| Living | `docs/.mathlint-include.txt` | Complete delimiter, Unicode-formula, complexity, and table checks |
| Legacy | `docs/.mathlint-legacy.txt` | Explicitly inventoried but not normative; promotion to living is reviewed |
| Append-only evidence | Path rules in `scripts/doc-math-prescan.raku` | Original bytes remain immutable; a dated erratum records corrections |

Every repository Markdown path must belong to exactly one of these classes. A
new path that appears in none fails as `unclassified-markdown`. This prevents a
new documentation subtree from escaping review merely because an allow-list
was not updated.

Rustdoc has two enforcement levels. Every discovered Rust source is checked
for byte-level delimiter corruption. The math-heavy automaton and temporal
kernel modules registered in `@STRICT-RUSTDOC-PATHS` additionally receive the
full prose checks. This distinction avoids treating phonetic letters, natural
language alphabets, or source examples as formulae while preserving strict
coverage where Rustdoc states mathematical contracts.

## Scanner model

`scripts/doc-math-prescan.raku` performs these operations in order:

1. Extract Markdown lines or rendered `//!`, `///`, `/*!`, and `/**` Rustdoc
   lines while preserving source line numbers.
2. Track source and math fences so examples are not parsed as prose.
3. Recognize valid inline spans and reject a stray backtick on either boundary.
4. Tokenize remaining inline code and reject dollar math nested inside code.
5. Detect bare dollar math, obsolete double-dollar displays, Unicode formulae,
   and undelimited asymptotic notation on strict prose surfaces.
6. Validate Markdown table column counts after blanking code and math spans.
7. Fail if repository discovery finds unclassified Markdown.

The scanner is read-only. Its contract suite in
`scripts/test-doc-math-prescan.raku` uses static positive, negative, Rustdoc,
append-only, and unclassified-path fixtures. Static fixtures make every accepted
and rejected construct reviewable and keep the tests independent of temporary
storage.

## Contributor workflow

Run the complete gate before submitting documentation:

```bash
scripts/doc-mathlint.sh
```

For a focused check, pass files directly. If a historical record contains bad
notation, do not rewrite the record. Add a dated erratum under
`docs/scientific-ledger/`, cite the affected path and scope, give the corrected
reading, and leave the evidence bytes intact. This preserves provenance while
ensuring that current readers have an unambiguous mathematical interpretation.
