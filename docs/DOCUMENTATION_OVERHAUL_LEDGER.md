# Documentation Overhaul Ledger

A scientific record of the documentation overhaul that brings `liblevenshtein`
into conformance with the **pgmcp documentation guidelines** (23 rules across
placement, coverage, pedagogy, diagrams, math notation, citations, and
algorithms), including a fully-coloured diagram suite built from the pgmcp
diagramming catalog.

- **Branch:** `docs/pgmcp-guidelines-overhaul`
- **Started:** 2026-06-19
- **Plan of record:** `~/.claude/plans/utilize-the-documentation-guidelines-humble-parasol.md`
- **Diagram conventions:** `docs/diagrams/README.md`

## Scope (user-confirmed 2026-06-19)

Maximal breadth: overhaul living docs **and** apply conformance to
`research/` · `theory/` · `mettail/` write-ups; build the **full 37-diagram
suite** + render pipeline; **approved batch structural cleanup** (manifest
re-listed for final confirm before execution); **author 8 new tutorials**.

### Append-only integrity rule

Dated scientific ledgers, hypothesis logs, experiment records, completion/phase
reports, and benchmark dumps are an **append-only** record. For these we add only
cross-links + diagrams and fix pure *notation* (backtick math, Unicode, broken
links) — we **never** alter recorded measurements, hypotheses, verdicts,
p-values, dates, or conclusions. `.v` Rocq theories and `.tla` specs are never
reworded. Prose-rewrites apply only to *explanatory/reference* documents.

## Status

| Phase | Workstream | Status | Notes |
|---|---|---|---|
| P0 | Diagram infrastructure & ledger | ✅ done | render.sh / Makefile / style guide / legend / this ledger |
| P1 | Index & navigation | ✅ done | docs/README.md rewritten · GLOSSARY.md refreshed (+35 terms) · 38 directory READMEs |
| P2 | 37-diagram suite | ✅ done | **49 diagrams** (PlantUML 25 · Graphviz 11 · D2 7 · Structurizr 2 · Pikchr 2 · Asymptote 2); all render, validate, in sync |
| P3 | Living-doc conformance (~45 files) | ✅ done | README +4 embeds · architecture/getting-started/concepts · user-guide (7) · guides (5) · phonetic-extraction (7) · algorithms layers (9) · design (15) · theory · dev-guide (4) |
| P4 | Coverage docs + 8 tutorials | ✅ done | SECURITY.md · architecture/overview.md · grammar/README.md · 8 tutorials + examples index |
| P5 | research/theory/mettail conformance | ✅ done | 35 explanatory docs conformed (𝒪() notation, DOIs incl. a corrected TCS-2011 DOI, backlinks); dated ledgers preserved (append-only) |
| P6 | rustdoc additions | ✅ done | cli/ (5) · repl/ (2) · 7 leaves; `cargo doc` → **0 missing_docs** |
| P7 | Structural cleanup + verification | ✅ done | cleanup ✅ (24 archived, 3 empty dirs removed); **all runnable gates green** — links 0 errors, diagrams 49/49, rustdoc 0 missing_docs, versions current |

**Overhaul complete (2026-06-19).** All eight phases done; see the closing summary at the end of the log.

## Verification gates

| Gate | Command | Status |
|---|---|---|
| rustdoc clean | `cargo doc --no-deps --all-features` → 0 `missing_docs` | ✅ (0 missing_docs; the 72 link/HTML warnings are pre-existing, outside the overhaul scope) |
| diagrams in sync | `docs/diagrams/render.sh --check` | ✅ (49/49 in sync) |
| links resolve | `lychee --offline` over living docs | ✅ core set (0 errors); broader living-doc repair in pass |
| version gate | `grep -rnE '0\.[4-8]\.[0-9]' <living docs>` → only historical hits | ✅ (no stale current-version claims in living docs) |
| DOI validity | DOIs are canonical/correct; HTTP resolution **not checkable** in this sandbox (no outbound network — curl returns 000) |
| define-before-use | living-doc acronyms ⊆ GLOSSARY headwords | ✅ (GLOSSARY refreshed with phonetics/MSM/automata/DSL/verification vocabulary) |

---

## Log

### 2026-06-19 — P0 — Diagram infrastructure & ledger ✅

- Created `docs/diagrams/render.sh` — discovers every `.puml/.dsl/.d2/.dot/.mmd/.pikchr/.asy`
  source and renders a committed sibling SVG; per-tool invocations verified by
  scratch tests (all 7 renderers exit 0 with valid SVG). Modes: build / `<glob>` /
  `--check` / `--list`. Structurizr `.dsl` exports to PlantUML then SVG; Mermaid
  uses `puppeteer-config.json` (`--no-sandbox`).
- Created `docs/diagrams/Makefile` (incremental rebuild + `check` + per-subsystem targets).
- Created `docs/diagrams/README.md` — the style guide: SVG-commit rule, tool-selection
  policy (mirrors pgmcp), the extended per-concept colour legend, per-tool house-style
  blocks, authoring checklist, embedding rules.
- Created `docs/diagrams/_legend/color-legend.d2` → `color-legend.svg` (canonical legend).
- **Verification:** full render produced 5 valid SVGs (`xmllint --noout` clean);
  `render.sh --check` reports "all 5 SVG(s) are in sync". Re-rendering the 4 seed
  diagrams only stripped the redundant embedded `<?plantuml-src?>` metadata blob
  (`-nometadata`); diagram geometry is byte-identical — the pipeline is deterministic.

### 2026-06-19 — P2 — 37-diagram suite ✅  (delivered 49)

Authored 45 new diagram sources (+4 seed) across 13 subsystem directories, all
rendered to committed sibling SVGs and fully coloured per the legend:
- **architectures/** (7): component-stack, crate-boundary, documentation-map,
  c4-context, c4-container, feature-flag-dag, module-dependency.
- **automata/** (10): levenshtein-nfa, nfa-transposition, nfa-merge-split,
  operation-sets, position-set-state, subsumption-lattice, automaton-implementations,
  characteristic-vector, wallbreaker-pigeonhole, wallbreaker-scdawg-walk.
- **traversal/** (7): query-flow, lazy-simulation, lockstep-dfs-sequence,
  lockstep-dfs-flow, query-iterator-hierarchy, value-filtered-pruning, end-to-end-spellcheck.
- **dictionary-structures/** (6): backend-taxonomy, backend-decision-tree,
  dictionary-traits, dawg-vs-trie, scdawg-structure, pruning-simd-bloom.
- **concurrency/** (2), **distance/** (1), **phonetic/** (5), **time-series/** (4),
  **contextual/** (2), **cache/** (1), **serialization/** (1), **grep/** (1),
  **bindings/** (1), **_legend/** (1).
- **Tools used** (pgmcp catalog): PlantUML 25 · Graphviz 11 · D2 7 · Structurizr 2 ·
  Pikchr 2 · Asymptote 2.
- **Verification:** `render.sh --check` → "all 49 SVG(s) are in sync"; `xmllint --noout`
  clean on all 49; no PlantUML error-graphics; label spot-checks pass; Asymptote SVGs
  carry width/height/viewBox.
- **Fixes during build:** hardened render.sh to continue past a failing source;
  Structurizr needs every block property on its own line (positional tags, multi-line
  styles); corrected the style guide's invalid single-line `skinparam` example.
- 18 of the 45 new diagrams were authored by three parallel subagents from precise
  per-diagram specs + the colour legend, each grounding labels against the code via
  pgmcp and self-validating via `render.sh`; reviewed and QA'd on integration.

### 2026-06-19 — P1 — Index & navigation ✅

- Rewrote `docs/README.md` — current to 0.9.1, complete nine-section documentation map,
  embeds `documentation-map.svg`, states the Living-vs-Historical rule and house math style.
- Refreshed `docs/GLOSSARY.md` — added ~35 post-2025 terms (parameterized/universal/
  generalized automata, OperationSet/SubstitutionSet, Myers, WallBreaker, SCDAWG, IPA,
  articulatory distance, phonetic normalization + the 7 phonetic algorithms, NFA product,
  Thompson construction, `.llev`/`.llre`/EBNF, MSM, DTW, TimeSeriesIndex, SAX,
  libdictenstein/duallity/deprecation-shim, Rocq/TLA+); added the crate-boundary note;
  fixed all stale code-links (dictionary → libdictenstein note; `state_pool`→`pool`,
  `query_ordered`→`ordered_query`, `simd/`→`simd.rs`, `draft`→`draft_buffer`). Glossary
  link check: **0 errors**.
- 38 missing directory READMEs created by a subagent (9 living navigational, 25 historical
  preserved-record, 4 formal-artifact pointers to the manifest); all back-links verified.

### 2026-06-19 — P3 — Living-doc conformance ✅ (core)

- `developer-guide/architecture.md` — full rewrite to 0.9.1: libdictenstein two-crate story,
  current module map, SIMD moved out of "future", 5 broken links fixed, 6 diagrams embedded.
- `user-guide/getting-started.md` — version pins, fixed fictitious example filenames, current
  backend table, end-to-end diagram.
- `concepts/LAZY_VS_EAGER_AUTOMATA.md` — backticked math, Schulz–Mihov DOI, I/M-type & oracle
  definitions, three automaton diagrams, "restricted substitutions" updated to available.
- `README.md` — 8 diagrams now embedded (added crate-boundary, backend-decision-tree,
  operation-sets, feature-flag-dag).
- Subagents conformed: `user-guide/*` (7), `guides/*` (5), `phonetic-extraction/*` (7, with
  canonical citations incl. Daitch–Mokotoff), the 9 algorithm-layer READMEs (diagrams + math),
  `design/*` + `theory/*` (in pass). Each verified embeds resolve, versions current, back-links
  present. (Remaining: 4 dev-guide docs — building/performance/contributing/publishing — version
  bumps, after the link-repair pass.)

### 2026-06-19 — P4 — Coverage docs + tutorials ✅

- `docs/SECURITY.md` — threat model with per-surface trust-boundary table (grep archive/
  document extraction, serialization, FFI contract, WASM, `.llre` ReDoS-resistance).
- `docs/architecture/overview.md` — inter-crate view (liblevenshtein ↔ libdictenstein ↔
  optional duallity ↔ DSL layer), the 0.9.0 extraction story, crate-boundary diagram.
- `docs/grammar/README.md` — `.llev`/`.llre`/regex EBNF reference with worked examples and the
  compilation-pipeline diagrams.
- 8 numbered tutorials (`docs/examples/01-08`) + an Examples & Tutorials index, each grounded in
  a real runnable `examples/*.rs` (29 API symbols verified), with embedded diagrams.

### 2026-06-19 — P5 — research/theory/mettail conformance ✅

- **35 explanatory docs** conformed across `research/levenshtein-automata/`,
  `research/universal-levenshtein/`, `research/weighted-levenshtein-automata/`,
  `research/bimachines/`, `research/comparative-analysis/`, and `mettail/{theoretical-foundations,
  reference,metta-ecosystem}/`: `O(...)`→`𝒪(...)` via a guarded raku transform (measurement/digit
  lines provably unchanged), backlinks, and verified DOIs (Schulz–Mihov, Wagner–Fischer [spelling
  corrected], Damerau, Ristad–Yianilos, Gerdjikov TCS-2019, Meredith–Radestock, …).
- **Citation-correctness fix:** a wrong TCS-2011 DOI/volume (`10.1016/j.tcs.2009.03.002`,
  "410(37-39)") propagated across 3 universal-levenshtein docs was corrected to the DBLP-verified
  **TCS 412(22):2340–2355, `10.1016/j.tcs.2011.01.013`**.
- `theory/` handled separately (design+theory pass). **Dated ledgers, hypothesis/experiment logs,
  phase/session/completion reports, benchmark dumps, and `.v`/`.tla` artifacts preserved verbatim**
  per the append-only rule (only pure-notation `𝒪()` where safe, no data/verdict/date altered).
  Whole tree indexed via the 38 READMEs + documentation-map Section 8.

### 2026-06-19 — P6 — rustdoc additions ✅

- Rich `//!` module headers: `src/cli/{mod,args,commands,detect,paths}.rs` (documenting the
  shared command keystone), `src/repl/{mod,highlighter}.rs`, and the 7 doc-less leaf files
  (`phonetic/named_classes/{phone_pattern,registry,lookup,algebra,tests}.rs`,
  `phonetic/regex/{parser,lexer}/tests.rs`).
- `cargo doc --no-deps --all-features` → **0 `missing_docs`**. (72 pre-existing intra-doc-link /
  HTML warnings in unrelated files are outside the overhaul scope; the one regression I
  introduced — a redundant explicit link target — was fixed.)

### 2026-06-19 — P7 — Structural cleanup + verification 🔄

- **Cleanup (approved batch):** `git mv` of 21 stray root `*.txt` benchmark dumps →
  `docs/archive/benchmarks/`, 2 stray `*.md` → `docs/analysis/fuzzy-maps/`, and
  `docs/README_ORG_MODE.md` → `docs/archive/`; removed 3 empty scaffolding dirs
  (`docs/deprecations`, `docs/references`, `docs/diagrams/performance`). Verified beforehand that
  no script/bench *reads* the moved dumps (`benchmark_optimizations.sh` only writes them). One
  resulting broken link (`HIERARCHICAL_SCOPE_COMPLETION.md`) repointed to the archive.
- **Gates:** diagram-sync ✅ (49/49 via `render.sh --check`); rustdoc ✅ (0 missing_docs);
  version ✅ (no stale current-version claims in living docs except the 4 dev-guide files, queued);
  links ✅ on the core set (0 errors), broader living-doc repair in pass. `markdownlint` and live
  DOI HTTP-resolution are not available in this sandbox (not installed / no outbound network);
  the DOIs used are canonical and verified-correct.
- **Living-doc link repair** (subagent): 99 broken relative links across 11 living files remapped
  to current targets or neutralized (unwritten sub-stubs) → living-doc link gate **0 errors**.
- **Dev-guide version bumps**: `building/performance/contributing/publishing.md` → 0.9.1.
- **design + theory conformance** (subagent): 40 files — 12 living design specs (each with an
  embedded diagram, backticked math, fixed links, back-links), 12 grammar-correction files
  (raku `𝒪()` pass + link-depth fixes), the theory READMEs + chapter files (math backticked,
  "moved to libdictenstein" pointers preserved). Dead `../../wfst/` forward-reference links
  (to a planned tree superseded by `mettail/correction-wfst/`) neutralized.

### 2026-06-19 — Closing summary ✅

The documentation overhaul is **complete** — all eight phases (P0–P7) done, conforming to the
pgmcp documentation guidelines.

- **Diagrams:** 49 fully-coloured diagrams (PlantUML 25 · Graphviz 11 · D2 7 · Structurizr 2 ·
  Pikchr 2 · Asymptote 2) from the pgmcp diagramming catalog, each source + committed SVG, behind
  a reproducible `render.sh`/Makefile pipeline, a style guide, and a shared colour legend.
- **Navigation:** `docs/README.md` rewritten to a complete nine-section map; `GLOSSARY.md`
  refreshed (+35 terms); 38 directory READMEs added.
- **Living-doc conformance:** README (8 embeds) · architecture (intra + new inter-crate overview) ·
  user-guide (8) · guides (6) · phonetic-extraction (8) · algorithm layers (9) · design (15) ·
  theory · developer-guide (5) — backticked Unicode math, defined terms, DOI'd citations, embedded
  diagrams, current versions, resolved links.
- **New coverage:** `SECURITY.md` threat model, `architecture/overview.md`, `grammar/README.md`
  DSL reference, 8 tutorials + examples index.
- **research/theory/mettail:** 35 explanatory docs conformed; all dated ledgers / `.v` / `.tla`
  artifacts preserved per the append-only rule.
- **rustdoc:** `cli/`, `repl/`, and 7 leaf modules documented → 0 `missing_docs`.
- **Cleanup:** 24 stray files archived, 3 empty dirs removed.

**Verification (runnable gates, all green):** living-doc links **0 errors** (1194 OK);
diagrams **49/49 in sync**; rustdoc **0 missing_docs**; version gate clean.
**Not runnable in this sandbox:** `markdownlint` (not installed); live DOI HTTP resolution
(no outbound network) — DOIs are canonical/verified-correct.

Work is on branch `docs/pgmcp-guidelines-overhaul`, uncommitted (commit on request).

### 2026-06-20 — Fix: D2 title/note clipping ✅

User-reported rendering bug: the D2 markdown title blocks (`title: |md  # … |`) were
emitted by D2 0.7.1 into an undersized `foreignObject` (e.g. 76px tall for an `<h1>`),
so long titles wrapped and the second line was vertically clipped (only glyph-tops
visible) — observed on `crate-boundary.svg` and `operation-sets.svg`. **Root cause:**
D2 underestimates the height of a wrapped markdown `# h1`. **Fix:** converted every D2
`title:`/`note:` markdown block (7 titles + 2 notes across all 7 `.d2` files) to a
native `shape: text` node, which the renderer sizes to its content. Result: **0
`foreignObject` blocks remain in any D2 SVG** — titles/notes are native `<text>`,
full-text, single-line, within the viewBox; all 49 diagrams re-rendered and back in
sync.

### 2026-06-20 — Fix: PersistentARTrie correctness + backend-family completeness ✅

User-reported: the backend diagrams (1) wrongly classified **PersistentARTrie as
static/immutable**, and (2) **omitted the disk-persisted family**. Verified against
libdictenstein: PersistentARTrie is `insert(&self)`/`remove(&self)`, `SyncStrategy::InternalSync`,
a **lock-free CAS overlay over memory-mapped disk** — i.e. *dynamic* and *disk-persisted*
("persistent" = non-volatile, NOT immutable). Enumerated the full `Dictionary` impl set:
in-memory (DoubleArrayTrie [static], DynamicDawg(/U64), SuffixAutomaton, Scdawg,
PathMapDictionary, BijectiveMap) + a disk-persisted mirror (PersistentARTrie(/U64),
PersistentScdawg, PersistentSuffixAutomaton, PersistentSuffixTree, PersistentVocabARTrie —
all dynamic, lock-free overlay). **Fixes:**
- `backend-decision-tree.dot` — rebuilt around two axes (access pattern × storage); all 12
  backend types present; PersistentARTrie + family under "disk-persisted, dynamic" (teal).
- `backend-taxonomy.puml` — split into In-memory and Disk-persisted packages; added the 5
  persistent types + BijectiveMap; corrected ART to dynamic.
- `concurrency-model.puml` — moved PersistentARTrie from the (wrong) "Persistent/immutable"
  group into "Wait-free / lock-free reads" (lock-free CAS); dropped the empty group; legend updated.
- `README.md` + `user-guide/getting-started.md` — corrected the "static backends" prose and the
  PersistentARTrie row (lock-free, dynamic), expanded the backend tables with the persistent
  family + BijectiveMap, and added a "persistent ≠ static" note.
- Verified: all 49 diagrams render + in sync; no error graphics; link gate 0 errors.

### 2026-06-20 — Repo-wide broken-link audit ✅

Scanned **all 573 markdown files** (custom resolver + `lychee --offline`):
- **176 directory links → `README.md`**: rewrote `](dir/)` / `](dir)` to `](dir/README.md)`
  across 53 files wherever the directory has a README (the index, layer cross-links, etc.).
- **44 broken links fixed** across ~22 files: 21 remaps (absolute `/docs/…`,`/src/…` → relative;
  wrong-depth `../../…` → `../../../…`, incl. the cross-repo `MORK`/`PathMap` sibling refs and
  `src/transducer/helpers.rs`; renames `PERFORMANCE.md`→`developer-guide/performance.md`,
  `ARCHITECTURE.md`→`developer-guide/architecture.md`, `FUTURE_ENHANCEMENTS.md`→`research/future-enhancements.md`,
  `phonetic_normalized/mod.rs`→`phonetic_normalized.rs`) and 20 neutralizations (dead `../wfst/…`
  forward-references to a never-created tree, the moved `pathmap.rs`, unwritten sub-pages).
- **Deliberately left untouched** (not real links): math notation in mettail category-theory docs
  (`[…](n, |(.(in(m…)`), rustdoc `crate::…` paths in a completion report, the fenced embedding
  *examples* in `diagrams/README.md`, and the gitignored generated `pkg/README.md`.
- **Result:** `lychee --offline docs` over the entire docs tree (3,344 links) → **0 errors**.

---

## Campaign 2 — 2026-07-12: MathJax migration + content refresh

A second overhaul campaign, opened after the pgmcp documentation guidelines evolved. This
section is **append-only** like the rest of this ledger; it records the campaign design,
the empirical baseline, and per-phase results as they land.

- **Branch:** `docs/mathjax-migration-campaign-2`
- **Plan of record:** `~/.claude/plans/rewrite-all-the-non-archived-synthetic-yeti.md`
- **Prompted by:** "Rewrite all the non-archived documentation under `docs/` to align with the
  documentation guidelines of pgmcp" + "update the documentation in the process; archive
  deprecated documentation that should not be rewritten; include the root `README.md`."

### Why (the reversal)

The 2026-06-19 overhaul (Campaign 1, above) deliberately standardized the corpus onto
**backticked Unicode-literal math** (`` `𝒪(∣W∣)` ``, bar = `∣` U+2223), and codified that
house style in `docs/README.md`. The pgmcp guidelines (26 rules;
`mcp__pgmcp__documentation_guidelines`) now require **MathJax LaTeX** instead — inline as a
backtick code span whose content is dollar-delimited (`` `$\mathcal{O}(\lvert W\rvert)$` ``)
and display as a fenced `math` block. Campaign 2 **reverses** the notation convention across
the living docs and, per the user's instruction, additionally refreshes drifted content and
archives deprecated docs.

### Empirical baseline (three parallel audits, 2026-07-12)

- **0** files use MathJax today; **150** files carry **1451** backticked-Unicode-math spans;
  **170** files carry ≈**1426** bare undelimited `O(...)`; genuine forbidden `$…$` is
  essentially absent (**1** real case — a BibTeX `$\pi$` in `mettail/reference/bibliography.md`).
- Content drift since Campaign 1 (20 `src/` commits): dictionary backends extracted to
  **`libdictenstein`** and the local re-exports now `#[deprecated]` (`src/lib.rs:185–265`),
  yet **31 living docs** still teach `liblevenshtein::dictionary::…`. Undocumented new
  features: `VersionedQueryCache`, the Phases 1–9 hardening campaign, bincode 1.3→2.0,
  MergeAndSplit-semantics correction.
- Diagram infra: **49** sources; **36** embed Unicode-literal math in labels; **0** use
  PlantUML `<latex>`.

### Decisions locked with the user (2026-07-12)

| # | Decision | Ruling |
|---|---|---|
| Q1 | Dated append-only records (~150–170 files) | **Leave untouched** — rewrite only living/explanatory docs. |
| Q2 | `docs/theory/` deep-dives duplicated in `libdictenstein` | **Trim to integration pointers.** |
| Q3 | Depth of the content refresh | **Full refresh** — fix deprecated imports, document new features, correct stale claims. |

### Append-only integrity rule (unchanged, restated)

Campaign 1's rule stands verbatim: dated scientific ledgers, hypothesis/experiment/phase/
session/completion reports, and benchmark dumps are **append-only** — never altered. `.v`
Rocq theories and `.tla` specs are never reworded. Campaign 2 tightens the earlier "fix
pure notation" latitude: per Q1, the **records are not touched at all** (not even for
notation); only the LIVING/explanatory set is rewritten. A `git diff` gate proves zero
changes to record paths.

### Tooling (P0)

- `scripts/doc-math-prescan.raku` — fence-aware scanner: `--key` prints the Unicode→LaTeX
  conversion key; default mode briefs a rewrite agent per file; `--lint` is the machine gate.
  Honors the guards (µ U+00B5 ≠ μ U+03BC; MeTTa `$var`/currency/regex `$`; code fences;
  table pipes). Validated: correctly flags 48 constructs in `README.md`, skips fenced-code
  math, ignores converted `` `$…$` `` spans and `` ```math `` blocks.
- `scripts/doc-mathlint.sh` — CI gate wrapping the scanner over the living-doc allow-list
  manifest `docs/.mathlint-include.txt`; PASS ⇔ 0 residual old-style math in living docs.

### Phase log

| Phase | Workstream | Status |
|---|---|---|
| P0 | Conventions (`docs/README.md` + `diagrams/README.md`), tooling, ledger | ✅ done |
| P1 | Root `README.md` flagship rewrite | 🔄 |
| P2 | Living-doc cluster rewrites (~180 files) | ⏳ |
| P3 | `theory/` trim to pointers | ⏳ |
| P4 | Diagram-label LaTeX (25 `.puml`) + new figures | ⏳ |
| P5 | Archive deprecated docs (Bucket C) | ⏳ |
| P6 | Verification gates + close-out | ⏳ |

#### 2026-07-12 — P0 — Conventions, tooling, ledger ✅

- Rewrote the `docs/README.md` "Document conventions → **Math**" bullet from the Unicode-
  backtick rule to the MathJax backtick-dollar + fenced-`math` convention (drops the "bar is
  `∣` never ASCII `|`" clause; adds the pseudocode-vs-`math`-block distinction).
- Added a "Math in labels" subsection to `docs/diagrams/README.md` §4 (PlantUML/Structurizr/
  Asymptote `<latex>`/`$…$`; Unicode fallback for DOT/D2/Pikchr/Mermaid), and converted the
  one `∩` in its own diagram-registry prose to `` `$\cap$` ``.
- Authored + validated the two tooling scripts above.
- This campaign section appended to the ledger (Campaign 1 untouched).

#### 2026-07-13 — P1/P3/P5 + P2 (partial) progress ✅🔄

- **P1 ✅** root `README.md`: 48 math spans → MathJax, MSM recurrence → a ```math `cases` block,
  pseudocode kept as literate fences, install pin `0.8`→`0.9`, exemplar import path corrected
  (`double_array_trie_char`→`double_array_trie`).
- **P3 ✅** `theory/{scdawg,disk-tries}/README.md` trimmed to libdictenstein integration pointers;
  the 14 duplicated deep-dive chapters `git mv`'d to `docs/archive/theory/` (indexed there).
- **P2 🔄 (partial)** conformed & scanner-verified (0 findings) via parallel subagents:
  `guides`+`grammar`+`llre`, `GLOSSARY.md`, `user-guide`+`concepts`+`architecture`, caching (algo 08),
  `developer-guide`+`examples`+`migration`, `phonetic-extraction`+`integration`, algorithm layers
  01–06 + 09 + indexes, `SECURITY.md`, `design/hierarchical-correction.md`. Subagents also fixed
  ~200 deprecated `liblevenshtein::dictionary::…` → `libdictenstein::…` imports, several µ-codepoint
  errors (U+03BC→U+00B5 for microseconds), stale versions, a wrong Inenaga DOI, and broken
  placeholder URLs. **Remaining (theory-heavy, bare-unicode/display-math — needs judgment):** mettail
  foundations+simplification+applied subtrees, `research/universal-levenshtein` explanatory subset,
  `research/levenshtein-automata/{glossary,implementation-mapping}`, `weighted` README, algorithm 07
  (contextual), grammar-correction (incl. `MAIN_DESIGN.md`), design leftovers, verification reference set.
- **libdictenstein canonical import paths** (verified against its `src/lib.rs` module re-exports):
  types are re-exported at module level — `double_array_trie::{DoubleArrayTrie,DoubleArrayTrieChar,
  DoubleArrayTrieZipper,DoubleArrayTrieCharZipper}`, `scdawg::{Scdawg,ScdawgChar}`, `prefix_zipper::…`,
  crate-root traits `{Dictionary,DictionaryNode,SyncStrategy}`. Deep `::char::`/`::zipper::` paths and
  the non-existent `double_array_trie_char`/`_zipper` top-level modules are normalized at P6.
- **Tooling:** added `scripts/doc-math-convert.raku`, a guarded mechanical converter (backticked-math
  + bare-`O(` → MathJax). It is reliable for clean symbolic math but NOT for pseudocode spans bearing
  incidental Greek (it cannot tell a code signature from a formula), so it is used only on clean-math
  files; the theory files use judgment-based subagents.
- **P5 ✅ (archive)** `git mv` of the Bucket-C deprecation manifest into `docs/archive/` (28 files:
  `research/RESEARCH_{INITIATIVES,TRACKING}`, the universal-levenshtein BTreeSet trio + PHASE4/DIAGONAL
  debug scratch, one-off development/verification/formal-verification debug & session notes, 10 scratch
  `test_*.v`); removed the 3 empty `formal-verification/proofs/{07,08,09}` dirs and fixed the
  `proofs/README.md` bullets; repointed the living `research/README.md` planning section. `.v` files
  moved verbatim, never reworded. (Optional `optimizations/`→`optimization/dynamic-dawg/` fold deferred.)
- **Process note:** parallel subagents are budget-expensive (~200k tokens each); running 11 at once
  exhausted the session window. Future waves are paced to 2–3 concurrent.

#### 2026-07-13 — Mechanical conversion pass (session endpoint) 🔄

Built a full guarded converter `scripts/doc-math-convert.raku` (fence-aware; backticked-math + bare-`O(`
+ a bare-Unicode run-detector with a "stop at any 2+-letter word" prose guard, an embedded-context
guard against fragmenting `A^{ND,χ}_n`-style notation, a `--bare-only` mode, and `*`-excluded MATHOPS
so markdown bold never breaks a run). Applied across the living set with per-batch backups + a
mangling/fragmentation restore loop; verified **0 fragmentation** and **balanced delimiters** (3
apparent imbalances are JS-template / `$1` regex false positives in code spans — not math).

**State:** ~**186 / 219 living docs fully MathJax-conformant** (0 scanner findings). **33 files retain
356 findings that require whole-expression JUDGMENT** (not safely mechanizable): ~257 are the
`docs/research/universal-levenshtein/` dense TCS-2011 automata notation (ASCII `^{}_` super/subscripts
interleaved with Unicode — e.g. `A^{ND,χ}_n(w)`, `L^χ_Lev`), and ~99 are display-math inference rules
(→ ```math blocks), named-function applications (`cost(T1∘T2)`, `|pattern|`), and Greek-in-terminology
(`λ-Theory`, `ε-transition` in headings — legitimately Unicode; converting breaks anchors). These, plus
P4 (diagram `<latex>`) and P6 (final gates, PathMap-concurrency currency, nav refresh, memory update),
are the agent-appropriate remainder — blocked only by the subagent session budget (resets 04:30 ET).
Full residual inventory: `scratchpad/RESIDUALS.txt` + `scratchpad/CAMPAIGN_STATE.md`.

#### 2026-07-13 — P2/P4/P6 completion — all gates green ✅

The remainder flagged at the previous session endpoint is **done**; every verification gate now
passes. Work completed, and the defects the campaign's own tooling introduced (found and repaired).

**Math (P2 close-out).** The residual judgment-heavy files were converted and the living-doc
allow-list `docs/.mathlint-include.txt` was **completed to 248 entries** — the earlier 228-entry
manifest was missing the whole `formal-verification/proofs/06_contextual_completion/` prose-proof
set (8 files), the six navigational TOC READMEs (`benchmarks/`, `bug-reports/`, `completion-reports/`,
`implementation-status/`, `optimization/`, `research/`), and four `universal-levenshtein/`
explanatory docs — so the gate had never scanned them. Guard #7 was applied to the Coq prose proofs:
inline-code lemma statements keep their **ASCII Coq operators** (`<=`, `>=`, `<>`) rather than being
forced into MathJax, matching sibling claims like `` `parent = child` ``.

**Converter defects found and repaired (this is the important finding).** The guarded converter
`scripts/doc-math-convert.raku` produced two classes of corruption that the math-lint gate could not
see (it checks glyphs, not markdown well-formedness):

1. **76 malformed adjacent spans** across 7 files — e.g. `` `$A^\forall$``$,\chi _n$` ``. CommonMark
   cannot close a 1-backtick span with a 2-backtick run, so these rendered as literal `$…$` garbage.
   Repaired by deleting the 4-char sequence `` $``$ ``, which merges each pair into one valid span
   (`` `$A^\forall,\chi _n$` `` — faithful to the original `A^∀,χ_n`).
2. **52 table rows whose math span swallowed the cell delimiter** across 15 files — e.g.
   `` | Match `$| \langle 1,1,0\rangle |$` Identity | `` — destroying the table structure. Repaired by
   splitting each span at its **unescaped** pipes and re-emitting the delimiters (rows that correctly
   used an **escaped** `\|` inside math were left untouched).

Also removed formulas from section **headings** (they produce ugly, fragile GitHub slugs and had
broken every TOC link into them); the notation is now restated in the section body instead.

**Content currency (P2).** Beyond math: the removed backends **`DawgDictionary` / `OptimizedDawg`**
(absent from both crates) were still taught as usable in 9 living docs — sections, imports,
constructors, benchmark rows, glossary entries, the algorithms index — all corrected to
`DynamicDawg` / `DoubleArrayTrie` with explicit "removed in 0.9.x" notes. The remaining deprecated
`liblevenshtein::dictionary::…` imports in the `07-contextual-completion/` cluster were repointed to
`libdictenstein::…`. `mork/` version labels (v0.8.0 → 0.9.1) and a `version = "0.4"` dependency
example were refreshed.

**Concurrency model corrected — the largest content defect.** Verified against libdictenstein 0.2.0
(the `path = "../libdictenstein"` dependency this repo actually builds): **every in-memory dictionary
backend is now lock-free** and reports `SyncStrategy::InternalSync` — `DynamicDawg(Char)` /
`DynamicDawgU64` (`LockFreeDawg`, per-node `ArcSwap<EdgeList>` + `compare_exchange`),
`SuffixAutomaton(Char)` (`LockFreeSuffixAutomaton`), `Scdawg(Char)` (`LockFreeScdawg`),
`PathMapDictionary(Char)` (`Arc<ArcSwap<PathMapState>>`), and `BijectiveMap`. `parking_lot::RwLock`
survives **only** inside the disk-backed `persistent_artrie` engine — never on a dictionary read path.
The docs still described the *retired* RwLock model (and `thread-safety.md` even carried an "intended
direction" note for a migration that had already shipped), which contradicted the observable runtime
contract. Corrected across ~20 files (`thread-safety.md`, `backends.md`, `features.md`, root
`README.md`, `architecture.md`, `getting-started.md`, `examples/`, the five per-backend implementation
docs, four design docs) plus the `concurrency-model` diagram. Measured historical figures (the PathMap
**~3.82× RwLock read-throughput** result; the ~10–20 ns / ~50–100 ns lock overheads) are **preserved as
dated historical notes**, not deleted. RwLock mentions that are *legitimate* were verified against
`src/` and left: the contextual-completion engine (`src/contextual/engine.rs` really does hold
`Arc<RwLock<ContextTree>>` + `Arc<RwLock<Transducer>>`), the cache/eviction layer, distance
memoization, and a user-code hot-swap example.

**P4 diagrams ✅.** PlantUML `<latex>` was validated end-to-end (JLaTeXMath typesets it to vector paths
in this repo's pipeline), then applied: **16 `.puml` converted**, every rendered label now LaTeX
(Unicode remains only in non-rendered source comments, which is correct). All **49 SVGs re-rendered
and in sync**; `xmllint` clean; no PlantUML error graphics; no leaked `<latex>` literals. The
`arcswap-vs-rwlock` diagram is retained as a *conceptual* contrast (it names no backend).

**Gate results (all green).**

| Gate | Result |
|---|---|
| #1 math-lint (`scripts/doc-mathlint.sh`) | ✅ **0** old-style constructs across **248** living docs |
| #2 diagrams (`render.sh --check`) | ✅ all **49** SVGs in sync; xmllint clean; no error graphics |
| #3 links (`lychee --offline --include-fragments`) | ✅ **0 errors** / 2602 OK (was 29 — 4 archive-move file links + 25 broken anchors) |
| #4 deprecated imports | ✅ only the deliberate deprecation *prose* in `architecture/overview.md:40` |
| #4b removed types | ✅ no `DawgDictionary`/`OptimizedDawg` presented as usable |
| #6 append-only integrity | ✅ **0** `.v`/`.tla` content changes (archive moves are pure `git mv`); ledger **+142 / −0** |
| #7 rustdoc | ✅ **0** `missing_docs`; `cargo doc` exit 0 |

**Known follow-up (deliberately NOT guessed at).** Several docs still show the **bincode 1.x** API
(`bincode::serialize` / `deserialize`), which cannot compile against the pinned `bincode = "2.0"`.
The correct replacement could not be established: `serialization.md` documents
`dict.serialize(file, Format::Bincode)`, but no such method is locatable in `src/`, `tests/`, or
`examples/` (tests use `serialize_paths`), so the real serialization API — likely split across
liblevenshtein and libdictenstein — needs a dedicated audit. Rewriting these examples against a
guessed API would have shipped *wrong code*, which is worse than the current consistent staleness;
they are therefore left as-is and recorded here.

#### 2026-07-13 — Serialization API corrected — supersedes the "known follow-up" above ✅

The preceding entry closed with a **deferral**: the bincode-1.x examples were left in place
because "the correct replacement could not be established". That was a **failure of
investigation, not an actual dead end** — and it is now resolved. The deferral is withdrawn.

**What the earlier investigation missed.** It concluded "no such method exists" from greps
over `src/serialization/*.rs`, never noticing that file is a **39-line re-export shim**
(`pub use libdictenstein::serialization::*`). The whole API lives in **libdictenstein**, and
the repo already shipped a *working, compiling* `examples/serialization.rs` demonstrating it.

**The real API** (verified against `libdictenstein/src/serialization/`, and against that
existing example):

- One **serializer type per format**, all implementing the `DictionarySerializer` trait:
  `BincodeSerializer`, `JsonSerializer`, `PlainTextSerializer`, `ProtobufSerializer`,
  `OptimizedProtobufSerializer`, `DatProtobufSerializer`, `SuffixAutomatonProtobufSerializer`,
  and the generic wrapper `GzipSerializer<S>` (used as `GzipSerializer::<BincodeSerializer>`).
- `fn serialize<D, W>(dict: &D, writer: W)` — bounded on **`D::Node: DictionaryNode<Unit = u8>`**
  — and `fn deserialize<D, R>(reader: R)` — bounded on `D: DictionaryFromTerms`. It encodes the
  dictionary's *terms* through the `Dictionary` trait; it does **not** go through serde.
- **`libdictenstein::serialization::bincode_compat`** — a public shim restoring the bincode-1.x
  free functions (`serialize`, `deserialize`, `serialize_into`, `deserialize_from`) on top of
  bincode 2.x, pinned to the legacy fixint-LE wire format. This is the drop-in for the old calls.

**Two defects this uncovered, both worse than the bincode calls themselves:**

1. **`docs/user-guide/serialization.md` documented a wholly fictional API.** A `Format` enum
   (`Format::Bincode`, `::Json`, `::Text`, `::ProtobufV2`, `::BincodeGz`, …), a
   `DictionaryDeserializer` trait, and a method-style `dict.serialize(file, Format::X)` — **none
   of which exist anywhere in either crate**. All 11 code blocks were rewritten to the real
   serializer types.
2. **`pathmap-dictionary.md` taught a serde round-trip that is impossible** — `PathMapDictionary`
   implements **no serde traits at all**. It must use `BincodeSerializer` (which works via the
   `Dictionary`/`DictionaryFromTerms` traits). Conversely `DynamicDawgChar` is `Unit = char`, so
   the `DictionarySerializer` trait does *not* apply to it; it derives serde and must go through
   `bincode_compat`. The docs now state both constraints explicitly.

Also corrected while in there: `double-array-trie.md` claimed memory-mapping gave a "zero-copy"
dictionary that "references memory-mapped data" — false; `deserialize` rebuilds an owned,
heap-resident dictionary. The claim now says what mmap actually buys and points at the
`Persistent*` family for genuine on-disk reads.

**Gate #5 (code-snippet validity) is now enforced, not spot-checked.** Added
`examples/doc_serialization_check.rs` (registered in `Cargo.toml` with
`required-features = ["serialization"]`), which reproduces **every** serialization pattern the
docs teach — including the `&D`-receiver case, the char-vs-byte split, the gzip wrapper, and the
versioned-struct `serialize_into`. It compiles with **0 errors and 0 warnings** under both
`--features serialization` and `--all-features`, and **runs green** (the round-trip asserts pass).
Any future drift in a documented serialization API now breaks the build.

Writing that check immediately caught a real bug in my own first draft: `PathMapDictionary::contains`
is a `Dictionary` **trait** method (the other backends also expose an inherent one), so the trait
must be in scope — proof that the compile gate earns its keep.

**Final gate results — all green.**

| Gate | Result |
|---|---|
| #1 math-lint | ✅ 0 across 248 living docs |
| #2 diagrams | ✅ 49/49 SVGs in sync |
| #3 links | ✅ **0 errors** / 2603 OK |
| #4 deprecated imports | ✅ only `phonetic_normalized` (legitimate) + the deprecation prose |
| #5 code validity | ✅ **`doc_serialization_check` compiles (0 warnings) and runs** |
| #6 append-only integrity | ✅ 0 `.v`/`.tla` content changes |
| #7 rustdoc | ✅ 0 `missing_docs` |

**No deferrals remain.**

---

## Correction — 2026-07-15: inline-math delimiters were transposed campaign-wide

**Defect.** The 2026-07-13 converter (`scripts/doc-math-convert.raku`) emitted inline math with its
delimiters **transposed** — a backtick code span whose *content* is dollar-delimited, instead of a
dollar-delimited span whose *content* is a backtick code span:

```
  WRONG  (GitHub renders it as literal monospace text, e.g. "$W$"):   `$…$`    ← backticks outside
  RIGHT  (GitHub renders it as MathJax):                              $`…`$    ← dollars outside

  converter emission, per site:   WRONG  '`$' ~ … ~ '$`'        RIGHT  '$`' ~ … ~ '`$'
```

So **every inline formula across the living docs was broken**. The lint scanner shared the same
inverted model (its kind-e "hygiene" rule even treated the backtick-then-dollar sequence as the
*opening* delimiter), so it never flagged the form. Root cause: all five emission sites in the
converter. Blast radius: **3,804 spans across 148 documents**, including `README.md`.

**Symptom fix.** New idempotent, fence- and tokenizer-aware repair `scripts/doc-math-transpose.raku`
transposed all 3,804 spans (backticks-outside → dollars-outside). The tokenizer (shared with the
scanner) guarantees it never matches the glue between two already-correct adjacent spans and is a
no-op on a second run. This ledger's 4 illustrative meta-examples of the broken form are guarded
and preserved.

**Secondary defect — swallowed markdown.** The same converter's span-bounding logic had also
**absorbed adjacent markdown into the math**: bold `**`, list bullets (`- ` / `N. `), and table
`|` delimiters ended up *inside* the math spans (94 lines). New conservative repair
`scripts/doc-math-fix-boundaries.raku` hoists those tokens back out (gated: table pipes only on
pipe-leading rows, bullets only on line-start spans), bailing on tangled multi-cell or
`**`-embedded spans; the ~8 irregular cases (Rholang-parallel associativity table, `[π]↑e`,
`β|N/A` cell, `i#e` split, two bold-embedded labels) were hand-reconstructed.

**Root-cause + prevention.**
- `scripts/doc-math-convert.raku` — all 5 emission sites corrected (dollars now emitted outside the
  backtick span); header and hygiene comments corrected.
- `scripts/doc-math-prescan.raku` — added `code-wrapped-dollar-math` (detects the transposed form),
  fixed the inverted kind-e opener, and added `table-column-mismatch` (flags **unescaped `|` in
  table cells** — the failure that swallowed pipes into math; counts delimiters with code spans
  blanked and `\|` neutralised, flagging rows with more cells than the `|---|` separator). Fixed the
  2 Bucket-A table rows it surfaced (`TCS_2011_PAPER_ANALYSIS.md`, `PROOF_INDEX.md`).

**Residual (intentional).** Three Bucket-B append-only records still carry unescaped-`|` table rows
(`archive/theory/scdawg/02-suffix-automaton.md`, `research/wallbreaker/scientific-ledger.md`,
`verification/FINDINGS_LEDGER.md`); the linter now flags them but, per the append-only rule, they
are left frozen.

**Verification.** `scripts/doc-mathlint.sh` → ✅ PASS (0 constructs across 248 living docs; the pass
message now correctly describes the dollars-outside form). Tokenizer re-scan: 0 residual
backticks-outside spans, 0 bold-in-span, 0 swallowed bullets, 0 trapped table pipes (excluding the
guarded ledger examples). Both repair scripts are idempotent (second `--dry` run = 0 changes).
