# Mathematical notation errata — 2026-08-30

This append-only erratum corrects a rendering defect in earlier scientific,
research, release, and universal-automaton records. It does not alter their
claims, measurements, hypotheses, or original bytes.

## Defect and normative reading

The affected lines contain an extra U+0060 GRAVE ACCENT adjacent to an inline
MathJax delimiter. GitHub can consequently render the intended formula as
literal code. For every affected mathematical span listed below, the normative
reading removes only the extraneous grave accent so that the dollar signs
surround exactly one backtick span. For example, the intended recurrence
parameter is $`\nu=1`$, and the intended complexity is
$`\mathcal{O}(\lvert Q\rvert)`$.

No numerical value or mathematical operator changes. This is a presentation
correction only.

## Affected immutable records

The audit found the malformed delimiter byte pattern at these paths. “Sites”
counts byte-pattern sites, not distinct scientific claims; a line can contain
more than one site.

| Immutable record | Sites |
|---|---:|
| `docs/releases/4.0.0-rc.6.md` | 2 |
| `docs/research/evaluation-methodology/NORVIG_CORPUS_INTEGRATION_PLAN.md` | 1 |
| `docs/scientific-ledger/affine-gap-2026-08-01.md` | 9 |
| `docs/scientific-ledger/downstream-query-surfaces-2026-08-02.md` | 5 |
| `docs/scientific-ledger/dtw-kernel-2026-08-01.md` | 2 |
| `docs/scientific-ledger/elastic-kernel-extraction-2026-07-31.md` | 2 |
| `docs/scientific-ledger/elastic-ucr-harness-2026-08-01.md` | 21 |
| `docs/scientific-ledger/erp-kernel-2026-07-31.md` | 1 |
| `docs/scientific-ledger/position-kind-zero-cost-2026-08-01.md` | 11 |
| `docs/scientific-ledger/true-damerau-2026-08-01.md` | 14 |
| `docs/scientific-ledger/twed-kernel-2026-08-01.md` | 11 |
| `docs/universal/merge_split_phase3_complete.md` | 10 |

The source records remain authoritative for their evidence and chronology;
this erratum is authoritative only for the corrected mathematical rendering.
Living documentation was repaired in place because it is explanatory rather
than append-only evidence.
