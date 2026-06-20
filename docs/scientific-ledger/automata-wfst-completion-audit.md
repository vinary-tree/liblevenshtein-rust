# Automata WFST Completion Audit

Date: 2026-06-19

Root tracker item: `lev-phonetic-wfst-scientific-evaluation`

This audit ties the user-requested automata and WFST evaluation program to the
repository commits, pgmcp experiment ledger, and local hygiene checks available
at this point in the session. The structured pgmcp experiment records remain the
source of truth for sample data, locked criteria, statistical tests, and trusted
experiment verdicts.

## Evidence Classes

- **Accepted experiment**: pgmcp experiment status is `decided`, with a verdict
  supporting the treatment or boundary decision.
- **Conformance gate**: deterministic acceptance evidence for a correctness
  boundary, typically `correctness_pass_rate = 1.0` under a memory cap.
- **Repository commit**: a stable git commit containing implementation,
  documentation, rule, or verification updates.
- **Lifecycle gate**: pgmcp work-item status has accepted the available evidence
  through its own transition graph.

## Requirement Coverage

| Requirement | Evidence |
|-------------|----------|
| Record architecture, design, and engineering ideas for native Levenshtein automata, phonetic automata, and WFST integration. | Pgmcp root item `lev-phonetic-wfst-scientific-evaluation`; repository ledger `docs/scientific-ledger/automata-wfst-evaluation.md`; commit `3b0dc71`. |
| Evaluate every proposed automata and WFST experiment in priority order and make empirical time/space decisions. | Accepted or rejected treatments are recorded for LEV-001, LEV-002, LEV-005, LEV-006, LEV-007, LEV-008, LEV-010, PHON-001/002, PHON-008, PHON-009, PHON-010, PHON-011, PHON-012, PHON-013, BENCH-005, DUAL-001, DUAL-003, DUAL-008, DUAL-009, DUAL-010, and WFST-004. |
| Use academic-style benchmark coverage rather than only local smoke cases. | BENCH-005 covers Birkbeck/Fawthrop spelling correction, Mitton-style misspelling suites, CMUdict homophone behavior, Pizza&Chili-style text throughput/RSS, and OpenSLR/LibriSpeech-style WFST lexicon fixtures. |
| Explain and debug the poor CMUdict homophone smoke-test behavior. | PHON-008 found that the full CMUdict extension plus exact whole-word split index produced 3960/3960 expected terms in full results over the first 2048 CMUdict cases. The top-k misses were classified as `top_k_ceiling` or `ambiguous_query_pronunciation_ranking`, with zero normalized-index/query bugs found in that diagnostic. |
| Add English rule coverage when the evidence supported a rule-coverage cause. | PHON-009 kept English LLev extensions before primary rules for whole-word coverage. Commit history includes the CMUdict homophone LLev extension and subsequent conformance gates for multi-symbol outputs, composite imports, and compound contexts. |
| Keep `lling-llang` agnostic of liblevenshtein while integrating the WFST path. | DUAL-010 keeps dictionary-integrated `PhoneticWfst` as a dictionary-side scorer/acceptor boundary and uses `RewriteWfst` or `PhoneticNfaWfst` plus Levenshtein WFST composition for query-to-dictionary transduction. |
| Select crate ownership for the lattice/WFST boundary. | WFST-004 selected `lling-llang` as the lattice graph engine and WFST algorithm crate, `duallity` as the liblevenshtein adapter, `llattice` as algebraic lattice traits, and `libgrammstein` as language-model/weight infrastructure. |
| Reference `mettail-rust` without modifying it. | The WFST and weighted-automata notes treat `mettail-rust` as a research reference only. This repository session did not modify that project. |
| Commit at stable points with major changes enumerated. | Recent commits include `640c710`, `3b0dc71`, `609c343`, `4a3dcaf`, `af20f20`, and `dd534ae`, each with scoped verification or ledger content. |
| Cap heavy processes after the OOM report. | Expensive builds, tests, and benches were run through `systemd-run --user --scope` with `MemoryMax` and `MemorySwapMax=0`. Core Rocq proof compilation OOM events were contained inside the capped scope and recorded in `docs/verification/SUMMARY.md`. |
| Clean task-owned temporary files. | Current checks found no task-owned top-level files in `/tmp` matching the liblevenshtein, Levenshtein, phonetic, or WFST scratch prefixes used in this session. |
| Reconcile tracker state through the established pgmcp API rather than source edits. | On 2026-06-20, `work_item_completion` for `lev-phonetic-wfst-scientific-evaluation` reported `verified_fraction = 1.0` with 21/21 evidence-verified leaves. |

## Experiment Decisions

| Area | Accepted treatments | Non-adopted or inconclusive treatments |
|------|---------------------|----------------------------------------|
| Native Levenshtein automata | Arena-backed query state, arena-index parent paths, ordered top-k arena paths, cached priority-query term keys, bounded transition slice scan, exact OSA hybrid for long transposition cases. | Lazy path-arena capacity tuning was measured as correct but slower/larger. Parent-path byte reduction was inconclusive for unordered queries even though allocation count and latency improved. |
| Phonetic automata and LLev | Trie-product phonetic regex traversal, English extension ordering for whole-word coverage, multi-symbol `RuleSetChar` output expansion, composite LLev import preservation, compound context integration, UTF-8 multi-character substitutions. | The original CMUdict smoke-test gap was not a traversal-index bug under the accepted diagnostic; it was resolved as coverage/oracle/ranking behavior for the measured cases. |
| WFST and lattice architecture | Duallity label semantics fix, rewrite WFST multi-symbol chains, phonetic NFA `StateSource` expansion, dictionary-side `PhoneticWfst` boundary, lling-llang lattice/WFST ownership. | No accepted evidence supports moving the lattice graph engine into liblevenshtein or making lling-llang depend on liblevenshtein. |

## Tracker Lifecycle Reconciliation

The experiment and conformance evidence has been reconciled through pgmcp's
public work-item and experiment APIs. The root completion API reports every leaf
as evidence-verified:

```text
work_item_completion("lev-phonetic-wfst-scientific-evaluation")
verified_fraction = 1.0
verified_leaves = 21
leaf_count = 21
```

The rows below are original non-leaf bug/task rows whose verified experiment
closure leaves carry the trusted evidence. Their parent statuses remain
`triage` or `verifying` because the agent API cannot force legacy parent rows to
`verified`; those states are tracker lifecycle artifacts, not repository source
incompletion.

| Parent work item | Parent status | Verified evidence leaf | Evidence state |
|------------------|---------------|------------------------|----------------|
| `dual-003-wfst-label-semantics` | `triage` | `dual-003-wfst-label-semantics-evidence-closure` | Accepted label-semantics experiment. |
| `phon-008-llev-zompist-recall-parity-root-cause` | `triage` | `phon-008-llev-zompist-recall-parity-root-cause-evidence-closure` | Accepted root-cause experiment with full-result recall closure for measured cases. |
| `dual-008-rewrite-wfst-multisymbol-output` | `triage` | `dual-008-rewrite-wfst-multisymbol-output-evidence-closure` | Accepted duallity conformance gate. |
| `dual-009-phonetic-nfa-statesource-pending` | `triage` | `dual-009-phonetic-nfa-statesource-evidence-closure` | Accepted duallity conformance gate. |
| `phon-010-llev-apply-multisymbol-output` | `triage` | `phon-010-llev-apply-multisymbol-output-evidence-closure` | Accepted liblevenshtein conformance gate. |
| `phon-011-llre-import-composite-llev-symbols` | `triage` | `phon-011-llre-import-composite-llev-symbols-evidence-closure` | Accepted liblevenshtein conformance gate. |
| `phon-012-llev-compound-context-integration-coverage` | `verifying` | `phon-012-llev-compound-context-integration-coverage-evidence-closure` | Accepted liblevenshtein conformance gate. |

Three lifecycle reconciliation attempts were made after the evidence was
recorded:

- Bulk `set_status` from `triage` to `in_progress` for the six triage rows
  failed for each row with `no transition 'triage' -> 'in_progress' exists`.
- Bulk `set_status` from `triage` to `ready` for the same six rows failed for
  each row with `no transition 'triage' -> 'ready' exists`.
- Direct `set_status` from `verifying` to `claimed_done` for
  `phon-012-llev-compound-context-integration-coverage` failed with
  `no transition 'verifying' -> 'claimed_done' exists`.

The current audit therefore treats the scientific evaluation leaves as complete.
No repository source changes remain for these rows in `liblevenshtein-rust`.

## Local Hygiene Snapshot

- `git status --short`: clean before this audit file was added; rechecked clean
  after the 2026-06-20 MSM checkpoint commit `dd534ae`.
- Stale-status marker scan over active source, docs, and LLev rule files:
  expected matches are limited to four literal CMUdict word entries in
  `data/rules/english/cmudict_homophones.llev` at lines 22341-22344.
- `/tmp` top-level scratch scan: no task-owned files matching the session
  scratch prefixes.
- Heavy commands: use capped `systemd-run --user --scope` forms as documented in
  `docs/scientific-ledger/automata-wfst-evaluation.md`.
