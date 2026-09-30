# CBC theory baseline evidence audit

This is the entry audit for pgmcp task `cbc-01-baseline` in epic
`metric-automata-cbc-theory` (#8567). The baseline commit is
`919c99352b74c0ab0ba8cbf45540fec982e7b7a4`. At the audit, the worktree
also contained uncommitted edits to both theory manuscripts and
`LazyProductOperations.v`, plus new CBC proof and model files. Consequently a
claim in a new or modified file is **worktree evidence**, not a theorem already
present in that baseline commit. The Rust and test edits already present when
this audit began are separate pre-existing work. A subsequent vector TWED
strict-origin correction and regression are recorded in the
[domain capability contract](CBC_DOMAIN_CAPABILITIES.md), not counted as
baseline evidence.

The [formal manifest](FORMAL_VERIFICATION_MANIFEST.tsv) records which files a
configured verifier treats as trusted, partial, or draft. That designation
does not establish an application-specific simulation. The actual claims are
the theorem statements in the linked proof files, with all section variables
and imported theorems retained as premises. A finite TLC run, symbolic
arithmetic check, or passing property test has its own bounded scope.

At the baseline commit, the manifest already listed `LazyWeightedFrontier.v`,
`LazyProductOperations.v`, `RangeCertificates.v`, and
`TemporalLazyProduct.tla` as `trusted/light`. The baseline
`LazyProductOperations.v` entry ended at unordered scheduler membership; its
BF-1 lexicographic stopping theorems belong to the later worktree overlay.
`OrderedTheoryRefinements.v`, the `Certified*.v` files, the lexicographic and
CBC TLA models, and the physical-time TWED transfer files were not tracked at
that baseline commit. Their current manifest entries describe newly supplied
worktree evidence. This repository audit found manifest declarations but no
saved per-artifact runner receipts for those four baseline files. A manifest
row is therefore recorded as the prior evidence *inventory*, not substituted
for a fresh proof replay or production validation.

## Evidence kinds and claim map

| Claim family | Exact source and current evidence | Scope established | Open premise for a production claim |
|---|---|---|---|
| ORC `R1–R4`, `C1–C2`, `D1–D2`, `B1–B2` | [ORC §§2–5](../theory/ordered-residual-calculus.md#2-residual-realization-and-its-limits), manuscript proofs; [Wolfram companion](../../scripts/verify-ordered-residual-calculus.wls) checks selected algebraic formulas and finite examples | Ordered residual, cost algebra, differential and backward-budget results under stated mathematical hypotheses | Chosen kernel recurrence, actual arithmetic order, cutoff behavior, and Rust transition correspondence |
| ORC `F1–F3`, `K1–K2`, `H1–H2`, `G1–G2`, `S1`, `M1–M3` | [ORC §§6–12](../theory/ordered-residual-calculus.md#6-frontier-algebra-and-certified-elimination), manuscript proofs; manifest-listed [LazyWeightedFrontier.v](temporal_automata/theories/LazyWeightedFrontier.v) proves selected generic frontier, abstraction, and retention lemmas | Conditional algebra, simulation, context, approximation, streaming, and metric results | Instance-specific order, finite-state, quotient, numerical, resource, and source obligations; manuscript IDs are not automatically mechanized |
| ORC `O1–O14` | [ORC §13 rule catalogue](../theory/ordered-residual-calculus.md#13-optimization-rules-and-selection), with the cited `R`, `C`, `F`, `B`, `H`, `K`, `G`, and `S` premises in their earlier sections; selected generic refinements in [OrderedTheoryRefinements.v](temporal_automata/theories/OrderedTheoryRefinements.v) | Conditional transformation permissions for the row-specific score, bounded-score, context, completed-result, or discrepancy observation; the catalogue itself is manuscript-level | Accepted instantiated rewrite certificates, complete source/target semantics, exact versus lower-bound type, numeric authority, concrete product and zipper laws, witness/resource observations, and Rust correspondence for each adopted rule |
| ORC `O15–O16` | [ORC §13 offset and scheduling arguments](../theory/ordered-residual-calculus.md#13-optimization-rules-and-selection); [LazyProductOperations.v](temporal_automata/theories/LazyProductOperations.v) supplies only generic BF-1 stopping lemmas | O15's exact min-plus translation law with retained output offset and separate all-infinite state; exact additive consistent-potential path reweighting; O16's conditional schedule and lexicographic top-`k` result order | Prove homogeneity for the selected recurrence and output, preserve budget/witness observations, establish consistent potential and terminal offsets where used, cover every pending original, validate machine tie/cost bounds, and account for scheduling/conversion costs; rounded or bottleneck algebra does not inherit the exact-additive argument |
| LOCPA `CA-1`, `FA-1` | [LOCPA registry](../theory/lazy-ordered-cost-product-automata.md#121-normative-theorem-schema-registry); [LazyProductOperations.v](temporal_automata/theories/LazyProductOperations.v) `admitted_final_is_within_cutoff` and `over_cutoff_final_is_rejected`; ORC cutoff proof | Generic cutoff and final-admission statements | Concrete finalizer, rounded cutoff comparison, and scheduler/result correspondence |
| LOCPA `RM-1`, `RM-2` | [LOCPA §§3–4](../theory/lazy-ordered-cost-product-automata.md#3-residuals-are-the-generalized-automaton-states), manuscript derivations | Conditional residual realization and deterministic quotient minimality | Per-kernel recurrence and reachable-state correspondence; no claim of NFA, byte, or time minimality |
| LOCPA `SP-1`, `SP-2`, `AN-1..3`, `AW-1` | [LOCPA §§4–5](../theory/lazy-ordered-cost-product-automata.md#4-canonical-frontiers-and-proved-subsumption); [LazyWeightedFrontier.v](temporal_automata/theories/LazyWeightedFrontier.v) `epsilon_dominance_is_residual_simulation`, `canonical_frontier_permutation_invariant`, `canonical_interning_is_permutation_sound` | Generic dominance, canonicalization, and permutation statements under supplied relation | Executable preorder, all predecessor schedules, normalization completeness, quotient-width instance |
| LOCPA `OC-1`, `CC-1`, `CC-2` | [LOCPA §5](../theory/lazy-ordered-cost-product-automata.md#5-exact-and-abstract-query-transitions); [LazyProductOperations.v](temporal_automata/theories/LazyProductOperations.v) observation/cache section | Generic observation congruence and exact cache refinement | Concrete decode, complete key, scope invalidation, error/resource observations, eviction behavior |
| LOCPA `AI-1..3`, `AP-1`, `EV-1` | [LOCPA §§5–6](../theory/lazy-ordered-cost-product-automata.md#5-exact-and-abstract-query-transitions); [LazyWeightedFrontier.v](temporal_automata/theories/LazyWeightedFrontier.v) `interval_additive_step_is_lower_simulation`, `point_interval_additive_step_is_exact`, `abstract_rejection_is_safe`, `exact_leaf_verification_has_no_false_positives` | Generic lower simulation and exact-leaf safety | Every concrete transition, captured originals, collision multiplicity, whole-search completion, independent exact verifier |
| LOCPA `MQ-1`, `MQ-2` | [LOCPA §9](../theory/lazy-ordered-cost-product-automata.md#9-metric-qualification-is-a-separate-theorem) and [ORC §12](../theory/ordered-residual-calculus.md#12-metric-and-quotient-qualification) | Conditional exact cutoff recognition and metric/quotient transfer | Per-measure domain proof and machine-arithmetic contract; no blanket metric claim for all kernels |
| LOCPA `ZP-1..7`, `PS-1..5` | [LOCPA §§6–7](../theory/lazy-ordered-cost-product-automata.md#6-lazy-synchronized-products); [LazyProductOperations.v](temporal_automata/theories/LazyProductOperations.v) `descend_preserves_snapshot_revision`, `product_child_components`, `query_first_child_is_product_equivalent`, `rejected_projection_constructs_no_child`; [TemporalLazyProduct.tla](tla/TemporalLazyProduct.tla) finite lifecycle model | Generic focus/product lemmas and bounded model behavior | Every selected backend's zipper laws, full original coverage, fair/explicitly incomplete scheduling, source mapping |
| LOCPA `GR-1..3`, `ST-1`, `TX-1`, `BO-1..2` | [LOCPA §8](../theory/lazy-ordered-cost-product-automata.md#8-online-semantics-stability-and-stack-safety); [LazyWeightedFrontier.v](temporal_automata/theories/LazyWeightedFrontier.v) `generational_retention_is_prefix_independent`, `page_then_resume_equals_uninterrupted`, `rejected_child_preflight_is_atomic`; finite [streaming](tla/TemporalStreamingGenerations.tla) and [DFS](tla/TemporalDfsStackArena.tla) models | Abstract bounded generations, logical resumption, preflight, and finite-state lifecycle checks | Actual byte accounting, allocation failures, arbitrary input lengths/pages, and Rust continuation mapping |
| Existing arena-reference claim | Baseline [manifest row](FORMAL_VERIFICATION_MANIFEST.tsv) and [LazyWeightedFrontier.v](temporal_automata/theories/LazyWeightedFrontier.v) `push_fresh_state_preserves_valid_references`; the baseline theorem prepended a state but assigned the old length as its ID | At baseline, this established only index bounds. For `states=[10]`, old ID `0`, and new state `20`, the modeled list became `[20,10]`: old ID `0` changed referent and proposed new ID `1` named `10`. The manifest's stable-reference wording was too strong for that proof. | The corrected append-only model now adds `append_fresh_state_preserves_old_referents`, `append_fresh_state_binds_new_identifier`, and `push_fresh_state_preserves_frame_referents`. Concrete Rust arena correspondence and capacity/allocation semantics remain open. |
| LOCPA `RS-1`, `CL-1`, `RA-1`, `RA-2` | [OrderedTheoryRefinements.v](temporal_automata/theories/OrderedTheoryRefinements.v) `arbitrary_adaptive_switches_preserve_score`, `ordered_cursor_matches_independent_lookups`, `admissible_bound_first_preserves_completed_search`, `arbitrary_page_boundaries_preserve_private_edge_refinement` | Generic finite-trace switching, lookup order/cost, bound-first result membership, private-edge paging | Real conversion, scheduled row order, per-original bound admissibility, cumulative Rust ledger, checked arithmetic and allocation |
| LOCPA `BF-1` | [LazyProductOperations.v](temporal_automata/theories/LazyProductOperations.v) `lex_stop_preserves_topk_selection`, `certified_summary_pruning_is_sound`; finite [LexicographicKnn.tla](tla/LexicographicKnn.tla) | Abstract natural-number rank stopping with complete/unknown/empty tie summaries; finite counterexample to equality-cost stopping | Sound machine cost/tie lower bounds, complete candidate ownership, actual heap/tie semantics, snapshot and result-page mapping |
| CBC `CBC-1`, `CBC-2` | [CertifiedMetricExecution.v](temporal_automata/theories/CertifiedMetricExecution.v) `local_certificates_lift_to_traces`, `step_certificates_compose`, `decreasing_rank_bounds_internal_steps` | Conditional finite trace simulation/composition and rank bound | Checked local simulation, concrete decreasing rank, no-stuck-state proof, terminal observation and source relation |
| CBC exact/lower certificate kernel | [CertifiedContracts.v](temporal_automata/theories/CertifiedContracts.v) `accepted_exact_certificate_preserves_every_environment`, `accepted_lower_certificate_bounds_every_environment`; finite [replay model](tla/MetricCertificateReplay.tla) | Executable checker for exact natural, score-only expression rewrites and distinct lower bounds; finite rejection abstraction | Binding opaque scope fields to live inputs, rounded arithmetic, order/witness/resources, proof-to-Rust realization |
| CBC `BF-2`, `BF-3`, `BF-4` | [CertifiedMetricExecution.v](temporal_automata/theories/CertifiedMetricExecution.v) `threshold_floor_pruning_is_sound`, `maximum_of_rank_certificates_is_sound`; [CBC §6](../theory/certified-metric-automata.md#6-stronger-ordered-pruning-with-scoped-rank-certificates) | Natural-number equality-slice and whole-pair maximum lemmas, plus information-limit argument | General rank order, strict-bound algebra, complete equality slice, scoped summary constructors and concrete tie keys |
| CBC `NC-1` | [CBC §5 enclosure theorem](../theory/certified-metric-automata.md#5-numerical-soundness-and-metric-enclosures), manuscript triangle-inequality proof; [numeric authority profile](CBC_NUMERIC_AUTHORITY.md) maps its applicable score profiles | Ideal metric lower bound from certified lower/upper pivot enclosures; directed-rounding extension is conditional | Concrete enclosure primitive proofs and valid ranges, exact-verifier rounding/error relation, machine comparison authority, and source-mapped implementation; no binary64 geometric pruning follows from the ideal theorem alone |
| CBC session and split | [CertifiedSearchSession.v](temporal_automata/theories/CertifiedSearchSession.v) ownership/cursor theorems; [CertifiedRegionPartition.v](temporal_automata/theories/CertifiedRegionPartition.v) `accepted_split_has_exact_original_coverage`; [CertifiedStructuralSplit.v](temporal_automata/theories/CertifiedStructuralSplit.v) `constructed_split_is_accepted`; finite [cursor model](tla/RangeCursorOwnership.tla) | Explicit finite original-list ownership and split preservation, plus bounded cursor counterexamples | Derive original lists and labels from captured Rust traversal; heap order, resources, errors and compact hot-path certificate |
| CBC `QO-1`, `QO-2` | [CertifiedMetricExecution.v](temporal_automata/theories/CertifiedMetricExecution.v) `local_potential_bounds_total_work`, `selected_minimum_is_model_optimal` | Conditional potential bound and minimum of a finite supplied feasible portfolio | Actual primitive costs, live/peak bytes, full feasible alternatives, workload distribution, selection cost and production measurements |
| CBC finite acyclic planning argument | [CBC §7.2](../theory/certified-metric-automata.md#72-qo-2-optimum-within-a-certified-portfolio), manuscript topological-induction proof | Least complete exact cost among paths represented in a finite acyclic graph with declared terminal and conversion costs | Complete feasible action graph, exact workload-dependent primitive costs, graph construction and selection overhead, resource feasibility, and source correspondence; it does not prove optimality against omitted alternatives or online unknown futures |
| Physical-time TWED transfer | [TimestampedMetricTransfer.v](twed/theories/Metric/TimestampedMetricTransfer.v) anchored recurrence and strict-domain theorems; [TwedSourceMetric.v](twed/theories/Metric/TwedSourceMetric.v) sample/time product metric | Strict-origin admission and conditional transfer; ideal product sample/time metric | Full local sequence-level source metric theorem, binary64 metric laws and Rust score correspondence |

Rows naming multiple schemas enumerate all registry IDs in that group; they do
not assert that one listed lemma discharges every schema. The exact theorem
statement, including its premises and local definitions, takes precedence over
the prose summary. Worktree-only CBC artifacts above cannot be used as evidence
about the baseline commit.

## Draft-correction checklist

| Potential overstatement | Required correction and current treatment |
|---|---|
| A generic file compiles, therefore the production Rust automaton is CBC | Reject. The claim ladder in the [obligation ledger](../theory/metric-automata-cbc-obligations.md#claim-levels-and-evidence-rule) requires a concrete instance and production correspondence. The new CBC draft calls those steps open. |
| A theorem over ideal reals or naturals licenses binary64 pruning | Reject. The contract names arithmetic and primitive order; the machine-specific transfer is open. |
| A local terminal/finalizer lemma proves completed range or kNN behavior | Reject. Whole-session original coverage, tie order, completion, and result-page observations need separate proofs. |
| A finite TLC or Wolfram corpus proves unbounded executable behavior | Reject. Model state bounds and symbolic formula scopes are stated separately from source correspondence. |
| A metric or residual state-minimality proof gives minimum bytes or runtime | Reject. The optimization objective must include acquisition, conversion, storage, and workload costs. |
| A lower-bound or abstract certificate proves the exact score | Reject. Exact and lower certificate types and observations remain separate. |
| An index-validity theorem proves stable arena referents | Reject for the baseline proof. Prepending kept every frame index in bounds while changing its referent. The corrected append-only model requires separate old- and new-identifier theorems. |
| Snapshot node sharing preserves original multiplicity automatically | Reject. Original path/slot identities, collision buckets, and explicit ownership must be accounted for. |

## Reproduction and limits

The frozen baseline is the explicit commit
`919c99352b74c0ab0ba8cbf45540fec982e7b7a4`; inspect it with
`git show 919c99352b74c0ab0ba8cbf45540fec982e7b7a4:<path>`. The current
revision is reported separately by `git rev-parse HEAD`, and any worktree
overlay by `git -c core.fsmonitor=false status --short` and focused `git diff`.
The manifest identifies configured file-level gates. The
[ORC evidence section](../theory/ordered-residual-calculus.md#14-evidence-controls-and-reproducibility)
lists the Wolfram command and explains its exact/finite scope. The
[LOCPA registry](../theory/lazy-ordered-cost-product-automata.md#121-normative-theorem-schema-registry)
lists the schema-specific executable gates. During this audit,
`CertifiedStructuralSplit.v` was compiled with resource-limited `coqc` and
checked by `coqchk`; `scripts/verify-formal.sh audit`,
`scripts/doc-mathlint.sh`, and `git diff --check` passed. Those runs do not
revalidate every artifact in this table, and no production optimization or
benchmark experiment was run.
