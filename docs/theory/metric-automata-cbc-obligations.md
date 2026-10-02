# Metric automata CBC obligation ledger

**Campaign:** pgmcp epic `metric-automata-cbc-theory` (#8567).
**Baseline revision:** `919c99352b74c0ab0ba8cbf45540fec982e7b7a4`, with pre-existing uncommitted theory and temporal changes retained.
**Status:** contract inventory and proof map, not an assertion that the whole library is certified.

This ledger extends the claim registry in [LOCPA](lazy-ordered-cost-product-automata.md#12-formal-theory-and-executable-gates). It is the campaign's readable index of exact obligations. The [formal manifest](../verification/FORMAL_VERIFICATION_MANIFEST.tsv) remains the authority for which artifacts pass a configured gate. The [certified execution draft](certified-metric-automata.md) states candidate theorems; its generic proof kernels must be instantiated and connected to executable artifacts before granting a production CBC claim.
The [baseline evidence audit](../verification/CBC_BASELINE_EVIDENCE.md)
records the exact starting revision, current worktree overlay, claim scope,
prior evidence type, and draft-correction checklist.
The [source-mapped domain capability contract](../verification/CBC_DOMAIN_CAPABILITIES.md)
expands the family inventory below with constructor links, quotient identities,
and negative controls.
The [numeric authority contract](../verification/CBC_NUMERIC_AUTHORITY.md)
maps every major lower-bound and exact-verifier path to its arithmetic profile,
including signed-zero cutoff equality and the machine-transfer proof still
required for ideal-real bounds.
The [result and witness contract](../verification/CBC_RESULT_WITNESS_CONTRACT.md)
instantiates the identity, multiplicity, order, and witness components below
for the current source surfaces.
The [observation profile](../verification/CBC_OBSERVATION_PROFILES.md) records
validation, error, pause/resume, and resource-trace obligations for those
surfaces.
The [resource event contract](../verification/CBC_RESOURCE_EVENT_CONTRACT.md)
defines primitive work and storage accounting separately from public budget
charges.
The [claim/instance matrix](../verification/CBC_CLAIM_INSTANCE_MATRIX.md)
indexes every downstream epic task against its proof level, source family,
external premises, and enforcement gate while the formal manifest remains the
artifact-status authority.

## Claim levels and evidence rule

| Level | Required evidence | Current meaning |
|---|---|---|
| Mathematical | Correct statement and complete proof under named domain/arithmetic hypotheses | Applies only to the stated measure |
| Generic mechanized | Checked Rocq theorem with every explicit premise shown | Does not discharge a concrete instance |
| Concrete instance | Every generic premise proved for one specified kernel and observation profile | May still describe a model |
| Enforced construction | Checked certificate rejection and acceptance semantics for a generated artifact | Proof checker and trust boundary identified |
| Production correspondence | A proved lowering, sound translation validator, or direct source refinement for the named executable | Required before calling that executable CBC |

A checked file with a `Context` parameter proves an implication whose parameter is a premise. A checked marker, regression test, or source hash does not discharge that premise. Review every transitive theorem dependency and any imported source theorem. When one level is missing, record it as **open** for that particular claim.

## Contract dimensions

Write a selected operation contract as $`(D,q,\theta,\mathcal N,\sigma,\tau,\prec,W,B,O)`$:

| Field | Required choice | Source to audit |
|---|---|---|
| `D` | Validated input domain, sequence identity or quotient, lawful empty cases | `src/time_series/metric_domains.rs`; `src/time_series/timestamped_twed.rs`; kernel constructors |
| $`q,\theta`$ | Query and fixed metric/quantizer/configuration parameters | Kernel and index constructors; snapshot manifest |
| `N` | Exact integer/rational, ideal real, or the actual ordered binary64 operation graph | Kernel scorer, interval automaton, `src/time_series/automaton/cost.rs` |
| $`\sigma`$ | Captured dictionary revision and full-precision original set | `src/time_series/elastic/walker/snapshot.rs`; index generation |
| $`\tau`$ | Inclusive semantic distance cutoff | `src/time_series/bounded.rs`; concrete scoring path |
| $`\prec`$ | Candidate identity and deterministic total rank | Elastic bucket/slot path; timestamped episode ID |
| `W` | Score-only, any valid witness, or canonical witness with tie rule | Exact verifier and witness replay |
| `B` | Hard cumulative limits, per-call page limits, live and peak storage | `ResourceLimits`, `PageBudget`, `ResourceUsage` |
| `O` | Required result, error, resumption, page, and resource observations | API-specific profile below |

Never infer a numeric bound under one `N` from a theorem under another `N` without a transfer proof. The public score authority for a concrete binary64 API is its actual sequence of machine operations and result tags. Equal algebraic real expressions can round differently after reassociation or fusion.

### Operation profiles

| Profile | Required complete result | Additional observations to model |
|---|---|---|
| Online score | Exact or cutoff-tagged score for every committed prefix | Validation order, partial progress, numeric failure |
| Exact range | All and only originals whose authoritative score satisfies the inclusive cutoff, in the promised order | Collision multiplicity, snapshot, optional witness and certificate |
| Exact kNN | First `min(k, eligible population)` distinct originals by score and declared tie key | Underfull heap, k=0 validation, threshold monotonicity |
| Resumable range/kNN | Same completed result when sufficient resources and resumed fairly | Partial result contract, reason tag, continuation ownership, cumulative ledger |
| Resource-sensitive | Declared work and storage accounting | Page boundaries, cache-path differences, actual versus reserved versus charged work |

`OperationOutcome::Complete` means successful exhaustive completion. `Incomplete` carries a reason, optional partial value, optional continuation, and usage. `ExactDecision::WithinCutoff`, `AboveCutoff`, and `NoFiniteAlignment` are distinct. A valid request that exceeds a budget is not an empty complete result. The exact precedence between multiple simultaneous invalid conditions belongs to the selected concrete API and must be read from its validation path; this matrix does not invent one global order.

The semantic cutoff $`\tau`$, the session-wide `ResourceLimits` ceilings, and `PageBudget` per-call slice are separate. `ResourceUsage` fields may count executed work or capacity reserved before a nonresumable operation. Thus a claim about *actual executed work* requires a separate primitive event model, and a claim about live storage requires more than the existing peak logical counters. A semantic rollback after charged work cannot make executed work disappear.

### Domain and capability inventory

| Family | Validated mathematical identity and law boundary | Current evidence and missing gate |
|---|---|---|
| Standard Levenshtein | Finite strings, ordinary equality, unit integer edit cost | Core metric proof and a checked exact-natural streaming-row recurrence bridge exist; the chosen production automaton still needs its full session/certificate/source correspondence chain |
| Weighted strings | Fixed lawful nonnegative operation costs; metricity requires the stated symmetry, separation, and triangle conditions | No blanket weighted metric grant from the generic ordered residual theory |
| ERP | Raw gap-valued sequences form a pseudometric; `ErpQuotientSeries` removes all gap samples for a fixed finite gap | Quotient constructor and metric-labeled wrapper exist; concrete numeric and whole-search correspondence remain separate |
| Scalar/vector discrete Fréchet | Nonempty paths on a fixed ground domain; consecutive stutter quotient identifies zero-distance sequences | Canonical `FrechetStutterClass` exists for scalar paths; vector metric qualification needs its own domain map |
| MSM | Lawful positive move/split/merge configuration and admitted finite series | Reviewed metric marker and named lower-bound proofs exist; the partial MSM proof tree and source/numeric gaps prevent a library-wide closed CBC claim |
| Unit-grid TWED | Positive stiffness and nonnegative gap penalty on a common uniform grid | Metric config and interval/recurrence proof islands exist; rounded metric laws and full executable correspondence remain open |
| Physical-time TWED | Same unit and origin; finite scalar or typed vector values; strictly increasing timestamps; first timestamp strictly after origin for metric qualification | The committed vector series guard enforces the origin restriction; the committed scalar raw API does not. An exact-natural 2007 alignment/DP minimum theorem is checked. `TimestampedMetricTransfer.v` proves the origin-equal obstruction for any nonnegative reflexive point cost, but still imports ideal-real source metricity and source-distance identification as premises; machine metricity and Rust correspondence remain open |
| Banded/general DTW | Exact recurrence and separately certified admissible filters; general family is nonmetric | Keep as a capability control; no triangle-dependent pruning license |
| Soft-DTW | Soft algebra/approximation profile, not ordinary exact metric rank without a separate theorem | Keep as a capability control |

For timestamped TWED the committed scalar raw constructor admits an initial
timestamp equal to the origin. The already checked 1-by-2 counterexample gives
distinct raw sequences at zero distance when the gap penalty is zero. A scalar
strict wrapper is present only in an uncommitted worktree overlay, so this
domain admission issue remains open in committed source. Such a wrapper would
still leave the external ideal metric theorem and binary64
separation/triangle laws open.

### Candidate identity and rank inventory

A candidate is an **original occurrence**, not merely a compressed node or an abstract bucket. The indexed snapshot may share physical dictionary states while preserving separate original paths. Elastic bounded kNN uses exact kernel cost and the original encounter sequence for ties; bounded range finalization uses original vector position. The legacy convenience walker has its own first-encounter sequence and defensive ID coalescing. Bucket/slot coordinates identify originals during traversal and must map to the selected public tie rule. Timestamped TWED ranking uses `(distance.total_cmp, episode_id)`; the episode ID is part of the identity contract. A result order theorem must prove that every eligible original is covered once, that the comparator is total on admitted scores, and that the heap maintains the first `min(k, |U|)` verified candidates. Zero-cost ties do not authorize deduplication.

For range results, specify whether promised order is score order, dictionary order, or another concrete API order. Do not transport an unordered-set proof to an ordered-result profile. A witness proof must name whether any optimal witness or the canonical tie-selected witness is promised.

## Existing proof islands and exact open steps

| Claim | Checked input | Remaining prerequisite for executable CBC |
|---|---|---|
| ORC R2/R2a residual/representation preservation | Ordered theory and `OrderedTheoryRefinements.v` generic finite traces | Termination between progress steps, concrete layout conversion, and observation profile |
| LOCPA BF-1 ordered stopping | `LazyProductOperations.v` generic lex rank and unknown/empty/known summaries | Whole-session candidate ownership, heap correspondence, machine admissibility, tie mapping |
| BF-2/BF-3 draft extension | `CertifiedMetricExecution.v` natural-number slice/maximum examples | General order, strict-bound algebra, complete union, scoped summary constructors, concrete instances |
| CBC-1 finite-step refinement | `CertifiedMetricExecution.v` implication from local simulation | Checked local certificates, terminal observation, source-to-executable relation |
| CBC-02 finite certificate composition | [CertifiedCertificateComposition.v](../verification/temporal_automata/theories/CertifiedCertificateComposition.v) proves that an ordered list of accepted exact-natural, score-only local certificates preserves the emitted score sequence through a finite execution ending in explicit completion; a two-instruction control emits distinct scores across silent steps, while an infinite-silent-step control shows why progress is separate | Prove the checker covers the chosen ORC operations, bind instruction steps to source transitions and completed observations, and discharge progress, witness/resource, numeric, and Rust correspondence separately |
| CBC-2 finite internal rank | `CertifiedMetricExecution.v` implication from decreasing natural rank | Concrete rank, no-stuck-state proof, finite reference progress, conditional resumption |
| CBC-02 checked scope reuse | [CertifiedCertificateScopeReuse.v](../verification/temporal_automata/theories/CertifiedCertificateScopeReuse.v) checks exact-request replay and stable-field equality for cutoff-independent exact-natural score-relation replay; stale query, gap, stiffness, revision, arithmetic profile, and raised-cutoff cached-decision controls are checked | Bind proof-level tokens and score environments to live query, parameters, numeric implementation, captured snapshot, and Rust cache entries; prove a fresh cutoff comparison before reusing a prune at a raised cutoff |
| CBC-04 scoped rank restriction | [CertifiedRankRestriction.v](../verification/temporal_automata/theories/CertifiedRankRestriction.v) restricts parent certificates only with equal scope and child inclusion; combines two rank floors by lawful lexicographic maximum only with common-scope coverage. Independent global cost/tie floors admit coordinatewise maxima, while a concrete conditioned-floor control refutes that operation | Prove source region inclusion/coverage and scope binding, validate each bound producer and comparator under the selected numeric authority, then relate certificate application to heap and Rust search behavior |
| CBC-04 original rank identity | [CertifiedOriginalRankInstances.v](../verification/temporal_automata/theories/CertifiedOriginalRankInstances.v) proves exact-natural lex comparator consistency and identity from injective tie maps: elastic originals use a modeled dictionary path plus terminal-bucket slot with an assumed injective encounter ordinal; timestamped originals use an assumed unique episode ID. Two originals in one bucket and one compressed node remain distinct | Prove captured path/slot enumeration, ordinal and episode-ID assignment, actual score and `f64::total_cmp` order, heap/sort behavior, and source/Rust correspondence; the natural-cost model does not certify binary64 ranking |
| CBC-04 per-original threshold coverage | [CertifiedThresholdCoverage.v](../verification/temporal_automata/theories/CertifiedThresholdCoverage.v) models finite bucket/slot originals and includes every enumerated exact-cost-at-most-cutoff original in $`S_Q(c)`$ when its own lower bound is sound. A complete equality-cost slice with a region cost floor yields an attained minimum tie floor and conditioned certificate; an empty complete slice yields a strict-cost cut. Unknown and Empty enumeration differ, and one bucket representative misses a second eligible original | Prove full captured-original enumeration, per-original machine lower-bound inequality, cost-floor and equality-slice evidence, actual tie identity/comparator, heap use, and Rust correspondence; neither source coverage nor binary64 admissibility follows from this finite model |
| CBC-05 nonvacuous evidence stopping | [EvidenceStopping.v](../verification/temporal_automata/theories/EvidenceStopping.v) defines finite eligible originals, exact observations, closed cost intervals, and feasible completions; constructs a completion under coherence/bound premises, proves full verified best-k is universally safe iff no feasible unresolved original can outrank its worst member, and proves underfull safety iff eligible unresolved originals are exhausted. Contradictory evidence, impossible intervals, improving completion, and safe/unsafe inhabited controls are checked | Produce sound intervals, exact verified scores, finite captured eligibility, and full verified heap choice from a concrete operation; connect cost order, numerical enclosure, result tags, and snapshot/Rust execution. This is semantic stopping, not an executable evidence solver |
| CBC-05 evidence indistinguishability | [EvidenceIndistinguishability.v](../verification/temporal_automata/theories/EvidenceIndistinguishability.v) proves feasible assignments agree on verified scores, and an unverified eligible score may be reassigned inside its independent interval while preserving evidence. Two inhabited completions with the same evidence have distinct, unique top-one answers, forcing a sound deterministic evidence-only decider to defer; contradictory evidence and the exhibited unresolved underfull vacancy also force deferral | Obtain the finite eligible set and independent interval semantics from a captured source operation; connect exact verified scores, rank order, binary64 bounds, evidence acquisition, and Rust result tags. The theorem is an information limit for this evidence language, not a minimum-work claim |
| CBC-05 attainable endpoint completion | [EvidenceAttainableCompletion.v](../verification/temporal_automata/theories/EvidenceAttainableCompletion.v) constructs a feasible endpoint assignment from well-formed identities, coherent exact observations, ordered independent intervals, and enclosed observations. For a full verified best-k, an eligible unverified lower-endpoint rank before the worst gives an attained answer-changing completion; at equal cost, an earlier tie refuses and a later tie does not. An unresolved underfull vacancy changes the answer regardless of rank, with inhabited controls | Produce the declared independent interval evidence and verified heap choice from one captured operation; prove any additional metric/recurrence correlations, actual comparator and numeric bounds, source validation, and Rust result behavior. No universal runtime-optimality claim follows |
| NC-1 geometric enclosure | [CertifiedGeometricEnclosure.v](../verification/temporal_automata/theories/CertifiedGeometricEnclosure.v) proves nonnegative lower and valid lower/upper endpoints from an exact-rational metric, a nonnegative radius, and each member's distance-to-center bound; strict-cutoff pruning follows. A two-point metric refutes an unscoped region and endpoints shifted inward | Produce a valid center/radius enclosure for every captured original; prove outward binary64 endpoint conversion and the actual verifier's numeric inequality before pruning; bind source regions, snapshot, and Rust branch behavior |
| QO-1 work potential | `CertifiedMetricExecution.v` telescoping inequality | Concrete primitive costs, independent credits, useful potentials, live/peak accounting |
| QO-2 finite portfolio minimum | `CertifiedMetricExecution.v` finite natural objective selection | Certified feasibility, acquisition/selection/conversion cost, workload and objective scope |
| Standard integer edit recurrence | [IntegerLevenshteinRecurrence.v](../verification/core/theories/Conformance/IntegerLevenshteinRecurrence.v) proves seed, one-label transition, streamed terminal, and both empty cases against `Core.LevDistance`; checked-add and checked-cell lemmas bind a finite maximum | Unicode/Rust label extraction, characteristic-vector and positional-state transitions, allocation/error semantics, and full source/session correspondence |
| Timestamped TWED finite source paths | [TwedSourceAlignments.v](../verification/twed/theories/Metric/TwedSourceAlignments.v) and [TwedSourceDpPath.v](../verification/twed/theories/Metric/TwedSourceDpPath.v) prove finite exact-natural paths on natural value/time pairs with unit-stiffness coordinate-sum distance and natural gap penalty, 2007 infinite axes, recurrence equivalence in both directions, and attained minimum; an altered λ-free axis yields 3 versus lawful 4 | Ideal-real source-distance and sequence-metric theorem, then binary64 operation graph, interval bound, index session, and Rust correspondence |
| Timestamped TWED metric transfer | `TimestampedMetricTransfer.v` recurrence/anchor transfer and abstract strict-domain admission | Committed scalar strict-domain API, ideal-real source-distance identification and local source sequence-metric theorem, binary64 and source correspondence separately |
| Range certificates and snapshot | `RangeCertificates.v`, `ElasticSnapshot.v` | Integrate with whole-session ownership and selected Rust execution paths |
| Original-occurrence ownership | `CertifiedSearchSession.v` proves pending/private/verified/excluded permutation, sound exclusion preservation, and conditional completed-winner retention | Region-level split, borrowed-parent publication, heap/rank, resource/error observations, and Rust correspondence |
| Explicit region split | `CertifiedRegionPartition.v` checks exact permutation of a supplied parent original list into terminal and child lists, including collision/duplicate controls | Prove captured snapshot enumeration, compact structural split certificate, and Rust iterator correspondence without enumerating whole subtrees in the hot path |
| Constructed region split | `CertifiedStructuralSplit.v` classifies every supplied parent original into a terminal or uniquely labelled child bucket, preserves exact multiplicity, and produces a package accepted by the explicit checker | Prove classifier and parent enumeration against a captured dictionary snapshot, then derive a compact certificate and source correspondence without scanning full subtrees in production |
| Exact workspace resources | `ExactWorkspaceResources.v` and `ResourceLedger` API | Model actual/reserved/charged work, vector capacity, private scratch, and allocation failure on all selected operations |

The standard integer edit bridge represents rows as total functions over
natural indices and uses the core ASCII `Char` model. Its cell-refinement
theorem applies to query indices within the query length, assumes the
predecessor row is reference-correct at those indices, and supplies the exact
new row's preceding horizontal cell. It certifies the unit-substitution
weight and the terminal coordinate, including an off-by-one control. Its
`checked_transition` accepts a cell only when **all three** candidate sums
fit the declared maximum. The checked overflow control uses arbitrary
primitive operands $`3,1,0`$ with maximum $`2`$ to show that this conservative
gate can reject when the exact minimum still fits; it does not exhibit a
reachable Levenshtein DP row. This theorem therefore does not
justify imposing that gate on a production scorer, nor does it prove a Rust
integer-overflow policy. A faster, less restrictive machine rule needs its
own soundness proof under the actual checked arithmetic and result tags.

A complete search-session invariant must count originals held by private work or prove that their public parent retains ownership until atomic publication. In the bounded range continuation, advancing the candidate cursor removes an original from its pending region; a `pending_match` awaiting a result-page slot therefore requires exclusive private ownership. Uncommitted node splitting can still borrow its pending parent. Verification and emission *histories* are ghost state; the concrete best-k heap, result vector, and continuation remain separately accounted. Cache entries are reusable only when complete and scoped to the same query, arithmetic, parameters, and snapshot. A pure cache-hit successor may preserve completed results while changing page boundaries or resource status.

## Validation and evidence protocol

The entry task audits the existing files and their theorem premises at the baseline revision. Formal changes use `systemd-run` resource limits and are recorded with exact commands and logs. A Rocq compile and `coqchk` establish kernel acceptance of the supplied proof terms; they do not establish an unstated Rust mapping. TLA+ finite model checks seek short counterexamples and keep their state bound visible. Differential and property tests are supporting execution evidence. Negative controls include origin-equal timestamped TWED, unknown summary treated as empty, duplicate-original node sharing, an omitted collision member, uncharged private work, incomplete dependency cuts, and changed arithmetic order.

The epic's planned review checkpoints A-D require the actual deliverables. The approval to create this ledger and epic is not a signoff on their later proof results.

## Checkpoint A review packet

The artifacts available for the first design review are this contract and
claim matrix, the certificate section of
[Certified metric execution](certified-metric-automata.md#3-construction-certificates-and-their-composition),
and [CertifiedContracts.v](../verification/temporal_automata/theories/CertifiedContracts.v).
The latter contains theorems
`accepted_exact_certificate_preserves_every_environment`,
`accepted_realization_implements_its_measure`,
`accepted_exact_certificates_compose_semantically`, and
`accepted_lower_certificate_bounds_every_environment`. The explicit stale
scope, wrong rule, broken chain, fabricated target, and lower-bound examples
are checked in the same file. Resource-limited `coqc` and `coqchk` have passed
for this finite natural-cost expression kernel; the manifest and README name
its narrow scope.

The checker now rejects `RoundedBinary64`, `OrderedResult`, and
`CanonicalWitness` even when both request and certificate carry the same
profile. It also binds the inclusive cutoff conservatively by exact equality.
This closes the illustrative arithmetic and observation-tag loopholes: its
natural-number cost proof cannot certify a rounded operation graph or a
canonical witness. The finite
[TLA+ acceptance model](../verification/tla/MetricCertificateReplay.tla)
explores 19 reason-classified scenarios, including a two-step replay and
separate exact/filter claims; the clean configuration passes, while bypassing
scope validation violates `ScopeMutationDetected`. Its witness field checks
presence only, and no TLA state identifies an actual Rust certificate payload. The
[model report](../verification/CBC_CERTIFICATE_REPLAY_REPORT.md) gives the
commands and exact limits.

**Proposed first source route:** direct refinement of the standard integer
Levenshtein transition path, from `StandardV::successors` through
`transition_standard_into`. The route must account for position fields,
characteristic-vector generation, checked index/error arithmetic,
subsumption, final cost, and any traversal/finalization covered by a claim.
A source-level proof technology such as Verus must establish the relevant
Rust statements; a separate relation to this Rocq expression model is needed
where conclusions cross tools. The trusted boundary includes the Rocq kernel,
the chosen Rust verifier and its Rust semantics, any translation bridge, the
compiler/runtime assumptions, and the actual binding of scope tokens to
request/revision data. These are **open proof obligations**, not consequences
of the four checked expression theorems.

The [source-correspondence decision](../verification/CBC_STANDARD_SOURCE_ROUTE.md)
specifies a first real executable path at cutoff 4 and names the exact
dispatch, characteristic cache, transition, closure, generated-state cache,
finality, and traversal edges that need source proofs. Its proof graph does
not promote the current Rocq and Verus mirrors to a Rust refinement claim.

The review decision is about (1) the contract and claim-level vocabulary,
(2) the acceptance interface and its narrow demonstrated checker, and (3)
the direct-refinement route and named trust boundary. It does not certify a
production automaton. Checkpoint B will review the complete search-session
model; C will review rank, numerical, and metric instances; D will review
quantitative results and any proposed production adoption.

**Checkpoint A design decision:** On 2026-09-29 the user selected
“Approve Checkpoint A design” for the presented packet. This approval covers
the design judgment, not the unproved executable correspondence, later
checker refinements, or Checkpoints B–D. The tracker dependency and evidence
gates still need to be satisfied before the checkpoint can be recorded as
verified.
