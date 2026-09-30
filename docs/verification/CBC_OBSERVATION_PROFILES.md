# Validation, failure, and resumption observations

This source-mapped contract serves pgmcp task `cbc-01-errors`. It refines the
observation profiles in [certified metric automata](../theory/certified-metric-automata.md#2-observation-profiles-and-lawful-executions)
and the result identities in the [result contract](CBC_RESULT_WITNESS_CONTRACT.md).
It describes current source behavior and the proof target for future rewrites;
it does not certify every implementation path. The
[numeric authority](CBC_NUMERIC_AUTHORITY.md) records known overflow and
result-tag gaps. No production experiment is executed here.

## 1. Two different preservation claims

Fix an operation and input request, a captured index revision, and a declared
score authority. **Completed-result equivalence** means that, whenever two
valid executions exhaust the same search and return `Complete`, their candidate
identities, scores, order, and promised witnesses agree. It allows different
internal schedules and cumulative work counts. It does not assert that both
executions complete at the same resource limit. A separate per-execution
fail-closed obligation forbids turning invalidity, resource failure, or an
unresolved search into a complete empty result. Validation-domain and
first-error preservation need their own stated claim; completed-result
equivalence alone does not establish either.

The **resource-sensitive trace profile** additionally retains the sequence of
committed prefixes or pages, pause reasons, exact partial subsets, resource
charges and peaks, cache observables where exposed, and error precedence. It
is a strictly stronger claim. Equal final answers do not imply equal trace
profiles, even under the same limits. Each optimization must state which
projection it preserves and prove that its *changed* trace remains within its
own declared safety and budget contract.

For a bounded call, [OperationOutcome](../../src/time_series/bounded.rs#L315)
separates `Complete { value, usage }` from
`Incomplete { partial, reason, continuation, usage }`.
[TemporalValidationError](../../src/time_series/bounded.rs#L271) is a separate
`Result::Err`, and [IncompleteReason](../../src/time_series/bounded.rs#L236)
distinguishes budget, arithmetic, numeric, allocation, stored-data,
cancellation, and unsupported failures. `AboveCutoff` and
`NoFiniteAlignment` are completed exact decisions only on an operation whose
result-tag correspondence has been proved; the scalar empty-side overflow
gap in [numeric authority](CBC_NUMERIC_AUTHORITY.md) prevents a blanket claim.
`OperationOutcome::usage` is cumulative for the session and records retained
peaks; [online step usage](../../src/time_series/automaton/mod.rs#L58) is for
one transition. The [resource ledger](../../src/time_series/bounded.rs#L432)
previews `charge_many` atomically and reports the first failing charge in its
listed order. A logical DP/work precharge is a budget convention, not a
measurement of CPU instructions or elapsed time.

| Profile | Required observable boundary | An admissible change needs |
|---|---|---|
| Online | Constructor error, consumed-prefix length, exact within-cutoff observation after every `Advanced`, and `Incomplete` with no consumed new unit. | Transition/validation equivalence at each committed prefix; rollback and charge proof for a failed step. |
| Bounded scalar/vector score | Request `Err` versus completed cutoff tag versus `Incomplete` reason; score and witness type when present. | Same domain and numeric authority, exact result mapping, and explicit overflow/structural distinction. |
| Completed range | Exactly the eligible originals and declared order after exhaustive completion. | Coverage and exact verifier proof; a changed partial page is allowed only under a completed-result projection. |
| Completed kNN | First $`\min(k,n)`$ eligible ranks under the operation's tie rule. | Heap and unresolved-region exclusion, including inclusive equality and $`k=0`$ validation. |
| Witness | Replayable feasible path, authoritative optimum score, and canonical tie selection only when promised. | Extraction and replay correspondence, plus numeric and tie proofs; replay of a merely feasible trace does not suffice. |
| Resumable range | Snapshot, committed cursor/private match, retained exact subset, page stop reason, and remaining work. | If execution completes, page partitions give the same completed answer; eventual completion requires page grants large enough for atomic phases, adequate cumulative budgets, and a progress premise. Cancel and terminal failure remain tagged incomplete. |
| Certified result | Accepted certificate, query/cutoff/snapshot binding, replay result, and typed rejection. | Proof that validation precedes admission; no malformed or stale certificate can become a completed result. |

The profiles are composable only when their named observations agree. In
particular, a proof of score-only equality does not imply witness, page,
resource, or error-precedence equality. A throughput improvement does not
authorize widening the semantic cutoff or suppressing an error.

## 2. Source precedence map

Precedence below is the order of the listed checks in the named function; it
is not a universal ordering across APIs. For example, a request with both a
bad cutoff and a nonfinite query may report different first errors on two
surfaces. Reordering checks is an observable change if the profile exposes
the error value. The table states only what is present in committed source.

| Surface | Checks before candidate work | Outcome boundary and limitation |
|---|---|---|
| [Shared scalar tagged score core](../../src/time_series/scoring.rs#L22) | Cutoff validity, kernel normalization, query length/finite samples, then candidate length/finite samples; structural reachability and empty-side arithmetic precede DP charging. | The ERP, TWED, and Fréchet wrappers call this core directly. Invalid requests return `Err`; resource/numeric failures return `Incomplete`; finite cutoff may classify a nonfinite online observation as `AboveCutoff`, while unbounded cutoff reports `NumericOverflow`. The [empty-side path](../../src/time_series/scoring.rs#L48) currently mislabels a reachable overflow as `NoFiniteAlignment`; CBC cannot use it as evidence of tag correctness. |
| [Banded DTW tagged score wrapper](../../src/time_series/scoring.rs#L193) | Cutoff validity, then band-width budget, then a length-difference structural branch that validates samples before returning `NoFiniteAlignment`, then squared cutoff and the shared core. | An over-budget band returns `Incomplete` before a nonfinite sample can be reported. Root-distance output and squared-cutoff arithmetic need a separate numeric correspondence argument. |
| [Raw and metric MSM tagged score](../../src/time_series/msm.rs#L337) | Raw MSM checks split/merge configuration, cutoff, then query/candidate length and finite samples. The [metric wrapper](../../src/time_series/msm.rs#L595) rejects either empty operand before delegating. | A metric empty-series error precedes an invalid cutoff; raw MSM permits the empty boundary. Neither uses the shared scalar core's validation order. |
| [Physical-time TWED tagged score](../../src/time_series/timestamped_twed.rs) | Typed operands are constructed separately; distance checks units, origins, cutoff, then query/candidate length before charging DP, work, and scratch. | A unit or origin mismatch precedes an invalid cutoff. A finite cutoff can complete `AboveCutoff` where an unbounded score path reports `Incomplete(NumericOverflow)`. |
| [Generic elastic online constructor](../../src/time_series/automaton/column.rs#L462) | Query length, finite samples, normalized kernel, finite valid cutoff, column domain, then frontier/resource setup. | [OnlineStepOutcome](../../src/time_series/automaton/mod.rs#L58) promises that an incomplete step leaves the preceding prefix committed. The source recurrence and every charge still need an instance proof. |
| [ERP online constructor](../../src/time_series/automaton/erp.rs#L402) | Query length, finite samples, cutoff (positive infinity allowed), positive frontier limit, then checked allocation and gap arithmetic. | Its `+inf` path can report numeric overflow where a finite cutoff can safely exclude an over-cutoff score. Do not substitute the finite-cutoff online profile. |
| [Physical-time TWED online constructor and step](../../src/time_series/automaton/timestamped_twed.rs#L75) | Constructor checks target unit, finite origin, matching origin, finite cutoff, then query length and state resources. [Advance](../../src/time_series/automaton/timestamped_twed.rs#L229) checks finite value, finite timestamp, monotonicity or origin, then depth and step work. | A first target timestamp equal to origin passes the online guard; later timestamps must strictly increase. Failed validation or incomplete work retains the preceding committed prefix. Constructor resource failures use `Err(TimestampedTwedError::Resource(...))`. |
| [Elastic bounded range](../../src/time_series/elastic/walker.rs#L2931) | Kernel cutoff validity, query length/finite samples, then continuation/query storage and workspace setup. | Paused pages keep the exact subset in the continuation with `partial: None`; [`exact_partial`](../../src/time_series/elastic/walker.rs#L1193) borrows it. Cancel or terminal failure may return a sorted `partial: Some`, or `None` if finalization fails. A complete empty vector is an exhaustive claim only after these gates. |
| [ERP automaton bounded range](../../src/time_series/elastic/walker.rs#L4654) | [Frontier construction](../../src/time_series/elastic/walker.rs#L4664) precedes query copying and traversal setup; its query/cutoff/resource checks are those of `ErpFrontierMachine`, not the generic range constructor. | Construction and query-copy allocation failures can be `Err(TemporalAutomatonError::Resource(...))`, while later terminal/page failures use `OperationOutcome::Incomplete`. This path needs its own error and continuation correspondence. |
| [Elastic bounded kNN](../../src/time_series/elastic/walker.rs#L3124) | Query length/finite samples precede $`k=0`$/empty-index return; then result charge/reservation and scorer setup. | A resource stop returns `Incomplete` with no public partial heap. [DP precharge](../../src/time_series/elastic/walker.rs#L3221) currently occurs before the candidate bound; a bound-first rewrite changes charges and possibly completion at a fixed budget. |
| [Timestamped TWED range](../../src/time_series/timestamped_twed_index.rs#L387) | Query unit/origin identity, query length, then cutoff, then product construction. | Page work/results checks precede candidate scoring. [Resume/cancel](../../src/time_series/timestamped_twed_index.rs#L1196) distinguish continuation-owned exact partials from terminal partials. |
| [Timestamped TWED kNN](../../src/time_series/timestamped_twed_index.rs#L420) | Query unit/origin identity and length precede $`k=0`$/empty-index return; then result charge, checked scratch sizing and allocation, heap reservation, and scan. | Checked scratch-size arithmetic currently returns `Err(TimestampedTwedIndexError::Resource(...))`. Other setup resource failures return `Incomplete` with `partial: None`; failures during scanning return `Incomplete` with a sorted verified heap in `partial: Some`. That heap is not a certified final top-$`k`$. |
| [MSM witness score](../../src/time_series/alignment.rs#L451) | MSM configuration, cutoff, query length/finite samples, candidate length/finite samples, then empty boundary and matrix setup. | Witness replay validates a path and score, not global optimality. Other [temporal witness scorers](../../src/time_series/alignment.rs#L1536) validate cutoff before query/candidate samples; their kernel-specific setup and result tags remain distinct. [Banded DTW witness](../../src/time_series/alignment.rs#L1610) checks band budget between cutoff and sample validation, so it can return `Incomplete` first. |
| [Vector bounded scorers](../../src/time_series/vector.rs#L966) | ERP validates both typed series before cutoff/length; vector TWED and banded DTW validate the pair before cutoff/length; Fréchet checks dimensions before cutoff. | This is a family of source-specific `VectorMetricError` precedences, not the scalar order above. [Vector DTW](../../src/time_series/vector.rs#L1620) reports an over-limit band as `Err(InvalidConfiguration)` after pair/cutoff/length checks, unlike scalar DTW's `Incomplete`. The [vector Fréchet online constructor](../../src/time_series/vector.rs#L2106) instead checks its finite cutoff first. |
| [Certified elastic range](../../src/time_series/elastic/walker.rs#L3859) | Cutoff, query finite validation, supported nonempty interval domain, then checked proof/workspace resource setup. | Typed `ElasticCertificateError` and complete evidence are required; no partial certificate is promoted after storage failure. Certificate validation binds exact query, cutoff, and snapshot. |
| [Elastic range certificate verifier](../../src/time_series/elastic/walker.rs#L4620) | Snapshot, cutoff, query length, and query-bit binding are compared before generating and comparing fresh evidence. | A binding mismatch returns `Ok(false)` before query/cutoff validation. If binding matches, generation can return a typed error. This is a distinct precedence from certificate construction. |
| [Generalized string `scaled_distance`](../../src/transducer/generalized/automaton.rs#L677) | Fallible scale/operation validation and checked relaxation occur in the operation. | `try_accepts` preserves the error; [`accepts`](../../src/transducer/generalized/automaton.rs#L663) collapses it to `false`. A Boolean `false` therefore is not an above-budget certificate. Base dictionary query iterators return items without the bounded temporal `OperationOutcome` profile. |

The vector and timestamped constructors also validate their typed operands at
construction. A public `k=0` rule applies *after the checks actually performed
by that API*, not before all validation by fiat. No table row grants
permission to reorder them without either preserving the same first error or
explicitly selecting a weaker, reviewed observation projection.

## 3. Continuations, pages, and private work

The [elastic range continuation](../../src/time_series/elastic/walker.rs#L1248)
stores accumulated results and a `pending_match` separately. If an exact
candidate has been scored but the result page is full, it retains that match
before advancing to another candidate; [resumption](../../src/time_series/elastic/walker.rs#L1470)
publishes it after the next result-slot preflight. A page pause exposes
`partial: None` and `continuation: Some(...)`, while `exact_partial()` borrows
the retained exact subset. Cancellation has no continuation and remains
`Incomplete` even when its partial vector is nonempty.

The [TWED range continuation](../../src/time_series/timestamped_twed_index.rs#L1277)
tests page result/work limits and charges candidate, DP-cell, and cumulative
work limits before exact scoring. It advances the candidate cursor after the
score, then reserves output storage and charges a result slot for an eligible
match. Failure in that latter phase terminates `Incomplete` without a
continuation; the just-scored match is absent from the terminal partial.
A proof for a changed schedule must account for all private phases and avoid
losing an eligible original on a resumable pause. The [session ownership theory](../theory/certified-metric-automata.md#4-a-session-invariant-that-covers-the-whole-search)
models this as a partition of pending, private, verified, and soundly excluded
originals; its abstract theorem is not yet Rust correspondence.

An online [incomplete transition](../../src/time_series/automaton/mod.rs#L58)
leaves the prior prefix as the public state. This semantic rollback does not
erase already performed work or peak storage from the ledger. Cache and arena
updates must be published only after their scope and resource obligations
hold; a resource failure is not a semantic dead state eligible for caching.

## 4. Two negative controls for trace equivalence

**Cache hit versus miss.** Suppose an exact state/observation key has a
certified pure successor. A hit can reuse that successor while a miss must
construct it, compare exact keys, and possibly reserve an arena/cache entry.
Under a page work ceiling between the hit and miss costs, the hit may advance
and the miss may pause. Both can yield the same completed result after enough
budget. [LOCPA CC-1](../theory/lazy-ordered-cost-product-automata.md#7-operations-cursors-and-product-zippers)
authorizes successor equality only with complete keys and a pure transition;
it does not equate work traces. The [bounded ERP transition cache](../../src/time_series/automaton/erp.rs#L233)
and [TWED product transition cache](../../src/time_series/timestamped_twed_index.rs#L1476)
are source sites whose actual hit/miss charges and publication order need a
separate correspondence proof. The budget example is a counterexample to an
*inference* of trace equality, not a measured claim about those sites.

**Bound-first versus DP precharge.** The current bounded elastic kNN scan
[charges worst-case DP work](../../src/time_series/elastic/walker.rs#L3221)
before evaluating the candidate lower bound. An admissible bound-first
schedule could charge the bound, reject a candidate whose bound is strictly
above the inclusive cutoff, and avoid that DP charge. At a budget between
those charges, the old path may stop `Incomplete` while the new path can
finish. [LOCPA RA-1](../theory/lazy-ordered-cost-product-automata.md#8-online-semantics-stability-and-stack-safety)
proves a generic completed-result preservation rule under bound admissibility,
preflight, and ownership premises. It does not prove the Rust bound, all
charges, or equality of status/page traces. Equality-bound candidates cannot
be rejected by this rule without the operation's separate tie certificate.

These examples require an explicit review decision before claiming full
resource-sensitive trace refinement. They do not prevent a completed-result
optimization that preserves validation and fail-closed status while charging
and reporting the work actually performed.
