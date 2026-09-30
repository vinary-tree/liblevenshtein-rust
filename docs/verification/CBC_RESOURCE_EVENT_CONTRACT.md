# Primitive work, storage, and budget contract

This source-mapped design serves pgmcp task `cbc-01-resources`. It refines the
resource field of the [CBC obligation ledger](../theory/metric-automata-cbc-obligations.md)
and the quantitative semantics of [certified metric automata](../theory/certified-metric-automata.md#7-quantitative-semantics-and-precise-optimality-claims).
It specifies the event vocabulary needed to study bounded temporal scores and
dictionary products. It is not a proof that all current Rust paths implement
the model, nor a benchmark or production optimization.

## 1. Separate five quantities

For one operation or resumable session, use these nonnegative quantities:

| Symbol | Meaning | Reset and measurement boundary |
|---|---|---|
| $`W_e`$ | Primitive work actually executed, including failed probes, hashing, comparison, conversion, and finalization | Monotone through the whole session; an incomplete or rolled-back transition does not refund it. This is a *specified logical count*, not elapsed time or CPU instructions. |
| $`W_r`$ | Work capacity reserved before a phase, including worst-case DP precharge | A reservation can exceed later executed work. It is not an observed execution count. Track its committed or unused status explicitly. |
| $`W_c`$ | Work units charged to the public `ResourceLedger` | The existing `ResourceUsage.work_units` is cumulative and may be precharged or charged after work, depending on the operation. No universal equality with $`W_e`$ or $`W_r`$ follows from the API. |
| $`M`$ | Bytes currently owned and live inside the selected operation | Sum disjoint ownership compartments, including heap referents owned by generic payloads; release or transfer changes $`M`$. Caller-owned input, pre-existing shared index storage, and allocator metadata are outside this selected logical boundary and must be stated separately. |
| $`P`$ | Maximum live bytes reached so far | $`P'=\max(P,M')`$ after every allocation, release, transfer, or temporary phase. It is not process RSS. |

The semantic inclusive cutoff $`\tau`$ filters distances. The hard
[ResourceLimits](../../src/time_series/bounded.rs#L71) cap cumulative counts
and selected peaks for one session; [PageBudget](../../src/time_series/bounded.rs#L151)
limits one resume call's logical work and newly accepted results. A page
counter resets on resume; hard cumulative usage and retained state do not.
None of these resource ceilings changes $`\tau`$ or authorizes a different
distance comparison. A zero page allowance may pause indefinitely; eventual
completion requires sufficient page grants and a progress premise.

The public [ResourceUsage](../../src/time_series/bounded.rs#L170) combines
cumulative counts and resource-specific peaks. Its `scratch_bytes` and
`continuation_bytes` can describe overlapping storage from different views;
adding those fields is not a total-live-byte calculation. `Results` can count
reserved slots before any item is produced, and it never reports their
allocation capacity in bytes. `ResourceLedger` treats `QueueEntries` as a
peak count; an online step may instead report its current active-position
count for that transition. Neither is backing bytes. A total-space theorem
therefore needs the disjoint
ownership map below, even when each public counter stays within its
own ceiling.

## 2. Primitive event semantics

Let an event carry an operation ID, phase, owner, resource kind, amount, and
outcome. All arithmetic is checked before committing a finite counter. The
table is a specification for future correspondence checks; current Rust
functions implement only some of these distinctions explicitly.

| Event | State update and unit | Required failure or observation rule |
|---|---|---|
| Validate input/configuration | Establish domain, dimensions, origin/unit, cutoff, and structural conditions; no work budget becomes a semantic cutoff. | Preserve the selected API's [first-error order](CBC_OBSERVATION_PROFILES.md#2-source-precedence-map). Invalidity is an error, not a complete empty answer. |
| Preflight cumulative charge | Preview $`W_c+w`$ or another cumulative count, in that resource's units, against its hard ceiling. | Checked overflow gives `ArithmeticOverflow`; excess gives `BudgetExceeded`. [`charge_many`](../../src/time_series/bounded.rs#L432) previews atomically and reports the first failure in its listed charge order. |
| Reserve work | Add a declared upper-bound reservation $`r`$ to $`W_r`$ and, if the API uses precharge, to $`W_c`$. | Must happen before a phase whose hard work bound depends on it. A reservation is evidence of budget admission only with a proof that the phase's executed work is covered. Do not label it executed work. |
| Execute primitive | Add its nonnegative logical cost to $`W_e`$; include unsuccessful searches and comparisons. | Never refund on rollback, cancellation, cache eviction, or retry. A bounded execution must show either prior coverage by reservation or safe per-step preflight. |
| Check page allowance | Preview call-local work or new-result count against `PageBudget`. | Pause before the next indivisible phase if it will not fit; retain its original or private ownership in a continuation. This does not reset the cumulative ledger. |
| Acquire allocation | Add retained capacity and owned referent bytes to exactly one owner in $`M`$; update $`P`$. | Check size arithmetic and relevant logical ceiling before use. `try_reserve` may grow by more than requested; `Vec` growth, map rehash, and conversion may temporarily retain old and new allocations together. These overlaps and subsequent capacity must be included in any concrete live-byte proof. Allocation failure is `Incomplete` or `Err(Resource)` as specified by the API. |
| Release or transfer | Subtract bytes only after actual release; transfer moves one allocation between owners without duplicating it. | No decrease in $`P`$, $`W_e`$, or cumulative $`W_c`$. A moved result vector is one allocation, not two. |
| Observe peak | Check $`M`$ or a declared category projection after construction and every growth point. | Peak observation is not a substitute for preflight if allocation can breach a hard ceiling before observation. [`observe_peak`](../../src/time_series/bounded.rs#L445) records a selected category maximum, not the whole operation's RSS. |
| Publish result/transition | Transfer a fully scored original or complete successor from private state to visible results/cache. | Preflight storage and result-slot charges for the chosen schedule; on failure retain resumable ownership or terminate `Incomplete`. Never publish a partially proved result. |
| Finish, cancel, or fail | Release or return owned state; report completed answer only after exhaustion. | Report actual public charges/peaks and the selected error/partial profile. Failure may leave work spent and a partial result; it does not imply the search space was exhausted. |

Reservation accounting needs a phase invariant. If $`R_i`$ is the unused part
of a proved reservation for phase $`i`$, executing cost $`w_i`$ requires
$`w_i\le R_i`$ and updates $`R_i'=R_i-w_i`$. It does **not** decrement
$`W_r`$, $`W_c`$, or $`W_e`$. A phase that has no reservation must preflight a
new charge before executing work if its public limit is meant to be hard.
The source currently has mixed precharge and incremental-charge schedules;
their trace profiles cannot be equated merely because final answers agree.

## 3. Ownership compartments and source witnesses

At the selected abstraction level, partition operation-owned storage into
the following disjoint compartments. A real allocation belongs to one
compartment at a time, even if several public resource categories describe it.
The list covers the selected scalar/vector bounded scorer, temporal online
machine, elastic/ERP product, physical-time TWED product, witness, and
certificate paths. A concrete theorem must enumerate additional allocations
introduced by its specific implementation.

| Owner | Retained and transient contents | Source witness and required accounting |
|---|---|---|
| Query and normalization | Owned query copy, normalized kernel/configuration, and typed value/time conversion with any owned referents | [ERP online constructor](../../src/time_series/automaton/erp.rs#L402) and [TWED typed series](../../src/time_series/timestamped_twed.rs). Caller-borrowed input is excluded; an owned copy is included. |
| Exact workspace | Query plan, two frontier generations, active-row arrays, candidate-local scratch and conversion temporaries | [`ExactPointWorkspace`](../../src/time_series/automaton/column.rs#L31) exposes retained and construction-peak sizes. [ExactWorkspaceResources.v](temporal_automata/theories/ExactWorkspaceResources.v) proves a natural-number plan-first maximum, conditional on sizes; it does not account for allocator overhead or all Rust growth points. |
| Product arena | Interned canonical states, position vectors, fingerprint table and collision buckets, plus temporary candidate state | [`TemporalStateArena::intern`](../../src/time_series/automaton/arena.rs#L67) has state and position ceilings plus fallible reserves. Distinct-state count alone does not bound backing capacity, map bucket/control storage, or temporary overlap. |
| Transition cache | Hash table, FIFO order, key/value copies, and retained empty-successor entries | [`BoundedTransitionCache`](../../src/time_series/automaton/cache.rs#L9) accounts table and order capacities. An entry ceiling is not itself a byte ceiling; cache hit/miss also changes executed work. |
| Traversal metadata | DFS frames, inline edge pages, cursor metadata, pending exact match, and any query-owned traversal arena or graph | [Elastic range peak observation](../../src/time_series/elastic/walker.rs#L1371) and [TWED product peak observation](../../src/time_series/timestamped_twed_index.rs#L1660) combine this compartment with arena/cache, scratch, query, and results in their continuation projections. [TraversalSession](../../src/transducer/dictionary_traversal.rs#L971) can hold an owned arena, a graph `Arc` and owner, or a native owner; a cursor's unvisited dictionary fanout alone is not a materialized frame. |
| Ranked results | Range result vector or kNN heap with capacity, owned `V` payload referents, and temporary finalization storage if simultaneously live | [Elastic bounded kNN](../../src/time_series/elastic/walker.rs#L3124) precharges result slots. [Bounded range finalization](../../src/time_series/elastic/walker.rs#L3732) allocates a permutation vector but rearranges the existing result vector in place; [bounded kNN finalization](../../src/time_series/elastic/walker.rs#L3791) allocates an overlapping output vector. [Cloning an ID](../../src/time_series/elastic/walker.rs#L1510) may allocate independently of `size_of::<V>()`. Count a move once and simultaneous buffers while both live. |
| Witness and proof | Predecessor matrix, alignment operation vector, certificate evidence, query binding, verification scratch | [MSM witness extraction](../../src/time_series/alignment.rs#L451) preflights matrix/witness resources; [elastic certificate construction](../../src/time_series/elastic/walker.rs#L3859) has separate witness/work ceilings. A returned witness transfers ownership to the caller; it leaves operation-owned $`M`$ at that handoff, but remains live in a whole-process accounting model. |
| Snapshot output and I/O | Emitted rolling-window copies, serialization/deserialization buffers, and bytes read or written | [`ResourceKind::SnapshotBytes`](../../src/time_series/bounded.rs#L62) is documented as persistent-snapshot I/O, but [rolling-window usage](../../src/time_series/rolling.rs#L299) reports bytes of snapshots emitted by one step. Its interpretation is API-specific; neither count alone bounds peak live snapshot buffers. A copied rolling snapshot is live until transferred or released. |

The continuation is an **aggregate holder**, not an extra ownership row: it
contains or borrows the query, traversal metadata, arena, cache, workspace,
and results above. A source map must partition actual ownership within it.
Container headers, map control/bucket arrays, and recursively owned payloads
must either be charged in the concrete owner or listed as explicit modeling
exclusions. A `HashMap.capacity() * size_of::<entry>()` estimate covers payload
slots only; it is not a complete bound on that map's allocations. The same
applies to generic `V`, kernel, query-plan, and dictionary-node referents.
This contract does not claim current source counters include them.

A pre-existing shared dictionary graph is attributed once to the index or
snapshot environment, not once per query holding an `Arc`. The query counts
its own handle, any owned traversal arena, and any graph materialized by the
query. Its `Arc` may pin the shared graph's lifetime, so a whole-library peak
claim must include the graph until the last holder releases it. An operation
peak that excludes it must say so. Result and witness handoff likewise moves
ownership to the caller; peak process memory may remain high after the
operation's $`M`$ decreases.

One implementation can project the same owned allocation into `ScratchBytes`
and `ContinuationBytes` for two independent ceilings. It must not count it
twice in the disjoint $`M`$ sum. Conversely, omitting workspace scratch from
both the total-live calculation and every applicable category would make a
space certificate unsound. [ERP's frontier machine](../../src/time_series/automaton/erp.rs#L107)
splits remaining logical scratch allowance between the required arena and
optional cache; that split is a ceiling policy, not evidence that both consume
their entire allowance or that the allocator uses exactly those byte totals.

## 4. Source accounting and trace obligations

The source-level [ledger](../../src/time_series/bounded.rs#L377) stores one
`ResourceLimits` value and committed `ResourceUsage`. Its `charge` uses checked
addition, `charge_many` uses preview then commit, and `observe_peak` retains a
maximum. [`ResourceKind` and `ResourceLimits`](../../src/time_series/bounded.rs#L36)
classify length/dimension/band validation and resource ceilings;
[`ResourceUsage`](../../src/time_series/bounded.rs#L170) records cumulative
DP/work/node/edge/candidate/result charges and scratch/queue/witness/
continuation peaks. Its snapshot field is surface-specific: cumulative
persistent I/O in its declared contract, or per-step emitted output in the
rolling-window source. Length, dimension, and band have no usage fields.
These are useful deterministic budget observations. They do not expose all
five quantities in section 1.

For example, [elastic bounded kNN](../../src/time_series/elastic/walker.rs#L3221)
charges worst-case DP and work before testing a candidate lower bound. If the
bound excludes the candidate, the charged units exceed that candidate's
executed DP work. A bound-first schedule may complete at a hard work limit
where the present schedule stops. The [observation contract](CBC_OBSERVATION_PROFILES.md#4-two-negative-controls-for-trace-equivalence)
therefore grants conditional completed-answer comparison only, unless a
stronger same-budget trace proof is supplied. Online
[`OnlineStepOutcome::usage`](../../src/time_series/automaton/mod.rs#L58)
instead reports one transition; a proof cannot silently add that quantity to
an unrelated cumulative session ledger.

All future concrete instantiations must give a source event map: where a
phase is preflighted, where each allocation may grow, where work executes,
when a cursor advances, where a result/cache entry is published, and which
error is returned on each failure. The event map must include construction,
failed probes, collision checks, finalization, and replay. It must establish
checked arithmetic, a no-use-before-admission rule for hard ceilings, and
logical live/peak upper bounds under the operation's selected ownership
boundary. A proof of per-category `ResourceUsage` bounds alone is narrower.

## 5. Negative controls and exact limits of this design

**Omitted scratch.** Suppose a plan retains $`a`$ bytes, a two-generation
frontier retains $`b`$ bytes, and a later arena retains $`c`$ bytes, all live
together. A claimed live bound $`a+c`$ omits $`b`$ and fails whenever
$`b>0`$ and the hard ceiling lies between $`a+c`$ and $`a+b+c`$. The checked
[workspace peak theorem](temporal_automata/theories/ExactWorkspaceResources.v)
uses retained workspace plus later state. When its `plan_retained`, `frontier`,
and `later` arguments are correctly bound to $`a,b,c`$, its preflight rejects
the bound $`L`$ with $`a+c\le L<a+b+c`$. The theorem does not inspect the
Rust heap or automatically detect an omitted source allocation. A source
instance still has to prove its $`a,b,c`$ correspond to actual retained capacities and
temporary construction overlap.

**Refunded executed work.** Suppose a failed transition hashes a key and
builds a candidate successor at positive logical cost $`h`$, then leaves the
semantic prefix unchanged. A model that restores $`W_e`$ to its old value
after rollback reports zero work for the failure. Repeating the same failed
step $`n`$ times executes at least $`n h`$ work while that model reports zero.
The monotone execute-event rule rejects the refund. A precharged public ledger
may report a different number; it must not be mislabeled as actual work.

These controls test the vocabulary and selected model. No present theorem
proves exact CPU cycles, allocator metadata, peak RSS, or globally minimum
time/space for a Rust automaton. A later adoption decision uses the
[predeclared performance gate](../theory/certified-metric-automata.md#74-future-production-adoption-time-and-space-gate)
alongside concrete correctness and resource correspondence.
