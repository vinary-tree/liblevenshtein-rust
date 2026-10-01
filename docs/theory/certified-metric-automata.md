# Certified execution and optimization of metric automata

**Status:** theory extension and CBC design contract; generic proof kernels
are mechanized, production adoption is separately identified.
**Companions:** [Ordered Residual Calculus](ordered-residual-calculus.md) (ORC)
and [lazy ordered-cost product automata](lazy-ordered-cost-product-automata.md)
(LOCPA).

A **correct-by-construction (CBC)** automaton is obtained through constructions
whose evidence establishes its requested observations. This requires a
connection between the specification, each accepted transformation, and the
executable artifact. A correct recurrence, a reviewed marker trait, or a
proof about an analogous model supplies part of that connection.

This extension specifies execution observations and the certificates needed
to preserve them; strengthens ordered kNN pruning with cost-conditioned tie
information; and gives separate meanings to optimization of results, pruning,
representations, and execution resources.

## 1. What the review changes

| Existing result | Additional obligation or stronger result |
|---|---|
| R2 preserves outputs after each input | Execution must account for invalidity, witnesses, incomplete outcomes, and divergence |
| R2a permits any finite switching schedule | Internal optimization steps need a progress argument; infinitely many conversions can prevent completion |
| BF-1 uses a global subtree tie floor | BF-2 needs a tie floor only among candidates that can attain the cost bound |
| Admissible scalar cost bounds combine by maximum | Conditioned rank bounds combine as whole lexicographic pairs; their coordinates cannot be mixed freely |
| A finite residual quotient is state-minimal | State count supplies neither minimum memory bytes nor fastest execution |
| A transformation preserves completed answers | Its changed resource trace and page behavior require their own contract |
| A certificate describes a mathematical machine | Acceptance must bind domain, arithmetic, observation, scope, and executable correspondence |

“Highly optimized” means choosing among certified alternatives under an
explicit resource objective and workload. A stronger abstract bound can reduce
exact verification while increasing total work. A smaller state machine can
require more expensive transitions. These tradeoffs belong in the model.

## 2. Observation profiles and lawful executions

Fix a contract
$`\Gamma=(D,q,\theta,\mathcal N,\sigma,\tau,\prec,\mathcal W,\mathcal B)`$:

- $`D`$ is the validated input domain, including any quotient identity;
- $`q,\theta`$ are the query and immutable scoring parameters;
- $`\mathcal N`$ fixes arithmetic, primitive order, rounding, and overflow;
- $`\sigma`$ is the captured dictionary revision, where a dictionary is used;
- $`\tau`$ is the inclusive semantic cutoff;
- $`\prec`$ specifies result identity, ordering, and ties;
- $`\mathcal W`$ specifies witness observations; and
- $`\mathcal B`$ specifies work, storage, and paging limits and their units.

A **search snapshot** is the immutable logical view of indexed data captured
when an operation begins. It fixes the meanings of original identities,
buckets, summaries, payloads, and any cache observations used by that
operation. The revision $`\sigma`$ identifies the view; a revision number by
itself does not prove that subsequent reads use the same data. Publishing a
newer index view must not retarget an active operation's reads. A snapshot need
not be a full physical copy or an on-disk serialized snapshot: shared storage
is valid when every retained referent continues to denote the captured view.

The snapshot obligation applies to **indexed search operations**, not to every
metric automaton. A standalone online automaton has a fixed owned query and
configuration and consumes a target stream; it has no dictionary revision to
capture (for example,
[`TimestampedTwedOnlineAutomaton`](../../src/time_series/automaton/timestamped_twed.rs)).
An indexed range or kNN session must keep its dictionary reads,
original identities, summaries, and cache evidence tied to one logical view
across every page and resume. The
[elastic traversal](../../src/time_series/elastic/walker.rs) captures a root
and its range continuation immutably borrows its index; the
[timestamped TWED range continuation](../../src/time_series/timestamped_twed_index.rs)
retains a captured root and term count. These are existing implementation
mechanisms to audit against that obligation. They do not, by themselves,
establish the full source-to-model proof. The portable, checksummed snapshot
file format is a separate persistence mechanism. This logical contract does
not prescribe copying the index or adding a snapshot allocation to each
automaton transition. The [indexed session lifecycle diagram](../diagrams/architectures/cbc-indexed-session-lifecycle.svg)
shows which private phases and retained state must remain bound to that view.

The distance cutoff, cumulative resource ceilings, and a per-call page budget
are different quantities. Changing one does not authorize changing the others.

An **observation profile** selects what a client can distinguish.

| Profile | Required observation |
|---|---|
| Online score | Validity and exact or cutoff-tagged score after every committed prefix |
| Completed range | Exactly the declared candidates, scores, identities, and promised order |
| Completed kNN | The first $`\min(k,|U|)`$ exact ranked candidates in the finite eligible universe $`U`$ |
| Witness | The specified witness or canonical optimum, including tie decisions |
| Resumable operation | Snapshot, committed progress, reason tag, and a continuation denoting all remaining work |
| Resource safety | Charges, reservations, live storage, and peak bounds in declared units |

A kNN contract must define eligibility, including structural nonmatches and
numeric failures. The expression above assumes each member of $`U`$ has an
eligible exact rank; failures are represented separately. For $`k=0`$,
the empty result follows only after the contract's required validation.
Error precedence is part of the API when observable.

Write a step as $`s\xrightarrow{\alpha,w}s'`$, where $`\alpha`$ is a finite
list of observable events and $`w`$ records resource use. Empty $`\alpha`$
denotes internal work, such as a cache probe or a representation conversion.
An observation projection may erase internal events; it must not erase
invalidity, overflow, completion tags, or required witness distinctions.
A change of result schedule needs a profile allowing that schedule, or an
explicit relation between traces. Erasing events alone does not justify
permuting ordered outputs.

Cache hit and miss paths may compute the same semantic successor while
pausing at different positions under a fixed work limit. Thus LOCPA CC-1
certifies the pure successor; it does not assert equality of resource traces
or partial pages.

## 3. Construction certificates and their composition

### 3.1 The certificate package

A checked artifact binds evidence to one specification and realization.
The observation profile determines which obligations apply.

| Evidence | Proposition it establishes |
|---|---|
| Domain | Accepted inputs satisfy $`D`$; invalid inputs follow the declared rejection behavior |
| Realization | Seed, transitions, and final observations satisfy R2 |
| Abstraction | Every represented original is covered and each bound is admissible in $`\mathcal N`$ |
| Reduction | Normalization, interning, and caches preserve observations under complete scope keys |
| Execution | Concrete transitions simulate reference transitions and preserve the session invariant |
| Progress | Internal work terminates or returns a permitted incomplete outcome; completion has no missing work |
| Resources | Executed work and retained allocations satisfy the ledger and storage model |
| Geometry | The untruncated measure has the requested metric or quotient laws on $`D`$ |
| Artifact correspondence | The executable realizes the model to which these propositions refer |

A certificate contains checked proof terms or a successful check by a validator
whose soundness has been established. A Boolean, source hash, or sealed marker
alone does not prove a semantic proposition. Hashes locate artifacts; canonical
payload equality and the declared integrity mechanism bind their identity.
Source changes invalidate correspondence evidence unless a new certificate
explicitly covers them.

An implementation can use a proved generator with a proved lowering, or
validate each generated implementation separately. Its trust boundary names
the proof checker, primitive arithmetic model, extraction or translation path,
and runtime assumptions. Monomorphized recurrences, SIMD operations, and
specialized storage are compatible with CBC when their correspondence is
covered. This follows the established separation of proof production and
checking in [proof-carrying code](https://people.eecs.berkeley.edu/~necula/Papers/pcc_fmcs97.pdf).

### 3.2 CBC-1: local certificates compose into execution preservation

Let $`I`$ be an implementation, $`S`$ a reference system, and $`R`$ relate
their states. Require related initial states. For every related pair and
implementation step, require

```math
R(i,s)\land i\xrightarrow{\alpha}_I i'
\Longrightarrow
\exists s'.\ s\xRightarrow{\alpha}_S s'\land R(i',s').
```

Here $`\xRightarrow{\alpha}`$ is a finite sequence of reference steps whose
event lists concatenate to $`\alpha`$. An internal implementation step may
correspond to zero reference steps. Require separately that a concrete
completed state relates to a reference completed state with the same selected
result observation.

**Theorem CBC-1.** These obligations imply that every finite implementation
execution has a reference execution with the same projected trace and related
final states. Completed observations therefore satisfy the reference contract.
Two such refinements compose by relational composition.

**Proof.** The empty execution uses the initial relation. For a nonempty
execution, the local certificate supplies a matching reference fragment and
the relation at its end. Apply induction to the remaining execution and
concatenate the reference fragments. For composition, lift the second local
certificate over the entire intermediate fragment using the same induction.
The intermediate related state witnesses relational composition. Finally
apply the completed-state obligation.

[CompCert's small-step framework](https://compcert.org/doc/html/compcert.common.Smallstep.html)
provides a primary example of trace-based simulation methodology.
[CertifiedMetricExecution.v](../verification/temporal_automata/theories/CertifiedMetricExecution.v)
mechanizes the finite execution and composition argument here. Instantiating
the reference with the Rust library remains a separate proof.
Its `two_local_certificates_preserve_completed_observation` theorem also makes
the two initial relations, concatenated event segments, and both terminal
observation relations explicit. It permits finite silent administrative steps
through the local simulation premise. The coinductive
`infinite_zero_conversions` control has a sound silent local refinement and an
infinite conversion-only execution that never reaches its completion state.
Thus this finite theorem supplies no total termination claim; the rank
obligation in CBC-2 is separate.

[CertifiedCertificateTrace.v](../verification/temporal_automata/theories/CertifiedCertificateTrace.v)
instantiates CBC-1 for the checker below. A fixed program contains source and
target score expressions with a scoped certificate for each instruction. An
executable score step is admitted only when the certificate checker accepts
an `Equivalent` exact-natural, score-only rewrite; it emits the target value.
The reference step emits the source value. The checker theorem makes those
values equal for every variable environment, so each executable step has a
matching reference event. A silent administrative step increments a concrete
counter and stutters in the reference. From related empty histories at cursor
zero, the finite-trace theorem preserves the concatenated emitted score list;
if the executable cursor reaches the program length, the reference reaches it
with the same output history. A checked one-score program actually completes,
while a coinductive infinite administrative run from that same unfinished
program shows why finite refinement does not establish progress. This is a
checker-to-model bridge for one expression fragment, not a correspondence for
Rust evaluation, resource charges, witnesses, or the whole ORC realization.

### 3.2a A checked exact-integer certificate kernel

[CertifiedContracts.v](../verification/temporal_automata/theories/CertifiedContracts.v)
gives a small executable checker for expressions built from natural-valued
variables, constants, addition, and minimum. A certificate contains a sequence
of named rewrites and the source and target expression at each step. The
checker recomputes every rewrite target, checks adjacency between steps, and
checks equality of the supplied contract scope, including a conservative
exact cutoff match. It accepts only the explicit `ExactNaturals` arithmetic
profile and `ScoreOnly` observation profile; rounded arithmetic, ordered
results, and canonical-witness claims require different checkers. Its theorem proves that an
accepted exact certificate preserves expression value for *every* variable
environment and therefore every finite list of score observations. A distinct
lower-certificate type proves only that its result does not exceed the exact
expression. The file checks stale-snapshot,
changed-query, changed-parameter, changed-arithmetic, changed-observation,
missing-step, wrong-rule, broken-chain, fabricated-target, and valid-lower
examples. An integer
measure specification names the source expression; acceptance proves the
named target expression implements that measure in every environment.

This is a complete checker **for that expression language**. Its scope fields
are opaque tokens: the caller still has to bind them to the actual query,
parameters, arithmetic, snapshot, and observation profile. The language does
not encode the whole ORC realization IR, overflow, Rust control flow,
dictionary ownership, witnesses, or resources. Its acceptance theorem is
therefore one local certificate obligation, not executable CBC for the
library.

[CertificateScope.v](../verification/temporal_automata/theories/CertificateScope.v)
refines reuse for this score-only, exact-natural fragment. It compares the
authoritative request's domain, query, parameter token, gap, stiffness,
snapshot, revision, arithmetic, observation, and label context before a
cached chain can be replayed. A cache key is only a lookup hint; collisions
cannot replace those comparisons. The accepted chain proves a score relation
for every variable environment, so that particular relation can be reused
across cutoffs while retaining the same stable request. A pruning decision
must compare the certified lower expression against the **new** inclusive
cutoff. A cached strict-prune decision can be carried to a lower cutoff, but
an increased cutoff requires a fresh comparison. The same-scope gate requires
exact cutoff equality where an operation may depend on the cutoff. Neither
gate certifies whole range/kNN output. The explicit gap and stiffness values
are exact-natural placeholders; floating-point representation and the binding
of these request fields to actual Rust inputs remain separate proof duties.
The cached-decision theorem uses the same exact score environment for both
cutoffs; a source instance must establish that stability before using the
rule. The proof checker replays old evidence here, so no runtime speedup is
claimed by this formalization.

The finite [certificate rejection model](../verification/CBC_CERTIFICATE_REPLAY_REPORT.md)
enumerates 19 acceptance and rejection scenarios. It checks the reason for
each malformed certificate, including a truncated two-step chain and a lower
result disguised as an exact claim. Disabling scope equality makes its
stale-revision invariant fail. This bounded model tests the acceptance
interface; it does not prove the expression rewrite rules, witness validity,
or Rust correspondence.

For the first concrete source connection, use a direct refinement of a named
standard integer Levenshtein transition path: the dispatch in
`src/transducer/variants/standard.rs` delegates to
`transition_standard_into` in `src/transducer/transition.rs`. The proof must relate the source
transition fields, checked integer arithmetic, final observation, and any
dictionary traversal it claims to the corresponding formal state. A small
verified transition is an intermediate instance; a whole-operation CBC claim
also requires the session and finalization proofs. This route minimizes
translation machinery while exposing the source-language and toolchain trust
boundary. Source hashes detect drift after the proof but cannot establish the
relation themselves.

The [first source-correspondence route](../verification/CBC_STANDARD_SOURCE_ROUTE.md)
selects the actual positional Standard path at cutoff 4, where packed
dispatch is ineligible, and gives a proof graph from the public units-native
query through transition, cache, finality, and original enumeration. Each
unproved source edge remains open; the route document is an architecture
decision rather than an executable CBC certificate.

### 3.2b Typed operation contracts

[CBCContractInterpretation.v](../verification/temporal_automata/theories/CBCContractInterpretation.v)
defines profile-indexed request and payload types for online scores,
completed range, completed kNN, canonical witnesses, resumable operations,
and resource observations. Its `measure_specification` fixes a validated
domain, quotient identity, parameters, numerical authority, reference
measure, and ordered valid-cost carrier. Its `operation_contract` fixes a
validated query, snapshot and distinct original identities, eligibility,
separate range and kNN tie orders, inclusive range cutoff, requested $`k`$,
budgets, and a nonempty set of selected profiles. The online request is a
target input; the witness request is an original occurrence; search and
resource requests use the fixed query and captured snapshot. Invalid-query
validation and error precedence occur before this valid-query contract is
constructed and remain source-instance obligations.

`ideal_range_law` covers exactly the eligible originals within the inclusive
cutoff in declared range order. `ideal_knn_law` covers all eligible originals
in cost-and-tie order independently of that cutoff. A completed kNN result
must equal the first $`k`$ of this independent order even when a range query
would fail. Online observations must correspond one-for-one with a declared,
nonempty attempted-prefix sequence; each prefix event must match its domain
check or exact/above-cutoff reference cost. Only the final attempt may fail or
stop incomplete, and either outcome leaves the preceding prefix committed.
An incomplete range or kNN
partial may contain only distinct, correctly ordered rows from its ideal
universe. It cannot claim exhaustion or final top-$`k`$ status.

A completed witness has a feasible path at the reference cost, no cheaper
feasible path, and no earlier equal-cost witness under the declared tie rule;
`Completed None` requires the original to be ineligible for the cutoff. A
completed session has no private or remaining work and no continuation. A
visible incomplete session partitions the captured originals, keeps published
and excluded originals sound with respect to an explicit range/kNN session
goal, and either
denotes its private/remaining work by a continuation or satisfies an explicit
terminal-incomplete predicate. Resource observations carry primitive events
and reported counters; the contract requires an event-accounting relation,
budget safety, and monotone cumulative charges and peaks. The concrete
meaning, units, and disjointness of charged and reserved work remain instance
premises.

`lawful_reference` requires the ideal-list laws and every selected
`observation_law`. `exact_realization` requires that lawful reference plus
equality for each selected profile. A generic theorem transfers the relevant
observation law to any exact realization. `lower_simulation`,
`metric_geometry`, and `resource_refinement` remain distinct propositions.
The file checks an inhabited one-original example where zero is a sound lower
bound for exact cost one but cannot be an exact realization; it also rejects a
foreign completed kNN row, omitted online prefix, unsound completed exclusion,
and unjustified partial row. The numerical descriptor, tie keys, prefix
sequence, request-specific failure and incomplete entitlement, witness
relation, continuation
denotation, and event accounting must be proved for each concrete operation.
The full checker, session refinement, and Rust correspondence are later gates.

### 3.2c Scoped finite rewrite replay

[CertificateChecking.v](../verification/temporal_automata/theories/CertificateChecking.v)
extends the exact-natural expression fragment with a typed, finite rewrite
certificate. Each step records a named rule, source and target expressions,
forward or backward orientation, the exact required-premise list, arithmetic
profile, witness effect, and a scope containing the query, parameters, domain,
cutoff, snapshot, dictionary revision, and label context. The checker compares
every step scope with the requested scope, then recomputes the rule result and
its premise list. It cannot accept an unknown rule identifier or a lower rule
in the reverse direction. The only admitted witness effect is the score-only
effect; a later witness-capable rule needs its own soundness theorem.

`replay_step_identifies_exact_rule_and_premises` establishes that an accepted
step names a supported rule, supplies precisely its required premises, and
matches that rule's computed result. `replay_steps_iff_finite_replay` relates
the executable checker to a finite inductive replay relation.
`accepted_scoped_certificate_sound` proves that accepted exact chains preserve
every natural-valued score, while accepted lower chains remain lower bounds.
The generic rejection theorems cover unknown rules, wrong premise lists,
reversed lower rules, and rounded-arithmetic steps; concrete controls also
reject revision changes, unsupported witness effects, and lower certificates
offered as exact equivalence. The acceptance corollaries state exact score
equality and lower-bound admissibility separately for every variable
environment. An omitted final rewrite and a fabricated rule tag fail even
when the advertised source and target are related by a real rule. Premise
names are audit labels: the checker
derives their actual validity from the finite rule definitions and their
kernel-checked soundness lemmas, rather than trusting a supplied name.
The scope fields are opaque tokens. Exact token comparison does not establish
that a token names the executable's actual query, parameters, dictionary
revision, or label interpretation; that binding is an instance obligation.

This checks the declared expression fragment only. It does not replay all
nodes of the ORC realization IR, prove floating-point rewrites, establish
witness preservation, or connect the certificate to Rust execution. Those
require additional checked rules and instance/correspondence proofs.

### 3.3 CBC-2: progress is a separate obligation

A machine that silently switches dense to sparse and back forever can satisfy
the finite-trace condition. Correct finite outputs do not imply that an output
is eventually produced.

Assign a natural rank $`r(i)`$ to consecutive internal steps that match no
reference work, and require strict decrease on each such step. If there are
$`n`$ internal steps from $`i`$ to $`i'`$, then

```math
n+r(i')\le r(i).
```

**Proof.** The zero-step case is equality. Each further step reduces the rank
by at least one; adding that inequality to the induction hypothesis proves
the result. The Rocq file checks this argument.

If every reference execution for a finite request has finitely many steps,
every other concrete step matches positive reference progress, no active
state is stuck, and the internal rank condition holds between progress
steps, the concrete request terminates or reaches an allowed suspension.
Otherwise an infinite execution would have either infinitely many reference
steps or an infinite internal segment, contradicting a premise. Client
resumption and external allocation availability are environment conditions;
no theorem forces a client to resume.

For adaptive layout choice, hysteresis can reduce switching but does not
replace this rank or an equivalent termination proof.

## 4. A session invariant that covers the whole search

One **search session** is a validated range or kNN operation from admission
through a complete or incomplete outcome. It may pause at a page boundary and
resume from retained work. Its runtime state stores the captured search
snapshot and pending work; proof-only ghost state records facts needed to show
that no original was lost, duplicated, or excluded without evidence. A pause
retains the same logical operation and snapshot.

Represent a session by $`(\Gamma,P,V,X,E,A,C,H,L,K)`$:

- $`P`$: pending product occurrences and their remaining candidate regions;
- $`V`$: ghost history of authoritative verifications, including candidates
  subsequently displaced from a best-$`k`$ heap;
- $`X`$: discarded candidates with sound exclusion evidence;
- $`E`$: retained exact range results or the selected best-$`k`$ heap;
- $`A,C`$: state arena and complete transition/summary caches;
- $`H`$: private work in progress, including phase and resumable cursors;
- $`L`$: the resource ledger; and
- $`K`$: ownership and reconstruction context for dictionary paths.

A **product occurrence** includes enough path context to identify its concrete
originals. Different occurrences may share a dictionary node or automaton
state. Sharing a representation does not identify their result identities.

The ownership discipline depends on the publication phase. An uncommitted
node split may borrow its parent region while that region remains in $`P`$;
publishing the complete split transfers its originals atomically to child
regions, terminal work, or sound exclusions. Once a candidate cursor advances
past an original, that original instead belongs to exclusive private work
$`H_o`$ until verification, exclusion, or result publication. These two kinds
of private work cannot be counted twice. Suspension may retain $`H_o`$;
terminal failure or cancellation may discard the unfinished session only
under an `Incomplete` outcome. A `Complete` outcome must exhaust all owned
work.

The bounded range implementation demonstrates why exclusive private ownership
is necessary: it increments a frame or scan cursor before applying the exact
candidate decision, and an accepted result may wait in `pending_match` across
a result-page pause. The cursor no longer covers that original while the
continuation still owns it. The [range continuation](../../src/time_series/elastic/walker.rs)
also keeps `results` and `pending_match` separate; a cancelled partial result
is a subset and need not include the latter.

| Bounded range source phase | Ownership effect | Remaining correspondence proof |
|---|---|---|
| Read bucket/slot and check the page work limit | Original stays in the pending cursor region | Prove no cursor increment on the budget-pause branch |
| Increment `next_candidate` or scan `slot` before lower-bound/exact scoring | Transfer one original to ephemeral private scoring | Prove each bucket member is enumerated once, including collisions |
| Reject by an admissible lower bound or exact `AboveCutoff`/`NoFiniteAlignment` | Transfer private original to sound exclusion | Bind the Rust bound and exact decision to the requested numeric contract |
| Store a verified result in `pending_match` on result-page exhaustion | Retain exclusive private ownership across resumption | Show the next `advance` publishes this match before reading a new candidate |
| Append a verified result to `results` | Transfer private ownership to retained results | Prove identity, exact cost, and capacity/charge observations |
| Finish or cancel | Sort completed results or return an exact partial subset | Prove the score/encounter tie order and keep `Incomplete` distinct from exhaustive completion |

The [finite cursor model report](../verification/CBC_RANGE_CURSOR_MODEL_REPORT.md)
checks this ownership lifecycle with separate original identities at a shared
node. Its omitted-private, skipped-collision, and early-pending-finish mutants
each violate the intended property. Its conditional fairness result addresses
only that bounded lifecycle.

For the simplest ownership discipline, maintain a disjoint cover

```math
U=E_o\uplus X\uplus H_o\uplus\biguplus_{Q\in P}U_Q.
```

Here $`E_o`$ is the set of original identities retained in $`E`$, $`X`$
contains soundly excluded originals (including verified candidates displaced
from a full best-$`k`$ heap), $`H_o`$ contains originals exclusively owned by
private work, and $`U_Q`$ is the unresolved region owned by pending occurrence
$`Q`$. The ghost verification history $`V`$ may overlap $`E_o`$ and $`X`$;
it is not another ownership class. The implementation can retain only its
compact best-$`k`$ heap and needed continuation state; ghost history is not
charged as live storage.
A backend with overlapping regions must instead prove an ownership or
deduplication invariant with equivalent coverage and multiplicity.
Set-union coverage alone cannot rule out duplicate emission.

[CertifiedRegionPartition.v](../verification/temporal_automata/theories/CertifiedRegionPartition.v)
provides a complete finite checker for an **explicit** parent list and a
terminal-plus-children split package. Acceptance is equivalent to a
permutation of original occurrences, so it preserves coverage and
multiplicity in any surrounding ownership context. The theorem is generic in
the decidable original identity; a checked path-and-bucket-slot instance
keeps two occurrences distinct even when they share a physical node. The checked controls
reject a missing terminal, a skipped collision member at a shared physical
node, a duplicate original, and a foreign original. The checker does not
derive the parent list from a captured dictionary or prove the Rust edge
iterator enumerates the submitted child lists. It materializes and scans
original lists, so it is a proof/reference checker rather than a proposed
hot-path production check; a compact structural certificate needs its own
source and resource proof.

[CertifiedStructuralSplit.v](../verification/temporal_automata/theories/CertifiedStructuralSplit.v)
constructs the explicit split from a total classifier over a supplied parent
list. It proves a permutation of all supplied originals, unique child labels,
and checker acceptance. Given a duplicate-free parent list, the result is
also duplicate-free. The constructor's classifier and parent enumeration are
still abstract inputs; the proof does not identify them with a captured Rust
dictionary snapshot. Its list buckets are a proof/reference representation,
not a proposal to scan complete subtrees during production search.

[CertifiedSearchSession.v](../verification/temporal_automata/theories/CertifiedSearchSession.v)
checks the exclusive-private fragment in which each **original occurrence** moves from
pending to private ownership, then atomically to verified or excluded. It
proves a disjoint list permutation, exclusion preservation, no duplicates,
and equality of winner **membership** after filtering the verified list when
both pending and private lists are empty. The model does not prove ordered
ranking of that set. A cursor-shaped specialization adds `Scoring` and
`AwaitingResult` phases, proves revision and ownership preservation across
pick, reject, accept, publish, and logical pause steps, and proves exact range
membership after cursor exhaustion when exact classification is supplied.
Its result list is reverse encounter history for membership proofs, not the
public range order. It does not prove the borrowed-parent split phase,
the exact Rust `pending_match`/cursor correspondence, the correctness of a concrete exclusion bound, ranked heap ordering,
snapshot/resource observations, or Rust correspondence. Those are separate
whole-session obligations.

The same Rocq file now defines `CompleteState.session_runtime` and a separate
`session_ghost`. The runtime type holds the captured contract, query, snapshot
and revision; compact pending occurrences with cursors and a ghost region
interpretation from the captured snapshot;
private split, scoring, publication, and cache-building phases; range results
or a bounded kNN store; emission position; arena and complete cache entries;
resource ledger; reconstruction context; and active, suspended, completed, or
failed status. The ghost type holds verification history, prior emitted
results, and exclusions with evidence. `session_abstraction` projects these
into the original-occurrence partition and requires exact coverage, live
borrowed parents, scoped cache entries, authoritative retained/emitted scores,
sound exclusions, arena and reconstruction validity, cache-entry semantic
validity, basic ledger inequalities, and exhaustion on completion.
The region interpretation is a proof parameter, not a list retained by each
frame. The storage measure takes only the runtime type. A checked example has a
best-one store with one retained result and two ghost verifications, so the
model does not require the concrete heap to retain every verified rank.

| Formal component | Source fields that motivate it | Still required for source refinement |
|---|---|---|
| Identity and pending occurrences | `RangeContinuation::{index, query, tau, mode}` and `BoundedRangeFrame::{state, final_bucket, next_candidate, edges}`; timestamped TWED's captured root and `ProductFrame` | Map each frame and scan cursor to its exact remaining originals in one immutable revision |
| Private phase and selected results | `RangeContinuation::{pending_match, results}` and the exact kNN queue/result structures | Prove cursor transfer, borrowed-parent publication, rank admission, displacement, and result order |
| Arena, cache, and reconstruction | `RangeSessionMode::Trie::{traversal, states, cache}` and the timestamped TWED state arena and transitions | Prove complete cache publication, scope, state IDs, path reconstruction, and failure atomicity |
| Ledger, emission, and status | `ResourceLedger`, `OperationOutcome`, result-page state, `terminal`, and `Done` | Connect abstract executed/reserved/charged work and live/peak bytes to actual allocation and continuation observations |
| Ghost history and exclusions | Proof-only verification/emission history and sound-exclusion witnesses | Prove each source transition preserves the abstraction; ghost history is erased from runtime storage accounting |

The type and example establish the state vocabulary and an abstraction
boundary. They do not yet prove that a Rust continuation satisfies that
relation; the following operational tasks must discharge each source edge.

[CertifiedSessionOwnership.v](../verification/temporal_automata/theories/CertifiedSessionOwnership.v)
proves the initial whole-session abstraction for an empty range store or an
empty best-$`k`$ store, provided the root's snapshot-relative region is exactly
the duplicate-free universe and its arena and reconstruction interpretations
are valid. Two checked controls keep separate originals at a shared node and
at different slots of one collision bucket; coalescing by node loses an
original. A third control takes an explicit [ownership-quotient step](../verification/temporal_automata/theories/CertifiedSearchSession.v)
from the public initial state into the complete state projection with one
exclusive private candidate, then shows that omitting private ownership loses
that original. This is a finite model execution, not a Rust cursor step: the
Rust root enumeration, bucket cursor, and continuation fields still need
source correspondence proofs.

[CertifiedSessionSplit.v](../verification/temporal_automata/theories/CertifiedSessionSplit.v)
connects the finite split checker to a complete runtime's borrowed-parent
phase. Given a live parent at the specified pending position, the very child
handles staged in that phase, and an accepted snapshot-relative terminal and
child package, one atomic publication replaces the parent with those handles
and clears the private phase. The proof preserves the exact multiplicity of
original ownership, the exclusion record, the captured session identity, and
the full session abstraction. Staged children already satisfy the old
abstraction's arena and reconstruction conditions; newly built terminal
handles must satisfy those conditions explicitly. An accepted region package
alone cannot establish arena validity, as a checked counterexample shows.
A finite control publishes two distinct paths through one shared physical
node; three rejected packages omit a terminal, omit a collision member, or
duplicate an original. The proof assumes the region interpretation and
terminal enumeration are accurate for the captured snapshot. Matching a Rust
split transition and checking the new terminal handles remain source
refinement obligations below.

[CertifiedSessionVerification.v](../verification/temporal_automata/theories/CertifiedSessionVerification.v)
models one candidate verification after the candidate takes exclusive private
ownership. The abstract verifier has distinct `VerifiedWithin`,
`VerifiedBeyond`, `VerifiedNoAlignment`, `NumericFailure`, and
`ResourceFailure` tags. Its soundness premise binds a successful score to the
captured query and original payload, applies the supplied cutoff relation, binds the
tie key to the snapshot, and validates the witness. Only that successful tag
adds a rank to ghost history and an `AwaitingPublication` heap input. A
rejection transfers ownership to a sound exclusion; either failure retains
private ownership and adds no rank. The transition preserves the complete
session abstraction and a separate ghost provenance list. Negative theorems
reject a successful rank with no finite reference score and an
`AboveCutoff` tag for a within-cutoff score. A finite-or-infinity control
instantiates an inclusive natural cutoff and accepts genuine success,
above-cutoff, and resource-failure tags while rejecting any attempt to turn
the latter two into an exact infinity rank. The generic theorem neither
requires an inclusive cutoff relation nor fixes a finite-score policy; each
source instance must establish both. The failure status codes are
abstract markers, not Rust error discriminants. The proof assumes verifier
soundness, snapshot payload and tie interpretation, and a fixed cutoff. It
does not prove the executed Rust verifier's result tags, rounded arithmetic,
heap insertion, or failure continuation. `run_verifier` consumes an existing
decision and copies the ledger, so verification work and allocation charges
need source-specific proofs. Witness and tie provenance are ghost-only; their
runtime retention and reconstruction are also separate obligations.

[CertifiedSessionResume.v](../verification/temporal_automata/theories/CertifiedSessionResume.v)
models suspension as a status change on the entire retained runtime and ghost
configuration. It proves that every active private phase can pause and resume
with the same pending cursors, snapshot identity, selected results, arena,
cache, reconstruction, emission cursor, ownership, and cumulative work ledger.
It rules out a cursor reset during a pause and a ledger recharge during a
resume. A counted paged execution erases to a productive logical execution
with the same number of productive steps. Two completed executions have the
same retained and emitted results, and the same productive-step count, when
the productive relation is deterministic and both erasures are terminal. An
inhabited already-scored control pauses before publication, then publishes
once under arbitrary legal paging; it rejects cursor reset, ledger recharge,
and a return to the scoring phase. This is
a conditional result: the proof does not establish that a client supplies
sufficient future pages, that an allocation succeeds, or that each Rust
`resume` path implements the modeled step and ledger behavior. In particular,
the source-specific `pending_match` publication, result-slot preflight, and
private split cursor still need correspondence proofs.

[CertifiedSessionFailure.v](../verification/temporal_automata/theories/CertifiedSessionFailure.v)
separates request rejection from admitted execution. For
`TimestampedTwedIndex::search_range_bounded`, its executable validation
model follows the public source order: unit, origin bit pattern, query
length, then cutoff. A prepared action retains the complete predecessor,
records distinct executed-event identities, and separately records work
charged before execution. Every modeled allocation, budget, arithmetic,
numeric, or stored-data failure returns `Incomplete` with the prior semantic
contents and the accumulated ledger. A retained continuation carries that
same ledger. Checked controls exercise failure before charge and after a
charge and execution; they reject a false `Complete`, a rolled-back charge,
and publication of a result from a failed action.

| Source stage | Precedence and observable result | Accounting consequence |
|---|---|---|
| Public range validation | Unit mismatch, then origin-bit mismatch, then query-length limit, then invalid cutoff; first failing check returns an API validation error | No continuation or execution work starts; length validation uses an empty temporary ledger |
| Continuation start | Product-state limit, checked width/scratch arithmetic and peak check, root recurrence, state intern, snapshot traversal, root-node charge, then fallible frame and DP-buffer reservations, then a final state-peak observation | A later failure returns `Incomplete` with the committed ledger reached at that stage; the root-node charge is not undone by a later allocation failure, and a rejected peak is not recorded |
| Active candidate | Candidate/page checks and `charge_many` precede exact scoring; result reservation and result charge precede `results.push`; a fallible state-peak observation follows that publication | Scoring failure retains committed charges. Peak failure after a result push returns terminal `Incomplete`; the already published result may appear in the exact partial value, without a continuation or a `Complete` claim |
| Active transition | Edge charges are first applied to a copied ledger and installed only after edge extraction; child-node charge is copied and installed only after successful frame reservation/open; a fallible peak observation follows child-frame push | A rejected provisional charge does not enter the session ledger. A post-push peak failure is terminal `Incomplete`; no whole-iteration rollback is claimed |
| Successful exhaustion | Only a successful exhaustive traversal enters the `Complete` branch | Final usage remains attached to the completed result |

The table records branch order in the reviewed source, not a proof that every
Rust edge implements the Rocq transaction. The Rocq action boundary is one
prepublication attempt: later peak checks are separate actions after a
successful publication, so the model does not claim whole-loop rollback.
The model's `Incomplete` prefix
is a logical predecessor view; the public Rust partial value may contain only
previously committed result rows. Its event identities and separate executed
versus charged counters need an instance mapping to `ResourceLedger` and the
actual primitive operations. `ResourceLedger::observe_peak` leaves a rejected
amount out of reported usage, so that counter is not an exact physical peak
measurement. Physical allocation capacities, cleanup, and fallible result
finalization also require source-specific refinement.

[CertifiedSessionSnapshot.v](../verification/temporal_automata/theories/CertifiedSessionSnapshot.v)
models an immutable captured image and a catalog of distinct revisions. An
original occurrence is addressed by its dictionary path and bucket slot; the
physical node identifier may be shared by different paths. Every accepted
bucket, summary, cache, or payload observation must agree with the row at
that path and slot in the captured image, including the original identity and
revision. Finite-trace preservation proves that a session can keep reading
its captured revision after a newer image is published. Concrete controls
reject a new bucket combined with old summary/cache observations, and reject
an attempt to relabel new data with the old revision. The pending-key update
checks referential validity only; the ownership and split proofs must also
establish coverage and multiplicity. This is a ghost/reference image, not a
per-frame copy or production lookup requirement. The source refinement must
prove that each Rust root, bucket, summary, cache, and payload read denotes
this same captured image. The timestamped TWED continuation's immutable index
borrow and captured root provide the intended route, but the proof does not
yet connect those fields to the Rocq image or establish cross-revision storage
semantics for other automata.

| Action | Preconditions | Invariant-preserving effect |
|---|---|---|
| Split a region | Complete child and terminal enumeration on $`\sigma`$ | Partition unresolved originals among children and terminal verification work |
| Advance a candidate cursor | Cursor points to an unconsumed original | Move that original from $`P`$ to $`H_o`$ before scoring |
| Verify an original | Original payload and exact verifier agree with $`\Gamma`$ | Add its rank to ghost $`V`$; retain ownership in $`H_o`$ until it enters $`E_o`$ or soundly excluded $`X`$ |
| Pause on a result-page limit | Exact candidate already verified but no result slot available | Keep it in `pending_match` and $`H_o`$; do not advance the cursor again |
| Prune a region | A bound excludes every owned original for the current threshold | Move ownership to $`X`$ with that evidence |
| Tighten kNN threshold | Retain the best $`k`$ distinct verified ranks | Earlier discarded regions stay excluded as the kth rank decreases |
| Resolve a summary | Every relevant child and live terminal is covered | Publish a complete scope-bound certificate |
| Suspend | All committed fields satisfy the invariant | Retain private phase, remaining ownership, ledger, and snapshot |
| Complete | All unresolved work is exhausted or soundly excluded | Materialize the contract's exact ordered result |

A verified kNN candidate is provisional until no unresolved rank can precede
it. Streaming it as an irrevocable ranked result requires that additional
certificate. Range operations with dictionary order need their own emission
invariant.

### Best-$`k`$ selection and heap boundary

The [checked best-$`k`$ model](../verification/temporal_automata/theories/CertifiedKnnHeap.v)
uses one distinct original identity and one distinct tie key per verified
candidate. It orders exact natural costs lexicographically by cost and tie
key. Its certificate partitions every verified candidate into a sorted
retained sequence and a rejected sequence, bounds retained length by $`k`$,
forbids rejection while underfull, and requires every retained rank to
precede every rejected rank. The canonical answer is the first $`k`$
entries of the verified candidates sorted by this same ordering; the Rocq
theorem `certified_best_k_is_canonical` proves equality of the entire retained
entry sequence, including original identities. Thus the certificate is an
exact selection specification, rather than merely a plausible heap size.

Admission fills an underfull selection. Once full, a strictly better rank
replaces its worst entry; an equal or worse rank is rejected. The checked
`best_k_step_preserves_certificate` theorem covers all four cases, including
$`k=0`$. The `kth_rank_iff_full_best_k` theorem permits a pruning cutoff
exactly when $`k>0`$ and at least $`k`$ candidates have been verified.
`full_kth_rank_nonincreasing` proves that later full-heap cutoffs cannot
increase. A concrete max-heap representation may permute selected entries;
when its root dominates the rest, `heap_root_has_kth_rank` proves the root's
cost and tie key equal the canonical kth rank. Equal-cost reversed-tie and
premature-underfull-cutoff examples violate the certificate.

These are exact-natural reference proofs. The source-specific obligations
remain: bind each Rust candidate identity and tie rule to the model; prove
that the binary64 comparison and TOP policy refine the declared rank; show
that heap insertion, replacement, and root maintenance preserve the abstract
step; and connect threshold use to the session's exclusion rule. The reference
sort is a proof oracle and adds no sort or scan to the production hot path.

### Prune certificates and persistent exclusions

The [checked pruning model](../verification/temporal_automata/theories/CertifiedKnnPruning.v)
distinguishes `Unknown`, `Empty`, and `Known` rank-floor summaries.
`Unknown` never authorizes pruning. `Empty` requires proof that the region
contains no candidates and may be discarded before the heap fills. A `Known`
floor must be at or below every candidate's exact rank. It authorizes pruning
only when a full heap supplies a kth rank no greater than that floor. Thus
equal cost alone is insufficient: the tie coordinate decides the equality
slice. The negative control has a verified rank $`(5,1)`$ and a region rank
$`(5,0)`$; a cost-only equality cut would lose the better region candidate.

The certificate also carries an exact scope identity. The generic scope
must be instantiated with the full query, parameters, snapshot/revision,
region occurrence, arithmetic profile, and observation contract. The decision
checks exact scope equality; its soundness and action theorems additionally
require a semantic bound proof at that scope. Matching a hash or cache key
alone is insufficient. A changed-revision control rejects the old bound even
when the underlying unscoped decision would return true.

`scoped_prune_action_preserves_ownership_and_exclusions` moves the complete
region from unresolved to excluded ownership and establishes that no member
can beat the active kth rank. `pruned_region_cannot_change_best_k` proves a
stronger reference property: even if all hidden exact ranks in the region
were added to the oracle universe, the selected best-$`k`$ sequence would
be unchanged, assuming distinct original identities and ties. This is a
counterfactual correctness proof; runtime pruning does not evaluate those
candidates. The `pruned_region_survives_best_k_step` and
`pruned_summary_survives_best_k_step` theorems show that exclusions and
reusable summary decisions persist after a full-heap insertion or replacement
lowers the worst rank. An explicit threshold-increase control fails that
persistence property, so a wider cutoff requires a new proof.

The remaining instance obligations are to construct and validate region
bounds from the actual automaton, prove that their scope matches the active
operation, account for machine arithmetic and TOP, and connect Rust prune
transitions and ownership storage to this model. The ghost oracle and its
counterfactual sort add no production scan or allocation.

The publication boundary covers arena IDs, cache entries, pending cursors,
and witnesses together. Private work may accumulate charges while the public
predecessor stays unchanged. An allocation failure after charged work must
retain or explicitly account for that work; rolling back semantic state does
not erase resource use. A cached resource failure cannot replace a semantic
dead transition.

The [checked cache and arena model](../verification/temporal_automata/theories/CertifiedSessionCache.v)
refines this boundary for one logical transition. A cache entry contains only
a completed `Dead` or `Live(id)` answer. Cache lookup compares the complete
operation scope, exact source-state identifier, and label before returning
that answer. An absent entry is a miss; it is distinct from a cached `Dead`.
`ResourceFailure` is a separate action result and cannot be stored as a dead
entry. `cache_hit_is_semantically_complete` and
`completed_action_refines_semantic_step` prove equality with the abstract
successor, conditional on the complete-entry invariant and source state.
Eviction retains the invariant and permits recomputation; it does not assert
equal work, allocation counts, or latency. A checked example has the same
semantic answer with different resource reports.

The query-local interner indexes immutable canonical residuals by a
fingerprint hint, then compares the full residual before reusing an ID.
The invariant requires every arena state to have a fingerprint index entry
and every entry to name the correct state. It also requires unique canonical
residuals. A hit denotes exact equality even when two states have the same
fingerprint; a miss proves the residual is new. Fresh publication appends the
residual before its ID becomes visible, preserving old ID referents and all
cache and publication evidence. `publication_action` accepts only a successor
whose arena referent, source transition, cursor, and witness are valid.
The checked premature-publication mutant violates the invariant when the ID
has no arena referent; a changed-scope cache lookup misses even if its source
ID and label match.

The model reflects the complete-hit and append-only ordering in
[`TimestampedTwedRangeContinuation::transition`](../../src/time_series/timestamped_twed_index.rs)
and `ProductStateArena::intern_at_fingerprint`. A concrete refinement still
must prove the binary64/canonical-bit residual equality, actual scope fields,
source-token decode, allocation and reservation effects, ledger charges,
and Rust publication sequence. The logical-state failure theorem does not
claim unchanged physical capacity after a failed reserve.

Coverage is also a **cut condition** on the recurrence dependency graph. A
vertex denotes a complete continuation context and its authoritative prefix
cost. The graph includes ordinary recurrence cells in retained generations
and pending multi-label or macro operations; an edge represents one lawful
dependency step. Let $`L`$ be the represented live vertices and $`D`$ the
processed vertices, whose accepting members have been verified and recorded.
For every accepting path
$`p`$ from the root to endpoint $`e`$, the cut invariant is:

```math
e\in D\quad\lor\quad p\cap L\ne\varnothing.
```

[CertifiedDependencyCut.v](../verification/temporal_automata/theories/CertifiedDependencyCut.v)
proves that this invariant holds initially and survives any finite sequence
of expansions. An expansion of $`x\in L`$ removes $`x`$, inserts an enumerated
successor set containing **every semantic edge** out of $`x`$, and adds $`x`$
to $`D`$. If $`x`$ is accepting, that addition requires its exact endpoint
obligation to have been proved. The successor-completeness premise is local
to each expansion, so a kernel may enumerate different operation families at
different vertices.
If each semantic edge satisfies $`c(u)\le c(v)`$, any unresolved accepting
path meets a live vertex whose cost does not exceed its endpoint cost. Thus a
nonempty live cut has floor $`\min_{x\in L}c(x)`$ no greater than every
unresolved completion, and strict cost pruning is sound when every live
cost exceeds the inclusive cutoff. The theorem does not supply a
lexicographic kNN tie floor; that needs the separate rank certificate.

The checked negative control has two paths: a current-row vertex of cost six
leading to an accepting cost seven, and a pending macro vertex of cost one
that jumps across that row to an accepting cost four. At inclusive cutoff
five, the current row alone appears prunable but would lose the second
answer. The complete cut retains both vertices. An operation instance must
prove its concrete recurrence edges are all enumerated, include bypasses and
retained generations in its live-state interpretation, justify exact endpoint
resolution, and establish cost monotonicity for its numerical authority.
The abstract vertex must retain enough path context to make its prefix cost
well defined; merging different costs requires a separate simulation proof.
No Rust recurrence or floating-point instance is certified by this generic
graph theorem alone. This connects ORC S1's dependency model to product
pruning without assuming a current-row minimum is always sufficient.

| Source operation | Required cut refinement |
|---|---|
| One-label recurrence step | Every source cell edge appears in the enumerated successor set; the live projection retains the needed current and prior generations |
| Multi-label or macro step | Pending work or a direct bypass edge remains represented until its destination is live or its accepting endpoint is verified |
| Dense/sparse conversion and cache reuse | The converted or reused state denotes the same live semantic dependencies; no bypass successor disappears |
| Exact endpoint verification | An accepting vertex enters the processed set only with authoritative score, eligibility, and witness evidence for that endpoint |
| Cost-based pruning | The instance proves nondecreasing cost on every semantic edge in the declared arithmetic profile; a stronger completion estimate needs its own bound proof |

## 5. Numerical certificates that authorize machine pruning

Fix the exact authority: a mathematical real measure, an exact rational or
integer measure, or a particular binary64 expression graph. These can have
different observable answers.

Machine pruning requires $`\widehat L_{\mathcal N}\le d_{\mathcal N}`$ under
the public comparator. A real inequality $`L\le d`$ is insufficient on its
own. Cutoff equality uses exact comparison without an epsilon. Overflow,
invalid input, structural impossibility, and above-cutoff results retain the
profile's distinct tags. Reassociation, fused multiply-add, vector reductions,
and flush-to-zero modes need correspondence evidence when they change the
graph. Underflow can erase a mathematically positive cost, so domain
validation does not prove separation for rounded scores. Signed-zero
canonicalization must agree with the declared observations and comparator.

**NC-1 — enclosure-based geometric lower bound.** Suppose an exact metric
$`d`$ has certified enclosures
$`\ell(q,p)\le d(q,p)`$ and $`d(x,p)\le u(x,p)`$. Then

```math
\max(0,\ell(q,p)-u(x,p))\le d(q,x).
```

**Proof.** The triangle inequality gives
$`d(q,p)-d(x,p)\le d(q,x)`$. Replacing the first term by a lower bound and
the subtracted term by an upper bound can only decrease the left side.
Nonnegativity permits its maximum with zero. Downward-rounded subtraction
and maximum retain the inequality if their enclosure semantics are proved.

This licenses real-distance rejection. Comparing against a machine score
also needs its relation to the real score. If, for example,
$`d_{\mathcal N}(q,x)\ge d(q,x)-\epsilon(q,x)`$, subtract that certified
error allowance from the real lower bound with downward rounding.
Error estimates only on pivot distances do not cover rounding in the exact
candidate verifier. A nonnegative machine score also permits clipping the
resulting bound at zero.

Primitive enclosure certificates compose through monotone expression nodes
by induction on the evaluation DAG. Each node needs its own directed-rounding
or error theorem, including valid intermediate ranges. NC-1 is proved here
mathematically; a concrete binary64 enclosure implementation is an additional
artifact.

## 6. Stronger ordered pruning with scoped rank certificates

### 6.1 BF-2: the equality slice is enough

Use the exact rank $`R(x)=(d(q,x),t_\sigma(x))`$ and total lexicographic
order of LOCPA BF-1. A **rank certificate** for region $`Q`$ is a pair
$`(L,M)`$ satisfying

```math
\forall x\in U_Q,\qquad (L,M)\le_{\rm lex}R(x).
```

Given cost admissibility $`L\le d(q,x)`$ for every member, this certificate
is equivalent to the weaker secondary premise

```math
d(q,x)=L\Longrightarrow M\le t_\sigma(x).
```

**Proof.** When $`L<d(q,x)`$, lexicographic order holds regardless of the
tie coordinate. When the costs are equal, it is precisely the displayed
tie inequality. These are all cases under cost admissibility. Rocq checks
this characterization for natural ranks in
[CertifiedConditionedTieFloor.v](../verification/temporal_automata/theories/CertifiedConditionedTieFloor.v),
including an empty equality slice, the full verified kth-rank gate, and
canonical best-k preservation under fresh identities and ties. The abstract
argument uses only a total cost order; the checked instance uses natural
costs.

A global tie floor satisfies this condition but may be weaker. To derive a
better floor without knowing exact scores, take a certified superset
$`S_Q(c)`$ of originals whose exact cost is at most $`c`$. For example,
with per-original lower bounds $`b(x)\le d(q,x)`$, use

```math
S_Q(c)=\{x\in U_Q:b(x)\le c\}.
```

Every exact at-or-below-cutoff original belongs to this set, since
$`b(x)\le d(q,x)\le c`$. A floor over every member of $`S_Q(L)`$ therefore
supplies the equality-slice premise. Enumerating all originals may be
expensive; a structural summary must prove the same coverage to claim the
same floor.

**Theorem BF-2.** Once the heap contains $`k`$ exact distinct results with
worst rank $`W=(L,t_k)`$, a region with cost lower bound $`L`$ and a
certified floor $`M\ge t_k`$ over $`S_Q(L)`$ can be pruned.

**Proof.** The covering-set argument supplies a rank certificate. BF-1's
transitivity argument then excludes any unverified rank below $`W`$.
No metric axiom is used. Rocq proves coverage and this pruning implication.

For example, let the worst verified rank be $`(5,7)`$. A region contains
potential ranks $`(6,1)`$ and $`(5,9)`$, with respective cost lower bounds
six and five. Its global lower pair $`(5,1)`$ cannot prune. Its equality-slice
certificate $`(5,9)`$ can: the earlier tie belongs to a candidate certified
to cost more than five.

If $`S_Q(L)`$ is empty, every remaining original costs strictly more than
$`L`$. This is useful even if no larger numeric lower bound is representable.
An incomplete inspection is not an empty-set certificate.
The certificate algebra therefore needs a tagged strict cost cut
$`d(q,x)>L`$; encoding it as an ordinary pair with an invented next cost or
greatest tie key is unsound for general cost orders.

### 6.2 BF-3: combine whole rank certificates

Two lower rank certificates for the same region combine by lexicographic
maximum. A parent certificate remains valid on a child subset; combine it
with the child's new certificate in the same way. For a finite union of
nonempty child regions, the lexicographic minimum of their certificates
is a lower certificate for the union.

**Proof.** Each candidate rank lies above both lower certificates, hence above
their maximum in a total order. Subsets preserve universal inequalities.
For a union member, choose its owning child: the minimum of certificates
is no greater than that child's certificate, which is no greater than its
rank. All three statements require complete coverage and common scope.

A conditioned tie floor stays attached to its cost. Both $`(5,9)`$ and
$`(6,1)`$ are valid lower certificates for candidate $`(6,1)`$. Their
coordinatewise maximum $`(6,9)`$ is not. The safe lexicographic maximum is
$`(6,1)`$. Rocq proves safe maximum composition and the counterexample.

Independent **global** cost and tie floors can still be combined
coordinatewise from their separate universal bounds, as in BF-1.
The certificate type must distinguish global floors from conditioned floors.

A threshold summary's scope includes the revision, region occurrence, query,
parameters, arithmetic, observation, and threshold. With fixed per-original
bounds, lowering $`c`$ shrinks $`S_Q(c)`$, so an old floor remains sound.
Raising $`c`$ can admit an earlier tie and requires new evidence. Rocq checks
subset monotonicity and the costs-five-and-six counterexample. If the bound
function also changes, prove subset inclusion before reusing the floor.

For scalar cost bounds alone, the maximum of parent and local bounds gives
monotone child priorities whenever child regions are subsets. Conditioned
pairs require the lexicographic rule above.

### 6.3 BF-4: the limit imposed by available information

Let $`\mathcal I`$ denote all checked evidence. Let
$`\mathcal M(\mathcal I)`$ be the complete score assignments consistent with
that evidence, domain, snapshot identities, and verified scores. Require this
set to be nonempty: inconsistent evidence is a certificate-validation failure,
not a vacuous proof of safe stopping. Consider
algorithms that return only exactly verified candidates.

**Theorem BF-4.** Returning the current best $`k`$ verified results is valid
for every assignment in $`\mathcal M(\mathcal I)`$ exactly when no assignment
has an unverified candidate ranking strictly before the worst selected rank.

**Proof.** If no such candidate exists, the selected set precedes every
unverified candidate in every assignment and is already the best verified
set. Conversely, an assignment with an earlier unverified candidate makes
that selected result incorrect: it displaces the worst selected member.
Unique identities and injective tie keys resolve equality. This argument
assumes a full $`k`$-element selection; smaller selections require exhaustion
or certified ineligibility of unresolved candidates.

This defines optimal use of available evidence. It does not compute
$`\mathcal M(\mathcal I)`$ or show that BF-2 decides every safe case.
Completeness of a proposed stopping test requires an **attainable completion**
lemma: whenever the test refuses to stop, a lawful assignment consistent with
all evidence has an unresolved candidate that changes the selected result.
Independent closed intervals with attainable endpoints permit a simple
construction. Metric triangle constraints, shared recurrence structure, and
relational group bounds can make an independently chosen assignment
impossible; those evidence languages need their own construction theorem.
Relational certificates can exclude assignments admitted by independent
bounds. Faster acquisition of such evidence is a separate cost question.

## 7. Quantitative semantics and precise optimality claims

### 7.1 QO-1: amortized work certificates

Separate cumulative work $`W`$, current live bytes $`M`$, and peak bytes
$`P`$. Allocations and releases update $`M`$; peak updates satisfy
$`P'=\max(P,M')`$. Arena capacity, private scratch, cache entries, path
ownership, results, and continuations contribute to byte accounting.
Allocation totals and peak live storage describe different properties.
Process RSS also includes allocator/runtime behavior outside a logical model.

For a nonnegative potential $`\Phi(s)`$, let a step use $`w_i`$ work and
receive $`a_i`$ units of modeled credit. Require

```math
w_i+\Phi(s_{i+1})\le a_i+\Phi(s_i).
```

Every finite execution then satisfies

```math
\sum_i w_i+\Phi(s_n)\le\sum_i a_i+\Phi(s_0).
```

**Proof.** Sum the step inequalities and cancel intermediate potentials, or
induct on execution length. Nonnegativity permits dropping the final
potential to bound total work. Rocq checks the inductive argument.

Credits must come from an independent bound, such as inspected edges or input
symbols; choosing observed work itself as credit proves no useful improvement.
Conversion, failed probes, normalization, collision checks, and summary
construction must appear in step costs. A potential cannot hide uncharged work.

The potential method connects local rules to sequence-level bounds; its use
with operational cost semantics is developed in
[Hoffmann and Hofmann's resource analysis](https://www.cs.cmu.edu/~janh/assets/pdf/HoffmannH102.pdf).
Here it supplies a contract for cursor advances, cache amortization, adaptive
layout conversions, and witness replay. Each application needs a concrete
potential and correspondence theorem.

### 7.2 QO-2: optimum within a certified portfolio

For workload $`x`$ and contract $`\Gamma`$, let $`\mathcal A`$ be a finite
nonempty portfolio of certified feasible realizations. Let $`J(a,x)`$ be
an exactly evaluated declared objective with a deterministic tie rule.
Selecting its minimum gives

```math
a^*\in\mathcal A,\qquad
\forall a\in\mathcal A,\quad J(a^*,x)\le J(a,x).
```

**Proof.** Start from one member and compare every remaining member with the
current minimum. Each update retains membership and the minimum invariant.
Induction gives both properties. Membership in the certified feasible set
preserves the semantic and resource certificates. Rocq checks membership,
minimum objective, and preservation of an arbitrary certification predicate
for a natural-valued objective.

This is model optimality within the stated portfolio. Measured wall time has
uncertainty and may change with the workload. Minimizing a proved work bound
does not prove minimum latency.

For several objectives, use a vector such as
$`(\text{work},\text{peak bytes},\text{first-result work})`$.
One artifact dominates another only when it is no worse in every coordinate
and strictly better in at least one, under the same workload and resource
model. Select from the Pareto frontier or declare scalar weights.
Fewer states alone is not a dominance certificate.

Adaptive selection additionally pays conversion costs. Choosing the cheapest
layout for each step may spend more than retaining one layout. A finite
acyclic planning graph can include layout, remaining work, and conversion
edges; backward minimum-cost dynamic programming is optimal on that graph.

**Proof.** A terminal node has its declared terminal cost. At any earlier
node, every complete plan starts with one outgoing edge followed by a plan
from its successor. By topological induction, the stored successor cost is
the least such continuation cost. Taking the minimum of edge plus successor
cost therefore gives exactly the least complete cost from this node.

The graph must include all choices being compared and use an exact cost
model. Graph construction and optimization effort also consume resources.
Online policies with unknown future labels need their own competitive or
distributional analysis; the finite-graph theorem supplies neither.

### 7.3 Optimization opportunities and their proof obligations

| Opportunity | Expected work reduction | Mandatory certificate | Main cost to measure |
|---|---|---|---|
| Conditioned tie summaries | Fewer exact verifications at equality | BF-2 coverage and scoped floor | Summary construction and retained metadata |
| Inherited rank certificates | Fewer stale expansions | BF-3 and child-region inclusion | Priority updates and heap maintenance |
| Dense/sparse adaptation | Fewer inactive cell evaluations | R2a, CBC-2, and QO-1 | Conversion, scratch, normalization, and repeated switching |
| Monotone source cursors | Avoid repeated binary searches | CL-1 request order | Source prefix traversed versus number of requests |
| Combined abstract domains | Stronger pruning through joint information | Concrete-set preservation and lower simulation | Relational-domain maintenance |
| Interning and observation caches | Reuse states and transitions | Exact equality, complete scope, pure-successor certificate | Hashing, collision comparisons, retained memory |
| Witness checkpoints | Reduce stored history | Replay and canonical witness equivalence | Replay work and result materialization |

For combined abstract domains with concretizations $`\gamma_i(a_i)`$, use
their intersection as the represented feasible set. A reduction between
components is sound when it preserves that set. Scalar bounds combine by
maximum; a reduced product may derive stronger bounds from correlations
that maxima alone do not capture. An empty intersection is a pruning
certificate only after proving preservation of every represented original.

Cursor stepping illustrates why an asymptotic improvement is conditional.
CL-1 gives work proportional to requests plus the source portion traversed.
For a few requests near the end of a long random-access frontier, independent
binary search can be cheaper. Request count and source density belong in the
portfolio cost model. The cursor theorem establishes answer equality without
promising a speedup for every transition.

### 7.4 Future production adoption: time and space gate

The theory campaign does not change production automata or run the deferred
optimization experiments. A later proposal to adopt a realization must pass
both its semantic/certificate gates and a **predeclared performance gate**.
This gate compares end-to-end operations, not only transition microbenchmarks.

1. Freeze the baseline and candidate revisions, compiler and build profile,
   CPU/memory limits, allocator, hardware, input snapshots, random seeds, and
   measurement scripts before gathering results. Compare the same validated
   inputs and exact output observations. Record warm and cold cache separately.
2. Cover short and long queries, sparse and dense frontiers, varied cutoff and
   $`k`$ values, equality-heavy ties, collision multiplicity, empty/invalid
   inputs, repeated resumptions, and the actual target workloads. Declare the
   primary workload mix and hard memory ceilings in advance. Avoid a pooled
   average that hides a serious regression in a primary case.
3. Count construction and certificate checking, evidence acquisition,
   conversions, cache hashing/collision comparisons, witness replay,
   finalization, and allocation/release. Report total work, throughput,
   first-result latency, median and tail latency (including p95/p99), and
   allocations. Report logical live and peak bytes, retained capacity,
   scratch, continuations, cache/arena occupancy, and measured peak RSS.
4. Use repeated paired measurements, disclose variance and confidence
   intervals, and specify the nonregression margin for each primary measure
   before observing the candidate. A candidate passes only when semantic
   equivalence is established, hard space ceilings hold, and the predeclared
   time/space noninferiority criteria hold. An unfavorable or uncertain result
   stays unadopted pending more evidence or an explicit user-reviewed
   tradeoff. Compare Pareto coordinates within each workload; state count,
   number of prunes, or a single local speedup cannot substitute for them.

This protocol can support a measured nonregression decision for specified
workloads and hardware. It cannot establish universal minimum wall time or
minimum RSS. A useful negative control is a bound that prunes more candidates
but takes longer after summary construction; another is a compressed state
layout that raises peak RSS through conversion scratch. Both fail an
unqualified improvement claim.

## 8. Study model, evidence, and enforcement

![The observation contract feeds reference semantics and local certificates. A checker validates executable correspondence and resource evidence before admitting realizations to an optimization portfolio.](../diagrams/architectures/certified-metric-execution.svg)

The model retains enough state to explain failures, beyond checking a final
distance.

| Layer | Modeled state and transitions | Required study questions |
|---|---|---|
| Measure | Domain, recurrence boundaries, numeric graph, quotient | Which metric laws and identities hold on the precise domain? |
| Residual | Context registers, generations, closure, normalizer | Which information suffices for every continuation and promised witness? |
| Product | Occurrence identity, region ownership, revision | Can splitting, sharing, pruning, or finalization lose or duplicate an original? |
| Scheduler | Priorities, verified top-k, emitted prefix, pending work | Can an equality tie or stale certificate change the ordered result? |
| Execution | Private phases, publication, pause/resume, progress rank | Can a finite request silently diverge or publish partial state? |
| Resources | Work, reservations, live/peak bytes, conversions | Are charges truthful, limits respected, and predicted savings obtained? |
| Construction | Certificate scope, accepted rewrite chain, executable identity | Can an unproved or stale artifact enter certified execution? |

The [Rocq kernel](../verification/temporal_automata/theories/CertifiedMetricExecution.v)
checks CBC-1 finite-trace composition, CBC-2's internal-rank bound, BF-2
eligible-set coverage and pruning, BF-3 maximum composition, QO-1 telescoping,
and QO-2 finite selection. Its rank arithmetic uses natural costs and tie keys.
NC-1, BF-3's union rule, BF-4, and the acyclic planning argument have
mathematical proofs above; they are not claimed as machine-checked Rust
theorems. The full session invariant and checker-to-executable connection
are normative designs requiring instance evidence.

A production CBC gate binds each theorem to the exact source or generated
artifact, kernel configuration, numeric profile, and observation profile.
It rejects missing evidence at construction or build time. Repository proof
success plus a matching source hash can detect drift, but only extraction,
validated translation, or a refinement proof establishes correspondence.
Existing traits and proof files support individual claims; they do not yet
enforce this complete construction discipline.

The next planned investigations have explicit controls:

1. **Certificate composition:** compare trace projections under each profile;
   reject a score-only certificate offered for a canonical-witness operation.
2. **Progress:** allow adversarial repeated conversion and cache maintenance;
   reject internal cycles without reference progress or a decreasing rank.
3. **Coverage:** exercise shared suffixes, collisions, multi-label operations,
   and terminals; reject duplicate ownership and skipped dependencies.
4. **Conditioned floors:** compare global and conditioned summaries; reject
   incomplete minima, raised-threshold reuse, and coordinatewise pair merging.
5. **Numerics:** include equality cutoffs, signed zero, subnormals, overflow,
   and reassociation; reject real-only bounds presented as machine bounds.
6. **Resources and selection:** account for summary/conversion work and
   private scratch, compare with the declared portfolio optimum, and reject
   models that omit those costs.

These are theory and experiment specifications. Production treatments and
benchmark campaigns remain planned. Formal proof compilation and
counterexample checking provide evidence about the stated models, and do not
supply performance measurements.
