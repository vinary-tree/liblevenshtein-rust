# Performance adoption gate for metric automata

**Status:** prospective protocol for `cbc-08-performance-gate`. No production
optimization or deferred experiment is executed by this document. An accepted
mathematical optimization still needs a source correspondence proof and this
measurement gate before adoption.

The gate applies to a proposed realization of one named metric-automaton
operation, such as scalar scoring, range search, ordered $`k`$-nearest-neighbor
search, or a resumable indexed search. The **baseline** is the current
production path for the same operation and observation contract. The
**candidate** is the proposed replacement. A **primary stratum** is a
predeclared workload class whose regression can independently block adoption.
The decision is made per stratum; a weighted average may inform a deployment
choice but cannot erase a failed primary stratum.

## Preregister the comparison

Before running a candidate, save the following in the associated pgmcp
experiment record and a versioned machine-readable run manifest:

| Field | Required value |
|---|---|
| Artifacts | Baseline and candidate commits, build artifact hashes, exact operation and API profile, target triple, compiler version, feature flags, optimization profile, allocator, and enabled CPU features |
| Platform | CPU and microcode, core pinning, governor, memory and process limits, operating system, thermal policy, clock policy, and competing load policy |
| Inputs | Captured dictionary or episode snapshot hash, query corpus hash, seeds, parameter values, cutoff and $`k`$ distribution, invalid-input cases, and expected multiplicity/tie policy |
| Execution | Warm versus cold cache definition, cache reset or preload procedure, process lifetime, concurrency, ordering of paired runs, and repetition count |
| Decision | Primary strata, hard ceilings, primary time and space measures, noninferiority margins, familywise error budget, confidence method, and maximum sample budget |

A source hash proves only that a measured source revision is identified. It
does not prove correspondence to the formal model. Different operation
profiles, output order, numeric admission rules, or snapshots create a new
comparison and require their own correctness gate.

## Correctness gate before timing

Use the same validated inputs for both artifacts. Check the operation's full
observable contract: exact scores or certified bounds as declared, original
identity and collision multiplicity, tie order, range membership, canonical
witnesses where promised, result and error tags, cutoff inclusivity, and
resumption behavior. Compare against the reference contract as well as the
baseline; shared baseline bugs cannot authorize a candidate. State and run
negative controls for stale certificate scope, equality-bound ties, underfull
heaps, arithmetic rejection, and snapshot changes when relevant.

Only an artifact connected to the checked model through the selected source
route can be called *CBC*. Differential and property tests remain valuable
fault detection, but are not the source correspondence proof. Correctness
failure rejects adoption without a performance decision.

## Workload design

At minimum, partition actual target traffic and adversarial cases along these
axes. Record the source of each distribution and its fixed sampling seed.

| Axis | Cases that must remain distinguishable |
|---|---|
| Input size and shape | Empty, short, medium, and long query; small and large captured snapshot; shallow and deep traversal |
| Frontier | Sparse and dense active state; low and high branch factor; early and late terminal |
| Search constraint | Zero, small, equality-heavy, and loose cutoff; $`k=0`$, underfull, full, and large $`k`$ |
| Originals | Unique terminals, multiple originals in one bucket, shared compressed nodes, repeated equal-cost ties |
| Evidence | Strong, weak, and unknown bounds; cache hit and miss; cold and warmed summary construction |
| Continuation | Single completion, repeated suspensions, quota pressure, and resumed completion |
| Arithmetic | Exact integer boundary, finite binary64 edge values, and rejected invalid values in the applicable domain |

The real workload mix determines which strata are primary. Adversarial
strata remain explicit even if rare; they protect correctness and resource
ceilings. Do not combine warm and cold runs or successful and incomplete
operations into one timing distribution. For a cold run, measure index and
summary construction if the operation's deployment path pays it; for a warm
run, include maintenance and amortize initialization over the declared reuse
horizon. Report that horizon. If preprocessing is shared by several queries,
allocate its cost by the recorded query count and also show total batch cost.

## Time and space observations

Time covers the full public operation: setup, certificate checking if runtime,
bound or evidence acquisition, transitions, heap and queue work, conversion,
cache lookup and collision comparison, arena maintenance, witness replay,
finalization, and cleanup. Report throughput, total latency, time to first
result, median, p95, and p99 where the operation streams or serves queries.
Keep transition-only microbenchmarks as diagnostics.

Record executed primitive work and allocated object counts separately from
elapsed time. For space, record logical live and peak bytes, retained
capacity, scratch and conversion buffers, heap/queue, cache and arena,
continuations, witness checkpoints, allocation count and bytes, and process
peak RSS. A logical state-count reduction is not a space win if scratch,
retention, or allocator behavior raises peak RSS. Measure RSS in an isolated
process when concurrent components would make attribution ambiguous. State
whether snapshot storage is charged to the operation or shared by the
baseline and candidate.

## Paired measurement and decision

Use randomized paired blocks on the same input snapshot and query batches.
Randomize baseline/candidate order within blocks, then repeat enough
independent blocks to meet the preregistered precision or sample budget.
The unit of uncertainty is the independent block, not each query inside a
shared warmed process. Keep raw per-block observations and disclose all
failed runs. For latency quantiles, compute each block's operation-level
quantile on a fixed query batch; for throughput, use completed operations per
fixed interval. Use a seeded block bootstrap or another preregistered
cluster-aware method to form simultaneous one-sided confidence bounds across
all primary comparisons. Divide the familywise error budget (default 0.05)
among those comparisons before data collection; report the resulting bound
and interval method. Stop only at the fixed sample budget or at a
preregistered sequential boundary with valid coverage.

For each primary stratum $`s`$ and lower-is-better time measure $`T`$, let
$`R_{s,T}`$ be the candidate/baseline ratio. For lower-is-better space
measure $`S`$, define $`R_{s,S}`$ similarly. The default margins are zero:

```math
U(R_{s,T})\le 1,\qquad U(R_{s,S})\le 1,
```

where $`U`$ is the simultaneous one-sided upper confidence bound. An
operation-specific strictly positive margin can be used only if its value
and rationale were approved in the experiment record before the run. Hard
byte or RSS ceilings apply in addition to the ratios and are checked against
the maximum observed value and the stated uncertainty rule; a ceiling cannot
be offset by a faster median. A primary stratum with an unfavorable or
inconclusive bound does not pass. State exact logical work and storage
invariants separately; statistical intervals do not replace proofs of a
hard resource contract.

Show each workload's vector of time, peak space, first-result latency, and
other declared objectives. Candidate $`A`$ Pareto-dominates baseline $`B`$
only when it is no worse on every coordinate under the same workload and
strictly better on at least one. If neither dominates, record the tradeoff
and obtain explicit review before adoption. A candidate that fails a primary
nonregression criterion or hard space ceiling is rejected under this gate;
any exception needs a separate, explicit adoption decision.

Two mandatory interpretation controls are a terminal scorer that is locally
faster while end-to-end search becomes slower after summary construction,
and a compact logical frontier that raises peak RSS through conversion
scratch. Both fail an unqualified optimization claim. Fewer automaton states,
fewer exact verifications, or a stronger bound are explanatory counters,
never substitute decision metrics.

This gate tests a named artifact on named workloads and hardware. It cannot
prove universal minimum wall time or memory, and it does not make the
deferred experiments an authorization to modify production automata.
