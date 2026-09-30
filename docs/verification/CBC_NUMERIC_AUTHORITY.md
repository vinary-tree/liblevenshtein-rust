# Numeric authority for metric-automata certificates

This is the source-mapped numeric contract for pgmcp task
`cbc-01-numeric-authority`. It complements the [domain contract](CBC_DOMAIN_CAPABILITIES.md)
and [baseline audit](CBC_BASELINE_EVIDENCE.md). A *score authority* is the
arithmetic and comparison rule that decides a public score or cutoff result.
Every pruning theorem must name that authority. An ideal-real inequality,
an exact integer identity, and a particular rounded binary64 computation are
different propositions even when they use the same formula on paper.

## Profiles and admission

| Profile | Carrier and source | What can be claimed | What cannot be inferred |
|---|---|---|---|
| `N-I` | [UnitCost](../../src/cost/unit.rs) uses `usize`, saturating addition, ordinary integer order, and `usize::MAX` as top. | Exact natural arithmetic below saturation; cutoff equality is inclusive. | A saturated top is not a finite unbounded natural score. A proof using finite naturals must show its reachable values stay below top or use an explicit cutoff quotient. |
| `N-Q` | [CostScale](../../src/cost/scale.rs) parses each configured weight's shortest round-tripping *decimal text* as an exact rational and uses checked scaled integers. | Exact calculations relative to the admitted decimal rational configuration and chosen denominator. | The rational denoted by the source `f64` bit pattern is not automatically the parsed decimal rational. `from_scaled` is a rounded presentation conversion, not an exact binary64 identity. |
| `N-R` | The ideal real measure in [ORC](../theory/ordered-residual-calculus.md#33-scalar-algebras-and-numerical-claims), [LOCPA](../theory/lazy-ordered-cost-product-automata.md#9-metric-qualification-is-a-separate-theorem), and measure-specific proof files. | Metric, recurrence, and lower-bound theorems under their exact domain and real-arithmetic premises. | No direct authority over a Rust `f64` cutoff decision or over reordered/fused arithmetic. |
| `N-F` | A specified binary64 operation graph: [WeightedCost](../../src/cost/weighted.rs) accumulates with `+`, [BottleneckCost](../../src/cost/bottleneck.rs) with `max`, and kernel methods evaluate their own ordered primitives. | Machine claims only when each primitive, intermediate range, result tag, and comparator are covered by a source correspondence or direct machine proof. | Algebraic reassociation, FMA, vector reduction changes, subnormal handling, and real-to-machine bound transfer are not automatic. [WeightedCostFloat.v](core/theories/Conformance/WeightedCostFloat.v) gives a limited Flocq reassociation envelope, explicitly excluding overflow. |
| `N-E` | Directed lower/upper enclosures of an `N-R` value, with specified rounding at each expression node. | A real lower bound or triangle-derived bound when the enclosure and every operation's rounding direction are proved. | A one-step `next_down`/`next_up` around a rounded expression is not itself a theorem about that expression's real value; a machine verifier still needs its own transfer. |
| `N-S` | [SoftDtwConfig](../../src/time_series/kernels/soft_dtw.rs) uses stabilized log-sum-exp and a distinct soft-loss observation. | Complete soft-loss/gradient analysis under its own numeric contract. | Min-cost antichain, ordinary metric, or exact trie-pruning claims. |

The [exact certificate checker](temporal_automata/theories/CertifiedContracts.v)
accepts `ExactNaturals` and `ScoreOnly` rewrites. Its lower-certificate type
establishes only a lower relation, and its rounded-scope rejection examples
prevent an `N-I` rewrite from being relabelled as `N-F`. Scope tokens still
need binding to live query, parameters, cutoff, and snapshot.

## Comparison and result authority

| Operation | Current source rule | Certificate obligation |
|---|---|---|
| Inclusive floating cutoff | `WeightedCost::within` and `BottleneckCost::within` use numeric `<=`; NaN compares false and `-0.0` equals `+0.0` for membership. The [elastic kernel admission](../../src/time_series/elastic/mod.rs#L211) uses `within(ZERO, cutoff)` and an upper-top check, admitting either zero sign. | A bound may prune only after proving it is above the cutoff in the *same* authority; no test epsilon widens admission. |
| Deterministic total ranking | Both floating carriers use `f64::total_cmp` for `compare`; public candidate ranking has an additional tie key where specified. | The rank proof must use the actual score order and prove how signed-zero canonicalization and candidate ties are represented. Numeric cutoff equality and total rank order are deliberately distinct relations. |
| Canonical state identity | [CanonicalFinite/CanonicalCost](../../src/time_series/automaton/cost.rs) admit finite, nonnegative values and merge signed zeros into one bit key; floating carrier `canonical_state_key` likewise rejects NaN and canonicalizes zero. | Equal keys must imply equal future transitions under the selected operation graph, not merely approximately equal scores. Cache scope also includes query, parameters, arithmetic, and snapshot. |
| Exact bounded outcome | [ExactDecision](../../src/time_series/bounded.rs#L357) separates `WithinCutoff { distance }`, `AboveCutoff`, and `NoFiniteAlignment`; [OperationOutcome](../../src/time_series/bounded.rs#L315) separately distinguishes `Complete` and `Incomplete`. | Numeric overflow, structural absence, budget exhaustion, and above-cutoff score cannot be collapsed to one empty result. The full operation profile determines error precedence. |
| Floating top | `WeightedCost` and `BottleneckCost` use positive infinity as absorbing top. Kernel-specific structural predicates distinguish impossible alignments from a reachable computation that overflowed to top. | A proof must state whether top is structural, a saturated machine value, or a valid unbounded cutoff. Real finite-cost proofs do not classify floating overflow. |
| Test envelopes | `CostMonoid::EPSILON` is a rounding/testing envelope. Main MSM transition paths use a zero `COST_EPSILON`. | No nonzero epsilon may silently enter cutoff membership, canonical equality, subsumption, or pruning. A tolerance-based test is supporting evidence, never a pruning certificate. |

The signed-zero boundary is pinned by
[cost-monoid tests](../../tests/cost_monoid_laws.rs) and
[elastic-kernel admission tests](../../tests/elastic_kernel_contract.rs).
The former verifies both floating carriers at zero and adjacent positive
cutoffs; the latter verifies `-0.0` admission and NaN rejection. These tests
exercise the chosen contract but do not establish binary64 admissibility for
every lower bound.

## Bound and verifier applicability

All bounds below are used only through a comparison in the named operation.
The generic [ElasticKernel seam](../../src/time_series/elastic/mod.rs#L170)
declares interval columns, prefix bounds, candidate bounds, and the exact
verifier. The [walker](../../src/time_series/elastic/walker.rs) calls
`K::Monoid::within` for pruning and final admission. These shared calls fix
the comparator, while each kernel still owes an instance theorem that its
bound cannot exceed its *own* authoritative exact score.

| Bound or verifier | Numeric path and evidence | Remaining machine obligation |
|---|---|---|
| K1 interval column and prefix bound | [MSM](../../src/time_series/msm_kernel.rs), [ERP](../../src/time_series/kernels/erp.rs), [Fréchet](../../src/time_series/kernels/frechet.rs), [TWED](../../src/time_series/kernels/twed.rs), and [banded DTW](../../src/time_series/kernels/dtw.rs) implement `step_column`; DTW also supplies `prefix_lower_bound`. Manifest-listed measure proofs establish selected ideal/abstract inequalities. | For each kernel and interval encoding, relate every rounded relaxed transition, retained row, and prefix carry to the exact point-frontier score under the same `N-F` graph. An ideal K1 theorem alone does not discharge this. |
| K2 path inflation and K3 structural/length rejection | [Walker transitions](../../src/time_series/elastic/walker.rs) combine parent bounds with kernel increments and structural predicates. | Prove monotonicity and checked/top behavior for the executed arithmetic; account for branch-specific empty and no-alignment rules. |
| K4 candidate bound | Each of the [MSM](../../src/time_series/msm_kernel.rs), [ERP](../../src/time_series/kernels/erp.rs), [Fréchet](../../src/time_series/kernels/frechet.rs), [TWED](../../src/time_series/kernels/twed.rs), and [DTW](../../src/time_series/kernels/dtw.rs) kernels defines `candidate_lower_bound`. | Prove `bound <= exact` in the named machine profile, including overflow and any different evaluation order, before rejecting at equality or above cutoff. |
| Exact full-precision verifier | The same kernels implement `exact_with_cutoff`; strict bounded operations reuse exact point-frontier workspace rather than trusting quantized labels. | Prove source recurrence correspondence and public result/tag mapping, including cutoff equality, empty inputs, and numeric failure. |
| Typed vector boxes and candidate filters | [FixedChannelMetric](../../src/time_series/vector.rs#L258) computes point/box bounds; vector ERP, TWED, Fréchet, and banded DTW use their own ordered `f64` paths. [Verus vector kernel](verus/vector_kernels.rs) checks selected typed interval laws. | Fixed layout and positive weights are domain premises. Different rounding or operation order between box bound and exact scorer requires a machine inequality, not only coordinatewise real inclusion. |
| Physical-time TWED product bins | [timestamped_twed_index.rs](../../src/time_series/timestamped_twed_index.rs#L2145) widens computed bin endpoints with `next_down`/`next_up`. | Prove that the entire quantizer map and those endpoints enclose every admitted original under binary64 operations, then compose with K1 and the exact verifier. A single-ULP widening is not self-certifying. |
| Triangle-derived geometric bound | [CBC NC-1](../theory/certified-metric-automata.md#5-numerical-certificates-that-authorize-machine-pruning) proves an ideal-real pivot inequality from certified enclosures. | Add downward-rounded subtraction and a bound on the verifier's own error; a real-only pivot proof cannot authorize machine pruning. |

For each pruning event, the proof record must identify `(operation, domain,
query, parameters, snapshot, cutoff, numeric profile, exact verifier,
candidate identity, bound constructor)`. Its judgment is a machine inequality
under the selected comparator. If an `N-R` or `N-E` certificate is used with
an `N-F` verifier, the record must include the transfer theorem and its
intermediate-range conditions. A missing transfer makes the bound ineligible;
no epsilon or fallback comparison supplies it.
