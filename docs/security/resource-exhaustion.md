# Resource-exhaustion controls for automata and dynamic programs

**Audience:** application developers accepting untrusted patterns, budgets, or
time series · **Status:** living security guidance

Correct automata can still be unsafe services when a caller controls the size
of a state space. This guide identifies each expansion surface, its asymptotic
failure mode, the library guard, and deployment guidance. It complements the
broader [security and threat model](../SECURITY.md).

## 1. Security model

A **resource guard** rejects or bounds work before an attacker can force
unbounded memory, stack depth, or CPU time. An **admissible bound** is a value
that never overstates the best completion cost and can therefore prune without
false negatives. A **state ceiling** is policy, not a mathematical property:
smaller inputs may still be expensive, and trusted callers may choose a
different ceiling through lower-level APIs.

The library treats pattern text, regular-expression repetition counts, edit
budgets, bracket-kind/depth pairs, time-series values, and serialized sizes as
potentially attacker-controlled.

## 2. Control matrix

| Surface | Explosion mode | Required control | Current status |
|---|---|---|---|
| regex to NFA to language product | left-deep AST work and exponential subset diversity | pre-parse source ceiling, expansion-aware NFA estimate, 4,096-state post-check | enforced by `query_regex` |
| generic `LanguageProduct::new` | caller-supplied state set and $`k+1`$ frontier levels | trusted-input API; caller checks `state_count()` | documented contract |
| generalized operation grid | reachable source/target alignment cells grow with both inputs and cheap operations | exact budget pruning, checked coordinates/costs, hard 1,000,000-cell ceiling | enforced by `try_accepts` / `scaled_distance` |
| generated `OperationSet` | many rules or extreme consumptions multiply slice and edge work at every reachable cell | eager structural validation and aggregate source-plus-target consumption ceiling 4,096 | enforced by construction/evaluation boundaries |
| `SmallDfa` | bit shift outside fixed representation | reject more than 31 real states | enforced |
| bracket kinds $`b`$ and depth $`h`$ | $`N(b,h)=(b^{h+1}-1)/(b-1)`$ states for $`b>1`$ | reject above 4,096 states | enforced before allocation |
| true Damerau budget $`k`$ | up to $`\mathcal{O}(k^2)`$ pending positions per state | hard representability ceiling 255; recommend 1–3; measured `SmallVec` spill | enforced and measured |
| unbanded dynamic time warping | lower bound can remain zero, causing a full dictionary scan | require a Sakoe–Chiba band | enforced by the public constructor's type shape |
| TWED length and carry-aware columns | $`\mathcal{O}(mn)`$ exact work; broad bins or $`\lambda=0`$ weaken pruning | cap both lengths, visited work, and candidate evaluations independently of pruning | implemented semantics; deployment policy required |
| scaled affine costs | multiplication and accumulated-cost overflow; zero extension widens the transition window | exact fallible scaling, checked arithmetic, budget-derived windows, service-level length/budget caps | enforced; deployment caps remain caller policy |
| NaN or infinite series values | expressions such as $`+\infty+(-\infty)`$ become NaN | explicit `is_finite` guards and fuzz cases | retained in MSM; required for every elastic kernel |

The status column describes the shipped boundary. Service-level length,
concurrency, and deadline policy remains the embedding application's
responsibility even where a library guard is enforced.

## 3. Regular-language queries

### 3.1 Why a post-compilation check is insufficient

A Thompson NFA for a long concatenation can be expensive to construct before
`num_states()` is available. Counted repetition is more dangerous: a short
pattern such as `a{1000000}` denotes an enormous construction. Checking only
after compilation permits the resource event one intended to prevent.

`Transducer::query_regex` therefore applies three checks:

1. It limits source scalar count before parsing. The rule is intentionally
   conservative for escaped or syntax-heavy input.
2. It applies flags and resolves group references, then computes a saturating
   Thompson-state estimate. Exact and ranged repetition multiply the inner
   estimate; overflow becomes `usize::MAX` rather than wrapping.
3. It checks the constructed NFA again before starting dictionary traversal.

All three errors use `ParseErrorKind::PatternTooComplex` and report the computed
size and maximum.

### 3.2 State-set diversity

The fixed cost frontier prevents multiplicative growth in edit histories, but
different dictionary prefixes can still produce many distinct NFA subsets.
The theoretical worst case remains exponential in language-state count. The
4,096-state ceiling limits one dimension; service operators should also apply
wall-clock deadlines, request concurrency limits, and pattern-specific quotas.

### 3.3 Stack safety and dictionary cycles

The query iterator is iterative and has no fixed depth 100, so a deep key does
not consume Rust call-stack frames. Parent links consume heap memory
proportional to reached prefixes. Ordinary trie and directed acyclic word graph
backends are finite. Do not expose a cyclic custom `DictionaryNode::edges`
graph without a finite-walk contract.

## 4. Edit budgets

The language frontier has exactly $`k+1`$ slots. Its public budget is `u8`, so
the absolute level count is 256 and checked addition cannot overflow. Runtime
still grows at least linearly in $`k`$; production services should choose a
smaller domain-specific limit. Spell-check APIs commonly need only one to three
edits.

True Damerau has an additional pending-delta dimension. Its compact position
stores delta in one byte, so `Algorithm::MAX_DAMERAU_DISTANCE` is 255. Every
public unit-transition, state, pooled-state, and initial-state entry point
checks this contract before traversal. A larger budget panics instead of
silently dropping unrepresentable macro transitions. Because reachable state
size is still $`\mathcal{O}(k^2)`$, deployments should impose a much smaller
fallible request limit; the measured spell-correction range is 1–3.

At $`k=1,2,3`$, the preregistered maximum reachable state sizes were 2, 4,
and 7. One-position successor buffers held 2, 3, and 4 values respectively and
did not spill their inline `SmallVec<[Position; 4]>`. This evidence justifies
the current inline capacity only over that measured range; it is not a promise
that larger states avoid allocation.

True Damerau is unit-cost and history-dependent. `PositionF64` and the phonetic
NFA product have no pending-delta carrier. Their public boundaries reject the
selector explicitly. Treat such a panic as a programmer configuration error;
validate an untrusted algorithm name at the service boundary and return an
application error rather than invoking an unsupported product.

### 4.1 Generalized operation sets

`GeneralizedAutomaton` materializes only alignment cells reachable within its
exact scaled budget. This is normally a narrow band, but a caller can supply
many cheap operations and long inputs. The initial cell counts toward
`MAX_GENERALIZED_ALIGNMENT_STATES`, and the fallible APIs check the next unique
discovery before inserting a vacant frontier entry. The frontier therefore
cannot allocate past the ceiling while unprocessed cells accumulate. The APIs
distinguish exhaustion from an invalid cost scale; all coordinate,
discovery-count, scale, and accumulated-cost additions are checked. The
Boolean `accepts` compatibility method fails closed; services that need
diagnostics should call `try_accepts`.

`GeneralizedState::try_transition` exactly accumulates scaled weights for its
bounded streaming vocabulary: one-scalar rules and $`\langle2,2\rangle`$,
$`\langle2,1\rangle`$, and $`\langle1,2\rangle`$ intermediates. It returns a
typed unsupported-arity error outside that vocabulary. Use
`GeneralizedAutomaton` for operation-complete validation of arbitrary non-zero
consumption pairs. Neither API silently treats a fractional operation as free.

Call `OperationSet::validate()` before retaining a generated or untrusted rule
collection. Validation rejects non-progressing rules, negative or non-finite
costs, free length-changing rules, checked consumption overflow, and aggregate
declared source-plus-target consumption above
`MAX_OPERATION_SET_TOTAL_CONSUMPTION` (4,096). The aggregate is checked after
every rule, so overflow or a prefix crossing the ceiling fails before any
alignment cell is expanded. `try_with_operations` enforces the same boundary
eagerly; fallible evaluation repeats it so the legacy infallible constructor
cannot bypass the policy. Boolean acceptance maps validation failures to
rejection.

This limit bounds declared rule work, not input work. The separate one-million
reachable-cell ceiling remains necessary, and services should still cap input
length, budget, and wall-clock time. The built-in Hamming, indel, and
bounded-skip presets are tiny validated sets. Specialized dictionary walkers
are not shipped because their preregistered structural edge-reduction gate
failed; do not assume a preset implies trie pruning.

### 4.2 Affine-gap costs and windows

`AffineGapParams::new` accepts only finite, non-negative decimal inputs that
have an exact representable fixed-point scale. Query budgets pass through the
same conversion. Every transition uses checked integer addition; the reference
Gotoh implementation also guards matrix-size multiplication and sentinel
arithmetic. Overflow makes a route unreachable or returns `ScaleError`; it
never wraps into a low-cost route.

The characteristic-vector width is derived from affordable **operations**, not
from the scaled integer budget. For current scaled cost $`c`$, maximum cost
$`k`$, and positive extension cost $`g_e`$, the kernel inspects at most:

```math
W(c)=\left\lfloor\frac{k-c}{g_e}\right\rfloor+1
```

query units, capped by the remaining query length. This prevents decimal scale
from multiplying per-edge work. When $`g_e=0`$, an arbitrarily long gap is
affordable, so correctness requires the full remaining-query window. Services
that accept caller-selected costs must cap query length, dictionary depth,
exact scaled budget, and wall-clock work independently; a zero extension cost
should receive the most restrictive limits.

## 5. Time-series inputs

Dynamic programs over real-valued series must reject or explicitly define NaN
and infinite values. IEEE-754 total ordering does not make arithmetic such as
$`+\infty+(-\infty)`$ meaningful. Interval lower-bound code must keep its
finite-cell guards even when a generic cost abstraction is introduced.

`ElasticTransducer` work is proportional to visited trie edges times the
kernel's live DP width. A loose bound or $`+\infty`$ cutoff can intentionally visit the
whole trie, so deployments must cap indexed-series length, query length, result
count, and wall-clock work at the service boundary. Both convenience and strict
range walks use explicit iterative DFS frames; dictionary depth therefore
consumes heap state rather than process stack. The strict surface also caps
frames, interned states, cached transitions, results, and continuation bytes,
and can resume without reconstructing an unbounded prefix.

Paused range outcomes do not duplicate their accumulated result vector.
Callers inspect the continuation's borrowed `exact_partial()` subset and must
not treat it as absence evidence. Completion charges one
$`2n\operatorname{sizeof}(\mathtt{usize})`$ permutation together with live
workspace scratch before allocation and applies it in place. Generic bounded
kNN similarly charges its fallible output conversion. Timestamped TWED uses
its eventual public output vector as the max heap, so its final sort is
allocation-free. These guarantees cover allocations controlled by the
library; an opaque generic `V: Clone` implementation remains a caller trust
boundary because Rust has no standard fallible-clone contract.

Kernel authors must satisfy K1–K4 as specified in the
[elastic-kernel design](../design/elastic-kernels.md). A heuristic mislabeled as
K1 or K4 can either drop true matches or force unexpectedly broad scans. New
kernels require differential, interval-admissibility, degenerate-bin,
leaf-exactness, and resource-guard tests.

Non-finite queries do not enter interval DP or priority queues. Exact range
search falls back to a deterministic scan, while kNN admits only exact costs
strictly below `TOP`. This preserves semantics but makes the worst-case
$`O(n)`$ behavior explicit; reject non-finite samples at an untrusted API
boundary if that scan is not acceptable.

Dynamic time warping requires an explicit band in `DtwConfig::new(band)`.
Without a band, a dictionary interval column can maintain a zero lower bound
through arbitrary stutters, converting an indexed walk into an
attacker-controlled full scan. A supplied band is still untrusted: when
$`w\ge\max(m,n)`$, live work approaches $`\mathcal{O}(mn)`$. Cap $`w`$
independently of the sequence lengths and reject endpoint length gaps wider
than the configured policy before search.

DTW's query plan allocates lower/upper envelopes and suffix extrema
proportional to query length, not the numeric band. Its incremental interval
LB_Keogh gate runs before child-column allocation, but pruning is an
optimization rather than a quota. Monitor prefix prunes, columns built, column
prunes, exact evaluations, and cutoff abandons. Native costs are squared;
public thresholds are root-valued. Unit confusion can accidentally square the
permitted work region, so use only `DtwTransducer` at public boundaries.

DTW has `IS_METRIC = false` and no `MetricElasticKernel` implementation. Never
circumvent that type gate to place it in a triangle-inequality-dependent index:
the formally checked band-one counterexample is an integrity failure for such
pruning, not merely a performance caveat.

ERP accepts empty sequences and samples equal to its gap value. Those cases can
have zero distance despite different raw lengths, so services must not use a
length-difference guard as an ERP security or pruning bound. Exact ERP takes
$`\mathcal{O}(mn)`$ time; its row cutoff is opportunistic, not a guaranteed
work limit. Cap both sequence lengths before constructing the DP, and reject
non-finite samples rather than relying on the deterministic scan fallback.

TWED also has $`\mathcal{O}(mn)`$ worst-case exact time and
$`\mathcal{O}(\min(m,n))`$ DP memory. Its interval recurrence must carry the
previous target bin, but that fixed-size carry is not a work cap. Broad bins can
weaken the two-sample leaf minima, and $`\lambda=0`$ reduces the length bound
to zero. Cap query length, indexed-series length, visited edges, exact candidate
evaluations, and wall time independently of observed pruning. Empty/nonempty
distance is finite, so a final trie root remains a legitimate exact candidate.
Reject non-finite samples at the service boundary.

`MetricTwedConfig` validates algebraic premises, not resource limits. Its
strict $`\nu>0`$ guarantee makes metric-dependent use sound but does not make
the quadratic recurrence subquadratic. Conversely, raw `TwedConfig` at
$`\nu=0`$ is valid for exact trie search but must never cross a
triangle-dependent index boundary.

Discrete Fréchet also takes $`\mathcal{O}(mn)`$ worst-case exact time and
$`\mathcal{O}(\min(m,n))`$ DP memory. Its candidate cascade copies and sorts
one side for a one-sided Hausdorff bound, adding $`\mathcal{O}(n)`$ temporary
memory and $`\mathcal{O}(n\log n)`$ preprocessing per exact candidate check.
Broad quantization bins or a permissive cutoff can keep interval columns below
threshold and visit the entire trie. Cap sequence lengths, result count,
visited-edge work, and wall time; do not treat successful pruning on typical
data as a resource guarantee. Consecutive stutters may tie at zero and must not
be deduplicated by raw-vector identity unless the application explicitly uses
run-length collapse.

### 5.9 Operation-set binary decoders

Operation sets accept only a versioned bincode envelope or the versioned
Protocol Buffers schema; gzip is an optional outer transport wrapper. The
bincode decoder validates its 20-byte header and declared payload length before
decoding. The protobuf decoder scans varints, keys, wire types, nested message
lengths, operation counts, restriction counts, name bytes, and restriction text
bytes before `prost` constructs schema objects. Both paths validate semantic
arity, applicability, and finite non-negative weights after decoding.

The executable Rocq byte parser proves exact header and length consumption,
bounded ten-byte `uint64` varints, exact length-delimited boundaries, and all
pre-allocation count postconditions. Rust properties feed arbitrary bytes to
each decoder and assert panic freedom plus post-admission limits. Gzip adds a
compressed-input ceiling, an inflated-output ceiling, checksum enforcement,
and exact single-member consumption. DEFLATE and checksum computation inside
`flate2` are explicitly trusted third-party behavior; formal claims begin at
the returned decompressor observation and cover only the crate-owned adapter.

## 6. Deployment checklist

- Prefer `query_regex` over manually compiling an untrusted pattern and calling
  `query_language`; the latter assumes the automaton has already passed policy.
- Cap request pattern length and edit budget again at the service boundary.
- Apply timeouts and bounded concurrency; mathematical pruning does not replace
  operational isolation.
- Enable `perf-instrumentation` only for measurement builds; counters add work
  and should not be interpreted as a security limit.
- Fuzz parsers, repetition counts, Unicode combining sequences, empty inputs,
  maximum budgets, NaN, and both infinities.
- Treat panic-free rejection as part of the public contract. Constructors for
  attacker-controlled objects should return `Result`.
- Monitor nodes visited and edges enumerated, not wall time alone, when tuning
  a guard across machines.
- `search_knn_with_stats` is observational and uses saturating increments; its
  counters do not cap work. Reject or cancel requests using independent policy
  limits, then use `accounting_is_consistent` to detect incomplete telemetry.
- For banded DTW, record prefix prunes and columns built separately so a cheap
  first gate cannot conceal an unexpectedly broad $`\mathcal{O}(w)`$ stage.

## 7. Verification and tests

The standalone [`fuzz/` workspace](../../fuzz/README.md) keeps cargo-fuzz and
`libfuzzer-sys` out of the runtime dependency graph. Each target consumes the
attacker-controlled representation directly and checks an invariant that also
appears in deterministic tests or a formal model:

| Fuzz target | Deterministic trip test | Formal or mathematical invariant |
|---|---|---|
| `regex_nfa_resource` | `query_regex_rejects_automata_above_the_resource_ceiling` covers a 2,050-scalar literal and `a{1000000}` | saturating preflight and the 4,096-state postcondition; language-product frontier and bit-shift obligations are checked by Rocq, Verus, Z3, and cvc5 |
| `bracket_state_growth` | `exponential_guard_precedes_allocation` checks $`N(3,10)=88{,}573>4{,}096`$ and the diagnostic | `bracket_state_count_is_depth_monotone`, the exact witness, and the policy rejection are proved in Rocq and mirrored in Dafny and Verus |
| `true_damerau_budget` | `damerau_budget_ceiling_rejects_incomplete_semantics` trips at 256 while the adjacent test accepts 255 | Rocq proves the one-byte positive-delta representation and $`k^2`$ frontier envelope; Verus, TLA+, Z3, and cvc5 check the executable refinements |
| `banded_dtw` | the `DtwConfig::default()` compile-fail test proves that callers cannot omit the band | Rocq, Verus, Z3, and cvc5 prove band reachability, recurrence symmetry, admissible bounds, and non-negativity |
| `cost_scale_overflow` | `rejects_inexact_nonfinite_negative_and_overflowing_values` forces conversion and rescaling overflow | Rocq, Verus, Z3, and cvc5 check guarded scale multiplication and monotonic non-wrapping accumulation |
| `msm_nonfinite` | `invalid_numeric_inputs_fail_closed_without_nan_cells` forces NaN, both infinities, reversed intervals, invalid constants, and malformed predecessor cells | Rocq proves the interval move, merge, and split minima over explicit finite and infinite endpoints; Rust guards connect IEEE-754 invalid values to the model's valid domain |

The true-Damerau differential target deliberately combines a character-labelled
dictionary with the character-counting Lowrance–Wagner oracle. A byte-labelled
dictionary counts UTF-8 code units and would make non-ASCII comparisons
dimensionally invalid rather than expose an automaton defect.

The fuzz invariants are intentionally stronger than “does not panic”: accepted
true-Damerau results must equal the reference set *and* distance, DTW must be
symmetric and non-NaN, and MSM must never retain a NaN cell. Short smoke runs
are bounded validation evidence, not a proof of absence. The formal results and
their trust classifications are indexed in the
[formal-verification manifest](../verification/FORMAL_VERIFICATION_MANIFEST.tsv).
The generalized-operation properties additionally check budget monotonicity,
empty-side rejection, exact fractional accumulation, and Unicode restrictions;
Rocq, Verus, Z3, and cvc5 mirror the path-cost and coordinate invariants.
