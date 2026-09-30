# First executable CBC correspondence route: Standard Levenshtein

**Decision status:** architecture and proof-obligation design. No Rust
function in this route is currently certified by this document.
**Inspected source revision:** `6d058c93` on
`codex/julia-distance-family-parity`. The files named below must be reviewed
again if they change. A source hash can detect drift but cannot prove a
semantic relation.

## Decision and executable target

Use **direct refinement of the Rust source** for the first integer instance.
The first named executable artifact is
[`transition_standard_into`](../../src/transducer/transition.rs), called by
[`StandardV::successors`](../../src/transducer/variants/standard.rs). The
initial proof scope is unit-cost Standard Levenshtein over `u8` units with
`Unrestricted` substitution policy, complete-string matching, and semantic
cutoff 4. On this scope, the packed implementation is ineligible:
[`PackedEditLaneLayout::eligible`](../../src/transducer/packed_lanes.rs)
requires a cutoff at most 3. The ordinary public
[`query_units_with_distance`](../../src/transducer/mod.rs) path therefore
selects `UnitCostMachine::Positional` without a benchmark-only override, when
the dictionary is not suffix-based. This identifies a real production path;
calling the transition helper alone would establish only a local theorem.
The eventual named library artifact is the `liblevenshtein` package's
`rlib` from [Cargo.toml](../../Cargo.toml), built from `src/lib.rs` with its
declared default `parking_lot` feature and a recorded target/toolchain. Its
public entry is `Transducer::<D, Unrestricted>::query_units_with_distance`
for a dictionary whose unit type is `u8`. This
document identifies that executable build route; it contains neither a
source proof nor a certified build of the artifact.

The choice of cutoff 4 is a **first proof instance**, not a final limit on
Standard Levenshtein CBC. General cutoffs, packed execution, string
conversion, suffix/prefix matching, restricted substitution, ordered output,
and resource profiles need their own checked extensions. The first completed
source theorem must state exactly which observation it covers. A local
transition proof cannot be labeled as a certified query operation.

| Route considered | Repository constraint | Decision for the first instance |
|---|---|---|
| Proved lowering or extraction from Rocq | The checked Rocq expression language has no proved lowering to this Rust transition engine, its `SmallVec` output, or its dictionary traversal | Requires a new verified compiler or extraction path before it reaches the current executable |
| Translation validation of generated Rust | No source-to-model validator covers `Position`, checked `usize`, closures, generated-state caches, or traversal | Appropriate for later specialized code generation only after the validator itself is proved sound |
| Direct source refinement | The existing Rust function and its call chain are named and inspectable; Verus and Rust-shaped proof islands exist in the repository | Selected; every source step remains an explicit proof obligation until verified against the actual function or a proved equivalent executable core |

Verus is available in the repository's formal toolchain. Existing Verus files
are Rust-shaped mathematical mirrors; their success alone is **not** a proof
of the implementation above. A source proof must verify code that the
production build actually executes, or prove a source-to-core equivalence and
show that the production caller uses that core. A separately copied model
paired with unchanged Rust is insufficient.

## The concrete call path

For the stated scope, the proof graph begins at the public units-native
query. The iterator captures a dictionary traversal root and constructs a
unit-cost machine. `PackedStandardMachine::new` returns `None` at cutoff 4,
so `UnitCostMachine::seeded_positional` stores an epsilon-closed initial
state. The traversal processes each edge through a prepared positional row
or `UnitCostMachine::step_positional`, which reaches
`CachedUnitTransitions::transition_generated_prepared`. A cache miss
materializes a canonical source state and calls
`transition_epsilon_closed_state_pooled_cached`. Its inner loop obtains a
characteristic window and calls `StandardV::successors`, hence
`transition_standard_into`. A cache hit bypasses that call and must be proved
equivalent to the certified miss. Finality uses `FinishMode::Complete` and
`State::infer_distance`, then the iterator applies the inclusive cutoff and
materializes the matched original. The public `QueryIterator::next` dispatches
to `PathQueryIteratorCore::next_match` and `advance`; the selected
`ResultPathStrategy::materialize_units` reconstructs the original units,
`TraversalSession::accepts_final_units` checks final-unit eligibility, and
the `QueryResult` implementation for `UnitCandidate<u8>` constructs the
visible `(units, distance)` result.

| Edge in the production path | Source anchor | Correspondence required |
|---|---|---|
| Public input to captured query | `src/transducer/mod.rs::query_units_with_distance`; `src/transducer/query.rs::with_traversal_root_and_units` | Query units, cutoff, policy, dictionary kind, and captured revision agree with the formal request |
| Dispatch to positional engine | `src/transducer/transition.rs::UnitCostMachine::seeded`; `src/transducer/packed_lanes.rs::eligible` | At cutoff 4 the packed arm is unreachable; alternative algorithm and benchmark controls are outside this instance |
| Seed to generated frontier | `transition.rs::seeded_positional`, `initial_state`, `seed_generated_state` | Initial positions and epsilon closure preserve the reference empty-prefix residual language and minimum cost |
| Edge label to characteristic class | `transition.rs::CharacteristicCache::class_for`, `matches`, `CachedCharacteristics::window` | Each Boolean is the exact `u8` equality result at the corresponding query index; padding is false and indexing is checked |
| Position successor | `variants/standard.rs::StandardV::successors`; `transition.rs::transition_standard_into` | Pushed successors are sound and their residual language/minimum cost covers the reference transition after proved dominance reduction; `Some(0)` deliberately omits dominated insertion/substitution alternatives |
| State successor and closure | `transition.rs::transition_epsilon_closed_state_into`, `epsilon_closure_mut`; `state.rs::insert_with` | Canonical antichain/subsumption and epsilon closure preserve the same residual language and minimum cost |
| Generated cache and interning | `transition.rs::transition_generated_prepared`, `intern_positions`, generated target table | Hit and miss denote the same successor; `UNCOMPUTED`, `EMPTY`, and state IDs are distinct and scoped to the query/configuration |
| Final score and admission | `transition.rs::finish_distance`; `state.rs::infer_distance`; `query.rs::advance` | The returned score equals the reference distance when at most 4; results above 4 are excluded by the inclusive comparison |
| Original enumeration | `query.rs::queue_children_and_finality`; `dictionary_traversal.rs::TraversalSession::accepts_final_units` | Every eligible captured original is visited once or retained as pending; no collision member is omitted |
| Public result construction | `query.rs::QueryIterator::next`, `PathQueryIteratorCore::next_match`, `advance`; `dictionary_traversal.rs::ResultPathStrategy::materialize_units`; `query_result.rs::QueryResult for UnitCandidate` | The yielded unit sequence and score are the checked original and authoritative distance, including final-unit eligibility and multiplicity |

The path has both row-prepared and ordinary step entry points. The
correspondence proof must cover the production branch selected by each
scheduler that claims this instance. A proof of only the ordinary
`step_positional` method cannot certify a prepared-row query that calls
`transition_generated_prepared` directly. The fast path for zero cutoff is
outside this first cutoff-4 scope; a general-cutoff theorem must include it.

## Proof dependency graph

The graph is ordered by what a later theorem may assume. Each node names a
proposition and its evidence boundary; no unchecked arrow is treated as a
proved refinement.

| Node | Proposition to establish | Depends on | Evidence that would close it |
|---|---|---|---|
| B: byte encoding | A Rust `u8` sequence has an equality-preserving, length-preserving encoding as a Rocq `ascii` list | Exact eight-bit conversion and its inverse | Checked encoding lemmas for every byte, sequence, equality test, and length; `Core/Definitions.v` fixes `Char := ascii` rather than a polymorphic element type |
| M: mathematical score | Unit-cost Levenshtein on finite `u8` sequences has the intended recurrence and metric laws | B; valid sequence domain and ordinary equality | Transport the Rocq recurrence in `core/theories/Core/LevDistance.v` and metric theorem in `core/theories/MainTheorems.v` across B, recording their exact hypotheses |
| L: local position transition | `transition_standard_into` preserves the reference residual language and minimum cost for a normal position and correct characteristic window, after dominance reduction | B, M; checked `usize` arithmetic, `SmallVec` behavior, and overrun-position semantics | Source-level proof of every branch plus a reduction theorem showing why omitted higher-cost successors in `Some(0)` are dominated; raw successor-set equality is not claimed |
| C: closure and canonical state | Epsilon closure and subsumption preserve the represented residual language and minimum cost | L; ordering/antichain invariant | Proof of `State::insert_with`, closure, initial seed, and terminal inference against the same reference state relation |
| K: cached generated transition | Cache hit, miss, and state interning produce equivalent canonical successors for one query/configuration | C; characteristic-class correctness | Source proof of class construction, cached table tags, equality/full-key scope, and ID lifetime |
| D: dispatch reachability | The selected public query at cutoff 4 executes the positional branch and only certified transition paths | K; exact selector predicate | Source proof for `seeded`, prepared-row dispatch, and feature/configuration constraints |
| O: completed query observation | The iterator emits exactly the captured originals with authoritative score at most 4 | D; final-score inference; traversal coverage and ownership | Source proof of finality, cutoff comparison, collision handling, pending work, and completion; use the appropriate observation profile |
| E: executable binding | The verified source is the code built for the named artifact and feature set | L–O | Verified in-source body or proved source-to-executable-core link plus build/feature provenance; hashes serve only as drift checks |
| R: resource and optimization claim | Runtime work, memory, and code path meet a stated objective without changing accepted observations | E; separate resource contracts | Bounded resource accounting and controlled performance measurements after source correspondence, with no certificate replay in the hot loop unless its cost is included |

The Rocq theorem underlying `M` and the generic certificate/session lemmas
are presently proof islands. The byte bridge `B` and source nodes `L` through
`E` are **open**. In particular,
[`CertifiedContracts.v`](temporal_automata/theories/CertifiedContracts.v)
proves facts about expression syntax, not this source transition; and
[`CertifiedMetricExecution.v`](temporal_automata/theories/CertifiedMetricExecution.v)
proves a conditional composition theorem, not its Rust premises. Combining
their names in a manifest does not close `L` through `E`.

## Source and toolchain trust boundary

The first local source proof must specify a relation between a Rust
`Position` and the formal residual position, with `kind=Normal`,
`aux=0`, and `num_errors` within the cutoff. **`term_index` may exceed the
query length.** For an empty query at cutoff 4, the padded false window can
send `(0,0)` to both `(0,1)` and `(1,1)`, and Standard subsumption retains
both. The representation relation must give such overrun positions their
actual suffix/finish semantics and account for their possible continued
dictionary transitions; imposing a query-length bound would exclude real
execution states.
It must prove that `checked_add` failures drop only positions that cannot be
represented by the supported machine domain; treating overflow as an
ordinary missing successor without a domain proof is unsound. The
`index_of_match` window must be shown to correspond to the exact query
suffix after `CharacteristicCache` padding and slicing. An empty output
buffer is a precondition of `transition_standard_into`; `debug_assert!` alone
does not enforce it in optimized builds.

The trusted boundary includes Rocq's kernel for imported mathematics, the
source verifier and its model of Rust integers/collections, any formal bridge
between Rocq and Verus propositions, the Rust compiler and target semantics,
and the chosen dictionary implementation's snapshot/lifetime contract.
`SmallVec`, `FxHashMap`, `Arc`, `NonNull`, and `libdictenstein` traversal APIs
need verified specifications or source proofs where they affect the claim.
If the proof uses an ideal sequence model for `SmallVec`, it must prove the
actual `clear`, `push`, iteration, and allocation/error behavior realizes
that model. The `u8`-to-Rocq-`ascii` bridge is another separate obligation:
the current Rocq `Char` is concrete `ascii`, so it does not automatically
describe Rust `u8` or Unicode `char`. The proof statement must bind the exact
compilation features, pointer width, policy type, algorithm, cutoff, and
observation profile.

The local and query proof obligations are intentionally separate. A mutation
of `transition_standard_into` that leaves the Rocq model unchanged must
invalidate `L` or `E`; a valid mathematical certificate must never authorize
an unrelated Rust function. Similarly, a proof for cutoff 4 cannot be used
to certify packed cutoffs 0–3, witness output, ordered kNN, or resource
usage. Those claims require their own source nodes and measurements. Offline
proof checking has no per-transition cost; any runtime guard or certificate
checker introduced later must be counted in the time and space objective.
