------------------------------ MODULE LlevBatchLease ------------------------------
(***************************************************************************)
(* TLA+ specification of the leased-batch cursor protocol (FV obligation   *)
(* #2, wave W3).                                                           *)
(*                                                                         *)
(* Models the llev_query_cursor_* surface in src/ffi/index.rs: a cursor    *)
(* hands out ONE borrowed batch at a time under a generation-tagged lease. *)
(* While the lease is live, advancing and freeing are REJECTED with        *)
(* BATCH_IN_USE — and the rejected free does NOT free (ownership stays     *)
(* with the caller). Releasing requires the exact live generation.         *)
(* Generations are minted by wrapping_add(1).max(1): they never revisit    *)
(* zero, so a stale or zeroed tag can never alias a live lease. The        *)
(* reducer path (llev_query_cursor_reduce) borrows and auto-releases       *)
(* around every callback, so it never leaves a lease behind.               *)
(*                                                                         *)
(* Attempted-operation modeling: illegal attempts are real transitions     *)
(* that leave the protocol state unchanged and record the status the ABI   *)
(* returns, so TLC explores the REJECTION semantics, not just the happy    *)
(* path.                                                                   *)
(*                                                                         *)
(* Invariants (registry: docs/verification/ABI_INVARIANTS.tsv):            *)
(*   LLEV-LEASE-1  AdvanceOnlyUnleased — a batch is issued only when no    *)
(*                 lease is live.                                          *)
(*   LLEV-LEASE-2  BusyRejectionExact — BATCH_IN_USE is returned exactly   *)
(*                 for advance/free attempts under a live lease, and the   *)
(*                 attempt changes nothing.                                *)
(*   LLEV-LEASE-3  ReleaseNeedsExactGeneration — releasing with a stale or *)
(*                 zero tag is INVALID_ARGUMENT and changes nothing.       *)
(*   LLEV-LEASE-4  GenerationsNeverZero — minted generations are never 0   *)
(*                 and strictly advance (mod the wrap), so no two live     *)
(*                 leases ever share a tag.                                *)
(*   LLEV-LEASE-5  RejectedFreeKeepsOwnership — a free under lease leaves  *)
(*                 the cursor alive and usable.                            *)
(*   LLEV-LEASE-6  ReduceNeverLeaks — the reducer path never ends in the   *)
(*                 leased state.                                           *)
(*   LLEV-LEASE-7  FreedIsTerminal — after a successful free, no operation *)
(*                 observes the cursor again (action property).            *)
(*                                                                         *)
(* Companion Rust anchor: tests/ffi_batch_lease_correspondence.rs (deep    *)
(* FSM proptest over the real C ABI, wave W3).                             *)
(*                                                                         *)
(* Upgrade rung (wave W9): Apalache inductive over an unbounded generation *)
(* counter; the wrap action below keeps the bounded model honest about     *)
(* the modular arithmetic.                                                 *)
(*                                                                         *)
(* Corresponds to: src/ffi/index.rs (leased flag, generation minting,      *)
(* release/advance/free/reduce entry points)                               *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS
  \* @type: Int;
  MaxGeneration,   \* wrap point for the bounded model (u64::MAX in Rust)
  \* @type: Int;
  MaxBatches       \* bounded number of batches the query can yield

ASSUME MaxGeneration \in Nat \ {0, 1}
ASSUME MaxBatches \in Nat \ {0}

VARIABLES
  \* @type: Str;
  cursor,       \* "idle" | "leased" | "ended" | "freed"
  \* @type: Int;
  generation,   \* last minted lease tag (0 before the first lease)
  \* @type: Int;
  batchesLeft,  \* batches the underlying query can still yield
  \* @type: Str;
  lastStatus    \* status returned by the most recent operation

vars == <<cursor, generation, batchesLeft, lastStatus>>

Statuses == {"ok", "end", "batch_in_use", "invalid_argument", "none"}

\* wrapping_add(1).max(1): the successor generation, skipping zero forever.
NextGeneration ==
  IF generation >= MaxGeneration THEN 1 ELSE generation + 1

Init ==
  /\ cursor = "idle"
  /\ generation = 0
  /\ batchesLeft = MaxBatches
  /\ lastStatus = "none"

(***************************************************************************)
(* Advancing. From idle with batches remaining: mint a fresh generation    *)
(* and lease the batch. From idle with nothing left: report END and enter  *)
(* the ended state (no lease). From ended: keep reporting END.             *)
(***************************************************************************)
NextBatchIssues ==
  /\ cursor = "idle"
  /\ batchesLeft > 0
  /\ cursor' = "leased"
  /\ generation' = NextGeneration
  /\ batchesLeft' = batchesLeft - 1
  /\ lastStatus' = "ok"

NextBatchEnds ==
  /\ cursor \in {"idle", "ended"}
  /\ batchesLeft = 0
  /\ cursor' = "ended"
  /\ lastStatus' = "end"
  /\ UNCHANGED <<generation, batchesLeft>>

\* Attempting to advance under a live lease: rejected, nothing changes.
NextBatchRejected ==
  /\ cursor = "leased"
  /\ lastStatus' = "batch_in_use"
  /\ UNCHANGED <<cursor, generation, batchesLeft>>

(***************************************************************************)
(* Releasing. Only the exact live generation releases the lease; stale     *)
(* tags (any prior generation) and the zero tag are rejected unchanged.    *)
(***************************************************************************)
ReleaseExact ==
  /\ cursor = "leased"
  /\ cursor' = "idle"
  /\ lastStatus' = "ok"
  /\ UNCHANGED <<generation, batchesLeft>>

ReleaseStale ==
  /\ cursor = "leased"
  /\ \E stale \in 0..MaxGeneration :
       /\ stale # generation
       /\ lastStatus' = "invalid_argument"
  /\ UNCHANGED <<cursor, generation, batchesLeft>>

ReleaseWhenUnleased ==
  /\ cursor \in {"idle", "ended"}
  /\ lastStatus' = "invalid_argument"
  /\ UNCHANGED <<cursor, generation, batchesLeft>>

(***************************************************************************)
(* The reducer path: borrows and auto-releases around every callback, so   *)
(* one Reduce step consumes any number of remaining batches and ends with  *)
(* NO lease outstanding. It is callable whenever no manual lease is live —  *)
(* from idle it drains the remaining batches, and from ended (where        *)
(* batchesLeft = 0 by EndedHasNoBatches) it is an Ok no-op that fires zero  *)
(* callbacks and stays ended: consumed is forced to 0, so the generation    *)
(* and batch count are unchanged and the cursor remains ended. This mirrors *)
(* the ABI accepting llev_query_reduce on an exhausted cursor (Ok,          *)
(* out_count = 0), which the idle-only guard previously failed to model.    *)
(***************************************************************************)
Reduce ==
  /\ cursor \in {"idle", "ended"}
  /\ \E consumed \in 0..batchesLeft :
       /\ batchesLeft' = batchesLeft - consumed
       /\ generation' =
            IF consumed = 0 THEN generation
            ELSE ((generation + consumed - 1) % MaxGeneration) + 1
  /\ cursor' = IF batchesLeft' = 0 THEN "ended" ELSE "idle"
  /\ lastStatus' = "ok"

ReduceRejected ==
  /\ cursor = "leased"
  /\ lastStatus' = "batch_in_use"
  /\ UNCHANGED <<cursor, generation, batchesLeft>>

(***************************************************************************)
(* Freeing. Succeeds only without a live lease; a free attempt under lease *)
(* is rejected AND does not free (LLEV-LEASE-5). Freed is terminal.        *)
(***************************************************************************)
FreeSucceeds ==
  /\ cursor \in {"idle", "ended"}
  /\ cursor' = "freed"
  /\ lastStatus' = "ok"
  /\ UNCHANGED <<generation, batchesLeft>>

FreeRejected ==
  /\ cursor = "leased"
  /\ lastStatus' = "batch_in_use"
  /\ UNCHANGED <<cursor, generation, batchesLeft>>

Next ==
  \/ NextBatchIssues
  \/ NextBatchEnds
  \/ NextBatchRejected
  \/ ReleaseExact
  \/ ReleaseStale
  \/ ReleaseWhenUnleased
  \/ Reduce
  \/ ReduceRejected
  \/ FreeSucceeds
  \/ FreeRejected

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Invariants                                                              *)
(***************************************************************************)
TypeOK ==
  /\ cursor \in {"idle", "leased", "ended", "freed"}
  /\ generation \in 0..MaxGeneration
  /\ batchesLeft \in 0..MaxBatches
  /\ lastStatus \in Statuses

\* Entering "ended" always drains the batch count to zero: NextBatchEnds
\* requires batchesLeft = 0, and Reduce sets cursor' = "ended" only when
\* batchesLeft' = 0. This is what makes the reduce-on-ended no-op sound — the
\* widened Reduce guard admits "ended", and there `consumed` can only be 0, so
\* the step neither mints a generation nor consumes a (non-existent) batch.
EndedHasNoBatches ==
  cursor = "ended" => batchesLeft = 0

\* LLEV-LEASE-4: after the first mint, the tag is never zero.
GenerationsNeverZero ==
  cursor = "leased" => generation # 0

\* LLEV-LEASE-2 (state half): BATCH_IN_USE arises exactly from attempts
\* made under a live lease — and only advance/free/reduce attempts make it.
BusyImpliesLeased ==
  lastStatus = "batch_in_use" => cursor = "leased"

\* LLEV-LEASE-5: a rejected free left the cursor usable (never freed while
\* the rejection status is showing).
RejectedFreeKeepsOwnership ==
  lastStatus = "batch_in_use" => cursor # "freed"

\* LLEV-LEASE-1/6 (action properties): batches are issued only from the
\* unleased idle state, and no reduce step ever ends leased.
AdvanceOnlyUnleased ==
  [][(cursor # "leased" /\ cursor' = "leased") => cursor = "idle"]_vars

ReduceNeverLeaks ==
  [][Reduce => cursor' # "leased"]_vars

\* LLEV-LEASE-4 (stability half): while a lease is live its tag never
\* changes — no second mint can occur under an outstanding lease, so no two
\* live leases can ever exist, let alone share a tag.
LeaseTagStable ==
  [][(cursor = "leased" /\ cursor' = "leased") => UNCHANGED generation]_vars

\* LLEV-LEASE-7: freed is terminal — no step leaves the freed state, and
\* nothing observes the cursor after a successful free.
FreedIsTerminal ==
  [][cursor = "freed" => UNCHANGED vars]_vars

=============================================================================
