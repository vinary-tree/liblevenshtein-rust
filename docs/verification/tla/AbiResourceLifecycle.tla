--------------------------- MODULE AbiResourceLifecycle ---------------------------
(***************************************************************************)
(* TLA+ specification of the vinary-tree-interop retained-resource         *)
(* lifecycle (FV obligation #1; interop-level, hosted in liblevenshtein).  *)
(*                                                                         *)
(* Models the protocol every VtResource producer and consumer must obey    *)
(* (vinary-tree-interop/src/lib.rs, VtResource / VtResourceVTable):        *)
(*                                                                         *)
(*   - a published resource carries ONE owned retain;                      *)
(*   - copying the two words does NOT retain (a raw borrow);               *)
(*   - a holder may add an owned retain only through a live reference      *)
(*     while the resource is alive;                                        *)
(*   - every owned retain is released exactly once;                        *)
(*   - provider callbacks (and provider-owned interface vtable pointers)   *)
(*     may be used only under a live owned retain;                         *)
(*   - when the provider's internal count reaches zero the resource is     *)
(*     destroyed and never resurrected.                                    *)
(*                                                                         *)
(* The provider's internal atomic counter is modeled EXPLICITLY as         *)
(* `count`, distinct from the ledger of holders' owned retains, so the     *)
(* central balance law is a checked invariant rather than a definition:    *)
(*                                                                         *)
(*   VT-LIFE-1  LedgerBalanced        count = sum of owned retains         *)
(*   VT-LIFE-2  NoUseAfterFinalRelease callbacks imply count > 0           *)
(*   VT-LIFE-3  NoNegativeCount        count >= 0                          *)
(*   VT-LIFE-4  CopyNeutrality         borrow steps never move the count   *)
(*                                     (action property)                   *)
(*   VT-LIFE-5  ReleaseReachesZero     with bounded work and fair release, *)
(*                                     the count reaches zero (liveness)   *)
(*   VT-LIFE-6  NoUseAfterDestroy      destroyed => no callbacks, count 0, *)
(*                                     and destruction is permanent        *)
(*                                                                         *)
(* Correct-consumer discipline (retains via live references only, release  *)
(* of a retain needed by one's own live callback forbidden) is part of the *)
(* NEXT-state relation: the model checks that the PROTOCOL, followed,      *)
(* guarantees the invariants above under every interleaving.               *)
(*                                                                         *)
(* Companion Rust anchor test:                                             *)
(*   tests/abi_resource_lifecycle_correspondence.rs                        *)
(* which drives the REAL counting provider                                 *)
(*   (tests/support/interop_dictionary.rs) through randomized lifecycle    *)
(* scripts and asserts the same invariants by ID.                          *)
(*                                                                         *)
(* Apalache upgrade path (wave W9): the inductive invariant candidate is   *)
(*   IndInv == TypeOK /\ LedgerBalanced /\ CallbackHoldersOwn              *)
(*             /\ DestroyedIsFinal                                         *)
(* over count as an unbounded Int; type annotations are kept Snowcat-      *)
(* compatible for that attempt.                                            *)
(*                                                                         *)
(* Corresponds to: vinary-tree-interop/src/lib.rs (VtResource,             *)
(*                 VtResourceVTable retain/release/query_interface)        *)
(*                 docs/bindings resource lifecycle documentation          *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS
  \* @type: Set(Str);
  Holders,             \* consumer identities sharing one published resource
  \* @type: Int;
  MaxRetainsPerHolder, \* per-holder cap on concurrently owned retains
  \* @type: Int;
  MaxAcquireOps        \* total budget of retain/borrow steps (bounded work)

ASSUME MaxRetainsPerHolder \in Nat \ {0}
ASSUME MaxAcquireOps \in Nat

VARIABLES
  \* @type: Int;
  count,        \* the provider's internal reference counter
  \* @type: Str -> Int;
  owned,        \* per-holder ledger of owned retains
  \* @type: Str -> Bool;
  borrowed,     \* per-holder possession of a raw two-word copy
  \* @type: Set(Str);
  inCallback,   \* holders currently inside a provider callback
  \* @type: Bool;
  destroyed,    \* the count reached zero and the provider freed the resource
  \* @type: Int;
  opsLeft       \* remaining acquire-step budget (drives the liveness bound)

vars == <<count, owned, borrowed, inCallback, destroyed, opsLeft>>

SumOwned ==
  LET Sum[hs \in SUBSET Holders] ==
        IF hs = {} THEN 0
        ELSE LET h == CHOOSE x \in hs : TRUE
             IN owned[h] + Sum[hs \ {h}]
  IN Sum[Holders]

(***************************************************************************)
(* Initial state: some creator publishes the resource holding its one      *)
(* retain and, having constructed it, the raw words.                       *)
(***************************************************************************)
Init ==
  /\ \E creator \in Holders :
       /\ owned = [h \in Holders |-> IF h = creator THEN 1 ELSE 0]
       /\ borrowed = [h \in Holders |-> h = creator]
  /\ count = 1
  /\ inCallback = {}
  /\ destroyed = FALSE
  /\ opsLeft = MaxAcquireOps

(***************************************************************************)
(* A holder with any live reference hands the raw two words to another     *)
(* holder. "Copying the two words does not retain": the provider counter   *)
(* is untouched (VT-LIFE-4 is checked as an action property over exactly   *)
(* these steps).                                                            *)
(***************************************************************************)
CopyWords(giver, taker) ==
  /\ ~destroyed
  /\ opsLeft > 0
  /\ giver # taker
  /\ borrowed[giver] \/ owned[giver] > 0
  /\ ~borrowed[taker]
  /\ borrowed' = [borrowed EXCEPT ![taker] = TRUE]
  /\ opsLeft' = opsLeft - 1
  /\ UNCHANGED <<count, owned, inCallback, destroyed>>

(***************************************************************************)
(* A holder converts a live reference into one owned retain. The resource  *)
(* must still be alive; the provider increments its counter.               *)
(***************************************************************************)
Retain(h) ==
  /\ ~destroyed
  /\ opsLeft > 0
  /\ borrowed[h] \/ owned[h] > 0
  /\ owned[h] < MaxRetainsPerHolder
  /\ count' = count + 1
  /\ owned' = [owned EXCEPT ![h] = @ + 1]
  /\ opsLeft' = opsLeft - 1
  /\ UNCHANGED <<borrowed, inCallback, destroyed>>

(***************************************************************************)
(* Releasing one owned retain. A holder never releases the retain its own  *)
(* live callback depends on (the discipline the consumer wrappers enforce  *)
(* by construction). When the counter reaches zero the provider destroys   *)
(* the resource permanently.                                                *)
(***************************************************************************)
Release(h) ==
  /\ owned[h] > 0
  /\ ~(h \in inCallback /\ owned[h] = 1)
  /\ count' = count - 1
  /\ owned' = [owned EXCEPT ![h] = @ - 1]
  /\ destroyed' = (count - 1 = 0)
  /\ UNCHANGED <<borrowed, inCallback, opsLeft>>

(***************************************************************************)
(* Provider callbacks: entered only under a live owned retain.             *)
(***************************************************************************)
EnterCallback(h) ==
  /\ ~destroyed
  /\ owned[h] > 0
  /\ h \notin inCallback
  /\ inCallback' = inCallback \cup {h}
  /\ UNCHANGED <<count, owned, borrowed, destroyed, opsLeft>>

ExitCallback(h) ==
  /\ h \in inCallback
  /\ inCallback' = inCallback \ {h}
  /\ UNCHANGED <<count, owned, borrowed, destroyed, opsLeft>>

Next ==
  \/ \E g, t \in Holders : CopyWords(g, t)
  \/ \E h \in Holders : Retain(h)
  \/ \E h \in Holders : Release(h)
  \/ \E h \in Holders : EnterCallback(h)
  \/ \E h \in Holders : ExitCallback(h)

(***************************************************************************)
(* Fairness: exits and releases may not stall forever, so the bounded      *)
(* work budget forces the count to zero (VT-LIFE-5).                       *)
(*                                                                         *)
(* Release needs STRONG fairness: a holder that forever alternates         *)
(* callbacks (enter/exit) has Release(h) enabled only infinitely often,    *)
(* not continuously — TLC exhibits exactly that lasso under WF. Exits stay *)
(* continuously enabled once inside, so weak fairness suffices there. The  *)
(* acquire budget (opsLeft) already guarantees retains cannot refill       *)
(* forever, so strong-fair releases drain the counter.                     *)
(***************************************************************************)
Fairness ==
  /\ \A h \in Holders : SF_vars(Release(h))
  /\ \A h \in Holders : WF_vars(ExitCallback(h))

Spec == Init /\ [][Next]_vars /\ Fairness

(***************************************************************************)
(* Invariants                                                              *)
(***************************************************************************)
TypeOK ==
  /\ count \in 0..(Cardinality(Holders) * MaxRetainsPerHolder + 1)
  /\ owned \in [Holders -> 0..MaxRetainsPerHolder]
  /\ borrowed \in [Holders -> BOOLEAN]
  /\ inCallback \subseteq Holders
  /\ destroyed \in BOOLEAN
  /\ opsLeft \in 0..MaxAcquireOps

\* VT-LIFE-1: the provider's counter equals the outstanding owned retains.
LedgerBalanced == count = SumOwned

\* VT-LIFE-2: nobody is inside a callback after the final release.
NoUseAfterFinalRelease == inCallback # {} => count > 0

\* VT-LIFE-3: the counter never goes negative.
NoNegativeCount == count >= 0

\* Support for VT-LIFE-2/6: every callback holder owns a live retain.
CallbackHoldersOwn == \A h \in inCallback : owned[h] > 0

\* VT-LIFE-6: destruction is exactly count-zero, excludes callbacks, and
\* (with DestroyedIsFinal below) is permanent.
NoUseAfterDestroy ==
  destroyed => /\ count = 0
               /\ inCallback = {}

\* VT-LIFE-4 (action property): steps that only move raw words never move
\* the provider counter.
CopyNeutrality == [][borrowed' # borrowed => count' = count]_vars

\* VT-LIFE-6 (action property): once destroyed, permanently destroyed and
\* the counter never resurrects.
DestroyedIsFinal == [][destroyed => (destroyed' /\ count' = 0)]_vars

\* VT-LIFE-5 (liveness): the bounded-work system drains to destruction.
ReleaseReachesZero == <>[](count = 0 /\ destroyed)

=============================================================================
