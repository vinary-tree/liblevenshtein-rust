------------------------------ MODULE LlevCallGate ------------------------------
(***************************************************************************)
(* TLA+ specification of the interop call-gate serialization contract     *)
(* (FV obligation #3; interop-level, hosted in liblevenshtein).           *)
(*                                                                         *)
(* Models CallGate in src/bindings.rs (~:203-206): calls into ONE captured *)
(* provider object are serialized through a per-provider mutex unless the  *)
(* provider advertises PARALLEL_REENTRANT                                  *)
(* (vinary-tree-interop dictionary_flags / wfst_flags bit 0), in which     *)
(* case callers enter freely. The gate protects only the callback object;  *)
(* it does not serialize the library, other providers, or other captured   *)
(* instances of the same underlying resource.                              *)
(*                                                                         *)
(* VT-GATE-1 (spec pin): the serialization DOMAIN is one captured          *)
(*   Provider. The model's very shape encodes this — `gate` and            *)
(*   `inCallback` are per-provider functions with no shared state between  *)
(*   providers, so nothing about provider p1 can constrain provider p2.    *)
(*   The companion Rust test demonstrates the resulting cross-provider     *)
(*   concurrency empirically.                                              *)
(* VT-GATE-2 (invariant MutualExclusion): a non-reentrant provider never   *)
(*   has two threads inside a callback simultaneously.                     *)
(* VT-GATE-3 (structure + invariant HolderIsInside): the gate is released  *)
(*   on EVERY exit path — normal return and error return are separate      *)
(*   actions with identical release semantics, and the gate is held        *)
(*   exactly while its holder is inside the callback.                      *)
(* NoStarvation (liveness): with fair scheduling and bounded work, every   *)
(*   thread finishes all its calls — serialization does not starve.        *)
(*                                                                         *)
(* Companion Rust anchor test: tests/abi_callgate_correspondence.rs        *)
(* (threaded overlap detector over providers with and without the          *)
(* PARALLEL_REENTRANT claim, plus the cross-provider concurrency demo).    *)
(*                                                                         *)
(* Corresponds to: src/bindings.rs (CallGate::Serial / parallel path)      *)
(*                 vinary-tree-interop/src/lib.rs (PARALLEL_REENTRANT)     *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS
  \* @type: Set(Str);
  Threads,             \* caller threads
  \* @type: Set(Str);
  Providers,           \* captured provider objects (each with its own gate)
  \* @type: Set(Str);
  ReentrantProviders,  \* subset advertising PARALLEL_REENTRANT
  \* @type: Str;
  NoThread,            \* sentinel: gate not held
  \* @type: Int;
  MaxCallsPerThread    \* bounded work per thread (drives liveness)

ASSUME ReentrantProviders \subseteq Providers
ASSUME NoThread \notin Threads
ASSUME MaxCallsPerThread \in Nat \ {0}

VARIABLES
  \* @type: Str -> Str;
  gate,        \* per-provider mutex holder (NoThread when free)
  \* @type: Str -> Set(Str);
  inCallback,  \* per-provider set of threads inside a callback
  \* @type: Str -> Int;
  callsLeft    \* per-thread remaining call budget

vars == <<gate, inCallback, callsLeft>>

Init ==
  /\ gate = [p \in Providers |-> NoThread]
  /\ inCallback = [p \in Providers |-> {}]
  /\ callsLeft = [t \in Threads |-> MaxCallsPerThread]

(***************************************************************************)
(* A thread enters a NON-REENTRANT provider: acquire that provider's gate  *)
(* and enter in one step (the Rust mutex guard makes lock+call atomic from *)
(* the provider's perspective).                                            *)
(***************************************************************************)
EnterSerial(t, p) ==
  /\ p \notin ReentrantProviders
  /\ callsLeft[t] > 0
  /\ gate[p] = NoThread
  /\ t \notin inCallback[p]
  /\ gate' = [gate EXCEPT ![p] = t]
  /\ inCallback' = [inCallback EXCEPT ![p] = @ \cup {t}]
  /\ callsLeft' = [callsLeft EXCEPT ![t] = @ - 1]

(***************************************************************************)
(* A thread enters a REENTRANT provider: no gate involved; concurrent      *)
(* entries are the provider's advertised capability.                       *)
(***************************************************************************)
EnterReentrant(t, p) ==
  /\ p \in ReentrantProviders
  /\ callsLeft[t] > 0
  /\ t \notin inCallback[p]
  /\ inCallback' = [inCallback EXCEPT ![p] = @ \cup {t}]
  /\ callsLeft' = [callsLeft EXCEPT ![t] = @ - 1]
  /\ UNCHANGED gate

(***************************************************************************)
(* Exit paths. The callback returning Ok and the callback failing (error   *)
(* status, recorded fault) are DISTINCT actions with IDENTICAL gate        *)
(* release semantics: VT-GATE-3 is that no exit path leaks the gate.       *)
(***************************************************************************)
ReleaseIfHeld(t, p) ==
  gate' = [gate EXCEPT ![p] = IF @ = t THEN NoThread ELSE @]

ExitOk(t, p) ==
  /\ t \in inCallback[p]
  /\ inCallback' = [inCallback EXCEPT ![p] = @ \ {t}]
  /\ ReleaseIfHeld(t, p)
  /\ UNCHANGED callsLeft

ExitError(t, p) ==
  /\ t \in inCallback[p]
  /\ inCallback' = [inCallback EXCEPT ![p] = @ \ {t}]
  /\ ReleaseIfHeld(t, p)
  /\ UNCHANGED callsLeft

Next ==
  \/ \E t \in Threads, p \in Providers : EnterSerial(t, p)
  \/ \E t \in Threads, p \in Providers : EnterReentrant(t, p)
  \/ \E t \in Threads, p \in Providers : ExitOk(t, p)
  \/ \E t \in Threads, p \in Providers : ExitError(t, p)

(***************************************************************************)
(* Fairness. Exits are continuously enabled while inside (weak fairness    *)
(* suffices). Serial entry is only intermittently enabled — the gate       *)
(* toggles as competitors enter and exit — so NoStarvation needs STRONG    *)
(* fairness on entry (the same WF-vs-SF distinction TLC demonstrated on    *)
(* the lifecycle model's Release action).                                  *)
(***************************************************************************)
Fairness ==
  /\ \A t \in Threads, p \in Providers : SF_vars(EnterSerial(t, p))
  /\ \A t \in Threads, p \in Providers : WF_vars(EnterReentrant(t, p))
  /\ \A t \in Threads, p \in Providers : WF_vars(ExitOk(t, p))

Spec == Init /\ [][Next]_vars /\ Fairness

(***************************************************************************)
(* Invariants                                                              *)
(***************************************************************************)
TypeOK ==
  /\ gate \in [Providers -> Threads \cup {NoThread}]
  /\ inCallback \in [Providers -> SUBSET Threads]
  /\ callsLeft \in [Threads -> 0..MaxCallsPerThread]

\* VT-GATE-2: serialized providers admit at most one thread at a time.
MutualExclusion ==
  \A p \in Providers \ ReentrantProviders : Cardinality(inCallback[p]) <= 1

\* VT-GATE-3 (state form): a held gate's holder is inside that callback,
\* and a serialized callback occupant holds the gate — the gate can never
\* be leaked by an exit path or held by a bystander.
HolderIsInside ==
  \A p \in Providers \ ReentrantProviders :
    /\ gate[p] # NoThread => gate[p] \in inCallback[p]
    /\ \A t \in inCallback[p] : gate[p] = t

\* Reentrant providers never engage the gate at all.
ReentrantGateUnused ==
  \A p \in ReentrantProviders : gate[p] = NoThread

\* NoStarvation: every thread finishes its budget and every callback ends.
AllWorkCompletes ==
  <>[](/\ \A t \in Threads : callsLeft[t] = 0
       /\ \A p \in Providers : inCallback[p] = {})

=============================================================================
