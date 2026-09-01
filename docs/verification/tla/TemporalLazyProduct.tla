-------------------------- MODULE TemporalLazyProduct --------------------------
(***************************************************************************)
(* Bounded, resumable dictionary x lazy-temporal-automaton product.         *)
(*                                                                         *)
(* Every pending entry carries a compact automaton state identifier. The    *)
(* arena and transition cache contain only states/transitions constructed   *)
(* by reached dictionary edges. A page stop is Paused, never Complete.      *)
(***************************************************************************)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS Tau, PageLimit, MaxQueue, MaxStates, MaxTransitions

Nodes == 1..8
Root == 1
States == 0..5
Refs == {10, 30, 40, 60, 70}

Children == (1 :> {2, 5}) @@ (2 :> {3, 4, 7, 8}) @@ (3 :> {}) @@ (4 :> {})
         @@ (5 :> {6}) @@ (6 :> {}) @@ (7 :> {}) @@ (8 :> {})
NodeState == (1 :> 0) @@ (2 :> 1) @@ (3 :> 2) @@ (4 :> 3)
          @@ (5 :> 4) @@ (6 :> 5) @@ (7 :> 2) @@ (8 :> 3)
NodeFinal == (1 :> TRUE) @@ (2 :> FALSE) @@ (3 :> TRUE) @@ (4 :> TRUE)
          @@ (5 :> FALSE) @@ (6 :> TRUE) @@ (7 :> TRUE) @@ (8 :> TRUE)
NodeRef == (1 :> 10) @@ (2 :> 10) @@ (3 :> 30) @@ (4 :> 40)
        @@ (5 :> 10) @@ (6 :> 60) @@ (7 :> 70) @@ (8 :> 70)
LowerBound == (1 :> 0) @@ (2 :> 1) @@ (3 :> 2) @@ (4 :> 1)
           @@ (5 :> 5) @@ (6 :> 6) @@ (7 :> 5) @@ (8 :> 2)
Exact == (1 :> 0) @@ (2 :> 0) @@ (3 :> 2) @@ (4 :> 3)
      @@ (5 :> 0) @@ (6 :> 7) @@ (7 :> 6) @@ (8 :> 10)

RECURSIVE ReachFrom(_)
ReachFrom(n) == {n} \cup UNION {ReachFrom(child) : child \in Children[n]}

AdmissibleTable ==
  \A node \in Nodes : \A terminal \in ReachFrom(node) :
    NodeFinal[terminal] => LowerBound[node] <= Exact[terminal]

Edges == UNION {
  {<<source, child>> : child \in Children[source]} : source \in Nodes}

Expected == {NodeRef[node] : node \in
  {candidate \in Nodes : NodeFinal[candidate] /\ Exact[candidate] <= Tau}}

VARIABLES pending, done, emitted, arena, cache, totalWork, pageWork, status
vars == <<pending, done, emitted, arena, cache, totalWork, pageWork, status>>

Unprocessed == pending \ done
Exhausted == Unprocessed = {}

Init ==
  /\ pending = {Root}
  /\ done = {}
  /\ emitted = {}
  /\ arena = {NodeState[Root]}
  /\ cache = {}
  /\ totalWork = 0
  /\ pageWork = 0
  /\ status = "Running"

Process ==
  /\ status = "Running"
  /\ pageWork < PageLimit
  /\ ~Exhausted
  /\ \E node \in Unprocessed :
      LET admitted == LowerBound[node] <= Tau
          children == IF admitted THEN Children[node] ELSE {}
          newPending == pending \cup children
          newArena == arena \cup {NodeState[child] : child \in children}
          newCache == cache \cup {<<node, child>> : child \in children}
      IN
      /\ Cardinality(newPending \ (done \cup {node})) <= MaxQueue
      /\ Cardinality(newArena) <= MaxStates
      /\ Cardinality(newCache) <= MaxTransitions
      /\ pending' = newPending
      /\ done' = done \cup {node}
      /\ arena' = newArena
      /\ cache' = newCache
      /\ emitted' =
          IF admitted /\ NodeFinal[node] /\ Exact[node] <= Tau
          THEN emitted \cup {NodeRef[node]}
          ELSE emitted
      /\ totalWork' = totalWork + 1
      /\ pageWork' = pageWork + 1
      /\ UNCHANGED status

Pause ==
  /\ status = "Running"
  /\ pageWork = PageLimit
  /\ ~Exhausted
  /\ status' = "Paused"
  /\ UNCHANGED <<pending, done, emitted, arena, cache, totalWork, pageWork>>

Resume ==
  /\ status = "Paused"
  /\ status' = "Running"
  /\ pageWork' = 0
  /\ UNCHANGED <<pending, done, emitted, arena, cache, totalWork>>

Finish ==
  /\ status = "Running"
  /\ Exhausted
  /\ status' = "Complete"
  /\ UNCHANGED <<pending, done, emitted, arena, cache, totalWork, pageWork>>

Stutter ==
  /\ status = "Complete"
  /\ UNCHANGED vars

Next == Process \/ Pause \/ Resume \/ Finish \/ Stutter
Spec == Init /\ [][Next]_vars /\ WF_vars(Process) /\ WF_vars(Pause)
        /\ WF_vars(Resume) /\ WF_vars(Finish)

TypeOK ==
  /\ pending \subseteq Nodes
  /\ done \subseteq Nodes
  /\ emitted \subseteq Refs
  /\ arena \subseteq States
  /\ cache \subseteq Edges
  /\ totalWork \in Nat
  /\ pageWork \in 0..PageLimit
  /\ status \in {"Running", "Paused", "Complete"}
  /\ AdmissibleTable

QueueBound == Cardinality(Unprocessed) <= MaxQueue
StateArenaBound == Cardinality(arena) <= MaxStates
TransitionCacheBound == Cardinality(cache) <= MaxTransitions

NoFalsePositives ==
  \A ref \in emitted : \E node \in Nodes :
    NodeFinal[node] /\ NodeRef[node] = ref /\ Exact[node] <= Tau

PruneSound ==
  \A node \in done : LowerBound[node] > Tau =>
    \A terminal \in ReachFrom(node) :
      NodeFinal[terminal] => Exact[terminal] > Tau

CompleteOnlyAfterExhaustion == status = "Complete" => Exhausted
CompleteIsExact == status = "Complete" => emitted = Expected
PausedIsExplicitlyIncomplete == status = "Paused" => ~Exhausted
EventuallyCompletes == <> (status = "Complete")

THEOREM Spec => []TypeOK
THEOREM Spec => []QueueBound
THEOREM Spec => []StateArenaBound
THEOREM Spec => []TransitionCacheBound
THEOREM Spec => []NoFalsePositives
THEOREM Spec => []PruneSound
THEOREM Spec => []CompleteOnlyAfterExhaustion
THEOREM Spec => []CompleteIsExact
THEOREM Spec => []PausedIsExplicitlyIncomplete
THEOREM Spec => EventuallyCompletes

=============================================================================
