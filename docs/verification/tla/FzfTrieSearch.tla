------------------------------ MODULE FzfTrieSearch ------------------------------
(***************************************************************************)
(* Finite max-score trie traversal with fzf's unstarted local alignment.    *)
(***************************************************************************)

EXTENDS Integers, FiniteSets, TLC

CONSTANT Cutoff

Nodes == 1..6
Root == 1
Finals == {3, 4, 6}
Children == (1 :> {2, 5}) @@ (2 :> {3, 4}) @@ (3 :> {})
         @@ (4 :> {}) @@ (5 :> {6}) @@ (6 :> {})
Exact == (1 :> 0) @@ (2 :> 0) @@ (3 :> 20)
      @@ (4 :> 40) @@ (5 :> 0) @@ (6 :> 25)

QueryLen == 2
Beta == 20
Capacity == (1 :> 3) @@ (2 :> 2) @@ (3 :> 0)
         @@ (4 :> 0) @@ (5 :> 1) @@ (6 :> 0)
ActiveRemaining == (1 :> 2) @@ (2 :> 1) @@ (3 :> 0)
                @@ (4 :> 0) @@ (5 :> 1) @@ (6 :> 0)
Completed == (1 :> 0) @@ (2 :> 0) @@ (3 :> 20)
          @@ (4 :> 40) @@ (5 :> 0) @@ (6 :> 25)

(* Node 2 is the load-bearing recurrence case: its active projection is 30,
   but a descendant can start later and score 40. QueryLen <= Capacity[2]
   retains the unstarted term. At node 5 the query no longer fits and the
   active recurrence projection supplies the exact bound 25. *)
Active == (1 :> 0) @@ (2 :> 10) @@ (3 :> 20)
       @@ (4 :> 40) @@ (5 :> 5) @@ (6 :> 25)
Unstarted == (1 :> 50) @@ (2 :> 40) @@ (3 :> 0)
          @@ (4 :> 0) @@ (5 :> 25) @@ (6 :> 0)
Max2(a, b) == IF a >= b THEN a ELSE b
Feasible(ok, completed, term) == IF ok THEN term ELSE completed
Bound[n \in Nodes] ==
  Max2(Completed[n],
    Max2(Feasible(QueryLen <= Capacity[n], Completed[n], Unstarted[n]),
         Feasible(ActiveRemaining[n] <= Capacity[n], Completed[n],
                  Active[n] + ActiveRemaining[n] * Beta)))

RECURSIVE ReachFrom(_)
ReachFrom(n) == {n} \cup UNION {ReachFrom(c) : c \in Children[n]}

BoundSound == \A n \in Nodes : \A d \in ReachFrom(n) :
                d \in Finals => Exact[d] <= Bound[n]
LegacyActiveOnlyIsUnsound == Active[2] + ActiveRemaining[2] * Beta < Exact[4]

VARIABLES pending, done, pruned, emitted
vars == <<pending, done, pruned, emitted>>

Init == /\ pending = {Root}
        /\ done = {}
        /\ pruned = {}
        /\ emitted = {}

Step == \E n \in pending \ done :
          /\ done' = done \cup {n}
          /\ IF Bound[n] < Cutoff
             THEN /\ pending' = pending
                  /\ pruned' = pruned \cup {n}
                  /\ UNCHANGED emitted
             ELSE /\ pending' = pending \cup Children[n]
                  /\ UNCHANGED pruned
                  /\ emitted' = IF n \in Finals /\ Exact[n] >= Cutoff
                                 THEN emitted \cup {n} ELSE emitted

Terminated == \A n \in pending : n \in done
Next == Step \/ (Terminated /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Step)

Expected == {n \in Finals : Exact[n] >= Cutoff}
TypeOK == /\ pending \subseteq Nodes
          /\ done \subseteq Nodes
          /\ pruned \subseteq Nodes
          /\ emitted \subseteq Finals
          /\ BoundSound
          /\ LegacyActiveOnlyIsUnsound
PruneSound == \A n \in pruned : \A d \in ReachFrom(n) :
                d \in Finals => Exact[d] < Cutoff
NoFalsePositive == \A n \in emitted : Exact[n] >= Cutoff
CompleteWhenTerminated == Terminated => emitted = Expected
EventuallyTerminates == <>Terminated

THEOREM Spec => []TypeOK
THEOREM Spec => []PruneSound
THEOREM Spec => []NoFalsePositive
THEOREM Spec => []CompleteWhenTerminated
THEOREM Spec => EventuallyTerminates

================================================================================
