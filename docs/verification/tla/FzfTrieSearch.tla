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

(* Node 2 is the load-bearing case: Active[2] is too small for a descendant
   that starts later. Retaining Unstarted[2] repairs the upper bound. *)
Active == (1 :> 0) @@ (2 :> 10) @@ (3 :> 20)
       @@ (4 :> 40) @@ (5 :> 5) @@ (6 :> 25)
Unstarted == (1 :> 50) @@ (2 :> 40) @@ (3 :> 0)
          @@ (4 :> 0) @@ (5 :> 25) @@ (6 :> 0)
Max2(a, b) == IF a >= b THEN a ELSE b
Bound[n \in Nodes] == Max2(Active[n], Unstarted[n])

RECURSIVE ReachFrom(_)
ReachFrom(n) == {n} \cup UNION {ReachFrom(c) : c \in Children[n]}

BoundSound == \A n \in Nodes : \A d \in ReachFrom(n) :
                d \in Finals => Exact[d] <= Bound[n]
LegacyActiveOnlyIsUnsound == Active[2] < Exact[4]

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
