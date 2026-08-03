---------------------------- MODULE ElasticTrieSearch ----------------------------
(***************************************************************************)
(* Generic ElasticTransducer range traversal.                              *)
(*                                                                         *)
(* The concrete table contains a final root (an indexed empty series), a   *)
(* subtree rejected by a constant-time prefix gate before column building, *)
(* a subtree rejected by its interval column, two emitted non-root leaves,  *)
(* a leaf rejected by K4, and a leaf admitted by both lower bounds but      *)
(* rejected by K3 exact rescoring. The processing order is                  *)
(* nondeterministic, matching the fact that correctness is independent of  *)
(* dictionary edge order.                                                  *)
(***************************************************************************)

EXTENDS Integers, FiniteSets, TLC

CONSTANT Tau

Nodes == 1..9
Root == 1
Refs == {10, 30, 40, 60, 70, 80, 90}

NodeFinal == (1 :> TRUE)  @@ (2 :> FALSE) @@ (3 :> TRUE) @@ (4 :> TRUE)
          @@ (5 :> FALSE) @@ (6 :> TRUE)  @@ (7 :> TRUE) @@ (8 :> TRUE)
          @@ (9 :> TRUE)
PrefixLB == (1 :> 0) @@ (2 :> 1) @@ (3 :> 2) @@ (4 :> 1)
          @@ (5 :> 5) @@ (6 :> 6) @@ (7 :> 2) @@ (8 :> 2) @@ (9 :> 1)
NodeLB == (1 :> 0) @@ (2 :> 1) @@ (3 :> 2) @@ (4 :> 1)
       @@ (5 :> 5) @@ (6 :> 6) @@ (7 :> 2) @@ (8 :> 2) @@ (9 :> 5)
CandidateLB == (1 :> 0) @@ (2 :> 0) @@ (3 :> 2) @@ (4 :> 2)
            @@ (5 :> 0) @@ (6 :> 6) @@ (7 :> 5) @@ (8 :> 2) @@ (9 :> 1)
Exact == (1 :> 0) @@ (2 :> 0) @@ (3 :> 2) @@ (4 :> 3)
      @@ (5 :> 0) @@ (6 :> 7) @@ (7 :> 6) @@ (8 :> 10) @@ (9 :> 9)
NodeRef == (1 :> 10) @@ (2 :> 0) @@ (3 :> 30) @@ (4 :> 40)
        @@ (5 :> 0) @@ (6 :> 60) @@ (7 :> 70) @@ (8 :> 80) @@ (9 :> 90)
Children == (1 :> {2, 5, 9}) @@ (2 :> {3, 4, 7, 8}) @@ (3 :> {}) @@ (4 :> {})
         @@ (5 :> {6}) @@ (6 :> {}) @@ (7 :> {}) @@ (8 :> {}) @@ (9 :> {})

RECURSIVE ReachFrom(_)
ReachFrom(n) == {n} \cup UNION {ReachFrom(c) : c \in Children[n]}
Subtree(n) == ReachFrom(n)

(***************************************************************************)
(* Executable K1 and K4 facts for this model. No abstract ASSUME is used.   *)
(***************************************************************************)
PrefixK1Table == \A n \in Nodes : \A m \in Subtree(n) :
                    NodeFinal[m] => PrefixLB[n] <= Exact[m]
K1Table == \A n \in Nodes : \A m \in Subtree(n) :
              NodeFinal[m] => NodeLB[n] <= Exact[m]
K4Table == \A n \in Nodes : NodeFinal[n] => CandidateLB[n] <= Exact[n]

VARIABLES pending, done, prefixPruned, columnPruned, columnsBuilt, emitted
vars == <<pending, done, prefixPruned, columnPruned, columnsBuilt, emitted>>

Init ==
    /\ pending = {Root}
    /\ done = {}
    /\ prefixPruned = {}
    /\ columnPruned = {}
    /\ columnsBuilt = {}
    /\ emitted = {}

Step ==
    \E n \in pending \ done :
        /\ done' = done \cup {n}
        /\ IF PrefixLB[n] > Tau
           THEN /\ pending' = pending
                /\ prefixPruned' = prefixPruned \cup {n}
                /\ UNCHANGED <<columnPruned, columnsBuilt, emitted>>
           ELSE /\ columnsBuilt' = columnsBuilt \cup {n}
                /\ UNCHANGED prefixPruned
                /\ IF NodeLB[n] > Tau
                   THEN /\ pending' = pending
                        /\ columnPruned' = columnPruned \cup {n}
                        /\ UNCHANGED emitted
                   ELSE /\ pending' = pending \cup Children[n]
                        /\ UNCHANGED columnPruned
                        /\ emitted' =
                             IF NodeFinal[n]
                                /\ CandidateLB[n] <= Tau
                                /\ Exact[n] <= Tau
                             THEN emitted \cup {NodeRef[n]}
                             ELSE emitted

Terminated == \A n \in pending : n \in done
Next == Step \/ (Terminated /\ UNCHANGED vars)
Spec == Init /\ [][Next]_vars /\ WF_vars(Step)

Expected == {NodeRef[n] : n \in {m \in Nodes : NodeFinal[m] /\ Exact[m] <= Tau}}

TypeOK ==
    /\ pending \subseteq Nodes
    /\ done \subseteq Nodes
    /\ prefixPruned \subseteq Nodes
    /\ columnPruned \subseteq Nodes
    /\ columnsBuilt \subseteq Nodes
    /\ emitted \subseteq Refs
    /\ PrefixK1Table
    /\ K1Table
    /\ K4Table

NoFalsePositives ==
    \A ref \in emitted :
        \E n \in Nodes : NodeFinal[n] /\ NodeRef[n] = ref /\ Exact[n] <= Tau

PrefixPruneSound ==
    \A n \in prefixPruned :
        \A m \in Subtree(n) : NodeFinal[m] => Exact[m] > Tau

ColumnPruneSound ==
    \A n \in columnPruned :
        \A m \in Subtree(n) : NodeFinal[m] => Exact[m] > Tau

PrefixGatePrecedesColumn == prefixPruned \cap columnsBuilt = {}

CandidatePruneSound ==
    \A n \in done :
        (NodeFinal[n] /\ CandidateLB[n] > Tau) => Exact[n] > Tau

RootTerminalComplete == Root \in done => NodeRef[Root] \in emitted

CompleteWhenTerminated == Terminated => emitted = Expected
EventuallyTerminates == <>Terminated

THEOREM Spec => []TypeOK
THEOREM Spec => []NoFalsePositives
THEOREM Spec => []PrefixPruneSound
THEOREM Spec => []ColumnPruneSound
THEOREM Spec => []PrefixGatePrecedesColumn
THEOREM Spec => []CandidatePruneSound
THEOREM Spec => []RootTerminalComplete
THEOREM Spec => []CompleteWhenTerminated
THEOREM Spec => EventuallyTerminates

================================================================================
