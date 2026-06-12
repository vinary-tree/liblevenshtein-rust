-------------------------------- MODULE MsmTrieSearch --------------------------------
(***************************************************************************)
(* TLA+ specification for the interval-MSM trie search.                     *)
(*                                                                          *)
(* Models the non-empty-query traversal of `MsmTransducer::search_range`     *)
(* (src/time_series/msm_transducer.rs): a walk over a trie of quantized      *)
(* reference series that, at each node,                                      *)
(* carries an interval-relaxed MSM column lower bound, PRUNES a subtree when *)
(* that bound exceeds the threshold tau, and at each final node re-scores    *)
(* the candidate against the stored full-precision original with the EXACT   *)
(* MSM distance, emitting only genuine matches.                              *)
(*                                                                          *)
(* The Coq development (Liblevenshtein.MSM.Indexing.IntervalColumn) proves   *)
(* the mathematical core: the per-node column lower bound never exceeds the  *)
(* exact MSM distance of any reference in its subtree (admissibility +       *)
(* pruning soundness, lemmas column_lb_le_deeper / lb_prune_sound_msm). This *)
(* model assumes that proved relationship as a constraint on the abstracted  *)
(* distance/bound table (ASSUME AdmissibleTable / MonotoneDownEdges) and     *)
(* verifies the OPERATIONAL consequences of the traversal itself:            *)
(*                                                                          *)
(*   - NoFalsePositives : every emitted id has exact MSM <= tau (the exact   *)
(*                        re-scoring step is what guarantees this);          *)
(*   - NoMissedMatches  : every processed in-threshold final is emitted;     *)
(*   - PruneSound       : a pruned node's subtree contains no in-threshold   *)
(*                        final (operational form of lb_prune_sound_msm);    *)
(*   - EventuallyTerminates : the walk is finite.                            *)
(*                                                                          *)
(* The nondeterministic processing order makes TLC verify these are          *)
(* order-independent (the Rust recursion order does not matter).             *)
(*                                                                          *)
(* Distances and bounds are abstracted as a small finite integer table; the *)
(* Coq proofs supply the only nontrivial fact relating them (admissibility). *)
(*                                                                          *)
(* Corresponds to: src/time_series/msm_transducer.rs                        *)
(*                 src/time_series/msm_interval.rs                          *)
(***************************************************************************)

EXTENDS Integers, FiniteSets, TLC

CONSTANTS Tau            \* MSM threshold (no-arg search returns {id : MSM <= Tau})

ASSUME Tau >= 0

(***************************************************************************)
(* A small concrete trie of quantized references.                           *)
(*                                                                          *)
(*   node 1 : root, non-final            NodeLB 0    children {2,5}          *)
(*   node 2 : depth1, non-final          NodeLB 1    children {3,4,8}        *)
(*   node 3 : final, ref 30              NodeLB 2, ExactMSM 2  (in range)    *)
(*   node 4 : final, ref 40              NodeLB 1, ExactMSM 3  (in range)    *)
(*   node 8 : final, ref 80              NodeLB 2, ExactMSM 10 (LB in range  *)
(*                                                   but exact OUT: rejected *)
(*                                                   by the verify step)     *)
(*   node 5 : depth1, non-final          NodeLB 5    children {6}  (PRUNED)  *)
(*   node 6 : final, ref 60              NodeLB 6, ExactMSM 7  (in pruned    *)
(*                                                   subtree; out of range)  *)
(***************************************************************************)

Nodes == 1..6 \cup {8}
Root  == 1
Refs  == {30, 40, 60, 80}

NodeFinal == (1 :> FALSE) @@ (2 :> FALSE) @@ (3 :> TRUE) @@ (4 :> TRUE)
          @@ (5 :> FALSE) @@ (6 :> TRUE)  @@ (8 :> TRUE)
NodeLB    == (1 :> 0) @@ (2 :> 1) @@ (3 :> 2) @@ (4 :> 1)
          @@ (5 :> 5) @@ (6 :> 6) @@ (8 :> 2)
ExactMSM  == (1 :> 0) @@ (2 :> 0) @@ (3 :> 2) @@ (4 :> 3)
          @@ (5 :> 0) @@ (6 :> 7) @@ (8 :> 10)
NodeRef   == (1 :> 0) @@ (2 :> 0) @@ (3 :> 30) @@ (4 :> 40)
          @@ (5 :> 0) @@ (6 :> 60) @@ (8 :> 80)
NodeChildren == (1 :> {2,5}) @@ (2 :> {3,4,8}) @@ (3 :> {}) @@ (4 :> {})
          @@ (5 :> {6}) @@ (6 :> {}) @@ (8 :> {})

(* Reflexive-transitive closure of the child relation: the subtree at n. *)
RECURSIVE ReachFrom(_)
ReachFrom(n) == {n} \cup UNION { ReachFrom(c) : c \in NodeChildren[n] }
Subtree(n) == ReachFrom(n)

(***************************************************************************)
(* Table consistency, as proved in Coq:                                     *)
(*                                                                          *)
(*  - AdmissibleTable: a node's column lower bound never exceeds the exact   *)
(*    MSM of any final in its subtree (Coq column_lb_le_deeper). Hence       *)
(*    pruning on the bound is sound.                                         *)
(*  - MonotoneDownEdges: the bound is non-decreasing down a trie edge (the   *)
(*    interval column minimum only grows as more target elements are         *)
(*    consumed; Coq mcol_min_le_succ).                                       *)
(***************************************************************************)
ASSUME AdmissibleTable ==
    \A n \in Nodes : \A m \in Subtree(n) :
        NodeFinal[m] => NodeLB[n] <= ExactMSM[m]

ASSUME MonotoneDownEdges ==
    \A n \in Nodes : \A c \in NodeChildren[n] : NodeLB[n] <= NodeLB[c]

(***************************************************************************)
(* State                                                                    *)
(***************************************************************************)

VARIABLES
    pending,   \* frontier nodes still to explore
    done,      \* nodes already processed
    emitted    \* set of emitted reference ids

vars == <<pending, done, emitted>>

Init ==
    /\ pending = {Root}
    /\ done = {}
    /\ emitted = {}

(***************************************************************************)
(* One processed node, mirroring `MsmTransducer::walk_range` for one node:   *)
(*   - if the node's column lower bound exceeds tau, PRUNE: mark done but do *)
(*     NOT enqueue children (skip the whole subtree);                        *)
(*   - otherwise DESCEND: enqueue children, and if this is a final whose     *)
(*     EXACT MSM is within tau, emit its id (the exact re-scoring step).     *)
(***************************************************************************)
Step ==
    \E n \in pending \ done :
        /\ done' = done \cup {n}
        /\ IF NodeLB[n] > Tau
           THEN /\ pending' = pending
                /\ UNCHANGED emitted
           ELSE /\ pending' = pending \cup NodeChildren[n]
                /\ emitted' = IF NodeFinal[n] /\ ExactMSM[n] <= Tau
                              THEN emitted \cup {NodeRef[n]}
                              ELSE emitted

Terminated == \A n \in pending : n \in done

Next == Step \/ (Terminated /\ UNCHANGED vars)

Spec == Init /\ [][Next]_vars /\ WF_vars(Step)

(***************************************************************************)
(* Invariants                                                               *)
(***************************************************************************)

TypeOK ==
    /\ pending \subseteq Nodes
    /\ done \subseteq Nodes
    /\ emitted \subseteq Refs

\* NO FALSE POSITIVES: every emitted id corresponds to a final whose EXACT
\* MSM distance is within tau. Guaranteed by the exact re-scoring step.
NoFalsePositives ==
    \A r \in emitted :
        \E n \in Nodes : NodeFinal[n] /\ NodeRef[n] = r /\ ExactMSM[n] <= Tau

\* NO MISSED MATCHES: every processed final that is genuinely within tau has
\* been emitted. (Pruned-away finals are covered by PruneSound below.)
NoMissedMatches ==
    \A n \in done :
        (NodeFinal[n] /\ ExactMSM[n] <= Tau) => (NodeRef[n] \in emitted)

\* PRUNING SOUNDNESS: a pruned node's subtree contains no in-threshold final.
\* This is the operational form of the Coq theorem lb_prune_sound_msm /
\* column_lb_le_deeper: a subtree whose bound exceeds tau holds no true match.
PruneSound ==
    \A n \in done :
        (NodeLB[n] > Tau) =>
            (\A m \in Subtree(n) : NodeFinal[m] => ExactMSM[m] > Tau)

(***************************************************************************)
(* Temporal property: the walk terminates (the trie is finite).             *)
(***************************************************************************)

EventuallyTerminates == <>Terminated

(***************************************************************************)
(* Theorems (checked by TLC).                                               *)
(***************************************************************************)

THEOREM Spec => []TypeOK
THEOREM Spec => []NoFalsePositives
THEOREM Spec => []NoMissedMatches
THEOREM Spec => []PruneSound
THEOREM Spec => EventuallyTerminates

================================================================================
